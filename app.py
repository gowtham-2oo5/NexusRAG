from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel
from typing import List, Optional
import os, tempfile, requests, time, logging, hashlib, asyncio, json
from urllib.parse import urlparse
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiohttp

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from openai import RateLimitError

# Initialize FastAPI app FIRST
app = FastAPI(title="HackRX Document Q&A API", version="2.5")

# ========== Constants ==========
GPT_PRIMARY = "gpt-4o-mini"
GPT_FALLBACK = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}
MAX_WORKERS = 8  # Increased for hackathon performance
EMBEDDING_BATCH_SIZE = 20  # Larger batches - you have good rate limits

# Rate limit optimized settings
GPT_4O_CONCURRENT = 8      # 500 RPM = ~8 per second max
GPT_4O_MINI_CONCURRENT = 15 # 500 RPM but use for domain detection only
EMBEDDING_CONCURRENT = 12   # Good embedding rate limits

# ========== Thread Pool for CPU-bound tasks ==========
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

# ========== Logging Setup ==========
log_dir = "logs-v2"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "hackrx_api.log")

app_logger = logging.getLogger("hackrx")
app_logger.setLevel(logging.INFO)
app_logger.propagate = False  # Prevent propagation to root logger

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File handler with rotation
file_handler = RotatingFileHandler(
    filename=log_file_path,
    mode='a',
    maxBytes=10 * 1024 * 1024,  # 10MB
    backupCount=5,
    encoding="utf-8",
    delay=False
)
file_handler.setFormatter(formatter)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Clear existing handlers to avoid duplication
if app_logger.hasHandlers():
    app_logger.handlers.clear()

# Add both handlers
app_logger.addHandler(file_handler)
app_logger.addHandler(console_handler)

app_logger.info("✅ Logger initialized: Console + File")
app_logger.info(f"🧪 Logging test: File path is {log_file_path}")

# ========== Enhanced Request Logging Middleware ==========
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Capture ALL request details
    client_ip = request.client.host if request.client else "unknown"
    method = request.method
    url = str(request.url)
    headers = dict(request.headers)
    
    # Create detailed request info structure
    request_details = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "method": method,
        "url": url,
        "path": request.url.path,
        "query_params": dict(request.query_params) if request.query_params else {},
        "headers": headers,
        "client_ip": client_ip,
        "user_agent": request.headers.get("user-agent", "Unknown")
    }
    
    app_logger.info(f"🌐 INCOMING REQUEST: {method} {url}")
    app_logger.info(f"🔍 Client IP: {client_ip}")
    
    # Log headers in a readable format
    app_logger.info(f"📋 Headers: {json.dumps(headers, indent=2)}")
    
    # For POST/PUT/PATCH requests, capture and log the complete body
    if method in ["POST", "PUT", "PATCH"]:
        try:
            # Read the body
            body = await request.body()
            
            if body:
                try:
                    # Try to decode as UTF-8
                    body_str = body.decode('utf-8')
                    
                    if request.headers.get("content-type", "").startswith("application/json"):
                        try:
                            body_json = json.loads(body_str)
                            request_details["body"] = body_json
                            
                            # Log the complete JSON body with formatting
                            app_logger.info(f"📝 Complete Request Body (JSON):")
                            app_logger.info(json.dumps(body_json, indent=2, ensure_ascii=False))
                            
                            # Also log questions count and document URL separately for quick reference
                            if isinstance(body_json, dict):
                                if "questions" in body_json:
                                    app_logger.info(f"❓ Questions Count: {len(body_json.get('questions', []))}")
                                    for i, q in enumerate(body_json.get('questions', []), 1):
                                        app_logger.info(f"   Q{i}: {q}")
                                if "documents" in body_json:
                                    app_logger.info(f"📄 Document URL: {body_json.get('documents', 'N/A')}")
                                    
                        except json.JSONDecodeError:
                            request_details["body"] = body_str
                            app_logger.info(f"📝 Complete Request Body (Invalid JSON):\n{body_str}")
                    else:
                        request_details["body"] = body_str
                        app_logger.info(f"📝 Complete Request Body (Raw):\n{body_str}")
                        
                except UnicodeDecodeError:
                    request_details["body"] = f"<Binary data: {len(body)} bytes>"
                    app_logger.info(f"📝 Complete Request Body (Binary): {len(body)} bytes")
                    # For binary data, show first 500 bytes as hex
                    hex_preview = body[:500].hex()
                    app_logger.info(f"📝 Binary Preview (hex): {hex_preview}")
                    
            else:
                request_details["body"] = None
                app_logger.info("📝 Request Body: Empty")
                
        except Exception as e:
            request_details["body_error"] = str(e)
            app_logger.warning(f"⚠️ Could not read request body: {e}")
    
    # Log query parameters if any
    if request.query_params:
        app_logger.info(f"🔗 Query Parameters: {dict(request.query_params)}")
    
    # Log the complete structured request details
    app_logger.info(f"🔍 COMPLETE REQUEST STRUCTURE:")
    app_logger.info(json.dumps(request_details, indent=2, ensure_ascii=False))
    
    # Process request
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log response details
        response_details = {
            "status_code": response.status_code,
            "processing_time": f"{process_time:.2f}s",
            "response_headers": dict(response.headers)
        }
        
        app_logger.info(f"✅ RESPONSE: {response.status_code} | Time: {process_time:.2f}s")
        app_logger.info(f"📤 Response Details: {json.dumps(response_details, indent=2)}")
        
        return response
        
    except Exception as e:
        process_time = time.time() - start_time
        error_details = {
            "error": str(e),
            "processing_time": f"{process_time:.2f}s",
            "error_type": type(e).__name__
        }
        app_logger.error(f"❌ REQUEST FAILED: {str(e)} | Time: {process_time:.2f}s")
        app_logger.error(f"❌ ERROR DETAILS: {json.dumps(error_details, indent=2)}")
        app_logger.exception("Full error traceback:")
        raise

# ========== Env Load ==========
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
os.environ["OPENAI_API_KEY"] = openai_api_key
app_logger.info("✅ OpenAI API key loaded")

# ========== Prompts ==========
domain_detection_template = PromptTemplate(
    input_variables=["document_content"],
    template="""
Analyze the following document content and identify its primary domain/expertise area.

Document Content (first 2000 characters):
{document_content}

Based on this content, identify the PRIMARY domain this document belongs to. Choose from these categories or suggest a more specific one:
- Automobile/Vehicle/Motorcycle
- Insurance/Finance
- Healthcare/Medical
- Technology/Software
- Legal/Law
- Education/Academic
- Real Estate/Property
- Food/Nutrition
- Travel/Tourism
- Manufacturing/Industrial
- Other (specify)

Respond with ONLY the domain name (e.g., "Automobile", "Insurance", "Healthcare", etc.). Be specific - if it's about motorcycles, say "Motorcycle" not just "Automobile".
"""
)

# NEW: Fact-checking prompt for general knowledge questions
fact_checking_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template="""
You are a fact-checking expert assistant. The user asked a general knowledge question that may not be directly related to the document.

Document Context (for reference):
{context}

Document Domain: {domain}

Question: {question}

Instructions:
1. First, check if the document context contains ANY relevant information about this question
2. If the document has relevant info, use it as the PRIMARY source and provide accurate information
3. If the document has NO relevant information, provide accurate general knowledge answer
4. If the document contains FALSE or misleading information about this topic, correct it with accurate facts
5. Be helpful and educational while being factually accurate

Respond in this format:
- If document has relevant accurate info: "Based on the document, [answer with document info]"
- If document has no relevant info: "While this isn't covered in the document, [provide accurate general knowledge answer]"  
- If document has false info: "The document contains incorrect information. The accurate answer is: [correct answer]"

Answer:
"""
)

enhanced_prompt_template = PromptTemplate(
    input_variables=["context", "question", "domain"],
    template="""
You are an expert assistant in {domain} helping users understand documents and related concepts.

Instructions:
- Use the provided document context as your PRIMARY source of information
- If the document context doesn't contain a direct answer but the question is related to {domain}, use your expert knowledge in {domain} to provide a helpful, accurate answer
- Combine document information with your domain expertise when appropriate
- If the question is completely unrelated to {domain} and the document topic, respond with: "Sorry, this question is out of the context of the document."
- Be helpful, accurate, and professional. Draw from both the document and your {domain} expertise.
- Provide concise but complete answers (2-3 sentences when needed for clarity)
For example:
    -If the document is about automobile user manuals and the question is "Can I put thums up instead of oil" th answer should be "No, you should not use thumbs up instead of oil. The document specifies that only the recommended oil type should be used for optimal performance."
    -If the document is about Indian Consitution and the question is "Article 21 significance" the answer should be "Article 21 of the Indian Constitution guarantees the right to life and personal liberty. It ensures that no person shall be deprived of their life or personal liberty except according to the procedure established by law."
    -If the document is about automobile user manuals and the question is "Write a code to check if a number is prime" the answer should be "Sorry, this question is out of the context of the document."
Document Context:
{context}

Question (as a {domain} expert): {question}

Answer:
"""
)

# ========== Schema ==========
class QARequest(BaseModel):
    documents: str
    questions: List[str]

# ========== File Helpers ==========
def get_faiss_folder(doc_hash: str) -> str:
    folder = os.path.join("faiss_indexes", doc_hash)
    os.makedirs(folder, exist_ok=True)
    return folder

def faiss_index_exists(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, "index.faiss")) and os.path.exists(os.path.join(folder, "index.pkl"))

def get_domain_cache_path(doc_hash: str) -> str:
    return os.path.join("faiss_indexes", doc_hash, "domain.txt")

async def save_domain_to_cache(doc_hash: str, domain: str):
    """Async version of save_domain_to_cache"""
    cache_path = get_domain_cache_path(doc_hash)
    async with aiofiles.open(cache_path, 'w') as f:
        await f.write(domain)
    app_logger.info(f"💾 Cached domain '{domain}' for document hash: {doc_hash}")

async def load_domain_from_cache(doc_hash: str) -> Optional[str]:
    """Async version of load_domain_from_cache"""
    path = get_domain_cache_path(doc_hash)
    if os.path.exists(path):
        async with aiofiles.open(path, 'r') as f:
            content = await f.read()
            domain = content.strip()
        app_logger.info(f"✅ Loaded cached domain '{domain}' for document hash: {doc_hash}")
        return domain
    return None
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(ROOT_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

async def download_and_hash_document(url: str, ext: str) -> (str, str):
    """Async version using aiohttp for concurrent downloads"""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: .{ext}")
    
    app_logger.info(f"📥 Downloading document from: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content = await response.read()

    # Hash calculation in thread pool to avoid blocking
    doc_hash = await asyncio.get_event_loop().run_in_executor(
        executor, lambda: hashlib.sha256(content).hexdigest()
    )
    
    file_path = os.path.join(DOCS_DIR, f"{doc_hash}.{ext}")
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    app_logger.info(f"📄 Saved document to: {file_path} (SHA-256: {doc_hash})")
    return file_path, doc_hash


def load_document_sync(file_path: str, ext: str):
    """Synchronous document loading for thread pool execution"""
    loader = {
        "pdf": PyPDFLoader,
        "txt": lambda p: TextLoader(p, encoding="utf-8"),
        "docx": Docx2txtLoader
    }.get(ext.lower())
    if not loader:
        raise ValueError(f"Unsupported file extension: .{ext}")
    return loader(file_path).load()

async def load_document(file_path: str, ext: str):
    """Async wrapper for document loading"""
    app_logger.info(f"📚 Loading document with extension: {ext}")
    return await asyncio.get_event_loop().run_in_executor(
        executor, load_document_sync, file_path, ext
    )

def split_document_sync(docs):
    """Synchronous document splitting for thread pool execution"""
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_documents(docs)

async def split_document(docs):
    """Async wrapper for document splitting"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, split_document_sync, docs
    )

# ========== FAISS ==========
async def embed_chunks_parallel(chunks, batch_size=EMBEDDING_BATCH_SIZE):
    """Embed chunks in parallel batches optimized for your rate limits"""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    
    # Split chunks into batches for parallel processing
    chunk_batches = [chunks[i:i + batch_size] for i in range(0, len(chunks), batch_size)]
    
    # Use semaphore to respect embedding rate limits
    embedding_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENT)
    
    async def embed_batch_with_limit(batch_idx, batch):
        async with embedding_semaphore:
            texts = [chunk.page_content for chunk in batch]
            app_logger.info(f"🔢 Embedding batch {batch_idx + 1} of {len(chunk_batches)}")
            return await embeddings.aembed_documents(texts)
    
    app_logger.info(f"🔥 Embedding {len(chunks)} chunks in {len(chunk_batches)} parallel batches")
    start_time = time.time()
    
    # Process all batches concurrently with rate limiting
    batch_embeddings = await asyncio.gather(
        *(embed_batch_with_limit(i, batch) for i, batch in enumerate(chunk_batches)),
        return_exceptions=True
    )
    
    app_logger.info(f"⚡ Embedding completed in {time.time() - start_time:.2f}s")
    
    # Flatten results and handle exceptions
    all_embeddings = []
    for i, batch_result in enumerate(batch_embeddings):
        if isinstance(batch_result, Exception):
            app_logger.error(f"Embedding batch {i} failed: {batch_result}")
            # Fallback to sync embedding for failed batch
            embeddings_sync = OpenAIEmbeddings(model=EMBED_MODEL)
            texts = [chunk.page_content for chunk in chunk_batches[i]]
            batch_result = embeddings_sync.embed_documents(texts)
        all_embeddings.extend(batch_result)
    
    return all_embeddings

def build_faiss_index_sync(chunks, doc_hash, embedded_vectors):
    """Build FAISS index with pre-computed embeddings"""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    text_embeddings = list(zip([chunk.page_content for chunk in chunks], embedded_vectors))
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings, 
        embedding=embeddings, 
        metadatas=[chunk.metadata for chunk in chunks]
    )
    vectorstore.save_local(get_faiss_folder(doc_hash))
    return vectorstore

async def build_faiss_index(chunks, doc_hash):
    """Build FAISS index with parallel embedding generation"""
    app_logger.info(f"🔧 Building new FAISS index for hash: {doc_hash}")
    # Generate embeddings in parallel
    embedded_vectors = await embed_chunks_parallel(chunks)
    
    # Build index in thread pool with pre-computed embeddings
    vectorstore = await asyncio.get_event_loop().run_in_executor(
        executor, build_faiss_index_sync, chunks, doc_hash, embedded_vectors
    )
    app_logger.info(f"💾 Saved FAISS index to {get_faiss_folder(doc_hash)}")
    return vectorstore

def load_faiss_index_sync(doc_hash):
    """Synchronous FAISS index loading for thread pool execution"""
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(get_faiss_folder(doc_hash), embeddings, allow_dangerous_deserialization=True)

async def load_faiss_index(doc_hash):
    """Async wrapper for FAISS index loading"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, load_faiss_index_sync, doc_hash
    )

# ========== LLM ==========
async def detect_document_domain(docs: List[Document]) -> str:
    sample_content = "\n".join([doc.page_content for doc in docs[:3]])[:2000]
    llm = ChatOpenAI(model_name=GPT_FALLBACK, temperature=0.1)
    prompt = domain_detection_template.format(document_content=sample_content)
    result = await llm.ainvoke(prompt)
    return result.content.strip()

def is_general_knowledge_question(question: str) -> bool:
    """Detect if a question is general knowledge/fact-checking type"""
    general_knowledge_indicators = [
        "what is the capital of", "where can we find", "what are clouds made of",
        "how many", "who is", "what is the name of", "when was", "how old is",
        "what color is", "how tall is", "what does", "where is", "which country",
        "what planet", "what galaxy", "what ocean", "what continent"
    ]
    question_lower = question.lower()
    return any(indicator in question_lower for indicator in general_knowledge_indicators)

async def ask_with_context(question: str, context_docs: List[Document], domain: str) -> str:
    context = "\n\n".join([doc.page_content for doc in context_docs])
    
    # Choose prompt based on question type
    if is_general_knowledge_question(question):
        app_logger.info(f"🔍 Detected general knowledge question: {question[:50]}...")
        prompt = fact_checking_template.format(context=context, question=question, domain=domain)
    else:
        prompt = enhanced_prompt_template.format(context=context, question=question, domain=domain)

    try:
        # Use GPT-4o for main answers (you have 500 RPM = good for hackathon)
        llm = ChatOpenAI(model_name=GPT_PRIMARY, temperature=0.3)
        result = await llm.ainvoke(prompt)
        return result.content.strip()
    except RateLimitError:
        app_logger.warning(f"⚠️ GPT-4o rate limited. Falling back to 4o-mini (you have 500 RPM for mini)")
        # Fallback has even better rate limits (500 RPM, 200k TPM)
        fallback_llm = ChatOpenAI(model_name=GPT_FALLBACK, temperature=0.3)
        result = await fallback_llm.ainvoke(prompt)
        return result.content.strip()
    except Exception as e:
        app_logger.exception("❌ GPT error")
        raise e

# ========== Concurrent Retrieval Helpers ==========
async def parallel_retrieval(ensemble_retriever, questions, max_concurrent=5):
    """Retrieve documents for all questions in parallel"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    async def retrieve_for_question(question):
        async with semaphore:
            return await ensemble_retriever.ainvoke(question)
    
    return await asyncio.gather(
        *(retrieve_for_question(q) for q in questions),
        return_exceptions=True
    )

def create_bm25_retriever_sync(chunks):
    """Synchronous BM25 retriever creation"""
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2
    return bm25_retriever

async def create_bm25_retriever(chunks):
    """Async wrapper for BM25 retriever creation"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, create_bm25_retriever_sync, chunks
    )

async def async_remove_file(file_path: str):
    """Async file removal"""
    try:
        await asyncio.get_event_loop().run_in_executor(executor, os.remove, file_path)
    except Exception as e:
        app_logger.warning(f"Failed to remove temp file {file_path}: {e}")

# ========== Main Endpoint ==========
@app.post("/hackrx/run")
async def run_rag(req: Request, authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    rawBody = await req.body()
    try:
       body_json = rawBody.decode("utf-8")
       app_logger.info(f" incoming request body: {body_json}")
       data = json.loads(body_json)
       
       req = QARequest(**data)
    except Exception as e:
       app_logger.error(f" Error parsing request: {e}")
       raise HTTPException(status_code = 400, detail="invalid requuest format")

    total_start_time = time.time()
    app_logger.info(f"🎯 Starting RAG pipeline for {len(req.questions)} questions")

    ext = req.documents.split('.')[-1].split('?')[0].lower()
    file_path, doc_hash = await download_and_hash_document(req.documents, ext)

    try:
        faiss_folder = get_faiss_folder(doc_hash)
        
        # Run multiple operations concurrently
        cached_domain_task = asyncio.create_task(load_domain_from_cache(doc_hash))
        docs_task = asyncio.create_task(load_document(file_path, ext))
        
        # Wait for document loading and domain cache check
        cached_domain, docs = await asyncio.gather(cached_domain_task, docs_task)
        app_logger.info(f"📄 Document loaded: {len(docs)} pages")
        
        # Split documents asynchronously
        chunks = await split_document(docs)
        app_logger.info(f"✂️ Document split into {len(chunks)} chunks")

        if faiss_index_exists(faiss_folder):
            app_logger.info(f"✅ Loading cached FAISS index for hash: {doc_hash}")
            # Load existing index and create retrievers concurrently
            vectorstore_task = asyncio.create_task(load_faiss_index(doc_hash))
            bm25_task = asyncio.create_task(create_bm25_retriever(chunks))
            
            if cached_domain:
                domain = cached_domain
                app_logger.info(f"✅ Loaded cached domain '{domain}' for document hash: {doc_hash}")
                vectorstore, bm25_retriever = await asyncio.gather(vectorstore_task, bm25_task)
            else:
                # Detect domain concurrently with loading operations
                domain_task = asyncio.create_task(detect_document_domain(chunks))
                vectorstore, bm25_retriever, domain = await asyncio.gather(
                    vectorstore_task, bm25_task, domain_task
                )
                app_logger.info(f"🔍 Detected document domain: {domain}")
                # Save domain to cache without blocking
                asyncio.create_task(save_domain_to_cache(doc_hash, domain))
        else:
            app_logger.info(f"🔧 Building new FAISS index for hash: {doc_hash}")
            # Build new index and detect domain concurrently
            vectorstore_task = asyncio.create_task(build_faiss_index(chunks, doc_hash))
            domain_task = asyncio.create_task(detect_document_domain(chunks))
            bm25_task = asyncio.create_task(create_bm25_retriever(chunks))
            
            vectorstore, domain, bm25_retriever = await asyncio.gather(
                vectorstore_task, domain_task, bm25_task
            )
            app_logger.info(f"🎯 Detected document domain: {domain}")
            # Save domain to cache without blocking
            asyncio.create_task(save_domain_to_cache(doc_hash, domain))

        # Create ensemble retriever
        vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever], 
            weights=[0.4, 0.6]
        )
        app_logger.info("🔍 Created Ensemble Retriever (BM25 + FAISS)")

        # ULTRA-FAST PARALLEL PROCESSING: Retrieve docs for ALL questions at once
        app_logger.info(f"🚀 Processing {len(req.questions)} questions in parallel")
        start_time = time.time()
        
        retrievals = await parallel_retrieval(ensemble_retriever, req.questions, max_concurrent=10)
        app_logger.info(f"📄 Document retrieval completed in {time.time() - start_time:.2f}s")
        
        # Process Q&A for all questions concurrently - optimized for your 500 RPM limit
        # 500 RPM = ~8 requests per second max, so 8 concurrent is perfect
        qa_semaphore = asyncio.Semaphore(GPT_4O_CONCURRENT)
        
        async def process_question_with_docs(question, docs):
            if isinstance(docs, Exception):
                app_logger.error(f"Retrieval failed for question: {docs}")
                return f"Error retrieving context: {str(docs)}"
            
            async with qa_semaphore:
                return await ask_with_context(question, docs, domain)

        qa_start_time = time.time()
        # Process all Q&A concurrently
        answers = await asyncio.gather(
            *(process_question_with_docs(q, docs) 
              for q, docs in zip(req.questions, retrievals)),
            return_exceptions=True
        )
        
        app_logger.info(f"🤖 LLM processing completed in {time.time() - qa_start_time:.2f}s")
        app_logger.info(f"🏆 Total processing time: {time.time() - start_time:.2f}s")
        
        # Handle any exceptions in answers
        processed_answers = []
        for i, answer in enumerate(answers):
            if isinstance(answer, Exception):
                app_logger.error(f"Error processing question {i}: {answer}")
                processed_answers.append(f"Error processing question: {str(answer)}")
            else:
                processed_answers.append(answer)

        total_time = time.time() - total_start_time
        app_logger.info(f"✅ Total time: {total_time:.2f}s for {len(req.questions)} questions in domain '{domain}'")
        
        # Log all questions and answers like in original code
        for i, (q, a) in enumerate(zip(req.questions, processed_answers), start=1):
            app_logger.info(f"📌 Q{i}: {q.strip()}")
            app_logger.info(f"📝 A{i}: {a.strip()}")

        return {
            "answers": processed_answers, 
            "detected_domain": domain,
            "performance": {
                "total_time": f"{total_time:.2f}s",
                "questions_processed": len(req.questions),
                "chunks_created": len(chunks)
            }
        }

    finally:
        # Clean up temp file asynchronously
        if os.path.exists(file_path):
            asyncio.create_task(async_remove_file(file_path))

# ========== Startup/Shutdown Events ==========
@app.on_event("startup")
async def startup_event():
    app_logger.info("🚀 API starting up with concurrency optimizations")

@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("🔥 Shutting down thread pool executor")
    executor.shutdown(wait=True)

# ========== Health Check ==========
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.5", "concurrency": "enabled"}

# ========== Root Endpoint ==========
@app.get("/")
async def root():
    return {"message": "HackRX Document Q&A API", "version": "2.5", "docs": "/docs"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

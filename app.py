from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
from typing import List, Optional
import os, tempfile, requests, time, logging, hashlib
from urllib.parse import urlparse
import asyncio
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.docstore.document import Document as LCDocument

# ========== App Setup ==========
app = FastAPI(title="HackRx LLM RAG API", version="2.4")

# ========== Logging Setup ==========
# ========== Logging Setup ==========
import logging
from logging.handlers import RotatingFileHandler

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

log_file_path = os.path.join(log_dir, "hackrx_api.log")

# Create a logger
app_logger = logging.getLogger("hackrx")
app_logger.setLevel(logging.INFO)
app_logger.propagate = False  # Prevent propagation to root logger

# Formatter
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# File handler
file_handler = RotatingFileHandler(
    filename=log_file_path,
    mode='a',
    maxBytes=10 * 1024 * 1024,
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

# Add handlers
app_logger.addHandler(file_handler)
app_logger.addHandler(console_handler)

app_logger.info("✅ Logger initialized: Console + File")
app_logger.info(f"🧪 Logging test: File path is {log_file_path}")


# ========== Environment Setup ==========
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
os.environ["OPENAI_API_KEY"] = openai_api_key
app_logger.info("✅ OpenAI API key loaded from environment variables")

# ========== Model Config ==========
DOMAIN_LLM_MODEL = "gpt-4o-mini"
QA_LLM_MODEL = "gpt-4o-mini"

# ========== Prompt Templates ==========
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

# ========== Request Schema ==========
class QARequest(BaseModel):
    documents: str
    questions: List[str]

# ========== Helper Functions ==========
def get_faiss_folder(doc_hash: str) -> str:
    folder = os.path.join("faiss_indexes", doc_hash)
    os.makedirs(folder, exist_ok=True)
    return folder

def faiss_index_exists(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, "index.faiss")) and os.path.exists(os.path.join(folder, "index.pkl"))

def get_domain_cache_path(doc_hash: str) -> str:
    return os.path.join("faiss_indexes", doc_hash, "domain.txt")

def save_domain_to_cache(doc_hash: str, domain: str):
    with open(get_domain_cache_path(doc_hash), 'w') as f:
        f.write(domain)
    app_logger.info(f"💾 Cached domain '{domain}' for document hash: {doc_hash}")

def load_domain_from_cache(doc_hash: str) -> Optional[str]:
    path = get_domain_cache_path(doc_hash)
    if os.path.exists(path):
        with open(path, 'r') as f:
            domain = f.read().strip()
        app_logger.info(f"✅ Loaded cached domain '{domain}' for document hash: {doc_hash}")
        return domain
    return None

def download_and_hash_document(url: str, ext: str) -> (str, str):
    app_logger.info(f"📥 Downloading document from: {url}")
    response = requests.get(url)
    content = response.content
    doc_hash = hashlib.sha256(content).hexdigest()
    temp_path = os.path.join(tempfile.gettempdir(), f"document.{ext}")
    with open(temp_path, 'wb') as f:
        f.write(content)
    app_logger.info(f"📄 Saved document to: {temp_path} (SHA-256: {doc_hash})")
    return temp_path, doc_hash

def load_document(file_path: str, ext: str):
    loader = {
        "pdf": PyPDFLoader,
        "txt": lambda p: TextLoader(p, encoding="utf-8"),
        "docx": Docx2txtLoader
    }.get(ext.lower())
    if not loader:
        raise ValueError(f"Unsupported file extension: .{ext}")
    app_logger.info(f"📚 Loading document with extension: {ext}")
    return loader(file_path).load()

def split_document(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=50)
    return splitter.split_documents(docs)

def embed_chunks_in_batches(chunks, embeddings, batch_size=20):
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        app_logger.info(f"🔢 Embedding batch {i//batch_size + 1} of {((len(texts) - 1) // batch_size + 1)}")
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)
    return all_embeddings, texts, metadatas

def load_or_build_faiss_index(chunks, doc_hash):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    folder = get_faiss_folder(doc_hash)
    if faiss_index_exists(folder):
        app_logger.info(f"✅ Loading cached FAISS index for hash: {doc_hash}")
        return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    app_logger.info(f"🔧 Building new FAISS index for hash: {doc_hash}")
    embedded_vectors, texts, metadatas = embed_chunks_in_batches(chunks, embeddings)
    vectorstore = FAISS.from_embeddings(list(zip(embedded_vectors, texts)), embeddings, metadatas=metadatas)
    vectorstore.save_local(folder)
    app_logger.info(f"💾 Saved FAISS index to {folder}")
    return vectorstore

# ========== LLM Helpers ==========
async def detect_document_domain(docs: List[Document]) -> str:
    sample_content = "\n".join([doc.page_content for doc in docs[:3]])[:2000]
    domain_llm = ChatOpenAI(model_name=DOMAIN_LLM_MODEL, temperature=0.1)
    prompt = domain_detection_template.format(document_content=sample_content)
    result = await asyncio.to_thread(domain_llm.invoke, prompt)
    domain = result.content.strip()
    app_logger.info(f"🎯 Detected document domain: {domain}")
    return domain

async def ask_with_context(question: str, context_docs: List[Document], domain: str) -> str:
    context = "\n\n".join([doc.page_content for doc in context_docs])
    qa_llm = ChatOpenAI(model_name=QA_LLM_MODEL, temperature=0.3)
    prompt = enhanced_prompt_template.format(context=context, question=question, domain=domain)
    result = await asyncio.to_thread(qa_llm.invoke, prompt)
    return result.content.strip()

# ========== Main Endpoint ==========
@app.post("/hackrx/run")
async def run_rag(req: QARequest, authorization: Optional[str] = Header(None)):
    try:
        total_start = time.time()

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")

        ext = req.documents.split('.')[-1].split('?')[0]

        t1 = time.time()
        file_path, doc_hash = download_and_hash_document(req.documents, ext)
        faiss_folder = get_faiss_folder(doc_hash)
        app_logger.info(f"⏱️ Step 1 - Download & Hash: {(time.time() - t1):.2f}s")

        t2 = time.time()
        cached_domain = load_domain_from_cache(doc_hash)
        app_logger.info(f"⏱️ Step 2 - Domain Cache Check: {(time.time() - t2):.2f}s")

        t3 = time.time()
        if faiss_index_exists(faiss_folder):
            vectorstore = load_or_build_faiss_index(None, doc_hash)
            if not cached_domain:
                docs = load_document(file_path, ext)
                chunks = split_document(docs)
                domain = await detect_document_domain(chunks)
                save_domain_to_cache(doc_hash, domain)
            else:
                domain = cached_domain
        else:
            docs = load_document(file_path, ext)
            chunks = split_document(docs)
            vectorstore = load_or_build_faiss_index(chunks, doc_hash)
            domain = await detect_document_domain(chunks)
            save_domain_to_cache(doc_hash, domain)
        app_logger.info(f"⏱️ Step 3 - FAISS + Domain Detection: {(time.time() - t3):.2f}s")

        t4 = time.time()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        app_logger.info(f"⏱️ Step 4 - Retriever Setup: {(time.time() - t4):.2f}s")

        t5 = time.time()
        async def process_question(q):
            relevant_docs = await retriever.ainvoke(q)
            return await ask_with_context(q, relevant_docs, domain)

        answers = await asyncio.gather(*(process_question(q) for q in req.questions))
        app_logger.info(f"⏱️ Step 5 - Answer Generation: {(time.time() - t5):.2f}s")

        app_logger.info(f"✅ Total time: {(time.time() - total_start):.2f}s for {len(req.questions)} questions in domain '{domain}'")
        for i, (q, a) in enumerate(zip(req.questions, answers), start=1):
            app_logger.info(f"📌 Q{i}: {q.strip()}")
            app_logger.info(f"📝 A{i}: {a.strip()}")

        return {"answers": answers, "detected_domain": domain}

    except Exception as e:
        app_logger.exception("❌ Error during /hackrx/run")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

# ========== Health Check ==========
@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "2.4"}

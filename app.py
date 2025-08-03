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
app = FastAPI(title="HackRx LLM RAG API", version="2.3")

# Logging setup: Console + Rotating File
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, "hackrx_api.log")

file_handler = RotatingFileHandler(
    log_file_path, maxBytes=100 * 1024 * 1024, backupCount=5
)
file_handler.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])
logging.info("✅ Logging initialized (console + file)")

# Load .env vars
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
os.environ["OPENAI_API_KEY"] = openai_api_key
logging.info("✅ OpenAI API key loaded from environment variables")

# ========== Prompt Template ==========
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a knowledgeable assistant helping users understand the contents and implications of a document.

Instructions:
- Use the document context to answer accurately and clearly.
- If the document does not contain a direct answer, but the question is clearly related to the document's subject (e.g., motorcycles, insurance, policies), use your general knowledge to give a helpful, reasonable answer.
- If the question is unrelated to the document’s topic (e.g., asking for code when the document is a user manual), respond with: "Sorry, this question is not related to the document."
- Do not just say "information not available" if you can give a reasonable, domain-relevant answer.
- Be helpful, concise, and professional (1–2 sentences). Paraphrase where possible.

Context:
{context}

Question:
{question}

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

def download_and_hash_document(url: str, ext: str) -> (str, str):
    logging.info(f"📥 Downloading document from: {url}")
    response = requests.get(url)
    content = response.content
    doc_hash = hashlib.sha256(content).hexdigest()
    temp_path = os.path.join(tempfile.gettempdir(), f"document.{ext}")
    with open(temp_path, 'wb') as f:
        f.write(content)
    logging.info(f"📄 Saved document to: {temp_path} (SHA-256: {doc_hash})")
    return temp_path, doc_hash

def load_document(file_path: str, ext: str):
    loader = {
        "pdf": PyPDFLoader,
        "txt": lambda p: TextLoader(p, encoding="utf-8"),
        "docx": Docx2txtLoader
    }.get(ext.lower())
    if not loader:
        raise ValueError(f"Unsupported file extension: .{ext}")
    logging.info(f"📚 Loading document with extension: {ext}")
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
        logging.info(f"🔢 Embedding batch {i//batch_size + 1} of {((len(texts) - 1) // batch_size + 1)}")
        batch_embeddings = embeddings.embed_documents(batch)
        all_embeddings.extend(batch_embeddings)

    return all_embeddings, texts, metadatas

def load_or_build_faiss_index(chunks, doc_hash):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    folder = get_faiss_folder(doc_hash)

    if faiss_index_exists(folder):
        logging.info(f"✅ Loading cached FAISS index for hash: {doc_hash}")
        return FAISS.load_local(folder, embeddings, allow_dangerous_deserialization=True)
    else:
        logging.info(f"🔧 Building new FAISS index for hash: {doc_hash}")
        embedded_vectors, texts, metadatas = embed_chunks_in_batches(chunks, embeddings)
        vectorstore = FAISS.from_embeddings(embedded_vectors, texts, metadatas=metadatas)
        vectorstore.save_local(folder)
        logging.info(f"💾 Saved FAISS index to {folder}")
        return vectorstore

async def ask_with_context(llm, question: str, context_docs: List[Document]) -> str:
    context = "\n\n".join([doc.page_content for doc in context_docs])
    prompt = prompt_template.format(context=context, question=question)
    result = await asyncio.to_thread(llm.invoke, prompt)
    return result.content.strip()

# ========== Main Endpoint ==========
@app.post("/hackrx/run")
async def run_rag(req: QARequest, authorization: Optional[str] = Header(None)):
    try:
        total_start = time.time()

        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Unauthorized")

        ext = req.documents.split('.')[-1].split('?')[0]
        file_path, doc_hash = download_and_hash_document(req.documents, ext)
        faiss_folder = get_faiss_folder(doc_hash)

        if faiss_index_exists(faiss_folder):
            vectorstore = load_or_build_faiss_index(None, doc_hash)
        else:
            docs = load_document(file_path, ext)
            chunks = split_document(docs)
            vectorstore = load_or_build_faiss_index(chunks, doc_hash)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.3)

        async def process_question(q):
            relevant_docs = retriever.get_relevant_documents(q)
            return await ask_with_context(llm, q, relevant_docs)

        answers = await asyncio.gather(*(process_question(q) for q in req.questions))

        duration = time.time() - total_start
        logging.info(f"✅ Processed {len(req.questions)} questions in {duration:.2f}s")
        return {"answers": answers}

    except Exception as e:
        logging.exception("❌ Error during /hackrx/run")
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")

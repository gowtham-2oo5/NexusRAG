from fastapi import FastAPI, HTTPException, Header, Request, Response
from pydantic import BaseModel
from typing import List, Optional
import os, tempfile, requests, time, logging, hashlib, asyncio, json, base64
from urllib.parse import urlparse
from dotenv import load_dotenv
from logging.handlers import RotatingFileHandler
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import aiohttp
import pandas as pd  # For Excel processing
import zipfile  # For ZIP file handling

from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from openai import RateLimitError

# NEW IMPORTS for PPT processing
import subprocess
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Initialize FastAPI app FIRST
app = FastAPI(title="HackRX Document Q&A API", version="3.0")

# ========== Constants ==========
GPT_PRIMARY = "gpt-4o-mini"
GPT_FALLBACK = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx", "ppt", "pptx", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "xlsx", "xls", "zip", "bin"}  # Added ppt, pptx, images, excel, zip, bin
MAX_WORKERS = 8  # Increased for hackathon performance
EMBEDDING_BATCH_SIZE = 20  # Larger batches - you have good rate limits
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB file size limit

# Rate limit optimized settings
GPT_4O_CONCURRENT = 8  # 500 RPM = ~8 per second max
GPT_4O_MINI_CONCURRENT = 15  # 500 RPM but use for domain detection only
EMBEDDING_CONCURRENT = 12  # Good embedding rate limits

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
gemini_api_key = os.getenv("GEMINI_API_KEY")  # NEW: Gemini API key
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found in .env")
if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
os.environ["OPENAI_API_KEY"] = openai_api_key

# Configure Gemini
genai.configure(api_key=gemini_api_key)

app_logger.info("✅ OpenAI API key loaded")
app_logger.info("✅ Gemini API key loaded")

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
- For structured data (Excel, CSV), look for exact matches in names, IDs, phone numbers, addresses, etc.
- When searching for people by name, check all name variations and partial matches
- For numerical queries (highest/lowest values), scan through all records to find the correct answer
- Always provide specific details like phone numbers, addresses, or other requested information when available

For example:
    -If the document is about automobile user manuals and the question is "Can I put thums up instead of oil" the answer should be "No, you should not use thumbs up instead of oil. The document specifies that only the recommended oil type should be used for optimal performance."
    -If the document is about Indian Constitution and the question is "Article 21 significance" the answer should be "Article 21 of the Indian Constitution guarantees the right to life and personal liberty. It ensures that no person shall be deprived of their life or personal liberty except according to the procedure established by law."
    -If the document is about automobile user manuals and the question is "Write a code to check if a number is prime" the answer should be "Sorry, this question is out of the context of the document."
    -If the document contains salary data and the question is "Who is the highest paid individual in pincode 400001?" then search through all records with pincode 400001 and find the person with the highest salary, including their contact details.

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


def get_extracted_text_cache_path(doc_hash: str) -> str:
    """Get path for cached extracted text content"""
    return os.path.join("faiss_indexes", doc_hash, "extracted_text.txt")


async def save_domain_to_cache(doc_hash: str, domain: str):
    """Async version of save_domain_to_cache"""
    cache_path = get_domain_cache_path(doc_hash)
    async with aiofiles.open(cache_path, 'w') as f:
        await f.write(domain)
    app_logger.info(f"💾 Cached domain '{domain}' for document hash: {doc_hash}")


async def save_extracted_text_to_cache(doc_hash: str, extracted_text: str):
    """Cache extracted text content to avoid re-processing"""
    cache_path = get_extracted_text_cache_path(doc_hash)
    async with aiofiles.open(cache_path, 'w', encoding='utf-8') as f:
        await f.write(extracted_text)
    app_logger.info(f"💾 Cached extracted text ({len(extracted_text)} chars) for document hash: {doc_hash}")


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


async def load_extracted_text_from_cache(doc_hash: str) -> Optional[str]:
    """Load cached extracted text content"""
    path = get_extracted_text_cache_path(doc_hash)
    if os.path.exists(path):
        async with aiofiles.open(path, 'r', encoding='utf-8') as f:
            content = await f.read()
        app_logger.info(f"✅ Loaded cached extracted text ({len(content)} chars) for document hash: {doc_hash}")
        return content
    return None


# ========== NEW: PPT/PPTX Processing Functions ==========
def convert_ppt_to_pdf_sync(ppt_path: str) -> str:
    """Convert PPT/PPTX to PDF using LibreOffice"""
    pdf_dir = os.path.dirname(ppt_path)
    try:
        # Use LibreOffice to convert PPT to PDF
        result = subprocess.run([
            r"C:\Program Files\LibreOffice\program\soffice.exe", '--headless', '--convert-to', 'pdf',
            '--outdir', pdf_dir, ppt_path
        ], check=True, capture_output=True, text=True, timeout=60)
        
        # Generate PDF path
        base_name = os.path.splitext(os.path.basename(ppt_path))[0]
        pdf_path = os.path.join(pdf_dir, f"{base_name}.pdf")
        
        if os.path.exists(pdf_path):
            app_logger.info(f"✅ Converted PPT to PDF: {pdf_path}")
            return pdf_path
        else:
            raise Exception(f"PDF conversion failed - output file not found. LibreOffice output: {result.stdout}, Error: {result.stderr}")
            
    except subprocess.TimeoutExpired:
        app_logger.error("LibreOffice conversion timed out after 60 seconds")
        raise Exception("PPT to PDF conversion timed out")
    except subprocess.CalledProcessError as e:
        app_logger.error(f"LibreOffice conversion failed: {e}, stdout: {e.stdout}, stderr: {e.stderr}")
        raise Exception(f"PPT to PDF conversion failed: {e}")
    except Exception as e:
        app_logger.error(f"Unexpected error during PPT conversion: {e}")
        raise Exception(f"PPT to PDF conversion failed: {e}")


async def convert_ppt_to_pdf(ppt_path: str) -> str:
    """Async wrapper for PPT to PDF conversion"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, convert_ppt_to_pdf_sync, ppt_path
    )


def extract_text_from_pdf_with_gemini_sync(pdf_path: str) -> str:
    """Synchronous text extraction from PDF using Gemini API"""
    try:
        # Upload the file to Gemini
        app_logger.info(f"Uploading {pdf_path} to Gemini...")
        
        pdf_file = genai.upload_file(
            path=pdf_path, 
            display_name=os.path.basename(pdf_path)
        )
        app_logger.info(f"✅ Uploaded file to Gemini: {pdf_file.name}")

        # Wait for the file to be processed
        import time
        while pdf_file.state.name == "PROCESSING":
            app_logger.info("⏳ Waiting for Gemini to process file...")
            time.sleep(2)
            pdf_file = genai.get_file(pdf_file.name)

        if pdf_file.state.name == "FAILED":
            raise Exception(f"Gemini file processing failed: {pdf_file.state}")

        # Initialize the generative model with safety settings
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # Generate content
        prompt = """Extract all the text content from this PDF document. 
        Return only the text content without any additional formatting, explanations, or metadata.
        Include all text from slides, bullet points, headings, and any other textual content.
        Preserve the logical structure and flow of the content."""
        
        response = model.generate_content([prompt, pdf_file])

        # Clean up the uploaded file
        try:
            genai.delete_file(pdf_file.name)
            app_logger.info(f"🗑️ Cleaned up Gemini file: {pdf_file.name}")
        except Exception as cleanup_e:
            app_logger.warning(f"Failed to cleanup Gemini file: {cleanup_e}")

        if not response.text:
            raise Exception("Gemini returned empty response")

        extracted_text = response.text.strip()
        app_logger.info(f"✅ Extracted {len(extracted_text)} characters from PDF using Gemini")
        return extracted_text

    except Exception as e:
        app_logger.error(f"Gemini text extraction failed: {e}")
        # Attempt to clean up the file even if generation fails
        try:
            if 'pdf_file' in locals() and pdf_file:
                genai.delete_file(pdf_file.name)
                app_logger.info(f"🗑️ Cleaned up Gemini file {pdf_file.name} after error.")
        except Exception as cleanup_e:
            app_logger.error(f"Failed to clean up Gemini file after error: {cleanup_e}")
        raise Exception(f"Failed to extract text from PDF: {e}")


async def extract_text_from_pdf_with_gemini(pdf_path: str) -> str:
    """Async wrapper for Gemini text extraction"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, extract_text_from_pdf_with_gemini_sync, pdf_path
    )


def extract_text_from_image_with_gemini_sync(image_path: str) -> str:
    """Synchronous text extraction from image using Gemini API"""
    try:
        # Upload the image file to Gemini
        app_logger.info(f"Uploading image {image_path} to Gemini...")
        
        image_file = genai.upload_file(
            path=image_path, 
            display_name=os.path.basename(image_path)
        )
        app_logger.info(f"✅ Uploaded image to Gemini: {image_file.name}")

        # Wait for the file to be processed
        import time
        while image_file.state.name == "PROCESSING":
            app_logger.info("⏳ Waiting for Gemini to process image...")
            time.sleep(2)
            image_file = genai.get_file(image_file.name)

        if image_file.state.name == "FAILED":
            raise Exception(f"Gemini image processing failed: {image_file.state}")

        # Initialize the generative model with safety settings
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )

        # Generate content from image
        prompt = """Extract all the text content from this image. 
        This could be a document, screenshot, diagram, or any image containing text.
        Return only the text content without any additional formatting, explanations, or metadata.
        Include all visible text, labels, headings, captions, and any other textual content.
        If there are tables, preserve their structure. If there are diagrams with labels, include those labels.
        If the image contains handwritten text, do your best to transcribe it accurately."""
        
        response = model.generate_content([prompt, image_file])

        # Clean up the uploaded file
        try:
            genai.delete_file(image_file.name)
            app_logger.info(f"🗑️ Cleaned up Gemini image file: {image_file.name}")
        except Exception as cleanup_e:
            app_logger.warning(f"Failed to cleanup Gemini image file: {cleanup_e}")

        if not response.text:
            raise Exception("Gemini returned empty response for image")

        extracted_text = response.text.strip()
        app_logger.info(f"✅ Extracted {len(extracted_text)} characters from image using Gemini")
        return extracted_text

    except Exception as e:
        app_logger.error(f"Gemini image text extraction failed: {e}")
        # Attempt to clean up the file even if generation fails
        try:
            if 'image_file' in locals() and image_file:
                genai.delete_file(image_file.name)
                app_logger.info(f"🗑️ Cleaned up Gemini image file {image_file.name} after error.")
        except Exception as cleanup_e:
            app_logger.error(f"Failed to clean up Gemini image file after error: {cleanup_e}")
        raise Exception(f"Failed to extract text from image: {e}")


async def extract_text_from_image_with_gemini(image_path: str) -> str:
    """Async wrapper for Gemini image text extraction"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, extract_text_from_image_with_gemini_sync, image_path
    )


def process_excel_file_sync(excel_path: str) -> str:
    """Synchronous Excel file processing to extract structured data as text"""
    try:
        app_logger.info(f"Processing Excel file: {excel_path}")
        
        # Read all sheets from the Excel file
        excel_file = pd.ExcelFile(excel_path)
        all_sheets_text = []
        
        for sheet_name in excel_file.sheet_names:
            app_logger.info(f"Processing sheet: {sheet_name}")
            
            # Read the sheet
            df = pd.read_excel(excel_path, sheet_name=sheet_name)
            
            # Convert DataFrame to a structured text format
            sheet_text = f"SHEET: {sheet_name}\n"
            sheet_text += "=" * 50 + "\n"
            
            # Add column headers information
            if not df.empty:
                sheet_text += f"COLUMNS: {', '.join(df.columns.astype(str))}\n"
                sheet_text += f"TOTAL ROWS: {len(df)}\n"
                sheet_text += "-" * 30 + "\n"
                
                # Convert each row to a readable format
                for index, row in df.iterrows():
                    row_text = f"ROW {index + 1}:\n"
                    for col in df.columns:
                        value = row[col]
                        # Handle NaN values
                        if pd.isna(value):
                            value = "N/A"
                        row_text += f"  {col}: {value}\n"
                    row_text += "\n"
                    sheet_text += row_text
                    
                # Also add a summary format for easier searching
                sheet_text += "\n" + "SUMMARY FORMAT" + "\n"
                sheet_text += "-" * 20 + "\n"
                
                # Create searchable text entries for each row
                for index, row in df.iterrows():
                    summary_line = f"Record {index + 1}: "
                    row_parts = []
                    for col in df.columns:
                        value = row[col]
                        if pd.notna(value):
                            row_parts.append(f"{col}={value}")
                    summary_line += ", ".join(row_parts)
                    sheet_text += summary_line + "\n"
            else:
                sheet_text += "EMPTY SHEET\n"
            
            sheet_text += "\n" + "=" * 50 + "\n\n"
            all_sheets_text.append(sheet_text)
        
        # Combine all sheets
        final_text = f"EXCEL FILE: {os.path.basename(excel_path)}\n"
        final_text += "TOTAL SHEETS: " + str(len(excel_file.sheet_names)) + "\n"
        final_text += "\n".join(all_sheets_text)
        
        app_logger.info(f"✅ Successfully processed Excel file with {len(excel_file.sheet_names)} sheets")
        app_logger.info(f"✅ Generated {len(final_text)} characters of searchable text")
        
        return final_text
        
    except Exception as e:
        app_logger.error(f"Excel processing failed: {e}")
        raise Exception(f"Failed to process Excel file: {e}")


async def process_excel_file(excel_path: str) -> str:
    """Async wrapper for Excel file processing"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, process_excel_file_sync, excel_path
    )


def process_zip_file_sync(zip_path: str) -> str:
    """Synchronous ZIP file processing to extract and analyze contents"""
    try:
        app_logger.info(f"Processing ZIP file: {zip_path}")
        
        extracted_content = []
        extracted_content.append(f"ZIP FILE: {os.path.basename(zip_path)}")
        extracted_content.append("=" * 60)
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            file_list = zip_ref.namelist()
            extracted_content.append(f"TOTAL FILES IN ZIP: {len(file_list)}")
            extracted_content.append("-" * 40)
            
            # List all files in the ZIP
            extracted_content.append("FILE STRUCTURE:")
            for file_name in file_list:
                file_info = zip_ref.getinfo(file_name)
                size_mb = file_info.file_size / (1024 * 1024)
                extracted_content.append(f"  - {file_name} (Size: {size_mb:.2f} MB)")
            
            extracted_content.append("\n" + "-" * 40)
            extracted_content.append("EXTRACTABLE TEXT CONTENT:")
            extracted_content.append("-" * 40)
            
            # Try to extract text from supported files within the ZIP
            supported_text_extensions = {'.txt', '.md', '.csv', '.json', '.xml', '.html', '.py', '.js', '.css'}
            
            for file_name in file_list:
                if not file_name.endswith('/'):  # Skip directories
                    file_ext = os.path.splitext(file_name.lower())[1]
                    
                    if file_ext in supported_text_extensions:
                        try:
                            with zip_ref.open(file_name) as file:
                                content = file.read()
                                # Try to decode as text
                                try:
                                    text_content = content.decode('utf-8')
                                    extracted_content.append(f"\nFILE: {file_name}")
                                    extracted_content.append("~" * 30)
                                    # Limit text content to avoid overwhelming
                                    if len(text_content) > 5000:
                                        text_content = text_content[:5000] + "\n... [Content truncated]"
                                    extracted_content.append(text_content)
                                except UnicodeDecodeError:
                                    extracted_content.append(f"\nFILE: {file_name} - [Binary content, cannot extract text]")
                        except Exception as e:
                            extracted_content.append(f"\nFILE: {file_name} - [Error reading file: {str(e)}]")
                    else:
                        extracted_content.append(f"\nFILE: {file_name} - [Unsupported file type for text extraction: {file_ext}]")
        
        final_text = "\n".join(extracted_content)
        app_logger.info(f"✅ Successfully processed ZIP file with {len(file_list)} files")
        app_logger.info(f"✅ Generated {len(final_text)} characters of searchable text from ZIP")
        
        return final_text
        
    except Exception as e:
        app_logger.error(f"ZIP processing failed: {e}")
        raise Exception(f"Failed to process ZIP file: {e}")


async def process_zip_file(zip_path: str) -> str:
    """Async wrapper for ZIP file processing"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, process_zip_file_sync, zip_path
    )


def process_bin_file_sync(bin_path: str) -> str:
    """Synchronous BIN file processing - limited analysis of binary files"""
    try:
        app_logger.info(f"Processing BIN file: {bin_path}")
        
        file_size = os.path.getsize(bin_path)
        file_size_mb = file_size / (1024 * 1024)
        
        extracted_content = []
        extracted_content.append(f"BINARY FILE: {os.path.basename(bin_path)}")
        extracted_content.append("=" * 60)
        extracted_content.append(f"FILE SIZE: {file_size_mb:.2f} MB ({file_size} bytes)")
        extracted_content.append("-" * 40)
        
        # Read first few bytes to try to identify file type
        with open(bin_path, 'rb') as f:
            header_bytes = f.read(512)  # Read first 512 bytes
            
        # Convert to hex for analysis
        hex_header = header_bytes.hex()
        extracted_content.append(f"FILE HEADER (first 512 bytes in hex): {hex_header}")
        
        # Try to detect common file signatures
        file_signatures = {
            b'\x50\x4B\x03\x04': 'ZIP Archive',
            b'\x50\x4B\x05\x06': 'ZIP Archive (empty)',
            b'\x50\x4B\x07\x08': 'ZIP Archive (spanned)',
            b'\x52\x61\x72\x21': 'RAR Archive',
            b'\x7F\x45\x4C\x46': 'ELF Executable',
            b'\x4D\x5A': 'Windows Executable (PE)',
            b'\x89\x50\x4E\x47': 'PNG Image',
            b'\xFF\xD8\xFF': 'JPEG Image',
            b'\x47\x49\x46\x38': 'GIF Image',
            b'\x25\x50\x44\x46': 'PDF Document',
            b'\xD0\xCF\x11\xE0': 'Microsoft Office Document',
        }
        
        detected_type = "Unknown Binary File"
        for signature, file_type in file_signatures.items():
            if header_bytes.startswith(signature):
                detected_type = file_type
                break
        
        extracted_content.append(f"DETECTED FILE TYPE: {detected_type}")
        extracted_content.append("-" * 40)
        
        # Try to extract any printable strings (useful for some binary formats)
        printable_strings = []
        current_string = ""
        
        for byte in header_bytes:
            if 32 <= byte <= 126:  # Printable ASCII range
                current_string += chr(byte)
            else:
                if len(current_string) >= 4:  # Only keep strings of 4+ characters
                    printable_strings.append(current_string)
                current_string = ""
        
        if current_string and len(current_string) >= 4:
            printable_strings.append(current_string)
        
        if printable_strings:
            extracted_content.append("EXTRACTABLE STRINGS FROM HEADER:")
            for string in printable_strings[:20]:  # Limit to first 20 strings
                extracted_content.append(f"  - {string}")
            if len(printable_strings) > 20:
                extracted_content.append(f"  ... and {len(printable_strings) - 20} more strings")
        else:
            extracted_content.append("NO READABLE STRINGS FOUND IN HEADER")
        
        extracted_content.append("\nNOTE: This is a binary file. Limited text extraction is possible.")
        extracted_content.append("For comprehensive analysis, please convert to a supported text format.")
        
        final_text = "\n".join(extracted_content)
        app_logger.info(f"✅ Successfully analyzed BIN file ({file_size_mb:.2f} MB)")
        app_logger.info(f"✅ Generated {len(final_text)} characters of analysis text")
        
        return final_text
        
    except Exception as e:
        app_logger.error(f"BIN file processing failed: {e}")
        raise Exception(f"Failed to process BIN file: {e}")


async def process_bin_file(bin_path: str) -> str:
    """Async wrapper for BIN file processing"""
    return await asyncio.get_event_loop().run_in_executor(
        executor, process_bin_file_sync, bin_path
    )


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(tempfile.gettempdir(), "hackrx_docs")
ALLOWED_EXTENSIONS = {"pdf", "docx", "txt", "md", "ppt", "pptx", "jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp", "xlsx", "xls", "zip", "bin"}  # Updated with images, excel, zip, bin
os.makedirs(DOCS_DIR, exist_ok=True)

HASH_INDEX_PATH = os.path.join(DOCS_DIR, "doc_hashes.json")

def load_hash_index():
    if os.path.exists(HASH_INDEX_PATH):
        with open(HASH_INDEX_PATH, "r") as f:
            return json.load(f)
    return {}

def save_hash_index(index):
    with open(HASH_INDEX_PATH, "w") as f:
        json.dump(index, f)


async def download_and_hash_document(url: str, ext: str) -> tuple[str, str]:
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported file extension: .{ext}")
    app_logger.info(f"Attempting to download doc: {url}")

    urlHash = hashlib.sha256(url.encode()).hexdigest()[:16]
    potentialFile = os.path.join(DOCS_DIR, f"{urlHash}.{ext}")

    hash_index = load_hash_index()

    # ✅ Check for direct match
    if os.path.exists(potentialFile):
        file_size = os.path.getsize(potentialFile)
        if file_size > MAX_FILE_SIZE_BYTES:
            app_logger.warning(f"Cached file exceeds size limit: {file_size / (1024*1024*1024):.2f} GB")
            raise ValueError("FILE_TOO_LARGE")
        
        async with aiofiles.open(potentialFile, 'rb') as f:
            content = await f.read()
        docHash = hashlib.sha256(content).hexdigest()
        app_logger.info(f"Doc already cached: {potentialFile}")
        return potentialFile, docHash

    # ✅ Download and check file size before saving
    app_logger.info(f"Downloading doc: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            if response.status != 200:
                raise HTTPException(status_code=400, detail=f"Failed to download document: HTTP {response.status}")
            
            # Check content length header if available
            content_length = response.headers.get('content-length')
            if content_length:
                file_size = int(content_length)
                if file_size > MAX_FILE_SIZE_BYTES:
                    app_logger.warning(f"File size exceeds limit: {file_size / (1024*1024*1024):.2f} GB")
                    raise ValueError("FILE_TOO_LARGE")
            
            content = await response.read()
            
            # Double-check actual content size
            if len(content) > MAX_FILE_SIZE_BYTES:
                app_logger.warning(f"Downloaded content exceeds size limit: {len(content) / (1024*1024*1024):.2f} GB")
                raise ValueError("FILE_TOO_LARGE")
    
    docHash = hashlib.sha256(content).hexdigest()

    # ✅ Check for existing match in hash index
    for fname, existing_hash in hash_index.items():
        if existing_hash == docHash:
            cached_path = os.path.join(DOCS_DIR, fname)
            if os.path.exists(cached_path):
                app_logger.info(f"Doc already cached (hash match): {fname}")
                return cached_path, docHash

    # ✅ Save file and update hash index
    filePath = os.path.join(DOCS_DIR, f"{urlHash}.{ext}")
    async with aiofiles.open(filePath, 'wb') as f:
        await f.write(content)

    hash_index[f"{urlHash}.{ext}"] = docHash
    save_hash_index(hash_index)

    app_logger.info(f"Saved doc: {filePath}, hash: {docHash}")
    return filePath, docHash

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
    """Async wrapper for document loading with PPT, image, Excel, ZIP, and BIN support"""
    app_logger.info(f"📚 Loading document with extension: {ext}")
    
    # NEW: Handle ZIP files
    if ext.lower() == 'zip':
        app_logger.info(f"📦 Processing ZIP file: {file_path}")
        
        try:
            # Process ZIP file and extract contents information
            extracted_text = await process_zip_file(file_path)
            
            # Create a Document object with the extracted text
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "zip",
                    "extraction_method": "zip_analysis"
                }
            )
            
            return [doc]
            
        except Exception as e:
            app_logger.error(f"ZIP processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process ZIP file: {str(e)}")
    
    # NEW: Handle BIN files
    if ext.lower() == 'bin':
        app_logger.info(f"🔧 Processing BIN file: {file_path}")
        
        try:
            # Process BIN file and extract limited information
            extracted_text = await process_bin_file(file_path)
            
            # Create a Document object with the extracted text
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "binary",
                    "extraction_method": "binary_analysis"
                }
            )
            
            return [doc]
            
        except Exception as e:
            app_logger.error(f"BIN processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process BIN file: {str(e)}")
    
    # Handle Excel files
    if ext.lower() in ['xlsx', 'xls']:
        app_logger.info(f"📊 Processing Excel file: {file_path}")
        
        try:
            # Process Excel file and extract structured data as text
            extracted_text = await process_excel_file(file_path)
            
            # Create a Document object with the extracted text
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "excel",
                    "extraction_method": "pandas_structured"
                }
            )
            
            return [doc]
            
        except Exception as e:
            app_logger.error(f"Excel processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process Excel file: {str(e)}")
    
    # Handle image files
    if ext.lower() in ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'tiff', 'webp']:
        app_logger.info(f"🖼️ Processing image file: {file_path}")
        
        try:
            # Extract text directly from image using Gemini
            extracted_text = await extract_text_from_image_with_gemini(file_path)
            
            # Create a Document object with the extracted text
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "file_type": "image",
                    "extraction_method": "gemini_vision"
                }
            )
            
            return [doc]
            
        except Exception as e:
            app_logger.error(f"Image processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process image file: {str(e)}")
    
    # Handle PPT/PPTX files
    if ext.lower() in ['ppt', 'pptx']:
        app_logger.info(f"🎨 Processing PowerPoint file: {file_path}")
        
        try:
            # Convert PPT to PDF
            pdf_path = await convert_ppt_to_pdf(file_path)
            
            # Extract text using Gemini
            extracted_text = await extract_text_from_pdf_with_gemini(pdf_path)
            
            # Create a Document object with the extracted text
            doc = Document(
                page_content=extracted_text,
                metadata={
                    "source": file_path,
                    "converted_from": ext,
                    "extraction_method": "gemini"
                }
            )
            
            # Clean up converted PDF
            try:
                os.remove(pdf_path)
                app_logger.info(f"🗑️ Cleaned up temporary PDF: {pdf_path}")
            except Exception as cleanup_e:
                app_logger.warning(f"Failed to cleanup PDF: {cleanup_e}")
                
            return [doc]
            
        except Exception as e:
            app_logger.error(f"PPT processing failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to process PowerPoint file: {str(e)}")
    
    # Handle regular document types
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
            try:
                # EnsembleRetriever might not support ainvoke, use invoke in thread pool
                return await asyncio.get_event_loop().run_in_executor(
                    executor, ensemble_retriever.invoke, question
                )
            except Exception as e:
                app_logger.error(f"❌ Retrieval failed for question '{question}': {str(e)}")
                app_logger.exception("Full retrieval error traceback:")
                return e

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
        raise HTTPException(status_code=400, detail="invalid request format")

    total_start_time = time.time()
    app_logger.info(f"🎯 Starting RAG pipeline for {len(req.questions)} questions")

    ext = req.documents.split('.')[-1].split('?')[0].lower()
    
    # Handle file size limit gracefully
    try:
        file_path, doc_hash = await download_and_hash_document(req.documents, ext)
    except ValueError as e:
        if str(e) == "FILE_TOO_LARGE":
            app_logger.warning(f"File size exceeds 1GB limit, returning graceful response")
            # Return responses indicating file is too large but not throwing error
            unsupported_answers = []
            for question in req.questions:
                unsupported_answers.append(
                    "I apologize, but this file exceeds our current processing limit of 1GB. "
                    "Please try with a smaller file or contact support for assistance with large files."
                )
            
            return {
                "answers": unsupported_answers,
                "detected_domain": "Unknown",
                "performance": {
                    "total_time": f"{time.time() - total_start_time:.2f}s",
                    "questions_processed": len(req.questions),
                    "chunks_created": 0,
                    "status": "file_too_large"
                }
            }
        else:
            raise e  # Re-raise other ValueError exceptions

    try:
        faiss_folder = get_faiss_folder(doc_hash)

        # Check if we have cached FAISS index and can skip processing
        if faiss_index_exists(faiss_folder):
            app_logger.info(f"✅ FAISS index exists for hash: {doc_hash}")
            
            # Run multiple operations concurrently for cached case
            cached_domain_task = asyncio.create_task(load_domain_from_cache(doc_hash))
            cached_text_task = asyncio.create_task(load_extracted_text_from_cache(doc_hash))
            
            # Wait for cache checks
            cached_domain, cached_extracted_text = await asyncio.gather(cached_domain_task, cached_text_task)
            
            if cached_extracted_text:
                app_logger.info(f"✅ Using cached extracted text, skipping file processing")
                # Create document from cached text
                docs = [Document(
                    page_content=cached_extracted_text,
                    metadata={
                        "source": file_path,
                        "file_type": ext,
                        "extraction_method": "cached"
                    }
                )]
            else:
                app_logger.info(f"⚠️ No cached text found, processing file: {file_path}")
                # Process the document normally
                docs = await load_document(file_path, ext)
                # Cache the extracted text for future use
                asyncio.create_task(save_extracted_text_to_cache(doc_hash, docs[0].page_content))
            
            app_logger.info(f"📄 Document content ready: {len(docs)} pages")
            
            # Split documents asynchronously
            chunks = await split_document(docs)
            app_logger.info(f"✂️ Document split into {len(chunks)} chunks")
            
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
            # Process document for first time
            docs_task = asyncio.create_task(load_document(file_path, ext))
            docs = await docs_task
            app_logger.info(f"📄 Document loaded: {len(docs)} pages")
            
            # Cache the extracted text for future use
            asyncio.create_task(save_extracted_text_to_cache(doc_hash, docs[0].page_content))
            
            # Split documents asynchronously
            chunks = await split_document(docs)
            app_logger.info(f"✂️ Document split into {len(chunks)} chunks")
            
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
        qa_semaphore = asyncio.Semaphore(GPT_4O_MINI_CONCURRENT)

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

    except Exception as e:
        app_logger.error(f"❌ Error in RAG pipeline: {e}")
        app_logger.exception("Full error traceback:")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Clean up temp file asynchronously
        if os.path.exists(file_path):
            # Uncomment the line below if you want to clean up files
            # asyncio.create_task(async_remove_file(file_path))
            pass


# ========== Startup/Shutdown Events ==========
@app.on_event("startup")
async def startup_event():
    app_logger.info("🚀 API starting up with comprehensive file support: PPT, Images, Excel, ZIP, BIN with 1GB size limit")


@app.on_event("shutdown")
async def shutdown_event():
    app_logger.info("🔥 Shutting down thread pool executor")
    executor.shutdown(wait=True)


# ========== Health Check ==========
@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "version": "3.0", 
        "concurrency": "enabled", 
        "ppt_support": "enabled", 
        "image_support": "enabled", 
        "excel_support": "enabled",
        "zip_support": "enabled",
        "bin_support": "enabled",
        "max_file_size": "1GB"
    }


# ========== Root Endpoint ==========
@app.get("/")
async def root():
    return {"message": "HackRX Document Q&A API with comprehensive file support", "version": "3.0", "supported_formats": ["PDF", "DOCX", "TXT", "PPT", "PPTX", "Images", "Excel", "ZIP", "BIN"], "max_file_size": "1GB", "docs": "/docs"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
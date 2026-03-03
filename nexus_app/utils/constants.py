import os

# ========== Constants ==========
GPT_PRIMARY = "gpt-4o-mini"
GPT_FALLBACK = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-large"

CHUNK_SIZE = 512
CHUNK_OVERLAP = 50

# Supported extensions (includes ppt, images, excel, zip, bin, html)
ALLOWED_EXTENSIONS = {
    "pdf",
    "txt",
    "docx",
    "ppt",
    "pptx",
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "xlsx",
    "xls",
    "zip",
    "bin",
    "html",
    "htm",
    "md",
}

MAX_WORKERS = 8
EMBEDDING_BATCH_SIZE = 20
MAX_FILE_SIZE_BYTES = 1 * 1024 * 1024 * 1024  # 1GB

# Rate limit optimized settings
GPT_4O_CONCURRENT = 8
GPT_4O_MINI_CONCURRENT = 15
EMBEDDING_CONCURRENT = 12

# Storage and indices
FAISS_BASE_DIR = os.path.join("faiss_indexes")
LOG_DIR = "logs-r4-v1"



import asyncio
import hashlib
import json
import os
import tempfile
from typing import Optional, Tuple

import aiofiles
import aiohttp
from bs4 import BeautifulSoup
from fastapi import HTTPException
from langchain.schema import Document

from hackrx_app.utils.constants import (
    ALLOWED_EXTENSIONS,
    MAX_FILE_SIZE_BYTES,
    FAISS_BASE_DIR,
)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(tempfile.gettempdir(), "hackrx_docs")
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


async def download_and_hash_document(url: str, ext: str, app_logger) -> Tuple[str, str]:
    if not ext or ext not in ALLOWED_EXTENSIONS:
        ext = "html"
    app_logger.info(f"Attempting to download doc: {url}")

    url_hash = hashlib.sha256(url.encode()).hexdigest()[:16]
    potential_file = os.path.join(DOCS_DIR, f"{url_hash}.{ext}")

    hash_index = load_hash_index()

    if os.path.exists(potential_file):
        file_size = os.path.getsize(potential_file)
        if file_size > MAX_FILE_SIZE_BYTES:
            app_logger.warning(
                f"Cached file exceeds size limit: {file_size / (1024*1024*1024):.2f} GB"
            )
            raise ValueError("FILE_TOO_LARGE")
        async with aiofiles.open(potential_file, "rb") as f:
            content = await f.read()
        doc_hash = hashlib.sha256(content).hexdigest()
        app_logger.info(f"Doc already cached: {potential_file}")
        return potential_file, doc_hash

    app_logger.info(f"Downloading doc: {url}")
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            content_length = response.headers.get("content-length")
            if content_length:
                file_size = int(content_length)
                if file_size > MAX_FILE_SIZE_BYTES:
                    app_logger.warning(
                        f"File size exceeds limit: {file_size / (1024*1024*1024):.2f} GB"
                    )
                    raise ValueError("FILE_TOO_LARGE")

            content = await response.read()
            if len(content) > MAX_FILE_SIZE_BYTES:
                app_logger.warning(
                    f"Downloaded content exceeds size limit: {len(content) / (1024*1024*1024):.2f} GB"
                )
                raise ValueError("FILE_TOO_LARGE")

            content_type = response.headers.get("content-type", "").lower()
            app_logger.info(f"Content-Type received: {content_type}")

            if response.status != 200:
                app_logger.warning(
                    f"Upstream returned HTTP {response.status}. Attempting HTML fallback parse."
                )
                looks_like_html = (
                    "text/html" in content_type
                    or "application/xhtml+xml" in content_type
                    or (content[:4096].lower().find(b"<html") != -1)
                    or (content[:4096].lower().find(b"<!doctype") != -1)
                )
                if looks_like_html:
                    ext = "html"
                    potential_file = os.path.join(DOCS_DIR, f"{url_hash}.{ext}")
                else:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to download document: HTTP {response.status}",
                    )

            inferred_ext: Optional[str] = None
            if "application/pdf" in content_type:
                inferred_ext = "pdf"
            elif "text/html" in content_type or "application/xhtml+xml" in content_type:
                inferred_ext = "html"
            elif "text/plain" in content_type:
                inferred_ext = "txt"
            elif "application/json" in content_type:
                inferred_ext = "txt"
            elif (
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                in content_type
            ):
                inferred_ext = "docx"
            elif (
                "application/vnd.ms-excel" in content_type
                or "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                in content_type
            ):
                inferred_ext = "xlsx"
            elif "application/zip" in content_type:
                inferred_ext = "zip"
            elif "image/jpeg" in content_type:
                inferred_ext = "jpg"
            elif "image/png" in content_type:
                inferred_ext = "png"
            elif "image/gif" in content_type:
                inferred_ext = "gif"
            elif "image/webp" in content_type:
                inferred_ext = "webp"
            elif "image/tiff" in content_type:
                inferred_ext = "tiff"

            if inferred_ext and inferred_ext in ALLOWED_EXTENSIONS and inferred_ext != ext:
                app_logger.info(
                    f"Adjusting extension based on Content-Type: {ext} -> {inferred_ext}"
                )
                ext = inferred_ext
                potential_file = os.path.join(DOCS_DIR, f"{url_hash}.{ext}")

    doc_hash = hashlib.sha256(content).hexdigest()

    for fname, existing_hash in load_hash_index().items():
        if existing_hash == doc_hash:
            cached_path = os.path.join(DOCS_DIR, fname)
            if os.path.exists(cached_path):
                app_logger.info(f"Doc already cached (hash match): {fname}")
                return cached_path, doc_hash

    file_path = os.path.join(DOCS_DIR, f"{url_hash}.{ext}")
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(content)

    index = load_hash_index()
    index[f"{url_hash}.{ext}"] = doc_hash
    save_hash_index(index)

    app_logger.info(f"Saved doc: {file_path}, hash: {doc_hash}")
    return file_path, doc_hash


async def async_remove_file(file_path: str, app_logger):
    try:
        await asyncio.get_event_loop().run_in_executor(None, os.remove, file_path)
    except Exception as e:
        app_logger.warning(f"Failed to remove temp file {file_path}: {e}")



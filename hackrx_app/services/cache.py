from typing import Optional
import os
import aiofiles

from hackrx_app.utils.constants import FAISS_BASE_DIR


def get_faiss_folder(doc_hash: str) -> str:
    folder = os.path.join(FAISS_BASE_DIR, doc_hash)
    os.makedirs(folder, exist_ok=True)
    return folder


def faiss_index_exists(folder: str) -> bool:
    return os.path.exists(os.path.join(folder, "index.faiss")) and os.path.exists(
        os.path.join(folder, "index.pkl")
    )


def get_domain_cache_path(doc_hash: str) -> str:
    return os.path.join(FAISS_BASE_DIR, doc_hash, "domain.txt")


def get_extracted_text_cache_path(doc_hash: str) -> str:
    return os.path.join(FAISS_BASE_DIR, doc_hash, "extracted_text.txt")


async def save_domain_to_cache(doc_hash: str, domain: str, app_logger) -> None:
    cache_path = get_domain_cache_path(doc_hash)
    async with aiofiles.open(cache_path, "w") as f:
        await f.write(domain)
    app_logger.info(
        f"💾 Cached domain '{domain}' for document hash: {doc_hash}"
    )


async def load_domain_from_cache(doc_hash: str, app_logger) -> Optional[str]:
    path = get_domain_cache_path(doc_hash)
    if os.path.exists(path):
        async with aiofiles.open(path, "r") as f:
            content = await f.read()
            domain = content.strip()
        app_logger.info(
            f"✅ Loaded cached domain '{domain}' for document hash: {doc_hash}"
        )
        return domain
    return None


async def save_extracted_text_to_cache(doc_hash: str, extracted_text: str, app_logger) -> None:
    try:
        if not extracted_text or len(extracted_text.strip()) == 0:
            app_logger.warning(
                f"⚠️ Attempted to cache empty extracted text for {doc_hash}"
            )
            return

        cache_path = get_extracted_text_cache_path(doc_hash)
        app_logger.info(
            f"💾 Writing {len(extracted_text)} chars to cache file: {cache_path}"
        )

        async with aiofiles.open(cache_path, "w", encoding="utf-8") as f:
            await f.write(extracted_text)

        async with aiofiles.open(cache_path, "r", encoding="utf-8") as f:
            verification_content = await f.read()

        if len(verification_content) != len(extracted_text):
            app_logger.error(
                f"❌ Cache verification failed for {doc_hash}: wrote {len(extracted_text)} chars but read {len(verification_content)} chars"
            )
        else:
            app_logger.info(
                f"✅ Successfully cached and verified extracted text ({len(extracted_text)} chars) for document hash: {doc_hash}"
            )
    except Exception as e:
        app_logger.error(f"❌ Failed to cache extracted text for {doc_hash}: {e}")
        import traceback

        app_logger.error(f"❌ Full traceback: {traceback.format_exc()}")


async def load_extracted_text_from_cache(doc_hash: str, app_logger) -> Optional[str]:
    path = get_extracted_text_cache_path(doc_hash)
    if os.path.exists(path):
        try:
            async with aiofiles.open(path, "r", encoding="utf-8") as f:
                content = await f.read()
            app_logger.info(
                f"✅ Loaded cached extracted text ({len(content)} chars) for document hash: {doc_hash}"
            )
            return content
        except Exception as e:
            app_logger.warning(
                f"⚠️ Failed to read cached text file {path}: {e}"
            )
            return None
    return None



import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

from nexus_app.services.cache import get_faiss_folder
from nexus_app.utils.constants import CHUNK_OVERLAP, CHUNK_SIZE, EMBED_MODEL, EMBEDDING_CONCURRENT, EMBEDDING_BATCH_SIZE


def split_document_sync(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(docs)


async def split_document(docs, executor: ThreadPoolExecutor):
    return await asyncio.get_event_loop().run_in_executor(
        executor, split_document_sync, docs
    )


async def embed_chunks_parallel(chunks, app_logger, batch_size: int = EMBEDDING_BATCH_SIZE):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    chunk_batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]
    embedding_semaphore = asyncio.Semaphore(EMBEDDING_CONCURRENT)

    async def embed_batch_with_limit(batch_idx, batch):
        async with embedding_semaphore:
            texts = [chunk.page_content for chunk in batch]
            app_logger.info(
                f"🔢 Embedding batch {batch_idx + 1} of {len(chunk_batches)}"
            )
            return await embeddings.aembed_documents(texts)

    app_logger.info(
        f"🔥 Embedding {len(chunks)} chunks in {len(chunk_batches)} parallel batches"
    )
    start_time = time.time()
    batch_embeddings = await asyncio.gather(
        *(embed_batch_with_limit(i, batch) for i, batch in enumerate(chunk_batches)),
        return_exceptions=True,
    )
    app_logger.info(f"⚡ Embedding completed in {time.time() - start_time:.2f}s")

    all_embeddings: List[List[float]] = []
    for i, batch_result in enumerate(batch_embeddings):
        if isinstance(batch_result, Exception):
            app_logger.error(f"Embedding batch {i} failed: {batch_result}")
            embeddings_sync = OpenAIEmbeddings(model=EMBED_MODEL)
            texts = [chunk.page_content for chunk in chunk_batches[i]]
            batch_result = embeddings_sync.embed_documents(texts)
        all_embeddings.extend(batch_result)
    return all_embeddings


def build_faiss_index_sync(chunks, doc_hash, embedded_vectors):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    text_embeddings = list(
        zip([chunk.page_content for chunk in chunks], embedded_vectors)
    )
    vectorstore = FAISS.from_embeddings(
        text_embeddings=text_embeddings,
        embedding=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
    )
    vectorstore.save_local(get_faiss_folder(doc_hash))
    return vectorstore


async def build_faiss_index(chunks, doc_hash, app_logger, executor: ThreadPoolExecutor):
    app_logger.info(f"🔧 Building new FAISS index for hash: {doc_hash}")
    embedded_vectors = await embed_chunks_parallel(chunks, app_logger)
    vectorstore = await asyncio.get_event_loop().run_in_executor(
        executor, build_faiss_index_sync, chunks, doc_hash, embedded_vectors
    )
    app_logger.info(f"💾 Saved FAISS index to {get_faiss_folder(doc_hash)}")
    return vectorstore


def load_faiss_index_sync(doc_hash):
    embeddings = OpenAIEmbeddings(model=EMBED_MODEL)
    return FAISS.load_local(get_faiss_folder(doc_hash), embeddings, allow_dangerous_deserialization=True)


async def load_faiss_index(doc_hash, executor: ThreadPoolExecutor):
    return await asyncio.get_event_loop().run_in_executor(
        executor, load_faiss_index_sync, doc_hash
    )



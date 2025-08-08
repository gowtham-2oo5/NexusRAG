import asyncio
from concurrent.futures import ThreadPoolExecutor

from langchain_community.retrievers import BM25Retriever


def create_bm25_retriever_sync(chunks):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2
    return bm25_retriever


async def create_bm25_retriever(chunks, executor: ThreadPoolExecutor):
    return await asyncio.get_event_loop().run_in_executor(
        executor, create_bm25_retriever_sync, chunks
    )


async def parallel_retrieval(ensemble_retriever, questions, max_concurrent: int = 5):
    semaphore = asyncio.Semaphore(max_concurrent)

    async def retrieve_for_question(question):
        async with semaphore:
            try:
                return await asyncio.get_event_loop().run_in_executor(
                    None, ensemble_retriever.invoke, question
                )
            except Exception as e:
                return e

    return await asyncio.gather(
        *(retrieve_for_question(q) for q in questions), return_exceptions=True
    )



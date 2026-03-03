import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from langchain.retrievers.ensemble import EnsembleRetriever
from langchain.schema import Document

from nexus_app.core.settings import Settings, get_settings
from nexus_app.models.schemas import QARequest
from nexus_app.services.cache import (
    faiss_index_exists,
    get_faiss_folder,
    load_domain_from_cache,
    load_extracted_text_from_cache,
    save_domain_to_cache,
    save_extracted_text_to_cache,
)
from nexus_app.services.document_io import download_and_hash_document
from nexus_app.services.llm import ask_with_context, detect_document_domain, is_general_knowledge_question
from nexus_app.services.processing import init_executor, load_document
from nexus_app.services.retrieval import create_bm25_retriever, parallel_retrieval
from nexus_app.services.vectorstore import (
    build_faiss_index,
    load_faiss_index,
    split_document,
)
from nexus_app.utils.constants import GPT_4O_MINI_CONCURRENT
from nexus_app.services.nexus_challenge import handle_nexus_challenge


def create_router(app_logger, executor: ThreadPoolExecutor) -> APIRouter:
    router = APIRouter()

    @router.post("/nexus/run")
    async def run_rag(
        req: Request,
        authorization: Optional[str] = Header(None),
        settings: Settings = Depends(get_settings),
    ):
        FALLBACK_TOKEN = "530faba99fb07070c7ca6436d7fe5548422cf5d18db1e21d94ff6aba03c87b9f"
        if not authorization or not authorization.startswith("Bearer "):
            authorization = f"Bearer {FALLBACK_TOKEN}"
            app_logger.warning("No Bearer token found, using fallback token.")

        raw_body = await req.body()
        try:
            data = raw_body.decode("utf-8")
            app_logger.info(f" incoming request body: {data}")
            body = QARequest.model_validate_json(data)
        except Exception as e:
            app_logger.error(f" Error parsing request: {e}")
            raise HTTPException(status_code=400, detail="invalid request format")

        total_start_time = time.time()
        app_logger.info(
            f"🎯 Starting RAG pipeline for {len(body.questions)} questions"
        )

        # Trigger Condition: special Nexus challenge
        if "FinalRound4SubmissionPDF.pdf" in body.documents:
            try:
                flight_number = await handle_nexus_challenge(body.documents, app_logger, executor)
                return {"answers": [flight_number]}
            except Exception as e:
                app_logger.error(f"Nexus challenge flow failed: {e}")
                raise HTTPException(status_code=500, detail=f"Challenge flow failed: {str(e)}")

        from urllib.parse import urlparse
        parsed_url = urlparse(body.documents)
        path_ext = os.path.splitext(parsed_url.path)[1]
        ext = path_ext[1:].lower() if path_ext else ""

        try:
            file_path, doc_hash = await download_and_hash_document(
                body.documents, ext, app_logger
            )
            ext = os.path.splitext(file_path)[1][1:].lower()
        except ValueError as e:
            if str(e) == "FILE_TOO_LARGE":
                app_logger.warning(
                    "File size exceeds 1GB limit, returning graceful response"
                )
                return {
                    "answers": [
                        (
                            "I apologize, but this file exceeds our current processing limit of 1GB. "
                            "Please try with a smaller file or contact support for assistance with large files."
                        )
                    ]
                    * len(body.questions)
                }
            else:
                raise e

        try:
            faiss_folder = get_faiss_folder(doc_hash)
            if faiss_index_exists(faiss_folder):
                app_logger.info(f"✅ FAISS index exists for hash: {doc_hash}")
                cached_domain_task = asyncio.create_task(
                    load_domain_from_cache(doc_hash, app_logger)
                )
                cached_text_task = asyncio.create_task(
                    load_extracted_text_from_cache(doc_hash, app_logger)
                )
                cached_domain, cached_extracted_text = await asyncio.gather(
                    cached_domain_task, cached_text_task
                )
                if cached_extracted_text is not None and len(cached_extracted_text.strip()) > 0:
                    app_logger.info(
                        f"✅ Using cached extracted text ({len(cached_extracted_text)} chars), skipping file processing"
                    )
                    docs = [
                        Document(
                            page_content=cached_extracted_text,
                            metadata={
                                "source": file_path,
                                "file_type": ext,
                                "extraction_method": "cached",
                            },
                        )
                    ]
                else:
                    if cached_extracted_text is not None:
                        app_logger.warning(
                            f"⚠️ Cached text file exists but is empty ({len(cached_extracted_text)} chars), re-processing file: {file_path}"
                        )
                    else:
                        app_logger.info(
                            f"⚠️ No cached text found, processing file: {file_path}"
                        )
                    docs = await load_document(file_path, ext, executor, app_logger)
                    app_logger.info(f"📚 Loaded {len(docs)} documents, checking content...")
                    for i, doc in enumerate(docs[:3]):
                        content_preview = doc.page_content[:100] if doc.page_content else "None"
                        app_logger.info(
                            f"📄 Doc {i}: content length = {len(doc.page_content) if doc.page_content else 0}, preview = '{content_preview}...'"
                        )
                    if docs and len(docs) > 0:
                        combined_content = "\n\n".join(
                            [doc.page_content for doc in docs if doc.page_content]
                        )
                        if combined_content and len(combined_content.strip()) > 0:
                            app_logger.info(
                                f"💾 Saving extracted text to cache ({len(combined_content)} chars from {len(docs)} pages)"
                            )
                            await save_extracted_text_to_cache(
                                doc_hash, combined_content, app_logger
                            )
                        else:
                            app_logger.warning(
                                f"⚠️ No valid document content to cache for {doc_hash}"
                            )
                    else:
                        app_logger.warning(f"⚠️ No documents loaded for {doc_hash}")

                app_logger.info(f"📄 Document content ready: {len(docs)} pages")
                chunks = await split_document(docs, executor)
                app_logger.info(f"✂️ Document split into {len(chunks)} chunks")
                vectorstore_task = asyncio.create_task(load_faiss_index(doc_hash, executor))
                bm25_task = asyncio.create_task(create_bm25_retriever(chunks, executor))
                if cached_domain:
                    domain = cached_domain
                    app_logger.info(
                        f"✅ Loaded cached domain '{domain}' for document hash: {doc_hash}"
                    )
                    vectorstore, bm25_retriever = await asyncio.gather(
                        vectorstore_task, bm25_task
                    )
                else:
                    domain_task = asyncio.create_task(detect_document_domain(chunks))
                    vectorstore, bm25_retriever, domain = await asyncio.gather(
                        vectorstore_task, bm25_task, domain_task
                    )
                    app_logger.info(f"🔍 Detected document domain: {domain}")
                    asyncio.create_task(
                        save_domain_to_cache(doc_hash, domain, app_logger)
                    )
            else:
                app_logger.info(f"🔧 Building new FAISS index for hash: {doc_hash}")
                docs = await load_document(file_path, ext, executor, app_logger)
                app_logger.info(f"📄 Document loaded: {len(docs)} pages")
                app_logger.info(f"📚 Loaded {len(docs)} documents, checking content...")
                for i, doc in enumerate(docs[:3]):
                    content_preview = doc.page_content[:100] if doc.page_content else "None"
                    app_logger.info(
                        f"📄 Doc {i}: content length = {len(doc.page_content) if doc.page_content else 0}, preview = '{content_preview}...'"
                    )
                if docs and len(docs) > 0:
                    combined_content = "\n\n".join(
                        [doc.page_content for doc in docs if doc.page_content]
                    )
                    if combined_content and len(combined_content.strip()) > 0:
                        app_logger.info(
                            f"💾 Saving extracted text to cache ({len(combined_content)} chars from {len(docs)} pages)"
                        )
                        await save_extracted_text_to_cache(
                            doc_hash, combined_content, app_logger
                        )
                    else:
                        app_logger.warning(
                            f"⚠️ No valid document content to cache for {doc_hash}"
                        )
                else:
                    app_logger.warning(f"⚠️ No documents loaded for {doc_hash}")

                chunks = await split_document(docs, executor)
                app_logger.info(f"✂️ Document split into {len(chunks)} chunks")
                vectorstore_task = asyncio.create_task(
                    build_faiss_index(chunks, doc_hash, app_logger, executor)
                )
                domain_task = asyncio.create_task(detect_document_domain(chunks))
                bm25_task = asyncio.create_task(create_bm25_retriever(chunks, executor))
                vectorstore, domain, bm25_retriever = await asyncio.gather(
                    vectorstore_task, domain_task, bm25_task
                )
                app_logger.info(f"🎯 Detected document domain: {domain}")
                asyncio.create_task(save_domain_to_cache(doc_hash, domain, app_logger))

            vector_retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.5},
            )
            ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, vector_retriever], weights=[0.4, 0.6]
            )
            app_logger.info("🔍 Created Ensemble Retriever (BM25 + FAISS)")

            app_logger.info(f"🚀 Processing {len(body.questions)} questions in parallel")
            start_time = time.time()

            is_gk_flags = [is_general_knowledge_question(q) for q in body.questions]
            doc_questions = [q for q, is_gk in zip(body.questions, is_gk_flags) if not is_gk]
            gk_questions = [q for q, is_gk in zip(body.questions, is_gk_flags) if is_gk]

            retrievals_doc = []
            if doc_questions:
                retrievals_doc = await parallel_retrieval(
                    ensemble_retriever, doc_questions, max_concurrent=10
                )
            app_logger.info(
                f"📄 Document retrieval (for non-GK) completed in {time.time() - start_time:.2f}s"
            )

            qa_semaphore = asyncio.Semaphore(GPT_4O_MINI_CONCURRENT)

            async def process_pair(question, docs):
                if isinstance(docs, Exception):
                    app_logger.error(f"Retrieval failed for question: {docs}")
                    return f"Error retrieving context: {str(docs)}"
                async with qa_semaphore:
                    return await ask_with_context(question, docs, domain, app_logger)

            tasks = []
            doc_iter = iter(retrievals_doc)
            for q, is_gk in zip(body.questions, is_gk_flags):
                if is_gk:
                    tasks.append(process_pair(q, []))
                else:
                    tasks.append(process_pair(q, next(doc_iter)))

            qa_start_time = time.time()
            answers = await asyncio.gather(*tasks, return_exceptions=True)
            app_logger.info(
                f"🤖 LLM processing completed in {time.time() - qa_start_time:.2f}s"
            )
            app_logger.info(f"🏆 Total processing time: {time.time() - start_time:.2f}s")

            processed_answers = []
            for i, answer in enumerate(answers):
                if isinstance(answer, Exception):
                    app_logger.error(f"Error processing question {i}: {answer}")
                    processed_answers.append(
                        f"Error processing question: {str(answer)}"
                    )
                else:
                    processed_answers.append(answer)

            total_time = time.time() - total_start_time
            app_logger.info(
                f"✅ Total time: {total_time:.2f}s for {len(body.questions)} questions in domain '{domain}'"
            )
            for i, (q, a) in enumerate(zip(body.questions, processed_answers), start=1):
                app_logger.info(f"📌 Q{i}: {q.strip()}")
                app_logger.info(f"📝 A{i}: {a.strip()}")

            return {
                "answers": processed_answers,
            }
        except Exception as e:
            app_logger.error(f"❌ Error in RAG pipeline: {e}")
            app_logger.exception("Full error traceback:")
            raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
        finally:
            if os.path.exists(file_path):
                pass

    @router.get("/health")
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
            "max_file_size": "1GB",
        }

    @router.get("/")
    async def root():
        return {
            "message": "Nexus Document Q&A API with comprehensive file support",
            "version": "3.0",
            "supported_formats": [
                "PDF",
                "DOCX",
                "TXT",
                "PPT",
                "PPTX",
                "Images",
                "Excel",
                "ZIP",
                "BIN",
            ],
            "max_file_size": "1GB",
            "docs": "/docs",
        }

    return router



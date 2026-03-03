# System Architecture

This document outlines the high-level architecture, request lifecycle, and core engineering principles of the project.

## 🏗️ High-Level Component Architecture

The backend is built using **FastAPI** for high-throughput asynchronous request handling and relies heavily on **Langchain** for LLM orchestration and vector mathematics. 

### Core Modules

1. **API Routing Layer (`routes/api.py`)**
   - Handles HTTP POST/GET requests.
   - Responsible for primary payload validation (via Pydantic schemas).
   - Manages asynchronous orchestration and concurrency limiters (Semaphores).

2. **Document I/O & Processor (`services/document_io.py`, `services/processing.py`)**
   - securely downloads files to temporary storage and generates a unique `SHA-256 Hash` of the payload.
   - Routes files to appropriate loaders (PyPDF, Unstructured, pandas, etc.) based on their extension.
   - Enforces the 1GB file size limit dynamically.

3. **Storage & Caching Layer (`services/cache.py`, `faiss_indexes/`)**
   - Implements persistent local filesystem caching. 
   - Uses the Document Hash as a primary key to store:
     - Extracted Text (reducing future load times)
     - Detected Document Domain (e.g., "Financial Report", "Technical Manual")
     - FAISS Vector Indexes

4. **Retrieval Engine (`services/retrieval.py`, `services/vectorstore.py`)**
   - **FAISS (Facebook AI Similarity Search):** Splits loaded documents into chunks, calculates dense vector embeddings using OpenAI, and indexes them for high-speed similarity search.
   - **BM25 Retriever:** Creates a sparse representation of chunks for exact-keyword frequency matching.
   - **Ensemble Retriever:** Merges the results of both FAISS (weighted 0.6) and BM25 (weighted 0.4) to return the most contextually relevant and exact chunks.

5. **LLM Orchestration Layer (`services/llm.py`, `services/nexus_challenge.py`)**
   - Interacts directly with OpenAI (`gpt-4o-mini`) and Google Gemini.
   - Includes custom conversational loops (the Universal LLM pipeline) that dynamically decide whether to extract data, calculate mathematics, or make external API calls based on the document's context.

---

## 🔄 Request Lifecycle (RAG Pipeline)

When a user submits a POST request to `/nexus/run` with a document and a batch of questions, the system follows this execution graph:

1. **Ingestion & Hashing:** 
   The application intercepts the document URL, downloads it natively, and computes its SHA-256 hash.

2. **Cache Verification:**
   The `faiss_indexes` directory is checked against the hash. 
   - *If Match Found:* Immediately load the extracted text, FAISS index, and Document Domain directly from Disk into RAM.
   - *If No Match:* Parse the downloaded document, extract all text, split into embeddings, write the FAISS index to disk, detect the domain via LLM, and cache all artifacts. 

3. **Question Triage:**
   Since a user can pass an array of `$N` questions, the system evaluates all questions in parallel to determine `is_general_knowledge`. 
   - Questions requiring context are queued for Vector Retrieval.
   - General knowledge questions bypass the Document Vector step.

4. **Parallel Retrieval:**
   Using the `EnsembleRetriever`, the system synchronously gathers the top 5 most relevant document chunks for each context-requiring question.

5. **Concurrent Inference:**
   All questions (both Document-based and General Knowledge) are fired asynchronously to the LLM within a constrained `Semaphore` pool (to gracefully throttle API rates).

6. **Aggregation:**
   The answers are aggregated in the exact original order requested by the client, and the response is serialized and returned.

---

## ⚡ Concurrency & Optimization

- **Thread Pooling:** A global `ThreadPoolExecutor` is initialized at startup. It is used to run CPU-bound Langchain operations (like FAISS indexing and PDF text extraction) outside the main Python `asyncio` event loop.
- **Asynchronous I/O:** All HTTP requests, disk I/O bindings, and LLM inferences use `async/await` to ensure non-blocking networking.
- **Memory Footprint:** By caching extracted texts to disk rather than keeping historical context maps in RAM, the API process remains lightweight. 

---

## 📂 Project Structure Map

```text
nexus/
├── app.py                      # Application Entry Point (Uvicorn wrapper)
├── requirements.txt            # Python dependencies
├── .env                        # Environment Secrets (ignored in git)
├── faiss_indexes/              # Persistent Volume for Cached Vector stores
└── nexus_app/
    ├── __init__.py
    ├── app_factory.py          # FastAPI application builder & lifecycle
    ├── core/
    │   ├── logging_config.py   # Standardized Emoji-Logger
    │   ├── middleware.py       # Request Latency & Trace middleware
    │   └── settings.py         # Type-safe Environment configuration
    ├── models/
    │   └── schemas.py          # Pydantic Request/Response models
    ├── routes/
    │   └── api.py              # Primary Endpoints (/nexus/run, /health)
    ├── services/
    │   ├── cache.py            # Local FileSystem Caching logic
    │   ├── document_io.py      # Downloader & Hash computer
    │   ├── nexus_challenge.py # Universal RAG/CAG multi-step agent logic
    │   ├── nexus_challenge_agent.py
    │   ├── llm.py              # Direct API interactions to OpenAI/Gemini
    │   ├── processing.py       # Threaded document loaders chunking
    │   ├── retrieval.py        # Ensemble, FAISS, and BM25 retrievers
    │   └── vectorstore.py      # Vector Mathematics boundary
    └── utils/
        └── constants.py        # Global limits (e.g., MAX_WORKERS, Concurrency)
```

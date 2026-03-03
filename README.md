# NexusRAG

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white)

An advanced Document Question & Answering API built for extreme scale, accuracy, and speed. Originally developed as a Nexus submission, this project leverages a high-performance Retrieval-Augmented Generation (RAG) and Context-Augmented Generation (CAG) architecture to extract intelligence from massive enterprise files.

---

## 🎯 The Problem

Large Language Models (LLMs) are revolutionary, but they suffer from severe context limitations. When presented with massive, multi-format organizational documents (like 500-page PDFs, detailed spreadsheets, or slide decks), standard AI struggles to ingest, process, and accurately reason over the data without hallucinating or running out of memory. 

Furthermore, traditional RAG systems often fail at complex reasoning tasks that require multi-step problem solving, external data lookups, and specialized calculations.

## 💡 How NexusRAG solves it

NexusRAG overcomes these limitations by implementing a **Universal LLM Pipeline** that acts as an intelligent orchestrator:
1. **Intelligent Ingestion:** It seamlessly downloads, hashes, and parses completely structured and unstructured data up to 1GB in size.
2. **Hybrid Retrieval (Ensemble):** Instead of relying purely on semantic vector search (which can miss specific keywords), it combines **FAISS** (dense vector search) with **BM25** (sparse keyword search). 
3. **Smart Workflows:** For complex problems, the engine breaks down the document's domain into a dynamic, multi-step execution graph (extract -> calculate -> API lookup) guided by interactive feedback loops.
4. **Aggressive Caching:** We heavily cache extracted texts, vector indexes, and domain classifications locally based on document hashes. If you ask 10 new questions about a previously uploaded document, the system skips ingestion entirely and answers instantly.

## ✨ Key Features

- **Comprehensive Format Support:** Natively processes `PDF`, `DOCX`, `TXT`, `PPT`, `PPTX`, `XLSX`, `Images`, `ZIP`, and `BIN` payloads.
- **Ensemble Retrieval System:** 60% Semantic Search (FAISS) + 40% Keyword Match (BM25) guarantees both conceptual understanding and precise data retrieval.
- **Massive Concurrency:** Built on FastAPI with `asyncio` and `ThreadPoolExecutor`. If an end-user submits 20 questions simultaneously, the LLM processes them in parallel dynamically.
- **General Knowledge Bypassing:** The system auto-detects if a question requires document context or is general knowledge, bypassing expensive retrieval steps to save time and compute.
- **Graceful File Handling:** Strict 1GB guardrails with non-crashing fallback responses for oversized files.
- **Universal Challenge Agent:** Includes a specialized Step-Function-like CAG prompt loop designed to solve deeply complex logic puzzles over documents.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- OpenAI API Key
- Google Gemini API Key

### Installation

1. Clone the repository and navigate into the project directory:
   ```bash
   cd nexus
   ```

2. Activate your virtual environment (or create one):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory and add your API credentials:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

### Running the Application

Start the FastAPI server using Uvicorn:
```bash
python app.py
```
*(The server will start on `http://0.0.0.0:8000`)*

### API Documentation

Once running, navigate to the auto-generated Swagger UI to test the endpoints:
- **Swagger UI:** `http://localhost:8000/docs`
- **Redoc:** `http://localhost:8000/redoc`

---

## 🛣️ Core Endpoint

### `POST /nexus/run`
Submit a document URL and a list of questions to receive parallelized answers.

**Request Body:**
```json
{
  "documents": "https://example.com/massive-report.pdf",
  "questions": [
    "What is the Q3 revenue target?",
    "Summarize the executive roadmap in 3 bullets.",
    "What is the capital of France?" 
  ]
}
```
*(Notice how the 3rd question is general knowledge; the API will answer it immediately without scanning the PDF!)*

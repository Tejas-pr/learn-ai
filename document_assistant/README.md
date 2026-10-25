# 📑 Document Assistant API

> Part of the [learn-ai](https://github.com/Tejas-pr/learn-ai) repository.

A fast, lightweight, and local **FastAPI** service for document ingestion, text chunking, persistent vector embedding with **ChromaDB**, and semantic search powered by **Ollama**.

---

## 🚀 Features

- **Chunking Pipeline**: Splits long text documents into overlapping character chunks to avoid slicing context across boundaries.
- **Local Embeddings**: Generates 768-dimensional vector embeddings using Ollama's `nomic-embed-text`.
- **Persistent Vector Store**: Stores chunks and embeddings to disk in ChromaDB (`./chroma_db`).
- **Upsert / Deduplication**: Re-ingesting a document automatically clears previous chunks for that `document_id`.
- **Semantic Search**: Computes query embeddings and performs nearest-neighbor search with cosine distance.
- **AI Chat Endpoint**: Direct AI conversation powered by `qwen2.5-coder:7b`.
- **Interactive Documentation**: Auto-generated Swagger UI at `/docs`.

---

## 🛠️ Tech Stack

- **Framework**: [FastAPI](https://fastapi.tiangolo.com/) + [Uvicorn](https://www.uvicorn.org/)
- **Validation**: [Pydantic v2](https://docs.pydantic.dev/)
- **LLM & Embeddings Provider**: [Ollama](https://ollama.ai/)
  - Embeddings: `nomic-embed-text:latest`
  - Chat: `qwen2.5-coder:7b`
- **Vector Database**: [ChromaDB](https://www.trychroma.com/) (Persistent SQLite)

---

## 📋 Prerequisites

1. Install and run [Ollama](https://ollama.ai/).
2. Pull required models:
   ```bash
   ollama pull nomic-embed-text
   ollama pull qwen2.5-coder:7b
   ```

---

## 💻 Setup & Installation

1. Navigate to the project directory:
   ```bash
   cd document_assistant
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install fastapi uvicorn chromadb ollama pydantic
   ```

4. Start the FastAPI development server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

5. Open your browser to test endpoints interactively:
   - **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
   - **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

---

## 📡 API Endpoints

### 1. Ingest Document
Split text into chunks, generate embeddings, and store them in ChromaDB.

- **Method**: `POST`
- **Endpoint**: `/ingest`
- **Payload**:
  ```json
  {
    "document_id": "doc_101",
    "text": "Antigravity is an advanced agentic coding assistant developed by Google DeepMind..."
  }
  ```
- **Response**:
  ```json
  {
    "document_id": "doc_101",
    "chunks_created": 1
  }
  ```

---

### 2. Semantic Search
Search through stored documents using vector similarity.

- **Method**: `POST`
- **Endpoint**: `/search`
- **Payload**:
  ```json
  {
    "question": "What is Antigravity?",
    "top_k": 3
  }
  ```
- **Response**:
  ```json
  {
    "question": "What is Antigravity?",
    "results": [
      {
        "text": "Antigravity is an advanced agentic coding assistant...",
        "metadata": {
          "document_id": "doc_101"
        },
        "distance": 0.28
      }
    ]
  }
  ```

---

### 3. AI Chat
Ask general questions directly to the local LLM.

- **Method**: `POST`
- **Endpoint**: `/chat`
- **Payload**:
  ```json
  {
    "question": "Explain vector databases in one sentence."
  }
  ```
- **Response**:
  ```json
  {
    "question": "Explain vector databases in one sentence.",
    "answer": "A vector database is an indexing system optimized to store, index, and retrieve high-dimensional mathematical embeddings based on semantic similarity."
  }
  ```

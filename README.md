# 🧠 Learn AI

Welcome to **Learn AI**! This repository is dedicated to learning, experimenting with, and building hands-on Artificial Intelligence (AI) and Large Language Model (LLM) applications.

The projects here focus on local, private, and open-source AI workflows—leveraging local models via Ollama, vector databases (ChromaDB), Retrieval-Augmented Generation (RAG), and modern backend/frontend frameworks.

---

## 📦 Cloning the Repository

Because this repository uses **Git submodules** to manage standalone projects, clone with the `--recurse-submodules` flag to ensure all nested project code is downloaded:

```bash
git clone --recurse-submodules https://github.com/Tejas-pr/learn-ai.git
cd learn-ai
```

### If you already cloned without `--recurse-submodules`:
Initialize and pull the submodules with:
```bash
git submodule update --init --recursive
```

To pull the latest updates across the repository and all submodules:
```bash
git pull --recurse-submodules
```

---

## 📂 Projects Overview

| Project | Description | Stack | Submodule Repo |
| :--- | :--- | :--- | :--- |
| **[Chat with PDF](./chat_with_pdf/)** | 100% offline RAG app to query and converse with PDF documents in real-time. | Streamlit, Ollama, ChromaDB, pypdf | [chat_with_pdf](https://github.com/Tejas-pr/chat_with_pdf) |
| **[Document Assistant](./document_assistant/)** | Production-ready FastAPI service for document chunking, vector embedding, semantic retrieval, and AI chat. | FastAPI, Uvicorn, Ollama, ChromaDB, Pydantic | Standalone folder |

---

## 🛠️ Global Prerequisites

Most projects in this repository run models locally:

1. **Python 3.9+**
2. **[Ollama](https://ollama.ai/)** installed and running on your system.
3. Download required local models:
   ```bash
   # Embeddings model
   ollama pull nomic-embed-text

   # Generative models
   ollama pull llama3.2:latest
   ollama pull qwen2.5-coder:7b
   ```

---

## 📑 Project Details

### 1. [Chat with PDF](./chat_with_pdf/)
- **Goal**: Full RAG pipeline with interactive web UI and conversational memory.
- **Features**: PDF parsing, sliding window chunking with overlap, semantic search via ChromaDB vector persistence, live token streaming.
- **Quickstart**:
  ```bash
  cd chat_with_pdf
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt # or: pip install pypdf chromadb ollama streamlit
  streamlit run app.py
  ```

### 2. [Document Assistant](./document_assistant/)
- **Goal**: Clean REST API exposing document ingestion, chunking, vector storage, and semantic search endpoints.
- **Features**: Document deduplication/re-indexing, Pydantic input validation, cosine distance scoring, modular RAG backend.
- **Quickstart**:
  ```bash
  cd document_assistant
  python3 -m venv venv
  source venv/bin/activate
  pip install fastapi uvicorn chromadb ollama pydantic
  uvicorn main:app --reload
  ```
  Explore the interactive Swagger API documentation at: [http://localhost:8000/docs](http://localhost:8000/docs)

---

## 🎯 Key AI Concepts Explored

- **Retrieval-Augmented Generation (RAG)**: Overcoming LLM hallucinations by grounding queries in private document context.
- **Vector Embeddings**: Converting textual semantics into multi-dimensional vector spaces using `nomic-embed-text`.
- **Chunking Strategies**: Exploring fixed vs overlapping sliding window chunking to preserve phrase continuity.
- **Vector Databases**: Managing persistent collections, embeddings storage, and similarity search with ChromaDB.
- **Local Model Serving**: Running open-weights models (`llama3.2`, `qwen2.5-coder`) locally via Ollama with zero external API costs or data leakage.

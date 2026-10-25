from fastapi import FastAPI, HTTPException
import ollama
from pydantic import BaseModel, Field
import chromadb

app = FastAPI(title="Document Assistant", description="A FastAPI application for document assistance.", version="1.0.0")

ollama_client = ollama.Client(
    host="http://localhost:11434",
)

db = chromadb.PersistentClient(path="./chroma_db")

collection = db.get_or_create_collection(
    name="documents"
)

# ----------------- pydantic models for request validation

class ChatResponse(BaseModel):
    question: str = Field(min_length=1)

class IngestRequest(BaseModel):
    document_id: str = Field(min_length=1)
    text: str = Field(min_length=1)

class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)

# ----------------- Helper function.

# Chunk text into smaller pieces for processing
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(text):
            break

        start = end - overlap

    return chunks

# Generate embeddings locally
def embed_texts(texts: list[str]) -> list[list[float]]:
    response = ollama_client.embed(
        model="nomic-embed-text:latest",
        input=texts,
    )
    return response["embeddings"]

# ----------------- FastAPI endpoints

@app.get("/")
def home():
    return {"message": "Welcome to the Document Assistant API!"}

@app.post("/chat")
def chat(request: ChatResponse):
    try:
        response = ollama_client.chat(
            model="qwen2.5-coder:7b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful AI assistant. "
                        "Explain concepts clearly."
                    ),
                },
                {
                    "role": "user",
                    "content": request.question,
                },
            ]
        )
        return {
            "question": request.question,
            "answer": response["message"]["content"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama request failed: {str(e)}",
        )

@app.post("/ingest")
def ingest(request: IngestRequest):
    chunks = chunk_text(request.text)

    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="No usable text found",
        )

    try:
        embeddings = embed_texts(chunks)

        # Remove old chunks if this document is re-ingested.
        collection.delete(
            where={"document_id": request.document_id}
        )

        ids = [f"{request.document_id}_{i}" for i in range(len(chunks))]

        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=[{"document_id": request.document_id} for _ in chunks]
        )

        return {
            "document_id": request.document_id,
            "chunks_created": len(chunks),
        }
    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Failed to ingest document: {str(e)}",
        )

# Retrieve relevant document chunks
@app.post("/search")
def search(request: ChatRequest):
    if collection.count() == 0:
        raise HTTPException(
            status_code=404,
            detail="No documents found in the database.",
        )

    try:
        query_embedding = embed_texts([request.question])[0]

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(3, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        matches = []

        for i, document in enumerate(results["documents"][0]):
            matches.append({
                "text": document,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })

        return {
            "question": request.question,
            "results": matches,
        }

    except Exception as e:
        raise HTTPException(
            status_code=502,
            detail=f"Search failed: {e}",
        )

from fastapi import FastAPI, Request
from pydantic import BaseModel
import chromadb
import os
import requests
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = client.get_or_create_collection("nitsri")

app = FastAPI()

class QueryRequest(BaseModel):
    query: str

def get_embedding(text: str):
    """Embed the query using OpenAI embeddings."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",
        "input": text
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    result = response.json()
    return result["data"][0]["embedding"]

def query_chroma_db(query: str, top_k: int = 5):
    """Query ChromaDB for similar chunks."""
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )
    return results

def ask_openai(question: str, context: str):
    """Ask OpenAI GPT model using retrieved context."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    messages = [
        {"role": "system", "content": "You are a helpful assistant answering questions about NIT Srinagar."},
        {"role": "user", "content": f"Use the following context to answer:\n\n{context}\n\nQuestion: {question}"}
    ]
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.4
    }
    response = requests.post(url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.post("/query")
async def query_endpoint(req: QueryRequest):
    try:
        # 1. Search ChromaDB
        results = query_chroma_db(req.query, top_k=5)
        context_chunks = results["documents"][0] if results["documents"] else []

        # 2. Concatenate retrieved chunks
        context_text = "\n---\n".join(context_chunks)

        # 3. Ask OpenAI
        answer = ask_openai(req.query, context_text)

        return {"answer": answer, "source_urls": [m["url"] for m in results["metadatas"][0]]}
    
    except Exception as e:
        return {"error": str(e)}

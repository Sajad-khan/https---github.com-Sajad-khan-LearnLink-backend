import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import logging
import traceback

# Load environment variables
load_dotenv()

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI and ChromaDB
app = FastAPI(title="NIT Srinagar Knowledge Base")
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection("nit_knowledge")
openai_client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    sources: Optional[List[str]] = None  # ["scraped", "user"], or None for both

class SourceResult(BaseModel):
    source: str
    metadata: dict
    relevance_score: float

class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceResult]
    confidence: float

def generate_answer(query: str, relevant_contexts: List[str]) -> str:
    try:
        if not relevant_contexts:
            return "I couldn't find specific information to answer your query."

        prompt = f"""
Given the following context, answer the query as precisely as possible:

Context: {' '.join(relevant_contexts)}

Query: {query}

Answer:
"""
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for NIT Srinagar information."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content or "No specific information found."
    except Exception as e:
        logger.error(f"Answer generation error: {e}\n{traceback.format_exc()}")
        return f"Error generating answer: {str(e)}"

@app.post("/query", response_model=QueryResponse)
async def semantic_search(request: QueryRequest):
    try:
        # Log collection count
        doc_count = collection.count()
        logger.info(f"Collection contains {doc_count} documents")
        
        if doc_count == 0:
            logger.warning("Collection is empty! No documents to search.")
            return QueryResponse(
                answer="The knowledge base is empty. Please add documents first.",
                sources=[],
                confidence=0.0
            )
            
        # Generate query embedding
        query_embedding = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=[request.query]
        ).data[0].embedding

        # Prepare query for ChromaDB
        query_params = {
            "query_embeddings": [query_embedding],
            "n_results": min(request.top_k, doc_count)  # Don't request more than we have
        }

        # Log search parameters
        if request.sources:
            query_params["where"] = {"source": {"$in": request.sources}}
            logger.info(f"Filtering by sources: {request.sources}")
        else:
            # Don't add any where filter - this will search all documents
            logger.info("No source filter applied, searching all documents")

        # Query ChromaDB
        logger.info(f"Executing query: '{request.query}' with top_k={query_params['n_results']}")
        results = collection.query(**query_params)

        # Log results count
        ids = results.get("ids", [[]])
        result_count = len(ids[0]) if ids and ids[0] else 0
        logger.info(f"Query returned {result_count} results")
        
        if result_count == 0:
            return QueryResponse(
                answer="No relevant information found in the knowledge base for your query.",
                sources=[],
                confidence=0.0
            )

        sources = []
        contexts = []
        metadatas = results.get("metadatas", [[]])[0]
        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        # Log the first result for debugging
        if result_count > 0:
            logger.info(f"First result source: {metadatas[0].get('source', 'unknown')}")
            logger.info(f"First result distance: {distances[0]}")

        for i in range(result_count):
            source_result = SourceResult(
                source=metadatas[i].get("source", "unknown"),
                metadata=metadatas[i],
                relevance_score=1.0 - distances[i] if distances else 0.0  # Convert distance to relevance
            )
            sources.append(source_result)
            contexts.append(docs[i])

        # Generate answer and calculate confidence
        answer = generate_answer(request.query, contexts)
        
        # Calculate confidence (inverted distance)
        mean_distance = sum(distances) / len(distances) if distances else 1.0
        confidence = 1.0 - mean_distance  # Convert distance to confidence
        
        logger.info(f"Generated answer with confidence: {confidence:.4f}")

        return QueryResponse(
            answer=answer,
            sources=sources,
            confidence=confidence
        )
    except Exception as e:
        logger.error(f"Query processing error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/sources")
async def get_available_sources():
    """
    Utility endpoint: List which 'source' values are present in the collection.
    """
    try:
        # Get all metadatas, extract sources (first 1000 documents)
        metadatas = []
        num_results = 100
        offset = 0
        while True:
            results = collection.get(
                include=["metadatas"],
                limit=num_results,
                offset=offset
            )
            if not results["metadatas"]:
                break
            metadatas.extend(results["metadatas"])
            if len(results["metadatas"]) < num_results:
                break
            offset += num_results
        
        # Count documents by source
        source_counts = {}
        for m in metadatas:
            if m and "source" in m:
                source = m["source"]
                source_counts[source] = source_counts.get(source, 0) + 1
                
        return {
            "sources": list(source_counts.keys()),
            "counts": source_counts,
            "total_documents": len(metadatas)
        }
    except Exception as e:
        logger.error(f"Error getting sources: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/collection-stats")
async def get_collection_stats():
    """
    Utility endpoint: Show stats about the collection
    """
    try:
        count = collection.count()
        
        # Get sample documents to verify content
        sample = collection.get(
            limit=5,
            include=["metadatas", "documents"]
        )
        
        sources_count = {}
        if sample["metadatas"]:
            for metadata in sample["metadatas"]:
                source = metadata.get("source", "unknown")
                sources_count[source] = sources_count.get(source, 0) + 1
        
        return {
            "total_documents": count,
            "sample_sources": sources_count,
            "has_documents": count > 0,
            "sample_document_ids": sample.get("ids", [])[:5]
        }
    except Exception as e:
        logger.error(f"Error getting collection stats: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
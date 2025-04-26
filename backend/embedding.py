import os
import json
import requests
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Updated ChromaDB path for Windows
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
OPENAI_API_KEY = os.getenv("OPEN_AI_API_KEY")  # Note: Getting from OPEN_AI_API_KEY as specified
SCRAPED_FILE = "../crawler/data/scraped_data.json"
MAX_CHUNK_SIZE = 8000  # Max tokens for OpenAI embeddings (text-embedding-3-small supports up to 8K tokens)
CHUNK_OVERLAP = 200    # Overlap between chunks to maintain context

# Validate API key
if not OPENAI_API_KEY:
    raise ValueError("OPEN_AI_API_KEY not found in environment variables")

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

# Get or create collection
collection = client.get_or_create_collection("nitsri")

def chunk_text(text, max_size=MAX_CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Split text into chunks of max_size with overlap."""
    words = text.split()
    
    # If text is small enough, return as single chunk
    if len(words) <= max_size:
        return [text]
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(words):
        # Calculate end index for this chunk
        end_idx = min(start_idx + max_size, len(words))
        
        # Create chunk from words
        chunk = " ".join(words[start_idx:end_idx])
        chunks.append(chunk)
        
        # Move start index forward, accounting for overlap
        start_idx = end_idx - overlap if end_idx < len(words) else end_idx
    
    return chunks

def get_embedding(text):
    """Get embedding for text using OpenAI API."""
    url = "https://api.openai.com/v1/embeddings"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-3-small",  # Most cost-effective embedding model
        "input": text
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()  # Raise exception for HTTP errors
        result = response.json()
        
        if "data" in result and result["data"]:
            return result["data"][0]["embedding"]
        else:
            print(f"❌ Failed to embed: {result}")
            return None
    except requests.exceptions.RequestException as e:
        print(f"❌ API request failed: {e}")
        return None
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return None

def process_data():
    """Process scraped data, chunk it, and store in ChromaDB."""
    # Track stats
    processed_docs = 0
    processed_chunks = 0
    skipped = 0
    failed = 0
    
    try:
        # Update path to be Windows-compatible and check if file exists
        adjusted_path = os.path.normpath(os.path.join(os.path.dirname(__file__), SCRAPED_FILE))
        if not os.path.exists(adjusted_path):
            print(f"❌ File not found: {adjusted_path}")
            return
            
        print(f"📂 Processing file: {adjusted_path}")
        
        with open(adjusted_path, "r", encoding="utf-8") as f:
            # Get existing IDs to avoid duplicates
            try:
                existing_ids = set(collection.get()["ids"]) if collection.count() > 0 else set()
                print(f"📊 Found {len(existing_ids)} existing documents in the collection")
            except Exception as e:
                print(f"⚠️ Could not retrieve existing IDs: {e}")
                existing_ids = set()
            
            # Count total lines for progress reporting
            total_lines = sum(1 for _ in open(adjusted_path, "r", encoding="utf-8"))
            print(f"📄 Total documents to process: {total_lines}")
            
            for idx, line in enumerate(f):
                doc_base_id = f"doc_{idx}"
                
                # Print progress every 10 documents
                if idx % 10 == 0:
                    print(f"⏳ Processing document {idx+1}/{total_lines}...")
                
                try:
                    item = json.loads(line)
                    content = item.get("content", "").strip()
                    url = item.get("url", "")
                    
                    # Skip if content is too short
                    if len(content.split()) < 20:
                        skipped += 1
                        continue
                    
                    # Create chunks from content
                    chunks = chunk_text(content)
                    chunk_success = 0
                    
                    # Process each chunk
                    for chunk_idx, chunk in enumerate(chunks):
                        # Generate a unique ID for this chunk
                        chunk_id = f"{doc_base_id}_chunk_{chunk_idx}"
                        
                        # Skip if already processed
                        if chunk_id in existing_ids:
                            skipped += 1
                            continue
                        
                        # Get embedding for chunk
                        embedding = get_embedding(chunk)
                        if not embedding:
                            failed += 1
                            continue
                        
                        # Add to ChromaDB with the embedding
                        collection.add(
                            documents=[chunk],
                            embeddings=[embedding],
                            metadatas=[{
                                "url": url,
                                "doc_id": doc_base_id,
                                "chunk_idx": chunk_idx,
                                "total_chunks": len(chunks)
                            }],
                            ids=[chunk_id]
                        )
                        
                        chunk_success += 1
                        processed_chunks += 1
                    
                    if chunk_success > 0:
                        processed_docs += 1
                        print(f"✅ Embedded doc_{idx}: {url} ({chunk_success}/{len(chunks)} chunks)")
                    else:
                        print(f"❌ Failed to embed any chunks for doc_{idx}: {url}")
                        
                except json.JSONDecodeError:
                    print(f"❌ Invalid JSON at line {idx+1}")
                    failed += 1
                except Exception as e:
                    print(f"❌ Failed to process line {idx+1}: {e}")
                    failed += 1
    except Exception as e:
        print(f"❌ Error processing file: {e}")
    
    print(f"📊 Summary: {processed_docs} documents processed ({processed_chunks} chunks), {skipped} skipped, {failed} failed")

def query_similar(query_text, n_results=5):
    """Query for similar documents based on a text query."""
    # Get embedding for query
    query_embedding = get_embedding(query_text)
    if not query_embedding:
        print("❌ Failed to get embedding for query")
        return []
    
    # Query the collection
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    
    return results

if __name__ == "__main__":
    process_data()
    
    # Uncomment to test querying
    # test_query = "What is NITSRI?"
    # results = query_similar(test_query)
    # print(f"Query results for '{test_query}':")
    # for i, (doc, metadata, distance) in enumerate(zip(
    #     results["documents"][0], 
    #     results["metadatas"][0],
    #     results["distances"][0]
    # )):
    #     print(f"\n--- Result {i+1} (Distance: {distance:.4f}) ---")
    #     print(f"URL: {metadata['url']}")
    #     print(f"Document ID: {metadata['doc_id']}, Chunk {metadata['chunk_idx']+1}/{metadata['total_chunks']}")
    #     print(f"Preview: {doc[:200]}...")
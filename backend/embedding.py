import os
import json
import logging
from typing import List, Dict, Any
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv
import time
from tqdm import tqdm
import colorama
from termcolor import colored
import tiktoken  # For accurate token counting

# Initialize colorama
colorama.init()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format=colored('%(asctime)s', 'cyan') + ' - ' + 
           colored('%(levelname)s', 'green') + ' - ' + 
           colored('%(message)s', 'white')
)
logger = logging.getLogger(__name__)

class EmbeddingService:
    def __init__(self, collection_name='nit_knowledge'):
        # Start timing the initialization
        start_time = time.time()
        
        # Determine the absolute path of the script
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        
        # Initialize tokenizer
        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        
        logger.info(colored("🚀 Initializing Embedding Service", "blue"))
        logger.info(colored(f"📍 Base Directory: {self.BASE_DIR}", "cyan"))
        
        # Initialize ChromaDB client
        try:
            db_path = os.path.join(self.BASE_DIR, "chroma_db")
            logger.info(colored(f"📂 ChromaDB Path: {db_path}", "cyan"))
            
            # Ensure chroma_db directory exists
            os.makedirs(db_path, exist_ok=True)
            
            self.chroma_client = chromadb.PersistentClient(path=db_path)
            logger.info(colored("✅ ChromaDB Client Initialized", "green"))
        except Exception as e:
            logger.error(colored(f"❌ ChromaDB Initialization Failed: {e}", "red"))
            raise
        
        # Initialize OpenAI client
        try:
            api_key = os.getenv('OPEN_AI_API_KEY')
            if not api_key:
                logger.error(colored("❌ OpenAI API key not found in environment variables", "red"))
                raise ValueError("OpenAI API key not found")
                
            self.openai_client = OpenAI(api_key=api_key)
            logger.info(colored("✅ OpenAI Client Initialized", "green"))
        except Exception as e:
            logger.error(colored(f"❌ OpenAI Client Initialization Failed: {e}", "red"))
            raise
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Log collection count
        doc_count = self.collection.count()
        logger.info(colored(f"📂 Collection '{collection_name}' Ready with {doc_count} documents", "magenta"))
        
        # Log initialization time
        init_time = time.time() - start_time
        logger.info(colored(f"⏱️  Initialization completed in {init_time:.2f} seconds", "yellow"))

    def chunk_text(self, text: str, max_tokens: int = 3000, overlap: int = 500) -> List[str]:
        """
        Chunk text based on token count with overlap.
        Chunks will be at most 3000 tokens with 500 token overlap by default.
        """
        if not text or text.strip() == "":
            logger.warning(colored("⚠️ Empty text provided for chunking", "yellow"))
            return []
            
        # Tokenize the entire text
        tokens = self.tokenizer.encode(text)
        logger.info(colored(f"📏 Text length: {len(tokens)} tokens", "cyan"))
        
        chunks = []
        start = 0
        
        while start < len(tokens):
            # Select a chunk of tokens
            chunk_tokens = tokens[start:start + max_tokens]
            
            # Decode the chunk
            chunk = self.tokenizer.decode(chunk_tokens)
            chunks.append(chunk)
            
            # Move start position with overlap
            start += max_tokens - overlap
        
        logger.info(colored(f"✂️  Created {len(chunks)} chunks (max {max_tokens} tokens)", "green"))
        return chunks

    def generate_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for given texts using OpenAI
        """
        if not texts:
            logger.warning(colored("⚠️ No texts provided for embedding generation", "yellow"))
            return []
            
        try:
            embeddings = []
            for text in tqdm(texts, desc=colored("Generating Embeddings", "blue")):
                response = self.openai_client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=[text]
                )
                embeddings.append(response.data[0].embedding)
            
            logger.info(colored(f"✅ Generated {len(embeddings)} embeddings", "green"))
            return embeddings
        
        except Exception as e:
            logger.error(colored(f"❌ Embedding Generation Error: {e}", "red"))
            raise

    def embed_scraped_data(self, relative_path: str):
        """
        Embed scraped web data by chunking large files
        """
        start_time = time.time()
        
        # Construct absolute path
        scraped_data_path = os.path.normpath(os.path.join(self.BASE_DIR, relative_path))
        
        logger.info(colored(f"🌐 Starting embedding of scraped data", "blue"))
        logger.info(colored(f"📁 Scraped Data Path: {scraped_data_path}", "cyan"))

        try:
            # Validate file existence
            if not os.path.exists(scraped_data_path):
                logger.error(colored(f"❌ File not found: {scraped_data_path}", "red"))
                raise FileNotFoundError(f"Scraped data file not found: {scraped_data_path}")

            # Log file details
            file_size = os.path.getsize(scraped_data_path)
            logger.info(colored(f"📊 File Size: {file_size / 1024:.2f} KB", "magenta"))

            # Read the entire file content
            with open(scraped_data_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
                # Check if the file is JSON
                if scraped_data_path.lower().endswith('.json'):
                    try:
                        # Parse JSON file
                        json_data = json.loads(file_content)
                        
                        # Handle different JSON structures
                        if isinstance(json_data, list):
                            # JSON array - process each item
                            logger.info(colored(f"📊 JSON array with {len(json_data)} items", "cyan")) 
                            combined_text = ""
                            for item in json_data:
                                if isinstance(item, dict):
                                    # Convert dict values to string and join
                                    item_text = " ".join(str(v) for v in item.values() if v)
                                    combined_text += item_text + "\n\n"
                                else:
                                    # Just convert to string
                                    combined_text += str(item) + "\n\n"
                            file_content = combined_text
                        elif isinstance(json_data, dict):
                            # Single JSON object - extract values
                            logger.info(colored("📊 Single JSON object", "cyan"))
                            file_content = " ".join(str(v) for v in json_data.values() if v)
                    except json.JSONDecodeError:
                        # Not valid JSON, continue with raw content
                        logger.warning(colored("⚠️ File has .json extension but content is not valid JSON", "yellow"))
                        pass

            # Chunk the content (now max 3000 tokens per chunk)
            content_chunks = self.chunk_text(file_content)
            logger.info(colored(f"📦 Created {len(content_chunks)} chunks", "cyan"))
            
            if not content_chunks:
                logger.warning(colored("⚠️ No chunks were created. File might be empty.", "yellow"))
                return

            # Generate embeddings for chunks
            embeddings = self.generate_embeddings(content_chunks)
            
            if not embeddings:
                logger.warning(colored("⚠️ No embeddings were generated.", "yellow"))
                return

            # Add chunks to ChromaDB
            for idx, (chunk, embedding) in enumerate(zip(content_chunks, embeddings)):
                # Calculate token count for logging
                token_count = len(self.tokenizer.encode(chunk))
                
                metadata = {
                    'source': 'scraped',
                    'original_source': os.path.basename(scraped_data_path),
                    'chunk_number': idx,
                    'token_count': token_count
                }

                self.collection.add(
                    embeddings=[embedding],
                    documents=[chunk],
                    ids=[f"scraped_{os.path.basename(scraped_data_path)}_{idx}"],
                    metadatas=[metadata]
                )
                logger.info(colored(f"✅ Added chunk {idx+1}/{len(content_chunks)} to ChromaDB", "green"))

            # Get updated document count
            doc_count = self.collection.count()
            
            # Compute and log processing time
            processing_time = time.time() - start_time
            logger.info(colored(f"✅ Successfully embedded {len(content_chunks)} chunks", "green"))
            logger.info(colored(f"📊 Collection now has {doc_count} total documents", "magenta"))
            logger.info(colored(f"⏱️  Total processing time: {processing_time:.2f} seconds", "yellow"))

        except Exception as e:
            logger.error(colored(f"❌ Embedding Process Failed: {e}", "red"))
            raise

# Executable script section
if __name__ == "__main__":
    try:
        # Use relative path to scraped data
        SCRAPED_FILE = "../crawler/data/scraped_data.json"
        
        embedding_service = EmbeddingService()
        embedding_service.embed_scraped_data(SCRAPED_FILE)
    except Exception as e:
        logger.error(colored(f"Fatal Error: {e}", "red"))
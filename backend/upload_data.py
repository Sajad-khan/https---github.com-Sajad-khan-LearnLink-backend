import os
import json
import logging
from embedding import EmbeddingService
import time
from tqdm import tqdm
import colorama
from termcolor import colored

# Initialize colorama
colorama.init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format=colored('%(asctime)s', 'cyan') + ' - ' +
           colored('%(levelname)s', 'green') + ' - ' +
           colored('%(message)s', 'white')
)
logger = logging.getLogger(__name__)

class DocumentUploader:
    def __init__(self):
        start_time = time.time()
        logger.info(colored("🚀 Initializing Document Uploader", "blue"))
        try:
            self.embedding_service = EmbeddingService()
            doc_count = self.embedding_service.collection.count()
            logger.info(colored(f"📊 Collection has {doc_count} existing documents", "cyan"))
            init_time = time.time() - start_time
            logger.info(colored(f"⏱️  Initialization completed in {init_time:.2f} seconds", "yellow"))
        except Exception as e:
            logger.error(colored(f"❌ Initialization Error: {e}", "red"))
            raise

    def upload_and_embed_document(self, file_path: str):
        """
        Upload and embed user-provided JSON data as one coherent context,
        preserving full detail and ensuring a single embedding per file.
        """
        start_time = time.time()
        logger.info(colored(f"📤 Starting upload for: {file_path}", "blue"))
        try:
            if not os.path.exists(file_path):
                logger.error(colored(f"❌ File not found: {file_path}", "red"))
                raise FileNotFoundError(f"Document file not found: {file_path}")

            ext = os.path.splitext(file_path)[1].lower()
            if ext != '.json':
                logger.error(colored(f"❌ Unsupported file type: {ext}", "red"))
                return False

            # Load JSON data
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            records = data if isinstance(data, list) else [data]
            if not records:
                logger.warning(colored("⚠️ No records to process", "yellow"))
                return False

            # Build detailed context string
            contexts = []
            for rec in records:
                if isinstance(rec, dict):
                    parts = [f"{k}: {v}" for k, v in rec.items() if v is not None]
                    if parts:
                        contexts.append("; ".join(parts))
            combined_text = "\n\n".join(contexts)
            logger.info(colored(f"📄 Combined context length: {len(combined_text)} characters", "cyan"))

            # Generate a single embedding for the full context
            # Skip chunking to maintain coherence on small datasets
            embeddings = self.embedding_service.generate_embeddings([combined_text])
            if not embeddings:
                logger.warning(colored("⚠️ Embedding failed", "yellow"))
                return False

            emb = embeddings[0]
            metadata = {
                'source': 'user',
                'filename': os.path.basename(file_path),
                'record_count': len(records)
            }
            doc_id = f"user_{os.path.basename(file_path)}"
            self.embedding_service.collection.add(
                embeddings=[emb],
                documents=[combined_text],
                ids=[doc_id],
                metadatas=[metadata]
            )
            logger.info(colored(f"✅ Added embedding for {doc_id}", "green"))

            total_docs = self.embedding_service.collection.count()
            elapsed = time.time() - start_time
            logger.info(colored(f"📊 Collection now has {total_docs} documents", "magenta"))
            logger.info(colored(f"⏱️  Processing time: {elapsed:.2f}s", "yellow"))

            return True
        except Exception as e:
            logger.error(colored(f"❌ Upload failed: {e}", "red"))
            return False

    def check_collection_status(self):
        try:
            count = self.embedding_service.collection.count()
            logger.info(colored(f"📊 Collection has {count} documents", "cyan"))
            if count > 0:
                sample = self.embedding_service.collection.get(limit=3, include=["metadatas", "documents"])
                sources = {}
                for m in sample["metadatas"]:
                    s = m.get('source', 'unknown')
                    sources[s] = sources.get(s, 0) + 1
                logger.info(colored(f"📋 Sample sources: {sources}", "cyan"))
            return {"count": count, "has_documents": count > 0}
        except Exception as e:
            logger.error(colored(f"❌ Check failed: {e}", "red"))
            return {"count": 0, "has_documents": False, "error": str(e)}

if __name__ == "__main__":
    uploader = DocumentUploader()
    status = uploader.check_collection_status()
    if not status.get('has_documents', False):
        uploader.upload_and_embed_document('data.json')
    uploader.check_collection_status()

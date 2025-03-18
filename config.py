import os
from dotenv import load_dotenv

load_dotenv()

LLM_MODE = os.getenv("LLM_MODE", "togetherai")  # Options: "ollama" or "togetherai"

# If using API-based
TOGETHERAI_API_KEY = os.getenv("TOGETHERAI_API_KEY")  # Get from https://together.ai/

# Select embedding mode: "huggingface" (local) or "togetherai" (API)
EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "togetherai")

# TogetherAI Embedding Model
TOGETHERAI_EMBEDDING_MODEL = os.getenv("TOGETHERAI_EMBEDDING_MODEL", "togethercomputer/m2-bert-80M-8k-retrieval")

# Select Similarity Mode: "cosine" or "Euclidean"
SIMILARITY_MODE = os.getenv("SIMILARITY_MODE", "euclidean")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# PDF_DIR = "data/"
PDF_DIR = os.path.join(BASE_DIR, "data")

FAISS_INDEX_PATH = os.path.join(BASE_DIR, "index", "faiss_index.bin")

CACHE_DB_PATH = os.path.join(BASE_DIR, "cache", "cache.db")

# Llama Model
LLAMA_MODEL = "llama3"  # Llama 3 via Ollama IGNORE

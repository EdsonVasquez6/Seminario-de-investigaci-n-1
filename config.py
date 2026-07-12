import os
from dotenv import load_dotenv

load_dotenv()

# API Key 
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "API KEY")

# Modelos
GENERATION_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL  = "text-embedding-3-small"

# ChromaDB 
CHROMA_PERSIST_DIR = "vectorstore/"
CHROMA_COLLECTION  = "iso27002_knowledge_base"

# Chunking
CHUNK_SIZE    = 800
CHUNK_OVERLAP = 150

# Generación 
MAX_TOKENS  = 2048
TEMPERATURE = 0.2

# Recuperación
DEFAULT_K = 4
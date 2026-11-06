# config.py
from dotenv import load_dotenv
import os

load_dotenv()

# LLM
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.1-70b-versatile"        # for answers + summaries
VISION_MODEL = "llama-3.2-90b-vision-preview" # for image descriptions

# Embeddings
EMBED_MODEL = "multi-qa-mpnet-base-cos-v1"   # from paper's appendix
EMBED_DIM = 768

# Chunking (from paper's Implementation Details)
IN_PAGE_CHUNK_SIZE = 300
IN_PAGE_OVERLAP = 50
CROSS_PAGE_CHUNK_SIZE = 100

# Retrieval (from paper's Parameter Analysis)
TOP_K_CHUNKS = 20         # chunks before parent-page expansion
TOP_K_PAGES = 10          # parent pages after LLM re-ranking
TOP_K_SUMMARIES = 10

# Paths
CHROMA_DIR = "chroma_db"
DATA_RAW = "data/raw"
DATA_PROCESSED = "data/processed"
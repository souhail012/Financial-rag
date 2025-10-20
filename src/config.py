import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PDF_FOLDER = os.path.join(ROOT, "data", "pdfs")
PROCESSED_FOLDER = os.path.join(ROOT, "data", "processed")
EMBEDDINGS_FOLDER = os.path.join(ROOT, "data", "embeddings")
QA_DATASET = os.path.join(ROOT, "data", "qa_dataset.csv")

# Chunk params (we may change later)
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Where to store FAISS index and metadata
VECTOR_DB_PATH = "data/vectorstore/financial_index"

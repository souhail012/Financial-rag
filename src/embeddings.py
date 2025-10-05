import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer
from src.config import PROCESSED_FOLDER, EMBEDDINGS_FOLDER

# Initialize embedding model
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L12-v2", device="cuda")

def generate_embeddings_and_index(
    input_csv="data/processed/clean_chunks.csv",
    emb_file="embeddings.npy",
    faiss_file="faiss_index.idx"
):
    # Load cleaned chunks
    df = pd.read_csv(input_csv)
    texts = df["text"].tolist()

    # Generate embeddings
    print("🔹 Generating embeddings...")
    embeddings = embedding_model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)  # normalize

    # Add index column
    df["embedding_index"] = np.arange(len(embeddings))

    # Save embeddings and dataframe
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
    np.save(os.path.join(EMBEDDINGS_FOLDER, emb_file), embeddings)
    df.to_csv(os.path.join(EMBEDDINGS_FOLDER, "indexed_chunks.csv"), index=False)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product for cosine similarity
    index.add(embeddings)
    faiss.write_index(index, os.path.join(EMBEDDINGS_FOLDER, faiss_file))

    print("✅ Embeddings generated and FAISS index saved.")
    return embeddings, index, df

if __name__ == "__main__":
    generate_embeddings_and_index()

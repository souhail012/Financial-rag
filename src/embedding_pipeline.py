import logging
import pandas as pd
from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from src.config import EMBEDDING_MODEL, VECTOR_DB_PATH, PROCESSED_FOLDER

# ----------------------------------------------------------------------
# 🔧 LOGGING SETUP
# ----------------------------------------------------------------------
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 🧠 EMBEDDING PIPELINE (LangChain + FAISS)
# ----------------------------------------------------------------------

def create_vector_database(
    chunks_csv: str = f"{PROCESSED_FOLDER}/clean_chunks.csv",
    model_name: str = EMBEDDING_MODEL,
    index_output: str = VECTOR_DB_PATH
):
    """Create FAISS vector database using LangChain + HuggingFace embeddings."""
    logger.info("🔹 Starting embedding process...")

    # --- Load preprocessed text chunks
    df = pd.read_csv(chunks_csv)
    if "text" not in df.columns:
        raise ValueError("CSV must contain a 'text' column.")

    texts = df["text"].tolist()
    metadatas = df.drop(columns=["text"]).to_dict(orient="records")

    # --- Create embeddings
    logger.info(f"Loading embedding model: {model_name}")
    embedding_model = HuggingFaceEmbeddings(model_name=model_name)

    logger.info("Encoding text chunks into embeddings...")
    vectorstore = FAISS.from_texts(
        texts=texts,
        embedding=embedding_model,
        metadatas=metadatas
    )

    # --- Save FAISS index + metadata (.faiss + .pkl)
    save_dir = Path(index_output).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving FAISS index to {index_output}")
    vectorstore.save_local(index_output)

    logger.info("✅ Vector database successfully created!")
    return vectorstore


# ----------------clear------------------------------------------------------
# 🚀 MAIN
# ----------------------------------------------------------------------
if __name__ == "__main__":
    create_vector_database()

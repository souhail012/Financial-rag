import os
import logging
import pandas as pd
import faiss
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document  

from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def build_langchain_retriever():
    """Load FAISS index + metadata and return a LangChain retriever."""
    faiss_path = Path(VECTOR_DB_PATH).with_suffix(".faiss")
    meta_path = Path(VECTOR_DB_PATH).with_name(f"{Path(VECTOR_DB_PATH).stem}_meta.csv")

    if not faiss_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {faiss_path}")

    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata CSV not found at {meta_path}")

    logger.info(f"Loading FAISS index from {faiss_path}")

    # Load embeddings
    embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    # Load FAISS index
    index = faiss.read_index(str(faiss_path))

    # Load metadata
    df = pd.read_csv(meta_path)
    docs = [Document(page_content=row["text"], metadata=row.to_dict()) for _, row in df.iterrows()]

    # ✅ Proper InMemoryDocstore
    docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(docs)})
    index_to_docstore_id = {i: str(i) for i in range(len(docs))}

    # Build FAISS vectorstore
    vectorstore = FAISS(
        embedding_function=embedding,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    logger.info("✅ FAISS retriever successfully loaded.")
    return retriever

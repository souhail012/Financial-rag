# src/retriever.py

import numpy as np
import pandas as pd
import faiss
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from src.config import EMBEDDINGS_FOLDER, OLLAMA_MODEL

def load_vectorstore():
    """Load embeddings, build FAISS index, and return a LangChain vectorstore."""
    print("🔹 Loading embeddings and metadata...")
    metadata = pd.read_csv(f"{EMBEDDINGS_FOLDER}/indexed_chunks.csv")
    embeddings_array = np.load(f"{EMBEDDINGS_FOLDER}/embeddings.npy")

    # Initialize embedding model
    embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")

    # Build FAISS index
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_array)

    # Create LangChain-compatible FAISS store
    docs = [
        {"page_content": row["text"], "metadata": {"filename": row["filename"], "chunk_id": row["chunk_id"]}}
        for _, row in metadata.iterrows()
    ]

    vectorstore = FAISS.from_embeddings(
        text_embeddings=list(zip([d["page_content"] for d in docs], embeddings_array)),
        embedding=embeddings_model,
        metadatas=[d["metadata"] for d in docs]
    )
    return vectorstore

def build_langchain_qa():
    """Build and return the RetrievalQA chain."""
    vectorstore = load_vectorstore()
    llm = OllamaLLM(model=OLLAMA_MODEL) 
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff"
    )
    return qa_chain

if __name__ == "__main__":
    qa_chain = build_langchain_qa()
    query = "What was the company's revenue in 2023?"
    print("🔍 Query:", query)
    answer = qa_chain.run(query)
    print("\n💬 Answer:\n", answer)

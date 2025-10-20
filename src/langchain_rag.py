import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA  # ✅ Correct for LangChain 1.0
from src.config import VECTOR_DB_PATH, EMBEDDING_MODEL, OLLAMA_MODEL

# ----------------------------------------------------------------------
# ✅ Load FAISS vectorstore created by embedding_pipeline.py
# ----------------------------------------------------------------------
print("📦 Loading FAISS vectorstore...")

embedding = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = FAISS.load_local(
    VECTOR_DB_PATH,
    embeddings=embedding,
    allow_dangerous_deserialization=True
)

print("✅ FAISS vectorstore loaded successfully!")

# ----------------------------------------------------------------------
# 🤖 Initialize LLM + RAG pipeline
# ----------------------------------------------------------------------
llm = Ollama(model=OLLAMA_MODEL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# ----------------------------------------------------------------------
# 🧠 Example query
# ----------------------------------------------------------------------
if __name__ == "__main__":
    query = "What was the company's revenue in 2023?"
    print("\nQ:", query)
    print("A:", qa_chain.run(query))

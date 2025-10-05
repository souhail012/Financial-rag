from langchain.vectorstores import FAISS
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
import faiss, numpy as np, pandas as pd
from src.config import EMBEDDINGS_FOLDER, OLLAMA_MODEL

# Load FAISS index + metadata
metadata = pd.read_csv(f"{EMBEDDINGS_FOLDER}/indexed_chunks.csv")
embeddings_array = np.load(f"{EMBEDDINGS_FOLDER}/embeddings.npy")
dim = embeddings_array.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(embeddings_array)

# LangChain wrapper
embeddings_model = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
vectorstore = FAISS(embeddings_model.embed_query, index, metadata.to_dict("records"))

llm = Ollama(model=OLLAMA_MODEL)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

if __name__ == "__main__":
    query = "What was the company's revenue in 2023?"
    print("Q:", query)
    print("A:", qa_chain.run(query))

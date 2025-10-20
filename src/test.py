from src.retriever import build_langchain_retriever as get_retriever
retriever = get_retriever()

query = "the value of B-BBEE level (South Africa) in Absa document in 2019"
docs = retriever.invoke(query)  # ✅ new API for LangChain 0.2+

print(f"Retrieved {len(docs)} documents.")
for i, doc in enumerate(docs[:3]):
    print(f"\n--- Document {i+1} ---")
    print(doc.page_content[:500])

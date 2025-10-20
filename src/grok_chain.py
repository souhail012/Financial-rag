from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_core.output_parsers import StrOutputParser
from src.retriever import build_langchain_retriever
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
You are "Financial RAG GPT", an expert assistant specialized in financial analysis.

Your role is to answer questions based strictly on the retrieved context from corporate filings,
annual reports, ESG reports, investor presentations, and market data.

Guidelines:
- ✅ Use ONLY the provided context to answer. Do not invent, guess, or assume anything.
- 🧾 Always include relevant numerical values, company names, and report filenames if available.
- 📊 Structure answers clearly and concisely, focusing on key figures or facts.
- 🕵️ If the context does not contain enough information to answer, say: 
  "The context does not provide enough information to answer this question."
- 🚫 Do NOT add any external information or personal opinions.
- 🧠 When multiple data points are present, choose the most directly relevant to the question.

Your output will be evaluated for:
- Faithfulness: how well your answer aligns with the retrieved context.
- Context Relevance: how well the retrieved context supports the question.

Answer in a professional, factual tone.
"""


QA_TEMPLATE = PromptTemplate(
    template=SYSTEM_PROMPT + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
    input_variables=["context", "question"],
)

def get_grok_chain():
    """Build a LangChain RAG pipeline with Groq and FAISS retriever."""
    retriever = build_langchain_retriever()

 
    llm = ChatGroq(
    temperature=0,
    model="llama-3.1-8b-instant",  
    api_key=os.getenv("GROQ_API_KEY"),
)


    rag_chain = (
        RunnableMap({
            "context": lambda x: retriever.invoke(x["question"]),
            "question": RunnablePassthrough(),
        })
        | (lambda x: {
            "context": "\n\n".join([d.page_content for d in x["context"]]),
            "question": x["question"],
        })
        | QA_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return rag_chain


def ask_grok(question: str):
    """Ask Grok a financial question."""
    chain = get_grok_chain()
    print("🔍 Searching documents...")
    answer = chain.invoke({"question": question})

    print("\n💬 **Answer:**")
    print(answer)


if __name__ == "__main__":
    ask_grok("What was the total revenue reported by Apple in 2023?")

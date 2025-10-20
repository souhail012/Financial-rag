import streamlit as st
from src.grok_chain import get_grok_chain

st.set_page_config(page_title="💹 Financial RAG (Groq)", page_icon="💬")

st.title("💬 Financial RAG — Powered by Groq")
st.write("Ask questions about your financial reports and filings.")

# Initialize chain once
if "rag_chain" not in st.session_state:
    with st.spinner("Loading Groq RAG pipeline..."):
        st.session_state.rag_chain = get_grok_chain()

# Input area
query = st.text_input("Enter your financial question:")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("🔍 Searching and generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke({"question": query})
                st.markdown(f"### 💬 Answer\n{answer}")
            except Exception as e:
                st.error(f"❌ Error: {e}")

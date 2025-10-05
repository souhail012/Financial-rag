import streamlit as st
from src.retriever import build_langchain_qa

# Load your QA chain once
st.session_state.qa_chain = st.session_state.get("qa_chain", build_langchain_qa())

st.title("💬 Financial RAG GPT")
st.write("Ask questions about your financial documents!")

# Input box
query = st.text_input("Enter your question:")

if st.button("Ask"):
    if query.strip() == "":
        st.warning("Please type a question!")
    else:
        with st.spinner("Fetching answer..."):
            try:
                # Run the QA chain
                answer = st.session_state.qa_chain.run(query)
                st.markdown(f"**Answer:** {answer}")
            except Exception as e:
                st.error(f"Error: {e}")

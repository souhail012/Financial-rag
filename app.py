import streamlit as st
from src.grok_chain import get_grok_chain

# -------------------------------
# ⚙️ PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="📊 RAG Assistant for Annual & Financial Reports",
    page_icon="💼",
    layout="centered"
)

# -------------------------------
# 🎨 CUSTOM STYLES
# -------------------------------
st.markdown(
    """
    <style>
        /* -------------------- GLOBAL STYLING -------------------- */
        .main {
            max-width: 80%;
            margin: auto;
            background-color: #f8f9fa;
            padding: 3rem 2.5rem;
            border-radius: 14px;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        }

        /* -------------------- TITLE -------------------- */
        .title {
            font-size: 3.5rem;
            font-weight: 700;
            color: #1f4e79;
            text-align: center;
            margin-bottom: 1rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        /* -------------------- SUBTITLE -------------------- */
        .subtitle {
            text-align: center;
            color: #555;
            font-size: 1.25rem;
            margin-bottom: 3rem;
            max-width: 750px;
            margin-left: auto;
            margin-right: auto;
        }

        /* -------------------- INPUT -------------------- */
        .stTextInput > div > div > input {
            font-size: 1.1rem;
            border-radius: 10px;
            padding: 0.8rem 1rem;
            border: 2px solid #e0e0e0;
            transition: border-color 0.3s ease;
        }
        .stTextInput > div > div > input:focus {
            border-color: #1f4e79;
        }

        /* -------------------- BUTTON -------------------- */
        .stButton > button {
            background-color: #1f4e79;
            color: white;
            font-weight: 600;
            border-radius: 8px;
            height: 3rem;
            width: 220px;
            font-size: 1.1rem;
            transition: all 0.2s ease;
            border: none;
        }
        .stButton > button:hover {
            background-color: #163a5a;
            transform: translateY(-2px);
        }

        /* -------------------- ANSWER BOX -------------------- */
        .answer-box {
            background-color: white;
            border-radius: 12px;
            padding: 2rem;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.08);
            margin-top: 2rem;
        }

        /* -------------------- FOOTER -------------------- */
        .footer {
            text-align: center;
            color: #777;
            font-size: 0.95rem;
            margin-top: 3rem;
        }

        hr {
            border: 1px solid #ddd;
            margin-top: 3rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# 💼 HEADER
# -------------------------------
st.markdown(
    """
    <div class="title">RAG Assistant for Annual & Financial Reports</div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Ask questions about corporate filings, ESG reports, and financial data — get clear, factual answers.</p>',
    unsafe_allow_html=True
)

# -------------------------------
# ⚙️ INITIALIZE RAG CHAIN
# -------------------------------
if "rag_chain" not in st.session_state:
    with st.spinner("⚙️ Initializing the RAG pipeline..."):
        st.session_state.rag_chain = get_grok_chain()

# -------------------------------
# 💬 USER INPUT
# -------------------------------
st.markdown("## 🔎 Ask a Financial Question")

query = st.text_input(
    "Enter your question:",
    placeholder="e.g. What was Absa’s B-BBEE level in 2020?",
)

col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit = st.button("💬 Generate Answer")

# -------------------------------
# 🚀 PROCESS QUERY
# -------------------------------
if submit:
    if not query.strip():
        st.warning("⚠️ Please enter a question before submitting.")
    else:
        with st.spinner("🔍 Searching reports and generating answer..."):
            try:
                answer = st.session_state.rag_chain.invoke({"question": query})
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown("### 💬 Answer")
                st.write(answer)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"❌ An error occurred: {e}")

# -------------------------------
# 📘 FOOTER
# -------------------------------
st.markdown(
    """
    <hr>
    <p class="footer">© 2025 <b>RAG Assistant for Annual & Financial Reports</b> — Built with ❤️ using Streamlit & Groq</p>
    """,
    unsafe_allow_html=True
)

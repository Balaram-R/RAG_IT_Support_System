# app.py
# Run this with: streamlit run app.py
# Opens a web app in your browser automatically

import streamlit as st
from rag_engine import ask    # import our engine

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="IT Support AI",
    page_icon="🖥️",
    layout="centered"
)

# ── Header ────────────────────────────────────────────────────
st.title("🖥️ IT Support AI Assistant")
st.caption("Ask any IT question. Powered by Internal Sources — your data never leaves your machine.")

# ── Sidebar — shows system info ───────────────────────────────
with st.sidebar:
    st.header("About this system :")
    st.markdown("""
**Model:** Llama 3.2 (local)

**Architecture:** RAG with ChromaDB

**Data:** IT Support Knowledge Base

Built by: Balaram R
""")
    
#
# 
# 
#     
    st.info("All processing happens locally. No data sent to any server.")

# ── Main input ────────────────────────────────────────────────
question = st.text_input(
    "Describe your IT problem:",
    placeholder="e.g. My WiFi keeps disconnecting every hour"
)

# ── Handle submission ─────────────────────────────────────────
if st.button("Get Answer", type="primary") or question:
    if question.strip():

        # Show a spinner while thinking
        with st.spinner("Searching knowledge base..."):
            result = ask(question)      # call our engine

        # ── Display the answer ────────────────────────────────────
        st.markdown("### Answer")
        st.write(result["answer"])

        # ── Show confidence level ─────────────────────────────────
        confidence = result["confidence"]
        if confidence == "high":
            st.success("High confidence answer")
        elif confidence == "medium":
            st.warning("Medium confidence — verify if critical")
        elif confidence == "low":
            st.error("Not found in knowledge base")
        elif confidence == "error":
            st.error("System error — please try again")

        # ── Show sources used ─────────────────────────────────────
        if result["sources"]:
            with st.expander("Sources used"):
                for source in result["sources"]:
                    st.write(f"📄 {source}")
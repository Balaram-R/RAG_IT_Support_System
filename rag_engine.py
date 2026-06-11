# rag_engine.py
# The core engine. All 5 architect layers are here.
# Import this into app.py

import logging
import ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# ── Layer 5: Set up logging before anything else ──────────────
# Every error gets written to rag_errors.log automatically
# You can open this file any time to see what went wrong
logging.basicConfig(
    filename="rag_errors.log",
    level=logging.ERROR,
    format="%(asctime)s — %(message)s"
)

# ── Load vector database once when this file is imported ─────
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"
)
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model
)

# ── The main function — call this with any IT question ────────
def ask(question: str) -> dict:
    """
    Takes a question, returns a dict with answer, confidence, sources.
    Always returns something — never crashes the app.
    """

    # ── Layer 1: Input validation ─────────────────────────────────
    if not question or not question.strip():
        return {
            "answer": "Please type a question.",
            "confidence": "none",
            "sources": []
        }

    if len(question.strip()) > 300:
        return {
            "answer": "Question too long. Please keep it under 300 characters.",
            "confidence": "none",
            "sources": []
        }

    question = question.strip()

    try:
        # ── Layer 2: Retrieve relevant chunks ────────────────────────
        results = vectorstore.similarity_search_with_score(
            question,
            k=3             # get top 3 most relevant chunks
        )

        # Check if retrieval found anything useful
        # Score below 1.0 means relevant (ChromaDB uses distance not similarity)
        # If best score is above 1.5 — nothing useful found
        if not results or results[0][1] > 1.5:
            return {
                "answer": "I don't have information about this in my knowledge base. Please contact IT support directly.",
                "confidence": "low",
                "sources": []
            }

        # Build context from the top chunks
        context = "\n\n".join([doc.page_content for doc, score in results])
        sources = list(set([doc.metadata.get("source", "docs") for doc, _ in results]))

        # ── Layer 3: Call local LLM via Ollama ───────────────────────
        prompt = f"""You are a helpful IT support assistant.
Answer the question using ONLY the information provided below.
If the answer is not in the information, say "I don't have that information."
Be concise and clear. Use numbered steps if giving instructions.

KNOWLEDGE BASE:
{context}

QUESTION: {question}

ANSWER:"""

        response = ollama.chat(
            model="llama3.2",            # your local model
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0}   # always consistent, no randomness
        )

        answer = response["message"]["content"].strip()

        # ── Layer 4: Return structured output ────────────────────────
        return {
            "answer": answer,
            "confidence": "high" if results[0][1] < 0.8 else "medium",
            "sources": sources
        }

    # ── Layer 5: Catch every possible failure ────────────────────
    except Exception as e:
        logging.error(f"RAG failed for question '{question}': {e}")
        return {
            "answer": "Something went wrong. Please try again in a moment.",
            "confidence": "error",
            "sources": []
        }
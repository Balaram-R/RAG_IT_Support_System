# app.py
# Run this with: streamlit run app.py
# Opens a web app in your browser automatically

import threading

import streamlit as st
from rag_engine import (
    ask, get_kb_stats, ollama_health, OLLAMA_MODEL, EMBEDDING_MODEL,
    preload_reranker,
)

# Preload the cross-encoder reranker in the background so the first
# query doesn't pay the ~8s model-loading cost.
threading.Thread(target=preload_reranker, daemon=True).start()

# ── Page configuration ────────────────────────────────────────
st.set_page_config(
    page_title="IT Support AI",
    page_icon="🖥️",
    layout="centered",
)

# ── LED strip renderer ────────────────────────────────────────
# status -> (color, glow)
LED_COLORS = {
    "pending": ("#9ca3af", "0 0 0 rgba(0,0,0,0)"),
    "running": ("#f59e0b", "0 0 14px #f59e0b"),
    "done":    ("#22c55e", "0 0 14px #22c55e"),
    "skipped": ("#6b7280", "0 0 0 rgba(0,0,0,0)"),
    "error":   ("#ef4444", "0 0 14px #ef4444"),
}


def led_strip_html(stages) -> str:
    """Renders the 7 pipeline stages as LED lights with labels."""
    cells = []
    for s in stages:
        color, glow = LED_COLORS.get(s["status"], LED_COLORS["pending"])
        pulse = "animation:ledpulse 1s infinite;" if s["status"] == "running" else ""
        cells.append(
            f'<div style="text-align:center;width:118px;flex:0 0 auto;">'
            f'<div title="{s["message"]}" style="width:24px;height:24px;border-radius:50%;'
            f'margin:0 auto;background:{color};box-shadow:{glow};{pulse}'
            f'border:2px solid rgba(255,255,255,.25);"></div>'
            f'<div style="font-size:11px;color:#6b7280;margin-top:6px;line-height:1.2;">'
            f'{s["name"]}</div></div>'
        )
    return (
        '<style>@keyframes ledpulse{0%,100%{opacity:1}50%{opacity:.35}}</style>'
        '<div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:center;'
        'padding:14px 8px;background:#111827;border-radius:14px;'
        'border:1px solid #374151;">' + "".join(cells) + "</div>"
    )


def led_legend() -> str:
    items = [
        ("#22c55e", "Done"),
        ("#f59e0b", "Running"),
        ("#ef4444", "Error"),
        ("#9ca3af", "Pending"),
        ("#6b7280", "Skipped"),
    ]
    spans = "".join(
        f'<span style="display:inline-flex;align-items:center;margin-right:14px;">'
        f'<span style="width:12px;height:12px;border-radius:50%;background:{c};'
        f'display:inline-block;margin-right:5px;"></span>{t}</span>'
        for c, t in items
    )
    return f'<div style="font-size:12px;color:#6b7280;margin-top:6px;">{spans}</div>'


# ── Header ────────────────────────────────────────────────────
st.title("🖥️ IT Support AI Assistant")
st.caption("Ask any IT question. Powered by Internal Sources — your data never leaves your machine.")

# ── Sidebar — system info ─────────────────────────────────────
with st.sidebar:
    st.header("About this system")
    st.markdown(f"""
**Model:** {OLLAMA_MODEL} (local)

**Embeddings:** {EMBEDDING_MODEL}

**Re-ranker:** cross-encoder/ms-marco-MiniLM-L-6-v2

**Architecture:** RAG with ChromaDB + re-ranking

**Data:** IT Support Knowledge Base

Built by: Balaram R
""")

    # Ollama status LED
    ok = ollama_health()
    dot = "#22c55e" if ok else "#ef4444"
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:6px 0;">'
        f'<span style="width:14px;height:14px;border-radius:50%;background:{dot};'
        f'box-shadow:0 0 10px {dot};display:inline-block;"></span>'
        f'<span>Ollama: {"online" if ok else "offline"}</span></div>',
        unsafe_allow_html=True,
    )

    # Knowledge base stats
    stats = get_kb_stats()
    if stats["ok"]:
        st.info(f"Knowledge base: **{stats['chunks']} chunks** ready")
    else:
        st.warning("Knowledge base empty — run ingest.py")

    st.info("All processing happens locally. No data sent to any server.")

# ── Main input ────────────────────────────────────────────────
with st.form("ask_form", clear_on_submit=False):
    question = st.text_input(
        "Describe your IT problem:",
        placeholder="e.g. My WiFi keeps disconnecting every hour",
    )
    submitted = st.form_submit_button("Get Answer", type="primary")

if submitted and question.strip():
    # ── Live LED pipeline ────────────────────────────────────────
    st.markdown("### Pipeline status")
    led_placeholder = st.empty()
    led_placeholder.markdown(
        led_strip_html([
            {"name": n, "status": "pending", "message": ""}
            for n in ["Input validation", "Query embedding", "ChromaDB retrieval",
                      "Re-ranking", "Context assembly", "Ollama processing",
                      "Grounding check", "Confidence scoring"]
        ]),
        unsafe_allow_html=True,
    )

    def on_progress(stages):
        led_placeholder.markdown(led_strip_html(stages), unsafe_allow_html=True)

    result = ask(question, progress_callback=on_progress)

    # Final LED state
    led_placeholder.markdown(led_strip_html(result["pipeline"]), unsafe_allow_html=True)
    st.markdown(led_legend(), unsafe_allow_html=True)

    # ── Confidence score ─────────────────────────────────────────
    score = result["confidence_score"]
    label = result["confidence"]
    if label == "high":
        color, icon = "#22c55e", "🟢"
    elif label == "medium":
        color, icon = "#f59e0b", "🟡"
    elif label == "low":
        color, icon = "#ef4444", "🔴"
    else:
        color, icon = "#6b7280", "⚪"

    st.markdown("### Confidence score")
    st.markdown(
        f'<div style="display:flex;align-items:center;gap:12px;">'
        f'<span style="font-size:34px;font-weight:700;color:{color};">{score}</span>'
        f'<span style="font-size:15px;color:{color};">/ 100 {icon} {label.upper()}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )
    st.progress(score / 100)

    # ── Answer ───────────────────────────────────────────────────
    st.markdown("### Answer")
    st.write(result["answer"])

    # ── Sources with scores ──────────────────────────────────────
    if result["sources"]:
        with st.expander(f"Sources used ({len(result['sources'])})"):
            for src in result["sources"]:
                st.markdown(
                    f"📄 **{src['source']}** — relevance **{src['score']}/100**"
                )
                st.caption(src["snippet"])
    elif result["status"] == "ok":
        st.warning("No sources matched — answer may be generic.")

elif submitted:
    st.warning("Please type a question first.")
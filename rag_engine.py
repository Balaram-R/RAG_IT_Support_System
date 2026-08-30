# rag_engine.py
# The core engine. Runs a 7-stage pipeline with guard rails,
# re-ranking and a confidence score. Import this into app.py.
#
# Pipeline stages (shown as LEDs in the UI):
#   1. input_validation   - guard rails on the question
#   2. query_embedding    - turn the question into a vector
#   3. chroma_retrieval   - fetch candidate chunks from ChromaDB
#   4. reranking          - cross-encoder re-scores the candidates
#   5. context_assembly   - build a tight, deduplicated context
#   6. ollama_generation  - local LLM answers from that context
#   7. confidence_scoring - 0-100 score from retrieval + answer checks

import logging
import re
import threading
import time

import ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# ── Settings ────────────────────────────────────────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"  # ~22 MB, one-time download
OLLAMA_MODEL = "qwen2.5:3b"
OLLAMA_HOST = "http://127.0.0.1:11434"  # pinned — ignores broken OLLAMA_HOST env vars
CHROMA_DIR = "./chroma_db"
INITIAL_K = 15      # candidates pulled from Chroma
FINAL_K = 5         # kept after re-ranking
MAX_QUERY_LEN = 500
OLLAMA_TIMEOUT = 120
RETRIEVAL_GATE = 0.40  # min top-chunk similarity to allow the LLM to answer

# Ollama client pinned to a working address
ollama_client = ollama.Client(host=OLLAMA_HOST)

# ── Layer 5: logging (every error lands in rag_errors.log) ─────
logging.basicConfig(
    filename="rag_errors.log",
    level=logging.ERROR,
    format="%(asctime)s — %(message)s",
)

# ── Guard rail: prompt-injection patterns (case-insensitive) ───
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions|prompts|messages)",
    r"disregard\s+(all\s+)?(previous|prior|above)\s+(instructions|prompts)",
    r"forget\s+(everything|all)\s+(you\s+)?(learned|know)",
    r"system\s+prompt",
    r"reveal\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"show\s+(me\s+)?(your|the)\s+(system\s+)?(prompt|instructions)",
    r"print\s+(your|the)\s+(system\s+)?(prompt|instructions)",
    r"you\s+are\s+now\s+",
    r"act\s+as\s+(if\s+you\s+are\s+)?(a\s+)?(dan|jailbreak)",
    r"developer\s+mode",
    r"jailbreak",
    r"bypass\s+(your|the|all)\s+(rules|restrictions|safety|guardrails)",
    r"override\s+(your|the|all)\s+(instructions|rules|guidelines)",
    r"new\s+instructions",
    r"do\s+not\s+follow\s+(your|the)\s+(instructions|guidelines|rules)",
    r"pretend\s+you\s+are",
    r"simulate\s+(a\s+)?(dan|jailbreak|unfiltered)",
    r"secret\s+instructions",
    r"hidden\s+instructions",
    r"what\s+are\s+your\s+(instructions|rules|guidelines)",
    r"how\s+do\s+you\s+work\s+internally",
]

# ── Guard rail: IT-domain keywords (soft check, lowers confidence) ─
IT_KEYWORDS = [
    "wifi", "wi-fi", "wireless", "network", "internet", "ethernet", "lan", "vlan",
    "dns", "dhcp", "ip address", "ipconfig", "gateway", "router", "switch", "firewall",
    "vpn", "port", "tcp", "udp", "ping", "tracert", "traceroute", "nslookup",
    "packet", "latency", "bandwidth", "connectivity", "connection", "disconnect",
    "slow", "speed", "signal", "ssid", "radius", "802.1x", "authentication",
    "password", "credential", "certificate", "proxy", "server", "host", "domain",
    "email", "outlook", "printer", "laptop", "desktop", "computer", "pc", "windows",
    "linux", "mac", "browser", "application", "software", "install", "update",
    "crash", "freeze", "blue screen", "bsod", "error", "log", "troubleshoot",
    "troubleshooting", "fix", "issue", "problem", "not working", "can't connect",
    "cannot connect", "no internet", "offline", "slow internet", "wifi keeps",
    "ip address", "obtain", "lease", "subnet", "mask", "gateway", "dns server",
    "malware", "virus", "antivirus", "backup", "restore", "recovery", "boot",
    "startup", "shutdown", "restart", "reboot", "driver", "hardware", "usb",
    "monitor", "keyboard", "mouse", "headset", "camera", "microphone", "speaker",
    "share", "folder", "file", "permission", "access", "account", "login", "sign in",
    "lock", "unlock", "reset", "forgot", "password reset", "mfa", "2fa", "otp",
    "ticket", "incident", "support", "helpdesk", "it support", "help",
]

# ── Guard rail: network-domain keywords (hard gate) ─────────────
# The knowledge base ONLY covers network support topics (docs/).
# If the question contains none of these terms, it cannot be
# answered from the KB — refuse instead of retrieving a loosely
# related doc and answering the wrong question.
NETWORK_KEYWORDS = [
    "wifi", "wi-fi", "wireless", "network", "internet", "ethernet",
    "lan", "vlan", "dns", "dhcp", "ip", "ipconfig", "gateway",
    "router", "switch", "firewall", "vpn", "port", "tcp", "udp",
    "ping", "tracert", "traceroute", "nslookup", "packet", "latency",
    "bandwidth", "connectivity", "connection", "disconnect", "signal",
    "ssid", "radius", "802.1x", "subnet", "lease", "hostname",
    "proxy", "mac address", "arp", "route", "routing", "wired",
    "uplink", "access point", "controller", "packet loss",
    "ip address", "dns server", "dhcp scope", "dhcp relay",
    "network support", "incident response",
]

# ── Pipeline tracker ────────────────────────────────────────────
STAGE_NAMES = {
    "input_validation": "Input validation",
    "query_embedding": "Query embedding",
    "chroma_retrieval": "ChromaDB retrieval",
    "reranking": "Re-ranking",
    "context_assembly": "Context assembly",
    "ollama_generation": "Ollama processing",
    "grounding_check": "Grounding check",
    "confidence_scoring": "Confidence scoring",
}


class Pipeline:
    """Tracks each stage's status so the UI can render LED lights."""

    def __init__(self, callback=None):
        self.stages = []
        self.callback = callback
        for key in STAGE_NAMES:
            self.stages.append({
                "key": key,
                "name": STAGE_NAMES[key],
                "status": "pending",   # pending | running | done | skipped | error
                "message": "",
            })

    def _emit(self):
        if self.callback:
            try:
                self.callback(self.to_list())
            except Exception:
                pass  # UI callback must never break the engine

    def start(self, key, message=""):
        for s in self.stages:
            if s["key"] == key:
                s["status"] = "running"
                s["message"] = message
        self._emit()

    def done(self, key, message=""):
        for s in self.stages:
            if s["key"] == key:
                s["status"] = "done"
                s["message"] = message
        self._emit()

    def skip(self, key, message=""):
        for s in self.stages:
            if s["key"] == key:
                s["status"] = "skipped"
                s["message"] = message
        self._emit()

    def error(self, key, message=""):
        for s in self.stages:
            if s["key"] == key:
                s["status"] = "error"
                s["message"] = message
        self._emit()

    def to_list(self):
        return [dict(s) for s in self.stages]


# ── Load vector database once when this file is imported ───────
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embedding_model,
)

# ── Ollama health check (cached, non-blocking) ─────────────────
_ollama_ok = None
_ollama_checked_at = 0.0
_ollama_lock = threading.Lock()


def ollama_health(force=False) -> bool:
    """True if the Ollama server is reachable. Result cached for 30 s."""
    global _ollama_ok, _ollama_checked_at
    now = time.time()
    if not force and _ollama_ok is not None and (now - _ollama_checked_at) < 30:
        return _ollama_ok
    with _ollama_lock:
        try:
            ollama_client.list()
            _ollama_ok = True
        except Exception:
            _ollama_ok = False
        _ollama_checked_at = time.time()
    return _ollama_ok


# ── Re-ranker (lazy-loaded so app startup stays fast) ──────────
_reranker = None
_reranker_lock = threading.Lock()


def _get_reranker():
    global _reranker
    if _reranker is None:
        with _reranker_lock:
            if _reranker is None:
                from sentence_transformers import CrossEncoder
                _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


# ── Guard rail helpers ──────────────────────────────────────────
def _check_injection(text: str):
    """Returns the first matched injection pattern, or None."""
    lowered = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered):
            return pattern
    return None


def _is_it_related(text: str) -> bool:
    lowered = text.lower()
    return any(kw in lowered for kw in IT_KEYWORDS)


def _is_network_related(text: str) -> bool:
    """True if the question mentions any network-support concept.

    The KB only covers network topics, so a question with zero
    network terms can never be answered from docs/ — refuse early.
    """
    lowered = text.lower()
    return any(kw in lowered for kw in NETWORK_KEYWORDS)


# Phrases the LLM uses when it refuses to answer. Used to detect
# cautious refusals (retry with a nudge) vs genuine non-answers.
FALLBACK_PHRASES = [
    "i don't have that information",
    "i don't have information",
    "not in the information",
    "not provided in the",
    "i do not have",
    "cannot answer",
    "no information",
]


def _is_fallback_answer(text: str) -> bool:
    lowered = text.lower()
    return any(p in lowered for p in FALLBACK_PHRASES)


def _distance_to_sim(distance: float) -> float:
    """Chroma L2 distance -> 0..1 similarity."""
    return 1.0 / (1.0 + distance)


# ── Guard rail: grounding verification ──────────────────────────
# Ensures the LLM's answer is actually supported by the retrieved
# docs. Ungrounded sentences (hallucinations) get the answer rejected
# or the ungrounded tail trimmed off.
GROUNDING_SIM_THRESHOLD = 0.42   # min cosine sim for long answer units
GROUNDING_LEXICAL_MIN = 0.50     # min token overlap for short units
SHORT_UNIT_MAX = 60              # units shorter than this use the lexical check


def _cosine(a, b):
    import math
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1.0
    nb = math.sqrt(sum(x * x for x in b)) or 1.0
    return dot / (na * nb)


def _split_units(answer: str) -> list:
    """Split an answer into checkable units: lines first, then long
    lines on sentence boundaries. Keeps numbered list items intact."""
    units = []
    for line in answer.split("\n"):
        line = line.strip()
        if not line:
            continue
        if len(line) <= SHORT_UNIT_MAX:
            units.append(line)
        else:
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", line) if s.strip()]
            units.extend(sents)
    return units


def _lexical_overlap(unit: str, ctx_tokens: set) -> float:
    tokens = {t for t in re.findall(r"[a-zA-Z]{4,}", unit.lower())}
    if not tokens:
        return 1.0  # e.g. bare list numbering "2." — nothing to verify
    return len(tokens & ctx_tokens) / len(tokens)


def _grounding_check(answer: str, context_chunks: list, question: str = None) -> tuple:
    """
    Returns (grounded: bool, detail: str, trimmed: str|None).
    - Short units (list items, commands) are verified lexically against
      the docs — they embed poorly but their words appear verbatim.
    - Long units (prose) are embedded and compared against the docs.
    - Units that closely match the question itself (question echo) pass.
    - If the answer has a grounded prefix followed by ungrounded units,
      the tail is trimmed off. If the very first unit is ungrounded,
      the whole answer is rejected.
    """
    if not answer.strip():
        return False, "Empty answer", None

    units = _split_units(answer)
    if not units:
        return False, "No content to verify", None

    ctx_tokens = set()
    for chunk in context_chunks:
        ctx_tokens.update(re.findall(r"[a-zA-Z]{4,}", chunk.lower()))

    long_units = [u for u in units if len(u) > SHORT_UNIT_MAX]
    ctx_vecs = None
    if long_units:
        ctx_vecs = embedding_model.embed_documents(context_chunks)
        long_vecs = embedding_model.embed_documents(long_units)
        long_sims = {u: max(_cosine(v, cv) for cv in ctx_vecs)
                     for u, v in zip(long_units, long_vecs)}
    else:
        long_sims = {}

    # Question-echo detection: units that closely match the question
    # itself are not hallucinations (the model just echoed the input).
    q_vec = None
    if question and question.strip():
        q_vec = embedding_model.embed_query(question)

    checks = []  # (score, threshold) per unit
    for u in units:
        if len(u) > SHORT_UNIT_MAX:
            ctx_score = long_sims[u]
            threshold = GROUNDING_SIM_THRESHOLD
        else:
            ctx_score = _lexical_overlap(u, ctx_tokens)
            threshold = GROUNDING_LEXICAL_MIN
        if ctx_score >= threshold:
            checks.append((ctx_score, threshold))
            continue
        if q_vec is not None:
            u_vec = embedding_model.embed_query(u)
            q_sim = _cosine(u_vec, q_vec)
            if q_sim >= 0.60:
                checks.append((q_sim, 0.60))  # question echo — pass
                continue
        checks.append((ctx_score, threshold))

    first_bad = next(
        (i for i, (s, t) in enumerate(checks) if s < t),
        None,
    )
    if first_bad is None:
        return True, f"Min unit score {min(s for s, _ in checks):.2f}", None
    if first_bad == 0:
        return False, f"First unit ungrounded (score {checks[0][0]:.2f})", None

    trimmed = " ".join(units[:first_bad])
    return (True,
            f"Trimmed {len(units) - first_bad} ungrounded unit(s)",
            trimmed)


# ── KB stats for the sidebar ────────────────────────────────────
def get_kb_stats() -> dict:
    try:
        count = vectorstore._collection.count()
        return {"chunks": count, "ok": True}
    except Exception as e:
        logging.error(f"Could not read KB stats: {e}")
        return {"chunks": 0, "ok": False}


# ── The main function — call this with any IT question ──────────
def ask(question: str, progress_callback=None) -> dict:
    """
    Runs the full 7-stage pipeline.
    Returns a dict with answer, confidence (label), confidence_score,
    sources (with scores), pipeline (LED states) and status.
    Never crashes the app — always returns a dict.
    """
    pipe = Pipeline(progress_callback)

    # ── Stage 1: Input validation (guard rails) ───────────────────
    pipe.start("input_validation")
    if not question or not question.strip():
        pipe.error("input_validation", "Empty question")
        return _result("Please type a question.", 0, "none", [], pipe, "blocked")

    question = question.strip()
    if len(question) > MAX_QUERY_LEN:
        pipe.error("input_validation", f"Question too long (>{MAX_QUERY_LEN} chars)")
        return _result(
            f"Question too long. Please keep it under {MAX_QUERY_LEN} characters.",
            0, "none", [], pipe, "blocked",
        )

    injection = _check_injection(question)
    if injection:
        pipe.error("input_validation", "Prompt-injection pattern blocked")
        return _result(
            "That question looks like an attempt to override my instructions, "
            "so I can't answer it. Please rephrase as a normal IT support question.",
            0, "none", [], pipe, "blocked",
        )

    it_related = _is_it_related(question)
    pipe.done("input_validation",
              "OK" if it_related else "Passed (not clearly IT-related)")

    # Network-domain gate: the KB only covers network support. If the
    # question has no network term at all, refuse before retrieval —
    # otherwise the model answers a loosely-related doc to a question
    # it was never asked (e.g. "my computer is slow" -> WiFi steps).
    if not _is_network_related(question):
        pipe.skip("query_embedding", "Skipped — out of scope")
        pipe.skip("chroma_retrieval", "Skipped — out of scope")
        pipe.skip("reranking", "Skipped — out of scope")
        pipe.skip("context_assembly", "Skipped — out of scope")
        pipe.skip("ollama_generation", "Skipped — out of scope")
        pipe.skip("grounding_check", "Skipped — no answer generated")
        pipe.done("confidence_scoring", "Out of scope — low confidence")
        return _result(
            "This question is outside the knowledge base, which covers "
            "network support topics only. Please contact IT support directly.",
            15, "low", [], pipe, "no_match",
        )

    try:
        # ── Stage 2: Query embedding ────────────────────────────────
        pipe.start("query_embedding")
        query_vector = embedding_model.embed_query(question)
        pipe.done("query_embedding", f"Vector dim {len(query_vector)}")

        # ── Stage 3: ChromaDB retrieval ─────────────────────────────
        pipe.start("chroma_retrieval")
        results = vectorstore.similarity_search_with_score(
            question, k=INITIAL_K
        )
        if not results:
            pipe.skip("chroma_retrieval", "No chunks in knowledge base")
            pipe.skip("reranking", "Nothing to re-rank")
            pipe.skip("context_assembly", "No context available")
            pipe.skip("ollama_generation", "Skipped — no context")
            pipe.skip("grounding_check", "Skipped — no answer generated")
            pipe.done("confidence_scoring", "No retrieval")
            return _result(
                "The knowledge base is empty. Run ingest.py first to build it.",
                0, "low", [], pipe, "no_match",
            )
        pipe.done("chroma_retrieval", f"{len(results)} candidates")

        # ── Stage 4: Re-ranking (cross-encoder) ────────────────────
        pipe.start("reranking")
        reranker = _get_reranker()
        pairs = [(question, doc.page_content) for doc, _ in results]
        rerank_scores = reranker.predict(pairs)
        ranked = sorted(
            zip(results, rerank_scores),
            key=lambda x: x[1],
            reverse=True,
        )[:FINAL_K]
        pipe.done("reranking", f"Top {len(ranked)} of {len(results)} kept")

        # ── Strict retrieval gate (guard rail) ─────────────────────
        # If the best chunk is not relevant enough, we REFUSE to answer
        # without calling the LLM at all. This guarantees the model can
        # never answer from its own training knowledge — only from docs/.
        top_sim = _distance_to_sim(ranked[0][0][1])
        if top_sim < RETRIEVAL_GATE:
            pipe.skip("context_assembly", "No relevant chunk found")
            pipe.skip("ollama_generation", "Skipped — nothing relevant in docs")
            pipe.skip("grounding_check", "Skipped — no answer generated")
            pipe.done("confidence_scoring", "No relevant retrieval")
            return _result(
                "I don't have information about this in the knowledge base. "
                "Please contact IT support directly.",
                15, "low", [], pipe, "no_match",
            )

        # ── Stage 5: Context assembly ───────────────────────────────
        pipe.start("context_assembly")
        context_parts = []
        seen_sources = set()
        for (doc, distance), rscore in ranked:
            source = doc.metadata.get("source", "docs")
            seen_sources.add(source)
            context_parts.append(
                f"[Source: {source}]\n{doc.page_content}"
            )
        context = "\n\n---\n\n".join(context_parts)
        pipe.done("context_assembly", f"{len(context_parts)} chunks, {len(context)} chars")

        # ── Stage 6: Ollama generation ─────────────────────────────
        pipe.start("ollama_generation")
        if not ollama_health():
            pipe.error("ollama_generation", "Ollama server not reachable")
            pipe.skip("grounding_check", "Skipped — no answer generated")
            pipe.skip("confidence_scoring", "No answer generated")
            return _result(
                "Ollama is not running. Start it (run `ollama serve` or open the "
                "Ollama app) and try again.",
                0, "error", [], pipe, "error",
            )

        prompt = _build_prompt(question, context)
        answer = _generate(question, context)
        pipe.done("ollama_generation", f"Model {OLLAMA_MODEL}")

        # ── Stage 7: Grounding check (guard rail) ──────────────────
        # Verify the answer is actually supported by the retrieved docs.
        # We check against ALL retrieved candidates (all from docs/), not
        # just the top context, so valid answers that reference other
        # parts of the same docs are accepted while hallucinations are not.
        pipe.start("grounding_check")
        candidate_chunks = [doc.page_content for doc, _ in results]

        # Cautious-refusal retry: if the model says it has no information
        # even though relevant docs were retrieved, retry once with a nudge.
        if _is_fallback_answer(answer) and top_sim >= RETRIEVAL_GATE:
            pipe.done("grounding_check", "Model refused — retrying with nudge")
            answer = _generate(question, context, nudge=True)
            if _is_fallback_answer(answer):
                # Still refuses. The docs DO contain relevant info, so show
                # the sources and a medium confidence instead of a hard miss.
                pipe.done("grounding_check", "Docs relevant but model declined")
                pipe.done("confidence_scoring", "Model declined twice")
                sources = _build_sources(ranked)
                return _result(
                    "I don't have that information. However, the knowledge base "
                    "contains related material — see the sources below.",
                    45, "medium", sources, pipe, "ok",
                )
            pipe.done("ollama_generation", f"Model {OLLAMA_MODEL} (retry)")

        grounded, detail, trimmed = _grounding_check(answer, candidate_chunks, question)
        if not grounded:
            pipe.error("grounding_check", f"Rejected — {detail}")
            pipe.done("confidence_scoring", "Answer rejected")
            return _result(
                "I don't have information about this in the knowledge base. "
                "Please contact IT support directly.",
                15, "low", [], pipe, "no_match",
            )
        if trimmed is not None:
            answer = trimmed
        pipe.done("grounding_check", f"Verified — {detail}")

        # ── Stage 8: Confidence scoring ────────────────────────────
        pipe.start("confidence_scoring")

        # Relative re-rank score: min-max over the batch so confidence
        # reflects "best candidate vs the rest", not query style.
        all_logits = [rs for _, rs in ranked]
        lo, hi = min(all_logits), max(all_logits)
        if hi - lo < 1e-6:
            rel_rerank = 1.0
        else:
            rel_rerank = (ranked[0][1] - lo) / (hi - lo)

        top_sim = _distance_to_sim(ranked[0][0][1])
        score = 0.60 * rel_rerank + 0.40 * top_sim

        # Guard rail: LLM said it has no information -> reduce confidence.
        # Hard cap only when retrieval was genuinely weak (top_sim < 0.40);
        # otherwise the small model is just being cautious and the answer
        # may still be useful, so cap at medium.
        if _is_fallback_answer(answer):
            score = min(score, 0.50 if top_sim >= 0.40 else 0.25)

        # Guard rail: answer too short to be useful
        if len(answer) < 20:
            score -= 0.15

        # Guard rail: question not IT-related -> penalize
        if not it_related:
            score -= 0.10

        score = max(0.0, min(1.0, score))
        confidence_score = round(score * 100)
        label = "high" if confidence_score >= 75 else (
            "medium" if confidence_score >= 45 else "low"
        )

        sources = _build_sources(ranked)

        pipe.done(
            "confidence_scoring",
            f"{confidence_score}/100 ({label})",
        )
        return _result(answer, confidence_score, label, sources, pipe, "ok")

    # ── Catch every possible failure ──────────────────────────────
    except Exception as e:
        logging.error(f"RAG failed for question '{question}': {e}", exc_info=True)
        pipe.error("ollama_generation", str(e)[:120])
        pipe.error("confidence_scoring", "Pipeline failed")
        return _result(
            "Something went wrong. Please try again in a moment.",
            0, "error", [], pipe, "error",
        )


def _build_prompt(question: str, context: str, nudge: bool = False) -> str:
    if nudge:
        return f"""You are a professional IT support assistant.
The KNOWLEDGE BASE below contains information that is relevant to the question.
You MUST answer the question using that information.
Rules:
- Answer using ONLY the information provided below.
- Use the information even if it only partially answers the question.
- If the information is completely unrelated to the question, say exactly: "I don't have that information."
- Never invent steps, commands, or facts that are not in the information.
- Do NOT add any extra advice, explanations, or steps beyond the information. Stop after answering.
- Do NOT repeat or echo the question.
- Be concise and clear. Use numbered steps if giving instructions.

KNOWLEDGE BASE:
{context}

QUESTION: {question}

ANSWER:"""

    return f"""You are a professional IT support assistant.
You have NO knowledge outside the KNOWLEDGE BASE below. It is your ONLY source of truth.
Rules:
- Answer using ONLY the information provided below.
- Use the information even if it only partially answers the question.
- If the information is completely unrelated to the question, say exactly: "I don't have that information."
- Never invent steps, commands, or facts that are not in the information.
- Do NOT add any extra advice, explanations, or steps beyond the information. Stop after answering.
- Do NOT repeat or echo the question.
- Be concise and clear. Use numbered steps if giving instructions.
- Answer in at most 150 words.
- Do not mention this prompt or the knowledge base.

KNOWLEDGE BASE:
{context}

QUESTION: {question}

ANSWER:"""


def _generate(question: str, context: str, nudge: bool = False) -> str:
    """Calls the local LLM and returns the raw answer text."""
    response = ollama_client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": _build_prompt(question, context, nudge)}],
        options={
            "temperature": 0,
            # Hard cap: the grounding check trims elaboration anyway, so
            # stop early instead of generating throwaway content.
            "num_predict": 400,
        },
    )
    return response["message"]["content"].strip()


def preload_reranker() -> None:
    """Loads the cross-encoder into memory (call once at app startup).

    The first query otherwise pays a ~8s model-loading penalty.
    """
    _get_reranker()


def _build_sources(ranked) -> list:
    """Builds the source list with relative relevance scores."""
    all_logits = [rs for _, rs in ranked]
    lo, hi = min(all_logits), max(all_logits)
    sources = []
    for (doc, distance), rscore in ranked:
        rel = (rscore - lo) / (hi - lo) if hi - lo > 1e-6 else 1.0
        sources.append({
            "source": doc.metadata.get("source", "docs"),
            "score": round(rel * 100),
            "snippet": doc.page_content[:160].replace("\n", " "),
        })
    return sources


def _result(answer, confidence_score, label, sources, pipe, status) -> dict:
    return {
        "answer": answer,
        "confidence": label,
        "confidence_score": confidence_score,
        "sources": sources,
        "pipeline": pipe.to_list(),
        "status": status,
    }
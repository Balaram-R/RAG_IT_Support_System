# IT Support Knowledge Assistant

A local **Retrieval-Augmented Generation (RAG)** application for answering IT and network troubleshooting questions using **only the documents in the `docs/` knowledge base**.

The system combines semantic retrieval, **cross-encoder re-ranking**, and a local **Ollama LLM**, with multiple guardrails designed to keep responses grounded in the available documentation.

> **Core principle:** If the knowledge base does not contain sufficient information to answer a question, the system refuses instead of guessing.

---

## Key Features

* **Knowledge-base grounded answers** — responses are generated from documents in `docs/`
* **8-stage RAG pipeline** with live LED status indicators
* **Cross-encoder re-ranking** using `ms-marco-MiniLM-L-6-v2`
* **Grounding verification** for generated answer units
* **Confidence scoring** from `0–100`
* **Confidence labels** — `HIGH`, `MEDIUM`, `LOW`
* **Prompt-injection blocking**
* **Network-domain gate** to reject unrelated questions
* **Retrieval gate** to prevent unsupported LLM generation
* **ChromaDB** vector database
* **Local LLM** using Ollama and `qwen2.5:3b`
* **Streamlit** web interface
* **Offline operation** after local models and dependencies are installed
* No external LLM API costs

---

## How It Works

The application does not simply send a user question directly to an LLM.

Instead, the question passes through a controlled pipeline:

```text
User Question
      │
      ▼
┌──────────────────────────────┐
│ 1. Input Validation          │
│    Injection + Domain Gate   │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 2. Query Embedding           │
│    all-MiniLM-L6-v2          │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 3. ChromaDB Retrieval        │
│    Top 15 Candidates         │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 4. Cross-Encoder Re-ranking  │
│    Keep Top 5                │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 5. Context Assembly          │
│    Retrieved Documentation   │
└──────────────┬───────────────┘
               ▼
        Retrieval Gate
               │
        ┌──────┴──────┐
        │             │
     Pass           Fail
        │             │
        ▼             ▼
┌──────────────┐   Refuse
│ 6. Ollama    │
│ Generation   │
└──────┬───────┘
       ▼
┌──────────────────────────────┐
│ 7. Grounding Check           │
│    Verify Answer Units       │
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ 8. Confidence Scoring        │
│    0–100 + Label             │
└──────────────┬───────────────┘
               ▼
        Answer + Sources
```

---

## Pipeline Stages

### 1. Input Validation

The system checks the incoming question before retrieval.

It includes:

* Prompt-injection detection
* Network-domain validation

Questions that clearly fall outside the supported knowledge domain can be rejected before expensive retrieval or generation.

---

### 2. Query Embedding

The question is converted into a vector representation using:

```text
all-MiniLM-L6-v2
```

This allows the system to search the knowledge base based on semantic similarity rather than requiring exact keyword matches.

---

### 3. ChromaDB Retrieval

The embedded query is searched against the ChromaDB vector database.

The initial retrieval stage returns:

```text
Top 15 candidates
```

These candidates are then passed to the re-ranking stage.

---

### 4. Cross-Encoder Re-ranking

The retrieved candidates are re-ranked using:

```text
ms-marco-MiniLM-L-6-v2
```

The system keeps the strongest:

```text
Top 5
```

This second-stage ranking is intended to improve the relevance of the context supplied to the LLM.

---

### 5. Context Assembly

The highest-ranked document chunks are assembled into the context supplied to the generation stage.

The context originates from:

```text
docs/
```

The knowledge base is treated as the application's source of truth.

---

### 6. Ollama Generation

If the retrieval gate passes, the system sends the retrieved context to the local Ollama model:

```text
qwen2.5:3b
```

The LLM is therefore not called when the system determines that the knowledge base does not provide sufficient retrieval evidence.

---

### 7. Grounding Check

The generated response is checked against the retrieved documentation.

The system uses:

* Lexical checking for short answer units
* Embedding similarity for prose
* A minimum embedding similarity threshold of `0.42`

Ungrounded content can be removed.

If the generated answer is completely unsupported, the system rejects it rather than returning an ungrounded response.

---

### 8. Confidence Scoring

The final response receives a confidence score:

```text
0 – 100
```

with one of three labels:

```text
HIGH
MEDIUM
LOW
```

The confidence information is exposed through the Streamlit interface.

---

# Guardrails

The system contains multiple controls designed to prevent unsupported generation.

## Prompt-Injection Blocking

Patterns such as:

```text
ignore previous instructions
```

are detected during input validation and blocked.

---

## Network-Domain Gate

The current knowledge base focuses on network and IT support.

Questions without relevant network-support terminology can be refused before retrieval.

For example:

```text
"My computer is slow"
```

may be rejected because the current knowledge base does not cover general computer-performance troubleshooting.

---

## Retrieval Gate

The system checks the retrieval quality before calling the LLM.

If the best matching chunk has a similarity score below:

```text
0.40
```

the LLM is **not called**.

Instead, the system returns a knowledge-base refusal.

```text
I don't have information about this in the knowledge base.
```

This prevents the LLM from generating an answer when the retrieval evidence is insufficient.

---

## Grounding Gate

Generated answer units are compared against the retrieved documentation.

For prose, the system uses an embedding similarity threshold of:

```text
0.42
```

Unsupported content can be trimmed, and completely ungrounded responses are rejected.

---

## Cautious-Refusal Retry

If relevant documentation was retrieved but the model refuses to answer, the system can retry once using a more explicit prompt requesting an answer from the supplied evidence.

---

# Knowledge Base

The application's knowledge source is the:

```text
docs/
```

directory.

The current knowledge base contains **10 documents** and produces **47 indexed chunks** in the documented test configuration.

Example document categories include:

* Network troubleshooting fundamentals
* DNS troubleshooting
* DHCP troubleshooting
* TCP/IP connectivity
* Wi-Fi troubleshooting
* VPN and remote access
* Firewall and ports
* Routing and switching
* Network performance and packet loss
* Network incident response

---

# Project Structure

```text
RAG_IT_Support_System/
│
├── app.py
│   └── Streamlit UI
│
├── ingest.py
│   └── Builds / rebuilds the ChromaDB index
│
├── rag_engine.py
│   └── RAG pipeline, retrieval, re-ranking and guardrails
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
├── Readme.docx
│   └── Documentation source
│
├── .gitignore
│
├── docs/
│   └── Knowledge-base input documents
│
├── chroma_db/
│   └── Generated vector database
│
└── rag_errors.log
    └── Runtime warnings / errors
```

> `chroma_db/` and `rag_errors.log` are generated/runtime artifacts and are ignored by Git.

---

# Technology Stack

| Component         | Technology             |
| ----------------- | ---------------------- |
| Language          | Python 3.12            |
| UI                | Streamlit              |
| RAG Framework     | LangChain              |
| Vector Database   | ChromaDB               |
| Local LLM Runtime | Ollama                 |
| Generation Model  | qwen2.5:3b             |
| Embedding Model   | all-MiniLM-L6-v2       |
| Re-ranker         | ms-marco-MiniLM-L-6-v2 |
| Retrieval         | Semantic Vector Search |
| Generation        | Local LLM              |

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/Balaram-R/RAG_IT_Support_System.git
cd RAG_IT_Support_System
```

---

## 2. Create a Virtual Environment

Windows:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\activate
```

---

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Install Ollama

Install Ollama and download the required model:

```powershell
ollama pull qwen2.5:3b
```

Make sure Ollama is running.

The application connects to:

```text
http://127.0.0.1:11434
```

---

## 5. Add Knowledge Documents

Place the supported knowledge-base documents inside:

```text
docs/
```

These documents become the source material used by the RAG system.

---

## 6. Build the Vector Database

Run:

```powershell
python ingest.py
```

This creates/rebuilds the local ChromaDB index from the documents in `docs/`.

---

## 7. Launch the Application

Run:

```powershell
streamlit run app.py
```

The Streamlit interface will open in your browser.

---

# Example Flow

A supported question might look like:

```text
Why can a DNS lookup fail even when the network connection is working?
```

The system then performs:

```text
Question
   ↓
Validation
   ↓
Embedding
   ↓
ChromaDB Retrieval
   ↓
Cross-Encoder Re-ranking
   ↓
Context Assembly
   ↓
Retrieval Gate
   ↓
Ollama
   ↓
Grounding Check
   ↓
Confidence Score
   ↓
Answer + Sources
```

If the knowledge base does not provide sufficient evidence, the system refuses instead of relying on the model's general training knowledge.

---

# Design Principle

The central design decision of this project is:

> **The LLM should not be allowed to answer simply because it can.**

Generation is conditional on retrieval quality, and generated content is subsequently checked against the retrieved evidence.

This makes the system more controlled than a simple:

```text
Question → LLM → Answer
```

architecture.

Instead:

```text
Question
   ↓
Validate
   ↓
Retrieve
   ↓
Re-rank
   ↓
Gate
   ↓
Generate
   ↓
Verify
   ↓
Score
   ↓
Answer
```

---

# Current Scope

The current knowledge base is focused on **IT and network troubleshooting**.

The system is therefore intentionally designed to refuse questions outside the available documentation rather than attempting to act as a general-purpose assistant.

---

# Author

**Balaram R**

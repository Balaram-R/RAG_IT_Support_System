# ingest.py
# Run this once whenever you add new documents:
#     .venv\Scripts\python.exe ingest.py
#
# It REBUILDS the knowledge base from scratch (no duplicates).
# The chroma_db folder is wiped and recreated with fresh chunks.

import os
import shutil
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# ── Settings (keep in sync with rag_engine.py) ─────────────────
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs offline
CHROMA_DIR = "./chroma_db"
DOCS_DIR = "./docs"

# Chunking: 800 chars with 120 overlap keeps paragraphs and
# numbered troubleshooting steps intact (500 chars cut mid-list).
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# ── STEP 0: Wipe old database so we never get duplicate chunks ─
if os.path.exists(CHROMA_DIR):
    print(f"Removing old database at {CHROMA_DIR} ...")
    shutil.rmtree(CHROMA_DIR)

# ── STEP 1: Load all .txt / .md files from the docs folder ─────
print("Loading documents...")
documents = []
for ext in ("*.txt", "*.md"):
    loader = DirectoryLoader(
        DOCS_DIR,
        glob=f"**/{ext}",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True,
    )
    documents.extend(loader.load())
if not documents:
    print(f"No documents found in {DOCS_DIR}. Add .txt or .md files and re-run.")
    raise SystemExit(1)
print(f"Loaded {len(documents)} documents")

# ── STEP 2: Split into chunks ──────────────────────────────────
# Separators respect paragraph and sentence boundaries so each
# chunk is a coherent block of instructions, not a cut-off list.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)
chunks = splitter.split_documents(documents)

# Add useful metadata to every chunk (source file + position)
for i, chunk in enumerate(chunks):
    chunk.metadata["chunk_index"] = i
    chunk.metadata["total_chunks"] = len(chunks)

print(f"Created {len(chunks)} chunks")

# ── STEP 3: Convert chunks to vectors and store ────────────────
print("Building vector database...")
embedding_model = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_DIR,
)

print("Done. Knowledge base ready.")
print(f"Stored {len(chunks)} searchable chunks in {CHROMA_DIR}/")
print(f"Embedding model: {EMBEDDING_MODEL}")
# ingest.py
# Run this once whenever you add new documents
# It builds the searchable knowledge base

import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma

# ── STEP 1: Load all .txt files from the docs folder ──────────
print("Loading documents...")
loader = DirectoryLoader(
    "./docs",                    # folder to read from
    glob="**/*.txt",             # only read .txt files
    loader_cls=TextLoader        # how to read them
)
documents = loader.load()
print(f"Loaded {len(documents)} documents")

# ── STEP 2: Split into chunks ──────────────────────────────────
# Why? LLMs have a limit on how much text they can read at once.
# Small chunks = more precise retrieval.
# 500 characters per chunk, 50 overlap so meaning is not lost at edges.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)
print(f"Created {len(chunks)} chunks")

# ── STEP 3: Convert chunks to vectors and store ───────────────
# SentenceTransformer converts text meaning into numbers
# Similar meaning = similar numbers = found together in search
print("Building vector database...")
embedding_model = SentenceTransformerEmbeddings(
    model_name="all-MiniLM-L6-v2"   # small, fast, runs offline
)

# Chroma saves these vectors to a folder called chroma_db
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="./chroma_db"  # saved here on your disk
)

print("Done. Knowledge base ready.")
print(f"Stored {len(chunks)} searchable chunks in chroma_db/")
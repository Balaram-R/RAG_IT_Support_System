IT Support Knowledge Assistant

A local Retrieval-Augmented Generation (RAG) application that helps users troubleshoot IT and networking issues using a custom knowledge base.
The application combines semantic search with a local Large Language Model (LLM) to provide accurate answers based on technical documentation, troubleshooting guides, and support knowledge bases.
Features
•	Local LLM powered by Ollama and Llama 3.2
•	ChromaDB vector database for semantic search
•	Retrieval-Augmented Generation (RAG)
•	Streamlit web interface
•	Offline operation after setup
•	Fast document retrieval
•	Custom knowledge base support
•	No external API costs

Use Cases
•	Network troubleshooting
•	DNS issues
•	Wi-Fi connectivity problems
•	VPN troubleshooting
•	Windows administration
•	Active Directory basics
•	Helpdesk knowledge retrieval
•	Internal IT documentation search

Technology Stack
•	Python
•	Streamlit
•	LangChain
•	ChromaDB
•	Ollama
•	Llama 3.2
•	Sentence Transformers
## Project Structure

```text
IT-Support-Knowledge-Assistant/
├── app.py
├── ingest.py
├── rag_engine.py
├── requirements.txt
├── README.md
├── .gitignore
├── data/
│   ├── troubleshooting-guides/
│   ├── networking/
│   └── documentation/
├── chroma_db/
└── assets/
```
Installation
Create Virtual Environment
python -m venv venv

Activate Virtual Environment
Windows:
venv\Scripts\activate

Install Dependencies
pip install -r requirements.txt

Install Ollama Model
ollama pull llama3.2

Add Documents
Place your documentation, troubleshooting guides, and knowledge base files into the data folder.

Build Vector Database
python ingest.py

Launch Application
streamlit run app.py

## Project Structure

```text
Architecture
Documents
    │
    ▼
Document Loader
    │
    ▼
Text Chunking
    │
    ▼
Embeddings
    │
    ▼
ChromaDB
    │
    ▼
Context Retrieval
    │
    ▼
Llama 3.2 (Ollama)
    │
    ▼
Generated Answer
```

Author: <br> Balaram R

<div align="center">

# 🧠 SmartAI Notes
### RAG-powered Document Intelligence — Ask anything about your files

[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-7c3aed?style=for-the-badge)](https://trychroma.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20Gateway-0f172a?style=for-the-badge)](https://openrouter.ai)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> *Upload your PDFs, docs, and notes — then have a conversation with them using state-of-the-art LLMs.*

</div>

---

## 📖 Overview

**SmartAI Notes** is a full-stack **Retrieval-Augmented Generation (RAG)** application that lets you upload documents in multiple formats and ask natural language questions about them. The system retrieves the most semantically relevant passages using **ChromaDB** vector search and generates accurate, cited answers using **LLMs via OpenRouter** (Llama, Mistral, Gemma, DeepSeek, and more).

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        RAG Pipeline                              │
│                                                                  │
│  📄 Document Upload                                              │
│       │                                                          │
│       ▼                                                          │
│  📝 Document Processor   ←── PDF / TXT / MD / DOCX / CSV / JSON │
│       │  (parse + chunk)                                         │
│       ▼                                                          │
│  🔢 SentenceTransformer  ←── all-MiniLM-L6-v2 (local, free)    │
│       │  (embed chunks)                                          │
│       ▼                                                          │
│  🗄️  ChromaDB            ←── Persistent vector store            │
│       │  (store embeddings)                                      │
│                                                                  │
│  💬 User Question                                                │
│       │                                                          │
│       ▼                                                          │
│  🔢 SentenceTransformer  (embed question)                        │
│       │                                                          │
│       ▼                                                          │
│  🗄️  ChromaDB Query      (cosine similarity → top-K chunks)     │
│       │                                                          │
│       ▼                                                          │
│  🤖 OpenRouter LLM       ←── Llama / Mistral / Gemma / DeepSeek │
│       │  (RAG prompt + context → answer)                        │
│       ▼                                                          │
│  📤 Cited Answer + Sources                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 **Multi-format** | PDF, TXT, Markdown, DOCX, CSV, JSON |
| 🔍 **Semantic Search** | ChromaDB + SentenceTransformer embeddings (local, no API cost) |
| 🤖 **Multiple LLMs** | Llama 3.3, Mistral 7B, Gemma 3, DeepSeek R1, Qwen 2.5 — all free via OpenRouter |
| 💬 **Chat History** | Multi-turn conversations with memory |
| ⚡ **Streaming** | Token-by-token streaming responses |
| 📚 **Source Citations** | Every answer cites the source document and page |
| 🗂️ **Document Management** | Upload, list, filter, and delete indexed documents |
| 🎨 **Beautiful UI** | Dark-mode Streamlit interface with chat bubbles |
| 🐳 **Docker Ready** | One-command startup with Docker Compose |
| 🔧 **FastAPI Backend** | Full REST API with Swagger docs at `/docs` |

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Frontend** | Streamlit | Interactive chat UI |
| **Backend** | FastAPI | REST API server |
| **Vector DB** | ChromaDB (persistent) | Semantic document storage |
| **Embeddings** | SentenceTransformer `all-MiniLM-L6-v2` | Local text embeddings |
| **LLM Gateway** | OpenRouter API | Access to 50+ LLMs |
| **LLMs** | Llama 3.3, Mistral, Gemma, DeepSeek | Answer generation |
| **Doc Parsing** | pypdf, python-docx | Multi-format document parsing |
| **Containerization** | Docker + Docker Compose | Easy deployment |

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
# Clone the repo
git clone https://github.com/Devendra-Pudi/SmartAI_Notes.git
cd SmartAI_Notes

# Set your OpenRouter API key
cp .env.example .env
# Edit .env → add your OPENROUTER_API_KEY

# Start everything
docker compose up --build

# Open in browser
# Frontend: http://localhost:8501
# API Docs:  http://localhost:8000/docs
```

### Option 2 — Manual Setup

**1. Backend**
```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp ../.env.example .env
# Edit .env → add OPENROUTER_API_KEY

# Start FastAPI server
uvicorn main:app --reload --port 8000
```

**2. Frontend** (new terminal)
```bash
cd frontend
pip install -r requirements.txt
streamlit run app.py
```

**3. Open your browser**
- 🎨 **Streamlit App**: http://localhost:8501
- 📖 **API Docs**: http://localhost:8000/docs

---

## 🔑 Getting Your OpenRouter API Key

1. Go to **[openrouter.ai](https://openrouter.ai)**
2. Sign up (free)
3. Navigate to **Keys** → Create new key
4. Copy and paste into `.env` or the Streamlit sidebar

> 💡 All models marked `(Free)` work with **$0 credits**. No billing required.

---

## 📡 API Reference

### Upload a Document
```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@your_document.pdf"
```

### Ask a Question
```bash
curl -X POST http://localhost:8000/api/v1/query/ \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the key findings?",
    "top_k": 5,
    "model": "meta-llama/llama-3.3-8b-instruct:free"
  }'
```

### List Documents
```bash
curl http://localhost:8000/api/v1/documents/
```

### Delete a Document
```bash
curl -X DELETE http://localhost:8000/api/v1/documents/{file_id}
```

Full interactive API docs: **http://localhost:8000/docs**

---

## 📁 Project Structure

```
SmartAI_Notes/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   ├── upload.py       # File upload endpoints
│   │   │   ├── query.py        # RAG query endpoints
│   │   │   └── health.py       # Health check
│   │   ├── core/
│   │   │   ├── config.py       # Settings & environment
│   │   │   └── logger.py       # Logging setup
│   │   ├── models/
│   │   │   └── schemas.py      # Pydantic schemas
│   │   └── services/
│   │       ├── document_processor.py  # Parse & chunk documents
│   │       ├── vector_store.py        # ChromaDB operations
│   │       ├── llm_service.py         # OpenRouter LLM calls
│   │       └── rag_pipeline.py        # Orchestrates RAG flow
│   ├── main.py                 # FastAPI app entry point
│   ├── requirements.txt
│   └── Dockerfile
├── frontend/
│   ├── app.py                  # Streamlit UI
│   ├── requirements.txt
│   └── Dockerfile
├── tests/
│   └── test_rag.py             # Unit tests
├── docker-compose.yml
├── .env.example
├── .gitignore
└── README.md
```

---

## 🤖 Available LLM Models (via OpenRouter)

| Model | Provider | Context | Free |
|-------|---------|---------|------|
| Llama 3.3 8B Instruct | Meta | 131K | ✅ |
| Llama 3.1 70B Instruct | Meta | 131K | ✅ |
| Mistral 7B Instruct | Mistral AI | 32K | ✅ |
| Gemma 3 4B Instruct | Google | 131K | ✅ |
| DeepSeek R1 | DeepSeek | 163K | ✅ |
| Qwen 2.5 72B Instruct | Alibaba | 131K | ✅ |
| GPT-4o Mini | OpenAI | 128K | 💳 |
| Claude 3 Haiku | Anthropic | 200K | 💳 |

---

## 🧪 Running Tests

```bash
cd backend
pip install pytest
pytest ../tests/ -v
```

---

## 📄 License

Licensed under the **MIT License** — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🧠 by [Devendra Prasad Pudi](https://github.com/Devendra-Pudi)

⭐ **Star this repo if you found it useful!**

</div>

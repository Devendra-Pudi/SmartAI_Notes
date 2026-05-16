---
title: SmartAI Notes
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: true
license: mit
short_description: RAG-powered Document Q&A — Upload files, ask questions, get cited answers
---

<div align="center">

# 🧠 SmartAI Notes
### RAG-powered Document Intelligence — Ask anything about your files

[![Gradio](https://img.shields.io/badge/Gradio-5.x-FF4B4B?style=for-the-badge&logo=gradio&logoColor=white)](https://gradio.app)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector%20Store-7c3aed?style=for-the-badge)](https://trychroma.com)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-LLM%20Gateway-0f172a?style=for-the-badge)](https://openrouter.ai)
[![HuggingFace](https://img.shields.io/badge/🤗%20Spaces-Deployed-yellow?style=for-the-badge)](https://huggingface.co/spaces/Devendra-Pudi/SmartAI-Notes)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

> *Upload your PDFs, docs, and notes — then have a conversation with them using state-of-the-art LLMs. Entirely free.*

**[🚀 Try it Live on HuggingFace Spaces](https://huggingface.co/spaces/Devendra-Pudi/SmartAI-Notes)**

</div>

---

## 📖 Overview

**SmartAI Notes** is a full-stack **Retrieval-Augmented Generation (RAG)** application deployed on **HuggingFace Spaces**. Upload documents in multiple formats and have a natural language conversation with them. The system finds the most semantically relevant passages using **ChromaDB** vector search and generates accurate, cited answers via **OpenRouter LLMs** (Llama, Mistral, Gemma, DeepSeek — all free).

The entire stack — Gradio UI + FastAPI backend + ChromaDB + embeddings — runs in a **single HuggingFace Space** at zero cost.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  HuggingFace Space (Single Process)                  │
│                                                                      │
│   ┌─────────────────┐    internal    ┌──────────────────────────┐   │
│   │   Gradio UI     │ ─────calls──►  │     FastAPI Backend       │   │
│   │  (port 7860)    │                │   /api/* endpoints        │   │
│   │                 │                │                           │   │
│   │  💬 Chat tab    │                │  ┌────────────────────┐   │   │
│   │  📄 Docs tab    │                │  │  DocumentProcessor │   │   │
│   │  🔌 API tab     │                │  │  (PDF/TXT/DOCX...) │   │   │
│   └─────────────────┘                │  └────────┬───────────┘   │   │
│                                      │           │               │   │
│                                      │  ┌────────▼───────────┐   │   │
│                                      │  │  SentenceTransfor  │   │   │
│                                      │  │  mer Embeddings    │   │   │
│                                      │  │  (local, free)     │   │   │
│                                      │  └────────┬───────────┘   │   │
│                                      │           │               │   │
│                                      │  ┌────────▼───────────┐   │   │
│                                      │  │   ChromaDB         │   │   │
│                                      │  │   (persistent disk)│   │   │
│                                      │  └────────────────────┘   │   │
│                                      └──────────────────────────┘   │
│                                                   │                  │
│                                          OpenRouter API              │
│                                     (Llama / Mistral / Gemma...)     │
└─────────────────────────────────────────────────────────────────────┘

                    Optional: HF Dataset repo for cross-restart
                              ChromaDB persistence
```

---

## ✨ Features

| Feature | Details |
|---------|---------|
| 📄 **Multi-format** | PDF, TXT, Markdown, DOCX, CSV, JSON |
| 🔍 **Semantic Search** | ChromaDB + local SentenceTransformer (no API cost) |
| 🤖 **6 Free LLMs** | Llama 3.3, Mistral 7B, Gemma 3, DeepSeek R1, Qwen 2.5 via OpenRouter |
| 💬 **Streaming** | Real-time token-by-token response rendering |
| 📚 **Source Citations** | Every answer cites document + page number |
| 🗂️ **Document Management** | Upload, list, filter by document, delete |
| 💾 **Persistence** | Optional HF Dataset backup for cross-restart ChromaDB survival |
| 🔌 **REST API** | Full FastAPI backend with Swagger UI at `/api/docs` |
| 🐳 **Local Dev** | Run locally with a single command |
| 🎨 **Dark UI** | Beautiful dark-mode Gradio interface |

---

## 🚀 Deploy to HuggingFace Spaces

### Option A — Fork this Space (Easiest)
1. Go to this Space on HuggingFace
2. Click **"Duplicate this Space"**
3. Add your `OPENROUTER_API_KEY` in Space **Settings → Secrets**
4. Done! 🎉

### Option B — Deploy from GitHub

```bash
# 1. Clone this repo
git clone https://github.com/Devendra-Pudi/SmartAI_Notes.git
cd SmartAI_Notes

# 2. Create a new HuggingFace Space
#    → huggingface.co/new-space
#    → SDK: Gradio
#    → Name: SmartAI-Notes

# 3. Add HF remote
git remote add hf https://huggingface.co/spaces/Devendra-Pudi/SmartAI-Notes

# 4. Push
git push hf main
```

**5. Add secrets in Space Settings:**

| Secret Key | Value |
|------------|-------|
| `OPENROUTER_API_KEY` | `sk-or-v1-...` (from openrouter.ai) |
| `HF_TOKEN` | *(optional)* HF write token for persistence |
| `HF_DATASET_REPO` | *(optional)* `your-username/smartai-db` |
| `USE_HF_PERSISTENCE` | `true` *(optional)* |

---

## 💾 ChromaDB Persistence (Cross-Restart)

HF Spaces free tier uses **ephemeral storage** — data is lost on restart. SmartAI Notes solves this with optional **HF Dataset backup**:

```
On startup  → Pull chroma_db.tar.gz from your HF Dataset repo
On shutdown → Push chroma_db.tar.gz back to your HF Dataset repo
```

**Setup:**
1. Create a **private** HF Dataset repo: `your-username/smartai-notes-db`
2. Generate an HF token with **write** access at `huggingface.co/settings/tokens`
3. Add `HF_TOKEN`, `HF_DATASET_REPO`, `USE_HF_PERSISTENCE=true` to Space secrets

Without this, documents must be re-uploaded after each Space restart.

---

## 🖥️ Run Locally

```bash
git clone https://github.com/Devendra-Pudi/SmartAI_Notes.git
cd SmartAI_Notes

# Install dependencies
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env → add OPENROUTER_API_KEY

# Launch (Gradio + FastAPI together on port 7860)
python app.py

# Open: http://localhost:7860
# API:  http://localhost:7860/api/docs
```

---

## 🔑 Getting Your OpenRouter API Key

1. Visit **[openrouter.ai](https://openrouter.ai)**
2. Sign up — no credit card needed
3. Go to **Keys** → **Create key**
4. Paste it into the **🔑 API Key** field in the app sidebar

> 💡 All models marked **(Free)** work with **$0 balance**. You get intelligent answers at zero cost.

---

## 🤖 Available LLM Models

| Model | Provider | Context | Cost |
|-------|---------|---------|------|
| Llama 3.3 8B Instruct | Meta | 131K | ✅ Free |
| Llama 3.1 70B Instruct | Meta | 131K | ✅ Free |
| Mistral 7B Instruct | Mistral AI | 32K | ✅ Free |
| Gemma 3 4B Instruct | Google | 131K | ✅ Free |
| DeepSeek R1 | DeepSeek | 163K | ✅ Free |
| Qwen 2.5 72B Instruct | Alibaba | 131K | ✅ Free |
| GPT-4o Mini | OpenAI | 128K | 💳 Paid |
| Claude 3 Haiku | Anthropic | 200K | 💳 Paid |

---

## 📡 API Reference

The FastAPI backend is fully accessible:

```bash
# Upload document
curl -X POST https://your-space.hf.space/api/documents/upload \
  -F "file=@report.pdf"

# Ask a question
curl -X POST https://your-space.hf.space/api/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key findings?", "top_k": 5}'

# List documents
curl https://your-space.hf.space/api/documents

# Health check
curl https://your-space.hf.space/api/health

# Swagger UI
open https://your-space.hf.space/api/docs
```

---

## 📁 Project Structure

```
SmartAI_Notes/
├── app.py                          # 🚀 Entry point — Gradio + FastAPI mounted together
├── requirements.txt                # All dependencies
├── .env.example                    # Environment variable template
├── README.md                       # This file (also HF Space card)
│
├── app/
│   ├── core/
│   │   ├── config.py               # Settings via env vars / HF Secrets
│   │   ├── logger.py               # Structured logging
│   │   └── persistence.py          # HF Dataset ChromaDB backup
│   ├── models/
│   │   └── schemas.py              # Pydantic request/response models
│   ├── services/
│   │   ├── document_processor.py   # Parse & chunk PDF/TXT/DOCX/CSV/JSON
│   │   ├── vector_store.py         # ChromaDB CRUD + semantic search
│   │   ├── llm_service.py          # OpenRouter API + streaming
│   │   └── rag_pipeline.py         # Orchestrate full RAG flow
│   └── api/
│       └── routes.py               # All FastAPI endpoints
│
└── tests/
    └── test_rag.py                 # Unit tests
```

---

## 🧪 Running Tests

```bash
pip install pytest
pytest tests/ -v
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">

Built with 🧠 by [Devendra Prasad Pudi](https://github.com/Devendra-Pudi)

⭐ **Star this repo if you found it useful!**

**[🚀 Try it Live](https://huggingface.co/spaces/Devendra-Pudi/SmartAI-Notes)** · **[📖 GitHub](https://github.com/Devendra-Pudi/SmartAI_Notes)**

</div>

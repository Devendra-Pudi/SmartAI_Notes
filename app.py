"""
SmartAI Notes — HuggingFace Spaces Entry Point
Gradio UI + FastAPI backend mounted in a single process.

Architecture:
  - FastAPI handles all RAG logic (upload, embed, query)
  - Gradio provides the chat + upload UI
  - gr.mount_gradio_app() serves both on port 7860
"""
import os
import sys
import asyncio
import threading
import tempfile
import json
import time
from pathlib import Path
from typing import List, Optional, Tuple

# ── Ensure local app package is importable ────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import gradio as gr
import httpx
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.core.logger import get_logger
from app.core.persistence import pull_db_from_hf, push_db_to_hf
from app.api.routes import router as api_router

settings = get_settings()
logger = get_logger("app")

# ── Ensure data directories exist ─────────────────────────────────────────────
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

# ── Restore ChromaDB from HF Dataset on startup ───────────────────────────────
pull_db_from_hf()

# ── FastAPI app ───────────────────────────────────────────────────────────────
fastapi_app = FastAPI(
    title="SmartAI Notes API",
    version=settings.APP_VERSION,
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)
fastapi_app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
fastapi_app.include_router(api_router, prefix="/api")


# ── Internal API client (calls FastAPI from within the same process) ──────────
API_BASE = "http://127.0.0.1:7860/api"


def sync_post(endpoint: str, **kwargs):
    try:
        with httpx.Client(timeout=120) as client:
            return client.post(f"{API_BASE}{endpoint}", **kwargs)
    except Exception as e:
        return None


def sync_get(endpoint: str, **kwargs):
    try:
        with httpx.Client(timeout=10) as client:
            return client.get(f"{API_BASE}{endpoint}", **kwargs)
    except Exception:
        return None


def sync_delete(endpoint: str):
    try:
        with httpx.Client(timeout=10) as client:
            return client.delete(f"{API_BASE}{endpoint}")
    except Exception:
        return None


# ── File upload handler ───────────────────────────────────────────────────────
def handle_upload(files) -> str:
    if not files:
        return "⚠️ No files selected."

    results = []
    for file_path in files:
        filename = Path(file_path).name
        ext = Path(filename).suffix.lower()

        with open(file_path, "rb") as f:
            content = f.read()

        resp = sync_post(
            "/documents/upload",
            files={"file": (filename, content, "application/octet-stream")},
        )

        if resp and resp.status_code == 200:
            data = resp.json()
            results.append(f"✅ **{filename}** — {data['chunks_created']} chunks indexed")
        else:
            err = resp.json().get("detail", "Unknown error") if resp else "Connection error"
            results.append(f"❌ **{filename}** — {err}")

    # Push updated DB to HF for persistence
    push_db_to_hf()
    return "\n".join(results)


def get_documents() -> Tuple[str, List[dict]]:
    resp = sync_get("/documents")
    if resp and resp.status_code == 200:
        docs = resp.json().get("documents", [])
        if not docs:
            return "📭 No documents indexed yet.", []

        ICONS = {".pdf": "📕", ".txt": "📄", ".md": "📝", ".docx": "📘", ".csv": "📊", ".json": "🗂️"}
        lines = []
        for d in docs:
            icon = ICONS.get(d.get("file_type", ".txt"), "📄")
            lines.append(f"{icon} **{d['filename']}** — {d['chunks']} chunks | `{d['file_id'][:8]}...`")
        return "\n".join(lines), docs
    return "⚠️ Could not fetch documents.", []


def delete_all_docs() -> str:
    resp = sync_delete("/documents")
    push_db_to_hf()
    if resp and resp.status_code == 200:
        return "🗑️ All documents cleared."
    return "❌ Failed to clear documents."


def get_models() -> List[str]:
    resp = sync_get("/query/models")
    if resp and resp.status_code == 200:
        models = resp.json().get("models", [])
        return [m["name"] for m in models], {m["name"]: m["id"] for m in models}
    default_names = [
        "Llama 3.3 8B (Free)", "Llama 3.1 70B (Free)",
        "Mistral 7B (Free)", "Gemma 3 4B (Free)",
        "DeepSeek R1 (Free)", "Qwen 2.5 72B (Free)",
    ]
    default_ids = {
        "Llama 3.3 8B (Free)":  "meta-llama/llama-3.3-8b-instruct:free",
        "Llama 3.1 70B (Free)": "meta-llama/llama-3.1-70b-instruct:free",
        "Mistral 7B (Free)":    "mistralai/mistral-7b-instruct:free",
        "Gemma 3 4B (Free)":    "google/gemma-3-4b-it:free",
        "DeepSeek R1 (Free)":   "deepseek/deepseek-r1:free",
        "Qwen 2.5 72B (Free)":  "qwen/qwen-2.5-72b-instruct:free",
    }
    return default_names, default_ids


# ── Chat handler ──────────────────────────────────────────────────────────────
model_name_to_id = {}  # populated on build


def chat(
    message: str,
    history: List,
    model_name: str,
    top_k: int,
    openrouter_key: str,
    filter_docs: str,
) -> Tuple[List, str]:
    """
    Handle a chat message:
    - Build chat_history from Gradio history format
    - POST to /api/query
    - Stream response back
    """
    if not message.strip():
        return history, ""

    # Override API key if provided via UI
    if openrouter_key.strip():
        os.environ["OPENROUTER_API_KEY"] = openrouter_key.strip()
        # Reload settings singleton
        get_settings.cache_clear()

    model_id = model_name_to_id.get(model_name, settings.DEFAULT_LLM_MODEL)

    # Convert Gradio history → API format
    api_history = []
    for h in history[-6:]:
        if isinstance(h, (list, tuple)) and len(h) == 2:
            if h[0]:
                api_history.append({"role": "user", "content": h[0]})
            if h[1]:
                api_history.append({"role": "assistant", "content": h[1]})

    # Resolve file filter
    file_ids = None
    if filter_docs and filter_docs != "All Documents":
        _, docs = get_documents()
        file_ids = [d["file_id"] for d in docs if d["filename"] == filter_docs]

    payload = {
        "question": message,
        "model": model_id,
        "top_k": top_k,
        "chat_history": api_history,
        "file_ids": file_ids,
    }

    # Stream response
    partial_answer = ""
    history = history + [[message, ""]]

    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", f"{API_BASE}/query/stream", json=payload) as resp:
                for token in resp.iter_text():
                    partial_answer += token
                    history[-1][1] = partial_answer
                    # Gradio streaming: yield intermediate state
    except Exception as e:
        partial_answer = f"❌ Error: {str(e)}"
        history[-1][1] = partial_answer

    return history, ""


def chat_stream(
    message: str,
    history: List,
    model_name: str,
    top_k: int,
    openrouter_key: str,
    filter_docs: str,
):
    """Gradio streaming generator for smooth token-by-token display."""
    if not message.strip():
        yield history, ""
        return

    if openrouter_key.strip():
        os.environ["OPENROUTER_API_KEY"] = openrouter_key.strip()
        get_settings.cache_clear()

    model_id = model_name_to_id.get(model_name, settings.DEFAULT_LLM_MODEL)

    api_history = []
    for h in history[-6:]:
        if isinstance(h, (list, tuple)) and len(h) == 2:
            if h[0]:
                api_history.append({"role": "user", "content": h[0]})
            if h[1]:
                api_history.append({"role": "assistant", "content": h[1]})

    file_ids = None
    if filter_docs and filter_docs != "All Documents":
        _, docs = get_documents()
        file_ids = [d["file_id"] for d in docs if d["filename"] == filter_docs]

    payload = {
        "question": message,
        "model": model_id,
        "top_k": top_k,
        "chat_history": api_history,
        "file_ids": file_ids,
    }

    history = history + [[message, "⏳ Thinking..."]]
    yield history, ""

    partial = ""
    try:
        with httpx.Client(timeout=120) as client:
            with client.stream("POST", f"{API_BASE}/query/stream", json=payload) as resp:
                for token in resp.iter_text():
                    if token:
                        partial += token
                        history[-1][1] = partial
                        yield history, ""
    except Exception as e:
        history[-1][1] = f"❌ Error: {str(e)}"
        yield history, ""


# ── Gradio UI ─────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
).set(
    body_background_fill="#0f1117",
    body_background_fill_dark="#0f1117",
    block_background_fill="#1a1f2e",
    block_background_fill_dark="#1a1f2e",
    block_border_color="#2d3748",
    block_border_color_dark="#2d3748",
    block_title_text_color="#f0f6fc",
    block_title_text_color_dark="#f0f6fc",
    body_text_color="#c9d1d9",
    body_text_color_dark="#c9d1d9",
    button_primary_background_fill="linear-gradient(135deg, #1d4ed8, #2563eb)",
    button_primary_background_fill_dark="linear-gradient(135deg, #1d4ed8, #2563eb)",
    button_primary_text_color="white",
    input_background_fill="#21262d",
    input_background_fill_dark="#21262d",
    input_border_color="#30363d",
)

CSS = """
/* Overall */
.gradio-container { max-width: 1200px !important; margin: 0 auto !important; }

/* Header */
.header-box {
    background: linear-gradient(135deg, #0f2027, #1a3a5c, #0f2027);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    text-align: center;
    border: 1px solid #2d4a6e;
}
.header-title {
    font-size: 2.2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}
.header-sub {
    color: #64748b;
    font-size: 0.95rem;
    margin-top: 6px;
}

/* Chatbot */
.chatbot-box { border-radius: 12px !important; border: 1px solid #2d3748 !important; }
.message.user { background: linear-gradient(135deg, #1d4ed8, #2563eb) !important; border-radius: 18px 18px 4px 18px !important; }
.message.bot  { background: #1a2535 !important; border: 1px solid #2d4a6e !important; border-radius: 18px 18px 18px 4px !important; }

/* Source accordion */
.source-panel { background: #1a2535 !important; border: 1px solid #2d4a6e !important; border-radius: 8px !important; }

/* Tab styling */
.tab-nav button { color: #94a3b8 !important; font-weight: 600 !important; }
.tab-nav button.selected { color: #38bdf8 !important; border-bottom: 2px solid #38bdf8 !important; }

/* Upload area */
.upload-box { border: 2px dashed #2d4a6e !important; border-radius: 12px !important; background: #1a2535 !important; }

/* Stats */
.stat-card {
    background: #1a2535;
    border: 1px solid #2d3748;
    border-radius: 10px;
    padding: 14px;
    text-align: center;
}
.stat-num { font-size: 1.8rem; font-weight: 700; }
.stat-label { color: #64748b; font-size: 0.78rem; margin-top: 2px; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #1a1f2e; }
::-webkit-scrollbar-thumb { background: #2d3748; border-radius: 3px; }
"""


def build_ui():
    global model_name_to_id

    model_names, model_name_to_id = get_models()
    _, initial_docs = get_documents()
    doc_names = ["All Documents"] + [d["filename"] for d in initial_docs]

    with gr.Blocks(theme=THEME, css=CSS, title="SmartAI Notes") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-box">
            <div class="header-title">🧠 SmartAI Notes</div>
            <div class="header-sub">
                RAG-powered Document Intelligence &nbsp;·&nbsp;
                Upload files &nbsp;·&nbsp; Ask questions &nbsp;·&nbsp; Get cited answers
            </div>
        </div>
        """)

        with gr.Tabs() as tabs:

            # ════════════════════════════════════════════════════════════════
            # TAB 1 — Chat
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("💬 Chat", id="chat"):
                with gr.Row():
                    # Left — Chat
                    with gr.Column(scale=3):
                        chatbot = gr.Chatbot(
                            label="SmartAI Notes",
                            height=520,
                            show_copy_button=True,
                            bubble_full_width=False,
                            render_markdown=True,
                            elem_classes=["chatbot-box"],
                            placeholder=(
                                "### 👋 Welcome to SmartAI Notes!\n\n"
                                "Upload your documents in the **📄 Documents** tab, "
                                "then come back here to ask questions.\n\n"
                                "I'll find the most relevant passages and generate "
                                "accurate, cited answers for you."
                            ),
                        )

                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask a question about your documents...",
                                show_label=False,
                                scale=5,
                                lines=1,
                                max_lines=4,
                                container=False,
                            )
                            send_btn = gr.Button("🚀 Ask", variant="primary", scale=1, min_width=80)

                        # Quick prompts
                        gr.Markdown("**💡 Quick prompts:**")
                        with gr.Row():
                            q1 = gr.Button("📋 Summarize key points", size="sm")
                            q2 = gr.Button("🔍 Main topics covered", size="sm")
                        with gr.Row():
                            q3 = gr.Button("📅 Important dates/numbers", size="sm")
                            q4 = gr.Button("❓ What conclusions were drawn?", size="sm")

                        clear_btn = gr.Button("🧹 Clear Chat", size="sm", variant="secondary")

                    # Right — Settings
                    with gr.Column(scale=1, min_width=220):
                        gr.Markdown("### ⚙️ Settings")

                        openrouter_key = gr.Textbox(
                            label="🔑 OpenRouter API Key",
                            placeholder="sk-or-v1-...",
                            type="password",
                            info="Get free key at openrouter.ai",
                            value=os.environ.get("OPENROUTER_API_KEY", ""),
                        )

                        model_dropdown = gr.Dropdown(
                            choices=model_names,
                            value=model_names[0] if model_names else "Llama 3.3 8B (Free)",
                            label="🤖 LLM Model",
                            info="All (Free) models cost $0",
                        )

                        top_k_slider = gr.Slider(
                            minimum=1, maximum=15, value=5, step=1,
                            label="🔍 Top-K Chunks",
                            info="More chunks = more context",
                        )

                        filter_docs = gr.Dropdown(
                            choices=doc_names,
                            value="All Documents",
                            label="📂 Search Scope",
                            info="Filter to specific document",
                        )

                        gr.Markdown("---")
                        gr.Markdown("### 📊 Session Info")
                        health_display = gr.Markdown("*Loading...*")

            # ════════════════════════════════════════════════════════════════
            # TAB 2 — Documents
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("📄 Documents", id="docs"):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📤 Upload Documents")
                        gr.Markdown(
                            "Supported formats: **PDF · TXT · Markdown · DOCX · CSV · JSON**  \n"
                            "Max file size: **50 MB per file**"
                        )
                        file_uploader = gr.File(
                            label="Drop files here or click to browse",
                            file_count="multiple",
                            file_types=[".pdf", ".txt", ".md", ".docx", ".csv", ".json"],
                            elem_classes=["upload-box"],
                            height=180,
                        )
                        upload_btn = gr.Button("📥 Index Documents", variant="primary", size="lg")
                        upload_status = gr.Markdown("")

                    with gr.Column(scale=2):
                        gr.Markdown("### 🗂️ Indexed Documents")
                        refresh_btn = gr.Button("🔄 Refresh List", size="sm")
                        docs_display = gr.Markdown("*Loading...*")
                        gr.Markdown("---")
                        clear_all_btn = gr.Button("🗑️ Clear All Documents", variant="stop", size="sm")
                        clear_status = gr.Markdown("")

                with gr.Accordion("ℹ️ How RAG Works", open=False):
                    gr.Markdown("""
```
📄 You upload a file
        │
        ▼
📝 DocumentProcessor parses it (PDF → text, DOCX → paragraphs, etc.)
        │
        ▼
✂️  Text is split into overlapping chunks (~800 chars each)
        │
        ▼
🔢 SentenceTransformer embeds each chunk into a vector (384 dimensions)
        │
        ▼
🗄️  Vectors stored in ChromaDB (persistent on disk)

💬 You ask a question
        │
        ▼
🔢 Question is embedded into a vector
        │
        ▼
🧲 ChromaDB finds Top-K most similar chunks (cosine similarity)
        │
        ▼
📝 Chunks + question → structured prompt
        │
        ▼
🤖 OpenRouter LLM generates an answer with citations
        │
        ▼
💡 You get an accurate, cited answer!
```
                    """)

            # ════════════════════════════════════════════════════════════════
            # TAB 3 — API Explorer
            # ════════════════════════════════════════════════════════════════
            with gr.TabItem("🔌 API", id="api"):
                gr.Markdown("### 🔌 REST API — Direct Access")
                gr.Markdown(
                    "The full FastAPI backend is available at `/api/`. "
                    "Visit [/api/docs](/api/docs) for the interactive Swagger UI."
                )
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("#### Upload a file")
                        gr.Code(
                            value='curl -X POST https://your-space.hf.space/api/documents/upload \\\n  -F "file=@document.pdf"',
                            language="bash",
                        )
                        gr.Markdown("#### Ask a question")
                        gr.Code(
                            value='curl -X POST https://your-space.hf.space/api/query \\\n  -H "Content-Type: application/json" \\\n  -d \'{"question": "What are the key findings?", "top_k": 5}\'',
                            language="bash",
                        )
                    with gr.Column():
                        gr.Markdown("#### List documents")
                        gr.Code(
                            value="curl https://your-space.hf.space/api/documents",
                            language="bash",
                        )
                        gr.Markdown("#### Health check")
                        gr.Code(
                            value="curl https://your-space.hf.space/api/health",
                            language="bash",
                        )
                        gr.Markdown("#### Swagger UI")
                        gr.Code(
                            value="https://your-space.hf.space/api/docs",
                            language="bash",
                        )

        # ── Event handlers ────────────────────────────────────────────────────

        # Send message (streaming)
        def submit(message, history, model_name, top_k, key, filter_doc):
            for h, _ in chat_stream(message, history, model_name, top_k, key, filter_doc):
                yield h, ""

        send_btn.click(
            fn=submit,
            inputs=[msg_input, chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs],
            outputs=[chatbot, msg_input],
        )
        msg_input.submit(
            fn=submit,
            inputs=[msg_input, chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs],
            outputs=[chatbot, msg_input],
        )

        # Quick prompts
        for btn, prompt_text in [
            (q1, "Summarize the key points from my documents."),
            (q2, "What are the main topics covered in these documents?"),
            (q3, "List all important dates and numbers mentioned in the documents."),
            (q4, "What conclusions or recommendations were drawn in these documents?"),
        ]:
            btn.click(
                fn=lambda h, m, k, f, t, p=prompt_text: next(
                    iter(chat_stream(p, h, m, k, t, f))
                ),
                inputs=[chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs],
                outputs=[chatbot, msg_input],
            ).then(
                fn=lambda h, m, k, f, t, p=prompt_text: (yield from chat_stream(p, h, m, k, t, f)),
                inputs=[chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs],
                outputs=[chatbot, msg_input],
            )

        # Better quick prompt handler
        def make_quick_fn(prompt):
            def fn(history, model_name, top_k, key, filter_doc):
                for h, _ in chat_stream(prompt, history, model_name, top_k, key, filter_doc):
                    yield h, ""
            return fn

        q1.click(make_quick_fn("Summarize the key points from my documents."), [chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs], [chatbot, msg_input])
        q2.click(make_quick_fn("What are the main topics covered in these documents?"), [chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs], [chatbot, msg_input])
        q3.click(make_quick_fn("List all important dates and numbers mentioned in the documents."), [chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs], [chatbot, msg_input])
        q4.click(make_quick_fn("What conclusions or recommendations were drawn in these documents?"), [chatbot, model_dropdown, top_k_slider, openrouter_key, filter_docs], [chatbot, msg_input])

        clear_btn.click(lambda: ([], ""), outputs=[chatbot, msg_input])

        # Upload
        def do_upload(files):
            if not files:
                return "⚠️ No files selected."
            result = handle_upload(files)
            return result

        upload_btn.click(fn=do_upload, inputs=[file_uploader], outputs=[upload_status])

        # Refresh docs list + filter dropdown
        def refresh_docs():
            text, docs = get_documents()
            names = ["All Documents"] + [d["filename"] for d in docs]
            return text, gr.Dropdown(choices=names, value="All Documents")

        refresh_btn.click(fn=refresh_docs, outputs=[docs_display, filter_docs])
        upload_btn.click(fn=refresh_docs, outputs=[docs_display, filter_docs])

        # Clear all docs
        def do_clear():
            msg = delete_all_docs()
            return msg, "📭 No documents indexed yet.", gr.Dropdown(choices=["All Documents"], value="All Documents")

        clear_all_btn.click(fn=do_clear, outputs=[clear_status, docs_display, filter_docs])

        # Health status on load
        def load_health():
            resp = sync_get("/health")
            if resp and resp.status_code == 200:
                h = resp.json()
                return (
                    f"**Status:** ✅ Online  \n"
                    f"**Chunks indexed:** {h['documents_indexed']}  \n"
                    f"**Version:** {h['version']}  \n"
                    f"**Model:** `{h['llm_model'].split('/')[-1]}`"
                )
            return "**Status:** ⚠️ Starting up..."

        def load_docs():
            text, _ = get_documents()
            return text

        demo.load(fn=load_health, outputs=[health_display])
        demo.load(fn=load_docs, outputs=[docs_display])

    return demo


# ── Mount Gradio into FastAPI ─────────────────────────────────────────────────
demo = build_ui()
app = gr.mount_gradio_app(fastapi_app, demo, path="/")


# ── Shutdown hook — push DB to HF ─────────────────────────────────────────────
import atexit
atexit.register(push_db_to_hf)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")

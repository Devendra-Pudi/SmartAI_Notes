"""
SmartAI Notes — Streamlit Frontend
A beautiful RAG-powered document Q&A interface.
"""
import streamlit as st
import requests
import json
import time
from pathlib import Path
from typing import List, Optional

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartAI Notes",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────────
API_BASE = "http://localhost:8000/api/v1"
HEALTH_URL = "http://localhost:8000/health"

FILE_TYPE_ICONS = {
    ".pdf": "📕",
    ".txt": "📄",
    ".md": "📝",
    ".docx": "📘",
    ".csv": "📊",
    ".json": "🗂️",
}

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #12151f 100%);
        border-right: 1px solid #2d3748;
    }

    /* Chat bubbles */
    .user-bubble {
        background: linear-gradient(135deg, #2d3a8c, #1e40af);
        border-radius: 18px 18px 4px 18px;
        padding: 12px 16px;
        margin: 8px 0;
        color: white;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(29, 64, 175, 0.3);
    }
    .ai-bubble {
        background: linear-gradient(135deg, #1a2535, #1e2a3a);
        border: 1px solid #2d4a6e;
        border-radius: 18px 18px 18px 4px;
        padding: 12px 16px;
        margin: 8px 0;
        color: #e2e8f0;
        max-width: 90%;
        box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    }

    /* Source cards */
    .source-card {
        background: #1a2535;
        border: 1px solid #2d4a6e;
        border-left: 3px solid #38bdf8;
        border-radius: 8px;
        padding: 10px 14px;
        margin: 4px 0;
        font-size: 0.82rem;
        color: #94a3b8;
    }
    .source-score {
        color: #38bdf8;
        font-weight: 600;
    }

    /* Upload zone */
    .upload-zone {
        border: 2px dashed #2d4a6e;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: #1a2535;
    }

    /* Doc badge */
    .doc-badge {
        background: #1e3a5f;
        border: 1px solid #2d6a9f;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.78rem;
        color: #7dd3fc;
        display: inline-block;
        margin: 2px;
    }

    /* Metric card */
    .metric-card {
        background: #1a2535;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Title gradient */
    .main-title {
        background: linear-gradient(90deg, #38bdf8, #818cf8, #34d399);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2rem;
        font-weight: 800;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── Session state ─────────────────────────────────────────────────────────────
def init_session():
    defaults = {
        "chat_history": [],
        "documents": [],
        "selected_docs": [],
        "selected_model": "meta-llama/llama-3.3-8b-instruct:free",
        "top_k": 5,
        "api_key_set": False,
        "openrouter_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()


# ── API helpers ───────────────────────────────────────────────────────────────
def api_headers():
    return {"Content-Type": "application/json"}


def check_health() -> dict:
    try:
        r = requests.get(HEALTH_URL, timeout=3)
        return r.json() if r.status_code == 200 else {}
    except Exception:
        return {}


def fetch_documents() -> List[dict]:
    try:
        r = requests.get(f"{API_BASE}/documents/", timeout=5)
        if r.status_code == 200:
            return r.json().get("documents", [])
    except Exception:
        pass
    return []


def fetch_models() -> List[dict]:
    try:
        r = requests.get(f"{API_BASE}/query/models", timeout=5)
        if r.status_code == 200:
            return r.json().get("models", [])
    except Exception:
        pass
    return []


def upload_file(file_bytes, filename) -> dict:
    try:
        files = {"file": (filename, file_bytes, "application/octet-stream")}
        r = requests.post(f"{API_BASE}/documents/upload", files=files, timeout=120)
        return r.json()
    except Exception as e:
        return {"status": "failed", "message": str(e)}


def delete_document(file_id: str) -> bool:
    try:
        r = requests.delete(f"{API_BASE}/documents/{file_id}", timeout=10)
        return r.status_code == 200
    except Exception:
        return False


def query_rag(question: str, file_ids: list, model: str, top_k: int, history: list) -> dict:
    payload = {
        "question": question,
        "file_ids": file_ids if file_ids else None,
        "model": model,
        "top_k": top_k,
        "chat_history": history,
    }
    try:
        r = requests.post(
            f"{API_BASE}/query/",
            json=payload,
            timeout=90,
            headers=api_headers(),
        )
        if r.status_code == 200:
            return r.json()
        return {"answer": f"❌ API error: {r.status_code} — {r.text[:200]}", "sources": []}
    except requests.Timeout:
        return {"answer": "⏱️ Request timed out. The LLM is taking too long. Try a shorter question or different model.", "sources": []}
    except Exception as e:
        return {"answer": f"❌ Connection error: {str(e)}", "sources": []}


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="main-title">🧠 SmartAI Notes</div>', unsafe_allow_html=True)
    st.caption("RAG-powered Document Intelligence")
    st.divider()

    # Health status
    health = check_health()
    if health.get("status") == "ok":
        st.success(f"✅ API Online | {health.get('documents_indexed', 0)} chunks indexed")
    else:
        st.error("❌ API Offline — Start the FastAPI backend first")
        st.code("cd backend && uvicorn main:app --reload", language="bash")

    st.divider()

    # ── OpenRouter API Key ────────────────────────────────────────────────────
    st.markdown("#### 🔑 OpenRouter API Key")
    key_input = st.text_input(
        "API Key",
        type="password",
        placeholder="sk-or-v1-...",
        value=st.session_state.openrouter_key,
        help="Get your free key at openrouter.ai",
        label_visibility="collapsed",
    )
    if key_input:
        st.session_state.openrouter_key = key_input
        st.session_state.api_key_set = True
        st.caption("✅ Key saved for this session")
    else:
        st.caption("⚠️ [Get free key at openrouter.ai](https://openrouter.ai)")

    st.divider()

    # ── Model selector ────────────────────────────────────────────────────────
    st.markdown("#### 🤖 LLM Model")
    models = fetch_models()
    model_options = {m["name"]: m["id"] for m in models} if models else {
        "Llama 3.3 8B (Free)": "meta-llama/llama-3.3-8b-instruct:free",
        "Mistral 7B (Free)": "mistralai/mistral-7b-instruct:free",
        "Gemma 3 4B (Free)": "google/gemma-3-4b-it:free",
        "DeepSeek R1 (Free)": "deepseek/deepseek-r1:free",
        "Qwen 2.5 72B (Free)": "qwen/qwen-2.5-72b-instruct:free",
    }
    selected_model_name = st.selectbox(
        "Model",
        list(model_options.keys()),
        label_visibility="collapsed",
    )
    st.session_state.selected_model = model_options[selected_model_name]

    free_models = [m for m in (models or []) if m.get("free")]
    if free_models:
        st.caption(f"💡 {len(free_models)} free models available")

    st.divider()

    # ── RAG settings ──────────────────────────────────────────────────────────
    st.markdown("#### ⚙️ RAG Settings")
    st.session_state.top_k = st.slider(
        "Top-K chunks to retrieve",
        min_value=1,
        max_value=15,
        value=st.session_state.top_k,
        help="More chunks = more context but slower",
    )

    st.divider()

    # ── Document management ───────────────────────────────────────────────────
    st.markdown("#### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md", "docx", "csv", "json"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        if st.button("📤 Index Documents", use_container_width=True, type="primary"):
            progress = st.progress(0)
            for i, f in enumerate(uploaded_files):
                with st.spinner(f"Indexing {f.name}..."):
                    result = upload_file(f.read(), f.name)
                    if result.get("status") == "ready":
                        st.success(
                            f"✅ {f.name} — {result.get('chunks_created', 0)} chunks"
                        )
                    else:
                        st.error(f"❌ {f.name}: {result.get('message', 'Failed')}")
                progress.progress((i + 1) / len(uploaded_files))
            st.session_state.documents = fetch_documents()
            time.sleep(0.5)
            st.rerun()

    st.divider()

    # ── Indexed documents ─────────────────────────────────────────────────────
    st.markdown("#### 🗂️ Indexed Documents")
    if not st.session_state.documents:
        st.session_state.documents = fetch_documents()

    docs = st.session_state.documents

    if docs:
        st.caption(f"{len(docs)} document(s) indexed")

        # Filter selection
        all_option = "🌐 All Documents"
        doc_options = [all_option] + [d["filename"] for d in docs]
        selected = st.multiselect(
            "Search in",
            doc_options,
            default=[all_option],
            label_visibility="collapsed",
        )

        if all_option in selected or not selected:
            st.session_state.selected_docs = []
        else:
            st.session_state.selected_docs = [
                d["file_id"] for d in docs if d["filename"] in selected
            ]

        # List docs with delete button
        for doc in docs:
            icon = FILE_TYPE_ICONS.get(doc.get("file_type", ".txt"), "📄")
            col1, col2 = st.columns([5, 1])
            with col1:
                st.markdown(
                    f'<span class="doc-badge">{icon} {doc["filename"][:28]} '
                    f'<span style="color:#64748b">({doc["chunks"]} chunks)</span></span>',
                    unsafe_allow_html=True,
                )
            with col2:
                if st.button("🗑", key=f"del_{doc['file_id']}", help="Delete"):
                    if delete_document(doc["file_id"]):
                        st.session_state.documents = fetch_documents()
                        st.rerun()

        if st.button("🗑️ Clear All", use_container_width=True):
            requests.delete(f"{API_BASE}/documents/", timeout=10)
            st.session_state.documents = []
            st.session_state.selected_docs = []
            st.rerun()
    else:
        st.info("📭 No documents yet. Upload files above.")

    st.divider()
    if st.button("🧹 Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────
col_main, col_info = st.columns([3, 1])

with col_main:
    st.markdown('<div class="main-title">🧠 SmartAI Notes</div>', unsafe_allow_html=True)
    st.markdown(
        "Ask questions about your uploaded documents. The AI retrieves relevant passages and generates cited answers."
    )
    st.divider()

    # ── Chat history display ──────────────────────────────────────────────────
    chat_container = st.container()

    with chat_container:
        if not st.session_state.chat_history:
            st.markdown(
                """
                <div style="text-align:center; padding: 60px 20px; color: #4a5568;">
                    <div style="font-size: 3rem; margin-bottom: 10px;">💬</div>
                    <div style="font-size: 1.1rem; color: #718096;">
                        Upload documents and start asking questions!
                    </div>
                    <div style="font-size: 0.85rem; color: #4a5568; margin-top: 8px;">
                        Supports PDF · TXT · Markdown · DOCX · CSV · JSON
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            for msg in st.session_state.chat_history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="user-bubble">🧑‍💻 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="ai-bubble">🧠 {msg["content"]}</div>',
                        unsafe_allow_html=True,
                    )
                    # Show sources if available
                    if msg.get("sources"):
                        with st.expander(f"📚 {len(msg['sources'])} Source(s) used", expanded=False):
                            for src in msg["sources"]:
                                score_pct = int(src.get("score", 0) * 100)
                                page_info = f" | Page {src['page']}" if src.get("page") else ""
                                st.markdown(
                                    f"""<div class="source-card">
                                    <strong>📄 {src['source']}{page_info}</strong>
                                    <span class="source-score"> · {score_pct}% match</span>
                                    <br/><span style="color:#64748b;font-size:0.8rem;">{src['content'][:300]}...</span>
                                    </div>""",
                                    unsafe_allow_html=True,
                                )

    # ── Input ─────────────────────────────────────────────────────────────────
    st.divider()

    # Quick prompt suggestions
    if not st.session_state.chat_history:
        st.markdown("**💡 Try asking:**")
        suggestions = [
            "Summarize the key points from my documents",
            "What are the main topics covered?",
            "List all important dates or numbers mentioned",
            "Explain [topic] in simple terms",
        ]
        cols = st.columns(2)
        for i, sug in enumerate(suggestions):
            if cols[i % 2].button(f"💬 {sug}", key=f"sug_{i}", use_container_width=True):
                st.session_state["_pending_question"] = sug
                st.rerun()

    with st.form("chat_form", clear_on_submit=True):
        question = st.text_area(
            "Ask a question",
            placeholder="e.g. What are the main conclusions of the report?",
            height=80,
            label_visibility="collapsed",
            value=st.session_state.pop("_pending_question", ""),
        )
        submitted = st.form_submit_button("🚀 Ask", use_container_width=True, type="primary")

    if submitted and question.strip():
        docs = st.session_state.documents
        if not docs:
            st.warning("⚠️ Please upload at least one document before asking questions.")
        elif not health.get("status") == "ok":
            st.error("❌ Backend API is offline. Please start the FastAPI server.")
        else:
            # Add user message to history
            st.session_state.chat_history.append(
                {"role": "user", "content": question}
            )

            # Build history for API (exclude current question)
            api_history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.chat_history[:-1]
            ]

            with st.spinner("🔍 Retrieving relevant context..."):
                result = query_rag(
                    question=question,
                    file_ids=st.session_state.selected_docs,
                    model=st.session_state.selected_model,
                    top_k=st.session_state.top_k,
                    history=api_history,
                )

            answer = result.get("answer", "Sorry, I couldn't generate an answer.")
            sources = result.get("sources", [])
            model_used = result.get("model_used", "")
            tokens = result.get("tokens_used", 0)

            # Add AI response to history
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                    "model": model_used,
                    "tokens": tokens,
                }
            )
            st.rerun()


# ── Right column: stats & info ────────────────────────────────────────────────
with col_info:
    st.markdown("### 📊 Session Stats")

    docs = st.session_state.documents
    total_chunks = sum(d.get("chunks", 0) for d in docs)
    num_messages = len([m for m in st.session_state.chat_history if m["role"] == "user"])

    st.markdown(
        f"""
        <div class="metric-card" style="margin-bottom:10px;">
            <div style="font-size:1.8rem; color:#38bdf8; font-weight:700;">{len(docs)}</div>
            <div style="color:#64748b; font-size:0.8rem;">Documents</div>
        </div>
        <div class="metric-card" style="margin-bottom:10px;">
            <div style="font-size:1.8rem; color:#34d399; font-weight:700;">{total_chunks}</div>
            <div style="color:#64748b; font-size:0.8rem;">Indexed Chunks</div>
        </div>
        <div class="metric-card" style="margin-bottom:10px;">
            <div style="font-size:1.8rem; color:#818cf8; font-weight:700;">{num_messages}</div>
            <div style="color:#64748b; font-size:0.8rem;">Questions Asked</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown("### 🔍 Active Scope")
    selected_docs = st.session_state.selected_docs
    if not selected_docs:
        st.info("🌐 All documents")
    else:
        for fid in selected_docs:
            doc = next((d for d in docs if d["file_id"] == fid), None)
            if doc:
                icon = FILE_TYPE_ICONS.get(doc.get("file_type", ".txt"), "📄")
                st.markdown(f"`{icon} {doc['filename'][:20]}`")

    st.divider()
    st.markdown("### 🤖 Current Model")
    model_id = st.session_state.selected_model
    short_name = model_id.split("/")[-1].replace(":free", "")
    st.code(short_name, language=None)
    if ":free" in model_id:
        st.success("💚 Free tier")

    st.divider()
    st.markdown("### ℹ️ How It Works")
    st.markdown(
        """
        <div style="font-size:0.82rem; color:#64748b; line-height:1.7;">
        1. 📄 <b>Upload</b> your documents<br/>
        2. 🔍 <b>Embed</b> text into ChromaDB<br/>
        3. 💬 <b>Ask</b> a question<br/>
        4. 🧲 <b>Retrieve</b> relevant chunks<br/>
        5. 🤖 <b>Generate</b> answer via LLM<br/>
        6. 📚 <b>Cite</b> source documents
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div style="font-size:0.75rem; color:#4a5568; text-align:center;">Built with ❤️ by <a href="https://github.com/Devendra-Pudi" style="color:#38bdf8;">Devendra Prasad Pudi</a></div>',
        unsafe_allow_html=True,
    )

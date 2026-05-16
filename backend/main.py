"""
SmartAI Notes — FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from app.core.config import get_settings
from app.core.logger import get_logger
from app.api import upload, query, health

settings = get_settings()
logger = get_logger("main")

# ── App factory ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="SmartAI Notes API",
    description="""
## 🧠 SmartAI Notes — RAG-powered Document Q&A

Upload your documents and ask questions. The AI retrieves the most relevant passages
from your files and generates accurate, cited answers.

### Features
- 📄 **Multi-format support**: PDF, TXT, MD, DOCX, CSV, JSON
- 🔍 **Semantic search**: ChromaDB + SentenceTransformer embeddings
- 🤖 **Multiple LLMs**: Llama 3, Mistral, Gemma, DeepSeek via OpenRouter
- 💬 **Chat history**: Multi-turn conversation support
- ⚡ **Streaming**: Token-by-token streaming responses
    """,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── Middleware ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(upload.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


# ── Startup / shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup():
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting up...")
    logger.info(f"📦 ChromaDB path: {settings.CHROMA_PERSIST_DIR}")
    logger.info(f"🤖 Default LLM: {settings.DEFAULT_LLM_MODEL}")
    logger.info("✅ Server ready. Visit /docs for API documentation.")


@app.on_event("shutdown")
async def shutdown():
    logger.info("👋 SmartAI Notes shutting down.")

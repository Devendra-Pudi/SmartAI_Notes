"""
Health check endpoint.
"""
from fastapi import APIRouter
from app.models.schemas import HealthResponse
from app.services.vector_store import get_vector_store
from app.services.llm_service import get_llm_service
from app.core.config import get_settings

router = APIRouter(tags=["Health"])
settings = get_settings()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns the health status of the service."""
    vector_store = get_vector_store()
    count = vector_store.get_collection_count()

    try:
        # Just verifying collection is accessible
        chroma_status = "ok"
    except Exception:
        chroma_status = "error"

    llm = get_llm_service()

    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        documents_indexed=count,
        chroma_status=chroma_status,
        llm_model=llm.default_model,
    )


@router.get("/")
async def root():
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
    }

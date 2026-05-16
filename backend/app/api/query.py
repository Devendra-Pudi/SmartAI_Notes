"""
Query / RAG endpoints.
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.models.schemas import QueryRequest, QueryResponse, ModelsResponse
from app.services.rag_pipeline import get_rag_pipeline
from app.services.llm_service import get_llm_service
from app.core.logger import get_logger

router = APIRouter(prefix="/query", tags=["Query"])
logger = get_logger(__name__)


@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """
    Ask a question about the uploaded documents.
    Returns a generated answer with source citations.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    try:
        pipeline = get_rag_pipeline()
        response = await pipeline.query(request)
        return response
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@router.post("/stream")
async def stream_query(request: QueryRequest):
    """
    Ask a question and get a streamed response (SSE / text stream).
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    pipeline = get_rag_pipeline()

    async def token_generator():
        try:
            async for token in pipeline.stream_query(request):
                yield token
        except Exception as e:
            yield f"\n\n⚠️ Stream error: {str(e)}"

    return StreamingResponse(
        token_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/models", response_model=ModelsResponse)
async def get_models():
    """List all available LLM models via OpenRouter."""
    llm = get_llm_service()
    return ModelsResponse(
        models=llm.get_available_models(),
        default_model=llm.default_model,
    )

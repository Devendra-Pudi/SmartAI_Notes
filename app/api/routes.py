"""
All FastAPI routes — upload, query, health.
"""
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.schemas import (
    UploadResponse, DocumentStatus, DocumentListResponse,
    DocumentInfo, DeleteResponse, QueryRequest, QueryResponse, HealthResponse,
)
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import get_vector_store
from app.services.rag_pipeline import get_rag_pipeline
from app.services.llm_service import get_llm_service

router = APIRouter()
logger = get_logger(__name__)
settings = get_settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


# ── Health ────────────────────────────────────────────────────────────────────
@router.get("/health", response_model=HealthResponse)
async def health():
    vs = get_vector_store()
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        documents_indexed=vs.get_collection_count(),
        chroma_status="ok",
        llm_model=settings.DEFAULT_LLM_MODEL,
    )


@router.get("/")
async def root():
    return {"app": settings.APP_NAME, "version": settings.APP_VERSION, "docs": "/api/docs"}


# ── Documents ─────────────────────────────────────────────────────────────────
@router.post("/documents/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}")

    content = await file.read()
    if len(content) / 1024 / 1024 > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(413, f"File too large. Max: {settings.MAX_FILE_SIZE_MB} MB")

    safe_name = Path(file.filename).name
    temp_path = os.path.join(settings.UPLOAD_DIR, safe_name)

    try:
        with open(temp_path, "wb") as f:
            f.write(content)

        processor = DocumentProcessor()
        file_id, chunks = processor.process_file(temp_path, safe_name)
        vs = get_vector_store()
        n = vs.add_chunks(chunks, safe_name)

        return UploadResponse(
            file_id=file_id, filename=safe_name,
            status=DocumentStatus.READY, chunks_created=n,
            message=f"✅ Indexed {n} chunks from '{safe_name}'",
        )
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(500, str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/documents/upload/batch", response_model=List[UploadResponse])
async def upload_batch(files: List[UploadFile] = File(...)):
    responses = []
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            responses.append(UploadResponse(file_id="", filename=file.filename, status=DocumentStatus.FAILED, chunks_created=0, message=f"Unsupported: {ext}"))
            continue
        content = await file.read()
        safe_name = Path(file.filename).name
        temp_path = os.path.join(settings.UPLOAD_DIR, safe_name)
        try:
            with open(temp_path, "wb") as f:
                f.write(content)
            processor = DocumentProcessor()
            file_id, chunks = processor.process_file(temp_path, safe_name)
            n = get_vector_store().add_chunks(chunks, safe_name)
            responses.append(UploadResponse(file_id=file_id, filename=safe_name, status=DocumentStatus.READY, chunks_created=n, message=f"✅ {n} chunks"))
        except Exception as e:
            responses.append(UploadResponse(file_id="", filename=file.filename, status=DocumentStatus.FAILED, chunks_created=0, message=str(e)))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    return responses


@router.get("/documents", response_model=DocumentListResponse)
async def list_documents():
    docs = get_vector_store().list_documents()
    return DocumentListResponse(
        documents=[DocumentInfo(file_id=d["file_id"], filename=d["filename"], file_type=d.get("file_type", ".txt"), chunks=d["chunks"], uploaded_at=d.get("uploaded_at", ""), status=DocumentStatus.READY) for d in docs],
        total=len(docs),
    )


@router.delete("/documents/{file_id}", response_model=DeleteResponse)
async def delete_document(file_id: str):
    ok = get_vector_store().delete_document(file_id)
    if not ok:
        raise HTTPException(404, f"Document {file_id} not found.")
    return DeleteResponse(file_id=file_id, message="Deleted.", success=True)


@router.delete("/documents", response_model=DeleteResponse)
async def clear_all():
    ok = get_vector_store().clear_all()
    return DeleteResponse(file_id="all", message="Cleared." if ok else "Failed.", success=ok)


# ── Query ─────────────────────────────────────────────────────────────────────
@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    try:
        return await get_rag_pipeline().query(request)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(500, str(e))


@router.post("/query/stream")
async def stream_query(request: QueryRequest):
    if not request.question.strip():
        raise HTTPException(400, "Question cannot be empty.")
    pipeline = get_rag_pipeline()

    async def gen():
        async for token in pipeline.stream_query(request):
            yield token

    return StreamingResponse(gen(), media_type="text/plain", headers={"Cache-Control": "no-cache"})


@router.get("/query/models")
async def get_models():
    llm = get_llm_service()
    return {"models": llm.get_available_models(), "default": llm.default_model}

"""
File upload endpoints.
"""
import os
import shutil
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logger import get_logger
from app.models.schemas import UploadResponse, DocumentStatus, DeleteResponse, DocumentListResponse, DocumentInfo
from app.services.document_processor import DocumentProcessor
from app.services.vector_store import get_vector_store

router = APIRouter(prefix="/documents", tags=["Documents"])
logger = get_logger(__name__)
settings = get_settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)


@router.post("/upload", response_model=UploadResponse)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document and index it into ChromaDB for RAG retrieval.
    Supports: PDF, TXT, MD, DOCX, CSV, JSON
    """
    # Validate extension
    ext = Path(file.filename).suffix.lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {settings.ALLOWED_EXTENSIONS}",
        )

    # Validate file size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_FILE_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_FILE_SIZE_MB} MB",
        )

    # Save file temporarily
    safe_name = Path(file.filename).name
    temp_path = os.path.join(settings.UPLOAD_DIR, safe_name)
    with open(temp_path, "wb") as f:
        f.write(content)

    logger.info(f"Received upload: {safe_name} ({size_mb:.2f} MB)")

    # Process and index
    try:
        processor = DocumentProcessor()
        file_id, chunks = processor.process_file(temp_path, safe_name)

        vector_store = get_vector_store()
        chunks_added = vector_store.add_chunks(chunks, safe_name)

        return UploadResponse(
            file_id=file_id,
            filename=safe_name,
            status=DocumentStatus.READY,
            chunks_created=chunks_added,
            message=f"✅ Successfully indexed {chunks_added} chunks from '{safe_name}'",
        )

    except Exception as e:
        logger.error(f"Failed to process {safe_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/upload/batch", response_model=List[UploadResponse])
async def upload_multiple_documents(files: List[UploadFile] = File(...)):
    """Upload multiple documents at once."""
    responses = []
    for file in files:
        try:
            # Reuse single upload logic
            ext = Path(file.filename).suffix.lower()
            if ext not in settings.ALLOWED_EXTENSIONS:
                responses.append(
                    UploadResponse(
                        file_id="",
                        filename=file.filename,
                        status=DocumentStatus.FAILED,
                        chunks_created=0,
                        message=f"Skipped: unsupported type '{ext}'",
                    )
                )
                continue

            content = await file.read()
            safe_name = Path(file.filename).name
            temp_path = os.path.join(settings.UPLOAD_DIR, safe_name)

            with open(temp_path, "wb") as f:
                f.write(content)

            processor = DocumentProcessor()
            file_id, chunks = processor.process_file(temp_path, safe_name)

            vector_store = get_vector_store()
            chunks_added = vector_store.add_chunks(chunks, safe_name)

            responses.append(
                UploadResponse(
                    file_id=file_id,
                    filename=safe_name,
                    status=DocumentStatus.READY,
                    chunks_created=chunks_added,
                    message=f"✅ Indexed {chunks_added} chunks",
                )
            )

            if os.path.exists(temp_path):
                os.remove(temp_path)

        except Exception as e:
            responses.append(
                UploadResponse(
                    file_id="",
                    filename=file.filename,
                    status=DocumentStatus.FAILED,
                    chunks_created=0,
                    message=f"❌ Error: {str(e)}",
                )
            )

    return responses


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """List all indexed documents."""
    vector_store = get_vector_store()
    docs = vector_store.list_documents()
    return DocumentListResponse(
        documents=[
            DocumentInfo(
                file_id=d["file_id"],
                filename=d["filename"],
                file_type=d.get("file_type", ".txt"),
                chunks=d["chunks"],
                uploaded_at=d.get("uploaded_at", ""),
                status=DocumentStatus.READY,
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.delete("/{file_id}", response_model=DeleteResponse)
async def delete_document(file_id: str):
    """Delete a specific document from the index."""
    vector_store = get_vector_store()
    success = vector_store.delete_document(file_id)

    if success:
        return DeleteResponse(
            file_id=file_id,
            message=f"Document {file_id} deleted successfully.",
            success=True,
        )
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Document {file_id} not found or could not be deleted.",
        )


@router.delete("/", response_model=DeleteResponse)
async def clear_all_documents():
    """Delete ALL documents from the index. Use with caution."""
    vector_store = get_vector_store()
    success = vector_store.clear_all()
    return DeleteResponse(
        file_id="all",
        message="All documents cleared." if success else "Failed to clear documents.",
        success=success,
    )

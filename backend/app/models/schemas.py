"""
Pydantic schemas for request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class DocumentStatus(str, Enum):
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


class UploadResponse(BaseModel):
    file_id: str
    filename: str
    status: DocumentStatus
    chunks_created: int
    message: str


class DocumentInfo(BaseModel):
    file_id: str
    filename: str
    file_type: str
    chunks: int
    uploaded_at: str
    status: DocumentStatus


class DocumentListResponse(BaseModel):
    documents: List[DocumentInfo]
    total: int


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: Optional[str] = None


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    file_ids: Optional[List[str]] = None  # None = search all docs
    model: Optional[str] = None
    top_k: Optional[int] = Field(default=5, ge=1, le=20)
    chat_history: Optional[List[ChatMessage]] = []


class SourceChunk(BaseModel):
    content: str
    source: str
    page: Optional[int] = None
    score: float
    chunk_index: int


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    model_used: str
    tokens_used: Optional[int] = None
    question: str


class DeleteResponse(BaseModel):
    file_id: str
    message: str
    success: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    documents_indexed: int
    chroma_status: str
    llm_model: str


class ModelsResponse(BaseModel):
    models: List[dict]
    default_model: str

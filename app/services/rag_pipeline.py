"""
RAG Pipeline — Retrieve → Augment → Generate
"""
from typing import List, Optional, AsyncGenerator

from app.services.vector_store import get_vector_store
from app.services.llm_service import get_llm_service, build_rag_prompt
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.core.logger import get_logger

logger = get_logger(__name__)

NO_DOCS_MESSAGE = (
    "⚠️ **No documents are indexed yet.**\n\n"
    "Please upload one or more files using the **📄 Upload Documents** tab, then ask your question again."
)

NO_CONTEXT_MESSAGE = (
    "🔍 **No relevant content found** for your question in the uploaded documents.\n\n"
    "Try:\n"
    "- Rephrasing your question\n"
    "- Uploading more relevant files\n"
    "- Lowering the relevance threshold by increasing **Top-K** in settings"
)


class RAGPipeline:
    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm_service()

    async def query(self, request: QueryRequest) -> QueryResponse:
        question = request.question.strip()
        top_k = request.top_k or 5
        model = request.model or None

        logger.info(f"Query: '{question[:80]}' | top_k={top_k}")

        # Step 1 — Retrieve
        raw_chunks = self.vector_store.query(question=question, file_ids=request.file_ids, top_k=top_k)

        if not raw_chunks:
            is_empty = self.vector_store.get_collection_count() == 0
            return QueryResponse(
                answer=NO_DOCS_MESSAGE if is_empty else NO_CONTEXT_MESSAGE,
                sources=[],
                model_used="system",
                tokens_used=0,
                question=question,
            )

        # Step 2 — Filter low-relevance
        filtered = [c for c in raw_chunks if c["score"] >= 0.25] or raw_chunks[:2]
        logger.info(f"Using {len(filtered)} chunks (top score: {filtered[0]['score']:.3f})")

        # Step 3 — Build prompt
        messages = build_rag_prompt(question, filtered, request.chat_history)

        # Step 4 — Generate
        data = await self.llm.generate(messages, model=model)
        answer, tokens = self.llm.extract_answer(data)
        used_model = data.get("model", model or self.llm.default_model)

        # Step 5 — Build response
        sources = [
            SourceChunk(
                content=c["content"][:400] + "..." if len(c["content"]) > 400 else c["content"],
                source=c["source"],
                page=c.get("page"),
                score=c["score"],
                chunk_index=c["chunk_index"],
            )
            for c in filtered
        ]
        return QueryResponse(answer=answer, sources=sources, model_used=used_model, tokens_used=tokens, question=question)

    async def stream_query(self, request: QueryRequest) -> AsyncGenerator[str, None]:
        question = request.question.strip()
        raw_chunks = self.vector_store.query(question=question, file_ids=request.file_ids, top_k=request.top_k or 5)

        if not raw_chunks:
            is_empty = self.vector_store.get_collection_count() == 0
            yield NO_DOCS_MESSAGE if is_empty else NO_CONTEXT_MESSAGE
            return

        filtered = [c for c in raw_chunks if c["score"] >= 0.25] or raw_chunks[:2]
        messages = build_rag_prompt(question, filtered, request.chat_history)

        async for token in self.llm.stream_generate(messages, model=request.model):
            yield token


_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = RAGPipeline()
    return _pipeline

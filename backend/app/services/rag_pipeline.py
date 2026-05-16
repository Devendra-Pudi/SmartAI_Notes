"""
RAG Pipeline — orchestrates retrieval + generation.
"""
from typing import List, Optional, Dict, Any

from app.services.vector_store import get_vector_store
from app.services.llm_service import get_llm_service, build_rag_prompt
from app.models.schemas import QueryRequest, QueryResponse, SourceChunk
from app.core.logger import get_logger

logger = get_logger(__name__)


class RAGPipeline:
    """
    Retrieval-Augmented Generation pipeline.

    Flow:
      1. Embed the question
      2. Retrieve top-K relevant chunks from ChromaDB
      3. Build prompt with context
      4. Send to LLM via OpenRouter
      5. Return structured response
    """

    def __init__(self):
        self.vector_store = get_vector_store()
        self.llm = get_llm_service()

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Execute the full RAG pipeline for a user question.
        """
        question = request.question.strip()
        top_k = request.top_k or 5
        model = request.model or None

        logger.info(f"RAG query: '{question[:80]}...' | top_k={top_k} | model={model}")

        # ── Step 1: Retrieve relevant chunks ─────────────────────────────────
        raw_chunks = self.vector_store.query(
            question=question,
            file_ids=request.file_ids,
            top_k=top_k,
        )

        if not raw_chunks:
            # No context — still ask the LLM but inform it
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are SmartAI Notes. No documents have been indexed yet. "
                        "Politely tell the user to upload documents first."
                    ),
                },
                {"role": "user", "content": question},
            ]
            response_data = await self.llm.generate(messages, model=model)
            answer, tokens = self.llm.extract_answer(response_data)
            return QueryResponse(
                answer=answer,
                sources=[],
                model_used=model or self.llm.default_model,
                tokens_used=tokens,
                question=question,
            )

        logger.info(f"Retrieved {len(raw_chunks)} chunks (top score: {raw_chunks[0]['score']:.3f})")

        # ── Step 2: Filter low-relevance chunks ──────────────────────────────
        # Keep chunks with cosine similarity > 0.25
        filtered_chunks = [c for c in raw_chunks if c["score"] >= 0.25]
        if not filtered_chunks:
            filtered_chunks = raw_chunks[:2]  # Always use at least 2

        # ── Step 3: Build RAG prompt ──────────────────────────────────────────
        messages = build_rag_prompt(
            question=question,
            context_chunks=filtered_chunks,
            chat_history=request.chat_history,
        )

        # ── Step 4: Generate LLM response ────────────────────────────────────
        response_data = await self.llm.generate(messages, model=model)
        answer, tokens = self.llm.extract_answer(response_data)
        used_model = response_data.get("model", model or self.llm.default_model)

        logger.info(f"Answer generated. Tokens used: {tokens}")

        # ── Step 5: Build response ────────────────────────────────────────────
        sources = [
            SourceChunk(
                content=c["content"][:400] + "..." if len(c["content"]) > 400 else c["content"],
                source=c["source"],
                page=c.get("page"),
                score=c["score"],
                chunk_index=c["chunk_index"],
            )
            for c in filtered_chunks
        ]

        return QueryResponse(
            answer=answer,
            sources=sources,
            model_used=used_model,
            tokens_used=tokens,
            question=question,
        )

    async def stream_query(self, request: QueryRequest):
        """
        Stream the RAG response token by token.
        Yields string tokens.
        """
        question = request.question.strip()
        top_k = request.top_k or 5
        model = request.model or None

        raw_chunks = self.vector_store.query(
            question=question,
            file_ids=request.file_ids,
            top_k=top_k,
        )

        if not raw_chunks:
            yield "⚠️ No documents found. Please upload files first."
            return

        filtered_chunks = [c for c in raw_chunks if c["score"] >= 0.25] or raw_chunks[:2]
        messages = build_rag_prompt(question, filtered_chunks, request.chat_history)

        async for token in self.llm.stream_generate(messages, model=model):
            yield token


# Singleton
_rag_pipeline: Optional[RAGPipeline] = None


def get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline()
    return _rag_pipeline

"""
Vector store service using ChromaDB with local sentence-transformer embeddings.
Handles storing, retrieving, and deleting document chunks.
"""
import os
from typing import List, Optional, Dict, Any
from datetime import datetime

import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings
from app.core.logger import get_logger
from app.services.document_processor import TextChunk

logger = get_logger(__name__)
settings = get_settings()


class VectorStoreService:
    """
    Manages ChromaDB collection for RAG document storage.
    Uses local SentenceTransformer embeddings (no external API cost).
    """

    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._collection = None
        self._embedding_model: Optional[SentenceTransformer] = None
        self._doc_registry: Dict[str, dict] = {}  # file_id → metadata

    def _get_client(self) -> chromadb.Client:
        if self._client is None:
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB initialized at: {settings.CHROMA_PERSIST_DIR}")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(
                f"Collection '{settings.CHROMA_COLLECTION_NAME}' ready. "
                f"Items: {self._collection.count()}"
            )
        return self._collection

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded ✅")
        return self._embedding_model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of texts."""
        model = self._get_embedding_model()
        embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
        return embeddings.tolist()

    def add_chunks(self, chunks: List[TextChunk], filename: str) -> int:
        """
        Add document chunks to ChromaDB.
        Returns number of chunks added.
        """
        if not chunks:
            return 0

        collection = self._get_collection()

        texts = [c.content for c in chunks]
        ids = [f"{c.file_id}__{c.chunk_index}" for c in chunks]
        metadatas = [
            {
                "source": c.source,
                "file_id": c.file_id,
                "chunk_index": c.chunk_index,
                "page": c.page,
                "filename": filename,
                "uploaded_at": datetime.utcnow().isoformat(),
            }
            for c in chunks
        ]

        # Embed in batches of 64 to avoid memory issues
        all_embeddings = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            all_embeddings.extend(self._embed(batch))

        # Add to ChromaDB
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=all_embeddings,
            metadatas=metadatas,
        )

        # Register document
        file_id = chunks[0].file_id
        self._doc_registry[file_id] = {
            "file_id": file_id,
            "filename": filename,
            "chunks": len(chunks),
            "uploaded_at": datetime.utcnow().isoformat(),
        }

        logger.info(f"Added {len(chunks)} chunks for '{filename}' (file_id={file_id})")
        return len(chunks)

    def query(
        self,
        question: str,
        file_ids: Optional[List[str]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Semantic similarity search against stored chunks.
        Optionally filter by file_ids.
        Returns list of result dicts with content, metadata, and score.
        """
        collection = self._get_collection()

        if collection.count() == 0:
            logger.warning("Collection is empty — no documents indexed yet.")
            return []

        query_embedding = self._embed([question])[0]

        # Build optional where filter
        where = None
        if file_ids:
            if len(file_ids) == 1:
                where = {"file_id": {"$eq": file_ids[0]}}
            else:
                where = {"file_id": {"$in": file_ids}}

        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            query_kwargs["where"] = where

        results = collection.query(**query_kwargs)

        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # Cosine distance → similarity score (0–1, higher = better)
            score = round(1 - dist, 4)
            output.append(
                {
                    "content": doc,
                    "source": meta.get("source", "unknown"),
                    "file_id": meta.get("file_id", ""),
                    "page": meta.get("page", 0),
                    "chunk_index": meta.get("chunk_index", 0),
                    "score": score,
                }
            )

        return output

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        Return a list of all distinct documents in the collection.
        """
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        # Get all items and deduplicate by file_id
        all_items = collection.get(include=["metadatas"])
        seen = {}
        for meta in all_items["metadatas"]:
            fid = meta.get("file_id", "unknown")
            if fid not in seen:
                seen[fid] = {
                    "file_id": fid,
                    "filename": meta.get("filename", meta.get("source", "unknown")),
                    "file_type": "." + meta.get("filename", ".txt").rsplit(".", 1)[-1]
                    if "." in meta.get("filename", "") else ".txt",
                    "chunks": 0,
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "status": "ready",
                }
            seen[fid]["chunks"] += 1

        return list(seen.values())

    def delete_document(self, file_id: str) -> bool:
        """Delete all chunks belonging to a file_id."""
        collection = self._get_collection()

        try:
            collection.delete(where={"file_id": {"$eq": file_id}})
            self._doc_registry.pop(file_id, None)
            logger.info(f"Deleted document: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {file_id}: {e}")
            return False

    def get_collection_count(self) -> int:
        """Return total number of chunks in the collection."""
        try:
            return self._get_collection().count()
        except Exception:
            return 0

    def clear_all(self) -> bool:
        """Delete the entire collection (nuclear option)."""
        try:
            client = self._get_client()
            client.delete_collection(settings.CHROMA_COLLECTION_NAME)
            self._collection = None
            self._doc_registry = {}
            logger.warning("All documents cleared from ChromaDB.")
            return True
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False


# Singleton instance
_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store

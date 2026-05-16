"""
ChromaDB vector store with local SentenceTransformer embeddings.
Data is persisted to ./data/chroma_db and optionally backed up to HF Dataset.
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
    def __init__(self):
        self._client: Optional[chromadb.Client] = None
        self._collection = None
        self._embedding_model: Optional[SentenceTransformer] = None

    def _get_client(self) -> chromadb.Client:
        if self._client is None:
            os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=settings.CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            logger.info(f"ChromaDB ready at: {settings.CHROMA_PERSIST_DIR}")
        return self._client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            self._collection = client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Collection ready — {self._collection.count()} chunks stored")
        return self._collection

    def _get_embedding_model(self) -> SentenceTransformer:
        if self._embedding_model is None:
            logger.info(f"Loading embedding model: {settings.EMBEDDING_MODEL}")
            self._embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
            logger.info("Embedding model loaded ✅")
        return self._embedding_model

    def _embed(self, texts: List[str]) -> List[List[float]]:
        model = self._get_embedding_model()
        return model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

    def add_chunks(self, chunks: List[TextChunk], filename: str) -> int:
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
        # Embed in batches of 64
        all_embeddings = []
        for i in range(0, len(texts), 64):
            all_embeddings.extend(self._embed(texts[i: i + 64]))

        collection.upsert(ids=ids, documents=texts, embeddings=all_embeddings, metadatas=metadatas)
        logger.info(f"Indexed {len(chunks)} chunks for '{filename}'")
        return len(chunks)

    def query(self, question: str, file_ids: Optional[List[str]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []

        query_embedding = self._embed([question])[0]
        where = None
        if file_ids:
            where = {"file_id": {"$eq": file_ids[0]}} if len(file_ids) == 1 else {"file_id": {"$in": file_ids}}

        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, collection.count()),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = collection.query(**kwargs)
        if not results["documents"] or not results["documents"][0]:
            return []

        output = []
        for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
            output.append({
                "content": doc,
                "source": meta.get("source", "unknown"),
                "file_id": meta.get("file_id", ""),
                "page": meta.get("page", 0),
                "chunk_index": meta.get("chunk_index", 0),
                "score": round(1 - dist, 4),
            })
        return output

    def list_documents(self) -> List[Dict[str, Any]]:
        collection = self._get_collection()
        if collection.count() == 0:
            return []
        all_items = collection.get(include=["metadatas"])
        seen = {}
        for meta in all_items["metadatas"]:
            fid = meta.get("file_id", "unknown")
            if fid not in seen:
                ext = "." + meta.get("filename", "file.txt").rsplit(".", 1)[-1] if "." in meta.get("filename", "") else ".txt"
                seen[fid] = {
                    "file_id": fid,
                    "filename": meta.get("filename", meta.get("source", "unknown")),
                    "file_type": ext,
                    "chunks": 0,
                    "uploaded_at": meta.get("uploaded_at", ""),
                    "status": "ready",
                }
            seen[fid]["chunks"] += 1
        return list(seen.values())

    def delete_document(self, file_id: str) -> bool:
        try:
            self._get_collection().delete(where={"file_id": {"$eq": file_id}})
            logger.info(f"Deleted document: {file_id}")
            return True
        except Exception as e:
            logger.error(f"Delete failed: {e}")
            return False

    def clear_all(self) -> bool:
        try:
            self._get_client().delete_collection(settings.CHROMA_COLLECTION_NAME)
            self._collection = None
            logger.warning("All documents cleared.")
            return True
        except Exception as e:
            logger.error(f"Clear failed: {e}")
            return False

    def get_collection_count(self) -> int:
        try:
            return self._get_collection().count()
        except Exception:
            return 0


_vector_store: Optional[VectorStoreService] = None


def get_vector_store() -> VectorStoreService:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStoreService()
    return _vector_store

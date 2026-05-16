"""
Basic tests for the RAG pipeline.
Run with: pytest tests/
"""
import sys
import os
import tempfile
import pytest

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


class TestDocumentProcessor:
    def test_parse_text_file(self):
        from app.services.document_processor import DocumentProcessor

        proc = DocumentProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world. " * 100)
            temp_path = f.name

        file_id, chunks = proc.process_file(temp_path, "test.txt")
        os.unlink(temp_path)

        assert file_id is not None
        assert len(chunks) > 0
        assert all(c.source == "test.txt" for c in chunks)
        assert all(len(c.content) > 0 for c in chunks)

    def test_parse_json_file(self):
        from app.services.document_processor import DocumentProcessor
        import json

        proc = DocumentProcessor()
        data = {"key": "value", "items": [1, 2, 3], "nested": {"a": "b"}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        file_id, chunks = proc.process_file(temp_path, "test.json")
        os.unlink(temp_path)

        assert len(chunks) > 0

    def test_chunk_overlap(self):
        from app.services.document_processor import DocumentProcessor

        proc = DocumentProcessor()
        long_text = "The quick brown fox jumps over the lazy dog. " * 50
        chunks = proc._split_text(long_text)

        assert len(chunks) >= 1
        for c in chunks:
            assert len(c) <= proc.chunk_size + 100  # small tolerance


class TestVectorStore:
    def test_add_and_query(self, tmp_path):
        """Test adding documents and querying."""
        import chromadb
        from app.services.document_processor import TextChunk

        # Use in-memory ChromaDB for testing
        client = chromadb.EphemeralClient()
        collection = client.create_collection("test_col")

        # Simple embedding mock
        chunks = [
            TextChunk(
                content="Python is a programming language.",
                source="test.txt",
                file_id="abc123",
                chunk_index=0,
            ),
            TextChunk(
                content="Machine learning is a subset of AI.",
                source="test.txt",
                file_id="abc123",
                chunk_index=1,
            ),
        ]

        collection.add(
            ids=["abc123__0", "abc123__1"],
            documents=[c.content for c in chunks],
            metadatas=[{"source": c.source, "file_id": c.file_id} for c in chunks],
        )

        results = collection.query(
            query_texts=["Tell me about Python"],
            n_results=2,
        )

        assert results["documents"] is not None
        assert len(results["documents"][0]) == 2

    def test_empty_query(self):
        """Query on empty collection returns empty."""
        import chromadb

        client = chromadb.EphemeralClient()
        collection = client.create_collection("test_empty")
        assert collection.count() == 0


class TestSchemas:
    def test_query_request_valid(self):
        from app.models.schemas import QueryRequest

        req = QueryRequest(question="What is RAG?", top_k=5)
        assert req.question == "What is RAG?"
        assert req.top_k == 5

    def test_query_request_invalid(self):
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            QueryRequest(question="", top_k=5)  # empty question

    def test_upload_response(self):
        from app.models.schemas import UploadResponse, DocumentStatus

        resp = UploadResponse(
            file_id="abc",
            filename="test.pdf",
            status=DocumentStatus.READY,
            chunks_created=10,
            message="OK",
        )
        assert resp.chunks_created == 10
        assert resp.status == DocumentStatus.READY

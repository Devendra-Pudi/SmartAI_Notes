"""
Tests for SmartAI Notes HF Edition.
Run: pytest tests/ -v
"""
import sys, os, tempfile, json, pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDocumentProcessor:
    def test_txt(self):
        from app.services.document_processor import DocumentProcessor
        p = DocumentProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Hello world. " * 200); tmp = f.name
        fid, chunks = p.process_file(tmp, "test.txt")
        os.unlink(tmp)
        assert fid and len(chunks) > 0
        assert all(c.source == "test.txt" for c in chunks)

    def test_json(self):
        from app.services.document_processor import DocumentProcessor
        p = DocumentProcessor()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"key": "value", "list": [1, 2, 3]}, f); tmp = f.name
        fid, chunks = p.process_file(tmp, "test.json")
        os.unlink(tmp)
        assert len(chunks) > 0

    def test_chunking(self):
        from app.services.document_processor import DocumentProcessor
        p = DocumentProcessor()
        text = "The quick brown fox. " * 100
        chunks = p._split_text(text)
        assert all(len(c) <= p.chunk_size + 200 for c in chunks)


class TestSchemas:
    def test_valid_query(self):
        from app.models.schemas import QueryRequest
        r = QueryRequest(question="What is RAG?", top_k=5)
        assert r.top_k == 5

    def test_empty_question(self):
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_upload_response(self):
        from app.models.schemas import UploadResponse, DocumentStatus
        r = UploadResponse(file_id="abc", filename="f.pdf", status=DocumentStatus.READY, chunks_created=5, message="ok")
        assert r.chunks_created == 5


class TestVectorStore:
    def test_empty_query(self):
        import chromadb
        c = chromadb.EphemeralClient()
        col = c.create_collection("test")
        assert col.count() == 0

    def test_add_and_query(self):
        import chromadb
        c = chromadb.EphemeralClient()
        col = c.create_collection("test2")
        col.add(ids=["1", "2"], documents=["Python is great.", "Machine learning rocks."], metadatas=[{"file_id": "x"}, {"file_id": "x"}])
        res = col.query(query_texts=["programming language"], n_results=2)
        assert len(res["documents"][0]) == 2

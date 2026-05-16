"""
Document processing — parse and chunk uploaded files.
Supports: PDF, TXT, MD, DOCX, CSV, JSON
"""
import os
import uuid
import json
import csv
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass, field

from app.core.config import get_settings
from app.core.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    content: str
    source: str
    file_id: str
    chunk_index: int
    page: int = 0
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def process_file(self, file_path: str, filename: str) -> Tuple[str, List[TextChunk]]:
        file_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()
        logger.info(f"Processing: {filename} ({ext})")

        if ext == ".pdf":
            raw_pages = self._parse_pdf(file_path)
        elif ext in (".txt", ".md"):
            raw_pages = self._parse_text(file_path)
        elif ext == ".docx":
            raw_pages = self._parse_docx(file_path)
        elif ext == ".csv":
            raw_pages = self._parse_csv(file_path)
        elif ext == ".json":
            raw_pages = self._parse_json(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        chunks = self._chunk_pages(raw_pages, file_id, filename)
        logger.info(f"Created {len(chunks)} chunks from '{filename}'")
        return file_id, chunks

    def _parse_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        try:
            import pypdf
            pages = []
            with open(file_path, "rb") as f:
                reader = pypdf.PdfReader(f)
                for i, page in enumerate(reader.pages):
                    text = page.extract_text() or ""
                    if text.strip():
                        pages.append((i + 1, text.strip()))
            return pages
        except Exception as e:
            raise RuntimeError(f"PDF parse error: {e}")

    def _parse_text(self, file_path: str) -> List[Tuple[int, str]]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return [(1, f.read())]

    def _parse_docx(self, file_path: str) -> List[Tuple[int, str]]:
        try:
            from docx import Document
            doc = Document(file_path)
            text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            return [(1, text)]
        except ImportError:
            raise ImportError("python-docx not installed")

    def _parse_csv(self, file_path: str) -> List[Tuple[int, str]]:
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for row in reader:
                rows.append(" | ".join(f"{k}: {v}" for k, v in row.items() if v))
        text = f"Columns: {', '.join(headers)}\n\n" + "\n".join(rows)
        return [(1, text)]

    def _parse_json(self, file_path: str) -> List[Tuple[int, str]]:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(1, json.dumps(data, indent=2))]

    def _chunk_pages(self, pages: List[Tuple[int, str]], file_id: str, filename: str) -> List[TextChunk]:
        chunks = []
        chunk_idx = 0
        for page_num, text in pages:
            for chunk_text in self._split_text(text):
                if chunk_text.strip():
                    chunks.append(TextChunk(
                        content=chunk_text.strip(),
                        source=filename,
                        file_id=file_id,
                        chunk_index=chunk_idx,
                        page=page_num,
                        metadata={
                            "source": filename,
                            "file_id": file_id,
                            "page": page_num,
                            "chunk_index": chunk_idx,
                        },
                    ))
                    chunk_idx += 1
        return chunks

    def _split_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        for sep in ["\n\n", "\n", ". ", " ", ""]:
            if sep and sep in text:
                parts = text.split(sep)
                current = ""
                for part in parts:
                    test = current + sep + part if current else part
                    if len(test) <= self.chunk_size:
                        current = test
                    else:
                        if current:
                            chunks.append(current)
                        overlap = current[-self.chunk_overlap:] if len(current) > self.chunk_overlap else current
                        current = overlap + sep + part if overlap else part
                if current:
                    chunks.append(current)
                return chunks or [text]

        return [text[i: i + self.chunk_size] for i in range(0, len(text), self.chunk_size - self.chunk_overlap)]

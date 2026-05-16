"""
Document processing service — loads, parses, and chunks uploaded files.
Supports: PDF, TXT, MD, DOCX, CSV, JSON
"""
import os
import uuid
import json
import csv
import io
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

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
    metadata: dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentProcessor:
    """Handles parsing and chunking of uploaded documents."""

    def __init__(self):
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    def process_file(self, file_path: str, filename: str) -> Tuple[str, List[TextChunk]]:
        """
        Process a file and return (file_id, list of TextChunk).
        """
        file_id = str(uuid.uuid4())
        ext = Path(filename).suffix.lower()

        logger.info(f"Processing file: {filename} (type: {ext})")

        try:
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
            logger.info(f"Created {len(chunks)} chunks from {filename}")
            return file_id, chunks

        except Exception as e:
            logger.error(f"Failed to process {filename}: {e}")
            raise

    def _parse_pdf(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse PDF, returns list of (page_num, text)."""
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
        except ImportError:
            # fallback: try pdfplumber
            try:
                import pdfplumber
                pages = []
                with pdfplumber.open(file_path) as pdf:
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text() or ""
                        if text.strip():
                            pages.append((i + 1, text.strip()))
                return pages
            except ImportError:
                raise ImportError("Install pypdf or pdfplumber: pip install pypdf")

    def _parse_text(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse plain text / markdown files."""
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        return [(1, text)]

    def _parse_docx(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse DOCX files."""
        try:
            from docx import Document
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            text = "\n\n".join(paragraphs)
            return [(1, text)]
        except ImportError:
            raise ImportError("Install python-docx: pip install python-docx")

    def _parse_csv(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse CSV — convert rows to readable text."""
        rows = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            reader = csv.DictReader(f)
            headers = reader.fieldnames or []
            for i, row in enumerate(reader):
                row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
                rows.append(row_text)
        text = f"Columns: {', '.join(headers)}\n\n" + "\n".join(rows)
        return [(1, text)]

    def _parse_json(self, file_path: str) -> List[Tuple[int, str]]:
        """Parse JSON — convert to indented string."""
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        text = json.dumps(data, indent=2)
        return [(1, text)]

    def _chunk_pages(
        self, pages: List[Tuple[int, str]], file_id: str, filename: str
    ) -> List[TextChunk]:
        """Split page texts into overlapping chunks."""
        chunks = []
        chunk_idx = 0

        for page_num, text in pages:
            page_chunks = self._split_text(text)
            for chunk_text in page_chunks:
                if chunk_text.strip():
                    chunks.append(
                        TextChunk(
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
                        )
                    )
                    chunk_idx += 1

        return chunks

    def _split_text(self, text: str) -> List[str]:
        """
        Smart recursive text splitter.
        Tries to split on paragraphs → sentences → characters.
        """
        if len(text) <= self.chunk_size:
            return [text]

        chunks = []
        # Try splitting by double newline (paragraphs)
        separators = ["\n\n", "\n", ". ", " ", ""]

        for sep in separators:
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
                        # Handle overlap
                        if len(current) > self.chunk_overlap:
                            overlap_text = current[-self.chunk_overlap:]
                            current = overlap_text + sep + part
                        else:
                            current = part
                if current:
                    chunks.append(current)
                return chunks if chunks else [text]

        # Force split by character if nothing else works
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunks.append(text[i : i + self.chunk_size])
        return chunks

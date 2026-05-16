"""
Configuration — reads from environment variables / HF Secrets.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "SmartAI Notes"
    APP_VERSION: str = "2.0.0"

    # OpenRouter
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_LLM_MODEL: str = "meta-llama/llama-3.3-8b-instruct:free"

    # Embeddings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB — stored inside HF Space persistent dir
    CHROMA_PERSIST_DIR: str = "./data/chroma_db"
    CHROMA_COLLECTION_NAME: str = "smartai_notes"

    # RAG
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5

    # Uploads (temp, HF Space ephemeral)
    UPLOAD_DIR: str = "./data/uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".md", ".docx", ".csv", ".json"]

    # HF Dataset persistence (optional — for cross-restart persistence)
    HF_TOKEN: str = ""
    HF_DATASET_REPO: str = ""   # e.g. "Devendra-Pudi/smartai-notes-db"
    USE_HF_PERSISTENCE: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

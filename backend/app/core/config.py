"""
Configuration settings for SmartAI Notes backend.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # App
    APP_NAME: str = "SmartAI Notes"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # OpenRouter API
    OPENROUTER_API_KEY: str = ""
    OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"
    DEFAULT_LLM_MODEL: str = "meta-llama/llama-3.3-8b-instruct:free"
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"

    # ChromaDB
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    CHROMA_COLLECTION_NAME: str = "smartai_notes"

    # RAG settings
    CHUNK_SIZE: int = 800
    CHUNK_OVERLAP: int = 150
    TOP_K_RESULTS: int = 5
    MAX_CONTEXT_TOKENS: int = 4000

    # File upload
    UPLOAD_DIR: str = "./uploads"
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_EXTENSIONS: list = [".pdf", ".txt", ".md", ".docx", ".csv", ".json"]

    # CORS
    ALLOWED_ORIGINS: list = ["http://localhost:8501", "http://127.0.0.1:8501"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()

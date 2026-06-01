from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    llm_provider: Literal["openai", "anthropic"] = "openai"
    openai_api_key: str = ""
    llm_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-3-5-sonnet-20241022"

    # RAG
    chunk_size: int = 800
    chunk_overlap: int = 150
    top_k_retrieval: int = 4

    # Vector store
    chroma_persist_dir: str = "./chroma_db"
    chroma_collection_name: str = "documents"

    # Database
    database_url: str = "postgresql://documind:documind_secret@localhost:5432/documind"

    # App
    app_env: str = "development"
    log_level: str = "INFO"
    max_upload_size_mb: int = 50
    allowed_extensions: str = "pdf,txt,md"

    @property
    def allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    return Settings()

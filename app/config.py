"""
Centralized application configuration using Pydantic BaseSettings.

All settings are loaded from environment variables and/or a .env file.
Import ``settings`` from this module everywhere instead of calling
``os.getenv()`` directly.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Set


class Settings(BaseSettings):
    """Application-wide settings – single source of truth."""

    # ── Server ───────────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    # ── Database ─────────────────────────────────────────────────────────
    db_path: str = "pma_metadata.db"
    schema_path: str = "app/storage/schema.sql"

    # ── ChromaDB ─────────────────────────────────────────────────────────
    chroma_persist_dir: str = "chroma_db"

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_batch_size: int = 64

    # ── Indexing ─────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 50
    max_file_size_mb: int = 50
    supported_extensions: str = ".txt,.md,.pdf,.docx,.csv,.json,.py,.js,.ts,.java,.c,.cpp,.rs,.go,.rb,.html,.css,.xml,.yaml,.yml,.toml,.ini,.cfg,.sh,.bat"
    index_concurrency: int = 4

    # ── LLM ──────────────────────────────────────────────────────────────
    gemini_api_key: str = ""
    gemini_model: str = "gemini-flash-latest"
    gemini_timeout: float = 30.0
    ollama_url: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3"
    ollama_timeout: float = 60.0

    # ── RAG / Retrieval ──────────────────────────────────────────────────
    rrf_fts_weight: float = 0.4
    rrf_semantic_weight: float = 0.6
    rrf_k: int = 60
    rrf_score_scale: int = 1000
    summary_boost_factor: float = 1.25
    retrieval_top_k: int = 10
    context_max_tokens: int = 4000

    # ── Dev / Feature Flags ──────────────────────────────────────────────
    dev_mode: bool = True  # Default on for hackathon; set to False for prod
    log_level: str = "INFO"

    @property
    def max_file_size_bytes(self) -> int:
        return self.max_file_size_mb * 1024 * 1024

    @property
    def extensions_set(self) -> Set[str]:
        return {e.strip() for e in self.supported_extensions.split(",") if e.strip()}

    model_config = {
        "env_prefix": "PMA_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton – import this everywhere
settings = Settings()

"""Runtime configuration, read from environment variables (or a .env file)."""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The pgvector column is declared as vector(512) in migrations/001_init.sql.
# Every embedding provider must produce vectors of exactly this size.
EMBEDDING_DIM = 512


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://docuchat:docuchat@localhost:5433/docuchat"
    # Optional Postgres schema to work in (used by tests/eval to isolate data).
    db_schema: str | None = None

    # Chunking
    chunk_size: int = Field(default=800, gt=50, description="max characters per chunk")
    chunk_overlap: int = Field(default=150, ge=0)
    max_upload_bytes: int = 10 * 1024 * 1024

    # Retrieval
    candidate_pool: int = Field(default=30, gt=0, description="candidates fetched per retriever")
    rrf_k: int = Field(default=60, gt=0)
    # Refuse to answer only when BOTH confidence signals are weak: the best
    # query-chunk cosine similarity is below `min_similarity` AND no retrieved
    # chunk contains at least `min_term_coverage` of the question's content
    # terms. Chosen with the sweep printed by `python -m eval.run_eval` for the
    # local embedder; re-tune when switching embedding models.
    min_similarity: float = Field(default=0.18, ge=-1.0, le=1.0)
    min_term_coverage: float = Field(default=0.5, ge=0.0, le=1.0)

    # Providers: "local" (offline, deterministic) or "openai" (any OpenAI-compatible API)
    embedding_provider: str = "local"
    llm_provider: str = "local"
    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str | None = None
    openai_embedding_model: str = "text-embedding-3-small"
    openai_chat_model: str = "gpt-4o-mini"
    openai_timeout_s: float = 60.0

    cors_origins: str = "http://localhost:3000"

    @model_validator(mode="after")
    def _check(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")
        for name in ("embedding_provider", "llm_provider"):
            if getattr(self, name) not in ("local", "openai"):
                raise ValueError(f"{name.upper()} must be 'local' or 'openai'")
        return self

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

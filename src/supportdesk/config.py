from datetime import date
from decimal import Decimal
from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model card: BAAI/bge-small-en-v1.5 outputs 384 dimensions and accepts 512 tokens.
# Revision is the Hub commit already in the local tokenizer cache. "main" is not allowed.
BGE_SMALL_DIMENSIONS = 384
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
# Hub id. model_info resolves the hyphenated L-6 name to this repo.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_REVISION = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
# USD per 1,000,000 tokens, read 2026-10-30. gemini-3.8-flash steps up after 2026-12-31.
PRICE_AS_OF = date(2026, 10, 30)
GENERATOR_PRICE_UNTIL = date(2026, 12, 31)
GEMINI_3_8_FLASH = "gemini-3.8-flash"
GEMINI_2_5_FLASH = "gemini-2.5-flash"
GEMINI_3_8_FLASH_INPUT_UNTIL = Decimal("0.75")
GEMINI_3_8_FLASH_OUTPUT_UNTIL = Decimal("3.75")
GEMINI_3_8_FLASH_INPUT_AFTER = Decimal("1.50")
GEMINI_3_8_FLASH_OUTPUT_AFTER = Decimal("7.50")
GEMINI_2_5_FLASH_INPUT = Decimal("0.30")
GEMINI_2_5_FLASH_OUTPUT = Decimal("2.50")
DEFAULT_JWT_SECRET = "supportdesk-dev-secret"
DEFAULT_CRM_API_KEY = "supportdesk-crm-dev"
CACHE_TTL_SECONDS = 86400


class Settings(BaseSettings):
    """Runtime configuration. Model ids are settings, not literals at the call site.

    The judge model must differ from the generator. The judge temperature is fixed
    at 0. A real Gemini generate calls client.models.list() once before
    generate_content and refuses if the generator id or the judge id is absent.
    Automatic function calling stays disabled. Tests inject a fake and do not call Gemini.

    Token prices are the module constants above, not environment variables.
    A short query uses cheap_model. The exact cache key includes the access scope
    and the corpus version. Semantic lookup is off unless semantic_cache_distance
    is set. When ENV is production, the default JWT secret and CRM key are refused.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    database_url: str = "postgresql+psycopg://supportdesk:supportdesk@localhost:5433/supportdesk"
    corpus_dir: str = "data/corpus"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_revision: str = EMBEDDING_REVISION
    embedding_dimensions: int = BGE_SMALL_DIMENSIONS
    rerank_enabled: bool = True
    rerank_model: str = RERANK_MODEL
    rerank_revision: str = RERANK_REVISION
    chunk_token_budget: int = 400
    chunk_overlap_tokens: int = 60
    generator_model: str = GEMINI_3_8_FLASH
    judge_model: str = GEMINI_2_5_FLASH
    cheap_model: str = GEMINI_2_5_FLASH
    simple_query_max_chars: int = 80
    judge_temperature: float = 0.0
    gemini_api_key: str = ""
    generator_temperature: float = 0.0
    env: str = "development"
    jwt_secret: str = DEFAULT_JWT_SECRET
    crm_base_url: str = "http://localhost:8001"
    crm_api_key: str = DEFAULT_CRM_API_KEY
    crm_timeout_seconds: float = 2.0
    crm_retry_attempts: int = 3
    crm_retry_backoff_seconds: float = 0.2
    llm_retry_attempts: int = 3
    llm_retry_backoff_seconds: float = 0.2
    circuit_failure_threshold: int = 3
    circuit_reset_seconds: float = 30.0
    semantic_cache_distance: float | None = None
    cache_ttl_seconds: int = CACHE_TTL_SECONDS
    max_context_tokens: int = 2000
    min_context_score: float | None = None

    @model_validator(mode="after")
    def check_models_and_chunks(self) -> "Settings":
        if self.judge_model == self.generator_model:
            raise ValueError("judge_model must be a different model id than generator_model")
        if self.judge_temperature != 0:
            raise ValueError("judge_temperature must be 0")
        if self.chunk_overlap_tokens < 0:
            raise ValueError("chunk_overlap_tokens must be >= 0")
        if self.chunk_overlap_tokens >= self.chunk_token_budget:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_token_budget")
        if self.embedding_dimensions != BGE_SMALL_DIMENSIONS:
            raise ValueError(
                "embedding_dimensions must stay 384 until a migration rebuilds the vector column"
            )
        if not self.jwt_secret.strip():
            raise ValueError("jwt_secret must not be empty")
        if not self.crm_api_key.strip():
            raise ValueError("crm_api_key must not be empty")
        if self.env == "production":
            if self.jwt_secret == DEFAULT_JWT_SECRET:
                raise ValueError("jwt_secret must not be the default when env is production")
            if self.crm_api_key == DEFAULT_CRM_API_KEY:
                raise ValueError("crm_api_key must not be the default when env is production")
        if self.crm_timeout_seconds <= 0:
            raise ValueError("crm_timeout_seconds must be > 0")
        if self.crm_retry_attempts < 1:
            raise ValueError("crm_retry_attempts must be >= 1")
        if self.crm_retry_backoff_seconds < 0:
            raise ValueError("crm_retry_backoff_seconds must be >= 0")
        if self.llm_retry_attempts < 1:
            raise ValueError("llm_retry_attempts must be >= 1")
        if self.llm_retry_backoff_seconds < 0:
            raise ValueError("llm_retry_backoff_seconds must be >= 0")
        if self.circuit_failure_threshold < 1:
            raise ValueError("circuit_failure_threshold must be >= 1")
        if self.circuit_reset_seconds < 0:
            raise ValueError("circuit_reset_seconds must be >= 0")
        if not self.cheap_model.strip():
            raise ValueError("cheap_model must not be empty")
        if self.cheap_model == self.generator_model:
            raise ValueError("cheap_model must be a different model id than generator_model")
        if self.simple_query_max_chars < 0:
            raise ValueError("simple_query_max_chars must be >= 0")
        if self.semantic_cache_distance is not None and self.semantic_cache_distance < 0:
            raise ValueError("semantic_cache_distance must be >= 0")
        if self.cache_ttl_seconds < 1:
            raise ValueError("cache_ttl_seconds must be >= 1")
        if self.max_context_tokens < 1:
            raise ValueError("max_context_tokens must be >= 1")
        _require_commit(self.embedding_revision, "embedding_revision")
        _require_commit(self.rerank_revision, "rerank_revision")
        return self


def _require_commit(value: str, field: str) -> None:
    if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{field} must be a 40-character commit id, not a branch name")


@lru_cache
def get_settings() -> Settings:
    return Settings()

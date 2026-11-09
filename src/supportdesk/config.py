from functools import lru_cache

from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Model card: BAAI/bge-small-en-v1.5 outputs 384 dimensions and accepts 512 tokens.
# Revision is the Hub commit already in the local tokenizer cache. "main" is not allowed.
BGE_SMALL_DIMENSIONS = 384
EMBEDDING_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"


class Settings(BaseSettings):
    """Runtime configuration. Model ids are settings, not literals at the call site.

    The judge model must differ from the generator. The judge temperature is fixed
    at 0. The first Gemini call checks both ids with client.models.list() and
    refuses to generate if either id is absent. This phase does not call Gemini.
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
    chunk_token_budget: int = 400
    chunk_overlap_tokens: int = 60
    generator_model: str = "gemini-3.8-flash"
    judge_model: str = "gemini-2.5-flash"
    judge_temperature: float = 0.0
    gemini_api_key: str = ""

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
        if len(self.embedding_revision) != 40 or any(
            char not in "0123456789abcdef" for char in self.embedding_revision
        ):
            raise ValueError(
                "embedding_revision must be a 40-character commit id, not a branch name"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()

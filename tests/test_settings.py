import pytest
from pydantic import ValidationError

from supportdesk.config import BGE_SMALL_DIMENSIONS, EMBEDDING_REVISION, RERANK_REVISION, Settings


def test_judge_model_must_differ_from_generator() -> None:
    with pytest.raises(ValidationError):
        Settings(generator_model="gemini-3.8-flash", judge_model="gemini-3.8-flash")


def test_judge_temperature_is_zero() -> None:
    with pytest.raises(ValidationError):
        Settings(judge_temperature=0.2)


def test_default_models_and_chunk_overlap() -> None:
    settings = Settings(
        generator_model="gemini-3.8-flash",
        judge_model="gemini-2.5-flash",
        judge_temperature=0,
        chunk_token_budget=400,
        chunk_overlap_tokens=60,
        embedding_dimensions=384,
    )
    assert settings.generator_model != settings.judge_model
    assert settings.judge_temperature == 0
    assert settings.embedding_dimensions == BGE_SMALL_DIMENSIONS
    assert settings.chunk_overlap_tokens / settings.chunk_token_budget == 0.15
    assert settings.embedding_revision == EMBEDDING_REVISION
    assert settings.rerank_revision == RERANK_REVISION
    assert settings.rerank_enabled is True


def test_embedding_revision_must_be_a_commit() -> None:
    with pytest.raises(ValidationError):
        Settings(embedding_revision="main")


def test_jwt_secret_must_not_be_empty() -> None:
    with pytest.raises(ValidationError):
        Settings(jwt_secret="  ")


def test_rerank_revision_must_be_a_commit() -> None:
    with pytest.raises(ValidationError):
        Settings(rerank_revision="main")


def test_crm_retry_settings() -> None:
    settings = Settings()
    assert settings.crm_base_url == "http://localhost:8001"
    assert settings.crm_retry_attempts == 3
    assert settings.crm_timeout_seconds == 2.0
    with pytest.raises(ValidationError):
        Settings(crm_api_key="  ")
    with pytest.raises(ValidationError):
        Settings(crm_timeout_seconds=0)
    with pytest.raises(ValidationError):
        Settings(crm_retry_attempts=0)

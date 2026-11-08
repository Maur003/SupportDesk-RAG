import pytest
from pydantic import ValidationError

from supportdesk.config import BGE_SMALL_DIMENSIONS, Settings


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

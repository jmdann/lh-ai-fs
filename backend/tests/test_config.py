"""Tests for backend.config.Settings — minimal coverage of the twelve-factor
III boundary: required api key, sane defaults, env overrides, secret masking,
and rejection of out-of-bounds values."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from backend.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_cache() -> None:
    get_settings.cache_clear()


def _build(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    for key in [
        "OPENAI_API_KEY",
        "OPENAI_MODEL",
        "OPENAI_TEMPERATURE",
        "EVAL_SEED",
        "LOG_LEVEL",
    ]:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    class _Settings(Settings):
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    return _Settings()


def test_api_key_required(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    class _Settings(Settings):
        model_config = SettingsConfigDict(env_file=None, extra="ignore")

    with pytest.raises(ValidationError, match="openai_api_key"):
        _Settings()


def test_defaults_with_only_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _build({"OPENAI_API_KEY": "sk-test"}, monkeypatch)
    assert s.openai_model == "gpt-4o-2024-08-06"
    assert s.openai_temperature == 0.0
    assert s.eval_seed == 42
    assert s.log_level == "INFO"


def test_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _build(
        {
            "OPENAI_API_KEY": "sk-test",
            "OPENAI_MODEL": "gpt-5-pinned",
            "OPENAI_TEMPERATURE": "0.7",
            "LOG_LEVEL": "DEBUG",
        },
        monkeypatch,
    )
    assert s.openai_model == "gpt-5-pinned"
    assert s.openai_temperature == 0.7
    assert s.log_level == "DEBUG"


def test_secret_str_masks_in_repr_and_json(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _build({"OPENAI_API_KEY": "sk-supersecret"}, monkeypatch)
    assert "sk-supersecret" not in repr(s)
    assert "sk-supersecret" not in s.model_dump_json()
    assert s.openai_api_key.get_secret_value() == "sk-supersecret"


def test_rejects_out_of_bounds(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(ValidationError):
        _build({"OPENAI_API_KEY": "sk-test", "OPENAI_TEMPERATURE": "2.5"}, monkeypatch)
    with pytest.raises(ValidationError):
        _build({"OPENAI_API_KEY": "sk-test", "EVAL_SEED": "-1"}, monkeypatch)
    with pytest.raises(ValidationError):
        _build({"OPENAI_API_KEY": "sk-test", "LOG_LEVEL": "TRACE"}, monkeypatch)


def test_extra_env_vars_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _build({"OPENAI_API_KEY": "sk-test", "UNRELATED": "noise"}, monkeypatch)
    assert isinstance(s, Settings)

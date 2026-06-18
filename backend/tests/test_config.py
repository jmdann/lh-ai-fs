"""Tests for backend.config.Settings — the twelve-factor III boundary.

The rule we are defending: every knob comes from the environment, defaults
are sane, secrets never leak through repr/dict, and the cached factory does
not silently serve stale state across tests.
"""

from __future__ import annotations

import pytest
from pydantic import SecretStr, ValidationError
from pydantic_settings import SettingsConfigDict

from backend.config import LogLevel, Settings, get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> None:
    get_settings.cache_clear()


def _build(env: dict[str, str], monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Build a Settings ignoring any .env file so tests are deterministic."""
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
        model_config = SettingsConfigDict(
            env_file=None,
            extra="ignore",
            case_sensitive=False,
            frozen=True,
        )

    return _Settings()


class TestRequiredFields:
    def test_api_key_required(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        class _Settings(Settings):
            model_config = SettingsConfigDict(env_file=None, extra="ignore", frozen=True)

        with pytest.raises(ValidationError, match="openai_api_key"):
            _Settings()


class TestDefaults:
    def test_defaults_when_only_api_key_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build({"OPENAI_API_KEY": "sk-test"}, monkeypatch)
        assert s.openai_model == "gpt-4o-2024-08-06"
        assert s.openai_temperature == 0.0
        assert s.eval_seed == 42
        assert s.log_level == "INFO"


class TestEnvOverrides:
    def test_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build(
            {"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": "gpt-5-pinned"},
            monkeypatch,
        )
        assert s.openai_model == "gpt-5-pinned"

    def test_temperature_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build(
            {"OPENAI_API_KEY": "sk-test", "OPENAI_TEMPERATURE": "0.7"},
            monkeypatch,
        )
        assert s.openai_temperature == 0.7

    def test_log_level_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build({"OPENAI_API_KEY": "sk-test", "LOG_LEVEL": "DEBUG"}, monkeypatch)
        assert s.log_level == "DEBUG"

    def test_log_level_lowercase_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # case_sensitive=False applies to env-var KEYS (LOG_LEVEL vs log_level),
        # not VALUES. The Literal is exact — "debug" lowercased is not "DEBUG".
        # If we wanted permissive levels we would coerce via a validator, but
        # the contract is "use the stdlib names verbatim".
        with pytest.raises(ValidationError):
            _build({"OPENAI_API_KEY": "sk-test", "LOG_LEVEL": "debug"}, monkeypatch)

    def test_extra_env_vars_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build(
            {"OPENAI_API_KEY": "sk-test", "TOTALLY_UNRELATED": "garbage"},
            monkeypatch,
        )
        assert isinstance(s, Settings)


class TestValidation:
    def test_empty_model_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            _build({"OPENAI_API_KEY": "sk-test", "OPENAI_MODEL": ""}, monkeypatch)

    def test_temperature_lower_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            _build(
                {"OPENAI_API_KEY": "sk-test", "OPENAI_TEMPERATURE": "-0.1"},
                monkeypatch,
            )

    def test_temperature_upper_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            _build({"OPENAI_API_KEY": "sk-test", "OPENAI_TEMPERATURE": "2.5"}, monkeypatch)

    def test_eval_seed_negative_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            _build({"OPENAI_API_KEY": "sk-test", "EVAL_SEED": "-1"}, monkeypatch)

    def test_log_level_unknown_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        with pytest.raises(ValidationError):
            _build({"OPENAI_API_KEY": "sk-test", "LOG_LEVEL": "TRACE"}, monkeypatch)


class TestSecretMasking:
    def test_api_key_is_secret_str(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build({"OPENAI_API_KEY": "sk-supersecret"}, monkeypatch)
        assert isinstance(s.openai_api_key, SecretStr)

    def test_repr_does_not_leak_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build({"OPENAI_API_KEY": "sk-supersecret"}, monkeypatch)
        assert "sk-supersecret" not in repr(s)

    def test_model_dump_json_does_not_leak_secret(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build({"OPENAI_API_KEY": "sk-supersecret"}, monkeypatch)
        assert "sk-supersecret" not in s.model_dump_json()

    def test_secret_is_recoverable_via_get_secret_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        s = _build({"OPENAI_API_KEY": "sk-supersecret"}, monkeypatch)
        assert s.openai_api_key.get_secret_value() == "sk-supersecret"


class TestImmutability:
    def test_frozen(self, monkeypatch: pytest.MonkeyPatch) -> None:
        s = _build({"OPENAI_API_KEY": "sk-test"}, monkeypatch)
        with pytest.raises(ValidationError):
            s.openai_model = "other"  # type: ignore[misc]


class TestGetSettingsFactory:
    def test_get_settings_is_cached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        get_settings.cache_clear()
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_get_settings_cache_clear_re_reads_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-first")
        get_settings.cache_clear()
        first = get_settings()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-second")
        get_settings.cache_clear()
        second = get_settings()
        assert first is not second
        assert second.openai_api_key.get_secret_value() == "sk-second"


class TestLogLevelType:
    def test_loglevel_alias_covers_stdlib_levels(self) -> None:
        # Smoke check that the Literal lines up with logging stdlib names.
        import logging

        for name in LogLevel.__args__:  # type: ignore[attr-defined]
            assert logging.getLevelName(name) != f"Level {name}"

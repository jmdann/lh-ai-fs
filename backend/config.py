"""Application configuration. Twelve-Factor III: every knob comes from
the environment, none are hardcoded.

The Settings object is constructed once per process via ``get_settings``
and injected through FastAPI ``Depends`` at the API edge. Agents take
what they need through their constructors — they never read
``os.environ`` directly."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    openai_api_key: SecretStr = Field(description="Required at startup; missing = fail fast.")
    openai_model: str = Field(default="gpt-4o-2024-08-06", min_length=1)
    openai_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    eval_seed: int = Field(default=42, ge=0)
    log_level: LogLevel = "INFO"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """FastAPI dependency. Cached because Settings is environment-pinned."""
    return Settings()

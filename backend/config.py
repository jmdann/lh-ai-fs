"""Application configuration. Twelve-Factor III: every knob comes from the
environment, none are hardcoded.

The Settings object is constructed once (``get_settings``) and injected
through FastAPI ``Depends`` at the API edge. Agents take what they need
through their constructors — they never read ``os.environ`` directly.

See specs/001-foundation-evals-crossdoc/spec.md § 8 and STANDARDS.md § 2
(twelve-factor table) for the rules these fields implement.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]


class Settings(BaseSettings):
    """Pinned at process start; treat as immutable for the request lifetime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        frozen=True,
    )

    openai_api_key: SecretStr = Field(
        description="OpenAI API key. Required at startup; missing = fail fast."
    )
    openai_model: str = Field(
        default="gpt-4o-2024-08-06",
        description=(
            "Pinned model id. Same model is used by the API and by run_evals.py "
            "so dev == eval == prod (twelve-factor X)."
        ),
        min_length=1,
    )
    openai_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature; 0.0 unless intentionally varied for ablations.",
    )
    eval_seed: int = Field(
        default=42,
        ge=0,
        description="Seed for any eval-side sampling (fixture shuffle, gold-set order).",
    )
    log_level: LogLevel = Field(
        default="INFO",
        description="Threshold for the structured JSON logger.",
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """FastAPI dependency. Cached because Settings is environment-pinned;
    if the env changes mid-process, that is a deployment bug, not a feature."""
    return Settings()

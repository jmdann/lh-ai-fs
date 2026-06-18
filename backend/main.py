"""FastAPI entrypoint. STANDARDS § 3.2: ``Depends`` lives ONLY at this
edge; nothing inside ``agents/`` or ``orchestrator.py`` imports FastAPI.
That means swapping the LLM provider (or wiring a fake for tests via
``app.dependency_overrides``) is a one-liner."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import Settings, get_settings
from backend.documents_io import load_case_documents
from backend.llm.client import LLMClient, OpenAIClient
from backend.models import Document, VerificationReport
from backend.observability import configure_logging
from backend.orchestrator import Orchestrator
from backend.sources import SourceRegistry

CASE_NAME = "Rivera v. Harmon Construction Group"


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Twelve-Factor XI: configure structured logging exactly once at boot."""
    configure_logging(get_settings().log_level)
    yield


app = FastAPI(title="BS Detector", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5175"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependencies (every Depends() in this module lives here) ──────────────


def get_llm_client(
    settings: Annotated[Settings, Depends(get_settings)],
) -> LLMClient:
    """LLM client factory. ``Settings`` is cached so the same OpenAI client
    is reused across requests within a process."""
    return OpenAIClient(settings)


@lru_cache(maxsize=1)
def _load_documents_cached() -> dict[str, Document]:
    return load_case_documents()


def get_documents() -> dict[str, Document]:
    return _load_documents_cached()


def get_orchestrator(
    llm: Annotated[LLMClient, Depends(get_llm_client)],
) -> Orchestrator:
    return Orchestrator(llm, SourceRegistry())


# ── Routes ─────────────────────────────────────────────────────────────────


@app.post("/analyze", response_model=VerificationReport)
async def analyze(
    orchestrator: Annotated[Orchestrator, Depends(get_orchestrator)],
    documents: Annotated[dict[str, Document], Depends(get_documents)],
) -> VerificationReport:
    return await orchestrator.run(documents, case_name=CASE_NAME)


@app.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}

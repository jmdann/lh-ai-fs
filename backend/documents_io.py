"""Loads the four case-file documents from disk into typed ``Document``
objects. Kept separate from ``main.py`` so the loader has unit tests and
the API stays focused on routing."""

from __future__ import annotations

from pathlib import Path

from backend.models import Document, DocumentKind

_DOCUMENTS_DIR = Path(__file__).parent / "documents"

# Filename (without extension) → (Document.id, DocumentKind). The id and the
# DocumentKind.value happen to coincide today; keeping them as a single
# source of truth here so the orchestrator can use either lookup direction.
_FILENAME_TO_DOC: dict[str, tuple[str, DocumentKind]] = {
    "motion_for_summary_judgment": ("motion", DocumentKind.MOTION),
    "police_report": ("police_report", DocumentKind.POLICE_REPORT),
    "medical_records_excerpt": ("medical_records_excerpt", DocumentKind.MEDICAL_RECORDS),
    "witness_statement": ("witness_statement", DocumentKind.WITNESS_STATEMENT),
}


def load_case_documents(documents_dir: Path | None = None) -> dict[str, Document]:
    """Read every recognized ``.txt`` file under ``documents_dir`` and return
    a mapping keyed by ``Document.id``.

    Files that don't match the recognized list are skipped silently — this
    keeps the loader robust if a stray README or hidden file lands in the
    directory. Missing files raise ``FileNotFoundError`` with a clear
    message so deploys fail fast."""
    directory = documents_dir or _DOCUMENTS_DIR
    out: dict[str, Document] = {}
    for stem, (doc_id, kind) in _FILENAME_TO_DOC.items():
        path = directory / f"{stem}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Expected case-file document not found: {path}")
        out[doc_id] = Document(id=doc_id, kind=kind, text=path.read_text())
    return out

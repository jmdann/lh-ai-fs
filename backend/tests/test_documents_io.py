"""Tests for ``backend.documents_io.load_case_documents``."""

from __future__ import annotations

from pathlib import Path

import pytest

from backend.documents_io import load_case_documents
from backend.models import DocumentKind


def test_loads_all_four_real_documents() -> None:
    docs = load_case_documents()
    assert set(docs.keys()) == {
        "motion",
        "police_report",
        "medical_records_excerpt",
        "witness_statement",
    }
    assert docs["motion"].kind is DocumentKind.MOTION
    assert docs["police_report"].kind is DocumentKind.POLICE_REPORT
    assert docs["medical_records_excerpt"].kind is DocumentKind.MEDICAL_RECORDS
    assert docs["witness_statement"].kind is DocumentKind.WITNESS_STATEMENT
    for doc in docs.values():
        assert len(doc.text) > 0


def test_missing_file_raises_clear_error(tmp_path: Path) -> None:
    # Empty directory — no motion file present.
    with pytest.raises(FileNotFoundError, match="motion_for_summary_judgment"):
        load_case_documents(tmp_path)


def test_extra_files_ignored(tmp_path: Path) -> None:
    # Seed all four expected files plus a stray README.
    for stem in [
        "motion_for_summary_judgment",
        "police_report",
        "medical_records_excerpt",
        "witness_statement",
    ]:
        (tmp_path / f"{stem}.txt").write_text(f"body of {stem}")
    (tmp_path / "README.md").write_text("ignored")
    (tmp_path / "stray.txt").write_text("also ignored")

    docs = load_case_documents(tmp_path)
    assert set(docs.keys()) == {
        "motion",
        "police_report",
        "medical_records_excerpt",
        "witness_statement",
    }

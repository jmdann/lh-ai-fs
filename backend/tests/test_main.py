"""End-to-end test for the ``/analyze`` route.

Uses FastAPI's ``app.dependency_overrides`` to swap ``get_llm_client`` for a
``FakeLLMClient``, so the test never hits the network. STANDARDS § 3.2:
``Depends`` only at the API edge — this is what makes the swap one line.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from backend.agents.citation_extractor import (
    _ExtractedCitations,
    _LLMCitation,
    extract_authority_strings,
)
from backend.agents.cross_doc_checker import (
    _ExtractedDiscrepancies,
    _LLMFactDiscrepancy,
)
from backend.agents.prompts.citation_extractor import SYSTEM as CITATION_SYSTEM
from backend.agents.prompts.citation_extractor import (
    build_user_prompt as build_citation_prompt,
)
from backend.agents.prompts.cross_doc_checker import SYSTEM as CROSS_DOC_SYSTEM
from backend.agents.prompts.cross_doc_checker import (
    build_user_prompt as build_cross_doc_prompt,
)
from backend.llm.client import FakeLLMClient, LLMClient
from backend.main import app, get_llm_client, load_case_documents
from backend.models import Document, Span


@pytest.fixture
def real_documents() -> dict[str, Document]:
    return load_case_documents()


@pytest.fixture
def seeded_fake(real_documents: dict[str, Document]) -> FakeLLMClient:
    """A FakeLLMClient pre-seeded with one citation + one fact discrepancy
    grounded in the real case-file documents so the orchestrator's
    grounding boundary accepts both."""
    fake = FakeLLMClient()
    motion = real_documents["motion"]

    # Citation: use a real authority string from the motion + a sentence
    # that actually appears in the motion text so it grounds.
    citation_candidates = extract_authority_strings(motion.text)
    citation_sentence = "Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)."
    fake.queue(
        system=CITATION_SYSTEM,
        user=build_citation_prompt(motion.text, citation_candidates),
        response=_ExtractedCitations(
            citations=[
                _LLMCitation(
                    cited_authority="Privette v. Superior Court, 5 Cal.4th 689, 695 (1993)",
                    proposition=Span(doc_id="motion", quote=citation_sentence),
                    quoted_text=None,
                )
            ]
        ),
    )

    # Cross-doc: motion's harness claim vs. police report's contrary
    # observation. Both quotes are substrings of the respective docs.
    record_triples = [
        ("police_report", "POLICE REPORT", real_documents["police_report"].text),
        (
            "medical_records_excerpt",
            "MEDICAL RECORDS",
            real_documents["medical_records_excerpt"].text,
        ),
        ("witness_statement", "WITNESS STATEMENT", real_documents["witness_statement"].text),
    ]
    fake.queue(
        system=CROSS_DOC_SYSTEM,
        user=build_cross_doc_prompt("motion", motion.text, record_triples),
        response=_ExtractedDiscrepancies(
            discrepancies=[
                _LLMFactDiscrepancy(
                    motion_claim=Span(doc_id="motion", quote="Rivera was a journeyman scaffolder"),
                    contradicting_evidence=[Span(doc_id="police_report", quote="Rivera")],
                    description="Test discrepancy",
                    severity="minor",
                )
            ]
        ),
    )
    return fake


@pytest.fixture
def client(seeded_fake: FakeLLMClient) -> Iterator[TestClient]:
    def _override() -> LLMClient:
        return seeded_fake

    app.dependency_overrides[get_llm_client] = _override
    yield TestClient(app)
    app.dependency_overrides.clear()


class TestAnalyzeRoute:
    def test_returns_200_with_verification_report_shape(self, client: TestClient) -> None:
        response = client.post("/analyze")
        assert response.status_code == 200, response.text
        payload = response.json()
        assert payload["case_name"] == "Rivera v. Harmon Construction Group"
        assert "generated_at" in payload
        assert isinstance(payload["citations"], list)
        assert isinstance(payload["findings"], list)
        assert isinstance(payload["agent_results"], list)
        # All six agent_results present (spec 003 adds confidence + memo).
        kinds = {r["kind"] for r in payload["agent_results"]}
        assert kinds == {
            "citations",
            "discrepancies",
            "quote_check",
            "authority_check",
            "confidence",
            "memo",
        }

    def test_agent_results_include_latency(self, client: TestClient) -> None:
        response = client.post("/analyze")
        payload = response.json()
        for ar in payload["agent_results"]:
            assert "latency_ms" in ar
            assert ar["latency_ms"] >= 0

    def test_health_check(self, client: TestClient) -> None:
        response = client.get("/healthz")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

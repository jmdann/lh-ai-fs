"""Stable intermediate representation for the BS Detector pipeline.

Every agent consumes and emits these Pydantic v2 models. Raw text never
crosses an agent boundary. See specs/001-foundation-evals-crossdoc/spec.md § 4
and STANDARDS.md §§ 3.1, 3.6 for the load-bearing decisions:

* LLMs emit ``Span(doc_id, quote)``; the orchestrator grounds to offsets
  in code via ``SourceRegistry.find``. ``start`` / ``end`` are derived.
* ``AgentResult`` is a discriminated union, not ``Generic[T]`` — Pydantic v2
  schema generation is cleaner that way and OpenAPI consumers benefit.
* The full IR ships in PR [1/3] (including spec-002 envelopes like
  ``QuoteCheck`` and ``AuthorityCheck``) so the type surface does not
  churn across PRs.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# ── Documents ──────────────────────────────────────────────────────────────


class DocumentKind(StrEnum):
    MOTION = "motion_for_summary_judgment"
    POLICE_REPORT = "police_report"
    MEDICAL_RECORDS = "medical_records_excerpt"
    WITNESS_STATEMENT = "witness_statement"
    EXTERNAL_AUTHORITY = "external_authority"


class Document(BaseModel):
    """A loaded source document. ``id`` is the stable identifier used by Spans;
    ``kind`` is the document type for prompt construction and agent routing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1, description="Stable id, e.g. 'motion' or 'auth:privette'.")
    kind: DocumentKind
    text: str = Field(min_length=1)


# ── Spans: quote is authoritative, offsets are derived ────────────────────


class Span(BaseModel):
    """A pointer into a document.

    The LLM emits ``doc_id`` and ``quote``. The orchestrator fills ``start`` and
    ``end`` after grounding the quote via ``SourceRegistry.find``. Ungrounded
    spans are dropped at the orchestrator boundary; see STANDARDS § 3.6.
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _offsets_consistent(self) -> Span:
        if (self.start is None) != (self.end is None):
            raise ValueError("start and end must both be set or both be None")
        if self.start is not None and self.end is not None and self.end <= self.start:
            raise ValueError("end must be strictly greater than start")
        return self

    @property
    def is_grounded(self) -> bool:
        return self.start is not None and self.end is not None


# ── Claims, Citations, Quotes ─────────────────────────────────────────────


class Claim(BaseModel):
    """A factual proposition asserted in the motion.

    Spec 001 emits ``FactDiscrepancy`` directly without a Claim hop; the type
    exists now so spec 002's ``ClaimExtractor`` (if it lands) can populate it
    without churning the IR.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^claim-\d+$")
    proposition: Span


class Citation(BaseModel):
    """A legal authority cited by the motion. Spec 001 emits these; spec 002
    feeds them to ``QuoteChecker`` and ``AuthoritySupportChecker``."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^cite-\d+$")
    proposition: Span
    cited_authority: str = Field(min_length=1)
    quoted_text: str | None = None


class Quote(BaseModel):
    """A direct quotation attributed to a cited authority. Spec 002 populates."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(pattern=r"^cite-\d+$")
    text: str = Field(min_length=1)
    attributed_doc_hint: str | None = None


QuoteVerdict = Literal["exact", "paraphrase", "altered", "fabricated", "unverifiable"]


class QuoteCheck(BaseModel):
    """``QuoteChecker`` (spec 002) output. Validators enforce the verdict /
    matched_span pairing so an LLM cannot claim ``exact`` without a span."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(pattern=r"^cite-\d+$")
    verdict: QuoteVerdict
    matched_span: Span | None = None
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _verdict_span_pairing(self) -> QuoteCheck:
        needs_span = {"exact", "paraphrase", "altered"}
        if self.verdict in needs_span and self.matched_span is None:
            raise ValueError(f"verdict={self.verdict} requires matched_span")
        if self.verdict in {"fabricated", "unverifiable"} and self.matched_span is not None:
            raise ValueError(f"verdict={self.verdict} forbids matched_span")
        return self


AuthorityVerdict = Literal["supports", "contradicts", "unverifiable"]


class AuthorityCheck(BaseModel):
    """``AuthoritySupportChecker`` (spec 002) output. ``unverifiable`` is a
    first-class outcome; the validator forces the agent to ground every
    supports/contradicts judgment in an actual source span."""

    model_config = ConfigDict(extra="forbid")

    citation_id: str = Field(pattern=r"^cite-\d+$")
    verdict: AuthorityVerdict
    source_basis: Span | None = None
    reasoning: str = Field(min_length=1)

    @model_validator(mode="after")
    def _verdict_basis_pairing(self) -> AuthorityCheck:
        if self.verdict in {"supports", "contradicts"} and self.source_basis is None:
            raise ValueError(f"verdict={self.verdict} requires source_basis")
        if self.verdict == "unverifiable" and self.source_basis is not None:
            raise ValueError("verdict=unverifiable forbids source_basis")
        return self


class EvidenceRef(BaseModel):
    """A typed reference from a Finding to its supporting span."""

    model_config = ConfigDict(extra="forbid")

    span: Span
    role: Literal["primary", "supporting", "contradicting"]


# ── Findings ──────────────────────────────────────────────────────────────


class FindingKind(StrEnum):
    FACT_DISCREPANCY = "fact_discrepancy"
    QUOTE_ALTERED = "quote_altered"
    QUOTE_FABRICATED = "quote_fabricated"
    AUTHORITY_UNSUPPORTED = "authority_unsupported"
    AUTHORITY_UNVERIFIABLE = "authority_unverifiable"


Severity = Literal["minor", "material", "dispositive"]


class FactDiscrepancy(BaseModel):
    """``CrossDocConsistencyChecker`` output: a motion claim contradicted by
    one or more record-document spans."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^disc-\d+$")
    motion_claim: Span
    contradicting_evidence: list[Span] = Field(min_length=1)
    description: str = Field(min_length=1)
    severity: Severity


class Finding(BaseModel):
    """A single verification finding. ``summary`` is human-readable and is
    excluded from eval matching to prevent prompt phrasing from leaking
    into the score (STANDARDS § 5)."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^find-\d+$")
    kind: FindingKind
    summary: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)
    agent: str = Field(min_length=1)
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    citation_id: str | None = Field(default=None, pattern=r"^cite-\d+$")
    claim_id: str | None = Field(default=None, pattern=r"^claim-\d+$")
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_reasoning: str | None = None

    @model_validator(mode="after")
    def _confidence_reasoning_required(self) -> Finding:
        if self.confidence is not None and not self.confidence_reasoning:
            raise ValueError("confidence_reasoning is required when confidence is set")
        return self


# ── Agent results: discriminated union, one class per kind ─────────────────


AgentOutcome = Literal["success", "failure", "partial", "timeout"]


class _AgentResultBase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    agent: str = Field(min_length=1)
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    outcome: AgentOutcome
    error: str | None = None
    latency_ms: int = Field(ge=0)


class CitationsResult(_AgentResultBase):
    kind: Literal["citations"] = "citations"
    data: list[Citation] = Field(default_factory=list)


class DiscrepanciesResult(_AgentResultBase):
    kind: Literal["discrepancies"] = "discrepancies"
    data: list[FactDiscrepancy] = Field(default_factory=list)


class QuoteCheckResult(_AgentResultBase):
    kind: Literal["quote_check"] = "quote_check"
    data: list[QuoteCheck] = Field(default_factory=list)


class AuthorityCheckResult(_AgentResultBase):
    kind: Literal["authority_check"] = "authority_check"
    data: list[AuthorityCheck] = Field(default_factory=list)


class ConfidenceResult(_AgentResultBase):
    kind: Literal["confidence"] = "confidence"
    data: list[Finding] = Field(default_factory=list)


class MemoResult(_AgentResultBase):
    kind: Literal["memo"] = "memo"
    data: str | None = None


AgentResult = Annotated[
    CitationsResult
    | DiscrepanciesResult
    | QuoteCheckResult
    | AuthorityCheckResult
    | ConfidenceResult
    | MemoResult,
    Field(discriminator="kind"),
]


# ── Report ────────────────────────────────────────────────────────────────


class VerificationReport(BaseModel):
    """The response shape of ``POST /analyze``. ``agent_results`` is the
    canonical log surface — every agent run shows up here regardless of
    whether its data is empty, so reviewers can debug a failed pipeline
    from the report alone."""

    model_config = ConfigDict(extra="forbid")

    case_name: str = Field(min_length=1)
    generated_at: datetime
    citations: list[Citation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    agent_results: list[AgentResult] = Field(default_factory=list)
    judicial_memo: str | None = None

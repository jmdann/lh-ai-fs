"""Stable intermediate representation for the BS Detector pipeline.

Every agent consumes and emits these Pydantic v2 models. Raw text never
crosses an agent boundary. See specs/001-foundation-evals-crossdoc/spec.md § 4
and STANDARDS.md §§ 3.1, 3.6 for the load-bearing decisions:

* ``Span`` is ``(doc_id, quote)``. No character offsets in the IR — the
  type system cannot be tricked by an LLM that emits integer offsets,
  because there are no integer fields. Grounding is exclusively a
  ``SourceRegistry`` boundary check inside the orchestrator: hit → keep,
  miss → drop the finding. Spec 003 may add a separate ``ResolvedSpan``
  type if UI rendering needs offsets; until then, offsets stay out.
* ``AgentResult`` is a discriminated union, not ``Generic[T]`` — Pydantic v2
  schema generation is cleaner that way and OpenAPI consumers benefit.
* The IR ships spec 001 + spec 002 surface in this PR. Spec 003 additions
  (``Finding.confidence``, ``judicial_memo``, ``ConfidenceResult``,
  ``MemoResult``, ``outcome="timeout"``) land in their owning PR — no
  forward-shipping of fields nothing in this PR populates.
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


# ── Spans: quote is the whole IR ──────────────────────────────────────────


class Span(BaseModel):
    """A pointer into a document.

    The LLM emits ``doc_id`` and ``quote``. There are no character offsets
    here on purpose — STANDARDS § 3.6 says the type system, not policy,
    enforces "LLMs can't emit offsets". Grounding is the orchestrator's
    job: ``SourceRegistry.find(doc_id, quote)`` either succeeds (keep the
    finding) or fails (drop it, log ``grounding_integrity_failure``).
    """

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)


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
    """``QuoteChecker`` (spec 002) output. The validator enforces the verdict /
    matched_span pairing so an LLM cannot claim ``exact`` without naming a
    span. Whether that span grounds in the source corpus is the orchestrator's
    concern, not the model's — keep type validation orthogonal to retrieval."""

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
    first-class outcome; the validator forces the agent to name a source span
    for every supports/contradicts judgment."""

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
    into the score (STANDARDS § 5).

    Confidence + reasoning are added in spec 003; they are intentionally
    absent from this PR's IR (no field populated by spec 001 / 002 agents
    means no field on the model).
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^find-\d+$")
    kind: FindingKind
    summary: str = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(min_length=1)
    agent: str = Field(min_length=1)
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    citation_id: str | None = Field(default=None, pattern=r"^cite-\d+$")
    claim_id: str | None = Field(default=None, pattern=r"^claim-\d+$")


# ── Agent results: discriminated union, one class per kind ─────────────────


AgentOutcome = Literal["success", "failure", "partial"]


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


AgentResult = Annotated[
    CitationsResult | DiscrepanciesResult | QuoteCheckResult | AuthorityCheckResult,
    Field(discriminator="kind"),
]


# ── Report ────────────────────────────────────────────────────────────────


class VerificationReport(BaseModel):
    """The response shape of ``POST /analyze``. ``agent_results`` is the
    canonical log surface — every agent run shows up here regardless of
    whether its data is empty, so reviewers can debug a failed pipeline
    from the report alone.

    Spec 003 adds ``judicial_memo``; the field is intentionally absent
    here because no spec 001 / 002 agent populates it.
    """

    model_config = ConfigDict(extra="forbid")

    case_name: str = Field(min_length=1)
    generated_at: datetime
    citations: list[Citation] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    agent_results: list[AgentResult] = Field(default_factory=list)

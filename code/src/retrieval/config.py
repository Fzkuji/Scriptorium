"""Validated retrieval settings."""

from dataclasses import dataclass

from .schemas import normalize_memory_components


SEARCH_TOOL_SETS = {
    "split": ("bm25_search", "embedding_search"),
    "fused": ("memory_search",),
}


@dataclass(frozen=True)
class QueryConfig:
    max_turns: int = 20
    max_budget_usd: float | None = None
    verify_sources: bool = True
    search_tools: str = "split"
    memory_components: tuple[str, ...] | str | None = None
    evidence_ledger_enabled: bool = False
    evidence_ledger_max_entries: int = 24
    reasoning_ledger_enabled: bool = False
    retrieval_plan_enabled: bool = False
    pipeline_enabled: bool = False
    pipeline_version: str = "p0-v2"
    pipeline_evidence_packet_enabled: bool = False
    pipeline_evidence_gate_enabled: bool = False
    pipeline_supplement_enabled: bool = False
    reflective_retrieval_enabled: bool = False
    adaptive_workspace_enabled: bool = False
    claim_evidence_state_enabled: bool = False
    claim_evidence_loop_enabled: bool = False
    claim_evidence_loop_initial_batch_size: int = 2
    claim_evidence_loop_initial_organizer_enabled: bool = True
    claim_evidence_loop_event_guidance_enabled: bool = False

    def __post_init__(self) -> None:
        if self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        if self.max_budget_usd is not None and self.max_budget_usd <= 0:
            raise ValueError("max_budget_usd must be positive")
        if self.search_tools not in SEARCH_TOOL_SETS:
            raise ValueError(
                "search_tools must be one of "
                + ", ".join(sorted(SEARCH_TOOL_SETS))
            )
        if self.evidence_ledger_max_entries < 1:
            raise ValueError("evidence_ledger_max_entries must be positive")
        if self.claim_evidence_loop_initial_batch_size < 1:
            raise ValueError(
                "claim-evidence loop initial batch size must be positive"
            )
        if self.reasoning_ledger_enabled and not self.evidence_ledger_enabled:
            raise ValueError("reasoning ledger requires evidence ledger")
        if self.retrieval_plan_enabled and not self.evidence_ledger_enabled:
            raise ValueError("retrieval plan requires evidence ledger")
        if self.pipeline_enabled and (
            self.evidence_ledger_enabled
            or self.reasoning_ledger_enabled
            or self.retrieval_plan_enabled
        ):
            raise ValueError("P0 pipeline cannot be combined with legacy ledgers")
        if self.pipeline_enabled and self.reflective_retrieval_enabled:
            raise ValueError("reflective retrieval cannot be combined with pipeline")
        if self.reflective_retrieval_enabled and (
            self.evidence_ledger_enabled
            or self.reasoning_ledger_enabled
            or self.retrieval_plan_enabled
        ):
            raise ValueError("reflective retrieval is an independent experiment")
        if self.adaptive_workspace_enabled and (
            self.pipeline_enabled
            or self.reflective_retrieval_enabled
            or self.evidence_ledger_enabled
            or self.reasoning_ledger_enabled
            or self.retrieval_plan_enabled
        ):
            raise ValueError("adaptive workspace is an independent A0 extension")
        if self.claim_evidence_state_enabled and (
            self.pipeline_enabled
            or self.reflective_retrieval_enabled
            or self.evidence_ledger_enabled
            or self.reasoning_ledger_enabled
            or self.retrieval_plan_enabled
            or self.adaptive_workspace_enabled
        ):
            raise ValueError("claim-evidence state is an independent A0 extension")
        if self.claim_evidence_loop_enabled and (
            self.pipeline_enabled
            or self.reflective_retrieval_enabled
            or self.evidence_ledger_enabled
            or self.reasoning_ledger_enabled
            or self.retrieval_plan_enabled
            or self.adaptive_workspace_enabled
            or self.claim_evidence_state_enabled
        ):
            raise ValueError("claim-evidence loop is an independent A0 extension")
        if self.pipeline_version not in {"p0-v2", "p0-v3", "p0b", "p0b-r1", "m3"}:
            raise ValueError("unsupported pipeline_version")
        if self.pipeline_evidence_packet_enabled and not self.pipeline_enabled:
            raise ValueError("pipeline evidence packet requires pipeline")
        if (
            self.pipeline_evidence_packet_enabled
            and self.pipeline_version not in {"p0b", "m3"}
        ):
            raise ValueError("evidence packet requires p0b or m3")
        if (
            self.pipeline_evidence_gate_enabled
            and not self.pipeline_evidence_packet_enabled
        ):
            raise ValueError("P2 evidence gate requires the P1 evidence packet")
        if (
            self.pipeline_supplement_enabled
            and not self.pipeline_evidence_packet_enabled
        ):
            raise ValueError("P3 supplement requires the P1 evidence packet")
        if self.pipeline_supplement_enabled and self.pipeline_evidence_gate_enabled:
            raise ValueError("P3 supplement must branch directly from P1, not P2")
        object.__setattr__(
            self,
            "memory_components",
            normalize_memory_components(self.memory_components),
        )

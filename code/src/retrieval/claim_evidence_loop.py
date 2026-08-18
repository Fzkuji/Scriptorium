"""Code-enforced LLM organization after each novel retrieval batch."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


_CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "candidate_id": {"type": "string"},
        "claim": {"type": "string"},
        "status": {
            "type": "string",
            "enum": ["supported", "excluded", "uncertain"],
        },
        "evidence_refs": {"type": "array", "items": {"type": "string"}},
        "counterevidence_refs": {
            "type": "array", "items": {"type": "string"},
        },
        "relations": {"type": "array", "items": {"type": "string"}},
        "reason": {"type": "string"},
        "answer_impact": {"type": "string"},
        "verification_priority": {"type": "string"},
    },
    "required": [
        "candidate_id", "claim", "status", "evidence_refs",
        "counterevidence_refs", "relations", "reason", "answer_impact",
        "verification_priority",
    ],
    "additionalProperties": False,
}

ORGANIZER_PATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "question_target": {"type": "string"},
        "upsert_candidates": {"type": "array", "items": _CANDIDATE_SCHEMA},
        "remove_candidate_ids": {
            "type": "array", "items": {"type": "string"},
        },
        "remaining_gaps": {"type": "array", "items": {"type": "string"}},
        "ready_to_answer": {"type": "boolean"},
        "revision_reason": {"type": "string"},
    },
    "required": [
        "question_target", "upsert_candidates", "remove_candidate_ids",
        "remaining_gaps", "ready_to_answer", "revision_reason",
    ],
    "additionalProperties": False,
}


def _parse_patch(result: Any) -> dict[str, Any]:
    if isinstance(result.structured_output, dict):
        return dict(result.structured_output)
    text = result.text.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    elif text.startswith("```") and text.endswith("```"):
        text = text[3:-3].strip()
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("organizer patch must be an object")
    return parsed


@dataclass
class ClaimEvidenceOrganizer:
    runtime: Any
    question: str
    question_date: str
    memory_dir: Path
    initial_batch_size: int = 2
    initial_organizer_enabled: bool = True
    event_guidance_enabled: bool = False
    seen_batches: set[str] = field(default_factory=set)
    pending_batches: list[tuple[str, str]] = field(default_factory=list)
    candidates: dict[str, dict[str, Any]] = field(default_factory=dict)
    candidate_order: list[str] = field(default_factory=list)
    question_target: str = ""
    remaining_gaps: list[str] = field(default_factory=list)
    ready_to_answer: bool = False
    versions: list[dict[str, Any]] = field(default_factory=list)
    organizer_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    anthropic_equivalent_cost_usd: float = 0.0
    event_reminder_emitted: bool = False
    event_reminders: int = 0

    def queue(self, evidence: str, *, tool_name: str) -> bool:
        digest = hashlib.sha256(evidence.strip().encode("utf-8")).hexdigest()
        if digest in self.seen_batches:
            return False
        self.seen_batches.add(digest)
        self.pending_batches.append((tool_name, evidence))
        return (
            self.initial_organizer_enabled
            and
            self.organizer_calls == 0
            and len(self.pending_batches) >= self.initial_batch_size
        )

    def state(self) -> dict[str, Any]:
        return {
            "version": len(self.versions),
            "question_target": self.question_target,
            "candidates": [
                self.candidates[candidate_id]
                for candidate_id in self.candidate_order
                if candidate_id in self.candidates
            ],
            "remaining_gaps": list(self.remaining_gaps),
            "ready_to_answer": self.ready_to_answer,
        }

    def take_event_reminder(self) -> bool:
        """Emit one nonbinding reminder per sufficiently mature evidence cycle."""
        if (
            not self.event_guidance_enabled
            or self.event_reminder_emitted
            or len(self.pending_batches) < 3
        ):
            return False
        self.event_reminder_emitted = True
        self.event_reminders += 1
        return True

    def render(self) -> str:
        return (
            "<current_claim_evidence_state>\n"
            + json.dumps(self.state(), ensure_ascii=False, indent=2)
            + "\n</current_claim_evidence_state>"
        )

    async def organize_pending(
        self,
        *,
        trigger_reason: str = "initial_evidence_scale",
        proposed_answer: str = "",
        force: bool = False,
    ) -> dict[str, Any]:
        if not self.pending_batches and not force:
            raise ValueError("organizer has no pending evidence")
        pending = self.pending_batches
        self.pending_batches = []
        self.event_reminder_emitted = False
        tool_name = "+".join(name for name, _evidence in pending) or trigger_reason
        evidence = "\n\n".join(
            f"<evidence_batch tool={name!r}>\n{text}\n</evidence_batch>"
            for name, text in pending
        ) or "(No new evidence batch; perform the mandatory final state review.)"
        prompt = (
            f"Question date: {self.question_date}\nQuestion: {self.question}\n\n"
            f"Organizer trigger: {trigger_reason}\n"
            "Current candidate state:\n"
            + json.dumps(self.state(), ensure_ascii=False, indent=2)
            + f"\n\nNew evidence batch from {tool_name}:\n{evidence}\n\n"
            + (
                f"Proposed answer awaiting final review:\n{proposed_answer}\n\n"
                if proposed_answer else ""
            )
            + (
                "Return one JSON patch matching the supplied schema. Organize "
                "answer-level claims, not search keywords. Infer the operation "
                "requested by the question from its wording (for example recall, "
                "compare, synthesize, or recommend), and do not silently replace "
                "it with a stricter task such as finding an exact prior statement. "
                "Keep direct facts distinct from justified synthesis. Preserve "
                "supported candidates unless new evidence changes them; merge "
                "duplicates and do not invent evidence. Emit upserts only for new "
                "or materially changed candidates; do not restate unchanged "
                "candidates. ready_to_answer is advisory: the query agent retains "
                "the stop decision."
            )
        )
        result = await asyncio.to_thread(
            self.runtime.agent.run,
            prompt=prompt,
            system_prompt=(
                "You are the organizer inside a general retrieve-organize loop. "
                "Update a compact candidate/evidence state from the new evidence. "
                "Do not choose retrieval tools and do not answer the user."
            ),
            cwd=self.memory_dir,
            tools=[],
            max_turns=2,
            max_budget_usd=None,
            output_schema=ORGANIZER_PATCH_SCHEMA,
            builtin_tools_enabled=False,
        )
        self.runtime.log_agent_result(result, phase="memory_qa_c4_organizer")
        patch = _parse_patch(result)
        self._apply(patch)
        self.organizer_calls += 1
        self.input_tokens += result.input_tokens
        self.output_tokens += result.output_tokens
        self.anthropic_equivalent_cost_usd += float(
            result.anthropic_equivalent_cost_usd or 0.0
        )
        next_version = len(self.versions) + 1
        state_snapshot = self.state()
        state_snapshot["version"] = next_version
        version = {
            "version": next_version,
            "tool_name": tool_name,
            "trigger_reason": trigger_reason,
            "patch": patch,
            "state": state_snapshot,
        }
        self.versions.append(version)
        return version

    def _apply(self, patch: dict[str, Any]) -> None:
        target = str(patch.get("question_target") or "").strip()
        if target:
            self.question_target = target
        for candidate_id in patch.get("remove_candidate_ids") or []:
            candidate_id = str(candidate_id)
            self.candidates.pop(candidate_id, None)
        for candidate in patch.get("upsert_candidates") or []:
            candidate_id = str(candidate["candidate_id"])
            if candidate_id not in self.candidate_order:
                self.candidate_order.append(candidate_id)
            self.candidates[candidate_id] = dict(candidate)
        self.remaining_gaps = [
            str(gap) for gap in patch.get("remaining_gaps") or []
        ]
        self.ready_to_answer = bool(patch.get("ready_to_answer"))

    def metrics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "mode": (
                "c4-event-driven" if self.initial_organizer_enabled
                else "c5a-final-only"
            ),
            "initial_batch_size": self.initial_batch_size,
            "initial_organizer_enabled": self.initial_organizer_enabled,
            "event_guidance_enabled": self.event_guidance_enabled,
            "event_reminders": self.event_reminders,
            "organizer_calls": self.organizer_calls,
            "state_versions": len(self.versions),
            "mechanism_active": self.organizer_calls > 0,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "anthropic_equivalent_cost_usd": self.anthropic_equivalent_cost_usd,
            "versions": self.versions,
            "pending_unorganized_batches": len(self.pending_batches),
            "final_state": self.state(),
        }

"""LLM-managed, evidence-grounded workspace for one retrieval question."""

from __future__ import annotations

import copy
import re
from datetime import date
from typing import Any

from .evidence_ledger import EvidenceLedger


_TIME = re.compile(r"\d{4}(?:-\d{2}(?:-\d{2})?)?")
_REQUIREMENT_STATUSES = {"missing", "partial", "supported", "conflicted"}
_SUFFICIENCY = {"insufficient", "provisional", "sufficient"}
_PRECISIONS = {"none", "year", "month", "day"}
_RESOLUTION_POLICIES = {
    "none", "latest_assertion", "historical_at_time", "same_event_dedup",
    "status_filter",
}
_MODES = {
    "direct_fact",
    "knowledge_update",
    "preference_transfer",
    "relation_check",
    "set_enumeration",
    "temporal_ordering",
    "value_resolution",
}


REASONING_LEDGER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning_mode": {"type": "string", "enum": sorted(_MODES)},
        "requirements": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "status": {
                        "type": "string",
                        "enum": sorted(_REQUIREMENT_STATUSES),
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["claim", "status", "evidence_ids"],
            },
        },
        "candidates": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "properties": {
                    "candidate_id": {"type": "string"},
                    "value": {"type": "string"},
                    "predicate": {"type": "string"},
                    "event_time": {"type": ["string", "null"]},
                    "time_precision": {
                        "type": "string",
                        "enum": sorted(_PRECISIONS),
                    },
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                    "compatible": {"type": "boolean"},
                    "exclusion_reason": {"type": ["string", "null"]},
                },
                "required": [
                    "candidate_id", "value", "predicate", "event_time",
                    "time_precision", "evidence_ids", "compatible",
                    "exclusion_reason",
                ],
            },
        },
        "conflicts": {
            "type": "array",
            "maxItems": 12,
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "evidence_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["description", "evidence_ids"],
            },
        },
        "resolution": {
            "type": "object",
            "properties": {
                "policy": {
                    "type": "string",
                    "enum": sorted(_RESOLUTION_POLICIES),
                },
                "selected_candidate_ids": {
                    "type": "array", "items": {"type": "string"},
                },
                "rejected_candidate_ids": {
                    "type": "array", "items": {"type": "string"},
                },
                "justification_evidence_ids": {
                    "type": "array", "items": {"type": "string"},
                },
            },
            "required": [
                "policy", "selected_candidate_ids", "rejected_candidate_ids",
                "justification_evidence_ids",
            ],
        },
        "sufficiency": {"type": "string", "enum": sorted(_SUFFICIENCY)},
    },
    "required": [
        "reasoning_mode", "requirements", "candidates", "conflicts",
        "resolution", "sufficiency",
    ],
}


class ReasoningLedger:
    """Validate semantic state proposed by the answer LLM."""

    def __init__(self, evidence: EvidenceLedger) -> None:
        self.evidence = evidence
        self.state: dict[str, Any] = {
            "reasoning_mode": "direct_fact",
            "requirements": [],
            "candidates": [],
            "conflicts": [],
            "resolution": {
                "policy": "none",
                "selected_candidate_ids": [],
                "rejected_candidate_ids": [],
                "justification_evidence_ids": [],
            },
            "sufficiency": "insufficient",
        }
        self.updates = 0
        self.rejections = 0

    def _known_ids(self) -> set[str]:
        return {entry.evidence_id for entry in self.evidence.entries}

    @staticmethod
    def _text(value: Any, field: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        return text

    def _evidence_ids(self, value: Any, field: str) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be an array")
        ids = list(dict.fromkeys(self._text(item, field) for item in value))
        unknown = sorted(set(ids) - self._known_ids())
        if unknown:
            raise ValueError(f"{field} contains unknown evidence IDs: {unknown}")
        return ids

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            state = self._validate(payload)
        except ValueError:
            self.rejections += 1
            raise
        self.state = state
        self.updates += 1
        return copy.deepcopy(state)

    def _validate(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("reasoning workspace must be an object")
        mode = str(payload.get("reasoning_mode", ""))
        if mode not in _MODES:
            raise ValueError("reasoning_mode is invalid")
        requirements = payload.get("requirements")
        candidates = payload.get("candidates")
        conflicts = payload.get("conflicts")
        if not isinstance(requirements, list) or len(requirements) > 12:
            raise ValueError("requirements must contain at most 12 items")
        if not isinstance(candidates, list) or len(candidates) > 24:
            raise ValueError("candidates must contain at most 24 items")
        if not isinstance(conflicts, list) or len(conflicts) > 12:
            raise ValueError("conflicts must contain at most 12 items")

        checked_requirements = []
        for number, item in enumerate(requirements):
            if not isinstance(item, dict):
                raise ValueError(f"requirements[{number}] must be an object")
            status = str(item.get("status", ""))
            if status not in _REQUIREMENT_STATUSES:
                raise ValueError(f"requirements[{number}].status is invalid")
            ids = self._evidence_ids(
                item.get("evidence_ids"), f"requirements[{number}].evidence_ids"
            )
            if status == "supported" and not ids:
                raise ValueError("a supported requirement needs evidence")
            checked_requirements.append({
                "claim": self._text(item.get("claim"), "requirement claim"),
                "status": status,
                "evidence_ids": ids,
            })

        checked_candidates = []
        candidate_ids: set[str] = set()
        for number, item in enumerate(candidates):
            if not isinstance(item, dict):
                raise ValueError(f"candidates[{number}] must be an object")
            candidate_id = self._text(
                item.get("candidate_id"), f"candidates[{number}].candidate_id"
            )
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", candidate_id):
                raise ValueError(f"candidates[{number}].candidate_id is invalid")
            if candidate_id in candidate_ids:
                raise ValueError(f"duplicate candidate_id: {candidate_id}")
            candidate_ids.add(candidate_id)
            ids = self._evidence_ids(
                item.get("evidence_ids"), f"candidates[{number}].evidence_ids"
            )
            compatible = item.get("compatible")
            if not isinstance(compatible, bool):
                raise ValueError(f"candidates[{number}].compatible must be boolean")
            if compatible and not ids:
                raise ValueError("a compatible candidate needs evidence")
            event_time = item.get("event_time")
            if event_time is not None:
                event_time = self._text(event_time, "event_time")
                if not _TIME.fullmatch(event_time):
                    raise ValueError(f"candidates[{number}].event_time is invalid")
            precision = str(item.get("time_precision", ""))
            if precision not in _PRECISIONS:
                raise ValueError(f"candidates[{number}].time_precision is invalid")
            exclusion = item.get("exclusion_reason")
            if exclusion is not None:
                exclusion = self._text(exclusion, "exclusion_reason")
            if not compatible and exclusion is None:
                raise ValueError("an incompatible candidate needs an exclusion reason")
            checked_candidates.append({
                "candidate_id": candidate_id,
                "value": self._text(item.get("value"), "candidate value"),
                "predicate": self._text(item.get("predicate"), "candidate predicate"),
                "event_time": event_time,
                "time_precision": precision,
                "evidence_ids": ids,
                "compatible": compatible,
                "exclusion_reason": exclusion,
            })

        checked_conflicts = []
        for number, item in enumerate(conflicts):
            if not isinstance(item, dict):
                raise ValueError(f"conflicts[{number}] must be an object")
            checked_conflicts.append({
                "description": self._text(
                    item.get("description"), "conflict description"
                ),
                "evidence_ids": self._evidence_ids(
                    item.get("evidence_ids"), f"conflicts[{number}].evidence_ids"
                ),
            })

        resolution = payload.get("resolution")
        if not isinstance(resolution, dict):
            raise ValueError("resolution must be an object")
        policy = str(resolution.get("policy", ""))
        if policy not in _RESOLUTION_POLICIES:
            raise ValueError("resolution.policy is invalid")
        selected = self._candidate_ids(
            resolution.get("selected_candidate_ids"),
            "resolution.selected_candidate_ids", candidate_ids,
        )
        rejected = self._candidate_ids(
            resolution.get("rejected_candidate_ids"),
            "resolution.rejected_candidate_ids", candidate_ids,
        )
        if set(selected) & set(rejected):
            raise ValueError("selected and rejected candidate IDs must be disjoint")
        justification = self._evidence_ids(
            resolution.get("justification_evidence_ids"),
            "resolution.justification_evidence_ids",
        )
        by_id = {item["candidate_id"]: item for item in checked_candidates}
        if any(not by_id[candidate_id]["compatible"] for candidate_id in selected):
            raise ValueError("selected candidates must be compatible")
        if policy == "none" and (selected or rejected):
            raise ValueError("resolution policy none cannot select or reject candidates")
        if policy != "none" and not selected:
            raise ValueError("a resolution policy requires a selected candidate")
        if policy != "none" and not justification:
            raise ValueError("a resolution policy requires justification evidence")
        if policy in {"same_event_dedup", "status_filter"} and not rejected:
            raise ValueError(f"{policy} requires a rejected candidate")
        if policy == "latest_assertion":
            self._validate_latest_selection(selected, checked_candidates)

        sufficiency = str(payload.get("sufficiency", ""))
        if sufficiency not in _SUFFICIENCY:
            raise ValueError("sufficiency is invalid")
        if sufficiency == "sufficient" and any(
            item["status"] != "supported"
            for item in checked_requirements
        ):
            raise ValueError("sufficiency cannot be sufficient with unmet requirements")
        if sufficiency == "sufficient" and not any(
            item["compatible"] for item in checked_candidates
        ):
            raise ValueError("sufficiency requires a compatible candidate")
        if sufficiency == "sufficient" and checked_conflicts and policy == "none":
            raise ValueError("sufficiency with conflicts requires a resolution policy")
        if mode == "temporal_ordering" and sufficiency == "sufficient":
            self._validate_temporal_order(checked_candidates)
        return {
            "reasoning_mode": mode,
            "requirements": checked_requirements,
            "candidates": checked_candidates,
            "conflicts": checked_conflicts,
            "resolution": {
                "policy": policy,
                "selected_candidate_ids": selected,
                "rejected_candidate_ids": rejected,
                "justification_evidence_ids": justification,
            },
            "sufficiency": sufficiency,
        }

    @staticmethod
    def _candidate_ids(value: Any, field: str, known: set[str]) -> list[str]:
        if not isinstance(value, list):
            raise ValueError(f"{field} must be an array")
        ids = list(dict.fromkeys(str(item).strip() for item in value))
        if any(not item for item in ids):
            raise ValueError(f"{field} contains an empty candidate ID")
        unknown = sorted(set(ids) - known)
        if unknown:
            raise ValueError(f"{field} contains unknown candidate IDs: {unknown}")
        return ids

    def _validate_latest_selection(
        self, selected: list[str], candidates: list[dict[str, Any]]
    ) -> None:
        timed = [
            item for item in candidates
            if item["compatible"] and item["event_time"] is not None
            and item["time_precision"] != "none"
        ]
        if len(timed) < 2:
            raise ValueError("latest_assertion requires competing timed candidates")
        starts = {
            item["candidate_id"]: self._time_interval(
                item["event_time"], item["time_precision"]
            )[0]
            for item in timed
        }
        latest = max(starts.values())
        if any(starts.get(candidate_id) != latest for candidate_id in selected):
            raise ValueError("latest_assertion must select the latest candidate")

    @staticmethod
    def _time_interval(value: str, precision: str) -> tuple[date, date]:
        parts = [int(part) for part in value.split("-")]
        if precision == "year":
            return date(parts[0], 1, 1), date(parts[0] + 1, 1, 1)
        if precision == "month":
            start = date(parts[0], parts[1], 1)
            end = (
                date(parts[0] + 1, 1, 1)
                if parts[1] == 12
                else date(parts[0], parts[1] + 1, 1)
            )
            return start, end
        start = date(parts[0], parts[1], parts[2])
        return start, date.fromordinal(start.toordinal() + 1)

    def _validate_temporal_order(self, candidates: list[dict[str, Any]]) -> None:
        timed = [
            item for item in candidates
            if item["compatible"] and item["event_time"] is not None
            and item["time_precision"] != "none"
        ]
        if len(timed) < 2:
            raise ValueError("temporal ordering requires at least two timed candidates")
        intervals = [
            (item, self._time_interval(item["event_time"], item["time_precision"]))
            for item in timed
        ]
        for index, (left, (left_start, left_end)) in enumerate(intervals):
            for right, (right_start, right_end) in intervals[index + 1:]:
                if left_start < right_end and right_start < left_end:
                    raise ValueError(
                        "temporal candidates overlap at their recorded precision: "
                        f"{left['value']!r} and {right['value']!r}"
                    )

    def render(self) -> str:
        rows = ["<reasoning_workspace>"]
        rows.append(f"reasoning_mode={self.state['reasoning_mode']}")
        rows.append(f"sufficiency={self.state['sufficiency']}")
        for item in self.state["requirements"]:
            rows.append(
                f"- requirement [{item['status']}]: {item['claim']} "
                f"evidence={','.join(item['evidence_ids']) or 'none'}"
            )
        for item in self.state["candidates"]:
            disposition = "compatible" if item["compatible"] else (
                "excluded: " + str(item["exclusion_reason"])
            )
            time = item["event_time"] or "unknown"
            rows.append(
                f"- candidate {item['candidate_id']}: {item['value']} | "
                f"predicate={item['predicate']} | "
                f"time={time}/{item['time_precision']} | {disposition} | "
                f"evidence={','.join(item['evidence_ids']) or 'none'}"
            )
        for item in self.state["conflicts"]:
            rows.append(
                f"- conflict: {item['description']} "
                f"evidence={','.join(item['evidence_ids']) or 'none'}"
            )
        resolution = self.state["resolution"]
        rows.append(
            f"resolution={resolution['policy']} "
            f"selected={','.join(resolution['selected_candidate_ids']) or 'none'} "
            f"rejected={','.join(resolution['rejected_candidate_ids']) or 'none'} "
            "evidence="
            f"{','.join(resolution['justification_evidence_ids']) or 'none'}"
        )
        rows.append("</reasoning_workspace>")
        return "\n".join(rows)

    def metrics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "updates": self.updates,
            "rejections": self.rejections,
            "state": copy.deepcopy(self.state),
        }

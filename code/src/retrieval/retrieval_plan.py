"""Pre-retrieval question contract and adaptive tool budget."""

from __future__ import annotations

import copy
from typing import Any


_SHAPES = {
    "direct_fact", "count", "list", "ordered_list", "preference",
    "relation_check", "abstainable",
}
_BUDGETS = {
    "direct_fact": 5,
    "count": 10,
    "list": 10,
    "ordered_list": 10,
    "preference": 6,
    "relation_check": 7,
    "abstainable": 7,
}

RETRIEVAL_PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer_shape": {"type": "string", "enum": sorted(_SHAPES)},
        "target_predicate": {"type": "string"},
        "entities": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "time_scope": {"type": ["string", "null"]},
        "inclusion_rules": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "exclusion_rules": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "required_evidence": {"type": "array", "maxItems": 8, "items": {"type": "string"}},
        "search_queries": {"type": "array", "maxItems": 6, "items": {"type": "string"}},
        "source_check": {"type": "string"},
        "stop_rule": {"type": "string"},
        "revision_reason": {"type": ["string", "null"]},
    },
    "required": [
        "answer_shape", "target_predicate", "entities", "time_scope",
        "inclusion_rules", "exclusion_rules", "required_evidence",
        "search_queries", "source_check", "stop_rule", "revision_reason",
    ],
}


class RetrievalPlan:
    """Validated one-shot plan created before any retrieval call."""

    def __init__(self) -> None:
        self.state: dict[str, Any] | None = None
        self.updates = 0
        self.rejections = 0

    @staticmethod
    def _text(value: Any, field: str) -> str:
        text = str(value).strip()
        if not text:
            raise ValueError(f"{field} must not be empty")
        return text

    def _texts(self, value: Any, field: str, maximum: int) -> list[str]:
        if not isinstance(value, list) or len(value) > maximum:
            raise ValueError(f"{field} must contain at most {maximum} items")
        return list(dict.fromkeys(self._text(item, field) for item in value))

    def update(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self.updates >= 2:
            self.rejections += 1
            raise ValueError("retrieval plan permits at most one revision")
        try:
            if not isinstance(payload, dict):
                raise ValueError("retrieval plan must be an object")
            shape = str(payload.get("answer_shape", ""))
            if shape not in _SHAPES:
                raise ValueError("answer_shape is invalid")
            state = {
                "answer_shape": shape,
                "target_predicate": self._text(
                    payload.get("target_predicate"), "target_predicate"
                ),
                "entities": self._texts(payload.get("entities"), "entities", 8),
                "time_scope": (
                    None if payload.get("time_scope") is None else self._text(
                        payload.get("time_scope"), "time_scope"
                    )
                ),
                "inclusion_rules": self._texts(
                    payload.get("inclusion_rules"), "inclusion_rules", 8
                ),
                "exclusion_rules": self._texts(
                    payload.get("exclusion_rules"), "exclusion_rules", 8
                ),
                "required_evidence": self._texts(
                    payload.get("required_evidence"), "required_evidence", 8
                ),
                "search_queries": self._texts(
                    payload.get("search_queries"), "search_queries", 6
                ),
                "source_check": self._text(
                    payload.get("source_check"), "source_check"
                ),
                "stop_rule": self._text(payload.get("stop_rule"), "stop_rule"),
                "revision_reason": (
                    None if payload.get("revision_reason") is None else self._text(
                        payload.get("revision_reason"), "revision_reason"
                    )
                ),
                "tool_budget": _BUDGETS[shape],
            }
            if not state["required_evidence"]:
                raise ValueError("required_evidence must not be empty")
            if not state["search_queries"]:
                raise ValueError("search_queries must not be empty")
            if self.updates == 0 and state["revision_reason"] is not None:
                raise ValueError("initial plan cannot have a revision_reason")
            if self.updates == 1 and state["revision_reason"] is None:
                raise ValueError("revised plan requires revision_reason")
        except ValueError:
            self.rejections += 1
            raise
        self.state = state
        self.updates += 1
        return copy.deepcopy(state)

    @property
    def tool_budget(self) -> int:
        return int(self.state["tool_budget"]) if self.state else 0

    @property
    def hard_budget(self) -> int:
        return self.tool_budget + 2

    def render(self) -> str:
        assert self.state is not None
        return (
            "<retrieval_plan>\n"
            f"shape={self.state['answer_shape']} budget={self.tool_budget}\n"
            f"predicate={self.state['target_predicate']}\n"
            f"required={'; '.join(self.state['required_evidence'])}\n"
            f"stop={self.state['stop_rule']}\n"
            "</retrieval_plan>"
        )

    def metrics(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "updates": self.updates,
            "rejections": self.rejections,
            "state": copy.deepcopy(self.state),
        }

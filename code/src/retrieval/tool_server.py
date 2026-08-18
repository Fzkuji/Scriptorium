"""Claude Code MCP tools for read-only memory retrieval."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_agent_sdk import tool

from ..runtime.tokenization import TokenCounter
from .evidence_ledger import EvidenceLedger
from .reasoning_ledger import REASONING_LEDGER_SCHEMA, ReasoningLedger
from .retrieval_plan import RETRIEVAL_PLAN_SCHEMA, RetrievalPlan
from .tools import execute_tool_call
from .views import tools_for
from .claim_evidence_loop import ClaimEvidenceOrganizer


@dataclass
class RetrievalToolState:
    trace: list[dict[str, Any]]
    evidence: list[dict[str, str]]
    model: str
    indexes: dict[str, Any] = field(default_factory=dict)
    tool_calls: int = 0
    visible_tokens: int = 0
    ledger: EvidenceLedger | None = None
    ledger_visible_tokens: int = 0
    reasoning_ledger: ReasoningLedger | None = None
    reasoning_visible_tokens: int = 0
    retrieval_plan: RetrievalPlan | None = None
    retrieval_calls: int = 0
    no_new_evidence_streak: int = 0
    reflections: list[dict[str, Any]] = field(default_factory=list)
    adaptive_workspace: list[dict[str, Any]] = field(default_factory=list)
    claim_evidence_states: list[dict[str, Any]] = field(default_factory=list)
    claim_evidence_organizer: ClaimEvidenceOrganizer | None = None


def retrieval_tools(
    runtime: Any,
    *,
    memory_dir: Path,
    files: list[Path],
    condition: str,
    include_recent: bool,
    state: RetrievalToolState,
    components: tuple[str, ...] | str | None = None,
    reflective_enabled: bool = False,
    adaptive_workspace_enabled: bool = False,
    claim_evidence_state_enabled: bool = False,
    search_tools: str = "split",
) -> list[Any]:
    definitions = tools_for(
        condition, components=components, search_tools=search_tools
    )

    def make_tool(definition: dict[str, Any]):
        function = definition["function"]
        name = function["name"]

        @tool(
            name,
            function.get("description", ""),
            function["parameters"],
        )
        async def invoke(arguments: dict[str, Any]) -> dict[str, Any]:
            if state.retrieval_plan is not None:
                if state.retrieval_plan.state is None:
                    message = "Submit set_retrieval_plan before retrieval."
                    state.trace.append({
                        "type": name, "executed": False, "accepted": False,
                        "error": message,
                    })
                    return {"content": [{"type": "text", "text": message}], "is_error": True}
                if state.retrieval_calls >= state.retrieval_plan.hard_budget:
                    message = "Absolute retrieval ceiling reached; answer from current evidence."
                    state.trace.append({
                        "type": name, "executed": False, "accepted": False,
                        "error": message,
                    })
                    return {"content": [{"type": "text", "text": message}], "is_error": True}
                if state.no_new_evidence_streak >= 3:
                    message = "Three retrieval calls added no evidence; stop searching."
                    state.trace.append({
                        "type": name, "executed": False, "accepted": False,
                        "error": message,
                    })
                    return {"content": [{"type": "text", "text": message}], "is_error": True}
                state.retrieval_calls += 1
            state.tool_calls += 1
            before_entries = len(state.ledger.entries) if state.ledger is not None else 0
            try:
                output, executed, accepted = execute_tool_call(
                    runtime,
                    name,
                    arguments,
                    memory_dir=memory_dir,
                    files=files,
                    condition=condition,
                    include_recent=include_recent,
                    components=components,
                    indexes=state.indexes,
                )
                is_error = output.startswith(("Command rejected:", "Tool error:"))
            except Exception as exc:  # MCP reports the failure to Claude Code
                output = f"{type(exc).__name__}: {exc}"
                executed = False
                accepted = None
                is_error = True
            nonempty = (
                bool(output.strip())
                and not is_error
                and output not in {
                    "No BM25 matches.",
                    "No embedding matches.",
                    "No memory matches.",
                }
            )
            rendered_output = output
            if nonempty and state.ledger is not None and name != "list_memory_files":
                state.ledger.ingest_tool_output(
                    tool_name=name,
                    arguments=arguments,
                    output=output,
                    retrieval_round=state.tool_calls,
                )
                ledger_text = state.ledger.render_delta()
                if ledger_text:
                    rendered_output = output + "\n\n" + ledger_text
                    state.ledger_visible_tokens += TokenCounter.resolve(
                        requested_model=state.model
                    ).count(ledger_text)
            if state.retrieval_plan is not None and state.ledger is not None:
                after_entries = len(state.ledger.entries)
                state.no_new_evidence_streak = (
                    0 if after_entries > before_entries
                    else state.no_new_evidence_streak + 1
                )
                warnings = []
                if state.retrieval_calls >= state.retrieval_plan.tool_budget:
                    warnings.append(
                        "Advisory retrieval budget reached; answer now unless one "
                        "specific unresolved evidence requirement remains."
                    )
                if state.no_new_evidence_streak >= 2:
                    warnings.append(
                        "Two calls added no evidence; revise the plan once or answer."
                    )
                if warnings:
                    rendered_output += "\n\n<retrieval_budget_warning>" + " ".join(
                        warnings
                    ) + "</retrieval_budget_warning>"
            tokens = TokenCounter.resolve(
                requested_model=state.model
            ).count(rendered_output) if nonempty else 0
            state.visible_tokens += tokens
            row = {
                "type": name,
                "args": arguments,
                "executed": executed,
                "nonempty": nonempty,
                "visible_tokens": tokens,
                "cumulative_visible_tokens": state.visible_tokens,
            }
            if accepted is not None:
                row["accepted"] = accepted
            state.trace.append(row)
            if nonempty and name != "list_memory_files":
                state.evidence.append({"text": output, "date": ""})
            if state.claim_evidence_organizer is not None:
                version = None
                if (
                    nonempty
                    and name != "list_memory_files"
                    and state.claim_evidence_organizer.queue(
                        output, tool_name=name
                    )
                ):
                    version = await state.claim_evidence_organizer.organize_pending(
                        trigger_reason="initial_evidence_scale"
                    )
                organizer_text = state.claim_evidence_organizer.render()
                rendered_output += "\n\n" + organizer_text
                reminder_text = ""
                if state.claim_evidence_organizer.take_event_reminder():
                    reminder_text = (
                        "<organizer_event_reminder>Enough novel evidence has "
                        "accumulated for a semantic checkpoint. If it directly "
                        "determines an answer candidate, materially changes a "
                        "candidate status, reveals a conflict, or changes the "
                        "retrieval direction, call report_retrieval_event now. "
                        "Otherwise continue without organizing.</organizer_event_reminder>"
                    )
                    rendered_output += "\n\n" + reminder_text
                organizer_tokens = TokenCounter.resolve(
                    requested_model=state.model
                ).count(organizer_text + reminder_text)
                state.visible_tokens += organizer_tokens
                row["organizer_triggered"] = version is not None
                row["organizer_state_version"] = len(
                    state.claim_evidence_organizer.versions
                )
                row["organizer_visible_tokens"] = organizer_tokens
                row["organizer_event_reminder"] = bool(reminder_text)
                row["cumulative_visible_tokens"] = state.visible_tokens
            return {
                "content": [{
                    "type": "text", "text": rendered_output or "(no output)"
                }],
                "is_error": is_error,
            }

        return invoke

    result = [make_tool(definition) for definition in definitions]
    if state.claim_evidence_organizer is not None:
        event_schema = {
            "type": "object",
            "properties": {
                "event": {
                    "type": "string",
                    "enum": [
                        "decisive_evidence",
                        "candidate_status_change",
                        "retrieval_direction_change",
                        "evidence_conflict",
                    ],
                },
                "reason": {"type": "string"},
            },
            "required": ["event", "reason"],
            "additionalProperties": False,
        }

        @tool(
            "report_retrieval_event",
            (
                "Report a genuine semantic checkpoint after new evidence: direct "
                "answer-determining evidence, a material candidate-status change, "
                "a retrieval-direction change, or an evidence conflict. The "
                "application organizes pending novel evidence before retrieval "
                "continues. Do not call this merely because another query is tried."
            ),
            event_schema,
        )
        async def report_retrieval_event(
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            organizer = state.claim_evidence_organizer
            version = None
            if organizer.pending_batches:
                version = await organizer.organize_pending(
                    trigger_reason=str(arguments["event"])
                )
            state.trace.append({
                "type": "report_retrieval_event",
                "accepted": True,
                "event": arguments["event"],
                "reason": arguments["reason"],
                "organizer_triggered": version is not None,
                "organizer_state_version": len(organizer.versions),
            })
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        ("Pending evidence organized.\n\n" if version else
                         "No novel pending evidence; no organizer call made.\n\n")
                        + organizer.render()
                    ),
                }],
                "is_error": False,
            }

        result.append(report_retrieval_event)
    if adaptive_workspace_enabled:
        workspace_schema = {
            "type": "object",
            "properties": {
                "current_interpretation": {"type": "string"},
                "candidate_answers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "candidate_id": {"type": "string"},
                            "value": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["possible", "supported", "rejected"],
                            },
                            "evidence": {
                                "type": "array", "items": {"type": "string"},
                            },
                        },
                        "required": ["candidate_id", "value", "status", "evidence"],
                        "additionalProperties": False,
                    },
                },
                "rejected_candidates": {
                    "type": "array", "items": {"type": "string"},
                },
                "open_questions": {
                    "type": "array", "items": {"type": "string"},
                },
                "revision_reason": {"type": ["string", "null"]},
            },
            "required": [
                "current_interpretation", "candidate_answers",
                "rejected_candidates", "open_questions", "revision_reason",
            ],
            "additionalProperties": False,
        }

        @tool(
            "update_adaptive_workspace",
            (
                "Optionally replace your current mutable retrieval workspace. "
                "Use it when evidence changes the interpretation, candidates, "
                "or remaining questions. This is a revisable scratch state, not "
                "a binding plan and not an answer validator. Evidence entries may "
                "name visible paths, lines, or concise retrieved facts."
            ),
            workspace_schema,
        )
        async def update_adaptive_workspace(arguments: dict[str, Any]) -> dict[str, Any]:
            snapshot = {
                "current_interpretation": str(arguments["current_interpretation"]),
                "candidate_answers": list(arguments["candidate_answers"]),
                "rejected_candidates": list(arguments["rejected_candidates"]),
                "open_questions": list(arguments["open_questions"]),
                "revision_reason": arguments.get("revision_reason"),
            }
            state.adaptive_workspace.append(snapshot)
            state.trace.append({
                "type": "update_adaptive_workspace",
                "accepted": True,
                "update": len(state.adaptive_workspace),
                "candidate_count": len(snapshot["candidate_answers"]),
                "open_question_count": len(snapshot["open_questions"]),
                "revision": bool(snapshot["revision_reason"]),
            })
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        "Adaptive workspace replaced. Continue retrieving, revise "
                        "again if evidence changes the interpretation, or answer "
                        "when the current evidence is sufficient."
                    ),
                }],
                "is_error": False,
            }

        result.append(update_adaptive_workspace)
    if claim_evidence_state_enabled:
        relation_schema = {
            "type": "object",
            "properties": {
                "type": {
                    "type": "string",
                    "enum": [
                        "same_event", "duplicate_of", "part_of", "before",
                        "after", "supports", "contradicts", "supersedes",
                        "unrelated", "other",
                    ],
                },
                "target_candidate_id": {"type": "string"},
                "evidence_refs": {"type": "array", "items": {"type": "string"}},
                "reason": {"type": "string"},
            },
            "required": ["type", "target_candidate_id", "evidence_refs", "reason"],
            "additionalProperties": False,
        }
        candidate_schema = {
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
                "relations": {"type": "array", "items": relation_schema},
                "reason": {"type": "string"},
                "answer_impact": {"type": "string"},
                "resolvable": {"type": "boolean"},
                "next_check": {"type": ["string", "null"]},
            },
            "required": [
                "candidate_id", "claim", "status", "evidence_refs",
                "counterevidence_refs", "relations", "reason",
                "answer_impact", "resolvable", "next_check",
            ],
            "additionalProperties": False,
        }
        claim_state_schema = {
            "type": "object",
            "properties": {
                "question_target": {"type": "string"},
                "candidates": {"type": "array", "items": candidate_schema},
                "remaining_gaps": {"type": "array", "items": {"type": "string"}},
                "ready_to_answer": {"type": "boolean"},
                "revision_reason": {"type": ["string", "null"]},
            },
            "required": [
                "question_target", "candidates", "remaining_gaps",
                "ready_to_answer", "revision_reason",
            ],
            "additionalProperties": False,
        }

        @tool(
            "update_claim_evidence_state",
            (
                "Replace the current answer-level claim-evidence state after "
                "examining retrieved evidence. Keep retrieval under your control. "
                "This is not a pre-retrieval plan: propose candidates only after "
                "evidence is visible, cite retrieved source/path/line references, "
                "and revise, merge, support, exclude, or leave candidates uncertain "
                "as later evidence warrants. An uncertain candidate justifies more "
                "retrieval only when it can change the answer and has a specific "
                "untried next_check. Before the final answer, submit a final state "
                "with ready_to_answer=true and answer from that state."
            ),
            claim_state_schema,
        )
        async def update_claim_evidence_state(
            arguments: dict[str, Any],
        ) -> dict[str, Any]:
            snapshot = {
                "question_target": str(arguments["question_target"]),
                "candidates": list(arguments["candidates"]),
                "remaining_gaps": list(arguments["remaining_gaps"]),
                "ready_to_answer": bool(arguments["ready_to_answer"]),
                "revision_reason": arguments.get("revision_reason"),
            }
            state.claim_evidence_states.append(snapshot)
            state.trace.append({
                "type": "update_claim_evidence_state",
                "accepted": True,
                "update": len(state.claim_evidence_states),
                "candidate_count": len(snapshot["candidates"]),
                "supported_count": sum(
                    row["status"] == "supported" for row in snapshot["candidates"]
                ),
                "excluded_count": sum(
                    row["status"] == "excluded" for row in snapshot["candidates"]
                ),
                "uncertain_count": sum(
                    row["status"] == "uncertain" for row in snapshot["candidates"]
                ),
                "ready_to_answer": snapshot["ready_to_answer"],
                "revision": bool(snapshot["revision_reason"]),
            })
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        "Claim-evidence state replaced. You retain control of "
                        "retrieval. Continue only for a concrete unresolved claim "
                        "that can change the answer, or answer from the final state."
                    ),
                }],
                "is_error": False,
            }

        result.append(update_claim_evidence_state)
    if reflective_enabled:
        reflection_schema = {
            "type": "object",
            "properties": {
                "supported_facts": {
                    "type": "array", "items": {"type": "string"},
                },
                "conflicts": {
                    "type": "array", "items": {"type": "string"},
                },
                "missing_information": {
                    "type": "array", "items": {"type": "string"},
                },
                "can_answer": {"type": "boolean"},
                "follow_up_queries": {
                    "type": "array", "items": {"type": "string"},
                },
            },
            "required": [
                "supported_facts", "conflicts", "missing_information",
                "can_answer", "follow_up_queries",
            ],
            "additionalProperties": False,
        }

        @tool(
            "update_retrieval_reflection",
            (
                "Record the current evidence sufficiency assessment. Call this "
                "after inspecting general_first_recall and again whenever new "
                "retrieval changes the assessment. State only evidence-grounded "
                "facts and conflicts. If can_answer is false, give focused, "
                "general follow-up queries; if true, leave follow_up_queries empty."
            ),
            reflection_schema,
        )
        async def update_retrieval_reflection(
            arguments: dict[str, Any]
        ) -> dict[str, Any]:
            can_answer = bool(arguments.get("can_answer"))
            follow_ups = list(arguments.get("follow_up_queries") or [])
            if can_answer and follow_ups:
                message = "follow_up_queries must be empty when can_answer is true"
                state.trace.append({
                    "type": "update_retrieval_reflection", "accepted": False,
                    "error": message,
                })
                return {
                    "content": [{"type": "text", "text": f"Rejected: {message}"}],
                    "is_error": True,
                }
            if not can_answer and not follow_ups:
                message = "provide at least one follow-up query when evidence is insufficient"
                state.trace.append({
                    "type": "update_retrieval_reflection", "accepted": False,
                    "error": message,
                })
                return {
                    "content": [{"type": "text", "text": f"Rejected: {message}"}],
                    "is_error": True,
                }
            reflection = {
                "supported_facts": list(arguments.get("supported_facts") or []),
                "conflicts": list(arguments.get("conflicts") or []),
                "missing_information": list(arguments.get("missing_information") or []),
                "can_answer": can_answer,
                "follow_up_queries": follow_ups,
            }
            state.reflections.append(reflection)
            state.trace.append({
                "type": "update_retrieval_reflection", "accepted": True,
                "round": len(state.reflections), "can_answer": can_answer,
                "missing_count": len(reflection["missing_information"]),
                "follow_up_count": len(follow_ups),
            })
            return {
                "content": [{
                    "type": "text",
                    "text": (
                        "Reflection recorded. Answer now from the evidence."
                        if can_answer else
                        "Reflection recorded. Use the focused follow-up queries, then reassess."
                    ),
                }],
                "is_error": False,
            }

        result.append(update_retrieval_reflection)
    if state.retrieval_plan is not None:
        @tool(
            "set_retrieval_plan",
            (
                "Before any retrieval, submit a concise question contract. "
                "Specify the answer shape, exact target predicate, entities, "
                "time scope, inclusion/exclusion rules, required evidence, "
                "focused search queries, source-verification trigger, and an "
                "evidence-based stopping rule. You may revise it once only after "
                "retrieval evidence contradicts the initial interpretation; a "
                "revision must provide revision_reason."
            ),
            RETRIEVAL_PLAN_SCHEMA,
        )
        async def set_retrieval_plan(arguments: dict[str, Any]) -> dict[str, Any]:
            try:
                if state.retrieval_plan.updates == 1 and state.retrieval_calls == 0:
                    raise ValueError("plan cannot be revised before retrieval evidence")
                state.retrieval_plan.update(arguments)
                rendered = state.retrieval_plan.render()
                state.trace.append({
                    "type": "set_retrieval_plan", "accepted": True,
                    "answer_shape": arguments.get("answer_shape"),
                    "tool_budget": state.retrieval_plan.tool_budget,
                    "revision": state.retrieval_plan.updates == 2,
                })
                return {"content": [{"type": "text", "text": rendered}], "is_error": False}
            except ValueError as exc:
                state.trace.append({
                    "type": "set_retrieval_plan", "accepted": False,
                    "error": str(exc),
                })
                return {"content": [{"type": "text", "text": f"Rejected: {exc}"}], "is_error": True}

        result.insert(0, set_retrieval_plan)
    if state.reasoning_ledger is not None:
        @tool(
            "update_reasoning_ledger",
            (
                "Replace the evidence-grounded reasoning workspace. Use after "
                "retrieving evidence and before answering. Separate question "
                "requirements, candidate values and their predicates, temporal "
                "precision, incompatibilities, and unresolved conflicts. Cite "
                "only evidence IDs shown by evidence_ledger_delta. Choose a "
                "reasoning_mode from the schema. For set_enumeration, list each "
                "physical item or event before counting and do not collapse "
                "distinct pickup/return obligations. For preference_transfer, "
                "extract grounded preferences and apply them to the new setting; "
                "the new location need not occur in memory. For value_resolution, "
                "enumerate every relevant number and bind it to its predicate. "
                "For relation_check, require the question's exact roles and every "
                "relation edge. For temporal_ordering, distinguish planned from "
                "completed events and resolve overlapping date precision. For "
                "knowledge_update, surface conflicting old and new values."
            ),
            REASONING_LEDGER_SCHEMA,
        )
        async def update_reasoning_ledger(
            arguments: dict[str, Any]
        ) -> dict[str, Any]:
            try:
                state.reasoning_ledger.update(arguments)
                rendered = state.reasoning_ledger.render()
                state.reasoning_visible_tokens += TokenCounter.resolve(
                    requested_model=state.model
                ).count(rendered)
                state.trace.append({
                    "type": "update_reasoning_ledger",
                    "accepted": True,
                    "sufficiency": arguments.get("sufficiency"),
                })
                return {
                    "content": [{"type": "text", "text": rendered}],
                    "is_error": False,
                }
            except ValueError as exc:
                state.trace.append({
                    "type": "update_reasoning_ledger",
                    "accepted": False,
                    "error": str(exc),
                })
                return {
                    "content": [{"type": "text", "text": f"Rejected: {exc}"}],
                    "is_error": True,
                }

        result.append(update_reasoning_ledger)
    return result

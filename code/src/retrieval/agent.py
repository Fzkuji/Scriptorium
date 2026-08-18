"""Agent-controlled Scriptorium retrieval through Claude Code."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from ..runtime.tokenization import TokenCounter
from .config import QueryConfig
from .claim_evidence_loop import ClaimEvidenceOrganizer
from .context import initialize_context
from .evidence_ledger import EvidenceLedger
from .evidence_packet import build_evidence_packet
from .evidence_gate import build_evidence_gate
from .pipeline import (
    PipelineConfig, build_pipeline_context, supplement_pipeline_rows,
)
from .prompts import EVIDENCE_GATE_PROMPT, EVIDENCE_PACKET_PROMPT, PIPELINE_PROMPT
from .reasoning_ledger import ReasoningLedger
from .reflective import (
    REFLECTION_SCHEMA,
    build_general_first_recall,
    render_reflective_evidence,
    retrieve_follow_up_queries,
)
from .retrieval_plan import RetrievalPlan
from .tool_server import RetrievalToolState, retrieval_tools
from .views import memory_files


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
_REFLECTION_RE = re.compile(
    r"<reflection>\s*(\{.*?\})\s*</reflection>", re.DOTALL | re.IGNORECASE
)


def _parse_reflection(result: Any) -> dict[str, Any]:
    if isinstance(result.structured_output, dict):
        return dict(result.structured_output)
    match = _REFLECTION_RE.search(result.text)
    raw = match.group(1) if match else result.text.strip()
    if raw.startswith("```json") and raw.endswith("```"):
        raw = raw[7:-3].strip()
    elif raw.startswith("```") and raw.endswith("```"):
        raw = raw[3:-3].strip()
    if not raw:
        raise ValueError(
            "empty reflection text "
            f"(stop_reason={result.stop_reason!r}, turns={result.num_turns})"
        )
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid reflection JSON: {raw[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("reflection must be a JSON object")
    return parsed


def collect_answer(
    runtime: Any,
    item: dict[str, Any],
    memory_dir: Path,
    turn_index: dict[str, Any],
    condition: str = "native",
    *,
    include_recent: bool = True,
    config: QueryConfig | None = None,
) -> tuple[list[dict[str, str]], int, str, list[dict[str, Any]]]:
    del turn_index
    config = config or getattr(runtime, "query_config", None) or QueryConfig()
    components = config.memory_components
    memory_dir = Path(memory_dir).resolve()
    files = memory_files(
        memory_dir,
        condition,
        include_recent=include_recent,
        components=components,
    )
    condition_label = (
        "components:" + ",".join(components)
        if components is not None
        else condition
    )
    sources_visible = any(
        path.relative_to(memory_dir).parts[:1] == ("sources",)
        for path in files
    )
    prompt, trace, evidence, initial_tokens = initialize_context(
        memory_dir=memory_dir,
        files=files,
        condition=condition_label,
        item=item,
        verify_sources=config.verify_sources,
        model=runtime.model,
    )
    ledger = None
    reasoning_ledger = None
    retrieval_plan = None
    if config.evidence_ledger_enabled:
        ledger = EvidenceLedger(
            memory_dir=memory_dir,
            files=files,
            max_entries=config.evidence_ledger_max_entries,
        )
        core_path = memory_dir / "core.md"
        recent_path = memory_dir / "recent_events.jsonl"
        core = core_path.read_text(encoding="utf-8") if core_path in files else ""
        recent = (
            recent_path.read_text(encoding="utf-8")
            if recent_path in files
            else ""
        )
        ledger.ingest_initial(core=core, recent=recent)
        # Core and recent are already visible in the native prompt. Register
        # their presence for audit without repeating either document.
        if config.reasoning_ledger_enabled:
            reasoning_ledger = ReasoningLedger(ledger)
        if config.retrieval_plan_enabled:
            retrieval_plan = RetrievalPlan()
    state = RetrievalToolState(
        trace=trace,
        evidence=evidence,
        model=runtime.model,
        visible_tokens=initial_tokens,
        ledger=ledger,
        reasoning_ledger=reasoning_ledger,
        retrieval_plan=retrieval_plan,
    )
    if config.claim_evidence_loop_enabled:
        state.claim_evidence_organizer = ClaimEvidenceOrganizer(
            runtime=runtime,
            question=str(item["question"]),
            question_date=str(item.get("question_date") or ""),
            memory_dir=memory_dir,
            initial_batch_size=config.claim_evidence_loop_initial_batch_size,
            initial_organizer_enabled=(
                config.claim_evidence_loop_initial_organizer_enabled
            ),
            event_guidance_enabled=(
                config.claim_evidence_loop_event_guidance_enabled
            ),
        )
    pipeline_metrics = None
    pipeline_config = None
    pipeline_rows: list[dict[str, Any]] = []
    reflective_metrics = None
    if config.reflective_retrieval_enabled:
        recall_text, recall_rows, reflective_metrics = build_general_first_recall(
            runtime,
            memory_dir=memory_dir,
            files=files,
            query=str(item["question"]),
        )
        prompt += "\n\n" + recall_text
        recall_tokens = TokenCounter.resolve(
            requested_model=runtime.model
        ).count(recall_text)
        state.visible_tokens += recall_tokens
        trace.append({
            "type": "general_first_recall",
            "executed": True,
            "visible_tokens": recall_tokens,
            "cumulative_visible_tokens": state.visible_tokens,
            **reflective_metrics,
        })
        evidence.extend(
            {"text": row["content"], "date": row["date"]}
            for row in recall_rows
        )
    if config.pipeline_enabled:
        pipeline_config = PipelineConfig.for_version(config.pipeline_version)
        pipeline_text, pipeline_rows, pipeline_metrics = build_pipeline_context(
            runtime,
            memory_dir=memory_dir,
            files=files,
            query=str(item["question"]),
            question_date=str(item.get("question_date") or ""),
            config=pipeline_config,
        )
        if config.pipeline_evidence_packet_enabled:
            pipeline_text, packet_metrics = build_evidence_packet(
                pipeline_rows,
                pipeline_metrics["question_contract"],
            )
            pipeline_metrics["evidence_packet"] = packet_metrics
        if config.pipeline_supplement_enabled:
            shape = pipeline_metrics["question_contract"]["answer_shape"]
            completed = packet_metrics["state_counts"].get("completed", 0)
            should_supplement = (
                shape in {"count", "list", "ordered_list"}
                and completed == 1
            )
            supplement_metrics = {
                "triggered": False,
                "lookup_count": 0,
                "reason": (
                    "single_completed_candidate_for_enumeration"
                    if should_supplement else "no_code_detectable_gap"
                ),
                "added_count": 0,
                "added_event_ids": [],
            }
            if should_supplement:
                added, supplement_metrics = supplement_pipeline_rows(
                    runtime,
                    memory_dir=memory_dir,
                    files=files,
                    query=str(item["question"]),
                    question_date=str(item.get("question_date") or ""),
                    existing_rows=pipeline_rows,
                )
                pipeline_rows.extend(added)
                pipeline_text, packet_metrics = build_evidence_packet(
                    pipeline_rows,
                    pipeline_metrics["question_contract"],
                )
                pipeline_metrics["evidence_packet"] = packet_metrics
            pipeline_metrics["supplement"] = supplement_metrics
        gate_text = ""
        if config.pipeline_evidence_gate_enabled:
            gate_text, gate_metrics = build_evidence_gate(
                question=str(item["question"]),
                rows=pipeline_rows,
                contract=pipeline_metrics["question_contract"],
                packet_metrics=pipeline_metrics["evidence_packet"],
            )
            pipeline_metrics["evidence_gate"] = gate_metrics
        core_path = memory_dir / "core.md"
        recent_path = memory_dir / "recent_events.jsonl"
        prompt_template = (
            EVIDENCE_GATE_PROMPT if config.pipeline_evidence_gate_enabled
            else EVIDENCE_PACKET_PROMPT if config.pipeline_evidence_packet_enabled
            else PIPELINE_PROMPT
        )
        prompt = prompt_template.format(
            core_memory=(
                core_path.read_text(encoding="utf-8")
                if core_path in files else "(empty)"
            ),
            recent_memory=(
                recent_path.read_text(encoding="utf-8")
                if recent_path in files else "(empty)"
            ),
            pipeline_context=pipeline_text,
            evidence_packet=pipeline_text,
            evidence_gate=gate_text,
            question_date=item.get("question_date", ""),
            question=item["question"],
        )
        pipeline_tokens = TokenCounter.resolve(
            requested_model=runtime.model
        ).count(pipeline_text)
        state.visible_tokens += pipeline_tokens
        trace.append({
            "type": "pipeline_retrieval",
            "executed": True,
            "visible_tokens": pipeline_tokens,
            "cumulative_visible_tokens": state.visible_tokens,
            **pipeline_metrics,
        })
        evidence.extend(
            {"text": row["content"], "date": row["date"]}
            for row in pipeline_rows
        )
    if config.reflective_retrieval_enabled:
        reflective_rows = list(recall_rows)
        seen_queries = {str(item["question"]).strip()}
        reflections: list[dict[str, Any]] = []
        total_turns = 0
        usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "anthropic_equivalent_cost_usd": 0.0,
        }
        answer = ""
        termination_reason = "reflective_safety_ceiling"
        for reflection_round in range(1, config.max_turns + 1):
            reflection_prompt = (
                "Question date: " + str(item.get("question_date") or "") + "\n"
                "Question: " + str(item["question"]) + "\n\n"
                + render_reflective_evidence(reflective_rows)
                + "\n\nAssess whether this evidence is sufficient to answer the exact "
                "question. Do not assume missing facts. If insufficient, propose "
                "focused follow-up search queries derived from the observed gaps. "
                "If sufficient, provide the final answer and no follow-up queries. "
                "Return exactly <reflection> followed by one JSON object with "
                "keys supported_facts, conflicts, missing_information, can_answer, "
                "follow_up_queries, and answer, followed by </reflection>."
            )
            result = runtime.agent.run(
                prompt=reflection_prompt,
                system_prompt=(
                    "You are the evidence sufficiency and answering stage of a "
                    "general memory retrieval system. Use only retrieved_evidence. "
                    "Do not classify benchmark question types or use task-specific "
                    "rules. Return the requested structured assessment."
                ),
                cwd=memory_dir,
                tools=[],
                max_turns=2,
                max_budget_usd=config.max_budget_usd,
                builtin_tools_enabled=False,
            )
            runtime.log_agent_result(result, phase="memory_qa_reflection")
            total_turns += result.num_turns
            usage["input_tokens"] += result.input_tokens
            usage["output_tokens"] += result.output_tokens
            usage["anthropic_equivalent_cost_usd"] += float(
                result.anthropic_equivalent_cost_usd or 0.0
            )
            reflection = _parse_reflection(result)
            reflection["round"] = reflection_round
            reflections.append(reflection)
            trace.append({
                "type": "retrieval_reflection",
                "accepted": True,
                **reflection,
            })
            if reflection.get("can_answer"):
                answer = str(reflection.get("answer") or "").strip()
                if answer:
                    termination_reason = "evidence_sufficient"
                    break
            queries = [
                str(query).strip()
                for query in reflection.get("follow_up_queries") or []
                if str(query).strip() and str(query).strip() not in seen_queries
            ]
            seen_queries.update(queries)
            if not queries:
                answer = "Insufficient information."
                termination_reason = "no_new_follow_up_queries"
                break
            added, follow_up_metrics = retrieve_follow_up_queries(
                runtime,
                memory_dir=memory_dir,
                files=files,
                queries=queries,
                existing_rows=reflective_rows,
            )
            reflective_rows.extend(added)
            evidence.extend(
                {"text": row["content"], "date": row["date"]}
                for row in added
            )
            trace.append({
                "type": "reflective_follow_up_recall",
                "executed": True,
                "round": reflection_round,
                **follow_up_metrics,
            })
        termination = {
            "type": "termination",
            "termination_reason": termination_reason,
            "retrieval_rounds": len(reflections),
            "tool_calls": 0,
            "memory_visible_tokens": state.visible_tokens,
            "source_verification": config.verify_sources and sources_visible,
            "source_verification_requested": config.verify_sources,
            "memory_components": list(components) if components is not None else None,
            **usage,
            "reflective_retrieval": {
                **reflective_metrics,
                "orchestration": "application_controlled_llm_queries",
                "reflection_count": len(reflections),
                "follow_up_query_count": len(seen_queries) - 1,
                "final_candidate_count": len(reflective_rows),
                "reflections": reflections,
            },
        }
        trace.append(termination)
        return evidence, total_turns, answer or "Insufficient information.", trace
    result = runtime.agent.run(
        prompt=prompt,
        system_prompt=(
            (
                "Answer only from the supplied fixed Pipeline Context. No tools "
                "are available. Do not claim to search or read workspace files."
                if config.pipeline_enabled
                else "Retrieve evidence from the supplied memory workspace and "
                "answer the question. Use only the available tools and evidence."
            )
            + (
                " Before using any retrieval tool, call set_retrieval_plan once. "
                "Follow its target predicate, inclusion/exclusion rules, source "
                "check, and stop rule. Treat the tool budget as an advisory limit; "
                "an absolute ceiling applies two calls later. If retrieved evidence "
                "contradicts the initial interpretation, revise the plan at most "
                "once with a concrete revision_reason. Do not broaden the task "
                "without such evidence."
                if retrieval_plan is not None else ""
            )
            + (
                " Before answering, call update_reasoning_ledger after retrieval. "
                "List every hard requirement and plausible candidate, bind values "
                "to predicates and evidence IDs, preserve time precision, exclude "
                "role/relation mismatches, record conflicts, and answer only when "
                "the workspace is sufficient; otherwise abstain. Select the "
                "reasoning mode by the operation the question requires. A new "
                "location does not block transfer of a grounded user preference, "
                "but factual roles and relations must match exactly. Count and "
                "ordering questions require a complete candidate list; numerical "
                "questions require competing values to be bound to predicates. "
                "Give every candidate a stable ID. Detect conflicts separately "
                "from resolving them: when evidence supports a general policy, "
                "record latest_assertion, historical_at_time, same_event_dedup, "
                "or status_filter with selected/rejected candidate IDs and "
                "justification evidence. Do not call a conflict resolved merely "
                "to make the workspace sufficient. Use policy none when no "
                "resolution is justified."
                if reasoning_ledger is not None else ""
            )
            + (
                " Inspect general_first_recall before searching. Then call "
                "update_retrieval_reflection with supported facts, conflicts, "
                "missing information, answer sufficiency, and focused follow-up "
                "queries. If evidence is insufficient, use ordinary retrieval "
                "tools to investigate those gaps and reflect again. Continue "
                "until the evidence is sufficient or the existing runtime safety "
                "ceiling is reached. Do not classify the benchmark question, "
                "apply task-specific rules, or follow a fixed normal call budget."
                if config.reflective_retrieval_enabled else ""
            )
            + (
                " You may use update_adaptive_workspace as a mutable scratchpad "
                "whenever evidence changes your interpretation, candidates, or "
                "remaining questions. It is optional and revisable, not a binding "
                "plan. Before the final answer, lightly check for unsupported or "
                "duplicate candidates, contradictions, and retrieved evidence "
                "that should revise your current interpretation. Retain pragmatic "
                "contextual reasoning; do not demand literal wording when a short "
                "well-supported relation is clear."
                if config.adaptive_workspace_enabled else ""
            )
            + (
                " Preserve the original A0 behavior: you decide which memory "
                "views and retrieval tools to use, what to query, when to switch "
                "views, and when retrieval is no longer useful. After the first "
                "meaningful evidence batch, call update_claim_evidence_state to "
                "record answer-level candidate claims rather than merely listing "
                "retrieved snippets. Bind each candidate to raw evidence, record "
                "counterevidence and semantic relations, and mark it supported, "
                "excluded, or uncertain. Revise this state after evidence changes "
                "a candidate or before changing retrieval direction. Do not search "
                "merely because uncertainty exists: continue only when an uncertain "
                "candidate can change the answer and has a specific untried check. "
                "Before answering, submit a final ready_to_answer=true state and "
                "derive the answer from the supported, non-duplicate candidates. "
                "This state is formed after evidence and remains revisable; it is "
                "not a question classifier, immutable plan, or literal-entailment "
                "gate. Use pragmatic contextual reasoning when evidence supports it."
                if config.claim_evidence_state_enabled else ""
            )
            + (
                " Preserve A0 tool and stopping control. "
                + (
                    "The first time novel effective evidence reaches the initial scale "
                    "threshold, the application invokes an independent organizer and "
                    "appends current_claim_evidence_state. "
                    if config.claim_evidence_loop_initial_organizer_enabled else
                    "The application does not force an early organizer checkpoint. "
                )
                + (
                    "Call report_retrieval_event when accumulated new evidence directly "
                    "determines an answer candidate, materially changes a candidate's "
                    "status, reveals a conflict, or changes the retrieval direction; "
                    "code then organizes only pending evidence. "
                    if config.claim_evidence_loop_event_guidance_enabled else
                    "Afterward, call report_retrieval_event only when new evidence causes "
                    "a genuine retrieval-direction change or reveals a conflict; code "
                    "then organizes pending evidence. "
                )
                + "Use the latest state for retrieval decisions. "
                "You may stop even if gaps remain; uncertainty alone does not force "
                "retrieval. The application performs a mandatory final organization and "
                "answer review after you propose an answer."
                if config.claim_evidence_loop_enabled else ""
            )
        ),
        cwd=memory_dir,
        tools=(
            []
            if config.pipeline_enabled
            else retrieval_tools(
                runtime,
                memory_dir=memory_dir,
                files=files,
                condition=condition,
                include_recent=include_recent,
                components=components,
                state=state,
                reflective_enabled=config.reflective_retrieval_enabled,
                adaptive_workspace_enabled=config.adaptive_workspace_enabled,
                claim_evidence_state_enabled=config.claim_evidence_state_enabled,
                search_tools=config.search_tools,
            )
        ),
        max_turns=(
            pipeline_config.answer_max_turns
            if pipeline_config is not None else config.max_turns
        ),
        max_budget_usd=config.max_budget_usd,
    )
    runtime.log_agent_result(result, phase="memory_qa")
    match = _ANSWER_RE.search(result.text)
    answer = (match.group(1) if match else result.text).strip()
    c4_final_result = None
    if state.claim_evidence_organizer is not None:
        asyncio.run(state.claim_evidence_organizer.organize_pending(
            trigger_reason="final_answer_review",
            proposed_answer=answer,
            force=True,
        ))
        final_state_text = state.claim_evidence_organizer.render()
        c4_final_result = runtime.agent.run(
            prompt=(
                f"Question date: {item.get('question_date', '')}\n"
                f"Question: {item['question']}\n\n"
                f"Proposed answer:\n{answer}\n\n{final_state_text}\n\n"
                "Return the final answer inside <answer>...</answer>. Preserve the "
                "proposed answer when supported; otherwise correct it from the final "
                "candidate/evidence state. Do not retrieve more evidence."
            ),
            system_prompt=(
                "You are the final answer acceptance step of a general memory system. "
                "Use the proposed answer and final evidence state. Do not add unsupported "
                "facts or demand an exact historical statement unless the question asks "
                "for one."
            ),
            cwd=memory_dir,
            tools=[],
            max_turns=2,
            max_budget_usd=config.max_budget_usd,
            builtin_tools_enabled=False,
        )
        runtime.log_agent_result(c4_final_result, phase="memory_qa_c4_final")
        final_match = _ANSWER_RE.search(c4_final_result.text)
        answer = (
            final_match.group(1) if final_match else c4_final_result.text
        ).strip()
    if reasoning_ledger is not None and (
        reasoning_ledger.updates < 1
        or reasoning_ledger.state["sufficiency"] != "sufficient"
    ):
        answer = "Insufficient information."
    termination = {
        "type": "termination",
        "termination_reason": result.stop_reason or "complete",
        "retrieval_rounds": result.num_turns,
        "tool_calls": state.tool_calls,
        "memory_visible_tokens": state.visible_tokens,
        "source_verification": config.verify_sources and sources_visible,
        "source_verification_requested": config.verify_sources,
        "memory_components": list(components) if components is not None else None,
        "input_tokens": result.input_tokens + (
            c4_final_result.input_tokens if c4_final_result else 0
        ),
        "output_tokens": result.output_tokens + (
            c4_final_result.output_tokens if c4_final_result else 0
        ),
        "anthropic_equivalent_cost_usd": float(
            result.anthropic_equivalent_cost_usd or 0.0
        ) + float(
            c4_final_result.anthropic_equivalent_cost_usd or 0.0
            if c4_final_result else 0.0
        ),
    }
    if ledger is not None:
        termination["evidence_ledger"] = ledger.metrics()
        termination["evidence_ledger_visible_tokens"] = state.ledger_visible_tokens
    if reasoning_ledger is not None:
        termination["reasoning_ledger"] = reasoning_ledger.metrics()
        termination["reasoning_ledger_visible_tokens"] = (
            state.reasoning_visible_tokens
        )
    if retrieval_plan is not None:
        termination["retrieval_plan"] = retrieval_plan.metrics()
        termination["retrieval_calls"] = state.retrieval_calls
        termination["no_new_evidence_streak"] = state.no_new_evidence_streak
    if pipeline_metrics is not None:
        termination["pipeline"] = pipeline_metrics
    if reflective_metrics is not None:
        termination["reflective_retrieval"] = {
            **reflective_metrics,
            "reflection_count": len(state.reflections),
            "reflections": state.reflections,
        }
    if config.adaptive_workspace_enabled:
        termination["adaptive_workspace"] = {
            "enabled": True,
            "update_count": len(state.adaptive_workspace),
            "revisions": sum(
                bool(snapshot.get("revision_reason"))
                for snapshot in state.adaptive_workspace
            ),
            "final_state": (
                state.adaptive_workspace[-1]
                if state.adaptive_workspace else None
            ),
        }
    if config.claim_evidence_state_enabled:
        final_claim_state = (
            state.claim_evidence_states[-1]
            if state.claim_evidence_states else None
        )
        termination["claim_evidence_state"] = {
            "enabled": True,
            "update_count": len(state.claim_evidence_states),
            "revisions": sum(
                bool(snapshot.get("revision_reason"))
                for snapshot in state.claim_evidence_states
            ),
            "mechanism_active": len(state.claim_evidence_states) >= 2,
            "final_ready": bool(
                final_claim_state and final_claim_state.get("ready_to_answer")
            ),
            "final_state": final_claim_state,
        }
    if state.claim_evidence_organizer is not None:
        loop_metrics = state.claim_evidence_organizer.metrics()
        loop_metrics["final_answer_review"] = {
            "executed": c4_final_result is not None,
            "input_tokens": c4_final_result.input_tokens if c4_final_result else 0,
            "output_tokens": c4_final_result.output_tokens if c4_final_result else 0,
            "anthropic_equivalent_cost_usd": (
                c4_final_result.anthropic_equivalent_cost_usd
                if c4_final_result else 0.0
            ),
        }
        termination["claim_evidence_loop"] = loop_metrics
    trace.append(termination)
    return evidence, result.num_turns, answer or "Insufficient information.", trace

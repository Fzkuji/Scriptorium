#!/usr/bin/env python3
"""Execute frozen LongMemEval-S M1 Mem0 or Graphiti backend plans.

The CLI is formal-only: it launches a dedicated local evidence proxy, executes
the backend builder with GPT-5.5, performs retrieval, applies the hard 20K
visible-token gate, and publishes the frozen ``answer_input.json`` consumed by
the shared answer stage.  Synthetic callers can inject an adapter into
``execute_backend_plan``; the CLI does not expose a synthetic mode.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import re
import signal
import subprocess
import sys
import time
import uuid
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, ContextManager, Mapping, Protocol, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from baselines.longmemeval_m1 import longmemeval_m1_backend_contract as backend_contract  # noqa: E402
from baselines.longmemeval_m1 import longmemeval_m1_contract as contract  # noqa: E402
from scripts.evaluation import durable_model_ledger as durable  # noqa: E402
from scripts.evaluation.visible_token_budget import (  # noqa: E402
    VisibleTokenBudgetGate,
    snapshot_memory_path,
)
from baselines.gateways import openai_gpt55_flex_gateway_evidence as flex_evidence  # noqa: E402


PROXY_SCRIPT = SCRIPTS / "controlled_r301_model_proxy.py"
LOCAL_GATEWAY_KEY_ENV = "LONGMEMEVAL_M1_LOCAL_GATEWAY_KEY"


class BackendExecutionError(RuntimeError):
    """Raised when formal backend execution cannot be proven complete."""


class BackendAdapter(Protocol):
    observer: Any

    def operation(self, operation_id: str) -> ContextManager[None]: ...

    def ingest(self, session: Mapping[str, Any]) -> dict[str, Any]: ...

    def retrieve(self, question: str) -> list[dict[str, Any]]: ...

    def close(self) -> None: ...


AdapterFactory = Callable[..., BackendAdapter]


def _atomic_json_no_clobber(path: Path, value: Mapping[str, Any]) -> str:
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    digest = durable.atomic_json(path, dict(value))
    return digest


def _parse_reference_time(value: str) -> datetime:
    normalized = re.sub(r"\s*\([A-Za-z]{3}\)\s*", " ", value).strip()
    try:
        parsed = datetime.strptime(normalized, "%Y/%m/%d %H:%M")
    except ValueError as exc:
        raise BackendExecutionError(f"unsupported LongMemEval date: {value!r}") from exc
    return parsed.replace(tzinfo=timezone.utc)


class Mem0Adapter:
    def __init__(
        self,
        *,
        workspace: Mapping[str, Any],
        item: Mapping[str, Any],
        item_index: int,
        proxy_base_url: str,
        proxy_log: Path | None,
        ledger: durable.HashChainLedger,
        artifact_root: Path,
        tokenizer: contract.FormalTokenizer,
        formal: bool,
    ) -> None:
        del item_index
        os.environ["MEM0_TELEMETRY"] = "false"
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from mem0 import Memory  # type: ignore[import-not-found]
        from mem0.configs.base import MemoryConfig  # type: ignore[import-not-found]
        from mem0.memory import main as mem0_main  # type: ignore[import-not-found]
        from mem0.memory import telemetry as mem0_telemetry  # type: ignore[import-not-found]
        from openai import OpenAI  # type: ignore[import-not-found]
        import httpx  # type: ignore[import-not-found]

        mem0_telemetry.MEM0_TELEMETRY = False
        mem0_main.MEM0_TELEMETRY = False

        config = MemoryConfig(
            vector_store={
                "provider": "qdrant",
                "config": {
                    "collection_name": workspace["collection_name"],
                    "embedding_model_dims": contract.EMBEDDING_DIMS,
                    "path": workspace["qdrant_path"],
                    "on_disk": True,
                },
            },
            llm={
                "provider": "openai",
                "config": {
                    "model": contract.EXPECTED_MODEL,
                    "temperature": 0,
                    "api_key": "controlled-local-proxy",
                    "openai_base_url": proxy_base_url,
                    "is_reasoning_model": False,
                },
            },
            embedder={
                "provider": "huggingface",
                "config": {
                    "model": contract.EMBEDDING_MODEL,
                    "embedding_dims": contract.EMBEDDING_DIMS,
                    "model_kwargs": {
                        "local_files_only": True,
                        "trust_remote_code": False,
                    },
                },
            },
            history_db_path=workspace["history_db"],
            version="v1.1",
        )
        # Mem0 gives ambient OPENROUTER_API_KEY precedence over an explicit
        # OpenAI-compatible base URL.  Hide only that selector during client
        # construction, then restore the caller's environment unchanged.
        ambient_openrouter_key = os.environ.pop("OPENROUTER_API_KEY", None)
        try:
            self.memory = Memory(config=config)
        finally:
            if ambient_openrouter_key is not None:
                os.environ["OPENROUTER_API_KEY"] = ambient_openrouter_key
        # Bind a transport that cannot inherit HTTP(S)_PROXY from the shell.
        # Every builder request must first reach the dedicated localhost proxy.
        self.memory.llm.client.close()
        self._http_client = httpx.Client(trust_env=False)
        self.memory.llm.client = OpenAI(
            api_key="controlled-local-proxy",
            base_url=proxy_base_url,
            http_client=self._http_client,
        )
        actual_base_url = str(self.memory.llm.client.base_url).rstrip("/")
        if actual_base_url != proxy_base_url.rstrip("/"):
            self.memory.close()
            raise BackendExecutionError(
                "Mem0 OpenAI client is not bound to the dedicated proxy"
            )
        self.owner = str(item["question_id"])
        self.item = item
        self.observer = durable.DurableModelObserver(
            ledger=ledger,
            artifact_root=artifact_root,
            token_counter=tokenizer,
            expected_model=contract.EXPECTED_MODEL,
            proxy_log=proxy_log,
            formal=formal,
        )
        self.observer.install(self.memory.llm.client.chat.completions)

    def operation(self, operation_id: str) -> ContextManager[None]:
        return self.observer.operation(operation_id)

    def ingest(self, session: Mapping[str, Any]) -> dict[str, Any]:
        before_failed = sum(
            record.get("event") == "model_call_failed"
            for record in self.observer.ledger.records
        )
        session_index = int(session["dataset_session_index"])
        messages = [
            {"role": str(turn["role"]), "content": str(turn["content"])}
            for turn in self.item["haystack_sessions"][session_index]
        ]
        result = self.memory.add(
            messages,
            user_id=self.owner,
            metadata={
                "source_session_id": session["source_session_id"],
                "dataset_session_index": session["dataset_session_index"],
                "source_document_sha256": session["text_sha256"],
                "source_date": session["date"],
                "history_owner_question_id": self.owner,
            },
            infer=True,
        )
        after_failed = sum(
            record.get("event") == "model_call_failed"
            for record in self.observer.ledger.records
        )
        if after_failed != before_failed:
            raise BackendExecutionError("Mem0 suppressed a failed builder model call")
        rows = result.get("results") if isinstance(result, Mapping) else None
        if not isinstance(rows, list):
            raise BackendExecutionError("Mem0 add result is invalid")
        return {
            "backend": "mem0",
            "source_session_id": session["source_session_id"],
            "dataset_session_index": session["dataset_session_index"],
            "result_count": len(rows),
            "memory_ids": [row.get("id") for row in rows if isinstance(row, Mapping)],
            "events": [row.get("event") for row in rows if isinstance(row, Mapping)],
        }

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        value = self.memory.search(
            question,
            top_k=backend_contract.RETRIEVAL_LIMIT,
            filters={"user_id": self.owner},
            threshold=0.1,
            rerank=False,
            explain=False,
        )
        rows = value.get("results") if isinstance(value, Mapping) else None
        if not isinstance(rows, list):
            raise BackendExecutionError("Mem0 search result is invalid")
        results: list[dict[str, Any]] = []
        for rank, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise BackendExecutionError("Mem0 search row is invalid")
            metadata = row.get("metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            source_id = metadata.get("source_session_id")
            source_index = metadata.get("dataset_session_index")
            document_hash = metadata.get("source_document_sha256")
            text = row.get("memory")
            if (
                not isinstance(source_id, str)
                or not source_id
                or not isinstance(source_index, int)
                or isinstance(source_index, bool)
                or not isinstance(document_hash, str)
                or not isinstance(text, str)
                or not text.strip()
                or row.get("user_id") != self.owner
            ):
                raise BackendExecutionError("Mem0 result lacks a stable source mapping")
            results.append(
                {
                    "rank": rank,
                    "backend_record_id": str(row.get("id", "")),
                    "text": text,
                    "score": row.get("score"),
                    "source_session_ids": [source_id],
                    "source_dataset_session_indices": [source_index],
                    "source_document_sha256s": [document_hash],
                    "backend_trace": {"metadata": dict(metadata)},
                }
            )
        return results

    def close(self) -> None:
        self.observer.restore()
        self.memory.llm.client.close()
        vector_client = getattr(
            getattr(self.memory, "vector_store", None), "client", None
        )
        if vector_client is not None and hasattr(vector_client, "close"):
            vector_client.close()
        self.memory.close()


class _LocalSentenceTransformerEmbedder:
    def __init__(self) -> None:
        from graphiti_core.embedder.client import (  # type: ignore[import-not-found]
            EmbedderConfig,
        )
        from sentence_transformers import (  # type: ignore[import-not-found]
            SentenceTransformer,
        )

        self.config = EmbedderConfig(embedding_dim=contract.EMBEDDING_DIMS)
        self.model = SentenceTransformer(
            contract.EMBEDDING_MODEL,
            local_files_only=True,
            trust_remote_code=False,
        )

    async def create(self, input_data: Any) -> list[float]:
        if isinstance(input_data, str):
            text = input_data
        elif (
            isinstance(input_data, list)
            and len(input_data) == 1
            and isinstance(input_data[0], str)
        ):
            text = input_data[0]
        else:
            raise BackendExecutionError(
                "Graphiti local embedder requires one text input"
            )
        value = self.model.encode(text, convert_to_numpy=True).tolist()
        if len(value) != contract.EMBEDDING_DIMS:
            raise BackendExecutionError("Graphiti embedding dimension differs")
        return [float(item) for item in value]

    async def create_batch(self, input_data_list: list[str]) -> list[list[float]]:
        if not all(isinstance(value, str) for value in input_data_list):
            raise BackendExecutionError("Graphiti local embedder batch is invalid")
        values = self.model.encode(input_data_list, convert_to_numpy=True).tolist()
        if any(len(value) != contract.EMBEDDING_DIMS for value in values):
            raise BackendExecutionError("Graphiti embedding dimension differs")
        return [[float(item) for item in value] for value in values]


class _ForbiddenCrossEncoder:
    async def rank(self, query: str, passages: list[str]) -> list[tuple[str, float]]:
        del query, passages
        raise BackendExecutionError("Graphiti RRF search invoked a cross encoder")


class GraphitiAdapter:
    def __init__(
        self,
        *,
        workspace: Mapping[str, Any],
        item: Mapping[str, Any],
        item_index: int,
        proxy_base_url: str,
        proxy_log: Path | None,
        ledger: durable.HashChainLedger,
        artifact_root: Path,
        tokenizer: contract.FormalTokenizer,
        formal: bool,
    ) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["GRAPHITI_TELEMETRY_ENABLED"] = "false"
        from graphiti_core import Graphiti  # type: ignore[import-not-found]
        from graphiti_core.driver.kuzu_driver import (  # type: ignore[import-not-found]
            KuzuDriver,
        )
        from graphiti_core.embedder.client import (  # type: ignore[import-not-found]
            EmbedderClient,
        )
        from graphiti_core.llm_client import LLMConfig  # type: ignore[import-not-found]
        from graphiti_core.llm_client.openai_generic_client import (  # type: ignore[import-not-found]
            OpenAIGenericClient,
        )
        from graphiti_core.cross_encoder.client import (  # type: ignore[import-not-found]
            CrossEncoderClient,
        )
        from openai import AsyncOpenAI  # type: ignore[import-not-found]
        import httpx  # type: ignore[import-not-found]

        embedder_implementation = _LocalSentenceTransformerEmbedder()

        class LocalEmbedder(EmbedderClient):
            def __init__(self) -> None:
                self.config = embedder_implementation.config

            async def create(self, input_data: Any) -> list[float]:
                return await embedder_implementation.create(input_data)

            async def create_batch(
                self, input_data_list: list[str]
            ) -> list[list[float]]:
                return await embedder_implementation.create_batch(input_data_list)

        class ForbiddenCrossEncoder(CrossEncoderClient):
            async def rank(
                self, query: str, passages: list[str]
            ) -> list[tuple[str, float]]:
                return await _ForbiddenCrossEncoder().rank(query, passages)

        self.item_index = item_index
        self.question_id = str(item["question_id"])
        self.group_id = str(workspace["group_id"])
        self.episode_sources: dict[str, dict[str, Any]] = {}
        self.previous_episode_uuid: str | None = None
        self.loop = asyncio.new_event_loop()
        self._http_client = httpx.AsyncClient(trust_env=False)
        controlled_client = AsyncOpenAI(
            api_key="controlled-local-proxy",
            base_url=proxy_base_url,
            http_client=self._http_client,
        )
        llm_client = OpenAIGenericClient(
            config=LLMConfig(
                api_key="controlled-local-proxy",
                model=contract.EXPECTED_MODEL,
                small_model=contract.EXPECTED_MODEL,
                base_url=proxy_base_url,
                temperature=0,
            ),
            cache=False,
            client=controlled_client,
            max_tokens=16_384,
            structured_output_mode="json_object",
        )
        actual_base_url = str(llm_client.client.base_url).rstrip("/")
        if actual_base_url != proxy_base_url.rstrip("/"):
            self.loop.close()
            raise BackendExecutionError(
                "Graphiti OpenAI client is not bound to the dedicated proxy"
            )
        self.graphiti = Graphiti(
            graph_driver=KuzuDriver(db=str(workspace["kuzu_path"])),
            llm_client=llm_client,
            embedder=LocalEmbedder(),
            cross_encoder=ForbiddenCrossEncoder(),
            store_raw_episode_content=True,
            max_coroutines=1,
        )
        self.observer = backend_contract.AsyncDurableModelObserver(
            ledger=ledger,
            artifact_root=artifact_root,
            token_counter=tokenizer,
            expected_model=contract.EXPECTED_MODEL,
            proxy_log=proxy_log,
            formal=formal,
        )
        self.observer.install(llm_client.client.chat.completions)

    def operation(self, operation_id: str) -> ContextManager[None]:
        return self.observer.operation(operation_id)

    def ingest(self, session: Mapping[str, Any]) -> dict[str, Any]:
        from graphiti_core.nodes import EpisodeType  # type: ignore[import-not-found]

        source_id = str(session["source_session_id"])
        episode_uuid = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"longmemeval-m1:{self.item_index}:{self.question_id}:"
                f"{session['dataset_session_index']}:{source_id}",
            )
        )
        previous = [self.previous_episode_uuid] if self.previous_episode_uuid else None
        result = self.loop.run_until_complete(
            self.graphiti.add_episode(
                name=f"longmemeval_session_{int(session['dataset_session_index']):04d}",
                episode_body=str(session["text"]),
                source_description=(
                    "LongMemEval-S session; "
                    f"source_session_id={source_id}; "
                    f"dataset_session_index={session['dataset_session_index']}"
                ),
                reference_time=_parse_reference_time(str(session["date"])),
                source=EpisodeType.message,
                group_id=self.group_id,
                uuid=episode_uuid,
                update_communities=False,
                previous_episode_uuids=previous,
            )
        )
        if str(result.episode.uuid) != episode_uuid:
            raise BackendExecutionError("Graphiti episode UUID differs")
        self.episode_sources[episode_uuid] = {
            "source_session_id": source_id,
            "dataset_session_index": session["dataset_session_index"],
            "source_document_sha256": session["text_sha256"],
        }
        self.previous_episode_uuid = episode_uuid
        return {
            "backend": "graphiti",
            "source_session_id": source_id,
            "dataset_session_index": session["dataset_session_index"],
            "episode_uuid": episode_uuid,
            "episodic_edge_count": len(result.episodic_edges),
            "entity_node_count": len(result.nodes),
            "entity_edge_count": len(result.edges),
        }

    def retrieve(self, question: str) -> list[dict[str, Any]]:
        edges = self.loop.run_until_complete(
            self.graphiti.search(
                question,
                group_ids=[self.group_id],
                num_results=backend_contract.RETRIEVAL_LIMIT,
            )
        )
        results: list[dict[str, Any]] = []
        for rank, edge in enumerate(edges):
            if edge.group_id != self.group_id:
                raise BackendExecutionError("Graphiti returned another item group")
            episode_ids = list(dict.fromkeys(str(value) for value in edge.episodes))
            if not episode_ids or any(
                value not in self.episode_sources for value in episode_ids
            ):
                raise BackendExecutionError(
                    "Graphiti result lacks an item-local episode source"
                )
            sources = [self.episode_sources[value] for value in episode_ids]
            fact = str(edge.fact)
            if not fact.strip():
                raise BackendExecutionError("Graphiti returned an empty fact")
            results.append(
                {
                    "rank": rank,
                    "backend_record_id": str(edge.uuid),
                    "text": fact,
                    "score": None,
                    "source_session_ids": [
                        source["source_session_id"] for source in sources
                    ],
                    "source_dataset_session_indices": [
                        source["dataset_session_index"] for source in sources
                    ],
                    "source_document_sha256s": [
                        source["source_document_sha256"] for source in sources
                    ],
                    "backend_trace": {
                        "episode_uuids": episode_ids,
                        "edge_name": str(edge.name),
                        "valid_at": (
                            edge.valid_at.isoformat() if edge.valid_at else None
                        ),
                        "invalid_at": (
                            edge.invalid_at.isoformat() if edge.invalid_at else None
                        ),
                    },
                }
            )
        return results

    def close(self) -> None:
        self.observer.restore()
        self.loop.run_until_complete(self.graphiti.llm_client.client.close())
        self.loop.run_until_complete(self.graphiti.close())
        self.loop.close()


def production_adapter_factory(method: str, **kwargs: Any) -> BackendAdapter:
    if method == "mem0":
        return Mem0Adapter(**kwargs)
    if method == "graphiti":
        return GraphitiAdapter(**kwargs)
    raise BackendExecutionError(f"unsupported backend method: {method}")


class ManagedProxy:
    def __init__(
        self,
        *,
        run_dir: Path,
        run_id: str,
        upstream: str,
        api_key_env: str = LOCAL_GATEWAY_KEY_ENV,
    ) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.upstream = upstream
        self.api_key_env = api_key_env
        self.process: subprocess.Popen[Any] | None = None
        self.log: Path | None = None
        self.ready: Path | None = None
        self.base_url: str | None = None

    def start(self) -> None:
        directory = self.run_dir / "proxy"
        directory.mkdir(parents=True, exist_ok=True)
        launch = len(list(directory.glob("launch-*.ready.json"))) + 1
        self.log = directory / f"launch-{launch:03d}.jsonl"
        self.ready = directory / f"launch-{launch:03d}.ready.json"
        environment = os.environ.copy()
        environment[self.api_key_env] = "local-flex-child-proxy"
        self.process = subprocess.Popen(
            [
                sys.executable,
                str(PROXY_SCRIPT),
                "--port",
                "0",
                "--upstream",
                self.upstream,
                "--api-key-env",
                self.api_key_env,
                "--expected-requested-model",
                contract.EXPECTED_MODEL,
                "--accepted-actual-model-regex",
                backend_contract.ACCEPTED_ACTUAL_MODEL_REGEX,
                "--log",
                str(self.log),
                "--ready",
                str(self.ready),
                "--run-id",
                f"{self.run_id}:launch-{launch:03d}",
            ],
            cwd=ROOT,
            start_new_session=True,
            env=environment,
        )
        deadline = time.monotonic() + 20
        try:
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise BackendExecutionError(
                        "exclusive proxy exited before readiness"
                    )
                if self.ready.is_file():
                    value = contract.read_json(self.ready)
                    if not isinstance(value, dict):
                        raise BackendExecutionError(
                            "exclusive proxy readiness is invalid"
                        )
                    self.base_url = str(value["base_url"])
                    return
                time.sleep(0.05)
            raise BackendExecutionError("exclusive proxy readiness timed out")
        except BaseException:
            self.stop()
            raise

    def stop(self) -> None:
        if self.process is None or self.process.poll() is not None:
            return
        os.killpg(self.process.pid, signal.SIGTERM)
        try:
            self.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait(timeout=5)


def _run_identity(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    dataset_path: Path,
    preregistration_path: Path,
    formal: bool,
) -> dict[str, Any]:
    return {
        "benchmark": "LongMemEval-S",
        "milestone": "M1 controlled baselines",
        "method": plan["method"],
        "scope": plan["scope"],
        "formal": formal,
        "execution_version": backend_contract.BACKEND_EXECUTION_VERSION,
        "plan_path": str(plan_path),
        "plan_sha256": contract.sha256_file(plan_path),
        "plan_content_sha256": plan["plan_content_sha256"],
        "dataset_path": str(dataset_path),
        "dataset_sha256": contract.sha256_file(dataset_path),
        "preregistration_path": str(preregistration_path),
        "preregistration_sha256": contract.sha256_file(preregistration_path),
        "configuration": plan["configuration"],
        "dependencies": plan["dependencies"],
        "retrieval_limit": backend_contract.RETRIEVAL_LIMIT,
        "tokenizer": contract.FormalTokenizer.resolve().identity,
        "requested_builder_model": contract.EXPECTED_MODEL,
        "accepted_actual_model_regex": backend_contract.ACCEPTED_ACTUAL_MODEL_REGEX,
        "source_hashes": backend_contract.source_hashes(),
    }


def _new_manifest(
    *, identity: Mapping[str, Any], run_dir: Path, expected_items: int
) -> dict[str, Any]:
    return {
        "schema_version": contract.RUN_MANIFEST_SCHEMA_VERSION,
        "backend_execution_schema": backend_contract.BACKEND_RUN_SCHEMA,
        "status": "running",
        "created_at": contract.utc_now(),
        "updated_at": contract.utc_now(),
        "run_dir": str(run_dir),
        "identity": copy.deepcopy(dict(identity)),
        "completed_items": 0,
        "expected_items": expected_items,
        "abstention_items": 0,
        "model_calls": 0,
        "network_calls": 0,
        "answer_stage": "not_started",
        "items": [],
    }


def _operation(
    *,
    ledger: durable.HashChainLedger,
    adapter: BackendAdapter | None,
    operation_id: str,
    kind: str,
    input_sha256: str,
    artifact_root: Path,
    artifact_path: Path,
    action: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    ledger.append(
        "operation_started",
        operation_id=operation_id,
        kind=kind,
        operation_input_sha256=input_sha256,
    )
    try:
        context = (
            adapter.operation(operation_id) if adapter is not None else nullcontext()
        )
        with context:
            result = dict(action())
        _atomic_json_no_clobber(artifact_path, result)
        ledger.append(
            "operation_committed",
            operation_id=operation_id,
            kind=kind,
            operation_input_sha256=input_sha256,
            artifact_path=str(artifact_path.relative_to(artifact_root)),
            artifact_sha256=contract.sha256_file(artifact_path),
        )
        return result
    except BaseException as exc:
        ledger.append(
            "operation_failed",
            operation_id=operation_id,
            kind=kind,
            operation_input_sha256=input_sha256,
            error=f"{type(exc).__name__}: {exc}",
        )
        raise


def _normalize_retrieval_records(
    *,
    method: str,
    item: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    sessions = contract.rendered_sessions(item)
    source_by_id = {str(session["source_session_id"]): session for session in sessions}
    normalized: list[dict[str, Any]] = []
    backend_ids: set[str] = set()
    for rank, raw in enumerate(records):
        record = dict(raw)
        backend_id = record.get("backend_record_id")
        text = record.get("text")
        source_ids = record.get("source_session_ids")
        source_indices = record.get("source_dataset_session_indices")
        source_hashes = record.get("source_document_sha256s")
        if (
            record.get("rank") != rank
            or not isinstance(backend_id, str)
            or not backend_id
            or backend_id in backend_ids
            or not isinstance(text, str)
            or not text.strip()
            or not isinstance(source_ids, list)
            or not source_ids
            or len(set(source_ids)) != len(source_ids)
            or not isinstance(source_indices, list)
            or not isinstance(source_hashes, list)
            or not (len(source_ids) == len(source_indices) == len(source_hashes))
        ):
            raise BackendExecutionError(f"{method} retrieval record {rank} is invalid")
        backend_ids.add(backend_id)
        for source_id, source_index, source_hash in zip(
            source_ids, source_indices, source_hashes
        ):
            source = source_by_id.get(str(source_id))
            if (
                source is None
                or source["dataset_session_index"] != source_index
                or source["text_sha256"] != source_hash
            ):
                raise BackendExecutionError(
                    f"{method} retrieval record {rank} source mapping differs"
                )
        normalized.append(
            {
                "rank": rank,
                "backend_record_id": backend_id,
                "text": text,
                "text_sha256": contract.sha256_text(text),
                "score": record.get("score"),
                "source_session_ids": list(source_ids),
                "source_dataset_session_indices": list(source_indices),
                "source_document_sha256s": list(source_hashes),
                "backend_trace": record.get("backend_trace", {}),
            }
        )
    return normalized


def _usage_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    finished = [
        record for record in records if record.get("event") == "model_call_finished"
    ]
    totals: dict[str, int | None] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        values = [
            record.get("usage", {}).get(key)
            for record in finished
            if isinstance(record.get("usage"), Mapping)
        ]
        totals[key] = sum(value for value in values if isinstance(value, int))
    totals["model_calls"] = len(finished)
    return totals


def _freeze_retrieval(
    *,
    method: str,
    item: Mapping[str, Any],
    item_index: int,
    tokenizer: contract.FormalTokenizer,
    records: Sequence[Mapping[str, Any]],
    gate: VisibleTokenBudgetGate,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sessions = contract.rendered_sessions(item)
    session_by_id = {str(session["source_session_id"]): session for session in sessions}
    pieces: list[str] = []
    events: list[dict[str, Any]] = []
    all_delivered_sources: list[str] = []
    truncated = 0
    for record in records:
        source_ids = [str(value) for value in record["source_session_ids"]]
        header = (
            f"[Retrieved {int(record['rank']) + 1} | method={method} | "
            f"backend_record_id={record['backend_record_id']} | "
            f"source_session_ids={','.join(source_ids)}]"
        )
        raw_text = f"{header}\n{record['text']}"
        if pieces:
            raw_text = "\n\n" + raw_text
        delivered = gate.deliver_tool_result(
            event_id=f"retrieval-{int(record['rank']):04d}",
            raw_text=raw_text,
            tool_name=f"{method}.search",
            tool_call_id=str(record["backend_record_id"]),
            metadata={
                "rank": record["rank"],
                "backend_record_id": record["backend_record_id"],
                "source_session_ids": source_ids,
            },
        )
        if delivered.delivered_text is None:
            break
        if f"source_session_ids={','.join(source_ids)}" not in delivered.delivered_text:
            raise BackendExecutionError(
                "visible-token truncation removed source identity"
            )
        if delivered.decision == "truncated":
            truncated += 1
        primary = session_by_id[source_ids[0]]
        event = contract._event_payload(  # noqa: SLF001 - shared frozen schema
            rank=int(record["rank"]),
            session=primary,
            raw_text=raw_text,
            delivered_text=delivered.delivered_text,
            raw_tokens=tokenizer.count(raw_text),
            delivered_tokens=delivered.delivered_tokens,
            decision=(
                "delivered_truncated"
                if delivered.decision == "truncated"
                else "delivered_full"
            ),
            cumulative=delivered.cumulative_visible_tokens,
        )
        event.update(
            {
                "backend_record_id": record["backend_record_id"],
                "source_session_ids": source_ids,
                "source_dataset_session_indices": record[
                    "source_dataset_session_indices"
                ],
                "source_document_sha256s": record["source_document_sha256s"],
            }
        )
        events.append(event)
        pieces.append(delivered.delivered_text)
        all_delivered_sources.extend(source_ids)
        if delivered.decision == "truncated":
            break
    context = "".join(pieces)
    unique_sources = list(dict.fromkeys(all_delivered_sources))
    answer_input = contract._base_answer_input(  # noqa: SLF001
        method=method,
        item=item,
        item_index=item_index,
        tokenizer=tokenizer,
        context=context,
        context_events=events,
        budget={
            "policy": "hard_visible_total",
            "configured_visible_budget_tokens": contract.VISIBLE_BUDGET_TOKENS,
            "overflow_policy": "truncate_current_then_stop",
            "truncated_events": truncated,
            "cumulative_visible_tokens": gate.cumulative_visible_tokens,
            "exhausted": gate.exhausted,
        },
    )
    answer_input["context"]["source_session_ids"] = unique_sources
    if answer_input["context"]["visible_tokens"] != gate.cumulative_visible_tokens:
        raise BackendExecutionError("visible-token gate and frozen context differ")
    private_audit = {
        "source_mapping": contract._source_mapping(item, unique_sources),  # noqa: SLF001
        "raw_session_count": len(sessions),
        "retrieval_record_count": len(records),
        "delivered_record_count": len(events),
        "delivered_session_count": len(unique_sources),
    }
    return answer_input, private_audit


def _completed_item(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    item_dir: Path,
    formal: bool,
) -> dict[str, Any]:
    item_index = int(plan_record["dataset_index"])
    method = str(plan["method"])
    ledger_path = item_dir / "attempts.jsonl"
    answer_path = item_dir / "answer_input.json"
    checkpoint_path = item_dir / "checkpoint.json"
    events = contract.read_jsonl(ledger_path)
    terminal = contract.validate_item_ledger(
        events,
        method=method,
        item_index=item_index,
        question_id=str(item["question_id"]),
        require_complete=True,
    )
    if terminal is None:
        raise BackendExecutionError(f"item {item_index} terminal is missing")
    required = {
        "answer": answer_path,
        "checkpoint": checkpoint_path,
        "source_trace": item_dir / "source_trace.json",
        "retrieval": item_dir / "retrieval_raw.json",
        "proxy_slice": item_dir / "proxy_slice.json",
        "budget_trace": item_dir / "retrieval_budget.jsonl",
        "budget_manifest": item_dir / "retrieval_budget.manifest.json",
        "model_ledger": item_dir / "model_evidence/ledger.jsonl",
    }
    for label, path in required.items():
        if path.is_symlink() or not path.is_file():
            raise BackendExecutionError(f"item {item_index} {label} is missing")
    answer = contract.read_json(answer_path)
    checkpoint = contract.read_json(checkpoint_path)
    if not isinstance(answer, dict) or not isinstance(checkpoint, dict):
        raise BackendExecutionError(f"item {item_index} artifact is invalid")
    contract.validate_answer_input_allowlist(answer)
    if (
        checkpoint.get("schema_version") != contract.CHECKPOINT_SCHEMA_VERSION
        or checkpoint.get("backend_execution_schema")
        != backend_contract.BACKEND_RUN_SCHEMA
        or checkpoint.get("status") != "complete"
        or checkpoint.get("method") != method
        or checkpoint.get("dataset_index") != item_index
        or checkpoint.get("question_id") != item["question_id"]
        or checkpoint.get("history_content_sha256")
        != contract.history_content_hash(item)
        or checkpoint.get("history_owner_sha256") != contract.history_owner_hash(item)
        or checkpoint.get("answer_input_sha256") != contract.sha256_file(answer_path)
        or checkpoint.get("ledger_sha256") != contract.sha256_file(ledger_path)
        or terminal.get("answer_input_sha256") != contract.sha256_file(answer_path)
        or terminal.get("model_ledger_sha256")
        != contract.sha256_file(required["model_ledger"])
    ):
        raise BackendExecutionError(f"item {item_index} checkpoint evidence differs")
    model_summary = backend_contract.model_ledger_summary(
        ledger_path=required["model_ledger"],
        artifact_root=item_dir / "model_evidence",
        formal=formal,
    )
    if (
        checkpoint.get("model_calls") != model_summary["model_calls"]
        or checkpoint.get("network_calls") != model_summary["network_calls"]
    ):
        raise BackendExecutionError(f"item {item_index} model accounting differs")
    workspace = Path(plan_record["workspace"]["workspace"])
    final_snapshot = snapshot_memory_path(workspace).descriptor
    if checkpoint.get("workspace_after_retrieval") != final_snapshot:
        raise BackendExecutionError(f"item {item_index} final workspace hash differs")
    return {
        "dataset_index": item_index,
        "question_id": item["question_id"],
        "abstention": str(item["question_id"]).endswith("_abs"),
        "item_dir": item_dir.name,
        "answer_input_sha256": contract.sha256_file(answer_path),
        "checkpoint_sha256": contract.sha256_file(checkpoint_path),
        "ledger_sha256": contract.sha256_file(ledger_path),
        "history_content_sha256": checkpoint["history_content_sha256"],
        "history_owner_sha256": checkpoint["history_owner_sha256"],
        "visible_tokens": answer["context"]["visible_tokens"],
        "rendered_prompt_tokens": answer["answer_protocol_interface"][
            "rendered_prompt_tokens"
        ],
        "delivered_source_session_ids": answer["context"]["source_session_ids"],
        "source_mapping_session_recall": checkpoint["private_audit"]["source_mapping"][
            "session_recall"
        ],
        "model_calls": model_summary["model_calls"],
        "network_calls": model_summary["network_calls"],
        "workspace": str(workspace),
        "workspace_after_retrieval_sha256": final_snapshot["sha256"],
        "model_ledger_sha256": model_summary["ledger_sha256"],
        "proxy_slice_sha256": contract.sha256_file(required["proxy_slice"]),
        "source_trace_sha256": contract.sha256_file(required["source_trace"]),
        "retrieval_sha256": contract.sha256_file(required["retrieval"]),
        "budget_trace_sha256": contract.sha256_file(required["budget_trace"]),
        "budget_manifest_sha256": contract.sha256_file(required["budget_manifest"]),
    }


def _process_item(
    *,
    plan: Mapping[str, Any],
    item: Mapping[str, Any],
    plan_record: Mapping[str, Any],
    preregistration_sha256: str,
    tokenizer: contract.FormalTokenizer,
    proxy_base_url: str,
    proxy_log: Path | None,
    adapter_factory: AdapterFactory,
    formal: bool,
) -> dict[str, Any]:
    method = str(plan["method"])
    item_index = int(plan_record["dataset_index"])
    output_root = Path(str(plan["output_root"]))
    item_dir = (
        output_root
        / method
        / "items"
        / contract.item_directory_name(item_index, str(item["question_id"]))
    )
    item_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = item_dir / "attempts.jsonl"
    checkpoint_path = item_dir / "checkpoint.json"
    answer_path = item_dir / "answer_input.json"
    with contract.advisory_lock(item_dir / ".item.lock"):
        events = contract.read_jsonl(ledger_path)
        if events:
            terminal = contract.validate_item_ledger(
                events,
                method=method,
                item_index=item_index,
                question_id=str(item["question_id"]),
                require_complete=False,
            )
            if terminal is not None:
                return _completed_item(
                    plan=plan,
                    item=item,
                    plan_record=plan_record,
                    item_dir=item_dir,
                    formal=formal,
                )
        else:
            unexpected = [
                path.name for path in item_dir.iterdir() if path.name != ".item.lock"
            ]
            if unexpected:
                raise BackendExecutionError(
                    f"item {item_index} has artifacts without an attempt ledger: "
                    f"{sorted(unexpected)}"
                )
        attempt_id = str(uuid.uuid4())
        configuration_sha256 = contract.canonical_hash(plan["configuration"])
        started = contract.append_jsonl_fsync(
            ledger_path,
            {
                "event": "attempt_started",
                "attempt_id": attempt_id,
                "method": method,
                "dataset_index": item_index,
                "question_id": item["question_id"],
                "history_content_sha256": contract.history_content_hash(item),
                "history_owner_sha256": contract.history_owner_hash(item),
                "preregistration_sha256": preregistration_sha256,
                "configuration_sha256": configuration_sha256,
                "plan_content_sha256": plan["plan_content_sha256"],
                "requested_builder_model": contract.EXPECTED_MODEL,
            },
        )
        workspace = Path(plan_record["workspace"]["workspace"])
        if workspace.exists() or workspace.is_symlink():
            raise BackendExecutionError(
                f"item {item_index} backend workspace already exists"
            )
        workspace.mkdir(parents=True)
        workspace_before = snapshot_memory_path(workspace).descriptor
        _atomic_json_no_clobber(
            item_dir / "workspace_before.json",
            {
                "schema_version": backend_contract.BACKEND_RUN_SCHEMA,
                "dataset_index": item_index,
                "question_id": item["question_id"],
                "snapshot": workspace_before,
            },
        )
        model_root = item_dir / "model_evidence"
        model_root.mkdir()
        model_ledger_path = model_root / "ledger.jsonl"
        model_ledger = durable.HashChainLedger(
            model_ledger_path,
            run_id=f"{plan['plan_content_sha256']}:{method}:{item_index:04d}",
        )
        proxy_start = durable.proxy_prefix(proxy_log)
        adapter: BackendAdapter | None = None
        try:
            init_path = model_root / "operations/initialize.json"

            def initialize() -> Mapping[str, Any]:
                nonlocal adapter
                adapter = adapter_factory(
                    method,
                    workspace=plan_record["workspace"],
                    item=item,
                    item_index=item_index,
                    proxy_base_url=proxy_base_url,
                    proxy_log=proxy_log,
                    ledger=model_ledger,
                    artifact_root=model_root,
                    tokenizer=tokenizer,
                    formal=formal,
                )
                return {
                    "schema_version": backend_contract.BACKEND_RUN_SCHEMA,
                    "operation": "initialize",
                    "method": method,
                    "workspace": plan_record["workspace"],
                }

            _operation(
                ledger=model_ledger,
                adapter=None,
                operation_id="initialize",
                kind="backend_initialize",
                input_sha256=contract.canonical_hash(plan_record["workspace"]),
                artifact_root=model_root,
                artifact_path=init_path,
                action=initialize,
            )
            if adapter is None:
                raise BackendExecutionError("backend adapter initialization failed")
            sessions = contract.rendered_sessions(item)
            ingest_records: list[dict[str, Any]] = []
            for session in sessions:
                operation_id = f"ingest-{int(session['dataset_session_index']):04d}"
                receipt_path = model_root / "operations" / f"{operation_id}.json"
                receipt = _operation(
                    ledger=model_ledger,
                    adapter=adapter,
                    operation_id=operation_id,
                    kind="history_ingest",
                    input_sha256=str(session["text_sha256"]),
                    artifact_root=model_root,
                    artifact_path=receipt_path,
                    action=lambda session=session: adapter.ingest(session),
                )
                ingest_records.append(receipt)
            source_trace = {
                "schema_version": backend_contract.BACKEND_SOURCE_TRACE_SCHEMA,
                "method": method,
                "dataset_index": item_index,
                "question_id": item["question_id"],
                "history_content_sha256": contract.history_content_hash(item),
                "history_owner_sha256": contract.history_owner_hash(item),
                "session_count": len(sessions),
                "sessions": [
                    {
                        **session,
                        "text": None,
                        "ingest": receipt,
                    }
                    for session, receipt in zip(sessions, ingest_records)
                ],
            }
            _atomic_json_no_clobber(item_dir / "source_trace.json", source_trace)
            workspace_after_build = snapshot_memory_path(workspace).descriptor
            raw_records: list[dict[str, Any]] = []

            def retrieve() -> Mapping[str, Any]:
                nonlocal raw_records
                raw_records = _normalize_retrieval_records(
                    method=method,
                    item=item,
                    records=adapter.retrieve(str(item["question"])),
                )
                return {
                    "schema_version": backend_contract.BACKEND_RETRIEVAL_SCHEMA,
                    "method": method,
                    "dataset_index": item_index,
                    "question_id": item["question_id"],
                    "question_sha256": contract.sha256_text(str(item["question"])),
                    "retrieval_limit": backend_contract.RETRIEVAL_LIMIT,
                    "workspace_after_build": workspace_after_build,
                    "record_count": len(raw_records),
                    "records": raw_records,
                }

            retrieval = _operation(
                ledger=model_ledger,
                adapter=adapter,
                operation_id="retrieve",
                kind="backend_retrieve",
                input_sha256=contract.sha256_text(str(item["question"])),
                artifact_root=model_root,
                artifact_path=model_root / "operations/retrieve.json",
                action=retrieve,
            )
            _atomic_json_no_clobber(item_dir / "retrieval_raw.json", retrieval)
            workspace_after_search = snapshot_memory_path(workspace).descriptor
            gate = VisibleTokenBudgetGate(
                trace_path=item_dir / "retrieval_budget.jsonl",
                manifest_path=item_dir / "retrieval_budget.manifest.json",
                run_id=f"{plan['plan_content_sha256']}:{method}:{item_index:04d}:gate",
                configured_budget_tokens=contract.VISIBLE_BUDGET_TOKENS,
                tokenizer=tokenizer,
                memory_before=snapshot_memory_path(workspace),
                overflow_policy="truncate",
                metadata={
                    "method": method,
                    "dataset_index": item_index,
                    "question_id": item["question_id"],
                },
            )
            with gate:
                answer_input, private_audit = _freeze_retrieval(
                    method=method,
                    item=item,
                    item_index=item_index,
                    tokenizer=tokenizer,
                    records=raw_records,
                    gate=gate,
                )
                _atomic_json_no_clobber(answer_path, answer_input)
                _operation(
                    ledger=model_ledger,
                    adapter=adapter,
                    operation_id="close",
                    kind="backend_close",
                    input_sha256=workspace_after_search["sha256"],
                    artifact_root=model_root,
                    artifact_path=model_root / "operations/close.json",
                    action=lambda: (
                        adapter.close()
                        or {
                            "schema_version": backend_contract.BACKEND_RUN_SCHEMA,
                            "operation": "close",
                        }
                    ),
                )
                adapter = None
                workspace_after_retrieval = snapshot_memory_path(workspace).descriptor
                gate.finalize(
                    memory_after=snapshot_memory_path(workspace),
                    actual_model_usage=_usage_summary(model_ledger.records),
                    metadata={"answer_input_sha256": contract.sha256_file(answer_path)},
                )
            model_summary = backend_contract.model_ledger_summary(
                ledger_path=model_ledger_path,
                artifact_root=model_root,
                formal=formal,
            )
            proxy_value = backend_contract.proxy_slice(
                path=proxy_log,
                start=proxy_start,
                logical_call_ids=model_summary["logical_call_ids"],
                formal=formal,
            )
            _atomic_json_no_clobber(item_dir / "proxy_slice.json", proxy_value)
            terminal = contract.append_jsonl_fsync(
                ledger_path,
                {
                    "event": "attempt_completed",
                    "attempt_id": attempt_id,
                    "method": method,
                    "dataset_index": item_index,
                    "question_id": item["question_id"],
                    "start_event_id": started["event_id"],
                    "answer_input_sha256": contract.sha256_file(answer_path),
                    "answer_input_content_sha256": contract.canonical_hash(
                        answer_input
                    ),
                    "private_audit_sha256": contract.canonical_hash(private_audit),
                    "model_ledger_sha256": model_summary["ledger_sha256"],
                    "source_trace_sha256": contract.sha256_file(
                        item_dir / "source_trace.json"
                    ),
                    "retrieval_sha256": contract.sha256_file(
                        item_dir / "retrieval_raw.json"
                    ),
                    "proxy_slice_sha256": contract.sha256_file(
                        item_dir / "proxy_slice.json"
                    ),
                    "visible_tokens": answer_input["context"]["visible_tokens"],
                    "rendered_prompt_tokens": answer_input["answer_protocol_interface"][
                        "rendered_prompt_tokens"
                    ],
                    "model_calls": model_summary["model_calls"],
                    "network_calls": model_summary["network_calls"],
                },
            )
            checkpoint = {
                "schema_version": contract.CHECKPOINT_SCHEMA_VERSION,
                "backend_execution_schema": backend_contract.BACKEND_RUN_SCHEMA,
                "status": "complete",
                "method": method,
                "scope": plan["scope"],
                "dataset_index": item_index,
                "question_id": item["question_id"],
                "attempt_id": attempt_id,
                "start_event_id": started["event_id"],
                "terminal_event_id": terminal["event_id"],
                "history_owner_question_id": item["question_id"],
                "history_content_sha256": contract.history_content_hash(item),
                "history_owner_sha256": contract.history_owner_hash(item),
                "preregistration_sha256": preregistration_sha256,
                "configuration_sha256": configuration_sha256,
                "plan_content_sha256": plan["plan_content_sha256"],
                "answer_input": answer_path.name,
                "answer_input_sha256": contract.sha256_file(answer_path),
                "ledger": ledger_path.name,
                "ledger_sha256": contract.sha256_file(ledger_path),
                "model_ledger": "model_evidence/ledger.jsonl",
                "model_ledger_sha256": model_summary["ledger_sha256"],
                "source_trace_sha256": contract.sha256_file(
                    item_dir / "source_trace.json"
                ),
                "retrieval_sha256": contract.sha256_file(
                    item_dir / "retrieval_raw.json"
                ),
                "proxy_slice_sha256": contract.sha256_file(
                    item_dir / "proxy_slice.json"
                ),
                "budget_trace_sha256": contract.sha256_file(
                    item_dir / "retrieval_budget.jsonl"
                ),
                "budget_manifest_sha256": contract.sha256_file(
                    item_dir / "retrieval_budget.manifest.json"
                ),
                "workspace_before": workspace_before,
                "workspace_after_build": workspace_after_build,
                "workspace_after_search": workspace_after_search,
                "workspace_after_retrieval": workspace_after_retrieval,
                "private_audit": private_audit,
                "answer_stage": "not_started",
                "requested_builder_model": contract.EXPECTED_MODEL,
                "actual_builder_model": (
                    contract.EXPECTED_MODEL if model_summary["model_calls"] else None
                ),
                "model_calls": model_summary["model_calls"],
                "network_calls": model_summary["network_calls"],
                "completed_at": contract.utc_now(),
            }
            contract.atomic_json_no_clobber(checkpoint_path, checkpoint)
            return _completed_item(
                plan=plan,
                item=item,
                plan_record=plan_record,
                item_dir=item_dir,
                formal=formal,
            )
        finally:
            if adapter is not None:
                try:
                    adapter.close()
                except Exception:  # noqa: BLE001 - original failure remains primary
                    pass


def execute_backend_plan(
    *,
    plan_path: Path,
    dataset_path: Path,
    preregistration_path: Path,
    adapter_factory: AdapterFactory,
    proxy_base_url: str,
    proxy_log: Path | None,
    formal: bool,
    smoke_audit_path: Path | None = None,
) -> dict[str, Any]:
    """Execute or validate every item in one frozen backend plan."""

    plan_path = plan_path.expanduser().resolve()
    dataset_path = dataset_path.expanduser().resolve()
    preregistration_path = preregistration_path.expanduser().resolve()
    try:
        plan, data, _ = backend_contract.validate_plan(
            plan_path=plan_path,
            dataset_path=dataset_path,
            preregistration_path=preregistration_path,
        )
    except backend_contract.BackendContractError as exc:
        raise BackendExecutionError(str(exc)) from exc
    method = str(plan["method"])
    run_dir = Path(str(plan["output_root"])) / method
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "items").mkdir(exist_ok=True)
    identity = _run_identity(
        plan=plan,
        plan_path=plan_path,
        dataset_path=dataset_path,
        preregistration_path=preregistration_path,
        formal=formal,
    )
    run_id = contract.canonical_hash(identity)
    if plan["scope"] == "formal":
        if not formal:
            pass
        elif smoke_audit_path is None:
            raise BackendExecutionError("formal plan requires --smoke-audit")
        else:
            try:
                smoke_report = backend_contract.validate_smoke_audit(
                    path=smoke_audit_path,
                    method=method,
                    formal_workspace_paths=[
                        Path(record["workspace"]["workspace"])
                        for record in plan["items"]
                    ],
                )
                from baselines.longmemeval_m1.audit_longmemeval_m1_backends import (  # noqa: PLC0415
                    audit_backend_run,
                )

                fresh_smoke_report = audit_backend_run(
                    plan_path=Path(str(smoke_report["plan_path"])),
                    dataset_path=dataset_path,
                    preregistration_path=preregistration_path,
                    formal=True,
                )
                if any(
                    fresh_smoke_report.get(field) != smoke_report.get(field)
                    for field in (
                        "run_manifest_sha256",
                        "workspace_paths",
                        "model_calls",
                        "network_calls",
                    )
                ):
                    raise BackendExecutionError(
                        "persisted smoke audit differs from a fresh offline audit"
                    )
            except (
                backend_contract.BackendContractError,
                contract.ContractError,
            ) as exc:
                raise BackendExecutionError(str(exc)) from exc
    manifest_path = run_dir / "run_manifest.json"
    with contract.advisory_lock(run_dir / ".run.lock"):
        if manifest_path.exists():
            manifest = contract.read_json(manifest_path)
            if not isinstance(manifest, dict) or manifest.get("identity") != identity:
                raise BackendExecutionError(
                    "existing backend run manifest identity differs"
                )
        else:
            manifest = _new_manifest(
                identity=identity,
                run_dir=run_dir,
                expected_items=len(plan["items"]),
            )
            manifest["run_id"] = run_id
            contract.atomic_json_replace(manifest_path, manifest)
        expected_names = {
            contract.item_directory_name(
                int(record["dataset_index"]), str(record["question_id"])
            )
            for record in plan["items"]
        }
        actual_names = {
            path.name
            for path in (run_dir / "items").iterdir()
            if path.is_dir() and not path.is_symlink()
        }
        if not actual_names.issubset(expected_names):
            raise BackendExecutionError("backend item directory inventory differs")
        completed: list[dict[str, Any]] = []
        for plan_record in plan["items"]:
            item_index = int(plan_record["dataset_index"])
            item = data[item_index]
            try:
                completed_item = _process_item(
                    plan=plan,
                    item=item,
                    plan_record=plan_record,
                    preregistration_sha256=contract.sha256_file(preregistration_path),
                    tokenizer=contract.FormalTokenizer.resolve(),
                    proxy_base_url=proxy_base_url,
                    proxy_log=proxy_log,
                    adapter_factory=adapter_factory,
                    formal=formal,
                )
            except (
                contract.ContractError,
                backend_contract.BackendContractError,
            ) as exc:
                raise BackendExecutionError(str(exc)) from exc
            completed.append(completed_item)
            manifest["status"] = "running"
            manifest["updated_at"] = contract.utc_now()
            manifest["completed_items"] = len(completed)
            manifest["abstention_items"] = sum(row["abstention"] for row in completed)
            manifest["model_calls"] = sum(row["model_calls"] for row in completed)
            manifest["network_calls"] = sum(row["network_calls"] for row in completed)
            manifest["items"] = completed
            contract.atomic_json_replace(manifest_path, manifest)
        item_inventory = contract.regular_tree_inventory(run_dir / "items")
        manifest["status"] = "inputs_complete"
        manifest["updated_at"] = contract.utc_now()
        manifest["completed_at"] = contract.utc_now()
        manifest["completed_items"] = len(completed)
        manifest["items"] = completed
        manifest["model_calls"] = sum(row["model_calls"] for row in completed)
        manifest["network_calls"] = sum(row["network_calls"] for row in completed)
        manifest["item_tree_inventory"] = {
            "entries": len(item_inventory),
            "bytes": sum(entry["bytes"] for entry in item_inventory),
            "root_sha256": contract.inventory_root(item_inventory),
        }
        manifest["visible_token_summary"] = {
            "min": min(row["visible_tokens"] for row in completed),
            "max": max(row["visible_tokens"] for row in completed),
            "sum": sum(row["visible_tokens"] for row in completed),
        }
        manifest["rendered_prompt_token_summary"] = {
            "min": min(row["rendered_prompt_tokens"] for row in completed),
            "max": max(row["rendered_prompt_tokens"] for row in completed),
            "sum": sum(row["rendered_prompt_tokens"] for row in completed),
        }
        contract.atomic_json_replace(manifest_path, manifest)
        return manifest


def _next_provider_evidence_dir(run_dir: Path) -> tuple[Path, int]:
    root = run_dir / "provider_evidence"
    root.mkdir(parents=True, exist_ok=True)
    numbers: list[int] = []
    for path in root.iterdir():
        match = re.fullmatch(r"invocation-(\d{4})", path.name)
        if match:
            numbers.append(int(match.group(1)))
    number = max(numbers, default=0) + 1
    return root / f"invocation-{number:04d}", number


def _provider_evidence_inventory(run_dir: Path) -> dict[str, Any]:
    root = run_dir / "provider_evidence"
    rows: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.glob("invocation-*/invocation.json")):
            try:
                report = flex_evidence.audit_invocation(contract.read_json(path))
            except (flex_evidence.EvidenceError, contract.ContractError) as exc:
                raise BackendExecutionError(
                    f"Flex provider invocation audit failed: {exc}"
                ) from exc
            rows.append(
                {
                    "path": str(path.relative_to(run_dir)),
                    "sha256": contract.sha256_file(path),
                    "requests": report["requests"],
                    "segment_sha256": report["segment_sha256"],
                    "committed_cost_nanos": report["committed_cost_nanos"],
                }
            )
    return {
        "schema": "longmemeval-m1-flex-provider-evidence/v1",
        "transport": "exclusive-child-proxy-to-openai-gpt55-flex-gateway",
        "invocations": rows,
        "requests": sum(int(row["requests"]) for row in rows),
    }


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    if not args.allow_model_requests:
        raise BackendExecutionError(
            "formal backend execution requires --allow-model-requests"
        )
    plan, _, _ = backend_contract.validate_plan(
        plan_path=args.plan,
        dataset_path=args.dataset,
        preregistration_path=args.preregistration,
    )
    run_dir = Path(str(plan["output_root"])) / str(plan["method"])
    identity = _run_identity(
        plan=plan,
        plan_path=args.plan.expanduser().resolve(),
        dataset_path=args.dataset.expanduser().resolve(),
        preregistration_path=args.preregistration.expanduser().resolve(),
        formal=True,
    )
    evidence_dir, invocation_number = _next_provider_evidence_dir(run_dir)
    invocation = None
    try:
        invocation = flex_evidence.begin_child_invocation(
            args.gateway_root,
            evidence_dir,
            run_id=(
                f"lme-m1-{plan['method']}-{contract.canonical_hash(identity)[:20]}-"
                f"{invocation_number:04d}"
            ),
        )
    except flex_evidence.EvidenceError as exc:
        raise BackendExecutionError(str(exc)) from exc
    proxy = ManagedProxy(
        run_dir=run_dir,
        run_id=contract.canonical_hash(identity),
        upstream=invocation.base_url,
    )
    result: dict[str, Any] | None = None
    try:
        proxy.start()
        if proxy.base_url is None or proxy.log is None:
            raise BackendExecutionError("exclusive proxy did not publish its binding")
        result = execute_backend_plan(
            plan_path=args.plan,
            dataset_path=args.dataset,
            preregistration_path=args.preregistration,
            adapter_factory=production_adapter_factory,
            proxy_base_url=proxy.base_url,
            proxy_log=proxy.log,
            formal=True,
            smoke_audit_path=args.smoke_audit,
        )
    finally:
        proxy.stop()
        try:
            invocation.finish()
        except flex_evidence.EvidenceError as exc:
            raise BackendExecutionError(
                f"cannot close Flex provider invocation: {exc}"
            ) from exc
    if result is None:
        raise BackendExecutionError("backend execution produced no manifest")
    provider_evidence = _provider_evidence_inventory(run_dir)
    if provider_evidence["requests"] != result.get("network_calls"):
        raise BackendExecutionError(
            "backend model-call count differs from exact Flex provider requests"
        )
    manifest_path = run_dir / "run_manifest.json"
    with contract.advisory_lock(run_dir / ".run.lock"):
        current = contract.read_json(manifest_path)
        if not isinstance(current, dict) or current.get("identity") != identity:
            raise BackendExecutionError("backend manifest changed before provider binding")
        current["provider_evidence"] = provider_evidence
        current["updated_at"] = contract.utc_now()
        contract.atomic_json_replace(manifest_path, current)
    return current


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=contract.DEFAULT_DATASET)
    parser.add_argument("--preregistration", type=Path, required=True)
    parser.add_argument(
        "--gateway-root",
        type=Path,
        required=True,
        help="active OpenAI GPT-5.5 Flex gateway result root",
    )
    parser.add_argument(
        "--allow-model-requests",
        action="store_true",
        help="required acknowledgement that backend construction sends GPT-5.5 requests",
    )
    parser.add_argument(
        "--smoke-audit",
        type=Path,
        help="required for a formal-500 plan; must be a passed isolated one-item audit",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    args.gateway_root = args.gateway_root.expanduser().resolve()
    result = run_formal(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

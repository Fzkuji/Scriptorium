"""In-process shim launcher for the EverMemOS/EverOS server (used by run_evermemos.py).

Why this exists (instead of `everos server start`):

1. Local embedder injection. EverOS only ships an OpenAI-compatible HTTP
   embedding provider (everos/component/embedding/openai_provider.py) and
   build_embedding_provider() hard-requires model/api_key/base_url. Our
   protocol mandates a LOCAL sentence-transformers all-MiniLM-L6-v2 embedder.
   This launcher monkeypatches `build_embedding_provider` IN PROCESS (both
   `everos.component.embedding` package attr and the `.factory` module attr,
   BEFORE `create_app()` imports the cascade lifespan which binds the name at
   module import time) to return a local MiniLM provider implementing the
   EmbeddingProvider protocol. No file in third_party/evermemos is modified.

2. Dimension note: every EverOS LanceDB table hardcodes Vector(1024)
   (everos/infra/persistence/lancedb/tables/*.py: _DIM = 1024). MiniLM is
   384d, so the provider zero-pads 384 -> 1024. Zero-padding both stored and
   query vectors preserves cosine similarity exactly, so retrieval ranking is
   unchanged vs a native 384d index.

3. Drain visibility. Extraction is asynchronous: /memory/flush writes episode
   markdown, then the OME engine (atomic facts etc.) and the cascade daemon
   (md -> LanceDB indexing) finish in the background. The adapter must not
   search before indexing settles, so this launcher adds a GET /debug/drained
   route exposing OME idleness + cascade queue counters + LanceDB row counts.

Usage: <evermemos-venv-python> src/adapters/_evermemos_server_shim.py <port>
Env (set by run_evermemos.py): EVEROS_ROOT, EVEROS_LLM__*, EVEROS_EMBEDDING__*
(dummy values so the settings guard in everos/service/search.py passes; the
patched factory ignores them), EVEROS_MEMORIZE__MODE=chat.
"""

import asyncio
import sys


def _install_local_embedder():
    """Monkeypatch build_embedding_provider with a local MiniLM provider."""
    import everos.component.embedding as emb_pkg
    import everos.component.embedding.factory as emb_factory

    from sentence_transformers import SentenceTransformer

    class LocalMiniLMProvider:
        """Local sentence-transformers embedder satisfying EmbeddingProvider.

        dim=1024 (LanceDB table shape); real MiniLM output is 384d and is
        zero-padded, which preserves cosine ordering (see module docstring).
        """

        dim = 1024
        _NATIVE_DIM = 384

        def __init__(self):
            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

        def _pad(self, vec):
            out = list(map(float, vec)) + [0.0] * (self.dim - self._NATIVE_DIM)
            return out

        async def embed(self, text):
            return (await self.embed_batch([text]))[0]

        async def embed_batch(self, texts):
            texts = list(texts)
            if not texts:
                return []
            vecs = await asyncio.to_thread(
                self._model.encode, texts, show_progress_bar=False
            )
            return [self._pad(v) for v in vecs]

    provider = LocalMiniLMProvider()

    def _patched_build_embedding_provider(settings, *, dim=1024):  # noqa: ARG001
        return provider

    emb_factory.build_embedding_provider = _patched_build_embedding_provider
    emb_pkg.build_embedding_provider = _patched_build_embedding_provider


def _add_debug_routes(app):
    """GET /debug/drained -> OME idle + cascade queue + LanceDB row counts."""

    async def drained():
        from everos.infra.persistence.lancedb.repos import (
            atomic_fact_repo,
            episode_repo,
        )
        from everos.infra.persistence.sqlite import md_change_state_repo
        from everos.service.memorize import _get_engine  # noqa: SLF001

        engine = _get_engine()
        ome_idle = await engine.wait_idle(timeout=0.05)
        s = await md_change_state_repo.queue_summary()
        episodes = await (await episode_repo._table()).count_rows()  # noqa: SLF001
        facts = await (await atomic_fact_repo._table()).count_rows()  # noqa: SLF001
        return {
            "ome_idle": ome_idle,
            "cascade_pending": s.pending,
            "cascade_lag": max(0, s.max_lsn - s.last_processed_lsn),
            "failed_retryable": s.failed_retryable,
            "failed_permanent": s.failed_permanent,
            "episodes": episodes,
            "atomic_facts": facts,
        }

    app.add_api_route("/debug/drained", drained, methods=["GET"])


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8377

    _install_local_embedder()

    from everos.core.observability.logging import configure_logging

    configure_logging(level="INFO")

    from everos.entrypoints.api.app import create_app

    app = create_app()
    _add_debug_routes(app)

    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    main()

"""Universal LLM usage tracker for baseline adapters.

Patches the OpenAI SDK at the lowest level (sync + async chat.completions
create), so EVERY LLM call made by any framework inside the adapter process
is counted — calls, prompt tokens, completion tokens, wall time. This works
because all 15 adapted frameworks use the bare OpenAI SDK (verified in
docs/experiments/baselines/baseline_adapters.md; no litellm bindings).

Usage in an adapter:
    from _usage_tracker import tracker
    tracker.install()

    tracker.reset("build")
    ... build memory ...
    build_usage = tracker.snapshot("build")
    # {'calls': N, 'tokens_in': X, 'tokens_out': Y, 'llm_time_s': T}

    tracker.reset("retrieval")   # per question: reset per phase or diff totals
    ...

Embeddings are local sentence-transformers everywhere → no API cost, not
counted. Matches LightMem Table 2 accounting (tokens in thousands, calls,
runtime) — see docs/experiments/protocols/unified_evaluation_protocol.md efficiency section.
"""

import functools
import threading
import time

_lock = threading.Lock()


class UsageTracker:
    def __init__(self):
        self._installed = False
        self._counters = {}  # phase -> dict
        # 并发答题下每题一个线程、各自一个 phase：_record 只记到调用线程的
        # phase（thread-local）；未设置时（如单线程 build）回退到"记进所有 phase"，
        # 保持原行为不变。
        self._tls = threading.local()

    # ------------------------------------------------------------------ #
    def install(self):
        """Patch openai SDK create methods (idempotent)."""
        if self._installed:
            return
        import openai.resources.chat.completions as _c

        tracker = self

        def _wrap_sync(orig):
            @functools.wraps(orig)
            def wrapper(*args, **kwargs):
                t0 = time.monotonic()
                resp = orig(*args, **kwargs)
                tracker._record(resp, time.monotonic() - t0)
                return resp
            return wrapper

        def _wrap_async(orig):
            @functools.wraps(orig)
            async def wrapper(*args, **kwargs):
                t0 = time.monotonic()
                resp = await orig(*args, **kwargs)
                tracker._record(resp, time.monotonic() - t0)
                return resp
            return wrapper

        _c.Completions.create = _wrap_sync(_c.Completions.create)
        _c.AsyncCompletions.create = _wrap_async(_c.AsyncCompletions.create)

        # Responses API (used by some newer frameworks) — patch if present
        try:
            import openai.resources.responses as _r
            _r.Responses.create = _wrap_sync(_r.Responses.create)
            _r.AsyncResponses.create = _wrap_async(_r.AsyncResponses.create)
        except Exception:  # noqa: BLE001 — older SDK without responses
            pass

        self._installed = True

    # ------------------------------------------------------------------ #
    def _record(self, resp, elapsed):
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "prompt_tokens", None)
        tout = getattr(usage, "completion_tokens", None)
        if tin is None:
            tin = getattr(usage, "input_tokens", 0) or 0   # responses API
        if tout is None:
            tout = getattr(usage, "output_tokens", 0) or 0
        phase = getattr(self._tls, "phase", None)
        with _lock:
            targets = ([self._counters[phase]] if phase in self._counters
                       else self._counters.values() if phase is None
                       else [])  # phase 已被 pop 的孤儿调用：不记
            for c in targets:
                c["calls"] += 1
                c["tokens_in"] += tin or 0
                c["tokens_out"] += tout or 0
                c["llm_time_s"] += elapsed

    # ------------------------------------------------------------------ #
    def bind_thread(self, phase):
        """把当前线程的所有 LLM 调用记账绑定到 phase（并发答题：每题一个 phase）。
        用 with 语法：`with tracker.bind_thread("q0"): ...`。退出后恢复。"""
        return _ThreadPhase(self, phase)

    def reset(self, phase="default"):
        with _lock:
            self._counters[phase] = {
                "calls": 0, "tokens_in": 0, "tokens_out": 0, "llm_time_s": 0.0}

    def snapshot(self, phase="default", pop=True):
        with _lock:
            c = dict(self._counters.get(phase, {
                "calls": 0, "tokens_in": 0, "tokens_out": 0, "llm_time_s": 0.0}))
            c["llm_time_s"] = round(c["llm_time_s"], 2)
            if pop:
                self._counters.pop(phase, None)
        return c


class _ThreadPhase:
    def __init__(self, tracker, phase):
        self._t, self._phase = tracker, phase

    def __enter__(self):
        self._prev = getattr(self._t._tls, "phase", None)
        self._t._tls.phase = self._phase
        return self

    def __exit__(self, *exc):
        self._t._tls.phase = self._prev
        return False


tracker = UsageTracker()

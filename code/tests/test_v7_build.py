import os, sys, json, importlib
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "src", "adapters"))

def test_build_memory_v7_calls_agent_and_rhythms(tmp_path, monkeypatch):
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v7")
    import nativemem, run_nativemem
    importlib.reload(run_nativemem)

    seen = {"chunks": 0, "tidy": 0, "reorg": 0}
    def fake_pca(chunk_text, chunk_date, memory_dir, dia_ids, running_summary="", mode="oneshot"):
        seen["chunks"] += 1
        assert dia_ids and dia_ids[0].startswith("D")  # dia_id 传进来了
        return f"summary {seen['chunks']}"
    monkeypatch.setattr(run_nativemem, "process_chunk_agent", fake_pca)
    monkeypatch.setattr(run_nativemem, "tidy_local", lambda *a, **k: seen.__setitem__("tidy", seen["tidy"]+1) or 0)
    # 让节奏3 一定触发
    monkeypatch.setattr(run_nativemem, "library_grew_past_threshold", lambda *a, **k: True)
    monkeypatch.setattr(run_nativemem, "reorganize_library", lambda *a, **k: seen.__setitem__("reorg", seen["reorg"]+1) or 0)
    monkeypatch.setattr(run_nativemem, "measure_library", lambda d: {"files":0,"dirs":0,"bytes":0,"entries":0})

    with open(os.path.join(_ROOT, "benchmarks/locomo/data/locomo10.json")) as f:
        conv = json.load(f)[0]["conversation"]
    run_nativemem.build_memory(conv, str(tmp_path / "mem"), max_sessions=2)

    assert seen["chunks"] > 0      # 每 chunk 走 agent
    assert seen["tidy"] >= 2       # 每 session 末 tidy（2 个 session）
    assert seen["reorg"] >= 1      # 增长触发全局重构

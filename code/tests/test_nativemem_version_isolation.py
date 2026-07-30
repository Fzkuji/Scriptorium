import pytest
from types import SimpleNamespace


def test_each_version_has_its_own_adapter():
    from src.nativemem_versions import load_version

    modules = {name: load_version(name) for name in ("v8", "v10", "v11")}

    assert len({module.__name__ for module in modules.values()}) == 3
    assert all(callable(module.build_memory) for module in modules.values())


def test_unknown_version_is_rejected():
    from src.nativemem_versions import load_version

    with pytest.raises(ValueError, match="unsupported NativeMem version"):
        load_version("v12")


def test_public_builder_dispatches_to_selected_version(monkeypatch, tmp_path):
    from src.adapters import run_nativemem

    called = []
    selected = SimpleNamespace(
        build_memory=lambda conv, memory_dir, max_sessions=None: called.append(
            (conv, memory_dir, max_sessions)
        ) or (1.0, 2)
    )
    monkeypatch.setenv("NATIVEMEM_PROMPT", "v11")
    monkeypatch.setattr(run_nativemem, "load_version", lambda name: selected)

    result = run_nativemem.build_memory({"session_1": []}, tmp_path, 1)

    assert result == (1.0, 2)
    assert called == [({"session_1": []}, tmp_path, 1)]

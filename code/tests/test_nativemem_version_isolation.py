import pytest


def test_each_version_has_its_own_adapter():
    from src.nativemem_versions import load_version

    modules = {name: load_version(name) for name in ("v8", "v10", "v11")}

    assert len({module.__name__ for module in modules.values()}) == 3
    assert all(callable(module.build_memory) for module in modules.values())


def test_unknown_version_is_rejected():
    from src.nativemem_versions import load_version

    with pytest.raises(ValueError, match="unsupported NativeMem version"):
        load_version("v12")

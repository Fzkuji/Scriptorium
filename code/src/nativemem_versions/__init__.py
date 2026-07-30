"""Version-specific NativeMem implementations."""

from importlib import import_module


def load_version(name: str):
    if name not in {"v8", "v10", "v11"}:
        raise ValueError(f"unsupported NativeMem version: {name}")
    return import_module(f"src.nativemem_versions.{name}.adapter")

"""NativeMem frozen-memory ablation runner."""


def main() -> int:
    from .runner import main as run

    return run()


__all__ = ["main"]

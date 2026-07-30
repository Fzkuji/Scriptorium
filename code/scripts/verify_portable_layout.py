#!/usr/bin/env python3
from pathlib import Path
import sys


def project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "Model-Aligned-Wiki.html").is_file():
            return parent
    raise RuntimeError("project root not found")


def main() -> int:
    root = project_root()
    required = (
        "README.md",
        "requirements.txt",
        "requirements-dev.txt",
        "setup.sh",
        "Model-Aligned-Wiki.html",
        "docs/experiment-plan.html",
    )
    missing = [name for name in required if not (root / name).is_file()]
    for name in (
        "src",
        "scripts",
        "tests",
        "benchmarks",
        "experiments",
        "figures",
        "gold_memory",
        "results",
        "third_party",
    ):
        path = root / name
        if not path.is_symlink() or path.resolve() != (root / "code" / name).resolve():
            missing.append(f"{name} -> code/{name}")
    if missing:
        print("Portable layout check failed:", file=sys.stderr)
        for item in missing:
            print(f"  - {item}", file=sys.stderr)
        return 1
    print("Portable layout check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

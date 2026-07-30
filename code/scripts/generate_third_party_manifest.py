#!/usr/bin/env python3
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
THIRD_PARTY = ROOT / "code" / "third_party"


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


repositories = []
for child in sorted(THIRD_PARTY.iterdir()):
    if not (child / ".git").exists():
        continue
    repositories.append(
        {
            "name": child.name,
            "remote": git(child, "remote", "get-url", "origin"),
            "commit": git(child, "rev-parse", "HEAD"),
            "dirty": bool(git(child, "status", "--porcelain")),
        }
    )

manifest = {
    "generated_at": datetime.now(timezone.utc).isoformat(),
    "repositories": repositories,
}
(THIRD_PARTY / "manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
)
print(f"Wrote {len(repositories)} repositories to {THIRD_PARTY / 'manifest.json'}")

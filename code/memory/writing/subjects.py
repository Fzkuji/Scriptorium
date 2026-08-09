"""Where a fact lands: subject and kind decide the path, not the model.

One folder per sort of subject, one file per subject. The model names the
subject; where that lands is a rule, and a rule the model re-derives is a
rule it gets wrong differently every time — "Caroline" filed once under
people/ and once under relationships/caroline-and-melanie.md is two subjects
where there is one person.
"""

from __future__ import annotations

import re
from pathlib import Path

_FOLDERS = {
    "person": "people",
    "project": "projects",
    "relationship": "relationships",
    "theme": "themes",
}


def path_for(subject: str, kind: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", subject.lower()).strip("-")
    if not slug:
        raise ValueError(f"subject has no name in it: {subject!r}")
    return f"topics/{_FOLDERS.get(kind, 'themes')}/{slug}.md"


def known_subjects(memory_dir: str | Path) -> str:
    """The subjects memory already has, so a name is reused and not reinvented.

    Two turns of every pass used to go on listing the workspace to find this
    out, and a subject spelled a second way is a second file about one person.
    """
    root = Path(memory_dir) / "topics"
    names = sorted(
        f"{path.stem.replace('-', ' ')} ({path.parent.name})"
        for path in root.rglob("*.md")
    ) if root.is_dir() else []
    if not names:
        return "\nMemory is empty; every subject here is a new one.\n"
    return (
        "\nSubjects memory already holds, to reuse rather than restate:\n"
        + "".join(f"  {name}\n" for name in names)
    )

"""Deterministic normalization and validation of Topic edits."""

import hashlib
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from ..markdown import (
    BLOCK_ID_LENGTH,
    definition_match,
    parse_topic_tree,
    render_definition,
)


class TopicNormalizationMixin:
    @staticmethod
    def _topic_fingerprints(root: Path) -> dict[str, str]:
        if not root.exists():
            return {}
        return {
            path.relative_to(root).as_posix(): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in root.rglob("*.md")
        }

    @staticmethod
    def _stable_local_id(
        seed: str, used: set[str], *, prefix: str = ""
    ) -> str:
        counter = 0
        while True:
            value = prefix + hashlib.sha256(
                f"{seed}|{counter}".encode()
            ).hexdigest()[: 10 if prefix else BLOCK_ID_LENGTH]
            if value not in used:
                used.add(value)
                return value
            counter += 1

    def _normalize_topic_edits(self, existing_block_ids: set[str]) -> None:
        # Callers that need to report assigned IDs read these afterwards.
        self.last_block_id_map: dict[str, str] = {}
        self.last_evidence_id_map: dict[str, str] = {}
        topics = self.stage_dir / "topics"
        if not topics.exists():
            return
        paths = sorted(topics.rglob("*.md"))
        core = self.stage_dir / "core.md"
        if core.is_file():
            paths.append(core)
        texts = {path: path.read_text(encoding="utf-8") for path in paths}
        current_block_ids = {
            match.group(1)
            for text in texts.values()
            for match in re.finditer(r"(?m)\^([A-Za-z0-9-]+)\s*$", text)
        }
        block_placeholders = {
            block_id for block_id in current_block_ids
            if block_id.startswith("new-block-")
            or block_id not in existing_block_ids
        }
        used_blocks = current_block_ids - block_placeholders | existing_block_ids
        block_ids = {
            placeholder: self._stable_local_id(
                f"block|{placeholder}|" + "".join(texts.values()),
                used_blocks,
            )
            for placeholder in sorted(block_placeholders)
        }
        used_evidence = {
            match.group(1)
            for text in texts.values()
            for match in re.finditer(r"(?m)^\[\^([A-Za-z0-9_-]+)\]:", text)
            if not match.group(1).startswith("new-evidence-")
        }
        evidence_placeholders = {
            match.group(1)
            for text in texts.values()
            for match in re.finditer(
                r"(?m)^\[\^(new-evidence-[A-Za-z0-9-]+)\]:", text
            )
        }
        evidence_ids = {
            placeholder: self._stable_local_id(
                f"evidence|{placeholder}|" + "".join(texts.values()),
                used_evidence,
                prefix="e-",
            )
            for placeholder in sorted(evidence_placeholders)
        }
        self.last_block_id_map = dict(block_ids)
        self.last_evidence_id_map = dict(evidence_ids)
        for path, original in texts.items():
            text = original
            for placeholder, stable in block_ids.items():
                text = text.replace(f"^{placeholder}", f"^{stable}")
            for placeholder, stable in evidence_ids.items():
                text = text.replace(f"[^{placeholder}]", f"[^{stable}]")
            rendered = []
            topic_path = (
                Path("core.md")
                if path == core
                else Path("topics") / path.relative_to(topics)
            )
            for line in text.splitlines():
                match = definition_match(line)
                if match:
                    raw_sources = match.group("sources")
                    linked_sources = re.findall(
                        r"\[[^]]+\]\([^)]+\)", raw_sources
                    )
                    values = linked_sources or re.split(
                        r"\s*(?:,|·)\s*", raw_sources
                    )
                    sources = []
                    for value in values:
                        value = value.strip()
                        if re.fullmatch(r"\[[^]]+\]\([^)]+\)", value):
                            sources.append(value)
                        elif value:
                            sources.append(
                                self._source_link(topic_path, value)
                            )
                    line = render_definition(
                        match.group("id"),
                        None
                        if match.group("when") == "undated"
                        else match.group("when"),
                        sources,
                    )
                else:
                    line = re.sub(
                        r"[ \t]+\[\^([A-Za-z0-9_-]+)\]",
                        r"[^\1]",
                        line,
                    )
                    line = re.sub(
                        r"[ \t]*\^([A-Za-z0-9-]+)\s*$",
                        r" ^\1",
                        line,
                    )
                rendered.append(line)
            normalized = "\n".join(rendered).rstrip() + "\n"
            if normalized != original:
                path.write_text(normalized, encoding="utf-8")

    def _validate_topic_contract(self, before_units: list[Any]) -> None:
        """Reject invalid Topic links introduced by an edit."""
        before = {unit.memory_id: unit for unit in before_units}
        for unit in parse_topic_tree(self.stage_dir / "topics"):
            previous = before.get(unit.memory_id)
            if previous is None and not re.fullmatch(
                rf"[0-9a-f]{{{BLOCK_ID_LENGTH}}}", unit.memory_id
            ):
                raise ValueError(
                    "new memory blocks must use ^new-block-<label>; "
                    "Runtime assigns the stable block ID"
                )
            if previous is not None and (
                unit.content,
                unit.evidence,
            ) == (
                previous.content,
                previous.evidence,
            ):
                continue
            for target in re.findall(
                r"\[[^]\n]+\]\(([^)\n]+)\)", unit.content
            ):
                path, separator, fragment = target.partition("#")
                if (
                    not path
                    or Path(path).is_absolute()
                    or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", path)
                ):
                    continue
                relative = Path(os.path.normpath(
                    str(Path(unit.topic_path).parent / unquote(path))
                ))
                if (
                    relative.suffix.lower() == ".md"
                    and ".." not in relative.parts
                    and (
                        not separator
                        or re.fullmatch(r"\^[A-Za-z0-9-]+", fragment) is None
                    )
                ):
                    raise ValueError(
                        "Topic-to-Topic link must target #^block-id: "
                        f"{unit.topic_path} -> {target}"
                    )

    @staticmethod
    def _tree_fingerprint(root: Path) -> str:
        digest = hashlib.sha256()
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    digest.update(
                        path.relative_to(root).as_posix().encode()
                    )
                    digest.update(path.read_bytes())
        return digest.hexdigest()

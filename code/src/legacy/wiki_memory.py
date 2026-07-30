"""
Wiki-based LLM Memory System
- Retrieval-Simulated Placement (RSP): store memories where the model would look for them
- Retrieval-Feedback Reorganization (RFR): move memories when retrieval goes wrong
"""

import os
import json
import subprocess
import re
from pathlib import Path
from datetime import datetime


REPO_DIR = str(Path(__file__).resolve().parents[3])  # Research-Wiki root (git repo)
import tempfile


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """Call LLM via codex exec (uses ChatGPT subscription, not API quota)."""
    # Write prompt to temp file, feed to codex via stdin with "-" arg
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(prompt)
        prompt_file = f.name

    try:
        result = subprocess.run(
            ["codex", "exec", "-"],
            capture_output=True, text=True, timeout=120,
            cwd=REPO_DIR,
            stdin=open(prompt_file, "r")
        )
    finally:
        os.unlink(prompt_file)

    output = result.stdout.strip()
    # Parse: output ends with "tokens used\nNNNN\n<answer>"
    if "tokens used" in output:
        parts = output.split("tokens used")
        after = parts[-1].strip()
        lines = after.split("\n", 1)
        if len(lines) > 1:
            return lines[1].strip()
        before = parts[0].strip()
        if "\ncodex\n" in before:
            return before.split("\ncodex\n")[-1].strip()
    return output


class WikiMemory:
    def __init__(self, wiki_dir: str, model: str = "gpt-4o-mini"):
        self.wiki_dir = Path(wiki_dir)
        self.model = model
        self.wiki_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.wiki_dir / "index.md"
        if not self.index_path.exists():
            self.index_path.write_text("# Memory Index\n\n")
        self.stats = {"writes": 0, "retrievals": 0, "reorganizations": 0, "llm_calls": 0}

    def _call(self, prompt: str) -> str:
        self.stats["llm_calls"] += 1
        return call_llm(prompt, self.model)

    def get_index(self) -> str:
        return self.index_path.read_text()

    def get_file_headings(self, filepath: str) -> str:
        """Extract markdown headings from a file."""
        full_path = self.wiki_dir / filepath
        if not full_path.exists():
            return ""
        lines = full_path.read_text().splitlines()
        return "\n".join(l for l in lines if l.startswith("#"))

    def list_files(self, subdir: str = "") -> list[str]:
        target = self.wiki_dir / subdir
        if not target.exists():
            return []
        result = []
        for item in sorted(target.iterdir()):
            if item.name.startswith(".") or item.name == "index.md":
                continue
            rel = str(item.relative_to(self.wiki_dir))
            if item.is_dir():
                result.append(rel + "/")
            elif item.suffix == ".md":
                result.append(rel)
        return result

    # =========================================================================
    # RSP: Retrieval-Simulated Placement
    # =========================================================================

    def store(self, memory_text: str, timestamp: str = "") -> dict:
        """Store a memory using Retrieval-Simulated Placement."""
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d")

        index_content = self.get_index()
        files = self.list_files()

        # Step 1: Ask model where it would LOOK for this info
        prompt = f"""You are managing a personal memory wiki. The wiki currently has these files:

{index_content}

File list: {json.dumps(files, ensure_ascii=False)}

A new piece of information needs to be stored:
"{memory_text}"

Imagine you have FORGOTTEN this information and need to find it again later.
Looking at the wiki structure above, which file would you open first to find it?

Rules:
- If an existing file is suitable, respond with just the filename (e.g. "people/zhang-san.md")
- If no existing file fits, respond with "NEW: <filename>" (e.g. "NEW: projects/db-migration.md")
- Use concrete nouns for filenames, not abstract words
- Keep filenames short

Respond with ONLY the filename or NEW: filename, nothing else."""

        response = self._call(prompt).strip()

        # Parse response
        if response.startswith("NEW:"):
            target_file = response[4:].strip()
            if not target_file.endswith(".md"):
                target_file += ".md"
            # Create the file and parent dirs
            full_path = self.wiki_dir / target_file
            full_path.parent.mkdir(parents=True, exist_ok=True)
            title = Path(target_file).stem.replace("-", " ").replace("_", " ").title()
            full_path.write_text(f"# {title}\n\n")
        else:
            target_file = response.strip().strip('"').strip("'")
            # Clean up: remove leading ./ or /
            target_file = target_file.lstrip("./")
            if not target_file or target_file == str(self.wiki_dir) or target_file == self.wiki_dir.name:
                target_file = "general.md"
            if not target_file.endswith(".md"):
                target_file += ".md"
            candidate = self.wiki_dir / target_file
            if candidate.is_dir():
                target_file = target_file.rstrip("/") + "/general.md"
            if not (self.wiki_dir / target_file).exists():
                full_path = self.wiki_dir / target_file
                full_path.parent.mkdir(parents=True, exist_ok=True)
                title = Path(target_file).stem.replace("-", " ").replace("_", " ").title()
                full_path.write_text(f"# {title}\n\n")

        # Step 2: Determine position within file (simplified: append)
        full_path = self.wiki_dir / target_file
        content = full_path.read_text()
        entry = f"\n[{timestamp}] {memory_text}\n"
        full_path.write_text(content + entry)

        # Step 3: Update index.md
        self._update_index(target_file, memory_text)

        # Step 4: Check cross-links
        self._check_links(target_file, memory_text)

        self.stats["writes"] += 1
        return {"file": target_file, "memory": memory_text}

    def _update_index(self, target_file: str, memory_text: str):
        """Update index.md with new/updated file entry."""
        index = self.get_index()
        if target_file in index:
            return  # Already listed

        prompt = f"""Write a one-line summary (under 15 words) for a memory wiki file that contains:
"{memory_text}"

Respond with ONLY the summary, nothing else."""

        summary = self._call(prompt).strip()

        entry = f"- [{Path(target_file).stem}]({target_file}) — {summary}\n"
        self.index_path.write_text(index + entry)

    def _check_links(self, source_file: str, memory_text: str):
        """Check if this memory should link to other files."""
        index = self.get_index()
        files = self.list_files()
        other_files = [f for f in files if f != source_file and not f.endswith("/")]

        if not other_files:
            return

        prompt = f"""A memory was just stored in "{source_file}":
"{memory_text}"

Other files in the wiki:
{json.dumps(other_files, ensure_ascii=False)}

Does this memory mention or relate to any of these other files?
If yes, list the filenames (one per line). If no, respond "NONE".
Respond with ONLY filenames or NONE, nothing else."""

        response = self._call(prompt).strip()
        if response == "NONE" or not response:
            return

        # Add cross-reference links
        full_path = self.wiki_dir / source_file
        content = full_path.read_text()
        for line in response.splitlines():
            linked_file = line.strip().strip("-").strip()
            if linked_file in other_files:
                rel_path = os.path.relpath(
                    self.wiki_dir / linked_file,
                    (self.wiki_dir / source_file).parent
                )
                link = f"[→ {Path(linked_file).stem}]({rel_path})"
                if link not in content:
                    content += f"\nSee also: {link}\n"
        full_path.write_text(content)

    # =========================================================================
    # Retrieval
    # =========================================================================

    def retrieve(self, question: str) -> dict:
        """Retrieve memories relevant to a question by browsing the wiki."""
        index_content = self.get_index()
        files_visited = []
        content_found = []

        # Step 1: Read index, pick file
        prompt = f"""You are looking through your personal memory wiki to answer a question.

Question: "{question}"

Your wiki index:
{index_content}

Which file would you open first to find the answer?
Respond with ONLY the filename, nothing else."""

        chosen = self._call(prompt).strip().strip('"').strip("'").lstrip("./")
        if not chosen or chosen == str(self.wiki_dir) or chosen == self.wiki_dir.name:
            chosen = "general.md"
        if not chosen.endswith(".md"):
            chosen += ".md"

        # Step 2: Read the chosen file
        chosen_path = self.wiki_dir / chosen
        if chosen_path.is_dir():
            # Model returned a directory, list its files and pick the first .md
            sub_files = [f for f in chosen_path.iterdir() if f.suffix == ".md"]
            if sub_files:
                chosen_path = sub_files[0]
                chosen = str(chosen_path.relative_to(self.wiki_dir))
            else:
                chosen_path = self.wiki_dir / "index.md"
                chosen = "index.md"
        if chosen_path.exists():
            files_visited.append(chosen)
            file_content = chosen_path.read_text()
            content_found.append(file_content)

            # Check if we need to follow links
            links_in_file = re.findall(r'\[→[^\]]*\]\(([^)]+)\)', file_content)
            if links_in_file:
                prompt2 = f"""You opened "{chosen}" and found:
{file_content}

Question: "{question}"

Is the answer here? If yes, respond "FOUND".
If not but there are links to follow, respond with the link filename to follow.
Respond with ONLY "FOUND" or a filename."""

                response2 = self._call(prompt2).strip()
                if response2 != "FOUND" and response2:
                    # Follow the link
                    link_target = response2.strip('"').strip("'")
                    # Resolve relative path
                    link_path = (chosen_path.parent / link_target).resolve()
                    if link_path.exists():
                        files_visited.append(str(link_path.relative_to(self.wiki_dir)))
                        content_found.append(link_path.read_text())
        else:
            # Model picked a non-existent file, try all files
            for f in self.list_files():
                if not f.endswith("/"):
                    fp = self.wiki_dir / f
                    files_visited.append(f)
                    content_found.append(fp.read_text())

        self.stats["retrievals"] += 1

        # Step 3: Generate answer
        all_content = "\n---\n".join(content_found) if content_found else "(no relevant memories found)"

        prompt_answer = f"""Based on the following memories from your personal wiki:

{all_content}

Answer this question: "{question}"

If the answer is in the memories, give a concise answer. If not found, say "I don't have this information in my memory."
Respond with ONLY the answer."""

        answer = self._call(prompt_answer).strip()

        # RFR: if we visited 2+ files, trigger reorganization
        if len(files_visited) > 1:
            self._reorganize(question, files_visited, content_found)

        return {
            "answer": answer,
            "files_visited": files_visited,
            "num_hops": len(files_visited)
        }

    # =========================================================================
    # RFR: Retrieval-Feedback Reorganization
    # =========================================================================

    def _reorganize(self, question: str, files_visited: list, content_found: list):
        """When retrieval took multiple hops, consider reorganizing."""
        self.stats["reorganizations"] += 1
        # For now, just log it. Full implementation would move memories.
        print(f"  [RFR] Retrieval for '{question[:50]}...' visited {len(files_visited)} files: {files_visited}")


# =============================================================================
# Baseline: Simple embedding-free retrieval (keyword match)
# =============================================================================

class SimpleMemory:
    """Baseline: store memories as flat list, retrieve by asking LLM to search."""

    def __init__(self, storage_path: str, model: str = "gpt-4o-mini"):
        self.storage_path = Path(storage_path)
        self.model = model
        self.memories = []
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.mem_file = self.storage_path / "memories.json"
        if self.mem_file.exists():
            self.memories = json.loads(self.mem_file.read_text())
        self.stats = {"writes": 0, "retrievals": 0, "llm_calls": 0}

    def _call(self, prompt: str) -> str:
        self.stats["llm_calls"] += 1
        return call_llm(prompt, self.model)

    def store(self, memory_text: str, timestamp: str = ""):
        if not timestamp:
            timestamp = datetime.now().strftime("%Y-%m-%d")
        self.memories.append({"text": memory_text, "timestamp": timestamp})
        self.mem_file.write_text(json.dumps(self.memories, ensure_ascii=False, indent=2))
        self.stats["writes"] += 1

    def retrieve(self, question: str) -> dict:
        if not self.memories:
            return {"answer": "No memories stored.", "num_hops": 0}

        mem_text = "\n".join(
            f"[{m['timestamp']}] {m['text']}" for m in self.memories
        )

        prompt = f"""Here are all stored memories:

{mem_text}

Question: "{question}"

If the answer is in the memories, give a concise answer. If not found, say "I don't have this information in my memory."
Respond with ONLY the answer."""

        answer = self._call(prompt).strip()
        self.stats["retrievals"] += 1
        self.stats["llm_calls"] += 1

        return {"answer": answer, "num_hops": 1}

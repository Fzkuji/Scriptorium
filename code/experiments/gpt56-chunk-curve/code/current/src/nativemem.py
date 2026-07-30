"""
Memory Builder V2: NativeMem with RAM (Retrieval-Aligned Memorization).

Two-layer memory: R (raw archive) + M (abstract memory).
RAM: single-session tool calling where the model retrieves first, then writes.
Source links: each entry in M links back to R.

Usage:
    python3 src/nativemem.py --data benchmarks/locomo/data/locomo10.json --sample 0  # 从项目根目录运行
"""

import json
import argparse
import os
import sys
import time
import re
import random
from pathlib import Path
from openai import OpenAI

sys.stdout.reconfigure(line_buffering=True)

ALIYUN_KEY = (os.environ.get("BUILDER_KEY")
              or os.environ.get("ALIYUN_KEY")
              or "not-set")
ALIYUN_MODEL = os.environ.get("MODEL") or os.environ.get("BUILDER_MODEL", "deepseek-v4-flash")
ALIYUN_BASE = os.environ.get("BUILDER_BASE",
    "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1")

# trust_env=False makes httpx ignore the macOS system proxy, which otherwise
# intercepts large localhost:8199 requests and returns 502 (repeatedly hit).
# NATIVEMEM_TRUST_PROXY=1 时改走系统代理——OpenRouter/OpenAI 系模型对国内
# 直连 IP 地区封锁(403 region),必须经代理;阿里云跑保持默认直连不受影响。
import httpx as _httpx  # noqa: E402
_TRUST_PROXY = os.environ.get("NATIVEMEM_TRUST_PROXY") == "1"
_OPENAI_MAX_RETRIES = int(os.environ.get("NATIVEMEM_OPENAI_MAX_RETRIES", "2"))
_HTTP_TIMEOUT = float(os.environ.get("NATIVEMEM_HTTP_TIMEOUT", "180"))
client = OpenAI(
    api_key=ALIYUN_KEY,
    base_url=ALIYUN_BASE,
    max_retries=_OPENAI_MAX_RETRIES,
    http_client=_httpx.Client(trust_env=_TRUST_PROXY, timeout=_HTTP_TIMEOUT),
)


def _chat_create(*args, **kwargs):
    """Create a Chat Completion under the frozen runtime reasoning policy.

    Most providers default this field differently.  The experiment runner sets
    ``NATIVEMEM_REASONING_EFFORT=none`` so every call, including tool-using
    maintenance calls, carries the same explicit policy.  Ordinary runs leave
    the environment variable unset and retain the provider default.
    """
    reasoning_effort = os.environ.get("NATIVEMEM_REASONING_EFFORT")
    if reasoning_effort:
        kwargs.setdefault("reasoning_effort", reasoning_effort)
    return client.chat.completions.create(*args, **kwargs)

TOTAL_CALLS = 0
TOTAL_TOKENS = 0
CALL_LOG = []
NAV_STEPS = []

# v8.6 起整理/答题走 ThreadPoolExecutor 并发，log_usage 会被多线程同时调；
# 全局计数用锁护住，避免 TOTAL_CALLS 丢更新、CALL_LOG 竞态。
import threading as _threading  # noqa: E402
_usage_lock = _threading.Lock()

_ENTRY_RE = re.compile(r"^\[\d{4}-\d{2}-\d{2}\]")

def log_usage(response, phase="unknown"):
    global TOTAL_CALLS, TOTAL_TOKENS, CALL_LOG
    with _usage_lock:
        TOTAL_CALLS += 1
        entry = {
            "call_id": TOTAL_CALLS,
            "phase": phase,
            "model": ALIYUN_MODEL,
            "response_model": getattr(response, "model", None),
            "response_id": getattr(response, "id", None),
        }
        if response.usage:
            u = response.usage
            entry["prompt_tokens"] = u.prompt_tokens
            entry["completion_tokens"] = u.completion_tokens
            entry["total_tokens"] = u.total_tokens
            prompt_details = getattr(u, "prompt_tokens_details", None)
            completion_details = getattr(u, "completion_tokens_details", None)
            entry["cached_tokens"] = int(
                getattr(prompt_details, "cached_tokens", 0) or 0
            )
            entry["reasoning_tokens"] = int(
                getattr(completion_details, "reasoning_tokens", 0) or 0
            )
            TOTAL_TOKENS += u.total_tokens
        CALL_LOG.append(entry)


# ==================== Tools ====================

TOOLS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": "Execute a shell command in the memory folder. The working directory is always the root of the memory folder. Use standard Linux commands (ls, cat, grep, mkdir, tee, etc.) to read and write files.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "The shell command to execute"}
        }, "required": ["command"]}
    }},
]

READ_TOOLS = TOOLS  # same tool, prompt controls what commands are allowed


def execute_tool(tool_name, args, base_dir, hide_raw=False):
    """Run a bash command in base_dir.

    hide_raw=True (RETRIEVAL口径 fair mode): the raw/ folder holding the
    original transcript is INVISIBLE. Without this, `grep -rin term .` in the
    memory dir would hit raw/ and leak the original conversation into
    retrieval — the answerer would be reading the source, not the memory. We
    prune raw from any find/grep/ls recursion and refuse commands that name
    raw directly, so no-original mode is genuinely no-original.
    """
    if tool_name == "bash":
        import subprocess
        cmd = args.get("command", "")
        if hide_raw:
            # refuse direct access to raw/
            if re.search(r"(^|[\s/'\"])raw($|[\s/'\"])", cmd):
                return "(no output)"
            # make recursive grep/find/ls skip raw/ automatically
            cmd = re.sub(r"\bgrep\b(?!\S)",
                         "grep --exclude-dir=raw", cmd)
            cmd = re.sub(r"\bfind\s+\.",
                         "find . -path ./raw -prune -o", cmd)
        try:
            result = subprocess.run(
                cmd, shell=True, cwd=base_dir,
                capture_output=True, text=True, timeout=10
            )
            output = result.stdout
            if result.stderr:
                output += result.stderr
            if hide_raw:
                # drop any output line that references the raw/ folder in any
                # form: 'raw/...', 'raw:' (grep prefix), a bare 'raw' entry from
                # ls, or an 'ls -la' row ending in ' raw'. raw stays invisible.
                def _mentions_raw(ln):
                    if "raw/" in ln or "raw:" in ln:
                        return True
                    # bare 'raw', './raw', '/raw' as a path component at line end
                    return re.search(r"(^|[\s/])raw\s*$", ln) is not None
                output = "\n".join(ln for ln in output.splitlines()
                                   if not _mentions_raw(ln))
            return output.strip() if output.strip() else "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out"
        except Exception as e:
            return f"Error: {e}"
    return f"Unknown tool: {tool_name}"


# ==================== RAM Prompt ====================

SYSTEM_PROMPT = """You are a personal memory manager. You maintain a memory folder of text files for a user.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

## Retrieval-Aligned Memorization (RAM)

For each piece of information you want to store:
1. THINK about what question someone might ask about this in the future.
2. BROWSE the memory folder to find where you would look for it.
3. STORE it at the location you navigated to. If no suitable location exists, create a new file at the path you would intuitively search.

## Content Rules
- COPY the original conversation text. Do not rephrase, summarize, or generalize.
- Only store user-specific information (personal facts, preferences, experiences, decisions, plans).
- Skip generic assistant knowledge.
- Each entry must have a timestamp: [YYYY-MM-DD] followed by the original text.
- Each entry must end with a source link: [source](SOURCE_PATH)
- When new information contradicts existing memory, append the new info with its timestamp below the old entry. Do NOT overwrite or delete. Keep both versions.

## Structure Rules
- File and folder names must be concrete nouns (person names, place names, project names).
- Organize files into subdirectories by category (e.g., people/, groups/, events/).
- Each file focuses on one entity or topic.
- Use markdown headings (##, ###) to organize sections within a file.
- Each heading in a file must appear exactly once — never create a duplicate heading.
- When a file grows beyond 100 lines, split it: move each ## section into its own sub-file under a directory named after the entity, and replace the original file with an index of links.
- When an entry mentions entities that already have their own files, add a cross-reference link: See also: [EntityName](path/to/entity.md)
- If a topic appears as a heading in 3 or more different files, extract it into its own dedicated file and replace the scattered entries with cross-reference links.

## Process
For each piece of information to store:

1. **See the folder structure**: `ls -lhR` — shows all folders, subfolders, files, and file sizes. This tells you what entities and topics already exist and how much content each file has.
2. **See a file's heading structure**: `grep '^#' path/to/file.md` — shows all heading levels (##, ###) in the file. This tells you what topics are already covered.
3. **Read the target section**: check the file size from `ls -lhR`. If the file is small (under a few KB), `cat` it directly. If the file is large, first `grep '^#'` to see headings, then use `sed -n '/^## Topic$/,/^## /p' path/to/file.md` to read only the section you need.
4. **Write to the correct section**: use `sed` to insert content at the right position in the file, or rewrite the file if necessary. If a heading already covers this topic, add the new entry under that existing heading. If no matching heading exists, append a new section. Never create a duplicate heading.
5. After writing, add cross-reference links in related entity files that already exist.

You can use any standard shell commands: ls, cat, grep, find, wc, mkdir -p, sed, head, tail, tee, etc.
"""


# ==================== O1: Two-Layer Memory variant (distilled + verbatim) ====
# Diff vs baseline SYSTEM_PROMPT: the "Content Rules" section is replaced with a
# two-layer entry format — a distilled fact line (with absolute dates + all
# specific nouns preserved) followed by the verbatim original underneath.
# Borrows: Mem0 (Observation-Date time anchor + Never-Generalize), Memobase
# (precision-adaptive dates), LightMem (source回指). Selected via
# NATIVEMEM_PROMPT=v4. Everything else (RAM, structure, process) is identical.

_DUALLAYER_CONTENT_RULES = """## Content Rules — Two-Layer Entries

Store each piece of user information as a TWO-LAYER entry:

    [YYYY-MM-DD] <distilled fact — one clear sentence>
      > "<verbatim original conversation text>"
      [source](SOURCE_PATH)

**The distilled fact line** (this is what future searches will match — make it findable):
- Write ONE self-contained sentence. Replace all pronouns with the specific person's name (never "she"/"they" — write "Caroline"/"Melanie").
- PRESERVE every specific detail — never generalize. Keep proper nouns (book/movie/place/brand names), exact numbers, and precise qualifiers EXACTLY. Examples: keep "assistant manager" NOT "manager"; "aerial yoga" NOT "yoga"; "a cup with a dog face" NOT "a cup"; "the book 'Becoming Nicole'" NOT "a book". Proper nouns are the HIGHEST value — users search by name, and a fact without the name is unfindable.
- Convert relative time to ABSOLUTE dates using the conversation date as the anchor (see below). Write "in June 2023" not "next month"; "the week before 15 July 2023" not "last Friday".
- Keep necessary context in the same sentence (fact + why/what), 10-40 words. Do NOT split into fragments.

**The verbatim line** (`> "..."`): copy the original conversation text unchanged, so any detail the distilled line missed can still be recovered and checked.

**Time anchoring (CRITICAL):** The chunk's date [YYYY-MM-DD] is the conversation date — your ONLY anchor for resolving relative time. "yesterday" → day before it; "last week" → week before it; "next month" → month after it. Only resolve to the precision you actually know: if only the year is inferable, write just the year; if you know "the week before" but not the exact day, write "the week before <date>". Never invent a precision you don't have. Never turn an absolute reference vague.

**Other rules:**
- Only store user-specific information (personal facts, preferences, experiences, decisions, plans). Skip generic assistant knowledge and pure small-talk.
- Process the conversation top to bottom; do not skip a fact because it looks minor. A chunk usually yields several distinct facts — if you extracted only one from a multi-topic chunk, re-read it.
- When new information contradicts existing memory, append the new two-layer entry with its date below the old one. Do NOT overwrite or delete. Keep both versions.
"""


def _build_system_prompt():
    """Return the active build prompt. Baseline unless NATIVEMEM_PROMPT=v4."""
    if os.environ.get("NATIVEMEM_PROMPT") in ("v4", "v5"):
        # Swap the baseline Content-Rules block for the two-layer one.
        import re as _re
        return _re.sub(
            r"## Content Rules.*?(?=## Structure Rules)",
            _DUALLAYER_CONTENT_RULES + "\n",
            SYSTEM_PROMPT, flags=_re.DOTALL)
    return SYSTEM_PROMPT


# ==================== V5: decoupled distill → store ====================
# Root cause of Qwen's collapse (agent analysis 2026-07-04): the baseline
# does distill + navigate + date-math + write + cross-ref + split all in ONE
# bash tool session. Weak models overload and drop the date math and detail
# preservation. Mem0/LightMem stay robust by making distillation a single
# stateless text→JSON step with a structured date anchor, leaving storage to
# code. V5 mirrors that: Step A distills with NO tools; Step B stores.

_DISTILL_PROMPT = """You extract memorable user facts from one conversation snippet. Output JSON only — no tools, no file access, just read and extract.

## Observation Date (the day this conversation happened — your ONLY anchor for relative time)
{obs_date}

Do NOT use today's real-world date to resolve time. Only the Observation Date above.

## What to extract
Extract every distinct piece of user-specific information. Read top to bottom, don't stop at the first — a snippet usually holds several facts. Extract BOTH kinds and never skip the first kind:
1. **EVENTS (highest priority — these answer "when did X happen")**: something the person DID, went to, made, attended, or will do, together with WHEN. "I went to a support group yesterday" → an event on the day before the Observation Date. Always capture the action AND its time. Do not collapse an event into a vague preference — "went to a support group yesterday" must NOT become "cares about support groups".
2. **STATES**: preferences, opinions, ongoing situations, plans, relationships.
Skip generic assistant knowledge and pure small-talk.

## For each fact, produce TWO things:
- "distilled": ONE self-contained sentence. Rules:
  - Replace pronouns with the specific person's name (write "Caroline"/"Melanie", never "she"/"they").
  - PRESERVE every specific detail exactly — proper nouns (book/movie/place/brand names), exact numbers, precise qualifiers. Keep "assistant manager" NOT "manager"; "a cup with a dog face" NOT "a cup"; "the book 'Becoming Nicole'" NOT "a book". Proper nouns are highest value — a fact without the name is unfindable.
  - For relative time, you do NOT need to compute an exact date. Write the relative reference together with the anchor, e.g. "the week before {obs_date}", "in the month after {obs_date}", or the plain year if that's all that's known. Do not guess a precision you don't have.
- "verbatim": the original conversation text for this fact, copied unchanged.

## Output format (JSON only, nothing else):
{{"facts": [{{"person": "Caroline", "topic": "short 2-4 word topic like 'music' or 'adoption plans'", "distilled": "...", "verbatim": "..."}}, ...]}}

If nothing worth storing, output {{"facts": []}}.
"""


# ==================== V6: layered distillation (design §2.4) ====================
# One memory = summarized skeleton (searchable) + verbatim detail tokens
# (proper nouns / numbers never generalized). The model also lists the key
# entities it must preserve; code then verifies none were dropped and flags
# misses for backfill — turning the "keep details" rule from soft (prompt,
# which weak models violate) into a hard guarantee. Selected via
# NATIVEMEM_PROMPT=v6.

_LAYERED_DISTILL_PROMPT = """You turn one conversation snippet into memory notes. Output JSON only — no tools, just read and extract.

## Observation Date (the day this conversation happened — your ONLY anchor for relative time)
{obs_date}
Do NOT use today's real date. Resolve "yesterday"/"last week" against the Observation Date only.

## What to extract
Read top to bottom, don't stop at the first — a snippet usually holds several facts. Capture BOTH, never skip events:
1. **EVENTS (highest priority — answer "when did X happen")**: something the person DID / went to / made / attended / will do, WITH its time. "went to a support group yesterday" is an event on the day before {obs_date}. Never collapse an event into a vague preference.
2. **STATES**: preferences, opinions, plans, relationships, ongoing situations.
Skip generic assistant knowledge and pure small-talk.

## For each fact, produce THREE fields — this is LAYERED distillation:
- "skeleton": ONE clear self-contained sentence summarizing the fact (this is what gets searched, so make it clean and findable):
   - Make the sentence self-contained: name the person at least once (the subject). You MAY keep natural pronouns after that — write "Caroline went to a group and it helped her", NOT "Caroline went to a group and it helped Caroline". Do not repeat the name in every clause.
   - Summarize the ACTION / relation / time freely into clear wording — you MAY rephrase this part.
   - Anchor relative time to the Observation Date ("the week before {obs_date}", or the plain year if that's all you know). Do not invent precision.
- "keep_verbatim": a LIST of the specific detail tokens that must survive EXACTLY, copied from the original word-for-word — proper nouns (book/movie/place/brand/person names), exact numbers, and precise qualifiers ("dog face", "assistant manager"). These are the highest-value, unfindable-if-lost tokens. The skeleton MUST contain every token in this list, spelled exactly. If a fact has no such token, use an empty list.
- "verbatim": the original conversation sentence(s) for this fact, copied unchanged.

Rule: the skeleton is a summary EXCEPT for the keep_verbatim tokens, which appear in it untouched. Summarize the connective tissue, preserve the nouns/numbers.

## WORKED EXAMPLE (Observation Date = 2023-05-08)
Input:
  Melanie: I moved here from Stockholm about 4 years ago. Last weekend I finally read "Becoming Nicole" by Amy Ellis Nutt — it really moved me.
  Caroline: That's great! How are the kids?
  Melanie: Good! My youngest just turned 3. We're planning a camping trip next month.

Correct output:
{{"facts": [
  {{"person": "Melanie", "topic": "background", "skeleton": "Melanie moved to her current city from Stockholm about 4 years before 2023-05-08.", "keep_verbatim": ["Stockholm", "4 years"], "verbatim": "I moved here from Stockholm about 4 years ago."}},
  {{"person": "Melanie", "topic": "reading", "skeleton": "Melanie read the book Becoming Nicole by Amy Ellis Nutt the weekend before 2023-05-08 and found it moving.", "keep_verbatim": ["Becoming Nicole", "Amy Ellis Nutt"], "verbatim": "Last weekend I finally read \\"Becoming Nicole\\" by Amy Ellis Nutt — it really moved me."}},
  {{"person": "Melanie", "topic": "family", "skeleton": "Melanie's youngest child just turned 3 as of 2023-05-08.", "keep_verbatim": ["3"], "verbatim": "My youngest just turned 3."}},
  {{"person": "Melanie", "topic": "plans", "skeleton": "Melanie is planning a camping trip in the month after 2023-05-08.", "keep_verbatim": [], "verbatim": "We're planning a camping trip next month."}}
]}}

Note in the example: skeletons summarize freely but keep Stockholm / Becoming Nicole / Amy Ellis Nutt / 3 exactly; the person is named once then pronouns are fine ("found it moving", not "Melanie found Melanie moving"); relative times are anchored ("the weekend before 2023-05-08", "the month after 2023-05-08"); keep_verbatim holds only proper nouns/numbers, empty when there are none (the camping fact). "How are the kids?" is Caroline's filler and produces no fact.

## Output (JSON only, same shape as the example):
{{"facts": [...]}}

If nothing worth storing: {{"facts": []}}
"""


def _extract_proper_nouns(text):
    """Heuristic: capitalized multi-word names, quoted titles, and numbers.
    Used to check whether the skeleton dropped a detail the original had."""
    nouns = set()
    # Quoted titles: "Becoming Nicole", 'The Name of the Wind'
    nouns.update(re.findall(r"[\"']([A-Z][^\"']{2,40})[\"']", text))
    # Capitalized sequences (proper-noun phrases), excluding sentence starts is
    # hard without NLP; we accept some noise — the model confirms, code only flags.
    nouns.update(re.findall(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b", text))
    # Standalone numbers with optional unit (years, counts, pages)
    nouns.update(re.findall(r"\b(\d{2,4})\b", text))
    return {n.strip() for n in nouns if len(n.strip()) > 1}


def _is_backfillable(token):
    """Only backfill tokens that look like a proper noun or number — a
    capitalized word/name, a quoted title, or a digit. Plain phrases the model
    over-eagerly put in keep_verbatim ('accept and support me', 'my rocks')
    are NOT backfilled — they add noise that hurts strong models."""
    t = token.strip()
    if re.search(r"\d", t):                      # has a number
        return True
    if re.match(r"^[A-Z][a-zA-Z]", t) and len(t.split()) <= 4:  # Proper-noun-ish
        return True
    return False


def verify_details(fact):
    """Hard check (design §2.1/§2.4): flag proper-noun/number detail tokens that
    the skeleton dropped, for backfill. Only genuine proper nouns/numbers —
    not the plain phrases weak models over-list in keep_verbatim."""
    skel = fact.get("skeleton", "")
    missing = [t for t in fact.get("keep_verbatim", [])
               if t and _is_backfillable(t) and t.lower() not in skel.lower()]
    # cross-check verbatim's proper nouns against skeleton
    for n in _extract_proper_nouns(fact.get("verbatim", "")):
        if n.lower() not in skel.lower() and n not in missing:
            missing.append(n)
    return missing


def normalize_date(raw):
    """Normalize benchmark timestamps to ``YYYY-MM-DD``.

    Supports LoCoMo's ``8 May, 2023`` form and LongMemEval/BEAM's leading
    ``YYYY/MM/DD`` or ``YYYY-MM-DD`` form.  Falls back to the original value
    when no calendar date can be parsed.
    """
    if not raw:
        return raw
    raw = str(raw).strip()
    iso = re.search(r"(?<!\d)(\d{4})[-/](\d{1,2})[-/](\d{1,2})(?!\d)", raw)
    if iso:
        year, month, day = map(int, iso.groups())
        try:
            from datetime import date
            return date(year, month, day).isoformat()
        except ValueError:
            return raw
    m = re.search(r"(\d{1,2})\s+([A-Za-z]+),?\s+(\d{4})", raw)
    if not m:
        return raw
    day, month_name, year = m.group(1), m.group(2), m.group(3)
    months = {mn: i for i, mn in enumerate(
        ["January", "February", "March", "April", "May", "June", "July",
         "August", "September", "October", "November", "December"], 1)}
    mo = months.get(month_name.capitalize())
    if not mo:
        return raw
    return f"{year}-{mo:02d}-{int(day):02d}"


def distill_chunk(chunk_text, chunk_date, max_retry=6):
    """Step A: stateless text→JSON extraction, NO tools. Returns list of facts.

    v6 (NATIVEMEM_PROMPT=v6): layered distillation — skeleton + keep_verbatim
    + verbatim, with a code-side detail check that backfills dropped proper
    nouns/numbers. Other modes: original distilled+verbatim.
    """
    layered = os.environ.get("NATIVEMEM_PROMPT") == "v6"
    obs_date = normalize_date(chunk_date)
    prompt = _LAYERED_DISTILL_PROMPT if layered else _DISTILL_PROMPT
    messages = [
        {"role": "system", "content": prompt.format(obs_date=obs_date)},
        {"role": "user", "content": chunk_text},
    ]
    # Don't hard-depend on response_format=json_object — the local ChatGPT
    # proxy silently ignores it and returns non-JSON, causing distill to fail
    # empty. Ask for JSON in the prompt and extract robustly below.
    use_json_mode = os.environ.get("NATIVEMEM_JSON_MODE") == "1"
    kwargs = {"response_format": {"type": "json_object"}} if use_json_mode else {}
    for retry in range(max_retry):
        try:
            resp = _chat_create(
                model=ALIYUN_MODEL, messages=messages,
                max_tokens=4000, temperature=0.2, **kwargs)
            break
        except Exception:  # noqa: BLE001
            if retry < max_retry - 1:
                time.sleep(3 * (retry + 1))
            else:
                return []
    log_usage(resp, phase="distill")
    text = resp.choices[0].message.content or ""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
    facts = None
    try:
        obj = json.loads(text)
        facts = obj.get("facts", []) if isinstance(obj, dict) else obj
    except Exception:  # noqa: BLE001
        # proxy may wrap JSON in prose — extract the first {...} that parses
        m = re.search(r"\{.*\"facts\".*\}", text, re.DOTALL)
        if m:
            try:
                facts = json.loads(m.group(0)).get("facts", [])
            except Exception:  # noqa: BLE001
                facts = None
    if not isinstance(facts, list):
        return []

    if not layered:
        return [f for f in facts if isinstance(f, dict) and f.get("distilled")]

    # v6: normalize skeleton→distilled, run the detail check, backfill misses.
    out = []
    for f in facts:
        if not isinstance(f, dict) or not f.get("skeleton"):
            continue
        missing = verify_details(f)
        skel = f["skeleton"]
        if missing:
            # Hard guarantee: append the dropped detail tokens so they are
            # searchable/answerable even if the model summarized them away.
            skel = skel.rstrip(". ") + " (" + ", ".join(missing) + ")."
            f["_backfilled"] = missing
        f["distilled"] = skel  # unify field name for downstream store_facts
        out.append(f)
    return out


def measure_library(memory_dir):
    """测量库规模：files/dirs/bytes/entries。跳过 raw/ 和隐藏。entries = 全库
    [YYYY-MM-DD] 开头的行数。是三节奏增长触发的基线。"""
    files = dirs = total_bytes = entries = 0
    for root, dirnames, filenames in os.walk(memory_dir):
        dirnames[:] = [d for d in dirnames if d != "raw" and not d.startswith(".")]
        for d in dirnames:
            dirs += 1
        for fn in filenames:
            if fn.startswith("."):
                continue
            files += 1
            path = os.path.join(root, fn)
            try:
                total_bytes += os.path.getsize(path)
                with open(path, encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if _ENTRY_RE.match(line):
                            entries += 1
            except OSError:
                pass
    return {"files": files, "dirs": dirs, "bytes": total_bytes, "entries": entries}


def library_grew_past_threshold(old, new, growth_ratio=2.0, file_delta=8):
    """自上次全局重构以来是否增长过阈值：文件+文件夹数增量 >= file_delta，
    或 字节比值 >= growth_ratio，任一为真。"""
    struct_delta = (new["files"] + new["dirs"]) - (old["files"] + old["dirs"])
    if struct_delta >= file_delta:
        return True
    if old["bytes"] > 0 and new["bytes"] / old["bytes"] >= growth_ratio:
        return True
    return False


def _top_level_view(memory_dir):
    """只列记忆根目录的直接子项（模型 drill-in 的起点，不展开文件内部）。
    文件列文件名；目录列 'name/  (N items)'；跳过隐藏文件和 raw/。"""
    try:
        names = sorted(os.listdir(memory_dir))
    except (FileNotFoundError, NotADirectoryError):
        return "(empty memory)"
    lines = []
    for name in names:
        if name.startswith(".") or name == "raw":
            continue
        full = os.path.join(memory_dir, name)
        if os.path.isdir(full):
            n = len([x for x in os.listdir(full) if not x.startswith(".")])
            lines.append(f"{name}/  ({n} items)")
        else:
            lines.append(name)
    return "\n".join(lines) if lines else "(empty memory)"


def _count_entries_in_file(path):
    """Count [YYYY-MM-DD] entries in a single file."""
    n = 0
    try:
        with open(path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if _ENTRY_RE.match(line):
                    n += 1
    except OSError:
        pass
    return n


def inspect_structure(memory_dir, big=150, small=8, wide=20):
    """结构体检：纯测量，把失衡维度用阈值标 ⚠。返回 {report, warnings}。
    维度：单文件过大(>big)/过小(<small)、目录过宽(>wide)、近义重名文件。
    只测量不动手——修由 tidy_local/reorganize_library 交给模型。"""
    warnings, lines = [], []
    all_md = []  # (relpath, entries)
    for root, dirnames, filenames in os.walk(memory_dir):
        dirnames[:] = [d for d in dirnames if d != "raw" and not d.startswith(".")]
        # 目录过宽
        rel_root = os.path.relpath(root, memory_dir)
        visible = [x for x in filenames if not x.startswith(".")] + dirnames
        if len(visible) > wide:
            tgt = rel_root if rel_root != "." else "(root)"
            warnings.append({"kind": "wide", "target": tgt + "/",
                             "detail": f"{len(visible)} 个直接子项 (>{wide})"})
            lines.append(f"- {tgt}/: {len(visible)} 个直接子项     ⚠ 目录过宽 (>{wide})")
        for fn in filenames:
            if fn.startswith(".") or not fn.endswith(".md"):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, memory_dir)
            n = _count_entries_in_file(path)
            all_md.append((rel, fn, n))
            if n > big:
                warnings.append({"kind": "big", "target": rel, "detail": f"{n} 条 (>{big})"})
                lines.append(f"- {rel}: {n} 条     ⚠ 过大 (>{big})")
            elif n < small:
                warnings.append({"kind": "small", "target": rel, "detail": f"{n} 条 (<{small})"})
                lines.append(f"- {rel}: {n} 条     ⚠ 过小 (<{small})")
    # 近义重名文件（复用 _find_merge_candidates，把文件名当 heading 喂进去）
    # 重要: 剥离 .md 后缀再喂进相似度计算，否则 .md 本身会污染 Jaccard 相似度
    # (所有文件都有 .md，导致 Career.md vs Diet.md 的相似度被拉高)。
    # 但保留原始带 .md 的名字用于 warnings 展示。
    basenames_clean = []
    basename_to_original = {}
    for _, fn, _ in all_md:
        clean = fn[:-3] if fn.endswith(".md") else fn
        basenames_clean.append(clean)
        basename_to_original[clean] = fn
    for group in _find_merge_candidates(basenames_clean):
        if len(group) > 1:
            # 映射回原始带 .md 的名字用于 warnings
            original_group = [basename_to_original.get(name, name) for name in group]
            warnings.append({"kind": "dup", "target": ", ".join(original_group),
                             "detail": "疑似近义重名"})
            lines.append(f"- 近义文件名: [{', '.join(original_group)}]     ⚠ 疑重复")
    total = measure_library(memory_dir)
    summary = (f"全库: {total['files']} 文件 / {total['dirs']} 目录 / "
               f"{total['entries']} 条")
    report = "memory/ 结构报告:\n" + ("\n".join(lines) if lines else "(无失衡)") + "\n" + summary
    return {"report": report, "warnings": warnings}


def split_into_chunks(session, size=10):
    """把一个 session 的 turn 列表按 size 切 chunk。
    返回 [(chunk_text, dia_ids), ...]：chunk_text 是 'speaker: text' 拼接，
    dia_ids 是该 chunk 覆盖的 turn['dia_id'] 列表。跳过非 dict / 空 text 的 turn。"""
    turns = [t for t in session
             if isinstance(t, dict) and str(t.get("text", "")).strip()]
    chunks = []
    for cs in range(0, len(turns), size):
        group = turns[cs:cs + size]
        text = "".join(f"{t.get('speaker', 'user')}: {t.get('text', '')}\n\n"
                       for t in group)
        dia_ids = [t["dia_id"] for t in group if "dia_id" in t]
        chunks.append((text, dia_ids))
    return chunks


def split_into_chunks_structured(session, size=10):
    """Structured variant of split_into_chunks: same turn selection/grouping,
    but returns [(turns_list, dia_ids), ...] where turns_list is the list of
    [(speaker, text), ...] tuples — NOT joined into free text.

    v8 needs per-turn boundaries to number blocks 1:1 with dia_ids. Reparsing
    the joined text (split_into_chunks) is ambiguous: a blank line INSIDE one
    turn's text is indistinguishable from the "\\n\\n" turn separator, so a
    turn with an internal blank line splits into 2+ blocks and shifts every
    subsequent block's number vs dia_ids (wrong dia_id / dropped events). This
    keeps the boundary explicit so numbering is 1:1 by construction.

    IMPORTANT: only ADD this — split_into_chunks stays byte-identical (v4/v6/v7
    depend on it)."""
    turns = [t for t in session
             if isinstance(t, dict) and str(t.get("text", "")).strip()]
    chunks = []
    for cs in range(0, len(turns), size):
        group = turns[cs:cs + size]
        turns_list = [(t.get("speaker", "user"), t.get("text", "")) for t in group]
        dia_ids = [t["dia_id"] for t in group if "dia_id" in t]
        chunks.append((turns_list, dia_ids))
    return chunks


# ==================== Raw Archive ====================

def save_to_raw_archive(chunk_text, chunk_date, session_idx, chunk_idx, raw_dir):
    os.makedirs(raw_dir, exist_ok=True)
    filename = f"session_{session_idx}_chunk_{chunk_idx}.md"
    filepath = os.path.join(raw_dir, filename)
    with open(filepath, "w") as f:
        f.write(f"# Session {session_idx}, Chunk {chunk_idx}\nDate: {chunk_date}\n\n{chunk_text}")
    return filename


# ==================== V5 Step B: store pre-distilled facts ====================

_STORE_PROMPT = """You file already-extracted memory entries into a personal memory folder. The distillation is DONE — do NOT re-summarize or change any wording. Your only job is to put each entry in the right file, using RAM (store it where you'd later look for it).

You have a bash tool. Working directory is the memory folder root.

## For the entries below
Each entry is: person / topic / a distilled fact line / its verbatim quote.
Store each as this two-layer block, under a file named after the person (e.g. people/Caroline.md), grouped by a `## topic` heading. Format (copy the distilled and verbatim text exactly, do NOT add angle brackets):

    [{obs_date}] Caroline started playing acoustic guitar in 2018.
      > "I started playing acoustic guitar about five years ago."
      [source]({source_path})

## Steps
1. `ls` to see which person files already exist.
2. For each entry, append its block to people/<Person>.md under the matching `## topic` heading (create the file or heading if absent). Use `cat >>`, `tee -a`, or `sed`. Never alter the distilled/verbatim text.
3. Do not split files, do not add cross-references — just file the entries. Structure cleanup happens later.

## Entries to store
{entries}
"""


def store_facts_code(facts, chunk_date, memory_dir, source_path):
    """Step B (v6, deterministic): CODE files each fact into
    people/<Person>.md under a `## <topic>` heading — no model, no bash. The
    model already produced structured fields (person/topic/distilled/verbatim);
    filing them is a mechanical write, so keep it deterministic. This stops
    weak models from building runaway deep folder trees (Qwen made 27 files /
    20 headings) and keeps a flat, navigable one-file-per-person structure that
    consolidate_topics can then tidy.
    """
    if not facts:
        return 0, []
    obs = normalize_date(chunk_date)
    people_dir = os.path.join(memory_dir, "people")
    os.makedirs(people_dir, exist_ok=True)
    written = 0
    for f in facts:
        person = re.sub(r"[^\w\- ]", "", str(f.get("person", "User"))).strip() or "User"
        topic = str(f.get("topic", "misc")).strip().lower()
        distilled = f.get("distilled", "")
        verbatim = f.get("verbatim", "")
        if not distilled:
            continue
        path = os.path.join(people_dir, f"{person}.md")
        block = (f"[{obs}] {distilled}\n"
                 f'  > "{verbatim}"\n'
                 f"  [source]({source_path})\n")
        heading = f"## {topic}"
        # read existing, insert under matching heading or append new section
        if os.path.exists(path):
            content = open(path).read()
        else:
            content = f"# {person}\n\n"
        if heading in content:
            # insert block right after the heading line
            lines = content.splitlines(keepends=True)
            out = []
            for ln in lines:
                out.append(ln)
                if ln.rstrip("\n") == heading:
                    out.append(block)
            content = "".join(out)
        else:
            content = content.rstrip("\n") + f"\n\n{heading}\n{block}"
        open(path, "w").write(content)
        written += 1
    return written, []


def store_facts(facts, chunk_date, memory_dir, source_path, max_rounds=12):
    """Step B: file pre-distilled facts. v6 uses deterministic code write;
    other modes let the model file via bash (legacy)."""
    if os.environ.get("NATIVEMEM_PROMPT") == "v6":
        return store_facts_code(facts, chunk_date, memory_dir, source_path)
    if not facts:
        return 0, []
    entries = "\n".join(
        f"- person: {f.get('person','User')} | topic: {f.get('topic','misc')}\n"
        f"  distilled: {f['distilled']}\n  verbatim: {f.get('verbatim','')}"
        for f in facts)
    messages = [
        {"role": "system", "content": _STORE_PROMPT.format(
            obs_date=normalize_date(chunk_date), source_path=source_path,
            entries=entries)},
        {"role": "user", "content": "File all the entries now."},
    ]
    tool_trace = []
    for _ in range(max_rounds):
        for retry in range(6):
            try:
                response = _chat_create(
                    model=ALIYUN_MODEL, messages=messages,
                    tools=TOOLS, max_tokens=2000, temperature=0.2)
                break
            except Exception:  # noqa: BLE001
                if retry < 5:
                    time.sleep(3 * (retry + 1))
                else:
                    raise
        log_usage(response, phase="store")
        msg = response.choices[0].message
        if msg.content:
            msg.content = re.sub(r"<think>.*?</think>", "", msg.content,
                                 flags=re.DOTALL).strip()
        if not msg.tool_calls:
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            tool_trace.append({"tool": tc.function.name, "args": fn_args})
            result = execute_tool(tc.function.name, fn_args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
    return len(messages), tool_trace


# ==================== Core: Single-session RAM ====================

STRUCTURE_STANDARD = """\
好结构的标准（尽量遵守，做不到没关系，整理环节会兜底）：
- 文件别太大：单文件条目过多会让检索捞不准。
- 文件别太碎：一堆只有几条的小文件会让检索跳很多次。
- 目录别太宽：一个文件夹平铺几十个文件难定位。
- 同主题别分散：一个话题散在多个文件会漏检。
- 命名统一：同一实体/主题别造近义重名（LGBTQ_Support_Group vs lgbtq-support-group）。
- 默认浅、按需深：小规模一主体一文件即可；大到超标才拆子目录、留摘要页+链接。
"""

_AGENT_STORE_PROMPT = """\
你在维护一个用文件系统组织的记忆库。工作目录就是记忆库根目录。
你只能用 bash（ls / cat / grep / sed / mkdir / tee / cat >> 等）读写文件。

## 当前记忆库顶层结构
{top_view}

## 上文（前几段对话的滚动总结，帮你判断新事实该并到哪）
{running_summary}

## 这批要存的事实（已提炼好，每条含 skeleton 骨架句 / verbatim 原文 / 相关 dia_id）
{facts}

## 本批对话覆盖的 turn id（写 source 时从中选与该条事实相关的）
{dia_ids}

## 每条记忆固定写成这三行块
[YYYY-MM-DD] 骨架句
  > "原文引用"
  [source](dia_id)          # 来自多句就 [source](D1:3, D1:5)

## 即时维护三规则（局部就能判断，务必遵守）
1. 放最相关的已有文件：先 ls/cat 看顶层和相关文件，把新事实放进语义最近的已有文件/## 小标题下；确实没有对应主体才新建文件。别新开文件堆一起。
2. 见重复就地合并/更新：同一事实的新版本（时间更新/细节补充/状态改变）就地改已有条目或紧挨着补，别无脑追加造重复。每条事实只写一次；写完这批就停，别反复写同一条。
3. 命名沿用惯例：新文件/小标题命名看已有的怎么起就怎么来，别造近义重名。

{structure_standard}

存完后，用一句话总结这批对话讲了什么，放在 <summary></summary> 里（给下一批当上下文）。
"""

def process_chunk(chunk_text, chunk_date, memory_dir, source_path, max_rounds=15):
    """Process one chunk. v5/v6 = decoupled distill(A)→store(B); else single."""
    if os.environ.get("NATIVEMEM_PROMPT") in ("v5", "v6"):
        facts = distill_chunk(chunk_text, chunk_date)          # Step A: no tools
        return store_facts(facts, chunk_date, memory_dir, source_path)  # Step B
    messages = [
        {"role": "system", "content": _build_system_prompt()},
        {"role": "user", "content": f"New conversation ({chunk_date}):\n\n{chunk_text}\n\nSource path for entries: {source_path}\n\nPlease extract and store important user information. For each item, first browse the folder to find the right location (RAM), then store it there with timestamp and source link."}
    ]

    tool_trace = []
    for round_i in range(max_rounds):
        for retry in range(6):
            try:
                response = _chat_create(
                    model=ALIYUN_MODEL, messages=messages,
                    tools=TOOLS, max_tokens=4000, temperature=0.3,
                )
                break
            except Exception as e:
                # Proxy backend occasionally returns 502/timeout on long
                # builds; exponential backoff so one blip doesn't kill the run.
                if retry < 5:
                    time.sleep(3 * (retry + 1))
                else:
                    raise

        log_usage(response, phase="build")
        msg = response.choices[0].message

        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                fn_args = {}
            tool_trace.append({"tool": fn_name, "args": fn_args, "round": round_i})
            result = execute_tool(fn_name, fn_args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return len(messages), tool_trace


# ==================== v7: agent 自组织写入 ====================

_SUMMARY_RE = re.compile(r"<summary>(.*?)</summary>", re.S)


def _format_facts_for_agent(facts):
    lines = []
    for i, f in enumerate(facts, 1):
        skel = f.get("distilled") or f.get("skeleton", "")
        lines.append(f"{i}. skeleton: {skel}\n   verbatim: {f.get('verbatim', '')}")
    return "\n".join(lines)


_WRITE_CMD_RE = re.compile(r">>|\btee\b|\bsed\s+-i\b|>\s*[^&|;]+\.md")


def _is_write_command(cmd):
    """粗判一条 bash 命令是否往 .md 文件追加/写入内容（用于终止压力，不做内容改写）。"""
    return bool(_WRITE_CMD_RE.search(cmd or ""))


_STOP_AFTER_WRITE_REMINDER = (
    "你已经把这批事实写进文件了。现在检查是否有重复写入——如果同一条 "
    "[source](dia_id) 在文件里出现多次，用 sed 删掉多余的，只留一条。"
    "确认无重复后，不要再写任何新内容，直接输出 <summary>一句话总结</summary> 结束。"
)


def _run_store_agent(memory_dir, top_view, facts_text, dia_ids,
                      running_summary="", max_rounds=4):
    """跑一轮 bash-agent 存储会话，返回最终文本（同 process_chunk 的 tool-call 处理方式）。

    弱模型写完一批事实后经常不知道停，会把同一批 fact 反复 `cat >> file.md`
    好几轮。这里只加终止压力（HOW），不做内容侧去重改写（WHAT 仍由模型自己
    组织）：第一次检测到写入命令后，下一轮前注入一条提醒，让模型自查重复、
    删多余、然后停手；同时把 max_rounds 从 15 降到 6 收窄误行为的爆炸半径。
    """
    prompt = _AGENT_STORE_PROMPT.format(
        top_view=top_view, facts=facts_text,
        dia_ids=", ".join(dia_ids) if dia_ids else "(none)",
        running_summary=running_summary.strip() if running_summary else "（无）",
        structure_standard=STRUCTURE_STANDARD)
    messages = [{"role": "user", "content": prompt}]
    final_text = ""
    has_written = False
    reminder_injected = False
    for _round_i in range(max_rounds):
        if has_written and not reminder_injected:
            messages.append({"role": "user", "content": _STOP_AFTER_WRITE_REMINDER})
            reminder_injected = True
        for retry in range(6):
            try:
                resp = _chat_create(
                    model=ALIYUN_MODEL, messages=messages, tools=TOOLS,
                    max_tokens=2000, temperature=0.2)
                break
            except Exception:  # noqa: BLE001
                if retry < 5:
                    time.sleep(3 * (retry + 1))
                else:
                    raise
        log_usage(resp, phase="v7_store")
        msg = resp.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()
        final_text = msg.content or final_text
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            if _is_write_command(args.get("command", "")):
                has_written = True
            # 写入期不 hide_raw：记忆库本就没有 raw/ 原文
            out = execute_tool(tc.function.name, args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return final_text


def process_chunk_agent(chunk_text, chunk_date, memory_dir, dia_ids,
                         running_summary="", mode="oneshot"):
    """v7 写入主体：提炼 → agent 用 bash 自组织存 → 返回本 chunk 一句总结。
    mode: 'oneshot'（一次会话存+总结，拿不到 summary 就返回 ""）/
          'threecall'（存、总结分调；拿不到 summary 时补一次总结调用）。
    提炼两模式都由 distill_chunk 独立完成（spec §2.4：提炼这一步独立）。"""
    os.makedirs(memory_dir, exist_ok=True)
    facts = distill_chunk(chunk_text, chunk_date)
    if not facts:
        return ""
    top_view = _top_level_view(memory_dir)
    facts_text = _format_facts_for_agent(facts)
    final_text = _run_store_agent(memory_dir, top_view, facts_text, dia_ids,
                                   running_summary=running_summary)
    m = _SUMMARY_RE.search(final_text or "")
    if m:
        return m.group(1).strip()
    if mode == "threecall":
        # 存储会话没给 summary，再单独调一次要总结
        resp = _chat_create(
            model=ALIYUN_MODEL, max_tokens=120, temperature=0.2,
            messages=[{"role": "user",
                       "content": f"用一句话总结这段对话：\n{chunk_text}"}])
        log_usage(resp, phase="v7_summary")
        return (resp.choices[0].message.content or "").strip()
    return ""


# ==================== Structure Rebalancing (design §rhythm 2/3) ====================
# Two rhythms share one bash-agent: tidy_local (rhythm 2, light per-session
# rebalance, skipped entirely when the library is already clean) and
# reorganize_library (rhythm 3, full reshuffle triggered by library growth).
# Whether a same-name summary page gets left behind after splitting a big file
# into a subdirectory is an experiment switch, controlled by the
# NATIVEMEM_SUMMARY_PAGE env var ("on"/"off", default "off").

_SUMMARY_PAGE_RULE_ON = "拆子目录后，在父层留一个同名摘要页（每个子主题一行摘要 + 链接）。"
_SUMMARY_PAGE_RULE_OFF = "拆成子目录，子文件名即话题。"

_REBALANCE_PROMPT = """\
你在整理一个用文件系统组织的记忆库。工作目录就是记忆库根目录，只能用 bash 操作。

## 结构体检报告（代码测量，⚠ 是失衡提示，不是强制命令）
{report}

## 顶层结构
{top_view}

{structure_standard}

你有全局视野，把这个库重排成任何时刻都好检索的样子：拆过大文件、合分散主题、
按子主题给过宽目录建子目录、并孤儿小文件、统一命名。
{summary_page_rule}
拆合建删随你判断。
红线：别丢任何记忆内容，别改每条的 [source](dia_id) 锚。
"""


def _rebalance_summary_page_rule():
    """读 NATIVEMEM_SUMMARY_PAGE 环境变量，返回填进 prompt 的那句指导（或空串）。"""
    mode = os.environ.get("NATIVEMEM_SUMMARY_PAGE", "off")
    return _SUMMARY_PAGE_RULE_ON if mode == "on" else _SUMMARY_PAGE_RULE_OFF


def _structure_standard_for_rebalance():
    """STRUCTURE_STANDARD 本体不改（Task 5 产出，_AGENT_STORE_PROMPT 仍用原文）。
    但它有一条泛泛提到「摘要页」的准则，会在 NATIVEMEM_SUMMARY_PAGE=off 时和
    _REBALANCE_PROMPT 的显式开关矛盾（off 时 prompt 里不该出现「摘要页」字样）。
    这里只在喂给 rebalance agent 的这一份拷贝里，把该条准则替换成不提摘要页的
    版本；is off 时用 _SUMMARY_PAGE_RULE_OFF 的说法保持一致，on 时原样保留。"""
    if os.environ.get("NATIVEMEM_SUMMARY_PAGE", "off") == "on":
        return STRUCTURE_STANDARD
    target = "大到超标才拆子目录、留摘要页+链接。"
    replacement = "大到超标才拆子目录，子文件名即话题。"
    assert target in STRUCTURE_STANDARD, "STRUCTURE_STANDARD wording drifted; update _structure_standard_for_rebalance"
    return STRUCTURE_STANDARD.replace(target, replacement)


def _run_rebalance_agent(memory_dir, report, top_view, max_rounds=15):
    """跑一轮 bash-agent 重排会话（同 _run_store_agent 的 messages 拼装/tool_call
    处理/退出条件写法），返回实际执行的 bash 轮数。"""
    prompt = _REBALANCE_PROMPT.format(
        report=report, top_view=top_view,
        structure_standard=_structure_standard_for_rebalance(),
        summary_page_rule=_rebalance_summary_page_rule())
    messages = [{"role": "user", "content": prompt}]
    rounds = 0
    for _round_i in range(max_rounds):
        for retry in range(6):
            try:
                resp = _chat_create(
                    model=ALIYUN_MODEL, messages=messages, tools=TOOLS,
                    max_tokens=2000, temperature=0.2)
                break
            except Exception:  # noqa: BLE001
                if retry < 5:
                    time.sleep(3 * (retry + 1))
                else:
                    raise
        log_usage(resp, phase="v7_reorg")
        rounds += 1
        msg = resp.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()
        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break
        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            out = execute_tool(tc.function.name, args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": out})
    return rounds


def tidy_local(memory_dir, big=150, small=8, wide=20):
    """节奏2：每 session 末轻量再平衡。无 ⚠ 直接返回 0（省 API）。
    有则文件内并近义标题（consolidate_topics，确定性）+ 对 big/dup 项让模型局部修，
    返回处理的 ⚠ 数。"""
    res = inspect_structure(memory_dir, big=big, small=small, wide=wide)
    if not res["warnings"]:
        return 0
    handled = 0
    # 先文件内并近义标题（步骤A，已实现，确定性）
    for root, dirnames, filenames in os.walk(memory_dir):
        dirnames[:] = [d for d in dirnames if d != "raw" and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".md"):
                consolidate_topics(os.path.join(root, fn))
    # 再对 big/dup 局部让模型修（用同一 rebalance agent，但只喂当前报告）
    if any(w["kind"] in ("big", "dup") for w in res["warnings"]):
        top_view = _top_level_view(memory_dir)
        _run_rebalance_agent(memory_dir, res["report"], top_view)
        handled = len(res["warnings"])
    return handled


def reorganize_library(memory_dir, big=150, small=8, wide=20):
    """节奏3：库规模增长触发的全局重构（= 记忆初始化）。模型拿全局报告+顶层视图重排。
    返回 bash 轮数。"""
    res = inspect_structure(memory_dir, big=big, small=small, wide=wide)
    top_view = _top_level_view(memory_dir)
    return _run_rebalance_agent(memory_dir, res["report"], top_view)


# ==================== Topic Consolidation (design §B / §5.2) ====================
# The core reorganization job: merge near-synonym topic headings that the
# incremental writer created blindly (## mental health career / work /
# counseling → one). Code finds candidate pairs by title similarity; the model
# only judges whether each candidate pair should merge and what to name it.
# This is the "code finds problems, model does semantic judgment" split.

_MERGE_PROMPT = """You are tidying the topic headings inside one person's memory file. The writer added headings incrementally and created some near-duplicates. Below are candidate heading groups that MIGHT be about the same thing.

For each group, decide: should these headings be merged into one? If yes, give the single best heading name (reuse the clearest existing name or a short better one). If they are actually distinct topics, keep them separate.

Candidate groups:
{groups}

Output JSON only:
{{"merges": [{{"from": ["## mental health career", "## mental health work"], "into": "## mental health career", "merge": true}}, {{"from": ["## art", "## adoption"], "merge": false}}]}}
"""


def _heading_similarity(a, b):
    """Cheap lexical similarity between two topic headings (0..1). Word overlap
    (Jaccard) — enough to nominate candidates; the model makes the real call."""
    wa = set(re.sub(r"[^a-z0-9 ]", " ", a.lower().lstrip("# ")).split())
    wb = set(re.sub(r"[^a-z0-9 ]", " ", b.lower().lstrip("# ")).split())
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


_STOPWORDS = {"the", "a", "an", "of", "and", "to", "in", "for", "with",
              "my", "her", "his", "their", "s",
              # generic topic words that co-occur across unrelated topics —
              # clustering on these creates bad candidates (career plans vs
              # adoption plans). Keep clustering to the DISTINCTIVE noun.
              "plans", "plan", "journey", "work", "life", "story", "stories",
              "memory", "memories", "experience", "experiences", "activity",
              "activities", "event", "events", "update", "updates", "general"}


def _head_words(h):
    return [w for w in re.sub(r"[^a-z0-9 ]", " ", h.lower().lstrip("# ")).split()
            if w and w not in _STOPWORDS]


def _find_merge_candidates(headings, threshold=0.34):
    """Nominate candidate groups of headings that MIGHT be the same topic.

    Two signals, unioned: (a) pairwise word-overlap ≥ threshold; (b) sharing a
    salient content word (so 'adoption research' / 'adoption council meeting'
    that don't overlap pairwise still land in one 'adoption' group). The model
    makes the real merge decision — over-nominating is fine, it just judges."""
    uniq = list(dict.fromkeys(headings))
    # (b) cluster by shared salient word
    word_to_heads = {}
    for h in uniq:
        for w in _head_words(h):
            if len(w) >= 4:  # skip tiny words like 'art' vs 'arts'? keep >=4 salient
                word_to_heads.setdefault(w, []).append(h)
    groups = []
    seen_group = set()
    for w, hs in word_to_heads.items():
        if len(hs) >= 2:
            key = tuple(sorted(hs))
            if key not in seen_group:
                seen_group.add(key)
                groups.append(hs)
    # (a) also add short-word shared groups (e.g. 'art')  + pairwise overlap
    for i, h in enumerate(uniq):
        grp = [h]
        for other in uniq[i + 1:]:
            if _heading_similarity(h, other) >= threshold:
                grp.append(other)
        if len(grp) >= 2:
            key = tuple(sorted(grp))
            if key not in seen_group:
                seen_group.add(key)
                groups.append(grp)
    return groups


def consolidate_topics(md_path, max_retry=4):
    """Merge near-synonym ## headings in one markdown file.

    Steps (design §5.2): (1) code reads all ## headings; (2) code finds
    similar candidate groups; (3) model judges which groups truly merge;
    (4) code rewrites the file, moving entries under the merged heading and
    deduping exact-duplicate entries. Returns number of merges applied.
    """
    p = Path(md_path)
    if not p.exists():
        return 0
    lines = p.read_text().splitlines()
    # collect headings and the entry blocks under each
    sections = {}          # heading -> list of body lines
    order = []             # heading order of first appearance
    cur = None
    for ln in lines:
        if ln.startswith("## "):
            if ln not in sections:
                sections[ln] = []
                order.append(ln)
            cur = ln
        elif cur is not None:
            sections[cur].append(ln)
    headings = list(order)
    if len(headings) < 2:
        return 0

    # exact-duplicate heading merge is free (code): already handled by dict, but
    # entries from repeated same-name headings were concatenated above.
    candidates = _find_merge_candidates(headings)
    if not candidates:
        # still rewrite to collapse any exact-dup headings we merged in the dict
        _rewrite_sections(p, order, sections, {})
        return 0

    groups_txt = "\n".join(
        f"Group {i+1}: " + " | ".join(g) for i, g in enumerate(candidates))
    for retry in range(max_retry):
        try:
            resp = _chat_create(
                model=ALIYUN_MODEL,
                messages=[{"role": "system",
                           "content": _MERGE_PROMPT.format(groups=groups_txt)}],
                max_tokens=1500, temperature=0.1,
                response_format={"type": "json_object"})
            break
        except Exception:  # noqa: BLE001
            if retry < max_retry - 1:
                time.sleep(3 * (retry + 1))
            else:
                return 0
    log_usage(resp, phase="consolidate")
    try:
        merges = json.loads(re.sub(r"<think>.*?</think>", "",
                            resp.choices[0].message.content or "",
                            flags=re.DOTALL)).get("merges", [])
    except Exception:  # noqa: BLE001
        return 0

    # apply merges: remap headings, then rewrite
    remap = {}
    n = 0
    for m in merges:
        if not m.get("merge") or not m.get("into"):
            continue
        into = m["into"] if m["into"].startswith("##") else "## " + m["into"].lstrip("# ")
        for h in m.get("from", []):
            if h in sections and h != into:
                remap[h] = into
                n += 1
    if not remap:
        _rewrite_sections(p, order, sections, {})
        return 0
    _rewrite_sections(p, order, sections, remap)
    return n


def _rewrite_sections(path, order, sections, remap):
    """Rewrite file: merged headings share one section, entries deduped."""
    # target heading order (first occurrence, remapped)
    final_order = []
    merged = {}   # target heading -> combined body lines
    for h in order:
        tgt = remap.get(h, h)
        if tgt not in merged:
            merged[tgt] = []
            final_order.append(tgt)
        merged[tgt].extend(sections.get(h, []))
    # dedup exact-duplicate entry lines within each section (keep order)
    out = []
    # preserve any pre-heading preamble (title line etc.)
    header = []
    with open(path) as f:
        for ln in f:
            ln = ln.rstrip("\n")
            if ln.startswith("## "):
                break
            header.append(ln)
    out.extend(header)
    for h in final_order:
        out.append(h)
        seen = set()
        for bl in merged[h]:
            key = bl.strip()
            if key and key in seen:
                continue
            if key:
                seen.add(key)
            out.append(bl)
    Path(path).write_text("\n".join(out) + "\n")


# ==================== Structure Reorganization ====================

REORG_PROMPT = """You are reorganizing a personal memory folder to improve navigation efficiency.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

The following structural issues were detected:
{issues}

For each issue, fix it:
- **File too long**: split it into multiple sub-files under a directory named after the entity. Create an index file with links to each sub-file. Update any cross-reference links in other files.
- **Directory too crowded**: group related files into subdirectories by theme. Update any cross-reference links.

Use standard shell commands (ls, cat, grep, sed, mkdir -p, mv, etc.) to perform the reorganization.
Preserve all content — do not delete any information, only reorganize it.
"""

def check_and_reorganize(memory_dir, max_file_lines=100, max_dir_files=15, max_rounds=15):
    """Check memory structure and reorganize if needed."""
    mem = Path(memory_dir)
    issues = []

    # Check for oversized files
    for f in mem.rglob("*.md"):
        if "raw" in f.parts:
            continue
        line_count = len(f.read_text().splitlines())
        rel = f.relative_to(mem)
        if line_count > max_file_lines:
            issues.append(f"File too long: {rel} ({line_count} lines, limit {max_file_lines})")

    # Check for crowded directories
    for d in [mem] + [x for x in mem.rglob("*") if x.is_dir() and "raw" not in x.parts]:
        children = [c for c in d.iterdir() if not c.name.startswith(".") and c.name != "raw"]
        rel = d.relative_to(mem) if d != mem else Path(".")
        if len(children) > max_dir_files:
            issues.append(f"Directory too crowded: {rel}/ ({len(children)} items, limit {max_dir_files})")

    if not issues:
        return 0

    print(f"  Reorganizing: {len(issues)} issues found")
    for iss in issues:
        print(f"    {iss}")

    messages = [
        {"role": "system", "content": REORG_PROMPT.format(issues="\n".join(f"- {i}" for i in issues))},
        {"role": "user", "content": "Please reorganize the memory folder to fix these issues."}
    ]

    for round_i in range(max_rounds):
        try:
            response = _chat_create(
                model=ALIYUN_MODEL, messages=messages,
                tools=TOOLS, max_tokens=4000, temperature=0.3,
            )
        except:
            time.sleep(2)
            continue

        log_usage(response, phase="reorganize")
        msg = response.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            messages.append({"role": "assistant", "content": msg.content or ""})
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                fn_args = json.loads(tc.function.arguments)
            except:
                fn_args = {}
            result = execute_tool(tc.function.name, fn_args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return len(issues)


# ==================== Retrieval ====================

RETRIEVAL_PROMPT = """You are answering questions using ONLY a personal memory folder.
You have a bash tool to execute shell commands. The working directory is the root of the memory folder.

## Strategy
1. `ls -lhR` to see the full folder structure, file names, and sizes.
2. Pick the file or directory whose name is most relevant to the question.
3. `grep '^#' path/to/file.md` to see its heading structure before reading the full file.
4. `cat path/to/file.md` or `sed -n '/^## Heading$/,/^## /p' path/to/file.md` to read the relevant section.
5. If the answer involves multiple entities, follow cross-reference links (See also: [...]) to related files.
6. If your first choice does not contain the answer, try another file.

## Rules
- ONLY use information found in the files. Do NOT use your training knowledge.
- If the information does not exist in the files, answer exactly: "No information available."
- Answer as briefly as possible — just the key facts, no explanation or elaboration.
"""

def retrieve(question, memory_dir, max_rounds=10):
    """Retrieve answer from memory."""
    messages = [
        {"role": "system", "content": RETRIEVAL_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]
    trace = []
    for _ in range(max_rounds):
        try:
            resp = _chat_create(
                model=ALIYUN_MODEL, messages=messages,
                tools=READ_TOOLS, max_tokens=2000, temperature=0.3
            )
        except:
            time.sleep(2)
            continue

        log_usage(resp, phase="retrieve")
        msg = resp.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            return msg.content or "NOT FOUND", trace

        messages.append(msg)
        for tc in msg.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except:
                args = {}
            trace.append({"tool": tc.function.name, "args": args})
            result = execute_tool(tc.function.name, args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return "NOT FOUND", trace


# ==================== Validation ====================

def validate_memory(memory_dir, raw_dir, num_samples=5):
    """Validate M against R."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        return []

    raw_files = list(raw_path.glob("*.md"))
    if not raw_files:
        return []

    samples = random.sample(raw_files, min(num_samples, len(raw_files)))
    results = []

    for raw_file in samples:
        content = raw_file.read_text()
        lines = [l for l in content.splitlines() if l.strip() and not l.startswith("#") and not l.startswith("Date:")]
        if not lines:
            continue
        passage = random.choice(lines)

        # Generate question
        try:
            resp = _chat_create(
                model=ALIYUN_MODEL,
                messages=[{"role": "user", "content": f"Generate one short factual question about:\n{passage}\nOutput ONLY the question."}],
                max_tokens=100, temperature=0.3
            )
            log_usage(resp, phase="validate_gen")
            question = resp.choices[0].message.content.strip()
            question = re.sub(r'<think>.*?</think>', '', question, flags=re.DOTALL).strip()
        except:
            continue

        answer, trace = retrieve(question, memory_dir)
        steps = len(trace)

        source_exists = any(raw_file.name in f.read_text() for f in Path(memory_dir).rglob("*.md"))

        found = "NOT FOUND" not in (answer or "NOT FOUND")
        # Also check if steps are abnormally high (structural drift)
        high_steps = steps > 15
        if found and not high_steps:
            status = "OK"
        elif found and high_steps:
            status = "SLOW"
        elif source_exists:
            status = "STRUCTURAL"
        else:
            status = "MISSING"

        # Repair loop: up to 2 rounds
        status_after = status
        final_steps = steps
        repaired = False
        if status != "OK":
            for repair_round in range(2):
                repaired = repair_memory(memory_dir, raw_dir, raw_file.name,
                                          passage, question, status_after, trace)
                if not repaired:
                    break
                # Re-validate
                answer2, trace2 = retrieve(question, memory_dir)
                final_steps = len(trace2)
                found2 = "NOT FOUND" not in (answer2 or "NOT FOUND")
                high_steps2 = final_steps > 15
                if found2 and not high_steps2:
                    status_after = "REPAIRED"
                    break
                elif found2 and high_steps2:
                    status_after = "SLOW"
                    trace = trace2  # use new trace for next repair round
                else:
                    status_after = "STRUCTURAL" if source_exists else "MISSING"
                    trace = trace2

        if status_after != status:
            print(f"  Validation: {status} -> {status_after} (steps: {steps} -> {final_steps}) - {question[:60]}")
        else:
            print(f"  Validation: {status} (steps={steps}) - {question[:60]}")

        results.append({
            "raw_file": raw_file.name, "passage": passage, "question": question,
            "found": found, "steps": steps, "source_exists": source_exists,
            "status": status, "status_after_repair": status_after,
            "repaired": repaired, "answer": answer,
        })

    return results


REPAIR_PROMPT = """You are repairing the organization of a personal memory folder.

A validation check found that the following information could not be efficiently retrieved.

Failure type: {failure_type}
Question that failed: {question}
Passage that should be findable: {passage}
Source file: {source_file}

The retriever attempted to find this information and took the following navigation path:
{trace_description}

Based on the failure type and the navigation trace:
- STRUCTURAL: The information is stored somewhere but the retriever navigated to the wrong location. Look at where the retriever went (the trace above) and where the information actually is. Add a cross-reference link FROM the location the retriever visited TO the actual location of the information, so next time the retriever will find it.
- SLOW: The information was eventually found but the retriever took a roundabout path. Add a cross-reference link at the location the retriever first visited, pointing to where the information actually is.
- MISSING: The information was never stored. First browse the folder to find where you would search for it (RAM), then store the passage there with a source link.

Use shell commands (ls, cat, grep, sed, etc.) to inspect and repair. Do NOT rewrite existing content. Only add links, rename, or add new entries."""


def repair_memory(memory_dir, raw_dir, source_file, passage, question, failure_type, trace):
    """Repair memory structure based on validation failure."""
    # Format trace into readable description
    trace_lines = []
    for i, step in enumerate(trace):
        args = step.get("args", {})
        cmd = args.get("command", str(args))
        trace_lines.append(f"  Step {i+1}: {cmd}")
    trace_description = "\n".join(trace_lines) if trace_lines else "  (no navigation steps recorded)"

    messages = [
        {"role": "system", "content": REPAIR_PROMPT.format(
            failure_type=failure_type, question=question,
            passage=passage, source_file=source_file,
            trace_description=trace_description,
        )},
        {"role": "user", "content": f"Please inspect the memory folder and repair this issue. "
                                     f"The source file is raw/{source_file}."}
    ]

    tool_trace = []
    for round_i in range(10):
        for retry in range(3):
            try:
                response = _chat_create(
                    model=ALIYUN_MODEL, messages=messages,
                    tools=TOOLS, max_tokens=4000, temperature=0.3,
                )
                break
            except Exception as e:
                if retry < 2:
                    time.sleep(2)
                else:
                    return False

        log_usage(response, phase="repair")
        msg = response.choices[0].message
        if msg.content:
            msg.content = re.sub(r'<think>.*?</think>', '', msg.content, flags=re.DOTALL).strip()

        if not msg.tool_calls:
            break

        messages.append(msg)
        for tc in msg.tool_calls:
            fn_name = tc.function.name
            try:
                fn_args = json.loads(tc.function.arguments)
            except:
                fn_args = {}
            tool_trace.append({"tool": fn_name, "args": fn_args})
            result = execute_tool(fn_name, fn_args, memory_dir)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

    return len(tool_trace) > 0




# ==================== CLI ====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="benchmarks/locomo/data/locomo10.json")
    parser.add_argument("--sample", type=int, default=0)
    parser.add_argument("--max-sessions", type=int, default=None)
    parser.add_argument("--outdir", default="results/run")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-samples", type=int, default=10)
    args = parser.parse_args()

    with open(args.data) as f:
        data = json.load(f)

    sample = data[args.sample]
    conv = sample["conversation"]
    speaker_a = conv.get("speaker_a", "Person A")
    speaker_b = conv.get("speaker_b", "Person B")

    memory_dir = os.path.join(args.outdir, f"memory_sample{args.sample}")
    raw_dir = os.path.join(args.outdir, f"raw_sample{args.sample}")
    os.makedirs(memory_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    sessions, dates = [], []
    i = 1
    while f"session_{i}" in conv:
        sessions.append(conv[f"session_{i}"])
        dates.append(conv.get(f"session_{i}_date_time", ""))
        i += 1

    if args.max_sessions:
        sessions = sessions[:args.max_sessions]
        dates = dates[:args.max_sessions]

    print(f"Sample {args.sample}: {len(sessions)} sessions, speakers: {speaker_a} & {speaker_b}")
    print(f"Model: {ALIYUN_MODEL}")

    t0 = time.time()
    chunk_idx = 0
    CHUNK_SIZE = 10

    for si, (session, date) in enumerate(zip(sessions, dates)):
        turns = session if isinstance(session, list) else [{"speaker": "user", "text": session}] if isinstance(session, str) else []
        for cs in range(0, len(turns), CHUNK_SIZE):
            chunk = turns[cs:cs + CHUNK_SIZE]
            chunk_text = ""
            for t in chunk:
                if isinstance(t, dict):
                    speaker = t.get("speaker", t.get("role", "user"))
                    text = t.get("text", t.get("content", ""))
                elif isinstance(t, str):
                    speaker, text = "user", t
                else:
                    continue
                chunk_text += f"{speaker}: {text}\n\n"

            if not chunk_text.strip():
                continue

            chunk_idx += 1
            source_file = save_to_raw_archive(chunk_text, date, si, chunk_idx, raw_dir)
            source_path = f"raw/{source_file}"

            print(f"\nChunk {chunk_idx} (session {si+1}/{len(sessions)}):")
            try:
                n_msgs, trace = process_chunk(chunk_text, date, memory_dir, source_path)
                n_tools = len(trace)
                print(f"  Done: {n_tools} tool calls, {n_msgs} messages")
            except Exception as e:
                print(f"  Error: {e}")

    elapsed = time.time() - t0
    file_count = len(list(Path(memory_dir).rglob("*.md")))

    print(f"\n=== Build Complete ===")
    print(f"Time: {elapsed:.0f}s")
    print(f"Wiki files: {file_count}")
    print(f"Total LLM calls: {TOTAL_CALLS}")
    print(f"Total tokens: {TOTAL_TOKENS:,}")

    stats = {
        "sample": args.sample, "model": ALIYUN_MODEL,
        "sessions": len(sessions), "chunks": chunk_idx,
        "files": file_count, "calls": TOTAL_CALLS,
        "tokens": TOTAL_TOKENS, "time_s": round(elapsed, 1),
        "call_log": CALL_LOG,
    }
    with open(os.path.join(args.outdir, f"build_stats_sample{args.sample}.json"), "w") as f:
        json.dump(stats, f, indent=2)

    # Structure reorganization
    print(f"\n=== Structure Reorganization ===")
    reorgs = check_and_reorganize(memory_dir)
    print(f"Reorganizations performed: {reorgs}")

    if args.validate:
        print(f"\n=== Validation ({args.validate_samples} samples) ===")
        results = validate_memory(memory_dir, raw_dir, args.validate_samples)
        found = sum(1 for r in results if r.get("status_after_repair") in ("OK", "REPAIRED") or r.get("status") == "OK")
        print(f"Final pass rate: {found}/{len(results)}")
        with open(os.path.join(args.outdir, f"validation_sample{args.sample}.json"), "w") as f:
            json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()

"""Tool schemas and view definitions exposed to retrieval models."""


MEMORY_COMPONENTS = (
    "topics",
    "timeline",
    "sources",
    "core",
    "recent",
    "relations",
)


def normalize_memory_components(value):
    """Return a stable component tuple from a CLI string or iterable."""
    if value is None:
        return None
    raw = value.split(",") if isinstance(value, str) else value
    result = tuple(dict.fromkeys(
        str(component).strip().lower()
        for component in raw
        if str(component).strip()
    ))
    if not result:
        raise ValueError("memory_components must not be empty")
    invalid = tuple(
        component for component in result if component not in MEMORY_COMPONENTS
    )
    if invalid:
        raise ValueError(
            "unknown memory components: " + ", ".join(invalid)
        )
    return result

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": (
            "Run a read-only POSIX Bash command in the visible memory workspace, "
            "regardless of the host operating system. Supported "
            "commands: find, rg, grep, cat, sed -n, ls, head, tail, wc, sort, "
            "uniq, cut, and pwd. Claude Code manages tool-result context."
        ),
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string"},
        }, "required": ["command"]},
    }},
    {"type": "function", "function": {
        "name": "list_memory_files",
        "description": "List visible memory files.",
        "parameters": {"type": "object", "properties": {
            "prefix": {"type": "string"},
        }},
    }},
    {"type": "function", "function": {
        "name": "read_memory_file",
        "description": (
            "Read one visible memory file. Optional offset and limit select a "
            "1-based line window."
        ),
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"},
            "offset": {"type": "integer", "minimum": 1},
            "limit": {"type": "integer", "minimum": 1},
        }, "required": ["path"]},
    }},
    {"type": "function", "function": {
        "name": "bm25_search",
        "description": "Rank topic and source memory by sparse lexical relevance.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "path_prefix": {"type": "string"},
            "date_from": {"type": "string", "description": "Inclusive lower bound: YYYY, YYYY-MM, or YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Inclusive upper bound: YYYY, YYYY-MM, or YYYY-MM-DD."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "embedding_search",
        "description": "Rank topic and source memory by dense semantic similarity.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "date_from": {"type": "string", "description": "Inclusive lower bound: YYYY, YYYY-MM, or YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Inclusive upper bound: YYYY, YYYY-MM, or YYYY-MM-DD."},
        }, "required": ["query"]},
    }},
    {"type": "function", "function": {
        "name": "memory_search",
        "description": (
            "Rank topic and source memory by fused lexical + semantic relevance "
            "(BM25 and embedding merged with reciprocal rank fusion)."
        ),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string"},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 10},
            "path_prefix": {"type": "string"},
            "date_from": {"type": "string", "description": "Inclusive lower bound: YYYY, YYYY-MM, or YYYY-MM-DD."},
            "date_to": {"type": "string", "description": "Inclusive upper bound: YYYY, YYYY-MM, or YYYY-MM-DD."},
        }, "required": ["query"]},
    }},
]

CONDITION_VIEWS = {
    "dual_source": ("topics", "timeline", "sources", "recent"),
    "topic_source": ("topics", "sources", "recent"),
    "timeline_source": ("timeline", "sources", "recent"),
    "dual_no_source": ("topics", "timeline", "recent"),
}

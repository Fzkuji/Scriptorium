"""Tool schemas and view definitions exposed to retrieval models."""

TOOL_DEFINITIONS = [
    {"type": "function", "function": {
        "name": "bash",
        "description": (
            "Run a read-only command in the visible memory workspace. Supported "
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
]

CONDITION_VIEWS = {
    "dual_source": ("topics", "timeline", "sources", "recent"),
    "topic_source": ("topics", "sources", "recent"),
    "timeline_source": ("timeline", "sources", "recent"),
    "dual_no_source": ("topics", "timeline", "recent"),
}

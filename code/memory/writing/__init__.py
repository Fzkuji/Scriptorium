"""Phase 1: conversation becomes memory.

The model decides one thing per fact — remember, update or forget — and
calls the matching verb. Where the fact is filed, its block ID and its
footnote are computed, not decided, so the format cannot come out wrong
underneath a judgement the model was never asked to make.
"""

from .session import write_sessions
from .subjects import path_for
from .tools import writing_tools

__all__ = ["path_for", "write_sessions", "writing_tools"]

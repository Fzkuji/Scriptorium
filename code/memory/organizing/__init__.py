"""Phase 2: memory stays usable.

Writing only ever adds; left alone a workspace grows duplicate facts, stale
structure and files that outgrew their subject, all of which make memory
slower to search and harder for a model to reorganize. `tidy` and `split`
remove what needs no judgement to fix — duplicates a string comparison
settles, a file only its size decided to break up; `reorganize` runs both
and then a model pass for the judgement calls that are left, including the
worded-differently pairs `tidy.merge_candidates` shortlists; `unreachable`
checks the result actually stayed findable.
"""

from .tidy import tidy
from .split import split
from .tools import organizing_tools
from .reorganize import reorganize
from .verify import unreachable

__all__ = ["organizing_tools", "reorganize", "split", "tidy", "unreachable"]

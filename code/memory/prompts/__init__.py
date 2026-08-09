"""Every prompt the memory system sends to a model, one file per stage.

``system`` holds the contract shared by the agents that edit memory; the
other modules hold the task each stage is given.

    system         SYSTEM_PROMPT, FEW_SHOT_INSTRUCTIONS
    write          WRITE_MEMORY
    organize       ORGANIZE_MEMORY
    verification   VERIFICATION_PROBE_TASK, _RETRIEVAL_TASK, _REPAIR_TASK
    answer         ANSWER_PROMPT
    retrieve       RETRIEVAL_PROMPT, FIND_MEMORY

Judge templates live with the evaluation scripts: they belong to the
benchmarks being scored, not to the memory system.
"""

from .organize import ORGANIZE_MEMORY
from .answer import ANSWER_PROMPT
from .retrieve import FIND_MEMORY, RETRIEVAL_PROMPT
from .system import FEW_SHOT_INSTRUCTIONS, SYSTEM_PROMPT
from .verification import (
    VERIFICATION_PROBE_TASK,
    VERIFICATION_REPAIR_TASK,
    VERIFICATION_RETRIEVAL_TASK,
)
from .write import WRITE_MEMORY

__all__ = [
    "ANSWER_PROMPT",
    "FEW_SHOT_INSTRUCTIONS",
    "FIND_MEMORY",
    "ORGANIZE_MEMORY",
    "RETRIEVAL_PROMPT",
    "SYSTEM_PROMPT",
    "VERIFICATION_PROBE_TASK",
    "VERIFICATION_REPAIR_TASK",
    "VERIFICATION_RETRIEVAL_TASK",
    "WRITE_MEMORY",
]

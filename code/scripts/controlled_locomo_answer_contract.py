"""Moved to scripts/controlled_locomo/controlled_locomo_answer_contract.py; kept so stored paths keep working."""

from scripts.controlled_locomo import controlled_locomo_answer_contract as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

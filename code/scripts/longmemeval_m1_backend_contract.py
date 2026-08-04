"""Moved to scripts/longmemeval_m1/longmemeval_m1_backend_contract.py; kept so stored paths keep working."""

from scripts.longmemeval_m1 import longmemeval_m1_backend_contract as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

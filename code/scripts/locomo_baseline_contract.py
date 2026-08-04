"""Moved to scripts/locomo_baselines/locomo_baseline_contract.py; kept so stored paths keep working."""

from scripts.locomo_baselines import locomo_baseline_contract as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

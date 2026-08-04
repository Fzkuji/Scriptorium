"""Moved to scripts/beam_controls/r115_beam_control_contract.py; kept so stored paths keep working."""

from scripts.beam_controls import r115_beam_control_contract as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

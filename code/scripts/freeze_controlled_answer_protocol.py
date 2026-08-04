"""Moved to scripts/controlled_locomo/freeze_controlled_answer_protocol.py; kept so stored paths keep working."""

from scripts.controlled_locomo import freeze_controlled_answer_protocol as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

if __name__ == "__main__":
    raise SystemExit(_moved.main())

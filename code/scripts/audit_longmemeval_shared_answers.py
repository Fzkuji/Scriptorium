"""Moved to scripts/longmemeval_m1/audit_longmemeval_shared_answers.py; kept so stored paths keep working."""

from scripts.longmemeval_m1 import audit_longmemeval_shared_answers as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

if __name__ == "__main__":
    raise SystemExit(_moved.main())

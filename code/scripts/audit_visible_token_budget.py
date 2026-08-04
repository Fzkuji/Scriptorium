"""Moved to scripts/token_budget/audit_visible_token_budget.py; kept so stored paths keep working."""

from scripts.token_budget import audit_visible_token_budget as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

if __name__ == "__main__":
    raise SystemExit(_moved.main())

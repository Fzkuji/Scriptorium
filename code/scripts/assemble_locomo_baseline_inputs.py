"""Moved to scripts/locomo_baselines/assemble_locomo_baseline_inputs.py; kept so stored paths keep working."""

from scripts.locomo_baselines import assemble_locomo_baseline_inputs as _moved

_SKIP = {'__name__', '__file__', '__loader__', '__spec__', '__package__', '__doc__'}
globals().update({k: v for k, v in vars(_moved).items() if k not in _SKIP})

if __name__ == "__main__":
    raise SystemExit(_moved.main())

"""Compatibility alias for the versioned V8 implementation."""

import sys

from src.nativemem_versions.v8 import memory

sys.modules[__name__] = memory

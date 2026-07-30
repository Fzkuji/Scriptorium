"""Compatibility alias for the versioned V11 implementation."""

import sys

from src.nativemem_versions.v11 import memory

sys.modules[__name__] = memory

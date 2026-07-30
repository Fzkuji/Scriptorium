"""Compatibility alias for the versioned V10 implementation."""

import sys

from src.nativemem_versions.v10 import memory

sys.modules[__name__] = memory

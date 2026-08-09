"""Make `code/` importable, so tests reach `src` and `scripts` as packages."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

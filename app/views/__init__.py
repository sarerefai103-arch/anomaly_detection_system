"""UI views for the top-level tabs.

Adds the app directory to ``sys.path`` so view modules can use the shared
helpers by their flat module names (``ui``, ``charts``, ...) regardless of
which module is imported first.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

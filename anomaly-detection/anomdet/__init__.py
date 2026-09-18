"""anomdet: a from-scratch anomaly detector.

By default this package exposes YOUR implementations from anomdet/core.py.

Set the environment variable ANOMDET_USE_REFERENCE=1 to swap in the reference
implementation instead.  Use that to (a) see what a finished stage is supposed
to produce before you write any code, and (b) diff your behaviour against the
reference when a test fails for reasons you cannot see.

    python3 stages/stage2_engines_2d.py                      # your code
    ANOMDET_USE_REFERENCE=1 python3 stages/stage2_engines_2d.py   # reference
"""

import os as _os
import sys as _sys
from pathlib import Path as _Path

_ROOT = _Path(__file__).resolve().parent.parent

if _os.environ.get("ANOMDET_USE_REFERENCE", "") in ("1", "true", "yes"):
    _sys.path.insert(0, str(_ROOT / "reference"))
    import reference_impl as core  # noqa: F401
    USING_SOLUTION = True
else:
    from . import core  # noqa: F401
    USING_SOLUTION = False

from .core_api import *  # noqa: F401,F403,E402
from .core_api import __all__  # noqa: F401,E402

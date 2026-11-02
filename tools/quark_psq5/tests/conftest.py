import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parents[2]
_VENDOR_QUARK = _TOOLS.parents[0] / "vendor" / "quark"
for path in (_TOOLS, _VENDOR_QUARK):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))

"""Minimal conftest for shadylib tests – shadylib is a pure Python package,
no HA stubs needed. Just ensure the package is importable."""
import sys, pathlib

# When running from the repo root, shadylib is installed or on the path already.
# When running from the shadylib/ directory directly, add src/ to sys.path.
_src = pathlib.Path(__file__).parent.parent / "src"
if _src.is_dir() and str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

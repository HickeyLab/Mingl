from importlib.metadata import PackageNotFoundError, version

from . import pl, pp, tl

# Try to get the installed package version; fall back if not installed
try:
    __version__ = version("MINGL")  # must match [project].name in pyproject.toml
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = ["pl", "pp", "tl", "__version__"]

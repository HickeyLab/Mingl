"""Compatibility package for older ``import MINGLE`` users."""

from importlib import import_module

from mingl import *  # noqa: F403
from mingl import __all__, __version__

_mingl = import_module("mingl")
pl = import_module("mingl.pl")
pp = import_module("mingl.pp")
tl = import_module("mingl.tl")
__path__ = _mingl.__path__

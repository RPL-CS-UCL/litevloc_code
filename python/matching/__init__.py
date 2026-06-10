"""Compatibility shim: re-export vismatch as 'matching'.

New code should import vismatch directly.
"""
import sys
import os

_vismatch_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../vismatch')
)
if _vismatch_dir not in sys.path:
    sys.path.insert(0, _vismatch_dir)

from vismatch import available_models, get_matcher  # noqa: F401
from vismatch import viz as viz2d                   # noqa: F401

__all__ = ["available_models", "get_matcher", "viz2d"]

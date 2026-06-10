import sys
import os

_vismatch_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../vismatch')
if _vismatch_dir not in sys.path:
    sys.path.insert(0, _vismatch_dir)

from vismatch.utils import get_image_pairs_paths, to_numpy

__all__ = ["get_image_pairs_paths", "to_numpy"]

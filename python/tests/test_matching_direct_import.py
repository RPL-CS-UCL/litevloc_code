from __future__ import annotations
import sys
import importlib
from pathlib import Path
from unittest.mock import MagicMock

LITEVLOC_PY  = Path(__file__).resolve().parents[1]
VISMATCH_ROOT = LITEVLOC_PY.parents[1] / "vismatch"

_OPEN3D_MOCK = MagicMock()


def _fresh(name: str, monkeypatch):
    for key in [k for k in sys.modules if k == name or k.startswith(name + ".")]:
        monkeypatch.delitem(sys.modules, key, raising=False)
    return importlib.import_module(name)


def test_utils_image_matching_imports_vismatch_directly(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(LITEVLOC_PY))
    monkeypatch.syspath_prepend(str(VISMATCH_ROOT))
    mod = _fresh("utils.utils_image_matching_method", monkeypatch)
    import vismatch
    print(f"[test_direct_import] get_matcher module={mod.get_matcher.__module__}")
    assert mod.get_matcher is vismatch.get_matcher


def test_utils_pipeline_imports_vismatch_directly(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(LITEVLOC_PY))
    monkeypatch.syspath_prepend(str(VISMATCH_ROOT))
    monkeypatch.setitem(sys.modules, "open3d", _OPEN3D_MOCK)
    mod = _fresh("utils.utils_pipeline", monkeypatch)
    import vismatch
    print(f"[test_pipeline_import] available_models id match={mod.available_models is vismatch.available_models}")
    assert mod.available_models is vismatch.available_models


def test_default_matcher_is_sift_lightglue(monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(LITEVLOC_PY))
    monkeypatch.syspath_prepend(str(VISMATCH_ROOT))
    monkeypatch.setitem(sys.modules, "open3d", _OPEN3D_MOCK)
    mod = _fresh("utils.utils_pipeline", monkeypatch)
    print(f"[test_default_matcher] sift-lightglue={'sift-lightglue' in mod.available_models}, "
          f"mickey={'mickey' in mod.available_models}")
    assert "sift-lightglue" in mod.available_models
    assert "mickey" not in mod.available_models


def test_save_output_uses_canonical_inlier_keys(tmp_path, monkeypatch) -> None:
    import numpy as np
    monkeypatch.syspath_prepend(str(LITEVLOC_PY))
    monkeypatch.syspath_prepend(str(VISMATCH_ROOT))
    mod = _fresh("utils.utils_image_matching_method", monkeypatch)
    (tmp_path / "preds").mkdir()
    result = {
        "num_inliers": 2,
        "H": np.eye(3),
        "inlier_kpts0": np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        "inlier_kpts1": np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
    }
    p = mod.save_output(result, "a.jpg", "b.jpg", "master", 2048, [512, 288], tmp_path, 0)
    print(f"[test_save_output] dict_path={p}, exists={Path(p).exists()}")
    assert Path(p).exists()


def test_save_visualization_uses_plot_matches_high_level(tmp_path, monkeypatch) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    monkeypatch.syspath_prepend(str(LITEVLOC_PY))
    monkeypatch.syspath_prepend(str(VISMATCH_ROOT))
    mod = _fresh("utils.utils_image_matching_method", monkeypatch)
    img = torch.zeros(3, 16, 16)
    kpts = np.array([[1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
    out_dir = tmp_path
    viz_path = mod.save_visualization(img, img, kpts, kpts, out_dir, 0, text="2 matches")
    print(f"[test_save_viz] viz_path={viz_path}, exists={viz_path.exists()}")
    assert viz_path.exists()
    plt.close("all")

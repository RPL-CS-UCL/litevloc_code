"""Unit tests for ros_loc_pipeline.py.

No ROS master, GPU, or model weights are needed: rospy and the other ROS
packages, plus loc_pipeline/image_node/utils.utils_ros/utils.utils_stamped_poses
(which pull in cv_bridge, tf2_msgs, gtsam, ... -- unavailable or unneeded here)
are replaced with MagicMock stand-ins before import, following the pattern in
test_matching_direct_import.py.
"""
from __future__ import annotations

import sys
import threading
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

LITEVLOC_PY = Path(__file__).resolve().parents[1]

_ROS_STUB_MODULES = [
	"rospy",
	"message_filters",
	"nav_msgs",
	"nav_msgs.msg",
	"sensor_msgs",
	"sensor_msgs.msg",
	"std_srvs",
	"std_srvs.srv",
	"loc_pipeline",
	"image_node",
	"utils.utils_ros",
	"utils.utils_stamped_poses",
]


@pytest.fixture()
def mod(monkeypatch):
	"""Import a fresh ros_loc_pipeline module with ROS/heavy deps stubbed out."""
	monkeypatch.syspath_prepend(str(LITEVLOC_PY))
	for name in _ROS_STUB_MODULES:
		monkeypatch.setitem(sys.modules, name, MagicMock())
	monkeypatch.delitem(sys.modules, "ros_loc_pipeline", raising=False)
	import ros_loc_pipeline as module
	yield module
	monkeypatch.delitem(sys.modules, "ros_loc_pipeline", raising=False)


def _header(frame_id="camera", stamp_sec=1.0):
	return types.SimpleNamespace(frame_id=frame_id, stamp=types.SimpleNamespace(to_sec=lambda: stamp_sec))


def _make_rgb_msg(width, height, pixel_bgr=(10, 20, 30), encoding="bgr8"):
	b, g, r = pixel_bgr
	row = bytes([b, g, r]) * width
	data = row * height
	return types.SimpleNamespace(
		encoding=encoding, width=width, height=height, step=width * 3,
		data=data, header=_header(),
	)


def _make_depth_msg(width, height, mm_value, encoding="16UC1"):
	arr = np.full((height, width), mm_value, dtype="<u2")
	return types.SimpleNamespace(
		encoding=encoding, width=width, height=height, step=width * 2,
		is_bigendian=False, data=arr.tobytes(), header=_header(),
	)


def _make_camera_info_msg(width, height):
	K = [float(x) for x in np.eye(3).flatten()]
	return types.SimpleNamespace(K=K, width=width, height=height)


def _make_args():
	return SimpleNamespace(image_size=None, device="cpu", global_pos_threshold=10.0)


def _make_loc(args):
	loc = SimpleNamespace()
	loc.has_global_pos = False
	loc.has_local_pos = False
	loc.ref_map_node = None
	loc.args = args
	loc.depth_range = (0.0, 100.0)
	loc.vpr_model = MagicMock(return_value=torch.zeros((1, 8)))
	loc.image_graph = MagicMock()
	loc.perform_global_loc = MagicMock(return_value={"succ": False, "map_id": None})
	loc.perform_local_loc = MagicMock(return_value={"succ": False, "T_w_obs": None})
	loc.publish_message = MagicMock()
	loc.child_frame_id = None
	loc.obs_id = 0
	loc.local_fail_count = 0
	loc.force_global_event = threading.Event()
	loc.curr_query_descs = []
	loc.main_freq = 50
	loc.curr_obs_node = None
	return loc


# --- 1. Regression: meters conversion + RGB order, process_frame must not raise ---

def test_process_frame_regression_meters_and_rgb_order(mod):
	args = _make_args()
	loc = _make_loc(args)

	rgb_msg = _make_rgb_msg(2, 2, pixel_bgr=(10, 20, 30))  # B=10 G=20 R=30
	depth_msg = _make_depth_msg(2, 2, mm_value=2500)  # -> 2.5 m
	cam_msg = _make_camera_info_msg(2, 2)

	captured = {}

	def desc_fn(rgb_img: np.ndarray) -> np.ndarray:
		captured["rgb"] = rgb_img.copy()
		return np.zeros(8, dtype=np.float32)

	real_depth_to_tensor = mod.depth_image_to_tensor

	def spy_depth_to_tensor(depth_img, depth_scale=1.0):
		captured["depth"] = depth_img.copy()
		return real_depth_to_tensor(depth_img, depth_scale)

	import pytest as _pytest  # local alias to avoid unused-import warnings in some linters
	_pytest  # no-op reference

	with _monkeypatch_attr(mod, "depth_image_to_tensor", spy_depth_to_tensor):
		mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=desc_fn)

	# Depth is in meters, not raw millimeters.
	np.testing.assert_allclose(captured["depth"], np.full((2, 2), 2.5, dtype=np.float32), atol=2e-3)
	# RGB channel order: bgr8 (10, 20, 30) must come out as RGB (30, 20, 10).
	np.testing.assert_array_equal(captured["rgb"][0, 0], [30, 20, 10])


class _monkeypatch_attr:
	"""Tiny context manager to patch-and-restore one module attribute (avoids pytest's
	function-scoped monkeypatch fixture inside a helper that already took `mod`)."""

	def __init__(self, obj, name, value):
		self._obj = obj
		self._name = name
		self._value = value

	def __enter__(self):
		self._original = getattr(self._obj, self._name)
		setattr(self._obj, self._name, self._value)

	def __exit__(self, *exc_info):
		setattr(self._obj, self._name, self._original)


# --- 2. perform_localization: thread must not exit on a per-frame exception ---

def test_perform_localization_survives_frame_exception(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.main_freq = 50

	mod.rgb_depth_queue.put((object(), object(), object()))
	mod.rgb_depth_queue.put((object(), object(), object()))

	calls = {"n": 0}

	def fake_process_frame(*_args, **_kwargs):
		calls["n"] += 1
		if calls["n"] == 1:
			raise RuntimeError("boom")

	with _monkeypatch_attr(mod, "process_frame", fake_process_frame):
		rate_mock = MagicMock()
		mod.rospy.Rate.return_value = rate_mock
		mod.rospy.is_shutdown = MagicMock(side_effect=[False, False, False, True])
		mod.rospy.logerr = MagicMock()

		mod.perform_localization(loc, args, desc_fn=None)

	assert calls["n"] == 2
	assert mod.rospy.logerr.call_count == 1
	# r.sleep() must be called every loop iteration, including empty-queue ones.
	assert rate_mock.sleep.call_count == 3


def test_perform_localization_exits_quietly_on_shutdown_and_survives_time_jump(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.main_freq = 50

	class ROSInterruptException(Exception):
		pass

	class ROSTimeMovedBackwardsException(ROSInterruptException):
		pass

	mod.rospy.exceptions.ROSInterruptException = ROSInterruptException
	mod.rospy.exceptions.ROSTimeMovedBackwardsException = ROSTimeMovedBackwardsException
	rate_mock = MagicMock()
	# Sim time jumps back once (keep going), then ROS shuts down (leave the loop, no traceback).
	rate_mock.sleep.side_effect = [ROSTimeMovedBackwardsException(), None, ROSInterruptException()]
	mod.rospy.Rate.return_value = rate_mock
	mod.rospy.is_shutdown = MagicMock(return_value=False)

	mod.perform_localization(loc, args, desc_fn=None)

	assert rate_mock.sleep.call_count == 3


# --- 3. local localization failing 3x in a row resets to global; a success in between resets the counter ---

def test_local_loc_fail_three_times_resets_to_global(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.has_global_pos = True
	loc.ref_map_node = SimpleNamespace(trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))
	mod.fused_poses.find_closest = MagicMock(return_value=(None, None))
	loc.perform_local_loc = MagicMock(return_value={"succ": False, "T_w_obs": None})

	rgb_msg = _make_rgb_msg(1, 1)
	depth_msg = _make_depth_msg(1, 1, mm_value=1000)
	cam_msg = _make_camera_info_msg(1, 1)

	for _ in range(3):
		mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=lambda img: np.zeros(4, dtype=np.float32))

	assert loc.has_global_pos is False
	assert loc.ref_map_node is None
	assert loc.local_fail_count == 0


def test_local_loc_fail_twice_then_succeed_resets_counter_without_global_reset(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.has_global_pos = True
	loc.ref_map_node = SimpleNamespace(trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))
	mod.fused_poses.find_closest = MagicMock(return_value=(None, None))
	loc.perform_local_loc = MagicMock(
		side_effect=[
			{"succ": False, "T_w_obs": None},
			{"succ": False, "T_w_obs": None},
			{"succ": True, "T_w_obs": np.eye(4)},
		]
	)

	rgb_msg = _make_rgb_msg(1, 1)
	depth_msg = _make_depth_msg(1, 1, mm_value=1000)
	cam_msg = _make_camera_info_msg(1, 1)

	for _ in range(3):
		mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=lambda img: np.zeros(4, dtype=np.float32))

	assert loc.local_fail_count == 0
	assert loc.has_global_pos is True
	assert loc.ref_map_node is not None


# --- 4. /vloc/force_global_loc callback sets the event; next frame goes back to global loc ---

def test_force_global_loc_callback_sets_event_and_triggers_global_relocalization(mod):
	args = _make_args()
	loc = _make_loc(args)

	callback = mod._build_force_global_loc_callback(loc)
	response = callback(None)

	assert loc.force_global_event.is_set()
	mod.TriggerResponse.assert_called_once_with(
		success=True, message="global localization will rerun on the next frame"
	)
	assert response is mod.TriggerResponse.return_value

	# Simulate a frame already globally+locally localized, which should be
	# discarded and re-run through global localization on the next frame.
	loc.has_global_pos = True
	loc.ref_map_node = SimpleNamespace(trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))
	loc.curr_query_descs = [np.zeros(4, dtype=np.float32)]
	loc.perform_global_loc = MagicMock(return_value={"succ": False, "map_id": None})

	rgb_msg = _make_rgb_msg(1, 1)
	depth_msg = _make_depth_msg(1, 1, mm_value=1000)
	cam_msg = _make_camera_info_msg(1, 1)
	mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=lambda img: np.zeros(4, dtype=np.float32))

	assert not loc.force_global_event.is_set()
	loc.perform_global_loc.assert_called_once()
	assert loc.curr_query_descs == []


# --- 5. has_local_pos bug fix: a reset-then-failed-global frame must not keep a stale True ---

def test_has_local_pos_cleared_when_global_reset_and_global_loc_fails(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.has_global_pos = False  # already reset (e.g. by the 3-fail rule above)
	loc.has_local_pos = True  # stale value from a previous successful frame
	loc.perform_global_loc = MagicMock(return_value={"succ": False, "map_id": None})

	rgb_msg = _make_rgb_msg(1, 1)
	depth_msg = _make_depth_msg(1, 1, mm_value=1000)
	cam_msg = _make_camera_info_msg(1, 1)
	mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=lambda img: np.zeros(4, dtype=np.float32))

	assert loc.has_local_pos is False


# --- 6. desc_fn hook: called with original-size RGB uint8, loc.vpr_model untouched; shape (1, D) ---

def test_desc_fn_hook_used_instead_of_vpr_model(mod):
	args = _make_args()
	loc = _make_loc(args)

	rgb_msg = _make_rgb_msg(3, 2, pixel_bgr=(1, 2, 3))
	depth_msg = _make_depth_msg(3, 2, mm_value=500)
	cam_msg = _make_camera_info_msg(3, 2)

	captured = {}

	def desc_fn(rgb_img: np.ndarray) -> np.ndarray:
		captured["shape"] = rgb_img.shape
		captured["dtype"] = rgb_img.dtype
		return np.arange(5, dtype=np.float32)

	mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=desc_fn)

	assert captured["shape"] == (2, 3, 3)  # HxWx3, original size (pre-resize)
	assert captured["dtype"] == np.uint8
	loc.vpr_model.assert_not_called()

	call_args = mod.ImageNode.call_args
	desc_arg = call_args[0][3]
	assert desc_arg.shape == (1, 5)


def test_no_desc_fn_falls_back_to_vpr_model(mod):
	args = _make_args()
	loc = _make_loc(args)

	rgb_msg = _make_rgb_msg(2, 2)
	depth_msg = _make_depth_msg(2, 2, mm_value=500)
	cam_msg = _make_camera_info_msg(2, 2)

	mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=None)

	loc.vpr_model.assert_called_once()

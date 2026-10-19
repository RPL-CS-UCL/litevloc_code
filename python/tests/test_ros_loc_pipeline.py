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
	"std_msgs",
	"std_msgs.msg",
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


def _make_args(map_path="/maps/old"):
	return SimpleNamespace(image_size=None, device="cpu", global_pos_threshold=10.0, map_path=map_path)


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
	loc.paused_event = threading.Event()
	loc.pending_map_path = None
	loc.map_path_event = threading.Event()
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
	paused_pub = MagicMock()

	callback = mod._build_force_global_loc_callback(loc, paused_pub)
	response = callback(None)

	assert loc.force_global_event.is_set()
	mod.TriggerResponse.assert_called_once_with(
		success=True, message="global localization will rerun on the next frame"
	)
	assert response is mod.TriggerResponse.return_value
	# Was not paused, so the pause status must not be touched or republished.
	assert not loc.paused_event.is_set()
	paused_pub.publish.assert_not_called()

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


# --- 7. /vloc/set_paused: pause ---

def test_set_paused_true_sets_event_and_publishes_once(mod):
	args = _make_args()
	loc = _make_loc(args)
	paused_pub = MagicMock()
	callback = mod._build_set_paused_callback(loc, paused_pub)

	response = callback(SimpleNamespace(data=True))

	assert loc.paused_event.is_set()
	paused_pub.publish.assert_called_once_with(mod.Bool(data=True))
	mod.SetBoolResponse.assert_called_with(success=True, message="now paused")
	assert response is mod.SetBoolResponse.return_value

	# Repeating the same request (already paused) must not republish, but must
	# still report success.
	paused_pub.publish.reset_mock()
	mod.SetBoolResponse.reset_mock()
	response2 = callback(SimpleNamespace(data=True))

	assert loc.paused_event.is_set()
	paused_pub.publish.assert_not_called()
	mod.SetBoolResponse.assert_called_once_with(success=True, message="already paused")
	assert response2 is mod.SetBoolResponse.return_value


# --- 8. /vloc/set_paused: resume ---

def test_set_paused_false_clears_event_keeps_local_state_and_publishes_once(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.paused_event.set()
	paused_pub = MagicMock()
	callback = mod._build_set_paused_callback(loc, paused_pub)

	response = callback(SimpleNamespace(data=False))

	assert not loc.paused_event.is_set()
	# resuming continues with local localization; no full global relocalization
	assert not loc.force_global_event.is_set()
	paused_pub.publish.assert_called_once_with(mod.Bool(data=False))
	mod.SetBoolResponse.assert_called_with(success=True, message="now running")
	assert response is mod.SetBoolResponse.return_value

	# Resuming again (already running) must not republish, and must still report success.
	paused_pub.publish.reset_mock()
	mod.SetBoolResponse.reset_mock()
	response2 = callback(SimpleNamespace(data=False))

	assert not loc.force_global_event.is_set()
	paused_pub.publish.assert_not_called()
	mod.SetBoolResponse.assert_called_once_with(success=True, message="already running")
	assert response2 is mod.SetBoolResponse.return_value


# --- 9. /vloc/force_global_loc while paused also resumes ---

def test_force_global_loc_while_paused_resumes_and_publishes(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.paused_event.set()
	paused_pub = MagicMock()
	callback = mod._build_force_global_loc_callback(loc, paused_pub)

	response = callback(None)

	assert loc.force_global_event.is_set()
	assert not loc.paused_event.is_set()
	paused_pub.publish.assert_called_once_with(mod.Bool(data=False))
	assert response is mod.TriggerResponse.return_value


# --- 10. Localization loop while paused: queue drained, process_frame not called; resumes cleanly ---

def test_run_one_cycle_paused_drops_frame_without_processing(mod):
	args = _make_args()
	loc = _make_loc(args)
	loc.paused_event.set()
	mod.rgb_depth_queue.put((object(), object(), object()))

	calls = {"n": 0}

	def fake_process_frame(*_args, **_kwargs):
		calls["n"] += 1

	with _monkeypatch_attr(mod, "process_frame", fake_process_frame):
		mod._run_one_cycle(loc, args, desc_fn=None)

	assert calls["n"] == 0
	assert mod.rgb_depth_queue.empty()


def test_run_one_cycle_resumes_and_processes_next_frame_normally(mod):
	args = _make_args()
	loc = _make_loc(args)
	# Was paused, now resumed (e.g. via set_paused): the queue has a fresh frame.
	loc.paused_event.clear()
	mod.rgb_depth_queue.put((object(), object(), object()))

	calls = {"n": 0}

	def fake_process_frame(*_args, **_kwargs):
		calls["n"] += 1

	with _monkeypatch_attr(mod, "process_frame", fake_process_frame):
		mod._run_one_cycle(loc, args, desc_fn=None)

	assert calls["n"] == 1
	assert mod.rgb_depth_queue.empty()


# --- 11. /vloc/map_path: same path as currently loaded -> ignored, no reload ---

def _make_map_switch_loc(args):
	"""_make_loc() plus the pieces _reload_map() touches (read_covis_graph_from_files,
	init_vpr_match_model, DB_Node_IDS)."""
	loc = _make_loc(args)
	loc.DB_Node_IDS = ["kf_000", "kf_001", "kf_002"]
	loc.read_covis_graph_from_files = MagicMock()
	loc.init_vpr_match_model = MagicMock()
	return loc


def test_map_path_callback_only_records_pending_path(mod):
	args = _make_args()
	loc = _make_map_switch_loc(args)
	callback = mod._build_map_path_callback(loc)

	callback(SimpleNamespace(data="/maps/new"))

	# Recorded for the localization thread to pick up; nothing reloaded yet.
	assert loc.pending_map_path == "/maps/new"
	assert loc.map_path_event.is_set()
	loc.read_covis_graph_from_files.assert_not_called()
	loc.init_vpr_match_model.assert_not_called()


def test_run_one_cycle_ignores_map_path_update_for_already_loaded_map(mod):
	args = _make_args(map_path="/maps/old")
	loc = _make_map_switch_loc(args)
	config = {"resize": None, "depth_scale": 1.0, "load_rgb": True, "load_depth": False, "normalized": False}
	map_loaded_pub = MagicMock()

	callback = mod._build_map_path_callback(loc)
	callback(SimpleNamespace(data="/maps/old"))  # same path as already loaded

	mod._run_one_cycle(loc, args, config, map_loaded_pub, desc_fn=None)

	loc.read_covis_graph_from_files.assert_not_called()
	loc.init_vpr_match_model.assert_not_called()
	map_loaded_pub.publish.assert_not_called()
	assert loc.args.map_path == "/maps/old"


# --- 12. /vloc/map_path: different path -> reloaded on the next cycle, state cleared ---

def test_run_one_cycle_reloads_map_on_path_change_and_resets_state(mod):
	args = _make_args(map_path="/maps/old")
	loc = _make_map_switch_loc(args)
	loc.has_global_pos = True
	loc.ref_map_node = SimpleNamespace(trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))
	loc.local_fail_count = 2
	loc.curr_query_descs = [np.zeros(4, dtype=np.float32)]
	config = {"resize": None, "depth_scale": 1.0, "load_rgb": True, "load_depth": False, "normalized": False}
	map_loaded_pub = MagicMock()
	# Queued frame against the old map must be dropped by the reload, not processed.
	mod.rgb_depth_queue.put((object(), object(), object()))

	callback = mod._build_map_path_callback(loc)
	callback(SimpleNamespace(data="/maps/new"))

	calls = {"n": 0}

	def fake_process_frame(*_args, **_kwargs):
		calls["n"] += 1

	with _monkeypatch_attr(mod, "process_frame", fake_process_frame):
		mod._run_one_cycle(loc, args, config, map_loaded_pub, desc_fn=None)

	loc.read_covis_graph_from_files.assert_called_once_with(config)
	loc.init_vpr_match_model.assert_called_once()
	assert loc.args.map_path == "/maps/new"
	assert loc.has_global_pos is False
	assert loc.ref_map_node is None
	assert loc.local_fail_count == 0
	assert loc.curr_query_descs == []
	assert mod.rgb_depth_queue.empty()
	map_loaded_pub.publish.assert_called_once_with(mod.String(data="/maps/new"))
	# The queued frame was dropped by the reload itself, not processed as a new frame.
	assert calls["n"] == 0
	assert not loc.map_path_event.is_set()


# --- 13. /vloc/map_path: reload failure keeps the old map and does not crash ---

def test_run_one_cycle_reload_failure_keeps_old_map(mod):
	args = _make_args(map_path="/maps/old")
	loc = _make_map_switch_loc(args)
	loc.read_covis_graph_from_files.side_effect = RuntimeError("corrupt map files")
	loc.has_global_pos = True
	loc.ref_map_node = SimpleNamespace(trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]))
	config = {"resize": None, "depth_scale": 1.0, "load_rgb": True, "load_depth": False, "normalized": False}
	map_loaded_pub = MagicMock()

	callback = mod._build_map_path_callback(loc)
	callback(SimpleNamespace(data="/maps/broken"))

	mod.rospy.logerr = MagicMock()
	mod._run_one_cycle(loc, args, config, map_loaded_pub, desc_fn=None)

	# Old map kept in use; localization state untouched by the failed reload.
	assert loc.args.map_path == "/maps/old"
	assert loc.has_global_pos is True
	assert loc.ref_map_node is not None
	loc.init_vpr_match_model.assert_not_called()
	map_loaded_pub.publish.assert_not_called()
	mod.rospy.logerr.assert_called_once()


def test_reload_failure_halfway_restores_whole_old_map(mod):
	"""Graph already replaced when the match model fails: every map-derived attribute goes back."""
	args = _make_args(map_path="/maps/old")
	loc = _make_map_switch_loc(args)
	loc.image_graph, loc.DB_Node_IDS, loc.vpr_match_model = "old_graph", [1, 2], "old_matcher"

	def half_load(_config):
		loc.image_graph, loc.DB_Node_IDS = "new_graph", [7]

	loc.read_covis_graph_from_files.side_effect = half_load
	loc.init_vpr_match_model.side_effect = RuntimeError("bad descriptors")
	config = {"resize": None, "depth_scale": 1.0, "load_rgb": True, "load_depth": False, "normalized": False}
	map_loaded_pub = MagicMock()
	mod._build_map_path_callback(loc)(SimpleNamespace(data="/maps/new"))
	mod.rospy.logerr = MagicMock()
	mod._run_one_cycle(loc, args, config, map_loaded_pub, desc_fn=None)

	assert loc.args.map_path == "/maps/old"
	assert (loc.image_graph, loc.DB_Node_IDS, loc.vpr_match_model) == ("old_graph", [1, 2], "old_matcher")
	map_loaded_pub.publish.assert_not_called()


# --- M5-B7 / B7c: /global_planner/active_nodes prior for global relocalization ---
#
# `current_candidate_subset()` / `active_nodes_callback()` are pure ros_loc_pipeline logic
# (no loc_pipeline involved), so the existing `mod` fixture (loc_pipeline mocked away) is
# enough for those. The actual verify-subset-then-fall-back-to-full-map behavior lives in
# `LocPipeline.perform_global_loc` itself, so those tests use `loc_pipeline_mod` below, which
# imports the *real* loc_pipeline module (only the ROS / optional message packages it never
# touches here are stubbed, same idea as the `mod` fixture's doc-string).

def test_active_nodes_callback_then_current_candidate_subset_fresh(mod):
	mod.time.time = lambda: 100.0
	mod.active_nodes_callback(SimpleNamespace(data=[3, 1, 4]))
	mod.time.time = lambda: 100.0 + mod.ACTIVE_NODES_MAX_AGE_S - 0.1  # 刚好还没过期
	assert mod.current_candidate_subset() == {1, 3, 4}


def test_current_candidate_subset_expired_returns_none(mod):
	mod.time.time = lambda: 100.0
	mod.active_nodes_callback(SimpleNamespace(data=[3, 1, 4]))
	mod.time.time = lambda: 100.0 + mod.ACTIVE_NODES_MAX_AGE_S + 0.1  # 超过 10 秒
	assert mod.current_candidate_subset() is None


def test_current_candidate_subset_none_when_never_received(mod):
	assert mod.current_candidate_subset() is None


def test_process_frame_passes_fresh_candidate_subset_to_perform_global_loc(mod):
	args = _make_args()
	loc = _make_loc(args)
	mod.time.time = lambda: 100.0
	mod.active_nodes_callback(SimpleNamespace(data=[5, 6]))

	rgb_msg = _make_rgb_msg(2, 2)
	depth_msg = _make_depth_msg(2, 2, mm_value=2500)
	cam_msg = _make_camera_info_msg(2, 2)
	mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=lambda img: np.zeros(4, dtype=np.float32))

	loc.perform_global_loc.assert_called_once_with(save_viz=False, candidate_subset={5, 6})


def test_process_frame_passes_none_when_active_nodes_expired(mod):
	args = _make_args()
	loc = _make_loc(args)
	mod.time.time = lambda: 100.0
	mod.active_nodes_callback(SimpleNamespace(data=[5, 6]))
	mod.time.time = lambda: 100.0 + mod.ACTIVE_NODES_MAX_AGE_S + 1.0  # 过期

	rgb_msg = _make_rgb_msg(2, 2)
	depth_msg = _make_depth_msg(2, 2, mm_value=2500)
	cam_msg = _make_camera_info_msg(2, 2)
	mod.process_frame(loc, args, rgb_msg, depth_msg, cam_msg, desc_fn=lambda img: np.zeros(4, dtype=np.float32))

	loc.perform_global_loc.assert_called_once_with(save_viz=False, candidate_subset=None)


# ---------- loc_pipeline.LocPipeline.perform_global_loc 本身的 candidate_subset 行为 ----------

_LOC_PIPELINE_ROS_STUBS = [
	"rospy", "std_msgs", "std_msgs.msg", "nav_msgs", "nav_msgs.msg",
	"sensor_msgs", "sensor_msgs.msg", "sensor_msgs.point_cloud2",
	"geometry_msgs", "geometry_msgs.msg", "visualization_msgs", "visualization_msgs.msg",
	"tf2_ros", "tf2_msgs", "tf2_msgs.msg", "cv_bridge", "gtsam", "tf", "tf.transformations",
]


@pytest.fixture()
def loc_pipeline_mod(monkeypatch):
	"""Import the *real* loc_pipeline module (not mocked away like in the `mod` fixture above),
	so LocPipeline.perform_global_loc's M5-B7 candidate_subset branch actually runs. Only the
	ROS / optional heavy message packages it transitively imports but never calls in these
	tests are stubbed out."""
	monkeypatch.syspath_prepend(str(LITEVLOC_PY))
	for name in _LOC_PIPELINE_ROS_STUBS:
		monkeypatch.setitem(sys.modules, name, MagicMock())
	monkeypatch.delitem(sys.modules, "loc_pipeline", raising=False)
	import loc_pipeline as module
	yield module
	monkeypatch.delitem(sys.modules, "loc_pipeline", raising=False)


class _FakeMatchModel:
	"""``match()`` just returns a fixed, already-ranked (best-first) list of DB indices,
	truncated to ``recall_values`` -- same contract as the real VPR match models (see
	``PlaceRecognitionSeqMatching._locate_best_match``: ``np.argsort(...)[:recall_values]``,
	a full sort then a slice, so asking for more values never changes the first few)."""

	def __init__(self, ranking, seq_len=1):
		self.seqLen = seq_len
		self._ranking = list(ranking)

	def match(self, query_descs, recall_values=1):
		preds = self._ranking[:recall_values]
		return preds, preds[0], 1.0


class _FakeImgMatcher:
	"""``num_inliers`` per DB node id, given up front; records which node ids it was asked
	to verify (so a test can assert "only the subset was checked")."""

	def __init__(self, inliers_by_node_id):
		self._inliers = inliers_by_node_id
		self.checked_node_ids = []

	def __call__(self, map_rgb_image, _obs_rgb_image):
		node_id = map_rgb_image.node_id
		self.checked_node_ids.append(node_id)
		return {"num_inliers": self._inliers.get(node_id, 0)}


def _make_real_loc_pipeline(loc_pipeline_mod, *, ranking, inliers_by_node_id, num_db=5):
	"""Bare ``LocPipeline`` (``__init__`` skipped: no log dir / rospy needed) with just the
	attributes ``perform_global_loc`` touches."""
	pipeline = loc_pipeline_mod.LocPipeline.__new__(loc_pipeline_mod.LocPipeline)
	pipeline.DB_Node_IDS = list(range(num_db))
	pipeline.DB_DESCRIPTORS = np.zeros((num_db, 4), dtype=np.float32)
	pipeline.args = SimpleNamespace(vpr_descriptors_dimension=4)
	pipeline.curr_obs_node = SimpleNamespace(
		get_descriptor=lambda: np.zeros((1, 4), dtype=np.float32),
		rgb_img_name="obs.jpg", id="obs", rgb_image="obs_rgb",
	)
	pipeline.curr_query_descs = []
	pipeline.vpr_match_model = _FakeMatchModel(ranking)
	nodes = {
		i: SimpleNamespace(rgb_img_name=f"kf_{i}.jpg", rgb_image=SimpleNamespace(node_id=i))
		for i in range(num_db)
	}
	pipeline.image_graph = SimpleNamespace(get_node=lambda node_id: nodes[node_id])
	pipeline.img_matcher = _FakeImgMatcher(inliers_by_node_id)
	return pipeline


def test_perform_global_loc_candidate_subset_verifies_only_the_subset(loc_pipeline_mod):
	"""有先验且子集里能过:只核验子集里的候选,不碰子集外的(即便它们排名更靠前)。"""
	# 全量重排后最像的是 kf2 / kf0 / kf1,但先验说大概率在 kf3 / kf4 附近,且 kf3 核验能过。
	pipeline = _make_real_loc_pipeline(
		loc_pipeline_mod, ranking=[2, 0, 1, 3, 4], inliers_by_node_id={2: 200, 3: 150}
	)
	result = pipeline.perform_global_loc(candidate_subset={3, 4})
	assert result == {"succ": True, "map_id": 3}
	assert sorted(pipeline.img_matcher.checked_node_ids) == [3, 4]  # kf0/kf1/kf2 没被核验过


def test_perform_global_loc_candidate_subset_falls_back_to_full_map_when_it_fails(loc_pipeline_mod):
	"""子集里一个都核验不过:退回全图做一次原流程,照样能找到正确的那个。"""
	pipeline = _make_real_loc_pipeline(
		loc_pipeline_mod, ranking=[2, 0, 1, 3, 4], inliers_by_node_id={2: 200}
	)
	result = pipeline.perform_global_loc(candidate_subset={0, 1})  # 子集里两个都核验不过(0 分)
	assert result == {"succ": True, "map_id": 2}
	# 子集先核验过(0、1),退回全图后把原本 Top-5(2、0、1、3、4)按原流程又核验一遍。
	assert pipeline.img_matcher.checked_node_ids.count(2) == 1
	assert set(pipeline.img_matcher.checked_node_ids) >= {0, 1, 2}


def test_perform_global_loc_candidate_subset_none_keeps_original_flow(loc_pipeline_mod):
	"""candidate_subset 不给(默认 None):行为跟加这个参数之前完全一样,只核验 Top-5。"""
	pipeline = _make_real_loc_pipeline(
		loc_pipeline_mod, ranking=[2, 0, 1, 3, 4], inliers_by_node_id={2: 200}
	)
	result = pipeline.perform_global_loc()
	assert result == {"succ": True, "map_id": 2}
	assert sorted(pipeline.img_matcher.checked_node_ids) == [0, 1, 2, 3, 4]

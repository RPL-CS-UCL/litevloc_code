"""Unit tests for ros_pose_fusion.py with real gtsam, no ROS master.

The callbacks are driven directly with real nav_msgs/Odometry messages; ROS time is set by hand
(the same mechanism rospy uses for /clock), publishers are replaced by recorders.
Synthetic run: the camera drives along a curve at 15 Hz; the local odometry has a 5 % scale drift
and lives in a different origin than the map; the global (LiteVLoc-like) poses are exact map
poses of every 15th frame, delivered one second late.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

gtsam = pytest.importorskip("gtsam")
rospy = pytest.importorskip("rospy")
from geometry_msgs.msg import TransformStamped  # noqa: E402
from nav_msgs.msg import Odometry, Path as PathMsg  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

LITEVLOC_PY = Path(__file__).resolve().parents[1]
RATE_HZ = 15.0
GLOBAL_EVERY = 15  # one global pose per second
GLOBAL_DELAY = 15  # delivered one second (15 local frames) later
CAMERA = "camera"


class _Recorder:
	def __init__(self, connections=1):
		self.msgs = []
		self.connections = connections

	def get_num_connections(self):
		return self.connections

	def publish(self, msg):
		self.msgs.append(msg)


class _IdentityTf:
	def lookup_transform(self, target, source, _stamp):
		tf = TransformStamped()
		tf.header.frame_id, tf.child_frame_id = target, source
		tf.transform.rotation.w = 1.0
		return tf


def _matrix(x, y, yaw_deg):
	T = np.eye(4)
	T[:3, :3] = Rotation.from_euler("z", yaw_deg, degrees=True).as_matrix()
	T[:3, 3] = [x, y, 0.5]
	return T


def _odom(T, stamp, frame_id):
	msg = Odometry()
	msg.header.stamp = rospy.Time.from_sec(stamp)
	msg.header.frame_id = frame_id
	msg.child_frame_id = CAMERA
	q = Rotation.from_matrix(T[:3, :3]).as_quat()
	p = msg.pose.pose
	p.position.x, p.position.y, p.position.z = T[:3, 3]
	p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = q
	return msg


def _pose_of(msg):
	p = msg.pose.pose
	T = np.eye(4)
	T[:3, :3] = Rotation.from_quat(
		[p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w]).as_matrix()
	T[:3, 3] = [p.position.x, p.position.y, p.position.z]
	return T


def _trajectory(n, t0, origin):
	"""Ground-truth camera poses in the map and the drifting local odometry of the same frames."""
	stamps = t0 + np.arange(n) / RATE_HZ
	gt, local = [], []
	T_local_map = np.linalg.inv(origin)
	for i in range(n):
		s = i / RATE_HZ
		T_map_cam = _matrix(0.4 * s, 2.0 * np.sin(0.05 * s), 10.0 * np.sin(0.03 * s))
		drifted = T_map_cam.copy()
		drifted[:3, 3] = origin[:3, 3] + 1.05 * (T_map_cam[:3, 3] - origin[:3, 3])
		gt.append(T_map_cam)
		local.append(T_local_map @ drifted)
	return stamps, gt, local


@pytest.fixture()
def fusion(monkeypatch):
	monkeypatch.syspath_prepend(str(LITEVLOC_PY))
	monkeypatch.setattr(sys, "argv", ["ros_pose_fusion.py"])
	monkeypatch.delitem(sys.modules, "ros_pose_fusion", raising=False)
	import ros_pose_fusion as mod
	from pose_fusion import PoseFusion

	rospy.rostime.set_rostime_initialized(True)
	pf = PoseFusion(SimpleNamespace(isam_params=False))
	pf.pub_odom, pf.pub_path, pf.pub_path_opt = _Recorder(), _Recorder(), _Recorder()
	pf.path_msg, pf.path_msg_opt = PathMsg(), PathMsg()
	mod.pose_fusion = pf
	mod.tf_buffer = _IdentityTf()
	mod.frame_id_map = "nav_map"
	return mod


def _run(mod, stamps, gt, local, start=0):
	"""Feed local frames [start, len) and the delayed global poses; return published messages."""
	pub = mod.pose_fusion.pub_odom
	for i in range(start, len(stamps)):
		rospy.rostime._set_rostime(rospy.Time.from_sec(stamps[i]))
		j = i - GLOBAL_DELAY
		if j >= start and (j - start) % GLOBAL_EVERY == 0:
			mod.odom_global_callback(_odom(gt[j], stamps[j], "nav_map"))
		mod.odom_local_callback_locked(_odom(local[i], stamps[i], "world"))
	return pub.msgs


def _trans_err(msg, T_gt):
	return float(np.linalg.norm(_pose_of(msg)[:3, 3] - T_gt[:3, 3]))


@pytest.mark.parametrize("yaw_deg", [40.0, 170.0])
def test_fusion_follows_map_and_beats_drifting_odometry(fusion, yaw_deg):
	# 170 deg between the local and the map frame: identity initial values would put the first output
	# metres off; the seeded initial values keep it at the centimetre level
	origin = _matrix(3.0, -2.0, yaw_deg)
	stamps, gt, local = _trajectory(int(60 * RATE_HZ), 1000.0, origin)
	msgs = _run(fusion, stamps, gt, local)

	# the first global pose predates every buffered local pose (local poses are only kept once the
	# global frame id is known) and is dropped; from the second one on, one output per local frame
	assert len(msgs) == len(stamps) - GLOBAL_DELAY - GLOBAL_EVERY
	by_stamp = {round(m.header.stamp.to_sec(), 6): m for m in msgs}
	assert all(m.header.frame_id == "nav_map" and m.child_frame_id == CAMERA for m in msgs)
	errs = [_trans_err(by_stamp[round(s, 6)], g) for s, g in zip(stamps, gt) if round(s, 6) in by_stamp]
	# dead reckoning from the first global pose with 5 % scale drift: ~1.2 m off after 24 m
	dead_reckoning = gt[0] @ np.linalg.inv(local[0]) @ local[-1]
	assert np.linalg.norm(dead_reckoning[:3, 3] - gt[-1][:3, 3]) > 1.0
	assert errs[0] < 0.05
	assert max(errs) < 0.15


def test_reset_clears_graph_and_drops_globals_from_before_reset(fusion):
	origin = _matrix(3.0, -2.0, 40.0)
	stamps, gt, local = _trajectory(int(40 * RATE_HZ), 1000.0, origin)
	half = len(stamps) // 2
	_run(fusion, stamps[:half], gt[:half], local[:half])
	assert fusion.init_system

	reset_at = stamps[half]
	rospy.rostime._set_rostime(rospy.Time.from_sec(reset_at))
	resp = fusion.reset_callback(None)
	assert resp.success and not fusion.init_system and len(fusion.poses_local) == 0
	n_before = len(fusion.pose_fusion.pub_odom.msgs)

	# new task: the local odometry origin jumps to the current camera pose
	new_origin = gt[half]
	new_local = [np.linalg.inv(new_origin) @ g for g in gt]
	# a global pose computed from an image taken before the reset must be ignored
	fusion.odom_global_callback(_odom(gt[half - 3], stamps[half - 3], "nav_map"))
	for i in range(half, half + 5):
		rospy.rostime._set_rostime(rospy.Time.from_sec(stamps[i]))
		fusion.odom_local_callback_locked(_odom(new_local[i], stamps[i], "world"))
	assert len(fusion.pose_fusion.pub_odom.msgs) == n_before
	assert not fusion.init_system

	msgs = _run(fusion, stamps, gt, new_local, start=half + 5)
	new_msgs = msgs[n_before:]
	assert len(new_msgs) == len(stamps) - (half + 5) - GLOBAL_DELAY
	gt_by_stamp = {round(s, 6): g for s, g in zip(stamps, gt)}
	assert max(_trans_err(m, gt_by_stamp[round(m.header.stamp.to_sec(), 6)]) for m in new_msgs) < 0.05


def test_path_keeps_only_recent_window(fusion, monkeypatch):
	# /pose_fusion/path is resent whole on every publish; it keeps only the last PATH_KEEP_S seconds
	monkeypatch.setattr(fusion, "PATH_KEEP_S", 10.0)
	origin = _matrix(3.0, -2.0, 40.0)
	stamps, gt, local = _trajectory(int(30 * RATE_HZ), 1000.0, origin)
	_run(fusion, stamps, gt, local)
	poses = fusion.pose_fusion.path_msg.poses
	first, last = poses[0].header.stamp.to_sec(), poses[-1].header.stamp.to_sec()
	assert last == pytest.approx(stamps[-1])
	assert last - 10.0 <= first < last - 10.0 + 1.5 / RATE_HZ
	assert len(poses) == pytest.approx(10.0 * RATE_HZ, abs=2)
	assert fusion.pose_fusion.pub_path.msgs[-1].poses[0].header.stamp.to_sec() >= last - 10.0 - 1.0

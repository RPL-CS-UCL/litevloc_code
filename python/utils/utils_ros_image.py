"""Decode ROS sensor_msgs Image / CompressedImage payloads into numpy arrays.

Only numpy and cv2 are used here (no rospy / cv_bridge / torch), so these
functions can be unit tested with plain fake message objects (e.g.
``types.SimpleNamespace``) and do not require a ROS environment.

This replaces ``cv_bridge.imgmsg_to_cv2(..., "passthrough")``, which returns a
read-only view into ``msg.data`` -- mutating it in place (as the old pipeline
did with ``depth_img *= 0.001``) raises, and it only recognizes ``mono16``
depth images, not the ``16UC1`` encoding our cameras publish.
"""
from typing import Any

import cv2
import numpy as np


def decode_depth_msg(msg: Any) -> np.ndarray:
	"""Decode a depth Image message into a new, writable, float32 meters array.

	``16UC1`` / ``mono16`` raw values are millimeters (scaled by 0.001);
	``32FC1`` values are already meters (NaN / inf are mapped to 0).
	"""
	encoding = msg.encoding
	endian = ">" if msg.is_bigendian else "<"
	width, height, step = int(msg.width), int(msg.height), int(msg.step)

	if encoding in ("16UC1", "mono16"):
		dtype = np.dtype(endian + "u2")
		itemsize = 2
		scale = np.float32(0.001)
	elif encoding == "32FC1":
		dtype = np.dtype(endian + "f4")
		itemsize = 4
		scale = np.float32(1.0)
	else:
		raise ValueError(f"Unsupported depth image encoding: {encoding}")

	row_items = step // itemsize
	raw = np.frombuffer(msg.data, dtype=dtype).reshape(height, row_items)[:, :width]
	depth = raw.astype(np.float32) * scale  # astype always copies -> new, writable array

	if encoding == "32FC1":
		depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

	return depth


def filter_depth_range(depth: np.ndarray, min_m: float, max_m: float) -> np.ndarray:
	"""Zero out depth values outside of [min_m, max_m], in place, and return it."""
	depth[(depth < min_m) | (depth > max_m)] = 0.0
	return depth


def decode_rgb_msg(msg: Any) -> np.ndarray:
	"""Decode a raw or compressed color image message into a new, writable HxWx3 uint8 RGB array."""
	if hasattr(msg, "encoding"):
		return _decode_raw_rgb_msg(msg)
	return _decode_compressed_rgb_msg(msg)


def _decode_raw_rgb_msg(msg: Any) -> np.ndarray:
	encoding = msg.encoding
	width, height, step = int(msg.width), int(msg.height), int(msg.step)

	if encoding == "rgb8":
		channels, bgr = 3, False
	elif encoding == "bgr8":
		channels, bgr = 3, True
	elif encoding == "rgba8":
		channels, bgr = 4, False
	elif encoding == "bgra8":
		channels, bgr = 4, True
	else:
		raise ValueError(f"Unsupported rgb image encoding: {encoding}")

	raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(height, step)[:, : width * channels]
	img = raw.reshape(height, width, channels)[:, :, :3]  # drop alpha if present
	if bgr:
		img = img[:, :, ::-1]
	return np.array(img, dtype=np.uint8)  # new, writable array


def _decode_compressed_rgb_msg(msg: Any) -> np.ndarray:
	buf = np.frombuffer(msg.data, dtype=np.uint8)
	bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
	if bgr is None:
		fmt = getattr(msg, "format", "?")
		raise ValueError(f"Failed to decode compressed image with format: {fmt}")
	return np.array(bgr[:, :, ::-1], dtype=np.uint8)  # BGR -> RGB, new writable array

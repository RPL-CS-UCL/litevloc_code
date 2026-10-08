"""Unit tests for utils.utils_ros_image: pure numpy/cv2 decoding of ROS image messages.

Fake messages are plain types.SimpleNamespace objects; `data` is always a bytes
object (read-only), exactly like a real ROS message, so these tests also guard
against the original cv_bridge bug (read-only array that cannot be scaled
in place, and that only recognized mono16).
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

LITEVLOC_PY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LITEVLOC_PY))

from utils.utils_ros_image import decode_depth_msg, decode_rgb_msg, filter_depth_range  # noqa: E402


def _depth_msg(values, encoding, step=None, is_bigendian=False, itemsize=2, dtype_char="u2"):
	values = np.asarray(values)
	height, width = values.shape
	endian = ">" if is_bigendian else "<"
	row_dtype = np.dtype(endian + dtype_char)
	row_items = step // itemsize if step is not None else width
	padded = np.zeros((height, row_items), dtype=row_dtype)
	padded[:, :width] = values.astype(row_dtype)
	data = padded.tobytes()
	return types.SimpleNamespace(
		encoding=encoding,
		width=width,
		height=height,
		step=row_items * itemsize,
		is_bigendian=is_bigendian,
		data=data,
	)


def test_decode_depth_16uc1_little_endian():
	msg = _depth_msg([[1000, 2500], [0, 65535]], "16UC1")
	depth = decode_depth_msg(msg)
	np.testing.assert_allclose(depth, [[1.0, 2.5], [0.0, 65.535]], rtol=0, atol=2e-3)
	assert depth.dtype == np.float32


def test_decode_depth_mono16_same_as_16uc1():
	msg = _depth_msg([[1000, 2500], [0, 65535]], "mono16")
	depth = decode_depth_msg(msg)
	np.testing.assert_allclose(depth, [[1.0, 2.5], [0.0, 65.535]], rtol=0, atol=2e-3)


def test_decode_depth_returns_writable_copy_not_tied_to_msg_data():
	msg = _depth_msg([[1000, 2500], [0, 65535]], "16UC1")
	original_data = msg.data
	depth = decode_depth_msg(msg)
	depth[0, 0] = 99.0  # must not raise, and must not touch msg.data
	assert depth[0, 0] == 99.0
	assert msg.data == original_data


def test_decode_depth_row_padding():
	# step covers 3 u16 items per row but only the first 2 columns are real.
	msg = _depth_msg([[1000, 2500], [500, 1500]], "16UC1", step=6)
	depth = decode_depth_msg(msg)
	np.testing.assert_allclose(depth, [[1.0, 2.5], [0.5, 1.5]], rtol=0, atol=2e-3)


def test_decode_depth_big_endian():
	msg = _depth_msg([[1000, 2500], [0, 65535]], "16UC1", is_bigendian=True)
	depth = decode_depth_msg(msg)
	np.testing.assert_allclose(depth, [[1.0, 2.5], [0.0, 65.535]], rtol=0, atol=2e-3)


def test_decode_depth_32fc1_nan_and_inf_to_zero():
	msg = _depth_msg(
		[[1.5, np.nan], [np.inf, -np.inf]], "32FC1", itemsize=4, dtype_char="f4"
	)
	depth = decode_depth_msg(msg)
	np.testing.assert_allclose(depth, [[1.5, 0.0], [0.0, 0.0]], rtol=0, atol=2e-3)


def test_decode_depth_unknown_encoding_raises():
	msg = _depth_msg([[1, 2]], "unknown_encoding")
	with pytest.raises(ValueError, match="unknown_encoding"):
		decode_depth_msg(msg)


def test_filter_depth_range():
	depth = np.array([[0.05, 1.0], [10.0, 20.0]], dtype=np.float32)
	out = filter_depth_range(depth, 0.1, 15.0)
	np.testing.assert_allclose(out, [[0.0, 1.0], [10.0, 0.0]])
	assert out is depth  # modifies and returns the same array


def _rgb_msg(rgb_hwc: np.ndarray, encoding: str, step=None, alpha=255):
	height, width = rgb_hwc.shape[:2]
	if encoding == "rgb8":
		raw = rgb_hwc
	elif encoding == "bgr8":
		raw = rgb_hwc[:, :, ::-1]
	elif encoding in ("rgba8", "bgra8"):
		alpha_chan = np.full((height, width, 1), alpha, dtype=np.uint8)
		base = rgb_hwc if encoding == "rgba8" else rgb_hwc[:, :, ::-1]
		raw = np.concatenate([base, alpha_chan], axis=2)
	else:
		raise AssertionError(f"unsupported test encoding {encoding}")

	channels = raw.shape[2]
	row_items = step if step is not None else width * channels
	padded = np.zeros((height, row_items), dtype=np.uint8)
	padded[:, : width * channels] = raw.reshape(height, width * channels)
	return types.SimpleNamespace(
		encoding=encoding,
		width=width,
		height=height,
		step=row_items,
		data=padded.tobytes(),
	)


def test_decode_rgb_rgb8_passthrough():
	rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
	msg = _rgb_msg(rgb, "rgb8")
	out = decode_rgb_msg(msg)
	np.testing.assert_array_equal(out, rgb)
	assert out.dtype == np.uint8


def test_decode_rgb_bgr8_swaps_channels():
	rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
	msg = _rgb_msg(rgb, "bgr8")
	out = decode_rgb_msg(msg)
	np.testing.assert_array_equal(out, rgb)


def test_decode_rgb_bgr8_with_row_padding():
	rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
	msg = _rgb_msg(rgb, "bgr8", step=2 * 3 + 5)  # 5 bytes of row padding
	out = decode_rgb_msg(msg)
	np.testing.assert_array_equal(out, rgb)


def test_decode_rgb_bgra8_drops_alpha_and_swaps():
	rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
	msg = _rgb_msg(rgb, "bgra8")
	out = decode_rgb_msg(msg)
	np.testing.assert_array_equal(out, rgb)
	assert out.shape == (1, 2, 3)


def test_decode_rgb_writable_copy():
	rgb = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
	msg = _rgb_msg(rgb, "rgb8")
	out = decode_rgb_msg(msg)
	out[0, 0, 0] = 255  # must not raise
	assert out[0, 0, 0] == 255


def test_decode_rgb_unknown_encoding_raises():
	rgb = np.zeros((1, 2, 3), dtype=np.uint8)
	msg = _rgb_msg(rgb, "rgb8")
	msg.encoding = "weird_encoding"
	with pytest.raises(ValueError, match="weird_encoding"):
		decode_rgb_msg(msg)


def test_decode_rgb_compressed_png_roundtrip():
	bgr = np.zeros((4, 4, 3), dtype=np.uint8)
	bgr[:, :, 0] = 10  # B
	bgr[:, :, 1] = 20  # G
	bgr[:, :, 2] = 30  # R
	ok, buf = cv2.imencode(".png", bgr)
	assert ok
	msg = types.SimpleNamespace(format="png", data=buf.tobytes())
	out = decode_rgb_msg(msg)
	assert out.shape == (4, 4, 3)
	np.testing.assert_array_equal(out[0, 0], [30, 20, 10])  # RGB order


def test_decode_rgb_compressed_bad_data_raises():
	msg = types.SimpleNamespace(format="png", data=b"not a real image")
	with pytest.raises(ValueError):
		decode_rgb_msg(msg)

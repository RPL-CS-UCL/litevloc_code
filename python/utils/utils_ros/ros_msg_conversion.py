#!/usr/bin/env python

import numpy as np
import sensor_msgs.point_cloud2 as pc2

from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, CompressedImage, PointCloud2
from tf2_msgs.msg import TFMessage
from visualization_msgs.msg import Marker
from scipy.spatial.transform import Rotation

##### Sensor data
def convert_cvimg_to_rosimg(image, encoding, header, compressed=False):
	bridge = CvBridge()
	image_16UC1 = image.astype(np.uint16)[:, :, 0] if encoding == 'mono16' else image
	if not compressed:
		img_msg = bridge.cv2_to_imgmsg(image_16UC1, encoding=encoding, header=header)
	else:
		img_msg = bridge.cv2_to_compressed_imgmsg(image_16UC1)
		img_msg.header = header
	return img_msg

def convert_rosimg_to_cvimg(img_msg):
	bridge = CvBridge()
	if isinstance(img_msg, CompressedImage):
		compressed = True
	else:
		compressed = False
	if compressed:
		cv_image = bridge.compressed_imgmsg_to_cv2(img_msg, "rgb8")
	else:
		cv_image = bridge.imgmsg_to_cv2(img_msg, "passthrough")
		if img_msg.encoding == "32FC1":
			cv_image = np.nan_to_num(cv_image, nan=0.0)
	return cv_image

def convert_pts_to_rospts(header, pts, intensity=None, color=None, label=None):
	msg = PointCloud2()
	msg.header = header
	# Define the fields
	fields = [
		pc2.PointField(name='x', offset=0, datatype=pc2.PointField.FLOAT32, count=1),
		pc2.PointField(name='y', offset=4, datatype=pc2.PointField.FLOAT32, count=1),
		pc2.PointField(name='z', offset=8, datatype=pc2.PointField.FLOAT32, count=1),
		pc2.PointField(name='intensity', offset=12, datatype=pc2.PointField.FLOAT32, count=1),
		pc2.PointField(name='rgb', offset=16, datatype=pc2.PointField.UINT32, count=1),
		pc2.PointField(name='label', offset=20, datatype=pc2.PointField.FLOAT32, count=1)
	]
	msg.fields.extend(fields)
	# Prepare points array
	dtype = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("intensity", "f4"), ("rgba", "u4"), ("label", "f4")])
	pointsColor = np.zeros(pts.shape[0], dtype=dtype)
	pointsColor["x"], pointsColor["y"], pointsColor["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
	if intensity is not None:
		pointsColor["intensity"] = intensity
	if color is not None:
		pointsColor["rgba"] = color.view('uint32')
	if label is not None:
		pointsColor["label"] = label
	msg.data = pointsColor.tobytes()
	msg.point_step = 24
	msg.height = 1
	msg.width = pts.shape[0]
	msg.row_step = msg.point_step * msg.width
	msg.is_bigendian = False
	return msg

def convert_rospts_to_pts(msg):
	"""Convert PointCloud2 message to numpy array."""
	import ros_numpy
	pc_array = ros_numpy.numpify(msg)
	pc = np.zeros([len(pc_array), 3])
	pc[:, 0] = pc_array['x']
	pc[:, 1] = pc_array['y']
	pc[:, 2] = pc_array['z']
	return pc

##### Odometry
def convert_rosodom_to_vec(odom, mode='xyzw'):
	if mode == 'xyzw':
		trans = np.array([
			odom.pose.pose.position.x, 
			odom.pose.pose.position.y, 
			odom.pose.pose.position.z]
		)
		quat = np.array([
			odom.pose.pose.orientation.x, 
			odom.pose.pose.orientation.y, 
			odom.pose.pose.orientation.z, 
			odom.pose.pose.orientation.w]
		)
	elif mode == 'wxyz':
		trans = np.array([
			odom.pose.pose.position.x, 
			odom.pose.pose.position.y, 
			odom.pose.pose.position.z]
		)
		quat = np.array([
			odom.pose.pose.orientation.w, 
			odom.pose.pose.orientation.x, 
			odom.pose.pose.orientation.y, 
			odom.pose.pose.orientation.z]
		)
	return trans, quat

def convert_rosodom_to_matrix(odom):
	trans = np.array([
		odom.pose.pose.position.x, 
		odom.pose.pose.position.y, 
		odom.pose.pose.position.z]
	)
	quat = np.array([
		odom.pose.pose.orientation.x, 
		odom.pose.pose.orientation.y, 
		odom.pose.pose.orientation.z, 
		odom.pose.pose.orientation.w]
	)
	T = np.eye(4)
	T[:3, :3] = Rotation.from_quat(quat).as_matrix()
	T[:3, 3] = trans
	return T

def convert_vec_to_rosodom(trans, quat, header, child_frame_id, mode='xyzw'):
	if mode == 'xyzw':
		odom = convert_vec_to_rosodom_scale(
			trans[0], trans[1], trans[2], 
			quat[0], quat[1], quat[2], quat[3], 
			header, child_frame_id
		)
	elif mode == 'wxyz':
		odom = convert_vec_to_rosodom_scale(
			trans[0], trans[1], trans[2], 
			quat[3], quat[0], quat[1], quat[2], 
			header, child_frame_id
		)
	return odom

def convert_vec_to_rosodom_scale(tx, ty, tz, qx, qy, qz, qw, header, child_frame_id):
	odom = Odometry()
	odom.header = header
	odom.child_frame_id = child_frame_id
	odom.pose.pose.position.x = tx
	odom.pose.pose.position.y = ty
	odom.pose.pose.position.z = tz
	odom.pose.pose.orientation.x = qx
	odom.pose.pose.orientation.y = qy
	odom.pose.pose.orientation.z = qz
	odom.pose.pose.orientation.w = qw
	return odom

def convert_vec_to_rospose(trans, quat, header, mode='xyzw'):
	if mode == 'xyzw':
		pose = convert_vec_to_rospose_scale(trans, quat, header)
	elif mode == 'wxyz':
		pose = convert_vec_to_rospose_scale(trans, np.roll(quat, 1), header)
	return pose

def convert_vec_to_rospose_scale(trans, quat, header):
	pose = PoseStamped()
	pose.header = header
	pose.pose.position.x = trans[0]
	pose.pose.position.y = trans[1]
	pose.pose.position.z = trans[2]
	pose.pose.orientation.x = quat[0]
	pose.pose.orientation.y = quat[1]
	pose.pose.orientation.z = quat[2]
	pose.pose.orientation.w = quat[3]
	return pose

def convert_vec_to_ros_tfmsg(trans, quat, header, child_frame_id, mode='xyzw'):
	tf_msg = TFMessage()
	tf_data = convert_vec_to_rostf(trans, quat, header, child_frame_id, mode)
	tf_msg.transforms.append(tf_data)
	return tf_msg

def convert_vec_to_rostf(trans, quat, header, child_frame_id, mode='xyzw'):
	if mode == 'xyzw':
		tf_msg = convert_vec_to_rostf_scale(
			trans[0], trans[1], trans[2], 
			quat[0], quat[1], quat[2], quat[3], 
			header, child_frame_id)
	elif mode == 'wxyz':
		tf_msg = convert_vec_to_rostf_scale(
			trans[0], trans[1], trans[2], 
			quat[3], quat[0], quat[1], quat[2], 
			header, child_frame_id)
	return tf_msg

def convert_vec_to_rostf_scale(tx, ty, tz, qx, qy, qz, qw, header, child_frame_id):
	tf_msg = TransformStamped()
	tf_msg.header = header
	tf_msg.child_frame_id = child_frame_id
	tf_msg.transform.translation.x = tx
	tf_msg.transform.translation.y = ty
	tf_msg.transform.translation.z = tz
	tf_msg.transform.rotation.x = qx
	tf_msg.transform.rotation.y = qy
	tf_msg.transform.rotation.z = qz
	tf_msg.transform.rotation.w = qw
	return tf_msg

def convert_rostf_to_vec(tf_msg, mode='xyzw'):
	if mode == 'xyzw':
		trans = np.array([
			tf_msg.transform.translation.x, 
			tf_msg.transform.translation.y, 
			tf_msg.transform.translation.z]
		)
		quat = np.array([
			tf_msg.transform.rotation.x, 
			tf_msg.transform.rotation.y, 
			tf_msg.transform.rotation.z, 
			tf_msg.transform.rotation.w]
		)
	else:
		trans = np.array([
			tf_msg.transform.translation.x, 
			tf_msg.transform.translation.y, 
			tf_msg.transform.translation.z]
		)
		quat = np.array([
			tf_msg.transform.rotation.w, 
			tf_msg.transform.rotation.x, 
			tf_msg.transform.rotation.y, 
			tf_msg.transform.rotation.z]
		)
	return trans, quat

def convert_rostf_to_matrix(tf_msg):
	trans = np.array([
		tf_msg.transform.translation.x, 
		tf_msg.transform.translation.y, 
		tf_msg.transform.translation.z]
	)
	quat = np.array([
		tf_msg.transform.rotation.x, 
		tf_msg.transform.rotation.y, 
		tf_msg.transform.rotation.z, 
		tf_msg.transform.rotation.w]
	)
	T = np.eye(4)
	T[:3, :3] = Rotation.from_quat(quat).as_matrix()
	T[:3, 3] = trans
	return T

def convert_odom_to_rospose(odom):
	pose = PoseStamped()
	pose.header = odom.header
	pose.pose = odom.pose.pose
	return pose

def convert_odom_to_rostf(odom):
	tf_msg = TransformStamped()
	tf_msg.header = odom.header
	tf_msg.child_frame_id = odom.child_frame_id
	tf_msg.transform.translation.x = odom.pose.pose.position.x
	tf_msg.transform.translation.y = odom.pose.pose.position.y
	tf_msg.transform.translation.z = odom.pose.pose.position.z
	tf_msg.transform.rotation.x = odom.pose.pose.orientation.x
	tf_msg.transform.rotation.y = odom.pose.pose.orientation.y
	tf_msg.transform.rotation.z = odom.pose.pose.orientation.z
	tf_msg.transform.rotation.w = odom.pose.pose.orientation.w
	return tf_msg		

##### Visualization message
def get_ros_marker_camera_frustum(header, position, orientation, length=10.0):
	marker = Marker()
	marker.header = header
	marker.ns = "frustum"
	marker.type = Marker.LINE_LIST
	marker.action = Marker.ADD
	marker.id = 0
	marker.pose.position.x = position[0]
	marker.pose.position.y = position[1]
	marker.pose.position.z = position[2]
	marker.pose.orientation.x = orientation[0]
	marker.pose.orientation.y = orientation[1]
	marker.pose.orientation.z = orientation[2]
	marker.pose.orientation.w = orientation[3]
	marker.scale.x = 0.25  # width
	marker.color.r = 1.0
	marker.color.g = 0.0
	marker.color.b = 0.0
	marker.color.a = 1.0
	# Define frustum points
	points = [
			[-length/2, -length/2, length/2], [length/2, -length/2, length/2],
			[-length/2, -length/2, length/2], [-length/2, length/2, length/2],
			[length/2, length/2, length/2], [-length/2, length/2, length/2],
			[length/2, length/2, length/2], [length/2, -length/2, length/2],
			[-length/2, -length/2, length/2], [0.0, 0.0, 0.0],
			[-length/2, -length/2, length/2], [0.0, 0.0, 0.0],
			[length/2, length/2, length/2], [0.0, 0.0, 0.0],
			[length/2, length/2, length/2], [0.0, 0.0, 0.0]
	]
	marker.points = [Point(x=p[0], y=p[1], z=p[2]) for p in points]
	return marker

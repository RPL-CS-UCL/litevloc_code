import rospy
import queue
import gtsam
import numpy as np
import threading
from std_msgs.msg import Header
from nav_msgs.msg import Odometry

from pose_fusion import parse_arguments, PoseFusion

from pycpptools.src.python.utils_algorithm.stamped_poses import StampedPoses
from pycpptools.src.python.utils_math.tools_eigen import convert_vec_gtsam_pose3
from pycpptools.src.python.utils_ros import ros_msg

lock_odom_global = threading.Lock()

odom_global_queue = queue.Queue()
poses_local = StampedPoses()

curr_stamped_pose = (0, gtsam.Pose3())
marginal_cov = np.eye(6, 6)
init_system = False

def odom_local_callback(odom_msg):
	global odom_global_queue, poses_local
	global curr_stamped_pose, marginal_cov, init_system

	"""Process local odometry"""
	curr_time = odom_msg.header.stamp.to_sec()
	trans, quat = ros_msg.convert_rosodom_to_vec(odom_msg)
	curr_pose_local = convert_vec_gtsam_pose3(trans, quat)
	curr_idx = len(poses_local)
	# Add odometry factor
	if len(poses_local) > 0:
		prev_idx = len(poses_local) - 1
		_, stamped_prev_pose_local = poses_local.get_item(prev_idx)
		sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
		pose_fusion.add_odometry_factor(prev_idx, stamped_prev_pose_local[1], curr_idx, curr_pose_local, sigma)
		# Update the current pose
		curr_stamped_pose = (curr_time, 
							 curr_stamped_pose[1] * stamped_prev_pose_local[1].between(curr_pose_local))
	poses_local.add(curr_time, curr_pose_local)

	# the odometry is aligned with the global frame
	if init_system:
		pose_fusion.add_init_estimate(curr_idx, curr_stamped_pose[1])
	# the odometry is not aligned with the global frame
	else:
		pose_fusion.add_init_estimate(curr_idx, gtsam.Pose3())

	"""Process global odometry"""
	if not odom_global_queue.empty():
		while not odom_global_queue.empty():
			# Skip the localization odometry that is newer than the current odometry
			# cannot determine the id of variable
			if odom_msg.header.stamp < odom_global_queue.queue[0].header.stamp: return
			lock_odom_global.acquire()
			odom_global_msg = odom_global_queue.get()
			lock_odom_global.release()
			pose_time = odom_global_msg.header.stamp.to_sec()
			trans, quat = ros_msg.convert_rosodom_to_vec(odom_global_msg)
			pose3 = convert_vec_gtsam_pose3(trans, quat)
			idx_closest, _ = poses_local.find_closest(pose_time)
			# print(f"Closest pose to global odometry at time {pose_time:.05f} is {idx_closest}")

			# Add prior factor
			sigma = np.array([np.deg2rad(5.), np.deg2rad(5.), np.deg2rad(5.), 0.1, 0.1, 0.1])
			pose_fusion.add_prior_factor(idx_closest, pose3, sigma)
		
		# Perform the isam2 optimization 
		pose_fusion.perform_optimization()
		# Update the current pose after optimization
		curr_stamped_pose = (curr_time, pose_fusion.current_estimate.atPose3(curr_idx))
		marginal_cov = pose_fusion.get_margin_covariance(curr_idx)
		if not init_system:
			init_system = True
			print("System initialized")

	if init_system:
		# Print the current pose
		# Publish the odometry
		trans = curr_stamped_pose[1].translation()
		quat = curr_stamped_pose[1].rotation().toQuaternion().coeffs() # xyzw
		print(f"Current pose at time {curr_time}: {trans}")
		header = Header(stamp=rospy.Time.from_sec(curr_stamped_pose[0]), frame_id='map')
		fusion_odom = ros_msg.convert_vec_to_rosodom(trans, quat, header, 'camera')
		fusion_odom.pose.covariance = list(marginal_cov.flatten())
		pose_fusion.pub_odom.publish(fusion_odom)
		# Publish the path
		pose_msg = ros_msg.convert_odom_to_rospose(fusion_odom)
		pose_fusion.path_msg.header = header
		pose_fusion.path_msg.poses.append(pose_msg)
		pose_fusion.pub_path.publish(pose_fusion.path_msg)

def odom_global_callback(odom_msg):
	print("----------------- Global odometry callback")
	lock_odom_global.acquire()
	odom_global_queue.put(odom_msg)
	while odom_global_queue.qsize() > 100: odom_global_queue.get()
	lock_odom_global.release()

if __name__ == '__main__':
	args = parse_arguments()

	rospy.init_node('ros_pose_fusion', anonymous=True)
	pose_fusion = PoseFusion(args)
	pose_fusion.setup_ros_objects()
	
	# Subscribe different sources of odometry
	# 1. local odometry
	# 2, global odometry
	odom_local = rospy.Subscriber('/depth_reg/odometry', Odometry, odom_local_callback)
	odom_global = rospy.Subscriber('/vloc/odometry', Odometry, odom_global_callback)

	rospy.spin()
import rospy
import tf
from nav_msgs.msg import Odometry
from pose_fusion import parse_arugment, PoseFusion
import threading

from pycpptools.src.python.utils_algorithm.stamped_poses import StampedPoses
from pycpptools.src.python.utils_math.tools_eigen import convert_vec_gtsam_pose3
from pycpptools.src.python.utils_ros import ros_msg

lock_odom_local = threading.Lock()
lock_odom_local = threading.Lock()

odom_local_queue = queue.Queue()
odom_global_queue = queue.Queue()

poses_local = StampedPoses()

curr_stamped_pose = (0, gtsam.Pose3())
init_system = False

def odom_local_callback(odom_msg):
	if odom_msg.header.stamp < odom_global_queue.queue[0].header.stamp: return

	"""Process local odometry"""
	curr_time = odom_msg.header.stamp.to_sec()
	trans, quat = ros_msg.convert_rosodom_to_vec(odom_msg)
	curr_pose_local = convert_vec_gtsam_pose3(trans, quat)
	curr_idx = len(poses_local)
	# Add odometry factor
	if len(poses_local) > 0:
		prev_idx = len(poses_local) - 1
		prev_pose_local = poses_local.get_item(prev_idx)
		sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
		pose_fusion.add_odometry_factor(prev_idx, prev_pose_local, curr_idx, curr_pose_local, sigma)
		# Indicate that the system has received the VLOC
		if init_system:
			curr_stamped_pose = (curr_time, 
								 curr_stamped_pose * prev_pose_local.between(curr_pose_local))
			pose_fusion.add_init_estimate(curr_idx, curr_stamped_pose[1])
	poses_local.add_pose(curr_time, curr_pose_local)

	"""Process global odometry"""
	if not odom_global_queue.empty():
		while not odom_global_queue.empty():
			# Skip the localization odometry that is newer than the current odometry
			if odom_msg.header.stamp < odom_global_queue.queue[0].header.stamp: return

			lock_odom_global.acquire()
			odom_global_msg = odom_global_queue.get()
			lock_odom_global.release()
			pose_time = odom_global_msg.header.stamp.to_sec()
			trans, quat = ros_msg.convert_rosodom_to_vec(odom_global_msg)
			pose3 = convert_vec_gtsam_pose(trans, quat)
			idx_closest, pose_closest = poses_local.find_closest(pose_time)
			# Add prior factor
			sigma = np.array([np.deg2rad(1.), np.deg2rad(1.), np.deg2rad(1.), 0.01, 0.01, 0.01])
			pose_fusion.add_prior_factor(idx_closest, pose_closest, sigma)
			pose_fusion.add_init_estimate(idx_closest, pose_closest)
		pose_fusion.perform_optimization()
		curr_stamped_pose = (curr_time, pose_fusion.current_estimate.atPose3(curr_idx))
		init_system = True

	if init_system:
		# Print the current pose
		print(f"Current pose at time {curr_time}: {curr_stamped_pose[1].translation()}")
		# Publish the odometry
		trans = curr_stamped_pose[1].translation()
		quat = curr_stamped_pose[1].rotation().quaternion()
		header = Header()
		header.stamp = rospy.Time.from_sec(curr_stamped_pose[0])
		header.frame = 'map'
		fusion_odom = ros_msg.convert_vec_to_rosodom(trans, quat, header, 'camera')
		pose_fusion.pub_odom.publish(fusion_odom)
		# Publish the path
		pose_msg = ros_msg.convert_odom_to_rospose(fusion_odom)
		pose_fusion.path_msg.header = header
		pose_fusion.path_msg.poses.append(pose_msg)
		pose_fusion.pub_path.publish(path_msg)

def odom_global_callback(odom_msg):
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
	odom_local = rospy.Subscriber('/vloc/odometry', Odometry, odom_local_callback)
	odom_global = rospy.Subscriber('/depth_reg/odometry', Odometry, odom_global_callback)

	# Start the fusion thread
	fusion_thread = threading.Thread(target=perform_pose_fusion, args=(pose_fusion, args, ))
	fusion_thread.start()

	rospy.spin()
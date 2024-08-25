#! /Rocket_ssd/miniconda3/envs/topo_loc/bin/python

"""
Usage: 
python loc_pipeline.py \
--dataset_path /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_17DRP5sb8fy/out_map \
--image_size 288 512 --device=cuda \
--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=512 --save_descriptors --num_preds_to_save 3 \
--img_matcher master --save_img_matcher \
--pose_solver pnp --config_pose_solver config/dataset/matterport3d.yaml \
--viz

Usage: 
rosbag record -O /Titan/dataset/data_topo_loc/anymal_lab_upstair_20240722_0/vloc.bag \
/vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs
"""

import os
import sys
import pathlib
import numpy as np
import torch
import time
import cv2
import rospy
from std_msgs.msg import Header
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseArray
from visualization_msgs.msg import MarkerArray
import tf2_ros

import rospkg
rospkg = rospkg.RosPack()
pack_path = rospkg.get_path('topo_loc')
sys.path.append(os.path.join(pack_path, '../image_matching_models'))

from matching.utils import to_numpy
from utils.utils_vpr_method import initialize_vpr_model, perform_knn_search
from utils.utils_vpr_method import save_visualization as save_vpr_visualization
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image import load_rgb_image, load_depth_image
from utils.utils_pipeline import *
from utils.pose_solver import get_solver
from utils.pose_solver_default import cfg
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode

import pycpptools.src.python.utils_math as pytool_math
import pycpptools.src.python.utils_ros as pytool_ros
import pycpptools.src.python.utils_sensor as pytool_sensor

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

class LocPipeline:
	def __init__(self, args, log_dir):
		self.args = args
		self.log_dir = log_dir
		self.has_global_pos = False
		self.frame_id_map = 'map'

	def init_vpr_model(self):
		self.vpr_model = initialize_vpr_model(self.args.vpr_method, self.args.vpr_backbone, self.args.vpr_descriptors_dimension, self.args.device)
		logging.info(f"VPR model: {self.args.vpr_method}")

	def init_img_matcher(self):
		self.img_matcher = initialize_img_matcher(self.args.img_matcher, self.args.device, self.args.n_kpts)
		self.img_matcher_lighter = initialize_img_matcher('eloftr', self.args.device, self.args.n_kpts)
		logging.info(f"Image matcher: {self.args.img_matcher}")
		logging.info(f"Image matcher (lighter): eloftr")
		
	def init_pose_solver(self):
		cfg.merge_from_file(self.args.config_pose_solver)
		self.pose_solver = get_solver(self.args.pose_solver, cfg)
		logging.info(f"Pose solver: {self.args.pose_solver}")

	def initalize_ros(self):
		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)
		
		self.pub_odom = rospy.Publisher('/vloc/odometry', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/vloc/path', Path, queue_size=10)
		self.pub_path_gt = rospy.Publisher('/vloc/path_gt', Path, queue_size=10)
		self.pub_map_obs = rospy.Publisher('/vloc/image_map_obs', Image, queue_size=10)

		self.br = tf2_ros.TransformBroadcaster()
		self.path_msg = Path()
		self.path_gt_msg = Path()

	def read_map_from_file(self):
		data_path = self.args.dataset_path
		self.image_graph = GraphLoader.load_data(
			data_path,
			self.args.image_size,
			depth_scale=self.args.depth_scale,
			normalized=False
		)
		logging.info(f"Loaded {self.image_graph} from {data_path}")

		# Extract VPR descriptors for all nodes in the map
		self.DB_DESCRIPTORS_ID = np.array(self.image_graph.get_all_id())
		self.DB_DESCRIPTORS = np.array([map_node.get_descriptor() for _, map_node in self.image_graph.nodes.items()], dtype="float32")
		print(f"IDs: {self.DB_DESCRIPTORS_ID} extracted {self.DB_DESCRIPTORS.shape} VPR descriptors.")
		self.DB_POSES = np.empty((self.image_graph.get_num_node(), 7), dtype="float32")
		for indices, (_, map_node) in enumerate(self.image_graph.nodes.items()):
			self.DB_POSES[indices, :3] = map_node.trans
			self.DB_POSES[indices, 3:] = map_node.quat

	def perform_vpr(self, db_descs, query_desc):
		query_desc_arr = np.empty((1, self.args.vpr_descriptors_dimension), dtype="float32")
		query_desc_arr[0] = query_desc
		dis, pred = perform_knn_search(
			db_descs,
			query_desc_arr,
			self.args.vpr_descriptors_dimension,
			self.args.recall_values
		)
		return dis, pred

	def perform_image_matching(self, matcher, map_node, obs_node):
		try:
			matcher_result = matcher(map_node.rgb_image, obs_node.rgb_image)

			"""Save matching results"""
			if self.args.save_img_matcher:
				num_inliers, H, mkpts0, mkpts1 = (
					matcher_result["num_inliers"],
					matcher_result["H"],
					matcher_result["inliers0"],
					matcher_result["inliers1"],
				)
				# out_str = f"Paths: map_id ({map_node.id}), obs_id ({obs_node.id}). "
				# out_str += f"Found {num_inliers} inliers after RANSAC. "
				save_img_matcher_visualization(
					obs_node.rgb_image, map_node.rgb_image,
					mkpts0, mkpts1, self.log_dir, obs_node.id, n_viz=100)
				
			return matcher_result
		except Exception as e:
			logging.error(f"Error in image matching: {e}")
		return None

	# Search potential keyframes using the covisiblity graph
	def search_keyframe_from_graph(self, obs_node):
		query_pose = obs_node.trans.reshape(1, 3)
		dis, pred = perform_knn_search(self.DB_POSES[:, :3], query_pose, 3, [1])
		if len(pred[0]) == 0 or dis[0][0] > self.args.global_pos_threshold: return None
		closest_map_node = self.image_graph.get_node(self.DB_DESCRIPTORS_ID[pred[0][0]])
		all_nodes, all_dis = [nei_node for nei_node, _ in closest_map_node.edges] + [closest_map_node], []
		alpha = 0.3
		for node in all_nodes:
			dis_trans, dis_angle = self.curr_obs_node.compute_distance(node)
			dis = alpha * min(dis_trans / 5.0, 1.0) + (1 - alpha) * min(dis_angle / 360.0, 1.0)
			all_dis.append(dis)
		sorted_nodes = [all_nodes[i] for i in np.argsort(all_dis)]

		start_time = time.time()
		all_num_inliers = []
		for node in sorted_nodes:
			matcher_result = self.perform_image_matching(self.img_matcher_lighter, node, obs_node)
			all_num_inliers.append(matcher_result["num_inliers"])
			if time.time() - start_time > 0.3: break
		out_str = ' '.join([f'{node.id}: {num_inliers}' for node, num_inliers in zip(sorted_nodes, all_num_inliers)])
		node_max_inliers = sorted_nodes[np.argmax(all_num_inliers)]
		return node_max_inliers

	def perform_global_loc(self, save=False):
		vpr_dis, vpr_pred = self.perform_vpr(self.DB_DESCRIPTORS, self.curr_obs_node.get_descriptor())
		vpr_dis, vpr_pred = vpr_dis[0, :], vpr_pred[0, :]
		if len(vpr_pred) == 0:
			print('No start node found, cannot determine the global position.')
			return {'succ': False, 'map_id': None}
		# Save VPR visualization for the top-k predictions
		if save:
			list_of_images_paths = [self.curr_obs_node.rgb_img_path]
			for i in range(len(vpr_pred[:self.args.num_preds_to_save])):
				map_node = self.image_graph.get_node(self.DB_DESCRIPTORS_ID[vpr_pred[i]])
				list_of_images_paths.append(map_node.rgb_img_path)
			preds_correct = [None] * len(list_of_images_paths)
			save_vpr_visualization(self.log_dir, 0, list_of_images_paths, preds_correct)
		return {'succ': True, 'map_id': self.DB_DESCRIPTORS_ID[vpr_pred[0]]}
	
	def perform_local_loc(self):
		search_start_time = time.time()
		ref_map_node = self.search_keyframe_from_graph(self.curr_obs_node)
		if ref_map_node is None: return {'succ': False, 'T_w_obs': None, 'solver_inliers': 0}
		print(f"Search keyframe costs: {time.time() - search_start_time:.3f}s, Found the reference map node: {ref_map_node.id}")
		
		matching_start_time = time.time()
		self.ref_map_node = ref_map_node
		matcher_result = self.perform_image_matching(self.img_matcher, self.ref_map_node, self.curr_obs_node)
		print(f"Image matching costs: {time.time() - matching_start_time: .3f}s")

		if matcher_result is None or matcher_result["num_inliers"] < self.args.min_inliers_threshold:
			return {'succ': False, 'T_w_obs': None, 'solver_inliers': 0}
		try:
			print(f'Number of inliers: {matcher_result["num_inliers"]}')
			T_mapnode_obs = None
			mkpts0, mkpts1 = (matcher_result["inliers0"], matcher_result["inliers1"])
			mkpts0_raw = mkpts0 * [self.ref_map_node.raw_img_size[0] / self.ref_map_node.img_size[0], 
								   self.ref_map_node.raw_img_size[1] / self.ref_map_node.img_size[1]]
			mkpts1_raw = mkpts1 * [self.curr_obs_node.raw_img_size[0] / self.curr_obs_node.img_size[0], 
								   self.curr_obs_node.raw_img_size[1] / self.curr_obs_node.img_size[1]]
			if self.args.img_matcher == "mickey":
				inliers = matcher_result["num_inliers"]
				R, t = self.img_matcher.scene["R"].squeeze(0), self.img_matcher.scene["t"].squeeze(0)
				R, t = to_numpy(R), to_numpy(t)
				T_mapnode_obs = np.eye(4)
				T_mapnode_obs[:3, :3], T_mapnode_obs[:3, 3] = R, t
				print(f'Mickey Solver:\n', T_mapnode_obs)
			else:
				depth_img0 = to_numpy(self.curr_obs_node.depth_image.squeeze(0))
				R, t, inliers = self.pose_solver.estimate_pose(
					mkpts1_raw, mkpts0_raw,
					self.curr_obs_node.raw_K, self.ref_map_node.raw_K,
					depth_img0, None)
				T_mapnode_obs = np.eye(4)
				T_mapnode_obs[:3, :3], T_mapnode_obs[:3, 3] = R, t.reshape(3)
				print(f'{self.args.pose_solver}: Number of inliers: {inliers}')

			if T_mapnode_obs is not None:
				T_w_mapnode = pytool_math.tools_eigen.convert_vec_to_matrix(
					self.ref_map_node.trans_gt, self.ref_map_node.quat_gt, 'xyzw')
				T_w_obs = T_w_mapnode @ T_mapnode_obs
				self.ref_map_node.set_matched_kpts(mkpts0, inliers)
				self.curr_obs_node.set_matched_kpts(mkpts1, inliers)
				return {'succ': True, 'T_w_obs': T_w_obs, 'solver_inliers': inliers}
		except Exception as e:
			print(f'Failed to estimate pose with {self.args.pose_solver}:', e)
			return {'succ': False, 'T_w_obs': None, 'solver_inliers': inliers}

	def publish_message(self):
		header = Header(stamp=rospy.Time.now(), frame_id=self.frame_id_map)
		tf_msg = pytool_ros.ros_msg.convert_vec_to_rostf(np.array([0, 0, -2.0]), np.array([0, 0, 0, 1]), header, f"{self.frame_id_map}_graph")
		self.br.sendTransform(tf_msg)
		header = Header(stamp=rospy.Time.now(), frame_id=f"{self.frame_id_map}_graph")
		pytool_ros.ros_vis.publish_graph(self.image_graph, header, self.pub_graph, self.pub_graph_poses)

		if self.curr_obs_node is not None:
			header = Header(stamp=rospy.Time.from_sec(self.curr_obs_node.time), frame_id=self.frame_id_map)
			
			odom = pytool_ros.ros_msg.convert_vec_to_rosodom(self.curr_obs_node.trans, self.curr_obs_node.quat, header, self.child_frame_id)
			self.pub_odom.publish(odom)

			pose_msg = pytool_ros.ros_msg.convert_odom_to_rospose(odom)
			self.path_msg.header = header
			self.path_msg.poses.append(pose_msg)
			self.pub_path.publish(self.path_msg)

			if self.curr_obs_node.has_pose_gt:
				pose_msg = pytool_ros.ros_msg.convert_vec_to_rospose(self.curr_obs_node.trans_gt, self.curr_obs_node.quat_gt, header)
				self.path_gt_msg.header = header
				self.path_gt_msg.poses.append(pose_msg)
				self.pub_path_gt.publish(self.path_gt_msg)

			if self.ref_map_node is not None and self.args.viz:
				n_viz = 10 # visualize n_viz matched keypoints
				rgb_img_ref = (np.transpose(to_numpy(self.ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				rgb_img_obs = (np.transpose(to_numpy(self.curr_obs_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				mkpts_map, num_inliers = self.ref_map_node.get_matched_kpts()
				mkpts_obs, _ = self.curr_obs_node.get_matched_kpts()
				step_size = max(1, len(mkpts_map) // n_viz)
				rgb_img_ref_bgr = cv2.cvtColor(rgb_img_ref, cv2.COLOR_RGB2BGR)
				rgb_img_obs_bgr = cv2.cvtColor(rgb_img_obs, cv2.COLOR_RGB2BGR)
				merged_img = np.hstack((rgb_img_ref_bgr, rgb_img_obs_bgr))
				for i in range(0, len(mkpts_map), step_size):
					x0, y0 = mkpts_map[i]
					x1, y1 = mkpts_obs[i]
					cv2.circle(rgb_img_ref_bgr, (int(x0), int(y0)), 3, (0, 255, 0), -1)
					cv2.circle(rgb_img_obs_bgr, (int(x1), int(y1)), 3, (0, 255, 0), -1)
					cv2.line(merged_img, (int(x0), int(y0)), (int(x1) + rgb_img_ref.shape[1], int(y1)), (0, 255, 0), 2)	
				text = f'Matched inliers kpts: {num_inliers}'
				text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
				text_x = (merged_img.shape[1] - text_size[0])
				text_y = (merged_img.shape[0] - text_size[1])
				cv2.putText(merged_img, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3, cv2.LINE_AA)
				img_msg = pytool_ros.ros_msg.convert_cvimg_to_rosimg(merged_img, "bgr8", header, compressed=False)
				self.pub_map_obs.publish(img_msg)

def perform_localization(loc: LocPipeline, args):
	"""Main loop for processing observations"""
	obs_poses_gt = np.loadtxt(os.path.join(args.dataset_path, '../out_general', 'poses.txt'))
	obs_cam_intrinsics = np.loadtxt(os.path.join(args.dataset_path, '../out_general', 'intrinsics.txt'))
	resize = args.image_size
	loc.last_obs_node = None

	for obs_id in range(0, len(obs_poses_gt), 30):
		if rospy.is_shutdown(): break
		print(f"Loading observation with id {obs_id}")

		rgb_img_path = os.path.join(args.dataset_path, '../out_general/seq', f'{obs_id:06d}.color.jpg')
		rgb_img = load_rgb_image(rgb_img_path, resize, normalized=False)

		depth_img_path = os.path.join(args.dataset_path, '../out_general/seq', f'{obs_id:06d}.depth.png')
		depth_img = load_depth_image(depth_img_path, depth_scale=args.depth_scale)

		raw_K = np.array([obs_cam_intrinsics[obs_id, 0], 0, obs_cam_intrinsics[obs_id, 2], 0, 
						  obs_cam_intrinsics[obs_id, 1], obs_cam_intrinsics[obs_id, 3], 
						  0, 0, 1], dtype=np.float32).reshape(3, 3)
		raw_img_size = (int(obs_cam_intrinsics[obs_id, 4]), int(obs_cam_intrinsics[obs_id, 5])) # width, height
		K = pytool_sensor.utils.correct_intrinsic_scale(raw_K, resize[0] / raw_img_size[0], resize[1] / raw_img_size[1]) if resize is not None else raw_K
		img_size = (int(resize[0]), int(resize[1])) if resize is not None else raw_img_size
		# Create observation node
		obs_node = ImageNode(obs_id, rgb_img, depth_img, None,
							 rospy.Time.now().to_sec(),
							 np.zeros(3), np.array([0, 0, 0, 1]),
							 K, img_size,
							 rgb_img_path, depth_img_path)
		obs_node.set_raw_intrinsics(raw_K, raw_img_size)
		obs_node.set_pose_gt(obs_poses_gt[obs_id, 1:4], obs_poses_gt[obs_id, 4:])
		loc.curr_obs_node = obs_node

		"""Perform global localization via. visual place recognition"""
		if not loc.has_global_pos:
			loc_start_time = time.time()
			if loc.curr_obs_node.get_descriptor() is None:
				with torch.no_grad():
					desc = loc.vpr_model(loc.curr_obs_node.rgb_image.unsqueeze(0).to(args.device)).cpu().numpy()
				loc.curr_obs_node.set_descriptor(desc)
			result = loc.perform_global_loc(save=(args.num_preds_to_save!=0))
			print(f"Global localization costs: {time.time() - loc_start_time:.3f}s")
			if result['succ']:
				matched_map_id = result['map_id']
				loc.has_global_pos = True
				loc.ref_map_node = loc.image_graph.get_node(matched_map_id)
				loc.curr_obs_node.set_pose(loc.ref_map_node.trans, loc.ref_map_node.quat)
			else:
				print('Failed to determine the global position.')
				continue
		else:
			init_trans, init_quat = loc.last_obs_node.trans, loc.last_obs_node.quat
			loc.curr_obs_node.set_pose(init_trans, init_quat)

		"""Perform local localization via. image matching"""
		if loc.has_global_pos:
			loc_start_time = time.time()
			result = loc.perform_local_loc()
			print(f"Local localization costs: {time.time() - loc_start_time:.3f}s")
			if result['succ']:
				T_w_obs = result['T_w_obs']
				trans, quat = pytool_math.tools_eigen.convert_matrix_to_vec(T_w_obs, 'xyzw')
				loc.curr_obs_node.set_pose(trans, quat)
				print(f'Groundtruth Poses: {loc.curr_obs_node.trans_gt.T}')
				print(f'Estimated Poses: {trans.T}\n')
			else:
				print('Failed to determine the local position.')
				continue

		loc.publish_message()
		# Set as the initial guess of the next observation
		loc.last_obs_node = loc.curr_obs_node
		time.sleep(0.01)
		input("Press Enter to continue...")

if __name__ == '__main__':
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.dataset_path, 'output_loc_pipeline'))
	out_dir.mkdir(exist_ok=True, parents=True)
	log_dir = setup_log_environment(out_dir, args)

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, log_dir)
	print('Initialize VPR Model')
	loc_pipeline.init_vpr_model()
	print('Initialize Image Matcher')
	loc_pipeline.init_img_matcher()
	print('Initialize Pose Solver')
	loc_pipeline.init_pose_solver()
	loc_pipeline.read_map_from_file()

	rospy.init_node('loc_pipeline_node', anonymous=True)
	loc_pipeline.initalize_ros()
	loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'map')
	loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_localization(loc_pipeline, args)

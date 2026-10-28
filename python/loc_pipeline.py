#! /usr/bin/env python

"""
Usage:
python python/loc_pipeline.py \
	--map_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--query_data_path /Rocket_ssd/dataset/data_litevloc/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
	--device=cuda \
	--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 \
	--img_matcher edm \
	--pose_solver pnp --config_pose_solver python/config/dataset/matterport3d.yaml \
	--global_pos_threshold 10.0 \
	--save_img_matcher --viz --num_preds_to_save 3
# image_size / n_kpts / 核验门槛(gv_min_inliers(_nodepth))按 --img_matcher 从
# config/matcher_profiles.yaml 读(M5-B4);不给就用那份配置,给了就以命令行为准。

Usage: 
rosbag record -O /Titan/dataset/data_litevloc/anymal_lab_upstair_20240722_0/vloc.bag \
	/vloc/odometry /vloc/path /vloc/path_gt /vloc/image_map_obs
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pathlib
import numpy as np
import time
import cv2
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import matplotlib
# M5-B4b:ROS 相关 import(rospy / tf2_ros / std_msgs 等消息类型 / utils.utils_ros)从模块顶部挪进
# 真正用到它们的方法里(用到时才 import)——这样 `import loc_pipeline` 在没有 ROS 的 PYTHONPATH 下
# (run_vloc_offline_rerun.py 的运行环境,见 utils/geom_verify.py 里 verify_match_result 的文档
# 字符串)也能成功;在线节点(ros_loc_pipeline.py)该导的还是照样导,只是导入时机从"模块加载时"
# 改成"方法被调用时"(Python 的 import 本身有缓存,重复 import 不是重新加载)。

from utils.utils_pipeline import parse_arguments, setup_log_environment
from utils.utils_geom import convert_vec_to_matrix, convert_matrix_to_vec, compute_pose_error, convert_pose_inv
from utils.utils_vpr_method import initialize_vpr_model, initialize_match_model, perform_knn_search, compute_euclidean_dis
from utils.utils_vpr_method import save_visualization as save_vpr_visualization
from utils.utils_image_matching_method import initialize_img_matcher
from utils.utils_image_matching_method import save_visualization as save_img_matcher_visualization
from utils.utils_image import load_rgb_image, load_depth_image, to_numpy
from utils.pose_solver import get_solver
from utils import geom_verify
from rpe_default import cfg
from image_graph import ImageGraphLoader as GraphLoader
from image_node import ImageNode


@dataclass
class LocResult:
	"""`LocPipeline.localize_obs()` 的返回值(M5-B4:在线节点 `process_frame`、`loc_pipeline.py`
	自带的无 ROS 测试循环 `perform_localization`、离线重跑 `run_vloc_offline_rerun.py` 原来各自抄
	一份"核验候选 + 解位姿"的逻辑,现在只有这一份——`run_vloc_offline_rerun.py` 不能 import 这个
	模块(见 utils/geom_verify.py 的 verify_match_result 文档字符串),改成直接调 geom_verify 里
	同一份核验函数,跟这里用的是同一个实现,不是又抄了一份)。

	succ: 这一轮定位成功与否(succ=True 时 map_id/T_world_obs 才有意义;mode="global" 且查询没
		深度时,succ=True 只表示"认出了是哪个关键帧",T_world_obs 仍是 None)。
	stage: "global" / "local" / "none"(没找到/没解出来)。
	map_id: 命中的地图关键帧 id(DB_Node_IDS 里的值),没命中是 None。
	T_world_obs: 解出的查询相机在世界(地图)系下的位姿(4x4),没深度或解不出来时是 None。
	verify_inliers: 统一核验分数(网格去重后的 PnP-RANSAC 内点数,没深度时是基础矩阵内点数)。
	n_matches_run: 实际跑了几次图像匹配(global 模式下早停时 < 候选数)。
	t_vpr_s / t_verify_s / t_local_s: 各段耗时(秒);不适用的段落留 0.0。
	"""

	succ: bool
	stage: str
	map_id: Optional[int]
	T_world_obs: Optional[np.ndarray]
	verify_inliers: int
	n_matches_run: int
	t_vpr_s: float
	t_verify_s: float
	t_local_s: float

# This is to be able to use matplotlib also without a GUI
if not hasattr(sys, "ps1"):	matplotlib.use("Agg")

class LocPipeline:
	def __init__(self, args, out_dir):
		self.args = args

		out_dir.mkdir(exist_ok=True, parents=True)
		self.log_dir = setup_log_environment(out_dir, args)
	
		self.has_global_pos = False
		self.has_local_pos = False
		self.frame_id_map = 'map'
		self.depth_range = (0.1, 15.0)

		# Pipeline
		self.vpr_model = None
		self.vpr_match_model = None
		self.img_matcher = None
		self.pose_solver = None

		# Variable
		self.ref_map_node = None
		
		# ROS Publish Rate
		self.publish_graph = False

	def init_vpr_model(self):
		import rospy
		self.vpr_model = initialize_vpr_model(self.args.vpr_method, self.args.vpr_backbone, self.args.vpr_descriptors_dimension, self.args.device)
		rospy.loginfo(f"Initialize VPR model: {self.args.vpr_method}")

	def init_vpr_match_model(self):
		import rospy
		self.vpr_match_model = initialize_match_model(
			self.args.vpr_match_model,
			self.args.vpr_match_seq_len
		)
		self.vpr_match_model.initialize_model(self.DB_DESCRIPTORS)
		self.curr_query_descs = []
		rospy.loginfo(f"Initialize VPR Match Model: {self.args.vpr_match_model}")

	def init_img_matcher(self):
		import rospy
		# M5-B4:按 --img_matcher 从 config/matcher_profiles.yaml 查到的额外构造参数(比如 master
		# 系列的 min_conf_thr、edm 的 ransac_model)在构造时就传给 get_matcher,不再构造完事后
		# 按 hasattr(img_matcher, "min_conf_thr") 判断、事后赋值。
		self.img_matcher = initialize_img_matcher(
			self.args.img_matcher, self.args.device, self.args.n_kpts,
			matcher_kwargs=getattr(self.args, "matcher_kwargs", None),
		)
		rospy.loginfo(f"Initialize Image matcher: {self.args.img_matcher}")

	def init_pose_solver(self):
		import rospy
		cfg.merge_from_file(self.args.config_pose_solver)
		self.pose_solver = get_solver(self.args.pose_solver, cfg)
		rospy.loginfo(f"Initialize Pose solver: {self.args.pose_solver}")

	def initalize_ros(self):
		import rospy
		import tf2_ros
		from geometry_msgs.msg import PoseArray
		from nav_msgs.msg import Odometry, Path
		from sensor_msgs.msg import Image
		from visualization_msgs.msg import MarkerArray

		self.pub_graph = rospy.Publisher('/graph', MarkerArray, queue_size=10)
		self.pub_graph_poses = rospy.Publisher('/graph/poses', PoseArray, queue_size=10)

		self.pub_odom = rospy.Publisher('/vloc/odometry', Odometry, queue_size=10)
		self.pub_path = rospy.Publisher('/vloc/path', Path, queue_size=10)
		self.pub_path_gt = rospy.Publisher('/vloc/path_gt', Path, queue_size=10)
		self.pub_map_obs = rospy.Publisher('/vloc/image_map_obs', Image, queue_size=10)

		self.br = tf2_ros.TransformBroadcaster()
		self.path_msg = Path()
		self.path_gt_msg = Path()

	def read_covis_graph_from_files(self, config):
		import rospy
		map_root = pathlib.Path(self.args.map_path)
		self.image_graph = GraphLoader.load_data(
			map_root=map_root,
			resize=config['resize'],
			depth_scale=config['depth_scale'],
			load_rgb=config['load_rgb'], 
			load_depth=config['load_depth'],
			normalized=config['normalized'],
			edge_type='covis'
		)
		rospy.loginfo(f"Loading Covisiblity Graph: {str(self.image_graph)}")

		# Extract VPR descriptors for all nodes in the map
		self.DB_Node_IDS = [node.id for node in self.image_graph.nodes.values()]
		self.DB_DESCRIPTORS = np.array([node.get_descriptor() for node in self.image_graph.nodes.values()], dtype="float32")
		self.DB_POSES = np.empty((self.image_graph.get_num_node(), 7), dtype="float32")
		for indices, (_, node) in enumerate(self.image_graph.nodes.items()):
			self.DB_POSES[indices, :3] = node.trans
			self.DB_POSES[indices, 3:] = node.quat
		logging.debug(f"Extracting {self.DB_DESCRIPTORS.shape} VPR descriptors from the map.")

	def perform_image_matching(self, matcher, map_node, obs_node):
		try:
			matcher_result = matcher(map_node.rgb_image, obs_node.rgb_image)

			if self.args.save_img_matcher:
				mkpts0, mkpts1 = matcher_result["inlier_kpts0"], matcher_result["inlier_kpts1"],
				save_img_matcher_visualization(
					map_node.rgb_image,
					obs_node.rgb_image,
					mkpts0, mkpts1, 
					self.log_dir, 
					obs_node.id, 
					n_step=50
				)

			return matcher_result
		
		except Exception as e:
			logging.error(f"Error in image matching: {e}")
			null_kpts = np.zeros((0, 2), dtype=np.float32)
			return {"num_inliers": 0, "inlier_kpts0": null_kpts, "inlier_kpts1": null_kpts}

	def search_keyframe_from_graph(self, obs_node):
		"""
		This method searches for the visual-closest keyframe in the covisibility graph for a given observation node.
		Parameters:
			obs_node (Node): The observation node for which to find the closest keyframe. 
		Returns:
			Node: The closest keyframe node in the graph.
				If no keyframe is found within the global position threshold, it returns None.
		"""
		# Search for the physical closest keyframe in the graph
		query_pose = obs_node.trans.reshape(1, 3)
		dis, pred = perform_knn_search(self.DB_POSES[:, :3], query_pose, 3, [1])
		if len(pred[0]) == 0: return None

		# Search for the visual closest keyframe in the graph
		node_id = self.DB_Node_IDS[pred[0][0]]
		closest_map_node = self.image_graph.get_node(node_id)
		tmp_nodes = [nei_node for nei_node, _ in closest_map_node.edges.values()] + [closest_map_node]
		all_nei_nodes = []
		for node in tmp_nodes:
			dis_t, dis_angle = node.compute_distance(obs_node)
			if dis_t < self.args.global_pos_threshold and dis_angle < 90.0:
				all_nei_nodes.append(node)

		if len(all_nei_nodes) == 0: return None

		list_dis = [compute_euclidean_dis(obs_node.get_descriptor(), node.get_descriptor()) for node in all_nei_nodes]
		node_min_dis = all_nei_nodes[np.argmin(list_dis)]
		out_str = 'Keyframe candidate: '
		out_str += ' '.join([f'{node.id}({dis:.2f})' for node, dis in zip(all_nei_nodes, list_dis)]) + f' Visual Similar node: {node_min_dis.id}'
		# M5-B4b:这个方法是 localize_obs(mode="local") 没给 ref_map_node 时的回退路径,离线重跑
		# (run_vloc_offline_rerun.py)也会走到这里,但它的运行环境没有 rospy。在线节点有 rospy 时
		# 照旧走 rospy.loginfo;没有时退到标准库 logging(跟 localize_obs 自己一贯的做法一致),
		# 不是换成静默吞掉。
		try:
			import rospy
			rospy.loginfo(out_str)
		except ImportError:
			logging.info(out_str)
		
		return node_min_dis

	def perform_global_loc(self, save_viz: bool = False, candidate_subset: Optional[Set[int]] = None) -> Dict[str, Any]:
		"""
		Args:
			save_viz: save VPR top-K visualization (unchanged).
			candidate_subset: Optional[Set[int]]. M5-B7: when the global planner has recently
				(within ACTIVE_NODES_MAX_AGE_S, checked by the ROS caller) told us roughly where
				the robot should be on the route, this is the set of DB node ids (poses.txt row
				numbers) around there. We verify only inside this subset first, and fall back to
				the regular full-map flow if nothing in the subset passes geometric verification.
				Default None: behaves exactly like before this parameter was added.
		"""
		import rospy
		query_desc = self.curr_obs_node.get_descriptor()
		assert query_desc.shape[1] == self.DB_DESCRIPTORS.shape[1]
		assert query_desc.shape[1] == self.args.vpr_descriptors_dimension
		self.curr_query_descs.append(query_desc)
		self.curr_query_descs = self.curr_query_descs[-self.vpr_match_model.seqLen:]

		if len(self.curr_query_descs) >= self.vpr_match_model.seqLen:
			# Perform sequence match
			query_descs = np.array(self.curr_query_descs).reshape(-1, query_desc.shape[1])
			recall_preds, _, _ = self.vpr_match_model.match(query_descs, recall_values=5)
			rospy.loginfo(recall_preds)

			if save_viz:
				if hasattr(self.curr_obs_node, 'map_root'):
					img_paths = [str(self.curr_obs_node.map_root / self.curr_obs_node.rgb_img_name)]
				else:
					img_paths = [str(self.image_graph.map_root / self.curr_obs_node.rgb_img_name)]

				for pred in recall_preds:
					node_id = self.DB_Node_IDS[pred]
					map_node = self.image_graph.get_node(node_id)
					img_paths.append(str(self.image_graph.map_root / map_node.rgb_img_name))

				preds_correct = [None] * len(img_paths)
				save_vpr_visualization(self.log_dir, self.curr_obs_node.id, img_paths, preds_correct)

				if hasattr(self.vpr_match_model, 'compute_diff_matrix'):
					D_all = self.vpr_match_model.compute_diff_matrix(query_descs)
					self.vpr_match_model.save_diff_matrix_fitting(
						os.path.join(self.log_dir, 'preds'),
						[], [], [], [], D_all, None, None
					)
				else:
					D_all = None

			if candidate_subset:
				# M5-B7:有先验时,先在候选节点子集里核验(全量重排一遍再按子集过滤取前 5,
				# 不是直接拿上面已经截断到 5 个的 recall_preds 去过滤——那 5 个很可能全不在子集里)。
				all_preds, _, _ = self.vpr_match_model.match(query_descs, recall_values=len(self.DB_Node_IDS))
				# faiss 等 k > 数据库大小时会拿 -1 占位,这里顺手滤掉非法下标
				subset_preds = [
					p for p in all_preds
					if 0 <= p < len(self.DB_Node_IDS) and self.DB_Node_IDS[p] in candidate_subset
				][:5]
				if subset_preds:
					subset_result = self._verify_geometric(subset_preds)
					if subset_result['succ']:
						rospy.loginfo(f"[Global Loc] 先验候选子集里核验通过: {subset_result['map_id']}")
						return subset_result
				rospy.loginfo("[Global Loc] 先验候选子集里一个都没核验过,退回全图匹配")

			# Perform geometric verification on the Top-K match
			# Return the one with the maximum number of inliers
			return self._verify_geometric(recall_preds)

		return {'succ': False, 'map_id': None}

	def _verify_one_candidate(self, map_node, obs_node, has_depth: bool):
		"""跟一个候选关键帧做一次图像匹配 + 统一核验(M5-B4:utils.geom_verify.verify_match_result)。

		Returns:
			(verify_inliers, T_world_obs, match_result):统一核验分数;解出的位姿(没深度或
			PnP/基础矩阵没解出来时为 None);匹配器的原始输出(调用方用来设 matched_kpts 做可视化、
			或者取 'num_inliers' 记日志)。
		"""
		match_result = self.perform_image_matching(self.img_matcher, map_node, obs_node)
		depth_np = to_numpy(obs_node.depth_image.squeeze(0)) if has_depth else None
		T_w_map = convert_vec_to_matrix(map_node.trans, map_node.quat, 'xyzw')
		result = geom_verify.verify_match_result(
			match_result,
			map_node.raw_K, map_node.raw_img_size, map_node.img_size,
			obs_node.raw_K, obs_node.raw_img_size, obs_node.img_size,
			T_w_map, depth_np, self.pose_solver,
		)
		return result.inliers, result.T_world_query, match_result

	def localize_obs(self, obs_node, mode: str, prior: Optional[Dict[str, Any]] = None) -> LocResult:
		"""M5-B4 统一的单帧定位引擎:候选核验(mode="global")或局部精化(mode="local"),核验口径
		统一走 utils.geom_verify(查询图网格去重 + PnP-RANSAC 内点数;查询没深度时退化成基础矩阵
		内点数,只认出关键帧、不解位姿)。不 import rospy,日志走标准库 logging——这样
		run_vloc_offline_rerun.py(运行环境剔除了 ROS,不能 import 这个模块)能通过
		utils.geom_verify.verify_match_result 跑同一份核验逻辑,不需要再抄一份。

		Args:
			obs_node: 当前观测(ImageNode);obs_node.depth_image is None 时走无深度分支。
			mode: "global" 或 "local"。
				"global":prior 必须给 "candidate_ids"(VPR 按相似度排好序的 DB node id 列表,
					不是 DB_DESCRIPTORS 的行号)。可选 "early_exit"(默认 True:遇到第一个核验分数
					达标的候选就停,直接用它核验时解出的位姿当结果,不用再跑一次局部匹配;False:
					全部候选跑完,取分数最高的那个——这是在线节点改这个默认值之前的行为,
					离线重跑 --gv_early_exit 0 用来跟新默认值对照)、"t_vpr_s"(只填进返回结果,
					不参与计算)。
				"local":prior 可选 "ref_map_node"(已经选好的参考关键帧);不给就调
					self.search_keyframe_from_graph(obs_node) 自己选(跟原 perform_local_loc 一样:
					"最近关键帧 + 邻居里挑描述子最近的")。

		Returns:
			LocResult。stage="none" 表示没找到候选、或核验分数没过门槛(全局用 gv_min_inliers,
			本地跟踪用 gv_min_inliers_local,没深度时两处都用 gv_min_inliers_nodepth)。
		"""
		prior = prior or {}
		has_depth = getattr(obs_node, "depth_image", None) is not None
		gv_threshold = self.args.gv_min_inliers if has_depth else self.args.gv_min_inliers_nodepth

		if mode == "global":
			candidate_ids: List[int] = list(prior.get("candidate_ids") or [])
			early_exit: bool = prior.get("early_exit", True)
			t_start = time.time()
			best_inliers, best_map_id, best_T = 0, None, None
			n_matches = 0
			for cand_id in candidate_ids:
				map_node = self.image_graph.get_node(cand_id)
				inliers, T_world_obs, _match_result = self._verify_one_candidate(map_node, obs_node, has_depth)
				n_matches += 1
				logging.info(
					f"[GV] Query {obs_node.rgb_img_name} - DB {map_node.rgb_img_name} verify_inliers={inliers}"
				)
				if inliers > best_inliers:
					best_inliers, best_map_id, best_T = inliers, cand_id, T_world_obs
				if early_exit and inliers >= gv_threshold:
					break
			t_verify_s = time.time() - t_start
			succ = best_map_id is not None and best_inliers >= gv_threshold
			logging.info(f"[Global Loc] best map node: {best_map_id} verify_inliers={best_inliers}")
			return LocResult(
				succ=succ, stage=("global" if succ else "none"),
				map_id=(best_map_id if succ else None),
				T_world_obs=(best_T if succ else None),
				verify_inliers=best_inliers, n_matches_run=n_matches,
				t_vpr_s=float(prior.get("t_vpr_s", 0.0)), t_verify_s=t_verify_s, t_local_s=0.0,
			)

		if mode == "local":
			t_start = time.time()
			ref_map_node = prior.get("ref_map_node")
			if ref_map_node is None:
				ref_map_node = self.search_keyframe_from_graph(obs_node)
			if ref_map_node is None:
				return LocResult(False, "none", None, None, 0, 0, 0.0, 0.0, time.time() - t_start)

			self.ref_map_node = ref_map_node
			inliers, T_world_obs, match_result = self._verify_one_candidate(ref_map_node, obs_node, has_depth)
			# 跟原 perform_local_loc 行为一致:不管核验过没过,都把这次匹配到的点记到两个节点上,
			# 给 publish_message() 的可视化用(匹配器自己的原始内点,未经网格去重)。
			num_inliers_raw = match_result["num_inliers"]
			ref_map_node.set_matched_kpts(match_result["inlier_kpts0"], num_inliers_raw)
			obs_node.set_matched_kpts(match_result["inlier_kpts1"], num_inliers_raw)
			logging.info(f"Number of matched kpts inliers: {num_inliers_raw} (统一核验: {inliers})")

			# 本地跟踪(已经知道大概在哪,只跟参考关键帧核对)单独一个门槛 gv_min_inliers_local,
			# 和全局定位(在全图里找)的 gv_min_inliers 分开;配置没给就跟全局用同一个数。
			local_threshold = (
				getattr(self.args, "gv_min_inliers_local", None) or self.args.gv_min_inliers
			) if has_depth else self.args.gv_min_inliers_nodepth
			succ = T_world_obs is not None and inliers >= local_threshold
			return LocResult(
				succ=succ, stage=("local" if succ else "none"),
				map_id=(ref_map_node.id if succ else None),
				T_world_obs=(T_world_obs if succ else None),
				verify_inliers=inliers, n_matches_run=1,
				t_vpr_s=0.0, t_verify_s=0.0, t_local_s=time.time() - t_start,
			)

		raise ValueError(f"localize_obs: unknown mode {mode!r}")

	def _verify_geometric(self, recall_preds: List[int], early_exit: bool = True) -> Dict[str, Any]:
		"""对一串 VPR 候选(DB_DESCRIPTORS 行号列表,不是 node id)做统一核验,返回
		{'succ', 'map_id', 'T_w_obs'}。

		M5-B4:核验逻辑整个转发到 localize_obs(mode="global")——核验分数不再是匹配器自己的单应
		内点数(GV_SCORE_THRESHOLD),而是网格去重后的 PnP-RANSAC 内点数,门槛是
		self.args.gv_min_inliers。early_exit=True(默认,新行为,之前在线节点是全部候选跑完取
		最大值,离线重跑原来默认是遇到第一个就停,现在两边统一成后者):第一个核验分数达标的候选
		就停,返回值里的 'T_w_obs' 就是那次核验时核验用的查询深度解出来的位姿——不是 None 时,
		调用方(perform_global_loc 的调用方,即 ros_loc_pipeline.process_frame /
		loc_pipeline.perform_localization)可以跳过这一帧的局部定位,直接用这个位姿。
		"""
		if self.img_matcher is None:
			# 没配图像匹配器:退化成"VPR 说是哪个就是哪个",不解位姿。
			return {'succ': True, 'map_id': self.DB_Node_IDS[recall_preds[0]], 'T_w_obs': None}

		candidate_ids = [self.DB_Node_IDS[pred] for pred in recall_preds]
		result = self.localize_obs(
			self.curr_obs_node, mode="global",
			prior={"candidate_ids": candidate_ids, "early_exit": early_exit},
		)
		return {'succ': result.succ, 'map_id': result.map_id, 'T_w_obs': result.T_world_obs}

	def perform_local_loc(self) -> Dict[str, Any]:
		"""M5-B4:整个转发到 localize_obs(mode="local")。返回值的 key 不变(仍是
		'succ'/'T_w_obs'/'solver_inliers'),调用方(ros_loc_pipeline.process_frame、本文件的
		perform_localization)不用改;'solver_inliers' 现在是统一核验分数(网格去重后的
		PnP-RANSAC 内点数),门槛也从 self.args.min_solver_inliers_thre 换成了
		self.args.gv_min_inliers(跟全局定位核验用同一个门槛,见简报"同一个门槛"的要求)。
		"""
		result = self.localize_obs(self.curr_obs_node, mode="local")
		return {'succ': result.succ, 'T_w_obs': result.T_world_obs, 'solver_inliers': result.verify_inliers}

	def publish_message(self):
		import rospy
		from std_msgs.msg import Header
		from utils.utils_ros import ros_msg
		# Publish Sensor Data
		if self.curr_obs_node is not None:
			header = Header(stamp=rospy.Time.from_sec(self.curr_obs_node.time), frame_id=self.frame_id_map)
			
			# Publish odometry and path if the local position is available
			if self.has_local_pos:
				odom = ros_msg.convert_vec_to_rosodom(self.curr_obs_node.trans, self.curr_obs_node.quat, header, self.child_frame_id)
				self.pub_odom.publish(odom)
				pose_msg = ros_msg.convert_odom_to_rospose(odom)
				self.path_msg.header = header
				self.path_msg.poses.append(pose_msg)
				self.pub_path.publish(self.path_msg)

			if self.curr_obs_node.has_pose_gt:
				pose_msg = ros_msg.convert_vec_to_rospose(self.curr_obs_node.trans_gt, self.curr_obs_node.quat_gt, header)
				self.path_gt_msg.header = header
				self.path_gt_msg.poses.append(pose_msg)
				self.pub_path_gt.publish(self.path_gt_msg)

			# Only draw and publish the match image when --viz is on and someone subscribes
			if self.ref_map_node is not None and self.args.viz and self.pub_map_obs.get_num_connections() > 0:
				n_viz = 10 # visualize n_viz matched keypoints
				rgb_img_ref = (np.transpose(to_numpy(self.ref_map_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				rgb_img_obs = (np.transpose(to_numpy(self.curr_obs_node.rgb_image), (1, 2, 0)) * 255).astype(np.uint8)
				mkpts_map, num_inliers = self.ref_map_node.get_matched_kpts()
				mkpts_obs, _ = self.curr_obs_node.get_matched_kpts()
				if mkpts_map is not None and mkpts_obs is not None:
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
					img_msg = ros_msg.convert_cvimg_to_rosimg(merged_img, "bgr8", header, compressed=False)
					self.pub_map_obs.publish(img_msg)

# Test loc_pipeline without using ROS
def perform_localization(loc: LocPipeline, args):
	import rospy
	from utils.utils_geom import read_intrinsics, read_poses, read_descriptors, correct_intrinsic_scale

	"""Main loop for processing observations"""
	poses = read_poses(os.path.join(args.query_data_path, 'poses.txt'))
	intrs = read_intrinsics(os.path.join(args.query_data_path, 'intrinsics.txt'))
	descs = read_descriptors(os.path.join(args.query_data_path, 'database_descriptors.txt'))
	resize = args.image_size # WxH
	loc.last_obs_node = None
	
	for node_id, (rgb_img_name, pose) in enumerate(poses.items()):
		if rospy.is_shutdown(): break	
		print(f"Loading observation {rgb_img_name}")

		rgb_img_path = os.path.join(args.query_data_path, rgb_img_name)
		rgb_img = load_rgb_image(rgb_img_path, resize)

		depth_img_name = rgb_img_name.replace('color.jpg', 'depth.png')
		depth_img_path = os.path.join(args.query_data_path, depth_img_name)
		depth_img = load_depth_image(depth_img_path, depth_scale=0.001)

		intr = intrs[rgb_img_name]
		width, height = int(intr[4]), int(intr[5])
		raw_K = np.array([intr[0], 0, intr[2], 0, intr[1], intr[3], 0, 0, 1], dtype=np.float32).reshape(3, 3)
		raw_size = (width, height)
		if resize is not None:
			K = correct_intrinsic_scale(
				raw_K, resize[0] / width, resize[1] / height
			) 
			img_size = np.array([int(resize[0]), int(resize[1])])
		else:
			K = raw_K
			img_size = raw_size

		# Create observation node
		obs_node = ImageNode(
			node_id, rgb_img, depth_img, descs[rgb_img_name].reshape(1, -1),
			rospy.Time.now().to_sec(),
			np.zeros(3), np.array([0, 0, 0, 1]),
			K, img_size,
			rgb_img_name, depth_img_name
		)
		obs_node.set_raw_intrinsics(raw_K, raw_size)
		trans, quat = convert_pose_inv(
			pose[4:], 
			np.roll(pose[:4], -1), 
			'xyzw'
		)
		obs_node.set_pose_gt(trans, quat)
		obs_node.map_root = pathlib.Path(args.query_data_path)
		loc.curr_obs_node = obs_node

		"""Perform global localization via. visual place recognition"""
		# M5-B4: perform_global_loc's unified verification already ran PnP on the winning
		# candidate; when it produced a usable pose, this frame is already fully localized
		# and the local-matching step below is skipped for it (same as ros_loc_pipeline.process_frame).
		just_localized_from_global = False
		if not loc.has_global_pos:
			loc_start_time = time.time()
			result = loc.perform_global_loc(save_viz=(args.num_preds_to_save!=0))
			rospy.loginfo(f"Global localization costs: {time.time() - loc_start_time:.3f}s")
			if result['succ']:
				matched_map_id = result['map_id']
				loc.has_global_pos = True
				loc.ref_map_node = loc.image_graph.get_node(matched_map_id)
				if result.get('T_w_obs') is not None:
					trans, quat = convert_matrix_to_vec(result['T_w_obs'], 'xyzw')
					loc.curr_obs_node.set_pose(trans, quat)
					loc.has_local_pos = True
					just_localized_from_global = True
					rospy.logwarn(f'[Succed] Found VPR Node in global position: {matched_map_id} (位姿已在核验时解出)')
				else:
					loc.curr_obs_node.set_pose(loc.ref_map_node.trans, loc.ref_map_node.quat)
					rospy.logwarn(f'[Succed] Found VPR Node in global position: {matched_map_id}')
			else:
				loc.ref_map_node = None
				rospy.logwarn('[Fail] to determine the global position since no VPR results.')
				continue
		else:
			if loc.last_obs_node is not None:
				init_trans, init_quat = loc.last_obs_node.trans, loc.last_obs_node.quat
				loc.curr_obs_node.set_pose(init_trans, init_quat)
				dis_trans, _ = compute_pose_error(
					(init_trans, init_quat), 
					(loc.ref_map_node.trans, loc.ref_map_node.quat), 
					mode='vector'
				)
				if dis_trans > loc.args.global_pos_threshold * 2.0:
					rospy.logwarn('Too far distance from the ref_map_node. Losing Visual Tracking. Reset the global position.')
					loc.has_global_pos = False
					loc.ref_map_node = None
			else:
				rospy.logwarn('[Fail] to determine the global position since not correct VPR.')
				continue				

		"""Perform local localization via. image matching"""
		if loc.has_global_pos and not just_localized_from_global:
			loc_start_time = time.time()
			result = loc.perform_local_loc()
			rospy.loginfo(f"Local localization costs: {time.time() - loc_start_time:.3f}s")
			if result['succ']:
				T_w_obs = result['T_w_obs']
				trans, quat = convert_matrix_to_vec(T_w_obs, 'xyzw')
				loc.curr_obs_node.set_pose(trans, quat)
				loc.has_local_pos = True
				rospy.loginfo(f'Groundtruth Poses: {loc.curr_obs_node.trans_gt.T}')
				rospy.loginfo(f'Estimated Poses: {trans.T}\n')
			else:
				loc.has_local_pos = False
				rospy.logwarn('[Fail] to determine the local position.')
				# DEBUG(gogojjh):
				loc.has_global_pos = False
				loc.ref_map_node = None


		loc.publish_message()
		# Set as the initial guess of the next observation
		loc.last_obs_node = loc.curr_obs_node
		time.sleep(0.01)
		#input()

if __name__ == '__main__':
	import rospy
	args = parse_arguments()
	out_dir = pathlib.Path(os.path.join(args.map_path, 'tmp/output_loc_pipeline'))
	config = dict(
		resize=args.image_size, depth_scale=args.depth_scale, 
		load_rgb=True, load_depth=False, normalized=False,
		color_correct=False
	)

	# Initialize the localization pipeline
	loc_pipeline = LocPipeline(args, out_dir)
	loc_pipeline.init_vpr_model()
	loc_pipeline.init_img_matcher()
	loc_pipeline.init_pose_solver()
	loc_pipeline.read_covis_graph_from_files(config)
	loc_pipeline.init_vpr_match_model()

	rospy.init_node('loc_pipeline_node', anonymous=True)
	loc_pipeline.initalize_ros()
	loc_pipeline.frame_id_map = rospy.get_param('~frame_id_map', 'vloc_map')
	loc_pipeline.child_frame_id = rospy.get_param('~child_frame_id', 'camera')

	perform_localization(loc_pipeline, args)

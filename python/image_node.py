import numpy as np
from utils.base_node import BaseNode

class ImageNode(BaseNode):
	def __init__(self, 
			  	 node_id: int, 
				 rgb_image: str,
				 depth_image: str,
				 global_descriptor: np.ndarray, 
				 time: float, 
				 trans: np.ndarray, 
				 quat: np.ndarray, 
				 K: np.ndarray, 
				 img_size: np.ndarray,
				 rgb_img_name: str, 
				 depth_img_name: str):
		super().__init__(node_id, trans, quat)

		# RGB and depth images
		self.rgb_image = rgb_image
		self.depth_image = depth_image

		# Path to the RGB and depth images
		self.rgb_img_name = rgb_img_name
		self.depth_img_name = depth_img_name

		# VPR descriptor: numpy.array
		self.global_descriptor = global_descriptor

		# Time of the image
		self.time = time

		# Camera intrinsics
		self.K = K
		self.img_size = img_size # width, height

		self.raw_K = K
		self.raw_img_size = img_size

		# Next node using in the shortest path
		self.next_node = None

		# Matched keypoints
		self.mkpts = None
		self.inliers = None

	def __str__(self):
		out_str = f'Image Node ID: {self.id} with edge number: {len(self.edges)}'
		out_str += f', intrinsics: [' + ', '.join([str(x) for x in self.raw_K.flatten()]) + ']'
		return out_str

	def set_descriptor(self, global_descriptor: np.ndarray):
		self.global_descriptor = global_descriptor

	def get_descriptor(self):
		return self.global_descriptor
	
	def set_matched_kpts(self, mkpts, num_inliers):
		self.mkpts = mkpts
		self.inliers = num_inliers
	
	def get_matched_kpts(self):
		return self.mkpts, self.inliers

	def set_raw_intrinsics(self, raw_K: np.ndarray, raw_img_size: np.ndarray):
		self.raw_K = raw_K
		self.raw_img_size = raw_img_size
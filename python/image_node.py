import os

from pycpptools.src.python.utils_algorithm.base_node import BaseNode

class ImageNode(BaseNode):
	def __init__(self, id, 
				rgb_image, depth_image, global_descriptor, 
				time, trans, quat, K, img_size,
				rgb_img_path, depth_img_path):
		super().__init__(id, trans, quat)

		# RGB and depth images
		self.rgb_image = rgb_image
		self.depth_image = depth_image

		# Path to the RGB and depth images
		self.rgb_img_path = rgb_img_path
		self.depth_img_path = depth_img_path

		# VPR descriptor: numpy.array
		self.global_descriptor = global_descriptor

		# Time of the image
		self.time = time

		# Camera intrinsics
		self.K = K
		self.img_size = img_size # width, height

		# Next node using in the shortest path
		self.next_node = None

		# Matched keypoints
		self.mkpts = None
		self.inliers = None

	def set_descriptor(self, global_descriptor):
		self.global_descriptor = global_descriptor

	def get_descriptor(self):
		return self.global_descriptor
	
	def set_matched_kpts(self, mkpts, num_inliers):
		self.mkpts = mkpts
		self.inliers = num_inliers
	
	def get_matched_kpts(self):
		return self.mkpts, self.inliers

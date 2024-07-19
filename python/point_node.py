import os

from pycpptools.src.python.utils_algorithm.base_node import BaseNode

class PointNode(BaseNode):
	def __init__(self, id, global_descriptor, time, trans, quat, rgb_img_path, depth_img_path):
		super().__init__(id, trans, quat)

		# Image paths
		self.rgb_img_path = rgb_img_path
		self.depth_img_path = depth_img_path

		# VPR descriptor
		self.global_descriptor = global_descriptor

		# Time of the image
		self.time = time

	def set_descriptor(self, global_descriptor):
		self.global_descriptor = global_descriptor

	def get_descriptor(self):
		return self.global_descriptor

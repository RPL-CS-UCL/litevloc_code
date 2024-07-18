import os

from pycpptools.src.python.utils_algorithm.base_node import BaseNode

class ImageNode(BaseNode):
	def __init__(self, id, 
							 rgb_image, depth_image, global_descriptor, 
							 time, t_w_cam, quat_w_cam, 
							 rgb_img_path, depth_img_path):
		super().__init__(id)

		self.rgb_image = rgb_image
		self.depth_image = depth_image

		self.global_descriptor = global_descriptor

		self.time = time
		self.t_w_cam = t_w_cam
		self.quat_w_cam = quat_w_cam

		self.rgb_img_path = rgb_img_path
		self.depth_img_path = depth_img_path

		self.next_node = None

	def set_descriptor(self, global_descriptor):
		self.global_descriptor = global_descriptor

	def get_descriptor(self):
		return self.global_descriptor

	def add_next_node(self, next_node):
		self.next_node = next_node

	def get_next_node(self):
		return self.next_node
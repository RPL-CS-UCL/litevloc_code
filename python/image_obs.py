import os
import numpy as np

from utils.utils_image_matching_method import load_image

class ImageObsLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(obs_path, image_size, sample_obs):
		image_obs = ImageObs()
		
		poses_w_cam = np.loadtxt(os.path.join(obs_path, 'obs_camera_pose_gt.txt'))
		for i, pose_w_cam in enumerate(poses_w_cam[::sample_obs, :]):
			img_path = os.path.join(obs_path, 'obs_rgb', f'{i:06}.png')
			image = load_image(img_path, image_size)

			time = pose_w_cam[0]
			t_w_cam = pose_w_cam[1:4]
			quat_w_cam = np.roll(pose_w_cam[4:], 1) # [qw, qx, qy, qz]

			node = Node(i, image, f'image node {i}', time, t_w_cam, quat_w_cam)
			image_obs.add_node(node)
			if i > 5:
				break
		return image_obs

class Node:
	def __init__(self, id, image, descriptor, time, t_w_cam, quat_w_cam):
		self.id = id
		self.image = image
		self.descriptor = descriptor

		self.time = time
		self.t_w_cam = t_w_cam
		self.quat_w_cam = quat_w_cam

class ImageObs:
	def __init__(self):
		self.nodes = {}

	def add_node(self, new_node):
		if new_node.id not in self.nodes:
			self.nodes[new_node.id] = new_node

	def get_node(self, id):
		if id in self.nodes:
			return self.nodes[id]
		else:
			return None

class TestImageObs():
	def __init__(self):
		pass
	
	def run_test(self):
		pass

if __name__ == '__main__':
	test_image_obs = TestImageObs()
	test_image_obs.run_test()
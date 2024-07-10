import os
import numpy as np

from utils.utils_image import load_image

class ImageObsLoader:
	def __init__(self):
		pass

	@staticmethod
	def load_data(obs_path, image_size, normalized=False, num_sample=1, num_load=10000):
		image_obs = ImageObs()
		
		poses_w_cam = np.loadtxt(os.path.join(obs_path, 'camera_pose_gt.txt'))
		for i in range(0, 
								 	 min(poses_w_cam.shape[0], num_load * num_sample), 
									 num_sample):
			img_path = os.path.join(obs_path, 'rgb', f'{i:06}.png')
			image = load_image(img_path, image_size, normalized)

			pose_w_cam = poses_w_cam[i, :]
			time = pose_w_cam[0]
			t_w_cam = pose_w_cam[1:4]
			quat_w_cam = np.roll(pose_w_cam[4:], 1) # [qw, qx, qy, qz]

			node = Node(i, image, f'image node {i}', time, t_w_cam, quat_w_cam, img_path)
			image_obs.add_node(node)

			if i / num_sample > num_load:
				break

		return image_obs

class Node:
	def __init__(self, id, image, descriptor, time, t_w_cam, quat_w_cam, img_path):
		self.id = id
		self.image = image
		self.descriptor = descriptor

		self.time = time
		self.t_w_cam = t_w_cam
		self.quat_w_cam = quat_w_cam

		self.img_path = img_path

	def set_descriptor(self, descriptor):
		self.descriptor = descriptor

	def get_descriptor(self):
		return self.descriptor

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
		
	def get_num_node(self):
		return len(self.nodes)

class TestImageObs():
	def __init__(self):
		pass
	
	def run_test(self):
		pass

if __name__ == '__main__':
	test_image_obs = TestImageObs()
	test_image_obs.run_test()
import numpy as np
from utils.base_node import BaseNode

class PointNode(BaseNode):
	def __init__(
			self, 
			id: int, 
			time: float, 
			trans: np.ndarray,
			quat: np.ndarray,
			gps_data: np.ndarray = None):
		super().__init__(id, trans, quat)

		# Data collection moment of this node in UTC timestamp
		self.time = time

		# GPS data
		self.gps_data = gps_data


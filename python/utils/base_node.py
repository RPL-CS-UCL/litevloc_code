#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../"))

import numpy as np
from utils.utils_geom import compute_pose_error

class BaseNode:
	def __init__(self, id, trans=np.zeros(3), quat=np.array([0.0, 0.0, 0.0, 1.0]), time=0.0):
		self.id = id
		self._edges = {}  # Ensure unique id: {nodeB.id: (nodeB, weight)}
		
		self.trans = trans  # xyz
		self.quat = quat    # xyzw
		
		self.has_pose_gt = False
		self.trans_gt = np.zeros(3)
		self.quat_gt = np.array([0.0, 0.0, 0.0, 1.0])
		
		self.next_node = None  # For shortest path tracking
		
	def __str__(self):
		return f'Node ID: {self.id} with edge number: {len(self._edges)}'
	
	def __lt__(self, other):
		return self.id < other.id
	
	@property
	def edges(self):
		return self._edges
	
	def set_pose(self, trans, quat):
		self.trans = trans
		self.quat = quat
	
	def set_pose_gt(self, trans_gt, quat_gt):
		self.has_pose_gt = True
		self.trans_gt = trans_gt
		self.quat_gt = quat_gt
	
	def set_edge(self, new_edges):
		self._edges = new_edges

	def add_edge(self, next_node, weight):
		"""Add/replace edge with O(1) time complexity"""
		self._edges[next_node.id] = (next_node, weight)
	
	def add_next_node(self, next_node):
		self.next_node = next_node
	
	def get_next_node(self):
		return self.next_node
		
	def compute_distance(self, next_node):
		dis_trans, dis_angle = compute_pose_error(
			(self.trans, self.quat), 
			(next_node.trans, next_node.quat),
			mode='vector'
		)
		return dis_trans, dis_angle
	
	def compute_gt_distance(self, next_node):
		dis_trans, dis_angle = compute_pose_error(
			(self.trans_gt, self.quat_gt), 
			(next_node.trans_gt, next_node.quat_gt),
			mode='vector'
		)
		return dis_trans, dis_angle	
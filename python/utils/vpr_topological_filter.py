#! /usr/bin/env python

import faiss
import torch
import numpy as np
from typing import Union

class PlaceRecognitionTopologicalFilter:
    '''
    Adapted from https://github.com/mingu6/ProbFiltersVPR/blob/master/src/models/TopologicalFilter.py
    '''
    def __init__(self):
        pass

    def initialize_model(self, db_descriptors, delta=5):
        """
        Initialize the VPRTopologicalFilter object.
        Initialize the belief distribution - uniform distribution

        Args:
            db_descriptors (numpy.ndarray): The map descriptors.
            delta (int, optional): The delta value. Defaults to 5.
            prop_radius (float, optional): The propagation radius. Defaults to 10.0.
        """
        # get map descriptors
        self.db_descriptors = db_descriptors

        # initialize hidden states and obs likelihood parameters
        self.delta = delta
        self.lambda1 = None
        self.belief = None

        self.belief = np.ones(self.db_descriptors.shape[0]) / self.db_descriptors.shape[0]

    # DEBUG(ggoojjh): the current implementation depends on the topological map, where nodes are not connected from the tail to head
    def get_back_prop_node(self, node) -> list:
        preds = {node.id}
        for edge in node.edges.values():
            if node.id > edge[0].id:
                preds.add(edge[0].id)
            # for sub_edge in edge[0].edges.values():
            #     if edge[0].id > sub_edge[0].id:
            #         preds.add(sub_edge[0].id)
        preds = list(set(preds))
        return preds

    def comp_dist_descriptor(self, descriptor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        ##### Option 1: cosine similarity
        # dists = np.sqrt(2 - 2 * np.dot(self.db_descriptors, descriptor.reshape(-1)))
        ##### Option 2: euclidean distance
        dists = np.linalg.norm(self.db_descriptors - descriptor, axis=1)
        return dists

    def obs_lhood(self, descriptor: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        '''Observation likelihood of the query descriptor'''
        dists = self.comp_dist_descriptor(descriptor)
        vsim = np.exp(-self.lambda1 * dists)
        return vsim

    def match(self, db_map, query_desc: Union[np.ndarray, torch.Tensor], recall_values=1):
        '''
        Match the query image to the topological map.

        Runs a prediction step followed by a measurement step:
        - Prediction: Propagate belief mass using the transition model
        - Measurement: Update belief mass using the observation likelihood

        After the process, the map node with the highest probability is
        returned as the subgoal.

        Returns:
        - recall_preds: the top recall indices of the matched map nodes
        - pred: the index of the matched map node
        - prob: the probability of the matching
        '''
        # Initialize the lambda
        if self.lambda1 is None:
            dists = self.comp_dist_descriptor(query_desc)
            descriptor_quantiles = np.quantile(dists, [0.025, 0.975])
            self.lambda1 = np.log(self.delta) / (descriptor_quantiles[1] - descriptor_quantiles[0])

        # Prediction step
        belief_pred = np.zeros_like(self.belief)
        for node_id in range(len(belief_pred)):
            node = db_map.get_node(node_id)
            back_prop_node_id = self.get_back_prop_node(node)
            belief_pred[node_id] = np.sum(self.belief[back_prop_node_id])
        obs_lhood = self.obs_lhood(query_desc)
        # Measurement step
        self.belief = obs_lhood * belief_pred
        self.belief /= self.belief.sum()

        # print('Prediction: Belief')
        # str = ' '.join([f'{x:.2f}' for x in belief_pred])
        # print(str)
        # print('Measurement Update: Belief')
        # str = ' '.join([f'{x:.2f}' for x in self.belief])
        # print(str)

        # Get the top recall values
        recall_preds = np.argsort(self.belief)[-recall_values:][::-1]
        pred = np.argmax(self.belief)
        prob = self.belief[pred]
    
        return recall_preds, pred, prob

if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
    import argparse
    from image_graph import ImageGraphLoader as GraphLoader
    from utils.utils_geom import compute_pose_error
    from tqdm import tqdm

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_map_path", type=str, help="Path to the database map file")
    parser.add_argument("--query_map_path", type=str, help="Path to the query map file")
    args = parser.parse_args()
    # Load database and query
    db_map = GraphLoader.load_data(
        args.db_map_path,
        [512, 288],
        depth_scale=0.0,
        load_rgb=True,
        load_depth=False,
        normalized=False
    )
    query_map = GraphLoader.load_data(
        args.query_map_path,
        [512, 288],
        depth_scale=0.0,
        load_rgb=True,
        load_depth=False,
        normalized=False
    )
    # Performance test
    db_descriptors = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")
    model = PlaceRecognitionTopologicalFilter()
    model.initialize_model(db_descriptors)
    preds = []
    for node in tqdm(query_map.nodes.values()):
        query_desc = node.get_descriptor()
        recall_preds, pred, score = model.match(db_map, query_desc.reshape(1, -1), recall_values=5)
        preds.append(recall_preds)

    succ = 0
    for i, node in enumerate(query_map.nodes.values()):
        ref_map_node = db_map.nodes[preds[i][0]]
        dis_tsl, dis_angle = compute_pose_error(
            (node.trans_gt, node.quat_gt), 
            (ref_map_node.trans_gt, ref_map_node.quat_gt),
            mode='vector'
        )
        if dis_tsl < 10.0 and dis_angle < 90.0:
            succ += 1
            print(f"Correct prediction: Query {node.id} - DB: {preds[i][0]}")
        else:
            print(f"Wrong prediction: Query {node.id} - DB: {preds[i][0]}")
    print(f"Success rate: {succ / len(query_map.nodes)}")
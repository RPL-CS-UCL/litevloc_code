#! /usr/bin/env python

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))
from python.utils.vpr_topological_filter import PlaceRecognitionTopologicalFilter
import numpy as np
import argparse

from image_graph import ImageGraphLoader as GraphLoader
from image_graph import ImageGraph

from utils.utils_map_merging import save_vis_coarse_loc
from utils.utils_vpr_method import perform_knn_search, save_vis_brief_function, save_vis_diff_matrix

import pycpptools.src.python.utils_math as pytool_math

import matplotlib.pyplot as plt

log_dir = '/Rocket_ssd/dataset/tmp'

def test_pr_raw(db_map, query_map):
    db_descriptors = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")
    query_descriptors = np.array([node.get_descriptor() for _, node in query_map.nodes.items()], dtype="float32")

    dist, preds = perform_knn_search(db_descriptors, query_descriptors, db_descriptors.shape[1], recall_values=[5])    
    save_vis_coarse_loc(log_dir, db_map, query_map, 2, preds, suffex='raw')

    succ = 0
    for i, node in enumerate(query_map.nodes.values()):
        ref_map_node = db_map.nodes[preds[i][0]]
        dis_tsl, dis_angle = \
            pytool_math.tools_eigen.compute_relative_dis(\
                node.trans_gt, node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
        if dis_tsl < 10.0 and dis_angle < 90.0:
            succ += 1
        else:
            print(f"Wrong prediction: node id {node.id}, wrong map node id {preds[i][0]}")

    return preds, succ / len(query_map.nodes)

def test_pr_topological_filter(db_map, query_map):
    db_descriptors = np.array([node.get_descriptor() for _, node in db_map.nodes.items()], dtype="float32")   
    # Initialize the Bayesian filter
    topo_filter = PlaceRecognitionTopologicalFilter(db_descriptors, window_lower=-1, window_upper=1, recall_values=5)
    preds = []
    for node in query_map.nodes.values():
        query_desc = node.get_descriptor()
        if topo_filter.belief is None:
            recall_preds, pred, score = topo_filter.initialize_model(query_desc)
        else:
            recall_preds, pred, score = topo_filter.match(query_desc)
        preds.append(recall_preds)
        save_vis_brief_function(log_dir, topo_filter.belief, node.id)

    preds = np.array(preds)
    save_vis_coarse_loc(log_dir, db_map, query_map, 2, preds, suffex='filter')

    succ = 0
    for i, node in enumerate(query_map.nodes.values()):
        ref_map_node = db_map.nodes[preds[i][0]]
        dis_tsl, dis_angle = \
            pytool_math.tools_eigen.compute_relative_dis(\
                node.trans_gt, node.quat_gt, ref_map_node.trans_gt, ref_map_node.quat_gt)
        if dis_tsl < 10.0 and dis_angle < 90.0:
            succ += 1
        else:
            print(f"Wrong prediction: node id {node.id}, wrong map node id {preds[i][0]}")

    return preds, succ / len(query_map.nodes)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, help="Path to the map file")
    parser.add_argument("--num_submap", type=int, help="Number of submaps in the map file")
    args = parser.parse_args()

    submaps = []
    for i in range(args.num_submap):
        submap_id = len(submaps)
        submap_path = os.path.join(args.dataset_path, f'out_map{submap_id}')
        image_graph = GraphLoader.load_data(
            submap_path,
            [512, 288],
            depth_scale=0.0,
            load_rgb=True,
            load_depth=False,
            normalized=False
        )
        submaps.append((submap_id, image_graph))
        print(f"Loaded {image_graph} from {submap_path}")
    print(f"Loaded {len(submaps)} submaps.")

    db_submap_id, query_submap_id = 1, 3

    # Method 1: without using sequential information
    preds, succ_ratio = test_pr_raw(submaps[db_submap_id][1], submaps[query_submap_id][1])
    print(f"VPR Raw: Success ratio = {succ_ratio:.3f}")

    # Method 2: with topological filter
    preds, succ_ratio = test_pr_topological_filter(submaps[db_submap_id][1], submaps[query_submap_id][1])
    print(f"VPR with Filter: Success ratio = {succ_ratio:.3f}")

    # Draw the difference matrix
    query_descriptors = np.array([node.get_descriptor() for _, node in submaps[query_submap_id][1].nodes.items()], dtype="float32")
    db_descriptors = np.array([node.get_descriptor() for _, node in submaps[db_submap_id][1].nodes.items()], dtype="float32")   
    diff_matrix = np.zeros((db_descriptors.shape[0], query_descriptors.shape[0]))
    for i, query_desc in enumerate(query_descriptors):
        diff_matrix[:, i] = np.linalg.norm(db_descriptors - query_desc, axis=1)
    save_vis_diff_matrix(log_dir, diff_matrix)
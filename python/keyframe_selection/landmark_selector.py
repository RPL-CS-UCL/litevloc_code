import numpy as np

from image_node import ImageNode
from image_graph import ImageGraph
from utils.utils_vpr_method import perform_knn_search

import math

class LandmarkSelector:
    def __init__(self):
        # Parameters for probability calculation
        self.Q_th = 30.0     # Midpoint for quality sigmoid
        self.k_Q = 0.6       # Quality sigmoid steepness

        self.R_th = 0.4      # Information redundancy threshold
        self.k_R = 0.1       # Information redundancy sensitivity (higher, more sensitive)

        self.G_th = 0.4      # Information gain threshold
        self.k_G = 1.0
        
        self.T_th = 3600.0   # Timestamp threshold (second)
        self.lambda_T = 0.05 # Timestamp sensitivity

    # The prbability of keeping the frame
    def quality_probability(self, Q):
        """Sigmoid function for image quality (0-100). Higher is better."""
        return 1 / (1 + math.exp(-self.k_Q * (Q - self.Q_th)))

    def redundancy_probability(self, R):
        """Exponential decay function for redundancy (0-1). Lower is better."""
        return 1 / (1 + math.exp(-self.k_R * 10.0 * (R - self.R_th)))

    def gain_probability(self, G):
        """Exponential increase function for information gain (0-1). Higher is better."""
        return 1 / (1 + math.exp(-self.k_G * 10.0 * (G - self.G_th)))

    def time_probability(self, T):
        """Exponential decay based on time elapsed. Smaller (recent) is better."""
        return math.exp(-self.lambda_T * T / self.T_th)

    def compute_accept_prob(self, Q, G):
        """Calculate input probability to determine whether accepting a new keyframe."""
        P_Q = self.quality_probability(Q)
        P_G = self.gain_probability(G)

        acc_prob = P_Q * P_G
        return acc_prob

    def compute_keep_prob(self, Q, R, G, T):
        """Calculate posterior probability for a keyframe."""
        P_Q = self.quality_probability(Q)
        # P_R = self.redundancy_probability(R)
        P_G = self.gain_probability(G)
        P_T = self.time_probability(T)

        keep_prob = P_Q * P_G * P_T + 1e-6
        
        return keep_prob

    def update_keyframes(self, submap, graph, timestamps, descriptors, iqa_scores, info_redu, info_gain):
        if graph.get_num_node() == 0:
            for img_name in submap['frames']:
                curr_node = ImageNode(img_name, None, None, descriptors[img_name], timestamps[img_name][0], None, None, None, None, None, None, None)
                curr_node.iqa_score = iqa_scores[img_name][0]
                graph.add_node(curr_node)
        else:
            db_descriptors = np.array([node.get_descriptor() for node in graph.nodes.values()], dtype=np.float32)
            for img_name in submap['frames']:
                curr_node = ImageNode(img_name, None, None, descriptors[img_name], timestamps[img_name][0], None, None, None, None, None, None, None)
                curr_node.iqa_score = iqa_scores[img_name][0]

                # Find the closest node in the graph
                query_descriptor = curr_node.get_descriptor().reshape(1, -1)
                dis, pred = perform_knn_search(db_descriptors, query_descriptor, query_descriptor.shape[1], [1])
                for idx, node in enumerate(graph.nodes.values()):
                    if idx == pred[0][0]:
                        closest_node = node
                        print(f"VPR: {closest_node.id} -> {curr_node.id}")
                        break

                # Determine whether to add new frame
                acc_prob = self.compute_accept_prob(
                    curr_node.iqa_score, 
                    info_gain[(curr_node.id, closest_node.id)] # how much information is gained by curr_node
                )
                print(f"{curr_node.id} with accept probability {acc_prob:.3f} w.r.t. {closest_node.id}")
                graph.add_node(curr_node)

                # Add new frame to the graph
                edge_info = {
                    'R': info_redu[(closest_node.id, curr_node.id)],
                    'G': info_gain[(closest_node.id, curr_node.id)],
                    'R_inv': info_redu[(curr_node.id, closest_node.id)],
                    'G_inv': info_gain[(curr_node.id, closest_node.id)],
                    'dt': curr_node.time - closest_node.time,
                }
                closest_node.add_edge(curr_node, edge_info)
            
            # Check whether old keyframe should be deleted
            for db_node in graph.nodes.values():
                if not db_node.edges:
                    continue
                
                # A node should be deleted if it has much redudancy information and less new information
                # max_R = max([edge[1]['R'] for edge in db_node.edges]) # the maximum information redudancy of db_node
                # min_G = min([edge[1]['G'] for edge in db_node.edges]) # the minimum information gain of db_node
                # print(f"{db_node.id}: max_R={max_R:.3f}, min_G={min_G:.3f}")
                # TODO:
                # if max_R > self.R_th and min_G < self.G_th:
                #     print(f"Delete node {db_node.id}")
                #     graph.nodes.pop(db_node.id)
                #     graph.edges.pop(db_node.id)

                # min_G = min([edge[1]['G'] for edge in db_node.edges]) # the minimum information gain of db_node

                P_keep = min([
                    self.compute_keep_prob(db_node.iqa_score, edge[1]['R'], edge[1]['G'], edge[1]['dt'])
                    for edge in db_node.edges
                ])
                
                for edge in db_node.edges:
                    P_Q = self.quality_probability(db_node.iqa_score)
                    P_R = self.redundancy_probability(edge[1]['R'])
                    P_G = self.gain_probability(edge[1]['G'])
                    P_T = self.time_probability(edge[1]['dt'])
                    # P = P_Q * P_R * P_G * P_T + 1e-3
                    P = P_Q * P_G * P_T + 1e-3
                    print(f"{db_node.id} -> {edge[0].id} with keep probability {P:.3f}")                    
                    print(f"Q: {db_node.iqa_score}, R: {edge[1]['R']}, G: {edge[1]['G']}, dT: {edge[1]['dt']}")
                    print(f"PQ: {P_Q:.3f}, PR: {P_R:.3f}, PG: {P_G:.3f}, PT: {P_T:.3f}")
                print(f"P_keep: {P_keep:.3f}")

                print()            
                input()

    def select_keyframes(self, timestamps, descriptors, iqa_scores, info_redu, info_gain, submap_database, max_frames=100):
        """
        Main method to select keyframes from provided data.
        timestamps, descriptors, iqa_scores, info_redu, info_gain: metadata dictionaries
        submap_database: list of submap dicts containing frame names
        """

        # Graph to store keyframes and their overlapping relationships
        graph = ImageGraph(map_root=None)
        
        # Process each submap
        for submap in submap_database:
            self.update_keyframes(submap, graph, timestamps, descriptors, iqa_scores, info_redu, info_gain)

        for key in graph.keys():
            if key[0] not in keyframes:
                keyframes.append(key[0])
                
            if key[1] not in keyframes:
                keyframes.append(key[1])

        return keyframes

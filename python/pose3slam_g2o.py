#! /usr/bin/env python

"""
 * @file Pose3SLAMExample_initializePose3.cpp
 * @brief A 3D Pose SLAM example that reads input from g2o, and initializes the
 *  Pose3 using InitializePose3
 * @date Jan 17, 2019
 * @author Vikrant Shah based on CPP example by Luca Carlone
"""

import argparse
import matplotlib.pyplot as plt
import numpy as np
import gtsam
import time
from utils.gtsam_pose_graph import PoseGraph

# Using GTSAM
def optimize_pose_graph(graph, initial, verbose=False):
    """
    Optimizes a pose graph using the Levenberg-Marquardt algorithm.

    This function adds a prior factor to the first key in the initial estimate to anchor the graph,
    then optimizes the graph to minimize the error.

    Args:
        graph (gtsam.NonlinearFactorGraph): The pose graph containing factors (constraints).
        initial (gtsam.Values): Initial estimates for the variables (poses) in the graph.

    Returns:
        gtsam.Values: The optimized values (poses) after the optimization process.
    """    
    # Set up the optimizer
    params = gtsam.LevenbergMarquardtParams()
    if verbose:
        params.setVerbosity("Termination")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    
    return result

def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array."""
    return np.array([x, y, z, a, b, c], dtype=float)

def parser_arugments():
    parser = argparse.ArgumentParser(
        description="A 3D Pose SLAM example that reads input from g2o, and "
        "initializes Pose3")
    parser.add_argument('-i', '--input', help='input file g2o format')
    parser.add_argument('-o', '--output', help="the path to the output file with optimized graph")
    parser.add_argument("-p", "--plot", action="store_true", help="Flag to plot results")
    parser.add_argument("--viz", action="store_true", help="Only visualize graph")
    args = parser.parse_args()
    return args

def main():
    """Main runner."""
    args = parser_arugments()
    g2o_file = args.input

    is3D = True
    graph, initial = gtsam.readG2o(g2o_file, is3D)
    print(f"Graph Info: ---------------------")
    print(f"Number of factors: {graph.size()}")
    print(f"Number of variables: {len(graph.keyVector())}")

    # Add prior factor on each disconnected graph
    priorModel = gtsam.noiseModel.Diagonal.Variances(vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))
    
    start_time = time.time()
    comp_graph_keys = PoseGraph.find_connected_components(graph)
    print(f"Search connected components costs time: {time.time() - start_time:.3f}s")
    for comp_id, comp_keys in enumerate(comp_graph_keys):
        init_estimate = initial.atPose3(comp_keys[0])
        graph.add(gtsam.PriorFactorPose3(comp_keys[0], init_estimate, priorModel))
        print(f"Add prior factor: {comp_keys[0]} to the {comp_id} subgraph with node number {len(comp_keys)}")

    if args.viz:
        result = initial
        print("initial error = ", graph.error(initial))
    else:
        result = optimize_pose_graph(graph, initial, True)
        print("Optimization complete")
        print("initial error = ", graph.error(initial))
        print("final error = ", graph.error(result))

    if args.output is None:
        # print("Final Result:\n{}".format(result))
        pass
    else:
        outputFile = args.output
        print("Writing results to file: ", outputFile)
        graphNoKernel, _ = gtsam.readG2o(g2o_file, is3D)
        gtsam.writeG2o(graphNoKernel, result, outputFile)
        print("Done!")

    if args.plot:
        PoseGraph.plot_pose_graph(None, graph, result, mode='3d')

if __name__ == "__main__":
    main()

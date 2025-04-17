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
import copy

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

    ##### Load an exiting G2O file
    is3D = True
    graph, initial = gtsam.readG2o(g2o_file, is3D)
    print(f"Graph Info: ---------------------")
    print(f"Number of factors: {graph.size()}")
    print(f"Number of variables: {len(graph.keyVector())}")

    # Add prior factor on each disconnected graph
    # Orientation and Translation Variance
    priorModel = gtsam.noiseModel.Diagonal.Sigmas(np.array([np.deg2rad(1.0)] * 3 + [0.1] * 3) / 1e2)

    start_time = time.time()
    comp_graph_keys = PoseGraph.find_connected_components(graph)
    print(f"----- Search connected components costs: {time.time() - start_time:.3f}s")
    for comp_id, comp_keys in enumerate(comp_graph_keys):
        init_estimate = initial.atPose3(comp_keys[0])
        graph.add(gtsam.PriorFactorPose3(comp_keys[0], init_estimate, priorModel))
        print(f"Add prior factor: {comp_keys[0]} to the {comp_id} subgraph with node number {len(comp_keys)}")

    ##### Optimize the graph
    if args.viz:
        result = initial
        print("initial error = ", graph.error(initial))
        if args.plot:
            PoseGraph.plot_pose_graph(None, graph, [initial, initial], ['Before PGO', 'Before PGO'], mode='3d')
    else:
        start_time = time.time()
        result = PoseGraph.optimize_pose_graph_with_LM(
            graph, initial, 
            verbose=True, 
            robust_kernel=True
        )
        print(f"----- Optimization costs: {time.time() - start_time:.3f}s")
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
            PoseGraph.plot_pose_graph(None, graph, [initial, result], ['Before PGO', 'After PGO'], mode='3d')

if __name__ == "__main__":
    main()

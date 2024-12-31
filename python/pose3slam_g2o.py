#! /usr/bin/env python

"""
 * @file Pose3SLAMExample_initializePose3.cpp
 * @brief A 3D Pose SLAM example that reads input from g2o, and initializes the
 *  Pose3 using InitializePose3
 * @date Jan 17, 2019
 * @author Vikrant Shah based on CPP example by Luca Carlone
"""
# pylint: disable=invalid-name, E1101

from __future__ import print_function

import argparse

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

import gtsam
from gtsam.utils import plot

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
    args = parser.parse_args()
    return args

def main():
    """Main runner."""
    args = parser_arugments()
    g2o_file = args.input

    is3D = True
    graph, initial = gtsam.readG2o(g2o_file, is3D)

    # Add Prior on the first key
    print("Adding prior to g2o file ")
    priorModel = gtsam.noiseModel.Diagonal.Variances(vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))
    firstKey = initial.keys()[0]
    init_estimate = initial.atPose3(firstKey)
    graph.add(gtsam.PriorFactorPose3(firstKey, init_estimate, priorModel))

    # Set up the optimizer
    params = gtsam.LevenbergMarquardtParams()
    params.setVerbosity("Termination")  # this will show info about stopping conds
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial, params)
    result = optimizer.optimize()
    # result = initial
    # Output the results
    print("Optimization complete")
    print("initial error = ", graph.error(initial))
    print("final error = ", graph.error(result))

    if args.output is None:
        print("Final Result:\n{}".format(result))
    else:
        outputFile = args.output
        print("Writing results to file: ", outputFile)
        graphNoKernel, _ = gtsam.readG2o(g2o_file, is3D)
        gtsam.writeG2o(graphNoKernel, result, outputFile)
        print("Done!")

    if args.plot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        resultPoses = gtsam.utilities.allPose3s(result)
        x_coords = [resultPoses.atPose3(i).translation()[0] for i in range(resultPoses.size())]
        y_coords = [resultPoses.atPose3(i).translation()[1] for i in range(resultPoses.size())]
        z_coords = [resultPoses.atPose3(i).translation()[2] for i in range(resultPoses.size())]
        plt.plot(x_coords, y_coords, z_coords, 'o', color='b', label='Est. Trajectory')

        for key in graph.keyVector():
            factor = graph.at(key)
            if isinstance(factor, gtsam.BetweenFactorPose3):
                key1, key2 = factor.keys()
                tsl1 = result.atPose3(key1).translation()
                tsl2 = result.atPose3(key2).translation()
                plt.plot([tsl1[0], tsl2[0]], [tsl1[1], tsl2[1]], [tsl1[2], tsl2[2]], '.-', color='g')

        ax.set_xlabel('X [m]')
        ax.set_ylabel('Y [m]')
        ax.set_zlabel('Z [m]')
        ax.view_init(elev=55, azim=60)
        plt.tight_layout()
        plt.axis('equal')
        plt.show()

if __name__ == "__main__":
    main()

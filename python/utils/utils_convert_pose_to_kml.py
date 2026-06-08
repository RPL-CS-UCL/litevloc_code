#!/usr/bin/env python

import os
import sys
import math
import argparse
import pathlib
import numpy as np
import pymap3d as pm
import simplekml

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from python.point_graph import PointGraphLoader as GraphLoader
from python.point_graph import PointGraph
from python.utils.utils_setting_color_font import acquire_color_palette

PALLETE = acquire_color_palette()  # Call function to get color palette

def save_coords_to_kml(directory, filename, coords, graph_id):
    kml = simplekml.Kml()
    kml.document.name = directory.split('/')[-1]
    lin = kml.newlinestring(name=directory.split('/')[-1], description='GPS trajectory', coords=coords)
    lin.style.linestyle.color = simplekml.Color.rgb(
        int(PALLETE[graph_id][0] * 255),
        int(PALLETE[graph_id][1] * 255),
        int(PALLETE[graph_id][2] * 255)
    )
    lin.style.linestyle.width = 6
    kml.save(os.path.join(directory, filename))

def read_trav_graph_from_files(map_path):
    map_root = pathlib.Path(map_path)
    point_graph = GraphLoader.load_data(map_root, edge_type='trav')
    print(f"Loading Traversability Graph: {str(point_graph)}")
    return point_graph

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--map_path", type=str, default="matterport3d", help="Path to map directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    point_graph = read_trav_graph_from_files(args.map_path)

    ##### Step 1: Compute the transformation matrix between ENU to local world frame
    all_coords = []
    valid_pairs = []  # Stores (local_pose, GPS) pairs for transformation
    latitude_origin, longitude_origin, altitude_origin = 0.0, 0.0, 0.0
    origin_set = False

    # First pass: Collect valid GPS-pose pairs and set ENU origin
    for node in point_graph.nodes.values():
        gps_data = node.gps_data
        if gps_data is not None:
            lat, lon = gps_data[:2]
            alt = gps_data[2] if len(gps_data) > 2 else 0.0
            if math.isnan(alt):
                alt = 0.0
            if not any(math.isnan(v) for v in (lat, lon, alt)):
                if not origin_set:
                    latitude_origin, longitude_origin, altitude_origin = lat, lon, alt
                    origin_set = True
                valid_pairs.append((node.trans, (lat, lon, alt)))

    if not origin_set:
        raise ValueError("Not enough valid GPS data to compute transformation matrix")

    # Compute transformation matrix from local coordinate to ENU if sufficient data
    T_ini = np.eye(4)
    if len(valid_pairs) >= 2:
        enu_points, local_points = [], []
        for pose, gps in valid_pairs:
            e, n, u = pm.geodetic2enu(gps[0], gps[1], gps[2], 
                                    latitude_origin, longitude_origin, altitude_origin)
            enu_points.append([e, n, u])
            local_points.append(pose[:3] if len(pose) >= 3 else [0,0,0])
        
        enu_arr = np.array(enu_points)
        local_arr = np.array(local_points)
        
        # Kabsch algorithm to find optimal rotation and translation
        centroid_enu = np.mean(enu_arr, axis=0)
        centroid_local = np.mean(local_arr, axis=0)
        centered_enu = enu_arr - centroid_enu
        centered_local = local_arr - centroid_local
        
        H = centered_local.T @ centered_enu
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # Ensure right-handed coordinate system
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        
        t = centroid_enu - R @ centroid_local
        T_ini[:3, :3] = R
        T_ini[:3, 3] = t

    print(T_ini)

    ##### Step 2: Handle each subgraph
    subgraphs = []
    last_time = 0
    for node in point_graph.nodes.values():
        if abs(node.time - last_time) > 3600.0 * 24:
            subgraphs.append(PointGraph(point_graph.map_root, point_graph.edge_type))
        current_graph = subgraphs[-1]
        current_graph.nodes[node.id] = node
        last_time = node.time

    # Process each subgraph to convert local poses to ENU and generate KML
    subgraphs = [point_graph]
    for graph_id, graph in enumerate(subgraphs):
        all_coords = []
        # Second pass: Apply transformation and convert to geographic
        for id, node in enumerate(graph.nodes.values()):
            tx, ty, tz = (T_ini[:3, :3] @ node.trans + T_ini[:3, 3])
            lat, lon, _ = pm.enu2geodetic(
                tx, ty, tz, 
                latitude_origin, longitude_origin, altitude_origin
            )
            
            if id % 10 == 0:
                all_coords.append((lon, lat))

        # Save KML file
        kml_path = pathlib.Path(args.map_path) / 'preds/kml'
        kml_path.mkdir(parents=True, exist_ok=True)
        save_coords_to_kml(
            str(kml_path),
            f'gps_traj_{graph_id}.kml',
            all_coords,
            graph_id % len(PALLETE)
        )
        print(f"Saved trajectory {graph_id} with {len(all_coords)} points")
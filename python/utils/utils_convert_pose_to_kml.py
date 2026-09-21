#!/usr/bin/env python

import os
import sys
import argparse
import pathlib
import numpy as np
import simplekml

sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
from python.point_graph import PointGraphLoader as GraphLoader
from python.point_graph import PointGraph
from python.utils.utils_setting_color_font import acquire_color_palette
from python.utils.utils_gps_align import collect_gps_pairs, compute_local_to_enu, local_to_geodetic

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
    nodes = list(point_graph.nodes.values())
    valid_pairs, origin = collect_gps_pairs([n.trans for n in nodes], [n.gps_data for n in nodes])
    if origin is None:
        raise ValueError("Not enough valid GPS data to compute transformation matrix")
    T_ini = compute_local_to_enu(valid_pairs, origin)

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
        geo = local_to_geodetic(T_ini, [n.trans for n in graph.nodes.values()], origin)
        for id, (lat, lon, _) in enumerate(geo):
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
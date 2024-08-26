"""
python camera_keyframe_select.py \
--path_dataset /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_17DRP5sb8fy/out_general/ \
--out_dir /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_17DRP5sb8fy/out_map \
--grid_resolution 0.1 --num_select_cam 15 --coverage_threshold 0.9

python camera_keyframe_select.py \
--path_dataset /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_EDJbREhghzL/out_general/ \
--out_dir /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_EDJbREhghzL/out_map \
--grid_resolution 0.1 --num_select_cam 25 --coverage_threshold 0.9

python camera_keyframe_select.py \
--path_dataset /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_B6ByNegPMK/out_general/ \
--out_dir /Rocket_ssd/dataset/data_topo_loc/matterport3d/out_B6ByNegPMK/out_map \
--grid_resolution 0.1 --num_select_cam 75 --coverage_threshold 0.9

python camera_keyframe_select.py \
--path_dataset /Rocket_ssd/dataset/data_topo_loc/ucl_campus/out/out_general/ \
--out_dir /Rocket_ssd/dataset/data_topo_loc/ucl_campus/out/out_map/ \
--grid_resolution 1.0 --num_select_cam 70 --coverage_threshold 0.8
"""

import os

import numpy as np
import argparse
import CMap2D
from map2d import gridshow
import matplotlib.pyplot as plt
import copy

from PIL import Image
import open3d as o3d

from pycpptools.src.python.utils_sensor.tools_depth_image import depth_image_to_point_cloud, transform_point_cloud
from pycpptools.src.python.utils_math.tools_eigen import convert_vec_to_matrix, compute_relative_dis_TF
from pycpptools.src.python.utils_visualization.tools_vis_camera import plot_cameras, plot_connected_cameras

def parse_arguments():
    parser = argparse.ArgumentParser(description="Select a fixed number of camera keyframes to maximize region coverage.")
    parser.add_argument('--path_dataset', type=str, required=True, help="Path to the dataset file")
    parser.add_argument('--out_dir', type=str, required=True, help="Path to the stored file")
    parser.add_argument('--thre_trans', type=float, default=0.1, help="Translation threshold")
    parser.add_argument('--thre_rot', type=float, default=1, help="Rotation threshold")
    parser.add_argument('--num_select_cam', type=int, default=10, help="Number of selected cameras")
    parser.add_argument('--grid_resolution', type=float, default=1.0, help="Resolution of the grid")
    parser.add_argument('--coverage_threshold', type=float, default=0.95, help="Coverage threshold")
    parser.add_argument('--viz', action='store_true', help="Visualize the camera poses")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    return parser.parse_args()

def crop_points(points, args):
    if 'matterport3d' in args.path_dataset:
        """Simu Matterport3d"""
        max_depth = 7.0
        points = points[(points[:, 1] > -0.3) & (points[:, 1] < 0.3)]
        points = points[(points[:, 2] < max_depth)]
        points = points[np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2) > 0.1]
    else:
        """Real Anymal"""
        max_depth = 8.0
        points = points[(points[:, 1] > -1.5) & (points[:, 1] < -0.7)]
        points = points[(points[:, 2] < max_depth)]
        points = points[np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2) > 0.5]    
    return points

def check_connection(grid_map, reso, pose_i, pose_j):
    goal = grid_map.xy_to_ij(pose_i[:2].reshape(1, 2))
    start = grid_map.xy_to_ij(pose_j[:2].reshape(1, 2))
    grid = grid_map.dijkstra(goal[0], inv_value=1000, connectedness=8)
    path, jumps = CMap2D.path_from_dijkstra_field(grid, start[0], connectedness=8)
    # physical length
    phy_length = np.linalg.norm(pose_i[:2] - pose_j[:2])
    # shortest path length length
    path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1) * reso)
    """DEBUG(gogojjh):"""
    # print(path)
    # print(start, goal)
    # fig = plt.figure()
    # gridshow(grid)
    # plt.plot(path[:, 0], path[:, 1], 'b-')
    # plt.show()
    # print(f"Path Length: {path_length}, Phy Length: {phy_length}")
    """"""
    if ((path[-1] == goal).all()) and (path_length < 20.0) and (path_length <= phy_length * 2.0):
        return path_length
    else:
        return None

class KeyFrameSelect:
    def __init__(self, args):
        self.args = args

        # Load the dataset         
        if 'matterport3d' in args.path_dataset:
            """Simu Matterport3d"""
            self.start_indice = 0
        else:
            """Real Anymal"""
            self.start_indice = 12100

        path_pose = os.path.join(args.path_dataset, 'poses.txt')
        self.poses = np.loadtxt(path_pose)  # time, tx, ty, tz, qx, qy, qz, qw
        path_intrinsics = os.path.join(args.path_dataset, 'intrinsics.txt')
        self.intrinsics = np.loadtxt(path_intrinsics)[0, :]  # fx, fy, cx, cy, width, height

        # Maually select valid poses according to the relative pose threshold
        self.valid_pose = [0] * len(self.poses)
        for i in range(self.start_indice, len(self.poses)):
            if i == self.start_indice: 
                self.valid_pose[i] = 1
            else:
                T0 = convert_vec_to_matrix(self.poses[i-1, 1:4], self.poses[i-1, 4:])
                T1 = convert_vec_to_matrix(self.poses[i, 1:4], self.poses[i, 4:])
                dis_trans, dis_angle = compute_relative_dis_TF(T0, T1)
                if dis_trans > args.thre_trans or dis_angle > args.thre_rot: 
                    self.valid_pose[i] = 1

        self.img_size = (int(self.intrinsics[4]), int(self.intrinsics[5])) # width, height
        self.K = np.array([[self.intrinsics[0], 0, self.intrinsics[2]], [0, self.intrinsics[1], self.intrinsics[3]], [0, 0, 1]])
        self.num_select_cam = self.args.num_select_cam

        # Create a grid map
        min_x, max_x = np.min(self.poses[:, 1]) - 1, np.max(self.poses[:, 1]) + 1
        min_y, max_y = np.min(self.poses[:, 2]) - 1, np.max(self.poses[:, 2]) + 1
        width = int((abs(max_x - min_x)) / self.args.grid_resolution)
        height = int((abs(max_y - min_y)) / self.args.grid_resolution)
        occupancy = np.zeros((width, height), dtype=np.float32) # x, y
        origin = (min_x, min_y)

        self.inc_grid_map = CMap2D.CMap2D()
        self.inc_grid_map.from_array(occupancy, origin, self.args.grid_resolution)

        self.full_grid_map = CMap2D.CMap2D()
        self.full_grid_map.from_array(occupancy, origin, self.args.grid_resolution)

        # Store all depth points
        self.world_depth_points_dict = {}

    def store_depth_points(self):
        for indice, pose in enumerate(self.poses):
            if self.valid_pose[indice] == 0: continue
            if self.args.viz:
                print(f"Storing Depth Points: {indice}")
            path_depth = os.path.join(self.args.path_dataset, f'seq/{indice:06d}.depth.png')
            depth_img = np.array(Image.open(path_depth), dtype=np.float32) / 1000.0
            points = depth_image_to_point_cloud(depth_img, self.K, self.img_size)
            points = crop_points(points, self.args)

            trans, quat = pose[1:4], pose[4:]
            T_w_cam = convert_vec_to_matrix(trans, quat)
            points_world = transform_point_cloud(points, T_w_cam)
            depth_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world))
            depth_cloud = depth_cloud.voxel_down_sample(voxel_size=self.args.grid_resolution / 5)
            self.world_depth_points_dict[indice] = np.asarray(depth_cloud.points)

    def build_occupancy_map(self):
        for cam_id, points in self.world_depth_points_dict.items():
            self.update_covered_space(self.full_grid_map, points, 0.05)

    def get_num_occupancy(self, grid_map):
        return np.sum(grid_map.occupancy() >= 0.999)

    def calculate_new_covered_area(self, grid_map, points):
        num_new_covered = 0
        points_ij = grid_map.xy_to_ij(points[:, :2], clip_if_outside=True).astype(int)
        for point_ij in points_ij:
            if grid_map._occupancy[point_ij[0], point_ij[1]] == 0.0:
                num_new_covered += 1
        return num_new_covered

    def update_covered_space(self, grid_map, points, value):
        points_ij = grid_map.xy_to_ij(points[:, :2], clip_if_outside=True).astype(int)
        for point_ij in points_ij:
            grid_map._occupancy[point_ij[0], point_ij[1]] = min(grid_map._occupancy[point_ij[0], point_ij[1]] + value, 1.0)

    def select_greedy(self):
        select_cam_id = []
        all_cam_id = [k for k in self.world_depth_points_dict.keys()]
        num_total_occu = self.get_num_occupancy(self.full_grid_map)
        while True:
            remain_occu_ratio = self.get_num_occupancy(self.inc_grid_map) / num_total_occu
            print(f"Remain Occupancy Ratio: {remain_occu_ratio:.5f}, Num of Select Cameras: {len(select_cam_id)}")
            if remain_occu_ratio > self.args.coverage_threshold or len(select_cam_id) > self.num_select_cam: 
                break

            random_cam_id = np.random.choice(all_cam_id, size=min(20, len(all_cam_id)), replace=False)
            num_new_covered_list = []
            for cam_id in random_cam_id:
                points_world = self.world_depth_points_dict[cam_id]
                num_new_covered = self.calculate_new_covered_area(self.inc_grid_map, points_world)
                num_new_covered_list.append(num_new_covered)
            
            optimal_cam_id = random_cam_id[np.argmax(num_new_covered_list)]
            select_cam_id.append(optimal_cam_id)
            all_cam_id.remove(optimal_cam_id)
            self.update_covered_space(self.inc_grid_map, self.world_depth_points_dict[optimal_cam_id], 0.2)

            if self.args.debug and self.args.viz:
                print(f"Optimal Camera ID: {optimal_cam_id}, Num of New Covered: {np.max(num_new_covered_list)}")
                gridshow(self.inc_grid_map.occupancy(), extent=self.inc_grid_map.get_extent_xy())
                plt.xlabel('X')
                plt.ylabel('Y')
                plt.show()

        result = np.array(select_cam_id)
        result.sort()
        return result

def main():
    args = parse_arguments()
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(os.path.join(args.out_dir, 'seq'), exist_ok=True)

    print('Creating KeyFrameSelect Object')
    keyframe_select = KeyFrameSelect(args)
    print(f'Number of cameras to be loaded: {len(keyframe_select.poses)}, Maximiumly Selected cameras: {keyframe_select.num_select_cam}')

    print('Storing Depth Points ...')
    keyframe_select.store_depth_points()

    print(f'Building Full Occupancy Map with {len(keyframe_select.world_depth_points_dict)} cameras')
    keyframe_select.build_occupancy_map()

    print('Selecting Keyframes ...')
    select_cam_id = keyframe_select.select_greedy()

    fig = plt.figure()
    ax = fig.add_subplot(121)
    gridshow(keyframe_select.full_grid_map.occupancy(), extent=keyframe_select.full_grid_map.get_extent_xy())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Full Occupancy Map')
    ax = fig.add_subplot(122)
    gridshow(keyframe_select.inc_grid_map.occupancy(), extent=keyframe_select.inc_grid_map.get_extent_xy())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Inc Occupancy Map')
    if args.viz:
        plt.show()
    else:
        plt.savefig(os.path.join(args.out_dir, 'occupancy_map.png'))
    print(f'Selected Cameras: ' + ' '.join([str(i) for i in select_cam_id]))

    """Save the selected keyframes"""
    new_cam_id = 0
    for cam_id in select_cam_id:
        path_img = os.path.join(args.path_dataset, f'seq/{cam_id:06d}.color.jpg')
        new_path_img = os.path.join(args.out_dir, f'seq/{new_cam_id:06d}.color.jpg')
        os.system(f'cp {path_img} {new_path_img}')

        path_img = os.path.join(args.path_dataset, f'seq/{cam_id:06d}.depth.png')
        new_path_img = os.path.join(args.out_dir, f'seq/{new_cam_id:06d}.depth.png')
        os.system(f'cp {path_img} {new_path_img}')

        path_img = os.path.join(args.path_dataset, f'seq/{cam_id:06d}.semantic.png')
        new_path_img = os.path.join(args.out_dir, f'seq/{new_cam_id:06d}.semantic.png')
        os.system(f'cp {path_img} {new_path_img}')
        new_cam_id += 1
    new_poses = keyframe_select.poses[select_cam_id, :]
    new_intrinsics = np.loadtxt(os.path.join(args.path_dataset, 'intrinsics.txt'))[select_cam_id, :]
    np.savetxt(os.path.join(args.out_dir, 'poses.txt'), new_poses, fmt='%.5f')
    np.savetxt(os.path.join(args.out_dir, 'intrinsics.txt'), new_intrinsics, fmt='%.5f')
    np.savetxt(os.path.join(args.out_dir, 'selected_keyframes_rawid.txt'), select_cam_id.reshape(-1, 1), fmt='%d')

    """Create edge list"""
    edge_list = np.empty((0, 3), dtype=np.float32)
    dij_map = copy.copy(keyframe_select.full_grid_map)
    valid_ij = np.where(dij_map.occupancy() < 0.999)
    for va in valid_ij:
        dij_map._occupancy[va[0], va[1]] = 0.0
    for i in range(len(select_cam_id)):
        for j in range(i+1, len(select_cam_id)):
            pose_i, pose_j = keyframe_select.poses[select_cam_id[i], 1:], keyframe_select.poses[select_cam_id[j], 1:]
            weight = check_connection(dij_map, args.grid_resolution, pose_i, pose_j)
            if weight is not None:
                edge_list = np.vstack((edge_list, np.array([i, j, weight]).reshape(1, 3)))
    np.savetxt(os.path.join(args.out_dir, 'edge_list.txt'), edge_list, fmt='%.5f')
    
    fig = plt.figure()
    plot_cameras(keyframe_select.poses[:, 1:], 10, title='Raw Camera Observations', ax=fig.add_subplot(221))
    plot_cameras(keyframe_select.poses[select_cam_id, 1:], 1, title='Selected Camera Observations', ax=fig.add_subplot(222))
    plot_connected_cameras(keyframe_select.poses[select_cam_id, 1:], edge_list, title='Selected Camera Obs With Connection', ax=fig.add_subplot(223))
    ax = fig.add_subplot(224)
    gridshow(keyframe_select.full_grid_map.occupancy(), extent=keyframe_select.full_grid_map.get_extent_xy())
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.title('Full Occupancy Map')    
    if args.viz:
        plt.show()
    else:
        plt.savefig(os.path.join(args.out_dir, 'camera_poses.png'))

if __name__ == "__main__":
    main()

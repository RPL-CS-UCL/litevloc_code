import os
import sys
import numpy as np
import argparse
from CMap2D import CMap2D
from map2d import gridshow
import matplotlib.pyplot as plt

from PIL import Image
import open3d as o3d

from pycpptools.src.python.utils_sensor.tools_depth_image import depth_image_to_point_cloud, transform_point_cloud
from pycpptools.src.python.utils_math.tools_eigen import convert_vec_to_matrix
from pycpptools.src.python.utils_visualization.tools_vis_camera import plot_cameras
from pycpptools.src.python.utils_sensor.camera_pinhole import CameraPinhole

def parse_arguments():
    parser = argparse.ArgumentParser(description="Select a fixed number of camera keyframes to maximize region coverage.")
    parser.add_argument('--path_dataset', type=str, required=True, help="Path to the dataset file")
    parser.add_argument('--select_cam_ratio', type=float, default=0.1, help="Sample ratio for cameras")
    parser.add_argument('--grid_resolution', type=float, default=1.0, help="Resolution of the grid")
    parser.add_argument('--coverage_threshold', type=float, default=0.95, help="Coverage threshold")
    parser.add_argument('--viz', action='store_true', help="Visualize the camera poses")
    parser.add_argument('--debug', action='store_true', help="Debug mode")
    return parser.parse_args()

def crop_points(points):
    """Real Anymal"""
    # points = points[(points[:, 1] > -2.0) & (points[:, 1] < -0.5)]
    # points = points[(points[:, 2] < 8.0)]
    # points = points[np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2) > 0.5]    

    """Simu Matterport3d"""
    points = points[(points[:, 1] > -1.0) & (points[:, 1] < 0.5)]
    points = points[(points[:, 2] < 5.0)]
    points = points[np.sqrt(points[:, 0]**2 + points[:, 1]**2 + points[:, 2]**2) > 0.1]
    return points

class KeyFrameSelect:
    def __init__(self, args):
        self.args = args

        # Load the dataset         
        path_pose = os.path.join(args.path_dataset, 'poses.txt')
        self.poses = np.loadtxt(path_pose)[:, 1:] # tx, ty, tz, qx, qy, qz, qw
        path_intrinsics = os.path.join(args.path_dataset, 'intrinsics.txt')
        intrinsics = np.loadtxt(path_intrinsics)[0, :] # fx, fy, cx, cy, width, height
        self.img_size = (int(intrinsics[4]), int(intrinsics[5])) # width, height
        self.K = np.array([[intrinsics[0], 0, intrinsics[2]], [0, intrinsics[1], intrinsics[3]], [0, 0, 1]])
        self.num_select_cam = int(len(self.poses) * self.args.select_cam_ratio)
        self.sample_step = 5

        # Create a grid map
        min_x, max_x = np.min(self.poses[:, 0]) - 1, np.max(self.poses[:, 0]) + 1
        min_y, max_y = np.min(self.poses[:, 1]) - 1, np.max(self.poses[:, 1]) + 1
        width = int((abs(max_x - min_x)) / self.args.grid_resolution)
        height = int((abs(max_y - min_y)) / self.args.grid_resolution)
        occupancy = np.zeros((width, height), dtype=np.float32) # x, y
        origin = (min_x, min_y)

        self.inc_grid_map = CMap2D()
        self.inc_grid_map.from_array(occupancy, origin, self.args.grid_resolution)

        self.full_grid_map = CMap2D()
        self.full_grid_map.from_array(occupancy, origin, self.args.grid_resolution)

        # Store all depth points
        self.world_depth_points_dict = {}

    def store_depth_points(self):
        for indice, pose in enumerate(self.poses):
            if indice % self.sample_step != 0: continue
            if self.args.viz:
                print(f"Storing Depth Points: {indice}")
            path_depth = os.path.join(self.args.path_dataset, f'seq/{indice:06d}.depth.png')
            depth_img = np.array(Image.open(path_depth), dtype=np.float32) / 1000.0
            points = depth_image_to_point_cloud(depth_img, self.K, self.img_size)
            points = crop_points(points)

            trans, quat = pose[:3], pose[3:]
            T_w_cam = convert_vec_to_matrix(trans, quat)
            points_world = transform_point_cloud(points, T_w_cam)
            depth_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points_world))
            depth_cloud = depth_cloud.voxel_down_sample(voxel_size=0.05)
            self.world_depth_points_dict[indice] = np.asarray(depth_cloud.points)

    def build_occupancy_map(self):
        for cam_id, points in self.world_depth_points_dict.items():
            self.update_covered_space(self.full_grid_map, points, 1.0)

    def get_num_occupancy(self, grid_map):
        return np.sum(grid_map.occupancy() >= 0.999)

    def calculate_covered_area(self, grid_map, points):
        num_new_covered = 0
        points_ij = grid_map.xy_to_ij(points[:, :2], clip_if_outside=True).astype(int)
        for point_ij in points_ij:
            if grid_map._occupancy[point_ij[0], point_ij[1]] < 1.0:
                num_new_covered += 1
        return num_new_covered

    def update_covered_space(self, grid_map, points, value):
        points_ij = grid_map.xy_to_ij(points[:, :2], clip_if_outside=True).astype(int)
        for point_ij in points_ij:
            grid_map._occupancy[point_ij[0], point_ij[1]] = value

    def run(self):
        select_cam_id = []
        all_cam_id = [k for k in self.world_depth_points_dict.keys()]
        num_total_occu = self.get_num_occupancy(self.full_grid_map)
        while True:
            remain_occu_ratio = self.get_num_occupancy(self.inc_grid_map) / num_total_occu
            print(f"Remain Occupancy Ratio: {remain_occu_ratio:.5f}, Num of Select Cameras: {len(select_cam_id)}")
            if remain_occu_ratio > self.args.coverage_threshold or len(select_cam_id) > self.num_select_cam: break

            random_cam_id = np.random.choice(all_cam_id, size=10, replace=False)
            num_new_covered_list = []
            for cam_id in random_cam_id:
                points_world = self.world_depth_points_dict[cam_id]
                num_new_covered = self.calculate_covered_area(self.inc_grid_map, points_world)
                num_new_covered_list.append(num_new_covered)
            
            optimal_cam_id = random_cam_id[np.argmax(num_new_covered_list)]
            select_cam_id.append(optimal_cam_id)
            all_cam_id.remove(optimal_cam_id)
            self.update_covered_space(self.inc_grid_map, self.world_depth_points_dict[optimal_cam_id], 1.0)

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
    
    print('Creating KeyFrameSelect Object')
    keyframe_select = KeyFrameSelect(args)
    print(f'Number of cameras: {len(keyframe_select.poses)}, Selected cameras: {keyframe_select.num_select_cam}')

    print('Storing Depth Points ...')
    keyframe_select.store_depth_points()

    print('Building Full Occupancy Map ...')
    keyframe_select.build_occupancy_map()
    if args.viz:
        gridshow(keyframe_select.full_grid_map.occupancy(), extent=keyframe_select.full_grid_map.get_extent_xy())
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Full Occupancy Map')
        plt.show()

    print('Selecting Keyframes ...')
    select_cam_id = keyframe_select.run()
    if args.viz:
        gridshow(keyframe_select.inc_grid_map.occupancy(), extent=keyframe_select.inc_grid_map.get_extent_xy())
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Inc Occupancy Map From Selected Cameras')
        plt.show()

    print(f'Selected Cameras: ' + ' '.join([str(i) for i in select_cam_id]))
    if args.viz:
        plot_cameras(keyframe_select.poses, 10, title='Raw Camera Observations')
        plot_cameras(keyframe_select.poses[select_cam_id], 1, title='Selected Camera Observations')
        plt.show()

    path_output = os.path.join(args.path_dataset, 'selected_keyframes.txt')
    np.savetxt(path_output, select_cam_id.reshape(-1, 1), fmt='%d')

if __name__ == "__main__":
    main()

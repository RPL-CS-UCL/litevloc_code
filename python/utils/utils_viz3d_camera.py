#! /usr/bin/env python
"""MapFree dataset visualization tool with camera frustums and images."""

"""Format of output Map-Free-Reloc dataset
Map_free_reloc/
	dataset_name/
		s00000/
			seq0/
				frame_000000.jpg
                frame_000000.(zed, xxx).png (optional)
			seq1/
				frame_000000.jpg
                frame_000000.(zed, xxx).png (optional)
				frame_000001.jpg
                frame_000001.(zed, xxx).png (optional)
			poses.txt (format: image_name qw qx qy qz tx ty tz) - transforms points in seq0/frame_000000.jpg into seq1/frame_X.jpg
			intrinsics.txt (format: image_name fx fy cx cy width height)
			timestamps.txt (format: image_name timestamp)
			gps_data.txt (format: image_name latitude longtitude attitude ...)
"""

"""Usage
# Visualize specific scenes
# python viz.py --dataset_dir /path/to/mapfree --scenes s00000 --cam_scale 0.2 --show_image
"""

"""MapFree dataset visualization tool with camera frustums and images."""

import argparse
import hashlib
import os
from glob import glob

import cv2
import numpy as np
import PIL.Image
import trimesh
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw, ImageFont

# Predefined color palette for consistent scene coloring (RGB tuples 0-255)
def spec(N):
    t = np.linspace(-510, 510, N)
    return np.round(np.clip(np.stack([-t, 510 - np.abs(t), t], axis=1), 0, 255)).astype("float32") / 255

SCENE_COLORS = spec(10)

# colormap: https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html
SCENE_COLORS[0] = [0, 1.0 * 152 / 255, 1.0 * 83 / 255]  # green
SCENE_COLORS[1] = [1.0 * 228 / 255, 1.0 * 53 / 255, 1.0 * 39 / 255]  # red
SCENE_COLORS[2] = [1.0 * 140 / 255, 1.0 * 3 / 255, 1.0 * 120 / 255]  # purple
SCENE_COLORS[3] = [0, 1.0 * 95 / 255, 1.0 * 170 / 255]  # blue
SCENE_COLORS[4] = [0.9290, 0.6940, 0.1250]
SCENE_COLORS[5] = [0.6350, 0.0780, 0.1840]
SCENE_COLORS[6] = [0.494, 0.184, 0.556]
SCENE_COLORS[7] = [0.850, 0.3250, 0.0980]

# Coordinate system transformation matrix (OpenGL convention)
OPENGL_MAT = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, 1]
])

def parse_arguments():
    """Parses command line arguments for dataset visualization.
    
    Returns:
        argparse.Namespace: Parsed arguments with dataset_dir and scenes.
    """
    parser = argparse.ArgumentParser(description='Visualize MapFree dataset scenes')
    parser.add_argument(
        '--dataset_dir', required=True,
        help='Root directory containing MapFree scenes'
    )
    parser.add_argument(
        '--scenes', nargs='+', default=['all'],
        help='Space-separated list of scenes to visualize (e.g. s00000 s00001)'
    )
    parser.add_argument(
        '--cam_size', type=float, default=0.2,
        help='Base size of camera frustums in meters'
    )
    parser.add_argument(
        '--show_image', action="store_true",
        help="Show image"
    )
    parser.add_argument(
        '--step', type=int, default=1,
        help="Sample step of data"
    )
    return parser.parse_args()

def load_scene_data(dataset_dir, target_scenes):
    """Loads specified scenes from dataset directory.
    
    Args:
        dataset_dir (str): Root directory containing scenes
        target_scenes (list): List of scene names or ['all']
    
    Returns:
        dict: Scene data containing intrinsics, poses and images
    """
    if 'all' in target_scenes or len(target_scenes) > 1:
        print("Show multiple scenes using poses_abs.txt")
        pose_file_name = 'poses_abs.txt'
        is_multi_frame = True
    else:
        print("Showing single scene using poses.txt")
        pose_file_name = 'poses.txt'
        is_multi_frame = False

    scene_paths = []
    if 'all' in target_scenes:
        scene_paths = sorted(glob(os.path.join(dataset_dir, "s*")))
    else:
        for scene in target_scenes:
            path = os.path.join(dataset_dir, scene)
            if os.path.exists(path):
                scene_paths.append(path)

    scene_data = {}
    for path in sorted(scene_paths):
        intr_path = os.path.join(path, 'intrinsics.txt')
        pose_path = os.path.join(path, pose_file_name)
        if not os.path.exists(pose_path) and 'poses_abs' in pose_path:
            pose_path = pose_path.replace('poses_abs', 'poses_abs_gt')
       
        scene_name = os.path.basename(path)
        scene_data[scene_name] = {
            'intrinsics': _load_intrinsics(intr_path),
            'poses': _load_poses(pose_path),
            'images': _collect_images(path)
        }

    return scene_data, is_multi_frame

def _load_intrinsics(filepath):
    """Loads camera intrinsics from text file."""
    intrinsics = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            frame_path = os.path.join(os.path.dirname(filepath), parts[0])
            intrinsics[frame_path] = np.array(list(map(float, parts[1:])))
    return intrinsics
def _load_poses(filepath):
    """Loads camera poses from text file (world-to-camera format)."""
    poses = {}
    if not os.path.exists(filepath) and 'poses_abs' in filepath:
        filepath = filepath.replace('poses_abs', 'poses_abs_gt')
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split()
            frame_path = os.path.join(os.path.dirname(filepath), parts[0])
            quat = list(map(float, parts[1:5]))
            trans = list(map(float, parts[5:8]))
            
            pose = np.eye(4)
            pose[:3, :3] = Rotation.from_quat(np.roll(quat, -1)).as_matrix()
            pose[:3, 3] = trans
            poses[frame_path] = pose
    return poses

def _collect_images(scene_path):
    """Collects all image paths in a scene."""
    images = []
    for seq in ['seq0', 'seq1', 'seq']:
        seq_path = os.path.join(scene_path, seq)
        if os.path.exists(seq_path):
            images.extend(glob(os.path.join(seq_path, '*.jpg')))
    images.sort()
    return images

def _get_scene_color(scene_name):
    """Generates consistent color for a scene using hash-based selection.
    
    Args:
        scene_name (str): Name of the scene
    
    Returns:
        tuple: Normalized RGB color (0-1 range)
    """
    hash_int = int(hashlib.md5(scene_name.encode()).hexdigest()[:8], 16)
    return tuple(c for c in SCENE_COLORS[hash_int % len(SCENE_COLORS)])

def _geotrf(Trf, pts, ncol=None, norm=False):
    """ Apply a geometric transformation to a list of 3-D points.

    H: 3x3 or 4x4 projection matrix (typically a Homography)
    p: numpy/torch/tuple of coordinates. Shape must be (...,2) or (...,3)

    ncol: int. number of columns of the result (2 or 3)
    norm: float. if != 0, the resut is projected on the z=norm plane.

    Returns an array of projected 2d points.
    """
    assert Trf.ndim >= 2
    if isinstance(Trf, np.ndarray):
        pts = np.asarray(pts)

    # adapt shape if necessary
    output_reshape = pts.shape[:-1]
    ncol = ncol or pts.shape[-1]

    # optimized code
    if Trf.ndim >= 3:
        n = Trf.ndim - 2
        assert Trf.shape[:n] == pts.shape[:n], 'batch size does not match'
        Trf = Trf.reshape(-1, Trf.shape[-2], Trf.shape[-1])

        if pts.ndim > Trf.ndim:
            # Trf == (B,d,d) & pts == (B,H,W,d) --> (B, H*W, d)
            pts = pts.reshape(Trf.shape[0], -1, pts.shape[-1])
        elif pts.ndim == 2:
            # Trf == (B,d,d) & pts == (B,d) --> (B, 1, d)
            pts = pts[:, None, :]

    if pts.shape[-1] + 1 == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf[..., :-1, :] + Trf[..., -1:, :]
    elif pts.shape[-1] == Trf.shape[-1]:
        Trf = Trf.swapaxes(-1, -2)  # transpose Trf
        pts = pts @ Trf
    else:
        pts = Trf @ pts.T
        if pts.ndim >= 2:
            pts = pts.swapaxes(-1, -2)

    if norm:
        pts = pts / pts[..., -1:]  # DONT DO /= BECAUSE OF WEIRD PYTORCH BUG
        if norm != 1:
            pts *= norm

    res = pts[..., :ncol].reshape(*output_reshape, ncol)
    return res

def _add_scene_cam(scene, scene_name, pose_w2c, color, image, focal, imsize, cam_size, show_image=True):
    """Adds a camera mesh to the scene using trimesh.
    
    Args:
        scene (trimesh.Scene): Target scene to add camera
        pose_w2c (np.array): World-to-camera pose matrix
        color (tuple): RGB camera color (0-1 range)
        image (np.array): Camera image for texture
        focal (float): Focal length in pixels
        imsize (tuple): Image dimensions (width, height)
        cam_size (float): Base size of camera frustum
    """

    ##### Add image    
    if image is not None:
        image = np.asarray(image)
        if image.dtype != np.uint8:
            image = np.uint8(255 * image)

    # Convert to camera-to-world pose with OpenGL convention
    pose_c2w = np.linalg.inv(pose_w2c)
    aspect_ratio = imsize[0] / imsize[1]

    # Create camera cone mesh
    height = max(cam_size/10, focal * cam_size / imsize[1])
    width = cam_size * 0.5**0.5
    cone = trimesh.creation.cone(width, height, sections=4)

    # Transform mesh to correct pose
    rot45 = np.eye(4)
    rot45[:3, :3] = Rotation.from_euler('z', np.deg2rad(45)).as_matrix()
    rot45[2, 3] = -height  # set the tip of the cone = optical center
    aspect_ratio = np.eye(4)
    aspect_ratio[0, 0] = imsize[0] / imsize[1]
    transform = pose_c2w @ OPENGL_MAT @ aspect_ratio @ rot45
    cam = trimesh.creation.cone(width, height, sections=4)  # , transform=transform)

    vertices = _geotrf(transform, cam.vertices[[4, 5, 1, 3]])
    faces = np.array([[0, 1, 2], [0, 2, 3], [2, 1, 0], [3, 2, 0]])
    img = trimesh.Trimesh(vertices=vertices, faces=faces)
    uv_coords = np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    # this is the image
    if image is not None and show_image:
        img.visual = trimesh.visual.TextureVisuals(uv_coords, image=PIL.Image.fromarray(image))
    else:
        img.visual.face_colors = [*[255, 255, 255], 0.3]  # RGBA with 10% opacity
    scene.add_geometry(img)

    ##### Add camera
    # this is the camera mesh
    rot2 = np.eye(4)
    rot2[:3, :3] = Rotation.from_euler('z', np.deg2rad(2)).as_matrix()
    vertices = np.r_[cam.vertices, 0.95*cam.vertices, _geotrf(rot2, cam.vertices)]
    vertices = _geotrf(transform, vertices)
    faces = []
    for face in cam.faces:
        if 0 in face:
            continue
        a, b, c = face
        a2, b2, c2 = face + len(cam.vertices)
        a3, b3, c3 = face + 2*len(cam.vertices)

        # add 3 pseudo-edges
        faces.append((a, b, b2))
        faces.append((a, a2, c))
        faces.append((c2, b, c))

        faces.append((a, b, b3))
        faces.append((a, a3, c))
        faces.append((c3, b, c))

    # no culling
    faces += [(c, b, a) for a, b, c in faces]

    cam = trimesh.Trimesh(vertices=vertices, faces=faces)
    cam.visual.face_colors[:, :3] = color
    scene.add_geometry(cam)

    ##### Add text - scene_name
    # TODO(gogojjh): cannnot visualize text
    # Create text mesh with scaling based on camera size
    camera_position = pose_c2w[:3, 3]
    
    text_size = cam_size * 0.5  # Adjust text size relative to camera
    text_depth = text_size * 0.05  # Text extrusion depth
        
    # Create 3D text mesh (oriented to face default view)
    img_text = Image.new('RGB', (256, 64), (255, 255, 255, 255))
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 40)
    except:
        font = ImageFont.load_default()
    d = ImageDraw.Draw(img_text)
    d.text((60, 10), scene_name, 
           fill=(0, 0, 0, 255), 
           font=font, 
           stroke_fill=(0, 0, 0, 255))

    text_mesh = trimesh.creation.box((text_size, text_size*0.5, 0.001))
    text_mesh.visual = trimesh.visual.TextureVisuals(
        uv=[[0,0], [1,0], [1,1], [0,1]],
        image=img_text
    )

    text_offset = np.array([0, text_size*1.5, text_size*0.5])
    text_transform = np.eye(4)
    text_transform[:3, 3] = camera_position + text_offset

    # Align text to face the camera's viewing direction
    text_transform[:3, :3] = pose_c2w[:3, :3] @ Rotation.from_euler('x', -45).as_matrix()
    text_mesh.apply_transform(text_transform)

    # Add text to scene
    scene.add_geometry(text_mesh)    

def visualize_scenes(scene_data, is_multi_frame, cam_size=0.03, show_image=True, step=1):
    """Visualizes multiple scenes with cameras and images.
    
    Args:
        scene_data (dict): Loaded scene data from load_scene_data
        cam_size (float): Base size of camera frustums in meters
    """
    scene = trimesh.Scene()

    for scene_name, data in scene_data.items():
        scene_color = _get_scene_color(scene_name)
        
        for idx, img_path in enumerate(data['poses']):
            if idx % step != 0: 
                continue

            if img_path in data['images']:
                # The rgb image exists
                exist_image = True
            else:
                # The rgb image is removed
                exist_image = False          

            # Get camera parameters
            pose_w2c = data['poses'][img_path]
            if is_multi_frame:
                data_tmp = scene_data[next(iter(scene_data))]
                # Normalize poses to the coordinate frame of the first camera c0
                pose_w2c0 = data_tmp['poses'][data_tmp['images'][0]]
                # T^ct_c0 = T^ct_w @ T^w_c0
                pose_w2c = pose_w2c @ np.linalg.inv(pose_w2c0)

            try:
                if exist_image:
                    show_cam_size = cam_size
                    fx, fy, cx, cy, width, height = data['intrinsics'][img_path]
                    image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                    imsize = (int(width), int(height))
                else:
                    show_cam_size = cam_size * 5
                    fx, fy, cx, cy, width, height = data['intrinsics'][next(iter(data['intrinsics']))]
                    image = None
                    imsize = (int(width), int(height))

                if 'seq0/frame_000000' in img_path or 'seq/000000' in img_path:
                    show_cam_size = cam_size * 5
                    _add_scene_cam(
                        scene=scene,
                        scene_name=scene_name,
                        pose_w2c=pose_w2c,
                        color=scene_color,
                        image=image,
                        focal=fx,
                        imsize=imsize,
                        cam_size=show_cam_size,
                        show_image=(True and exist_image)
                    )
                else:
                    _add_scene_cam(
                        scene=scene,
                        scene_name=scene_name,
                        pose_w2c=pose_w2c,
                        color=scene_color,
                        image=image,
                        focal=fx,
                        imsize=imsize,
                        cam_size=show_cam_size,
                        show_image=(show_image and exist_image)
                    )
            except Exception as e:
                print(f"Error processing {img_path}: {str(e)}")

    scene.show()

if __name__ == '__main__':
    args = parse_arguments()
    scene_data, is_multi_frame = load_scene_data(args.dataset_dir, args.scenes)
    
    if scene_data:
        print(f"Visualizing {len(scene_data)} scenes: {', '.join(scene_data.keys())}")
        visualize_scenes(scene_data, is_multi_frame, args.cam_size, args.show_image, args.step)
    else:
        print("No valid scenes found, exiting...")

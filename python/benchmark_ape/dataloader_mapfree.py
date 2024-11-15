# Code adapted from Map-free benchmark: https://github.com/nianticlabs/map-free-reloc

import os
import sys
import random
from pathlib import Path
import torch
import torch.utils.data as data
import numpy as np
from transforms3d.quaternions import qinverse, qmult, rotate_vector, quat2mat

# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map_free_reloc"))
# from lib.datasets.utils import read_color_image, read_depth_image, correct_intrinsic_scale

SEED = 42 # Set constant random seed for reproducibility

def correct_intrinsic_scale(K, scale_x, scale_y):
    """Given an intrinsic matrix (3x3) and two scale factors, returns the new intrinsic matrix corresponding to
    the new coordinates x' = scale_x * x; y' = scale_y * y
    Source: https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    """

    transform = np.eye(3)
    transform[0, 0] = scale_x
    transform[0, 2] = scale_x / 2 - 0.5
    transform[1, 1] = scale_y
    transform[1, 2] = scale_y / 2 - 0.5
    Kprime = transform @ K

    return Kprime

class MapFreeScene(data.Dataset):
    def __init__(
            self, scene_root, resize, overlap_limits=None, N_query=1, top_K=2, 
            transforms=None, test_scene=False):
        super().__init__()

        self.scene_root = Path(scene_root)
        self.resize = resize
        self.transforms = transforms
        self.test_scene = test_scene

        # load absolute poses
        self.poses = self.read_poses(self.scene_root)

        # read intrinsics
        self.K, self.K_ori = self.read_intrinsics(self.scene_root, resize)

        # load pairs
        self.pairs = self.load_pairs(self.scene_root, overlap_limits, N_query, top_K)

    @staticmethod
    def read_intrinsics(scene_root: Path, resize=None):
        Ks = {}
        K_ori = {}
        with (scene_root / 'intrinsics.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                fx, fy, cx, cy, W, H = map(float, line[1:])

                K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                K_ori[img_name] = K
                if resize is not None:
                    K = correct_intrinsic_scale(K, resize[0] / W, resize[1] / H)
                Ks[img_name] = K
        return Ks, K_ori

    @staticmethod
    def read_poses(scene_root: Path):
        """
        Returns a dictionary that maps: img_path -> (q, t) where
        np.array q = (qw, qx qy qz) quaternion encoding rotation matrix;
        np.array t = (tx ty tz) translation vector;
        (q, t) encodes absolute pose (world-to-camera), i.e. X_c = R(q) X_W + t
        """
        poses = {}
        with (scene_root / 'poses.txt').open('r') as f:
            for line in f.readlines():
                if '#' in line:
                    continue

                line = line.strip().split(' ')
                img_name = line[0]
                qt = np.array(list(map(float, line[1:])))
                poses[img_name] = (qt[:4], qt[4:])
        return poses

    def load_pairs(self, scene_root: Path, overlap_limits: tuple = None, N_query: int = 1, top_K: int = 2):
        """
        Load pairs of frames based on overlap for training scenes, or form pairs for test/val scenes.
        Sets a fixed seed and generates random indices from [0, len(self.poses)].

        Args:
        scene_root (Path): The root directory of the scene.
        overlap_limits (tuple, optional): Min and max overlap range for filtering pairs.
        N_query (int, optional): specifies the number of query for test.
        top_K (int, optional): specifies the number of images for localization test.

        Returns:
        idxx: np.ndarray of shape [N+1,], containing indices [0, k1, k2, ..., kN].
        pairs: np.ndarray [Npairs, 4] with each column representing seaA, imA, seqB, imB, respectively.
        """
        # Generate random pairs
        random.seed(SEED)
        random_idxs = np.random.randint(0, len(self.poses) - 1, size=(N_query, top_K))
        idxs = np.zeros((N_query, 3 + top_K), dtype=np.uint16)
        idxs[:, 2] = 1
        idxs[:, 3:] = random_idxs
        pairs = idxs.copy()
        return pairs

    def get_pair_path(self, pair):
        seqA_id, imgA_id, seqB_id, list_imgB_id = pair[0], pair[1], pair[2], pair[3:]
        tar_img_name = f'seq{seqA_id}/frame_{imgA_id:05}.jpg'
        list_ref_img_name = [f'seq{seqB_id}/frame_{imgB_id:05}.jpg' for imgB_id in list_imgB_id]
        return tar_img_name, list_ref_img_name

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        # image paths (relative to scene_root)
        im_path_tar, list_im_path_ref = self.get_pair_path(self.pairs[index, :])
        
        im_path_tar_full = os.path.join(str(self.scene_root), im_path_tar)
        list_im_path_ref_full = [os.path.join(str(self.scene_root), im_path_ref) for im_path_ref in list_im_path_ref]

        # # load color images
        # image_tar = read_color_image(self.scene_root / im_path_tar,
        #                              self.resize, augment_fn=self.transforms)
        # list_image_ref = [read_color_image(self.scene_root / im_path_ref, 
        #                                    self.resize, augment_fn=self.transforms)
        #                   for im_path_ref in list_im_path_ref]

        # load intrinsics
        list_K_ref = [torch.from_numpy(self.K[im_path_ref]) for im_path_ref in list_im_path_ref]
        list_K_ori_ref = [torch.from_numpy(self.K_ori[im_path_ref]) for im_path_ref in list_im_path_ref]
        K_tar = self.K[im_path_tar]
        K_ori_tar = self.K_ori[im_path_tar]

        # get absolute pose of reference_images and target_image
        if self.test_scene:
            list_img_ref_pose = []
            for im_path in list_im_path_ref:
                q, t = self.poses[im_path]
                c = rotate_vector(-t, qinverse(q))
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = quat2mat(qinverse(q))
                T[:3, 3] = c
                list_img_ref_pose.append(torch.from_numpy(T))

            q, t = np.zeros([4]), np.zeros([3])
            T = np.zeros([4, 4])
            img_tar_pose = torch.from_numpy(T)
        else:
            list_img_ref_pose = []
            for im_path in list_im_path_ref:
                q, t = self.poses[im_path]
                c = rotate_vector(-t, qinverse(q))
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = quat2mat(qinverse(q))
                T[:3, 3] = c
                list_img_ref_pose.append(torch.from_numpy(T))

            q, t = self.poses[im_path_tar]
            c = rotate_vector(-t, qinverse(q))
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = quat2mat(qinverse(q))
            T[:3, 3] = c
            img_tar_pose = torch.from_numpy(T)

        data = {
            'list_image0_path_full': list_im_path_ref_full,  # list of paths of reference images
            'image1_path_full': im_path_tar_full, # path of target image
            'list_K_color0': list_K_ref,  # list of (3, 3)
            'K_color1': K_tar,  # (3, 3)
            'list_Kori_color0': list_K_ori_ref,  # list of (3, 3)
            'Kori_color1': K_ori_tar,  # (3, 3)
            'list_image0_pose': list_img_ref_pose,  # list of (4, 4)
            'image1_pose': img_tar_pose,
            'top_K': torch.tensor(len(list_im_path_ref)),
            'dataset_name': 'Mapfree',
            'scene_id': self.scene_root.stem,
            'scene_root': str(self.scene_root),
            'pair_id': index,
            'pair_names': (list_im_path_ref, im_path_tar),
        }

        return data

class MapFreeDataset(data.ConcatDataset):
    def __init__(self, cfg, mode, transforms=None):
        assert mode in ['train', 'val', 'test'], 'Invalid dataset mode'
        assert cfg.DATASET.TOP_K >= 2 # At least 2 images for metric-based localization
        assert cfg.DATASET.N_QUERY >= 1 # At least 1 query for localization

        data_root = Path(cfg.DATASET.DATA_ROOT) / mode

        if mode == 'test':
            test_scene = True
        else:
            test_scene = False

        scenes = cfg.DATASET.SCENES
        if scenes is None:
            # Locate all scenes of the current dataset
            scenes = [s.name for s in data_root.iterdir() if s.is_dir()]

        # Init dataset objects for each scene
        data_srcs = [
            MapFreeScene(
                data_root / scene, resize=None, overlap_limits=None, N_query=cfg.DATASET.N_QUERY, top_K=cfg.DATASET.TOP_K, 
                transforms=None, test_scene=False) for scene in scenes]
        super().__init__(data_srcs)

# [Computation]:
# python submission.py --config ../config/dataset/mapfree.yaml --models master --pose_solver essentialmatrixmetricmean --out_dir /Titan/dataset/data_mapfree/results --split val
# [Evaluation] (in mickey folder):
# python -m benchmark.mapfree --submission_path /Titan/dataset/data_mapfree/results/master_essentialmatrixmetricmean/submission.zip --dataset_path /Titan/dataset/data_mapfree --split val --log error

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from zipfile import ZipFile

import time
import numpy as np
from tqdm import tqdm

from transforms3d.quaternions import mat2quat

# from matching import available_models, get_matcher
from matching.utils import to_numpy, to_tensor

from ape_default import cfg
from datamodules import DataModule

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map_free_reloc"))
from lib.utils.data import data_to_model_device

from pycpptools.src.python.utils_sensor.utils import correct_intrinsic_scale

def get_ape_method(model, device="cuda"):
    return model

@dataclass
class Pose:
    top_K: int
    list_ref_image_name: list
    tar_image_name: str
    q: np.ndarray
    t: np.ndarray

    def __str__(self) -> str:
        formatter = {"float": lambda v: f"{v:.6f}"}
        max_line_width = 1000
        q_str = np.array2string(
            self.q, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        t_str = np.array2string(
            self.t, formatter=formatter, max_line_width=max_line_width
        )[1:-1]
        str_ref_image_names = " ".join(ref_image_name for ref_image_name in self.list_ref_image_name)
        return f"{self.top_K} {self.tar_image_name} {str_ref_image_names} {q_str} {t_str}"

def predict(loader, ape_method, str_ape_metho):
    results_dict = defaultdict(list)
    running_time = []
    for data in tqdm(loader):
        # try:
            # data = data_to_model_device(data, ape_method) # in torch.tensor, but in cpu
            list_ref_img  = [img.squeeze(0) for img in data['list_image0']]
            list_ref_img_K = [K.squeeze(0) for K in data['list_K_color0']]
            list_ref_img_Kori = [K.squeeze(0) for K in data['list_Kori_color0']]
            list_ref_img_pose = [pose.squeeze(0) for pose in data['list_image0_pose']]          
            list_ref_img_name = [name[0] for name in data['pair_names'][0]]
            top_K = data['top_K'].squeeze(0)
            print(list_ref_img_name)

            tar_img = data['image1'].squeeze(0)
            tar_img_K = data['K_color1'].squeeze(0)
            tar_img_Kori = data['Kori_color1'].squeeze(0)
            tar_img_name = data['pair_names'][1][0]

            """Absolute Pose Estimation"""
            start_time = time.time()
            # ape_result = ape_method(list_ref_img, list_ref_img_K, list_ref_img_pose, tar_img, tar_img_K)
            ape_result = {"R": to_tensor(np.eye(3), 'cpu'), "t": to_tensor(np.zeros(3), 'cpu')}
            ape_time = time.time() - start_time
            running_time.append(ape_time)

            """Definition of solver output"""
            # Rwc (numpy.ndarray): Estimated rotation matrix from world (reference frame) to camera
            # twc (numpy.ndarray): Estimated translation vector. Shape: [3, 1] that translate depth_img1 to depth_img0.
            # inliers_solver (int): Number of inliers used in the final pose estimation.
            R, t = ape_result["R"].squeeze(0), ape_result["t"].squeeze(0)
            Rwc, twc = R, t
            # inliers = ape_result["inliers"].squeeze(0)
            Twc = np.eye(4); Twc[:3, :3] = Rwc; Twc[:3,  3] = twc.reshape(3)
            Tcw = np.linalg.inv(Twc); R = Twc[:3, :3]; t = Twc[:3,  3].reshape(3, 1)

            """Save Results"""
            scene = data['scene_id'][0]
            if np.isnan(R).any() or np.isnan(t).any() or np.isinf(t).any():
                raise ValueError("Estimated pose is NaN or infinite.")

            # populate results_dict
            estimated_pose = Pose(top_K=top_K,
                                list_ref_image_name=list_ref_img_name, 
                                tar_image_name=tar_img_name,
                                q=mat2quat(R).reshape(-1),
                                t=t.reshape(-1))
            results_dict[scene].append(estimated_pose)

            # if args.debug:
            #     print(t.T)
            #     if num_inliers < 100:
            #         print(f"Inliers number < 100: {num_inliers} at {data['scene_id'][0]}/{data['pair_names']}")
            #     out_match_dir = Path(os.path.join(args.out_dir, f"{str_matcher}_{str_solver}"))
            #     out_match_dir.mkdir(parents=True, exist_ok=True)
            #     Path(out_match_dir / "preds").mkdir(parents=True, exist_ok=True)
            #     text = f"{len(mkpts1)} matches: {scene}-{query_img.split('/')[1]}" # "N matches: s00000-frame_000000.jpg"
            #     save_visualization(rgb_img0, rgb_img1, mkpts0, mkpts1, out_match_dir, save_indice, n_viz=30, line_width=0.6, text=text)
            #     save_indice += 1
            #     if str_matcher == "duster" and args.viz: matcher.scene.show(cam_size=0.05)
        # except Exception as e:
        #     # scene = data['scene_id'][0]
        #     # query_img = data['pair_names'][1][0]
        #     # tqdm.write(f"Error with {str_matcher}: {e}")
        #     # tqdm.write(f"(duster) May occur due to no overlapping regions or insufficient matching at {scene}/{query_img}.")
        #     pass

    avg_runtime = running_time[0] if len(running_time) == 1 else np.mean(running_time)
    return results_dict, avg_runtime

def save_submission(results_dict: dict, output_path: Path):
    with ZipFile(output_path, "w") as zip:
        for scene, poses in results_dict.items():
            poses_str = "\n".join((str(pose) for pose in poses))
            zip.writestr(f"pose_{scene}.txt", poses_str.encode("utf-8"))

def eval(args):
    # Load configs
    cfg.merge_from_file(args.config)

    # Create dataloader for different datasets
    if args.split == 'test':
        cfg.TRAINING.BATCH_SIZE = 1
        cfg.TRAINING.NUM_WORKERS = 1
        dataloader = DataModule(cfg).test_dataloader()
    elif args.split == 'val':
        cfg.TRAINING.BATCH_SIZE = 1
        cfg.TRAINING.NUM_WORKERS = 1
        dataloader = DataModule(cfg).val_dataloader()
    else:
        raise NotImplemented(f'Invalid split: {args.split}')

    output_root = Path(args.out_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    with open(output_root / "runtime_results.txt", "w") as f:
        # for model in args.models:
        for model in ['master']:
            ape_method = get_ape_method(model, device=args.device)
            print(f"Running APE Method: {model}")
            results_dict, avg_runtime = predict(dataloader, ape_method, model)

            # Save runtimes to txt
            runtime_str = f"{model}: {avg_runtime:.3f}s"
            f.write(runtime_str + "\n")
            tqdm.write(runtime_str)

            # Save predictions to txt per scene within zip
            log_dir = Path(os.path.join(output_root, f"{model}"))
            log_dir.mkdir(parents=True, exist_ok=True)
            save_submission(results_dict, log_dir / "submission.zip")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="path to config file")
    # parser.add_argument(
    #     "--models",
    #     type=str,
    #     nargs="+",
    #     default="all",
    #     choices=available_models
    # )
    # parser.add_argument(
    #     "--pose_solvers",
    #     type=str,
    #     nargs="+",
    #     default="all",
    #     choices=available_solvers,
    # )
    parser.add_argument(
        "--device", type=str, default="cuda", choices=["cpu", "cuda"]
    )
    parser.add_argument(
        "--viz",
        action="store_true",
        help="pass --viz to avoid saving visualizations",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="pass --debug to visualize intermediate results",
    )    
    parser.add_argument(
        "--out_dir", type=str, default=None, help="path where outputs are saved"
    )
    parser.add_argument(
        "--num-iters",
        type=int,
        default=1,
        help="number of interations to run benchmark and average over",
    )    
    parser.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Dataset split to use for evaluation. Choose from test or val. Default: test",
    )
    args = parser.parse_args()
    # if args.models == "all":
    #     args.models = available_models
    # if args.pose_solvers == "all":
    #     args.pose_solvers = available_solvers
    eval(args)

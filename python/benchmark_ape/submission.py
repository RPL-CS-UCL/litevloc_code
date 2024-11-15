# [Computation]:
# python submission.py --config ../config/dataset/matterport3d.yaml --split test --out_dir xx --models master --debug
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

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../pose_estimation_models"))
from estimator import available_models, get_estimator

from ape_default import cfg
from datamodules import DataModule

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../map_free_reloc"))
from lib.utils.data import data_to_model_device

from pycpptools.src.python.utils_sensor.utils import correct_intrinsic_scale

@dataclass
class Pose:
    top_K: int
    list_ref_image_name: list
    tar_image_name: str
    q: np.ndarray
    t: np.ndarray
    loss: float

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
        return f"{self.top_K} {self.tar_image_name} {str_ref_image_names} {q_str} {t_str} {self.loss:.3f}"

def predict(loader, estimator, str_estimator, cfg):
    results_dict = defaultdict(list)
    running_time = []
    save_indice = 0
    for data in tqdm(loader):
        # try:
            # data = data_to_model_device(data, estimator)
            top_K = data['top_K'].detach().cpu().item()
            list_ref_img_path_full = [path[0] for path in data['list_image0_path_full']]
            tar_img_path_full = data['image1_path_full']

            list_ref_img_K = [K.squeeze(0) for K in data['list_K_color0']]
            # list_ref_img_Kori = [K.squeeze(0) for K in data['list_Kori_color0']]
            list_ref_img_pose = [pose.squeeze(0) for pose in data['list_image0_pose']]

            tar_img_K = data['K_color1'].squeeze(0)
            # tar_img_Kori = data['Kori_color1'].squeeze(0)
            init_img1_pose = data['image1_pose'].squeeze(0)

            """Absolute Pose Estimation"""
            start_time = time.time()
            option = {
                'resize': 512,
                'opt_cam': 'single'
            }
            est_result = estimator(list_ref_img_path_full, tar_img_path_full, 
                                   list_ref_img_pose, init_img1_pose, 
                                   list_ref_img_K, tar_img_K, 
                                   option)
            est_time = time.time() - start_time
            running_time.append(est_time)

            """Definition of solver output"""
            # Rwc (numpy.ndarray): Estimated rotation matrix from world (reference frame) to camera
            # twc (numpy.ndarray): Estimated translation vector. Shape: [3, 1] that translate depth_img1 to depth_img0.
            im_pose, loss = est_result["im_pose"], est_result["loss"]
            Rwc, twc = im_pose[:3, :3], im_pose[:3, 3].reshape(3, 1)
            Twc = np.eye(4); Twc[:3, :3] = Rwc; Twc[:3,  3] = twc.reshape(3)
            Tcw = np.linalg.inv(Twc); Rcw = Tcw[:3, :3]; tcw = Tcw[:3,  3].reshape(3, 1)

            """Save Results"""
            scene = data['scene_id'][0]
            if np.isnan(im_pose).any():
                raise ValueError("Estimated pose is NaN or infinite.")

            # populate results_dict
            tar_img_name = data['pair_names'][1][0]
            list_ref_img_name = [name[0] for name in data['pair_names'][0]]
            estimated_pose = Pose(top_K=top_K,
                                  list_ref_image_name=list_ref_img_name, 
                                  tar_image_name=tar_img_name,
                                  q=mat2quat(Rcw).reshape(-1),
                                  t=tcw.reshape(-1),
                                  loss=loss)
            results_dict[scene].append(estimated_pose)

            if args.debug:
                print(tcw.T)
                if args.viz:
                    estimator.scene.show(cam_size=cfg.DATASET.VIZ_CAM_SIZE)
                out_est_dir = Path(os.path.join(args.out_dir, f"{str_estimator}"))
                out_est_dir.mkdir(parents=True, exist_ok=True)
                Path(out_est_dir / "preds").mkdir(parents=True, exist_ok=True)
                # text = f"{len(mkpts1)} matches: {scene}-{query_img.split('/')[1]}" # "N matches: s00000-frame_000000.jpg"
                # save_visualization(rgb_img0, rgb_img1, mkpts0, mkpts1, out_match_dir, save_indice, n_viz=30, line_width=0.6, text=text)
                save_indice += 1
        # except Exception as e:
        #     scene = data['scene_id'][0]
        #     top_K = data['top_K'].detach().cpu().item()
        #     list_ref_img_name = [name[0] for name in data['pair_names'][0]]
        #     tar_img_name = data['pair_names'][1][0]
        #     estimated_pose = Pose(top_K=top_K,
        #                           list_ref_image_name=list_ref_img_name, 
        #                           tar_image_name=tar_img_name,
        #                           q=np.array([None, None, None, None]).reshape(-1),
        #                           t=np.array([None, None, None, None]).reshape(-1),
        #                           loss=0)
        #     results_dict[scene].append(estimated_pose)
        #     tqdm.write(f"Error with {str_estimator}: {e}")
        #     tqdm.write(f"May occur due to no overlapping regions or insufficient matching at {scene}/{tar_img_name}.")

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
        for model in args.models:
            estimator = get_estimator(model, device=args.device)
            print(f"Running APE Method: {model}")
            results_dict, avg_runtime = predict(dataloader, estimator, model, cfg)

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
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default="all",
        choices=available_models
    )
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
    if args.models == "all":
        args.models = available_models
    eval(args)

# LiteVLoc Offline Visual Localization

## Overview

LiteVLoc is a three-stage hierarchical visual localization pipeline that works without ROS dependencies:
1. **Global localization (VPR)** — coarse retrieval from a pre-built topometric map
2. **Image matching** — fine-grained keypoint matching between query and map node
3. **Pose estimation** — PnP solver with depth for full 6-DoF camera pose

Results are saved to a **Rerun `.rrd`** file for interactive 3D visualization.

---

## Quick Start

```bash
cd /Titan/code/robohike_ws/src/opennavmap
conda activate opennavmap

# Run on one environment — outputs a .rrd file
bash third_party/litevloc_code/scripts/run_vloc_offline_rerun.sh s17DRP5sb8fy

# Open the recording in Rerun Viewer
rerun third_party/litevloc_code/output/vloc_s17DRP5sb8fy.rrd
```

Supported environments: `s17DRP5sb8fy`, `sB6ByNegPMK`, `sEDJbREhghzL`

---

## Manual Command (Full Control)

```bash
cd /Titan/code/robohike_ws/src/opennavmap
LD_LIBRARY_PATH=/root/miniconda3/envs/opennavmap/lib:$LD_LIBRARY_PATH \
PYTHONPATH=third_party/litevloc_code/python:third_party/VPR-methods-evaluation \
python third_party/litevloc_code/python/run_vloc_offline_rerun.py \
  --map_path /Titan/dataset/data_opennavmap/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
  --query_data_path /Titan/dataset/data_opennavmap/vnav_eval/matterport3d/s17DRP5sb8fy/merge_finalmap \
  --output_rrd third_party/litevloc_code/output/vloc_s17DRP5sb8fy.rrd \
  --image_size 512 288 --device cuda \
  --vpr_method cosplace --vpr_backbone ResNet18 --vpr_descriptors_dimension 256 \
  --vpr_match_model single_match --vpr_match_seq_len 1 \
  --img_matcher master --pose_solver pnp \
  --config_pose_solver third_party/litevloc_code/python/config/dataset/matterport3d.yaml \
  --global_pos_threshold 10.0 --min_master_conf_thre 1.5 --min_solver_inliers_thre 200
```

---

## Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--map_path` | *(required)* | Directory of the pre-built topometric map |
| `--query_data_path` | *(required)* | Directory of query observations |
| `--output_rrd` | `output/vloc_result.rrd` | Output `.rrd` path |
| `--image_size` | `512 288` | Resize width and height |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--vpr_method` | `cosplace` | VPR model: `cosplace`, `netvlad`, `eigenplaces`, etc. |
| `--vpr_backbone` | `ResNet18` | Backbone for VPR model |
| `--vpr_match_model` | `sequence_match` | VPR matching: `single_match`, `sequence_match` |
| `--vpr_match_seq_len` | `5` | Sequence length for sequence match |
| `--img_matcher` | `master` | Feature matcher: `master`, `superpoint`, `disk`, etc. |
| `--n_kpts` | `2048` | Number of keypoints for image matcher |
| `--pose_solver` | `pnp` | Pose solver: `pnp` |
| `--config_pose_solver` | `matterport3d.yaml` | YACS config path |
| `--global_pos_threshold` | `10.0` | Distance threshold (m) for keyframe candidates |
| `--min_master_conf_thre` | `1.5` | Confidence threshold for Mast3R matcher |
| `--min_solver_inliers_thre` | `200` | Minimum inliers for PnP solver success |
| `--depth_scale` | `0.001` | Depth image scale (mm → m) |

---

## Expected Output

Each query frame prints a structured log:

```
Loading observation seq/000000.color.jpg
Global localization costs: 0.012s
Found VPR Node in global position: 0
Keyframe candidate: 1(1.06) 3(1.02) 5(1.15) Closest node: 0
Number of matched inliers: 2229
Image matching costs: 0.269s
[Succ] sufficient number 2229 solver inliers
Local localization costs: 0.273s
Groundtruth Poses: [-0.60591018  1.01205952  1.        ]
Estimated Poses: [-0.605915    1.0120701   0.99999773]
t_err=0.000m r_err=0.02deg inliers=2229
```

---

## Rerun `.rrd` Visualization

Open with:
```bash
rerun third_party/litevloc_code/output/vloc_s17DRP5sb8fy.rrd
```

### What You See

| View | Content |
|------|---------|
| **3D View** | Green boxes = map nodes; blue lines = topology edges; red/green world axes at origin |
| **Zoom on a node** | Camera frustum with the node's RGB image textured on it |
| **Play timeline** | Green trajectory = GT; red trajectory = estimated poses |
| **query/matching** | Combined ref+query images with green match lines (up to 20 keypoint pairs) |

---

## Data Format

The map directory must contain:

```
map_root/
├── seq/                        # RGB-D image sequences
│   ├── 000000.color.jpg
│   └── 000000.depth.png        # uint16 in mm, optional (needed for PnP)
├── intrinsics.txt              # fx fy cx cy width height per image
├── poses.txt                   # qw qx qy qz tx ty tz per image
├── database_descriptors.txt    # VPR global descriptors (256-dim CosPlace)
├── edges_covis.txt             # Covisibility edges (node_a node_b weight)
├── edges_trav.txt              # Traversability edges
└── timestamps.txt              # Image timestamps
```

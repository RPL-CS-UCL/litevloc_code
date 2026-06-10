#!/bin/bash
# Run LiteVLoc offline localization with Rerun visualization (no ROS required).
#
# Usage:
#   bash scripts/run_vloc_offline_rerun.sh [ENV_ID] [MAP_ROOT]
#
# Arguments:
#   ENV_ID   - Matterport3D environment id (default: s17DRP5sb8fy)
#   MAP_ROOT - Root directory containing environment subdirs
#              (default: /Titan/dataset/data_opennavmap/vnav_eval/matterport3d)
#
# Output:
#   third_party/litevloc_code/output/vloc_<ENV_ID>.rrd
#
# Run from the OpenNavMap workspace root:
#   cd /Titan/code/robohike_ws/src/opennavmap
#   bash third_party/litevloc_code/scripts/run_vloc_offline_rerun.sh s17DRP5sb8fy
#   bash third_party/litevloc_code/scripts/run_vloc_offline_rerun.sh sB6ByNegPMK
#   bash third_party/litevloc_code/scripts/run_vloc_offline_rerun.sh sEDJbREhghzL

set -euo pipefail

ENV_ID="${1:-s17DRP5sb8fy}"
MAP_ROOT="${2:-/Titan/dataset/data_opennavmap/vnav_eval/matterport3d}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITEVLOC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_ROOT="$(cd "${LITEVLOC_ROOT}/../.." && pwd)"

MAP_PATH="${MAP_ROOT}/${ENV_ID}/merge_finalmap"
OUTPUT_RRD="${LITEVLOC_ROOT}/output/vloc_${ENV_ID}.rrd"
CONFIG_POSE="${LITEVLOC_ROOT}/python/config/dataset/matterport3d.yaml"
CONDA_PROFILE="/root/miniconda3/etc/profile.d/conda.sh"
CONDA_ENV="opennavmap"

source "${CONDA_PROFILE}"
conda activate "${CONDA_ENV}"

export LD_LIBRARY_PATH="/root/miniconda3/envs/${CONDA_ENV}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="${LITEVLOC_ROOT}/python:${WORKSPACE_ROOT}/third_party/VPR-methods-evaluation:${PYTHONPATH:-}"

echo "============================================"
echo " LiteVLoc Offline Rerun Visualization"
echo "============================================"
echo "  ENV_ID     : ${ENV_ID}"
echo "  MAP_PATH   : ${MAP_PATH}"
echo "  OUTPUT_RRD : ${OUTPUT_RRD}"
echo "============================================"

PYTHON="${LITEVLOC_ROOT}/python/run_vloc_offline_rerun.py"

python "${PYTHON}" \
  --map_path            "${MAP_PATH}" \
  --query_data_path     "${MAP_PATH}" \
  --output_rrd          "${OUTPUT_RRD}" \
  --image_size          512 288 \
  --device              cuda \
  --vpr_method          cosplace \
  --vpr_backbone        ResNet18 \
  --vpr_descriptors_dimension 256 \
  --vpr_match_model     single_match \
  --vpr_match_seq_len   1 \
  --img_matcher         master \
  --pose_solver         pnp \
  --config_pose_solver  "${CONFIG_POSE}" \
  --global_pos_threshold    10.0 \
  --min_master_conf_thre    1.5 \
  --min_solver_inliers_thre 200

echo ""
echo "Done. Open recording with:"
echo "  rerun ${OUTPUT_RRD}"

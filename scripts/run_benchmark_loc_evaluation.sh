"""
Dense matching: 
  roma tiny-roma duster master
Semi-dense matching:
  loftr eloftr matchformer xfeat-star
Sparse matching:
  sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat
"""

export MAP_FREE_PATH="/Titan/code/robohike_ws/src/image_matching_models/third_party/mickey"
export DATASET_PATH="/Rocket_ssd/dataset/data_topo_loc/matterport3d/eval/"
models=("roma" "tiny-roma" "duster" "master" "loftr" "eloftr" "matchformer" "xfeat-star" "sift-lg" "superpoint-lg" "gim-lg" "xfeat-lg" "sift-nn" "orb-nn" "gim-dkm" "xfeat")
models=("master")

cd $MAP_FREE_PATH
for model in "${models[@]}"
do
  echo "Evaluate matching methods: "$model"_pnp"
  python -m benchmark.mapfree --submission_path $DATASET_PATH/results/"$model"_pnp/submission.zip --dataset_path $DATASET_PATH --split test --log error
  echo ""
done

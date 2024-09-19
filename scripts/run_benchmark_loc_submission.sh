# Dense matching: 
#   roma tiny-roma duster master
# Semi-dense matching:
#   loftr eloftr matchformer xfeat-star
# Sparse matching:
#   sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat

export PROJECT_PATH="/Titan/code/robohike_ws/src/topo_loc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/matterport3d.yaml"
export OUT_DIR="/Rocket_ssd/dataset/data_topo_loc/matterport3d/map_free_eval/results"
export MODELS="roma tiny-roma duster master loftr eloftr matchformer xfeat-star sift-lg superpoint-lg gim-lg xfeat-lg sift-nn orb-nn gim-dkm xfeat"

python $PROJECT_PATH/python/benchmark_loc/submission.py --config $CONFIG_FILE --models $MODELS --pose_solver pnp --out_dir $OUT_DIR --split test --debug
# Direct regression
#   duster master

export PROJECT_PATH="/Titan/code/robohike_ws/src/litevloc"
export CONFIG_FILE="$PROJECT_PATH/python/config/dataset/hkustgz_campus.yaml"
export OUT_DIR="/Rocket_ssd/dataset/data_litevloc/hkustgz_campus/map_free_eval/results_ape"
export MODELS="master"

python $PROJECT_PATH/python/benchmark_ape/submission.py --config $CONFIG_FILE --models $MODELS --out_dir $OUT_DIR --split test --debug --viz
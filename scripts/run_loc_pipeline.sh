export MAP_PATH='/Rocket_ssd/dataset/data_litevloc/vnav_eval/ucl_campus/sops_msg_olympic_202505'
export QUERY_DATA_PATH='/Rocket_ssd/dataset/data_litevloc/map_multisession_eval/ucl_campus_robot/s00000_data/0'

rosrun litevloc loc_pipeline.py \
	--map_path $MAP_PATH --query_data_path $QUERY_DATA_PATH \
	--image_size 512 288 --device=cuda \
	--vpr_method cosplace --vpr_backbone=ResNet18 --vpr_descriptors_dimension=256 \
	--img_matcher master \
	--pose_solver pnp --config_pose_solver python/config/dataset/matterport3d.yaml \
	--global_pos_threshold 5.0 --min_solver_inliers_thre 50 --min_master_conf_thre 1.0 \
	--num_preds_to_save 3 --save_img_matcher --viz

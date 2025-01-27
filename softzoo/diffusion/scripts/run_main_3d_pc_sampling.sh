


export task_cond="--task_cond"
export task_cond=""

export resume_checkpoint=""


# export exp_tag=allegro_lotsballs_singleinst_taskcond_v2_

export exp_tag=allegro_test_bt_cylinder_

export save_interval=100

export debug=""
export single_inst=""


export exp_tag=allegro_test_bt_cylinder_singleinst_
# export debug="--debug"
export single_inst="--single_inst"


export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/save_info_v6_statistics.npy"

# export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/save_info_v6_statistics_single.npy"


### training settings ###
# export training_setting='regular_training'
export training_setting='trajectory_translations'
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_cylinder_/model000005400.pt"
export single_inst="--single_inst"
### training settings ###


### training settings ###
export training_setting="trajectory_translations"
# export training_setting="regular_training"

# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_cylinder_/model000013700.pt"
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_cylinder_trajtrans_2/model000015700.pt"
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_cylinder_trajtrans_3/model000016100.pt"
export single_inst=""
export save_interval=100
export exp_tag="allegro_test_bt_cylinder_trajtrans_3"
### training settings ###

# export batch_size=8

export sampling=""


export sampling="--sampling"
export use_t=500

export cuda_ids=6

# bash scripts/run_main_3d_pc_sampling.sh

CUDA_VISIBLE_DEVICES=${cuda_ids} python main_3d_pc.py  ${task_cond}  --resume_checkpoint=${resume_checkpoint} --exp_tag=${exp_tag} --save_interval=${save_interval}  ${debug} --statistics_info_fn=${statistics_info_fn} --training_setting=${training_setting} ${single_inst} ${sampling} --use_t=${use_t} 
# --batch_size=${batch_size}

# ${single_inst}
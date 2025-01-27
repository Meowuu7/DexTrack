


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
export single_inst=""
export save_interval=100
export exp_tag="allegro_test_bt_cylinder_trajtrans_3"
### training settings ###



### training settings for trajectory_translations_cond ###
export training_setting="trajectory_translations_cond"
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_cylinder_trajtrans_2/model000015700.pt"
export single_inst=""
export save_interval=100
export exp_tag="allegro_test_bt_cylinder_trajtrans_cond"
### training settings ###

export training_use_jointspace_seq=""


##### training setting #####
export training_setting='regular_training'
export resume_checkpoint=""
export single_inst=""
export save_interval=1000
export exp_tag="allegro_test_bt_rotationv1_jointspace"
export training_use_jointspace_seq="--training_use_jointspace_seq"
### TODO: change the satistics info ###
export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/save_info_v6_statistics.npy"

##### regular jtraining setting #####
export training_setting='regular_training'
export resume_checkpoint=""
export single_inst=""
export save_interval=1000
export exp_tag="allegro_test_bt_rotationv1_box"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export training_use_jointspace_seq=""
### TODO: change the satistics info ###
export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"


##### regular jtraining setting #####
export training_setting='regular_training'
export resume_checkpoint=""
export single_inst=""
export save_interval=1000
export exp_tag="allegro_test_bt_rotationv1_box_jtspace"
export exp_tag="allegro_test_bt_rotationv1_cylinder_jtspace"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export task_cond=""
# export training_use_jointspace_seq=""
### TODO: change the satistics info ###
export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"
### cylinder ###
export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_cylinder_wsmallsigma_svres_v2__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"


##### regular jtraining setting #####
# export training_setting='regular_training'
# export resume_checkpoint=""
# export single_inst=""
# export save_interval=1000
# export exp_tag="allegro_test_bt_rotationv1_box_jtspace_taskcond"
# export training_use_jointspace_seq="--training_use_jointspace_seq"
# export task_cond="--task_cond"
# # export training_use_jointspace_seq=""
# ### TODO: change the satistics info ###
# export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_rotationv1_box_jtspace/model000034000.pt"

####### ===== task space ===== ######
export training_setting='regular_training'
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_rotationv1_box_jtspace_taskspace/model000010000.pt"
export single_inst=""
export save_interval=10000
# export save_interval=10
export resume_checkpoint=""
export exp_tag="allegro_test_bt_rotationv1_box_jtspace_taskspace_v4"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export task_cond=""
# export training_use_jointspace_seq=""
### TODO: change the satistics info ###
export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"
export diff_task_space="--diff_task_space"
export diff_task_translations=""


# export training_setting='regular_training'
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_test_bt_rotationv1_box_jtspace_taskspace_v2/model000050000.pt" ## TODO
export single_inst=""
export save_interval=10000
# export save_interval=10
export exp_tag="allegro_test_bt_rotationv1_box_jtspace_taskspace_difftasktrans_v3"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export task_cond="--task_cond"
export diff_task_translations="--diff_task_translations"
export diff_task_space="--diff_task_space"
# export training_use_jointspace_seq=""
### TODO: change the satistics info ###
export statistics_info_fn="/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"



# kine_diff # 
export resume_checkpoint=""
export kine_diff="--kine_diff"
# export save_interval=10
export exp_tag="allegro_tracking_task_kine_diff"
export exp_tag="allegro_tracking_task_kine_diff_ncat_"
export exp_tag="allegro_tracking_task_kine_diff_ncatonlyobjpc_"
export task_cond=""
export diff_task_translations=""
export diff_task_space=""
export tracking_ctl_diff=""
# export training_use_jointspace_seq=
export save_interval=10000
export single_inst="--single_inst"
export exp_tag="allegro_tracking_task_kine_diff_v2_si_"

export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_/model001060000.pt'
export single_inst=""
export exp_tag="allegro_tracking_task_kine_diff_v2_"
# 

export single_inst="--single_inst"
# export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_si_/model001090000.pt'
# export single_inst=""
export exp_tag="allegro_tracking_task_kine_diff_v2_si_"


###### AE Diff ######
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_si_/model000020000.pt"
export AE_Diff="--AE_Diff"
export train_AE="--train_AE"
export train_Diff=""
export save_interval=10000
export single_inst=""
# export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_si_"
export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_"
###### AE Diff ######


###### Train Diffusion ######
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_/model000090000.pt"
export AE_Diff="--AE_Diff"
export train_AE=""
export train_Diff="--train_Diff"
export save_interval=10000
export single_inst=""
export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_"
# export single_inst="--single_inst"
# export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_si_"
###### Train Diffusion ######


###### AE_Diff task conditional #######
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_/model000170000.pt"
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_/model000290000.pt"
export AE_Diff="--AE_Diff"
export train_AE=""
export train_Diff="--train_Diff"
export task_cond="--task_cond"
export save_interval=10000
export single_inst="--single_inst"
export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_si_"
export single_inst=""
export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_"

# export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_trainallp_/model000330000.pt'
# export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_trainallp_/model000330000.pt'

export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_trainallp_"


# export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_/model000170000.pt'
export exp_ta='allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_traincond_'

###### AE_Diff task conditional ####### # if


# export single_inst="--single_inst"
# export exp_tag="allegro_tracking_task_kine_diff_ncatonlyobjpc_si_"

# tracking control sequence diffusion ##
###### ===== tracking control sequence diffusion ===== #####
export AE_Diff=""
export train_AE=""
export train_Diff=""
export kine_diff=""
export diff_task_space=""
export tracking_ctl_diff="--tracking_ctl_diff"
export training_use_jointspace_seq="--training_use_jointspace_seq"

# export exp_tag="allegro_tracking_task_ctl_diff_si_"

# export single_inst=""
# export exp_tag="allegro_tracking_task_ctl_diff_"

export single_inst=""
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_"

# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_v2_/_model002180000.pt"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_v2_"

export task_cond=""
# export single_inst="--single_inst"
export resume_checkpoint=""
# export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_v2_/_model002180000.pt'
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_si_"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_"


# export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_/model002240000.pt'
# export single_inst=""
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_"
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_/model000170000.pt"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_"
###### ===== tracking control sequence diffusio ===== #####




export multi_inst=""

# # ###### ==== task conditional training ==== ####
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_/model000170000.pt"
# # export single_inst="--single_inst"
# export task_cond="--task_cond"
# export save_interval=10000
# # export exp_tag="allegro_tracking_task_ctl_diff_wscale_taskcond_v2_"
# # export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_taskcond_"
# # export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_taskcond_v2_"


# export cond_diff_allparams="--cond_diff_allparams"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_taskcond_diffallp_"

# ## multiple instances ##
# export multi_inst="--multi_inst"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_taskcond_diffallp_multiinst_"
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_/model001390000.pt"
# # ###### ==== task conditional training ==== ####



##### Task control traj modeling #####
export AE_Diff="--AE_Diff"
export train_AE="--train_AE"
export train_Diff=""
export kine_diff=""
export diff_task_space=""
export tracking_ctl_diff="--tracking_ctl_diff"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export single_inst=""
export task_cond=""
export resume_checkpoint=""
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_fixedorder_"



export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_/model000480100.pt"
export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_fixedorder_/model000010000.pt"
export train_AE=""
export train_Diff="--train_Diff"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_fixedorder_"
export save_interval=10000
##### Task control traj modeling #####


#####  NOTE: TASK COND -- Task control traj modeling #####
export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_/model000480100.pt"
# resume the checkpoint #
export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_fixedorder_/model000030000.pt"
export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_/model001180000.pt"
export single_inst="--single_inst"
export single_inst=""
export task_cond="--task_cond"
export train_AE=""
export train_Diff="--train_Diff"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_"
#####  NOTE: TASK COND -- Task control traj modeling #####
export batch_size=8

##### Task control traj AE modeling (PyBullet) #####
export sim_platform='pybullet'
export slicing_ws=30
export slicing_data="--slicing_data"
export AE_Diff="--AE_Diff"
export train_AE="--train_AE"
export train_Diff=""
export kine_diff=""
export diff_task_space=""
export tracking_ctl_diff="--tracking_ctl_diff"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export single_inst=""
export task_cond=""
export resume_checkpoint=""
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_slicing_"
export batch_size=64


export train_AE=""
export train_Diff="--train_Diff"
export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_slicing_/model000050000.pt"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_slicing_"


export single_inst="--single_inst"
export single_inst=""
export task_cond="--task_cond"
export train_AE=""
export train_Diff="--train_Diff"
export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_slicing_/model000050000.pt"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_slicing_2_"
##### Task control traj AE modeling (PyBullet) #####


export task_cond_type='future'
export task_cond_type='history_future' 

export sub_task_cond_type='full'


export use_kine_obj_pos_canonicalization=""

##### Task control traj AE modeling (Isaac) #####
export sim_platform='isaac'
export grab_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy'
export grab_inst_tag_to_opt_stat_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res.npy'
export slicing_ws=30
# export slicing_ws=15
# export slicing_ws=5
# export slicing_ws=2
# export history_ws=30

# export slicing_ws=5
export history_ws=5
export step_size=1

# export slicing_ws=50
# export step_size=1
# export history_ws=50


# export slicing_ws=150
# export step_size=150
# export history_ws=2

# export slicing_ws=100
# export step_size=1
# export history_ws=2


# export slicing_ws=50
# export step_size=1
# export history_ws=2



# export slicing_ws=10
# export step_size=1
# export history_ws=2



export slicing_data="--slicing_data"
export AE_Diff="--AE_Diff"
export train_AE="--train_AE"
export train_Diff=""
export kine_diff=""
export diff_task_space=""
export tracking_ctl_diff="--tracking_ctl_diff"
export training_use_jointspace_seq="--training_use_jointspace_seq"
export single_inst=""
# export single_inst="--single_inst"
export task_cond=""
export resume_checkpoint=""

#  export history_ws=60 
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_slicing_"
#### add the slicing window size ####
#### Add the task cond type into `exp_tag` ####
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_v3_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_v4_
# history_ws
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_v4_

export glb_rot_use_quat="--glb_rot_use_quat"
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_glbrotquat_v4_

export glb_rot_use_quat=""
export use_kine_obj_pos_canonicalization="--use_kine_obj_pos_canonicalization"
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_glbrotquat_kineobjcanon_v4_

export batch_size=256
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_/model000010000.pt"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_res_"




export train_AE=""
export train_Diff="--train_Diff"
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_15_/model001500000.pt"
##### task cond in the history future format #####
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_taskcond_history_future_/model000490000.pt"
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_taskcond_history_future_v2_/model000350000.pt"
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_15_taskcond_history_future_v2_/model000410000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_5_taskcond_history_future_v2_/model000430000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_2_taskcond_history_future_v2_/model000440000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_taskcond_history_future_v3_/model003740000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_15_taskcond_history_future_v3_/model003770000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_5_taskcond_history_future_v3_/model002410000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_2_taskcond_history_future_v3_/model004300000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_taskcond_history_future_v4_/model000550000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_5_taskcond_history_future_v4_/model001000000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_2_taskcond_history_future_v4_/model001570000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_hist_30_taskcond_history_future_v4_w_tacograbinterp_/model000140000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000180000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_5_hist_5_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000310000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_50_hist_50_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000140000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_150_hist_2_step_150_taskcond_history_future_v4_w_tacograbinterp_/model000150000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_100_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000310000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_50_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000520000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_10_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000670000.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_evalv2_/model000420000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_hist_5_step_1_taskcond_history_future_glbrotquat_v4_w_tacograbinterp_/model000520000.pt'
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_v4_
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_glbrotquat_v4_
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_30_hist_5_step_1_taskcond_history_future_glbrotquat_kineobjcanon_v4_w_tacograbinterp_woquat_kineobjcanon_/model000190001.pt'
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_glbrotquat_kineobjcanon_v4_




################## Task Conditions ##################
export single_inst="--single_inst"
export single_inst=""
export task_cond="--task_cond"
export train_AE="" 
export train_Diff="--train_Diff"
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_/model000550000.pt"
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_taskcond_history_future_/model000830000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_taskcond_history_future_v2_/model000380000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_15_taskcond_history_future_v2_/model000440000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_5_taskcond_history_future_v2_/model000460000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_2_taskcond_history_future_v2_/model000470000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_taskcond_history_future_v3_/model004170000.pt' ## resume the checkpoint ##
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_15_taskcond_history_future_v3_/model004280000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_5_taskcond_history_future_v3_/model002920000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_2_taskcond_history_future_v3_/model004810000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_taskcond_history_future_v4_/model000720000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_5_taskcond_history_future_v4_/model001100000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_2_taskcond_history_future_v4_/model001670000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_5_taskcond_history_future_v4_/model002840000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_taskcond_history_future_v4_/model001210000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_taskcond_history_future_v4_w_tacograbinterp_/model000170000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000190000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_5_hist_5_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000310000.pt'
# export history_ws=${slicing_ws}
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
################## Task Conditions ##################

### history 30 ###
# export sub_task_cond_type='obj_shape_pose'
# export sub_task_cond_type='full_woornt'
# # export debug="--debug"


# ##### use a large future window size and a large history window size #####
# # export history_ws=30
# # export history_ws=60 
# # export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_histws_${history_ws}_taskcond_${task_cond_type}_v4_
# # export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_histws_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_v4_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_histws_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_


# ###### slicing ws = 150, history ws = 30, step size = 150 ######
# export history_ws=30
# export slicing_ws=150 
# export step_size=150
# export sub_task_cond_type='full_wohistory'

# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_150_hist_2_step_150_taskcond_history_future_v4_w_tacograbinterp_/model000370000.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
# ###### slicing ws = 150, history ws = 30, step size = 150 ######



# ###### slicing ws = 30, history ws = 30, step size = 1 ######
# export history_ws=30
# export history_ws=15

############### expeirmental settings ###############
export history_ws=5
export slicing_ws=30
export step_size=1
# export sub_task_cond_type='full'
# export expsetting_additional_tag='_newenc_noptenc'
# export expsetting_additional_tag='_newenc_noptenc_lesstr'
export sub_task_cond_type='full_wohistory'
# export expsetting_additional_tag='_newenc_noptenc_lesstrwohist_v2'
export expsetting_additional_tag='_newenc_noptenc_lesstrwohist_v3'
export expsetting_additional_tag='_newenc_noptenc_lesstrwohist_v3_ntransformers'

# export sub_task_cond_type='full'
# # export expsetting_additional_tag='_newenc_noptenc_lesstrwohist_v3'
# export expsetting_additional_tag='_newenc_noptenc_lesstrwohist_v3_ntransformers_whist5'
# # export expsetting_additional_tag='_newenc_quat_v4'
############### expeirmental settings ###############



############### expeirmental settings -- v2, history = 1 ###############
# export history_ws=1
# export slicing_ws=30
# export step_size=1
# export sub_task_cond_type='full'
# export expsetting_additional_tag='_newenc_noptenc_lesstr'
############### expeirmental settings -- v2, history = 1 ###############



export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model002620000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000190000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_evalv2_/model000760000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_evalv2_/model001400000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_evalv2_/model001870000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_30_step_1_taskcond_history_future_v4_w_tacograbinterp_evalv2_/model001740000.pt'
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_30_hist_5_step_1_taskcond_history_future_glbrotquat_kineobjcanon_v4_w_tacograbinterp_woquat_kineobjcanon_/model000620002.pt'

export expsetting_additional_tag='_kineobjcanon_nenctransformers_'
export expsetting_additional_tag='_kineobjcanon_trainenctransformers_'
export expsetting_additional_tag='debugging'
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
# ###### slicing ws = 30, history ws = 30, step size = 1 ######



###### slicing ws = 30, history ws = 30, step size = 1 ######
# export history_ws=30
# export slicing_ws=100
# export step_size=1
# export sub_task_cond_type='full'
# export expsetting_additional_tag='_newenc'

# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_100_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000770000.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
###### slicing ws = 30, history ws = 30, step size = 1 ######


###### slicing ws = 50, history ws = 30, step size = 1 ######
# export history_ws=30
# export slicing_ws=50
# export step_size=1
# export sub_task_cond_type='full'
# export expsetting_additional_tag='_newenc'

# # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_100_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000770000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_50_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model001380000.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
###### slicing ws = 50, history ws = 30, step size = 1 ######


###### slicing ws = 10, history ws = 5, step size = 1 ######
# export history_ws=5
# export slicing_ws=10
# export step_size=1
# export sub_task_cond_type='full'
# export expsetting_additional_tag='_newenc'

# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_10_hist_2_step_1_taskcond_history_future_v4_w_tacograbinterp_/model002440000.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
# ###### slicing ws = 50, history ws = 30, step size = 1 ######


# ###### slicing ws = 5, history ws = 1, step size = 1 ######
# export history_ws=1
# export slicing_ws=5
# export step_size=1
# export sub_task_cond_type='full'
# export expsetting_additional_tag='_newenc'

# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_5_hist_5_step_1_taskcond_history_future_v4_w_tacograbinterp_/model003110000.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_subtype_${sub_task_cond_type}_v4_
###### slicing ws = 50, history ws = 30, step size = 1 ######


##### Task control traj AE modeling (Isaac) #####




##### for grab-only training settings #####
export taco_inst_tag_to_optimized_res_fn=""
export taco_interped_fr_grab_tag=""
export taco_interped_data_sv_additional_tag=""
export exp_additional_tag=''
##### for grab-only training settings #####



##### for taco training settings #####
# export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval/statistics/data_inst_tag_to_optimized_res.npy"
# export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
# export grab_inst_tag_to_optimized_res_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy'
export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export taco_interped_fr_grab_tag="ori_grab_s2_phone_call_1"
export taco_interped_data_sv_additional_tag="v2"
export exp_additional_tag='w_tacograbinterp'
# export exp_additional_tag='w_tacograbinterp_evalv2'
export exp_additional_tag='w_tacograbinterp_wquat'
export exp_additional_tag='w_tacograbinterp_woquat_kineobjcanon'
export exp_additional_tag='w_tacograbinterp_woquat_kineobjcanon_debugging'
export exp_additional_tag='w_tacograbinterp_woquat_v3_kine_'
export exp_additional_tag='w_tacograbinterp_woquat_v3_kine_wtaco_'
export exp_additional_tag='w_tacograbinterp_woquat_v3_kine_canonv2_'
export exp_additional_tag=${exp_additional_tag}${expsetting_additional_tag}
export exp_tag=${exp_tag}${exp_additional_tag}_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_v4_

##### for taco training settings #####



export sampling=""




############ Samping Code ############
# export target_grab_inst_tag="ori_grab_s2_hammer_lift"
# export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-23-46-55/ts_to_hand_obj_obs_reset_1_sorted_best.npy"

# export target_grab_inst_tag="ori_grab_s8_hammer_lift"
# export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-37-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy"

# # export target_grab_inst_tag="ori_grab_s8_apple_lift" 
# # export target_grab_inst_opt_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-59-53/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

# # export target_grab_inst_tag="ori_grab_s8_banana_eat_1" 
# # export target_grab_inst_opt_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-52-28/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

# # export target_grab_inst_tag="ori_grab_s8_banana_peel_1" 
# # export target_grab_inst_opt_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-20-51-58/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export batch_size=256


# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v2_/model000510000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_15_taskcond_history_future_v2_/model000610000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v2_/model000630000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v2_/model000660000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/model000980000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/model001940000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/model001860000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/model001900000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v4_/model002560000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000190000.pt'
# export sampling="--sampling"
############ Samping Code ############


########### Default settings ###########
export obj_type_to_kinematics_traj_dict_fn=''
export canonicalize_features=""
export canonicalize_obj_pts=""
export centralize_info=""
export glb_feat_per_skip=1
export history_window_size=60
export scale_clip_data=""
export load_excuted_optimized_res=""
export use_clip_glb_features=""
export text_feature_version='v1'
########### Default settings ###########


export kine_diff_version="v1"

############ Kinematics Diff ############ 
export kine_diff="--kine_diff"
export AE_Diff="--AE_Diff"
export train_AE="--train_AE"
export train_Diff=""
export diff_task_space=""
export tracking_ctl_diff=""
export task_cond=""
export glb_rot_use_quat=""
export task_inherit_info_fn=""
export use_kine_obj_pos_canonicalization=""
export training_use_jointspace_seq="--training_use_jointspace_seq"
# export debug="--debug"
export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_
export resume_checkpoint=""

export kine_diff_version='v2'

# export use_taco_data="--use_taco_data"
export kine_diff_version='v1'
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_canonv2_

# export kine diff version ##

export train_AE=""
export train_Diff="--train_Diff"
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/model001070000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_/model000620000.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_/model000140001.pt'
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_canonv2_/model000050001.pt'
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_hist_${history_ws}_step_${step_size}_taskcond_${task_cond_type}_v4_
# ###### exp tag for the train diff ###### ## train diff ## slicing ws ##
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_v3data_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_canonv2_
# # # ###### vnew ######

export sampling=''
export target_grab_inst_tag=""


export num_frames=300
export maxx_inst_nn=2000


############################ Auto-regressive style trajectory forcasting (deterministic modeling) ############################
####### Auto-regressive style trajectory forcasting (deterministic modeling) #######
export training_setting='trajectory_forcasting'
####### Auto-regressive style trajectory forcasting (deterministic modeling) #######


export resume_checkpoint=''

####### Whether to add the trajectory features in the condition #######
export w_glb_traj_feat_cond="--w_glb_traj_feat_cond"
# export w_glb_traj_feat_cond=""
####### Whether to add the trajectory features in the condition #######

####### Canonicalize feature setting #######
# export canonicalize_features="--canonicalize_features"
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_trajfeatcond_canon_feats_
####### Canonicalize feature setting #######
############################ Auto-regressive style trajectory forcasting (deterministic modeling) ############################





####### Resume the checkpoitn setting #########
export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_/model000300001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_v2/model000220001.pt'
export resume_checkpoint=''
####### Resume the checkpoint setting #########

####### Whether to add the trajectory features in the condition #######
export w_glb_traj_feat_cond="--w_glb_traj_feat_cond"
# export w_glb_traj_feat_cond=""
####### Whether to add the trajectory features in the condition #######


####### Object type to kinematics traj dict setting #######
export obj_type_to_kinematics_traj_dict_fn='../../assets/obj_type_to_kinematics_traj_dict.npy'
export obj_type_to_kinematics_traj_dict_fn=''
####### Object type to kinematics traj dict setting #######


####### Canonicalization setting #######
# export canonicalize_obj_pts="--canonicalize_obj_pts"
####### Canonicalization setting #######

####### Forcasting window size ########
export forcasting_window_size=60
# export forcasting_window_size=30
# export forcasting_window_size=50
# export forcasting_window_size=100
# export forcasting_window_size=120
# export forcasting_window_size=140
# export forcasting_window_size=150
# export forcasting_window_size=160
# export forcasting_window_size=180
# export forcasting_window_size=300
####### Forcasting window size ########


####### Whether to add the history window index ########
export w_history_window_index=""
export w_history_window_index="--w_history_window_index"
####### Whether to add the history window index ########


############################ Diffusion  style trajectory forcasting (deterministic modeling) ############################
####### Auto-regressive style trajectory forcasting (deterministic modeling) #######
# export training_setting='trajectory_forcasting_diffusion'
####### Auto-regressive style trajectory forcasting (deterministic modeling) #######


###### Contact Diff #######
export diff_contact_sequence=""
# export diff_contact_sequence="--diff_contact_sequence"
###### Contact Diff #######



###### partial hand info and partial object info #########
export partial_hand_info=""
export partial_obj_info=""
export partial_obj_pos_info=""
# export partial_hand_info="--partial_hand_info"
# export partial_obj_info="--partial_obj_info"
# export partial_obj_pos_info="--partial_obj_pos_info"
###### partial hand info and partial object info #########



####### AE Diffusion Settings -- train_AE #######
# export training_setting='trajectory_forcasting_diffusion'
# export kine_diff="--kine_diff"
# export AE_Diff="--AE_Diff"
# export train_AE="--train_AE"
# export train_Diff=""
# export diff_task_space=""
# export tracking_ctl_diff=""
# export task_cond=""
# export resume_checkpoint=""
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws300_v2/model000240001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_v2/model000220001.pt'
####### AE Diffusion Settings -- train_AE #######


####### AE Diffusion Settings -- train_Diff #######
# export training_setting='trajectory_forcasting_diffusion'
# export kine_diff="--kine_diff"
# export AE_Diff="--AE_Diff"
# export train_AE=""
# export train_Diff="--train_Diff"
# export diff_task_space=""
# export tracking_ctl_diff=""
# export task_cond="" # 
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE/model000100000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_v2/model000220001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2/model002580001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2/model003550000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws100_v2/model002170000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws300_v2/model000090001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2_partialhand_/model004490001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2_partialobj_/model004490001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_ws60_v2_partialobjposinfo_/model000210001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_ws60_canoninput_v2_partialobjposinfo_/model000160001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2_partialobjposinfo_/model000390002.pt'
####### AE Diffusion Settings -- train_Diff #######




####### AE Diffusion Settings -- train_Diff with task cond #######
# export training_setting='trajectory_forcasting_diffusion'
# export kine_diff="--kine_diff"
# export AE_Diff="--AE_Diff"
# export train_AE=""
# export train_Diff="--train_Diff"
# export diff_task_space=""
# export tracking_ctl_diff=""
# export task_cond="--task_cond"
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v2/model000390002.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v4/model000970002.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2/model004410001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2/model004990001.pt'
# # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2_partialhand_/model004570002.pt'
# # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws300_v2/model001370002.pt'
# # # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws100_v2/model002900001.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2_partialobjposinfo_/model000390002.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_canoninput_v2_partialobjposinfo_/model000350002.pt'
####### AE Diffusion Settings -- train_Diff with task cond #######

####### Whether to add the history window index -- for diffusion conditions ########
export w_history_window_index=""
export w_history_window_index="--w_history_window_index" # whether to add history window index is not a critical thing here 
####### Whether to add the history window index -- for diffusion conditions ######## 




###### Whether to use the masked future conditional input ########
export w_masked_future_cond=""
# export w_masked_future_cond="--w_masked_future_cond" # the partial to complete trajectory translation problem where the mdoel is trained to preoduce teh full trajecotry by conditioning on partial information sequence #
###### Whether to use the masked future conditional input ########


###### Conditional model setting #######
export st_ed_state_cond=""
# export st_ed_state_cond="--st_ed_state_cond"
###### Conditional model setting #######


####### Canonicalization setting ####### -> avg and std level canonicalization
export canonicalize_obj_pts="--canonicalize_obj_pts"
####### Canonicalization setting ####### -> avg and std level canonicalization



####### Additional data settings and conditional settings #######
export centralize_info="--centralize_info"
export glb_feat_per_skip=6
export history_window_size=1
export scale_clip_data="--scale_clip_data"
####### Additional data settings and conditional settings #######


####### Canonicalization setting #######
export canonicalize_obj_pts=""
####### Canonicalization setting #######




####### Additional data settings and conditional settings #######
export centralize_info="--centralize_info"
export glb_feat_per_skip=6
export history_window_size=1
export scale_clip_data=""
export canonicalize_obj_pts=""
####### Additional data settings and conditional settings #######


####### Inversed kinematics info forecasting frequency #######
export inv_kine_freq=1
# export inv_kine_freq=5
####### Inversed kinematics info forecasting frequency #######


export excuted_inst_tag_to_optimized_res=""

####### Whether to use optimized trajectories for the training #######
# export load_excuted_optimized_res="--load_excuted_optimized_res"
# export excuted_inst_tag_to_optimized_res='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests6_/statistics/data_inst_tag_to_optimized_res.npy'
# export excuted_inst_tag_to_optimized_res='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res_merged.npy'
###### Whether to use optimized trajectories for the training #######



####### Whether to use clip features as the global features #######
export use_clip_glb_features="--use_clip_glb_features"
export clip_feat_dim=512
export text_feature_version='v2'
####### Whether to use clip features as the global features #######


####### Inversed forecasting freqeuency #######
export inv_forecasting_freq=1
export inv_forecasting_freq=5

export forcasting_window_size=10
export inv_forecasting_freq=10

export forcasting_window_size=6
export inv_forecasting_freq=10
export centralize_info=""
####### Inversed forecasting freqeuency #######


# ####### Canonicalization setting #######
# export canonicalize_="--canonicalize_obj_pts"
# ####### Canonicalization setting #######


####### Trajectory and object pc encoding and decoding training ########
# export training_setting='trajectory_encdec'
#### whether to use the sampling mode ####
# export sampling="--sampling"
####### Trajectory and object pc encoding and decoding training ########


####### Canonicalize feature setting #######
# export canonicalize_features="--canonicalize_features"
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_trajfeatcond_canon_feats_
####### Canonicalize feature setting #######
############################ Diffusion  style trajectory forcasting (deterministic modeling) ############################


# export save_interval=10


####### Exp Tags #######
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_trajfeatcond_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_trajfeatcond_skippedglb_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_v2_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_v2data_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_v2data_wcanon_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_v2_wcanon_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws100_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws120_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws150_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws160_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_v2_resume
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v4
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v4_taskcond
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws100_histindex_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws50_histindex_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws120_histindex_ # killed 
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws130_histindex_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws140_histindex_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws150_histindex_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws120_histindex_
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_obj_feat_training_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v2_taskcond
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2_contact
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws100_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws100_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws100_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_histindex_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws300_v2
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws100_histindex_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws300_v2_resume
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_histindex_maskedfuture_v2
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2_partialhand_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2_partialobj_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2_partialhand_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2_partialobj_
# generate the partial hand <- while the conditions may not need to change ## tsk cond #
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2_partialhand_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2_partialhand_condhandonly_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_partialhand_ws300_v2
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_stedgoalcond_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_ws60_v2_partialobjposinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_v2_partialobjposinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_ws60_canoninput_v2_partialobjposinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_canoninput_v2_partialobjposinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2_partialobjposinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_canoninput_v2_partialobjposinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws120_v2_tessssss
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_centralize_partialhandinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_centralize_scaleclip_partialhandinfo_
export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_centralize_handobjinfo_


# clip features #

export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_handobjinfo_wglbfeat_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_invfreq5_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_woptdata_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_wclipglbfeat_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_wclipglbfeatv3_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_wclipglbfeatv2_woptresall_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws150_v2
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_wclipglbfeat_invfreq5_
export exp_tag=allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_wclipglbfeat_invfreq5v2_
export exp_tag=allegro_tracking_traj_forcasting_ws60_wglbfeat_wclipglbfeat_invfreq10ws10_
export exp_tag=allegro_tracking_traj_forcasting_ws60_wglbfeat_wclipglbfeat_invfreq10ws6_
export exp_tag=allegro_tracking_traj_forcasting_ws60_wglbfeat_wclipglbfeat_woptresall_invfreq10ws6_
# =======
export exp_tag=allegro_tracking_traj_forcasting_ws60_wglbfeat_wclipglbfeat_invfreq10ws6_s2_
export exp_tag=allegro_tracking_traj_forcasting_ws60_wglbfeat_wclipglbfeat_invfreq10ws6_s2_justgpu_
####### Exp Tags ####### goal pos of the the object #

# 

export batch_size=128




# ### they are settings for the control sequence diffusion ###
# export glb_rot_use_quat=""
# export use_kine_obj_pos_canonicalization=""
# ### they are settings for the control sequence diffusion ###
# export task_cond="--task_cond" 
# # export task_inherit_info_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy"
# export task_inherit_info_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task_list.npy"
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnew_/model001070000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnew_/model001200000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_v3data_/model000330002.pt'
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_v3data_taskcond_




# ######### Sampling Code #########
# export sampling='--sampling'
# export data_statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_kinematics_diff_w_tacograbinterp_woquat_kineobjcanon_debuggingdebugging.npy'
# # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/tracking_ori_grab_s10_apple_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-15-22-32
# # export target_grab_inst_tag="ori_grab_s10_apple_eat_1"
# export target_grab_inst_tag="ori_grab_s1_hammer_use_1"
# export sampling="--sampling"
# # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_/model001210000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_/model001210002.pt'
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_sample_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_


# # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_/model000010000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_/model000010001.pt'
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_/model000030001.pt'
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_kinediffv2_samples_
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_canonv2_/model000040001.pt'
# # export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_
# export exp_tag=allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_canonv2_samples_
# export sampling="--sampling"
# export target_grab_inst_tag=""
# # export data_statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_kinematics_diff_w_tacograbinterp_woquat_v3_kine_debugging.npy'
# # export data_statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_kinematics_diff_w_tacograbinterp_woquat_v3_kine_wtaco_debugging.npy'
# export data_statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_kinematics_diff_w_tacograbinterp_woquat_v3_kine_canonv2_debugging.npy'
####### Sampling Code #########
############ Kinematics Diff ############




export use_t=1000

# export batch_size=4

export cuda_ids=4




export debug=""
# export debug="--debug"
# export save_interval=10

# 



# bash scripts/K2_run_main_3d_pc_traj_forcasting.sh


CUDA_VISIBLE_DEVICES=${cuda_ids} python main_3d_pc.py  ${task_cond}  --resume_checkpoint=${resume_checkpoint} --exp_tag=${exp_tag} --save_interval=${save_interval}  ${debug} --statistics_info_fn=${statistics_info_fn} --training_setting=${training_setting} ${single_inst} --batch_size=${batch_size} ${training_use_jointspace_seq} ${diff_task_space} ${diff_task_translations} ${kine_diff} ${tracking_ctl_diff} ${sampling} --use_t=${use_t} --target_grab_inst_tag=${target_grab_inst_tag} --target_grab_inst_opt_fn=${target_grab_inst_opt_fn} ${AE_Diff} ${train_AE} ${train_Diff} ${cond_diff_allparams} ${multi_inst} --slicing_ws=${slicing_ws} ${slicing_data} --sim_platform=${sim_platform} --grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} --grab_inst_tag_to_opt_stat_fn=${grab_inst_tag_to_opt_stat_fn} --task_cond_type=${task_cond_type} --history_ws=${history_ws} --taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn} --taco_interped_fr_grab_tag=${taco_interped_fr_grab_tag} --taco_interped_data_sv_additional_tag=${taco_interped_data_sv_additional_tag} --exp_additional_tag=${exp_additional_tag}  --sub_task_cond_type=${sub_task_cond_type} --task_inherit_info_fn=${task_inherit_info_fn} ${glb_rot_use_quat} ${use_kine_obj_pos_canonicalization} --data_statistics_info_fn=${data_statistics_info_fn} --kine_diff_version=${kine_diff_version} ${use_taco_data} --num_frames=${num_frames} --maxx_inst_nn=${maxx_inst_nn} ${w_glb_traj_feat_cond} ${canonicalize_features} --obj_type_to_kinematics_traj_dict_fn=${obj_type_to_kinematics_traj_dict_fn} ${canonicalize_obj_pts} --forcasting_window_size=${forcasting_window_size} ${w_history_window_index} ${partial_hand_info} ${partial_obj_info} ${diff_contact_sequence} ${w_masked_future_cond} ${st_ed_state_cond} ${partial_obj_pos_info} ${centralize_info} --glb_feat_per_skip=${glb_feat_per_skip} --history_window_size=${history_window_size} ${scale_clip_data} --inv_kine_freq=${inv_kine_freq} ${load_excuted_optimized_res} --excuted_inst_tag_to_optimized_res=${excuted_inst_tag_to_optimized_res} ${use_clip_glb_features} --clip_feat_dim=${clip_feat_dim} --text_feature_version=${text_feature_version} --inv_forecasting_freq=${inv_forecasting_freq}
   


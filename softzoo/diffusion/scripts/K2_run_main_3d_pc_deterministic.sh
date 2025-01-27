


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





##### Task control traj AE modeling (Isaac) #####
export sim_platform='isaac'
export grab_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy'
export grab_inst_tag_to_opt_stat_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res.npy'
# export slicing_ws=30
# export slicing_ws=15
# export slicing_ws=5
export slicing_ws=1
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
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_v3_
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_v4_
export batch_size=64
# export resume_checkpoint="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_/model000010000.pt"
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_isaac_slicing_res_"




export train_AE=""
export train_Diff="--train_Diff"
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_v4_


export single_inst="--single_inst"
export single_inst=""
export task_cond="--task_cond"
export train_AE=""
export train_Diff="--train_Diff"

export resume_checkpoint=''



export history_ws=30
# export history_ws=60 
# export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_taskcond_${task_cond_type}_v4_
export exp_tag=allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_${slicing_ws}_histws_${history_ws}_taskcond_${task_cond_type}_deterministic_v2_

##### Task control traj AE modeling (Isaac) #####






export sampling=""


############ Samping Code ############
# # #### Training set test ####
# # export target_grab_inst_tag="ori_grab_s2_hammer_use_1"
# export target_grab_inst_tag="ori_grab_s2_hammer_lift"
# # export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/tds_rl_exp_ctlfreq_10_rew_v2new_pkretar_/logs_PPO/allegro_ctlfreq_10_bt_grab_ori_grab_s2_hammer_use_1_trajtag_ori_grab_s2_hammer_use_1_tracking_v3_mact_smass_ctlfreq10_ctlbiaskinematics_rewv2_new__objtype_ori_grab_s2_hammer_use_1_objinitxyz_0.0_0.0_0.0_/fr_objname_ori_grab_s2_hammer_use_1_decom_5_trajobjname_ori_grab_s2_hammer_use_1/best_res_tau_4848_best_rew_rnk_3_rew_64.487.npy"
# export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-23-46-55/ts_to_hand_obj_obs_reset_1_sorted_best.npy"


# export target_grab_inst_tag="ori_grab_s8_hammer_lift"
# # export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/tds_rl_exp_ctlfreq_10_rew_v2new_pkretar_/logs_PPO/allegro_ctlfreq_10_bt_grab_ori_grab_s2_hammer_use_1_trajtag_ori_grab_s2_hammer_use_1_tracking_v3_mact_smass_ctlfreq10_ctlbiaskinematics_rewv2_new__objtype_ori_grab_s2_hammer_use_1_objinitxyz_0.0_0.0_0.0_/fr_objname_ori_grab_s2_hammer_use_1_decom_5_trajobjname_ori_grab_s2_hammer_use_1/best_res_tau_4848_best_rew_rnk_3_rew_64.487.npy"
# export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-37-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
# # #### Training set test ####

# export target_grab_inst_tag="ori_grab_s8_apple_lift" 
# export target_grab_inst_opt_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-59-53/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

# export target_grab_inst_tag="ori_grab_s8_banana_eat_1" 
# export target_grab_inst_opt_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-52-28/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

# ###### the banana peel 1 ######
# ## we should have the target_grab_inst_tag here for the inst opt fn ##
# ## target grab inst tag ##
# export target_grab_inst_tag="ori_grab_s8_banana_peel_1" 
# export target_grab_inst_opt_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-20-51-58/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export batch_size=256

# # export target_grab_inst_tag="ori_grab_s8_hammer_lift"
# # export target_grab_inst_opt_fn="/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s8_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-37-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy"

# # ### directly apply that into the histroy ###
# # export target_grab_inst_tag="ori_grab_s2_phone_call_1"
# # export target_grab_inst_opt_fn="/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s8_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-37-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy"

# # #### Test set test ####
# # export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_slicing_/model001760000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v2_/model000510000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_15_taskcond_history_future_v2_/model000610000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v2_/model000630000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v2_/model000660000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/model000980000.pt'
# ### ws = 30 ckpt for the sampling ###
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/model001940000.pt'
# ### ws = 5 ckpt for sampling ###
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/model001860000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/model001900000.pt'
# export resume_checkpoint='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v4_/model002560000.pt'
# export sampling="--sampling"
# # export target_grab_inst_tag="ori_grab_s3_duck_inspect_1"
# # export target_grab_inst_opt_fn="/cephfs/xueyi/uni_manip/tds_rl_exp_ctlfreq_10_rew_v2new_pkretar_/logs_PPO/allegro_ctlfreq_10_bt_grab_ori_grab_s3_duck_inspect_1_trajtag_ori_grab_s3_duck_inspect_1_tracking_v3_mact_smass_ctlfreq10_ctlbiaskinematics_rewv2_new__objtype_ori_grab_s3_duck_inspect_1_objinitxyz_0.0_0.0_0.0_/best_res_tau_15192_best_rew_rnk_3_rew_62.95.npy"
# # #### Test set test ####
############ Samping Code ############



# export debug="--debug"

export use_t=1000


export use_deterministic="--use_deterministic"
# export save_interval=100



export cuda_ids=6
# ##### ==== sampling code ==== #####
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_taskcond_/model000630000.pt"
# export sampling="--sampling"




# ##### ==== sampling code ==== #####
# export target_grab_inst_tag="ori_grab_s2_mug_pass_1"
# export target_grab_inst_opt_fn="/cephfs/yilaa/uni_manip/tds_rl_exp_ctlfreq_10_new/logs_PPO/allegro_ctlfreq_10_bt_grab_ori_grab_s2_mug_pass_1_trajtag_ori_grab_s2_mug_pass_1_tracking_v3_mact_smass_ctlfreq10_ctlbiaskinematics__objtype_ori_grab_s2_mug_pass_1_objinitxyz_0.0_0.0_0.0_/best_res_tau_29602_best_rew_rnk_2_rew_15.294.npy"
# ##### ==== sampling code ==== #####


# /root/diffsim/softzoo/softzoo/diffusion/scripts/K2_run_main_3d_pc_deterministic.sh



# bash scripts/K2_run_main_3d_pc_deterministic.sh 



CUDA_VISIBLE_DEVICES=${cuda_ids} python main_3d_pc.py  ${task_cond}  --resume_checkpoint=${resume_checkpoint} --exp_tag=${exp_tag} --save_interval=${save_interval}  ${debug} --statistics_info_fn=${statistics_info_fn} --training_setting=${training_setting} ${single_inst} --batch_size=${batch_size} ${training_use_jointspace_seq} ${diff_task_space} ${diff_task_translations} ${kine_diff} ${tracking_ctl_diff} ${sampling} --use_t=${use_t} --target_grab_inst_tag=${target_grab_inst_tag} --target_grab_inst_opt_fn=${target_grab_inst_opt_fn} ${AE_Diff} ${train_AE} ${train_Diff} ${cond_diff_allparams} ${multi_inst} --slicing_ws=${slicing_ws} ${slicing_data} --sim_platform=${sim_platform} --grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} --grab_inst_tag_to_opt_stat_fn=${grab_inst_tag_to_opt_stat_fn} --task_cond_type=${task_cond_type} --history_ws=${history_ws} ${use_deterministic}




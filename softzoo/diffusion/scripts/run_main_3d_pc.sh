


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
export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_si_/model001090000.pt'
# export single_inst=""
export exp_tag="allegro_tracking_task_kine_diff_v2_si_"


###### AE Diff ######
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_si_/model000020000.pt"
export AE_Diff="--AE_Diff"
export train_AE="--train_AE"
export train_Diff=""
export save_interval=10000
export single_inst=""
# export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_si_"
export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_"
###### AE Diff ######


###### Train Diffusion ######
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainAE_/model000090000.pt"
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
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_/model000290000.pt"
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
export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_trainallp_/model000330000.pt'

export exp_tag="allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_taskcond_trainallp_"


export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_/model000170000.pt'
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

export exp_tag="allegro_tracking_task_ctl_diff_si_"

# export single_inst=""
# export exp_tag="allegro_tracking_task_ctl_diff_"

export single_inst=""
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_"

export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_v2_/_model002180000.pt"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_v2_"

export task_cond=""
export single_inst="--single_inst"
export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_v2_/_model002180000.pt'
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_si_"


export resume_checkpoint='/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_/model002240000.pt'
export single_inst=""
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_"
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_kine_diff_v2_AE_Diff_trainDiff_/model000170000.pt"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_"
###### ===== tracking control sequence diffusio ===== #####



# # ###### ==== task conditional training ==== ####
export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_/model003870000.pt"
# export single_inst="--single_inst"
export task_cond="--task_cond"
export save_interval=10000
# export exp_tag="allegro_tracking_task_ctl_diff_wscale_taskcond_v2_"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_taskcond_"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_taskcond_v2_"


export cond_diff_allparams="--cond_diff_allparams"
export exp_tag="allegro_tracking_task_ctl_diff_wscale_rew_v2_new_taskcond_v2_trainallp_"
# # ###### ==== task conditional training ==== ####

export sampling=""
export use_t=1000
export target_grab_inst_tag=''
export target_grab_inst_opt_fn=''

export batch_size=8

# export batch_size=4

export cuda_ids=1

# ##### ==== sampling code ==== #####
# export resume_checkpoint="/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_taskcond_/model000630000.pt"
# export sampling="--sampling"
# ### === free sampling with conditions === ###

# # target_grab_inst_tag: '' # #
# # target_grab_inst_opt_fn: '' # #

# ### === sampling with the target instance === ###
# export target_grab_inst_tag="ori_grab_s2_mug_pass_1" #
# export target_grab_inst_opt_fn="/cephfs/yilaa/uni_manip/tds_rl_exp_ctlfreq_10_new/logs_PPO/allegro_ctlfreq_10_bt_grab_ori_grab_s2_mug_pass_1_trajtag_ori_grab_s2_mug_pass_1_tracking_v3_mact_smass_ctlfreq10_ctlbiaskinematics__objtype_ori_grab_s2_mug_pass_1_objinitxyz_0.0_0.0_0.0_/best_res_tau_29602_best_rew_rnk_2_rew_15.294.npy"
# # taco sequences ##### taco sequences ##
# ### === sampling with the target instance === ### ##
# ##### ==== sampling code ==== ####


# bash scripts/run_main_3d_pc.sh

CUDA_VISIBLE_DEVICES=${cuda_ids} python main_3d_pc.py  ${task_cond}  --resume_checkpoint=${resume_checkpoint} --exp_tag=${exp_tag} --save_interval=${save_interval}  ${debug} --statistics_info_fn=${statistics_info_fn} --training_setting=${training_setting} ${single_inst} --batch_size=${batch_size} ${training_use_jointspace_seq} ${diff_task_space} ${diff_task_translations} ${kine_diff} ${tracking_ctl_diff} ${sampling} --use_t=${use_t} --target_grab_inst_tag=${target_grab_inst_tag} --target_grab_inst_opt_fn=${target_grab_inst_opt_fn} ${AE_Diff} ${train_AE} ${train_Diff} ${cond_diff_allparams}




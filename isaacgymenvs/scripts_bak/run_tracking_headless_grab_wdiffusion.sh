

export numEnvs=10240
export minibatch_size=10240

export numEnvs=8000
export minibatch_size=8000

export mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy'

export test=False

# export checkpoint=''
export checkpoint=runs/Humanoid_03-09-34-46/nn/Humanoid.pth 

export checkpoint=runs/Humanoid_03-16-01-13/nn/Humanoid.pth


export goal_cond=True
export goal_cond=False


### Object task setting ###
export object_name=''
export mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy'
### Object task setting ###

export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'


# export object_name='ori_grab_s2_camera_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_camera_pass_1.npy'


# export object_name='ori_grab_s2_mug_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_mug_pass_1.npy'


# export object_name='ori_grab_s2_hand_inspect_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_hand_inspect_1.npy'

# export object_name='ori_grab_s2_flashlight_on_2'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_flashlight_on_2.npy'

# export object_name='ori_grab_s2_headphones_lift'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_headphones_lift.npy'

# # /cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_pass_1.npy

# export object_name='ori_grab_s2_phone_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_pass_1.npy'

export use_kinematics_bias=False

# useRelativeControl #
export use_relative_control=True
export use_relative_control=False

export w_obj_ornt=False

### Object task setting ###
export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'
export checkpoint=''
export goal_cond=True
export dt=0.0166
export use_kinematics_bias=True
### Object task setting ###



### Object task setting ###
export object_name='ori_grab_s2_headphones_lift'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift.npy'
export checkpoint=''
export goal_cond=True
export dt=0.0166
export use_kinematics_bias=False
export use_kinematics_bias=True
### Object task setting ###

# hand inspect
### hand inspect sequence with the object ornt reward ###
export object_name='ori_grab_s2_hand_inspect_1'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_hand_inspect_1.npy'
export checkpoint=''
export goal_cond=False 
export dt=0.0166
export use_kinematics_bias=False
export w_obj_ornt=True
### hand inspect sequence with the object ornt reward ###


# kinematics_only # 


### Object task setting ###
# export goal_cond=False
# export dt=0.0166 # 1 / 60 s
# export dt=0.00833  # 1 / 120 s
# export test=True
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_06-03-10-52/nn/last_tracking_ori_grab_s2_train_lift_ep_1900_rew_152.4565.pth'
### Object task setting ###


### Object task setting ###
# export goal_cond=True
# export dt=0.0166 # 1 / 60 s
# export test=True
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_05-13-20-04/nn/last_tracking_ori_grab_s2_train_lift_ep_10000_rew__106.25_.pth'
### Object task setting ###

### Training and Test Setting ###
# export object_name='ori_grab_s2_headphones_lift'
# export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift.npy'
# export goal_cond=False
# export test=True
# export test=False 
# export checkpoint='/data1/xueyi/exp/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-19-03-28/nn/last_Humanoid_ep_10000_rew__218.01_.pth'
# export checkpoint=''
# export dt=0.0166 # 1 / 60 s
# export dt=0.00833  
# export goal_cond=True
### Training and Test Setting ###


# ### Training setting ###
export object_name='ori_grab_s2_camera_pass_1'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_camera_pass_1.npy'
export test=False 
export goal_cond=False
export checkpoint=''
# ### Training setting ###



# ### Training setting ###
# export object_name='ori_grab_s2_flute_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_flute_pass_1.npy'
# export test=False 
# export goal_cond=False # goal cond #
# export checkpoint=''
# ### Training setting ###



# ### Test setting ###
# export object_name='ori_grab_s2_train_lift'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_05-13-33-32/nn/last_tracking_ori_grab_s2_train_lift_ep_2850_rew_153.66248.pth'
# export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'
# export goal_cond=False
# export test=True
# export dt=0.00833  
# # export dt=0.0166 # 1 / 60 s
# ### Test setting ###

export use_kinematics_bias_wdelta=False
export use_unified_canonical_state=False

####### Object settings #######
export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'


export object_name='ori_grab_s2_hand_inspect_1'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_hand_inspect_1.npy'


# export object_name='ori_grab_s2_flute_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_flute_pass_1.npy'



# export object_name='ori_grab_s2_headphones_lift'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_headphones_lift.npy'

# export object_name='ori_grab_s2_flashlight_on_2'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_flashlight_on_2.npy'

# export object_name='ori_grab_s2_phone_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_pass_1.npy'
# # ## phone pass #

# export object_name='ori_grab_s2_phone_call_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1.npy'
# # # ## phone pass #
####### Object settings #######


####### whether to use the pure_state as the observation #####
export obs_type='pure_state'






# export kinematics_only=True
# export object_name='ori_grab_s2_train_lift'
export checkpoint=''

export goal_cond=False # 
export test=False # True
# export dt=0.00833  
export dt=0.0166 # 1 / 60 s 
export use_kinematics_bias=True
export use_kinematics_bias=False
export use_relative_control=True
export glb_trans_vel_scale=1
export glb_rot_vel_scale=1
# export kinematics_only=True
##### tracking headless ######




##### tracking the camear pass sequence #####
# export object_name='ori_grab_s2_camera_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_camera_pass_1.npy'
# export checkpoint=''
# export use_unified_canonical_state=True
##### tracking the camear pass sequence #####







####### use canonical state ########
export use_canonical_state=True

export separate_stages=False
export additiona_tag="canonstate"


export use_unified_canonical_state=True
export use_unified_canonical_state=False
# export use_canonical_state=False
# export additiona_tag="ncanon_liftingtargets"


export separate_stages=True
export additiona_tag="purecanonstate_liftingtargets"
# export additiona_tag="ncanonstate_liftingtargets"

export additiona_tag="purecanonstate_liftingtargets_lowwerzero"
export additiona_tag="canonstate_liftingtargets_lowwerzero"
# export additiona_tag="ncanonstate_liftingtargets"



export use_unified_canonical_state=False
export use_canonical_state=False
export use_kinematics_bias=True
export additiona_tag="kinebias"
export kinematics_only=True
export kinematics_only=False
export use_fingertips=False
export use_fingertips=True
export rigid_obj_density=500
# export rigid_obj_density=100


##### use_kinematics_bias_wdelta ######
export use_kinematics_bias_wdelta=True
export additiona_tag="kinebais_wdelta"
export additiona_tag="kinebais_wdelta_rewhandpos_dist_"
export obs_type='pure_state_wref_wdelta'
##### use_kinematics_bias_wdelta ######

export glb_trans_vel_scale=0.1
export glb_rot_vel_scale=0.1

export w_obj_ornt=False

export hand_pose_guidance_glb_trans_coef=0.6
export hand_pose_guidance_glb_rot_coef=0.1
export hand_pose_guidance_fingerpose_coef=0.1

export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.5

### Train_lift_1 #####
# export obs_type='pure_state_wref'
# # export additiona_tag="kinebias_"
# export glb_trans_vel_scale=10
# export glb_rot_vel_scale=1
# export goal_cond=True
### Train_lift_1 #####


######## test setting ########
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_08-03-48-05/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate.pth'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_08-04-56-44/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate.pth'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_08-06-17-40/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate.pth'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_liftingtargets_08-06-57-46/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_liftingtargets.pth'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_08-06-17-40/nn/last_tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_ep_1850_rew_385.39832.pth'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_liftingtargets_08-08-38-07/nn/last_tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_liftingtargets_ep_1250_rew_285.8653.pth'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_08-06-17-40/nn/last_tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_ep_2800_rew_386.63733.pth'
# # export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_liftingtargets_density_100_08-18-32-47/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_canonstate_liftingtargets_density_100.pth'
# export checkpoint='runs/tracking_ori_grab_s2_camera_pass_1_obs_pure_state_density_500_purecanonstate_liftingtargets_08-19-36-31/nn/tracking_ori_grab_s2_camera_pass_1_obs_pure_state_density_500_purecanonstate_liftingtargets.pth'
# # # export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_density_500_kinebias_09-01-21-00/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_density_500_kinebias.pth'
export use_kinematics_bias_wdelta=False
export obs_type="pure_state_wref"

export use_kinematics_bias_wdelta=True
export obs_type="pure_state_wref_wdelta"


export glb_trans_vel_scale=0.5
export glb_rot_vel_scale=0.5
export dofSpeedScale=20

export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.0

# kine # 
# export use_kinematics_bias_wdelta=False
# export obs_type="pure_state_wref"
# # export additional_tag=kinebias_rewfingerdist_${rew_finger_obj_dist_coef}_rewdeltahandpose_${rew_delta_hand_pose_coef}

# export glb_trans_vel_scale=10
# export glb_rot_vel_scale=10
# export dofSpeedScale=40

export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.0

export rew_finger_obj_dist_coef=0.0
export rew_delta_hand_pose_coef=0.5
export separate_stages=False

export object_name='ori_grab_s2_apple_pass_1'
# export object_name='ori_grab_s2_hammer_use_1'
export object_name='ori_grab_s2_cylindermedium_inspect_1'
export object_name='ori_grab_s2_airplane_fly_1'
export object_name='ori_grab_s2_toothpaste_lift'
export object_name='ori_grab_s2_knife_pass_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export checkpoint='runs/tracking_ori_grab_s2_cylindermedium_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-18-23-11/nn/tracking_ori_grab_s2_cylindermedium_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_knife_pass_1_OPTFR_ori_grab_s2_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-00-05-56/nn/tracking_ori_grab_s2_knife_pass_1_OPTFR_ori_grab_s2_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-22-42-47/ts_to_hand_obj_obs_reset_1.npy'

export object_name='ori_grab_s2_phone_call_1_nf_300'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export checkpoint='runs/tracking_ori_grab_s2_phone_call_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-12-51-55/nn/tracking_ori_grab_s2_phone_call_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export pre_optimized_traj=''


export object_name='ori_grab_s2_hammer_lift'
export object_name='ori_grab_s8_hammer_lift'
# export object_name='ori_grab_s8_apple_lift'
export object_name='ori_grab_s8_duck_inspect_1'
# export object_name='ori_grab_s8_hand_lift'
# export object_name='ori_grab_s8_mouse_lift'
# export object_name='ori_grab_s9_toothpaste_pass_1'
# export object_name='ori_grab_s2_toothpaste_lift'
# # export object_name='ori_grab_s8_banana_eat_1'
# export object_name='ori_grab_s8_banana_peel_1'
export object_name='ori_grab_s2_apple_lift'
# export object_name='ori_grab_s10_hand_inspect_1'
# export object_name='ori_grab_s2_hand_inspect_1'
# export object_name='ori_grab_s10_hammer_use_2'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy

### Set the checkpoint to the ckpt to the diffusion model ###
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/model000760000.pt'
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/model000870000.pt'
### Set the checkpoint to the path to the statistics info fn ###
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_15.npy'
export slicing_ws=30

export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_2_step_1.npy'


export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v4_/model002560000.pt'
export slicing_ws=2

export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/model002120000.pt'
export slicing_ws=5
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_5_step_2.npy'

export history_ws=${slicing_ws}


# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_60_taskcond_history_future_v4_/model002430000.pt'
# export slicing_ws=30
# export history_ws=60
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_15.npy'

### model setting for slicing_ws=30, history_ws=30, step_size=1 ####
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000190000.pt'
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/model000560000.pt'
export slicing_ws=30
export history_ws=30
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp.npy'

export sub_task_cond_type='full'


### wp history ###
export slicing_ws=30
export history_ws=5
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_wohistory_v4_w_tacograbinterp_newenc_noptenc_lesstrwohist_/model000410000.pt'
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstr.npy'
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstrwohist.npy'
export sub_task_cond_type='full_wohistory'

### with history ###
export slicing_ws=30
export history_ws=5
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_noptenc_lesstr_/model000390000.pt'
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstr.npy'
export sub_task_cond_type='full'
# use_vision # 


export slicing_ws=30
export history_ws=1
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_1_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_noptenc_lesstr_/model000890000.pt'
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstr.npy'
export sub_task_cond_type='full'


### wp history ###
export slicing_ws=30
export history_ws=5
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_wohistory_v4_w_tacograbinterp_newenc_noptenc_lesstrwohist_/model000410000.pt'
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstr.npy'
export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstrwohist.npy'

export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_evalv2_newenc_noptenc_lesstrwohist.npy'
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_wohistory_v4_w_tacograbinterp_evalv2_newenc_noptenc_lesstrwohist_/model000990000.pt'
export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_wohistory_v4_w_tacograbinterp_evalv2_newenc_noptenc_lesstrwohist_/model001200000.pt'
export sub_task_cond_type='full_wohistory'

# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc_lesstrwohist_v2.npy'
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_wohistory_v4_w_tacograbinterp_newenc_noptenc_lesstrwohist_v2_/model001500000.pt'




# too in-domain #
# another thing is training with different history windows #


# export history_ws=30
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc.npy'
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_30_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_noptenc_/model002760000.pt'


# export history_ws=15
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc.npy'
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_15_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_noptenc_/model002750000.pt'


# export history_ws=5
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc_noptenc.npy'
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_5_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_noptenc_/model002730000.pt'



# ###### slicing ws = 150 ######
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_150_hist_30_step_150_taskcond_history_future_subtype_full_wohistory_v4_w_tacograbinterp_/model000370000.pt'
# export slicing_ws=150
# export history_ws=30
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_150_step_1_w_tacograbinterp.npy'
# export sub_task_cond_type='full_wohistory'
# ###### slicing ws = 150 ######


# ###### the hand shacks sverely ######
# ###### slicing ws = 100 ######
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_100_hist_30_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_/model000770000.pt'
# export slicing_ws=100
# export history_ws=30
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_100_step_1_w_tacograbinterp_newenc.npy'
# export sub_task_cond_type='full'
# ###### slicing ws = 100 ######



# ###### slicing ws = 30 ######
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_hist_30_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_/model002660000.pt'
# export slicing_ws=30
# export history_ws=30
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_1_w_tacograbinterp_newenc.npy'
# export sub_task_cond_type='full'
# ###### slicing ws = 30 ######


# ###### slicing ws = 10 ######
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_10_hist_5_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_/model002550000.pt'
# export slicing_ws=10
# export history_ws=5
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_10_step_1_w_tacograbinterp_newenc.npy'
# export sub_task_cond_type='full'
# ###### slicing ws = 10 ######



## reason? ##
# ###### slicing ws = 5 ######
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_hist_1_step_1_taskcond_history_future_subtype_full_v4_w_tacograbinterp_newenc_/model003250000.pt'
# export slicing_ws=5
# export history_ws=1
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_5_step_1_w_tacograbinterp_newenc.npy'
# export sub_task_cond_type='full'
# ###### slicing ws = 5 ######


export predict_ws=1


# export sub_task_cond_type='obj_shape_pose'
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_5_step_1_w_tacograbinterp.npy'
# export resume_checkpoint_pc='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_histws_5_step_1_taskcond_history_future_subtype_obj_shape_pose_v4_w_tacograbinterp_/model000540000.pt'

export use_deterministic=False

####### NOTE: deterministic mapping model #######
# export slicing_ws=1
# export use_deterministic=True
# export statistics_info_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_1_step_1.npy'
# export resume_checkpoint_pc='' ## TODO
# export history_ws=30
####### NOTE: deterministic mapping model #######


export numEnvs=2
export minibatch_size=2



##### test the pre_optimized_traj #####
# export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/samples_ep_0_batch_0_merged.npy'
# export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/samples_ep_0_batch_0_merged.npy'
# export kinematics_only=True
##### test the pre_optimized_traj #####





export test=True
# ####### test setting ########



export train_name=tracking_${object_name}_obs_${obs_type}_density_${rigid_obj_density}_trans_${glb_trans_vel_scale}_rot_${glb_rot_vel_scale}_goalcond_${goal_cond}_${additiona_tag}
export full_experiment_name=${train_name}


# export test=False
export checkpoint=''


export tag=tracking_${object_name}
export cuda_idx=3

export task_type=AllegroHandTrackingDiff
export train_type=HumanoidPPO


######################### =================================== #########################

# bash scripts/run_tracking_headless_grab_wdiffusion.sh


CUDA_VISIBLE_DEVICES=${cuda_idx} python train.py task=${task_type} train=${train_type} sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs=${numEnvs} train.params.config.minibatch_size=${minibatch_size}  task.env.useRelativeControl=${use_relative_control}  train.params.config.max_epochs=10000 task.env.mocap_sv_info_fn=${mocap_sv_info_fn} checkpoint=${checkpoint} task.env.goal_cond=${goal_cond} task.env.object_name=${object_name} tag=${tag} train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name} task.sim.dt=${dt} test=${test} task.env.use_kinematics_bias=${use_kinematics_bias} task.env.w_obj_ornt=${w_obj_ornt} task.env.observationType=${obs_type} task.env.use_canonical_state=${use_canonical_state} task.env.separate_stages=${separate_stages} task.env.rigid_obj_density=${rigid_obj_density} task.env.use_unified_canonical_state=${use_unified_canonical_state} task.env.kinematics_only=${kinematics_only}  task.env.use_fingertips=${use_fingertips}  task.env.glb_trans_vel_scale=${glb_trans_vel_scale} task.env.glb_rot_vel_scale=${glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta=${use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} task.env.dofSpeedScale=${dofSpeedScale} task.env.pre_optimized_traj=${pre_optimized_traj} task.diffusion.resume_checkpoint_pc=${resume_checkpoint_pc} task.diffusion.statistics_info_fn=${statistics_info_fn} task.diffusion.slicing_ws=${slicing_ws}  task.diffusion.history_ws=${history_ws} task.diffusion.use_deterministic=${use_deterministic} task.diffusion.predict_ws=${predict_ws} task.diffusion.sub_task_cond_type=${sub_task_cond_type}

# task.diffusion.grab_inst_tag_to_opt_stat_fn=${grab_inst_tag_to_opt_stat_fn} task.diffusion.grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn}

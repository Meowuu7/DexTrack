

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
export rew_delta_hand_pose_coef=0.0

### Train_lift_1 #####
# export obs_type='pure_state_wref' # pure state w_ref #
# # export additiona_tag="kinebias_"
# export glb_trans_vel_scale=10
# export glb_rot_vel_scale=1
# export goal_cond=True
### Train_lift_1 #####

export additional_tag=kinebias_wdelta_rewfingerdist_${rew_finger_obj_dist_coef}_rewdeltahandpose_${rew_delta_hand_pose_coef}



export use_kinematics_bias_wdelta=False
export obs_type="pure_state_wref"
export additional_tag=kinebias_rewfingerdist_${rew_finger_obj_dist_coef}_rewdeltahandpose_${rew_delta_hand_pose_coef}

export glb_trans_vel_scale=1
export glb_rot_vel_scale=1
export additional_tag=kinebias_t${glb_trans_vel_scale}r${glb_rot_vel_scale}_rewfingerdist_${rew_finger_obj_dist_coef}_rewdeltahandpose_${rew_delta_hand_pose_coef}



export use_kinematics_bias_wdelta=True
export obs_type="pure_state_wref_wdelta"
export additional_tag=kinebias_rewfingerdist_${rew_finger_obj_dist_coef}_rewdeltahandpose_${rew_delta_hand_pose_coef}

export glb_trans_vel_scale=0.5
export glb_rot_vel_scale=0.5
export dofSpeedScale=20
export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.5

# export use_kinematics_bias_wdelta=False
# export obs_type="pure_state_wref"
# # export additional_tag=kinebias_rewfingerdist_${rew_finger_obj_dist_coef}_rewdeltahandpose_${rew_delta_hand_pose_coef}

# export glb_trans_vel_scale=10
# export glb_rot_vel_scale=10
# export dofSpeedScale=40

export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=0.5
export separate_stages=False



export st_idx=8
export additional_tag=kinebias_t${glb_trans_vel_scale}r${glb_rot_vel_scale}f${dofSpeedScale}_rfd_${rew_finger_obj_dist_coef}_rh_${rew_delta_hand_pose_coef}


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
# # # # export checkpoint='runs/tracking_ori_grab_s2_train_lift_obs_pure_state_density_500_kinebias_09-01-21-00/nn/tracking_ori_grab_s2_train_lift_obs_pure_state_density_500_kinebias.pth'
# export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_hand_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.1_rot_0.1_goalcond_False_kinebais_wdelta_rewhandpos_dist__09-17-22-16/nn/tracking_ori_grab_s2_hand_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.1_rot_0.1_goalcond_False_kinebais_wdelta_rewhandpos_dist_.pth'
# export test=True # runs / track the ori #
# ####### test setting ########

export nn_gpus=8
# export nn_gpus=1
# export nn_gpus=4


export train_name=tracking_${object_name}_obs_${obs_type}_density_${rigid_obj_density}_trans_${glb_trans_vel_scale}_rot_${glb_rot_vel_scale}_goalcond_${goal_cond}_${additiona_tag}
export full_experiment_name=${train_name}

export debug=""
# export debug="--debug"

export tag=tracking_${object_name}
export cuda_idx=2



export tracking_data_sv_root="/cephfs/xueyi/data/GRAB_Tracking_PK/data"
export num_frames=150
# export num_frames=300

##### #####
export data_inst_flag=''
export data_inst_flag='ori_grab_s8_apple_lift'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'

export data_inst_flag='ori_grab_s8_banana_eat_1'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'

export data_inst_flag='ori_grab_s8_banana_peel_1'
# export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'


export data_inst_flag='ori_grab_s8_apple_lift'
# export data_inst_flag='ori_grab_s8_apple_eat_1'
# export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__23-18-20-08/ts_to_hand_obj_obs_reset_1.npy'
export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_eat_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__23-20-35-13/ts_to_hand_obj_obs_reset_1.npy'


export data_inst_flag='ori_grab_s8_toothpaste_pass_1'
export pre_optimized_traj='runs/tracking_ori_grab_s8_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-14-23/ts_to_hand_obj_obs_reset_1.npy'

export data_inst_flag='ori_grab_s8_mouse_lift'
export pre_optimized_traj='runs/tracking_ori_grab_s8_mouse_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-13-31/ts_to_hand_obj_obs_reset_2.npy'


export data_inst_flag='ori_grab_s8_hand_lift'
export pre_optimized_traj='runs/tracking_ori_grab_s8_hand_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-12-00/ts_to_hand_obj_obs_reset_2.npy'


export data_inst_flag='ori_grab_s8_duck_inspect_1'
export pre_optimized_traj='runs/tracking_ori_grab_s8_duck_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-11-09/ts_to_hand_obj_obs_reset_2.npy'


export data_inst_flag='ori_grab_s8_apple_lift'
export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-09-14/ts_to_hand_obj_obs_reset_2.npy'


export data_inst_flag='ori_grab_s9_toothpaste_pass_1'
export pre_optimized_traj='runs/tracking_ori_grab_s8_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-19-58-03/ts_to_hand_obj_obs_reset_1.npy'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s2_toothpaste_squeeze_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-00-30-56/ts_to_hand_obj_obs_reset_1.npy'


export data_inst_flag='ori_grab_s10_hand_inspect_1'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s7_hand_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-14-09-43/ts_to_hand_obj_obs_reset_1.npy'


export data_inst_flag='ori_grab_s8_hammer_lift'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/samples_ep_0_batch_0_merged.npy'


export data_inst_flag='ori_grab_s2_toothpaste_lift'
export pre_optimized_traj='runs/tracking_ori_grab_s2_toothpaste_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-23-23-23/ts_to_hand_obj_obs_reset_1.npy'

export data_inst_flag='ori_grab_s10_hand_inspect_1'
export pre_optimized_traj='runs/tracking_ori_grab_s2_hand_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-01-44-54/ts_to_hand_obj_obs_reset_1.npy'

export data_inst_flag='ori_grab_s10_hammer_use_2'
export pre_optimized_traj='runs/tracking_ori_grab_s10_hammer_use_2_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-01-53-17/ts_to_hand_obj_obs_reset_1.npy'

export data_inst_flag='ori_grab_s2_hand_inspect_1'
export pre_optimized_traj='runs/tracking_ori_grab_s2_hand_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-16-25-34/ts_to_hand_obj_obs_reset_1.npy'

export data_inst_flag='ori_grab_s2_apple_lift'
export pre_optimized_traj='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-15-50-22/ts_to_hand_obj_obs_reset_1.npy'

export data_inst_flag='ori_grab_s8_apple_lift'
export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-18-33-30/ts_to_hand_obj_obs_reset_1.npy'


export data_inst_flag='ori_grab_s8_duck_inspect_1'
export pre_optimized_traj='runs/tracking_ori_grab_s8_duck_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-19-00-27/ts_to_hand_obj_obs_reset_1.npy'


export data_inst_flag='ori_grab_s2_apple_lift' 
export pre_optimized_traj='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__28-10-05-24/ts_to_hand_obj_obs_reset_1.npy'
export pre_optimized_traj='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__28-09-21-17/ts_to_hand_obj_obs_reset_1.npy'
export pre_optimized_traj='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__28-19-23-38/ts_to_hand_obj_obs_reset_1.npy'

export hand_type='allegro'



# 
#### hand type ####

####### experiments for leap ########
# export hand_type='leap'
# export pre_optimized_traj=''
# export data_inst_flag='ori_grab_s2_apple_lift'
# export data_inst_flag='ori_grab_s2_apple_pass_1'
# export tracking_data_sv_root="/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data"
####### experiments for leap ########

export subj_nm='s9'

export test=False

export use_twostage_rew=False
# export st_idx=2
export st_idx=6


#### whether to sue the generalist policy ####
export use_generalist_policy=False
export use_generalist_policy=True
#### whether to use the hand actions reward ####
export use_hand_actions_rew=False

export supervised_training=True

##### not ues the supervised training but with the hand actions rewards #######
# export supervised_training=False
# export use_hand_actions_rew=True
# export use_hand_actions_rew=False
# ##### not ues the supervised training but with the hand actions rewards #######


########### Default geneal settings ###########
export preload_experiences_tf=False
export test_inst_tag=''
export test_optimized_res=''
export single_instance_state_based_test=False
export preload_experiences_path=''
export single_instance_training_config=False
export generalist_tune_all_instnaces=False
export obj_type_to_pre_optimized_traj=""
export pre_load_trajectories=False
export sampleds_with_object_code_fn=""
export pure_supervised_training=False
export log_path='./runs'
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data'
export tracking_info_st_tag='passive_active_info_'
export only_training_on_succ_samples=False
export maxx_inst_nn=10000
export rew_filter=False
export rew_low_threshold=0.0
export use_strict_maxx_nn_ts=False
export grab_train_test_setting=False
export use_strict_maxx_nn_ts=False
export strict_maxx_nn_ts=150
export bound_loss_coef=0.0
export rew_grab_thres=100.0
export rew_taco_thres=200.0
export rew_smoothness_coef=0.0
export use_base_traj=False
export rew_thres_with_selected_insts=False
export selected_inst_idxes_dict=''
export customize_damping=False
export customize_global_damping=False
export train_on_all_trajs=False
export eval_split_trajs=False
export single_instance_state_based_train=False
########### Default geneal settings ###########


# export use_twostage_rew=True
export debug="--debug"
# export debug=""
# export st_idx=0

export episodeLength=1000
export max_epochs=1000


export grab_obj_type_to_opt_res_fn=''
export taco_obj_type_to_opt_res_fn=''
export obj_type_to_base_traj_fn=''


## grab_inst_tag_to_optimized_res_fn, taco_inst_tag_to_optimized_res_fn ##
export grab_inst_tag_to_optimized_res_fn="/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn=""

##### Generalist regular training settings --- settings for training the generalist ######
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-19-38-44/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-19-38-44/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-21-05-45/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-20-34-27/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # superivsed training weights #
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-22-09-35/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-03-11-08/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'   # supervised training weights; hand training only # 
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-03-33-32/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # supervised training weights; hand and obj training together #
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-13-30-52/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # supervised training weights; hand only 
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-13-29-28/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # supervised training weights; hand & obj; 200 instances #
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-17-16-22/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-18-36-16/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-22-12-14/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # the 1000 supervised learning results #
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-02-23-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-12-34-06/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-21-33-40/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-21-53-00/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-22-35-57/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint=''
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-22-43-19/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-22-57-44/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export max_epochs=5000 # 
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-23-22-45/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint=''
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-01-40-22/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-00-42-45/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-10-43-11/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint=''
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-14-42-40/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-15-30-13/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
##### Generalist regular training settings --- settings for training the generalist ######



##### Generalist regular training settings --- training mode ######
export training_mode='regular'
# export checkpoint=''
# export training_mode='offline_supervised'
export checkpoint=''
##### Generalist regular training settings --- training mode ######


export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy"


##### Generalist regular training settings --- training mode; v2 dataset ######
export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn=""
# export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
# export taco_inst_tag_to_optimized_res_fn=""
export single_instance_training=False
export generalist_tune_all_instnaces=False
export supervised_loss_coef=0.0005
# export supervised_loss_coef=0.001
# export pure_supervised_training=True
# export supervised_loss_coef=1.0
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/inst_tag_to_obj_feat.npy'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs_v3/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-03-29-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
##### Generalist regular training settings --- training mode; v2 dataset ######


##### Generalist regular training settings --- training mode; v2 dataset; with grab-train-test setting ######
export grab_train_test_setting=True
export checkpoint=''
##### Generalist regular training settings --- training mode; v2 dataset; with grab-train-test setting ######

# single instance training features #

##### latent feature #####
export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
export checkpoint=''
##### latent feature #####


##### Generalist regular training settings --- GRAB instances with 300 frames ######
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
# export maxx_inst_nn=5
# export num_frames=300 # 300 frames #
# export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy"
# export taco_inst_tag_to_optimized_res_fn=""
# export checkpoint=''
##### Generalist regular training settings --- GRAB instances with 300 frames ######



##### only trained on succ trajectories setting #######
export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
export taco_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res_beta.npy'
export checkpoint=''
# ##### only trained on succ trajectories setting (loss masks) #######


# 

##### only trained on succ trajectories setting -- only such samples #######
export only_training_on_succ_samples=True
##### only trained on succ trajectories setting -- only such samples #######




##### LEAP hand setting ######
# export numEnvs=5000
# export minibatch_size=5000
# export maxx_inst_nn=2
# export maxx_inst_nn=0
# export grab_obj_type_to_opt_res_fn=''
# export taco_obj_type_to_opt_res_fn=''
# export hand_type='leap'
# export checkpoint=''
# # export supervised_training=False
# export supervised_loss_coef=0.0000
# # tracking_save_info_fn, tracking_info_st_tag
# # /cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data/leap_passive_active_info_ori_grab_s1_alarmclock_pass_1.npy
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
# export tracking_info_st_tag='leap_passive_active_info_'
##### LEAP hand setting ######


##### only trained on TACO settings #####
# export maxx_inst_nn=0
# export maxx_inst_nn=2
# export maxx_inst_nn=200
# export supervised_loss_coef=0.0000
# export checkpoint=''
# # export grab_obj_type_to_opt_res_fn=''
# # export taco_obj_type_to_opt_res_fn=''
# export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
# export taco_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res_beta.npy'
# export grab_inst_tag_to_optimized_res_fn=""
# export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy" # taoc inst tag to optimized res fn #
# ## object features and the taco features ##
# export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
# export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
# # export inst_tag_to_latent_feature_fn=''
# export supervised_loss_coef=0.001
# export use_strict_maxx_nn_ts=True
# export use_strict_maxx_nn_ts=True
# # export use_strict_maxx_nn_ts=False
# export strict_maxx_nn_ts=200
# ##### only trained on TACO settings #####



##### only trained on TACO settings; interpfr_60_interpfr2_60_nntrans_40 #####
export dataset_type='taco'
export maxx_inst_nn=0
export maxx_inst_nn=2
export maxx_inst_nn=10000
export supervised_loss_coef=0.0000
export checkpoint=''
export grab_obj_type_to_opt_res_fn=""
# export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
export taco_obj_type_to_opt_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta.npy"
export grab_inst_tag_to_optimized_res_fn=""
# export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy'
export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
export supervised_loss_coef=0.001
export use_strict_maxx_nn_ts=True
# export use_strict_maxx_nn_ts=False
# export strict_maxx_nn_ts=200
# export strict_maxx_nn_ts=150
export strict_maxx_nn_ts=185
# /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230919_053_v2_interpfr_60_interpfr2_60_nntrans_40.npy
# export taco_interped_data_sv_additional_tag=''
export taco_interped_data_sv_additional_tag='interpfr_60_interpfr2_60_nntrans_40'
### on 123 machie ###
export checkpoint='./runs_taco_grab_trajs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-17-48-25/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export use_local_canonical_state=True
export checkpoint=''
export use_local_canonical_state=False
export checkpoint=''
export use_local_canonical_state=True
export checkpoint=''
export supervised_loss_coef=0.000
# export supervised_loss_coef=0.0005
# export supervised_loss_coef=0.0001
# export bound_loss_coef=0.01
export supervised_loss_coef=0.000
export rew_taco_thres=100.0
export rew_taco_thres=50.0
export obj_type_to_base_traj_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res_rew_95.0.npy'
# export obj_type_to_base_traj_fn=''
export rew_smoothness_coef=0.001
export rew_smoothness_coef=0.000
# export rew_taco_thres=200.0
# export rew_taco_thres=100.0
# export strict_maxx_nn_ts=400

export rew_taco_thres=200.0

export rew_taco_thres=50.0
# export rew_smoothness_coef=0.001
export rew_smoothness_coef=0.000
export customize_damping=True
export supervised_loss_coef=0.0005
export customize_global_damping=True
##### only trained on TACO settings; interpfr_60_interpfr2_60_nntrans_40 #####



##### only trained on TACO settings; interpfr_60_interpfr2_60_nntrans_40; all trajs #####
export strict_maxx_nn_ts=185
export obj_type_to_base_traj_fn=''
export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_tot.npy"
export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_tot_filtered.npy"
export train_on_all_trajs=True
export taco_obj_type_to_opt_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy"
export only_training_on_succ_samples=False
export customize_global_damping=False




# export maxx_inst_nn=200
# export num_frames=150
# export strict_maxx_nn_ts=185
# # export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
# export numEnvs=5000
# export minibatch_size=5000
# export hand_type='leap'
# export supervised_loss_coef=0.000
# export tracking_info_st_tag='leap_passive_active_info_'
# export tracking_save_info_fn='/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data'
# export tracking_data_sv_root='/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data'
##### only trained on TACO settings; interpfr_60_interpfr2_60_nntrans_40; all trajs #####




# #### only trained on GRAB settings #####
# export maxx_inst_nn=0
# export maxx_inst_nn=2
# export maxx_inst_nn=10000
# export supervised_loss_coef=0.0000
# export checkpoint=''
# # export grab_obj_type_to_opt_res_fn=""
# export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
# # export taco_obj_type_to_opt_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta.npy"
# export taco_obj_type_to_opt_res_fn=""
# # export grab_inst_tag_to_optimized_res_fn=""
# export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
# # export taco_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy'
# export taco_inst_tag_to_optimized_res_fn=""
# export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
# export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
# export supervised_loss_coef=0.001
# export use_strict_maxx_nn_ts=True
# # export use_strict_maxx_nn_ts=False
# # export strict_maxx_nn_ts=200
# # export strict_maxx_nn_ts=150
# export strict_maxx_nn_ts=185
# # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230919_053_v2_interpfr_60_interpfr2_60_nntrans_40.npy
# # export taco_interped_data_sv_additional_tag=''
# export taco_interped_data_sv_additional_tag='interpfr_60_interpfr2_60_nntrans_40'
# ### on 123 machie ###
# # export checkpoint='./runs_taco_grab_trajs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-17-48-25/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# # export use_local_canonical_state=True
# # export checkpoint=''
# # # export grab_obj_type_to_opt_res_fn=""
# # export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
# # # export taco_obj_type_to_opt_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta.npy"
# # export taco_obj_type_to_opt_res_fn=""
# # # export grab_inst_tag_to_optimized_res_fn=""
# # export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
# # # export taco_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy'
# # export taco_inst_tag_to_optimized_res_fn=""
# # export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
# # export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
# # export supervised_loss_coef=0.001
# # export use_strict_maxx_nn_ts=True
# # # export use_strict_maxx_nn_ts=False
# # # export strict_maxx_nn_ts=200
# # # export strict_maxx_nn_ts=150
# # export strict_maxx_nn_ts=185
# # # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230919_053_v2_interpfr_60_interpfr2_60_nntrans_40.npy
# # # export taco_interped_data_sv_additional_tag=''
# # export taco_interped_data_sv_additional_tag='interpfr_60_interpfr2_60_nntrans_40'
# # ### on 123 machie ###
# # # export checkpoint='./runs_taco_grab_trajs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-17-48-25/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export use_local_canonical_state=True
# export checkpoint=''
# export supervised_loss_coef=0.000
# export supervised_loss_coef=0.0005
# # export bound_loss_coef=0.01
# export rew_grab_thres=100.0
# export rew_taco_thres=200.0
# export rew_grab_thres=50.0
# ##### only trained on succ trajectories setting -- only such samples #######
# export only_training_on_succ_samples=False
# ##### only trained on succ trajectories setting -- only such samples #######
# export rew_smoothness_coef=0.0001

# # # rew_thres_with_selected_insts, selected_inst_idxes_dict # 
# export rew_grab_thres=100.0
# export rew_thres_with_selected_insts=True
# export rew_smoothness_coef=0.001
# export selected_inst_idxes_dict='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/selected_obj_types_idxes.npy'





export use_window_future_selection=False
export forcasting_history_ws=60
export forcasting_inv_freq=60


##### tune all instances settings; trajectory initialization; TACO #####
export checkpoint=''
export grab_obj_type_to_opt_res_fn=''
export only_training_on_succ_samples=False
export generalist_tune_all_instnaces=True
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v4/statistics/data_inst_tag_to_optimized_res.npy'
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res.npy'
export sampleds_with_object_code_fn=''
export debug=""
export pre_load_trajectories=True
export log_path='./runs_init_fr_traj_translations'
export obj_type_to_optimized_res_fn=''
export rew_filter=True
export rew_low_threshold=10.0
export rew_filter=False
export rew_low_threshold=10.0
export dataset_type='taco'
export use_base_traj=True
export rew_smoothness_coef=0.001
export early_terminate=False 
##### tune all instances settings; trajectory initialization; TACO #####


# #### tune all instances settings; no initialization; simple training #####
# export numEnvs=4000
# export minibatch_size=4000
# export num_frames=150
# export maxx_inst_nn=1000
# export checkpoint=''
# export generalist_tune_all_instnaces=True
# # export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
# export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
# # export sampleds_with_object_code_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_canonv2_taskcond_samples_/samples_ep_0_batch_0.npy'
# export sampleds_with_object_code_fn=''
# export subj_nm='s1'
# export debug="--debug" # debug #
# export debug=""
# export pre_load_trajectories=True
# # export log_path='./runs_init_fr_traj_translations'
# export log_path='./runs_leap_hand_traj'
# export grab_obj_type_to_opt_res_fn=''
# export taco_obj_type_to_opt_res_fn=''
# export checkpoint=''
# export only_training_on_succ_samples=False 
# #### tune all instances settings; no initialization; simple training #####



##### tune all instances settings; no initialization; simple training; with customized damping; GRAB-300 #####
export dataset_type='grab'
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
export numEnvs=8000
export minibatch_size=8000
export num_frames=300
export maxx_inst_nn=1000
export checkpoint=''
export generalist_tune_all_instnaces=True
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
export sampleds_with_object_code_fn=''
export subj_nm='s1'
export debug="--debug"
export debug=""
export pre_load_trajectories=True
# export log_path='./runs_init_fr_traj_translations'
# export log_path='./runs_leap_hand_traj'
export grab_obj_type_to_opt_res_fn=''
export taco_obj_type_to_opt_res_fn=''
export checkpoint=''
export only_training_on_succ_samples=False 
export customize_damping=True
export rew_smoothness_coef=0.000


export num_frames=150
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
export numEnvs=5000
export minibatch_size=5000
export hand_type='leap'
export tracking_info_st_tag='leap_passive_active_info_'
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
##### tune all instances settings; no initialization; simple training; with customized damping; GRAB-300 #####

##### some default settings #####
export downsample=False
export use_base_traj=False
export use_teacher_model=False
export teacher_model_path=''
export teacher_model_inst_tags_fn=''
export use_history_obs=False
export history_length=5
export w_franka=False
export good_inst_opt_res=''
export early_terminate=False
export use_forcasting_model=False
export forcasting_model_weights=''
export forcasting_model_n_layers=7
export w_glb_traj_feat_cond=False
export forcasting_inv_freq=1
export forcasting_history_ws=30
export sv_info_during_training=False
export w_impedance_bias_control=False
export impedance_stiffness_low=1.0
export impedance_stiffness_high=50.0
export w_obj_latent_features=True
export w_inst_latent_features=False
export net_type='v4'
export history_freq=1
export use_future_obs=False
export randomize_conditions=False 
export w_history_window_index=True
export masked_mimic_training=False
export masked_mimic_teacher_model_path=''
export forcasting_model_training=False
export forcasting_model_lr=0.0001
export forcasting_model_weight_decay=0.00005
export contact_info_sv_root="/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag"
export add_contact_conditions=False
export st_ed_state_cond=False
export forcasting_diffusion_model=False
# random_shift_cond, random_shift_cond_freq, maxx_inv_cond_freq
export random_shift_cond=False
export random_shift_cond_freq=False
export maxx_inv_cond_freq=30
export only_use_hand_first_frame=False
export teacher_index_to_weights=''
export teacher_index_to_inst_tags=''
##### some default settings #####


##### tune all instances settings; no initialization; simple training; with customized damping; GRAB-300 #####
export dataset_type='grab'
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
export numEnvs=8000
export minibatch_size=8000
export num_frames=300
export maxx_inst_nn=1000
export checkpoint=''
export generalist_tune_all_instnaces=False
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
export sampleds_with_object_code_fn=''
export subj_nm='s1'
export debug="--debug"
export debug=""
export pre_load_trajectories=True
# export log_path='./runs_init_fr_traj_translations'
# export log_path='./runs_leap_hand_traj'
export grab_obj_type_to_opt_res_fn=''
export taco_obj_type_to_opt_res_fn=''
export checkpoint=''
export only_training_on_succ_samples=False 
export customize_damping=True
export customize_global_damping=False
export rew_smoothness_coef=0.000


export strict_maxx_nn_ts=300
# export strict_maxx_nn_ts=150
export hand_type='allegro'
export supervised_loss_coef=0.000
export grab_train_test_setting=True
export grab_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_300_v12/statistics/data_inst_tag_to_optimized_res.npy'
export taco_inst_tag_to_optimized_res_fn=''
export obj_type_to_base_traj_fn=''
export data_inst_flag='ori_grab_s2_apple_lift_nf_300' 
export tracking_info_st_tag='passive_active_info_'
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data'



export grab_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
export taco_inst_tag_to_optimized_res_fn=''
# export only_training_on_succ_samples
export supervised_loss_coef=0.0005
export supervised_loss_coef=0.0000

export controlFrequencyInv=3


export controlFrequencyInv=1 # control freq inv #
export supervised_loss_coef=0.0005
export supervised_loss_coef=0.0000
# export supervised_loss_coef=0.001





###### Downsample settings #######
# export dt=0.00833  
# export controlFrequencyInv=2

# export dt=0.0166
# export controlFrequencyInv=1
# export controlFrequencyInv=2
# export supervised_loss_coef=0.0
# # export supervised_loss_coef=0.001
# # export supervised_loss_coef=0.005
# # export maxx_inst_nn=10
# # export strict_maxx_nn_ts=151
# export downsample=True
# export target_inst_tag_list_fn=''
###### Downsample settings #######


###### Teacher model training setting ######
export teacher_subj_idx=9
# export teacher_subj_idx=9
export supervised_loss_coef=0.0
export use_base_traj=False
export dt=0.0166
export controlFrequencyInv=1
export downsample=False
export target_inst_tag_list_fn=/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_s${teacher_subj_idx}.npy
# export target_inst_tag_list_fn=''
###### Teacher model training setting ######




# ###### Teacher-student model setting #######
# export use_teacher_model=True
# export teacher_model_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_s10/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-04-17-19/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export teacher_model_inst_tags_fn='/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_s9.npy'
# export supervised_loss_coef=0.0000
# export supervised_loss_coef=0.0005
# export supervised_loss_coef=0.1
# # export teacher_index_to_weights='/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights_v2.npy'
# # export teacher_index_to_inst_tags='/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_inst_tags_v2.npy'
# export teacher_index_to_weights='/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights_v3.npy'
# export teacher_index_to_inst_tags='/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_inst_tags_v3.npy'
# export teacher_index_to_weights='/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights_v4.npy'
# export teacher_index_to_inst_tags='/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_inst_tags_v4.npy'
# # export good_inst_opt_res="/root/diffsim/IsaacGymEnvs2/assets/good_inst_opt_res.npy"
# export good_inst_opt_res=''
# # export maxx_inst_nn=10
# # export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s10/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-17-24-40/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint=''
# export supervised_loss_coef=0.0000
# export supervised_loss_coef=0.0001
# export supervised_loss_coef=0.0005
# # export supervised_loss_coef=0.0000
# # export inst_tag_to_latent_feature_fn=''
# ###### Teacher-student model setting #######



###### With franka setting #######
# export w_franka=True
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka_v2/data'
# export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka_v2/data'
# ###### With franka setting #######


###### Use history observation setting ######
# export use_history_obs=True
# export history_length=5
# # export maxx_inst_nn=10
###### Use history observation setting ######



###### Early terminate config ######
# export early_terminate=True
###### Early terminate config ######


###### Sim parameters #######
export dt=0.05
export substeps=12
###### Sim parameters #######


###### Sim parameters #######
export dt=0.0166
export substeps=4
export substeps=2
###### Sim parameters #######


###### Resume setting #######
# export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_sup0d0001_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-00-30-56/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint=''
###### Resume setting #######


###### Whether to use the instnace latent features #######
# # export inst_tag_to_latent_feature_fn=''
# export use_teacher_model=False

# # export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy"  
# # export inst_tag_to_latent_feature_fn=''

# # export w_obj_latent_features=False
# export net_type='v4'
# # export net_type='v3'
# # export net_type='v1'
###### Whether to use the instnace latent features #######


###### whether to use history #######
export use_history_obs=True
export history_length=5
export history_freq=1
export history_freq=5
export history_freq=8
export history_freq=10
###### whether to use history #######



###### whether to use the future obs #######
export use_future_obs=True
###### whether to use the future obs #######


###### Per-subject training settings ######
# export target_inst_tag_list_fn="/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_s2.npy"
# export target_inst_tag_list_fn=''
#### Per-subject training settings ######

###### Activate the per-subject training settings ######
# export target_inst_tag_list_fn="/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_s10.npy"
# export target_inst_tag_list_fn="/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_s9.npy"
# export target_inst_tag_list_fn=''
###### Activate the per-subject training settings ######



### history conditions ###

###### History and the future observation settings #######
# export use_future_obs=True
export use_future_obs=False
export use_history_obs=False
export history_freq=5
# export history_freq=1
export history_length=5
# export history_length=4
###### History and the future observation settings #######




# ###### Random shift conditions setting #######
# export random_shift_cond=True
# export random_shift_cond_freq=False
# # export random_shift_cond=False
# # export random_shift_cond_freq=True
# export maxx_inv_cond_freq=30
# ###### Random shift conditions setting #######


###### Whether to only use hand first frame setting ######
# export only_use_hand_first_frame=True
# export single_instance_state_based_train=True
###### Whether to only use hand first frame setting ######


###### Instnace feature fn -- adjusting the observation dimension here #######
# export inst_tag_to_latent_feature_fn=''
###### Instnace feature fn -- adjusting the observation dimension here #######



###### Randomized condition training #######
# export randomize_conditions=True
# export randomize_condition_type='random'
###### Randomized condition training #######

###### contact condition setting ########
# export contact_info_sv_root="/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag"
# export add_contact_conditions=True
###### contact condition setting ########


###### Randomized condition training #######
# export inst_tag_to_latent_feature_fn=''
# export supervised_loss_coef=0.0005
# export masked_mimic_training=True
# export masked_mimic_teacher_model_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-17-15-37/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_
###### Randomized condition training #######


####### Use the leap model #########
# export hand_type='leap' 
# # when the trajectory length goes up; the model often has difficulties in tracking them #
# export tracking_info_st_tag='leap_passive_active_info_'
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
# export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'

# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data'
# export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data'

# # export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced/data'
# # export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced/data'

# # export maxx_inst_nn=10
####### Use the leap model #########




####### Forcasting model setting #########
# export use_forcasting_model=True
# export forcasting_model_n_layers=7
# # export w_glb_traj_feat_cond=False
# export w_glb_traj_feat_cond=True
# export use_history_obs=True
# export use_history_obs=False
# export maxx_inst_nn=10
# export maxx_inst_nn=1
# export use_teacher_model=False
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_/model000240001.pt'
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_/model000670001.pt'
# export single_instance_state_based_train=True
# export w_history_window_index=True

# export forcasting_history_ws=60
# export forcasting_inv_freq=60

# # export forcasting_history_ws=100
# # export forcasting_inv_freq=100

# # export forcasting_history_ws=120
# # export forcasting_inv_freq=120

# # export forcasting_history_ws=150
# # export forcasting_inv_freq=150

# # export forcasting_history_ws=160
# # export forcasting_inv_freq=160

# export sv_info_during_training=True
# export target_inst_tag_list_fn=''
# ### forcasting model training setting ###
# export forcasting_model_training=True
# export forcasting_model_lr=0.0001
# export forcasting_model_weight_decay=0.00005
# ### forcasting model training setting ###


# ### single isntane w/o forcasting model comparison ###
# # export use_forcasting_model=False
# # export sv_info_during_training=False
# # export net_type='v1'
# ### single instance w/o forcasting model comparison ###

# #### forcasting diffusion setting ####
# # export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_histindex_v2/model005520002.pt'
# # export forcasting_diffusion_model=True
# #### forcasting diffusion setting ####
####### Forcasting model setting #########





###### Forcasting diffusion model setting #######
# export forcasting_diffusion_model=True
# export forcasting_model_weights=''
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_stedgoalcond_/model005400002.pt'
# # export forcasting_model_weights="/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep4_wforcasting_model_nhist_nbias_glbtraj_singletraj_sn_toursesmall_widx_forefreq60_inst1_train_diff/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-03-35-23/last_forcasting_model_weight.pth"
###### Forcasting diffusion model setting #######



####### Compliance ########
# export use_forcasting_model=False
# export sv_info_during_training=True
# export impedance_stiffness_low=1.0
# export impedance_stiffness_high=50.0
# export w_impedance_bias_control=True
# export w_impedance_bias_control=False
####### Compliance ########


####### Use the window future selection setting #######
# export use_window_future_selection=True
# # export use_window_future_selection=False
# export use_forcasting_model=False
# export use_history_obs=False
####### Use the window future selection setting #######


########### Use window future ###########
# export num_frames=150 # 300
# export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
# export numEnvs=5000
# export minibatch_size=5000
# export hand_type='leap'
# export tracking_info_st_tag='leap_passive_active_info_'
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
# export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
##### tune all instances settings; no initialization; simple training; with customized damping; GRAB-300 #####


##### params introduced for forecasting model training #####
export comput_reward_traj_hand_qpos=True
export use_future_ref_as_obs_goal=True
# export use_future_ref_as_obs_goal=False
##### params introduced for forecasting model training #####


##### include obj ornt into the obs #####
export w_obj_ornt=True
export include_obj_rot_in_obs=True
export w_obj_ornt=False
export include_obj_rot_in_obs=False
##### include obj ornt into the obs #####

##### Conditional states setting -- using start and end frame as the conditions ######
# export st_ed_state_cond=True
##### Conditional states setting -- using start and end frame as the conditions ######


#### teacher model training ####
export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s10'
export max_epochs=10000

export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s8'
export max_epochs=10000

export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s9'
export max_epochs=10000

export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s${teacher_subj_idx}_v2

export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s${teacher_subj_idx}_v3goal
export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s${teacher_subj_idx}_v3goal_wornt
export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal
export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal_frteacherv4


export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s${teacher_subj_idx}_v3goal_v2

export max_epochs=10000
#### teacher model training ####



#### teacher-student training ####
# export log_path='./isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s1_supcoef_0.01'
# export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_sup0d001_s2tos8_'
# export max_epochs=100000

# # 
# # export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_sup0d005_s2tos8_'
# # export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_'
# export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s9_wfuturews5freq5_'
# export max_epochs=100000


# # 
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcondv2_'
# export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcondfreq_'
# export max_epochs=100000
# # 
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcondv3condpartialhand_'
# export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_sup0d005_s2tos10_'
# export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_wfuturews5freq5_sup0d005_s2tos10_'
# export max_epochs=100000

# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcond_singleinsttest_'
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcond_singleinsttest_ncondhandonly_'
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcond_singleinsttest_ncondhandonly_handfirstfr_'
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s2_randshiftcond_singleinsttest_ncondhandonly_nhandfirstfr_'
# export max_epochs=100000
#### teacher-student training ####


##### Subject training #####
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s2_'
# export max_epochs=1000000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws4_s2_'
# export max_epochs=100000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_randcond_'
# export max_epochs=100000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_freq5_ws5_s10_'
# export max_epochs=100000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_randcond_distill_'
# export max_epochs=100000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_randcond_distilltest_'
# export max_epochs=100000
# # export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s3_'
# # export max_epochs=100000
# # export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s5_'
# # export max_epochs=100000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s8_'
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s10_'
# # export max_epochs=100000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_contactcond_distll_'
# export max_epochs=100000
##### Subject training #####



########## Forcasting model training ############
# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep4_wforcasting_model_v4forcasting_nhist_nbias_glbtraj_singletraj_sn_toursesmall_forefreq10_inst3'
# export max_epochs=1000000
# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep4_wforcasting_model_v4forcasting_nhist_nbias_glbtraj_singletraj_sn_toursesmall_widx_forefreq60_inst3'
# export max_epochs=10000
# export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep4_wforcasting_model_nhist_nbias_glbtraj_singletraj_sn_toursesmall_widx_forefreq60_inst1_train_diff'
# export max_epochs=10000
########## Forcasting model training ############




########## Impedance control ############
# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_impedance_'
# export max_epochs=100000

# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nimpedance_'
# export max_epochs=100000
########## Impedance control ############





#### teacher-student training ####
# export log_path='./isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_wfranka'
# export max_epochs=10000
#### teacher-student training ####



#### leap hand model training ####
# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_wcustomdamping_newtrainall_v2_'
# export max_epochs=10000

# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_wcustomdamping_newtrainall_v2_s1_'

# export log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_wcustomdamping_newtrainall_v2_s2_'
# export max_epochs=10000
#### leap hand model training ####





export numEnvs=11000
export minibatch_size=11000


export numEnvs=25000
export minibatch_size=25000

# export numEnvs=50000
# export minibatch_size=50000


export target_inst_tag_list_fn=/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_obj_duck.npy
export log_path=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_duck_v3goal_v2
export numEnvs=20000
export minibatch_size=20000


#### bash scripts ####
# export maxx_inst_nn=5
# export maxx_inst_nn=2




export st_idx=0



# bash scripts/run_tracking_headless_grab_whltrain_v1_local.sh




CUDA_VISIBLE_DEVICES=${cuda_idx} python train_pool.py --additional_tag=${additional_tag} --hand_type=${hand_type} --numEnvs=${numEnvs} --minibatch_size=${minibatch_size} --goal_cond=${goal_cond}  --test=${test} --use_relative_control=${use_relative_control} --use_kinematics_bias=${use_kinematics_bias} --w_obj_ornt=${w_obj_ornt} --obs_type=${obs_type} --separate_stages=${separate_stages} --rigid_obj_density=${rigid_obj_density}  --kinematics_only=${kinematics_only} --use_fingertips=${use_fingertips}  --use_kinematics_bias_wdelta=${use_kinematics_bias_wdelta} --hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} --hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} --hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} --dt=${dt} --glb_trans_vel_scale=${glb_trans_vel_scale} --glb_rot_vel_scale=${glb_rot_vel_scale} --rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} --rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} ${debug} --nn_gpus=${nn_gpus} --num_frames=${num_frames} --tracking_data_sv_root=${tracking_data_sv_root} --subj_nm=${subj_nm}  --st_idx=${st_idx} --dofSpeedScale=${dofSpeedScale} --use_twostage_rew=${use_twostage_rew} --episodeLength=${episodeLength} --data_inst_flag=${data_inst_flag} --pre_optimized_traj=${pre_optimized_traj} --use_generalist_policy=${use_generalist_policy} --use_hand_actions_rew=${use_hand_actions_rew} --supervised_training=${supervised_training} --checkpoint=${checkpoint} --max_epochs=${max_epochs} --training_mode=${training_mode} --test_inst_tag=${test_inst_tag} --test_optimized_res=${test_optimized_res} --preload_experiences_tf=${preload_experiences_tf} --preload_experiences_path=${preload_experiences_path} --single_instance_training=${single_instance_training} --generalist_tune_all_instnaces=${generalist_tune_all_instnaces} --obj_type_to_pre_optimized_traj=${obj_type_to_pre_optimized_traj} --pre_load_trajectories=${pre_load_trajectories} --sampleds_with_object_code_fn=${sampleds_with_object_code_fn} --log_path=${log_path} --grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} --taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn} --single_instance_tag=${single_instance_tag} --obj_type_to_optimized_res_fn=${obj_type_to_optimized_res_fn} --supervised_loss_coef=${supervised_loss_coef} --pure_supervised_training=${pure_supervised_training} --inst_tag_to_latent_feature_fn=${inst_tag_to_latent_feature_fn} --object_type_to_latent_feature_fn=${object_type_to_latent_feature_fn} --grab_obj_type_to_opt_res_fn=${grab_obj_type_to_opt_res_fn} --taco_obj_type_to_opt_res_fn=${taco_obj_type_to_opt_res_fn}  --maxx_inst_nn=${maxx_inst_nn} --tracking_save_info_fn=${tracking_save_info_fn} --tracking_info_st_tag=${tracking_info_st_tag} --only_training_on_succ_samples=${only_training_on_succ_samples} --rew_filter=${rew_filter} --rew_low_threshold=${rew_low_threshold} --use_strict_maxx_nn_ts=${use_strict_maxx_nn_ts} --taco_interped_data_sv_additional_tag=${taco_interped_data_sv_additional_tag} --strict_maxx_nn_ts=${strict_maxx_nn_ts} --grab_train_test_setting=${grab_train_test_setting} --checkpoint=${checkpoint} --use_local_canonical_state=${use_local_canonical_state} --bound_loss_coef=${bound_loss_coef} --rew_grab_thres=${rew_grab_thres} --rew_taco_thres=${rew_taco_thres} --dataset_type=${dataset_type} --rew_smoothness_coef=${rew_smoothness_coef} --use_base_traj=${use_base_traj} --obj_type_to_base_traj_fn=${obj_type_to_base_traj_fn} --rew_thres_with_selected_insts=${rew_thres_with_selected_insts} --selected_inst_idxes_dict=${selected_inst_idxes_dict} --customize_damping=${customize_damping} --customize_global_damping=${customize_global_damping} --train_on_all_trajs=${train_on_all_trajs} --eval_split_trajs=${eval_split_trajs} --single_instance_state_based_train=${single_instance_state_based_train} --controlFrequencyInv=${controlFrequencyInv} --downsample=${downsample} --target_inst_tag_list_fn=${target_inst_tag_list_fn} --use_teacher_model=${use_teacher_model} --teacher_model_path=${teacher_model_path} --teacher_model_inst_tags_fn=${teacher_model_inst_tags_fn} --teacher_index_to_weights=${teacher_index_to_weights} --teacher_index_to_inst_tags=${teacher_index_to_inst_tags} --use_history_obs=${use_history_obs} --history_length=${history_length} --good_inst_opt_res=${good_inst_opt_res} --w_franka=${w_franka}  --early_terminate=${early_terminate} --substeps=${substeps} --use_forcasting_model=${use_forcasting_model} --forcasting_model_weights=${forcasting_model_weights} --forcasting_model_n_layers=${forcasting_model_n_layers} --w_glb_traj_feat_cond=${w_glb_traj_feat_cond} --use_window_future_selection=${use_window_future_selection} --forcasting_inv_freq=${forcasting_inv_freq} --forcasting_history_ws=${forcasting_history_ws} --sv_info_during_training=${sv_info_during_training} --impedance_stiffness_low=${impedance_stiffness_low} --impedance_stiffness_high=${impedance_stiffness_high} --w_impedance_bias_control=${w_impedance_bias_control} --w_obj_latent_features=${w_obj_latent_features} --net_type=${net_type} --history_freq=${history_freq} --use_future_obs=${use_future_obs} --w_history_window_index=${w_history_window_index} --randomize_conditions=${randomize_conditions} --w_inst_latent_features=${w_inst_latent_features} --masked_mimic_training=${masked_mimic_training} --masked_mimic_teacher_model_path=${masked_mimic_teacher_model_path} --forcasting_model_training=${forcasting_model_training} --forcasting_model_lr=${forcasting_model_lr} --forcasting_model_weight_decay=${forcasting_model_weight_decay} --randomize_condition_type=${randomize_condition_type} --add_contact_conditions=${add_contact_conditions} --contact_info_sv_root=${contact_info_sv_root} --st_ed_state_cond=${st_ed_state_cond} --forcasting_diffusion_model=${forcasting_diffusion_model} --random_shift_cond=${random_shift_cond} --random_shift_cond_freq=${random_shift_cond_freq} --maxx_inv_cond_freq=${maxx_inv_cond_freq} --only_use_hand_first_frame=${only_use_hand_first_frame} --comput_reward_traj_hand_qpos=${comput_reward_traj_hand_qpos} --use_future_ref_as_obs_goal=${use_future_ref_as_obs_goal} --include_obj_rot_in_obs=${include_obj_rot_in_obs}
 


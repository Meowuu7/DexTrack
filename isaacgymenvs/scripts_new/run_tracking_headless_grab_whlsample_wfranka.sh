

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
export st_idx=0


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
export use_local_canonical_state=False
export obj_type_to_ckpt_fn=''
export use_base_traj=False
export obj_type_to_base_trajs=''
export customize_damping=False
export tracking_info_st_tag='passive_active_info_'
export train_on_all_trajs=False
export test_on_taco_test_set=False
export single_instance_state_based_train=False
export w_obj_latent_features=True
export forcasting_diffusion_model=False
export use_future_obs=False
export randomize_conditions=False
export randomize_condition_type='random'
export history_freq=5
export history_length=5
export partial_obj_info=False
export partial_hand_info=False
export use_partial_to_complete_model=False
export partial_to_complete_model_weights=''
export st_ed_state_cond=False
export add_contact_conditions=False
export wo_vel_obs=False
export disable_hand_obj_contact=False
export closed_loop_to_real=False 
export hand_glb_mult_factor_scaling_coef=1.0
export hand_glb_mult_scaling_progress_after=900
export wo_fingertip_rot_vel=False
export include_obj_rot_in_obs=False
export arm_stiffness=400
export arm_effort=400
export arm_damping=80
export estimate_vels=False
export use_v2_leap_warm_urdf=False

# export use_twostage_rew=True
export debug="--debug"
# export debug=""
# export st_idx=0

export episodeLength=1000
export max_epochs=1000


## grab_inst_tag_to_optimized_res_fn, taco_inst_tag_to_optimized_res_fn ##
export grab_inst_tag_to_optimized_res_fn="/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn=""



##### Generalist regular training settings --- training mode ######
export training_mode='regular'
# export checkpoint=''
# export training_mode='offline_supervised'
export checkpoint=''
##### Generalist regular training settings --- training mode ######




##### Generalist regular training settings --- training mode; v2 dataset ######
export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
# export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
# export taco_inst_tag_to_optimized_res_fn=""
export single_instance_training=False
export generalist_tune_all_instnaces=False
export generalist_tune_all_instnaces=True
export supervised_loss_coef=0.0005
# export supervised_loss_coef=0.001
# export pure_supervised_training=True
# export supervised_loss_coef=1.0
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/inst_tag_to_obj_feat.npy'


### Checkpoint trained on all trajs ###
### Checkpoint trained on only succ trajs ###
# export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_35_runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-22-56-41/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/inst_tag_to_obj_feat.npy'
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
export pre_load_trajectories=True
##### Generalist regular training settings --- training mode; v2 dataset ######

export dataset_type='grab'

export num_frames=150
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data'
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data'

###### GRAB-300 test settting ########
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
# # export maxx_inst_nn=5
# export num_frames=300
# export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy"
# export taco_inst_tag_to_optimized_res_fn=""
# export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
# # export checkpoint=''
###### GRAB-300 test settting ########



export downsample=False
export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy"


###### GRAB test setting #######
export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
export grab_obj_type_to_opt_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
export taco_obj_type_to_opt_res_fn=""
export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn=''
export supervised_loss_coef=0.001
export use_strict_maxx_nn_ts=True
export strict_maxx_nn_ts=185
export use_local_canonical_state=True
# export use_local_canonical_state=False
export taco_interped_data_sv_additional_tag='interpfr_60_interpfr2_60_nntrans_40'
### model trained on GRAB s1-s9 trajectories ###
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_training/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-22-24-59/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_training/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-01-16-11/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_samples_rewthres50/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-14-41-56/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_training/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-22-48-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_samples_rewthres50/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-14-41-56/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_650_rew_-9.833794.pth'
# last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_100_rew_-26.675728.pth
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_samples_rewthres50/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-14-41-56/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1500_rew_-3.7702148.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_samples_rewthres50/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-14-41-56/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2800_rew_-1.8776069.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_samples_rewthres50/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-14-41-56/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_100_rew_-26.675728.pth'
###### GRAB test setting ########


###### GRAB-300 test settting ########
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK/data'
export num_frames=300
export use_strict_maxx_nn_ts=False
export use_strict_maxx_nn_ts=True
export strict_maxx_nn_ts=300
export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy"
export taco_inst_tag_to_optimized_res_fn=""
# obj type to pre optimized traj #
export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
# ######## Per-subject trained models #########
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s2/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-01-16-10/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_s10/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-04-17-19/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_sup0d0001_early_terminate_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-00-31-07/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# ######## Per-subject trained models #########


export use_window_future_selection=False


export grab_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
export taco_inst_tag_to_optimized_res_fn=''
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data'
export taco_inst_tag_to_optimized_res_fn=''
export obj_type_to_base_traj_fn=''
export data_inst_flag='ori_grab_s2_apple_lift_nf_300' 
export tracking_info_st_tag='passive_active_info_'
export hand_type='allegro'

export customize_damping=True
export customize_global_damping=False

export controlFrequencyInv=1


###### Sim parameters #######
export dt=0.0166
export substeps=4

export dt=0.0166
export substeps=2
###### Sim parameters #######



####### Forcasting model setting #########
export use_forcasting_model=True
export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_/model000400001.pt' # 
export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_/model000310001.pt'
export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_/model000580001.pt'
export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_/model000240001.pt'
export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_/model000670001.pt'
export forcasting_model_n_layers=7
export w_glb_traj_feat_cond=False
export w_glb_traj_feat_cond=True
# export use_history_obs=True
export use_history_obs=False
export maxx_inst_nn=10
export use_teacher_model=False
# export use_forcasting_model=False
export w_history_window_index=False
export w_history_window_index=True
export single_instance_test_tag='ori_grab_s6_torusmedium_inspect_1_nf_300'

export forcasting_history_ws=60
export forcasting_inv_freq=60
####### Forcasting model setting #########


####### turn off the forcasting model ########
export use_forcasting_model=False
####### turn off the forcasting model ########


####### not use the obj and inst feat setting ######
export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'
# export object_type_to_latent_feature_fn=''
# export inst_tag_to_latent_feature_fn=''
export w_obj_latent_features=False
####### not use the obj and inst feat setting ######

####### obj latent features ######
export w_obj_latent_features=True
# export inst_tag_to_latent_feature_fn=''
####### obj latent features ######

####### Distable single instance test tag #######
export single_instance_test_tag=''
####### Distable single instance test tag #######


###### Preset cond type ######
export preset_cond_type=0
export preset_inv_cond_freq=1
# export preset_inv_cond_freq=10
###### Preset cond type ######

###### Randomized condition training #######
# export use_history_obs=False
# export use_future_obs=True
# export randomize_conditions=True
# export randomize_condition_type='random'
# export randomize_condition_type='hand'
# # export randomize_condition_type='obj'
# export history_freq=5
# export history_length=5

# ### partial info setting ###
# export partial_obj_info=False
# export partial_hand_info=True
# ### partial info setting ###
###### Randomized condition training #######


###### Randomized condition training #######
# export randomize_conditions=True
# export randomize_condition_type='random'
###### Randomized condition training #######

###### contact condition setting ########
# export contact_info_sv_root="/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag"
# export add_contact_conditions=True
# export use_forcasting_model=False
# export use_future_obs=True
# export use_history_obs=False
# export history_freq=5
# export history_length=5
###### contact condition setting ########




# ###### Forcasting diffusion model setting #######
# export forcasting_diffusion_model=True
# export forcasting_model_weights=''
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_histindex_v2/model005520002.pt'
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2_partialhand_/model004710003.pt'
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2_partialhand_condhandonly_/model004900003.pt'
# export forcasting_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_stedgoalcond_/model005400002.pt'
# # export forcasting_model_weights="/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep4_wforcasting_model_nhist_nbias_glbtraj_singletraj_sn_toursesmall_widx_forefreq60_inst1_train_diff/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-10-16-12/last_forcasting_model_weight.pth"
# # export forcasting_model_weights
# ###### Forcasting diffusion model setting #######


###### Partial to complete model setting #######
# use_partial_to_complete_model, partial_to_complete_model_weights
# export use_partial_to_complete_model=True
# export partial_to_complete_model_weights='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_histindex_maskedfuture_v2/model006150002.pt'
##### Partial to complete model setting #######




# export st_ed_state_cond=True


######## Open loop testing code ########
export open_loop_test=False
# export open_loop_test=True

# export obj_type_to_pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
# export obj_type_to_pre_optimized_traj='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res_to_s1.npy'
# export pre_load_trajectories=True
# export use_base_traj=True
######## Open loop testing code ########


export use_future_ref_as_obs_goal=True


export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal_franka/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-26-53/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/home/xymeow/xueyi/IsaacGymEnvs2/isaacgymenvs/ckpts/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_randcond_distill_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-21-21-20/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_wfuture_freq5_ws5_s2_contactcond_distll_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-11-15-38/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
####### Forcasting model setting #########






####### Use the window future selection setting #######
# export use_window_future_selection=True
# export use_window_future_selection=False
# export use_forcasting_model=False
# export use_history_obs=False
####### Use the window future selection setting #######



######## Downsample configs ########
# export dt=0.0166  
# export controlFrequencyInv=1
# export controlFrequencyInv=2
# export downsample=True
######## Downsample configs ########





###### GRAB-300 test setting ########






# export max_epochs=8000
# export max_epochs=4500
# export max_epochs=2100
# export max_epochs=1000
export max_epochs=8000





export debug=""
export subj_nm='s9'
export subj_nm='s7'
export subj_nm=''
export subj_nm='s1'
export subj_nm='s1'
# export subj_nm='s2'
export subj_nm='s2'
# export subj_nm='s10'
export subj_nm='s6'
export subj_nm='s5'
export subj_nm='s4'
export subj_nm='s2'
# export subj_nm='s1'
# export subj_nm=''


export single_inst_tag='ori_grab_s2_flute_pass_1_nf_300'
export single_inst_tag='ori_grab_s2_hammer_use_2_nf_300'
export single_inst_tag='ori_grab_s2_duck_inspect_1_nf_300'
export single_inst_tag='ori_grab_s2_flashlight_on_1_nf_300'
# export single_inst_tag='ori_grab_s2_duck_inspect_1_nf_300'

export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_condobj_' 
export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_condhandobj_' 
export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_condhandobj_invfreq10_' 
export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_' 
export log_path='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_' 




export exclude_inst_tag_to_opt_res_fn=''
export st_idx=0

export test=True


# export numEnvs=10
# export minibatch_size=10

export numEnvs=1000
export minibatch_size=1000


export numEnvs=100
export minibatch_size=100




export w_franka=True
export tracking_save_info_fn='./data/GRAB_Tracking_PK_OFFSET_Reduced/data'
export tracking_data_sv_root='./data/GRAB_Tracking_PK_OFFSET_Reduced/data'
export table_z_dim=0.6
export tracking_save_info_fn='./data/GRAB_Tracking_PK_OFFSET_0d4/data'
export tracking_data_sv_root='./data/GRAB_Tracking_PK_OFFSET_0d4/data'
export table_z_dim=0.4
export maxx_inst_nn=1
export use_local_canonical_state=False 
export add_table=True

export grab_inst_tag_to_optimized_res_fn='./data/statistics/data_inst_tag_to_optimized_res.npy'
export object_type_to_latent_feature_fn='./data/statistics/obj_type_to_obj_feat.npy'
export inst_tag_to_latent_feature_fn='./data/statistics/inst_tag_to_obj_feat.npy'
export mocap_sv_info_fn='./data/GRAB_Tracking_PK_OFFSET_Reduced/data/passive_active_info_ori_grab_s2_apple_lift_nf_300.npy'



export headless=True
export st_idx=0

export gpu_offset_idx=3


export pure_test_inst_tag="ori_grab_s2_flashlight_on_1"
export pure_test_inst_tag="ori_grab_s2_hammer_use_2"
## mass in sim = 70g #
# export pure_test_inst_tag="ori_grab_s2_banana_eat_1"
# export pure_test_inst_tag="ori_grab_s2_duck_inspect_1"
# export pure_test_inst_tag="ori_grab_s2_apple_eat_1"
# export pure_test_inst_tag="ori_grab_s2_hand_inspect_1"
# export pure_test_inst_tag="ori_grab_s2_elephant_inspect_1"
# export pure_test_inst_tag="ori_grab_s1_banana_peel_1"
# export pure_test_inst_tag="ori_grab_s1_watch_set_2"
# export pure_test_inst_tag='ori_grab_s1_waterbottle_pour_1'
# export pure_test_inst_tag="ori_grab_s1_waterbottle_pass_1"
# export pure_test_inst_tag="ori_grab_s1_wineglass_drink_2"
# export pure_test_inst_tag="ori_grab_s1_wineglass_toast_1"
# export pure_test_inst_tag="ori_grab_s2_cubesmall_inspect_1"
# export pure_test_inst_tag='ori_grab_s1_lightbulb_pass_1'
# export pure_test_inst_tag='ori_grab_s3_phone_call_1'
# export pure_test_inst_tag='ori_grab_s3_toothpaste_squeeze_1'
# export pure_test_inst_tag='ori_grab_s1_banana_peel_1'
# export pure_test_inst_tag='ori_grab_s6_banana_peel_2'
# export pure_test_inst_tag='ori_grab_s3_mouse_pass_1'
# export pure_test_inst_tag='ori_grab_s2_train_pass_1'
# export pure_test_inst_tag='ori_grab_s2_elephant_inspect_1'
# export pure_test_inst_tag='ori_grab_s2_flashlight_on_1'
# export pure_test_inst_tag='ori_grab_s2_hand_inspect_1'
# export pure_test_inst_tag="ori_grab_s1_cup_pour_1"
# export pure_test_inst_tag='ori_grab_s8_phone_call_1'
# export pure_test_inst_tag='ori_grab_s3_phone_call_1'
# export pure_test_inst_tag='ori_grab_s2_knife_pass_1'
# export pure_test_inst_tag='ori_grab_s8_phone_call_1'
# export pure_test_inst_tag='ori_grab_s8_toothpaste_squeeze_1'
# export pure_test_inst_tag='ori_grab_s9_banana_pass_1'
# export pure_test_inst_tag='ori_grab_s6_mug_toast_1'



export single_inst_tag=${pure_test_inst_tag}_nf_300
# export single_inst_tag='ori_grab_s2_gamecontroller_pass_1_nf_300'
# export single_inst_tag='ori_grab_s2_hand_lift_nf_300'
# export single_inst_tag='ori_grab_s2_apple_eat_1_nf_300'
# export single_inst_tag='ori_grab_s2_elephant_inspect_1_nf_300'
# export single_inst_tag='ori_grab_s2_hammer_lift_nf_300'
# export single_inst_tag='ori_grab_s2_apple_lift_nf_300'
# export single_inst_tag='ori_grab_s2_hand_inspect_1_nf_300'




export hand_pose_guidance_glb_trans_coef=0.1
export hand_pose_guidance_glb_rot_coef=0.1
export hand_pose_guidance_fingerpose_coef=0.1



export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.1
export rew_delta_hand_pose_coef=0.05



export w_finger_pos_rew=True
export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.3
export hand_pose_guidance_glb_trans_coef=0.6
export hand_pose_guidance_glb_rot_coef=0.6
export hand_pose_guidance_fingerpose_coef=0.2
export kine_info_with_arm_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d6_0d4_warm/data'
export franka_delta_delta_mult_coef=1.0
export franka_delta_delta_mult_coef=0.5



export load_kine_info_retar_with_arm=True
export kine_info_with_arm_sv_root='./data/GRAB_Tracking_PK_OFFSET_warm/data'
# export kine_info_with_arm_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_warm/data'
export kine_info_with_arm_sv_root=/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d4_0d5_warm/data
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d4_0d5_warm/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d4_0d5_warm/data'
export table_z_dim=0.5


export control_arm_via_ik=False


##### control strategy ######
export control_arm_via_ik=True
##### control strategy ######




# ###### LEAP hand setting ######
export hand_type='leap'
export kine_info_with_arm_sv_root=/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm/data
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm/data'
export table_z_dim=0.5
export tracking_info_st_tag='leap_passive_active_info_'
# ###### LEAP hand setting ######



export wo_vel_obs=False
export wo_vel_obs=True


###### LEAP hand with version 2 urdf ######
export kine_info_with_arm_sv_root=/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data'
export use_v2_leap_warm_urdf=False
###### LEAP hand with version 2 urdf ######





##### LEAP hand setting --- checkpoints #####
## elephant ##
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s2_apple_eat_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-15-08-45/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_600_rew_155.34076.pth'
############### banana (no dr) ###############
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s1_banana_peel_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-20-43-59/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s1_banana_peel_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-20-42-22/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
############### flashlight ###############
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-02-13-33/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-01-38-56/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev41_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-02-20-34/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_152.14603.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev41_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-02-14-30/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# ############### banana ###############
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev41_rewv5_franka_ori_grab_s1_banana_peel_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-19-06-03/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsTrue_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-10-09-43/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_123.363365.pth'
############### elephant, with DR, behaviour is strange, but can work a little bit ###############
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev41_rewv5_franka_ori_grab_s2_elephant_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-01-33-26/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_104.26583.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_elephant_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-22-52-43/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_172.04007.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsTrue_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev41_rewv5_franka_ori_grab_s2_elephant_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-18-41-22/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-12-39-23/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1400_rew_189.31245.pth'
############### duck with more DR ###############
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev40_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-02-21/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_99.85559.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev40_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-02-25-25/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
##### LEAP hand setting --- checkpoints #####
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_elephant_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-22-52-43/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_158.08421.pth'
###### LEAP hand, waterbottle jpass 1 #########
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s1_waterbottle_pass_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-03-02-25/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
###### LEAP hand, cubesmall inspect 1 #########
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_cubesmall_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-03-05-39/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
###### LEAP hand, wineglass drink 2 #########
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s1_wineglass_drink_2_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-12-05-16/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
###### LEAP hand, wineglass toast 1 #########
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s1_wineglass_toast_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-15-10-48/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftrotvelTrue_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev14_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-12-05-57/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1.pth'


####### use_v2_leap_warm_urdf setting #######
export kine_info_with_arm_sv_root=/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data'
export use_v2_leap_warm_urdf=True
# ####### use_v2_leap_warm_urdf setting #######



# ################# Rorientation test settings ##################
# ### flashlight on 1 ###
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-12-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1000_rew_99.20196.pth'
# #### duck insepct 1 ###
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-20-05-49/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1600_rew_170.89615.pth'
# ### apple eat 1 ####
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_apple_eat_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-19-17/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1000_rew_121.545876.pth'
# # #### hammer use 1 ####
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_hammer_use_2_nf_300_armmult2.0_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-20-19-16/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1600_rew_71.641.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_hammer_use_2_nf_300_armmult2.0_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-20-22-49/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1400_rew_118.88766.pth'
# ####  duck reoreinetaitn settings ####
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasFalse_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-20-13-46/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_114.719604.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-20-16-03/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_238.4407.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-20-14-13/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_125.938545.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-16-24/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# # export wo_vel_obs=False
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasFalse_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-07-05/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drTruev35_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-23-24-06/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1000_rew_-1.9951135.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_apple_eat_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-19-17/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1600_rew_148.37556.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_hammer_use_2_nf_300_armmult2.0_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-20-19-16/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2800_rew_85.14848.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_hammer_use_2_nf_300_armmult2.0_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-20-22-49/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2600_rew_133.28648.pth'
# # export wo_vel_obs=False
# # export checkpoint=
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_rewv7_reornt_franka_teacher_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_16-16-33-14/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev29_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-12-01-27/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_143.54118.pth'
# ### without fingertip velocities and orientations ###
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_rewv7_reornt_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-10-43-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_600_rew_149.85504.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_rewv7_reornt_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-10-43-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1200_rew_166.40463.pth'
# # hammer reorient dr 35 # # 
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drTruev35_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-21-56-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1000_rew_-8.863782.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-04-56/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasFalse_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-07-05/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_154.10251.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev29_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-12-01-27/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2400_rew_173.7865.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-04-56/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev29_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-12-01-27/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2200_rew_173.42041.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-16-24/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_138.7791.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-12-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2400_rew_130.03404.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_apple_eat_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-19-17/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2400_rew_159.57864.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasFalse_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-07-05/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1400_rew_166.33434.pth'
# # duck inspect, without the reorientation, without randomizations #
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drTruev35_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-28-26/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1600_rew_176.63686.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasFalse_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-07-05/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3600_rew_171.68793.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-16-24/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3400_rew_180.27975.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-04-56/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3400_rew_306.11948.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-12-00-17/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3600_rew_173.37283.pth'

# ########## s1_watch_set_2 ##########
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_watch_set_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-19-00-09/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1200_rew_136.20778.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_watch_set_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-19-00-09/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3400_rew_155.39694.pth'

# # ####### Flashlight can work #######
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_flashlight_on_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-12-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2600_rew_133.09106.pth'

# ####### apple eat ########
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_apple_eat_1_nf_300_armmult2.0_table0.5_wovelTrue_reornt/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-19-17/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2600_rew_161.19783.pth'


# ### hammer can wrok ###
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_hammer_use_2_nf_300_armmult2.0_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-20-19-16/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2800_rew_85.14848.pth'

# # #### v1 urdf in the reorientation setting ######
# # export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v7
# # export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v7
# # export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v7
# # export use_v2_leap_warm_urdf=False
# # #### v1 urdf in the reorientation setting ######

# ##### waterbottle pass 1 #####
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_waterbottle_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-10-08-07/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2200_rew_46.514874.pth'

# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-02-29-12/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3000_rew_176.0377.pth'

# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_cubesmall_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-02-39-06/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_5200_rew_152.64369.pth'
# export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-09-59-35/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2400_rew_89.54613.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-23-37-13/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# # export checkpoint='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-10-53-26/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_141.42554.pth'

# ##### v2 urdf in the reorientation setting #####
# export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
# export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
# export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
# export use_v2_leap_warm_urdf=True
# ##### v2 urdf in the reorientation setting #####

# # ##### v3 urdf in the reorientation setting #####
# # export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v9
# # export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v9
# # export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v9
# # export use_v2_leap_warm_urdf=True
# # ##### v3 urdf in the reorientation setting #####


# export hand_glb_mult_factor_scaling_coef=0.00001
# export hand_glb_mult_scaling_progress_after=120
# export w_obj_ornt=True
# export include_obj_rot_in_obs=True
# ################# Rorientation test settings ##################

export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
export use_v2_leap_warm_urdf=True


export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v6urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v15
export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v6urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v15
export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v6urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v15
export use_v2_leap_warm_urdf=True



export add_physical_params_in_obs=False
export controlFrequencyInv=3


export dofSpeedScale=1
# export dofSpeedScale=20

export wo_fingertip_pos=False

export not_use_kine_bias=True


export warm_trans_actions_mult_coef=0.01
export warm_rot_actions_mult_coef=0.01
# export warm_trans_actions_mult_coef=0.005
# export warm_rot_actions_mult_coef=0.005

# 
# 
###### Whether to diable the contact to vis the actions rollout ######
export disable_hand_obj_contact=True
export disable_hand_obj_contact=False
###### Whether to diable the contact to vis the actions rollout ######


export wo_fingertip_vel=False
export wo_fingertip_rot_vel=False

# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# export wo_fingertip_rot_vel=False
# export wo_fingertip_vel=True


# export wo_fingertip_rot_vel=False
# export wo_fingertip_vel=False
# export wo_fingertip_pos=True


# # reset_obj_mass: False
# obj_mass_reset: 0.27
# recompute_inertia: False


export reset_obj_mass=False
export obj_mass_reset=0.27
export recompute_inertia=False

# export add_physical_params_in_obs=True

# export reset_obj_mass=True
# export obj_mass_reset=0.120
# export obj_mass_reset=0.143
# export obj_mass_reset=0.020
# export obj_mass_reset=0.0833
# export recompute_inertia=False

# export arm_stiffness=100000
# export arm_effort=100000
# export arm_damping=80


export gpu_offset_idx=1



export numEnvs=200
export minibatch_size=200



# export not_use_kine_bias=False



########################## Local test settings; headless = False configs ##########################
export estimate_vels=False
### control freq for real tests ###
export table_z_dim=0.5
export numEnvs=200
export minibatch_size=200
#### only effective for the duck and elephant instance ####
export tracking_save_info_fn='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data'
export tracking_data_sv_root='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data'
export kine_info_with_arm_sv_root='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data'
export use_v2_leap_warm_urdf=False
export tracking_save_info_fn='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v3/data'
export tracking_data_sv_root='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v3/data'
export kine_info_with_arm_sv_root='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v3/data'
export controlFrequencyInv=3
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1.pth'
# export checkpoint='/home/xymeow/xueyi/IsaacGymEnvs2/isaacgymenvs/ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_ctlfreq3_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_ctlfreq3_delta1_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_ctlfreq3_nkinebias_trcoef0.01_delta1_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_185.60869.pth'
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_dr1.pth'
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_dr2.pth'
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_wovelrot.pth'
#### models trained with randomiztions ####
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_drv4_voftvels.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_151.04553.pth'
#### models trained with randomiztions ####
#### models trained with randomization -- full observations ####
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_600_rew_133.84709.pth'
#### models trained with randomization -- full observations ####
#### models trained with randomization v5 -- partial observations ####
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_157.99556.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_169.98018.pth'
#### models trained with randomization v5 -- partial observations ####
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_95.38914.pth'
#### DR ####
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_drv14.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_153.24663_drv20.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1000_rew_164.09914_dr22.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2600_rew_166.43976_drmassv2.pth'
export Checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2400_rew_129.99924_handinspect_drv26.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2400_rew_172.04095_handinspect_ndr.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_171.44658_appleeat.pth'
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_elephant_drv41.pth'
export checkpoint=ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_bananapeel_dr41.pth
export checkpoint=ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_123.363365_bananapeel_dr35.pth
export checkpoint='ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_duckinspect_drv40.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_152.14603_flashlight_drv41.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1600_rew_176.63686_duckreorient_dr35.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2800_rew_85.14848_hammer_reorient_wovels.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2600_rew_133.09106_flashlight_reorient.pth'
# export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1200_rew_109.517334_dr23.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_elephant_1.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_distill_duck.pth'
# export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_181.60353.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_dr2.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_3.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_185.60869.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_dr1.pth'
# export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_dr2.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_600_rew_133.84709.pth'
# export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_157.99556.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_169.98018.pth'
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_95.38914.pth'
export checkpoint='./ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_drv14.pth'
#### elephant inspect ####
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_158.08421_elephant_inspect.pth'
export checkpoint='ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_200_rew_171.44658_appleeat.pth'
#### banana peeel ####
export checkpoint='ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_banana_peel_ndr.pth'
#### bottle pass ####
export checkpoint='ckpts/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_1_waterbottle_pass.pth'
export gpu_offset_idx=0
export headless=False
export log_path='./isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_' 
# ### TO Real setting ###
export toreal=1
export headless=True
# export closed_loop_to_real=True
# export numEnvs=2
# export minibatch_size=2
# ### TO Real setting ###
export gpu_offset_idx=0
# export disable_hand_obj_contact=True
export disable_hand_obj_contact=False
# export estimate_vels=True




###### reorientation local test settings ########
export checkpoint='./ckpts/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1600_rew_177.48398_reornt.pth'

### v1 retargeted kinematcis folders ###
export kine_info_with_arm_sv_root=./data/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v7
export tracking_save_info_fn=./data/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v7
export tracking_data_sv_root=./data/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v7
export use_v2_leap_warm_urdf=False
### v1 retargeted kinematics folders ###

### v2 retargeted kinematics folders ###
export kine_info_with_arm_sv_root=data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
export tracking_save_info_fn=data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
export tracking_data_sv_root=data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
export use_v2_leap_warm_urdf=True
### v2 retargeted kinematics folders ###


export kine_info_with_arm_sv_root=data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export tracking_save_info_fn=data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export tracking_data_sv_root=data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export use_v2_leap_warm_urdf=True


# ### v2 retargeted kinematics folders ###
# export kine_info_with_arm_sv_root=data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v9
# export tracking_save_info_fn=data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v9
# export tracking_data_sv_root=data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v9
# export use_v2_leap_warm_urdf=True
# ### v2 retargeted kinematics folders ###


export hand_glb_mult_factor_scaling_coef=0.00001
export hand_glb_mult_scaling_progress_after=120
export w_obj_ornt=True
export include_obj_rot_in_obs=True
###### reorientation local test settings #########




# ####### local presaved action files ######
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_22340_duck_reorient_presaved_acts.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_126800_hammer_reorient_presaved_acts.npy'
# export preload_action_target_env_idx=99
# export preload_action_start_frame=0
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_117160_waterbottle_reorient_presaved_acts.npy'
# export preload_action_target_env_idx=50
# export preload_action_start_frame=49
# # export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_113520_watch_reorient_presaved_acts.npy'
# # export preload_action_target_env_idx=45
# # export preload_action_start_frame=206
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_41320_hammeruse2_v8_reorient.npy'
# export preload_action_target_env_idx=62
# export preload_action_start_frame=22
# ## not work... ##
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_41560_s1watchset2_reorient.npy'
# export preload_action_target_env_idx=80
# export preload_action_start_frame=14
# # export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_52580_s1waterbottlepour1_reorient.npy'
# # export preload_action_target_env_idx=0
# # export preload_action_start_frame=130
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_47680_s2appleeat1_reorient.npy'
# export preload_action_target_env_idx=52
# export preload_action_start_frame=9
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_47660_s1lightbulbpass1_reorient.npy'
# export preload_action_target_env_idx=69
# export preload_action_start_frame=48
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_4620_duck_inspect_1_stiffv2.npy'
# export preload_action_target_env_idx=6
# export preload_action_start_frame=295
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_5500_cubesmall_inspect_1.npy'
# export preload_action_target_env_idx=97
# export preload_action_start_frame=47
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_7880_waterbottle_pour_1.npy'
# export preload_action_target_env_idx=34
# export preload_action_start_frame=94
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_24040_s1wineglasstoast.npy'
# export preload_action_target_env_idx=56
# export preload_action_start_frame=198
# # export reset_obj_mass=True
# # export obj_mass_reset=0.104
# # export recompute_inertia=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_78460_s3phonecall.npy'
# export preload_action_target_env_idx=14
# export preload_action_start_frame=95
# export preload_action_target_env_idx=41
# export preload_action_start_frame=149
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_36600_s1lightbulbpass1.npy'
# export preload_action_target_env_idx=71
# export preload_action_start_frame=130
# export reset_obj_mass=True
# export obj_mass_reset=0.104
# export recompute_inertia=True
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42220_lightbulb_stiffnessv2.npy'
# export preload_action_target_env_idx=44
# export preload_action_start_frame=200
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.090
# # export obj_mass_reset=0.104
# export recompute_inertia=False
# <<<<<<< HEAD
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_78460_s3phonecall.npy'
export preload_action_target_env_idx=14
export preload_action_start_frame=95
export preload_action_target_env_idx=41
export preload_action_start_frame=149
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_36600_s1lightbulbpass1.npy'
export preload_action_target_env_idx=71
export preload_action_start_frame=130
export reset_obj_mass=True
export obj_mass_reset=0.104
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42220_lightbulb_stiffnessv2.npy'
export preload_action_target_env_idx=44
export preload_action_start_frame=200
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.090
# export obj_mass_reset=0.104
export recompute_inertia=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_37400_watchset2.npy'
export preload_action_target_env_idx=17
export preload_action_start_frame=228
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.090
# export obj_mass_reset=0.104
export recompute_inertia=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_41560_s1watchset2_reorient.npy'
export preload_action_target_env_idx=80
export preload_action_start_frame=14
export stiffness_coef=100
export damping_coef=4
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_117840_s1watchset2_reornt_v2.npy'
export preload_action_target_env_idx=52
export preload_action_start_frame=249
# =======
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_37400_watchset2.npy'
# export preload_action_target_env_idx=17
# export preload_action_start_frame=228
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.090
# # export obj_mass_reset=0.104
# export recompute_inertia=False
# >>>>>>> 3b30b961f722dd05798b0f2409f17eb9c476a331
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_40020_bananapeel.npy'
# export preload_action_target_env_idx=51
# export preload_action_start_frame=80
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.090
# # export obj_mass_reset=0.104
# export recompute_inertia=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_22160_hammer_use_2_v2.npy'
# export preload_action_target_env_idx=61
# export preload_action_start_frame=212
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_28400_hammer_use_v8_v18.npy'
# export preload_action_target_env_idx=52
# export preload_action_start_frame=249
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_21440_s6_banana_peel_2.npy'
# export preload_action_target_env_idx=99
# export preload_action_start_frame=176
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_43020_s3_mouse_pass_1.npy'
# export preload_action_target_env_idx=5
# export preload_action_start_frame=23
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.10
# export recompute_inertia=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_16040_s6_banana_peel_2_v2.npy'
# export preload_action_target_env_idx=45
# export preload_action_start_frame=26
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# <<<<<<< HEAD
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_11820_s2flashlighton1_reorient.npy'
export preload_action_target_env_idx=87
export preload_action_start_frame=160
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_313680_gene_s2.npy'
export preload_action_target_env_idx=61
export preload_action_start_frame=242
export preload_action_target_env_idx=10
export preload_action_start_frame=200
export stiffness_coef=100
export damping_coef=4
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_9800_banana_peel_2_reornt_v2.npy'
export preload_action_target_env_idx=7
export preload_action_start_frame=246
export stiffness_coef=100
export damping_coef=4
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_45300_lightbulb_pass_1_reorient_v22.npy'
export preload_action_target_env_idx=24
export preload_action_start_frame=149
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_45300_lightbulb_pass_1_reorient_v322.npy'
export preload_action_target_env_idx=27
export preload_action_start_frame=216
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_36620_handinspect_1_reorient.npy'
export preload_action_target_env_idx=38
export preload_action_start_frame=70
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_50980_lightbulb_pass_lightermass.npy'
export preload_action_target_env_idx=92
export preload_action_start_frame=241
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.0431
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_24780_cuppourreorient.npy'
export preload_action_target_env_idx=42
export preload_action_start_frame=190
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_29100_s8phonecall_oreinet.npy'
export preload_action_target_env_idx=46
export preload_action_start_frame=76
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42300_phone_reorient_v1.npy'
export preload_action_target_env_idx=9
export preload_action_start_frame=121
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_43540_phone_reorinet_v2.npy'
export preload_action_target_env_idx=47
export preload_action_start_frame=5
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_59900_s8_toothpaste_squeeze_reorint.npy'
export preload_action_target_env_idx=30
export preload_action_start_frame=0
export stiffness_coef=210
export damping_coef=20
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_16920_s8_phone_call_reorient.npy'
# export preload_action_target_env_idx=24
# export preload_action_start_frame=95
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_158300_knife_pass_reorient_v1.npy'
# export preload_action_target_env_idx=90
# export preload_action_start_frame=169
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
export preload_action_targets_fn="ckpts/ts_to_hand_obj_obs_reset_4580_reorient_phone_v3.npy"
export preload_action_target_env_idx=37
export preload_action_start_frame=249
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.0681
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_5500_phone_reorient_v4.npy'
export preload_action_target_env_idx=61
export preload_action_start_frame=32
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.0681
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_7700_phone_reorient_v5.npy'
export preload_action_target_env_idx=8
export preload_action_start_frame=137
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.0681
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_12620_reorient_phone_v7.npy'
export preload_action_target_env_idx=26
export preload_action_start_frame=54
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.0681
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_8780_banana_eat_reorient_v1.npy'
export preload_action_target_env_idx=15
export preload_action_start_frame=88
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=True
export obj_mass_reset=0.0435
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_17440_banana_eat_reorient_v8.npy'
export preload_action_target_env_idx=0
export preload_action_start_frame=144
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42780_banana_eat_reorient_v9.npy'
export preload_action_target_env_idx=0
export preload_action_start_frame=110
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_44100_banana_eat_reorient_v10.npy'
export preload_action_target_env_idx=31
export preload_action_start_frame=42
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_41860_banana_eat_reorient_v11.npy'
export preload_action_target_env_idx=33
export preload_action_start_frame=251
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_23160_banana_eat_reorient_v12.npy'
export preload_action_target_env_idx=13
export preload_action_start_frame=186
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_21620_banana_eat_reorient_v13.npy'
export preload_action_target_env_idx=26
export preload_action_start_frame=162
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_1_hammer_reorient_sample_6.npy'
export preload_action_target_env_idx=35
export preload_action_start_frame=0
export stiffness_coef=210
export damping_coef=20
export reset_obj_mass=False
export obj_mass_reset=0.0435
export recompute_inertia=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_1_hammer_reorient_sample_9.npy'
export preload_action_target_env_idx=3
export preload_action_start_frame=0
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_54460_s6_mugtoast_1_reorient_v1.npy'
export preload_action_target_env_idx=34
export preload_action_start_frame=135
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_59540_s6_mugtoast_1_reorient_v2.npy'
export preload_action_target_env_idx=93
export preload_action_start_frame=169
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_47740_hammer_reorient_v13.npy'
export preload_action_target_env_idx=59
export preload_action_start_frame=3
export reset_obj_mass=True
export obj_mass_reset=0.0833
export recompute_inertia=True
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_90800_hammer_reorient_v14.npy'
export preload_action_target_env_idx=9
export preload_action_start_frame=207
export reset_obj_mass=False
export add_global_movements=True
export add_global_movements_af_step=367
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_146040_hammer_reorient_v15.npy'
# export preload_action_target_env_idx=73
# export preload_action_start_frame=231
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_120600_hammer_reorient_v16.npy'
# export preload_action_target_env_idx=90
# export preload_action_start_frame=2
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_1_hammer_sample_30_vxx.npy'
export preload_action_target_env_idx=52
export preload_action_start_frame=0
export add_global_movements=False
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_64800_hammer_reornt_vxx2.npy'
export preload_action_target_env_idx=34
export preload_action_start_frame=39
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_106140_hammer_reoreint_v2urdf_v8.npy'
export preload_action_target_env_idx=22
export preload_action_start_frame=141
export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_41320_hammeruse2_v8_reorient.npy'
export preload_action_target_env_idx=62
export preload_action_start_frame=22
export add_global_movements=True
export add_global_movements_af_step=220
########################## Local test settings; headless = False configs ##########################
# =======
# ########################## Local test settings; headless = False configs ##########################
# >>>>>>> 3b30b961f722dd05798b0f2409f17eb9c476a331






####### local use_v2_leap_warm_urdf setting #######
# export kine_info_with_arm_sv_root=./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data
# export tracking_save_info_fn='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data'
# export tracking_data_sv_root='./data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data'
# export tracking_save_info_fn=./data/modified_kinematics_data_leap_wfranka_v17urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export kine_info_with_arm_sv_root=./data/modified_kinematics_data_leap_wfranka_v17urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export tracking_save_info_fn=./data/modified_kinematics_data_leap_wfranka_v17urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export tracking_data_sv_root=./data/modified_kinematics_data_leap_wfranka_v17urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
export kine_ed_tag="_sample_0.npy"
export strict_maxx_nn_ts=500
export use_actual_traj_length=True
export use_v2_leap_warm_urdf=True
####### local use_v2_leap_warm_urdf setting #######




# export kine_info_with_arm_sv_root=/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d6_0d4_warm/data
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d6_0d4_warm/data'
# export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d6_0d4_warm/data'
# export table_z_dim=0.6
# export franka_delta_delta_mult_coef=1.0

# export hand_pose_guidance_glb_trans_coef=0.02
# export hand_pose_guidance_glb_rot_coef=0.02
# export hand_pose_guidance_fingerpose_coef=0.1

# export single_instance_state_based_train=True

##### activate the single instance state based train config #####
# export single_instance_state_based_train=True
export train_on_all_trajs=True
##### activate the single instance state based train config #####


# export preload_action_targets_fn=''

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_duck_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-09-59-57/ts_to_hand_obj_obs_reset_22340.npy'
# export preload_action_target_env_idx=0
# export preload_action_start_frame=190



# ###### waterbottle pass 1, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_waterbottle_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-10-03-46/ts_to_hand_obj_obs_reset_117160.npy' # start frame = 49, idxx = 50
# export preload_action_target_env_idx=50
# export preload_action_start_frame=49
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_waterbottle_pass_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-22-59-57/ts_to_hand_obj_obs_reset_1.npy
# ###### waterbottle pass 1, reorientation ######


# ###### hammer use 2, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-09-59-35/ts_to_hand_obj_obs_reset_126800.npy'
# export preload_action_target_env_idx=99
# export preload_action_start_frame=0
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_hammer_use_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-03-12/ts_to_hand_obj_obs_reset_1.npy
# ###### hammer use 2, reorientation ######



# ###### watch set 2, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_watch_set_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-10-01-30/ts_to_hand_obj_obs_reset_113520.npy'
# export preload_action_target_env_idx=45
# export preload_action_start_frame=206
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_watch_set_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-09-51/ts_to_hand_obj_obs_reset_1.npy
# ###### watch set 2, reorientation ######


# ###### hammer use 2, v8, v2urdf, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-16-11-56/ts_to_hand_obj_obs_reset_41320.npy'
# export preload_action_target_env_idx=62
# export preload_action_start_frame=22
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_hammer_use_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-12-48/ts_to_hand_obj_obs_reset_1.npy
# ###### hammer use 2, v8, v2urdf, reorientation ######


# # ###### s1 watch set 2, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_watch_set_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-16-14-53/ts_to_hand_obj_obs_reset_41560.npy'
# export preload_action_target_env_idx=80
# export preload_action_start_frame=14
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_watch_set_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-16-56/ts_to_hand_obj_obs_reset_1.npy
# # ###### s1 watch set 2, reorientation ######


# ###### s1 waterbottle pour, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_waterbottle_pour_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-16-43-35/ts_to_hand_obj_obs_reset_52580.npy'
# export preload_action_target_env_idx=0
# export preload_action_start_frame=130
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_waterbottle_pour_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-21-11/ts_to_hand_obj_obs_reset_1.npy
# ###### s1 waterbottle pour, reorientation ######


# ###### s2 apple eat 1, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_apple_eat_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-18-54-40/ts_to_hand_obj_obs_reset_47680.npy'
# export preload_action_target_env_idx=52
# export preload_action_start_frame=9
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_apple_eat_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-27-57/ts_to_hand_obj_obs_reset_1.npy
# ###### s2 apple eat 1, reorientation ######


# ###### lightbulb pass 1, reorientation ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_lightbulb_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-18-58-00/ts_to_hand_obj_obs_reset_47660.npy'
# export preload_action_target_env_idx=69
# export preload_action_start_frame=48
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_lightbulb_pass_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-30-18/ts_to_hand_obj_obs_reset_1.npy
# ###### lightbulb pass 1, reorientation ######


# #### not good... #####
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drTruev35_rewv7_franka_ori_grab_s1_watch_set_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_bash scripts_new/run_tracking_headless_grab_whlsample_wfranka.shdensity_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-19-01-01/ts_to_hand_obj_obs_reset_37400.npy'
# export preload_action_target_env_idx=31
# export preload_action_start_frame=0
# #### not good... #####

# # #### not good... #####
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drTruev35_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-19-02-59/ts_to_hand_obj_obs_reset_46860.npy'
# export preload_action_target_env_idx=99
# export preload_action_start_frame=147
# export reset_obj_mass=True
# export obj_mass_reset=0.0833
# export recompute_inertia=False
# ##### PD v2 #####
# export stiffness_coef=210
# export damping_coef=20
# ##### PD v2 #####
# # #### not good... #####



# # ###### duck inspect 1 ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_duck_inspect_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-18-56-30/ts_to_hand_obj_obs_reset_4620.npy'
# export preload_action_target_env_idx=6
# export preload_action_start_frame=295
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True
# # ###### duck inspect 1 ######


# ###### cubesmall inspect 1 ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka_ori_grab_s2_cubesmall_inspect_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-18-57-37/ts_to_hand_obj_obs_reset_5500.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_5500_cubesmall_inspect_1.npy'
# export preload_action_target_env_idx=97
# export preload_action_start_frame=47
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True
# # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_cubesmall_inspect_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-21-50-51/ts_to_hand_obj_obs_reset_1.npy
# ###### cubesmall inspect 1 ######

# ###### waterbottle pour 1 ######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s1_waterbottle_pour_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-18-58-27/ts_to_hand_obj_obs_reset_7880.npy'
# /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_waterbottle_pour_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-22-49-07/ts_to_hand_obj_obs_reset_1.npy
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_7880_waterbottle_pour_1.npy'
# export preload_action_target_env_idx=34
# export preload_action_start_frame=94
# ###### waterbottle pour 1 ######

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s1_wineglass_toast_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-22-45-41/ts_to_hand_obj_obs_reset_24040.npy'
# export preload_action_target_env_idx=56
# export preload_action_start_frame=198
# export reset_obj_mass=True
# export obj_mass_reset=0.104
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# 14, 95
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka__armmult2.0_table0.5_subj3_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-18-49-52/ts_to_hand_obj_obs_reset_78460.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_78460_s3phonecall.npy'
# export preload_action_target_env_idx=14
# export preload_action_start_frame=95
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# # 41, 149
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka__armmult2.0_table0.5_subj3_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-18-49-52/ts_to_hand_obj_obs_reset_78460.npy'
# export preload_action_target_env_idx=41
# export preload_action_start_frame=149
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s1_lightbulb_pass_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-00-16-54/ts_to_hand_obj_obs_reset_36600.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_36600_s1lightbulbpass1.npy'
# export preload_action_target_env_idx=71
# export preload_action_start_frame=130
# export reset_obj_mass=True
# export obj_mass_reset=0.104
# export recompute_inertia=True
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


######## lightbulb pass 1, stiffness v2 #######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s1_lightbulb_pass_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-00-18-42/ts_to_hand_obj_obs_reset_42220.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42220_lightbulb_stiffnessv2.npy'
# export preload_action_target_env_idx=44
# export preload_action_start_frame=200
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.090
# # export obj_mass_reset=0.104
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True
######## lightbulb pass 1, stiffness v2 #######


# ######## watch set 2, stiffness v2 #######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s1_watch_set_2_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-00-35-55/ts_to_hand_obj_obs_reset_37400.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_37400_watchset2.npy'
# export preload_action_target_env_idx=17
# export preload_action_start_frame=228
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.090
# # export obj_mass_reset=0.104
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True
# ######## watch set 2, stiffness v2 #######



# ######## s1 banana peel 1, stiffness v2 #######
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s1_banana_peel_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-00-22-30/ts_to_hand_obj_obs_reset_40020.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_40020_bananapeel.npy'
# export preload_action_target_env_idx=51
# export preload_action_start_frame=80
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.090
# # export obj_mass_reset=0.104
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True
# ######## s1 banana peel 1, stiffness v2 #######



# # ### single target frame ###
# export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
# export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
# export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
# export use_v2_leap_warm_urdf=True
# ### single target frame ###
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s6_banana_peel_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-00-17-02/ts_to_hand_obj_obs_reset_21440.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_21440_s6_banana_peel_2.npy'
# export preload_action_target_env_idx=99
# export preload_action_start_frame=176
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s6_banana_peel_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-10-41-09/ts_to_hand_obj_obs_reset_16040.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_16040_s6_banana_peel_2_v2.npy'
# export preload_action_target_env_idx=45
# export preload_action_start_frame=26
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False



# ### dense hand and object target frames ###
# export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
# export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
# export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v8
# export use_v2_leap_warm_urdf=True
# ### dense hand and object target frames ###
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-00-37-31/ts_to_hand_obj_obs_reset_22160.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_22160_hammer_use_2_v2.npy'
# export preload_action_target_env_idx=61
# export preload_action_start_frame=212
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True



# ### single target frame ###
# export kine_info_with_arm_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
# export tracking_save_info_fn=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
# export tracking_data_sv_root=/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf/GRAB_Tracking_PK_reduced_300_resampled_${pure_test_inst_tag}_v18
# export use_v2_leap_warm_urdf=True
# ### single target frame ###
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_09-20-35-29/ts_to_hand_obj_obs_reset_28400.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_28400_hammer_use_v8_v18.npy'
# export preload_action_target_env_idx=52
# export preload_action_start_frame=249
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True



# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drTruev35_rewv5_franka_ori_grab_s3_mouse_pass_1_nf_300_armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-15-28-40/ts_to_hand_obj_obs_reset_43020.npy'
# <<<<<<< HEAD
# # export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_43020_s3_mouse_pass_1.npy'
# =======
# >>>>>>> 2c7abe03fe89343d6cce65f9a43615473eac897f
# export preload_action_target_env_idx=5
# export preload_action_start_frame=23
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.10
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_watch_set_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-16-14-53/ts_to_hand_obj_obs_reset_117840.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_117840_s1watchset2_reornt_v2.npy'
# export preload_action_target_env_idx=52
# export preload_action_start_frame=249



# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_trcoef0.01_drFalsev34_rewv5_franka__armmult2.0_table0.5_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-15-38-29/ts_to_hand_obj_obs_reset_313680.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_313680_gene_s2.npy'
# export preload_action_target_env_idx=61
# export preload_action_start_frame=242
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# export obj_mass_reset=0.10
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# # 10 200 elephant
# export preload_action_target_env_idx=10
# export preload_action_start_frame=200


# ### reudced control frequency ##
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s6_banana_peel_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-02-07-51/ts_to_hand_obj_obs_reset_9800.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_9800_banana_peel_2_reornt_v2.npy'
# export preload_action_target_env_idx=7
# export preload_action_start_frame=246
# export stiffness_coef=100
# export damping_coef=4
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_flashlight_on_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-11-24-35/ts_to_hand_obj_obs_reset_11820.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_11820_s2flashlighton1_reorient.npy'
# export preload_action_target_env_idx=87
# export preload_action_start_frame=160
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_lightbulb_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-11-28-23/ts_to_hand_obj_obs_reset_45300.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_45300_lightbulb_pass_1_reorient_v22.npy'
# export preload_action_target_env_idx=24
# export preload_action_start_frame=149
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_lightbulb_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-11-28-23/ts_to_hand_obj_obs_reset_28740.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_45300_lightbulb_pass_1_reorient_v322.npy'
# export preload_action_target_env_idx=27
# export preload_action_start_frame=216


# # export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hand_inspect_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-02-13-18/ts_to_hand_obj_obs_reset_36620.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_36620_handinspect_1_reorient.npy'
# export preload_action_target_env_idx=38
# export preload_action_start_frame=70
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_lightbulb_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-01-52-53/ts_to_hand_obj_obs_reset_50980.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_50980_lightbulb_pass_lightermass.npy'
# export preload_action_target_env_idx=92
# export preload_action_start_frame=241
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.0431
# export recompute_inertia=True
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True



# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s1_cup_pour_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-02-21-04/ts_to_hand_obj_obs_reset_24780.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_24780_cuppourreorient.npy'
# export preload_action_target_env_idx=42
# export preload_action_start_frame=190
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export obj_mass_reset=0.0431
# export recompute_inertia=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s8_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-02-30-24/ts_to_hand_obj_obs_reset_29100.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_29100_s8phonecall_oreinet.npy'
# export preload_action_target_env_idx=46
# export preload_action_start_frame=76
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s3_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_15-01-59-20/ts_to_hand_obj_obs_reset_43540.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_43540_phone_reorinet_v2.npy'
# export preload_action_target_env_idx=47
# export preload_action_start_frame=5
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True



# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s3_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_15-02-00-08/ts_to_hand_obj_obs_reset_42300.npy'
# # export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42300_phone_reorient_v1.npy'
# export preload_action_target_env_idx=9
# export preload_action_start_frame=121
# export stiffness_coef=210
# export damping_coef=20



# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_knife_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-10-13-43/ts_to_hand_obj_obs_reset_158300.npy'
# export checkpoint='ckpts/ts_to_hand_obj_obs_reset_158300_knife_pass_reorient_v1.npy'
# export preload_action_target_env_idx=90
# export preload_action_start_frame=169
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export wo_fingertip_vel=False
# export wo_fingertip_rot_vel=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s8_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-02-30-24/ts_to_hand_obj_obs_reset_16920.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_16920_s8_phone_call_reorient.npy'
# export preload_action_target_env_idx=24
# export preload_action_start_frame=95
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s8_toothpaste_squeeze_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-16-25-19/ts_to_hand_obj_obs_reset_59900.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_59900_s8_toothpaste_squeeze_reorint.npy'
# export preload_action_target_env_idx=30
# export preload_action_start_frame=0
# export stiffness_coef=210
# export damping_coef=20


# export preload_action_targets_fn="/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s3_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-10-54-49/ts_to_hand_obj_obs_reset_4580.npy"
# export preload_action_targets_fn="ckpts/ts_to_hand_obj_obs_reset_4580_reorient_phone_v3.npy"
# export preload_action_target_env_idx=37
# export preload_action_start_frame=249
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.0681
# export recompute_inertia=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s3_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-13-09-17/ts_to_hand_obj_obs_reset_5500.npy'
# <<<<<<< HEAD
# export preload_action_target_env_idx=61
# export preload_action_start_frame=32
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s3_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-12-12-09/ts_to_hand_obj_obs_reset_7700.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_7700_phone_reorient_v5.npy'
# export preload_action_target_env_idx=8
# export preload_action_start_frame=137
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.0681
# export recompute_inertia=True
# # =======
# # export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_5500_phone_reorient_v4.npy'
# # export preload_action_target_env_idx=61
# # export preload_action_start_frame=32
# # export stiffness_coef=210
# # export damping_coef=20
# # export reset_obj_mass=True
# # export obj_mass_reset=0.0681
# # export recompute_inertia=True
# >>>>>>> 72dda129a67c054adb37b5c65947e8d5b2874b94

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s3_phone_call_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-10-54-49/ts_to_hand_obj_obs_reset_12620.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_12620_reorient_phone_v7.npy'
# export preload_action_target_env_idx=26
# export preload_action_start_frame=54
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.0681
# export recompute_inertia=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-14-23-51/ts_to_hand_obj_obs_reset_8780.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_8780_banana_eat_reorient_v1.npy'
# export preload_action_target_env_idx=15
# export preload_action_start_frame=88
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.0435
# export recompute_inertia=True


# TO TEST #
# _sample_0.npy
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-14-18-51/ts_to_hand_obj_obs_reset_41860.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_41860_banana_eat_reorient_v11.npy'
# export preload_action_target_env_idx=33
# export preload_action_start_frame=251
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=True
# export obj_mass_reset=0.0435
# export recompute_inertia=True

# _sample_4.npy
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-14-21-55/ts_to_hand_obj_obs_reset_44100.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_44100_banana_eat_reorient_v10.npy'
# export preload_action_target_env_idx=31
# export preload_action_start_frame=42


# _sample_7.npy
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-14-23-51/ts_to_hand_obj_obs_reset_42780.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_42780_banana_eat_reorient_v9.npy'
# export preload_action_target_env_idx=0
# export preload_action_start_frame=110


# sample_7
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-20-18-16/ts_to_hand_obj_obs_reset_17440.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_17440_banana_eat_reorient_v8.npy'
# export preload_action_target_env_idx=0
# export preload_action_start_frame=144


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-02-08/ts_to_hand_obj_obs_reset_21620.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_21620_banana_eat_reorient_v13.npy'
# export preload_action_target_env_idx=26
# export preload_action_start_frame=162


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s9_banana_pass_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-01-44/ts_to_hand_obj_obs_reset_23160.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_23160_banana_eat_reorient_v12.npy'
# export preload_action_target_env_idx=13
# export preload_action_start_frame=186


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-03-09/ts_to_hand_obj_obs_reset_1.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_1_hammer_reorient_sample_6.npy'
# export preload_action_target_env_idx=35
# export preload_action_start_frame=0
# export stiffness_coef=210
# export damping_coef=20
# export reset_obj_mass=False
# export obj_mass_reset=0.0435
# export recompute_inertia=False

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-16-16/ts_to_hand_obj_obs_reset_1.npy'
# export preload_action_target_env_idx=3
# export preload_action_start_frame=0


# ## TO TEST (s6_mug_toast_1) ##
# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s6_mug_toast_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-09-26-09/ts_to_hand_obj_obs_reset_54460.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_54460_s6_mugtoast_1_reorient_v1.npy'
# export preload_action_target_env_idx=34
# export preload_action_start_frame=135


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s6_mug_toast_1_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-09-26-36/ts_to_hand_obj_obs_reset_59540.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_59540_s6_mugtoast_1_reorient_v2.npy'
# export preload_action_target_env_idx=93
# export preload_action_start_frame=169


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-01-16-25/ts_to_hand_obj_obs_reset_47740.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_47740_hammer_reorient_v13.npy'
# export preload_action_target_env_idx=59
# export preload_action_start_frame=3
# export reset_obj_mass=True
# export obj_mass_reset=0.0833
# export recompute_inertia=True


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-01-49-53/ts_to_hand_obj_obs_reset_90800.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_90800_hammer_reorient_v14.npy'
# export preload_action_target_env_idx=9
# export preload_action_start_frame=207
# export reset_obj_mass=False


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_ori_grab_s2_hammer_use_2_nf_300_reornt_hodistFalse_mt_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-12-06-33/ts_to_hand_obj_obs_reset_146040.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_146040_hammer_reorient_v15.npy'
# export preload_action_target_env_idx=73
# export preload_action_start_frame=231
# export reset_obj_mass=False


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-01-52-31/ts_to_hand_obj_obs_reset_120600.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_120600_hammer_reorient_v16.npy'
# export preload_action_target_env_idx=90
# export preload_action_start_frame=2

# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_ori_grab_s2_hammer_use_2_nf_300_reornt_hodistFalse_mt_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-23-28-12/ts_to_hand_obj_obs_reset_1.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_1_hammer_sample_30_vxx.npy'
# export preload_action_target_env_idx=52
# export preload_action_start_frame=0


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-22-46-32/ts_to_hand_obj_obs_reset_64800.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_64800_hammer_reornt_vxx2.npy'
# export preload_action_target_env_idx=34
# export preload_action_start_frame=39


# export preload_action_targets_fn='/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelTrue/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-16-11-56/ts_to_hand_obj_obs_reset_106140.npy'
# export preload_action_targets_fn='ckpts/ts_to_hand_obj_obs_reset_106140_hammer_reoreint_v2urdf_v8.npy'
# export preload_action_target_env_idx=22
# export preload_action_start_frame=141


# /data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_ori_grab_s2_hammer_use_2_nf_300_reornt_hodistFalse_mt_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_16-23-33-41/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth

export record_experiences=False
### record the experience and use more environments for testing ###
# export record_experiences=True
# export numEnvs=5000
# export minibatch_size=5000
### record the experience and use more environments for testing ###


# bash scripts_new/run_tracking_headless_grab_whlsample_wfranka.sh



CUDA_VISIBLE_DEVICES=${cuda_idx} python test_generalist_pool.py --additional_tag=${additional_tag} --hand_type=${hand_type} --numEnvs=${numEnvs} --minibatch_size=${minibatch_size} --goal_cond=${goal_cond}  --test=${test} --use_relative_control=${use_relative_control} --use_kinematics_bias=${use_kinematics_bias} --w_obj_ornt=${w_obj_ornt} --obs_type=${obs_type} --separate_stages=${separate_stages} --rigid_obj_density=${rigid_obj_density}  --kinematics_only=${kinematics_only} --use_fingertips=${use_fingertips}  --use_kinematics_bias_wdelta=${use_kinematics_bias_wdelta} --hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} --hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} --hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} --dt=${dt} --glb_trans_vel_scale=${glb_trans_vel_scale} --glb_rot_vel_scale=${glb_rot_vel_scale} --rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} --rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} ${debug} --nn_gpus=${nn_gpus} --num_frames=${num_frames} --tracking_data_sv_root=${tracking_data_sv_root} --subj_nm=${subj_nm}  --st_idx=${st_idx} --dofSpeedScale=${dofSpeedScale} --use_twostage_rew=${use_twostage_rew} --episodeLength=${episodeLength} --data_inst_flag=${data_inst_flag} --pre_optimized_traj=${pre_optimized_traj} --use_generalist_policy=${use_generalist_policy} --use_hand_actions_rew=${use_hand_actions_rew} --supervised_training=${supervised_training} --checkpoint=${checkpoint} --max_epochs=${max_epochs} --training_mode=${training_mode} --test_inst_tag=${test_inst_tag} --test_optimized_res=${test_optimized_res} --preload_experiences_tf=${preload_experiences_tf} --preload_experiences_path=${preload_experiences_path} --single_instance_training=${single_instance_training} --generalist_tune_all_instnaces=${generalist_tune_all_instnaces} --obj_type_to_pre_optimized_traj=${obj_type_to_pre_optimized_traj} --pre_load_trajectories=${pre_load_trajectories} --sampleds_with_object_code_fn=${sampleds_with_object_code_fn} --log_path=${log_path} --grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} --taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn} --single_instance_tag=${single_instance_tag} --obj_type_to_optimized_res_fn=${obj_type_to_optimized_res_fn} --supervised_loss_coef=${supervised_loss_coef} --pure_supervised_training=${pure_supervised_training} --inst_tag_to_latent_feature_fn=${inst_tag_to_latent_feature_fn} --exclude_inst_tag_to_opt_res_fn=${exclude_inst_tag_to_opt_res_fn} --dataset_type=${dataset_type} --object_type_to_latent_feature_fn=${object_type_to_latent_feature_fn} --tracking_save_info_fn=${tracking_save_info_fn} --taco_interped_data_sv_additional_tag=${taco_interped_data_sv_additional_tag} --use_strict_maxx_nn_ts=${use_strict_maxx_nn_ts} --strict_maxx_nn_ts=${strict_maxx_nn_ts} --use_local_canonical_state=${use_local_canonical_state} --obj_type_to_ckpt_fn=${obj_type_to_ckpt_fn} --use_base_traj=${use_base_traj} --obj_type_to_base_trajs=${obj_type_to_base_trajs} --customize_damping=${customize_damping} --tracking_info_st_ta=${tracking_info_st_tag} --train_on_all_trajs=${train_on_all_trajs} --test_on_taco_test_set=${test_on_taco_test_set} --single_instance_state_based_train=${single_instance_state_based_train} --controlFrequencyInv=${controlFrequencyInv} --downsample=${downsample}  --use_forcasting_model=${use_forcasting_model} --forcasting_model_weights=${forcasting_model_weights} --forcasting_model_n_layers=${forcasting_model_n_layers} --w_glb_traj_feat_cond=${w_glb_traj_feat_cond} --use_history_obs=${use_history_obs} --substeps=${substeps} --use_window_future_selection=${use_window_future_selection} --w_history_window_index=${w_history_window_index} --single_instance_test_tag=${single_instance_test_tag} --w_obj_latent_features=${w_obj_latent_features} --forcasting_inv_freq=${forcasting_inv_freq} --forcasting_history_ws=${forcasting_history_ws} --use_future_obs=${use_future_obs} --randomize_condition_type=${randomize_condition_type} --randomize_conditions=${randomize_conditions} --history_freq=${history_freq} --history_length=${history_length} --forcasting_diffusion_model=${forcasting_diffusion_model} --partial_obj_info=${partial_obj_info} --partial_hand_info=${partial_hand_info} --use_partial_to_complete_model=${use_partial_to_complete_model} --partial_to_complete_model_weights=${partial_to_complete_model_weights} --contact_info_sv_root=${contact_info_sv_root} --add_contact_conditions=${add_contact_conditions} --st_ed_state_cond=${st_ed_state_cond} --preset_cond_type=${preset_cond_type} --preset_inv_cond_freq=${preset_inv_cond_freq} --open_loop_test=${open_loop_test} --single_inst_tag=${single_inst_tag} --use_future_ref_as_obs_goal=${use_future_ref_as_obs_goal} --w_franka=${w_franka} --add_table=${add_table} --table_z_dim=${table_z_dim}  --gpu_offset_idx=${gpu_offset_idx} --headless=${headless} --maxx_inst_nn=${maxx_inst_nn}  --load_kine_info_retar_with_arm=${load_kine_info_retar_with_arm} --kine_info_with_arm_sv_root=${kine_info_with_arm_sv_root} --w_finger_pos_rew=${w_finger_pos_rew} --franka_delta_delta_mult_coef=${franka_delta_delta_mult_coef} --control_arm_via_ik=${control_arm_via_ik} --warm_trans_actions_mult_coef=${warm_trans_actions_mult_coef} --warm_rot_actions_mult_coef=${warm_rot_actions_mult_coef} --wo_vel_obs=${wo_vel_obs} --not_use_kine_bias=${not_use_kine_bias} --disable_hand_obj_contact=${disable_hand_obj_contact} --closed_loop_to_real=${closed_loop_to_real} --hand_glb_mult_factor_scaling_coef=${hand_glb_mult_factor_scaling_coef} --hand_glb_mult_scaling_progress_after=${hand_glb_mult_scaling_progress_after} --wo_fingertip_rot_vel=${wo_fingertip_rot_vel} --wo_fingertip_vel=${wo_fingertip_vel} --include_obj_rot_in_obs=${include_obj_rot_in_obs}  --arm_stiffness=${arm_stiffness} --arm_effort=${arm_effort} --arm_damping=${arm_damping} --estimate_vels=${estimate_vels} --use_v2_leap_warm_urdf=${use_v2_leap_warm_urdf} --wo_fingertip_pos=${wo_fingertip_pos} --reset_obj_mass=${reset_obj_mass} --obj_mass_reset=${obj_mass_reset} --recompute_inertia=${recompute_inertia} --add_physical_params_in_obs=${add_physical_params_in_obs} --preload_action_targets_fn=${preload_action_targets_fn} --preload_action_target_env_idx=${preload_action_target_env_idx} --preload_action_start_frame=${preload_action_start_frame} --record_experiences=${record_experiences} --stiffness_coef=${stiffness_coef} --damping_coef=${damping_coef} --kine_ed_tag=${kine_ed_tag} --use_actual_traj_length=${use_actual_traj_length} --add_global_movements=${add_global_movements} --add_global_movements_af_step=${add_global_movements_af_step}
  

# --mocap_sv_info_fn=${mocap_sv_info_fn}


# export train_on_all_trajs=True
# export test_on_taco_test_set=False

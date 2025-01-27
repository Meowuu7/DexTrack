

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

### TOTEST ###
# get the object name #
# key: ('ori_grab_s2_airplane_fly_1', 'ori_grab_s2_airplane_fly_1'), cur_val: ['/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-00-01-18/ts_to_hand_obj_obs_reset_1.npy']

export object_name='ori_grab_s2_airplane_fly_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-17-38-45/nn/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-23-46-56/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-18-56-44/nn/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-00-12-20/ts_to_hand_obj_obs_reset_1.npy'



export object_name='ori_grab_s8_banana_eat_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'
export checkpoint='runs/tracking_ori_grab_s8_banana_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-03-51-53/nn/tracking_ori_grab_s8_banana_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'


####### test settings for the banana peel ####### banana peel #######
####### test settings #######
export object_name='ori_grab_s8_banana_peel_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'
# export checkpoint='runs/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-14-28-40/nn/last_tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_350_rew_16.446312.pth'
export checkpoint='runs/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-14-28-40/nn/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export pre_optimized_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'
export checkpoint='runs/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-16-31-57/nn/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-16-31-57/nn/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s8_apple_lift'
# export object_name='ori_grab_s8_apple_eat_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
# export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__23-18-20-08/ts_to_hand_obj_obs_reset_1.npy'
export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__23-18-20-08/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-19-18-32/nn/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s8_apple_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-19-52-41/nn/tracking_ori_grab_s8_apple_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export checkpoint='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-19-18-32/nn/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-20-16-41/nn/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export checkpoint='runs/tracking_ori_grab_s8_apple_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-19-52-41/nn/tracking_ori_grab_s8_apple_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

## load the pre optimized traj ##
export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_eat_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__23-20-35-13/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-20-38-25/nn/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s8_toothpaste_pass_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='runs/tracking_ori_grab_s8_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-14-23/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s8_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-18-48-09/nn/tracking_ori_grab_s8_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'


export object_name='ori_grab_s8_mouse_lift'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='runs/tracking_ori_grab_s8_mouse_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-18-13-31/ts_to_hand_obj_obs_reset_2.npy'
export checkpoint='runs/tracking_ori_grab_s8_mouse_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-18-59-05/nn/tracking_ori_grab_s8_mouse_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'


export object_name='ori_grab_s9_toothpaste_pass_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export checkpoint='runs/tracking_ori_grab_s9_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-20-05-58/nn/last_tracking_ori_grab_s9_toothpaste_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_400_rew_18.06042.pth'


export object_name='ori_grab_s10_hand_inspect_1'
export object_name='ori_grab_s2_apple_lift'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s7_hand_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-14-09-43/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s10_hand_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-21-33-19/nn/tracking_ori_grab_s10_hand_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export pre_optimized_traj='runs/tracking_ori_grab_s10_hand_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__24-23-50-29/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s10_hand_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-00-22-44/nn/tracking_ori_grab_s10_hand_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export pre_optimized_traj='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-15-50-22/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-16-30-58/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s8_duck_inspect_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='runs/tracking_ori_grab_s8_duck_inspect_1_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-19-00-27/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s8_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-19-59-26/nn/tracking_ori_grab_s8_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s8_apple_lift'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__25-18-33-30/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='runs/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-19-57-33/nn/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s9_airplane_fly_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples/tracking_ori_grab_s9_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_slicing_hist_pred_deter_26-12-47-18/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_reopt/tracking_ori_grab_s9_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-01-15-32/nn/tracking_ori_grab_s9_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s9_pyramidmedium_inspect_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples/tracking_ori_grab_s9_pyramidlarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_slicing_hist_pred_deter_26-12-18-08/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_reopt/tracking_ori_grab_s9_pyramidmedium_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_26-23-44-37/nn/tracking_ori_grab_s9_pyramidmedium_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='ori_grab_s2_eyeglasses_clean_1'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples/tracking_ori_grab_s2_eyeglasses_clean_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_slicing_hist_pred_deter_27-07-32-23/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_reopt/tracking_ori_grab_s2_eyeglasses_clean_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-11-44-06/nn/tracking_ori_grab_s2_eyeglasses_clean_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'


export object_name='ori_grab_s8_watch_lift'
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export pre_optimized_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_slicing_hist_pred_deter_26-01-39-53/ts_to_hand_obj_obs_reset_1.npy'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_reopt/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-00-18-17/nn/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
####### test settings #######



export hand_type='allegro'


####### experiments for leap ########
# export hand_type='leap'
# export pre_optimized_traj=''
# # export object_name='ori_grab_s2_apple_lift'
# export object_name='ori_grab_s2_apple_pass_1'
# export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data/leap_passive_active_info_${object_name}.npy
# # export tracking_data_sv_root="/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data"
# export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_apple_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_26-15-00-27/nn/tracking_ori_grab_s2_apple_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_pass_1_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-16-24-46/nn/last_tracking_ori_grab_s2_apple_pass_1_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_950_rew_-15.459991.pth'
# export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_apple_pass_1_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-16-24-46/nn/tracking_ori_grab_s2_apple_pass_1_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'


####### experiments for leap ########



export test=True # runs / track the ori grab s2 banana peel 1 # # runs / track # #
# ####### test setting ########



export train_name=tracking_${object_name}_obs_${obs_type}_hand_${hand_type}_density_${rigid_obj_density}_trans_${glb_trans_vel_scale}_rot_${glb_rot_vel_scale}_goalcond_${goal_cond}_${additiona_tag}
export full_experiment_name=${train_name}


# /cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data/passive_active_info_ori_grab_s1_airplane_offhand_1.npy
export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data'
export tracking_info_st_tag='passive_active_info_'


# model in the joint space #
# model in the point cloud space #
# both of them are non-trivial to train and learn? #
# both of them are non-trivial to train and learn? #

# export test=False
# export checkpoint='' # 

# given the vision input we should distill them into the #
# control sequences # #
# but not a policy ? # # 
# then you get a succesul policy #
# you do not have the multi-task polcy #
# so the only thing is distill each single policy # 
# from the state input to the vision input #
# then a dagger is needed #
# another thing is distill each single policy to a general and unified policy #


export tag=tracking_${object_name}
export cuda_idx=2


export task_type="AllegroHandTracking"
export train_type="HumanoidPPO"


# runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-14-08-19/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-13-58-46/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-14-08-19/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-19-38-44/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-19-38-44/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-19-39-13/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 64.13 # obj reward #
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-21-05-45/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-20-34-27/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-06-35/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-40-52/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-00-32-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-00-32-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-00-50-56/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # hand and obj rew, 1000
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-22-17-35/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-02-33-27/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # hand and obj rew, 200 # av reward: -7.11323828125 av steps: 149.0
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-03-45-07/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # obj rew, 200 # av reward: -7.11323828125 av steps: 149.0
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-03-33-32/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # supervised training with obj rew
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-03-11-08/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'  # supervised training without obj rew
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-11-29-21/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-12-34-01/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-12-34-01/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 2 instances, sup training, ver2 #
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-13-30-52/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 2 instances, sup training, ver1 #
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-13-29-28/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 200 instances, sup training, ver2 
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-15-37-26/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-14-49-40/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-17-16-22/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-17-16-22/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # largenet #
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-19-16-44/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' #  re-trained with weight 0.1
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-19-46-58/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-18-36-16/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # large net v3 # 
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-21-36-30/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_450_rew_-9.710324.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-22-17-35/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-20-51-48/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_02-02-23-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth' # 
export pre_optimized_traj='' # 
export task_type="AllegroHandTrackingGeneralist"
export train_type="HumanoidPPO"


export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy"
# export grab_inst_tag_to_optimized_res_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
export grab_inst_tag_to_optimized_res_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy' # grab inst jto optimized 
export test_subj_nm='s2'
export grab_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
export test_subj_nm='s9'
# export test_subj_nm='s2'
# export test_subj_nm='s1'
export test_subj_nm=''

export taco_inst_tag_to_optimized_res_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"




####### Multiple instance test setting ###### Generalist ######
# export grab_inst_tag_to_optimized_res_fn='/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy'
############# ========= Checkpoints trained on dataset version 1 ========= #############
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-17-28-12/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-17-28-12/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1300_rew_-7.6574783.pth'
############# ========= Checkpoints trained on dataset version 1 ========= #############

############# ========= Checkpoints trained on dataset version 2 ========= #############
export checkpoint='./runs_generalist_x/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-12-02-47/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_1200_rew_-7.647374.pth'
export checkpoint='./runs_generalist_x/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-12-02-47/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_8000_rew_-17.171999.pth'
# export checkpoint='./runs_generalist_x/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-10-51-50/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_8000_rew__-14.02_.pth'
export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-15-25-30/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_300_rew_12.338114.pth'
export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-15-25-30/nn/last_tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_650_rew_20.20599.pth'
export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-17-59-42/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
############# ========= Checkpoints trained on dataset version 2 ========= #############

### use the checkpoint from the training ###
export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-03-51-57/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-17-59-42/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export single_instance_state_based_test=False
export record_experiences=False
# export test_subj_nm=''
####### Multiple instance test setting ###### Generalist ######



####### Multiple instance test settings ######## --- generalist --- ########
export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-03-29-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs_v2/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-03-29-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs_v3/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-03-29-55/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_v3.pth'
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/inst_tag_to_obj_feat.npy'

export taco_inst_tag_to_optimized_res_fn=''




##### LEAP hand setting ######
# export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy"
# export numEnvs=1000
# export minibatch_size=1000
# export maxx_inst_nn=2
# export maxx_inst_nn=0
# export grab_obj_type_to_opt_res_fn=''
# export taco_obj_type_to_opt_res_fn=''
# export hand_type='leap'
# # export supervised_training=False
# # export supervised_loss_coef=0.0000
# # tracking_save_info_fn, tracking_info_st_tag #
# # /cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data/leap_passive_active_info_ori_grab_s1_alarmclock_pass_1.npy
# export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data'
# export tracking_info_st_tag='leap_passive_active_info_'
# export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/inst_tag_to_obj_feat.npy'
# export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-19-57-43/nn/tracking_ori_grab_s2_apple_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export checkpoint='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_35_runs/tracking_ori_grab_s2_apple_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-19-57-43/nn/tracking_ori_grab_s2_apple_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
##### LEAP hand setting ######




# ### get the checpoint and the inst tag to latent feature fn ###
# export checkpoint='./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-17-59-42/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# export inst_tag_to_latent_feature_fn=''
# ### get the checpoint and the inst tag to latent feature fn ###


export object_type_to_latent_feature_fn="/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/inst_tag_to_obj_feat.npy'

##### single instance settings ##### # single instance and the test #
export test_inst_tag='ori_grab_s1_camera_takepicture_1'
export test_inst_tag='ori_grab_s1_headphones_pass_1'
export test_inst_tag='ori_grab_s1_stamp_offhand_1'
export test_inst_tag='ori_grab_s1_alarmclock_pass_1'
export test_inst_tag='ori_grab_s1_headphones_use_1'
export test_inst_tag='ori_grab_s1_spheremedium_offhand_1'
export test_inst_tag='ori_grab_s1_teapot_pass_1'
export test_inst_tag='ori_grab_s1_teapot_pour_1'
export test_inst_tag='ori_grab_s1_toruslarge_inspect_1'
export test_inst_tag='ori_grab_s1_torusmedium_pass_1'
# export test_optimized_res='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s9_waterbottle_pour_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-23-35-57/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
export test_optimized_res='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s1_camera_takepicture_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-21-16-18/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export sampleds_with_object_code_fn="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__03-22-56-23/ts_to_hand_obj_obs_reset_1.npy"
export sampleds_with_object_code_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_canonv2_taskcond_samples_/samples_ep_0_batch_0.npy'
export sampleds_with_object_code_fn='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_vnewv2_canonv2_taskcond_samples_/samples_ep_0_batch_0.npy'
# export checkpoint='runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_04-18-20-12/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='./runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_camera_takepicture_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-12-21-09/nn/tracking_ori_grab_s1_camera_takepicture_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_headphones_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-19-18-14/nn/tracking_ori_grab_s1_headphones_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_stamp_offhand_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-01-28-37/nn/tracking_ori_grab_s1_stamp_offhand_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_alarmclock_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-19-18-14/nn/tracking_ori_grab_s1_alarmclock_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_headphones_use_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-04-52-11/nn/tracking_ori_grab_s1_headphones_use_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_spheremedium_offhand_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-19-18-14/nn/tracking_ori_grab_s1_spheremedium_offhand_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_teapot_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-04-52-12/nn/tracking_ori_grab_s1_teapot_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_teapot_pour_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-22-27-52/nn/tracking_ori_grab_s1_teapot_pour_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_toruslarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-22-27-52/nn/tracking_ori_grab_s1_toruslarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_init_fr_traj_translations_filter/tracking_ori_grab_s1_torusmedium_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-04-52-11/nn/tracking_ori_grab_s1_torusmedium_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export checkpoint=''
##### single instance settings #####
# export replay_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/tracking_ori_grab_s1_torusmedium_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-19-45-23/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
export replay_fn='../assets/optimized_res/tracking_ori_grab_s1_torusmedium_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-19-45-23/ts_to_hand_obj_obs_reset_1_sorted_best.npy'



export test_inst_tag='taco_20231020_036'
export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231020_036_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-50-08/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231020_036_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-50-08/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231020_036_v2_interpfr_60_interpfr2_60_nntrans_40.npy'



# export test_inst_tag='taco_20231027_008'
# export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231027_008_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231027_008_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231027_008_v2_interpfr_60_interpfr2_60_nntrans_40.npy'


# export test_inst_tag='taco_20231104_035'
# export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231104_035_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231104_035_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231104_035_v2_interpfr_60_interpfr2_60_nntrans_40.npy'


export test_inst_tag='taco_20231024_176'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_176_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-03-58-21/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_176_v2_interpfr_60_interpfr2_60_nntrans_40.npy'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_176_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-03-58-21/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_176_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-30-18/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy


# export test_inst_tag='taco_20231024_169'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_169_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-04-07-14/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_169_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-04-07-14/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_169_v2_interpfr_60_interpfr2_60_nntrans_40.npy'
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_169_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-10-20/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy



# export test_inst_tag='taco_20231024_124'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_124_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-04-02-46/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_124_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-04-02-46/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_124_v2_interpfr_60_interpfr2_60_nntrans_40.npy'
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_124_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-02-07-21/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy


# export test_inst_tag='taco_20231024_070'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_070_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-03-58-22/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_070_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-03-58-22/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_070_v2_interpfr_60_interpfr2_60_nntrans_40.npy'
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_070_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-01-52-31/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy


# export test_inst_tag='taco_20231024_045'
# export replay_fn='../assets/optimized_res/tracking_taco_20231024_045_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-03-58-22/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_045_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-03-58-22/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_045_v2_interpfr_60_interpfr2_60_nntrans_40.npy
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_045_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-56-34/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_007'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_007_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-52-23/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_007_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-16-02/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
export replay_fn=../assets/optimized_res/tracking_taco_20231024_007_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-52-23/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_007_v2_interpfr_60_interpfr2_60_nntrans_40.npy

# export test_inst_tag='taco_20231024_010'
# export replay_fn=../assets/optimized_res/tracking_taco_20231024_010_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-36-53/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy


export test_inst_tag='taco_20231024_019'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_019_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-14-43-33/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_043'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_043_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-06-49/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_056'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_056_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-14-53-50/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_061'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_061_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-31-45/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
export replay_fn=../assets/optimized_res/tracking_taco_20231024_061_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-17-29/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
export replay_fn='../assets/optimized_res/tracking_taco_20231024_061_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-17-29/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231024_061_v2_interpfr_60_interpfr2_60_nntrans_40.npy'

export test_inst_tag='taco_20231024_076'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_076_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-21-29/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_081'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_081_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-10-24/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy


export test_inst_tag='taco_20231024_087'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_087_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-18-30/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'

export test_inst_tag='taco_20231024_137'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_137_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-22-24/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'

export test_inst_tag='taco_20231024_191'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_191_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-22-24/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_194'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_194_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-21-29/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'

export test_inst_tag='taco_20231024_196'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_196_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-14-43-33/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'

export test_inst_tag='taco_20231024_197'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_197_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-14-39-46/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy


export test_inst_tag='taco_20231024_236'
export replay_fn=../assets/optimized_res/tracking_taco_20231024_236_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-15-18-30/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231024_300'
export replay_fn='../assets/optimized_res/tracking_taco_20231024_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-14-48-17/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy'


export test_inst_tag='taco_20231104_035'
export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231104_035_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
export replay_fn='../assets/optimized_res/tracking_TACO_taco_20231104_035_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231104_035_v2_interpfr_60_interpfr2_60_nntrans_40.npy'
export replay_fn=../assets/optimized_res/tracking_taco_20231104_035_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-32-07/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
export replay_fn=../assets/optimized_res/tracking_taco_20231104_035_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-32-07/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

# export test_inst_tag='ori_grab_s1_banana_offhand_1_nf_300'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_banana_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-29-18/ts_to_hand_obj_obs_reset_1_sorted_best.npy


# export test_inst_tag='ori_grab_s1_duck_offhand_2_nf_300'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_duck_offhand_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-07-10/ts_to_hand_obj_obs_reset_1_sorted_best.npy
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_duck_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-09/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# export test_inst_tag='ori_grab_s1_flute_pass_1_nf_300'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flute_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-55-09/ts_to_hand_obj_obs_reset_1_sorted_best.npy


# export test_inst_tag='ori_grab_s1_flute_play_1_nf_300'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flute_play_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-43-25/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='ori_grab_s1_hammer_use_1_nf_300'
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hammer_use_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-09/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hammer_use_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-51-06/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hammer_use_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-09/passive_active_info_ori_grab_s1_hammer_use_1_nf_300.npy


# export test_inst_tag='ori_grab_s1_hammer_use_2_nf_300'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hammer_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-01-17/ts_to_hand_obj_obs_reset_1_sorted_best.npy


# export test_inst_tag='ori_grab_s1_hammer_use_3_nf_300'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_hammer_use_3_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-09/passive_active_info_ori_grab_s1_hammer_use_3_nf_300.npy'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hammer_use_3_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-09/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# export test_inst_tag='ori_grab_s1_hand_pass_1_nf_300'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_hand_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-01-17/ts_to_hand_obj_obs_reset_1_sorted_best.npy'


# # do not use the cmp
# export test_inst_tag='ori_grab_s1_phone_call_1_nf_300'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_phone_call_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-24-43/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# # export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_phone_call_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-43-24/ts_to_hand_obj_obs_reset_1_sorted_best.npy
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_phone_call_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-43-24/passive_active_info_ori_grab_s1_phone_call_1_nf_300.npy

# export test_inst_tag='ori_grab_s1_pyramidmedium_lift_nf_300'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_pyramidmedium_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-01-17/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_pyramidmedium_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-08/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

# export test_inst_tag='ori_grab_s1_pyramidsmall_pass_1_nf_300'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_pyramidsmall_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-33-09/ts_to_hand_obj_obs_reset_1_sorted_best.npy'


# no cmp
# export test_inst_tag='ori_grab_s1_stapler_pass_1_nf_300'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_stapler_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-24-43/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_stapler_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-24-43/passive_active_info_ori_grab_s1_stapler_pass_1_nf_300.npy'
# export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_stapler_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-11-57/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

# export test_inst_tag='ori_grab_s1_watch_lift_nf_300'
# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_watch_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-09-58/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_watch_set_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-35-23/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_waterbottle_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-09-57/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_waterbottle_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-22-58/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# export test_inst_tag='ori_grab_s1_banana_pass_1'
# export replay_fn='../assets/optimized_res/tracking_ori_grab_s1_banana_pass_1_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_15-00-17-04/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export test_inst_tag='ori_grab_s1_cubesmall_pass_1'
# export replay_fn='../assets/optimized_res/tracking_ori_grab_s1_cubesmall_pass_1_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_15-00-22-18/ts_to_hand_obj_obs_reset_1_sorted_best.npy'
# export hand_type='leap'


export test_inst_tag='ori_grab_s1_cylindersmall_pass_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_cylindersmall_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-07-44-29/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_cylindersmall_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-58-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='ori_grab_s1_flute_play_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flute_play_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-09-03-55/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flute_play_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-09-03-55/passive_active_info_ori_grab_s1_flute_play_1_nf_300.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flute_play_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-43-25/ts_to_hand_obj_obs_reset_1_sorted_best.npy


# no cmp
export test_inst_tag='ori_grab_s1_toothpaste_squeeze_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_toothpaste_squeeze_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-20-48-28/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_toothpaste_squeeze_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-20-48-28/passive_active_info_ori_grab_s1_toothpaste_squeeze_1_nf_300.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_toothpaste_squeeze_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-42-53/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='ori_grab_s1_doorknob_use_2_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_doorknob_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-14-03-48/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='ori_grab_s1_headphones_offhand_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_headphones_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-19-36-05/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_headphones_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-19-36-05/passive_active_info_ori_grab_s1_headphones_offhand_1_nf_300.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_headphones_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-41-24/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='ori_grab_s1_flashlight_lift_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flashlight_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-06-20-28/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='ori_grab_s1_cubesmall_inspect_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_cubesmall_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-02-42-28/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_cubesmall_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-02-42-28/passive_active_info_ori_grab_s1_cubesmall_inspect_1_nf_300.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_cubesmall_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-21-18/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_cubesmall_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-02-33/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='ori_grab_s1_flashlight_on_2_nf_300'
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flashlight_on_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-46-02/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flashlight_on_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-03-36-51/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flashlight_on_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-03-36-51/passive_active_info_ori_grab_s1_flashlight_on_2_nf_300.npy

export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flashlight_on_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-22-47-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flashlight_on_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-42-53/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flashlight_on_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-42-53/passive_active_info_ori_grab_s1_flashlight_on_1_nf_300.npy


export test_inst_tag='ori_grab_s1_hand_inspect_1_nf_300'
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hand_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-23-29-15/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hand_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-25-38/ts_to_hand_obj_obs_reset_1_sorted_best.npy
# export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_hand_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-15-52-50/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_hand_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-15-52-50/passive_active_info_ori_grab_s1_hand_pass_1_nf_300.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_hand_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-01-17/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='ori_grab_s1_spheremedium_lift_nf_300'
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_spheremedium_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-01-12/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='ori_grab_s1_flute_lift_nf_300'
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_flute_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-27-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flute_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-18-36-27/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_flute_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-18-36-27/passive_active_info_ori_grab_s1_flute_lift_nf_300.npy


export test_inst_tag='ori_grab_s1_airplane_fly_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_airplane_fly_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-09-17-21/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='ori_grab_s1_spheresmall_pass_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_spheresmall_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-06-47-42/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_spheresmall_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-06-47-42/passive_active_info_ori_grab_s1_spheresmall_pass_1_nf_300.npy
export replay_fn='../assets/optimized_res_grab/tracking_ori_grab_s1_spheresmall_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-27-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy'


export test_inst_tag='ori_grab_s1_spheremedium_inspect_1_nf_300'
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_spheremedium_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-22-49-42/ts_to_hand_obj_obs_reset_1_sorted_best.npy
# export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_spheremedium_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-13-36-34/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_spheremedium_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-13-36-34/passive_active_info_ori_grab_s1_spheremedium_inspect_1_nf_300.npy


export test_inst_tag='ori_grab_s1_eyeglasses_offhand_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_eyeglasses_offhand_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_29-13-39-57/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='ori_grab_s1_apple_pass_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_apple_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-20-53-01/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_apple_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-51-33/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_grab/tracking_ori_grab_s1_apple_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-00-51-33/passive_active_info_ori_grab_s1_apple_pass_1_nf_300.npy


export test_inst_tag='ori_grab_s1_eyeglasses_wear_1_nf_300'
export replay_fn=../assets/optimized_res_grab_v2/tracking_ori_grab_s1_eyeglasses_wear_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-08-22-35/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20230919_021'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230919_021_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-23-22-52/ts_to_hand_obj_obs_reset_1_sorted_best.npy

# inst tag #
export test_inst_tag='taco_20231105_009'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231105_009_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-09-14-26/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231105_009_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-09-14-26/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231105_009_v2_interpfr_60_interpfr2_60_nntrans_40.npy


export test_inst_tag='taco_20231104_010'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_010_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-00-55-16/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_010_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-00-55-16/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231104_010_v2_interpfr_60_interpfr2_60_nntrans_40.npy

export test_inst_tag='taco_20231104_040'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_040_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-17-53-50/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231104_069'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_069_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-08-40-16/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='taco_20231104_073'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_073_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-13-32-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231104_112'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_112_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-16-08-54/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export test_inst_tag='taco_20231104_141'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_141_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-14-35-12/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231104_176'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_176_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-19-35-47/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231104_205'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_205_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-21-07-13/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_206_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-15-39-17/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231104_210'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_210_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-20-38-08/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_211_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-10-11/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_212_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-08-06-40/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_214_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-07-32-41/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_215_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-04-44-14/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231104_216_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-23-22-52/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20230928_029'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_029_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-16-47-59/ts_to_hand_obj_obs_reset_1_sorted_best.npy


export test_inst_tag='taco_20230928_032'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_032_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-17-21-33/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20230928_034'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_034_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-18-24-28/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export test_inst_tag='taco_20230928_035'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_035_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-16-16-09/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export test_inst_tag='taco_20230928_036'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_036_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-02-26-59/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20230928_037'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-17-15-44/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20230928_042'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_042_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-16-46-20/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20230928_043'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_043_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-18-56-25/ts_to_hand_obj_obs_reset_1_sorted_best.npy
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_043_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-18-56-25/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_043_v2_interpfr_60_interpfr2_60_nntrans_40.npy

export test_inst_tag='taco_20230928_044'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_044_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-03-33-41/ts_to_hand_obj_obs_reset_1_sorted_best.npy
# export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230928_044_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-03-33-41/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_044_v2_interpfr_60_interpfr2_60_nntrans_40.npy


export test_inst_tag='taco_20230930_003'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230930_003_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-17-15-44/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20230930_035' # good 
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230930_035_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-18-24-09/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20230930_057'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20230930_057_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-18-12-52/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231005_017'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231005_017_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-16-21-44/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231005_112'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231005_112_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-17-21-33/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231005_124'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231005_124_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-16-27-31/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
export test_inst_tag='taco_20231005_126'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231005_126_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-12-27-04/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231015_240'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231015_240_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-17-38-54/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231015_240_v2_interpfr_60_interpfr2_60_nntrans_40.npy
# export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231015_240_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-17-38-54/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231015_243'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231015_243_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-16-27-31/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy

export test_inst_tag='taco_20231015_248'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231015_248_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-07-32-41/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export test_inst_tag='taco_20231015_252'
export replay_fn='../assets/optimized_res_taco_v2/tracking_TACO_taco_20231015_252_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_03-21-37-41/ts_to_hand_obj_obs_reset_1_sorted_best.npy'

export test_inst_tag='taco_20231015_266'
export replay_fn=../assets/optimized_res_taco_v2/tracking_TACO_taco_20231015_266_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-05-51-21/ts_to_hand_obj_obs_reset_1_sorted_best.npy

export w_franka=False


export w_franka=True

# replay fn #


export cuda_idx=6

export use_hand_actions_rew=False
export supervised_training=False



export numEnvs=64
export minibatch_size=64


export task_type="AllegroHandTrackingPlay"
export train_type="HumanoidPPO"


# ./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-03-51-57/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth



# bash scripts/run_tracking_headless_grab_play.sh



# CUDA_VISIBLE_DEVICES=${cuda_idx}

python train.py task=${task_type} train=${train_type} sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=True headless=False   task.env.numEnvs=${numEnvs} train.params.config.minibatch_size=${minibatch_size}  task.env.useRelativeControl=${use_relative_control}  train.params.config.max_epochs=10000 task.env.mocap_sv_info_fn=${mocap_sv_info_fn} checkpoint=${checkpoint} task.env.goal_cond=${goal_cond} task.env.object_name=${object_name} tag=${tag} train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name} task.sim.dt=${dt} test=${test} task.env.use_kinematics_bias=${use_kinematics_bias} task.env.w_obj_ornt=${w_obj_ornt} task.env.observationType=${obs_type} task.env.use_canonical_state=${use_canonical_state} task.env.separate_stages=${separate_stages} task.env.rigid_obj_density=${rigid_obj_density} task.env.use_unified_canonical_state=${use_unified_canonical_state} task.env.kinematics_only=${kinematics_only}  task.env.use_fingertips=${use_fingertips}  task.env.glb_trans_vel_scale=${glb_trans_vel_scale} task.env.glb_rot_vel_scale=${glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta=${use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} task.env.dofSpeedScale=${dofSpeedScale} task.env.pre_optimized_traj=${pre_optimized_traj} task.env.hand_type=${hand_type} task.env.use_hand_actions_rew=${use_hand_actions_rew} task.env.supervised_training=${supervised_training} task.env.grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} task.env.test_subj_nm=${test_subj_nm}  task.env.test_inst_tag=${test_inst_tag} task.env.test_optimized_res=${test_optimized_res} task.env.single_instance_state_based_test=${single_instance_state_based_test} task.env.sampleds_with_object_code_fn=${sampleds_with_object_code_fn} task.env.inst_tag_to_latent_feature_fn=${inst_tag_to_latent_feature_fn}  task.env.taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn} task.env.maxx_inst_nn=${maxx_inst_nn} task.env.tracking_save_info_fn=${tracking_save_info_fn} task.env.tracking_info_st_tag=${tracking_info_st_tag} task.env.object_type_to_latent_feature_fn=${object_type_to_latent_feature_fn} task.env.replay_fn=${replay_fn} task.env.w_franka=${w_franka}

# --inst_tag_to_latent_feature_fn=${inst_tag_to_latent_feature_fn}

# train.params.config.record_experiences=${record_experiences} 


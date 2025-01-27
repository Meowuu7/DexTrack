

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

# # 
# # /cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_pass_1.npy

# export object_name='ori_grab_s2_phone_pass_1'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_pass_1.npy'

export use_kinematics_bias=False


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
# export obs_type='pure_state_wref' # pure state w_ref #
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
export mocap_sv_info_fn=/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_${object_name}.npy
export checkpoint='runs/tracking_ori_grab_s2_cylindermedium_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-18-23-11/nn/tracking_ori_grab_s2_cylindermedium_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'


export use_twostage_rew=False
export disable_gravity=False
export ground_distance=-0.0


#### TACO ####
export object_name='taco_20231104_161'
export object_name='taco_20231104_206'
export object_name='taco_20231104_203'
# export object_name='taco_20231104_016'

export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}.npy
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-03-14-26/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'

export object_name='taco_20231104_016'
export object_name='taco_20231104_203'
# export object_name='taco_20231104_125'
export checkpoint='runs/tracking_taco_20231104_016_obs_pure_state_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-15-29-52/nn/last_tracking_taco_20231104_016_obs_pure_state_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2000_rew__42.09_.pth'
# /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-03-14-26/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
export checkpoint='runs/tracking_taco_20231104_125_obs_pure_state_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-15-33-06/nn/last_tracking_taco_20231104_125_obs_pure_state_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2000_rew__-42.38_.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-15-22-23/nn/last_tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_2000_rew__-36.16_.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_7.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-20-27-18/nn/last_tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_7.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3950_rew_-37.3057.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_7.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-20-31-31/nn/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_7.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_3.0_rot_10.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-00-37-36/nn/last_tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_3.0_rot_10.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_3050_rew_-38.47241.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_7.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-10-04-49/nn/tracking_taco_20231104_203_obs_pure_state_density_500.0_trans_7.0_rot_1.0_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# two stage rewards 

export use_twostage_rew=True
export use_canonical_state=True
export episodeLength=1000
export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.0
# export glb_trans_vel_scale=7
# export glb_rot_vel_scale=1
export glb_trans_vel_scale=7
export glb_rot_vel_scale=10
export use_kinematics_bias_wdelta=False
export use_kinematics_bias=False
export use_relative_control=True
export obs_type='pure_state'
export right_hand_dist_thres=0.2 
export use_real_twostage_rew=False


export glb_trans_vel_scale=7
export glb_rot_vel_scale=1
export use_real_twostage_rew=True
# export use_relative_control=False

export checkpoint="runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-19-46-29/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
export controlFrequencyInv=1
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-20-22-19/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-20-22-19/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_t7r0.5f20_rfd_0.5_rh_0.0_14-09-40-40/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_t7r0.5f20_rfd_0.5_rh_0.0.pth'
# export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-20-22-19/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
# in the 
export controlFrequencyInv=4
export start_grasping_fr=True
export glb_trans_vel_scale=0.5
export glb_rot_vel_scale=0.5
export dofSpeedScale=20
export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=0.5
# export rew_finger_obj_dist_coef=0.3
# # export rew_delta_hand_pose_coef=0.5
# export rew_delta_hand_pose_coef=0.2
export rew_finger_obj_dist_coef=0.1
export rew_delta_hand_pose_coef=0.5
export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.0
export use_real_twostage_rew=False
export use_twostage_rew=False
export use_kinematics_bias_wdelta=True
export use_canonical_state=False
export obs_type="pure_state_wref_wdelta"
export use_canonical_state=True
export separate_stages=True
export glb_trans_vel_scale=7 # 2
export glb_rot_vel_scale=0.5
export use_interpolated_data=False
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_t7r0.5f20_rfd_0.5_rh_0.0_14-09-40-40/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_t7r0.5f20_rfd_0.5_rh_0.0.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_t7r0.5f20_rfd_0.5_rh_0.0_14-09-40-40/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_t7r0.5f20_rfd_0.5_rh_0.0.pth'



##### Using the interpolated data #####
export use_interpolated_data=True
export separate_stages=False
export controlFrequencyInv=1
export use_canonical_state=False
export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=0.5
export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.0
export glb_trans_vel_scale=0.5
export glb_rot_vel_scale=0.5
export dofSpeedScale=20
export start_grasping_fr=False
export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}_modifed_interped.npy
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_t0.5r0.5f20_rfd_0.3_rh_0.5_14-16-58-42/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_interpeddata_t7r0.5f20_rfd_0.5_rh_0.0_14-18-26-48/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_interpeddata_t7r0.5f20_rfd_0.5_rh_0.0.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_interpeddata_t7r0.5f20_rfd_0.5_rh_0.0_14-18-26-48/nn/last_tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_7.0_rot_0.5_goalcond_False_kinebias_interpeddata_t7r0.5f20_rfd_0.5_rh_0.0_ep_3450_rew_-26.32089.pth'
export episodeLength=1000

export goal_dist_thres=0.0
export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=0.5
export rew_obj_pose_coef=1.0
export start_frame=60
export additional_tag=kinebias_interpeddata_stfr${start_frame}_t${glb_trans_vel_scale}r${glb_rot_vel_scale}f${dofSpeedScale}_rfd_${rew_finger_obj_dist_coef}_rh_${rew_delta_hand_pose_coef}
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_0.5_15-02-36-51/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.0_15-11-18-34/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.0.pth'
export lifting_separate_stages=False


# from a to b and c to d #

export goal_dist_thres=0.2
export rew_finger_obj_dist_coef=0.5
export rew_delta_hand_pose_coef=0.0
export rew_obj_pose_coef=1.0
export lifting_separate_stages=True
export data_inst_flag="taco_20231104_203"
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue_15-15-55-56/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue_15-17-14-02/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue.pth'
# export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue_15-13-23-49/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue_15-15-55-56/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue_15-17-32-43/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue.pth'
### checkpoint and xxx ###

export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue_15-20-06-23/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.2_liftingsepTrue.pth'
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.4_liftingsepTrue_15-21-16-29/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.4_liftingsepTrue.pth'
export data_inst_flag="taco_20231104_016"
export object_name='taco_20231104_016'
export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}_modifed_interped.npy
export checkpoint='runs/tracking_taco_20231104_016_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.4_liftingsepTrue_15-23-34-53/nn/tracking_taco_20231104_016_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.4_liftingsepTrue.pth'
export checkpoint='runs/tracking_taco_20231104_016_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.4_liftingsepTrue_16-00-17-57/nn/tracking_taco_20231104_016_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.5_rh_0.0_robj_1.0_gd_0.4_liftingsepTrue.pth'
export strict_lifting_separate_stages=True 



export goal_dist_thres=0.2 ## set the goal_dist_thres to a number larger than 0.0 for encouraging the hand-object contacts
export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=0.5
export rew_obj_pose_coef=1.0
export lifting_separate_stages=False 
export strict_lifting_separate_stages=False
export data_inst_flag='taco_20231104_203' # 
export object_name='taco_20231104_203'
export disable_gravity=True
export ground_distance=-0.1




export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}_modifed_interped.npy
export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}_modifed_interped_transformed.npy
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-13-44-23/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'

export hand_pose_guidance_glb_trans_coef=0.6
export hand_pose_guidance_glb_rot_coef=0.6 # get the glb rot coef #
export hand_pose_guidance_fingerpose_coef=0.1
export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=1.0




export disable_gravity=False
export ground_distance=0.0

export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-14-01-46/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'



export disable_gravity=False
export ground_distance=0.0

export add_table=True
export table_z_dim=0.5
export disable_gravity=True

# export kinematics_only=True


export start_frame=0
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-17-17-05/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'

export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-19-36-38/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'

export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-20-54-40/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'


export disable_gravity=False
export ground_distance=0.0
export add_table=False
export table_z_dim=0.0
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-22-08-42/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'



export hand_pose_guidance_glb_trans_coef=0.6
export hand_pose_guidance_glb_rot_coef=0.6 # orientation is also important 
export hand_pose_guidance_fingerpose_coef=0.1
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2_16-22-20-33/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_1.0_robj_1.0_gd_0.2.pth'



export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2_17-00-17-55/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2.pth'

export disable_gravity=True
export ground_distance=0.0
export add_table=True
export table_z_dim=0.5
export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2_17-02-38-45/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr0_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2.pth'


export disable_gravity=False
export ground_distance=0.0
export add_table=False
export table_z_dim=0.0

export hand_pose_guidance_glb_trans_coef=0.1
export hand_pose_guidance_glb_rot_coef=0.0 # orientation is also important 
export hand_pose_guidance_fingerpose_coef=0.6

export start_frame=40
export episodeLength=150

export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr40_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2_17-14-47-55/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr40_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2.pth'


export start_frame=20
export episodeLength=150

# export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}_modifed_interped_transformed.npy
# passive_active_info_{traj_grab_data_tag}_zrot_3.141592653589793_modifed_interped
export mocap_sv_info_fn=/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_${object_name}_zrot_3.141592653589793_modifed_interped.npy

export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr10_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2_17-15-58-35/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr10_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.2.pth'

# export hand_pose_guidance_glb_trans_coef=0.0 # 0.6
# export hand_pose_guidance_glb_rot_coef=0.1
# export hand_pose_guidance_fingerpose_coef=0.1
# export rew_finger_obj_dist_coef=0.3
# export rew_delta_hand_pose_coef=0.5
# export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.4_liftingsepTrue_15-23-21-17/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.3_rh_0.5_robj_1.0_gd_0.4_liftingsepTrue.pth'
# # 

## rew obj pose coef ##
# export rew_finger_obj_dist_coef=0.0
# export rew_delta_hand_pose_coef=0.5
# export rew_obj_pose_coef=0.0
## lifting stages #
# export checkpoint='runs/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.0_rh_0.5_robj_0.0_15-02-45-30/nn/tracking_taco_20231104_203_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_interpeddata_stfr60_t0.5r0.5f20_rfd_0.0_rh_0.5_robj_0.0.pth'

# 
##### the 



# export glb_trans_vel_scale=7
# export glb_rot_vel_scale=0.5
# export separate_stages=True


##### Using the interpolated data #####


#### taco ####
# export use_twostage_rew=False
# export use_canonical_state=False
# export disable_gravity=True
# export ground_distance=-0.1

# export kinematics_only=True


### TOTEST
# runs/tracking_ori_grab_s2_stanfordbunny_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-20-33-47/nn/tracking_ori_grab_s2_stanfordbunny_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# runs/tracking_ori_grab_s2_cubesmall_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-20-33-47/nn/tracking_ori_grab_s2_cubesmall_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# runs/tracking_ori_grab_s2_cylindermedium_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-20-33-47/nn/tracking_ori_grab_s2_cylindermedium_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# [bad] runs/tracking_ori_grab_s2_cubelarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-20-33-47/nn/tracking_ori_grab_s2_cubelarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# runs/tracking_ori_grab_s2_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-21-34-16/nn/tracking_ori_grab_s2_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# runs/tracking_ori_grab_s2_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-21-34-16/nn/tracking_ori_grab_s2_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# [bad] runs/tracking_ori_grab_s2_spherelarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-21-34-16/nn/tracking_ori_grab_s2_spherelarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
# [bad] runs/tracking_ori_grab_s2_toruslarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-21-34-16/nn/tracking_ori_grab_s2_toruslarge_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth

export test=True
# ####### test setting ########


export train_name=tracking_${object_name}_obs_${obs_type}_density_${rigid_obj_density}_trans_${glb_trans_vel_scale}_rot_${glb_rot_vel_scale}_goalcond_${goal_cond}_${additiona_tag}
export full_experiment_name=${train_name}



export tag=tracking_${object_name}
export cuda_idx=3


# bash scripts/run_tracking_headless.sh

CUDA_VISIBLE_DEVICES=${cuda_idx} python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs=${numEnvs} train.params.config.minibatch_size=${minibatch_size}  task.env.useRelativeControl=${use_relative_control}  train.params.config.max_epochs=10000 task.env.mocap_sv_info_fn=${mocap_sv_info_fn} checkpoint=${checkpoint} task.env.goal_cond=${goal_cond} task.env.object_name=${object_name} tag=${tag} train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name} task.sim.dt=${dt} test=${test} task.env.use_kinematics_bias=${use_kinematics_bias} task.env.w_obj_ornt=${w_obj_ornt} task.env.observationType=${obs_type} task.env.use_canonical_state=${use_canonical_state} task.env.separate_stages=${separate_stages} task.env.rigid_obj_density=${rigid_obj_density} task.env.use_unified_canonical_state=${use_unified_canonical_state} task.env.kinematics_only=${kinematics_only}  task.env.use_fingertips=${use_fingertips}  task.env.glb_trans_vel_scale=${glb_trans_vel_scale} task.env.glb_rot_vel_scale=${glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta=${use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} task.env.dofSpeedScale=${dofSpeedScale} task.env.use_twostage_rew=${use_twostage_rew} task.env.disable_obj_gravity=${disable_gravity} task.env.ground_distance=${ground_distance} task.env.right_hand_dist_thres=${right_hand_dist_thres} task.env.use_real_twostage_rew=${use_real_twostage_rew} task.env.start_grasping_fr=${start_grasping_fr} task.env.controlFrequencyInv=${controlFrequencyInv} task.env.episodeLength=${episodeLength} task.env.start_frame=${start_frame} task.env.rew_obj_pose_coef=${rew_obj_pose_coef} task.env.goal_dist_thres=${goal_dist_thres} task.env.lifting_separate_stages=${lifting_separate_stages} task.env.strict_lifting_separate_stages=${strict_lifting_separate_stages} task.env.add_table=${add_table} task.env.table_z_dim=${table_z_dim} 
 
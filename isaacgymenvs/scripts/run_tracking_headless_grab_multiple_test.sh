
export test=False

export goal_cond=True
export goal_cond=False


### Object task setting ###
export object_name=''
export mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy'
### Object task setting ###

export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'



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

export nn_gpus=8
# export nn_gpus=1
# export nn_gpus=4


export train_name=tracking_${object_name}_obs_${obs_type}_density_${rigid_obj_density}_trans_${glb_trans_vel_scale}_rot_${glb_rot_vel_scale}_goalcond_${goal_cond}_${additiona_tag}
export full_experiment_name=${train_name}

export tag=tracking_${object_name}
export cuda_idx=2

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
export supervised_loss_coef=0.000
export obj_type_to_base_traj_fn='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res_rew_95.0.npy'
# export obj_type_to_base_traj_fn=''
export rew_smoothness_coef=0.001
export rew_smoothness_coef=0.000

export rew_taco_thres=50.0
# export rew_smoothness_coef=0.001
export rew_smoothness_coef=0.000
export customize_damping=True
export supervised_loss_coef=0.0005
export customize_global_damping=True


export train_on_all_trajs=True
export only_training_on_succ_samples=False
export customize_global_damping=False





export use_window_future_selection=False
export forcasting_history_ws=60
export forcasting_inv_freq=60


##### tune all instances settings; trajectory initialization; TACO #####
export checkpoint=''
export grab_obj_type_to_opt_res_fn=''
export only_training_on_succ_samples=False
export generalist_tune_all_instnaces=True
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




export pre_load_trajectories=True
export grab_obj_type_to_opt_res_fn=''
export taco_obj_type_to_opt_res_fn=''
export checkpoint=''
export only_training_on_succ_samples=False 
export customize_damping=True
export rew_smoothness_coef=0.000


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



export dataset_type='grab'
export num_frames=300
export obj_type_to_pre_optimized_traj='../statistics/data_inst_tag_to_optimized_res.npy'
export sampleds_with_object_code_fn=''
export debug=""
export pre_load_trajectories=True
export grab_obj_type_to_opt_res_fn=''
export taco_obj_type_to_opt_res_fn=''
export checkpoint=''
export only_training_on_succ_samples=False 
export customize_damping=True
export customize_global_damping=False
export rew_smoothness_coef=0.000


export strict_maxx_nn_ts=300
export hand_type='allegro'
export grab_train_test_setting=True

export obj_type_to_base_traj_fn=''
export data_inst_flag='ori_grab_s2_apple_lift_nf_300' 



export grab_inst_tag_to_optimized_res_fn='./isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
export taco_inst_tag_to_optimized_res_fn=''


export supervised_loss_coef=0.0000
# export supervised_loss_coef=0.001




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




###### Sim parameters #######
export dt=0.0166
export substeps=2
###### Sim parameters #######


###### History and the future observation settings #######
# export use_future_obs=True
export use_future_obs=False
export use_history_obs=False
export history_freq=5
# export history_freq=1
export history_length=5
# export history_length=4
###### History and the future observation settings #######




GPUS=$1
TRAJ=$2
CKPT=$3
HEADKESS=$4


##### params introduced for forecasting model training #####
export comput_reward_traj_hand_qpos=True
export use_future_ref_as_obs_goal=True
# export use_future_ref_as_obs_goal=False
##### params introduced for forecasting model training #####


##### include obj ornt into the obs #####
# export w_obj_ornt=True
# export include_obj_rot_in_obs=True
export w_obj_ornt=False
export include_obj_rot_in_obs=False
##### include obj ornt into the obs #####

export max_epochs=10000


export tracking_info_st_tag='passive_active_info_'
export tracking_save_info_fn='./data/GRAB_Tracking_PK_reduced_300/data'
export tracking_data_sv_root='./data/GRAB_Tracking_PK_reduced_300/data'
export object_type_to_latent_feature_fn="../assets/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='../assets/inst_tag_to_obj_feat.npy'



# export subj_idx=2
# export subj_nm=${SUBJNM}
export single_instance_training=False
export generalist_tune_all_instnaces=False
export maxx_inst_nn=10000
export single_instance_state_based_train=False


# export target_inst_tag_list_fn=../assets/inst_tag_list_${subj_nm}.npy
# export log_path=./logs/isaacgym_rl_allegro_${subj_nm}

# export target_inst_tag_list_fn=''
# export log_path=./logs/isaacgym_rl_allegro_all


# export target_inst_tag_list_fn=${TAGLISTFN}
export log_path=./logs/isaacgym_rl_allegro_multiple




export subj_nm=''
export object_type_to_latent_feature_fn="../assets/obj_type_to_obj_feat.npy"
export inst_tag_to_latent_feature_fn='../assets/inst_tag_to_obj_feat.npy'
export generalist_tune_all_instnaces=False
export pure_inst_tag=${TRAJ}
# export data_inst_flag='ori_grab_s2_duck_inspect_1_nf_300' 
# export data_inst_flag='ori_grab_s2_camera_takepicture_2_nf_300' 
# export data_inst_flag='ori_grab_s2_apple_eat_1_nf_300' 
# export data_inst_flag='ori_grab_s2_cubesmall_inspect_1_nf_300' 
# export data_inst_flag="ori_grab_s1_flute_play_1_nf_300"
# export data_inst_flag="ori_grab_s9_hammer_use_2_nf_300"
export data_inst_flag=${pure_inst_tag}_nf_300
export target_inst_tag_list_fn=''
export test_inst_tag=${data_inst_flag}
export single_instance_training=True # the learning model would not print additional infos in this setting




##### sampling code #####
export checkpoint=${CKPT}
export test=True
export numEnvs=100  
export minibatch_size=100
export headless=${HEADKESS} # True # False
##### sampling code #####






export st_idx=${GPUS}



# bash scripts/run_tracking_headless_grab_multiple_test.sh 0 s2_duck_inspect_1 /root/diffsim/DexTrack/isaacgymenvs/./logs/isaacgym_rl_allegro_multiple/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-01-18-08/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth




CUDA_VISIBLE_DEVICES=${cuda_idx} python train_pool_2.py --additional_tag=${additional_tag} --hand_type=${hand_type} --numEnvs=${numEnvs} --minibatch_size=${minibatch_size} --goal_cond=${goal_cond}  --test=${test} --use_relative_control=${use_relative_control} --use_kinematics_bias=${use_kinematics_bias} --w_obj_ornt=${w_obj_ornt} --obs_type=${obs_type} --separate_stages=${separate_stages} --rigid_obj_density=${rigid_obj_density}  --kinematics_only=${kinematics_only} --use_fingertips=${use_fingertips}  --use_kinematics_bias_wdelta=${use_kinematics_bias_wdelta} --hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} --hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} --hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} --dt=${dt} --glb_trans_vel_scale=${glb_trans_vel_scale} --glb_rot_vel_scale=${glb_rot_vel_scale} --rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} --rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} ${debug} --nn_gpus=${nn_gpus} --num_frames=${num_frames} --tracking_data_sv_root=${tracking_data_sv_root} --subj_nm=${subj_nm}  --st_idx=${st_idx} --dofSpeedScale=${dofSpeedScale} --use_twostage_rew=${use_twostage_rew} --episodeLength=${episodeLength} --data_inst_flag=${data_inst_flag} --pre_optimized_traj=${pre_optimized_traj} --use_generalist_policy=${use_generalist_policy} --use_hand_actions_rew=${use_hand_actions_rew} --supervised_training=${supervised_training} --checkpoint=${checkpoint} --max_epochs=${max_epochs} --training_mode=${training_mode} --test_inst_tag=${test_inst_tag} --test_optimized_res=${test_optimized_res} --preload_experiences_tf=${preload_experiences_tf} --preload_experiences_path=${preload_experiences_path} --single_instance_training=${single_instance_training} --generalist_tune_all_instnaces=${generalist_tune_all_instnaces} --obj_type_to_pre_optimized_traj=${obj_type_to_pre_optimized_traj} --pre_load_trajectories=${pre_load_trajectories} --sampleds_with_object_code_fn=${sampleds_with_object_code_fn} --log_path=${log_path} --grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} --taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn} --single_instance_tag=${single_instance_tag} --obj_type_to_optimized_res_fn=${obj_type_to_optimized_res_fn} --supervised_loss_coef=${supervised_loss_coef} --pure_supervised_training=${pure_supervised_training} --inst_tag_to_latent_feature_fn=${inst_tag_to_latent_feature_fn} --object_type_to_latent_feature_fn=${object_type_to_latent_feature_fn} --grab_obj_type_to_opt_res_fn=${grab_obj_type_to_opt_res_fn} --taco_obj_type_to_opt_res_fn=${taco_obj_type_to_opt_res_fn}  --maxx_inst_nn=${maxx_inst_nn} --tracking_save_info_fn=${tracking_save_info_fn} --tracking_info_st_tag=${tracking_info_st_tag} --only_training_on_succ_samples=${only_training_on_succ_samples} --rew_filter=${rew_filter} --rew_low_threshold=${rew_low_threshold} --use_strict_maxx_nn_ts=${use_strict_maxx_nn_ts} --taco_interped_data_sv_additional_tag=${taco_interped_data_sv_additional_tag} --strict_maxx_nn_ts=${strict_maxx_nn_ts} --grab_train_test_setting=${grab_train_test_setting} --checkpoint=${checkpoint} --use_local_canonical_state=${use_local_canonical_state} --bound_loss_coef=${bound_loss_coef} --rew_grab_thres=${rew_grab_thres} --rew_taco_thres=${rew_taco_thres} --dataset_type=${dataset_type} --rew_smoothness_coef=${rew_smoothness_coef} --use_base_traj=${use_base_traj} --obj_type_to_base_traj_fn=${obj_type_to_base_traj_fn} --rew_thres_with_selected_insts=${rew_thres_with_selected_insts} --selected_inst_idxes_dict=${selected_inst_idxes_dict} --customize_damping=${customize_damping} --customize_global_damping=${customize_global_damping} --train_on_all_trajs=${train_on_all_trajs}  --single_instance_state_based_train=${single_instance_state_based_train} --controlFrequencyInv=${controlFrequencyInv} --downsample=${downsample} --target_inst_tag_list_fn=${target_inst_tag_list_fn} --use_teacher_model=${use_teacher_model} --teacher_model_path=${teacher_model_path} --teacher_model_inst_tags_fn=${teacher_model_inst_tags_fn} --teacher_index_to_weights=${teacher_index_to_weights} --teacher_index_to_inst_tags=${teacher_index_to_inst_tags} --use_history_obs=${use_history_obs} --history_length=${history_length} --good_inst_opt_res=${good_inst_opt_res} --w_franka=${w_franka}  --early_terminate=${early_terminate} --substeps=${substeps} --use_forcasting_model=${use_forcasting_model} --forcasting_model_weights=${forcasting_model_weights} --forcasting_model_n_layers=${forcasting_model_n_layers} --w_glb_traj_feat_cond=${w_glb_traj_feat_cond} --use_window_future_selection=${use_window_future_selection} --forcasting_inv_freq=${forcasting_inv_freq} --forcasting_history_ws=${forcasting_history_ws} --sv_info_during_training=${sv_info_during_training} --impedance_stiffness_low=${impedance_stiffness_low} --impedance_stiffness_high=${impedance_stiffness_high} --w_impedance_bias_control=${w_impedance_bias_control} --w_obj_latent_features=${w_obj_latent_features} --net_type=${net_type} --history_freq=${history_freq} --use_future_obs=${use_future_obs} --w_history_window_index=${w_history_window_index} --randomize_conditions=${randomize_conditions} --w_inst_latent_features=${w_inst_latent_features} --masked_mimic_training=${masked_mimic_training} --masked_mimic_teacher_model_path=${masked_mimic_teacher_model_path} --forcasting_model_training=${forcasting_model_training} --forcasting_model_lr=${forcasting_model_lr} --forcasting_model_weight_decay=${forcasting_model_weight_decay} --randomize_condition_type=${randomize_condition_type} --add_contact_conditions=${add_contact_conditions} --contact_info_sv_root=${contact_info_sv_root} --st_ed_state_cond=${st_ed_state_cond} --forcasting_diffusion_model=${forcasting_diffusion_model} --random_shift_cond=${random_shift_cond} --random_shift_cond_freq=${random_shift_cond_freq} --maxx_inv_cond_freq=${maxx_inv_cond_freq} --only_use_hand_first_frame=${only_use_hand_first_frame} --comput_reward_traj_hand_qpos=${comput_reward_traj_hand_qpos} --use_future_ref_as_obs_goal=${use_future_ref_as_obs_goal} --include_obj_rot_in_obs=${include_obj_rot_in_obs}
 


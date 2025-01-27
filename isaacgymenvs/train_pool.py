# Script to train policies in Isaac Gym
#
# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
import logging
import os
# from datetime import datetime

# noinspection PyUnresolvedReferences
import isaacgym
import sys
sys.path.append('.')
sys.path.append('..')




from omegaconf import open_dict


from multiprocessing import Pool
from multiprocessing import Process
import argparse
import numpy as np

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ##### pool settings ####
    parser.add_argument("--launch_type", type=str, default='trajectory')
    parser.add_argument("--tracking_data_sv_root", type=str, default='/cephfs/yilaa/data/GRAB_Tracking/data')
    parser.add_argument("--subj_nm", type=str, default='')
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--num_frames", type=int, default=150, help="number of vector envs")
    parser.add_argument("--base_dir", type=str, default='/cephfs/yilaa/uni_manip/tds_rl_exp', help="Mocap save info file")
   
   
    ##### experiment settings #####
    parser.add_argument("--hand_type",  type=str, default='allegro', help="path to the optimized qtars")
    
    
    ##### isaacgym settings #####
    parser.add_argument("--numEnvs", type=int, default=8000)
    parser.add_argument("--minibatch_size", type=int, default=8000)
    parser.add_argument("--use_relative_control", type=str2bool,  default=True)
    parser.add_argument("--goal_cond", type=str2bool, default=False)
    # parser.add_argument("--object_name", type=str, default='')
    parser.add_argument("--obs_type", type=str, default='pure_state_wref_wdelta')
    parser.add_argument("--rigid_obj_density", type=float, default=500)
    parser.add_argument("--glb_trans_vel_scale", type=float, default=0.1)
    parser.add_argument("--glb_rot_vel_scale", type=float, default=0.1)
    parser.add_argument("--additional_tag", type=str, default='')
    parser.add_argument("--dt", type=float, default=0.0166)
    parser.add_argument("--test", type=str2bool,  default=False)
    parser.add_argument("--use_kinematics_bias", type=str2bool,  default=True)  
    parser.add_argument("--w_obj_ornt",  type=str2bool,   default=False)  
    parser.add_argument("--separate_stages",  type=str2bool,  default=False)  
    parser.add_argument("--kinematics_only",  type=str2bool,   default=False) 
    parser.add_argument("--use_fingertips",  type=str2bool,   default=True) 
    parser.add_argument("--use_kinematics_bias_wdelta",  type=str2bool,  default=True)  
    parser.add_argument("--hand_pose_guidance_glb_trans_coef", type=float, default=0.6)
    parser.add_argument("--hand_pose_guidance_glb_rot_coef", type=float, default=0.1)
    parser.add_argument("--hand_pose_guidance_fingerpose_coef", type=float, default=0.1)
    parser.add_argument("--rew_finger_obj_dist_coef", type=float, default=0.5)
    parser.add_argument("--rew_delta_hand_pose_coef", type=float, default=0.5)
    parser.add_argument("--nn_gpus", type=int, default=8)
    parser.add_argument("--st_idx", type=int, default=0)
    parser.add_argument("--dofSpeedScale", type=float, default=20)
    # use_twostage_rew
    parser.add_argument("--use_twostage_rew", type=str2bool,  default=False)
    parser.add_argument("--dataset_type", type=str, default='grab')
    
    ### sim steup ###
    parser.add_argument("--ground_distance", type=float, default=0.0)
    parser.add_argument("--use_canonical_state",  type=str2bool,   default=False) 
    parser.add_argument("--disable_gravity",  type=str2bool,   default=False) 
    parser.add_argument("--data_inst_flag", type=str, default='')
    # right_hand_dist_thres #
    parser.add_argument("--right_hand_dist_thres", type=float, default=0.12)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--max_epochs", type=int, default=1000)
    # use_real_twostage_rew # use_real_twostage_rew # # two stage rewards # #
    parser.add_argument("--use_real_twostage_rew",  type=str2bool,   default=False) 
    # start_grasping_fr
    parser.add_argument("--start_grasping_fr",  type=str2bool,   default=False) 
    # 
    # controlFrequencyInv
    parser.add_argument("--controlFrequencyInv", type=int, default=1)
    parser.add_argument("--use_interpolated_data",  type=str2bool,   default=False) 
    # episodeLength
    parser.add_argument("--episodeLength", type=int, default=1000)
    # start_frame
    parser.add_argument("--start_frame", type=int, default=0)
    # rew_obj_pose_coef
    parser.add_argument("--rew_obj_pose_coef", type=float, default=1.0)
    # goal_dist_thres
    parser.add_argument("--goal_dist_thres", type=float, default=0.0)
    # lifting_separate_stages
    parser.add_argument("--lifting_separate_stages",  type=str2bool,   default=False) 
    
    # strict_lifting_separate_stages
    parser.add_argument("--strict_lifting_separate_stages",  type=str2bool,   default=False) 
    parser.add_argument("--add_table",  type=str2bool,   default=False) 
    # table_z_dim
    parser.add_argument("--table_z_dim", type=float, default=0.5)
    parser.add_argument("--headless", type=str2bool,   default=True)
    # parser.add_argument("--target_object_name", type=str, default='')
    # parser.add_argument("--target_mocap_sv_fn", type=str, default='')
    # use_taco_obj_traj
    parser.add_argument("--use_taco_obj_traj", type=str2bool,   default=True)
    # pre_optimized_traj
    parser.add_argument("--pre_optimized_traj", type=str, default='')
    # pre optimized traj 
    # pre_load_trajectories, obj_type_to_pre_optimized_traj
    ### TODO: add pre_load_trajectories, obj_type_to_pre_optimized_traj ###
    parser.add_argument("--pre_load_trajectories", type=str2bool,   default=False)
    parser.add_argument("--obj_type_to_pre_optimized_traj", type=str, default='')
    parser.add_argument("--subj_idx", type=int, default=0)
    # parser.add_argument("--hand_type", type=str, default='allegro')
    # use_vision
    parser.add_argument("--use_vision", type=str2bool,   default=False)
    parser.add_argument("--use_dagger", type=str2bool,   default=False)
    parser.add_argument("--use_generalist_policy", type=str2bool,   default=True)
    # 
    parser.add_argument("--use_hand_actions_rew", type=str2bool,   default=False)
    # supervised_training
    parser.add_argument("--supervised_training", type=str2bool,   default=True)
    # checkpoint
    parser.add_argument("--test_inst_tag", type=str, default='')
    parser.add_argument("--test_optimized_res", type=str, default='')
    parser.add_argument("--training_mode", type=str, default='regular')

    parser.add_argument("--preload_experiences_tf", type=str2bool,   default=False)
    parser.add_argument("--preload_experiences_path", type=str,   default=None)
    # single_instance_training
    parser.add_argument("--single_instance_training", type=str2bool,   default=False)
    # parser.add_argument("--checkpoint", type=str, default='') # 
    parser.add_argument("--generalist_tune_all_instnaces", type=str2bool,   default=False)
    # Related to base trajectory
    parser.add_argument("--sampleds_with_object_code_fn", type=str, default='')
    # log_path
    parser.add_argument("--log_path", type=str, default='./runs')
    # grab_inst_tag_to_optimized_res_fn, taco_inst_tag_to_optimized_res_fn
    parser.add_argument("--grab_inst_tag_to_optimized_res_fn", type=str, default='')
    parser.add_argument("--taco_inst_tag_to_optimized_res_fn", type=str, default='')
    parser.add_argument("--single_instance_tag", type=str, default='')
    parser.add_argument("--obj_type_to_optimized_res_fn", type=str, default='')
    # supervised_loss_coef
    parser.add_argument("--supervised_loss_coef", type=float, default=0.0005)
    # pure_supervised_training
    parser.add_argument("--pure_supervised_training", type=str2bool,   default=False)
    # inst_tag_to_latent_feature_fn
    parser.add_argument("--inst_tag_to_latent_feature_fn", type=str, default='')
    
    parser.add_argument("--object_type_to_latent_feature_fn", type=str, default='')
    parser.add_argument("--grab_obj_type_to_opt_res_fn", type=str, default='')
    parser.add_argument("--taco_obj_type_to_opt_res_fn", type=str, default='')
    parser.add_argument("--maxx_inst_nn", type=int, default=10000)
    # tracking_save_info_fn, tracking_info_st_tag
    parser.add_argument("--tracking_save_info_fn", type=str, default="/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data")
    parser.add_argument("--tracking_info_st_tag", type=str, default='passive_active_info_')
    parser.add_argument("--only_training_on_succ_samples", type=str2bool,   default=False)
    parser.add_argument("--exclude_inst_tag_to_opt_res_fn", type=str, default='')
    parser.add_argument("--rew_filter", type=str2bool,   default=False)
    parser.add_argument("--rew_low_threshold", type=float, default=0.0)
    # use_teacher_model
    parser.add_argument("--use_teacher_model", type=str2bool,   default=False)
    # use_strict_maxx_nn_ts
    parser.add_argument("--use_strict_maxx_nn_ts", type=str2bool,   default=True)
    # taco_interped_data_sv_additional_tag
    parser.add_argument("--taco_interped_data_sv_additional_tag", type=str, default='')
    # strict_maxx_nn_ts
    parser.add_argument("--strict_maxx_nn_ts", type=int, default=300)
    # parser.add_argument("--inst_tag_to_latent_feature_fn", type=str, default='')
    parser.add_argument("--grab_train_test_setting", type=str2bool,   default=False)
    # use_local_canonical_state
    parser.add_argument("--use_local_canonical_state", type=str2bool,   default=False)
    # bound_loss_coef
    parser.add_argument("--bound_loss_coef", type=float, default=0.0001)
    # rew_grab_thres, rew_taco_thres
    parser.add_argument("--rew_grab_thres", type=float, default=50.0)
    parser.add_argument("--rew_taco_thres", type=float, default=200.0)
    # rew_smoothness_coef
    parser.add_argument("--rew_smoothness_coef", type=float, default=0.0)
    # obj_type_to_base_traj_fn
    parser.add_argument("--obj_type_to_base_traj_fn", type=str, default='')
    parser.add_argument("--use_base_traj", type=str2bool,   default=False)

    parser.add_argument("--rew_thres_with_selected_insts", type=str2bool,   default=False)
    parser.add_argument("--selected_inst_idxes_dict", type=str, default='')
    # customize_damping
    parser.add_argument("--customize_damping", type=str2bool,   default=True)
    # customize_global_damping
    parser.add_argument("--customize_global_damping", type=str2bool,   default=False)
    parser.add_argument("--train_on_all_trajs", type=str2bool,   default=False)

    parser.add_argument("--single_instance_state_based_train", type=str2bool,   default=False)
    # data_selection_ratio
    parser.add_argument("--data_selection_ratio", type=float,   default=1.0)
    # test_taco_tag = 'taco_20231024_'
    # export test_taco_tag='taco_20231024_
    parser.add_argument("--test_taco_tag", type=str, default='taco_20231024_')
    # wo_vel_obs
    parser.add_argument("--wo_vel_obs", type=str2bool,   default=False)
    # TODO: we may only need to downsample the kinematics trajs #
    parser.add_argument("--downsample", type=str2bool,   default=False)
    # target_inst_tag_list_fn #
    parser.add_argument("--target_inst_tag_list_fn", type=str, default='')
    # controlFrequencyInv
    # parser.add_argument("--controlFrequencyInv", type=int,   default=1)
    # use_teacher_model, teacher_model_path, teacher_model_inst_tags_fn
    # parser.add_argument("--use_teacher_model", type=str2bool,   default=False)
    parser.add_argument("--teacher_model_path", type=str, default='')
    parser.add_argument("--teacher_model_inst_tags_fn", type=str, default='')
    parser.add_argument("--teacher_index_to_weights", type=str, default='')
    parser.add_argument("--teacher_index_to_inst_tags", type=str, default='')
    parser.add_argument("--obj_type_to_base_trajs_config", type=str, default='')
    # use_history_obs, history_length
    parser.add_argument("--use_history_obs", type=str2bool,   default=False)
    parser.add_argument("--history_length", type=int, default=5)
    # good_inst_opt_res
    parser.add_argument("--good_inst_opt_res", type=str, default='')
    # w_franka
    parser.add_argument("--w_franka", type=str2bool,   default=False)
    # randomize
    parser.add_argument("--randomize", type=str2bool,   default=False)
    # early_terminate
    parser.add_argument("--early_terminate", type=str2bool,   default=False)
    parser.add_argument("--substeps", type=int, default=2)
    
    # use_forcasting_model, forcasting_model_weights, forcasting_model_n_layers, w_glb_traj_feat_cond #
    # parser.add_argument("--use_forcasting_model", type=str2bool,   default=False)
    # parser.add_argument("--forcasting_model_weights", type=str,   default='')
    # parser.add_argument("--forcasting_model_n_layers", type=int, default=7)
    # w_glb_traj_feat_cond #
    # parser.add_argument("--w_glb_traj_feat_cond", type=str2bool, default=False)
    # use_window_future_selection
    # parser.add_argument("--use_window_future_selection", type=str2bool, default=False)
    # forcasting_inv_freq
    # parser.add_argument("--forcasting_inv_freq", type=int, default=1)
    
    # stiffness_coef
    parser.add_argument("--stiffness_coef", type=float,   default=100.0)
    parser.add_argument("--damping_coef", type=float,   default=4.0)
    parser.add_argument("--effort_coef", type=float,   default=0.95)
    # forcasting_history_ws
    # parser.add_argument("--forcasting_history_ws", type=int, default=1)
    # sv_info_during_training
    # parser.add_argument("--sv_info_during_training", type=str2bool, default=False)
    # impedance_stiffness_low, impedance_stiffness_high, w_impedance_bias_control #
    # parser.add_argument("--impedance_stiffness_low", type=float,   default=1.0)
    # parser.add_argument("--impedance_stiffness_high", type=float,   default=50.0)
    # parser.add_argument("--w_impedance_bias_control", type=str2bool, default=False)
    # # w_obj_latent_features
    parser.add_argument("--w_obj_latent_features", type=str2bool, default=True)
    parser.add_argument("--net_type", type=str,   default='v4')
    # history_freq
    parser.add_argument("--history_freq", type=int, default=1)
    parser.add_argument("--use_future_obs", type=str2bool, default=False)
    # w_history_window_index
    parser.add_argument("--w_history_window_index", type=str2bool, default=False)
    # randomize_conditions
    parser.add_argument("--randomize_conditions", type=str2bool, default=False)
    # w_inst_latent_features
    parser.add_argument("--w_inst_latent_features", type=str2bool, default=False)
    # masked_mimic_training, masked_mimic_teacher_model_path
    # parser.add_argument("--masked_mimic_training", type=str2bool, default=False)
    # parser.add_argument("--masked_mimic_teacher_model_path", type=str,   default='')
    # forcasting_model_training, forcasting_model_lr, forcasting_model_weight_decay
    # parser.add_argument("--forcasting_model_training", type=str2bool, default=False)
    # parser.add_argument("--forcasting_model_lr", type=float,   default=1e-4)
    # parser.add_argument("--forcasting_model_weight_decay", type=float,   default=5e-5)
    # randomize_condition_type
    # parser.add_argument("--randomize_condition_type", type=str,   default='random')
    # add_contact_conditions, contact_info_sv_root
    parser.add_argument("--add_contact_conditions", type=str2bool, default=False)
    # contact_info_sv_root
    # parser.add_argument("--contact_info_sv_root", type=str,   default='')
    # parser.add_argument("--partial_hand_info", type=str2bool, default=False)
    # parser.add_argument("--partial_obj_info", type=str2bool, default=False)
    # st_ed_state_cond
    # parser.add_argument("--st_ed_state_cond", type=str2bool, default=False)
    # parser.add_argument("--forcasting_diffusion_model", type=str2bool, default=False)
    
    # --history_window_size=${history_window_size} --glb_feat_per_skip=${glb_feat_per_skip} --centralize_info=${centralize_info}
    parser.add_argument("--history_window_size", type=int, default=60)
    parser.add_argument("--glb_feat_per_skip", type=int, default=1)
    parser.add_argument("--centralize_info", type=str2bool, default=False)
    
    # random_shift_cond, random_shift_cond_freq, maxx_inv_cond_freq
    parser.add_argument("--random_shift_cond", type=str2bool, default=False)
    parser.add_argument("--random_shift_cond_freq", type=str2bool, default=False)
    parser.add_argument("--maxx_inv_cond_freq", type=int, default=30) ## 30 maxx inv cond freq ##
    # only_use_hand_first_frame
    parser.add_argument("--only_use_hand_first_frame", type=str2bool, default=False)
    # parser.add_argument("--only_use_hand_first_frame", type=str2bool, default=False)
    # forecasting_obs_dim: 797
    # forecasting_act_dim: 29
    # forecasting_nn_frames: 10 # 
    parser.add_argument("--forecasting_obs_dim", type=int, default=797) 
    parser.add_argument("--forecasting_act_dim", type=int, default=29) 
    parser.add_argument("--forecasting_nn_frames", type=int, default=10)     
    
    parser.add_argument("--w_forecasting_model", type=str2bool, default=False)
    # use_world_model
    parser.add_argument("--use_world_model", type=str2bool, default=False)
    # train_controller
    parser.add_argument("--train_controller", type=str2bool, default=False)
    # train_forecasting_model
    parser.add_argument("--train_forecasting_model", type=str2bool, default=False)
    # forecasting_model_weight_fn
    parser.add_argument("--forecasting_model_weight_fn", type=str,   default='')
    # single_inst_tag
    parser.add_argument("--single_inst_tag", type=str,   default='')
    # activate_forecaster
    parser.add_argument("--activate_forecaster", type=str2bool, default=True)
    # comput_reward_traj_hand_qpos
    parser.add_argument("--comput_reward_traj_hand_qpos", type=str2bool, default=True)
    # use_future_ref_as_obs_goal
    parser.add_argument("--use_future_ref_as_obs_goal", type=str2bool, default=True)
    # forecast_obj_pos
    parser.add_argument("--forecast_obj_pos", type=str2bool, default=False)
    # multiple_kine_source_trajs_fn
    parser.add_argument("--multiple_kine_source_trajs_fn", type=str,   default='')
    # use_multiple_kine_source_trajs
    parser.add_argument("--use_multiple_kine_source_trajs", type=str2bool, default=False)
    # include_obj_rot_in_obs 
    parser.add_argument("--include_obj_rot_in_obs", type=str2bool, default=False)
    # compute_hand_rew_buf_threshold
    parser.add_argument("--compute_hand_rew_buf_threshold", type=int, default=500) 
    # parser.add_argument("--vis_train", type=str2bool, default=False)
    # load_kine_info_retar_with_arm, kine_info_with_arm_sv_root
    parser.add_argument("--load_kine_info_retar_with_arm", type=str2bool, default=False) 
    parser.add_argument("--kine_info_with_arm_sv_root", type=str, default='') 
    # w_finger_pos_rew
    parser.add_argument("--w_finger_pos_rew", type=str2bool, default=False) 
    # franka_delta_delta_mult_coef
    parser.add_argument("--franka_delta_delta_mult_coef", type=float,   default=1.0)
    # control_arm_via_ik
    parser.add_argument("--control_arm_via_ik", type=str2bool, default=False) 
    # warm_trans_actions_mult_coef, warm_rot_actions_mult_coef
    parser.add_argument("--warm_trans_actions_mult_coef", type=float,   default=0.04)
    parser.add_argument("--warm_rot_actions_mult_coef", type=float,   default=0.04)
    # hand_qpos_rew_coef
    parser.add_argument("--hand_qpos_rew_coef", type=float,   default=0.00)
    # 
    parser.add_argument("--schedule_ornt_rew_coef", type=str2bool, default=False)
    parser.add_argument("--lowest_ornt_rew_coef", type=float,   default=0.03)
    parser.add_argument("--highest_ornt_rew_coef", type=float,   default=0.33)
    parser.add_argument("--ornt_rew_coef_warm_starting_steps", type=int, default=100) 
    parser.add_argument("--ornt_rew_coef_increasing_steps", type=int, default=200) 
    
    parser.add_argument("--schedule_hodist_rew_coef", type=str2bool, default=False)
    parser.add_argument("--lowest_rew_finger_obj_dist_coef", type=float,   default=0.1)
    parser.add_argument("--highest_rew_finger_obj_dist_coef", type=float,   default=0.5)
    parser.add_argument("--hodist_rew_coef_warm_starting_steps", type=int, default=100) 
    parser.add_argument("--hodist_rew_coef_increasing_steps", type=int, default=300) 
    
    parser.add_argument("--log_root", type=str, default='')
    # scaling factor, scaling progress after #
    parser.add_argument("--hand_glb_mult_factor_scaling_coef", type=float,   default=1.0)
    parser.add_argument("--hand_glb_mult_scaling_progress_after", type=int, default=900) 
    parser.add_argument("--hand_glb_mult_scaling_progress_before", type=int, default=900) 
    parser.add_argument("--not_use_kine_bias", type=str2bool, default=False)
    parser.add_argument("--disable_hand_obj_contact", type=str2bool, default=False)
    # wo_fingertip_rot_vel
    parser.add_argument("--wo_fingertip_rot_vel", type=str2bool, default=False)
    # wo_fingertip_vel
    parser.add_argument("--wo_fingertip_vel", type=str2bool, default=False)
    # arm_stiffness
    parser.add_argument("--arm_stiffness", type=float,   default=400.0)
    parser.add_argument("--arm_effort", type=float,   default=200.0)
    parser.add_argument("--arm_damping", type=float,   default=80.0)
    # train_student_model, ts_teacher_model_obs_dim, ts_teacher_model_weights_fn
    parser.add_argument("--train_student_model", type=str2bool, default=False)
    parser.add_argument("--ts_teacher_model_obs_dim", type=int, default=731) 
    parser.add_argument("--ts_teacher_model_weights_fn", type=str, default='') 
    # randomize_obj_init_pos, randomize_obs_more
    parser.add_argument("--randomize_obj_init_pos", type=str2bool, default=False)
    parser.add_argument("--randomize_obs_more", type=str2bool, default=False)
    parser.add_argument("--obj_init_pos_rand_sigma", type=float,   default=0.1)
    # 
    parser.add_argument("--obs_simplified", type=str2bool, default=False)
    # w_traj_modifications
    parser.add_argument("--w_traj_modifications", type=str2bool, default=False)
    # wo_fingertip_pos
    parser.add_argument("--wo_fingertip_pos", type=str2bool, default=False)
    # rand_obj_mass_lowest_range, rand_obj_mass_highest_range
    parser.add_argument("--rand_obj_mass_lowest_range", type=float,   default=0.5)
    parser.add_argument("--rand_obj_mass_highest_range", type=float,   default=1.5)
    # use_v2_leap_warm_urdf
    parser.add_argument("--use_v2_leap_warm_urdf", type=str2bool, default=False)
    # hand_specific_randomizations
    parser.add_argument("--hand_specific_randomizations", type=str2bool, default=False)
    parser.add_argument("--action_specific_randomizations", type=str2bool, default=False)
    parser.add_argument("--action_specific_rand_noise_scale", type=float,   default=0.5)
    # reset_obj_mass: False
    # obj_mass_reset: 0.27
    # recompute_inertia: False # recompute # previledged info #
    parser.add_argument("--reset_obj_mass", type=str2bool, default=False)
    parser.add_argument("--obj_mass_reset", type=float,   default=0.27)
    parser.add_argument("--recompute_inertia", type=str2bool, default=False)
    parser.add_argument("--use_vision_obs", type=str2bool, default=False)
    ## w_rotation_axis_rew
    parser.add_argument("--w_rotation_axis_rew", type=str2bool, default=False)
    # add_physical_params_in_obs
    parser.add_argument("--add_physical_params_in_obs", type=str2bool, default=False)
    # whether_randomize_obs_act
    parser.add_argument("--whether_randomize_obs_act", type=str2bool, default=True)
    # obs_rand_noise_scale
    parser.add_argument("--obs_rand_noise_scale", type=float,   default=100.0)
    # dagger_style_training
    parser.add_argument("--dagger_style_training", type=str2bool, default=False)
    # whether_randomize_obs
    parser.add_argument("--whether_randomize_obs", type=str2bool, default=True)
    # whether_randomize_act
    parser.add_argument("--whether_randomize_act", type=str2bool, default=True)
    parser.add_argument("--teacher_subj_idx", type=int, default=2) 
    parser.add_argument("--action_chunking", type=str2bool, default=False )
    # action_chunking_frames
    parser.add_argument("--action_chunking_frames", type=int, default=1) 
    #  rollout_teacher_model, rollout_student_model
    parser.add_argument("--rollout_teacher_model", type=str2bool, default=False)
    parser.add_argument("--rollout_student_model", type=str2bool, default=True)
    parser.add_argument("--horizon_length", type=int, default=32) 
    # bc_style_training
    parser.add_argument("--bc_style_training", type=str2bool, default=False)
    # export demonstration_tuning_model, demonstration_tuning_model_freq
    parser.add_argument("--demonstration_tuning_model", type=str2bool, default=False)
    parser.add_argument("--demonstration_tuning_model_freq", type=int, default=1) 
    # bc_relative_targets
    parser.add_argument("--bc_relative_targets", type=str2bool, default=False)
    # distill_full_to_partial
    parser.add_argument("--distill_full_to_partial", type=str2bool, default=False)
    # train_free_hand
    parser.add_argument("--train_free_hand", type=str2bool, default=False)
    parser.add_argument("--simreal_modeling", type=str2bool, default=False)
    # more_allegro_stiffness
    parser.add_argument("--more_allegro_stiffness", type=str2bool, default=False)
    # add distinuguish base trajectories #
    # distinguish_kine_with_base_traj
    parser.add_argument("--distinguish_kine_with_base_traj", type=str2bool, default=False)
    # distill_delta_targets
    parser.add_argument("--distill_delta_targets", type=str2bool, default=False)
    parser.add_argument("--preset_cond_type", type=int, default=0)
    # preload_all_saved_exp_buffers
    parser.add_argument("--preload_all_saved_exp_buffers", type=str2bool, default=False)
    # use_actual_prev_targets_in_obs
    parser.add_argument("--use_actual_prev_targets_in_obs", type=str2bool, default=False)
    parser.add_argument("--use_no_obj_pose", type=str2bool, default=False)
    parser.add_argument("--multi_traj_use_joint_order_in_sim", type=str2bool, default=False)
    parser.add_argument("--action_chunking_skip_frames", type=int, default=1)
    parser.add_argument("--multi_inst_chunking", type=str2bool, default=False)
    parser.add_argument("--add_obj_features", type=str2bool, default=False)
    # parser.add_argument("--reset_obj_mass", type=str2bool, default=False)
    # kine_ed_tag
    parser.add_argument("--kine_ed_tag", type=str,   default='.npy')
    # only_rot_axis_guidance
    parser.add_argument("--only_rot_axis_guidance", type=str2bool, default=False)
    parser.add_argument("--add_global_motion_penalty", type=str2bool, default=False)
    parser.add_argument("--add_torque_penalty", type=str2bool, default=False)
    parser.add_argument("--add_work_penalty", type=str2bool, default=False)
    #
    parser.add_argument("--use_multi_step_control", type=str2bool, default=False)
    parser.add_argument("--nn_control_substeps", type=int, default=10)
    parser.add_argument("--schedule_episod_length", type=str2bool, default=False)
    parser.add_argument("--episod_length_low", type=int, default=270)
    parser.add_argument("--episod_length_high", type=int, default=500)
    parser.add_argument("--episod_length_warming_up_steps", type=int, default=130)
    parser.add_argument("--episod_length_increasing_steps", type=int, default=200)
    # use_actual_traj_length
    parser.add_argument("--use_actual_traj_length", type=str2bool, default=False)
    # randomize_reset_frame
    parser.add_argument("--randomize_reset_frame", type=str2bool, default=False)
    # add_forece_obs
    parser.add_argument("--add_forece_obs", type=str2bool, default=False)
    parser.add_argument("--distill_via_bc", type=str2bool, default=False)
    parser.add_argument("--record_experiences", type=str2bool, default=False)
    # learning_rate
    parser.add_argument("--learning_rate", type=float,   default=5e-4)
    parser.add_argument("--forecasting_obs_with_original_obs", type=str2bool, default=False)
    parser.add_argument("--use_gene_task_v2", type=str2bool, default=False)
    parser.add_argument("--schedule_glb_action_penalty", type=str2bool, default=False)
    parser.add_argument("--glb_penalty_low", type=float, default=0.0003)
    parser.add_argument("--glb_penalty_high", type=float, default=1.0)
    parser.add_argument("--glb_penalty_warming_up_steps", type=int, default=50)
    parser.add_argument("--glb_penalty_increasing_steps", type=int, default=300)
    parser.add_argument("--add_hand_targets_smooth", type=str2bool, default=False)
    parser.add_argument("--hand_targets_smooth_coef", type=float, default=0.4)
    parser.add_argument("--preset_multi_traj_index", type=int, default=-1) 
    # save_experiences_via_ts
    parser.add_argument("--save_experiences_via_ts", type=str2bool, default=False)
    # load_experiences_maxx_ts
    parser.add_argument("--load_experiences_maxx_ts", type=int, default=600) 
    # switch_between_models, switch_to_trans_model_frame_after, switch_to_trans_model_ckpt_fn
    parser.add_argument("--switch_between_models", type=str2bool, default=False)
    parser.add_argument("--switch_to_trans_model_frame_after", type=int, default=310)
    parser.add_argument("--switch_to_trans_model_ckpt_fn", type=str,   default='')
    parser.add_argument("--traj_idx_to_experience_sv_folder", type=str,   default='')
    # load_chunking_experiences_v2
    parser.add_argument("--load_chunking_experiences_v2", type=str2bool, default=False)
    parser.add_argument("--history_chunking_obs_version", type=str,   default='v1')
    # load_chunking_experiences_from_real
    parser.add_argument("--load_chunking_experiences_from_real", type=str2bool, default=False)
    parser.add_argument("--closed_loop_to_real", type=str2bool, default=False)
    
    # use_transformer_model
    parser.add_argument("--use_transformer_model", type=str2bool, default=False)
    args = parser.parse_args()
    
    if len(args.subj_nm) > 0:
        args.subj_idx = int(args.subj_nm[1:])
    
    # 
    if len(args.obj_type_to_optimized_res_fn) > 0 and os.path.exists(args.obj_type_to_optimized_res_fn):
        obj_type_to_optimized_res = np.load(args.obj_type_to_optimized_res_fn, allow_pickle=True).item()
        pure_obj_type_to_optimized_res = {}
        for key in obj_type_to_optimized_res:
            if isinstance(key, tuple):
                pure_obj_type_to_optimized_res[key[0]] = obj_type_to_optimized_res[key]
            else:
                pure_obj_type_to_optimized_res[key] = obj_type_to_optimized_res[key]
        # print(f"obj_type_to_optimized_res: {pure_obj_type_to_optimized_res.keys()}")
    else:
        pure_obj_type_to_optimized_res = None 
    
    if len(args.exclude_inst_tag_to_opt_res_fn) > 0 and os.path.exists(args.exclude_inst_tag_to_opt_res_fn):
        exclude_inst_tag_to_opt_res_raw = np.load(args.exclude_inst_tag_to_opt_res_fn, allow_pickle=True).item()
        exclude_inst_tag_to_opt_res = {}
        for key in exclude_inst_tag_to_opt_res_raw:
            if isinstance(key, tuple):
                exclude_inst_tag_to_opt_res[key[0]] = exclude_inst_tag_to_opt_res_raw[key]
            else:
                exclude_inst_tag_to_opt_res[key] = exclude_inst_tag_to_opt_res_raw[key]
        print(f"exclude_inst_tag_to_opt_res: {exclude_inst_tag_to_opt_res.keys()}")
    else:
        exclude_inst_tag_to_opt_res = None    
        
            

    def launch_one_process(cur_grab_data_tag, traj_grab_data_tag, cuda_idx, pre_optimized_traj=None):
        # print(f"pre_optimized_traj: {pre_optimized_traj}")
        obs_type = args.obs_type
        # use_small_sigmas = args.use_small_sigmas
        # finger_urdf_template = args.finger_urdf_template
        # finger_near_palm_joint_idx = args.finger_near_palm_joint_idx
        # constraint_level = args.constraint_level
        # object_type = cur_grab_data_tag # cur_grab_dta_tag #
        object_name = cur_grab_data_tag
        # task_type = "mocap_tracking"
        if args.hand_type == 'allegro':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB_Tracking/data/passive_active_info_{traj_grab_data_tag}.npy"
            
    
            if args.dataset_type == 'grab':
                mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
            elif args.dataset_type == 'taco':
                # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
                # if args.use_interpolated_data:
                #     # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_modifed_interped_transformed.npy"
                #     # passive_active_info_taco_20231104_203_zrot_3.141592653589793_modifed_interped # 
                #     mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_zrot_3.141592653589793_modifed_interped.npy"
                #     if not os.path.exists(mocap_sv_info_fn):
                #         mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_modifed_interped.npy"
                #### by default we use interpolated data #### 
                if len(args.taco_interped_data_sv_additional_tag) > 0:
                    mocap_sv_info_fn = f"passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2_{args.taco_interped_data_sv_additional_tag}.npy"
                else:
                    mocap_sv_info_fn = f"passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2.npy"
                # mocap_sv_info_fn = f"passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2.npy"
                mocap_sv_info_fn = os.path.join(args.tracking_data_sv_root, mocap_sv_info_fn)
            
            else:
                raise ValueError
        
        elif args.hand_type == 'leap':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB _Tracking/data/leap_passive_active_info_{traj_grab_data_tag}.npy" # leap -- tracking data sv root#
            # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/leap_passive_active_info_{traj_grab_data_tag}.npy"
            mocap_sv_info_fn = f"/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2_interpfr_60_interpfr2_60_nntrans_40.npy"
        else:
            raise ValueError
        
        
        print(f"mocap_sv_info_fn: {mocap_sv_info_fn}")
        
        
        
        
        # checkpoint = ''
        tag = f"tracking_{object_name}"
        
        # if args.launch_type == 'trajectory':
        # if args.hand_type == 'allegro':
        train_name = f"tracking_{object_name}_{args.hand_type}_{args.additional_tag}"
        
        
        
        full_experiment_name = train_name
        
        if args.headless:
            capture_video = False
            force_render = False
        else:
            capture_video = False # True
            force_render = True
            
            if args.use_vision_obs:
                capture_video = True
                force_render = True
        
        
        
        cuda_visible_text = f"CUDA_VISIBLE_DEVICES={cuda_idx} "
        
        
        exp_dir = '.'
        
        ## Not in use now ##
        # if len(args.target_object_name) > 0:
        #     object_name=args.target_object_name
        
        # if len(args.target_mocap_sv_fn) > 0:
        #     mocap_sv_info_fn = args.target_mocap_sv_fn
            
        if pre_optimized_traj is not None and len(pre_optimized_traj) > 0:
            cur_pre_optimized_traj = pre_optimized_traj
        else:
            cur_pre_optimized_traj = args.pre_optimized_traj
            
        if args.use_vision:
            task_type = "AllegroHandTrackingVision"
            train_type = "HumanoidPPOVision"
            
            enableCameraSensors = True
            if args.use_dagger:
                task_type = "AllegroHandTrackingVision"
                train_type = "HumanoidPPOVisionDAgger"
            print(f"task_type: {task_type}, train_type: {train_type}")
        else:
            
            task_type = "AllegroHandTrackingGeneralist"
            train_type = "HumanoidPPO"
            
            
            task_type = "AllegroHandTrackingGeneralist"
            train_type = "HumanoidPPOSupervised"
            
            if args.single_instance_state_based_train:
                train_type = "HumanoidPPOSupervisedSN"
            
            
            
            training_mode_config = f"train.params.config.training_mode={args.training_mode}"
            test_inst_config = f"task.env.test_inst_tag={args.test_inst_tag} task.env.test_optimized_res={args.test_optimized_res} train.params.config.test_inst_tag={args.test_inst_tag}"
            preload_experience_config = f"train.params.config.preload_experiences_tf={args.preload_experiences_tf} train.params.config.preload_experiences_path={args.preload_experiences_path}"
            single_instance_training_config = f"train.params.config.single_instance_training={args.single_instance_training}"
            
            
            sampleds_with_object_code_fn_config = f"task.env.sampleds_with_object_code_fn={args.sampleds_with_object_code_fn}"
            
            
            if args.generalist_tune_all_instnaces:
                test_inst_config = f"task.env.test_inst_tag={cur_grab_data_tag} task.env.test_optimized_res={pre_optimized_traj}"
                single_instance_training_config = f"train.params.config.single_instance_training={True}"
                preload_experience_config = f"train.params.config.preload_experiences_tf={False} train.params.config.preload_experiences_path={''}"

            log_path_config = f"train.params.config.log_path={args.log_path}"
            train_dir_config = f"train.params.config.train_dir={args.log_path}"
            single_instance_tag_config = f"train.params.config.single_instance_tag={args.single_instance_tag}"
            obj_type_to_optimized_res_fn_config = f"train.params.config.obj_type_to_optimized_res_fn={args.obj_type_to_optimized_res_fn}"
            supervised_loss_coef_config = f"train.params.config.supervised_loss_coef={args.supervised_loss_coef}"
            pure_supervised_training_config = f"train.params.config.pure_supervised_training={args.pure_supervised_training}"
            inst_tag_to_latent_feature_fn_config = f"task.env.inst_tag_to_latent_feature_fn={args.inst_tag_to_latent_feature_fn}"
            object_type_to_latent_feature_fn_config = f"task.env.object_type_to_latent_feature_fn={args.object_type_to_latent_feature_fn}"
            obj_type_to_opt_res_config = f"task.env.grab_obj_type_to_opt_res_fn={args.grab_obj_type_to_opt_res_fn} task.env.taco_obj_type_to_opt_res_fn={args.taco_obj_type_to_opt_res_fn} train.params.config.grab_obj_type_to_opt_res_fn={args.grab_obj_type_to_opt_res_fn} train.params.config.taco_obj_type_to_opt_res_fn={args.taco_obj_type_to_opt_res_fn}"
            use_teacher_model_config = f"train.params.config.use_teacher_model={args.use_teacher_model}"
            bound_loss_coef_config = f"train.params.config.bounds_loss_coef={args.bound_loss_coef}"
            data_selection_ratio_config = f"task.env.data_selection_ratio={args.data_selection_ratio}"
            target_inst_tag_list_fn_config = f"task.env.target_inst_tag_list_fn={args.target_inst_tag_list_fn}"
            teacher_model_config = f"task.env.use_teacher_model={args.use_teacher_model} train.params.config.use_teacher_model={args.use_teacher_model} train.params.config.teacher_model_path={args.teacher_model_path} task.env.teacher_model_inst_tags_fn={args.teacher_model_inst_tags_fn}"
            multiple_teacher_model_config = f"train.params.config.teacher_index_to_weights={args.teacher_index_to_weights} task.env.teacher_index_to_inst_tags={args.teacher_index_to_inst_tags}"
            good_inst_opt_res_config = f"task.env.good_inst_opt_res={args.good_inst_opt_res}"
            w_franka_config = f"task.env.w_franka={args.w_franka} train.params.config.w_franka={args.w_franka}"
            randomize_config = f"task.task.randomize={args.randomize}"
            early_terminate_config = f"task.env.early_terminate={args.early_terminate}"
            substeps_config = f"task.sim.substeps={args.substeps}"
            # forcasting_model_config = f"task.env.use_forcasting_model={args.use_forcasting_model} task.env.forcasting_model_weights={args.forcasting_model_weights} task.env.forcasting_model_n_layers={args.forcasting_model_n_layers} task.env.w_glb_traj_feat_cond={args.w_glb_traj_feat_cond}"
            # use_window_future_selection_config = f"task.env.use_window_future_selection={args.use_window_future_selection}"
            # forcasting_inv_freq_config = f"task.env.forcasting_inv_freq={args.forcasting_inv_freq}"
            controller_setting_config = f"task.env.stiffness_coef={args.stiffness_coef} task.env.damping_coef={args.damping_coef} task.env.effort_coef={args.effort_coef}"
            
            
            grab_opt_res_config = f"task.env.grab_inst_tag_to_optimized_res_fn={args.grab_inst_tag_to_optimized_res_fn}"
            taco_opt_res_config = f"task.env.taco_inst_tag_to_optimized_res_fn={args.taco_inst_tag_to_optimized_res_fn}"
            maxx_inst_nn_config= f"task.env.maxx_inst_nn={args.maxx_inst_nn} train.params.config.maxx_inst_nn={args.maxx_inst_nn}"

            tracking_folder_info_config = f"task.env.tracking_save_info_fn={args.tracking_save_info_fn} task.env.tracking_info_st_tag={args.tracking_info_st_tag}"
            only_training_on_succ_samples_config = f"task.env.only_training_on_succ_samples={args.only_training_on_succ_samples}"
            
            use_strict_maxx_nn_ts_config = f"task.env.use_strict_maxx_nn_ts={args.use_strict_maxx_nn_ts}"
            taco_interped_data_sv_additional_tag_config = f"task.env.taco_interped_data_sv_additional_tag={args.taco_interped_data_sv_additional_tag}"
            strict_maxx_nn_ts_config=  f"task.env.strict_maxx_nn_ts={args.strict_maxx_nn_ts}"
            grab_train_test_setting_config = f"task.env.grab_train_test_setting={args.grab_train_test_setting}"
            use_local_canonical_state_config = f"task.env.use_local_canonical_state={args.use_local_canonical_state}"
                # rew_grab_thres, rew_taco_thres
            rew_thres_config = f"task.env.rew_grab_thres={args.rew_grab_thres} task.env.rew_taco_thres={args.rew_taco_thres}"
            rew_smoothness_coef_config = f"task.env.rew_smoothness_coef={args.rew_smoothness_coef}"
            obj_type_to_base_traj_fn_config = f"task.env.obj_type_to_base_traj_fn={args.obj_type_to_base_traj_fn}"
            use_base_traj_config = f"task.env.use_base_traj={args.use_base_traj}"
            # rew_thres_with_selected_insts, selected_inst_idxes_dict
            rew_thres_with_selected_insts_config = f"task.env.rew_thres_with_selected_insts={args.rew_thres_with_selected_insts}"
            selected_inst_idxes_dict_config = f"task.env.selected_inst_idxes_dict={args.selected_inst_idxes_dict}"
            customize_damping_config = f"task.env.customize_damping={args.customize_damping}"
            customize_global_damping_config = f"task.env.customize_global_damping={args.customize_global_damping}"
            train_on_all_trajs_config = f"task.env.train_on_all_trajs={args.train_on_all_trajs}"
            single_instance_state_based_train_config = f"task.env.single_instance_state_based_train={args.single_instance_state_based_train}"
            wo_vel_obs_config  = f"task.env.wo_vel_obs={args.wo_vel_obs}"
            downsample_config = f"task.env.downsample={args.downsample}"

            
            base_traj_config = ""
            
            
            history_setting_config = f"task.env.use_history_obs={args.use_history_obs} task.env.history_length={args.history_length} train.params.config.history_length={args.history_length}"
            w_obj_latent_features_config = f"task.env.w_obj_latent_features={args.w_obj_latent_features}"
            net_type_config = f"task.env.net_type={args.net_type}"
            history_freq_config = f"task.env.history_freq={args.history_freq}"
            use_future_obs_config = f"task.env.use_future_obs={args.use_future_obs}"
            w_history_window_index_config = f"task.env.w_history_window_index={args.w_history_window_index}"
            randomize_conditions_config = f"task.env.randomize_conditions={args.randomize_conditions}"
            w_inst_latent_features_config = f"task.env.w_inst_latent_features={args.w_inst_latent_features}"
            
            
            history_glbfeat_setting = f"task.env.history_window_size={args.history_window_size} task.env.glb_feat_per_skip={args.glb_feat_per_skip} task.env.centralize_info={args.centralize_info}"
            # random_shift_cond, random_shift_cond_freq, maxx_inv_cond_freq
            random_shift_conditions_setting = f"task.env.random_shift_cond={args.random_shift_cond} task.env.random_shift_cond_freq={args.random_shift_cond_freq} task.env.maxx_inv_cond_freq={args.maxx_inv_cond_freq}"
            only_use_hand_first_frame_config = f"task.env.only_use_hand_first_frame={args.only_use_hand_first_frame}"
            
            forecasting_agent_training_config = f"train.params.config.forecasting_obs_dim={args.forecasting_obs_dim} train.params.config.forecasting_act_dim={args.forecasting_act_dim} train.params.config.forecasting_nn_frames={args.forecasting_nn_frames}"
            use_world_model_config = f"train.params.config.use_world_model={args.use_world_model}"
            train_controller_config = f"train.params.config.train_controller={args.train_controller}"
            train_forecasting_model_config = f"train.params.config.train_forecasting_model={args.train_forecasting_model}"
            forecasting_model_weight_fn_config = f"train.params.config.forecasting_model_weight_fn={args.forecasting_model_weight_fn}"
            single_inst_tag_config = f"task.env.single_inst_tag={args.single_inst_tag}"
            activate_forecaster_config = f"task.env.activate_forecaster={args.activate_forecaster}"
            comput_reward_traj_hand_qpos_config = f"task.env.comput_reward_traj_hand_qpos={args.comput_reward_traj_hand_qpos}"
            use_future_ref_as_obs_goal_config = f"task.env.use_future_ref_as_obs_goal={args.use_future_ref_as_obs_goal}"
            forecast_obj_pos_config = f"task.env.forecast_obj_pos={args.forecast_obj_pos}"
            multiple_kine_source_trajs_fn_config = f"task.env.multiple_kine_source_trajs_fn={args.multiple_kine_source_trajs_fn}"
            use_multiple_kine_source_trajs_config = f"task.env.use_multiple_kine_source_trajs={args.use_multiple_kine_source_trajs}"
            include_obj_rot_in_obs_config = f"task.env.include_obj_rot_in_obs={args.include_obj_rot_in_obs}"
            compute_hand_rew_buf_threshold_config = f"task.env.compute_hand_rew_buf_threshold={args.compute_hand_rew_buf_threshold}"
            w_kine_retar_with_arm_config = f"task.env.load_kine_info_retar_with_arm={args.load_kine_info_retar_with_arm} task.env.kine_info_with_arm_sv_root={args.kine_info_with_arm_sv_root}"
            w_finger_pos_rew_config  = f"task.env.w_finger_pos_rew={args.w_finger_pos_rew}"
            franka_delta_delta_mult_coef_config = f"task.env.franka_delta_delta_mult_coef={args.franka_delta_delta_mult_coef}"
            control_arm_via_ik_config = f"task.env.control_arm_via_ik={args.control_arm_via_ik}"
            # warm_trans_actions_mult_coef, warm_rot_actions_mult_coef
            warm_actions_mult_coef_config = f"task.env.warm_trans_actions_mult_coef={args.warm_trans_actions_mult_coef} task.env.warm_rot_actions_mult_coef={args.warm_rot_actions_mult_coef}"
            hand_qpos_rew_coef_config = f"task.env.hand_qpos_rew_coef={args.hand_qpos_rew_coef}"
            ornt_rew_scheduling_config = f"task.env.schedule_ornt_rew_coef={args.schedule_ornt_rew_coef} task.env.lowest_ornt_rew_coef={args.lowest_ornt_rew_coef} task.env.highest_ornt_rew_coef={args.highest_ornt_rew_coef} task.env.ornt_rew_coef_warm_starting_steps={args.ornt_rew_coef_warm_starting_steps} task.env.ornt_rew_coef_increasing_steps={args.ornt_rew_coef_increasing_steps}"
            glb_mult_factor_scaling_config = f"task.env.hand_glb_mult_factor_scaling_coef={args.hand_glb_mult_factor_scaling_coef} task.env.hand_glb_mult_scaling_progress_after={args.hand_glb_mult_scaling_progress_after} task.env.hand_glb_mult_scaling_progress_before={args.hand_glb_mult_scaling_progress_before}"
            not_use_kine_bias_config = f"task.env.not_use_kine_bias={args.not_use_kine_bias}"
            disable_hand_obj_contact_config = f"task.env.disable_hand_obj_contact={args.disable_hand_obj_contact}"
            wo_fingertip_rot_vel_config = f"task.env.wo_fingertip_pos={args.wo_fingertip_pos} task.env.wo_fingertip_rot_vel={args.wo_fingertip_rot_vel} task.env.wo_fingertip_vel={args.wo_fingertip_vel}"
            arm_ctl_params_config = f"task.env.arm_stiffness={args.arm_stiffness} task.env.arm_effort={args.arm_effort} task.env.arm_damping={args.arm_damping}"
            # train_student_model, ts_teacher_model_obs_dim, ts_teacher_model_weights_fn
            ts_teacher_model_config = f"train.params.config.train_student_model={args.train_student_model} train.params.config.ts_teacher_model_obs_dim={args.ts_teacher_model_obs_dim} train.params.config.ts_teacher_model_weights_fn={args.ts_teacher_model_weights_fn} task.env.train_student_model={args.train_student_model} task.env.ts_teacher_model_obs_dim={args.ts_teacher_model_obs_dim}"
            randomize_setting_config = f"task.env.randomize_obj_init_pos={args.randomize_obj_init_pos} task.env.randomize_obs_more={args.randomize_obs_more}"
            obj_init_pos_rand_sigma_config = f"task.env.obj_init_pos_rand_sigma={args.obj_init_pos_rand_sigma}"
            obs_simplified_config = f"task.env.obs_simplified={args.obs_simplified}"
            w_traj_modifications_config = f"task.env.w_traj_modifications={args.w_traj_modifications}"
            # rand_obj_mass_lowest_range, rand_obj_mass_highest_range
            rand_obj_mass_range_config = f"task.task.rand_obj_mass_lowest_range={args.rand_obj_mass_lowest_range} task.task.rand_obj_mass_highest_range={args.rand_obj_mass_highest_range}"
            use_v2_leap_warm_urdf_config = f"task.env.use_v2_leap_warm_urdf={args.use_v2_leap_warm_urdf}"
            action_specific_rand_config = f"task.env.action_specific_randomizations={args.action_specific_randomizations} task.env.action_specific_rand_noise_scale={args.action_specific_rand_noise_scale}"
            hand_specific_randomizations_config = f"task.env.hand_specific_randomizations={args.hand_specific_randomizations}"
            schedule_hodist_coef_config = f"task.env.schedule_hodist_rew_coef={args.schedule_hodist_rew_coef} task.env.highest_rew_finger_obj_dist_coef={args.highest_rew_finger_obj_dist_coef} task.env.lowest_rew_finger_obj_dist_coef={args.lowest_rew_finger_obj_dist_coef} task.env.hodist_rew_coef_warm_starting_steps={args.hodist_rew_coef_warm_starting_steps} task.env.hodist_rew_coef_increasing_steps={args.hodist_rew_coef_increasing_steps}"
            reset_obj_mass_config = f"task.env.reset_obj_mass={args.reset_obj_mass} task.env.obj_mass_reset={args.obj_mass_reset} task.env.recompute_inertia={args.recompute_inertia}"
            use_vision_obs_config = f"task.env.use_vision_obs={args.use_vision_obs}"
            w_rotation_axis_rew_config = f"task.env.w_rotation_axis_rew={args.w_rotation_axis_rew}"
            add_physical_params_in_obs_config = f"task.env.add_physical_params_in_obs={args.add_physical_params_in_obs}"
            whether_randomize_obs_act_config = f"task.env.whether_randomize_obs_act={args.whether_randomize_obs_act} task.env.whether_randomize_obs={args.whether_randomize_obs} task.env.whether_randomize_act={args.whether_randomize_act}"
            obs_rand_noise_scale_config = f"task.env.obs_rand_noise_scale={args.obs_rand_noise_scale}"
            dagger_style_training_config = f"train.params.config.dagger_style_training={args.dagger_style_training} train.params.config.rollout_teacher_model={args.rollout_teacher_model} train.params.config.rollout_student_model={args.rollout_student_model}"
            teacher_subj_idx_config = f"train.params.config.teacher_subj_idx={args.teacher_subj_idx}"
            action_chunking_config = f"task.env.action_chunking={args.action_chunking} task.env.action_chunking_frames={args.action_chunking_frames} train.params.config.action_chunking={args.action_chunking} train.params.config.action_chunking_frames={args.action_chunking_frames}"

            cur_horizon_length = args.horizon_length
            horizon_length_config = f"train.params.config.horizon_length={cur_horizon_length}"
            bc_style_training_config = f"train.params.config.bc_style_training={args.bc_style_training} task.env.bc_style_training={args.bc_style_training}"
            demonstration_tuning_config = f"train.params.config.demonstration_tuning_model={args.demonstration_tuning_model} train.params.config.demonstration_tuning_model_freq={args.demonstration_tuning_model_freq}"
            bc_relative_targets_config = f"train.params.config.bc_relative_targets={args.bc_relative_targets}"
            distill_full_to_partial_config = f"task.env.distill_full_to_partial={args.distill_full_to_partial}"
            train_free_hand_config = f"task.env.train_free_hand={args.train_free_hand}"
            simreal_modeling_config = f"task.env.simreal_modeling={args.simreal_modeling} train.params.config.simreal_modeling={args.simreal_modeling}"
            more_allegro_stiffness_config = f"task.env.more_allegro_stiffness={args.more_allegro_stiffness}"
            distinguish_kine_with_base_traj_config = f"task.env.distinguish_kine_with_base_traj={args.distinguish_kine_with_base_traj}"
            distill_delta_targets_config = f"train.params.config.distill_delta_targets={args.distill_delta_targets}"
            preset_cond_type_config = f"task.env.preset_cond_type={args.preset_cond_type}"
            preload_all_saved_exp_buffers_config = f"train.params.config.preload_all_saved_exp_buffers={args.preload_all_saved_exp_buffers}"
            use_actual_prev_targets_in_obs_config = f"task.env.use_actual_prev_targets_in_obs={args.use_actual_prev_targets_in_obs}"
            use_no_obj_pose_config = f"train.params.config.use_no_obj_pose={args.use_no_obj_pose} task.env.use_no_obj_pose={args.use_no_obj_pose}"
            multi_traj_use_joint_order_in_sim_config = f"task.env.multi_traj_use_joint_order_in_sim={args.multi_traj_use_joint_order_in_sim}"
            action_chunking_skip_frames_config = f"train.params.config.action_chunking_skip_frames={args.action_chunking_skip_frames}"
            multi_inst_chunking_config = f"task.env.multi_inst_chunking={args.multi_inst_chunking}"
            add_obj_features_config = f"train.params.config.add_obj_features={args.add_obj_features} task.env.add_obj_features={args.add_obj_features}"
            kine_ed_tag_config = f"task.env.kine_ed_tag={args.kine_ed_tag}"
            only_rot_axis_guidance_config = f"task.env.only_rot_axis_guidance={args.only_rot_axis_guidance}"
            add_penalty_config = f"task.env.add_global_motion_penalty={args.add_global_motion_penalty} task.env.add_torque_penalty={args.add_torque_penalty} task.env.add_work_penalty={args.add_work_penalty}"
            use_multi_step_control_config = f"task.env.use_multi_step_control={args.use_multi_step_control} task.env.nn_control_substeps={args.nn_control_substeps}"
            schedule_episod_length_config = f"task.env.schedule_episod_length={args.schedule_episod_length} task.env.episod_length_low={args.episod_length_low} task.env.episod_length_high={args.episod_length_high} task.env.episod_length_warming_up_steps={args.episod_length_warming_up_steps} task.env.episod_length_increasing_steps={args.episod_length_increasing_steps}"
            use_actual_traj_length_config = f"task.env.use_actual_traj_length={args.use_actual_traj_length}"
            randomize_reset_frame_config = f"task.env.randomize_reset_frame={args.randomize_reset_frame}"
            add_forece_obs_config = f"task.env.add_forece_obs={args.add_forece_obs}"
            distill_via_bc_config = f"task.env.distill_via_bc={args.distill_via_bc} train.params.config.distill_via_bc={args.distill_via_bc}"
            record_experiences_config = f"train.params.config.record_experiences={args.record_experiences}"
            learning_rate_config = f"train.params.config.learning_rate={args.learning_rate}"
            forecasting_obs_with_original_obs_config = f"task.env.forecasting_obs_with_original_obs={args.forecasting_obs_with_original_obs} train.params.config.forecasting_obs_with_original_obs={args.forecasting_obs_with_original_obs}"
            glb_action_penalty_schedule_config = f"task.env.schedule_glb_action_penalty={args.schedule_glb_action_penalty} task.env.glb_penalty_low={args.glb_penalty_low} task.env.glb_penalty_high={args.glb_penalty_high} task.env.glb_penalty_warming_up_steps={args.glb_penalty_warming_up_steps} task.env.glb_penalty_increasing_steps={args.glb_penalty_increasing_steps}"
            hand_targets_smooth_config = f"task.env.add_hand_targets_smooth={args.add_hand_targets_smooth} task.env.hand_targets_smooth_coef={args.hand_targets_smooth_coef}" # hand targets smooth coef 
            preset_multi_traj_index_config = f"task.env.preset_multi_traj_index={args.preset_multi_traj_index}"
            save_experiences_via_ts_config = f"train.params.config.save_experiences_via_ts={args.save_experiences_via_ts}"
            load_experiences_maxx_ts_config = f"train.params.config.load_experiences_maxx_ts={args.load_experiences_maxx_ts}"
            # switch_between_models, switch_to_trans_model_frame_after, switch_to_trans_model_ckpt_fn
            switch_between_models_config = f"train.params.config.switch_between_models={args.switch_between_models} train.params.config.switch_to_trans_model_frame_after={args.switch_to_trans_model_frame_after} train.params.config.switch_to_trans_model_ckpt_fn={args.switch_to_trans_model_ckpt_fn} task.env.switch_between_models={args.switch_between_models} task.env.switch_to_trans_model_frame_after={args.switch_to_trans_model_frame_after} task.env.switch_to_trans_model_ckpt_fn={args.switch_to_trans_model_ckpt_fn}"
            traj_idx_to_experience_sv_folder_config = f"train.params.config.traj_idx_to_experience_sv_folder={args.traj_idx_to_experience_sv_folder}"
            load_chunking_experiences_v2_config  = f"train.params.config.load_chunking_experiences_v2={args.load_chunking_experiences_v2} task.env.load_chunking_experiences_v2={args.load_chunking_experiences_v2}"
            history_chunking_obs_version_config = f"train.params.config.history_chunking_obs_version={args.history_chunking_obs_version} task.env.history_chunking_obs_version={args.history_chunking_obs_version}"
            load_chunking_experiences_from_real_config = f"train.params.config.load_chunking_experiences_from_real={args.load_chunking_experiences_from_real}"
            closed_loop_to_real_config = f"task.env.closed_loop_to_real={args.closed_loop_to_real}"
            use_transformer_model_config = f"train.params.config.use_transformer_model={args.use_transformer_model}"
            
            enableCameraSensors = False
        
        
        cuda_idxx = 0
        
        
        cmd = f"{cuda_visible_text} python train.py task={task_type} train={train_type} sim_device='cuda:{cuda_idxx}' rl_device='cuda:{cuda_idxx}'  capture_video={capture_video} force_render={force_render} headless={args.headless}   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={args.test} task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={args.checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.controlFrequencyInv={args.controlFrequencyInv} task.env.episodeLength={args.episodeLength} task.env.start_frame={args.start_frame} task.env.rew_obj_pose_coef={args.rew_obj_pose_coef} task.env.goal_dist_thres={args.goal_dist_thres} task.env.lifting_separate_stages={args.lifting_separate_stages} task.env.strict_lifting_separate_stages={args.strict_lifting_separate_stages} task.env.table_z_dim={args.table_z_dim} task.env.add_table={args.add_table} exp_dir={exp_dir} task.env.use_taco_obj_traj={args.use_taco_obj_traj} task.env.pre_optimized_traj={ cur_pre_optimized_traj } task.env.hand_type={ args.hand_type } enableCameraSensors={enableCameraSensors} graphics_device_id={cuda_idx} task.env.use_hand_actions_rew={args.use_hand_actions_rew} task.env.supervised_training={args.supervised_training} {training_mode_config} {test_inst_config} {preload_experience_config} {single_instance_training_config} {sampleds_with_object_code_fn_config} {log_path_config} {train_dir_config} {grab_opt_res_config}  {taco_opt_res_config} {single_instance_tag_config} {obj_type_to_optimized_res_fn_config} {supervised_loss_coef_config} {pure_supervised_training_config} {inst_tag_to_latent_feature_fn_config} {object_type_to_latent_feature_fn_config} {obj_type_to_opt_res_config} {maxx_inst_nn_config} {tracking_folder_info_config} {only_training_on_succ_samples_config} {use_teacher_model_config} {use_strict_maxx_nn_ts_config} {taco_interped_data_sv_additional_tag_config} {strict_maxx_nn_ts_config} {grab_train_test_setting_config} {use_local_canonical_state_config} {bound_loss_coef_config} {rew_thres_config} {rew_smoothness_coef_config} {obj_type_to_base_traj_fn_config} {use_base_traj_config} {rew_thres_with_selected_insts_config} {selected_inst_idxes_dict_config} {customize_damping_config} {customize_global_damping_config} {train_on_all_trajs_config} {single_instance_state_based_train_config} {data_selection_ratio_config} {wo_vel_obs_config} {downsample_config} {target_inst_tag_list_fn_config} {teacher_model_config} {multiple_teacher_model_config} {base_traj_config} {history_setting_config} {good_inst_opt_res_config} {w_franka_config} {randomize_config} {early_terminate_config} {substeps_config} {controller_setting_config}  {w_obj_latent_features_config} {net_type_config} {history_freq_config} {use_future_obs_config} {w_history_window_index_config} {randomize_conditions_config} {w_inst_latent_features_config} {history_glbfeat_setting} {random_shift_conditions_setting} {only_use_hand_first_frame_config} {forecasting_agent_training_config} {use_world_model_config} {train_controller_config} {train_forecasting_model_config} {forecasting_model_weight_fn_config} {single_inst_tag_config} {activate_forecaster_config} {comput_reward_traj_hand_qpos_config} {use_future_ref_as_obs_goal_config} {forecast_obj_pos_config} {multiple_kine_source_trajs_fn_config} {use_multiple_kine_source_trajs_config} {include_obj_rot_in_obs_config} {compute_hand_rew_buf_threshold_config} {w_kine_retar_with_arm_config} {w_finger_pos_rew_config} {franka_delta_delta_mult_coef_config} {control_arm_via_ik_config} {warm_actions_mult_coef_config} {hand_qpos_rew_coef_config} {ornt_rew_scheduling_config} {glb_mult_factor_scaling_config} {not_use_kine_bias_config} {disable_hand_obj_contact_config} {wo_fingertip_rot_vel_config} {arm_ctl_params_config} {ts_teacher_model_config} {randomize_setting_config} {obj_init_pos_rand_sigma_config} {obs_simplified_config} {w_traj_modifications_config} {rand_obj_mass_range_config} {use_v2_leap_warm_urdf_config} {action_specific_rand_config} {hand_specific_randomizations_config} {schedule_hodist_coef_config} {reset_obj_mass_config} {use_vision_obs_config} {w_rotation_axis_rew_config} {add_physical_params_in_obs_config} {whether_randomize_obs_act_config} {obs_rand_noise_scale_config} {dagger_style_training_config} {teacher_subj_idx_config} {action_chunking_config} {horizon_length_config} {bc_style_training_config} {demonstration_tuning_config} {bc_relative_targets_config} {distill_full_to_partial_config} {train_free_hand_config} {simreal_modeling_config} {more_allegro_stiffness_config} {distinguish_kine_with_base_traj_config} {distill_delta_targets_config} {preset_cond_type_config} {preload_all_saved_exp_buffers_config} {use_actual_prev_targets_in_obs_config} {use_no_obj_pose_config} {multi_traj_use_joint_order_in_sim_config} {action_chunking_skip_frames_config} {multi_inst_chunking_config} {add_obj_features_config} {kine_ed_tag_config} {only_rot_axis_guidance_config} {add_penalty_config} {use_multi_step_control_config} {schedule_episod_length_config} {use_actual_traj_length_config} {randomize_reset_frame_config} {add_forece_obs_config} {distill_via_bc_config} {record_experiences_config} {learning_rate_config} {forecasting_obs_with_original_obs_config} {glb_action_penalty_schedule_config} {hand_targets_smooth_config} {preset_multi_traj_index_config} {save_experiences_via_ts_config} {load_experiences_maxx_ts_config} {switch_between_models_config} {traj_idx_to_experience_sv_folder_config} {load_chunking_experiences_v2_config} {history_chunking_obs_version_config} {load_chunking_experiences_from_real_config} {closed_loop_to_real_config} {use_transformer_model_config}"     
            
        print(cmd)
        os.system(cmd)
    
    
    tracking_data_sv_root = args.tracking_data_sv_root
    
    
    
    
    if args.dataset_type == 'grab':
        starting_str = "passive_active_info_ori_grab_"
        passive_active_info_tag = "passive_active_info_"
        
        if args.hand_type == 'leap':
            starting_str = "leap_" + starting_str
            passive_active_info_tag = "leap_" + passive_active_info_tag
        
        tot_tracking_data = os.listdir(tracking_data_sv_root)
        # if args.num_frames == 150:
        #     tot_tracking_data = [fn for fn in tot_tracking_data if fn[: len(starting_str)] == starting_str and fn.endswith(".npy") and "_nf_" not in fn]
        # else:
        nf_tag = f"_nf_{args.num_frames}"
        tot_tracking_data = [fn for fn in tot_tracking_data if fn[: len(starting_str)] == starting_str and fn.endswith(".npy") and nf_tag in fn]
        
        if len(args.subj_nm) > 0:
            subj_tag = f"_{args.subj_nm}_"
            tot_tracking_data = [fn for fn in tot_tracking_data if subj_tag in fn]
    elif args.dataset_type == 'taco':
        taso_inst_st_flag = 'taco_'
        # TODO: move such meshes into the ../assets folder
        mesh_sv_root = "../assets/meshdatav3_scaled/sem"
        # if not os.path.exists(mesh_sv_root):
        #     mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        tot_mesh_folders = os.listdir(mesh_sv_root)
        # find meshes directly #
        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        
        
        # if args.eval_split_trajs:
        #     test_taco_tag = args.test_taco_tag
        #     tot_mesh_folders = [fn for fn in tot_mesh_folders if test_taco_tag in fn]
        
        # modified_tag = "_modifed"
        # interped_tag = "_interped"
        # find tracking data
        tot_tracking_data = tot_mesh_folders
        passive_active_info_tag = ''
    # 
    else: # value error #
        raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}")
    
    nn_gpus = args.nn_gpus
    
    gpu_idx_list = [1, 2, 3, 4, 5, 6, 7]
    # gpu_idx_list = [0, 1, 2, 3, 4, 5, 6, 7]
    nn_gpus = len(gpu_idx_list)
    
    ### and also add the grab instance to the optimized res dict
    # pre_load_trajectories, obj_type_to_pre_optimized_traj
    # pre_load_trajectories = args.pre_load_trajectories
    # print(f"pre_load_trajectories: {pre_load_trajectories}")
    # if pre_load_trajectories:
    #     obj_type_to_pre_optimized_traj = args.obj_type_to_pre_optimized_traj 
    #     assert len(obj_type_to_pre_optimized_traj) > 0 and os.path.exists(obj_type_to_pre_optimized_traj)
    #     obj_type_to_pre_optimized_traj = np.load(obj_type_to_pre_optimized_traj, allow_pickle=True).item()
    #     ## ## obj type to pre optimized traj ## ##
    # else:  ## obj type to pre optimized traj ## ##
    obj_type_to_pre_optimized_traj = None
    ## obj type to pre optimized traj ##
    print(f"launch_type: {args.launch_type}")
    
    
    
    if not args.generalist_tune_all_instnaces:
        tot_tracking_data = tot_tracking_data[:10]
    
    ### tot tracking data ### ### tracking data ###
    ### tot tracking data ### ### tracking data ###
    print(f"tot_tracking_data: {tot_tracking_data}")
    
    if args.launch_type == 'trajectory':
        tot_grab_data_tag = []
        for cur_tracking_data in tot_tracking_data:
            cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
            print(f"cur_grab_data_tag: {cur_grab_data_tag}")
            
            if exclude_inst_tag_to_opt_res is not None and cur_grab_data_tag in exclude_inst_tag_to_opt_res:
                print(f"[Info] Trained. Continue...")
                continue
            
            # if cur_grab_data_tag != 'ori_grab_s6_torusmedium_inspect_1_nf_300':
            #     continue
            
            traj_grab_data_tag = cur_grab_data_tag
            
            # if pure_obj_type_to_optimized_res and args.rew_filter:
            #     if cur_grab_data_tag not in pure_obj_type_to_optimized_res:
            #         continue
            
            #     cur_obj_rew = pure_obj_type_to_optimized_res[cur_grab_data_tag][0]
            #     print(f"cur_grab_data_tag: {cur_grab_data_tag}, cur_obj_rew: {cur_obj_rew}")
            #     if cur_obj_rew > args.rew_low_threshold:
            #         continue
            #     print(f"cur_grab_data_tag: {cur_grab_data_tag}, cur_obj_rew: {cur_obj_rew}")
            
            # obj #
            if obj_type_to_pre_optimized_traj is not None:
                key_of_opt_traj = list(obj_type_to_pre_optimized_traj.keys())[0]
                
                if isinstance(key_of_opt_traj, tuple):
                    if 'taco' in cur_grab_data_tag:
                        cur_grab_data_tag_key = (cur_grab_data_tag, 'ori_grab_s2_phone_call_1')
                    else:
                        cur_grab_data_tag_key = ( cur_grab_data_tag, cur_grab_data_tag )
                    # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
                else: # grab data kye #
                    cur_grab_data_tag_key = cur_grab_data_tag
                    # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
                    
                
                if cur_grab_data_tag_key not in obj_type_to_pre_optimized_traj:
                    if args.train_on_all_trajs:
                        cur_pre_optimized_traj = ["/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1.npy"]
                    else:
                        continue
                else:
                    cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag_key ]
                # if isinstance(key_of_opt_traj, tuple): # grab data tag key #
                #     cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
                # else:
                #     cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
                cur_pre_optimized_traj = cur_pre_optimized_traj[0] # get the pre optimized traj #
                # 
                cur_pre_optimized_traj_sorted = cur_pre_optimized_traj.replace(".npy", "_sorted.npy")
                cur_pre_optimized_traj_sorted_best = cur_pre_optimized_traj_sorted.replace(".npy", "_best.npy")
                if not os.path.exists(cur_pre_optimized_traj_sorted_best):
                    continue
                cur_pre_optimized_traj = cur_pre_optimized_traj_sorted_best
            else:
                cur_pre_optimized_traj = None
            
            # cur grad data tag; #
            print(f"cur_grab_data_tag: {cur_grab_data_tag}, cur_pre_optimized_traj: {cur_pre_optimized_traj}")
            cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
            
            cur_cuda_idx = gpu_idx_list[cur_cuda_idx]
            
            tot_grab_data_tag.append(
                [cur_grab_data_tag, traj_grab_data_tag, cur_cuda_idx, cur_pre_optimized_traj]
            )
            
            if args.debug:
                break
    
    
    elif args.launch_type == 'trajectory_baseline_search':
        assert len(args.obj_type_to_base_trajs_config) > 0 and os.path.exists(args.obj_type_to_base_trajs_config)
        obj_type_to_base_trajs_config = np.load(args.obj_type_to_base_trajs_config, allow_pickle=True).item()
        # obj type to base trajs config #
        tot_grab_data_tag = []
        for cur_child_grab_data_tag in obj_type_to_base_trajs_config:
            tot_fa_data_tags = obj_type_to_base_trajs_config[cur_child_grab_data_tag]
            
            # i_fa = 0
            for cur_fa_data_tag in tot_fa_data_tags:
                # cur_fa_data_tag, cur_fa_traj_fn = cur_fa_data_tag_pair
                cur_fa_traj_fn = tot_fa_data_tags[cur_fa_data_tag]
                if not cur_fa_traj_fn.endswith("_sorted_best.npy"):
                    cur_fa_traj_fn = cur_fa_traj_fn.replace(".npy", "_sorted_best.npy")
                print(f"cur_grab_data_tag: {cur_child_grab_data_tag}, tot_fa_data_tags: {tot_fa_data_tags}, cur_fa_traj_fn: {cur_fa_traj_fn}")
                cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
                
                cur_cuda_idx = gpu_idx_list[cur_cuda_idx]
                
                tot_grab_data_tag.append(
                    [cur_child_grab_data_tag, cur_fa_data_tag, cur_cuda_idx, cur_fa_traj_fn]
                )
                
            
    elif args.launch_type == 'object_type':
        tot_grab_data_tag = []
        for cur_tracking_data in tot_tracking_data:
            cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
            traj_grab_data_tag = cur_grab_data_tag
            
            if '_nf_' in cur_grab_data_tag:
                pure_obj_type = cur_grab_data_tag.split('_nf_')[0] #
            else:
                pure_obj_type = cur_grab_data_tag
            cur_idx = data_nm_to_idx[pure_obj_type]
            # ori_grab_sx_xxx # 
            tot_data_names = list(data_nm_to_idx.keys())
            print(f"pure_obj_type: {pure_obj_type}")
            cur_obj_name = pure_obj_type.split("_")[3]
            grab_obj_idx_dict_fn = f"../assets/grab_obj_name_idx_dict.npy"
            grab_obj_idx_dict = np.load(grab_obj_idx_dict_fn, allow_pickle=True).item()
            grab_obj_nm_to_idx = grab_obj_idx_dict['grab_obj_name_to_idx']
            cru_obj_idx = grab_obj_nm_to_idx[cur_obj_name] # get the current object index #
            cur_obj_sorted_nei_idxes = find_similar_objs(cru_obj_idx)
            cur_obj_sorted_nei_names = [grab_obj_idx_dict['grab_idx_to_obj_name'][idx] for idx in cur_obj_sorted_nei_idxes]
            for i_obj, nei_obj_name in enumerate(cur_obj_sorted_nei_names):
                pure_nei_obj_name = None 
                for cur_candi_pure_obj_name in tot_data_names:
                    if nei_obj_name in cur_candi_pure_obj_name:
                        pure_nei_obj_name = cur_candi_pure_obj_name
                        break
                if pure_nei_obj_name is None:
                    continue
                cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
                tot_grab_data_tag.append(
                    [pure_nei_obj_name, traj_grab_data_tag, cur_cuda_idx, None]
                )
    else:
        raise ValueError(f"Launch type {args.launch_type} not supported")
    
    print(f"tot_tracking_data : {tot_tracking_data}")
    
    
    
    tot_grab_data_tag = tot_grab_data_tag[args.st_idx: ]
    
    if args.debug:
        tot_grab_data_tag = tot_grab_data_tag[:1]
    
    
    ######## Single isntance training setting ########
    if (not args.generalist_tune_all_instnaces) and (args.data_inst_flag is not None) and len(args.data_inst_flag) > 0:
        data_inst_flag = args.data_inst_flag
        cur_cuda_idx = args.st_idx
        if obj_type_to_pre_optimized_traj is not None:
            # key_of_opt_traj = obj_type_to_pre_optimized_traj.keys()[0]
            key_of_opt_traj = list(obj_type_to_pre_optimized_traj.keys())[0]
            if isinstance(key_of_opt_traj, tuple):
                cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ (data_inst_flag, data_inst_flag) ]
            else:
                cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ data_inst_flag ]
            # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ (data_inst_flag, data_inst_flag) ]
            # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ data_inst_flag ]
            cur_pre_optimized_traj = cur_pre_optimized_traj[0] 
        else:
            cur_pre_optimized_traj = None
        tot_grab_data_tag = [
            [data_inst_flag, data_inst_flag, cur_cuda_idx, cur_pre_optimized_traj]
        ]

    max_pool_size = nn_gpus * 1
    
    for i_st in range(0, len(tot_grab_data_tag), max_pool_size):
        i_ed = i_st + max_pool_size
        i_ed = min(i_ed, len(tot_grab_data_tag))
        cur_batch_grab_data_tags = tot_grab_data_tag[i_st: i_ed]
        
        cur_thread_processes = []
        
        for cur_grab_data_tag in cur_batch_grab_data_tags:
            # existing = judge_whether_trained(tot_tracking_logs, cur_grab_data_tag)
            # if existing:
            #     print(f" cur_grab_data_tag: {cur_grab_data_tag} has been trained")
            #     continue
            cur_thread_processes.append(
                Process(target=launch_one_process, args=(cur_grab_data_tag))
            )
            # 
            cur_thread_processes[-1].start()
        for p in cur_thread_processes:
            p.join()
    

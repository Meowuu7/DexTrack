# train.py
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

# import hydra
# from isaacgymenvs.learning import calm_agent, calm_models, calm_network_builder, calm_players
# from isaacgymenvs.learning import encamp_network_builder, encamp_agent
# from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank
# from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
# from omegaconf import DictConfig, OmegaConf
# from hydra.utils import to_absolute_path
# from isaacgymenvs.tasks import isaacgym_task_map
# import gym

# from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
# from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# import torch
# import numpy as np
# import random

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
    # parser.add_argument("--additional_tag",  type=str, default='', help="path to the optimized qtars")
    parser.add_argument("--hand_type",  type=str, default='allegro', help="path to the optimized qtars")
    
    
    ##### isaacgym settings #####
    parser.add_argument("--numEnvs", type=int, default=8000)
    parser.add_argument("--minibatch_size", type=int, default=8000)
    parser.add_argument("--use_relative_control", type=str2bool,  default=False)
    parser.add_argument("--goal_cond", type=str2bool, default=False)
    # parser.add_argument("--object_name", type=str, default='')
    parser.add_argument("--obs_type", type=str, default='pure_state_wref_wdelta')
    parser.add_argument("--rigid_obj_density", type=float, default=500)
    parser.add_argument("--glb_trans_vel_scale", type=float, default=0.1)
    parser.add_argument("--glb_rot_vel_scale", type=float, default=0.1)
    # export additiona_tag="kinebais_wdelta_rewhandpos_dist_"
    parser.add_argument("--additional_tag", type=str, default='kinebais_wdelta_rewhandpos_dist_')
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
    parser.add_argument("--target_object_name", type=str, default='')
    parser.add_argument("--target_mocap_sv_fn", type=str, default='')
    parser.add_argument("--use_taco_obj_traj", type=str2bool,   default=True)
    parser.add_argument("--pre_optimized_traj", type=str, default='')
    ### TODO: add pre_load_trajectories, obj_type_to_pre_optimized_traj ###
    parser.add_argument("--pre_load_trajectories", type=str2bool,   default=False)
    parser.add_argument("--obj_type_to_pre_optimized_traj", type=str, default='')
    parser.add_argument("--subj_idx", type=int, default=0)
    # parser.add_argument("--hand_type", type=str, default='allegro')
    parser.add_argument("--use_vision", type=str2bool,   default=False)
    parser.add_argument("--use_dagger", type=str2bool,   default=False)
    parser.add_argument("--use_generalist_policy", type=str2bool,   default=False)
    parser.add_argument("--use_hand_actions_rew", type=str2bool,   default=True)
    # supervised_training
    parser.add_argument("--supervised_training", type=str2bool,   default=False)
    # checkpoint
    parser.add_argument("--test_inst_tag", type=str, default='')
    parser.add_argument("--test_optimized_res", type=str, default='')
    parser.add_argument("--training_mode", type=str, default='regular')
    # preload_experiences_tf, preload_experiences_path
    parser.add_argument("--preload_experiences_tf", type=str2bool,   default=False)
    parser.add_argument("--preload_experiences_path", type=str,   default=None)
    # single_instance_training
    parser.add_argument("--single_instance_training", type=str2bool,   default=False)
    # parser.add_argument("--checkpoint", type=str, default='') # 
    parser.add_argument("--generalist_tune_all_instnaces", type=str2bool,   default=False)
    # sampleds_with_object_code_fn
    parser.add_argument("--sampleds_with_object_code_fn", type=str, default='')
    # log_path
    parser.add_argument("--log_path", type=str, default='./runs')
    # grab_inst_tag_to_optimized_res_fn: '/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy'
    # grab_inst_tag_to_optimized_res_fn, taco_inst_tag_to_optimized_res_fn
    parser.add_argument("--grab_inst_tag_to_optimized_res_fn", type=str, default='/root/diffsim/softzoo/softzoo/diffusion/assets/data_inst_tag_to_optimized_res.npy')
    parser.add_argument("--taco_inst_tag_to_optimized_res_fn", type=str, default='')
    # single_instance_tag # obj_type_to_optimized_res_fn #
    parser.add_argument("--single_instance_tag", type=str, default='')
    parser.add_argument("--obj_type_to_optimized_res_fn", type=str, default='')
    # supervised_loss_coef
    parser.add_argument("--supervised_loss_coef", type=float, default=0.0005)
    # pure_supervised_training
    parser.add_argument("--pure_supervised_training", type=str2bool,   default=False)
    # inst_tag_to_latent_feature_fn
    parser.add_argument("--inst_tag_to_latent_feature_fn", type=str, default='')
    parser.add_argument("--object_type_to_latent_feature_fn", type=str, default='')
    parser.add_argument("--exclude_inst_tag_to_opt_res_fn", type=str, default='')
    # tracking_save_info_fn
    parser.add_argument("--tracking_save_info_fn", type=str, default='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data')
    # taco_interped_data_sv_additional_tag
    parser.add_argument("--taco_interped_data_sv_additional_tag", type=str, default='')
    parser.add_argument("--use_strict_maxx_nn_ts", type=str2bool,  default=False)
    parser.add_argument("--strict_maxx_nn_ts", type=int, default=150)
    # use_local_canonical_state
    parser.add_argument("--use_local_canonical_state", type=str2bool,  default=False)
    parser.add_argument("--obj_type_to_ckpt_fn", type=str, default='')
    # use_base_traj
    parser.add_argument("--use_base_traj", type=str2bool,  default=False)
    parser.add_argument("--obj_type_to_base_trajs", type=str, default='')
    # customize_damping
    parser.add_argument("--customize_damping", type=str2bool,  default=False)
    # tracking_info_st_tag
    parser.add_argument("--tracking_info_st_tag", type=str, default='passive_active_info_')
    # train_on_all_trajs
    parser.add_argument("--train_on_all_trajs", type=str2bool,  default=False)
    parser.add_argument("--test_on_taco_test_set", type=str2bool,  default=False)
    # single_instance_state_based_train
    parser.add_argument("--single_instance_state_based_train", type=str2bool,  default=False)
    # downsample
    parser.add_argument("--downsample", type=str2bool,  default=False)
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
    # w_franka
    parser.add_argument("--w_franka", type=str2bool,  default=False)
    # randomize
    parser.add_argument("--randomize", type=str2bool,  default=False)
    
    parser.add_argument("--use_history_obs", type=str2bool,   default=False)
    parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--use_future_obs", type=str2bool, default=False)
    
    # use_forcasting_model, forcasting_model_weights, forcasting_model_n_layers, w_glb_traj_feat_cond #
    parser.add_argument("--use_forcasting_model", type=str2bool,   default=False)
    parser.add_argument("--forcasting_model_weights", type=str,   default='')
    parser.add_argument("--forcasting_model_n_layers", type=int, default=7)
    # w_glb_traj_feat_cond #
    parser.add_argument("--w_glb_traj_feat_cond", type=str2bool, default=False)
    # substeps
    parser.add_argument("--substeps", type=int, default=2)
    parser.add_argument("--use_window_future_selection", type=str2bool, default=False)
    parser.add_argument("--w_history_window_index", type=str2bool, default=False)
    parser.add_argument("--single_instance_test_tag", type=str,   default='')
    parser.add_argument("--w_obj_latent_features", type=str2bool, default=True)
    parser.add_argument("--forcasting_history_ws", type=int, default=60)
    parser.add_argument("--forcasting_inv_freq", type=int, default=60)
    parser.add_argument("--randomize_conditions", type=str2bool, default=False)
    # parser.add_argument("--history_length", type=int, default=5)
    parser.add_argument("--history_freq", type=int, default=1)
    parser.add_argument("--randomize_condition_type", type=str,   default='random')
    # forcasting_diffusion_model
    parser.add_argument("--forcasting_diffusion_model", type=str2bool, default=False)
    
    parser.add_argument("--stiffness_coef", type=float,   default=100.0)
    parser.add_argument("--damping_coef", type=float,   default=4.0)
    parser.add_argument("--effort_coef", type=float,   default=0.95)
    # partial_obj_info
    parser.add_argument("--partial_obj_info", type=str2bool, default=False)
    parser.add_argument("--partial_hand_info", type=str2bool, default=False)
    # partial_obj_pos_info
    parser.add_argument("--partial_obj_pos_info", type=str2bool, default=False)
    # use_partial_to_complete_model, partial_to_complete_model_weights
    parser.add_argument("--use_partial_to_complete_model", type=str2bool, default=False)
    parser.add_argument("--partial_to_complete_model_weights", type=str,   default='')
    parser.add_argument("--add_contact_conditions", type=str2bool, default=False)
    # contact_info_sv_root
    parser.add_argument("--contact_info_sv_root", type=str,   default='')
    # st_ed_state_cond
    parser.add_argument("--st_ed_state_cond", type=str2bool, default=False)
    # 
    # --history_window_size=${history_window_size} --glb_feat_per_skip=${glb_feat_per_skip} --centralize_info=${centralize_info}
    parser.add_argument("--history_window_size", type=int, default=60)
    parser.add_argument("--glb_feat_per_skip", type=int, default=1)
    parser.add_argument("--centralize_info", type=str2bool, default=False)
    parser.add_argument("--hist_cond_partial_hand_info", type=str2bool, default=False)
    parser.add_argument("--hist_cond_partial_obj_info", type=str2bool, default=False)
    parser.add_argument("--hist_cond_partial_obj_pos_info", type=str2bool, default=False)
    # preset_cond_type
    parser.add_argument("--preset_cond_type", type=int, default=0)
    # preset_inv_cond_freq
    parser.add_argument("--preset_inv_cond_freq", type=int, default=1)
    # random_shift_cond, random_shift_cond_freq, maxx_inv_cond_freq
    parser.add_argument("--random_shift_cond", type=str2bool, default=False)
    parser.add_argument("--random_shift_cond_freq", type=str2bool, default=False)
    parser.add_argument("--maxx_inv_cond_freq", type=int, default=30) 
    # use_clip_glb_features
    parser.add_argument("--use_clip_glb_features", type=str2bool, default=False)
    # forecasting_model_inv_freq
    parser.add_argument("--forecasting_model_inv_freq", type=int, default=1) 
    
    parser.add_argument("--forecasting_obs_dim", type=int, default=797) 
    parser.add_argument("--forecasting_act_dim", type=int, default=29) 
    parser.add_argument("--forecasting_nn_frames", type=int, default=10)     
    
    parser.add_argument("--w_forecasting_model", type=str2bool, default=False)
    # use_world_model
    parser.add_argument("--use_world_model", type=str2bool, default=False)
    parser.add_argument("--maxx_inst_nn", type=int, default=1000000) 
    # single_inst_tag
    parser.add_argument("--single_inst_tag", type=str,   default='')
    # open_loop_test
    parser.add_argument("--open_loop_test", type=str2bool, default=False)
    # comput_reward_traj_hand_qpos
    parser.add_argument("--comput_reward_traj_hand_qpos", type=str2bool, default=False)
    parser.add_argument("--train_controller", type=str2bool, default=False)
    # train_forecasting_model
    parser.add_argument("--forecast_obj_pos", type=str2bool, default=False)
    
    parser.add_argument("--train_forecasting_model", type=str2bool, default=False)
    parser.add_argument("--activate_forecaster", type=str2bool, default=True)
    parser.add_argument("--use_future_ref_as_obs_goal", type=str2bool, default=False)
    # parser.add_argument("--forecast_obj_pos", type=str2bool, default=False)
    parser.add_argument("--gpu_offset_idx", type=int, default=0)    
    # multiple_kine_source_trajs_fn
    parser.add_argument("--multiple_kine_source_trajs_fn", type=str,   default='')
    # use_multiple_kine_source_trajs
    parser.add_argument("--use_multiple_kine_source_trajs", type=str2bool, default=False)
    parser.add_argument("--include_obj_rot_in_obs", type=str2bool, default=False)
    # parser.add_argument("--w_franka", type=str2bool,   default=False)
    # # 
    # parser.add_argument("--add_table",  type=str2bool,   default=False) 
    # parser.add_argument("--table_z_dim", type=float, default=0.5)
    parser.add_argument("--load_kine_info_retar_with_arm", type=str2bool, default=False) 
    parser.add_argument("--kine_info_with_arm_sv_root", type=str, default='') 
    
    parser.add_argument("--w_finger_pos_rew", type=str2bool, default=False) 
    # franka_delta_delta_mult_coef
    parser.add_argument("--franka_delta_delta_mult_coef", type=float,   default=1.0)
    # control_arm_via_ik
    parser.add_argument("--control_arm_via_ik", type=str2bool, default=False) 
    # warm_trans_actions_mult_coef, warm_rot_actions_mult_coef
    parser.add_argument("--warm_trans_actions_mult_coef", type=float,   default=0.04)
    parser.add_argument("--warm_rot_actions_mult_coef", type=float,   default=0.04)
    parser.add_argument("--wo_vel_obs", type=str2bool,   default=False)
    parser.add_argument("--compute_hand_rew_buf_threshold", type=int, default=500) 
    parser.add_argument("--not_use_kine_bias", type=str2bool, default=False)
    parser.add_argument("--disable_hand_obj_contact", type=str2bool, default=False)
    # closed_loop_to_real
    parser.add_argument("--closed_loop_to_real", type=str2bool, default=False)
    parser.add_argument("--hand_glb_mult_factor_scaling_coef", type=float,   default=1.0)
    parser.add_argument("--hand_glb_mult_scaling_progress_after", type=int, default=900) 
    # hand_qpos_rew_coef
    parser.add_argument("--wo_fingertip_rot_vel", type=str2bool, default=False)
    parser.add_argument("--wo_fingertip_vel", type=str2bool, default=False)
    # arm_stiffness
    parser.add_argument("--arm_stiffness", type=float,   default=400.0)
    parser.add_argument("--arm_effort", type=float,   default=200.0)
    parser.add_argument("--arm_damping", type=float,   default=80.0)
    # estimate_vels
    parser.add_argument("--estimate_vels", type=str2bool, default=False)
    parser.add_argument("--use_v2_leap_warm_urdf", type=str2bool, default=False)
    # wo_fingertip_pos
    parser.add_argument("--wo_fingertip_pos", type=str2bool, default=False)
    # reset_obj_mass: False
    # obj_mass_reset: 0.27
    # recompute_inertia: False # recompute # previledged info #
    parser.add_argument("--reset_obj_mass", type=str2bool, default=False)
    parser.add_argument("--obj_mass_reset", type=float,   default=0.27)
    parser.add_argument("--recompute_inertia", type=str2bool, default=False)
    parser.add_argument("--add_physical_params_in_obs", type=str2bool, default=False)
    # record_experiences #
    parser.add_argument("--record_experiences", type=str2bool, default=False)
    # action_chunking; 
    parser.add_argument("--action_chunking", type=str2bool, default=False)
    # action_chunking_frames
    parser.add_argument("--action_chunking_frames", type=int, default=1) 
    parser.add_argument("--bc_style_training", type=str2bool, default=False)
    # bc_relative_targets
    parser.add_argument("--bc_relative_targets", type=str2bool, default=False)
    parser.add_argument("--distill_full_to_partial", type=str2bool, default=False)
    parser.add_argument("--train_free_hand", type=str2bool, default=False)
    # preload_action_targets_fn
    parser.add_argument("--preload_action_targets_fn", type=str,   default='')
    # tune_hand_pd: False
    parser.add_argument("--tune_hand_pd", type=str2bool, default=False)
    # record_for_distill_to_ctlv2
    parser.add_argument("--record_for_distill_to_ctlv2", type=str2bool, default=False)
    parser.add_argument("--more_allegro_stiffness", type=str2bool, default=False)
    parser.add_argument("--preload_action_target_env_idx", type=int, default=0) 
    parser.add_argument("--preload_action_start_frame", type=int, default=190) 
    parser.add_argument("--use_no_obj_pose", type=str2bool, default=False)
    # preset_multi_traj_index
    parser.add_argument("--preset_multi_traj_index", type=int, default=-1) 
    parser.add_argument("--multi_traj_use_joint_order_in_sim", type=str2bool, default=False)
    parser.add_argument("--use_actual_prev_targets_in_obs", type=str2bool, default=False)
    parser.add_argument("--action_chunking_skip_frames", type=int, default=1)
    parser.add_argument("--add_obj_features", type=str2bool, default=False)
    parser.add_argument("--distill_via_bc", type=str2bool, default=False)
    parser.add_argument("--kine_ed_tag", type=str,   default='.npy')
    parser.add_argument("--use_actual_traj_length", type=str2bool, default=False)
    parser.add_argument("--add_global_movements", type=str2bool, default=False)
    parser.add_argument("--add_global_movements_af_step", type=int, default=369) 
    
    args = parser.parse_args()
    
    if len(args.subj_nm) > 0:
        args.subj_idx = int(args.subj_nm[1:])
    else:
        args.subj_idx = ''
    
    # isnt tag #
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
        
    if len(args.obj_type_to_base_trajs) > 0 and os.path.exists(args.obj_type_to_base_trajs):
        obj_type_to_base_trajs = np.load(args.obj_type_to_base_trajs, allow_pickle=True).item()
    else:
        obj_type_to_base_trajs = None
    
    
    def launch_one_process(cur_grab_data_tag, traj_grab_data_tag, cuda_idx, pre_optimized_traj=None, traj_checkpoint=None, cur_base_traj=None):
        print(f"pre_optimized_traj: {pre_optimized_traj}")
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
                
                ####### Mocap sv info fn v1 #######
                # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
                # if args.use_interpolated_data:
                #     # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_modifed_interped_transformed.npy"
                #     # passive_active_info_taco_20231104_203_zrot_3.141592653589793_modifed_interped # 
                #     mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_zrot_3.141592653589793_modifed_interped.npy"
                #     if not os.path.exists(mocap_sv_info_fn):
                #         mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_modifed_interped.npy"
                ####### Mocap sv info fn v1 #######
                
                #### by default we use interpolated data ####
                if len(args.taco_interped_data_sv_additional_tag) > 0:
                    mocap_sv_info_fn = f"passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2_{args.taco_interped_data_sv_additional_tag}.npy"
                else:
                    mocap_sv_info_fn = f"passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2.npy"
                mocap_sv_info_fn = os.path.join(args.tracking_data_sv_root, mocap_sv_info_fn)
            
            else:
                raise ValueError
        
        elif args.hand_type == 'leap':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB_Tracking/data/leap_passive_active_info_{traj_grab_data_tag}.npy"
            # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/leap_passive_active_info_{traj_grab_data_tag}.npy"
            mocap_sv_info_fn = f"/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{traj_grab_data_tag}_v2_interpfr_60_interpfr2_60_nntrans_40.npy"
        else:
            raise ValueError
        
        # traj is not None #
        if traj_checkpoint is not None and len(traj_checkpoint) > 0 and os.path.exists(traj_checkpoint):
            cur_traj_checkpoint = traj_checkpoint
        else:
            cur_traj_checkpoint = args.checkpoint
        
        print(f"mocap_sv_info_fn: {mocap_sv_info_fn}")
        
        checkpoint = ''
        tag = f"tracking_{object_name}"
        
        if args.launch_type == 'trajectory':
            if args.hand_type == 'allegro':
                train_name = f"tracking_{object_name}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
            elif args.hand_type == 'leap':
                train_name = f"tracking_{object_name}_{args.hand_type}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
            else:
                raise ValueError
        elif args.launch_type == 'object_type':
            if args.hand_type == 'allegro':
                train_name = f"tracking_{object_name}_traj_{traj_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
            elif args.hand_type == 'leap':
                train_name = f"tracking_{object_name}_traj_{traj_grab_data_tag}_{args.hand_type}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
            else:
                raise ValueError
        else: 
            raise ValueError
        
        # model and the performance of the model #
        
        full_experiment_name = train_name
        
        if args.headless:
            capture_video = False
            force_render = False
        else:
            # capture_video = True
            capture_video = False
            force_render = True
        
        # # # # #
        # use_small_sigmas = "--use_small_sigmas" if args.use_small_sigmas else "" #
        # use_relaxed_model = "--use_relaxed_model" if args.use_relaxed_model else "" #
        # w_hand_table_collision = "--w_hand_table_collision" if args.w_hand_table_collision else "" #
        
        
        print(f"test: {args.test}")
        
        if args.headless:
            cuda_visible_text = f"CUDA_VISIBLE_DEVICES={cuda_idx} "
        else:
            # cuda_visible_text = ''
            cuda_visible_text = f"CUDA_VISIBLE_DEVICES={cuda_idx} "
        
        
        exp_dir = '/cephfs/xueyi/exp/IsaacGymEnvs/isaacgymenvs'
        if not os.path.exists(exp_dir):
            exp_dir = '.'
        
        if len(args.target_object_name) > 0:
            object_name=args.target_object_name
        
        if len(args.target_mocap_sv_fn) > 0:
            mocap_sv_info_fn = args.target_mocap_sv_fn
            
        if pre_optimized_traj is not None and len(pre_optimized_traj) > 0:
            cur_pre_optimized_traj = pre_optimized_traj
        else:
            cur_pre_optimized_traj = args.pre_optimized_traj
            
        if cur_base_traj is None:
            cur_base_traj = ''
            
        # argessue the visio n#
        if args.use_vision:
            task_type = "AllegroHandTrackingVision"
            train_type = "HumanoidPPOVision"
            
            enableCameraSensors = True
            if args.use_dagger:
                # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/cfg/train/HumanoidPPOSupervised.yaml
                task_type = "AllegroHandTrackingVision"
                train_type = "HumanoidPPOVisionDAgger"
            print(f"task_type: {task_type}, train_type: {train_type}")
        else:
            
            if args.use_generalist_policy:
                task_type = "AllegroHandTrackingGeneralist"
                train_type = "HumanoidPPO" 
                # test_inst_tag, test_optimized_res # optimized res # # optimized res # # optimized res # optimized res ##
                if args.supervised_training:
                    task_type = "AllegroHandTrackingGeneralist"
                    train_type = "HumanoidPPOSupervised"
                    
                    # training mode config # # #
                    # training mode config # # #
                    if args.single_instance_state_based_train:
                        train_type = "HumanoidPPOSupervisedSN"
                        print(f"using SN")
                    if args.w_forecasting_model:
                        # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/cfg/train/HumanoidPPOSupervisedWForecasting.yaml
                        train_type = "HumanoidPPOSupervisedWForecasting"
                        task_type = "AllegroHandTrackingGeneralistWForecasting"
                    
                    if args.action_chunking:
                        task_type = "AllegroHandTrackingGeneralistChunking"
                        # train_type = "HumanoidPPOSupervised"
                        train_type = "HumanoidPPOSupervisedChunking"
                    
                    # training mode config #
                    training_mode_config = f"train.params.config.training_mode={args.training_mode}"
                    test_inst_config = f"task.env.test_inst_tag={args.test_inst_tag} task.env.test_optimized_res={args.test_optimized_res}"
                    # # preload_experiences_tf, preload_experiences_path
                    preload_experience_config = f"train.params.config.preload_experiences_tf={args.preload_experiences_tf} train.params.config.preload_experiences_path={args.preload_experiences_path}"
                    single_instance_training_config = f"train.params.config.single_instance_training={args.single_instance_training}"
                    
                    # ### for the test setting -- ###
                    sampleds_with_object_code_fn_config = f"task.env.sampleds_with_object_code_fn={args.sampleds_with_object_code_fn}" # no base trajs are usd during the traiing #
                    
                    # # ori grab s2 apple lift #
                    if args.generalist_tune_all_instnaces: # preload ## preload #
                        test_inst_config = f"task.env.test_inst_tag={cur_grab_data_tag} task.env.test_optimized_res={pre_optimized_traj}"
                        single_instance_training_config = f"train.params.config.single_instance_training={False}"
                        preload_experience_config = f"train.params.config.preload_experiences_tf={False} train.params.config.preload_experiences_path={''}"

                    # samples with code fn # # samples with code fn # #
                    log_path_config = f"train.params.config.log_path={args.log_path}"
                    train_dir_config = f"train.params.config.train_dir={args.log_path}"
                    single_instance_tag_config = f"train.params.config.single_instance_tag={args.single_instance_tag}"
                    obj_type_to_optimized_res_fn_config = f"train.params.config.obj_type_to_optimized_res_fn={args.obj_type_to_optimized_res_fn}"
                    supervised_loss_coef_config = f"train.params.config.supervised_loss_coef={args.supervised_loss_coef}"
                    pure_supervised_training_config = f"train.params.config.pure_supervised_training={args.pure_supervised_training}"
                    # inst_tag_to_latent_feature_fn_config = f"task.env.inst_tag_to_latent_feature_fn={args.inst_tag_to_latent_feature_fn}"
                    # object_type_to_latent_feature_fn_config = f"task.env.object_type_to_latent_feature_fn={args.object_type_to_latent_feature_fn}" # force # static scene #
                    downsample_config = f"task.env.downsample={args.downsample}"
                    w_franka_config = f"task.env.w_franka={args.w_franka}"
                    randomize_config = f"task.task.randomize={args.randomize}"
                    forcasting_model_config = f"task.env.use_forcasting_model={args.use_forcasting_model} task.env.forcasting_model_weights={args.forcasting_model_weights} task.env.forcasting_model_n_layers={args.forcasting_model_n_layers} task.env.w_glb_traj_feat_cond={args.w_glb_traj_feat_cond}"
                    use_history_obs_config= f"task.env.use_history_obs={args.use_history_obs}"
                    substeps_config = f"task.sim.substeps={args.substeps}" # 
                    # use_forcasting_model, forcasting_model_weights, forcasting_model_n_layers, w_glb_traj_feat_cond #
                    # forcasting_model_config = f"task.env.use_forcasting_model={args.use_forcasting_model} task.env.forcasting_model_weights={args.forcasting_model_weights} task.env.forcasting_model_n_layers={args.forcasting_model_n_layers} task.env.w_glb_traj_feat_cond={args.w_glb_traj_feat_cond}" # # forcasting model # #
                    use_window_future_selection_config = f"task.env.use_window_future_selection={args.use_window_future_selection}"
                    # window 
                    # forcasting_inv_freq_config = f"task.env.forcasting_inv_freq={args.forcasting_inv_freq}"
                    # controller_setting_config = f"task.env.stiffness_coef={args.stiffness_coef} task.env.damping_coef={args.damping_coef} task.env.effort_coef={args.effort_coef}"
                    # forcasting_history_ws_config = f"task.env.forcasting_history_ws={args.forcasting_history_ws}"
                    # train_controller_config = "task"
                    
                else:
                    training_mode_config  = ""
                    test_inst_config = ""
                    single_instance_training_config = ""
                    sampleds_with_object_code_fn_config = "" # 
                    log_path_config = "" # 
                    single_instance_tag_config = ""
                    obj_type_to_optimized_res_fn_config = ""
                    supervised_loss_coef_config = ""
                    pure_supervised_training_config = ""
                    inst_tag_to_latent_feature_fn_config = ""
                    object_type_to_latent_feature_fn_config = ""
                    downsample_config = ""
                    w_franka_config = ""
                    randomize_config = ""
                    forcasting_model_config = ""
                    use_history_obs_config = ""
                    substeps_config = ""
                    use_window_future_selection_config = ""
                    train_controller_config = ""
                    train_forecasting_model_config = ""
                
                maxx_inst_nn_config= f"task.env.maxx_inst_nn={args.maxx_inst_nn} train.params.config.maxx_inst_nn={args.maxx_inst_nn}"
                # grab_opt_res_config  taco_opt_res_config # # taco opt res config #
                grab_opt_res_config = f"task.env.grab_inst_tag_to_optimized_res_fn={args.grab_inst_tag_to_optimized_res_fn}"
                taco_opt_res_config = f"task.env.taco_inst_tag_to_optimized_res_fn={args.taco_inst_tag_to_optimized_res_fn}"
                object_type_to_latent_feature_fn_config = f"task.env.object_type_to_latent_feature_fn={args.object_type_to_latent_feature_fn}"
                inst_tag_to_latent_feature_fn_config = f"task.env.inst_tag_to_latent_feature_fn={args.inst_tag_to_latent_feature_fn}"
                # task.env.taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn}
                tracking_save_info_fn_config = f"task.env.tracking_save_info_fn={args.tracking_save_info_fn}"
                taco_interped_data_sv_additional_tag_config = f"task.env.taco_interped_data_sv_additional_tag={args.taco_interped_data_sv_additional_tag}"
                tracking_info_st_tag_config = f"task.env.tracking_info_st_tag={args.tracking_info_st_tag}"
                
                use_strict_maxx_nn_ts_config = f"task.env.use_strict_maxx_nn_ts={args.use_strict_maxx_nn_ts}"
                strict_maxx_nn_ts_config=  f"task.env.strict_maxx_nn_ts={args.strict_maxx_nn_ts}"
                use_local_canonical_state_config = f"task.env.use_local_canonical_state={args.use_local_canonical_state}"
                use_base_traj_config = f"task.env.use_base_traj={args.use_base_traj}"
                base_traj_config = f"task.env.base_traj={cur_base_traj}"
                customize_damping_config = f"task.env.customize_damping={args.customize_damping}"
                single_instance_state_based_train_config = f"task.env.single_instance_state_based_train={args.single_instance_state_based_train}"
                history_setting_config = f"task.env.use_history_obs={args.use_history_obs} task.env.history_length={args.history_length}"
                w_history_window_index_config = f"task.env.w_history_window_index={args.w_history_window_index}"
                w_obj_latent_features_config = f"task.env.w_obj_latent_features={args.w_obj_latent_features}"
                
                forcasting_inv_freq_config = f"task.env.forcasting_inv_freq={args.forcasting_inv_freq}"
                controller_setting_config = f"task.env.stiffness_coef={args.stiffness_coef} task.env.damping_coef={args.damping_coef} task.env.effort_coef={args.effort_coef}"
                forcasting_history_ws_config = f"task.env.forcasting_history_ws={args.forcasting_history_ws}"
                use_future_obs_config = f"task.env.use_future_obs={args.use_future_obs}"
                randomize_conditions_config = f"task.env.randomize_conditions={args.randomize_conditions}"
                history_freq_config = f"task.env.history_freq={args.history_freq}"
                randomize_condition_type_config = f"task.env.randomize_condition_type={args.randomize_condition_type}"
                forcasting_diffusion_model_config = f"task.env.forcasting_diffusion_model={args.forcasting_diffusion_model}"
                partial_info_config = f"task.env.partial_obj_info={args.partial_obj_info} task.env.partial_hand_info={args.partial_hand_info}"
                # use_partial_to_complete_model, partial_to_complete_model_weights
                partial_to_complete_config = f"task.env.use_partial_to_complete_model={args.use_partial_to_complete_model} task.env.partial_to_complete_model_weights={args.partial_to_complete_model_weights}"
                contact_condition_setting = f"task.env.add_contact_conditions={args.add_contact_conditions} task.env.contact_info_sv_root={args.contact_info_sv_root}"
                st_ed_state_cond_setting = f"task.env.st_ed_state_cond={args.st_ed_state_cond}"
                partial_obj_pos_info_setting = f"task.env.partial_obj_pos_info={args.partial_obj_pos_info}"
                
                # # --history_window_size=${history_window_size} --glb_feat_per_skip=${glb_feat_per_skip} --centralize_info=${centralize_info}
                history_glbfeat_setting = f"task.env.history_window_size={args.history_window_size} task.env.glb_feat_per_skip={args.glb_feat_per_skip} task.env.centralize_info={args.centralize_info}"
                hist_cond_partial_info_setting = f"task.env.hist_cond_partial_hand_info={args.hist_cond_partial_hand_info} task.env.hist_cond_partial_obj_info={args.hist_cond_partial_obj_info} task.env.hist_cond_partial_obj_pos_info={args.hist_cond_partial_obj_pos_info}" 
                preset_cond_type_config = f"task.env.preset_cond_type={args.preset_cond_type}"
                preset_inv_cond_freq_config = f"task.env.preset_inv_cond_freq={args.preset_inv_cond_freq}"
                # random_shift_cond, random_shift_cond_freq, maxx_inv_cond_freq
                random_shift_conditions_setting = f"task.env.random_shift_cond={args.random_shift_cond} task.env.random_shift_cond_freq={args.random_shift_cond_freq} task.env.maxx_inv_cond_freq={args.maxx_inv_cond_freq}"
                use_clip_glb_features_config = f"task.env.use_clip_glb_features={args.use_clip_glb_features}"
                forecasting_model_inv_freq_config = f"task.env.forecasting_model_inv_freq={args.forecasting_model_inv_freq}"
                forecasting_agent_training_config = f"train.params.config.forecasting_obs_dim={args.forecasting_obs_dim} train.params.config.forecasting_act_dim={args.forecasting_act_dim} train.params.config.forecasting_nn_frames={args.forecasting_nn_frames}"
                use_world_model_config = f"train.params.config.use_world_model={args.use_world_model}"
                open_loop_test_config = f"task.env.open_loop_test={args.open_loop_test}"
                comput_reward_traj_hand_qpos_config = f"task.env.comput_reward_traj_hand_qpos={args.comput_reward_traj_hand_qpos}"
                train_controller_config = f"train.params.config.train_controller={args.train_controller}"
                train_forecasting_model_config = f"train.params.config.train_forecasting_model={args.train_forecasting_model}"
                activate_forecaster_config = f"task.env.activate_forecaster={args.activate_forecaster}"
                use_future_ref_as_obs_goal_config = f"task.env.use_future_ref_as_obs_goal={args.use_future_ref_as_obs_goal}"
                forecast_obj_pos_config = f"task.env.forecast_obj_pos={args.forecast_obj_pos}"
                multiple_kine_source_trajs_fn_config = f"task.env.multiple_kine_source_trajs_fn={args.multiple_kine_source_trajs_fn}"
                use_multiple_kine_source_trajs_config = f"task.env.use_multiple_kine_source_trajs={args.use_multiple_kine_source_trajs}"
                include_obj_rot_in_obs_config = f"task.env.include_obj_rot_in_obs={args.include_obj_rot_in_obs}"
                w_kine_retar_with_arm_config = f"task.env.load_kine_info_retar_with_arm={args.load_kine_info_retar_with_arm} task.env.kine_info_with_arm_sv_root={args.kine_info_with_arm_sv_root}"
                w_finger_pos_rew_config  = f"task.env.w_finger_pos_rew={args.w_finger_pos_rew}"
                franka_delta_delta_mult_coef_config = f"task.env.franka_delta_delta_mult_coef={args.franka_delta_delta_mult_coef}"
                control_arm_via_ik_config = f"task.env.control_arm_via_ik={args.control_arm_via_ik}"
                warm_actions_mult_coef_config = f"task.env.warm_trans_actions_mult_coef={args.warm_trans_actions_mult_coef} task.env.warm_rot_actions_mult_coef={args.warm_rot_actions_mult_coef}"
                wo_vel_obs_config  = f"task.env.wo_vel_obs={args.wo_vel_obs}"
                compute_hand_rew_buf_threshold_config = f"task.env.compute_hand_rew_buf_threshold={args.compute_hand_rew_buf_threshold}"
                not_use_kine_bias_config = f"task.env.not_use_kine_bias={args.not_use_kine_bias}"
                disable_hand_obj_contact_config = f"task.env.disable_hand_obj_contact={args.disable_hand_obj_contact}"
                closed_loop_to_real_config = f"task.env.closed_loop_to_real={args.closed_loop_to_real}"
                glb_mult_factor_scaling_config = f"task.env.hand_glb_mult_factor_scaling_coef={args.hand_glb_mult_factor_scaling_coef} task.env.hand_glb_mult_scaling_progress_after={args.hand_glb_mult_scaling_progress_after}"
                wo_fingertip_rot_vel_config = f"task.env.wo_fingertip_pos={args.wo_fingertip_pos} task.env.wo_fingertip_rot_vel={args.wo_fingertip_rot_vel} task.env.wo_fingertip_vel={args.wo_fingertip_vel}"
                arm_ctl_params_config = f"task.env.arm_stiffness={args.arm_stiffness} task.env.arm_effort={args.arm_effort} task.env.arm_damping={args.arm_damping}"
                estimate_vels_config = f"task.env.estimate_vels={args.estimate_vels}"
                use_v2_leap_warm_urdf_config = f"task.env.use_v2_leap_warm_urdf={args.use_v2_leap_warm_urdf}"
                action_chunking_config = f"task.env.action_chunking={args.action_chunking} task.env.action_chunking_frames={args.action_chunking_frames} train.params.config.action_chunking={args.action_chunking} train.params.config.action_chunking_frames={args.action_chunking_frames}"
                # reset_obj_mass: False
                # obj_mass_reset: 0.27
                # recompute_inertia: False
                reset_obj_mass_config = f"task.env.reset_obj_mass={args.reset_obj_mass} task.env.obj_mass_reset={args.obj_mass_reset} task.env.recompute_inertia={args.recompute_inertia}"
                add_physical_params_in_obs_config = f"task.env.add_physical_params_in_obs={args.add_physical_params_in_obs}"
                record_experiences_config = f"train.params.config.record_experiences={args.record_experiences}"
                bc_style_training_config = f"train.params.config.bc_style_training={args.bc_style_training} task.env.bc_style_training={args.bc_style_training}"
                bc_relative_targets_config = f"train.params.config.bc_relative_targets={args.bc_relative_targets} task.env.bc_relative_targets={args.bc_relative_targets}"
                distill_full_to_partial_config = f"task.env.distill_full_to_partial={args.distill_full_to_partial}"
                train_free_hand_config = f"task.env.train_free_hand={args.train_free_hand}"
                preload_action_targets_fn_config = f"task.env.preload_action_targets_fn={args.preload_action_targets_fn} task.env.preload_action_target_env_idx={args.preload_action_target_env_idx} task.env.preload_action_start_frame={args.preload_action_start_frame}"
                tune_hand_pd_config = f"task.env.tune_hand_pd={args.tune_hand_pd}"
                record_for_distill_to_ctlv2_config = f"train.params.config.record_for_distill_to_ctlv2={args.record_for_distill_to_ctlv2}"
                more_allegro_stiffness_config = f"task.env.more_allegro_stiffness={args.more_allegro_stiffness}"
                use_no_obj_pose_config = f"train.params.config.use_no_obj_pose={args.use_no_obj_pose} task.env.use_no_obj_pose={args.use_no_obj_pose}"
                preset_multi_traj_index_config = f"task.env.preset_multi_traj_index={args.preset_multi_traj_index}"
                multi_traj_use_joint_order_in_sim_config = f"task.env.multi_traj_use_joint_order_in_sim={args.multi_traj_use_joint_order_in_sim}"
                use_actual_prev_targets_in_obs_config = f"task.env.use_actual_prev_targets_in_obs={args.use_actual_prev_targets_in_obs}"
                action_chunking_skip_frames_config = f"task.env.action_chunking_skip_frames={args.action_chunking_skip_frames}"
                add_obj_features_config = f"task.env.add_obj_features={args.add_obj_features}"
                distill_via_bc_config = f"task.env.distill_via_bc={args.distill_via_bc} train.params.config.distill_via_bc={args.distill_via_bc}"
                kine_ed_tag_config = f"task.env.kine_ed_tag={args.kine_ed_tag}"
                use_actual_traj_length_config = f"task.env.use_actual_traj_length={args.use_actual_traj_length}"
                add_global_movements_config=  f"task.env.add_global_movements={args.add_global_movements} task.env.add_global_movements_af_step={args.add_global_movements_af_step}"
            else:
                
                task_type = "AllegroHandTracking" 
                train_type = "HumanoidPPO"
                training_mode_config = ""
                test_inst_config = ""
                
                maxx_inst_nn_config = ""

                single_instance_training_config = ""
                sampleds_with_object_code_fn_config = ""
                log_path_config = ""
                
                grab_opt_res_config = f""
                taco_opt_res_config = f""
                
                single_instance_tag_config = ""
                obj_type_to_optimized_res_fn_config = "" #
                
                supervised_loss_coef_config = ""
                pure_supervised_training_config = ""

                inst_tag_to_latent_feature_fn_config = ""
                object_type_to_latent_feature_fn_config = ""
                tracking_save_info_fn_config = ""
                taco_interped_data_sv_additional_tag_config = ""
                
                use_strict_maxx_nn_ts_config = ""
                strict_maxx_nn_ts_config = ""
                use_local_canonical_state_config = ""
                
                use_base_traj_config = ""
                base_traj_config = ""
                customize_damping_config = ""
                tracking_info_st_tag_config = ""
                single_instance_state_based_train_config = ""
                history_setting_config = ""
                use_future_obs_config = ""
                randomize_conditions_config = ""
                history_freq_config= ""
                randomize_condition_type_config = ""
                forcasting_diffusion_model_config = ""
                partial_info_config = ""
                partial_to_complete_config = ""
                contact_condition_setting = ""
                st_ed_state_cond_setting = ""
                partial_obj_pos_info_setting = ""
                history_glbfeat_setting = ""
                hist_cond_partial_info_setting = ""
                preset_cond_type_config = ""
                preset_inv_cond_freq_config = ""
                random_shift_conditions_setting = ""
                use_clip_glb_features_config = ""
                forecasting_model_inv_freq_config=  ""
                open_loop_test_config = ""
                comput_reward_traj_hand_qpos_config = ""
                train_controller_config = ""
                train_forecasting_model_config = ""
                activate_forecaster_config = ""
                use_future_ref_as_obs_goal_config = ""
                forecast_obj_pos_config=""
                multiple_kine_source_trajs_fn_config = ""
                use_multiple_kine_source_trajs_config = ""
                include_obj_rot_in_obs_config = ""
                w_kine_retar_with_arm_config = ""
                w_finger_pos_rew_config = ""
                franka_delta_delta_mult_coef_config = ""
                control_arm_via_ik_config = ""
                warm_actions_mult_coef_config = ""
                wo_vel_obs_config = ""
                compute_hand_rew_buf_threshold_config = ""
                not_use_kine_bias_config = ""
                disable_hand_obj_contact_config = ""
                closed_loop_to_real_config = ""
                glb_mult_factor_scaling_config = ""
                wo_fingertip_rot_vel_config = ""
                arm_ctl_params_config = ""
                estimate_vels_config = ""
                use_v2_leap_warm_urdf_config = ""
                record_experiences_config = ""
                use_actual_prev_targets_in_obs_config = ""
            enableCameraSensors = False
        # if args.use_generalist_policy: #### taco ####
        # grab_inst_tag_to_optimized_res_fn, taco_inst_tag_to_optimized_res_fn
        
        
        if args.use_vision:
            cuda_visible_text = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  " 
            cmd = f"{cuda_visible_text} python train.py task={task_type} train={train_type} sim_device='cuda:{cuda_idx}' rl_device='cuda:{cuda_idx}'  capture_video={capture_video} force_render={force_render} headless={args.headless}   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={args.test} task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={cur_traj_checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.controlFrequencyInv={args.controlFrequencyInv} task.env.episodeLength={args.episodeLength} task.env.start_frame={args.start_frame} task.env.rew_obj_pose_coef={args.rew_obj_pose_coef} task.env.goal_dist_thres={args.goal_dist_thres} task.env.lifting_separate_stages={args.lifting_separate_stages} task.env.strict_lifting_separate_stages={args.strict_lifting_separate_stages} task.env.table_z_dim={args.table_z_dim} task.env.add_table={args.add_table} exp_dir={exp_dir} task.env.use_taco_obj_traj={args.use_taco_obj_traj} task.env.pre_optimized_traj={ cur_pre_optimized_traj } task.env.hand_type={ args.hand_type } enableCameraSensors={enableCameraSensors} graphics_device_id={cuda_idx} {downsample_config} {randomize_config} {forcasting_model_config} {use_history_obs_config} {substeps_config} {history_setting_config} {use_window_future_selection_config}"
        else:
            cmd = f"{cuda_visible_text} python train.py task={task_type} train={train_type} sim_device='cuda:0' rl_device='cuda:0'  capture_video={capture_video} force_render={force_render} headless={args.headless}   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={ True } task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={cur_traj_checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.controlFrequencyInv={args.controlFrequencyInv} task.env.episodeLength={args.episodeLength} task.env.start_frame={args.start_frame} task.env.rew_obj_pose_coef={args.rew_obj_pose_coef} task.env.goal_dist_thres={args.goal_dist_thres} task.env.lifting_separate_stages={args.lifting_separate_stages} task.env.strict_lifting_separate_stages={args.strict_lifting_separate_stages} task.env.table_z_dim={args.table_z_dim} task.env.add_table={args.add_table} exp_dir={exp_dir} task.env.use_taco_obj_traj={args.use_taco_obj_traj} task.env.pre_optimized_traj={ cur_pre_optimized_traj } task.env.hand_type={ args.hand_type } enableCameraSensors={enableCameraSensors} graphics_device_id={cuda_idx} task.env.use_hand_actions_rew={args.use_hand_actions_rew} task.env.supervised_training={args.supervised_training} {training_mode_config} {test_inst_config} {preload_experience_config} {single_instance_training_config} {sampleds_with_object_code_fn_config} {log_path_config} {train_dir_config} {grab_opt_res_config}  {taco_opt_res_config} {single_instance_tag_config} {obj_type_to_optimized_res_fn_config} {supervised_loss_coef_config} {pure_supervised_training_config} {inst_tag_to_latent_feature_fn_config} {object_type_to_latent_feature_fn_config} {tracking_save_info_fn_config} {taco_interped_data_sv_additional_tag_config} {use_strict_maxx_nn_ts_config} {strict_maxx_nn_ts_config} {use_local_canonical_state_config} {use_base_traj_config} {base_traj_config} {customize_damping_config} {tracking_info_st_tag_config} {single_instance_state_based_train_config} {downsample_config} {w_franka_config} {randomize_config} {forcasting_model_config} {use_history_obs_config} {substeps_config} {history_setting_config} {use_window_future_selection_config} {w_history_window_index_config} {w_obj_latent_features_config} {forcasting_inv_freq_config} {forcasting_history_ws_config} {use_future_obs_config} {randomize_conditions_config} {history_freq_config} {randomize_condition_type_config} {forcasting_diffusion_model_config} {controller_setting_config} {partial_info_config} {partial_to_complete_config} {contact_condition_setting} {st_ed_state_cond_setting} {partial_obj_pos_info_setting} {history_glbfeat_setting} {hist_cond_partial_info_setting} {preset_cond_type_config} {preset_inv_cond_freq_config} {random_shift_conditions_setting} {use_clip_glb_features_config} {forecasting_model_inv_freq_config} {forecasting_agent_training_config} {use_world_model_config } {maxx_inst_nn_config} {open_loop_test_config} {comput_reward_traj_hand_qpos_config} {train_controller_config} {train_forecasting_model_config} {activate_forecaster_config} {use_future_ref_as_obs_goal_config} {forecast_obj_pos_config} {multiple_kine_source_trajs_fn_config} {use_multiple_kine_source_trajs_config} {include_obj_rot_in_obs_config} {w_kine_retar_with_arm_config} {w_finger_pos_rew_config} {franka_delta_delta_mult_coef_config} {control_arm_via_ik_config} {warm_actions_mult_coef_config} {wo_vel_obs_config} {compute_hand_rew_buf_threshold_config} {not_use_kine_bias_config} {disable_hand_obj_contact_config} {closed_loop_to_real_config} {glb_mult_factor_scaling_config} {wo_fingertip_rot_vel_config} {arm_ctl_params_config} {estimate_vels_config} {use_v2_leap_warm_urdf_config} {reset_obj_mass_config} {add_physical_params_in_obs_config} {record_experiences_config} {action_chunking_config} {bc_style_training_config} {bc_relative_targets_config} {distill_full_to_partial_config} {train_free_hand_config} {preload_action_targets_fn_config} {tune_hand_pd_config} {record_for_distill_to_ctlv2_config} {more_allegro_stiffness_config} {use_no_obj_pose_config} {preset_multi_traj_index_config} {multi_traj_use_joint_order_in_sim_config} {use_actual_prev_targets_in_obs_config} {action_chunking_skip_frames_config} {add_obj_features_config} {distill_via_bc_config} {kine_ed_tag_config} {use_actual_traj_length_config} {add_global_movements_config}"   
            
            
        print(cmd)
        os.system(cmd)
    
    
    # base_dir = '/cephfs/yilaa/uni_manip/tds_rl_exp'
    tracking_data_sv_root = args.tracking_data_sv_root
    
    obj_type_to_ckpt_fn = args.obj_type_to_ckpt_fn
    if len(obj_type_to_ckpt_fn) > 0 and os.path.exists(obj_type_to_ckpt_fn):
        obj_type_to_ckpt = np.load(obj_type_to_ckpt_fn, allow_pickle=True).item()
    else:
        obj_type_to_ckpt = None
    
    
    if args.launch_type != 'trajectory':
        # grab_data_nm_idx_dict #
        # grab_tracking_data_root = args.tracking_data_sv_root
        if args.subj_idx == 2 or args.subj_idx < 1:
            data_nm_idx_dict_sv_fn = "grab_data_nm_idx_dict.npy"
        else:
            data_nm_idx_dict_sv_fn = f"grab_data_nm_idx_dict_s{args.subj_idx}.npy"
        data_nm_idx_dict_sv_fn = os.path.join(tracking_data_sv_root, data_nm_idx_dict_sv_fn)
        data_nm_idx_dict = np.load(data_nm_idx_dict_sv_fn, allow_pickle=True).item()
        data_nm_to_idx = data_nm_idx_dict['data_nm_to_idx'] # idx to data nm # 
        idx_to_data_nm = data_nm_idx_dict['idx_to_data_nm'] # data nm to idx # 
        
    
    def find_similar_objs(obj_index):
        grab_cross_obj_diff_arr_fn = "../assets/grab_cross_obj_verts_diff.npy"
        grab_cross_obj_diff_arr = np.load(grab_cross_obj_diff_arr_fn)
        cur_obj_diff_arr = grab_cross_obj_diff_arr[obj_index]
        cur_obj_sorted_nei_idxes = np.argsort(cur_obj_diff_arr, axis=0)
        cur_obj_sorted_nei_idxes = cur_obj_sorted_nei_idxes[1: 1 + 10]
        cur_obj_sorted_nei_idxes = cur_obj_sorted_nei_idxes.tolist()
        return cur_obj_sorted_nei_idxes

    

    if args.dataset_type == 'grab':
        # passive_active_info_ori_grab_s2_pyramidlarge_lift.npy
        # starting_str = "passive_active_info_ori_grab_"
        # passive_active_info_tag = "passive_active_info_"
        
        # passive_active_info_ori_grab_s2_pyramidlarge_lift.npy
        starting_str = "passive_active_info_ori_grab_"
        passive_active_info_tag = "passive_active_info_"
        
        if args.hand_type == 'leap':
            starting_str = "leap_" + starting_str
            passive_active_info_tag = "leap_" + passive_active_info_tag
        
        tot_tracking_data = os.listdir(tracking_data_sv_root)
        
        # print(f"tot_tracking_data: {tot_tracking_data}")
        if args.num_frames == 150:
            tot_tracking_data = [fn for fn in tot_tracking_data if fn[: len(starting_str)] == starting_str and fn.endswith(".npy") and "_nf_" not in fn]
        else:
            nf_tag = f"_nf_{args.num_frames}"
            tot_tracking_data = [fn for fn in tot_tracking_data if fn[: len(starting_str)] == starting_str and fn.endswith(".npy") and nf_tag in fn]
        
        if len(args.subj_nm) > 0:
            subj_tag = f"_{args.subj_nm}_"
            tot_tracking_data = [fn for fn in tot_tracking_data if subj_tag in fn]
    elif args.dataset_type == 'taco':
        taso_inst_st_flag = 'taco_'
        mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        if not os.path.exists(mesh_sv_root):
            mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        tot_mesh_folders = os.listdir(mesh_sv_root)
        # find meshes directly #
        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        
        # test_on_taco_test_set #
        if args.test_on_taco_test_set:
            test_taco_tag = 'taco_20231024_'
            tot_mesh_folders = [fn for fn in tot_mesh_folders if test_taco_tag in fn]
        
        
        # modified_tag = "_modifed"
        # interped_tag = "_interped"
        # find tracking data
        tot_tracking_data = tot_mesh_folders
        passive_active_info_tag = ''
    
    else:
        raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}")
    
    print(f"tot_tracking_data: {tot_tracking_data}")
    nn_gpus = args.nn_gpus
    
    # gpu_idx_list = [1, 2, 3, 4, 5, 6, 7]
    gpu_idx_list = [0, 1, 2, 3, 4, 5, 6, 7]
    nn_gpus = len(gpu_idx_list)
    
    
    ### and also add the grab instance to the optimized res dict
    # pre_load_trajectories, obj_type_to_pre_optimized_traj
    pre_load_trajectories = args.pre_load_trajectories
    print(f"pre_load_trajectories: {pre_load_trajectories}") # load trajectories #
    if pre_load_trajectories and len(args.obj_type_to_pre_optimized_traj ) > 0 and os.path.exists(args.obj_type_to_pre_optimized_traj ): # load pre optimized trajectories ##
        obj_type_to_pre_optimized_traj = args.obj_type_to_pre_optimized_traj 
        print(f"obj_type_to_pre_optimized_traj: {obj_type_to_pre_optimized_traj}")
        assert len(obj_type_to_pre_optimized_traj) > 0 and os.path.exists(obj_type_to_pre_optimized_traj)
        print(f"obj_type_to_pre_optimized_traj start loading..")
        obj_type_to_pre_optimized_traj = np.load(obj_type_to_pre_optimized_traj, allow_pickle=True).item()
        print(f"Loaded")
        #### obj type to pre optimized traj ####
    else:
        obj_type_to_pre_optimized_traj = None
    
    # print(f"obj_type_to_pre_optimized_traj: {obj_type_to_pre_optimized_traj.keys()}")
    if obj_type_to_ckpt is not None :
        print(f"obj_type_to_ckpt: {obj_type_to_ckpt.keys()}")
        
    # first_ten_traj_tags = [
    #     "ori_grab_s10_alarmclock_pass_1", "ori_grab_s10_stanfordbunny_pass_1", "ori_grab_s2_knife_pass_1", "ori_grab_s4_train_lift", "ori_grab_s4_watch_set_2", "ori_grab_s4_waterbottle_drink_1", "ori_grab_s5_duck_inspect_1", "ori_grab_s6_torusmedium_inspect_1", "ori_grab_s8_binoculars_pass_1", "ori_grab_s8_cylinderlarge_inspect_1"
    # ]
    first_ten_traj_tags = [
        "ori_grab_s10_alarmclock_pass_1", "ori_grab_s2_knife_pass_1", "ori_grab_s4_train_lift", "ori_grab_s4_watch_set_2", "ori_grab_s4_waterbottle_drink_1", "ori_grab_s6_torusmedium_inspect_1", "ori_grab_s8_binoculars_pass_1", "ori_grab_s8_cylinderlarge_inspect_1"
    ]
    first_ten_traj_tags = [
        fn + "_nf_300" for fn in first_ten_traj_tags
    ]
    

    
    use_first_ten = False 
    # use_first_ten = True
    
    if len(args.single_instance_test_tag) != 0:
        use_first_ten = True
        first_ten_traj_tags = [args.single_instance_test_tag]
        
    print(f"first_ten_traj_tags: {first_ten_traj_tags}")
    
    if len(args.single_inst_tag) > 0:
        # tot_tracking_data = ["passive_active_info_" + args.single_inst_tag + ".npy"] + tot_tracking_data
        if args.hand_type == 'leap':
            tot_tracking_data = ["leap_passive_active_info_" + args.single_inst_tag + ".npy"]
        else:
            tot_tracking_data = ["passive_active_info_" + args.single_inst_tag + ".npy"]
    
    print(f"tot_tracking_data: {tot_tracking_data}")
    if args.launch_type == 'trajectory':
        
        tot_grab_data_tag = []
        for cur_tracking_data in tot_tracking_data: # cur tracking 
            cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
            
            if exclude_inst_tag_to_opt_res is not None and cur_grab_data_tag in exclude_inst_tag_to_opt_res:
                print(f"Exclude {cur_grab_data_tag}")
                continue
            
            if use_first_ten:
                if cur_grab_data_tag not in first_ten_traj_tags:
                    continue
                print(f"data in first ten data set: {cur_grab_data_tag}")
            
            traj_grab_data_tag = cur_grab_data_tag
            
            # if obj_type_to_pre_optimized_traj is not None:
            #     key_of_opt_traj = list(obj_type_to_pre_optimized_traj.keys())[0]
            #     if isinstance(key_of_opt_traj, tuple):
            #         if 'taco' in cur_grab_data_tag:
            #             cur_grab_data_tag_key = (cur_grab_data_tag, 'ori_grab_s2_phone_call_1')
            #         else:
            #             cur_grab_data_tag_key = ( cur_grab_data_tag, cur_grab_data_tag )
            #         # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
            #     else:
            #         cur_grab_data_tag_key = cur_grab_data_tag
            #         # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
            #     if cur_grab_data_tag_key in obj_type_to_pre_optimized_traj:
            #             # continue
            #         # print
            #         cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag_key ]
            #         # if isinstance(key_of_opt_traj, tuple):
            #         #     cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
            #         # else:
            #         #     cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
            #         cur_pre_optimized_traj = cur_pre_optimized_traj[0] # get the pre optimized traj #
            #         # 
            #         cur_pre_optimized_traj_sorted = cur_pre_optimized_traj.replace(".npy", "_sorted.npy")
            #         cur_pre_optimized_traj_sorted_best = cur_pre_optimized_traj_sorted.replace(".npy", "_best.npy")
            #         if not os.path.exists(cur_pre_optimized_traj_sorted_best):
            #             cur_pre_optimized_traj =  None
            #         else:
            #                 # continue
            #             cur_pre_optimized_traj = cur_pre_optimized_traj_sorted_best
            #     else:
            #         cur_pre_optimized_traj = None
            # else:
            #     cur_pre_optimized_traj = None
            
            # 
            
            if obj_type_to_base_trajs is not None:
                key_of_opt_traj = list(obj_type_to_base_trajs.keys())[0]
                if isinstance(key_of_opt_traj, tuple):
                    if 'taco' in cur_grab_data_tag:
                        cur_grab_data_tag_key = (cur_grab_data_tag, 'ori_grab_s2_phone_call_1')
                    else:
                        cur_grab_data_tag_key = ( cur_grab_data_tag, cur_grab_data_tag )
                else:
                    cur_grab_data_tag_key = cur_grab_data_tag
                if cur_grab_data_tag_key in obj_type_to_base_trajs:
                    cur_base_traj = obj_type_to_base_trajs[ cur_grab_data_tag_key ]
                    cur_base_traj = cur_base_traj[0]
                else:
                    cur_base_traj = None
            else:
                cur_base_traj = None
            
            if obj_type_to_pre_optimized_traj is not None:
                key_of_opt_traj = list(obj_type_to_pre_optimized_traj.keys())[0]
                
                # print(f"key_of_opt_traj: {key_of_opt_traj}, cur_grab_data_tag_key: {cur_grab_data_tag}")
                if isinstance(key_of_opt_traj, tuple):
                    if 'taco' in cur_grab_data_tag:
                        cur_grab_data_tag_key = (cur_grab_data_tag, 'ori_grab_s2_phone_call_1')
                    else:
                        cur_grab_data_tag_key = ( cur_grab_data_tag, cur_grab_data_tag )
                    # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
                else:
                    cur_grab_data_tag_key = cur_grab_data_tag
                    # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
                if cur_grab_data_tag_key in obj_type_to_pre_optimized_traj:
                        # continue
                    # print
                    cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag_key ]
                    # if isinstance(key_of_opt_traj, tuple):
                    #     cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
                    # else:
                    #     cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
                    cur_pre_optimized_traj = cur_pre_optimized_traj[0] # get the pre optimized traj #
                    # 
                    print(f"cur_pre_optimized_traj: {cur_pre_optimized_traj}")
                    cur_pre_optimized_traj_sorted = cur_pre_optimized_traj.replace(".npy", "_sorted.npy")
                    cur_pre_optimized_traj_sorted_best = cur_pre_optimized_traj_sorted.replace(".npy", "_best.npy")
                    if not os.path.exists(cur_pre_optimized_traj_sorted_best):
                        if args.train_on_all_trajs:
                            cur_pre_optimized_traj = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
                            if not os.path.exists(cur_pre_optimized_traj):
                                cur_pre_optimized_traj = "./data/GRAB_Tracking_PK_OFFSET_Reduced/data/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
                        else:
                            continue
                    else:
                        cur_pre_optimized_traj = cur_pre_optimized_traj_sorted_best
                else:
                    if args.train_on_all_trajs:
                        # get the pre optimized traj #
                        cur_pre_optimized_traj = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
                        if not os.path.exists(cur_pre_optimized_traj):
                            cur_pre_optimized_traj = "./data/GRAB_Tracking_PK_OFFSET_Reduced/data/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
                    else:
                        continue
            else:
                cur_pre_optimized_traj = None
                
            if obj_type_to_ckpt is not None:
                key_of_opt_traj = list(obj_type_to_ckpt.keys())[0]
                
                if isinstance(key_of_opt_traj, tuple):
                    if 'taco' in cur_grab_data_tag:
                        cur_grab_data_tag_key = (cur_grab_data_tag, 'ori_grab_s2_phone_call_1')
                    else:
                        cur_grab_data_tag_key = ( cur_grab_data_tag, cur_grab_data_tag )
                    # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_grab_data_tag, cur_grab_data_tag)]
                else:
                    cur_grab_data_tag_key = cur_grab_data_tag
                    # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ cur_grab_data_tag ]
                if cur_grab_data_tag_key not in obj_type_to_ckpt: # not in the dict #
                    continue
                cur_traj_opt_ckpt = obj_type_to_ckpt[ cur_grab_data_tag_key ][1]
            else:
                cur_traj_opt_ckpt = None
                
            
            # cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
            
            cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
            
            cur_cuda_idx = (cur_cuda_idx + args.gpu_offset_idx) % len(gpu_idx_list) # find cuda
            
            cur_cuda_idx = gpu_idx_list[cur_cuda_idx]
            
            
            tot_grab_data_tag.append(
                [cur_grab_data_tag, traj_grab_data_tag, cur_cuda_idx, cur_pre_optimized_traj, cur_traj_opt_ckpt, cur_base_traj]
            )
    elif args.launch_type == 'object_type': # # model and how to make the model use optimized trajectories # # # cur bae trajs # a new sample process # and an optimization with the smoothness added #
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
    
    # print(f"tot_tracking_data : {tot_tracking_data}")
    
    ### tot grab data tag ###
    ### tot grab data tag ###
    tot_grab_data_tag = tot_grab_data_tag[args.st_idx: ]
    
    if args.debug: # debug # # 
        tot_grab_data_tag = tot_grab_data_tag[:1]
    
    
    print(f"tot_grab_data_tag: {tot_grab_data_tag}")
    # # generalist_tune_all_instnaces 
    # if (not args.generalist_tune_all_instnaces) and (args.data_inst_flag is not None) and len(args.data_inst_flag) > 0:
    #     data_inst_flag = args.data_inst_flag
    #     cur_cuda_idx = args.st_idx
    #     if obj_type_to_pre_optimized_traj is not None:
    #         # key_of_opt_traj = obj_type_to_pre_optimized_traj.keys()[0]
    #         key_of_opt_traj = list(obj_type_to_pre_optimized_traj.keys())[0]
    #         if isinstance(key_of_opt_traj, tuple):
    #             cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ (data_inst_flag, data_inst_flag) ]
    #         else:
    #             cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ data_inst_flag ]
    #         ### ### xxx xxx ### ###
    #         # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ (data_inst_flag, data_inst_flag) ]
    #         # cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[ data_inst_flag ]
    #         cur_pre_optimized_traj = cur_pre_optimized_traj[0] 
    #     else:
    #         cur_pre_optimized_traj = None
    #     tot_grab_data_tag = [
    #         [data_inst_flag, data_inst_flag, cur_cuda_idx, cur_pre_optimized_traj]
    #     ]
    # obj type optimized traj #
    max_pool_size = nn_gpus * 1
    print(f"tot_grab_data_tag: {len(tot_grab_data_tag)}, max_pool_size: {max_pool_size}")
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
    
    
    # launch_rlg_hydra()

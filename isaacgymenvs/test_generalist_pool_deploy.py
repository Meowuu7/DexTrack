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
    parser.add_argument("--use_deploy_generalist", type=str2bool,  default=False)
    
    
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
                
                #### by default we use interpolated data #### # single trajectories #
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
        else: # model and the performance of the model #
            raise ValueError
        
        
        full_experiment_name = train_name
        
        if args.headless:
            capture_video = False
            force_render = False
        else:
            capture_video = True
            force_render = True
        
        # 
        # use_small_sigmas = "--use_small_sigmas" if args.use_small_sigmas else ""
        # use_relaxed_model = "--use_relaxed_model" if args.use_relaxed_model else ""
        # w_hand_table_collision = "--w_hand_table_collision" if args.w_hand_table_collision else ""
        
        
        print(f"test: {args.test}")
        
        if args.headless:
            cuda_visible_text = f"CUDA_VISIBLE_DEVICES={cuda_idx} "
        else:
            cuda_visible_text = ''
            
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
            
        # if args.use_vision:
        #     task_type = "AllegroHandTrackingVision"
        #     train_type = "HumanoidPPOVision"
            
        #     enableCameraSensors = True
        #     if args.use_dagger:
        #         # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/cfg/train/HumanoidPPOSupervised.yaml
        #         task_type = "AllegroHandTrackingVision"
        #         train_type = "HumanoidPPOVisionDAgger"
        #     print(f"task_type: {task_type}, train_type: {train_type}")
        # else:
            
        if args.use_generalist_policy:
            task_type = "AllegroHandTrackingGeneralist"
            train_type = "HumanoidPPO" 
            # test_inst_tag, test_optimized_res # optimized res # # optimized res # # optimized res # optimized res #
            if args.supervised_training:
                task_type = "AllegroHandTrackingGeneralist"
                train_type = "HumanoidPPOSupervised"
                
                
                if args.use_deploy_generalist:
                    task_type = "AllegroHandTrackingGeneralistDeploy"
                    train_type = "HumanoidPPOSupervised"
                
                if args.single_instance_state_based_train:
                    train_type = "HumanoidPPOSupervisedSN"
                    print(f"using SN")
                
                training_mode_config = f"train.params.config.training_mode={args.training_mode}"
                test_inst_config = f"task.env.test_inst_tag={args.test_inst_tag} task.env.test_optimized_res={args.test_optimized_res}"
                # # preload_experiences_tf, preload_experiences_path
                preload_experience_config = f"train.params.config.preload_experiences_tf={args.preload_experiences_tf} train.params.config.preload_experiences_path={args.preload_experiences_path}"
                single_instance_training_config = f"train.params.config.single_instance_training={args.single_instance_training}"
                
                # ### for the test setting -- ###
                sampleds_with_object_code_fn_config = f"task.env.sampleds_with_object_code_fn={args.sampleds_with_object_code_fn}" # no base trajs are usd during the traiing #
                
                # # ori grab s2 apple lift #
                # if args.generalist_tune_all_instnaces: # pre load # 
                test_inst_config = f"task.env.test_inst_tag={cur_grab_data_tag} task.env.test_optimized_res={pre_optimized_traj}"
                single_instance_training_config = f"train.params.config.single_instance_training={False}"
                preload_experience_config = f"train.params.config.preload_experiences_tf={False} train.params.config.preload_experiences_path={''}"

                # samples with code fn #
                log_path_config = f"train.params.config.log_path={args.log_path}"
                train_dir_config = f"train.params.config.train_dir={args.log_path}"
                single_instance_tag_config = f"train.params.config.single_instance_tag={args.single_instance_tag}"
                obj_type_to_optimized_res_fn_config = f"train.params.config.obj_type_to_optimized_res_fn={args.obj_type_to_optimized_res_fn}"
                supervised_loss_coef_config = f"train.params.config.supervised_loss_coef={args.supervised_loss_coef}"
                pure_supervised_training_config = f"train.params.config.pure_supervised_training={args.pure_supervised_training}"
                # use_deploy_generalist_config = f
                # inst_tag_to_latent_feature_fn_config = f"task.env.inst_tag_to_latent_feature_fn={args.inst_tag_to_latent_feature_fn}"
                # object_type_to_latent_feature_fn_config = f"task.env.object_type_to_latent_feature_fn={args.object_type_to_latent_feature_fn}"
                
            else:
                training_mode_config  = ""
                test_inst_config = ""
                single_instance_training_config = ""
                sampleds_with_object_code_fn_config = ""
                log_path_config = ""
                single_instance_tag_config = ""
                obj_type_to_optimized_res_fn_config = ""
                supervised_loss_coef_config = ""
                pure_supervised_training_config = ""
                inst_tag_to_latent_feature_fn_config = ""
                object_type_to_latent_feature_fn_config = ""
            
            # grab_opt_res_config  taco_opt_res_config
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
        else:
            
            task_type = "AllegroHandTracking"
            train_type = "HumanoidPPO"
            training_mode_config = ""
            test_inst_config = ""

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
            
        enableCameraSensors = False
    # if args.use_generalist_policy: #### taco ##### targets ######
    # grab_inst_tag_to_optimized_res_fn, taco_inst_tag_to_optimized_res_fn
        
        
        # if args.use_vision:
        #     cuda_visible_text = f"CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  " 
        #     # # 
        #     cmd = f"{cuda_visible_text} python train.py task={task_type} train={train_type} sim_device='cuda:{cuda_idx}' rl_device='cuda:{cuda_idx}'  capture_video={capture_video} force_render={force_render} headless={args.headless}   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={args.test} task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={cur_traj_checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.controlFrequencyInv={args.controlFrequencyInv} task.env.episodeLength={args.episodeLength} task.env.start_frame={args.start_frame} task.env.rew_obj_pose_coef={args.rew_obj_pose_coef} task.env.goal_dist_thres={args.goal_dist_thres} task.env.lifting_separate_stages={args.lifting_separate_stages} task.env.strict_lifting_separate_stages={args.strict_lifting_separate_stages} task.env.table_z_dim={args.table_z_dim} task.env.add_table={args.add_table} exp_dir={exp_dir} task.env.use_taco_obj_traj={args.use_taco_obj_traj} task.env.pre_optimized_traj={ cur_pre_optimized_traj } task.env.hand_type={ args.hand_type } enableCameraSensors={enableCameraSensors} graphics_device_id={cuda_idx}"
        # else:
        # deploy and the deploy jours #
        cmd = f"{cuda_visible_text} python deploy_ours.py task={task_type} train={train_type} sim_device='cuda:0' rl_device='cuda:0'  capture_video={capture_video} force_render={force_render} headless={args.headless}   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={ True } task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={cur_traj_checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.controlFrequencyInv={args.controlFrequencyInv} task.env.episodeLength={args.episodeLength} task.env.start_frame={args.start_frame} task.env.rew_obj_pose_coef={args.rew_obj_pose_coef} task.env.goal_dist_thres={args.goal_dist_thres} task.env.lifting_separate_stages={args.lifting_separate_stages} task.env.strict_lifting_separate_stages={args.strict_lifting_separate_stages} task.env.table_z_dim={args.table_z_dim} task.env.add_table={args.add_table} exp_dir={exp_dir} task.env.use_taco_obj_traj={args.use_taco_obj_traj} task.env.pre_optimized_traj={ cur_pre_optimized_traj } task.env.hand_type={ args.hand_type } enableCameraSensors={enableCameraSensors} graphics_device_id={cuda_idx} task.env.use_hand_actions_rew={args.use_hand_actions_rew} task.env.supervised_training={args.supervised_training} {training_mode_config} {test_inst_config} {preload_experience_config} {single_instance_training_config} {sampleds_with_object_code_fn_config} {log_path_config} {train_dir_config} {grab_opt_res_config}  {taco_opt_res_config} {single_instance_tag_config} {obj_type_to_optimized_res_fn_config} {supervised_loss_coef_config} {pure_supervised_training_config} {inst_tag_to_latent_feature_fn_config} {object_type_to_latent_feature_fn_config} {tracking_save_info_fn_config} {taco_interped_data_sv_additional_tag_config} {use_strict_maxx_nn_ts_config} {strict_maxx_nn_ts_config} {use_local_canonical_state_config} {use_base_traj_config} {base_traj_config} {customize_damping_config} {tracking_info_st_tag_config} {single_instance_state_based_train_config}" 
            # # # #
            
        print(cmd)
        os.system(cmd)
    
    
    
    cur_grab_data_tag = args.test_inst_tag
    launch_one_process(cur_grab_data_tag, cur_grab_data_tag, 0)
    exit(0)
    
    
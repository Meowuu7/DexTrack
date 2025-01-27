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
import numpy as np
# import random

from omegaconf import open_dict


from multiprocessing import Pool
from multiprocessing import Process
import argparse

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
    # use_taco_obj_traj
    parser.add_argument("--use_taco_obj_traj", type=str2bool,   default=True)
    # parser.add_argument("--", type=str2bool,   default=True)
    
    parser.add_argument("--exp_type", type=str, default='taco_obj_grab_seq')
    parser.add_argument("--taco_interped_data_sv_additional_tag", type=str, default='')
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res.npy 
    # 
    parser.add_argument("--excluded_fr_existing_res_fn", type=str, default='')
    parser.add_argument("--log_path", type=str, default='./runs')
    args = parser.parse_args()
    
    # 
    
    def launch_one_process(cur_grab_data_tag, traj_grab_data_tag, cuda_idx):
        
        obs_type = args.obs_type
        # use_small_sigmas = args.use_small_sigmas
        # finger_urdf_template = args.finger_urdf_template
        # finger_near_palm_joint_idx = args.finger_near_palm_joint_idx
        # constraint_level = args.constraint_level
        # object_type = cur_grab_data_tag
        object_name = cur_grab_data_tag
        
        if len(args.target_object_name) > 0:
            object_name = args.target_object_name # 
        
        # task_type = "mocap_tracking"
        if args.hand_type == 'allegro':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB_Tracking/data/passive_active_info_{traj_grab_data_tag}.npy"
            # tracking_data_sv_roo #
    
            if args.dataset_type == 'grab':
                mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
            elif args.dataset_type == 'taco':
                mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
                if args.use_interpolated_data:
                    # mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_modifed_interped_transformed.npy"
                    # passive_active_info_taco_20231104_203_zrot_3.141592653589793_modifed_interped # 
                    mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_zrot_3.141592653589793_modifed_interped.npy"
                    if not os.path.exists(mocap_sv_info_fn):
                        mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}_modifed_interped.npy"
            else:
                raise ValueError
    
        
        elif args.hand_type == 'leap':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB_Tracking/data/leap_passive_active_info_{traj_grab_data_tag}.npy"
            mocap_sv_info_fn = f"{args.tracking_data_sv_root}/leap_passive_active_info_{traj_grab_data_tag}.npy"
        else:
            raise ValueError
        # launch one process #
        
        
        
        print(f"mocap_sv_info_fn: {mocap_sv_info_fn}")
        
        if args.exp_type == 'taco_obj_grab_seq':
            args.use_taco_obj_traj = False ## not use the taco_obj_traj ##
        elif args.exp_type == 'taco_obj_taco_seq':
            args.use_taco_obj_traj = True
        #     ## do not use 
        #     args.additional_tag = f"TACOGRABSEQ_{object_name}_" ### 
        
        checkpoint = ''
        tag = f"tracking_{object_name}"
        
        if args.exp_type == 'taco_obj_grab_seq':
            train_name = f"tracking_TACO_{object_name}_GRABSEQ_{traj_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
        elif args.exp_type == 'taco_obj_taco_seq':
            train_name = f"tracking_TACO_{object_name}_GRABHANDSEQ_{traj_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
        elif args.exp_type == 'taco_interp_seq':
            # taco_interped_data_sv_additional_tag #
            if len(args.taco_interped_data_sv_additional_tag) == 0:
                train_name = f"tracking_TACO_{object_name}_INTERPSEQ_{traj_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
                mocap_sv_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{object_name}_v2.npy'
            else:
                #### set train_name with the taco interped data sv additional tag ####
                train_name = f"tracking_TACO_{object_name}_INTERPSEQ_{traj_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}_{args.taco_interped_data_sv_additional_tag}"
                # mocap_sv_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{object_name}_v2.npy'
                ###### taco_interped_data_sv_additional_tag ######
                mocap_sv_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{object_name}_v2_{args.taco_interped_data_sv_additional_tag}.npy'
        elif args.exp_type == 'taco_interp_seq_v1':
            train_name = f"tracking_TACO_{object_name}_INTERPSEQV1_{traj_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
            mocap_sv_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{object_name}_v1.npy'
        else:
            raise ValueError(f"Unrecognized exp_type: {args.exp_type}")
        
        # train_name = f"tracking_{object_name}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
        full_experiment_name = train_name
        
        if args.headless:
            capture_video = False
            force_render = False
        else:
            capture_video = True
            force_render = True
        
        # use_small_sigmas = "--use_small_sigmas" if args.use_small_sigmas else ""
        # use_relaxed_model = "--use_relaxed_model" if args.use_relaxed_model else ""
        # w_hand_table_collision = "--w_hand_table_collision" if args.w_hand_table_collision else ""
        # 
        
        
        print(f"test: {args.test}")
        
        if args.headless:
            cuda_visible_text = f"CUDA_VISIBLE_DEVICES={cuda_idx} "
        else:
            cuda_visible_text = ''
            
        exp_dir = '/cephfs/xueyi/exp/IsaacGymEnvs/isaacgymenvs'
        if not os.path.exists(exp_dir):
            exp_dir = '.'
        
        # if len(args.target_object_name) > 0:
        #     object_name=args.target_object_name
        
        if len(args.target_mocap_sv_fn) > 0:
            mocap_sv_info_fn = args.target_mocap_sv_fn
        
        
        trian_log_path_config = f"train.params.config.log_path={args.log_path} train.params.config.train_dir={args.log_path}"
        
        
        # if args.headless: # add the taco instance tag # add the taco instance ? #
        cmd = f"{cuda_visible_text} python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video={capture_video} force_render={force_render} headless={args.headless}   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={args.test} task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={args.checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.controlFrequencyInv={args.controlFrequencyInv} task.env.episodeLength={args.episodeLength} task.env.start_frame={args.start_frame} task.env.rew_obj_pose_coef={args.rew_obj_pose_coef} task.env.goal_dist_thres={args.goal_dist_thres} task.env.lifting_separate_stages={args.lifting_separate_stages} task.env.strict_lifting_separate_stages={args.strict_lifting_separate_stages} task.env.table_z_dim={args.table_z_dim} task.env.add_table={args.add_table} exp_dir={exp_dir} task.env.use_taco_obj_traj={args.use_taco_obj_traj} {trian_log_path_config}"
        
        print(cmd)
        os.system(cmd)
    
    # trian pool
    
    # base_dir = '/cephfs/yilaa/uni_manip/tds_rl_exp'
    tracking_data_sv_root = args.tracking_data_sv_root
    
    
    ###### 
    # if args.dataset_type == 'grab':
    #     # passive_active_info_ori_grab_s2_pyramidlarge_lift.npy
    #     starting_str = "passive_active_info_ori_grab_"
    #     passive_active_info_tag = "passive_active_info_"
    #     tot_tracking_data = os.listdir(tracking_data_sv_root)
    #     if args.num_frames == 150:
    #         tot_tracking_data = [fn for fn in tot_tracking_data if fn[: len(starting_str)] == starting_str and fn.endswith(".npy") and "_nf_" not in fn]
    #     else:
    #         nf_tag = f"_nf_{args.num_frames}"
    #         tot_tracking_data = [fn for fn in tot_tracking_data if fn[: len(starting_str)] == starting_str and fn.endswith(".npy") and nf_tag in fn]
        
    #     if len(args.subj_nm) > 0:
    #         subj_tag = f"_{args.subj_nm}_"
    #         tot_tracking_data = [fn for fn in tot_tracking_data if subj_tag in fn]
    # elif args.dataset_type == 'taco':
    #     taso_inst_st_flag = 'taco_'
    #     mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    #     if not os.path.exists(mesh_sv_root):
    #         mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    #     tot_mesh_folders = os.listdir(mesh_sv_root)
    #     # find meshes directly #
    #     tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
    #     # modified_tag = "_modifed"
    #     # interped_tag = "_interped"
    #     # find tracking data #
    #     tot_tracking_data = tot_mesh_folders  # get the tracking data 
    #     passive_active_info_tag = ''
    
    # else:
    #     raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}")
    
    if args.exp_type == 'taco_obj_grab_seq':
        taso_inst_st_flag = 'taco_'
        mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        if not os.path.exists(mesh_sv_root):
            mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        tot_mesh_folders = os.listdir(mesh_sv_root)
        # find meshes directly #
        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        tot_tracking_data = tot_mesh_folders
        # if args.debug:
            
        passive_active_info_tag = ''
    elif args.exp_type == 'taco_obj_taco_seq':
        # runs/tracking_TACO_taco_20231104_205_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_23-15-53-42/nn/tracking_TACO_taco_20231104_205_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
        # obj_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{self.object_name}_zrot_3.141592653589793_modifed_interped.npy'
        taso_inst_st_flag = 'taco_'
        mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        if not os.path.exists(mesh_sv_root):
            mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        tot_mesh_folders = os.listdir(mesh_sv_root)
        # find meshes directly #
        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        
        ## tot mesh folders ##
        forbid_taco_cat_tag = "taco_20231104_"
        tot_mesh_folders = [ fn for fn in tot_mesh_folders if forbid_taco_cat_tag not in fn]
        
        tot_tracking_data = tot_mesh_folders
        # if args.debug: ## # args.debug ##
            
        passive_active_info_tag = ''

        new_tot_tracking_data = []
        for cur_mesh_fn in tot_mesh_folders:
            cur_grab_data_tag = cur_mesh_fn.split(".")[0][len(passive_active_info_tag):]
            cur_grab_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{cur_grab_data_tag}_zrot_3.141592653589793_modifed_interped.npy'
            if os.path.exists(cur_grab_mocap_info_fn):
                new_tot_tracking_data.append(cur_mesh_fn)
        tot_tracking_data = new_tot_tracking_data
    elif args.exp_type == 'taco_interp_seq':
        ##### Get all candidate data via the pre-saved meshes and the retargeted mocap data ####
        # obj_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{self.object_name}_zrot_3.141592653589793_modifed_interped.npy'
        taso_inst_st_flag = 'taco_'
        mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        if not os.path.exists(mesh_sv_root):
            mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        tot_mesh_folders = os.listdir(mesh_sv_root)

        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        tot_tracking_data = tot_mesh_folders
            
        passive_active_info_tag = ''

        new_tot_tracking_data = []
        for cur_mesh_fn in tot_mesh_folders:
            cur_grab_data_tag = cur_mesh_fn.split(".")[0][len(passive_active_info_tag):]
            # cur_grab_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{cur_grab_data_tag}_zrot_3.141592653589793_modifed_interped.npy'
            if len(args.taco_interped_data_sv_additional_tag) == 0:
                cur_grab_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{cur_grab_data_tag}_v2.npy'
            else: #
                cur_grab_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{cur_grab_data_tag}_v2_{args.taco_interped_data_sv_additional_tag}.npy'
            if os.path.exists(cur_grab_mocap_info_fn):
                new_tot_tracking_data.append(cur_mesh_fn)
        tot_tracking_data = new_tot_tracking_data
    
    elif args.exp_type == 'taco_interp_seq_v1':
        # obj_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{self.object_name}_zrot_3.141592653589793_modifed_interped.npy'
        taso_inst_st_flag = 'taco_'
        mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        if not os.path.exists(mesh_sv_root):
            mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
        tot_mesh_folders = os.listdir(mesh_sv_root)
        # find meshes directly #
        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        tot_tracking_data = tot_mesh_folders
        # if args.debug:
            
        passive_active_info_tag = ''

        new_tot_tracking_data = []
        for cur_mesh_fn in tot_mesh_folders:
            cur_grab_data_tag = cur_mesh_fn.split(".")[0][len(passive_active_info_tag):]
            # cur_grab_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{cur_grab_data_tag}_zrot_3.141592653589793_modifed_interped.npy'
            cur_grab_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{cur_grab_data_tag}_v1.npy'
            if os.path.exists(cur_grab_mocap_info_fn):
                new_tot_tracking_data.append(cur_mesh_fn)
        tot_tracking_data = new_tot_tracking_data
    
    else:
        
        
        raise NotImplementedError
        
    
    if len(args.excluded_fr_existing_res_fn) > 0:
        existing_data_inst_to_optimized_res = np.load(args.excluded_fr_existing_res_fn, allow_pickle=True).item()
        print(f"existing_data_inst_to_optimized_res: {len(existing_data_inst_to_optimized_res)}")
    else:
        existing_data_inst_to_optimized_res = {}
    
    nn_gpus = args.nn_gpus
    
    if args.launch_type == 'trajectory':
        
        tot_grab_data_tag = []
        for cur_tracking_data in tot_tracking_data:
            ### get the tracking data --- passive active info tag ### ### passive active info tag ###
            cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
            traj_grab_data_tag = args.data_inst_flag
            #
            if cur_grab_data_tag in existing_data_inst_to_optimized_res: # optimized res #
                continue
            
            cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
            tot_grab_data_tag.append(
                [cur_grab_data_tag, traj_grab_data_tag, cur_cuda_idx]
            )
    else:
        raise ValueError(f"Launch type {args.launch_type} not supported")

    
    # print()
    print(f"tot_tracking_data : {len(tot_grab_data_tag)}, first data: {tot_grab_data_tag[0]}")
    
    # tot grab data tag #
    tot_grab_data_tag = tot_grab_data_tag[args.st_idx: ]
    
    if args.debug:
        tot_grab_data_tag = tot_grab_data_tag[:1]
        
    # if args.data_inst_flag is not None and len(args.data_inst_flag) > 0:
    #     data_inst_flag = args.data_inst_flag
    #     cur_cuda_idx = args.st_idx
    #     tot_grab_data_tag = [
    #         [data_inst_flag, data_inst_flag, cur_cuda_idx] # # # re-luanch the related exps? ## 
    #     ]  #### a new exp for searching? ####
    
    max_pool_size = nn_gpus * 1
    
    for i_st in range(0, len(tot_grab_data_tag), max_pool_size):
        i_ed = i_st + max_pool_size
        i_ed = min(i_ed, len(tot_grab_data_tag))
        cur_batch_grab_data_tags = tot_grab_data_tag[i_st: i_ed]
        
        cur_thread_processes = []
        
        for cur_grab_data_tag in cur_batch_grab_data_tags:
            # for cur_grab_data_tag in tot_grab_data_tag:
            # existing = judge_whether_trained(tot_tracking_logs, cur_grab_data_tag)
            # if existing:
            #     print(f"cur_grab_data_tag: {cur_grab_data_tag} has been trained")
            #     continue
            
            cur_thread_processes.append(
                Process(target=launch_one_process, args=(cur_grab_data_tag))
            )
            
            cur_thread_processes[-1].start()
        for p in cur_thread_processes:
            p.join()
    
    
    # launch_rlg_hydra()

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
import numpy as np 
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
    # simreal_modeling
    
    ### sim steup ###
    parser.add_argument("--ground_distance", type=float, default=0.0)
    parser.add_argument("--use_canonical_state",  type=str2bool,   default=False) 
    parser.add_argument("--disable_gravity",  type=str2bool,   default=False) 
    parser.add_argument("--data_inst_flag", type=str, default='')
    # right_hand_dist_thres #
    parser.add_argument("--right_hand_dist_thres", type=float, default=0.12)
    parser.add_argument("--checkpoint", type=str, default='')
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--optimized_data_sv_root", type=str, default='')
    parser.add_argument("--obj_type_to_optimized_res_sv_fn", type=str, default='')
    # test full for the real testing trajs #
    # tracking_data_sv_root = /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab
    # pre_load_trajectories, obj_type_to_pre_optimized_traj
    ### TODO: add pre_load_trajectories, obj_type_to_pre_optimized_traj ###
    parser.add_argument("--pre_load_trajectories", type=str2bool,   default=False)
    parser.add_argument("--obj_type_to_pre_optimized_traj", type=str, default='')
    # 
    # 
    args = parser.parse_args()
    
    def parse_obj_type_fr_folder_name(folder_nm):
        # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_hammer_use_1_obs_pure_state_wref_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t1r1_rewfingerdist_0.5_rewdeltahandpose_0.0_10-17-19-17
        folder_st_tag = "tracking_"
        remains_folder_nm = folder_nm.split("/")[-1][len(folder_st_tag): ]
        
        folder_nm_segs = remains_folder_nm.split("_")
        st_idx = 0
        for ed_idx in range(st_idx, len(folder_nm_segs)):
            cur_seg = folder_nm_segs[ed_idx]
            if cur_seg == 'obs':
                break
        obj_type = folder_nm_segs[st_idx: ed_idx]
        obj_type = "_".join(obj_type)
        return obj_type
    
    def find_best_rew(folder_nm):
        nn_folder = os.path.join(folder_nm, "nn")
        if not os.path.exists(nn_folder):
            return -9999.9, None
        tot_ckpts = os.listdir(nn_folder)
        tot_ckpts = [
            fn for fn in tot_ckpts if fn.endswith(".pth")
        ]
        tot_rews = []
        cur_best_rew = -9999.9
        best_ckpt_fn = None
        for cur_ckpt_fn in tot_ckpts:
            # cur_full_ckpt_fn = os.path.join(nn_folder, cur_ckpt_fn)
            cur_ckpt_pure_fn = cur_ckpt_fn.split(".pth")[0]
            cur_ckpt_pure_fn_segs = cur_ckpt_pure_fn.split("_")
            try:
                if len(cur_ckpt_pure_fn_segs[-1]) == 0:
                    cur_rew = float(cur_ckpt_pure_fn_segs[-2])
                else:
                    cur_rew = float(cur_ckpt_pure_fn_segs[-1])
            except:
                print(cur_ckpt_pure_fn_segs)
                continue
            tot_rews.append(cur_rew)
            if cur_rew > cur_best_rew:
                cur_best_rew = cur_rew
                best_ckpt_fn = os.path.join(nn_folder, cur_ckpt_fn)
        maxx_rew = max(tot_rews)
        # 
        return cur_best_rew, best_ckpt_fn
    
    
    
    
    def launch_one_process(cur_grab_data_tag, traj_grab_data_tag, checkpoint, cuda_idx, pre_optimized_traj=None): 
        
        obs_type = args.obs_type
        # use_small_sigmas = args.use_small_sigmas
        # finger_urdf_template = args.finger_urdf_template
        # finger_near_palm_joint_idx = args.finger_near_palm_joint_idx #
        # constraint_level = args.constraint_level
        # object_type = cur_grab_data_tag #
        object_name = cur_grab_data_tag
        if args.hand_type == 'allegro':
            if args.dataset_type == 'grab':
                mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
            elif args.dataset_type == 'taco':
                mocap_sv_info_fn = f"{args.tracking_data_sv_root}/passive_active_info_{traj_grab_data_tag}.npy"
            else:
                raise ValueError
        
        elif args.hand_type == 'leap':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB_Tracking/data/leap_passive_active_info_{traj_grab_data_tag}.npy"
            mocap_sv_info_fn = f"{args.tracking_data_sv_root}/leap_passive_active_info_{traj_grab_data_tag}.npy"
        else:
            raise ValueError
        
        print(f"mocap_sv_info_fn: {mocap_sv_info_fn}")
        
        # checkpoint = ''
        tag = f"tracking_{object_name}"
        train_name = f"tracking_{object_name}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
        full_experiment_name = train_name
        
        
        
        if pre_optimized_traj is not None and len(pre_optimized_traj) > 0:
            cur_pre_optimized_traj = pre_optimized_traj
        else:
            cur_pre_optimized_traj = ''
        
        # use_small_sigmas = "--use_small_sigmas" if args.use_small_sigmas else ""
        # use_relaxed_model = "--use_relaxed_model" if args.use_relaxed_model else ""
        # w_hand_table_collision = "--w_hand_table_collision" if args.w_hand_table_collision  else ""
        # w hand table collision
        
        # cmd = f"CUDA_VISIBLE_DEVICES={cuda_idx} python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={args.test} task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={args.checkpoint}"

        # print(f"test: {args.test}")
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_idx} python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs=10000 task.env.mocap_sv_info_fn={mocap_sv_info_fn} checkpoint={checkpoint} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={ True } task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type} task.env.use_canonical_state={args.use_canonical_state} task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density} task.env.use_unified_canonical_state={ False } task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.disable_obj_gravity={args.disable_gravity} task.env.ground_distance={args.ground_distance} task.env.right_hand_dist_thres={args.right_hand_dist_thres} task.env.pre_optimized_traj={ cur_pre_optimized_traj } "
        print(cmd)
        os.system(cmd)
    
    
    
    def get_obj_type_to_optimized_res(optimized_root_folder):
        tot_folders = os.listdir(optimized_root_folder)
        tracking_st_tag = 'tracking_'
        tot_folders = [
            fn for fn in tot_folders if fn[: len(tracking_st_tag)] == tracking_st_tag
        ]
        obj_type_to_optimized_res = {}
        for cur_folder in tot_folders:
            cur_full_folder = os.path.join(optimized_root_folder, cur_folder)
            cur_obj_type = parse_obj_type_fr_folder_name(cur_full_folder)
            
            cur_best_rew, best_ckpt_fn = find_best_rew(cur_full_folder)
            if best_ckpt_fn is None:
                continue
            obj_type_to_optimized_res[cur_obj_type] = (cur_best_rew, best_ckpt_fn)
        return obj_type_to_optimized_res
    
    optimized_data_sv_root = args.optimized_data_sv_root
    
    tracking_data_statistics_folder = os.path.join(optimized_data_sv_root, "statistics")
    os.makedirs(tracking_data_statistics_folder, exist_ok=True)
    # if obj type to optimized res 
    if args.obj_type_to_optimized_res_sv_fn is None or len(args.obj_type_to_optimized_res_sv_fn) == 0:
        obj_type_to_optimized_res_sv_fn = "obj_type_to_optimized_res.npy"
    else:
        obj_type_to_optimized_res_sv_fn = args.obj_type_to_optimized_res_sv_fn
    obj_type_to_optimized_res_sv_fn = os.path.join(tracking_data_statistics_folder, obj_type_to_optimized_res_sv_fn) 
    if not os.path.exists(obj_type_to_optimized_res_sv_fn):
        obj_type_to_optimized_res = get_obj_type_to_optimized_res(optimized_data_sv_root) # get the tracking data sv root #
        print(f"obj_type_to_optimized_res: {obj_type_to_optimized_res}")
        np.save(obj_type_to_optimized_res_sv_fn, obj_type_to_optimized_res) # save the optimized res
        print(f"obj_type_to_optimized_res saved to {obj_type_to_optimized_res_sv_fn}")
    # obj type to optimized res #
    else:
        obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_sv_fn, allow_pickle=True).item()
    
    
    
    
    # TODO: get the traj relations from previously calculated res for bullet #
    # obj type to optimized res #
    
    tot_grab_data_tag = []
    
    
    
    # # base_dir = '/cephfs/yilaa/uni_manip/tds_rl_exp'
    # tracking_data_sv_root = args.tracking_data_sv_root
    
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
    #     tot_mesh_folders = os.listdir(mesh_sv_root)
    #     tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
    #     tot_tracking_data = tot_mesh_folders  # get the tracking data 
    #     passive_active_info_tag = ''
    
    # else:
    #     raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}")
    
    nn_gpus = args.nn_gpus
    
    ### and also add the grab instance to the optimized res dict ###
    pre_load_trajectories = args.pre_load_trajectories
    print(f"pre_load_trajectories: {pre_load_trajectories}")
    if pre_load_trajectories: # load pre optimized trajectories #
        obj_type_to_pre_optimized_traj = args.obj_type_to_pre_optimized_traj 
        assert len(obj_type_to_pre_optimized_traj) > 0 and os.path.exists(obj_type_to_pre_optimized_traj)
        obj_type_to_pre_optimized_traj = np.load(obj_type_to_pre_optimized_traj, allow_pickle=True).item()
        ### #### obj type to pre optimized traj #### ###
    else:
        obj_type_to_pre_optimized_traj = None
    
    # if args.launch_type == 'trajectory':
        
    #     tot_grab_data_tag = []
    #     for cur_tracking_data in tot_tracking_data:
            
    #         # if args.
    #         cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
    #         traj_grab_data_tag = cur_grab_data_tag
            
    #         cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
    #         tot_grab_data_tag.append(
    #             [cur_grab_data_tag, traj_grab_data_tag, cur_cuda_idx]
    #         )
    # else:
    #     raise ValueError(f"Launch type {args.launch_type} not supported")

    
    
    # print(f"tot_tracking_data : {tot_tracking_data}")
    

    # tot_grab_data_tag = tot_grab_data_tag[args.st_idx: ]
    
    # if args.debug:
    #     tot_grab_data_tag = tot_grab_data_tag[:1]
        
    cur_gpu_idx = 0
    for cur_obj_type in obj_type_to_optimized_res:
        if args.subj_nm is not None and len(args.subj_nm) > 0:
            if f"_{args.subj_nm}_" not in cur_obj_type: # cur obj type 
                continue
        cur_optimized_res = obj_type_to_optimized_res[cur_obj_type]
        opt_rew, opt_ckpt_fn = cur_optimized_res[0], cur_optimized_res[1]
        
        if obj_type_to_pre_optimized_traj is not None:
            cur_pre_optimized_traj = obj_type_to_pre_optimized_traj[(cur_obj_type, cur_obj_type)]
            cur_pre_optimized_traj = cur_pre_optimized_traj[0] # get the pre optimized traj #
        else:
            cur_pre_optimized_traj = None
    
        tot_grab_data_tag.append(
            (cur_obj_type, cur_obj_type, opt_ckpt_fn, (cur_gpu_idx % nn_gpus), cur_pre_optimized_traj)
        )
        cur_gpu_idx = (cur_gpu_idx + 1 ) % nn_gpus
        
    if args.debug:
        tot_grab_data_tag = tot_grab_data_tag[:1]
        
    # if args.data_inst_flag is not None and len(args.data_inst_flag) > 0:
    #     data_inst_flag = args.data_inst_flag
    #     cur_cuda_idx = args.st_idx
    #     tot_grab_data_tag = [
    #         [data_inst_flag, data_inst_flag, cur_cuda_idx]
    #     ]
    
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
    
    
    
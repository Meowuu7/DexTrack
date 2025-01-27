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
import numpy as np
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

# train selfexp # # train  online evolved # # offline evolved # # as the base trajectories; identify good baseline trajectories  ---- we can use per-train but with different baseline trajectories #
# sue the offlien strateyg 

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    ##### pool settings ####
    parser.add_argument("--launch_type", type=str, default='trajectory')
    parser.add_argument("--tracking_data_sv_root", type=str, default='/cephfs/yilaa/data/GRAB_Tracking/data')
    parser.add_argument("--subj_nm", type=str, default='')
    parser.add_argument("--debug", action='store_true', default=False)
    parser.add_argument("--num_frames", type=int, default=150, help="number of vector envs")
    parser.add_argument("--base_dir", type=str, default='/cephfs/yilaa/uni_manip/tds_rl_exp', help="Mocap save info file")
   
   
    ##### experiment settings ##### #
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
    parser.add_argument("--subj_idx", type=int, default=2)
    parser.add_argument("--obj_type_to_optimized_traj_fn", type=str, default='') # obj type to optimized traj fn #
    # 
    # 
    args = parser.parse_args()
    
    
    # grab_data_nm_idx_dict #
    grab_tracking_data_root = args.tracking_data_sv_root
    if args.subj_idx == 2 or args.subj_idx < 1:
        data_nm_idx_dict_sv_fn = "grab_data_nm_idx_dict.npy"
    else:
        data_nm_idx_dict_sv_fn = f"grab_data_nm_idx_dict_s{args.subj_idx}.npy"
    data_nm_idx_dict_sv_fn = os.path.join(grab_tracking_data_root, data_nm_idx_dict_sv_fn)
    data_nm_idx_dict = np.load(data_nm_idx_dict_sv_fn, allow_pickle=True).item()
    data_nm_to_idx = data_nm_idx_dict['data_nm_to_idx'] # idx to data nm # 
    idx_to_data_nm = data_nm_idx_dict['idx_to_data_nm'] # data nm to idx # 
    
    # add the train --- nmae # 
    # tracking_object_type_OPTFR_xxx # 
    # add the tracking #
    
    # if '_nf_' in args.object_type:
    #     pure_obj_type = args.object_type.split('_nf_')[0]
    # else:
    #     pure_obj_type = args.object_type
    # cur_idx = data_nm_to_idx[pure_obj_type]
    
    def find_topk_nearest_trajs(cur_idx, topk=10, traj_tracking_dir="/cephfs/yilaa/data/GRAB_Tracking/data", subj_idx=2):
        # traj_tracking_dir = "/cephfs/yilaa/data/GRAB_Tracking/data"
        if subj_idx == 2 or subj_idx < 1: # find the subj idx #
            grab_diff_arr_fn = f"grab_diff_arr.npy"
        else:
            grab_diff_arr_fn = f"grab_diff_arr_s{subj_idx}.npy"
        # grab_diff_arr_fn = "grab_diff_arr.npy"
        grab_diff_arr_fn = os.path.join(traj_tracking_dir, grab_diff_arr_fn) 
        grab_diff_arr = np.load(grab_diff_arr_fn) # grab_diff_arr:  nn_seq x nn_seq
        cur_seq_diff_arr = grab_diff_arr[cur_idx]
        cur_seq_sorted_neighbours = np.argsort(cur_seq_diff_arr, axis=0) # 
        cur_seq_sorted_neighbours = cur_seq_sorted_neighbours[1: 1 + topk]
        cur_seq_sorted_neighbours = cur_seq_sorted_neighbours.tolist()
        return cur_seq_sorted_neighbours
        ## TODO: load the idx to seq name array ###
        ## TODO: get curresponding object_name and traj_name ##
        pass
    
    #  #
    def load_obj_type_to_optimized_traj(obj_type_to_optimized_traj_fn):
        obj_type_to_optimized_traj = np.load(obj_type_to_optimized_traj_fn, allow_pickle=True).item()
        return obj_type_to_optimized_traj
    
    def find_similar_objs(obj_index):
        grab_cross_obj_diff_arr_fn = "../assets/grab_cross_obj_verts_diff.npy"
        grab_cross_obj_diff_arr = np.load(grab_cross_obj_diff_arr_fn)
        cur_obj_diff_arr = grab_cross_obj_diff_arr[obj_index]
        cur_obj_sorted_nei_idxes = np.argsort(cur_obj_diff_arr, axis=0)
        cur_obj_sorted_nei_idxes = cur_obj_sorted_nei_idxes[1: 1 + 10]
        cur_obj_sorted_nei_idxes = cur_obj_sorted_nei_idxes.tolist()
        return cur_obj_sorted_nei_idxes

    
    def launch_one_process(cur_grab_data_tag, fa_grab_data_tag, fa_grab_optimized_res_fn, cuda_idx): # obj-tag and the traj-obj-tag #
        
        obs_type = args.obs_type
        # use_small_sigmas = args.use_small_sigmas
        # finger_urdf_template = args.finger_urdf_template
        # finger_near_palm_joint_idx = args.finger_near_palm_joint_idx
        # constraint_level = args.constraint_level
        # object_type = cur_grab_data_tag
        # 
        object_name = cur_grab_data_tag
        traj_grab_data_tag = cur_grab_data_tag
        # task_type = "mocap_tracking"
        if args.hand_type == 'allegro':
            # mocap_save_info_fn = f"/cephfs/yilaa/data/GRAB_Tracking/data/passive_active_info_{traj_grab_data_tag}.npy"
            # tracking_data_sv_roo #
    
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
        # launch one process #
        
        print(f"mocap_sv_info_fn: {mocap_sv_info_fn}")
        
        
        checkpoint = ''
        tag = f"tracking_{object_name}"
        
        train_name = f"tracking_{object_name}_OPTFR_{fa_grab_data_tag}_obs_{obs_type}_density_{args.rigid_obj_density}_trans_{args.glb_trans_vel_scale}_rot_{args.glb_rot_vel_scale}_goalcond_{args.goal_cond}_{args.additional_tag}"
        
        full_experiment_name = train_name
        
        pre_optimized_traj = fa_grab_optimized_res_fn # get teh grab d
        
        # use_small_sigmas = "--use_small_sigmas" if args.use_small_sigmas else ""
        # use_relaxed_model = "--use_relaxed_model" if args.use_relaxed_model else ""
        # w_hand_table_collision = "--w_hand_table_collision" if args.w_hand_table_collision  else ""
        # w hand table collision # == == #

        print(f"test: {args.test}")
        cmd = f"CUDA_VISIBLE_DEVICES={cuda_idx} python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs={args.numEnvs} train.params.config.minibatch_size={args.minibatch_size}  task.env.useRelativeControl={args.use_relative_control}  train.params.config.max_epochs={args.max_epochs} task.env.mocap_sv_info_fn={mocap_sv_info_fn} task.env.goal_cond={args.goal_cond} task.env.object_name={object_name} tag={tag} train.params.config.name={train_name} train.params.config.full_experiment_name={full_experiment_name} task.sim.dt={args.dt} test={args.test} task.env.use_kinematics_bias={args.use_kinematics_bias} task.env.w_obj_ornt={args.w_obj_ornt} task.env.observationType={obs_type}  task.env.separate_stages={args.separate_stages} task.env.rigid_obj_density={args.rigid_obj_density}   task.env.kinematics_only={args.kinematics_only}  task.env.use_fingertips={args.use_fingertips}  task.env.glb_trans_vel_scale={args.glb_trans_vel_scale} task.env.glb_rot_vel_scale={args.glb_rot_vel_scale} task.env.use_kinematics_bias_wdelta={args.use_kinematics_bias_wdelta} task.env.hand_pose_guidance_glb_trans_coef={args.hand_pose_guidance_glb_trans_coef} task.env.hand_pose_guidance_glb_rot_coef={args.hand_pose_guidance_glb_rot_coef} task.env.hand_pose_guidance_fingerpose_coef={args.hand_pose_guidance_fingerpose_coef} task.env.rew_finger_obj_dist_coef={args.rew_finger_obj_dist_coef} task.env.rew_delta_hand_pose_coef={args.rew_delta_hand_pose_coef} task.env.dofSpeedScale={args.dofSpeedScale} task.env.use_twostage_rew={args.use_twostage_rew} task.env.ground_distance={args.ground_distance} task.env.use_canonical_state={args.use_canonical_state} task.env.disable_obj_gravity={args.disable_gravity} train.params.config.save_best_after=50 task.env.right_hand_dist_thres={args.right_hand_dist_thres} checkpoint={args.checkpoint} task.env.use_real_twostage_rew={args.use_real_twostage_rew} task.env.start_grasping_fr={args.start_grasping_fr} task.env.pre_optimized_traj={pre_optimized_traj}" #
        print(cmd)
        os.system(cmd)
    
    # trian pool 
    # 
    
    obj_type_to_optimized_traj = load_obj_type_to_optimized_traj(args.obj_type_to_optimized_traj_fn) # get he traj fn #
    # base_dir = '/cephfs/yilaa/uni_manip/tds_rl_exp'
    tracking_data_sv_root = args.tracking_data_sv_root
    
    if args.dataset_type == 'grab':
        # passive_active_info_ori_grab_s2_pyramidlarge_lift.npy
        starting_str = "passive_active_info_ori_grab_"
        passive_active_info_tag = "passive_active_info_"
        tot_tracking_data = os.listdir(tracking_data_sv_root)
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
        tot_mesh_folders = os.listdir(mesh_sv_root)
        tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag]
        tot_tracking_data = tot_mesh_folders  # get the tracking data 
        passive_active_info_tag = ''
    else:
        raise ValueError(f"Unrecognized dataset_type: {args.dataset_type}")
    
    nn_gpus = args.nn_gpus
    
    if args.launch_type == 'trajectory':
        
        tot_grab_data_tag = []
        for cur_tracking_data in tot_tracking_data:
            
            
            cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
            traj_grab_data_tag = cur_grab_data_tag
            
            if '_nf_' in cur_grab_data_tag:
                pure_obj_type = cur_grab_data_tag.split('_nf_')[0] #
            else:
                pure_obj_type = cur_grab_data_tag
            cur_idx = data_nm_to_idx[pure_obj_type]
            ## subject idx ##
            cur_seq_sorted_neighbours = find_topk_nearest_trajs(cur_idx, topk=10, traj_tracking_dir=tracking_data_sv_root, subj_idx=args.subj_idx)
            
            ### i_traj and the traj_idx ###
            for i_traj, traj_idx in enumerate(cur_seq_sorted_neighbours):
                cur_obj_name = idx_to_data_nm[traj_idx]
                cur_traj_name = cur_obj_name
                
                if isinstance(list(obj_type_to_optimized_traj.keys())[0], tuple):
                    if (cur_obj_name, cur_traj_name) in obj_type_to_optimized_traj:
                        cur_obj_optimized_fn = obj_type_to_optimized_traj[(cur_obj_name, cur_traj_name)][0] ### get the optimized obj type and the traj name ###
                        ### cur obj optimized fn ###
                        # cur_obj_optimized_fn
                    else:
                        continue
                else:
                    if cur_obj_name in obj_type_to_optimized_traj:
                        cur_obj_optimized_fn = obj_type_to_optimized_traj[cur_obj_name] # get the optimized traj # 
                    else:
                        continue
                
                ## get the self-exp ## self-exp ##
                cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
                tot_grab_data_tag.append(
                    [ cur_grab_data_tag, cur_obj_name, cur_obj_optimized_fn, cur_cuda_idx ]
                )
                
                #### get the obj type to the optimized traj ####
                # if cur_obj_name in obj_type_to_optimized_traj:
                #     cur_obj_optimized_fn = obj_type_to_optimized_traj[cur_obj_name]
                #     # cur_grab_data_tag, cur_obj_name, cur_obj_optimized_fn, cuda_idx #
                #     cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
                #     tot_grab_data_tag.append(
                #         [cur_grab_data_tag, cur_obj_name, cur_obj_optimized_fn, cur_cuda_idx]
                #     )
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
            cur_obj_name = pure_obj_type.split("_")[-1]
            grab_obj_idx_dict_fn = f"../assets/grab_obj_name_idx_dict.npy"
            grab_obj_idx_dict = np.load(grab_obj_idx_dict_fn, allow_pickle=True).item()
            grab_obj_nm_to_idx = grab_obj_idx_dict['grab_obj_name_to_idx']
            cru_obj_idx = grab_obj_nm_to_idx[cur_obj_name] # get the current object index #
            cur_obj_sorted_nei_idxes = find_similar_objs(cru_obj_idx)
            cur_obj_sorted_nei_names = [grab_obj_idx_dict['grab_obj_idx_to_name'][idx] for idx in cur_obj_sorted_nei_idxes]
            for i_obj, nei_obj_name in enumerate(cur_obj_sorted_nei_names):
                cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
                tot_grab_data_tag.append(
                    [cur_grab_data_tag, nei_obj_name, '', cur_cuda_idx]
                )
            # for i_obj, obj_idx in enumerate(cur_obj_sorted_nei_idxes):
            #     cur_obj_name = grab_obj_idx_dict['grab_obj_idx_to_name'][obj_idx]
            #     cur_cuda_idx = len(tot_grab_data_tag) % nn_gpus
            #     tot_grab_data_tag.append(
            #         [cur_grab_data_tag, cur_obj_name, '', cur_cuda_idx]
    else:
        raise ValueError(f"Launch type {args.launch_type} not supported")

    
    
    print(f"tot_tracking_data : {tot_tracking_data}")
    

    tot_grab_data_tag = tot_grab_data_tag[args.st_idx: ]
    
    if args.debug:
        tot_grab_data_tag = tot_grab_data_tag[:1]
        
    if args.data_inst_flag is not None and len(args.data_inst_flag) > 0:
        data_inst_flag = args.data_inst_flag
        cur_cuda_idx = args.st_idx
        tot_grab_data_tag = [
            [data_inst_flag, data_inst_flag, cur_cuda_idx]
        ]
    
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

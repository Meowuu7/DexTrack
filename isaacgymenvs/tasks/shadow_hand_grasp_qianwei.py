# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import numpy as np
import os
import os.path as osp
import random
import subprocess
import sys
import time
from pyparsing import And
import torch
import trimesh
import pyrender
import polyscope as ps


sys.path.append('/home/dermark/LEAP_Hand_API/python')
sys.path.append('/home/dermark/deoxys_control')
# sys.path.append('/cephfs/hanqianwei/LEAP_Hand_API/python')
# sys.path.append('/cephfs/hanqianwei/deoxys_control/deoxys')

import argparse
import pickle
import threading
import time
from pathlib import Path
import torch
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.ik_utils import IKWrapper


from utils.torch_jit_utils import *
from utils.data_info import plane2euler
from tasks.hand_base.base_task import BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi
from scipy.spatial.transform import Rotation as R

class ShadowHandGrasp(BaseTask):
    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):

        self.cfg = cfg
        self.sim_params = sim_params
        print("self.sim_params",self.sim_params.dt)
        self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)

        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)
        ############################################################改下####################################
        num_obs = 236 + 64
        self.num_obs_dict = {
            "full_state": num_obs
        }
        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.fingertips = ["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"]
        self.hand_center = ["palm_lower"]
        self.num_fingertips = len(self.fingertips) 
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = 23
        self.cfg["device_type"] = device_type
        self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        ################################################################################################
        
        npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_elephant_inspect_1_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_apple_lift_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_hammer_use_1_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_hammer_use_1_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_hammer_use_1_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_knife_pass_1_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_train_lift_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_mug_lift_nf_300.npy"
        # npy_file = "/home/dermark/GRAB/grab_save3/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"












        dataaa = np.load(npy_file, allow_pickle=True).item()

        # 打印出所有的 key
        # print("11111111111111")
        # print(dataaa.keys())
        passive_meshes=dataaa['passive_meshes']
        self.object_transl = dataaa['object_transl'] 
        self.object_rot_quat = dataaa['object_rot_quat']
        self.robot_delta_states_weights_np=dataaa['robot_delta_states_weights_np']
        # print("self.object_transl",self.object_transl[0:100,0:3],self.object_transl[100:200,0:3],self.object_transl[200:300,0:3])
        # print("self.robot_delta_states_weights_np",self.robot_delta_states_weights_np[0:100,0:6],self.robot_delta_states_weights_np[100:200,0:6],self.robot_delta_states_weights_np[200:300,0:6])
        temp = np.copy(self.robot_delta_states_weights_np[:, 0])

        # 交换 [:, 0] 和 [:, 2] 的位置
        # self.robot_delta_states_weights_np[:, 0] = self.robot_delta_states_weights_np[:, 2]
        # self.robot_delta_states_weights_np[:, 2] = temp
        # print("Shape of passive_meshes:", passive_meshes.shape)
        # print("robot_delta_states_weights_np", self.robot_delta_states_weights_np.shape)
        # print("what?",self.object_transl[0],self.object_transl.shape)
        # print("what?",self.object_rot_quat[0],self.object_rot_quat.shape)
        # 要加的偏移量
        self.offset = np.array([0.4, 0, 0.6])


        ############################################################改上####################################
        super().__init__(cfg=self.cfg, enable_camera_sensors=False)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "hand"))
        # print("self.jacobian_tensor",self.jacobian_tensor.shape)
        
        #print("actor_root_state_tensor:",actor_root_state_tensor)
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                    self.num_shadow_hand_dofs + self.num_object_dofs)
            self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_dof_force_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        self.shadow_hand_default_dof_pos[:7] = torch.tensor([0, -0.7853, 0, -2.35539, 0, 1.57,0], dtype=torch.float, device=self.device)

        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        # print("self.shadow_hand_dof_pos",self.shadow_hand_dof_pos.shape,self.shadow_hand_dof_pos[0])
        

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        # print("self.root_state_tensor:",self.root_state_tensor.shape)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.refer_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.cartesian_error =torch.zeros((self.num_envs, 7), device=self.device)
        self.total_successes = 0
        self.total_resets = 0
        self.relative_scale_tensor = torch.full((self.num_envs, 1), 0.2, device=self.device)

        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "palm_lower", gymapi.DOMAIN_ENV)
        # print("self.hand_base_rigid_body_index",self.hand_base_rigid_body_index)
        
        # self.hand_base_dof_index=self.gym.find_actor_dof_index(self.envs[0], self.hand_indices[0], "15", gymapi.DOMAIN_ENV)
        # print("self.hand_base_dof_index",self.hand_base_dof_index)
        #self.hand_dof_pos_index =self.gym.get_actor_dof_names()



        # replay_fn="/home/dermark/GRAB/grab_save/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
        # # 加载 .npy 文件
        # data = np.load(replay_fn, allow_pickle=True).item()
        # # 打印出所有的 key
        # print(data.keys())

########################################################################################referhand#############################################################
        
        # # 加载 .npy 文件
        # npy_file = "/home/dermark/GRAB/grab_save/leap_passive_active_info_ori_grab_s2_spheremedium_pass_1_nf_300.npy"
        # dataaa = np.load(npy_file, allow_pickle=True).item()

        # # 打印出所有的 key
        # print("11111111111111")
        # print(dataaa.keys())

        # passive_meshes = dataaa['robot_delta_states_weights_np'] 
        # print("Shape of passive_meshes:", passive_meshes.shape)
        # print("what?",passive_meshes[100])
        # mesh_vertices = passive_meshes[100]  # 获取第一个帧的 mesh 顶点数据
        # mesh_vertices = torch.tensor(mesh_vertices, dtype=torch.float, device=self.device)

        # armdofposss = torch.zeros(23, device=self.device)




        # # 获取雅可比矩阵
        # jeef = self.jacobian_tensor[2, self.hand_base_rigid_body_index-1, :, :7]
        # print("jeef",jeef.shape,jeef)
        # print("mesh_vertices[0:6]",mesh_vertices[0:6])
        # dampings = 0.03
        # mesh_vertices[5]=mesh_vertices[5]
        # mesh_vertices[4]=mesh_vertices[4]
        # mesh_vertices[3]=mesh_vertices[3]
        # mesh_vertices[2]=mesh_vertices[2]  ## z axis
        # mesh_vertices[1]=mesh_vertices[1]
        # mesh_vertices[0]=mesh_vertices[0] 

        # # 求解阻尼最小二乘
        # jeef_T = jeef.T
        # lmbda = torch.eye(6, device=self.device) * (dampings ** 2)

        # # 确保所有参与计算的张量在相同设备上
        # u = (jeef_T @ torch.inverse(jeef @ jeef_T +lmbda) @ mesh_vertices[0:6])

        # # 更新 armdofposss 的值
        # armdofposss[0:7] = u  # 将 u 的形状展平成一维以适应 armdofposss
        # armdofposss[7:23] = mesh_vertices[6:22].to(self.device)  # 确保 mesh_vertices 在正确的设备上
        # print("armdofposss",armdofposss)
        # # 设置 shadow_hand_default_dof_pos
        # self.shadow_hand_default_dof_pos = armdofposss
        

        #npy_file = "/home/dermark/GRAB/grab_save/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
        




        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        self.robot_delta_states_weights_np = self.robot_delta_states_weights_np[:, joint_idxes_inversed_ordering]

        # offset = [0.4, 0.0, 0.6] # 

        # 将前 3 个维度加上偏移量
        self.object_transl[:, :3] += self.offset
        # self.robot_delta_states_weights_np[:, :3] += self.offset
        self.robot_delta_states_weights_tt=torch.zeros((300,23), device=self.device)
        self.robot_delta_states_weights_66=torch.zeros((300,22), device=self.device)
        self.robot_delta_states_weights_tt[:,7:]=torch.tensor(self.robot_delta_states_weights_np[:,6:], dtype=torch.float, device=self.device)
        self.robot_delta_states_weights_66[:,6:]=torch.tensor(self.robot_delta_states_weights_np[:,6:], dtype=torch.float, device=self.device)
        refer_init_pos=self.robot_delta_states_weights_np[0,:3]
        euler_angles = np.array(self.robot_delta_states_weights_np[0,3:6])
        # print("self.robot_delta_states_weights_np[50,3:6]",self.robot_delta_states_weights_np[50,0:6])
      
        #######################################################################leap_hand_retartget_base_to_palm###########################

        # Joint 变换,根据urdf来的，用的是fly_v3
        joint_xyz = [0, 0.038, 0.098]
        joint_rpy = [0, -1.57, 0]
        joint_translation = joint_xyz
        joint_rotation = R.from_euler('xyz', joint_rpy, degrees=False).as_matrix()

        T_joint = np.eye(4)
        T_joint[:3, :3] = joint_rotation
        T_joint[:3, 3] = joint_translation
        T_joint_inv = np.linalg.inv(T_joint) 
        new_6d_pose=[]
        new_7d_pose=[]
        for i in range(300):
            print(i)
            base_pose = self.robot_delta_states_weights_np[i,:6]
            base_translation = base_pose[:3]
            base_rotation = R.from_euler('xyz', base_pose[3:], degrees=False).as_matrix()

            T_base = np.eye(4)
            T_base[:3, :3] = base_rotation
            T_base[:3, 3] = base_translation
            # print("T_base",T_base)

            # T_joint = np.array([
            #     [0, 0, -1, 0],
            #     [0, 1, 0, 0.038],
            #     [1, 0, 0, 0.098],
            #     [0, 0, 0, 1]
            # ])
            # 计算下一个 link 的变换矩阵
            # T_link = np.dot(T_base, T_joint)
            # T_link = np.dot( T_joint_inv,T_base)
            T_link = T_base
            # 提取新的平移和旋转
            new_translation = T_link[:3, 3]
            new_rotation = R.from_matrix(T_link[:3, :3]).as_euler('xyz', degrees=False)
            new_rotation_quat = R.from_matrix(T_link[:3, :3]).as_quat()
            # 组合新的 6D pose
            
            new_6d_pose.append(np.concatenate((new_translation, new_rotation)))
            new_7d_pose.append(np.concatenate((new_translation, new_rotation_quat)))

            # print("下一个 link 的 6D pose：", new_6d_pose)
        ##########################################################leap_hand_to_palm_change_hardcode########################################################################
        
        # new_6d_pose=[]
        # new_7d_pose=[]
        # for i in range(300):
        #     print(i)
            
        #     base_pose = self.robot_delta_states_weights_np[i,:6]
        #     print("base_pose",base_pose)
        #     newx=-base_pose[2]
        #     newy=base_pose[1]+0.038
        #     newz=base_pose[0]+0.098
        #     newxr=-base_pose[5]
        #     newyr=base_pose[4]
        #     newzr=base_pose[3]
        #     # newxr=base_pose[3]
        #     # newyr=base_pose[4]
        #     # newzr=base_pose[5]
        #     new_6d_pose.append(np.array([newx, newy,newz,newxr,newyr,newzr]))
        #     new_rotation=R.from_euler('xyz', new_6d_pose[-1][3:], degrees=False).as_matrix()
        #     T_base = np.eye(4)
        #     T_base[:3, :3] = new_rotation
        #     new_rotation_quat = R.from_matrix(T_base[:3, :3]).as_quat()
        #     new_7d_pose.append(np.array([newx, newy,newz,*new_rotation_quat]))







        ##################################################################################################
        new_6d_pose = np.array(new_6d_pose)
        # print("new_6d_pose",new_6d_pose[0:100,:],new_6d_pose[200:300,:],new_6d_pose[200:300,:])
        new_6d_pose[:,:3] += self.offset
        np.set_printoptions(precision=6) 
        print("new_6d_pose[:,:3]",new_6d_pose[0:100,:3],new_6d_pose[100:200,:3],new_6d_pose[200:300,:3])
        self.refhand_6d_pose=torch.tensor(new_6d_pose, dtype=torch.float, device=self.device)
        # print("self.refhand_6d_pose",self.refhand_6d_pose)
        # self.refhand_6d_pose[:,3:6]=0
        # new_6d_pose[:,3:6]=0
        # self.refhand_6d_pose[:,5]=0
        new_7d_pose=np.array(new_7d_pose)
        new_7d_pose[:,:3] += self.offset
        init_frame=0
        # print("new_6d_pose[init_frame,3:]",self.refhand_6d_pose[init_frame,:])
        new_6d_rotation = R.from_euler('xyz', new_6d_pose[init_frame,3:], degrees=False).as_matrix()
        # print("new_6d_rotation",new_6d_rotation)
        new_6d_translation=new_6d_pose[init_frame,:3]
        self.robot_delta_states_weights_tt[:,:7]=torch.tensor(new_7d_pose, dtype=torch.float, device=self.device)
        self.robot_delta_states_weights_66[:,:6]=torch.tensor(new_6d_pose, dtype=torch.float, device=self.device)
        # self.robot_delta_states_weights_tt[:,3:6]=0
        # self.robot_delta_states_weights_tt[:,7]=1














        # init dof pos clone #
        initdofposclone=self.shadow_hand_default_dof_pos[:7]
        # print("self.shadow_hand_default_dof_pos[:7]",self.shadow_hand_default_dof_pos[:7])
        # self.shadow_hand_default_dof_pos[:7]= torch.tensor(self.ik_wrapper.inverse_kinematics(self.ik_wrapper.model, self.ik_wrapper.data, 
        #                                             refer_init_rot_mat, refer_init_pos, initdofposclone.tolist()), dtype=torch.float, device=self.device)
        # inverse kinematics for the ik wrapper model; ik wrapper model.data #
        self.shadow_hand_default_dof_pos[:7]= torch.tensor(self.ik_wrapper.inverse_kinematics(self.ik_wrapper.model, self.ik_wrapper.data, 
                                                    new_6d_rotation, new_6d_translation, initdofposclone.tolist()), dtype=torch.float, device=self.device)
        # self.shadow_hand_default_dof_pos[:7]=self.shadow_hand_default_dof_pos[:7]+delta_pos7
        # print("self.shadow_hand_default_dof_pos[:7]",self.shadow_hand_default_dof_pos[:7])


        # iidx = self.hand_body_idx_dict["palm"]
        # print("iidx",iidx)
        # endeff_init_pos = self.rigid_body_states[2, iidx-1, 0:3]
        # endeff_init_rot = self.rigid_body_states[2, iidx-1, 3:7]
        # endeff_init_rot = endeff_init_rot.unsqueeze(0)
        # endeffxyz=get_euler_xyz(endeff_init_rot)
        # endeffxyz=torch.tensor(endeffxyz, dtype=torch.float, device=self.device)
        # endeffxyz=torch.tensor(endeffxyz, dtype=torch.float, device=self.device)
        # endeff_init_pose=torch.cat((endeff_init_pos, endeffxyz), dim=-1)
        # print("endeff_init_pose",endeff_init_pose)
        # init_dpose=self.robot_delta_states_weights_np[0,0:6]-endeff_init_pose
        # print("init_dpose",init_dpose)



        
        # # 打印结果验证
        # #print(robot_delta_states_weights_np)

        self.reference_hand_data = self.robot_delta_states_weights_tt.unsqueeze(0).repeat(self.num_envs, 1, 1)
        # print("self.reference_hand_data",self.reference_hand_data.shape)
        self.reference_hand_data_6d =self.robot_delta_states_weights_66.unsqueeze(0).repeat(self.num_envs, 1, 1)
        self.object_transl_tensor = torch.tensor(self.object_transl, dtype=torch.float, device=self.device)
        self.object_rot_quat_tensor = torch.tensor(self.object_rot_quat, dtype=torch.float, device=self.device)
        self.ref_obj_pos =self.object_transl_tensor.unsqueeze(0).repeat(self.num_envs,1,1)
        self.ref_obj_rot =self.object_rot_quat_tensor.unsqueeze(0).repeat(self.num_envs,1,1)
        self.shadow_hand_default_dof_pos[7:] = self.robot_delta_states_weights_tt[0,7:]
        # print("self.robot_delta_states_weights_tt[0,6:]",self.robot_delta_states_weights_tt[0,7:])
        #self.shadow_hand_default_dof_pos[7:] = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0], dtype=torch.float, device=self.device)
        # #'''
        # #self.shadow_hand_default_dof_pos[7:] = torch.tensor([0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0], dtype=torch.float, device=self.device)
        # jeef = self.jacobian_tensor[2, self.hand_base_rigid_body_index-1, :, :7]
        # print("jeef",jeef.shape,jeef)
        # dampings = 0.05
        # jeef_T = jeef.T
        # lmbda = torch.eye(6, device=self.device) * (dampings ** 2)
        # u = (jeef_T @ torch.inverse(jeef @ jeef_T +lmbda) @ init_dpose)
        # print("u",u)
        # print("self.robot_delta_states_weights_np[0,0:6]",self.robot_delta_states_weights_np[0,0:6])
        # print("self.shadow_hand_default_dof_pos[:7]",self.shadow_hand_default_dof_pos[:7])
        
        
        #'''

####################################################################referhandup###############################################################

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)
    ############################################################改下####################################
    def _create_envs(self, num_envs, spacing, num_per_row):
        object_scale_dict = self.cfg['env']['object_code_dict']
        self.object_code_list = list(object_scale_dict.keys())
        ############################################################改上####################################
        all_scales = set()
        for object_scales in object_scale_dict.values():
            for object_scale in object_scales:
                all_scales.add(object_scale)
        self.id2scale = []
        self.scale2id = {}
        for scale_id, scale in enumerate(all_scales):
            self.id2scale.append(scale)
            self.scale2id[scale] = scale_id

        self.object_scale_id_list = []
        for object_scales in object_scale_dict.values():
            object_scale_ids = [self.scale2id[object_scale] for object_scale in object_scales]
            self.object_scale_id_list.append(object_scale_ids)
        self.repose_z = self.cfg['env']['repose_z']

        
        #if read from one data 
        assets_path = '../assets'
        dataset_root_path = osp.join(assets_path, 'datasetv4.1_posedata.npy')
        self.grasp_data_np = np.load(dataset_root_path, allow_pickle=True).item()
        keys_to_convert = ['target_qpos', 'target_hand_pos', 'target_hand_rot', 'object_euler_xy', 'object_init_z']
        self.grasp_data = {object_code: {scale: {key: None for key in keys_to_convert} for scale in self.grasp_data_np[object_code]} for object_code in self.grasp_data_np}

        for object_code in list(self.grasp_data_np.keys()):
            if object_code not in self.object_code_list:
                continue
            data_per_object = self.grasp_data[object_code]
            data_per_object_np = self.grasp_data_np[object_code]
            for scale in list(data_per_object_np.keys()):
                if scale not in object_scale_dict[object_code]:
                    continue
                for key in keys_to_convert:
                    data_per_object[scale][key] = [torch.tensor(item, dtype=torch.float, device=self.device) for item in data_per_object_np[scale][key]]
                    # In UniDexGrasp++ we don't use the grasp pose in data so we simply set this to 0
                    if key in ['target_qpos', 'target_hand_pos', 'target_hand_rot']:
                        data_per_object[scale][key] = [value * 0 for value in data_per_object[scale][key]]


        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)
        self.refertarget_dof_pos =torch.zeros((self.num_envs,16),device=self.device)
        self.delta=torch.zeros((self.num_envs, 7), device=self.device)
        ##################################################################################################################################























        #################################################################################################################################
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)
        ############################################################改下####################################
        asset_root = "../../assets"
        #shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        shadow_hand_asset_file = "urdf/franka_description/robots/leap_hand/franka_panda2.urdf"
        
        #leap_hand_asset_file = "urdf/franka_description/robots"
        ############################################################改上####################################
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file)

        # load shadow hand_asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 1
        asset_options.linear_damping = 1
        asset_options.armature = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)
        ############################################################改下####################################
        # tendon set up
        '''
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        for i in range(self.num_shadow_hand_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping

        self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        '''
        ############################################################改上##################################
        #actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        #print("actuated_dof_names",actuated_dof_names)
        #self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]
        self.actuated_dof_indices =[i for i in range(23)]
        self.actuated_dof_indices1 =[i for i in range(16)]
        # set shadow_hand dof properties
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)
        field_names = shadow_hand_dof_props.dtype.names
        # print("字段名称: ", field_names)
        # print("shadow_hand_dof_props:",shadow_hand_dof_props)
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        for i in range(7,self.num_shadow_hand_dofs):
            shadow_hand_dof_props['friction'][i] = 0.01
            shadow_hand_dof_props['armature'][i] = 0.001
            shadow_hand_dof_props['damping'][i] = 0.1
            shadow_hand_dof_props['effort'][i] = 4
            shadow_hand_dof_props['stiffness'][i] = 100
        '''
        for i in range(7):
            shadow_hand_dof_props['stiffness'][i] = [100.0, 100.0, 100.0, 100.0, 75.0, 150.0, 50.0]
            #shadow_hand_dof_props['effort'][i] = 200
            shadow_hand_dof_props['damping'][i] = [20.0, 20.0, 20.0, 20.0, 7.5, 15.0, 5.0]
            #shadow_hand_dof_props['friction'][i] = 1
        '''     
        
        ############################################################################################3
        
        
        shadow_hand_dof_props['stiffness'][0] = 100
        shadow_hand_dof_props['stiffness'][1]= 100
        shadow_hand_dof_props['stiffness'][2]= 100
        shadow_hand_dof_props['stiffness'][3]= 100
        shadow_hand_dof_props['stiffness'][4]= 75
        shadow_hand_dof_props['stiffness'][5]= 150
        shadow_hand_dof_props['stiffness'][6]= 50

        
        #shadow_hand_dof_props['effort'][i] = 200
        shadow_hand_dof_props['damping'][0] = 20
        shadow_hand_dof_props['damping'][1]= 20
        shadow_hand_dof_props['damping'][2]= 20
        shadow_hand_dof_props['damping'][3]= 20
        shadow_hand_dof_props['damping'][4]= 7.5
        shadow_hand_dof_props['damping'][5]= 15
        shadow_hand_dof_props['damping'][6]= 5

        #shadow_hand_dof_props['friction'][i] = 1

















        ############################################################################################3


            
        #print("shadow_hand_dof_props:",shadow_hand_dof_props)
        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.actuated_dof_indices1 = to_torch(self.actuated_dof_indices1, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        #self.shadow_hand_dof_default_pos[:7] = torch.tensor([0, -0.7853, 0, -2.35539, 0, 1.57,0], dtype=torch.float, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)

        # visual feature
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
        }

        object_scale_idx_pairs = []
        visual_feat_root = osp.realpath(osp.join(assets_path, 'meshdatav3_pc_feat'))
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64), device=self.device)
        
        for object_id in range(len(self.object_code_list)):
            object_code = self.object_code_list[object_id]
            self.visual_feat_data[object_id] = {}
            for scale_id in self.object_scale_id_list[object_id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    object_scale_idx_pairs.append([object_id, scale_id])
                else:
                    print(f'prior not found: {object_code}/{scale}')
                file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale2str[scale]}.npy')
                with open(file_dir, 'rb') as f:
                    feat = np.load(f)
                self.visual_feat_data[object_id][scale_id] = torch.tensor(feat, device=self.device)        

        object_asset_dict = {}
        goal_asset_dict = {}

        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            object_asset_options.density = 500
            object_asset_options.fix_base_link = False
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 50000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE  
            object_asset = None
            
            for obj_id, scale_id in object_scale_idx_pairs:
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
                    scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file,
                                                              object_asset_options)
                    if obj_id not in object_asset_dict:
                        object_asset_dict[object_id] = {}
                    object_asset_dict[object_id][scale_id] = scaled_object_asset

                    if object_asset is None:
                        object_asset = scaled_object_asset
            
            assert object_asset is not None
            object_asset_options.disable_gravity = True    
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset)
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
        ############################################################改下###########################################################

        # create table asset
        table_dims = gymapi.Vec3(0.4, 0.4, 0.60)
        #table_dims = gymapi.Vec3(1.0, 0.6, 0.01)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())

        shadow_hand_start_pose = gymapi.Transform()
        shadow_hand_start_pose = gymapi.Transform()
        #shadow_hand_start_pose.p = gymapi.Vec3(-0.2, 0.3, 0.3)  # gymapi.Vec3(0.1, 0.1, 0.65)
        shadow_hand_start_pose.p = gymapi.Vec3(0, 0, 0)  # gymapi.Vec3(0.1, 0.1, 0.65)
        #shadow_hand_start_pose.p = gymapi.Vec3(0, 0, 0)
        #shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0 ,-1.57)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0 ,0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)



        object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(0.0, -0.2, 0.67)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.p = gymapi.Vec3(self.offset[0],self.offset[1],self.offset[2])  # gymapi.Vec3(0.0, 0.0, 0.72)
        additional_pose=gymapi.Vec3(self.object_transl[0,0],self.object_transl[0,1],self.object_transl[0,2])
        object_start_pose.p=object_start_pose.p+additional_pose
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        additional_quat = gymapi.Quat(self.object_rot_quat[0,0],self.object_rot_quat[0,1],self.object_rot_quat[0,2],self.object_rot_quat[0,3])
        object_start_pose.r=object_start_pose.r*additional_quat
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0
        print("object_start_pose",object_start_pose.r)
        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = gymapi.Vec3(self.object_transl[0,0],self.object_transl[0,1],self.object_transl[0,2])
        goal_start_pose.r = gymapi.Quat(self.object_rot_quat[0,0],self.object_rot_quat[0,1],self.object_rot_quat[0,2],self.object_rot_quat[0,3])  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.55, 0, 0.5 * table_dims.z)
        #table_pose.p = gymapi.Vec3(0.0, -1.5, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1  ##
        max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1  ##
        ############################################################改上#########################################################
        self.shadow_hands = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        ############################################################改下###########################################################
        body_names = {
            
            'palm': 'palm_lower',
            'thumb': 'thumb_tip_head',
            'index': 'index_tip_head',
            'middle': 'middle_tip_head',
            'ring': 'ring_tip_head',
            
        }

        ############################################################改上#########################################################
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)
                #print("meibaocuo")

        self.object_scale_buf = {}

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            hand_color = [147/255, 215/255, 160/255]

            ############################################################改下###########################################################
            #hand_rigid_body_index = [[0,1,2,3,4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25,26,27,28]]

            ############################################################改上###########################################################
            #for n in self.agent_index[0]:
            #    for m in n:
            #        for o in hand_rigid_body_index[m]:
            #            self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
            #                                    gymapi.Vec3(*hand_color))

            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)


            id = int(i / self.num_envs * len(self.object_code_list))
            object_code = self.object_code_list[id]
            available_scale = []
            for scale_id in self.object_scale_id_list[id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    available_scale.append(scale)
                else:
                    print(f'prior not found: {object_code}/{scale}')
            scale = available_scale[i % len(available_scale)]
            scale_id = self.scale2id[scale]
            self.object_scale_buf[i] = scale
            self.object_id_buf[i] = id

            self.visual_feat_buf[i] = self.visual_feat_data[id][scale_id]

            # add object
            # object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i+2*self.num_envs, 0, 0)
            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            # hand_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, shadow_hand_actor)
            # table_shape_props[0].friction = 1
            # object_shape_props[0].friction = 1
            table_shape_props[0].friction = 0.2
            object_shape_props[0].friction = 0.2
            # hand_shape_props[0].friction =1
            # print(len(hand_shape_props))
            # print("hand_shape_props",hand_shape_props[0].shape)
            # hand_shape_props[0].friction = 4
            # hand_shape_props[1].friction = 4
            # hand_shape_props[2].friction = 4
            # hand_shape_props[3].friction = 4
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            # self.gym.set_actor_rigid_shape_properties(env_ptr, shadow_hand_actor, hand_shape_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        #print("hand_indices",self.hand_indices)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions, id=-1):
        self.dof_pos = self.shadow_hand_dof_pos
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_init_z, 
            self.reference_hand_data,
            self.delta,
            self.cartesian_error,
            #self.delta_qpos, 
            #self.delta_target_hand_pos, 
            #self.delta_target_hand_rot,
            self.id, self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,self.obj_last_pos,
            self.right_hand_pos, self.right_hand_rot ,self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]


        ############################################################改下###########################################################
        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        #self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        #self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        #self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                              
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        #self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        #self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                           
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        #self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        ############################################################改上#########################################################
        # self.goal_pose = self.ref_obj_pos
        self.goal_pos = self.ref_obj_pos[:,self.progress_buf[0], 0:3]
        self.goal_rot = self.ref_obj_rot[:,self.progress_buf[0], 0:4]
        self.obj_last_pos = self.ref_obj_pos[:,299, 0:3]
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # def world2obj_vec(vec):
        #     return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        # def obj2world_vec(vec):
        #     return quat_apply(self.object_rot, vec) + self.object_pos
        # def world2obj_quat(quat):
        #     return quat_mul(quat_conjugate(self.object_rot), quat)
        # def obj2world_quat(quat):
        #     return quat_mul(self.object_rot, quat)

        self.delta_target_hand_pos = self.reference_hand_data[:,self.progress_buf[0],0:3] - self.right_hand_pos
        # self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.reference_hand_data[:,self.progress_buf[0],3:7], quat_conjugate(self.right_hand_rot))
        # self.delta_qpos = self.target_qpos
        self.delta_qpos =  self.reference_hand_data[:,self.progress_buf[0],7:] - self.shadow_hand_dof_pos[:,7:]
        self.compute_full_state()

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state

    def compute_full_state(self, asymm_obs=False):
        # print("prigressbuf",self.progress_buf)
        self.get_unpose_quat()
        ############################################################改下###########################################################
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 52 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 24 ##

        # 0:69
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        self.obs_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)





        fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        #print("fingertip_obs_start",fingertip_obs_start)
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(4):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 69:121: ft states
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 121:161: ft sensors: do not need repose
        #self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :24]

        hand_pose_start = fingertip_obs_start + 52 
        #print("hand_pose_start",hand_pose_start)
        # 121:127: hand_pose
        self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
        euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
        self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        action_obs_start = hand_pose_start + 6
        
        #print("action_obs_start",action_obs_start)
        # 127:150: action
        aux = self.actions[:, :23]
        #aux2=scale(torch.cat((self.delta[:, :7], self.actions[:, 7:23]), dim=1),self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        
        #print(aux)
        #aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        #aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        self.obs_buf[:, action_obs_start:action_obs_start + 23] = aux
        #self.obs_buf[:, action_obs_start:action_obs_start + 23] = aux2
        obj_obs_start = action_obs_start + 23  # 144
        #print("obj_obs_start",obj_obs_start)
        # 150:166 object_pose, goal_pos
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.goal_pos - self.object_pos
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 14] = self.goal_rot - self.object_rot
        self.obs_buf[:, obj_obs_start + 14:obj_obs_start + 17] = self.obj_last_pos -self.object_pos
        # print("self.goal_pos - self.object_pos",self.goal_pos - self.object_pos)
        # print(self.goal_pos.shape,self.object_pos.shape,self.goal_pos,self.object_pos)
        # 207:236 goal
        # In UniDexGrasp++, we don't use the target goal grasp pose so we simply set
        # this observation all to zero
        hand_goal_start = obj_obs_start + 17
        self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos # self.delta_target_hand_pos
        self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot # self.delta_target_hand_rot
        self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 23] = self.delta_qpos # self.delta_qpos
        # self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = 0 # self.delta_target_hand_pos
        # self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = 0 # self.delta_target_hand_rot
        # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 23] = 0 # self.delta_qpos
        #print(" hand_goal_start", hand_goal_start)
        # 236: visual feature
        visual_feat_start = hand_goal_start + 23
        #print("visual_feat_start",visual_feat_start)
        # 236: 300: visual feature
        #self.obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf
        self.obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0
        #print("60:",self.obs_buf[2,0:69])
        return

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        rand_length = torch_rand_float(0.3, 0.5, (len(env_ids), 1), device=self.device)
        rand_angle = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0
        ############################################################改上#########################################################
    def reset(self, env_ids, goal_env_ids):
            
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        # self.reset_target_pose(env_ids)

        if self.random_prior:
            for env_id in env_ids:
                i = env_id.item()
                object_code = self.object_code_list[self.object_id_buf[i]]
                scale = self.object_scale_buf[i]

                data = self.grasp_data[object_code][scale] # data for one object one scale
               
                buf = data['object_euler_xy']
                #print("buf.shape",len(buf))
                prior_idx = random.randint(0, len(buf) - 1)
                # prior_idx = 0 ## use only one data
                #print("shapedata",data['target_qpos'][prior_idx].shape)
                self.target_qpos[i:i+1] = data['target_qpos'][prior_idx]
                self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
                self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
                self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
                self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.refer_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        #print("self.refer_targets",self.refer_targets)
        self.refer_targets[env_ids, 7] = 1
        #print("self.refer_targets",self.refer_targets)
        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        #print("self.dof_state",self.dof_state,self.dof_state.shape)
        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        #print("self.prev_targets",self.prev_targets[0],self.prev_targets.shape)
        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##

        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]

        #theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        theta = torch.zeros(len(env_ids), device=self.device)
        #reset obejct with all data:
        #new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta)
        new_object_rot = quat_from_euler_xyz(theta+1.57,theta, theta)
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # coordinate transform according to theta(object)/ prior_rot_z(hand)
        self.z_theta[env_ids] = prior_rot_z
        prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(len(env_ids), 1)[:, 0], torch.zeros_like(theta), prior_rot_z)

        self.hand_orientations[hand_indices.to(torch.long), :] = prior_rot_quat
        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0

        self.root_state_tensor[self.hand_indices[env_ids]] = self.hand_start_states[env_ids].clone()

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        #self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids],
                                              self.table_indices[env_ids], ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))

        if self.random_time:
            self.random_time = False
            #self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
            self.progress_buf[env_ids] = 0
            #print("hi if",self.progress_buf)
        else:
            self.progress_buf[env_ids] = 0
            #print("hi else")
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        

    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        self.get_pose_quat()
        #actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        #actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.clone().to(self.device)
        #print("pre_physics_step")
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            ############################################################改下###########################################################
            #self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            #self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            #self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            # # x-arm control
            #pos_err= self.actions[:, 0:3]
            #rot_err= self.actions[:, 3:6]
            '''
            fixed_value1 = -0.01 
            fixed_value2 = 0.0
            self.actions[:, 0]=fixed_value2
            self.actions[:, 1]=fixed_value2
            self.actions[:, 2]=fixed_value2
            self.actions[:, 3]=fixed_value2
            self.actions[:, 4]=fixed_value2
            self.actions[:, 5]=fixed_value2
            self.actions[:, 6:23]=fixed_value2
            print("self.actions",self.actions)
            '''
            # pos_err= 0.02*self.actions[:, 0:3]
            # rot_err= 0.02*self.actions[:, 3:6]
            # pos_err= 0.1*self.actions[:, 0:3]
            # rot_err= 0.1*self.actions[:, 3:6]
            # pos_err= 0.4*self.actions[:, 0:3]
            # rot_err= 0.4*self.actions[:, 3:6]
            pos_err= 0.04*self.actions[:, 0:3]
            rot_err= 0.04*self.actions[:, 3:6]
            # pos_err= 0.004*self.actions[:, 0:3]
            # rot_err= 0.004*self.actions[:, 3:6]

            #dpose =torch.cat([pose_err], -1).unsqueeze(-1)
            dpose = torch.cat((pos_err, rot_err), dim=1)
            #print("dpose",dpose)
            #print("dpose",dpose)
            #self.delta = control_ik(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, dpose, self.num_envs)

            '''
            lambda_val=0.03
            jacobian_T = torch.transpose(self.jacobian_tensor[:, self.hand_base_rigid_body_index-1, :, :7], dim0=1, dim1=2) # n, q, 6
            lambda_matrix = (lambda_val**2) * torch.eye(n=6, device=self.device)
            kin_matrix = torch.bmm(self.jacobian_tensor[:, self.hand_base_rigid_body_index-1, :, :7], jacobian_T) + lambda_matrix[None, ...] # n, 6, 6
            delta_joint_pos = torch.bmm(jacobian_T, torch.linalg.solve(kin_matrix, dpose.unsqueeze(-1)))
            print("self.jacobian_tensor",self.jacobian_tensor.shape)
            print("self.hand_base_rigid_body_index",self.hand_base_rigid_body_index)
            '''
            



            dpose =dpose.unsqueeze(-1)
            #print("dpose",dpose.shape)
            '''
            jeefwrong=self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7]


            jeef_copy = jeefwrong.clone()
            jeef_copy[:, 1, :] = jeefwrong[:, 0, :]
            jeef_copy[:, 0, :] = -jeefwrong[:, 1, :]
            jeef_copy[:, 4, :] = jeefwrong[:, 3, :]
            jeef_copy[:, 3, :] = -jeefwrong[:, 4, :]
            jeef=jeef_copy
            #jeef=jeefwrong
            print("jeef",jeef.shape)
            print("jeefcopy",jeef[0])
            '''
            jeef=self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7]
            

            dampings = 0.03
            # solve damped least squares
            jeef_T = torch.transpose(jeef, 1, 2)
            #print("jeef_T",jeef_T.shape)
            lmbda = torch.eye(6, device=self.device) * (dampings ** 2)
            u = (jeef_T @ torch.inverse(jeef @ jeef_T + lmbda) @ dpose).view(self.num_envs, 7)
            # u = torch.zeros(self.num_envs, 7).to(self.device)
            
            self.delta=u
            
            
            self.cur_targets[:, :7] = self.prev_targets[:, :7] + self.delta[:, :7]
            #self.cur_targets[:, :7] = self.prev_targets[:, :7]
            self.cur_targets[:, :7]=tensor_clamp(self.cur_targets[:,:7], self.shadow_hand_dof_lower_limits[:7],self.shadow_hand_dof_upper_limits[:7])

            targets=scale(self.actions[:, self.actuated_dof_indices1 + 7],self.shadow_hand_dof_lower_limits[7:],
                           self.shadow_hand_dof_upper_limits[7:])
           
            self.cur_targets[:, 7:]=targets
           

        self.prev_targets = self.cur_targets.clone()
        #print("self.cur_targets_out",self.cur_targets)
        #print("self.prev_targets",self.prev_targets)
        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        #print("all_hand_indices",all_hand_indices)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                       gymtorch.unwrap_tensor(self.prev_targets),
                                                       gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))











        ############################################keshihua#######################################################################################################








        # print("self.progress_buf[0]",self.progress_buf[0])


        # # refer_init_pos=self.robot_delta_states_weights_np[self.progress_buf[0]+1,:3]
        # # euler_angles = np.array(self.robot_delta_states_weights_np[self.progress_buf[0]+1,3:6])
        # # # 使用 'xyz' 表示旋转顺序是绕 x, y, z 轴
        # # rotation = R.from_euler('xyz', euler_angles)
        # # # 获取旋转矩阵
        # # refer_init_rot_mat = rotation.as_matrix()
        # initdofposclone=self.shadow_hand_dof_pos[0,0:7]
        
        # new_6d_pose = self.refhand_6d_pose[:,:6].cpu().numpy()
        # #print("self.refhand_6d_pose[:,:6]",new_6d_pose)
        # init_frame=0
        # new_6d_rotation = R.from_euler('xyz', new_6d_pose[self.progress_buf[0]+1,3:], degrees=False).as_matrix()
        # print("new_6d_rotation",new_6d_rotation)
        # new_6d_translation=new_6d_pose[self.progress_buf[0]+1,:3]
        
        # self.cur_targets[0, :7]= torch.tensor(self.ik_wrapper.inverse_kinematics(self.ik_wrapper.model, self.ik_wrapper.data, 
        #                                             new_6d_rotation, new_6d_translation, initdofposclone.tolist()), dtype=torch.float, device=self.device)


        # # self.cur_targets[0, [11,1,2,3,4,5,6]] = torch.tensor(self.ik_wrapper.inverse_kinematics
        # #                                                      (self.ik_wrapper.model, self.ik_wrapper.data, refer_init_rot_mat, refer_init_pos, initdofposclone.tolist()), dtype=torch.float, device=self.device)
        

        
        # #self.cur_targets[0, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]] = self.robot_delta_states_weights_tt[self.progress_buf[0],6:]
        # # self.cur_targets[0, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]] = self.robot_delta_states_weights_tt[self.progress_buf[0]+1,
        # #                                                                                                         [8,7,9,10,19,20,21,22,12,11,13,14,16,15,17,18]]
        # self.cur_targets[0, 7:] = self.robot_delta_states_weights_tt[self.progress_buf[0]+1, 7:]
        # # self.cur_targets[0, 7:] = 0
        # print("self.robot_delta_states_weights_tt[self.progress_buf[0],6:]",self.robot_delta_states_weights_tt[self.progress_buf[0],7:])
        # #self.cur_targets[0, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]] = torch.tensor([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0], dtype=torch.float, device=self.device)
        # for i in range(self.num_envs):
        #     self.cur_targets[i]=self.cur_targets[0]


        # print("self.cur_targets[:,:7]",self.cur_targets[:,:7])
        
        # self.cur_targets2 = self.cur_targets.reshape(460, 1)
        # print("self.cur_targets2",self.cur_targets2)
        # num_dofs = 460
        # # positions = self.cur_targets[:, 0]  # 假设你有目标位置
        # velocities = torch.zeros((num_dofs, 1), device=self.device)  # 创建一个速度为0的张量

        # # 合并位置和速度
        # self.cur_targets2 = torch.cat((self.cur_targets2, velocities), dim=1)

        # ###########################################################set object#######################################################################################
        # ref_objpos=torch.tensor(self.object_transl[self.progress_buf[0]+1],dtype=torch.float,device=self.device)
        # ref_objrot=torch.tensor(self.object_rot_quat[self.progress_buf[0]+1],dtype=torch.float,device=self.device)
        # cur_obj_replay_pose = torch.cat(
        #     [ ref_objpos, ref_objrot ], dim=-1
        # )


        # self.root_state_tensor[self.object_indices, :7] = cur_obj_replay_pose.clone()
        # self.root_state_tensor[self.object_indices, 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices, 7:13])
        # obj_indices = self.object_indices.to(torch.int32)
        # self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
        #                                              gymtorch.unwrap_tensor(obj_indices), len(obj_indices))

        # self.gym.set_dof_state_tensor(self.sim,
        #                                                 gymtorch.unwrap_tensor(self.cur_targets2))


        ##############################################################################################################################################
        # self.cur_targets[0, :]=0
        # self.cur_targets[0, 10]=1

        # for i in range(self.num_envs):
        #     self.cur_targets[i]=self.cur_targets[0]

        # self.gym.set_dof_position_target_tensor(self.sim,
        #                                                 gymtorch.unwrap_tensor(self.cur_targets))
        #print("self.cur_targets",self.cur_targets[0])
        
        #print("self.progress_buf",self.progress_buf)
        
        # self.prev_targets = self.cur_targets.clone()
        


        
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.cartesian_error = torch.norm(self.cur_targets[:, :7] - self.shadow_hand_dof_pos[:, :7], dim=-1)

        self.compute_observations()
        self.compute_reward(self.actions, self.id)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])

    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
        object_init_z, 
        reference_hand_data,
        delta,
        cartesian_error,
        #delta_qpos, 
        #delta_target_hand_pos, 
        #delta_target_hand_rot,
        id: int, object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,obj_last_pos,
        right_hand_pos,right_hand_rot, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    ############################################################改下###########################################################

    #print("reference_hand_data",reference_hand_data.shape)

    #print("dof_pos",dof_pos.shape)
    #print("progress_buf",progress_buf.shape)
    #print("object_handle_pos",object_handle_pos.shape)

    # print("lowest",object_pos[15,:])
    # print("target_pos",target_pos.shape,target_pos)

    reference_hand_data_pos=reference_hand_data[:,progress_buf[0],:]
    #print("reference_hand_data_pos",reference_hand_data_pos.shape)



    hand_pos_dist=torch.norm(reference_hand_data_pos[:,:3] - right_hand_pos[:,:3], p=2, dim=-1)


        
    
    reference_quat = reference_hand_data_pos[:, 3:7]  # Tensor with shape [batch_size, 4]
    right_hand_quat = right_hand_rot[:, 0:4]           # Tensor with shape [batch_size, 4]
    # print("reference_quat",reference_quat.shape)
    # print("right_hand_quat",right_hand_quat.shape)
    # print(reference_quat.dtype, right_hand_quat.dtype)
    # print(type(reference_quat), reference_quat.dtype)
    # print(type(right_hand_quat), right_hand_quat.dtype)
    # print("1705")
    quat_diff =quat_mul(right_hand_quat, quat_conjugate(reference_quat))
    # print("1707")
    #print("quat_diff",quat_diff.shape,quat_diff)
    # 计算旋转误差
    hand_rot_dist = torch.norm(quat_diff[:, 0:3], p=2, dim=-1)  # 只计算旋转部分（x, y, z）的距离





    # rotation_error=reference_hand_data_pos[:,3:7] * right_hand_pos[:,3:7].inv()
    # hand_rot_dist=np.linalg.norm(rotation_error) ** 2
    qpos_dist=torch.norm(reference_hand_data_pos[:,7:] - dof_pos[:,7:], p=2, dim=-1)
    #print("pos_dist",pos_dist)
    #print("reference_hand_data_pos[:,6:]",reference_hand_data_pos[:,6:])
    #print("dof_pos[:,7:]",dof_pos[:,7:])


    obj_last_dist = torch.norm(obj_last_pos - object_pos,p=2,dim=-1)
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    object_handle_pos2=object_handle_pos.clone()
    object_handle_pos2[:,2]=object_handle_pos[:,2]
    right_hand_dist = torch.norm(object_handle_pos2 - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 1.5, 1.5 + 0 * right_hand_dist, right_hand_dist)
    right_hand_dist = torch.where(right_hand_dist <= 0.065, 0.065, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos2 - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos2 - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos2 - right_hand_rf_pos, p=2, dim=-1) + torch.norm(object_handle_pos2 - right_hand_th_pos, p=2, dim=-1))
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= 5.0, 5.0 + 0 * right_hand_finger_dist,right_hand_finger_dist)
    #right_hand_finger_dist = torch.where(right_hand_finger_dist <= 0.8, 0.8 ,right_hand_finger_dist)
    right_hand_dist_rew = right_hand_dist
    #print("dof_pos",dof_pos.shape)

    # goal_obj_dist=torch.norm(object_handle_pos2 - right_hand_pos, p=2, dim=-1)



    right_hand_finger_dist_rew=right_hand_finger_dist
 
    action_penalty = torch.sum(actions ** 2, dim=-1)
    drop_panalty=torch.zeros_like(right_hand_finger_dist)
    delta_penalty=torch.sum(delta ** 2, dim=-1)
    cartesian_error_penalty=torch.sum(cartesian_error ** 2, dim=-1)
    # print("1750")
    # print("object_rot",object_rot.shape)
    # print("target_rot",target_rot.shape)
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # print("1752")
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    lowest = object_pos[:, 2]
    lift_z = object_init_z[:, 0] + 0.6638

    drop_panalty=torch.where(lowest<=0.2,0.5,drop_panalty)
    
    flag = (right_hand_finger_dist <= 0.32).int() + (right_hand_dist <= 0.15).int()
    #print("flag",flag[3])

    task_rew = torch.zeros_like(right_hand_finger_dist)
    task_rew = torch.where(flag == 1, 1, task_rew)
    task_rew = torch.where(flag == 2, 1.5, task_rew)
    # if flag[5]==2:
    #     print(2)
    # if flag[5]==1:
    #     print(right_hand_finger_dist[5],right_hand_dist[5])
    goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
    goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)
    #print("goal_hand_rew:",goal_hand_rew)
    #print("lowest",lowest)
    #print("lift_z",lift_z)
    hand_up = torch.zeros_like(right_hand_finger_dist)
    hand_up = torch.where(lowest >= lift_z+0.01, torch.where(flag == 2, 0.1 +1*right_hand_pos[:,2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 2 - obj_last_dist * 1, hand_up), hand_up)
    hand_track = torch.zeros_like(right_hand_finger_dist)
    hand_track = torch.where(flag == 2, 0.2 +1*right_hand_pos[:,2], hand_track)
    hand_track = torch.where(lowest >= 0.80, torch.where(flag == 2, 2 - obj_last_dist * 1, hand_track), hand_track)
    #flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

    dist_rew=torch.zeros_like(right_hand_finger_dist)
    dist_rew=torch.where(flag == 2, 1 * (10 - 4 * hand_pos_dist), dist_rew)
    #reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + 2*goal_hand_rew + 0.2*hand_up
    #reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + 2*goal_hand_rew + 0.2*hand_up + 10*bonus-drop_panalty-delta_penalty-0.004*cartesian_error_penalty
    # reward =  - 1*pos_dist-1*hand_pos_dist-1*hand_rot_dist+1
    # reward = -0.3 * right_hand_finger_dist_rew - 0.3 * right_hand_dist_rew  - 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist -goal_dist+task_rew -obj_last_dist+hand_track
    ######################## all hand object track############################
    # reward = torch.where(flag == 2,- 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist -goal_dist+task_rew -obj_last_dist+hand_track \
    # , -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist -goal_dist -obj_last_dist+hand_track)
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist - goal_dist - obj_last_dist + hand_track+bonus
    # ######################### only hand##########################################
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist 
    # reward = torch.where(flag == 2,- 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist +task_rew  \
    # , -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist )
    # ######################### hand with track####################################
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_track
    # reward = torch.where(flag == 2,- 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist +task_rew +hand_track \
    # , -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist +hand_track)
    # ###############only hand qpos05,02,00###################################################################################################
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.2*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #      - 1.0*hand_pos_dist - 0.5*hand_rot_dist
    # ###############hand with track qpos05,02,00###################################################################################################
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_track
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.2*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_track
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #      - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_track
    # ###############hand with hand_up qpos05,02,00###################################################################################################
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.5*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_up
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #     - 0.2*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_up
    # reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
    #      - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_up
    ###############hand with hand_up qpos05,02,00#########################################################################
    reward =  -0.5 * right_hand_finger_dist_rew -0.5 * right_hand_dist_rew  \
        - 0.2*qpos_dist - 1.0*hand_pos_dist - 0.5*hand_rot_dist+hand_up




    # reward = - 1.0*pos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist-goal_dist
    
    
    # reward =  (- 1*pos_dist-10.0*hand_pos_dist-5*hand_rot_dist+2 )/100
    # print("reference_hand_data_pos[:,:3]",reference_hand_data_pos[10,:3])
    # print("progress_buf",progress_buf[0])
    # print("right_hand_pos[:,:3]",right_hand_pos[10,:3])
    # print("right_hand_quat",right_hand_quat[10])
    # print("reference_quat",reference_quat[10])
    # reward =  - 1*pos_dist-1.0*hand_pos_dist-0.5*hand_rot_dist-goal_dist+2 
    # print("right_hand_finger_dist_rew",right_hand_finger_dist_rew[5])
    # print("right_hand_dist_rew",right_hand_dist_rew[5])
    # print("pos_dist",pos_dist[5])
    # print("hand_pos_dist",hand_pos_dist[5])
    # print("hand_rot_dist",hand_rot_dist[5])
    # print("task_rew",task_rew[5])
    #print("reward",reward)
    #print("delta_penalty",delta_penalty)
    #print("cartesian_error_penalty",cartesian_error_penalty)
    #print("hand_up",hand_up)
    
    
    
    resets = reset_buf
    #print("progress_buf",progress_buf)
    ############################################################改上###########################################################
    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())
    current_successes = torch.where(resets, successes, current_successes)

    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


def control_ik(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    damping = 0.01
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, 7)
    return u


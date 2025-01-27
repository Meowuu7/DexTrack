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

from pyparsing import And
import torch

from utils.torch_jit_utils import *
from utils.data_info import plane2euler
# from tasks.base.base_task import BaseTask
from isaacgymenvs.tasks.vec_task import VecTask
# from isaacgymenvs.tasks.base.vec_task import VecTask
from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, get_axis_params
from utils import torch_utils


'''
goal position是固定位置


'''
# _create_envs最后几行
# self.gym.get_asset_dof_names(fullbody_asset) 
# 09:'right_elbow' 10:'robot0:WRJ1' 33:'robot0:THJ0' 34:'left_shoulder_x' 
# body_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,]
# hand_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ,31, 32, 33]

# 10:'robot0:WRJ1' 11:'robot0:WRJ0' 去掉
# 31:'robot0:THJ0' 32:'left_shoulder_x' 
# body_idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,]
# hand_idx = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30 ,31,]


table_z = 1.0
start_x_pos = -0.5
# end_x_pos = 0.8
speeds = 0.005 * 2
grasp_front_dist = 0.25
goal_sphere_size = 0.01
class MovingArm(VecTask):
    # def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
    #              agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg

        self._pd_control = self.cfg["env"]["pdControl"]
        self.power_scale = self.cfg["env"]["powerScale"]
        
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        # self.aggregate_mode = 0
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

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
        self.fullbody_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
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

        num_obs = 236 + 64
        self.num_obs_dict = {
            "full_state": num_obs
        }
        self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        self.up_axis = 'z'
        self.fingertips = ["robot0:ffdistal", "robot0:mfdistal", "robot0:rfdistal", "robot0:lfdistal",
                           "robot0:thdistal"]
        self.hand_center = ["robot0:hand mount"]
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
        # self.cfg["env"]["numActions"] = 24 
        self.cfg["headless"] = headless

        
        key_bodies = self.cfg["env"]["keyBodies"]
        self._setup_character_props(key_bodies)
        self.cfg["env"]["numObservations"] = self.get_obs_size()
        self.cfg["env"]["numActions"] = self.get_action_size()

        self._local_root_obs = self.cfg["env"].get("localRootObs", True)
        self._root_height_obs = self.cfg["env"].get("rootHeightObs", True)

        # self.cfg["env"]["numObservations"] = self.get_obs_size()
        # self.cfg["env"]["numActions"] = self.get_action_size()
        self._enable_early_termination = self.cfg["env"]["enableEarlyTermination"]


        # self.num_environments = self.cfg["env"]["numEnvs"]
        # self.obs_buf2 = torch.zeros((self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)

        # ************************* called before super() *************************
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)
        dt = self.cfg["sim"]["dt"]
        self.dt = self.control_freq_inv * dt

        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)


        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            # self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, -1)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                    self.num_fullbody_dofs + self.num_object_dofs)
            self.dof_force_tensor = self.dof_force_tensor[:, :self.num_fullbody_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)

        # create some wrapper tensors for different slices
        self.fullbody_default_dof_pos = torch.zeros(self.num_fullbody_dofs, dtype=torch.float, device=self.device)
        self.fullbody_default_dof_pos[0] = self.fullbody_dof_lower_limits[0]
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.fullbody_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_fullbody_dofs]
        self.fullbody_dof_pos = self.fullbody_dof_state[..., 0]
        self.fullbody_dof_vel = self.fullbody_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.root_state_tensor_cache = self.root_state_tensor.clone()

        self.saved_fullbody_dof_pos = self.fullbody_dof_pos.clone() 
        self.saved_rigidbody_pos = self.rigid_body_states[:,:,:3].clone()

        # tscale = 0.05
        # for sid in self.dof_static_id:
        #     lower = self.fullbody_dof_lower_limits[sid]
        #     upper = self.fullbody_dof_upper_limits[sid]
        #     length = upper - lower
        #     delta = 0.5*tscale*length
        #     if self.fullbody_dof_lower_limits[sid] < self.saved_fullbody_dof_pos[:,sid].mean().item() - delta:
        #         self.fullbody_dof_lower_limits[sid] = self.saved_fullbody_dof_pos[:,sid].mean().item() - delta
        #     if self.fullbody_dof_upper_limits[sid] > self.saved_fullbody_dof_pos[:,sid].mean().item() + delta:
        #         self.fullbody_dof_upper_limits[sid] = self.saved_fullbody_dof_pos[:,sid].mean().item() + delta
        # self.fullbody_dof_lower_limits = to_torch(self.fullbody_dof_lower_limits, device=self.device)
        # self.fullbody_dof_upper_limits = to_torch(self.fullbody_dof_upper_limits, device=self.device)


        num_actors = self.get_num_actors_per_env()
        self.root_positions = self.root_state_tensor[:, 0:3]
        self.root_orientations = self.root_state_tensor[:, 3:7]
        self.root_linvels = self.root_state_tensor[:, 7:10]
        self.root_angvels = self.root_state_tensor[:, 10:13]


        # bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        # rigid_body_state_reshaped = self._rigid_body_state.view(self.num_nvs, bodies_per_env, 13)
        self._rigid_body_pos = self.rigid_body_states[..., :self.num_bodies, 0:3]
        self._rigid_body_rot = self.rigid_body_states[..., :self.num_bodies, 3:7]
        self._rigid_body_vel = self.rigid_body_states[..., :self.num_bodies, 7:10]
        self._rigid_body_ang_vel = self.rigid_body_states[..., :self.num_bodies, 10:13]
        
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        # bodies_per_env = self.rigid_body_states.shape[0] // self.num_envs
        # bodies_per_env应该是41， fullbody38个，再加上1个object, 1个goal，1个table
        self._contact_forces = contact_force_tensor.view(self.num_envs, -1, 3)[..., :self.num_bodies, :]

        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.long, device=self.device)
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device)
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.total_successes = 0
        self.total_resets = 0

        self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)

        # self._terminate_buf = torch.ones(self.num_envs, device=self.device, dtype=torch.long)
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._build_termination_heights()

        contact_bodies = self.cfg["env"]["contactBodies"]
        self._key_body_ids = self._build_key_body_ids_tensor(key_bodies)
        self._contact_body_ids = self._build_contact_body_ids_tensor(contact_bodies)

        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)

    def get_num_actors_per_env(self):
        num_actors = self.root_state_tensor.shape[0] // self.num_envs
        return num_actors
    


    
    def _build_termination_heights(self):
        head_term_height = 0.3
        shield_term_height = 0.32

        termination_height = self.cfg["env"]["terminationHeight"]
        self._termination_heights = np.array([termination_height] * self.num_bodies)

        head_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.fullbody_actor_handles[0], "head")
        self._termination_heights[head_id] = max(head_term_height, self._termination_heights[head_id])

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        if asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            left_arm_id = self.gym.find_actor_rigid_body_handle(self.envs[0], self.fullbody_actor_handles[0], "left_lower_arm")
            self._termination_heights[left_arm_id] = max(shield_term_height, self._termination_heights[left_arm_id])
        
        self._termination_heights = to_torch(self._termination_heights, device=self.device)
        return
    

    def _setup_character_props(self, key_bodies):
        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        # # 24个body, 23个joint, 22个dof
        # self._dof_body_ids = [2, 3, 4, 5, 
        #                       6, 7, 8, 9, 
        #                       10, 11, 12, 13, 
        #                       14, 15, 16, 17, 18,
        #                       19, 20, 21, 22, 23
        #                     ]
        # self._dof_offsets = [0, 
        #                      1, 2, 3, 4, 
        #                      5, 6, 7, 8,
        #                      9, 10, 11, 12,
        #                      13, 14, 15, 16, 17, 
        #                      18, 19, 20, 21, 22
        #                     ]4096*41'robot0:rfmiddle', 'robot0:rfdistal', 'robot0:lfmetacarpal', 
        #     'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal', 
        #     'robot0:thbase', 'robot0:thproximal', 'robot0:thhub', 'robot0:thmiddle', 
        #     'robot0:thdistal', 'left_upper_arm', 'left_lower_arm', 'left_hand', 
        #     'right_thigh', 'right_shin', 'right_foot', 'left_thigh', 
        #     'left_shin', 'left_foot'
        # ]


        # 上面是旧版的，wrj1不应该加在hand mount上，应该放在palm里面

        self._dof_body_ids = [1, 2, 4, 
                              5, 6, 7, 8, 
                              9, 10, 11, 12, 
                              13, 14, 15, 16,
                              17, 18, 19, 20, 21,
                              22, 23, 24, 25, 26,
                              ]

        dof_size_list =      [1, 3, 3,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1,
                              1, 1, 1, 1, 1,
                              1, 1, 1, 1, 1, 
                              ]
        self._dof_offsets = [0]
        for dof_size in dof_size_list:
            self._dof_offsets.append(self._dof_offsets[-1]+dof_size)
        # [0, 
        #      1, 4,
        #      3, 6, 9, 10,
        #      12, 13, 14, 15,
        #      16, 17, 18, 19, 20,
        #      21, 22, 23, 24, 25,
        #      26, 27, 28, 29, 30,
        #      31, 32, 33, 34, 
        #      37, 38, 41, 42, 45,
        #      48, 49, 52,
        #     ]


         
        # 132 = 22*6
        self._dof_obs_size = (len(self._dof_offsets) - 1)*6

        # humanoid: 28, hand: 22+2，这里的2是WRJ1和WRJ0
        # self.gym.get_asset_dof_names(fullbody_asset)
        self._num_actions = 31
        # TODO 这里暂定1000，多余的地方可以全部设置成0，或者一会根据具体结果来更新
        self._num_obs = 1000 
        

        return

    def _build_key_body_ids_tensor(self, key_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.fullbody_actor_handles[0]
        body_ids = []

        for body_name in key_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids

    def _build_contact_body_ids_tensor(self, contact_body_names):
        env_ptr = self.envs[0]
        actor_handle = self.fullbody_actor_handles[0]
        body_ids = []

        for body_name in contact_body_names:
            body_id = self.gym.find_actor_rigid_body_handle(env_ptr, actor_handle, body_name)
            assert(body_id != -1)
            body_ids.append(body_id)

        body_ids = to_torch(body_ids, device=self.device, dtype=torch.long)
        return body_ids
    
    def get_obs_size(self):
        return self._num_obs

    def get_action_size(self):
        return self._num_actions
     
    def create_sim(self):
        self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dt = self.sim_params.dt
        # self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        object_scale_dict = self.cfg['env']['object_code_dict']
        self.object_code_list = list(object_scale_dict.keys())
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

        self.grasp_data = {}
        assets_path = '../assets/hand'
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        for object_code in self.object_code_list:
            data_per_object = {}
            dataset_path = dataset_root_path + '/' + object_code
            data_num_list = os.listdir(dataset_path)
            for num in data_num_list:
                data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                qpos = data_dict['qpos'].item() #goal
                scale_inverse = data_dict['scale'].item()  # the inverse of the object's scale
                scale = round(1 / scale_inverse, 2)
                assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                target_qpos = torch.tensor(list(qpos.values())[:22], dtype=torch.float, device=self.device)
                target_hand_rot_xyz = torch.tensor(list(qpos.values())[22:25], dtype=torch.float, device=self.device)  # 3
                target_hand_rot = quat_from_euler_xyz(target_hand_rot_xyz[0], target_hand_rot_xyz[1], target_hand_rot_xyz[2])  # 4
                target_hand_pos = torch.tensor(list(qpos.values())[25:28], dtype=torch.float, device=self.device)
                plane = data_dict['plane']  # plane parameters (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                translation, euler = plane2euler(plane, axes='sxyz')  # object
                object_euler_xy = torch.tensor([euler[0], euler[1]], dtype=torch.float, device=self.device)
                object_init_z = torch.tensor([translation[2]], dtype=torch.float, device=self.device)

                if object_init_z > 0.05:
                    continue

                if scale in data_per_object:
                    data_per_object[scale]['target_qpos'].append(target_qpos)
                    data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                    data_per_object[scale]['target_hand_rot'].append(target_hand_rot)
                    data_per_object[scale]['object_euler_xy'].append(object_euler_xy)
                    data_per_object[scale]['object_init_z'].append(object_init_z)
                else:
                    data_per_object[scale] = {}
                    data_per_object[scale]['target_qpos'] = [target_qpos]
                    data_per_object[scale]['target_hand_pos'] = [target_hand_pos]
                    data_per_object[scale]['target_hand_rot'] = [target_hand_rot]
                    data_per_object[scale]['object_euler_xy'] = [object_euler_xy]
                    data_per_object[scale]['object_init_z'] = [object_init_z]
            self.grasp_data[object_code] = data_per_object

        self.goal_cond = self.cfg["env"]["goal_cond"]
        self.random_prior = self.cfg['env']['random_prior']
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = "../assets/hand"
        # fullbody_asset_file = "mjcf/open_ai_assets/hand/fullbody.xml"
        # fullbody_asset_file = "mjcf/open_ai_assets/hand/fullhand.xml"
        fullbody_asset_file = "mjcf/open_ai_assets/hand/fullbody_full.xml"
        fullbody_asset_file = "mjcf/open_ai_assets/hand/fullbody_full_plus.xml"
        fullbody_asset_file = "mjcf/open_ai_assets/hand/right_arm.xml"
        # fullbody_asset_file = "mjcf/open_ai_assets/hand/fullbody_nowrj.xml"
        # fullbody_asset_file = "fullbody.xml"
        # fullbody_asset_file = "mjcf/fullbody_24.xml"
        table_texture_files = "../assets/hand/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        # TODO 用yaml来配置
        # if "asset" in self.cfg["env"]:
        #     asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
        #     fullbody_asset_file = self.cfg["env"]["asset"].get("assetFileName", fullbody_asset_file)

        # load shadow hand_ asset
        # asset_options = gymapi.AssetOptions()
        # asset_options.flip_visual_attachments = False
        # asset_options.fix_base_link = False
        # asset_options.collapse_fixed_joints = True
        # # asset_options.disable_gravity = False 
        # asset_options.thickness = 0.001
        # asset_options.angular_damping = 100
        # asset_options.linear_damping = 100

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        asset_options.fix_base_link = True 
        # asset_options.disable_gravity = True 


        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        fullbody_asset = self.gym.load_asset(self.sim, asset_root, fullbody_asset_file, asset_options)
        self.fullbody_asset = fullbody_asset
        self.num_fullbody_bodies = self.gym.get_asset_rigid_body_count(fullbody_asset)
        self.num_fullbody_shapes = self.gym.get_asset_rigid_shape_count(fullbody_asset)
        self.num_fullbody_dofs = self.gym.get_asset_dof_count(fullbody_asset)
        self.num_fullbody_actuators = self.gym.get_asset_actuator_count(fullbody_asset)
        self.num_fullbody_tendons = self.gym.get_asset_tendon_count(fullbody_asset)

        dof_names = self.gym.get_asset_dof_names(fullbody_asset)
        # self.dof_name2num_dict = {dof_names[i]:i for i in range(len(dof_names))}
        # dof_static_name_list = ['left_hip_x', 'left_hip_y', 'left_hip_z', 
        #                         'left_knee', 
        #                         'left_ankle_x', 'left_ankle_y', 'left_ankle_z',

        #                         'right_hip_x', 'right_hip_y', 'right_hip_z', 
        #                         'right_knee', 
        #                         'right_ankle_x', 'right_ankle_y', 'right_ankle_z',

        #                         'abdomen_x', 'abdomen_y', 'abdomen_z',
        #                         'neck_x', 'neck_y', 'neck_z',

        #                         'left_shoulder_x', 'left_shoulder_y', 'left_shoulder_z',
        #                         'left_elbow',

        #                         ]
        # self.dof_static_id = [self.dof_name2num_dict[i] for i in dof_static_name_list]
        # self.dof_change_id = set(range(len(dof_static_name_list))) - set(self.dof_static_id)

        rigidbody_name = self.gym.get_asset_rigid_body_names(fullbody_asset)
        # self.rigidbody_name2num_dict = {rigidbody_name[i]:i for i in range(len(rigidbody_name))}
        # rigidbody_static_name_list = [
        #     'pelvis', 'torso', 'head', 'left_upper_arm', 'left_lower_arm', 'left_hand', 
        #     'right_thigh', 'right_shin', 'right_foot',
        #     'left_thigh', 'left_shin', 'left_foot'
        # ]
        # self.rigidbody_static_id = [self.rigidbody_name2num_dict[i] for i in rigidbody_static_name_list]

        print("self.num_fullbody_bodies: ", self.num_fullbody_bodies)
        print("self.num_fullbody_shapes: ", self.num_fullbody_shapes)
        print("self.num_fullbody_dofs: ", self.num_fullbody_dofs)
        print("self.num_fullbody_actuators: ", self.num_fullbody_actuators)
        print("self.num_fullbody_tendons: ", self.num_fullbody_tendons)

        # tendon set up
        limit_stiffness = 30
        t_damping = 0.1
        relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        tendon_props = self.gym.get_asset_tendon_properties(fullbody_asset)

        for i in range(self.num_fullbody_tendons):
            for rt in relevant_tendons:
                if self.gym.get_asset_tendon_name(fullbody_asset, i) == rt:
                    tendon_props[i].limit_stiffness = limit_stiffness
                    tendon_props[i].damping = t_damping

        self.gym.set_asset_tendon_properties(fullbody_asset, tendon_props)

        actuated_dof_names = [self.gym.get_asset_actuator_joint_name(fullbody_asset, i) for i in range(self.num_fullbody_actuators)]
        self.actuated_dof_indices = [self.gym.find_asset_dof_index(fullbody_asset, name) for name in actuated_dof_names]
        # actuated_dof_names_hand = actuated_dof_names[:18]
        # self.actuated_dof_indices_hand = [self.gym.find_asset_dof_index(fullbody_asset, name) for name in actuated_dof_names_hand] 
        # self.actuated_dof_indices_hand = to_torch(self.actuated_dof_indices_hand, dtype=torch.long, device=self.device)

        # set fullbody dof properties
        fullbody_dof_props = self.gym.get_asset_dof_properties(fullbody_asset)

        self.fullbody_dof_lower_limits = []
        self.fullbody_dof_upper_limits = []
        self.fullbody_dof_default_pos = []
        self.fullbody_dof_default_vel = []
        self.fullbody_actor_handles = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_fullbody_dofs):
            self.fullbody_dof_lower_limits.append(fullbody_dof_props['lower'][i])
            self.fullbody_dof_upper_limits.append(fullbody_dof_props['upper'][i])
            self.fullbody_dof_default_pos.append(0.0)
            self.fullbody_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.fullbody_dof_lower_limits = to_torch(self.fullbody_dof_lower_limits, device=self.device)
        self.fullbody_dof_upper_limits = to_torch(self.fullbody_dof_upper_limits, device=self.device)
        self.fullbody_dof_default_pos = to_torch(self.fullbody_dof_default_pos, device=self.device)
        self.fullbody_dof_default_vel = to_torch(self.fullbody_dof_default_vel, device=self.device)

        self.dof_limits_lower = []
        self.dof_limits_upper = []
        # self.gym.get_asset_dof_names(fullbody_asset) 
        # 09:'right_elbow' 10:'robot0:WRJ1' 33:'robot0:THJ0' 34:'left_shoulder_x' 
        dof_names_list = self.gym.get_asset_dof_names(fullbody_asset)
        body_idx = []
        hand_idx = []
        for ii in range(len(dof_names_list)):
            if dof_names_list[ii].startswith('robot0:'):
                hand_idx.append(ii)
            else:
                body_idx.append(ii)
        self.body_idx = body_idx
        self.hand_idx =hand_idx
        
        self.dof_limits_lower = [self.fullbody_dof_lower_limits[i] for i in body_idx]
        self.dof_limits_upper = [self.fullbody_dof_upper_limits[i] for i in body_idx]

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.hand_dof_limits_lower = [self.fullbody_dof_lower_limits[i] for i in hand_idx]
        self.hand_dof_limits_upper = [self.fullbody_dof_upper_limits[i] for i in hand_idx]
        self.hand_dof_limits_lower = to_torch(self.hand_dof_limits_lower, device=self.device)
        self.hand_dof_limits_upper = to_torch(self.hand_dof_limits_upper, device=self.device)

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
            # object_asset_options.density = 500000
            object_asset_options.fix_base_link = False
            # object_asset_options.fix_base_link = True 
            # object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            # object_asset_options.angular_damping = 100000
            # object_asset_options.linear_damping = 100000
            
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
            goal_asset = self.gym.create_sphere(self.sim, goal_sphere_size, object_asset_options)
            # >>>>>>>> mx_debug <<<<<<<<
            # goal_asset = self.gym.create_sphere(self.sim, 0.03, object_asset_options)
            # >>>>>>>> mx_debug <<<<<<<<
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


        # create table asset
        # table_z = 1.0
        table_len = 0.4
        object2table_disp= 0.0
        table_dims = gymapi.Vec3(table_len, table_len, table_z)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        # asset_options.flip_visual_attachments = True
        # asset_options.collapse_fixed_joints = True
        # asset_options.disable_gravity = True
        # asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

        fullbody_start_pose = gymapi.Transform()
        # fullbody_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.8)  # gymapi.Vec3(0.1, 0.1, 0.65)

        # rx = random.uniform(10.0, 12.3)
        rx = 0.3
        # ry = random.uniform(2.0, 2.3)
        ry = -0.3
        # rz = random.uniform(1.1, 1.3)
        # rx, ry = 0.2, 0.2
        rz = 1.0 
        # fullbody_start_pose.p = gymapi.Vec3(0.3, 0.3, 1.2)  # gymapi.Vec3(0.1, 0.1, 0.65)
        # fullbody_start_pose.p = gymapi.Vec3(rx, ry, rz)  # gymapi.Vec3(0.1, 0.1, 0.65)
        # fullbody_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)

        # fullbody_start_pose.p = gymapi.Vec3(rx, ry, rz)  # gymapi.Vec3(0.1, 0.1, 0.65)
        # fullbody_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1

        char_h = 0.88
        char_h = 1.28
        fullbody_start_pose.p = gymapi.Vec3(*get_axis_params(char_h, self.up_axis_idx))
        fullbody_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)


        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(rx-object2table_disp, ry, table_z + 0.002)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # object_start_pose.p = gymapi.Vec3(rx-object2table_disp, ry, table_z + 0.02)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)

        goal_start_pose.p.z -= 0.0

        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(rx, ry, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        max_agg_bodies = self.num_fullbody_bodies * 1 + 2 * self.num_object_bodies + 1  ##
        max_agg_shapes = self.num_fullbody_shapes * 1 + 2 * self.num_object_shapes + 1  ##

        self.fullbodys = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.fullbody_start_states = []
        self.fullbody_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(fullbody_asset, name) for name in self.fingertips]
        t_hand_handle = ["robot0:hand mount", "robot0:palm"]
        self.hand_handles = [self.gym.find_asset_rigid_body_index(fullbody_asset, name) for name in t_hand_handle] 
        body_names = {
            # 'wrist': 'robot0:wrist',
            'palm': 'robot0:palm',
            'thumb': 'robot0:thdistal',
            'index': 'robot0:ffdistal',
            'middle': 'robot0:mfdistal',
            'ring': 'robot0:rfdistal',
            'little': 'robot0:lfdistal'
        }
        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(fullbody_asset, body_name)

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(fullbody_asset, ft_handle, sensor_pose)
            
            for h_handle in self.hand_handles:
                self.gym.create_asset_force_sensor(fullbody_asset, h_handle, sensor_pose) 

        self.object_scale_buf = {}

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            # fullbody_actor = self.gym.create_actor(env_ptr, fullbody_asset, fullbody_start_pose, "fullbody", i, -1, 0)
            fullbody_actor = self.gym.create_actor(env_ptr, fullbody_asset, fullbody_start_pose, "fullbody", i, -1, 0)
            self.fullbody_start_states.append(
                [fullbody_start_pose.p.x, fullbody_start_pose.p.y, fullbody_start_pose.p.z,
                 fullbody_start_pose.r.x, fullbody_start_pose.r.y, fullbody_start_pose.r.z,
                 fullbody_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, fullbody_actor, fullbody_dof_props)
            fullbody_idx = self.gym.get_actor_index(env_ptr, fullbody_actor, gymapi.DOMAIN_SIM)
            self.fullbody_indices.append(fullbody_idx)
            self.fullbody_actor_handles.append(fullbody_actor)

            # randomize colors and textures for rigid body
            num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, fullbody_actor)
            hand_color = [147/255, 215/255, 160/255]
            hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            for n in self.agent_index[0]:
                for m in n:
                    for o in hand_rigid_body_index[m]:
                        self.gym.set_rigid_body_color(env_ptr, fullbody_actor, o, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(*hand_color))

            # create fingertip force-torque sensors
            if self.obs_type == "full_state" or self.asymmetric_obs:
                self.gym.enable_actor_dof_force_sensors(env_ptr, fullbody_actor)
            
            fullbody_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, fullbody_actor)
            
            # for sid in self.rigidbody_static_id:
            #     fullbody_body_props[sid].mass = fullbody_body_props[sid].mass*10
            # # table_body_props[0].invMass = 1/table_body_props[0].mass 
            # self.gym.set_actor_rigid_body_properties(env_ptr, fullbody_actor, fullbody_body_props) 


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

            
            # # >>>>>>>> mx_debug <<<<<<<
            # # 给handle添加一些移动，发现在测试中效果就变差了
            # goal_start_pose.p.x += 0.001
            # goal_start_pose.p.y += 0.001
            # # goal_start_pose.p.z += 0.0001 
            # # >>>>>>>> mx_debug <<<<<<<

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
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            # table_body_props = self.gym.get_actor_rigid_body_properties(env_ptr, table_handle)
            # table_body_props[0].mass = 1e4
            # # table_body_props[0].invMass = 1/table_body_props[0].mass 
            # self.gym.set_actor_rigid_body_properties(env_ptr, table_handle, table_body_props) 

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.fullbodys.append(fullbody_actor)


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        # self.goal_init_state = self.goal_states.clone()
        self.fullbody_start_states = to_torch(self.fullbody_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.fullbody_indices = to_torch(self.fullbody_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)


        

        self._build_pd_action_offset_scale()


    def compute_reward(self, actions):
        self.dof_pos = self.fullbody_dof_pos
        # self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
        #     self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
        #     self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
        #     self.progress_buf, self.successes, self.consecutive_successes,
        #     self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
        #     self.goal_pos, self.goal_rot,
        #     self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
        #     self.right_hand_lf_pos, self.right_hand_th_pos,
        #     self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
        #     self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
        #     self.max_consecutive_successes, self.av_factor,self.goal_cond
        # )

        
        self.static_body_delta = 0
        self.rew_buf[:], _, self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_lf_pos, self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.hand_sensor, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond,
            self.static_body_delta,
            self._rigid_body_vel,
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes
        self.extras["terminate"] = self.successes

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
    
    def compute_reward_v1(self, actions):
        self.rew_buf[:] = compute_reach_reward(self._rigid_body_pos)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]

        self.goal_pose = self.root_state_tensor[self.goal_object_indices, 0:7]
        self.gaol_pos = self.root_state_tensor[self.goal_object_indices, 0:3]
        self.goal_rot = self.root_state_tensor[self.goal_object_indices, 3:7]
        
        self.object_handle_pos = self.object_pos  ##+ quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]



        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                              
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['little']
        self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # self.cur_static_body = self.rigid_body_states[:, self.rigidbody_static_id, 0:3]
        # self.init_static_body = self.saved_rigidbody_pos[:, self.rigidbody_static_id, 0:3]
        # # self.save 
        # self.static_body_delta = torch.abs(self.cur_static_body-self.init_static_body)

        # 0-29是5个手指tips的3force3torque 
        # 30-35 hand mount
        # 36-42 palm
        # self.hand_sensor = self.vec_sensor_tensor[:, [32, 38]]
        self.hand_sensor = self.vec_sensor_tensor[:, [i for i in range(30, 42)]]

        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)

        self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        # self.delta_qpos = self.fullbody_dof_pos - self.target_qpos
        self.delta_qpos = self.target_qpos

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

        body_pos = self._rigid_body_pos
        body_rot = self._rigid_body_rot
        body_vel = self._rigid_body_vel
        body_ang_vel = self._rigid_body_ang_vel

        # obs.shape : envs_num, 613 
        body_obs = compute_body_observations_max(body_pos, body_rot, body_vel, body_ang_vel, self._local_root_obs,
                                                self._root_height_obs)

        # self.obs_buf[:, 0:body_obs.shape[1]] = body_obs 
        self.get_unpose_quat()

        # hand_obs_start = body_obs.shape[1]
        hand_obs_start = 0

        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0:66
        self.obs_buf[:, hand_obs_start:hand_obs_start+self.num_fullbody_dofs] = unscale(self.fullbody_dof_pos,
                                                               self.fullbody_dof_lower_limits,
                                                               self.fullbody_dof_upper_limits)
        self.obs_buf[:,hand_obs_start+self.num_fullbody_dofs:hand_obs_start+2*self.num_fullbody_dofs] = self.vel_obs_scale * self.fullbody_dof_vel
        self.obs_buf[:,hand_obs_start+2*self.num_fullbody_dofs:hand_obs_start+3*self.num_fullbody_dofs] = self.force_torque_obs_scale * self.dof_force_tensor
        
        fingertip_obs_start = hand_obs_start + 3*self.num_fullbody_dofs
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        for i in range(5):
            aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
        # 66:131: ft states
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 131:161: ft sensors: do not need repose
        self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :30]

        # hand_pose_start = fingertip_obs_start + 95
        # # 161:167: hand_pose
        # self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
        # euler_xyz = get_euler_xyz(self.unpose_quat(self.root_orientations[self.fullbody_indices, :]))
        # self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        # self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        # self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)

        action_obs_start = fingertip_obs_start + 95
        # 167:191: action
        aux = self.actions
        # TODO: 检查是否需要对前6个进行unpose
        # aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        # aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        self.obs_buf[:, action_obs_start:action_obs_start + self.actions.shape[1]] = aux

        obj_obs_start = action_obs_start + self.actions.shape[1]  # 144
        # 191:207 object_pose, goal_pos
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

        # set 0, non-gpal-condition
        hand_goal_start = obj_obs_start + 16 

        # 0, 1, 2 goal, 3 table 
        self.obs_buf[:,hand_goal_start:hand_goal_start+3] = body_pos[:,-2,:]
        hand_goal_start += 3
        self.obs_buf[:, hand_goal_start:] = 0
        
        degug = 987
        #  # 207:236 goal
        # hand_goal_start = obj_obs_start + 16
        # self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
        # self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
        # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos

        # # 236: visual feature
        # visual_feat_start = hand_goal_start + 29

        # # 236: 300: visual feature
        # self.obs_buf[:, visual_feat_start:visual_feat_start + 64] = 0.1 * self.visual_feat_buf

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


    def reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            # env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long) 
        
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_fullbody_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        if self.random_prior:
            for env_id in env_ids:
                i = env_id.item()
                object_code = self.object_code_list[self.object_id_buf[i]]
                scale = self.object_scale_buf[i]

                data = self.grasp_data[object_code][scale] # data for one object one scale
                buf = data['object_euler_xy']
                prior_idx = random.randint(0, len(buf) - 1)
                # prior_idx = 0 ## use only one data

                self.target_qpos[i:i+1] = data['target_qpos'][prior_idx]
                self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
                self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
                self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
                self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]

        # reset shadow hand
        delta_max = self.fullbody_dof_upper_limits - self.fullbody_dof_default_pos
        delta_min = self.fullbody_dof_lower_limits - self.fullbody_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_fullbody_dofs]

        pos = self.fullbody_default_dof_pos  # + self.reset_dof_pos_noise * rand_delta
        self.fullbody_dof_pos[env_ids, :] = pos

        self.fullbody_dof_vel[env_ids, :] = self.fullbody_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_fullbody_dofs:5 + self.num_fullbody_dofs * 2]

        self.prev_targets[env_ids, :self.num_fullbody_dofs] = pos
        self.cur_targets[env_ids, :self.num_fullbody_dofs] = pos

        all_fullbody_indices = torch.unique(torch.cat([self.fullbody_indices[env_ids].to(torch.int32)]).to(torch.int32))
        all_object_indices = torch.unique(torch.cat([self.object_indices[env_ids].to(torch.int32)]).to(torch.int32)) 
        all_goal_object_indices = torch.unique(torch.cat([self.goal_object_indices[env_ids].to(torch.int32)]).to(torch.int32)) 
        all_table_indices = torch.unique(torch.cat([self.table_indices[env_ids].to(torch.int32)]).to(torch.int32)) 


        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_fullbody_indices), len(all_fullbody_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_fullbody_indices), len(all_fullbody_indices))

        all_indices = torch.unique(torch.cat([all_fullbody_indices, all_object_indices, all_table_indices, ]).to(torch.int32))  ##

        self.root_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.root_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]

        theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]

        #reset obejct with all data:
        new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta)
        # new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[:,0], self.object_init_euler_xy[:,1], theta)
        prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # coordinate transform according to theta(object)/ prior_rot_z(hand)
        self.z_theta[env_ids] = prior_rot_z
        # prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(self.num_envs, 1)[:, 0], torch.zeros_like(theta), prior_rot_z)
        prior_rot_quat = quat_from_euler_xyz(torch.zeros_like(theta), torch.zeros_like(theta), torch.zeros_like(theta))
        self.root_orientations[all_fullbody_indices.to(torch.long), :] = prior_rot_quat
        self.root_linvels[all_fullbody_indices.to(torch.long), :] = 0
        self.root_angvels[all_fullbody_indices.to(torch.long), :] = 0
        # self.root_orientations[self.object_indices.to(torch.long), :] = prior_rot_quat 

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot  # reset object rotation
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        all_indices = torch.unique(torch.cat([all_fullbody_indices,
                                              all_object_indices,
                                              all_goal_object_indices,
                                              all_table_indices, ]).to(torch.int32))

        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))


        self.root_state_tensor_cache[all_fullbody_indices, 0] = start_x_pos
        # arm_root_state = self.root_state_tensor.clone()
        # arm_root_state[all_fullbody_indices,0] += 0.02
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor_cache),
                                                         gymtorch.unwrap_tensor(all_fullbody_indices),
                                                         len(all_fullbody_indices))
        
        
        
        
         
        
        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
    
    def reset(self):
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self.reset_idx(env_ids)
        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)


    def pre_physics_step(self, actions):
        hand_idx = self.hand_idx
        body_idx = self.body_idx
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.get_pose_quat()
        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        self.actions = actions.clone().to(self.device)

        # self.actions_body = self.actions[:, body_idx]
        pd_tar = self._action_to_pd_targets(self.actions)[:, body_idx]
        
        self.actions_hand = self.actions[:, hand_idx]



        # pd_tar = self._action_to_pd_targets(self.actions_body)

        # body除了右手臂，都用set_dof_state，保持静止

        self.cur_targets[:, body_idx] = pd_tar 
        # self.cur_targets[:, body_idx[1:]] = pd_tar[:,1:]
        all_fullbody_indices = torch.unique(torch.cat([self.fullbody_indices]).to(torch.int32))

        # self.dof_limits_lower
        self.cur_targets[:, hand_idx] = scale(self.actions_hand,self.hand_dof_limits_lower,self.hand_dof_limits_upper)
        self.cur_targets[:, hand_idx] = self.act_moving_average * self.cur_targets[:, hand_idx] + (1.0 - self.act_moving_average) * self.prev_targets[:, hand_idx]
        self.cur_targets[:, hand_idx] = tensor_clamp(self.cur_targets[:, hand_idx],self.hand_dof_limits_lower,self.hand_dof_limits_upper)
        
        
        
        self.prev_targets = self.cur_targets 
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_fullbody_indices), len(all_fullbody_indices)) 

        self.root_state_tensor_cache[all_fullbody_indices,0] += speeds
        self.gym.refresh_actor_root_state_tensor(self.sim)
        arm_root_state = self.root_state_tensor.clone()
        arm_root_state[all_fullbody_indices,0] = self.root_state_tensor_cache[all_fullbody_indices,0]

        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(arm_root_state),
                                                         gymtorch.unwrap_tensor(all_fullbody_indices),
                                                         len(all_fullbody_indices))

        
        
        
        # arm_root_state[all_goal_indices,0:1] = arm_root_state[all_fullbody_indices,0:1] + grasp_front_dist
        # arm_root_state[all_goal_indices,1:2] = arm_root_state[all_fullbody_indices,1:2]
         
        # # self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_init_state[env_ids, 0:3]  # + self.goal_displacement_tensor
        # # self.root_state_tensor[self.goal_object_indices[env_ids], 0:1] = self.root_state_tensor[self.fullbody_indices[env_ids], 0:1] + 0.01 
        # # self.root_state_tensor[self.goal_object_indices[env_ids], 1:2] = self.root_state_tensor[self.fullbody_indices[env_ids], 1:2] 
        # # self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_init_state[env_ids, 3:7]
        
        # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(arm_root_state),
        #                                                  gymtorch.unwrap_tensor(all_goal_indices),
        #                                                  len(all_goal_indices))
        # # self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(arm_root_state),
        # #                                                  gymtorch.unwrap_tensor(all_fullbody_indices),
        # #                                                  len(all_fullbody_indices))
        
        


    def _build_pd_action_offset_scale(self):
        tmp_dof_offsets = self._dof_offsets
        # tmp_dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        num_joints = len(tmp_dof_offsets) - 1
        
        lim_low = self.fullbody_dof_lower_limits.cpu().numpy()
        lim_high = self.fullbody_dof_upper_limits.cpu().numpy()

        for j in range(num_joints):
            dof_offset = tmp_dof_offsets[j]
            dof_size = tmp_dof_offsets[j + 1] - tmp_dof_offsets[j]

            if dof_size == 3:
                curr_low = lim_low[dof_offset:(dof_offset + dof_size)]
                curr_high = lim_high[dof_offset:(dof_offset + dof_size)]
                curr_low = np.max(np.abs(curr_low))
                curr_high = np.max(np.abs(curr_high))
                curr_scale = max([curr_low, curr_high])
                curr_scale = 1.2 * curr_scale
                curr_scale = min([curr_scale, np.pi])

                lim_low[dof_offset:(dof_offset + dof_size)] = -curr_scale
                lim_high[dof_offset:(dof_offset + dof_size)] = curr_scale

            elif dof_size == 1:
                curr_low = lim_low[dof_offset]
                curr_high = lim_high[dof_offset]
                curr_mid = 0.5 * (curr_high + curr_low)
                
                # extend the action range to be a bit beyond the joint limits so that the motors
                # don't lose their strength as they approach the joint limits
                curr_scale = 0.7 * (curr_high - curr_low)
                curr_low = curr_mid - curr_scale
                curr_high = curr_mid + curr_scale

                lim_low[dof_offset] = curr_low
                lim_high[dof_offset] =  curr_high

        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)
        self._pd_action_offset = to_torch(self._pd_action_offset, device=self.device)
        self._pd_action_scale = to_torch(self._pd_action_scale, device=self.device)

        return
     
    def _action_to_pd_targets(self, action):
        pd_tar = self._pd_action_offset + self._pd_action_scale * action
        return pd_tar
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        self._compute_reset()
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        arm_root_state = self.root_state_tensor.clone()
        all_goal_indices = torch.unique(torch.cat([self.goal_object_indices]).to(torch.int32))
        all_fullbody_indices = torch.unique(torch.cat([self.fullbody_indices]).to(torch.int32)) 
        arm_root_state[all_goal_indices,0:1] = arm_root_state[all_fullbody_indices,0:1] + grasp_front_dist
        arm_root_state[all_goal_indices,1:2] = arm_root_state[all_fullbody_indices,1:2]
        
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(arm_root_state),
                                                         gymtorch.unwrap_tensor(all_goal_indices),
                                                         len(all_goal_indices))

        # env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        # if len(env_ids) > 0:
        #     self.reset_idx(env_ids)
        

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])


    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = compute_humanoid_reset(self.reset_buf, self.progress_buf,
                                                                           self._contact_forces, self._contact_body_ids,
                                                                           self._rigid_body_pos,
                                                                           self.max_episode_length,
                                                                           self._enable_early_termination,
                                                                           self._termination_heights,
                                                                           self.object_pos,
                                                                           )

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


# @torch.jit.script
def compute_hand_reward_v0(
        object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
                object_handle_pos - right_hand_lf_pos, p=2, dim=-1) + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist,right_hand_finger_dist)

    right_hand_dist_rew = right_hand_dist
    right_hand_finger_dist_rew = right_hand_finger_dist

    action_penalty = torch.sum(actions ** 2, dim=-1)

    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)
    delta_value = 0.6 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.1 * delta_qpos_value 
    target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
        
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    lowest = object_pos[:, 2]
    lift_z = object_init_z[:, 0] + 0.6 +0.003

    if goal_cond:
        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)
        
        flag2 = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5*delta_value

    else:
        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= 0.630, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus
    
    
    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    # reward = reward*0
    return reward, resets, goal_resets, progress_buf, successes, cons_successes

# @torch.jit.script
def compute_hand_reward(
        object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_lf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        force_sensor, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool,
        static_body_delta,
        rigid_body_vel,
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1) + torch.norm(
                object_handle_pos - right_hand_lf_pos, p=2, dim=-1) + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist,right_hand_finger_dist)

    right_hand_dist_rew = right_hand_dist
    right_hand_finger_dist_rew = right_hand_finger_dist

    # action_penalty = torch.sum(actions ** 2, dim=-1)
    
    delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    delta_qpos_value = torch.norm(delta_qpos, p=1, dim=-1)
    delta_value = 0.6 * delta_hand_pos_value + 0.04 * delta_hand_rot_value + 0.1 * delta_qpos_value 
    target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
        
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    lowest = object_pos[:, 2]
    lift_z = object_init_z[:, 0] + 0.6 +0.003

    if goal_cond:
        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 5, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)
        
        flag2 = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 0.80, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5*delta_value

    else:
        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

        hand_up = torch.zeros_like(right_hand_finger_dist)
        # hand_up = torch.where(lowest >= 1.030, torch.where(flag == 2, 0.1 + 0.01 * force_sensor[:, 2], hand_up), hand_up)
        # rigid_body_vel[:,[5,6,10,14], 2].max()
        hand_up = torch.where(lowest >= 1.030, torch.where(flag == 2, 0.1 + 0.025 * rigid_body_vel[:,5, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= 1.20, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        flag = (right_hand_finger_dist <= 0.6).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus
    
    # static_penalty = -static_body_delta.sum(dim=(1,2))*100 
    # static_penalty = -static_body_delta.sum(dim=(1,2)) 
    # reward += static_penalty
    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    # reward = reward*0
    return reward, resets, goal_resets, progress_buf, successes, cons_successes



# @torch.jit.script
def compute_reach_reward(rigid_body_pos,):
    # tar_pos
    # tar_rx = 1.0, tar_ry = 0.0

    foot_body_ids = torch.tensor([34, 37])
    reach_body_pos = rigid_body_pos[:,foot_body_ids,:]
    pos_err_scale = 4.0

    tar_pos = torch.zeros_like(reach_body_pos)
    tar_pos[:,:,0] = 1


    pos_diff = tar_pos[:,:,:2] - reach_body_pos[:,:,:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    reward = pos_reward
    reward = reward.mean(dim=1)
    return reward

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))



@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot


@torch.jit.script
def compute_body_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs):
    # type: (Tensor, Tensor, Tensor, Tensor, bool, bool) -> Tensor
    root_pos = body_pos[:, 0, :]
    root_rot = body_rot[:, 0, :]

    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
    flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    
    root_pos_expand = root_pos.unsqueeze(-2)
    local_body_pos = body_pos - root_pos_expand
    flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
    flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
    local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
    local_body_pos = local_body_pos[..., 3:] # remove root pos

    flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
    flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
    flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
    local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
    
    if not local_root_obs:
        root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
        local_body_rot_obs[..., 0:6] = root_rot_obs

    flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
    flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
    local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
    
    flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
    flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
    local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])
    
    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel), dim=-1)
    return obs


# @torch.jit.script
def compute_humanoid_reset(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, object_pos):
    terminated = torch.zeros_like(reset_buf)

    terminated = torch.where(object_pos[...,-1]<table_z, torch.ones_like(reset_buf), terminated)
    # arm is to far
    terminated = torch.where(rigid_body_pos[...,0,0]>(-start_x_pos+0.3), torch.ones_like(reset_buf), terminated)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated

# @torch.jit.script
def compute_humanoid_reset_vv(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, object_pos):
    terminated = torch.zeros_like(reset_buf)
    # self.num_fullbody_bodies 38
    
    if enable_early_termination:
        masked_contact_buf = contact_buf.clone()
        masked_contact_buf[:, contact_body_ids, :] = 0
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        fall_contact = torch.any(fall_contact, dim=-1)

        # body_height = rigid_body_pos[..., 2]
        body_height = rigid_body_pos[:, :38, 2]
        # fall_height = body_height < 0.05
        fall_height = body_height < 0.15
        fall_height[:, contact_body_ids] = False
        fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        has_fallen *= (progress_buf > 1)
        terminated = torch.where(has_fallen, torch.ones_like(reset_buf), terminated)
    
    terminated = torch.where(object_pos[...,-1]<table_z-0.3, torch.ones_like(reset_buf), terminated)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)

    return reset, terminated


# @torch.jit.script
def compute_humanoid_reset_v1(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, object_pos):
    terminated = torch.zeros_like(reset_buf)

    non_contact_body_ids = torch.tensor([0, 1, 2, 3, 4,
                                         29, 30, 31, 32, 33, 35, 36
                                         ])
    foot_body_ids = torch.tensor([34, 37])
                            
    if enable_early_termination:
        # masked_contact_buf = contact_buf[:,:-3,:].clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0

        masked_contact_buf = contact_buf[:,non_contact_body_ids,:].clone() 
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[...,non_contact_body_ids, 2]
        # fall_height = body_height < termination_heights[0]
        # fall_height = body_height < 0.1 
        fall_height = body_height < 0.2 
        # fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = torch.any(has_fallen, dim=-1)

        
        # float_height = rigid_body_pos[...,foot_body_ids, 2]>
        float_height = rigid_body_pos[...,foot_body_ids, 2]>1.3
        # float_height = rigid_body_pos[...,foot_body_ids, 2]>0.086
        float_heights = torch.any(float_height, dim=-1)
        terminate_ = torch.logical_or(has_fallen, float_heights)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        terminate_ *= (progress_buf > 1)
        terminated = torch.where(terminate_, torch.ones_like(reset_buf), terminated)
        # terminated = torch.zeros_like(terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    
    # reset = torch.where(table_z >= object_pos[:,2], torch.ones_like(reset), reset)
    
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), torch.zeros_like(reset_buf)) 
    return reset, terminated


def compute_humanoid_reset_v2(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, object_pos):
    terminated = torch.zeros_like(reset_buf)

    non_contact_body_ids = torch.tensor([0, 1, 2, 3, 4,
                                         29, 30, 31, 32, 33, 35, 36
                                         ])
    foot_body_ids = torch.tensor([34, 37])
                            
    if enable_early_termination:
        # masked_contact_buf = contact_buf[:,:-3,:].clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0

        masked_contact_buf = contact_buf[:,non_contact_body_ids,:].clone() 
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[...,non_contact_body_ids, 2]
        # fall_height = body_height < termination_heights[0]
        # fall_height = body_height < 0.1 
        fall_height = body_height < 0.8
        # fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = torch.any(has_fallen, dim=-1)

        
        # float_height = rigid_body_pos[...,foot_body_ids, 2]>
        float_height = rigid_body_pos[...,foot_body_ids, 2]>0.3
        # float_height = rigid_body_pos[...,foot_body_ids, 2]>0.086
        float_heights = torch.any(float_height, dim=-1)
        terminate_ = torch.logical_or(has_fallen, float_heights)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        terminate_ *= (progress_buf > 1)
        terminated = torch.where(terminate_, torch.ones_like(reset_buf), terminated)
        # terminated = torch.zeros_like(terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    
    # reset = torch.where(table_z >= object_pos[:,2], torch.ones_like(reset), reset)
    
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), torch.zeros_like(reset_buf)) 
    return reset, terminated


# @torch.jit.script
def compute_humanoid_reset_v(reset_buf, progress_buf, contact_buf, contact_body_ids, rigid_body_pos,
                           max_episode_length, enable_early_termination, termination_heights, object_pos):
    terminated = torch.zeros_like(reset_buf)

    non_contact_body_ids = torch.tensor([0, 1, 2, 3, 4,
                                         29, 30, 31, 32, 33, 35, 36
                                         ])
    foot_body_ids = torch.tensor([34, 37])
                            
    if enable_early_termination:
        # masked_contact_buf = contact_buf[:,:-3,:].clone()
        # masked_contact_buf[:, contact_body_ids, :] = 0

        masked_contact_buf = contact_buf[:,non_contact_body_ids,:].clone() 
        fall_contact = torch.any(torch.abs(masked_contact_buf) > 0.1, dim=-1)
        # fall_contact = torch.any(fall_contact, dim=-1)

        body_height = rigid_body_pos[...,non_contact_body_ids, 2]
        # fall_height = body_height < termination_heights[0]
        # fall_height = body_height < 0.1 
        fall_height = body_height < 0.6 
        # fall_height = torch.any(fall_height, dim=-1)

        has_fallen = torch.logical_and(fall_contact, fall_height)
        has_fallen = torch.any(has_fallen, dim=-1)

        
        # float_height = rigid_body_pos[...,foot_body_ids, 2]>
        float_height = rigid_body_pos[...,foot_body_ids, 2]>1.3
        # float_height = rigid_body_pos[...,foot_body_ids, 2]>0.086
        float_heights = torch.any(float_height, dim=-1)
        terminate_ = torch.logical_or(has_fallen, float_heights)

        # first timestep can sometimes still have nonzero contact forces
        # so only check after first couple of steps
        terminate_ *= (progress_buf > 1)
        terminated = torch.where(terminate_, torch.ones_like(reset_buf), terminated)
        # terminated = torch.zeros_like(terminated)

    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), terminated)
    
    reset = torch.where(table_z >= object_pos[:,2], torch.ones_like(reset), reset)
    
    # reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), torch.zeros_like(reset_buf)) 
    return reset, terminated
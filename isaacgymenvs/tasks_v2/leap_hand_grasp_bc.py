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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi

from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp  
from isaacgymenvs.tasks.base.vec_task import VecTask


class LeapHandGraspbc(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):

        self.cfg = cfg

        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]

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
        self.torque_control = self.cfg["env"].get("torqueControl", True)

        self.force_scale = self.cfg["env"].get("forceScale", 0.0)
        self.force_prob_range = self.cfg["env"].get("forceProbRange", [0.001, 0.1])
        self.force_decay = self.cfg["env"].get("forceDecay", 0.99)
        self.force_decay_interval = self.cfg["env"].get("forceDecayInterval", 0.08)

        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.1)

        self.object_type = self.cfg["env"]["objectType"]
        assert self.object_type in ["block", "egg", "pen","bottle","ycb/bottle","duck","elephant","banana", "cube"]

        self.ignore_z = (self.object_type == "pen")

        self.asset_files_dict = {
            "block": "urdf/objects/cube_multicolor.urdf",
            "egg": "mjcf/open_ai_assets/hand/egg.xml",
            "pen": "mjcf/open_ai_assets/hand/pen.xml",
            "ycb/bottle": "urdf/ycb/006_mustard_bottle/006_mustard_bottle.urdf",
            "bottle": "urdf/bottle.urdf",
            "duck": "urdf/duck.urdf",
            "elephant": "urdf/elephant.urdf",
            "banana": "urdf/banana.urdf",
            "cube": "urdf/cube.urdf",
        }

        if "asset" in self.cfg["env"]:
            self.asset_files_dict["block"] = self.cfg["env"]["asset"].get("assetFileNameBlock", self.asset_files_dict["block"])
            self.asset_files_dict["egg"] = self.cfg["env"]["asset"].get("assetFileNameEgg", self.asset_files_dict["egg"])
            self.asset_files_dict["pen"] = self.cfg["env"]["asset"].get("assetFileNamePen", self.asset_files_dict["pen"])

        # can be "full_no_vel", "full", "full_state"
        self.obs_type = self.cfg["env"]["observationType"]

        if not (self.obs_type in ["full_no_vel", "full", "full_state","debug_realworldbc", "debug_realworldbc_w_velocity"]):
            raise Exception(
                "Unknown type of observations!\nobservationType should be one of: [openai, full_no_vel, full, full_state]")
        
        self.debug_realbc = self.cfg["env"]["debug_realworldbc"]
        self.wo_velocity = self.cfg["env"]["wo_velocity"]

        if self.debug_realbc:
            if self.wo_velocity:
                self.obs_type = "debug_realworldbc"
            else:
                self.obs_type = "debug_realworldbc_w_velocity"

        print("Obs type:", self.obs_type)

        self.num_obs_dict = {
            "full_no_vel": 50,
            "full": 72,
            "full_state": 162,
            "debug_realworldbc": 68, 
            "debug_realworldbc_w_velocity": 161
        }

        self.is_distillation = self.cfg["env"]["is_distillation"]

        if self.is_distillation:
            self.num_student_obs = 33 #55

        self.up_axis = 'z'

        self.fingertips = ["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"]
        self.hand_center = ["palm_center"]

        self.num_fingertips = len(self.fingertips)

        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]

        num_states = 0
        if self.asymmetric_obs:
            num_states = 88

        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        self.cfg["env"]["numStates"] = num_states
        self.cfg["env"]["numActions"] = 22

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        self.dt = self.sim_params.dt
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        if self.reset_time > 0.0:
            self.max_episode_length = int(round(self.reset_time/(control_freq_inv * self.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(1.2, 0.19, 2.0)
            cam_target = gymapi.Vec3(-0.1, 0.19, 0.6)
            # cam_pos = gymapi.Vec3(0.25, 0.8, 1.3)
            # cam_target = gymapi.Vec3(-0.05, 0.0, 0.6)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self.jacobian_tensor = gymtorch.wrap_tensor(self.gym.acquire_jacobian_tensor(self.sim, "hand"))
        # dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        # self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arm_hand_dofs)

        # if self.obs_type == "full_state" or self.asymmetric_obs:
        # #     sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
        # #     self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

        #     dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        #     self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_arm_hand_dofs)
        #     self.dof_force_tensor = self.dof_force_tensor[:, :self.num_arm_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        # create some wrapper tensors for different slices
        self.arm_hand_default_dof_pos = torch.zeros(self.num_arm_hand_dofs, dtype=torch.float, device=self.device)
        self.arm_hand_default_dof_pos[:7] = torch.tensor([0, -0.7853, 0, -2.35539, 0, 1.57,0.785], device=self.device)
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.arm_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_arm_hand_dofs]
        self.arm_hand_dof_pos = self.arm_hand_dof_state[..., 0]
        self.arm_hand_dof_vel = self.arm_hand_dof_state[..., 1]

        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]

        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)

        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0

        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print("Num dofs: ", self.num_dofs)

        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.last_actions = torch.zeros((self.num_envs, 22), dtype=torch.float, device=self.device)

        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs, -1)
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
        self.cartesian_error =torch.zeros((self.num_envs, 3), device=self.device)

        self.total_successes = 0
        self.total_resets = 0

        self.relative_scale_tensor = torch.full((self.num_envs, 1), 0.2, device=self.device)

        self.hand_base_rigid_body_index = self.gym.find_actor_rigid_body_index(self.envs[0], self.hand_indices[0], "panda_link7", gymapi.DOMAIN_ENV)
        print("self.hand_base_rigid_body_index",self.hand_base_rigid_body_index)
        
        self.hand_base_dof_index=self.gym.find_actor_dof_index(self.envs[0], self.hand_indices[0], "panda_joint7", gymapi.DOMAIN_ENV)
        print("self.hand_base_dof_index",self.hand_base_dof_index)


        self.p_gain_val_hand = 100.0
        self.d_gain_val_hand = 4.0
        self.p_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.p_gain_val_hand
        self.d_gain = torch.ones((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float) * self.d_gain_val_hand
        # joint_kp: [100., 100., 100., 100., 75., 150., 50.]
        # joint_kd: [20., 20., 20., 20., 7.5, 15.0, 5.0]
        self.p_gain_val_arm_0123 = 100
        self.d_gain_val_arm_0123 = 20

        self.p_gain_val_arm_4 = 130
        self.d_gain_val_arm_4 = 7.5

        self.p_gain_val_arm_5 = 130
        self.d_gain_val_arm_5 = 15

        self.p_gain_val_arm_6 = 100
        self.d_gain_val_arm_6 = 5

        self.torques = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)
        self.dof_vel_finite_diff = torch.zeros((self.num_envs, self.num_dofs), device=self.device, dtype=torch.float)

    def create_sim(self):
        self.dt = self.sim_params.dt
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2

        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        arm_hand_asset_file = "urdf/franka_description/robots/franka_panda_leaphand.urdf"

        if "asset" in self.cfg["env"]:
            asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root)
            allegro_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", arm_hand_asset_file)

        object_asset_file = self.asset_files_dict[self.object_type]

        # load arm hand asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.angular_damping = 0.01

        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS

        arm_hand_asset = self.gym.load_asset(self.sim, asset_root, arm_hand_asset_file, asset_options)

        self.num_arm_hand_bodies = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        self.num_arm_hand_shapes = self.gym.get_asset_rigid_shape_count(arm_hand_asset)
        self.num_arm_hand_dofs = self.gym.get_asset_dof_count(arm_hand_asset)
        print("Num dofs: ", self.num_arm_hand_dofs)
        self.num_shadow_hand_actuators = self.num_arm_hand_dofs

        self.actuated_dof_indices = [i for i in range(7, self.num_arm_hand_dofs)]

        # set shadow_hand dof properties
        robot_dof_props = self.gym.get_asset_dof_properties(arm_hand_asset)

        self.arm_hand_dof_lower_limits = []
        self.arm_hand_dof_upper_limits = []
        self.arm_hand_dof_default_pos = []
        self.arm_hand_dof_default_vel = []

        robot_lower_qpos = []
        robot_upper_qpos = []

        self.sensors = []
        sensor_pose = gymapi.Transform()

        # franka_stiffness = [50., 50., 50., 50., 75., 150., 50.]
        # franka_damping = [20., 20., 20., 20., 7.5, 15.0, 5.0]

        # franka_stiffness = [100., 100., 50., 50., 75., 150., 50.]
        # franka_damping = [20., 20., 20., 20., 7.5, 15.0, 5.0]

        franka_stiffness = [100., 100., 100., 100., 75., 150., 50.]
        franka_damping =  [20., 20., 20., 20., 7.5, 15.0, 5.0]

        # franka_stiffness = [10., 10., 10., 10., 7.5, 15., 5.]
        # franka_damping = [20., 20., 20., 20., 7.5, 15.0, 5.0]

        # This part is very important (damping)
        for i in range(23):
            robot_lower_qpos.append(robot_dof_props['lower'][i])
            robot_upper_qpos.append(robot_dof_props['upper'][i])
            self.arm_hand_dof_default_pos.append(0.0)
            self.arm_hand_dof_default_vel.append(0.0)

            if not self.torque_control:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS
                if i < 7:
                    robot_dof_props['damping'][i] = franka_damping[i]
                    robot_dof_props['stiffness'][i] = franka_stiffness[i]
                else:
                    robot_dof_props['friction'][i] = 0.01
                    robot_dof_props['armature'][i] = 0.001
                    robot_dof_props['damping'][i] = 4
                    robot_dof_props['effort'][i] = 0.95
                    robot_dof_props['stiffness'][i] = 100
            else:
                robot_dof_props['driveMode'][i] = gymapi.DOF_MODE_EFFORT
                robot_dof_props['stiffness'][i] = 0.
                robot_dof_props['damping'][i] = 0.

                robot_dof_props['effort'][i] = 20.0

                robot_dof_props['friction'][i] = 0.1
                robot_dof_props['armature'][i] = 0.1

                if i < 7:
                    robot_dof_props['velocity'][i] = 1.0
                else:
                    robot_dof_props['velocity'][i] = 3.14

                if i < 7:
                    robot_dof_props['damping'][i] = 5 #100.0
                else:
                    robot_dof_props['damping'][i] = 0.0 

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.arm_hand_dof_lower_limits = to_torch(robot_lower_qpos, device=self.device)
        self.arm_hand_dof_upper_limits = to_torch(robot_upper_qpos, device=self.device)
        self.arm_hand_dof_default_pos = to_torch(self.arm_hand_dof_default_pos, device=self.device)
        self.arm_hand_dof_default_vel = to_torch(self.arm_hand_dof_default_vel, device=self.device)

        # load manipulated object and goal assets
        object_asset_options = gymapi.AssetOptions()
        object_asset_options.disable_gravity = False
        object_asset_options.fix_base_link = False
        object_asset = self.gym.load_asset(self.sim, asset_root, object_asset_file, object_asset_options)
        self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
        self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)

        object_asset_options.disable_gravity = True
        goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)

        # create table asset
        table_dims = gymapi.Vec3(1.5, 1.5, 0.6)
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001

        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)
        table_pose = gymapi.Transform()
        table_pose.p = gymapi.Vec3(0.0, 0.19, 0.5 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        arm_hand_start_pose = gymapi.Transform()
        arm_hand_start_pose.p = gymapi.Vec3(-0.9, 0.0, 0.6-0.135)
        arm_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)

        object_start_pose = gymapi.Transform()
        object_start_pose.p = gymapi.Vec3(-0.2, 0.0, 0.6281)
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 1.57)

        if self.object_type == "pen":
            object_start_pose.p.z = arm_hand_start_pose.p.z + 0.02

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.20)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement

        goal_start_pose.p.z -= 0.0

        # compute aggregate size
        max_agg_bodies = self.num_arm_hand_bodies + 2 * self.num_object_bodies + 1
        max_agg_shapes = self.num_arm_hand_shapes + 2 * self.num_object_shapes + 1

        self.arm_hands = []
        self.envs = []

        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []

        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(arm_hand_asset, name) for name in self.fingertips]

        body_names = {
            
            'palm': 'palm_center',
            'thumb': 'thumb_tip_head',
            'index': 'index_tip_head',
            'middle': 'middle_tip_head',
            'ring': 'ring_tip_head',
            
        }

        self.hand_body_idx_dict = {}
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(arm_hand_asset, body_name)

        arm_hand_rb_count = self.gym.get_asset_rigid_body_count(arm_hand_asset)
        object_rb_count = self.gym.get_asset_rigid_body_count(object_asset)
        self.object_rb_handles = list(range(arm_hand_rb_count, arm_hand_rb_count + object_rb_count))

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
            arm_hand_actor = self.gym.create_actor(env_ptr, arm_hand_asset, arm_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append([arm_hand_start_pose.p.x, arm_hand_start_pose.p.y, arm_hand_start_pose.p.z,
                                           arm_hand_start_pose.r.x, arm_hand_start_pose.r.y, arm_hand_start_pose.r.z, arm_hand_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.gym.set_actor_dof_properties(env_ptr, arm_hand_actor, robot_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, arm_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            # self.gym.enable_actor_dof_force_sensors(env_ptr, arm_hand_actor)

            # add object
            object_handle = self.gym.create_actor(env_ptr, object_asset, object_start_pose, "object", i, 0, 0)
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z, object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0]) 

            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)

            # add goal object
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
            # self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
            table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
            self.gym.set_rigid_body_color(
                env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(1, 0.9, 0.8)
            )
            self.table_indices.append(table_idx)

            # set friction
            table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            table_shape_props[0].friction = 1
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))           

            if self.object_type != "block":
                self.gym.set_rigid_body_color(
                    env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))
                self.gym.set_rigid_body_color(
                    env_ptr, goal_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(0.6, 0.72, 0.98))

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.arm_hands.append(arm_hand_actor)

        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)

        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        #print("hand_indices",self.hand_indices)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

    def compute_reward(self, actions):
        self.dof_pos = self.arm_hand_dof_pos.clone()
        # work and torque penalty
        torque_penalty = (self.torques ** 2).sum(-1)
        work_penalty = ((self.torques * self.dof_vel_finite_diff).sum(-1)) ** 2

        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
            self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_rot,
            self.goal_pos, self.goal_rot,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor, torque_penalty, work_penalty, self.cartesian_error
        )

        self.extras['successes'] = self.successes
        self.extras['consecutive_successes'] = self.consecutive_successes.mean()

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum()
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(direct_average_successes/(self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(self.total_successes/self.total_resets))

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            # self.gym.refresh_dof_force_tensor(self.sim)

        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]

        self.hand_base_pose = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:7]
        self.hand_base_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3]
        self.hand_base_rot = self.rigid_body_states[:, self.hand_base_rigid_body_index, 3:7]
        self.hand_base_linvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 7:10]
        self.hand_base_angvel = self.rigid_body_states[:, self.hand_base_rigid_body_index, 10:13]

        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]

        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_ff_linvel = self.rigid_body_states[:, idx, 7:10]
        self.right_hand_ff_angvel = self.rigid_body_states[:, idx, 10:13]

        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_mf_linvel = self.rigid_body_states[:, idx, 7:10]
        self.right_hand_mf_angvel = self.rigid_body_states[:, idx, 10:13]

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_rf_linvel = self.rigid_body_states[:, idx, 7:10]
        self.right_hand_rf_angvel = self.rigid_body_states[:, idx, 10:13]


        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        self.right_hand_th_linvel = self.rigid_body_states[:, idx, 7:10]
        self.right_hand_th_angvel = self.rigid_body_states[:, idx, 10:13]      

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]

        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        self.arm_hand_finger_dist = (torch.norm(self.object_pos - self.right_hand_ff_pos, p=2, dim=-1) + torch.norm(self.object_pos - self.right_hand_mf_pos, p=2, dim=-1)
                                + torch.norm(self.object_pos - self.right_hand_rf_pos, p=2, dim=-1) + torch.norm(self.object_pos - self.right_hand_th_pos, p=2, dim=-1))
        
        relative_pos = self.hand_base_pos - self.hand_base_last_pos
        self.cartesian_error = torch.norm(( relative_pos - self.dpose[:, 0:3] ), dim = 1)

        if self.obs_type == "full_no_vel":
            self.compute_full_observations(True)
        elif self.obs_type == "full":
            self.compute_full_observations()
        elif self.obs_type == "full_state":
             self.compute_full_state()
        elif self.obs_type == "debug_realworldbc":
            self.compute_full_state()
        elif self.obs_type == "debug_realworldbc_w_velocity":
            self.compute_full_state()
        else:
            print("Unknown observations type!")

        if self.asymmetric_obs:
            self.compute_full_state(True)

    def compute_full_observations(self, no_vel=False):
        if no_vel:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)

            self.obs_buf[:, 16:23] = self.object_pose
            self.obs_buf[:, 23:30] = self.goal_pose
            self.obs_buf[:, 30:34] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 34:50] = self.actions
        else:
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                                   self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel

            # 2*16 = 32 -16
            self.obs_buf[:, 32:39] = self.object_pose
            self.obs_buf[:, 39:42] = self.object_linvel
            self.obs_buf[:, 42:45] = self.vel_obs_scale * self.object_angvel

            self.obs_buf[:, 45:52] = self.goal_pose
            self.obs_buf[:, 52:56] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            self.obs_buf[:, 56:72] = self.actions

    def compute_full_state(self, asymm_obs=False):
        if asymm_obs:
            self.states_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                      self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
            self.states_buf[:, self.num_shadow_hand_dofs:2*self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            self.states_buf[:, 2*self.num_shadow_hand_dofs:3*self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            obj_obs_start = 3*self.num_shadow_hand_dofs  # 48
            self.states_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            self.states_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            self.states_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel

            goal_obs_start = obj_obs_start + 13  # 61
            self.states_buf[:, goal_obs_start:goal_obs_start + 7] = self.goal_pose
            self.states_buf[:, goal_obs_start + 7:goal_obs_start + 11] = quat_mul(self.object_rot, quat_conjugate(self.goal_rot))

            fingertip_obs_start = goal_obs_start + 11  # 72

            # obs_end = 96 + 65 + 30 = 191
            # obs_total = obs_end + num_actions = 72 + 16 = 88
            obs_end = fingertip_obs_start
            self.states_buf[:, obs_end:obs_end + self.num_actions] = self.actions
        else:
            if not self.debug_realbc:
                num_ft_states = 13 * int(self.num_fingertips)  # 52 ##
                # 0:69
                self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                        self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                self.obs_buf[:, self.num_arm_hand_dofs:2*self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel

                self.obs_buf[:, 46:49] = self.right_hand_ff_pos
                self.obs_buf[:, 49:52] = self.right_hand_rf_pos
                self.obs_buf[:, 52:55] = self.right_hand_mf_pos
                self.obs_buf[:, 55:58] = self.right_hand_th_pos

                self.obs_buf[:, 58:80] = self.actions
                self.obs_buf[:, 80:87] = self.hand_base_pose

                self.obs_buf[:, 87:90] = self.hand_base_linvel
                self.obs_buf[:, 90:93] = self.hand_base_angvel

                self.obs_buf[:, 93:97] = self.right_hand_ff_rot  
                self.obs_buf[:, 97:100] = self.right_hand_ff_linvel
                self.obs_buf[:, 100:103] = self.right_hand_ff_angvel

                self.obs_buf[:, 103:107] = self.right_hand_mf_rot  
                self.obs_buf[:, 107:110] = self.right_hand_mf_linvel
                self.obs_buf[:, 110:113] = self.right_hand_mf_angvel

                self.obs_buf[:, 113:117] = self.right_hand_rf_rot  
                self.obs_buf[:, 117:120] = self.right_hand_rf_linvel
                self.obs_buf[:, 120:123] = self.right_hand_rf_angvel

                self.obs_buf[:, 123:127] = self.right_hand_th_rot  
                self.obs_buf[:, 127:130] = self.right_hand_th_linvel
                self.obs_buf[:, 130:133] = self.right_hand_th_angvel

                self.obs_buf[:, 133:140] = self.object_pose
                self.obs_buf[:, 140:143] = self.object_linvel
                self.obs_buf[:, 143:146] = self.vel_obs_scale * self.object_angvel
                self.obs_buf[:, 146:149] = self.goal_pos - self.object_pos

                self.obs_buf[:, 149:152] = self.object_pos - self.right_hand_ff_pos
                self.obs_buf[:, 152:155] = self.object_pos - self.right_hand_mf_pos
                self.obs_buf[:, 155:158] = self.object_pos - self.right_hand_rf_pos
                self.obs_buf[:, 158:161] = self.object_pos - self.right_hand_th_pos

                self.obs_buf[:, 161:162] = self.arm_hand_finger_dist.unsqueeze(-1)
            
            else:
                if self.wo_velocity:
                    self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                            self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                    # Observation randomization.
                    self.obs_buf[:, 7:23] += (torch.rand_like(self.obs_buf[:, 7:23]) - 0.5) * 2 * 0.05
                    self.obs_buf[:, 0:7] += (torch.rand_like(self.obs_buf[:, 0:7]) - 0.5) * 2 * 0.02

                    self.obs_buf[:, 23:26] = self.right_hand_ff_pos
                    self.obs_buf[:, 26:30] = self.right_hand_ff_rot
                    self.obs_buf[:, 30:33] = self.right_hand_rf_pos
                    self.obs_buf[:, 33:37] = self.right_hand_rf_rot
                    self.obs_buf[:, 37:40] = self.right_hand_mf_pos
                    self.obs_buf[:, 40:44] = self.right_hand_mf_rot
                    self.obs_buf[:, 44:47] = self.right_hand_th_pos
                    self.obs_buf[:, 47:51] = self.right_hand_th_rot

                    self.obs_buf[:, 51:58] = self.hand_base_pose

                    self.obs_buf[:, 58:65] = self.object_pose

                    self.obs_buf[:, 65:68] = self.goal_pos - self.object_pos


                else:
                    self.obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                            self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                    self.obs_buf[:, self.num_arm_hand_dofs:2*self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel

                    # Observation randomization.
                    self.obs_buf[:, 7:23] += (torch.rand_like(self.obs_buf[:, 7:23]) - 0.5) * 2 * 0.05
                    self.obs_buf[:, 0:7] += (torch.rand_like(self.obs_buf[:, 0:7]) - 0.5) * 2 * 0.02

                    self.obs_buf[:, 46:49] = self.right_hand_ff_pos
                    self.obs_buf[:, 49:52] = self.right_hand_rf_pos
                    self.obs_buf[:, 52:55] = self.right_hand_mf_pos
                    self.obs_buf[:, 55:58] = self.right_hand_th_pos

                    self.obs_buf[:, 58:80] = self.actions
                    self.obs_buf[:, 80:87] = self.hand_base_pose

                    self.obs_buf[:, 87:90] = self.hand_base_linvel
                    self.obs_buf[:, 90:93] = self.hand_base_angvel

                    self.obs_buf[:, 93:97] = self.right_hand_ff_rot  
                    self.obs_buf[:, 97:100] = self.right_hand_ff_linvel
                    self.obs_buf[:, 100:103] = self.right_hand_ff_angvel

                    self.obs_buf[:, 103:107] = self.right_hand_mf_rot  
                    self.obs_buf[:, 107:110] = self.right_hand_mf_linvel
                    self.obs_buf[:, 110:113] = self.right_hand_mf_angvel

                    self.obs_buf[:, 113:117] = self.right_hand_rf_rot  
                    self.obs_buf[:, 117:120] = self.right_hand_rf_linvel
                    self.obs_buf[:, 120:123] = self.right_hand_rf_angvel

                    self.obs_buf[:, 123:127] = self.right_hand_th_rot  
                    self.obs_buf[:, 127:130] = self.right_hand_th_linvel
                    self.obs_buf[:, 130:133] = self.right_hand_th_angvel

                    self.obs_buf[:, 133:140] = self.object_pose
                    self.obs_buf[:, 140:143] = self.object_linvel
                    self.obs_buf[:, 143:146] = self.vel_obs_scale * self.object_angvel
                    self.obs_buf[:, 146:149] = self.goal_pos - self.object_pos

                    self.obs_buf[:, 149:152] = self.object_pos - self.right_hand_ff_pos
                    self.obs_buf[:, 152:155] = self.object_pos - self.right_hand_mf_pos
                    self.obs_buf[:, 155:158] = self.object_pos - self.right_hand_rf_pos
                    self.obs_buf[:, 158:161] = self.object_pos - self.right_hand_th_pos

            if self.is_distillation:
                # self.student_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                #                                                       self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                # self.student_obs_buf[:, 23:26] = self.right_hand_ff_pos
                # self.student_obs_buf[:, 26:29] = self.right_hand_rf_pos
                # self.student_obs_buf[:, 29:32] = self.right_hand_mf_pos
                # self.student_obs_buf[:, 32:35] = self.right_hand_th_pos
                # self.student_obs_buf[:, 35:57] = self.actions
                # self.student_obs_buf[:, 57:64] = self.hand_base_pose
                # self.student_obs_buf[:, 64:68] = self.right_hand_ff_rot
                # self.student_obs_buf[:, 68:72] = self.right_hand_mf_rot
                # self.student_obs_buf[:, 72:76] = self.right_hand_rf_rot
                # self.student_obs_buf[:, 76:80] = self.right_hand_th_rot
                # self.student_obs_buf[:, 80:87] = self.object_pose
                # self.student_obs_buf[:, 87:90] = self.goal_pos - self.object_pos

                # # trying to take out fingertip pose

                # self.student_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                #                                                       self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                # self.student_obs_buf[:, 23:45] = self.actions
                # self.student_obs_buf[:, 45:52] = self.hand_base_pose
                # self.student_obs_buf[:, 52:59] = self.object_pose
                # self.student_obs_buf[:, 59:62] = self.goal_pos - self.object_pos

                # trying to take out fingertip pose & hand_base pose & actions

                self.student_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                                                                      self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                self.student_obs_buf[:, 23:30] = self.object_pose
                self.student_obs_buf[:, 30:33] = self.goal_pos - self.object_pos

                # # debuging......

                # self.student_obs_buf[:, 0:self.num_arm_hand_dofs] = unscale(self.arm_hand_dof_pos,
                #                                                         self.arm_hand_dof_lower_limits, self.arm_hand_dof_upper_limits)
                # self.student_obs_buf[:, self.num_arm_hand_dofs:2*self.num_arm_hand_dofs] = self.vel_obs_scale * self.arm_hand_dof_vel

                # self.student_obs_buf[:, 46:49] = self.right_hand_ff_pos
                # self.student_obs_buf[:, 49:52] = self.right_hand_rf_pos
                # self.student_obs_buf[:, 52:55] = self.right_hand_mf_pos
                # self.student_obs_buf[:, 55:58] = self.right_hand_th_pos

                # self.student_obs_buf[:, 58:80] = self.actions
                # self.student_obs_buf[:, 80:87] = self.hand_base_pose

                # self.student_obs_buf[:, 87:90] = self.hand_base_linvel
                # self.student_obs_buf[:, 90:93] = self.hand_base_angvel

                # self.student_obs_buf[:, 93:97] = self.right_hand_ff_rot  
                # self.student_obs_buf[:, 97:100] = self.right_hand_ff_linvel
                # self.student_obs_buf[:, 100:103] = self.right_hand_ff_angvel

                # self.student_obs_buf[:, 103:107] = self.right_hand_mf_rot  
                # self.student_obs_buf[:, 107:110] = self.right_hand_mf_linvel
                # self.student_obs_buf[:, 110:113] = self.right_hand_mf_angvel

                # self.student_obs_buf[:, 113:117] = self.right_hand_rf_rot  
                # self.student_obs_buf[:, 117:120] = self.right_hand_rf_linvel
                # self.student_obs_buf[:, 120:123] = self.right_hand_rf_angvel

                # self.student_obs_buf[:, 123:127] = self.right_hand_th_rot  
                # self.student_obs_buf[:, 127:130] = self.right_hand_th_linvel
                # self.student_obs_buf[:, 130:133] = self.right_hand_th_angvel

                # self.student_obs_buf[:, 133:140] = self.object_pose
                # self.student_obs_buf[:, 140:143] = self.object_linvel
                # self.student_obs_buf[:, 143:146] = self.vel_obs_scale * self.object_angvel
                # self.student_obs_buf[:, 146:149] = self.goal_pos - self.object_pos

                # self.student_obs_buf[:, 149:152] = self.object_pos - self.right_hand_ff_pos
                # self.student_obs_buf[:, 152:155] = self.object_pos - self.right_hand_mf_pos
                # self.student_obs_buf[:, 155:158] = self.object_pos - self.right_hand_rf_pos
                # self.student_obs_buf[:, 158:161] = self.object_pos - self.right_hand_th_pos

                # self.student_obs_buf[:, 161:162] = self.arm_hand_finger_dist.unsqueeze(-1)

            # self.obs_buf[:, 2*self.num_arm_hand_dofs:3*self.num_arm_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor

            # fingertip_obs_start = 3*self.num_arm_hand_dofs  # 69
            # aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # # for i in range(4):
            # #     aux[:, i * 13:(i + 1) * 13] = aux[:, i * 13:(i + 1) * 13]
            # # 69:121: ft states
            # self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

            # hand_base_start = fingertip_obs_start + 52 

            # # 121:128: ft states

            # self.obs_buf[:, hand_base_start:hand_base_start + 7] = self.hand_base_pose

            # # hand_pose_start = hand_base_start + 7

            # # # 128:128: hand pose
            # # self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.right_hand_pos
            # # self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 7] = self.right_hand_rot

            # action_obs_start = hand_base_start + 7

            # aux = self.actions[:, :22]

            # # 128:150: actions
            # self.obs_buf[:, action_obs_start:action_obs_start + 22] = aux

            # obj_obs_start = action_obs_start + 22

            # # 150:166: object state
            # self.obs_buf[:, obj_obs_start:obj_obs_start + 7] = self.object_pose
            # self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.object_linvel
            # self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.object_angvel
            # self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.goal_pos - self.object_pos


    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]
        self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3] # + self.goal_displacement_tensor
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]
        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                         gymtorch.unwrap_tensor(self.root_state_tensor),
                                                         gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0

    def update_controller(self):
        previous_dof_pos = self.arm_hand_dof_pos.clone()
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)
        # self.gym.refresh_dof_force_tensor(self.sim)

        if self.torque_control:
            dof_pos = self.arm_hand_dof_pos
            dof_vel = (dof_pos - previous_dof_pos) / self.dt
            self.dof_vel_finite_diff = dof_vel.clone()
            # print("self.p_gain = ",self.p_gain)
            torques = self.p_gain * (self.cur_targets - dof_pos) - self.d_gain * dof_vel
            # print(torch.norm(torques[0]))
            # print("torques = ", torques[0,4])
            self.torques = torch.clip(torques, -20.0, 20.0).clone()
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
        else:
            self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self.cur_targets))
        return


    def reset_idx(self, env_ids, goal_env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)
        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_arm_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
        self.reset_target_pose(env_ids)

        # # reset rigid body forces
        # self.rb_forces[env_ids, :, :] = 0.0

        # reset the pd-gain.
        self.randomize_p_gain_lower = self.p_gain_val_hand * 0.8
        self.randomize_p_gain_upper = self.p_gain_val_hand * 1.2
        self.randomize_d_gain_lower = self.d_gain_val_hand * 0.8
        self.randomize_d_gain_upper = self.d_gain_val_hand * 1.2
        
        new_values = torch_rand_float(self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 16), device=self.device)
        self.p_gain[env_ids, 7:23] = new_values.squeeze(1)

        self.randomize_p_gain_lower = self.p_gain_val_arm_0123 * 0.5
        self.randomize_p_gain_upper = self.p_gain_val_arm_0123 * 2.0
        self.randomize_d_gain_lower = self.d_gain_val_arm_0123 * 0.5
        self.randomize_d_gain_upper = self.d_gain_val_arm_0123 * 2

        new_values = torch_rand_float(self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 4), device=self.device)
        self.p_gain[env_ids, 0:4] = new_values.squeeze(1)

        self.randomize_p_gain_lower = self.p_gain_val_arm_4 * 0.5
        self.randomize_p_gain_upper = self.p_gain_val_arm_4 * 2.0
        self.randomize_d_gain_lower = self.d_gain_val_arm_4 * 0.5
        self.randomize_d_gain_upper = self.d_gain_val_arm_4 * 2.0
        new_values = torch_rand_float(self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 1), device=self.device)
        self.p_gain[env_ids, 4] = new_values.squeeze(1)

        self.randomize_p_gain_lower = self.p_gain_val_arm_5 * 0.5
        self.randomize_p_gain_upper = self.p_gain_val_arm_5 * 2.0
        self.randomize_d_gain_lower = self.d_gain_val_arm_5 * 0.5
        self.randomize_d_gain_upper = self.d_gain_val_arm_5 * 2.0
        new_values = torch_rand_float(self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 1), device=self.device)
        self.p_gain[env_ids, 5] = new_values.squeeze(1)

        self.randomize_p_gain_lower = self.p_gain_val_arm_6 * 0.5
        self.randomize_p_gain_upper = self.p_gain_val_arm_6 * 2.0
        self.randomize_d_gain_lower = self.d_gain_val_arm_6 * 0.5
        self.randomize_d_gain_upper = self.d_gain_val_arm_6 * 2.0
        new_values = torch_rand_float(self.randomize_p_gain_lower, self.randomize_p_gain_upper, (len(env_ids), 1), device=self.device)
        self.p_gain[env_ids, 6] = new_values.squeeze(1)

        # reset object
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        self.root_state_tensor[self.object_indices[env_ids], 0:2] = self.object_init_state[env_ids, 0:2] + \
            self.reset_position_noise * rand_floats[:, 0:2]
        self.root_state_tensor[self.object_indices[env_ids], self.up_axis_idx] = self.object_init_state[env_ids, self.up_axis_idx] + \
            self.reset_position_noise * rand_floats[:, self.up_axis_idx]

        # new_object_rot = randomize_rotation(rand_floats[:, 3], rand_floats[:, 4], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])
        # if self.object_type == "pen":
        #     rand_angle_y = torch.tensor(0.3)
        #     new_object_rot = randomize_rotation_pen(rand_floats[:, 3], rand_floats[:, 4], rand_angle_y,
        #                                             self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids], self.z_unit_tensor[env_ids])

        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        # self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])

        object_indices = torch.unique(torch.cat([self.object_indices[env_ids],
                                                 self.goal_object_indices[env_ids],
                                                 self.goal_object_indices[goal_env_ids]]).to(torch.int32))
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(object_indices), len(object_indices))

        # # reset random force probabilities
        # self.random_force_prob[env_ids] = torch.exp((torch.log(self.force_prob_range[0]) - torch.log(self.force_prob_range[1]))
        #                                             * torch.rand(len(env_ids), device=self.device) + torch.log(self.force_prob_range[1]))

        # reset arm hand
        delta_max = self.arm_hand_dof_upper_limits - self.arm_hand_dof_default_pos
        delta_min = self.arm_hand_dof_lower_limits - self.arm_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * (rand_floats[:, 5:5+self.num_arm_hand_dofs] + 1)

        pos = self.arm_hand_default_dof_pos.clone()  
        # arm_hand start pos randomization
        pos += (torch.rand_like(pos) - 0.5) * 2 * 0.02

        self.arm_hand_dof_pos[env_ids, :] = pos
        self.arm_hand_dof_vel[env_ids, :] = self.arm_hand_dof_default_vel + \
            self.reset_dof_vel_noise * rand_floats[:, 5+self.num_arm_hand_dofs:5+self.num_arm_hand_dofs*2]
        self.prev_targets[env_ids, :self.num_arm_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_arm_hand_dofs] = pos

        hand_indices = self.hand_indices[env_ids].to(torch.int32)
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        self.progress_buf[env_ids] = 0
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
            self.reset_idx(env_ids, goal_env_ids)

        self.actions = actions.clone().to(self.device)
        if self.use_relative_control:
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets,
                                                                          self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, 6:22],
                                                                        self.arm_hand_dof_lower_limits[self.actuated_dof_indices],
                                                                        self.arm_hand_dof_upper_limits[self.actuated_dof_indices])

            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,
                                                                                                        self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]

            self.relative_scale_tensor = torch.full_like(self.relative_scale_tensor, 0.2)  # *                                 (1 + (torch.rand_like(self.relative_scale_tensor) - 0.5) * 1)
            self.dpose = self.actions[:, 0:6] * self.relative_scale_tensor

            self.hand_base_last_pos = self.rigid_body_states[:, self.hand_base_rigid_body_index, 0:3].clone()
            # print("dpose sim= ",self.dpose)
            # dpose =dpose.unsqueeze(-1) # for control_ik()
            delta = control_ik_pseudo_inverse(self.jacobian_tensor[:, self.hand_base_rigid_body_index - 1, :, :7], self.device, self.dpose, self.num_envs)

            # print("delta sim= ", delta[0,:7])

            self.cur_targets[:, :7] = delta[:,:7]  + self.arm_hand_dof_pos[:,:7]  # self.prev_targets[:,:7] self.arm_hand_dof_pos[:,:7]

            self.cur_targets[:, :] = tensor_clamp(self.cur_targets[:, :],
                                            self.arm_hand_dof_lower_limits[:],
                                            self.arm_hand_dof_upper_limits[:])            
            
            self.prev_targets[:, :] = self.cur_targets[:, :].clone()


    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        self.compute_observations()
        self.compute_reward(self.actions)

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                targetx = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                targety = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                targetz = (self.goal_pos[i] + quat_apply(self.goal_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.goal_pos[i].cpu().numpy() + self.goal_displacement_tensor.cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetx[0], targetx[1], targetx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targety[0], targety[1], targety[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], targetz[0], targetz[1], targetz[2]], [0.1, 0.1, 0.85])

                objectx = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                objecty = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                objectz = (self.object_pos[i] + quat_apply(self.object_rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                p0 = self.object_pos[i].cpu().numpy()
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectx[0], objectx[1], objectx[2]], [0.85, 0.1, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objecty[0], objecty[1], objecty[2]], [0.1, 0.85, 0.1])
                self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], objectz[0], objectz[1], objectz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================jit functions=========================###
#####################################################################


@torch.jit.script
def compute_hand_reward(
    dof_pos, rew_buf, reset_buf, reset_goal_buf, 
    progress_buf, successes, consecutive_successes, 
    max_episode_length: float, object_pos, object_rot, 
    target_pos, target_rot,
    arm_hand_pos ,
    arm_hand_ff_pos, arm_hand_mf_pos, arm_hand_rf_pos, arm_hand_th_pos, 
    dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
    actions, action_penalty_scale: float,
    success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
    fall_penalty: float, max_consecutive_successes: int, av_factor: float, torque_penalty, work_penalty, cartesian_error
):
    #######################################
    #######       grasping reward   #######
    #######################################
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - arm_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_pos - arm_hand_pos, p=2, dim=-1)
    # right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)
    # right_hand_dist = torch.where(right_hand_dist <= 0.1, 0.1, right_hand_dist)

    object_handle_pos2  = object_pos.clone()
    object_handle_pos2[:,2] = object_pos[:,2]

    right_hand_finger_dist = (torch.norm(object_handle_pos2 - arm_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos2 - arm_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos2 - arm_hand_rf_pos, p=2, dim=-1) + 3 * torch.norm(object_handle_pos2 - arm_hand_th_pos, p=2, dim=-1))
    # right_hand_finger_dist = torch.where(right_hand_finger_dist >= 3.0, 3.0 + 0 * right_hand_finger_dist,right_hand_finger_dist)
    # finger dist threshold #                    

    lowest = object_pos[:, 2]
    # print("lowest = ",lowest)
    # lift_z =  0.6281 # 0.6281 ycb/bottle, 0.6376 bottle  

    right_hand_finger_dist_thres = 0.10 * 4
    right_hand_dist_thres = 0.10
    flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= right_hand_dist_thres).int()
    # print("right_hand_finger_dist = ",right_hand_finger_dist[0])
    # print("right_hand_dist = ",right_hand_dist[0])

    # if flag[0].item() == 2:  
    #     print("\033[31mflag = 2\033[0m")

    # if flag[0].item() == 1: 
    #     print("flag = 1")
    #     if right_hand_dist[0] <= 0.10:
    #         print("plam is close") 
    #     if right_hand_finger_dist[0] <= right_hand_finger_dist_thres:
    #         print("finger is close") 

    goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
    goal_hand_rew = torch.where(flag == 2, 1 * (0.9 - 2 * goal_dist), goal_hand_rew)

    hand_up = torch.zeros_like(right_hand_finger_dist)
    hand_up = torch.where(lowest >= 0.640, torch.where(flag == 2, 0.1 + 0.1 * arm_hand_pos[:,2], hand_up), hand_up)
    hand_up = torch.where(lowest >= 0.80, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)


    flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= right_hand_dist_thres).int()
    bonus = torch.zeros_like(goal_dist)
    bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

    action_penalty = torch.sum(actions ** 2, dim=-1)

    controller_penalty = cartesian_error ** 2

    reward = - 0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus #- 0.003 * torque_penalty - 0.001 * action_penalty #- 1 * controller_penalty #-2 * work_penalty - 1 * controller_penalty

    resets = torch.where(right_hand_dist >= 0.9, torch.ones_like(reset_buf), reset_buf)
    # resets = torch.where(right_hand_finger_dist >= 3.3, torch.ones_like(reset_buf), reset_buf)   

    # resets = torch.where(progress_buf >= 400, 
    #             torch.where(right_hand_finger_dist >= right_hand_finger_dist_thres, torch.ones_like(resets), resets), resets)

    # resets = reset_buf
    timed_out = progress_buf >= max_episode_length - 1
    resets = torch.where(timed_out, torch.ones_like(resets), resets)

    goal_resets = resets
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)



    return reward, resets, goal_resets, progress_buf, successes, cons_successes


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot

def orientation_error(desired, current):
	cc = quat_conjugate(current)
	q_r = quat_mul(desired, cc)
	return q_r[:, 0:3] * torch.sign(q_r[:, 3]).unsqueeze(-1)

def control_ik(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    damping = 0.05
    j_eef = j_eef.float()      # or .double()
    dpose = dpose.float() 
    # solve damped least squares
    j_eef_T = torch.transpose(j_eef, 1, 2)
    lmbda = torch.eye(6, device=device) * (damping ** 2)
    u = (j_eef_T @ torch.inverse(j_eef @ j_eef_T + lmbda) @ dpose).view(num_envs, -1)
    return u

def control_ik_pseudo_inverse(j_eef, device, dpose, num_envs):
	# Set controller parameters
	# IK params
    lambda_val = 0.05
    j_eef = j_eef.float()      # or .double()
    dpose = dpose.float() 
    # computation
    jacobian_T = torch.transpose(j_eef, dim0=1, dim1=2) # n, q, 6
    lambda_matrix = (lambda_val**2) * torch.eye(n=6, device=device)
    kin_matrix = torch.bmm(j_eef, jacobian_T) + lambda_matrix[None, ...] # n, 6, 6
    delta_joint_pos = torch.bmm(jacobian_T, torch.linalg.solve(kin_matrix, dpose.unsqueeze(-1)))
    delta_joint_pos = delta_joint_pos.squeeze(-1)
    return delta_joint_pos

def recover_action(action, limit):
    action = (action + 1) / 2 * (limit[:, 1] - limit[:, 0]) + limit[:, 0]
    return action
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

from enum import Enum

from .hand_amp_base import HandAMPBase
from gym import spaces
from utils.torch_utils import *

from utils import torch_utils
from utils.motion_lib import MotionLib


class HandAMP(HandAMPBase):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        # state_init = 'Default' 
        self._state_init = HandAMP.StateInit[state_init]

        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]
        self._num_amp_obs_enc_steps = cfg["env"].get("numAMPEncObsSteps", self._num_amp_obs_steps)

        self._equal_motion_weights = cfg["env"].get("equal_motion_weights", False)
        assert(self._num_amp_obs_steps >= 2)

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super().__init__(cfg=self.cfg, 
                         rl_device=rl_device, 
                         sim_device=sim_device, 
                         graphics_device_id=graphics_device_id, 
                         headless=headless, 
                         virtual_screen_capture=virtual_screen_capture, 
                         force_render=force_render,
                         agent_index=agent_index,
                         is_multi_agent=is_multi_agent, 
                        )

        motion_file = cfg['env']['motion_file']
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/" + motion_file)
        # motion_file_path = os.path.join(asset_root, "amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step
        self.num_amp_obs_enc = self._num_amp_obs_enc_steps * self._num_amp_obs_per_step
        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)
        


        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]

        return

    @property
    def amp_observation_space(self):
        return self._amp_obs_space

    @property
    def task_obs_size(self):
        return 0 

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return
    
    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self._num_amp_obs_steps * self._num_amp_obs_per_step


    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self.shadow_hand_dof_pos, self.shadow_hand_dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._dof_offsets)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                     self._rigid_body_rot[env_ids][:, 0, :],
                                                                     self._rigid_body_vel[env_ids][:, 0, :],
                                                                     self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                     self.shadow_hand_dof_pos[env_ids], self.shadow_hand_dof_vel[env_ids],
                                                                     key_body_pos[env_ids],
                                                                     self._local_root_obs, self._root_height_obs,
                                                                     self._dof_obs_size, self._dof_offsets)
        return



    def fetch_amp_obs_demo(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        truncate_time = self.dt * (self._num_amp_obs_steps - 1)
        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=truncate_time)
        motion_times0 += truncate_time

        amp_obs_demo_flat = self.build_amp_obs_demo(motion_ids, motion_times0, self._num_amp_obs_steps).to(self.device).view(-1, self.get_num_amp_obs())

        return amp_obs_demo_flat

    def fetch_amp_obs_demo_per_id(self, num_samples, motion_id):
        motion_ids = torch.tensor([motion_id for _ in range(num_samples)], dtype=torch.long).view(-1)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        enc_amp_obs_demo = self.build_amp_obs_demo(motion_ids, enc_motion_times, self._num_amp_obs_enc_steps).view(-1,
                                                                                                                   self._num_amp_obs_enc_steps,
                                                                                                                   self._num_amp_obs_per_step)

        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device).view(-1, self.get_num_enc_amp_obs())

        return motion_ids, enc_motion_times, enc_amp_obs_demo_flat

    def fetch_amp_obs_demo_enc_pair(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        enc_motion_times = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        # make sure not to add more than motion clip length, negative amp_obs will show zero index amp_obs instead
        enc_motion_times += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        # sub-window-size is for the amp_obs contained within the enc-amp-obs. make sure we sample only within the valid portion of the motion
        sub_window_size = torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size) - self.dt * self._num_amp_obs_steps
        motion_times = enc_motion_times - torch.rand(enc_motion_times.shape, device=self.device) * sub_window_size

        enc_amp_obs_demo = self.build_amp_obs_demo(motion_ids, enc_motion_times, self._num_amp_obs_enc_steps).view(-1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step)
        amp_obs_demo = self.build_amp_obs_demo(motion_ids, motion_times, self._num_amp_obs_steps).view(-1, self._num_amp_obs_steps, self._num_amp_obs_per_step)

        enc_amp_obs_demo_flat = enc_amp_obs_demo.to(self.device).view(-1, self.get_num_enc_amp_obs())
        amp_obs_demo_flat = amp_obs_demo.to(self.device).view(-1, self.get_num_amp_obs())

        return motion_ids, enc_motion_times, enc_amp_obs_demo_flat, motion_times, amp_obs_demo_flat

    def fetch_amp_obs_demo_pair(self, num_samples):
        motion_ids = self._motion_lib.sample_motions(num_samples)
        cat_motion_ids = torch.cat((motion_ids, motion_ids), dim=0)

        # since negative times are added to these values in build_amp_obs_demo,
        # we shift them into the range [0 + truncate_time, end of clip]
        enc_window_size = self.dt * (self._num_amp_obs_enc_steps - 1)

        motion_times0 = self._motion_lib.sample_time(motion_ids, truncate_time=enc_window_size)
        motion_times0 += torch.clip(self._motion_lib._motion_lengths[motion_ids], max=enc_window_size)

        motion_times1 = motion_times0 + torch.rand(motion_times0.shape, device=self._motion_lib._device) * 0.5
        motion_times1 = torch.min(motion_times1, self._motion_lib._motion_lengths[motion_ids])

        motion_times = torch.cat((motion_times0, motion_times1), dim=0)

        amp_obs_demo = self.build_amp_obs_demo(cat_motion_ids, motion_times, self._num_amp_obs_enc_steps).view(-1, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step)
        amp_obs_demo0, amp_obs_demo1 = torch.split(amp_obs_demo, num_samples)

        amp_obs_demo0_flat = amp_obs_demo0.to(self.device).view(-1, self.get_num_enc_amp_obs())

        amp_obs_demo1_flat = amp_obs_demo1.to(self.device).view(-1, self.get_num_enc_amp_obs())

        return motion_ids, motion_times0, amp_obs_demo0_flat, motion_times1, amp_obs_demo1_flat

    def build_amp_obs_demo(self, motion_ids, motion_times0, num_steps):
        dt = self.dt

        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, num_steps])
        motion_times = motion_times0.unsqueeze(-1)
        time_steps = -dt * torch.arange(0, num_steps, device=self.device)
        motion_times = torch.clip(motion_times + time_steps, min=0)

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              dof_pos, dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self._dof_obs_size, self._dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        self._enc_amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_enc_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return

    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        asset_file = self.cfg["env"]["asset"]["assetFileName"]
        num_key_bodies = len(key_bodies)

        # if asset_file == "mjcf/amp_humanoid.xml":
        #     self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        # elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
        #     self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        # else:
        #     print("Unsupported character config file: {s}".format(asset_file))
        #     assert False

        # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        # self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies 
        # self._num_amp_obs_per_step = 13 + self._dof_obs_size + 22 + 3 * num_key_bodies 
        self._num_amp_obs_per_step = 6 + self._dof_obs_size + 22 + 3 * num_key_bodies 
        return

    def _load_motion(self, motion_file):
        assert(self._dof_offsets[-1] == self.num_shadow_hand_dofs)
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self._dof_body_ids,
                                     dof_offsets=self._dof_offsets,
                                     key_body_ids=self._key_body_ids.cpu().numpy(),
                                     equal_motion_weights=self._equal_motion_weights,
                                     device=self.device)
        # self._motion_lib = MotionLib(motion_file=motion_file, 
        #                              num_dofs=self.num_dof,
        #                              key_body_ids=self._key_body_ids.cpu().numpy(), 
        #                              device=self.device)
        return
    
    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)
        self._init_amp_obs(env_ids)

        return

    def reset_idx(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        if len(env_ids) == 0:
            return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        
        self._reset_actors(env_ids)
        super().reset_idx(env_ids)
        self._init_amp_obs(env_ids)

        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def _reset_actors(self, env_ids):
        if self._state_init == HandAMP.StateInit.Default:
            self._reset_default(env_ids)
        elif (self._state_init == HandAMP.StateInit.Start
              or self._state_init == HandAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == HandAMP.StateInit.Hybrid:
            self._reset_hybrid_state_init(env_ids)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        return
    
    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        motion_ids = self._motion_lib.sample_motions(num_envs)
        
        if (self._state_init == HandAMP.StateInit.Random
            or self._state_init == HandAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == HandAMP.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)

        # self._set_env_state(
        #     env_ids=env_ids,
        #     root_pos=root_pos,
        #     root_rot=root_rot,
        #     dof_pos=dof_pos,
        #     root_vel=root_vel,
        #     root_ang_vel=root_ang_vel,
        #     dof_vel=dof_vel
        # )

        self._reset_ref_env_ids = env_ids
        self._reset_ref_motion_ids = motion_ids
        self._reset_ref_motion_times = motion_times
        return

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]
        if len(ref_reset_ids) > 0:
            self._reset_ref_state_init(ref_reset_ids)

        default_reset_ids = env_ids[torch.logical_not(ref_init_mask)]
        if len(default_reset_ids) > 0:
            self._reset_default(default_reset_ids)

        return

    def _init_amp_obs(self, env_ids):
        self._compute_amp_observations(env_ids)

        if len(self._reset_default_env_ids) > 0:
            self._init_amp_obs_default(self._reset_default_env_ids)

        if len(self._reset_ref_env_ids) > 0:
            self._init_amp_obs_ref(self._reset_ref_env_ids, self._reset_ref_motion_ids,
                                   self._reset_ref_motion_times)
        
        return

    def _init_amp_obs_default(self, env_ids):
        curr_amp_obs = self._curr_amp_obs_buf[env_ids].unsqueeze(-2)
        self._hist_amp_obs_buf[env_ids] = curr_amp_obs
        return

    def _init_amp_obs_ref(self, env_ids, motion_ids, motion_times):
        dt = self.dt
        motion_ids = torch.tile(motion_ids.unsqueeze(-1), [1, self._num_amp_obs_steps - 1])
        motion_times = motion_times.unsqueeze(-1)
        time_steps = -dt * (torch.arange(0, self._num_amp_obs_steps - 1, device=self.device) + 1)
        motion_times = motion_times + time_steps

        motion_ids = motion_ids.view(-1)
        motion_times = motion_times.view(-1)
        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
               = self._motion_lib.get_motion_state(motion_ids, motion_times)
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, 
                                              dof_pos, dof_vel, key_pos, 
                                              self._local_root_obs, self._root_height_obs, 
                                              self._dof_obs_size, self._dof_offsets)
        self._hist_amp_obs_buf[env_ids] = amp_obs_demo.view(self._hist_amp_obs_buf[env_ids].shape)
        return

    def get_task_obs_size(self):
        return 0

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return




#####################################################################
###=========================jit functions=========================###
#####################################################################

# @torch.jit.script
def build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, dof_offsets):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, bool, bool, int, List[int]) -> Tensor
    root_h = root_pos[:, 2:3]
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)

    if local_root_obs:
        root_rot_obs = quat_mul(heading_rot, root_rot)
    else:
        root_rot_obs = root_rot
    root_rot_obs = torch_utils.quat_to_tan_norm(root_rot_obs)
    
    if not root_height_obs:
        root_h_obs = torch.zeros_like(root_h)
    else:
        root_h_obs = root_h
    
    local_root_vel = quat_rotate(heading_rot, root_vel)
    local_root_ang_vel = quat_rotate(heading_rot, root_ang_vel)

    root_pos_expand = root_pos.unsqueeze(-2)
    local_key_body_pos = key_body_pos - root_pos_expand
    
    heading_rot_expand = heading_rot.unsqueeze(-2)
    heading_rot_expand = heading_rot_expand.repeat((1, local_key_body_pos.shape[1], 1))
    flat_end_pos = local_key_body_pos.view(local_key_body_pos.shape[0] * local_key_body_pos.shape[1], local_key_body_pos.shape[2])
    flat_heading_rot = heading_rot_expand.view(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                               heading_rot_expand.shape[2])
    local_end_pos = quat_rotate(flat_heading_rot, flat_end_pos)
    flat_local_key_pos = local_end_pos.view(local_key_body_pos.shape[0], local_key_body_pos.shape[1] * local_key_body_pos.shape[2])
    
    dof_obs = dof_to_obs(dof_pos, dof_obs_size, dof_offsets)
    # obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    obs = torch.cat((local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def dof_to_obs(pose, dof_obs_size, dof_offsets):
    # type: (Tensor, int, List[int]) -> Tensor
    joint_obs_size = 6
    num_joints = len(dof_offsets) - 1

    dof_obs_shape = pose.shape[:-1] + (dof_obs_size,)
    dof_obs = torch.zeros(dof_obs_shape, device=pose.device)
    dof_obs_offset = 0

    for j in range(num_joints):
        dof_offset = dof_offsets[j]
        dof_size = dof_offsets[j + 1] - dof_offsets[j]
        joint_pose = pose[:, dof_offset:(dof_offset + dof_size)]

        # assume this is a spherical joint
        if dof_size == 3:
            joint_pose_q = torch_utils.exp_map_to_quat(joint_pose)
        elif dof_size == 1:
            axis = torch.tensor([0.0, 1.0, 0.0], dtype=joint_pose.dtype, device=pose.device)
            joint_pose_q = quat_from_angle_axis(joint_pose[..., 0], axis)
        else:
            assert False, "Unsupported joint type"

        joint_dof_obs = torch_utils.quat_to_tan_norm(joint_pose_q)
        dof_obs[:, (j * joint_obs_size):((j + 1) * joint_obs_size)] = joint_dof_obs

    assert((num_joints * joint_obs_size) == dof_obs_size)

    return dof_obs

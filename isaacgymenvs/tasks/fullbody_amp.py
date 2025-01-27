# Copyright (c) 2018-2022, NVIDIA Corporation
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

from enum import Enum
import numpy as np
import torch

# from tasks.humanoid_amp_base import FullBodyAMPBase, dof_to_obs
from tasks.fullbody import FullBody
from utils.motion_lib import MotionLib
# from isaacgymenvs.tasks.amp.utils_amp.motion_lib import MotionLib
from utils.torch_utils import *

from utils import torch_utils
import os
from gym import spaces

# 15
humanoid_amp_rigid_body_name_list = [
    'pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 
    'right_hand', 
    'left_upper_arm', 'left_lower_arm', 'left_hand', 
    'right_thigh', 'right_shin', 'right_foot', 
    'left_thigh', 'left_shin', 'left_foot'
]

# 38
fullbody_amp_rigid_body_name_list = [
    'pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 
    'robot0:hand mount', 'robot0:palm', 
    'robot0:ffknuckle', 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
    'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 'robot0:mfdistal', 
    'robot0:rfknuckle', 'robot0:rfproximal', 'robot0:rfmiddle', 'robot0:rfdistal', 
    'robot0:lfmetacarpal', 'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal', 
    'robot0:thbase', 'robot0:thproximal', 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal', 
    'left_upper_arm', 'left_lower_arm', 'left_hand', 
    'right_thigh', 'right_shin', 'right_foot', 
    'left_thigh', 'left_shin', 'left_foot'
]

amp_rigid_body_name_list = [
    'pelvis', 'torso', 'head', 
    'left_upper_arm', 'left_lower_arm', 'left_hand', 
    'right_thigh', 'right_shin', 'right_foot', 
    'left_thigh', 'left_shin', 'left_foot'
]
# amp_rigid_body_name_list = [
#     'pelvis', 'torso', 'head', 'right_upper_arm', 'right_lower_arm', 
#     'left_upper_arm', 'left_lower_arm', 'left_hand', 
#     'right_thigh', 'right_shin', 'right_foot', 
#     'left_thigh', 'left_shin', 'left_foot'
# ]
humanoid_amp_rigid_bodies_id = [humanoid_amp_rigid_body_name_list.index(ns) for ns in amp_rigid_body_name_list]
fullbody_amp_rigid_bodies_id = [fullbody_amp_rigid_body_name_list.index(ns) for ns in amp_rigid_body_name_list]


class FullBodyAMP(FullBody):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        state_init = cfg["env"]["stateInit"]
        self._state_init = FullBodyAMP.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]
        self._num_amp_obs_steps = cfg["env"]["numAMPObsSteps"]

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
                        )
        
        amp_key_bodies = self.cfg["env"]["AMPkeyBodies"] 
        fullbody_amp_key_bodies_ids = [fullbody_amp_rigid_body_name_list.index(ns) for ns in amp_key_bodies]
        self.fullbody_amp_key_bodies_ids = to_torch(fullbody_amp_key_bodies_ids, device=self.device, dtype=torch.long)
        humanoid_amp_key_bodies_ids = [humanoid_amp_rigid_body_name_list.index(ns) for ns in amp_key_bodies]
        self.humanoid_amp_key_bodies_ids = to_torch(humanoid_amp_key_bodies_ids, device=self.device, dtype=torch.long)
        
        motion_file = cfg['env']['motion_file']
        asset_root = self.cfg["env"]["asset"]["assetRoot"]
        # motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/" + motion_file)
        motion_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets/amp/motions/new_walk/dataset_walking.yaml")
        # motion_file_path = os.path.join(asset_root, "amp/motions/" + motion_file)
        self._load_motion(motion_file_path)

        
        self.num_amp_obs = self._num_amp_obs_steps * self._num_amp_obs_per_step
        self._amp_obs_space = spaces.Box(np.ones(self.num_amp_obs) * -np.Inf, np.ones(self.num_amp_obs) * np.Inf)

        self._amp_obs_buf = torch.zeros((self.num_envs, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float)
        self._curr_amp_obs_buf = self._amp_obs_buf[:, 0]
        self._hist_amp_obs_buf = self._amp_obs_buf[:, 1:]
        
        self._amp_obs_demo_buf = None
        self._prev_root_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        
        self._tar_speed = self.cfg["env"]["tarSpeed"]
        self._tar_change_steps_min = self.cfg["env"]["tarChangeStepsMin"]
        self._tar_change_steps_max = self.cfg["env"]["tarChangeStepsMax"]
        self._tar_dist_max = self.cfg["env"]["tarDistMax"]
         
        return

    def post_physics_step(self):
        super().post_physics_step()

        self._update_hist_amp_obs()
        self._compute_amp_observations()

        amp_obs_flat = self._amp_obs_buf.view(-1, self.get_num_amp_obs())
        self.extras["amp_obs"] = amp_obs_flat

        return

    def get_num_amp_obs(self):
        return self.num_amp_obs

    @property
    def amp_observation_space(self):
        return self._amp_obs_space


    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)

        self.humanoid_dof_body_ids = [1, 2, 3, 4, 6, 7, 9, 10, 11, 12, 13, 14]
        self.humanoid_dof_offsets = [0, 3, 6, 9, 10, 13, 14, 17, 18, 21, 24, 25, 28]
        self.humanoid_dof_size = [self.humanoid_dof_offsets[i+1] - self.humanoid_dof_offsets[i] \
            for i in range(len(self.humanoid_dof_offsets)-1)]
       
        humanoid_amp_rigid_bodies_id 
        self.humanoid_amp_dof_ids = []
        self.humanoid_amp_dof_offsets = [0]
        self.humanoid_amp_dof_size = []
        cnt = 0
        for i in range(len(self.humanoid_dof_body_ids)):
            offsets = self.humanoid_dof_offsets[i+1] - self.humanoid_dof_offsets[i]
            if self.humanoid_dof_body_ids[i] in humanoid_amp_rigid_bodies_id:
                for ii in range(offsets):
                    self.humanoid_amp_dof_ids.append(cnt+ii)
                self.humanoid_amp_dof_size.append(offsets)
            # else:
            cnt += offsets
        for dof_size in self.humanoid_amp_dof_size:
            self.humanoid_amp_dof_offsets.append(self.humanoid_amp_dof_offsets[-1]+dof_size)
        
        
        self.fullbody_amp_dof_ids = []
        self.fullbody_amp_dof_offsets = [0]
        self.fullbody_amp_dof_size = []
        cnt = 0
        for i in range(len(self._dof_body_ids)):
            offsets = self._dof_offsets[i+1] - self._dof_offsets[i]
            if self._dof_body_ids[i] in fullbody_amp_rigid_bodies_id:
                for ii in range(offsets):
                    self.fullbody_amp_dof_ids.append(cnt+ii)
                self.fullbody_amp_dof_size.append(offsets)
            # else:
            cnt += offsets
        for dof_size in self.fullbody_amp_dof_size:
            self.fullbody_amp_dof_offsets.append(self.fullbody_amp_dof_offsets[-1]+dof_size) 
        debug = 10
            
        
        amp_key_bodies = self.cfg["env"]["AMPkeyBodies"]
        
         
        # asset_file = self.cfg["env"]["asset"]["assetFileName"]
        self.amp_dof_obs_size = 6 * len(self.humanoid_amp_dof_size)

        # self.amp_key_bodies
        num_amp_key_bodies = len(amp_key_bodies)
        len_dof_vel = self.fullbody_amp_dof_offsets[-1] 
        self._num_amp_obs_per_step = 13 + self.amp_dof_obs_size + len_dof_vel + 3 * num_amp_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        return 
        
        if asset_file == "mjcf/amp_humanoid.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 28 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        elif asset_file == "mjcf/amp_humanoid_sword_shield.xml":
            self._num_amp_obs_per_step = 13 + self._dof_obs_size + 31 + 3 * num_key_bodies # [root_h, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos]
        else:
            print("Unsupported character config file: {s}".format(asset_file))
            assert False

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

        
        
        # amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
        #                                       dof_pos, dof_vel, key_pos,
        #                                       self._local_root_obs, self._root_height_obs,
        #                                       self.amp_dof_obs_size, self._dof_offsets)
        
        humanoid_amp_dof_vel = dof_vel[:,self.humanoid_amp_dof_ids]
        humanoid_amp_dof_pos = dof_pos[:,self.humanoid_amp_dof_ids]
        amp_obs_demo = build_amp_observations(root_pos, root_rot, root_vel, root_ang_vel,
                                              humanoid_amp_dof_pos, humanoid_amp_dof_vel, key_pos,
                                              self._local_root_obs, self._root_height_obs,
                                              self.amp_dof_obs_size, self.humanoid_amp_dof_offsets)
        return amp_obs_demo

    def _build_amp_obs_demo_buf(self, num_samples):
        self._amp_obs_demo_buf = torch.zeros((num_samples, self._num_amp_obs_steps, self._num_amp_obs_per_step), device=self.device, dtype=torch.float32)
        return



    def _load_motion(self, motion_file):
        self._motion_lib = MotionLib(motion_file=motion_file,
                                     dof_body_ids=self.humanoid_dof_body_ids,
                                     dof_offsets=self.humanoid_dof_offsets,
                                     key_body_ids=self.humanoid_amp_key_bodies_ids.cpu().numpy(),
                                     equal_motion_weights=self._equal_motion_weights,
                                     device=self.device)
        
        return
    
    
    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)

        self._reset_default_env_ids = env_ids
        self._reset_ref_env_ids = []
        self._init_amp_obs(env_ids)
     

    def _reset_actors(self, env_ids):
        if self._state_init == FullBodyAMP.StateInit.Default:
            self._reset_default(env_ids)
        elif (self._state_init == FullBodyAMP.StateInit.Start
              or self._state_init == FullBodyAMP.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif self._state_init == FullBodyAMP.StateInit.Hybrid:
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
        
        if (self._state_init == FullBodyAMP.StateInit.Random
            or self._state_init == FullBodyAMP.StateInit.Hybrid):
            motion_times = self._motion_lib.sample_time(motion_ids)
        elif self._state_init == FullBodyAMP.StateInit.Start:
            motion_times = torch.zeros(num_envs, device=self.device)
        else:
            assert False, "Unsupported state initialization strategy: {:s}".format(str(self._state_init))

        root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel, key_pos \
            = self._motion_lib.get_motion_state(motion_ids, motion_times)

        self._set_env_state(
            env_ids=env_ids,
            root_pos=root_pos,
            root_rot=root_rot,
            dof_pos=dof_pos,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            dof_vel=dof_vel
        )

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
                                              self.amp_dof_obs_size, self._dof_offsets)
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

    def _update_hist_amp_obs(self, env_ids=None):
        if env_ids is None:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[:, i + 1] = self._amp_obs_buf[:, i]
        else:
            for i in reversed(range(self._amp_obs_buf.shape[1] - 1)):
                self._amp_obs_buf[env_ids, i + 1] = self._amp_obs_buf[env_ids, i]
        return

    def _compute_amp_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self.fullbody_amp_key_bodies_ids, :]
        dof_pos = self.fullbody_dof_pos[:,self.fullbody_amp_dof_ids] 
        dof_vel = self.fullbody_dof_vel[:,self.fullbody_amp_dof_ids] 
        
        if env_ids is None:
            self._curr_amp_obs_buf[:] = build_amp_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               dof_pos, dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self.amp_dof_obs_size, self.fullbody_amp_dof_offsets)
        else:
            self._curr_amp_obs_buf[env_ids] = build_amp_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                     self._rigid_body_rot[env_ids][:, 0, :],
                                                                     self._rigid_body_vel[env_ids][:, 0, :],
                                                                     self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                     dof_pos[env_ids], dof_vel[env_ids],
                                                                     key_body_pos[env_ids],
                                                                     self._local_root_obs, self._root_height_obs,
                                                                     self.amp_dof_obs_size, self.fullbody_amp_dof_offsets)
        
        return


    def pre_physics_step(self, actions):
        super().pre_physics_step(actions)
        # self._prev_root_pos[:] = self._humanoid_root_states[..., 0:3]
        self._prev_root_pos[:] = self.rigid_body_states[...,0,0:3]
        return  
    
    # def compute_reward(self, actions):
    #     super().compute_reward(actions)
    #     self.rew_buf[:] = torch.ones_like(self.rew_buf[:])

    # def compute_reward(self, actions):

    #     root_pos = self.rigid_body_states[...,0,0:3]
    #     root_rot = self.rigid_body_states[...,0,3:7]

    #     object_pos = self.root_state_tensor[self.object_indices, 0:3]
        
    #     tmp_rew_buf_location = compute_location_reward(root_pos, self._prev_root_pos, root_rot,
    #                                              object_pos[...,:2], self._tar_speed,
    #                                              self.dt)
        
        
    #     tmp_rew_buf_hand, _, self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.consecutive_successes[:] = compute_hand_reward(
    #         self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
    #         self.object_id_buf, self.fullbody_dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
    #         self.progress_buf, self.successes, self.consecutive_successes,
    #         self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
    #         self.goal_pos, self.goal_rot,
    #         self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
    #         self.right_hand_lf_pos, self.right_hand_th_pos,
    #         self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.hand_sensor, self.action_penalty_scale,
    #         self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
    #         self.max_consecutive_successes, self.av_factor,self.goal_cond,
    #         self.static_body_delta,
    #         self._rigid_body_vel,
    #     )

    #     pos_diff = object_pos[...,:2]-root_pos[...,:2]
    #     pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    #     # self.rew_buf[:] = torch.where(dist>0.5, tmp_rew_buf_location, tmp_rew_buf_hand+0.5*torch.ones_like(tmp_rew_buf_location))
    #     self.rew_buf[:] = torch.where(pos_err>0.5, tmp_rew_buf_location, tmp_rew_buf_hand+tmp_rew_buf_location)
    #     # self.rew_buf[:] = tmp_rew_buf_location
    #     self.extras['successes'] = self.successes
    #     self.extras['consecutive_successes'] = self.consecutive_successes
    #     self.extras["terminate"] = self.successes
        

    #     pass
    
    
    def render(self, sync_frame_time=False):
        super().render(sync_frame_time)

        if self.viewer:
            self._draw_task()
        return
    
    def _draw_task(self):

        object_pos = self.root_state_tensor[self.object_indices, 0:3]
        root_pos = self.rigid_body_states[...,0,0:3]
        
        cols = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)

        self.gym.clear_lines(self.viewer)

        starts = root_pos
        ends = object_pos

        verts = torch.cat([starts, ends], dim=-1).cpu().numpy()

        for i, env_ptr in enumerate(self.envs):
            curr_verts = verts[i]
            curr_verts = curr_verts.reshape([1, 6])
            self.gym.add_lines(self.viewer, env_ptr, curr_verts.shape[0], curr_verts, cols)

        return
#####################################################################
###=========================jit functions=========================###
#####################################################################

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
    obs = torch.cat((root_h_obs, root_rot_obs, local_root_vel, local_root_ang_vel, dof_obs, dof_vel, flat_local_key_pos), dim=-1)
    return obs


@torch.jit.script
def compute_location_reward(root_pos, prev_root_pos, root_rot, tar_pos, tar_speed, dt):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float) -> Tensor
    dist_threshold = 0.5

    pos_err_scale = 0.5
    vel_err_scale = 4.0

    pos_reward_w = 0.5
    vel_reward_w = 0.4
    face_reward_w = 0.1
    
    pos_diff = tar_pos - root_pos[..., 0:2]
    pos_err = torch.sum(pos_diff * pos_diff, dim=-1)
    pos_reward = torch.exp(-pos_err_scale * pos_err)

    tar_dir = tar_pos - root_pos[..., 0:2]
    tar_dir = torch.nn.functional.normalize(tar_dir, dim=-1)
    
    
    delta_root_pos = root_pos - prev_root_pos
    root_vel = delta_root_pos / dt
    tar_dir_speed = torch.sum(tar_dir * root_vel[..., :2], dim=-1)
    tar_vel_err = tar_speed - tar_dir_speed
    tar_vel_err = torch.clamp_min(tar_vel_err, 0.0)
    vel_reward = torch.exp(-vel_err_scale * (tar_vel_err * tar_vel_err))
    speed_mask = tar_dir_speed <= 0
    vel_reward[speed_mask] = 0


    heading_rot = torch_utils.calc_heading_quat(root_rot)
    facing_dir = torch.zeros_like(root_pos)
    facing_dir[..., 0] = 1.0
    facing_dir = quat_rotate(heading_rot, facing_dir)
    facing_err = torch.sum(tar_dir * facing_dir[..., 0:2], dim=-1)
    facing_reward = torch.clamp_min(facing_err, 0.0)


    dist_mask = pos_err < dist_threshold
    # facing_reward[dist_mask] = 1.0
    vel_reward[dist_mask] = 1.0
    pos_reward[dist_mask] = 1.0

    reward = pos_reward_w * pos_reward + vel_reward_w * vel_reward + face_reward_w * facing_reward

    return reward

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

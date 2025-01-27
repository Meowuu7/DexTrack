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
from datetime import datetime


from attr import has
import isaacgym
import torch
import xml.etree.ElementTree as ET
import os
import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.utils import to_absolute_path
from leapsim.utils.reformat import omegaconf_to_dict, print_dict
from leapsim.utils.utils import set_np_formatting, set_seed, get_current_commit_hash
from leapsim.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, get_rlgames_env_creator
from rl_games.common import env_configurations, vecenv
from rl_games.torch_runner import Runner, _override_sigma, _restore
from rl_games.algos_torch import model_builder
from leapsim.learning import amp_continuous
from leapsim.learning import amp_players
from leapsim.learning import amp_models
from leapsim.learning import amp_network_builder
import numpy as np
from gym import spaces
import matplotlib.pyplot as plt
from collections import deque
import math
import random
from scipy.spatial.transform import Rotation as R

# noinspection PyUnresolvedReferences
import isaacgym
import sys
sys.path.append('.')
sys.path.append('..')

import hydra
# from isaacgymenvs.learning import calm_agent, calm_models, calm_network_builder, calm_players
# from isaacgymenvs.learning import encamp_network_builder, encamp_agent
from isaacgymenvs.utils.rlgames_utils import multi_gpu_get_rank

from isaacgymenvs.pbt.pbt import PbtAlgoObserver, initial_pbt_check
from omegaconf import DictConfig, OmegaConf
from hydra.utils import to_absolute_path
from isaacgymenvs.tasks import isaacgym_task_map
import gym

from isaacgymenvs.utils.reformat import omegaconf_to_dict, print_dict
from isaacgymenvs.utils.utils import set_np_formatting, set_seed

# from isaacgymenvs.

import torch
import numpy as np
import random

from omegaconf import open_dict

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # import numpy
    # # seed = 666
    # random.seed(seed)
    # numpy.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ['PYTHONHASHSEED'] = str(seed)
    # # torch.use_deterministic_algorithms(True, warn_only=True)


# setup_seed(666)

# 
# 
def preprocess_train_config(cfg, config_dict): # config dict #
    """
    Adding common configuration parameters to the rl_games train config.
    An alternative to this is inferring them in task-specific .yaml files, but that requires repeating the same
    variable interpolations in each config.
    """

    train_cfg = config_dict['params']['config']

    train_cfg['device'] = cfg.rl_device

    train_cfg['population_based_training'] = cfg.pbt.enabled
    train_cfg['pbt_idx'] = cfg.pbt.policy_idx if cfg.pbt.enabled else None

    train_cfg['full_experiment_name'] = cfg.get('full_experiment_name')

    print(f'Using rl_device: {cfg.rl_device}')
    print(f'Using sim_device: {cfg.sim_device}')
    print(train_cfg)

    try:
        model_size_multiplier = config_dict['params']['network']['mlp']['model_size_multiplier']
        if model_size_multiplier != 1:
            units = config_dict['params']['network']['mlp']['units']
            for i, u in enumerate(units):
                units[i] = u * model_size_multiplier
            print(f'Modified MLP units by x{model_size_multiplier} to {config_dict["params"]["network"]["mlp"]["units"]}')
    except KeyError:
        pass

    return config_dict

def copysign(a, b):
    # type: (float, Tensor) -> Tensor
    a = torch.tensor(a, device=b.device, dtype=torch.float).repeat(b.shape[0])
    return torch.abs(a) * torch.sign(b)



def get_euler_xyz(q):
    qx, qy, qz, qw = 0, 1, 2, 3
    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (q[:, qw] * q[:, qx] + q[:, qy] * q[:, qz])
    cosr_cosp = q[:, qw] * q[:, qw] - q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] + q[:, qz] * q[:, qz]
    roll = torch.atan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (q[:, qw] * q[:, qy] - q[:, qz] * q[:, qx])
    pitch = torch.where(torch.abs(sinp) >= 1, copysign(
        np.pi / 2.0, sinp), torch.asin(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (q[:, qw] * q[:, qz] + q[:, qx] * q[:, qy])
    cosy_cosp = q[:, qw] * q[:, qw] + q[:, qx] * \
        q[:, qx] - q[:, qy] * q[:, qy] - q[:, qz] * q[:, qz]
    yaw = torch.atan2(siny_cosp, cosy_cosp)

    return roll % (2*np.pi), pitch % (2*np.pi), yaw % (2*np.pi)


def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class HardwarePlayer(object):
    def __init__(self, config):
        self.config = omegaconf_to_dict(config)
        self.set_defaults()
        self.action_scale = 1 / 24
        self.actions_num = 16
        self.device = 'cuda'

        self.debug_viz = self.config["task"]['env']['enableDebugVis']

        # hand setting
        self.init_pose = self.fetch_grasp_state()
        self.get_dof_limits()
        # self.leap_dof_lower = torch.from_numpy(np.array([
        #     -1.5716, -0.4416, -1.2216, -1.3416,  1.0192,  0.0716,  0.2516, -1.3416,
        #     -1.5716, -0.4416, -1.2216, -1.3416, -1.5716, -0.4416, -1.2216, -1.3416
        # ])).to(self.device)
        # self.leap_dof_upper = torch.from_numpy(np.array([
        #     1.5584, 1.8584, 1.8584, 1.8584, 1.7408, 1.0684, 1.8584, 1.8584, 1.5584,
        #     1.8584, 1.8584, 1.8584, 1.5584, 1.8584, 1.8584, 1.8584
        # ])).to(self.device)



        # self.leap_dof_lower[4] = -0.519205
        # self.leap_dof_upper[5] = 1.96841
        # self.leap_dof_lower[5] = -0.57159
        # self.leap_dof_lower[6] = -0.25159

        if self.debug_viz:
            self.setup_plot()
            
        self.progress_buf = 0

        nn_dofs = 22
        self.prev_delta_targets = torch.zeros((nn_dofs, ), dtype=torch.float32).cuda()
        self.cur_delta_targets = torch.zeros((nn_dofs, ), dtype=torch.float32).cuda()
        
        
        self.cfg = self.config
        
        self.glb_trans_vel_scale = self.cfg["env"]["glb_trans_vel_scale"]
        self.glb_rot_vel_scale = self.cfg["env"]["glb_rot_vel_scale"] #
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        self.dt = self.cfg['sim']['dt']
        self.tight_obs = self.cfg["env"]["tight_obs"]
        
        self.wo_vel_obs = self.cfg['env'].get('wo_vel_obs', False)
        # 
        # self.shadow_hand_dof_speed_scale_list = [1.0] * 6 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        self.shadow_hand_dof_speed_scale_list = [self.glb_trans_vel_scale] * 3 + [self.glb_rot_vel_scale] * 3 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        self.shadow_hand_dof_speed_scale_tsr = torch.tensor(self.shadow_hand_dof_speed_scale_list, dtype=torch.float32).cuda()
        self.supervised_training = self.cfg['env'].get('supervised_training', False)
        self.use_inst_latent_features = self.cfg['env'].get('use_inst_latent_features', False)
        
        self.num_envs = 1
        
        #  # self.shadow_hand_dof_speed_scale_list = [1.0] * 6 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        # self.shadow_hand_dof_speed_scale_list = [self.glb_trans_vel_scale] * 3 + [self.glb_rot_vel_scale] * 3 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        # self.shadow_hand_dof_speed_scale_tsr = torch.tensor(self.shadow_hand_dof_speed_scale_list, device=self.rl_device, dtype=torch.float) # #
        
        # 
        


    def _load_single_tracking_kine_info(self, data_inst_tag, cur_base_traj_fn=None):
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        
        self.cfg = self.config
        self.test_inst_tag = self.config['env'].get('test_inst_tag', '')
        self.test_optimized_res = self.cfg['env'].get('test_optimized_res', '')
        # GRAB redced files # # from the test inst tag ot get htejgrab resudece jfiles # #tracking results -- not necesary ito load #
        leap_grab_kine_root = os.path.join(f"../assets", "GRAB_Tracking_LEAP_PK_Reduced", "data")
        grab_kine_fn = f"leap_passive_active_info_{self.test_inst_tag}.npy"
        grab_kine_fn = os.path.join(leap_grab_kine_root, grab_kine_fn)
        
        grab_kine_data = np.load(grab_kine_fn, allow_pickle=True ).item()
        cur_kine_data = grab_kine_data
        
        hand_qs = cur_kine_data['robot_delta_states_weights_np']
        # maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
        maxx_ws = hand_qs.shape[0]
        hand_qs = hand_qs[:maxx_ws]
        
        obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
        obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
        
        
        kine_obj_rot_euler_angles = []
        for i_fr in range(obj_ornt.shape[0]):
            cur_rot_quat = obj_ornt[i_fr]
            cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True)
            kine_obj_rot_euler_angles.append(cur_rot_euler)
        kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
        kine_info_dict = { # 
            # 'obj_verts': obj_verts, 
            'hand_qs': hand_qs[..., self.joint_idxes_inversed_ordering],
            # 'base_traj_hand_qs': base_traj_hand_qs, # 
            'obj_trans': obj_trans,
            'obj_ornt': obj_ornt ,
            'obj_rot_euler': kine_obj_rot_euler_angles
        }
        
        self.hand_qs = torch.from_numpy(hand_qs).float().cuda()
        self.obj_trans = torch.from_numpy(obj_trans).float().cuda()
        self.obj_ornt = torch.from_numpy(obj_ornt).float().cuda()
        self.obj_rot_euler = torch.from_numpy(kine_obj_rot_euler_angles).float().cuda()
        
        
        return kine_info_dict
    
    def _load_obj_inst_features_fn(self):
        self.cfg=  self.config
        self.object_type_to_latent_feature_fn = self.cfg['env']['object_type_to_latent_feature_fn']
        self.inst_tag_to_latent_feature_fn = self.cfg['env'].get('inst_tag_to_latent_feature_fn', '')
        
        self.object_type_to_latent_feature = np.load(self.object_type_to_latent_feature_fn, allow_pickle=True).item()
        self.inst_tag_to_latent_features = np.load(self.inst_tag_to_latent_feature_fn, allow_pickle=True).item() # 
    
    
        self.env_object_latent_feat = self.object_type_to_latent_feature[self.test_inst_tag]
        self.env_inst_latent_feat = self.inst_tag_to_latent_features[self.test_inst_tag]
        self.env_object_latent_feat = torch.from_numpy(self.env_object_latent_feat).float().cuda().unsqueeze(0)
        self.env_inst_latent_feat = torch.from_numpy(self.env_inst_latent_feat).float().cuda().unsqueeze(0)
    
    
    def _set_some_paras(self, ):
        self.num_fingertips = 4
        
    

    def real_to_sim(self, values):
        if not hasattr(self, "real_to_sim_indices"):
            self.construct_sim_to_real_transformation()

        return values[:, self.real_to_sim_indices]

    def sim_to_real(self, values):
        if not hasattr(self, "sim_to_real_indices"):
            self.construct_sim_to_real_transformation()
        
        return values[:, self.sim_to_real_indices]

    def construct_sim_to_real_transformation(self):
        self.sim_to_real_indices = self.config["task"]["env"]["sim_to_real_indices"]
        self.real_to_sim_indices= self.config["task"]["env"]["real_to_sim_indices"]

    def get_dof_limits(self):
        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../')
        hand_asset_file = self.config['task']['env']['asset']['handAsset']

        tree = ET.parse(os.path.join(asset_root, hand_asset_file))
        root = tree.getroot()

        self.leap_dof_lower = [0 for _ in range(16)]
        self.leap_dof_upper = [0 for _ in range(16)]

        for child in root.getchildren():
            if child.tag == "joint":
                joint_idx = int(child.attrib['name'])

                for gchild in child.getchildren():
                    if gchild.tag == "limit":
                        lower = float(gchild.attrib['lower'])
                        upper = float(gchild.attrib['upper'])

                        self.leap_dof_lower[joint_idx] = lower
                        self.leap_dof_upper[joint_idx] = upper

        self.leap_dof_lower = torch.tensor(self.leap_dof_lower).to(self.device)[None, :] 
        self.leap_dof_upper = torch.tensor(self.leap_dof_upper).to(self.device)[None, :] 

        self.leap_dof_lower = self.real_to_sim(self.leap_dof_lower).squeeze()
        self.leap_dof_upper = self.real_to_sim(self.leap_dof_upper).squeeze()

    def plot_callback(self):
        self.fig.canvas.restore_region(self.bg)

        # self.ydata.append(self.object_rpy[0, 2].item())
        self.ydata.append(self.cur_obs_joint_angles[0, 9].item())
        self.ydata2.append(self.cur_obs_joint_angles[0, 4].item())

        self.ln.set_ydata(list(self.ydata))
        self.ln.set_xdata(range(len(self.ydata)))

        self.ln2.set_ydata(list(self.ydata2))
        self.ln2.set_xdata(range(len(self.ydata2)))

        self.ax.draw_artist(self.ln)
        self.ax.draw_artist(self.ln2)
        self.fig.canvas.blit(self.fig.bbox)
        self.fig.canvas.flush_events()
    
    def setup_plot(self):   
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(-1, 1)
        self.ydata = deque(maxlen=100) # Plot 5 seconds of data 
        self.ydata2 = deque(maxlen=100)
        (self.ln,) = self.ax.plot(range(len(self.ydata)), list(self.ydata), animated=True)
        (self.ln2,) = self.ax.plot(range(len(self.ydata2)), list(self.ydata2), animated=True)
        plt.show(block=False)
        plt.pause(0.1)

        self.bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self.ax.draw_artist(self.ln)
        self.fig.canvas.blit(self.fig.bbox)

    def set_defaults(self):
        if "include_history" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_history"] = True

        if "include_targets" not in self.config["task"]["env"]:
            self.config["task"]["env"]["include_targets"] = True

    def fetch_grasp_state(self, s=1.0):
        self.grasp_cache_name = self.config['task']['env']['grasp_cache_name']
        grasping_states = np.load(f'cache/{self.grasp_cache_name}_grasp_50k_s{str(s).replace(".", "")}.npy')

        if "sampled_pose_idx" in self.config["task"]["env"]:
            idx = self.config["task"]["env"]["sampled_pose_idx"]
        else:
            idx = random.randint(0, grasping_states.shape[0] - 1)

        return grasping_states[idx][:16] # first 16 are hand dofs, last 16 is object state
    
    
    def compute_full_state(self, asymm_obs=False): #
        # hand dof pos, hand dof velocities, fingertip states, right hand pos, right hand rot, current actions, object states, next qpos ref, current delta targets
        # self.get_unpose_quat()

        
        # 2 * nn_hand_dofs + 13 * num_fingertips + 6 + nn_hand_dofs + 16 + 7 + nn_hand_dofs ## 
        # unscale to (-1，1) # 
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##
        
        # if self.use_local_canonical_state:
        # 
        # print(f"using local canonicalizations")
        canon_shadow_hand_dof_trans = self.shadow_hand_dof_pos[..., :3] - self.object_pos[..., :]
        canon_shadow_hand_dof = torch.cat(
            [ canon_shadow_hand_dof_trans, self.shadow_hand_dof_pos[..., 3:] ], dim=-1 
        )
        # else:
        #     canon_shadow_hand_dof = self.shadow_hand_dof_pos 
        
        
        self.cur_dof_vel[:, : ] = self.shadow_hand_dof_vel

        # # 0:66
        # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
        #                                                        self.shadow_hand_dof_lower_limits,
        #                                                        self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(canon_shadow_hand_dof,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        if self.wo_vel_obs:
            self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = 0.0
        else:
            self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        
        
        
        # if self.obs_type == "full_state" or asymm_obs:
        #     self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
        
        #     fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        # else:
        fingertip_obs_start = 2 * self.num_shadow_hand_dofs
        
        
        # finger tip state # 
        # if self.use_local_canonical_state:
        canon_fingertip_pos = self.fingertip_pos - self.object_pos.unsqueeze(1)
        canon_fingertip_pose = torch.cat(
            [ canon_fingertip_pos, self.fingertip_state[..., 3:] ], dim=-1
        )
        # else:
        #     canon_fingertip_pose = self.fingertip_state
        
        # aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
        aux = canon_fingertip_pose.reshape(self.num_envs, num_ft_states)
        
        for i in range(self.num_fingertips):
            aux[:, i * 13:(i + 1) * 13] = aux[:, i * 13:(i + 1) * 13]
        # 66:131: ft states
        self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

        # 131:161: ft sensors: do not need repose
        # if self.obs_type == "full_state" or asymm_obs:
        # #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.force_sensor_tensor[:, :num_ft_force_torques] # full state or asymm_obs #
        # # else
        #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :num_ft_force_torques]

        #     hand_pose_start = fingertip_obs_start + num_ft_states + num_ft_force_torques #  95
        # else:
        hand_pose_start = fingertip_obs_start + num_ft_states
        # 161:167: hand_pose
        ### Global hand pose ###
        
        
        # if self.use_local_canonical_state:
        canon_right_hand_pos = self.right_hand_pos - self.object_pos
        # else:
        #     canon_right_hand_pos = self.right_hand_pos
        
        if self.tight_obs:
            # right_hand_rot

            # self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
            self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = (canon_right_hand_pos)
            euler_xyz = get_euler_xyz((self.right_hand_rot))
        else:
            # self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
            self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = (canon_right_hand_pos)
            euler_xyz = get_euler_xyz((self.hand_orientations[self.hand_indices, :]))
            
        self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
        self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)
            
        # Actions #
        action_obs_start = hand_pose_start + 6
        # 167:191: action #
        try:
            aux = self.actions[:, :self.num_shadow_hand_dofs]
        except: # using the
            aux = torch.zeros((self.num_envs, self.num_shadow_hand_dofs), dtype=torch.float32, device=self.device)
        aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
        aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
        self.obs_buf[:, action_obs_start:action_obs_start + self.num_shadow_hand_dofs] = aux

        # object pos and object pose ? #
        if self.use_local_canonical_state:
            canon_object_pos = self.object_pos - self.object_pos
        else:
            canon_object_pos = self.object_pos  

        obj_obs_start = action_obs_start + self.num_shadow_hand_dofs  # 144
        # 191:207 object_pose, goal_pos
        # self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = (canon_object_pos)
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = (self.object_pose[:, 3:7])
        if self.wo_vel_obs:
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = 0.0
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = 0.0
        else:
            self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = (self.object_linvel)
            self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * (self.object_angvel)
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = (self.goal_pos - self.object_pos)

        # + 6 + nn_dof (action) + 16 (obj) + 7 + nn_dof (goal) + 64
         # 207:236 goal # obj obs start # 
        hand_goal_start = obj_obs_start + 16
        
        if self.tight_obs:
            self.obs_buf[:, hand_goal_start: hand_goal_start +  self.num_shadow_hand_dofs] = self.delta_qpos
        else:
            self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
            self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
            # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos
            self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.delta_qpos

            hand_goal_start = hand_goal_start + 7

        
        # prue state wref # # add the delta targets # # target object pos #
        # if self.obs_type == 'pure_state_wref_wdelta' and self.use_kinematics_bias_wdelta:
            
        # tot_goal_hand_qs_th = self.tot_kine_qs
        # tot_goal_hand_qs_th = self.tot_hand_preopt_res
        # envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
        # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #

        
        nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
        
        if self.use_local_canonical_state:
            canon_hand_qpos_trans = self.nex_hand_qpos_ref[..., :3] - self.object_pos
            canon_hand_qpos_ref = torch.cat(
                [ canon_hand_qpos_trans, self.nex_hand_qpos_ref[..., 3:] ], dim=-1
            )
        else:
            canon_hand_qpos_ref = self.nex_hand_qpos_ref
        
        # unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        unscaled_nex_hand_qpos_ref = unscale(canon_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        
        # cur_hand_qpos_ref # the nex 
        # unscaled_nex_hand_qpos_ref = unscale(cur_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        
        # unscaled_nex_hand_qpos_ref = cur_hand_qpos_ref
        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
        
        cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
        self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs]
        
        obj_feat_st_idx = cur_delta_start + self.num_shadow_hand_dofs
            
        # elif self.obs_type == 'pure_state_wref': # pure stsate with ref 
        #     nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
        #     unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
        #     self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
            
        #     obj_feat_st_idx = nex_ref_start + self.num_shadow_hand_dofs
            
        # elif not self.obs_type == 'pure_state':
            
        #     # 236: visual feature 
        #     visual_feat_start = hand_goal_start + self.num_shadow_hand_dofs #  29
            
        #     # 236: 300: visual feature #
        #     self.obs_buf[:, visual_feat_start: visual_feat_start + 64] = 0.1 * self.visual_feat_buf
        #     self.obs_buf[:, visual_feat_start + 64: 300] = 0.0
            
        #     obj_feat_st_idx = 300
        
        
        # 
        # if not self.single_instance_state_based_test and not self.single_instance_state_based_train:
        ### add the obj latent features ###
        ### add the env obj latent features ###
        self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_object_latent_feat
        
        if self.use_inst_latent_features:
            obj_feat_st_idx = obj_feat_st_idx + self.object_feat_dim
            self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_inst_latent_feat
        
        # if self.supervised_training:
        #     # TODO: add expected actions here #
        #     nex_hand_qtars_st_idx = obj_feat_st_idx + self.object_feat_dim
        #     env_max_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) - 1
        #     # nn_envs,
        #     nex_progress_buf = torch.clamp(self.progress_buf + 1, min=torch.zeros_like(env_max_episode_length), max=env_max_episode_length)
        #     # env_hand_qtars = batched_index_select(self.env_hand_qs, self.env_inst_idxes, dim=0)
        #     maxx_env_idxes  = torch.max(self.env_inst_idxes).item()
        #     minn_env_idxes = torch.min(self.env_inst_idxes).item()
        #     # print(f"maxx_env_idxes: {maxx_env_idxes}, minn_env_idxes: {minn_env_idxes}, tot_hand_qtars: {self.tot_hand_qtars.size()}, tot_kine_qs: {self.tot_kine_qs.size()}")
            
        #     env_hand_qtars = batched_index_select(self.tot_hand_qtars, self.env_inst_idxes, dim=0)
        #     nex_env_hand_qtars = batched_index_select(env_hand_qtars, nex_progress_buf.unsqueeze(1), dim=1)
            
        #     nex_env_hand_qtars = nex_env_hand_qtars.squeeze(1)
            
            
        #     tot_envs_hand_qs = batched_index_select(self.tot_kine_qs, self.env_inst_idxes, dim=0) # nn_envs x 
        #     # envs_maxx_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        #     increased_progress_buf = nex_progress_buf
        #     ctl_kinematics_bias = batched_index_select(tot_envs_hand_qs, increased_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs # 
        #     ctl_kinematics_bias = ctl_kinematics_bias.squeeze(1)
            
            
        #     nex_delta_actions = nex_env_hand_qtars - ctl_kinematics_bias
        #     # print(f"nex_delta_actions: {nex_delta_actions.size()}, ")
        #     # print(f"cur_delta_targets: {self.cur_delta_targets.size()}, self.actuated_dof_indices: {self.actuated_dof_indices}")
        #     nex_delta_delta_actions = nex_delta_actions - self.cur_delta_targets[:, self.actuated_dof_indices]
        #     # print(f"nex_delta_delta_actions: {nex_delta_delta_actions.size()}, shadow_hand_dof_speed_scale_tsr: {self.shadow_hand_dof_speed_scale_tsr.size()}")
        #     # shadow hand dof speed sacle tsr #
        #     nex_actions = (nex_delta_delta_actions / self.dt) / self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0)
            
            
        #     if self.tot_hand_actions is not None:
        #         env_hand_actions = batched_index_select(self.tot_hand_actions, self.env_inst_idxes, dim=0)
        #         nex_env_hand_actions = batched_index_select(env_hand_actions, nex_progress_buf.unsqueeze(1), dim=1)
        #         nex_env_hand_actions = nex_env_hand_actions.squeeze(1)
        #         nex_actions = nex_env_hand_actions
            
        #     # # prev_detlat_targets # # prev delta targets #
        #     # delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
        #     # cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
        #     # self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
        #     # self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
            
        # self.obs_buf[:, nex_hand_qtars_st_idx: nex_hand_qtars_st_idx + self.num_actions] = nex_actions 
            
            # if self.grab_obj_type_to_opt_res is not None:
            #     self.obs_buf[:, nex_hand_qtars_st_idx + self.num_actions: nex_hand_qtars_st_idx + self.num_actions + 1] = self.env_rew_succ_list.unsqueeze(1)
            
            # # unscale(nex_env_hand_tars, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            # pass
        
        return

    
    def compute_observations(self):
        # compute observations #
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        # object pose # # root state tensor # # root state tensort #
        # shadow_hand_dof_pos # 
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        #
        self.object_handle_pos = self.object_pos  ## + quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos # + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        # object linvel # object angvel # object linvel #
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]



        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        idx = self.hand_body_idx_dict['index']
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                              
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        # idx = self.hand_body_idx_dict['little']
        # self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        # self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        # # self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3] # goal state #
        self.goal_rot = self.goal_states[:, 3:7]
        
        self.goal_lifting_pos = self.goal_states[:, 0:3] 
        
        
        if self.dataset_type == 'taco':
            progress_buf_indexes = torch.where(self.progress_buf >= self.hand_palm_world_poses.size(0), self.hand_palm_world_poses.size(0) - 1 + torch.zeros_like(self.progress_buf), self.progress_buf)
            # self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos #
            self.gt_hand_palm_pos = self.hand_palm_world_poses[progress_buf_indexes]
            self.gt_hand_thumb_pos = self.thumb_tip_world_poses[progress_buf_indexes]
            self.gt_hand_index_pos = self.index_tip_world_poses[progress_buf_indexes]
            self.gt_hand_middle_pos = self.middle_tip_world_poses[progress_buf_indexes]
            self.gt_hand_ring_pos = self.ring_tip_world_poses[progress_buf_indexes]
            
            
            
            
        # get fingertip states #
            
            
            
            
            
            
            
            
            
            
        
        # maxx_progress = torch.max(self.progress_buf)
        # minn_progress = torch.min(self.progress_buf)
        # print(f"maxx_progress: {maxx_progress}, minn_progress: {minn_progress}, goal_obj_trans_th: {self.goal_obj_trans_th.size()}")
        
        # goal obj trans # 
        
        # 
        # env_inst_idxes #
        
        # tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        # tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
        
        tot_goal_obj_trans_th = self.obj_trans
        tot_goal_obj_ornt_th = self.obj_ornt
        
        # values, indices, dims #
        # cur_dof_vel #

        # envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        # envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        # envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) # nn_envs x 1
        # # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}")
        # cur_progress_buf = torch.clamp(self.progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        # # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        
        cur_goal_pos = tot_goal_obj_trans_th[self.progress_buf]
        cur_goal_rot = tot_goal_obj_ornt_th[self.progress_buf]
        
        # cur_goal_pos = self.goal_obj_trans_th[self.progress_buf]
        # cur_goal_rot = self.goal_obj_rot_quat_th[self.progress_buf]
        
        
        self.goal_pos_ref = cur_goal_pos
        self.goal_rot_ref = cur_goal_rot
        self.goal_pose_ref = torch.cat(
            [self.goal_pos_ref, self.goal_rot_ref], dim=-1
        )
        
        # if self.separate_stages:
        #     obj_lowest_z_less_than_thres = (self.object_pos[:, 2] < 0.19)
        #     self.goal_pos = torch.where(
        #         obj_lowest_z_less_than_thres.unsqueeze(-1).repeat(1, 3), self.goal_pos, self.goal_pos_ref
        #     )
        #     self.goal_rot = torch.where(
        #         obj_lowest_z_less_than_thres.unsqueeze(-1).repeat(1, 4), self.goal_rot, self.goal_rot_ref
        #     )
        #     self.goal_pose = torch.cat(
        #         [self.goal_pos, self.goal_rot], dim=-1
        #     )
        # else:
        self.goal_pose = self.goal_pose_ref
        self.goal_rot = self.goal_rot_ref
        self.goal_pos  = self.goal_pos_ref
    
        
        #######################3
        # if self.use_twostage_rew:
        #     grasping_frame_obj_pos = self.goal_obj_trans_th[self.cur_grasp_fr] + to_torch([0.0, 0.0, 0.1], device=self.device)
        #     grasping_frame_obj_ornt = self.goal_obj_rot_quat_th[self.cur_grasp_fr]
        #     expanded_grasping_frame_obj_pos = grasping_frame_obj_pos.unsqueeze(0).repeat(self.num_envs, 1)
        #     expanded_grasping_frame_obj_ornt = grasping_frame_obj_ornt.unsqueeze(0).repeat( self.num_envs, 1)
        #     grasp_manip_stages_flag_pos = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, 3)
        #     grasp_manip_stages_flag_rot = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, 4)
            
        #     if self.use_real_twostage_rew:
        #         self.goal_pos = torch.where(
        #             grasp_manip_stages_flag_pos, expanded_grasping_frame_obj_pos, self.goal_pos
        #         )
        #         self.goal_rot = torch.where(
        #             grasp_manip_stages_flag_rot, expanded_grasping_frame_obj_ornt, self.goal_rot
        #         )   
        #     else:
        #         self.goal_pos = expanded_grasping_frame_obj_pos
        #         self.goal_rot = expanded_grasping_frame_obj_ornt
        #     self.goal_pose = torch.cat(
        #         [self.goal_pos, self.goal_rot], dim=-1
        #     )
        
        #######################
        # if self.lifting_separate_stages:
        #     lifting_frame_obj_pos = self.goal_obj_trans_th[self.lift_fr]
        #     lifting_frame_obj_ornt = self.goal_obj_rot_quat_th[self.lift_fr]
        #     expanded_lifting_frame_obj_pos = lifting_frame_obj_pos.unsqueeze(0).repeat(self.num_envs, 1)
        #     expanded_lifting_frame_obj_ornt = lifting_frame_obj_ornt.unsqueeze(0).repeat( self.num_envs, 1)
        #     lifting_manip_stages_flag_pos = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, 3)
        #     lifting_manip_stages_flag_rot = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, 4)
        #     self.goal_pos = torch.where(
        #         lifting_manip_stages_flag_pos, expanded_lifting_frame_obj_pos, self.goal_pos
        #     )
        #     self.goal_rot = torch.where(
        #         lifting_manip_stages_flag_rot, expanded_lifting_frame_obj_ornt, self.goal_rot
        #     )
        #     # # # #
        #     self.goal_pose = torch.cat(
        #         [self.goal_pos, self.goal_rot], dim=-1
        #     )
            
            
        
        ## TODO: goal velocities may be noisy; so we do not add it ##
        # goal_linvel, goal_angvel # # goal angvel #
        # cur_goal_lin_vels = self.goal_obj_lin_vels_th[self.progress_buf]
        # cur_goal_ang_vels = self.goal_obj_ang_vels_th[self.progress_buf]
        # self.goal_linvel = cur_goal_lin_vels
        # self.goal_angvel = cur_goal_ang_vels
        # self.goal_vels = torch.cat(
        #     [self.goal_linvel, self.goal_angvel], dim=-1
        # )

        # fingertip state #
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        # def world2obj_vec(vec):
        #     return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        # def obj2world_vec(vec):
        #     return quat_apply(self.object_rot, vec) + self.object_pos
        # def world2obj_quat(quat):
        #     return quat_mul(quat_conjugate(self.object_rot), quat)
        # def obj2world_quat(quat):
            # return quat_mul(self.object_rot, quat)

        # ### HACK : original implementation ####
        # self.delta_target_hand_pos = world2obj_vec(self.right_hand_pos) - self.target_hand_pos
        # self.rel_hand_rot = world2obj_quat(self.right_hand_rot)
        # self.delta_target_hand_rot = quat_mul(self.rel_hand_rot, quat_conjugate(self.target_hand_rot))
        # ### HACK : original implementation ####
        
        ### HACK ###
        self.delta_target_hand_pos = torch.zeros((3,), dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1).cuda()
        self.rel_hand_rot = torch.zeros((4,), dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1).cuda()
        self.delta_target_hand_rot = torch.zeros((4,), dtype=torch.float32).unsqueeze(0).repeat(self.num_envs, 1).cuda()
        ### HACK ###
        
        
        # tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        # tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
        
        # # values, indices, dims #
        # envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        # envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
    
        # # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        
        # cur_goal_pos = self.
        
        cur_goal_pos = tot_goal_obj_trans_th[self.progress_buf]
        cur_goal_rot = tot_goal_obj_ornt_th[self.progress_buf]
        cur_hand_qpos_ref  = self.hand_qs[self.progress_buf]
        
        
        # # tot_goal_hand_qs_th = self.tot_kine_qs
        # tot_goal_hand_qs_th = self.tot_hand_preopt_res
        # envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
        # # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #

        ### current target hand pose, and the difference from the reference hand pos ###
        # cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        
        
        self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        
        
        ### next progress buffer ###
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.hand_qs_th.size(0) - 1)
        # nex_hand_qpos_ref = self.hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        # self.nex_hand_qpos_ref = nex_hand_qpos_ref
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.maxx_kine_nn_ts - 1)
        
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, min=torch.zeros_like(envs_maxx_episode_length_per_traj), max=envs_maxx_episode_length_per_traj)
        
        nex_progress_buf = min(self.progress_buf + 1, self.hand_qs.size(0) - 1)
        # print(f"nex_progress_buf: {torch.max(nex_progress_buf)}, envs_goal_hand_qs: {envs_goal_hand_qs.size()}")
        # nex_hand_qpos_ref = self.goal_hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        # nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        nex_hand_qpos_ref = self.hand_qs[nex_progress_buf]
        self.nex_hand_qpos_ref = nex_hand_qpos_ref
        
        # # 
        # if self.use_twostage_rew: # two stage reward #
        #     # grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
        #     grasp_frame_hand_qpos = self.goal_hand_qs_th[self.cur_grasp_fr]
        #     # expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        #     # self.shadow_hand_dof_pos = torch.where(
        #     #     self.grasp_manip_stages == 0, expanded_grasp_frame_hand_qpos, self.shadow_hand_dof_pos
        #     # )
        #     self.grasp_frame_hand_qpos = grasp_frame_hand_qpos # 
        #     expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        #     grasp_manip_stages_flag_qpos = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
            
        #     if self.use_real_twostage_rew:
        #         self.nex_hand_qpos_ref = torch.where(
        #             grasp_manip_stages_flag_qpos, expanded_grasp_frame_hand_qpos, self.nex_hand_qpos_ref
        #         )
        #         cur_hand_qpos_ref = torch.where(
        #             grasp_manip_stages_flag_qpos, expanded_grasp_frame_hand_qpos, cur_hand_qpos_ref
        #         )
        #     else:
        #         self.nex_hand_qpos_ref = expanded_grasp_frame_hand_qpos
        #         cur_hand_qpos_ref = expanded_grasp_frame_hand_qpos

        #     self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
            
        # if self.lifting_separate_stages:
        #     # lifting_frame_hand_qpos = self.hand_qs_th[self.lift_fr]
        #     lifting_frame_hand_qpos = self.goal_hand_qs_th[self.lift_fr]
        #     expanded_lifting_frame_hand_qpos = lifting_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        #     lifting_manip_stages_flag_qpos = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
        #     self.nex_hand_qpos_ref = torch.where(
        #         lifting_manip_stages_flag_qpos, expanded_lifting_frame_hand_qpos, self.nex_hand_qpos_ref
        #     )
        #     cur_hand_qpos_ref = torch.where(
        #         lifting_manip_stages_flag_qpos, expanded_lifting_frame_hand_qpos, cur_hand_qpos_ref
        #     )
            # self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        
        
        # if self.test:
        #     # object pose np ## -- curretn step observations; # # 
        #     self.object_pose_np = self.object_pose.detach().cpu().numpy()
        #     self.shadow_hand_dof_pos_np = self.shadow_hand_dof_pos.detach().cpu().numpy()
        #     self.target_qpos_np = self.cur_targets.detach().cpu().numpy()[:, : self.num_shadow_hand_dofs]
        #     self.shadow_hand_dof_vel_np = self.shadow_hand_dof_vel.detach().cpu().numpy() 
        #     self.object_linvel_np = self.object_linvel.detach().cpu().numpy()
        #     self.object_angvel_np = self.object_angvel.detach().cpu().numpy()
        #     self.obs_buf_np = self.obs_buf.detach().cpu().numpy()[:100] ## get the observation buffers ##
        #     if self.ref_ts > 0:
        #         self.actions_np = self.actions.detach().cpu().numpy()[:100]
        #     else:
        #         self.actions_np = np.zeros((self.num_envs, self.num_shadow_hand_dofs), dtype=np.float32)[:100]
        #     # so that the obs include all obs buffer ? #
        #     # save observations, actions, angvel, linvel and other things #
        #     # then load hand dof pos and dof tars; dof tars #
        #     self.ts_to_hand_obj_states[self.ref_ts] = {
        #         'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np,
        #         'shadow_hand_dof_tars': self.target_qpos_np,
        #         'object_pose': self.object_pose_np,
        #         'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np,
        #         'object_linvel': self.object_linvel_np,
        #         'object_angvel': self.object_angvel_np,
        #         'actions': self.actions_np , 
        #         'observations': self.obs_buf_np
        #         # actions and the hand obs #
        #     }
        #     # self.ts_to_hand_obj_states[self.ref_ts]
        
        
        # self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos
        self.compute_full_state()

        if self.asymmetric_obs: 
            self.compute_full_state(True)

    
    

    def deploy(self):
        import rospy
        from hardware_controller import LeapHand
        
        # try to set up rospy
        num_obs = self.config['task']['env']['numObservations'] 
        num_obs_single = num_obs // 3 
        rospy.init_node('example')
        leap = LeapHand()
        leap.leap_dof_lower = self.leap_dof_lower.cpu().numpy()
        leap.leap_dof_upper = self.leap_dof_upper.cpu().numpy()
        leap.sim_to_real_indices = self.sim_to_real_indices
        leap.real_to_sim_indices = self.real_to_sim_indices
        # Wait for connections.
        rospy.wait_for_service('/leap_position')

        hz = 20
        self.control_dt = 1 / hz
        ros_rate = rospy.Rate(hz)

        print("command to the initial position")
        for _ in range(hz * 4):
        #     print("self.init_pose.shape = ",self.init_pose.shape)
        #     print("self.init_pose = ",self.init_pose)
        #     self.init_pose = np.array([-0.3073,  0.1722,  1.8214, -0.3657,  1.8289, -0.4562,  1.6719, -0.7970,
        # -0.2401, -0.8232,  1.5012, -0.1182,  0.7753, -0.7858,  0.5857,  1.1246])
        #     print("self.init_pose.shape after = ",self.init_pose.shape)
        #     print("self.init_pose after = ",self.init_pose)
            leap.command_joint_position(self.init_pose)
            # self.init_pose[:] = 0
            # self.init_pose[12] = 0.8
            print("self.init_pose = ",self.init_pose)
            obses, _ = leap.poll_joint_position()
            print("obses.shape = ",obses.shape)
            print("obses = ",obses)
            ros_rate.sleep()
        print("done")
        # 使用计时器代替 time.sleep
        start_time = rospy.get_time()
        while rospy.get_time() - start_time < 100:  # 100秒
            rospy.sleep(1)  # 让线程有机会响应中断

        obses, _ = leap.poll_joint_position()

        # hardware deployment buffer
        obs_buf = torch.from_numpy(np.zeros((1, 0)).astype(np.float32)).cuda()

        def unscale(x, lower, upper):
            return (2.0 * x - upper - lower) / (upper - lower)

        obses = torch.from_numpy(obses.astype(np.float32)).cuda()
        prev_target = obses[None].clone()
        cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]

        if self.config["task"]["env"]["include_history"]:
            num_append_iters = 3
        else:
            num_append_iters = 1

        for i in range(num_append_iters):   
            obs_buf = torch.cat([obs_buf, cur_obs_buf.clone()], dim=-1)
            
            if self.config["task"]["env"]["include_targets"]:
                obs_buf = torch.cat([obs_buf, prev_target.clone()], dim=-1)

            if "phase_period" in self.config["task"]["env"]:
                phase = torch.tensor([[0., 1.]], device=self.device)
                obs_buf = torch.cat([obs_buf, phase], dim=-1)

        if "obs_mask" in self.config["task"]["env"]:
            obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"]).cuda()[None, :]

        obs_buf = obs_buf.float()

        counter = 0 

        if "debug" in self.config["task"]["env"]:
            self.obs_list = []
            self.target_list = []

            if "record" in self.config["task"]["env"]["debug"]:
                self.record_duration = int(self.config["task"]["env"]["debug"]["record"]["duration"] / self.control_dt)

            if "actions_file" in self.config["task"]["env"]["debug"]:
                self.actions_list = torch.from_numpy(np.load(self.config["task"]["env"]["debug"]["actions_file"])).cuda()        
                self.record_duration = self.actions_list.shape[0]

        if self.player.is_rnn:
            print("is_rnn !!!!!!!!")
            self.player.init_rnn()

        while True:
            counter += 1
            # obs = self.running_mean_std(obs_buf.clone()) # ! Need to check if this is implemented
            
            if hasattr(self, "actions_list"):
                action = self.actions_list[counter-1][None, :]
            else:
                action = self.forward_network(obs_buf)
                
                
            
            # self.actions = action.clone().to(self.device)


            self.actions = action.clone()
            
            if len(self.actions.size()) > 1:
                self.actions = self.actions.squeeze(0)

            # if self.use_kinematics_bias_wdelta:
            # print(f"self.use_kinematics_bias_wdelta: {self.use_kinematics_bias_wdelta}")
            increased_progress_buf = self.progress_buf + 1
            
            # two instances? #
            # increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
            # get the kinematicsof the increaesd progres buf as the kinematics bias # 
            # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
            # ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]

            # tot_envs_hand_qs = self.tot_kine_qs
            # ### ### # # tot envs qs #
            
            tot_envs_hand_qs = self.tot_hand_preopt_res
            
            increased_progress_buf = min(increased_progress_buf, self.hand_qs.size(0) - 1)
            # obs_buf ##
            
            # maxx_env_inst_idx = torch.max(self.env_inst_idxes).item()
            # minn_env_inst_idx = torch.min(self.env_inst_idxes).item()
            # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_env_inst_idx: {maxx_env_inst_idx}, minn_env_inst_idx: {minn_env_inst_idx}")
            
            # tot_envs_hand_qs = batched_index_select(tot_envs_hand_qs, self.env_inst_idxes, dim=0) # nn_envs x 
            # envs_maxx_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
            # increased_progress_buf = torch.clamp(increased_progress_buf, min=torch.zeros_like(envs_maxx_episode_length), max=envs_maxx_episode_length)
            
            
            # maxx_increased_progress_buf = torch.max(increased_progress_buf).item()
            # minn_increased_progress_buf= torch.min(increased_progress_buf).item()
            # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_increased_progress_buf: {maxx_increased_progress_buf}, minn_increased_progress_buf: {minn_increased_progress_buf}")
            
            # ctl_kinematics_bias = batched_index_select(tot_envs_hand_qs, increased_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs # 
            ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # ctl_kinematics_bias.squeeze(1) # nn_envs x nn_hand_dofs #
            
            
            # if self.use_twostage_rew:
            #     grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
            #     expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            #     grasp_manip_stages_flag = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
            #     ctl_kinematics_bias = torch.where(
            #         grasp_manip_stages_flag, expanded_grasp_frame_hand_qpos, ctl_kinematics_bias
            #     )
            
            # self.shadow_hand_dof_speed_scale_tsr
            # self.dt 
            
            # self.shadow_hand_dof_lower_limits
            # self.shadow_hand_dof_upper_limits #
            
            
            
            # prev_detlat_targets # 
            delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            cur_delta_targets = self.prev_delta_targets + delta_delta_targets
            self.cur_delta_targets = cur_delta_targets
            self.prev_delta_targets = self.cur_delta_targets # [:, self.actuated_dof_indices]
            # if self.kinematics_only: # kinematics bais #
            #     cur_targets = ctl_kinematics_bias
            # else:
            cur_targets = ctl_kinematics_bias + self.cur_delta_targets # [:, self.actuated_dof_indices]
            # self.cur_targets = tensor_clamp(cur_targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            # self.cur_targets = torch.clamp(
            #     cur_targets, min=self.shadow_hand_dof_lower_limits, max=self.shadow_hand_dof_upper_limits
            # )
                

            # action = torch.clamp(action, -1.0, 1.0)

            # if "actions_mask" in self.config["task"]["env"]:
            #     action = action * torch.tensor(self.config["task"]["env"]["actions_mask"]).cuda()[None, :]

            target = cur_targets
            # target = prev_target + self.action_scale * action 
            target = torch.clip(target, self.leap_dof_lower, self.leap_dof_upper)
            prev_target = target.clone()
        
            # interact with the hardware
            commands = target.cpu().numpy()[0]
            # print("commands = ",commands)

            if "disable_actions" not in self.config["task"]["env"]:
                leap.command_joint_position(commands)

            ros_rate.sleep()  # keep 20 Hz command
            
            # command_list.append(commands)
            # get o_{t+1}
            obses, _ = leap.poll_joint_position()
            obses = torch.from_numpy(obses.astype(np.float32)).cuda()

            # obs_buf_list.append(obses.cpu().numpy().squeeze())
            cur_obs_buf = unscale(obses, self.leap_dof_lower, self.leap_dof_upper)[None]
            self.cur_obs_joint_angles = cur_obs_buf.clone()

            if self.debug_viz:
                self.plot_callback()

            if hasattr(self, "obs_list"):
                self.obs_list.append(cur_obs_buf[0].clone())
                self.target_list.append(target[0].clone().squeeze())

                if counter == self.record_duration - 1:
                    self.obs_list = torch.stack(self.obs_list, dim=0)
                    self.obs_list = self.obs_list.cpu().numpy()

                    self.target_list = torch.stack(self.target_list, dim=0)
                    self.target_list = self.target_list.cpu().numpy()

                    if "actions_file" in self.config["task"]["env"]["debug"]:
                        actions_file = os.path.basename(self.config["task"]["env"]["debug"]["actions_file"])
                        folder = os.path.dirname(self.config["task"]["env"]["debug"]["actions_file"])
                        suffix = "_".join(actions_file.split("_")[1:])
                        joints_file = os.path.join(folder, "joints_real_{}".format(suffix)) 
                        target_file = os.path.join(folder, "targets_real_{}".format(suffix))
                    else:
                        suffix = self.config["task"]["env"]["debug"]["record"]["suffix"]
                        joints_file = "debug/joints_real_{}.npy".format(suffix)
                        target_file = "debug/targets_real_{}.npy".format(suffix)

                    np.save(joints_file, self.obs_list)
                    np.save(target_file, self.target_list) 
                    exit()

            if self.config["task"]["env"]["include_history"]:
                obs_buf = obs_buf[:, num_obs_single:].clone()
            else:
                obs_buf = torch.zeros((1, 0), device=self.device)

            obs_buf = torch.cat([obs_buf, cur_obs_buf.clone()], dim=-1)

            if self.config["task"]["env"]["include_targets"]:
                obs_buf = torch.cat([obs_buf, target.clone()], dim=-1)

            if "phase_period" in self.config["task"]["env"]:
                omega = 2 * math.pi / self.config["task"]["env"]["phase_period"]
                phase_angle = (counter - 1) * omega / hz 
                num_envs = obs_buf.shape[0]
                phase = torch.zeros((num_envs, 2), device=obs_buf.device)
                phase[:, 0] = math.sin(phase_angle)
                phase[:, 1] = math.cos(phase_angle)
                obs_buf = torch.cat([obs_buf, phase.clone()], dim=-1)

            if "obs_mask" in self.config["task"]["env"]:
                obs_buf = obs_buf * torch.tensor(self.config["task"]["env"]["obs_mask"]).cuda()[None, :]

            obs_buf = obs_buf.float()

    def forward_network(self, obs):
        return self.player.get_action(obs, True)

    def restore_bak(self):
        rlg_config_dict = self.config['train']
        rlg_config_dict["params"]["config"]["env_info"] = {}
        self.num_obs = self.config["task"]["env"]["numObservations"]
        self.num_actions = 16
        observation_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        rlg_config_dict["params"]["config"]["env_info"]["observation_space"] = observation_space
        action_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)
        rlg_config_dict["params"]["config"]["env_info"]["action_space"] = action_space
        rlg_config_dict["params"]["config"]["env_info"]["agents"] = 1

        def build_runner(algo_observer):
            runner = Runner(algo_observer)
            runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
            runner.player_factory.register_builder('amp_continuous', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            model_builder.register_model('continuous_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
            model_builder.register_network('amp', lambda **kwargs : amp_network_builder.AMPBuilder())

            return runner

        runner = build_runner(RLGPUAlgoObserver())
        runner.load(rlg_config_dict)
        runner.reset()

        args = {
            'train': False,
            'play': True,
            'checkpoint' : self.config['checkpoint'],
            'sigma' : None
        }

        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)
    

    def restore(self):

        cfg = self.config
        try:
            cseed = cfg.task.seed
        except:
            cseed = 20

        setup_seed(cseed)

        if cfg.pbt.enabled:
            initial_pbt_check(cfg)
        
        from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
        from isaacgymenvs.utils.rlgames_utils import AMPRLGPUEnv
        # from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        from rl_games.algos_torch import model_builder
        from isaacgymenvs.learning import amp_continuous
        from isaacgymenvs.learning import amp_players
        from isaacgymenvs.learning import amp_models
        from isaacgymenvs.learning import amp_network_builder
        from isaacgymenvs.learning import visionppo_models, visionppo_network_builder
        from isaacgymenvs.learning import a2c_dagger_continuous
        from isaacgymenvs.learning import a2c_supervised
        from isaacgymenvs.learning import a2c_supervised_player
        from isaacgymenvs.learning import a2c_fromsupervised
        from isaacgymenvs.learning import a2c_supervised_deterministic
        import isaacgymenvs

        with open_dict(cfg):
            cfg.task.test = cfg.test
            
            
        tag = cfg.tag
        time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"{tag}_{cfg.wandb_name}_{time_str}"

        # ensure checkpoints can be specified as relative paths
        if cfg.checkpoint:
            cfg.checkpoint = to_absolute_path(cfg.checkpoint)

        cfg_dict = omegaconf_to_dict(cfg)
        print_dict(cfg_dict)

        # set numpy formatting for printing only #
        set_np_formatting()

        # global rank of the GPU #
        global_rank = int(os.getenv("RANK", "0"))

        # sets seed. if seed is -1 will pick a random one #
        cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
        
        
        exp_dir = cfg.exp_dir
        
        def create_isaacgym_env(**kwargs):
            envs = isaacgymenvs.make(
                cfg.seed, 
                cfg.task_name, 
                cfg.task.env.numEnvs, 
                cfg.sim_device,
                cfg.rl_device,
                cfg.graphics_device_id,
                cfg.headless,
                cfg.multi_gpu,
                cfg.capture_video,
                cfg.force_render,
                cfg,
                **kwargs,
            )
            if cfg.capture_video:
                envs.is_vector_env = True
                envs = gym.wrappers.RecordVideo(
                    envs, # record video #
                    os.path.join(exp_dir, f"videos/{run_name}"),
                    step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                    video_length=cfg.capture_video_len,
                )
            return envs

        env_configurations.register('rlgpu', {
            'vecenv_type': 'RLGPU',
            'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
        })

        ige_env_cls = isaacgym_task_map[cfg.task_name] # task map # 
        dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

        if dict_cls:
            # params network#
            obs_spec = {}
            actor_net_cfg = cfg.train.params.network
            obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
            if "central_value_config" in cfg.train.params.config:
                critic_net_cfg = cfg.train.params.config.central_value_config.network
                obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
            
            vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
        else:
            # try: # 
            #     if cfg.task.amprlgpu == True: # 
            #         vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: AMPRLGPUEnv(config_name, num_actors, **kwargs))
            #     else: # 
            #         vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
            # except: 
            #     vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
            vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
            


        rlg_config_dict = omegaconf_to_dict(cfg.train)
        rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

        observers = [RLGPUAlgoObserver()]

        if cfg.pbt.enabled:
            pbt_observer = PbtAlgoObserver(cfg)
            observers.append(pbt_observer)

        if cfg.wandb_activate:
            cfg.seed += global_rank
            if global_rank == 0: # rl gpu algo observer #
                # initialize wandb only once per multi-gpu run # rl gpu algo observer #
                wandb_observer = WandbAlgoObserver(cfg)
                observers.append(wandb_observer)
        
        # register new AMP network builder and agent #
        def build_runner(algo_observer):
            runner = Runner(algo_observer)
            # registermodel and register network? #
            runner.algo_factory.register_builder('humanoid_amp', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
            runner.player_factory.register_builder('humanoid_amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
            
            # A2CAgentFromSupervised # if we use the deterministic #
            runner.algo_factory.register_builder('a2c_fromsupervised', lambda **kwargs : a2c_fromsupervised.A2CAgentFromSupervised(**kwargs))
            # runner.algo_factory.register_builder('a2c_fromsupervised', lambda **kwargs : a2c_fromsupervised.A2CAgentFromSupervised(**kwargs))
            
            runner.algo_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised.A2CSupervisedAgent(**kwargs))
            # runner.algo_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised_deterministic.A2CSupervisedAgent(**kwargs))
            runner.player_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised_player.A2CSupervisedPlayer(**kwargs))
            
            runner.algo_factory.register_builder('a2c_continuous_dagger', lambda **kwargs : a2c_dagger_continuous.ContinuousA2CBaseDAGGER(**kwargs))
            
            model_builder.register_model('humanoid_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
            model_builder.register_network('humanoid_amp', lambda **kwargs : amp_network_builder.AMPBuilder())

            # visionppo_models, visionppo_network_builder
            model_builder.register_model('visionppo', lambda network, **kwargs : visionppo_models.ModelVisionPPO(network))
            model_builder.register_network('visionppo', lambda **kwargs : visionppo_network_builder.VisionPPOBuilder())
            
            # runner.algo_factory.register_builder('humanoid_calm', lambda **kwargs : calm_agent.CALMAgent(**kwargs))
            # runner.player_factory.register_builder('humanoid_calm', lambda **kwargs : calm_players.CALMPlayer(**kwargs))
            # model_builder.register_model('humanoid_calm', lambda network, **kwargs : calm_models.ModelCALMContinuous(network))
            # model_builder.register_network('humanoid_calm', lambda **kwargs : calm_network_builder.CALMBuilder())


            # runner.algo_factory.register_builder('encamp', lambda **kwargs : encamp_agent.ENCAMPAgent(**kwargs))
            # # runner.player_factory.register_builder('encamp', lambda **kwargs : calm_players.CALMPlayer(**kwargs))
            # # model_builder.register_model('encamp', lambda network, **kwargs : encamp_models.ModelENCAMPContinuous(network))
            # model_builder.register_network('encamp', lambda **kwargs : encamp_network_builder.ENCAMPBuilder())
            # # runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
            return runner
        # 

        # convert CLI arguments into dictionary
        # create runner and set the settings
        runner = build_runner(MultiObserver(observers))
        # rl games config dict #
        runner.load(rlg_config_dict)
        runner.reset()


        args = {
            'train': False,
            'play': True,
            'checkpoint' : self.config['checkpoint'],
            'sigma' : None
        }


        self.player = runner.create_player()
        _restore(self.player, args)
        _override_sigma(self.player, args)

        # ## TODO: if we are in the test mode; set the random_time config to False #
        # # exp_logging_dir = os.path.join('runs', cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        
        # exp_run_root_dir = cfg.train.params.config.log_path
        # exp_logging_dir = os.path.join(exp_run_root_dir, cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        
        # os.makedirs(exp_logging_dir, exist_ok=True)
        # try:
        #     cfg['task']['env']['exp_logging_dir'] = exp_logging_dir
        # except:
        #     pass

        # try:
        #     cfg['task']['env']['test'] = cfg.test
        # except:
        #     pass
        
        # if cfg.test:
        #     cfg['task']["env"]["random_time"] = False
        
        
        # # dump config dict #
        # if not cfg.test:
        #     try:
        #         if cfg.task.delog == True:
        #             pass
        #     except:
        #         # cfg_task_delog == True
        #         # exp_logging_dir = os.path.join('runs', cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        #         # experiment_dir = os.path.join(exp_dir, 'runs', cfg.train.params.config.name  + 
        #         # '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        #         experiment_dir = exp_logging_dir 
        #         #  os.path.join('runs', cfg.train.params.config.name  + 
        #         # '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
        #         # '_{}'.format() #
        #         os.makedirs(experiment_dir, exist_ok=True)
        #         with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
        #             f.write(OmegaConf.to_yaml(cfg))
        # # cfg['task']['test'] = 
        # if not 'checkpoint_b' in cfg:
        #     runner.run({
        #         'train': not cfg.test,
        #         'play': cfg.test,
        #         'checkpoint': cfg.checkpoint,
        #         'sigma': cfg.sigma if cfg.sigma != '' else None
        #     })
        # else:
        #     runner.run({
        #         'train': not cfg.test,
        #         'play': cfg.test,
        #         'checkpoint_b': cfg.checkpoint_b,
        #         'checkpoint_h': cfg.checkpoint_h,
        #         'sigma': cfg.sigma if cfg.sigma != '' else None
        #     })
      


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):

    try:
        cseed = cfg.task.seed
    except:
        cseed = 20

    setup_seed(cseed)

    if cfg.pbt.enabled:
        initial_pbt_check(cfg)
    
    from isaacgymenvs.utils.rlgames_utils import RLGPUEnv, RLGPUAlgoObserver, MultiObserver, ComplexObsRLGPUEnv
    from isaacgymenvs.utils.rlgames_utils import AMPRLGPUEnv
    # from isaacgymenvs.utils.wandb_utils import WandbAlgoObserver
    from rl_games.common import env_configurations, vecenv
    from rl_games.torch_runner import Runner
    from rl_games.algos_torch import model_builder
    from isaacgymenvs.learning import amp_continuous
    from isaacgymenvs.learning import amp_players
    from isaacgymenvs.learning import amp_models
    from isaacgymenvs.learning import amp_network_builder
    from isaacgymenvs.learning import visionppo_models, visionppo_network_builder
    from isaacgymenvs.learning import a2c_dagger_continuous
    from isaacgymenvs.learning import a2c_supervised
    from isaacgymenvs.learning import a2c_supervised_player
    from isaacgymenvs.learning import a2c_fromsupervised
    from isaacgymenvs.learning import a2c_supervised_deterministic
    import isaacgymenvs

    with open_dict(cfg):
        cfg.task.test = cfg.test
        
        
    tag = cfg.tag
    time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{tag}_{cfg.wandb_name}_{time_str}"

    # ensure checkpoints can be specified as relative paths
    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    cfg_dict = omegaconf_to_dict(cfg)
    print_dict(cfg_dict)

    # set numpy formatting for printing only #
    set_np_formatting()

    # global rank of the GPU #
    global_rank = int(os.getenv("RANK", "0"))

    # sets seed. if seed is -1 will pick a random one #
    cfg.seed = set_seed(cfg.seed, torch_deterministic=cfg.torch_deterministic, rank=global_rank)
    
    
    exp_dir = cfg.exp_dir
    
    def create_isaacgym_env(**kwargs):
        envs = isaacgymenvs.make(
            cfg.seed, 
            cfg.task_name, 
            cfg.task.env.numEnvs, 
            cfg.sim_device,
            cfg.rl_device,
            cfg.graphics_device_id,
            cfg.headless,
            cfg.multi_gpu,
            cfg.capture_video,
            cfg.force_render,
            cfg,
            **kwargs,
        )
        if cfg.capture_video:
            envs.is_vector_env = True
            envs = gym.wrappers.RecordVideo(
                envs, # record video #
                os.path.join(exp_dir, f"videos/{run_name}"),
                step_trigger=lambda step: step % cfg.capture_video_freq == 0,
                video_length=cfg.capture_video_len,
            )
        return envs

    env_configurations.register('rlgpu', {
        'vecenv_type': 'RLGPU',
        'env_creator': lambda **kwargs: create_isaacgym_env(**kwargs),
    })

    ige_env_cls = isaacgym_task_map[cfg.task_name] # task map # 
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        # params network#
        obs_spec = {}
        actor_net_cfg = cfg.train.params.network
        obs_spec['obs'] = {'names': list(actor_net_cfg.inputs.keys()), 'concat': not actor_net_cfg.name == "complex_net", 'space_name': 'observation_space'}
        if "central_value_config" in cfg.train.params.config:
            critic_net_cfg = cfg.train.params.config.central_value_config.network
            obs_spec['states'] = {'names': list(critic_net_cfg.inputs.keys()), 'concat': not critic_net_cfg.name == "complex_net", 'space_name': 'state_space'}
        
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: ComplexObsRLGPUEnv(config_name, num_actors, obs_spec, **kwargs))
    else:
        # try: # 
        #     if cfg.task.amprlgpu == True: # 
        #         vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: AMPRLGPUEnv(config_name, num_actors, **kwargs))
        #     else: # 
        #         vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        # except: 
        #     vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        vecenv.register('RLGPU', lambda config_name, num_actors, **kwargs: RLGPUEnv(config_name, num_actors, **kwargs))
        


    rlg_config_dict = omegaconf_to_dict(cfg.train)
    rlg_config_dict = preprocess_train_config(cfg, rlg_config_dict)

    observers = [RLGPUAlgoObserver()]

    if cfg.pbt.enabled:
        pbt_observer = PbtAlgoObserver(cfg)
        observers.append(pbt_observer)

    if cfg.wandb_activate:
        cfg.seed += global_rank
        if global_rank == 0: # rl gpu algo observer #
            # initialize wandb only once per multi-gpu run # rl gpu algo observer #
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)
    
    # register new AMP network builder and agent #
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        # registermodel and register network? #
        runner.algo_factory.register_builder('humanoid_amp', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('humanoid_amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        
        # A2CAgentFromSupervised # if we use the deterministic #
        runner.algo_factory.register_builder('a2c_fromsupervised', lambda **kwargs : a2c_fromsupervised.A2CAgentFromSupervised(**kwargs))
        # runner.algo_factory.register_builder('a2c_fromsupervised', lambda **kwargs : a2c_fromsupervised.A2CAgentFromSupervised(**kwargs))
        
        runner.algo_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised.A2CSupervisedAgent(**kwargs))
        # runner.algo_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised_deterministic.A2CSupervisedAgent(**kwargs))
        runner.player_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised_player.A2CSupervisedPlayer(**kwargs))
        
        runner.algo_factory.register_builder('a2c_continuous_dagger', lambda **kwargs : a2c_dagger_continuous.ContinuousA2CBaseDAGGER(**kwargs))
        
        model_builder.register_model('humanoid_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('humanoid_amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        # visionppo_models, visionppo_network_builder
        model_builder.register_model('visionppo', lambda network, **kwargs : visionppo_models.ModelVisionPPO(network))
        model_builder.register_network('visionppo', lambda **kwargs : visionppo_network_builder.VisionPPOBuilder())
        
        # runner.algo_factory.register_builder('humanoid_calm', lambda **kwargs : calm_agent.CALMAgent(**kwargs))
        # runner.player_factory.register_builder('humanoid_calm', lambda **kwargs : calm_players.CALMPlayer(**kwargs))
        # model_builder.register_model('humanoid_calm', lambda network, **kwargs : calm_models.ModelCALMContinuous(network))
        # model_builder.register_network('humanoid_calm', lambda **kwargs : calm_network_builder.CALMBuilder())


        # runner.algo_factory.register_builder('encamp', lambda **kwargs : encamp_agent.ENCAMPAgent(**kwargs))
        # # runner.player_factory.register_builder('encamp', lambda **kwargs : calm_players.CALMPlayer(**kwargs))
        # # model_builder.register_model('encamp', lambda network, **kwargs : encamp_models.ModelENCAMPContinuous(network))
        # model_builder.register_network('encamp', lambda **kwargs : encamp_network_builder.ENCAMPBuilder())
        # # runner.algo_factory.register_builder('amp_continuous', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        return runner
    # 

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    # rl games config dict #
    runner.load(rlg_config_dict)
    runner.reset()

    ## TODO: if we are in the test mode; set the random_time config to False #
    # exp_logging_dir = os.path.join('runs', cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
    
    exp_run_root_dir = cfg.train.params.config.log_path
    exp_logging_dir = os.path.join(exp_run_root_dir, cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
    
    os.makedirs(exp_logging_dir, exist_ok=True)
    try:
        cfg['task']['env']['exp_logging_dir'] = exp_logging_dir
    except:
        pass

    try:
        cfg['task']['env']['test'] = cfg.test
    except:
        pass
    
    if cfg.test:
        cfg['task']["env"]["random_time"] = False
    
    
    # dump config dict #
    if not cfg.test:
        try:
            if cfg.task.delog == True:
                pass
        except:
            # cfg_task_delog == True
            # exp_logging_dir = os.path.join('runs', cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
            # experiment_dir = os.path.join(exp_dir, 'runs', cfg.train.params.config.name  + 
            # '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
            experiment_dir = exp_logging_dir 
            #  os.path.join('runs', cfg.train.params.config.name  + 
            # '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
            # '_{}'.format() #
            os.makedirs(experiment_dir, exist_ok=True)
            with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as f:
                f.write(OmegaConf.to_yaml(cfg))
    # cfg['task']['test'] = 
    if not 'checkpoint_b' in cfg:
        runner.run({
            'train': not cfg.test,
            'play': cfg.test,
            'checkpoint': cfg.checkpoint,
            'sigma': cfg.sigma if cfg.sigma != '' else None
        })
    else:
        runner.run({
            'train': not cfg.test,
            'play': cfg.test,
            'checkpoint_b': cfg.checkpoint_b,
            'checkpoint_h': cfg.checkpoint_h,
            'sigma': cfg.sigma if cfg.sigma != '' else None
        })


if __name__ == "__main__":
    launch_rlg_hydra()



# python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=True force_render=True headless=False  

##### train using the prev_state control mode #####
# python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=True force_render=True headless=False   task.env.numEnvs=1024 train.params.config.minibatch_size=1024  task.env.useRelativeControl=True train.params.config.max_epochs=10000 


# NOTE: goal_cond=False, w_obj_ornt=False
# checkpoint=runs/Humanoid_02-14-52-59/nn/Humanoid.pth

##### train using the prev_state control mode #####
# CUDA_VISIBLE_DEVICES=2 python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs=10240 train.params.config.minibatch_size=10240  task.env.useRelativeControl=True train.params.config.max_epochs=10000 task.env.w_obj_ornt=True  task.env.goal_cond=True 



# NOTE: tracking 
##### train using the prev_state control mode #####
# CUDA_VISIBLE_DEVICES=7 python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs=10240 train.params.config.minibatch_size=10240  task.env.useRelativeControl=True train.params.config.max_epochs=10000 task.env.mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy' checkpoint=runs/Humanoid_03-09-34-46/nn/Humanoid.pth task.env.goal_cond=True 


# NOTE: tracking -- capture video
# python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=True force_render=True headless=False   task.env.numEnvs=1024 train.params.config.minibatch_size=1024  task.env.useRelativeControl=True train.params.config.max_epochs=10000 task.env.mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy' checkpoint=runs/Humanoid_03-09-34-46/nn/Humanoid.pth


# task.env.w_obj_ornt=True  task.env.goal_cond=True 


# 
# task.env.w_obj_ornt=True # TODO: add this one
# checkpoint=runs/Humanoid_02-11-50-17/nn/last_Humanoid_ep_1000_rew__-91.1_.pth
# 
# checkpoint=runs/Humanoid_02-10-48-18/nn/Humanoid.pth



# python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:4' rl_device='cuda:4'  capture_video=False force_render=False headless=False   task.env.numEnvs=1024 train.params.config.minibatch_size=1024 task.env.useRelativeControl=True

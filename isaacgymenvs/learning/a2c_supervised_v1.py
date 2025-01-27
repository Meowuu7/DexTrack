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

from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import torch_ext
from rl_games.common import a2c_common
from rl_games.common import schedulers
from rl_games.common import vecenv


from rl_games.algos_torch import a2c_continuous

from isaacgymenvs.utils.torch_jit_utils import to_torch

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 
from rl_games.common import common_losses
from rl_games.algos_torch import  model_builder

from tensorboardX import SummaryWriter

from rl_games.common.experience import ExperienceBuffer
# common agent torch #
import os


# 

def swap_and_flatten01(arr):
    """
    swap and then flatten axes 0 and 1
    """
    if arr is None:
        return arr
    s = arr.size()
    return arr.transpose(0, 1).reshape(s[0] * s[1], *s[2:])

def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action = action * d + m
    return scaled_action


def print_statistics(print_stats, curr_frames, step_time, step_inference_time, total_time, epoch_num, max_epochs, frame, max_frames):
    if print_stats:
        step_time = max(step_time, 1e-9)
        fps_step = curr_frames / step_time
        fps_step_inference = curr_frames / step_inference_time
        fps_total = curr_frames / total_time

        if max_epochs == -1 and max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}')
        elif max_epochs == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f} frames: {frame:.0f}/{max_frames:.0f}')
        elif max_frames == -1:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}')
        else:
            print(f'fps step: {fps_step:.0f} fps step and policy inference: {fps_step_inference:.0f} fps total: {fps_total:.0f} epoch: {epoch_num:.0f}/{max_epochs:.0f} frames: {frame:.0f}/{max_frames:.0f}')


class A2CSupervisedAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, params):
        #  supervised agenet #
        super().__init__(base_name, params)
        # if self.normalize_value:
        #     self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        # if self._normalize_amp_input:
        #     self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        
        # mode=regular: then we should combine the supervised loss with the regular rl loss for training the actor and critic together
        # mode=offline_supervised: only train the actor using the supervised loss
        self.params = params.copy()
        
        self.training_mode = params['config'].get('training_mode', 'regular')
        # self.training_mode = params['config']['training_mode']
        self.preload_experiences_tf = params['config'].get('preload_experiences_tf', False)
        self.preload_experiences_path = params['config'].get('preload_experiences_path', None)
        self.single_instance_training = params['config'].get('single_instance_training', False)
        self.presaved_experience_st_idx = 0
        self.play_presaved_freq = 10
        # 
        self.single_instance_tag = params['config'].get('single_instance_tag', '') # 
        self.obj_type_to_optimized_res_fn = params['config'].get('obj_type_to_optimized_res_fn', '')
        self.supervised_loss_coef = params['config'].get('supervised_loss_coef', 0.0005)
        self.pure_supervised_training = params['config'].get('pure_supervised_training', False)
        
        self.grab_obj_type_to_opt_res_fn = params['config'].get('grab_obj_type_to_opt_res_fn', '')
        self.taco_obj_type_to_opt_res_fn = params['config'].get('taco_obj_type_to_opt_res_fn', '')
        self.use_teacher_model = params['config'].get('use_teacher_model', False)
        
        self.teacher_model_path = params['config'].get('teacher_model_path', '')
        self.w_franka = params['config'].get('w_franka', False)
        
        self.optimizing_with_teacher_net = len(self.teacher_model_path) > 0 and os.path.exists(self.teacher_model_path) #  get the teacher network model path #
        
        
        ###### Teacher models for masked mimicingg ######
        self.masked_mimic_training = params['config'].get('masked_mimic_training', False) # get the masked mimicing training #
        self.masked_mimic_teacher_model_path = params['config'].get('masked_mimic_teacher_model_path', '') # get the masked mimicing teacher model path #
        if self.masked_mimic_training:
            self._init_mask_mimic_teacher_model()
        
        
          
        # teacher_index_to_weights, use_multiple_teacher, _init_multiple_teacher_models
        # self.teacher_index_to_inst_tags = self.cfg['env'].get('teacher_index_to_inst_tags', '') 
        self.teacher_index_to_weights_fn = params['config'].get('teacher_index_to_weights', '') # get the teacher index to weights # 
        self.use_multiple_teacher = False
        if self.use_teacher_model and len(self.teacher_index_to_weights_fn) > 0 and os.path.exists(self.teacher_index_to_weights_fn):
            self.teacher_index_to_weights = np.load(self.teacher_index_to_weights_fn, allow_pickle=True).item()
            self.use_multiple_teacher = True
            self.optimizing_with_teacher_net = True
            self.nn_teacher = len(self.teacher_index_to_weights)
            # use mulitple teacher # #
            self._init_multiple_teacher_models()
            
            self.cur_teacher_idx = 0
            
        ###### Tacher models for specialist-generalist training ######
        if self.use_teacher_model and self.optimizing_with_teacher_net and (not self.use_multiple_teacher):
            self._init_teacher_models_single()
          
        
        
        
        if self.w_franka:
            
            self.nn_act_dims = 23
        else:
            self.nn_act_dims = 22
            
        return
    
    # build the teacher model #
    def _init_teacher_models(self, ):
        obj_feat_shape = 256
        obj_type_to_optimized_res_fn = self.obj_type_to_optimized_res_fn 
        obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item() 
        print(f"obj_type_to_optimized_res: {obj_type_to_optimized_res}")
        self.inst_tag_to_model_struct = {}
        if self.single_instance_tag == 'apple':
            self.teacher_obs_dim = self.obs_shape[0] 
        else:
            self.teacher_obs_dim = self.obs_shape[0]  - obj_feat_shape
            
        
        self.teacher_network_params = self.params.copy()
        if self.single_instance_tag == 'apple':
            self.teacher_network_params['network']['mlp']['units'] = [8192, 4096, 2048, 1024, 512, 256, 128]
        else:
            self.teacher_network_params['network']['mlp']['units'] = [400, 200, 100]
        self.teacher_network_builder = model_builder.ModelBuilder()
        self.teacher_network_builder = self.teacher_network_builder.load(self.teacher_network_params)
        
        # supervised training #
        # for cur_obj_type in obj_type_to_optimized_res: # #
        for cur_obj_type in self.interested_instance_tags:
            if isinstance(obj_type_to_optimized_res[cur_obj_type], tuple):
                cur_optimized_res = obj_type_to_optimized_res[cur_obj_type][1]
            else:
                cur_optimized_res = obj_type_to_optimized_res[cur_obj_type]
            
            cur_obj_model_build_config = {
                'actions_num' : self.actions_num,
                'input_shape' : (self.teacher_obs_dim, ),
                'num_seqs' : self.num_actors * self.num_agents,
                'value_size': self.env_info.get('value_size',1),
                'normalize_value' : self.normalize_value,
                'normalize_input': self.normalize_input, # # normalize # #
            }
            cur_obj_model = self.teacher_network_builder.build(cur_obj_model_build_config).to(self.ppo_device)
            
            cur_optimized_res_wieghts = torch.load(cur_optimized_res, map_location='cpu')
            cur_obj_model.load_state_dict(cur_optimized_res_wieghts['model'])
            cur_obj_model.eval()
            self.inst_tag_to_model_struct[cur_obj_type] = cur_obj_model
            
            # init the teacher model # # init the teacher model # # 
            
            # cur_model_struct = cur_optimized_res['model_struct']
            # self.inst_tag_to_model_struct[cur_obj_type] = cur_model_struct
    
        
    
    # teacher_index_to_weights, use_multiple_teacher, _init_multiple_teacher_models
    def _init_multiple_teacher_models(self, ):
        
        assert self.use_multiple_teacher 
        assert self.optimizing_with_teacher_net
        
        obj_feat_shape = 256
        
        self.teacher_obs_dim = self.obs_shape[0] 
        self.teacher_network_params = self.params.copy()
        self.teacher_network_builder = model_builder.ModelBuilder()
        self.teacher_network_builder = self.teacher_network_builder.load(self.teacher_network_params)
        
        self.teacher_index_to_obs_dim = {}
        
        self.teacher_index_to_models = {}
        for teacher_index in self.teacher_index_to_weights:
            # cur_teacher_model_weight_fn = self.teacher_index_to_model_weights[teacher_index]
            # teacher_index_to_weights
            cur_teacher_model_weight_fn = self.teacher_index_to_weights[teacher_index]
            # cur_teacher_model_weight
            # a2c_network.actor_mlp.0.weight # 
            
            teacher_res_weights = torch.load(cur_teacher_model_weight_fn, map_location='cpu')['model']
            first_teacher_weight_name = 'a2c_network.actor_mlp.0.weight'
            teacher_model_obs_dim = teacher_res_weights[first_teacher_weight_name].size(1)
            
            self.teacher_index_to_obs_dim[teacher_index] = teacher_model_obs_dim
            
            cur_obj_model_build_config = {
                'actions_num' : self.actions_num,
                'input_shape' : (teacher_model_obs_dim, ),
                'num_seqs' : self.num_actors * self.num_agents,
                'value_size': self.env_info.get('value_size',1),
                'normalize_value' : self.normalize_value,
                'normalize_input': self.normalize_input, # # normalize # #
            }
            cur_obj_model = self.teacher_network_builder.build(cur_obj_model_build_config).to(self.ppo_device)
            
            cur_optimized_res_wieghts = torch.load(cur_teacher_model_weight_fn, map_location='cpu')
            cur_obj_model.load_state_dict(cur_optimized_res_wieghts['model'])
            cur_obj_model.eval()
            
            # teacher model cur obj model # # cur obj model #
            # self.teacher_model = cur_o bj_model
            
            self.teacher_index_to_models[teacher_index] = cur_obj_model 
    
       
    
    
    def _init_teacher_models_single(self, ):
        obj_feat_shape = 256
        
        self.teacher_obs_dim = self.obs_shape[0] 
        self.teacher_network_params = self.params.copy()
        self.teacher_network_builder = model_builder.ModelBuilder()
        self.teacher_network_builder = self.teacher_network_builder.load(self.teacher_network_params)
        
        cur_optimized_res = self.teacher_model_path
        cur_obj_model_build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : (self.teacher_obs_dim, ),
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input, # # normalize # #
        }
        cur_obj_model = self.teacher_network_builder.build(cur_obj_model_build_config).to(self.ppo_device)
        
        # cur optimized res weights # #
        print(f"loading the teacher mode from: {cur_optimized_res}") # # # cur optimized res # # # #
        cur_optimized_res_wieghts = torch.load(cur_optimized_res, map_location='cpu')
        cur_obj_model.load_state_dict(cur_optimized_res_wieghts['model'])
        cur_obj_model.eval()
        
        self.teacher_model = cur_obj_model
        
        # self.inst_tag_to_model_struct[cur_obj_type] = cur_obj_model
        
        
        # # obj_type_to_optimized_res_fn = self.obj_type_to_optimized_res_fn 
        # # obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item() 
        # # print(f"obj_type_to_optimized_res: {obj_type_to_optimized_res}")
        # # self.inst_tag_to_model_struct = {}
        # # if self.single_instance_tag == 'apple':
        # #     self.teacher_obs_dim = self.obs_shape[0] 
        # # else:
        # #     self.teacher_obs_dim = self.obs_shape[0]  - obj_feat_shape
            
        
        # # self.teacher_network_params = self.params.copy()
        # # if self.single_instance_tag == 'apple':
        # #     self.teacher_network_params['network']['mlp']['units'] = [8192, 4096, 2048, 1024, 512, 256, 128]
        # # else:
        # #     self.teacher_network_params['network']['mlp']['units'] = [400, 200, 100]
        # # self.teacher_network_builder = model_builder.ModelBuilder()
        # # self.teacher_network_builder = self.teacher_network_builder.load(self.teacher_network_params)
        
        # # supervised training #
        # # for cur_obj_type in obj_type_to_optimized_res: # #
        # for cur_obj_type in self.interested_instance_tags:
        #     if isinstance(obj_type_to_optimized_res[cur_obj_type], tuple):
        #         cur_optimized_res = obj_type_to_optimized_res[cur_obj_type][1]
        #     else:
        #         cur_optimized_res = obj_type_to_optimized_res[cur_obj_type]
            
        #     cur_obj_model_build_config = {
        #         'actions_num' : self.actions_num,
        #         'input_shape' : (self.teacher_obs_dim, ),
        #         'num_seqs' : self.num_actors * self.num_agents,
        #         'value_size': self.env_info.get('value_size',1),
        #         'normalize_value' : self.normalize_value,
        #         'normalize_input': self.normalize_input, # # normalize # #
        #     }
        #     cur_obj_model = self.teacher_network_builder.build(cur_obj_model_build_config).to(self.ppo_device)
            
        #     cur_optimized_res_wieghts = torch.load(cur_optimized_res, map_location='cpu')
        #     cur_obj_model.load_state_dict(cur_optimized_res_wieghts['model'])
        #     cur_obj_model.eval()
        #     self.inst_tag_to_model_struct[cur_obj_type] = cur_obj_model
            
        #     # init the teacher model # # init the teacher model # # 
            
        #     # cur_model_struct = cur_optimized_res['model_struct']
        #     # self.inst_tag_to_model_struct[cur_obj_type] = cur_model_struct
    
    
    def _init_mask_mimic_teacher_model(self, ):
        assert self.masked_mimic_training
        
        
        obj_feat_shape = 256
        # The same observation space but one with masks and one without #
        
        self.teacher_obs_dim = self.obs_shape[0] 
        self.teacher_network_params = self.params.copy()
        self.teacher_network_builder = model_builder.ModelBuilder()
        self.teacher_network_builder = self.teacher_network_builder.load(self.teacher_network_params)
        
        
        
        teacher_model_weight_fn = self.masked_mimic_teacher_model_path
        
        print(f"[Debug] Loading mimic teacher model from {teacher_model_weight_fn}")
        
        assert len(teacher_model_weight_fn) > 0 and os.path.exists(teacher_model_weight_fn)
        
        teacher_model_build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : (self.teacher_obs_dim, ),
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        }
        teacher_model = self.teacher_network_builder.build(teacher_model_build_config).to(self.ppo_device)
        teacher_model_weights = torch.load(teacher_model_weight_fn, map_location='cpu')
        teacher_model.load_state_dict(teacher_model_weights['model'])
        teacher_model.eval()
        self.mimic_teacher_model = teacher_model
        
    # def _get_mimic_teacher_action_values(self, obs, teacher_model):
        # pass
    
    
    def _get_teacher_action_values(self, obs, teacher_model, teacher_model_obs_dim=None):
        
        # if self.use_multiple_teacher:
        #     assert teacher_idx is not None
        #     cur_techer_model = self.teacher_index_to_models[teacher_idx]
        # else:
        #     cur_techer_model = teacher_model
        
        teacher_model_obs_dim = self.teacher_obs_dim if teacher_model_obs_dim is None else teacher_model_obs_dim
        
        processed_obs = self._preproc_obs(obs['obs'])
        
        if processed_obs.size(-1) < teacher_model_obs_dim:
            processed_obs = torch.cat([processed_obs, torch.zeros(processed_obs.size(0), teacher_model_obs_dim - processed_obs.size(-1), device=processed_obs.device)], dim=-1)
        
        teacher_model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., : teacher_model_obs_dim],
            'rnn_states' : self.rnn_states
        }

        # get the actions values #
        with torch.no_grad():
            res_dict = teacher_model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict 
    
    
    def _get_mimic_teacher_action_values(self, obs, teacher_model):
        
        processed_obs = self._preproc_obs(obs['mimic_teacher_obs'])
        teacher_model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., : self.teacher_obs_dim],
            'rnn_states' : self.rnn_states
        }

        # get the actions values #
        with torch.no_grad():
            res_dict = teacher_model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict 
    
    def build_demo_experience_buffer(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.demo_experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)
        val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.demo_current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.demo_current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.demo_current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.demo_dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        if self.is_rnn:
            self.demo_rnn_states = self.model.get_default_rnn_state()
            self.demo_rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
            self.demo_mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]

        batch_shape = self.demo_experience_buffer.obs_base_shape
        # 
        self.demo_experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (self.nn_act_dims,),
                                                                    device=self.ppo_device)
        
        
        
        # if self.mimic_teacher_model:
        
        
        
        # amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        # self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        # self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        # replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        # self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['gt_act']
        
        if self.masked_mimic_training:
            self.demo_experience_buffer.tensor_dict['mimic_teacher_obs'] = torch.zeros(batch_shape + (self.obs_shape[0], ),
                                                                    device=self.ppo_device)
            self.tensor_list += ['mimic_teacher_obs']
            
        
        
        
    
    def preload_demonstrations(self ):
        # and may have many demonstrations #
        demonstration_fn = ""
        demonstration_fn = np.load(demonstration_fn, allow_pickle=True).item()
         # 0 1 to the useful information j # 
        # print(f"demo_fn: {demonstration_fn.keys()}"
        # get obs and the acts from the buffer#
        self.tot_obs = []
        self.tot_acts = []
        tot_ts = list(demonstration_fn.keys())
        tot_ts = [cur_ts for cur_ts in tot_ts if isinstance(demonstration_fn[cur_ts], dict) and isinstance(cur_ts, int)]
        tot_ts =sorted(tot_ts)
        for i_ts in tot_ts:
            cur_ts_dict = demonstration_fn[i_ts]
            cur_ts_obs = cur_ts_dict['observations']
            cur_ts_acts = cur_ts_dict['actions']
            self.tot_obs.append(cur_ts_obs)
            self.tot_acts.append(cur_ts_acts)
        self.tot_obs = np.stack(self.tot_obs) # nn_ts x nn_envs x nn_obs_dim 
        self.tot_acts = np.stack(self.tot_acts) # nn_ts x nn_envs x nn_act_dim
        self.tot_obs = torch.from_numpy(self.tot_obs).to(self.ppo_device) 
        self.tot_acts = torch.from_numpy(self.tot_acts).to(self.ppo_device) 
        pass


    def init_tensors(self):
        super().init_tensors()
        self._build_gt_act_buffers()
        if self.preload_experiences_tf:
            self.preload_saved_experiences()
        ### TODO: add the demo experience buffer and the logic of preload demonstrations ###
        # self.build_demo_experience_buffer() #
        # self.preload_demonstrations() #
        return
    
    
    

    # prepare
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['gt_act'] = batch_dict['gt_act']
        if self.masked_mimic_training:
            self.dataset.values_dict['mimic_teacher_obs'] = batch_dict['mimic_teacher_obs']
        # self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        # self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        # self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        # return




    # if we use the deterministic network toher than ouputing the action distributions ? #
    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['obs'])
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., : self.obs_shape[0]],
            'rnn_states' : self.rnn_states
        }

        # get the actions values #
        with torch.no_grad(): # get mus and get actions --- in the network #
            res_dict = self.model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value: # has central value #
                states = obs['states']
                self.central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.model.eval()
                processed_obs = self._preproc_obs(obs['obs'])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs[..., : self.obs_shape[0]],
                    'rnn_states' : self.rnn_states
                }
                result = self.model(input_dict)
                value = result['values']
            return value
    
    def play_steps_gt_dataset(self ):
        update_list = self.update_list

        step_time = 0.0
        
        pass

    
    def play_demo_steps(self):
        update_list = self.update_list
        step_time = 0.0
        
        self.obs = self.env_reset()
        
        demo_play_length = self.tot_obs.size(0) - 1
        
        for n in range(demo_play_length):
            gt_act_val = self.tot_acts[n + 1]
            self.demo_experience_buffer.update_data('gt_act', n, gt_act_val)
            
            if 'obs' in self.obs:
                self.obs['obs'] = self.obs['obs'][..., : self.obs_shape[0]]
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            # in
            self.demo_experience_buffer.update_data('obses', n, self.obs['obs'][..., : self.obs_shape[0]])
            self.demo_experience_buffer.update_data('dones', n, self.demo_dones)
            
            
            for k in update_list:
                self.demo_experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.demo_experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.perf_counter()
            res_dict['actions'] = self.tot_acts[n + 1]
            
            self.obs, rewards, self.demo_dones, infos = self.env_step(res_dict['actions'])
            step_time_end = time.perf_counter()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos: # sahped rewards 
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.demo_experience_buffer.update_data('rewards', n, shaped_rewards)

            self.demo_current_rewards += rewards
            self.demo_current_shaped_rewards += shaped_rewards
            self.demo_current_lengths += 1
            all_done_indices = self.demo_dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            # game shaped rewards #
            self.game_rewards.update(self.demo_current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.demo_current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.demo_current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.demo_dones.float()

            self.demo_current_rewards = self.demo_current_rewards * not_dones.unsqueeze(1)
            self.demo_current_shaped_rewards = self.demo_current_shaped_rewards * not_dones.unsqueeze(1)
            self.demo_current_lengths = self.demo_current_lengths * not_dones


        # if 'obs' in self.obs:
        #     self.obs['obs'] = self.obs['obs'][..., : self.obs_shape[0]]
        last_values = self.get_values(self.obs)

        fdones = self.demo_dones.float()
        mb_fdones = self.demo_experience_buffer.tensor_dict['dones'].float()
        mb_values = self.demo_experience_buffer.tensor_dict['values']
        mb_rewards = self.demo_experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        # get transformed list?
        batch_dict = self.demo_experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        # print(f"batch_dict: {batch_dict.keys()}, ")
        # print(f"tensor_list: {self.tensor_list}")
        return batch_dict

    
    ## TODO: add a function to preload the saved experiences ##
    ## TODO: when we have loaded the saved experiences, add a function to sample from the pre-saved experiences and add to the current experience buffer ##
    def preload_saved_experiences(self):
        # preload_experiences_path = self.preload_experiences_path
        self.preload_experiences = np.load(self.preload_experiences_path, allow_pickle=True).item()
        ###### Convert the preloaded experiences to torch tensors ######
        self.preload_experiences = {
            key: torch.from_numpy(self.preload_experiences[key]).float().to(self.device) for key in self.preload_experiences
        }
        # self.num_actors #
        for key in self.preload_experiences:
            if self.preload_experiences[key].size(1) < self.num_actors:
                print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                self.preload_experiences[key] = torch.cat(
                    [
                        self.preload_experiences[key] for _ in range(self.num_actors // self.preload_experiences[key].size(1))
                    ], dim=1
                )
            # else:
            #     self.preload_experiences[key] = self.preload_experiences[key].reshape(self.num_actors, -1, *self.preload_experiences[key].shape[1:])
        ###### TODO: sample from the preloaded experiences and add to the experience buffer ######
        # pass
    
    def play_presaved_experiences(self ):
        update_list = self.update_list
        
        # obses: (300, 8000, 469)
        # rewards: (300, 8000, 1)
        # values: (300, 8000, 1)
        # neglogpacs: (300, 8000)
        # dones: (300, 8000)
        # actions: (300, 8000, 22)
        # mus: (300, 8000, 22)
        # sigmas: (300, 8000, 22)
        
        for i_n in range(self.horizon_length):
            n = self.presaved_experience_st_idx
            
            actions = self.preload_experiences['actions'][n] ## n-th actions ##
            obses = self.preload_experiences['obses'][n] ## n-th observations ##
            neglogpacs = self.preload_experiences['neglogpacs'][n] ## n-th neglogpacs ##
            values = self.preload_experiences['values'][n]  ## n-th values ##
            dones = self.preload_experiences['dones'][n]
            rewards = self.preload_experiences['rewards'][n]
            # if 'obs' in res_dict: # play presaved experiments #
            #     res_dict['obs'] = res_dict['obs'][..., : self.obs_shape[0]]
            
            self.experience_buffer.update_data('obses', i_n, obses)
            self.experience_buffer.update_data('dones', i_n, dones)
            for k in update_list:
                self.experience_buffer.update_data(k, i_n, self.preload_experiences[k][n]) # the preloaded experiences #
            if self.has_central_value:
                self.experience_buffer.update_data('states', i_n, self.preload_experiences['states'][n])
            
            
            step_time_start = time.perf_counter()
            # if we use deterministic actions? # 
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # TODO: try to change the actions to mus? #
            step_time_end = time.perf_counter()

            # step_time += (step_time_end - step_time_start)

            # shaped_rewards = self.rewards_shaper(rewards)
            # if self.value_bootstrap and 'time_outs' in infos:
            #     shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', i_n, rewards)

            # self.current_rewards += rewards
            # self.current_shaped_rewards += shaped_rewards
            # self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            # # if not self.epoch_num % 10 == 0:
            # self.game_rewards.update(self.current_rewards[env_done_indices])
            # self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            # self.game_lengths.update(self.current_lengths[env_done_indices])
            # self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            # self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            # self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            # self.current_lengths = self.current_lengths * not_dones
            
            self.presaved_experience_st_idx += 1
            maxx_steps = 149
            self.presaved_experience_st_idx = self.presaved_experience_st_idx % maxx_steps
            
        # last_values = self.get_values(self.obs)
        
        # if self.training_mode == 'offline_supervised':
        #     #### version 2 for creating the expert demonstration data ####
        #     last_values = torch.ones_like(last_values)
        #     #### version 2 for creating the expert demonstration data ####
            
        # # if self.epoch_num % 10 == 0:
        # #     #### version 2 for creating the expert demonstration data ####
        # #     last_values = torch.ones_like(last_values) # demonstration data #
        # #     #### version 2 for creating the expert demonstration data ####
        if self.presaved_experience_st_idx == 0:
            last_values = self.preload_experiences['values'][maxx_steps - 1]
        else:
            last_values = self.preload_experiences['values'][self.presaved_experience_st_idx]
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        # mb_returns = mb_values
        
        
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = 0.001
        
        return batch_dict


            
    # when training t # l
    
    def play_steps(self):
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length): 
            # print(f"self.obs: {self.obs['obs'].size()}, self.obs_shape: {self.obs_shape}")
            
            if self.masked_mimic_training:
                mimic_teacher_obs = self.obs['mimic_teacher_obs'][..., : ] # get the last obs features as teacher features here #
                # self.obs['obs'] = self.obs['obs'][..., : - self.obs_shape[0]]
                self.experience_buffer.update_data('mimic_teacher_obs', n, mimic_teacher_obs)
            
            gt_act_val = self.obs['obs'][..., self.obs_shape[0]: ]
            # if we do not use the gt act val? ---- where the value would be added ? #
            
            # the last obs_shape[0] dim is the gt_act_val 
            
            self.experience_buffer.update_data('gt_act', n, gt_act_val)
            
            # get the self.obs # 
            
            
            
            # if 'obs' in self.obs:
            #     self.obs['obs'] = self.obs['obs'][..., : self.obs_shape[0]]
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            # res_dict['actions'] = res_dict['mus']
            # if self.epoch_num % 10 == 0:
            #     ### use gtact to collect the dataset ####
            #     res_dict['actions'] = gt_act_val
            #     res_dict_mus = res_dict['mus']
            #     res_dict_sigmas = res_dict['sigmas']
            #     distr = torch.distributions.Normal(res_dict_mus, res_dict_sigmas, validate_args=False)
            #     neglogp = -distr.log_prob(res_dict['actions']).sum(dim=-1)
            #     res_dict['neglogpacs'] =  torch.squeeze(neglogp)
                ### use gtact to collect the dataset ####
                
            # if self.epoch_num % 10 == 0:
            #     #### version 2 for creating the expert demonstration data ####
            #     res_dict['actions'] = gt_act_val
            #     res_dict_mus = res_dict['mus']
            #     res_dict_sigmas = res_dict['sigmas']
            #     distr = torch.distributions.Normal(res_dict_mus, res_dict_sigmas, validate_args=False)
            #     neglogp = -distr.log_prob(res_dict['actions']).sum(dim=-1)
            #     neglogp = torch.zeros_like(neglogp)
            #     res_dict['neglogpacs'] =  neglogp
            #     res_dict['values'] = torch.ones_like(res_dict['values'])
            #     #### version 2 for creating the expert demonstration data ####
            
            if self.training_mode == 'offline_supervised' or self.pure_supervised_training:
            
                #### version 2 for creating the expert demonstration data ####
                res_dict['actions'] = gt_act_val
                res_dict_mus = res_dict['mus']
                res_dict_sigmas = res_dict['sigmas']
                distr = torch.distributions.Normal(res_dict_mus, res_dict_sigmas, validate_args=False)
                neglogp = -distr.log_prob(res_dict['actions']).sum(dim=-1)
                neglogp = torch.zeros_like(neglogp)
                res_dict['neglogpacs'] =  neglogp
                res_dict['values'] = torch.ones_like(res_dict['values'])
                #### version 2 for creating the expert demonstration data ####
            
            
            # if 'obs' in res_dict:
            #     res_dict['obs'] = res_dict['obs'][..., : self.obs_shape[0]]
            
            self.experience_buffer.update_data('obses', n, self.obs['obs'][..., : self.obs_shape[0]])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.perf_counter()
            # if we use deterministic actions? # 
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # TODO: try to change the actions to mus? #
            step_time_end = time.perf_counter()

            step_time += (step_time_end - step_time_start)

            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()

            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            # if not self.epoch_num % 10 == 0:
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones


        # if 'obs' in self.obs:
        #     self.obs['obs'] = self.obs['obs'][..., : self.obs_shape[0]]
        last_values = self.get_values(self.obs)
        
        if self.training_mode == 'offline_supervised' or self.pure_supervised_training:
            #### version 2 for creating the expert demonstration data ####
            last_values = torch.ones_like(last_values) # 
            #### version 2 for creating the expert demonstration data ####
            
        # if self.epoch_num % 10 == 0:
        #     #### version 2 for creating the expert demonstration data ####
        #     last_values = torch.ones_like(last_values) # demonstration data #
        #     #### version 2 for creating the expert demonstration data ####

        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values

        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        # print(f"batch_dict: {batch_dict.keys()}, ")
        # print(f"tensor_list: {self.tensor_list}")
        return batch_dict

    def actor_loss_supervised_bak(self, pred_actions, gt_actions):
        if len(self.grab_obj_type_to_opt_res_fn) > 0:
            
            gt_succ_flag = gt_actions[..., -1]
            avg_succ_flag = torch.sum(gt_succ_flag) / gt_succ_flag.size(0)
            # print(f"avg_succ_flag: {avg_succ_flag}")
            gt_actions = gt_actions[..., :-1]
            pred_actions = pred_actions[..., :]
            loss = torch.sum(
                torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='none'), dim=-1
            )
            loss = loss * gt_succ_flag
        else:
            loss = torch.sum(
                torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='none'), dim=-1
            )
        # print(f"pred_actions: {pred_actions.size()}, gt_actions: {gt_actions.size()}, loss: {loss.size()}")
        return loss
    
    # def teacher_loss_supervised()
    
    def actor_loss_supervised(self, pred_actions, gt_actions):
        
        
        
        # if len(self.grab_obj_type_to_opt_res_fn) > 0:
            
        gt_succ_flag = gt_actions[..., -1]
        avg_succ_flag = torch.sum(gt_succ_flag) / gt_succ_flag.size(0)
        # print(f"avg_succ_flag: {avg_succ_flag}")
        gt_actions = gt_actions[..., :-1]
        pred_actions = pred_actions[..., :]
        loss = torch.sum(
            torch.nn.functional.mse_loss(pred_actions[..., :self.nn_act_dims], gt_actions, reduction='none'), dim=-1
        )
        loss = loss * gt_succ_flag
        # else:
        #     loss = torch.sum(
        #         torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='none'), dim=-1
        #     )
        # print(f"pred_actions: {pred_actions.size()}, gt_actions: {gt_actions.size()}, loss: {loss.size()}")
        return loss
    
    
    def actor_loss_mimic_teacher(self, pred_actions, gt_actions):
        
        gt_actions = gt_actions[..., :]
        pred_actions = pred_actions[..., :]
        loss = torch.sum(
            torch.nn.functional.mse_loss(pred_actions[..., :self.nn_act_dims], gt_actions, reduction='none'), dim=-1
        )
        
        return loss


    def calc_gradients(self, input_dict):
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        # input #
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions'] # 
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        env_obs_batch = obs_batch[..., : self.obs_shape[0]] 
        
        
        # the current state #
        gt_act_batch = input_dict['gt_act']#
        
        if self.masked_mimic_training:
            mimic_teacher_obs_batch = input_dict['mimic_teacher_obs']
        
        if self.use_teacher_model and self.optimizing_with_teacher_net:
            # obj_inst_tag = self.single_instance_tag
            # obj_teacher_model = self.inst_tag_to_model_struct[obj_inst_tag]
            # teacher_res_dict = self._get_teacher_action_values(input_dict, obj_teacher_model)
            # gt_act_batch = teacher_res_dict['actions'] 
            
            if self.use_multiple_teacher:
                # added to  the gt acts # 
                
                ######## multiple teacher supervision strategy 1 ###########
                # encoded_succ_idxes = gt_act_batch[..., -1:].int()
                # single_succ_code = encoded_succ_idxes[0, 0].item()
                # single_succ_code = int(single_succ_code) # get the single succ code #
                # # single succ code # 
                # # nn_teacher # 
                # cur_teacher_idx = single_succ_code % self.nn_teacher
                # obj_teacher_model = self.teacher_index_to_models[cur_teacher_idx]
                # gt_succ_index = (encoded_succ_idxes - cur_teacher_idx) // self.nn_teacher 
                
                # encoded_teaecher_idx = encoded_succ_idxes - gt_succ_index * self.nn_teacher # if the encoded teacher idx is enqual to 
                # gt_succ_index = (encoded_teaecher_idx == cur_teacher_idx).int() + gt_succ_index
                # gt_succ_index = (gt_succ_index == 2).int()
                
                # # gt succ index #
                # gt_succ_index = gt_succ_index.float()
                ######## multiple teacher supervision strategy 1 ###########
                
                
                
                ######## multiple teacher supervision strategy 2 ###########
                obj_teacher_model = self.teacher_index_to_models[self.cur_teacher_idx]
                envs_teacher_idxes = gt_act_batch[..., -1:].int()
                gt_succ_index = (envs_teacher_idxes == self.cur_teacher_idx).float()
                gt_succ_index = gt_succ_index.float()
                teacher_model_obs_dim = self.teacher_index_to_obs_dim[self.cur_teacher_idx]
                ######## multiple teacher supervision strategy 2 ###########
                
                # get the teacher supervision strategy # # get the teacher supervision strategy # #get the teacher supervision  
                # print(f"single_succ_code: {single_succ_code}, cur_teacher_idx: {cur_teacher_idx}, nn_teacher: {self.nn_teacher}") # nothing for # nothing for the teacher #
                # print(f"gt_succ_index: {gt_succ_index.mean()}")
            else:
                obj_teacher_model = self.teacher_model
                gt_succ_index = gt_act_batch[..., -1:] 
                teacher_model_obs_dim = None
                
            
            # obj_teacher_model = self.teacher_model
            teacher_res_dict = self._get_teacher_action_values(input_dict, obj_teacher_model, teacher_model_obs_dim=teacher_model_obs_dim)
            teacher_gt_act_batch = teacher_res_dict['actions']
            
            # gt_act_batch = torch.cat( # searched res # and a is better than b # and a is better than b #
            #     [ teacher_gt_act_batch, gt_act_batch[..., -1:] ], dim=-1 # use the teacher model to # searced res # and a is better than b #
            # ) # 
            
            gt_act_batch = torch.cat(
                [ teacher_gt_act_batch, gt_succ_index ], dim=-1
            )
            # print(f"gt_act_batch: {gt_act_batch.size()}")
        if self.masked_mimic_training:
            # mimic_teacher_obs_batch = input_dict['mimic_teacher_obs']
            # mimic_teacher_obs_batch = self._preproc_obs(mimic_teacher_obs_batch)
            teacher_res_dict = self._get_mimic_teacher_action_values(input_dict, self.mimic_teacher_model)
            teacher_gt_act_batch = teacher_res_dict['actions']
            # env_obs_batch = torch.cat([env_obs_batch, mimic_teacher_obs_batch], dim=-1)
            gt_act_batch = teacher_gt_act_batch
        
        
        
        lr_mul = 1.0
        curr_e_clip = self.e_clip

        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs' : env_obs_batch,
        }

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            


        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values'] 
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            
            sigma = res_dict['sigmas']
            
            # sigma = torch.zeros_like(sigma)
            
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            # ####### Supervised loss Version 1 ####### #
            # rnd_noise = torch.randn_like(mu) * sigma
            # sampled_actions = mu + rnd_noise
            # supervised_a_loss = self.actor_loss_supervised(sampled_actions, gt_act_batch)
            # ####### Supervised loss Version 1 ####### #

            # ####### Supervised loss Version 2 ####### #
            # TODO: what's the role of rnns here? #
            # TODO: and also what's the role of rnn_masks? #
            if self.masked_mimic_training:
                supervised_a_loss = self.actor_loss_mimic_teacher(mu, gt_act_batch)
            else:
                supervised_a_loss = self.actor_loss_supervised(mu, gt_act_batch)
            # ####### Supervised loss Version 2 ####### #
            
            # training mode #
            # 
            if self.training_mode == 'offline_supervised':
                # ####### Supervised loss Version 3 ####### #
                policy_distr = torch.distributions.Normal(mu, sigma, validate_args=False) 
                neglog_gt_acts = -policy_distr.log_prob(gt_act_batch).sum(dim=-1)
                supervised_a_loss = neglog_gt_acts
                # ####### Supervised loss Version 3 ####### #
                
            
            
            
            
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model,value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound':
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([supervised_a_loss.unsqueeze(1), a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            supervised_a_loss, a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3], losses[4]

            # ##### Version 1 --- only use the action supervised loss for training #####
            # a_loss = supervised_a_loss
            # c_loss = a_loss
            # entropy = torch.zeros_like(a_loss)
            # b_loss = torch.zeros_like(a_loss)
            # loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            # ##### Version 1 #####
            
            # supervised_a_loss_coef = 0.0005
            # supervised_a_loss_coef = 1.0
            # supervised_a_loss_coef = 0.0
            # supervised_a_loss_coef = 0.0005
            supervised_a_loss_coef = self.supervised_loss_coef
            if self.single_instance_training:
                # print(f"single_instance_training: {self.single_instance_training}")
                supervised_a_loss_coef = 0.0
            a_loss_coef = 1.0
            c_loss_coef = 1.0
            entropy_coef = 1.0
            bounds_loss_coef = 1.0
            if self.pure_supervised_training:
                a_loss_coef = 0.0
                supervised_a_loss_coef = 1.0
                entropy_coef = 0.0
                bounds_loss_coef = 0.0
                c_loss_coef = 0.0
            if self.training_mode == 'offline_supervised':
                a_loss_coef = 0.0
                supervised_a_loss_coef = 1.0
                entropy_coef = 1.0
                bounds_loss_coef = 1.0
                c_loss_coef = 0.0
            if self.preload_experiences_tf and self.epoch_num % self.play_presaved_freq == 0:
                supervised_a_loss_coef = 0.0
            # a_loss_coef = 0.0
            # supervised_a_loss_coef = 1.0
            # if self.epoch_num % 2 == 0:
            #     a_loss_coef = 0.0
            #     supervised_a_loss_coef = 1.0
            ##### Version 2 -- with supervised action loss #####
            loss = supervised_a_loss * supervised_a_loss_coef +  a_loss * a_loss_coef + 0.5 * c_loss * self.critic_coef * c_loss_coef - entropy * self.entropy_coef * entropy_coef + b_loss * self.bounds_loss_coef * bounds_loss_coef
            ##### Version 2 -- with supervised action loss #####
            
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            # loss =  a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None

        self.scaler.scale(loss).backward()
        #TODO: Refactor this ugliest code of they year
        self.trancate_gradients_and_step()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            # mu detach; #
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask #

        self.diagnostics.mini_batch(self,
        {
            'values' : value_preds_batch,
            'returns' : return_batch,
            'new_neglogp' : action_log_probs,
            'old_neglogp' : old_action_log_probs_batch,
            'masks' : rnn_masks
        }, curr_e_clip, 0)      

        self.train_result = (a_loss, c_loss, entropy, \
            kl_dist, self.last_lr, lr_mul, \
            mu.detach(), sigma.detach(), b_loss)
        
        # print(f"supervised_a_loss: {supervised_a_loss.size()}")
        self.supervised_a_loss_np = supervised_a_loss.detach().cpu().numpy()
        self.a_loss_np = a_loss.detach().cpu().numpy()
        self.c_loss_np = c_loss.detach().cpu().numpy()
        self.entropy_loss_np = -entropy.detach().cpu().numpy()
        self.b_loss_np = b_loss.detach().cpu().numpy()
        return 


    def train_epoch(self):
        super().train_epoch()

        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            if self.preload_experiences_tf and self.epoch_num % self.play_presaved_freq == 0:
                batch_dict = self.play_presaved_experiences()
            else:
                if self.is_rnn:
                    batch_dict = self.play_steps_rnn()
                else:
                    batch_dict = self.play_steps()

        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()
        rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            ep_act_supervised_losses = []
            ep_a_losses = []
            ep_c_losses = []
            ep_entropy_losses = []
            ep_b_losses = []
            for i in range(len(self.dataset)):
                # actor
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(self.dataset[i])
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                self.dataset.update_mu_sigma(cmu, csigma)
                if self.schedule_type == 'legacy':
                    av_kls = kl
                    if self.multi_gpu:
                        dist.all_reduce(kl, op=dist.ReduceOp.SUM)
                        av_kls /= self.world_size
                    self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                    self.update_lr(self.last_lr)

                ep_act_supervised_losses.append(self.supervised_a_loss_np)
                ep_a_losses.append(self.a_loss_np)
                ep_c_losses.append(self.c_loss_np)
                ep_entropy_losses.append(self.entropy_loss_np)
                ep_b_losses.append(self.b_loss_np)
                
            
            ep_act_supervised_losses = np.array(ep_act_supervised_losses)
            avg_ep_act_supervised_loss = np.mean(ep_act_supervised_losses).item()
            ep_a_losses = np.array(ep_a_losses)
            avg_ep_a_loss = np.mean(ep_a_losses).item()
            ep_c_losses = np.array(ep_c_losses)
            avg_ep_c_loss = np.mean(ep_c_losses).item()
            ep_entropy_losses = np.array(ep_entropy_losses)
            avg_ep_entropy_loss = np.mean(ep_entropy_losses).item()
            ep_b_losses = np.array(ep_b_losses)
            avg_ep_b_loss = np.mean(ep_b_losses).item()
            
            # single_instance_training 
            if not self.single_instance_training:
                print(f"avg_ep_act_supervised_loss: {avg_ep_act_supervised_loss}, avg_ep_a_loss: {avg_ep_a_loss}, avg_ep_c_loss: {avg_ep_c_loss}, avg_ep_entropy_loss: {avg_ep_entropy_loss}, avg_ep_b_loss: {avg_ep_b_loss}")
                
            av_kls = torch_ext.mean_list(ep_kls)
            if self.multi_gpu:
                dist.all_reduce(av_kls, op=dist.ReduceOp.SUM)
                av_kls /= self.world_size
            if self.schedule_type == 'standard':
                self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
                self.update_lr(self.last_lr)

            kls.append(av_kls)
            self.diagnostics.mini_epoch(self, mini_ep)
            if self.normalize_input:
                self.model.running_mean_std.eval() # don't need to update statstics more than one miniepoch

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


    def train(self):
        self.init_tensors()
        self.last_mean_rewards = -100500
        start_time = time.perf_counter()
        total_time = 0
        rep_count = 0
        self.obs = self.env_reset()
        # print(f"obs_dict: {self.obs.keys()}")
        # print(f"second time env reset...") # 
        # self.obs = self.env_reset()
        self.curr_frames = self.batch_size_envs

        if self.multi_gpu:
            torch.cuda.set_device(self.local_rank)
            print("====================broadcasting parameters")
            model_params = [self.model.state_dict()]
            dist.broadcast_object_list(model_params, 0)
            self.model.load_state_dict(model_params[0])

        while True:
            epoch_num = self.update_epoch()
            step_time, play_time, update_time, sum_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul = self.train_epoch()
            total_time += sum_time
            frame = self.frame // self.num_agents

            # cleaning memory to optimize space
            self.dataset.update_values_dict(None)
            should_exit = False

            if self.global_rank == 0:
                self.diagnostics.epoch(self, current_epoch = epoch_num)
                # do we need scaled_time? # train epoch #
                scaled_time = self.num_agents * sum_time
                scaled_play_time = self.num_agents * play_time
                curr_frames = self.curr_frames * self.world_size if self.multi_gpu else self.curr_frames
                self.frame += curr_frames

                print_statistics(self.print_stats, curr_frames, step_time, scaled_play_time, scaled_time, 
                                epoch_num, self.max_epochs, frame, self.max_frames)

                self.write_stats(total_time, epoch_num, step_time, play_time, update_time,
                                a_losses, c_losses, entropies, kls, last_lr, lr_mul, frame,
                                scaled_time, scaled_play_time, curr_frames)

                if len(b_losses) > 0:
                    self.writer.add_scalar('losses/bounds_loss', torch_ext.mean_list(b_losses).item(), frame)

                if self.has_soft_aug:
                    self.writer.add_scalar('losses/aug_loss', np.mean(aug_losses), frame)

                if self.game_rewards.current_size > 0:
                    mean_rewards = self.game_rewards.get_mean()
                    mean_shaped_rewards = self.game_shaped_rewards.get_mean()
                    mean_lengths = self.game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]
                    print('mean_rewards: ', mean_rewards, 'last_mean_rewards: ', self.last_mean_rewards)
                    for i in range(self.value_size):
                        rewards_name = 'rewards' if i == 0 else 'rewards{0}'.format(i)
                        self.writer.add_scalar(rewards_name + '/step'.format(i), mean_rewards[i], frame)
                        self.writer.add_scalar(rewards_name + '/iter'.format(i), mean_rewards[i], epoch_num)
                        self.writer.add_scalar(rewards_name + '/time'.format(i), mean_rewards[i], total_time)
                        self.writer.add_scalar('shaped_' + rewards_name + '/step'.format(i), mean_shaped_rewards[i], frame)
                        self.writer.add_scalar('shaped_' + rewards_name + '/iter'.format(i), mean_shaped_rewards[i], epoch_num)
                        self.writer.add_scalar('shaped_' + rewards_name + '/time'.format(i), mean_shaped_rewards[i], total_time)

                    self.writer.add_scalar('episode_lengths/step', mean_lengths, frame)
                    self.writer.add_scalar('episode_lengths/iter', mean_lengths, epoch_num)
                    self.writer.add_scalar('episode_lengths/time', mean_lengths, total_time)

                    if self.has_self_play_config:
                        self.self_play_manager.update(self)

                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) + '_rew_' + str(mean_rewards[0])

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

                    if mean_rewards[0] > self.last_mean_rewards and epoch_num >= self.save_best_after:
                        print('saving next best rewards: ', mean_rewards)
                        self.last_mean_rewards = mean_rewards[0]
                        self.save(os.path.join(self.nn_dir, self.config['name']))
                        
                        # NOTE: add the log to log best rewards and the per-epoch reward
                        logging_fn = os.path.join(self.nn_dir, "logs.txt")
                        # logging_fn #
                        with open(logging_fn, "a") as wf:
                            wf.write(f"epoch: {epoch_num}, mean_rewards: {mean_rewards[0]}\n")
                            wf.close()

                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                if epoch_num >= self.max_epochs and self.max_epochs != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max epochs reached before any env terminated at least once')
                        mean_rewards = -np.inf # mean rewards # 

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_ep_' + str(epoch_num) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX EPOCHS NUM!')
                    should_exit = True

                if self.frame >= self.max_frames and self.max_frames != -1:
                    if self.game_rewards.current_size == 0:
                        print('WARNING: Max frames reached before any env terminated at least once')
                        mean_rewards = -np.inf

                    self.save(os.path.join(self.nn_dir, 'last_' + self.config['name'] + '_frame_' + str(self.frame) \
                        + '_rew_' + str(mean_rewards).replace('[', '_').replace(']', '_')))
                    print('MAX FRAMES NUM!')
                    should_exit = True

                update_time = 0


            
            if self.use_multiple_teacher:
                if (epoch_num + 1) % 5 == 0:
                    self.cur_teacher_idx = (self.cur_teacher_idx + 1) % self.nn_teacher

            if self.multi_gpu:
                should_exit_t = torch.tensor(should_exit, device=self.device).float()
                dist.broadcast(should_exit_t, 0)
                should_exit = should_exit_t.float().item()
            if should_exit:
                return self.last_mean_rewards, epoch_num

            if should_exit:
                return self.last_mean_rewards, epoch_num


    def _build_gt_act_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        # add the 
        
        self.experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (self.nn_act_dims + 1,),
                                                                    device=self.ppo_device)
        
        
        
        # if len(self.grab_obj_type_to_opt_res_fn) > 0:
        #     self.experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (23,),
        #                                                             device=self.ppo_device)
        # else:
        #     self.experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (22,),
        #                                                                 device=self.ppo_device)
        
        # amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        # self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        # self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        # replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        # self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['gt_act']
        
        if self.masked_mimic_training:
            self.experience_buffer.tensor_dict['mimic_teacher_obs'] = torch.zeros(batch_shape + (self.obs_shape[0], ),
                                                                    device=self.ppo_device)
            self.tensor_list += ['mimic_teacher_obs']
        return

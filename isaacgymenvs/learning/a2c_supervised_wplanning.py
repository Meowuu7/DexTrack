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
import gym

from rl_games.algos_torch import a2c_continuous

from isaacgymenvs.utils.torch_jit_utils import to_torch

import time
from datetime import datetime
import numpy as np
from torch import optim
import torch 
from torch import nn

from rl_games.algos_torch.moving_mean_std import GeneralizedMovingStats
import isaacgymenvs.learning.replay_buffer as replay_buffer
import isaacgymenvs.learning.common_agent as common_agent 
from rl_games.common import datasets
from rl_games.common import common_losses
from rl_games.algos_torch import  model_builder
from rl_games.algos_torch import central_value

from tensorboardX import SummaryWriter

from rl_games.common.experience import ExperienceBuffer
# common agent torch #
import os


# the question of # if the world model 
# if the world model can learn useful things w.r.t. (current state, tracking targets) -> actual arrived states;  then it can be used to give the forecasting model correct guidances # 
# one thing is that if we regard this as the tracking problem, and we want that the controller can sucessfully track the forecasted frames, then we should use  the world modle to guide the foreacasting model to predict exactly the current tracking targets #
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

def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
class A2CSupervisedAgentWForecasting(a2c_continuous.A2CAgent):

    def __init__(self, base_name, params): # supervised with forecasting #
        #  supervised agenet #
        super().__init__(base_name, params)
        # if self.normalize_value:
        #     self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        # if self._normalize_amp_input:
        #     self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        
        
        # mode=regular: then we should combine the supervised loss with the regular rl loss for training the actor and critic together
        # mode=offline_supervised: only train the actor using the supervised loss # mode is the supervised #
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
        
        self.train_controller = params['config'].get('train_controller', False)
        self.train_forecasting_model = params['config'].get('train_forecasting_model', True)
        
        self.forecasting_model_weight_fn = params['config'].get('forecasting_model_weight_fn', '')
        
        self.forecasting_obs_with_original_obs = params['config'].get('forecasting_obs_with_original_obs', False)
        
        self.maxx_inst_nn = params['config'].get('maxx_inst_nn', 100000)
        self.tuning_single_instance = True if self.maxx_inst_nn == 1 else False
        
        self.optimizing_with_teacher_net = len(self.teacher_model_path) > 0 and os.path.exists(self.teacher_model_path)
        print(f"train_controller: {self.train_controller}")
        print(f"train_forecasting_model: {self.train_forecasting_model}")
        
        
        """ Forecasting model settings """
        self.w_forecasting_model = True
        self.forecasting_obs_dim = params['config'].get('forecasting_obs_dim', 797)
        if self.tuning_single_instance:
            # self.forecasting_obs_dim =  797 - 256 - 512
            self.forecasting_obs_dim =  (797 - 256 - 512) * 2
        else:
            # self.forecasting_obs_dim = 797 
            # self.forecasting_obs_dim = 22 + 3 + 4 + 22 + (22 + 3 + 4) + 256
            self.forecasting_obs_dim = 22 + 3 + 4 + (22 + 3 + 4) + (22 + 3 + 4) + 256
            if self.forecasting_obs_with_original_obs:
                self.forecasting_obs_dim = 22 + 3 + 4 + (22 + 3 + 4) + (22 + 3 + 4) + self.obs_shape[0] # 
                print(f"forecasting_obs_dim: {self.forecasting_obs_dim}")
            
        self.forecasting_act_dim = params['config'].get('forecasting_act_dim', 29)
        self.forecasting_nn_frames = params['config'].get('forecasting_nn_frames', 10)
        self.forecasting_nn_frames = 2
        self.forecasting_act_dim_per_frame = self.forecasting_act_dim
        self.forecasting_act_dim = self.forecasting_act_dim * self.forecasting_nn_frames 
        self.actions_low = torch.cat(
            [
                self.actions_low, torch.zeros(self.forecasting_act_dim, device=self.ppo_device) - 1.0
            ], dim=-1
        )
        self.actions_high = torch.cat(
            [
                self.actions_high, torch.zeros(self.forecasting_act_dim, device=self.ppo_device) + 1.0
            ], dim=-1
        )
        
        if self.w_forecasting_model:
            self._init_forecasting_model()
            
        if self.has_central_value:
            cv_config = {
                'state_shape' : self.state_shape, # state shape # # state shape #
                'value_size' : self.value_size,
                'ppo_device' : self.ppo_device, 
                'num_agents' : self.num_agents, 
                'horizon_length' : self.horizon_length,
                'num_actors' : self.num_actors, 
                'num_actions' : self.actions_num, 
                'seq_length' : self.seq_length,
                'normalize_value' : self.normalize_value,
                'network' : self.central_value_config['network'],
                'config' : self.central_value_config, 
                'writter' : self.writer, # 
                'max_epochs' : self.max_epochs,
                'multi_gpu' : self.multi_gpu,
                'zero_rnn_on_done' : self.zero_rnn_on_done
            }
            self.forecasting_central_value_net = central_value.CentralValueTrain(**cv_config).to(self.ppo_device)
        
        self.forecasting_dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        if self.normalize_value:
            self.value_mean_std_forecasting = self.central_value_net.model.value_mean_std if self.has_central_value else self.forecasting_model.value_mean_std
        self.forecasting_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        
        if self.normalize_advantage and self.normalize_rms_advantage:
            momentum = self.config.get('adv_rms_momentum', 0.5)
            self.advantage_mean_std_forecasting = GeneralizedMovingStats((1,), momentum=momentum).to(self.ppo_device)
        self.forecasting_game_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.forecasting_game_shaped_rewards = torch_ext.AverageMeter(self.value_size, self.games_to_track).to(self.ppo_device)
        self.forecasting_game_lengths = torch_ext.AverageMeter(1, self.games_to_track).to(self.ppo_device)
        """ Forecasting model settings """
        
        self.use_world_model = params['config'].get('use_world_model', False)
        if self.use_world_model:
            self._init_world_model()
            pass
        
        
        """ Mask mimicing teacher model """ 
        self.masked_mimic_training = params['config'].get('masked_mimic_training', False)
        self.masked_mimic_teacher_model_path = params['config'].get('masked_mimic_teacher_model_path', '')
        if self.masked_mimic_training: # masked mimic training #
            self._init_mask_mimic_teacher_model()
        """ Mask mimicing teacher model """ 
        
        """ Multiple teacher models """
        self.teacher_index_to_weights_fn = params['config'].get('teacher_index_to_weights', '')
        self.use_multiple_teacher = False
        if self.use_teacher_model and len(self.teacher_index_to_weights_fn) > 0 and os.path.exists(self.teacher_index_to_weights_fn):
            self.teacher_index_to_weights = np.load(self.teacher_index_to_weights_fn, allow_pickle=True).item()
            self.use_multiple_teacher = True
            self.optimizing_with_teacher_net = True
            self.nn_teacher = len(self.teacher_index_to_weights)
            # use mulitple teacher # 
            self._init_multiple_teacher_models()
            
            self.cur_teacher_idx = 0
        """ Multiple teacher models """
        
        ##### init teacher models #####
        # hand dof; object pos, object orientation, with the action -> (world model) -> the next state #
        # construct these tuples # tuple # construct these tuples # -> train the world model in the calc_gradient function #
        # in the first step, we can use the MLP as the world model architecture #
        # forecastng obs -- it has sates --- after that we can have actions from the pre-trained controller #
        # with the three types of inputs, we can get the final transited states # transited states # # transited states # #
        
        """ Single teacher model """
        if self.use_teacher_model and self.optimizing_with_teacher_net and (not self.use_multiple_teacher):
            self._init_teacher_models_single()
        """ Single teacher model """
        
        
        if self.w_franka:
            
            self.nn_act_dims = 23
        else:
            self.nn_act_dims = 22
            
        return
    
    
    def set_weights(self, weights):
        # if not (self.train_controller and not self.train_forecasting_model):
        print(f"loading the model weights")
        self.model.load_state_dict(weights['model'])
        if 'forecasting_model' in weights:
            print(f"Loading the forecasting model")
            self.forecasting_model.load_state_dict(weights['forecasting_model'])
        if 'world_model' in weights:
            self.world_model.load_state_dict(weights['world_model'])
        if not (self.train_controller and not self.train_forecasting_model):
            self.set_stats_weights(weights)
        
        
    def set_full_state_weights(self, weights, set_epoch=True):

        self.set_weights(weights)
        if set_epoch:
            self.epoch_num = weights['epoch']
            self.frame = weights['frame']

        if self.has_central_value:
            self.central_value_net.load_state_dict(weights['assymetric_vf_nets'])

        if not (self.train_controller and not self.train_forecasting_model):
            self.optimizer.load_state_dict(weights['optimizer'])

        self.last_mean_rewards = weights.get('last_mean_rewards', -1000000000)

        if self.vec_env is not None:
            env_state = weights.get('env_state', None)
            self.vec_env.set_env_state(env_state)

    
    
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
            'num_seqs' : self.num_actors * self.num_agents, # init 
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input, # # normalize # #
        }
        cur_obj_model = self.teacher_network_builder.build(cur_obj_model_build_config).to(self.ppo_device)
        
        
        
        print(f"loading the teacher mode from: {cur_optimized_res}")
        cur_optimized_res_wieghts = torch.load(cur_optimized_res, map_location='cpu')
        cur_obj_model.load_state_dict(cur_optimized_res_wieghts['model'])
        cur_obj_model.eval()
        
        self.teacher_model = cur_obj_model
        
        
        
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
        
    def _init_forecasting_model(self, ):
        assert self.w_forecasting_model
        
        obj_feat_shape = 256
        
        self.forecasting_network_params = self.params.copy()
        if self.tuning_single_instance:
            self.forecasting_network_params['network']['mlp']['units'] = [1024, 512, 256, 128] 
        else:
            print(f"Not tuning single instance!")
            self.forecasting_network_params['network']['mlp']['units'] = [8192, 4096, 2048, 1024, 512, 256, 128]
        
        self.forecasting_network_builder = model_builder.ModelBuilder()
        self.forecasting_network_builder = self.forecasting_network_builder.load(self.forecasting_network_params)
        
        forecasting_model_build_config = {
            'actions_num' : self.forecasting_act_dim,
            'input_shape' : (self.forecasting_obs_dim, ),
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        } # and also the changes in the world model #
        forecasting_model = self.forecasting_network_builder.build(forecasting_model_build_config).to(self.ppo_device)
        # teacher_model_weights = torch.load(teacher_model_weight_fn, map_location='cpu')
        # teacher_model.load_state_dict(teacher_model_weights['model'])
        forecasting_model.train()
        self.forecasting_model = forecasting_model
        
        self.forecasting_states = None
        self.init_rnn_from_model(self.forecasting_model)
        # self.last_lr = float(self.last_lr)
        # self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        self.forecasting_optimizer = optim.Adam(self.forecasting_model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        
        if len(self.forecasting_model_weight_fn) > 0 and os.path.exists(self.forecasting_model_weight_fn):
            forecasting_model_weight = torch.load(self.forecasting_model_weight_fn, map_location='cpu')
            print(f"Initializing forecasting model weights from {self.forecasting_model_weight_fn}")
            self.forecasting_model.load_state_dict(forecasting_model_weight['forecasting_model'])
            
    
    def _init_world_model(self, ):
        assert self.use_world_model
        
        ## TODO: add world model initialization ##
        ## TODO: add world model optimizer ##
        ## TODO: add wordl model dataset ##
        ## TODO: what's the output obs dim? it should be the same with forecasting obs dim #
        
        
        act_dim = 22 + 3 + 4
        
        if self.tuning_single_instance:
            self.world_model = nn.Sequential(
                nn.Linear(self.forecasting_obs_dim + act_dim, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, self.forecasting_obs_dim)
            ).to(self.ppo_device)
        else:
            self.world_model = nn.Sequential(
                nn.Linear(self.forecasting_obs_dim + act_dim, 8192), nn.ReLU(),
                nn.Linear(8192, 4096), nn.ReLU(),
                nn.Linear(4096, 2048), nn.ReLU(),
                nn.Linear(2048, 1024), nn.ReLU(),
                nn.Linear(1024, 512), nn.ReLU(),
                nn.Linear(512, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, self.forecasting_obs_dim)
            ).to(self.ppo_device)
        
        
        self.world_model.train()
        self.world_model_optimizer = optim.Adam(self.world_model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        
        ## TODO: add the world model dataset ##
        ## TODO: add the world model dataset ##
        
        self.world_model_dataset = datasets.PPODataset(self.batch_size, self.minibatch_size, self.is_discrete, self.is_rnn, self.ppo_device, self.seq_length)
        
        self._init_world_model_experience_buffer()
        # TODO: add the play step and the training code for the world model #
        # #
    
    
    def _init_world_model_experience_buffer(self, ): 
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks,
        }
        _forecasting_act_dim = self.forecasting_act_dim // self.forecasting_nn_frames
        self.world_model_env_info = self.env_info.copy()
        self.world_model_env_info['observation_space'] = gym.spaces.Box(low=-1, high=1,shape=(self.forecasting_obs_dim, ), dtype=np.float32) 
        # tuple(list(self.obs_shape)[:-1] + [self.forecasting_obs_dim]) # add action space observation space, and the next observation space
        self.world_model_env_info['action_space'] =gym.spaces.Box(low=-1, high=1,shape=(_forecasting_act_dim, ), dtype=np.float32) 
        # self.world_model_env_info['next_observation_space'] = gym.spaces.Box(low=-1, high=1,shape=(self.forecasting_obs_dim, ), dtype=np.float32)
        self.world_model_experience_buffer = ExperienceBuffer(self.world_model_env_info, algo_info, self.ppo_device)
        
        self.world_model_experience_buffer.tensor_dict['next_obses'] = torch.zeros( self.world_model_experience_buffer.obs_base_shape + (self.forecasting_obs_dim, ), device=self.ppo_device)
        
        # world model experience buffer #
        # reset no larger than 1000 is still not enough... # # add the world model experience buffer #
        # # val_shape = (self.horizon_length, batch_size, self.value_size)
        # current_rewards_shape = (batch_size, self.value_size)
        # self.f_current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        # self.forecasting_current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        # self.forecasting_current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        # self.forecasting_dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        # self.forecasting_rnn_states = None
        # if self.is_rnn:
        #     self.forecasting_rnn_states = self.forecasting_model.get_default_rnn_state()
        #     self.forecasting_rnn_states = [s.to(self.ppo_device) for s in self.forecasting_rnn_states]

        #     total_agents = self.num_agents * self.num_actors
        #     num_seqs = self.horizon_length // self.seq_length
        #     assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
        #     self.forecasting_mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.forecasting_rnn_states]

        # batch_shape = self.experience_buffer.obs_base_shape
        # self.forecasting_experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (self.forecasting_act_dim, ),
        #                                                             device=self.ppo_device)
        
        
        
    
    
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
    
    


    def init_tensors(self):
        super().init_tensors()
        self._build_gt_act_buffers()
        # if self.preload_experiences_tf:
        #     self.preload_saved_experiences()
        
        return
    
    
    
    def prepare_dataset_forecasting(self, batch_dict):
        rnn_masks = batch_dict.get('rnn_masks', None)
        returns = batch_dict['returns']
        values = batch_dict['values']
        actions = batch_dict['actions']
        neglogpacs = batch_dict['neglogpacs']
        dones = batch_dict['dones']
        mus = batch_dict['mus']
        sigmas = batch_dict['sigmas']
        rnn_states = batch_dict.get('rnn_states', None)
        rnn_masks = batch_dict.get('rnn_masks', None)
        
        obses = batch_dict['obses']
        advantages = returns - values

        if self.normalize_value:
            self.value_mean_std_forecasting.train()
            values = self.value_mean_std_forecasting(values)
            returns = self.value_mean_std_forecasting(returns)
            self.value_mean_std_forecasting.eval()
        
        advantages = torch.sum(advantages, axis=1)

        if self.normalize_advantage:
            if self.is_rnn:
                if self.normalize_rms_advantage:
                    advantages = self.advantage_mean_std(advantages, mask=rnn_masks)
                else:
                    advantages = torch_ext.normalization_with_masks(advantages, rnn_masks)
            else:
                if self.normalize_rms_advantage: # the adtanges 
                    advantages = self.advantage_mean_std_forecasting(advantages)
                else:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        dataset_dict = {}
        dataset_dict['old_values'] = values
        dataset_dict['old_logp_actions'] = neglogpacs
        dataset_dict['advantages'] = advantages
        dataset_dict['returns'] = returns
        dataset_dict['actions'] = actions
        dataset_dict['obs'] = obses
        dataset_dict['dones'] = dones
        dataset_dict['rnn_states'] = rnn_states
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['rnn_masks'] = rnn_masks
        dataset_dict['mu'] = mus
        dataset_dict['sigma'] = sigmas


        # if self.use_action_masks:
        #     dataset_dict['action_masks'] = batch_dict['action_masks']

        # update the dataset 
        self.forecasting_dataset.update_values_dict(dataset_dict)
        if self.has_central_value:
            dataset_dict = {}
            dataset_dict['old_values'] = values
            dataset_dict['advantages'] = advantages
            dataset_dict['returns'] = returns
            dataset_dict['actions'] = actions
            dataset_dict['dones'] = dones
            dataset_dict['obs'] = batch_dict['states'] 
            dataset_dict['rnn_masks'] = rnn_masks
            self.forecasting_central_value_net.update_dataset(dataset_dict)
        self.forecasting_dataset.values_dict['gt_act'] = batch_dict['gt_act']
    
    def prepare_dataset_world_model(self, batch_dict):
        self.world_model_dataset.update_values_dict(batch_dict)
    
    
    
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['gt_act'] = batch_dict['gt_act']
        if self.masked_mimic_training:
            self.dataset.values_dict['mimic_teacher_obs'] = batch_dict['mimic_teacher_obs']
        # if 'fore' # super # env set train info  # # state with those tracking targets -> 
        # only usingthe hand conditions whtile the object # # 

    # 


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
        with torch.no_grad():
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
    
    
    def get_forecasting_action_values(self, obs):
        processed_obs = self._preproc_obs(obs['forecasting_obs'][..., : self.forecasting_obs_dim])
        self.forecasting_model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., :],
            'rnn_states' : self.forecasting_rnn_states
        }

        # get the actions values #
        with torch.no_grad():
            res_dict = self.forecasting_model(input_dict)
            if self.has_central_value:
                states = obs['forecasting_states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
            # print(f"res_dict: {res_dict.}") # forecasting model #
        return res_dict

    def get_values(self, obs):
        with torch.no_grad():
            if self.has_central_value:
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
        
        
    def get_forecasting_values(self, obs):
        with torch.no_grad():
            if self.has_central_value: # has central value #
                states = obs['forecasting_states']
                self.forecasting_central_value_net.eval()
                input_dict = {
                    'is_train': False,
                    'states' : states,
                    'actions' : None,
                    'is_done': self.dones,
                }
                value = self.get_central_value(input_dict)
            else:
                self.forecasting_model.eval()
                processed_obs = self._preproc_obs(obs['forecasting_obs'][..., : self.forecasting_obs_dim])
                input_dict = {
                    'is_train': False,
                    'prev_actions': None, 
                    'obs' : processed_obs[..., : ],
                    'rnn_states' : self.forecasting_rnn_states
                }
                result = self.forecasting_model(input_dict)
                value = result['values']
            return value
    
    
    
    
    def play_steps(self):
        
        
        
        update_list = self.update_list

        step_time = 0.0
        
        # print(f"self.actions_low: {self.actions_low}, self.actions_high: {self.actions_high}")

        for n in range(self.horizon_length): 
            # print(f"self.obs: {self.obs['obs'].size()}, self.obs_shape: {self.obs_shape}")
            
            # if self.masked_mimic_training:
            #     mimic_teacher_obs = self.obs['mimic_teacher_obs'][..., : ] # get the last obs features as teacher features here #
            #     # self.obs['obs'] = self.obs['obs'][..., : - self.obs_shape[0]]
            #     self.experience_buffer.update_data('mimic_teacher_obs', n, mimic_teacher_obs)
            
            gt_act_val = self.obs['obs'][..., self.obs_shape[0]: ]
            
            
            
            self.experience_buffer.update_data('gt_act', n, gt_act_val) # find the detla qpos of  the #
            # shadow hand qpos and the cur hand qpos ref #
            
            
            
            # and is not aims at "track" the arbitrary hand command or the arbitrary obj position commands #
            # at each tiemstep, use the current obs to get the # get the action values, obs, #
            # delta are calculated based on the previous 
            
            
            # if self.use_action_masks:
            #     masks = self.vec_env.get_action_masks()
            #     res_dict = self.get_masked_action_values(self.obs, masks)
            # else: # res dict #  # 
            #     res_dict = self.get_action_values(self.obs)
            
            
            
            # print("foecasting_obs:", self.obs['forecasting_obs'].size())
            # print(f"forecasting_obs_dim: {self.forecasting_obs_dim}")
                
            forecasting_res_dict = self.get_forecasting_action_values(self.obs)
            
            
            
            
            if self.train_forecasting_model:
            
                tmp_forecasting_actions = forecasting_res_dict['actions']
                ## update obs via the forecasted actions #
                # tmp_forecasting_actions: nn_envs x (nn_act_dim x nn_futruer_obs)
                tmp_forecasting_actions = tmp_forecasting_actions.contiguous().view(tmp_forecasting_actions.size(0), -1, self.forecasting_act_dim_per_frame)
                hand_dof_nn = self.vec_env.env.num_shadow_hand_dofs
                tmp_forecasting_actions = torch.cumsum(tmp_forecasting_actions, dim=1) # 
                
                cur_step_forecasting_actions = tmp_forecasting_actions[:, 0, :][..., :hand_dof_nn]
                nex_step_forecasting_actions = tmp_forecasting_actions[:, 1, :][..., :hand_dof_nn]
                
                cur_step_goal_obj_pos = tmp_forecasting_actions[:, 0, :][..., hand_dof_nn: hand_dof_nn + 3]
                
                cur_step_hand_dof_pos = self.obs['forecasting_obs'][..., : hand_dof_nn] # nn_envs x nn_hand_dof 
                shadow_hand_dof_speed_scale_tsr = self.vec_env.env.shadow_hand_dof_speed_scale_tsr # nn_hand_dof
                ### TODO: change it to the accumulated version ###
                ### TODO: initilaize the pred_targets --- the first frame should be the first frame of the original kine refs #
                cur_step_forecasting_actions = cur_step_forecasting_actions * shadow_hand_dof_speed_scale_tsr * self.vec_env.env.dt
                nex_step_forecasting_actions = nex_step_forecasting_actions *   shadow_hand_dof_speed_scale_tsr * self.vec_env.env.dt
                cur_step_forecasting_actions = cur_step_forecasting_actions + cur_step_hand_dof_pos # n_envs x nn_hand_dof
                nex_step_forecasting_actions = nex_step_forecasting_actions + cur_step_hand_dof_pos
                hand_goal_start = int(self.obs['hand_goal_start'][0].item())
                # print(f"[forecaster] hand_goal_start: {hand_goal_start}, hand_dof_nn: {hand_dof_nn}, shadow_hand_dof_speed_scale_tsr: {shadow_hand_dof_speed_scale_tsr}, dt: {self.vec_env.env.dt}")
                delta_qpos = cur_step_hand_dof_pos - cur_step_forecasting_actions
                
                # if self.vec_env.env.use_future_ref_as_obs_goal:
                #     delta_qpos = cur_step_hand_dof_pos - nex_step_forecasting_actions

                # print(f"Modifying the obs with forecasting actions")
                self.obs['obs'][..., hand_goal_start + 7 : hand_goal_start + 7 + hand_dof_nn] = delta_qpos
                
                ###### replace the nex ref ######
                nex_ref_start = int(self.obs['hand_goal_start'][1].item())
                if self.vec_env.env.use_local_canonical_state: # nex_hand_qpos_ref #
                    canon_hand_qpos_trans = nex_step_forecasting_actions[..., :3] - self.vec_env.env.object_pos
                    canon_hand_qpos_ref = torch.cat(
                        [ canon_hand_qpos_trans, nex_step_forecasting_actions[..., 3:] ], dim=-1
                    )
                else:
                    canon_hand_qpos_ref = nex_step_forecasting_actions
                
                unscaled_nex_hand_qpos_ref = unscale(canon_hand_qpos_ref, self.vec_env.env.shadow_hand_dof_lower_limits, self.vec_env.env.shadow_hand_dof_upper_limits)
                
                # self.obs['obs'][..., nex_ref_start : nex_ref_start + hand_dof_nn] = unscaled_nex_hand_qpos_ref
                
                if self.vec_env.env.forecast_obj_pos:
                    # print(f"Modifying the obs with forecasting actions")
                    ######### Replace the object goal pos ############
                    obj_obs_start = int(self.obs['hand_goal_start'][2].item())
                    cur_step_goal_obj_pos = cur_step_goal_obj_pos * shadow_hand_dof_speed_scale_tsr.unsqueeze(0)[..., :3] * self.vec_env.env.dt
                    self.obs['obs'][..., obj_obs_start + 13:obj_obs_start + 16] = cur_step_goal_obj_pos #  self.goal_pos - self.object_pos
                    ######### Replace the object goal pos ############
            
            
            
            
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else: # res dict #
                res_dict = self.get_action_values(self.obs)
            
            # things regarding both the delta_qpos 
            # you should cahnge self.obs['obs] 
            # 
            # compute the shadow current dof from the forecasting_obs #
            # compute the 
            
            # res_dict['actions'] = res_dict['mus']
            # if self.epoch_num % 10 == 0: #
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
            
            # if self.training_mode == 'offline_supervised' or self.pure_supervised_training:
            
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
            
            # if 'obs' in res_dict:
            #     res_dict['obs'] = res_dict['obs'][..., : self.obs_shape[0]]
            
            ##### Update necessary values in the experience buffer #####
            self.experience_buffer.update_data('obses', n, self.obs['obs'][..., : self.obs_shape[0]])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list:
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])
            ##### Update necessary values in the experience buffer #####
            
            
            ##### Update necessary values in the experience buffer -- forecasting model #####
            self.forecasting_experience_buffer.update_data('obses', n, self.obs['forecasting_obs'][..., : self.forecasting_obs_dim])
            self.forecasting_experience_buffer.update_data('dones', n, self.dones)
            
            forecasting_gt_act = self.obs['forecasting_obs'][..., -self.forecasting_act_dim:  ]
            self.forecasting_experience_buffer.update_data('gt_act', n, forecasting_gt_act)

            for k in update_list: # 
                self.forecasting_experience_buffer.update_data(k, n, forecasting_res_dict[k]) 
            if self.has_central_value:
                self.forecasting_experience_buffer.update_data('states', n, self.obs['forecasting_states'])
            ##### Update necessary values in the experience buffer -- forecasting model #####
            
            if self.use_world_model:
                self.world_model_experience_buffer.update_data('obses', n, self.obs['forecasting_obs'][..., : self.forecasting_obs_dim])
                controller_actions = self.obs['nex_tracking_targets'] # nn_envs x next tracking targes #
                self.world_model_experience_buffer.update_data('actions', n, controller_actions)
                
                #### get the next obses ####
                if n >= 1:
                    # print(self.obs['forecasting_obs'][..., : self.forecasting_obs_dim].size(), self.world_model_experience_buffer.tensor_dict['next_obses'].size())
                    self.world_model_experience_buffer.update_data('next_obses', n - 1, self.obs['forecasting_obs'][..., : self.forecasting_obs_dim])
            
            ## TODO: add it and try it ##
            step_time_start = time.perf_counter()
            # we need to cat these two actions? # 
            policy_actions = res_dict['actions']
            
            if self.use_world_model:
                forecasting_actions = forecasting_res_dict['mus']
            else:
                forecasting_actions = forecasting_res_dict['actions']
            
            # forecasting_actions = forecasting_res_dict['mus']
            
            cat_actions = torch.cat(
                [ policy_actions, forecasting_actions ], dim=-1
            )
            self.obs, rewards, self.dones, infos = self.env_step(cat_actions)
            step_time_end = time.perf_counter()

            step_time += (step_time_end - step_time_start)
            
            
            ## TODO:  forecasting_reward -- should be set in the observation dict ##
            
            
            """ Calculate rewards and update rewards in the buffer """
            shaped_rewards = self.rewards_shaper(rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                # print("values:", res_dict['values'].size(), "; time_outs:", self.cast_obs(infos['time_outs']).size())
                shaped_rewards += self.gamma * res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            
            self.experience_buffer.update_data('rewards', n, shaped_rewards)

            self.current_rewards += rewards
            self.current_shaped_rewards += shaped_rewards
            self.current_lengths += 1
            
            all_done_indices = self.dones.nonzero(as_tuple=False)
            env_done_indices = all_done_indices[::self.num_agents]

            """ Controller's reward """
            self.game_rewards.update(self.current_rewards[env_done_indices])
            self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            self.game_lengths.update(self.current_lengths[env_done_indices])
            self.algo_observer.process_infos(infos, env_done_indices)
            
            not_dones = 1.0 - self.dones.float()

            self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            self.current_lengths = self.current_lengths * not_dones
            """ Calculate rewards and update rewards in the buffer """
            
            """ Calculate rewards and update rewards in the buffer -- forecasting model """
            forecasting_rewards = self.obs['forecasting_rewards'] # get the forecasting rewards # # get 
            # print(f"rewards: {rewards.size()}, forecasting_rewards: {forecasting_rewards.size()}")
            forecasting_shaped_rewards = self.rewards_shaper(forecasting_rewards)
            if self.value_bootstrap and 'time_outs' in infos:
                # print("values:", forecasting_res_dict['values'].size(), "; time_outs:", self.cast_obs(infos['time_outs']).size())
                forecasting_shaped_rewards += self.gamma * forecasting_res_dict['values'] * self.cast_obs(infos['time_outs']).unsqueeze(1).float()
            self.forecasting_experience_buffer.update_data('rewards', n, forecasting_shaped_rewards)
            
            self.forecasting_current_rewards += forecasting_rewards
            self.forecasting_current_shaped_rewards += forecasting_shaped_rewards
            self.forecasting_current_lengths += 1
            
            """ Forecasting model's reward """
            self.forecasting_game_rewards.update(self.forecasting_current_rewards[env_done_indices])
            self.forecasting_game_shaped_rewards.update(self.forecasting_current_shaped_rewards[env_done_indices])
            self.forecasting_game_lengths.update(self.forecasting_current_lengths[env_done_indices])
            # self.algo_observer.process_infos(infos, env_done_indices)
            
            self.forecasting_current_rewards = self.forecasting_current_rewards * not_dones.unsqueeze(1)
            self.forecasting_current_shaped_rewards = self.forecasting_current_shaped_rewards * not_dones.unsqueeze(1)
            self.forecasting_current_lengths = self.forecasting_current_lengths * not_dones
            """ Calculate rewards and update rewards in the buffer -- forecasting model """
            
        if self.use_world_model:
            cur_forecasting_obs = self.obs['forecasting_obs'][..., : self.forecasting_obs_dim]
            self.world_model_experience_buffer.update_data('next_obses', self.horizon_length - 1, cur_forecasting_obs)
            world_model_tensor_list = ['obses', 'actions', 'next_obses']
            world_model_batch_dict = self.world_model_experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, world_model_tensor_list)
        
        # add get value for the forecasting model #
        ### TODO: add get value for the forecasting model ###
        
        """ Last values and calculate sometimes  """
        last_values = self.get_values(self.obs)
        
        # if self.training_mode == 'offline_supervised' or self.pure_supervised_training:
        #     #### version 2 for creating the expert demonstration data ####
        #     last_values = torch.ones_like(last_values) # 
        #     #### version 2 for creating the expert demonstration data ####
        
        fdones = self.dones.float()
        mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
        mb_values = self.experience_buffer.tensor_dict['values']
        mb_rewards = self.experience_buffer.tensor_dict['rewards']
        mb_advs = self.discount_values(fdones, last_values, mb_fdones, mb_values, mb_rewards)
        mb_returns = mb_advs + mb_values
        """ Last values and calculate sometimes  """
        
        """ Last values and calculate sometimes  -- forecasting model """
        forecasting_last_values = self.get_forecasting_values(self.obs)
        # fdones = self.dones.float()
        forecasting_mb_fdones = self.forecasting_experience_buffer.tensor_dict['dones'].float()
        forecasting_mb_values = self.forecasting_experience_buffer.tensor_dict['values']
        forecasting_mb_rewards = self.forecasting_experience_buffer.tensor_dict['rewards']
        forecasting_mb_advs = self.discount_values(fdones, forecasting_last_values, forecasting_mb_fdones, forecasting_mb_values, forecasting_mb_rewards)
        forecasting_mb_returns = forecasting_mb_advs + forecasting_mb_values
        """ Last values and calculate sometimes  -- forecasting model """

        
        """ Prepare the batch dict """
        batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
        batch_dict['played_frames'] = self.batch_size
        batch_dict['step_time'] = step_time
        """ Prepare the batch dict """
        
        """ Prepare the forecasting batch dict """
        forecasting_batch_dict = self.forecasting_experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
        forecasting_batch_dict['returns'] = a2c_common.swap_and_flatten01(forecasting_mb_returns)
        forecasting_batch_dict['played_frames'] = self.batch_size
        forecasting_batch_dict['step_time'] = step_time 
        
        

        
        # forecasting_batch_dict = {
        #     f"forecasting_{k}": forecasting_batch_dict[k] for k in forecasting_batch_dict
        # }
        
        print(f"forecasting_batch_dict: {forecasting_batch_dict.keys()}, batch_dict: {batch_dict.keys()}")
        """ Prepare the forecasting batch dict """
        
        
        batch_dict.update({'forecasting_dict': forecasting_batch_dict})
        # batch_dict.update(forecasting_batch_dict)
        
        if self.use_world_model:
            batch_dict.update({'world_model_dict': world_model_batch_dict})
        
        # print(f"batch_dict: {batch_dict.keys()}, ")
        # print(f"tensor_list: {self.tensor_list}")
        return batch_dict

    # actor loss  save the 
    def actor_loss_supervised(self, pred_actions, gt_actions):
        
        if pred_actions.size(-1) == gt_actions.size(-1):
            gt_actions = gt_actions[..., :]
            pred_actions = pred_actions[..., :]
            loss = torch.sum(
                torch.nn.functional.mse_loss(pred_actions[..., :], gt_actions, reduction='none'), dim=-1
            )
        else:
        
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
        
        
        if 'forecasting_obs' in input_dict:
            forecasting_value_preds_batch = input_dict['forecasting_old_values']
            forecasting_old_action_log_probs_batch = input_dict['forecasting_old_logp_actions']
            forecasting_advantage = input_dict['forecasting_advantages']
            forecasting_old_mu_batch = input_dict['forecasting_mu']
            forecasting_old_sigma_batch = input_dict['forecasting_sigma']
            forecasting_return_batch = input_dict['forecasting_returns']
            forecasting_actions_batch = input_dict['forecasting_actions']
            forecasting_obs_batch = input_dict['forecasting_obs']
            forecasting_act_gt_batch = input_dict['forecasting_gt_act']
            forecasting_obs_batch = self._preproc_obs(forecasting_obs_batch)
        else:
            forecasting_obs_batch = None
        # forecasting obs batch #
        
        
        
        gt_act_batch = input_dict['gt_act']
        
        # if self.masked_mimic_training:
        #     mimic_teacher_obs_batch = input_dict['mimic_teacher_obs']
        
        if self.use_teacher_model and self.optimizing_with_teacher_net:
            # obj_inst_tag = self.single_instance_tag
            # obj_teacher_model = self.inst_tag_to_model_struct[obj_inst_tag]
            # teacher_res_dict = self._get_teacher_action_values(input_dict, obj_teacher_model)
            # gt_act_batch = teacher_res_dict['actions'] 
            
            if self.use_multiple_teacher:
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

        # batch size #
        batch_dict = {
            'is_train': True,
            'prev_actions': actions_batch,
            'obs' : env_obs_batch,
        }
        
        
        if forecasting_obs_batch is not None:
            forecasting_batch_dict = {
                'is_train': True, #
                'prev_actions': forecasting_actions_batch,
                'obs' : forecasting_obs_batch, #
            }
        else:
            forecasting_batch_dict = None
        
        

        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict['rnn_masks']
            batch_dict['rnn_states'] = input_dict['rnn_states']
            batch_dict['seq_length'] = self.seq_length

            if self.zero_rnn_on_done:
                batch_dict['dones'] = input_dict['dones']            

        # use a neurla objective to get the actions? #
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            res_dict = self.model(batch_dict)
            action_log_probs = res_dict['prev_neglogp']
            values = res_dict['values'] 
            entropy = res_dict['entropy']
            mu = res_dict['mus']
            
            sigma = res_dict['sigmas']
            
            # sigma = torch.zeros_like(sigma)
            
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)


            if forecasting_batch_dict is not None:
                forecasting_res_dict = self.forecasting_model(forecasting_batch_dict)
                forecasting_action_log_probs = forecasting_res_dict['prev_neglogp']
                forecasting_values = forecasting_res_dict['values']
                forecasting_entropy = forecasting_res_dict['entropy']
                forecasting_mu = forecasting_res_dict['mus']
                forecasting_sigma = forecasting_res_dict['sigmas']
                forecasting_actions = torch.randn_like(forecasting_mu) * forecasting_sigma + forecasting_mu
                
                ## TODO: whether self.ppo need to be changed here? # # whether self.ppo need to be changed here? #
                # 
                forecasting_a_loss = self.actor_loss_func(forecasting_old_action_log_probs_batch, forecasting_action_log_probs, forecasting_advantage, self.ppo, curr_e_clip)
            
            
            
            # ####### Supervised loss Version 1 #######
            # rnd_noise = torch.randn_like(mu) * sigma
            # sampled_actions = mu + rnd_noise
            # supervised_a_loss = self.actor_loss_supervised(sampled_actions, gt_act_batch) # 
            # ####### Supervised loss Version 1 #######

            # ####### Supervised loss Version 2 ####### # supervised #
            # TODO: what's the role of rnns here? #
            # TODO: and also what's the role of rnn_masks? # # we need to tra ina modle taht fits the 
            if self.masked_mimic_training:
                supervised_a_loss = self.actor_loss_mimic_teacher(mu, gt_act_batch)
            else:
                supervised_a_loss = self.actor_loss_supervised(mu, gt_act_batch)
            # ####### Supervised loss Version 2 ####### # # supervised # # 
            
            
            # has value loss #
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


            if forecasting_batch_dict is not None:
                # train the actor ? #
                if self.has_value_loss:
                    forecasting_c_loss = common_losses.critic_loss(self.forecasting_model, forecasting_value_preds_batch, forecasting_values, curr_e_clip, forecasting_return_batch, self.clip_value)
                else:
                    forecasting_c_loss = torch.zeros(1, device=self.ppo_device)
                if self.bound_loss_type == 'regularisation':
                    forecasting_b_loss = self.reg_loss(forecasting_mu)
                elif self.bound_loss_type == 'bound':
                    forecasting_b_loss = self.bound_loss(forecasting_mu)
                else:
                    forecasting_b_loss = torch.zeros(1, device=self.ppo_device)
                forecasting_supervised_a_loss = self.actor_loss_supervised(forecasting_mu, forecasting_act_gt_batch)    
            
                # has value loss #
                if self.use_world_model:
                    nn_hand_dofs = 22
                    to_track_hand_obj_nn_dofs = 22 + 3 + 4
                    cur_hand_dof_state = forecasting_obs_batch[..., : nn_hand_dofs]
                    to_track_res = forecasting_act_gt_batch.contiguous().view(forecasting_act_gt_batch.size(0), -1, to_track_hand_obj_nn_dofs)
                    to_track_hand_state = to_track_res[..., : nn_hand_dofs] + cur_hand_dof_state.unsqueeze(1)
                    to_track_obj_pos = to_track_res[..., nn_hand_dofs: nn_hand_dofs + 3]
                    to_track_obj_ornt = to_track_res[..., nn_hand_dofs + 3: nn_hand_dofs + 7]
                    # pred_ # actions shoud be the absolute tracking targets? # # this action is not that actions ... # get that thing from the observations #
                    pred_to_track_res = forecasting_mu.contiguous().view(forecasting_mu.size(0), -1, to_track_hand_obj_nn_dofs)
                    pred_to_track_res = pred_to_track_res[:, 1] # nn_envs x to_track_hand_dof_nn-dofs
                    pred_to_track_res[..., : nn_hand_dofs] = pred_to_track_res[..., : nn_hand_dofs ] + cur_hand_dof_state # .unsqueeze(1)
                    in_world_model_batch = torch.cat(
                        [ forecasting_obs_batch , pred_to_track_res], dim=-1 # in world model batch 
                    )
                    world_model_pred = self.world_model(in_world_model_batch) # 
                    world_model_pred_tracked_hand_state = world_model_pred[..., : nn_hand_dofs]
                    world_model_pred_tracked_obj_pos = world_model_pred[..., nn_hand_dofs: nn_hand_dofs + 3]
                    world_model_pred_tracked_obj_ornt = world_model_pred[..., nn_hand_dofs + 3: nn_hand_dofs + 7]
                    
                    diff_hand_state = torch.norm(world_model_pred_tracked_hand_state - to_track_hand_state[:, 1], dim=-1, p=2)
                    diff_obj_pos = torch.norm(world_model_pred_tracked_obj_pos - to_track_obj_pos[:, 1], dim=-1, p=2)
                    diff_obj_ornt = torch.norm(world_model_pred_tracked_obj_ornt - to_track_obj_ornt[:, 1], dim=-1, p=2)
                    # forecasting_supervised_a_loss = diff_hand_state + diff_obj_pos + diff_obj_ornt + forecasting_supervised_a_loss
                    forecasting_supervised_a_loss = forecasting_supervised_a_loss * 10 + diff_hand_state + diff_obj_pos + diff_obj_ornt 
                    # forecasting_supervised_a_loss = forecasting_supervised_a_loss * 10
                else:
                    # forecasting actions --- why 
                    forecasting_supervised_a_loss_actions = self.actor_loss_supervised(forecasting_actions, forecasting_act_gt_batch)
                    # forecasting_supervised_a_loss = (forecasting_supervised_a_loss + forecasting_supervised_a_loss_actions) / 2.0 # 
                    forecasting_supervised_a_loss = (forecasting_supervised_a_loss * 10.0 + forecasting_supervised_a_loss_actions) / 2.0 # 
                
                
                forecasting_losses, sum_mask = torch_ext.apply_masks([forecasting_supervised_a_loss.unsqueeze(1), forecasting_a_loss.unsqueeze(1), forecasting_c_loss, forecasting_entropy.unsqueeze(1), forecasting_b_loss.unsqueeze(1)], rnn_masks)
                forecasting_supervised_a_loss, forecasting_a_loss, forecasting_c_loss, forecasting_entropy, forecasting_b_loss = forecasting_losses[0], forecasting_losses[1], forecasting_losses[2], forecasting_losses[3], forecasting_losses[4]
            
            
            # supervised_a_loss_coef = 0.0005
            # supervised_a_loss_coef = 1.0
            # supervised_a_loss_coef = 0.0
            
            
            # supervised_a_loss_coef = 0.0005 #
            supervised_a_loss_coef = self.supervised_loss_coef
            if self.single_instance_training:
                # print(f"single_instance_training: {self.single_instance_training}") # just focusing on-your own things #
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
                c_loss_coef = 0.0 # forecasting #
            if self.preload_experiences_tf and self.epoch_num % self.play_presaved_freq == 0:
                supervised_a_loss_coef = 0.0
            # a_loss_coef = 0.0
            # supervised_a_loss_coef = 1.0
            # if self.epoch_num % 2 == 0:
            #     a_loss_coef = 0.0
            #     supervised_a_loss_coef = 1.0
            ##### Version 2 -- with supervised action loss #####
            loss = supervised_a_loss * supervised_a_loss_coef +  a_loss * a_loss_coef + 0.5 * c_loss * self.critic_coef * c_loss_coef - entropy * self.entropy_coef * entropy_coef + b_loss * self.bounds_loss_coef * bounds_loss_coef
            
            
            if forecasting_batch_dict is not None:
                if self.use_world_model: # use world model # # what actions can you track? #
                    forecasting_loss = forecasting_supervised_a_loss
                else:
                    forecasting_sup_loss_coef = 1.0
                    # forecasting_sup_loss_coef = 0.1
                    # forecasting_sup_loss_coef = 0.01
                    # forecasting_sup_loss_coef = 0.001
                    # forecasting_sup_loss_coef = 0.0001
                    # forecasting_sup_loss_coef = 0.00001
                    forecasting_sup_loss_coef = 0.0
                    forecasting_loss = forecasting_supervised_a_loss * (forecasting_sup_loss_coef) + forecasting_a_loss * a_loss_coef + 0.5 * forecasting_c_loss * self.critic_coef * c_loss_coef - forecasting_entropy * self.entropy_coef * entropy_coef + forecasting_b_loss * self.bounds_loss_coef * bounds_loss_coef
                    # forecasting_loss = forecasting_supervised_a_loss + forecasting_a_loss * a_loss_coef + 0.5 * forecasting_c_loss * self.critic_coef * c_loss_coef - forecasting_entropy * self.entropy_coef * entropy_coef + forecasting_b_loss * self.bounds_loss_coef * bounds_loss_coef
                ##### Version 2 -- with supervised action loss #####
            
            # loss = loss # + forecasting_loss # 
            
            
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            # loss =  a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            
            if self.multi_gpu:
                self.optimizer.zero_grad()
                if forecasting_batch_dict is not None:
                    self.forecasting_optimizer.zero_grad()
            else:
                for param in self.model.parameters():
                    param.grad = None
                if forecasting_batch_dict is not None:
                    for param in self.forecasting_model.parameters():
                        param.grad = None

        if self.train_controller:
            self.scaler.scale(loss).backward()
        # print(f"forecasting_loss: {forecasting_loss}, forecasting_a_loss: {forecasting_a_loss}, forecasting_c_loss: {forecasting_c_loss}, forecasting_entropy: {forecasting_entropy}, forecasting_b_loss: {forecasting_b_loss}")
        
        
        if self.train_forecasting_model:
            self.forecasting_scaler.scale(forecasting_loss).backward() # backward the forecasting loss
        
        # aware 
        self.trancate_gradients_and_step()

        with torch.no_grad():
            """ Controller training statistics """
            reduce_kl = rnn_masks is None
            if self.train_controller: # policy ik #
                kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
                if rnn_masks is not None:
                    kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask # 
            else:
                """ Forecasting model training statistics """ 
                kl_dist = torch_ext.policy_kl(forecasting_mu.detach(), forecasting_sigma.detach(), forecasting_old_mu_batch, forecasting_old_sigma_batch, reduce_kl)
                if rnn_masks is not None:
                    kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()  #/ sum_mask # # 
                
        if self.train_controller:
            self.diagnostics.mini_batch(self,
            {
                'values' : value_preds_batch,
                'returns' : return_batch,
                'new_neglogp' : action_log_probs,
                'old_neglogp' : old_action_log_probs_batch,
                'masks' : rnn_masks
            }, curr_e_clip, 0)      
        else:
            self.diagnostics.mini_batch(self,
            {
                'values' : forecasting_value_preds_batch,
                'returns' : forecasting_return_batch,
                'new_neglogp' : forecasting_action_log_probs,
                'old_neglogp' : forecasting_old_action_log_probs_batch,
                'masks' : rnn_masks
            }, curr_e_clip, 0)      


        if self.train_controller:
            """ Controller training statistics """
            self.train_result = (a_loss, c_loss, entropy, \
                kl_dist, self.last_lr, lr_mul, \
                mu.detach(), sigma.detach(), b_loss)
        else:
            """ Forecasting model training statistics """
            self.train_result = (forecasting_a_loss, forecasting_c_loss, forecasting_entropy, \
                kl_dist, self.last_lr, lr_mul, \
                forecasting_mu.detach(), forecasting_sigma.detach(), forecasting_b_loss)
        
        
        if self.train_forecasting_model:
            self.forecasting_mu_detach = forecasting_mu.detach()
            self.forecasting_sigma_detach = forecasting_sigma.detach()
            
        if self.train_controller:
            self.controller_mu_detach = mu.detach()
            self.controller_sigma_detach = sigma.detach()
        
        
        if self.train_controller: # 
            """ Controller training statistics """
            # print(f"supervised_a_loss: {supervised_a_loss.size()}")
            self.supervised_a_loss_np = supervised_a_loss.detach().cpu().numpy()
            self.a_loss_np = a_loss.detach().cpu().numpy()
            self.c_loss_np = c_loss.detach().cpu().numpy()
            self.entropy_loss_np = -entropy.detach().cpu().numpy()
            self.b_loss_np = b_loss.detach().cpu().numpy()
        else:
            """ Forecasting model training statistics """
            # print(f"supervised_a_loss: {supervised_a_loss.size()}")
            self.supervised_a_loss_np = forecasting_supervised_a_loss.detach().cpu().numpy()
            self.a_loss_np = forecasting_a_loss.detach().cpu().numpy()
            self.c_loss_np = forecasting_c_loss.detach().cpu().numpy()
            self.entropy_loss_np = -forecasting_entropy.detach().cpu().numpy()
            self.b_loss_np = forecasting_b_loss.detach().cpu().numpy()
        return 

    def trancate_gradients_and_step(self): # 
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.model.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.model.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            if self.train_controller:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            if self.train_forecasting_model:
                self.forecasting_scaler.unscale_(self.forecasting_optimizer)
                nn.utils.clip_grad_norm_(self.forecasting_model.parameters(), self.grad_norm)

        if self.train_controller:
            # print(f"self.scaler.step(self.optimizer)")
            self.scaler.step(self.optimizer)
            self.scaler.update()
        
        if self.train_forecasting_model:
            self.forecasting_scaler.step(self.forecasting_optimizer)
            self.forecasting_scaler.update()
    
    
    def set_train(self):
        if self.train_controller:
            self.model.train()
        else:
            self.model.eval()
        if self.train_forecasting_model:
            self.forecasting_model.train()
        else:
            self.forecasting_model.eval()
        
        if self.normalize_rms_advantage:
            if self.train_controller:
                self.advantage_mean_std.train()
            else:
                self.advantage_mean_std.eval()
            if self.train_forecasting_model:
                self.advantage_mean_std_forecasting.train()
            else:
                self.advantage_mean_std_forecasting.eval()
        
        if self.use_world_model:
            self.world_model.train()
            
            
    def set_eval(self):
        self.model.eval()
        self.forecasting_model.eval()
        if self.normalize_rms_advantage:
            self.advantage_mean_std.eval()
            self.advantage_mean_std_forecasting.eval()
            
        if self.use_world_model: # 
            self.world_model.eval() # 
    
    def train_central_value(self):
        if self.train_controller:
            self.central_value_net.train_net()
        if self.train_forecasting_model:
            self.forecasting_central_value_net.train_net()
        # else:
            
        return 
        
        
    def update_lr(self, lr):
        if self.multi_gpu:
            lr_tensor = torch.tensor([lr], device=self.device)
            dist.broadcast(lr_tensor, 0)
            lr = lr_tensor.item()

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        for param_group in self.forecasting_optimizer.param_groups:
            param_group['lr'] = lr
        #if self.has_central_value:
        #    self.central_value_net.update_lr(lr)


    def train_world_model(self, world_model_dict):
        # self.world_model.train_net()
        # print(f"world_model_dict: {world_model_dict.keys()}")
        self.world_model.train()
        obses = world_model_dict['obses']
        actions = world_model_dict['actions']
        next_obses = world_model_dict['next_obses']
        obses_w_actions = torch.cat(
            [ obses, actions ], dim=-1
        )
        
        pred_next_obses = self.world_model(obses_w_actions)
        
        # nn_bsz x nn_obs_dim  
        diff_pred_next_obses = pred_next_obses - next_obses
        loss = torch.sum(diff_pred_next_obses ** 2, dim=-1) # nn_envs x nn_obs_dim
        loss = loss.mean()
        
        self.world_model_optimizer.zero_grad()
        loss.backward()
        self.world_model_optimizer.step()
        
        # print(f"world model loss: {loss.item()}")
        
        return loss.item()


    def train_epoch(self):
        if self.train_controller:
            super().train_epoch()
        else:
            self.vec_env.set_train_info(self.frame, self)

        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            # if self.preload_experiences_tf and self.epoch_num % self.play_presaved_freq == 0:
            #     batch_dict = self.play_presaved_experiences()
            # else:
            if self.is_rnn:
                batch_dict = self.play_steps_rnn()
            else:
                batch_dict = self.play_steps()
        
        
        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()
        # rnn_masks = batch_dict.get('rnn_masks', None)

        self.set_train() # train the controller 
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        
        forecasting_dict = batch_dict['forecasting_dict']
        self.prepare_dataset_forecasting(forecasting_dict)
        
        if self.use_world_model:
            world_model_dict = batch_dict['world_model_dict']
            self.prepare_dataset_world_model(world_model_dict)
        
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
            ep_world_model_losses = []
            
            for i in range(len(self.dataset)):
                
                """ Update world model """
                if self.use_world_model:
                    self.world_model.train()
                    cur_world_model_dict = self.world_model_dataset[i]
                    world_model_loss = self.train_world_model(cur_world_model_dict)
                    ep_world_model_losses.append(world_model_loss)
                    
                    self.world_model.eval()
                
                cur_dict = self.dataset[i]
                cur_forecasting_dict = self.forecasting_dataset[i]
                cur_forecasting_dict = {
                    f"forecasting_{k}": cur_forecasting_dict[k] for k in cur_forecasting_dict
                }
                
                cur_dict.update(cur_forecasting_dict)
                # print(f"cur_forecasting_dict: {cur_forecasting_dict.keys()}, cur_dict: {cur_dict.keys()}")
                a_loss, c_loss, entropy, kl, last_lr, lr_mul, cmu, csigma, b_loss = self.train_actor_critic(cur_dict)
                a_losses.append(a_loss)
                c_losses.append(c_loss)
                ep_kls.append(kl)
                entropies.append(entropy)
                if self.bounds_loss_coef is not None:
                    b_losses.append(b_loss)

                """ Update the mu and sigma in the dataset """
                if self.train_controller:
                    self.dataset.update_mu_sigma(self.controller_mu_detach, self.controller_sigma_detach) # 
                    
                """ Update the mu and sigma in the forecasting dataset """
                if self.train_forecasting_model:
                    self.forecasting_dataset.update_mu_sigma(self.forecasting_mu_detach, self.forecasting_sigma_detach)
                # del self.forecasting_mu_detach, self.forecasting_sigma_detach
                
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
                
                
                
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
                torch.cuda.empty_cache()
            
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
            if self.use_world_model:
                ep_world_model_losses = np.array(ep_world_model_losses)
                avg_ep_world_model_loss = np.mean(ep_world_model_losses).item()
                # print(f"avg_ep_world_model_loss: {avg_ep_world_model_loss}")
            else:
                avg_ep_world_model_loss = 0.0
            
            # single_instance_training 
            if not self.single_instance_training:
                print(f"avg_ep_act_supervised_loss: {avg_ep_act_supervised_loss}, avg_ep_a_loss: {avg_ep_a_loss}, avg_ep_c_loss: {avg_ep_c_loss}, avg_ep_entropy_loss: {avg_ep_entropy_loss}, avg_ep_b_loss: {avg_ep_b_loss}, avg_ep_world_model_loss: {avg_ep_world_model_loss}")
                
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
                self.forecasting_model.running_mean_std.eval()

        update_time_end = time.perf_counter()
        play_time = play_time_end - play_time_start
        update_time = update_time_end - update_time_start
        total_time = update_time_end - play_time_start

        return batch_dict['step_time'], play_time, update_time, total_time, a_losses, c_losses, b_losses, entropies, kls, last_lr, lr_mul


    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        state['forecasting_model'] = self.forecasting_model.state_dict()
        if self.use_world_model:
            state['world_model'] = self.world_model.state_dict()
        return state


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

                if self.forecasting_game_rewards.current_size > 0:
                    mean_rewards = self.forecasting_game_rewards.get_mean()
                    mean_shaped_rewards = self.forecasting_game_shaped_rewards.get_mean()
                    mean_lengths = self.forecasting_game_lengths.get_mean()
                    self.mean_rewards = mean_rewards[0]
                    
                    mean_controller_rewards = self.game_rewards.get_mean()
                    self.mean_controller_rewards = mean_controller_rewards[0]
                    print('mean_rewards: ', mean_rewards, 'mean_controller_rewards: ', mean_controller_rewards, 'last_mean_rewards: ', self.last_mean_rewards)
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

                    # save 
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
                            wf.write(f"epoch: {epoch_num}, mean_rewards: {mean_rewards[0]}, mean_controller_rewards: {mean_controller_rewards[0]}\n")
                            wf.close()

                        # last mean rewards #
                        if 'score_to_win' in self.config:
                            if self.last_mean_rewards > self.config['score_to_win']:
                                print('Maximum reward achieved. Network won!')
                                self.save(os.path.join(self.nn_dir, checkpoint_name))
                                should_exit = True

                # epoch num #
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
        
        
        
        self.tensor_list += ['gt_act']
        
        if self.masked_mimic_training:
            self.experience_buffer.tensor_dict['mimic_teacher_obs'] = torch.zeros(batch_shape + (self.obs_shape[0], ),
                                                                    device=self.ppo_device)
            self.tensor_list += ['mimic_teacher_obs']
            
        
        if self.w_forecasting_model:
            self._build_forecasting_module_buffers()
            
        #     # input #
        #     # value_preds_batch = input_dict['old_values']
        #     # old_action_log_probs_batch = input_dict['old_logp_actions'] # 
        #     # advantage = input_dict['advantages']
        #     # old_mu_batch = input_dict['mu']
        #     # old_sigma_batch = input_dict['sigma']
        #     # return_batch = input_dict['returns']
        #     # actions_batch = input_dict['actions']
        #     # obs_batch = input_dict['obs']
        #     self.experience_buffer.tensor_dict['forecast_old_values'] = torch.zeros(batch_shape + (1,),
        #                                                             device=self.ppo_device)
        #     self.experience_buffer.tensor_dict['forecast_old_logp_actions'] = torch.zeros(batch_shape + (1,),
        #                                                             device=self.ppo_device)
        #     self.experience_buffer.tensor_dict['forecast_old_logp_actions'] = torch.zeros(batch_shape + (1,),
        #                                                             device=self.ppo_device)
        #     # whether we need to add the 
        #     ## TODO: add the planning model's buffer here for the training with planning model #
        #     ## planning obs, planning 
        #     pass
        
        return



    def _build_forecasting_module_buffers(self):
        batch_size = self.num_agents * self.num_actors
        algo_info = {
            'num_actors' : self.num_actors,
            'horizon_length' : self.horizon_length,
            'has_central_value' : self.has_central_value,
            'use_action_masks' : self.use_action_masks
        }
        self.forecasting_env_info = self.env_info.copy()
        self.forecasting_env_info['observation_space'] = gym.spaces.Box(low=-1, high=1,shape=(self.forecasting_obs_dim, ), dtype=np.float32) 
        # tuple(list(self.obs_shape)[:-1] + [self.forecasting_obs_dim])
        self.forecasting_env_info['action_space'] =gym.spaces.Box(low=-1, high=1,shape=(self.forecasting_act_dim, ), dtype=np.float32) 
        self.forecasting_experience_buffer = ExperienceBuffer(self.forecasting_env_info, algo_info, self.ppo_device)
        
        
        
        # val_shape = (self.horizon_length, batch_size, self.value_size)
        current_rewards_shape = (batch_size, self.value_size)
        self.forecasting_current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.forecasting_current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
        self.forecasting_current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
        self.forecasting_dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

        self.forecasting_rnn_states = None
        if self.is_rnn:
            self.forecasting_rnn_states = self.forecasting_model.get_default_rnn_state()
            self.forecasting_rnn_states = [s.to(self.ppo_device) for s in self.forecasting_rnn_states]

            total_agents = self.num_agents * self.num_actors
            num_seqs = self.horizon_length // self.seq_length
            assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
            self.forecasting_mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.forecasting_rnn_states]

        batch_shape = self.experience_buffer.obs_base_shape
        self.forecasting_experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (self.forecasting_act_dim, ),
                                                                    device=self.ppo_device)
        
        return

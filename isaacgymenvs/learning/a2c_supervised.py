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

from isaacgymenvs.learning.transformer_layers import TransformerFeatureProcessing

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


def batched_index_select(values, indices, dim = 1):
  value_dims = values.shape[(dim + 1):]
  values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
  indices = indices[(..., *((None,) * len(value_dims)))]
  indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
  value_expand_len = len(indices_shape) - (dim + 1)
  values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

  value_expand_shape = [-1] * len(values.shape)
  expand_slice = slice(dim, (dim + value_expand_len))
  value_expand_shape[expand_slice] = indices.shape[expand_slice]
  values = values.expand(*value_expand_shape)

  dim += value_expand_len
  return values.gather(dim, indices)




class A2CSupervisedAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, params):
        
        self.params = params
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
        # preload_experiences_tf, preload_experiences_path #
        self.preload_experiences_tf = params['config'].get('preload_experiences_tf', False)
        self.preload_experiences_path = params['config'].get('preload_experiences_path', None)
        self.single_instance_training = params['config'].get('single_instance_training', False)
        self.presaved_experience_st_idx = 0
        self.play_presaved_freq = 10
        # 10 # 
        self.single_instance_tag = params['config'].get('single_instance_tag', '') # 
        self.obj_type_to_optimized_res_fn = params['config'].get('obj_type_to_optimized_res_fn', '')
        self.supervised_loss_coef = params['config'].get('supervised_loss_coef', 0.0005)
        self.pure_supervised_training = params['config'].get('pure_supervised_training', False)
        
        self.grab_obj_type_to_opt_res_fn = params['config'].get('grab_obj_type_to_opt_res_fn', '')
        self.taco_obj_type_to_opt_res_fn = params['config'].get('taco_obj_type_to_opt_res_fn', '')
        self.use_teacher_model = params['config'].get('use_teacher_model', False)
        
        self.teacher_model_path = params['config'].get('teacher_model_path', '')
        self.w_franka = params['config'].get('w_franka', False)
        
        self.dagger_style_training = params['config'].get('dagger_style_training', False)
        self.rollout_teacher_model = params['config'].get('rollout_teacher_model', False)
        self.rollout_student_model = params['config'].get('rollout_student_model', True)
        
        if self.dagger_style_training and self.preload_experiences_tf:
            self.play_presaved_freq = 1 # every time we would play the preseaved experiences #
        
        self.optimizing_with_teacher_net = len(self.teacher_model_path) > 0 and os.path.exists(self.teacher_model_path) 
        
        # chunk_start_frame, play_presaved_experience_idx, play_presaved_experience_changing_freq
        self.chunk_start_frame = 0
        self.play_presaved_experience_idx = 0
        self.play_presaved_experience_changing_freq = 10
        
        ## action chunking settings ##
        self.action_chunking = params['config'].get('action_chunking', False)
        self.action_chunking_frames = params['config'].get('action_chunking_frames', 1)
        self.action_chunking_skip_frames = params['config'].get('action_chunking_skip_frames', 1)
        self.bc_style_training = params['config'].get('bc_style_training', False)
        self.bc_relative_targets = params['config'].get('bc_relative_targets', False)
        self.simreal_modeling = params['config'].get('simreal_modeling', False)
        self.use_no_obj_pose = params['config'].get('use_no_obj_pose', False)
        ## action chunking settings ##
        
        
        ## distill via bc ##
        self.distill_via_bc = params['config'].get('distill_via_bc', False)
        
        # demonstration_tuning_model, demonstration_tuning_model_freq
        self.demonstration_tuning_model = params['config'].get('demonstration_tuning_model', False)
        self.demonstration_tuning_model_freq = params['config'].get('demonstration_tuning_model_freq', 1) # use the demonstration to tune the model add the demonstration tuning model freq # only current epoch multiplies of the tuning model freq, can we activte the play experience buffer step # have many and many demonstrations --- then we use these demonstrations and the actions contained in those demonstrations to train the model # # 
        
        
        self.save_experiences_via_ts = params['config'].get('save_experiences_via_ts', False)
        self.load_experiences_maxx_ts = params['config'].get('load_experiences_maxx_ts', False)
        
        # 
        self.history_length = params['config'].get('history_length', 1)
        
        ###### Teacher models for masked mimicing ######
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
            self._init_multiple_teacher_models()
            
            self.cur_teacher_idx = 0
        
        
        
        ###### Tacher models for specialist-generalist training ######
        if self.use_teacher_model and self.optimizing_with_teacher_net and (not self.use_multiple_teacher):
            self._init_teacher_models_single()
        
        
        ###### Initialize student models ########
        self.train_student_model = params['config'].get('train_student_model', False) # bool value #
        self.ts_teacher_model_obs_dim = params['config'].get('ts_teacher_model_obs_dim', self.obs_shape[0])  # get the student model observation dimension #
        self.ts_teacher_model_weights_fn = params['config'].get('ts_teacher_model_weights_fn', '') # get the student model weights #
        if self.train_student_model:
            self._init_ts_teacher_models()
        # after that we sshould initlaize the studnet model -- the mus and the sigmas should be supervised ? #
        
        
        
        self.distill_action_space = params['config'].get('distill_action_space', False)
        
        self.distill_delta_targets = params['config'].get('distill_delta_targets', False)
        
        self.preload_all_saved_exp_buffers = params['config'].get('preload_all_saved_exp_buffers', False)
        
        self.test_inst_tag = params['config'].get('test_inst_tag', '')
        self.add_obj_features = params['config'].get('add_obj_features', False)
        
        
        self.traj_idx_to_experience_sv_folder = params['config'].get('traj_idx_to_experience_sv_folder', '')
        
        self.load_chunking_experiences_v2 = params['config'].get('load_chunking_experiences_v2', False)
        self.history_chunking_obs_version = params['config'].get('history_chunking_obs_version', 'v1')
        self.load_chunking_experiences_from_real = params['config'].get('load_chunking_experiences_from_real', False)
        
        self.use_transformer_model = params['config'].get('use_transformer_model', False)
        
        if self.use_transformer_model:
            self._init_model_transformers()
            self.transformer_encoder_decoder.train()
        
        if self.w_franka:
            self.nn_act_dims = 23
        else:
            self.nn_act_dims = 22
        
        # if self.vec_env.env.use_history_obs:
        #     self.nn_act_dims = self.nn_act_dims * self.vec_env.env.history_length
        
        if self.action_chunking:
            self.nn_act_dims = self.nn_act_dims * self.action_chunking_frames
        
        # if self.action_chunking:
        #     self.horizon_length = self.horizon_length // self.action_chunking_frames
        
        return
    
    # initialize teacher model; initialize the model of a different architecture? #
    
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
    
    
    def _init_ts_teacher_models(self, ):
        # self.train_student_model
        ts_teacher_model_obs_dim = self.ts_teacher_model_obs_dim
        self.ts_teacher_network_params = self.params.copy()
        self.ts_teacher_network_builder = model_builder.ModelBuilder()
        self.ts_teacher_network_builder = self.ts_teacher_network_builder.load(self.ts_teacher_network_params)
        
        ts_teacher_model_build_config = {
            'actions_num' : self.actions_num, #  
            'input_shape' : (ts_teacher_model_obs_dim, ),
            'num_seqs' : self.num_actors * self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input, # # normalize # #
        }
        
        ts_teacher_model = self.ts_teacher_network_builder.build(ts_teacher_model_build_config).to(self.ppo_device)
        
        ts_teacher_model_weights_fn = self.ts_teacher_model_weights_fn
        
        ts_teacher_model_weights = torch.load(ts_teacher_model_weights_fn, map_location='cpu')
        ts_teacher_model.load_state_dict(ts_teacher_model_weights['model'])
        ts_teacher_model.eval()
        self.ts_teacher_model = ts_teacher_model # init the ts teacher model # # add the full obs fn # 
        
        # self.student_network_params = self.params.copy() # studnet model obs dim, --- #
        # # self.student_network_params['network']['mlp']['units'] = [8192, 4096, 2048, 1024, 512, 256, 128] # sue the same network paramsters as the teacher model #
        # self.student_network_builder = model_builder.ModelBuilder()
        # self.student_network_builder = self.student_network_builder.load(self.student_network_params)
        
        # student_model_build_config = {
        #     'actions_num' : self.actions_num,
        #     'input_shape' : (student_model_obs_dim, ),
        #     'num_seqs' : self.num_actors * self.num_agents,
        #     'value_size': self.env_info.get('value_size',1),
        #     'normalize_value' : self.normalize_value,
        #     'normalize_input': self.normalize_input, # # normalize # # 
        # } # 
        # student_model = self.student_network_builder.build(student_model_build_config).to(self.ppo_device) # studnet model #
        
        # self.student_model


    
    # teacher_index_to_weights, use_multiple_teacher, _init_multiple_teacher_models
    def _init_multiple_teacher_models(self, ):
        
        assert self.use_multiple_teacher 
        assert self.optimizing_with_teacher_net
        
        # obj_feat_shape = 256 #
        
        self.teacher_obs_dim = self.obs_shape[0] 
        self.teacher_network_params = self.params.copy()
        self.teacher_network_builder = model_builder.ModelBuilder()
        self.teacher_network_builder = self.teacher_network_builder.load(self.teacher_network_params)
        
        self.teacher_index_to_obs_dim = {}
        
        self.teacher_index_to_models = {}
        
        tot_teacher_indexes = list(self.teacher_index_to_weights.keys())
        
        # if self.dagger_style_training:
        #     tot_teacher_indexes = [7]
        
        # for teacher_index in self.teacher_index_to_weights:
        # for teacher_index in [7]:
        for teacher_index in tot_teacher_indexes:
            # cur_teacher_model_weight_fn = self.teacher_index_to_model_weights[teacher_index]
            # teacher_index_to_weights
            cur_teacher_model_weight_fn = self.teacher_index_to_weights[teacher_index]
            # a2c_network.actor_mlp.0.weight # 
            
            ## two instance model ##
            # cur_teacher_model_weight_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s9_v3goal_v2/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-16-57-06/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
            # ## all s9 instances model ##
            # cur_teacher_model_weight_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s9_v3goal_v2/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-15-57-15/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
            
            print(f"Loading teacher model from {cur_teacher_model_weight_fn}, teacher_idx: {teacher_index}")
            
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
        # 
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
        
    
    
    def _init_model_transformers(self, ):
        # transformer encoder and the transformer decoder #
        # during the traiing, we need to set this model to train
        # we also need to add an argument to control this model's usage
        nn_latents = 512
        self.transformer_encoder_decoder = TransformerFeatureProcessing(nn_latents=nn_latents, dropout=0.1)
        self.transformer_encoder_decoder = self.transformer_encoder_decoder.to(self.device)
        # self.transformer_encoder_decoder.train() #
        self.transformer_optimizer = optim.Adam(self.transformer_encoder_decoder.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.transformer_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        pass
    
    def _forward_transformer_model(self, input_dict):
        
        input_obses = input_dict['obs']
        output_mus = self.transformer_encoder_decoder(input_obses, nn_history=self.history_length, nn_future=self.action_chunking_frames)
        
        prev_neglogp = torch.ones_like(output_mus )[:, 0] * 0.1
        values = torch.ones_like(output_mus)[:, 0]
        entropy = torch.ones_like(output_mus ) * 0.1
        mus = output_mus
        sigmas = torch.ones_like(mus) * 0.1
        
        res_dict = {
            'prev_neglogp': prev_neglogp,
            'values': values,
            'entropy': entropy,
            'mus': mus,
            'sigmas': sigmas
        }
        return res_dict
        pass
    
    
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
        if self.preload_experiences_tf: # init tensors #
            if self.distill_via_bc:
                self.preload_saved_experiences_simdemo()
            elif self.action_chunking and self.bc_style_training:
                if self.load_chunking_experiences_v2:
                    if self.load_chunking_experiences_from_real:
                        self.preload_multiple_saved_experiences_realdemo()
                    else:   
                        self.preload_multiple_saved_experiences_chunking_bc_simdemo()
                else:
                    self.preload_saved_experiences_chunking_bc()
            else:
                if self.preload_all_saved_exp_buffers:
                    self.preload_all_saved_experiences()
                else:
                    self.preload_saved_experiences()
        ### TODO: add the demo experience buffer and the logic of preload demonstrations ###
        # self.build_demo_experience_buffer() # experiences # training #
        # self.preload_demonstrations() # 
        return
    
    
    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['gt_act'] = batch_dict['gt_act']
        if self.masked_mimic_training:
            self.dataset.values_dict['mimic_teacher_obs'] = batch_dict['mimic_teacher_obs']
        if self.train_student_model:
            self.dataset.values_dict['full_obs'] = batch_dict['full_obs']
        if self.distill_delta_targets:
            self.dataset.values_dict['delta_targets'] = batch_dict['delta_targets']
        # self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        # self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        # self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        # return

    def get_action_values_dagger_style(self, obs):
        
        processed_obs = self._preproc_obs(obs['obs'])
        # self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., : self.obs_shape[0]],
            'rnn_states' : self.rnn_states
        }
        gt_act_batch = obs['obs'][..., -1:]
        
        with torch.no_grad(): 
            key_to_res_batch = {}
            
            # tot_teacher_mus_batch = []
            # tot_teacher_sigmas_batch = []
            for i_teacher in range(0, self.nn_teacher):
                cur_teaacher_model = self.teacher_index_to_models[i_teacher]
                cur_teaacher_model.eval()
                
                teacher_model_obs_dim = self.teacher_index_to_obs_dim[i_teacher]
                teacher_res_dict = self._get_teacher_action_values(input_dict, cur_teaacher_model, teacher_model_obs_dim=teacher_model_obs_dim)
                
                for key in teacher_res_dict:
                    if key not in key_to_res_batch:
                        key_to_res_batch[key] = [teacher_res_dict[key]]
                    else:
                        key_to_res_batch[key].append(teacher_res_dict[key])
                # cur_teacher_mus = teacher_res_dict['mus'] # nn_bsz x nn_act_dim
                # cur_teacher_sigmas = teacher_res_dict['sigmas']
                # tot_teacher_mus_batch.append(cur_teacher_mus)
                # tot_teacher_sigmas_batch.append(cur_teacher_sigmas)
            
            envs_teacher_idxes = gt_act_batch[..., -1:].long() # nn_envs x 1 
            for key in key_to_res_batch:
                if key_to_res_batch[key][0] is None:
                    key_to_res_batch[key] = None
                else:
                    key_to_res_batch[key] = torch.stack(key_to_res_batch[key], dim=1)
            
            res_dict = {}
            for key in key_to_res_batch:
                if key_to_res_batch[key] is not None:
                    res_dict[key] = batched_index_select(key_to_res_batch[key], envs_teacher_idxes, dim=1).squeeze(1) 
                else:
                    res_dict[key] = None
        # tot_teacher_mus_batch = torch.stack(tot_teacher_mus_batch, dim=1) # nn_bsz x nn_teac x ...
        # tot_teacher_sigmas_batch = torch.stack(tot_teacher_sigmas_batch, dim=1)
        
        # envs_teacher_idxes = gt_act_batch[..., -1:].long() # nn_envs x 1 
        # maxx_teacher_idx = torch.max(envs_teacher_idxes)
        # minn_teacher_idx = torch.min(envs_teacher_idxes)
        # # print(f"maxx_teacher_idx: {maxx_teacher_idx}, minn_teacher_idx: {minn_teacher_idx}, tot_teacher_mus_batch: {tot_teacher_mus_batch.size()}, tot_teacher_sigmas_batch: {tot_teacher_sigmas_batch.size()}")
        # # print(f"teacher_index_to_weights: {self.teacher_index_to_weights}")
        
        # envs_teacher_mus = batched_index_select(tot_teacher_mus_batch, envs_teacher_idxes, dim=1) # nn_bsz x 1 x nn_act_dim
        # envs_teacher_mus = envs_teacher_mus.squeeze(1) # nn_bsz x nn_act_dim
        # envs_teacher_sigmas = batched_index_select(tot_teacher_sigmas_batch, envs_teacher_idxes, dim=1) 
        # envs_teacher_sigmas = envs_teacher_sigmas.squeeze(1) # nn_bsz x nn_act_dim
        return res_dict


    # avg a loss #
    # if we use the deterministic network toher than ouputing the action distributions ? #
    def get_action_values(self, obs): # action values #
        
        if self.dagger_style_training:
            res_dict = self.get_action_values_dagger_style(obs)
            return res_dict

        
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
            # if self.dagger_style_training:
            #     res_dict = self.teacher_index_to_models[7](input_dict)
            # else:   
            #     res_dict = self.model(input_dict)
            
            if self.dagger_style_training:
                if self.rollout_student_model:
                    res_dict = self.model(input_dict)
                elif self.rollout_teacher_model:
                    res_dict = self.teacher_index_to_models[7](input_dict)
                else:
                    raise ValueError("Invalid training mode")
            elif self.train_student_model:
                if self.rollout_student_model:
                    res_dict = self.model(input_dict)
                elif self.rollout_teacher_model:
                    self.ts_teacher_model.eval()
                    # print(f"ts_teacher_model: {self.ts_teacher_model}")
                    # input_dict['obs'] = self._preproc_obs(obs['full_obs'])[..., : self.obs_shape[0]]    
                    input_dict['obs'] = obs['full_obs']
                    res_dict = self.ts_teacher_model(input_dict)
                else:
                    raise ValueError("Invalid training mode")
            else:
                res_dict = self.model(input_dict)
            
            
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        
        ### try to test it ###        
        
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
    
    
    def get_ts_teacher_action_values(self, obs, is_train=False):
        processed_obs = self._preproc_obs(obs['full_obs'])
        self.ts_teacher_model.eval()
        input_dict = {
            'is_train': is_train,
            'prev_actions': obs['actions'], 
            'obs' : processed_obs[..., : self.ts_teacher_model_obs_dim],
            'rnn_states' : self.rnn_states
        }

        # get the actions values #
        with torch.no_grad(): 
            res_dict = self.ts_teacher_model(input_dict)
            if self.has_central_value:
                states = obs['states']
                input_dict = {
                    'is_train': False,
                    'states' : states,
                }
                value = self.get_central_value(input_dict)
                res_dict['values'] = value
        return res_dict
    
    # observation should be passed to the file #
    # ad then inference directly apply the actions in the step function # 
    # for action chunking, actually we need a different 
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
            
            # # demo experience buffer # #
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
    
    # [act_real, obs_real] x seq_len --> [obs_sim] x seq_len #
    
    
    def preload_saved_experiences_chunking_bc_simdemo_dict(self, tot_preload_exp):
        obj_latent_feat = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
        obj_latent_feat_dict = np.load(obj_latent_feat, allow_pickle=True).item()
        self.preload_experiences  = {}
        tot_obses, tot_dones, tot_rewards, tot_gt_act =     [], [], [], []
        for obj_code in tot_preload_exp:
            print(f"loading experience for {obj_code}")
            cur_obj_feat_np = obj_latent_feat_dict[obj_code]
            cur_obj_exp_fn = tot_preload_exp[obj_code]
            preload_exp = np.load(cur_obj_exp_fn, allow_pickle=True).item()
            
            tot_ts  = list(preload_exp.keys())
        
            tot_ts = [ int(cur_ts) for cur_ts in tot_ts if isinstance(cur_ts, int) ]
            tot_ts = sorted(tot_ts)
            tot_qtars = []
            tot_obj_pose = []
            tot_states = []
            for i_ts in tot_ts:
                cur_ts_exp_dict = preload_exp[i_ts]
                shadow_hand_dof_pos = cur_ts_exp_dict['shadow_hand_dof_pos']
                shadow_hand_dof_tars = cur_ts_exp_dict['shadow_hand_dof_tars']
                object_pose = cur_ts_exp_dict['object_pose']
                
                shadow_hand_dof_pos = torch.from_numpy(shadow_hand_dof_pos, ).float().to(self.device)
                shadow_hand_dof_tars = torch.from_numpy(shadow_hand_dof_tars, ).float().to(self.device)
                object_pose = torch.from_numpy(object_pose, ).float().to(self.device)
                
                tot_qtars.append(shadow_hand_dof_tars)
                tot_obj_pose.append(object_pose)
                tot_states.append(shadow_hand_dof_pos)
            # tot_qtars  = np.stack()
            tot_qtars = torch.stack(tot_qtars, dim=0)
            tot_obj_pose = torch.stack(tot_obj_pose, dim=0)
            tot_states = torch.stack(tot_states, dim=0)
            
            real_arm_pos  = tot_states[..., :7]
            real_leap_pos_to_sim = tot_states[..., 7:]
            real_object_pose = tot_obj_pose
            cur_step_already_execute_actions = tot_qtars
            
            maxx_frame = 300
            real_arm_pos = real_arm_pos[: maxx_frame]
            real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
            real_object_pose = real_object_pose[: maxx_frame]
            cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
            
            
            history_len = self.history_length
            action_chunks = 10
            obses = []
            dones = []
            rewards = []
            gt_act = []
            for i_ts in range(real_arm_pos.shape[0]):
                tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
                tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
                # tot history ts #
                tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
                # nn_ts x nn_envs x nn_feature_dim # # nn_feature_dim #
                history_real_arm_pos = real_arm_pos[tot_history_ts]
                history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
                history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
                
                tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
                tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
                tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
                future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
                
                if self.use_no_obj_pose:
                    history_real_object_pose[..., :] = 0.0
                    history_real_object_pose[..., -1] = 1.0
                
                cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose], dim=-1)
                cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
                cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
                
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
                
                if i_ts < real_arm_pos.shape[0] - 1:
                    cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                else:
                    cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
                gt_act.append(future_cur_step_already_execute_actions)
                obses.append(cur_obs)
                dones.append(cur_dones)
                rewards.append(cur_rewards)
                
            obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
            dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
            rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
            gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
            
            cur_obj_code_th = torch.from_numpy(cur_obj_feat_np).float().to(self.device)
            obses = torch.cat(
                [ obses, cur_obj_code_th.unsqueeze(0).unsqueeze(0).repeat(obses.size(0), obses.size(1), 1) ], dim=-1
            )
            
            tot_obses.append(obses)
            tot_dones.append(dones)
            tot_rewards.append(rewards)
            tot_gt_act.append(gt_act)
            # self.preload_experiences = {
            #     'obses' : obses,
            #     'dones' : dones,
            #     'rewards' : rewards,
            #     'actions' : gt_act,
            #     'neglogpacs': torch.zeros_like(dones),
            #     'values': torch.zeros_like(dones).unsqueeze(-1),
            #     'mus' : gt_act,
            #     'sigmas' : gt_act,
            # }
            
            # for key in self.preload_experiences:
            #     val = self.preload_experiences[key]
            #     print(f"key: {key}, val: {val.size()}")
        tot_obses = torch.cat(tot_obses, dim=1)
        tot_dones = torch.cat(tot_dones, dim=1)
        tot_rewards = torch.cat(tot_rewards, dim=1)
        tot_gt_act = torch.cat(tot_gt_act, dim=1)
        print(f"tot_obses: {tot_obses.size()}, tot_dones: {tot_dones.size()}, tot_rewards: {tot_rewards.size()}, tot_gt_act: {tot_gt_act.size()}")
        self.preload_experiences = {
            'obses' : tot_obses,
            'dones' : tot_dones,
            'rewards' : tot_rewards,
            'actions' : tot_gt_act,
            'neglogpacs': torch.zeros_like(tot_dones),
            'values': torch.zeros_like(tot_dones).unsqueeze(-1),
            'mus' : tot_gt_act,
            'sigmas' : tot_gt_act,
        }
        pass
    
    ## TODO: add a loading from sim option and loading from this function ##
    def preload_saved_experiences_chunking_bc_simdemo(self, ):
        preload_exp = np.load(self.preload_experiences_path, allow_pickle=True).item()
        
        if 'object_code_list' not in preload_exp:
            self.preload_saved_experiences_chunking_bc_simdemo_dict(preload_exp)
            return
        
        tot_ts  = list(preload_exp.keys())
        
        tot_ts = [ int(cur_ts) for cur_ts in tot_ts if isinstance(cur_ts, int) ]
        tot_ts = sorted(tot_ts)
        tot_qtars = []
        tot_obj_pose = []
        tot_states = []
        for i_ts in tot_ts:
            cur_ts_exp_dict = preload_exp[i_ts]
            shadow_hand_dof_pos = cur_ts_exp_dict['shadow_hand_dof_pos']
            shadow_hand_dof_tars = cur_ts_exp_dict['shadow_hand_dof_tars']
            object_pose = cur_ts_exp_dict['object_pose']
            
            shadow_hand_dof_pos = torch.from_numpy(shadow_hand_dof_pos, ).float().to(self.device)
            shadow_hand_dof_tars = torch.from_numpy(shadow_hand_dof_tars, ).float().to(self.device)
            object_pose = torch.from_numpy(object_pose, ).float().to(self.device)
            
            tot_qtars.append(shadow_hand_dof_tars)
            tot_obj_pose.append(object_pose)
            tot_states.append(shadow_hand_dof_pos)
        # tot_qtars  = np.stack()
        tot_qtars = torch.stack(tot_qtars, dim=0)
        tot_obj_pose = torch.stack(tot_obj_pose, dim=0)
        tot_states = torch.stack(tot_states, dim=0)
        
        real_arm_pos  = tot_states[..., :7]
        real_leap_pos_to_sim = tot_states[..., 7:]
        real_object_pose = tot_obj_pose
        cur_step_already_execute_actions = tot_qtars
        
        maxx_frame = 300
        real_arm_pos = real_arm_pos[: maxx_frame]
        real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
        real_object_pose = real_object_pose[: maxx_frame]
        cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
        
        
        history_len = self.history_length
        # action_chunks = 10
        action_chunks = self.action_chunking_frames
        obses = []
        dones = []
        rewards = []
        gt_act = []
        for i_ts in range(real_arm_pos.shape[0]):
            tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
            tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
            # tot history ts #
            tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
            # nn_ts x nn_envs x nn_feature_dim # # nn_feature_dim #
            history_real_arm_pos = real_arm_pos[tot_history_ts]
            history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
            history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
            
            tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
            tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
            tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
            future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
            
            if self.use_no_obj_pose:
                history_real_object_pose[..., :] = 0.0
                history_real_object_pose[..., -1] = 1.0
            
            cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose], dim=-1)
            cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
            cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
            
            future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
            future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
            
            if i_ts < real_arm_pos.shape[0] - 1:
                cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
            else:
                cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
            cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
            gt_act.append(future_cur_step_already_execute_actions)
            obses.append(cur_obs)
            dones.append(cur_dones)
            rewards.append(cur_rewards)
            
        obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
        dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
        rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
        gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
        self.preload_experiences = {
            'obses' : obses,
            'dones' : dones,
            'rewards' : rewards,
            'actions' : gt_act,
            'neglogpacs': torch.zeros_like(dones),
            'values': torch.zeros_like(dones).unsqueeze(-1),
            'mus' : gt_act,
            'sigmas' : gt_act,
        }
        
        for key in self.preload_experiences:
            val = self.preload_experiences[key]
            print(f"key: {key}, val: {val.size()}")
    
    
    
    def preload_multiple_saved_experiences_chunking_bc_simdemo(self, ):
        traj_idx_to_preload_exp = np.load(self.preload_experiences_path, allow_pickle=True).item()
        
        # if 'object_code_list' not in preload_exp:
        #     self.preload_saved_experiences_chunking_bc_simdemo_dict(preload_exp)
        #     return
        
        tot_obses = []
        tot_dones = []
        tot_rewards = []
        tot_gt_act = []
        
        for traj_idx in traj_idx_to_preload_exp:
            preload_exp_fn = traj_idx_to_preload_exp[traj_idx]
            preload_exp = np.load(preload_exp_fn, allow_pickle=True).item()
            
            obses = []
            dones = []
            rewards = []
            gt_act = []
            
        
            tot_ts  = list(preload_exp.keys())
            
            tot_ts = [ int(cur_ts) for cur_ts in tot_ts if isinstance(cur_ts, int) ]
            tot_ts = sorted(tot_ts)
            tot_qtars = []
            tot_obj_pose = []
            tot_states = []
            
            tot_goal_hand_qpos = []
            tot_goal_obj_pose = []
            for i_ts in tot_ts:
                cur_ts_exp_dict = preload_exp[i_ts]
                shadow_hand_dof_pos = cur_ts_exp_dict['shadow_hand_dof_pos'] # actual states
                shadow_hand_dof_tars = cur_ts_exp_dict['shadow_hand_dof_tars'] # actual actions
                object_pose = cur_ts_exp_dict['object_pose'] # actual object poses
                
                goal_hand_qpos = cur_ts_exp_dict['next_ref_np'] # goal hand qpos
                goal_obj_pose = cur_ts_exp_dict['goal_pose_ref_np'] # goal object pose
                
                shadow_hand_dof_pos = torch.from_numpy(shadow_hand_dof_pos, ).float().to(self.device)
                shadow_hand_dof_tars = torch.from_numpy(shadow_hand_dof_tars, ).float().to(self.device)
                object_pose = torch.from_numpy(object_pose, ).float().to(self.device)
                
                goal_hand_qpos = torch.from_numpy(goal_hand_qpos, ).float().to(self.device)
                goal_obj_pose = torch.from_numpy(goal_obj_pose, ).float().to(self.device)
                
                
                tot_qtars.append(shadow_hand_dof_tars)
                tot_obj_pose.append(object_pose)
                tot_states.append(shadow_hand_dof_pos)
                
                tot_goal_hand_qpos.append(goal_hand_qpos)
                tot_goal_obj_pose.append(goal_obj_pose)
                
                
            # tot_qtars  = np.stack()
            tot_qtars = torch.stack(tot_qtars, dim=0)
            tot_obj_pose = torch.stack(tot_obj_pose, dim=0)
            tot_states = torch.stack(tot_states, dim=0)
            tot_goal_obj_pose = torch.stack(tot_goal_obj_pose, dim=0)
            tot_goal_hand_qpos = torch.stack(tot_goal_hand_qpos, dim=0)
            
            
            real_arm_pos  = tot_states[..., :7]
            real_leap_pos_to_sim = tot_states[..., 7:]
            real_object_pose = tot_obj_pose
            cur_step_already_execute_actions = tot_qtars
        
            maxx_frame = 300
            real_arm_pos = real_arm_pos[: maxx_frame]
            real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
            real_object_pose = real_object_pose[: maxx_frame]
            cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
            
            tot_goal_obj_pose = tot_goal_obj_pose[: maxx_frame]
            tot_goal_hand_qpos = tot_goal_hand_qpos[: maxx_frame]
        
        
            history_len = self.history_length
            # action_chunks = 10
            action_chunks = self.action_chunking_frames
            
            for i_ts in range(real_arm_pos.shape[0]):
                tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
                tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
                # tot history ts #
                tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
                # nn_ts x nn_envs x nn_feature_dim # # nn_feature_dim #
                history_real_arm_pos = real_arm_pos[tot_history_ts]
                history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
                history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
                
                history_actions = tot_qtars[tot_history_ts]
                
                tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
                tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
                tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
                future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
                
                
                
                tot_ref_ts = [i_ts + i for i in range(0, action_chunks)]
                tot_ref_ts = [ min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_ref_ts]
                tot_ref_ts = torch.tensor(tot_ref_ts, ).to(self.device).long()
                goal_hand_qpos = tot_goal_hand_qpos[tot_ref_ts]
                goal_obj_pose = tot_goal_obj_pose[tot_ref_ts]
                
                
                if self.use_no_obj_pose:
                    history_real_object_pose[..., :] = 0.0
                    history_real_object_pose[..., -1] = 1.0
                
                if self.history_chunking_obs_version == 'v1':
                    cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose, history_actions], dim=-1)
                elif self.history_chunking_obs_version == 'v2':
                    cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose], dim=-1)
                else:
                    raise ValueError(f"Unrecognized history chunking obs version: {self.history_chunking_obs_version}")
                
                cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
                cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
                
                goal_obs = torch.cat([goal_hand_qpos, goal_obj_pose], dim=-1)
                goal_obs = goal_obs.contiguous().transpose(1, 0) 
                goal_obs = goal_obs.contiguous().view(goal_obs.size(0), -1).contiguous() 
                
                
                if self.history_chunking_obs_version in ['v1']:
                    cur_obs = torch.cat(
                        [ cur_obs, goal_obs ], dim=-1
                    )
                    
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
                
                if i_ts < real_arm_pos.shape[0] - 1:
                    cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                else:
                    cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
                
                
                gt_act.append(future_cur_step_already_execute_actions)
                obses.append(cur_obs)
                dones.append(cur_dones)
                rewards.append(cur_rewards)
                
            obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
            dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
            rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
            gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
            
            tot_obses.append(obses)
            tot_dones.append(dones)
            tot_rewards.append(rewards)
            tot_gt_act.append(gt_act)
        
        tot_obses = torch.cat(tot_obses, dim=1)
        tot_dones = torch.cat(tot_dones, dim=1)
        tot_rewards = torch.cat(tot_rewards, dim=1)
        tot_gt_act = torch.cat(tot_gt_act, dim=1)
            
        self.preload_experiences = {
            'obses' : tot_obses,
            'dones' : tot_dones,
            'rewards' : tot_rewards,
            'actions' : tot_gt_act,
            'neglogpacs': torch.zeros_like(tot_dones),
            'values': torch.zeros_like(tot_dones).unsqueeze(-1),
            'mus' : tot_gt_act,
            'sigmas' : tot_gt_act,
        }
        
        for key in self.preload_experiences:
            val = self.preload_experiences[key]
            print(f"key: {key}, val: {val.size()}")
    
    
    def preload_multiple_saved_experiences_realdemo(self, ):
        # preload # # preload #
        # 
        
        tot_real_replay_info_fns = os.listdir(self.preload_experiences_path)
        tot_real_replay_info_fns = [
            fn for fn in tot_real_replay_info_fns if fn.endswith(".npy")
        ]
        tot_obses = []
        tot_dones = []
        tot_rewards = []
        tot_gt_act = []
        
        goal_kine_qs_w_arm = self.vec_env.env.tot_kine_qs_w_arm # nn_insts x nn_ts x nn_dims 
        goal_kine_obj_trans = self.vec_env.env.tot_kine_obj_trans
        goal_kine_obj_ornt = self.vec_env.env.tot_kine_obj_ornt
        
        print(f"goal_kine_qs_w_arm: {goal_kine_qs_w_arm.size()}, goal_kine_obj_trans: {goal_kine_obj_trans.size()}, goal_kine_obj_ornt: {goal_kine_obj_ornt.size()} ")
        
        if len(goal_kine_qs_w_arm.size()) == 3:
            goal_kine_qs_w_arm = goal_kine_qs_w_arm[0].unsqueeze(1)
            goal_kine_obj_trans = goal_kine_obj_trans[0].unsqueeze(1)
            goal_kine_obj_ornt = goal_kine_obj_ornt[0].unsqueeze(1)
            
        goal_kine_obj_pose = torch.cat(
            [ goal_kine_obj_trans, goal_kine_obj_ornt ], dim=-1
        )
        
        for cur_real_replay_info_fn in tot_real_replay_info_fns:
            cur_full_replay_info_fn = os.path.join(self.preload_experiences_path, cur_real_replay_info_fn)
            cur_full_replay_info = np.load(cur_full_replay_info_fn, allow_pickle=True).item() #
            # { ts: dict_keys(['real_arm_pos', 'real_leap_pos_to_sim', 'real_object_pose', 'cur_step_already_execute_actions', 'sim_hand_qpos', 'sim_fingertip_pos', 'sim_object_pose']) }
            tot_ts = list(cur_full_replay_info.keys())
            tot_ts = [ int(cur_ts)for  cur_ts in tot_ts if isinstance(cur_ts, int) ]
            tot_ts = sorted(tot_ts)
            
            # 
            
            tot_qtars = []
            tot_obj_pose = []
            tot_states = []
            
            obses = []
            dones = []
            rewards = []
            gt_act = []
            
            for i_ts in tot_ts:
                cur_ts_exp_dict = cur_full_replay_info[i_ts]
                cur_real_arm_pos = cur_ts_exp_dict['real_arm_pos']
                cur_real_pos_to_sim = cur_ts_exp_dict['real_leap_pos_to_sim']
                cur_real_obj_pose = cur_ts_exp_dict['real_object_pose']
                cur_qtars = cur_ts_exp_dict['cur_step_already_execute_actions']
                
                cur_state = np.concatenate(
                    [ cur_real_arm_pos, cur_real_pos_to_sim ], axis=-1
                )
                tot_states.append(cur_state)
                tot_obj_pose.append(cur_real_obj_pose)
                tot_qtars.append(cur_qtars)
            tot_states = np.stack(tot_states, axis=0) # 
            tot_obj_pose = np.stack(tot_obj_pose, axis=0)
            tot_qtars = np.stack(tot_qtars, axis=0) # nn_ts x nn_feat_dim
            
            
            print(f"tot_states: {tot_states.shape},tot_obj_pose: {tot_obj_pose.shape}, tot_qtars: {tot_qtars.shape} ")
            
            tot_states = torch.from_numpy(tot_states).float().to(self.device).unsqueeze(1)
            tot_obj_pose = torch.from_numpy(tot_obj_pose).float().to(self.device).unsqueeze(1)
            tot_qtars = torch.from_numpy(tot_qtars).float().to(self.device).unsqueeze(1)
            
            
            real_arm_pos  = tot_states[..., :7]
            real_leap_pos_to_sim = tot_states[..., 7:]
            real_object_pose = tot_obj_pose
            cur_step_already_execute_actions = tot_qtars
        
            maxx_frame = 300
            real_arm_pos = real_arm_pos[: maxx_frame]
            real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
            real_object_pose = real_object_pose[: maxx_frame]
            cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
            
            # tot_goal_obj_pose = tot_goal_obj_pose[: maxx_frame]
            # tot_goal_hand_qpos = tot_goal_hand_qpos[: maxx_frame]
            
            history_len = self.history_length
            # action_chunks = 10
            action_chunks = self.action_chunking_frames
            
            for i_ts in range(real_arm_pos.shape[0]):
                tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
                tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
                # tot history ts #
                tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
                # nn_ts x nn_envs x nn_feature_dim # # nn_feature_dim #
                history_real_arm_pos = real_arm_pos[tot_history_ts]
                history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
                history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
                
                history_actions = tot_qtars[tot_history_ts]
                
                tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
                tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
                tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
                future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
                
                
                
                tot_ref_ts = [i_ts + i for i in range(0, action_chunks)]
                tot_ref_ts = [ min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_ref_ts]
                tot_ref_ts = torch.tensor(tot_ref_ts, ).to(self.device).long()
                goal_hand_qpos = goal_kine_qs_w_arm[tot_ref_ts]
                goal_obj_pose = goal_kine_obj_pose[tot_ref_ts]
                
                
                if self.use_no_obj_pose:
                    history_real_object_pose[..., :] = 0.0
                    history_real_object_pose[..., -1] = 1.0
                
                if self.history_chunking_obs_version == 'v1':
                    cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose, history_actions], dim=-1)
                elif self.history_chunking_obs_version == 'v2':
                    cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose], dim=-1)
                else:
                    raise ValueError(f"Unrecognized history chunking obs version: {self.history_chunking_obs_version}")
                
                # history_length x 1 x nn_dim
                cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
                cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
                
                goal_obs = torch.cat([goal_hand_qpos, goal_obj_pose], dim=-1)
                goal_obs = goal_obs.contiguous().transpose(1, 0) 
                goal_obs = goal_obs.contiguous().view(goal_obs.size(0), -1).contiguous() 
                
                
                if self.history_chunking_obs_version in ['v1']:
                    cur_obs = torch.cat(
                        [ cur_obs, goal_obs ], dim=-1
                    )
                    
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
                
                if i_ts < real_arm_pos.shape[0] - 1:
                    cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                else:
                    cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
                
                gt_act.append(future_cur_step_already_execute_actions)
                obses.append(cur_obs)
                dones.append(cur_dones)
                rewards.append(cur_rewards)
                
            obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
            dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
            rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
            gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
            
            tot_obses.append(obses)
            tot_dones.append(dones)
            tot_rewards.append(rewards)
            tot_gt_act.append(gt_act)
        
        
        tot_obses = torch.cat(tot_obses, dim=1)
        tot_dones = torch.cat(tot_dones, dim=1)
        tot_rewards = torch.cat(tot_rewards, dim=1)
        tot_gt_act = torch.cat(tot_gt_act, dim=1)
        
        # num_actors #
        
        self.preload_experiences = {
            'obses' : tot_obses,
            'dones' : tot_dones,
            'rewards' : tot_rewards,
            'actions' : tot_gt_act,
            'neglogpacs': torch.zeros_like(tot_dones),
            'values': torch.zeros_like(tot_dones).unsqueeze(-1),
            'mus' : tot_gt_act,
            'sigmas' : tot_gt_act,
        }
        
        for key in self.preload_experiences:
            if self.preload_experiences[key].size(1) < self.num_actors:
                print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                self.preload_experiences[key] = torch.cat(
                    [
                        self.preload_experiences[key] for _ in range(self.num_actors // self.preload_experiences[key].size(1))
                    ], dim=1
                )
                if self.preload_experiences[key].size(1) < self.num_actors:
                    # print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                    self.preload_experiences[key] = torch.cat(
                        [
                            self.preload_experiences[key], self.preload_experiences[key][:, : self.num_actors - self.preload_experiences[key].size(1)]
                        ], dim=1
                    )
        
        
        for key in self.preload_experiences:
            val = self.preload_experiences[key]
            print(f"key: {key}, val: {val.size()}")
        
        
        
    
    
    # distill_via_bc #
    # TODO: should increase the max step aprameter in the pre presaved experience function #
    def preload_saved_experiences_simdemo(self, ):
        cur_preload_exp_path = self.preload_experiences_path
        preload_exp = np.load(cur_preload_exp_path, allow_pickle=True).item() # load info from the file #
        # preload_exp_keys = list(preload_exp.keys())
        
        hand_dof_pos = preload_exp['shadow_hand_dof_pos']
        hand_dof_tars = preload_exp['shadow_hand_dof_tars']
        ref_dof_pos = preload_exp['next_ref_np']
        obj_pose = preload_exp['object_pose']
        ref_obj_pose = preload_exp['goal_pose_ref_np'] # goal pose ref np and the next ref np #
        
        hand_dof_pos = torch.from_numpy(hand_dof_pos).float().to(self.device)
        hand_dof_tars = torch.from_numpy(hand_dof_tars).float().to(self.device)
        ref_dof_pos = torch.from_numpy(ref_dof_pos).float().to(self.device)
        obj_pose = torch.from_numpy(obj_pose).float().to(self.device)
        ref_obj_pose = torch.from_numpy(ref_obj_pose).float().to(self.device)
        
        # print(f"")
        for key in preload_exp:
            cur_val = preload_exp[key]
            print(f"key: {key}, cur_val: {cur_val.shape}")
        # maxx_frame = 500
        # real_arm_pos = real_arm_pos[: maxx_frame]
        # real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
        # real_object_pose = real_object_pose[: maxx_frame]
        # cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
        
        ## TODO: should set theose parameters (in the action chunking fashion)
        ## TODO: set parameters -- history_length, action_chunking_frames, action_chunking_skip_frames
        
        # TODO: should carefully design the observations and the actions #
        # history is still the dof pos and obejct pose --- with the histry stack #
        # observation should include the nex chunk's ref poses #
        # action is the next action to be executed
        
        
        history_len = self.history_length
        # action_chunks = 10
        action_chunks = self.action_chunking_frames
        obses = []
        dones = []
        rewards = []
        gt_act = []
        for i_ts in range(0, hand_dof_pos.shape[0], self.action_chunking_skip_frames):
            tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
            tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
            # tot history ts #
            tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
            # nn_ts x nn_envs x nn_feature_dim #
            
            tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
            tot_future_ts = [min(hand_dof_tars.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
            tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
            
            tot_ref_ts = [i_ts + i for i in range(0, action_chunks)]
            tot_ref_ts = [ min(cur_ts, hand_dof_tars.size(0) - 1) for cur_ts in tot_ref_ts  ]
            tot_ref_ts = torch.tensor(tot_ref_ts).to(self.device).long()
            
            history_hand_dof_pos = hand_dof_pos[tot_history_ts]
            history_obj_pose = obj_pose[tot_history_ts]
            # future_ref_hand
            ref_hand_dof_pos = ref_dof_pos[tot_ref_ts]
            cur_ref_obj_pose = ref_obj_pose[tot_ref_ts]
            
            future_hand_dof_qtars = hand_dof_tars[tot_future_ts] # future actions 
            
            history_hand_w_obj_info = torch.cat(
                [ history_hand_dof_pos, history_obj_pose ], dim=-1 # nn_history_ts x nn_envs x (nn_hand_dof + 7)
            )
            ref_hand_w_obj_info = torch.cat(
                [ ref_hand_dof_pos, cur_ref_obj_pose ], dim=-1 # nn_ref_ts x nn_envs x (nn_hand_dof + 7)
            )
            
            history_hand_w_obj_info = history_hand_w_obj_info.contiguous().transpose(1, 0).contiguous()
            history_hand_w_obj_info = history_hand_w_obj_info.contiguous().view(history_hand_w_obj_info.size(0), -1).contiguous()
            
            ref_hand_w_obj_info = ref_hand_w_obj_info.contiguous().transpose(1, 0).contiguous()
            ref_hand_w_obj_info = ref_hand_w_obj_info.contiguous().view(ref_hand_w_obj_info.size(0), -1).contiguous()
            
            
            cur_obs = torch.cat(
                [ history_hand_w_obj_info, ref_hand_w_obj_info ], dim=-1 # nn_envs x nn_obs_dim --- calculate the observations
            )
            # history_hand_w_obj_info = history_hand_w_obj_info.contiguous()
            # # .view(-1) 
            # ref_hand_w_obj_info = ref_hand_w_obj_info.contiguous().view(-1) # 
            
            cur_actions = future_hand_dof_qtars.contiguous().transpose(1, 0).contiguous() # nn_envs x nn_future_ts x nn_dof_pos
            cur_actions = cur_actions.contiguous().view(cur_actions.size(0), -1).contiguous() # nn_envs x (nn_future_ts x nn_dof_pos)
            # the actions is the real targets -! #
            
            
            
            # history_real_arm_pos = real_arm_pos[tot_history_ts]
            # history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
            # history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
            
            # tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
            # tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
            # tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
            # future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
            
            # if self.use_no_obj_pose:
            #     history_real_object_pose[..., :] = 0.0
            #     history_real_object_pose[..., -1] = 1.0
            
            # cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose], dim=-1)
            # cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
            # cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
            
            # future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
            # future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
            # hand dof tars #
            if i_ts < hand_dof_tars.size(0) - 1:
                cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
            else:
                cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
            cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
            gt_act.append(cur_actions)
            obses.append(cur_obs)
            dones.append(cur_dones)
            rewards.append(cur_rewards)
            
        # nn_ts x nn_envs x nn_feature_dim #
        obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
        
        # TODO: add_obj_features can be set to False in the beginning #
        if self.add_obj_features:
            obj_latent_feat = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
            obj_latent_feat_dict = np.load(obj_latent_feat, allow_pickle=True).item()
            obj_feature = obj_latent_feat_dict[self.test_inst_tag.split('_nf_300')[0]]
            obj_feature = torch.from_numpy(obj_feature).float().to(self.device)
            obj_feature = obj_feature.unsqueeze(0).unsqueeze(0).repeat(obses.size(0), obses.size(1), 1)
            obses = torch.cat([ obses, obj_feature ], dim=-1)
            
        
        dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
        rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
        gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
        self.preload_experiences = {
            'obses' : obses,
            'dones' : dones,
            'rewards' : rewards,
            'actions' : gt_act,
            'neglogpacs': torch.zeros_like(dones),
            'values': torch.zeros_like(dones).unsqueeze(-1),
            'mus' : gt_act,
            'sigmas' : gt_act,
        }
        
        for key in self.preload_experiences:
            val = self.preload_experiences[key]
            print(f"key: {key}, val: {val.size()}")
        
        pass

    def preload_saved_experiences_chunking_bc(self, ):
        ### TODO: load the data and also use the data to compute observations and the actions and others that are needed to compute the information needed to train the network ##
        ### TODO: add the corresponding folder information ###
        # key: 215, real_arm_pos, sub_val: (7,)
        # key: 215, sub_key: real_leap_pos_to_sim, sub_val: (16,)
        # key: 215, sub_key: real_object_pose, sub_val: (7,)
        # key: 215, sub_key: cur_step_already_execute_actions, sub_val: (23,)
        # key: 215, sub_key: sim_hand_qpos, sub_val: (23,)
        # key: 215, sub_key: sim_fingertip_pos, sub_val: (4, 3)
        # key: 215, sub_key: sim_object_pose, sub_val: (7,)
        ### key and sub_key ###
        
        if self.simreal_modeling:
            self.preload_saved_experiences_chunking_bc_simrealseq()
            return
        
        if os.path.isdir(self.preload_experiences_path):
            def load_one_experience_path(cur_full_fn):
                print(f"Loading from {cur_full_fn}")
                preload_experiences = np.load(cur_full_fn, allow_pickle=True).item()
                tot_ts = list(preload_experiences.keys())
                tot_ts = [int(cur_ts) for cur_ts in tot_ts]
                tot_subkeys = list(preload_experiences[tot_ts[0]].keys())
                subkey_to_tot_vals = {}
                for cur_ts in tot_ts:
                    for cur_subkey in tot_subkeys:
                        cur_sub_val = preload_experiences[cur_ts][cur_subkey] # 
                        if cur_subkey not in subkey_to_tot_vals:
                            subkey_to_tot_vals[cur_subkey] = [cur_sub_val]
                        else:
                            subkey_to_tot_vals[cur_subkey].append(cur_sub_val)
                for subkey in subkey_to_tot_vals:
                    subkey_to_tot_vals[subkey] = np.stack(subkey_to_tot_vals[subkey], axis=0) # nnn_ts x nn_feature_dim
                return subkey_to_tot_vals

            tot_fns = os.listdir(self.preload_experiences_path)
            tot_fns = [ fn for fn in tot_fns if fn.endswith('.npy')]
            self.preload_experiences = {}
            for cur_fn in tot_fns:
                cur_full_fn = os.path.join(self.preload_experiences_path, cur_fn)
                cur_subkey_to_tot_vals = load_one_experience_path(cur_full_fn)
                for subkey in cur_subkey_to_tot_vals:
                    if subkey not in self.preload_experiences:
                        self.preload_experiences[subkey] = [ cur_subkey_to_tot_vals[subkey] ]
                    else:
                        self.preload_experiences[subkey].append(cur_subkey_to_tot_vals[subkey])
            for subkey in self.preload_experiences:
                self.preload_experiences[subkey] = np.stack(self.preload_experiences[subkey], axis=1) # nn_ts x nn_envs x nn_feature_dim
            
            ## TODO: construct the observations and the actions and the other information needed to train the network ##
            # obs = hist_obs * (23 + 7)
            # actions = action_chuking_nn * 23
            # obses, dones, rewards, gt_act
            real_arm_pos = self.preload_experiences['real_arm_pos']
            real_leap_pos_to_sim = self.preload_experiences['real_leap_pos_to_sim']
            real_object_pose = self.preload_experiences['real_object_pose']
            cur_step_already_execute_actions = self.preload_experiences['cur_step_already_execute_actions']
            
            
            # if 'shadow_hand_dof_tars' in self.preload_experiences: 
            #     # demonstrations from the sim #
            #     shadow_hand_dof_tars = self.preload_experiences['shadow_hand_dof_tars'] # 
            #     pass
            
            # ### try to only use the first trajectory ###
            # print(f"real_arm_pos: {real_arm_pos.shape}, real_leap_pos_to_sim: {real_leap_pos_to_sim.shape}, real_object_pose: {real_object_pose.shape}, cur_step_already_execute_actions: {cur_step_already_execute_actions.shape}")
            # ### try to only use the first trajectory ###
            # tot_envs = real_arm_pos.shape[1]
            # real_arm_pos = real_arm_pos[:, 0:1] 
            # real_leap_pos_to_sim = real_leap_pos_to_sim[:, 0:1] # .repeat(1, tot_envs, 1).contiguous()
            # real_object_pose = real_object_pose[:, 0:1] # .repeat(1, tot_envs, 1).contiguous()
            # cur_step_already_execute_actions = cur_step_already_execute_actions[:, 0:1] # .repeat(1, tot_envs, 1).contiguous()
            # real_arm_pos = np.concatenate(
            #     [ real_arm_pos for _ in range(tot_envs) ], axis=1
            # )
            # real_leap_pos_to_sim = np.concatenate(
            #     [ real_leap_pos_to_sim for _ in range(tot_envs) ], axis=1
            # )
            # real_object_pose = np.concatenate(
            #     [ real_object_pose for _ in range(tot_envs) ], axis=1
            # )
            # cur_step_already_execute_actions = np.concatenate(
            #     [ cur_step_already_execute_actions for _ in range(tot_envs) ], axis=1
            # )
            # ### try to only use the first trajectory ###
            
            
            
            real_arm_pos = torch.from_numpy(real_arm_pos).float().to(self.device)   
            real_leap_pos_to_sim = torch.from_numpy(real_leap_pos_to_sim).float().to(self.device)
            real_object_pose = torch.from_numpy(real_object_pose).float().to(self.device)
            cur_step_already_execute_actions = torch.from_numpy(cur_step_already_execute_actions).float().to(self.device)
            
            if self.bc_relative_targets:
                # cur_step_already_execute_actions: nn_ts x nn_envs x nn_act_dim
                real_init_state = torch.cat( # nn_ts x nn_envs x nn_act_dim
                    [ real_arm_pos, real_leap_pos_to_sim ], dim=-1
                )[0:1]
                # real_init_state = 
                # # 
                every_step_targets = torch.cat(
                    [ real_init_state, cur_step_already_execute_actions ], dim=0 # (nn_ts, nn_envs, nn_act_dim)
                )
                cur_step_already_execute_actions = every_step_targets[1:] - every_step_targets[:-1]
                # 
                
            maxx_frame = 300
            real_arm_pos = real_arm_pos[: maxx_frame]
            real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
            real_object_pose = real_object_pose[: maxx_frame]
            cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
            
            
            
            history_len = self.history_length
            # action_chunks = 10
            action_chunks = self.action_chunking_frames
            obses = []
            dones = []
            rewards = []
            gt_act = []
            for i_ts in range(0, real_arm_pos.shape[0], self.action_chunking_skip_frames):
                tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
                tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
                # tot history ts #
                tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
                # nn_ts x nn_envs x nn_feature_dim # # nn_feature_dim #
                history_real_arm_pos = real_arm_pos[tot_history_ts]
                history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
                history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
                
                tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
                tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
                tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
                future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
                
                if self.use_no_obj_pose:
                    history_real_object_pose[..., :] = 0.0
                    history_real_object_pose[..., -1] = 1.0
                
                cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_real_object_pose], dim=-1)
                cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
                cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
                
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
                
                if i_ts < real_arm_pos.shape[0] - 1:
                    cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                else:
                    cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
                gt_act.append(future_cur_step_already_execute_actions)
                obses.append(cur_obs)
                dones.append(cur_dones)
                rewards.append(cur_rewards)
                
            obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
            
            if self.add_obj_features:
                obj_latent_feat = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
                obj_latent_feat_dict = np.load(obj_latent_feat, allow_pickle=True).item()
                obj_feature = obj_latent_feat_dict[self.test_inst_tag.split('_nf_300')[0]]
                obj_feature = torch.from_numpy(obj_feature).float().to(self.device)
                obj_feature = obj_feature.unsqueeze(0).unsqueeze(0).repeat(obses.size(0), obses.size(1), 1)
                obses = torch.cat([ obses, obj_feature ], dim=-1)
                
            
            dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
            rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
            gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
            self.preload_experiences = {
                'obses' : obses,
                'dones' : dones,
                'rewards' : rewards,
                'actions' : gt_act,
                'neglogpacs': torch.zeros_like(dones),
                'values': torch.zeros_like(dones).unsqueeze(-1),
                'mus' : gt_act,
                'sigmas' : gt_act,
            }
            
            for key in self.preload_experiences:
                val = self.preload_experiences[key]
                print(f"key: {key}, val: {val.size()}")
        else:
            self.preload_saved_experiences_chunking_bc_simdemo()
        pass
    
    def preload_saved_experiences_chunking_bc_simrealseq(self, ):
        ### TODO: load the data and also use the data to compute observations and the actions and others that are needed to compute the information needed to train the network ##
        ### TODO: add the corresponding folder information ###
        # key: 215, real_arm_pos, sub_val: (7,)
        # key: 215, sub_key: real_leap_pos_to_sim, sub_val: (16,)
        # key: 215, sub_key: real_object_pose, sub_val: (7,)
        # key: 215, sub_key: cur_step_already_execute_actions, sub_val: (23,)
        # key: 215, sub_key: sim_hand_qpos, sub_val: (23,)
        # key: 215, sub_key: sim_fingertip_pos, sub_val: (4, 3)
        # key: 215, sub_key: sim_object_pose, sub_val: (7,)
        ### key and sub_key ###
        
        if os.path.isdir(self.preload_experiences_path):
            def load_one_experience_path(cur_full_fn):
                print(f"Loading from {cur_full_fn}")
                preload_experiences = np.load(cur_full_fn, allow_pickle=True).item()
                tot_ts = list(preload_experiences.keys())
                tot_ts = [int(cur_ts) for cur_ts in tot_ts]
                tot_subkeys = list(preload_experiences[tot_ts[0]].keys())
                subkey_to_tot_vals = {}
                for cur_ts in tot_ts:
                    for cur_subkey in tot_subkeys:
                        cur_sub_val = preload_experiences[cur_ts][cur_subkey] # 
                        if cur_subkey not in subkey_to_tot_vals:
                            subkey_to_tot_vals[cur_subkey] = [cur_sub_val]
                        else:
                            subkey_to_tot_vals[cur_subkey].append(cur_sub_val)
                for subkey in subkey_to_tot_vals:
                    subkey_to_tot_vals[subkey] = np.stack(subkey_to_tot_vals[subkey], axis=0) # nnn_ts x nn_feature_dim
                return subkey_to_tot_vals

            tot_fns = os.listdir(self.preload_experiences_path)
            tot_fns = [ fn for fn in tot_fns if fn.endswith('.npy')]
            self.preload_experiences = {}
            for cur_fn in tot_fns:
                cur_full_fn = os.path.join(self.preload_experiences_path, cur_fn)
                cur_subkey_to_tot_vals = load_one_experience_path(cur_full_fn)
                for subkey in cur_subkey_to_tot_vals:
                    if subkey not in self.preload_experiences:
                        self.preload_experiences[subkey] = [ cur_subkey_to_tot_vals[subkey] ]
                    else:
                        self.preload_experiences[subkey].append(cur_subkey_to_tot_vals[subkey])
            for subkey in self.preload_experiences:
                self.preload_experiences[subkey] = np.stack(self.preload_experiences[subkey], axis=1) # nn_ts x nn_envs x nn_feature_dim
            
            ## TODO: construct the observations and the actions and the other information needed to train the network ##
            # obs = hist_obs * (23 + 7)
            # actions = action_chuking_nn * 23
            # obses, dones, rewards, gt_act
            real_arm_pos = self.preload_experiences['real_arm_pos']
            real_leap_pos_to_sim = self.preload_experiences['real_leap_pos_to_sim']
            real_object_pose = self.preload_experiences['real_object_pose']
            cur_step_already_execute_actions = self.preload_experiences['cur_step_already_execute_actions']
            # sim
            sim_hand_qpos = self.preload_experiences['sim_hand_qpos']
            
            # if 'shadow_hand_dof_tars' in self.preload_experiences: 
            #     # demonstrations from the sim #
            #     shadow_hand_dof_tars = self.preload_experiences['shadow_hand_dof_tars'] # 
            #     pass
            
            # ### try to only use the first trajectory ###
            # print(f"real_arm_pos: {real_arm_pos.shape}, real_leap_pos_to_sim: {real_leap_pos_to_sim.shape}, real_object_pose: {real_object_pose.shape}, cur_step_already_execute_actions: {cur_step_already_execute_actions.shape}")
            # ### try to only use the first trajectory ###
            # tot_envs = real_arm_pos.shape[1]
            # real_arm_pos = real_arm_pos[:, 0:1] 
            # real_leap_pos_to_sim = real_leap_pos_to_sim[:, 0:1] # .repeat(1, tot_envs, 1).contiguous()
            # real_object_pose = real_object_pose[:, 0:1] # .repeat(1, tot_envs, 1).contiguous()
            # cur_step_already_execute_actions = cur_step_already_execute_actions[:, 0:1] # .repeat(1, tot_envs, 1).contiguous()
            # real_arm_pos = np.concatenate(
            #     [ real_arm_pos for _ in range(tot_envs) ], axis=1
            # )
            # real_leap_pos_to_sim = np.concatenate(
            #     [ real_leap_pos_to_sim for _ in range(tot_envs) ], axis=1
            # )
            # real_object_pose = np.concatenate(
            #     [ real_object_pose for _ in range(tot_envs) ], axis=1
            # )
            # cur_step_already_execute_actions = np.concatenate(
            #     [ cur_step_already_execute_actions for _ in range(tot_envs) ], axis=1
            # )
            # ### try to only use the first trajectory ###
            
            
            
            real_arm_pos = torch.from_numpy(real_arm_pos).float().to(self.device)   
            real_leap_pos_to_sim = torch.from_numpy(real_leap_pos_to_sim).float().to(self.device)
            real_object_pose = torch.from_numpy(real_object_pose).float().to(self.device)
            cur_step_already_execute_actions = torch.from_numpy(cur_step_already_execute_actions).float().to(self.device)
            sim_hand_qpos = torch.from_numpy(sim_hand_qpos).float().to(self.device) # sim hand qpos #
            
            # bc relative targets #
            if self.bc_relative_targets:
                # cur_step_already_execute_actions: nn_ts x nn_envs x nn_act_dim
                # get the step execute actions #
                real_init_state = torch.cat( # nn_ts x nn_envs x nn_act_dim
                    [ real_arm_pos, real_leap_pos_to_sim ], dim=-1
                )[0:1]
                # real_init_state = 
                # # 
                every_step_targets = torch.cat(
                    [ real_init_state, cur_step_already_execute_actions ], dim=0 # (nn_ts, nn_envs, nn_act_dim)
                )
                cur_step_already_execute_actions = every_step_targets[1:] - every_step_targets[:-1]
                # 
                
            maxx_frame = 300
            real_arm_pos = real_arm_pos[: maxx_frame]
            real_leap_pos_to_sim = real_leap_pos_to_sim[: maxx_frame]
            real_object_pose = real_object_pose[: maxx_frame]
            cur_step_already_execute_actions = cur_step_already_execute_actions[: maxx_frame]
            sim_hand_qpos = sim_hand_qpos[: maxx_frame]
            
            
            history_len = self.history_length
            action_chunks = 10 
            obses = []
            dones = []
            rewards = []
            gt_act = []
            for i_ts in range(real_arm_pos.shape[0]): 
                tot_history_ts = [ i_ts - history_len + i + 1 for i in range(0, history_len) ]
                tot_history_ts = [ max(0, cur_ts) for cur_ts in tot_history_ts ] # tot history ts # #
                # tot history ts #
                tot_history_ts = torch.tensor(tot_history_ts).to(self.device).long() # (nn_history_len, )
                # nn_ts x nn_envs x nn_feature_dim # # nn_feature_dim #
                history_real_arm_pos = real_arm_pos[tot_history_ts]
                history_real_leap_pos_to_sim = real_leap_pos_to_sim[tot_history_ts]
                history_real_object_pose = real_object_pose[tot_history_ts] # real object pose #
                history_sim_hand_qpos = sim_hand_qpos[tot_history_ts] # nn_history_len x nn_hand_qpos #
                # tot history #
                history_actions = cur_step_already_execute_actions[tot_history_ts] # nn_history_len x nn_act_dim #
                
                # tot_future_ts = [i_ts + i + 1 for i in range(0, action_chunks)]
                # tot_future_ts = [min(real_arm_pos.shape[0] - 1, cur_ts) for cur_ts in tot_future_ts]
                # tot_future_ts = torch.tensor(tot_future_ts).to(self.device).long()
                # future_cur_step_already_execute_actions = cur_step_already_execute_actions[tot_future_ts]
                
                cur_obs = torch.cat([history_real_arm_pos, history_real_leap_pos_to_sim, history_actions], dim=-1)
                cur_obs = cur_obs.contiguous().transpose(1, 0) ## nn_envs x nn_ts x nn_feature_dim
                cur_obs = cur_obs.contiguous().view(cur_obs.size(0), -1).contiguous() # get the cur_obs
                
                # future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().transpose(1, 0)
                # future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
                
                future_cur_step_already_execute_actions = history_sim_hand_qpos.contiguous().transpose(1, 0)
                future_cur_step_already_execute_actions = future_cur_step_already_execute_actions.contiguous().view(future_cur_step_already_execute_actions.size(0), -1).contiguous() # nn_envs x (some feature dims)
                
                if i_ts < real_arm_pos.shape[0] - 1:
                    cur_dones = torch.zeros(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                else:
                    cur_dones = torch.ones(cur_obs.size(0), dtype=torch.uint8, device=self.device)
                cur_rewards = torch.zeros(cur_obs.size(0), 1, dtype=torch.float32, device=self.device)
                gt_act.append(future_cur_step_already_execute_actions)
                obses.append(cur_obs)
                dones.append(cur_dones)
                rewards.append(cur_rewards)
                
            obses = torch.stack(obses, dim=0) # nn_ts x nn_envs x nn_feature_dim
            dones = torch.stack(dones, dim=0) # nn_ts x nn_envs
            rewards = torch.stack(rewards, dim=0) # nn_ts x nn_envs x 1
            gt_act = torch.stack(gt_act, dim=0) # nn_ts x nn_envs x nn_feature_dim
            self.preload_experiences = {
                'obses' : obses,
                'dones' : dones,
                'rewards' : rewards,
                'actions' : gt_act,
                'neglogpacs': torch.zeros_like(dones),
                'values': torch.zeros_like(dones).unsqueeze(-1),
                'mus' : gt_act,
                'sigmas' : gt_act,
            }
            
            for key in self.preload_experiences:
                val = self.preload_experiences[key]
                print(f"key: {key}, val: {val.size()}")
        else:
            self.preload_saved_experiences_chunking_bc_simdemo()
        pass
    
    
    def preload_all_saved_experiences(self, ):
        print(f"Start loading all saved experiences...")
        saved_experiences_folder_path_st = f"/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_specialist_s"
        tot_saved_experience_folders = [ f"{saved_experiences_folder_path_st}{i}_" for i in range(2, 11)]
        tot_saved_experience_fns = []
        experience_st_tag = "experience_buffer_sv_"
        for cur_folder in tot_saved_experience_folders:
            if not os.path.exists(cur_folder):
                continue
            tot_exp_subfolders = os.listdir(cur_folder)
            for cur_exp_subfolder in tot_exp_subfolders:
                cur_full_exp_subfolder = os.path.join(cur_folder, cur_exp_subfolder)    
                tot_exp_fns = os.listdir(cur_full_exp_subfolder)
                tot_exp_fns = [ cur_fn for cur_fn in tot_exp_fns if cur_fn.endswith(".npy") and cur_fn[: len(experience_st_tag)] == experience_st_tag ]
                if len(tot_exp_fns) == 0:
                    continue
                cur_exp_subfolder_fn = tot_exp_fns[0]
                cur_exp_subfolder_full_fn = os.path.join(cur_full_exp_subfolder, cur_exp_subfolder_fn)
                tot_saved_experience_fns.append(cur_exp_subfolder_full_fn)
        
        self.tot_saved_experience_fns = tot_saved_experience_fns
        self.exp_loading_st_idx = 0
        self.reload_experiences()
        
    def reload_experiences(self, ):        
        
        self.preload_experiences = {}
        cur_fn_st = self.exp_loading_st_idx
        cur_fn_ed = self.exp_loading_st_idx + 5
        # cur_fn_ed = self.exp_loading_st_idx + 10
        for i_fn in range(cur_fn_st, cur_fn_ed):
        # for i_fn, cur_fn in enumerate(tot_saved_experience_fns):
        #     print(f"{i_fn}/{len(tot_saved_experience_fns)} Loading from {cur_fn}")
            cur_fn = self.tot_saved_experience_fns[i_fn]
            preload_experiences = np.load(cur_fn, allow_pickle=True).item()
            maxx_nn = 2000
            for key in preload_experiences:
                if key not in self.preload_experiences:
                        self.preload_experiences[key] = [ torch.from_numpy(preload_experiences[key][:, :maxx_nn]).float().to(self.device) ]
                else:
                    self.preload_experiences[key].append( torch.from_numpy(preload_experiences[key][:, :maxx_nn]).float().to(self.device) )  #
        
        self.exp_loading_st_idx = cur_fn_ed
        
        for key in self.preload_experiences:
            print(f"Processing {key}") # nn_ts x nn_envs x n_feature_dim #
            # self.preload_experiences[key] = np.concatenate(self.preload_experiences[key], axis=1)
            # print(f"After concatenating, start moving to torch and the gpu")
            # self.preload_experiences[key] = torch.from_numpy(self.preload_experiences[key]).float().to(self.device)
            
            self.preload_experiences[key] = torch.cat(
                self.preload_experiences[key], dim=1
            )
        for key in self.preload_experiences:
            if self.preload_experiences[key].size(1) < self.num_actors:
                print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                self.preload_experiences[key] = torch.cat(
                    [
                        self.preload_experiences[key] for _ in range(self.num_actors // self.preload_experiences[key].size(1))
                    ], dim=1
                )
                if self.preload_experiences[key].size(1) < self.num_actors:
                    # print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                    self.preload_experiences[key] = torch.cat(
                        [
                            self.preload_experiences[key], self.preload_experiences[key][:, : self.num_actors - self.preload_experiences[key].size(1)]
                        ], dim=1
                    )
                    
    def reload_experiences_inplace(self, ):        
        
        # self.preload_experiences = {}
        cur_fn_st = self.exp_loading_st_idx
        # cur_fn_ed = self.exp_loading_st_idx + 5
        cur_fn_ed = self.exp_loading_st_idx + 10
        cur_fn_ed = min(cur_fn_ed, len(self.tot_saved_experience_fns))
        cur_idxx = 0
        for i_fn in range(cur_fn_st, cur_fn_ed):
        # for i_fn, cur_fn in enumerate(tot_saved_experience_fns):
        #     print(f"{i_fn}/{len(tot_saved_experience_fns)} Loading from {cur_fn}")
            
            cur_fn = self.tot_saved_experience_fns[i_fn]
            print(f"Loading {i_fn}-th file: {self.tot_saved_experience_fns[i_fn]}")
            preload_experiences = np.load(cur_fn, allow_pickle=True).item()
            tot_exp_data_keys = list(preload_experiences.keys())
            maxx_nn = 2000
            actual_nn = min(maxx_nn, preload_experiences[tot_exp_data_keys[0]].shape[1])
            for key in preload_experiences:
                self.preload_experiences[key][:, cur_idxx: cur_idxx + actual_nn] = torch.from_numpy(preload_experiences[key][:, :actual_nn]).float().to(self.device)
                
                # if key not in self.preload_experiences:
                #         self.preload_experiences[key] = [ torch.from_numpy(preload_experiences[key][:, :maxx_nn]).float().to(self.device) ]
                # else:
                #     self.preload_experiences[key].append( torch.from_numpy(preload_experiences[key][:, :maxx_nn]).float().to(self.device) )  #
            cur_idxx += actual_nn #  preload_experiences[key].shape[1]
        
        # tot_sizes = preload_experiences[key].shape[1] * (cur_fn_ed - cur_fn_st)
        # tot_sizes = cur_idxx
        tot_sizes = actual_nn * (cur_fn_ed - cur_fn_st)
        self.exp_loading_st_idx = cur_fn_ed
        self.exp_loading_st_idx = self.exp_loading_st_idx % len(self.tot_saved_experience_fns)
        
        # for key in self.preload_experiences:
        #     print(f"Processing {key}") # nn_ts x nn_envs x n_feature_dim #
        #     # self.preload_experiences[key] = np.concatenate(self.preload_experiences[key], axis=1)
        #     # print(f"After concatenating, start moving to torch and the gpu")
        #     # self.preload_experiences[key] = torch.from_numpy(self.preload_experiences[key]).float().to(self.device)
            
        #     self.preload_experiences[key] = torch.cat(
        #         self.preload_experiences[key], dim=1
        #     )
        
        while cur_idxx < self.preload_experiences[key].size(1):
            cur_ed_idxx = cur_idxx + tot_sizes
            if cur_ed_idxx > self.preload_experiences[key].size(1):
                cur_ed_idxx = self.preload_experiences[key].size(1)
            for curr_key in self.preload_experiences:
                self.preload_experiences[curr_key][:, cur_idxx: cur_ed_idxx] = self.preload_experiences[curr_key][:, 0: cur_ed_idxx - cur_idxx]
            cur_idxx = cur_ed_idxx
        
        # for key in self.preload_experiences:
        #     if self.preload_experiences[key].size(1) < self.num_actors:
        #         print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
        #         self.preload_experiences[key] = torch.cat(
        #             [
        #                 self.preload_experiences[key] for _ in range(self.num_actors // self.preload_experiences[key].size(1))
        #             ], dim=1
        #         )
        #         if self.preload_experiences[key].size(1) < self.num_actors:
        #             # print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
        #             self.preload_experiences[key] = torch.cat(
        #                 [
        #                     self.preload_experiences[key], self.preload_experiences[key][:, : self.num_actors - self.preload_experiences[key].size(1)]
        #                 ], dim=1
        #             )

    def preload_saved_experiences_multitraj_via_ts(self,):
        assert len(self.traj_idx_to_experience_sv_folder) > 0 and os.path.exists(self.traj_idx_to_experience_sv_folder)
        assert self.save_experiences_via_ts
        traj_idx_to_experience_sv_folder =np.load(self.traj_idx_to_experience_sv_folder, allow_pickle=True).item() 
        self.traj_idx_to_experience_sv_folder = traj_idx_to_experience_sv_folder
        self.tot_traj_idxes = list(self.traj_idx_to_experience_sv_folder.keys())
        self.tot_traj_idxes = sorted(self.tot_traj_idxes)
        
        self.cur_ordered_traj_idx = 0
        
        cur_traj_idx = self.tot_traj_idxes[self.cur_ordered_traj_idx]
        
        self.tot_ts_preload_experiences = {}
        
        
        # for cur_traj_idx in traj_idx_to_experience_sv_folder:
            
        cur_traj_experiences_sv_fn = traj_idx_to_experience_sv_folder[cur_traj_idx]
        assert os.path.isdir(cur_traj_experiences_sv_fn)
        experience_st_tag = "experience_buffer_sv_"
        tot_fns = os.listdir(cur_traj_experiences_sv_fn)
        tot_fns = [ fn for fn in tot_fns if fn[: len(experience_st_tag)] == experience_st_tag and fn.endswith('.npy')] 
        
        
        
        for i_fn, cur_fn in enumerate(tot_fns):
            
            cur_file_ts = cur_fn.split("/")[-1].split(".npy")[0].split("_ts_")[-1] 
            cur_file_ts = int(cur_file_ts)
            if cur_file_ts > self.load_experiences_maxx_ts:
                continue
            print(f"[{cur_traj_idx}/{len(traj_idx_to_experience_sv_folder)}] [{i_fn}/{len(tot_fns)}] loading {cur_fn}")
            # then we should load cur_fn #
            cur_full_fn = os.path.join(cur_traj_experiences_sv_fn, cur_fn)
            cur_ts_preloaded_experiences = np.load(cur_full_fn, allow_pickle=True).item() # 
            # cur ts preloaded experiences #
            
            if cur_file_ts not in self.tot_ts_preload_experiences:
                self.tot_ts_preload_experiences[cur_file_ts] = {}
                for tsr_key in cur_ts_preloaded_experiences:
                    # self.tot_ts_preload_experiences[cur_file_ts][tsr_key] = [cur_ts_preloaded_experiences[tsr_key]]
                    self.tot_ts_preload_experiences[cur_file_ts][tsr_key] = cur_ts_preloaded_experiences[tsr_key]
                # self.tot_ts_preload_experiences[cur_file_ts] = cur_ts_preloaded_experiences #  []
            else:
                for tsr_key in cur_ts_preloaded_experiences:
                    # self.tot_ts_preload_experiences[cur_file_ts][tsr_key].append(cur_ts_preloaded_experiences[tsr_key])
                    
                    self.tot_ts_preload_experiences[cur_file_ts][tsr_key] = np.concatenate(
                        [ self.tot_ts_preload_experiences[cur_file_ts][tsr_key], cur_ts_preloaded_experiences[tsr_key] ], axis=0
                    )
            # self.tot_ts_preload_experiences[cur_file_ts].append(cur_ts_preloaded_experiences)
            
            # self.tot_ts_preload_experiences[cur_file_ts] = cur_ts_preloaded_experiences

                
            
        total_ts = list(self.tot_ts_preload_experiences.keys())
        total_ts = sorted(total_ts)
        self.total_ts = total_ts 
        self.maxx_preload_experience_ts = max(total_ts) 
        print(f"total_ts: {self.total_ts}")
        
        self.tot_demo_nns = self.tot_ts_preload_experiences[self.maxx_preload_experience_ts]['obses'].shape[0]
        print(f"tot_demo_nns: {self.tot_demo_nns}")
        
        ### allocate bufers for play presaved expeirences ####
        self.preload_experiences = {}
        for tsr_key in self.tot_ts_preload_experiences[self.maxx_preload_experience_ts]:
            cur_ts_shape = [self.horizon_length, self.num_actors] + list(self.tot_ts_preload_experiences[self.maxx_preload_experience_ts][tsr_key].shape[1:]) # 
            cur_ts_shape = tuple(cur_ts_shape)
            cur_tsr_buffer = torch.zeros(cur_ts_shape, dtype=torch.float32).to(self.device) 
            self.preload_experiences[tsr_key] = cur_tsr_buffer
        ### allocate bufers for play presaved expeirences ####

    
    
    def reload_inplace_saved_experiences_multitraj_via_ts(self,):
        # assert len(self.traj_idx_to_experience_sv_folder) > 0 and os.path.exists(self.traj_idx_to_experience_sv_folder)
        # assert self.save_experiences_via_ts
        # traj_idx_to_experience_sv_folder =np.load(self.traj_idx_to_experience_sv_folder, allow_pickle=True).item() 
        # self.traj_idx_to_experience_sv_folder = traj_idx_to_experience_sv_folder
        # self.tot_traj_idxes = list(self.traj_idx_to_experience_sv_folder.keys())
        # self.tot_traj_idxes = sorted(self.tot_traj_idxes)
        
        # self.cur_ordered_traj_idx = 0
        
        self.cur_ordered_traj_idx = self.cur_ordered_traj_idx + 1
        self.cur_ordered_traj_idx = self.cur_ordered_traj_idx % len(self.tot_traj_idxes)
        
        cur_traj_idx = self.tot_traj_idxes[self.cur_ordered_traj_idx]
        
        # self.tot_ts_preload_experiences = {}
        
        
        # for cur_traj_idx in traj_idx_to_experience_sv_folder:
            
        cur_traj_experiences_sv_fn = self.traj_idx_to_experience_sv_folder[cur_traj_idx]
        assert os.path.isdir(cur_traj_experiences_sv_fn)
        experience_st_tag = "experience_buffer_sv_"
        tot_fns = os.listdir(cur_traj_experiences_sv_fn)
        tot_fns = [ fn for fn in tot_fns if fn[: len(experience_st_tag)] == experience_st_tag and fn.endswith('.npy')] 
        
        print(f"[INFO] Reloading presaved_experiences from folder: {cur_traj_experiences_sv_fn}")
        
        for i_fn, cur_fn in enumerate(tot_fns):
            
            cur_file_ts = cur_fn.split("/")[-1].split(".npy")[0].split("_ts_")[-1] 
            cur_file_ts = int(cur_file_ts)
            if cur_file_ts > self.load_experiences_maxx_ts:
                continue
            # print(f"[{cur_traj_idx}/{len(self.traj_idx_to_experience_sv_folder)}] [{i_fn}/{len(tot_fns)}] loading {cur_fn}")
            # then we should load cur_fn #
            cur_full_fn = os.path.join(cur_traj_experiences_sv_fn, cur_fn)
            cur_ts_preloaded_experiences = np.load(cur_full_fn, allow_pickle=True).item() # 
            
            for tsr_key in cur_ts_preloaded_experiences:
                self.tot_ts_preload_experiences[cur_file_ts][tsr_key][:] = cur_ts_preloaded_experiences[tsr_key][:]
            
            # if cur_file_ts not in self.tot_ts_preload_experiences:
            #     self.tot_ts_preload_experiences[cur_file_ts] = {}
            #     for tsr_key in cur_ts_preloaded_experiences:
            #         # self.tot_ts_preload_experiences[cur_file_ts][tsr_key] = [cur_ts_preloaded_experiences[tsr_key]]
            #         self.tot_ts_preload_experiences[cur_file_ts][tsr_key] = cur_ts_preloaded_experiences[tsr_key]
            #     # self.tot_ts_preload_experiences[cur_file_ts] = cur_ts_preloaded_experiences #  []
            # else:
            #     for tsr_key in cur_ts_preloaded_experiences:
            #         # self.tot_ts_preload_experiences[cur_file_ts][tsr_key].append(cur_ts_preloaded_experiences[tsr_key])
                    
            #         self.tot_ts_preload_experiences[cur_file_ts][tsr_key] = np.concatenate(
            #             [ self.tot_ts_preload_experiences[cur_file_ts][tsr_key], cur_ts_preloaded_experiences[tsr_key] ], axis=0
            #         )
            # self.tot_ts_preload_experiences[cur_file_ts].append(cur_ts_preloaded_experiences)
            
            # self.tot_ts_preload_experiences[cur_file_ts] = cur_ts_preloaded_experiences

                
            
        # total_ts = list(self.tot_ts_preload_experiences.keys())
        # total_ts = sorted(total_ts)
        # self.total_ts = total_ts 
        # self.maxx_preload_experience_ts = max(total_ts) 
        # print(f"total_ts: {self.total_ts}")
        
        # self.tot_demo_nns = self.tot_ts_preload_experiences[self.maxx_preload_experience_ts]['obses'].shape[0]
        # print(f"tot_demo_nns: {self.tot_demo_nns}")
        
        # ### allocate bufers for play presaved expeirences ####
        # self.preload_experiences = {}
        # for tsr_key in self.tot_ts_preload_experiences[self.maxx_preload_experience_ts]:
        #     cur_ts_shape = [self.horizon_length, self.num_actors] + list(self.tot_ts_preload_experiences[self.maxx_preload_experience_ts][tsr_key].shape[1:]) # 
        #     cur_ts_shape = tuple(cur_ts_shape)
        #     cur_tsr_buffer = torch.zeros(cur_ts_shape, dtype=torch.float32).to(self.device) 
        #     self.preload_experiences[tsr_key] = cur_tsr_buffer
        # ### allocate bufers for play presaved expeirences ####

    
    ## TODO: add a function to preload the saved experiences ##
    ## TODO: when we have loaded the saved experiences, add a function to sample from the pre-saved experiences ##
    def preload_saved_experiences(self):
        
        if len(self.traj_idx_to_experience_sv_folder) > 0 and os.path.exists(self.traj_idx_to_experience_sv_folder):
            self.preload_saved_experiences_multitraj_via_ts()
            return
        
        
        if os.path.isdir(self.preload_experiences_path):
            experience_st_tag = "experience_buffer_sv_"
            tot_fns = os.listdir(self.preload_experiences_path)
            tot_fns = [ fn for fn in tot_fns if fn[: len(experience_st_tag)] == experience_st_tag and fn.endswith('.npy')] 
            
            
            if self.save_experiences_via_ts:
                
                self.tot_ts_preload_experiences = {}
                # get preload experiences via ts #
                # _ts_ # # _ts_ #
                for i_fn, cur_fn in enumerate(tot_fns):
                    cur_file_ts = cur_fn.split("/")[-1].split(".npy")[0].split("_ts_")[-1] 
                    cur_file_ts = int(cur_file_ts)
                    if cur_file_ts > self.load_experiences_maxx_ts:
                        continue
                    print(f"[{i_fn}/{len(tot_fns)}] loading {cur_fn}")
                    # then we should load cur_fn #
                    cur_full_fn = os.path.join(self.preload_experiences_path, cur_fn)
                    cur_ts_preloaded_experiences = np.load(cur_full_fn, allow_pickle=True).item() # 
                    # cur ts preloaded experiences #
                    self.tot_ts_preload_experiences[cur_file_ts] = cur_ts_preloaded_experiences
                total_ts = list(self.tot_ts_preload_experiences.keys())
                total_ts = sorted(total_ts) # 
                self.total_ts = total_ts
                self.maxx_preload_experience_ts = max(total_ts) 
                print(f"total_ts: {self.total_ts}")
                
                self.tot_demo_nns = self.tot_ts_preload_experiences[self.maxx_preload_experience_ts]['obses'].shape[0]
                print(f"tot_demo_nns: {self.tot_demo_nns}")
            
                
                ### allocate bufers for play presaved expeirences ####
                self.preload_experiences = {}
                for tsr_key in self.tot_ts_preload_experiences[self.maxx_preload_experience_ts]:
                    # 
                    cur_ts_shape = [self.horizon_length, self.num_actors] + list(self.tot_ts_preload_experiences[self.maxx_preload_experience_ts][tsr_key].shape[1:]) # 
                    cur_ts_shape = tuple(cur_ts_shape)
                    cur_tsr_buffer = torch.zeros(cur_ts_shape, dtype=torch.float32).to(self.device) 
                    self.preload_experiences[tsr_key] = cur_tsr_buffer
                ### allocate bufers for play presaved expeirences ####
            
            else:
                self.preload_experiences = {}
                
                for cur_fn in tot_fns: #
                    cur_full_fn = os.path.join(self.preload_experiences_path, cur_fn)
                    print(f"Loading from {cur_full_fn}")
                    preload_experiences = np.load(cur_full_fn, allow_pickle=True).item()
                    print(f"preload_experiences: {preload_experiences.keys()}")
                    for key in preload_experiences:
                        maxx_nn = 2000
                        if key not in self.preload_experiences:
                            self.preload_experiences[key] = [ torch.from_numpy(preload_experiences[key][:, :maxx_nn]).float().to(self.device) ]
                        else:
                            self.preload_experiences[key].append( torch.from_numpy(preload_experiences[key][:, :maxx_nn]).float().to(self.device) )  #
                for key in self.preload_experiences:
                    print(f"Processing {key}") # nn_ts x nn_envs x n_feature_dim #
                    # self.preload_experiences[key] = np.concatenate(self.preload_experiences[key], axis=1)
                    # print(f"After concatenating, start moving to torch and the gpu")
                    # self.preload_experiences[key] = torch.from_numpy(self.preload_experiences[key]).float().to(self.device)
                    
                    self.preload_experiences[key] = torch.cat(
                        self.preload_experiences[key], dim=1
                    )
                for key in self.preload_experiences:
                    if self.preload_experiences[key].size(1) < self.num_actors:
                        print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                        self.preload_experiences[key] = torch.cat(
                            [
                                self.preload_experiences[key] for _ in range(self.num_actors // self.preload_experiences[key].size(1))
                            ], dim=1
                        )
                        if self.preload_experiences[key].size(1) < self.num_actors:
                            # print(f"self.preload_experiences[key]: {self.preload_experiences[key].size()}")
                            self.preload_experiences[key] = torch.cat(
                                [
                                    self.preload_experiences[key], self.preload_experiences[key][:, : self.num_actors - self.preload_experiences[key].size(1)]
                                ], dim=1
                            )
        else:
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
    
    
    def prepare_tsr_buffer(self, ):
        st_n = self.presaved_experience_st_idx
        for cur_ts_n in range(st_n, st_n + self.horizon_length):
            cur_real_ts_n = self.total_ts[cur_ts_n % len(self.total_ts)]
            # cur_real_ts_n #
            cur_ts_preload_experience = self.tot_ts_preload_experiences[cur_real_ts_n]
            for tsr_key in cur_ts_preload_experience:
                # 
                if cur_ts_preload_experience[tsr_key].shape[0] < self.preload_experiences[tsr_key].size(1):
                    cur_shp_st = 0
                    while cur_shp_st < self.preload_experiences[tsr_key].size(1):
                        cur_shp_ed = min(cur_shp_st + cur_ts_preload_experience[tsr_key].shape[0], self.preload_experiences[tsr_key].size(1)) # #
                        self.preload_experiences[tsr_key][cur_ts_n - st_n, cur_shp_st: cur_shp_ed] = torch.from_numpy(cur_ts_preload_experience[tsr_key][: cur_shp_ed  - cur_shp_st]).float().to(self.device)
                        cur_shp_st = cur_shp_ed
                else:
                    self.preload_experiences[tsr_key][cur_ts_n - st_n] = torch.from_numpy(cur_ts_preload_experience[tsr_key][self.chunk_start_frame: self.chunk_start_frame + self.num_actors]).float().to(self.device)
                


    
    def play_presaved_experiences(self ):
        update_list = self.update_list
        
        ### play presaved experiences ###
        # obses: (300, 8000, 469)
        # rewards: (300, 8000, 1)
        # values: (300, 8000, 1)
        # neglogpacs: (300, 8000)
        # dones: (300, 8000)
        # actions: (300, 8000, 22)
        # mus: (300, 8000, 22)
        # sigmas: (300, 8000, 22)
        
        
        # chunk start frame #
        # chunk_start_frame, play_presaved_experience_idx, play_presaved_experience_changing_freq
        # 
        
        if self.save_experiences_via_ts:
            
            chunk_start_frame = 0
            chunk_end_frame = self.num_actors
        else:
            chunk_start_frame = self.chunk_start_frame
            chunk_end_frame = chunk_start_frame + self.num_actors
        
        
        # training the model using a large amount of data? #
        
        if self.save_experiences_via_ts:
            self.prepare_tsr_buffer()
        
        for i_n in range(self.horizon_length):
            
            if self.save_experiences_via_ts:
                n = i_n
            else:
                n = self.presaved_experience_st_idx # preload experiences st idx #
            
            
            
            # actions 
            actions = self.preload_experiences['actions'][n][chunk_start_frame: chunk_end_frame] ## n-th actions ##
            obses = self.preload_experiences['obses'][n][chunk_start_frame: chunk_end_frame] ## n-th observations ##
            # neglogpacs = self.preload_experiences['neglogpacs'][n] # chunk start frame: chunk end frame
            # values = self.preload_experiences['values'][n]
            dones = self.preload_experiences['dones'][n][chunk_start_frame: chunk_end_frame]
            rewards = self.preload_experiences['rewards'][n][chunk_start_frame: chunk_end_frame]
            # if 'obs' in res_dict: # play presaved experiments #
            #     res_dict['obs'] = res_dict['obs'][..., : self.obs_shape[0]]
            
            self.experience_buffer.update_data('obses', i_n, obses)
            self.experience_buffer.update_data('dones', i_n, dones)
            for k in update_list:
                # print(f"Updating {k}, val size: {self.preload_experiences[k][n][chunk_start_frame: chunk_end_frame].size()}")
                self.experience_buffer.update_data(k, i_n, self.preload_experiences[k][n][chunk_start_frame: chunk_end_frame]) # the preloaded experiences #
            if self.has_central_value:
                self.experience_buffer.update_data('states', i_n, self.preload_experiences['states'][n][chunk_start_frame: chunk_end_frame])
            
            if self.bc_style_training and self.action_chunking:
                ## TODO: make sure that the gt_act is with the same size of nn_dof_pos * action_chunks ##
                self.experience_buffer.update_data('gt_act', i_n, actions)
            
            if 'delta_targets' in self.preload_experiences:
                self.experience_buffer.update_data('delta_targets', i_n, self.preload_experiences['delta_targets'][n][chunk_start_frame: chunk_end_frame]) # 
            
            # step_time_start = time.perf_counter() # g
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
            # self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            # TODO: try to change the actions to mus? #
            # step_time_end = time.perf_counter()

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

            # # if not self.epoch_num % 10 == 0: # not dones #
            # self.game_rewards.update(self.current_rewards[env_done_indices])
            # self.game_shaped_rewards.update(self.current_shaped_rewards[env_done_indices])
            # self.game_lengths.update(self.current_lengths[env_done_indices])
            # self.algo_observer.process_infos(infos, env_done_indices)

            not_dones = 1.0 - self.dones.float()

            # self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
            # self.current_shaped_rewards = self.current_shaped_rewards * not_dones.unsqueeze(1)
            # self.current_lengths = self.current_lengths * not_dones
            
            self.presaved_experience_st_idx += 1
            # maxx_steps = 149
            # maxx_steps = 290
            maxx_steps = 270
            if self.save_experiences_via_ts:
                maxx_steps = min(maxx_steps, self.maxx_preload_experience_ts + 1)
            else:
                maxx_steps = min(maxx_steps, self.preload_experiences['obses'].size(0))
            self.presaved_experience_st_idx = self.presaved_experience_st_idx % maxx_steps
        
        # presaved expereience #
        # last_values = self.get_values(self.obs)
        
        # if self.training_mode == 'offline_supervised':
        #     #### version 2 for creating the expert demonstration data ####
        #     last_values = torch.ones_like(last_values)
        #     #### version 2 for creating the expert demonstration data ####
            
        # # if self.epoch_num % 10 == 0:
        # #     #### version 2 for creating the expert demonstration data ####
        # #     last_values = torch.ones_like(last_values) # demonstration data #
        # #     #### version 2 for creating the expert demonstration data ####
        
        if self.save_experiences_via_ts:
            if self.presaved_experience_st_idx == 0:
                last_values = self.preload_experiences['values'][ - 1][chunk_start_frame: chunk_end_frame]
            else:
                last_values = self.preload_experiences['values'][-1][chunk_start_frame: chunk_end_frame]
        else:
            if self.presaved_experience_st_idx == 0:
                last_values = self.preload_experiences['values'][maxx_steps - 1][chunk_start_frame: chunk_end_frame]
            else:
                last_values = self.preload_experiences['values'][self.presaved_experience_st_idx][chunk_start_frame: chunk_end_frame]
        
        
        
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
        
        
        # chunk_start_frame, play_presaved_experience_idx, play_presaved_experience_changing_freq
        self.play_presaved_experience_idx = self.play_presaved_experience_idx + 1
        
        if self.play_presaved_experience_idx % self.play_presaved_experience_changing_freq == 0:
            # modify 
            if self.preload_all_saved_exp_buffers:
                # self.exp_loading_st_idx = self.exp_loading_st_idx + 1
                # self.exp_loading_st_idx = self.exp_loading_st_idx % len(self.tot_saved_experience_fns)
                # for key in self.preload_experiences:
                #     del self.preload_experiences[key]
                
                
                # for key in self.preload_experiences:
                #     self.preload_experiences[key] = self.preload_experiences[key].detach().cpu().numpy()
                
                # del self.preload_experiences
                # torch.cuda.empty_cache()
                # torch.cuda.empty_cache()
                # torch.cuda.empty_cache()
                # torch.cuda.empty_cache()
                
                # self.reload_experiences()
                
                self.reload_experiences_inplace()
                self.chunk_start_frame = 0
                pass
            else:
                
                if self.save_experiences_via_ts:
                    
                    self.reload_inplace_saved_experiences_multitraj_via_ts()
                    self.chunk_start_frame = 0
                    
                    # if self.chunk_start_frame + self.num_actors >= self.tot_demo_nns:
                    #     self.chunk_start_frame = 0
                    # else:
                    #     self.chunk_start_frame = min(self.tot_demo_nns - self.num_actors, self.chunk_start_frame + self.num_actors)
                else:
                    if self.chunk_start_frame + self.num_actors >= self.preload_experiences['obses'].size(1):
                        self.chunk_start_frame = 0
                    else:
                        self.chunk_start_frame = min(self.preload_experiences['obses'].size(1) - self.num_actors, self.chunk_start_frame + self.num_actors)
            # self.chunk_start_frame = self.chunk_start_frame + 1
            # self.play_presaved_experience_idx = 0
        
        
        return batch_dict

    
    
    def play_steps(self):
        
        if self.preload_experiences_tf and self.dagger_style_training or (self.preload_experiences_tf and self.bc_style_training) or (self.preload_experiences_tf and self.distill_delta_targets) or (self.preload_experiences_tf and self.demonstration_tuning_model and self.demonstration_tuning_model_freq == 1):
            batch_dict = self.play_presaved_experiences()
            return batch_dict 
        
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length):
            # print(f"self.obs: {self.obs['obs'].size()}, self.obs_shape: {self.obs_shape}") # 
            
            if self.masked_mimic_training:
                mimic_teacher_obs = self.obs['mimic_teacher_obs'][..., : ] # get the last obs features as teacher features here #
                # self.obs['obs'] = self.obs['obs'][..., : - self.obs_shape[0]]
                self.experience_buffer.update_data('mimic_teacher_obs', n, mimic_teacher_obs)
            
            gt_act_val = self.obs['obs'][..., self.obs_shape[0]: ]
            # if we do not use the gt act val? ---- where the value would be added ? # #
            # print(f"self.obs_shape[0]: {self.obs_shape[0]}, gt_act_val: {gt_act_val.size()}") #
            # the last obs_shape[0] dim is the gt_act_val # #
            
            self.experience_buffer.update_data('gt_act', n, gt_act_val)
            
            
            if self.train_student_model:
                full_obs = self.obs['full_obs']
                self.experience_buffer.update_data('full_obs', n, full_obs)
            
            
            
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
            if self.dagger_style_training:
                # self.obs, rewards, self.dones, infos = self.env_step(res_dict['mus'])
                self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
            else:
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
    
    
    
    def actor_loss_supervised(self, pred_actions, gt_actions):
        
        
        
        # if len(self.grab_obj_type_to_opt_res_fn) > 0:
            
        gt_succ_flag = gt_actions[..., -1]
        avg_succ_flag = torch.sum(gt_succ_flag) / gt_succ_flag.size(0)
        # print(f"avg_succ_flag: {avg_succ_flag}")
        # gt_actions = gt_actions[..., :-1]
        gt_actions = gt_actions[..., :self.nn_act_dims]
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


    def trancate_gradients_and_step_transformer(self): # 
        if self.multi_gpu:
            # batch allreduce ops: see https://github.com/entity-neural-network/incubator/pull/220
            all_grads_list = []
            for param in self.transformer_encoder_decoder.parameters():
                if param.grad is not None:
                    all_grads_list.append(param.grad.view(-1))

            all_grads = torch.cat(all_grads_list)
            dist.all_reduce(all_grads, op=dist.ReduceOp.SUM)
            offset = 0
            for param in self.transformer_encoder_decoder.parameters():
                if param.grad is not None:
                    param.grad.data.copy_(
                        all_grads[offset : offset + param.numel()].view_as(param.grad.data) / self.world_size
                    )
                    offset += param.numel()

        if self.truncate_grads:
            # if self.train_controller:
            #     self.scaler.unscale_(self.optimizer)
            #     nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            if self.use_transformer_model:
                self.transformer_scaler.unscale_(self.transformer_optimizer)
                nn.utils.clip_grad_norm_(self.transformer_encoder_decoder.parameters(), self.grad_norm)

        # if self.train_controller:
        # print(f"self.scaler.step(self.optimizer)")
        self.transformer_scaler.step(self.transformer_optimizer)
        self.transformer_scaler.update()
        
        # if self.train_forecasting_model:
        #     self.forecasting_scaler.step(self.forecasting_optimizer)
        #     self.forecasting_scaler.update()
    
    def calc_gradients(self, input_dict):
        """Compute gradients needed to step the networks of the algorithm.

        Core algo logic is defined here

        Args:
            input_dict (:obj:`dict`): Algo inputs as a dict.

        """
        
        value_preds_batch = input_dict['old_values']
        old_action_log_probs_batch = input_dict['old_logp_actions']
        advantage = input_dict['advantages']
        old_mu_batch = input_dict['mu']
        old_sigma_batch = input_dict['sigma']
        return_batch = input_dict['returns']
        actions_batch = input_dict['actions']
        obs_batch = input_dict['obs']
        obs_batch = self._preproc_obs(obs_batch)
        
        env_obs_batch = obs_batch[..., : self.obs_shape[0]] 
        
        
        # the current state #
        gt_act_batch = input_dict['gt_act']
        
        if self.masked_mimic_training:
            mimic_teacher_obs_batch = input_dict['mimic_teacher_obs']
        
        # if self.train_student_model:
        #     full_obs = input_dict['full_obs'] # get the full_obs for inferring the ts_teacher_model #
        
        
        if self.use_teacher_model and self.optimizing_with_teacher_net and (not self.dagger_style_training):
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
        
        if self.dagger_style_training:
            assert self.use_teacher_model and self.use_multiple_teacher and self.optimizing_with_teacher_net, "Dagger style training requires the use of teacher model and optimizing with teacher net"
            
            def get_temp_ts_teacher_action_values(obs, ts_model, ts_obs_dim, is_train=False):
                processed_obs = self._preproc_obs(obs['obs'])
                ts_model.eval()
                input_dict = {
                    'is_train': is_train,
                    'prev_actions': obs['actions'], 
                    'obs' : processed_obs[..., : ts_obs_dim],
                    'rnn_states' : self.rnn_states
                }

                # get the actions values #
                with torch.no_grad(): 
                    res_dict = ts_model(input_dict)
                    if self.has_central_value:
                        states = obs['states']
                        input_dict = {
                            'is_train': False,
                            'states' : states,
                        }
                        value = self.get_central_value(input_dict)
                        res_dict['values'] = value
                return res_dict
            
            tot_teacher_mus_batch = []
            tot_teacher_sigmas_batch = []
            for i_teacher in range(0, self.nn_teacher):
                cur_teaacher_model = self.teacher_index_to_models[i_teacher]
                teacher_model_obs_dim = self.teacher_index_to_obs_dim[i_teacher]
                teacher_res_dict = get_temp_ts_teacher_action_values(input_dict, cur_teaacher_model, teacher_model_obs_dim, is_train=True)
                
                cur_teacher_mus = teacher_res_dict['mus'] # nn_bsz x nn_act_dim
                cur_teacher_sigmas = teacher_res_dict['sigmas']
                tot_teacher_mus_batch.append(cur_teacher_mus)
                tot_teacher_sigmas_batch.append(cur_teacher_sigmas)
            tot_teacher_mus_batch = torch.stack(tot_teacher_mus_batch, dim=1) # nn_bsz x nn_teac x ...
            tot_teacher_sigmas_batch = torch.stack(tot_teacher_sigmas_batch, dim=1)
            
            envs_teacher_idxes = gt_act_batch[..., -1:].long() # nn_envs x 1 
            # maxx_teacher_idx = torch.max(envs_teacher_idxes)
            # minn_teacher_idx = torch.min(envs_teacher_idxes)
            # print(f"maxx_teacher_idx: {maxx_teacher_idx}, minn_teacher_idx: {minn_teacher_idx}, tot_teacher_mus_batch: {tot_teacher_mus_batch.size()}, tot_teacher_sigmas_batch: {tot_teacher_sigmas_batch.size()}")
            # print(f"teacher_index_to_weights: {self.teacher_index_to_weights}")
            
            envs_teacher_mus = batched_index_select(tot_teacher_mus_batch, envs_teacher_idxes, dim=1) # nn_bsz x 1 x nn_act_dim
            envs_teacher_mus = envs_teacher_mus.squeeze(1) # nn_bsz x nn_act_dim
            envs_teacher_sigmas = batched_index_select(tot_teacher_sigmas_batch, envs_teacher_idxes, dim=1) 
            envs_teacher_sigmas = envs_teacher_sigmas.squeeze(1) # nn_bsz x nn_act_dim
            
            # #
            # #
            #####################   #####################
            # def get_temp_ts_teacher_action_values(obs, ts_model, ts_obs_dim, is_train=False):
            #     processed_obs = self._preproc_obs(obs['obs'])
            #     ts_model.eval()
            #     input_dict = {
            #         'is_train': is_train,
            #         'prev_actions': obs['actions'], 
            #         'obs' : processed_obs[..., : ts_obs_dim],
            #         'rnn_states' : self.rnn_states
            #     }

            #     # get the actions values #
            #     with torch.no_grad(): 
            #         res_dict = ts_model(input_dict)
            #         if self.has_central_value:
            #             states = obs['states']
            #             input_dict = {
            #                 'is_train': False,
            #                 'states' : states,
            #             }
            #             value = self.get_central_value(input_dict)
            #             res_dict['values'] = value
            #     return res_dict
            
            
            # cur_teacher_idx = 7
            # cur_teaacher_model = self.teacher_index_to_models[cur_teacher_idx]
            # teacher_model_obs_dim = self.teacher_index_to_obs_dim[cur_teacher_idx]
            # # teacher_res_dict = self._get_teacher_action_values(input_dict, cur_teaacher_model, teacher_model_obs_dim=teacher_model_obs_dim)
            # teacher_res_dict = get_temp_ts_teacher_action_values(input_dict, cur_teaacher_model, teacher_model_obs_dim, is_train=True)
            
            # cur_teacher_mus = teacher_res_dict['mus'] # nn_bsz x nn_act_dim
            # cur_teacher_sigmas = teacher_res_dict['sigmas']
            
            # envs_teacher_mus = cur_teacher_mus
            # envs_teacher_sigmas = cur_teacher_sigmas
            
            # print(f"envs_teacher_mus: {envs_teacher_mus.size()}, envs_teacher_sigmas: {envs_teacher_sigmas.size()}")
            #####################   #####################
                
        
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

        if self.use_transformer_model:
            self.transformer_encoder_decoder.train()


        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            if self.use_transformer_model:
                res_dict = self._forward_transformer_model(batch_dict)
            else:
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
            # TODO: what's the role of rnns here? # # 
            # TODO: and also what's the role of rnn_masks? #
            if self.masked_mimic_training:
                supervised_a_loss = self.actor_loss_mimic_teacher(mu, gt_act_batch)
            else:
                supervised_a_loss = self.actor_loss_supervised(mu, gt_act_batch)
            # ####### Supervised loss Version 2 ####### #
            
            
            # ts teacher action values #
            
            if self.train_student_model:
                ts_teacher_res_dict = self.get_ts_teacher_action_values(input_dict, is_train=True) # 
                ts_teacher_mus = ts_teacher_res_dict['mus']
                ts_teacher_sigmas = ts_teacher_res_dict['sigmas']
                diff_mus_with_ts_teacher_mus = torch.norm(mu - ts_teacher_mus, dim=-1, p=2)
                diff_sigmas_with_ts_teacher_sigmas = torch.norm(sigma - ts_teacher_sigmas, dim=-1, p=2)
                diff_mus_with_ts_teacher_mus = diff_mus_with_ts_teacher_mus # .mean()
                diff_sigmas_with_ts_teacher_sigmas = diff_sigmas_with_ts_teacher_sigmas # .mean()
                a_loss = diff_mus_with_ts_teacher_mus + diff_sigmas_with_ts_teacher_sigmas
                
                
            
            
            if self.training_mode == 'offline_supervised':
                # ####### Supervised loss Version 3 ####### #
                policy_distr = torch.distributions.Normal(mu, sigma, validate_args=False) 
                neglog_gt_acts = -policy_distr.log_prob(gt_act_batch).sum(dim=-1)
                supervised_a_loss = neglog_gt_acts
                # ####### Supervised loss Version 3 ####### #
                
            
            if self.dagger_style_training:
                diff_mus_with_ts_teacher_mus = torch.norm(mu - envs_teacher_mus, dim=-1, p=2)
                diff_sigmas_with_ts_teacher_sigmas = torch.norm(sigma - envs_teacher_sigmas, dim=-1, p=2)
                a_loss = diff_mus_with_ts_teacher_mus + diff_sigmas_with_ts_teacher_sigmas
                # a_loss = diff_mus_with_ts_teacher_mus
                # print(f"computing a_loss") 
                # supervised_a_loss = diff_sigmas_with_ts_teacher_sigmas + diff_mus_with_ts_teacher_mus
            
            # use action chunking for bc style training #
            if (self.action_chunking and self.bc_style_training):
                diff_mus_with_gt_acts = torch.norm(mu - gt_act_batch, dim=-1, p=2)
                a_loss = diff_mus_with_gt_acts
                if self.bc_relative_targets:
                    a_loss = a_loss * 10
                    
            if self.distill_via_bc:
                cur_acts = mu + torch.randn_like(sigma) * sigma
                a_loss = torch.norm(cur_acts - gt_act_batch, dim=-1, p=2)
            
            
            if self.demonstration_tuning_model and self.preload_experiences_tf and self.epoch_num % self.demonstration_tuning_model_freq == 0:
                cur_acts = torch.randn_like(sigma) * sigma + mu
                
                if self.distill_delta_targets:
                    gt_delta_targets = input_dict['delta_targets']
                    cur_delta_targets = cur_acts * self.vec_env.env.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.vec_env.env.dt
                    diff_cur_act_w_actions = torch.norm(cur_delta_targets - gt_delta_targets, dim=-1, p=2)
                else:
                    
                    # print(f"demonstrating tuning model")
                    
                    # ###### loss v1 ######
                    # diff_cur_act_w_actions = torch.norm(cur_acts - actions_batch, dim=-1, p=2)
                    # ###### loss v1 ######
                    
                    # diff_mus = ts_teacher_res_dict['mus']
                    # ts_teacher_sigmas = ts_teacher_res_dict['sigmas']
                    # diff_mus = torch.norm(mu - old_mu_batch, dim=-1, p=2)
                    # diff_sigmas = torch.norm(sigma - old_sigma_batch, dim=-1, p=2)
                    
                    # glb_diff_mus = torch.norm(mu[..., :6] - actions_batch[..., :6], dim=-1, p=2)
                    # finger_diff_mus = torch.norm(mu[..., 6:] - actions_batch[..., 6:], dim=-1, p=2)
                    # diff_cur_act_w_actions = glb_diff_mus * 10 + finger_diff_mus
                    
                    
                    diff_mus = torch.norm(mu - actions_batch, dim=-1, p=2)
                    diff_cur_act_w_actions = diff_mus # + diff_sigmas
                    
                a_loss = diff_cur_act_w_actions
                # pass # 
                
                
                
            
            
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
            ##### Version 2 -- with supervised action loss ##### # with ## with a loss coef #
            loss = supervised_a_loss * supervised_a_loss_coef +  a_loss * a_loss_coef + 0.5 * c_loss * self.critic_coef * c_loss_coef - entropy * self.entropy_coef * entropy_coef + b_loss * self.bounds_loss_coef * bounds_loss_coef
            
            
            if self.demonstration_tuning_model and self.preload_experiences_tf and self.epoch_num % self.demonstration_tuning_model_freq == 0:
                loss = a_loss * a_loss_coef
            # else: else we need to add the alos and the aloss coef #
            ### Two configs share the sime formulation of the loss ###
            elif self.dagger_style_training or  self.train_student_model or (self.action_chunking and self.bc_style_training) or self.distill_via_bc:
                if self.dagger_style_training or (  self.action_chunking and self.bc_style_training) or self.distill_via_bc: # 
                    loss = a_loss * a_loss_coef #  - entropy * self.entropy_coef * entropy_coef + b_loss * self.bounds_loss_coef * bounds_loss_coef
                    # loss = supervised_a_loss * supervised_a_loss_coef +  a_loss * a_loss_coef + 0.5 * c_loss * self.critic_coef * c_loss_coef - entropy * self.entropy_coef * entropy_coef + b_loss * self.bounds_loss_coef * bounds_loss_coef
                else:
                    loss = a_loss * a_loss_coef - entropy * self.entropy_coef * entropy_coef + b_loss * self.bounds_loss_coef * bounds_loss_coef
            ##### Version 2 -- with supervised action loss #####
            
            
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            # loss =  a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            
            
            if self.multi_gpu:
                if self.use_transformer_model:
                    self.transformer_optimizer.zero_grad()
                else:
                    self.optimizer.zero_grad()
            else:
                if self.use_transformer_model:
                    self.transformer_optimizer.zero_grad()
                    for param in self.transformer_encoder_decoder.parameters():
                        param.grad = None
                    
                else:
                    for param in self.model.parameters():
                        param.grad = None

        if self.use_transformer_model:
            # self.transformer_scaler.scale(loss).backward()
            # self.trancate_gradients_and_step_transformer()
            
            loss.backward()
            self.transformer_optimizer.step()
            
        else:
            self.scaler.scale(loss).backward()
            # TODO: Refactor this ugliest code of they year
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
        self.loss_np = loss.detach().cpu().numpy()
        return 


    def train_epoch(self):
        super().train_epoch()



        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
            # self.preload experiences tf #
            # activate preload_experiences_tf and also jump to the play presaved experiences in the play steps function #
            # demonstration_tuning_model, demonstration_tuning_model_freq
            if self.demonstration_tuning_model and self.preload_experiences_tf and self.epoch_num % self.demonstration_tuning_model_freq == 0:
                batch_dict = self.play_presaved_experiences()
            elif self.preload_experiences_tf and self.epoch_num % self.play_presaved_freq == 0:
                batch_dict = self.play_presaved_experiences()
            else:
                if self.is_rnn:
                    batch_dict = self.play_steps_rnn()
                else:
                    batch_dict = self.play_steps()


        play_time_end = time.perf_counter()
        update_time_start = time.perf_counter()
        # rnn_masks = batch_dict.get('rnn_masks', None)


        # if self.use_transformer_model:
        #     jself.

        self.set_train()
        self.curr_frames = batch_dict.pop('played_frames')
        self.prepare_dataset(batch_dict)
        self.algo_observer.after_steps()
        if self.has_central_value:
            self.train_central_value()

        a_losses = []
        c_losses = []
        b_losses = []
        # losses = []
        entropies = []
        kls = []

        for mini_ep in range(0, self.mini_epochs_num):
            ep_kls = []
            ep_act_supervised_losses = []
            ep_a_losses = []
            ep_c_losses = []
            ep_entropy_losses = []
            ep_b_losses = []
            ep_losses = []
            for i in range(len(self.dataset)): #
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
                ep_losses.append(self.loss_np)
                
            
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
            ep_losses = np.array(ep_losses)
            avg_ep_loss = np.mean(ep_losses).item()
            
            
            # single_instance_training 
            if not self.single_instance_training:
                print(f"avg_ep_act_supervised_loss: {avg_ep_act_supervised_loss}, avg_ep_a_loss: {avg_ep_a_loss}, avg_ep_c_loss: {avg_ep_c_loss}, avg_ep_entropy_loss: {avg_ep_entropy_loss}, avg_ep_b_loss: {avg_ep_b_loss}, avg_ep_loss: {avg_ep_loss}")
                
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


    def get_weights(self):
        state = self.get_stats_weights()
        state['model'] = self.model.state_dict()
        if self.use_transformer_model:
            state['transformer_encoder_decoder'] = self.transformer_encoder_decoder.state_dict()
        # if self.use_world_model:
        #     state['world_model'] = self.world_model.state_dict()
        print(f"getting weights with keys: {state.keys()}")
        return state

    def train(self):
        self.init_tensors() 
        self.last_mean_rewards = -100500 
        start_time = time.perf_counter() 
        total_time = 0 
        # rep_count = 0 
        self.obs = self.env_reset() 
        # print(f"obs_dict: {self.obs.keys()}") 
        # print(f"second time env reset...") 
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
                
                
                #### save the checkpoint in the corresponding file ####
                if self.preload_experiences_tf:
                    checkpoint_name = self.config['name'] + '_ep_' + str(epoch_num) 

                    if self.save_freq > 0:
                        if epoch_num % self.save_freq == 0:
                            self.save(os.path.join(self.nn_dir, 'last_' + checkpoint_name))

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
                # epoch num >= self.max_epochs #
                
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
        # build additional buffers #
        
        gt_act_dim = self.nn_act_dims + 1

        if (self.bc_style_training and self.action_chunking) or self.distill_via_bc:
            gt_act_dim = self.nn_act_dims 
            
        print(f"[_build_gt_act_buffers] gt_act_dim: {gt_act_dim}")
        
        self.experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (gt_act_dim,),
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
        
        self.experience_buffer.tensor_dict['delta_targets'] = torch.zeros(batch_shape + (self.nn_act_dims,),
                                                                    device=self.ppo_device)
        self.tensor_list += ['delta_targets']
        
        if self.masked_mimic_training:
            self.experience_buffer.tensor_dict['mimic_teacher_obs'] = torch.zeros(batch_shape + (self.obs_shape[0], ),
                                                                    device=self.ppo_device)
            self.tensor_list += ['mimic_teacher_obs']
            
        if self.train_student_model:
            self.experience_buffer.tensor_dict['full_obs'] = torch.zeros(batch_shape + (self.ts_teacher_model_obs_dim, ), device=self.ppo_device)
            self.tensor_list += ['full_obs'] 
            
        if self.distill_action_space:
            # self.experience_buffer.tensor_dict['prev_targets'] = torch.zeros(batch_shape + (self.ts_teacher_model_obs_dim, ), device=self.ppo_device) ### prev targets ### # delta_delta_targets + prev_deltas + kinematic bias --- so we can use them 
            self.experience_buffer.tensor_dict['prev_targets'] = torch.zeros(batch_shape + (self.nn_act_dims, ), device=self.ppo_device)
            self.tensor_list += ['prev_targets'] 
        
        return

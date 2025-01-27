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

from tensorboardX import SummaryWriter

from rl_games.common.experience import ExperienceBuffer
# common agent torch #

class A2CSupervisedAgent(a2c_continuous.A2CAgent):

    def __init__(self, base_name, params):
        super().__init__(base_name, params)
        # the observation is a combination of current observations with the actions #
        # if self.normalize_value:
        #     self.value_mean_std = self.central_value_net.model.value_mean_std if self.has_central_value else self.model.value_mean_std
        # if self._normalize_amp_input:
        #     self._amp_input_mean_std = RunningMeanStd(self._amp_observation_space.shape).to(self.ppo_device)
        # 
        return

    # def init_tensors(self):
    #     batch_size = self.num_agents * self.num_actors
    #     algo_info = {
    #         'num_actors' : self.num_actors,
    #         'horizon_length' : self.horizon_length,
    #         'has_central_value' : self.has_central_value,
    #         'use_action_masks' : self.use_action_masks
    #     }
    #     self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.ppo_device)

    #     val_shape = (self.horizon_length, batch_size, self.value_size)
    #     current_rewards_shape = (batch_size, self.value_size)
    #     self.current_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
    #     self.current_shaped_rewards = torch.zeros(current_rewards_shape, dtype=torch.float32, device=self.ppo_device)
    #     self.current_lengths = torch.zeros(batch_size, dtype=torch.float32, device=self.ppo_device)
    #     self.dones = torch.ones((batch_size,), dtype=torch.uint8, device=self.ppo_device)

    #     if self.is_rnn:
    #         self.rnn_states = self.model.get_default_rnn_state()
    #         self.rnn_states = [s.to(self.ppo_device) for s in self.rnn_states]

    #         total_agents = self.num_agents * self.num_actors
    #         num_seqs = self.horizon_length // self.seq_length
    #         assert((self.horizon_length * total_agents // self.num_minibatches) % self.seq_length == 0)
    #         self.mb_rnn_states = [torch.zeros((num_seqs, s.size()[0], total_agents, s.size()[2]), dtype = torch.float32, device=self.ppo_device) for s in self.rnn_states]


    def init_tensors(self):
        super().init_tensors()
        self._build_gt_act_buffers()
        return
    
    # def set_eval(self):
    #     super().set_eval()
    #     if self._normalize_amp_input:
    #         self._amp_input_mean_std.eval()
    #     return

    # def set_train(self):
    #     super().set_train()
    #     if self._normalize_amp_input:
    #         self._amp_input_mean_std.train()
        # return

    # def get_stats_weights(self):
    #     state = super().get_stats_weights()
    #     if self._normalize_amp_input:
    #         state['amp_input_mean_std'] = self._amp_input_mean_std.state_dict()
    #     return state

    # def set_stats_weights(self, weights):
    #     super().set_stats_weights(weights)
    #     if self._normalize_amp_input:
    #         self._amp_input_mean_std.load_state_dict(weights['amp_input_mean_std'])
    #     return

    # def play_steps(self):
    #     self.set_eval()

    #     epinfos = []
    #     update_list = self.update_list

    #     for n in range(self.horizon_length):
    #         self.obs, done_env_ids = self._env_reset_done()
    #         self.experience_buffer.update_data('obses', n, self.obs['obs'])

    #         if self.use_action_masks:
    #             masks = self.vec_env.get_action_masks()
    #             res_dict = self.get_masked_action_values(self.obs, masks)
    #         else:
    #             res_dict = self.get_action_values(self.obs)

    #         for k in update_list:
    #             self.experience_buffer.update_data(k, n, res_dict[k]) 

    #         if self.has_central_value:
    #             self.experience_buffer.update_data('states', n, self.obs['states'])

    #         self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
    #         shaped_rewards = self.rewards_shaper(rewards)
    #         self.experience_buffer.update_data('rewards', n, shaped_rewards)
    #         self.experience_buffer.update_data('next_obses', n, self.obs['obs'])
    #         self.experience_buffer.update_data('dones', n, self.dones)
    #         self.experience_buffer.update_data('amp_obs', n, infos['amp_obs'])

    #         terminated = infos['terminate'].float()
    #         terminated = terminated.unsqueeze(-1)
    #         next_vals = self._eval_critic(self.obs)
    #         next_vals *= (1.0 - terminated)
    #         self.experience_buffer.update_data('next_values', n, next_vals)

    #         self.current_rewards += rewards
    #         self.current_lengths += 1
    #         all_done_indices = self.dones.nonzero(as_tuple=False)
    #         done_indices = all_done_indices[::self.num_agents]
  
    #         self.game_rewards.update(self.current_rewards[done_indices])
    #         self.game_lengths.update(self.current_lengths[done_indices])
    #         self.algo_observer.process_infos(infos, done_indices)

    #         not_dones = 1.0 - self.dones.float()

    #         self.current_rewards = self.current_rewards * not_dones.unsqueeze(1)
    #         self.current_lengths = self.current_lengths * not_dones
        
    #         if (self.vec_env.env.viewer and (n == (self.horizon_length - 1))):
    #             # self._amp_debug(infos)
    #             pass

    #     mb_fdones = self.experience_buffer.tensor_dict['dones'].float()
    #     mb_values = self.experience_buffer.tensor_dict['values']
    #     mb_next_values = self.experience_buffer.tensor_dict['next_values']

    #     mb_rewards = self.experience_buffer.tensor_dict['rewards']
    #     mb_amp_obs = self.experience_buffer.tensor_dict['amp_obs']
    #     amp_rewards = self._calc_amp_rewards(mb_amp_obs)
    #     mb_rewards = self._combine_rewards(mb_rewards, amp_rewards)

    #     mb_advs = self.discount_values(mb_fdones, mb_values, mb_rewards, mb_next_values)
    #     mb_returns = mb_advs + mb_values

    #     batch_dict = self.experience_buffer.get_transformed_list(a2c_common.swap_and_flatten01, self.tensor_list)
    #     batch_dict['returns'] = a2c_common.swap_and_flatten01(mb_returns)
    #     batch_dict['played_frames'] = self.batch_size

    #     for k, v in amp_rewards.items():
    #         batch_dict[k] = a2c_common.swap_and_flatten01(v)

    #     return batch_dict

    def prepare_dataset(self, batch_dict):
        super().prepare_dataset(batch_dict)
        self.dataset.values_dict['gt_act'] = batch_dict['gt_act']
        # self.dataset.values_dict['amp_obs'] = batch_dict['amp_obs']
        # self.dataset.values_dict['amp_obs_demo'] = batch_dict['amp_obs_demo']
        # self.dataset.values_dict['amp_obs_replay'] = batch_dict['amp_obs_replay']
        # # return # 

    # def train_epoch(self):
    #     # 
    #     play_time_start = time.time()
    #     with torch.no_grad():
    #         if self.is_rnn:
    #             batch_dict = self.play_steps_rnn()
    #         else:
    #             batch_dict = self.play_steps() 

    #     play_time_end = time.time()
    #     update_time_start = time.time()
    #     rnn_masks = batch_dict.get('rnn_masks', None)
        
    #     self._update_amp_demos()
    #     num_obs_samples = batch_dict['amp_obs'].shape[0]
    #     amp_obs_demo = self._amp_obs_demo_buffer.sample(num_obs_samples)['amp_obs']
    #     batch_dict['amp_obs_demo'] = amp_obs_demo

    #     if (self._amp_replay_buffer.get_total_count() == 0):
    #         batch_dict['amp_obs_replay'] = batch_dict['amp_obs']
    #     else:
    #         batch_dict['amp_obs_replay'] = self._amp_replay_buffer.sample(num_obs_samples)['amp_obs']

    #     self.set_train()

    #     self.curr_frames = batch_dict.pop('played_frames')
    #     self.prepare_dataset(batch_dict)
    #     self.algo_observer.after_steps()

    #     if self.has_central_value:
    #         self.train_central_value()

    #     train_info = None

    #     if self.is_rnn:
    #         frames_mask_ratio = rnn_masks.sum().item() / (rnn_masks.nelement())
    #         print(frames_mask_ratio)

    #     for _ in range(0, self.mini_epochs_num):
    #         ep_kls = []
    #         for i in range(len(self.dataset)):
    #             curr_train_info = self.train_actor_critic(self.dataset[i])
                
    #             if self.schedule_type == 'legacy':
    #                 self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, curr_train_info['kl'].item())
    #                 self.update_lr(self.last_lr)

    #             if (train_info is None):
    #                 train_info = dict()
    #                 for k, v in curr_train_info.items():
    #                     train_info[k] = [v]
    #             else:
    #                 for k, v in curr_train_info.items():
    #                     train_info[k].append(v)
            
    #         av_kls = torch_ext.mean_list(train_info['kl'])

    #         if self.schedule_type == 'standard':
    #             self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
    #             self.update_lr(self.last_lr)

    #     if self.schedule_type == 'standard_epoch':
    #         self.last_lr, self.entropy_coef = self.scheduler.update(self.last_lr, self.entropy_coef, self.epoch_num, 0, av_kls.item())
    #         self.update_lr(self.last_lr)

    #     update_time_end = time.time()
    #     play_time = play_time_end - play_time_start
    #     update_time = update_time_end - update_time_start
    #     total_time = update_time_end - play_time_start

    #     self._store_replay_amp_obs(batch_dict['amp_obs'])

    #     train_info['play_time'] = play_time
    #     train_info['update_time'] = update_time
    #     train_info['total_time'] = total_time
    #     self._record_train_batch_info(batch_dict, train_info)

    #     return train_info

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
            # # NOTE: since it is the deterministic network , wejset actions to mus #
            # res_dict['actions'] = res_dict['mus'] # if we only train the actor -- then it is the problem... #
            #
            if self.has_central_value: # centoal values? #
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
            else: # #  # deterministic # # large action losses --- # why action losses get so large? # 
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

    def play_steps(self): # 
        update_list = self.update_list

        step_time = 0.0

        for n in range(self.horizon_length): # 
            # print(f"self.obs: {self.obs['obs'].size()}, self.obs_shape: {self.obs_shape}")
            gt_act_val = self.obs['obs'][..., self.obs_shape[0]: ]
            self.experience_buffer.update_data('gt_act', n, gt_act_val)
            
            # if 'obs' in self.obs:
            #     self.obs['obs'] = self.obs['obs'][..., : self.obs_shape[0]]
            if self.use_action_masks:
                masks = self.vec_env.get_action_masks()
                res_dict = self.get_masked_action_values(self.obs, masks)
            else:
                res_dict = self.get_action_values(self.obs)
            
            # if 'obs' in res_dict: # modify obs if wish to add to add to the replay buffer #
            #     res_dict['obs'] = res_dict['obs'][..., : self.obs_shape[0]] ## obs shape #
            
            self.experience_buffer.update_data('obses', n, self.obs['obs'][..., : self.obs_shape[0]])
            self.experience_buffer.update_data('dones', n, self.dones)

            for k in update_list: # add to therepaly buffer --- it is not the supervised training at all? ## 
                ## does such bias really useful for training the agent #
                ## 
                self.experience_buffer.update_data(k, n, res_dict[k]) 
            if self.has_central_value:
                self.experience_buffer.update_data('states', n, self.obs['states'])

            step_time_start = time.perf_counter()
            self.obs, rewards, self.dones, infos = self.env_step(res_dict['actions'])
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

    def actor_loss_supervised(self, pred_actions, gt_actions):
        
        loss = torch.sum(
            torch.nn.functional.mse_loss(pred_actions, gt_actions, reduction='none'), dim=-1
        )
        # print(f"pred_actions: {pred_actions.size()}, gt_actions: {gt_actions.size()}, loss: {loss.size()}")
        return loss

    def calc_gradients(self, input_dict): # calculate gradients # # 
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
        
        gt_act_batch = input_dict['gt_act'] # obs_batch[..., self.obs_shape[0]: ]

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

            ### previous rl actor loss func ### # loss funct ## ## get the observations nad smaples from an expert actor? ## 
            a_loss = self.actor_loss_func(old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip)

            
            # TODO: what's the role of rnns here? #
            # TODO: and also what's the role of rnn_masks? #
            supervised_a_loss = self.actor_loss_supervised(mu, gt_act_batch)
            
            if self.has_value_loss:
                c_loss = common_losses.critic_loss(self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value)
            else:
                c_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == 'regularisation':
                b_loss = self.reg_loss(mu)
            elif self.bound_loss_type == 'bound': # value pred batach # 
                b_loss = self.bound_loss(mu)
            else:
                b_loss = torch.zeros(1, device=self.ppo_device)
            losses, sum_mask = torch_ext.apply_masks([supervised_a_loss.unsqueeze(1), a_loss.unsqueeze(1), c_loss , entropy.unsqueeze(1), b_loss.unsqueeze(1)], rnn_masks)
            supervised_a_loss, a_loss, c_loss, entropy, b_loss = losses[0], losses[1], losses[2], losses[3], losses[4]

            # ##### Version 1 --- only use the action supervised loss for training ##### #
            # a_loss = supervised_a_loss 
            # c_loss = a_loss
            # entropy = torch.zeros_like(a_loss)
            # b_loss = torch.zeros_like(a_loss)
            # loss = a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            # ##### Version 1 ##### #
            
            supervised_a_loss_coef = 0.001
            ##### Version 2 -- with supervised action loss #####
            loss = supervised_a_loss * supervised_a_loss_coef +  a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef
            ##### Version 2 -- with supervised action loss #####
            
            # # ##### Version 3 -- without supervised action loss, for testing the correctness of this agent #####
            # loss =  a_loss + 0.5 * c_loss * self.critic_coef - entropy * self.entropy_coef + b_loss * self.bounds_loss_coef #
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
        super().train_epoch() # vecenv.settrain #

        self.set_eval()
        play_time_start = time.perf_counter()
        with torch.no_grad():
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

    # def _load_config_params(self, config):
    #     super()._load_config_params(config)
        
    #     self._task_reward_w = config['task_reward_w']
    #     self._disc_reward_w = config['disc_reward_w']

    #     self._amp_observation_space = self.env_info['amp_observation_space']
    #     self._amp_batch_size = int(config['amp_batch_size'])
    #     self._amp_minibatch_size = int(config['amp_minibatch_size'])
    #     assert(self._amp_minibatch_size <= self.minibatch_size)

    #     self._disc_coef = config['disc_coef']
    #     self._disc_logit_reg = config['disc_logit_reg']
    #     self._disc_grad_penalty = config['disc_grad_penalty']
    #     self._disc_weight_decay = config['disc_weight_decay']
    #     self._disc_reward_scale = config['disc_reward_scale']
    #     self._normalize_amp_input = config.get('normalize_amp_input', True)
    #     return

    # def _build_net_config(self):
    #     config = super()._build_net_config()
    #     config['amp_input_shape'] = self._amp_observation_space.shape
    #     return config

    # def _init_train(self):
    #     super()._init_train()
    #     self._init_amp_demo_buf()
    #     return

    # def _disc_loss(self, disc_agent_logit, disc_demo_logit, obs_demo):
    #     # prediction loss
    #     disc_loss_agent = self._disc_loss_neg(disc_agent_logit)
    #     disc_loss_demo = self._disc_loss_pos(disc_demo_logit)
    #     disc_loss = 0.5 * (disc_loss_agent + disc_loss_demo)

    #     # logit reg
    #     logit_weights = self.model.a2c_network.get_disc_logit_weights()
    #     disc_logit_loss = torch.sum(torch.square(logit_weights))
    #     disc_loss += self._disc_logit_reg * disc_logit_loss

    #     # grad penalty
    #     disc_demo_grad = torch.autograd.grad(disc_demo_logit, obs_demo, grad_outputs=torch.ones_like(disc_demo_logit),
    #                                          create_graph=True, retain_graph=True, only_inputs=True)
    #     disc_demo_grad = disc_demo_grad[0]
    #     disc_demo_grad = torch.sum(torch.square(disc_demo_grad), dim=-1)
    #     disc_grad_penalty = torch.mean(disc_demo_grad)
    #     disc_loss += self._disc_grad_penalty * disc_grad_penalty

    #     # weight decay
    #     if (self._disc_weight_decay != 0):
    #         disc_weights = self.model.a2c_network.get_disc_weights()
    #         disc_weights = torch.cat(disc_weights, dim=-1)
    #         disc_weight_decay = torch.sum(torch.square(disc_weights))
    #         disc_loss += self._disc_weight_decay * disc_weight_decay

    #     disc_agent_acc, disc_demo_acc = self._compute_disc_acc(disc_agent_logit, disc_demo_logit)

    #     disc_info = {
    #         'disc_loss': disc_loss,
    #         'disc_grad_penalty': disc_grad_penalty,
    #         'disc_logit_loss': disc_logit_loss,
    #         'disc_agent_acc': disc_agent_acc,
    #         'disc_demo_acc': disc_demo_acc,
    #         'disc_agent_logit': disc_agent_logit,
    #         'disc_demo_logit': disc_demo_logit
    #     }
    #     return disc_info

    # def _disc_loss_neg(self, disc_logits):
    #     bce = torch.nn.BCEWithLogitsLoss()
    #     loss = bce(disc_logits, torch.zeros_like(disc_logits))
    #     return loss
    
    # def _disc_loss_pos(self, disc_logits):
    #     bce = torch.nn.BCEWithLogitsLoss()
    #     loss = bce(disc_logits, torch.ones_like(disc_logits))
    #     return loss

    # def _compute_disc_acc(self, disc_agent_logit, disc_demo_logit):
    #     agent_acc = disc_agent_logit < 0
    #     agent_acc = torch.mean(agent_acc.float())
    #     demo_acc = disc_demo_logit > 0
    #     demo_acc = torch.mean(demo_acc.float())
    #     return agent_acc, demo_acc

    # def _fetch_amp_obs_demo(self, num_samples):
    #     amp_obs_demo = self.vec_env.env.fetch_amp_obs_demo(num_samples)
    #     return amp_obs_demo

    def _build_gt_act_buffers(self):
        batch_shape = self.experience_buffer.obs_base_shape
        # add the 
        self.experience_buffer.tensor_dict['gt_act'] = torch.zeros(batch_shape + (22,),
                                                                    device=self.ppo_device)
        
        # amp_obs_demo_buffer_size = int(self.config['amp_obs_demo_buffer_size'])
        # self._amp_obs_demo_buffer = replay_buffer.ReplayBuffer(amp_obs_demo_buffer_size, self.ppo_device)

        # self._amp_replay_keep_prob = self.config['amp_replay_keep_prob']
        # replay_buffer_size = int(self.config['amp_replay_buffer_size'])
        # self._amp_replay_buffer = replay_buffer.ReplayBuffer(replay_buffer_size, self.ppo_device)

        self.tensor_list += ['gt_act']
        return

    # def _init_amp_demo_buf(self):
    #     buffer_size = self._amp_obs_demo_buffer.get_buffer_size()
    #     num_batches = int(np.ceil(buffer_size / self._amp_batch_size))

    #     for i in range(num_batches):
    #         curr_samples = self._fetch_amp_obs_demo(self._amp_batch_size)
    #         self._amp_obs_demo_buffer.store({'amp_obs': curr_samples})

    #     return
    
    # def _update_amp_demos(self):
    #     new_amp_obs_demo = self._fetch_amp_obs_demo(self._amp_batch_size)
    #     self._amp_obs_demo_buffer.store({'amp_obs': new_amp_obs_demo})
    #     return

    # def _preproc_amp_obs(self, amp_obs):
    #     if self._normalize_amp_input:
    #         amp_obs = self._amp_input_mean_std(amp_obs)
    #     return amp_obs

    # def _combine_rewards(self, task_rewards, amp_rewards):
    #     disc_r = amp_rewards['disc_rewards']
    #     combined_rewards = self._task_reward_w * task_rewards + \
    #                      + self._disc_reward_w * disc_r
    #     return combined_rewards

    # def _eval_disc(self, amp_obs):
    #     proc_amp_obs = self._preproc_amp_obs(amp_obs)
    #     return self.model.a2c_network.eval_disc(proc_amp_obs)

    # def _calc_amp_rewards(self, amp_obs):
    #     disc_r = self._calc_disc_rewards(amp_obs)
    #     output = {
    #         'disc_rewards': disc_r
    #     }
    #     return output

    # def _calc_disc_rewards(self, amp_obs):
    #     with torch.no_grad():
    #         disc_logits = self._eval_disc(amp_obs)
    #         prob = 1 / (1 + torch.exp(-disc_logits)) 
    #         disc_r = -torch.log(torch.maximum(1 - prob, torch.tensor(0.0001, device=self.ppo_device)))
    #         disc_r *= self._disc_reward_scale
    #     return disc_r

    # def _store_replay_amp_obs(self, amp_obs):
    #     buf_size = self._amp_replay_buffer.get_buffer_size()
    #     buf_total_count = self._amp_replay_buffer.get_total_count()
    #     if (buf_total_count > buf_size):
    #         keep_probs = to_torch(np.array([self._amp_replay_keep_prob] * amp_obs.shape[0]), device=self.ppo_device)
    #         keep_mask = torch.bernoulli(keep_probs) == 1.0
    #         amp_obs = amp_obs[keep_mask]

    #     self._amp_replay_buffer.store({'amp_obs': amp_obs})
    #     return

    # def _record_train_batch_info(self, batch_dict, train_info):
    #     train_info['disc_rewards'] = batch_dict['disc_rewards']
    #     return

    # def _log_train_info(self, train_info, frame):
    #     super()._log_train_info(train_info, frame)

    #     self.writer.add_scalar('losses/disc_loss', torch_ext.mean_list(train_info['disc_loss']).item(), frame)

    #     self.writer.add_scalar('info/disc_agent_acc', torch_ext.mean_list(train_info['disc_agent_acc']).item(), frame)
    #     self.writer.add_scalar('info/disc_demo_acc', torch_ext.mean_list(train_info['disc_demo_acc']).item(), frame)
    #     self.writer.add_scalar('info/disc_agent_logit', torch_ext.mean_list(train_info['disc_agent_logit']).item(), frame)
    #     self.writer.add_scalar('info/disc_demo_logit', torch_ext.mean_list(train_info['disc_demo_logit']).item(), frame)
    #     self.writer.add_scalar('info/disc_grad_penalty', torch_ext.mean_list(train_info['disc_grad_penalty']).item(), frame)
    #     self.writer.add_scalar('info/disc_logit_loss', torch_ext.mean_list(train_info['disc_logit_loss']).item(), frame)

    #     disc_reward_std, disc_reward_mean = torch.std_mean(train_info['disc_rewards'])
    #     self.writer.add_scalar('info/disc_reward_mean', disc_reward_mean.item(), frame)
    #     self.writer.add_scalar('info/disc_reward_std', disc_reward_std.item(), frame)
    #     return

    # def _amp_debug(self, info):
    #     with torch.no_grad():
    #         amp_obs = info['amp_obs']
    #         amp_obs = amp_obs[0:1]
    #         disc_pred = self._eval_disc(amp_obs)
    #         amp_rewards = self._calc_amp_rewards(amp_obs)
    #         disc_reward = amp_rewards['disc_rewards']

    #         disc_pred = disc_pred.detach().cpu().numpy()[0, 0]
    #         disc_reward = disc_reward.cpu().numpy()[0, 0]
    #         print("disc_pred: ", disc_pred, disc_reward)
    #     return
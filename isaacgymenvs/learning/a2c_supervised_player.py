from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
from rl_games.common.experience import ExperienceBuffer
from datetime import datetime
import os
from rl_games.algos_torch import  model_builder


from isaacgymenvs.learning.transformer_layers import TransformerFeatureProcessing



def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action


class A2CSupervisedPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        
        self.params = params.copy()
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]
        
        
        full_experiment_name = params['config'].get('full_experiment_name', None)
        if full_experiment_name:
            print(f'Exact experiment name requested from command line: {full_experiment_name}')
            self.experiment_name = full_experiment_name
        else:
            self.experiment_name = params['config']['name'] + datetime.now().strftime("_%d-%H-%M-%S")
        
        self.train_dir = params['config'].get('train_dir', 'runs')
        # a folder inside of train_dir containing everything related to a particular experiment
        self.experiment_dir = os.path.join(self.train_dir, self.experiment_name)
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        self.experiment_dir  = self.env.exp_logging_dir # 
        
        # record experiences # # record experiences # # record experiences # # # record experiences #
        
        self.record_experiences = params['config'].get('record_experiences', False)
        
        if self.record_experiences:
            self.value_bootstrap = self.config.get('value_bootstrap')
            self.gamma = self.config.get('gamma')
            self.num_actors = params['config']['num_actors']
            self.central_value_config = params['config'].get('central_value_config', None)
            self.has_central_value = self.central_value_config is not None
            self.use_action_masks = params['config'].get('use_action_masks', False)
            self.rewards_shaper = self.config['reward_shaper']
            self.ppo_device = self.device
            self.update_list = ['actions', 'neglogpacs', 'values', 'mus', 'sigmas']
            if self.use_action_masks:
                self.update_list += ['action_masks']
            algo_info = {
                'num_actors' : self.num_actors,
                'horizon_length' : 300,
                'has_central_value' : self.has_central_value,
                'use_action_masks' : self.use_action_masks
            }
            self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.device)
            self._build_delta_targets_buffers()


        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        
        ## action chunking settings; action ## 
        self.action_chunking = params['config'].get('action_chunking', False)
        self.action_chunking_frames = params['config'].get('action_chunking_frames', 1)
        self.bc_style_training = params['config'].get('bc_style_training', False)
        self.distill_via_bc = params['config'].get('distill_via_bc', False)
        ## action chunking settings ##
        
        # self.bc_style_training = params['config'].get('bc_style_training', False)
        # 
        self.history_length = params['config'].get('history_length', 1)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()
        
        self.record_for_distill_to_ctlv2 = params['config'].get('record_for_distill_to_ctlv2', False)
        
        self.save_experiences_via_ts = params['config'].get('save_experiences_via_ts', False)
        
        # # switch between different models #
        # # use one model and them use a different model # 
        # print(f"switch_between_models:", params['config']['swtich_between_models'])
        # self.switch_between_models = params['config'].get('swtich_between_models', False)
        # self.switch_to_trans_model_frame_after = params['config'].get('switch_to_trans_model_frame_after', 310)
        # self.switch_to_trans_model_ckpt_fn = params['config'].get('switch_to_trans_model_ckpt_fn', '')
        
        self.switch_between_models = self.env.switch_between_models
        self.switch_to_trans_model_frame_after = self.env.switch_to_trans_model_frame_after
        self.switch_to_trans_model_ckpt_fn = self.env.switch_to_trans_model_ckpt_fn
        
        self.use_transformer_model = params['config'].get('use_transformer_model', False)
        
        if self.use_transformer_model:
            self._init_model_transformers()
            self.transformer_encoder_decoder.eval()
        
        
        print(f"switch_between_models: {self.switch_between_models}")
        if self.switch_between_models and len(self.switch_to_trans_model_ckpt_fn) > 0 and os.path.exists(self.switch_to_trans_model_ckpt_fn):
            self._build_trans_model()
    
    def _init_model_transformers(self, ):
        # transformer encoder and the transformer decoder #
        # during the traiing, we need to set this model to train
        # we also need to add an argument to control this model's usage
        self.transformer_encoder_decoder = TransformerFeatureProcessing(nn_latents=256, dropout=0.1)
        self.transformer_encoder_decoder = self.transformer_encoder_decoder.to(self.device)
        # self.transformer_encoder_decoder.train() #
        # self.transformer_optimizer = optim.Adam(self.transformer_encoder_decoder.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        # self.transformer_scaler = torch.cuda.amp.GradScaler(enabled=self.mixed_precision)
        pass
    
    def _build_trans_model(self, ): #
        self.trans_model_params = self.params.copy()
        self.trans_model_builder = model_builder.ModelBuilder()
        self.trans_model_builder = self.trans_model_builder.load(self.trans_model_params)
        
        cur_model_build_config = {
            'actions_num' : self.actions_num,
            'input_shape' : (self.obs_shape[0], ),
            'num_seqs' :  self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input, # # normalize # #
        }
        self.trans_model = self.trans_model_builder.build(cur_model_build_config)
        
        if len(self.switch_to_trans_model_ckpt_fn) > 0 and os.path.exists(self.switch_to_trans_model_ckpt_fn):
            print(f"[INFO] loading trans model weights from {self.switch_to_trans_model_ckpt_fn}")
            trans_model_weights = torch.load(self.switch_to_trans_model_ckpt_fn, map_location='cpu')
            self.trans_model.load_state_dict(trans_model_weights['model'])
            self.trans_model.to(self.device)
            self.trans_model.eval()
            

    
    # def init_tensors(self):
    #     super().init_tensors()
    #     # self._build_gt_act_buffers() # 
    #     # if self.preload_experiences_tf:
    #     #     if self.action_chunking and self.bc_style_training:
    #     #         self.preload_saved_experiences_chunking_bc()
    #     #     else:
    #     #         self.preload_saved_experiences()
    #     self._build_delta_targets_buffers()
    #     return
    
    def _build_delta_targets_buffers(self):
        print(f"start init tensors")
        batch_shape = self.experience_buffer.obs_base_shape
        # build additional buffers #
        act_dim = self.actions_num
        self.experience_buffer.tensor_dict['delta_targets'] = torch.zeros(batch_shape + (act_dim,),
                                                                    device=self.ppo_device)
        # self.tensor_list += ['delta_targets']
    
    
    def _forward_transformer_model(self, input_dict):
        
        input_obses = input_dict['obs']
        output_mus = self.transformer_encoder_decoder(input_obses, nn_history=self.history_length, nn_future=self.action_chunking_frames)
        
        prev_neglogp = torch.ones_like(output_mus )[:, 0] * 0.1
        values = torch.ones_like(output_mus)[:, 0]
        entropy = torch.ones_like(output_mus ) * 0.1
        mus = output_mus
        sigmas = torch.ones_like(mus) * 0.1
        
        res_dict = {
            'actions': mus,
            'prev_neglogp': prev_neglogp,
            'values': values,
            'entropy': entropy,
            'mus': mus,
            'sigmas': sigmas,
            'rnn_states': input_dict['rnn_states']
        }
        return res_dict
    
    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs[..., : self.obs_shape[0]],
            'rnn_states' : self.states
        }
        with torch.no_grad():
            if self.switch_between_models and self.steps[0].item() > self.switch_to_trans_model_frame_after:
                print(f"step: {self.steps[0].item()}, using trans_model")
                res_dict = self.trans_model(input_dict)
            
            if self.use_transformer_model:
                # res_dict = self.transformer_encoder_decoder(input_dict)
                res_dict = self._forward_transformer_model(input_dict)
            else:
                res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.bc_style_training:
            return current_action
        else:
            # return current_action
            if self.clip_actions:
                return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
            else:
                return current_action

    def restore(self, fn): # restore #
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        if 'transformer_encoder_decoder' in checkpoint:
            print(f"Loading transformer_encoder_decoder from {fn}")
            self.transformer_encoder_decoder.load_state_dict(checkpoint['transformer_encoder_decoder'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)
            
            
        # TODO: load the transformer_encoder_deocder model #

    def reset(self):
        self.init_rnn()
        
    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs)
        if self.use_transformer_model:
            self.transformer_encoder_decoder.eval()
        else:
            self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., : self.obs_shape[0]],
            'rnn_states' : self.states
        }

        # get the actions values #
        with torch.no_grad(): # get mus and get actions --- in the network #
            if self.use_transformer_model:
                # res_dict = self.transformer_encoder_decoder(input_dict)
                res_dict = self._forward_transformer_model(input_dict)
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
        return res_dict
        
    def run(self):
        n_games = self.games_num
        render = self.render_env
        n_game_life = self.n_game_life
        is_deterministic = self.is_deterministic
        sum_rewards = 0
        sum_steps = 0
        sum_game_res = 0
        n_games = n_games * n_game_life
        games_played = 0
        has_masks = False
        has_masks_func = getattr(self.env, "has_action_mask", None) is not None

        op_agent = getattr(self.env, "create_agent", None)
        if op_agent:
            agent_inited = True
            # print('setting agent weights for selfplay')
            # self.env.create_agent(self.env.config)
            # self.env.set_weights(range(8),self.get_weights())

        if has_masks_func:
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()
        
        if self.record_experiences:
            self.dones = torch.zeros((self.num_agents, ), dtype=torch.float32).to(self.device)
            update_list = self.update_list

        need_init_rnn = self.is_rnn
        for _ in range(n_games): 
            if games_played >= n_games: 
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses, batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)
            
            self.steps = steps

            print_game_res = False

            for n in range(self.max_steps):
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    if self.action_chunking and self.bc_style_training:
                        action = self.get_action(obses, True)
                    else:
                        # print(f"deterministic: {is_deterministic}")
                        action = self.get_action(obses, is_deterministic)
                        # action = self.get_action(obses, True)
                    
                if self.record_experiences:
                    res_dict = self.get_action_values(obses)
                    
                    # 1) observations obtained from 
                    # gt_act_val = obses['obs'][..., self.obs_shape[0]: ]
                    # self.experience_buffer.update_data('gt_act', n, gt_act_val)
                    
                    obses_to_record = obses[..., : self.obs_shape[0]].clone()
                    if self.record_for_distill_to_ctlv2:
                        print(f"record_for_distill_to_ctlv2: {self.record_for_distill_to_ctlv2}")
                        obses_to_record[:, self.env.cur_delta_start: self.env.cur_delta_start + self.env.num_shadow_hand_dofs] = self.env.prev_targets[:, : self.env.num_shadow_hand_dofs]
                    
                    # self.experience_buffer.update_data('obses', n, obses[..., : self.obs_shape[0]])
                    self.experience_buffer.update_data('obses', n, obses_to_record[..., : self.obs_shape[0]])
                    
                    self.experience_buffer.update_data('dones', n, self.dones)
                    
                    for k in update_list:
                        self.experience_buffer.update_data(k, n, res_dict[k]) 
                    if self.has_central_value:
                        self.experience_buffer.update_data('states', n, self.obs['states'])


                if self.switch_between_models and steps[0].item() >= self.switch_to_trans_model_frame_after:
                    self.env.use_local_canonical_state = True
                

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1
                
                
                # runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-17-28-12/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
                if self.record_experiences:
                    shaped_rewards = self.rewards_shaper(r).to(self.ppo_device).unsqueeze(1)
                    if self.value_bootstrap and 'time_outs' in info: # 
                        # print(f"shaped_rewards: {shaped_rewards.size()}, values: {res_dict['values'].size()}, time_outs: {self.cast_obs(info['time_outs']).size()}")
                        residual_rew = self.gamma * res_dict['values'] * self.cast_obs(info['time_outs']).unsqueeze(1).float()
                        # print(f"residual_rew: {residual_rew.size()}")
                        shaped_rewards += residual_rew # .squeeze(1)

                    self.experience_buffer.update_data('rewards', n, shaped_rewards)
                    try:
                        delta_targets = self.env.true_delta_targets
                        self.experience_buffer.update_data('delta_targets', n, delta_targets) 
                    except:
                        pass
                    

                if render:
                    self.env.render(mode='human')
                    time.sleep(self.render_sleep)

                all_done_indices = done.nonzero(as_tuple=False)
                done_indices = all_done_indices[::self.num_agents]
                done_count = len(done_indices)
                games_played += done_count

                if done_count > 0: # done count #
                    if self.is_rnn:
                        for s in self.states:
                            s[:, all_done_indices, :] = s[:,
                                                          all_done_indices, :] * 0.0

                    cur_rewards = cr[done_indices].sum().item()
                    cur_steps = steps[done_indices].sum().item()

                    cr = cr * (1.0 - done.float())
                    steps = steps * (1.0 - done.float())
                    sum_rewards += cur_rewards
                    sum_steps += cur_steps

                    game_res = 0.0
                    if isinstance(info, dict):
                        if 'battle_won' in info:
                            print_game_res = True
                            game_res = info.get('battle_won', 0.5)
                        if 'scores' in info:
                            print_game_res = True
                            game_res = info.get('scores', 0.5)

                    if self.print_stats:
                        cur_rewards_done = cur_rewards/done_count
                        cur_steps_done = cur_steps/done_count
                        if print_game_res:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f} w: {game_res}')
                        else:
                            print(f'reward: {cur_rewards_done:.2f} steps: {cur_steps_done:.1f}')

                    sum_game_res += game_res
                    if batch_size//self.num_agents == 1 or games_played >= n_games:
                        break
        
        
        
                # print(f"record_experiences: {self.record_experiences}")
                step_threshold = 269
                if self.record_experiences and steps[0].item() >= step_threshold: #  285: # if we record experiences, 
                    # from experience buffer to numpy #
                    tensor_dict_np = { 
                        key: self.experience_buffer.tensor_dict[key].cpu().numpy() for key in self.experience_buffer.tensor_dict
                    }
                    for key in tensor_dict_np:
                        print(f"{key}: {tensor_dict_np[key].shape}")
                    target_key_in_tensor_dict = list(tensor_dict_np.keys())[0]
                    tot_envs = tensor_dict_np[target_key_in_tensor_dict].shape[1]
                    maxx_save_envs_per_np = 1000
                    tot_chunks = tot_envs // maxx_save_envs_per_np
                    if tot_envs % maxx_save_envs_per_np > 0:
                        tot_chunks += 1
                        
                    if not self.save_experiences_via_ts:
                        for i in range(tot_chunks):
                            start_idx = i * maxx_save_envs_per_np
                            end_idx = start_idx + maxx_save_envs_per_np
                            tensor_dict_np_chunk = {
                                key: tensor_dict_np[key][:, start_idx:end_idx] for key in tensor_dict_np
                            }
                            tensor_dict_sv_fn = f"./experience_buffer_sv_{i}.npy"
                            tensor_dict_sv_fn = os.path.join(self.experiment_dir, tensor_dict_sv_fn)
                            print(f"experiences saved to {tensor_dict_sv_fn}")
                            np.save(tensor_dict_sv_fn, tensor_dict_np_chunk)
                    else:
                        ###### try to save the data via timestamps ######
                        tot_ts = tensor_dict_np[target_key_in_tensor_dict].shape[0]
                        for i_ts in range(tot_ts):
                            cur_ts_key_to_tensor_dict = {
                                key: tensor_dict_np[key][i_ts] for key in tensor_dict_np
                            }
                            cur_ts_dict_sv_fn = f"./experience_buffer_sv_ts_{i_ts}.npy"
                            cur_ts_dict_sv_fn = os.path.join(self.experiment_dir, cur_ts_dict_sv_fn)
                            print(f"experiences saved to {cur_ts_dict_sv_fn}")
                            np.save(cur_ts_dict_sv_fn, cur_ts_key_to_tensor_dict) 
                        
                        ###### try to save the data via timestamps ######
                    
                    
                    # for i_ts in range()
                
            self.steps = steps
            
            
            # for key in tensor_dict_np:
            #     # tot_nn_envs = tensor_dict_np[key]
            #     tensor_dict_np[key] = tensor_dict_np[key][:, :1000]
            # tensor_dict_sv_fn = "./experience_buffer_sv.npy"
            # np.save(tensor_dict_sv_fn, tensor_dict_np)
        print(sum_rewards)
        
        if print_game_res:
            print('av reward:', sum_rewards / games_played * n_game_life, 'av steps:', sum_steps /
                  games_played * n_game_life, 'winrate:', sum_game_res / games_played * n_game_life)
        else:
            print('av reward:', sum_rewards / games_played * n_game_life,
                  'av steps:', sum_steps / games_played * n_game_life)



class PpoPlayerContinuous(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]

        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)

        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        } 
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        mu = res_dict['mus']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if is_deterministic:
            current_action = mu
        else:
            current_action = action
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class PpoPlayerDiscrete(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)

        self.network = self.config['network']
        if type(self.action_space) is gym.spaces.Discrete:
            self.actions_num = self.action_space.n
            self.is_multi_discrete = False
        if type(self.action_space) is gym.spaces.Tuple:
            self.actions_num = [action.n for action in self.action_space]
            self.is_multi_discrete = True
        self.mask = [False]
        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        obs_shape = self.obs_shape
        config = {
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'num_seqs' : self.num_agents,
            'value_size': self.env_info.get('value_size',1),
            'normalize_value': self.normalize_value,
            'normalize_input': self.normalize_input,
        }

        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def get_masked_action(self, obs, action_masks, is_deterministic = True):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)
        action_masks = torch.Tensor(action_masks).to(self.device).bool()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'action_masks' : action_masks,
            'rnn_states' : self.states
        }
        self.model.eval()

        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=-1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self._preproc_obs(obs)

        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs,
            'rnn_states' : self.states
        }
        with torch.no_grad():
            res_dict = self.model(input_dict)
        logits = res_dict['logits']
        action = res_dict['actions']
        self.states = res_dict['rnn_states']
        if self.is_multi_discrete:
            if is_deterministic:
                action = [torch.argmax(logit.detach(), axis=1).squeeze() for logit in logits]
                return torch.stack(action,dim=-1)
            else:    
                return action.squeeze().detach()
        else:
            if is_deterministic:
                return torch.argmax(logits.detach(), axis=-1).squeeze()
            else:    
                return action.squeeze().detach()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()


class SACPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.action_range = [
            float(self.env_info['action_space'].low.min()),
            float(self.env_info['action_space'].high.max())
        ]

        obs_shape = self.obs_shape
        self.normalize_input = self.config.get('normalize_input', False)
        config = {
            'obs_dim': self.env_info["observation_space"].shape[0],
            'action_dim': self.env_info["action_space"].shape[0],
            'actions_num' : self.actions_num,
            'input_shape' : obs_shape,
            'value_size': self.env_info.get('value_size', 1),
            'normalize_value': False,
            'normalize_input': self.normalize_input,
        }  
        self.model = self.network.build(config)
        self.model.to(self.device)
        self.model.eval()
        self.is_rnn = self.model.is_rnn()

    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.sac_network.actor.load_state_dict(checkpoint['actor'])
        self.model.sac_network.critic.load_state_dict(checkpoint['critic'])
        self.model.sac_network.critic_target.load_state_dict(checkpoint['critic_target'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def get_action(self, obs, is_deterministic=False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        obs = self.model.norm_obs(obs)
        dist = self.model.actor(obs)
        actions = dist.sample() if not is_deterministic else dist.mean
        actions = actions.clamp(*self.action_range).to(self.device)
        if self.has_batch_dimension == False:
            actions = torch.squeeze(actions.detach())
        return actions

    def reset(self):
        pass
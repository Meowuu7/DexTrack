from rl_games.common.player import BasePlayer
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common.tr_helpers import unsqueeze_obs
import gym
import torch 
from torch import nn
import numpy as np
from rl_games.common.experience import ExperienceBuffer
from rl_games.algos_torch import  model_builder


def rescale_actions(low, high, action):
    d = (high - low) / 2.0
    m = (high + low) / 2.0
    scaled_action =  action * d + m
    return scaled_action

def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


class A2CSupervisedWForecastingPlayer(BasePlayer):

    def __init__(self, params):
        BasePlayer.__init__(self, params)
        
        self.params = params.copy()
        self.network = self.config['network']
        self.actions_num = self.action_space.shape[0] 
        self.actions_low = torch.from_numpy(self.action_space.low.copy()).float().to(self.device)
        self.actions_high = torch.from_numpy(self.action_space.high.copy()).float().to(self.device)
        self.mask = [False]
        
        
        self.ppo_device = self.device
        # record the experience #
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
            self.update_list = ['actions', 'neglogpacs', 'values']
            if self.use_action_masks:
                self.update_list += ['action_masks']
            algo_info = {
                'num_actors' : self.num_actors,
                'horizon_length' : 300,
                'has_central_value' : self.has_central_value,
                'use_action_masks' : self.use_action_masks
            }
            self.experience_buffer = ExperienceBuffer(self.env_info, algo_info, self.device)


        self.normalize_input = self.config['normalize_input']
        self.normalize_value = self.config.get('normalize_value', False)
        self.use_world_model = params['config'].get('use_world_model', False)
        
        
        # get the 
        self.train_controller = params['config'].get('train_controller', False)
        self.train_forecasting_model = params['config'].get('train_forecasting_model', True)
        

        """ Load model """
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
        
        """ Load the forecasting model """
        self.w_forecasting_model = True
        self.forecasting_obs_dim = params['config'].get('forecasting_obs_dim', 797)
        self.forecasting_obs_dim =  (797 - 256 - 512) * 2
        self.forecasting_act_dim = params['config'].get('forecasting_act_dim', 29)
        self.forecasting_nn_frames = params['config'].get('forecasting_nn_frames', 10)
        self.forecasting_nn_frames = 2
        self.forecasting_act_dim_per_frame = self.forecasting_act_dim
        self.forecasting_act_dim = self.forecasting_act_dim * self.forecasting_nn_frames 
        self.actions_low = torch.cat(
            [
                self.actions_low, torch.zeros(self.forecasting_act_dim, device=self.device) - 1.0
            ], dim=-1
        )
        self.actions_high = torch.cat(
            [
                self.actions_high, torch.zeros(self.forecasting_act_dim, device=self.device) + 1.0
            ], dim=-1
        )
        
        if self.w_forecasting_model:
            self._init_forecasting_model()
            
        self.vec_env = self.env
            
    def obs_to_torch(self, obs):
        if isinstance(obs, dict):
            # if 'obs' in obs:
            #     obs = obs['obs'] # 
            if isinstance(obs, dict):
                upd_obs = {}
                for key, value in obs.items():
                    upd_obs[key] = self._obs_to_tensors_internal(value, False)
            else:
                upd_obs = self.cast_obs(obs)
        else:
            upd_obs = self.cast_obs(obs)
        return upd_obs
            
    def _init_forecasting_model(self, ):
        assert self.w_forecasting_model
        
        obj_feat_shape = 256
        
        self.forecasting_network_params = self.params.copy()
        self.forecasting_network_params['network']['mlp']['units'] = [1024, 512, 256, 128] 
        self.forecasting_network_builder = model_builder.ModelBuilder()
        self.forecasting_network_builder = self.forecasting_network_builder.load(self.forecasting_network_params)
        
        # wplanning #
        forecasting_model_build_config = {
            'actions_num' : self.forecasting_act_dim,
            'input_shape' : (self.forecasting_obs_dim, ),
            'num_seqs' : self.num_agents, # num seqs? #
            'value_size': self.env_info.get('value_size',1),
            'normalize_value' : self.normalize_value,
            'normalize_input': self.normalize_input,
        } # ppo device #
        forecasting_model = self.forecasting_network_builder.build(forecasting_model_build_config).to(self.device)
        # teacher_model_weights = torch.load(teacher_model_weight_fn, map_location='cpu')
        # teacher_model.load_state_dict(teacher_model_weights['model'])
        forecasting_model.eval()
        self.forecasting_model = forecasting_model
        
        self.forecasting_states = None
        
        
        # self.init_rnn_from_model(self.forecasting_model)
        # self.last_lr = float(self.last_lr)
        # self.bound_loss_type = self.config.get('bound_loss_type', 'bound') # 'regularisation' or 'bound'
        # self.forecasting_optimizer = optim.Adam(self.forecasting_model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)


    # if it; and for each model? #
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
            # print(f"res_dict: {res_dict.}")
            # sicne the modle has uncertainty here #
            
    # 
    def get_action(self, obs, is_deterministic = False):
        if self.has_batch_dimension == False:
            obs = unsqueeze_obs(obs)
        
        
        
        
        """ Infer the forward forecasting results """
        forecasting_obs_val = obs['forecasting_obs']
        if not self.has_batch_dimension:
            forecasting_obs_val = forecasting_obs_val.unsqueeze(0)
        forecasting_obs_val = self._preproc_obs(forecasting_obs_val[..., :self.forecasting_obs_dim])
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : forecasting_obs_val[..., :],
            'rnn_states' : None
        }
        
        with torch.no_grad():
            forecasting_res_dict = self.forecasting_model(input_dict)
        """ Infer the forward forecasting results """
        
        if self.train_forecasting_model and self.vec_env.activate_forecaster: # only in this way can we modify the observation space input to the controller module #
            """ Use the forecasting result to change the delta_qpos in the observation """
            # print(f"[forecaster] forecasting_res_dict: {forecasting_res_dict.keys()}")
            tmp_forecasting_actions = forecasting_res_dict['actions']
            ## update obs via the forecasted actions #
            # tmp_forecasting_actions: nn_envs x (nn_act_dim x nn_futruer_obs)
            tmp_forecasting_actions = tmp_forecasting_actions.contiguous().view(tmp_forecasting_actions.size(0), -1, self.forecasting_act_dim_per_frame)
            hand_dof_nn = self.vec_env.num_shadow_hand_dofs
            tmp_forecasting_actions = torch.cumsum(tmp_forecasting_actions, dim=1) # 
            cur_step_forecasting_actions = tmp_forecasting_actions[:, 0, :][..., :hand_dof_nn]
            nex_step_forecasting_actions = tmp_forecasting_actions[:, 1, :][..., :hand_dof_nn]

            cur_step_goal_obj_pos = tmp_forecasting_actions[:, 0, :][..., hand_dof_nn: hand_dof_nn + 3]
            # 
            # # # hand dof pose # # # # to inference # # to 
            cur_step_hand_dof_pos = obs['forecasting_obs'][..., : hand_dof_nn] # nn_envs x nn_hand_dof 
            shadow_hand_dof_speed_scale_tsr = self.vec_env.shadow_hand_dof_speed_scale_tsr # nn_hand_dof
            ### TODO: change it to the accumulated version ###
            ### TODO: initilaize the pred_targets --- the first frame should be the first frame of the original kine refs #
            cur_step_forecasting_actions = cur_step_forecasting_actions * shadow_hand_dof_speed_scale_tsr * self.vec_env.dt
            nex_step_forecasting_actions = nex_step_forecasting_actions * shadow_hand_dof_speed_scale_tsr * self.vec_env.dt
            cur_step_forecasting_actions = cur_step_forecasting_actions + cur_step_hand_dof_pos # n_envs x nn_hand_dof
            nex_step_forecasting_actions = nex_step_forecasting_actions + cur_step_hand_dof_pos
            hand_goal_start = int(obs['hand_goal_start'][0].item())
            delta_qpos = cur_step_hand_dof_pos - cur_step_forecasting_actions
            
            # print(f"[forecaster] hand_goal_start: {hand_goal_start}, hand_dof_nn: {hand_dof_nn}, shadow_hand_dof_speed_scale_tsr: {shadow_hand_dof_speed_scale_tsr}, dt: {self.vec_env.dt}")
            obs['obs'][..., hand_goal_start + 7 : hand_goal_start + 7 + hand_dof_nn] = delta_qpos
            # gou #  # obs and delta qpos #
            
            
            nex_ref_start = int(obs['hand_goal_start'][1].item())
            if self.vec_env.use_local_canonical_state: # nex_hand_qpos_ref #
                canon_hand_qpos_trans = nex_step_forecasting_actions[..., :3] - self.vec_env.object_pos
                canon_hand_qpos_ref = torch.cat( # obj pos #
                    [ canon_hand_qpos_trans, nex_step_forecasting_actions[..., 3:] ], dim=-1
                )
            else:
                canon_hand_qpos_ref = nex_step_forecasting_actions
            
            # unscale the qpos # 
            unscaled_nex_hand_qpos_ref = unscale(canon_hand_qpos_ref, self.vec_env.shadow_hand_dof_lower_limits, self.vec_env.shadow_hand_dof_upper_limits)
            # obs['obs'][..., nex_ref_start : nex_ref_start + hand_dof_nn] = unscaled_nex_hand_qpos_ref
            
            
            if self.vec_env.forecast_obj_pos:
                ######### Replace the object goal pos ############
                obj_obs_start = int(obs['hand_goal_start'][2].item())
                cur_step_goal_obj_pos = cur_step_goal_obj_pos * shadow_hand_dof_speed_scale_tsr.unsqueeze(0)[..., :3] * self.vec_env.dt
                obs['obs'][..., obj_obs_start + 13:obj_obs_start + 16] = cur_step_goal_obj_pos #  self.goal_pos - self.object_pos
                ######### Replace the object goal pos ############
            """ Use the forecasting result to change the delta_qpos in the observation """
        
        
        
        obs = self._preproc_obs(obs)
        
        obs_val = obs['obs']
        if not self.has_batch_dimension:
            obs_val = obs_val.unsqueeze(0)
        obs_val = self._preproc_obs(obs_val)
        
        
        
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : obs_val[..., : self.obs_shape[0]],
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
            
        
        # forecasting_obs_val = obs['forecasting_obs']
        # if not self.has_batch_dimension:
        #     forecasting_obs_val = forecasting_obs_val.unsqueeze(0)
        # forecasting_obs_val = self._preproc_obs(forecasting_obs_val[..., :self.forecasting_obs_dim])
        # input_dict = {
        #     'is_train': False,
        #     'prev_actions': None, 
        #     'obs' : forecasting_obs_val[..., :],
        #     'rnn_states' : None
        # }
        
        # with torch.no_grad():
        #     forecasting_res_dict = self.forecasting_model(input_dict)
        
        forecasting_mu = forecasting_res_dict['mus']
        forecasting_actions = forecasting_res_dict['actions']
        if is_deterministic:
            current_forecasting_action = forecasting_mu
        else:
            if self.use_world_model:
                current_forecasting_action = forecasting_mu
            else:
                current_forecasting_action = forecasting_actions
        
        current_action = torch.cat(
            [ current_action, current_forecasting_action ], dim=-1
        )
        # get the actions and the current states # 
        # but inference model  cannot; reference motion # reference motion #
        # reference model # predicted results # predicted results 3
        # second one forecasting model # 
        # time 0 -- get observation - use the forcasted results as the next step ref in the observation - inference the model - predict action - step the environment - use the step reward as the forecasting model's reward - use the forecasting model's reward as the model's reward -- 
        # time 0 -- get observation #
        
        
        if self.has_batch_dimension == False:
            current_action = torch.squeeze(current_action.detach())

        if self.clip_actions:
            return rescale_actions(self.actions_low, self.actions_high, torch.clamp(current_action, -1.0, 1.0))
        else:
            return current_action

    # add a model in front of the model to get the forecasted results #
    def restore(self, fn):
        checkpoint = torch_ext.load_checkpoint(fn)
        self.model.load_state_dict(checkpoint['model'])
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.model.running_mean_std.load_state_dict(checkpoint['running_mean_std'])

        print(f"Loading the forecasting model from {fn}")
        self.forecasting_model.load_state_dict(checkpoint['forecasting_model'])
        
        env_state = checkpoint.get('env_state', None)
        if self.env is not None and env_state is not None:
            self.env.set_env_state(env_state)

    def reset(self):
        self.init_rnn()
        
    def get_action_values(self, obs):
        processed_obs = self._preproc_obs(obs)
        self.model.eval()
        input_dict = {
            'is_train': False,
            'prev_actions': None, 
            'obs' : processed_obs[..., : self.obs_shape[0]],
            'rnn_states' : self.states
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

        if has_masks_func: # has the mask #
            has_masks = self.env.has_action_mask()

        self.wait_for_checkpoint()
        
        if self.record_experiences:
            self.dones = torch.zeros((self.num_agents, ), dtype=torch.float32).to(self.device)
            update_list = self.update_list

        need_init_rnn = self.is_rnn
        
        for _ in range(n_games): # run #
            if games_played >= n_games: # 
                break

            obses = self.env_reset(self.env)
            batch_size = 1
            batch_size = self.get_batch_size(obses['obs'], batch_size)

            if need_init_rnn:
                self.init_rnn()
                need_init_rnn = False

            cr = torch.zeros(batch_size, dtype=torch.float32)
            steps = torch.zeros(batch_size, dtype=torch.float32)

            print_game_res = False

            for n in range(self.max_steps):
                if self.evaluation and n % self.update_checkpoint_freq == 0:
                    self.maybe_load_new_checkpoint()

                if has_masks:
                    masks = self.env.get_action_mask()
                    action = self.get_masked_action(
                        obses, masks, is_deterministic)
                else:
                    action = self.get_action(obses, is_deterministic)
                    
                if self.record_experiences:
                    res_dict = self.get_action_values(obses)
                    
                    # gt_act_val = obses['obs'][..., self.obs_shape[0]: ]
                    # self.experience_buffer.update_data('gt_act', n, gt_act_val)
                    self.experience_buffer.update_data('obses', n, obses[..., : self.obs_shape[0]])
                    self.experience_buffer.update_data('dones', n, self.dones)
                    
                    for k in update_list:
                        self.experience_buffer.update_data(k, n, res_dict[k]) 
                    if self.has_central_value:
                        self.experience_buffer.update_data('states', n, self.obs['states'])


                

                obses, r, done, info = self.env_step(self.env, action)
                cr += r
                steps += 1
                
                
                # record # record # record #
                # runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-17-28-12/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
                if self.record_experiences: # record experiences #
                    shaped_rewards = self.rewards_shaper(r).to(self.device).unsqueeze(1)
                    if self.value_bootstrap and 'time_outs' in info: # 
                        # print(f"shaped_rewards: {shaped_rewards.size()}, values: {res_dict['values'].size()}, time_outs: {self.cast_obs(info['time_outs']).size()}")
                        residual_rew = self.gamma * res_dict['values'] * self.cast_obs(info['time_outs']).unsqueeze(1).float()
                        # print(f"residual_rew: {residual_rew.size()}")
                        shaped_rewards += residual_rew # .squeeze(1)

                    self.experience_buffer.update_data('rewards', n, shaped_rewards)
                

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
                    # done count #
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
        
        print(sum_rewards)
        
        if self.record_experiences: # if we record experiences, 
            tensor_dict_np = {
                key: self.experience_buffer.tensor_dict[key].cpu().numpy() for key in self.experience_buffer.tensor_dict
            }
            for key in tensor_dict_np:
                print(f"{key}: {tensor_dict_np[key].shape}")
            for key in tensor_dict_np:
                tensor_dict_np[key] = tensor_dict_np[key][:, :1000]
            tensor_dict_sv_fn = "./experience_buffer_sv.npy"
            np.save(tensor_dict_sv_fn, tensor_dict_np)
            
        
        if print_game_res: # get thegame rewards # 
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
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
    from isaacgymenvs.learning import a2c_supervised_wplanning
    from isaacgymenvs.learning import a2c_supervised_wplanning_player
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
    
    
    net_type = cfg.task.env.net_type
    
    # if net_type == 'v4':
    #     cfg.train.params.network.mlp.units = [8192, 4096, 2048, 1024, 512, 256, 128]
    # elif net_type == 'v3':
    #     cfg.train.params.network.mlp.units = [4096, 2048, 1024, 512, 256, 128]
    # elif net_type == 'v2':
    #     cfg.train.params.network.mlp.units = [2048, 1024, 512, 256, 128]
    # elif net_type == 'v1':
    #     cfg.train.params.network.mlp.units = [1024, 512, 256, 128]
    # else:
    #     raise ValueError(f"Unknown net_type: {net_type}")
    
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
        
        
        runner.algo_factory.register_builder('a2c_continuous_supervised_wplanning', lambda **kwargs : a2c_supervised_wplanning.A2CSupervisedAgentWForecasting(**kwargs))
        runner.player_factory.register_builder('a2c_continuous_supervised_wplanning', lambda **kwargs : a2c_supervised_wplanning_player.A2CSupervisedWForecastingPlayer(**kwargs)) # player #
        # A2CSupervisedWForecastingPlayer
        # # runner.algo_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised_deterministic.A2CSupervisedAgent(**kwargs))
        # runner.player_factory.register_builder('a2c_continuous_supervised', lambda **kwargs : a2c_supervised_player.A2CSupervisedPlayer(**kwargs))
        
        
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


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

import torch
import numpy as np
import random

from omegaconf import open_dict
import argparse
from hydra import compose, initialize
from diffusion.dataset.get_data import get_dataset_loader_3d_pc, get_dataset_loader_3d_v3_pc, get_dataset_loader_3d_v5_pc, get_dataset_loader_3d_v6_pc, get_dataset_loader_3d_v7_pc
from diffusion.model_util import create_model_and_diffusion_3d_pc
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--sampling", default=False, action='store_true', help="Decay factor for exp")
parser.add_argument("--use_shadow_test_data", default=False, action='store_true', help="Decay factor for exp")
parser.add_argument("--task_cond", default=False, action='store_true', help="Decay factor for exp")
parser.add_argument("--resume_checkpoint", type=str, default='', help="Render mode for the environment")
parser.add_argument("--specified_test_subfolder", type=str, default='', help="Render mode for the environment")
parser.add_argument("--exp_tag", type=str, default='', help="Render mode for the environment")
parser.add_argument("--debug", default=False, action='store_true', help="debug flag")
parser.add_argument("--save_interval", type=int, default=20000, help="Render mode for the environment")
parser.add_argument("--statistics_info_fn", type=str, default='', help="Render mode for the environment")
parser.add_argument("--single_inst", default=False, action='store_true', help="single_inst flag")
parser.add_argument("--training_setting", type=str, default='regular_training', help="Render mode for the environment")
parser.add_argument("--use_t", type=int, default=200, help="Render mode for the environment")
parser.add_argument("--batch_size", type=int, default=16, help="Render mode for the environment")
parser.add_argument("--training_use_jointspace_seq", default=False, action='store_true', help="single_inst flag")
parser.add_argument("--diff_task_space", default=False, action='store_true')
parser.add_argument("--diff_task_translations", default=False, action='store_true')
parser.add_argument("--kine_diff", default=False, action='store_true')
parser.add_argument("--concat_two_dims", default=False, action='store_true')
parser.add_argument("--tracking_ctl_diff", default=False, action='store_true')
parser.add_argument("--AE_Diff", default=False, action='store_true') 
parser.add_argument("--train_AE", default=False, action='store_true')  
parser.add_argument("--train_Diff", default=False, action='store_true')  
parser.add_argument("--target_grab_inst_tag", type=str, default='', help="Render mode for the environment")
parser.add_argument("--target_grab_inst_opt_fn", type=str, default='', help="Render mode for the environment")
parser.add_argument("--cond_diff_allparams", default=False, action='store_true') 
parser.add_argument("--succ_rew_threshold", type=float, default=50.0, help="Render mode for the environment")
parser.add_argument("--multi_inst", default=False, action='store_true') 
parser.add_argument("--slicing_ws", type=int, default=30)
parser.add_argument("--slicing_data", default=False, action='store_true') 
parser.add_argument("--grab_inst_tag_to_opt_stat_fn", type=str, default='')
parser.add_argument("--sim_platform", type=str, default='pybullet')
parser.add_argument("--grab_inst_tag_to_optimized_res_fn", type=str, default='')
parser.add_argument("--task_cond_type", type=str, default='future')

pre_args = parser.parse_args()



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


@hydra.main(config_name="config", config_path="./cfg")
def launch_rlg_hydra(cfg: DictConfig):
    
    
    ### TODO: merge them into the tracking env file ## 
    ### TODO: how to pass the configurations ? ###
    with initialize(version_base="1.3", config_path="cfgs", job_name="test_app"):
        if os.path.exists("/cephfs/xueyi/backup"):
            diff_cfg = compose(config_name="K2_config_3d_k8s")
        elif os.path.exists("/root/diffsim/softzoo"):
            diff_cfg = compose(config_name="config_3d_k8s")
        else:
            raise ValueError("Please run this code on the k8s cluster")

    args = diff_cfg
    
    args.sampling.sampling = pre_args.sampling
    args.training.resume_checkpoint_pc = pre_args.resume_checkpoint
    args.sampling.use_shadow_test_data = pre_args.use_shadow_test_data
    args.sampling.specified_test_subfolder = pre_args.specified_test_subfolder
    args.training.task_cond = pre_args.task_cond
    args.training.save_interval = pre_args.save_interval
    args.dataset_3d_pc.statistics_info_fn = pre_args.statistics_info_fn
    args.dataset_3d_pc.single_inst = pre_args.single_inst
    args.training.setting = pre_args.training_setting
    args.sampling.use_t = pre_args.use_t
    args.training.batch_size = pre_args.batch_size
    args.training.use_jointspace_seq = pre_args.training_use_jointspace_seq
    args.training.diff_task_translations = pre_args.diff_task_translations
    args.training.diff_task_space = pre_args.diff_task_space
    args.training.kine_diff = pre_args.kine_diff # kine diff #
    args.training.concat_two_dims = pre_args.concat_two_dims
    args.training.tracking_ctl_diff = pre_args.tracking_ctl_diff
    # target_grab_inst_tag: ''
#   target_grab_inst_opt_fn: ''
    args.sampling.target_grab_inst_tag = pre_args.target_grab_inst_tag
    args.sampling.target_grab_inst_opt_fn = pre_args.target_grab_inst_opt_fn
    args.training.AE_Diff = pre_args.AE_Diff
    args.training.train_AE = pre_args.train_AE
    args.training.train_Diff = pre_args.train_Diff
    args.training.cond_diff_allparams = pre_args.cond_diff_allparams
    args.training.succ_rew_threshold = pre_args.succ_rew_threshold
    args.training.slicing_data = pre_args.slicing_data
    args.training.slicing_ws = pre_args.slicing_ws
    args.training.grab_inst_tag_to_opt_stat_fn = pre_args.grab_inst_tag_to_opt_stat_fn
    args.training.grab_inst_tag_to_optimized_res_fn = pre_args.grab_inst_tag_to_optimized_res_fn
    
    args.dataset_3d_pc.multi_inst = pre_args.multi_inst
    args.dataset_3d_pc.sim_platform = pre_args.sim_platform
    
    args.training.task_cond_type = pre_args.task_cond_type
    args.training.debug = pre_args.debug
    
    args.debug = pre_args.debug
    if len(pre_args.exp_tag) > 0:
        args.exp_tag = pre_args.exp_tag
        
    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    else:
        os.makedirs(args.save_dir, exist_ok=True) 
        exp_tag = args.exp_tag
        args.save_dir = os.path.join(args.save_dir, exp_tag)
        os.makedirs(args.save_dir, exist_ok=True)
        


    # shutil.copyfile(src, dst)
    config_path = "cfgs/config.yaml"
    dst_config_folder = args.save_dir
    shutil.copy(config_path, dst_config_folder)
    
    #### TODO: do not create the datasets --- do not load all training data --- but just load statistics realted files to the dataset ###
    print("creating data loader...")
    if args.dataset_3d_pc.data_tag == "v6":
        data = get_dataset_loader_3d_v6_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    elif args.dataset_3d_pc.data_tag == "v7":
        data = get_dataset_loader_3d_v7_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)        
    elif args.dataset_3d_pc.data_tag == "v5":
        print(f"getting dataset for model with arch: {args.model.model_arch}")
        data = get_dataset_loader_3d_v5_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    elif args.model.model_arch == "transformer_v2":
        print(f"getting dataset for model with arch: {args.model.model_arch}")
        data = get_dataset_loader_3d_v3_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    else:
        data = get_dataset_loader_3d_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)

    print("creating model and diffusion...")
    
    # model, diffusion = create_model_and_diffusion(args, data)
    model, diffusion = create_model_and_diffusion_3d_pc(args)
    model.cuda()
    
    
    
    
    

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

    ige_env_cls = isaacgym_task_map[cfg.task_name]
    dict_cls = ige_env_cls.dict_obs_cls if hasattr(ige_env_cls, 'dict_obs_cls') and ige_env_cls.dict_obs_cls else False

    if dict_cls:
        
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
        if global_rank == 0:
            # initialize wandb only once per multi-gpu run
            wandb_observer = WandbAlgoObserver(cfg)
            observers.append(wandb_observer)
    
    # register new AMP network builder and agent #
    def build_runner(algo_observer):
        runner = Runner(algo_observer)
        runner.algo_factory.register_builder('humanoid_amp', lambda **kwargs : amp_continuous.AMPAgent(**kwargs))
        runner.player_factory.register_builder('humanoid_amp', lambda **kwargs : amp_players.AMPPlayerContinuous(**kwargs))
        model_builder.register_model('humanoid_amp', lambda network, **kwargs : amp_models.ModelAMPContinuous(network))
        model_builder.register_network('humanoid_amp', lambda **kwargs : amp_network_builder.AMPBuilder())

        
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

    # convert CLI arguments into dictionary
    # create runner and set the settings
    runner = build_runner(MultiObserver(observers))
    runner.load(rlg_config_dict)
    runner.reset()

    ## TODO: if we are in the test mode; set the random_time config to False #
    exp_logging_dir = os.path.join('runs', cfg.train.params.config.name + '_{date:%d-%H-%M-%S}'.format(date=datetime.now()))
    
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

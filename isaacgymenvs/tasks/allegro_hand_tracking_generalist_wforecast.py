# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
from unittest import TextTestRunner
import xxlimited
from matplotlib.pyplot import axis
import numpy as np
import os
import os.path as osp
import random

from pyparsing import And
import torch

from utils.torch_jit_utils import *
from utils.data_info import plane2euler
# from tasks.hand_base.base_task import BaseTask
from isaacgymenvs.tasks.vec_task import VecTask as BaseTask
from isaacgym import gymtorch
from isaacgym import gymapi

from scipy.spatial.transform import Rotation as R
import trimesh
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples
from copy import deepcopy

# the forcasting model #
# from softzoo.diffusion.diffusion.resample import create_named_schedule_sampler
# from softzoo.diffusion.models.transformer_model import Transformer_Net_PC_Seq_V3_KineDiff_AE_V3, Transformer_Net_PC_Seq_V3_KineDiff_AE_V4, Transformer_Net_PC_Seq_V3_KineDiff_AE_V6
# from softzoo.diffusion.diffusion import gaussian_diffusion_3d_pc as gd_3d_pc
# from softzoo.diffusion.diffusion.respace import space_timesteps, SpacedDiffusion3DPC # as SpacedDiffusion
# SpacedDiffusion_OursV5
import torch.nn as nn
from torch.optim import AdamW

try:
    import clip
except:
    pass
import torch




MASK_HAND = 0
MASK_OBJ = 1
MASK_HAND_RNDIDX = 2

COND_HAND_OBJ = 0
COND_PARTIALHAND_OBJ = 2
COND_OBJ = 1

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


# change the action dim 
# change observation dicts
# 

class AllegroHandTrackingGeneralistWForecasting(BaseTask):
    # def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
    #              agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, exp_logging_dir=None):
        self.cfg = cfg
        
        self.w_forecasting = True
        # self.test = self.cfg['task']['test']
        # self.sim_params = sim_params
        # self.physics_engine = physics_engine
        self.agent_index = agent_index
        self.is_multi_agent = is_multi_agent
        self.test = self.cfg['env']['test']
        self.randomize = self.cfg["task"]["randomize"]
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]
        self.dist_reward_scale = self.cfg["env"]["distRewardScale"]
        self.rot_reward_scale = self.cfg["env"]["rotRewardScale"]
        self.action_penalty_scale = self.cfg["env"]["actionPenaltyScale"]
        self.success_tolerance = self.cfg["env"]["successTolerance"]
        self.reach_goal_bonus = self.cfg["env"]["reachGoalBonus"]
        self.fall_dist = self.cfg["env"]["fallDistance"]
        self.fall_penalty = self.cfg["env"]["fallPenalty"]
        self.rot_eps = self.cfg["env"]["rotEps"]
        self.vel_obs_scale = 0.2  # scale factor of velocity based observations
        self.force_torque_obs_scale = 10.0  # scale factor of velocity based observations
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        
        
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.max_episode_length = self.cfg["env"]["episodeLength"]
        self.reset_time = self.cfg["env"].get("resetTime", -1.0)
        self.print_success_stat = self.cfg["env"]["printNumSuccesses"]
        self.max_consecutive_successes = self.cfg["env"]["maxConsecutiveSuccesses"]
        self.w_obj_ornt = self.cfg["env"]["w_obj_ornt"]
        self.w_obj_vels = self.cfg["env"]["w_obj_vels"]
        self.av_factor = self.cfg["env"].get("averFactor", 0.01)
        print("Averaging factor: ", self.av_factor)
        self.rl_device = rl_device
        self.exp_logging_dir = exp_logging_dir
        
        if self.exp_logging_dir is None:
            self.exp_logging_dir = self.cfg['env']['exp_logging_dir']
        
        self.object_name = self.cfg["env"]["object_name"] #
        
        
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.mocap_sv_info_fn = self.cfg["env"]["mocap_sv_info_fn"]
        
        if 'taco_' in self.object_name and 'TACO' in self.mocap_sv_info_fn and 'ori_grab' not in self.mocap_sv_info_fn:
            self.dataset_type = 'taco'
        elif 'grab' in self.object_name or 'GRAB' in self.mocap_sv_info_fn or 'ori_grab' in self.mocap_sv_info_fn:
            self.dataset_type = 'grab'
        else:
            raise ValueError(f"Unknown dataset type for object: {self.object_name}")

        
        
        self.use_kinematics_bias = self.cfg['env']['use_kinematics_bias']
        self.kinematics_only = self.cfg['env']['kinematics_only']
        self.use_kinematics_bias_wdelta = self.cfg['env']['use_kinematics_bias_wdelta']
        # #
        
        self.use_canonical_state = self.cfg['env']['use_canonical_state']
        self.separate_stages = self.cfg['env']['separate_stages']
        self.use_unified_canonical_state = self.cfg['env']['use_unified_canonical_state']
        
        self.rigid_obj_density = self.cfg['env']['rigid_obj_density']
        # self.density = self.cfg["env"]["rigid_obj_density"]
        self.use_fingertips = self.cfg["env"]["use_fingertips"]
        self.glb_trans_vel_scale = self.cfg["env"]["glb_trans_vel_scale"]
        self.glb_rot_vel_scale = self.cfg["env"]["glb_rot_vel_scale"]
        self.tight_obs = self.cfg["env"]["tight_obs"]
        # hand_pose_guidance_glb_trans_coef, hand_pose_guidance_glb_rot_coef, hand_pose_guidance_fingerpose_coef #
        self.hand_pose_guidance_glb_trans_coef = self.cfg["env"]["hand_pose_guidance_glb_trans_coef"]
        self.hand_pose_guidance_glb_rot_coef = self.cfg["env"]["hand_pose_guidance_glb_rot_coef"]
        self.hand_pose_guidance_fingerpose_coef = self.cfg["env"]["hand_pose_guidance_fingerpose_coef"]
        self.rew_finger_obj_dist_coef = self.cfg["env"]["rew_finger_obj_dist_coef"]
        self.rew_delta_hand_pose_coef = self.cfg["env"]["rew_delta_hand_pose_coef"]
        self.use_real_twostage_rew = self.cfg["env"]["use_real_twostage_rew"]
        self.start_grasping_fr = self.cfg["env"]["start_grasping_fr"]
        self.start_frame = self.cfg['env']['start_frame']
        self.rew_obj_pose_coef = self.cfg['env']['rew_obj_pose_coef']
        self.object_feat_dim = self.cfg['env']['object_feat_dim']
        self.use_hand_actions_rew = self.cfg['env']['use_hand_actions_rew']
        self.supervised_training = self.cfg['env'].get('supervised_training', False)
        # test_inst_tag, test_optimized_res
        self.test_inst_tag = self.cfg['env'].get('test_inst_tag', '')
        self.test_optimized_res = self.cfg['env'].get('test_optimized_res', '')
        # test optimized res #
        self.use_local_canonical_state = self.cfg['env'].get('use_local_canonical_state', False)
        self.obj_type_to_base_traj_fn = self.cfg['env'].get('obj_type_to_base_traj_fn', '')
        # rew_thres_with_selected_insts, selected_inst_idxes_dict
        self.rew_thres_with_selected_insts = self.cfg['env'].get('rew_thres_with_selected_insts', False)
        self.selected_inst_idxes_dict = self.cfg['env'].get('selected_inst_idxes_dict', '')
        self.train_on_all_trajs = self.cfg['env'].get('train_on_all_trajs', False )
        self.wo_vel_obs = self.cfg['env'].get('wo_vel_obs', False)
        
        self.use_history_obs = self.cfg['env'].get('use_history_obs', False)
        self.history_length = self.cfg['env'].get('history_length', 5) # 5-frame history #
        self.w_franka = self.cfg['env'].get('w_franka', 5) 
        self.early_terminate = self.cfg['env'].get('early_terminate', False)
        self.use_future_obs = self.cfg['env'].get('use_future_obs', False)
        self.rl_tracking_targets = self.cfg['env'].get('rl_tracking_targets', False)
        
        # with #
        self.sv_info_during_training = self.cfg['env'].get('sv_info_during_training', False)
        
        # with history windowjindex # # with 
        # with history window index # # # # # history window index? # window index #
        # w_history_window_index # # with history window index # window index #
        self.w_history_window_index = self.cfg['env'].get('w_history_window_index', False)
        
        # if we randomize the conditions # keyframes, sparse joints # 
        ## TODO: 1) train with randomized masks, 2) train with randomized keyframes, 3) train with the contact conditions --- but how to get such contact conditions? #
        ## TODO: 1) maks them with strategies? #
        self.randomize_conditions = self.cfg['env'].get('randomize_conditions', False)
        self.w_inst_latent_features = self.cfg['env'].get('w_inst_latent_features', False)
        
        self.masked_mimic_training = self.cfg['env'].get('masked_mimic_training', False)
        
        
        self.randomize_condition_type = self.cfg['env'].get('randomize_condition_type', False)
        
        self.add_contact_conditions = self.cfg['env'].get('add_contact_conditions', False)
        self.contact_info_sv_root = self.cfg['env'].get('contact_info_sv_root', '')
        # history_window_size, glb_feat_per_skip, centralize_info #
        self.history_window_size = self.cfg['env'].get('history_window_size', 30)
        self.glb_feat_per_skip = self.cfg['env'].get('glb_feat_per_skip', 1)
        self.centralize_info = self.cfg['env'].get('centralize_info', False)
        self.using_forcast_res_step_threshold = 60
        self.forecast_future_freq = self.cfg['env'].get('forecast_future_freq', 1)
        
        self.hist_cond_partial_hand_info = self.cfg['env'].get('hist_cond_partial_hand_info', False)
        self.hist_cond_partial_obj_info = self.cfg['env'].get('hist_cond_partial_obj_info', False)
        self.hist_cond_partial_obj_pos_info = self.cfg['env'].get('hist_cond_partial_obj_pos_info', False)
        self.include_obj_rot_in_obs = self.cfg['env'].get('include_obj_rot_in_obs', False)
        
        
        
        # step 1: add random shift cond and check the performance?
        # step 2: add the random shift cond freq -- which is used to change the goal target frequencies and check the performance #
        self.random_shift_cond = self.cfg['env'].get('random_shift_cond', False)
        # # object reward and the hand dofs reward # #
        self.random_shift_cond_freq = self.cfg['env'].get('random_shift_cond_freq', False)
        self.maxx_inv_cond_freq = self.cfg['env'].get('maxx_inv_cond_freq', 30)
        
        self.only_use_hand_first_frame = self.cfg['env'].get('only_use_hand_first_frame', False)
        
        self.forecasting_model_inv_freq = self.cfg['env'].get('forecasting_model_inv_freq', 1)
        
        
        self.forward_forecasting_nn_steps = self.cfg['env'].get('forward_forecasting_nn_steps', 10)
        self.forward_forecasting_nn_steps = 2
        
        self.use_clip_glb_features = self.cfg['env'].get('use_clip_glb_features', False)
        
        self.forecast_obj_pos = self.cfg['env'].get('forecast_obj_pos', False)
        
        self.use_multi_step_control = self.cfg['env'].get('use_multi_step_control', False)
        self.distill_via_bc = self.cfg['env'].get('distill_via_bc', False)
        
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        # model, preprocess = clip.load("ViT-B/32", device=device)
        # self.
        
        # if self.use_clip_glb_features:
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.rl_device)
        
        self.ref_ts = 0 
        
        # right_hand_dist_thres # pre optimized traj #
        try:
            self.pre_optimized_traj = self.cfg['env']['pre_optimized_traj'] # pre optimized traj #
        except:
            
            self.pre_optimized_traj = None
        
        #### right_hand_dist_thres #### ## thres ##
        try:
            self.right_hand_dist_thres = self.cfg['env']['right_hand_dist_thres']
        except:
            # right_hand_dist_thres #
            self.right_hand_dist_thres = 0.12
        
        print(f"right_hand_dist_thres: {self.right_hand_dist_thres}")
        
        
        try:
            self.ground_distance = self.cfg['env']['ground_distance']
        except:
            self.ground_distance = 0.0
            
        try:
            self.disable_obj_gravity = self.cfg['env']['disable_obj_gravity']
        except:
            self.disable_obj_gravity = False
        
        try:
            self.use_twostage_rew = self.cfg['env']['use_twostage_rew']
        except:
            self.use_twostage_rew = False

        try:
            self.goal_dist_thres = self.cfg['env']['goal_dist_thres']
        except:
            self.goal_dist_thres = 0.0
            
        # lifting_separate_stages # lifting separate stages #
        
        try : 
            self.lifting_separate_stages = self.cfg['env']['lifting_separate_stages']
        except:
            self.lifting_separate_stages = False
        
        try : 
            self.strict_lifting_separate_stages = self.cfg['env']['strict_lifting_separate_stages']
        except:
            self.strict_lifting_separate_stages = False
        try:
            self.add_table = self.cfg['env']['add_table']
        except:
            self.add_table = False
        
        try: # table z dim #
            self.table_z_dim = self.cfg['env']['table_z_dim']
        except:
            self.table_z_dim = 0.0
            
        try:
            self.use_taco_obj_traj = self.cfg['env']['use_taco_obj_traj']
        except:
            self.use_taco_obj_traj = False
        
        
        try:
            self.hand_type = self.cfg["env"]["hand_type"]
        except:
            self.hand_type = "allegro"
        
        
        
        self.test_subj_nm = self.cfg['env'].get('test_subj_nm', '')
        
        ### base task ###
        ### TODO: add such stats fn; trained models ###### trained models ###
        ### NOTE: the hand tracking generalist task is used for the generalist tracking task ###
        self.grab_inst_tag_to_opt_stat_fn = self.cfg['env']['grab_inst_tag_to_opt_stat_fn']
        self.grab_inst_tag_to_optimized_res_fn = self.cfg['env']['grab_inst_tag_to_optimized_res_fn']
        self.taco_inst_tag_to_optimized_res_fn = self.cfg['env']['taco_inst_tag_to_optimized_res_fn']
        self.object_type_to_latent_feature_fn = self.cfg['env']['object_type_to_latent_feature_fn'] # # object_type_to_latent_featu #
        
        self.inst_tag_to_latent_feature_fn = self.cfg['env'].get('inst_tag_to_latent_feature_fn', '') #
        self.use_inst_latent_features = len(self.inst_tag_to_latent_feature_fn) > 0 #
        # /cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy #
        self.tracking_save_info_fn = self.cfg['env']['tracking_save_info_fn'] #
        self.single_instance_state_based_test = self.cfg['env'].get('single_instance_state_based_test', False) #
        # load the samples with object code? #
        self.sampleds_with_object_code_fn = self.cfg['env'].get('sampleds_with_object_code_fn', '')
        
        self.grab_obj_type_to_opt_res_fn = self.cfg['env'].get('grab_obj_type_to_opt_res_fn', '')
        self.taco_obj_type_to_opt_res_fn = self.cfg['env'].get('taco_obj_type_to_opt_res_fn', '')
        
        self.only_training_on_succ_samples = self.cfg['env'].get('only_training_on_succ_samples', False)
        self.grab_train_test_setting = self.cfg['env'].get('grab_train_test_setting', False)
        
        self.maxx_inst_nn = self.cfg['env'].get('maxx_inst_nn', 10000)
        
        self.tuning_single_instance = True if self.maxx_inst_nn == 1 else False

        
        # self.tuning_single_instance 
        
        self.last_step = -1
        self.last_rand_step = -1
        self.first_randomization = True
        
        
        # tracking info st tag # # whole hierarchical space # and a universal conditional sapce #
        
        # self.tracking_info_st_tag = "passive_active_info_"
        self.tracking_info_st_tag = self.cfg['env'].get('tracking_info_st_tag', 'passive_active_info_')
        self.use_strict_maxx_nn_ts = self.cfg['env'].get('use_strict_maxx_nn_ts', False) # 
        self.strict_maxx_nn_ts = self.cfg['env'].get('strict_maxx_nn_ts', 150)
        self.taco_interped_data_sv_additional_tag = self.cfg['env'].get('taco_interped_data_sv_additional_tag', False)
        # rew_grab_thres: 50.0
        # rew_taco_thres: 200.0
        self.rew_grab_thres = self.cfg['env'].get('rew_grab_thres', 50.0)
        self.rew_taco_thres = self.cfg['env'].get('rew_taco_thres', 200.0)
        self.rew_smoothness_coef = self.cfg['env'].get('rew_smoothness_coef', 0.0)
        self.use_base_traj = self.cfg['env'].get('use_base_traj', False) # use_base_traj #
        self.base_traj = self.cfg['env'].get('base_traj', '')
        
        # customize_global_damping and the customized damping #
        self.customize_damping = self.cfg['env'].get('customize_damping', False)
        self.customize_global_damping = self.cfg['env'].get('customize_global_damping', False)
        
        self.single_instance_state_based_train = self.cfg['env'].get('single_instance_state_based_train', False)
        self.data_selection_ratio = self.cfg['env'].get('data_selection_ratio', 1.0)
        
        self.downsample = self.cfg['env'].get('downsample', False)
        self.target_inst_tag_list_fn = self.cfg['env'].get('target_inst_tag_list_fn', '')
        
        ######### Single instance training settings  #########
        self.single_instance_training = False
        ######### Single instance training settings  #########
        
        
        
        ########## Teacher model settings ##########
        self.use_teacher_model = self.cfg['env'].get('use_teacher_model', False)
        self.teacher_model_inst_tags_fn = self.cfg['env'].get('teacher_model_inst_tags_fn', '')
        self.teacher_index_to_inst_tags_fn = self.cfg['env'].get('teacher_index_to_inst_tags', '')
        self.good_inst_opt_res = self.cfg['env'].get('good_inst_opt_res', '')
        ########## Teacher model settings ##########
        
        ########## (forcasting target reference trajectories) window selection settings ##########
        self.use_window_future_selection = self.cfg['env'].get('use_window_future_selection', False)
        ########## (forcasting target reference trajectories) window selection settings ##########
        
        ########## Forcasting model setting ##########
        self.use_forcasting_model = self.cfg['env'].get('use_forcasting_model', False)
        self.forcasting_model_weights = self.cfg['env'].get('forcasting_model_weights', '')
        self.forcasting_model_n_layers = self.cfg['env'].get('forcasting_model_n_layers', 7)
        self.w_glb_traj_feat_cond = self.cfg['env'].get('w_glb_traj_feat_cond', False)
        # self.forcasting_history_ws = 30
        self.forcasting_history_ws = self.cfg['env'].get('forcasting_history_ws', 30)
        self.forcasting_diffusion_model = self.cfg['env'].get('forcasting_diffusion_model', False)
        self.forcasting_model_training = self.cfg['env'].get('forcasting_model_training', False)
        self.forcasting_model_lr = self.cfg['env'].get('forcasting_model_lr', 1e-4)
        self.forcasting_model_weight_decay = self.cfg['env'].get('forcasting_model_weight_decay', 5e-5)
        ########### Whether to add the partial obj info and partial hand info ###########
        self.partial_obj_info = self.cfg['env'].get('partial_obj_info', False)
        self.partial_hand_info = self.cfg['env'].get('partial_hand_info', False)
        self.partial_obj_pos_info = self.cfg['env'].get('partial_obj_pos_info', False)
        ########### Whether to add the partial obj info and partial hand info ###########
        
        self.use_partial_to_complete_model = self.cfg['env'].get('use_partial_to_complete_model', False)
        self.partial_to_complete_model_weights = self.cfg['env'].get('partial_to_complete_model_weights', '') 
        
        ##### Conditioning setting --- whether to use the start-end state conditions #####
        self.st_ed_state_cond = self.cfg['env'].get('st_ed_state_cond', False)
        ##### Conditioning setting --- whether to use the start-end state conditions #####
        
        self.preset_cond_type = self.cfg['env'].get('preset_cond_type', 0)
        self.preset_inv_cond_freq = self.cfg['env'].get('preset_inv_cond_freq', 1)
        
        
        
        # getthe forcasting model #
        if self.use_forcasting_model: 
            self._load_forcasting_model()
            
        if self.use_partial_to_complete_model:
            self._load_partial_to_complete_model()
        
        self.forcasting_inv_freq = self.cfg['env'].get('forcasting_inv_freq', 1)
        # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot #
        self.already_forcasted = False 
        # self.forcasting_diffusion_model = self.cfg['env'].get('forcasting_diffusion_model', False)
        ########## Forcasting model setting ##########
        
        self.single_inst_tag = self.cfg['env'].get('single_inst_tag', '')
        
        
        ########## Add system parameters ###########
        self.stiffness_coef = self.cfg['env'].get('stiffness_coef', 100)
        self.damping_coef = self.cfg['env'].get('damping_coef', 4.0)
        self.effort_coef = self.cfg['env'].get('effort_coef', 0.95)
        ########## Add system parameters ###########
        
        ########## The impedance control setting #########
        # TODO: tune the sitffness low and the high range values #
        # impedance_stiffness_low, impedance_stiffness_high, w_impedance_bias_control #
        self.impedance_stiffness_low = self.cfg['env'].get('impedance_stiffness_low', 1.0)
        self.impedance_stiffness_high = self.cfg['env'].get('impedance_stiffness_high', 50.0)
        # impedance stiffness low and the impedance stiffness high # 
        self.w_impedance_bias_control = self.cfg['env'].get('w_impedance_bias_control', False)
        ########### The impedance control setting ###########
        
        self.comput_reward_traj_hand_qpos = self.cfg['env'].get('comput_reward_traj_hand_qpos', False)
        
        
        ########### Whether to add object  ###########
        self.w_obj_latent_features = self.cfg['env'].get('w_obj_latent_features', True)
        self.history_freq = self.cfg['env'].get('history_freq', 1)
        
        self.activate_forecaster = self.cfg['env'].get('activate_forecaster', True)
        
        self.use_future_ref_as_obs_goal = self.cfg['env'].get('use_future_ref_as_obs_goal', False)
        
        
        self.forecasting_obs_with_original_obs = self.cfg['env'].get('forecasting_obs_with_original_obs', False )
        
        # load some tracking trajectories and also save some trajectories for training 
        # load the tracking trajectories and use them as the tracking targets 
        
        # only when single instanc estate based train is activated would we use the tracking trajs as the tracking
        
        
        
        # use_multiple_kine_source_trajs, multiple_kine_source_trajs_fn
        self.use_multiple_kine_source_trajs = self.cfg['env'].get('use_multiple_kine_source_trajs', False)
        self.multiple_kine_source_trajs_fn = self.cfg['env'].get('multiple_kine_source_trajs_fn', '')
        self.compute_hand_rew_buf_threshold = self.cfg['env'].get('compute_hand_rew_buf_threshold', 500)
        
        #### orientation reward coeficient scheduling ####
        self.schedule_ornt_rew_coef = self.cfg['env'].get('schedule_ornt_rew_coef', False)
        self.lowest_ornt_rew_coef = self.cfg['env'].get('lowest_ornt_rew_coef', 0.03)
        self.highest_ornt_rew_coef = self.cfg['env'].get('highest_ornt_rew_coef', 0.33)
        self.ornt_rew_coef_warm_starting_steps = self.cfg['env'].get('ornt_rew_coef_warm_starting_steps', 100)
        self.ornt_rew_coef_increasing_steps = self.cfg['env'].get('ornt_rew_coef_increasing_steps', 200)
        self.cur_ornt_rew_coef = 0.03
        self.rew_env_reset_nn = 0
        #### orientation reward coeficient scheduling ####
        
        
        #### add the global mult factor scaling setting ####
        # hand_glb_mult_factor_scaling_coef, hand_glb_mult_scaling_progress_after
        self.hand_glb_mult_factor_scaling_coef = self.cfg['env'].get('hand_glb_mult_factor_scaling_coef', 1.0)
        self.hand_glb_mult_scaling_progress_after = self.cfg['env'].get('hand_glb_mult_scaling_progress_after', 600)
        #### add the global mult factor scaling setting ####
        
        ### student model distillation settings ###
        self.train_student_model = self.cfg['env'].get('train_student_model', False)
        self.ts_teacher_model_obs_dim = self.cfg['env'].get('ts_teacher_model_obs_dim', 731)
        ### student model distillation settings ###
        print(f"[INFO] train_student_model: {self.train_student_model}")
        
        
        self.already_predicted_targets = False
        
        self.num_environments = self.cfg["env"]["numEnvs"]
        # 
        
        # use_multiple_kine_source_trajs #
        # if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
        #     self.multiple_kine_source_trajs = np.load(self.multiple_kine_source_trajs_fn, allow_pickle=True).item() # multiple kine source  trajs; multiple kine source trajs 
        #     self.multiple_kine_source_trajs = torch.from_numpy(self.multiple_kine_source_trajs).float().to(self.rl_device) # nn_trajs x nn_hand_dofs 
        #     self.envs_kine_source_trajs_idxes = torch.tensor(
        #         [0] * self.num_envs, dtype=torch.long
        #     ).to(self.device) # (nn_envs, ) #
        #     # multiple kine source trajs #
        
        # 
        # impedance stiffness low and impedance stiffness high #
        # impedance_stiffness_low, impedance_stiffness_high, w_impedance_bias_control
        # prev_targets_impedance, cur_targets_impedance, prev_delta_targets_impedance, cur_delta_targets_impedance
        
        
        
        ######### Loading the teacher model #########
        self.use_multiple_teacher_model = False
        if self.use_teacher_model and len(self.teacher_index_to_inst_tags_fn) > 0 and os.path.exists(self.teacher_index_to_inst_tags_fn):
            self.teacher_index_to_inst_tags = np.load(self.teacher_index_to_inst_tags_fn, allow_pickle=True).item()
            self.teacher_index_to_inst_tags_dict = {}
            for teacher_index in self.teacher_index_to_inst_tags:
                cur_inst_tags_fn = self.teacher_index_to_inst_tags[teacher_index] 
                # curi nst tags #
                cur_inst_tags_dict = np.load(cur_inst_tags_fn, allow_pickle=True).item()
                self.teacher_index_to_inst_tags_dict[teacher_index] = cur_inst_tags_dict
            self.use_multiple_teacher_model = True
            self.cur_teacher_index = 0
            self.teacher_model_idx = 0
            self.nn_teachers = len(self.teacher_index_to_inst_tags)
            self.tot_reset_nn = 0
            self.change_teacher_freq = 10
        ######### Teacher index to instance tags dict #########
        
        
        if len(self.good_inst_opt_res) > 0 and os.path.exists(self.good_inst_opt_res):
            self.good_inst_opt_res = np.load(self.good_inst_opt_res, allow_pickle=True).item()
        else:
            self.good_inst_opt_res = None
        
        
        self.maxx_kine_nn_ts = 150 
        if len(self.grab_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.grab_inst_tag_to_optimized_res_fn):
            if '_nf_300' in self.grab_inst_tag_to_optimized_res_fn:
                self.maxx_kine_nn_ts = 300
        
        ### make the maximum kienmatics nn ts ###
        # self.maxx_kine_nn_ts = 1000 # maximum kinematics nn ts ---- maximum lenght of the kinematic sequence #
        self.maxx_nn_pts = 512
        if len(self.taco_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.taco_inst_tag_to_optimized_res_fn):
            self.maxx_kine_nn_ts = 1000
            
        if self.use_strict_maxx_nn_ts:
            # self.maxx_kine_nn_ts = 150
            self.maxx_kine_nn_ts = self.strict_maxx_nn_ts
            
        self.glb_rot_use_quat = False



        if len(self.grab_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.grab_inst_tag_to_optimized_res_fn):
            if len(self.taco_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.taco_inst_tag_to_optimized_res_fn):
                self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn, self.taco_inst_tag_to_optimized_res_fn] 
                # self.grab_inst_tag_to_optimized_res_fn = [self.taco_inst_tag_to_optimized_res_fn] 
                # get the grab and taco inst tagt to optimized res fn ##
            else:
                self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn]
        else:
            self.grab_inst_tag_to_optimized_res_fn = [self.taco_inst_tag_to_optimized_res_fn]
            
        if len(self.obj_type_to_base_traj_fn) > 0 and os.path.exists(self.obj_type_to_base_traj_fn):
            self.obj_type_to_base_traj = np.load(self.obj_type_to_base_traj_fn, allow_pickle=True).item()
        else:
            self.obj_type_to_base_traj =None
        
        
        print(f"grab_inst_tag_to_optimized_res_fn: {self.grab_inst_tag_to_optimized_res_fn}")
        
        
        self.tot_grab_inst_tag_to_opt_res = {}
        for cur_fn in self.grab_inst_tag_to_optimized_res_fn: # test grab inst tag to opt res #
            cur_opt_res = np.load(cur_fn, allow_pickle=True).item()
            self.tot_grab_inst_tag_to_opt_res.update(cur_opt_res)
        self.grab_inst_tag_to_opt_res = self.tot_grab_inst_tag_to_opt_res
        
        
        
        ## NOTE: get all grab instance tags ##
        def get_all_grab_tags():
            # tracking_save_info_fn
            tracking_save_info_fn = self.tracking_save_info_fn
            # tracking_info_st_tag # 
            # for all the data train them on that # 
            tot_fns = os.listdir(tracking_save_info_fn)
            tot_fns = [cur_fn for cur_fn in tot_fns if self.tracking_info_st_tag in cur_fn and cur_fn.endswith('.npy')]
            tot_grab_inst_tags = []
            for cur_fn in tot_fns:
                if '_taco_' in cur_fn:
                    continue
                cur_grab_inst_tag = cur_fn.split(".")[0][len(self.tracking_info_st_tag): ]
                
                if self.grab_train_test_setting and self.maxx_inst_nn > 1: # judge whether the modle is in the training split mode 
                    if '_s1_' in cur_grab_inst_tag: # skip the test split instances # very difficult cases ## ---- using cups and wineglasses #
                        continue
                
                # for cur_obj_nm in ['mug', 'cup', 'wineglass']:
                #     if cur_obj_nm 
                if 'mug' in cur_grab_inst_tag or 'cup' in cur_grab_inst_tag or 'wineglass' in cur_grab_inst_tag:
                    continue
                
                cur_grab_mesh_fn = cur_grab_inst_tag.split("_nf_")[0]
                cur_grab_mesh_fn = os.path.join("/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/datasetv4.1/sem", cur_grab_mesh_fn)
                if not os.path.exists(cur_grab_mesh_fn):
                    continue
                
                tot_grab_inst_tags.append(cur_grab_inst_tag)
            return tot_grab_inst_tags
        
        tot_grab_inst_tag = get_all_grab_tags()
        for cur_inst_tag in tot_grab_inst_tag:
            cur_inst_tag_tuple = (cur_inst_tag, cur_inst_tag)
            if cur_inst_tag_tuple not in self.grab_inst_tag_to_opt_res:
                self.grab_inst_tag_to_opt_res[cur_inst_tag_tuple] = ['']
        
        
        # 
        if len(self.target_inst_tag_list_fn) > 0 and os.path.exists(self.target_inst_tag_list_fn):
            print(f"Loading target_inst_tag_list from: {self.target_inst_tag_list_fn}")
            self.target_inst_tag_list = np.load(self.target_inst_tag_list_fn, allow_pickle=True).item()
            self.grab_inst_tag_to_opt_res_new = {}
            for cur_inst_tag in self.grab_inst_tag_to_opt_res:
                if cur_inst_tag[0] not in self.target_inst_tag_list:
                    continue
                self.grab_inst_tag_to_opt_res_new[cur_inst_tag] = self.grab_inst_tag_to_opt_res[cur_inst_tag]
            self.grab_inst_tag_to_opt_res = self.grab_inst_tag_to_opt_res_new
            
        # self.use_teacher_model, self.teacher_model_inst_tags_fn #
        if self.use_teacher_model:
            assert len(self.teacher_model_inst_tags_fn) > 0 and os.path.exists(self.teacher_model_inst_tags_fn)
            self.teacher_model_inst_tags = np.load(self.teacher_model_inst_tags_fn, allow_pickle=True).item()
            
        
        # print(f"[2] loaded grab_inst_tag_to_opt_res: {self.grab_inst_tag_to_opt_res}")
        
        
        
        self.grab_inst_tag_to_best_opt_res = {}
        for cur_fn in self.grab_inst_tag_to_optimized_res_fn:
            cur_best_opt_res_all_fn = cur_fn.replace('data_inst_tag_to_optimized_res.npy', 'data_inst_tag_to_best_opt_res_all.npy')
            # cur best opt res all fn #
            if not os.path.exists(cur_best_opt_res_all_fn):
                continue
            cur_best_opt_res_all = np.load(cur_best_opt_res_all_fn, allow_pickle=True).item()
            self.grab_inst_tag_to_best_opt_res.update(cur_best_opt_res_all  )
        
        
        # supervision strategy is strange #
        ### Load and use the obj type to opt res to filter out unsuccessful trajectories ###
        if len(self.grab_obj_type_to_opt_res_fn) > 0 and os.path.exists(self.grab_obj_type_to_opt_res_fn):
            self.grab_obj_type_to_opt_res = np.load(self.grab_obj_type_to_opt_res_fn, allow_pickle=True).item()
        else:
            self.grab_obj_type_to_opt_res = None
        
        
        if len(self.taco_obj_type_to_opt_res_fn) > 0 and os.path.exists(self.taco_obj_type_to_opt_res_fn):
            self.taco_obj_type_to_opt_res = np.load(self.taco_obj_type_to_opt_res_fn, allow_pickle=True).item()
        else:
            self.taco_obj_type_to_opt_res = None ## else we do no have the 
            
        # self.use_teacher_model, self.teacher_model_inst_tags_fn #


        print(f"Loaded inst_tag_to_optimized_res with number of total instances {len(self.grab_inst_tag_to_opt_res)}")
        # load the inst tag to optimized res # load the inst tag #
        
        
        
        # rew_thres_with_selected_insts, selected_inst_idxes_dict
        if self.rew_thres_with_selected_insts: # rew threshold with 
            print(f"Loading selected_inst_idxes_dict from: {self.selected_inst_idxes_dict}")
            self.selected_inst_idxes_dict = np.load(self.selected_inst_idxes_dict, allow_pickle=True).item()
            sorted_inst_idxes = sorted(self.selected_inst_idxes_dict.items(), key=lambda x: x[1])
            # # 
            maxx_selected_inst_nn = 400
            sorted_inst_idxes = sorted_inst_idxes[:maxx_selected_inst_nn]
            sorted_inst_idxes_dict = { item[0]: item[1] for item in sorted_inst_idxes }
        
        
        
        if self.w_franka:
            joint_idxes_ordering = [_ for _ in range(11)] + [_ + 15 for _ in range(0, 8)] + [11, 12, 13, 14]
            joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
            joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
            self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
            self.joint_idxes_ordering_th = torch.from_numpy(joint_idxes_ordering).long().to(self.rl_device)
            self.inversed_joint_idxes_ordering_th = torch.from_numpy(joint_idxes_inversed_ordering).long().to(self.rl_device)
        else: # inversed joint idxes #
            joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
            joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
            joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
            self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
            self.joint_idxes_ordering_th = torch.from_numpy(joint_idxes_ordering).long().to(self.rl_device)
            self.inversed_joint_idxes_ordering_th = torch.from_numpy(joint_idxes_inversed_ordering).long().to(self.rl_device)
        
        # inversed joint idxes #
        
        # grab_inst_tag_to_opt_stat = np.load(grab_inst_tag_to_opt_stat_fn, allow_pickle=True).item()
        self.data_list  = []
        self.data_inst_tag_list = []
        self.object_code_list = []
        self.object_rew_succ_dict = {}
        self.rew_succ_threshold = self.rew_grab_thres # 50.0 #
        self.taco_rew_succ_threshold = self.rew_taco_thres # 200.0 #
        
        self.data_base_traj = []
        
        # test 
        # single  testing optimized res #
        # single testing instance tag #
        # single testing object type #
        # test_inst_tag, test_optimized_res #
        # preoptimized trajectories --- as the supervision 
        
        if self.use_base_traj:
            if len(self.sampleds_with_object_code_fn) == 0:
                self.sampleds_with_object_code_fn = self.pre_optimized_traj
            print(f"sampleds_with_object_code_fn: {self.sampleds_with_object_code_fn}")
            
        if len(self.base_traj) > 0 and os.path.exists(self.base_traj):
            self.sampleds_with_object_code_fn = self.base_traj
        
        
        # single_test_forbit_tags = ["ori_grab_s8_cylinderlarge_inspect_1_nf_300"]
        # ori_grab_s6_torusmedium_inspect_1
        # single_test_forbit_tags = ["ori_grab_s8_cylinderlarge_inspect_1_nf_300", "ori_grab_s6_torusmedium_inspect_1_nf_300", "ori_grab_s4_train_lift_nf_300", "ori_grab_s10_alarmclock_pass_1_nf_300"]
        single_test_forbit_tags = ["ori_grab_s2_duck_inspect_1_nf_300"]
        
        if len(self.single_inst_tag) > 0:
            single_test_forbit_tags = [self.single_inst_tag]
        single_test_forbit_tags = single_test_forbit_tags[:1]
        
        
        if len(self.test_inst_tag) > 0:
            print(f"test_inst_tag: {self.test_inst_tag}, test_optimized_res: {self.test_optimized_res}")
            single_test_forbit_tags = [self.test_inst_tag]
            self.grab_inst_tag_to_opt_res[(self.test_inst_tag, self.test_inst_tag)] = ['']
        
        
        self.obj_tag_to_teacher_idx = {}
        
         
        if len(self.test_inst_tag) > 0 and len(self.test_optimized_res) > 0 and os.path.exists(self.test_optimized_res):
            self.single_instance_training = True
            
            self.data_list = [  self.test_optimized_res ]
            self.data_inst_tag_list = [ self.test_inst_tag ]
            self.object_code_list = [ self.test_inst_tag ] 
            self.data_base_traj = [ None ]
            
            if self.test_inst_tag.endswith('.npy'):
                test_inst_dict = np.load(self.test_inst_tag, allow_pickle=True).item()
                test_inst_object_type = test_inst_dict['object_type']
                self.object_code_list = [ test_inst_object_type ]
            
            # with obj code file name # # # file name #
            if len(self.sampleds_with_object_code_fn) > 0 and os.path.exists(self.sampleds_with_object_code_fn):
                print(f"loading sampleds_with_object_code_fn: {self.sampleds_with_object_code_fn}")
                self.samples_with_object_code = np.load(self.sampleds_with_object_code_fn, allow_pickle=True).item()
                if 'optimized_hand_qtars' in self.samples_with_object_code:
                    self.tot_dof_targets = self.samples_with_object_code['optimized_hand_qtars']
                    self.tot_dof_targets = self.tot_dof_targets[0]
                    self.maxx_kine_nn_ts = min(self.maxx_kine_nn_ts, self.tot_dof_targets.shape[0])
                    print(f"Loaded optimized hand qtars with shape: {self.tot_dof_targets.shape}")
                elif 'samples' in self.samples_with_object_code:
                    samples = self.samples_with_object_code['samples']
                    if 'data_nm' in samples:
                        samples_data_nm = samples['data_nm']
                        for idx, cur_data_nm in enumerate(samples_data_nm):
                            if cur_data_nm == self.test_inst_tag:
                                break
                        self.tot_dof_targets = samples['hand_qs'][idx]
                    else:
                        self.tot_dof_targets = samples['hand_qs'][0]
                    
                    self.maxx_kine_nn_ts = min(self.maxx_kine_nn_ts, self.tot_dof_targets.shape[0])
                    print(f"Loaded optimized hand qtars with shape: {self.tot_dof_targets.shape}")
                elif 'env_object_codes' in self.samples_with_object_code:
                    env_obj_codes = self.samples_with_object_code['env_object_codes']
                    # pre_optimized_fr_tag = 'ori_grab_s9_waterbottle_pour_1'
                    pre_optimized_fr_tag = self.test_inst_tag
                    for i_env, cur_env_obj_code in enumerate(env_obj_codes):
                        if cur_env_obj_code == pre_optimized_fr_tag:
                            # self.test_obj_code = self.sampleds_with_object_code['object_codes'][i_env]
                            break
                    # print(f"Loading from sampled trajectories for the instance: {cur_env_obj_code}")
                    tot_ts_idxes = [ key for key in self.samples_with_object_code if isinstance(key, int) ]
                    tot_ts_idxes = sorted(tot_ts_idxes) # the sorted indexes for timesteps #
                    tot_dof_targets = []
                    for i_ts, cur_ts in enumerate(tot_ts_idxes):
                        cur_ts_stats = self.samples_with_object_code[cur_ts]
                        cur_ts_dof_tars = cur_ts_stats['shadow_hand_dof_tars']
                        cur_ts_dof_tars = cur_ts_dof_tars[i_env]
                        tot_dof_targets.append( cur_ts_dof_tars )
                    tot_dof_targets = np.stack(tot_dof_targets, axis=0) # nn_ts x nn_evs x nn_han-dof_
                    self.tot_dof_targets = tot_dof_targets
                # elif 'optimized_hand_qtars' in self.samples_with_object_code:
                #     self.tot_dof_targets = self.samples_with_object_code['optimized_hand_qtars']
                #     self.tot_dof_targets = self.tot_dof_targets[0] # nn_envs x nn_ts x nn_hand_dof_
                else:
                    i_env = 0
                
            else:
                self.tot_dof_targets = None
            
        else:
            
            self.maxx_obj_nn = 1 
            self.maxx_obj_nn = 1000
            self.maxx_obj_nn = 10000
            # self.maxx_obj_nn = 50
            self.maxx_obj_nn = self.maxx_inst_nn
            self.tot_dof_targets = None
            # self.maxx_obj_nn = 200
            # self.maxx_obj_nn = 100
            # self.maxx_obj_nn = 50
            for i_inst_grab, grab_inst_tag in enumerate(self.grab_inst_tag_to_opt_res):
                print(f"[Debug] Loading {grab_inst_tag}, single_test_forbit_tags: {single_test_forbit_tags}")
                
                # if i_inst_grab == 0:
                #     continue
                # print(f"grab_inst_tag: {grab_inst_tag}, val: {self.grab_inst_tag_to_opt_res[grab_inst_tag]}")
                if isinstance(grab_inst_tag, tuple):
                    cur_grab_obj_type, cur_grab_traj_obj_type = grab_inst_tag
                else:
                    cur_grab_obj_type = grab_inst_tag
                if self.maxx_inst_nn == 1 and cur_grab_obj_type not in single_test_forbit_tags:
                    continue
                # hand type #
                
                ######## Previous code for setting the kinematic trajectory root ########
                if self.hand_type == 'leap' and 'grab_' in cur_grab_obj_type:
                    kine_root = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data/"
                    # /cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data
                    kine_root = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data"
                    kine_fn = f"leap_passive_active_info_{cur_grab_obj_type}.npy"
                    kine_fn  = os.path.join(kine_root, kine_fn)
                    if not os.path.exists(kine_fn):
                        continue
                    print(f"[Debug] kine_fn for leap: {kine_fn}")
                    # kine_root = '/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced/data'
                    # kine_fn = f"leap_passive_active_info_{cur_grab_obj_type.split('_nf_')[0]}.npy"
                    # kine_fn  = os.path.join(kine_root, kine_fn)
                    # if not os.path.exists(kine_fn):
                    #     continue
                ######## Previous code for setting the kinematic trajectory root ########
                
                
                ### TODO: in the test pool all the taco instances are used for training ###
                
                
                test_taco_tag = 'taco_20231024_'
                if self.hand_type != 'leap' and self.train_on_all_trajs:
                    if test_taco_tag in cur_grab_obj_type:
                        continue
                    
                
                
                # taco_20231024_124 
                # 
                target_taco_tag = 'taco_20231024_124'
                if self.hand_type == 'leap':
                    if 'GRAB' not in self.tracking_save_info_fn:
                        train_taco_tag = 'taco_20231104_'
                        # if train_taco_tag not in cur_grab_obj_type:
                        #     continue # 
                        if target_taco_tag not in cur_grab_obj_type:
                            continue
                    
                
                if self.only_training_on_succ_samples:
                    
                    if 'taco' in cur_grab_obj_type and self.taco_obj_type_to_opt_res is not None:
                        if cur_grab_obj_type not in self.taco_obj_type_to_opt_res:
                            continue
                        cur_inst_opt_res = self.taco_obj_type_to_opt_res[cur_grab_obj_type][0]
                        # if cur_inst_opt_res < self.rew_succ_threshold:
                        if self.obj_type_to_base_traj is None:
                            if cur_inst_opt_res < self.taco_rew_succ_threshold:
                                # cur_random_val = np.random.rand()
                                continue
                        else: # rew threshold #
                            if grab_inst_tag not in self.obj_type_to_base_traj and cur_inst_opt_res < self.taco_rew_succ_threshold:
                                continue
                        cur_random_val = np.random.rand()
                        if cur_random_val > self.data_selection_ratio:
                                continue
                    else:
                        if cur_grab_obj_type not in self.grab_obj_type_to_opt_res:
                            continue
                        cur_inst_opt_res = self.grab_obj_type_to_opt_res[cur_grab_obj_type][0] # with the succ threshold
                        if self.rew_thres_with_selected_insts:
                            if cur_inst_opt_res < self.rew_succ_threshold and cur_grab_obj_type not in sorted_inst_idxes_dict:
                                continue
                        else:
                            if cur_inst_opt_res < self.rew_succ_threshold: 
                                continue
                        cur_random_val = np.random.rand()
                        if cur_random_val > self.data_selection_ratio:
                                continue #
                    
                    # if we only train on succ samples --- for the grab instances, filter out unsucc trajectories #

                # only train on succ trajectories #
                if 'taco' in cur_grab_obj_type:
                    cur_grab_traj_obj_type = cur_grab_obj_type
                    print(f"cur_grab_obj_type: {cur_grab_obj_type}, grab_inst_tag: {grab_inst_tag}")
                    
                
                
                # print(f"cur_grab_obj_type: {cur_grab_obj_type}")
                if 'ori_grab' in cur_grab_obj_type:  #
                    # if cur_grab_obj_type not in grab_inst_tag_to_opt_stat:
                    #     continue
                    if self.test_subj_nm is not None and len(self.test_subj_nm) > 0:
                        if self.test_subj_nm not in cur_grab_obj_type:
                            continue
                        
                    if self.grab_train_test_setting and self.maxx_inst_nn > 1:
                        if '_s1_' in cur_grab_obj_type: # use the s1 as the test split
                            continue
                
                pure_cur_grab_obj_type = cur_grab_obj_type.split("_nf_")[0]
                
                # Generalist # inst opt fns # # simulation frequency and the control frequency #
                cur_inst_opt_fns = self.grab_inst_tag_to_opt_res[grab_inst_tag]
                
                if isinstance(cur_inst_opt_fns, tuple):
                    cur_inst_opt_fns = [cur_inst_opt_fns[1]]
                    
                if self.obj_type_to_base_traj is not None:
                    if grab_inst_tag in self.obj_type_to_base_traj:
                        print(f"grab_inst_tag: {grab_inst_tag}, val: {self.obj_type_to_base_traj[grab_inst_tag]}")
                        cur_obj_base_traj = self.obj_type_to_base_traj[grab_inst_tag][0]
                    else:
                        cur_obj_base_traj = None
                else:
                    cur_obj_base_traj = None
                
                # print(f"cur_inst_opt_fns: {cur_inst_opt_fns}")
                
                for i_inst, cur_inst_fn in enumerate(cur_inst_opt_fns):
                    
                    cur_inst_sorted_val_fn = cur_inst_fn.replace(".npy", "_sorted.npy")
                    if not os.path.exists(cur_inst_sorted_val_fn):
                        cur_inst_sorted_val_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
                        self.data_list.append(cur_inst_sorted_val_fn)
                        self.data_inst_tag_list.append(grab_inst_tag)
                        self.object_code_list.append(pure_cur_grab_obj_type)
                        self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                        
                        # cur_grab_obj_type # 
                        if self.use_teacher_model:
                            
                            if self.good_inst_opt_res is not None:
                                if (cur_grab_obj_type, cur_grab_obj_type) not in self.good_inst_opt_res:
                                    self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                                    continue
                            
                            if self.use_multiple_teacher_model:
                                found_teacher = False
                                for teacher_index in self.teacher_index_to_inst_tags_dict:
                                    cur_teacher_inst_tags_dict = self.teacher_index_to_inst_tags_dict[teacher_index]
                                    
                                    if cur_grab_obj_type in cur_teacher_inst_tags_dict:
                                        self.obj_tag_to_teacher_idx[pure_cur_grab_obj_type] = teacher_index
                                        self.object_rew_succ_dict[pure_cur_grab_obj_type] = 1
                                        found_teacher = True
                                        break
                                if not found_teacher:
                                    self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                            else:
                                if cur_grab_obj_type in self.teacher_model_inst_tags:
                                    self.object_rew_succ_dict[pure_cur_grab_obj_type] = 1
                                else:
                                    self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                        continue 
                    else:
                        if not os.path.exists(cur_inst_sorted_val_fn):
                            continue
                        cur_inst_sorted_val_fn_best = cur_inst_sorted_val_fn.replace(".npy", "_best.npy")
                        if os.path.exists(cur_inst_sorted_val_fn_best):
                            cur_inst_sorted_val_fn = cur_inst_sorted_val_fn_best
                        # cur_full_sorted_val_fn = os.path.join(data_folder, cur_inst_sorted_val_fn)
                        self.data_list.append(cur_inst_sorted_val_fn)
                        self.data_inst_tag_list.append(grab_inst_tag)
                        self.object_code_list.append(pure_cur_grab_obj_type)
                    
                    if grab_inst_tag in self.grab_inst_tag_to_best_opt_res:
                        cur_best_opt_res = self.grab_inst_tag_to_best_opt_res[grab_inst_tag]
                        cur_obj_rot_diff = cur_best_opt_res['obj_rot_diff'].item()
                        cur_obj_trans_diff = cur_best_opt_res['obj_pos_diff'].item()
                        ### defin the threshold for the rot diff adn the trans diff ###
                        succ_obj_trans_threshold = 0.10
                        succ_obj_trans_threshold = 0.05
                        succ_obj_rot_threshold = 0.3490658503988659
                        if cur_obj_trans_diff <= succ_obj_trans_threshold and cur_obj_rot_diff <= succ_obj_rot_threshold:
                            self.object_rew_succ_dict[pure_cur_grab_obj_type] = 1
                        else:
                            self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                    else:
                        self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                    
                    
                    if self.use_teacher_model:
                        
                        if self.good_inst_opt_res is not None:
                            if (cur_grab_obj_type, cur_grab_obj_type) not in self.good_inst_opt_res:
                                self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                                continue
                        
                        if self.use_multiple_teacher_model:
                            found_teacher = False # found teacher # # found teacher #
                            for teacher_index in self.teacher_index_to_inst_tags_dict:
                                cur_teacher_inst_tags_dict = self.teacher_index_to_inst_tags_dict[teacher_index]
                                if cur_grab_obj_type in cur_teacher_inst_tags_dict:
                                    self.obj_tag_to_teacher_idx[pure_cur_grab_obj_type] = teacher_index
                                    self.object_rew_succ_dict[pure_cur_grab_obj_type] = 1
                                    found_teacher = True
                                    break
                            if not found_teacher:
                                self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                        else:
                            if cur_grab_obj_type in self.teacher_model_inst_tags:
                                self.object_rew_succ_dict[pure_cur_grab_obj_type] = 1
                            else:
                                self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                        
                        
                        # if cur_grab_obj_type in self.teacher_model_inst_tags:
                        #     self.object_rew_succ_dict[pure_cur_grab_obj_type] = 1
                        # else:
                        #     self.object_rew_succ_dict[pure_cur_grab_obj_type] = 0
                            
                    
                    # if self.only_training_on_succ_samples:
                    #     self.object_rew_succ_dict[cur_grab_obj_type] = 1 
                    # else: # 
                    #     if 'taco' in cur_grab_obj_type:
                    #         if self.taco_obj_type_to_opt_res is not None:
                    #             if cur_grab_obj_type in self.taco_obj_type_to_opt_res:
                    #                 cur_obj_opt_res = self.taco_obj_type_to_opt_res[cur_grab_obj_type][0]
                    #                 if cur_obj_opt_res >= self.rew_succ_threshold:
                    #                     print(f"only_training_on_succ_samples: {self.only_training_on_succ_samples}, cur_grab_obj_type: {cur_grab_obj_type}, cur_obj_opt_res: {cur_obj_opt_res}")
                    #                     cur_random_val = np.random.rand()
                    #                     if cur_random_val > self.data_selection_ratio:
                    #                         self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    #                     else:
                    #                         self.object_rew_succ_dict[cur_grab_obj_type] = 1
                    #                 else:
                    #                     self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    #             else:
                    #                 self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    #     else: # grab obj type to opt res 
                    #         if self.grab_obj_type_to_opt_res is not None:
                                
                    #             # if 'taco' in cur_grab_obj_type and self.taco_obj_type_to_opt_res is not None:
                                
                    #             if cur_grab_obj_type in self.grab_obj_type_to_opt_res:
                    #                 cur_obj_opt_res = self.grab_obj_type_to_opt_res[cur_grab_obj_type][0]
                    #                 if self.rew_thres_with_selected_insts:
                    #                     if cur_obj_opt_res >= self.rew_succ_threshold or cur_grab_obj_type in sorted_inst_idxes_dict:
                    #                         # self.object_rew_succ_dict[cur_grab_obj_type] = 1
                    #                         cur_random_val = np.random.rand()
                    #                         if cur_random_val > self.data_selection_ratio:
                    #                             self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    #                         else:
                    #                             self.object_rew_succ_dict[cur_grab_obj_type] = 1
                    #                         print(f"cur_grab_obj_type: {cur_grab_obj_type}, cur_obj_opt_res: {cur_obj_opt_res}")
                    #                     else:
                    #                         self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    #                 else:
                    #                     if cur_obj_opt_res >= self.rew_succ_threshold:
                    #                         # self.object_rew_succ_list.append(1)
                    #                         self.object_rew_succ_dict[cur_grab_obj_type] = 1
                    #                         print(f"cur_grab_obj_type: {cur_grab_obj_type}, cur_obj_opt_res: {cur_obj_opt_res}")
                    #                     else:
                    #                         # self.object_rew_succ_list.append(0)
                    #                         self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    #             else:
                    #                 # self.object_rew_succ_list.append(0)
                    #                 self.object_rew_succ_dict[cur_grab_obj_type] = 0
                    
                
                self.data_base_traj.append(cur_obj_base_traj)
                
                
                if len(self.data_list) >= self.maxx_obj_nn:
                    break
        
        self.tot_obj_codes = self.object_code_list
        self.data_name_to_data = {}
        self.data_name_to_object_code = {}
        self.data_name_to_kine_info = {}
        
        
        
        tot_succ_nn = 0
        
        
        
        for obj_type in self.object_rew_succ_dict:
            if self.object_rew_succ_dict[obj_type] == 1:
                tot_succ_nn += 1
        print(f"[3] tot_succ_nn / tot_nn: {tot_succ_nn} / {len(self.object_rew_succ_dict)}")
        
        # self.tracking_info_st_tag = "passive_active_info_"
        
        self.maxx_trajectory_length = 0
        self._preload_mocap_tracking_ctl_data() 
        self._load_tracking_kine_info()
        self._load_object_type_to_feature()
        
        
        
        self._prepare_expert_traj_infos()
        if self.use_inst_latent_features:
            self._load_inst_tag_to_features() # load inst tag to optimized features #
        ## TODO: add a pre-trained point cloud encoder; ##
        ## TODO: add object point cloud features from that ##
        #### NOTE: Load data lis tand data instance tag list #### # perhaps #
        
        
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        self.control_freq_inv = control_freq_inv
        if self.reset_time > 0.0: # 
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)
        print(f"controlFrequencyInv: {self.control_freq_inv}")



        if self.w_franka:
            self.nn_hand_dof = 23
            self.glb_hand_dof = 7
        else: # leap #
            self.nn_hand_dof = 22
            self.glb_hand_dof = 6
        
        
        
        # self.shadow_hand_dof_speed_scale_list = [1.0] * 6 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        
        if self.w_franka:
            self.shadow_hand_dof_speed_scale_list = [self.shadow_hand_dof_speed_scale] * (self.nn_hand_dof )
        else:
            self.shadow_hand_dof_speed_scale_list = [self.glb_trans_vel_scale] * 3 + [self.glb_rot_vel_scale] * 3 + [self.shadow_hand_dof_speed_scale] * (self.nn_hand_dof - 6)
        
        
        
        self.shadow_hand_dof_speed_scale_tsr = torch.tensor(self.shadow_hand_dof_speed_scale_list, device=self.rl_device, dtype=torch.float) # #
        
        
        self.up_axis = 'z'
        # 'palm': 'palm_link',
        # 'thumb': 'link_15_tip',
        # 'index': 'link_3_tip', # link 3 tip
        # 'middle': 'link_7_tip', # link 3 tip
        # 'ring': 'link_11_tip' # link 3 tip #
        # self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        if self.use_fingertips:
            if self.hand_type == 'allegro':
                self.fingertips = ["link_15_tip", "link_3_tip", "link_7_tip", "link_11_tip"]
            elif self.hand_type == 'leap':
                # body_names = { # leap fingertips #
                #     'palm': 'palm_lower',
                #     'thumb': 'thumb_tip_head',
                #     'index': 'index_tip_head',
                #     'middle': 'middle_tip_head',
                #     'ring': 'ring_tip_head',
                # }
                # body_names = { # leap fingertips #
                #     'palm': 'palm_lower',
                #     'thumb': 'thumb_fingertip',
                #     'index': 'fingertip',
                #     'middle': 'fingertip_2',
                #     'ring': 'fingertip_3',
                # } # reward is not correct ? #
                self.fingertips = ["thumb_tip_head", "index_tip_head", "middle_tip_head", "ring_tip_head"]
                # self.fingertips = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
        else:
            self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        self.hand_center = ["palm_link"]
        self.num_fingertips = len(self.fingertips) 
        
        # you query the observations, and it gives you the current step (t)'s observation, with the origianl t+1's prediction as the target
        # but it is not the reward of the current prediction #
        # you query the # 
        
        
        self.mocap_sv_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_apple_lift.npy'
        
        # self.mocap_sv_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_apple_lift.npy'
        if not os.path.exists(self.mocap_sv_info_fn):
            self.mocap_sv_info_fn = './data/GRAB_Tracking_PK_OFFSET_Reduced/data/passive_active_info_ori_grab_s2_apple_lift_nf_300.npy'
        # if self.dataset_type == 'taco':
        #     self.mocap_sv_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_taco_s2_apple_lift.npy'
        self._load_mocap_info()
        
        self.max_episode_length = min(self.max_episode_length, self.hand_qs.shape[0] - 1)
        
        
        ### TODO xxx ###
        # self.max_episode_length = 149
        
        # 13 
        # self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        #  + 6 + nn_dof (action) + 
        # 16 (obj) + 7 + nn_dof (goal) + 64
        self.num_hand_obs = 66 + 76 + self.nn_hand_dof + 6 # self.glb_hand_dof  # 191 =  22*3 + (65+30) + 24
        
        # 2 * 22 + 13 * 4 + 6 + 22 + 16 + 7 + 22
        # num_pure_obs = 2 * self.nn_hand_dof + 13 * self.num_fingertips + 6 + self.nn_hand_dof + 16 + 7 + self.nn_hand_dof # 169
        
        
        
        if self.use_history_obs:
            num_pure_obs = (2 * self.nn_hand_dof + 13 * self.num_fingertips + 6 + self.nn_hand_dof + 7) * self.history_length + 9 + 7 + self.nn_hand_dof 
        else:
            num_pure_obs = 2 * self.nn_hand_dof + 13 * self.num_fingertips + 6 + self.nn_hand_dof + 16 + 7 + self.nn_hand_dof # 169
        
        if self.use_future_obs:
            num_pure_obs = num_pure_obs + (self.nn_hand_dof+ 7) * self.history_length # shoulld add 7 
            
            
        if self.include_obj_rot_in_obs:
            num_pure_obs = num_pure_obs + 4
        
            
        
        num_obs = self.num_hand_obs + 16 + 7 + self.nn_hand_dof + 64 #  236 + 64
        self.num_obs_dict = {
            "full_state": num_obs,
            "full_state_nforce": num_obs + 300 - num_obs, #  num_obs - self.nn_hand_dof - 24 # 24 -- fingertip forces
            "pure_state": num_pure_obs, # number obs - self.nnhanddofs #
            "pure_state_wref": num_pure_obs + self.nn_hand_dof,
            "pure_state_wref_wdelta": num_pure_obs + self.nn_hand_dof + self.nn_hand_dof
        } 
        
        
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        
        if not self.single_instance_state_based_test and not self.single_instance_state_based_train:
            if self.w_obj_latent_features:
                self.cfg['env']['numObservations'] += self.object_feat_dim  # get the obj feat dim # 
            
        # 
        if self.use_inst_latent_features: # #
            self.cfg['env']['numObservations'] += self.object_feat_dim 
        
        # tight obs #
        if self.tight_obs: # # 
            self.cfg['env']['numObservations'] -= 7
        
        print(f"obs_type: {self.obs_type}, num_obs: {self.cfg['env']['numObservations']}")
        
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = self.nn_hand_dof # 24 # # get the model #
        # self.cfg["device_type"] = device_type # # #
        # self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        
        
        
        if not self.single_instance_state_based_test and self.supervised_training:
            self.cfg['env']['numObservationsWActions'] = self.cfg['env']['numObservations'] + self.cfg['env']['numActions']
            
            # if self.grab_obj_type_to_opt_res is not None:
            self.cfg['env']['numObservationsWActions'] += 1
            
            print(f"numObservationsWActions: {self.cfg['env']['numObservationsWActions']}, numActions: {self.cfg['env']['numActions']}, numObservations: {self.cfg['env']['numObservations']}")
        
        if self.w_impedance_bias_control:
            self.cfg['env']['numActions'] = self.cfg['env']['numActions'] + 22 + 22
        
        # super().__init__(cfg=self.cfg, enable_camera_sensors=False) # camera #
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        print(f"num_shadow_hand_dofs: {self.num_shadow_hand_dofs}")
        
        if self.viewer != None:
            cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            cam_target = gymapi.Vec3(6.0, 5.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        rigid_body_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)
            self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, self.num_fingertips * 6)

            dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
            self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs,
                                                    self.num_shadow_hand_dofs + self.num_object_dofs)
            self.dof_force_tensor = self.dof_force_tensor[:, :self.num_shadow_hand_dofs]

        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        self.z_theta = torch.zeros(self.num_envs, device=self.device)
        
        
        

        print(f"Registering the history observations...")
        # shadow_hand_dof_pos, shadow_hand_dof_vel, fingertip_state, right_hand_pos, right_hand_rot, actions, object_pose # --- they need the histroy #
        # shadow hand dof pos # dof pos # dof vel #
        # get the history object pos # history buf # use the history buf # how to fill the history buf here ? # torch.where progress buf == this progress buf && history buf 
        # envs x xxxx info --- histroy buf == the current progress buf as the condition #
        # history_shadow_hand_dof_pos, history_shadow_hand_dof_vel, history_fingertip_state, history_right_hand_pos, history_right_hand_rot, history_right_hand_actions, history_object_pose
        self.history_buf_length = self.maxx_trajectory_length
        self.history_shadow_hand_dof_pos = torch.zeros((self.history_buf_length, self.num_envs, self.num_shadow_hand_dofs), device=self.device)
        self.history_shadow_hand_dof_vel = torch.zeros((self.history_buf_length, self.num_envs, self.num_shadow_hand_dofs), device=self.device)
        self.history_fingertip_state = torch.zeros((self.history_buf_length, self.num_envs, self.num_fingertips, 13), device=self.device)
        self.history_right_hand_pos = torch.zeros((self.history_buf_length, self.num_envs, 3), device=self.device)
        self.history_right_hand_rot = torch.zeros((self.history_buf_length, self.num_envs, 4), device=self.device)
        self.history_right_hand_actions = torch.zeros((self.history_buf_length, self.num_envs, self.num_shadow_hand_dofs), device=self.device)
        self.history_object_pose = torch.zeros((self.history_buf_length, self.num_envs, 7), device=self.device)
        
        
        
        if self.use_forcasting_model:
            # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot #
            self.forcast_buf_length = self.maxx_trajectory_length + 2
            # print(f"")
            print(f"forcast_buf_length: {self.forcast_buf_length}")
            # forcast shadow hand dof pose # 
            # try to save the forcasted dof values ##
            # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot #
            self.forcast_shadow_hand_dof_pos = torch.zeros((self.num_envs, self.forcast_buf_length, self.num_shadow_hand_dofs), device=self.device) # device #
            self.forcast_obj_pos = torch.zeros((self.num_envs, self.forcast_buf_length, 3), device=self.device) # 
            self.forcast_obj_rot = torch.zeros((self.num_envs, self.forcast_buf_length, 4), device=self.device) # get the forcast buf length #
        
        
        
        
        print(f"[Debug] shadow_hand_default_dof_pos: {self.shadow_hand_default_dof_pos.size()}")
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        print(f"[Debug] dof_state: {self.dof_state.size()}")
        # self.dof_state[:, 0] = self.shadow_hand_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).view(-1).contiguous()
        self.dof_state[:, 0] = self.shadow_hand_default_dof_pos.view(-1).contiguous()
        self.shadow_hand_dof_state = self.dof_state.view(self.num_envs, -1, 2)[:, :self.num_shadow_hand_dofs]
        self.shadow_hand_dof_pos = self.shadow_hand_dof_state[..., 0]
        self.shadow_hand_dof_vel = self.shadow_hand_dof_state[..., 1]
        self.rigid_body_states = gymtorch.wrap_tensor(rigid_body_tensor).view(self.num_envs, -1, 13)
        self.num_bodies = self.rigid_body_states.shape[1]
        self.root_state_tensor = gymtorch.wrap_tensor(actor_root_state_tensor).view(-1, 13)
        self.hand_positions = self.root_state_tensor[:, 0:3]
        self.hand_orientations = self.root_state_tensor[:, 3:7]
        self.hand_linvels = self.root_state_tensor[:, 7:10]
        self.hand_angvels = self.root_state_tensor[:, 10:13]
        self.saved_root_tensor = self.root_state_tensor.clone()
        self.saved_root_tensor[self.object_indices, 9:10] = 0.0
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs
        print(f"[Debug] num_dofs: {self.num_dofs}")
        
        # prev delta target and the current delta targets #
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        
        """ Targets buffers for the tracking targets prediction model """
        # self.prev_delta_tracking_targets = # #
        # TODO: 1) directly decode the tracking targets for both the hand and the object #
        self.pred_tracking_targets_hand_qs = torch.zeros((self.num_envs, self.maxx_trajectory_length, self.num_shadow_hand_dofs), dtype=torch.float, device=self.device)
        self.pred_tracking_targets_obj_pos = torch.zeros((self.num_envs, self.maxx_trajectory_length, 3), dtype=torch.float, device=self.device)
        self.pred_tracking_targets_obj_rot = torch.zeros((self.num_envs, self.maxx_trajectory_length, 4), dtype=torch.float, device=self.device)
        self.forecasting_rew_buf = torch.zeros_like(self.rew_buf)
        # nn_envs x max_episode_length x shadow hand dofs # 
        
        ##### Initialize the pred_tracking_targets_hand_qs ######
        self.pred_tracking_targets_hand_qs[..., 0, :] = self.shadow_hand_default_dof_pos.clone()
        ##### Initialize the pred_tracking_targets_hand_qs ######
        
        
        
        # prev_targets_impedance, cur_targets_impedance, prev_delta_targets_impedance, cur_delta_targets_impedance #
        if self.w_impedance_bias_control: # num envs x num dofs #
            self.prev_targets_impedance = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
            self.cur_targets_impedance = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
            self.prev_delta_targets_impedance = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
            self.cur_delta_targets_impedance = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        
        if self.rl_tracking_targets:
            self.prev_delta_tracking_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
            self.cur_delta_tracking_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device) # prev snad the cur tracking targets #
        
        
        self.prev_dof_vel = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_dof_vel = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.maxx_grasping_steps = 150
        self.maxx_grasping_steps = 300
        if self.use_twostage_rew:
            # grasping_progress_buf, grasp_manip_stages, grasping_succ_buf
            self.grasping_progress_buf = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
            
            self.grasp_manip_stages =  torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
            self.grasping_succ_buf = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device)
        
        self.global_indices = torch.arange(self.num_envs * 3, dtype=torch.int32, device=self.device).view(self.num_envs,-1)
        self.x_unit_tensor = to_torch([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = to_torch([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = to_torch([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.reset_goal_buf = self.reset_buf.clone()
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.current_successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.consecutive_successes = torch.zeros(1, dtype=torch.float, device=self.device) # consecutive successes #
        self.av_factor = to_torch(self.av_factor, dtype=torch.float, device=self.device) # apply force #
        self.apply_forces = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        self.apply_torque = torch.zeros((self.num_envs, self.num_bodies, 3), device=self.device, dtype=torch.float)
        # whether it has reached the lifting stage? #
        
        self.env_cond_type = torch.zeros((self.num_envs, ), dtype=torch.long, device=self.device) # zero is the COND_HAND_OBJ #
        self.env_cond_type = self.env_cond_type + self.preset_cond_type
        
        
        # self.preset_inv_cond_freq # self.preset_inv_cond_freq #
        self.env_inv_cond_freq = torch.ones_like(self.env_cond_type) * self.preset_inv_cond_freq
        
        # envs cond hand masks #
        self.env_cond_hand_masks = torch.ones((self.num_envs, self.nn_hand_dof), dtype=torch.float, device=self.device)
        
        if self.preset_cond_type == COND_PARTIALHAND_OBJ:
            self.env_cond_hand_masks[:, 6:] = 0.0 # 
        
        #### Add masked mimicing teacher observations ####
        if self.masked_mimic_training:
            self.mimic_teacher_obs_buf = torch.zeros((self.num_envs, self.cfg['env']['numObservations']), dtype=torch.float, device=self.device)
        
        # hand_palm_fingers_obj_contact_buf, right_hand_dist_buf
        self.hand_palm_fingers_obj_contact_buf = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.right_hand_dist_buf = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)    
        
        self.reach_lifting_stage = torch.zeros((self.num_envs), dtype=torch.float, device=self.device) # all zeros for the reach ifting stages # # lifting stages # # lifting stages #
        # a # multiple teachers with the random selected # # # # # all zeros for the reach lifting stages #
        
        self.total_successes = 0
        self.total_resets = 0
        
        self.ts_to_hand_obj_states = {}
        
        self.ref_ts =  0
        self.reset_nn = 0
        
    
        
    
    def _load_object_type_to_feature(self, ):
        self.object_type_to_latent_feature = np.load(self.object_type_to_latent_feature_fn, allow_pickle=True).item()
        # print(f"object_type_to_latent_feature: {self.object_type_to_latent_feature.keys()}")
        
    
    def _load_inst_tag_to_features(self, ):
        self.inst_tag_to_latent_features = np.load(self.inst_tag_to_latent_feature_fn, allow_pickle=True).item()
        
        
    # partial  to complete model and the model forcasting stetting #
    def _load_partial_to_complete_model(self, ):
        input_dims = {
            'X': 3,
            'feat': 22 , # + 3 + 3,
            # 'concat_two_dims': concat_two_dims
        }
        
        output_dims = {
            'X': input_dims['X'],
            'feat': input_dims['feat']
        }
        
        hidden_mlp_dims = {
            'X': 256,
            'feat':  256, # 
            't': 256,
        }
        
        input_dims = {
            'X': 3,
            'feat': 22 , # + 3 + 3,
            # 'concat_two_dims': concat_two_dims
        }
        
        n_layers = self.forcasting_model_n_layers
    
    
        # if self.forcasting_diffusion_model:
        input_dims = {
            'X': 3,
            'feat': 22 +3 +4, # + 3 + 3, # the difference is that the conditions is not jthe same -- it may be masked jfeatures of the full conditional features with 
            'concat_two_dims': False 
        }
        output_dims = {
            'X': input_dims['X'],
            'feat': input_dims['feat']
        }
        
        # if self.partial_hand_info:
        #     input_dims['feat'] = 22
        #     output_dims['feat'] = 22
            
        # if self.partial_obj_info:
        #     input_dims['feat'] = 3 + 4
        #     output_dims['feat'] = 3 + 4
            
        # input_dims['cond_feat'] = 29
        # output_dims['cond_feat'] = 29
        
        # 
        
        self.partial_to_complete_model = Transformer_Net_PC_Seq_V3_KineDiff_AE_V4(
            n_layers=n_layers,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            output_dims=output_dims, # output dims #
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            traj_cond=True,
            w_history_window_index=self.w_history_window_index
        )
        
        ######## Create diffusion model ########
        predict_xstart = True
        steps = 1000
        scale_beta = 1.
        timestep_respacing = ''
        learn_sigma = False
        rescale_timesteps = False

        noise_schedule = 'linear'
        sigma_small = True
        # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
        betas = gd_3d_pc.get_named_beta_schedule(noise_schedule, steps)
        loss_type = gd_3d_pc.LossType.MSE

        if not timestep_respacing:
            timestep_respacing = [steps]

        # partial to complete forwading mdoel 
        # partial to complete diffusion model
        self.partial_to_complete_diffusion = SpacedDiffusion3DPC(
            use_timesteps=space_timesteps(steps, timestep_respacing),
            betas=betas,
            model_mean_type=(
                gd_3d_pc.ModelMeanType.EPSILON if not predict_xstart else gd_3d_pc.ModelMeanType.START_X
            ),
            model_var_type=(
                (
                    gd_3d_pc.ModelVarType.FIXED_LARGE
                    if not sigma_small
                    else gd_3d_pc.ModelVarType.FIXED_SMALL
                )
                if not learn_sigma
                else gd_3d_pc.ModelVarType.LEARNED_RANGE
            ),
            loss_type=loss_type,
            rescale_timesteps=rescale_timesteps,
        )
        ######## Create diffusion model ########
        
        
        # else:
            
        #     if self.w_history_window_index:
        #         model_class = Transformer_Net_PC_Seq_V3_KineDiff_AE_V6
        #     else:
        #         model_class = Transformer_Net_PC_Seq_V3_KineDiff_AE_V3
            
        #     self.forcasting_model = model_class(
        #         n_layers=n_layers,
        #         input_dims=input_dims,
        #         hidden_mlp_dims=hidden_mlp_dims,
        #         output_dims=output_dims,
        #         act_fn_in=nn.ReLU(),
        #         act_fn_out=nn.ReLU(), 
        #         traj_cond=True,
        #         w_glb_traj_feat_cond=self.w_glb_traj_feat_cond
        #     )
        
        state_dicts = torch.load(
            self.partial_to_complete_model_weights, map_location='cpu'
        )
        # forcasting_model_state_dict = self.forcasting_model.state_dict() # get the generalist #
        # forcasting_model_state_dict.update(state_dicts)
        # self.forcasting_model.load_state_dict(forcasting_model_state_dict)
        self.partial_to_complete_model.load_state_dict(state_dicts)
        self.partial_to_complete_model = self.partial_to_complete_model.to(self.rl_device)
        
        if self.forcasting_model_training:
            self.partial_to_complete_model.train()
            # add the train function after each forrwarding # 
            # yes we do not need to do the eval function since the output in the training mode can still be in use # 
            trainable_params = []
            trainable_params += list(self.partial_to_complete_model.parameters())
            # self.1e-4 # add th 
            # TODO: add loss to train the forcasting model #
            self.partial_to_complete_model_opt = AdamW(
                trainable_params, lr=self.forcasting_model_lr, weight_decay=self.forcasting_model_weight_decay # get the weight decay parameter
            ) 
            self.partial_to_complete_model.train()
        else:
            self.partial_to_complete_model.eval()
    
        
        
    # partial  to complete model and the model forcasting stetting #
    def _load_forcasting_model(self, ):
        input_dims = {
            'X': 3,
            'feat': 22 , # + 3 + 3,
            # 'concat_two_dims': concat_two_dims
        }
        
        output_dims = {
            'X': input_dims['X'],
            'feat': input_dims['feat']
        }
        
        hidden_mlp_dims = {
            'X': 256,
            'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
            't': 256,
        }
        
        input_dims = {
            'X': 3,
            'feat': 22 , # + 3 + 3,
            # 'concat_two_dims': concat_two_dims
        }
        
        n_layers = self.forcasting_model_n_layers
    
    
        if self.forcasting_diffusion_model:
            input_dims = {
                'X': 3,
                'feat': 22 +3 +4, # + 3 + 3,
                'concat_two_dims': False 
            }
            output_dims = {
                'X': input_dims['X'],
                'feat': input_dims['feat']
            }
            
            if self.partial_hand_info:
                input_dims['feat'] = 22
                output_dims['feat'] = 22
                
            if self.partial_obj_info:
                input_dims['feat'] = 3 + 4
                output_dims['feat'] = 3 + 4
                
            input_dims['cond_feat'] = 29
            output_dims['cond_feat'] = 29
            
            if self.hist_cond_partial_obj_pos_info:
                input_dims['hist_cond_feat'] = 3
                output_dims['hist_cond_feat'] = 3
            if self.hist_cond_partial_hand_info:
                input_dims['hist_cond_feat'] = 22
                output_dims['hist_cond_feat'] = 22
            if self.hist_cond_partial_obj_info:
                input_dims['hist_cond_feat'] = 3 + 4
                output_dims['hist_cond_feat'] = 3 + 4
            
            st_ed_state_cond = self.st_ed_state_cond
            
            self.forcasting_model = Transformer_Net_PC_Seq_V3_KineDiff_AE_V4(
                n_layers=n_layers,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                output_dims=output_dims, # output dims #
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(),
                traj_cond=True,
                w_history_window_index=self.w_history_window_index,
                st_ed_state_cond=st_ed_state_cond
            )
            
            ######## Create diffusion model ########
            predict_xstart = True
            steps = 1000
            scale_beta = 1.
            timestep_respacing = ''
            learn_sigma = False
            rescale_timesteps = False

            noise_schedule = 'linear'
            sigma_small = True
            # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
            betas = gd_3d_pc.get_named_beta_schedule(noise_schedule, steps)
            loss_type = gd_3d_pc.LossType.MSE

            if not timestep_respacing:
                timestep_respacing = [steps]

            self.diffusion = SpacedDiffusion3DPC(
                use_timesteps=space_timesteps(steps, timestep_respacing),
                betas=betas,
                model_mean_type=(
                    gd_3d_pc.ModelMeanType.EPSILON if not predict_xstart else gd_3d_pc.ModelMeanType.START_X
                ),
                model_var_type=(
                    (
                        gd_3d_pc.ModelVarType.FIXED_LARGE
                        if not sigma_small
                        else gd_3d_pc.ModelVarType.FIXED_SMALL
                    )
                    if not learn_sigma
                    else gd_3d_pc.ModelVarType.LEARNED_RANGE
                ),
                loss_type=loss_type,
                rescale_timesteps=rescale_timesteps,
            )
            
            if self.forcasting_model_training:
                self.schedule_sampler_type = 'uniform'
                self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, self.diffusion)
                self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
            ######## Create diffusion model ########
        else:
            
            if self.w_history_window_index:
                model_class = Transformer_Net_PC_Seq_V3_KineDiff_AE_V6
            else:
                model_class = Transformer_Net_PC_Seq_V3_KineDiff_AE_V3
            
            use_clip_glb_features = self.use_clip_glb_features
            
            
            self.forcasting_model = model_class(
                n_layers=n_layers,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(), 
                traj_cond=True,
                w_glb_traj_feat_cond=self.w_glb_traj_feat_cond,
                use_clip_glb_features=use_clip_glb_features
            )
        
        if self.forcasting_model_weights.endswith('.pth'):
            state_dicts = torch.load(
                self.forcasting_model_weights, weights_only=True, map_location='cpu'
            )
        else:
            state_dicts = torch.load(
                self.forcasting_model_weights, map_location='cpu'
            )
        # forcasting_model_state_dict = self.forcasting_model.state_dict() # get the generalist #
        # forcasting_model_state_dict.update(state_dicts)
        # self.forcasting_model.load_state_dict(forcasting_model_state_dict)
        self.forcasting_model.load_state_dict(state_dicts)
        self.forcasting_model = self.forcasting_model.to(self.rl_device)
        
        if self.forcasting_model_training:
            self.forcasting_model.train()
            # add the train function after each forrwarding # 
            # yes we do not need to do the eval function since the output in the training mode can still be in use # 
            trainable_params = []
            trainable_params += list(self.forcasting_model.parameters())
            # self.1e-4 # add th 
            # TODO: add loss to train the forcasting model #
            self.forcasting_model_opt = AdamW(
                trainable_params, lr=self.forcasting_model_lr, weight_decay=self.forcasting_model_weight_decay # get the weight decay parameter
            ) 
            self.forcasting_model.train()
        else:
            self.forcasting_model.eval()
    
    
    
    def _forward_forcasting_model(self, ):
        
        
        # progress buf # progress buf #
        forcasting_progress_buf = self.progress_buf.unsqueeze(-1)
        # forcasting_ranged_ws = torch.range(0, self.forcasting_history_ws).to(self.rl_device) # nn_envs x nn_progress #
        # forcasting_progress_buf = forcasting_progress_buf + forcasting_ranged_ws.unsqueeze(0) # nn_envs x nn_progress #
        # forcasting_ranged_ws = forcasting_ranged_ws[::-1]
        forcasting_ranged_ws = torch.range(self.forcasting_history_ws - 1, 0, -1).to(self.rl_device) # ws - 1, ws - 2, ..., 0 #
        ### NOTE: multiple with the forcasting inv freq to cond only on those sparse frames ###
        forcasting_ranged_ws = forcasting_ranged_ws * self.forecasting_model_inv_freq 
        ### is the current progress recorded in the histroy dof pos and histroy object pose? ###
        # print(f"[Debug] forcasting_ranged_ws: {forcasting_ranged_ws.size()}")
        # print(f"[Debug] {forcasting_ranged_ws}") forcasting ranged ws # # forcasting ws forcasting process buf 
        forcasting_progress_buf = forcasting_progress_buf - forcasting_ranged_ws.unsqueeze(0) # nn_envs x nn_progress #
        forcasting_progress_buf = forcasting_progress_buf.long() # episode length #
        
        envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        envs_episode_length = envs_episode_length.unsqueeze(-1).repeat(1, self.forcasting_history_ws)
        forcasting_progress_buf = torch.clamp(forcasting_progress_buf, min=torch.zeros_like(envs_episode_length), max=envs_episode_length)
        # forcasting_history_hand_pose, forcasting_history_obj_pos
        forcasting_history_hand_pose = batched_index_select(self.history_shadow_hand_dof_pos.transpose(1,0).contiguous(), forcasting_progress_buf, dim=1) # nn_envs x n_histroy_ts x nn_hand_pse
        forcasting_history_obj_pos = batched_index_select(self.history_object_pose.transpose(1,0).contiguous(), forcasting_progress_buf, dim=1) # nn_envs x n_histroy_ts x nn_obj_pos
        # forcasting_history_obj_rot = batched_index_select(self.history_object_pose.transpose(1,0).contiguous(), forcasting_progress_buf, dim=1) # nn_envs x n_histroy_ts x nn_obj_rot
        
        # 
        
        forcasting_history_hand_pose = forcasting_history_hand_pose[..., self.joint_idxes_ordering_th] # nn_envs x nn_traj_len x envs_inst_idx #
        
        
        
        
        
        
        ## TODO: we need to alter the hand pose dof orders ##
        if self.w_glb_traj_feat_cond:
            # self.tot_kine_qs self.tot_kine_obj_trans, self.tot_kine_obj_ornt #
            # # self.tot_kine_obj_trans, self.tot_kine_obj_ornt #
            traj_hand_qs = batched_index_select(self.tot_kine_qs, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
            traj_obj_pos = batched_index_select(self.tot_kine_obj_trans, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
            traj_obj_rot = batched_index_select(self.tot_kine_obj_ornt, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
            # traj_obj_pts = batched_index_select(self.tot_obj_pts, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
            traj_hand_qs = traj_hand_qs[..., self.joint_idxes_ordering_th]
            
            traj_hand_qs = traj_hand_qs[:, ::self.forecasting_model_inv_freq]
            traj_obj_pos = traj_obj_pos[:, ::self.forecasting_model_inv_freq]
            traj_obj_rot = traj_obj_rot[:, ::self.forecasting_model_inv_freq]
            
            traj_hand_qs = traj_hand_qs[:, ::10, ]
            traj_obj_pos = traj_obj_pos[:, ::10, ]
            traj_obj_rot = traj_obj_rot[:, ::10, ]
            
            # # history_window_size, glb_feat_per_skip, centralize_info # # 
            traj_hand_qs = torch.cat(
                [ traj_hand_qs[:, ::self.glb_feat_per_skip], traj_hand_qs[:, -1:] ], dim=1
            )
            traj_obj_pos = torch.cat(
                [ traj_obj_pos[:, ::self.glb_feat_per_skip], traj_obj_pos[:, -1:] ], dim=1
            )
            traj_obj_rot = torch.cat(
                [ traj_obj_rot[:, ::self.glb_feat_per_skip], traj_obj_rot[:, -1:] ], dim=1
            )
        
        
        # if self.use_clip_glb_features:
        envs_text_features = batched_index_select(self.tot_text_features, self.env_inst_idxes, dim=0)
        
        # #### NOTE: DEBUG ####
        # envs_kine_traj_hand_qs = batched_index_select(self.tot_kine_qs, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
        # envs_kine_traj_obj_pos = batched_index_select(self.tot_kine_obj_trans, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
        # envs_kine_traj_obj_rot = batched_index_select(self.tot_kine_obj_ornt, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
        # # traj_obj_pts = batched_index_select(self.tot_obj_pts, self.env_inst_idxes, dim=0) # nn_envs x nn_traj_len x envs_inst_idx #
        # envs_kine_traj_hand_qs = envs_kine_traj_hand_qs[..., self.joint_idxes_ordering_th]
        # forcasting_history_hand_pose = batched_index_select(envs_kine_traj_hand_qs.contiguous(), forcasting_progress_buf, dim=1) # nn_envs x n_histroy_ts x nn_hand_pse
        # forcasting_history_obj_pos_pos = batched_index_select(envs_kine_traj_obj_pos.contiguous(), forcasting_progress_buf, dim=1) # nn_envs x n_histroy_ts x nn_obj_pos
        # forcasting_history_obj_pos_rot = batched_index_select(envs_kine_traj_obj_rot.contiguous(), forcasting_progress_buf, dim=1) # nn_envs x n_histroy_ts x nn_obj_pos
        # forcasting_history_obj_pos = torch.cat(
        #     [forcasting_history_obj_pos_pos, forcasting_history_obj_pos_rot], dim=-1
        # )
        # #### NOTE: DEBUG ####
        
        # conditions on the obj pose; condition on the hand pose #
        # the high condition on the obj pose only but not the hand pos only #
        # nn_envs x nn_obj_pts x 3 
        forcasting_obj_pts = batched_index_select(self.tot_obj_pts, self.env_inst_idxes, dim=0) # nn_envs x n_histroy_ts x nn_obj_pts
        forcasting_obj_pts_expanded = forcasting_obj_pts.unsqueeze(1).repeat(1, forcasting_history_obj_pos.size(1), 1, 1) # nn_envs x n_histroy_ts x nn_obj_pts x 3
        if self.w_glb_traj_feat_cond:
            forcasting_obj_pts_expanded_tot = forcasting_obj_pts.unsqueeze(1).repeat(1, traj_hand_qs.size(1), 1, 1)
        else:
            forcasting_obj_pts_expanded_tot = None
        tot_history_qs  = torch.cat([forcasting_history_hand_pose, forcasting_history_obj_pos], dim=-1) # nn_envs x n_histroy_ts x (nn_hand_pse + nn_obj_pos)
        
        tot_history_qs_tot_ts = tot_history_qs.clone()
        
        # nn_envs x nn_ts x nn_q_dim #
        tot_history_qs = tot_history_qs[:, -self.history_window_size:]
        # forcasting_obj_pts_expanded_tot = forcasting_obj_pts_expanded_tot[:, -self.history_window_size:]
        forcasting_obj_pts_expanded = forcasting_obj_pts_expanded[:, -self.history_window_size:] # forcasting pts #
        
        # tot history qs #
        tot_history_qs_full = tot_history_qs.clone()
        
        # if self.partial_hand_info:
        #     tot_history_qs = tot_history_qs[..., :22]
        # elif self.partial_obj_info:
        #     tot_history_qs = tot_history_qs[..., 22:]
        
        if self.w_glb_traj_feat_cond:
            tot_glb_qs = torch.cat([traj_hand_qs, traj_obj_pos, traj_obj_rot], dim=-1)
        else:
            tot_glb_qs = None
            
            
        if self.w_history_window_index:
            ws_offset_aranged = torch.arange(self.forcasting_history_ws - 1, -1, step=-1, dtype=torch.long, device=self.rl_device).unsqueeze(0).repeat(self.num_envs, 1) # nn_envs x history ws
            # print(f"ws_offset_aranged: {ws_offset_aranged}")
            ws_offset_aranged = ws_offset_aranged * self.forecasting_model_inv_freq
            offset_progress_buf = self.progress_buf.unsqueeze(-1) - ws_offset_aranged # .unsqueeze(0) # nn_envs x history_ws #
            self.num_frames = 300
            offset_progress_buf = torch.clamp(offset_progress_buf, min=torch.zeros_like(offset_progress_buf), max=envs_episode_length) # envs episode length 
            offset_progress_buf = offset_progress_buf.float() / float(self.num_frames)
            rescaled_num_frames = 1000 # rescaled num frames # # rescaled num frames #
            factorized_history_window_info = (offset_progress_buf * rescaled_num_frames).long()
            
            ws_future_offset_aranged = torch.arange(1, self.forcasting_history_ws + 1, dtype=torch.long, device=self.rl_device).unsqueeze(0).repeat(self.num_envs, 1) # nn_envs x history ws
            future_offset_pogress_buf = self.progress_buf.unsqueeze(-1) + ws_future_offset_aranged # .unsqueeze(0) # nn_envs x history_ws # # nn_envs x nn_future_ts
            future_offset_pogress_buf = torch.clamp(future_offset_pogress_buf, min=torch.zeros_like(future_offset_pogress_buf), max=envs_episode_length) # envs episode length
            future_offset_pogress_buf = future_offset_pogress_buf.float() / float(self.num_frames)
            factorized_future_window_info = (future_offset_pogress_buf * rescaled_num_frames).long()
            
            # get the history future window info #
            factorized_future_window_info = factorized_future_window_info[:,  -self.history_window_size: ] 
        
        
        # if self.forcasting_model_training:
        ############ Get future object pos and rot; Get future hand pose ############
        ### jet he ground truth forcasting information ###
        future_progress_buf = self.progress_buf.unsqueeze(-1)
        # to forcastg --- cur_progress + 1
        future_ranged_ws = torch.range(1, self.forcasting_history_ws, 1).to(self.rl_device)
        future_ranged_ws = future_ranged_ws * self.forecasting_model_inv_freq
        future_progress_buf = future_progress_buf + future_ranged_ws.unsqueeze(0) #
        future_progress_buf = future_progress_buf.long()
        ### nn_envs x nn_future_window ###
        future_progress_buf = torch.clamp(future_progress_buf, min=torch.zeros_like(future_progress_buf), max=envs_episode_length)
        
        # tot_goal_hand_qs_th = self.tot_kine_qs
        tot_goal_hand_qs_th = self.tot_hand_preopt_res
        envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0)
        # nn_envs x nn_future_ws x nn_hand_pose #
        cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, future_progress_buf, dim=1) # .squeeze(1) # nn_envs x len(hand_qs) #


        tot_goal_obj_trans_th = self.tot_kine_obj_trans
        tot_goal_obj_ornt_th = self.tot_kine_obj_ornt 
        

        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) 
        
        # envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) 
        
        cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, future_progress_buf, dim=1)
        cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, future_progress_buf, dim=1)
        
        cur_hand_qpos_ref = cur_hand_qpos_ref[..., self.joint_idxes_ordering_th]
        ############ Get future object pos and rot; Get future hand pose ############
        
        # inference_bsz = 128 # inference batch size # -- #
        # inference_bsz = 64
        inference_bsz = 512  
        inference_bsz = 128
        # inference_bsz = 256
        # inference_bsz = 1024
        # if self.w_glb_traj_feat_cond:
        #     inference_bsz = 8
        rt_val_dict = {} 
        tot_bsz_loss = []
        for i_st in range(0, forcasting_obj_pts_expanded.size(0), inference_bsz):
            i_ed = i_st + inference_bsz
            i_ed = min(i_ed, forcasting_obj_pts_expanded.size(0))
            cur_bz_obj_pts_expanded = forcasting_obj_pts_expanded[i_st: i_ed]
            cur_bz_obj_pts = forcasting_obj_pts[i_st: i_ed]
            cur_bz_history_qs = tot_history_qs_full[i_st: i_ed] #  tot_history_qs[i_st: i_ed]
            
            # if self.partial_hand_info: # history qs #
            #     cur_bz_history_qs[..., self.nn_hand_dof: ] = 0.0
            
            
            cur_bz_history_qs_input = tot_history_qs[i_st: i_ed]
            if self.w_glb_traj_feat_cond:
                cur_bsz_tot_glb_qs = tot_glb_qs[i_st: i_ed]
                cur_obj_pts_tot = forcasting_obj_pts_expanded_tot[i_st: i_ed]
            else:
                cur_bsz_tot_glb_qs = None
                cur_obj_pts_tot = None
                
            if self.centralize_info:
                # centralize the history qs and global qs # centralize #
                last_frame_hand_transl = cur_bz_history_qs[:, -1:, :3]
                cur_bz_history_qs[:, :, :3] = cur_bz_history_qs[:, :, :3] - last_frame_hand_transl
                cur_bz_history_qs[:, :, self.nn_hand_dof: self.nn_hand_dof + 3 ] = cur_bz_history_qs[:, :, self.nn_hand_dof: self.nn_hand_dof + 3 ] - last_frame_hand_transl
                cur_bsz_tot_glb_qs[:, :, :3] = cur_bsz_tot_glb_qs[:, :, :3]  - last_frame_hand_transl
                cur_bsz_tot_glb_qs[:, :, self.nn_hand_dof: self.nn_hand_dof + 3 ] = cur_bsz_tot_glb_qs[:, :, self.nn_hand_dof: self.nn_hand_dof + 3 ] - last_frame_hand_transl
            
            
            if self.use_clip_glb_features:
                cur_bsz_envs_text_features = envs_text_features[i_st: i_ed]
                cur_bsz_tot_glb_qs = cur_bsz_envs_text_features
            
            
            if self.forcasting_diffusion_model:
                
                if self.forcasting_model_training:
                    with torch.enable_grad():
                        self.forcasting_model.train()
                        self.forcasting_model_opt.zero_grad()
                        cur_bsz_obj_pts = forcasting_obj_pts[i_st: i_ed]
                        X = cur_bsz_obj_pts # nn_micro_bsz x nn_pts x 3 
                        # E = 
                        cur_bsz_goal_pos = cur_goal_pos[i_st: i_ed]
                        cur_bsz_goal_rot = cur_goal_rot[i_st: i_ed]
                        cur_bsz_hand_qpos_ref = cur_hand_qpos_ref[i_st: i_ed]
                        E =torch.cat(
                            [cur_bsz_hand_qpos_ref, cur_bsz_goal_pos, cur_bsz_goal_rot], dim=-1
                        )
                        E_cond = cur_bsz_tot_glb_qs
                        history_E_cond = cur_bz_history_qs
                        
                        micro = {
                            'X': X,
                            'E': E,
                            'X_cond': X,
                            'E_cond': E_cond,
                            'history_E_cond': history_E_cond
                        }
                        if self.w_history_window_index:
                            cur_bsz_factorized_history_window_info = factorized_history_window_info[i_st: i_ed]
                            micro.update(
                                {
                                    'history_E_window_idx': cur_bsz_factorized_history_window_info
                                }
                            )
                        # for key in micro:
                        #     val = micro[key]
                        #     print(f"key: {key}, val: {val.size()}")
                        
                        t, weights = self.schedule_sampler.sample(cur_bsz_goal_pos.size(0), self.rl_device)

                        
                        training_losses_func = self.diffusion.training_losses_AE_Diff
                        
                        calculate_loss_keys = ['E']
                        # print(f"self.model: {self.model}, diffusion: {self.diffusion}") # autoencoding #
                        losses = training_losses_func( ## get the losses from the diffusion model ##
                            self.forcasting_model,  # model
                            micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                            t,  # [bs](int) sampled timesteps
                            model_kwargs={'y': micro},
                            calculate_loss_keys=calculate_loss_keys
                        )
                        
                        # if isinstance(self.schedule_sampler, LossAwareSampler):
                        #     self.schedule_sampler.update_with_local_losses(
                        #         t, losses["loss"].detach()
                        #     )
                        
                        loss = (losses["loss"] * weights).mean()
                        
                        loss.backward()
                        self.forcasting_model_opt.step()
                        
                        tot_bsz_loss.append(loss.detach().cpu().item())

                        pass
                
                sample_fn = self.diffusion.p_sample_loop_AE_Diff
                X = cur_bz_obj_pts # nn bsz x nn bsz #
                E = tot_history_qs_tot_ts[i_st: i_ed] #  cur_bz_history_qs_input
                
                if self.partial_hand_info:
                    E = E[..., :self.nn_hand_dof]
                
                X_cond = cur_bz_obj_pts
                E_cond = cur_bsz_tot_glb_qs
                history_E_cond = cur_bz_history_qs
                
                cur_bsz_goal_pos = cur_goal_pos[i_st: i_ed]
                cur_bsz_goal_rot = cur_goal_rot[i_st: i_ed]
                cur_bsz_hand_qpos_ref = cur_hand_qpos_ref[i_st: i_ed]
                
                if self.centralize_info:
                    cur_bsz_goal_pos[:, :, :3] = cur_bsz_goal_pos[:, :, :3] - last_frame_hand_transl
                    cur_bsz_hand_qpos_ref[:, :, :3] = cur_bsz_hand_qpos_ref[:, :, :3]  - last_frame_hand_transl
                            
                if self.hist_cond_partial_hand_info:
                    history_E_cond = history_E_cond[..., :self.nn_hand_dof]
                elif self.hist_cond_partial_obj_info:
                    history_E_cond = history_E_cond[..., self.nn_hand_dof:]
                elif self.hist_cond_partial_obj_pos_info:
                    history_E_cond = history_E_cond[..., self.nn_hand_dof: self.nn_hand_dof + 3]
                
                # ## NOTE: currently we only consider the partial object postion conditioning and forecasting ##
                # # we should reduce the history_E_cond to the current bsz #
                # history_E_cond = history_E_cond[:, -1:, :]
                
                
                # get the conditional model #
                # history_E_cond[..., : self.nn_hand_dof] = 0.0
                # history_E_cond[..., self.nn_hand_dof + 3: ] = 0.0
                
                micro = {'X': X, 'E': E, 'X_cond': X_cond, 'E_cond': E_cond, 'history_E_cond': history_E_cond}
                
                if self.w_history_window_index: # get forcasted window indexes #
                    cur_bsz_factorized_history_window_info = factorized_history_window_info[i_st: i_ed]
                    
                    # ## NOTE: currently we only consider the partial object postion conditioning and forecasting ##
                    # cur_bsz_factorized_history_window_info = cur_bsz_factorized_history_window_info[:, -1:]
                    cur_bsz_factorized_history_window_info = cur_bsz_factorized_history_window_info[:, -self.history_window_size:]
                    micro['history_E_window_idx'] = cur_bsz_factorized_history_window_info
                    print(f"micro: {micro.keys()}")
                    for key in micro:
                        print(f"key: {key}, val: {micro[key].size()}")
                        # print(f"E_cond: {E_cond[0][-1]}")
                
                
                # we should
                # we should add the history window length for describing history size ##
                
                shape = {
                    key: micro[key].shape for key in micro
                }
                samples = sample_fn(
                    self.forcasting_model, 
                    shape,
                    noise=None,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                    progress=True,
                    use_t=False,
                    data=micro,
                    # ret_encoded_feat=True # sparsse conditional info #
                )
                sampled_feats = samples['E']
                sampled_nn_ts = sampled_feats.size(1)
                # bsz x nn_ts x
                
                if self.partial_hand_info:
                    cur_bz_rt_val_dict = {
                        'hand_pose': sampled_feats[..., : self.nn_hand_dof],
                        'obj_pos': cur_bsz_goal_pos[:, 0:1].repeat(1, sampled_nn_ts, 1), # torch.zeros_like(sampled_feats[..., : self.nn_hand_dof])[..., :3],
                        'obj_ornt': cur_bsz_goal_rot[:, 0:1].repeat(1, sampled_nn_ts, 1) #  torch.zeros_like(sampled_feats[..., : self.nn_hand_dof])[..., :4],
                        # 'obj_pos': sampled_feats[..., self.nn_hand_dof: self.nn_hand_dof + 3],
                        # 'obj_ornt': sampled_feats[..., self.nn_hand_dof + 3: self.nn_hand_dof + 7]
                    }
                elif self.partial_obj_info:
                    cur_bz_rt_val_dict = {
                        'hand_pose': cur_bsz_hand_qpos_ref[:, 0:1].repeat(1, sampled_nn_ts, 1), #  torch.zeros((sampled_feats.size(0), self.nn_hand_dof), device=self.rl_device).float(),
                        'obj_pos': sampled_feats[..., :3 ],
                        'obj_ornt': sampled_feats[..., 3 : 3 + 4 ],
                        # 'obj_pos': sampled_feats[..., self.nn_hand_dof: self.nn_hand_dof + 3],
                        # 'obj_ornt': sampled_feats[..., self.nn_hand_dof + 3: self.nn_hand_dof + 7]
                    }
                elif self.partial_obj_pos_info: 
                    cur_bz_rt_val_dict = {
                        'hand_pose': cur_bsz_hand_qpos_ref[:, 0:1].repeat(1, sampled_nn_ts, 1), #  torch.zeros((sampled_feats.size(0), self.nn_hand_dof), device=self.rl_device).float(),
                        'obj_pos': sampled_feats[..., :3 ],
                        'obj_ornt':  cur_bsz_goal_rot[:, 0:1].repeat(1, sampled_nn_ts, 1), # torch.zeros((sampled_feats.size(0), 4), device=self.rl_device).float(),
                    }
                else:
                    cur_bz_rt_val_dict = {
                        'hand_pose': sampled_feats[..., : self.nn_hand_dof],
                        'obj_pos': sampled_feats[..., self.nn_hand_dof: self.nn_hand_dof + 3],
                        'obj_ornt': sampled_feats[..., self.nn_hand_dof + 3: self.nn_hand_dof + 7]
                    }
                    
                if self.use_partial_to_complete_model:
                    hand_pose = cur_bz_rt_val_dict['hand_pose']
                    obj_pos = cur_bz_rt_val_dict['obj_pos']
                    obj_ornt = cur_bz_rt_val_dict['obj_ornt']
                    cur_history_E_cond = torch.cat(
                        [ hand_pose, obj_pos, obj_ornt], dim=-1
                    )
                    
                    sample_fn = self.partial_to_complete_diffusion.p_sample_loop_AE_Diff
                    X = cur_bz_obj_pts # nn bsz x nn bsz #
                    E = tot_history_qs_full[i_st: i_ed] #  cur_bz_history_qs_input
                    X_cond = cur_bz_obj_pts
                    E_cond = cur_bsz_tot_glb_qs
                    history_E_cond = cur_history_E_cond # get the history qs # # 
                    micro = {'X': X, 'E': E, 'X_cond': X_cond, 'E_cond': E_cond, 'history_E_cond': history_E_cond}
                    
                    if self.w_history_window_index: # get forcasted window indexes # # conditional tuning #
                        cur_bsz_factorized_history_window_info = factorized_future_window_info[i_st: i_ed]
                        micro['history_E_window_idx'] = cur_bsz_factorized_history_window_info
                        print(f"micro: {micro.keys()}")
                    
                    shape = {
                        key: micro[key].shape for key in micro
                    }
                    samples = sample_fn(
                        self.partial_to_complete_model, 
                        shape,
                        noise=None,
                        clip_denoised=False,
                        denoised_fn=None,
                        cond_fn=None,
                        model_kwargs=None,
                        progress=True,
                        use_t=False,
                        data=micro,
                        # ret_encoded_feat=True # sparsse conditional info # # 
                    )  # 
                    
                    sampled_feats = samples['E']
                    cur_bz_rt_val_dict = {
                        'hand_pose': sampled_feats[..., : self.nn_hand_dof],
                        'obj_pos': sampled_feats[..., self.nn_hand_dof: self.nn_hand_dof + 3],
                        'obj_ornt': sampled_feats[..., self.nn_hand_dof + 3: self.nn_hand_dof + 7]
                    }
                
                
            else:
                if self.forcasting_model_training: # forcasting model training # # model
                    
                    with torch.enable_grad():
                        self.forcasting_model.train()
                        self.forcasting_model_opt.zero_grad()
                        if self.w_history_window_index:
                            cur_bsz_factorized_history_window_info = factorized_history_window_info[i_st: i_ed]
                            cur_bz_rt_val_dict =  self.forcasting_model(cur_bz_obj_pts_expanded, cur_bz_history_qs, tot_feat_feat=cur_bsz_tot_glb_qs, tot_obj_pts=cur_obj_pts_tot, history_window_index=cur_bsz_factorized_history_window_info)
                        else:
                            cur_bz_rt_val_dict =  self.forcasting_model(cur_bz_obj_pts_expanded, cur_bz_history_qs, tot_feat_feat=cur_bsz_tot_glb_qs, tot_obj_pts=cur_obj_pts_tot)
                        
                        cur_bsz_goal_pos = cur_goal_pos[i_st: i_ed]
                        cur_bsz_goal_rot = cur_goal_rot[i_st: i_ed]
                        cur_bsz_hand_qpos_ref = cur_hand_qpos_ref[i_st: i_ed]
                        
                        if self.centralize_info:
                            cur_bsz_goal_pos[:, :, :3] = cur_bsz_goal_pos[:, :, :3] - last_frame_hand_transl
                            cur_bsz_hand_qpos_ref[:, :, :3] = cur_bsz_hand_qpos_ref[:, :, :3]  - last_frame_hand_transl
                            
                        
                        loss_pred_goal_pos = torch.mean(torch.sum((cur_bsz_goal_pos - cur_bz_rt_val_dict['obj_pos']) ** 2, dim=-1))
                        loss_pred_goal_ornt = torch.mean(torch.sum((cur_bsz_goal_rot - cur_bz_rt_val_dict['obj_ornt']) ** 2, dim=-1))
                        loss_pred_hand_qtars = torch.mean(torch.sum((cur_bsz_hand_qpos_ref - cur_bz_rt_val_dict['hand_pose']) ** 2, dim=-1))

                        # print(f"loss_pred_goal_pos: {loss_pred_goal_pos}, grad: {cur_bz_rt_val_dict['obj_pos'].grad}")
                        # print(f"loss_pred_goal_pos: {loss_pred_goal_ornt}, grad: {loss_pred_goal_ornt.grad}")
                        # print(f"loss_pred_goal_pos: {loss_pred_hand_qtars}, grad: {loss_pred_hand_qtars.grad}")
                        tot_loss = loss_pred_goal_pos + loss_pred_goal_ornt + loss_pred_hand_qtars
                        
                        
                        tot_loss.backward()
                        self.forcasting_model_opt.step()
                        
                        tot_bsz_loss.append(tot_loss.detach().cpu().item())
                        pass
                else:
                    if self.w_history_window_index:
                        
                        cur_bsz_factorized_history_window_info = factorized_history_window_info[i_st: i_ed]
                        print(f"cur_bsz_factorized_history_window_info: {cur_bsz_factorized_history_window_info.size()}, cur_bz_history_qs: {cur_bz_history_qs.size()}, ")
                        cur_bz_rt_val_dict =  self.forcasting_model(cur_bz_obj_pts_expanded, cur_bz_history_qs, tot_feat_feat=cur_bsz_tot_glb_qs, tot_obj_pts=cur_obj_pts_tot, history_window_index=cur_bsz_factorized_history_window_info)
                    else:
                        cur_bz_rt_val_dict =  self.forcasting_model(cur_bz_obj_pts_expanded, cur_bz_history_qs, tot_feat_feat=cur_bsz_tot_glb_qs, tot_obj_pts=cur_obj_pts_tot)
                
                
                
            # 
            if self.centralize_info:
                cur_bz_rt_val_dict['hand_pose'][:, :, :3] = cur_bz_rt_val_dict['hand_pose'][:, :, :3] + last_frame_hand_transl
                cur_bz_rt_val_dict['obj_pos'][:, :, :3] = cur_bz_rt_val_dict['obj_pos'][:, :, :3] + last_frame_hand_transl
                
            
            
            for key in cur_bz_rt_val_dict:
                if key not in rt_val_dict:
                    rt_val_dict[key] = [ cur_bz_rt_val_dict[key].detach() ]
                else:
                    rt_val_dict[key].append(cur_bz_rt_val_dict[key].detach())
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            torch.cuda.empty_cache()
            
        for key in rt_val_dict:
            rt_val_dict[key] = torch.cat(rt_val_dict[key], dim=0)
            
        rt_val_dict_np = {}
        cur_input_traj_hand_qs = tot_history_qs.detach().cpu().numpy() 
        for key in rt_val_dict:
            rt_val_dict_np[key] = rt_val_dict[key].detach().cpu().numpy() # 
        # returned hand qs #
        forcasting_model_forwarding_info_dict = {
            'input': cur_input_traj_hand_qs, 
            'output': rt_val_dict_np # rt val dict #
        }
        self.forcasting_model_forwarding_info_dict = forcasting_model_forwarding_info_dict
        
        
        
        if self.forcasting_model_training:
            tot_bsz_loss = sum(tot_bsz_loss) / float(len(tot_bsz_loss))
            print(f"Training the forcasting model with loss: {tot_bsz_loss}")
        
        
        
        pred_hand_pose = rt_val_dict['hand_pose']
        pred_obj_pos = rt_val_dict['obj_pos']
        pred_obj_ornt = rt_val_dict['obj_ornt']
        
        ## pred obj pos ##
        pred_hand_pose = pred_hand_pose[..., self.inversed_joint_idxes_ordering_th] # inversed joint ordering #
        
        # they are the future kines and the base trajs here #
        
        # TODO: replace the pred hand pose an the pred hannornt #
        
        self.already_forcasted = True
        
        # the question is how to set them to the forcasted models? #
        # # pred hand pose #
        print(f"pred_obj_pos: {pred_obj_pos.size(), }, pred_hand_pose: {pred_hand_pose.size()}")
        envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        # forecast_future_freq = self.forecast_future_freq
        forecast_future_freq = self.forecasting_model_inv_freq
        for i_ws in range(pred_obj_pos.size(1)):
            
            # forcast_obj_pos: nn_envs x nn_forcasting_ts x obj_pos_dim # 
            for i_sub_assign_idx in range(forecast_future_freq):
                cur_forcast_assign_idx = self.progress_buf + (i_ws * forecast_future_freq + i_sub_assign_idx) + 1
                # forcast assign idx #
                cur_forcast_assign_idx = torch.clamp(cur_forcast_assign_idx, min=torch.zeros_like(cur_forcast_assign_idx), max=envs_episode_length)
                # objpos: nn_envs x nn_length x 3
                aranged_forcasting_idxes = torch.arange(self.forcast_obj_pos.size(1), device=self.rl_device).unsqueeze(0).repeat(self.forcast_obj_pos.size(0), 1)
                # aranged forcasting idxes #
                assign_mask = (aranged_forcasting_idxes == cur_forcast_assign_idx.unsqueeze(-1))
                self.forcast_obj_pos = torch.where(
                    assign_mask.unsqueeze(-1).repeat(1, 1, 3), pred_obj_pos[:, i_ws].unsqueeze(1).repeat(1, self.forcast_obj_pos.size(1), 1), self.forcast_obj_pos
                )
                self.forcast_obj_rot = torch.where(
                    assign_mask.unsqueeze(-1).repeat(1, 1, 4), pred_obj_ornt[:, i_ws].unsqueeze(1).repeat(1, self.forcast_obj_pos.size(1), 1), self.forcast_obj_rot
                )
                self.forcast_shadow_hand_dof_pos = torch.where(
                    assign_mask.unsqueeze(-1).repeat(1, 1, self.forcast_shadow_hand_dof_pos.size(-1)), pred_hand_pose[:, i_ws].unsqueeze(1).repeat(1, self.forcast_obj_pos.size(1), 1), self.forcast_shadow_hand_dof_pos
                )
            
            
            # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot #
            # self.forcast_obj_pos[:, i_ws] = pred_obj_pos[:, i_ws]
            # self.forcast_obj_rot[:, i_ws] = pred_obj_ornt[:, i_ws]
            # self.forcast_shadow_hand_dof_pos[:, i_ws] = pred_hand_pose[:, i_ws]
            
            
            
    
    
    def _preload_single_tracking_ctl_data(self, data_fn, add_to_dict=True, key_data_fn=None):
        
        
        if key_data_fn is None:
            key_data_fn = data_fn
        
        if not os.path.exists(data_fn):
            optimized_obj_pose = np.zeros((300, 7), dtype=np.float32)
            optimized_hand_qtars = np.zeros((1, 300, 22), dtype=np.float32)
            optimized_hand_qs = np.zeros((1, 300, 22), dtype=np.float32)
            cur_data= {}
        else:
            cur_data = np.load(data_fn, allow_pickle=True).item()
            # if self.use_jointspace_seq:
            # if self.sim_platform == 'isaac':
            optimized_obj_pose = cur_data['optimized_obj_pose']
            optimized_hand_qtars = cur_data['optimized_hand_qtars']
            optimized_hand_qs = cur_data['optimized_hand_qs']
            # # TODO: use top-k other than using the best evaluated res? 
            
            # hand_qs_np = optimized_hand_qs[0][..., self.joint_idxes_inversed_ordering]
            # hand_qtars_np = optimized_hand_qtars[0][..., self.joint_idxes_inversed_ordering]
        
        
        
        # # print(f"loading from {data_fn}")
        # cur_data = np.load(data_fn, allow_pickle=True).item()
        # # if self.use_jointspace_seq:
        # # if self.sim_platform == 'isaac':
        # optimized_obj_pose = cur_data['optimized_obj_pose']
        # optimized_hand_qtars = cur_data['optimized_hand_qtars']
        # optimized_hand_qs = cur_data['optimized_hand_qs']
        # # # TODO: use top-k other than using the best evaluated res? 
        
        # hand_qs_np = optimized_hand_qs[0][..., self.joint_idxes_inversed_ordering]
        # hand_qtars_np = optimized_hand_qtars[0][..., self.joint_idxes_inversed_ordering]
        
        # 
        hand_qs_np = optimized_hand_qs[0]
        hand_qtars_np = optimized_hand_qtars[0]
        
        
        # if self.glb_rot_use_quat:
        #     hand_glb_rot_np = hand_qs_np[..., 3:6]
        #     hand_glb_qtar_np = hand_qtars_np[..., 3:6]
        #     hand_glb_rot_th = torch.from_numpy(hand_glb_rot_np)
        #     hand_glb_tar_rot_th = torch.from_numpy(hand_glb_qtar_np)
        #     hand_glb_quat_th = quat_from_euler_xyz(hand_glb_rot_th[..., 0], hand_glb_rot_th[..., 1], hand_glb_rot_th[..., 2])
        #     hand_glb_tar_quat_th = quat_from_euler_xyz(hand_glb_tar_rot_th[..., 0], hand_glb_tar_rot_th[..., 1], hand_glb_tar_rot_th[..., 2])
        #     hand_glb_rot_np = hand_glb_quat_th.numpy()
        #     hand_glb_qtar_np = hand_glb_tar_quat_th.numpy()
            
        #     hand_qs_np = np.concatenate(
        #         [ hand_qs_np[..., :3], hand_glb_rot_np, hand_qs_np[..., 6:] ], axis=-1
        #     )
        #     hand_qtars_np = np.concatenate(
        #         [ hand_qtars_np[..., :3], hand_glb_qtar_np, hand_qtars_np[..., 6:] ], axis=-1
        #     )
        #     # hand_qs_np[..., 3:6] = hand_glb_rot_np
        #     # hand_qtars_np[..., 3:6] = hand_glb_qtar_np
            # obj_pose_np = cu
            
        # else:
        #     ts_to_hand_qs = cur_data['ts_to_hand_qs'] 
        #     ts_to_hand_qtars = cur_data['ts_to_qtars']
        #     if self.slicing_data:
        #         sorted_ts = sorted(list(ts_to_hand_qs.keys()))
        #         hand_qs_np = [
        #             ts_to_hand_qs[i_ts] for i_ts in sorted_ts
        #         ]
        #         hand_qtars_np = [
        #             ts_to_hand_qtars[i_ts] for i_ts in sorted_ts
        #         ]
        #     else:
        #         if 'ts_to_optimized_q_tars_wcontrolfreq' in cur_data:
        #             ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
        #         else:
        #             ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_hand_tot_ctl_qtars']
        #         ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                    
        #         max_hand_qs_kd = max(list(ts_to_hand_qs.keys()))
        #         ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
        #         ctl_freq_tss = sorted(ctl_freq_tss) # ctl freq tss #
        #         ctl_freq = 10 # 
        #         ctl_freq_tss_expanded = [ min(max_hand_qs_kd, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
        #         ts_to_hand_qs = {
        #             ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
        #         }
        #         hand_qs_np = [
        #             ts_to_hand_qs[cur_ts] for cur_ts in ctl_freq_tss
        #         ]
        #         hand_qtars_np = [ # 
        #             ts_to_hand_qtars[cur_ts] for cur_ts in ctl_freq_tss
        #         ]
        #     hand_qs_np = np.stack(hand_qs_np, axis=0)
        #     hand_qtars_np = np.stack(hand_qtars_np, axis=0) 
        
        cur_clip_data = {
            'hand_qs': hand_qs_np,
            'hand_qtars': hand_qtars_np
        }
        
        if 'actions' in cur_data:
            actions = cur_data['actions']
            cur_clip_data['actions'] = actions

        
        
        # cur_clip_data = {
        #     'tot_verts': hand_qs_np[None],
        #     # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
        #     'tot_verts_integrated_qdd_tau': hand_qtars_np[None], 
        # }
        
        # TODO: 
        # history fture? -- not the model yet ... #
        # if self.task_cond_type == 'history_future':
        #     obj_pose_np = optimized_obj_pose[0]
        #     cur_clip_data.update(
        #         {
        #             'tot_obj_pose': obj_pose_np[None]
        #         }
        #     )
        # cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
        if add_to_dict:
            self.data_name_to_data[key_data_fn] = cur_clip_data
        return cur_clip_data, hand_qs_np, hand_qtars_np
    
    
    def _preload_mocap_tracking_ctl_data(self,):  
        # print(f"Entering func _preload_mocap_tracking_ctl_data")
        # self.data_list #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        # if self.single_inst: # 
        #     self.data_list = self.data_list[:1]
        #     self.data_inst_tag_list = self.data_inst_tag_list[:1]
        # elif self.multi_inst:
        #     self.data_list = self.data_list[:10]
        #     self.data_inst_tag_list = self.data_inst_tag_list[:10]
        # tot_expanded_passive #
        forbid_data_inst_tags = ["ori_grab_s2_phone_call_1", "ori_grab_s2_phone_pass_1"]
        
        for i_data_inst, data_fn in enumerate(self.data_list): # preload trajectories #
            print(f"{i_data_inst}/{len(self.data_list)} Loading from {data_fn}")
            excluded = False 
            for cur_forbid_inst_tag in forbid_data_inst_tags:
                if cur_forbid_inst_tag in data_fn and 'taco' not in data_fn:
                    excluded = True
                    break
            if excluded:
                continue
            
            cur_obj_code = self.object_code_list[i_data_inst]
            
            if 'taco' in cur_obj_code:
                if self.taco_obj_type_to_opt_res is not None:
                    if self.object_rew_succ_dict[cur_obj_code] == 1:
                        real_data_fn = data_fn
                    else:
                        real_data_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230930_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-20-05-36/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
                    # if cur_obj_code in self.taco_obj_type_to_opt_res:
                    #     # cur_obj_opt_res = self.taco_obj_type_to_opt_res[cur_obj_code][0]
                    #     # if cur_obj_opt_res >= self.rew_succ_threshold:
                    #     #     self.object_rew_succ_dict[cur_obj_code] = 1
                    #     # else:
                    #     #     self.object_rew_succ_dict[cur_obj_code] = 0
                    #     real_data_fn = data_fn
                    # else:
                    #     self.object_rew_succ_dict[cur_obj_code] = 0
                else:
                    real_data_fn = data_fn
            else:
                real_data_fn = data_fn
            
            
            print(f"{i_data_inst}/{len(self.data_list)} Loading from {data_fn}, len(data_name_to_data): {len(self.data_name_to_data)}")
            # load tracking single ctl data # preload single tracking ctl data #
            
            if data_fn in self.data_name_to_data:
                # data_fn_key = data_fn + str(i_data_inst)
                data_fn_key = data_fn + str(i_data_inst)
            else:
                # data_fn_key = data_fn
                data_fn_key = data_fn
            
            # key_data_fn
            cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(real_data_fn, key_data_fn=data_fn_key)
            tot_data_hand_qs.append(hand_qs_np)
            tot_data_hand_qtars.append(hand_qtars_np)
            
            
            self.data_name_to_data[data_fn_key] = cur_clip_data
            self.data_name_to_object_code[data_fn_key] = self.object_code_list[i_data_inst]
            
            cur_traj_maxx_length = max(hand_qs_np.shape[0], hand_qtars_np.shape[0])
            self.maxx_trajectory_length  = max(self.maxx_trajectory_length, cur_traj_maxx_length)
            
            if 'taco' in data_fn:
                print(f'Loading from {data_fn}, cur_traj_maxx_length: {cur_traj_maxx_length}, maxx_trajectory_length: {self.maxx_trajectory_length}')
        print(f"Existing func _preload_mocap_tracking_ctl_data")
     
    def _load_single_tracking_kine_info(self, data_inst_tag, cur_base_traj_fn=None):
        print(f"Loading single with data_inst_tag: {data_inst_tag}")
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            if 'taco' in kine_data_fn:
                passive_act_pure_tag = "passive_active_info_ori_grab_s2_phone_call_1_interped_"
                # self.objtype_to_tracking_sv_info = {}
                # for cur_sv_info in tracking_save_info:
                cur_objtype = kine_data_fn.split("/")[-1].split(".")[0]
                # cur_objtype = cur_objtype.split("_nf_")[0]
                cur_objtype = cur_objtype[len(passive_act_pure_tag): ]
                cur_objtype_segs = cur_objtype.split("_")
                cur_objtype = "_".join(cur_objtype_segs[0: 3])
                kine_object_type= cur_objtype
                # self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)
            else:
                kine_data_tag = "passive_active_info_"
                kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
                kine_object_type = kine_object_type.split("_nf_")[0]
            return kine_object_type
        
        if isinstance(data_inst_tag, str):
            
            if data_inst_tag.endswith('.npy'):
                # object type can be the grab inst tag or the taco inst tag #
                kine_info_fn = data_inst_tag
                cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
                cur_object_type = cur_kine_data['object_type']
            else:
                if 'taco' in data_inst_tag:
                    #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy
                    # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
                    self.taco_interped_fr_grab_tag = "ori_grab_s2_phone_call_1"
                    # self.taco_interped_data_sv_additional_tag = self.taco_interped_data_sv_additional_tag #  ""
                    if len(self.taco_interped_data_sv_additional_tag) == 0:
                        # traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2.npy'
                        traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_{data_inst_tag}_v2.npy'
                    else:
                        # traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                        traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_{data_inst_tag}_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                    if self.hand_type == 'allegro':
                        taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                        # kine_info_fn = os.path.join(taco_kine_sv_root, traj_kine_info)
                    else:
                        # kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
                        taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data'
                        # kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
                    # taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                    kine_info_fn = os.path.join(taco_kine_sv_root, traj_kine_info)
                else:
                    kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
                    kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
                # get he kinemati info file #
                
                if self.add_contact_conditions:
                    pure_kine_data_fn = kine_info_fn.split("/")[-1].split(".")[0] # split #
                    pure_contact_data_fn = pure_kine_data_fn + "_contact_flag.npy"
                    full_contact_data_fn = os.path.join(self.contact_info_sv_root, pure_contact_data_fn) # contact data fn # 
                    cur_inst_contact_data = np.load(full_contact_data_fn)
                
                
                pure_kine_object_type = parse_kine_data_fn_into_object_type(kine_info_fn)
                
                cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
                cur_object_type = data_inst_tag
                pure_cur_object_type = data_inst_tag.split("_nf_")[0]
            
            hand_qs = cur_kine_data['robot_delta_states_weights_np'] # weights -- kinematics qs #
            maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            hand_qs = hand_qs[:maxx_ws]
            
            obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
            
            # then segment the data_inst_tag to get the mesh file name #
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{pure_cur_object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            while len(to_sample_fr_idxes) < self.maxx_nn_pts:
                to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
        elif isinstance(data_inst_tag, tuple): # 
            obj_type, traj_obj_type = data_inst_tag
            
            if 'ori_grab' in obj_type:
                
                # if self.hand_type == 'leap':
                #     pure_obj_type = obj_type.split("_nf_")[0]
                #     traj_kine_info  = f"{self.tracking_info_st_tag}{pure_obj_type}.npy"
                # else:
                #     traj_kine_info  = f"{self.tracking_info_st_tag}{traj_obj_type}.npy"
                    
                traj_kine_info  = f"{self.tracking_info_st_tag}{traj_obj_type}.npy"
                
                
                traj_kine_info = os.path.join(self.tracking_save_info_fn, traj_kine_info)
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
                
                if self.add_contact_conditions:
                    pure_kine_data_fn = traj_kine_info.split("/")[-1].split(".")[0] # split #
                    pure_contact_data_fn = pure_kine_data_fn + "_contact_flag.npy"
                    full_contact_data_fn = os.path.join(self.contact_info_sv_root, pure_contact_data_fn) # contact data fn # 
                    cur_inst_contact_data = np.load(full_contact_data_fn)
                    
                    
                
                
                
                hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
                maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
                # maxx_ws = hand_qs.shape[0]
                hand_qs = hand_qs[:maxx_ws]
                
                obj_trans = traj_kine_info_data['object_transl'][:maxx_ws] # nn_ts x 3 
                obj_ornt = traj_kine_info_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
                # then segment the data_inst_tag to get the mesh file name #
                self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
                
                if not os.path.exists(self.grab_obj_mesh_sv_folder):
                    self.grab_obj_mesh_sv_folder = "../../tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
                
                pure_obj_type = obj_type.split("_nf_")[0]
                
                grab_mesh_fn = f"{pure_obj_type}.obj"
                grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
                # get the object mesh #
                obj_mesh = trimesh.load(grab_mesh_fn, force='mesh')
                obj_verts = obj_mesh.vertices # nn_pts x 3
                # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
                # random_sampled_idxes = np.random.sample()
                to_sample_fr_idxes = list(range(obj_verts.shape[0]))
                while len(to_sample_fr_idxes) < self.maxx_nn_pts:
                    to_sample_fr_idxes += list(range(obj_verts.shape[0]))
                random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
                random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
                obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            elif 'taco' in obj_type:
                #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy
                # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
                self.taco_interped_fr_grab_tag = "ori_grab_s2_phone_call_1"
                # self.taco_interped_data_sv_additional_tag = "" # zero out the addtional sv tag here #
                # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230919_021_v2_interpfr_60_interpfr2_60_nntrans_40.npy
                # if len(self.taco_interped_data_sv_additional_tag) == 0:
                #     traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2.npy'
                # else:
                #     traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                
                if len(self.taco_interped_data_sv_additional_tag) == 0:
                    # traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2.npy'
                    traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_{obj_type}_v2.npy'
                else:
                    # traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                    traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_{obj_type}_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                
                # taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data' # TACO tracking pk # 
                if self.hand_type == 'allegro':
                    taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                    # kine_info_fn = os.path.join(taco_kine_sv_root, traj_kine_info)
                else:
                    # kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
                    taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data'
                traj_kine_info = os.path.join(taco_kine_sv_root, traj_kine_info) # get kinematics sv root # kinematics data #
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                print(f"traj_kine_info: {traj_kine_info}")
                hand_qs = traj_kine_info_data['robot_delta_states_weights_np'] # delta states #
                maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
                hand_qs = hand_qs[:maxx_ws]
                
                obj_trans = traj_kine_info_data['object_transl'][:maxx_ws] # nn_ts x 3 
                obj_ornt = traj_kine_info_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
                # then segment the data_inst_tag to get the mesh file name #
                self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
                grab_mesh_fn = f"{obj_type}.obj"
                grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
                # get the object mesh #
                obj_mesh = trimesh.load(grab_mesh_fn, force='mesh')
                obj_verts = obj_mesh.vertices # nn_pts x 3
                # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
                # random_sampled_idxes = np.random.sample()
                to_sample_fr_idxes = list(range(obj_verts.shape[0]))
                while len(to_sample_fr_idxes) < self.maxx_nn_pts:
                    to_sample_fr_idxes += list(range(obj_verts.shape[0]))
                random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
                random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
                obj_verts = obj_verts[random_sampled_idxes]
            else:
                raise ValueError(f"Cannot parse the dataset type from obj_type: {obj_type}")
            # grab_mesh_fn = f"{data_inst_tag}.obj"
            # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
        else: #
            raise ValueError(f"Unrecognized grab_data_inst_type: {type(data_inst_tag)}")
            
        # hand_qs = cur_kine_data['robot_delta_states_weights_np'][:maxx_ws] # nn_ts x 22
        
        if self.glb_rot_use_quat:
            hand_qs_rot_np = hand_qs[..., 3:6]
            hand_qs_rot_th = torch.from_numpy(hand_qs_rot_np)
            hand_qs_rot_quat_th = quat_from_euler_xyz(hand_qs_rot_th[..., 0], hand_qs_rot_th[..., 1], hand_qs_rot_th[..., 2])
            hand_qs_rot_quat_np = hand_qs_rot_quat_th.numpy()
            hand_qs= np.concatenate(
                [hand_qs[..., :3], hand_qs_rot_quat_np, hand_qs[..., 6:]], axis=-1
            )
        
        cur_maxx_kine_traj_length = max(hand_qs.shape[0], max(obj_trans.shape[0], obj_ornt.shape[0]))
        self.maxx_trajectory_length = max(self.maxx_trajectory_length, cur_maxx_kine_traj_length)
        
        if cur_base_traj_fn is not None:
            cur_base_traj_data = np.load(cur_base_traj_fn, allow_pickle=True).item()
            base_traj_hand_qs = cur_base_traj_data['optimized_hand_qtars'][0]
            # self.tot_dof_targets = self.samples_with_object_code['optimized_hand_qtars']
            #         self.tot_dof_targets = self.tot_dof_targets[0] 
        else:
            base_traj_hand_qs = hand_qs[..., self.joint_idxes_inversed_ordering]
        
        kine_obj_rot_euler_angles = []
        for i_fr in range(obj_ornt.shape[0]):
            cur_rot_quat = obj_ornt[i_fr]
            cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True)
            kine_obj_rot_euler_angles.append(cur_rot_euler)
        kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
        
        # if self.compute_hand_rew_buf_threshold > hand
        
        kine_info_dict = { # 
            'obj_verts': obj_verts, 
            'hand_qs': hand_qs[..., self.joint_idxes_inversed_ordering],
            'base_traj_hand_qs': base_traj_hand_qs, # 
            'obj_trans': obj_trans,
            'obj_ornt': obj_ornt ,
            'obj_rot_euler': kine_obj_rot_euler_angles
        }
        if self.add_contact_conditions:
            kine_info_dict['contact_conditions'] = cur_inst_contact_data
        return kine_info_dict
    
    
    
    def _load_tracking_kine_info(self, ):
        # self.maxx_kine_nn_ts = 300
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        
        
        #### iterate over all data instance tag and load the kinematics info ####
        for i_inst, data_inst_tag in enumerate(self.data_inst_tag_list):
            print(f"[Loading tracking kine info] {i_inst}/{len(self.data_inst_tag_list)}: {data_inst_tag}")
            cur_base_traj_fn = self.data_base_traj[i_inst]
            kine_info_dict = self._load_single_tracking_kine_info(data_inst_tag, cur_base_traj_fn)
            
            cur_data_fn = self.data_list[i_inst]
            if cur_data_fn in self.data_name_to_kine_info:
                key_data_fn = cur_data_fn + str(i_inst)
            else:
                key_data_fn = cur_data_fn
            
            
            obj_trans, kine_obj_rot_euler_angles, hand_qs, obj_verts = kine_info_dict['obj_trans'], kine_info_dict['obj_rot_euler'], kine_info_dict['hand_qs'], kine_info_dict['obj_verts']
            
            # obj trans, obj rot euler angles, hand qs, obj verts #

            #### NOTE: down sampling #### # down sample #
            if self.downsample:
                if obj_trans.shape[0] > 200:
                    
                    base_traj_hand_qs = kine_info_dict['base_traj_hand_qs']
                    obj_ornt = kine_info_dict['obj_ornt']
                    
                    idxes = [ii for ii in range(obj_trans.shape[0]) if ii % 2 == 0]
                    idxes =  np.array(idxes, dtype=np.int32)
                    obj_trans = obj_trans[idxes]
                    kine_obj_rot_euler_angles = kine_obj_rot_euler_angles[idxes]
                    hand_qs = hand_qs[idxes]
                    base_traj_hand_qs = base_traj_hand_qs[idxes]
                    obj_ornt = obj_ornt[idxes]
                    print(f"down sampling, hand_qs: {hand_qs.shape}")
                    
                    kine_info_dict['obj_trans'] = obj_trans
                    kine_info_dict['obj_rot_euler'] = kine_obj_rot_euler_angles
                    kine_info_dict['hand_qs'] = hand_qs
                    kine_info_dict['base_traj_hand_qs'] = base_traj_hand_qs
                    kine_info_dict['obj_ornt'] = obj_ornt
                    
                    if self.add_contact_conditions:
                        cur_contact_info = kine_info_dict['kine_info_dict']
                        cur_contact_info = cur_contact_info[idxes]
                        # add the contact conditions #
                        kine_info_dict['contact_conditions'] = cur_contact_info
            
            # and cond #
            
            if self.w_franka:
                hand_qs = kine_info_dict['hand_qs']
                base_traj_hand_qs = kine_info_dict['base_traj_hand_qs']
                obj_trans = kine_info_dict['obj_trans']
                
                #### with franka  #####
                # hand_qs[..., 2] = hand_qs[..., 2] + 0.7
                # hand_qs[..., 0] = hand_qs[..., 0] + 0.6
                
                # base_traj_hand_qs[..., 2] = base_traj_hand_qs[..., 2] + 0.7
                # base_traj_hand_qs[..., 0] = base_traj_hand_qs[..., 0] + 0.6
                
                obj_trans[..., 2] = obj_trans[..., 2] + 0.7
                obj_trans[..., 0] = obj_trans[..., 0] + 0.6
                
                kine_info_dict['hand_qs'] = hand_qs
                kine_info_dict['base_traj_hand_qs'] = base_traj_hand_qs
                kine_info_dict['obj_trans'] = obj_trans
            
            kine_info_dict['data_inst_tag'] = data_inst_tag
            # kinematics info dictionary #
            self.data_name_to_kine_info[key_data_fn] = kine_info_dict

            tot_obj_transl.append(obj_trans)
            tot_obj_rot_euler.append(kine_obj_rot_euler_angles)
            tot_hand_qs.append(hand_qs)
            tot_object_verts.append(obj_verts)
            
            
            
            # use_multiple_kine_source_trajs # multiple kine source trajs fn; multiple kine srouce trajs fn # multiple kine source trajs fn #
            if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
                self.multiple_kine_source_trajs_tot = np.load(self.multiple_kine_source_trajs_fn, allow_pickle=True).item() # .item() # 
                
                if len(self.multiple_kine_source_trajs_tot['hand_qs'].shape) == 2:
                    for key in self.multiple_kine_source_trajs_tot:
                        self.multiple_kine_source_trajs_tot[key] = self.multiple_kine_source_trajs_tot[key][None]
                    self.multiple_kine_source_trajs_tot['hand_qs'] = self.multiple_kine_source_trajs_tot['hand_qs'][:, :, self.joint_idxes_inversed_ordering]
                        
                    
                
                
                self.multiple_kine_source_trajs = self.multiple_kine_source_trajs_tot['hand_qs']
                self.multiple_kine_source_obj_pos = self.multiple_kine_source_trajs_tot['obj_pos']
                
                wss = min(self.multiple_kine_source_trajs.shape[1], hand_qs.shape[0])
                self.multiple_kine_source_trajs = self.multiple_kine_source_trajs[:, :wss]
                self.multiple_kine_source_obj_pos = self.multiple_kine_source_obj_pos[:, :wss]
                
                print("multiple_kine_source_trajs:", self.multiple_kine_source_trajs.shape)
                
                self.multiple_kine_source_trajs = torch.from_numpy(self.multiple_kine_source_trajs).float().to(self.rl_device)
                self.multiple_kine_source_trajs = torch.cat(
                    [ torch.from_numpy(hand_qs).to(self.rl_device).unsqueeze(0), self.multiple_kine_source_trajs ], dim=0 # (nn_source_trajs + 1, nn_hand_dofs) # expand the traj sources #
                )
                self.multiple_kine_source_obj_pos = torch.from_numpy(self.multiple_kine_source_obj_pos).float().to(self.rl_device)
                self.multiple_kine_source_obj_pos = torch.cat(
                    [ torch.from_numpy(obj_trans).to(self.rl_device).unsqueeze(0),  self.multiple_kine_source_obj_pos ], dim=0
                )
                
                
                if 'obj_rot' in self.multiple_kine_source_trajs_tot:
                    self.multiple_kine_source_obj_rot = self.multiple_kine_source_trajs_tot['obj_rot']
                    self.multiple_kine_source_obj_rot = self.multiple_kine_source_obj_rot[:, :wss]
                    self.multiple_kine_source_obj_rot = torch.from_numpy(self.multiple_kine_source_obj_rot).float().to(self.rl_device)
                    self.multiple_kine_source_obj_rot = torch.cat(
                        [ torch.from_numpy(kine_info_dict['obj_ornt']).to(self.rl_device).unsqueeze(0),  self.multiple_kine_source_obj_rot ], dim=0
                    )
                else:
                    self.multiple_kine_source_obj_rot = None
                    
                # self.multiple_kine_source_obj_rot = None
                
                
                self.envs_kine_source_trajs_idxes = torch.tensor(
                    [0] * self.num_envs, dtype=torch.long
                ).to(self.rl_device) # (nn_envs, ) #
                
                
            # get the expert #
    # prepare expert #
    def _prepare_expert_traj_infos(self,):
        
        
        # expert trajectories #
        tot_data_fns = self.data_name_to_data.keys()
        tot_data_fns = sorted(tot_data_fns) # expert ## expert 
        self.data_fn_to_data_index = {}
        self.maxx_episode_length_per_traj = []
        # kine_info_dict = {
        #     'obj_verts': obj_verts, # solve an problem better #
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 # the foreacasted weights #
        #     'obj_trans': obj_trans, # obj verts; obj trans # 
        #     'obj_ornt': obj_ornt , # the eueler angles # the velocity guided problems? #
        #     'obj_rot_euler': kine_obj_rot_euler_angles
        # 
        # } 
        # 
        tot_hand_qs = []
        tot_hand_qtars = []
        tot_kine_qs = []
        tot_kine_obj_trans = []
        tot_kine_obj_ornt = []
        tot_kine_obj_rot_euler = []
        tot_obj_codes = []
        tot_hand_actions = []
        tot_hand_preopt_res =[]
        tot_base_traj_hand_qs = []
        tot_contact_infos = []
        tot_obj_pts = []
        
        tot_data_inst_tags = []
        
        # data name to kine info #
        for i_inst, cur_data_fn in enumerate(tot_data_fns):
            # use_clip_glb_features
            self.data_fn_to_data_index[cur_data_fn] = i_inst # i_inst, cur_data_fn # cur data fn # # try to 
            cur_tracking_data = self.data_name_to_data[cur_data_fn]
            cur_kine_data = self.data_name_to_kine_info[cur_data_fn] # get the kine data #
            # laod the trakcing results # 
            cur_hand_qs = cur_tracking_data['hand_qs']
            cur_hand_qtars = cur_tracking_data['hand_qtars']
            cur_kine_qs = cur_kine_data['hand_qs']
            cur_data_inst_tag = cur_kine_data['data_inst_tag']
            
            # forecast #
            if cur_hand_qtars.shape[1] != cur_kine_qs.shape[1]:
                cur_hand_qtars  = cur_kine_qs.copy()
                cur_hand_qs  = cur_kine_qs.copy()
            
            cur_kine_obj_trans = cur_kine_data['obj_trans']
            cur_kine_obj_ornt = cur_kine_data['obj_ornt']
            cur_kine_obj_rot_euler = cur_kine_data['obj_rot_euler']
            cur_base_traj_hand_qs = cur_kine_data['base_traj_hand_qs']
            
            
            
            print(f"cur_kine_qs: {cur_kine_qs.shape}")
            cur_data_episode_length = min( [ cur_hand_qs.shape[0], cur_hand_qtars.shape[0], cur_kine_qs.shape[0], cur_kine_obj_trans.shape[0], cur_kine_obj_ornt.shape[0], cur_kine_obj_rot_euler.shape[0] , cur_base_traj_hand_qs.shape[0] ])
            # self.maxx_episode_length_per_traj.append(cur_data_episode_length - 1)
            
            cur_traj_length = min(cur_data_episode_length, self.maxx_trajectory_length - 1)
            # self.maxx_episode_length_per_traj.append(self.maxx_trajectory_length - 1)
            self.maxx_episode_length_per_traj.append(cur_traj_length)
            
            # 
            # 
            # tot_dof_targets # online policy training v.s. offline policy training #
            # 
            
            if self.tot_dof_targets is not None:
                cur_hand_dof_targets = self.tot_dof_targets
                if cur_hand_dof_targets.shape[0] < self.maxx_trajectory_length:
                    cur_hand_dof_targets = np.concatenate(
                        [ cur_hand_dof_targets, np.zeros((self.maxx_trajectory_length - cur_hand_dof_targets.shape[0], cur_hand_dof_targets.shape[-1] ), dtype=np.float32) ], axis=0
                    )
                elif cur_hand_dof_targets.shape[0] > self.maxx_trajectory_length:
                    cur_hand_dof_targets = cur_hand_dof_targets[:self.maxx_trajectory_length]
                print(f"cur_hand_dof_targets: {cur_hand_dof_targets.shape}, cur_data_episode_length: {cur_data_episode_length}")
                tot_hand_preopt_res.append(cur_hand_dof_targets)
            
            if cur_base_traj_hand_qs.shape[0] < self.maxx_trajectory_length:
                cur_base_traj_hand_qs = np.concatenate(
                    [ cur_base_traj_hand_qs, np.zeros((self.maxx_trajectory_length - cur_base_traj_hand_qs.shape[0], cur_base_traj_hand_qs.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_base_traj_hand_qs.shape[0] > self.maxx_trajectory_length:
                cur_base_traj_hand_qs = cur_base_traj_hand_qs[:self.maxx_trajectory_length]
                
            
            if self.add_contact_conditions:
                contact_conditions = cur_kine_data['contact_conditions']
                if contact_conditions.shape[0]  < self.maxx_trajectory_length:
                    contact_conditions = np.concatenate(
                        [ contact_conditions, np.zeros((self.maxx_trajectory_length - contact_conditions.shape[0], contact_conditions.shape[-1]), dtype=np.float32) ], axis=0
                    )
                elif contact_conditions.shape[0] > self.maxx_trajectory_length:
                    contact_conditions = contact_conditions[:   self.maxx_trajectory_length] # maxx trajectory length 
                tot_contact_infos.append(contact_conditions)
            
            if cur_hand_qs.shape[0] < self.maxx_trajectory_length:
                cur_hand_qs = np.concatenate(
                    [ cur_hand_qs, np.zeros((self.maxx_trajectory_length - cur_hand_qs.shape[0], cur_hand_qs.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_hand_qs.shape[0] > self.maxx_trajectory_length:
                cur_hand_qs = cur_hand_qs[:self.maxx_trajectory_length]
                
            if cur_hand_qtars.shape[0] < self.maxx_trajectory_length:
                cur_hand_qtars = np.concatenate(
                    [ cur_hand_qtars, np.zeros((self.maxx_trajectory_length - cur_hand_qtars.shape[0], cur_hand_qtars.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_hand_qtars.shape[0] > self.maxx_trajectory_length:
                cur_hand_qtars = cur_hand_qtars[:self.maxx_trajectory_length]
                
            if cur_kine_qs.shape[0] < self.maxx_trajectory_length:
                cur_kine_qs = np.concatenate(
                    [ cur_kine_qs, np.zeros((self.maxx_trajectory_length - cur_kine_qs.shape[0], cur_kine_qs.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_kine_qs.shape[0] > self.maxx_trajectory_length:
                cur_kine_qs = cur_kine_qs[:self.maxx_trajectory_length]
                
            if cur_kine_obj_trans.shape[0] < self.maxx_trajectory_length:
                cur_kine_obj_trans = np.concatenate(
                    [ cur_kine_obj_trans, np.zeros((self.maxx_trajectory_length - cur_kine_obj_trans.shape[0], cur_kine_obj_trans.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_kine_obj_trans.shape[0] > self.maxx_trajectory_length:
                cur_kine_obj_trans = cur_kine_obj_trans[:self.maxx_trajectory_length]
                
            if cur_kine_obj_ornt.shape[0] < self.maxx_trajectory_length:
                cur_kine_obj_ornt = np.concatenate(
                    [ cur_kine_obj_ornt, np.zeros((self.maxx_trajectory_length - cur_kine_obj_ornt.shape[0], cur_kine_obj_ornt.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_kine_obj_ornt.shape[0] > self.maxx_trajectory_length:
                cur_kine_obj_ornt = cur_kine_obj_ornt[:self.maxx_trajectory_length]
                
            if cur_kine_obj_rot_euler.shape[0] < self.maxx_trajectory_length:
                cur_kine_obj_rot_euler = np.concatenate(
                    [ cur_kine_obj_rot_euler, np.zeros((self.maxx_trajectory_length - cur_kine_obj_rot_euler.shape[0], cur_kine_obj_rot_euler.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_kine_obj_rot_euler.shape[0] > self.maxx_trajectory_length:
                cur_kine_obj_rot_euler = cur_kine_obj_rot_euler[:self.maxx_trajectory_length]
            
            if 'actions' in cur_tracking_data:
                cur_hand_actions = cur_tracking_data['actions']
                if cur_hand_actions.shape[0] < self.maxx_trajectory_length:
                    cur_hand_actions = np.concatenate(
                        [ cur_hand_actions, np.zeros((self.maxx_trajectory_length - cur_hand_actions.shape[0], cur_hand_actions.shape[-1]), dtype=np.float32) ], axis=0
                    )
                elif cur_hand_actions.shape[0] > self.maxx_trajectory_length:
                    cur_hand_actions = cur_hand_actions[:self.maxx_trajectory_length]
                
                tot_hand_actions.append(cur_hand_actions) # cur hand tracking actions #
                
            cur_obj_pts = cur_kine_data['obj_verts']
            
            tot_obj_pts.append(cur_obj_pts)
                
            tot_hand_qs.append(cur_hand_qs)
            tot_hand_qtars.append(cur_hand_qtars)
            tot_kine_qs.append(cur_kine_qs)
            tot_kine_obj_trans.append(cur_kine_obj_trans)
            tot_kine_obj_ornt.append(cur_kine_obj_ornt)
            tot_kine_obj_rot_euler.append(cur_kine_obj_rot_euler)
            
            cur_obj_code = self.data_name_to_object_code[cur_data_fn]
            tot_obj_codes.append(cur_obj_code)
            tot_base_traj_hand_qs.append(cur_base_traj_hand_qs)
        
            # if self.use_clip_glb_features:
            # cur_pure_inst_tag = cur_data_fn.split("/")[-1].split(".npy")[0].split("_nf_")[0].split("_")[6:8]
            # cur_pure_inst_tag = cur_data_fn.split("/")[-1].split(".npy")[0].split("_nf_")[0].split("_")[5:8]
            # ori_grab_s2_cubesmall_lift_nf_300
            # cur_pure_inst_tag = cur_data_inst_tag.split("_")[3:5]
            if isinstance(cur_data_inst_tag, tuple):
                cur_data_inst_tag = cur_data_inst_tag[0]
            cur_pure_inst_tag = cur_data_inst_tag.split("_")[2:5]
            cur_pure_inst_tag = " ".join(cur_pure_inst_tag)
            print(f"cur_pure_inst_tag: {cur_pure_inst_tag }, cur_data_fn: {cur_data_fn}")
            tot_data_inst_tags.append(cur_pure_inst_tag)
        
        tot_hand_qs = np.stack(tot_hand_qs, axis=0)
        tot_hand_qtars = np.stack(tot_hand_qtars, axis=0)
        tot_kine_qs = np.stack(tot_kine_qs, axis=0)
        tot_kine_obj_trans = np.stack(tot_kine_obj_trans, axis=0)
        tot_kine_obj_ornt = np.stack(tot_kine_obj_ornt, axis=0)
        tot_kine_obj_rot_euler = np.stack(tot_kine_obj_rot_euler, axis=0)
        tot_base_traj_hand_qs = np.stack(tot_base_traj_hand_qs, axis=0)
        self.tot_base_traj_hand_qs = torch.from_numpy(tot_base_traj_hand_qs).float().to(self.rl_device)
        self.tot_hand_qs = torch.from_numpy(tot_hand_qs).float().to(self.rl_device)
        self.tot_hand_qtars = torch.from_numpy(tot_hand_qtars).float().to(self.rl_device)
        self.tot_kine_qs = torch.from_numpy(tot_kine_qs).float().to(self.rl_device) # kine qs #
        self.tot_kine_obj_trans = torch.from_numpy(tot_kine_obj_trans).float().to(self.rl_device)
        self.tot_kine_obj_ornt = torch.from_numpy(tot_kine_obj_ornt).float().to(self.rl_device)
        self.tot_kine_obj_rot_euler = torch.from_numpy(tot_kine_obj_rot_euler).float().to(self.rl_device)
        self.maxx_episode_length_per_traj = np.array(self.maxx_episode_length_per_traj, dtype=np.int32)
        self.maxx_episode_length_per_traj = torch.from_numpy(self.maxx_episode_length_per_traj).to(self.rl_device)
        self.tot_obj_pts = np.stack(tot_obj_pts, axis=0)
        self.tot_obj_pts = torch.from_numpy(self.tot_obj_pts).float().to(self.rl_device)
        
        # tot_data_inst_tags # 
        # if self.use_clip_glb_features:
        text_inputs = clip.tokenize(tot_data_inst_tags).to(self.rl_device)
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        # text_features = text_features[0].cpu().numpy()
        # sv_dict['text_features'] = text_features
        self.tot_text_features = text_features
    
    
        
        if self.add_contact_conditions:
            tot_contact_infos = np.stack(tot_contact_infos, axis=0)
            self.tot_contact_infos = torch.from_numpy(tot_contact_infos).float().to(self.rl_device)
        
        
        if len(tot_hand_preopt_res) > 0:
            tot_hand_preopt_res = np.stack(tot_hand_preopt_res, axis=0)
            self.tot_hand_preopt_res = torch.from_numpy(tot_hand_preopt_res).float().to(self.rl_device)
            
        else:
            self.tot_hand_preopt_res = self.tot_base_traj_hand_qs.clone()
        print(f"tot_hand_preopt_res: {self.tot_hand_preopt_res.size()}, tot_hand_qs: {self.tot_hand_qs.size()}, tot_hand_qtars: {self.tot_hand_qtars.size()}, tot_kine_qs: {self.tot_kine_qs.size()}, tot_kine_obj_trans: {self.tot_kine_obj_trans.size()}, tot_kine_obj_ornt: {self.tot_kine_obj_ornt.size()}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj}")
        
        
        
        
        if len(tot_hand_actions) > 0: # tot hand actions #
            tot_hand_actions = np.stack(tot_hand_actions, axis=0)
            self.tot_hand_actions = torch.from_numpy(tot_hand_actions).float().to(self.rl_device)
        else:
            self.tot_hand_actions = None
        
        self.tot_obj_codes = tot_obj_codes
        
        ##### Prepare for the full trajectory information arrays #####
        self.ori_tot_hand_preopt_res = self.tot_hand_preopt_res.clone()
        self.ori_tot_hand_qs = self.tot_hand_qs.clone()
        self.ori_tot_hand_qtars = self.tot_hand_qtars.clone()
        self.ori_tot_kine_qs = self.tot_kine_qs.clone()
        self.ori_tot_kine_obj_trans = self.tot_kine_obj_trans.clone()
        self.ori_tot_kine_obj_ornt =self.tot_kine_obj_ornt.clone()
        self.ori_tot_kine_obj_rot_euler = self.tot_kine_obj_rot_euler.clone()
        
        pass
       # allegro hand tracking #





    def _calculate_obj_vels(self, obj_trans, obj_rot_quat):
        mocap_freq = 120
        mocap_dt = float(1) / float(mocap_freq)
        obj_lin_vels = []
        obj_ang_vels = []
        for i_fr in range(obj_trans.shape[0]):
            nex_fr = i_fr + 1
            if nex_fr < obj_trans.shape[0]:
                cur_fr_trans = obj_trans[i_fr]
                nex_fr_trans = obj_trans[nex_fr]
                obj_lin_vel = (nex_fr_trans - cur_fr_trans) / float(mocap_dt)
                obj_lin_vels.append(obj_lin_vel)

                cur_fr_rot_euler = R.from_quat(obj_rot_quat[i_fr]).as_euler('xyz', degrees=False)
                nex_fr_rot_euler = R.from_quat(obj_rot_quat[nex_fr]).as_euler('xyz', degrees=False)
                obj_rot_vel = (nex_fr_rot_euler - cur_fr_rot_euler) / float(mocap_dt)
                obj_ang_vels.append(obj_rot_vel)
        obj_lin_vels.append(obj_lin_vels[-1])
        obj_ang_vels.append(obj_ang_vels[-1])
        obj_lin_vels = np.stack(obj_lin_vels, axis=0)
        obj_ang_vels = np.stack(obj_ang_vels, axis=0)
        return obj_lin_vels, obj_ang_vels
            
    def _find_grasp_frame(self, obj_transl, obj_ornt):
        # # def find_grasp_frame_from_mocap_data(mocap_data_fn):
        # data_dict = np.load(mocap_data_fn, allow_pickle=True).item()
        # hand_qs = data_dict['robot_delta_states_weights_np']
        # obj_transl = data_dict['object_transl'][:]
        # obj_ornt = data_dict['object_rot_quat'][: ]
        # 
        # nn_frames x 3 #
        # nn_frames x 4 #
        
        eps = 1e-2
        # # if the hand is close to the object --- add the hand pose guidance? #
        cur_grasp_fr = 0
        for cur_grasp_fr in range(0, obj_transl.shape[0] - 1):
            cur_fr_transl = obj_transl[cur_grasp_fr]
            cur_fr_ornt = obj_ornt[cur_grasp_fr]
            # print(f"cur_fr_transl: {cur_fr_transl}")
            nex_fr_transl = obj_transl[cur_grasp_fr + 1]
            nex_fr_ornt = obj_ornt[cur_grasp_fr + 1]
            diff_cur_nex_transl = np.linalg.norm(nex_fr_transl - cur_fr_transl)
            
            cur_fr_rot_euler = R.from_quat(cur_fr_ornt).as_euler('xyz', degrees=False)
            nex_fr_rot_euler = R.from_quat(nex_fr_ornt).as_euler('xyz', degrees=False)
            diff_cur_nex_rot = np.linalg.norm(nex_fr_rot_euler - cur_fr_rot_euler)
            if diff_cur_nex_transl > eps or diff_cur_nex_rot > eps:
                break
        return cur_grasp_fr 
    
    # run tracking headless test pool #
    # run jracking selfexp pool #
    # 
    
    def _load_optimized_traj_diffusion_samples(self, optimized_traj_fn):
        isaac_sv_info = np.load(optimized_traj_fn, allow_pickle=True).item()
        key = 'samples'
        # key = 'closest_training_data'
        sv_info = isaac_sv_info[key]
        ts_to_hand_qs_np = sv_info[0] ## nn_ts x nn_hand_dofs # 
        # ts_to_hand_qs_np = np.concatenate(
        #     [ self.pre_hand_qs[] ]
        # )
        print(f"loading from samples")
        ts_to_hand_qs_np = ts_to_hand_qs_np[:, self.joint_idxes_ordering]
        ts_to_hand_qs_np = np.concatenate(
            [ self.pre_hand_qs[0:1], ts_to_hand_qs_np ], axis=0
        )
        return ts_to_hand_qs_np
        
    
    def _load_optimized_traj_sorted_qtars(self, optimized_traj_fn):
        isaac_sv_info = np.load(optimized_traj_fn, allow_pickle=True).item()
        # tot_ts_list = list(isaac_sv_info.keys())
        # tot_ts_list = sorted(tot_ts_list)
        first_ts_isaac_sv_info = isaac_sv_info[0]
        assert 'shadow_hand_dof_tars' in first_ts_isaac_sv_info
        # if 'shadow_hand_dof_tars' in first_ts_isaac_sv_info:
        if '_sorted.npy' not in optimized_traj_fn:
            optimized_traj_fn = optimized_traj_fn.replace('.npy', '_sorted.npy')
            isaac_sv_info = np.load(optimized_traj_fn, allow_pickle=True).item()
        optimized_hand_qtars = isaac_sv_info['optimized_hand_qtars'] #### nn_envs x nn_ts x nn_hand_dofs 
        optimized_hand_qtars = optimized_hand_qtars[0]
        # ts_to_hand_qs_np = np.stack(ts_to_hand_qs_np, )
        ts_to_hand_qs_np = optimized_hand_qtars
        ts_to_hand_qs_np = ts_to_hand_qs_np[:, self.joint_idxes_ordering]
        return ts_to_hand_qs_np
        
    
    def _load_optimized_traj(self, optimized_traj_fn):
        print(f"Loading pre-optimized trajectory from {optimized_traj_fn}")
        isaac_sv_info = np.load(optimized_traj_fn, allow_pickle=True).item()
        
        
        if 'samples' in isaac_sv_info:
            ts_to_hand_qs_np = self._load_optimized_traj_diffusion_samples(optimized_traj_fn)
            return ts_to_hand_qs_np
            
        
        first_ts_sv_info = isaac_sv_info[0]
        if 'ts_to_hand_obj_obs_reset' not in optimized_traj_fn.split("/")[-1] and 'shadow_hand_dof_tars' in first_ts_sv_info:
            ts_to_hand_qs_np = self._load_optimized_traj_sorted_qtars(optimized_traj_fn)
            return ts_to_hand_qs_np
        
        
        
        tot_ts_list = list(isaac_sv_info.keys())
        tot_ts_list = sorted(tot_ts_list)
        minn_ts = min(tot_ts_list)
        maxx_ts = max(tot_ts_list)
        # for each val -- nn_envs x xxxx #
        ts_to_hand_qs = {}
        ts_to_obj_qs = {}
        idxx = 1000
        maxx_lowest_z = -9999.0
        
        for idx in range(isaac_sv_info[maxx_ts]['object_pose'].shape[0]):
            cur_last_z = isaac_sv_info[maxx_ts]['object_pose'][idx][2]
            if cur_last_z > maxx_lowest_z and cur_last_z < 2.0:
                maxx_lowest_z = cur_last_z
                idxx = idx
        print(idxx, maxx_lowest_z)
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        
        # self.joint_idxes_ordering = to
        
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        print(f"joint_idxes_ordering: {joint_idxes_ordering}")
        print(f"joint_idxes_inversed_ordering: {joint_idxes_inversed_ordering}")
        
        # idxx = 3 #
        tot_obj_rot = [] 
        for i_ts in tot_ts_list:
            cur_ts_sv_info = isaac_sv_info[i_ts]
            cur_ts_shadow_hand_dof_pos = cur_ts_sv_info['shadow_hand_dof_pos']
            cur_ts_object_pose = cur_ts_sv_info['object_pose'] #
            cur_ts_shadow_hand_dof_pos = cur_ts_shadow_hand_dof_pos[idxx]
            cur_ts_object_pose = cur_ts_object_pose[idxx]
            cur_ts_obj_trans = cur_ts_object_pose[:3]
            cur_ts_obj_rot = cur_ts_object_pose[3:]
            # get #
            ts_to_hand_qs[i_ts - minn_ts] = cur_ts_shadow_hand_dof_pos[joint_idxes_ordering]
            ts_to_obj_qs[i_ts - minn_ts] = cur_ts_object_pose
        sorted_ts = sorted(list(ts_to_hand_qs.keys()))
        ts_to_hand_qs_np = [
            ts_to_hand_qs[i_ts] for i_ts in sorted_ts
        ]
        ts_to_hand_qs_np = np.stack(ts_to_hand_qs_np, axis=0)
        return ts_to_hand_qs_np
    
    def _find_lifting_frame(self, object_transl):
        # 
        # 
        lift_trans_z_thres = 0.05
        if self.strict_lifting_separate_stages:
            lift_trans_z_thres = 0.10
        lift_fr = 0
        for i_fr in range(object_transl.shape[0]):
            cur_obj_transl_z= object_transl[i_fr][2].item()
            if cur_obj_transl_z > lift_trans_z_thres:
                lift_fr = i_fr
                print(f"lift_fr: {lift_fr}, cur_obj_transl: {object_transl[lift_fr]}")
                break
        if lift_fr == 0:
            lift_fr =object_transl.shape[0] - 1
        return lift_fr
        pass
    
    
    
    
    
    def _load_mocap_info(self,):
        print(f"==> Loading mocap reference information from {self.mocap_sv_info_fn}")
        save_info = np.load(self.mocap_sv_info_fn, allow_pickle=True).item()
        
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        self.joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        
        # mocap info #
        hand_qs = save_info['robot_delta_states_weights_np'][self.start_frame : ]
        hand_qs = hand_qs[: , : self.nn_hand_dof]
        goal_obj_trans = save_info['object_transl'][: ][self.start_frame : ]
        goal_obj_rot_quat = save_info['object_rot_quat'][: ][self.start_frame : ]
        
        
        self.pre_hand_qs = hand_qs.copy()
        goal_hand_qs = hand_qs.copy()
        
        
        if self.dataset_type == 'taco':
            
            # x_offset = -0.05
            # y_offset = -0.01
            # goal_obj_trans[:, 0] = goal_obj_trans[:, 0] + x_offset
            # goal_obj_trans[:, 1] = goal_obj_trans[:, 1] + y_offset
            
            ed_frame = min(hand_qs.shape[0], self.max_episode_length)
            hand_qs = hand_qs[: ed_frame    ]
            goal_obj_trans = goal_obj_trans[: ed_frame]
            goal_obj_rot_quat = goal_obj_rot_quat[: ed_frame]
        
        # # get the tasks and the taco #
        # if 'taco' in self.object_name and 'TACO' not in self.mocap_sv_info_fn and self.use_taco_obj_traj:
        #     # then we need to repose all related trajectories and also interpolate between all related trajectories #
        #     obj_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{self.object_name}_zrot_3.141592653589793_modifed_interped.npy'
        #     obj_mocap_info = np.load(obj_mocap_info_fn , allow_pickle=True ).item()
        #     goal_obj_trans = obj_mocap_info['object_transl'][: ][20 : ]
        #     goal_obj_rot_quat = obj_mocap_info['object_rot_quat'][: ][20 : ]
            
        #     cur_ws = min(goal_obj_trans.shape[0], hand_qs.shape[0])
        #     hand_qs = hand_qs[: cur_ws]
        #     goal_obj_trans = goal_obj_trans[: cur_ws]
        #     goal_obj_rot_quat = goal_obj_rot_quat[: cur_ws]
        
        # if self.dataset_type == 'taco':
        #     # link_name_to_poses # 
        #     link_name_to_poses = save_info['link_name_to_poses']
        #     self.link_name_to_poses = link_name_to_poses
            
        #     for link_name in self.link_name_to_poses: 
        #         self.link_name_to_poses[link_name][:, 2] -= self.ground_distance
        #         if self.add_table:
        #             self.link_name_to_poses[link_name][:, 2] += self.table_z_dim
            
        #     self.link_name_to_poses_th = {}
        #     for link_name in self.link_name_to_poses:
        #         ##### get the link name to poses #####
        #         self.link_name_to_poses_th[link_name] = torch.from_numpy(self.link_name_to_poses[link_name]).float().to(self.rl_device)
            
        #     if self.hand_type == 'allegro':
        #         self.hand_palm_link_name = 'palm_link'
        #         self.thumb_tip_link_name = 'link_15_tip'
        #         self.index_tip_link_name = 'link_3_tip'
        #         self.middle_tip_link_name = 'link_7_tip'
        #         self.ring_tip_link_name = 'link_11_tip'
        #     elif self.hand_type == 'leap':
        #         # body_names = { # leap fingertips #
        #         #     'palm': 'palm_lower',
        #         #     'thumb': 'thumb_tip_head',
        #         #     'index': 'index_tip_head',
        #         #     'middle': 'middle_tip_head',
        #         #     'ring': 'ring_tip_head',
        #         # }
        #         # body_names = { # leap fingertips #
        #         #     'palm': 'palm_lower',
        #         #     'thumb': 'thumb_fingertip',
        #         #     'index': 'fingertip',
        #         #     'middle': 'fingertip_2',
        #         #     'ring': 'fingertip_3',
        #         # }
        #         # self.hand_palm_link_name = 'palm_lower'
        #         # self.thumb_tip_link_name = 'thumb_tip_head'
        #         # self.index_tip_link_name = 'index_tip_head'
        #         # self.middle_tip_link_name = 'middle_tip_head'
        #         # self.ring_tip_link_name = 'ring_tip_head'
        #         self.hand_palm_link_name = 'palm_lower'
        #         self.thumb_tip_link_name = 'thumb_fingertip'
        #         self.index_tip_link_name = 'fingertip'
        #         self.middle_tip_link_name = 'fingertip_2'
        #         self.ring_tip_link_name = 'fingertip_3'
        #     self.hand_palm_world_poses = self.link_name_to_poses_th[self.hand_palm_link_name]  
        #     self.thumb_tip_world_poses = self.link_name_to_poses_th[self.thumb_tip_link_name]
        #     self.index_tip_world_poses = self.link_name_to_poses_th[self.index_tip_link_name]
        #     self.middle_tip_world_poses = self.link_name_to_poses_th[self.middle_tip_link_name]
        #     self.ring_tip_world_poses = self.link_name_to_poses_th[self.ring_tip_link_name]
            
        
        
        self.lift_fr = self._find_lifting_frame( goal_obj_trans  )
        self.lift_obj_pos = goal_obj_trans[self.lift_fr]
        self.lift_obj_pos_th = torch.from_numpy(self.lift_obj_pos).float().to(self.rl_device) 
        
        
        # TODO: in the new train pool file #
        # TODO: in the new train pool file, for each obj type --- find its neighbouring trajectory types; load the traj to optimized res; for each neighbouring traj, set th preoptimized traj to that traj #
        
        # if self.pre_optimized_traj is not None and len(self.pre_optimized_traj) > 0 and os.path.exists(self.pre_optimized_traj):
        #     hand_qs = self._load_optimized_traj(self.pre_optimized_traj)
        #     currr_ws = min(hand_qs.shape[0], goal_obj_trans.shape[0])
        #     currr_ws = min(currr_ws, goal_obj_rot_quat.shape[0])
        #     hand_qs = hand_qs[: currr_ws]
        #     goal_obj_trans = goal_obj_trans[: currr_ws]
        #     goal_obj_rot_quat = goal_obj_rot_quat[: currr_ws]
        #     # goal_hand_qs = self.pre_hand_qs.copy()[: currr_ws]
        #     goal_hand_qs = goal_hand_qs[: currr_ws]
        #     pass
        
        
        
        
        goal_hand_qs[:, 2] -= self.ground_distance
        hand_qs[:, 2] -= self.ground_distance
        goal_obj_trans[:, 2] -= self.ground_distance
        
        
        ## TODO: reset the table's initial translations ##
        ## offset the hand qs ###
        if self.add_table:
            if not self.w_franka:
                goal_hand_qs[:, 2] += self.table_z_dim
                hand_qs[:, 2] += self.table_z_dim
                goal_obj_trans[:, 2] += self.table_z_dim
        
        self.goal_hand_qs = goal_hand_qs # get the goal hand qs #
        self.hand_qs = hand_qs
        self.goal_obj_trans = goal_obj_trans
        self.goal_obj_rot_quat = goal_obj_rot_quat
        print(f"==> Info loaded with hand_qs: {hand_qs.shape}, goal_hand_qs: {goal_hand_qs.shape}, goal_obj_trans: {goal_obj_trans.shape}, goal_obj_rot_quat: {goal_obj_rot_quat.shape}")
        
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        print(f"joint_idxes_ordering: {joint_idxes_ordering}")
        print(f"joint_idxes_inversed_ordering: {joint_idxes_inversed_ordering}")
        
        
        
        self.goal_hand_qs_th = torch.from_numpy(goal_hand_qs[:, joint_idxes_inversed_ordering]).float().to(self.rl_device)
        self.hand_qs_th = torch.from_numpy(hand_qs[:, joint_idxes_inversed_ordering]).float().to(self.rl_device)
        self.goal_obj_trans_th = torch.from_numpy(goal_obj_trans).float().to(self.rl_device)
        self.goal_obj_rot_quat_th = torch.from_numpy(goal_obj_rot_quat).float().to(self.rl_device)
        
        self.hand_qs = self.hand_qs[:, joint_idxes_inversed_ordering]
        self.goal_hand_qs = self.goal_hand_qs[:, joint_idxes_inversed_ordering]
        
        
        
        if self.use_twostage_rew:
            self.cur_grasp_fr = self._find_grasp_frame(goal_obj_trans, goal_obj_rot_quat)
        
        
        # start grasping fr # 
        if self.start_grasping_fr:
            self.cur_grasp_fr = self._find_grasp_frame(goal_obj_trans, goal_obj_rot_quat)
            print(f"cur_grasp_fr: {self.cur_grasp_fr}")
            if self.control_freq_inv == 1:
                self.cur_grasp_fr = self.cur_grasp_fr + 5 # the grasp fr #
            # self.cur_grasp_fr # self.cur_grasp_fr #
            
            self.hand_qs = self.hand_qs[self.cur_grasp_fr: ]
            self.hand_qs_th = self.hand_qs_th[self.cur_grasp_fr: ]
            self.goal_obj_trans_th = self.goal_obj_trans_th[self.cur_grasp_fr: ]
            self.goal_obj_rot_quat_th = self.goal_obj_rot_quat_th[self.cur_grasp_fr: ]
            goal_obj_trans = goal_obj_trans[self.cur_grasp_fr: ]
            goal_obj_rot_quat = goal_obj_rot_quat[self.cur_grasp_fr: ]
            self.goal_obj_trans = self.goal_obj_trans[self.cur_grasp_fr: ]
            self.goal_obj_rot_quat = self.goal_obj_rot_quat[self.cur_grasp_fr: ]
            
            if self.control_freq_inv > 1:
                prev_hand_qs = [self.hand_qs[0][None] for _ in range(self.cur_grasp_fr)]
                prev_hand_qs_th = self.hand_qs_th[0].unsqueeze(0).repeat(self.cur_grasp_fr, 1)
                prev_goal_obj_trans_th = self.goal_obj_trans_th[0].unsqueeze(0).repeat(self.cur_grasp_fr, 1)
                prev_goal_obj_rot_quat_th = self.goal_obj_rot_quat_th[0].unsqueeze(0).repeat(self.cur_grasp_fr, 1)
                prev_goal_obj_trans = [goal_obj_trans[0][None] for _ in range(self.cur_grasp_fr)]
                prev_goal_obj_rot_quat = [goal_obj_rot_quat[0][None] for _ in range(self.cur_grasp_fr)]
                prev_self_goal_obj_trans = [self.goal_obj_trans[0][None] for _ in range(self.cur_grasp_fr)]
                prev_self_goal_obj_rot_quat = [self.goal_obj_rot_quat[0][None] for _ in range(self.cur_grasp_fr)]
                prev_hand_qs = np.concatenate(prev_hand_qs, axis=0)
                # prev_hand_qs_th = torch.cat(prev_hand_qs_th, dim=0)
                # prev_goal_obj_trans_th = torch.cat(prev_goal_obj_trans_th, dim=0)
                # prev_goal_obj_rot_quat_th = torch.cat(prev_goal_obj_rot_quat_th, dim=0)
                prev_goal_obj_trans = np.concatenate(prev_goal_obj_trans, axis=0)
                prev_goal_obj_rot_quat = np.concatenate(prev_goal_obj_rot_quat, axis=0)
                prev_self_goal_obj_trans = np.concatenate(prev_self_goal_obj_trans, axis=0)
                prev_self_goal_obj_rot_quat = np.concatenate(prev_self_goal_obj_rot_quat, axis=0)
                
                self.hand_qs = np.concatenate(
                    [prev_hand_qs, self.hand_qs], axis=0
                )
                self.hand_qs_th = torch.cat([prev_hand_qs_th, self.hand_qs_th], dim=0)
                self.goal_obj_trans_th = torch.cat([prev_goal_obj_trans_th, self.goal_obj_trans_th], dim=0)
                self.goal_obj_rot_quat_th = torch.cat([prev_goal_obj_rot_quat_th, self.goal_obj_rot_quat_th], dim=0)
                goal_obj_trans = np.concatenate([prev_goal_obj_trans, goal_obj_trans], axis=0)
                goal_obj_rot_quat = np.concatenate([prev_goal_obj_rot_quat, goal_obj_rot_quat], axis=0)
                self.goal_obj_trans = np.concatenate([prev_self_goal_obj_trans, self.goal_obj_trans], axis=0)
                self.goal_obj_rot_quat = np.concatenate([prev_self_goal_obj_rot_quat, self.goal_obj_rot_quat], axis=0)
            
            
        # 
        
        goal_obj_lin_vels, goal_obj_ang_vels = self._calculate_obj_vels(goal_obj_trans, goal_obj_rot_quat)
        self.goal_obj_lin_vels_th = torch.from_numpy(goal_obj_lin_vels).float().to(self.rl_device)
        self.goal_obj_ang_vels_th = torch.from_numpy(goal_obj_ang_vels).float().to(self.rl_device)
        

    def create_sim(self):
        self.object_id_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.dt = self.sim_params.dt # up axis #
        # self.up_axis_idx = self.set_sim_params_up_axis(self.sim_params, self.up_axis)
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))
        
        if self.randomize:
            self.apply_randomizations(self.randomization_params)




    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # plane_params.distance = self.ground_distance
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        object_scale_dict = self.cfg['env']['object_code_dict']
        
        # if len(self.object_name) > 0:
        #     if '_nf' in self.object_name: # from object name to the pure object name
        #         pure_obj_name = "_".join(self.object_name.split('_')[:-2])
        #     else:
        #         pure_obj_name = self.object_name
        #     # object_scale_dict = { f'sem/{self.object_name}' : [1.0] }
        #     object_scale_dict = { f'sem/{pure_obj_name}' : [1.0] }
        
        ### load all object codes -- tot_obj_codes ###
        object_scale_dict = {}
        for cur_obj_code in self.tot_obj_codes:
            if '_nf' in cur_obj_code:
                pure_obj_name = "_".join(cur_obj_code.split('_')[:-2])
            else:
                pure_obj_name = cur_obj_code
            object_scale_dict[ f'sem/{pure_obj_name}' ] = [1.0]
        
        
        # object code: sem/pure_obj_name #
        # self.object_code_list = list(object_scale_dict.keys())
        self.object_code_list = self.tot_obj_codes
        
        
        all_scales = set()
        for object_scales in object_scale_dict.values():
            for object_scale in object_scales:
                all_scales.add(object_scale)
        self.id2scale = []
        self.scale2id = {}
        for scale_id, scale in enumerate(all_scales): 
            self.id2scale.append(scale)
            self.scale2id[scale] = scale_id

        self.object_scale_id_list = []
        for object_scales in object_scale_dict.values():
            object_scale_ids = [self.scale2id[object_scale] for object_scale in object_scales]
            self.object_scale_id_list.append(object_scale_ids)
        self.repose_z = self.cfg['env']['repose_z']
        self.repose_z = False

        self.grasp_data = {}
        assets_path = '../assets'
        assets_path = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets"
        if not os.path.exists(assets_path):
            assets_path = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets"
        if not os.path.exists(assets_path):
            assets_path = "../../UniDexGrasp/dexgrasp_policy/assets"
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        # ## add object code ##
        # print(f"[Debug] object_code_list: {self.object_code_list}")
        # object shape inst list #
        
        for i_inst, object_code in enumerate(self.object_code_list):
            data_per_object = {}
            pure_object_code = object_code.split("_nf_")[0]
            dataset_path = dataset_root_path + '/sem/' + pure_object_code
            data_num_list = os.listdir(dataset_path)
            cur_inst_hand_qs = self.tot_kine_qs[i_inst]
            cur_inst_goal_obj_trans = self.tot_kine_obj_trans[i_inst]
            cur_inst_goal_obj_quat = self.tot_kine_obj_ornt[i_inst] 
            
            for num in data_num_list: # qpos, scale, target hand rot, target hand pos
                data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True)) # data path #
                qpos = data_dict['qpos'] # .item() # 
                scale_inverse = data_dict['scale'] # .item() #
                scale = round(1 / scale_inverse, 2)
                # print(f"[Debug] scale: {scale}")
                # assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                target_qpos = torch.from_numpy(qpos).float().to(self.device)
                target_hand_rot_xyz = torch.zeros((3, ), device=self.device) 
                target_hand_pos = torch.zeros((3, ), device=self.device)
                
                # target hand rot xyz #
                # target_qpos = torch.tensor(list(qpos.values())[:22], dtype=torch.float, device=self.device)
                # target_hand_rot_xyz = torch.tensor(list(qpos.values())[22:25], dtype=torch.float, device=self.device)  # 3
                target_hand_rot = quat_from_euler_xyz(target_hand_rot_xyz[0], target_hand_rot_xyz[1], target_hand_rot_xyz[2])  # 4
                # target_hand_pos = torch.tensor(list(qpos.values())[25:28], dtype=torch.float, device=self.device)
                # plane = data_dict['plane']  # plane parameters (A, B, C, D), Ax + By + Cz + D >= 0, A^2 + B^2 + C^2 = 1
                # translation, euler = plane2euler(plane, axes='sxyz')  # object
                # object_euler_xy = torch.tensor([euler[0], euler[1]], dtype=torch.float, device=self.device)
                # object_init_z = torch.tensor([translation[2]], dtype=torch.float, device=self.device)
                # ## TODO: inspct into the common setting of init_z in the original pipeline #
                
                # object_euler_xy = torch.tensor([0, 0], dtype=torch.float, device=self.device)
                # # object_init_z = torch.tensor([0.049], dtype=torch.float, device=self.device)
                # object_init_z = torch.tensor([0.0], dtype=torch.float, device=self.device)
                
                # 
                
                # self.init_hand_q = data_dict['qpos_init']
                
                # self.init_hand_q = self.hand_qs[0, :]
                # self.init_hand_q = torch.from_numpy(self.init_hand_q).float().to(self.device)
                
                # self.init_hand_q = cur_inst_han
                self.init_hand_q = cur_inst_hand_qs[0, :]
                
                # init_obj_tarns = self.goal_obj_trans[0, :]
                # init_obj_quat = self.goal_obj_rot_quat[0, :]
                
                # self.obj_trans_init = torch.from_numpy(init_obj_tarns).float().to(self.device)
                # self.obj_ornt_init = torch.from_numpy(init_obj_quat).float().to(self.device)
                
                self.obj_trans_init = cur_inst_goal_obj_trans[0, :]
                self.obj_ornt_init = cur_inst_goal_obj_quat[0, :]
                init_obj_quat = self.obj_ornt_init.detach().cpu().numpy() # xyz #
                init_obj_tarns = self.obj_trans_init.detach().cpu().numpy() # obj trans init #
                
                obj_ornt_init_euler_xyz = R.from_quat(init_obj_quat).as_euler('xyz', degrees=False)
                object_euler_xy = torch.tensor([obj_ornt_init_euler_xyz[0], obj_ornt_init_euler_xyz[1]], dtype=torch.float, device=self.device) # 
                object_init_z = torch.tensor([init_obj_tarns[2]], dtype=torch.float, device=self.device) # 
                
                
                # print(f"[Debug] object_init_z: {object_init_z}, object_euler_xy: {object_euler_xy}")
                
                # if object_init_z > 0.06:
                #     continue

                if scale in data_per_object:
                    data_per_object[scale]['target_qpos'].append(target_qpos)
                    data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                    data_per_object[scale]['target_hand_rot'].append(target_hand_rot)
                    data_per_object[scale]['object_euler_xy'].append(object_euler_xy)
                    data_per_object[scale]['object_init_z'].append(object_init_z)
                else:
                    data_per_object[scale] = {}
                    data_per_object[scale]['target_qpos'] = [target_qpos]
                    data_per_object[scale]['target_hand_pos'] = [target_hand_pos]
                    data_per_object[scale]['target_hand_rot'] = [target_hand_rot]
                    data_per_object[scale]['object_euler_xy'] = [object_euler_xy]
                    data_per_object[scale]['object_init_z'] = [object_init_z]
            self.grasp_data[object_code] = data_per_object

        ### NOTE: not a fly hand here ###
        # target pos is all joints ? # 
        self.goal_cond = self.cfg["env"]["goal_cond"] 
        self.random_prior = self.cfg['env']['random_prior']
        self.random_time = self.cfg["env"]["random_time"]
        # add the env test mode # 
        
        # taret qpos # hand pos 
        self.target_qpos = torch.zeros((self.num_envs, self.nn_hand_dof), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        
        
        # asset_root = "../../assets" # 
        # asset_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc"
        asset_root = "../assets"
        # shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
        
        
        if self.hand_type == 'allegro':
            shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf"
            if self.w_franka:
                shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        elif self.hand_type == 'leap':
            # shadow_hand_asset_file = f"leap_hand/leap_hand_right_fly_v3.urdf"
            # shadow_hand_asset_file = f"leap_hand/leap_hand_right_fly_v4.urdf"
            shadow_hand_asset_file = f"leap_hand/leap_hand_right_fly_v5.urdf"
        else:
            raise ValueError(f"Unknown hand type: {self.hand_type}") # hand type ##
        # if not os.path.exists(asset_root):
        #     asset_root = "/home/xueyi/diffsim/tiny-differentiable-simulator/python/examples/rsc"
            # shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd.urdf"
        
        
        # 
        # shadow_hand_asset_file = "mjcf/open_ai_assets/hand/shadow_hand.xml"
        table_texture_files = "../assets/textures/texture_stone_stone_texture_0.jpg"
        table_texture_handle = self.gym.create_texture_from_file(self.sim, table_texture_files)

        # if "asset" in self.cfg["env"]: # asset and the env #
        #     asset_root = self.cfg["env"]["asset"].get("assetRoot", asset_root) # asset and the env #
        #     shadow_hand_asset_file = self.cfg["env"]["asset"].get("assetFileName", shadow_hand_asset_file) # asset file name #

        print(f"[Debug] shadow_hand_asset_file: {shadow_hand_asset_file}")
        
        # load shadow hand_ asset
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = False # fixed base link # fix 
        asset_options.fix_base_link =  True # False
        # asset_options.collapse_fixed_joints = True
        if self.use_fingertips:
            asset_options.collapse_fixed_joints = False # 
        else:
            asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001 # 
        ## TODO: angulear damping and linear damping settings in the pybullet? 
        asset_options.angular_damping = 100
        asset_options.linear_damping = 100

        ## TODO: default dof drive mode? -- NONE ??
        if self.physics_engine == gymapi.SIM_PHYSX:
            asset_options.use_physx_armature = True
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
        shadow_hand_asset = self.gym.load_asset(self.sim, asset_root, shadow_hand_asset_file, asset_options)

        ### TODO: what's the difference between tendons and regular actuators? 
        self.num_shadow_hand_bodies = self.gym.get_asset_rigid_body_count(shadow_hand_asset)
        self.num_shadow_hand_shapes = self.gym.get_asset_rigid_shape_count(shadow_hand_asset)
        self.num_shadow_hand_dofs = self.gym.get_asset_dof_count(shadow_hand_asset)
        # self.num_shadow_hand_actuators = self.gym.get_asset_actuator_count(shadow_hand_asset)
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs
        self.num_shadow_hand_tendons = self.gym.get_asset_tendon_count(shadow_hand_asset)

        print("self.num_shadow_hand_bodies: ", self.num_shadow_hand_bodies)
        print("self.num_shadow_hand_shapes: ", self.num_shadow_hand_shapes)
        print("self.num_shadow_hand_dofs: ", self.num_shadow_hand_dofs)
        print("self.num_shadow_hand_actuators: ", self.num_shadow_hand_actuators)
        print("self.num_shadow_hand_tendons: ", self.num_shadow_hand_tendons)

        asset_rigid_body_names = [self.gym.get_asset_rigid_body_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_bodies)]
        print("asset_rigid_body_names: ", asset_rigid_body_names)
        
        # 
        
        # tendon set up
        # TODO: tendon set up? #
        limit_stiffness = 30
        t_damping = 0.1
        # relevant_tendons = ["robot0:T_FFJ1c", "robot0:T_MFJ1c", "robot0:T_RFJ1c", "robot0:T_LFJ1c"]
        # tendon_props = self.gym.get_asset_tendon_properties(shadow_hand_asset)

        # for i in range(self.num_shadow_hand_tendons):
        #     for rt in relevant_tendons:
        #         if self.gym.get_asset_tendon_name(shadow_hand_asset, i) == rt:
        #             tendon_props[i].limit_stiffness = limit_stiffness
        #             tendon_props[i].damping = t_damping

        # self.gym.set_asset_tendon_properties(shadow_hand_asset, tendon_props)
        # TODO: tendon set up? #

        # actuated_dof_names = [self.gym.get_asset_actuator_joint_name(shadow_hand_asset, i) for i in range(self.num_shadow_hand_actuators)]
        # self.actuated_dof_indices = [self.gym.find_asset_dof_index(shadow_hand_asset, name) for name in actuated_dof_names]
        
        # shadow hand actuators #
        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

 
        # table_z_dim # 
        # set shadow_hand dof properties # # dof properties # #
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        shadow_hand_dof_names = self.gym.get_asset_dof_names(shadow_hand_asset)
        print(f"[Debug] shadow_hand_dof_names: {shadow_hand_dof_names}")
        # ['WRJ0x', 'WRJ0y', 'WRJ0z', 'WRJ0rx', 'WRJ0ry', 'WRJ0rz', 'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_12', 'joint_13', 'joint_14', 'joint_15', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'joint_8', 'joint_9', 'joint_10', 'joint_11']
        
        self.thumb_dof_idxes = [10, 11, 12, 13]
        self.thumb_dof_idxes = torch.tensor(self.thumb_dof_idxes, dtype=torch.long, device=self.device)
        
        stiffness_coef = self.stiffness_coef
        damping_coef = self.damping_coef
        effort_coef = self.effort_coef
        
        self.shadow_hand_dof_lower_limits = []
        self.shadow_hand_dof_upper_limits = []
        self.shadow_hand_dof_default_pos = []
        self.shadow_hand_dof_default_vel = []
        self.sensors = []
        sensor_pose = gymapi.Transform()
        for i in range(self.num_shadow_hand_dofs):
            ### HACK ###
            print(f"i: {i}")
            # if i > 5:
            #     self.shadow_hand_dof_lower_limits.append(0.0) # add a table 
            # else: # self.table_z_dim --- #
            #     self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            
            
            # else:
            
            # if self.customize_global_damping:
                
            #     if i < 6:
            #         shadow_hand_dof_props['stiffness'][i] = 400
            #         shadow_hand_dof_props['effort'][i] = 200
            #         shadow_hand_dof_props['damping'][i] = 80
            #     elif i >= 6:
            #         shadow_hand_dof_props['velocity'][i] = 10.0
            #         shadow_hand_dof_props['effort'][i] = 0.7
            #         shadow_hand_dof_props['stiffness'][i] = 20
            #         shadow_hand_dof_props['damping'][i] = 1
            #     print(f"shadow_hand_dof_props: {shadow_hand_dof_props}")
                
                
            # if self.customize_damping:
            #     if i < 6:
            #         shadow_hand_dof_props['stiffness'][i] = 400
            #         shadow_hand_dof_props['effort'][i] = 200
            #         shadow_hand_dof_props['damping'][i] = 80
            #     elif i >= 6:
            #         shadow_hand_dof_props['velocity'][i] = 10.0
            #         shadow_hand_dof_props['effort'][i] = 0.7
            #         shadow_hand_dof_props['stiffness'][i] = 20
            #         shadow_hand_dof_props['damping'][i] = 1
            #     print(f"shadow_hand_dof_props: {shadow_hand_dof_props}")
                
            if self.customize_damping:
                if self.customize_global_damping:
                    if i < 6:
                        shadow_hand_dof_props['stiffness'][i] = 400
                        shadow_hand_dof_props['effort'][i] = 200
                        shadow_hand_dof_props['damping'][i] = 80
                
                
                if self.w_franka:
                    # if i < 7:
                    if i < 7:
                        shadow_hand_dof_props['stiffness'][i] = 400
                        shadow_hand_dof_props['effort'][i] = 200
                        shadow_hand_dof_props['damping'][i] = 80
                    else:
                        shadow_hand_dof_props['velocity'][i] = 10.0
                        shadow_hand_dof_props['effort'][i] = 0.7
                        shadow_hand_dof_props['stiffness'][i] = 20
                        shadow_hand_dof_props['damping'][i] = 1
                else:
                        
                    if i >= 6:
                        if self.hand_type == 'leap':
                            
                            # shadow_hand_dof_props['velocity'][i] = 10.0 # 
                            # shadow_hand_dof_props['effort'][i] = 10 #  20 #  0.95
                            # shadow_hand_dof_props['stiffness'][i] = 20 #  200
                            # # shadow_hand_dof_props['effort'][i] = 20 #  20 #  0.95
                            # # shadow_hand_dof_props['stiffness'][i] = 20 #  200
                            # # shadow_hand_dof_props['effort'][i] =  0.95
                            # # shadow_hand_dof_props['stiffness'][i] = 200
                            # shadow_hand_dof_props['damping'][i] = 0.1
                            
                            ##### v2 #####
                            # shadow_hand_dof_props['effort'][i] = 0.5
                            # shadow_hand_dof_props['stiffness'][i] =  3 # self.cfg['env']['controller']['pgain']
                            # shadow_hand_dof_props['damping'][i] = 0.1 #  self.cfg['env']['controller']['dgain']
                            # shadow_hand_dof_props['friction'][i] = 0.01
                            # shadow_hand_dof_props['armature'][i] = 0.001
                            ##### v2 #####
                            
                            # shadow_hand_dof_props['effort'][i] = 0.95
                            # shadow_hand_dof_props['stiffness'][i] =  100 # self.cfg['env']['controller']['pgain']
                            # shadow_hand_dof_props['damping'][i] = 4 #  self.cfg['env']['controller']['dgain']
                            
                            shadow_hand_dof_props['effort'][i] = self.effort_coef
                            shadow_hand_dof_props['stiffness'][i] =  self.stiffness_coef # self.cfg['env']['controller']['pgain']
                            shadow_hand_dof_props['damping'][i] = self.damping_coef #  self.cfg['env']['controller']['dgain']
                            shadow_hand_dof_props['friction'][i] = 0.01
                            shadow_hand_dof_props['armature'][i] = 0.001
                        else:
                            shadow_hand_dof_props['velocity'][i] = 10.0
                            shadow_hand_dof_props['effort'][i] = 0.7
                            shadow_hand_dof_props['stiffness'][i] = 20
                            shadow_hand_dof_props['damping'][i] = 1
                print(f"shadow_hand_dof_props: {shadow_hand_dof_props}")
            
            # shadow_hand_dof_props['velocity'][i]
            
            self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])
            self.shadow_hand_dof_upper_limits.append(shadow_hand_dof_props['upper'][i])
            self.shadow_hand_dof_default_pos.append(0.0)
            self.shadow_hand_dof_default_vel.append(0.0)

        self.actuated_dof_indices = to_torch(self.actuated_dof_indices, dtype=torch.long, device=self.device)
        self.shadow_hand_dof_lower_limits = to_torch(self.shadow_hand_dof_lower_limits, device=self.device)
        self.shadow_hand_dof_upper_limits = to_torch(self.shadow_hand_dof_upper_limits, device=self.device)
        self.shadow_hand_dof_default_pos = to_torch(self.shadow_hand_dof_default_pos, device=self.device)
        self.shadow_hand_dof_default_vel = to_torch(self.shadow_hand_dof_default_vel, device=self.device)
        
        
        
        
        
        print(f"[Debug] shadow_hand_dof_lower_limits: {self.shadow_hand_dof_lower_limits}")
        print(f"[Debug] shadow_hand_dof_upper_limits: {self.shadow_hand_dof_upper_limits}")

        shadow_hand_joint_dict = self.gym.get_asset_joint_dict(shadow_hand_asset)
        for key in shadow_hand_joint_dict:
            val = shadow_hand_joint_dict[key]
            print(f"key: {key} val: {val}")

        # visual feature
        scale2str = {
            0.06: '006',
            0.08: '008',
            0.10: '010',
            0.12: '012',
            0.15: '015',
            1: '1',
        } 

        object_scale_idx_pairs = []
        visual_feat_root = osp.realpath(osp.join(assets_path, 'meshdatav3_pc_feat'))
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64), device=self.device)
        
        # object id #
        for object_id in range(len(self.object_code_list)): # object id # 
            object_code = self.object_code_list[object_id]
            self.visual_feat_data[object_id] = {}
            for scale_id in self.object_scale_id_list[object_id]:
                scale = self.id2scale[scale_id]
                
                object_scale_idx_pairs.append([object_id, scale_id])
                self.visual_feat_data[object_id][scale_id] = torch.zeros((64, ), device=self.device, dtype=torch.float)
                
                # ##### original code #####
                # if scale in self.grasp_data[object_code]:
                #     object_scale_idx_pairs.append([object_id, scale_id])
                # else:
                #     print(f'prior not found: {object_code}/{scale}')
                # file_dir = osp.join(visual_feat_root, f'{object_code}/pc_feat_{scale2str[scale]}.npy')
                # with open(file_dir, 'rb') as f:
                #     feat = np.load(f)
                # self.visual_feat_data[object_id][scale_id] = torch.tensor(feat, device=self.device)     
                # ##### original code #####   

        object_asset_dict = {}
        goal_asset_dict = {}
        
        object_asset_list = []
        
        maxx_num_obj_bodies = 0
        maxx_num_obj_shapes = 0

        ##### Load the object asset for each object in the object code list #####
        ## TODO: decompose the object and add into the mesh_data_scaled ##
        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        for object_id, object_code in enumerate(self.object_code_list): # object_code 
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            # object_asset_options.density = 500
            object_asset_options.density = self.rigid_obj_density
            object_asset_options.fix_base_link = False
            # print(f"disable_obj_gravity: {self.disable_obj_gravity}")
            object_asset_options.disable_gravity = self.disable_obj_gravity
            object_asset_options.use_mesh_materials = True # mesh
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            
            for obj_id, scale_id in object_scale_idx_pairs: # # find # #
                # print(f"obj_id: {obj_id}, scale_id: {scale_id}, object_id: {object_id}") 
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    pure_object_code = object_code.split("_nf_")[0]
                    scaled_object_asset_file = 'sem/' + pure_object_code + f"/coacd/coacd_{scale_str}.urdf"
                    # print(f"scaled_object_asset_file: {scaled_object_asset_file}")
                    scaled_object_asset = self.gym.load_asset(self.sim, mesh_path, scaled_object_asset_file,
                                                              object_asset_options)
                    if obj_id not in object_asset_dict:
                        object_asset_dict[object_id] = {}
                    object_asset_dict[object_id][scale_id] = scaled_object_asset

                    if object_asset is None:
                        object_asset = scaled_object_asset
            
            assert object_asset is not None
            object_asset_options.disable_gravity = True    
            goal_asset = self.gym.create_sphere(self.sim, 0.005, object_asset_options)
            self.num_object_bodies = self.gym.get_asset_rigid_body_count(object_asset)
            self.num_object_shapes = self.gym.get_asset_rigid_shape_count(object_asset)
            
            maxx_num_obj_bodies = max(maxx_num_obj_bodies, self.num_object_bodies)
            maxx_num_obj_shapes = max(maxx_num_obj_shapes, self.num_object_shapes)

            # set object dof properties
            self.num_object_dofs = self.gym.get_asset_dof_count(object_asset)
            object_dof_props = self.gym.get_asset_dof_properties(object_asset) # asset dof properties #
            self.object_dof_lower_limits = []
            self.object_dof_upper_limits = []

            for i in range(self.num_object_dofs):
                self.object_dof_lower_limits.append(object_dof_props['lower'][i])
                self.object_dof_upper_limits.append(object_dof_props['upper'][i])

            # 
            self.object_dof_lower_limits = to_torch(self.object_dof_lower_limits, device=self.device)
            self.object_dof_upper_limits = to_torch(self.object_dof_upper_limits, device=self.device)
            print(f"[Debug] object_dof_lower_limits: {self.object_dof_lower_limits}")
            print(f"[Debug] object_dof_upper_limits: {self.object_dof_upper_limits}")


        # create table asset
        # table_dims = gymapi.Vec3(1, 1, 0.6)
        # table_dims = gymapi.Vec3(1, 1, 0.00001)
        table_dims = gymapi.Vec3(1, 1, self.table_z_dim)
        asset_options = gymapi.AssetOptions()
        asset_options.fix_base_link = True
        asset_options.flip_visual_attachments = True
        asset_options.collapse_fixed_joints = True
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        # object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        #### set the table asset options ####
        # table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

        shadow_hand_start_pose = gymapi.Transform() # gymapi.Vec3(0.0, )
        # shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # gymapi.Vec3(0.1, 0.1, 0.65)
        # shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, self.table_z_dim/2.0) 
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) 
        # shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0)

        object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        obj_init_x = self.obj_trans_init[0].item()
        obj_init_y = self.obj_trans_init[1].item()
        obj_init_z = self.obj_trans_init[2].item()
        object_start_pose.p = gymapi.Vec3(obj_init_x, obj_init_y, obj_init_z)  # gymapi.Vec3(0.0, 0.0, 0.72) # gymapi #
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0) # from euler zyx # from euler zyx #
        # pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0
        object_start_pose.r = gymapi.Quat(self.obj_ornt_init[0].item(), self.obj_ornt_init[1].item(), self.obj_ornt_init[2].item(), self.obj_ornt_init[3].item())
        
        

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        # goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        goal_start_pose.r = gymapi.Quat(self.obj_ornt_init[0].item(), self.obj_ornt_init[1].item(), self.obj_ornt_init[2].item(), self.obj_ornt_init[3].item())

        goal_start_pose.p.z -= 0.0 # goal start pose # # goal pose #

        table_pose = gymapi.Transform()
        ###### set table pose ######
        # by default # by default #
        # table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        
        if 'transformed' in self.mocap_sv_info_fn:
            table_pose.p = gymapi.Vec3(0.5 * table_dims.x, -0.5 * table_dims.y, 0.5 * table_dims.z)
        else:
            table_pose.p = gymapi.Vec3(-0.5 * table_dims.x, 0.5 * table_dims.y, 0.5 * table_dims.z)
        # table_pose.p = gymapi.Vec3(0.0, 0.0, -0.5 * table_dims.z)
        # table_pose.p = gymapi.Vec3(0.0, 0.0, -1.0 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size
        # max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1
        # max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1
        max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * maxx_num_obj_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * maxx_num_obj_shapes + 1

        self.shadow_hands = []
        self.envs = []
        self.object_init_state = []
        self.goal_init_state = []
        self.hand_start_states = []
        self.hand_indices = []
        self.fingertip_indices = []
        self.object_indices = []
        self.goal_object_indices = []
        self.table_indices = []

        self.fingertip_handles = [self.gym.find_asset_rigid_body_index(shadow_hand_asset, name) for name in self.fingertips]
        
        # ## TODO: change it to the alegro body name # # table pos p #
        # body_names = {
        #     # 'wrist': 'robot0:wrist',
        #     'palm': 'palm_link',
        #     'thumb': 'link_15',
        #     'index': 'link_3',
        #     'middle': 'link_7',
        #     'ring': 'link_11', # finger
        #     # 'little': 'robot0:lfdistal'
        # }
        
        ## ##
        
        if self.use_fingertips: #
            if self.hand_type == 'allegro': # 
                body_names = {
                    # 'wrist': 'robot0:wrist',
                    'palm': 'palm_link',
                    'thumb': 'link_15_tip',
                    'index': 'link_3_tip',
                    'middle': 'link_7_tip',
                    'ring': 'link_11_tip',
                    # 'little': 'robot0:lfdistal'
                }
            elif self.hand_type == 'leap':
                # body_names = { # leap fingertips #
                #     'palm': 'palm_lower',
                #     'thumb': 'thumb_fingertip',
                #     'index': 'fingertip',
                #     'middle': 'fingertip_2',
                #     'ring': 'fingertip_3',
                # }
                body_names = { # leap fingertips #
                    'palm': 'palm_lower',
                    'thumb': 'thumb_tip_head',
                    'index': 'index_tip_head',
                    'middle': 'middle_tip_head',
                    'ring': 'ring_tip_head',
                }
        else:
            ### TODO: get the non-fngertip setttings for the leap hand ###
            body_names = {
                # 'wrist': 'robot0:wrist',
                'palm': 'palm_link',
                'thumb': 'link_15',
                'index': 'link_3',
                'middle': 'link_7',
                'ring': 'link_11',
                # 'little': 'robot0:lfdistal'
            }
        
        print(f"[Debug] fingertips handles: {self.fingertip_handles}")
        self.hand_body_idx_dict = {} #
        for name, body_name in body_names.items():
            self.hand_body_idx_dict[name] = self.gym.find_asset_rigid_body_index(shadow_hand_asset, body_name)
            print(f"[Debug] body_name: {name}, idx: {self.hand_body_idx_dict[name]}")

        # create fingertip force sensors, if needed # if needed # if needed #
        if self.obs_type == "full_state" or self.asymmetric_obs: # fingertip sensors #
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)

        # 
        # add a table and # change the difficulty of the task? #
        self.object_scale_buf = {}
        self.tot_hand_dof_pos = []

        self.env_inst_idxes = []
        
        # 
        # self.env_object_latent_feat # latent feat # #
        # self.env_object_latent_feat # latent feat # #
        self.env_object_latent_feat = []
        self.env_object_codes = []
        self.env_inst_latent_feat = []
        self.env_rew_succ_list = []
        self.env_teacher_idx_list = []
        print(f"len(object_code_list): {len(self.object_code_list)}")
        for i in range(self.num_envs):
            
            data_inst_idx = i % len(self.object_code_list)
            cur_inst_hand_kine_qs = self.tot_kine_qs[data_inst_idx] 
            cur_inst_goal_obj_trans = self.tot_kine_obj_trans[data_inst_idx] 
            cur_inst_goal_obj_ornt = self.tot_kine_obj_ornt[data_inst_idx]
            
            first_frame_goal_obj_trans = cur_inst_goal_obj_trans[0, :]
            first_frame_goal_obj_ornt = cur_inst_goal_obj_ornt[0, :]
            
            first_frame_hand_kine_qs = cur_inst_hand_kine_qs[0, :]
            self.tot_hand_dof_pos.append(first_frame_hand_kine_qs)
            
            self.env_inst_idxes.append(data_inst_idx)
            
            cur_object_code = self.object_code_list[data_inst_idx]
            pure_object_code = cur_object_code.split("_nf_")[0]
            cur_object_latent_features = self.object_type_to_latent_feature[pure_object_code]
            self.env_object_latent_feat.append(cur_object_latent_features)
            
            # if self.grab_obj_type_to_opt_res is not None:
            # cur_obj_succ = self.object_rew_succ_dict[cur_object_code]
            if pure_object_code in self.object_rew_succ_dict:
                cur_obj_succ = self.object_rew_succ_dict[pure_object_code]
            else:
                cur_obj_succ = 0
            self.env_rew_succ_list.append(cur_obj_succ)
            
            if self.use_multiple_teacher_model:
                if pure_object_code in self.obj_tag_to_teacher_idx:
                    cur_obj_teacher_idx = self.obj_tag_to_teacher_idx[pure_object_code]
                else:
                    cur_obj_teacher_idx = -1 #
                self.env_teacher_idx_list.append(cur_obj_teacher_idx) # get the teacher index list #
                
            
            if self.use_inst_latent_features:
                cur_inst_tag = self.object_code_list[data_inst_idx]
                pure_cur_inst_tag = cur_inst_tag.split("_nf_")[0]
                cur_inst_latent_features = self.inst_tag_to_latent_features[pure_cur_inst_tag]
                self.env_inst_latent_feat.append(cur_inst_latent_features) # get the inst latent features #
            
            # create env instance
            self.env_object_codes.append(pure_object_code) # get the env object codes # # get the env object codes #
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                # print(f"Start the aggregation with mode {self.aggregate_mode}, obj_idx: {data_inst_idx}, obj_code: {cur_object_code}")
                # print(f"object_code_list: {self.object_code_list}")
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # 
            shadow_hand_actor = self.gym.create_actor(env_ptr, shadow_hand_asset, shadow_hand_start_pose, "hand", i, -1, 0)
            self.hand_start_states.append(
                [shadow_hand_start_pose.p.x, shadow_hand_start_pose.p.y, shadow_hand_start_pose.p.z,
                 shadow_hand_start_pose.r.x, shadow_hand_start_pose.r.y, shadow_hand_start_pose.r.z,
                 shadow_hand_start_pose.r.w,
                 0, 0, 0, 0, 0, 0])

            self.gym.set_actor_dof_properties(env_ptr, shadow_hand_actor, shadow_hand_dof_props)
            hand_idx = self.gym.get_actor_index(env_ptr, shadow_hand_actor, gymapi.DOMAIN_SIM)
            self.hand_indices.append(hand_idx)

            #### no colore ####
            # randomize colors and textures for rigid body
            # num_bodies = self.gym.get_actor_rigid_body_count(env_ptr, shadow_hand_actor)
            # hand_color = [147/255, 215/255, 160/255]
            # hand_rigid_body_index = [[0,1,2,3], [4,5,6,7], [8,9,10,11], [12,13,14,15], [16,17,18,19,20], [21,22,23,24,25]]
            # for n in self.agent_index[0]:
            #     for m in n:
            #         for o in hand_rigid_body_index[m]:
            #             self.gym.set_rigid_body_color(env_ptr, shadow_hand_actor, o, gymapi.MESH_VISUAL,
            #                                     gymapi.Vec3(*hand_color))

            # create fingertip force-torque sensors # # chagne the control strategy # #
            if self.obs_type == "full_state" or self.asymmetric_obs: 
                self.gym.enable_actor_dof_force_sensors(env_ptr, shadow_hand_actor)

            # 
            # id = int(i / self.num_envs * len(self.object_code_list))
            
            id = i % len(self.object_code_list)
            object_code = self.object_code_list[id]
            available_scale = []
            for scale_id in self.object_scale_id_list[id]:
                scale = self.id2scale[scale_id]
                if scale in self.grasp_data[object_code]:
                    available_scale.append(scale)
                else:
                    print(f'prior not found: {object_code}/{scale}')
            scale = available_scale[i % len(available_scale)]
            scale_id = self.scale2id[scale]
            self.object_scale_buf[i] = scale
            self.object_id_buf[i] = id
            # visual feat buf idx #
            self.visual_feat_buf[i] = self.visual_feat_data[id][scale_id]
            
            # print(f"Start creating obs")
            object_start_pose = gymapi.Transform()
            obj_init_x = first_frame_goal_obj_trans[0].item()
            obj_init_y = first_frame_goal_obj_trans[1].item()
            obj_init_z = first_frame_goal_obj_trans[2].item()
            object_start_pose.p = gymapi.Vec3(obj_init_x, obj_init_y, obj_init_z)
            object_start_pose.r = gymapi.Quat(first_frame_goal_obj_ornt[0].item(), first_frame_goal_obj_ornt[1].item(), first_frame_goal_obj_ornt[2].item(), first_frame_goal_obj_ornt[3].item())


            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i, 0, 0)
            
            
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)
            # set actor sacle #
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            

            #### NOTE: we have disabled table here ####
            if self.add_table: # add table --- 
                table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
                self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
                table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
                self.table_indices.append(table_idx)


                table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
                table_shape_props[0].friction = 1
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            
            # object shape props #
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)
            

            object_color = [90/255, 94/255, 173/255]
            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
            
            
        self.tot_hand_dof_pos = torch.stack(self.tot_hand_dof_pos, dim=0)
        self.shadow_hand_default_dof_pos = self.tot_hand_dof_pos
        
        # if self.use_canonical_state:
        #     # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
        #     # self.shadow_hand_default_dof_pos[2] = 0.2
        #     if self.use_unified_canonical_state:
        #         self.shadow_hand_default_dof_pos = torch.zeros_like(self.shadow_hand_default_dof_pos)
        #         self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
        #         self.shadow_hand_default_dof_pos[2] = 0.2
        #         self.shadow_hand_default_dof_pos[1] = 0.0
        #     else:
        #         print(f"setting the canonical state")
        #         # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
        #         # self.shadow_hand_default_dof_pos[2] = 0.2
        #         # self.shadow_hand_default_dof_pos[1] = -0.07
                
        #         # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
        #         self.shadow_hand_default_dof_pos[2] += 0.01
        #         # self.shadow_hand_default_dof_pos[1] = -0.07 # use the canonical state #
                # self.shadow_hand_default_dof_pos[6:] = 0.0
        
        ###### ######
        # set dof state tensor index #
        # self.gym.set_dof_state_tensor_indexed(self.sim, # two hands #
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(hand_indices), len(env_ids))

        # # create envs # #
        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone( )
        # self.goal_pose = self.goal_states[:, 0:7] #
        # self.goal_pos = self.goal_states[:, 0:3] #
        # self.goal_rot = self.goal_states[:, 3:7] #
        # self.goal_states[:, self.up_axis_idx] -= 0.04 #
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
        self.env_object_latent_feat = np.stack(self.env_object_latent_feat, axis=0) # nn_envs x nn_latent_feat_dim
        self.env_object_latent_feat = to_torch(self.env_object_latent_feat, dtype=torch.float32, device=self.device) 
        
        
        if len(self.env_rew_succ_list) > 0:
            self.env_rew_succ_list = np.array(self.env_rew_succ_list, dtype=np.float32)
            self.env_rew_succ_list = to_torch(self.env_rew_succ_list, dtype=torch.float32, device=self.device)
        
        
        if self.use_multiple_teacher_model:
            # if len(self.env_teacher_idx_list):
            assert len(self.env_teacher_idx_list) > 0
            self.env_teacher_idx_list = np.array(self.env_teacher_idx_list, dtype=np.int32)
            self.env_teacher_idx_list = to_torch(self.env_teacher_idx_list, dtype=torch.long, device=self.device) # get the env teacher idx list #
        
        if self.use_inst_latent_features:
            self.env_inst_latent_feat = np.stack(self.env_inst_latent_feat, axis=0)
            self.env_inst_latent_feat = to_torch(self.env_inst_latent_feat, dtype=torch.float32, device=self.device)
        
        self.env_inst_idxes = np.array(self.env_inst_idxes, dtype=np.int32)
        self.env_inst_idxes = to_torch(self.env_inst_idxes, dtype=torch.long, device=self.device) # ge the env inst idxes #
        
        if self.add_table: # remember the table asset initial poses ? #
            self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)
            
        ##### get object_rew_succ_dict #####
        succ_instst = [
            obj_key for obj_key, obj_val in self.object_rew_succ_dict.items() if obj_val == 1
        ]
        print(f"tot_obj_succ_nn: {len(succ_instst)}")

    # define some differnet generalization levels? # different levels of generalization #

    def compute_reward_two_stages(self, actions, id=-1):
        self.dof_pos = self.shadow_hand_dof_pos # shadow hand dof pos #
        # hand_up_threshold_1: float, hand_up_threshold_2: float
        
        # ##### NOTE: previous thresholds with tables in the scene #####
        # hand_up_threshold_1 = 0.630
        # hand_up_threshold_2 = 0.80
        # ##### NOTE: previous thresholds with tables in the scene #####
        
        ##### NOTE: current thresholds without tables #####
        hand_up_threshold_1 = 0.030
        hand_up_threshold_2 = 0.2
        ##### NOTE: current thresholds without tables #####
        # fingertips #
        
        # goal_linvel, goal_angvel #
        # object_linvel , object_pos #
         
        # current goal #
        
        # goal_lifting_pos
        
        # separate_stages #
        
        # # hand_pose_guidance_glb_trans_coef, hand_pose_guidance_glb_rot_coef, hand_pose_guidance_fingerpose_coef
        
        # grasping_frame_hand_pose, # grasping hand pose # 
        # grasping_progress_buf, # the grasping progress buffer --- #
        # grasping_manip_stage,
        # manip_frame_hand_pose,
        # hand_pose,
        # maxx_grasping_steps: int,
        # grasping succ buf; 
        # only with the object pose guidance? #
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:], self.grasping_succ_buf[:] = compute_hand_reward_tracking_twostages( # compute hand tracking reward ##
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.object_linvel, self.object_angvel, self.object_linvel, self.object_angvel,
            self.goal_pos, self.goal_rot, self.goal_lifting_pos,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos, # 
            self.grasp_frame_hand_qpos, # grasping frame
            self.grasping_progress_buf, # grasping progress buffer
            self.grasp_manip_stages, # grasping 
            self.ori_cur_hand_qpos_ref, # 
            self.shadow_hand_dof_pos,
            self.maxx_grasping_steps,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.test, self.right_hand_dist_thres
        )


        self.grasping_progress_buf = torch.where(
            self.grasping_succ_buf.int() == 1, torch.zeros_like(self.grasping_progress_buf), self.grasping_progress_buf
        )
        self.progress_buf = torch.where(
            self.grasping_succ_buf.int() == 1, self.cur_grasp_fr + torch.zeros_like(self.progress_buf), self.progress_buf
        )
        self.grasp_manip_stages = torch.where(
            self.grasping_succ_buf.int() == 1, 1 + torch.zeros_like(self.grasp_manip_stages), self.grasp_manip_stages
        )
        avg_grasp_manip_stages = torch.mean(self.grasp_manip_stages.float())
        if self.test:
            avg_grasping_succ_buf = torch.mean(self.grasping_succ_buf.float())
            print(f"avg_grasping_succ_buf: {avg_grasping_succ_buf}")
            print(f"avg_grasp_manip_stages: {avg_grasp_manip_stages}")

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum() # 
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance.
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_reward_taco(self, actions, id=-1):
        
        if self.use_twostage_rew:
            self.compute_reward_two_stages(actions, id)
            return
        
        self.dof_pos = self.shadow_hand_dof_pos # shadow hand dof pos #
        # hand_up_threshold_1: float, hand_up_threshold_2: float
        
        # ##### NOTE: previous thresholds with tables in the scene #####
        # hand_up_threshold_1 = 0.630
        # hand_up_threshold_2 = 0.80
        # ##### NOTE: previous thresholds with tables in the scene #####
        
        ##### NOTE: current thresholds without tables #####
        hand_up_threshold_1 = 0.030
        hand_up_threshold_2 = 0.2
        ##### NOTE: current thresholds without tables #####
        # fingertips #
        
        # goal_linvel, goal_angvel #
        # object_linvel , object_pos #
         
        # current goal #
        
        # goal_lifting_pos
        
        # separate_stages #
        
        # # hand_pose_guidance_glb_trans_coef, hand_pose_guidance_glb_rot_coef, hand_pose_guidance_fingerpose_coef
        
        if self.dataset_type == 'grab':
            compute_reward_func = compute_hand_reward_tracking
        elif self.dataset_type == 'taco':
            compute_reward_func = compute_hand_reward_tracking_taco
        else:
            raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        
        # self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos #
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:], self.hand_palm_fingers_obj_contact_buf[:], self.right_hand_dist_buf[:] = compute_reward_func( # compute hand tracking reward ##
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.object_linvel, self.object_angvel, self.object_linvel, self.object_angvel,
            self.lift_obj_pos_th,
            self.goal_pos, self.goal_rot, self.goal_lifting_pos,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos, # 
            self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos ,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.rew_obj_pose_coef, self.goal_dist_thres, self.lifting_separate_stages, self.reach_lifting_stage, self.strict_lifting_separate_stages
        )
        
        
        if self.lifting_separate_stages:
            if self.strict_lifting_separate_stages:
                obj_goal_dist_thres = 0.05
                dist_obj_pos_w_lift_pos = torch.sum(
                    (self.object_pos - self.lift_obj_pos_th.unsqueeze(0)) ** 2, dim=-1
                )
                dist_obj_pos_w_lift_pos = torch.sqrt(dist_obj_pos_w_lift_pos) # nn_envs -- distances from the lifting position goal 
                grasping_succ_buf = (self.reach_lifting_stage == 0).int() + (dist_obj_pos_w_lift_pos < obj_goal_dist_thres).int() + self.hand_palm_fingers_obj_contact_buf.int()
                self.reach_lifting_stage = torch.where(
                    grasping_succ_buf == 3, 1 + torch.zeros_like(self.reach_lifting_stage), self.reach_lifting_stage
                )
                self.progress_buf = torch.where(
                    grasping_succ_buf == 3, self.lift_fr + torch.zeros_like(self.progress_buf), self.progress_buf
                )
                
                grasping_failed_buf = (self.reach_lifting_stage == 0).int() + (((dist_obj_pos_w_lift_pos > obj_goal_dist_thres).int() + (self.hand_palm_fingers_obj_contact_buf.int() == 0).int()) >= 1).int() + (self.progress_buf >= self.maxx_grasping_steps).int()
                self.reset_buf = torch.where(
                    grasping_failed_buf == 3, 1 + torch.zeros_like(self.reset_buf), self.reset_buf
                )
                self.reset_goal_buf = torch.where(
                    grasping_failed_buf == 3, 1 + torch.zeros_like(self.reset_goal_buf), self.reset_goal_buf
                )
                
                right_hand_dist_buf_buf_terminate = self.right_hand_dist_buf >= 0.6
                self.reset_buf = torch.where(
                    right_hand_dist_buf_buf_terminate, 1 + torch.zeros_like(self.reset_buf), self.reset_buf
                )
                self.reset_goal_buf = torch.where(
                    right_hand_dist_buf_buf_terminate, 1 + torch.zeros_like(self.reset_goal_buf), self.reset_goal_buf
                )
                if self.test:
                    avg_reach_lift_stage = torch.mean(self.reach_lifting_stage.float())
                    reach_lift_stage_env_ids = torch.argsort(self.reach_lifting_stage, descending=True)
                    print(f"avg_reach_lift_stage: {avg_reach_lift_stage}, reach_lift_stage_env_ids: {reach_lift_stage_env_ids[:10]}")
                
            else:
                lowest = self.object_pos[:, 2] # .unsqueeze(-1).repeat(1, 3)
                lift_height_z = self.lift_obj_pos_th[2].item()
                # target_lifting_pos = self.lift_obj_pos_th.unsqueeze(0).contiguous().repeat(target_pos.size(0), 1).contiguous()
                # target_lifting_pos as  the target lifting pos #
                # target_pos = torch.where(lowest < lift_height_z, target_lifting_pos, target_pos )
                
                ## reach lifting stage flag ## 
                self.reach_lifting_stage_flag = lowest >= lift_height_z # larger than the lift-heightj-z ##
                if torch.sum(self.reach_lifting_stage_flag.float()) > 0.5:
                    cur_reached_lifting_stage_obj_pos = self.object_pos[self.reach_lifting_stage_flag]
                    avg_cur_obj_pos = cur_reached_lifting_stage_obj_pos.mean(dim=0)
                    # print(f"avg_cur_obj_pos: {avg_cur_obj_pos}")
                
                ## reach lifting stge ##
                new_reach_lifting_stage_flag = (self.reach_lifting_stage == 0).int() + self.reach_lifting_stage_flag.int()
                self.reach_lifting_stage = torch.where(
                    new_reach_lifting_stage_flag == 2, 1 + torch.zeros_like(self.reach_lifting_stage), self.reach_lifting_stage # change jor not to change th eflag
                )
                self.progress_buf = torch.where(
                    new_reach_lifting_stage_flag == 2, self.lift_fr + torch.zeros_like(self.progress_buf), self.progress_buf
                )
        else:
            right_hand_dist_buf_buf_terminate = self.right_hand_dist_buf >= 0.6
            self.reset_buf = torch.where(
                right_hand_dist_buf_buf_terminate, 1 + torch.zeros_like(self.reset_buf), self.reset_buf
            )
            self.reset_goal_buf = torch.where(
                right_hand_dist_buf_buf_terminate, 1 + torch.zeros_like(self.reset_goal_buf), self.reset_goal_buf
            )
            
            hand_obj_contact_terminate = (self.hand_palm_fingers_obj_contact_buf == 0  ).int() + (self.progress_buf >= 250).int()
            hand_obj_contact_terminate = (hand_obj_contact_terminate >= 2)
            self.reset_buf = torch.where(
                hand_obj_contact_terminate, 1 + torch.zeros_like(self.reset_buf), self.reset_buf
            )
            self.reset_goal_buf = torch.where(
                hand_obj_contact_terminate, 1 + torch.zeros_like(self.reset_goal_buf), self.reset_goal_buf
            )
            
        
        

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum() # 
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance. # consecutive #
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))


    def compute_reward(self, actions, id=-1):
        
        if self.dataset_type == 'taco':
            self.compute_reward_taco(actions, id)
            return
        
        if self.use_twostage_rew:
            self.compute_reward_two_stages(actions, id)
            return
        
        self.dof_pos = self.shadow_hand_dof_pos # shadow hand dof pos #
        # hand_up_threshold_1: float, hand_up_threshold_2: float
        
        # ##### NOTE: previous thresholds with tables in the scene #####
        # hand_up_threshold_1 = 0.630
        # hand_up_threshold_2 = 0.80
        # ##### NOTE: previous thresholds with tables in the scene #####
        
        ##### NOTE: current thresholds without tables #####
        hand_up_threshold_1 = 0.030
        hand_up_threshold_2 = 0.2
        ##### NOTE: current thresholds without tables #####
        
        
        # print(f"maxx_env_inst_idx: {torch.max(self.env_inst_idxes)}, tot_hand_qtars: {self.tot_hand_qtars.size()}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj.size()}")
        envs_hand_qtars = batched_index_select(self.tot_hand_qtars, self.env_inst_idxes, dim=0)
        
        # print(f"env_inst_idxes: {torch.max(self.env_inst_idxes)}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj.size()}, max_episode_length: {self.maxx_episode_length_per_traj}")
        envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        
        if self.use_window_future_selection:
            cur_progress_buf = torch.clamp(self.ws_selected_progress_buf, min=torch.zeros_like(envs_episode_length), max=envs_episode_length)
        else:
            cur_progress_buf = torch.clamp(self.progress_buf, min=torch.zeros_like(envs_episode_length), max=envs_episode_length)
        
        
        if self.random_shift_cond_freq or  self.preset_inv_cond_freq > 1:
            # print(f"[Debug] changing the progress buffer under the random_shift_cond_freq setting")
            moded_progress_buf = cur_progress_buf // self.env_inv_cond_freq
            increase_nn = (cur_progress_buf > moded_progress_buf * self.env_inv_cond_freq).int()
            cur_progress_buf = (moded_progress_buf + increase_nn) * self.env_inv_cond_freq
            cur_progress_buf = torch.clamp(cur_progress_buf, max=envs_episode_length, min=torch.zeros_like(cur_progress_buf))
        
        
        envs_hand_qtars = batched_index_select(envs_hand_qtars, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        
        
        
        if self.comput_reward_traj_hand_qpos or self.use_future_ref_as_obs_goal:
            # print(f"Debug====comput_reward_traj_hand_qpos: {self.comput_reward_traj_hand_qpos}")
            tot_goal_hand_qs_th = self.tot_hand_preopt_res
            
            
            envs_traj_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) # nn_envs x maxx episolid length x nn_hand_qs #
            
            
            
            if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
                envs_traj_goal_hand_qs = batched_index_select(self.multiple_kine_source_trajs, self.envs_kine_source_trajs_idxes, dim=0)
            
        
            # modify deta qpos 
            cur_hand_qpos_ref = batched_index_select(envs_traj_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
            self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
            
            
            tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
            tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
            
            envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
            envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
            
            
            if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
                envs_goal_obj_trans_th = batched_index_select(self.multiple_kine_source_obj_pos, self.envs_kine_source_trajs_idxes, dim=0)
                if self.multiple_kine_source_obj_rot is not None:
                    envs_goal_obj_ornt_th = batched_index_select(self.multiple_kine_source_obj_rot, self.envs_kine_source_trajs_idxes, dim=0)
            
            # multiple_kine_source_obj_pos
            
            cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
            cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
            
            self.goal_pos = cur_goal_pos
            self.goal_rot = cur_goal_rot
        
        
        # if self.dataset_type == 'grab':
        compute_reward_func = compute_hand_reward_tracking
        
        # if self.use_forcasting_model: # 
        #     tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        #     tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
            
        #     # values, indices, dims # # dims #
        #     envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        #     envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
    
        #     # batched index select #
        #     envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) # nn_envs x 1
        #     # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}")
        #     cur_progress_buf = torch.clamp(self.progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        #     # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        #     # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        #     cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        #     cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        #     self.goal_pos= cur_goal_pos
        #     self.goal_rot = cur_goal_rot
            
        #     # tot_goal_hand_qs_th = self.tot_kine_qs
        #     tot_goal_hand_qs_th = self.tot_hand_preopt_res
        #     envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
        #     # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        #     cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #


        #     # if self.use_forcasting_model and self.already_forcasted:
        #     #     # 
        #     #     cur_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #


        #     ### current target hand pose, and the difference from the reference hand pos ###
        #     # cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        #     # self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        #     # self.ori_cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        #     # cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
            
            
        #     self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
            
            # delta_qpos
            
        # if using the tracking traj to comput reward --- we should recalculate the delta qpos and others? #
        # self.delta_qpos 
        # self.goal_pos 
        # self.goal_rot
        
        
        # elif self.dataset_type == 'taco':
        #     compute_reward_func = compute_hand_reward_tracking_taco
        # else:
        #     raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        
        # compute hand reward thrshold = 150 -- compute hand reward threshold = 0 #
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_reward_func( # compute hand tracking reward ##
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            envs_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.object_linvel, self.object_angvel,self.object_linvel, self.object_angvel,
            self.goal_pos, self.goal_rot, self.goal_lifting_pos,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos, # 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.rew_obj_pose_coef, self.goal_dist_thres, envs_hand_qtars, self.cur_targets, self.use_hand_actions_rew, self.prev_dof_vel, self.cur_dof_vel, self.rew_smoothness_coef, self.early_terminate, self.env_cond_type, self.env_cond_hand_masks, self.compute_hand_rew_buf_threshold , self.cur_ornt_rew_coef
            # when calculating the reard -- you still use the # # # # # get env
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        if self.print_success_stat:
            self.total_resets = self.total_resets + self.reset_buf.sum() # 
            direct_average_successes = self.total_successes + self.successes.sum()
            self.total_successes = self.total_successes + (self.successes * self.reset_buf).sum()

            # The direct average shows the overall result more quickly, but slightly undershoots long term
            # policy performance. # consecutive #
            print("Direct average consecutive successes = {:.1f}".format(
                direct_average_successes / (self.total_resets + self.num_envs)))
            if self.total_resets > 0:
                print("Post-Reset average consecutive successes = {:.1f}".format(
                    self.total_successes / self.total_resets))

    def compute_observations(self):
        
        # randomize the condition modes? # 
        if self.randomize_conditions:
            self.generate_random_mask_config()
        
        # compute observations #
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)



        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        #
        self.object_handle_pos = self.object_pos  ## + quat_apply(self.object_rot, to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.06)
        self.object_back_pos = self.object_pos # + quat_apply(self.object_rot,to_torch([1, 0, 0], device=self.device).repeat(self.num_envs, 1) * 0.04)
        # object linvel #
        self.object_linvel = self.root_state_tensor[self.object_indices, 7:10]
        self.object_angvel = self.root_state_tensor[self.object_indices, 10:13]



        idx = self.hand_body_idx_dict['palm']
        self.right_hand_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.08)
        # self.right_hand_pos = self.right_hand_pos + quat_apply(self.right_hand_rot,to_torch([0, 1, 0], device=self.device).repeat(self.num_envs, 1) * -0.02)

        # right hand finger
        idx = self.hand_body_idx_dict['index'] # in #
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
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        
        self.goal_lifting_pos = self.goal_states[:, 0:3] 
        
        # fingertip state # # nn_envs x nn_fingertipsx 13
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        
        
        if self.dataset_type == 'taco':
            progress_buf_indexes = torch.where(self.progress_buf >= self.hand_palm_world_poses.size(0), self.hand_palm_world_poses.size(0) - 1 + torch.zeros_like(self.progress_buf), self.progress_buf)
            # self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos #
            self.gt_hand_palm_pos = self.hand_palm_world_poses[progress_buf_indexes]
            self.gt_hand_thumb_pos = self.thumb_tip_world_poses[progress_buf_indexes]
            self.gt_hand_index_pos = self.index_tip_world_poses[progress_buf_indexes]
            self.gt_hand_middle_pos = self.middle_tip_world_poses[progress_buf_indexes]
            self.gt_hand_ring_pos = self.ring_tip_world_poses[progress_buf_indexes]
            
        
        ##### get the history obs ######
        # TODO: check whether history-zero is correctly set here
        expanded_history_progress_buf = self.progress_buf.unsqueeze(0).contiguous().repeat(self.history_buf_length, 1).contiguous() #
        expanded_history_range = torch.arange(self.history_buf_length, device=self.device).unsqueeze(-1).repeat(1, self.num_envs).contiguous()
        # history_progress_buf = torch.clamp(expanded_history_progress_buf - expanded_history_range, min=torch.zeros_like(expanded_history_progress_buf))
        self.history_shadow_hand_dof_pos = torch.where( # only the samehistory cn 
            expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_pos.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_pos.size(-1)), self.shadow_hand_dof_pos.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_shadow_hand_dof_pos
        )  #get the da
        self.history_shadow_hand_dof_vel = torch.where( # only the samehistory cn 
            expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_vel.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_vel.size(-1)), self.shadow_hand_dof_vel.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_shadow_hand_dof_vel
        )  #get the da
        self.history_fingertip_state = torch.where( # only the samehistory cn 
            expanded_history_progress_buf.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.history_fingertip_state.size(-2), self.history_fingertip_state.size(-1)) == expanded_history_range.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.history_fingertip_state.size(-2), self.history_fingertip_state.size(-1)), self.fingertip_state.unsqueeze(0).repeat(self.history_buf_length, 1, 1, 1), self.history_fingertip_state
        )  #get the da
        self.history_right_hand_pos = torch.where( # only the samehistory cn 
            expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_right_hand_pos.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_right_hand_pos.size(-1)), self.right_hand_pos.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_right_hand_pos
        )  #get the  # history right hand rot #
        self.history_right_hand_rot = torch.where(
            expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_right_hand_rot.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_right_hand_rot.size(-1)), self.right_hand_rot.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_right_hand_rot
        ) 
        try:
            cur_actions = self.actions
        except:
            cur_actions = torch.zeros((self.num_envs, self.nn_hand_dof), dtype=torch.float32, device=self.device)
        self.history_right_hand_actions = torch.where( 
            expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_right_hand_actions.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_right_hand_actions.size(-1)), cur_actions.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_right_hand_actions
        )  #get the da
        self.history_object_pose = torch.where( 
            expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_object_pose.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_object_pose.size(-1)), self.object_pose.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_object_pose
        )  #get the da
        ##### get the history obs ######
        
        
        
        if self.use_forcasting_model:
            first_env_progress_buf = self.progress_buf[0].item()
            if (not self.already_forcasted) or  (first_env_progress_buf % self.forcasting_inv_freq) == 0:
                self._forward_forcasting_model()
                self.try_save_network_forwarding_info_dict()
            
        
        if self.use_window_future_selection:
            # batched index select #
            envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) # nn_envs x 1
            # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}") #
        
            # step 1: find the goal ref states in a time window #
            tot_kine_qs = self.tot_kine_qs
            tot_goal_obj_trans = self.tot_kine_obj_trans
            tot_goal_obj_ornt = self.tot_kine_obj_ornt
            envs_kine_qs = batched_index_select(tot_kine_qs, self.env_inst_idxes, dim=0)
            envs_obj_trans = batched_index_select(tot_goal_obj_trans, self.env_inst_idxes, dim=0)
            envs_obj_ornt = batched_index_select(tot_goal_obj_ornt, self.env_inst_idxes, dim=0)
            # nn envs x nn ts x nn feat dim #
            ws_selection = 10
            # ws_selection = 20
            # prev_progress_buf_ws = torch.arange(-ws_selection, -1, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            # nex_progress_buf_ws = torch.arange(0, ws_selection, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            prev_progress_buf_ws = torch.arange(-ws_selection, 0, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            nex_progress_buf_ws = torch.arange(1, ws_selection, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
            
            ###### Selectf from both previous history and the future states ######
            progress_buf_ws = torch.cat([ prev_progress_buf_ws, nex_progress_buf_ws], dim=-1) + self.progress_buf.unsqueeze(-1)
            ###### Selectf from both previous history and the future states ######
            
            ###### Selectf from previous history  ######
            progress_buf_ws = prev_progress_buf_ws + self.progress_buf.unsqueeze(-1) # torch.clamp(progress_buf_ws, max=envs_maxx_episode_length_per_traj.unsqueeze(-1).repeat(1, progress_buf_ws.size(-1)), min=torch.zeros_like(progress_buf_ws))
            ###### Selectf from previous history  ######
            
            # progress buf ws #
            progress_buf_ws = torch.clamp(progress_buf_ws, max=envs_maxx_episode_length_per_traj.unsqueeze(-1).repeat(1, progress_buf_ws.size(-1)), min=torch.zeros_like(progress_buf_ws))
            ws_kine_qs = batched_index_select(envs_kine_qs, progress_buf_ws, dim=1) # nn_envs x ws x 22
            ws_obj_trans = batched_index_select(envs_obj_trans, progress_buf_ws, dim=1) # nn_envs x ws x 3 
            ws_obj_ornt = batched_index_select(envs_obj_ornt, progress_buf_ws, dim=1) # nn_envs x ws x 4
            
            cur_kine_qs = self.shadow_hand_dof_pos # nn_envs x nn_dof #
            cur_obj_pos = self.object_pos # nn_envs x 3 #
            cur_obj_ornt = self.object_rot # nn_envs x 4 #
            diff_kine_qs_w_ws_qs = torch.sum((cur_kine_qs.unsqueeze(1) - ws_kine_qs) ** 2, dim=-1) # nn_envs x ws #
            diff_obj_trans_w_ws_obj_trans = torch.sum((ws_obj_trans - cur_obj_pos.unsqueeze(1)) ** 2, dim=-1) # nn_envs x ws # # nn_envs x ws # 
            weighted_diff = diff_kine_qs_w_ws_qs * 0.3 + diff_obj_trans_w_ws_obj_trans * 0.7 # # 
            # nn_envs x ws # weighted diff #
            minn_ws_idxes = torch.argmin(weighted_diff, dim=-1) # nn_envs # minn ws idxes # # nn_envs # 
            ws_selected_progress_buf = batched_index_select(progress_buf_ws, minn_ws_idxes.unsqueeze(1), dim=1).squeeze(1) # nn_envs --> the re-selected progress buf #
            ws_selected_qs = batched_index_select(ws_kine_qs, minn_ws_idxes.unsqueeze(1), dim=1).squeeze(1) # nn_envs x 22 # 
            ws_selected_pos = batched_index_select(ws_obj_trans, minn_ws_idxes.unsqueeze(1), dim=1).squeeze(1) # nn_envs x 3 #
            ws_selected_ornt = batched_index_select(ws_obj_ornt, minn_ws_idxes.unsqueeze(1), dim=1).squeeze(1) # nn_envs x 4 #
            # get the ws slected ornt and pos ## get the ws ##
            # shold add them to a buffer that stores these selected values # # self.progress_buf -- that utilizes # self.progress_bufs #
            # ws selected qs; --- selected pos #
            # ws selected qs #
            self.ws_selected_progress_buf = ws_selected_progress_buf
            pass
        
        
        tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
        
        # goal obj trans th # 

        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        
        
        
        envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) # nn_envs x 1
        # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}")
        
        if self.use_window_future_selection:
            cur_progress_buf = torch.clamp(self.ws_selected_progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
            # cur_goal_pos = batched_index_select(envs_goal_obj
        else:
        
            cur_progress_buf = torch.clamp(self.progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
            
            cur_progress_buf_obj =  cur_progress_buf 
            # torch.clamp(self.progress_buf, max=torch.zeros_like(envs_maxx_episode_length_per_traj) + 190, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
            
        if self.random_shift_cond_freq or  self.preset_inv_cond_freq > 1:
            moded_progress_buf = cur_progress_buf // self.env_inv_cond_freq
            increase_nn = (cur_progress_buf > moded_progress_buf * self.env_inv_cond_freq).int()
            cur_progress_buf = (moded_progress_buf + increase_nn) * self.env_inv_cond_freq
            cur_progress_buf = torch.clamp(cur_progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(cur_progress_buf))
            
        
        if self.already_predicted_targets: # envs # envs # alreay pred
            # print(f"already_predicted_targets: {self.already_predicted_targets}")
            # cur_goal_pos = self.pred_tracking_targets_obj_pos.clone()
            # cur_goal_rot = self.pred_tracking_targets_obj_rot.clone()
            # envs_goal_obj_trans_th = batched_index_select(self.pred_tracking_targets_obj_pos, self.env_inst_idxes, dim=0)
            # envs_goal_obj_ornt_th = batched_index_select(self.pred_tracking_targets_obj_rot, self.env_inst_idxes, dim=0)
            
            # envs_goal_obj_trans_th = self.pred_tracking_targets_obj_pos
            # envs_goal_obj_ornt_th = self.pred_tracking_targets_obj_rot
            pass
            # pass
            
        # # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        
        if self.use_future_ref_as_obs_goal:
            # use future ref as obs goal #
            cur_progress_buf_obj = torch.clamp(self.progress_buf + 1, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        
        
        if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
            envs_goal_obj_trans_th = batched_index_select(self.multiple_kine_source_obj_pos, self.envs_kine_source_trajs_idxes, dim=0)
            # multiple_kine_source_obj_pos
            if self.multiple_kine_source_obj_rot is not None:
                # print("[debug] computing ornt")
                envs_goal_obj_ornt_th = batched_index_select(self.multiple_kine_source_obj_rot, self.envs_kine_source_trajs_idxes, dim=0)
        
        
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        
        if self.use_forcasting_model and self.already_forcasted:
            # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot #
            # maxx_cur_progress_buf, _ = torch.max
            # print(f"maxx_cur_progress_buf: {torch.max(cur_progress_buf)}, minn_cur_progress_buf: {torch.min(cur_progress_buf)}, forcast_obj_pos: {self.forcast_obj_pos.size()}, forcast_obj_rot: {self.forcast_obj_rot.size()}")
            cur_goal_pos = batched_index_select(self.forcast_obj_pos, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 # # get the current forcasted obj pos 
            cur_goal_rot = batched_index_select(self.forcast_obj_rot, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 # 
        
        
        
        # if self.already_predicted_targets:
        #     cur_goal_pos = self.pred_tracking_targets_obj_pos.clone()
        #     cur_goal_rot = self.pred_tracking_targets_obj_rot.clone()
        
        
        
        # cur_goal_pos = self.goal_obj_trans_th[self.progress_buf]
        # cur_goal_rot = self.goal_obj_rot_quat_th[self.progress_buf]
        
        
        self.goal_pos_ref = cur_goal_pos
        self.goal_rot_ref = cur_goal_rot
        self.goal_pose_ref = torch.cat(
            [self.goal_pos_ref, self.goal_rot_ref], dim=-1
        )
        
        if self.separate_stages:
            obj_lowest_z_less_than_thres = (self.object_pos[:, 2] < 0.19)
            self.goal_pos = torch.where(
                obj_lowest_z_less_than_thres.unsqueeze(-1).repeat(1, 3), self.goal_pos, self.goal_pos_ref
            )
            self.goal_rot = torch.where(
                obj_lowest_z_less_than_thres.unsqueeze(-1).repeat(1, 4), self.goal_rot, self.goal_rot_ref
            )
            self.goal_pose = torch.cat(
                [self.goal_pos, self.goal_rot], dim=-1
            )
        else:
            self.goal_pose = self.goal_pose_ref
            self.goal_rot = self.goal_rot_ref
            self.goal_pos  = self.goal_pos_ref
        
        # goal pose #
        if self.use_twostage_rew:
            grasping_frame_obj_pos = self.goal_obj_trans_th[self.cur_grasp_fr] + to_torch([0.0, 0.0, 0.1], device=self.device)
            grasping_frame_obj_ornt = self.goal_obj_rot_quat_th[self.cur_grasp_fr]
            expanded_grasping_frame_obj_pos = grasping_frame_obj_pos.unsqueeze(0).repeat(self.num_envs, 1)
            expanded_grasping_frame_obj_ornt = grasping_frame_obj_ornt.unsqueeze(0).repeat( self.num_envs, 1)
            grasp_manip_stages_flag_pos = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, 3)
            grasp_manip_stages_flag_rot = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, 4)
            
            if self.use_real_twostage_rew:
                self.goal_pos = torch.where(
                    grasp_manip_stages_flag_pos, expanded_grasping_frame_obj_pos, self.goal_pos
                )
                self.goal_rot = torch.where(
                    grasp_manip_stages_flag_rot, expanded_grasping_frame_obj_ornt, self.goal_rot
                )   
            else:
                self.goal_pos = expanded_grasping_frame_obj_pos
                self.goal_rot = expanded_grasping_frame_obj_ornt
            self.goal_pose = torch.cat(
                [self.goal_pos, self.goal_rot], dim=-1
            )
        if self.lifting_separate_stages:
            lifting_frame_obj_pos = self.goal_obj_trans_th[self.lift_fr]
            lifting_frame_obj_ornt = self.goal_obj_rot_quat_th[self.lift_fr]
            expanded_lifting_frame_obj_pos = lifting_frame_obj_pos.unsqueeze(0).repeat(self.num_envs, 1)
            expanded_lifting_frame_obj_ornt = lifting_frame_obj_ornt.unsqueeze(0).repeat( self.num_envs, 1)
            lifting_manip_stages_flag_pos = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, 3)
            lifting_manip_stages_flag_rot = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, 4)
            self.goal_pos = torch.where(
                lifting_manip_stages_flag_pos, expanded_lifting_frame_obj_pos, self.goal_pos
            )
            self.goal_rot = torch.where(
                lifting_manip_stages_flag_rot, expanded_lifting_frame_obj_ornt, self.goal_rot
            )
            # # # #
            self.goal_pose = torch.cat(
                [self.goal_pos, self.goal_rot], dim=-1
            )
        
            
        
        ## TODO: goal velocities may be noisy; so we do not add it ##
        # goal_linvel, goal_angvel # # goal angvel #
        # cur_goal_lin_vels = self.goal_obj_lin_vels_th[self.progress_buf]
        # cur_goal_ang_vels = self.goal_obj_ang_vels_th[self.progress_buf]
        # self.goal_linvel = cur_goal_lin_vels
        # self.goal_angvel = cur_goal_ang_vels
        # self.goal_vels = torch.cat(
        #     [self.goal_linvel, self.goal_angvel], dim=-1 # another thing is the only first frame setting #
        # )

        # # fingertip state # # nn_envs x nn_fingertipsx 13
        # self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        # self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

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
        self.delta_target_hand_pos = torch.zeros((3,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.rel_hand_rot = torch.zeros((4,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.delta_target_hand_rot = torch.zeros((4,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ### HACK ###
        
        
        tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
        
        # values, indices, dims #
        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        
        
        
        # if self.already_predicted_targets:
        #     # cur_goal_pos = self.pred_tracking_targets_obj_pos.clone()
        #     # cur_goal_rot = self.pred_tracking_targets_obj_rot.clone()
        #     # envs_goal_obj_trans_th = batched_index_select(self.pred_tracking_targets_obj_pos, self.env_inst_idxes, dim=0)
        #     # envs_goal_obj_ornt_th = batched_index_select(self.pred_tracking_targets_obj_rot, self.env_inst_idxes, dim=0)
        #     # #### goal obj trans th #### # #

        #     # envs_goal_obj_trans_th = self.pred_tracking_targets_obj_pos
        #     # envs_goal_obj_ornt_th = self.pred_tracking_targets_obj_rot
        #     pass
    
        # # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        # # print(f"envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}, maxx_cur_progress_buf: {torch.max(cur_progress_buf)}, minn_cur_progress_buf: {torch.min(cur_progress_buf)}")
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        
        # if self.use_forcasting_model and self.already_forcasted:
        #     # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot #
        #     cur_goal_pos = batched_index_select(self.forcast_obj_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 # # get the current forcasted obj pos 
        #     cur_goal_rot = batched_index_select(self.forcast_obj_rot, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 # # get the current forcasted obj rot # progress buf #
        
        # if self.already_predicted_targets:
        #     # cur_goal_pos = self.pred_tracking_targets_obj_pos.clone()
        #     # cur_goal_rot = self.pred_tracking_targets_obj_rot.clone() # 
        
        tot_goal_hand_qs_th = self.tot_hand_preopt_res
        envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
        envs_goal_hand_qs_kine_ref = envs_goal_hand_qs.clone()
        
        # if self.already_predicted_targets:
        #     envs_goal_hand_qs = self.pred_tracking_targets_hand_qs
        
        
        ### TODO: we first try to not to use the forcasted hand qpos ###
        if self.use_forcasting_model and self.already_forcasted:
            # # cur_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
            forcast_envs_goal_hand_qs = self.forcast_shadow_hand_dof_pos 
            # using_forcast_res_step_threshold = self.using_forcast_res_step_threshold
            # minn_ts_nn = min(envs_goal_hand_qs.size(1), forcast_envs_goal_hand_qs.size(1))
            # envs_goal_hand_qs[self.progress_buf >= using_forcast_res_step_threshold, : minn_ts_nn] = forcast_envs_goal_hand_qs[self.progress_buf >= using_forcast_res_step_threshold, : minn_ts_nn]
            
            envs_goal_hand_qs = forcast_envs_goal_hand_qs
            
            
        # if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
        if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
            # nn_kine_trajs x nn_hand_dofs 
            # (nn_envs, ) # (nn_envs, ) # multiple kine source trajs # # nn_envs x 
            envs_goal_hand_qs = batched_index_select(self.multiple_kine_source_trajs, self.envs_kine_source_trajs_idxes, dim=0)
            
        envs_goal_hand_qs_kine_ref = envs_goal_hand_qs.clone()
        
        # if self.already_predicted_targets:
        #     # cur_goal_pos = self.pred_tracking_targets_obj_pos.clone()
        #     # cur_goal_rot = self.pred_tracking_targets_obj_rot.clone()
        #     # envs_goal_obj_trans_th = batched_index_select(self.pred_tracking_targets_obj_pos, self.env_inst_idxes, dim=0)
        #     # envs_goal_obj_ornt_th = batched_index_select(self.pred_tracking_targets_obj_rot, self.env_inst_idxes, dim=0)
        #     envs_goal_hand_qs = batched_index_select(self.pred_tracking_targets_hand_pos, self.env_inst_idxes, dim=0)
        
        # envs_goal_hand_qs = self.pred_tracking_targets_hand_pos
        
        
        #####  comment out on 12.10 ######
        # if self.already_predicted_targets:
        #     envs_goal_hand_qs = self.pred_tracking_targets_hand_qs
        #####  comment out on 12.10 ######
        
        # use_future_ref_as_obs_goal
        cur_progress_buf_handqpos = cur_progress_buf.clone()
        
        
        # # use the futhre ref and n o activate the forecaster #
        # if self.use_future_ref_as_obs_goal and (not self.activate_forecaster):
        #     cur_progress_buf_handqpos = torch.clamp(self.progress_buf + 1, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
            
        # use the futhre ref and n o activate the forecaster #
        if self.use_future_ref_as_obs_goal:
            cur_progress_buf_handqpos = torch.clamp(self.progress_buf + 1, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        
        
        # when compute the reward --- goes back to 
        # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf_handqpos.unsqueeze(-1), dim=1).squeeze(1)

        # if self.already_predicted_targets: # predicted targets # 
        #     cur_hand_qpos_ref = self.pred_tracking_targets_obj_pos.clone()
        

        # ### TODO: we first try to not to use the forcasted hand qpos ###
        # if self.use_forcasting_model and self.already_forcasted:
        #     cur_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/tasks/allegro_hand_tracking_generalist.py
        

        ### current target hand pose, and the difference from the reference hand pos ###
        # cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        
        
        
        
        self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        
        # if self.already_predicted_targets:
        #     self.ori_cur_hand_qpos_ref = self.pred_tracking_targets_obj_pos.clone()
        
        
        ### TODO: we first try to not to use the forcasted hand qpos ### ## and only condition it on the object pos sequence prediction ###
        # if self.use_forcasting_model and self.already_forcasted:
        #     self.ori_cur_hand_qpos_ref = batched_index_select(self.forast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        
        ### next progress buffer ###
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.hand_qs_th.size(0) - 1)
        # nex_hand_qpos_ref = self.hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        # self.nex_hand_qpos_ref = nex_hand_qpos_ref # next progress buf #
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.maxx_kine_nn_ts - 1)
        
        
        #### NOTE: nex_hand_qpos_ref is used as the kinematic bias in the next step -- so we do not need to consider the effect of random_shift_cond_freq here --- just set it to the original nex_progress_buf! ####
        if self.use_window_future_selection:
            nex_progress_buf = torch.clamp(self.ws_selected_progress_buf + 1, min=torch.zeros_like(envs_maxx_episode_length_per_traj), max=envs_maxx_episode_length_per_traj)
        else:
            nex_progress_buf = torch.clamp(self.progress_buf + 1, min=torch.zeros_like(envs_maxx_episode_length_per_traj), max=envs_maxx_episode_length_per_traj)
        
        
        # if self.random_shift_cond_freq:
        #     moded_progress_buf = nex_progress_buf // self.env_inv_cond_freq
        #     increase_nn = (nex_progress_buf > moded_progress_buf * self.env_inv_cond_freq).int()
        #     nex_progress_buf = (moded_progress_buf + increase_nn) * self.env_inv_cond_freq
        #     nex_progress_buf = torch.clamp(nex_progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(nex_progress_buf))
        
        # print(f"envs_goal_hand_qs: {envs_goal_hand_qs.size()}, maxx_nex_progress_buf: {torch.max(nex_progress_buf)}, minn_nex_progress_buf: {torch.min(nex_progress_buf)}")
        
        ##### hand qpos ref ######
        nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        ##### hand qpos ref ######
        
        # # ##### hand qpos ref ######
        # nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs_kine_ref, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        # # ##### hand qpos ref ######
        
        
        # if self.already_predicted_targets:
        #     nex_hand_qpos_ref = self.pred_tracking_targets_obj_pos.clone()
        
        
        self.nex_hand_qpos_ref = nex_hand_qpos_ref
        
        
        # ###### NOTE: Version 3 of random conditions ######
        # if self.random_shift_cond:
        #     self.nex_hand_qpos_ref[self.env_cond_type == COND_OBJ] = envs_goal_hand_qs[self.env_cond_type == COND_OBJ, 0]
        # ###### NOTE: Version 3 of random conditions ######
        
        
        
        ### TODO: we first try to not to use the forcasted hand qpos ### ## and only condition it on the object pos sequence prediction ###
        # if self.use_forcasting_model and self.already_forcasted:
        #     # print(f"maxx_nex_progress_buf: {torch.max(nex_progress_buf)}, minn_nex_progress_buf: {torch.min(nex_progress_buf)}, forcast_shadow_hand_dof_pos: {self.forcast_shadow_hand_dof_pos.size()}")
        #     self.nex_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        #     nex_hand_qpos_ref  = self.nex_hand_qpos_ref # nex hand qpos ref # # nex hand qpos ref # # qpos ref # # video data -
        
        
        
        # if self.use_history_obs:
        #     # TODO: check whether history-zero is correctly set here
        #     expanded_history_progress_buf = self.progress_buf.unsqueeze(0).contiguous().repeat(self.history_buf_length, 1).contiguous() # get tu
        #     expanded_history_range = torch.arange(self.history_buf_length, device=self.device).unsqueeze(-1).repeat(1, self.num_envs).contiguous()
        #     # history_progress_buf = torch.clamp(expanded_history_progress_buf - expanded_history_range, min=torch.zeros_like(expanded_history_progress_buf))
        #     # use history obs #
        #     self.history_shadow_hand_dof_pos = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_pos.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_pos.size(-1)), self.shadow_hand_dof_pos.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_shadow_hand_dof_pos
        #     )  #get the da
        #     self.history_shadow_hand_dof_vel = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_vel.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_shadow_hand_dof_vel.size(-1)), self.shadow_hand_dof_vel.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_shadow_hand_dof_vel
        #     )  #get the da
        #     self.history_fingertip_state = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.history_fingertip_state.size(-2), self.history_fingertip_state.size(-1)) == expanded_history_range.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, self.history_fingertip_state.size(-2), self.history_fingertip_state.size(-1)), self.fingertip_state.unsqueeze(0).repeat(self.history_buf_length, 1, 1, 1), self.history_fingertip_state
        #     )  #get the da
        #     self.history_right_hand_pos = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_right_hand_pos.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_right_hand_pos.size(-1)), self.right_hand_pos.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_right_hand_pos
        #     )  #get the  # history right hand rot #
        #     self.history_right_hand_rot = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_right_hand_rot.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_right_hand_rot.size(-1)), self.right_hand_rot.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_right_hand_rot
        #     )  #get the da
        #     try:
        #         cur_actions = self.actions
        #     except:
        #         cur_actions = torch.zeros((self.num_envs, self.nn_hand_dof), dtype=torch.float32, device=self.device)
        #     self.history_right_hand_actions = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_right_hand_actions.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_right_hand_actions.size(-1)), cur_actions.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_right_hand_actions
        #     )  #get the da
        #     self.history_object_pose = torch.where( # only the samehistory cn 
        #         expanded_history_progress_buf.unsqueeze(-1).repeat(1, 1, self.history_object_pose.size(-1)) == expanded_history_range.unsqueeze(-1).repeat(1, 1, self.history_object_pose.size(-1)), self.object_pose.unsqueeze(0).repeat(self.history_buf_length, 1, 1), self.history_object_pose
        #     )  
        #     ##### get the history obs ###### # obs #
        # wforecast #
            
            
            
        if self.use_history_obs:
            self.tot_history_hand_dof_pos = []
            self.tot_history_hand_dof_vel = []
            self.tot_history_fingertip_state = []
            self.tot_history_right_hand_pos = []
            self.tot_history_right_hand_rot = []
            self.tot_history_right_hand_actions = []
            self.tot_history_object_pose = []
            # history_freq = 1
            history_freq = self.history_freq
            # for i_history_step in range(self.history_length):
            for i_history_step in range(self.history_length - 1, -1, -1):
                cur_progress_buf = torch.clamp(self.progress_buf - i_history_step * history_freq, min=torch.zeros_like(self.progress_buf))
                trans_shadow_hand_dof_pos = self.history_shadow_hand_dof_pos.contiguous().transpose(1, 0)
                cur_hist_shadow_hand_dof_pos = batched_index_select(trans_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                trans_shadow_hand_dof_vel = self.history_shadow_hand_dof_vel.contiguous().transpose(1, 0)
                cur_hist_shadow_hand_dof_vel = batched_index_select(trans_shadow_hand_dof_vel, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                trans_history_fingertip_state = self.history_fingertip_state.contiguous().transpose(1, 0)
                cur_hist_fingertip_state = batched_index_select(trans_history_fingertip_state, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                trans_history_right_hand_pos = self.history_right_hand_pos.contiguous().transpose(1, 0)
                cur_hist_right_hand_pos = batched_index_select(trans_history_right_hand_pos, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                trans_history_right_hand_rot = self.history_right_hand_rot.contiguous().transpose(1, 0)
                cur_hist_right_hand_rot = batched_index_select(trans_history_right_hand_rot, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                trans_history_right_hand_actions = self.history_right_hand_actions.contiguous().transpose(1, 0)
                cur_hist_right_hand_actions = batched_index_select(trans_history_right_hand_actions, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                trans_history_object_pose = self.history_object_pose.contiguous().transpose(1, 0)
                cur_hist_object_pose = batched_index_select(trans_history_object_pose, cur_progress_buf.unsqueeze(1), dim=1).squeeze(1)
                self.tot_history_hand_dof_pos.append(cur_hist_shadow_hand_dof_pos)
                self.tot_history_hand_dof_vel.append(cur_hist_shadow_hand_dof_vel)
                self.tot_history_fingertip_state.append(cur_hist_fingertip_state)
                self.tot_history_right_hand_pos.append(cur_hist_right_hand_pos)
                self.tot_history_right_hand_rot.append(cur_hist_right_hand_rot)
                self.tot_history_right_hand_actions.append(cur_hist_right_hand_actions)
                self.tot_history_object_pose.append(cur_hist_object_pose)
            self.tot_history_hand_dof_pos = torch.stack(self.tot_history_hand_dof_pos, dim=1)
            self.tot_history_hand_dof_vel = torch.stack(self.tot_history_hand_dof_vel, dim=1)
            self.tot_history_fingertip_state = torch.stack(self.tot_history_fingertip_state, dim=1)
            self.tot_history_right_hand_pos = torch.stack(self.tot_history_right_hand_pos, dim=1)
            self.tot_history_right_hand_rot = torch.stack(self.tot_history_right_hand_rot, dim=1)
            self.tot_history_right_hand_actions = torch.stack(self.tot_history_right_hand_actions, dim=1)
            self.tot_history_object_pose = torch.stack(self.tot_history_object_pose, dim=1) # nn_envs x history_buf_length x 7 #
            # nn_envs x nn_envs #
            pass
        
        
        
        if self.use_twostage_rew:
            # grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
            grasp_frame_hand_qpos = self.goal_hand_qs_th[self.cur_grasp_fr]
            # expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            # self.shadow_hand_dof_pos = torch.where(
            #     self.grasp_manip_stages == 0, expanded_grasp_frame_hand_qpos, self.shadow_hand_dof_pos
            # )
            self.grasp_frame_hand_qpos = grasp_frame_hand_qpos # 
            expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            grasp_manip_stages_flag_qpos = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
            
            if self.use_real_twostage_rew:
                self.nex_hand_qpos_ref = torch.where(
                    grasp_manip_stages_flag_qpos, expanded_grasp_frame_hand_qpos, self.nex_hand_qpos_ref
                )
                cur_hand_qpos_ref = torch.where(
                    grasp_manip_stages_flag_qpos, expanded_grasp_frame_hand_qpos, cur_hand_qpos_ref
                )
            else:
                self.nex_hand_qpos_ref = expanded_grasp_frame_hand_qpos
                cur_hand_qpos_ref = expanded_grasp_frame_hand_qpos

            self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
            
        if self.lifting_separate_stages:
            # lifting_frame_hand_qpos = self.hand_qs_th[self.lift_fr]
            lifting_frame_hand_qpos = self.goal_hand_qs_th[self.lift_fr]
            expanded_lifting_frame_hand_qpos = lifting_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            lifting_manip_stages_flag_qpos = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
            self.nex_hand_qpos_ref = torch.where(
                lifting_manip_stages_flag_qpos, expanded_lifting_frame_hand_qpos, self.nex_hand_qpos_ref
            )
            cur_hand_qpos_ref = torch.where(
                lifting_manip_stages_flag_qpos, expanded_lifting_frame_hand_qpos, cur_hand_qpos_ref
            )
            self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        
        
        if self.test:
            # object pose np ## -- curretn step observations; #
            self.object_pose_np = self.object_pose.detach().cpu().numpy()
            self.shadow_hand_dof_pos_np = self.shadow_hand_dof_pos.detach().cpu().numpy()
            self.target_qpos_np = self.cur_targets.detach().cpu().numpy()[:, : self.num_shadow_hand_dofs]
            self.shadow_hand_dof_vel_np = self.shadow_hand_dof_vel.detach().cpu().numpy() 
            self.object_linvel_np = self.object_linvel.detach().cpu().numpy()
            self.object_angvel_np = self.object_angvel.detach().cpu().numpy()
            self.obs_buf_np = self.obs_buf.detach().cpu().numpy()[:100] ## get the observation buffers ##
            if self.ref_ts > 0:
                self.actions_np = self.actions.detach().cpu().numpy()[:100]
            else:
                self.actions_np = np.zeros((self.num_envs, self.num_shadow_hand_dofs), dtype=np.float32)[:100]
            
            # cur_ts_forcast_hand_dof_pos = self.forcast_shadow_hand_dof_pos.detach().cpu().numpy()[:, self.ref_ts]
            # cur_ts_forcast_obj_pos = self.forcast_obj_pos.detach().cpu().numpy()[:, self.ref_ts]
            # cur_ts_forcast_obj_ornt = self.forcast_obj_rot.detach().cpu().numpy()[:, self.ref_ts]
            
            if self.use_forcasting_model and self.already_forcasted:
                cur_ts_forcast_hand_dof_pos = self.forcast_shadow_hand_dof_pos.detach().cpu().numpy()[:, self.ref_ts]
                cur_ts_forcast_obj_pos = self.forcast_obj_pos.detach().cpu().numpy()[:, self.ref_ts]
                cur_ts_forcast_obj_ornt = self.forcast_obj_rot.detach().cpu().numpy()[:, self.ref_ts]
            else:
                cur_ts_forcast_hand_dof_pos = self.shadow_hand_dof_pos_np
                cur_ts_forcast_obj_pos = self.object_pose_np[..., :3]
                cur_ts_forcast_obj_ornt = self.object_pose_np[..., 3:]
                
                
            if self.already_predicted_targets: # jointly 
                # self.cur_pred_targets_hand_qs = pred_targets_hand_qs[:, 1]
                # self.cur_pred_targets_obj_pos = pred_targets_obj_pos[:, 1]
                # self.cur_pred_targets_obj_rot = pred_targets_obj_rot[:, 1] # pred targets obj rot #
                cur_ts_forcast_hand_dof_pos = self.cur_pred_targets_hand_qs.detach().cpu().numpy()
                cur_ts_forcast_obj_pos = self.cur_pred_targets_obj_pos.detach().cpu().numpy()
                cur_ts_forcast_obj_ornt = self.cur_pred_targets_obj_rot.detach().cpu().numpy()
                
            self.ts_to_hand_obj_states[self.ref_ts] = {
                'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np,
                'shadow_hand_dof_tars': self.target_qpos_np,
                'object_pose': self.object_pose_np,
                'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np,
                'object_linvel': self.object_linvel_np,
                'object_angvel': self.object_angvel_np,
                'actions': self.actions_np , 
                'observations': self.obs_buf_np,
                'forcast_hand_dof_pos': cur_ts_forcast_hand_dof_pos,
                'forcast_obj_pos': cur_ts_forcast_obj_pos,
                'forcast_obj_ornt': cur_ts_forcast_obj_ornt
            }
            # self.ts_to_hand_obj_states[self.ref_ts] # 
        else:
            # if not self.single_instance_training and self.num_envs < 1000: # get the sv info during training #
            # if not self.single_instance_training and self.sv_info_during_training:
            if self.sv_info_during_training:
                # object pose np ## -- curretn step observations; # # 
                self.object_pose_np = self.object_pose.detach().cpu().numpy()
                self.shadow_hand_dof_pos_np = self.shadow_hand_dof_pos.detach().cpu().numpy()
                # self.target_qpos_np = self.cur_targets.detach().cpu().numpy()[:, : self.num_shadow_hand_dofs]
                # self.shadow_hand_dof_vel_np = self.shadow_hand_dof_vel.detach().cpu().numpy() 
                # self.object_linvel_np = self.object_linvel.detach().cpu().numpy()
                # self.object_angvel_np = self.object_angvel.detach().cpu().numpy()
                # self.obs_buf_np = self.obs_buf.detach().cpu().numpy()[:100] ## get the observation buffers ##
                # if self.ref_ts > 0:
                #     self.actions_np = self.actions.detach().cpu().numpy()[:100]
                # else:
                #     self.actions_np = np.zeros((self.num_envs, self.num_shadow_hand_dofs), dtype=np.float32)[:100]
                
                # if self.use_forcasting_model and self.already_forcasted:
                #     cur_ts_forcast_hand_dof_pos = self.forcast_shadow_hand_dof_pos.detach().cpu().numpy()[:, self.ref_ts]
                #     cur_ts_forcast_obj_pos = self.forcast_obj_pos.detach().cpu().numpy()[:, self.ref_ts]
                #     cur_ts_forcast_obj_ornt = self.forcast_obj_rot.detach().cpu().numpy()[:, self.ref_ts]
                # else:
                #     cur_ts_forcast_hand_dof_pos = self.shadow_hand_dof_pos_np
                #     cur_ts_forcast_obj_pos = self.object_pose_np[..., :3]
                #     cur_ts_forcast_obj_ornt = self.object_pose_np[..., 3:]
                nn_envs = 100
                self.ts_to_hand_obj_states[self.ref_ts] = {
                    'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np[:nn_envs],
                    # 'shadow_hand_dof_tars': self.target_qpos_np[:nn_envs],
                    'object_pose': self.object_pose_np[:nn_envs],
                    # 'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np[:nn_envs],
                    # 'object_linvel': self.object_linvel_np[:nn_envs],
                    # 'object_angvel': self.object_angvel_np[:nn_envs],
                    # 'actions': self.actions_np[:nn_envs] , 
                    # 'observations': self.obs_buf_np[:nn_envs],
                    # 'forcast_hand_dof_pos': cur_ts_forcast_hand_dof_pos[:nn_envs],
                    # 'forcast_obj_pos': cur_ts_forcast_obj_pos[:nn_envs],
                    # 'forcast_obj_ornt': cur_ts_forcast_obj_ornt[:nn_envs]
                }
                
        
        # self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos
        self.compute_full_state()

        if self.asymmetric_obs: 
            self.compute_full_state(True)

    def get_unpose_quat(self):
        if self.repose_z:
            self.unpose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                -self.z_theta,
            )
        return

    def unpose_point(self, point):
        if self.repose_z:
            return self.unpose_vec(point)
            # return self.origin + self.unpose_vec(point - self.origin)
        return point

    def unpose_vec(self, vec):
        if self.repose_z:
            return quat_apply(self.unpose_z_theta_quat, vec)
        return vec

    def unpose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.unpose_z_theta_quat, quat)
        return quat

    def unpose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.unpose_point(state[:, 0:3])
            state[:, 3:7] = self.unpose_quat(state[:, 3:7])
            state[:, 7:10] = self.unpose_vec(state[:, 7:10])
            state[:, 10:13] = self.unpose_vec(state[:, 10:13])
        return state

    def get_pose_quat(self):
        if self.repose_z:
            self.pose_z_theta_quat = quat_from_euler_xyz(
                torch.zeros_like(self.z_theta), torch.zeros_like(self.z_theta),
                self.z_theta,
            )
        return

    def pose_vec(self, vec): # p
        if self.repose_z:
            return quat_apply(self.pose_z_theta_quat, vec)
        return vec

    def pose_point(self, point):
        if self.repose_z:
            return self.pose_vec(point)
            # return self.origin + self.pose_vec(point - self.origin)
        return point

    def pose_quat(self, quat):
        if self.repose_z:
            return quat_mul(self.pose_z_theta_quat, quat)
        return quat

    def pose_state(self, state):
        if self.repose_z:
            state = state.clone()
            state[:, 0:3] = self.pose_point(state[:, 0:3])
            state[:, 3:7] = self.pose_quat(state[:, 3:7])
            state[:, 7:10] = self.pose_vec(state[:, 7:10])
            state[:, 10:13] = self.pose_vec(state[:, 10:13])
        return state
    
    
    
    
    # TODO: whether we need to change rewards? #
    def generate_random_mask_config(self, ):
        if self.add_contact_conditions:
            # tot_contact_infos
            condition_mask_type = torch.randint(0, 3, (1, ))
        else:
            # condition_mask_type = torch.randint(0, 4, (1, ))
            condition_mask_type = torch.randint(0, 3, (1, ))
        condition_mask_type = condition_mask_type[0].item()
        
        if self.randomize_condition_type == 'hand':
            condition_mask_type = MASK_HAND # mask hand 
        elif self.randomize_condition_type == 'obj':
            condition_mask_type = MASK_OBJ
        elif self.randomize_condition_type == 'hand_rndindex':
            condition_mask_type = MASK_HAND_RNDIDX 
        # elif self.randomize_condition_type[:len('frame')] == 'frame':
        #     condition_mask_type = 0
        # elif self.randomize_condition_type == 'contact':
        #     condition_mask_type = 4 
        # elif self.randomize_condition_type == 'objpos':
        #     condition_mask_type = 4
        
        if condition_mask_type == 2:
            rnd_masked_ratio = np.random.uniform(0, 1, (1,) )[0].item() # a masking ratio from zero to one #
            rnd_masked_joints_nn = int(rnd_masked_ratio * self.nn_hand_dof) # a masking ratio from zero to one #
            rnd_selected_hand_joints = np.random.permutation(self.nn_hand_dof)[: rnd_masked_joints_nn]
            rnd_selected_hand_joints = torch.tensor(rnd_selected_hand_joints, dtype=torch.long, device=self.rl_device) # 
            self.rnd_selected_hand_joints = rnd_selected_hand_joints
        self.condition_mask_type = condition_mask_type # add the condition mask type # # use the planned results as the values to train the controller #
    
    

    def compute_full_state(self, asymm_obs=False): #
        
        
        
        self.get_unpose_quat()
        
        

        # 2 * nn_hand_dofs + 13 * num_fingertips + 6 + nn_hand_dofs + 16 + 7 + nn_hand_dofs ## 

        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##
        
        if self.use_future_obs:
            # we have 0.25 possibility to mask out key frames # 1/nn_future_frame for masiking out each number of frame; then we randomly select frames of that number to mask out #
            # we have 0.25 possibility to mask out joints # we uniformly randomly select the mask ratio from 0.0 to 0.8; then we randomly select which to mask out #
            # we have 0.25 possibitilyt to mask all hand future conditions #
            # we have 0.25 possibility to mask out all object future conditions 
            
            
            # 1) random key frame masks -- 0.2, 2) random joint masks -- 0.4; disable some information #
            # use the futuer obs # use the futuer obs # # tot hand qtars #
            # envs_hand_qtars = batched_index_select(self.tot_hand_qtars, self.env_inst_idxes, dim=0)
            envs_hand_qtars = batched_index_select(self.tot_kine_qs, self.env_inst_idxes, dim=0)
            # envs hand qtars # # envs hand qtars #
            # print(f"env_inst_idxes: {torch.max(self.env_inst_idxes)}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj.size()}, max_episode_length: {self.maxx_episode_length_per_traj}")
            envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
            # cur_progress_buf = torch.clamp(self.progress_buf, min=torch.zeros_like(envs_episode_length), max=envs_episode_length) # 
            # envs_hand_qtars = batched_index_select(envs_hand_qtars, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # # squeeze # #

            future_ws = self.history_length
            future_freq = self.history_freq
            ranged_future_ws = torch.arange(future_ws, device=self.device).unsqueeze(0).repeat(self.num_envs, 1) * future_freq
            # nn_envs x nn_future_ws #
            increased_progress_buf = self.progress_buf.unsqueeze(-1).contiguous().repeat(1, future_ws).contiguous() + ranged_future_ws
            
            if self.random_shift_cond_freq or  self.preset_inv_cond_freq > 1:
                moded_progress_buf = increased_progress_buf // self.env_inv_cond_freq.unsqueeze(-1).repeat(1, future_ws).contiguous()
                increase_nn = (increased_progress_buf > moded_progress_buf * self.env_inv_cond_freq.unsqueeze(-1).repeat(1, future_ws).contiguous()).int()
                increased_progress_buf = (moded_progress_buf + increase_nn) * self.env_inv_cond_freq.unsqueeze(-1).repeat(1, future_ws).contiguous()
                
            
            
            future_progress_buf = torch.clamp(increased_progress_buf, min=torch.zeros_like(envs_episode_length).unsqueeze(-1).repeat(1, future_ws).contiguous(), max=envs_episode_length.unsqueeze(-1).repeat(1, future_ws).contiguous())
            
            ### TODO: add the shfited freq inv div for future progress buf ###
            
            
            #### get the future hand qtars #### # only track the next target state #
            # nn_envs x nn_ts x nn_qs_dim --> nn_envs x nn_future_ts x nn_q_dims
            future_hand_qtars = batched_index_select(envs_hand_qtars, future_progress_buf, dim=1)  # nn_envs x nn_future_ws x nn_hand_dof #
            #### get the future hand qtars ####
            
            #### get the future goal obj pos and obj rot ####
            envs_obj_goal_pos = batched_index_select(self.tot_kine_obj_trans, self.env_inst_idxes, dim=0) # 
            envs_obj_goal_rot = batched_index_select(self.tot_kine_obj_ornt, self.env_inst_idxes, dim=0)
            # cur_goal_pos = batched_index_select(envs_obj_goal_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
            # cur_goal_rot = batched_index_select(envs_obj_goal_rot, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
            future_goal_pos = batched_index_select(envs_obj_goal_pos, future_progress_buf, dim=1) # nn_envs x nn_future_ws x 3 #
            future_goal_rot = batched_index_select(envs_obj_goal_rot, future_progress_buf, dim=1) # nn_envs x nn_future_ws x 4 #
            #### get the future goal obj pos and obj rot ####
            
            
            
            
            if self.use_forcasting_model and self.already_forcasted:
                print(f"maxx_future_progress_buf: {torch.max(future_progress_buf)}, minn_future_progress_buf: {torch.min(future_progress_buf)}")
                future_hand_qtars = batched_index_select(self.forcast_shadow_hand_dof_pos, future_progress_buf, dim=1) # nn_envs x nn_future_ws x nn_hand_dof #
                future_goal_pos = batched_index_select(self.forcast_obj_pos, future_progress_buf, dim=1) # nn_envs x nn_future_ws x 3 #
                future_goal_rot = batched_index_select(self.forcast_obj_rot, future_progress_buf, dim=1) # 
                
            
            # we have 0.25 possibility to mask out key frames # 1/nn_future_frame for masiking out each number of frame; then we randomly select frames of that number to mask out #
            # we have 0.25 possibility to mask out joints # we uniformly randomly select the mask ratio from 0.0 to 0.8; then we randomly select which to mask out #
            # we have 0.25 possibitilyt to mask all hand future conditions #
            # we have 0.25 possibility to mask out all object future conditions #
            
            full_future_hand_qtars = future_hand_qtars.clone()
            full_future_goal_pos = future_goal_pos.clone()
            full_future_goal_rot = future_goal_rot.clone()
            
            ##### NOTE: Version 1 of the condition randomization #####
            # if self.randomize_conditions:
            #     # print(f"Randomizing conditions")
            #     # condition mask type # 
            #     # if we mask out the hand --- 
            #     # we have some random masked hand type #
            #     # 1) mask out  the total future hand #
            #     # 2) mask out 
            #     if self.add_contact_conditions:
            #         # tot_contact_infos
            #         condition_mask_type = torch.randint(0, 5, (1, ))
            #     # condition_mask_type = condition_mask_type[0].item()
            #     else:
            #         # condition_mask_type = torch.randint(0, 4, (1, ))
            #         condition_mask_type = torch.randint(0, 5, (1, ))
            #     condition_mask_type = condition_mask_type[0].item()
                
            #     if self.randomize_condition_type == 'hand':
            #         condition_mask_type = 3
            #     elif self.randomize_condition_type == 'obj':
            #         condition_mask_type = 2
            #     elif self.randomize_condition_type[:len('frame')] == 'frame':
            #         condition_mask_type = 0
            #     elif self.randomize_condition_type == 'contact':
            #         condition_mask_type = 4 
            #     elif self.randomize_condition_type == 'objpos':
            #         condition_mask_type = 4
                
                
            #     if condition_mask_type == 0: # conditional model training ##
            #         # nn_future_frames # # nn_future_frames; nn future frames #
            #         selected_nn_masked_frames = torch.randint(0, future_ws + 1, (1, ))
            #         selected_nn_masked_frames = selected_nn_masked_frames[0].item() # an int number #
            #         selected_future_frame_index = np.random.permutation(future_ws)[:selected_nn_masked_frames]   #j
            #         selected_future_frame_index = torch.from_numpy(selected_future_frame_index).to(self.device).long() # selected future frame index #
                    
            #         if self.randomize_condition_type[:len('frame')] == 'frame':
            #             frame_idx = int(self.randomize_condition_type[len('frame_'):])
            #             selected_future_frame_index = [_ for _ in range(future_ws) if _ != frame_idx]
            #             selected_future_frame_index = torch.from_numpy(np.array(selected_future_frame_index)).to(self.device).long()
                    
                    
            #         # mask out features in these frames # 
            #         future_hand_qtars[:, selected_future_frame_index] = 0.0
            #         future_goal_pos[:, selected_future_frame_index] = 0.0
            #         future_goal_rot[:, selected_future_frame_index] = 0.0
            #     elif condition_mask_type == 1:
            #         joint_mask_ratio = np.random.uniform(0.0, 0.8)
            #         nn_joints = future_hand_qtars.size(1)
            #         selected_nn_joints = int(joint_mask_ratio * nn_joints)
            #         selected_joint_index = np.random.permutation(nn_joints)[:selected_nn_joints]
            #         selected_joint_index = torch.from_numpy(selected_joint_index).to(self.device).long() # selected joint index #
            #         # mask out features in these joints #
            #         future_hand_qtars[..., selected_joint_index] = 0.0
            #     elif condition_mask_type == 2:
            #         future_hand_qtars[:] = 0.0
            #     elif condition_mask_type == 3:
            #         future_goal_pos[:] = 0.0
            #         future_goal_rot[:] = 0.0
            #     elif condition_mask_type == 4:
            #         if self.add_contact_conditions:
            #             ## TODO: it is for the model training ##
            #             ## TODO: we need the forcasting model to predict such contact maps and select future contact maps from the forecasted results directly #
            #             envs_contact_maps = batched_index_select(self.tot_contact_infos, self.env_inst_idxes, dim=0)
            #             future_contact_maps = batched_index_select(envs_contact_maps, future_progress_buf, dim=1)
            #             future_goal_pos[:] = 0.0
            #             future_goal_rot[:] = 0.0
            #             future_hand_qtars[:] = 0.0
            #             future_hand_qtars[..., :future_contact_maps.size(-1)] = future_contact_maps
            #         else:
            #             future_hand_qtars[:] = 0.0
            #             future_goal_rot[:] = 0.0
            #     # elif condition_mask_type == 5:
            #     #     future_hand_qtars[:] = 0.0
            #     #     future_goal_rot[:] = 0.0
            ##### NOTE: Version 1 of the condition randomization #####
            
            
            future_hand_qtars[..., :3] = future_hand_qtars[..., :3] - self.object_pos[..., :].unsqueeze(1)
            future_goal_pos = future_goal_pos - self.object_pos.unsqueeze(1)
            
            
            ##### NOTE: version 2 of the condition (goal) randomization #####
            if self.randomize_conditions:
                if self.condition_mask_type == MASK_HAND:
                    future_hand_qtars[:] = 0.0
                elif self.condition_mask_type == MASK_OBJ:
                    future_goal_pos[:] = 0.0
                    future_goal_rot[:] = 0.0
                elif self.condition_mask_type == MASK_HAND_RNDIDX:
                    future_hand_qtars[..., self.rnd_selected_hand_joints] = 0.0
            ##### NOTE: version 2 of the condition (goal) randomization #####
            
            ##### NOTE: version 3 of the condition (goal) randomization #####
            # if self.random_shift_cond:
            future_hand_qtars[self.env_cond_type == COND_OBJ] = 0.0
            # future_goal_pos[self.env_cond_type == COND_HAND] = 0.0
            # future_goal_rot[self.env_cond_type == COND_HAND] = 0.0
            future_hand_qtars[self.env_cond_type == COND_PARTIALHAND_OBJ] = future_hand_qtars[self.env_cond_type == COND_PARTIALHAND_OBJ] * self.env_cond_hand_masks[self.env_cond_type == COND_PARTIALHAND_OBJ].unsqueeze(1) # nn_envs_cond x nn_future_ts x nn_hand_dof xxxxxx nn_envs_cond x 1 x nn_hand_dof
            ##### NOTE: version 3 of the condition (goal) randomization #####
                
            future_feats = torch.cat([future_goal_pos, future_goal_rot, future_hand_qtars], dim=-1)
            future_feats = future_feats.contiguous().view(self.num_envs, -1).contiguous()
            
            full_future_hand_qtars[..., :3] = full_future_hand_qtars[..., :3] - self.object_pos[..., :].unsqueeze(1)
            full_future_goal_pos = full_future_goal_pos - self.object_pos.unsqueeze(1)
            full_future_feats = torch.cat([full_future_goal_pos, full_future_goal_rot, full_future_hand_qtars], dim=-1)
            full_future_feats = full_future_feats.contiguous().view(self.num_envs, -1).contiguous()
            
            
            
        
        
        if not self.use_history_obs:
            
            if self.use_local_canonical_state:
                # local canonicalizations #
                # print(f"using local canonicalizations")
                canon_shadow_hand_dof_trans = self.shadow_hand_dof_pos[..., :3] - self.object_pos[..., :]
                canon_shadow_hand_dof = torch.cat(
                    [ canon_shadow_hand_dof_trans, self.shadow_hand_dof_pos[..., 3:] ], dim=-1 
                )
            else:
                canon_shadow_hand_dof = self.shadow_hand_dof_pos 
            
            
            self.cur_dof_vel[:, : ] = self.shadow_hand_dof_vel

            # get the obs buf #
            # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos, # 
            #                                                        self.shadow_hand_dof_lower_limits, # 
            #                                                        self.shadow_hand_dof_upper_limits) #
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(canon_shadow_hand_dof,
                                                                self.shadow_hand_dof_lower_limits,
                                                                self.shadow_hand_dof_upper_limits)
            if self.wo_vel_obs:
                self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = 0.0
            else:
                self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
            
            if self.obs_type == "full_state" or asymm_obs:
                self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
            
                fingertip_obs_start = 3 * self.num_shadow_hand_dofs
            else:
                fingertip_obs_start = 2 * self.num_shadow_hand_dofs
            
            
            if self.use_local_canonical_state:
                canon_fingertip_pos = self.fingertip_pos - self.object_pos.unsqueeze(1)
                canon_fingertip_pose = torch.cat(
                    [ canon_fingertip_pos, self.fingertip_state[..., 3:] ], dim=-1
                )
            else:
                canon_fingertip_pose = self.fingertip_state
        
            # aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            aux = canon_fingertip_pose.reshape(self.num_envs, num_ft_states)
            for i in range(self.num_fingertips):
                aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
            # 66:131: ft states
            self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

            # 131:161: ft sensors: do not need repose
            if self.obs_type == "full_state" or asymm_obs:
                self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :num_ft_force_torques]

                hand_pose_start = fingertip_obs_start + num_ft_states + num_ft_force_torques #  95
            else:
                hand_pose_start = fingertip_obs_start + num_ft_states
            # 161:167: hand_pose
            ### Global hand pose ###
            
            
            if self.use_local_canonical_state:
                canon_right_hand_pos = self.right_hand_pos - self.object_pos
            else:
                canon_right_hand_pos = self.right_hand_pos
            
            if self.tight_obs:
                # self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
                self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.unpose_point(canon_right_hand_pos)
                euler_xyz = get_euler_xyz(self.unpose_quat(self.right_hand_rot))
            else:
                # self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
                self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(canon_right_hand_pos)
                euler_xyz = get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
            self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
            self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
            self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)
                
            # Actions #
            action_obs_start = hand_pose_start + 6
            # 167:191: action #
            try:
                aux = self.actions[:, :self.num_shadow_hand_dofs]
            except: # 
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
            self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(canon_object_pos)
            self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
            if self.wo_vel_obs:
                self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = 0.0
                self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = 0.0
            else: # object 
                self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
                self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        else:
            # print(f"[Debug] Adding history obs")
            if self.use_local_canonical_state: # using local #
                # print(f"using local canonicalizations") # using local #
                # tot_history_hand_dof_pos, tot_history_hand_dof_vel, tot_history_fingertip_state, tot_history_right_hand_pos, tot_history_right_hand_rot, tot_history_right_hand_actions, tot_history_object_pose #
                # histroy_hand_dof_pos: nn_envs x nn_hist_length x nn_hand_dof #
                canon_shadow_hand_dof_trans = self.tot_history_hand_dof_pos[..., :3] - self.object_pos[..., :].unsqueeze(1) # unsqueeze the history dimension
                canon_shadow_hand_dof = torch.cat(
                    [ canon_shadow_hand_dof_trans, self.tot_history_hand_dof_pos[..., 3:] ], dim=-1
                )
                # canon_shadow_hand_dof_trans = self.shadow_hand_dof_pos[..., :3] - self.object_pos[..., :]
                # canon_shadow_hand_dof = torch.cat( #
                #     [ canon_shadow_hand_dof_trans, self.shadow_hand_dof_pos[..., 3:] ], dim=-1 #
                # ) # canon shadow hand dof #
            else:
                canon_shadow_hand_dof = self.tot_history_hand_dof_pos 
            
            
            self.cur_dof_vel[:, : ] = self.shadow_hand_dof_vel

            # self.
            # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos, ##
            #                                                        self.shadow_hand_dof_lower_limits, ##
            #                                                        self.shadow_hand_dof_upper_limits) # upper limits ##
            # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(canon_shadow_hand_dof, ##
            #                                                     self.shadow_hand_dof_lower_limits, ##
            #                                                     self.shadow_hand_dof_upper_limits)  ##
            
            
            canon_shadow_hand_dof = unscale(canon_shadow_hand_dof,
                                            self.shadow_hand_dof_lower_limits,
                                            self.shadow_hand_dof_upper_limits)
            
            # 
            canon_shadow_hand_dof = canon_shadow_hand_dof.contiguous().view(canon_shadow_hand_dof.size(0), -1).contiguous() 
            self.obs_buf[:, 0:self.num_shadow_hand_dofs * self.history_length] = canon_shadow_hand_dof
            
            
            if self.wo_vel_obs: # previous hand poses and dofs and the preiv
                self.obs_buf[:, self.num_shadow_hand_dofs * self.history_length : 2 * self.num_shadow_hand_dofs * self.history_length] = 0.0
            else:
                self.obs_buf[:,self.num_shadow_hand_dofs * self.history_length :2 * self.num_shadow_hand_dofs * self.history_length ] = self.vel_obs_scale * self.tot_history_hand_dof_vel.contiguous().view(self.tot_history_hand_dof_vel.size(0), -1).contiguous() # get the hand dof velocities #
            
            if self.obs_type == "full_state" or asymm_obs:
                self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
            
                fingertip_obs_start = 3 * self.num_shadow_hand_dofs
            else:
                fingertip_obs_start = 2 * self.num_shadow_hand_dofs * self.history_length
            
            
            if self.use_local_canonical_state:
                
                history_fingertip_pos = self.tot_history_fingertip_state[..., :3]
                canon_fingertip_pos = history_fingertip_pos - self.object_pos.unsqueeze(1).unsqueeze(1)
                canon_fingertip_pose = torch.cat(
                    [ canon_fingertip_pos, self.tot_history_fingertip_state[..., 3:] ], dim=-1
                )
                # dynamics aware planning module training # # training #
                # canon_fingertip_pos = self.fingertip_pos - self.object_pos.unsqueeze(1)
                # canon_fingertip_pose = torch.cat(
                #     [ canon_fingertip_pos, self.fingertip_state[..., 3:] ], dim=-1
                # )
            else:
                canon_fingertip_pose = self.tot_history_fingertip_state
        
            # aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            aux = canon_fingertip_pose.reshape(self.num_envs, num_ft_states * self.history_length)
            aux = aux.contiguous().view(aux.size(0), -1).contiguous()
            # for i in range(self.num_fingertips):
            #     aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
            # 66:131: ft states
            self.obs_buf[:, fingertip_obs_start: fingertip_obs_start + num_ft_states * self.history_length] = aux

            # 131:161: ft sensors: do not need repose
            if self.obs_type == "full_state" or asymm_obs:
            #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.force_sensor_tensor[:, :num_ft_force_torques]
            # else
                self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.vec_sensor_tensor[:, :num_ft_force_torques]

                hand_pose_start = fingertip_obs_start + num_ft_states + num_ft_force_torques #  95
            else:
                hand_pose_start = fingertip_obs_start + num_ft_states * self.history_length
            # 161:167: hand_pose
            ### Global hand pose ###
            
            
            if self.use_local_canonical_state:
                canon_right_hand_pos = self.tot_history_right_hand_pos - self.object_pos.unsqueeze(1)
                # canon_right_hand_pos = self.right_hand_pos - self.object_pos
            else:
                canon_right_hand_pos = self.tot_history_right_hand_pos
            
            canon_right_hand_pos = canon_right_hand_pos.contiguous().view(canon_right_hand_pos.size(0), -1).contiguous()
            history_hand_rot = get_euler_xyz(self.tot_history_right_hand_rot.contiguous().view(self.tot_history_right_hand_rot.size(0) * self.tot_history_right_hand_rot.size(1), 4))
            history_hand_rot_x, history_hand_rot_y, history_hand_rot_z = history_hand_rot[0], history_hand_rot[1], history_hand_rot[2]
            history_hand_rot = torch.stack(
                [history_hand_rot_x, history_hand_rot_y, history_hand_rot_z], dim=-1
            )
            history_hand_rot = history_hand_rot.contiguous().view(self.num_envs, -1, 3)
            history_hand_rot = history_hand_rot.contiguous().view(history_hand_rot.size(0), -1).contiguous()
            
            if self.tight_obs:
                # self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
                self.obs_buf[:, hand_pose_start: hand_pose_start + 3 * self.history_length] = canon_right_hand_pos # self.unpose_point(canon_right_hand_pos)
                # history_ha
                euler_xyz = history_hand_rot #  get_euler_xyz(self.unpose_quat(self.right_hand_rot))
            else:
                # self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
                self.obs_buf[:, hand_pose_start:hand_pose_start + 3 * self.history_length ] = canon_right_hand_pos #  self.unpose_point(canon_right_hand_pos)
                euler_xyz = history_hand_rot #  get_euler_xyz(self.unpose_quat(self.hand_orientations[self.hand_indices, :]))
            
            # self.obs_buf[:, hand_pose_start + 3:hand_pose_start + 4] = euler_xyz[0].unsqueeze(-1)
            # self.obs_buf[:, hand_pose_start + 4:hand_pose_start + 5] = euler_xyz[1].unsqueeze(-1)
            # self.obs_buf[:, hand_pose_start + 5:hand_pose_start + 6] = euler_xyz[2].unsqueeze(-1)
            
            self.obs_buf[:, hand_pose_start + 3 * self.history_length: hand_pose_start + 6 * self.history_length] = euler_xyz
                
            # Actions #
            action_obs_start = hand_pose_start + 6 * self.history_length
            # 167:191: action #
            try:
                # aux = self.actions[:, :self.num_shadow_hand_dofs]
                aux = self.tot_history_right_hand_actions.contiguous().view(self.tot_history_right_hand_actions.size(0), -1).contiguous()
            except: # 
                aux = torch.zeros((self.num_envs, self.num_shadow_hand_dofs * self.history_length), dtype=torch.float32, device=self.device)
            # aux[:, 0:3] = self.unpose_vec(aux[:, 0:3])
            # aux[:, 3:6] = self.unpose_vec(aux[:, 3:6])
            self.obs_buf[:, action_obs_start:action_obs_start + self.num_shadow_hand_dofs * self.history_length] = aux

            # object pos and object pose ? #
            if self.use_local_canonical_state:
                canon_object_pos = self.tot_history_object_pose[..., :3] - self.object_pos.unsqueeze(1)
                # canon_object_pos = self.object_pos - self.object_pos
            else:
                canon_object_pos = self.tot_history_object_pose[..., :3] #  self.object_pos  
            canon_object_pos = canon_object_pos.contiguous().view(canon_object_pos.size(0), -1).contiguous()
            canon_object_ornt = self.tot_history_object_pose[..., 3:].contiguous().view(self.tot_history_object_pose.size(0), -1).contiguous()

            obj_obs_start = action_obs_start + self.num_shadow_hand_dofs  * self.history_length # 144
            # 191:207 object_pose, goal_pos
            # self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
            self.obs_buf[:, obj_obs_start:obj_obs_start + 3 * self.history_length ] = canon_object_pos #  self.unpose_point(canon_object_pos)
            self.obs_buf[:, obj_obs_start + 3 * self.history_length :obj_obs_start + 7 * self.history_length ] =  canon_object_ornt # self.unpose_quat(self.object_pose[:, 3:7])
            
            obj_obs_vel_start = obj_obs_start + 7 * self.history_length
            
            if self.wo_vel_obs:
                self.obs_buf[:, obj_obs_vel_start : obj_obs_vel_start + 3] = 0.0
                self.obs_buf[:, obj_obs_vel_start + 3: obj_obs_vel_start + 6] = 0.0
            else: # object 
                self.obs_buf[:, obj_obs_vel_start : obj_obs_vel_start + 3] = self.unpose_vec(self.object_linvel)
                self.obs_buf[:, obj_obs_vel_start + 3: obj_obs_vel_start + 6] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
                
            # print(f"[Debug] After adding history obs")
        
        
        
        #### Delta object pos #### nex obj pos, while when calculating the reward you should use the object pos in the referende trajs at the current timestep # # also for the hand ref state -- should use the next timestep's predictions as the delta pos; but the reward should be computed using the current step's result ####
        self.obj_obs_start = obj_obs_start
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)
        
        #### Delta object ornt ####
        if self.include_obj_rot_in_obs:
            self.obs_buf[:, obj_obs_start + 16:obj_obs_start + 20] = self.unpose_quat(self.goal_rot)
            hand_goal_start = obj_obs_start + 20
        else:
            hand_goal_start = obj_obs_start + 16
            
        # #### NOTE: version 2 of the randomize conditions ####
        # if self.randomize_conditions:
        #     if self.condition_mask_type == MASK_OBJ:
        #         self.obs_buf[:, obj_obs_start + 13: obj_obs_start + 16] = 0.0
        # #### NOTE: version 2 of the randomize conditions ####
        
        # #### NOTE: version 3 of the randomize conditions ####
        # if self.random_shift_cond:
        #     self.obs_buf[self.env_cond_type == COND_HAND, obj_obs_start + 13: obj_obs_start + 16] = 0.0
        # #### NOTE: version 3 of the randomize conditions ####
        
        # + 6 + nn_dof (action) + 16 (obj) + 7 + nn_dof (goal) + 64
        # 207:236 goal #
        # hand_goal_start = obj_obs_start + 16
        
        self.hand_goal_start = hand_goal_start
        
        # print(f"hand_goal_start: {hand_goal_start}")
        
        if self.tight_obs:
            self.obs_buf[:, hand_goal_start: hand_goal_start +  self.num_shadow_hand_dofs] = self.delta_qpos
        else:
            self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
            self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
            # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos
            self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.delta_qpos


            if self.masked_mimic_training:
                self.mimic_teacher_obs_buf[:, : hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.obs_buf[:, : hand_goal_start + 7+  self.num_shadow_hand_dofs].clone()

            #### NOTE: version 2 of the randomize conditions ####
            if self.randomize_conditions:
                if self.condition_mask_type == MASK_HAND:
                    self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = 0.0
                elif self.condition_mask_type == MASK_HAND_RNDIDX:
                    self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs][:, self.rnd_selected_hand_joints] = 0.0
                    # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs]] = 0.0
            #### NOTE: version 2 of the randomize conditions ####
            
            #### NOTE: version 3 of the randomize conditions ####
            # if self.random_shift_cond:
            self.obs_buf[self.env_cond_type == COND_OBJ, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = 0.0
            self.obs_buf[self.env_cond_type == COND_PARTIALHAND_OBJ, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.obs_buf[self.env_cond_type == COND_PARTIALHAND_OBJ, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] * self.env_cond_hand_masks[self.env_cond_type == COND_PARTIALHAND_OBJ] # nn_cond_envs x nn_hand_dof xxxxxx nn_cond_envs x nn_hand_dof #
            #### NOTE: version 3 of the randomize conditions ####
            
            hand_goal_start = hand_goal_start + 7




        if self.obs_type == 'pure_state_wref_wdelta' and self.use_kinematics_bias_wdelta:
            
            # tot_goal_hand_qs_th = self.tot_kine_qs
            # tot_goal_hand_qs_th = self.tot_hand_preopt_res
            # envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
            # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
            # print(f"[Debug] Start adding residual actions")
            
            nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
            
            if self.only_use_hand_first_frame:
                # first_frame_hand_qpos_ref = 
                tot_envs_hand_qs = self.tot_hand_preopt_res
                # maxx_env_inst_idx = torch.max(self.env_inst_idxes).item()
                # minn_env_inst_idx = torch.min(self.env_inst_idxes).item() # tot envs hand qs #
                # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_env_inst_idx: {maxx_env_inst_idx}, minn_env_inst_idx: {minn_env_inst_idx}")
                tot_envs_hand_qs = batched_index_select(tot_envs_hand_qs, self.env_inst_idxes, dim=0) # nn_envs x nn_envs #
                first_frame_envs_hand_qs = tot_envs_hand_qs[:, 0]
                self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = first_frame_envs_hand_qs
            else:
                if self.use_local_canonical_state: # nex_hand_qpos_ref #
                    canon_hand_qpos_trans = self.nex_hand_qpos_ref[..., :3] - self.object_pos
                    canon_hand_qpos_ref = torch.cat(
                        [ canon_hand_qpos_trans, self.nex_hand_qpos_ref[..., 3:] ], dim=-1
                    )
                else:
                    canon_hand_qpos_ref = self.nex_hand_qpos_ref
            
                # unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
                unscaled_nex_hand_qpos_ref = unscale(canon_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            
                # cur_hand_qpos_ref #  # current delta targets #
                # unscaled_nex_hand_qpos_ref = unscale(cur_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
                
                self.nex_ref_start = nex_ref_start
                
                # unscaled_nex_hand_qpos_ref = cur_hand_qpos_ref
                self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
                
            
            
            # delta poses, # delta targets # #
            
            cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
            self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs]
            
            if self.masked_mimic_training:
                self.mimic_teacher_obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref.clone()
                self.mimic_teacher_obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs].clone()
                
            
            obj_feat_st_idx = cur_delta_start + self.num_shadow_hand_dofs
            
        elif self.obs_type == 'pure_state_wref': # pure stsate with ref 
            nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
            unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
            
            obj_feat_st_idx = nex_ref_start + self.num_shadow_hand_dofs
            
        elif not self.obs_type == 'pure_state':
            
            # 236: visual feature 
            visual_feat_start = hand_goal_start + self.num_shadow_hand_dofs #  29
            
            # 236: 300: visual feature #
            self.obs_buf[:, visual_feat_start: visual_feat_start + 64] = 0.1 * self.visual_feat_buf
            self.obs_buf[:, visual_feat_start + 64: 300] = 0.0
            
            obj_feat_st_idx = 300
        
        
        
        if self.use_future_obs:
            
            if self.masked_mimic_training: # teacher should have the full kinematics observations # -- and it applies both for the nfuture setting and for the wfuture setting -> teacher -- no masked goals #
                # self.mimic_teacher_obs_buf[:, : obj_feat_st_idx] = self.obs_buf[:, : obj_feat_st_idx].clone()
                self.mimic_teacher_obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + full_future_feats.size(1) ] = full_future_feats
            
            
            # future_feats
            self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + future_feats.size(1)] = future_feats # future features
            obj_feat_st_idx = obj_feat_st_idx + future_feats.size(1)
        
        
        
        
        if not self.single_instance_state_based_test and not self.single_instance_state_based_train:
            ### add the obj latent features ###
            ### add the env obj latent features ###
            
            if self.w_obj_latent_features:
                self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_object_latent_feat
                
                if self.masked_mimic_training:
                    self.mimic_teacher_obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_object_latent_feat
            
            if self.use_inst_latent_features: # use the instane latent features 
                
                obj_feat_st_idx = obj_feat_st_idx + self.object_feat_dim
                if self.w_inst_latent_features:
                    self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_inst_latent_feat

                    if self.masked_mimic_training:
                        self.mimic_teacher_obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_inst_latent_feat

            
            
            
            if self.supervised_training:
                # TODO: add expected actions here #
                if self.w_obj_latent_features:
                    nex_hand_qtars_st_idx = obj_feat_st_idx + self.object_feat_dim
                else:
                    nex_hand_qtars_st_idx = obj_feat_st_idx
                env_max_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) - 1
                # nn_envs,
                if self.use_window_future_selection:
                    nex_progress_buf = torch.clamp(self.ws_selected_progress_buf + 1, min=torch.zeros_like(env_max_episode_length), max=env_max_episode_length)
                else:
                    nex_progress_buf = torch.clamp(self.progress_buf + 1, min=torch.zeros_like(env_max_episode_length), max=env_max_episode_length)
                # env_hand_qtars = batched_index_select(self.env_hand_qs, self.env_inst_idxes, dim=0)
                maxx_env_idxes  = torch.max(self.env_inst_idxes).item()
                minn_env_idxes = torch.min(self.env_inst_idxes).item()
                # print(f"maxx_env_idxes: {maxx_env_idxes}, minn_env_idxes: {minn_env_idxes}, tot_hand_qtars: {self.tot_hand_qtars.size()}, tot_kine_qs: {self.tot_kine_qs.size()}")
                env_hand_qtars = batched_index_select(self.tot_hand_qtars, self.env_inst_idxes, dim=0)
                nex_env_hand_qtars = batched_index_select(env_hand_qtars, nex_progress_buf.unsqueeze(1), dim=1)
                nex_env_hand_qtars = nex_env_hand_qtars.squeeze(1)
                
                
                tot_envs_hand_qs = batched_index_select(self.tot_kine_qs, self.env_inst_idxes, dim=0) # nn_envs x 
                # envs_maxx_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
                increased_progress_buf = nex_progress_buf
                ctl_kinematics_bias = batched_index_select(tot_envs_hand_qs, increased_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs # 
                ctl_kinematics_bias = ctl_kinematics_bias.squeeze(1)
                
                
                nex_delta_actions = nex_env_hand_qtars - ctl_kinematics_bias
                # print(f"nex_delta_actions: {nex_delta_actions.size()}, ")
                # print(f"cur_delta_targets: {self.cur_delta_targets.size()}, self.actuated_dof_indices: {self.actuated_dof_indices}")
                nex_delta_delta_actions = nex_delta_actions - self.cur_delta_targets[:, self.actuated_dof_indices]
                # print(f"nex_delta_delta_actions: {nex_delta_delta_actions.size()}, shadow_hand_dof_speed_scale_tsr: {self.shadow_hand_dof_speed_scale_tsr.size()}")
                # shadow hand dof speed sacle tsr #
                nex_actions = (nex_delta_delta_actions / self.dt) / self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0)
                
                
                if self.tot_hand_actions is not None:
                    env_hand_actions = batched_index_select(self.tot_hand_actions, self.env_inst_idxes, dim=0)
                    nex_env_hand_actions = batched_index_select(env_hand_actions, nex_progress_buf.unsqueeze(1), dim=1)
                    nex_env_hand_actions = nex_env_hand_actions.squeeze(1)
                    nex_actions = nex_env_hand_actions
                
                # # prev_detlat_targets # 
                # delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
                # cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
                # self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
                # self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
                
                self.obs_buf[:, nex_hand_qtars_st_idx: nex_hand_qtars_st_idx + self.num_actions] = nex_actions 
                
                
                
                if self.use_multiple_teacher_model:
                    
                    ######## multiple teacher supervision strategy 1 ###########
                    cur_env_succ_index = (self.env_teacher_idx_list > -0.5).int() + (self.env_teacher_idx_list == self.teacher_model_idx).int()
                    cur_env_succ_index = (cur_env_succ_index == 2).int() # 
                    cur_env_succ_encoded_value = cur_env_succ_index * self.nn_teachers + self.teacher_model_idx
                    # print(f"teacher_model_idx: {self.teacher_model_idx}, nn_teachers: {self.nn_teachers}, cur_env_succ_index: {cur_env_succ_encoded_value[: 10]}, cur_env_succ_index: {cur_env_succ_index.float().mean()}, env_teacher_idx_list: {self.env_teacher_idx_list.float().mean()}")
                    ######## multiple teacher supervision strategy 1 ###########
                    
                    ######## multiple teacher supervision strategy 2 ###########
                    cur_env_succ_encoded_value = self.env_teacher_idx_list
                    ######## multiple teacher supervision strategy 1 ###########
                    
                    
                    self.obs_buf[:, nex_hand_qtars_st_idx + self.num_actions: nex_hand_qtars_st_idx + self.num_actions + 1] = cur_env_succ_encoded_value.unsqueeze(1) # self.env_rew_succ_list.unsqueeze(1)
                else:
                    # if self.grab_obj_type_to_opt_res is not None: # self.grab obj type to opt res # # to opt res # to opt res #
                    # print(f"{sum(self.env_rew_succ_list)} / {len(self.env_rew_succ_list)}") # to opt res # # env ecoded value #
                    self.obs_buf[:, nex_hand_qtars_st_idx + self.num_actions: nex_hand_qtars_st_idx + self.num_actions + 1] = self.env_rew_succ_list.unsqueeze(1)
                
                # unscale(nex_env_hand_tars, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
                pass
        
        
        return

    def reset_target_pose(self, env_ids, apply_reset=False):
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), 4), device=self.device)

        new_rot = randomize_rotation(rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids])

        rand_length = torch_rand_float(0.3, 0.5, (len(env_ids), 1), device=self.device)
        rand_angle = torch_rand_float(-1.57, 1.57, (len(env_ids), 1), device=self.device)
        self.goal_states[env_ids, 0:3] = self.goal_init_state[env_ids, 0:3]

        # self.goal_states[env_ids, 3:7] = new_rot
        self.root_state_tensor[self.goal_object_indices[env_ids], 0:3] = self.goal_states[env_ids, 0:3]  # + self.goal_displacement_tensor # 
        self.root_state_tensor[self.goal_object_indices[env_ids], 3:7] = self.goal_states[env_ids, 3:7]

        self.root_state_tensor[self.goal_object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.goal_object_indices[env_ids], 7:13])

        if apply_reset:
            goal_object_indices = self.goal_object_indices[env_ids].to(torch.int32)
            self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_state_tensor), gymtorch.unwrap_tensor(goal_object_indices), len(env_ids))
        self.reset_goal_buf[env_ids] = 0



    # with the task and the hierarchical task description and the regenerated goal states for hand and object # 
    def random_shift_cond_type(self, env_ids=None):
        
        # ##### Prepare for the full trajectory information arrays #####
        # self.ori_tot_hand_preopt_res = self.tot_hand_preopt_res.clone()
        # self.ori_tot_hand_qs = self.tot_hand_qs.clone()
        # self.ori_tot_hand_qtars = self.tot_hand_qtars.clone()
        # self.ori_tot_kine_qs = self.tot_kine_qs.clone()
        # self.ori_tot_kine_obj_trans = self.tot_kine_obj_trans.clone()
        # self.ori_tot_kine_obj_ornt =self.tot_kine_obj_ornt.clone()
        # self.ori_tot_kine_obj_rot_euler = self.tot_kine_obj_rot_euler.clone()
        if env_ids is None : 
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        nn_reset_envs = env_ids.size(0)
        
        if self.random_shift_cond:
            
            
            ## TODO: for each cond type, we should calibrate the goals input to the observations # 
            ## TODO: for each cond type, we should change how does the reward is computed #
            ## TODO: if the cond type does not include the hand, we should set the base 
            ## for object cond, we should only track the object; -- 1) should mask out hand goal states related terms in the observation, should not compute the hand pose related rewards in the reward function, 2) should set the base  trajectory -- the pre_opt_res trajectory to the first frame # 
            # envs_rnd_type = torch.randint(0, 2, (nn_reset_envs, ), device=self.device, dtype=torch.long)
            envs_rnd_type = torch.randint(0, 3, (nn_reset_envs, ), device=self.device, dtype=torch.long)
            self.env_cond_type[env_ids] = envs_rnd_type
            
            self.env_cond_hand_masks[env_ids, :] = 1.0
            random_mask_hand_ratio = np.random.uniform(0.0, float(19/22), (1,))[0].item() 
            random_mask_nn_hand_joint = int(random_mask_hand_ratio * self.nn_hand_dof)
            if random_mask_nn_hand_joint > 0:
                random_mask_hand_joint_idxes = np.random.permutation(self.nn_hand_dof )[:random_mask_nn_hand_joint]
                random_mask_hand_joint_idxes = torch.from_numpy(random_mask_hand_joint_idxes).to(self.rl_device).long()
                # print(f"env_cond_hand_masks: {self.env_cond_hand_masks.size()}, env_ids: {env_ids.size()}, random_mask_hand_joint_idxes: {random_mask_hand_joint_idxes.size()}")
                self.env_cond_hand_masks[env_ids][:, random_mask_hand_joint_idxes] = self.env_cond_hand_masks[env_ids][:, random_mask_hand_joint_idxes] * 0.0
            
        
        ### TODO: after we have set the env_cond_type, we should then use them in the preopt_res selection, observation computation and the reward computation ###
        
        if self.random_shift_cond_freq :
            # env  # env inv cond freq #
            envs_rnd_inv_cond_freq = torch.randint(1, self.maxx_inv_cond_freq + 1, (nn_reset_envs, ), device=self.device, dtype=torch.long) # get the inv cond freq
            self.env_inv_cond_freq[env_ids] = envs_rnd_inv_cond_freq # 
        
    
    def schedule_ornt_reward_coef(self, env_ids=None):
        env_ids_list = env_ids.detach().cpu().numpy().tolist()
        env_ids_list = [int(cur_idx) for cur_idx in env_ids_list]
        target_env_idx = 0
        if target_env_idx in env_ids_list:
            if self.rew_env_reset_nn <= self.ornt_rew_coef_warm_starting_steps:
                self.cur_ornt_rew_coef = self.lowest_ornt_rew_coef
            elif self.rew_env_reset_nn <= self.ornt_rew_coef_warm_starting_steps + self.ornt_rew_coef_increasing_steps:
                self.cur_ornt_rew_coef = self.lowest_ornt_rew_coef + (self.highest_ornt_rew_coef - self.lowest_ornt_rew_coef) * (self.rew_env_reset_nn - self.ornt_rew_coef_warm_starting_steps) / (self.ornt_rew_coef_increasing_steps + self.ornt_rew_coef_warm_starting_steps)
            else:
                self.cur_ornt_rew_coef = self.highest_ornt_rew_coef
            self.rew_env_reset_nn += 1
            print(f"[INFO] rew_env_reset_nn: {self.rew_env_reset_nn}, ornt_rew_coef: {self.cur_ornt_rew_coef}")
            # pass

    def reset(self, env_ids=None, goal_env_ids=None): 
        
        
        
        
        if self.use_multiple_teacher_model:
            self.tot_reset_nn += 1
            if self.tot_reset_nn % self.change_teacher_freq == 0:
                self.teacher_model_idx = (self.teacher_model_idx + 1) % self.nn_teachers
        
        
        
        
        
        self.random_shift_cond_type(env_ids)
        
        # maxx_progress_buf = torch.max(self.progress_buf)
        # minn_progress_buf = torch.min(self.progress_buf)
        # print(f"maxx_progress_buf: {maxx_progress_buf}, minn_progress_buf: {minn_progress_buf}")
        
        # self.ref_ts = 0
        
        if env_ids is None : 
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
            
        
        if self.schedule_ornt_rew_coef:
            self.schedule_ornt_reward_coef(env_ids)
        
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses #
        self.reset_target_pose(env_ids) # 
        
        
        

        ### NOTE: we have disabled the random_prior setting 
        # if self.random_prior:
        #     for env_id in env_ids:
        #         i = env_id.item()
        #         object_code = self.object_code_list[self.object_id_buf[i]]
        #         scale = self.object_scale_buf[i]

        #         data = self.grasp_data[object_code][scale] # data for one object one scale # reset target pose #
        #         buf = data['object_euler_xy']
        #         prior_idx = random.randint(0, len(buf) - 1)
        #         # prior_idx = 0 ## use only one data

        #         self.target_qpos[i:i+1] = data['target_qpos'][prior_idx]
        #         self.target_hand_pos[i:i + 1] = data['target_hand_pos'][prior_idx]
        #         self.target_hand_rot[i:i + 1] = data['target_hand_rot'][prior_idx]
        #         self.object_init_euler_xy[i:i + 1] = data['object_euler_xy'][prior_idx]
        #         self.object_init_z[i:i + 1] = data['object_init_z'][prior_idx]
        ### NOTE: we have disabled the random_prior setting 

        # # reset shadow hand
        # delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        # delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        # rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        # pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        # 
        self.shadow_hand_dof_pos[env_ids, :] = self.shadow_hand_default_dof_pos[env_ids, :] # env_ids #

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]
        # 
        # self.dof_state[:, : ] #
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = self.shadow_hand_default_dof_pos[env_ids, : self.num_shadow_hand_dofs]
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = self.shadow_hand_default_dof_pos[env_ids, : self.num_shadow_hand_dofs]
        
        self.prev_delta_targets[env_ids, :] = 0
        self.cur_delta_targets[env_ids, :] = 0
        
        # cur_delta_targets_impedance, prev_delta_targets_impedance
        if self.w_impedance_bias_control:
            # clear the prev and the current target impedance #
            self.cur_delta_targets_impedance[env_ids, :] = 0
            self.prev_delta_targets_impedance[env_ids, :] = 0 # 
            # cur d

        hand_indices = self.hand_indices[env_ids].to(torch.int32) # hand indices #
        all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        self.gym.set_dof_position_target_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

        # all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], self.table_indices[env_ids], ]).to(torch.int32))  ##
        
        all_indices = torch.unique(torch.cat([all_hand_indices, self.object_indices[env_ids], ]).to(torch.int32))  ##

        # set hand positions; aet hand orientations #
        self.hand_positions[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 0:3]
        self.hand_orientations[all_indices.to(torch.long), :] = self.saved_root_tensor[all_indices.to(torch.long), 3:7]


        ## NOTE: we disable the object random rotations here ##
        # theta = torch_rand_float(-3.14, 3.14, (len(env_ids), 1), device=self.device)[:, 0]
        # #reset obejct with all data:
        # new_object_rot = quat_from_euler_xyz(self.object_init_euler_xy[env_ids,0], self.object_init_euler_xy[env_ids,1], theta)
        # prior_rot_z = get_euler_xyz(quat_mul(new_object_rot, self.target_hand_rot[env_ids]))[2]

        # # coordinate transform according to theta(object)/ prior_rot_z(hand)
        # self.z_theta[env_ids] = prior_rot_z
        # prior_rot_quat = quat_from_euler_xyz(torch.tensor(1.57, device=self.device).repeat(len(env_ids), 1)[:, 0], torch.zeros_like(theta), prior_rot_z)


        ## NOTE: we disable the object random rotations here ##
        # self.hand_orientations[hand_indices.to(torch.long), :] = prior_rot_quat
        self.hand_linvels[hand_indices.to(torch.long), :] = 0
        self.hand_angvels[hand_indices.to(torch.long), :] = 0


        # 
        self.root_state_tensor[self.object_indices[env_ids]] = self.object_init_state[env_ids].clone()
        ## NOTE: we disable the object random rotations here #
        # self.root_state_tensor[self.object_indices[env_ids], 3:7] = new_object_rot
        self.root_state_tensor[self.object_indices[env_ids], 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices[env_ids], 7:13])
        # 
        # 
        # all_indices = torch.unique(torch.cat([all_hand_indices,
        #                                       self.object_indices[env_ids],
        #                                       self.goal_object_indices[env_ids],
        #                                       self.table_indices[env_ids], ]).to(torch.int32))
        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids], ]).to(torch.int32))
        # state tensor indexed #
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        
        
        """ V1 forecasting -- data driven forecaster """
        if self.use_forcasting_model and (not self.already_forcasted) and self.random_time:
            # if self.use_forcasting_model:
            self.progress_buf[env_ids] = 0
            # first_env_progress_buf = self.progress_buf[0].item()
            # if (not self.already_forcasted) or  (first_env_progress_buf % self.forcasting_inv_freq) == 0:
            self._forward_forcasting_model()
            # self.try_save_network_forwarding_info_dict()
        """ V1 forecasting -- data driven forecaster """
        

        if self.random_time:
            self.random_time = False
            self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
            if self.use_twostage_rew:
                # self.cur_grasp_fr[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
                self.grasping_progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        else:
            self.progress_buf[env_ids] = 0
            if self.use_twostage_rew:
                self.grasping_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        
        """ NOTE: not in use currently """
        if self.use_twostage_rew:
            # grasping_progress_buf, grasp_manip_stages, grasping_succ_buf
            self.grasp_manip_stages[env_ids] = 0
            self.grasping_succ_buf[env_ids] = 0
        
        if self.lifting_separate_stages:
            self.reach_lifting_stage[env_ids] = 0
            # self.lift_fr[env_ids] = 0
            # self.reach_lifting_stage[goal_env_ids] = 0
            # self.lift_fr[goal_env_ids] = 0
        """ NOTE: not in use currently """
        
        
        ##### Reset the contact buffer and the distance buffer to zeros #####
        # hand_palm_fingers_obj_contact_buf, right_hand_dist_buf
        self.hand_palm_fingers_obj_contact_buf[env_ids] = 0
        self.right_hand_dist_buf[env_ids] = 0
        
        self.prev_dof_vel[env_ids] = 0
        self.cur_dof_vel[env_ids] = 0
        
        # after that, we can set states #
        # if self.cfg.test and self.reset_nn % 1000 == 0:
        # if self.reset_nn % 1000 == 0:
        #     print(f"reset_nn: {self.reset_nn}")
        #     logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
        #     logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
        #     np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states) # save the ts_to_hand_obj_states #
        #     print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")
        
        self.reset_nn += 1
        
        if self.use_forcasting_model:
            self.try_save_forcasting_model()
        
        self.compute_observations()
        
        
        envs_text_features = batched_index_select(self.tot_text_features, self.env_inst_idxes, dim=0)
        # self.forecasting_obs = torch.cat(
        #     [
        #         self.shadow_hand_dof_pos, self.object_pos, self.object_rot, self.env_object_latent_feat, envs_text_features
        #     ], dim=-1
        # )
        
        # find [progress_buf + 1, progress_buf + 2] -- the kine shadow hand dofs, kine obj pos, and kine obj rot #
        
        """ Get GT predicted res """
        envs_goal_obj_trans_th = batched_index_select(self.tot_kine_obj_trans, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(self.tot_kine_obj_ornt, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        envs_goal_hand_qs_th = batched_index_select(self.tot_hand_preopt_res, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x nn_hand_dofs
        
        
        tot_gt_target_hand_qs = []
        tot_gt_target_obj_pos  = []
        tot_gt_target_obj_rot = []
        
        
        
        
        # pred_tracking_targets_obj_pos, pred_tracking_targets_obj_rot, pred_tracking_targets_hand_qs
        envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        for i_ws in range(self.forward_forecasting_nn_steps):
            # forward forecasting nn steps # 
            cur_progress_buf = self.progress_buf + i_ws
            cur_progress_buf = torch.clamp(cur_progress_buf, min=torch.zeros_like(cur_progress_buf), max=envs_episode_length) # 
            
            gt_targets_hand_qs = batched_index_select(envs_goal_hand_qs_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x nn_hand_qs_dim #
            gt_targets_obj_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 3
            gt_targets_obj_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 4
            
            tot_gt_target_hand_qs.append(gt_targets_hand_qs)
            tot_gt_target_obj_pos.append(gt_targets_obj_pos)
            tot_gt_target_obj_rot.append(gt_targets_obj_rot)
        tot_gt_target_hand_qs = torch.stack(tot_gt_target_hand_qs, dim=1)
        tot_gt_target_obj_pos = torch.stack(tot_gt_target_obj_pos, dim=1)
        tot_gt_target_obj_rot = torch.stack(tot_gt_target_obj_rot, dim=1)
        
        nex_tracking_targets = torch.cat(
            [ tot_gt_target_hand_qs, tot_gt_target_obj_pos, tot_gt_target_obj_rot], dim=-1 # nn_envs x 
        )[:, 1] # next tracking targets # # tot_nn_envs x 
        
        # not using the GT to supervise total actions but at least using them to sup the mean of the model's predictions? #
        nex_tracking_targets_forecast_obs = nex_tracking_targets.clone()
        
        tot_gt_target_hand_qs = tot_gt_target_hand_qs - self.shadow_hand_dof_pos.unsqueeze(1)
        
        tot_gt_target =  torch.cat(
            [tot_gt_target_hand_qs, tot_gt_target_obj_pos, tot_gt_target_obj_rot], dim=-1
        )
        
        
        
        tot_gt_target_flatten = tot_gt_target.view(tot_gt_target.size(0), -1)
        
        
        if self.already_predicted_targets: # 
            envs_goal_obj_trans_th = self.pred_tracking_targets_obj_pos
            envs_goal_obj_ornt_th = self.pred_tracking_targets_obj_rot
            envs_goal_hand_qs_th = self.pred_tracking_targets_hand_qs
            
            tot_gt_target_hand_qs = []
            tot_gt_target_obj_pos  = []
            tot_gt_target_obj_rot = []
            
            # pred_tracking_targets_obj_pos, pred_tracking_targets_obj_rot, pred_tracking_targets_hand_qs
            envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
            for i_ws in range(self.forward_forecasting_nn_steps):
                # forward forecasting nn steps # 
                cur_progress_buf = self.progress_buf + i_ws
                cur_progress_buf = torch.clamp(cur_progress_buf, min=torch.zeros_like(cur_progress_buf), max=envs_episode_length) # 
                
                gt_targets_hand_qs = batched_index_select(envs_goal_hand_qs_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x nn_hand_qs_dim #
                gt_targets_obj_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 3
                gt_targets_obj_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 4
                
                tot_gt_target_hand_qs.append(gt_targets_hand_qs)
                tot_gt_target_obj_pos.append(gt_targets_obj_pos)
                tot_gt_target_obj_rot.append(gt_targets_obj_rot)
            tot_gt_target_hand_qs = torch.stack(tot_gt_target_hand_qs, dim=1)
            tot_gt_target_obj_pos = torch.stack(tot_gt_target_obj_pos, dim=1)
            tot_gt_target_obj_rot = torch.stack(tot_gt_target_obj_rot, dim=1)
            
            nex_tracking_targets = torch.cat(
                [ tot_gt_target_hand_qs, tot_gt_target_obj_pos, tot_gt_target_obj_rot], dim=-1 # nn_envs x 
            )[:, 1] 
        
        
        # nex tracking targets forecast obs #
        if self.tuning_single_instance:
            # self.forecasting_obs = torch.cat(
            #     [
            #         self.shadow_hand_dof_pos, self.object_pos, self.object_rot, tot_gt_target_flatten
            #         # self.env_object_latent_feat, envs_text_features
            #     ], dim=-1, 
            # )
            self.forecasting_obs = torch.cat(
                [
                    self.shadow_hand_dof_pos, self.object_pos, self.object_rot, nex_tracking_targets_forecast_obs, tot_gt_target_flatten
                    # self.env_object_latent_feat, envs_text_features
                ], dim=-1, # 
            )
        else:
            # self.forecasting_obs = torch.cat( # obs dimensions? #
            #     [   # do we have forecasting obs dimensions? #
            #         self.shadow_hand_dof_pos, self.object_pos, self.object_rot, self.env_object_latent_feat, envs_text_features, tot_gt_target_flatten
            #         # self.env_object_latent_feat, envs_text_features
            #     ], dim=-1, # 
            # )
            self.forecasting_obs = torch.cat( # obs dimensions? # # 22 + 3 + 4 + 22 + (22 + 3 + 4) + 256
                [   # do we have forecasting obs dimensions? #
                 
                    self.shadow_hand_dof_pos, self.object_pos, self.object_rot, nex_tracking_targets_forecast_obs, tot_gt_target_flatten, self.env_object_latent_feat, 
                    # envs_text_features, tot_gt_target_flatten
                    # self.env_object_latent_feat, envs_text_features
                ], dim=-1, # 
            )
            
            if self.forecasting_obs_with_original_obs:
                self.forecasting_obs = torch.cat( # obs dimensions? # # 22 + 3 + 4 + 22 + (22 + 3 + 4) + 256
                    [ 
                        self.shadow_hand_dof_pos, self.object_pos, self.object_rot, nex_tracking_targets_forecast_obs, tot_gt_target_flatten
                    ], dim=-1,
                )
                    
            
        hand_goal_start = self.hand_goal_start
        
        # print(f"[controller] hand_goal_start: {hand_goal_start}")
        
        hand_goal_start_tsr = torch.tensor([hand_goal_start, self.nex_ref_start, self.obj_obs_start], device=self.rl_device).float()
        
        
        # TODO: 1) in prephysics step, nothing needs to be modified; 2) in compute observations, use the env kine traj idxes to select proper kinematic trajs; 3) in compute observations, us the env kine traj idxes to select proper kinematic trajs 
        if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
            self.envs_kine_source_trajs_idxes = torch.randint(
                low=1, high=self.multiple_kine_source_trajs.size(0), size=(self.envs_kine_source_trajs_idxes.size(0), )
            ).to(self.rl_device)
            # print(f"multiple_kine_source_trajs: {self.multiple_kine_source_trajs.size()}")
            # self.envs_kine_source_trajs_idxes[:] = 1
            pass
        
        
        obs_dict = {}
        obs_dict["forecasting_obs"] = torch.clamp(self.forecasting_obs, -self.clip_obs, self.clip_obs).to(self.rl_device)
        obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        
        if self.forecasting_obs_with_original_obs:
            obs_dict['forecasting_obs'] = torch.cat(
                [ obs_dict["forecasting_obs"], obs_dict['obs'] ], dim=-1
            )
        
        obs_dict['nex_tracking_targets'] = nex_tracking_targets.to(self.rl_device) #
        obs_dict['hand_goal_start'] = hand_goal_start_tsr.to(self.rl_device)
        
        return obs_dict
        
        ## forecasting obs ##
        if self.masked_mimic_training:
            mimic_teacher_obs_buf_clamped = torch.clamp(self.mimic_teacher_obs_buf, -self.clip_obs, self.clip_obs)
            obs_dict = {}
            obs_dict["mimic_teacher_obs"] = mimic_teacher_obs_buf_clamped.to(self.rl_device)
            obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
            # hand current state, object state, object features, progress embedding, global features #
            
            return obs_dict
        else:
            return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def try_save(self, ):
        # if self.reset_nn % 1000 == 0:
        if self.reset_nn % 1 == 0:
            # print(f"reset_nn: {self.reset_nn}")
            self.ts_to_hand_obj_states['object_code_list'] = self.object_code_list
            self.ts_to_hand_obj_states['env_object_codes'] = self.env_object_codes
            # logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
            logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{1}.npy"
            logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
            # if self.ref_ts >= 147: #
            if self.ref_ts >= 285: # save the ts_to_hand_obj_states #
                np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states) # save the ts_to_hand_obj_states #
                print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")
            
        if self.reset_nn == 2:
            exit(0)
            
    def try_save_forcasting_model(self, ):
        if self.reset_nn % 100 == 0:
            cur_forcasting_model_weight = self.forcasting_model.state_dict()
            cur_forcasting_model_weight_fn = f"last_forcasting_model_weight.pth"
            cur_forcasting_model_weight_fn = os.path.join(self.exp_logging_dir, cur_forcasting_model_weight_fn)
            torch.save(cur_forcasting_model_weight, cur_forcasting_model_weight_fn)
            print(f"save the forcasting model weight to {cur_forcasting_model_weight_fn}")
            
            
    def try_save_train(self, ):
        # if self.reset_nn % 1000 == 0:
        if self.reset_nn % 20 == 0:
            # print(f"reset_nn: {self.reset_nn}")
            self.ts_to_hand_obj_states['object_code_list'] = self.object_code_list
            self.ts_to_hand_obj_states['env_object_codes'] = self.env_object_codes
            # logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
            logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
            logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
            # if self.ref_ts >= 147:
            if self.ref_ts >= 250:
                np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states)
                print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")
    
    
    
    ## TODO: we'd better can get the re-planed trajectories #
    def try_save_network_forwarding_info_dict(self, ):
        if self.ref_ts % 1000 == 0 or (self.test and self.ref_ts >= 250):
            forwarding_logging_fn = f"network_forwarding_info_dict_{self.ref_ts}.npy"
            forwarding_logging_fn = os.path.join(self.exp_logging_dir, forwarding_logging_fn)
            np.save(forwarding_logging_fn, self.forcasting_model_forwarding_info_dict)
            # 
            print(f"Network forwarding info dict saved to {forwarding_logging_fn}")
    
    # 
    
    
    # Apply randomizations only on resets, due to current PhysX limitations #
    def apply_randomizations(self, dr_params):
        # print(f"[Debug] Start randomization")
        rand_freq = dr_params.get("frequency", 1)

        # First, determine what to randomize: 
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything 
        self.last_step = self.gym.get_frame_count(self.sim) 
        if self.first_randomization: # first randomization
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else: # noenv 
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            # randomize_buf #
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf) # reset buf; # apply randomization # reset buf # rand envs #
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist() # rand envs #
            self.randomize_buf[rand_envs] = 0 # set randomize_buf of the rand_envs to zero #

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        # param setters map #
        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        ### TODO: check buckets ###
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        ### NOTE: in the first stage, we do not consider the nonphysical parameter randomization here ###
        # for nonphysical_param in ["observations", "actions"]:
        #     if nonphysical_param in dr_params and do_nonenv_randomize: # do noenv randomize #
        #         dist = dr_params[nonphysical_param]["distribution"]
        #         op_type = dr_params[nonphysical_param]["operation"]
        #         sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
        #         sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
        #         op = operator.add if op_type == 'additive' else operator.mul

        #         if sched_type == 'linear':
        #             sched_scaling = 1.0 / sched_step * \
        #                 min(self.last_step, sched_step)
        #         elif sched_type == 'constant': # constrant #
        #             sched_scaling = 0 if self.last_step < sched_step else 1
        #         else:
        #             sched_scaling = 1

        #         if dist == 'gaussian':
        #             mu, var = dr_params[nonphysical_param]["range"]
        #             mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

        #             if op_type == 'additive':
        #                 mu *= sched_scaling
        #                 var *= sched_scaling
        #                 mu_corr *= sched_scaling
        #                 var_corr *= sched_scaling
        #             elif op_type == 'scaling':
        #                 var = var * sched_scaling  # scale up var over time
        #                 mu = mu * sched_scaling + 1.0 * \
        #                     (1.0 - sched_scaling)  # linearly interpolate

        #                 var_corr = var_corr * sched_scaling  # scale up var over time
        #                 mu_corr = mu_corr * sched_scaling + 1.0 * \
        #                     (1.0 - sched_scaling)  # linearly interpolate

        #             def noise_lambda(tensor, param_name=nonphysical_param):
        #                 params = self.dr_randomizations[param_name]
        #                 corr = params.get('corr', None)
        #                 if corr is None:
        #                     corr = torch.randn_like(tensor)
        #                     params['corr'] = corr
        #                 corr = corr * params['var_corr'] + params['mu_corr']
        #                 return op(
        #                     tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

        #             self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

        #         elif dist == 'uniform':
        #             lo, hi = dr_params[nonphysical_param]["range"]
        #             lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

        #             if op_type == 'additive':
        #                 lo *= sched_scaling
        #                 hi *= sched_scaling
        #                 lo_corr *= sched_scaling
        #                 hi_corr *= sched_scaling
        #             elif op_type == 'scaling':
        #                 lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
        #                 hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
        #                 lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
        #                 hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

        #             def noise_lambda(tensor, param_name=nonphysical_param):
        #                 params = self.dr_randomizations[param_name]
        #                 corr = params.get('corr', None)
        #                 if corr is None:
        #                     corr = torch.randn_like(tensor)
        #                     params['corr'] = corr
        #                 corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
        #                 return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

        #             self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        # ##### in the first stage, we do not consider the simulation parameters randomization here ##### #

        ### NOTE: in the first stage, we do not consider the simulation parameters randomization here ###
        # if "sim_params" in dr_params and do_nonenv_randomize:
        #     prop_attrs = dr_params["sim_params"]
        #     prop = self.gym.get_sim_params(self.sim)

        #     if self.first_randomization: # first randomization # sim parameters #
        #         self.original_props["sim_params"] = {
        #             attr: getattr(prop, attr) for attr in dir(prop)}

        #     for attr, attr_randomization_params in prop_attrs.items():
        #         apply_random_samples( # prop 
        #             prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

        #     self.gym.set_sim_params(self.sim, prop)
        ### NOTE: in the first stage, we do not consider the simulation parameters randomization here ###
        

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids: # self.actor_params_generator #
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        for actor, actor_properties in dr_params["actor_params"].items():
            #### designed for randomize the object properties ####
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color': # get the actor handle and the actor properties #
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale': # attribute randomization params #
                        attr_randomization_params = prop_attrs
                        sample = generate_random_samples(attr_randomization_params, 1,
                                                         self.last_step, None)
                        og_scale = 1
                        if attr_randomization_params['operation'] == 'scaling':
                            new_scale = og_scale * sample
                        elif attr_randomization_params['operation'] == 'additive':
                            new_scale = og_scale + sample
                        self.gym.set_actor_scale(env, handle, new_scale) # set the actor scale --- scale of the actors -- # scale of the actors #
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items(): # random positioning or random rotationing #
                                # random positioning or random rotationing # # offline or online trajectory generations -- offline traj generation # 
                                # at every time , we sample for a random linear and angular velocity and apply the sampled velocity to the object position and rotation #
                                # only apply the position velocity to the hand trajectory, while leaving the hand rotations not changed #
                                # only apply the translational velocity to the hand trajectory, while leaving the hand rotations not changed #
                                # select a frame and apply these changes # # hand fingers and hand rotations should not be included in the reward term # 
                                # we want to let some behaviours automatically emerged # # these tracking rewards are for theose tracking where the hand information is not correct enough # # and wnat some interesting behaviours automatically emerged # # automatically emerged # #
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], p, attr)
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params, # 
                                    self.last_step, smpl)
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            smpl = None
                            if self.actor_params_generator is not None:
                                smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                    extern_sample, extern_offsets[env_id], prop, attr)
                            apply_random_samples(
                                prop, self.original_props[prop_name], attr,
                                attr_randomization_params, self.last_step, smpl)

                    # setter # params setters map #
                    setter = param_setters_map[prop_name]
                    default_args = param_setter_defaults_map[prop_name]
                    setter(env, handle, prop, *default_args)

        if self.actor_params_generator is not None:
            for env_id in env_ids:  # check that we used all dims in sample
                if extern_offsets[env_id] > 0:
                    extern_sample = self.extern_actor_params[env_id]
                    if extern_offsets[env_id] != extern_sample.shape[0]:
                        print('env_id', env_id,
                              'extern_offset', extern_offsets[env_id],
                              'vs extern_sample.shape', extern_sample.shape)
                        raise Exception("Invalid extern_sample size")

        self.first_randomization = False



    def pre_physics_step(self, actions):
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API 
        if len(goal_env_ids) > 0 and len(env_ids) == 0:
            self.reset_target_pose(goal_env_ids, apply_reset=True)
        # if goals need reset in addition to other envs, call set API in reset()
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)


        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        
        
        
        # maxx_progress_buf = torch.max(self.progress_buf).item()
        # minn_progress_buf = torch.min(self.progress_buf).item()
        # print(f"maxx_progress_buf: {maxx_progress_buf}, minn_progress_buf: {minn_progress_buf}") # minn progress buf #
        
        """ Hand actual actions """
        # self.actions = actions.clone().to(self.device)
        self.actions = actions.clone().to(self.device)[..., : self.nn_hand_dof]
        
        
        """ Predicted targets """
        # 
        # self.forecasting_actions = actions.clone().to(self.device)[..., self.nn_hand_dof : ]
        # self.pred_targets_hand_qs = actions.clone().to(self.device)[..., self.nn_hand_dof : self.nn_hand_dof * 2]
        # self.pred_targets_obj_pos = actions.clone().to(self.device)[..., self.nn_hand_dof * 2 : self.nn_hand_dof * 2 + 3]
        # self.pred_targets_obj_rot = actions.clone().to(self.device)[..., self.nn_hand_dof * 2 + 3 : self.nn_hand_dof * 2 + 7]
        # self.pred_targets_hand_qs = scale(self.pred_targets_hand_qs, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        # self.pred_targets_obj_rot = self.pred_targets_obj_rot / torch.clamp(torch.norm(self.pred_targets_obj_rot, dim=-1, keepdim=True), min=1e-6)
        
        
        """ Get predicted targets """ 
        # self.already_predicted_targets = False # # self already forecasted # # state, action -> state # state, action -> state #
        # given the current state and the predicted action, 
        # [current state, inferenced action] -> (world model) -> next state --> it should be close to the expected state to track #
        if self.reset_nn >= 10000:
            self.already_predicted_targets = True
        else:
            self.already_predicted_targets = True
            # self.already_predicted_targets = False
        
        if not self.activate_forecaster:
            self.already_predicted_targets = False
        
        # should be similar to the original hand qs and also the hand qs #
        self.forecasting_actions = actions.clone().to(self.device)[..., self.nn_hand_dof: ]
        per_ts_forecasting_action_dim = self.nn_hand_dof + 3 + 4
        forecasting_actions = self.forecasting_actions.view(self.num_envs, -1, per_ts_forecasting_action_dim)
        pred_targets_hand_qs = forecasting_actions[..., : self.nn_hand_dof]
        
        
        ### Modification 1: delta dtargets; not use obj targets ####
        pred_targets_hand_qs = pred_targets_hand_qs * self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0).unsqueeze(0) * self.dt 
        pred_targets_hand_qs = torch.cumsum(pred_targets_hand_qs, dim=1) ### nn_envs x nn_forecasting_steps x nn_dofs ###
        pred_targets_hand_qs = pred_targets_hand_qs * 1.0 + self.shadow_hand_dof_pos.unsqueeze(1)
        
        
        pred_targets_obj_pos = forecasting_actions[..., self.nn_hand_dof: self.nn_hand_dof + 3]
        pred_targets_obj_rot = forecasting_actions[..., self.nn_hand_dof + 3 :  self.nn_hand_dof + 7 ]
        # pred_targets_hand_qs = scale(pred_targets_hand_qs, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        pred_targets_obj_rot = pred_targets_obj_rot / torch.clamp(torch.norm(pred_targets_obj_rot, dim=-1, keepdim=True), min=1e-6)
        
        
        
        ####### ====== From predicted obj pose to compute the real forecasted obj pose (next ts targets) ====== #######
        ### nn_envs x nn_forecast_steps x 3 ###
        pred_targets_obj_pos = pred_targets_obj_pos * self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0).unsqueeze(0)[..., :3] * self.dt
        pred_targets_obj_pos = pred_targets_obj_pos * 1.0 + self.object_pos.unsqueeze(1)
        
        expanded_object_rot = self.object_rot.unsqueeze(1).contiguous().repeat(1, pred_targets_obj_rot.size(1), 1).contiguous()
        pred_targets_obj_rot = quat_mul(pred_targets_obj_rot, expanded_object_rot) # 
        ####### ====== From predicted obj pose to compute the real forecasted obj pose (next ts targets) ====== #######
        
        
        
        forecast_idx = 0
        self.cur_pred_targets_hand_qs = pred_targets_hand_qs[:, forecast_idx]
        self.cur_pred_targets_obj_pos = pred_targets_obj_pos[:, forecast_idx]
        self.cur_pred_targets_obj_rot = pred_targets_obj_rot[:, forecast_idx]
        
        """ Assign predicted results to buffers """
        envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        for i_ws in range(pred_targets_hand_qs.size(1)):
            # forcast_obj_pos: nn_envs x nn_forcasting_ts x obj_pos_dim #
            cur_forcast_assign_idx = self.progress_buf + i_ws + 1
            # forcast assign idx #
            cur_forcast_assign_idx = torch.clamp(cur_forcast_assign_idx, min=torch.zeros_like(cur_forcast_assign_idx), max=envs_episode_length)
            # objpos: nn_envs x nn_length x 3
            aranged_forcasting_idxes = torch.arange(self.pred_tracking_targets_hand_qs.size(1), device=self.rl_device).unsqueeze(0).repeat(self.pred_tracking_targets_hand_qs.size(0), 1)
            # aranged forcasting idxes #
            assign_mask = (aranged_forcasting_idxes == cur_forcast_assign_idx.unsqueeze(-1))
            # pred_tracking_targets_obj_pos, pred_tracking_targets_obj_rot, pred_tracking_targets_hand_qs
            self.pred_tracking_targets_obj_pos = torch.where(
                assign_mask.unsqueeze(-1).repeat(1, 1, 3), pred_targets_obj_pos[:, i_ws].unsqueeze(1).repeat(1, self.pred_tracking_targets_obj_pos.size(1), 1), self.pred_tracking_targets_obj_pos
            )
            self.pred_tracking_targets_obj_rot = torch.where(
                assign_mask.unsqueeze(-1).repeat(1, 1, 4), pred_targets_obj_rot[:, i_ws].unsqueeze(1).repeat(1, self.pred_tracking_targets_obj_rot.size(1), 1), self.pred_tracking_targets_obj_rot
            )
            self.pred_tracking_targets_hand_qs = torch.where(
                assign_mask.unsqueeze(-1).repeat(1, 1, self.pred_tracking_targets_hand_qs.size(-1)), pred_targets_hand_qs[:, i_ws].unsqueeze(1).repeat(1, self.pred_tracking_targets_hand_qs.size(1), 1), self.pred_tracking_targets_hand_qs
            )
        
        
        
        
        if self.use_kinematics_bias_wdelta: 
            # print(f"self.use_kinematics_bias_wdelta: {self.use_kinematics_bias_wdelta}")
            
            if self.use_window_future_selection:
                # increased_progress_buf = self.ws_selected_progress_buf + 1
                increased_progress_buf = self.progress_buf + 1
            else:
                increased_progress_buf = self.progress_buf + 1
            
            # two instances? # # -10 # cannot understand why it cannot become comparable to that one
            # increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1)
            # get the kinematicsof the increaesd progres buf as the kinematics bias
            # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
            # ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]

            # tot_envs_hand_qs = self.tot_kine_qs
            
            
            tot_envs_hand_qs = self.tot_hand_preopt_res
            
            # if self.already_predicted_targets:
            #     tot_envs_hand_qs = self.pred_tracking_targets_hand_qs
            
            # maxx_env_inst_idx = torch.max(self.env_inst_idxes).item()
            # minn_env_inst_idx = torch.min(self.env_inst_idxes).item()
            # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_env_inst_idx: {maxx_env_inst_idx}, minn_env_inst_idx: {minn_env_inst_idx}")
            
            tot_envs_hand_qs = batched_index_select(tot_envs_hand_qs, self.env_inst_idxes, dim=0) # nn_envs x nn_envs #
            
            
            
            
            if self.use_multiple_kine_source_trajs and len(self.multiple_kine_source_trajs_fn) > 0 and os.path.exists(self.multiple_kine_source_trajs_fn):
                # print(f"envs_kine_source_trajs_idxes: {self.envs_kine_source_trajs_idxes[10]}")
                tot_envs_hand_qs = batched_index_select(self.multiple_kine_source_trajs, self.envs_kine_source_trajs_idxes, dim=0)
            
        
            
            
            
            envs_maxx_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
            increased_progress_buf = torch.clamp(increased_progress_buf, min=torch.zeros_like(envs_maxx_episode_length), max=envs_maxx_episode_length)

            
            # maxx_increased_progress_buf = torch.max(increased_progress_buf).item()
            # minn_increased_progress_buf= torch.min(increased_progress_buf).item()
            # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_increased_progress_buf: {maxx_increased_progress_buf}, minn_increased_progress_buf: {minn_increased_progress_buf}")
            
            ctl_kinematics_bias = batched_index_select(tot_envs_hand_qs, increased_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs # 
            
            
            ctl_kinematics_bias = ctl_kinematics_bias.squeeze(1) # nn_envs x nn_hand_dofs #
            
            # random shift the cond fre q #
            # if self.random_shift_cond_freq:
            #     moded_progress_buf = increased_progress_buf // self.env_inv_cond_freq # get the moded progress buf
            #     moded_progress_buf = moded_progress_buf * self.env_inv_cond_freq # get the moded progress buf
            #     ctl_kinematics_bias = batched_index_select(tot_envs_hand_qs, moded_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs # 
            #     ctl_kinematics_bias = ctl_kinematics_bias.squeeze(1)
                
            
            
            # if self.random_shift_cond:
            #     first_frame_envs_hand_qs = tot_envs_hand_qs[:, 0, :] # nn_envs x nn_hand_dof
            #     ctl_kinematics_bias[self.env_cond_type == COND_OBJ] = first_frame_envs_hand_qs[self.env_cond_type == COND_OBJ]
            #     # increased_progress_buf = increased_progress_buf.squeeze(1)
            #     # increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1)
            #     # ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]
                
            if self.only_use_hand_first_frame:
                first_frame_envs_hand_qs = tot_envs_hand_qs[:, 0, :] # nn_envs x nn_hand_dof
                ctl_kinematics_bias = first_frame_envs_hand_qs
            
            
            # ### NOTE: assume we have the base trajectory ###
            # if self.use_forcasting_model and self.already_forcasted: # if the current progress buf is #
            #     forcast_ctl_kinematics_bias = batched_index_select(self.forcast_shadow_hand_dof_pos, increased_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs #
            #     forcast_ctl_kinematics_bias = forcast_ctl_kinematics_bias.squeeze(1) # nn_envs x nn_hand_dofs
            #     using_forcast_res_step_threshold = self.using_forcast_res_step_threshold #  60
            #     ctl_kinematics_bias[self.progress_buf >= using_forcast_res_step_threshold] = forcast_ctl_kinematics_bias[self.progress_buf >= using_forcast_res_step_threshold]
            
            
            if self.use_twostage_rew:
                grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
                expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
                grasp_manip_stages_flag = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
                ctl_kinematics_bias = torch.where(
                    grasp_manip_stages_flag, expanded_grasp_frame_hand_qpos, ctl_kinematics_bias
                )
            
            # delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            if not self.downsample:
                delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            else:
                delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions  * 2.0 * self.control_freq_inv
            
            # # hand_glb_mult_factor_scaling_coef, hand_glb_mult_scaling_progress_after
            if self.hand_glb_mult_scaling_progress_after < 500:
                shadow_hand_dof_speed_scale_tsr_expanded = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0).repeat(self.num_envs, 1) # nn_envs x nn_hand_dofs # 
                shadow_hand_dof_speed_scale_tsr_expanded[self.progress_buf >= self.hand_glb_mult_scaling_progress_after, :6] = self.hand_glb_mult_factor_scaling_coef * shadow_hand_dof_speed_scale_tsr_expanded[self.progress_buf >= self.hand_glb_mult_scaling_progress_after, :6]
            
            
            cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
            self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
            self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
            if self.kinematics_only:
                cur_targets = ctl_kinematics_bias
            else:
                cur_targets = ctl_kinematics_bias + self.cur_delta_targets[:, self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(cur_targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            
            
            if self.w_impedance_bias_control:
                impedance_delta_acts = actions.clone().to(self.device)[..., self.nn_hand_dof: self.nn_hand_dof * 2] # get the hand delta acts #
                impedance_joint_stiffness = actions.clone().to(self.device)[..., self.nn_hand_dof * 2: self.nn_hand_dof * 3 ] # the dof stiffnes values #
                impedance_delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * impedance_delta_acts # get the impedance delta acts #
                impedance_cur_delta_targets = self.prev_delta_targets_impedance[:, self.actuated_dof_indices] + impedance_delta_delta_targets
                self.cur_delta_targets_impedance[:, self.actuated_dof_indices] = impedance_cur_delta_targets
                self.prev_delta_targets_impedance[:, self.actuated_dof_indices] = self.cur_delta_targets_impedance[:, self.actuated_dof_indices]
                cur_targets_impedance =  ctl_kinematics_bias + self.cur_delta_targets_impedance[:, self.actuated_dof_indices]
                self.cur_targets_impedance[:, self.actuated_dof_indices] = tensor_clamp(cur_targets_impedance, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
                impedance_joint_stiffness = (impedance_joint_stiffness + 1) / 2.0
                impedance_forces = impedance_joint_stiffness * (self.shadow_hand_dof_pos - self.cur_targets_impedance)
                self.cur_targets_impedance[:, self.actuated_dof_indices] = impedance_forces
            
        
        
        elif self.use_kinematics_bias:
            increased_progress_buf = self.progress_buf + 1
            increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
            # get the kinematicsof the increaesd progres buf as the kinematics bias # 
            # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
            # hand_qs_th
            ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]
            # ctl kinematics bias #
            if self.kinematics_only:
                targets = ctl_kinematics_bias
            else:
                # targets = ctl_kinematics_bias + self.shadow_hand_dof_speed_scale * self.dt * self.actions 
                #### from actions to targets ####
                targets = ctl_kinematics_bias + self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            # kinematics_only # targets #
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        elif self.use_relative_control:
            # 
            # targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
            # shadow_hand_dof_speed_scale_tsr # 
            targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            # targets = self.prev_targets #
        else:
            self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

            ### TODO: check whether if it is because we use the shadow hand dof pos to set the dof state tensor, so we need to set the dof state tensor here ###
            # self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
            # self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

            # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
            #                                         gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        self.prev_dof_vel[:, :] = self.cur_dof_vel[:, :]
        
        if self.w_impedance_bias_control:
            self.gym.set_dof_actuation_force_tensor_indexed(
                self.sim, 
                gymtorch.unwrap_tensor(self.cur_targets_impedance),
                gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices)
            )
            pass
        
        

    def post_physics_step(self):
        
        self.progress_buf += 1
        self.randomize_buf += 1
        
        if self.use_twostage_rew:
            self.grasping_progress_buf += 1

        self.ref_ts += 1
        
        self.compute_observations()
        
        ## TODO: to compute the reward for the forecasting model ##
        ## TODO: to compute the reward for the forecasting model: it is composed of two parts, 1) the reward of the controller model, 2) the reward between the forecasted information and theses corresponding kinematics --- the time indexes should increase from progress_buf to progress_buf + self.forward_forecasting_nn_steps #
        
        
        
        # if self.use_forcasting_model:
        #     self._forward_forcasting_model() # forward the forcasting model #
        #     self.try_save_network_forwarding_info_dict() # try to save them #
        
        # print(f"To compute reward with ref_ts: {self.ref_ts}")
        # 
        self.compute_reward(self.actions)
        
        
        
        """Compute the part-2 reward for the forecasting model """
        envs_goal_obj_trans_th = batched_index_select(self.tot_kine_obj_trans, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(self.tot_kine_obj_ornt, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        envs_goal_hand_qs_th = batched_index_select(self.tot_hand_preopt_res, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x nn_hand_dofs
        
        
        
        tot_diff_pred_with_gt_targets_hand_qs = []
        tot_diff_pred_with_gt_targets_obj_pos = []
        tot_diff_pred_with_gt_targets_obj_ornt = []
        
        # pred_tracking_targets_obj_pos, pred_tracking_targets_obj_rot, pred_tracking_targets_hand_qs
        envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        for i_ws in range(self.forward_forecasting_nn_steps):
            # forward forecasting nn steps # 
            cur_progress_buf = self.progress_buf + i_ws
            cur_progress_buf = torch.clamp(cur_progress_buf, min=torch.zeros_like(cur_progress_buf), max=envs_episode_length) # 
            # nn_envs x nn_hand_qs_dim; nn_envs x 3; nn_envs x 4 #
            pred_tracking_targets_hand_qs = batched_index_select(self.pred_tracking_targets_hand_qs, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x nn_hand_qs_dim #
            pred_tracking_targets_obj_pos = batched_index_select(self.pred_tracking_targets_obj_pos, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 3 
            pred_tracking_targets_obj_rot = batched_index_select(self.pred_tracking_targets_obj_rot, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 4 
            # gt_targets_hand_qs = batched_index_select(self.)
            
            gt_targets_hand_qs = batched_index_select(envs_goal_hand_qs_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x nn_hand_qs_dim #
            gt_targets_obj_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 3
            gt_targets_obj_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(1), dim=1).contiguous().squeeze(1) # nn_envs x 4
            
            diff_pred_with_gt_targets_hand_qs = torch.norm(pred_tracking_targets_hand_qs - gt_targets_hand_qs, p=2, dim=-1) # nn_envs #
            # rewards --- return self.rewards? # 
            diff_pred_with_gt_targets_obj_pos = torch.norm(pred_tracking_targets_obj_pos - gt_targets_obj_pos, p=2, dim=-1) # nn_envs # 
            diff_pred_with_gt_targets_obj_rot = torch.norm(pred_tracking_targets_obj_rot - gt_targets_obj_rot, p=2, dim=-1) # nn_envs #
        
            tot_diff_pred_with_gt_targets_hand_qs.append(diff_pred_with_gt_targets_hand_qs)
            tot_diff_pred_with_gt_targets_obj_pos.append(diff_pred_with_gt_targets_obj_pos)
            tot_diff_pred_with_gt_targets_obj_ornt.append(diff_pred_with_gt_targets_obj_rot)
            
            # print(f"pred_tracking_targets_hand_qs: {(pred_tracking_targets_hand_qs - gt_targets_hand_qs).mean(dim=0)}")
            
        tot_diff_pred_with_gt_targets_hand_qs = torch.stack(tot_diff_pred_with_gt_targets_hand_qs, dim=1) # nn_envs x forward_forecasting_nn_steps #
        tot_diff_pred_with_gt_targets_obj_pos = torch.stack(tot_diff_pred_with_gt_targets_obj_pos, dim=1) # nn_envs x forward_forecasting_nn_steps #
        tot_diff_pred_with_gt_targets_obj_ornt = torch.stack(tot_diff_pred_with_gt_targets_obj_ornt, dim=1) # nn_envs x forward_forecasting_nn_steps #
        
        # nn_envs, nn_envs, nn_envs #
        diff_hand_qs = torch.mean(tot_diff_pred_with_gt_targets_hand_qs, dim=-1) # nn_envs #
        diff_obj_pos = torch.mean(tot_diff_pred_with_gt_targets_obj_pos, dim=-1) # nn_envs #
        diff_obj_ornt = torch.mean(tot_diff_pred_with_gt_targets_obj_ornt, dim=-1) # nn_envs #
        # diff hand qs, diff obj pos, diff obj ornt #
        # diff hand qs, diff obj pos, diff obj ornt #
        rew_diff_pred_targets_with_gt = 0.5 * diff_hand_qs # + 0.25 * diff_obj_pos + 0.25 * diff_obj_ornt
        # print(f"diff_hand_qs: {diff_hand_qs.mean()}, diff_obj_pos: {diff_obj_pos.mean()}, diff_obj_ornt: {diff_obj_ornt.mean()}, forward_forecasting_nn_steps: {self.forward_forecasting_nn_steps}")
        # rew_diff_pred_targets_with_gt = -1.0 * rew_diff_pred_targets_with_gt
        # rew_diff_pred_targets_with_gt = -0.05 * rew_diff_pred_targets_with_gt
        rew_diff_pred_targets_with_gt = -1* rew_diff_pred_targets_with_gt
        # self.forecasting_rew_buf[:] = rew_diff_pred_targets_with_gt + self.rew_buf[:] 
        self.forecasting_rew_buf[:] = rew_diff_pred_targets_with_gt * 0.0  + self.rew_buf[:] 
        
        
        
        
        if self.test:
            # if self.ref_ts >= self.max_episode_length - 3:
            self.try_save()
        else:
            # if not self.single_instance_training and self.num_envs < 1000:
            # if not self.single_instance_training and self.sv_info_during_training:
            if self.sv_info_during_training:
                self.try_save_train()
                if self.ref_ts >= 300 * 2:   
                    self.ref_ts = 0
                    
        # self.try_save_forcasting_model()

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                
                
                




    def add_debug_lines(self, env, pos, rot):
        posx = (pos + quat_apply(rot, to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
        posy = (pos + quat_apply(rot, to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
        posz = (pos + quat_apply(rot, to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

        p0 = pos.cpu().numpy()
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posx[0], posx[1], posx[2]], [0.85, 0.1, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posy[0], posy[1], posy[2]], [0.1, 0.85, 0.1])
        self.gym.add_lines(self.viewer, env, 1, [p0[0], p0[1], p0[2], posz[0], posz[1], posz[2]], [0.1, 0.1, 0.85])


#####################################################################
###=========================jit functions=========================###
#####################################################################



def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset+prop[attr].shape[0]]
        return smpl, offset+prop[attr].shape[0]
    else:
        return sample[offset], offset+1



@torch.jit.script
def compute_hanid_reward(
        object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot, target_pos, target_rot,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool, hand_up_threshold_1: float, hand_up_threshold_2: float, num_fingers: int, w_obj_ornt: bool
):
    # Distance from the hand to the object
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)  + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
                #               + torch.norm(
                # object_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
    
    
    finger_dist_threshold = 0.6 * num_fingers #                          
    
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= finger_dist_threshold, finger_dist_threshold + 0 * right_hand_finger_dist,right_hand_finger_dist)

    right_hand_dist_rew = right_hand_dist
    right_hand_finger_dist_rew = right_hand_finger_dist

    action_penalty = torch.sum(actions ** 2, dim=-1)

    # delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    # delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    
    delta_qpos_value = torch.norm(delta_qpos[:, 6:], p=1, dim=-1)
    delta_hand_pos_value = torch.norm(delta_qpos[:, :3], p=1, dim=-1)
    delta_hand_rot_value = torch.norm(delta_qpos[:, 3:6], p=1, dim=-1)
    
    # NOTE: try to change it to a different coefficient? #
    # hand_pos_rew_coef = 0.6
    # hand_rot_rew_coef = 0.04
    # hand_qpos_rew_coef = 0.1
    
    
    hand_pos_rew_coef = 0.6
    hand_rot_rew_coef = 0.1
    hand_qpos_rew_coef = 0.1

    
    delta_value = hand_pos_rew_coef * delta_hand_pos_value + hand_rot_rew_coef * delta_hand_rot_value + hand_qpos_rew_coef * delta_qpos_value 
    
    target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
        

    # orientation? #
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    rot_dist = 3.14 - torch.abs(rot_dist) 
    five_degree_rot_diff = torch.asin(5.0 / 180.0 * 3.1415926535)
    
    lowest = object_pos[:, 2]
    # lift_z = object_init_z[:, 0] + 0.6 +0.003
    lift_z = object_init_z[:, 0] + (hand_up_threshold_1 - 0.030) + 0.003

    if goal_cond:
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        
        inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist) + 0.33 * (3.14 - rot_dist)
        
        goal_hand_rew = torch.where(flag == 5, inhand_obj_pos_ornt_rew, goal_hand_rew)
        
        
        
        flag2 = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        hand_up = torch.zeros_like(right_hand_finger_dist) # lowest is bigger than lift_z #
        hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)
        
        if w_obj_ornt:
            obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            bonus = torch.zeros_like(goal_dist)
            bonus = torch.where(obj_bonus_flat == 2, 1.0 / (1 + 10 * goal_dist), bonus)

        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5*delta_value

    else:
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        
        
        inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist) + 0.33 * (3.14 - rot_dist)
        
        
        # no touch, no object positional reward #
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, inhand_obj_pos_ornt_rew, goal_hand_rew) # 

        ## NOTE: it seems like the following hand_up reward is designed for action space directly predicting actions ##
        ## but perhaps we can still use it in the prev_state based action representation since enouraging the velocity's direction is also reasonable ##
        ## but the lowest threshold should be changed to align with this setting ##
        
        ## NOTE: 1) if the object has been lifted up a little bit, then we continue to encourage the hand up-lifting action; 2) if the object has been lifted to the goal height, -- just give the corresponding reward!
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= hand_up_threshold_1, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        # 
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)

        
        if w_obj_ornt:
            obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            bonus = torch.zeros_like(goal_dist)
            bonus = torch.where(obj_bonus_flat == 2, 1.0 / (1 + 10 * goal_dist), bonus)
        
        reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus
    
    # 
    
    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)

    goal_resets = resets
    # sucesses depends on the goal dist #
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes



@torch.jit.script
def compute_hand_reward_tracking(
        object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length, object_pos, object_handle_pos, object_back_pos, object_rot,
        object_linvel, object_angvel, goal_linvel, goal_angvel,
        target_pos, target_rot, target_lifting_pos,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool, hand_up_threshold_1: float, hand_up_threshold_2: float, num_fingers: int, w_obj_ornt: bool, w_obj_vels: bool, separate_stages: bool, hand_pose_guidance_glb_trans_coef: float, hand_pose_guidance_glb_rot_coef: float , hand_pose_guidance_fingerpose_coef: float, rew_finger_obj_dist_coef: float, rew_delta_hand_pose_coef: float, rew_obj_pose_coef: float, goal_dist_thres: float , envs_hand_qtars, env_hand_cur_targets, use_hand_actions_rew: bool, prev_dof_vel, cur_dof_vel, rew_smoothness_coef: float, early_terminate: float, env_cond_type, env_cond_hand_masks, compute_hand_rew_buf_threshold: int, cur_ornt_rew_coef: float
):
    if separate_stages:
        lowest = object_pos[:, 2].unsqueeze(-1).repeat(1, 3)
        # calculate the target pos based on the target lifting pose #
        target_pos = torch.where(lowest < 0.19, target_lifting_pos, target_pos)
        # target pos, object pose # object pose #
    
    
    # obj_pos_cur_w_target_dist = torch.norm(target_pos - object_pos, p=)
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # target pos, right hand pos # target pos change frequency? ## pose change frequencies ##
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)  + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
                #               + torch.norm( 
                # object_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
    # idx=6 #
    # print(f"right_hand_dist: {right_hand_dist[idxxx]}, object_handle_pos: {object_handle_pos[idxxx]},right_hand_pos: {right_hand_pos[idxxx]}, object_pos: {object_pos[idxxx]}") 
    
    # finger dist threshold #
    finger_dist_threshold = 0.6 * num_fingers 
    
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= finger_dist_threshold, finger_dist_threshold + 0 * right_hand_finger_dist,right_hand_finger_dist)
    

    
    # right_hand_dist_rew = right_hand_dist
    # right_hand_finger_dist_rew = right_hand_finger_dist

    # action_penalty = torch.sum(actions ** 2, dim=-1)

    # delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    # delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    
    # delta targets #
    # delta_glb_pos_targets, delta_glb_rot_targets, delta_finger_pos_targets # 
    delta_targets = envs_hand_qtars - env_hand_cur_targets # 
    delta_glb_pos_targets = torch.norm(delta_targets[:, :3], p=1, dim=-1)
    delta_glb_rot_targets = torch.norm(delta_targets[:, 3:6], p=1, dim=-1)
    delta_finger_pos_targets = torch.norm(delta_targets[:, 6:], p=1, dim=-1)
    rew_glb_pos_targets = torch.exp(-50.0 * delta_glb_pos_targets)
    rew_glb_rot_targets = torch.exp(-50.0 * delta_glb_rot_targets)
    rew_finger_pos_targets = torch.exp(-200.0 * delta_finger_pos_targets)
    
    # sav the actions at each time? #
    
    
    delta_qpos[env_cond_type == 2] = delta_qpos[env_cond_type == 2] * env_cond_hand_masks[env_cond_type == 2] # envs x nn_hand_dof xxxxx envs x nn_hand_dof #
    
    delta_qpos_value = torch.norm(delta_qpos[:, 6:], p=1, dim=-1)
    delta_hand_pos_value = torch.norm(delta_qpos[:, :3], p=1, dim=-1)
    delta_hand_rot_value = torch.norm(delta_qpos[:, 3:6], p=1, dim=-1)
    
    # NOTE: try to change it to a different coefficient? #
    # hand_pos_rew_coef = 0.6
    # hand_rot_rew_coef = 0.04
    # hand_qpos_rew_coef = 0.1
    
    
    
    # encourage the hand pose rewards #
    # hand_pos_rew_coef = 0.6
    hand_pos_rew_coef = hand_pose_guidance_glb_trans_coef #  0.6
    hand_rot_rew_coef = hand_pose_guidance_glb_rot_coef #  0.6
    hand_qpos_rew_coef = hand_pose_guidance_fingerpose_coef #  0.6

    # delta_value[progress_buf >= compute_hand_rew_buf_threshold] = 0.0 # c
    delta_qpos_value[progress_buf >= compute_hand_rew_buf_threshold] = 0.0
    
    
    
    delta_value = hand_pos_rew_coef * delta_hand_pos_value + hand_rot_rew_coef * delta_hand_rot_value + hand_qpos_rew_coef * delta_qpos_value 
    
    # delta_value[progress_buf >= compute_hand_rew_buf_threshold] = 0.0 # c
    
    
    
    target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
    
    # quat_ # 

    # orientation? #
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    # rot_dist = 3.14 - torch.abs(rot_dist) 
    # rot
    # five_degree_rot_diff = torch.asin(5.0 / 180.0 * 3.1415926535) # 0.0874
    five_degree_rot_diff = 5.0 / 180.0 * 3.1415926535 # 0.08726646259722222 ##
    
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot)) # conjugate #  #
    # euler_xyz_diff = get_euler_xyz(quat_diff)
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(euler_xyz_diff, p=2, dim=-1), max=1.0))
    
    
    smoothness_rew = -torch.norm(
        (prev_dof_vel - cur_dof_vel), p=2, dim=-1
    ) * rew_smoothness_coef
    
    # # 
    lin_vel_rew = torch.zeros_like(goal_dist)
    ang_vel_rew = torch.zeros_like(goal_dist)
    lin_vel_bonus = torch.zeros_like(goal_dist)
    ang_vel_bonus = torch.zeros_like(goal_dist) # 
    
    if w_obj_vels:
        lin_vel_dist = torch.norm(object_linvel - goal_linvel, p=2, dim=-1) # (nn_envs, )
        ang_vel_dist = torch.norm(object_angvel - goal_angvel, p=2, dim=-1) # (nn_envs, )
        
        lin_vel_rew = 1 / float(120) * (120 * 0.9 - 2 * lin_vel_dist) #
        ang_vel_rew = 1 / float(120) * (120 * 0.9 - 2 * ang_vel_dist) #
        
        lin_vel_bonus_thres = 0.05 * 12 # 0.6 is toleratble # 
        ang_vel_bonus_thres = 0.05 * 12 # 
        
        lin_vel_bonus = torch.zeros_like(lin_vel_dist)
        lin_vel_bonus = torch.where(lin_vel_dist <= lin_vel_bonus_thres, 1.0 / (1 + 10 * (lin_vel_dist / float(120))), lin_vel_bonus)
        
        ang_vel_bonus = torch.zeros_like(ang_vel_dist) # ang vel bonus #
        ang_vel_bonus = torch.where(ang_vel_dist <= ang_vel_bonus_thres, 1.0 / (1 + 10 * (ang_vel_dist / float(120))), ang_vel_bonus)
        
    
    
    # lowest #
    lowest = object_pos[:, 2]
    # lift_z = object_init_z[:, 0] + 0.6 +0.003
    lift_z = object_init_z[:, 0] + (hand_up_threshold_1 - 0.030) + 0.003

    if goal_cond: ## 
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        
        ##### inhand obj #####
        inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + 0.33 * (3.14 - rot_dist)
        
        if w_obj_vels:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + lin_vel_rew + ang_vel_rew
        
        goal_hand_rew = torch.where(flag == 5, inhand_obj_pos_ornt_rew, goal_hand_rew)
        
        
        flag2 = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        hand_up = torch.zeros_like(right_hand_finger_dist) # lowest is bigger than lift_z # # 
        hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)
        
        if w_obj_ornt: 
            
            obj_bonus_flat = (rot_dist <= five_degree_rot_diff).int()
            # obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            obj_ornt_bonus = torch.zeros_like(rot_dist) 
            # use the rot_dist to compute the bonus #
            obj_ornt_bonus = torch.where(obj_bonus_flat == 1, 1.0 / (1 + 10 * rot_dist * 0.5), obj_ornt_bonus)
            # obj_ornt_bonus = torch.
            bonus = bonus + obj_ornt_bonus
        
        if w_obj_vels:
            bonus = bonus + lin_vel_bonus + ang_vel_bonus
            

        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5 * delta_value
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 5.0 * delta_value
        reward = goal_hand_rew + hand_up + bonus - 5.0 * delta_value 

    else:
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        
        
        progress_buf_rot_indicators = (progress_buf >= compute_hand_rew_buf_threshold).int()
        ##### original version #####
        # inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        ##### original version #####
        
        inhand_obj_pos_ornt_rew = 1 * (0.0 - 2 * goal_dist)
        
        if w_obj_ornt:
            # inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist) + 0.33 * (3.14 - rot_dist)
            # inhand_obj_pos_ornt_rew = 1 * (0.0 - 2 * goal_dist) + 0.33 * (0.174 - rot_dist)
            # rot_dist_rew = 0.33 * (0.0 - rot_dist)
            # # progress_buf_rot_indicators = (progress_buf >= compute_hand_rew_buf_threshold).int()
            # rot_dist_rew = rot_dist_rew * progress_buf_rot_indicators # progress buf not indicators #
            
            
            # rot_dist_rew = 0.03 * (0.0 - rot_dist)
            # rot_dist_rew = 0.33 * (0.0 - rot_dist)
            rot_dist_rew = cur_ornt_rew_coef * (0.0 - rot_dist)
            # progress_buf_rot_indicators = (progress_buf >= compute_hand_rew_buf_threshold).int()
            rot_dist_rew = rot_dist_rew # * progress_buf_rot_indicators # progress buf not indicators #
            # progress buf rot indicators #
            inhand_obj_pos_ornt_rew = 1 * (0.0 - 2 * goal_dist) + rot_dist_rew #  0.33 * (0.3491 - rot_dist)
        
        if w_obj_vels:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + lin_vel_rew + ang_vel_rew
        
        
        
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, inhand_obj_pos_ornt_rew, goal_hand_rew)

        # 
        ## NOTE: it seems like the following hand_up reward is designed for action space directly predicting actions ##
        ## but perhaps we can still use it in the prev_state based action representation since enouraging the velocity's direction is also reasonable ##
        ## but the lowest threshold should be changed to align with this setting ### yep #
        
        
        
        ## NOTE: 1) if the object has been lifted up a little bit, then we continue to encourage the hand up-lifting action; 2) if the object has been lifted to the goal height, -- just give the corresponding reward! #
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= hand_up_threshold_1, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)
        


        # hand_additional_up = torch.zeros_like(right_hand_finger_dist)
        # # actions are used to comptue the rewards ? # # 
        # hand_additional_up = torch.where(lowest < 0.1, torch.where(flag == 2, 0.1 * actions[:, 2], hand_up), hand_up)
        # hand addtional up # is below the threshold # below the threshold ##
        
        
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
        
        # if w_obj_ornt: # w # flag #
        #     obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
        #     bonus = torch.zeros_like(goal_dist) #
        #     bonus = torch.where(obj_bonus_flat == 2, 1.0 / (1 + 10 * goal_dist), bonus) # bonus #
        
        
        ####### calculate object orientation bonus ########
        if w_obj_ornt:  # obj ornt ---  # # obj ornt # #
            
            obj_bonus_flat = (rot_dist <= five_degree_rot_diff).int() + (right_hand_finger_dist <= right_hand_finger_dist_thres).int()
            # obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            obj_ornt_bonus = torch.zeros_like(rot_dist) 
            # use the rot_dist to compute the bonus #
            obj_ornt_bonus = torch.where(obj_bonus_flat == 2, 1.0 / (1 + 10 * rot_dist * 0.5), obj_ornt_bonus)
            
            obj_ornt_bonus = obj_ornt_bonus * progress_buf_rot_indicators # progress buf rot indicators #
            
            bonus = bonus #  + obj_ornt_bonus
        
        ####### calculate object velocity bonus ########
        if w_obj_vels:
            bonus = bonus + lin_vel_bonus + ang_vel_bonus
        
        ###################  ###################
        # print(f"right_hand_finger_dist: {right_hand_finger_dist.mean()}, goal_dist: {goal_dist.mean()}, right_hand_dist: {right_hand_dist.mean()}, goal_hand_rew: {goal_hand_rew.mean()}, bonus: {bonus.mean()},")
        
        #### goal_hand_rew ####
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + bonus + hand_up 
        
        ### original version ####
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + bonus + hand_up # + hand_additional_up
        ### original version ####
        
        # print(f"right_hand_finger_dist: {right_hand_finger_dist.mean()}, goal_dist: {goal_dist.mean()}, right_hand_dist: {right_hand_dist.mean()}, goal_hand_rew: {goal_hand_rew.mean()}, bonus: {bonus.mean()}, hand_up: {hand_up.mean()}")
        
        ### original version - 2 ###
        # reward = -0.5 * right_hand_finger_dist + goal_hand_rew + bonus + hand_up 
        ### original version - 2 ###
        
        # grasp frame # # 
        
        hand_dist_flag = (right_hand_dist <= 0.12).int()
        right_hand_finger_dist = torch.where(hand_dist_flag == 1, right_hand_finger_dist, 0.0 * right_hand_finger_dist)
        
        
        # delta_value[env_cond_type == COND_OBJ] = 0.0
        # goal_hand_rew[env_cond_type == COND_HAND] = 0.0
        
        delta_value[env_cond_type == 1] = 0.0
        # goal_hand_rew[env_cond_type == 2] = 0.0
        # delta_value[env_cond_type == 2] = delta_value[env_cond_type == 2] * 
        
        # delta_value[progress_buf >= compute_hand_rew_buf_threshold] = 0.0 # c
        
        
        
        
        reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)  + goal_hand_rew + bonus # + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
    
    
    
    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)
    
    if early_terminate:
        obj_pos_terminate_resets = torch.where(goal_dist >= 0.2, torch.ones_like(resets), torch.zeros_like(resets))
        tot_resets_flags = (obj_pos_terminate_resets.int() + resets.int()) >= 1 # if not >=1 then it must be 0 -- no reset and not early terminated #
        resets = torch.where(tot_resets_flags, torch.ones_like(resets), resets)
        
    
    step_resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), torch.zeros_like(resets))

    goal_resets = resets
    # sucesses depends on the goal dist #
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)
    
    
    
    
    if use_hand_actions_rew:
        # print(f"using use_hand_actions_rew")
        objreward_coef = 1.0
        # objreward_coef = 0.0
        finger_rew_coef = 1.0
    
        reward = reward * objreward_coef +  rew_glb_pos_targets + rew_glb_rot_targets + rew_finger_pos_targets   * finger_rew_coef
        # reward = reward +  rew_glb_pos_targets + rew_glb_rot_targets + rew_finger_pos_targets
    
    
    reward = reward + smoothness_rew
    
    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes



@torch.jit.script
def compute_hand_reward_tracking_taco(
        object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot,
        object_linvel, object_angvel, goal_linvel, goal_angvel,
        lift_obj_pos_th, 
        target_pos, target_rot, target_lifting_pos,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos,
        right_hand_th_pos,
        gt_hand_palm_pos, gt_hand_thumb_pos, gt_hand_index_pos, gt_hand_middle_pos, gt_hand_ring_pos ,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool, hand_up_threshold_1: float, hand_up_threshold_2: float, num_fingers: int, w_obj_ornt: bool, w_obj_vels: bool, separate_stages: bool, hand_pose_guidance_glb_trans_coef: float, hand_pose_guidance_glb_rot_coef: float , hand_pose_guidance_fingerpose_coef: float, rew_finger_obj_dist_coef: float, rew_delta_hand_pose_coef: float, rew_obj_pose_coef: float, goal_dist_thres: float, lifting_separate_stages: bool, reach_lifting_stage, strict_lifting_separate_stages: float 
):
    # Distance from the hand to the object
    # target hand pos, object pos 
    # print(f"in taco rew function")
    # use separte stages -> use the lifting pos #
    if separate_stages:
        ######## whether to use separate stages ########
        lowest = object_pos[:, 2].unsqueeze(-1).repeat(1, 3)
        # calculate the target pos based on the target lifting pose #
        target_pos = torch.where(lowest < 0.19, target_lifting_pos, target_pos) # target lifting pose # 
        # # target pos, object # use separate stages should be ture? 
        # use lifting separate stages? # # reach lifting stage # 
        
    if lifting_separate_stages:
        
        if strict_lifting_separate_stages:
            target_lifting_pos = lift_obj_pos_th.unsqueeze(0).contiguous().repeat(target_pos.size(0), 1).contiguous()
            change_target_flag = (reach_lifting_stage == 0).int() #
            change_target_flag = (change_target_flag == 1).unsqueeze(-1).repeat(1, 3).contiguous() 
            # if taret lifting pos is true --- set it to the target lifting pos # # set it to target # # # set it to the target # # set it to the target #
            target_pos = torch.where(change_target_flag, target_lifting_pos, target_pos )
        else:
            lowest = object_pos[:, 2] # .unsqueeze(-1).repeat(1, 3)
            lift_height_z = lift_obj_pos_th[2].item()
            target_lifting_pos = lift_obj_pos_th.unsqueeze(0).contiguous().repeat(target_pos.size(0), 1).contiguous()
            # change_target_flag = (lowest < lift_height_z).int() + (reach_lifting_stage == 0).int() # change target flag # 
            change_target_flag = (reach_lifting_stage == 0).int() # 
            change_target_flag = (change_target_flag == 1).unsqueeze(-1).repeat(1, 3).contiguous() # change the target flag #
            # target_pos = torch.where(lowest < lift_height_z, target_lifting_pos, target_pos )
            target_pos = torch.where(change_target_flag, target_lifting_pos, target_pos ) #
        
        
    # reach 
    ##### reach lifting stage #####
    # tot_reach_lifting_stage = torch.mean(reach_lifting_stage.float())
    # print(f"tot_reach_lifting_stage: {tot_reach_lifting_stage}")
        
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # target pos, right hand pos # target pos change frequency? #
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    # right hand finger dist #
    # right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
    #     object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)  + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    #             #               + torch.norm(
    #             # object_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
    
    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)  + 5.0 * torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
    idxxx= 6
    # print(f"right_hand_dist: {right_hand_dist[idxxx]}, object_handle_pos: {object_handle_pos[idxxx]},right_hand_pos: {right_hand_pos[idxxx]}, object_pos: {object_pos[idxxx]}")
    # idxxx = 6 and the idxxx # reach lift stage # # and additional thumb joint idexes and the corresponding rew term #?
    
    # finger dist threshold #
    finger_dist_threshold = 0.6 * num_fingers                         
    
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= finger_dist_threshold, finger_dist_threshold + 0 * right_hand_finger_dist,right_hand_finger_dist)


    # thumb_finger_dist = thumb_dof_idxes

    # right_hand_dist_rew = right_hand_dist
    # right_hand_finger_dist_rew = right_hand_finger_dist

    # action_penalty = torch.sum(actions ** 2, dim=-1)

    # delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    # delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    
    delta_qpos_value = torch.norm(delta_qpos[:, 6:], p=1, dim=-1)
    delta_hand_pos_value = torch.norm(delta_qpos[:, :3], p=1, dim=-1)
    delta_hand_rot_value = torch.norm(delta_qpos[:, 3:6], p=1, dim=-1)
    # [10, 11, 12, 13]
    
    
    
    delta_thumb_joint_value = torch.norm(delta_qpos[:, 10:14], p=1, dim=-1)
    
    diff_hand_palm_pos = torch.norm(right_hand_pos - gt_hand_palm_pos[:, :3], p=2, dim=-1)
    diff_hand_thumb_pos = torch.norm(right_hand_th_pos - gt_hand_thumb_pos[:, :3], p=2, dim=-1)
    diff_hand_index_pos = torch.norm(right_hand_ff_pos - gt_hand_index_pos[:, :3], p=2, dim=-1)
    diff_hand_middle_pos = torch.norm(right_hand_mf_pos - gt_hand_middle_pos[:, :3], p=2, dim=-1)
    diff_hand_ring_pos = torch.norm(right_hand_rf_pos - gt_hand_ring_pos[:, :3], p=2, dim=-1)
    
    thumb_coef = 10.0
    index_coef = 5.0
    delta_poses_value = diff_hand_palm_pos + thumb_coef * diff_hand_thumb_pos + index_coef * diff_hand_index_pos + diff_hand_middle_pos + diff_hand_ring_pos
    
    
    # NOTE: try to change it to a different coefficient? #
    # hand_pos_rew_coef = 0.6
    # hand_rot_rew_coef = 0.04
    # hand_qpos_rew_coef = 0.1
    
    
    
    # encourage the hand pose rewards #
    # hand_pos_rew_coef = 0.6
    hand_pos_rew_coef = hand_pose_guidance_glb_trans_coef #  0.6
    hand_rot_rew_coef = hand_pose_guidance_glb_rot_coef #  0.6
    hand_qpos_rew_coef = hand_pose_guidance_fingerpose_coef #  0.6
    hand_thumb_rew_coef = 1.0

    
    delta_value = hand_pos_rew_coef * delta_hand_pos_value + hand_rot_rew_coef * delta_hand_rot_value + hand_qpos_rew_coef * delta_qpos_value + hand_thumb_rew_coef * delta_thumb_joint_value 
    
    target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
    
    # quat_ # 

    # orientation? # # quat mul #
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    # rot_dist = 3.14 - torch.abs(rot_dist) 
    # rot
    # five_degree_rot_diff = torch.asin(5.0 / 180.0 * 3.1415926535) # 0.0874
    five_degree_rot_diff = 5.0 / 180.0 * 3.1415926535 # 0.08726646259722222 ##
    
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    # euler_xyz_diff = get_euler_xyz(quat_diff)
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(euler_xyz_diff, p=2, dim=-1), max=1.0))
    
    
    # # 
    lin_vel_rew = torch.zeros_like(goal_dist)
    ang_vel_rew = torch.zeros_like(goal_dist)
    lin_vel_bonus = torch.zeros_like(goal_dist)
    ang_vel_bonus = torch.zeros_like(goal_dist) # 
    
    if w_obj_vels: # thumb_dof_idxes # and the dof idxes #
        lin_vel_dist = torch.norm(object_linvel - goal_linvel, p=2, dim=-1) # (nn_envs, )
        ang_vel_dist = torch.norm(object_angvel - goal_angvel, p=2, dim=-1) # (nn_envs, )
        
        lin_vel_rew = 1 / float(120) * (120 * 0.9 - 2 * lin_vel_dist) #
        ang_vel_rew = 1 / float(120) * (120 * 0.9 - 2 * ang_vel_dist) #
        
        lin_vel_bonus_thres = 0.05 * 12 # 0.6 is toleratble # 
        ang_vel_bonus_thres = 0.05 * 12 # 
        
        lin_vel_bonus = torch.zeros_like(lin_vel_dist)
        lin_vel_bonus = torch.where(lin_vel_dist <= lin_vel_bonus_thres, 1.0 / (1 + 10 * (lin_vel_dist / float(120))), lin_vel_bonus)
        
        ang_vel_bonus = torch.zeros_like(ang_vel_dist) # ang vel bonus #
        ang_vel_bonus = torch.where(ang_vel_dist <= ang_vel_bonus_thres, 1.0 / (1 + 10 * (ang_vel_dist / float(120))), ang_vel_bonus)
        
    
    
    # lowest #
    lowest = object_pos[:, 2]
    # lift_z = object_init_z[:, 0] + 0.6 +0.003
    lift_z = object_init_z[:, 0] + (hand_up_threshold_1 - 0.030) + 0.003
    
    right_hand_finger_dist_thres = 0.12 * num_fingers
    hand_palm_fingers_obj_contact_flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
    hand_palm_fingers_obj_contact_flag = hand_palm_fingers_obj_contact_flag == 2 

    if goal_cond: ## 
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        
        ##### inhand obj #####
        inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + 0.33 * (3.14 - rot_dist)
        
        if w_obj_vels:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + lin_vel_rew + ang_vel_rew
        
        goal_hand_rew = torch.where(flag == 5, inhand_obj_pos_ornt_rew, goal_hand_rew)
        
        
        flag2 = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        hand_up = torch.zeros_like(right_hand_finger_dist) # lowest is bigger than lift_z # # 
        hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)
        
        if w_obj_ornt: 
            
            obj_bonus_flat = (rot_dist <= five_degree_rot_diff).int()
            # obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            obj_ornt_bonus = torch.zeros_like(rot_dist) 
            # use the rot_dist to compute the bonus #
            obj_ornt_bonus = torch.where(obj_bonus_flat == 1, 1.0 / (1 + 10 * rot_dist * 0.5), obj_ornt_bonus)
            # obj_ornt_bonus = torch.
            bonus = bonus + obj_ornt_bonus
        
        if w_obj_vels:
            bonus = bonus + lin_vel_bonus + ang_vel_bonus
            

        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5 * delta_value
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 5.0 * delta_value
        reward = goal_hand_rew + hand_up + bonus - 5.0 * delta_value 

    else:
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        
        ##### original version #####
        # inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        ##### original version #####
        
        # inhand_obj_pos_ornt_rew = 1 * (0.0 - 2 * goal_dist)
        # inhand_obj_pos_ornt_rew = 1 * (0.5 - 2 * goal_dist)
        # goal_dist_thres #
        inhand_obj_pos_ornt_rew = 1 * (goal_dist_thres - 2 * goal_dist)
        
        
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist) + 0.33 * (3.14 - rot_dist)
        
        if w_obj_vels:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + lin_vel_rew + ang_vel_rew
        
        # no touch, no object positional reward #
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, inhand_obj_pos_ornt_rew, goal_hand_rew) # 

        ## NOTE: it seems like the following hand_up reward is designed for action space directly predicting actions ##
        ## but perhaps we can still use it in the prev_state based action representation since enouraging the velocity's direction is also reasonable ##
        ## but the lowest threshold should be changed to align with this setting ##
        
        ## NOTE: 1) if the object has been lifted up a little bit, then we continue to encourage the hand up-lifting action; 2) if the object has been lifted to the goal height, -- just give the corresponding reward!
        # 
        hand_up = torch.zeros_like(right_hand_finger_dist)
        hand_up = torch.where(lowest >= hand_up_threshold_1, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)
        

        # additional up #
        # hand_additional_up = torch.zeros_like(right_hand_finger_dist)
        # # actions are used to comptue the rewards ? #  #  if object # -- # 
        # # 
        # hand_additional_up = torch.where(lowest < 0.1, torch.where(flag == 2, 0.1 * actions[:, 2], hand_up), hand_up)
        # hand addtional up # flag # # is below the threshold # below the threshold ##  # hand up -- without the hand addtional up ?#


        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
        
        # if w_obj_ornt: # w # flag #
        #     obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
        #     bonus = torch.zeros_like(goal_dist)
        #     bonus = torch.where(obj_bonus_flat == 2, 1.0 / (1 + 10 * goal_dist), bonus)
        
        
        ####### calculate object orientation bonus ########
        if w_obj_ornt:  # obj ornt ---  # 
            
            obj_bonus_flat = (rot_dist <= five_degree_rot_diff).int()
            # obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            obj_ornt_bonus = torch.zeros_like(rot_dist) 
            # use the rot_dist to compute the bonus #
            obj_ornt_bonus = torch.where(obj_bonus_flat == 1, 1.0 / (1 + 10 * rot_dist * 0.5), obj_ornt_bonus)
            # obj_ornt_bonus = torch.
            bonus = bonus + obj_ornt_bonus
        
        ####### calculate object velocity bonus ########
        if w_obj_vels:
            bonus = bonus + lin_vel_bonus + ang_vel_bonus
        
        ################### ######### if we use the strict lifting guidance as a strict guidance for the lifting stage ? ###
        # print(f"right_hand_finger_dist: {right_hand_finger_dist.mean()}, goal_dist: {goal_dist.mean()}, right_hand_dist: {right_hand_dist.mean()}, goal_hand_rew: {goal_hand_rew.mean()}, bonus: {bonus.mean()},")
        
        #### goal_hand_rew ####
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + bonus + hand_up 
        
        ### original version ####
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + bonus + hand_up # + hand_additional_up
        ### original version ####
        
        # print(f"right_hand_finger_dist: {right_hand_finger_dist.mean()}, goal_dist: {goal_dist.mean()}, right_hand_dist: {right_hand_dist.mean()}, goal_hand_rew: {goal_hand_rew.mean()}, bonus: {bonus.mean()}, hand_up: {hand_up.mean()}")
        
        ### original version - 2 ###
        # reward = -0.5 * right_hand_finger_dist + goal_hand_rew + bonus + hand_up 
        ### original version - 2 ###
        
        # grasp frame # 
        
        # hand_dist_flag = (right_hand_dist <= 0.12).int()
        # right_hand_finger_dist = torch.where(hand_dist_flag == 1, right_hand_finger_dist, 0.0 * right_hand_finger_dist)
        # hand up --- # 
        
        if lifting_separate_stages: 
            if strict_lifting_separate_stages:
                ### hand up ###
                hand_up = torch.where(
                    reach_lifting_stage == 0, hand_up, torch.where(flag == 2, torch.zeros_like(hand_up) + 0.2, torch.zeros_like(hand_up)) # lift the hand up # 
                )
        
            # exp -100 * delta value #
            # # rew
            # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)  + rew_obj_pose_coef * (goal_hand_rew + bonus) + hand_up 
            
            # use exp rewards # use exp rewards #
            rew_delta_value = torch.exp(-100.0 * delta_value)
            rew_hand_obj_dist = torch.exp(-100.0 * (right_hand_finger_dist + 2.0 * right_hand_dist))
            # rew_delta_value = torch.exp(-10.0 * delta_value)
            # rew_hand_obj_dist = torch.exp(-10.0 * (right_hand_finger_dist + 2.0 * right_hand_dist))
            rew_obj_pose = (goal_hand_rew + bonus)
            rew_hand_up = hand_up
            rew_delta_value = torch.where(
                right_hand_dist < 0.06, rew_delta_value, torch.zeros_like(rew_delta_value)
            )
            reward = (rew_delta_hand_pose_coef) * rew_delta_value + (rew_finger_obj_dist_coef) * rew_hand_obj_dist  + rew_obj_pose_coef * rew_obj_pose + rew_hand_up 
        else:
            # reward = (-rew_delta_hand_pose_coef) * (delta_value + delta_poses_value) + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)  + rew_obj_pose_coef * (goal_hand_rew + bonus) + hand_up 
            
            reward = (-rew_delta_hand_pose_coef) * (delta_value + delta_poses_value) + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist)  + rew_obj_pose_coef * (goal_hand_rew + bonus) + hand_up 
            
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
    
    # resets buf #
    resets = reset_buf
    
    # re

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    step_resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), torch.zeros_like(resets))

    goal_resets = resets
    # sucesses depends on the goal dist #
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes, hand_palm_fingers_obj_contact_flag, right_hand_dist


# two stages reward #

@torch.jit.script
def compute_hand_reward_tracking_twostages( 
        object_init_z, delta_qpos, delta_target_hand_pos, delta_target_hand_rot,
        object_id, dof_pos, rew_buf, reset_buf, reset_goal_buf, progress_buf, successes, current_successes, consecutive_successes,
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot,
        object_linvel, object_angvel, goal_linvel, goal_angvel,
        target_pos, target_rot, target_lifting_pos,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos,
        grasping_frame_hand_pose, # grasping hand pose # 
        grasping_progress_buf, # the grasping progress buffer #
        grasping_manip_stage,
        manip_frame_hand_pose,
        hand_pose,
        maxx_grasping_steps: int,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool, hand_up_threshold_1: float, hand_up_threshold_2: float, num_fingers: int, w_obj_ornt: bool, w_obj_vels: bool, separate_stages: bool, hand_pose_guidance_glb_trans_coef: float, hand_pose_guidance_glb_rot_coef: float , hand_pose_guidance_fingerpose_coef: float, rew_finger_obj_dist_coef: float, rew_delta_hand_pose_coef: float, test_mode: bool, right_hand_dist_thres: float
):
    # Distance from the hand to the object
    # target hand pos, object pos 
    # TODO: the target object pose should be set to the target obj pose at that frame
    
    # object pos and the target pos #
    # grasping frame -- how to use the grasping frame ? #
    avg_object_pos = torch.mean(object_pos, dim=0)
    if test_mode:
        print(f"object_pos: {object_pos.size()}")
        print(f"avg_object_pos: {avg_object_pos}")
        print(f"aaa")
    # 
    nn_envs = object_pos.size(0)
    expanded_grasping_frames = grasping_frame_hand_pose.unsqueeze(0).repeat(nn_envs, 1)
    # 0 
    grasping_manip_stage_flag = (grasping_manip_stage.int() == 0).unsqueeze(-1).repeat(1, manip_frame_hand_pose.size(-1))
    
    # goal_hand_pose = torch.where(grasping_manip_stage_flag, expanded_grasping_frames, manip_frame_hand_pose) # the grasping ad the manip frame hand pose 
    # 
    
    goal_hand_pose = expanded_grasping_frames
    
    delta_qpos = hand_pose - goal_hand_pose # nn_envs x nn_hand_dofs #
    
    # # 
    
    if test_mode:
        avg_obj_target_pos = torch.mean(target_pos, dim=0)
        print(f"avg_obj_target_pos: {avg_obj_target_pos}")
    
    # if separate_stages:
    #     lowest = object_pos[:, 2].unsqueeze(-1).repeat(1, 3)
    #     # calculate the target pos based on the target lifting pose #
    #     target_pos = torch.where(lowest < 0.19, target_lifting_pos, target_pos)
    #     # target pos, object pose # object pose #
    
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # target pos, right hand pos # target pos change frequency? ## pose change frequencies ##
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)  + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
                #               + torch.norm(
                # object_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
    
    # finger dist threshold #
    finger_dist_threshold = 0.6 * num_fingers                         
    
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= finger_dist_threshold, finger_dist_threshold + 0 * right_hand_finger_dist,right_hand_finger_dist)


    # right_hand_dist_rew = right_hand_dist
    # right_hand_finger_dist_rew = right_hand_finger_dist

    # action_penalty = torch.sum(actions ** 2, dim=-1)

    # delta_hand_pos_value = torch.norm(delta_target_hand_pos, p=1, dim=-1)
    # delta_hand_rot_value = 2.0 * torch.asin(torch.clamp(torch.norm(delta_target_hand_rot[:, 0:3], p=2, dim=-1), max=1.0))
    
    delta_qpos_value = torch.norm(delta_qpos[:, 6:], p=1, dim=-1)
    delta_hand_pos_value = torch.norm(delta_qpos[:, :3], p=1, dim=-1)
    delta_hand_rot_value = torch.norm(delta_qpos[:, 3:6], p=1, dim=-1)
    
    l2_delta_hand_pos_value = torch.norm(delta_qpos[:, :3], p=2, dim=-1) # nn_envs
    near_hand_pose_flag = l2_delta_hand_pos_value <= 0.1 # 5 cm 
    
    if test_mode:
        avg_near_hand_pose_flag = torch.mean(near_hand_pose_flag.float())
        print(f"avg_near_hand_pose_flag: {avg_near_hand_pose_flag}")
    
    
    # NOTE: try to change it to a different coefficient? #
    # hand_pos_rew_coef = 0.6
    # hand_rot_rew_coef = 0.04
    # hand_qpos_rew_coef = 0.1
    
    
    
    # encourage the hand pose rewards #
    # hand_pos_rew_coef = 0.6 # pose guidance glb trans coef #
    hand_pos_rew_coef = hand_pose_guidance_glb_trans_coef #  0.6
    hand_rot_rew_coef = hand_pose_guidance_glb_rot_coef #  0.6
    hand_qpos_rew_coef = hand_pose_guidance_fingerpose_coef #  0.6
    
    
    #
    delta_glb_value = hand_pos_rew_coef * delta_hand_pos_value + hand_rot_rew_coef * delta_hand_rot_value
    
    
    delta_value = hand_pos_rew_coef * delta_hand_pos_value + hand_rot_rew_coef * delta_hand_rot_value + hand_qpos_rew_coef * delta_qpos_value 
    
    grasping_and_not_near_hand_pose_flag = (grasping_manip_stage == 0).int() + (near_hand_pose_flag == 0).int()
    # delta_value = torch.where(grasping_and_not_near_hand_pose_flag == 2, delta_glb_value, delta_value)
    
    delta_value = delta_glb_value
    
    if test_mode:
        avg_delta_hand_pos = torch.mean(delta_hand_pos_value)
        avg_delta_hand_rot = torch.mean(delta_hand_rot_value)
        avg_delta_qpos = torch.mean(delta_qpos_value)
        print(f"avg_delta_hand_pos: {avg_delta_hand_pos}, avg_delta_hand_rot: {avg_delta_hand_rot}, avg_delta_qpos: {avg_delta_qpos}")
        
    target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 6.0).int()
    # target_flag = (delta_hand_pos_value <= 0.4).int() + (delta_hand_rot_value <= 1.0).int() + (delta_qpos_value <= 10.0).int()
    
    
    # quat_ # 


    ### NOTE: object orientation differences ###
    # orientation? #
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0)) # if we need to add the 
    # rot_dist = 3.14 - torch.abs(rot_dist) 
    # rot
    # five_degree_rot_diff = torch.asin(5.0 / 180.0 * 3.1415926535) # 0.0874
    five_degree_rot_diff = 5.0 / 180.0 * 3.1415926535 # 0.08726646259722222 ##
    ### NOTE: object orientation differences ###
     
    
    # quat_diff = quat_mul(object_rot, quat_conjugate(target_rot)) # conjugate #  #
    # euler_xyz_diff = get_euler_xyz(quat_diff)
    # rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(euler_xyz_diff, p=2, dim=-1), max=1.0))
    
    
    lin_vel_rew = torch.zeros_like(goal_dist)
    ang_vel_rew = torch.zeros_like(goal_dist)
    lin_vel_bonus = torch.zeros_like(goal_dist)
    ang_vel_bonus = torch.zeros_like(goal_dist) # 
    
    if w_obj_vels:
        lin_vel_dist = torch.norm(object_linvel - goal_linvel, p=2, dim=-1) # (nn_envs, )
        ang_vel_dist = torch.norm(object_angvel - goal_angvel, p=2, dim=-1) # (nn_envs, )
        
        lin_vel_rew = 1 / float(120) * (120 * 0.9 - 2 * lin_vel_dist) #
        ang_vel_rew = 1 / float(120) * (120 * 0.9 - 2 * ang_vel_dist) #
        
        lin_vel_bonus_thres = 0.05 * 12 # 0.6 is toleratble # 
        ang_vel_bonus_thres = 0.05 * 12 # 
        
        lin_vel_bonus = torch.zeros_like(lin_vel_dist)
        lin_vel_bonus = torch.where(lin_vel_dist <= lin_vel_bonus_thres, 1.0 / (1 + 10 * (lin_vel_dist / float(120))), lin_vel_bonus)
        
        ang_vel_bonus = torch.zeros_like(ang_vel_dist) # ang vel bonus #
        ang_vel_bonus = torch.where(ang_vel_dist <= ang_vel_bonus_thres, 1.0 / (1 + 10 * (ang_vel_dist / float(120))), ang_vel_bonus)
        
    
    
    # lowest #
    lowest = object_pos[:, 2]
    # lift_z = object_init_z[:, 0] + 0.6 +0.003
    lift_z = object_init_z[:, 0] + (hand_up_threshold_1 - 0.030) + 0.003

    if goal_cond: ## 
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()  + target_flag
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        
        ##### inhand obj #####
        inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + 0.33 * (3.14 - rot_dist)
        
        if w_obj_vels:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + lin_vel_rew + ang_vel_rew
        
        goal_hand_rew = torch.where(flag == 5, inhand_obj_pos_ornt_rew, goal_hand_rew)
        
        
        flag2 = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        hand_up = torch.zeros_like(right_hand_finger_dist) # lowest is bigger than lift_z # # 
        hand_up = torch.where(lowest >= lift_z, torch.where(flag2 == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag2 == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)

        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus)
        
        if w_obj_ornt: 
            
            obj_bonus_flat = (rot_dist <= five_degree_rot_diff).int()
            # obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            obj_ornt_bonus = torch.zeros_like(rot_dist) 
            # use the rot_dist to compute the bonus #
            obj_ornt_bonus = torch.where(obj_bonus_flat == 1, 1.0 / (1 + 10 * rot_dist * 0.5), obj_ornt_bonus)
            # obj_ornt_bonus = torch.
            bonus = bonus + obj_ornt_bonus
        
        if w_obj_vels:
            bonus = bonus + lin_vel_bonus + ang_vel_bonus
            

        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 0.5 * delta_value
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + hand_up + bonus  - 5.0 * delta_value
        reward = goal_hand_rew + hand_up + bonus - 5.0 * delta_value 

    else:
        right_hand_finger_dist_thres = 0.12 * num_fingers
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= right_hand_dist_thres).int()
        
        
        if test_mode:
            ### finger flag and hand flag ###
            finger_flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int()
            hand_flag = (right_hand_dist <= 0.2).int()
            avg_finger_flag = torch.mean(finger_flag.float()) # avg hand flag and avg hand flag #
            avg_hand_flag = torch.mean(hand_flag.float()) # 
            print(f"avg_finger_flag: {avg_finger_flag}, avg_hand_flag: {avg_hand_flag}")
            ### finger flag and hand flag #### avg hand flag ## avg hand flag ##


        
        ##### original version ##### ## 0.9 - 2 * goal_dist #
        # inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist) #
        ##### original version ##### ## 
        
        inhand_obj_pos_ornt_rew = 1 * (0.0 - 2 * goal_dist)
        
        if w_obj_ornt:
            inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist) + 0.33 * (3.14 - rot_dist)
        
        if w_obj_vels:
            inhand_obj_pos_ornt_rew = inhand_obj_pos_ornt_rew + lin_vel_rew + ang_vel_rew
        
        # no touch, no object positional reward #
        goal_hand_rew = torch.zeros_like(right_hand_finger_dist)
        goal_hand_rew = torch.where(flag == 2, inhand_obj_pos_ornt_rew, goal_hand_rew) # 

        ## NOTE: it seems like the following hand_up reward is designed for action space directly predicting actions ##
        ## but perhaps we can still use it in the prev_state based action representation since enouraging the velocity's direction is also reasonable ##
        ## but the lowest threshold should be changed to align with this setting ##
        
        ## NOTE: 1) if the object has been lifted up a little bit, then we continue to encourage the hand up-lifting action; 2) if the object has been lifted to the goal height, -- just give the corresponding reward!
        # 
        hand_up = torch.zeros_like(right_hand_finger_dist)
        # one possibility --- cannot lift the object up? # on 
        # hand_up = torch.where(lowest >= hand_up_threshold_1, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_1, torch.where(flag == 2, 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)
        
        if test_mode:
            avg_up_act = torch.mean(actions[:, 2])
            avg_hand_up = torch.mean(hand_up)
            print(f"avg_up_act: {avg_up_act}, avg_hand_up: {avg_hand_up}")
            
            

        # additional up # 
        # hand_additional_up = torch.zeros_like(right_hand_finger_dist)
        # # actions are used to comptue the rewards ? #  #  if object # -- # 
        # # 
        # hand_additional_up = torch.where(lowest < 0.1, torch.where(flag == 2, 0.1 * actions[:, 2], hand_up), hand_up)
        # hand addtional up # flag # # is below the threshold # below the threshold ##  # hand up -- without the hand addtional up ?#

        # hand up # flag = () # flag = () # why #
        # flag = () # 
        flag = (right_hand_finger_dist <= right_hand_finger_dist_thres).int() + (right_hand_dist <= 0.12).int()
        bonus = torch.zeros_like(goal_dist)
        bonus = torch.where(flag == 2, torch.where(goal_dist <= 0.05, 1.0 / (1 + 10 * goal_dist), bonus), bonus)
        
        # if w_obj_ornt: # w # flag #
        #     obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
        #     bonus = torch.zeros_like(goal_dist)
        #     bonus = torch.where(obj_bonus_flat == 2, 1.0 / (1 + 10 * goal_dist), bonus)
        
        
        ####### calculate object orientation bonus ########
        if w_obj_ornt:  # obj ornt ---  # 
            
            obj_bonus_flat = (rot_dist <= five_degree_rot_diff).int()
            # obj_bonus_flat = (goal_dist <= 0.05).int() + (rot_dist <= five_degree_rot_diff).int()
            obj_ornt_bonus = torch.zeros_like(rot_dist) 
            # use the rot_dist to compute the bonus #
            obj_ornt_bonus = torch.where(obj_bonus_flat == 1, 1.0 / (1 + 10 * rot_dist * 0.5), obj_ornt_bonus)
            # obj_ornt_bonus = torch.
            bonus = bonus + obj_ornt_bonus
        
        ####### calculate object velocity bonus ########
        if w_obj_vels:
            bonus = bonus + lin_vel_bonus + ang_vel_bonus
        
        ################### #####
        # print(f"right_hand_finger_dist: {right_hand_finger_dist.mean()}, goal_dist: {goal_dist.mean()}, right_hand_dist: {right_hand_dist.mean()}, goal_hand_rew: {goal_hand_rew.mean()}, bonus: {bonus.mean()},")
        
        #### goal_hand_rew ####
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + bonus + hand_up 
        
        ### original version ####
        # reward = -0.5 * right_hand_finger_dist - 1.0 * right_hand_dist + goal_hand_rew + bonus + hand_up # + hand_additional_up
        ### original version ####
        
        # print(f"right_hand_finger_dist: {right_hand_finger_dist.mean()}, goal_dist: {goal_dist.mean()}, right_hand_dist: {right_hand_dist.mean()}, goal_hand_rew: {goal_hand_rew.mean()}, bonus: {bonus.mean()}, hand_up: {hand_up.mean()}")
        
        ### original version - 2 ###
        # reward = -0.5 * right_hand_finger_dist + goal_hand_rew + bonus + hand_up 
        ### original version - 2 ###
        
        # grasp frame # 
        
        ###### Remove this logic ######
        # hand_dist_flag = (right_hand_dist <= 0.12).int() # goal hand rew 
        # right_hand_finger_dist = torch.where(hand_dist_flag == 1, right_hand_finger_dist, 0.0 * right_hand_finger_dist)
        ###### Remove this logic ######
        
        # reward #
        reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
    
    # resets buf ## reset buf ##
    resets = reset_buf
    # resets # 
    # Find out which envs hit the goal and update successes count # ones link # ones link # # 
    resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), resets)
    step_resets = torch.where(progress_buf >= max_episode_length, torch.ones_like(resets), torch.zeros_like(resets))

    grasping_z_succ = torch.abs((object_pos[:, 2] - target_pos[:, 2])) <= (0.05 / 3 )
    # grasping_succ = (goal_dist <= 0.05).int() + (grasping_manip_stage == 0).int()
    grasping_succ = (grasping_z_succ).int() + (grasping_manip_stage == 0).int()
    # grasping_succ = (grasping_manip_stage == 0).int() + (target_flag == 3).int()
    grasping_succ_flag = torch.zeros_like(grasping_succ)
    grasping_succ_flag = torch.where(grasping_succ == 2, 1 + grasping_succ_flag, grasping_succ_flag)
    
    grasping_exceed_steps = (grasping_manip_stage == 0).int() + (grasping_progress_buf >= maxx_grasping_steps).int()
    grasping_exceed_steps = (grasping_exceed_steps == 2).int() + (grasping_succ_flag == 0).int()
    grasping_exceed_steps_flag = torch.zeros_like(grasping_exceed_steps)
    grasping_exceed_steps_flag = torch.where(grasping_exceed_steps == 2, 1 + grasping_exceed_steps_flag, grasping_exceed_steps_flag) #
    
    resets = torch.where(grasping_exceed_steps_flag.int() == 1, torch.ones_like(resets), resets)
    # turning_to_manip_flag = 
    # 

    goal_resets = resets
    # sucesses depends on the goal dist #
    successes = torch.where(goal_dist <= 0.05, torch.ones_like(successes), successes)
    num_resets = torch.sum(resets)
    finished_cons_successes = torch.sum(successes * resets.float())

    current_successes = torch.where(resets, successes, current_successes)
    cons_successes = torch.where(num_resets > 0, av_factor * finished_cons_successes / num_resets + (
                1.0 - av_factor) * consecutive_successes, consecutive_successes)

    return reward, resets, goal_resets, progress_buf, successes, current_successes, cons_successes, grasping_succ_flag



@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(quat_from_angle_axis(rand0 * np.pi, x_unit_tensor),
                    quat_from_angle_axis(rand1 * np.pi, y_unit_tensor))


@torch.jit.script
def randomize_rotation_pen(rand0, rand1, max_angle, x_unit_tensor, y_unit_tensor, z_unit_tensor):
    rot = quat_mul(quat_from_angle_axis(0.5 * np.pi + rand0 * max_angle, x_unit_tensor),
                   quat_from_angle_axis(rand0 * np.pi, z_unit_tensor))
    return rot
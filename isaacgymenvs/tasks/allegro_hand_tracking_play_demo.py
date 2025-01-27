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

# in the test mode #
# load the trajectories to replay here # 
# disable the object gravity; 
# disable the collision between object , hand and the table # 
# then for each timestep --- directly set states and toogle the simulation evolution #

class AllegroHandTrackingPlayDemo(BaseTask):
    # def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
    #              agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, exp_logging_dir=None):
        self.cfg = cfg
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
        self.vel_obs_scale = 0.2 
        self.force_torque_obs_scale = 10.0 
        self.reset_position_noise = self.cfg["env"]["resetPositionNoise"]
        self.reset_rotation_noise = self.cfg["env"]["resetRotationNoise"]
        self.reset_dof_pos_noise = self.cfg["env"]["resetDofPosRandomInterval"]
        self.reset_dof_vel_noise = self.cfg["env"]["resetDofVelRandomInterval"]
        self.shadow_hand_dof_speed_scale = self.cfg["env"]["dofSpeedScale"]
        
        
        self.use_relative_control = self.cfg["env"]["useRelativeControl"]
        self.act_moving_average = self.cfg["env"]["actionsMovingAverage"]
        # debug viz #
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

        
        
        self.use_kinematics_bias = self.cfg['env']['use_kinematics_bias'] ## env and the kinematics bias #
        self.kinematics_only = self.cfg['env']['kinematics_only']
        self.use_kinematics_bias_wdelta = self.cfg['env']['use_kinematics_bias_wdelta']
        # get the control frequency #
        
        self.use_canonical_state = self.cfg['env']['use_canonical_state']
        self.separate_stages = self.cfg['env']['separate_stages']
        self.use_unified_canonical_state = self.cfg['env']['use_unified_canonical_state']
        
        self.rigid_obj_density = self.cfg['env']['rigid_obj_density']
        # self.density = self.cfg["env"]["rigid_obj_density"]
        self.use_fingertips = self.cfg["env"]["use_fingertips"]
        self.glb_trans_vel_scale = self.cfg["env"]["glb_trans_vel_scale"]
        self.glb_rot_vel_scale = self.cfg["env"]["glb_rot_vel_scale"] # get the rot vel scale #
        self.tight_obs = self.cfg["env"]["tight_obs"]
        # hand_pose_guidance_glb_trans_coef, hand_pose_guidance_glb_rot_coef, hand_pose_guidance_fingerpose_coef
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
        
        # isnt tag to pre optimized kinematic trajectories #
        # inst tag to pre optimized kinematic 
        
        # /root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem/ori_grab_s1_cubesmall_offhand_1/coacd/decomposed.obj
        # /root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem/ori_grab_s5_watch_pass_1/coacd/decomposed.obj
        self.ref_ts = 0 
        
        # right_hand_dist_thres # pre optimized traj #
        try:
            self.pre_optimized_traj = self.cfg['env']['pre_optimized_traj']
        except:
            
            self.pre_optimized_traj = None
        
        ## ## right_hand_dist_thres ## ## #
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
            
        # lifting_separate_stages 
        
        try : 
            self.lifting_separate_stages = self.cfg['env']['lifting_separate_stages']
        except:
            self.lifting_separate_stages = False
        
        try : 
            self.strict_lifting_separate_stages = self.cfg['env']['strict_lifting_separate_stages']
        except:
            self.strict_lifting_separate_stages = False
        # try:
        #     self.add_table = self.cfg['env']['add_table']
        # except:
        #     self.add_table = False

        self.add_table = True
        
        try:
            # self.table_z_dim = self.cfg['env']['table_z_dim']
            self.table_z_dim = 0.001
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
        
        ### TODO: add such stats fn; trained models ###### trained models ###
        ### NOTE: the hand tracking generalist task is used for the generalist tracking task ###
        self.grab_inst_tag_to_opt_stat_fn = self.cfg['env']['grab_inst_tag_to_opt_stat_fn']
        self.grab_inst_tag_to_optimized_res_fn = self.cfg['env']['grab_inst_tag_to_optimized_res_fn']
        self.taco_inst_tag_to_optimized_res_fn = self.cfg['env']['taco_inst_tag_to_optimized_res_fn']
        self.object_type_to_latent_feature_fn = self.cfg['env']['object_type_to_latent_feature_fn']
        self.inst_tag_to_latent_feature_fn = self.cfg['env'].get('inst_tag_to_latent_feature_fn', '')
        self.use_inst_latent_features = len(self.inst_tag_to_latent_feature_fn) > 0 
        # /cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy
        self.tracking_save_info_fn = self.cfg['env']['tracking_save_info_fn']
        self.single_instance_state_based_test = self.cfg['env'].get('single_instance_state_based_test', False)
        # load the samples with object code? #
        self.sampleds_with_object_code_fn = self.cfg['env'].get('sampleds_with_object_code_fn', '')
        
        self.grab_obj_type_to_opt_res_fn = self.cfg['env'].get('grab_obj_type_to_opt_res_fn', '')
        self.taco_obj_type_to_opt_res_fn = self.cfg['env'].get('taco_obj_type_to_opt_res_fn', '')
        
        self.only_training_on_succ_samples = self.cfg['env'].get('only_training_on_succ_samples', False)
        self.grab_train_test_setting = self.cfg['env'].get('grab_train_test_setting', False)
        
        self.maxx_inst_nn = self.cfg['env'].get('maxx_inst_nn', 10000)
        
        
        # tracking_save_info_fn, tracking_info_st_tag
        # self.tracking_info_st_tag = "passive_active_info_"
        # 
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
        # smoothness loss coefs #? 
        
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

        # _load_object_type_to_feature, object_type_to_latent_feature_fn #
        
        if len(self.grab_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.grab_inst_tag_to_optimized_res_fn):
            if len(self.taco_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.taco_inst_tag_to_optimized_res_fn):
                self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn, self.taco_inst_tag_to_optimized_res_fn] 
                # self.grab_inst_tag_to_optimized_res_fn = [self.taco_inst_tag_to_optimized_res_fn] 
                # get the grab and taco inst tagt to optimized res fn ## # isnt tag to optimized res fn #
            else:
                self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn]
        else:
            self.grab_inst_tag_to_optimized_res_fn = [self.taco_inst_tag_to_optimized_res_fn]
            
        # if len(self.obj_type_to_base_traj_fn) > 0 and os.path.exists(self.obj_type_to_base_traj_fn):
        #     self.obj_type_to_base_traj = np.load(self.obj_type_to_base_traj_fn, allow_pickle=True).item()
        # else:
        #     self.obj_type_to_base_traj =None
            
        self.obj_type_to_base_traj = None
        
        #  
        print(f"grab_inst_tag_to_optimized_res_fn: {self.grab_inst_tag_to_optimized_res_fn}")
        # pre trained object encoders #
        # load grab inst tag to opt res #
        
        # 
        # self.tot_grab_inst_tag_to_opt_res = {}
        # for cur_fn in self.grab_inst_tag_to_optimized_res_fn: # test grab inst tag to opt res #
        #     cur_opt_res = np.load(cur_fn, allow_pickle=True).item()
        #     self.tot_grab_inst_tag_to_opt_res.update(cur_opt_res)
        # self.grab_inst_tag_to_opt_res = self.tot_grab_inst_tag_to_opt_res
        
        
        # ### Load and use the obj type to opt res to filter out unsuccessful trajectories ###
        # if len(self.grab_obj_type_to_opt_res_fn) > 0 and os.path.exists(self.grab_obj_type_to_opt_res_fn):
        #     self.grab_obj_type_to_opt_res = np.load(self.grab_obj_type_to_opt_res_fn, allow_pickle=True).item()
        # else:
        #     self.grab_obj_type_to_opt_res = None
        self.grab_obj_type_to_opt_res = None
        
        
        # if len(self.taco_obj_type_to_opt_res_fn) > 0 and os.path.exists(self.taco_obj_type_to_opt_res_fn):
        #     self.taco_obj_type_to_opt_res = np.load(self.taco_obj_type_to_opt_res_fn, allow_pickle=True).item()
        # else:
        #     self.taco_obj_type_to_opt_res = None ## else we do no have the 
        self.taco_obj_type_to_opt_res = None ## else we do no have the 
            
        
        # print(f"Loaded inst_tag_to_optimized_res with number of total instances {len(self.grab_inst_tag_to_opt_res)}")
        # load the inst tag to optimized res #
        
        # opt state fn #
        # pure_inst_tag_to_opt_stat_fn = self.grab_inst_tag_to_opt_stat_fn.split("/")[-1]
        # local_inst_tag_to_opt_state_fn = os.path.join(f"./assets", pure_inst_tag_to_opt_stat_fn)
        # if os.path.exists(local_inst_tag_to_opt_state_fn): # local isnt tag and the glb inst tag #
        #     grab_inst_tag_to_opt_stat_fn = local_inst_tag_to_opt_state_fn
        # else: # grab inst tag to opt state #
        #     grab_inst_tag_to_opt_stat_fn = self.grab_inst_tag_to_opt_stat_fn # add grab isnt tag to optstat fn #
        

        # this is a single trajectory replay setting #
        ### add the replay trajectory fnhere ###
        self.replay_fn = self.cfg['env'].get('replay_fn', '')
        assert os.path.exists(self.replay_fn), f"Replay file {self.replay_fn} does not exist"
        # assert that the file exists #
        
        
        # self._load_replay_fn() # load replay fn # load replay fn ##
        # 
        
        self.datanm_to_replay_fn_dict_fn = self.cfg['env'].get('datanm_to_replay_fn_dict_fn', '')
        
        self._preload_all_replay_data(self.datanm_to_replay_fn_dict_fn)
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        # grab_inst_tag_to_opt_stat = np.load(grab_inst_tag_to_opt_stat_fn, allow_pickle=True).item()
        self.data_list  = []
        self.data_inst_tag_list = []
        # self.object_code_list = []
        self.object_rew_succ_dict = {}
        self.rew_succ_threshold = self.rew_grab_thres #  50.0
        self.taco_rew_succ_threshold = self.rew_taco_thres #  200.0
        
        self.data_base_traj = []
        
        # single  testing optimized res # 
        # single testing instance tag # 
        # single testing object type #
        # test_inst_tag, test_optimized_res #
        # preoptimized trajectories --- as the supervision 
        
        # if self.use_base_traj:
        #     if len(self.sampleds_with_object_code_fn) == 0:
        #         self.sampleds_with_object_code_fn = self.pre_optimized_traj
        #     print(f"sampleds_with_object_code_fn: {self.sampleds_with_object_code_fn}")
            
        # if len(self.base_traj) > 0 and os.path.exists(self.base_traj):
        #     self.sampleds_with_object_code_fn = self.base_traj
        
        # self.object_code_list = [ self.test_inst_tag ] 
            
        # if len(self.test_inst_tag) > 0 and len(self.test_optimized_res) > 0 and os.path.exists(self.test_optimized_res):
            
        #     # load the test inst tag, optimized res and the object code #
        #     # load single instance testing configs #
        #     # test inst tag # test optimized res #
        #     self.data_list = [  self.test_optimized_res ]
        #     self.data_inst_tag_list = [ self.test_inst_tag ]
        #     self.object_code_list = [ self.test_inst_tag ] 
        #     self.data_base_traj = [ None ]
            
        #     if self.test_inst_tag.endswith('.npy'):
        #         test_inst_dict = np.load(self.test_inst_tag, allow_pickle=True).item()
        #         test_inst_object_type = test_inst_dict['object_type']
        #         self.object_code_list = [ test_inst_object_type ]
            
            
        #     if len(self.sampleds_with_object_code_fn) > 0 and os.path.exists(self.sampleds_with_object_code_fn):
        #         self.samples_with_object_code = np.load(self.sampleds_with_object_code_fn, allow_pickle=True).item()
        #         if 'optimized_hand_qtars' in self.samples_with_object_code:
        #             self.tot_dof_targets = self.samples_with_object_code['optimized_hand_qtars']
        #             self.tot_dof_targets = self.tot_dof_targets[0]
        #             self.maxx_kine_nn_ts = min(self.maxx_kine_nn_ts, self.tot_dof_targets.shape[0])
        #             print(f"Loaded optimized hand qtars with shape: {self.tot_dof_targets.shape}")
        #         elif 'samples' in self.samples_with_object_code:
        #             samples = self.samples_with_object_code['samples']
        #             if 'data_nm' in samples:
        #                 samples_data_nm = samples['data_nm']
        #                 for idx, cur_data_nm in enumerate(samples_data_nm):
        #                     if cur_data_nm == self.test_inst_tag:
        #                         break
        #                 self.tot_dof_targets = samples['hand_qs'][idx]
        #             else:
        #                 self.tot_dof_targets = samples['hand_qs'][0]
                    
        #             self.maxx_kine_nn_ts = min(self.maxx_kine_nn_ts, self.tot_dof_targets.shape[0])
        #             print(f"Loaded optimized hand qtars with shape: {self.tot_dof_targets.shape}")
        #         elif 'env_object_codes' in self.samples_with_object_code:
        #             env_obj_codes = self.samples_with_object_code['env_object_codes']
        #             # pre_optimized_fr_tag = 'ori_grab_s9_waterbottle_pour_1'
        #             pre_optimized_fr_tag = self.test_inst_tag
        #             for i_env, cur_env_obj_code in enumerate(env_obj_codes):
        #                 if cur_env_obj_code == pre_optimized_fr_tag:
        #                     # self.test_obj_code = self.sampleds_with_object_code['object_codes'][i_env]
        #                     break
        #             # print(f"Loading from sampled trajectories for the instance: {cur_env_obj_code}")
        #             tot_ts_idxes = [ key for key in self.samples_with_object_code if isinstance(key, int) ]
        #             tot_ts_idxes = sorted(tot_ts_idxes) # the sorted indexes for timesteps #
        #             tot_dof_targets = []
        #             for i_ts, cur_ts in enumerate(tot_ts_idxes):
        #                 cur_ts_stats = self.samples_with_object_code[cur_ts]
        #                 cur_ts_dof_tars = cur_ts_stats['shadow_hand_dof_tars']
        #                 cur_ts_dof_tars = cur_ts_dof_tars[i_env]
        #                 tot_dof_targets.append( cur_ts_dof_tars )
        #             tot_dof_targets = np.stack(tot_dof_targets, axis=0) # nn_ts x nn_evs x nn_han-dof_
        #             self.tot_dof_targets = tot_dof_targets
        #         # elif 'optimized_hand_qtars' in self.samples_with_object_code:
        #         #     self.tot_dof_targets = self.samples_with_object_code['optimized_hand_qtars']
        #         #     self.tot_dof_targets = self.tot_dof_targets[0] # nn_envs x nn_ts x nn_hand_dof_
        #         else:
        #             i_env = 0
        #         # gather the env's optimized res # 
        #         # self.ts_to_hand_obj_states[self.ref_ts] = {
        #         #     'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np,
        #         #     'shadow_hand_dof_tars': self.target_qpos_np,
        #         #     'object_pose': self.object_pose_np,
        #         #     'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np,
        #         #     'object_linvel': self.object_linvel_np,
        #         #     'object_angvel': self.object_angvel_np,
        #         #     'actions': self.actions_np , 
        #         #     'observations': self.obs_buf_np
        #         #     # actions and the hand obs #
        #         # }
        #         # shadow_hand_dof_tars 
        #         # tot_ts_idxes = [ key for key in self.samples_with_object_code if isinstance(key, int) ]
        #         # tot_ts_idxes = sorted(tot_ts_idxes) # the sorted indexes for timesteps #
        #         # tot_dof_targets = []
        #         # for i_ts, cur_ts in enumerate(tot_ts_idxes):
        #         #     cur_ts_stats = self.samples_with_object_code[cur_ts]
        #         #     cur_ts_dof_tars = cur_ts_stats['shadow_hand_dof_tars'] # cur ts dof tars --- shadow hand dof tars --- nn_envs x nn-timesteps x 22
        #         #     cur_ts_dof_tars = cur_ts_dof_tars[i_env]
        #         #     tot_dof_targets.append( cur_ts_dof_tars )  # 
        #         # tot_dof_targets = np.stack(tot_dof_targets, axis=0) # nn_ts x nn_evs x nn_han-dof_
        #         # self.tot_dof_targets = tot_dof_targets
        #         # n_ts x nn_hand_ dof # #
        #         # self.tot_dof_targets = torch.from_numpy(self.tot_dof_targets).float().to(self.rl_device)
        #     else:
        #         self.tot_dof_targets = None
            
        # else:
            
            
                
            
        #     self.maxx_obj_nn = 1 
        #     self.maxx_obj_nn = 1000
        #     self.maxx_obj_nn = 10000
        #     # self.maxx_obj_nn = 50
        #     self.maxx_obj_nn = self.maxx_inst_nn
        #     self.tot_dof_targets = None
        #     # self.maxx_obj_nn = 200
        #     # self.maxx_obj_nn = 100
        #     # self.maxx_obj_nn = 50
        #     for i_inst_grab, grab_inst_tag in enumerate(self.grab_inst_tag_to_opt_res):
                
                
        #         # if i_inst_grab == 0:
        #         #     continue
        #         # self.sampleds_with_object_code_fn #
        #         # print(f"grab_inst_tag: {grab_inst_tag}, val: {self.grab_inst_tag_to_opt_res[grab_inst_tag]}")
        #         if isinstance(grab_inst_tag, tuple): # 
        #             cur_grab_obj_type, cur_grab_traj_obj_type = grab_inst_tag
        #         else:
        #             cur_grab_obj_type = grab_inst_tag
                
                
        #         if self.only_training_on_succ_samples: # active #
                    
        #             if 'taco' in cur_grab_obj_type and self.taco_obj_type_to_opt_res is not None:
        #                 if cur_grab_obj_type not in self.taco_obj_type_to_opt_res:
        #                     continue
        #                 cur_inst_opt_res = self.taco_obj_type_to_opt_res[cur_grab_obj_type][0]
        #                 # if cur_inst_opt_res < self.rew_succ_threshold:
        #                 if self.obj_type_to_base_traj is None:
        #                     if cur_inst_opt_res < self.taco_rew_succ_threshold:
        #                         continue
        #                 else:
        #                     if grab_inst_tag not in self.obj_type_to_base_traj and cur_inst_opt_res < self.taco_rew_succ_threshold:
        #                         continue
        #             else:
        #                 if cur_grab_obj_type not in self.grab_obj_type_to_opt_res:
        #                     continue
        #                 cur_inst_opt_res = self.grab_obj_type_to_opt_res[cur_grab_obj_type][0] # with the succ threshold
        #                 if cur_inst_opt_res < self.rew_succ_threshold:   # with succ threshold 
        #                     continue
                    
        #             # if we only train on succ samples --- for the grab instances, filter out unsucc trajectories #

                
        #         if 'taco' in cur_grab_obj_type:
        #             cur_grab_traj_obj_type = cur_grab_obj_type
        #             print(f"cur_grab_obj_type: {cur_grab_obj_type}, grab_inst_tag: {grab_inst_tag}")
                    
                
                
        #         # print(f"cur_grab_obj_type: {cur_grab_obj_type}")
        #         if 'ori_grab' in cur_grab_obj_type:  #
        #             # if cur_grab_obj_type not in grab_inst_tag_to_opt_stat:
        #             #     continue
        #             if self.test_subj_nm is not None and len(self.test_subj_nm) > 0:
        #                 if self.test_subj_nm not in cur_grab_obj_type:
        #                     continue
                        
        #             if self.grab_train_test_setting:
        #                 if '_s1_' in cur_grab_obj_type: # use the s1 as the test split
        #                     continue
                
        #         pure_cur_grab_obj_type = cur_grab_obj_type.split("_nf_")[0]
                
        #         # Generalist # inst opt fns #
        #         cur_inst_opt_fns = self.grab_inst_tag_to_opt_res[grab_inst_tag]
                
        #         if isinstance(cur_inst_opt_fns, tuple):
        #             cur_inst_opt_fns = [cur_inst_opt_fns[1]]
                    
        #         if self.obj_type_to_base_traj is not None:
        #             if grab_inst_tag in self.obj_type_to_base_traj:
        #                 print(f"grab_inst_tag: {grab_inst_tag}, val: {self.obj_type_to_base_traj[grab_inst_tag]}")
        #                 cur_obj_base_traj = self.obj_type_to_base_traj[grab_inst_tag][0]
        #             else:
        #                 cur_obj_base_traj = None
        #         else:
        #             cur_obj_base_traj = None
        #         self.data_base_traj.append(cur_obj_base_traj)
                
        #         for i_inst, cur_inst_fn in enumerate(cur_inst_opt_fns):
        #             cur_inst_sorted_val_fn = cur_inst_fn.replace(".npy", "_sorted.npy")
        #             cur_inst_sorted_val_fn_best = cur_inst_sorted_val_fn.replace(".npy", "_best.npy")
        #             if os.path.exists(cur_inst_sorted_val_fn_best):
        #                 cur_inst_sorted_val_fn = cur_inst_sorted_val_fn_best
        #             # cur_full_sorted_val_fn = os.path.join(data_folder, cur_inst_sorted_val_fn)
        #             self.data_list.append(cur_inst_sorted_val_fn)
        #             self.data_inst_tag_list.append(grab_inst_tag)
        #             self.object_code_list.append(pure_cur_grab_obj_type)
                    
        #             if self.only_training_on_succ_samples:
        #                 self.object_rew_succ_dict[cur_grab_obj_type] = 1 
        #             else: # # #
        #                 if 'taco' in cur_grab_obj_type:
        #                     if self.taco_obj_type_to_opt_res is not None:
        #                         if cur_grab_obj_type in self.taco_obj_type_to_opt_res:
        #                             cur_obj_opt_res = self.taco_obj_type_to_opt_res[cur_grab_obj_type][0]
        #                             if cur_obj_opt_res >= self.rew_succ_threshold:
        #                                 self.object_rew_succ_dict[cur_grab_obj_type] = 1
        #                             else:
        #                                 self.object_rew_succ_dict[cur_grab_obj_type] = 0
        #                         else:
        #                             self.object_rew_succ_dict[cur_grab_obj_type] = 0
        #                 else: # grab obj type to opt res #
        #                     if self.grab_obj_type_to_opt_res is not None:
                                
        #                         # if 'taco' in cur_grab_obj_type and self.taco_obj_type_to_opt_res is not None:
                                
        #                         if cur_grab_obj_type in self.grab_obj_type_to_opt_res:
        #                             cur_obj_opt_res = self.grab_obj_type_to_opt_res[cur_grab_obj_type][0]
        #                             if cur_obj_opt_res >= self.rew_succ_threshold:
        #                                 # self.object_rew_succ_list.append(1)
        #                                 self.object_rew_succ_dict[cur_grab_obj_type] = 1
        #                                 print(f"cur_grab_obj_type: {cur_grab_obj_type}, cur_obj_opt_res: {cur_obj_opt_res}")
        #                             else:
        #                                 # self.object_rew_succ_list.append(0)
        #                                 self.object_rew_succ_dict[cur_grab_obj_type] = 0
        #                         else:
        #                             # self.object_rew_succ_list.append(0)
        #                             self.object_rew_succ_dict[cur_grab_obj_type] = 0
        #             # object_code_list #
        #         if  i_inst_grab >= self.maxx_obj_nn:
        #             break
        
        self.tot_obj_codes = self.object_code_list
        self.data_name_to_data = {}
        self.data_name_to_object_code = {}
        self.data_name_to_kine_info = {}
        
        # self.tracking_info_st_tag = "passive_active_info_"
        
        self.maxx_trajectory_length = 0
        # self._preload_mocap_tracking_ctl_data() 
        # self._load_tracking_kine_info()
        # self._load_object_type_to_feature()
        # self._prepare_expert_traj_infos()
        # if self.use_inst_latent_features:
            # self._load_inst_tag_to_features() # load inst tag to optimized features #
        ## TODO: add a pre-trained point cloud encoder; ##
        ## TODO: add object point cloud features from that ##
        #### NOTE: Load data lis tand data instance tag list ####
        
        
        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        self.control_freq_inv = control_freq_inv
        if self.reset_time > 0.0: # 
            self.max_episode_length = int(round(self.reset_time / (control_freq_inv * self.sim_params.dt)))
            print("Reset time: ", self.reset_time)
            print("New episode length: ", self.max_episode_length)
        self.obs_type = self.cfg["env"]["observationType"]
        print("Obs type:", self.obs_type)
        print(f"controlFrequencyInv: {self.control_freq_inv}")

        self.nn_hand_dof = 22
        
        # 
        # self.shadow_hand_dof_speed_scale_list = [1.0] * 6 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        self.shadow_hand_dof_speed_scale_list = [self.glb_trans_vel_scale] * 3 + [self.glb_rot_vel_scale] * 3 + [self.shadow_hand_dof_speed_scale] * (22 - 6)
        self.shadow_hand_dof_speed_scale_tsr = torch.tensor(self.shadow_hand_dof_speed_scale_list, device=self.rl_device, dtype=torch.float) # #
        
        
        self.up_axis = 'z'
        
        if self.use_fingertips: 
            if self.hand_type == 'allegro':
                self.fingertips = ["link_15_tip", "link_3_tip", "link_7_tip", "link_11_tip"]
            elif self.hand_type == 'leap':
                self.fingertips = ["thumb_fingertip", "fingertip", "fingertip_2", "fingertip_3"]
        else:
            self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        self.hand_center = ["palm_link"] #  
        self.num_fingertips = len(self.fingertips) 
        
        
        # self.mocap_sv_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_apple_lift.npy'
        # self._load_mocap_info()
        
        # self.max_episode_length = min(self.max_episode_length, self.hand_qs.shape[0] - 1)
        # self.max_episode_length = min(self.max_episode_length, self.optimized_hand_qs.size(0) - 1)
        # tot_maxx_kine_nn
        self.max_episode_length = min(self.tot_maxx_kine_nn, self.max_episode_length)
        
        
        
        # 13 
        # self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        #  + 6 + nn_dof (action) + 
        # 16 (obj) + 7 + nn_dof (goal) + 64
        self.num_hand_obs = 66 + 76 + 22 + 6  # 191 =  22*3 + (65+30) + 24
        
        # 2 * 22 + 13 * 4 + 6 + 22 + 16 + 7 + 22
        num_pure_obs = 2 * self.nn_hand_dof + 13 * self.num_fingertips + 6 + self.nn_hand_dof + 16 + 7 + self.nn_hand_dof # 169
        
        num_obs = self.num_hand_obs + 16 + 7 + self.nn_hand_dof + 64 #  236 + 64
        self.num_obs_dict = {
            "full_state": num_obs,
            "full_state_nforce": num_obs + 300 - num_obs, #  num_obs - self.nn_hand_dof - 24 # 24 -- fingertip forces
            "pure_state": num_pure_obs, # number obs - self.nnhanddofs #
            "pure_state_wref": num_pure_obs + self.nn_hand_dof,
            "pure_state_wref_wdelta": num_pure_obs + self.nn_hand_dof + self.nn_hand_dof
        }   
        # decide the observation type and size #
        # num_obs_dict #
        
        
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        
        if not self.single_instance_state_based_test:
            self.cfg['env']['numObservations'] += self.object_feat_dim 
            
        if self.use_inst_latent_features:
            self.cfg['env']['numObservations'] += self.object_feat_dim 
        
        
        if self.tight_obs: # with next state references and with the current delta targets #
            self.cfg['env']['numObservations'] -= 7
        
        print(f"obs_type: {self.obs_type}, num_obs: {self.cfg['env']['numObservations']}")
        
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = self.nn_hand_dof #  24 
        # self.cfg["device_type"] = device_type # # 
        # self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless
        
        if not self.single_instance_state_based_test and self.supervised_training:
            self.cfg['env']['numObservationsWActions'] = self.cfg['env']['numObservations'] + self.cfg['env']['numActions']
            
            if self.grab_obj_type_to_opt_res is not None:
                self.cfg['env']['numObservationsWActions'] += 1
            
            print(f"numObservationsWActions: {self.cfg['env']['numObservationsWActions']}, numActions: {self.cfg['env']['numActions']}, numObservations: {self.cfg['env']['numObservations']}")
        
        # super().__init__(cfg=self.cfg, enable_camera_sensors=False)
        
        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        print(f"num_shadow_hand_dofs: {self.num_shadow_hand_dofs}")
        
        if self.viewer != None:
            # cam_pos = gymapi.Vec3(10.0, 5.0, 1.0)
            # cam_target = gymapi.Vec3(6.0, 5.0, 0.0)

            # cam_pos = gymapi.Vec3(10.0, 5.9, 1.0)
            cam_pos = gymapi.Vec3(10.0, 5.2, 1.0)
            cam_pos = gymapi.Vec3(9.7, 5.2, 1.0)
            cam_target = gymapi.Vec3(6.0, 7.9, 0.0)

            cam_pos = gymapi.Vec3(9.7, 5.2, 0.5)
            cam_target = gymapi.Vec3(6.0, 7.9, -0.5)
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

        # only train the sucessful trajectories #
        # use successful trajectories to provide rewards #
        # add rewards #
        # use succesu trajectories to provide rewars # --- use that ad the distillation process #
        # do not make things complex and just try to use the expert mimicing rewards at first --- #
        # to see whether we can distill the results into a single policy successfully #
        # 
        # create some wrapper tensors for different slices # add rewards #
        # self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        
        # dof_state_tensor[:, : self.num_shadow_hand_dofs, 0] = self.shadow_hand_default_dof_pos
        
        ## debug the hand pose ## 
        # is is the same as we expected? #
        print(f"[Debug] shadow_hand_default_dof_pos: {self.shadow_hand_default_dof_pos}")
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
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
        self.prev_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_delta_targets = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.prev_dof_vel = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.cur_dof_vel = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self.actions = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
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
        
        # hand_palm_fingers_obj_contact_buf, right_hand_dist_buf
        self.hand_palm_fingers_obj_contact_buf = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.right_hand_dist_buf = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)    
        
        self.reach_lifting_stage = torch.zeros((self.num_envs), dtype=torch.float, device=self.device) # all zeros for the reach ifting stages #
        # a
        
        self.total_successes = 0
        self.total_resets = 0
        
        self.ts_to_hand_obj_states = {}
        
        self.ref_ts =  0
        self.reset_nn = 0
        
    # _load_object_type_to_feature, object_type_to_latent_feature_fn
    def _load_object_type_to_feature(self, ):
        # object_type_to_latent_feature_fn = self.object_type_to_latent_feature_fn
        self.object_type_to_latent_feature = np.load(self.object_type_to_latent_feature_fn, allow_pickle=True).item()
        # print(f"object_type_to_latent_feature: {self.object_type_to_latent_feature.keys()}")
    
    def _load_inst_tag_to_features(self, ):
        self.inst_tag_to_latent_features = np.load(self.inst_tag_to_latent_feature_fn, allow_pickle=True).item() # 
        # use_inst_latent_features # 
    

    def _load_replay_fn(self, ):
        # if optimized in the file v.s. if optimized not in the file 
        self.replay_data = np.load(self.replay_fn, allow_pickle=True).item()
        
        
        if 'robot_delta_states_weights_np' in self.replay_data:
            hand_qs_key = 'robot_delta_states_weights_np'
            obj_pos_key = 'object_transl'
            obj_ornt_key = 'object_rot_quat'
            hand_qs = self.replay_data[hand_qs_key]
            obj_pos = self.replay_data[obj_pos_key]
            obj_ornt = self.replay_data[obj_ornt_key]
            
            joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
            joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
            
            joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
            print(f"joint_idxes_ordering: {joint_idxes_ordering}")
            print(f"joint_idxes_inversed_ordering: {joint_idxes_inversed_ordering}")
            
            hand_qs  = hand_qs[..., joint_idxes_inversed_ordering]
            self.optimized_hand_qs = torch.from_numpy(hand_qs).float().to(self.rl_device)
            obj_pose = np.concatenate([obj_pos, obj_ornt], axis=-1)
            self.optimized_obj_pose = torch.from_numpy(obj_pose).float().to(self.rl_device)
        else:
            # load the replay fn # select some trajs out form the res #
            hand_qs_key = 'optimized_hand_qs'
            obj_pose_key = 'optimized_obj_pose'
            # hand_qs_key #
            optimized_hand_qs = self.replay_data[hand_qs_key][0]
            optimized_obj_pose = self.replay_data[obj_pose_key][0]
            # hand qs key #
            self.optimized_hand_qs = torch.from_numpy(optimized_hand_qs).float().to(self.rl_device)
            self.optimized_obj_pose = torch.from_numpy(optimized_obj_pose).float().to(self.rl_device)

            
        if self.add_table:
            if self.hand_type == 'leap':
                self.optimized_hand_qs[..., 0] += self.table_z_dim
            else:
                self.optimized_hand_qs[..., 2] += self.table_z_dim
            self.optimized_obj_pose[..., 2] += self.table_z_dim
            # self.optimized_obj_pos[..., 2] += self.table_z_dim
        
        


        self.optimized_obj_pos = self.optimized_obj_pose[..., :3]
        self.optimized_obj_ornt = self.optimized_obj_pose[..., 3:]
        
        # cur_inst_hand_qs = self.tot_kine_qs[i_inst]
        # cur_inst_goal_obj_trans = self.tot_kine_obj_trans[i_inst]
        # cur_inst_goal_obj_quat = self.tot_kine_obj_ornt[i_inst] 
        
        self.tot_kine_qs = self.optimized_hand_qs.clone()
        self.tot_kine_obj_trans = self.optimized_obj_pose[..., :3].clone()
        self.tot_kine_obj_ornt = self.optimized_obj_pose[..., 3:].clone()
        
    
    
    def _preload_all_replay_data(self, data_fn):
        self.replay_data_dict  = np.load(data_fn, allow_pickle=True).item()
        tot_traj_kine_qs = []
        tot_traj_obj_trans = []
        tot_traj_obj_ornt = []
        self.object_code_list = []
        # optimized ahnd qs # 
        # replay the single ones # 
        hand_qs_key = 'optimized_hand_qs'
        obj_pose_key = 'optimized_obj_pose'
        tot_traj_maxx_kine_nn = [] 
        # env_maxx_progress_buf, tot_traj_maxx_kine_nn
        for data_inst_tag in self.replay_data_dict:
            cur_inst_replay_fn = self.replay_data_dict[data_inst_tag]
            # /root/diffsim/IsaacGymEnvs2
            cur_inst_replay_fn = cur_inst_replay_fn.replace("/root/diffsim/IsaacGymEnvs2", "..")
            # IsaacGymEnvs2/assets/optimized_res_taco_400_demo
            cur_inst_replay_fn = cur_inst_replay_fn.replace("/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_", "../assets/optimized_res_taco_400_demo")
            if not os.path.exists(cur_inst_replay_fn):
                cur_inst_replay_fn = cur_inst_replay_fn.replace("_best.npy", "_best_vv.npy")
            cur_inst_replay_data = np.load(cur_inst_replay_fn, allow_pickle=True).item()
            cur_inst_hand_qs = cur_inst_replay_data[hand_qs_key][0]
            cur_inst_obj_pose = cur_inst_replay_data[obj_pose_key][0]

            
            tot_traj_maxx_kine_nn.append(min(cur_inst_hand_qs.shape[0], cur_inst_obj_pose.shape[0]))
            
            # 400 # maxx length # 
            maxx_length = 400
            if cur_inst_hand_qs.shape[0] < maxx_length:
                cur_inst_hand_qs = np.concatenate(
                    [cur_inst_hand_qs, np.zeros((maxx_length - cur_inst_hand_qs.shape[0], cur_inst_hand_qs.shape[1]), dtype=np.float32)], axis=0
                )
            else:
                cur_inst_hand_qs = cur_inst_hand_qs[: maxx_length]

            if cur_inst_obj_pose.shape[0] < maxx_length:
                cur_inst_obj_pose = np.concatenate(
                    [cur_inst_obj_pose, np.zeros((maxx_length - cur_inst_obj_pose.shape[0], cur_inst_obj_pose.shape[1]), dtype=np.float32)], axis=0
                )
            else:
                cur_inst_obj_pose = cur_inst_obj_pose[: maxx_length]


            cur_inst_obj_pos, cur_inst_obj_ornt = cur_inst_obj_pose[..., :3], cur_inst_obj_pose[..., 3:]

            # cur_inst_hand_qs[..., 2] = cur_inst_hand_qs[..., 2] + self.table_z_dim
            # cur_inst_obj_pose[..., 2] = cur_inst_obj_pose[..., 2] + self.table_z_dim

            

            tot_traj_kine_qs.append(cur_inst_hand_qs)
            tot_traj_obj_trans.append(cur_inst_obj_pos)
            tot_traj_obj_ornt.append(cur_inst_obj_ornt)
            
            self.object_code_list.append(data_inst_tag)
        tot_traj_obj_trans = np.stack(tot_traj_obj_trans, axis=0)
        tot_traj_kine_qs = np.stack(tot_traj_kine_qs, axis=0)
        tot_traj_obj_ornt = np.stack(tot_traj_obj_ornt, axis=0)
        
        self.tot_traj_obj_trans = tot_traj_obj_trans
        self.tot_traj_kine_qs = tot_traj_kine_qs
        self.tot_traj_obj_ornt = tot_traj_obj_ornt
        self.tot_traj_maxx_kine_nn = np.array(tot_traj_maxx_kine_nn, dtype=np.int32) # 
        
        self.tot_traj_obj_trans = torch.from_numpy(self.tot_traj_obj_trans).float().to(self.rl_device)
        self.tot_traj_obj_ornt = torch.from_numpy(self.tot_traj_obj_ornt).float().to(self.rl_device)
        self.tot_traj_kine_qs = torch.from_numpy(self.tot_traj_kine_qs).float().to(self.rl_device)
        self.tot_traj_maxx_kine_nn = torch.from_numpy(self.tot_traj_maxx_kine_nn).long().to(self.rl_device)

        self.tot_kine_qs = self.tot_traj_kine_qs.clone()
        self.tot_kine_obj_trans = self.tot_traj_obj_trans.clone()
        self.tot_kine_obj_ornt = self.tot_traj_obj_ornt.clone()

        self.tot_maxx_kine_nn = torch.max(self.tot_traj_maxx_kine_nn).item()

        
        
        
    # preload # which ca nlaod multiple trackingcontorl dat a#
    # load single tracking ctl data #
    def _preload_single_tracking_ctl_data(self, data_fn, add_to_dict=True):
        
        # print(f"loading from {data_fn}")
        cur_data = np.load(data_fn, allow_pickle=True).item()
        # if self.use_jointspace_seq:
        # if self.sim_platform == 'isaac':
        optimized_obj_pose = cur_data['optimized_obj_pose']
        optimized_hand_qtars = cur_data['optimized_hand_qtars']
        optimized_hand_qs = cur_data['optimized_hand_qs']
        # # TODO: use top-k other than using the best evaluated res? 
        
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
            self.data_name_to_data[data_fn] = cur_clip_data
        return cur_clip_data, hand_qs_np, hand_qtars_np
    
       
    
    
    def _preload_mocap_tracking_ctl_data(self,):  
        print(f"Entering func _preload_mocap_tracking_ctl_data")
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
            
            excluded = False 
            for cur_forbid_inst_tag in forbid_data_inst_tags:
                if cur_forbid_inst_tag in data_fn and 'taco' not in data_fn:
                    excluded = True
                    break
            if excluded:
                continue
            
            # load mocap tracking ctl #
            # print(f"loading from {data_fn}") # display good results? #
            # load tracking single ctl data # preload single tracking ctl data #
            cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(data_fn)
            tot_data_hand_qs.append(hand_qs_np)
            tot_data_hand_qtars.append(hand_qtars_np)
            self.data_name_to_data[data_fn] = cur_clip_data
            self.data_name_to_object_code[data_fn] = self.object_code_list[i_data_inst]
            
            cur_traj_maxx_length = max(hand_qs_np.shape[0], hand_qtars_np.shape[0])
            self.maxx_trajectory_length  = max(self.maxx_trajectory_length, cur_traj_maxx_length)
            
            if 'taco' in data_fn:
                print(f'Loading from {data_fn}, cur_traj_maxx_length: {cur_traj_maxx_length}, maxx_trajectory_length: {self.maxx_trajectory_length}')
        print(f"Existing func _preload_mocap_tracking_ctl_data")
     
    def _load_single_tracking_kine_info(self, data_inst_tag, cur_base_traj_fn=None):
        
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
                        traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2.npy'
                    else:
                        traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                    taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                    kine_info_fn = os.path.join(taco_kine_sv_root, traj_kine_info)
                else:
                    kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
                    kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
                # get he kinemati info file #
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
            
                traj_kine_info  = f"{self.tracking_info_st_tag}{traj_obj_type}.npy"
                traj_kine_info = os.path.join(self.tracking_save_info_fn, traj_kine_info)
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
                hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
                maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
                hand_qs = hand_qs[:maxx_ws]
                
                obj_trans = traj_kine_info_data['object_transl'][:maxx_ws] # nn_ts x 3 
                obj_ornt = traj_kine_info_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
                # then segment the data_inst_tag to get the mesh file name #
                self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
                
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
                if len(self.taco_interped_data_sv_additional_tag) == 0:
                    traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2.npy'
                else:
                    traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_v2_{self.taco_interped_data_sv_additional_tag}.npy'
                taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data' # TACO tracking pk # 
                traj_kine_info = os.path.join(taco_kine_sv_root, traj_kine_info) # get kinematics sv root # kinematics data #
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
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
        kine_info_dict = { # 
            'obj_verts': obj_verts, 
            'hand_qs': hand_qs[..., self.joint_idxes_inversed_ordering],
            'base_traj_hand_qs': base_traj_hand_qs, # 
            'obj_trans': obj_trans,
            'obj_ornt': obj_ornt ,
            'obj_rot_euler': kine_obj_rot_euler_angles
        }
        return kine_info_dict
    
    
    
    def _load_tracking_kine_info(self, ):
        # self.maxx_kine_nn_ts = 300
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        
        # kine tracking info #
        #### iterate over all data instance tag and load the kinematics info ####
        for i_inst, data_inst_tag in enumerate(self.data_inst_tag_list):
            print(f"[Loading tracking kine info] {i_inst}/{len(self.data_inst_tag_list)}: {data_inst_tag}")
            cur_base_traj_fn = self.data_base_traj[i_inst]
            kine_info_dict = self._load_single_tracking_kine_info(data_inst_tag, cur_base_traj_fn)
            
            # kinematics info dictionary #
            self.data_name_to_kine_info[self.data_list[i_inst]] = kine_info_dict
            
            obj_trans, kine_obj_rot_euler_angles, hand_qs, obj_verts = kine_info_dict['obj_trans'], kine_info_dict['obj_rot_euler'], kine_info_dict['hand_qs'], kine_info_dict['obj_verts']

            tot_obj_transl.append(obj_trans)
            tot_obj_rot_euler.append(kine_obj_rot_euler_angles)
            tot_hand_qs.append(hand_qs)
            tot_object_verts.append(obj_verts)
        
        
        # 
        # tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        # tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        # tot_hand_qs = np.concatenate(tot_hand_qs, axis=0)
        # tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        # self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        # self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        # self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
        # self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
        # # avg hand qs and the std hand qs #
        # ## TODO: for the kinematics target data --- we should save them using a differnet name? #
        # # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        # # avg kine hand qs #
        # self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        
        
        # self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        # self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        
        # # avg obj verts and the kine hand qs and #
        # self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        # self.data_statistics['std_obj_verts'] = self.std_object_verts
        # self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        # self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        # self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        # self.data_statistics['std_obj_transl'] = self.std_obj_transl
        # self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        # self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
        
        # obj_verts = cur_kine_data['passive_meshes']
        # robot_hand_pts = cur_kine_data['ts_to_allegro']
        # robot_hand_qs = cur_kine_data['robot_delta_states_weights_np']
        # sv_dict = {
        #     'obj_verts': obj_verts, 
        #     'robot_hand_pts': robot_hand_pts, 
        #     'robot_hand_qs': robot_hand_qs
        # }
        # self.data_name_to_data[cur_kine_data_fn] = sv_dict # get the save dict #
        
        # # obj_verts: nn_ts x nn_pts x 3 #
        # # get he nn_ts and nnpts # 
        # expanded_obj_verts = obj_verts.reshape(obj_verts.shape[0] * obj_verts.shape[1], -1) # 
        
    def _prepare_expert_traj_infos(self,):
        
        
        # expert trajectories #
        tot_data_fns = self.data_name_to_data.keys()
        tot_data_fns = sorted(tot_data_fns)
        self.data_fn_to_data_index = {}
        self.maxx_episode_length_per_traj = []
        # kine_info_dict = {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3
        #     'obj_trans': obj_trans, # obj verts; obj trans;
        #     'obj_ornt': obj_ornt ,
        #     'obj_rot_euler': kine_obj_rot_euler_angles
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
        
        # data name to kine info #
        for i_inst, cur_data_fn in enumerate(tot_data_fns):
            self.data_fn_to_data_index[cur_data_fn] = i_inst # i_inst, cur_data_fn # cur data fn #
            cur_tracking_data = self.data_name_to_data[cur_data_fn]
            cur_kine_data = self.data_name_to_kine_info[cur_data_fn] # get the kine data # 
            # laod the trakcing results # 
            cur_hand_qs = cur_tracking_data['hand_qs']
            cur_hand_qtars = cur_tracking_data['hand_qtars']
            cur_kine_qs = cur_kine_data['hand_qs']
            cur_kine_obj_trans = cur_kine_data['obj_trans']
            cur_kine_obj_ornt = cur_kine_data['obj_ornt']
            cur_kine_obj_rot_euler = cur_kine_data['obj_rot_euler']
            cur_base_traj_hand_qs = cur_kine_data['base_traj_hand_qs']
            cur_data_episode_length = min( [ cur_hand_qs.shape[0], cur_hand_qtars.shape[0], cur_kine_qs.shape[0], cur_kine_obj_trans.shape[0], cur_kine_obj_ornt.shape[0], cur_kine_obj_rot_euler.shape[0] , cur_base_traj_hand_qs.shape[0] ])
            self.maxx_episode_length_per_traj.append(cur_data_episode_length - 1)
            # self.maxx_episode_length_per_traj.append(self.maxx_trajectory_length - 1)
            
            # tot_dof_targets
            
            if self.tot_dof_targets is not None:
                cur_hand_dof_targets = self.tot_dof_targets
                if cur_hand_dof_targets.shape[0] < self.maxx_trajectory_length:
                    cur_hand_dof_targets = np.concatenate(
                        [ cur_hand_dof_targets, np.zeros((self.maxx_trajectory_length - cur_hand_dof_targets.shape[0], cur_hand_dof_targets.shape[-1] ), dtype=np.float32) ], axis=0
                    )
                elif cur_hand_dof_targets.shape[0] > self.maxx_trajectory_length:
                    cur_hand_dof_targets = cur_hand_dof_targets[:self.maxx_trajectory_length]
                tot_hand_preopt_res.append(cur_hand_dof_targets)
            
            if cur_base_traj_hand_qs.shape[0] < self.maxx_trajectory_length:
                cur_base_traj_hand_qs = np.concatenate(
                    [ cur_base_traj_hand_qs, np.zeros((self.maxx_trajectory_length - cur_base_traj_hand_qs.shape[0], cur_base_traj_hand_qs.shape[-1]), dtype=np.float32) ], axis=0
                )
            elif cur_base_traj_hand_qs.shape[0] > self.maxx_trajectory_length:
                cur_base_traj_hand_qs = cur_base_traj_hand_qs[:self.maxx_trajectory_length]
                
                
            
            # tot_dof_targets #
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
                
            tot_hand_qs.append(cur_hand_qs)
            tot_hand_qtars.append(cur_hand_qtars)
            tot_kine_qs.append(cur_kine_qs)
            tot_kine_obj_trans.append(cur_kine_obj_trans)
            tot_kine_obj_ornt.append(cur_kine_obj_ornt)
            tot_kine_obj_rot_euler.append(cur_kine_obj_rot_euler)
            
            cur_obj_code = self.data_name_to_object_code[cur_data_fn]
            tot_obj_codes.append(cur_obj_code)
            tot_base_traj_hand_qs.append(cur_base_traj_hand_qs)
        
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
        
        print(f"tot_hand_qs: {self.tot_hand_qs.size()}, tot_hand_qtars: {self.tot_hand_qtars.size()}, tot_kine_qs: {self.tot_kine_qs.size()}, tot_kine_obj_trans: {self.tot_kine_obj_trans.size()}, tot_kine_obj_ornt: {self.tot_kine_obj_ornt.size()}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj}")
        
        
        if len(tot_hand_preopt_res) > 0: # tot hand preopt res; tot hand
            tot_hand_preopt_res = np.stack(tot_hand_preopt_res, axis=0)
            self.tot_hand_preopt_res = torch.from_numpy(tot_hand_preopt_res).float().to(self.rl_device)
            
        else:
            # self.tot_hand_preopt_res = self.tot_kine_qs.clone()
            self.tot_hand_preopt_res = self.tot_base_traj_hand_qs.clone()
        
        # 
        if len(tot_hand_actions) > 0: # tot hand actions #
            tot_hand_actions = np.stack(tot_hand_actions, axis=0)
            self.tot_hand_actions = torch.from_numpy(tot_hand_actions).float().to(self.rl_device)
        else:
            self.tot_hand_actions = None
        
        self.tot_obj_codes = tot_obj_codes
        pass
       # allegro hand tracking #



    
    # timesteps and the control frequences ? #
    # 120 HZ for the GRAB data # 
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
        
        # get the tasks and the taco #
        if 'taco' in self.object_name and 'TACO' not in self.mocap_sv_info_fn and self.use_taco_obj_traj:
            # then we need to repose all related trajectories and also interpolate between all related trajectories #
            obj_mocap_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{self.object_name}_zrot_3.141592653589793_modifed_interped.npy'
            obj_mocap_info = np.load(obj_mocap_info_fn , allow_pickle=True ).item()
            goal_obj_trans = obj_mocap_info['object_transl'][: ][20 : ]
            goal_obj_rot_quat = obj_mocap_info['object_rot_quat'][: ][20 : ]
            
            cur_ws = min(goal_obj_trans.shape[0], hand_qs.shape[0])
            hand_qs = hand_qs[: cur_ws]
            goal_obj_trans = goal_obj_trans[: cur_ws]
            goal_obj_rot_quat = goal_obj_rot_quat[: cur_ws]
        
        if self.dataset_type == 'taco':
            # link_name_to_poses # 
            link_name_to_poses = save_info['link_name_to_poses']
            self.link_name_to_poses = link_name_to_poses
            
            for link_name in self.link_name_to_poses: 
                self.link_name_to_poses[link_name][:, 2] -= self.ground_distance
                if self.add_table:
                    self.link_name_to_poses[link_name][:, 2] += self.table_z_dim
            
            self.link_name_to_poses_th = {}
            for link_name in self.link_name_to_poses:
                ##### get the link name to poses #####
                self.link_name_to_poses_th[link_name] = torch.from_numpy(self.link_name_to_poses[link_name]).float().to(self.rl_device)
            
            if self.hand_type == 'allegro':
                self.hand_palm_link_name = 'palm_link'
                self.thumb_tip_link_name = 'link_15_tip'
                self.index_tip_link_name = 'link_3_tip'
                self.middle_tip_link_name = 'link_7_tip'
                self.ring_tip_link_name = 'link_11_tip'
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
                # }
                # self.hand_palm_link_name = 'palm_lower'
                # self.thumb_tip_link_name = 'thumb_tip_head'
                # self.index_tip_link_name = 'index_tip_head'
                # self.middle_tip_link_name = 'middle_tip_head'
                # self.ring_tip_link_name = 'ring_tip_head'
                self.hand_palm_link_name = 'palm_lower'
                self.thumb_tip_link_name = 'thumb_fingertip'
                self.index_tip_link_name = 'fingertip'
                self.middle_tip_link_name = 'fingertip_2'
                self.ring_tip_link_name = 'fingertip_3'
            self.hand_palm_world_poses = self.link_name_to_poses_th[self.hand_palm_link_name]  
            self.thumb_tip_world_poses = self.link_name_to_poses_th[self.thumb_tip_link_name]
            self.index_tip_world_poses = self.link_name_to_poses_th[self.index_tip_link_name]
            self.middle_tip_world_poses = self.link_name_to_poses_th[self.middle_tip_link_name]
            self.ring_tip_world_poses = self.link_name_to_poses_th[self.ring_tip_link_name]
            
        
        
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



    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # plane_params.distance = self.ground_distance
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    
    def _create_envs(self, num_envs, spacing, num_per_row):
        object_scale_dict = self.cfg['env']['object_code_dict']
        
        # if len(self.object_name) > 0: # object name ##
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
        self.object_code_list = self.tot_obj_codes # obj codes #
        
        
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
        assets_path = '../../UniDexGrasp/dexgrasp_policy/assets'
        # assets_path = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets"
        
        # if not os.path.exists(assets_path):
        #     assets_path = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets"
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        # ## add object code ##
        # print(f"[Debug] object_code_list: {self.object_code_list}")
        # object shape inst list #
        
        for i_inst, object_code in enumerate(self.object_code_list):
            data_per_object = {}
            pure_object_code = object_code.split("_nf_")[0]
            dataset_path = dataset_root_path + '/sem/' + pure_object_code
            data_num_list = os.listdir(dataset_path)
            # 
            
            
            cur_inst_hand_qs = self.tot_kine_qs # [i_inst]
            cur_inst_goal_obj_trans = self.tot_kine_obj_trans # [i_inst]
            cur_inst_goal_obj_quat = self.tot_kine_obj_ornt # [i_inst] 
            
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
                init_obj_quat = self.obj_ornt_init.detach().cpu().numpy() # xyz 
                init_obj_tarns = self.obj_trans_init.detach().cpu().numpy() # obj trans init #
                
                obj_ornt_init_euler_xyz = R.from_quat(np.array([0,0,0,1])).as_euler('xyz', degrees=False)
                object_euler_xy = torch.tensor([obj_ornt_init_euler_xyz[0], obj_ornt_init_euler_xyz[1]], dtype=torch.float, device=self.device) # 
                object_init_z = torch.tensor([init_obj_tarns[2]], dtype=torch.float, device=self.device) # 
                
                
                

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
        
        # taret qpos # hand pos # target qpos # target pos #
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
        self.target_hand_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.target_hand_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.object_init_euler_xy = torch.zeros((self.num_envs, 2), device=self.device)
        self.object_init_z = torch.zeros((self.num_envs, 1), device=self.device)

        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        
        
        # asset_root = "../../assets"
        # asset_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc"
        asset_root = "../assets"
        # shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
        
        
        if self.hand_type == 'allegro':
            shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf"
        elif self.hand_type == 'leap':
            shadow_hand_asset_file = f"leap_hand/leap_hand_right_fly_v3.urdf"
        else:
            raise ValueError(f"Unknown hand type: {self.hand_type}")
        # if not os.path.exists(asset_root):
        #     asset_root = "/home/xueyi/diffsim/tiny-differentiable-simulator/python/examples/rsc"
            # shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd.urdf"
        
        
        # # shadow hand asset file # #
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
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        # asset_options.default_dof_drive_mode = gymapi.DOF_MODE_POS
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
        # set shadow_hand dof properties #
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

        # set shadow #
        shadow_hand_dof_names = self.gym.get_asset_dof_names(shadow_hand_asset)
        print(f"[Debug] shadow_hand_dof_names: {shadow_hand_dof_names}")
        # ['WRJ0x', 'WRJ0y', 'WRJ0z', 'WRJ0rx', 'WRJ0ry', 'WRJ0rz', 'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_12', 'joint_13', 'joint_14', 'joint_15', 'joint_4', 'joint_5', 'joint_6', 'joint_7', 'joint_8', 'joint_9', 'joint_10', 'joint_11']
        
        self.thumb_dof_idxes = [10, 11, 12, 13]
        self.thumb_dof_idxes = torch.tensor(self.thumb_dof_idxes, dtype=torch.long, device=self.device)
        
        
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
            #     self.shadow_hand_dof_lower_limits.append(0.0)
            # else: # self.table_z_dim --- #
            #     self.shadow_hand_dof_lower_limits.append(shadow_hand_dof_props['lower'][i])

            if i >= 6:
                shadow_hand_dof_props['velocity'][i] = 10.0
                shadow_hand_dof_props['effort'][i] = 0.7
                shadow_hand_dof_props['stiffness'][i] = 20
                shadow_hand_dof_props['damping'][i] = 1
            print(f"shadow_hand_dof_props: {shadow_hand_dof_props}")
        

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
        # object scale idx pairs #
        object_scale_idx_pairs = []
        visual_feat_root = osp.realpath(osp.join(assets_path, 'meshdatav3_pc_feat'))
        self.visual_feat_data = {}
        self.visual_feat_buf = torch.zeros((self.num_envs, 64), device=self.device)
        
        for object_id in range(len(self.object_code_list)):
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
            # object_asset_options.disable_gravity = self.disable_obj_gravity
            object_asset_options.disable_gravity = True
            object_asset_options.use_mesh_materials = True # mesh
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True 
            object_asset_options.override_inertia = True 
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            
            for obj_id, scale_id in object_scale_idx_pairs:
                # print(f"obj_id: {obj_id}, scale_id: {scale_id}, object_id: {object_id}") 
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    pure_object_code = object_code.split("_nf_")[0]
                    scaled_object_asset_file = 'sem/' + pure_object_code + f"/coacd/coacd_{scale_str}_vis.urdf"
                    print(f"scaled_object_asset_file: {scaled_object_asset_file}")
                    full_obj_asset_file = os.path.join(mesh_path, scaled_object_asset_file)
                    if not os.path.exists(full_obj_asset_file):
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

            # object dof lower limits # 
            # dof upper limits #
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
        
        # table #
        table_asset_options = gymapi.AssetOptions()
        table_asset_options.fix_base_link = True
        table_asset_options.flip_visual_attachments = True
        table_asset_options.collapse_fixed_joints = True # fixedjo
        table_asset_options.disable_gravity = True
        table_asset_options.thickness = 0.001
        table_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE

        #### set the table asset options ####
        # table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, gymapi.AssetOptions())
        table_asset = self.gym.create_box(self.sim, table_dims.x, table_dims.y, table_dims.z, table_asset_options)

        shadow_hand_start_pose = gymapi.Transform() # gymapi.Vec3(0.0, )
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # gymapi.Vec3(0.1, 0.1, 0.65)
        # shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0)

        object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # obj_init_x = self.obj_trans_init[0].item()
        # obj_init_y = self.obj_trans_init[1].item()
        # obj_init_z = self.obj_trans_init[2].item()
        # object_start_pose.p = gymapi.Vec3(obj_init_x, obj_init_y, obj_init_z)  # gymapi.Vec3(0.0, 0.0, 0.72)
        object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) 
        object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0) # from euler zyx #
        # pose_dx, pose_dy, pose_dz = -1.0, 0.0, -0.0
        # object_start_pose.r = gymapi.Quat(self.obj_ornt_init[0].item(), self.obj_ornt_init[1].item(), self.obj_ornt_init[2].item(), self.obj_ornt_init[3].item())
        
        

        self.goal_displacement = gymapi.Vec3(-0., 0.0, 0.2)
        self.goal_displacement_tensor = to_torch(
            [self.goal_displacement.x, self.goal_displacement.y, self.goal_displacement.z], device=self.device)
        goal_start_pose = gymapi.Transform()
        goal_start_pose.p = object_start_pose.p + self.goal_displacement
        # goal_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0)
        goal_start_pose.r = object_start_pose.r #  gymapi.Quat(self.obj_ornt_init[0].item(), self.obj_ornt_init[1].item(), self.obj_ornt_init[2].item(), self.obj_ornt_init[3].item())

        goal_start_pose.p.z -= 10.0 # goal start pose # # goal pose #

        table_pose = gymapi.Transform()
        ###### set table pose ######
        # by dfault
        # table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        
        # if 'transformed' in self.mocap_sv_info_fn:
        #     table_pose.p = gymapi.Vec3(0.5 * table_dims.x, -0.5 * table_dims.y, 0.5 * table_dims.z)
        # else:
        #     table_pose.p = gymapi.Vec3(-0.5 * table_dims.x, 0.5 * table_dims.y, 0.5 * table_dims.z)
        # # table_pose.p = gymapi.Vec3(0.0, 0.0, -0.5 * table_dims.z)
        # # table_pose.p = gymapi.Vec3(0.0, 0.0, -1.0 * table_dims.z)
        table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
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

        # finger tip handles #
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
                body_names = { # leap fingertips #
                    'palm': 'palm_lower',
                    'thumb': 'thumb_fingertip',
                    'index': 'fingertip',
                    'middle': 'fingertip_2',
                    'ring': 'fingertip_3',
                }
                # body_names = { # leap fingertips #
                #     'palm': 'palm_lower',
                #     'thumb': 'thumb_tip_head',
                #     'index': 'index_tip_head',
                #     'middle': 'middle_tip_head',
                #     'ring': 'ring_tip_head',
                # }
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

        # create fingertip force sensors, if needed
        if self.obs_type == "full_state" or self.asymmetric_obs: # 
            sensor_pose = gymapi.Transform()
            for ft_handle in self.fingertip_handles:
                self.gym.create_asset_force_sensor(shadow_hand_asset, ft_handle, sensor_pose)


        # add a table and # change the difficulty of the task? #
        self.object_scale_buf = {}
        self.tot_hand_dof_pos = []

        self.env_inst_idxes = []
        
        # self.env_object_latent_feat
        self.env_object_latent_feat = []
        self.env_object_codes = []
        self.env_inst_latent_feat = []
        self.env_rew_succ_list = []
        self.env_maxx_progress_buf = []
        # env_maxx_progress_buf, tot_traj_maxx_kine_nn
        # self.
        print(f"len(object_code_list): {len(self.object_code_list)}")
        for i in range(self.num_envs):
            
            data_inst_idx = i % len(self.object_code_list)
            
            cur_inst_maxx_kine_nn = self.tot_traj_maxx_kine_nn[data_inst_idx].item()
            self.env_maxx_progress_buf.append(cur_inst_maxx_kine_nn)

            cur_inst_hand_kine_qs = self.tot_kine_qs[data_inst_idx] # [data_inst_idx] 
            cur_inst_goal_obj_trans = self.tot_kine_obj_trans[data_inst_idx] # [data_inst_idx] 
            cur_inst_goal_obj_ornt = self.tot_kine_obj_ornt[data_inst_idx] # [data_inst_idx]

            # cur_inst_hand_kine_qs = self.optimized_hand_qs # [0] # goal hand qs 
            # cur_inst_goal_obj_pose = self.optimized_obj_pose # [0] # goal obj pose 
            
            cur_inst_goal_obj_pose = torch.cat(
                [cur_inst_goal_obj_trans, cur_inst_goal_obj_ornt], dim=-1
            )

            # 
            cur_inst_goal_obj_trans = cur_inst_goal_obj_pose[..., :3 ]
            cur_inst_goal_obj_ornt = cur_inst_goal_obj_pose[..., 3: ]


            first_frame_goal_obj_trans = cur_inst_goal_obj_trans[0, :]
            first_frame_goal_obj_ornt = cur_inst_goal_obj_ornt[0, :]
            
            first_frame_hand_kine_qs = cur_inst_hand_kine_qs[0, :]
            self.tot_hand_dof_pos.append(first_frame_hand_kine_qs) # first frame hand kine qs #
            
            self.env_inst_idxes.append(data_inst_idx)
            
            cur_object_code = self.object_code_list[data_inst_idx]
            pure_object_code = cur_object_code.split("_nf_")[0]
            # cur_object_latent_features = self.object_type_to_latent_feature[pure_object_code]
            # cur_object_latent_features 
            # self.env_object_latent_feat.append(cur_object_latent_features)
            
            # if self.grab_obj_type_to_opt_res is not None:
            #     cur_obj_succ = self.object_rew_succ_dict[cur_object_code]
            #     self.env_rew_succ_list.append(cur_obj_succ)
            
            # if self.use_inst_latent_features:
            #     cur_inst_tag = self.object_code_list[data_inst_idx]
            #     pure_cur_inst_tag = cur_inst_tag.split("_nf_")[0]
            #     cur_inst_latent_features = self.inst_tag_to_latent_features[pure_cur_inst_tag]
            #     self.env_inst_latent_feat.append(cur_inst_latent_features) # get the inst latent features
            
            self.env_object_codes.append(pure_object_code) # get the env object codes #
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1: # how does the agenet compute its gradient? #
                # print(f"Start the aggregation with mode {self.aggregate_mode}, obj_idx: {data_inst_idx}, obj_code: {cur_object_code}")
                # print(f"object_code_list: {self.object_code_list}")
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)
            # 
            # i + 2 * num_envs 0--- object #
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

            # create fingertip force-torque sensors #
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


            object_handle = self.gym.create_actor(env_ptr, object_asset_dict[id][scale_id], object_start_pose, "object", i + 2 * self.num_envs, 0, 0)
            
            
            self.object_init_state.append([object_start_pose.p.x, object_start_pose.p.y, object_start_pose.p.z,
                                           object_start_pose.r.x, object_start_pose.r.y, object_start_pose.r.z,
                                           object_start_pose.r.w,
                                           0, 0, 0, 0, 0, 0])
            self.goal_init_state.append([goal_start_pose.p.x, goal_start_pose.p.y, goal_start_pose.p.z,
                                         goal_start_pose.r.x, goal_start_pose.r.y, goal_start_pose.r.z,
                                         goal_start_pose.r.w,
                                         0, 0, 0, 0, 0, 0])
            # goal init state #
            # obj init state #
            object_idx = self.gym.get_actor_index(env_ptr, object_handle, gymapi.DOMAIN_SIM)
            self.object_indices.append(object_idx)
            self.gym.set_actor_scale(env_ptr, object_handle, 1.0)
            # set actor sacle #
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0) # set actor scale #

            

            #### NOTE: we have disabled table here ####
            if self.add_table: # add table --- 
                table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i + 3 * self.num_envs, -1, 0)
                self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle)
                table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
                self.table_indices.append(table_idx)


                table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
                table_shape_props[0].friction = 1
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            
            # object shape props # # obj properties #
            object_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, object_handle)
            object_shape_props[0].friction = 1
            self.gym.set_actor_rigid_shape_properties(env_ptr, object_handle, object_shape_props)

            

            object_color = [90/255, 94/255, 173/255]
            # self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*object_color))
            table_color = [150/255, 150/255, 150/255]
            # self.gym.set_rigid_body_color(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, gymapi.Vec3(*table_color))

            self.gym.set_rigid_body_color(env_ptr, object_handle, 0, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
            
            
            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            self.envs.append(env_ptr)
            self.shadow_hands.append(shadow_hand_actor)
            
        # tot hand dof pos #
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

        # # 
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
        # self.env_object_latent_feat = np.stack(self.env_object_latent_feat, axis=0) # nn_envs x nn_latent_feat_dim
        # self.env_object_latent_feat = to_torch(self.env_object_latent_feat, dtype=torch.float32, device=self.device) 
        
        self.env_maxx_progress_buf = torch.tensor(self.env_maxx_progress_buf, dtype=torch.long, device=self.device)
        # if len(self.env_rew_succ_list) > 0:
        #     self.env_rew_succ_list = np.array(self.env_rew_succ_list, dtype=np.float32)
        #     self.env_rew_succ_list = to_torch(self.env_rew_succ_list, dtype=torch.float32, device=self.device)
        
        # if self.use_inst_latent_features:
        #     self.env_inst_latent_feat = np.stack(self.env_inst_latent_feat, axis=0)
        #     self.env_inst_latent_feat = to_torch(self.env_inst_latent_feat, dtype=torch.float32, device=self.device)
        
        self.env_inst_idxes = np.array(self.env_inst_idxes, dtype=np.int32)
        self.env_inst_idxes = to_torch(self.env_inst_idxes, dtype=torch.long, device=self.device) # ge the env inst idxes #
        
        if self.add_table: # remember the table asset initial poses ? #
            self.table_indices = to_torch(self.table_indices, dtype=torch.long, device=self.device)

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
        
        # if self.dataset_type == 'grab':
        #     compute_reward_func = compute_hand_reward_tracking
        # elif self.dataset_type == 'taco':
        #     compute_reward_func = compute_hand_reward_tracking_taco
        # else:
        #     raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        
        # #
        
        # # lift_obj_pos_th # compute reward #
        
        # # self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos #
        
        # self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:], self.hand_palm_fingers_obj_contact_buf[:], self.right_hand_dist_buf[:] = compute_reward_func( # compute hand tracking reward ##
        #     self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
        #     self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
        #     self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
        #     self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
        #     self.object_linvel, self.object_angvel, self.object_linvel, self.object_angvel,
        #     self.lift_obj_pos_th,
        #     self.goal_pos, self.goal_rot, self.goal_lifting_pos,
        #     self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
        #     self.right_hand_th_pos, # 
        #     self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos ,
        #     self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
        #     self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
        #     self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.rew_obj_pose_coef, self.goal_dist_thres, self.lifting_separate_stages, self.reach_lifting_stage, self.strict_lifting_separate_stages
        # )
        
        # right_hand_dist_buf_buf #
        # ## set the strict_lifting_separate_stages to True ## #
        if self.lifting_separate_stages: # lifting the separate stages #
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
                self.reach_lifting_stage_flag = lowest >= lift_height_z # larger than the lift-heightj-z ##  # larger 
                if torch.sum(self.reach_lifting_stage_flag.float()) > 0.5:
                    cur_reached_lifting_stage_obj_pos = self.object_pos[self.reach_lifting_stage_flag]
                    avg_cur_obj_pos = cur_reached_lifting_stage_obj_pos.mean(dim=0)
                    # print(f"avg_cur_obj_pos: {avg_cur_obj_pos}")
                
                ## reach lifting stge ## reach lifting stage ##
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
        
        
        # # print(f"maxx_env_inst_idx: {torch.max(self.env_inst_idxes)}, tot_hand_qtars: {self.tot_hand_qtars.size()}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj.size()}")
        # envs_hand_qtars = batched_index_select(self.tot_hand_qtars, self.env_inst_idxes, dim=0)
        # # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_hand_qtars: {envs_hand_qtars.size()}")
        # envs_hand_qtars = batched_index_select(envs_hand_qtars, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        
        # # print(f"env_inst_idxes: {torch.max(self.env_inst_idxes)}, maxx_episode_length_per_traj: {self.maxx_episode_length_per_traj.size()}, max_episode_length: {self.maxx_episode_length_per_traj}")
        # envs_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        
        
        
        # # if self.dataset_type == 'grab':
        # compute_reward_func = compute_hand_reward_tracking
        # # elif self.dataset_type == 'taco':
        # #     compute_reward_func = compute_hand_reward_tracking_taco
        # # else:
        # #     raise ValueError(f"Invalid dataset type: {self.dataset_type}")
        
        # # s
        
        # self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_reward_func( # compute hand tracking reward ##
        #     self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
        #     self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
        #     self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
        #     envs_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
        #     self.object_linvel, self.object_angvel,self.object_linvel, self.object_angvel,
        #     self.goal_pos, self.goal_rot, self.goal_lifting_pos,
        #     self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
        #     self.right_hand_th_pos, # 
        #     self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
        #     self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
        #     self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.rew_obj_pose_coef, self.goal_dist_thres, envs_hand_qtars, self.cur_targets, self.use_hand_actions_rew, self.prev_dof_vel, self.cur_dof_vel, self.rew_smoothness_coef
        # )

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
        # compute observations #
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

    def compute_observations_bak(self):
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
            
            
            
        
        # maxx_progress = torch.max(self.progress_buf)
        # minn_progress = torch.min(self.progress_buf)
        # print(f"maxx_progress: {maxx_progress}, minn_progress: {minn_progress}, goal_obj_trans_th: {self.goal_obj_trans_th.size()}")
        
        # goal obj trans # 
        
        # 
        # env_inst_idxes #
        tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
        
        # values, indices, dims #
        # cur_dof_vel #

        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) # nn_envs x 1
        # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}")
        cur_progress_buf = torch.clamp(self.progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
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
        self.delta_target_hand_pos = torch.zeros((3,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.rel_hand_rot = torch.zeros((4,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.delta_target_hand_rot = torch.zeros((4,), dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        ### HACK ###
        
        
        tot_goal_obj_trans_th = self.tot_kine_obj_trans # nn_tot_envs x maximum_episode_length x 3
        tot_goal_obj_ornt_th = self.tot_kine_obj_ornt # nn_tot_envs x maximum_episode_length x 4
        
        # values, indices, dims #
        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
    
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        # tot_goal_hand_qs_th = self.tot_kine_qs
        tot_goal_hand_qs_th = self.tot_hand_preopt_res
        envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
        # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #

        ### current target hand pose, and the difference from the reference hand pos ###
        # cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        
        
        self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        
        
        ### next progress buffer ###
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.hand_qs_th.size(0) - 1)
        # nex_hand_qpos_ref = self.hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        # self.nex_hand_qpos_ref = nex_hand_qpos_ref
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.maxx_kine_nn_ts - 1)
        nex_progress_buf = torch.clamp(self.progress_buf + 1, min=torch.zeros_like(envs_maxx_episode_length_per_traj), max=envs_maxx_episode_length_per_traj)
        # print(f"nex_progress_buf: {torch.max(nex_progress_buf)}, envs_goal_hand_qs: {envs_goal_hand_qs.size()}")
        # nex_hand_qpos_ref = self.goal_hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        self.nex_hand_qpos_ref = nex_hand_qpos_ref
        
        # 
        if self.use_twostage_rew: # two stage reward #
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
            # object pose np ## -- curretn step observations; # # 
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
            # so that the obs include all obs buffer ? #
            # save observations, actions, angvel, linvel and other things #
            # then load hand dof pos and dof tars; dof tars #
            self.ts_to_hand_obj_states[self.ref_ts] = {
                'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np,
                'shadow_hand_dof_tars': self.target_qpos_np,
                'object_pose': self.object_pose_np,
                'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np,
                'object_linvel': self.object_linvel_np,
                'object_angvel': self.object_angvel_np,
                'actions': self.actions_np , 
                'observations': self.obs_buf_np
                # actions and the hand obs #
            }
            # self.ts_to_hand_obj_states[self.ref_ts]
        
        
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

    def compute_full_state(self, asymm_obs=False): #
        # hand dof pos, hand dof velocities, fingertip states, right hand pos, right hand rot, current actions, object states, next qpos ref, current delta targets
        self.get_unpose_quat()

        # 2 * nn_hand_dofs + 13 * num_fingertips + 6 + nn_hand_dofs + 16 + 7 + nn_hand_dofs ## 
        # unscale to (-1，1) # 
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##
        
        if self.use_local_canonical_state:
            # 
            # print(f"using local canonicalizations")
            canon_shadow_hand_dof_trans = self.shadow_hand_dof_pos[..., :3] - self.object_pos[..., :]
            canon_shadow_hand_dof = torch.cat(
                [ canon_shadow_hand_dof_trans, self.shadow_hand_dof_pos[..., 3:] ], dim=-1 
            )
        else:
            canon_shadow_hand_dof = self.shadow_hand_dof_pos     # 
        
        
        self.cur_dof_vel[:, : ] = self.shadow_hand_dof_vel

        # # 0:66
        # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
        #                                                        self.shadow_hand_dof_lower_limits,
        #                                                        self.shadow_hand_dof_upper_limits)
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(canon_shadow_hand_dof,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        
        if self.obs_type == "full_state" or asymm_obs:
            self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
        
            fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        else:
            fingertip_obs_start = 2 * self.num_shadow_hand_dofs
        
        
        # finger tip state # 
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
        #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.force_sensor_tensor[:, :num_ft_force_torques] # full state or asymm_obs #
        # else
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
            # right_hand_rot

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
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(canon_object_pos)
        self.obs_buf[:, obj_obs_start + 3:obj_obs_start + 7] = self.unpose_quat(self.object_pose[:, 3:7])
        self.obs_buf[:, obj_obs_start + 7:obj_obs_start + 10] = self.unpose_vec(self.object_linvel)
        self.obs_buf[:, obj_obs_start + 10:obj_obs_start + 13] = self.vel_obs_scale * self.unpose_vec(self.object_angvel)
        self.obs_buf[:, obj_obs_start + 13:obj_obs_start + 16] = self.unpose_vec(self.goal_pos - self.object_pos)

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
        if self.obs_type == 'pure_state_wref_wdelta' and self.use_kinematics_bias_wdelta:
            
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
        
        if not self.single_instance_state_based_test:
            ### add the obj latent features ###
            ### add the env obj latent features ###
            self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_object_latent_feat
            
            if self.use_inst_latent_features:
                obj_feat_st_idx = obj_feat_st_idx + self.object_feat_dim
                self.obs_buf[:, obj_feat_st_idx: obj_feat_st_idx + self.object_feat_dim] = self.env_inst_latent_feat
            
            if self.supervised_training:
                # TODO: add expected actions here #
                nex_hand_qtars_st_idx = obj_feat_st_idx + self.object_feat_dim
                env_max_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) - 1
                # nn_envs,
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
                
                # # prev_detlat_targets # # prev delta targets #
                # delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
                # cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
                # self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
                # self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
                
                self.obs_buf[:, nex_hand_qtars_st_idx: nex_hand_qtars_st_idx + self.num_actions] = nex_actions 
                
                if self.grab_obj_type_to_opt_res is not None:
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

    def reset(self, env_ids=None, goal_env_ids=None):
        
        # 
        maxx_progress_buf = torch.max(self.progress_buf)
        minn_progress_buf = torch.min(self.progress_buf)
        # print(f"maxx_progress_buf: {maxx_progress_buf}, minn_progress_buf: {minn_progress_buf}")
        
        # self.ref_ts = 0
        
        if env_ids is None : 
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        # randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        # generate random values
        rand_floats = torch_rand_float(-1.0, 1.0, (len(env_ids), self.num_shadow_hand_dofs * 2 + 5), device=self.device)

        # randomize start object poses
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

        # shadow_hand_default_dof_pos # dof pos #
        # self.dof_state[:, : self.] # dof pos #
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = self.shadow_hand_default_dof_pos[env_ids, : self.num_shadow_hand_dofs]
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = self.shadow_hand_default_dof_pos[env_ids, : self.num_shadow_hand_dofs]
        
        self.prev_delta_targets[env_ids, :] = 0
        self.cur_delta_targets[env_ids, :] = 0

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
        #TODO: do we need to reset the table? #

        # if self.random_time:
        #     self.random_time = False
        #     self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        #     if self.use_twostage_rew:
        #         # self.cur_grasp_fr[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        #         self.grasping_progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
        # else:
        self.progress_buf[env_ids] = 0
        if self.use_twostage_rew:
            self.grasping_progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        self.successes[env_ids] = 0
        
        
        if self.use_twostage_rew:
            # grasping_progress_buf, grasp_manip_stages, grasping_succ_buf
            self.grasp_manip_stages[env_ids] = 0
            # self.cur_grasp_fr[env_ids] = 0
            # self.grasp_manip_stages[goal_env_ids] = 0
            # self.grasping_progress_buf[env_ids] = 0
            # if self.random_time:
            #     ## TODO: waht's the role of random-time here ##
            #     self.random_time = False
            #     self.progress_buf[env_ids] = torch.randint(0, self.max_episode_length, (len(env_ids),), device=self.device)
            # else:
            #     self.progress_buf[env_ids] = 0
            self.grasping_succ_buf[env_ids] = 0
        
        if self.lifting_separate_stages:
            self.reach_lifting_stage[env_ids] = 0
            # self.lift_fr[env_ids] = 0
            # self.reach_lifting_stage[goal_env_ids] = 0
            # self.lift_fr[goal_env_ids] = 0
        
        
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
        
        self.compute_observations()
        
        
        return torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    def try_save(self, ):
        # if self.reset_nn % 1000 == 0:
        if self.reset_nn % 1 == 0:
            print(f"reset_nn: {self.reset_nn}")
            self.ts_to_hand_obj_states['object_code_list'] = self.object_code_list
            self.ts_to_hand_obj_states['env_object_codes'] = self.env_object_codes
            # logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
            logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{1}.npy"
            logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
            np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states) # save the ts_to_hand_obj_states #
            print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")

    def pre_physics_step(self, actions):
        # print(f" in physics step")
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)


        wether_to_reset = self.progress_buf >= self.env_maxx_progress_buf
        tot_envs_ids = torch.arange(self.num_envs, device=self.device)
        reset_env_idxs = tot_envs_ids[wether_to_reset]
        
        if len(reset_env_idxs) > 0:
            self.reset(reset_env_idxs)

        # if self.progress_buf[0].item() == self.optimized_hand_qs.size(0):
        #     self.reset()
            
        # progress buf #

        # if only goals need reset, then call set API 
        # if len(goal_env_ids) > 0 and len(env_ids) == 0:
        #     self.reset_target_pose(goal_env_ids, apply_reset=True)
        # # if goals need reset in addition to other envs, call set API in reset() #
        # elif len(goal_env_ids) > 0:
        #     self.reset_target_pose(goal_env_ids)

        # if len(env_ids) > 0:
        #     self.reset(env_ids, goal_env_ids)



        # print(f"progress_buf:{self.progress_buf}")

        # self.get_pose_quat() # # pre physics step #
        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3])
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6])
        
        # maxx_progress_buf = torch.max(self.progress_buf).item()
        # minn_progress_buf = torch.min(self.progress_buf).item()
        # print(f"maxx_progress_buf: {maxx_progress_buf}, minn_progress_buf: {minn_progress_buf}")
        
        self.actions = actions.clone().to(self.device)

        # use the progress buf to select hand replay qs and the object replay qs #
        cur_replay_progress_buf= torch.clamp(self.progress_buf, min=torch.zeros_like(self.progress_buf), max=torch.zeros_like(self.progress_buf) +  self.env_maxx_progress_buf - 1)
        envs_hand_replay_qs = batched_index_select(self.tot_kine_qs, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x len(hand_qs)

        cur_hand_replay_qs = batched_index_select(envs_hand_replay_qs, cur_replay_progress_buf.unsqueeze(1), dim=1)
        cur_hand_replay_qs = cur_hand_replay_qs.squeeze(1) # nn_envs x nn_hand_qs #
        print(f"hand_qs: {cur_hand_replay_qs.size()}, envs_hand_replay_qs: {envs_hand_replay_qs.size()}, cur_replay_progress_buf: {cur_replay_progress_buf.size()}, env_inst_idxes: {len(self.env_inst_idxes)}")
        # print(cur_hand_replay_qs)
        # self.shadow_hand_dof_pos[env_ids, :] = self.shadow_hand_default_dof_pos[env_ids, :] # env_ids #
        # self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
        #                                        self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]
        # # shadow_hand_default_dof_pos 
        # # self.dof_state[:, : self.]
        # self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = self.shadow_hand_default_dof_pos[env_ids, : self.num_shadow_hand_dofs]
        # self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = self.shadow_hand_default_dof_pos[env_ids, : self.num_shadow_hand_dofs]
        # hand_indices = self.hand_indices[env_ids].to(torch.int32) # hand indices #
        # all_hand_indices = torch.unique(torch.cat([hand_indices]).to(torch.int32))

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))



        # sset the dof state to hand replay qs
        # hand replay qs: nn_envs x nn_hand_dofs
        self.dof_state[:, 0] = cur_hand_replay_qs.contiguous().view(-1).contiguous() # get the dof state #
        self.dof_state[:, 1] = 0.0
        self.gym.set_dof_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.dof_state),
                                            gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        

        envs_optimized_obj_pos = batched_index_select(self.tot_kine_obj_trans, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x len(hand_qs)
        envs_optimized_obj_ornt = batched_index_select(self.tot_kine_obj_ornt, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x len(hand_qs)
        envs_optimized_obj_pose = torch.cat([envs_optimized_obj_pos, envs_optimized_obj_ornt], dim=-1)
        # nn_envs x nn_state_dims #
        cur_obj_replay_pos = batched_index_select(envs_optimized_obj_pose[..., :3], cur_replay_progress_buf.unsqueeze(-1), dim=1)
        cur_obj_replay_pos = cur_obj_replay_pos.squeeze(1)

        # nn_envs x 4 --- object orientations #
        cur_obj_replay_ornt = batched_index_select(envs_optimized_obj_pose[..., 3:], cur_replay_progress_buf.unsqueeze(-1), dim=1)
        cur_obj_replay_ornt = cur_obj_replay_ornt.squeeze(1)

        cur_obj_replay_pose = torch.cat(
            [ cur_obj_replay_pos, cur_obj_replay_ornt ], dim=-1
        )

        # roto state tensro #
        self.root_state_tensor[self.object_indices, :7] = cur_obj_replay_pose.clone()
        
        self.root_state_tensor[self.object_indices, 7:13] = torch.zeros_like(self.root_state_tensor[self.object_indices, 7:13])
        
        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices,
                                              self.goal_object_indices, ]).to(torch.int32))
        # state tensor indexed # actor state tensor indexed #
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        

        self.cur_targets[:, self.actuated_dof_indices] = cur_hand_replay_qs.clone()

        # if self.use_kinematics_bias_wdelta:
        #     # print(f"self.use_kinematics_bias_wdelta: {self.use_kinematics_bias_wdelta}")
        #     increased_progress_buf = self.progress_buf + 1
            
        #     # two instances? #
        #     # increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
        #     # get the kinematicsof the increaesd progres buf as the kinematics bias # 
        #     # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
        #     # ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]

        #     # tot_envs_hand_qs = self.tot_kine_qs
        #     # ### ### #
            
        #     tot_envs_hand_qs = self.tot_hand_preopt_res
            
        #     maxx_env_inst_idx = torch.max(self.env_inst_idxes).item()
        #     minn_env_inst_idx = torch.min(self.env_inst_idxes).item()
        #     # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_env_inst_idx: {maxx_env_inst_idx}, minn_env_inst_idx: {minn_env_inst_idx}")
            
        #     tot_envs_hand_qs = batched_index_select(tot_envs_hand_qs, self.env_inst_idxes, dim=0) # nn_envs x 
        #     envs_maxx_episode_length = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0)
        #     increased_progress_buf = torch.clamp(increased_progress_buf, min=torch.zeros_like(envs_maxx_episode_length), max=envs_maxx_episode_length)
            
            
        #     maxx_increased_progress_buf = torch.max(increased_progress_buf).item()
        #     minn_increased_progress_buf= torch.min(increased_progress_buf).item()
        #     # print(f"tot_envs_hand_qs: {tot_envs_hand_qs.size()}, maxx_increased_progress_buf: {maxx_increased_progress_buf}, minn_increased_progress_buf: {minn_increased_progress_buf}")
            
        #     ctl_kinematics_bias = batched_index_select(tot_envs_hand_qs, increased_progress_buf.unsqueeze(-1), dim=1) # nn_envs x nn_steps x nn_hand_dofs # 
        #     ctl_kinematics_bias = ctl_kinematics_bias.squeeze(1) # nn_envs x nn_hand_dofs #
            
            
        #     if self.use_twostage_rew:
        #         grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
        #         expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
        #         grasp_manip_stages_flag = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
        #         ctl_kinematics_bias = torch.where(
        #             grasp_manip_stages_flag, expanded_grasp_frame_hand_qpos, ctl_kinematics_bias
        #         )
            
        #     # prev_detlat_targets # 
        #     delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
        #     cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
        #     self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
        #     self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
        #     if self.kinematics_only:
        #         cur_targets = ctl_kinematics_bias
        #     else:
        #         cur_targets = ctl_kinematics_bias + self.cur_delta_targets[:, self.actuated_dof_indices]
        #     self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(cur_targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            
        # # if we use the kinematics motion as the bis # 
        # elif self.use_kinematics_bias:
        #     increased_progress_buf = self.progress_buf + 1
        #     increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
        #     # get the kinematicsof the increaesd progres buf as the kinematics bias # 
        #     # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
        #     # hand_qs_th
        #     ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]
        #     # ctl kinematics bias #
        #     if self.kinematics_only:
        #         targets = ctl_kinematics_bias
        #     else:
        #         # targets = ctl_kinematics_bias + self.shadow_hand_dof_speed_scale * self.dt * self.actions 
        #         #### from actions to targets ####
        #         targets = ctl_kinematics_bias + self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
        #     # kinematics_only # targets #
        #     self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        # elif self.use_relative_control: # use relative control #
        #     # 
        #     # targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale * self.dt * self.actions
        #     # shadow_hand_dof_speed_scale_tsr # 
        #     targets = self.prev_targets[:, self.actuated_dof_indices] + self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions
        #     self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices], self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        #     # targets = self.prev_targets #
        # else:
        #     self.cur_targets[:, self.actuated_dof_indices] = scale(self.actions[:, :],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        #     self.cur_targets[:, self.actuated_dof_indices] = self.act_moving_average * self.cur_targets[:,self.actuated_dof_indices] + (1.0 - self.act_moving_average) * self.prev_targets[:,self.actuated_dof_indices]
        #     self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(self.cur_targets[:, self.actuated_dof_indices],self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])

        #     ### TODO: check whether if it is because we use the shadow hand dof pos to set the dof state tensor, so we need to set the dof state tensor here ###
        #     # self.apply_forces[:, 1, :] = self.actions[:, 0:3] * self.dt * self.transition_scale * 100000
        #     # self.apply_torque[:, 1, :] = self.actions[:, 3:6] * self.dt * self.orientation_scale * 1000

        #     # self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(self.apply_forces),
        #     #                                         gymtorch.unwrap_tensor(self.apply_torque), gymapi.ENV_SPACE)

        # # prev targets an the current jarets #
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        # all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))

        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))
        
        
        
        

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        
        if self.use_twostage_rew:
            self.grasping_progress_buf += 1

        self.ref_ts += 1

        # # print(f"To compute observations with ref_ts: {self.ref_ts}")
        self.compute_observations()
        # print(f"To compute reward with ref_ts: {self.ref_ts}")
        self.compute_reward(self.actions)
        
        # if self.test: # test the test setting #
        #     # if self.ref_ts >= self.max_episode_length - 3: # try save #
        #     self.try_save()

        if self.viewer and self.debug_viz:
            # draw axes on target object
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.object_back_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.goal_pos[i], self.object_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_pos[i], self.right_hand_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.right_hand_th_pos[i], self.right_hand_th_rot[i])

                # self.add_debug_lines(self.envs[i], self.left_hand_ff_pos[i], self.right_hand_ff_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_mf_pos[i], self.right_hand_mf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_rf_pos[i], self.right_hand_rf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_lf_pos[i], self.right_hand_lf_rot[i])
                # self.add_debug_lines(self.envs[i], self.left_hand_th_pos[i], self.right_hand_th_rot[i])


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


@torch.jit.script
def compute_hand_reward(
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
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool, hand_up_threshold_1: float, hand_up_threshold_2: float, num_fingers: int, w_obj_ornt: bool, w_obj_vels: bool, separate_stages: bool, hand_pose_guidance_glb_trans_coef: float, hand_pose_guidance_glb_rot_coef: float , hand_pose_guidance_fingerpose_coef: float, rew_finger_obj_dist_coef: float, rew_delta_hand_pose_coef: float, rew_obj_pose_coef: float, goal_dist_thres: float , envs_hand_qtars, env_hand_cur_targets, use_hand_actions_rew: bool, prev_dof_vel, cur_dof_vel, rew_smoothness_coef: float
):
    if separate_stages:
        lowest = object_pos[:, 2].unsqueeze(-1).repeat(1, 3)
        # calculate the target pos based on the target lifting pose #
        target_pos = torch.where(lowest < 0.19, target_lifting_pos, target_pos)
        # target pos, object pose # object pose #
    
    goal_dist = torch.norm(target_pos - object_pos, p=2, dim=-1)
    # target pos, right hand pos # target pos change frequency? ## pose change frequencies ##
    goal_hand_dist = torch.norm(target_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.norm(object_handle_pos - right_hand_pos, p=2, dim=-1)
    right_hand_dist = torch.where(right_hand_dist >= 0.5, 0.5 + 0 * right_hand_dist, right_hand_dist)

    right_hand_finger_dist = (torch.norm(object_handle_pos - right_hand_ff_pos, p=2, dim=-1) + torch.norm(
        object_handle_pos - right_hand_mf_pos, p=2, dim=-1)+ torch.norm(object_handle_pos - right_hand_rf_pos, p=2, dim=-1)  + torch.norm(object_handle_pos - right_hand_th_pos, p=2, dim=-1))
                #               + torch.norm(
                # object_handle_pos - right_hand_lf_pos, p=2, dim=-1) 
    # idxxx= 6
    # print(f"right_hand_dist: {right_hand_dist[idxxx]}, object_handle_pos: {object_handle_pos[idxxx]},right_hand_pos: {right_hand_pos[idxxx]}, object_pos: {object_pos[idxxx]}")
    
    # finger dist threshold #
    finger_dist_threshold = 0.6 * num_fingers                         
    
    right_hand_finger_dist = torch.where(right_hand_finger_dist >= finger_dist_threshold, finger_dist_threshold + 0 * right_hand_finger_dist,right_hand_finger_dist)


    # thumb_finger_dist = 

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

    
    delta_value = hand_pos_rew_coef * delta_hand_pos_value + hand_rot_rew_coef * delta_hand_rot_value + hand_qpos_rew_coef * delta_qpos_value 
    
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
        
        ##### original version #####
        # inhand_obj_pos_ornt_rew = 1 * (0.9 - 2 * goal_dist)
        ##### original version #####
        
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
        hand_up = torch.where(lowest >= hand_up_threshold_1, torch.where(flag == 2, 0.1 + 0.1 * actions[:, 2], hand_up), hand_up)
        hand_up = torch.where(lowest >= hand_up_threshold_2, torch.where(flag == 2, 0.2 - goal_hand_dist * 0, hand_up), hand_up)
        

        # additional up #
        # hand_additional_up = torch.zeros_like(right_hand_finger_dist)
        # # actions are used to comptue the rewards ? #  #  if object # -- # 
        # # 
        # hand_additional_up = torch.where(lowest < 0.1, torch.where(flag == 2, 0.1 * actions[:, 2], hand_up), hand_up)
        # hand addtional up # flag # # is below the threshold # below the threshold ##  # hand up -- without the hand addtional up ?#

        # hand up # flag = () # flag = () #
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
        
        hand_dist_flag = (right_hand_dist <= 0.12).int()
        right_hand_finger_dist = torch.where(hand_dist_flag == 1, right_hand_finger_dist, 0.0 * right_hand_finger_dist)
        
        reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (right_hand_finger_dist + 2.0 * right_hand_dist)  + goal_hand_rew + bonus # + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
        # reward = (-rew_delta_hand_pose_coef) * delta_value + (-rew_finger_obj_dist_coef) * (2.0 * right_hand_dist)  + goal_hand_rew + bonus + hand_up 
    
    # resets buf #
    resets = reset_buf

    # Find out which envs hit the goal and update successes count
    resets = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(resets), resets)
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
        # compute rewards #
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
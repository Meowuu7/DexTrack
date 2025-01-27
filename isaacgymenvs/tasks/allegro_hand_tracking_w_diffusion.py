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
from hydra import compose, initialize

# from isaacgymenvs.diffusion.dataset.get_data import get_dataset_loader_3d_pc, get_dataset_loader_3d_v3_pc, get_dataset_loader_3d_v5_pc, get_dataset_loader_3d_v6_pc, get_dataset_loader_3d_v7_pc
# from model_util import create_model_and_diffusion_3d_pc, create_model_deterministic
import trimesh

# [Debug] scale: 1.0
# [Debug] object_init_z: tensor([0.0233], device='cuda:0'), object_euler_xy: tensor([-2.8889,  0.0305], device='cuda:0')
# [Error] [carb.gym.plugin] Graphics is nullptr in GymCreateTextureFromFile
# JointSpec type free not yet supported!
# self.num_shadow_hand_bodies:  24
# self.num_shadow_hand_shapes:  20
# self.num_shadow_hand_dofs:  22
# self.num_shadow_hand_actuators:  22
# self.num_shadow_hand_tendons:  4
#### dof lower limits ####
# [Debug] shadow_hand_dof_lower_limits: tensor([-0.3490,  0.0000,  0.0000,  0.0000, -0.3490,  0.0000,  0.0000,  0.0000,
#         -0.3490,  0.0000,  0.0000,  0.0000,  0.0000, -0.3490,  0.0000,  0.0000,
#          0.0000, -1.0470,  0.0000, -0.2090, -0.5240, -1.5710], device='cuda:0')
# [Debug] shadow_hand_dof_upper_limits: tensor([0.3490, 1.5710, 1.5710, 1.5710, 0.3490, 1.5710, 1.5710, 1.5710, 0.3490,
#         1.5710, 1.5710, 1.5710, 0.7850, 0.3490, 1.5710, 1.5710, 1.5710, 1.0470,
#         1.2220, 0.2090, 0.5240, 0.0000], device='cuda:0')
# scaled_object_asset_file: sem/Headphone/coacd/coacd_1.urdf
# Using VHACD cache directory '/root/.isaacgym/vhacd'
# Found existing convex decomposition for mesh '../assets/meshdatav3_scaled/sem/Headphone/coacd/decomposed.obj'
# [Debug] object_dof_lower_limits: tensor([], device='cuda:0')
# [Debug] object_dof_upper_limits: tensor([], device='cuda:0')
# num_shadow_hand_dofs: 22
# [Debug] shadow_hand_default_dof_pos: tensor([-0.1096,  0.1048,  0.1459,  1.7980, -0.1805, -1.0893,  0.1091,  0.1922,
#          0.2433,  0.0931, -0.1590,  0.3097,  0.3075,  0.0583,  0.1312,  0.6333,
#          0.2597,  0.0881,  1.0236,  0.2167,  1.0111,  0.2123], device='cuda:0')
# [Debug] num_dofs: 22


### NOTE: some functions in this scirt and the TrackingDiff environment calss amy lag behind that in the original Tracking environment #### 
### NOTE: but we use it only for the in-the-loop sampling purpose



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


class AllegroHandTrackingDiff(BaseTask):
    # def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless,
    #              agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False):
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render,
                 agent_index=[[[0, 1, 2, 3, 4, 5]], [[0, 1, 2, 3, 4, 5]]], is_multi_agent=False, exp_logging_dir=None): # 
        
        ## this one is only
        # with initialize(version_base="1.3", config_path="../../diffusion/cfgs", job_name="test_app"):
        #     if os.path.exists("/cephfs/xueyi/backup"):
        #         diff_cfg = compose(config_name="K2_config_3d_k8s")
        #     elif os.path.exists("/root/diffsim/softzoo"):
        #         diff_cfg = compose(config_name="config_3d_k8s")
        #     else:
        #         raise ValueError("Please run this code on the k8s cluster")
        self.rl_device = rl_device
        self.exp_logging_dir = exp_logging_dir
        self.device = self.rl_device
        
        class diff_cfg:
            def __init__(self, ):
                class Training:
                    def __init__(self, ):
                        self.task_cond = True
                        self.setting = 'regular_training'
                        self.batch_size = 64
                        self.use_jointspace_seq = True
                        self.diff_task_translations = False
                        self.diff_task_space = False
                        self.kine_diff = False 
                        self.concat_two_dims = False
                        self.tracking_ctl_diff = True
                        self.AE_Diff = True
                        self.train_AE = False
                        self.train_Diff = True
                        self.cond_diff_allparams = True
                        self.succ_rew_threshold = 50.0
                        self.slicing_data = True
                        self.slicing_ws = cfg["diffusion"]["slicing_ws"]
                        self.history_ws = cfg["diffusion"]["history_ws"]
                        self.debug = False
                        self.task_cond_type = 'history_future'
                        self.sub_task_cond_type = cfg["diffusion"]["sub_task_cond_type"]
                        self.glb_rot_use_quat = cfg['diffusion']['glb_rot_use_quat']
                        self.use_kine_obj_pos_canonicalization = cfg['diffusion']['use_kine_obj_pos_canonicalization']
                class Dataset_3d_pc:
                    def __init__(self, ):
                        self.multi_inst = False
                        self.sim_platform = 'isaac'
                        self.statistics_info_fn = cfg["diffusion"]["statistics_info_fn"]
                        self.single_inst = False
                        
                        
                class Model:
                    def __init__(self, ):
                        self.model_arch = 'transformer_v3'
                        self.n_layers_3d_pc = 7
                        # self.debug
                        class Hidden_mlp_dims_3d_pc:
                            def __init__(self, ):
                                self.X = 1024
                                self.feat = 2048
                                self.t = 256
                        self.hidden_mlp_dims_3d_pc = Hidden_mlp_dims_3d_pc()
                class Diffusion:
                    def __init__(self, ):
                        self.noise_schedule = 'linear'
                        self.sigma_small = True 
                        self.lambda_vel  = 0.0
                        self.lambda_rcxyz = 0.0
                        self.lambda_fc = 0.0
                        
                self.training = Training()
                self.dataset_3d_pc = Dataset_3d_pc()
                self.model = Model()
                self.diffusion = Diffusion()
                
        
        args = diff_cfg()
        
        
        # # args.sampling.sampling = True
        self.maxx_nn_pts = 512
        self.resume_checkpoint = cfg["diffusion"]["resume_checkpoint_pc"]
        
        self.use_deterministic = cfg["diffusion"]["use_deterministic"]
        
        self.predict_ws = cfg['diffusion']['predict_ws']
        self.glb_rot_use_quat = args.training.glb_rot_use_quat
        # args.dataset_3d_pc.multi_inst = False
        # args.dataset_3d_pc.sim_platform = 'isaac'
        
        # args.training.task_cond_type = 'history_future'
        # args.training.debug = False
        
        self.use_kine_obj_pos_canonicalization = args.training.use_kine_obj_pos_canonicalization
        self.statistics_info_fn = args.dataset_3d_pc.statistics_info_fn
        self.slicing_ws = args.training.slicing_ws
        self.microbatch = 256
        self.batch_size = 256
        
        # microbatch #
        
        # if len(pre_args.exp_tag) > 0:
        #     args.exp_tag = pre_args.exp_tag
            
        # if args.save_dir is None:
        #     raise FileNotFoundError('save_dir was not specified.')
        
        # else:
        #     os.makedirs(args.save_dir, exist_ok=True) 
        #     exp_tag = args.exp_tag
        #     args.save_dir = os.path.join(args.save_dir, exp_tag)
        #     os.makedirs(args.save_dir, exist_ok=True)
        
        #### TODO: do not create the datasets --- do not load all training data --- but just load statistics realted files to the dataset ###
        print("creating data loader...")
        # if args.dataset_3d_pc.data_tag == "v6":
        #     data = get_dataset_loader_3d_v6_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
        # elif args.dataset_3d_pc.data_tag == "v7":
        #     data = get_dataset_loader_3d_v7_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)        
        # elif args.dataset_3d_pc.data_tag == "v5":
        #     print(f"getting dataset for model with arch: {args.model.model_arch}")
        #     data = get_dataset_loader_3d_v5_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
        # elif args.model.model_arch == "transformer_v2":
        #     print(f"getting dataset for model with arch: {args.model.model_arch}")
        #     data = get_dataset_loader_3d_v3_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
        # else:
        #     data = get_dataset_loader_3d_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)

        print("creating model and diffusion...")
        
        # with that models and the pre-args #
        # add the cfg to the pre-args #
        self.kine_diff  = args.training.kine_diff
        self.task_cond = args.training.task_cond
        self.AE_Diff = args.training.AE_Diff
        self.train_AE = args.training.train_AE
        self.train_Diff = args.training.train_Diff
        self.cond_diff_allparams = args.training.cond_diff_allparams
        
        
        
        
        # self.slicing_ws = args.dataset_3d_pc.slicing_ws
        self.history_ws = args.training.history_ws
        self.future_ws = self.slicing_ws
        
        if self.use_deterministic: 
            model = create_model_deterministic(args)
            model.to(rl_device)
            diffusion = None
        else:
            # model, diffusion = create_model_and_diffusion(args, data)
            model, diffusion = create_model_and_diffusion_3d_pc(args)
            model.to(rl_device)
        self.prior_model = diffusion 
        self.ddp_model = model
        self.model = model
        
        
        print(f"=== Loading model parameters ===")
        ### laod the checkpoint ###
        self._load_and_sync_parameters()
        ## ## then the model should have the predict from model #
        print(f"=== Loading data statistics ===")
        self._load_data_statistics() # 
        
        
        

        
        
        # hand tracking #
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
        
        if self.exp_logging_dir is None:
            self.exp_logging_dir = self.cfg['env']['exp_logging_dir']
        
        self.object_name = self.cfg["env"]["object_name"]
        
        
        self.transition_scale = self.cfg["env"]["transition_scale"]
        self.orientation_scale = self.cfg["env"]["orientation_scale"]
        self.mocap_sv_info_fn = self.cfg["env"]["mocap_sv_info_fn"]
        
        if 'taco_' in self.object_name and 'TACO' in self.mocap_sv_info_fn and 'ori_grab' not in self.mocap_sv_info_fn:
            self.dataset_type = 'taco'
        elif 'grab' in self.object_name or 'GRAB' in self.mocap_sv_info_fn or 'ori_grab' in self.mocap_sv_info_fn:
            self.dataset_type = 'grab'
        else:
            raise ValueError(f"Unknown dataset type for object: {self.object_name}")

        
        
        
        data_inst_tag = self.object_name
        self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
        grab_mesh_fn = f"{data_inst_tag}.obj"
        grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
        # get the object mesh #
        obj_mesh = trimesh.load_mesh(grab_mesh_fn)
        obj_verts = obj_mesh.vertices # nn_pts x 3
        to_sample_fr_idxes = list(range(obj_verts.shape[0]))
        while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            to_sample_fr_idxes += list(range(obj_verts.shape[0]))
        random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
        random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
        obj_verts = obj_verts[random_sampled_idxes] 
        obj_verts = torch.from_numpy(obj_verts).float().to(self.device)
        self.obj_verts = obj_verts
        
        
        
        
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
        self.ref_ts = 0 
        # right_hand_dist_thres
        try:
            self.pre_optimized_traj = self.cfg['env']['pre_optimized_traj'] # pre optimized traj #
        except:
            
            self.pre_optimized_traj = None
        
        ## ## right_hand_dist_thres ## ##
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
        try:
            self.add_table = self.cfg['env']['add_table']
        except:
            self.add_table = False
        
        try:
            self.table_z_dim = self.cfg['env']['table_z_dim']
        except:
            self.table_z_dim = 0.0
            
        try:
            self.use_taco_obj_traj = self.cfg['env']['use_taco_obj_traj']
        except:
            self.use_taco_obj_traj = False
        # 

        control_freq_inv = self.cfg["env"].get("controlFrequencyInv", 1)
        self.control_freq_inv = control_freq_inv
        if self.reset_time > 0.0:
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
        self.shadow_hand_dof_speed_scale_tsr = torch.tensor(self.shadow_hand_dof_speed_scale_list, device=self.rl_device, dtype=torch.float) # speed scale tsr #
        
        
        self.up_axis = 'z'
        # 'palm': 'palm_link',
        # 'thumb': 'link_15_tip',
        # 'index': 'link_3_tip', # link 3 tip
        # 'middle': 'link_7_tip', # link 3 tip
        # 'ring': 'link_11_tip' # link 3 tip
        # self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        if self.use_fingertips:
            self.fingertips = ["link_15_tip", "link_3_tip", "link_7_tip", "link_11_tip"]
        else:
            self.fingertips = ["link_15", "link_3", "link_7", "link_11"]
        self.hand_center = ["palm_link"]
        self.num_fingertips = len(self.fingertips) 
        
        
        self._load_mocap_info()
        
        self.max_episode_length = min(self.max_episode_length, self.hand_qs.shape[0] - 1)
        
        print(f"max_episode_length: {self.max_episode_length}, hand_qs_shape: {self.hand_qs.shape }" ) ## get the hand qs shape and the episode length ##
        
        # self.num_hand_obs = 66 + 95 + 24 + 6  # 191 =  22*3 + (65+30) + 24
        #  + 6 + nn_dof (action) + 
        # 16 (obj) + 7 + nn_dof (goal) + 64
        self.num_hand_obs = 66 + 76 + 22 + 6  # 191 =  22*3 + (65+30) + 24
        
        num_pure_obs = 2 * self.nn_hand_dof + 13 * self.num_fingertips + 6 + self.nn_hand_dof + 16 + 7 + self.nn_hand_dof # # 
        
        num_obs = self.num_hand_obs + 16 + 7 + self.nn_hand_dof + 64 #  236 + 64
        self.num_obs_dict = {
            "full_state": num_obs,
            "full_state_nforce": num_obs + 300 - num_obs, #  num_obs - self.nn_hand_dof - 24 # 24 -- fingertip forces
            "pure_state": num_pure_obs, # number obs - self.nnhanddofs #
            "pure_state_wref": num_pure_obs + self.nn_hand_dof,
            "pure_state_wref_wdelta": num_pure_obs + self.nn_hand_dof + self.nn_hand_dof
        }   
        # decide the observation type and size #
        # num_obs_dict # hand tracking #
        
        
        self.use_vel_obs = False
        self.fingertip_obs = True
        self.asymmetric_obs = self.cfg["env"]["asymmetric_observations"]
        num_states = 0
        if self.asymmetric_obs:
            num_states = 211
        self.cfg["env"]["numObservations"] = self.num_obs_dict[self.obs_type]
        
        if self.tight_obs: # with next state references and with the current delta targets #
            self.cfg['env']['numObservations'] -= 7
        
        print(f"obs_type: {self.obs_type}, num_obs: {self.cfg['env']['numObservations']}")
        
        self.cfg["env"]["numStates"] = num_states
        self.num_agents = 1
        self.cfg["env"]["numActions"] = self.nn_hand_dof #  24 
        # self.cfg["device_type"] = device_type
        # self.cfg["device_id"] = device_id
        self.cfg["headless"] = headless

        # super().__init__(cfg=self.cfg, enable_camera_sensors=False)
        
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

        # create some wrapper tensors for different slices
        # self.shadow_hand_default_dof_pos = torch.zeros(self.num_shadow_hand_dofs, dtype=torch.float, device=self.device)
        
        # dof_state_tensor[:, : self.num_shadow_hand_dofs, 0] = self.shadow_hand_default_dof_pos
        
        ## debug the hand pose ## 
        # is is the same as we expected? #
        print(f"[Debug] shadow_hand_default_dof_pos: {self.shadow_hand_default_dof_pos}")
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_state[:, 0] = self.shadow_hand_default_dof_pos.unsqueeze(0).repeat(self.num_envs, 1).view(-1).contiguous()
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
        
        # hand_palm_fingers_obj_contact_buf, right_hand_dist_buf
        self.hand_palm_fingers_obj_contact_buf = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)
        self.right_hand_dist_buf = torch.zeros((self.num_envs, ), dtype=torch.float, device=self.device)    
        
        self.reach_lifting_stage = torch.zeros((self.num_envs), dtype=torch.float, device=self.device)
        
        
        ### TODO: initialize the hand dof buf, pose buf .. ##
        # history_obj_pos_buf, history_obj_ornt_buf, history_obj_rot_euler_buf #
        self.history_hand_dof_buf = torch.zeros((self.num_envs, self.max_episode_length, self.num_shadow_hand_dofs), dtype=torch.float, device=self.device)
        self.history_obj_pos_buf = torch.zeros((self.num_envs, self.max_episode_length, 3), dtype=torch.float, device=self.device)
        self.history_obj_ornt_buf = torch.zeros((self.num_envs, self.max_episode_length, 4), dtype=torch.float, device=self.device)
        self.history_obj_rot_euler_buf = torch.zeros((self.num_envs, self.max_episode_length, 3), dtype=torch.float, device=self.device)
        
        self.total_successes = 0
        self.total_resets = 0
        
        self.ts_to_hand_obj_states = {}
        
        self.ref_ts =  0
        self.reset_nn = 0
    
    
    
    def _load_data_statistics(self, ): ## data statistics ##
        # self.data_statistics #
        self.data_statistics = np.load(self.statistics_info_fn, allow_pickle=True).item()
        # dict_keys(['avg_hand_qs', 'std_hand_qs', 'avg_hand_qtars', 'std_hand_qtars', 'avg_obj_verts', 'std_obj_verts', 'avg_kine_hand_qs', 'std_kine_hand_qs', 'avg_obj_transl', 'std_obj_transl', 'avg_obj_rot_euler', 'std_obj_rot_euler'])
        ### add data related information from the dictionary ###
        self.avg_hand_qs = self.data_statistics['avg_hand_qs']
        self.std_hand_qs = self.data_statistics['std_hand_qs']
        self.avg_hand_qtars = self.data_statistics['avg_hand_qtars']
        self.std_hand_qtars = self.data_statistics['std_hand_qtars']
        self.avg_obj_verts = self.data_statistics['avg_obj_verts']
        self.std_obj_verts = self.data_statistics['std_obj_verts']
        self.avg_kine_hand_qs = self.data_statistics['avg_kine_hand_qs']
        self.std_kine_hand_qs = self.data_statistics['std_kine_hand_qs']
        self.avg_obj_transl = self.data_statistics['avg_obj_transl']
        self.std_obj_transl = self.data_statistics['std_obj_transl']
        self.avg_obj_rot_euler = self.data_statistics['avg_obj_rot_euler']
        self.std_obj_rot_euler = self.data_statistics['std_obj_rot_euler']
        
        self.avg_hand_qs = torch.from_numpy(self.avg_hand_qs).float().to(self.rl_device)
        self.std_hand_qs = torch.from_numpy(self.std_hand_qs).float().to(self.rl_device)
        self.avg_hand_qtars = torch.from_numpy(self.avg_hand_qtars).float().to(self.rl_device)
        self.std_hand_qtars = torch.from_numpy(self.std_hand_qtars).float().to(self.rl_device)
        self.avg_obj_verts = torch.from_numpy(self.avg_obj_verts).float().to(self.rl_device)
        self.std_obj_verts = torch.from_numpy(self.std_obj_verts).float().to(self.rl_device)
        self.avg_kine_hand_qs = torch.from_numpy(self.avg_kine_hand_qs).float().to(self.rl_device)
        self.std_kine_hand_qs = torch.from_numpy(self.std_kine_hand_qs).float().to(self.rl_device)
        self.avg_obj_transl = torch.from_numpy(self.avg_obj_transl).float().to(self.rl_device)
        self.std_obj_transl = torch.from_numpy(self.std_obj_transl).float().to(self.rl_device)
        self.avg_obj_rot_euler = torch.from_numpy(self.avg_obj_rot_euler).float().to(self.rl_device)
        self.std_obj_rot_euler = torch.from_numpy(self.std_obj_rot_euler).float().to(self.rl_device)
        
        
     

    def parse_resume_step_from_filename(self, filename):
        """
        Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        split = filename.split("model")
        if len(split) < 2:
            return 0
        split1 = split[-1].split(".")[0]
        try:
            return int(split1)
        except ValueError:
            return 0
        
    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict:
            print(f"k: {k}")
            if k in model_dict:
                if ori_dict[k].shape == model_dict[k].shape:
                    v = ori_dict[k]
                    part_dict[k] = v
                    tot_params_n += 1
        model_dict.update(part_dict)
        model.load_state_dict(model_dict)
        print(f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
    
    
    
    def _load_and_sync_parameters(self):
        resume_checkpoint =  self.resume_checkpoint

        if resume_checkpoint: # resume checkpoint # 
            self.resume_step = self.parse_resume_step_from_filename(resume_checkpoint)
            # logger.log(f"loading model from checkpoint: {resume_checkpoint}...") # safe load ckpt 
            print(f"loading model from checkpoint: {resume_checkpoint}...")
            
            self.safe_load_ckpt(self.model, torch.load(resume_checkpoint, map_location=self.rl_device)
                                    # dist_util.load_state_dict(
                                    #     resume_checkpoint, map_location=dist_util.dev()
                                    # )
                                )
    
    
    
    #### -> samples wiht the key 'E' <-> the sampled/decoded next step actions ####
    def predict_single_step_deterministic(self, batch, use_t=None):
        
        tot_samples = {key: [] for key in batch}
        
        nn_bsz = batch['X'].shape[0]
        interest_keys = ['X', 'E', 'X_cond', 'E_cond', 'obj_task_setting', 'history_E_cond']
        interest_keys = [key for key in interest_keys if key in batch]
        # 
        
        samples = {
            'E': []
        }
        
        for i in range(0, nn_bsz, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            
            task_cond = {
                'X': micro['X_cond'], 'E': micro['E_cond'], 'history_E': micro['history_E_cond']
            }
            
            decoded_acts = self.model(task_cond)
            
            
            samples['E'].append(decoded_acts)
            
            for key in samples:
                if key not in tot_samples:
                    tot_samples[key] = samples[key]
                else:
                    tot_samples[key] += samples[key]
            

            
        print(f"tot_samples: {tot_samples.keys()}")
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
                print(f"key: {key}, shape: {tot_samples[key].shape}")
            except:
                continue
        
        tot_samples['X'] = tot_samples['E']    
        
        return tot_samples

    
    
        
    def predict_single_step(self, batch, use_t=None):
        # self.mp_trainer.zero_grad()
        
        # tot_dec_disp_e_along_normals = []
        # tot_dec_disp_e_vt_normals = []
        # tot_pred_joints_quant = []
        tot_samples = {key: [] for key in batch}
        
        nn_bsz = batch['X'].shape[0]
        interest_keys = ['X', 'E', 'X_cond', 'E_cond', 'obj_task_setting', 'history_E_cond']
        interest_keys = [key for key in interest_keys if key in batch]
        
        for key in batch:
            cur_val = batch[key]
            print(f"key: {key}, cur_val: {cur_val.size()}")
        # 
        for i in range(0, nn_bsz, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            # ## micro batch ##
            # rhand_joints = micro['rhand_joints'] #
            # micro_cond = cond # micro cond and cond ##
            ## predict single step ##
            
            # last_batch = (i + self.microbatch) >= nn_bsz #  #
            # t, weights = self.schedule_sampler.sample(micro['X'].shape[0], dist_util.dev()) # 
            

            # shape = {
            #     key: micro[key].shape for key in micro
            # }
            
            shape = {
                key: micro[key].shape for key in interest_keys
            }
            
            ## sample from the model --- the target sample should be in the sahpe of micro['X'].shape ## nn_bsz ## interest keys ##
            # shape = micro['X'].shape
            
            
            sample_fn = self.prior_model.p_sample_loop
            
            if self.AE_Diff:
                if self.train_AE:
                    sample_fn = self.prior_model.p_sample_loop_AE
                else: # 
                    sample_fn = self.prior_model.p_sample_loop_AE_Diff
            
            samples = sample_fn(
                self.ddp_model,  ### ddp omodel ? ###
                shape,
                noise=None,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                progress=True,
                use_t=use_t,
                data=micro if (self.AE_Diff or (use_t is not None) or (self.task_cond or self.training_setting == 'trajectory_translations_cond')) else None
            )

            for key in samples:
                tot_samples[key].append(samples[key])
            for key in micro:
                if key not in samples:
                    tot_samples[key].append(micro[key])

            
            
            
        print(f"tot_samples: {tot_samples.keys()}")
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
                print(f"key: {key}, shape: {tot_samples[key].shape}")
            except:
                continue
        ### predict the samples ##
        return tot_samples

    
    
    def predict_from_model(self, ):
        ### predict from the model? #
        ### predict from the model? #
        # predict from the model --- #
        # the current state is progress_buf 
        # the next target ts is progress_buf + 1
        
        
        
        history_progress_buf = []
        # [current progress, progress - 1, ..., progress - history_ws + 1]
        for i_ts in range(self.history_ws):
            cur_history_progress_buf = self.progress_buf - i_ts
            cur_history_progress_buf = torch.where(
                cur_history_progress_buf < 0, torch.zeros_like(cur_history_progress_buf), cur_history_progress_buf
            )
            history_progress_buf.append(cur_history_progress_buf)
        # progress buf # 
        history_progress_buf = reversed(history_progress_buf)
        history_progress_buf = list(history_progress_buf)
        tot_history_progress_buf = torch.stack(history_progress_buf, dim=1)
        tot_history_hand_dof = batched_index_select(self.history_hand_dof_buf, tot_history_progress_buf, dim=1)
        tot_history_obj_pos = batched_index_select(self.history_obj_pos_buf, tot_history_progress_buf, dim=1)
        # tot_history_obj_ornt = batched_index_select(self.history_obj_ornt_buf, tot_history_progress_buf, dim=1) ## nn_envs x nn_history_ts x xxx
        tot_history_obj_rot_euler = batched_index_select(self.history_obj_rot_euler_buf, tot_history_progress_buf, dim=1)
        # tot_history_info = torch.cat([tot_history_hand_dof, tot_history_obj_pos, tot_history_obj_ornt], dim=-1) # 
        
        ##### compute the future kinematics targets ##### # future kinematic targets  #
        future_kine_progress_buf = []
        for i_ts in range(1, self.future_ws + 1):
            cur_progress_buf = self.progress_buf + i_ts
            cur_progress_buf = torch.where( # 
                cur_progress_buf >= self.hand_qs_th_original_order.size(0), torch.zeros_like(cur_progress_buf) + self.hand_qs_th_original_order.size(0) - 1, cur_progress_buf
            )
            future_kine_progress_buf.append(cur_progress_buf)
        future_kine_progress_buf = torch.stack(future_kine_progress_buf, dim=1)
        expanded_hand_qs_th = self.hand_qs_th_original_order.unsqueeze(0).repeat(self.num_envs, 1, 1)
        future_kine_hand_qs = batched_index_select(expanded_hand_qs_th, future_kine_progress_buf, dim=1)
        expanded_obj_pos_th = self.goal_obj_trans_th.unsqueeze(0).repeat(self.num_envs, 1, 1)
        expanded_obj_ornt_th = self.goal_obj_rot_quat_th.unsqueeze(0).repeat(self.num_envs, 1, 1)
        expanded_obj_euler_th = self.goal_obj_rot_euler_th.unsqueeze(0).repeat(self.num_envs, 1, 1)
        future_kine_obj_pos = batched_index_select(expanded_obj_pos_th, future_kine_progress_buf, dim=1)
        # future_kine_obj_ornt = batched_index_select(expanded_obj_ornt_th, future_kine_progress_buf, dim=1)
        future_kine_obj_euler = batched_index_select(expanded_obj_euler_th, future_kine_progress_buf, dim=1)
        
        # current progress buf #
        current_kine_obj_pos = batched_index_select(expanded_obj_pos_th, self.progress_buf.unsqueeze(1), dim=1) # nn_envs x 1 x 3 # 
        current_kine_obj_pos = current_kine_obj_pos[:, 0, :]
        
        
        
        cur_obj_pos = self.object_pos # nn_envs x 3 # 
        
        if self.use_kine_obj_pos_canonicalization:
            cur_obj_pos = current_kine_obj_pos  
        
        print(f"object_pos: {cur_obj_pos}")
        
        if self.glb_rot_use_quat:
            future_hand_qs_rot_euler = future_kine_hand_qs[..., 3:6]
            future_hand_qs_rot_quat = quat_from_euler_xyz(future_hand_qs_rot_euler[..., 0], future_hand_qs_rot_euler[..., 1], future_hand_qs_rot_euler[..., 2])
            future_kine_hand_qs = torch.cat(
                [ 
                    future_kine_hand_qs[..., :3], future_hand_qs_rot_quat, future_kine_hand_qs[..., 6:]
                ], dim=-1
            )
            
            tot_history_hand_rot_euler = tot_history_hand_dof[:, :, 3:6]
            tot_history_hand_rot_quat = quat_from_euler_xyz(tot_history_hand_rot_euler[:, :, 0], tot_history_hand_rot_euler[:, :, 1], tot_history_hand_rot_euler[:, :, 2])
            tot_history_hand_dof = torch.cat(
                [
                    tot_history_hand_dof[:, :, :3], tot_history_hand_rot_quat, tot_history_hand_dof[:, :, 6:]
                ], dim=-1
            )
        
        ###### translational canonicalization ######
        tot_history_obj_pos = tot_history_obj_pos - cur_obj_pos.unsqueeze(1)
        tot_history_hand_dof[:, :, :3] = tot_history_hand_dof[:, :, :3] - cur_obj_pos.unsqueeze(1)
        future_kine_obj_pos = future_kine_obj_pos - cur_obj_pos.unsqueeze(1)
        future_kine_hand_qs[:, :, :3] = future_kine_hand_qs[:, :, :3] - cur_obj_pos.unsqueeze(1)
        ###### translational canonicalization ######
    


        
        # future_kine_info = torch.cat([future_kine_hand_qs, future_kine_obj_pos, future_kine_obj_ornt], dim=-1) # nn_envs x nn_future_ws x xxx
        # TODO: load object information #
        # TODO: load statistics #
        # TODO: scale them #
        # then segment the data_inst_tag to get the mesh file name #
        # data_inst_tag = self.object_name
        # self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
        # grab_mesh_fn = f"{data_inst_tag}.obj" # meshes and the grab mesh fn #
        # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
        # # get the object mesh #
        # obj_mesh = trimesh.load_mesh(grab_mesh_fn)
        # obj_verts = obj_mesh.vertices # nn_pts x 3
        # to_sample_fr_idxes = list(range(obj_verts.shape[0]))
        # while len(to_sample_fr_idxes) < self.maxx_nn_pts:
        #     to_sample_fr_idxes += list(range(obj_verts.shape[0]))
        # random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
        # random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
        # obj_verts = obj_verts[random_sampled_idxes] 
        # obj_verts = torch.from_numpy(obj_verts).float().to(self.device)
        
        #### get the object vertices ####
        obj_verts = self.obj_verts.clone()
        
        
        eps = 1e-6
        ### Scale the task cond information, query the model ###
        scaled_cond_obj_verts = (obj_verts - self.avg_obj_verts.unsqueeze(0)) / (self.std_obj_verts.unsqueeze(0) + eps)
        # cond robot hand qs # ## nn_envs x nn_history_ws x nn_hand_qs_dof
        
        if not self.glb_rot_use_quat:
            scaled_cond_hand_qs = (future_kine_hand_qs - self.avg_kine_hand_qs.unsqueeze(0).unsqueeze(0)) / (self.std_kine_hand_qs.unsqueeze(0).unsqueeze(0) + eps)
            scaled_cond_obj_rot_euler = (future_kine_obj_euler - self.avg_obj_rot_euler.unsqueeze(0).unsqueeze(0)) / (self.std_obj_rot_euler.unsqueeze(0).unsqueeze(0) + eps)
            scaled_cond_obj_transl = (future_kine_obj_pos - self.avg_obj_transl.unsqueeze(0).unsqueeze(0)) / (self.std_obj_transl.unsqueeze(0).unsqueeze(0) + eps)
            # scaled cond obj transl # 
            scaled_history_hand_qs = (tot_history_hand_dof - self.avg_hand_qs.unsqueeze(0).unsqueeze(0)) / (self.std_hand_qs.unsqueeze(0).unsqueeze(0) + eps)
            scaled_history_obj_rot_euler = (tot_history_obj_rot_euler - self.avg_obj_rot_euler.unsqueeze(0).unsqueeze(0)) / (self.std_obj_rot_euler.unsqueeze(0).unsqueeze(0) + eps)
            scaled_history_obj_transl = (tot_history_obj_pos - self.avg_obj_transl.unsqueeze(0).unsqueeze(0))  / (self.std_obj_transl.unsqueeze(0).unsqueeze(0) + eps)
        else:
            scaled_cond_hand_qs = future_kine_hand_qs
            scaled_cond_obj_rot_euler = future_kine_obj_euler
            scaled_cond_obj_transl = future_kine_obj_pos
            scaled_history_hand_qs = tot_history_hand_dof
            scaled_history_obj_rot_euler = tot_history_obj_rot_euler
            scaled_history_obj_transl = tot_history_obj_pos
            
        
        cond_concat_feat = torch.cat(
            [ scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], dim=-1
        )
        
        history_cond_feat = torch.cat(
            [ scaled_history_hand_qs, scaled_history_obj_transl, scaled_history_obj_rot_euler ], dim=-1
        )
        
        cond_dict = {
            'X_cond': scaled_cond_obj_verts.unsqueeze(0).repeat(self.num_envs, 1, 1), # .unsqueeze(0).unsqueeze(0),
            'E_cond': cond_concat_feat.unsqueeze(1),
            'history_E_cond': history_cond_feat.unsqueeze(1)
        }
        # slicing ws #
        X = torch.zeros_like(scaled_cond_hand_qs)
        E = torch.zeros_like(scaled_cond_hand_qs)
        cond_dict.update(
            {
                'X': X.unsqueeze(1), 'E': E.unsqueeze(1)
            }
        )
        
        
        if self.use_deterministic:
            tot_samples = self.predict_single_step_deterministic(cond_dict)
            
            tot_samples['E'] = tot_samples['E'].unsqueeze(1)
            tot_samples['X'] = tot_samples['E'].clone() 
            
        else:
            ### get the total samples ###
            tot_samples = self.predict_single_step(cond_dict)
        ## TODO: scale the sampled data to get the scaled data #
        ## Get hand qs and hand qtars
        sampled_X = tot_samples['X']
        sampled_E = tot_samples['E']
        
        ##### Rescale the X and E #####
        ## nn_bsz x nn_ts x nn_hand_qs_dof ###
        if not self.glb_rot_use_quat:
            sampled_X = (sampled_X * (self.std_hand_qs.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qs.unsqueeze(0).unsqueeze(0)
            sampled_E = (sampled_E * (self.std_hand_qtars.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qtars.unsqueeze(0).unsqueeze(0)
        
        ### shifted hand qs and qtars --- nn_envs x nn_ts x nn_hand_qs_dof ###
        print(f"sampled_X: {sampled_X.size()}, sampled_E: {sampled_E.size()}, cur_obj_pose: {cur_obj_pos.size()}")
        sampled_X[..., :3] = sampled_X[..., :3] + cur_obj_pos.unsqueeze(1).unsqueeze(1)
        sampled_E[..., :3] = sampled_E[..., :3] + cur_obj_pos.unsqueeze(1).unsqueeze(1)
        
        if self.glb_rot_use_quat:
            sampled_E_rot_quat = sampled_E[..., 3: 7]
            sampled_E_rot_euler = get_euler_xyz(sampled_E_rot_quat)
            sampled_E = torch.cat(
                [
                    sampled_E[..., :3], sampled_E_rot_euler, sampled_E[..., 7:]
                ], dim=-1
            )
        
        ## TODO: use that samples for the following controlling processes ##
        ##### load statistics and use the statistics for scaling #####
        
        sampled_dict = {
            'qs': sampled_X,
            'qtars': sampled_E
        }
        self.predicted_qtars = sampled_E # nn_envs x nn_steps x nn_dofs #
        self.predicted_qtars = self.predicted_qtars[:, :, : self.predict_ws]
        return sampled_dict



    # def _load_data_statistics(self, ):
    #     # stats_fn = f"diffusion/assets/data_statistics_ws_{self.slicing_ws}.npy"
    #     stats_fn = self.statistics_info_fn
    #     data_stats = np.load(stats_fn, allow_pickle=True).item()
    #     self.avg_hand_qs = data_stats['avg_hand_qs']
    #     self.std_hand_qs = data_stats['std_hand_qs']
    #     self.avg_hand_qtars = data_stats['avg_hand_qtars']
    #     self.std_hand_qtars = data_stats['std_hand_qtars']
    #     self.avg_kine_hand_qs = data_stats['avg_kine_hand_qs']
    #     self.std_kine_hand_qs = data_stats['std_kine_hand_qs']
    #     self.avg_obj_transl = data_stats['avg_obj_transl']
    #     self.std_obj_transl = data_stats['std_obj_transl']
    #     self.avg_obj_rot_euler = data_stats['avg_obj_rot_euler']
    #     self.std_obj_rot_euler = data_stats['std_obj_rot_euler']
        
    #     self.avg_hand_qs = torch.from_numpy(self.avg_hand_qs).float().to(self.device)
    #     self.std_hand_qs = torch.from_numpy(self.std_hand_qs).float().to(self.device)
    #     self.avg_hand_qtars = torch.from_numpy(self.avg_hand_qtars).float().to(self.device)
    #     self.std_hand_qtars = torch.from_numpy(self.std_hand_qtars).float().to(self.device)
    #     self.avg_kine_hand_qs = torch.from_numpy(self.avg_kine_hand_qs).float().to(self.device)
    #     self.std_kine_hand_qs = torch.from_numpy(self.std_kine_hand_qs).float().to(self.device)
    #     self.avg_obj_transl = torch.from_numpy(self.avg_obj_transl).float().to(self.device)
    #     self.std_obj_transl = torch.from_numpy(self.std_obj_transl).float().to(self.device)
    #     self.avg_obj_rot_euler = torch.from_numpy(self.avg_obj_rot_euler).float().to(self.device)
    #     self.std_obj_rot_euler = torch.from_numpy(self.std_obj_rot_euler).float().to(self.device)
    
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
        # nn_frames x 3 
        # nn_frames x 4
        
        eps = 1e-2
        #  # if the hand is close to the object --- add the hand pose guidance? #
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
        key = 'closest_training_data'
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
        if 'shadow_hand_dof_tars' in first_ts_sv_info:
            ts_to_hand_qs_np = self._load_optimized_traj_sorted_qtars(optimized_traj_fn)
            return ts_to_hand_qs_np
        
        
        tot_ts_list = list(isaac_sv_info.keys())
        tot_ts_list = sorted(tot_ts_list)
        minn_ts = min(tot_ts_list)
        maxx_ts = max(tot_ts_list)
        # for each val -- nn_envs x xxxx #
        ts_to_hand_qs = {}
        ts_to_obj_qs = {}
        idxx = 1000 ## # 
        maxx_lowest_z = -9999.0
        
        for idx in range(isaac_sv_info[maxx_ts]['object_pose'].shape[0]): # isaac sv info #
            cur_last_z = isaac_sv_info[maxx_ts]['object_pose'][idx][2]
            if cur_last_z > maxx_lowest_z and cur_last_z < 2.0:
                maxx_lowest_z = cur_last_z
                idxx = idx
        print("selected inst index and the optimized object lowest_z", idxx, maxx_lowest_z)
        
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
        
        
        hand_qs = save_info['robot_delta_states_weights_np'][self.start_frame : ]
        hand_qs = hand_qs[: , : self.nn_hand_dof]
        goal_obj_trans = save_info['object_transl'][: ][self.start_frame : ]
        goal_obj_rot_quat = save_info['object_rot_quat'][: ][self.start_frame : ]
        
        self.pre_hand_qs = hand_qs.copy()
        
        if self.dataset_type == 'taco':
            
            # x_offset = -0.05
            # y_offset = -0.01
            # goal_obj_trans[:, 0] = goal_obj_trans[:, 0] + x_offset
            # goal_obj_trans[:, 1] = goal_obj_trans[:, 1] + y_offset
            
            ed_frame = min(hand_qs.shape[0], self.max_episode_length)
            hand_qs = hand_qs[: ed_frame    ]
            goal_obj_trans = goal_obj_trans[: ed_frame]
            goal_obj_rot_quat = goal_obj_rot_quat[: ed_frame]
        
        # get the tasks and the taco # # # # load mocap info # # #
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
            link_name_to_poses = save_info['link_name_to_poses'] # 
            self.link_name_to_poses = link_name_to_poses # get link name to poses #
            
            for link_name in self.link_name_to_poses:
                self.link_name_to_poses[link_name][:, 2] -= self.ground_distance
                if self.add_table:
                    self.link_name_to_poses[link_name][:, 2] += self.table_z_dim
            
            self.link_name_to_poses_th = {}
            for link_name in self.link_name_to_poses:
                ##### get the link name to poses #####
                self.link_name_to_poses_th[link_name] = torch.from_numpy(self.link_name_to_poses[link_name]).float().to(self.rl_device)
            
            self.hand_palm_link_name = 'palm_link'
            self.thumb_tip_link_name = 'link_15_tip'
            self.index_tip_link_name = 'link_3_tip'
            self.middle_tip_link_name = 'link_7_tip'
            self.ring_tip_link_name = 'link_11_tip'
            self.hand_palm_world_poses = self.link_name_to_poses_th[self.hand_palm_link_name]  
            self.thumb_tip_world_poses = self.link_name_to_poses_th[self.thumb_tip_link_name]
            self.index_tip_world_poses = self.link_name_to_poses_th[self.index_tip_link_name]
            self.middle_tip_world_poses = self.link_name_to_poses_th[self.middle_tip_link_name]
            self.ring_tip_world_poses = self.link_name_to_poses_th[self.ring_tip_link_name]
            
        
        
        self.lift_fr = self._find_lifting_frame( goal_obj_trans  )
        self.lift_obj_pos = goal_obj_trans[self.lift_fr]
        self.lift_obj_pos_th = torch.from_numpy(self.lift_obj_pos).float().to(self.rl_device) 
        
        
        # TODO: in the new train pool file # # 
        # TODO: in the new train pool file, for each obj type --- find its neighbouring trajectory types; load the traj to optimized res; for each neighbouring traj, set th preoptimized traj to that traj #
        if self.pre_optimized_traj is not None and len(self.pre_optimized_traj) > 0 and os.path.exists(self.pre_optimized_traj):
            
            hand_qs = self._load_optimized_traj(self.pre_optimized_traj)
            currr_ws = min(hand_qs.shape[0], goal_obj_trans.shape[0])
            currr_ws = min(currr_ws, goal_obj_rot_quat.shape[0])
            hand_qs = hand_qs[: currr_ws]
            goal_obj_trans = goal_obj_trans[: currr_ws]
            goal_obj_rot_quat = goal_obj_rot_quat[: currr_ws]
            
            pass
        
        hand_qs[:, 2] -= self.ground_distance
        goal_obj_trans[:, 2] -= self.ground_distance
        
        ## TODO: reset the table's initial translations ##
        ## offset the hand qs ###
        if self.add_table:
            hand_qs[:, 2] += self.table_z_dim
            goal_obj_trans[:, 2] += self.table_z_dim
        
        
        self.hand_qs = hand_qs
        self.goal_obj_trans = goal_obj_trans ## 
        self.goal_obj_rot_quat = goal_obj_rot_quat
        print(f"==> Info loaded with hand_qs: {hand_qs.shape}, goal_obj_trans: {goal_obj_trans.shape}, goal_obj_rot_quat: {goal_obj_rot_quat.shape}")
        
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        print(f"joint_idxes_ordering: {joint_idxes_ordering}")
        print(f"joint_idxes_inversed_ordering: {joint_idxes_inversed_ordering}")
        
        
        
        # hand_qs_th, goal_obj_trans_th, goal_obj_rot_quat_th
        self.hand_qs_th = torch.from_numpy(hand_qs[:, joint_idxes_inversed_ordering]).float().to(self.rl_device)
        
        self.hand_qs_th_original_order = torch.from_numpy(hand_qs).float().to(self.rl_device)
        
        self.goal_obj_trans_th = torch.from_numpy(goal_obj_trans).float().to(self.rl_device)
        self.goal_obj_rot_quat_th = torch.from_numpy(goal_obj_rot_quat).float().to(self.rl_device)
        
        self.hand_qs = self.hand_qs[:, joint_idxes_inversed_ordering]
        
        # object transl and rot quat #
        # hand qs and hand qs # # object transl #
        
        if self.use_twostage_rew:
            self.cur_grasp_fr = self._find_grasp_frame(goal_obj_trans, goal_obj_rot_quat)
        
        # start grasping fr # # start grasping frame #
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
            
        
        self.goal_obj_rot_euler_th =  get_euler_xyz(self.goal_obj_rot_quat_th)
        self.goal_obj_rot_euler_th = torch.stack(
            [ self.goal_obj_rot_euler_th[0], self.goal_obj_rot_euler_th[1], self.goal_obj_rot_euler_th[2] ], dim=-1
        )
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

    ## create envs ##
    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        # plane_params.distance = self.ground_distance
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    ## right_hand_dist_thres ##
    def _create_envs(self, num_envs, spacing, num_per_row):
        object_scale_dict = self.cfg['env']['object_code_dict']
        
        if len(self.object_name) > 0: #
            if '_nf' in self.object_name: # from object name to the pure object name #
                pure_obj_name = "_".join(self.object_name.split('_')[:-2])
            else:
                pure_obj_name = self.object_name
            # object_scale_dict = { f'sem/{self.object_name}' : [1.0] }
            object_scale_dict = { f'sem/{pure_obj_name}' : [1.0] }
        
        
        self.object_code_list = list(object_scale_dict.keys())
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
        dataset_root_path = osp.join(assets_path, 'datasetv4.1')
        
        print(f"[Debug] object_code_list: {self.object_code_list}")
        
        
        for object_code in self.object_code_list: # 
            data_per_object = {}
            dataset_path = dataset_root_path + '/' + object_code
            data_num_list = os.listdir(dataset_path)
            for num in data_num_list: # qpos, scale, target hand rot, target hand pos
                data_dict = dict(np.load(os.path.join(dataset_path, num), allow_pickle=True))
                qpos = data_dict['qpos'] # .item()
                scale_inverse = data_dict['scale'] # .item()
                scale = round(1 / scale_inverse, 2)
                print(f"[Debug] scale: {scale}")
                # assert scale in [0.06, 0.08, 0.10, 0.12, 0.15]
                target_qpos = torch.from_numpy(qpos).float().to(self.device)
                target_hand_rot_xyz = torch.zeros((3, ), device=self.device) 
                target_hand_pos = torch.zeros((3, ), device=self.device)
                
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
                
                
                # self.init_hand_q = data_dict['qpos_init']
                self.init_hand_q = self.hand_qs[0, :]
                self.init_hand_q = torch.from_numpy(self.init_hand_q).float().to(self.device)
                
                
                init_obj_tarns = self.goal_obj_trans[0, :]
                init_obj_quat = self.goal_obj_rot_quat[0, :]
                
                
                self.obj_trans_init = torch.from_numpy(init_obj_tarns).float().to(self.device)
                self.obj_ornt_init = torch.from_numpy(init_obj_quat).float().to(self.device)
                
                obj_ornt_init_euler_xyz = R.from_quat(init_obj_quat).as_euler('xyz', degrees=False) # not th degress # 
                object_euler_xy = torch.tensor([obj_ornt_init_euler_xyz[0], obj_ornt_init_euler_xyz[1]], dtype=torch.float, device=self.device) # 
                object_init_z = torch.tensor([init_obj_tarns[2]], dtype=torch.float, device=self.device) ## ge the devie # 
                
                # 
                
                
                print(f"[Debug] object_init_z: {object_init_z}, object_euler_xy: {object_euler_xy}")
                

                # if object_init_z > 0.06:
                #     continue

                if scale in data_per_object:
                    data_per_object[scale]['target_qpos'].append(target_qpos)
                    data_per_object[scale]['target_hand_pos'].append(target_hand_pos)
                    data_per_object[scale]['target_hand_rot'].append(target_hand_rot) # trget hand root #
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
        
        self.target_qpos = torch.zeros((self.num_envs, 22), device=self.device)
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
        shadow_hand_asset_file = f"allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf"
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
        asset_options.flip_visual_attachments = False # fixed base link #
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

        self.num_shadow_hand_actuators = self.num_shadow_hand_dofs

        self.actuated_dof_indices = [i for i in range(self.num_shadow_hand_dofs)]

 
        # table_z_dim # 
        # set shadow_hand dof properties # 
        shadow_hand_dof_props = self.gym.get_asset_dof_properties(shadow_hand_asset)

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

        ## TODO: decompose the object and add into the mesh_data_scaled #
        mesh_path = osp.join(assets_path, 'meshdatav3_scaled')
        for object_id, object_code in enumerate(self.object_code_list):
            # load manipulated object and goal assets
            object_asset_options = gymapi.AssetOptions()
            # object_asset_options.density = 500
            object_asset_options.density = self.rigid_obj_density
            object_asset_options.fix_base_link = False
            print(f"disable_obj_gravity: {self.disable_obj_gravity}")
            object_asset_options.disable_gravity = self.disable_obj_gravity
            object_asset_options.use_mesh_materials = True
            object_asset_options.mesh_normal_mode = gymapi.COMPUTE_PER_VERTEX
            object_asset_options.override_com = True
            object_asset_options.override_inertia = True
            object_asset_options.vhacd_enabled = True
            object_asset_options.vhacd_params = gymapi.VhacdParams()
            object_asset_options.vhacd_params.resolution = 300000
            object_asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            object_asset = None
            
            for obj_id, scale_id in object_scale_idx_pairs:
                if obj_id == object_id:
                    scale_str = scale2str[self.id2scale[scale_id]]
                    scaled_object_asset_file = object_code + f"/coacd/coacd_{scale_str}.urdf"
                    print(f"scaled_object_asset_file: {scaled_object_asset_file}")
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
        shadow_hand_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.0) # gymapi.Vec3(0.1, 0.1, 0.65)
        # shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(1.57, 0, 0)  # gymapi.Quat().from_euler_zyx(0, -1.57, 0)
        shadow_hand_start_pose.r = gymapi.Quat().from_euler_zyx(0.0, 0, 0)

        object_start_pose = gymapi.Transform()
        # object_start_pose.p = gymapi.Vec3(0.0, 0.0, 0.6 + 0.1)  # gymapi.Vec3(0.0, 0.0, 0.72)
        obj_init_x = self.obj_trans_init[0].item()
        obj_init_y = self.obj_trans_init[1].item()
        obj_init_z = self.obj_trans_init[2].item()
        object_start_pose.p = gymapi.Vec3(obj_init_x, obj_init_y, obj_init_z)  # gymapi.Vec3(0.0, 0.0, 0.72)
        # object_start_pose.r = gymapi.Quat().from_euler_zyx(0, 0, 0)  # gymapi.Quat().from_euler_zyx(1.57, 0, 0) # from euler zyx #
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
        # by dfault
        # table_pose.p = gymapi.Vec3(0.0, 0.0, 0.5 * table_dims.z)
        
        if 'transformed' in self.mocap_sv_info_fn:
            table_pose.p = gymapi.Vec3(0.5 * table_dims.x, -0.5 * table_dims.y, 0.5 * table_dims.z)
        else:
            table_pose.p = gymapi.Vec3(-0.5 * table_dims.x, 0.5 * table_dims.y, 0.5 * table_dims.z)
        # table_pose.p = gymapi.Vec3(0.0, 0.0, -0.5 * table_dims.z)
        # table_pose.p = gymapi.Vec3(0.0, 0.0, -1.0 * table_dims.z)
        table_pose.r = gymapi.Quat().from_euler_zyx(-0., 0, 0)

        # compute aggregate size # num shadowh hand bodies # 
        max_agg_bodies = self.num_shadow_hand_bodies * 1 + 2 * self.num_object_bodies + 1
        max_agg_shapes = self.num_shadow_hand_shapes * 1 + 2 * self.num_object_shapes + 1

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
        
        if self.use_fingertips:
            body_names = {
                # 'wrist': 'robot0:wrist',
                'palm': 'palm_link',
                'thumb': 'link_15_tip',
                'index': 'link_3_tip',
                'middle': 'link_7_tip',
                'ring': 'link_11_tip',
                # 'little': 'robot0:lfdistal'
            }
        else:
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

        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            if self.aggregate_mode >= 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # shadow hand actor --- # 
            # add hand - collision filter = -1 to use asset collision filters set in mjcf loader
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
            id = int(i / self.num_envs * len(self.object_code_list))
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

            self.visual_feat_buf[i] = self.visual_feat_data[id][scale_id]

            # add object
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

            # add goal object
            # goal_asset_dict[id][scale_id]
            goal_handle = self.gym.create_actor(env_ptr, goal_asset, goal_start_pose, "goal_object", i + self.num_envs, 0, 0)
            goal_object_idx = self.gym.get_actor_index(env_ptr, goal_handle, gymapi.DOMAIN_SIM)
            self.goal_object_indices.append(goal_object_idx)
            self.gym.set_actor_scale(env_ptr, goal_handle, 1.0)

            # add table
            #### NOTE: we have disabled table here ### create the table handle from the table asset #
            if self.add_table: # add table --- 
                table_handle = self.gym.create_actor(env_ptr, table_asset, table_pose, "table", i, -1, 0)
                self.gym.set_rigid_body_texture(env_ptr, table_handle, 0, gymapi.MESH_VISUAL, table_texture_handle) # table texture handle #
                table_idx = self.gym.get_actor_index(env_ptr, table_handle, gymapi.DOMAIN_SIM)
                self.table_indices.append(table_idx)


                table_shape_props = self.gym.get_actor_rigid_shape_properties(env_ptr, table_handle)
                table_shape_props[0].friction = 1
                self.gym.set_actor_rigid_shape_properties(env_ptr, table_handle, table_shape_props)
            
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

        # init hand q 
        self.shadow_hand_default_dof_pos = self.init_hand_q
        
        if self.use_canonical_state:
            # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
            # self.shadow_hand_default_dof_pos[2] = 0.2
            if self.use_unified_canonical_state:
                self.shadow_hand_default_dof_pos = torch.zeros_like(self.shadow_hand_default_dof_pos)
                self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
                self.shadow_hand_default_dof_pos[2] = 0.2
                self.shadow_hand_default_dof_pos[1] = 0.0
            else:
                print(f"setting the canonical state")
                # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
                # self.shadow_hand_default_dof_pos[2] = 0.2
                # self.shadow_hand_default_dof_pos[1] = -0.07
                
                # self.shadow_hand_default_dof_pos[4] = 0.5 * np.pi
                self.shadow_hand_default_dof_pos[2] += 0.01
                # self.shadow_hand_default_dof_pos[1] = -0.07 # use the canonical state #
                self.shadow_hand_default_dof_pos[6:] = 0.0
        
        ###### ######
        # set dof state tensor index #
        # self.gym.set_dof_state_tensor_indexed(self.sim, # two hands #
        #                                       gymtorch.unwrap_tensor(self.dof_state),
        #                                       gymtorch.unwrap_tensor(hand_indices), len(env_ids))


        self.object_init_state = to_torch(self.object_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_init_state = to_torch(self.goal_init_state, device=self.device, dtype=torch.float).view(self.num_envs, 13)
        self.goal_states = self.goal_init_state.clone()
        # self.goal_pose = self.goal_states[:, 0:7]
        # self.goal_pos = self.goal_states[:, 0:3]
        # self.goal_rot = self.goal_states[:, 3:7]
        # self.goal_states[:, self.up_axis_idx] -= 0.04
        self.goal_init_state = self.goal_states.clone()
        self.hand_start_states = to_torch(self.hand_start_states, device=self.device).view(self.num_envs, 13)
        self.fingertip_handles = to_torch(self.fingertip_handles, dtype=torch.long, device=self.device)
        self.hand_indices = to_torch(self.hand_indices, dtype=torch.long, device=self.device)
        self.object_indices = to_torch(self.object_indices, dtype=torch.long, device=self.device)
        self.goal_object_indices = to_torch(self.goal_object_indices, dtype=torch.long, device=self.device)
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
            self.object_linvel, self.object_angvel, self.goal_linvel, self.goal_angvel,
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
        
        #
        
        # lift_obj_pos_th # compute reward #
        
        # self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos #
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:], self.hand_palm_fingers_obj_contact_buf[:], self.right_hand_dist_buf[:] = compute_reward_func( # compute hand tracking reward ##
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.object_linvel, self.object_angvel, self.goal_linvel, self.goal_angvel,
            self.lift_obj_pos_th,
            self.goal_pos, self.goal_rot, self.goal_lifting_pos,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos, # 
            self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos ,
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.rew_obj_pose_coef, self.goal_dist_thres, self.lifting_separate_stages, self.reach_lifting_stage, self.strict_lifting_separate_stages
        )
        
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
        
        self.rew_buf[:], self.reset_buf[:], self.reset_goal_buf[:], self.progress_buf[:], self.successes[:], self.current_successes[:], self.consecutive_successes[:] = compute_reward_func( # compute hand tracking reward ##
            self.object_init_z, self.delta_qpos, self.delta_target_hand_pos, self.delta_target_hand_rot,
            self.object_id_buf, self.dof_pos, self.rew_buf, self.reset_buf, self.reset_goal_buf,
            self.progress_buf, self.successes, self.current_successes, self.consecutive_successes,
            self.max_episode_length, self.object_pos, self.object_handle_pos, self.object_back_pos, self.object_rot,
            self.object_linvel, self.object_angvel, self.goal_linvel, self.goal_angvel,
            self.goal_pos, self.goal_rot, self.goal_lifting_pos,
            self.right_hand_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos,
            self.right_hand_th_pos, # 
            self.dist_reward_scale, self.rot_reward_scale, self.rot_eps, self.actions, self.action_penalty_scale,
            self.success_tolerance, self.reach_goal_bonus, self.fall_dist, self.fall_penalty,
            self.max_consecutive_successes, self.av_factor,self.goal_cond, hand_up_threshold_1, hand_up_threshold_2 , len(self.fingertips), self.w_obj_ornt, self.w_obj_vels, self.separate_stages, self.hand_pose_guidance_glb_trans_coef, self.hand_pose_guidance_glb_rot_coef, self.hand_pose_guidance_fingerpose_coef, self.rew_finger_obj_dist_coef, self.rew_delta_hand_pose_coef, self.rew_obj_pose_coef, self.goal_dist_thres
        )

        self.extras['successes'] = self.successes
        self.extras['current_successes'] = self.current_successes
        self.extras['consecutive_successes'] = self.consecutive_successes

        print(f"reset_buf: {self.reset_buf}")
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
        ### TODO: about the link body pose and the object global pose? ###
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)


        # shadow_hand_dof_pos # 
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        # self. #
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
        
        self.goal_lifting_pos = self.goal_states[:, 0:3] # goal lifting 
        
        
        if self.dataset_type == 'taco':
            progress_buf_indexes = torch.where(self.progress_buf >= self.hand_palm_world_poses.size(0), self.hand_palm_world_poses.size(0) - 1 + torch.zeros_like(self.progress_buf), self.progress_buf)
            # self.gt_hand_palm_pos, self.gt_hand_thumb_pos, self.gt_hand_index_pos, self.gt_hand_middle_pos, self.gt_hand_ring_pos #
            self.gt_hand_palm_pos = self.hand_palm_world_poses[progress_buf_indexes]
            self.gt_hand_thumb_pos = self.thumb_tip_world_poses[progress_buf_indexes]
            self.gt_hand_index_pos = self.index_tip_world_poses[progress_buf_indexes]
            self.gt_hand_middle_pos = self.middle_tip_world_poses[progress_buf_indexes]
            self.gt_hand_ring_pos = self.ring_tip_world_poses[progress_buf_indexes]
            
            
            
        
        maxx_progress = torch.max(self.progress_buf)
        minn_progress = torch.min(self.progress_buf)
        # print(f"maxx_progress: {maxx_progress}, minn_progress: {minn_progress}, goal_obj_trans_th: {self.goal_obj_trans_th.size()}")
        
        # goal obj trans # 
        cur_goal_pos = self.goal_obj_trans_th[self.progress_buf]
        cur_goal_rot = self.goal_obj_rot_quat_th[self.progress_buf]
        
        
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
            self.goal_pos  = self.goal_pos_ref # get the goal rot, pos, and pose #
        
        
        if self.use_twostage_rew:
            grasping_frame_obj_pos = self.goal_obj_trans_th[self.cur_grasp_fr] + to_torch([0.0, 0.0, 0.1], device=self.device)
            grasping_frame_obj_ornt = self.goal_obj_rot_quat_th[self.cur_grasp_fr]
            expanded_grasping_frame_obj_pos = grasping_frame_obj_pos.unsqueeze(0).repeat(self.num_envs, 1)
            expanded_grasping_frame_obj_ornt = grasping_frame_obj_ornt.unsqueeze(0).repeat( self.num_envs, 1)
            grasp_manip_stages_flag_pos = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, 3)
            grasp_manip_stages_flag_rot = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, 4)
            
            if self.use_real_twostage_rew:
                self.goal_pos = torch.where(  # grasping stages pos 
                    grasp_manip_stages_flag_pos, expanded_grasping_frame_obj_pos, self.goal_pos
                )
                self.goal_rot = torch.where(
                    grasp_manip_stages_flag_rot, expanded_grasping_frame_obj_ornt, self.goal_rot
                )   
            else:
                self.goal_pos = expanded_grasping_frame_obj_pos # 
                self.goal_rot = expanded_grasping_frame_obj_ornt # obj ornt #
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
            
            
        
        # goal_linvel, goal_angvel #
        cur_goal_lin_vels = self.goal_obj_lin_vels_th[self.progress_buf]
        cur_goal_ang_vels = self.goal_obj_ang_vels_th[self.progress_buf]
        self.goal_linvel = cur_goal_lin_vels
        self.goal_angvel = cur_goal_ang_vels
        self.goal_vels = torch.cat(
            [self.goal_linvel, self.goal_angvel], dim=-1
        )

        # fingertip state #
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        def world2obj_vec(vec):
            return quat_apply(quat_conjugate(self.object_rot), vec - self.object_pos)
        def obj2world_vec(vec):
            return quat_apply(self.object_rot, vec) + self.object_pos
        def world2obj_quat(quat):
            return quat_mul(quat_conjugate(self.object_rot), quat)
        def obj2world_quat(quat):
            return quat_mul(self.object_rot, quat)

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
        

        ### current target hand pose, and the difference from the reference hand pos ###
        cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        self.ori_cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        
        ### next progress buffer ###
        nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.hand_qs_th.size(0) - 1)
        nex_hand_qpos_ref = self.hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        self.nex_hand_qpos_ref = nex_hand_qpos_ref
        
        if self.use_twostage_rew:
            grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
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
            lifting_frame_hand_qpos = self.hand_qs_th[self.lift_fr]
            expanded_lifting_frame_hand_qpos = lifting_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
            lifting_manip_stages_flag_qpos = (self.reach_lifting_stage.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
            self.nex_hand_qpos_ref = torch.where(
                lifting_manip_stages_flag_qpos, expanded_lifting_frame_hand_qpos, self.nex_hand_qpos_ref
            )
            cur_hand_qpos_ref = torch.where(
                lifting_manip_stages_flag_qpos, expanded_lifting_frame_hand_qpos, cur_hand_qpos_ref
            )
            self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
            
        
        
        # shadow_hand_dof_pos, object_pos #
        # self.shadow_hand_dof_pos # 
        # self.object_pose #
        # shadow hand dof vels #
        
        self.object_pose_np = self.object_pose.detach().cpu().numpy()
        self.shadow_hand_dof_pos_np = self.shadow_hand_dof_pos.detach().cpu().numpy()
        self.target_qpos_np = self.cur_targets.detach().cpu().numpy()[:, : self.num_shadow_hand_dofs]
        self.shadow_hand_dof_vel_np = self.shadow_hand_dof_vel.detach().cpu().numpy() #
        self.object_linvel_np = self.object_linvel.detach().cpu().numpy()
        self.object_angvel_np = self.object_angvel.detach().cpu().numpy()
        # so that the obs include all obs buffer ? # all obs buffer ? #
        self.ts_to_hand_obj_states[self.ref_ts] = {
            'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np,
            'shadow_hand_dof_tars': self.target_qpos_np,
            'object_pose': self.object_pose_np,
            'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np,
            'object_linvel': self.object_linvel_np,
            'object_angvel': self.object_angvel_np,
        }
        
        print(f"obj_trans_init: {self.obj_trans_init}")
        print(f"compute observation object_pose_np: {self.object_pose_np}")
        # self.ts_to_hand_obj_states[self.ref_ts] # object angvel np # 
        # object linvel np #
        
        # self.delta_qpos = self.shadow_hand_dof_pos - self.target_qpos
        self.compute_full_state()

        if self.asymmetric_obs: 
            self.compute_full_state(True)
        
        
        # only set the relevant reference motions in the current progress buf ?
        # history_obj_pos_buf, history_obj_ornt_buf, history_obj_rot_euler_buf
        
        ###### Update history buffers --- hand_qs, object_pos, object_ornt, object_rot_euler #######
        history_episode_range = torch.arange(self.max_episode_length, device=self.device).unsqueeze(0).repeat(self.num_envs, 1) # nn_envs x nn_max_episod
        expanded_history_episode_range = history_episode_range.unsqueeze(-1).repeat(1, 1, self.hand_qs_th.size(-1)) # nn envs x nn_max_episode x nn_ref_qs
        expanded_progress_buf = self.progress_buf.unsqueeze(-1).unsqueeze(-1).repeat(1, self.max_episode_length, self.hand_qs_th.size(-1)) # nn_envs x nn_max_episode x nn_ref_qs
        # self.history_hand_dof_buf
        expanded_cur_hand_dof_state = self.shadow_hand_dof_pos.unsqueeze(1).repeat(1, self.max_episode_length, 1)
        #### Update history hand dof buf ####
        self.history_hand_dof_buf = torch.where(
            expanded_history_episode_range == expanded_progress_buf, expanded_cur_hand_dof_state, self.history_hand_dof_buf
        )
        expanded_history_episode_range_obj = history_episode_range.unsqueeze(-1).repeat(1, 1, 7) # nn_envs x nn_max_episode x 7
        expanded_progress_buf_obj = self.progress_buf.unsqueeze(-1).unsqueeze(-1).repeat(1, self.max_episode_length, 7) # nn_envs x nn_max_episode x 7
        # expanded_cur_obj_state = self.object_pose.unsqueeze(1).repeat(1, self.max_episode_length, 1)
        expanded_cur_obj_pos = self.object_pos.unsqueeze(1).repeat(1, self.max_episode_length, 1)
        expanded_cur_obj_ornt = self.object_rot.unsqueeze(1).repeat(1, self.max_episode_length, 1)
        self.history_obj_pos_buf = torch.where(
            expanded_history_episode_range_obj[..., :3] == expanded_progress_buf_obj[..., :3], expanded_cur_obj_pos, self.history_obj_pos_buf
        )
        self.history_obj_ornt_buf = torch.where(
            expanded_history_episode_range_obj[..., 3:] == expanded_progress_buf_obj[..., 3:], expanded_cur_obj_ornt, self.history_obj_ornt_buf
        )
        self.object_rot_euler = get_euler_xyz(self.object_rot)
        self.object_rot_euler = torch.stack([self.object_rot_euler[0], self.object_rot_euler[1], self.object_rot_euler[ 2]], dim=-1)
        expanded_object_rot_euler = self.object_rot_euler.unsqueeze(1).repeat(1, self.max_episode_length, 1).contiguous()
        
        self.history_obj_rot_euler_buf = torch.where(
            expanded_history_episode_range_obj[..., :3] == expanded_progress_buf_obj[..., :3], expanded_object_rot_euler, self.history_obj_rot_euler_buf
        )
        ###### Update history buffers --- hand_qs, object_pos, object_ornt, object_rot_euler #######
        

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

    def compute_full_state(self, asymm_obs=False):

        self.get_unpose_quat()

        # 2 * nn_hand_dofs + 13 * num_fingertips + 6 + nn_hand_dofs + 16 + 7 + nn_hand_dofs
        # unscale to (-1，1)
        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        num_ft_force_torques = 6 * int(self.num_fingertips)  # 30 ##

        # 0:66
        self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos,
                                                               self.shadow_hand_dof_lower_limits,
                                                               self.shadow_hand_dof_upper_limits)
        self.obs_buf[:,self.num_shadow_hand_dofs:2 * self.num_shadow_hand_dofs] = self.vel_obs_scale * self.shadow_hand_dof_vel
        
        if self.obs_type == "full_state" or asymm_obs:
            self.obs_buf[:,2 * self.num_shadow_hand_dofs:3 * self.num_shadow_hand_dofs] = self.force_torque_obs_scale * self.dof_force_tensor[:, :self.num_shadow_hand_dofs]
        
            fingertip_obs_start = 3 * self.num_shadow_hand_dofs
        else:
            fingertip_obs_start = 2 * self.num_shadow_hand_dofs
        
        aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
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
        
        
        if self.tight_obs:
            # right_hand_rot
            self.obs_buf[:, hand_pose_start: hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
            euler_xyz = get_euler_xyz(self.unpose_quat(self.right_hand_rot))
        else:
            self.obs_buf[:, hand_pose_start:hand_pose_start + 3] = self.unpose_point(self.right_hand_pos)
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

        obj_obs_start = action_obs_start + self.num_shadow_hand_dofs  # 144
        # 191:207 object_pose, goal_pos
        self.obs_buf[:, obj_obs_start:obj_obs_start + 3] = self.unpose_point(self.object_pose[:, 0:3])
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

        
        # prue state wref #
        if self.obs_type == 'pure_state_wref_wdelta' and self.use_kinematics_bias_wdelta:
            nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
            unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
            
            cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
            self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs]
            
        elif self.obs_type == 'pure_state_wref':
            nex_ref_start = hand_goal_start + self.num_shadow_hand_dofs
            
            unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
            
        elif not self.obs_type == 'pure_state':
            
            # 236: visual feature 
            visual_feat_start = hand_goal_start + self.num_shadow_hand_dofs #  29
            
            # 236: 300: visual feature # # obs buf #
            self.obs_buf[:, visual_feat_start: visual_feat_start + 64] = 0.1 * self.visual_feat_buf
            self.obs_buf[:, visual_feat_start + 64: 300] = 0.0
        # use relative controls #
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
        
        # if self.randomize:
        #     self.apply_randomizations(self.randomization_params)

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

        # reset shadow hand
        delta_max = self.shadow_hand_dof_upper_limits - self.shadow_hand_dof_default_pos
        delta_min = self.shadow_hand_dof_lower_limits - self.shadow_hand_dof_default_pos
        rand_delta = delta_min + (delta_max - delta_min) * rand_floats[:, 5:5 + self.num_shadow_hand_dofs]

        pos = self.shadow_hand_default_dof_pos + self.reset_dof_pos_noise * rand_delta
        self.shadow_hand_dof_pos[env_ids, :] = pos # env_ids #

        self.shadow_hand_dof_vel[env_ids, :] = self.shadow_hand_dof_default_vel + \
                                               self.reset_dof_vel_noise * rand_floats[:, 5 + self.num_shadow_hand_dofs:5 + self.num_shadow_hand_dofs * 2]

        # shadow_hand_default_dof_pos #
        # self.dof_state[:, : self.]
        self.prev_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        self.cur_targets[env_ids, :self.num_shadow_hand_dofs] = pos
        
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
        # all indices #
        # all_indices = torch.unique(torch.cat([all_hand_indices, # 
        #                                       self.object_indices[env_ids],
        #                                       self.goal_object_indices[env_ids], # to torch.int32 #
        #                                       self.table_indices[env_ids], ]).to(torch.int32))
        all_indices = torch.unique(torch.cat([all_hand_indices,
                                              self.object_indices[env_ids],
                                              self.goal_object_indices[env_ids], ]).to(torch.int32))
        # state tensor indexed #
        self.gym.set_actor_root_state_tensor_indexed(self.sim,gymtorch.unwrap_tensor(self.root_state_tensor),
                                                     gymtorch.unwrap_tensor(all_indices), len(all_indices))
        #TODO: do we need to reset the table? #

        if self.random_time:
            ## TODO: waht's the role of random-time here ##
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
        
        # add 
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
             
            # logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy" # the hand up reward ? # 
            logging_sv_ts_to_obs_fn = f"ts_to_hand_obj_obs_reset_{self.reset_nn}.npy"
            logging_sv_ts_to_obs_fn = os.path.join(self.exp_logging_dir, logging_sv_ts_to_obs_fn)
            np.save(logging_sv_ts_to_obs_fn, self.ts_to_hand_obj_states) # save the ts_to_hand_obj_states #
            print(f"save the ts_to_hand_obj_states to {logging_sv_ts_to_obs_fn}")

    def pre_physics_step(self, actions):
        # pre physics step ##
        ## TODO: after each step, we should add the observed kinemati demonstrations to the history progress buffer ##
        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        goal_env_ids = self.reset_goal_buf.nonzero(as_tuple=False).squeeze(-1)

        # if only goals need reset, then call set API 
        if len(goal_env_ids) > 0 and len(env_ids) == 0: ## goal env ids and the env ids ##
            self.reset_target_pose(goal_env_ids, apply_reset=True) #### goal env ids and env ids #### ## goal env ids ##
        # if goals need reset in addition to other envs, call set API in reset() #
        elif len(goal_env_ids) > 0:
            self.reset_target_pose(goal_env_ids)

        if len(env_ids) > 0:
            self.reset(env_ids, goal_env_ids)

        # use_kinematics bias weights delta #
        
        if not self.kinematics_only:
           
            if self.ref_ts == 0 or self.predicted_qtars.size(2) == 0:
                self.predict_from_model()
                pred_qtars = self.predicted_qtars[:, :, 0] # nn_envs x nn_hand_qs #
                self.predicted_qtars = self.predicted_qtars[:, :, 1:] # nn_envs x nn_ts x nn_hand_qs #
            else:
                pred_qtars = self.predicted_qtars[:, :, 0] # nn_envs x nn_hand_qs #
                self.predicted_qtars = self.predicted_qtars[:, :, 1:] # nn_envs x nn_ts x nn_hand_qs #
            # pred_samples = self.predict_from_model()
            # pred_qtars = pred_samples['qtars' ][:, 0] # nn_envs x nn_ts x nn_hand_qs #
            print(f"predicted_qtars: {self.predicted_qtars.size()}")
            
            # print() # # print # # print #
        cur_progress = self.progress_buf # [0]
        print(f"current progress: {cur_progress}")
        
        
        # actions[:, 0:3] = self.pose_vec(actions[:, 0:3]) ##
        # actions[:, 3:6] = self.pose_vec(actions[:, 3:6]) ##
        
        self.actions = actions.clone().to(self.device)


        if self.use_kinematics_bias_wdelta: # bias delta #
            # print(f"self.use_kinematics_bias_wdelta: {self.use_kinematics_bias_wdelta}")
            # prev_delta_targets # 
            increased_progress_buf = self.progress_buf + 1
            increased_progress_buf = torch.clamp(increased_progress_buf, min=0, max=self.hand_qs.shape[0] - 1) # 
            # get the kinematicsof the increaesd progres buf as the kinematics bias # 
            # ctl_kinematics_bias = self.hand_qs[increased_progress_buf] # - self.shadow_hand_dof_pos
            # hand_qs_th # training data from that #
            
            ctl_kinematics_bias = self.hand_qs_th[increased_progress_buf]
            
            if self.use_twostage_rew:
                grasp_frame_hand_qpos = self.hand_qs_th[self.cur_grasp_fr]
                expanded_grasp_frame_hand_qpos = grasp_frame_hand_qpos.unsqueeze(0).repeat(self.num_envs, 1)
                grasp_manip_stages_flag = (self.grasp_manip_stages.int() == 0).unsqueeze(-1).repeat(1, self.nex_hand_qpos_ref.size(-1))
                ctl_kinematics_bias = torch.where(
                    grasp_manip_stages_flag, expanded_grasp_frame_hand_qpos, ctl_kinematics_bias
                )
            
            # prev_detlat_targets # prev delta targets #
            delta_delta_targets = self.shadow_hand_dof_speed_scale_tsr.unsqueeze(0) * self.dt * self.actions 
            cur_delta_targets = self.prev_delta_targets[:, self.actuated_dof_indices] + delta_delta_targets
            

            # cur delta targets #
            self.cur_delta_targets[:, self.actuated_dof_indices] = cur_delta_targets
            self.prev_delta_targets[:, self.actuated_dof_indices] = self.cur_delta_targets[:, self.actuated_dof_indices]
            
            if self.kinematics_only:
                cur_targets = ctl_kinematics_bias
            else:
                cur_targets = ctl_kinematics_bias + self.cur_delta_targets[:, self.actuated_dof_indices]
            
            if not self.kinematics_only:
                cur_targets = pred_qtars[:, 0, :] # get the pred_qtars # ## 
                
            self.cur_targets[:, self.actuated_dof_indices] = tensor_clamp(cur_targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
            
            
        # if we use the kinematics motion as the bis # 
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

        elif self.use_relative_control: # use relative control #
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

        # prev targets an the current jarets #
        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        all_hand_indices = torch.unique(torch.cat([self.hand_indices]).to(torch.int32))
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self.prev_targets),
                                                        gymtorch.unwrap_tensor(all_hand_indices), len(all_hand_indices))

    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1
        
        if self.use_twostage_rew:
            self.grasping_progress_buf += 1

        self.ref_ts += 1

        self.compute_observations()
        self.compute_reward(self.actions)
        
        if self.test: # test the test setting #
            # if self.ref_ts >= self.max_episode_length - 3: # try save #
            self.try_save()
            
            if self.reset_nn > 1:
                exit(0)

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
        max_episode_length: float, object_pos, object_handle_pos, object_back_pos, object_rot,
        object_linvel, object_angvel, goal_linvel, goal_angvel,
        target_pos, target_rot, target_lifting_pos,
        right_hand_pos, right_hand_ff_pos, right_hand_mf_pos, right_hand_rf_pos, right_hand_th_pos,
        dist_reward_scale: float, rot_reward_scale: float, rot_eps: float,
        actions, action_penalty_scale: float,
        success_tolerance: float, reach_goal_bonus: float, fall_dist: float,
        fall_penalty: float, max_consecutive_successes: int, av_factor: float, goal_cond: bool, hand_up_threshold_1: float, hand_up_threshold_2: float, num_fingers: int, w_obj_ornt: bool, w_obj_vels: bool, separate_stages: bool, hand_pose_guidance_glb_trans_coef: float, hand_pose_guidance_glb_rot_coef: float , hand_pose_guidance_fingerpose_coef: float, rew_finger_obj_dist_coef: float, rew_delta_hand_pose_coef: float, rew_obj_pose_coef: float, goal_dist_thres: float 
):
    # Distance from the hand to the object ## original goal pos ##
    # target hand pos, object pos ## original goal pos ##
    # 
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
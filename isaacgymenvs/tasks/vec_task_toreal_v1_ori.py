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
import os
import time
from datetime import datetime
from os.path import join
from typing import Dict, Any, Tuple, List, Set

import gym
from gym import spaces

from isaacgym import gymtorch, gymapi
from isaacgymenvs.utils.torch_jit_utils import to_torch
from isaacgymenvs.utils.dr_utils import get_property_setter_map, get_property_getter_map, \
    get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import torch
import numpy as np
import operator, random
from copy import deepcopy
from isaacgymenvs.utils.utils import nested_dict_get_attr, nested_dict_set_attr

from collections import deque

import sys

import abc
from abc import ABC

import pytorch_kinematics as pk

from utils.torch_jit_utils import *

# Janebek
from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
sys.path.append('/home/xymeow/xueyi/LEAP_Hand_API/python')
from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
import tf.transformations


class LeapNode:
    def __init__(self):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 800
        self.kI = 0
        self.kD = 200
        
        
        # self.kP = 100
        # self.kI = 0
        # self.kD = 4
        
        # self.kP = 25
        # self.kI = 0
        # self.kD = 4
        
        # # kPP # kP # kP # kP # kP # 
        # self.kP = 8
        # self.kI = 0
        # self.kD = 2
        
        self.curr_lim = 300
        # self.curr_lim = 500
        
        init_pos = [ 0.1451,  0.0797,  0.5086,  0.4145,  0.4700,  0.7530,  1.2950, -0.2015,
         0.2630, -0.1050,  0.3566,  0.2325,  0.3979, -0.1960,  0.2432,  0.3852]
        
        joint_idxes_ordering = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        
        init_pos = np.array(init_pos, dtype=np.float32)
        init_pos = init_pos[joint_idxes_ordering]
        
        # self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
        
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(init_pos)
           
        #You can put the correct port here or have the node auto-search for a hand at the first 3 ports.
        self.motors = motors = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        try:
            self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB0', 4000000)
            self.dxl_client.connect()
        except Exception:
            try:
                self.dxl_client = DynamixelClient(motors, '/dev/ttyUSB1', 4000000)
                self.dxl_client.connect()
            except Exception:
                self.dxl_client = DynamixelClient(motors, 'COM13', 4000000)
                self.dxl_client.connect()
        
        
        # #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        # self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        # self.dxl_client.set_torque_enabled(motors, True)
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        # self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        # self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
        # #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        # self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        
        
        
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.9), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write([1,5,9,12], np.ones(3) * (self.kP * 1.25), 84, 2) # Pgain stiffness for side to side should be a bit less
        
        #### new setting ####
        self.dxl_client.sync_write([1,5,9,12], np.ones(4) * (self.kP * 1.25), 84, 2)
        self.dxl_client.sync_write([0], np.ones(1) * (self.kP * 0.01), 84, 2)
        # self.dxl_client.sync_write([1], np.ones(1) * (100), 84, 2)
        
        self.dxl_client.sync_write([1], np.ones(1) * (self.kP), 84, 2) # version 2 
        
        
        # self.dxl_client.sync_write([2], np.ones(1) * (self.kP * 0.1), 84, 2) 
        # self.dxl_client.sync_write([1], np.ones(1) * (100), 84, 2)
        # self.dxl_client.sync_write([1], np.ones(1) * (self.kP * 0.01), 84, 2) # can work for duck #
        #### new setting ####
        
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.9), 80, 2) # Dgain damping for side to side should be a bit less
        self.dxl_client.sync_write([1,5,9,12], np.ones(3) * (self.kD * 1.25), 80, 2) # Dgain damping for side to side should be a bit less
        
        #### new setting ####
        self.dxl_client.sync_write([1,5,9,12], np.ones(4) * (self.kD * 1.25), 80, 2) # Dgain damping for side to side should be a bit less
        # self.dxl_client.sync_write([1], np.ones(1) * (self.kD * 0.01), 80, 2) # Dgain damping for side to side should be a bit less
        #### new setting ####
        
        #Max at current (in unit 1ma) so don't overheat and grip too hard #500 normal or #350 for lite
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.curr_lim, 102, 2)
        
        
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)

    #Receive LEAP pose and directly control the robot
    def set_leap(self, pose):
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #allegro compatibility
    def set_allegro(self, pose):
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #Sim compatibility, first read the sim value in range [-1,1] and then convert to leap
    def set_ones(self, pose):
        pose = self.sim_ones_to_LEAPhand(np.array(pose))
        self.prev_pos = self.curr_pos
        self.curr_pos = np.array(pose)
        self.dxl_client.write_desired_pos(self.motors, self.curr_pos)
    #read position
    def read_pos(self):
        return self.dxl_client.read_pos()
    #read velocity
    def read_vel(self):
        return self.dxl_client.read_vel()
    #read current
    def read_cur(self):
        return self.dxl_client.read_cur()

    def LEAPsim_limits(self,type = "regular"):
        sim_min = np.array([-1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -1.047, -0.314, -0.506, -0.366, -0.349, -0.47, -1.20, -1.34])
        sim_max = np.array([1.047,    2.23,  1.885,  2.042,  1.047,   2.23,  1.885,  2.042,  1.047,   2.23,  1.885,  2.042,  2.094,  2.443, 1.90,  1.88])
        return sim_min, sim_max
    
    def scale(self,x, lower, upper):
        return (0.5 * (x + 1.0) * (upper - lower) + lower)
    
    def sim_ones_to_LEAPhand(self,joints, hack_thumb = False):
        sim_min, sim_max = self.LEAPsim_limits(type = hack_thumb)
        joints = self.scale(joints, sim_min, sim_max)
        joints = self.LEAPsim_to_LEAPhand(joints)
        return joints
    
    def LEAPsim_to_LEAPhand(self,joints):
        joints = np.array(joints)
        ret_joints = joints + 3.14159
        return ret_joints

EXISTING_SIM = None
SCREEN_CAPTURE_RESOLUTION = (1027, 768)

def _create_sim_once(gym, *args, **kwargs):
    global EXISTING_SIM
    if EXISTING_SIM is not None:
        return EXISTING_SIM
    else:
        EXISTING_SIM = gym.create_sim(*args, **kwargs)
        return EXISTING_SIM


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

class Env(ABC):
    def __init__(self, config: Dict[str, Any], rl_device: str, sim_device: str, graphics_device_id: int, headless: bool): 
        """Initialise the env.

        Args:
            config: the configuration dictionary.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
        """

        split_device = sim_device.split(":")
        self.device_type = split_device[0]
        self.device_id = int(split_device[1]) if len(split_device) > 1 else 0

        self.device = "cpu"
        if config["sim"]["use_gpu_pipeline"]:
            if self.device_type.lower() == "cuda" or self.device_type.lower() == "gpu":
                self.device = "cuda" + ":" + str(self.device_id)
            else:
                print("GPU Pipeline can only be used with GPU simulation. Forcing CPU Pipeline.")
                config["sim"]["use_gpu_pipeline"] = False

        self.rl_device = rl_device

        # Rendering
        # if training in a headless mode
        self.headless = headless

        # enable_camera_sensors = config.get("enableCameraSensors", False)
        enable_camera_sensors = config['env'].get("enableCameraSensors", False)
        self.graphics_device_id = graphics_device_id
        print(f"enable_camera_sensors: {enable_camera_sensors}, graphics_device_id: {graphics_device_id}")
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_environments = config["env"]["numEnvs"]
        self.num_agents = config["env"].get("numAgents", 1)  # used for multi-agent environments

        self.num_observations = config["env"].get("numObservations", 0)
        self.num_states = config["env"].get("numStates", 0)
        self.num_observationswactions = config["env"].get("numObservationsWActions", self.num_observations)

        self.obs_space = spaces.Box(np.ones(self.num_obs) * -np.Inf, np.ones(self.num_obs) * np.Inf)
        self.state_space = spaces.Box(np.ones(self.num_states) * -np.Inf, np.ones(self.num_states) * np.Inf)

        self.num_actions = config["env"]["numActions"]
        self.control_freq_inv = config["env"].get("controlFrequencyInv", 1)

        self.act_space = spaces.Box(np.ones(self.num_actions) * -1., np.ones(self.num_actions) * 1.)

        self.clip_obs = config["env"].get("clipObservations", np.Inf)
        self.clip_actions = config["env"].get("clipActions", np.Inf)

        # Total number of training frames since the beginning of the experiment.
        # We get this information from the learning algorithm rather than tracking ourselves.
        # The learning algorithm tracks the total number of frames since the beginning of training and accounts for
        # experiments restart/resumes. This means this number can be > 0 right after initialization if we resume the
        # experiment.
        self.total_train_env_frames: int = 0

        # number of control steps
        self.control_steps: int = 0

        self.render_fps: int = config["env"].get("renderFPS", -1)
        self.last_frame_time: float = 0.0

        self.record_frames: bool = False
        self.record_frames_dir = join("recorded_frames", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    @abc.abstractmethod 
    def allocate_buffers(self):
        """Create torch buffers for observations, rewards, actions dones and any additional data."""

    @abc.abstractmethod
    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.
        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """

    @abc.abstractmethod
    def reset(self)-> Dict[str, torch.Tensor]:
        """Reset the environment.
        Returns:
            Observation dictionary
        """

    @abc.abstractmethod
    def reset_idx(self, env_ids: torch.Tensor):
        """Reset environments having the provided indices.
        Args:
            env_ids: environments to reset
        """

    @property
    def observation_space(self) -> gym.Space:
        """Get the environment's observation space."""
        return self.obs_space

    @property
    def action_space(self) -> gym.Space:
        """Get the environment's action space."""
        return self.act_space

    @property
    def num_envs(self) -> int:
        """Get the number of environments."""
        return self.num_environments

    @property
    def num_acts(self) -> int:
        """Get the number of actions in the environment."""
        return self.num_actions

    @property
    def num_obs(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observations
    
    @property
    def num_obs_w_actions(self) -> int:
        """Get the number of observations in the environment."""
        return self.num_observationswactions

    def set_train_info(self, env_frames, *args, **kwargs):
        """
        Send the information in the direction algo->environment.
        Most common use case: tell the environment how far along we are in the training process. This is useful
        for implementing curriculums and things such as that.
        """
        self.total_train_env_frames = env_frames
        # print(f'env_frames updated to {self.total_train_env_frames}')

    def get_env_state(self):
        """
        Return serializable environment state to be saved to checkpoint.
        Can be used for stateful training sessions, i.e. with adaptive curriculums.
        """
        return None

    def set_env_state(self, env_state):
        pass


class VecTask(Env):

    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 24}

    def __init__(self, config, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture: bool = False, force_render: bool = False): 
        """Initialise the `VecTask`.

        Args:
            config: config dictionary for the environment.
            sim_device: the device to simulate physics on. eg. 'cuda:0' or 'cpu'
            graphics_device_id: the device ID to render with.
            headless: Set to False to disable viewer rendering.
            virtual_screen_capture: Set to True to allow the users get captured screen in RGB array via `env.render(mode='rgb_array')`. 
            force_render: Set to True to always force rendering in the steps (if the `control_freq_inv` is greater than 1 we suggest stting this arg to True)
        """
        # super().__init__(config, rl_device, sim_device, graphics_device_id, headless, use_dict_obs)
        super().__init__(config, rl_device, sim_device, graphics_device_id, headless)
        self.virtual_screen_capture = virtual_screen_capture
        self.virtual_display = None
        if self.virtual_screen_capture:
            from pyvirtualdisplay.smartdisplay import SmartDisplay
            self.virtual_display = SmartDisplay(size=SCREEN_CAPTURE_RESOLUTION)
            self.virtual_display.start()
        self.force_render = force_render

        self.sim_params = self.__parse_sim_params(self.cfg["physics_engine"], self.cfg["sim"])
        if self.cfg["physics_engine"] == "physx":
            self.physics_engine = gymapi.SIM_PHYSX
        elif self.cfg["physics_engine"] == "flex":
            self.physics_engine = gymapi.SIM_FLEX
        else:
            msg = f"Invalid physics engine backend: {self.cfg['physics_engine']}"
            raise ValueError(msg)

        self.dt: float = self.sim_params.dt
        
        rospy.init_node('example')

        self.tf_buffer = tf2_ros.Buffer()
        self.listener = tf2_ros.TransformListener(self.tf_buffer)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)

        self.gym = gymapi.acquire_gym()
        # self.nn_substeps = config['env']['nn_substeps']
        self.first_randomization = True
        self.original_props = {}
        self.dr_randomizations = {}
        self.actor_params_generator = None
        self.extern_actor_params = {}
        self.last_step = -1
        self.last_rand_step = -1
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        # create envs, sim and viewer
        self.sim_initialized = False
        self.create_sim()
        self.gym.prepare_sim(self.sim)
        self.sim_initialized = True

        self.set_viewer()
        self.allocate_buffers()

        self.obs_dict = {}
        
        #Janebek
        ainterface_cfg="charmander.yml"
        acontroller_cfg="joint-impedance-controller.yml"
    

        self.robot_interface = FrankaInterface(
            config_root + f"/{ainterface_cfg}", use_visualizer=False
        )
        self.controller_cfg = YamlConfig(config_root + f"/{acontroller_cfg}").as_easydict()

        self.controller_type = "JOINT_IMPEDANCE"

        self.last_time = None
        
        self.leap_hand = LeapNode()
        
        if self.closed_loop_to_real:
            self.load_kinematics_chain_cl()


    def set_viewer(self):
        """Create the viewer."""

        # todo: read from config
        self.enable_viewer_sync = True
        self.viewer = None

        # if running with a viewer, set up keyboard shortcuts and camera
        if self.headless == False:
            # subscribe to keyboard shortcuts
            self.viewer = self.gym.create_viewer(
                self.sim, gymapi.CameraProperties())
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_ESCAPE, "QUIT")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_V, "toggle_viewer_sync")
            self.gym.subscribe_viewer_keyboard_event(
                self.viewer, gymapi.KEY_R, "record_frames")

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(20.0, 25.0, 3.0)
                cam_target = gymapi.Vec3(10.0, 15.0, 0.0)
            else:
                cam_pos = gymapi.Vec3(20.0, 3.0, 25.0)
                cam_target = gymapi.Vec3(10.0, 0.0, 15.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    def allocate_buffers(self):
        """Allocate the observation, states, etc. buffers.

        These are what is used to set observations and states in the environment classes which
        inherit from this one, and are read in `step` and other related functions.

        """

        # allocate buffers
        # self.obs_buf = torch.zeros(
        #     (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        # num_observationswactions
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_observationswactions), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.timeout_buf = torch.zeros(
             self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

    def create_sim(self, compute_device: int, graphics_device: int, physics_engine, sim_params: gymapi.SimParams):
        """Create an Isaac Gym sim object.

        Args:
            compute_device: ID of compute device to use.
            graphics_device: ID of graphics device to use.
            physics_engine: physics engine to use (`gymapi.SIM_PHYSX` or `gymapi.SIM_FLEX`)
            sim_params: sim params to use.
        Returns:
            the Isaac Gym sim object.
        """
        sim = _create_sim_once(self.gym, compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim

    def get_state(self):
        """Returns the state buffer of the environment (the privileged observations for asymmetric training)."""
        return torch.clamp(self.states_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

    @abc.abstractmethod
    def pre_physics_step(self, actions: torch.Tensor):
        """Apply the actions to the environment (eg by setting torques, position targets).

        Args:
            actions: the actions to apply
        """

    @abc.abstractmethod
    def post_physics_step(self):
        """Compute reward and observations, reset any environments that require it."""


    def step(self, actions: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """Step the physics of the environment.

        Args:
            actions: actions to apply
        Returns:
            Observations, rewards, resets, info
            Observations are dict of observations (currently only one member called 'obs')
        """
        # randomize actions
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        action_tensor = torch.clamp(actions, -self.clip_actions, self.clip_actions)
        # get the prephysics step # get the action tensor #
        self.pre_physics_step(action_tensor)
        
        # Janebek
        if self.last_time == None:
            self.last_time = time.time_ns()
        # current_time = time.time_ns()
        
        # remaining_time = 0.05 - (
        #     current_time - self.last_time
        # ) / (10**9)
        # print(f"remaining_time: {remaining_time}")
        # if 0.0001 < remaining_time:
        #     time.sleep(remaining_time)
        # self.last_time = time.time_ns()
        
        
        
        
        # cur_env_idx = 22
        cur_env_idx = 153
        self.cur_env_idx = cur_env_idx

        joint=self.cur_targets[cur_env_idx, 0:7].cpu().numpy()
        action2 = joint.tolist() + [-1.0]
        # print("actions = ",action2)
        
        # joint_idxes_ordering = [_ for _ in range(4)] + [_ + 8 for _ in range(0, 8)] + [4, 5, 6, 7]
        joint_idxes_ordering = [1, 0, 2, 3, 9, 8, 10, 11, 13, 12, 14, 15, 4, 5, 6, 7]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        self.joint_idxes_ordering = joint_idxes_ordering
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        # self.joint_idxes_ordering_th = torch.from_numpy(joint_idxes_ordering).long().to(self.rl_device)
        # self.inversed_joint_idxes_ordering_th = torch.from_numpy(joint_idxes_inversed_ordering).long().to(self.rl_device)
        
        
        # set_allegro_type = 'actions'
        # set_allegro_type = 'states'
        
        
        # self.cur_targets = tensor_clamp(self.cur_targets, self.shadow_hand_dof_lower_limits[self.actuated_dof_indices],self.shadow_hand_dof_upper_limits[self.actuated_dof_indices])
        
        cur_targets_to_set = self.cur_targets[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering].copy()
        
        
        ##### Attemp 1 -- not used now -- transform the targets to set #####
        # calibrate_joint_idx = 0
        # for i_joint in range(cur_targets_to_set.shape[-1]):
        #     if i_joint != calibrate_joint_idx:
        #         cur_targets_to_set[ i_joint] = 0.0
        #     while cur_targets_to_set[i_joint].item() > np.pi:
        #         cur_targets_to_set[i_joint] -= 2 * np.pi 
        #     while cur_targets_to_set[i_joint].item() < -np.pi:
        #         cur_targets_to_set[i_joint] += 2 * np.pi
        
        # print(f"cur_targets_to_set: {cur_targets_to_set}, {self.cur_targets[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering]}, pos: {self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering]}")
        ##### Attemp 1 -- not used now -- transform the targets to set #####
        
        
        ##### Read leap hand qpos from the leap_hand interface and transform qpos #####
        # real_leap_pos = self.leap_hand.read_pos()

        # for i_fr in range(real_leap_pos.shape[0]):
        #     # while real_leap_pos[i_fr].item() > np.pi:
        #     #     real_leap_pos[i_fr] = real_leap_pos[i_fr] - 2 * np.pi
        #     # while real_leap_pos[i_fr].item() < -np.pi:
        #     #     real_leap_pos[i_fr] = real_leap_pos[i_fr] + 2 * np.pi 
        #     real_leap_pos[i_fr] = real_leap_pos[i_fr] - np.pi
        
        # # print(f"[LEAP REAL] real_leap_pos.shape: {real_leap_pos.shape}")
        # print(f"[LEAP REAL] {real_leap_pos}")
        # print(f"[LEAP SIMM] {self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering]}")
        ##### Read leap hand qpos from the leap_hand interface and transform qpos #####
        
        
        # joint_idx_interested = 1
        # # print(f"cur_targets_to_set: {cur_targets_to_set[joint_idx_interested]}, {self.cur_targets[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering][joint_idx_interested]}, pos: {self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering][joint_idx_interested]}")
        
        # self.leap_hand.set_allegro(cur_targets_to_set)
        # # self.leap_hand.set_allegro(self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering])
    
        # self.robot_interface.control(
        #     controller_type=self.controller_type,
        #     action=action2,
        #     controller_cfg=self.controller_cfg,
        # )
        
        # 
        # time.sleep(10000)
        # input()
        
        # self.gym.simulate(self.sim)
        
        # step physics and render each frame
        for i in range(self.control_freq_inv): 
        # for i in range(20): 
        
        # for i in range(3): 

            self.gym.simulate(self.sim)
            if self.force_render:
                self.render()
        
        if self.force_render:
            self.render()

        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
        
        
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        
        
        current_time = time.time_ns()
        
        remaining_time = 0.05 - (
            current_time - self.last_time
        ) / (10**9)
        print(f"remaining_time: {remaining_time}")
        if 0.0001 < remaining_time:
            time.sleep(remaining_time)
        self.last_time = time.time_ns()
        
        
        # joint=self.cur_targets[cur_env_idx, 0:7].cpu().numpy()
        # action2 = joint.tolist() + [-1.0]
        ### set states as the actions ###
        arm_states = self.shadow_hand_dof_pos.detach().cpu().numpy()[cur_env_idx, :7].copy()
        action2 = arm_states.tolist() + [-1.0]
        ### set states as the actions ###
        
        
        
        joint_idx_interested = 1
        # print(f"cur_targets_to_set: {cur_targets_to_set[joint_idx_interested]}, {self.cur_targets[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering][joint_idx_interested]}, pos: {self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering][joint_idx_interested]}")
        
        self.leap_hand.set_allegro(cur_targets_to_set)
        # self.leap_hand.set_allegro(self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering])
    
        self.robot_interface.control(
            controller_type=self.controller_type,
            action=action2,
            controller_cfg=self.controller_cfg,
        )
        
        # 
        
        real_leap_pos = self.leap_hand.read_pos()

        for i_fr in range(real_leap_pos.shape[0]):
            # while real_leap_pos[i_fr].item() > np.pi:
            #     real_leap_pos[i_fr] = real_leap_pos[i_fr] - 2 * np.pi
            # while real_leap_pos[i_fr].item() < -np.pi:
            #     real_leap_pos[i_fr] = real_leap_pos[i_fr] + 2 * np.pi 
            real_leap_pos[i_fr] = real_leap_pos[i_fr] - np.pi
            
        self.real_leap_pos_to_sim = real_leap_pos[joint_idxes_inversed_ordering] 
        # 
        self.real_leap_pos_to_sim = torch.from_numpy(self.real_leap_pos_to_sim).to(self.rl_device) # --- 16-dim torch tensor as real leap pos to sim --- #
        
        arm_pos = self.robot_interface.last_q # 
        self.real_arm_pos  = torch.from_numpy(arm_pos).to(self.rl_device)
        




        if self.closed_loop_to_real:
            if rospy.is_shutdown():
                exit(0)
            self.post_physics_step_cl()
        else:
            # compute observations, rewards, resets, ...
            self.post_physics_step()
            
        
        
        # self.leap_hand.set_allegro(self.shadow_hand_dof_pos[cur_env_idx, 7:].cpu().numpy()[self.joint_idxes_ordering])
    

        self.control_steps += 1

        # fill time out buffer: set to 1 if we reached the max episode length AND the reset buffer is 1. Timeout == 1 makes sense only if the reset buffer is 1.
        self.timeout_buf = (self.progress_buf >= self.max_episode_length - 1) & (self.reset_buf != 0)

        # randomize observations
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)

        self.extras["time_outs"] = self.timeout_buf.to(self.rl_device)



        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
        
        if self.masked_mimic_training:
            mimic_teacher_obs_buf_clamped = torch.clamp(self.mimic_teacher_obs_buf, -self.clip_obs, self.clip_obs)
            self.obs_dict["mimic_teacher_obs"] = mimic_teacher_obs_buf_clamped.to(self.rl_device)

        

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
            
            
        try:
            if self.w_forecasting:
                
                
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
                )[:, 1] # next tracking targets # #
                
                nex_tracking_targets_forecast_obs = nex_tracking_targets.clone()
                # gt target # # gt target # #
                # gt target # # gt target # #
                tot_gt_target_hand_qs = tot_gt_target_hand_qs - self.shadow_hand_dof_pos.unsqueeze(1) # 
        

                
                tot_gt_target =  torch.cat(
                    [tot_gt_target_hand_qs, tot_gt_target_obj_pos, tot_gt_target_obj_rot], dim=-1
                )
                tot_gt_target_flatten = tot_gt_target.view(tot_gt_target.size(0), -1)
                
                
                if self.already_predicted_targets:
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
                    )[:, 1] # next tracking targets # #
                    
                    
            
        
                
                envs_text_features = batched_index_select(self.tot_text_features, self.env_inst_idxes, dim=0)
                if self.tuning_single_instance:
                    # self.forecasting_obs = torch.cat(
                    #     [
                    #         self.shadow_hand_dof_pos, self.object_pos, self.object_rot,  tot_gt_target_flatten
                    #         # self.env_object_latent_feat, envs_text_features
                    #     ], dim=-1
                    # )
                    self.forecasting_obs = torch.cat(
                        [
                            self.shadow_hand_dof_pos, self.object_pos, self.object_rot, nex_tracking_targets_forecast_obs, tot_gt_target_flatten
                            # self.env_object_latent_feat, envs_text_features
                        ], dim=-1, # 
                    )
                else:
                    self.forecasting_obs = torch.cat(
                        [
                            self.shadow_hand_dof_pos, self.object_pos, self.object_rot,  self.env_object_latent_feat, envs_text_features, tot_gt_target_flatten
                            #
                        ], dim=-1
                    )
                    
                    
                hand_goal_start = self.hand_goal_start
                hand_goal_start_tsr = torch.tensor([hand_goal_start, self.nex_ref_start, self.obj_obs_start], device=self.rl_device).float()
          
                    
                # obs_dict = {}
                self.obs_dict["forecasting_obs"] = torch.clamp(self.forecasting_obs, -self.clip_obs, self.clip_obs).to(self.rl_device)
                # obs_dict['obs'] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)
                if 'states' in self.obs_dict:
                    self.obs_dict["forecasting_states"] = self.obs_dict["states"].clone()
                
                self.obs_dict['forecasting_rewards'] = self.forecasting_rew_buf.to(self.rl_device).unsqueeze(1)
                self.obs_dict['nex_tracking_targets'] = nex_tracking_targets.to(self.rl_device)
                self.obs_dict['hand_goal_start'] = hand_goal_start_tsr.to(self.rl_device)
        except:
            pass

        return self.obs_dict, self.rew_buf.to(self.rl_device), self.reset_buf.to(self.rl_device), self.extras
    
    def load_kinematics_chain_cl(self):
        
        if self.hand_type == 'allegro':
            urdf_fn = "../assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
            
            self.palm_name = 'palm_link'
            self.first_tip_name = 'link_3_tip'
            self.second_tip_name = 'link_7_tip'
            self.third_tip_name = 'link_11_tip'
            self.forth_tip_name = 'link_15_tip'
        
        elif self.hand_type == 'leap':
            urdf_fn = "../assets/leap_hand/franka_panda_leaphand.urdf"
            allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
                
            self.palm_name = 'palm_lower'
            self.first_tip_name = 'thumb_tip_head'
            self.second_tip_name = 'index_tip_head'
            self.third_tip_name = 'middle_tip_head'
            self.forth_tip_name = 'ring_tip_head'
            
            # first_tip_name = 'thumb_fingertip'
            # second_tip_name = 'fingertip'
            # third_tip_name = 'fingertip_2'
            # forth_tip_name = 'fingertip_3'
        else:
            raise NotImplementedError
        
        print(f"[INFO] Building chain from urdf: {urdf_fn}")
        self.chain = pk.build_chain_from_urdf(open(urdf_fn).read())
        self.chain = self.chain.to(dtype=torch.float32, device=self.rl_device)
        
        
        
        pass
    
    def post_physics_step_cl(self):
        
        self.progress_buf += 1
        self.randomize_buf += 1
        
        if self.use_twostage_rew:
            self.grasping_progress_buf += 1

        self.ref_ts += 1
        
        self.compute_observations_cl()
        
        # if self.use_forcasting_model:
        #     self._forward_forcasting_model() # forward the forcasting model #
        #     self.try_save_network_forwarding_info_dict() # try to save them #
        
        # print(f"To compute reward with ref_ts: {self.ref_ts}")
        self.compute_reward(self.actions)
        
        if self.test:
            # if self.ref_ts >= self.max_episode_length - 3:
            self.try_save()
        else:
            # if not self.single_instance_training and self.num_envs < 1000:
            if not self.single_instance_training and self.sv_info_during_training:
                self.try_save_train()
                if self.ref_ts>=300:
                    self.ref_ts = 0
                    
        # self.try_save_forcasting_model()

        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            for i in range(self.num_envs):
                self.add_debug_lines(self.envs[i], self.object_pos[i], self.object_rot[i])
    
    
    def compute_observations_cl(self):
        
        
        if self.randomize_conditions:
            self.generate_random_mask_config()
        
        
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)

        if self.obs_type == "full_state" or self.asymmetric_obs:
            self.gym.refresh_force_sensor_tensor(self.sim)
            self.gym.refresh_dof_force_tensor(self.sim)

        if self.w_franka:
            self.gym.refresh_jacobian_tensors(self.sim)
        
        real_robot_qpos = torch.cat(
            [ self.real_arm_pos, self.real_leap_pos_to_sim ], dim=-1 # get the real arm pos and the real leap to --> construct real robot qpos
        ).float()
        # self.
        print(f"[Debug] real_robot_qpos.type: {real_robot_qpos.type}")
        tg_batch = self.chain.forward_kinematics(real_robot_qpos.unsqueeze(0))
        key_to_link_ornt = {}
        key_to_link_pos = {}
        for key in tg_batch:
            m = tg_batch[key].get_matrix()
            cur_link_pos = m[:, :3, 3].float()
            cur_link_ornt = m[:, :3, :3].float()
            cur_link_ornt = pk.matrix_to_quaternion(cur_link_ornt)
            key_to_link_pos[key] = cur_link_pos # get the cur link pos #
            key_to_link_ornt[key] = cur_link_ornt[:, [1, 2, 3, 0]] # convert the quaternion from wxyz to xyzw 
        
        
        # object pose #
        self.object_pose = self.root_state_tensor[self.object_indices, 0:7]
        self.object_pos = self.root_state_tensor[self.object_indices, 0:3]
        self.object_rot = self.root_state_tensor[self.object_indices, 3:7]
        
        
        debug_env_obj_pose = self.object_pose[self.cur_env_idx]
        
        trans = self.tf_buffer.lookup_transform("base_link", "object_frame", rospy.Time(0), rospy.Duration(0.5))
        x = trans.transform.translation.x
        y = trans.transform.translation.y
        z = trans.transform.translation.z
        qx = trans.transform.rotation.x
        qy = trans.transform.rotation.y
        qz = trans.transform.rotation.z
        qw = trans.transform.rotation.w

        obj_pose = [x, y, z, qx, qy, qz, qw]
        # print("obj_pose = ",obj_pose)
        
        obj_pose = np.array(obj_pose, dtype=np.float32)
        obj_pose = torch.from_numpy(obj_pose).float().to(self.rl_device)
        
        print(f"[REAL Obj] {obj_pose}")
        print(f"[SIMM Obj] {debug_env_obj_pose}")
        
        self.object_pose = obj_pose.unsqueeze(0).repeat(self.num_envs, 1)
        self.object_pos = self.object_pose[..., :3]
        self.object_rot = self.object_pose[..., 3:]
        
        
        
        
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

        self.right_hand_pos = key_to_link_pos[self.palm_name].repeat(self.num_envs, 1) # (nn_envs, 3) -- the palm link pos #
        self.right_hand_rot = key_to_link_ornt[self.palm_name].repeat(self.num_envs, 1) # (nn_envs, 4) -- the palm link ornt #
        


        idx = self.hand_body_idx_dict['index'] 
        self.right_hand_ff_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_ff_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_ff_pos = self.right_hand_ff_pos + quat_apply(self.right_hand_ff_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
        
        self.right_hand_ff_pos = key_to_link_pos[self.second_tip_name].repeat(self.num_envs, 1 ) # (nn_envs x 3) #  
        self.right_hand_ff_rot = key_to_link_ornt[self.second_tip_name].repeat(self.num_envs, 1)
                                             
                                                              
        idx = self.hand_body_idx_dict['middle']
        self.right_hand_mf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_mf_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_mf_pos = self.right_hand_mf_pos + quat_apply(self.right_hand_mf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)


        self.right_hand_mf_pos = key_to_link_pos[self.third_tip_name].repeat(self.num_envs, 1)
        self.right_hand_mf_rot = key_to_link_ornt[self.third_tip_name].repeat(self.num_envs, 1)


        idx = self.hand_body_idx_dict['ring']
        self.right_hand_rf_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_rf_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_rf_pos = self.right_hand_rf_pos + quat_apply(self.right_hand_rf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.right_hand_rf_pos = key_to_link_pos[self.forth_tip_name].repeat(self.num_envs, 1)
        self.right_hand_rf_rot = key_to_link_ornt[self.forth_tip_name].repeat(self.num_envs, 1)


        # idx = self.hand_body_idx_dict['little']
        # self.right_hand_lf_pos = self.rigid_body_states[:, idx, 0:3]
        # self.right_hand_lf_rot = self.rigid_body_states[:, idx, 3:7]
        # # self.right_hand_lf_pos = self.right_hand_lf_pos + quat_apply(self.right_hand_lf_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)
                                                                         
        idx = self.hand_body_idx_dict['thumb']
        self.right_hand_th_pos = self.rigid_body_states[:, idx, 0:3]
        self.right_hand_th_rot = self.rigid_body_states[:, idx, 3:7]
        # self.right_hand_th_pos = self.right_hand_th_pos + quat_apply(self.right_hand_th_rot,to_torch([0, 0, 1], device=self.device).repeat(self.num_envs, 1) * 0.02)

        self.right_hand_th_pos = key_to_link_pos[self.first_tip_name].repeat(self.num_envs, 1)
        self.right_hand_th_rot = key_to_link_ornt[self.first_tip_name].repeat(self.num_envs, 1)

        self.goal_pose = self.goal_states[:, 0:7]
        self.goal_pos = self.goal_states[:, 0:3]
        self.goal_rot = self.goal_states[:, 3:7]
        
        self.goal_lifting_pos = self.goal_states[:, 0:3] 
        
        # fingertip state # # 
        # fingertip state # # nn_envs x nn_fingertipsx 13
        self.fingertip_state = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:13]
        self.fingertip_pos = self.rigid_body_states[:, self.fingertip_handles][:, :, 0:3]

        self.fingertip_pos = torch.stack(
            [ self.right_hand_th_pos, self.right_hand_ff_pos, self.right_hand_mf_pos, self.right_hand_rf_pos ], dim=1 # nn_envsx nn_fingers x 3 
        )
        self.fingertip_rot = torch.stack(
            [ self.right_hand_th_rot, self.right_hand_ff_rot, self.right_hand_mf_rot, self.right_hand_rf_rot ], dim=1 # nn_envsx nn_fingers x 4
        )
        # self.fingertip_state = torch.cat(
        #     [ self.fingertip_pos, torch.zeros((self.num_envs, self.fingertip_pos.size(1), 13 - self.fingertip_pos.size(-1))).float().to(self.rl_device) ], dim=-1 # nn_envs x nn_fingers x 13 #
        # )
        self.fingertip_state = torch.cat(
            [ self.fingertip_pos, self.fingertip_rot, torch.zeros((self.num_envs, self.fingertip_pos.size(1), 13 - self.fingertip_pos.size(-1) - self.fingertip_rot.size(-1))).float().to(self.rl_device) ], dim=-1 # nn_envs x nn_fingers x 13 #
        )
        
        
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
        
        

        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
        
        

        envs_maxx_episode_length_per_traj = batched_index_select(self.maxx_episode_length_per_traj, self.env_inst_idxes, dim=0) # nn_envs x 1
        # print(f"progress_buf: {torch.max(self.progress_buf)}, envs_goal_obj_trans_th: {envs_goal_obj_trans_th.size()}")
        
        if self.use_window_future_selection:
            cur_progress_buf = torch.clamp(self.ws_selected_progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        else:
            cur_progress_buf = torch.clamp(self.progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        
        
        cur_progress_buf_obj = cur_progress_buf.clone()
        
        if self.use_future_ref_as_obs_goal:
            cur_progress_buf_obj = torch.clamp(self.progress_buf + 1, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        
        
        ### cond frequency ###
        if self.random_shift_cond_freq or  self.preset_inv_cond_freq > 1:
            moded_progress_buf = cur_progress_buf // self.env_inv_cond_freq
            increase_nn = (cur_progress_buf > moded_progress_buf * self.env_inv_cond_freq).int()
            cur_progress_buf = (moded_progress_buf + increase_nn) * self.env_inv_cond_freq
            cur_progress_buf = torch.clamp(cur_progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(cur_progress_buf))
            
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        # whether to
        if self.use_forcasting_model and self.already_forcasted:
            # print(f"maxx_cur_progress_buf: {torch.max(cur_progress_buf)}, minn_cur_progress_buf: {torch.min(cur_progress_buf)}, forcast_obj_pos: {self.forcast_obj_pos.size()}, forcast_obj_rot: {self.forcast_obj_rot.size()}")
            cur_goal_pos = batched_index_select(self.forcast_obj_pos, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
            cur_goal_rot = batched_index_select(self.forcast_obj_rot, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 # 

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
        
        
        
        cur_progress_buf_obj = cur_progress_buf.clone()
        
        if self.use_future_ref_as_obs_goal:
            cur_progress_buf_obj = torch.clamp(self.progress_buf + 1, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        
        # values, indices, dims #
        envs_goal_obj_trans_th = batched_index_select(tot_goal_obj_trans_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 3 
        envs_goal_obj_ornt_th = batched_index_select(tot_goal_obj_ornt_th, self.env_inst_idxes, dim=0) # nn_envs x maximum_episode_length x 4
    
        # cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        # cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        cur_goal_pos = batched_index_select(envs_goal_obj_trans_th, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 #
        cur_goal_rot = batched_index_select(envs_goal_obj_ornt_th, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 #
        
        
        if self.use_forcasting_model and self.already_forcasted:
            # forcast_shadow_hand_dof_pos, forcast_obj_pos, forcast_obj_rot # # use the forcasted obj pos # # 
            cur_goal_pos = batched_index_select(self.forcast_obj_pos, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 3 # # get the current forcasted obj pos 
            cur_goal_rot = batched_index_select(self.forcast_obj_rot, cur_progress_buf_obj.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x 4 # # get the current forcasted obj rot 
        
        
        
        tot_goal_hand_qs_th = self.tot_hand_preopt_res
        
        
        envs_goal_hand_qs = batched_index_select(tot_goal_hand_qs_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
        
        
        ### TODO: we first try to not to use the forcasted hand qpos ### # w forecasting model #
        if self.use_forcasting_model and self.already_forcasted:
            # # cur_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
            forcast_envs_goal_hand_qs = self.forcast_shadow_hand_dof_pos 
            # using_forcast_res_step_threshold = self.using_forcast_res_step_threshold
            # minn_ts_nn = min(envs_goal_hand_qs.size(1), forcast_envs_goal_hand_qs.size(1))
            # envs_goal_hand_qs[self.progress_buf >= using_forcast_res_step_threshold, : minn_ts_nn] = forcast_envs_goal_hand_qs[self.progress_buf >= using_forcast_res_step_threshold, : minn_ts_nn]
            envs_goal_hand_qs = forcast_envs_goal_hand_qs
        

        cur_progress_buf_handqpos = cur_progress_buf.clone()
        if self.use_future_ref_as_obs_goal:
            cur_progress_buf_handqpos = torch.clamp(self.progress_buf + 1, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(envs_maxx_episode_length_per_traj))
        
        # cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf_handqpos.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #


        # ### TODO: we first try to not to use the forcasted hand qpos ###
        # if self.use_forcasting_model and self.already_forcasted:
        #     cur_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1)

        

        ### current target hand pose, and the difference from the reference hand pos ###
        # cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.hand_qs_th[self.progress_buf]
        # cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        
        ## Related to the reward computation ##
        if self.w_franka:
            self.delta_qpos = self.shadow_hand_dof_pos - self.shadow_hand_dof_pos
            self.delta_qpos[..., 7:] = self.shadow_hand_dof_pos[..., 7:] - cur_hand_qpos_ref[..., 6:]
            
            if self.load_kine_info_retar_with_arm:
                tot_goal_hand_qs_warm_th = self.tot_kine_qs_w_arm
                envs_goal_hand_qs_warm = batched_index_select(tot_goal_hand_qs_warm_th, self.env_inst_idxes, dim=0) ## nn_envs x maximum_episode_length x len(hand_qs)
                cur_hand_qpos_ref_warm = batched_index_select(envs_goal_hand_qs_warm, cur_progress_buf_handqpos.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
                
                expanded_real_robot_qpos = real_robot_qpos.unsqueeze(0).repeat(self.num_envs, 1)
                
                self.expanded_real_robot_qpos = expanded_real_robot_qpos
                self.delta_qpos = expanded_real_robot_qpos - cur_hand_qpos_ref_warm
        
        else:
            self.delta_qpos = self.shadow_hand_dof_pos - cur_hand_qpos_ref
        # self.ori_cur_hand_qpos_ref = self.goal_hand_qs_th[self.progress_buf]
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, self.progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        # self.ori_cur_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        
        
        ### TODO: we first try to not to use the forcasted hand qpos #### qpos #
        # if self.use_forcasting_model and self.already_forcasted:
        #     self.ori_cur_hand_qpos_ref = batched_index_select(self.forcast_shadow_hand_dof_pos, cur_progress_buf.unsqueeze(-1), dim=1).squeeze(1) # nn_envs x len(hand_qs) #
        
        ### next progress buffer ###
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.hand_qs_th.size(0) - 1)
        # nex_hand_qpos_ref = self.hand_qs_th[nex_progress_buf] # get the next progress buf --- nn_envs x nn_ref_qs ##
        # self.nex_hand_qpos_ref = nex_hand_qpos_ref # next progress buf #
        # nex_progress_buf = torch.clamp(self.progress_buf + 1, 0, self.maxx_kine_nn_ts - 1)
        
        
        #### NOTE: nex_hand_qpos_ref is used as the kinematic bias in the next step -- so we do not need to consider the effect of random_shift_cond_freq here ####
        if self.use_window_future_selection:
            nex_progress_buf = torch.clamp(self.ws_selected_progress_buf + 1, min=torch.zeros_like(envs_maxx_episode_length_per_traj), max=envs_maxx_episode_length_per_traj)
        else:
            nex_progress_buf = torch.clamp(self.progress_buf + 1, min=torch.zeros_like(envs_maxx_episode_length_per_traj), max=envs_maxx_episode_length_per_traj)
        
        # if self.random_shift_cond_freq:
        #     moded_progress_buf = nex_progress_buf // self.env_inv_cond_freq
        #     increase_nn = (nex_progress_buf > moded_progress_buf * self.env_inv_cond_freq).int()
        #     nex_progress_buf = (moded_progress_buf + increase_nn) * self.env_inv_cond_freq
        #     nex_progress_buf = torch.clamp(nex_progress_buf, max=envs_maxx_episode_length_per_traj, min=torch.zeros_like(nex_progress_buf))
        
        
        if self.w_franka and self.load_kine_info_retar_with_arm:
            nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs_warm, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
            
            # nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        else:
            nex_hand_qpos_ref = batched_index_select(envs_goal_hand_qs, nex_progress_buf.unsqueeze(-1), dim=1).squeeze(1)
        
        self.nex_hand_qpos_ref = nex_hand_qpos_ref
        
        
            
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
            # self.ts_to_hand_obj_states[self.ref_ts]
        else:
            # if not self.single_instance_training and self.num_envs < 1000: # get the sv info during training #
            if not self.single_instance_training and self.sv_info_during_training:
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
                
                if self.use_forcasting_model and self.already_forcasted:
                    cur_ts_forcast_hand_dof_pos = self.forcast_shadow_hand_dof_pos.detach().cpu().numpy()[:, self.ref_ts]
                    cur_ts_forcast_obj_pos = self.forcast_obj_pos.detach().cpu().numpy()[:, self.ref_ts]
                    cur_ts_forcast_obj_ornt = self.forcast_obj_rot.detach().cpu().numpy()[:, self.ref_ts]
                else:
                    cur_ts_forcast_hand_dof_pos = self.shadow_hand_dof_pos_np
                    cur_ts_forcast_obj_pos = self.object_pose_np[..., :3]
                    cur_ts_forcast_obj_ornt = self.object_pose_np[..., 3:]
                nn_envs = 100
                self.ts_to_hand_obj_states[self.ref_ts] = {
                    'shadow_hand_dof_pos': self.shadow_hand_dof_pos_np[:nn_envs],
                    'shadow_hand_dof_tars': self.target_qpos_np[:nn_envs],
                    'object_pose': self.object_pose_np[:nn_envs],
                    'shadow_hand_dof_vel': self.shadow_hand_dof_vel_np[:nn_envs],
                    'object_linvel': self.object_linvel_np[:nn_envs],
                    'object_angvel': self.object_angvel_np[:nn_envs],
                    'actions': self.actions_np[:nn_envs] , 
                    'observations': self.obs_buf_np[:nn_envs],
                    'forcast_hand_dof_pos': cur_ts_forcast_hand_dof_pos[:nn_envs],
                    'forcast_obj_pos': cur_ts_forcast_obj_pos[:nn_envs],
                    'forcast_obj_ornt': cur_ts_forcast_obj_ornt[:nn_envs]
                }
        
        
        
        self.compute_full_state_cl()

        if self.asymmetric_obs: 
            self.compute_full_state_cl(True)



    def compute_full_state_cl(self, asymm_obs=False): #
        
        self.get_unpose_quat()
        
        
        # 2 * nn_hand_dofs + 13 * num_fingertips + 6 + nn_hand_dofs + 16 + 7 + nn_hand_dofs ## 

        num_ft_states = 13 * int(self.num_fingertips)  # 65 ##
        per_finger_nn_state = 13
        if self.wo_fingertip_rot_vel:
            num_ft_states = 3 * int(self.num_fingertips)  # 50 ##
            per_finger_nn_state = 3
        elif self.wo_fingertip_vel:
            num_ft_states = 7 * int(self.num_fingertips)  # 50 ##
            per_finger_nn_state = 7
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
                canon_shadow_hand_dof = self.expanded_real_robot_qpos #  self.shadow_hand_dof_pos 
            
            
            self.cur_dof_vel[:, : ] = self.shadow_hand_dof_vel

            # get the obs buf #
            # self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(self.shadow_hand_dof_pos, # 
            #                                                        self.shadow_hand_dof_lower_limits, # 
            #                                                        self.shadow_hand_dof_upper_limits) #
            self.obs_buf[:, 0:self.num_shadow_hand_dofs] = unscale(canon_shadow_hand_dof,
                                                                self.shadow_hand_dof_lower_limits,
                                                                self.shadow_hand_dof_upper_limits)
            if self.wo_vel_obs: # previous hand poses and dofs and the preiv
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
                
            if self.wo_fingertip_rot_vel:
                # canon_fingertip_pose[..., 3:] = canon_fingertip_pose[..., 3:] * 0.0 # zero out the fingertip rotation velocities #
                canon_fingertip_pose = self.fingertip_pos
        
            # # aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            # aux = canon_fingertip_pose.reshape(self.num_envs, num_ft_states)
            # for i in range(self.num_fingertips):
            #     aux[:, i * 13:(i + 1) * 13] = self.unpose_state(aux[:, i * 13:(i + 1) * 13])
            # # 66:131: ft states
            # self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

            
            # aux = self.fingertip_state.reshape(self.num_envs, num_ft_states)
            aux = canon_fingertip_pose.reshape(self.num_envs, num_ft_states)
            for i in range(self.num_fingertips):
                aux[:, i * per_finger_nn_state:(i + 1) * per_finger_nn_state] = self.unpose_state(aux[:, i * per_finger_nn_state:(i + 1) * per_finger_nn_state])
            # 66:131: ft states
            self.obs_buf[:, fingertip_obs_start:fingertip_obs_start + num_ft_states] = aux

            
            
            # 131:161: ft sensors: do not need repose
            if self.obs_type == "full_state" or asymm_obs:
            #     self.obs_buf[:, fingertip_obs_start + num_ft_states:fingertip_obs_start + num_ft_states + num_ft_force_torques] = self.force_torque_obs_scale * self.force_sensor_tensor[:, :num_ft_force_torques]
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
        
        
            if self.wo_fingertip_rot_vel:
                # canon_fingertip_pose[..., 3:] = canon_fingertip_pose[..., 3:] * 0.0 # zero out the fingertip rotation velocities #
                canon_fingertip_pose = history_fingertip_pos
                
        
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
            # print()
            
        
        
        #### Delta object pos ####
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
        # 207:236 goal # obj obs start # 
        # if 
        # hand_goal_start = obj_obs_start + 16
        
        if self.tight_obs:
            self.obs_buf[:, hand_goal_start: hand_goal_start +  self.num_shadow_hand_dofs] = self.delta_qpos
        else:
            self.obs_buf[:, hand_goal_start:hand_goal_start + 3] = self.delta_target_hand_pos
            self.obs_buf[:, hand_goal_start + 3:hand_goal_start + 7] = self.delta_target_hand_rot
            # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 29] = self.delta_qpos # 
            self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.delta_qpos


            if self.masked_mimic_training:
                self.mimic_teacher_obs_buf[:, : hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.obs_buf[:, : hand_goal_start + 7+  self.num_shadow_hand_dofs].clone()

            # #### NOTE: version 2 of the randomize conditions ####
            # if self.randomize_conditions:
            #     if self.condition_mask_type == MASK_HAND:
            #         self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = 0.0
            #     elif self.condition_mask_type == MASK_HAND_RNDIDX:
            #         self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs][:, self.rnd_selected_hand_joints] = 0.0
            #         # self.obs_buf[:, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs]] = 0.0
            # #### NOTE: version 2 of the randomize conditions ####
            
            # #### NOTE: version 3 of the randomize conditions ####
            # # if self.random_shift_cond:
            # self.obs_buf[self.env_cond_type == COND_OBJ, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = 0.0
            # self.obs_buf[self.env_cond_type == COND_PARTIALHAND_OBJ, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] = self.obs_buf[self.env_cond_type == COND_PARTIALHAND_OBJ, hand_goal_start + 7:hand_goal_start + 7+  self.num_shadow_hand_dofs] * self.env_cond_hand_masks[self.env_cond_type == COND_PARTIALHAND_OBJ] # nn_cond_envs x nn_hand_dof xxxxxx nn_cond_envs x nn_hand_dof #
            # #### NOTE: version 3 of the randomize conditions ####
            
            hand_goal_start = hand_goal_start + 7




        if self.obs_type == 'pure_state_wref_wdelta' and self.use_kinematics_bias_wdelta:
            
            # tot goal hand qs th #
            # tot_goal_hand_qs_th = self.tot_kine_qs # goal pos and goal rot #
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
            
            
                if self.w_franka:
                    unscaled_nex_hand_qpos_ref = canon_hand_qpos_ref
                else:
                    # unscaled_nex_hand_qpos_ref = unscale(self.nex_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
                    unscaled_nex_hand_qpos_ref = unscale(canon_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
            
                # cur_hand_qpos_ref #  # current delta targets #
                # unscaled_nex_hand_qpos_ref = unscale(cur_hand_qpos_ref, self.shadow_hand_dof_lower_limits, self.shadow_hand_dof_upper_limits)
                
                if self.w_franka:
                    if self.load_kine_info_retar_with_arm:
                        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs ] = unscaled_nex_hand_qpos_ref
                        cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs 
                        
                        # self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs - 1] = unscaled_nex_hand_qpos_ref
                        # cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs - 1
                    else:
                        self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs - 1] = unscaled_nex_hand_qpos_ref
                        cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs - 1
                else:
                    # unscaled_nex_hand_qpos_ref = cur_hand_qpos_ref
                    self.obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref
                    cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
            
            
            if self.w_franka:
                # cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
                if self.load_kine_info_retar_with_arm:
                    self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets_warm[:, : self.num_shadow_hand_dofs]
                    
                    # self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs - 1] = self.cur_delta_targets[:, : self.num_shadow_hand_dofs - 1]
                else:
                    self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs - 1] = self.cur_delta_targets[:, : self.num_shadow_hand_dofs - 1]
            else:
                # cur_delta_start = nex_ref_start + self.num_shadow_hand_dofs
                self.obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs]
            
            if self.masked_mimic_training:
                self.mimic_teacher_obs_buf[:, nex_ref_start : nex_ref_start + self.num_shadow_hand_dofs] = unscaled_nex_hand_qpos_ref.clone()
                self.mimic_teacher_obs_buf[:, cur_delta_start : cur_delta_start + self.num_shadow_hand_dofs] = self.cur_delta_targets[:, :self.num_shadow_hand_dofs].clone()
                
            
            if self.w_franka:
                if self.load_kine_info_retar_with_arm:
                    obj_feat_st_idx = cur_delta_start + self.num_shadow_hand_dofs
                    # obj_feat_st_idx = cur_delta_start + self.num_shadow_hand_dofs - 1
                else:
                    obj_feat_st_idx = cur_delta_start + self.num_shadow_hand_dofs - 1
            else:
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
                
                if self.w_franka:
                    # nex_delta_delta_actions = nex_delta_actions - self.cur_delta_targets[:, self.actuated_dof_indices]
                    pass
                else:
                    # nex_delta_delta_actions = torch.zero
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


    def zero_actions(self) -> torch.Tensor:
        """Returns a buffer with zero actions.

        Returns:
            A buffer of zero torch actions
        """
        actions = torch.zeros([self.num_envs, self.num_actions], dtype=torch.float32, device=self.rl_device)

        return actions

    def reset_idx(self, env_idx):
        """Reset environment with indces in env_idx. 
        Should be implemented in an environment class inherited from VecTask.
        """  
        pass

    def reset(self):
        """Is called only once when environment starts to provide the first observations.
        Doesn't calculate observations. Actual reset and observation calculation need to be implemented by user.
        Returns:
            Observation dictionary
        """
        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
            
        
        if self.masked_mimic_training:
            mimic_teacher_obs_buf_clamped = torch.clamp(self.mimic_teacher_obs_buf, -self.clip_obs, self.clip_obs)
            self.obs_dict["mimic_teacher_obs"] = mimic_teacher_obs_buf_clamped.to(self.rl_device)
        print(f"Retruning reset obs dict with keys: {self.obs_dict.keys()} ")

        return self.obs_dict

    def reset_done(self):
        """Reset the environment.
        Returns:
            Observation dictionary, indices of environments being reset
        """
        done_env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        if len(done_env_ids) > 0:
            self.reset_idx(done_env_ids)

        self.obs_dict["obs"] = torch.clamp(self.obs_buf, -self.clip_obs, self.clip_obs).to(self.rl_device)

        # asymmetric actor-critic
        if self.num_states > 0:
            self.obs_dict["states"] = self.get_state()
            
        if self.masked_mimic_training:
            mimic_teacher_obs_buf_clamped = torch.clamp(self.mimic_teacher_obs_buf, -self.clip_obs, self.clip_obs)
            self.obs_dict["mimic_teacher_obs"] = mimic_teacher_obs_buf_clamped.to(self.rl_device)
        print(f"Retruning reset obs dict with keys: {self.obs_dict.keys()} ")
        return self.obs_dict, done_env_ids

    def render(self, mode="rgb_array"):
        """Draw the frame to the viewer, and check for keyboard events."""
        if self.viewer:
            # check for window closed
            if self.gym.query_viewer_has_closed(self.viewer):
                sys.exit()

            # check for keyboard events
            for evt in self.gym.query_viewer_action_events(self.viewer):
                if evt.action == "QUIT" and evt.value > 0:
                    sys.exit()
                elif evt.action == "toggle_viewer_sync" and evt.value > 0:
                    self.enable_viewer_sync = not self.enable_viewer_sync
                elif evt.action == "record_frames" and evt.value > 0:
                    self.record_frames = not self.record_frames

            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)

                # Wait for dt to elapse in real time.
                # This synchronizes the physics simulation with the rendering rate.
                self.gym.sync_frame_time(self.sim)

                # it seems like in some cases sync_frame_time still results in higher-than-realtime framerate
                # this code will slow down the rendering to real time
                now = time.time()
                delta = now - self.last_frame_time
                if self.render_fps < 0:
                    # render at control frequency
                    render_dt = self.dt * self.control_freq_inv  # render every control step
                else:
                    render_dt = 1.0 / self.render_fps

                if delta < render_dt:
                    time.sleep(render_dt - delta)

                self.last_frame_time = time.time()

            else:
                self.gym.poll_viewer_events(self.viewer)

            if self.record_frames:
                if not os.path.isdir(self.record_frames_dir):
                    os.makedirs(self.record_frames_dir, exist_ok=True)

                self.gym.write_viewer_image_to_file(self.viewer, join(self.record_frames_dir, f"frame_{self.control_steps}.png"))

            if self.virtual_display and mode == "rgb_array":
                img = self.virtual_display.grab()
                return np.array(img)

    def __parse_sim_params(self, physics_engine: str, config_sim: Dict[str, Any]) -> gymapi.SimParams:
        """Parse the config dictionary for physics stepping settings.

        Args:
            physics_engine: which physics engine to use. "physx" or "flex"
            config_sim: dict of sim configuration parameters
        Returns
            IsaacGym SimParams object with updated settings.
        """
        sim_params = gymapi.SimParams()

        # check correct up-axis
        if config_sim["up_axis"] not in ["z", "y"]:
            msg = f"Invalid physics up-axis: {config_sim['up_axis']}"
            print(msg)
            raise ValueError(msg)

        # assign general sim parameters
        sim_params.dt = config_sim["dt"]
        sim_params.num_client_threads = config_sim.get("num_client_threads", 0)
        sim_params.use_gpu_pipeline = config_sim["use_gpu_pipeline"]
        sim_params.substeps = config_sim.get("substeps", 2)

        # assign up-axis
        if config_sim["up_axis"] == "z":
            sim_params.up_axis = gymapi.UP_AXIS_Z
        else:
            sim_params.up_axis = gymapi.UP_AXIS_Y

        # assign gravity
        sim_params.gravity = gymapi.Vec3(*config_sim["gravity"])
        # print(f"sim_params : {sim_params.keys()}")
        # config_sim['physx']['gpu_max_rigid_patch_count'] = 2**24
        # config_sim['physx']['gpu_found_lost_pairs_capacity'] = 2**24
        # config_sim['physx']['gpu_max_rigid_contact_count'] = 2**24
        # config_sim['physx']['gpu_max_soft_body_contacts'] = 2**24
        # config_sim['physx']['gpu_max_particle_contacts'] = 2**24
        
        config_sim['physx']['max_gpu_contact_pairs'] = 2**24
        config_sim['physx']['num_subscenes'] = 2**24

        # configure physics parameters
        if physics_engine == "physx":
            # set the parameters
            if "physx" in config_sim:
                for opt in config_sim["physx"].keys():
                    if opt == "contact_collection":
                        setattr(sim_params.physx, opt, gymapi.ContactCollection(config_sim["physx"][opt]))
                    else:
                        setattr(sim_params.physx, opt, config_sim["physx"][opt])
        else:
            # set the parameters
            if "flex" in config_sim:
                for opt in config_sim["flex"].keys():
                    setattr(sim_params.flex, opt, config_sim["flex"][opt])

        # return the configured params
        return sim_params

    """
    Domain Randomization methods
    """

    def get_actor_params_info(self, dr_params: Dict[str, Any], env):
        """Generate a flat array of actor params, their names and ranges.

        Returns:
            The array
        """

        if "actor_params" not in dr_params:
            return None
        params = []
        names = []
        lows = []
        highs = []
        param_getters_map = get_property_getter_map(self.gym)
        for actor, actor_properties in dr_params["actor_params"].items():
            handle = self.gym.find_actor_handle(env, actor)
            for prop_name, prop_attrs in actor_properties.items():
                if prop_name == 'color':
                    continue  # this is set randomly
                props = param_getters_map[prop_name](env, handle)
                if not isinstance(props, list):
                    props = [props]
                for prop_idx, prop in enumerate(props):
                    for attr, attr_randomization_params in prop_attrs.items():
                        name = prop_name+'_' + str(prop_idx) + '_'+attr
                        lo_hi = attr_randomization_params['range']
                        distr = attr_randomization_params['distribution']
                        if 'uniform' not in distr:
                            lo_hi = (-1.0*float('Inf'), float('Inf'))
                        if isinstance(prop, np.ndarray):
                            for attr_idx in range(prop[attr].shape[0]):
                                params.append(prop[attr][attr_idx])
                                names.append(name+'_'+str(attr_idx))
                                lows.append(lo_hi[0])
                                highs.append(lo_hi[1])
                        else:
                            params.append(getattr(prop, attr))
                            names.append(name)
                            lows.append(lo_hi[0])
                            highs.append(lo_hi[1])
        return params, names, lows, highs

    ## create envs ## ## randomizations ##
    def apply_randomizations(self, dr_params):
        """Apply domain randomizations to the environment.

        Note that currently we can only apply randomizations only on resets, due to current PhysX limitations

        Args:
            dr_params: parameters for domain randomization to use.
        """

        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

        ### 
        # First, determine what to randomize:
        #   - non-environment parameters when > frequency steps have passed since the last non-environment
        #   - physical environments in the reset buffer, which have exceeded the randomization frequency threshold
        #   - on the first call, randomize everything
        self.last_step = self.gym.get_frame_count(self.sim)
        if self.first_randomization:
            do_nonenv_randomize = True
            env_ids = list(range(self.num_envs))
        else:
            do_nonenv_randomize = (self.last_step - self.last_rand_step) >= rand_freq
            rand_envs = torch.where(self.randomize_buf >= rand_freq, torch.ones_like(self.randomize_buf), torch.zeros_like(self.randomize_buf))
            rand_envs = torch.logical_and(rand_envs, self.reset_buf)
            env_ids = torch.nonzero(rand_envs, as_tuple=False).squeeze(-1).tolist()
            self.randomize_buf[rand_envs] = 0

        if do_nonenv_randomize:
            self.last_rand_step = self.last_step

        param_setters_map = get_property_setter_map(self.gym)
        param_setter_defaults_map = get_default_setter_args(self.gym)
        param_getters_map = get_property_getter_map(self.gym)

        # On first iteration, check the number of buckets
        if self.first_randomization:
            check_buckets(self.gym, self.envs, dr_params)

        for nonphysical_param in ["observations", "actions"]:
            if nonphysical_param in dr_params and do_nonenv_randomize:
                dist = dr_params[nonphysical_param]["distribution"]
                op_type = dr_params[nonphysical_param]["operation"]
                sched_type = dr_params[nonphysical_param]["schedule"] if "schedule" in dr_params[nonphysical_param] else None
                sched_step = dr_params[nonphysical_param]["schedule_steps"] if "schedule" in dr_params[nonphysical_param] else None
                op = operator.add if op_type == 'additive' else operator.mul

                if sched_type == 'linear':
                    sched_scaling = 1.0 / sched_step * \
                        min(self.last_step, sched_step)
                elif sched_type == 'constant':
                    sched_scaling = 0 if self.last_step < sched_step else 1
                else:
                    sched_scaling = 1

                if dist == 'gaussian':
                    mu, var = dr_params[nonphysical_param]["range"]
                    mu_corr, var_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        mu *= sched_scaling
                        var *= sched_scaling
                        mu_corr *= sched_scaling
                        var_corr *= sched_scaling
                    elif op_type == 'scaling':
                        var = var * sched_scaling  # scale up var over time
                        mu = mu * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                        var_corr = var_corr * sched_scaling  # scale up var over time
                        mu_corr = mu_corr * sched_scaling + 1.0 * \
                            (1.0 - sched_scaling)  # linearly interpolate

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * params['var_corr'] + params['mu_corr']
                        return op(
                            tensor, corr + torch.randn_like(tensor) * params['var'] + params['mu'])

                    self.dr_randomizations[nonphysical_param] = {'mu': mu, 'var': var, 'mu_corr': mu_corr, 'var_corr': var_corr, 'noise_lambda': noise_lambda}

                elif dist == 'uniform':
                    lo, hi = dr_params[nonphysical_param]["range"]
                    lo_corr, hi_corr = dr_params[nonphysical_param].get("range_correlated", [0., 0.])

                    if op_type == 'additive':
                        lo *= sched_scaling
                        hi *= sched_scaling
                        lo_corr *= sched_scaling
                        hi_corr *= sched_scaling
                    elif op_type == 'scaling':
                        lo = lo * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi = hi * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        lo_corr = lo_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)
                        hi_corr = hi_corr * sched_scaling + 1.0 * (1.0 - sched_scaling)

                    def noise_lambda(tensor, param_name=nonphysical_param):
                        params = self.dr_randomizations[param_name]
                        corr = params.get('corr', None)
                        if corr is None:
                            corr = torch.randn_like(tensor)
                            params['corr'] = corr
                        corr = corr * (params['hi_corr'] - params['lo_corr']) + params['lo_corr']
                        return op(tensor, corr + torch.rand_like(tensor) * (params['hi'] - params['lo']) + params['lo'])

                    self.dr_randomizations[nonphysical_param] = {'lo': lo, 'hi': hi, 'lo_corr': lo_corr, 'hi_corr': hi_corr, 'noise_lambda': noise_lambda}

        if "sim_params" in dr_params and do_nonenv_randomize:
            prop_attrs = dr_params["sim_params"]
            prop = self.gym.get_sim_params(self.sim)

            if self.first_randomization:
                self.original_props["sim_params"] = {
                    attr: getattr(prop, attr) for attr in dir(prop)}

            for attr, attr_randomization_params in prop_attrs.items():
                apply_random_samples(
                    prop, self.original_props["sim_params"], attr, attr_randomization_params, self.last_step)

            self.gym.set_sim_params(self.sim, prop)

        # If self.actor_params_generator is initialized: use it to
        # sample actor simulation params. This gives users the
        # freedom to generate samples from arbitrary distributions,
        # e.g. use full-covariance distributions instead of the DR's
        # default of treating each simulation parameter independently.
        extern_offsets = {}
        if self.actor_params_generator is not None:
            for env_id in env_ids:
                self.extern_actor_params[env_id] = \
                    self.actor_params_generator.sample()
                extern_offsets[env_id] = 0

        # randomise all attributes of each actor (hand, cube etc..)
        # actor_properties are (stiffness, damping etc..)

        # Loop over actors, then loop over envs, then loop over their props 
        # and lastly loop over the ranges of the params 

        for actor, actor_properties in dr_params["actor_params"].items():

            # Loop over all envs as this part is not tensorised yet 
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                # randomise dof_props, rigid_body, rigid_shape properties 
                # all obtained from the YAML file
                # EXAMPLE: prop name: dof_properties, rigid_body_properties, rigid_shape properties  
                #          prop_attrs: 
                #               {'damping': {'range': [0.3, 3.0], 'operation': 'scaling', 'distribution': 'loguniform'}
                #               {'stiffness': {'range': [0.75, 1.5], 'operation': 'scaling', 'distribution': 'loguniform'}
                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue

                    if prop_name == 'scale':
                        setup_only = prop_attrs.get('setup_only', False)
                        if (setup_only and not self.sim_initialized) or not setup_only:
                            attr_randomization_params = prop_attrs
                            sample = generate_random_samples(attr_randomization_params, 1,
                                                             self.last_step, None)
                            og_scale = 1
                            if attr_randomization_params['operation'] == 'scaling':
                                new_scale = og_scale * sample
                            elif attr_randomization_params['operation'] == 'additive':
                                new_scale = og_scale + sample
                            self.gym.set_actor_scale(env, handle, new_scale)
                        continue

                    prop = param_getters_map[prop_name](env, handle)
                    set_random_properties = True

                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                setup_only = attr_randomization_params.get('setup_only', False)
                                if (setup_only and not self.sim_initialized) or not setup_only:
                                    smpl = None
                                    if self.actor_params_generator is not None:
                                        smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                            extern_sample, extern_offsets[env_id], p, attr)
                                    apply_random_samples(
                                        p, og_p, attr, attr_randomization_params,
                                        self.last_step, smpl)
                                else:
                                    set_random_properties = False
                    else:
                        if self.first_randomization:
                            self.original_props[prop_name] = deepcopy(prop)
                        for attr, attr_randomization_params in prop_attrs.items():
                            setup_only = attr_randomization_params.get('setup_only', False)
                            if (setup_only and not self.sim_initialized) or not setup_only:
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], prop, attr)
                                apply_random_samples(
                                    prop, self.original_props[prop_name], attr,
                                    attr_randomization_params, self.last_step, smpl)
                            else:
                                set_random_properties = False

                    if set_random_properties:
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




# pass

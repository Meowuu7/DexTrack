# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import sys
import os
import operator
from copy import deepcopy
import random
import time

from isaacgym import gymapi
from isaacgym.gymutil import get_property_setter_map, get_property_getter_map, get_default_setter_args, apply_random_samples, check_buckets, generate_random_samples

import numpy as np
import torch


#'''
sys.path.append('/home/dermark/LEAP_Hand_API/python')
sys.path.append('/home/dermark/deoxys_control')
"""Moving robot joint positions to initial pose for starting new experiments."""
import argparse
import pickle
import threading
import time
from pathlib import Path
import torch
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.ik_utils import IKWrapper

import numpy as np

from leap_hand_utils.dynamixel_client import *
import leap_hand_utils.leap_hand_utils as lhu
import time






sys.path.append('/home/dermark/deoxys_control')

"""Moving robot joint positions to initial pose for starting new experiments."""
import argparse
import pickle
import threading
import time
from pathlib import Path
import torch
import subprocess
import os
import matplotlib.pyplot as plt
import numpy as np

from deoxys import config_root
from deoxys.franka_interface import FrankaInterface
from deoxys.utils import YamlConfig
from deoxys.utils.input_utils import input2action
from deoxys.utils.io_devices import SpaceMouse
from deoxys.utils.log_utils import get_deoxys_example_logger
from deoxys.utils.ik_utils import IKWrapper

#'''


class LeapNode:
    def __init__(self):
        ####Some parameters
        # self.ema_amount = float(rospy.get_param('/leaphand_node/ema', '1.0')) #take only current
        self.kP = 800
        self.kI = 0
        self.kD = 200
        self.curr_lim = 300
        self.prev_pos = self.pos = self.curr_pos = lhu.allegro_to_LEAPhand(np.zeros(16))
           
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
        #Enables position-current control mode and the default parameters, it commands a position and then caps the current so the motors don't overload
        self.dxl_client.sync_write(motors, np.ones(len(motors))*5, 11, 1)
        self.dxl_client.set_torque_enabled(motors, True)
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kP, 84, 2) # Pgain stiffness     
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kP * 0.75), 84, 2) # Pgain stiffness for side to side should be a bit less
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kI, 82, 2) # Igain
        self.dxl_client.sync_write(motors, np.ones(len(motors)) * self.kD, 80, 2) # Dgain damping
        self.dxl_client.sync_write([0,4,8], np.ones(3) * (self.kD * 0.75), 80, 2) # Dgain damping for side to side should be a bit less
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
        print("pose1",pose)
        pose = lhu.allegro_to_LEAPhand(pose, zeros=False)
        print("pose2",pose)
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
    

# Base class for RL tasks
class BaseTask():

    def __init__(self, cfg, enable_camera_sensors=False, is_meta=False, task_num=0):
        self.id = -1
        self.gym = gymapi.acquire_gym()

        self.enable_camera_sensors = enable_camera_sensors

        self.device_type = cfg.get("device_type", "cuda")
        self.device_id = cfg.get("device_id", 0)

        self.device = "cpu"
        if self.device_type == "cuda" or self.device_type == "GPU":
            self.device = "cuda" + ":" + str(self.device_id)

        self.headless = cfg["headless"]

        # double check!
        #self.graphics_device_id = self.device_id
        # for vision, set graphics_device_id to 0 in cfg
        self.graphics_device_id = cfg.get("graphics_device_id", self.device_id)
        print("self.device_id:",self.device_id)
        print("self.graphics_device_id",self.graphics_device_id)
        if enable_camera_sensors == False and self.headless == True:
            self.graphics_device_id = -1

        self.num_envs = cfg["env"]["numEnvs"]
        if is_meta:
            self.num_envs = cfg["env"]["numEnvs"] * task_num
            
        self.num_obs = cfg["env"]["numObservations"]
        self.num_states = cfg["env"].get("numStates", 0)
        self.num_actions = cfg["env"]["numActions"]

        self.control_freq_inv = cfg["env"].get("controlFrequencyInv", 1)

        # optimization flags for pytorch JIT
        torch._C._jit_set_profiling_mode(False)
        torch._C._jit_set_profiling_executor(False)
        # allocate buffers
        self.obs_buf = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.obs_buf2 = torch.zeros(
            (self.num_envs, self.num_obs), device=self.device, dtype=torch.float)
        self.states_buf = torch.zeros(
            (self.num_envs, self.num_states), device=self.device, dtype=torch.float)
        self.rew_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float)
        self.reset_buf = torch.ones(
            self.num_envs, device=self.device, dtype=torch.long)
        self.progress_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.randomize_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.object_id_buf = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long)
        self.extras = {}

        
        '''
        ainterface_cfg="charmander.yml"
        acontroller_cfg="joint-impedance-controller.yml"
    

        self.robot_interface = FrankaInterface(
            config_root + f"/{ainterface_cfg}", use_visualizer=False
        )
        self.controller_cfg = YamlConfig(config_root + f"/{acontroller_cfg}").as_easydict()

        self.controller_type = "JOINT_IMPEDANCE"
        self.ik_wrapper = IKWrapper()
        
        self.delta_position = np.array([0.03, 0.0, 0.0])

        
        self.leap_hand = LeapNode()


        #'''
        self.last2_time = None
        self.original_props = {}
        self.dr_randomizations = {}
        self.first_randomization = True
        self.actor_params_generator = None
        self.extern_actor_params = {}
        for env_id in range(self.num_envs):
            self.extern_actor_params[env_id] = None

        self.last_step = -1
        self.last_rand_step = -1

        # create envs, sim and viewer
        self.create_sim()
        self.gym.prepare_sim(self.sim)

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

            # set the camera position based on up axis
            sim_params = self.gym.get_sim_params(self.sim)
            if sim_params.up_axis == gymapi.UP_AXIS_Z:
                cam_pos = gymapi.Vec3(10, 10, 3.0)
                cam_target = gymapi.Vec3(0, 0, 0.0)
            else:
                cam_pos = gymapi.Vec3(10, 10, 3.0)
                cam_target = gymapi.Vec3(0, 0, 0.0)

            self.gym.viewer_camera_look_at(
                self.viewer, None, cam_pos, cam_target)

    # set gravity based on up axis and return axis index
    def set_sim_params_up_axis(self, sim_params, axis):
        if axis == 'z':
            sim_params.up_axis = gymapi.UP_AXIS_Z
            sim_params.gravity.x = 0
            sim_params.gravity.y = 0
            sim_params.gravity.z = -9.81
            return 2
        return 1

    def create_sim(self, compute_device, graphics_device, physics_engine, sim_params):
        sim = self.gym.create_sim(compute_device, graphics_device, physics_engine, sim_params)
        if sim is None:
            print("*** Failed to create sim")
            quit()

        return sim



    
    def step(self, actions, id):
        
        self.id = id
        if self.dr_randomizations.get('actions', None):
            actions = self.dr_randomizations['actions']['noise_lambda'](actions)

        # apply actions
        self.pre_physics_step(actions)
        '''
        if self.last2_time == None:
            self.last2_time = time.time_ns()
        current_time = time.time_ns()
        
        remaining_time = 0.05 - (
            current_time - self.last2_time
        ) / (10**9)
        if 0.0001 < remaining_time:
            time.sleep(remaining_time)
        self.last2_time = time.time_ns()
        print("nowatime",time.time())

        self.last_q = np.array(self.robot_interface.last_q)
        self.delta_position=0.004*self.actions[8, 0:3]
        print("self.actions",self.actions[8, :])
        print("self.delta_position",self.delta_position)
        joint_traj, debug_info = self.ik_wrapper.ik_trajectory_delta_position(self.delta_position.cpu().numpy(), self.last_q.tolist(), num_points=10)
        #joint_traj = self.ik_wrapper.interpolate_dense_traj(joint_traj)
        print("joint_traj",joint_traj)


        joint2=self.cur_targets[8, [11,1,2,3,4,5,6]].cpu().numpy()
        print("self.cur_targets[2, [11,1,2,3,4,5,6]]",self.cur_targets[2, [11,1,2,3,4,5,6]])
        print("self.cur_targets[:, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]]",self.cur_targets[8, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]])

        joint=joint_traj[9]
        action2 = joint2.tolist() + [-1.0]
        
        
        print("self.shadow_hand_dof_pos",self.shadow_hand_dof_pos[8,[8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]])

        #sim_min, sim_max = LEAPsim_limits(type = hack_thumb)
        #joints = scale(joints, sim_min, sim_max)
        #self.leap_hand.set_allegro(self.cur_targets[8, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]].cpu().numpy())
        self.leap_hand.set_allegro(self.shadow_hand_dof_pos[8, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]].cpu().numpy())

        #self.leaphand.sim_ones_to_LEAPhand(self.cur_targets[8, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]].cpu().numpy())


        #self.leap_hand.set_ones(self.cur_targets[8, [8,7,9,10,19,20,21,22,12,0,13,14,16,15,17,18]].cpu().numpy())



        self.robot_interface.control(
            controller_type=self.controller_type,
            action=action2,
            controller_cfg=self.controller_cfg,
        )



        print("self.leap_hand.read_pos()",self.leap_hand.read_pos())
        print("self.leap_hand.read_vel()",self.leap_hand.read_vel())
        print("self.leap_hand.read_cur()",self.leap_hand.read_cur())



        #'''
        
        # step physics and render each frame
        for i in range(self.control_freq_inv):
            self.render()
            # self.gym.simulate(self.sim)
        # self.render()
        self.gym.simulate(self.sim)

        #current_time = time.time()
        #print("current_time",current_time)


        # to fix!
        if self.device == 'cpu':
            self.gym.fetch_results(self.sim, True)
 
        # compute observations, rewards, resets, ...
        self.post_physics_step()

        #time.sleep(1)
        
        if self.dr_randomizations.get('observations', None):
            self.obs_buf = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf)
            self.obs_buf2 = self.dr_randomizations['observations']['noise_lambda'](self.obs_buf2)

    def get_states(self):
        return self.states_buf

    def render(self, sync_frame_time=False):
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
            
            # fetch results
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)

            # step graphics
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)
                self.gym.draw_viewer(self.viewer, self.sim, True)
            else:
                self.gym.poll_viewer_events(self.viewer)
        elif self.enable_camera_sensors:
            if self.device != 'cpu':
                self.gym.fetch_results(self.sim, True)
            if self.enable_viewer_sync:
                self.gym.step_graphics(self.sim)

    def get_actor_params_info(self, dr_params, env):
        """Returns a flat array of actor params, their names and ranges."""
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
                        name = prop_name+'_'+str(prop_idx)+'_'+attr
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

    # Apply randomizations only on resets, due to current PhysX limitations
    def apply_randomizations(self, dr_params):
        # If we don't have a randomization frequency, randomize every step
        rand_freq = dr_params.get("frequency", 1)

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

        for actor, actor_properties in dr_params["actor_params"].items():
            for env_id in env_ids:
                env = self.envs[env_id]
                handle = self.gym.find_actor_handle(env, actor)
                extern_sample = self.extern_actor_params[env_id]

                for prop_name, prop_attrs in actor_properties.items():
                    if prop_name == 'color':
                        num_bodies = self.gym.get_actor_rigid_body_count(
                            env, handle)
                        for n in range(num_bodies):
                            self.gym.set_rigid_body_color(env, handle, n, gymapi.MESH_VISUAL,
                                                          gymapi.Vec3(random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)))
                        continue
                    if prop_name == 'scale':
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
                    if isinstance(prop, list):
                        if self.first_randomization:
                            self.original_props[prop_name] = [
                                {attr: getattr(p, attr) for attr in dir(p)} for p in prop]
                        for p, og_p in zip(prop, self.original_props[prop_name]):
                            for attr, attr_randomization_params in prop_attrs.items():
                                smpl = None
                                if self.actor_params_generator is not None:
                                    smpl, extern_offsets[env_id] = get_attr_val_from_sample(
                                        extern_sample, extern_offsets[env_id], p, attr)
                                apply_random_samples(
                                    p, og_p, attr, attr_randomization_params,
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
        raise NotImplementedError

    def post_physics_step(self):
        raise NotImplementedError


def get_attr_val_from_sample(sample, offset, prop, attr):
    """Retrieves param value for the given prop and attr from the sample."""
    if sample is None:
        return None, 0
    if isinstance(prop, np.ndarray):
        smpl = sample[offset:offset+prop[attr].shape[0]]
        return smpl, offset+prop[attr].shape[0]
    else:
        return sample[offset], offset+1
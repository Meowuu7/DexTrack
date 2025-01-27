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

import numpy as np
import os
import torch

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgymenvs.utils.torch_jit_utils import scale, unscale, quat_mul, quat_conjugate, quat_from_angle_axis, \
    to_torch, get_axis_params, torch_rand_float, tensor_clamp, compute_heading_and_up, compute_rot, normalize_angle

from isaacgymenvs.tasks.vec_task import VecTask
from PIL import Image
import clip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC


import time
import grequests
import requests
from requests.adapters import HTTPAdapter, Retry
from io import BytesIO
import pickle


class Humanoid(VecTask):

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg
        # rl_device = 'cuda:0'
        # sim_device = 'cuda:0'
        # graphics_device_id = 0 
        self.randomization_params = self.cfg["task"]["randomization_params"]
        self.randomize = self.cfg["task"]["randomize"]
        self.dof_vel_scale = self.cfg["env"]["dofVelocityScale"]
        self.angular_velocity_scale = self.cfg["env"].get("angularVelocityScale", 0.1)
        self.contact_force_scale = self.cfg["env"]["contactForceScale"]
        self.power_scale = self.cfg["env"]["powerScale"]
        self.heading_weight = self.cfg["env"]["headingWeight"]
        self.up_weight = self.cfg["env"]["upWeight"]
        self.actions_cost_scale = self.cfg["env"]["actionsCost"]
        self.energy_cost_scale = self.cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self.cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self.cfg["env"]["deathCost"]
        self.termination_height = self.cfg["env"]["terminationHeight"]

        self.debug_viz = self.cfg["env"]["enableDebugVis"]
        self.plane_static_friction = self.cfg["env"]["plane"]["staticFriction"]
        self.plane_dynamic_friction = self.cfg["env"]["plane"]["dynamicFriction"]
        self.plane_restitution = self.cfg["env"]["plane"]["restitution"]

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.cfg["env"]["numObservations"] = 108
        self.cfg["env"]["numActions"] = 21
        # self.cfg['enableCameraSensors']=True

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        if self.viewer != None:
            cam_pos = gymapi.Vec3(50.0, 25.0, 2.4)
            cam_target = gymapi.Vec3(45.0, 25.0, 0.0)
            self.gym.viewer_camera_look_at(self.viewer, None, cam_pos, cam_target)

        # get gym GPU state tensors
        actor_root_state = self.gym.acquire_actor_root_state_tensor(self.sim)
        dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        sensor_tensor = self.gym.acquire_force_sensor_tensor(self.sim)

        sensors_per_env = 2
        self.vec_sensor_tensor = gymtorch.wrap_tensor(sensor_tensor).view(self.num_envs, sensors_per_env * 6)

        dof_force_tensor = self.gym.acquire_dof_force_tensor(self.sim)
        self.dof_force_tensor = gymtorch.wrap_tensor(dof_force_tensor).view(self.num_envs, self.num_dof)

        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.root_states = gymtorch.wrap_tensor(actor_root_state)
        self.initial_root_states = self.root_states.clone()
        self.initial_root_states[:, 7:13] = 0

        # create some wrapper tensors for different slices
        self.dof_state = gymtorch.wrap_tensor(dof_state_tensor)
        self.dof_pos = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 0]
        self.dof_vel = self.dof_state.view(self.num_envs, self.num_dof, 2)[..., 1]
        self.initial_dof_pos = torch.zeros_like(self.dof_pos, device=self.device, dtype=torch.float)
        zero_tensor = torch.tensor([0.0], device=self.device)
        self.initial_dof_pos = torch.where(self.dof_limits_lower > zero_tensor, self.dof_limits_lower,
                                           torch.where(self.dof_limits_upper < zero_tensor, self.dof_limits_upper, self.initial_dof_pos))
        self.initial_dof_vel = torch.zeros_like(self.dof_vel, device=self.device, dtype=torch.float)

        # initialize some data used later on
        self.up_vec = to_torch(get_axis_params(1., self.up_axis_idx), device=self.device).repeat((self.num_envs, 1))
        self.heading_vec = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self.targets = to_torch([1000, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.target_dirs = to_torch([1, 0, 0], device=self.device).repeat((self.num_envs, 1))
        self.dt = self.cfg["sim"]["dt"]
        self.potentials = to_torch([-1000./self.dt], device=self.device).repeat(self.num_envs)
        self.prev_potentials = self.potentials.clone()

        # adding CLIP model

        # if not cfg['test']:
        #     self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        #     text_opt =[
        #         # "The humanoid stands upright, reaching towards an object on a pedestal, with a slightly rigid posture and one arm close to its side.",
        #         # "The humanoid stands relaxed, hand wrapped around an object on a pedestal, other arm hanging loosely, exuding a calm and natural demeanor.",
        #         # "The humanoid is grasping an object in a natural pose.",
        #         # "The humanoid is grasping an object in a unnatural pose.",
        #         # "The robot is sitting in a lotus position."
        #         # "a humanoid robot practicing gymnastics, doing the sid e splits."
        #         # "a humanoid robot is walking forward in a natural posture."
        #         # "a humanoid robot is standing staticly in a natural posture."
        #         # "a humanoid robot walks forward by bending its knees and swinging its arms, just like a human."
        #         # "alternate its legs with a narrower base and swing its arms in sync with the opposite leg. The knees should have a slight bend, not be fully extended or spread too wide. is walking. The legs move in a rhythmic stride where one foot steps forward while the other supports the body's weight. The knees bend naturally during each step. Simultaneously, the arms swing in opposition to the legs — as the left leg moves forward, the right arm swings forward, and vice versa. This arm swing helps in maintaining balance and momentum. The torso remains upright and may rotate slightly with each stride. This pattern of movement is an efficient way to maintain balance and forward momentum."
        #         # "In robot walking, arms and legs move in a coordinated, alternating rhythm. As the left leg steps forward, the right arm swings forward, and vice versa. Knees bend naturally with each step, helping to maintain balance and efficiency."
        #         "a standing humanoid robot walks forward by alternating its legs with a narrower base and swing its arms in sync with the opposite leg. The knees should have a slight bend, not be fully extended or spread too wide."
        #     ]

        #     text = clip.tokenize(text_opt).to(self.device)
        #     with torch.no_grad():
        #         text_features = self.clip_model.encode_text(text)

        #         # image_features /= image_features.norm(dim=-1, keepdim=True)
        #         text_features /= text_features.norm(dim=-1, keepdim=True)
        #     self.text_features = text_features 

        self.render_list = [[] for i in range(len(self.envs))]

    def create_sim(self):
        self.up_axis_idx = 2 # index of up axis: Y=1, Z=2
        self.sim = super().create_sim(self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)

        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

        # If randomizing, apply once immediately on startup before the fist sim step
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

    def _create_ground_plane(self):
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        plane_params.static_friction = self.plane_static_friction
        plane_params.dynamic_friction = self.plane_dynamic_friction
        plane_params.restitution = self.plane_restitution
        plane_params.segmentation_id = 1
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../assets')
        asset_file = "mjcf/nv_humanoid.xml"

        if "asset" in self.cfg["env"]:
            asset_file = self.cfg["env"]["asset"].get("assetFileName", asset_file)

        asset_path = os.path.join(asset_root, asset_file)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.angular_damping = 0.01
        asset_options.max_angular_velocity = 100.0
        # Note - DOF mode is set in the MJCF file and loaded by Isaac Gym
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
        humanoid_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)

        # Note - for this asset we are loading the actuator info from the MJCF
        actuator_props = self.gym.get_asset_actuator_properties(humanoid_asset)
        motor_efforts = [prop.motor_effort for prop in actuator_props]

        # create force sensors at the feet
        right_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "right_foot")
        left_foot_idx = self.gym.find_asset_rigid_body_index(humanoid_asset, "left_foot")
        sensor_pose = gymapi.Transform()
        self.gym.create_asset_force_sensor(humanoid_asset, right_foot_idx, sensor_pose)
        self.gym.create_asset_force_sensor(humanoid_asset, left_foot_idx, sensor_pose)

        self.max_motor_effort = max(motor_efforts)
        self.motor_efforts = to_torch(motor_efforts, device=self.device)

        self.torso_index = 0
        self.num_bodies = self.gym.get_asset_rigid_body_count(humanoid_asset)
        self.num_dof = self.gym.get_asset_dof_count(humanoid_asset)
        self.num_joints = self.gym.get_asset_joint_count(humanoid_asset)

        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*get_axis_params(1.34, self.up_axis_idx))
        start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        self.start_rotation = torch.tensor([start_pose.r.x, start_pose.r.y, start_pose.r.z, start_pose.r.w], device=self.device)

        self.humanoid_handles = []
        self.envs = []
        self.dof_limits_lower = []
        self.dof_limits_upper = []
        # self.camera_handles = []

        self.cam_img_tensors = []
        self.cam_seg_tensors = []

        res = 512 
        res = 224 
        # res = 128
        camera_props = gymapi.CameraProperties()
        camera_props.width = res 
        camera_props.height = res
        camera_props.enable_tensors = True
        for i in range(self.num_envs):
            # create env instance
            env_ptr = self.gym.create_env(
                self.sim, lower, upper, num_per_row
            )
            handle = self.gym.create_actor(env_ptr, humanoid_asset, start_pose, "humanoid", i, 0, segmentationId=2)

            self.gym.enable_actor_dof_force_sensors(env_ptr, handle)

            for j in range(self.num_bodies):
                self.gym.set_rigid_body_color(
                    env_ptr, handle, j, gymapi.MESH_VISUAL, gymapi.Vec3(0.97, 0.38, 0.06))


            # create env camera
            camera_handle = self.gym.create_camera_sensor(env_ptr, camera_props)

            # local_pos = gymapi.Vec3(3.0, -1, 1.)
            # target_pos = gymapi.Vec3(0., 0., 1.) 
            # self.gym.set_camera_location(camera_handle, env_ptr, local_pos, target_pos)
            
            # attach camera to rigid body
            # local_transform = gymapi.Transform()
            # local_transform.p = gymapi.Vec3(1,1,1)
            # local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,1,0), np.radians(45.0))
            # camera_offset = gymapi.Vec3(3, -1, 0)
            camera_offset = gymapi.Vec3(2, 0, 0)
            # camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 1, 0), np.deg2rad(135))
            camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(0, 0, 1), np.deg2rad(180))
            
            actor_handle = handle 
            body_handle = self.gym.get_actor_rigid_body_handle(env_ptr, actor_handle, 0)
            self.gym.attach_camera_to_body(camera_handle, env_ptr, body_handle, gymapi.Transform(camera_offset, camera_rotation), gymapi.FOLLOW_POSITION)

            img_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_COLOR))
            img_seg_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, env_ptr, camera_handle, gymapi.IMAGE_SEGMENTATION)) 
            self.cam_img_tensors.append(img_tensor)
            self.cam_seg_tensors.append(img_seg_tensor)
            
             
            self.envs.append(env_ptr)
            self.humanoid_handles.append(handle)
        
        dof_prop = self.gym.get_actor_dof_properties(env_ptr, handle)
        for j in range(self.num_dof):
            if dof_prop['lower'][j] > dof_prop['upper'][j]:
                self.dof_limits_lower.append(dof_prop['upper'][j])
                self.dof_limits_upper.append(dof_prop['lower'][j])
            else:
                self.dof_limits_lower.append(dof_prop['lower'][j])
                self.dof_limits_upper.append(dof_prop['upper'][j])

        self.dof_limits_lower = to_torch(self.dof_limits_lower, device=self.device)
        self.dof_limits_upper = to_torch(self.dof_limits_upper, device=self.device)

        self.extremities = to_torch([5, 8], device=self.device, dtype=torch.long)

    def compute_reward(self, actions):
        # self.rew_buf[:], self.reset_buf = compute_humanoid_reward(
        #     self.obs_buf,
        #     self.reset_buf,
        #     self.progress_buf,
        #     self.actions,
        #     self.up_weight,
        #     self.heading_weight,
        #     self.potentials,
        #     self.prev_potentials,
        #     self.actions_cost_scale,
        #     self.energy_cost_scale,
        #     self.joints_at_limit_cost_scale,
        #     self.max_motor_effort,
        #     self.motor_efforts,
        #     self.termination_height,
        #     self.death_cost,
        #     self.max_episode_length
        # )
        self.rew_buf[:] = self.computer_clip_reward()
        self.rew_buf[:] = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.rew_buf[:]) * self.death_cost, self.rew_buf[:])

        self.reset_buf = torch.where(self.obs_buf[:, 0] < self.termination_height, torch.ones_like(self.reset_buf), self.reset_buf)
        self.reset_buf = torch.where(self.progress_buf >= self.max_episode_length - 1, torch.ones_like(self.reset_buf), self.reset_buf)

    def compute_observations(self):
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_actor_root_state_tensor(self.sim)

        self.gym.refresh_force_sensor_tensor(self.sim)

        self.gym.refresh_dof_force_tensor(self.sim)
        self.obs_buf[:], self.potentials[:], self.prev_potentials[:], self.up_vec[:], self.heading_vec[:] = compute_humanoid_observations(
            self.obs_buf, self.root_states, self.targets, self.potentials,
            self.inv_start_rot, self.dof_pos, self.dof_vel, self.dof_force_tensor,
            self.dof_limits_lower, self.dof_limits_upper, self.dof_vel_scale,
            self.vec_sensor_tensor, self.actions, self.dt, self.contact_force_scale, self.angular_velocity_scale,
            self.basis_vec0, self.basis_vec1)

    def reset_idx(self, env_ids):
        # Randomization can happen only at reset time, since it can reset actor positions on GPU
        if self.randomize:
            self.apply_randomizations(self.randomization_params)

        positions = torch_rand_float(-0.2, 0.2, (len(env_ids), self.num_dof), device=self.device)
        velocities = torch_rand_float(-0.1, 0.1, (len(env_ids), self.num_dof), device=self.device)

        self.dof_pos[env_ids] = tensor_clamp(self.initial_dof_pos[env_ids] + positions, self.dof_limits_lower, self.dof_limits_upper)
        self.dof_vel[env_ids] = velocities

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.initial_root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        to_target = self.targets[env_ids] - self.initial_root_states[env_ids, 0:3]
        to_target[:, self.up_axis_idx] = 0
        self.prev_potentials[env_ids] = -torch.norm(to_target, p=2, dim=-1) / self.dt
        self.potentials[env_ids] = self.prev_potentials[env_ids].clone()

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0

    def pre_physics_step(self, actions):
        self.actions = actions.to(self.device).clone()
        forces = self.actions * self.motor_efforts.unsqueeze(0) * self.power_scale
        force_tensor = gymtorch.unwrap_tensor(forces)
        self.gym.set_dof_actuation_force_tensor(self.sim, force_tensor)


    def computer_clip_reward(self):
        url = "http://10.220.5.4:8085"
        sess = requests.Session()
        retries = Retry(total=1000, backoff_factor=1, status_forcelist=[500], allowed_methods=False)
        sess.mount("http://", HTTPAdapter(max_retries=retries))
        
        
        jpeg_images = []
        req_list = []
        
        
        if self.cfg['test']:
            return torch.zeros_like(self.rew_buf)
            
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        self.gym.render_all_camera_sensors(self.sim)
        envs_num = len(self.envs)
        self.gym.start_access_image_tensors(self.sim)
        #
        # User code to digest tensors
        #
        try:
            len(self.img_lists)
        except:
            self.img_lists = []
            self.mask_lists= []
            
            
        # self.cam_img_tensors
        # self.cam_seg_tensors
        delta_gap = 5
        cache_length = 30
        cat_lenth = 10
        # cache_length = 30
        # cat_lenth = 25
        # Image.fromarray(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[eid], self.camera_handles[eid], gymapi.IMAGE_COLOR)).cpu().numpy()).show()

        clip_img_tensor_list = []
        with torch.no_grad():
            for eid in range(envs_num):
                img_tensor = self.cam_img_tensors[eid]
                img_seg_tensor = self.cam_seg_tensors[eid]
                img_tensor = img_tensor[..., :3]
                
                cmask = img_seg_tensor
                try:
                    non_zero_indices = torch.nonzero(cmask==2, as_tuple=True)[1]
                    sk_beg, sk_end = non_zero_indices.min()-delta_gap, non_zero_indices.max()+delta_gap
                    img_dim = img_seg_tensor.shape[0]
                    if sk_beg < 0:
                        sk_beg = 0
                    if sk_end >= img_dim:
                        sk_end = img_dim - 1
                    
                    if len(self.render_list[eid])>=cache_length:
                        self.render_list[eid].pop(0)
                    self.render_list[eid].append(img_tensor[:,sk_beg:sk_end,:3].clone())
                except:
                    self.render_list[eid].append(img_tensor.clone())


                cat_list = self.render_list[eid]
                cat_list = cat_list[::-5][::-1]
                final_image = torch.concat(cat_list, dim=1)
                
                img = Image.fromarray(final_image.cpu().numpy())

                # img = Image.fromarray(img_np[0])
                buffer = BytesIO()
                img.save(buffer, format="JPEG", quality=80)
                jpeg_images.append(buffer.getvalue())

                bs = 64
                if len(jpeg_images) == bs:
                    data = {
                        "images": jpeg_images,
                        "queries": [["Answer concisely: what is going on in this image?"]]*bs,
                        "answers": [["a standing humanoid robot walks forward by alternating its legs with a narrower base and swing its arms in sync with the opposite leg. The knees should have a slight bend, not be fully extended or spread too wide."]]*bs,
                    }
                    data_bytes = pickle.dumps(data)
                    req_list.append(grequests.post(url, data=data_bytes, timeout=250))
                    jpeg_images = []
        
            self.gym.end_access_image_tensors(self.sim)
            start_ = time.time()
            res_list = grequests.map(req_list)
            end_ = time.time()

            score_list = [np.array(pickle.loads(res.content)['recall']).reshape(-1) for res in res_list]
            scores_np = np.concatenate(score_list, axis=0)

            similarity = torch.from_numpy(scores_np).to(self.device) 
     
            # clip_img_tensors = torch.stack(clip_img_tensor_list, dim=0)
            # image_features = self.clip_model.encode_image(clip_img_tensors)
            # image_features /= image_features.norm(dim=-1, keepdim=True)
            # similarity = (1.0 * image_features @ self.text_features.T).squeeze(1)
        
        # img_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs_num-1], self.camera_handles[envs_num-1], gymapi.IMAGE_COLOR)) 
        # img_seg_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs_num-1], self.camera_handles[envs_num-1], gymapi.IMAGE_SEGMENTATION)) 
        # img_tensor = img_tensor[..., :3]
        # # img_tensor = img_tensor * (img_seg_tensor.unsqueeze(-1)==2)
        # # img_tensor = img_tensor * (img_seg_tensor.unsqueeze(-1)>0)
        # img = Image.fromarray(img_tensor.cpu().numpy()) 
        
        img_dir ='./render_tmp3'
        if not os.path.exists(img_dir):
            os.makedirs(img_dir)
        img.save('%s/%s.png' %(img_dir, time.time())) 


        
        # self.img_lists.append(img_tensor.clone())
        # self.mask_lists.append(img_seg_tensor.clone())
        
        # if len(self.img_lists) == 32:
        #     debug = 10
        
           
        return similarity


     
     
    def computer_clip_reward_v1(self):
        self.gym.render_all_camera_sensors(self.sim)
        self.gym.fetch_results(self.sim, True)
        self.gym.step_graphics(self.sim)
        envs_num = len(self.envs)
        self.gym.start_access_image_tensors(self.sim)
        #
        # User code to digest tensors
        #
        try:
            len(self.img_lists)
        except:
            self.img_lists = []
            self.mask_lists= []
            self.bg = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs_num-1], self.camera_handles[envs_num-1], gymapi.IMAGE_COLOR))[...,:3].clone()
        
        
        img_list = []
        # Image.fromarray(gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[eid], self.camera_handles[eid], gymapi.IMAGE_COLOR)).cpu().numpy()).show()
        with torch.no_grad():
            for eid in range(envs_num):
                img_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[eid], self.camera_handles[eid], gymapi.IMAGE_COLOR))
                img = Image.fromarray(img_tensor.cpu().numpy())
                img_processed = self.clip_preprocess(img)
                img_list.append(img_processed)
            render_imgs = torch.stack(img_list, dim=0).to(self.device)
            image_features = self.clip_model.encode_image(render_imgs)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (1.0 * image_features @ self.text_features.T).squeeze(1)
        
        img_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs_num-1], self.camera_handles[envs_num-1], gymapi.IMAGE_COLOR)) 
        img_seg_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[envs_num-1], self.camera_handles[envs_num-1], gymapi.IMAGE_SEGMENTATION)) 
        img_tensor = img_tensor[..., :3]
        # img_tensor = img_tensor * (img_seg_tensor.unsqueeze(-1)==2)
        # img_tensor = img_tensor * (img_seg_tensor.unsqueeze(-1)>0)
        img = Image.fromarray(img_tensor.cpu().numpy()) 
        img.save('./render_tmp/%s.png' %(time.time())) 


        
        self.img_lists.append(img_tensor.clone())
        self.mask_lists.append(img_seg_tensor.clone())
        
        if len(self.img_lists) == 32:
            debug = 10
        
        self.gym.end_access_image_tensors(self.sim)        
        return similarity

            # self.clip_preprocess(render_imgs)
            
        # with torch.no_grad():
        #     envs_num = len(self.envs)
        #     for eid in range(envs_num):
        #         img_tensor = gymtorch.wrap_tensor(self.gym.get_camera_image_gpu_tensor(self.sim, self.envs[eid], self.camera_handles[eid], gymapi.IMAGE_COLOR))[...,:3]
        #         img_list.append(img_tensor)
        #     render_imgs = torch.stack(img_list, dim=0) 
        #     self.clip_preprocess(render_imgs)
        
    
    def post_physics_step(self):
        self.progress_buf += 1
        self.randomize_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()

        # if 255 in env_ids:
        #     debug = 10
            
        if len(env_ids) > 0:
            self.reset_idx(env_ids)
        

        self.compute_observations()
        self.compute_reward(self.actions) 
        for eid in env_ids:
            self.render_list[eid] = []

        # debug viz
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)

            points = []
            colors = []
            for i in range(self.num_envs):
                origin = self.gym.get_env_origin(self.envs[i])
                pose = self.root_states[:, 0:3][i].cpu().numpy()
                glob_pos = gymapi.Vec3(origin.x + pose[0], origin.y + pose[1], origin.z + pose[2])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.heading_vec[i, 0].cpu().numpy(),
                               glob_pos.y + 4 * self.heading_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.heading_vec[i, 2].cpu().numpy()])
                colors.append([0.97, 0.1, 0.06])
                points.append([glob_pos.x, glob_pos.y, glob_pos.z, glob_pos.x + 4 * self.up_vec[i, 0].cpu().numpy(), glob_pos.y + 4 * self.up_vec[i, 1].cpu().numpy(),
                               glob_pos.z + 4 * self.up_vec[i, 2].cpu().numpy()])
                colors.append([0.05, 0.99, 0.04])

            self.gym.add_lines(self.viewer, None, self.num_envs * 2, points, colors)

#####################################################################
###=========================jit functions=========================###
#####################################################################


# @torch.jit.script
def compute_humanoid_reward(
    obs_buf,
    reset_buf,
    progress_buf,
    actions,
    up_weight,
    heading_weight,
    potentials,
    prev_potentials,
    actions_cost_scale,
    energy_cost_scale,
    joints_at_limit_cost_scale,
    max_motor_effort,
    motor_efforts,
    termination_height,
    death_cost,
    max_episode_length
):
    # type: (Tensor, Tensor, Tensor, Tensor, float, float, Tensor, Tensor, float, float, float, float, Tensor, float, float, float) -> Tuple[Tensor, Tensor]

    # reward from the direction headed
    heading_weight_tensor = torch.ones_like(obs_buf[:, 11]) * heading_weight
    heading_reward = torch.where(obs_buf[:, 11] > 0.8, heading_weight_tensor, heading_weight * obs_buf[:, 11] / 0.8)

    # reward for being upright
    up_reward = torch.zeros_like(heading_reward)
    up_reward = torch.where(obs_buf[:, 10] > 0.93, up_reward + up_weight, up_reward)

    actions_cost = torch.sum(actions ** 2, dim=-1)

    # energy cost reward
    motor_effort_ratio = motor_efforts / max_motor_effort
    scaled_cost = joints_at_limit_cost_scale * (torch.abs(obs_buf[:, 12:33]) - 0.98) / 0.02
    dof_at_limit_cost = torch.sum((torch.abs(obs_buf[:, 12:33]) > 0.98) * scaled_cost * motor_effort_ratio.unsqueeze(0), dim=-1)

    electricity_cost = torch.sum(torch.abs(actions * obs_buf[:, 33:54]) * motor_effort_ratio.unsqueeze(0), dim=-1)

    # reward for duration of being alive
    alive_reward = torch.ones_like(potentials) * 2.0
    progress_reward = potentials - prev_potentials

    total_reward = progress_reward + alive_reward + up_reward + heading_reward - \
        actions_cost_scale * actions_cost - energy_cost_scale * electricity_cost - dof_at_limit_cost

    # adjust reward for fallen agents
    total_reward = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(total_reward) * death_cost, total_reward)

    # reset agents
    reset = torch.where(obs_buf[:, 0] < termination_height, torch.ones_like(reset_buf), reset_buf)
    reset = torch.where(progress_buf >= max_episode_length - 1, torch.ones_like(reset_buf), reset)
    
    return total_reward, reset


@torch.jit.script
def compute_humanoid_observations(obs_buf, root_states, targets, potentials, inv_start_rot, dof_pos, dof_vel,
                                  dof_force, dof_limits_lower, dof_limits_upper, dof_vel_scale,
                                  sensor_force_torques, actions, dt, contact_force_scale, angular_velocity_scale,
                                  basis_vec0, basis_vec1):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, float, Tensor, Tensor, float, float, float, Tensor, Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]

    torso_position = root_states[:, 0:3]
    torso_rotation = root_states[:, 3:7]
    velocity = root_states[:, 7:10]
    ang_velocity = root_states[:, 10:13]

    to_target = targets - torso_position
    to_target[:, 2] = 0

    prev_potentials_new = potentials.clone()
    potentials = -torch.norm(to_target, p=2, dim=-1) / dt

    torso_quat, up_proj, heading_proj, up_vec, heading_vec = compute_heading_and_up(
        torso_rotation, inv_start_rot, to_target, basis_vec0, basis_vec1, 2)

    vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target = compute_rot(
        torso_quat, velocity, ang_velocity, targets, torso_position)

    roll = normalize_angle(roll).unsqueeze(-1)
    yaw = normalize_angle(yaw).unsqueeze(-1)
    angle_to_target = normalize_angle(angle_to_target).unsqueeze(-1)
    dof_pos_scaled = unscale(dof_pos, dof_limits_lower, dof_limits_upper)

    # obs_buf shapes: 1, 3, 3, 1, 1, 1, 1, 1, num_dofs (21), num_dofs (21), 6, num_acts (21)
    obs = torch.cat((torso_position[:, 2].view(-1, 1), vel_loc, angvel_loc * angular_velocity_scale,
                     yaw, roll, angle_to_target, up_proj.unsqueeze(-1), heading_proj.unsqueeze(-1),
                     dof_pos_scaled, dof_vel * dof_vel_scale, dof_force * contact_force_scale,
                     sensor_force_torques.view(-1, 12) * contact_force_scale, actions), dim=-1)

    return obs, potentials, prev_potentials_new, up_vec, heading_vec

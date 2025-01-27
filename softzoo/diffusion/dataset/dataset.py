import torch
import torch.nn.functional as F
import torch.utils.data
import numpy as np
import os, glob
# from data_loaders.humanml.data.utils import random_rotate_np
# from manopth.manolayer import ManoLayer
import utils
import pickle

# import data_loaders.humanml.data.utils as utils

import random
import trimesh
from scipy.spatial.transform import Rotation as R

# import utils.common_utils as common_utils

from torch_cluster import fps


import clip
import torch

clip_device =  "cpu"



# from utils.anchor_utils import masking_load_driver, anchor_load_driver, recover_anchor_batch


def farthest_point_sampling(pos: torch.FloatTensor, n_sampling: int):
    bz, N = pos.size(0), pos.size(1)
    feat_dim = pos.size(-1)
    device = pos.device
    sampling_ratio = float(n_sampling / N)
    pos_float = pos.float()

    batch = torch.arange(bz, dtype=torch.long).view(bz, 1).to(device)
    mult_one = torch.ones((N,), dtype=torch.long).view(1, N).to(device)

    batch = batch * mult_one
    batch = batch.view(-1)
    pos_float = pos_float.contiguous().view(-1, feat_dim).contiguous() # (bz x N, 3)
    # sampling_ratio = torch.tensor([sampling_ratio for _ in range(bz)], dtype=torch.float).to(device)
    # batch = torch.zeros((N, ), dtype=torch.long, device=device)
    sampled_idx = fps(pos_float, batch, ratio=sampling_ratio, random_start=False)
    # shape of sampled_idx?
    return sampled_idx

def load_ply_data(ply_fn):
    obj_mesh = trimesh.load(ply_fn, process=False)
    # obj_mesh.remove_degenerate_faces(height=1e-06)

    verts_obj = np.array(obj_mesh.vertices)
    faces_obj = np.array(obj_mesh.faces)
    obj_face_normals = np.array(obj_mesh.face_normals)
    obj_vertex_normals = np.array(obj_mesh.vertex_normals)

    print(f"vertex: {verts_obj.shape}, obj_faces: {faces_obj.shape}, obj_face_normals: {obj_face_normals.shape}, obj_vertex_normals: {obj_vertex_normals.shape}")
    return verts_obj, faces_obj

# 

class Uni_Manip_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        ### it seems that you can create such data ###
        ### with the graph A, 
        ## with the graph A, graph B ##
        ## ## but the distribution can hardly be optimized
        ## what is a general model? ## 
        ## actually you can genrate some data from it ##
        # actually you can generate some data from it
        self.data_folder = data_folder
        self.cfg = cfg
        # st lengths; ed lengths #
        ## manip description ##
        ## the graph node features ##
        ## 21 graph node features ##
        ## for all npys in the data folder -> it contains the manipulator graph's node features, edge connections ## 
        self.data_list = os.listdir(self.data_folder)
        self.data_list = [fn for fn in self.data_list if fn.endswith(".npy") and "uni_manip" in fn]
        self.data_name_list = [fn[:-4] for fn in self.data_list]
        self.data_name_to_data = {}
        pass
    
    # def 
    def __len__(self):
        return len(self.data_list)
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm + ".npy"
        cur_data_fn = os.path.join(self.data_folder, cur_data_fn)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    ## scale the data -- the manipulator length varies from the 0.1 - 0.4 -- therefore x 4 - 1 is ok --> -0.6 - 0.4 ## 
    ## scale the data -- the manipulator link connection values varies from 0.0 to 1.0 -- therefore x 2 - 1 is ok ##
    
    
    
    
    # cloest_training_data = 
    def get_closest_training_data(self, data_dict):
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        nn_bsz = data_dict['X'].shape[0]
        cloest_training_X, cloest_training_E = [], []
        for i_sample in range(nn_bsz):
            cur_X, cur_E = data_dict['X'][i_sample], data_dict['E'][i_sample]
            minn_dist_w_training = 9999.9
            minn_training_nm = None
            for cur_data_nm in self.data_name_to_data:
                cur_data_X = self.data_name_to_data[cur_data_nm]['X']
                cur_data_E = self.data_name_to_data[cur_data_nm]['E'] ## X and E ##
                cur_dist = np.linalg.norm(cur_data_X - cur_X) + np.linalg.norm(cur_data_E - cur_E)
                if cur_dist < minn_dist_w_training or minn_training_nm is None:
                    minn_dist_w_training = cur_dist
                    minn_training_nm = cur_data_nm
            ## get the current X and the current E #
            cur_cloest_X, cur_cloest_E = self.data_name_to_data[minn_training_nm]['X'], self.data_name_to_data[minn_training_nm]['E']
            cloest_training_X.append(cur_cloest_X)
            cloest_training_E.append(cur_cloest_E)
        cloest_training_X = np.stack(cloest_training_X, axis=0)
        cloest_training_E = np.stack(cloest_training_E, axis=0)
        cloest_training_data = {
            'X': cloest_training_X,
            'E': cloest_training_E
        }
        return cloest_training_data
    
    ## constraint mapping ## from the data? ##
    
    def inv_scale_data(self, data_dict):
        data_X = data_dict['X']
        data_E = data_dict['E']
        data_E = data_E[..., 0]
        data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        return {
            'X': data_X_inv_scaled, ## 
            'E': data_E_inv_scaled ## data_E_inv_scaled ##
        }
    # 
    
    def scale_data(self, data_dict):
        data_X = data_dict['X']
        data_E = data_dict['E']
        data_E = data_E[:, :, None]
        # data_X_scaled = data_X * 4.0 - 1.0 
        data_X_scaled =( data_X * 10.0 - 1.0 ) / 2.0 # [-0.5, 0.5]
        data_E_scaled =( data_E * 2.0 - 1.0) / 2.0  # [-0.5, 0.5]
        return {
            'X': data_X_scaled,
            'E': data_E_scaled
        }
        
    def data_dict_to_th(self, data_dict_np):
        data_dict_th = {
            # key: torch.from_numpy(data_dict_np[key]).float().cuda() for key in data_dict_np
            key: torch.from_numpy(data_dict_np[key]).float() for key in data_dict_np
            ## ? ## TODO: add the self.device in the init according to cfgs ###
        }
        return data_dict_th
    
    
    # def get_data_via_index(self, index) --> getitem
    def __getitem__(self, index):
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ### ##
        
        cur_data_scaled = self.scale_data(cur_data)    
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        
        return cur_data_scaled_th ### get the scaled data in th format  # 
    
    # def get_item ## constriant space -> the space with all constraint representations ##
    # for data -> data optimized with the corresponding constraint ## ## the corresponding constraints ## 
    # for data -> data optimized with the corresponding constraint ## ## the corresponding constraints ##
    # the constriant g operates on the data space ##
    # to what extent it can decide itself and to what extent it is influenced by others ##
    ## the sims 

    
    
class Uni_Manip_Act_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        ## from this one to the data with optimied res ## -- checkpoint best and the checkpoint last?
        tmp_data_list = os.listdir(self.data_folder)
        
        tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(self.data_folder, fn))]
        
        self.ckpt_nm = "ckpt_best.npy"
        
        self.data_list = []
        for fn in tmp_data_list:
            cur_data_ckpt_folder = os.path.join(self.data_folder, fn, "checkpoints")
            if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
                best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, self.ckpt_nm)
                if os.path.exists(best_ckpt_data_fn): # exist in the data folder #
                    self.data_list.append(fn)
        ### TODO: mvoe this parameter to the config file ###
        self.fixed_nn_nodes = 21
        
        self.max_link_rot_acc = 10.0
        self.min_link_rot_acc = -29.0
        self.extent_link_rot_acc = self.max_link_rot_acc - self.min_link_rot_acc
        
        self.max_link_trans_acc = 61.0
        self.min_link_trans_acc = -30.0
        self.extent_link_trans_acc = self.max_link_trans_acc - self.min_link_trans_acc
        
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        
    def __len__(self):
        return len(self.data_list)
    
    # def inverse_scale_dadta()
    def inv_scale_data(self, data):
        link_rot_acc_data, link_trans_acc_data = data[..., 0], data[..., 1:]
        link_rot_acc_data = (link_rot_acc_data * self.extent_link_rot_acc) + self.min_link_rot_acc
        link_trans_acc_data = (link_trans_acc_data * self.extent_link_trans_acc) + self.min_link_trans_acc
        if isinstance(data, torch.Tensor):
            scaled_data = torch.cat(
                [link_rot_acc_data.unsqueeze(-1), link_trans_acc_data], dim=-1
            )
        else:
            scaled_data = np.concatenate(
                [link_rot_acc_data[..., None], link_trans_acc_data], axis=-1
            )
        return scaled_data
    
    
    
    def get_closest_training_data(self, sampled_data): 
        
        if len(self.data_name_to_data) == 0:
            tot_closest_training_data = np.zeros_like(sampled_data.detach().cpu().numpy())
            return tot_closest_training_data
        
        nn_bsz = sampled_data.shape[0]
        tot_closest_training_data = []
        
        for i_sample in range(nn_bsz):
            cur_sampled_data = sampled_data[i_sample]
            
            closest_training_dist = None
            closest_training_data_nm = None
            
            for cur_data_nm in self.data_name_to_data:
                cur_sampled_data_np = cur_sampled_data.detach().cpu().numpy()
                cur_ori_data = self.data_name_to_data[cur_data_nm]
                
                dist_sampled_data_w_ori_data = np.sum(
                    (cur_sampled_data_np - cur_ori_data) ** 2
                ).item()
                if closest_training_dist is None or (dist_sampled_data_w_ori_data < closest_training_dist):
                    closest_training_dist = dist_sampled_data_w_ori_data
                    closest_training_data_nm = cur_data_nm # cur data nm ##
            ori_data = self.data_name_to_data[closest_training_data_nm]
            tot_closest_training_data.append(ori_data)
        tot_closest_training_data = np.stack(tot_closest_training_data, axis=0)
        return tot_closest_training_data
    
    
    def load_data_from_data_name(self, data_nm):
        # print(f"data_fn: {data_nm}")
        cur_data_folder_name = os.path.join(self.data_folder, data_nm)
        cur_data_ckpt_folder = os.path.join(cur_data_folder_name, "checkpoints")
        cur_data_ckpt_fn = os.path.join(cur_data_ckpt_folder, self.ckpt_nm)
        cur_data = np.load(cur_data_ckpt_fn, allow_pickle=True).item()  ## get the data dict ##
        # dict_keys(['particle_xs', 'particle_link_idx', 'link_rotational_accs', 'link_translational_accs', 'rigid_linear_translations'])
        cur_links_rot_acc = cur_data['link_rotational_accs'] ## nn_frames x nn_links x link_rot_dof ##
        cur_links_trans_acc = cur_data['link_translational_accs']
        
        # nn_frames x nn_links x link_rot_dof #
        if cur_links_rot_acc.shape[1] < self.fixed_nn_nodes:
            cur_links_rot_acc = np.concatenate(
                [cur_links_rot_acc, np.zeros((cur_links_rot_acc.shape[0], self.fixed_nn_nodes - cur_links_rot_acc.shape[1], cur_links_rot_acc.shape[2]), dtype=np.float32)], axis=1
            )
            cur_links_trans_acc = np.concatenate(
                [cur_links_trans_acc, np.zeros((cur_links_trans_acc.shape[0], self.fixed_nn_nodes - cur_links_trans_acc.shape[1], cur_links_trans_acc.shape[2]), dtype=np.float32)], axis=1
            )
        
        cur_links_rot_acc = torch.from_numpy(cur_links_rot_acc).float()
        cur_links_trans_acc = torch.from_numpy(cur_links_trans_acc).float() ## th style
        cur_links_rot_acc = cur_links_rot_acc.contiguous().transpose(1, 0).contiguous() ## nn_links x nn_frames x link_rot_dof ##
        cur_links_trans_acc = cur_links_trans_acc.contiguous().transpose(1, 0).contiguous() ## nn_links x nn_frames 
        
        ##### === scale the data === #####
        cur_links_trans_acc = (cur_links_trans_acc - self.min_link_trans_acc) / self.extent_link_trans_acc
        cur_links_rot_acc = (cur_links_rot_acc - self.min_link_rot_acc) / self.extent_link_rot_acc
        ##### === scale the data === #####
        
        
        cur_link_actions = torch.cat(
            [cur_links_rot_acc, cur_links_trans_acc], dim=-1 ## link rota and trans accs
        )
        # self.data_name_to_data[data_nm] = cur_link_actions # link
        return cur_link_actions
        
    ### TODO: check the data scale and for the data scalings ###
    def __getitem__(self, index):
        
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self.load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data # 
        else:
            cur_data = self.data_name_to_data[cur_data_nm]
        
        rt_dict = {
            'X': cur_data
        }    
        return rt_dict
        
        ## TODO: data scaling ### ## data scaling ## scale the data ###
        ## it should be a consrained sampling where the constraints are imposed via the sampled graph structures ##
        ## it should be a sampling process guided by the graph structure ## --- guided by the graph structure ##
        ## the extent jlink trans acc; the extent link rot acc; the min link trans acc; what's the sampled ##
        ## what's the sampled data operated on the sampled graph ##
        
        
        # return cur_data # the foramat is the torch tensor ## the format is the torch tensor ##
    
        # return super().__getitem__(index)
        # 



class Uni_Manip_PC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.sampled_particle_idxes = None
        
        ## then you should get the task nn timesteps ## ## use the target data
        ## get the task nn timesteps ##
        
        # get_manipulator_infos(self, nn_links, fixed_y_len=0.1, base_x_len=0.1):
        if self.use_target_data:
            # TODO: get nn_links from the config file ##
            nn_links = 5
            print(f"Start getting target data with {nn_links} links")
            fixed_y_len = 0.05
            base_x_len = 0.1
            self.get_manipulator_infos(nn_links, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
            # self.get_manipulator_infos(nn_links, fixed_y_len=0.1, base_x_len=0.1)
        
        ## get manipulator infos ##
        data_task_err_thres = 0.03
        data_trans_constraints_thres = 0.01
        
        
        
        exp_tags = ["expv4_projected", "expv4_projected_task_0", "expv4_projected_task_2"]

        ## root_data_folder ##
        
        self.data_list = []
        
        for exp_tag in exp_tags:
            cur_data_folder = os.path.join(self.data_folder, exp_tag)
            tmp_data_list = os.listdir(cur_data_folder)
            
            ## valid data list ##
            valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
            valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tag, valid_data_list_sv_fn)

            with open(valid_data_list_sv_fn, "r") as rf:
                cur_data_list = rf.readlines()
                # self.data_list = [fn.strip() for fn in self.data_list]
                cur_data_list = [fn.strip() for fn in cur_data_list]
                cur_data_list = [exp_tag + "/" + fn for fn in cur_data_list]
                self.data_list += cur_data_list
            
        ## from this one to the data with optimied res
        # tmp_data_list = os.listdir(self.data_folder)
        
        # data_task_err_thres = 0.03
        # data_trans_constraints_thres = 0.01
        
        # valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
        # valid_data_list_sv_fn = os.path.join(self.data_folder, valid_data_list_sv_fn)
        
        ### valid data list sv fn ###
        # with open(valid_data_list_sv_fn, "r") as rf:
        #     self.data_list = rf.readlines()
        #     self.data_list = [fn.strip() for fn in self.data_list]
        ## get dataset and the dataset -- valid data statistics taskerrthres0.03_transconsthres0.01.npy ##
            
        valid_data_statistics_sv_fn = f"valid_data_statistics_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.npy"
        valid_data_statistics_sv_fn = os.path.join(self.data_folder, "expv4", valid_data_statistics_sv_fn)
        valid_data_statistics = np.load(valid_data_statistics_sv_fn, allow_pickle=True).item()


        self.avg_particle_init_xs = valid_data_statistics['avg_particle_init_xs']
        self.std_particle_init_xs = valid_data_statistics['std_particle_init_xs']
        
        self.avg_particle_accs = valid_data_statistics['avg_particle_accs']
        self.std_particle_accs = valid_data_statistics['std_particle_accs'] ##
        
        print(f"avg_particle_init_xs: {self.avg_particle_init_xs}, std_particle_init_xs: {self.std_particle_init_xs}")
        
        tmp_data_list = self.data_list ### get the valid data list ##
        
        tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(self.data_folder, fn))]
        
        self.ckpt_nm = "ckpt_best_diff.npy"
        
        self.data_list = []
        for fn in tmp_data_list:
            cur_data_ckpt_folder = os.path.join(self.data_folder, fn, "checkpoints")
            if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
                best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, self.ckpt_nm)
                if os.path.exists(best_ckpt_data_fn): # exist in the data folder #
                    self.data_list.append(fn)
        ### TODO: mvoe this parameter to the config file ###
        self.fixed_nn_nodes = 21
        
        
        
        self.maxx_nn_pts = 5000
        
        self.max_link_rot_acc = 10.0
        self.min_link_rot_acc = -29.0
        self.extent_link_rot_acc = self.max_link_rot_acc - self.min_link_rot_acc
        
        self.max_link_trans_acc = 61.0
        self.min_link_trans_acc = -30.0
        self.extent_link_trans_acc = self.max_link_trans_acc - self.min_link_trans_acc
        
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        return len(self.data_list)
    
    def get_target_pcd(self,):
        
        target_init_particles = self.target_init_particles
        nn_act_dim = 2
        target_particle_acts =  np.random.randn(self.nn_timesteps, target_init_particles.shape[0], nn_act_dim)
        
        target_init_particles = target_init_particles[None, :, :]
        
        unscaled_data_dict = {
            'particle_xs': target_init_particles,
            'particle_accs': target_particle_acts,
            # 'particle_link_idxes': self.target_particle_link_idxes,
            # 'link_joint_pos': self.target_link_joint_pos,
            # 'link_joint_dir': self.target_link_joint_dir,
            # 'link_parent_idx': self.target_link_parent_idx
        }
        
        # self.target_init_particles = generated_test_link_dict['particles_xs']
        # self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        # self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        # self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        # # self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        # if self.sample_wconstraints:
        #     scaled_data = self.scale_data_wconstraints(unscaled_data_dict)
        # else:
        scaled_data = self.scale_data(unscaled_data_dict)
        
        
        target_init_particles = scaled_data['X'] # [0]
        print(f"get target init particles: {target_init_particles.shape}")
        
        
        return target_init_particles
        
        
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy"
        cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    
    # generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len=fixed_y_len, base_x_len=base_x_len)

    #### generate test links general flexy ### 
    ## TODO: get dim, nn_links_one_side, len_one_side, fixed_y_len, base_x_len ##
    def generate_test_links_general_flexy(self, dim, nn_links_one_side, len_one_side, fixed_y_len=0.1, base_x_len=0.1):
        per_link_len = len_one_side / float(nn_links_one_side)
        
        ## get nn_links_ ##
        dim = 2
        quality = 1  # Use a larger value for higher-res simulations ##
        n_particles, n_grid = 9000 * quality**2, 128 * quality
        
        n_particles = n_particles // 3 ## get nn_particels for each link ##
        
        particle_density = n_particles / (base_x_len * fixed_y_len)
        base_link_n_particles = particle_density * (base_x_len * fixed_y_len)
        child_link_n_particles = particle_density * (per_link_len * fixed_y_len)
        
        base_link_n_particles = int(base_link_n_particles)
        child_link_n_particles = int(child_link_n_particles)
        
        ## 
        rnd_xys = np.random.rand(base_link_n_particles, dim) * np.array([base_x_len, fixed_y_len], dtype=np.float32)[None, :]
        base_link_xys = np.array([0.45, 0.45], dtype=np.float32)[None, :] + rnd_xys ## get the random initialized xys ##
        base_link_particles_link_idx = np.zeros((base_link_n_particles, ), dtype=np.int32) # the bae link particles idxes 
        
        link_xys = [base_link_xys]
        link_particle_link_idx = [base_link_particles_link_idx]
        link_joint_pos = [np.array([0.0, 0.0], dtype=np.float32)]
        link_parent_idx = [-1]
        
        
        child_link_idx = 1
        joint_x = 0.45
        joint_y = 0.45
        for i_link in range(nn_links_one_side):
            cur_link_st_x = joint_x - per_link_len
            rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, fixed_y_len], dtype=np.float32)
            cur_link_xys = np.array([cur_link_st_x, joint_y], dtype=np.float32)[None, :] + rnd_xys
            cur_link_idxes = np.ones((child_link_n_particles, ), dtype=np.int32) * child_link_idx
            cur_link_joint_pos = np.array([joint_x, joint_y], dtype=np.float32)
            cur_link_parent_idx = child_link_idx - 1
            
            link_xys.append(cur_link_xys)
            link_particle_link_idx.append(cur_link_idxes)
            link_joint_pos.append(cur_link_joint_pos)
            link_parent_idx.append(cur_link_parent_idx)
            
            joint_x -= per_link_len
            child_link_idx += 1
        
        joint_x = 0.55
        joint_y = 0.45
        for i_link in range(nn_links_one_side):
            cur_link_st_x = joint_x # - per_link_len
            rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, fixed_y_len], dtype=np.float32)
            cur_link_xys = np.array([cur_link_st_x, joint_y], dtype=np.float32)[None, :] + rnd_xys
            cur_link_idxes = np.ones((child_link_n_particles, ), dtype=np.int32) * child_link_idx
            cur_link_joint_pos = np.array([joint_x, joint_y], dtype=np.float32)
            cur_link_parent_idx = child_link_idx - 1 if i_link > 0 else 0
            
            link_xys.append(cur_link_xys)
            link_particle_link_idx.append(cur_link_idxes)
            link_joint_pos.append(cur_link_joint_pos)
            link_parent_idx.append(cur_link_parent_idx)
            
            joint_x += per_link_len
            child_link_idx += 1
        
        obj_particles = np.concatenate(link_xys, axis=0)
        particle_link_idxes = np.concatenate(link_particle_link_idx, axis=0)
        link_joint_pos = np.stack(link_joint_pos, axis=0)
        link_parent_idx = np.array(link_parent_idx, dtype=np.int32)
        
        link_joint_dir = [[1.0, 0.0] for _ in range(nn_links_one_side * 2 + 1)]
        link_joint_dir = np.array(link_joint_dir, dtype=np.float32)
        
        ## link joint dir ##
        obj_info = {
            'particles_xs': obj_particles, ### obj particles ###
            'particle_link_idxes': particle_link_idxes, ### particle link idxes ###
            'link_joint_pos': link_joint_pos, ### link joint pos ###
            'link_joint_dir': link_joint_dir,
            'link_parent_idx': link_parent_idx
        }
        
        return obj_info
        
        # asset_root_folder = os.path.join(PROJ_ROOT_FOLDER, "assets")
        # os.makedirs(asset_root_folder, exist_ok=True)
        
        # obj_info_sv_fn = os.path.join(PROJ_ROOT_FOLDER, f"assets", f"obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_baseX_{base_x_len}_Y_{fixed_y_len}.npy")
        # np.save(obj_info_sv_fn, obj_info)
        # print(f"Object information saved to {obj_info_sv_fn}")
    
    
    def get_manipulator_infos(self, nn_links, fixed_y_len=0.1, base_x_len=0.1):
        ### get the manipulator infos; st_len; 
        ### 
        # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8]
        # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
        
        # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
        # tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
        
        ## create a new manipualtror ##
        nn_links = 5  ### TODO: get nn_links from the parameters passed to the function ##
        nn_links_one_side = (nn_links - 1) // 2 ### get the number of links in one side ##
        ## nn links one side ##
        # /data/xueyi/softzoo/expv4_projected_task_0/n_links_5_tag_iinst_8_nlinks_5_childlinklen_0.26666666666666666_curri_v2__nreg_inherit_True_seed_67_contact_spring_d_0.5_damping_0.1_ # 
        # /data/xueyi/softzoo/expv4_projected_task_0/n_links_5_tag_iinst_8_nlinks_5_childlinklen_0.30000000000000004_curri_v2__nreg_inherit_True_seed_67_contact_spring_d_0.5_damping_0.1_
        # 
        # generate_test_links_general_flexy(self, dim, nn_links_one_side, len_one_side, fixed_y_len=0.1, base_x_len=0.1):
        
        dim = 2
        nn_links_one_side = nn_links_one_side
        len_one_side = (0.26666666666666666 + 0.30000000000000004) / 2.0
        # fixed_y_len = 0.1
        # base_x_len = 0.1 ## base x len 
        generated_test_link_dict = self.generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len, base_x_len) ## get the generated test link dict ##
        ### we have particles; and the partile link idxes ##
        ## sample for the trajectories from the data and the model for further usage ##
        
        ##### target init particles; target particles link idxes; target link joint pos; target link joint dir; target link parent idx #####
        self.target_init_particles = generated_test_link_dict['particles_xs']
        self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        
        
        
        ## get the manipulator info ##
        # tot_nn_links_one_side = []
        # tot_len_one_side = []
        # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages // 2 - 1)
        # st_nn_link_one_side = 1
        # for i in range(nn_stages):
        #     cur_link_len = st_len_one_side + (i // 2) * link_len_one_side_interval
            
        #     tot_len_one_side.append(cur_link_len)
        #     tot_nn_links_one_side.append(st_nn_link_one_side)
            
        #     if i % 2 == 0:
        #         st_nn_link_one_side = st_nn_link_one_side * 2
        
        # print("tot_nn_links_one_side: ", tot_nn_links_one_side)
        # print(f"tot_len_one_side: {tot_len_one_side}")
        # return tot_nn_links_one_side, tot_len_one_side
    
    
    def get_closest_training_data(self, data_dict):
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        nn_bsz = data_dict['X'].shape[0]
        cloest_training_X, cloest_training_E = [], []
        for i_sample in range(nn_bsz):
            cur_X, cur_E = data_dict['X'][i_sample], data_dict['E'][i_sample]
            minn_dist_w_training = 9999.9
            minn_training_nm = None
            for cur_data_nm in self.data_name_to_data:
                cur_data_X = self.data_name_to_data[cur_data_nm]['X']
                cur_data_E = self.data_name_to_data[cur_data_nm]['E'] ## X and E ##
                cur_dist = np.linalg.norm(cur_data_X - cur_X) + np.linalg.norm(cur_data_E - cur_E)
                if cur_dist < minn_dist_w_training or minn_training_nm is None:
                    minn_dist_w_training = cur_dist
                    minn_training_nm = cur_data_nm
            ## get the current X and the current E #
            cur_cloest_X, cur_cloest_E = self.data_name_to_data[minn_training_nm]['X'], self.data_name_to_data[minn_training_nm]['E']
            cloest_training_X.append(cur_cloest_X)
            cloest_training_E.append(cur_cloest_E)
        cloest_training_X = np.stack(cloest_training_X, axis=0)
        cloest_training_E = np.stack(cloest_training_E, axis=0)
        cloest_training_data = {
            'X': cloest_training_X,
            'E': cloest_training_E
        }
        return cloest_training_data
    
    
    def inv_scale_data(self, data_dict):
        data_X = data_dict['X']
        data_E = data_dict['E']
        # data_E = data_E[..., 0]
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        ## inv_scale data ##
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        if isinstance(data_X, torch.Tensor):
            data_E_inv_scaled = (data_E * (torch.from_numpy(self.std_particle_accs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_accs[None][None]).float().to(data_X.device)
            data_X_inv_scaled = (data_X * (torch.from_numpy(self.std_particle_init_xs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_init_xs[None][None]).float().to(data_X.device)
        else:
            data_E_inv_scaled = (data_E * (self.std_particle_accs[None][None] + 1e-6)) + self.avg_particle_accs[None][None]
            data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None][None] + 1e-6)) + self.avg_particle_init_xs[None][None]
            
        
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6))
        
        print(f"in inv scaled data")
        return {
            'X': data_X_inv_scaled,
            'E': data_E_inv_scaled
        }
        
    ## get some out of distribution data for the following inferences ##
    
    
    
    
    def transform_pcd_wact_dict(self, data_dict):
        init_xs = data_dict['X']
        ### TODO: get dt from configs? ###
        pts_acts = data_dict["E"]
        nn_bszs = init_xs.shape[0]

        # dt = 1e-1
        dt = self.dt
        
        tot_pts_vels = []
        tot_pts_vecs = []

        for i_bsz in range(nn_bszs):
            cur_init_xs = init_xs[i_bsz]
            cur_pts_acts = pts_acts[i_bsz]

            nn_pts_dims = cur_init_xs.shape[-1]
            cur_pts_acts = cur_pts_acts.reshape(cur_pts_acts.shape[0], -1, nn_pts_dims)

            #### ==== get pts act shapes ==== ####
            print(f"[Transform PCDs with ACTs] cur_pts_acts: {cur_pts_acts.shape}")
            cur_pts_acts = np.transpose(cur_pts_acts, (1, 0, 2))
            print(f"[Transform PCDs with ACTs] cur_pts_acts: {cur_pts_acts.shape}")
            ## 
            cur_pts_vels = []
            cur_pts_vecs = []
            for i_fr in range(cur_pts_acts.shape[0]):
                if i_fr == 0:
                    cur_fr_pts_vels = np.zeros_like(cur_pts_acts[i_fr])
                    cur_fr_pts_vecs = np.zeros_like(cur_pts_acts[i_fr])
                else:
                    prev_fr_pts_vels = cur_pts_vels[-1]
                    prev_fr_pts_vecs = cur_pts_vecs[-1]
                    cur_fr_pts_accs = cur_pts_acts[i_fr]

                    cur_fr_pts_vels = prev_fr_pts_vels + dt * cur_fr_pts_accs
                    cur_fr_pts_vecs = prev_fr_pts_vecs + dt * prev_fr_pts_vels + dt * dt * cur_fr_pts_accs
                    
                cur_pts_vels.append(cur_fr_pts_vels)
                cur_pts_vecs.append(cur_fr_pts_vecs)
            cur_pts_vels = np.stack(cur_pts_vels, axis=0)
            cur_pts_vecs = np.stack(cur_pts_vecs, axis=0)
            
            tot_pts_vels.append(cur_pts_vels)
            tot_pts_vecs.append(cur_pts_vecs)
        
        tot_pts_vels = np.stack(tot_pts_vels, axis=0)
        tot_pts_vecs = np.stack(tot_pts_vecs, axis=0)
        
        rt_dict = {
            'X': init_xs,
            'E': pts_acts,
            'pts_vels': tot_pts_vels,
            'pts_vecs': tot_pts_vecs ## get total pts vels and the vecs ##
        }
        return rt_dict

        
    
    def scale_data(self, data_dict):
        
        particle_xs = data_dict['particle_xs']
        particle_acts = data_dict['particle_accs'] # T x act_dim #
        
        ## TODO: for this setting, random permuting particle xs for sampling points is reasonable; but is not a good strategy for non-uniform pcs 
        sampled_particle_idxes = np.random.permutation(particle_xs.shape[1])[: self.maxx_nn_pts] ## jet the sampled pts indexes 
        particle_xs = particle_xs[:, sampled_particle_idxes]
        particle_acts = particle_acts[:, sampled_particle_idxes]
        
         
        init_pos = particle_xs[0]


        particle_xs = (init_pos - self.avg_particle_init_xs[None]) / (self.std_particle_init_xs[None] + 1e-6)
        particle_acts = np.transpose(particle_acts, (1, 0, 2))
        particle_acts = particle_acts.reshape(particle_acts.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        particle_acts = (particle_acts - self.avg_particle_accs[None]) / (self.std_particle_accs[None] + 1e-6)
        
        
        return {
            'X': particle_xs,
            'E': particle_acts,
        }
        
    def scale_data_batched(self, data_dict):
        init_particle_xs = data_dict['X']
        particle_acts = data_dict['E']
        ## bsz x nn_particles x 3 
        print(f"[Batched data scaling] init_particle_xs: {init_particle_xs.size()}, particle_acts: {particle_acts.size()}")
        th_avg_particle_init_xs = torch.from_numpy(self.avg_particle_init_xs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_std_particle_init_xs = torch.from_numpy(self.std_particle_init_xs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        th_avg_particle_accs = torch.from_numpy(self.avg_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_std_particle_accs = torch.from_numpy(self.std_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        init_particle_xs = (init_particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + 1e-6)
        # particle_acts = particle_acts.transpose(1, 0).contiguous().view(particle_acts.size(1), -1)
        particle_acts = (particle_acts - th_avg_particle_accs) / (th_std_particle_accs + 1e-6)
        
        scaled_data = {
            'X': init_particle_xs, 
            'E': particle_acts
        }
        return scaled_data
        
    def scale_data_wconstraints(self, data_dict):
        
        particle_xs = data_dict['particle_xs']
        particle_acts = data_dict['particle_accs'] # T x act_dim #
        
        # self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        # self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        # self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        # self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        particle_link_idxes = data_dict['particle_link_idxes'] ## nn_original_particles 
        link_joint_pos = data_dict['link_joint_pos']
        link_joint_dir = data_dict['link_joint_dir']
        link_parent_idx = data_dict['link_parent_idx']
        
        
        
        ## TODO: for this setting, random permuting particle xs for sampling points is reasonable; but is not a good strategy for non-uniform pcs 
        if self.sampled_particle_idxes is None:
            sampled_particle_idxes = np.random.permutation(particle_xs.shape[1])[: self.maxx_nn_pts] ## jet the sampled pts indexes 
            self.sampled_particle_idxes = sampled_particle_idxes
        else:
            sampled_particle_idxes = self.sampled_particle_idxes ## get a unified sampled particle idxes ## 
        
        
        particle_xs = particle_xs[:, sampled_particle_idxes]
        particle_acts = particle_acts[:, sampled_particle_idxes]
        
        particle_link_idxes = particle_link_idxes[sampled_particle_idxes]
        ## inv scale -> project -> scale -> resample ##
        
         
        init_pos = particle_xs[0]


        particle_xs = (init_pos - self.avg_particle_init_xs[None]) / (self.std_particle_init_xs[None] + 1e-6)
        particle_acts = np.transpose(particle_acts, (1, 0, 2))
        particle_acts = particle_acts.reshape(particle_acts.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        particle_acts = (particle_acts - self.avg_particle_accs[None]) / (self.std_particle_accs[None] + 1e-6)
        
        
        return {
            'X': particle_xs,
            'E': particle_acts,
            'particle_link_idxes': particle_link_idxes,
            'link_joint_pos': link_joint_pos,
            'link_joint_dir': link_joint_dir,
            'link_parent_idx': link_parent_idx
        }
        
    def data_dict_to_th(self, data_dict_np):
        data_dict_th = {
            # key: torch.from_numpy(data_dict_np[key]).float().cuda() for key in data_dict_np
            key: torch.from_numpy(data_dict_np[key]).float() for key in data_dict_np
            ## ? ## TODO: add the self.device in the init according to cfgs ###
        }
        return data_dict_th
    
    ## get the target data from them ##
    def get_target_data(self,):
        nn_timesteps = 10
        nn_act_dim = 2
        target_init_particles = self.target_init_particles ## constraint projections with the original cosntriants ## 
        ## fit for a set of rotations and the translations ---> to satisfy the segmentation constraints ##
        ## project the tranaltions to satisfy joint constraints ## --> finally we the particle sequences ##
        ## project 
        ### nn_timesstpes x nn_particles x nn_act_flatten_dim ###
        target_particle_acts =  np.random.randn(nn_timesteps, target_init_particles.shape[0], nn_act_dim)
        
        target_init_particles = target_init_particles[None, :, :]
        
        unscaled_data_dict = {
            'particle_xs': target_init_particles,
            'particle_accs': target_particle_acts,
            'particle_link_idxes': self.target_particle_link_idxes,
            'link_joint_pos': self.target_link_joint_pos,
            'link_joint_dir': self.target_link_joint_dir,
            'link_parent_idx': self.target_link_parent_idx
        }
        
        # self.target_init_particles = generated_test_link_dict['particles_xs']
        # self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        # self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        # self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        # self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        if self.sample_wconstraints:
            scaled_data = self.scale_data_wconstraints(unscaled_data_dict)
        else:
            scaled_data = self.scale_data(unscaled_data_dict)
        
        # scaled_data = self.scale_data(unscaled_data_dict) ## get the scaled dta and unscaled data dict ##
        
        return scaled_data
    
    # def get_data_via_index(self, index) -->getitem ##
    def __getitem__(self, index):
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ### ##
        
        ## TODO: data selecting, data parsing, and data scaling ##
        
        # if self.use_target_data:
        #     cur_data_scaled = self.get_target_data()
        # else:
        
        cur_data_scaled = self.scale_data(cur_data) ## scale the data
        
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = {
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        
        return cur_data_scaled_th ### get the scaled data in th format 
    




class Uni_Manip_PCSeg_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        ## from this one to the data with optimied res ## -- checkpoint best and the checkpoint last?
        tmp_data_list = os.listdir(self.data_folder)
        
        data_task_err_thres = 0.03
        data_trans_constraints_thres = 0.01
        
        valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
        valid_data_list_sv_fn = os.path.join(self.data_folder, valid_data_list_sv_fn)
        
        
        with open(valid_data_list_sv_fn, "r") as rf:
            self.data_list = rf.readlines()
            self.data_list = [fn.strip() for fn in self.data_list]
            
        valid_data_statistics_sv_fn = f"valid_data_statistics_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.npy"
        valid_data_statistics_sv_fn = os.path.join(self.data_folder, valid_data_statistics_sv_fn)
        valid_data_statistics = np.load(valid_data_statistics_sv_fn, allow_pickle=True).item() ##

        self.avg_particle_init_xs = valid_data_statistics['avg_particle_init_xs']
        self.std_particle_init_xs = valid_data_statistics['std_particle_init_xs']
        self.avg_particle_accs = valid_data_statistics['avg_particle_accs']
        self.std_particle_accs = valid_data_statistics['std_particle_accs'] ## 
        
        tmp_data_list = self.data_list ### get the valid data list ##    
        
        tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(self.data_folder, fn))]
        
        self.ckpt_nm = "ckpt_best_diff.npy"
        
        self.data_list = []
        for fn in tmp_data_list:
            cur_data_ckpt_folder = os.path.join(self.data_folder, fn, "checkpoints")
            if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
                best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, self.ckpt_nm)
                if os.path.exists(best_ckpt_data_fn): # exist in the data folder #
                    self.data_list.append(fn)
        ### TODO: mvoe this parameter to the config file ###
        self.fixed_nn_nodes = 21
        
        self.load_fr_eval = False
        
        ### TODO: add the filters for the task completion errors
        ### TODO: add filters for the joint constraints errors
        ### TODO: sames not necessary to use tetemore points ## 
        ### points ## get the categorical values ## --- actually it is a discrete types of the model but you can use continuous type of data for approximating ##
        ## points for getting the categorical values ##
        
        
        self.maxx_nn_pts = 5000
        
        self.max_link_rot_acc = 10.0
        self.min_link_rot_acc = -29.0
        self.extent_link_rot_acc = self.max_link_rot_acc - self.min_link_rot_acc
        
        self.max_link_trans_acc = 61.0
        self.min_link_trans_acc = -30.0
        self.extent_link_trans_acc = self.max_link_trans_acc - self.min_link_trans_acc
        
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy"
        cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        
        
        
        return cur_data
    
    
    def get_closest_training_data(self, data_dict):
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data

        
        nn_bsz = data_dict['X'].shape[0]
        cloest_training_X, cloest_training_E = [], []
        for i_sample in range(nn_bsz):
            cur_X, cur_E = data_dict['X'][i_sample], data_dict['E'][i_sample]
            minn_dist_w_training = 9999.9
            minn_training_nm = None
            for cur_data_nm in self.data_name_to_data:
                cur_data_X = self.data_name_to_data[cur_data_nm]['X']
                cur_data_E = self.data_name_to_data[cur_data_nm]['E'] ## X and E ##
                cur_dist = np.linalg.norm(cur_data_X - cur_X) + np.linalg.norm(cur_data_E - cur_E)
                if cur_dist < minn_dist_w_training or minn_training_nm is None:
                    minn_dist_w_training = cur_dist
                    minn_training_nm = cur_data_nm
            ## get the current X and the current E #
            cur_cloest_X, cur_cloest_E = self.data_name_to_data[minn_training_nm]['X'], self.data_name_to_data[minn_training_nm]['E']
            cloest_training_X.append(cur_cloest_X)
            cloest_training_E.append(cur_cloest_E)
        cloest_training_X = np.stack(cloest_training_X, axis=0)
        cloest_training_E = np.stack(cloest_training_E, axis=0)
        cloest_training_data = {
            'X': cloest_training_X,
            'E': cloest_training_E
        }
        return cloest_training_data
    
    ## inv scale data ### ## self.data_dict ## ## self
    def inv_scale_data(self, data_dict):
        data_X = data_dict['X']
        data_E = data_dict['E']
        # data_E = data_E[..., 0]
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        if isinstance(data_X, torch.Tensor):
            # data_E_inv_scaled = (data_E * (torch.from_numpy(self.std_particle_accs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_accs[None][None]).float().to(data_X.device)
            data_X_inv_scaled = (data_X * (torch.from_numpy(self.std_particle_init_xs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_init_xs[None][None]).float().to(data_X.device)
        else:
            # data_E_inv_scaled = (data_E * (self.std_particle_accs[None][None] + 1e-6)) + self.avg_particle_accs[None][None]
            data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None][None] + 1e-6)) + self.avg_particle_init_xs[None][None]
        
        data_E_inv_scaled = data_E
        
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6))
        
        return {
            'X': data_X_inv_scaled,
            'E': data_E_inv_scaled
        }
    
    
    
    def scale_data(self, data_dict):
        
        particle_xs = data_dict['particle_xs']
        # particle_acts = data_dict['particle_accs']
        
        particle_link_idxes = data_dict['particle_link_idx']
        
        sampled_particle_idxes = np.random.permutation(particle_xs.shape[1])[: self.maxx_nn_pts]
        particle_xs = particle_xs[:, sampled_particle_idxes]
        particle_link_idxes = particle_link_idxes[sampled_particle_idxes]
        
        init_pos = particle_xs[0]

        particle_xs = (init_pos - self.avg_particle_init_xs[None]) / (self.std_particle_init_xs[None] + 1e-6)

        nn_maxx_links = 21
        particle_link_segs = []
        for i_particle in range(particle_link_idxes.shape[0]):
            particle_link_seg_idx = particle_link_idxes[i_particle] ## the link seg idx ## 
            ### TODO: need to get the segmentation index --- but what's the format of the seg_idx? ###
            particle_link_seg_categorical_vec = np.zeros((nn_maxx_links, ), dtype=np.float32)
            particle_link_seg_categorical_vec[particle_link_seg_idx] = 1.0 ## 

            particle_link_segs.append(particle_link_seg_categorical_vec)
        particle_link_segs = np.stack(particle_link_segs, axis=0)


        return {
            'X': particle_xs,
            'E': particle_link_segs
        }
    
    
    
    def calibrate_data_dict_from_sv_dict(self, sv_dict_fn):
        data_dict = np.load(sv_dict_fn, allow_pickle=True).item()
        init_particle_xs = data_dict['X']
        
        for key in data_dict: # 
            cur_value = data_dict[key]
            print(f"key: {key}, cur_value: {cur_value.shape}")
        
        if isinstance(init_particle_xs, torch.Tensor):
            scaled_particle_xs = (init_particle_xs - torch.from_numpy(self.avg_particle_init_xs[None][None]).float().to(init_particle_xs.device)) / (torch.from_numpy(self.std_particle_init_xs[None][None]).float().to(init_particle_xs.device) + 1e-6)
            scaled_particle_xs = scaled_particle_xs.detach().cpu().numpy()
        else:
            scaled_particle_xs = (init_particle_xs - self.avg_particle_init_xs[None][None]) / (self.std_particle_init_xs[None][None] + 1e-6)
        
        # scaled_particle_xs = 
        
        nn_maxx_links = 21
        particle_link_segs = np.random.randn(scaled_particle_xs.shape[0], scaled_particle_xs.shape[1], nn_maxx_links)
        
        rt_data_dict = {
            'X': scaled_particle_xs,
            'E': particle_link_segs
        }
        
        # rt_data_dict = self.data_dict_to_th(rt_data_dict)
        
        self.data_name_list = []
        self.data_name_to_data = {}
        
        for i_bsz in range(scaled_particle_xs.shape[0]):
            self.data_name_list.append(i_bsz)
            self.data_name_to_data[i_bsz] = {
                'X': rt_data_dict['X'][i_bsz],
                'E': rt_data_dict['E'][i_bsz]
            }
        self.load_fr_eval = True
        
        return rt_data_dict
        
        
        
        
    def data_dict_to_th(self, data_dict_np):
        data_dict_th = {
            # key: torch.from_numpy(data_dict_np[key]).float().cuda() for key in data_dict_np
            key: torch.from_numpy(data_dict_np[key]).float() for key in data_dict_np
            ## ? ## TODO: add the self.device in the init according to cfgs ###
        }
        return data_dict_th
    
    # def get_data_via_index(self, index) -->getitem ##
    def __getitem__(self, index):
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ### ##
        
        ## TODO: data selecting, data parsing, and data scaling ##
        
        if not self.load_fr_eval:
            cur_data_scaled = self.scale_data(cur_data) ## scale the data
        else:
            cur_data_scaled = cur_data
    
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        
        return cur_data_scaled_th ### get the scaled data in th format 
  
  
# ## uni manip ## #

class Uni_Manip_3D_PC_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        
        self.sampled_particle_idxes = None
        
        # self.sampled_particle_idxes = None
        
        ## then you should get the task nn timesteps ##
        ## get the task nn timesteps ##
        
        # TODO: implement thsi logic where the target data is the target 3D dex manipulator #
        # if self.use_target_data:
        #     # TODO: get nn_links from the config file ##
        #     nn_links = 5
        #     print(f"Start getting target data with {nn_links} links")
        #     fixed_y_len = 0.05
        #     base_x_len = 0.1
        #     self.get_manipulator_infos(nn_links, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
        #     # self.get_manipulator_infos(nn_links, fixed_y_len=0.1, base_x_len=0.1)
        
        ## get manipulator infos ##
        # data_task_err_thres = 0.03
        # data_trans_constraints_thres = 0.01
        
        
        
        # TODO: add the valid data filters #
        
        
        self.data_inst_fn = "saved_info_accs_v2.npy"
        exp_tags = ["tds_exp_2"]

        ## root_data_folder ##
        self.data_list = []
        
        for exp_tag in exp_tags:
            
            cur_data_folder = os.path.join(self.data_folder, exp_tag)
            tmp_data_list = os.listdir(cur_data_folder)
            tmp_data_list = ["allegro_bouncing_ball_task_0_trail6_"]
            # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
            # tmp_data_list = ["allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_"]
            for cur_subfolder in tmp_data_list:
                inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                if os.path.isdir(inst_folder):
                    cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                    if os.path.exists(cur_inst_file):
                        self.data_list.append(cur_inst_file)

        #
        ### TODO: currently we do not have the threshold for task completion
        # valid_data_list_sv_fn = f"valid_data_statistics.npy"
        valid_data_list_sv_fn = f"valid_data_statistics_v2.npy"
        valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
        
        ### TODO: calculate such statistics ##
        valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        self.avg_particle_init_xs = valid_data_statistics['robot_init_visual_pts_avg']
        self.std_particle_init_xs = valid_data_statistics['robot_init_visual_pts_std']
        self.avg_particle_accs_tau = valid_data_statistics['pts_accs_tau_avg']
        self.std_particle_accs_tau = valid_data_statistics['pts_accs_tau_std']
        self.avg_particle_accs = valid_data_statistics['pts_accs_avg']
        self.std_particle_accs = valid_data_statistics['pts_accs_std']
        self.avg_particle_accs_final = valid_data_statistics['pts_accs_final_avg']
        self.std_particle_accs_final = valid_data_statistics['pts_accs_final_std']
        
        
        
        # valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
        # valid_data_list_sv_fn = os.path.join(self.data_folder, valid_data_list_sv_fn)
        
        
        
        # valid_data_statistics_sv_fn = f"valid_data_statistics_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.npy"
        # valid_data_statistics_sv_fn = os.path.join(self.data_folder, "expv4", valid_data_statistics_sv_fn)
        # valid_data_statistics = np.load(valid_data_statistics_sv_fn, allow_pickle=True).item()


        # self.avg_particle_init_xs = valid_data_statistics['avg_particle_init_xs']
        # self.std_particle_init_xs = valid_data_statistics['std_particle_init_xs']
        
        # self.avg_particle_accs = valid_data_statistics['avg_particle_accs']
        # self.std_particle_accs = valid_data_statistics['std_particle_accs'] ##
        
        # print(f"avg_particle_init_xs: {self.avg_particle_init_xs}, std_particle_init_xs: {self.std_particle_init_xs}")
        
        # tmp_data_list = self.data_list ### get the valid data list ##
        
        # tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(self.data_folder, fn))]
        
        # self.ckpt_nm = "ckpt_best_diff.npy"
        
        # self.data_list = []
        # for fn in tmp_data_list:
        #     cur_data_ckpt_folder = os.path.join(self.data_folder, fn, "checkpoints")
        #     if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
        #         best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, self.ckpt_nm)
        #         if os.path.exists(best_ckpt_data_fn): # exist in the data folder #
        #             self.data_list.append(fn)
        # ### TODO: mvoe this parameter to the config file ###
        # self.fixed_nn_nodes = 21
        
        
        
        self.maxx_nn_pts = 5000
        
        # self.max_link_rot_acc = 10.0
        # self.min_link_rot_acc = -29.0
        # self.extent_link_rot_acc = self.max_link_rot_acc - self.min_link_rot_acc
        
        # self.max_link_trans_acc = 61.0
        # self.min_link_trans_acc = -30.0
        # self.extent_link_trans_acc = self.max_link_trans_acc - self.min_link_trans_acc
        
        # with various constraint modeling # 
        
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        
        self._preload_data()
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        return len(self.data_name_list)
    
    ## TODO: currently not in use ##
    def get_target_pcd(self,):
        
        target_init_particles = self.target_init_particles
        nn_act_dim = 2
        target_particle_acts =  np.random.randn(self.nn_timesteps, target_init_particles.shape[0], nn_act_dim)
        
        target_init_particles = target_init_particles[None, :, :]
        
        unscaled_data_dict = {
            'particle_xs': target_init_particles,
            'particle_accs': target_particle_acts,
            # 'particle_link_idxes': self.target_particle_link_idxes,
            # 'link_joint_pos': self.target_link_joint_pos,
            # 'link_joint_dir': self.target_link_joint_dir,
            # 'link_parent_idx': self.target_link_parent_idx
        }
        
        # self.target_init_particles = generated_test_link_dict['particles_xs']
        # self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        # self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        # self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        # # self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        # if self.sample_wconstraints:
        #     scaled_data = self.scale_data_wconstraints(unscaled_data_dict)
        # else:
        scaled_data = self.scale_data(unscaled_data_dict)
        
        
        target_init_particles = scaled_data['X'] # [0]
        print(f"get target init particles: {target_init_particles.shape}")
        
        
        return target_init_particles
        
        
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy"
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    def _preload_data(self, ):
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
            
        
        for data_nm in self.data_name_list:
            cur_data = self._load_data_from_data_name(data_nm)
            self.data_name_to_data[data_nm] = cur_data
            
            init_verts = cur_data['tot_verts'][0]
            particle_init_xs_th = torch.from_numpy(init_verts).float()
            sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
            self.data_name_to_fps_idxes[data_nm] = sampled_particle_idxes
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")
    
    
    # generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len=fixed_y_len, base_x_len=base_x_len)

    ## TODO: currently not in use ##
    def generate_test_links_general_flexy(self, dim, nn_links_one_side, len_one_side, fixed_y_len=0.1, base_x_len=0.1):
        per_link_len = len_one_side / float(nn_links_one_side)
        
        ## get nn_links_ ##
        dim = 2
        quality = 1  # Use a larger value for higher-res simulations ##
        n_particles, n_grid = 9000 * quality**2, 128 * quality
        
        n_particles = n_particles // 3 ## get nn_particels for each link ##
        
        particle_density = n_particles / (base_x_len * fixed_y_len)
        base_link_n_particles = particle_density * (base_x_len * fixed_y_len)
        child_link_n_particles = particle_density * (per_link_len * fixed_y_len)
        
        base_link_n_particles = int(base_link_n_particles)
        child_link_n_particles = int(child_link_n_particles)
        
        ## 
        rnd_xys = np.random.rand(base_link_n_particles, dim) * np.array([base_x_len, fixed_y_len], dtype=np.float32)[None, :]
        base_link_xys = np.array([0.45, 0.45], dtype=np.float32)[None, :] + rnd_xys ## get the random initialized xys ##
        base_link_particles_link_idx = np.zeros((base_link_n_particles, ), dtype=np.int32) # the bae link particles idxes 
        
        link_xys = [base_link_xys]
        link_particle_link_idx = [base_link_particles_link_idx]
        link_joint_pos = [np.array([0.0, 0.0], dtype=np.float32)]
        link_parent_idx = [-1]
        
        
        child_link_idx = 1
        joint_x = 0.45
        joint_y = 0.45
        for i_link in range(nn_links_one_side):
            cur_link_st_x = joint_x - per_link_len
            rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, fixed_y_len], dtype=np.float32)
            cur_link_xys = np.array([cur_link_st_x, joint_y], dtype=np.float32)[None, :] + rnd_xys
            cur_link_idxes = np.ones((child_link_n_particles, ), dtype=np.int32) * child_link_idx
            cur_link_joint_pos = np.array([joint_x, joint_y], dtype=np.float32)
            cur_link_parent_idx = child_link_idx - 1
            
            link_xys.append(cur_link_xys)
            link_particle_link_idx.append(cur_link_idxes)
            link_joint_pos.append(cur_link_joint_pos)
            link_parent_idx.append(cur_link_parent_idx)
            
            joint_x -= per_link_len
            child_link_idx += 1
        
        joint_x = 0.55
        joint_y = 0.45
        for i_link in range(nn_links_one_side):
            cur_link_st_x = joint_x # - per_link_len
            rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, fixed_y_len], dtype=np.float32)
            cur_link_xys = np.array([cur_link_st_x, joint_y], dtype=np.float32)[None, :] + rnd_xys
            cur_link_idxes = np.ones((child_link_n_particles, ), dtype=np.int32) * child_link_idx
            cur_link_joint_pos = np.array([joint_x, joint_y], dtype=np.float32)
            cur_link_parent_idx = child_link_idx - 1 if i_link > 0 else 0
            
            link_xys.append(cur_link_xys)
            link_particle_link_idx.append(cur_link_idxes)
            link_joint_pos.append(cur_link_joint_pos)
            link_parent_idx.append(cur_link_parent_idx)
            
            joint_x += per_link_len
            child_link_idx += 1
        
        obj_particles = np.concatenate(link_xys, axis=0)
        particle_link_idxes = np.concatenate(link_particle_link_idx, axis=0)
        link_joint_pos = np.stack(link_joint_pos, axis=0)
        link_parent_idx = np.array(link_parent_idx, dtype=np.int32)
        
        link_joint_dir = [[1.0, 0.0] for _ in range(nn_links_one_side * 2 + 1)]
        link_joint_dir = np.array(link_joint_dir, dtype=np.float32)
        
        ## link joint dir ##
        obj_info = {
            'particles_xs': obj_particles, ### obj particles ###
            'particle_link_idxes': particle_link_idxes, ### particle link idxes ###
            'link_joint_pos': link_joint_pos, ### link joint pos ###
            'link_joint_dir': link_joint_dir,
            'link_parent_idx': link_parent_idx
        }
        
        return obj_info
        
        # asset_root_folder = os.path.join(PROJ_ROOT_FOLDER, "assets")
        # os.makedirs(asset_root_folder, exist_ok=True)
        
        # obj_info_sv_fn = os.path.join(PROJ_ROOT_FOLDER, f"assets", f"obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_baseX_{base_x_len}_Y_{fixed_y_len}.npy")
        # np.save(obj_info_sv_fn, obj_info)
        # print(f"Object information saved to {obj_info_sv_fn}")
    
    ## TODO: currently not in use ##
    def get_manipulator_infos(self, nn_links, fixed_y_len=0.1, base_x_len=0.1):
        ### get the manipulator infos; st_len; 
        ### 
        # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8]
        # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
        
        # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
        # tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
        
        ## create a new manipualtror ##
        nn_links = 5  ### TODO: get nn_links from the parameters passed to the function ##
        nn_links_one_side = (nn_links - 1) // 2 ### get the number of links in one side ##
        ## nn links one side ##
        # /data/xueyi/softzoo/expv4_projected_task_0/n_links_5_tag_iinst_8_nlinks_5_childlinklen_0.26666666666666666_curri_v2__nreg_inherit_True_seed_67_contact_spring_d_0.5_damping_0.1_ # 
        # /data/xueyi/softzoo/expv4_projected_task_0/n_links_5_tag_iinst_8_nlinks_5_childlinklen_0.30000000000000004_curri_v2__nreg_inherit_True_seed_67_contact_spring_d_0.5_damping_0.1_
        # 
        # generate_test_links_general_flexy(self, dim, nn_links_one_side, len_one_side, fixed_y_len=0.1, base_x_len=0.1):
        
        dim = 2
        nn_links_one_side = nn_links_one_side
        len_one_side = (0.26666666666666666 + 0.30000000000000004) / 2.0
        # fixed_y_len = 0.1
        # base_x_len = 0.1 ## base x len 
        generated_test_link_dict = self.generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len, base_x_len) ## get the generated test link dict ##
        ### we have particles; and the partile link idxes ##
        ## sample for the trajectories from the data and the model for further usage ##
        
        ##### target init particles; target particles link idxes; target link joint pos; target link joint dir; target link parent idx #####
        self.target_init_particles = generated_test_link_dict['particles_xs']
        self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        
        
        
        ## get the manipulator info ##
        # tot_nn_links_one_side = []
        # tot_len_one_side = []
        # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages // 2 - 1)
        # st_nn_link_one_side = 1
        # for i in range(nn_stages):
        #     cur_link_len = st_len_one_side + (i // 2) * link_len_one_side_interval
            
        #     tot_len_one_side.append(cur_link_len)
        #     tot_nn_links_one_side.append(st_nn_link_one_side)
            
        #     if i % 2 == 0:
        #         st_nn_link_one_side = st_nn_link_one_side * 2
        
        # print("tot_nn_links_one_side: ", tot_nn_links_one_side)
        # print(f"tot_len_one_side: {tot_len_one_side}")
        # return tot_nn_links_one_side, tot_len_one_side
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data(self, data_dict): # bsz x nn_particles x feat_dim #
        data_X = data_dict['X']
        data_E = data_dict['E']
        # data_E = data_E[..., 0]
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        ## inv_scale data ##
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        
        avg_particle_accs_all = np.concatenate([self.avg_particle_accs_tau, self.avg_particle_accs, self.avg_particle_accs_final], axis=-1)
        std_particle_accs_all = np.concatenate([self.std_particle_accs_tau, self.std_particle_accs, self.std_particle_accs_final], axis=-1)
        
        
        
        
        if isinstance(data_X, torch.Tensor):
            data_E_inv_scaled = (data_E * (torch.from_numpy(std_particle_accs_all[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(avg_particle_accs_all[None][None]).float().to(data_X.device)
            data_X_inv_scaled = (data_X * (torch.from_numpy(self.std_particle_init_xs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_init_xs[None][None]).float().to(data_X.device)
        else:
            data_E_inv_scaled = (data_E * (std_particle_accs_all[None][None] + 1e-6)) + avg_particle_accs_all[None][None]
            data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None][None] + 1e-6)) + self.avg_particle_init_xs[None][None]
            
        
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6))
        
        # bsz x nn_particles x (3  x nn_ts x 3 )
        tot_accs_dim = data_E_inv_scaled.shape[-1]
        single_type_accs_dim = tot_accs_dim // 3
        particle_accs_tau = data_E_inv_scaled[..., :single_type_accs_dim]
        particle_accs = data_E_inv_scaled[..., single_type_accs_dim: 2 * single_type_accs_dim]
        particle_accs_final = data_E_inv_scaled[..., 2 * single_type_accs_dim:]
        
        single_ts_acc_dim = 3 #  particle_accs_tau.shape[-1] // 3
        nn_ts = particle_accs_tau.shape[-1] // single_ts_acc_dim
        particle_accs_tau = particle_accs_tau.contiguous().view(particle_accs_tau.shape[0], -1, nn_ts, single_ts_acc_dim)
        particle_accs = particle_accs.contiguous().view(particle_accs.shape[0], -1, nn_ts, single_ts_acc_dim)
        particle_accs_final = particle_accs_final.contiguous().view(particle_accs_final.shape[0], -1, nn_ts, single_ts_acc_dim)
        
        particle_accs_tau = particle_accs_tau.contiguous().transpose(2, 1).contiguous()
        particle_accs = particle_accs.contiguous().transpose(2, 1).contiguous()
        particle_accs_final = particle_accs_final.contiguous().transpose(2, 1).contiguous()
        
        # init_pos = data_X_inv_scaled.contiguous().view(data_X_inv_scaled.shape[0], -1, 3)
        
        return {
            'tot_verts': data_X_inv_scaled,
            'tot_verts_dd_tau': particle_accs_tau,
            'tot_verts_dd': particle_accs,
            'tot_verts_dd_final': particle_accs_final
        }
        
    
    def inv_scale_data_v2(self, data_dict): # bsz x nn_particles x feat_dim #
        data_X = data_dict['X']
        data_E = data_dict['E']
        # data_E = data_E[..., 0]
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        ## inv_scale data ##
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        
        avg_particle_accs_all = np.concatenate([self.avg_particle_accs_tau, self.avg_particle_accs, self.avg_particle_accs_final], axis=-1)
        std_particle_accs_all = np.concatenate([self.std_particle_accs_tau, self.std_particle_accs, self.std_particle_accs_final], axis=-1)
        
        
        
        ###### ======= scale the data ======= ######
        if isinstance(data_X, torch.Tensor):
            data_E_inv_scaled = (data_E * (torch.from_numpy(std_particle_accs_all[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(avg_particle_accs_all[None][None]).float().to(data_X.device)
            data_X_inv_scaled = (data_X * (torch.from_numpy(self.std_particle_init_xs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_init_xs[None][None]).float().to(data_X.device)
        else:
            data_E_inv_scaled = (data_E * (std_particle_accs_all[None][None] + 1e-6)) + avg_particle_accs_all[None][None]
            data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None][None] + 1e-6)) + self.avg_particle_init_xs[None][None]
        ###### ======= scale the data ======= ######
        
        
        ###### ======= n-scale the data ======= ######
        # data_E_inv_scaled = data_E
        # data_X_inv_scaled = data_X
        ###### ======= n-scale the data ======= ######
        
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6))
        
        # bsz x nn_particles x (3  x nn_ts x 3 )
        tot_accs_dim = data_E_inv_scaled.shape[-1]
        single_type_accs_dim = tot_accs_dim // 3
        particle_accs_tau = data_E_inv_scaled[..., :single_type_accs_dim]
        particle_accs = data_E_inv_scaled[..., single_type_accs_dim: 2 * single_type_accs_dim]
        particle_accs_final = data_E_inv_scaled[..., 2 * single_type_accs_dim:]
        
        single_ts_acc_dim = 3 #  particle_accs_tau.shape[-1] // 3
        nn_ts = particle_accs_tau.shape[-1] // single_ts_acc_dim
        particle_accs_tau = particle_accs_tau.contiguous().view(particle_accs_tau.shape[0], -1, nn_ts, single_ts_acc_dim)
        particle_accs = particle_accs.contiguous().view(particle_accs.shape[0], -1, nn_ts, single_ts_acc_dim)
        particle_accs_final = particle_accs_final.contiguous().view(particle_accs_final.shape[0], -1, nn_ts, single_ts_acc_dim)
        
        particle_accs_tau = particle_accs_tau.contiguous().transpose(2, 1).contiguous()
        particle_accs = particle_accs.contiguous().transpose(2, 1).contiguous()
        particle_accs_final = particle_accs_final.contiguous().transpose(2, 1).contiguous()
        
        # init_pos = data_X_inv_scaled.contiguous().view(data_X_inv_scaled.shape[0], -1, 3)
        
        return {
            'tot_verts': data_X_inv_scaled,
            'tot_verts_dd_tau': particle_accs_tau,
            'tot_verts_dd': particle_accs,
            'tot_verts_dd_final': particle_accs_final
        }
        
     

    ## TODO: currently not in use ##
    def transform_pcd_wact_dict(self, data_dict):
        init_xs = data_dict['X']
        ### TODO: get dt from configs? ###
        pts_acts = data_dict["E"]
        nn_bszs = init_xs.shape[0]

        # dt = 1e-1
        dt = self.dt
        
        tot_pts_vels = []
        tot_pts_vecs = []

        for i_bsz in range(nn_bszs):
            cur_init_xs = init_xs[i_bsz]
            cur_pts_acts = pts_acts[i_bsz]

            nn_pts_dims = cur_init_xs.shape[-1]
            cur_pts_acts = cur_pts_acts.reshape(cur_pts_acts.shape[0], -1, nn_pts_dims)

            #### ==== get pts act shapes ==== ####
            print(f"[Transform PCDs with ACTs] cur_pts_acts: {cur_pts_acts.shape}")
            cur_pts_acts = np.transpose(cur_pts_acts, (1, 0, 2))
            print(f"[Transform PCDs with ACTs] cur_pts_acts: {cur_pts_acts.shape}")
            ## 
            cur_pts_vels = []
            cur_pts_vecs = []
            for i_fr in range(cur_pts_acts.shape[0]):
                if i_fr == 0:
                    cur_fr_pts_vels = np.zeros_like(cur_pts_acts[i_fr])
                    cur_fr_pts_vecs = np.zeros_like(cur_pts_acts[i_fr])
                else:
                    prev_fr_pts_vels = cur_pts_vels[-1]
                    prev_fr_pts_vecs = cur_pts_vecs[-1]
                    cur_fr_pts_accs = cur_pts_acts[i_fr]

                    cur_fr_pts_vels = prev_fr_pts_vels + dt * cur_fr_pts_accs
                    cur_fr_pts_vecs = prev_fr_pts_vecs + dt * prev_fr_pts_vels + dt * dt * cur_fr_pts_accs
                    
                cur_pts_vels.append(cur_fr_pts_vels)
                cur_pts_vecs.append(cur_fr_pts_vecs)
            cur_pts_vels = np.stack(cur_pts_vels, axis=0)
            cur_pts_vecs = np.stack(cur_pts_vecs, axis=0)
            
            tot_pts_vels.append(cur_pts_vels)
            tot_pts_vecs.append(cur_pts_vecs)
        
        tot_pts_vels = np.stack(tot_pts_vels, axis=0)
        tot_pts_vecs = np.stack(tot_pts_vecs, axis=0)
        
        rt_dict = {
            'X': init_xs,
            'E': pts_acts,
            'pts_vels': tot_pts_vels,
            'pts_vecs': tot_pts_vecs ## get total pts vels and the vecs ##
        }
        return rt_dict

        
    
    def scale_data(self, data_dict, data_nm):
        ## nn_ts x nn_particles x 3 ##
        particle_init_xs = data_dict['tot_verts'][0]
        particle_accs_tau = data_dict['tot_verts_dd_tau']
        particle_accs = data_dict['tot_verts_dd']
        particle_accs_final = data_dict['tot_verts_dd_final']
        
        # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
        sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        # if self.sampled_particle_idxes is None:
        #     particle_init_xs_th = torch.from_numpy(particle_init_xs).float()
        #     sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
        #     self.sampled_particle_idxes = sampled_particle_idxes # sampled_particle_idxes #
        # else: #
        #     sampled_particle_idxes = self.sampled_particle_idxes #
        
        sv_sampled_init_xs_fn = f"assets/sampled_init_xs.npy"
        if not os.path.exists(sv_sampled_init_xs_fn): # sampled init xs #
            sv_sampled_init_xs = {
                'particle_init_xs': particle_init_xs[sampled_particle_idxes],
                'sampled_particle_idxes': sampled_particle_idxes
            }
            np.save(sv_sampled_init_xs_fn, sv_sampled_init_xs)
            
        # 
        particle_init_xs = particle_init_xs[sampled_particle_idxes]
        particle_accs_tau = particle_accs_tau[:, sampled_particle_idxes]
        particle_accs = particle_accs[:, sampled_particle_idxes]
        particle_accs_final = particle_accs_final[:, sampled_particle_idxes] # 
        
        
        def scale_via_min_max(particle_data):
            minn_data = np.min(particle_data, axis=0, keepdims=True)
            maxx_data = np.max(particle_data, axis=0, keepdims=True)
            extent_data = maxx_data - minn_data
            particle_data = (particle_data - minn_data) / (extent_data + 1e-6)
            particle_data = particle_data * 1.0 - 0.5
            return particle_data
        
        
        
        # minn_particle_xs = np.min(particle_init_xs, axis=0, keepdims=True) #
        # maxx_particle_xs = np.max(particle_init_xs, axis=0, keepdims=True) #
        # extent_particle_xs = maxx_particle_xs - minn_particle_xs #
        # particle_init_xs = (particle_init_xs - minn_particle_xs) / (extent_particle_xs + 1e-6) #
        
        
        ######## ======== Uniform particle init_xs, particle_accs, and accs_final scaling ======== ########
        # particle_init_xs = scale_via_min_max(particle_init_xs)
        # particle_accs_tau = np.transpose(particle_accs_tau, (1, 0, 2)) 
        # particle_accs_tau = particle_accs_tau.reshape(particle_accs_tau.shape[0], -1) ## nn_particles x (nn_ts * 3)
        # particle_accs_tau = scale_via_min_max(particle_accs_tau)
        
        # particle_accs = np.transpose(particle_accs, (1, 0, 2))
        # particle_accs = particle_accs.reshape(particle_accs.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        # particle_accs = scale_via_min_max(particle_accs)
        
        
        # particle_accs_final = np.transpose(particle_accs_final, (1, 0, 2))
        # particle_accs_final = particle_accs_final.reshape(particle_accs_final.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        # particle_accs_final = scale_via_min_max(particle_accs_final)
        ######## ======== Uniform particle init_xs, particle_accs, and accs_final scaling ======== ########
        
        
        
        # init_pos = particle_init_xs[0] #
        particle_init_xs = (particle_init_xs - self.avg_particle_init_xs[None]) / (self.std_particle_init_xs[None] + 1e-6)
        particle_accs_tau = np.transpose(particle_accs_tau, (1, 0, 2)) # nn_particles x nn_ts x 3
        particle_accs_tau = particle_accs_tau.reshape(particle_accs_tau.shape[0], -1) ## nn_particles x (nn_ts * 3)
        particle_accs_tau = (particle_accs_tau - self.avg_particle_accs_tau[None]) / (self.std_particle_accs_tau[None] + 1e-6)
        particle_accs = np.transpose(particle_accs, (1, 0, 2))
        particle_accs = particle_accs.reshape(particle_accs.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        particle_accs = (particle_accs - self.avg_particle_accs[None]) / (self.std_particle_accs[None] + 1e-6)
        particle_accs_final = np.transpose(particle_accs_final, (1, 0, 2))
        particle_accs_final = particle_accs_final.reshape(particle_accs_final.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        particle_accs_final = (particle_accs_final - self.avg_particle_accs_final[None]) / (self.std_particle_accs_final[None] + 1e-6)
        ## particle accs ##
        
        
        sv_scaled_sampled_init_xs_fn = f"assets/scaled_sampled_init_xs.npy"
        if not os.path.exists(sv_scaled_sampled_init_xs_fn):
            sv_scaled_sampled_init_xs = {
                'particle_init_xs': particle_init_xs,
            }
            np.save(sv_scaled_sampled_init_xs_fn, sv_scaled_sampled_init_xs)
        
        
        
        particle_acts = np.concatenate([particle_accs_tau, particle_accs, particle_accs_final], axis=-1)
        
        # particle_xs = data_dict['particle_xs']
        # particle_acts = data_dict['particle_accs']

        # ## TODO: for this setting, random permuting particle xs for sampling points is reasonable; but is not a good strategy for non-uniform pcs 
        # sampled_particle_idxes = np.random.permutation(particle_xs.shape[1])[: self.maxx_nn_pts] ## jet the sampled pts indexes 
        # particle_xs = particle_xs[:, sampled_particle_idxes]
        # particle_acts = particle_acts[:, sampled_particle_idxes]
        
        
        return {
            'X': particle_init_xs,
            'E': particle_acts,
        }
        
    def scale_data_batched(self, data_dict):
        init_particle_xs = data_dict['X']
        particle_acts = data_dict['E']
        ## bsz x nn_particles x 3 
        print(f"[Batched data scaling] init_particle_xs: {init_particle_xs.size()}, particle_acts: {particle_acts.size()}")
        th_avg_particle_init_xs = torch.from_numpy(self.avg_particle_init_xs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_std_particle_init_xs = torch.from_numpy(self.std_particle_init_xs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        th_avg_particle_accs_tau = torch.from_numpy(self.avg_particle_accs_tau).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_std_particle_accs_tau = torch.from_numpy(self.std_particle_accs_tau).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_avg_particle_accs = torch.from_numpy(self.avg_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_std_particle_accs = torch.from_numpy(self.std_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_avg_particle_accs_final = torch.from_numpy(self.avg_particle_accs_final).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        th_std_particle_accs_final = torch.from_numpy(self.std_particle_accs_final).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        init_particle_xs = (init_particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + 1e-6)
        # particle_accs = particle_acts
        th_avg_accs_all = torch.cat([th_avg_particle_accs_tau, th_avg_particle_accs, th_avg_particle_accs_final], dim=-1)
        th_std_accs_all = torch.cat([th_std_particle_accs_tau, th_std_particle_accs, th_std_particle_accs_final], dim=-1)
        particle_acts = (particle_acts - th_avg_accs_all) / (th_std_accs_all + 1e-6)
        
        # th_avg_particle_accs = torch.from_numpy(self.avg_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_std_particle_accs = torch.from_numpy(self.std_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        # init_particle_xs = (init_particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + 1e-6)
        # # particle_acts = particle_acts.transpose(1, 0).contiguous().view(particle_acts.size(1), -1)
        # particle_acts = (particle_acts - th_avg_particle_accs) / (th_std_particle_accs + 1e-6)
        
        scaled_data = {
            'X': init_particle_xs, 
            'E': particle_acts
        }
        return scaled_data
        
    ## TODO: currently not in use ##
    def scale_data_wconstraints(self, data_dict):
        
        particle_xs = data_dict['particle_xs']
        particle_acts = data_dict['particle_accs'] # T x act_dim #
        
        # self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        # self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        # self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        # self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        particle_link_idxes = data_dict['particle_link_idxes'] ## nn_original_particles 
        link_joint_pos = data_dict['link_joint_pos']
        link_joint_dir = data_dict['link_joint_dir']
        link_parent_idx = data_dict['link_parent_idx']
        
        
        
        ## TODO: for this setting, random permuting particle xs for sampling points is reasonable; but is not a good strategy for non-uniform pcs 
        if self.sampled_particle_idxes is None:
            sampled_particle_idxes = np.random.permutation(particle_xs.shape[1])[: self.maxx_nn_pts] ## jet the sampled pts indexes 
            self.sampled_particle_idxes = sampled_particle_idxes
        else:
            sampled_particle_idxes = self.sampled_particle_idxes ## get a unified sampled particle idxes ## 
        
        
        particle_xs = particle_xs[:, sampled_particle_idxes]
        particle_acts = particle_acts[:, sampled_particle_idxes]
        
        particle_link_idxes = particle_link_idxes[sampled_particle_idxes]
        ## inv scale -> project -> scale -> resample ##
        
         
        init_pos = particle_xs[0]


        particle_xs = (init_pos - self.avg_particle_init_xs[None]) / (self.std_particle_init_xs[None] + 1e-6)
        particle_acts = np.transpose(particle_acts, (1, 0, 2))
        particle_acts = particle_acts.reshape(particle_acts.shape[0], -1) ## nn_particles x nn_act_feat_dim ##
        particle_acts = (particle_acts - self.avg_particle_accs[None]) / (self.std_particle_accs[None] + 1e-6)
        
        
        return {
            'X': particle_xs,
            'E': particle_acts,
            'particle_link_idxes': particle_link_idxes,
            'link_joint_pos': link_joint_pos,
            'link_joint_dir': link_joint_dir,
            'link_parent_idx': link_parent_idx
        }
     
      
    def data_dict_to_th(self, data_dict_np):
        data_dict_th = {
            # key: torch.from_numpy(data_dict_np[key]).float().cuda() for key in data_dict_np
            key: torch.from_numpy(data_dict_np[key]).float() for key in data_dict_np
            ## ? ## TODO: add the self.device in the init according to cfgs ###
        }
        return data_dict_th
    
    ## TODO: currently not in use ##
    def get_target_data(self,):
        nn_timesteps = 10
        nn_act_dim = 2
        target_init_particles = self.target_init_particles ## constraint projections with the original cosntriants ## 
        ## fit for a set of rotations and the translations ---> to satisfy the segmentation constraints ##
        ## project the tranaltions to satisfy joint constraints ## --> finally we the particle sequences ##
        ## project 
        ### nn_timesstpes x nn_particles x nn_act_flatten_dim ###
        target_particle_acts =  np.random.randn(nn_timesteps, target_init_particles.shape[0], nn_act_dim)
        
        target_init_particles = target_init_particles[None, :, :]
        
        unscaled_data_dict = {
            'particle_xs': target_init_particles,
            'particle_accs': target_particle_acts,
            'particle_link_idxes': self.target_particle_link_idxes,
            'link_joint_pos': self.target_link_joint_pos,
            'link_joint_dir': self.target_link_joint_dir,
            'link_parent_idx': self.target_link_parent_idx
        }
        
        # self.target_init_particles = generated_test_link_dict['particles_xs']
        # self.target_particle_link_idxes = generated_test_link_dict['particle_link_idxes']
        # self.target_link_joint_pos = generated_test_link_dict['link_joint_pos']
        # self.target_link_joint_dir = generated_test_link_dict['link_joint_dir'] ## get link joint dirs ##
        # self.target_link_parent_idx = generated_test_link_dict['link_parent_idx']
        if self.sample_wconstraints:
            scaled_data = self.scale_data_wconstraints(unscaled_data_dict)
        else:
            scaled_data = self.scale_data(unscaled_data_dict)
        
        # scaled_data = self.scale_data(unscaled_data_dict) ## get the scaled dta and unscaled data dict ##
        
        return scaled_data
    
    # def get_data_via_index(self, index) -->getitem ##
    def __getitem__(self, index):
        # print(f"data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
            # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ###
        
        ## TODO: data selecting, data parsing, and data scaling ##
        
        # if self.use_target_data:
        #     cur_data_scaled = self.get_target_data()
        # else
        # :
        
        cur_data_scaled = self.scale_data(cur_data, cur_data_nm) ## scale the data
        
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = {
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        return cur_data_scaled_th





class Uni_Manip_3D_PC_V3_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst # get the all clips inst #
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages ## get the nn)stages from the dataset config ##
        
        # self.sampled_particle_idxes = None
        
        ## then you should get the task nn timesteps ##
        ## get the task nn timesteps ##
        
        # TODO: implement thsi logic where the target data is the target 3D dex manipulator #
        # if self.use_target_data:
        #     # TODO: get nn_links from the config file ##
        #     nn_links = 5
        #     print(f"Start getting target data with {nn_links} links")
        #     fixed_y_len = 0.05
        #     base_x_len = 0.1
        #     self.get_manipulator_infos(nn_links, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
        #     # self.get_manipulator_infos(nn_links, fixed_y_len=0.1, base_x_len=0.1)
        
        ## get manipulator infos ##
        # data_task_err_thres = 0.03
        # data_trans_constraints_thres = 0.01
        
        
        
        # TODO: add the valid data filters #
        
        
        self.data_inst_fn = "saved_info_accs_v3.npy"
        exp_tags = ["tds_exp_2"]

        ## root_data_folder ##
        self.data_list = []
        
        
        if self.all_clips_inst: # get the all clips instances # #
            self.data_inst_fn = f"saved_info_accs_v4_nstages_{self.nn_stages}.npy"
            print(f"data_inst_fn changed to {self.data_inst_fn} with all_clips_inst: {self.all_clips_inst}")
            for exp_tag in exp_tags:
                cur_data_folder = os.path.join(self.data_folder, exp_tag)
                tmp_data_list = os.listdir(cur_data_folder)
                # /data/xueyi/uni_manip/tds_exp_2/allegro_bottle_5_taskstage2_objm0.39_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.2_0.37_objgoalrot_2.8274333477020264_0.0_0.0_ ## single inst model 
                ### single inst model ###
                
                # if self.single_inst:
                #     tmp_data_list = ["allegro_bottle_5_taskstage2_objm0.39_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.2_0.37_objgoalrot_2.8274333477020264_0.0_0.0_"]
                # # if only using the data with positive rotation angles
                
                # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
                # tmp_data_list = ["allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_"]
                for cur_subfolder in tmp_data_list: # 
                    
                    cur_subfolder_angle = cur_subfolder.split("_")
                    ####### ====== for positiive angles ====== #######
                    # for i_split in range(len(cur_subfolder_angle)):
                    #     if cur_subfolder_angle[i_split] == "objgoalrot":
                    #         cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                    #         break
                    # if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                    #     continue
                    ####### ====== for positiive angles ====== #######
                    inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                    if os.path.isdir(inst_folder):
                        cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                        if os.path.exists(cur_inst_file):
                            self.data_list.append(cur_inst_file)
        else:
            
            for exp_tag in exp_tags:
                cur_data_folder = os.path.join(self.data_folder, exp_tag)
                tmp_data_list = os.listdir(cur_data_folder)
                # /data/xueyi/uni_manip/tds_exp_2/allegro_bottle_5_taskstage2_objm0.39_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.2_0.37_objgoalrot_2.8274333477020264_0.0_0.0_ ## single inst model 
                ### single inst model ###
                
                if self.single_inst:
                    tmp_data_list = ["allegro_bottle_5_taskstage2_objm0.39_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.2_0.37_objgoalrot_2.8274333477020264_0.0_0.0_"]
                # if only using the data with positive rotation angles
                
                # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
                # tmp_data_list = ["allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_"]
                for cur_subfolder in tmp_data_list:
                    
                    cur_subfolder_angle = cur_subfolder.split("_")
                    ####### ====== for positiive angles ====== #######
                    for i_split in range(len(cur_subfolder_angle)):
                        if cur_subfolder_angle[i_split] == "objgoalrot":
                            cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                            break
                    if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                        continue
                    ####### ====== for positiive angles ====== #######
                    inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                    if os.path.isdir(inst_folder):
                        cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                        if os.path.exists(cur_inst_file):
                            self.data_list.append(cur_inst_file)
            
        
        ### solve the task better ### # 
        ### currently we do not have the ### currently we do not have the xxx
        ### TODO: currently we do not have the threshold for task completion #
        ### TODO: valid data statistics #
        ### TODO: the currently we do not have threshold for the task completion
        if self.all_clips_inst: # 
            valid_data_list_sv_fn = f"valid_data_statistics_v4.npy" 
        elif self.single_inst: 
            valid_data_list_sv_fn = f"valid_data_statistics_v3.npy" 
        else:
            # valid_data_list_sv_fn = f"valid_data_statistics_v3_all.npy" ## solve the task better? ##
            valid_data_list_sv_fn = f"valid_data_statistics_v3_positive_angles.npy"
        # valid_data_list_sv_fn = f"valid_data_statistics_v3.npy"
        valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
        
        # statistics = { # statistics and the statistics #
        #     'avg_verts_tot_cases_tot_ts': avg_verts_tot_cases_tot_ts,
        #     'std_verts_tot_cases_tot_ts': std_verts_tot_cases_tot_ts,
        #     'avg_verts_qdd_tau_tot_cases_tot_ts': avg_verts_qdd_tau_tot_cases_tot_ts,
        #     'std_verts_qdd_tau_tot_cases_tot_ts': std_verts_qdd_tau_tot_cases_tot_ts,
        #     # 'pts_accs_avg': avg_tot_pts_accs,
        #     # 'pts_accs_std': std_tot_pts_accs,
        #     # 'pts_accs_final_avg': avg_tot_pts_accs_final,
        #     # 'pts_accs_final_std': std_tot_pts_accs_final
        # } # statistics #
        
        ### TODO: calculate such statistics ##
        valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        
        self.avg_verts_tot_cases_tot_ts = valid_data_statistics['avg_verts_tot_cases_tot_ts']
        self.std_verts_tot_cases_tot_ts = valid_data_statistics['std_verts_tot_cases_tot_ts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['avg_verts_qdd_tau_tot_cases_tot_ts']
        self.std_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['std_verts_qdd_tau_tot_cases_tot_ts']
        
        # self.avg_particle # avg particle #
        # self.avg_particle_init_xs = valid_data_statistics['robot_init_visual_pts_avg']
        # self.std_particle_init_xs = valid_data_statistics['robot_init_visual_pts_std']
        # self.avg_particle_accs_tau = valid_data_statistics['pts_accs_tau_avg']
        # self.std_particle_accs_tau = valid_data_statistics['pts_accs_tau_std']
        # self.avg_particle_accs = valid_data_statistics['pts_accs_avg']
        # self.std_particle_accs = valid_data_statistics['pts_accs_std']
        # self.avg_particle_accs_final = valid_data_statistics['pts_accs_final_avg']
        # self.std_particle_accs_final = valid_data_statistics['pts_accs_final_std']
        
        
        
        # valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
        # valid_data_list_sv_fn = os.path.join(self.data_folder, valid_data_list_sv_fn)
        
        
        
        # valid_data_statistics_sv_fn = f"valid_data_statistics_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.npy"
        # valid_data_statistics_sv_fn = os.path.join(self.data_folder, "expv4", valid_data_statistics_sv_fn)
        # valid_data_statistics = np.load(valid_data_statistics_sv_fn, allow_pickle=True).item()


        # self.avg_particle_init_xs = valid_data_statistics['avg_particle_init_xs']
        # self.std_particle_init_xs = valid_data_statistics['std_particle_init_xs']
        
        # self.avg_particle_accs = valid_data_statistics['avg_particle_accs']
        # self.std_particle_accs = valid_data_statistics['std_particle_accs'] ##
        
        # print(f"avg_particle_init_xs: {self.avg_particle_init_xs}, std_particle_init_xs: {self.std_particle_init_xs}")
        
        # tmp_data_list = self.data_list ### get the valid data list ##
        
        # tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(self.data_folder, fn))]
        
        # self.ckpt_nm = "ckpt_best_diff.npy"
        
        # self.data_list = []
        # for fn in tmp_data_list:
        #     cur_data_ckpt_folder = os.path.join(self.data_folder, fn, "checkpoints")
        #     if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
        #         best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, self.ckpt_nm)
        #         if os.path.exists(best_ckpt_data_fn): # exist in the data folder #
        #             self.data_list.append(fn)
        # ### TODO: mvoe this parameter to the config file ###
        # self.fixed_nn_nodes = 21
        
        
        
        self.maxx_nn_pts = 512
        
        # self.max_link_rot_acc = 10.0
        # self.min_link_rot_acc = -29.0
        # self.extent_link_rot_acc = self.max_link_rot_acc - self.min_link_rot_acc
        
        # self.max_link_trans_acc = 61.0
        # self.min_link_trans_acc = -30.0
        # self.extent_link_trans_acc = self.max_link_trans_acc - self.min_link_trans_acc
        
        # with various constraint modeling # 
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        
        self._preload_data()
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        # data_name_to_data, data_name_to_fps_idxes #
        return len(self.data_name_to_data)
        # return len(self.data_name_list)
    
    
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    def _preload_data(self, ):
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        
        single_clip_length = 300
        sliding_window_length = 100
        
        for data_nm in self.data_name_list:
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            # tot_verts tot_verts_integrated_qdd_tau
            tot_verts = cur_data['tot_verts'] #
            tot_verts_integrated_qdd_tau = cur_data['tot_verts_integrated_qdd_tau'] #
            # nn_ts x nn_verts x 3 #
            
            ###### ===== get the verts integrated with qd ===== ######
            # tot_verts_integrated_qd = cur_data['tot_verts_integrated_qd'] #
            ###### ===== get the verts integrated with qd ===== ######
        
            # cur_verts_expanded, cur_verts_integrated_qdd_tau #
            tot_verts_expanded = []
            tot_verts_integrated_qdd_tau_expanded = []
            
        
            if self.all_clips_inst: 
                for i_starting_ts in range(0, tot_verts.shape[1] - single_clip_length, sliding_window_length):
                    cur_ending_ts = i_starting_ts + single_clip_length
                    cur_tot_verts = tot_verts[:, i_starting_ts:cur_ending_ts]
                    
                    ###### ===== get the verts integrated with qd ===== ######
                    # cur_tot_verts_integrated_qd = tot_verts_integrated_qd[:, i_starting_ts: cur_ending_ts]
                    ###### ===== get the verts integrated with qd ===== ######
                    
                    cur_tot_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau[:, i_starting_ts: cur_ending_ts] 
                    
                    
                    first_fr_cur_tot_verts = cur_tot_verts[:, 0]
                    ### TODO: another cnetralization strategy  ? ##
                    first_fr_verts_offset = first_fr_cur_tot_verts[0] ## (3,) - shape of the offset tensor ##
                    
                    cur_tot_verts = cur_tot_verts - first_fr_verts_offset[None][None] ## the cur_tot_verts - (1,1,3) # first frame vertices offset #
                    
                    ###### ===== get the verts integrated with qd ===== ######
                    # cur_tot_verts_integrated_qd = cur_tot_verts_integrated_qd - first_fr_verts_offset[None][None] 
                    ###### ===== get the verts integrated with qd ===== ######
                    
                    cur_tot_verts_integrated_qdd_tau = cur_tot_verts_integrated_qdd_tau - first_fr_verts_offset[None][None] 
                    
                    
                    
                    
                    
                    cur_data_nm = f"data_nm_ist_{i_starting_ts}_ied_{cur_ending_ts}"
                    cur_clip_data = {
                        'tot_verts': cur_tot_verts, 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': cur_tot_verts_integrated_qdd_tau
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                    
                    init_verts = cur_tot_verts[:, 0]
                    particle_init_xs_th = torch.from_numpy(init_verts).float()
                    
                    
                    sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                    if not os.path.exists(sampled_particle_idxes_sv_fn):
                        sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                        
                        np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)  # saved to the particles idxes #
                    else:
                        sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True) # 
                        
                
                    self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
                    
                    
                    # cur_verts_expanded, cur_verts_integrated_qdd_tau #
                    # cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
                    # cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                        
                        
                    tot_verts_expanded.append(cur_tot_verts)
                    tot_verts_integrated_qdd_tau_expanded.append(tot_verts_integrated_qdd_tau) ##
                    
                    # if self.single_inst:
                    #     cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
                    #     cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                        
                        
                    #     avg_cur_verts_expanded = np.mean(cur_verts_expanded, axis=0)
                    #     std_cur_verts_expanded = np.std(cur_verts_expanded, axis=0)
                    #     avg_cur_verts_integrated_qdd_tau = np.mean(cur_verts_integrated_qdd_tau, axis=0)
                    #     std_cur_verts_integrated_qdd_tau = np.std(cur_verts_integrated_qdd_tau, axis=0)
                        
                    #     self.avg_verts_tot_cases_tot_ts = avg_cur_verts_expanded
                    #     self.std_verts_tot_cases_tot_ts = std_cur_verts_expanded
                        
                    #     self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_cur_verts_integrated_qdd_tau
                    #     self.std_verts_qdd_tau_tot_cases_tot_ts = std_cur_verts_integrated_qdd_tau
                    #     break
                    
                #### single inst scaling v2 ####
                if self.single_inst:
                    
                    tot_verts_expanded = np.concatenate(tot_verts_expanded, axis=0)
                    tot_verts_integrated_qdd_tau_expanded = np.concatenate(tot_verts_integrated_qdd_tau_expanded, axis=0)
                    
                    tot_verts_expanded = tot_verts_expanded.reshape(-1, 3)
                    tot_verts_integrated_qdd_tau_expanded = tot_verts_integrated_qdd_tau_expanded.reshape(-1, 3)
                    
                    avg_verts_expanded = np.mean(tot_verts_expanded, axis=0)
                    std_verts_expanded = np.std(tot_verts_expanded, axis=0)
                    
                    avg_verts_integrated_qdd_tau = np.mean(tot_verts_integrated_qdd_tau_expanded, axis=0)
                    std_verts_integrated_qdd_tau = np.std(tot_verts_integrated_qdd_tau_expanded, axis=0)
                    
                    self.avg_verts_tot_cases_tot_ts = avg_verts_expanded
                    self.std_verts_tot_cases_tot_ts = std_verts_expanded
                    self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_verts_integrated_qdd_tau
                    self.std_verts_qdd_tau_tot_cases_tot_ts = std_verts_integrated_qdd_tau
                    
                    # cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
                    # cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                    
                    # avg_cur_verts_expanded = np.mean(cur_verts_expanded, axis=0)
                    # std_cur_verts_expanded = np.std(cur_verts_expanded, axis=0) # std cur #
                    # avg_cur_verts_integrated_qdd_tau = np.mean(cur_verts_integrated_qdd_tau, axis=0)
                    # std_cur_verts_integrated_qdd_tau = np.std(cur_verts_integrated_qdd_tau, axis=0)
                    
                    # self.avg_verts_tot_cases_tot_ts = avg_cur_verts_expanded
                    # self.std_verts_tot_cases_tot_ts = std_cur_verts_expanded
                    
                    # self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_cur_verts_integrated_qdd_tau
                    # self.std_verts_qdd_tau_tot_cases_tot_ts = std_cur_verts_integrated_qdd_tau
                    break
                    
                    
            else:
                self.data_name_to_data[data_nm] = cur_data
                
                init_verts = cur_data['tot_verts'][:, 0]
                particle_init_xs_th = torch.from_numpy(init_verts).float()
                
                # 
                sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                
                if not os.path.exists(sampled_particle_idxes_sv_fn):
                    sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    
                    np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)  # saved to the particles idxes #
                else:
                    sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True) # 
                
                # sampled particle idxes #
                
                # sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                
                
                self.data_name_to_fps_idxes[data_nm] = sampled_particle_idxes
        # self.data name to fps idxes #
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")

        self.data_name_list = list(self.data_name_to_data.keys())
    
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_v2(self, data_dict): # bsz x nn_particles x feat_dim #
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        

        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        
        
        inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
        inv_scaled_particle_xs_integrated_taus = (data_E * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
        
        
        
        # avg_particle_accs_all = np.concatenate([self.avg_particle_accs_tau, self.avg_particle_accs, self.avg_particle_accs_final], axis=-1)
        # std_particle_accs_all = np.concatenate([self.std_particle_accs_tau, self.std_particle_accs, self.std_particle_accs_final], axis=-1)
        
    
        
        
        
        # ###### ======= scale the data ======= ######
        # if isinstance(data_X, torch.Tensor):
        #     data_E_inv_scaled = (data_E * (torch.from_numpy(std_particle_accs_all[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(avg_particle_accs_all[None][None]).float().to(data_X.device)
        #     data_X_inv_scaled = (data_X * (torch.from_numpy(self.std_particle_init_xs[None][None]).float().to(data_X.device) + 1e-6)) + torch.from_numpy(self.avg_particle_init_xs[None][None]).float().to(data_X.device)
        # else:
        #     data_E_inv_scaled = (data_E * (std_particle_accs_all[None][None] + 1e-6)) + avg_particle_accs_all[None][None]
        #     data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None][None] + 1e-6)) + self.avg_particle_init_xs[None][None]
        # ###### ======= scale the data ======= ######
        
        
        ###### ======= n-scale the data ======= ######
        # data_E_inv_scaled = data_E
        # data_X_inv_scaled = data_X
        ###### ======= n-scale the data ======= ######
        
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6))
        
        # # bsz x nn_particles x (3  x nn_ts x 3 )
        # tot_accs_dim = data_E_inv_scaled.shape[-1]
        # single_type_accs_dim = tot_accs_dim // 3
        # particle_accs_tau = data_E_inv_scaled[..., :single_type_accs_dim]
        # particle_accs = data_E_inv_scaled[..., single_type_accs_dim: 2 * single_type_accs_dim]
        # particle_accs_final = data_E_inv_scaled[..., 2 * single_type_accs_dim:]
        
        # single_ts_acc_dim = 3 #  particle_accs_tau.shape[-1] // 3
        # nn_ts = particle_accs_tau.shape[-1] // single_ts_acc_dim
        # particle_accs_tau = particle_accs_tau.contiguous().view(particle_accs_tau.shape[0], -1, nn_ts, single_ts_acc_dim)
        # particle_accs = particle_accs.contiguous().view(particle_accs.shape[0], -1, nn_ts, single_ts_acc_dim)
        # particle_accs_final = particle_accs_final.contiguous().view(particle_accs_final.shape[0], -1, nn_ts, single_ts_acc_dim)
        
        # particle_accs_tau = particle_accs_tau.contiguous().transpose(2, 1).contiguous()
        # particle_accs = particle_accs.contiguous().transpose(2, 1).contiguous()
        # particle_accs_final = particle_accs_final.contiguous().transpose(2, 1).contiguous()
        
        # init_pos = data_X_inv_scaled.contiguous().view(data_X_inv_scaled.shape[0], -1, 3)
        # /data/xueyi/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v2_v1_pts_512_/samples000000200.npy...
        
        rt_dict = {
            'tot_verts': inv_scaled_particle_xs,
            'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
        }
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
        
        return rt_dict
        # {
        #     'tot_verts': inv_scaled_particle_xs,
        #     'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
        #     # 'tot_verts_dd': particle_accs,
        #     # 'tot_verts_dd_final': particle_accs_final
        # }
    
    
    def scale_data(self, data_dict, data_nm):
        
        # avg_verts_tot_cases_tot_ts, std_verts_tot_cases_tot_ts, avg_verts_qdd_tau_tot_cases_tot_ts, std_verts_qdd_tau_tot_cases_tot_ts
        
        ## nn_ts x nn_particles x 3 ##
        
        eps = 1e-6
        particle_xs = data_dict['tot_verts'] # nnp x nnts x 3
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        
        # nn_ts, nn_particles = particle_xs.shape[0], particle_xs.shape[1]
        # avg_verts_tot_cases_tot_ts = self.avg_verts_tot_cases_tot_ts
        # 
        
        # # nn_ts x nn_particles x 3 #
        # flatten_particle_xs = particle_xs.reshape(nn_ts * nn_particles, -1)
        # flatten_particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau.reshape(nn_ts * nn_particles, -1) ## get the particle qdd taus # 
        
        
        
        particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
        particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
        # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
        sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        # particle xs # 
        particle_xs = particle_xs[sampled_particle_idxes, :, :]
        particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
        # particlexs
        
        
        
        return {
            'X': particle_xs, # nnp x nnts x 3 #
            'E': particle_xs_integrated_qdd_tau,
            'sampled_idxes': sampled_particle_idxes,
        }
        
    def scale_data_batched(self, data_dict):
        particle_xs = data_dict['X']
        particle_acts = data_dict['E']
        
        eps = 1e-6
        
        # bsz x nn_ts x nn_particles x 3 #
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        

        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        particle_xs = (particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + eps)
        particle_acts = (particle_acts - th_avg_particle_xs_integrated_taus) / (th_std_particle_xs_integrated_taus + eps)
        
        # inv_scaled_particle_xs = (particle_xs * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
        # inv_scaled_particle_xs_integrated_taus = (particle_acts * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
        
        
        # ## bsz x nn_particles x 3 
        # print(f"[Batched data scaling] init_particle_xs: {init_particle_xs.size()}, particle_acts: {particle_acts.size()}")
        # th_avg_particle_init_xs = torch.from_numpy(self.avg_particle_init_xs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_std_particle_init_xs = torch.from_numpy(self.std_particle_init_xs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        # th_avg_particle_accs_tau = torch.from_numpy(self.avg_particle_accs_tau).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_std_particle_accs_tau = torch.from_numpy(self.std_particle_accs_tau).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_avg_particle_accs = torch.from_numpy(self.avg_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_std_particle_accs = torch.from_numpy(self.std_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_avg_particle_accs_final = torch.from_numpy(self.avg_particle_accs_final).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_std_particle_accs_final = torch.from_numpy(self.std_particle_accs_final).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        # init_particle_xs = (init_particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + 1e-6)
        # # particle_accs = particle_acts
        # th_avg_accs_all = torch.cat([th_avg_particle_accs_tau, th_avg_particle_accs, th_avg_particle_accs_final], dim=-1)
        # th_std_accs_all = torch.cat([th_std_particle_accs_tau, th_std_particle_accs, th_std_particle_accs_final], dim=-1)
        # particle_acts = (particle_acts - th_avg_accs_all) / (th_std_accs_all + 1e-6)
        
        # th_avg_particle_accs = torch.from_numpy(self.avg_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        # th_std_particle_accs = torch.from_numpy(self.std_particle_accs).float().to(init_particle_xs.device).unsqueeze(0).unsqueeze(0)
        
        # init_particle_xs = (init_particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + 1e-6)
        # # particle_acts = particle_acts.transpose(1, 0).contiguous().view(particle_acts.size(1), -1)
        # particle_acts = (particle_acts - th_avg_particle_accs) / (th_std_particle_accs + 1e-6)
        
        scaled_data = {
            'X': particle_xs, 
            'E': particle_acts
        }
        return scaled_data
    
    
    
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
        
        # # data_dict_th = {
        # #     key: torch.from_numpy(data_dict_np[key]).float() for key in data_dict_np
        # # }
        
        return data_dict_th
    
    
    
    # def get_data_via_index(self, index) -->getitem ##
    def __getitem__(self, index):
        # print(f"data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
            # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ###
        
        ## TODO: data selecting, data parsing, and data scaling ##
        # if self.use_target_data:
        #     cur_data_scaled = self.get_target_data()
        # else:
        # use target data for the scaling #
        
        cur_data_scaled = self.scale_data(cur_data, cur_data_nm) ## scale the data
        
        # ## ## #
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = {
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ") 
        return cur_data_scaled_th





class Uni_Manip_3D_PC_V5_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst # get the all clips inst #
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages ## get the nn)stages from the dataset config ##
        
        
        
        # self.data_inst_fn = "save_info_v5.npy"
        exp_tags = ["tds_exp_2"]

        ## root_data_folder ##
        self.data_list = []
        self.valid_subfolders = []
        
        # if self.all_clips_inst: # get the all clips instances # #
        self.data_inst_fn = f"save_info_v5.npy"
        print(f"data_inst_fn changed to {self.data_inst_fn} with all_clips_inst: {self.all_clips_inst}")
        for exp_tag in exp_tags:
            cur_data_folder = os.path.join(self.data_folder, exp_tag)
            tmp_data_list = os.listdir(cur_data_folder)
            
            for cur_subfolder in tmp_data_list:
                
                # cur_subfolder_angle = cur_subfolder.split("_")
                ####### ====== for positiive angles ====== #######
                # for i_split in range(len(cur_subfolder_angle)):
                #     if cur_subfolder_angle[i_split] == "objgoalrot":
                #         cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                #         break
                # if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                #     continue
                ####### ====== for positiive angles ====== #######
                
                
                inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                if os.path.isdir(inst_folder):
                    ####### ======= get the instance statistics info fn ====== ######
                    save_cur_inst_statistics_info_fn = "save_info_v5_statistics.npy"
                    save_cur_inst_statistics_info_fn = os.path.join(inst_folder, save_cur_inst_statistics_info_fn)

                    if not os.path.exists(save_cur_inst_statistics_info_fn):
                        continue
                    
                    cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                    if os.path.exists(cur_inst_file):
                        
                        self.data_list.append(cur_inst_file)
                        self.valid_subfolders.append(cur_subfolder)
        
        # if self.all_clips_inst:
        #     valid_data_list_sv_fn = f"valid_data_statistics_v4.npy" 
        # elif self.single_inst: 
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3.npy" 
        # else:
        #     # valid_data_list_sv_fn = f"valid_data_statistics_v3_all.npy" ## solve the task better? ##
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3_positive_angles.npy"
        # # valid_data_list_sv_fn = f"valid_data_statistics_v3.npy"
        
        ####### ======= get the single inst ======= #######
        if self.single_inst:
            valid_data_list_sv_fn = "save_info_v5_statistics.npy"
            valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], self.valid_subfolders[0], valid_data_list_sv_fn)
        else:
            valid_data_list_sv_fn = "save_info_v5_statistics.npy"
            valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
        
        
        valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        
        self.avg_verts_tot_cases_tot_ts = valid_data_statistics['avg_verts']
        self.std_verts_tot_cases_tot_ts = valid_data_statistics['std_verts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['avg_qtar_verts']
        self.std_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['std_qtar_verts']
        
        
        
        self.maxx_nn_pts = 512
        
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        
        self._preload_data()
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        # data_name_to_data, data_name_to_fps_idxes #
        return len(self.data_name_to_data)
        # return len(self.data_name_list)
    
    
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    def _preload_data(self, ):
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        
        single_clip_length = 300
        sliding_window_length = 100
        
        for data_nm in self.data_name_list:
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            # tot_verts tot_verts_integrated_qdd_tau
            tot_verts = cur_data['tot_verts'] #
            tot_verts_integrated_qdd_tau = cur_data['tot_qtar_verts'] #
            # nn_ts x nn_verts x 3 #
            
            mean_tot_verts = np.mean(tot_verts, axis=1)
            mean_tot_verts = np.mean(mean_tot_verts, axis=0)
            
            mean_tot_verts_qdd = np.mean(tot_verts_integrated_qdd_tau, axis=1)
            mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
            
            print(f"mean_tot_verts: {mean_tot_verts}, mean_tot_verts_qdd: {mean_tot_verts_qdd}")
            
            ###### ===== get the verts integrated with qd ===== ######
            # tot_verts_integrated_qd = cur_data['tot_verts_integrated_qd'] #
            ###### ===== get the verts integrated with qd ===== ######
        
            # cur_verts_expanded, cur_verts_integrated_qdd_tau #
            tot_verts_expanded = []
            tot_verts_integrated_qdd_tau_expanded = []
            
        
            # if self.all_clips_inst: 
            for i_starting_ts in range(0, tot_verts.shape[1] - single_clip_length, sliding_window_length):
                cur_ending_ts = i_starting_ts + single_clip_length
                cur_tot_verts = tot_verts[:, i_starting_ts:cur_ending_ts]
                
                ###### ===== get the verts integrated with qd ===== ######
                # cur_tot_verts_integrated_qd = tot_verts_integrated_qd[:, i_starting_ts: cur_ending_ts]
                ###### ===== get the verts integrated with qd ===== ######
                
                cur_tot_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau[:, i_starting_ts: cur_ending_ts] 
                
                
                first_fr_cur_tot_verts = cur_tot_verts[:, 0]
                ### TODO: another cnetralization strategy  ? ##
                first_fr_verts_offset = first_fr_cur_tot_verts[0] ## (3,) - shape of the offset tensor ##
                
                cur_tot_verts = cur_tot_verts - first_fr_verts_offset[None][None] ## the cur_tot_verts - (1,1,3) # first frame vertices offset #
                
                ###### ===== get the verts integrated with qd ===== ######
                # cur_tot_verts_integrated_qd = cur_tot_verts_integrated_qd - first_fr_verts_offset[None][None] 
                ###### ===== get the verts integrated with qd ===== ######
                
                cur_tot_verts_integrated_qdd_tau = cur_tot_verts_integrated_qdd_tau - first_fr_verts_offset[None][None] 
                
                
                
                
                
                cur_data_nm = f"data_nm_ist_{i_starting_ts}_ied_{cur_ending_ts}"
                cur_clip_data = {
                    'tot_verts': cur_tot_verts, 
                    # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                    'tot_verts_integrated_qdd_tau': cur_tot_verts_integrated_qdd_tau
                }
                self.data_name_to_data[cur_data_nm] = cur_clip_data
                
                init_verts = cur_tot_verts[:, 0]
                particle_init_xs_th = torch.from_numpy(init_verts).float()
                
                
                sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                if not os.path.exists(sampled_particle_idxes_sv_fn):
                    sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    
                    np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)  # saved to the particles idxes #
                else:
                    sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True) # 
                    
            
                self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
                
                
                # cur_verts_expanded, cur_verts_integrated_qdd_tau #
                # cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
                # cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                    
                    
                tot_verts_expanded.append(cur_tot_verts)
                tot_verts_integrated_qdd_tau_expanded.append(tot_verts_integrated_qdd_tau) ##
                
                # if self.single_inst:
                #     cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
                #     cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                    
                    
                #     avg_cur_verts_expanded = np.mean(cur_verts_expanded, axis=0)
                #     std_cur_verts_expanded = np.std(cur_verts_expanded, axis=0)
                #     avg_cur_verts_integrated_qdd_tau = np.mean(cur_verts_integrated_qdd_tau, axis=0)
                #     std_cur_verts_integrated_qdd_tau = np.std(cur_verts_integrated_qdd_tau, axis=0)
                    
                #     self.avg_verts_tot_cases_tot_ts = avg_cur_verts_expanded
                #     self.std_verts_tot_cases_tot_ts = std_cur_verts_expanded
                    
                #     self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_cur_verts_integrated_qdd_tau
                #     self.std_verts_qdd_tau_tot_cases_tot_ts = std_cur_verts_integrated_qdd_tau
                #     break
                
            #### single inst scaling v2 ####
            # if self.single_inst:
                
            #     tot_verts_expanded = np.concatenate(tot_verts_expanded, axis=0)
            #     tot_verts_integrated_qdd_tau_expanded = np.concatenate(tot_verts_integrated_qdd_tau_expanded, axis=0)
                
            #     tot_verts_expanded = tot_verts_expanded.reshape(-1, 3)
            #     tot_verts_integrated_qdd_tau_expanded = tot_verts_integrated_qdd_tau_expanded.reshape(-1, 3)
                
            #     avg_verts_expanded = np.mean(tot_verts_expanded, axis=0)
            #     std_verts_expanded = np.std(tot_verts_expanded, axis=0)
                
            #     avg_verts_integrated_qdd_tau = np.mean(tot_verts_integrated_qdd_tau_expanded, axis=0)
            #     std_verts_integrated_qdd_tau = np.std(tot_verts_integrated_qdd_tau_expanded, axis=0)
                
            #     self.avg_verts_tot_cases_tot_ts = avg_verts_expanded
            #     self.std_verts_tot_cases_tot_ts = std_verts_expanded
            #     self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_verts_integrated_qdd_tau
            #     self.std_verts_qdd_tau_tot_cases_tot_ts = std_verts_integrated_qdd_tau
                
            #     # cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
            #     # cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                
            #     # avg_cur_verts_expanded = np.mean(cur_verts_expanded, axis=0)
            #     # std_cur_verts_expanded = np.std(cur_verts_expanded, axis=0) # std cur #
            #     # avg_cur_verts_integrated_qdd_tau = np.mean(cur_verts_integrated_qdd_tau, axis=0)
            #     # std_cur_verts_integrated_qdd_tau = np.std(cur_verts_integrated_qdd_tau, axis=0)
                
            #     # self.avg_verts_tot_cases_tot_ts = avg_cur_verts_expanded
            #     # self.std_verts_tot_cases_tot_ts = std_cur_verts_expanded
                
            #     # self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_cur_verts_integrated_qdd_tau
            #     # self.std_verts_qdd_tau_tot_cases_tot_ts = std_cur_verts_integrated_qdd_tau
            #     break
                    
                    
            # else:
            #     self.data_name_to_data[data_nm] = cur_data
                
            #     init_verts = cur_data['tot_verts'][:, 0]
            #     particle_init_xs_th = torch.from_numpy(init_verts).float()
                
            #     # 
            #     sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                
            #     if not os.path.exists(sampled_particle_idxes_sv_fn):
            #         sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    
            #         np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)  # saved to the particles idxes #
            #     else:
            #         sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True) # 
                
            #     # sampled particle idxes #
                
            #     # sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                
                
            #     self.data_name_to_fps_idxes[data_nm] = sampled_particle_idxes
        
        
        # self.data name to fps idxes #
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")

        self.data_name_list = list(self.data_name_to_data.keys()) # data name to data # 
    
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_v2(self, data_dict): # bsz x nn_particles x feat_dim #
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        

        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        
        
        inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
        inv_scaled_particle_xs_integrated_taus = (data_E * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
        
        
        ###### ======= n-scale the data ======= ######
        # data_E_inv_scaled = data_E
        # data_X_inv_scaled = data_X
        ###### ======= n-scale the data ======= ######
        
        
        rt_dict = {
            'tot_verts': inv_scaled_particle_xs,
            'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
        }
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
        
        return rt_dict
    
    
    def scale_data(self, data_dict, data_nm):
        
        ## nn_ts x nn_particles x 3 ##
        
        eps = 1e-6
        particle_xs = data_dict['tot_verts']
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        
        
        particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
        particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
        # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
        sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        
        particle_xs = particle_xs[sampled_particle_idxes, :, :]
        particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
        
        
        return {
            'X': particle_xs,
            'E': particle_xs_integrated_qdd_tau,
            'sampled_idxes': sampled_particle_idxes,
        }
        
    def scale_data_batched(self, data_dict):
        particle_xs = data_dict['X']
        particle_acts = data_dict['E']
        
        eps = 1e-6
        
        ## 
        # bsz x nn_ts x nn_particles x 3 #
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus =  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        particle_xs = (particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + eps)
        particle_acts = (particle_acts - th_avg_particle_xs_integrated_taus) / (th_std_particle_xs_integrated_taus + eps)
        
        
        scaled_data = {
            'X': particle_xs, 
            'E': particle_acts
        }
        return scaled_data
    
    
    
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
        
        
        return data_dict_th
    
    
    
    # def get_data_via_index(self, index) -->getitem ##
    def __getitem__(self, index):
        # print(f"data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
            # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ###
        
        ## TODO: data selecting, data parsing, and data scaling ##
        # if self.use_target_data:
        #     cur_data_scaled = self.get_target_data()
        # else:
        # use target data for the scaling #
        
        cur_data_scaled = self.scale_data(cur_data, cur_data_nm) ## scale the data
        
        # ## ## #
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = {
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ") 
        return cur_data_scaled_th



 
class Uni_Manip_3D_PC_V6_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        # #
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages
        
        
        exp_tags = ["tds_exp_2"]

        ## root_data_folder ##
        self.data_list = []
        self.valid_subfolders = []
        
        # if self.all_clips_inst:
        self.data_inst_fn = f"save_info_v6.npy"
        print(f"data_inst_fn changed to {self.data_inst_fn} with all_clips_inst: {self.all_clips_inst}")
        for exp_tag in exp_tags:
            cur_data_folder = os.path.join(self.data_folder, exp_tag)
            tmp_data_list = os.listdir(cur_data_folder)
            
            for cur_subfolder in tmp_data_list: # getting data #
                
                # cur_subfolder_angle = cur_subfolder.split("_")
                ####### ====== for positiive angles ====== #######
                # for i_split in range(len(cur_subfolder_angle)):
                #     if cur_subfolder_angle[i_split] == "objgoalrot":
                #         cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                #         break
                # if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                #     continue
                ####### ====== for positiive angles ====== #######
                
                
                inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                if os.path.isdir(inst_folder):
                    ####### ======= get the instance statistics info fn ====== ######
                    save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
                    save_cur_inst_statistics_info_fn = os.path.join(inst_folder, save_cur_inst_statistics_info_fn)

                    if not os.path.exists(save_cur_inst_statistics_info_fn):
                        continue
                    
                    cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                    if os.path.exists(cur_inst_file):
                        
                        self.data_list.append(cur_inst_file)
                        self.valid_subfolders.append(cur_subfolder)
        
        # if self.all_clips_inst:
        #     valid_data_list_sv_fn = f"valid_data_statistics_v4.npy" 
        # elif self.single_inst: 
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3.npy" 
        # else:
        #     # valid_data_list_sv_fn = f"valid_data_statistics_v3_all.npy" ## solve the task better? ##
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3_positive_angles.npy"
        # # valid_data_list_sv_fn = f"valid_data_statistics_v3.npy"
        
        ####### ======= get the single inst ======= #######
        if self.single_inst: # save info v6 statistics #
            valid_data_list_sv_fn = "save_info_v6_statistics.npy"
            valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], self.valid_subfolders[0], valid_data_list_sv_fn)
        else:
            valid_data_list_sv_fn = "save_info_v6_statistics.npy"
            valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
        
        
        valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        
        self.avg_verts_tot_cases_tot_ts = valid_data_statistics['avg_verts']
        self.std_verts_tot_cases_tot_ts = valid_data_statistics['std_verts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['avg_qtar_verts']
        self.std_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['std_qtar_verts']
        
        
        
        self.maxx_nn_pts = 512 ## ##
        
        # self.maxx nn pts ##
        # 
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        
        self._preload_data()
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        # data_name_to_data, data_name_to_fps_idxes #
        return len(self.data_name_to_data)
        # return len(self.data_name_list)
    
    
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    def _preload_data(self, ):
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        # self.data_name_list #
        
        # single_clip_length = 300
        # sliding_window_length = 100
        
        for data_nm in self.data_name_list:
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            # selected_frame_verts, selected_frame_qtars_verts
            # tot_verts tot_verts_integrated_qdd_tau
            tot_verts = cur_data['selected_frame_verts'] # tot verts #
            tot_verts_integrated_qdd_tau = cur_data['selected_frame_qtars_verts'] #
            # nn_ts x nn_verts x 3 #
            
            mean_tot_verts = np.mean(tot_verts, axis=1)
            mean_tot_verts = np.mean(mean_tot_verts, axis=0)
            
            mean_tot_verts_qdd = np.mean(tot_verts_integrated_qdd_tau, axis=1)
            mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
            
            print(f"mean_tot_verts: {mean_tot_verts}, mean_tot_verts_qdd: {mean_tot_verts_qdd}")
            
            
            cur_data_nm = data_nm
            cur_clip_data = {
                'tot_verts': tot_verts, 
                # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                'tot_verts_integrated_qdd_tau': tot_verts_integrated_qdd_tau
            }
            self.data_name_to_data[cur_data_nm] = cur_clip_data
            
            init_verts = tot_verts[:, 0] # 
            particle_init_xs_th = torch.from_numpy(init_verts).float()
            
            
            ### get the particle idxes  ###
            # get partcle init xs #
            sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
            if not os.path.exists(sampled_particle_idxes_sv_fn):
                sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)
            else:
                sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True)
            
            self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
            
            
            # ###### ===== get the verts integrated with qd ===== ######
            # # tot_verts_integrated_qd = cur_data['tot_verts_integrated_qd'] #
            # ###### ===== get the verts integrated with qd ===== ######
        
            # # cur_verts_expanded, cur_verts_integrated_qdd_tau #
            # tot_verts_expanded = []
            # tot_verts_integrated_qdd_tau_expanded = []
            
        
            # # if self.all_clips_inst: 
            # for i_starting_ts in range(0, tot_verts.shape[1] - single_clip_length, sliding_window_length):
            #     cur_ending_ts = i_starting_ts + single_clip_length
            #     cur_tot_verts = tot_verts[:, i_starting_ts:cur_ending_ts]
                
            #     ###### ===== get the verts integrated with qd ===== ######
            #     # cur_tot_verts_integrated_qd = tot_verts_integrated_qd[:, i_starting_ts: cur_ending_ts]
            #     ###### ===== get the verts integrated with qd ===== ######
                
            #     cur_tot_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau[:, i_starting_ts: cur_ending_ts] 
                
                
            #     first_fr_cur_tot_verts = cur_tot_verts[:, 0]
            #     ### TODO: another cnetralization strategy  ? ##
            #     first_fr_verts_offset = first_fr_cur_tot_verts[0] ## (3,) - shape of the offset tensor ##
                
            #     cur_tot_verts = cur_tot_verts - first_fr_verts_offset[None][None] ## the cur_tot_verts - (1,1,3) # first frame vertices offset #
                
            #     ###### ===== get the verts integrated with qd ===== ######
            #     # cur_tot_verts_integrated_qd = cur_tot_verts_integrated_qd - first_fr_verts_offset[None][None] 
            #     ###### ===== get the verts integrated with qd ===== ######
                
            #     cur_tot_verts_integrated_qdd_tau = cur_tot_verts_integrated_qdd_tau - first_fr_verts_offset[None][None] 
                
                
                
                
                
            #     cur_data_nm = f"data_nm_ist_{i_starting_ts}_ied_{cur_ending_ts}"
            #     cur_clip_data = {
            #         'tot_verts': cur_tot_verts, 
            #         # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
            #         'tot_verts_integrated_qdd_tau': cur_tot_verts_integrated_qdd_tau
            #     }
            #     self.data_name_to_data[cur_data_nm] = cur_clip_data
                
            #     init_verts = cur_tot_verts[:, 0]
            #     particle_init_xs_th = torch.from_numpy(init_verts).float()
                
                
            #     sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
            #     if not os.path.exists(sampled_particle_idxes_sv_fn):
            #         sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    
            #         np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)  # saved to the particles idxes #
            #     else:
            #         sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True) # 
                    
            
            #     self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
                
                
            #     # cur_verts_expanded, cur_verts_integrated_qdd_tau #
            #     # cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
            #     # cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                    
                    
            #     tot_verts_expanded.append(cur_tot_verts)
            #     tot_verts_integrated_qdd_tau_expanded.append(tot_verts_integrated_qdd_tau) ##
                
            #     # if self.single_inst:
            #     #     cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
            #     #     cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                    
                    
            #     #     avg_cur_verts_expanded = np.mean(cur_verts_expanded, axis=0)
            #     #     std_cur_verts_expanded = np.std(cur_verts_expanded, axis=0)
            #     #     avg_cur_verts_integrated_qdd_tau = np.mean(cur_verts_integrated_qdd_tau, axis=0)
            #     #     std_cur_verts_integrated_qdd_tau = np.std(cur_verts_integrated_qdd_tau, axis=0)
                    
            #     #     self.avg_verts_tot_cases_tot_ts = avg_cur_verts_expanded
            #     #     self.std_verts_tot_cases_tot_ts = std_cur_verts_expanded
                    
            #     #     self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_cur_verts_integrated_qdd_tau
            #     #     self.std_verts_qdd_tau_tot_cases_tot_ts = std_cur_verts_integrated_qdd_tau
            #     #     break
                
            #### single inst scaling v2 ####
            # if self.single_inst:
                
            #     tot_verts_expanded = np.concatenate(tot_verts_expanded, axis=0)
            #     tot_verts_integrated_qdd_tau_expanded = np.concatenate(tot_verts_integrated_qdd_tau_expanded, axis=0)
                
            #     tot_verts_expanded = tot_verts_expanded.reshape(-1, 3)
            #     tot_verts_integrated_qdd_tau_expanded = tot_verts_integrated_qdd_tau_expanded.reshape(-1, 3)
                
            #     avg_verts_expanded = np.mean(tot_verts_expanded, axis=0)
            #     std_verts_expanded = np.std(tot_verts_expanded, axis=0)
                
            #     avg_verts_integrated_qdd_tau = np.mean(tot_verts_integrated_qdd_tau_expanded, axis=0)
            #     std_verts_integrated_qdd_tau = np.std(tot_verts_integrated_qdd_tau_expanded, axis=0)
                
            #     self.avg_verts_tot_cases_tot_ts = avg_verts_expanded
            #     self.std_verts_tot_cases_tot_ts = std_verts_expanded
            #     self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_verts_integrated_qdd_tau
            #     self.std_verts_qdd_tau_tot_cases_tot_ts = std_verts_integrated_qdd_tau
                
            #     # cur_verts_expanded = cur_tot_verts.reshape(-1, 3)
            #     # cur_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau.reshape(-1, 3) # 
                
            #     # avg_cur_verts_expanded = np.mean(cur_verts_expanded, axis=0)
            #     # std_cur_verts_expanded = np.std(cur_verts_expanded, axis=0) # std cur #
            #     # avg_cur_verts_integrated_qdd_tau = np.mean(cur_verts_integrated_qdd_tau, axis=0)
            #     # std_cur_verts_integrated_qdd_tau = np.std(cur_verts_integrated_qdd_tau, axis=0)
                
            #     # self.avg_verts_tot_cases_tot_ts = avg_cur_verts_expanded
            #     # self.std_verts_tot_cases_tot_ts = std_cur_verts_expanded
                
            #     # self.avg_verts_qdd_tau_tot_cases_tot_ts = avg_cur_verts_integrated_qdd_tau
            #     # self.std_verts_qdd_tau_tot_cases_tot_ts = std_cur_verts_integrated_qdd_tau
            #     break
                    
                    
            # else:
            #     self.data_name_to_data[data_nm] = cur_data
                
            #     init_verts = cur_data['tot_verts'][:, 0]
            #     particle_init_xs_th = torch.from_numpy(init_verts).float()
                
            #     # 
            #     sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                
            #     if not os.path.exists(sampled_particle_idxes_sv_fn):
            #         sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    
            #         np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)  # saved to the particles idxes #
            #     else:
            #         sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True) # 
                
            #     # sampled particle idxes #
                
            #     # sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                
                
            #     self.data_name_to_fps_idxes[data_nm] = sampled_particle_idxes
        
        
        # self.data name to fps idxes #
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")

        self.data_name_list = list(self.data_name_to_data.keys()) # data name to data # 
    
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_v2(self, data_dict): # bsz x nn_particles x feat_dim #
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        

        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        
        
        inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
        inv_scaled_particle_xs_integrated_taus = (data_E * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
        
        
        ###### ======= n-scale the data ======= ######
        # data_E_inv_scaled = data_E
        # data_X_inv_scaled = data_X
        ###### ======= n-scale the data ======= ######
        
        
        rt_dict = {
            'tot_verts': inv_scaled_particle_xs,
            'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
        }
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
        
        return rt_dict
    
    
    def scale_data(self, data_dict, data_nm):
        
        ## nn_ts x nn_particles x 3 ##
        
        eps = 1e-6
        particle_xs = data_dict['tot_verts']
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        
        
        particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
        particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
        # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
        sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        
        particle_xs = particle_xs[sampled_particle_idxes, :, :]
        particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
        
        
        return {
            'X': particle_xs,
            'E': particle_xs_integrated_qdd_tau,
            'sampled_idxes': sampled_particle_idxes,
        }
        
    def scale_data_batched(self, data_dict):
        particle_xs = data_dict['X']
        particle_acts = data_dict['E']
        
        eps = 1e-6
        
        ## 
        # bsz x nn_ts x nn_particles x 3 #
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus =  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(particle_xs.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        particle_xs = (particle_xs - th_avg_particle_init_xs) / (th_std_particle_init_xs + eps)
        particle_acts = (particle_acts - th_avg_particle_xs_integrated_taus) / (th_std_particle_xs_integrated_taus + eps)
        
        
        scaled_data = {
            'X': particle_xs, 
            'E': particle_acts
        }
        return scaled_data
    
    
    
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
        
        
        return data_dict_th
    
    
    
    # def get_data_via_index(self, index) -->getitem ##
    def __getitem__(self, index):
        # print(f"data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        cur_data_nm = self.data_name_list[index]
        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
            # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        else:
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ###
        
        ## TODO: data selecting, data parsing, and data scaling ##
        # if self.use_target_data:
        #     cur_data_scaled = self.get_target_data()
        # else:
        # use target data for the scaling #
        
        cur_data_scaled = self.scale_data(cur_data, cur_data_nm) ## scale the data
        
        # ## ## #
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = {
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ") # 
        return cur_data_scaled_th




# @torch.jit.script
def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)



class Uni_Manip_3D_PC_V7_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        self.debug = self.cfg.debug
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        #
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        self.multi_inst = cfg.dataset_3d_pc.multi_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst
        self.specified_hand_type = cfg.dataset_3d_pc.specified_hand_type 
        
        self.specified_object_type = cfg.dataset_3d_pc.specified_object_type
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages
        self.use_static_first_frame = cfg.dataset_3d_pc.use_static_first_frame
        self.use_shadow_test_data = cfg.sampling.use_shadow_test_data
        self.sampling = cfg.sampling.sampling
        
        # self.use_allegro_test_data = cfg.sampling.use_allegro_test_data
        self.specified_test_subfolder = cfg.sampling.specified_test_subfolder
        self.specified_statistics_info_fn = cfg.training.specified_statistics_info_fn
        self.specified_sampled_particle_idxes_fn = cfg.training.specified_sampled_particle_idxes_fn
        
        self.training_setting = cfg.training.setting ## training setting ## 
        self.use_jointspace_seq = cfg.training.use_jointspace_seq
        
        # 
        self.task_cond = cfg.training.task_cond # 
        self.diff_task_space = cfg.training.diff_task_space
        self.diff_task_translations = cfg.training.diff_task_translations
        
        self.kine_diff = cfg.training.kine_diff
        self.tracking_ctl_diff = cfg.training.tracking_ctl_diff
        
        # target_grab_inst_tag, target_grab_inst_opt_fn #
        ''' for sampling '''  #
        #### get the jtarget jrab inst tagandthe optimized fn ###
        self.target_grab_inst_tag = cfg.sampling.target_grab_inst_tag
        self.target_grab_inst_opt_fn = cfg.sampling.target_grab_inst_opt_fn
        
        ''' for training and the training data '''
        self.grab_inst_tag_to_optimized_res_fn = cfg.training.grab_inst_tag_to_optimized_res_fn
        self.taco_inst_tag_to_optimized_res_fn = cfg.training.taco_inst_tag_to_optimized_res_fn
        if len(self.taco_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.taco_inst_tag_to_optimized_res_fn):
            self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn, self.taco_inst_tag_to_optimized_res_fn]    
    
    
        try:
            self.use_taco_data = cfg.training.use_taco_data
        except:
            self.use_taco_data = False
    
        try:
            self.glb_rot_use_quat = cfg.training.glb_rot_use_quat
        except:
            self.glb_rot_use_quat = False
        self.succ_rew_threshold = cfg.training.succ_rew_threshold # 
        
        
        try:
            self.task_cond_type = cfg.training.task_cond_type
        except:
            self.task_cond_type = "future"
        
        try:
            self.slicing_ws = cfg.training.slicing_ws
        except:
            self.slicing_ws = 30
            pass
        
        ### TODO: a slicing ws with an additional history window ws for tracking ###
        ### trajs obtained via closed loop planning? ###
        
        try:
            self.history_ws = cfg.training.history_ws
        except:
            self.history_ws = self.slicing_ws
        
        
        try:
            self.use_kine_obj_pos_canonicalization = cfg.training.use_kine_obj_pos_canonicalization
        except:
            self.use_kine_obj_pos_canonicalization = False
        
        
        try:
            self.exp_additional_tag = cfg.training.exp_additional_tag
        except:
            self.exp_additional_tag = ''
    
        #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy
        # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
        try:
            self.taco_interped_fr_grab_tag = cfg.training.taco_interped_fr_grab_tag
        except:
            self.taco_interped_fr_grab_tag = "ori_grab_s2_phone_call_1"
        
        try:
            self.taco_interped_data_sv_additional_tag = cfg.training.taco_interped_data_sv_additional_tag
        except:
            self.taco_interped_data_sv_additional_tag = 'v2'
        
        try:
            self.num_frames = cfg.training.num_frames
        except:
            self.num_frames = 150
        
        valid_data_statistics = None 
        
        try:
            self.task_inherit_info_fn = cfg.training.task_inherit_info_fn
        except:
            self.task_inherit_info_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy"
            
        # dataset #
        # self.step_size = 30
        # dataset #
        
        # try:
        #     self.step_size = 
        # self.step_size = self.slicing_ws // 2
        
        # if self.step_size == 0:
        #     self.step_size = 1
            
        self.step_size = 1
        
        try:
            self.slicing_data = cfg.training.slicing_data
        except:
            self.slicing_data = False
        
        
        # diff_task_translations and diff_task_space and task_cond #
        self.obj_name_to_idx = {
            'box': 0,
            'cylinder': 1
        }
        
        self.nn_hands_dof = 22
        
        exp_tags = ["tds_exp_2"]
        
        
        self.sim_platform = cfg.dataset_3d_pc.sim_platform
        
        self.data_statistics_info_fn = cfg.dataset_3d_pc.data_statistics_info_fn
        
        self.statistics_info_fn = cfg.dataset_3d_pc.statistics_info_fn
        
        print(f"statistics_info_fn: {self.statistics_info_fn}")
        
        self.tot_inheriting_infos = []

        self.hybrid_dataset = False

        # self.tracking_save_info_fn = "/cephfs/yilaa/data/GRAB_Tracking/data"
        self.tracking_save_info_fn = cfg.dataset_3d_pc.tracking_save_info_fn 
        self.tracking_info_st_tag = "passive_active_info_"
        
        self.target_grab_data_nm = None
        
        if self.kine_diff:
            
            if self.sim_platform == 'isaac':
                passive_act_info_tag = 'passive_active_info_ori_grab'
                # tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
                tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
                print("Start enumerating retargeted tracking info")
                tracking_save_info = os.listdir(tracking_save_info_fn)
                tracking_save_info = [
                    fn for fn in tracking_save_info if fn.endswith(".npy") and fn[: len(passive_act_info_tag)] == passive_act_info_tag
                ]
                if self.num_frames == 150:
                    non_nf_tag = '_nf_'
                    tracking_save_info = [
                        fn for fn in tracking_save_info if non_nf_tag not in fn
                     ]
                else:
                    nf_tag = f'_nf_{self.num_frames}'
                    tracking_save_info = [
                        fn for fn in tracking_save_info if nf_tag in fn
                    ]
                
                passive_act_pure_tag = "passive_active_info_"
                self.objtype_to_tracking_sv_info = {}
                for cur_sv_info in tracking_save_info:
                    cur_objtype = cur_sv_info.split(".")[0]
                    cur_objtype = cur_objtype.split("_nf_")[0]
                    cur_objtype = cur_objtype[len(passive_act_pure_tag): ]
                    self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)
                    
                    # [len(passive_act_info_tag): ]
                    
                    
                    # cur_objtype = "ori_grab" + cur_objtype
                    # self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)
                # if self.num_frames == 150:
                #     pass
                tracking_save_info = [
                    os.path.join(tracking_save_info_fn, fn) for fn in tracking_save_info
                ]
                self.data_list = tracking_save_info
                
                if self.use_taco_data:
                    # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231002_050_v2.npy
                    passive_act_info_tag = 'passive_active_info_ori_grab_s2_phone_call_1_interped_'
                    # tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
                    tracking_save_info_fn = "/cephfs/xueyi/data/TACO_Tracking_PK/data"
                    print("Start enumerating retargeted tracking info")
                    tracking_save_info = os.listdir(tracking_save_info_fn)
                    tracking_save_info = [
                        fn for fn in tracking_save_info if fn.endswith("_v2.npy") and fn[: len(passive_act_info_tag)] == passive_act_info_tag
                    ]
                    # if self.num_frames == 150:
                    #     non_nf_tag = '_nf_'
                    #     tracking_save_info = [
                    #         fn for fn in tracking_save_info if non_nf_tag not in fn
                    #     ]
                    # else:
                    #     nf_tag = f'_nf_{self.num_frames}'
                    #     tracking_save_info = [
                    #         fn for fn in tracking_save_info if nf_tag in fn
                    #     ]
                    
                    passive_act_pure_tag = "passive_active_info_ori_grab_s2_phone_call_1_interped_"
                    self.objtype_to_tracking_sv_info = {}
                    for cur_sv_info in tracking_save_info:
                        cur_objtype = cur_sv_info.split(".")[0]
                        # cur_objtype = cur_objtype.split("_nf_")[0]
                        cur_objtype = cur_objtype[len(passive_act_pure_tag): ]
                        cur_objtype_segs = cur_objtype.split("_")
                        cur_objtype = "_".join(cur_objtype_segs[0: 3])
                        self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)
                        
                        # [len(passive_act_info_tag): ]
                        
                        
                        # cur_objtype = "ori_grab" + cur_objtype
                        # self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)
                    # if self.num_frames == 150:
                    #     pass
                    taco_tracking_save_info = [
                        os.path.join(tracking_save_info_fn, fn) for fn in tracking_save_info
                    ]
                    self.data_list += taco_tracking_save_info
                    
                
                
                print("End!")
                # 
                
                if self.sampling and len(self.target_grab_inst_tag) > 0:
                    target_inst_tag = self.target_grab_inst_tag
                    cur_inheriting_dict = {
                        'fa_objtype': target_inst_tag,
                        'fa_trajtype': target_inst_tag,
                        'ch_objtype': target_inst_tag,
                        'ch_trajtype': target_inst_tag
                    }
                    self.task_inheriting_dict_info = [cur_inheriting_dict]
                    
                    interested_task = set()
                    self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
                    for key in self.task_inheriting_dict_info[0]:
                        cur_val = self.task_inheriting_dict_info[0][key]
                        interested_task.add(cur_val)
                    self.data_list = [self.objtype_to_tracking_sv_info[cur_val] for cur_val in interested_task]
                
                else:
                    if self.task_cond:
                        # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy #
                        # statistics_folder = "/cephfs/xueyi/uni_manip/tds_rl_exp_ctlfreq_10_new/logs_PPO/statistics"
                        # kine_traj_task_inherit_info_fn = "child_task_to_fa_task.npy"
                        task_inherit_info = np.load(self.task_inherit_info_fn, allow_pickle=True).item() # task inherit info 
                        print(f"task_inherit_info: {len(task_inherit_info)}")
                        task_inheriting_dict_info = []
                        for child_task in task_inherit_info:
                            parenta_task = task_inherit_info[child_task]
                            ###### add all inheriting dict info here ######
                            if isinstance(parenta_task, list):
                                for cur_fa_task in parenta_task:
                                    cur_inheriting_dict = {
                                        'fa_objtype': cur_fa_task,
                                        'fa_trajtype': cur_fa_task,
                                        'ch_objtype': child_task,
                                        'ch_trajtype': child_task
                                    }
                                    task_inheriting_dict_info.append(cur_inheriting_dict)
                            else:
                                cur_inheriting_dict = {
                                    'fa_objtype': parenta_task,
                                    'fa_trajtype': parenta_task,
                                    'ch_objtype': child_task,
                                    'ch_trajtype': child_task
                                }
                                task_inheriting_dict_info.append(cur_inheriting_dict)
                        self.task_inheriting_dict_info = task_inheriting_dict_info 
                    
                    if self.task_cond and self.debug:
                        interested_task = set()
                        self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
                        for key in self.task_inheriting_dict_info[0]:
                            cur_val = self.task_inheriting_dict_info[0][key]
                            interested_task.add(cur_val)
                        self.data_list = [self.objtype_to_tracking_sv_info[cur_val] for cur_val in interested_task]
                
                
                # TODO: task cond setting #
                # TODO: hybrid dataset #
            else:
                passive_act_info_tag = "passive_active_info_ori_grab_s2"
                cur_statistics_info = np.load(self.statistics_info_fn, allow_pickle=True).item()
                valid_data_statistics = cur_statistics_info
                # tracking save info fn #
                tracking_save_info_fn = "/cephfs/yilaa/data/GRAB_Tracking/data"
                tracking_save_info = os.listdir(tracking_save_info_fn)
                tracking_save_info = [fn for fn in tracking_save_info if os.path.exists(os.path.join(tracking_save_info_fn, fn)) and fn.endswith(".npy") and fn and fn[:len(passive_act_info_tag)] == passive_act_info_tag] 
                ## get the tracking save info ##
                
                # get the objtype_to_tracking_sv_info #
                self.objtype_to_tracking_sv_info = {}
                for cur_sv_info in tracking_save_info:
                    cur_objtype = cur_sv_info.split(".")[0][len(passive_act_info_tag): ]
                    cur_objtype = "ori_grab_s2" + cur_objtype
                    self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)
                
                ## load the tracking info ##
                tracking_save_info = [os.path.join(tracking_save_info_fn, fn) for fn in tracking_save_info]
                self.data_list = tracking_save_info
                
                
                if self.task_cond:
                    ## ## task
                    statistsics_folder = "/cephfs/yilaa/uni_manip/tds_rl_exp_ctlfreq_10_new/logs_PPO/statistics"
                    # /cephfs/yilaa/uni_manip/tds_rl_exp_ctlfreq_10_new/logs_PPO/statistics/task_inheriting_traj_pairs.npy
                    task_inheriting_info_fn = "task_inheriting_traj_pairs.npy"
                    task_inheriting_info_fn = os.path.join(statistsics_folder, task_inheriting_info_fn)
                    task_inheriting_info = np.load(task_inheriting_info_fn, allow_pickle=True).item()
                    task_inheriting_dict_info = []
                    for child_task_info in task_inheriting_info:
                        cur_parent_task_settings = task_inheriting_info[child_task_info]
                        cur_child_objtype = child_task_info[0]
                        cur_child_trajtype = child_task_info[1]
                        for cur_parent_setting in cur_parent_task_settings:
                            cur_parent_objtype = cur_parent_setting[0]
                            cur_parent_trajtype = cur_parent_setting[1]
                            cur_inheriting_dict = {
                                'fa_objtype': cur_parent_objtype,
                                'fa_trajtype': cur_parent_trajtype,
                                'ch_objtype': cur_child_objtype,
                                'ch_trajtype': cur_child_trajtype
                            }
                            task_inheriting_dict_info.append(cur_inheriting_dict)
                    self.task_inheriting_dict_info = task_inheriting_dict_info # get the task inheritng dict info #
        
        
        elif self.tracking_ctl_diff:
            # tracking ctl diff #
            self.data_list = []
            self.data_inst_tag_list = []
            # history information # -- history ws may not be necessarily the same as the future ws for prediction # 
            # if self.target_grab_inst_tag is not None and len(self.target_grab_inst_tag) > 0:
            #     print(f"[Loading target grab data] target_grab_inst_tag: {self.target_grab_inst_tag}")
            #     cur_opt_res_fn = self.target_grab_inst_opt_fn
            #     data_root_folder = "/".join(cur_opt_res_fn.split("/")[:-1])
            #     raw_fn = ".".join(cur_opt_res_fn.split(".")[:-1])
            #     raw_saved_pts_fn = f"{raw_fn}_exported_pts.npy" # 
            #     full_raw_saved_pts_fn = os.path.join(data_root_folder, raw_saved_pts_fn)
                
            #     # if not os.path.exists(full_raw_saved_pts_fn):
            #     #         full_raw_saved_pts_fn = cur_opt_res_fn
                
            #     if not os.path.exists(full_raw_saved_pts_fn):
            #         full_raw_saved_pts_fn = self.target_grab_inst_opt_fn
                
            #     self.data_list.append(full_raw_saved_pts_fn)
            #     self.data_inst_tag_list.append(self.target_grab_inst_tag)
            
            # else:
                # grab_inst_tag_to_opt_res_fn = "grab_inst_tag_to_opt_res_merged_v1.npy"
                # grab_inst_tag_to_opt_res_fn = "grab_inst_tag_to_opt_res_merged.npy"
                # grab_inst_tag_to_opt_res_fn = os.path.join(rl_statistics_folder, grab_inst_tag_to_opt_res_fn)
            # 
            if self.sim_platform == 'isaac':
                
                ##### if we have moved them to the local folder, then use the local optimized res fn #####
                
                if isinstance(self.grab_inst_tag_to_optimized_res_fn, list):
                    self.hybrid_dataset = True
                    # pure_grab_inst_tag_to_opt_res_fn = [
                    #     cur_fn.split("/")[-1] for cur_fn in self.grab_inst_tag_to_optimized_res_fn
                    # ]
                    # local_grab_inst_tag_to_opt_res_fn = [ os.path.join(f"./assets", cur_fn) for cur ]
                    tot_grab_inst_tag_to_opt_res = {} # grab inst tag #
                    for cur_fn in self.grab_inst_tag_to_optimized_res_fn:
                        cur_opt_res = np.load(cur_fn, allow_pickle=True).item()
                        tot_grab_inst_tag_to_opt_res.update(cur_opt_res) # get the inst tag #
                    grab_inst_tag_to_opt_res =  tot_grab_inst_tag_to_opt_res
                else:
                    self.hybrid_dataset = False
                
                    pure_grab_inst_tag_to_opt_res_fn = self.grab_inst_tag_to_optimized_res_fn.split("/")[-1]
                    local_grab_inst_tag_to_opt_res_fn = os.path.join(f"./assets", pure_grab_inst_tag_to_opt_res_fn)
                    if os.path.exists(local_grab_inst_tag_to_opt_res_fn):
                        grab_inst_tag_to_opt_res_fn = local_grab_inst_tag_to_opt_res_fn
                    else:   
                        grab_inst_tag_to_opt_res_fn = self.grab_inst_tag_to_optimized_res_fn
                    # rew_threshold = 50.0
                    grab_inst_tag_to_opt_res = np.load(grab_inst_tag_to_opt_res_fn, allow_pickle=True).item()
                    
                pure_inst_tag_to_opt_stat_fn = self.cfg.training.grab_inst_tag_to_opt_stat_fn.split("/")[-1]
                local_inst_tag_to_opt_state_fn = os.path.join(f"./assets", pure_inst_tag_to_opt_stat_fn)
                if os.path.exists(local_inst_tag_to_opt_state_fn): # local isnt tag and the glb inst tag #
                    grab_inst_tag_to_opt_stat_fn = local_inst_tag_to_opt_state_fn
                else:
                    grab_inst_tag_to_opt_stat_fn = self.cfg.training.grab_inst_tag_to_opt_stat_fn
                
                grab_inst_tag_to_opt_stat = np.load(grab_inst_tag_to_opt_stat_fn, allow_pickle=True).item()
                
                ### TODO: add the tracking related statistics in the optimized info dictionary and use that to filter the training data ###
                ### TODO: figure out whether if there are some differences between the `key` used in such two dicts ###
                ### 
                for grab_inst_tag in grab_inst_tag_to_opt_res:
                    cur_grab_obj_type, cur_grab_traj_obj_type = grab_inst_tag
                    
                    if 'taco' in cur_grab_obj_type:
                        cur_grab_traj_obj_type = cur_grab_obj_type
                    
                    # if not self.hybrid_dataset:
                        # if cur_grab_obj_type not in grab_inst_tag_to_opt_stat:
                        #     continue
                        # cur_inst_opt_rew_val = grab_inst_tag_to_opt_stat[cur_grab_obj_type][0]
                    
                    if 'ori_grab' in cur_grab_obj_type:
                        if cur_grab_obj_type not in grab_inst_tag_to_opt_stat:
                            continue
                    
                    cur_inst_opt_fns = grab_inst_tag_to_opt_res[grab_inst_tag]
                    for i_inst, cur_inst_fn in enumerate(cur_inst_opt_fns):
                        cur_inst_sorted_val_fn = cur_inst_fn.replace(".npy", "_sorted.npy")
                        cur_inst_sorted_val_fn_best = cur_inst_sorted_val_fn.replace(".npy", "_best.npy")
                        if os.path.exists(cur_inst_sorted_val_fn_best):
                            cur_inst_sorted_val_fn = cur_inst_sorted_val_fn_best
                        # cur_full_sorted_val_fn = os.path.join(data_folder, cur_inst_sorted_val_fn)
                        self.data_list.append(cur_inst_sorted_val_fn)
                        self.data_inst_tag_list.append(grab_inst_tag)
                    
                    if self.debug:
                        break
            else:
                
                grab_inst_tag_to_opt_res_fn = self.grab_inst_tag_to_optimized_res_fn
                # grab_inst_tag_to_opt_res_f # to opt res fn ##
                grab_inst_tag_to_opt_res = np.load(grab_inst_tag_to_opt_res_fn, allow_pickle=True).item()
                # merged_statistics_info_fn = "grab_inst_tag_to_opt_res_merged.npy"
                # merged_statistics_info_fn = os.path.join(statistics_folder, merged_statistics_info_fn)
                # grab_inst_tag_to_opt_res = np.load(merged_statistics_info_fn, allow_pickle=True).item()
                
                for grab_inst_tag in  grab_inst_tag_to_opt_res:
                    # print(grab_inst_tag_to_opt_res[grab_inst_tag])
                    if len(grab_inst_tag_to_opt_res[grab_inst_tag]) == 0:
                        continue
                    
                    cur_opt_res_fn = grab_inst_tag_to_opt_res[grab_inst_tag][0][0]
                    
                    #### NOTE: newly added --- only the self-self optimized and `s2` subject are considered ####
                    cur_grab_obj_type, cur_grab_traj_obj_type = grab_inst_tag 
                    if cur_grab_obj_type != cur_grab_traj_obj_type:
                        continue
                    last_folder_nm = cur_opt_res_fn.split("/")[-2]
                    if '_s2_' not in last_folder_nm:
                        continue
                    #### NOTE: newly added --- only the self-self optimized and `s2` subject are considered ####
                    
                    
                    
                    cur_opt_res_rew_val = grab_inst_tag_to_opt_res[grab_inst_tag][0][1] # 
                    
                    if cur_opt_res_rew_val < self.succ_rew_threshold:
                        continue
                
                    # cur_opt_res = np.load(cur_opt_res_fn, allow_pickle=True).item()
                    # cur_opt_res.update(
                    #     get_pts_fr_qs_qtars(cur_opt_res_fn)
                    # ) # 
                    data_root_folder = "/".join(cur_opt_res_fn.split("/")[:-1])
                    raw_fn = ".".join(cur_opt_res_fn.split(".")[:-1])
                    raw_saved_pts_fn = f"{raw_fn}_exported_pts.npy" # 
                    full_raw_saved_pts_fn = os.path.join(data_root_folder, raw_saved_pts_fn)
                    if not os.path.exists(full_raw_saved_pts_fn):
                        full_raw_saved_pts_fn = cur_opt_res_fn
                    
                    # #### ==== export the raw saved fn ==== ####
                    # np.save(full_raw_saved_pts_fn, cur_opt_res) # save the results #
                    # print(f"exported points saved to {full_raw_saved_pts_fn}")
                    self.data_list.append(full_raw_saved_pts_fn) 
                    # self.data_list.append(grab_inst_tag) 
                    self.data_inst_tag_list.append(grab_inst_tag)
                    
                    if self.target_grab_inst_tag is not None and len(self.target_grab_inst_tag) > 0:
                        if self.target_grab_inst_tag == cur_grab_obj_type:
                            print(f"[Loading target grab data] target_grab_inst_tag: {self.target_grab_inst_tag}")
                            self.target_grab_data_nm = full_raw_saved_pts_fn
            # if self.targetgrab
            
            if os.path.exists(self.statistics_info_fn):
                cur_statistics_info = np.load(self.statistics_info_fn, allow_pickle=True).item()
    
                valid_data_statistics = cur_statistics_info
            else:
                valid_data_statistics = None



        ##### the number of the inheriting task data depends on corresponding values in statistics #####
        elif self.statistics_info_fn is not None and len(self.statistics_info_fn) > 0 and os.path.exists(self.statistics_info_fn): 
            print(f"loading from statistics {self.statistics_info_fn}")
            cur_statistics_info = np.load(self.statistics_info_fn, allow_pickle=True).item()
            valid_data_statistics = cur_statistics_info
            # training_setting # # conditiona l trans # 
            if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond'] :
                valid_data_folder = "/".join(self.statistics_info_fn.split("/")[:-1])
                tot_subfolders = os.listdir(valid_data_folder)
                print(f"valid_data_folder: {valid_data_folder}")
                tot_subfolders = [fn for fn in tot_subfolders if os.path.exists(os.path.join(valid_data_folder, fn))]
                ## data list ##
                self.data_list = [] ##
                
                for cur_fodler in tot_subfolders:
                    cur_full_folder = os.path.join(valid_data_folder, cur_fodler)
                    # saved_info = { # optimized info fn ## use the same statistics #
                    #     'unopt_verts': first_tot_verts,
                    #     'unopt_qtar_verts': first_tot_qtars_verts,
                    #     'opt_verts': last_tot_verts,
                    #     'opt_qtar_verts': last_tot_qtars_verts, 
                    # }
                    optimized_pts_info_fn = "optimization_info.npy"
                    optimized_pts_info_fn = os.path.join(cur_full_folder, optimized_pts_info_fn)
                    if not os.path.exists(optimized_pts_info_fn):
                        continue
                    # cur_optimized_pts_info = np.load(optimized_pts_info_fn, allow_pickle=True).item()
                    # unopt_verts = cur_optimized_pts_info['unopt_verts']
                    # unopt_qtar_verts = cur_optimized_pts_info['unopt_qtar_verts'] # 
                    # opt_verts = cur_optimized_pts_info['opt_verts']
                    # opt_qtar_verts = cur_optimized_pts_info['opt_qtar_verts'] # opt qtars verts 
                    self.data_list.append(optimized_pts_info_fn) ## data list ## data list ##
            else:
                file_list = cur_statistics_info['file_list']
                self.data_list = file_list
                
                inheriting_info = cur_statistics_info['inheriting_info']
                self.tot_inheriting_infos.append(inheriting_info)
                
        else:
            ## root_data_folder ##
            self.data_list = []
            self.valid_subfolders = []
            
            # if self.all_clips_inst:
            self.data_inst_fn = f"save_info_v6.npy"
            print(f"data_inst_fn changed to {self.data_inst_fn} with all_clips_inst: {self.all_clips_inst}")
            
            ### exp tag ###
            for exp_tag in exp_tags:
                cur_data_folder = os.path.join(self.data_folder, exp_tag)
                tmp_data_list = os.listdir(cur_data_folder)
                
                print(f"specified_test_subfolder: {self.specified_test_subfolder}, full_specified_test_subfolder: {os.path.join(cur_data_folder, self.specified_test_subfolder)}")
                
                if self.specified_test_subfolder is not None and len(self.specified_test_subfolder) > 0 and os.path.exists(os.path.join(cur_data_folder, self.specified_test_subfolder)):
                    print(f"[here] specified_test_subfolder: {self.specified_test_subfolder}")
                    tmp_data_list = [self.specified_test_subfolder]
                
                for cur_subfolder in tmp_data_list: # getting data ## specified test fn ##
                    
                    if self.specified_hand_type is not None:
                        if self.specified_hand_type == 'allegro_flat_fivefin_yscaled_finscaled':
                            if self.specified_hand_type not in cur_subfolder:
                                continue
                        elif self.specified_hand_type == 'allegro_flat_fivefin_yscaled':
                            if self.specified_hand_type not in cur_subfolder or 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder:
                                continue
                        elif self.specified_hand_type == 'allegro_flat_fivefin':
                            if self.specified_hand_type not in cur_subfolder or 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder or 'allegro_flat_fivefin_yscaled' in cur_subfolder:
                                continue
                        elif self.specified_hand_type == 'allegro':
                            if 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder or 'allegro_flat_fivefin_yscaled' in cur_subfolder or 'allegro_flat_fivefin' in cur_subfolder:
                                continue
                        else:
                            raise ValueError(f"Unrecognized specified_hand_type: {self.specified_hand_type}")
                    
                    if self.specified_object_type is not None:
                        if f"objtype_{self.specified_object_type}" not in cur_subfolder:
                            continue
                    
                    # cur_subfolder_angle = cur_subfolder.split("_")
                    ####### ====== for positiive angles ====== #######
                    # for i_split in range(len(cur_subfolder_angle)):
                    #     if cur_subfolder_angle[i_split] == "objgoalrot":
                    #         cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                    #         break
                    # if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                    #     continue
                    ####### ====== for positiive angles ====== #######
                    
                    
                    inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                    if os.path.isdir(inst_folder):
                        ####### ======= get the instance statistics info fn ====== ######
                        save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
                        save_cur_inst_statistics_info_fn = os.path.join(inst_folder, save_cur_inst_statistics_info_fn)

                        if not os.path.exists(save_cur_inst_statistics_info_fn):
                            continue
                        
                        cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                        if os.path.exists(cur_inst_file):
                            
                            self.data_list.append(cur_inst_file)
                            self.valid_subfolders.append(cur_subfolder)
            
            
            if (not self.single_inst) and self.specified_statistics_info_fn is not None and len(self.specified_statistics_info_fn) > 0 and os.path.exists(self.specified_statistics_info_fn):
                valid_data_list_sv_fn = self.specified_statistics_info_fn
            else:
                ####### ======= get the single inst ======= #######
                if self.single_inst:
                    valid_data_list_sv_fn = "save_info_v6_statistics.npy"
                    valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], self.valid_subfolders[0], valid_data_list_sv_fn)
                else:
                    valid_data_list_sv_fn = "save_info_v6_statistics.npy"
                    # valid_data_list_sv_fn = "/cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy"
                    valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
                
                
                ####### ====== only use the allegro statistics ====== #######
                valid_data_list_sv_fn = "/cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy"
                ####### ====== only use the allegro statistics ====== #######
            
            print(f"valid_data_list_sv_fn: {valid_data_list_sv_fn}")
            valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        
        
        
        ### NOTE: not in use yet ###
        if self.diff_task_translations:
            self.data_list = [
                cur_inherit_info['to_task_pts_info_fn'] for cur_inherit_info in self.tot_inheriting_infos
            ]
        
        ### NOTE: basically not in use yet ###
        if valid_data_statistics is not None:
            self.avg_verts_tot_cases_tot_ts = valid_data_statistics['avg_verts']
            self.std_verts_tot_cases_tot_ts = valid_data_statistics['std_verts']
            self.avg_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['avg_qtar_verts']
            self.std_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['std_qtar_verts']
            self.avg_verts_qdd_tau_tot_cases_tot_ts_s2 = valid_data_statistics['avg_qtar_verts_s2']
            self.std_verts_qdd_tau_tot_cases_tot_ts_s2 = valid_data_statistics['std_qtar_verts_s2']
        
        # initialize the data statistics #
        self.data_statistics = {}
        
        self.maxx_nn_pts = 512
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        self.data_name_to_kine_info = {}
        
        self.tot_data_dict_list = []
        
        if not self.hybrid_dataset:
            self.maxx_kine_nn_ts = 150 
        else:
            self.maxx_kine_nn_ts = 1000
            
        ### TODO: statistics info loading part for tracking_ctl_diff is not correct -- fix it! ###
        
        if self.kine_diff:
            # preload the data #
            self._preload_kine_data()
            
            if self.task_cond:
                self._preload_kine_taskcond_data()
            
            if self.sampling and len(self.target_grab_inst_tag) > 0 :
                
                self.data_statistics = np.load(self.data_statistics_info_fn, allow_pickle=True).item()
                # self.data_statistics = {
                #     'avg_obj_transl': self.avg_obj_transl, 
                #     'std_obj_transl': self.std_obj_transl,
                #     'avg_obj_rot_euler': self.avg_obj_rot_euler,
                #     'std_obj_rot_euler': self.std_obj_rot_euler,
                #     'avg_obj_verts': self.avg_object_verts,
                #     'std_obj_verts': self.std_object_verts,
                # }
                self.avg_obj_transl = self.data_statistics['avg_obj_transl']
                self.std_obj_transl = self.data_statistics['std_obj_transl']
                self.avg_obj_rot_euler = self.data_statistics['avg_obj_rot_euler']
                self.std_obj_rot_euler = self.data_statistics['std_obj_rot_euler']
                self.avg_object_verts = self.data_statistics['avg_obj_verts']
                self.std_object_verts = self.data_statistics['std_obj_verts']
                self.avg_hand_qs = self.data_statistics['avg_hand_qs']
                self.std_hand_qs = self.data_statistics['std_hand_qs']
                
                pass
            
            
            if not self.sampling and not self.debug:
                self._save_kine_data_statistics()
        
        # preload     
        ### load and slice the tracking ctl data from the pre-saved fns ###
        elif self.tracking_ctl_diff:
            ### self tracking ctl diff ###
            if self.sampling and self.target_grab_inst_tag is not None and len(self.target_grab_inst_tag) > 0:
                ## target grab inst tag ##
                self.step_size = self.slicing_ws // 2
                data_statistics_info_fn = f'./assets/data_statistics_ws_{self.slicing_ws}_step_{self.step_size}.npy' 
                
                print(f"==> Loading data statistics info from {data_statistics_info_fn}") # load the data statisitcs infos #
                
                data_statistics_info = np.load(data_statistics_info_fn, allow_pickle=True).item()
                print(f"data_statistics_info: {data_statistics_info.keys()}")
                # dict_keys(['avg_hand_qs', 'std_hand_qs', 'avg_hand_qtars', 'std_hand_qtars', 'avg_obj_verts', 'std_obj_verts', 'avg_kine_hand_qs', 'std_kine_hand_qs', 'avg_obj_transl', 'std_obj_transl', 'avg_obj_rot_euler', 'std_obj_rot_euler'])
                self.avg_hand_qs = data_statistics_info['avg_hand_qs']
                self.std_hand_qs = data_statistics_info['std_hand_qs']
                self.avg_hand_qtars = data_statistics_info['avg_hand_qtars']
                self.std_hand_qtars = data_statistics_info['std_hand_qtars']
                
                self.avg_kine_hand_qs = data_statistics_info['avg_kine_hand_qs']
                self.std_kine_hand_qs = data_statistics_info['std_kine_hand_qs']
                self.avg_obj_transl = data_statistics_info['avg_obj_transl']
                self.std_obj_transl = data_statistics_info['std_obj_transl']
                self.avg_obj_rot_euler = data_statistics_info['avg_obj_rot_euler']
                self.std_obj_rot_euler = data_statistics_info['std_obj_rot_euler']
                
                self.avg_object_verts = data_statistics_info['avg_obj_verts']
                self.std_object_verts = data_statistics_info['std_obj_verts']
                
            else:
                
                self._preload_mocap_tracking_ctl_data()
                ## NOTE: task_cond -- we need the kinematics data as the task condition ##
                ## NOTE: slicing data -- we need the kinematics data frames to canonicalize the data ##
                if self.task_cond or self.slicing_data:
                    self._load_tracking_kine_info()
                if self.slicing_data:
                    self._slicing_mocap_tracking_ctl_data()
                    self._slice_tracking_kine_data()
        
            if not self.sampling and not self.debug:
                ### Save statistics ###
                self._save_data_statistics()
        
        
        elif self.diff_task_translations: 
            self._preload_inheriting_data()
        else:
            self._preload_data()
        
        self.data_name_to_statistics = {}
        
        
        if self.sampling and self.kine_diff and self.target_grab_inst_tag is not None and len(self.target_grab_inst_tag) > 0:
            # load the target kinematics task conditional data #
            self._preload_kine_target_taskcond_data()
        
        
        ## NOTE: to support out-of-training-set test ## # and len(self.target_grab_inst_tag) > 0 ? #
        elif self.target_grab_inst_tag is not None and len(self.target_grab_inst_tag) > 0 and self.target_grab_data_nm is None:
            
            self.target_grab_data_nm = self.target_grab_inst_opt_fn
            
            if self.slicing_data: ## load data and slice data if we've ste the slicing data to true #
                # cur data dict #
                cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(self.target_grab_inst_opt_fn, add_to_dict=False)
                # if self.task_cond_type == 'history_future':
                #     tot_obj_pose = cur_clip_data['tot_obj_pose']
                kine_info_dict = self._load_single_tracking_kine_info(self.target_grab_inst_tag)
                self._slicing_single_mocap_tracking_ctl_data(self.target_grab_inst_opt_fn, cur_clip_data, kine_info_dict, add_to_dict=True)
            else:
                cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(self.target_grab_inst_opt_fn)
                kine_info_dict = self._load_single_tracking_kine_info(self.target_grab_inst_tag)
                self.data_name_to_data[self.target_grab_inst_opt_fn] = cur_clip_data
                
                self.data_name_to_kine_info[self.target_grab_inst_opt_fn] = kine_info_dict
                self.data_name_list.append(self.target_grab_inst_opt_fn)
            
            # cur clip data, hand qs np # # add object #
            # cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(target_data_fn) 
        elif self.target_grab_inst_tag is not None and len(self.target_grab_inst_tag) > 0 and self.target_grab_data_nm is not None:
            self.target_grab_data_nm = self.target_grab_inst_opt_fn
            if self.slicing_data:
                cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(self.target_grab_inst_opt_fn, add_to_dict=False)
                kine_info_dict = self._load_single_tracking_kine_info(self.target_grab_inst_tag)
                self._slicing_single_mocap_tracking_ctl_data(self.target_grab_inst_opt_fn, cur_clip_data, kine_info_dict, add_to_dict=False)
        

    
    
    def __len__(self): # 
        
        ### the lenght of the target ###
        if self.sampling and self.target_grab_data_nm is not None and self.slicing_data and len(self.target_grab_inst_tag) > 0:
            return len(self.tot_target_data_nm)
                # tot_target_data_nm #
        
        if self.kine_diff and self.task_cond:
            return len(self.tot_data_dict_list)
        else:
            return len(self.data_name_to_data)
        # return len(self.data_name_list) 
    
    
    def _save_kine_data_statistics(self, ):
        # data_statistics
        if len(self.exp_additional_tag) ==0:
            data_stats_sv_fn = f"data_statistics_kinematics_diff.npy"
        else:
            data_stats_sv_fn = f"data_statistics_kinematics_diff_{self.exp_additional_tag}.npy"
        data_stats_sv_fn = os.path.join(f"assets", data_stats_sv_fn)
        np.save(data_stats_sv_fn, self.data_statistics)
        print(f"[Kine Diff] Data statistics saved to {data_stats_sv_fn}")
    
    def _save_data_statistics(self, ):
        ##### data stats sv fn #####
        if len(self.exp_additional_tag) == 0:
            data_stats_sv_fn = f"data_statistics_ws_{self.slicing_ws}_step_{self.step_size}.npy"
        else:
            # exp_additional_tag
            data_stats_sv_fn = f"data_statistics_ws_{self.slicing_ws}_step_{self.step_size}_{self.exp_additional_tag}.npy"
        
        data_stats_sv_fn = os.path.join(f"assets", data_stats_sv_fn)
        np.save(data_stats_sv_fn, self.data_statistics)
        print(f"Data statistics saved to {data_stats_sv_fn}") 
        
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ## ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    # ''' Shadow '''
    def load_target_data(self, target_data_fn):
        
        eps = 1e-6
        target_pts = np.load(target_data_fn, allow_pickle=True).item()
        target_pts = target_pts['ts_to_optimized_pts_integrated_w_tau'] ## get the optimized pts #
        first_frame_pts = target_pts[0] # nn_pts x 3
        # tot_verts = 
        first_frame_pts = first_frame_pts[:, None]
        target_pts = np.repeat(first_frame_pts, self.nn_seq_len, axis=1) 
        ### scale the data ### 
        target_pts = (target_pts - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
        # particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
        # target_pts: nn_pts x nn_frames x 3 #
        data_E = np.concatenate(
            [target_pts, target_pts], axis=-1 ## get the target features
        )
        ## TODO: copy to the relative assets folder ##
        sampled_pts_idxes_fn = "/root/diffsim/tiny-differentiable-simulator/python/examples/assets/shadow_sampled_pts_idxes.npy"
        sampled_idxes = np.load(sampled_pts_idxes_fn) ## the int32 array ## 
        data_dict ={
            'X': target_pts,
            'E': data_E,
            'sampled_idxes': sampled_idxes
        }
        self.target_data_dict = data_dict
        return data_dict
    
    
    def _load_single_tracking_kine_info(self, data_inst_tag):
        if isinstance(data_inst_tag, str):
            
            kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
            kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
            # get he kinemati info file # # load #  # load from that # 
            cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
            
            hand_qs = cur_kine_data['robot_delta_states_weights_np']
            maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            hand_qs = hand_qs[:maxx_ws]
            
            obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
            
            # then segment the data_inst_tag to get the mesh file name #
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{data_inst_tag}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            while len(to_sample_fr_idxes) < self.maxx_nn_pts:
                to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
        elif isinstance(data_inst_tag, tuple): # obj
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
                obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws] # obj verts #
            elif 'taco' in obj_type:
                #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy
                # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
                traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_{self.taco_interped_data_sv_additional_tag}.npy'
                taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                traj_kine_info = os.path.join(taco_kine_sv_root, traj_kine_info) # get hejkineinfo s 
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
                hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
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
        
        
        kine_obj_rot_euler_angles = []
        for i_fr in range(obj_ornt.shape[0]):
            cur_rot_quat = obj_ornt[i_fr] # dkine obj rot quat #
            cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
            kine_obj_rot_euler_angles.append(cur_rot_euler)
        kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
        kine_info_dict = {
            'obj_verts': obj_verts, 
            'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
            'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
            'obj_ornt': obj_ornt ,
            'obj_rot_euler': kine_obj_rot_euler_angles
        }
        return kine_info_dict
    
    
    ### data name to kine info ###
    def _load_tracking_kine_info(self,):
        # self.maxx_kine_nn_ts = 300
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        ## maxx kine nn ts setting ##
        for i_inst, data_inst_tag in enumerate(self.data_inst_tag_list):
            print(f"[Loading tracking kine info] {i_inst}/{len(self.data_inst_tag_list)}: {data_inst_tag}")
            kine_info_dict = self._load_single_tracking_kine_info(data_inst_tag)
            # if isinstance(data_inst_tag, str):
            #     kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
            #     kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
            #     # get he kinemati info file # # load #  # load from that # 
            #     cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
                
            #     hand_qs = cur_kine_data['robot_delta_states_weights_np']
            #     maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            #     hand_qs = hand_qs[:maxx_ws]
                
            #     obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            #     obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
            #     # then segment the data_inst_tag to get the mesh file name #
            #     self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            #     grab_mesh_fn = f"{data_inst_tag}.obj"
            #     grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
            #     # get the object mesh #
            #     obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            #     obj_verts = obj_mesh.vertices # nn_pts x 3
            #     # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            #     to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            #     while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            #         to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            #     random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            #     random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            #     obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            # elif isinstance(data_inst_tag, tuple): # obj
            #     obj_type, traj_obj_type = data_inst_tag
            #     traj_kine_info  = f"{self.tracking_info_st_tag}{traj_obj_type}.npy"
            #     traj_kine_info = os.path.join(self.tracking_save_info_fn, traj_kine_info)
            #     traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
            #     hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
            #     maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            #     hand_qs = hand_qs[:maxx_ws]
                
            #     obj_trans = traj_kine_info_data['object_transl'][:maxx_ws] # nn_ts x 3 
            #     obj_ornt = traj_kine_info_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
            #     # then segment the data_inst_tag to get the mesh file name #
            #     self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            #     grab_mesh_fn = f"{obj_type}.obj"
            #     grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
            #     # get the object mesh #
            #     obj_mesh = trimesh.load(grab_mesh_fn, force='mesh')
            #     obj_verts = obj_mesh.vertices # nn_pts x 3
            #     # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            #     # random_sampled_idxes = np.random.sample()
            #     to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            #     while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            #         to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            #     random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            #     random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            #     obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws] # obj verts #
            #     # grab_mesh_fn = f"{data_inst_tag}.obj"
            #     # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            # else: #
            #     raise ValueError(f"Unrecognized grab_data_inst_type: {type(data_inst_tag)}")
                
            # # hand_qs = cur_kine_data['robot_delta_states_weights_np'][:maxx_ws] # nn_ts x 22
            
            # kine_obj_rot_euler_angles = []
            # for i_fr in range(obj_ornt.shape[0]):
            #     cur_rot_quat = obj_ornt[i_fr] # dkine obj rot quat #
            #     cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True) # get the rotation euler 
            #     kine_obj_rot_euler_angles.append(cur_rot_euler)
            # kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            
            # the task conditional settings #
            self.data_name_to_kine_info[self.data_list[i_inst]] = kine_info_dict
            # { # data list to the obj verts and the ahand qs # 
            #     'obj_verts': obj_verts, 
            #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
            #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
            #     'obj_ornt': obj_ornt ,
            #     'obj_rot_euler': kine_obj_rot_euler_angles
            # }
            obj_trans, kine_obj_rot_euler_angles, hand_qs, obj_verts = kine_info_dict['obj_trans'], kine_info_dict['obj_rot_euler'], kine_info_dict['hand_qs'], kine_info_dict['obj_verts']

            tot_obj_transl.append(obj_trans)
            tot_obj_rot_euler.append(kine_obj_rot_euler_angles)
            tot_hand_qs.append(hand_qs)
            tot_object_verts.append(obj_verts)
        
        
        tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
        self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
        # avg hand qs and the std hand qs #?
        ## TODO: for the kinematics target data --- we should save them using a differnet name? #
        # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        # avg kine hand qs #
        self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        
        
        # 
        self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        
        # avg obj verts and the kine hand qs and #
        self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        self.data_statistics['std_obj_verts'] = self.std_object_verts
        self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        self.data_statistics['std_obj_transl'] = self.std_obj_transl
        self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
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
        
    
    # preload single tracking data #
    # cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(target_data_fn)
    def _preload_single_tracking_ctl_data(self, data_fn, add_to_dict=True):
        
        print(f"loading from {data_fn}")
        cur_data = np.load(data_fn, allow_pickle=True).item()
        if self.use_jointspace_seq:
            if self.sim_platform == 'isaac':
                optimized_obj_pose = cur_data['optimized_obj_pose']
                optimized_hand_qtars = cur_data['optimized_hand_qtars']
                optimized_hand_qs = cur_data['optimized_hand_qs']
                # # TODO: use top-k other than using the best evaluated res ? # 
                hand_qs_np = optimized_hand_qs[0] 
                hand_qtars_np = optimized_hand_qtars[0] # 
                
                if self.glb_rot_use_quat:
                    hand_glb_rot_np = hand_qs_np[..., 3:6]
                    hand_glb_qtar_np = hand_qtars_np[..., 3:6]
                    hand_glb_rot_th = torch.from_numpy(hand_glb_rot_np)
                    hand_glb_tar_rot_th = torch.from_numpy(hand_glb_qtar_np)
                    hand_glb_quat_th = quat_from_euler_xyz(hand_glb_rot_th[..., 0], hand_glb_rot_th[..., 1], hand_glb_rot_th[..., 2])
                    hand_glb_tar_quat_th = quat_from_euler_xyz(hand_glb_tar_rot_th[..., 0], hand_glb_tar_rot_th[..., 1], hand_glb_tar_rot_th[..., 2])
                    hand_glb_rot_np = hand_glb_quat_th.numpy()
                    hand_glb_qtar_np = hand_glb_tar_quat_th.numpy()
                    
                    hand_qs_np = np.concatenate(
                        [ hand_qs_np[..., :3], hand_glb_rot_np, hand_qs_np[..., 6:] ], axis=-1
                    )
                    hand_qtars_np = np.concatenate(
                        [ hand_qtars_np[..., :3], hand_glb_qtar_np, hand_qtars_np[..., 6:] ], axis=-1
                    )
                    # hand_qs_np[..., 3:6] = hand_glb_rot_np
                    # hand_qtars_np[..., 3:6] = hand_glb_qtar_np
                # obj_pose_np = cu
            else:
                ts_to_hand_qs = cur_data['ts_to_hand_qs'] 
                ts_to_hand_qtars = cur_data['ts_to_qtars']
                if self.slicing_data:
                    sorted_ts = sorted(list(ts_to_hand_qs.keys()))
                    hand_qs_np = [
                        ts_to_hand_qs[i_ts] for i_ts in sorted_ts
                    ]
                    hand_qtars_np = [
                        ts_to_hand_qtars[i_ts] for i_ts in sorted_ts
                    ]
                else:
                    if 'ts_to_optimized_q_tars_wcontrolfreq' in cur_data:
                        ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
                    else:
                        ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_hand_tot_ctl_qtars']
                    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                        
                    max_hand_qs_kd = max(list(ts_to_hand_qs.keys()))
                    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
                    ctl_freq_tss = sorted(ctl_freq_tss) # ctl freq tss #
                    ctl_freq = 10 # 
                    ctl_freq_tss_expanded = [ min(max_hand_qs_kd, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
                    ts_to_hand_qs = {
                        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
                    }
                    hand_qs_np = [
                        ts_to_hand_qs[cur_ts] for cur_ts in ctl_freq_tss
                    ]
                    hand_qtars_np = [ # 
                        ts_to_hand_qtars[cur_ts] for cur_ts in ctl_freq_tss
                    ]
                hand_qs_np = np.stack(hand_qs_np, axis=0)
                hand_qtars_np = np.stack(hand_qtars_np, axis=0) 
                
            # cur_data_nm = data_nm #
            
            # tot_data_hand_qs.append(hand_qs_np)
            # tot_data_hand_qtars.append(hand_qtars_np)

            cur_clip_data = {
                'tot_verts': hand_qs_np[None],  # hand qtars np; tot verts # 
                # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
                'tot_verts_integrated_qdd_tau': hand_qtars_np[None], # 
                # 'task_setting': task_setting, # rotation euler angles? # 
                # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            } # grab inst tag #
            
            if self.task_cond_type == 'history_future':
                obj_pose_np = optimized_obj_pose[0]
                cur_clip_data.update(
                    {
                        'tot_obj_pose': obj_pose_np[None]
                    }
                )
                
            
            # cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
            # using the data fn as the data name # 
            if add_to_dict:
                self.data_name_to_data[data_fn] = cur_clip_data
        else:
            raise NotImplementedError(f"Unrecognized use_jointspace_seq: {self.use_jointspace_seq}")
        return cur_clip_data, hand_qs_np, hand_qtars_np
        
    
    def _preload_mocap_tracking_ctl_data(self,): # 
        # self.data_list #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        if self.single_inst: # 
            self.data_list = self.data_list[:1]
            self.data_inst_tag_list = self.data_inst_tag_list[:1]
        elif self.multi_inst:
            self.data_list = self.data_list[:10]
            self.data_inst_tag_list = self.data_inst_tag_list[:10]
        # tot_expanded_passive #
        forbid_data_inst_tags = ["ori_grab_s2_phone_call_1", "ori_grab_s2_phone_pass_1"]
        
        for i_data_inst, data_fn in enumerate(self.data_list):
            
            excluded = False 
            for cur_forbid_inst_tag in forbid_data_inst_tags:
                if cur_forbid_inst_tag in data_fn:
                    excluded = True
                    break
            if excluded: ## # excluded ##
                continue
            
            print(f"loading from {data_fn}")
            # preload the tracking #
            cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(data_fn)
            tot_data_hand_qs.append(hand_qs_np)
            tot_data_hand_qtars.append(hand_qtars_np)
            self.data_name_to_data[data_fn] = cur_clip_data
            # cur_data = np.load(data_fn, allow_pickle=True).item()
            
            # if self.use_jointspace_seq:
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
                    
            #     # cur_data_nm = data_nm
                
            #     tot_data_hand_qs.append(hand_qs_np)
            #     tot_data_hand_qtars.append(hand_qtars_np)
                

            #     cur_clip_data = {
            #         'tot_verts': hand_qs_np[None],  # hand qtars np; tot verts # 
            #         # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
            #         'tot_verts_integrated_qdd_tau': hand_qtars_np[None], # 
            #         # 'task_setting': task_setting, # rotation euler angles? # 
            #         # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            #     } # grab inst tag #
            #     cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
            #     # using the data fn as the data name # 
            #     self.data_name_to_data[data_fn] = cur_clip_data
            # else:
            #     raise NotImplementedError(f"Unrecognized use_jointspace_seq: {self.use_jointspace_seq}")
        
        tot_data_hand_qs = np.concatenate(tot_data_hand_qs, axis=0)
        tot_data_hand_qtars = np.concatenate(tot_data_hand_qtars, axis=0)
        self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
        self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
        self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
        self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
        
        ### add to data statistics ###
        ### TODO: we should put object stats here --- obj stats should be calculated from the tracked trajectories ###
        self.data_statistics['avg_hand_qs'] = self.avg_hand_qs
        self.data_statistics['std_hand_qs'] = self.std_hand_qs
        self.data_statistics['avg_hand_qtars'] = self.avg_hand_qtars
        self.data_statistics['std_hand_qtars'] = self.std_hand_qtars
        
    
    
    def _get_obj_slicing_rot_trans(self, data_nm, st_idx, ed_idx):
        
        if self.task_cond_type == 'history_future':
            obj_pose = self.data_name_to_data[data_nm]['tot_obj_pose'][0]
            #  get the kine obj pose states # history future #
            
            # sliced_obj_pose = obj_pose[st_idx: ed_idx]
            sliced_obj_pose = obj_pose[st_idx - 1: ed_idx - 1 ]
            sliced_obj_trans, sliced_obj_ornt = sliced_obj_pose[:, :3], sliced_obj_pose[:, 3:]
            
            if self.use_kine_obj_pos_canonicalization:
                kine_info_dict = self.data_name_to_kine_info[data_nm]
                obj_trans = kine_info_dict['obj_trans']
                obj_ornt = kine_info_dict['obj_ornt']
                sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx - 1: ed_idx - 1], obj_ornt[st_idx - 1: ed_idx - 1]
                
        else: # sliced obj trans and the ornt #
            kine_info_dict = self.data_name_to_kine_info[data_nm]
            obj_trans = kine_info_dict['obj_trans']
            obj_ornt = kine_info_dict['obj_ornt']
            sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx: ed_idx], obj_ornt[st_idx: ed_idx]
        return sliced_obj_trans, sliced_obj_ornt
        # {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
        #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
        #     'obj_ornt': obj_ornt ,
        #     'obj_rot_euler': kine_obj_rot_euler_angles
        # }
    
    
    def _slicing_single_mocap_tracking_ctl_data(self, grab_opt_data_fn, cur_data_dict, kine_data_dict, add_to_dict=True):
        kine_qs = cur_data_dict['tot_verts'][0]
        q_tars = cur_data_dict['tot_verts_integrated_qdd_tau'][0]
        if self.task_cond_type == 'history_future':
            obj_pose = cur_data_dict['tot_obj_pose'][0]
        obj_trans = kine_data_dict['obj_trans']
        obj_ornt = kine_data_dict['obj_ornt']
        
        # slice # 
        print(f"kine_qs: {kine_qs.shape}, q_tars: {q_tars.shape}, obj_trans: {obj_trans.shape}, obj_ornt: {obj_ornt.shape}")
        
        slicing_st_idx = 0
        if self.task_cond_type == 'history_future':
            slicing_st_idx = 1
        
        self.tot_target_data_nm = []
        for i_slice in range(slicing_st_idx, kine_qs.shape[0] - self.slicing_ws, self.step_size):
            st_idx = i_slice # the start index #
            ed_idx = i_slice + self.slicing_ws
            
            slicing_idxes = list(range(st_idx, ed_idx))
            slicing_idxes = np.array(slicing_idxes, dtype=np.int32) # get the slicing idxes # 
            slicing_idxes = np.clip(slicing_idxes, a_min=0, a_max=kine_qs.shape[0] - 1)
            
            # ed_idx = min(i_slice + self.slicing_ws, kine_qs.shape[0])
            
            
            
            # sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx: ed_idx], obj_ornt[st_idx: ed_idx]
            sliced_obj_trans, sliced_obj_ornt = obj_trans[slicing_idxes], obj_ornt[slicing_idxes]
            
            # ge the slicing data obj trans #
            if self.task_cond_type == 'history_future':
                sliced_obj_pose = obj_pose[st_idx - 1: ed_idx - 1 ]
                first_frame_obj_trans = sliced_obj_pose[0, :3]
                first_frame_obj_ornt = sliced_obj_pose[0, 3:]
            else:
                first_frame_obj_trans = sliced_obj_trans[0, :3] # first farme obj trans #
            
            # cur_slice_kine_qs = kine_qs[st_idx: ed_idx]
            # cur_slice_q_tars = q_tars[st_idx: ed_idx]
            
            cur_slice_kine_qs = kine_qs[slicing_idxes]
            cur_slice_q_tars = q_tars[slicing_idxes]
            
            #### NOTE: A simple canonicalization ####
            cur_slice_kine_qs[:, :3] = cur_slice_kine_qs[:, :3] - first_frame_obj_trans[ None]
            cur_slice_q_tars[:, :3] = cur_slice_q_tars[:, :3] - first_frame_obj_trans[ None]
            #### NOTE: A simple canonicalization ####

            cur_slice_data = {
                'tot_verts': cur_slice_kine_qs[None],
                'tot_verts_integrated_qdd_tau': cur_slice_q_tars[None]
            }
            
            
            # TODO: change all the jugdement logic to this # 
            # 
            if  self.task_cond_type == 'history_future': # history future #
                history_st_idx = st_idx - self.slicing_ws
                # history_st_idx = max(0, history_st_idx)
                # history_ed_idx = st_idx # + 1
                # history_ed_idx = min(kine_qs.shape[0], history_ed_idx)
                # history_st_idx = max(0, history_st_idx)
                history_ed_idx = st_idx # + 1
                # history_ed_idx = min(kine_qs.shape[0], history_ed_idx) # hsitory idxes #
                history_idxes = list(range(history_st_idx, history_ed_idx))
                history_idxes = np.array(history_idxes, dtype=np.int32)
                history_idxes = np.clip(history_idxes, a_min=0, a_max=kine_qs.shape[0] - 1)
                history_kine_qs = kine_qs[history_idxes]
                tot_obj_pose = cur_data_dict['tot_obj_pose'][0]
                history_obj_pose = tot_obj_pose[history_idxes]
                history_obj_pose[:, :3] = history_obj_pose[:, :3] - first_frame_obj_trans[ None]
                history_kine_qs[:, :3] = history_kine_qs[:, :3] - first_frame_obj_trans[ None] # minus the first frame obj trans #
                
                ## the obj eulers may not be a good representation ## ## a representation ## ## # a representation #
                ## TODO: the obj eulers may not be a good representation ## a good representation ##
                # ### from the obj quaternion ornt to the obj euler rotations ##### rotations #####
                history_obj_trans, history_obj_ornt = history_obj_pose[:, :3], history_obj_pose[:, 3:]
                history_obj_rot_euler = []
                for ii_fr in range(history_obj_ornt.shape[0]):
                    cur_fr_obj_ornt = history_obj_ornt[ii_fr]
                    cur_fr_obj_rot_euler = R.from_quat(cur_fr_obj_ornt).as_euler('xyz', degrees=False) # as the rot eulers # 
                    history_obj_rot_euler.append(cur_fr_obj_rot_euler)
                history_obj_rot_euler = np.stack(history_obj_rot_euler, axis= 0) 
                
                # add the history information #
                history_info = {
                    'history_obj_pose': history_obj_pose[None ],
                    'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                    'first_frame_obj_trans': first_frame_obj_trans,
                    'first_frame_obj_ornt': first_frame_obj_ornt, # first frae obj ornt # an the trans # 
                    'history_obj_trans': history_obj_trans[None ], 
                    'history_obj_rot_euler': history_obj_rot_euler[None ],
                }
                cur_slice_data.update(history_info )
                # have he st_idx ? # 
            
            # cur_slice_data_nm = f"{cur_data_fn}_sted_{st_idx}_{ed_idx}"
            cur_slice_data_nm = grab_opt_data_fn.replace(".npy", f"_STED_{st_idx}_{ed_idx}.npy")
            
            if add_to_dict:
                self.data_name_to_data[cur_slice_data_nm] = cur_slice_data #  
            
            self.tot_target_data_nm.append(cur_slice_data_nm)
            if add_to_dict:
                self.data_name_list.append(cur_slice_data_nm)
            
            # slicing tracking kienmatics data ###
            # print(f"[Slicing tracking kine data] data_nm: {data_nm}")
            # sted_info = data_nm.split("/")[-1].split(".npy")[0].split('_STED_')[1]
            # st_idx, ed_idx = sted_info.split("_")
            # st_idx, ed_idx = int(st_idx), int(ed_idx)
            # # else:
            # #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0] # sliced hand qs #
            
            # sliced_hand_qs = kine_data_dict['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
            # sliced_obj_trans = kine_data_dict['obj_trans'][st_idx: ed_idx]
            # sliced_obj_ornt = kine_data_dict['obj_ornt'][st_idx: ed_idx]
            # sliced_obj_rot_euler = kine_data_dict['obj_rot_euler'][st_idx: ed_idx]
            
            # slicing_idxes
            
            sliced_hand_qs = kine_data_dict['hand_qs'][slicing_idxes][:, :self.nn_hands_dof]
            sliced_obj_trans = kine_data_dict['obj_trans'][slicing_idxes]
            sliced_obj_ornt = kine_data_dict['obj_ornt'][slicing_idxes]
            sliced_obj_rot_euler = kine_data_dict['obj_rot_euler'][slicing_idxes]
            
            obj_verts = kine_data_dict['obj_verts']
            
            first_frame_obj_trans = sliced_obj_trans[0, :3]
            sliced_hand_qs[:, :3] = sliced_hand_qs[:, :3] - first_frame_obj_trans[None]
            sliced_obj_trans = sliced_obj_trans - first_frame_obj_trans[None]
            
            kine_info_dict = {
                'obj_verts': obj_verts, 
                'hand_qs': sliced_hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
                'obj_trans': sliced_obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
                'obj_ornt': sliced_obj_ornt ,
                'obj_rot_euler': sliced_obj_rot_euler
            }
            if add_to_dict:
                self.data_name_to_kine_info[cur_slice_data_nm] = kine_info_dict
                
    
    # slice mocap ctl data #
    def _slicing_mocap_tracking_ctl_data(self, ):
        ## slice the tracking ctl data ## # tracking ctl # use_kine_obj_pos_canonicalization # kine obj pos canonicalization #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        self.all_slices_data_inst_tag_list = []
        self.all_slices_data_name_to_data = {}
        for cur_data_fn in self.data_name_to_data:
            cur_data = self.data_name_to_data[cur_data_fn]
            kine_qs =cur_data['tot_verts'][0]
            q_tars = cur_data['tot_verts_integrated_qdd_tau'][0]
            
            slicing_st_idx = 0 
            if self.task_cond_type == 'history_future':
                slicing_st_idx = 1
        
            slicing_ed_idx = kine_qs.shape[0] - self.slicing_ws
            if slicing_ed_idx < slicing_st_idx + 1:
                slicing_ed_idx = slicing_st_idx + 1 ## increase the slicing ed idx 
            # step size = 30
            # print(f"kine_qs: {kine_qs.shape}, q_tars: {q_tars.shape}")
            for i_slice in range(slicing_st_idx, slicing_ed_idx, self.step_size):
                st_idx = i_slice
                ed_idx = i_slice + self.slicing_ws
                slicing_idxes = list(range(st_idx, ed_idx))
                slicing_idxes = np.array(slicing_idxes, dtype=np.int32) # 
                slicing_idxes = np.clip(slicing_idxes, a_min=0, a_max=kine_qs.shape[0] - 1) #
                # ed_idx = min(i_slice + self.slicing_ws, kine_qs.shape[0])
                
                # task cond should not ## obj slicing rot trans #
                sliced_obj_trans, sliced_obj_ornt = self._get_obj_slicing_rot_trans(cur_data_fn, st_idx, ed_idx)
                
                if sliced_obj_trans.shape[0] == 0:
                    continue
                
                first_frame_obj_trans = sliced_obj_trans[0, :3]
                first_frame_obj_ornt = sliced_obj_ornt[0, :]
                
                # cur_slice_kine_qs = kine_qs[st_idx: ed_idx]
                # cur_slice_q_tars = q_tars[st_idx: ed_idx]
                
                cur_slice_kine_qs = kine_qs[slicing_idxes]
                cur_slice_q_tars = q_tars[slicing_idxes]
                
                #### NOTE: A simple canonicalization ####
                cur_slice_kine_qs[:, :3] = cur_slice_kine_qs[:, :3] - first_frame_obj_trans[ None]
                cur_slice_q_tars[:, :3] = cur_slice_q_tars[:, :3] - first_frame_obj_trans[ None]
                #### NOTE: A simple canonicalization ####
                
                cur_slice_data = {
                    'tot_verts': cur_slice_kine_qs[None],
                    'tot_verts_integrated_qdd_tau': cur_slice_q_tars[None],
                    'slicing_idxes': slicing_idxes,
                }
                
                # TODO: change all the jugdement logic to this # 
                if  self.task_cond_type == 'history_future':
                    # history_st_idx = st_idx - self.slicing_ws
                    history_st_idx = st_idx - self.history_ws
                    # history_st_idx = max(0, history_st_idx)
                    history_ed_idx = st_idx # + 1
                    # history_ed_idx = min(kine_qs.shape[0], history_ed_idx)
                    history_idxes = list(range(history_st_idx, history_ed_idx))
                    history_idxes = np.array(history_idxes, dtype=np.int32)
                    history_idxes = np.clip(history_idxes, a_min=0, a_max=kine_qs.shape[0] - 1) # 
                    # history_kine_qs = kine_qs[history_st_idx : history_ed_idx]
                    history_kine_qs = kine_qs[history_idxes]
                    tot_obj_pose = cur_data['tot_obj_pose'][0]
                    # history_obj_pose = tot_obj_pose[history_st_idx: history_ed_idx]
                    history_obj_pose = tot_obj_pose[history_idxes]
                    history_obj_pose[:, :3] = history_obj_pose[:, :3] - first_frame_obj_trans[ None]
                    history_kine_qs[:, :3] = history_kine_qs[:, :3] - first_frame_obj_trans[ None] # minus the first frame obj trans # # j
                    
                    ## the obj eulers may not be a good representation ## ## a representation ## ## # a representation #
                    ## TODO: the obj eulers may not be a good representation ## a good representation ##
                    # ### from the obj quaternion ornt to the obj euler rotations ##### rotations #####
                    history_obj_trans, history_obj_ornt = history_obj_pose[:, :3], history_obj_pose[:, 3:]
                    history_obj_rot_euler = []
                    for ii_fr in range(history_obj_ornt.shape[0]):
                        cur_fr_obj_ornt = history_obj_ornt[ii_fr]
                        cur_fr_obj_rot_euler = R.from_quat(cur_fr_obj_ornt).as_euler('xyz', degrees=False) # as the rot eulers # 
                        history_obj_rot_euler.append(cur_fr_obj_rot_euler)
                    history_obj_rot_euler = np.stack(history_obj_rot_euler, axis= 0) 
                    
                    # add the history information #
                    history_info = {
                        'history_obj_pose': history_obj_pose[None ],
                        'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                        'first_frame_obj_trans': first_frame_obj_trans,
                        'first_frame_obj_ornt': first_frame_obj_ornt, # first frae obj ornt # an the trans # 
                        'history_obj_trans': history_obj_trans[None ], 
                        'history_obj_rot_euler': history_obj_rot_euler[None ],
                        'history_idxes': history_idxes
                    }
                    cur_slice_data.update(history_info )
                    
                    
                
                # cur_slice_data_nm = f"{cur_data_fn}_sted_{st_idx}_{ed_idx}"
                cur_slice_data_nm = cur_data_fn.replace(".npy", f"_STED_{st_idx}_{ed_idx}.npy")
                self.all_slices_data_name_to_data[cur_slice_data_nm] = cur_slice_data #  
                
                self.all_slices_data_inst_tag_list.append(cur_slice_data_nm)
                tot_data_hand_qs.append(cur_slice_kine_qs)
                tot_data_hand_qtars.append(cur_slice_q_tars)

        tot_data_hand_qs = np.concatenate(tot_data_hand_qs, axis=0)
        tot_data_hand_qtars = np.concatenate(tot_data_hand_qtars, axis=0)
        self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
        self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
        self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
        self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
        
        ### add to data statistics ###
        self.data_statistics['avg_hand_qs'] = self.avg_hand_qs
        self.data_statistics['std_hand_qs'] = self.std_hand_qs
        self.data_statistics['avg_hand_qtars'] = self.avg_hand_qtars
        self.data_statistics['std_hand_qtars'] = self.std_hand_qtars
        
        
        self.data_name_list = self.all_slices_data_inst_tag_list
        self.data_name_to_data = self.all_slices_data_name_to_data
    
    
    
    def _preload_kine_taskcond_data(self, ):
        if self.single_inst:
            self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
        
        
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            kine_data_tag = "passive_active_info_"
            kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
            return kine_object_type
           
        
        def load_traj_info(kine_dict_fn, maxx_ws):
            kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            kine_obj_transl = kine_dict['object_transl']
            kine_obj_rot_quat = kine_dict['object_rot_quat']
            kine_robot_hand_qs = kine_dict['robot_delta_states_weights_np']
            kine_obj_transl = kine_obj_transl[: maxx_ws]
            kine_obj_rot_quat = kine_obj_rot_quat[: maxx_ws]
            kine_robot_hand_qs = kine_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            kine_obj_rot_euler_angles = []
            for i_fr in range(kine_obj_rot_quat.shape[0]):
                cur_rot_quat = kine_obj_rot_quat[i_fr] #
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
                kine_obj_rot_euler_angles.append(cur_rot_euler)
            kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            return kine_obj_transl, kine_obj_rot_euler_angles, kine_robot_hand_qs
            
        def load_obj_pcs(kine_dict_fn, nn_sampled_pts):
            object_type = parse_kine_data_fn_into_object_type(kine_dict_fn)
            # kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            # kine_obj_transl = k
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes][:nn_sampled_pts]
            # tot_object_verts.append(obj_verts)/
            return obj_verts
        
        # task inheriting dict info ## inheritng dict info #
        maxx_ws = 149
        for i_data, data_dict in enumerate(self.task_inheriting_dict_info):
            # cur_inheriting_dict = {
            #     'fa_objtype': cur_parent_objtype, # 
            #     'fa_trajtype': cur_parent_trajtype, # 
            #     'ch_objtype': cur_child_objtype, # 
            #     'ch_trajtype': cur_child_trajtype
            # }
            cur_fa_objtype = data_dict['fa_objtype']
            cur_fa_trajtype = data_dict['fa_trajtype']
            cur_ch_objtype = data_dict['ch_objtype']
            cur_ch_trajtype = data_dict['ch_trajtype']
            
            ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_trajtype]
            fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_trajtype]
            
            ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_objtype]
            fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_objtype]
            
            ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
            fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
            
            ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
            fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
            
            
            
            # 
            sv_dict = {
                'obj_verts': fa_obj_verts,
                'robot_hand_qs': fa_robot_hand_qs,
                'obj_rot_euler': fa_obj_rot_euler,
                'obj_transl': fa_obj_transl
            }
            cond_sv_dict = {
                'cond_obj_verts': ch_obj_verts,
                'cond_robot_hand_qs': ch_robot_hand_qs,
                'cond_obj_rot_euler': ch_obj_rot_euler,
                'cond_obj_transl': ch_obj_transl
            }
            sv_dict.update(cond_sv_dict)
            
            
            for key in sv_dict:
                print(f"key: {key}, val: {sv_dict[key].shape }")
            
            # self.data_name_to_data[cur_kine_data_fn] = sv_dict
            self.tot_data_dict_list.append(sv_dict)
            
            # object_type = parse_kine_data_fn_into_object_type(cur_kine_data_fn)
            # self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            # grab_mesh_fn = f"{object_type}.obj"
            # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            
            # obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            # obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            # obj_verts = obj_verts[random_sampled_idxes][:maxx_ws]
            # tot_object_verts.append(obj_verts)
            
            # ch_kine_traj_dict = np.load(ch_kine_traj_dict_fn, allow_pickle=True).item()
            # fa_kine_traj_dict = np.load(fa_kine_traj_dict_fn, allow_pickle=True).item()
            
            # ch_obj_transl = ch_kine_traj_dict['object_transl']
            # ch_obj_rot_quat = ch_kine_traj_dict['object_rot_quat']
            # ch_robot_hand_qs = ch_kine_traj_dict['robot_delta_states_weights_np']
            # maxx_ws = 150
            # ch_obj_transl = ch_obj_transl[: maxx_ws]
            # ch_obj_rot_quat = ch_obj_rot_quat[: maxx_ws]
            # ch_robot_hand_qs = ch_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            
            # ch_obj_rot_euler_angles = []
            # for i_fr in range(ch_obj_rot_quat.shape[0]):
            #     cur_rot_quat = ch_obj_rot_quat[i_fr]
            #     cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True) # get the rotation euler 
            #     ch_obj_rot_euler_angles.append(cur_rot_euler)
            # ch_obj_rot_euler_angles = np.stack(ch_obj_rot_euler_angles, axis=0)
    
    
    def _preload_kine_target_taskcond_data(self, ):
        # if self.single_inst:
        #     self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
        
        # tot_obj_transl = []
        # tot_obj_rot_euler = []
        # tot_hand_qs = []
        # tot_obj_verts = []
        
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            kine_data_tag = "passive_active_info_"
            kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
            return kine_object_type
           
        
        def load_traj_info(kine_dict_fn, maxx_ws):
            kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            kine_obj_transl = kine_dict['object_transl']
            kine_obj_rot_quat = kine_dict['object_rot_quat']
            kine_robot_hand_qs = kine_dict['robot_delta_states_weights_np']
            kine_obj_transl = kine_obj_transl[: maxx_ws]
            kine_obj_rot_quat = kine_obj_rot_quat[: maxx_ws]
            kine_robot_hand_qs = kine_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            kine_obj_rot_euler_angles = []
            for i_fr in range(kine_obj_rot_quat.shape[0]):
                cur_rot_quat = kine_obj_rot_quat[i_fr] # dkine obj rot quat #
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
                kine_obj_rot_euler_angles.append(cur_rot_euler)
            kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            return kine_obj_transl, kine_obj_rot_euler_angles, kine_robot_hand_qs
            
        def load_obj_pcs(kine_dict_fn, nn_sampled_pts):
            object_type = parse_kine_data_fn_into_object_type(kine_dict_fn)
            # kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            # kine_obj_transl = k
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes][:nn_sampled_pts]
            # tot_object_verts.append(obj_verts)/
            return obj_verts
        
        
        maxx_ws = 149
        
        ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        
        ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        
        ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
        fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
        
        ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
        fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
        
    # 
        sv_dict = {
            'obj_verts': fa_obj_verts,
            'robot_hand_qs': fa_robot_hand_qs,
            'obj_rot_euler': fa_obj_rot_euler,
            'obj_transl': fa_obj_transl
        }
        cond_sv_dict = {
            'cond_obj_verts': ch_obj_verts,
            'cond_robot_hand_qs': ch_robot_hand_qs,
            'cond_obj_rot_euler': ch_obj_rot_euler,
            'cond_obj_transl': ch_obj_transl
        }
        sv_dict.update(cond_sv_dict)
        
        
        self.tot_data_dict_list = []
        self.tot_data_dict_list.append(sv_dict)
        
        # for i_data, data_dict in enumerate(self.task_inheriting_dict_info):
        #     # cur_inheriting_dict = {
        #     #     'fa_objtype': cur_parent_objtype, # 
        #     #     'fa_trajtype': cur_parent_trajtype, # 
        #     #     'ch_objtype': cur_child_objtype, # 
        #     #     'ch_trajtype': cur_child_trajtype
        #     # }
        #     cur_fa_objtype = data_dict['fa_objtype']
        #     cur_fa_trajtype = data_dict['fa_trajtype']
        #     cur_ch_objtype = data_dict['ch_objtype']
        #     cur_ch_trajtype = data_dict['ch_trajtype']
            
        #     ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_trajtype]
        #     fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_trajtype]
            
        #     ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_objtype]
        #     fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_objtype]
            
        #     ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
        #     fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
            
        #     ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
        #     fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
            
            
            
        #     # 
        #     sv_dict = {
        #         'obj_verts': fa_obj_verts,
        #         'robot_hand_qs': fa_robot_hand_qs,
        #         'obj_rot_euler': fa_obj_rot_euler,
        #         'obj_transl': fa_obj_transl
        #     }
        #     cond_sv_dict = {
        #         'cond_obj_verts': ch_obj_verts,
        #         'cond_robot_hand_qs': ch_robot_hand_qs,
        #         'cond_obj_rot_euler': ch_obj_rot_euler,
        #         'cond_obj_transl': ch_obj_transl
        #     }
        #     sv_dict.update(cond_sv_dict)
            
            
        #     for key in sv_dict:
        #         print(f"key: {key}, val: {sv_dict[key].shape }")
            
        #     # self.data_name_to_data[cur_kine_data_fn] = sv_dict
        #     self.tot_data_dict_list.append(sv_dict)
            
    

    
    def _preload_kine_data(self,) : # 
        if self.single_inst or self.debug:
            self.data_list = self.data_list[:1]
       
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
           
        # tot_expanded_passve_meshes = [] # tot expanded passive meshes #
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        print(f"Start loading kinematics data: {len(self.data_list)}")
        for i_kine, kine_fn in enumerate(self.data_list):
            print(f"[{i_kine}/{len(self.data_list)}] {kine_fn}")
            # kine_saved_info: dict_keys(['passive_meshes', 'active_meshes', 'passive_mesh_normals', 'object_transl', 'object_rot_quat', 'ts_to_allegro', 'ts_to_mano_fingers_np', 'ts_to_robot_fingers_np', 'robot_delta_states_weights_np']) 
            cur_kine_data_fn = self.data_list[i_kine]
            cur_kine_data = np.load(cur_kine_data_fn, allow_pickle=True).item()
            
            obj_transl = cur_kine_data['object_transl']
            obj_rot_quat = cur_kine_data['object_rot_quat']
            robot_hand_qs = cur_kine_data['robot_delta_states_weights_np']
            
            maxx_ws = 150
            maxx_ws = min(maxx_ws, obj_transl.shape[0])
            maxx_ws = min(maxx_ws, robot_hand_qs.shape[0])
            obj_transl = obj_transl[: maxx_ws]
            obj_rot_quat = obj_rot_quat[: maxx_ws]
            robot_hand_qs = robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            # transform the rot_quat # 
            obj_rot_euler_angles = []
            for i_fr in range(obj_rot_quat.shape[0]):
                cur_rot_quat = obj_rot_quat[i_fr]
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False)
                obj_rot_euler_angles.append(cur_rot_euler)
            obj_rot_euler_angles = np.stack(obj_rot_euler_angles, axis=0)
            
            # obj_transl and obj_rot_euler_angles # 
            tot_obj_transl.append(obj_transl)
            tot_obj_rot_euler.append(obj_rot_euler_angles)
            tot_hand_qs.append(robot_hand_qs)
        
            object_type = parse_kine_data_fn_into_object_type(cur_kine_data_fn)
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            tot_object_verts.append(obj_verts)
            
            sv_dict = {
                'obj_verts': obj_verts,
                'robot_hand_qs': robot_hand_qs,
                'obj_rot_euler': obj_rot_euler_angles,
                'obj_transl': obj_transl,
                'object_type': object_type,
            }
            self.data_name_to_data[cur_kine_data_fn] = sv_dict
            # # --- using pose trajectories and encode them into the pose trajectories #
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
            # tot_expanded_passve_meshes.append(expanded_obj_verts)
            # # 
            # tot kine data fn #
        
        tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        
        if not (self.sampling and len(self.target_grab_inst_tag) > 0):
            self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
            self.std_obj_transl = np.std(tot_obj_transl, axis=0)
            self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
            self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
            self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
            self.std_hand_qs = np.std(tot_hand_qs, axis=0)
            # 
            self.avg_object_verts = np.mean(tot_object_verts, axis=0)
            self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
            
            
            self.data_statistics = {
                'avg_obj_transl': self.avg_obj_transl, 
                'std_obj_transl': self.std_obj_transl,
                'avg_obj_rot_euler': self.avg_obj_rot_euler,
                'std_obj_rot_euler': self.std_obj_rot_euler,
                'avg_obj_verts': self.avg_object_verts,
                'std_obj_verts': self.std_object_verts,
                'avg_hand_qs': self.avg_hand_qs, 
                'std_hand_qs': self.std_hand_qs
            }
        
        # tot_expanded_passve_meshes = np.concatenate(tot_expanded_passve_meshes, axis=0) # 
        # avg_obj_verts = np.mean(tot_expanded_passve_meshes, axis=0)
        # std_obj_verts = np.std(tot_expanded_passve_meshes, axis=0) 
        # self.avg_obj_verts = avg_obj_verts
        # self.std_obj_verts = std_obj_verts ## avg and std object verts ##
        # pass
    
    def _preload_inheriting_data(self, ):
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
            
        for cur_inherit_info in self.tot_inheriting_infos:
            cur_fa_task_fn = cur_inherit_info['inherit_fr_pts_info_fn']
            cur_ch_task_fn = cur_inherit_info['to_task_pts_info_fn']
            
            ## fa and ch ##
            cur_fa_data = self._load_data_from_data_name(cur_fa_task_fn)
            cur_ch_data = self._load_data_from_data_name(cur_ch_task_fn)
            
            cur_ch_task_setting = [
                float(self.obj_name_to_idx[cur_ch_data['object_type']]) - 0.5, float(cur_ch_data['task_rot']), float(cur_ch_data['object_size_x'])
            ]
            cur_fa_task_setting = [
                float(self.obj_name_to_idx[cur_fa_data['object_type']]) - 0.5, float(cur_fa_data['task_rot']), float(cur_fa_data['object_size_x'])
            ]
            cur_inheri_data = {
                'fa_task_setting': cur_fa_task_setting, 
                'ch_task_setting' : cur_ch_task_setting
            }
            self.data_name_to_data[cur_ch_task_fn] = cur_inheri_data
    
    def _preload_data(self, ):
        
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        
        self.tot_object_type = []
        self.tot_rot = []
        self.tot_obj_size = []
        for data_nm in self.data_name_list:
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            ### training setting ###
            if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond']:
                unopt_verts = cur_data['unopt_verts']
                tot_verts = unopt_verts
                unopt_qtar_verts = cur_data['unopt_qtar_verts']
                opt_verts = cur_data['opt_verts']
                opt_qtar_verts = cur_data['opt_qtar_verts']
                
                ## unopt verts ##
                cur_clip_data = {
                    'unopt_verts': unopt_verts,
                    'unopt_qtar_verts': unopt_qtar_verts,
                    'opt_verts': opt_verts,
                    'opt_qtar_verts': opt_qtar_verts
                }
                cur_data_nm = data_nm
                self.data_name_to_data[data_nm] = cur_clip_data
            
            
            else: 
                if self.use_jointspace_seq:
                    ts_to_hand_qs = cur_data['ts_to_hand_qs']
                    # ts_to_hand_qtars = cur_data['ts_to_qtars'] 
    
                    ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
                    # ts_to_
                    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                     
                    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
                    ctl_freq_tss = sorted(ctl_freq_tss)
                    ctl_freq = 10
                    ctl_freq_tss_expanded = [ min(500 - 1, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
                    ts_to_hand_qs = {
                        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
                    }
                    # nn_ts x 
                    # tot_qs = tot_qs[None]
                    ts_keys_sorted = sorted(list(ts_to_hand_qs.keys()))
                    hand_qs_np = [
                        ts_to_hand_qs[cur_ts] for cur_ts in ts_to_hand_qs
                    ]
                    hand_qtars_np = [
                        ts_to_hand_qtars[cur_ts] for cur_ts in ts_to_hand_qtars
                    ]
                    hand_qs_np = np.stack(hand_qs_np, axis=0)
                    hand_qtars_np = np.stack(hand_qtars_np, axis=0) ## tethte qtarsnp 
                    
                    # hand_qs_np = hand_qs_np[]
                    
                    cur_data_nm = data_nm
                    
                    task_setting = {
                        'object_type': self.obj_name_to_idx[cur_data['object_type']],
                        'task_rot': cur_data['task_rot'],
                        'object_size_x': cur_data['object_size_x']
                    }
                    
                    self.tot_object_type.append(task_setting['object_type'])
                    self.tot_rot.append(task_setting['task_rot'])
                    self.tot_obj_size.append(task_setting['object_size_x']) ## get object size x ##
                    
                    
                    cur_clip_data = {
                        'tot_verts': hand_qs_np[None], 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': hand_qtars_np[None],
                        'task_setting': task_setting
                        # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                
                else:
                    # selected_frame_verts, selected_frame_qtars_verts
                    # tot_verts tot_verts_integrated_qdd_tau
                    ## convert to the interested first frame's pose ## then chnage the pose of that data ##
                    
                    tot_verts = cur_data['tot_verts']
                    
                    if self.use_static_first_frame:
                        tot_verts_first_frame = tot_verts[:, 0:1]
                        tot_verts = np.repeat(tot_verts_first_frame, tot_verts.shape[1], axis=1)
                    # print(f"tot_verts: {}")
                    
                    tot_verts_integrated_qdd_tau = cur_data['tot_qtar_verts']
                    if 'tot_qtar_verts_s2' not in cur_data:
                        tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts'].copy()
                    else:
                        tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts_s2']
                    
                    
                    # nn_ts x nn_verts x 3 #
                    # sequence length ? # # tot verts # #
                    self.nn_seq_len = tot_verts.shape[1]
                    
                    ##### inspect ######
                    mean_tot_verts = np.mean(tot_verts, axis=1)
                    mean_tot_verts = np.mean(mean_tot_verts, axis=0)
                    
                    mean_tot_verts_qdd = np.mean(tot_verts_integrated_qdd_tau, axis=1)
                    mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
                    
                    mean_tot_verts_qdd_s2 = np.mean(tot_verts_integrated_qdd_tau_s2, axis=1)
                    mean_tot_verts_qdd_s2 = np.mean(mean_tot_verts_qdd_s2, axis=0)
                    
                    print(f"mean_tot_verts: {mean_tot_verts}, mean_tot_verts_qdd: {mean_tot_verts_qdd}, mean_tot_verts_qdd_s2: {mean_tot_verts_qdd_s2}")
                    ##### inspect ######
                    
                    cur_data_nm = data_nm
                    cur_clip_data = {
                        'tot_verts': tot_verts, 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': tot_verts_integrated_qdd_tau,
                        'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                
            
            ###### not use jointspace seq ######
            if not self.use_jointspace_seq:
                init_verts = tot_verts[:, 0]
                particle_init_xs_th = torch.from_numpy(init_verts).float()
                
                if self.specified_sampled_particle_idxes_fn is not None and len(self.specified_sampled_particle_idxes_fn) > 0:
                    sampled_particle_idxes_sv_fn = self.specified_sampled_particle_idxes_fn
                else:
                    if 'allegro_flat_fivefin_yscaled_finscaled' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_finscaled_sampled_particle_idxes.npy")
                    elif 'allegro_flat_fivefin_yscaled' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_sampled_particle_idxes.npy")
                    elif 'allegro_flat_fivefin' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_sampled_particle_idxes.npy")
                    else:
                        ## al
                        ### get the particle idxes  ###
                        # get partcle init xs #
                        sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                
                if not os.path.exists(sampled_particle_idxes_sv_fn):
                    sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)
                else:
                    sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True)
                
                self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
        if self.use_jointspace_seq:
            self.tot_object_type = np.array(self.tot_object_type, dtype=np.float32)  ### (nn_instances, )
            self.tot_rot = np.array(self.tot_rot, dtype=np.float32)
            self.tot_obj_size = np.array(self.tot_obj_size, dtype=np.float32)
            self.avg_obj_type = np.mean(self.tot_object_type)
            self.avg_rot = np.mean(self.tot_rot)
            self.avg_obj_size = np.mean(self.tot_obj_size)
            self.std_obj_type = np.std(self.tot_object_type)
            self.std_rot = np.std(self.tot_rot)
            self.std_obj_size = np.std(self.tot_obj_size)
            
            self.avg_task_setting = np.array(
                [self.avg_obj_type, self.avg_rot, self.avg_obj_size], dtype=np.float32
            )
            self.std_task_setting = np.array(
                [self.std_obj_type, self.std_rot, self.std_obj_size], dtype=np.float32
            )
                
        
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")
        self.data_name_list = list(self.data_name_to_data.keys()) # data name to data # 
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_kine_v2(self, data_dict):
        # eps = 1e-6
        # obj_verts = (obj_verts - self.avg_obj_verts[None][None]) / (self.std_obj_verts[None][None] + eps)
        
        eps = 1e-6
        # inv scale data kine info #
        data_X = data_dict['X']
        data_E = data_dict['E']
        
        avg_obj_verts_th = torch.from_numpy(self.avg_object_verts).float().cuda()
        std_obj_verts_th = torch.from_numpy(self.std_object_verts).float().cuda()
        avg_hand_qs_th = torch.from_numpy(self.avg_hand_qs).float().cuda()
        std_hand_qs_th = torch.from_numpy(self.std_hand_qs).float().cuda()
        avg_obj_rot_euler_th = torch.from_numpy(self.avg_obj_rot_euler).float().cuda()
        std_obj_rot_euler_th = torch.from_numpy(self.std_obj_rot_euler).float().cuda()
        avg_obj_transl_th = torch.from_numpy(self.avg_obj_transl).float().cuda()
        std_obj_transl_th = torch.from_numpy(self.std_obj_transl).float().cuda()
        
        
        data_E = data_E[:, 0, :, :]
        dec_hand_qs = data_E[:, :, : self.nn_hands_dof]
        dec_obj_transl = data_E[:, :, self.nn_hands_dof: self.nn_hands_dof + 3]
        dec_obj_rot_euler = data_E[:, :, self.nn_hands_dof + 3: ]
        
        inv_scaled_hand_qs = (dec_hand_qs * (std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_transl = (dec_obj_transl * (std_obj_transl_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_transl_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_rot_euler = (dec_obj_rot_euler * (std_obj_rot_euler_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_rot_euler_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_verts = (data_X * (std_obj_verts_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_verts_th.unsqueeze(0).unsqueeze(0)
        
        
        if self.task_cond:
            cond_data_X = data_dict['X_cond']
            cond_data_E = data_dict['E_cond']
            cond_data_E = cond_data_E[:, 0, :, :]
            
            dec_cond_hand_qs = cond_data_E[:, :, : self.nn_hands_dof]
            dec_cond_obj_transl = cond_data_E[:, :, self.nn_hands_dof: self.nn_hands_dof + 3]
            dec_cond_obj_rot_euler = cond_data_E[:, :, self.nn_hands_dof + 3: ]
            
            inv_scaled_cond_hand_qs = (dec_cond_hand_qs * (std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
            inv_scaled_cond_obj_transl = (dec_cond_obj_transl * (std_obj_transl_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_transl_th.unsqueeze(0).unsqueeze(0)
            inv_scaled_cond_obj_rot_euler = (dec_cond_obj_rot_euler * (std_obj_rot_euler_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_rot_euler_th.unsqueeze(0).unsqueeze(0)
            
            inv_scaled_cond_obj_verts = (cond_data_X * (std_obj_verts_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_verts_th.unsqueeze(0).unsqueeze(0)
            cond_rt_dict = {
                'cond_obj_verts': inv_scaled_cond_obj_verts,
                'cond_hand_qs': inv_scaled_cond_hand_qs,
                'cond_obj_transl': inv_scaled_cond_obj_transl,
                'cond_obj_rot_euler': inv_scaled_cond_obj_rot_euler
            }
            
            # dec_obj_transl = cond_data_E[:, :, :3]
            
            
        
        # obj_verts_avg_th = torch.from_numpy(self.avg_obj_verts).float().cuda()
        # obj_verts_std_th = torch.from_numpy(self.std_obj_verts).float().cuda() ## get the avg and std object vertices # 
        # # (3,) - dim obj_verts_avg and obj_verts_std # 
        # data_E = (obj_verts_std_th.unsqueeze(0).unsqueeze(0).unsqueeze(0) + eps ) * data_E + obj_verts_avg_th.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # tot_verts = data_X
        # tot_verts_integrated_qdd_tau = data_E 
        rt_dict = {
            # 'tot_verts': data_X,
            # 'tot_verts_integrated_qdd_tau': data_E,
            # 'tot_verts_integrated_qdd_tau_s2': data_E # inv scaled #
            'obj_verts': inv_scaled_obj_verts,
            'hand_qs': inv_scaled_hand_qs,
            'obj_transl': inv_scaled_obj_transl,
            'obj_rot_euler': inv_scaled_obj_rot_euler
        }
        
        if self.task_cond:
            rt_dict.update(
                cond_rt_dict # get the cond rt dict #
            )
            
        for key in data_dict:
            if key not in rt_dict:
                rt_dict[key] = data_dict[key]
        
        return rt_dict
        
        
    
    def inv_scale_data_v2(self, data_dict, data_nm=None, notarget=False): # bsz x nn_particles x feat_dim #
        
        if self.kine_diff:
            rt_dict = self.inv_scale_data_kine_v2(data_dict=data_dict)
            return rt_dict
        
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        
        scaled_keys = []
        
        if self.use_jointspace_seq:
            
            data_X = data_X[:, 0]
            data_E = data_E[:, 0]
            
            if self.tracking_ctl_diff:
                # self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
                # self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
                # self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
                # self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
                
                self.avg_hand_qs_th = torch.from_numpy(self.avg_hand_qs).float().cuda() #
                self.std_hand_qs_th = torch.from_numpy(self.std_hand_qs).float().cuda() #
                self.avg_hand_qtars_th = torch.from_numpy(self.avg_hand_qtars).float().cuda() # 
                self.std_hand_qtars_th = torch.from_numpy(self.std_hand_qtars).float().cuda() #
                eps = 1e-6
                
                if not self.glb_rot_use_quat:
                    data_X = (data_X * (self.std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
                    data_E = (data_E * (self.std_hand_qtars_th.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qtars_th.unsqueeze(0).unsqueeze(0)
                
                # data_X: bsz x nn_ts x feat_dim # 
                if data_nm is not None:
                    print(f"data_nm: {data_nm[0]}")
                    tot_batch_data_transl = []
                    for cur_data_nm in data_nm:
                        cur_first_frame_obj_transl = self.data_name_to_data[cur_data_nm]['first_frame_obj_trans']
                        cur_first_frame_obj_transl = torch.from_numpy(cur_first_frame_obj_transl).float().cuda() # get the first fram obj transl 
                        tot_batch_data_transl.append(cur_first_frame_obj_transl)
                    tot_batch_data_transl  = torch.stack(tot_batch_data_transl, dim=0) #### nn_bsz x 3
                    
                    # first_frame_obj_transl = self.data_name_to_data[data_nm]['first_frame_obj_trans']
                    # first_frame_obj_transl = torch.from_numpy(first_frame_obj_transl, dtype=torch.float32).cuda()
                    data_X[..., :3] = data_X[..., :3] + tot_batch_data_transl.unsqueeze(1)
                    data_E[..., :3] = data_E[..., :3] + tot_batch_data_transl.unsqueeze(1)
                
            
            if self.diff_task_space:
                
                data_X = data_X[:, 0]
                obj_type = data_X[:, 0:1] + 0.5
                data_X = torch.cat(
                    [obj_type, data_X[:, 1:]], dim=-1
                )
                data_E = data_X.clone()
                
                # avg_task_setting_th = torch.from_numpy(self.avg_task_setting).float().cuda()
                # std_task_setting_th = torch.from_numpy(self.std_task_setting).float().cuda()
                
                # data_X = data_X * (std_task_setting_th.unsqueeze(0) + eps) + avg_task_setting_th.unsqueeze(0)
                # data_E = data_X.clone()
                
            
            rt_dict = {
                'tot_verts': data_X,
                'tot_verts_integrated_qdd_tau': data_E,
                # 'tot_verts_integrated_qdd_tau_s2': data_E # inv scaled #
            }
        else:
            th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            

            
            th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            
            th_avg_particle_xs_integrated_taus_s2 = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_xs_integrated_taus_s2=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            
            
            inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            
            data_verts, data_verts_s2 = data_E[..., :3], data_E[..., 3:]
            inv_scaled_particle_xs_integrated_taus = (data_verts * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
            
            inv_scaled_particle_xs_integrated_taus_s2 = (data_verts_s2 * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2 ## get the inv_scaled integrated taus ##
            
            ###### ======= n-scale the data ======= ######
            # data_E_inv_scaled = data_E
            # data_X_inv_scaled = data_X
            ###### ======= n-scale the data ======= ######
            
            
            rt_dict = {
                'tot_verts': inv_scaled_particle_xs,
                'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
                'tot_verts_integrated_qdd_tau_s2': inv_scaled_particle_xs_integrated_taus_s2 # inv scaled #
            }
        
        if self.training_setting == 'trajectory_translations' and (not notarget):
            # inv_scaled_particle_xs_targe
            data_X_target = data_dict['X_target']
            data_E_target = data_dict['E_target']
            inv_scaled_data_X_target = (data_X_target * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            
            data_verts_target, data_verts_s2_target = data_E_target[..., :3], data_E_target[..., 3:]
            
            inv_scaled_particle_xs_integrated_taus_target = (data_verts_target * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus #
            inv_scaled_particle_xs_integrated_taus_s2_target = (data_verts_s2_target * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus 
            
            inv_scaled_data_target_dict = {
                'tot_verts_target': inv_scaled_data_X_target,
                'tot_verts_integrated_qdd_tau_target': inv_scaled_particle_xs_integrated_taus_target,
                'tot_verts_integrated_qdd_tau_s2_target': inv_scaled_particle_xs_integrated_taus_s2_target
            }
            rt_dict.update(inv_scaled_data_target_dict)
        elif self.training_setting == 'trajectory_translations_cond' and (not notarget):
            data_X_cond = data_dict['X_cond']
            data_E_cond = data_dict['E_cond']
            inv_scaled_data_X_cond = (data_X_cond * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            data_verts_cond, data_verts_s2_cond = data_E_cond[..., :3], data_E_cond[..., 3:]
            
            inv_scaled_particle_xs_integrated_taus_cond = (data_verts_cond * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus
            inv_scaled_particle_xs_integrated_taus_s2_cond = (data_verts_s2_cond * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2
            
            inv_scaled_data_cond_dict = {
                'tot_verts_cond': inv_scaled_data_X_cond,
                'tot_verts_integrated_qdd_tau_cond': inv_scaled_particle_xs_integrated_taus_cond,
                'tot_verts_integrated_qdd_tau_s2_cond': inv_scaled_particle_xs_integrated_taus_s2_cond
            }
            rt_dict.update(inv_scaled_data_cond_dict)
        # elif self.use_jointspace_seq:
        #     data_X = 
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
            
        for key in data_dict:
            if key not in rt_dict:
                rt_dict[key] = data_dict[key]
        
        return rt_dict
    
    
    def scale_data_kine(self, data_dict, data_nm=None):
        
        # sv_dict = {
        #         'obj_verts': obj_verts,
        #         'robot_hand_qs': robot_hand_qs,
        #         'obj_rot_euler': obj_rot_euler_angles,
        #         'obj_transl': obj_transl
        #     }
        
        ## TODO: load kine data in the task conditioanl setting and scale the data here ##
        
        
        obj_verts = data_dict['obj_verts'] # codn obj verts and cond hand qs? # cond hand qs #
        robot_hand_qs = data_dict['robot_hand_qs']
        obj_rot_euler = data_dict['obj_rot_euler']
        obj_transl = data_dict['obj_transl'] 
        # object_type = data_dict['object_type'] # 
        
        eps = 1e-6 # 
        scaled_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        scaled_hand_qs = (robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        scaled_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        scaled_obj_transl = (obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
        
        concat_feat = np.concatenate(
            [scaled_hand_qs, scaled_obj_transl, scaled_obj_rot_euler ], axis=-1
        )
        
        if self.task_cond:
            cond_obj_verts = data_dict['cond_obj_verts']
            cond_robot_hand_qs = data_dict['cond_robot_hand_qs']
            cond_obj_rot_euler = data_dict['cond_obj_rot_euler']
            cond_obj_transl = data_dict['cond_obj_transl']
            # eps = 1e-6
            scaled_cond_obj_verts = (cond_obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
            # cond robot hand qs #
            scaled_cond_hand_qs = (cond_robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
            scaled_cond_obj_rot_euler = (cond_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
            scaled_cond_obj_transl = (cond_obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

            cond_concat_feat = np.concatenate(
                [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
            )

        
        # robot_hand_qs = data_dict['robot_hand_qs'][:, :self.nn_hands_dof] # ts x nn_qs # 
        # robot_hand_qs = robot_hand_qs[None] # 1 x ts x nn_qs # 
        # obj_verts = data_dict['obj_verts'] # ts x nn_obj_verts x 3 # 
        # obj_verts = obj_verts.transpose(1, 0, 2)[:, : robot_hand_qs.shape[1]] # nn_obj_verts x ts x 3 # 
        # # 
        # nn_pts = 512
        # rand_sampled_obj_verts_idxes = np.random.permutation(obj_verts.shape[0])[:nn_pts] # sampled idxes # 
        # obj_verts = obj_verts[rand_sampled_obj_verts_idxes] # 
        
        # avg_obj_verts_th = torc
        # eps = 1e-6
        # obj_verts = (obj_verts - self.avg_obj_verts[None][None]) / (self.std_obj_verts[None][None] + eps)
        
        rt_dict = {
            'X': scaled_obj_verts,
            'E': concat_feat[None],
            # 'object_type': object_type
        }
        if self.task_cond:
            rt_dict.update(
                {
                    'X_cond': scaled_cond_obj_verts,
                    'E_cond': cond_concat_feat[None]
                }
            )
        
        return rt_dict
    
    def _slice_tracking_kine_data(self, ):
        tot_hand_qs = []
        tot_obj_rot_eulers = []
        tot_obj_trans = []
        self.new_data_name_to_kine_info = {}
        for data_nm in self.data_name_to_data:
            # if self.slicing_data:
            pure_data_nm = data_nm.split('_STED_')[0] + ".npy"
            # else:
            #     pure_data_nm = data_nm # + ".npy"
            
            kine_traj_info = self.data_name_to_kine_info[pure_data_nm]
            
            # if self.slicing_data:
            print(f"[Slicing tracking kine data] data_nm: {data_nm}")
            sted_info = data_nm.split("/")[-1].split(".npy")[0].split('_STED_')[1]
            st_idx, ed_idx = sted_info.split("_")
            st_idx, ed_idx = int(st_idx), int(ed_idx)
            # else:
            #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0]
            
            if self.task_cond_type == 'history_future':
                slicing_idxes = self.data_name_to_data[data_nm]['slicing_idxes']
                hand_qs = kine_traj_info['hand_qs'][slicing_idxes][:, :self.nn_hands_dof]
                obj_trans = kine_traj_info['obj_trans'][slicing_idxes]
                obj_ornt = kine_traj_info['obj_ornt'][slicing_idxes]
                obj_rot_euler = kine_traj_info['obj_rot_euler'][slicing_idxes]
                obj_verts = kine_traj_info['obj_verts']
            else:
                hand_qs = kine_traj_info['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
                obj_trans = kine_traj_info['obj_trans'][st_idx: ed_idx]
                obj_ornt = kine_traj_info['obj_ornt'][st_idx: ed_idx]
                obj_rot_euler = kine_traj_info['obj_rot_euler'][st_idx: ed_idx]
                obj_verts = kine_traj_info['obj_verts']
            
            # if self.task_cond and self.task_cond_type == 'history_future':
            if self.task_cond_type == 'history_future':
                first_frame_obj_trans = self.data_name_to_data[data_nm]['first_frame_obj_trans'] # the first frametrans
            else:
                first_frame_obj_trans = obj_trans[0, :3]
            
            hand_qs[:, :3] = hand_qs[:, :3] - first_frame_obj_trans[None]
            obj_trans = obj_trans - first_frame_obj_trans[None]
            
            kine_info_dict = {
                'obj_verts': obj_verts, 
                'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
                'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
                'obj_ornt': obj_ornt ,
                'obj_rot_euler': obj_rot_euler
            }
            self.new_data_name_to_kine_info[data_nm] = kine_info_dict
            tot_hand_qs.append(hand_qs)
            tot_obj_rot_eulers.append(obj_rot_euler)
            tot_obj_trans.append(obj_trans)
        tot_obj_transl = np.concatenate(tot_obj_trans, axis=0)
        tot_obj_rot_eulers = np.concatenate(tot_obj_rot_eulers, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        # tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        self.avg_obj_rot_euler = np.mean(tot_obj_rot_eulers, axis=0)
        self.std_obj_rot_euler = np.std(tot_obj_rot_eulers, axis=0)
        # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        
        self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        # avg obj verts and the kine hand qs and #
        # self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        # self.data_statistics['std_obj_verts'] = self.std_object_verts
        self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        self.data_statistics['std_obj_transl'] = self.std_obj_transl
        self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
        
        # 
        # self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        # self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        self.data_name_to_kine_info = self.new_data_name_to_kine_info
        
        
    
    def scale_data_tracking_ctl(self, data_dict, data_nm):
        # print(f"data_nm: {data_nm}, data_dict: {data_dict.keys()}")
        # print(f"[Scale data tracking ctl] data_nm: {data_nm} glb_rot_use_quat: {self.glb_rot_use_quat}")
        particle_xs = data_dict['tot_verts']
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        
        ## NOTE: currently they are all qs and qtars ##
        eps = 1e-6
        
        if not self.glb_rot_use_quat:
            particle_xs = (particle_xs - self.avg_hand_qs[None][None]) / (self.std_hand_qs[None][None] + eps)
            particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_hand_qtars[None][None]) / (self.std_hand_qtars[None][None] + eps)
        
        # self.data_name_to_kine_info[data_inst_tag] = {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
        #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
        #     'obj_ornt': obj_ornt 
        # }
        
        assert particle_xs.shape[1] == self.slicing_ws
        
        if particle_xs.shape[1] < self.slicing_ws:
            padding_particle_xs = np.concatenate(
                [ particle_xs[:, -1:] for _ in range(self.slicing_ws - particle_xs.shape[1]) ], axis=1
            )
            particle_xs = np.concatenate(
                [ particle_xs, padding_particle_xs ], axis=1
            )
            
            padding_particle_xs_integrated_qdd_tau = np.concatenate(
                [ particle_xs_integrated_qdd_tau[:, -1:] for _ in range(self.slicing_ws - particle_xs_integrated_qdd_tau.shape[1]) ], axis=1
            )
            particle_xs_integrated_qdd_tau = np.concatenate(
                [ particle_xs_integrated_qdd_tau, padding_particle_xs_integrated_qdd_tau ], axis=1
            )

        
        
        rt_dict = {
            'X': particle_xs,
            'E': particle_xs_integrated_qdd_tau
        }
        
        if self.task_cond:
            
            # if '_STED_' in if 
            # if self.slicing_data:
            #     pure_data_nm = data_nm.split('_STED_')[0] + ".npy"
            # else:
            #     pure_data_nm = data_nm # + ".npy"
            pure_data_nm = data_nm
            
            kine_traj_info = self.data_name_to_kine_info[pure_data_nm]
            
            # if self.slicing_data:
            #     sted_info = data_nm.split("/")[-1].split(".")[0].split('_STED_')[1]
            #     st_idx, ed_idx = sted_info.split("_")
            #     st_idx, ed_idx = int(st_idx), int(ed_idx)
            # else:
            #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0]
            st_idx = 0
            ed_idx = kine_traj_info['hand_qs'].shape[0]
            
            hand_qs = kine_traj_info['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
            obj_trans = kine_traj_info['obj_trans'][st_idx: ed_idx]
            obj_ornt = kine_traj_info['obj_ornt'][st_idx: ed_idx]
            obj_rot_euler = kine_traj_info['obj_rot_euler'][st_idx: ed_idx]
            obj_verts = kine_traj_info['obj_verts']
            
            
            # first_frame_obj_trans = obj_trans[0, :3]

            ## TODO: eulers may not be a good representation ##
            
            scaled_cond_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
            # cond robot hand qs #
            
            if not self.glb_rot_use_quat:
                # scaled_cond_hand_qs = (hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
                scaled_cond_hand_qs = (hand_qs - self.avg_kine_hand_qs[None]) / (self.std_kine_hand_qs[None] + eps)
                scaled_cond_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
                scaled_cond_obj_transl = (obj_trans - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

            # E cond --- the conditional future data #  # obj trans #
            cond_concat_feat = np.concatenate(
                [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
            )
            
            assert cond_concat_feat.shape[0] == self.slicing_ws
            
            # cond concat feat --- for the concat feat #
            # cond input # 
            if cond_concat_feat.shape[0] < self.slicing_ws:
                if cond_concat_feat.shape[0] > 0:
                    padding_concat_feat = np.concatenate(
                        [ cond_concat_feat[-1:] for _ in range(self.slicing_ws - cond_concat_feat.shape[0]) ], axis=0
                    )
                    cond_concat_feat = np.concatenate(
                        [cond_concat_feat, padding_concat_feat], axis=0
                    )
                else:
                    cond_concat_feat = np.zeros((self.slicing_ws, cond_concat_feat.shape[-1]), dtype=np.float32)
                    
            
            
            rt_dict.update(
                {
                    'X_cond': scaled_cond_obj_verts,
                    'E_cond': cond_concat_feat[None]
                }
            )
            
            if self.task_cond_type == 'history_future':
                tracking_ctl_info_dict = self.data_name_to_data[data_nm] # 
                # history_info = {
                #     'history_obj_pose': history_obj_pose[None ], # 
                #     'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                #     'first_frame_obj_trans': first_frame_obj_trans,
                #     'first_frame_obj_ornt': first_frame_obj_ornt # first frae obj ornt # an the trans # 
                # }
                history_obj_pose = tracking_ctl_info_dict['history_obj_pose'] # history obj pose -- 1 x ws x nn_obj_dim 
                history_kine_qs = tracking_ctl_info_dict['history_kine_qs'][0] # history kine qs -- 1 x ws x nn_hand_qs 
                first_frame_obj_trans = tracking_ctl_info_dict['first_frame_obj_trans']
                # 'history_obj_trans': history_obj_trans[None ], 
                        # 'history_obj_rot_euler': history_obj_rot_euler[None ],
                history_obj_rot_euler = tracking_ctl_info_dict['history_obj_rot_euler'][0]
                history_obj_trans = tracking_ctl_info_dict['history_obj_trans'][0]
                
                
                if not self.glb_rot_use_quat:
                    scaled_history_kine_qs = (history_kine_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
                    scaled_history_obj_rot_euler = (history_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
                    scaled_history_obj_trans = (history_obj_trans - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
                
                
                # # nn_history_ws x 3 # # 
                # # nn_history_ws x 3 # # 
                # # nn_history_ws x 22 # # 
                # history_cond_contact_feat = np.concatenate(
                #     [history_kine_qs, history_obj_trans, history_obj_rot_euler], axis=-1 # history cond features # 
                # )
                history_cond_contact_feat = np.concatenate(
                    [scaled_history_kine_qs, scaled_history_obj_trans, scaled_history_obj_rot_euler], axis=-1
                )
                # history cond contact feat #
                
                assert history_cond_contact_feat.shape[0] == self.history_ws #  self.slicing_ws
                # print(f"history_cond_contact_feat: {history_cond_contact_feat.shape}")
                
                if history_cond_contact_feat.shape[0] < self.slicing_ws:
                    if history_cond_contact_feat.shape[0] > 0:
                        padding_history_E_cond_feat = np.zeros_like(history_cond_contact_feat[0:1])
                        padding_history_E_cond_feat = np.concatenate(
                            [ padding_history_E_cond_feat for _ in range(self.slicing_ws - history_cond_contact_feat.shape[0]) ], axis=0
                        )
                        history_cond_contact_feat = np.concatenate(
                            [ padding_history_E_cond_feat, history_cond_contact_feat ], axis=0
                        )
                    else:
                        history_cond_contact_feat = np.zeros((self.slicing_ws, history_cond_contact_feat.shape[-1]), dtype=np.float32)
                    
                    
                    
                # print(f"[After padding] history_cond_contact_feat: {history_cond_contact_feat.shape}")
                # if history_cond_contact_feat.
                history_cond_dict = {
                    'history_E_cond': history_cond_contact_feat[None]
                }
                rt_dict.update(history_cond_dict)
                
                pass
            
        return rt_dict
                
    
    def scale_data(self, data_dict, data_nm):
        
        if self.kine_diff:
            rt_dict = self.scale_data_kine(data_dict, data_nm)
            return rt_dict
        elif self.tracking_ctl_diff:
            rt_dict = self.scale_data_tracking_ctl(data_dict, data_nm)
            return rt_dict

        
        ## nn_ts x nn_particles x 3 ## ## get scaled data ##
        
        if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond']:
            unopt_xs = data_dict['unopt_verts']
            unopt_tar_xs = data_dict['unopt_qtar_verts']
            opt_xs = data_dict['opt_verts']
            opt_tar_xs = data_dict['opt_qtar_verts']
            
            eps = 1e-6
            
            # unopt_xs = (unopt_xs - self.)
            unopt_xs = (unopt_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            unopt_tar_xs = (unopt_tar_xs - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
            
            opt_xs = (opt_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            opt_tar_xs = (opt_tar_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            
            sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        
            unopt_xs = unopt_xs[sampled_particle_idxes, :, :]
            unopt_tar_xs = unopt_tar_xs[sampled_particle_idxes, :, :]
            opt_xs = opt_xs[sampled_particle_idxes, :, :]
            opt_tar_xs = opt_tar_xs[sampled_particle_idxes, :, :]
            
            unopt_E = np.concatenate(
                [unopt_tar_xs, unopt_tar_xs], axis=-1
            )    
            opt_E = np.concatenate(
                [opt_tar_xs, opt_tar_xs], axis=-1
            )
            if self.training_setting == 'trajectory_translations_cond':
                rt_dict = {
                    'X_cond': unopt_xs,
                    'E_cond': unopt_E,
                    'X': opt_xs,
                    'E': opt_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
            else:
                rt_dict = {
                    'X': unopt_xs,
                    'E': unopt_E,
                    'X_target': opt_xs,
                    'E_target': opt_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
        
        else:
            if self.use_jointspace_seq:
                
                if self.diff_task_translations:
                    fa_task_setting = data_dict['fa_task_setting']
                    task_setting = np.array(fa_task_setting, dtype=np.float32)
                    
                    ch_task_setting = data_dict['ch_task_setting']
                    ch_task_setting = [ch_task_setting[0] - 0.5, ch_task_setting[1], ch_task_setting[2]]
                    ch_task_setting = np.array(ch_task_setting, dtype=np.float32)
                    particle_xs = ch_task_setting[None][None ]
                    particle_xs_integrated_qdd_tau = particle_xs
                else:
                    
                    particle_xs = data_dict['tot_verts']
                    particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
                    
                    ## set task setting # obj_task_setting #
                    
                    # task_setting = {
                    #     'object_type': cur_data['object_type'],
                    #     'task_rot': cur_data['task_rot'],
                    #     'object_size_x': cur_data['object_size_x']
                    # }
                    #### [object_type, task_rot, object_size_x] ####
                    task_setting = [
                        data_dict['task_setting']['object_type'], data_dict['task_setting']['task_rot'], data_dict['task_setting']['object_size_x']
                    ]
                    task_setting = np.array(task_setting, dtype=np.float32)
                    
                    
                    ### 1) make it into the particle xs and also E ###
                    
                    if self.diff_task_space:
                        eps = 1e-6
                        task_setting_2 = [
                            float(data_dict['task_setting']['object_type']), data_dict['task_setting']['task_rot'], data_dict['task_setting']['object_size_x']
                        ]
                        task_setting_2 = np.array(task_setting_2, dtype=np.float32)
                        
                        task_setting_2 = (task_setting_2 - self.avg_task_setting) / (self.std_task_setting + eps)
                        
                        particle_xs = task_setting_2[None][None]
                        particle_xs_integrated_qdd_tau = task_setting_2[None][None]
                
                
                rt_dict = {
                    'X': particle_xs,
                    'E': particle_xs_integrated_qdd_tau,
                    'obj_task_setting': task_setting #### [object_type, task_rot, object_size_x] ####
                }
            
            else:
                eps = 1e-6
                particle_xs = data_dict['tot_verts']
                particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
                particle_xs_integrated_qdd_tau_s2 = data_dict['tot_verts_integrated_qdd_tau_s2']
                
                
                particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
                particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
                particle_xs_integrated_qdd_tau_s2 = (particle_xs_integrated_qdd_tau_s2 - self.avg_verts_qdd_tau_tot_cases_tot_ts_s2[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts_s2[None][None] + eps)
                # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
                sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
                
                particle_xs = particle_xs[sampled_particle_idxes, :, :]
                particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
                particle_xs_integrated_qdd_tau_s2 = particle_xs_integrated_qdd_tau_s2[sampled_particle_idxes, :, :]
                
                data_E = np.concatenate([particle_xs_integrated_qdd_tau, particle_xs_integrated_qdd_tau_s2], axis=-1)
            
                rt_dict = {
                    'X': particle_xs,
                    'E': data_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
        
        ### return the dict ###
        ### return the dict ###
        return rt_dict
    
    #### data_dict_to_th ####
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if isinstance(data_dict_np[key], str):
                data_dict_th[key] = data_dict_np[key]
            elif key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
        
        
        return data_dict_th
    
    
    
    # def get_data_via_index(self, index) --> getitem
    def __getitem__(self, index):
        # print(f"data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        
        if self.kine_diff and self.task_cond:
            cur_data = self.tot_data_dict_list[index] 
            cur_data_nm = index
        else:
            cur_data_nm = self.data_name_list[index]
            if self.sampling and len(self.target_grab_inst_tag) > 0 and self.target_grab_data_nm is not None: 
                if self.slicing_data: ### slicing data ###
                    cur_data_nm = self.tot_target_data_nm[index % len(self.tot_target_data_nm)]
                else:
                    cur_data_nm = self.target_grab_data_nm
            if cur_data_nm not in self.data_name_to_data:
                cur_data = self._load_data_from_data_name(cur_data_nm)
                self.data_name_to_data[cur_data_nm] = cur_data
                # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
            else:
                cur_data = self.data_name_to_data[cur_data_nm]
        
        
        ## TODO: data selecting, data parsing, and data scaling
        # if self.use_target_data:
        #     cur_data_scaled = self.get_target_data() # 
        # else:
        
        # print
        # print(f"cur_data_nm: {cur_data_nm}, cur_data: {cur_data.keys()}")
        
        if self.sampling and self.use_shadow_test_data:
            cur_data_scaled = self.target_data_dict
        else:
            cur_data_scaled = self.scale_data(cur_data, cur_data_nm)
        
        # object code features #
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = { ## data scaled ##
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        
        # for cur_key in cur_data_scaled_th:
        #     cur_data_th = cur_data_scaled_th[cur_key]
        #     print(f"cur_key: {cur_key}, cur_data_th: {cur_data_th.size()}")
        
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ") 
        cur_data_scaled_th.update(
            {'data_nm': cur_data_nm}
        )
        return cur_data_scaled_th





class Uni_Manip_3D_PC_V8_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        self.debug = self.cfg.debug
        
        # 
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        # 
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        self.multi_inst = cfg.dataset_3d_pc.multi_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst
        self.specified_hand_type = cfg.dataset_3d_pc.specified_hand_type 
        
        self.specified_object_type = cfg.dataset_3d_pc.specified_object_type
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages
        self.use_static_first_frame = cfg.dataset_3d_pc.use_static_first_frame
        self.use_shadow_test_data = cfg.sampling.use_shadow_test_data
        self.sampling = cfg.sampling.sampling
        
        # self.use_allegro_test_data = cfg.sampling.use_allegro_test_data
        self.specified_test_subfolder = cfg.sampling.specified_test_subfolder
        self.specified_statistics_info_fn = cfg.training.specified_statistics_info_fn
        self.specified_sampled_particle_idxes_fn = cfg.training.specified_sampled_particle_idxes_fn
        
        self.training_setting = cfg.training.setting
        self.use_jointspace_seq = cfg.training.use_jointspace_seq
        
        # 
        self.task_cond = cfg.training.task_cond # 
        self.diff_task_space = cfg.training.diff_task_space
        self.diff_task_translations = cfg.training.diff_task_translations
        
        self.kine_diff = cfg.training.kine_diff
        self.tracking_ctl_diff = cfg.training.tracking_ctl_diff
        
        try:
            self.use_clip_glb_features = cfg.training.use_clip_glb_features
        except:
            self.use_clip_glb_features = False
            
        if self.use_clip_glb_features:  
            self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=clip_device)
        
        
        try:
            self.diff_contact_sequence = cfg.training.diff_contact_sequence
            self.contact_info_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag"
        except:
            self.diff_contact_sequence = False
            self.contact_info_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag"
        
        # 
        ''' for sampling '''
        #### get the jtarget jrab inst tagandthe optimized fn ###
        self.target_grab_inst_tag = cfg.sampling.target_grab_inst_tag
        self.target_grab_inst_opt_fn = cfg.sampling.target_grab_inst_opt_fn
        
        self.w_glb_traj_feat_cond = cfg.training.w_glb_traj_feat_cond
        self.canonicalize_features = cfg.dataset.canonicalize_features
        
        ''' for training and the training data '''
        self.grab_inst_tag_to_optimized_res_fn = cfg.training.grab_inst_tag_to_optimized_res_fn
        self.taco_inst_tag_to_optimized_res_fn = cfg.training.taco_inst_tag_to_optimized_res_fn
        if len(self.taco_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.taco_inst_tag_to_optimized_res_fn):
            self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn, self.taco_inst_tag_to_optimized_res_fn]    

        try:
            self.inv_kine_freq = cfg.training.inv_kine_freq
        except:
            self.inv_kine_freq = 1
        

        try:
            self.use_taco_data = cfg.training.use_taco_data
        except:
            self.use_taco_data = False
    
        try:
            self.glb_rot_use_quat = cfg.training.glb_rot_use_quat
        except:
            self.glb_rot_use_quat = False
        self.succ_rew_threshold = cfg.training.succ_rew_threshold # 
        
        
        try:
            self.task_cond_type = cfg.training.task_cond_type
        except:
            self.task_cond_type = "future"
        
        try:
            self.slicing_ws = cfg.training.slicing_ws
        except:
            self.slicing_ws = 30
            pass
        
        ### TODO: a slicing ws with an additional history window ws for tracking ###
        ### trajs obtained via closed loop planning? ###
        
        try:
            self.history_ws = cfg.training.history_ws
        except:
            self.history_ws = self.slicing_ws
        
        
        try:
            self.use_kine_obj_pos_canonicalization = cfg.training.use_kine_obj_pos_canonicalization
        except:
            self.use_kine_obj_pos_canonicalization = False
        
        
        try:
            self.exp_additional_tag = cfg.training.exp_additional_tag
        except:
            self.exp_additional_tag = ''
    
        #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy #
        # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag #
        try:
            self.taco_interped_fr_grab_tag = cfg.training.taco_interped_fr_grab_tag
        except:
            self.taco_interped_fr_grab_tag = "ori_grab_s2_phone_call_1"
        
        try:
            self.taco_interped_data_sv_additional_tag = cfg.training.taco_interped_data_sv_additional_tag
        except:
            self.taco_interped_data_sv_additional_tag = 'v2'
        
        try:
            self.num_frames = cfg.dataset.num_frames
        except:
            self.num_frames = 150
        
        # valid_data_statistics = None 
        
        try:
            self.task_inherit_info_fn = cfg.training.task_inherit_info_fn
        except:
            self.task_inherit_info_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy"
        
        try:
            self.obj_type_to_kinematics_traj_dict_fn = cfg.dataset_3d_pc.obj_type_to_kinematics_traj_dict_fn
        except:
            self.obj_type_to_kinematics_traj_dict_fn = ''
        
        try:
            self.canonicalize_obj_pts = cfg.dataset_3d_pc.canonicalize_obj_pts
        except:
            self.canonicalize_obj_pts = False
            
        try:
            self.forcasting_window_size = cfg.dataset_3d_pc.forcasting_window_size
        except:
            self.forcasting_window_size = 30
            
        try:
            self.w_masked_future_cond = cfg.training.w_masked_future_cond
        except:
            self.w_masked_future_cond = False
        
        
        
        # load_excuted_optimized_res, excuted_inst_tag_to_optimized_res
        # try:
        #     self.data_inst_tag_to_optimized_res_fn = cfg.training.
        try:
            self.load_excuted_optimized_res = cfg.training.load_excuted_optimized_res
        except:
            self.load_excuted_optimized_res=  False
            
        
        
        # partial information # # partial information # # # partial information # # # # partial information #
        # TODO: define the succ threshhoulds which determines whether the current sampled res can be used #
        # It should be a fn to a dict that summarizes the isntance tag to the optimized results file #
        try:
            self.excuted_inst_tag_to_optimized_res = cfg.training.excuted_inst_tag_to_optimized_res
        except:
            self.excuted_inst_tag_to_optimized_res = ''
        # TODO: we need the thresholds as well #
        
        self.partial_obj_pos_info = cfg.training.partial_obj_pos_info 
        
        try:
            self.centralize_info = cfg.training.centralize_info
        except:
            self.centralize_info = False
        
        
        #### TODO: Centralize, without canonicalization, without st_ed conditions, glb_skip=6, partial hand info only, history_window_size=1 #####
        ### TODO: add the global information window skipping information ###
        try:
            self.glb_feat_per_skip = cfg.training.glb_feat_per_skip
        except:
            self.glb_feat_per_skip = 1
        
        try:
            self.history_window_size = cfg.dataset_3d_pc.history_window_size
        except:
            self.history_window_size = self.forcasting_window_size


        try:
            self.scale_clip_data = cfg.dataset_3d_pc.scale_clip_data
        except:
            self.scale_clip_data = False
        
        # scale clip data ## 
        try:
            self.text_feature_version = cfg.training.text_feature_version
        except:
            self.text_feature_version = 'v1'
        
        
        try:
            self.inv_forecasting_freq = cfg.training.inv_forecasting_freq
        except:
            self.inv_forecasting_freq = 1
        
        # inv scale the text feature versions # inv sliding data freuency # # inv slide data frequency #
        
        
        self.maxx_nn_pts = 512
        
        
        self.maxx_ws_size = 300
        
        # try:
        #     self.step_size = 
        # self.step_size = self.slicing_ws // 2
        
        # if self.step_size == 0:
        #     self.step_size = 1
        
        self.step_size = 1
        
        try:
            self.slicing_data = cfg.training.slicing_data
        except:
            self.slicing_data = False
        
        
        self.obj_name_to_idx = {
            'box': 0,
            'cylinder': 1
        }
        
        self.nn_hands_dof = 22
        
        # exp_tags = ["tds_exp_2"]
        
        
        self.sim_platform = cfg.dataset_3d_pc.sim_platform
        
        self.data_statistics_info_fn = cfg.dataset_3d_pc.data_statistics_info_fn
        
        self.statistics_info_fn = cfg.dataset_3d_pc.statistics_info_fn
        
        print(f"statistics_info_fn: {self.statistics_info_fn}")
        
        self.tot_inheriting_infos = []

        self.hybrid_dataset = False

        # self.tracking_save_info_fn = "/cephfs/yilaa/data/GRAB_Tracking/data"
        self.tracking_save_info_fn = cfg.dataset_3d_pc.tracking_save_info_fn 
        self.tracking_info_st_tag = "passive_active_info_"
        
        self.target_grab_data_nm = None
        
        subj_idx = None
        subj_idx = 2
        
        tracking_data_root_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
        
        
        
        if not os.path.exists(tracking_data_root_folder):
            self.load_n_data = True
            self.data_list = [f"{idxx}" for idxx in range(0, 1000)]
        else:
            self.load_n_data = False
            ## TODO: 1) add canonicalize_obj_pts; 2) add obj_type_to_kinematics_traj_dict_fn #
            if len(self.obj_type_to_kinematics_traj_dict_fn) != 0 and os.path.exists(self.obj_type_to_kinematics_traj_dict_fn):
                self.obj_type_to_kinematics_traj_dict = np.load(self.obj_type_to_kinematics_traj_dict_fn, allow_pickle=True).item()
                self.objtype_to_tracking_sv_info = self.obj_type_to_kinematics_traj_dict
                self.data_list = [
                    self.objtype_to_tracking_sv_info[obj_type] for obj_type in self.objtype_to_tracking_sv_info
                ]
            else:
                passive_act_info_tag = 'passive_active_info_ori_grab'
                # tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
                # tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
                tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
                local_data_tracking_save_info_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
                if os.path.exists(local_data_tracking_save_info_fn):
                    tracking_save_info_fn = local_data_tracking_save_info_fn
                print("Start enumerating retargeted tracking info")
                tracking_save_info = os.listdir(tracking_save_info_fn)
                tracking_save_info = [
                    fn for fn in tracking_save_info if fn.endswith(".npy") and fn[: len(passive_act_info_tag)] == passive_act_info_tag
                ]
                
                
                
                # if self.num_frames == 150:
                #     non_nf_tag = '_nf_'
                #     tracking_save_info = [
                #         fn for fn in tracking_save_info if non_nf_tag not in fn
                #         ]
                
                nf_tag = f'_nf_{self.num_frames}'
                tracking_save_info = [
                    fn for fn in tracking_save_info if nf_tag in fn
                ]
                
                if subj_idx is not None:
                    tracking_save_info = [
                        fn for fn in tracking_save_info if f'_s{subj_idx}_' in fn
                    ]
                
                # tracking_save_info = tracking_save_info[:10]
                
                passive_act_pure_tag = "passive_active_info_"
                self.objtype_to_tracking_sv_info = {}
                for cur_sv_info in tracking_save_info:
                    cur_objtype = cur_sv_info.split(".")[0]
                    cur_objtype = cur_objtype.split("_nf_")[0]
                    cur_objtype = cur_objtype[len(passive_act_pure_tag): ]
                    self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)

                    # cur_objtype = "ori_grab" + cur_objtype # cur obj type # # 
                    # self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)

                tracking_save_info = [
                    os.path.join(tracking_save_info_fn, fn) for fn in tracking_save_info
                ]
                self.data_list = tracking_save_info
        
        self.data_name_to_data = {}
        self.inst_tag_to_kine_traj_data = {}
        
        self._preload_kine_data()
        
        
        
        if self.load_excuted_optimized_res: # preload the optimized res data #
            print(f"Start loading excuted optimized res...") # so the tnex step -- you should add relevant files here and load them #
            self._preload_optimized_res_data()
        
        
        # sv_dict = {
        #     'obj_verts': obj_verts,
        #     'robot_hand_qs': robot_hand_qs,
        #     'obj_rot_euler': obj_rot_euler_angles,
        #     'obj_transl': obj_transl,
        #     'object_type': object_type,
        # }
        ### TODO: add the task description ###
        forcasting_window_size = self.forcasting_window_size #  30
        self.data_with_window_info_list = []
        
        self.tot_clip_hand_qs = []
        self.tot_clip_obj_pos = []
        
        
        for data_nm in self.data_name_to_data:
            cur_val_dict = self.data_name_to_data[data_nm]
            cur_obj_transl = cur_val_dict['obj_transl']
            print(f"cur_obj_transl: {cur_obj_transl.shape}")
            robot_hand_qs = cur_val_dict['robot_hand_qs']
            
            # sv_dict = {
            #     'obj_verts': obj_verts,
            #     'robot_hand_qs': robot_hand_qs,
            #     'obj_rot_euler': obj_rot_euler_angles,
            #     'obj_transl': obj_transl,
            #     'object_type': object_type,
            #     'obj_rot_quat': obj_rot_quat
            # }
            
            if forcasting_window_size == self.maxx_ws_size:
                history_ed_idx = 0
                history_last_idx = history_ed_idx - forcasting_window_size + 1
                future_st_idx = 0
                future_ed_idx = future_st_idx + forcasting_window_size - 1
                history_window_idxes = range(history_last_idx, history_ed_idx + 1)
                future_window_idxes = range(future_st_idx, future_ed_idx + 1)
                
                history_window_idxes  = np.array(history_window_idxes, dtype=np.int32)
                future_window_idxes = np.array(future_window_idxes, dtype=np.int32)
                history_window_idxes = np.clip(history_window_idxes, 0, cur_obj_transl.shape[0] - 1)
                future_window_idxes = np.clip(future_window_idxes, 0, cur_obj_transl.shape[0] - 1)
                self.data_with_window_info_list.append(
                    {
                        'data_nm': data_nm,
                        'history_window_idxes': history_window_idxes,
                        'future_window_idxes': future_window_idxes
                    }
                )
            else:
                
                # obj transl #
                for idx in range(cur_obj_transl.shape[0]):
                    history_last_idx = idx - forcasting_window_size + 1 
                    future_st_idx = idx + 1
                    future_ed_idx = idx + forcasting_window_size 
                    
                    history_window_idxes = range(forcasting_window_size - 1, -1, -1)
                    history_window_idxes = np.array(history_window_idxes, dtype=np.int32) * self.inv_forecasting_freq
                    
                    # forcasting window  size #
                    future_window_idxes = range(1, forcasting_window_size + 1) # inv forcasting freq #
                    future_window_idxes = np.array(future_window_idxes, dtype=np.int32) * self.inv_forecasting_freq
                    
                    # forcasting window size - 1, 1, -1 # # history; #
                    history_window_idxes = idx - history_window_idxes # future 
                    future_window_idxes = idx + future_window_idxes # future idxes and history idxes #
                    
                    
                    history_window_idxes = np.clip(history_window_idxes, 0, cur_obj_transl.shape[0] - 1)
                    future_window_idxes = np.clip(future_window_idxes, 0, cur_obj_transl.shape[0] - 1)
                    
                    
                    # 
                    # history_window_idxes = range(history_last_idx, idx + 1)
                    # future_window_idxes = range(future_st_idx, future_ed_idx + 1)
                    # # history window idxes #
                    # history_window_idxes  = np.array(history_window_idxes, dtype=np.int32)
                    # future_window_idxes = np.array(future_window_idxes, dtype=np.int32)
                    
                    # # history_window_idxes = historywindowidx
                    
                    # history_window_idxes = np.clip(history_window_idxes, 0, cur_obj_transl.shape[0] - 1)
                    # future_window_idxes = np.clip(future_window_idxes, 0, cur_obj_transl.shape[0] - 1)
                    
                    
                    if self.centralize_info:
                        history_hand_qs = robot_hand_qs[history_window_idxes].copy()
                        future_hand_qs = robot_hand_qs[future_window_idxes].copy()
                        history_obj_transl = cur_obj_transl[history_window_idxes].copy()
                        future_obj_transl = cur_obj_transl[future_window_idxes].copy()
                        last_frame_hand_qs = history_hand_qs[-1] # the last frame histroy hand qs # 
                        history_hand_qs[..., :3] = history_hand_qs[..., :3] - last_frame_hand_qs[None, :3]
                        future_hand_qs[..., :3] = future_hand_qs[..., :3] - last_frame_hand_qs[None, :3]
                        history_obj_transl = history_obj_transl - last_frame_hand_qs[None, :3]
                        future_obj_transl = future_obj_transl - last_frame_hand_qs[None, :3]
                        self.tot_clip_hand_qs.append(history_hand_qs)
                        self.tot_clip_hand_qs.append(future_hand_qs)
                        self.tot_clip_obj_pos.append(history_obj_transl)
                        self.tot_clip_obj_pos.append(future_obj_transl)
                    
                    # sparse -> dense targets #
                    # description with the current state -> future state # # that's true #
                    # #
                    self.data_with_window_info_list.append(
                        {
                            'data_nm': data_nm,
                            'history_window_idxes': history_window_idxes,
                            'future_window_idxes': future_window_idxes
                        }
                    )
            
            if not self.centralize_info:
                self.tot_clip_hand_qs.append(robot_hand_qs)
                self.tot_clip_obj_pos.append(cur_obj_transl)
            
            # pass
           
        #  avg_tot_clip_hand_qs, std_tot_clip_hand_qs, avg_tot_clip_obj_pos, std_tot_clip_obj_pos
        self.tot_clip_hand_qs = np.concatenate(self.tot_clip_hand_qs, axis=0)
        self.tot_clip_obj_pos = np.concatenate(self.tot_clip_obj_pos, axis=0)
        self.avg_tot_clip_hand_qs = np.mean(self.tot_clip_hand_qs, axis=0) # avg hand qs
        self.std_tot_clip_hand_qs = np.std(self.tot_clip_hand_qs, axis=0) # std hand qs 
        self.avg_tot_clip_obj_pos = np.mean(self.tot_clip_obj_pos, axis=0) # avg obj pos
        self.std_tot_clip_obj_pos = np.std(self.tot_clip_obj_pos, axis=0) # std obj pos
        
        self.clip_statistics_info = {
            'avg_tot_clip_hand_qs': self.avg_tot_clip_hand_qs,
            'std_tot_clip_hand_qs': self.std_tot_clip_hand_qs,
            'avg_tot_clip_obj_pos': self.avg_tot_clip_obj_pos,
            'std_tot_clip_obj_pos': self.std_tot_clip_obj_pos # getthe clip statistics info #
        }
        
        # 
        self._save_clip_data_statistics()
        
        
        
    
    
    def __len__(self):
        
        return len(self.data_with_window_info_list)
        
        ### the lenght of the target ###
        if self.sampling and self.target_grab_data_nm is not None and self.slicing_data and len(self.target_grab_inst_tag) > 0:
            return len(self.tot_target_data_nm)
                # tot_target_data_nm #
        
        if self.kine_diff and self.task_cond:
            return len(self.tot_data_dict_list)
        else:
            return len(self.data_name_to_data)
        # return len(self.data_name_list) 
    
    
    def _save_clip_data_statistics(self, ):
        data_statistics_info_fn = "../assets/clip_data_statistics_info.npy"
        np.save(data_statistics_info_fn, self.clip_statistics_info)
        print(f"Clip data statistics saved to {data_statistics_info_fn}")
    
    
    def _save_kine_data_statistics(self, ):
        # data_statistics
        if len(self.exp_additional_tag) ==0:
            data_stats_sv_fn = f"data_statistics_kinematics_diff.npy"
        else:
            data_stats_sv_fn = f"data_statistics_kinematics_diff_{self.exp_additional_tag}.npy"
        data_stats_sv_fn = os.path.join(f"assets", data_stats_sv_fn)
        np.save(data_stats_sv_fn, self.data_statistics)
        print(f"[Kine Diff] Data statistics saved to {data_stats_sv_fn}")
    
    def _save_data_statistics(self, ):
        ##### data stats sv fn #####
        if len(self.exp_additional_tag) == 0:
            data_stats_sv_fn = f"data_statistics_ws_{self.slicing_ws}_step_{self.step_size}.npy"
        else:
            # exp_additional_tag
            data_stats_sv_fn = f"data_statistics_ws_{self.slicing_ws}_step_{self.step_size}_{self.exp_additional_tag}.npy"
        
        data_stats_sv_fn = os.path.join(f"assets", data_stats_sv_fn)
        np.save(data_stats_sv_fn, self.data_statistics)
        print(f"Data statistics saved to {data_stats_sv_fn}") 
        
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ## ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    
    def _load_single_tracking_kine_info(self, data_inst_tag):
        if isinstance(data_inst_tag, str):
            
            kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
            kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
            # get he kinemati info file # # load #  # load from that # 
            cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
            
            hand_qs = cur_kine_data['robot_delta_states_weights_np']
            maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            hand_qs = hand_qs[:maxx_ws]
            
            obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
            
            # then segment the data_inst_tag to get the mesh file name #
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{data_inst_tag}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            while len(to_sample_fr_idxes) < self.maxx_nn_pts:
                to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
        elif isinstance(data_inst_tag, tuple): # obj
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
                obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws] # obj verts #
            elif 'taco' in obj_type:
                #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy
                # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
                traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_{self.taco_interped_data_sv_additional_tag}.npy'
                taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                traj_kine_info = os.path.join(taco_kine_sv_root, traj_kine_info) # get hejkineinfo s 
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
                hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
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
        
        
        kine_obj_rot_euler_angles = []
        for i_fr in range(obj_ornt.shape[0]):
            cur_rot_quat = obj_ornt[i_fr] # dkine obj rot quat #
            cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
            kine_obj_rot_euler_angles.append(cur_rot_euler)
        kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
        kine_info_dict = {
            'obj_verts': obj_verts, 
            'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
            'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
            'obj_ornt': obj_ornt ,
            'obj_rot_euler': kine_obj_rot_euler_angles
        }
        return kine_info_dict
    
    
    
    def _load_tracking_kine_info(self,):
        # self.maxx_kine_nn_ts = 300
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        # add the positional encodings into the model encodings? #
        # add the positional encodings into the model encoding? #
        # each is encoded positionals is encodied into the model #
        # give the number of timesteps and the timesteps #
        # give the total number of tiemsteps # tracking controls #
        # domain randomizations an the control alignmnet #
        # the relative positions of the frame sequence in the total trajectory to track # #
        # the relative positions # # relative # # # #
        
        ## maxx kine nn ts setting ##
        for i_inst, data_inst_tag in enumerate(self.data_inst_tag_list):
            print(f"[Loading tracking kine info] {i_inst}/{len(self.data_inst_tag_list)}: {data_inst_tag}")
            kine_info_dict = self._load_single_tracking_kine_info(data_inst_tag)
            # if isinstance(data_inst_tag, str):
            #     kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
            #     kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
            #     # get he kinemati info file # # load #  # load from that # 
            #     cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
                
            #     hand_qs = cur_kine_data['robot_delta_states_weights_np']
            #     maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            #     hand_qs = hand_qs[:maxx_ws]
                
            #     obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            #     obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
            #     # then segment the data_inst_tag to get the mesh file name #
            #     self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            #     grab_mesh_fn = f"{data_inst_tag}.obj"
            #     grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
            #     # get the object mesh #
            #     obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            #     obj_verts = obj_mesh.vertices # nn_pts x 3
            #     # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            #     to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            #     while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            #         to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            #     random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            #     random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            #     obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            # elif isinstance(data_inst_tag, tuple): # obj
            #     obj_type, traj_obj_type = data_inst_tag
            #     traj_kine_info  = f"{self.tracking_info_st_tag}{traj_obj_type}.npy"
            #     traj_kine_info = os.path.join(self.tracking_save_info_fn, traj_kine_info)
            #     traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
            #     hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
            #     maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            #     hand_qs = hand_qs[:maxx_ws]
                
            #     obj_trans = traj_kine_info_data['object_transl'][:maxx_ws] # nn_ts x 3 
            #     obj_ornt = traj_kine_info_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
            #     # then segment the data_inst_tag to get the mesh file name #
            #     self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            #     grab_mesh_fn = f"{obj_type}.obj"
            #     grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
            #     # get the object mesh #
            #     obj_mesh = trimesh.load(grab_mesh_fn, force='mesh')
            #     obj_verts = obj_mesh.vertices # nn_pts x 3
            #     # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            #     # random_sampled_idxes = np.random.sample()
            #     to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            #     while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            #         to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            #     random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            #     random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            #     obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws] # obj verts #
            #     # grab_mesh_fn = f"{data_inst_tag}.obj"
            #     # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            # else: #
            #     raise ValueError(f"Unrecognized grab_data_inst_type: {type(data_inst_tag)}")
                
            # # hand_qs = cur_kine_data['robot_delta_states_weights_np'][:maxx_ws] # nn_ts x 22
            
            # kine_obj_rot_euler_angles = []
            # for i_fr in range(obj_ornt.shape[0]):
            #     cur_rot_quat = obj_ornt[i_fr] # dkine obj rot quat #
            #     cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True) # get the rotation euler 
            #     kine_obj_rot_euler_angles.append(cur_rot_euler)
            # kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            
            # the task conditional settings #
            self.data_name_to_kine_info[self.data_list[i_inst]] = kine_info_dict
            # { # data list to the obj verts and the ahand qs # 
            #     'obj_verts': obj_verts, 
            #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
            #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
            #     'obj_ornt': obj_ornt ,
            #     'obj_rot_euler': kine_obj_rot_euler_angles
            # }
            obj_trans, kine_obj_rot_euler_angles, hand_qs, obj_verts = kine_info_dict['obj_trans'], kine_info_dict['obj_rot_euler'], kine_info_dict['hand_qs'], kine_info_dict['obj_verts']

            tot_obj_transl.append(obj_trans)
            tot_obj_rot_euler.append(kine_obj_rot_euler_angles)
            tot_hand_qs.append(hand_qs)
            tot_object_verts.append(obj_verts)
        
        
        tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
        self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
        # avg hand qs and the std hand qs #?
        ## TODO: for the kinematics target data --- we should save them using a differnet name? #
        # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        # avg kine hand qs #
        self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        
        
        # 
        self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        
        # avg obj verts and the kine hand qs and #
        self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        self.data_statistics['std_obj_verts'] = self.std_object_verts
        self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        self.data_statistics['std_obj_transl'] = self.std_obj_transl
        self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
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
        
    
    
    # let it condition on such information is more important #
    # cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(target_data_fn)
    def _preload_single_tracking_ctl_data(self, data_fn, add_to_dict=True):
        
        print(f"loading from {data_fn}")
        cur_data = np.load(data_fn, allow_pickle=True).item()
        if self.use_jointspace_seq:
            if self.sim_platform == 'isaac':
                optimized_obj_pose = cur_data['optimized_obj_pose']
                optimized_hand_qtars = cur_data['optimized_hand_qtars']
                optimized_hand_qs = cur_data['optimized_hand_qs']
                # # TODO: use top-k other than using the best evaluated res ? # 
                hand_qs_np = optimized_hand_qs[0] 
                hand_qtars_np = optimized_hand_qtars[0] # 
                
                if self.glb_rot_use_quat:
                    hand_glb_rot_np = hand_qs_np[..., 3:6]
                    hand_glb_qtar_np = hand_qtars_np[..., 3:6]
                    hand_glb_rot_th = torch.from_numpy(hand_glb_rot_np)
                    hand_glb_tar_rot_th = torch.from_numpy(hand_glb_qtar_np)
                    hand_glb_quat_th = quat_from_euler_xyz(hand_glb_rot_th[..., 0], hand_glb_rot_th[..., 1], hand_glb_rot_th[..., 2])
                    hand_glb_tar_quat_th = quat_from_euler_xyz(hand_glb_tar_rot_th[..., 0], hand_glb_tar_rot_th[..., 1], hand_glb_tar_rot_th[..., 2])
                    hand_glb_rot_np = hand_glb_quat_th.numpy()
                    hand_glb_qtar_np = hand_glb_tar_quat_th.numpy()
                    
                    hand_qs_np = np.concatenate(
                        [ hand_qs_np[..., :3], hand_glb_rot_np, hand_qs_np[..., 6:] ], axis=-1
                    )
                    hand_qtars_np = np.concatenate(
                        [ hand_qtars_np[..., :3], hand_glb_qtar_np, hand_qtars_np[..., 6:] ], axis=-1
                    )
                    # hand_qs_np[..., 3:6] = hand_glb_rot_np
                    # hand_qtars_np[..., 3:6] = hand_glb_qtar_np
                # obj_pose_np = cu
            else:
                ts_to_hand_qs = cur_data['ts_to_hand_qs'] 
                ts_to_hand_qtars = cur_data['ts_to_qtars']
                if self.slicing_data:
                    sorted_ts = sorted(list(ts_to_hand_qs.keys()))
                    hand_qs_np = [
                        ts_to_hand_qs[i_ts] for i_ts in sorted_ts
                    ]
                    hand_qtars_np = [
                        ts_to_hand_qtars[i_ts] for i_ts in sorted_ts
                    ]
                else:
                    if 'ts_to_optimized_q_tars_wcontrolfreq' in cur_data:
                        ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
                    else:
                        ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_hand_tot_ctl_qtars']
                    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                        
                    max_hand_qs_kd = max(list(ts_to_hand_qs.keys()))
                    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
                    ctl_freq_tss = sorted(ctl_freq_tss) # ctl freq tss #
                    ctl_freq = 10 # 
                    ctl_freq_tss_expanded = [ min(max_hand_qs_kd, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
                    ts_to_hand_qs = {
                        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
                    }
                    hand_qs_np = [
                        ts_to_hand_qs[cur_ts] for cur_ts in ctl_freq_tss
                    ]
                    hand_qtars_np = [ # 
                        ts_to_hand_qtars[cur_ts] for cur_ts in ctl_freq_tss
                    ]
                hand_qs_np = np.stack(hand_qs_np, axis=0)
                hand_qtars_np = np.stack(hand_qtars_np, axis=0) 
                
            # cur_data_nm = data_nm #
            
            # tot_data_hand_qs.append(hand_qs_np)
            # tot_data_hand_qtars.append(hand_qtars_np)

            cur_clip_data = {
                'tot_verts': hand_qs_np[None],  # hand qtars np; tot verts # 
                # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
                'tot_verts_integrated_qdd_tau': hand_qtars_np[None], # 
                # 'task_setting': task_setting, # rotation euler angles? # 
                # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            } # grab inst tag #
            
            if self.task_cond_type == 'history_future':
                obj_pose_np = optimized_obj_pose[0]
                cur_clip_data.update(
                    {
                        'tot_obj_pose': obj_pose_np[None]
                    }
                )
                
            
            # cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
            # using the data fn as the data name # 
            if add_to_dict:
                self.data_name_to_data[data_fn] = cur_clip_data
        else:
            raise NotImplementedError(f"Unrecognized use_jointspace_seq: {self.use_jointspace_seq}")
        return cur_clip_data, hand_qs_np, hand_qtars_np
        
    
    def _preload_mocap_tracking_ctl_data(self,): # 
        # self.data_list #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        if self.single_inst: # 
            self.data_list = self.data_list[:1]
            self.data_inst_tag_list = self.data_inst_tag_list[:1]
        elif self.multi_inst:
            self.data_list = self.data_list[:10]
            self.data_inst_tag_list = self.data_inst_tag_list[:10]
        # tot_expanded_passive #
        forbid_data_inst_tags = ["ori_grab_s2_phone_call_1", "ori_grab_s2_phone_pass_1"]
        
        for i_data_inst, data_fn in enumerate(self.data_list):
            
            excluded = False 
            for cur_forbid_inst_tag in forbid_data_inst_tags:
                if cur_forbid_inst_tag in data_fn:
                    excluded = True
                    break
            if excluded: ## # excluded ##
                continue
            
            print(f"loading from {data_fn}")
            # preload the tracking #
            cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(data_fn)
            tot_data_hand_qs.append(hand_qs_np)
            tot_data_hand_qtars.append(hand_qtars_np)
            self.data_name_to_data[data_fn] = cur_clip_data
            # cur_data = np.load(data_fn, allow_pickle=True).item()
            
            # if self.use_jointspace_seq:
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
                    
            #     # cur_data_nm = data_nm
                
            #     tot_data_hand_qs.append(hand_qs_np)
            #     tot_data_hand_qtars.append(hand_qtars_np)
                

            #     cur_clip_data = {
            #         'tot_verts': hand_qs_np[None],  # hand qtars np; tot verts # 
            #         # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
            #         'tot_verts_integrated_qdd_tau': hand_qtars_np[None], # 
            #         # 'task_setting': task_setting, # rotation euler angles? # 
            #         # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            #     } # grab inst tag #
            #     cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
            #     # using the data fn as the data name # 
            #     self.data_name_to_data[data_fn] = cur_clip_data
            # else:
            #     raise NotImplementedError(f"Unrecognized use_jointspace_seq: {self.use_jointspace_seq}")
        
        tot_data_hand_qs = np.concatenate(tot_data_hand_qs, axis=0)
        tot_data_hand_qtars = np.concatenate(tot_data_hand_qtars, axis=0)
        self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
        self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
        self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
        self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
        
        ### add to data statistics ###
        ### TODO: we should put object stats here --- obj stats should be calculated from the tracked trajectories ###
        self.data_statistics['avg_hand_qs'] = self.avg_hand_qs
        self.data_statistics['std_hand_qs'] = self.std_hand_qs
        self.data_statistics['avg_hand_qtars'] = self.avg_hand_qtars
        self.data_statistics['std_hand_qtars'] = self.std_hand_qtars
        
    
    
    def _get_obj_slicing_rot_trans(self, data_nm, st_idx, ed_idx):
        
        if self.task_cond_type == 'history_future':
            obj_pose = self.data_name_to_data[data_nm]['tot_obj_pose'][0]
            #  get the kine obj pose states # history future #
            
            # sliced_obj_pose = obj_pose[st_idx: ed_idx]
            sliced_obj_pose = obj_pose[st_idx - 1: ed_idx - 1 ]
            sliced_obj_trans, sliced_obj_ornt = sliced_obj_pose[:, :3], sliced_obj_pose[:, 3:]
            
            if self.use_kine_obj_pos_canonicalization:
                kine_info_dict = self.data_name_to_kine_info[data_nm]
                obj_trans = kine_info_dict['obj_trans']
                obj_ornt = kine_info_dict['obj_ornt']
                sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx - 1: ed_idx - 1], obj_ornt[st_idx - 1: ed_idx - 1]
                
        else: # sliced obj trans and the ornt #
            kine_info_dict = self.data_name_to_kine_info[data_nm]
            obj_trans = kine_info_dict['obj_trans']
            obj_ornt = kine_info_dict['obj_ornt']
            sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx: ed_idx], obj_ornt[st_idx: ed_idx]
        return sliced_obj_trans, sliced_obj_ornt
        # {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
        #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
        #     'obj_ornt': obj_ornt ,
        #     'obj_rot_euler': kine_obj_rot_euler_angles
        # }
    
    
    def _slicing_single_mocap_tracking_ctl_data(self, grab_opt_data_fn, cur_data_dict, kine_data_dict, add_to_dict=True):
        kine_qs = cur_data_dict['tot_verts'][0]
        q_tars = cur_data_dict['tot_verts_integrated_qdd_tau'][0]
        if self.task_cond_type == 'history_future':
            obj_pose = cur_data_dict['tot_obj_pose'][0]
        obj_trans = kine_data_dict['obj_trans']
        obj_ornt = kine_data_dict['obj_ornt']
        
        # slice # 
        print(f"kine_qs: {kine_qs.shape}, q_tars: {q_tars.shape}, obj_trans: {obj_trans.shape}, obj_ornt: {obj_ornt.shape}")
        
        slicing_st_idx = 0
        if self.task_cond_type == 'history_future':
            slicing_st_idx = 1
        
        self.tot_target_data_nm = []
        for i_slice in range(slicing_st_idx, kine_qs.shape[0] - self.slicing_ws, self.step_size):
            st_idx = i_slice # the start index #
            ed_idx = i_slice + self.slicing_ws
            
            slicing_idxes = list(range(st_idx, ed_idx))
            slicing_idxes = np.array(slicing_idxes, dtype=np.int32) # get the slicing idxes # 
            slicing_idxes = np.clip(slicing_idxes, a_min=0, a_max=kine_qs.shape[0] - 1)
            
            # ed_idx = min(i_slice + self.slicing_ws, kine_qs.shape[0])
            
            
            
            # sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx: ed_idx], obj_ornt[st_idx: ed_idx]
            sliced_obj_trans, sliced_obj_ornt = obj_trans[slicing_idxes], obj_ornt[slicing_idxes]
            
            # ge the slicing data obj trans #
            if self.task_cond_type == 'history_future':
                sliced_obj_pose = obj_pose[st_idx - 1: ed_idx - 1 ]
                first_frame_obj_trans = sliced_obj_pose[0, :3]
                first_frame_obj_ornt = sliced_obj_pose[0, 3:]
            else:
                first_frame_obj_trans = sliced_obj_trans[0, :3] # first farme obj trans #
            
            # cur_slice_kine_qs = kine_qs[st_idx: ed_idx]
            # cur_slice_q_tars = q_tars[st_idx: ed_idx]
            
            cur_slice_kine_qs = kine_qs[slicing_idxes]
            cur_slice_q_tars = q_tars[slicing_idxes]
            
            #### NOTE: A simple canonicalization ####
            cur_slice_kine_qs[:, :3] = cur_slice_kine_qs[:, :3] - first_frame_obj_trans[ None]
            cur_slice_q_tars[:, :3] = cur_slice_q_tars[:, :3] - first_frame_obj_trans[ None]
            #### NOTE: A simple canonicalization ####

            cur_slice_data = {
                'tot_verts': cur_slice_kine_qs[None],
                'tot_verts_integrated_qdd_tau': cur_slice_q_tars[None]
            }
            
            
            # TODO: change all the jugdement logic to this # 
            # 
            if  self.task_cond_type == 'history_future': # history future #
                history_st_idx = st_idx - self.slicing_ws
                # history_st_idx = max(0, history_st_idx)
                # history_ed_idx = st_idx # + 1
                # history_ed_idx = min(kine_qs.shape[0], history_ed_idx)
                # history_st_idx = max(0, history_st_idx)
                history_ed_idx = st_idx # + 1
                # history_ed_idx = min(kine_qs.shape[0], history_ed_idx) # hsitory idxes #
                history_idxes = list(range(history_st_idx, history_ed_idx))
                history_idxes = np.array(history_idxes, dtype=np.int32)
                history_idxes = np.clip(history_idxes, a_min=0, a_max=kine_qs.shape[0] - 1)
                history_kine_qs = kine_qs[history_idxes]
                tot_obj_pose = cur_data_dict['tot_obj_pose'][0]
                history_obj_pose = tot_obj_pose[history_idxes]
                history_obj_pose[:, :3] = history_obj_pose[:, :3] - first_frame_obj_trans[ None]
                history_kine_qs[:, :3] = history_kine_qs[:, :3] - first_frame_obj_trans[ None] # minus the first frame obj trans #
                
                ## the obj eulers may not be a good representation ## ## a representation ## ## # a representation #
                ## TODO: the obj eulers may not be a good representation ## a good representation ##
                # ### from the obj quaternion ornt to the obj euler rotations ##### rotations #####
                history_obj_trans, history_obj_ornt = history_obj_pose[:, :3], history_obj_pose[:, 3:]
                history_obj_rot_euler = []
                for ii_fr in range(history_obj_ornt.shape[0]):
                    cur_fr_obj_ornt = history_obj_ornt[ii_fr]
                    cur_fr_obj_rot_euler = R.from_quat(cur_fr_obj_ornt).as_euler('xyz', degrees=False) # as the rot eulers # 
                    history_obj_rot_euler.append(cur_fr_obj_rot_euler)
                history_obj_rot_euler = np.stack(history_obj_rot_euler, axis= 0) 
                
                # add the history information #
                history_info = {
                    'history_obj_pose': history_obj_pose[None ],
                    'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                    'first_frame_obj_trans': first_frame_obj_trans,
                    'first_frame_obj_ornt': first_frame_obj_ornt, # first frae obj ornt # an the trans # 
                    'history_obj_trans': history_obj_trans[None ], 
                    'history_obj_rot_euler': history_obj_rot_euler[None ],
                }
                cur_slice_data.update(history_info )
                # have he st_idx ? # 
            
            # cur_slice_data_nm = f"{cur_data_fn}_sted_{st_idx}_{ed_idx}"
            cur_slice_data_nm = grab_opt_data_fn.replace(".npy", f"_STED_{st_idx}_{ed_idx}.npy")
            
            if add_to_dict:
                self.data_name_to_data[cur_slice_data_nm] = cur_slice_data #  
            
            self.tot_target_data_nm.append(cur_slice_data_nm)
            if add_to_dict:
                self.data_name_list.append(cur_slice_data_nm)
            
            # slicing tracking kienmatics data ###
            # print(f"[Slicing tracking kine data] data_nm: {data_nm}")
            # sted_info = data_nm.split("/")[-1].split(".npy")[0].split('_STED_')[1]
            # st_idx, ed_idx = sted_info.split("_")
            # st_idx, ed_idx = int(st_idx), int(ed_idx)
            # # else:
            # #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0] # sliced hand qs #
            
            # sliced_hand_qs = kine_data_dict['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
            # sliced_obj_trans = kine_data_dict['obj_trans'][st_idx: ed_idx]
            # sliced_obj_ornt = kine_data_dict['obj_ornt'][st_idx: ed_idx]
            # sliced_obj_rot_euler = kine_data_dict['obj_rot_euler'][st_idx: ed_idx]
            
            # slicing_idxes
            
            sliced_hand_qs = kine_data_dict['hand_qs'][slicing_idxes][:, :self.nn_hands_dof]
            sliced_obj_trans = kine_data_dict['obj_trans'][slicing_idxes]
            sliced_obj_ornt = kine_data_dict['obj_ornt'][slicing_idxes]
            sliced_obj_rot_euler = kine_data_dict['obj_rot_euler'][slicing_idxes]
            
            obj_verts = kine_data_dict['obj_verts']
            
            first_frame_obj_trans = sliced_obj_trans[0, :3]
            sliced_hand_qs[:, :3] = sliced_hand_qs[:, :3] - first_frame_obj_trans[None]
            sliced_obj_trans = sliced_obj_trans - first_frame_obj_trans[None]
            
            kine_info_dict = {
                'obj_verts': obj_verts, 
                'hand_qs': sliced_hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
                'obj_trans': sliced_obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
                'obj_ornt': sliced_obj_ornt ,
                'obj_rot_euler': sliced_obj_rot_euler
            }
            if add_to_dict:
                self.data_name_to_kine_info[cur_slice_data_nm] = kine_info_dict
    # 
    # 
    # 
    def _slicing_mocap_tracking_ctl_data(self, ):
        ## slice the tracking ctl data ## # tracking ctl # use_kine_obj_pos_canonicalization # kine obj pos canonicalization #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        self.all_slices_data_inst_tag_list = []
        self.all_slices_data_name_to_data = {}
        for cur_data_fn in self.data_name_to_data:
            cur_data = self.data_name_to_data[cur_data_fn]
            kine_qs =cur_data['tot_verts'][0]
            q_tars = cur_data['tot_verts_integrated_qdd_tau'][0]
            
            slicing_st_idx = 0 
            if self.task_cond_type == 'history_future':
                slicing_st_idx = 1
        
            slicing_ed_idx = kine_qs.shape[0] - self.slicing_ws
            if slicing_ed_idx < slicing_st_idx + 1:
                slicing_ed_idx = slicing_st_idx + 1 ## increase the slicing ed idx 
            # step size = 30
            # print(f"kine_qs: {kine_qs.shape}, q_tars: {q_tars.shape}")
            for i_slice in range(slicing_st_idx, slicing_ed_idx, self.step_size):
                st_idx = i_slice
                ed_idx = i_slice + self.slicing_ws
                slicing_idxes = list(range(st_idx, ed_idx))
                slicing_idxes = np.array(slicing_idxes, dtype=np.int32) # 
                slicing_idxes = np.clip(slicing_idxes, a_min=0, a_max=kine_qs.shape[0] - 1) #
                # ed_idx = min(i_slice + self.slicing_ws, kine_qs.shape[0])
                
                # task cond should not ## obj slicing rot trans #
                sliced_obj_trans, sliced_obj_ornt = self._get_obj_slicing_rot_trans(cur_data_fn, st_idx, ed_idx)
                
                if sliced_obj_trans.shape[0] == 0:
                    continue
                
                first_frame_obj_trans = sliced_obj_trans[0, :3]
                first_frame_obj_ornt = sliced_obj_ornt[0, :]
                
                # cur_slice_kine_qs = kine_qs[st_idx: ed_idx]
                # cur_slice_q_tars = q_tars[st_idx: ed_idx]
                
                cur_slice_kine_qs = kine_qs[slicing_idxes]
                cur_slice_q_tars = q_tars[slicing_idxes]
                
                #### NOTE: A simple canonicalization ####
                cur_slice_kine_qs[:, :3] = cur_slice_kine_qs[:, :3] - first_frame_obj_trans[ None]
                cur_slice_q_tars[:, :3] = cur_slice_q_tars[:, :3] - first_frame_obj_trans[ None]
                #### NOTE: A simple canonicalization ####
                
                cur_slice_data = {
                    'tot_verts': cur_slice_kine_qs[None],
                    'tot_verts_integrated_qdd_tau': cur_slice_q_tars[None],
                    'slicing_idxes': slicing_idxes,
                }
                
                # TODO: change all the jugdement logic to this # 
                if  self.task_cond_type == 'history_future':
                    # history_st_idx = st_idx - self.slicing_ws
                    history_st_idx = st_idx - self.history_ws
                    # history_st_idx = max(0, history_st_idx)
                    history_ed_idx = st_idx # + 1
                    # history_ed_idx = min(kine_qs.shape[0], history_ed_idx)
                    history_idxes = list(range(history_st_idx, history_ed_idx))
                    history_idxes = np.array(history_idxes, dtype=np.int32)
                    history_idxes = np.clip(history_idxes, a_min=0, a_max=kine_qs.shape[0] - 1) # 
                    # history_kine_qs = kine_qs[history_st_idx : history_ed_idx]
                    history_kine_qs = kine_qs[history_idxes]
                    tot_obj_pose = cur_data['tot_obj_pose'][0]
                    # history_obj_pose = tot_obj_pose[history_st_idx: history_ed_idx]
                    history_obj_pose = tot_obj_pose[history_idxes]
                    history_obj_pose[:, :3] = history_obj_pose[:, :3] - first_frame_obj_trans[ None]
                    history_kine_qs[:, :3] = history_kine_qs[:, :3] - first_frame_obj_trans[ None] # minus the first frame obj trans # # j
                    
                    ## the obj eulers may not be a good representation ## ## a representation ## ## # a representation #
                    ## TODO: the obj eulers may not be a good representation ## a good representation ##
                    # ### from the obj quaternion ornt to the obj euler rotations ##### rotations #####
                    history_obj_trans, history_obj_ornt = history_obj_pose[:, :3], history_obj_pose[:, 3:]
                    history_obj_rot_euler = []
                    for ii_fr in range(history_obj_ornt.shape[0]):
                        cur_fr_obj_ornt = history_obj_ornt[ii_fr]
                        cur_fr_obj_rot_euler = R.from_quat(cur_fr_obj_ornt).as_euler('xyz', degrees=False) # as the rot eulers # 
                        history_obj_rot_euler.append(cur_fr_obj_rot_euler)
                    history_obj_rot_euler = np.stack(history_obj_rot_euler, axis= 0) 
                    
                    # add the history information #
                    history_info = {
                        'history_obj_pose': history_obj_pose[None ],
                        'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                        'first_frame_obj_trans': first_frame_obj_trans,
                        'first_frame_obj_ornt': first_frame_obj_ornt, # first frae obj ornt # an the trans # 
                        'history_obj_trans': history_obj_trans[None ], 
                        'history_obj_rot_euler': history_obj_rot_euler[None ],
                        'history_idxes': history_idxes
                    }
                    cur_slice_data.update(history_info )
                    
                    
                
                # cur_slice_data_nm = f"{cur_data_fn}_sted_{st_idx}_{ed_idx}"
                cur_slice_data_nm = cur_data_fn.replace(".npy", f"_STED_{st_idx}_{ed_idx}.npy")
                self.all_slices_data_name_to_data[cur_slice_data_nm] = cur_slice_data #  
                
                self.all_slices_data_inst_tag_list.append(cur_slice_data_nm)
                tot_data_hand_qs.append(cur_slice_kine_qs)
                tot_data_hand_qtars.append(cur_slice_q_tars)

        tot_data_hand_qs = np.concatenate(tot_data_hand_qs, axis=0)
        tot_data_hand_qtars = np.concatenate(tot_data_hand_qtars, axis=0)
        self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
        self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
        self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
        self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
        
        ### add to data statistics ###
        self.data_statistics['avg_hand_qs'] = self.avg_hand_qs
        self.data_statistics['std_hand_qs'] = self.std_hand_qs
        self.data_statistics['avg_hand_qtars'] = self.avg_hand_qtars
        self.data_statistics['std_hand_qtars'] = self.std_hand_qtars
        
        
        self.data_name_list = self.all_slices_data_inst_tag_list
        self.data_name_to_data = self.all_slices_data_name_to_data
    
    
    
    def _preload_kine_taskcond_data(self, ):
        if self.single_inst:
            self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
        
        
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            kine_data_tag = "passive_active_info_"
            kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
            return kine_object_type
           
        
        def load_traj_info(kine_dict_fn, maxx_ws):
            kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            kine_obj_transl = kine_dict['object_transl']
            kine_obj_rot_quat = kine_dict['object_rot_quat']
            kine_robot_hand_qs = kine_dict['robot_delta_states_weights_np']
            kine_obj_transl = kine_obj_transl[: maxx_ws]
            kine_obj_rot_quat = kine_obj_rot_quat[: maxx_ws]
            kine_robot_hand_qs = kine_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            kine_obj_rot_euler_angles = []
            for i_fr in range(kine_obj_rot_quat.shape[0]):
                cur_rot_quat = kine_obj_rot_quat[i_fr] #
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
                kine_obj_rot_euler_angles.append(cur_rot_euler)
            kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            return kine_obj_transl, kine_obj_rot_euler_angles, kine_robot_hand_qs
            
        def load_obj_pcs(kine_dict_fn, nn_sampled_pts):
            object_type = parse_kine_data_fn_into_object_type(kine_dict_fn)
            # kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            # kine_obj_transl = k
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes][:nn_sampled_pts]
            # tot_object_verts.append(obj_verts)/
            return obj_verts
        
        # task inheriting dict info ## inheritng dict info #
        maxx_ws = 149
        for i_data, data_dict in enumerate(self.task_inheriting_dict_info):
            # cur_inheriting_dict = {
            #     'fa_objtype': cur_parent_objtype, # 
            #     'fa_trajtype': cur_parent_trajtype, # 
            #     'ch_objtype': cur_child_objtype, # 
            #     'ch_trajtype': cur_child_trajtype
            # }
            cur_fa_objtype = data_dict['fa_objtype']
            cur_fa_trajtype = data_dict['fa_trajtype']
            cur_ch_objtype = data_dict['ch_objtype']
            cur_ch_trajtype = data_dict['ch_trajtype']
            
            ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_trajtype]
            fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_trajtype]
            
            ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_objtype]
            fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_objtype]
            
            ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
            fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
            
            ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
            fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
            
            
            
            # 
            sv_dict = {
                'obj_verts': fa_obj_verts,
                'robot_hand_qs': fa_robot_hand_qs,
                'obj_rot_euler': fa_obj_rot_euler,
                'obj_transl': fa_obj_transl
            }
            cond_sv_dict = {
                'cond_obj_verts': ch_obj_verts,
                'cond_robot_hand_qs': ch_robot_hand_qs,
                'cond_obj_rot_euler': ch_obj_rot_euler,
                'cond_obj_transl': ch_obj_transl
            }
            sv_dict.update(cond_sv_dict)
            
            
            for key in sv_dict:
                print(f"key: {key}, val: {sv_dict[key].shape }")
            
            # self.data_name_to_data[cur_kine_data_fn] = sv_dict
            self.tot_data_dict_list.append(sv_dict)
            
            # object_type = parse_kine_data_fn_into_object_type(cur_kine_data_fn)
            # self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            # grab_mesh_fn = f"{object_type}.obj"
            # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            
            # obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            # obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            # obj_verts = obj_verts[random_sampled_idxes][:maxx_ws]
            # tot_object_verts.append(obj_verts)
            
            # ch_kine_traj_dict = np.load(ch_kine_traj_dict_fn, allow_pickle=True).item()
            # fa_kine_traj_dict = np.load(fa_kine_traj_dict_fn, allow_pickle=True).item()
            
            # ch_obj_transl = ch_kine_traj_dict['object_transl']
            # ch_obj_rot_quat = ch_kine_traj_dict['object_rot_quat']
            # ch_robot_hand_qs = ch_kine_traj_dict['robot_delta_states_weights_np']
            # maxx_ws = 150
            # ch_obj_transl = ch_obj_transl[: maxx_ws]
            # ch_obj_rot_quat = ch_obj_rot_quat[: maxx_ws]
            # ch_robot_hand_qs = ch_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            
            # ch_obj_rot_euler_angles = []
            # for i_fr in range(ch_obj_rot_quat.shape[0]):
            #     cur_rot_quat = ch_obj_rot_quat[i_fr]
            #     cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True) # get the rotation euler 
            #     ch_obj_rot_euler_angles.append(cur_rot_euler)
            # ch_obj_rot_euler_angles = np.stack(ch_obj_rot_euler_angles, axis=0)
    
    
    def _preload_kine_target_taskcond_data(self, ):
        # if self.single_inst:
        #     self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
        
        # tot_obj_transl = []
        # tot_obj_rot_euler = []
        # tot_hand_qs = []
        # tot_obj_verts = []
        
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            kine_data_tag = "passive_active_info_"
            kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
            return kine_object_type
           
        
        def load_traj_info(kine_dict_fn, maxx_ws):
            kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            kine_obj_transl = kine_dict['object_transl']
            kine_obj_rot_quat = kine_dict['object_rot_quat']
            kine_robot_hand_qs = kine_dict['robot_delta_states_weights_np']
            kine_obj_transl = kine_obj_transl[: maxx_ws]
            kine_obj_rot_quat = kine_obj_rot_quat[: maxx_ws]
            kine_robot_hand_qs = kine_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            kine_obj_rot_euler_angles = []
            for i_fr in range(kine_obj_rot_quat.shape[0]):
                cur_rot_quat = kine_obj_rot_quat[i_fr] # dkine obj rot quat #
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
                kine_obj_rot_euler_angles.append(cur_rot_euler)
            kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            return kine_obj_transl, kine_obj_rot_euler_angles, kine_robot_hand_qs
            
        def load_obj_pcs(kine_dict_fn, nn_sampled_pts):
            object_type = parse_kine_data_fn_into_object_type(kine_dict_fn)
            # kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            # kine_obj_transl = k
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes][:nn_sampled_pts]
            # tot_object_verts.append(obj_verts)/
            return obj_verts
        
        
        maxx_ws = 149
        
        ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        
        ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        
        ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
        fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
        
        ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
        fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
        
    # 
        sv_dict = {
            'obj_verts': fa_obj_verts,
            'robot_hand_qs': fa_robot_hand_qs,
            'obj_rot_euler': fa_obj_rot_euler,
            'obj_transl': fa_obj_transl
        }
        cond_sv_dict = {
            'cond_obj_verts': ch_obj_verts,
            'cond_robot_hand_qs': ch_robot_hand_qs,
            'cond_obj_rot_euler': ch_obj_rot_euler,
            'cond_obj_transl': ch_obj_transl
        }
        sv_dict.update(cond_sv_dict)
        
        
        self.tot_data_dict_list = []
        self.tot_data_dict_list.append(sv_dict)

    
    
    def _preload_optimized_res_data(self, ):
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        self.joint_idxes_ordering = joint_idxes_ordering
        # self.joint_idxes_ordering_th = torch.from_numpy(joint_idxes_ordering).long().to(self.rl_device)
        # self.inversed_joint_idxes_ordering_th = torch.from_numpy(joint_idxes_inversed_ordering).long().to(self.rl_device)
        
        # 
        # excuted_inst_tag_to_optimized_res
        inst_tag_to_optimized_res = np.load(self.excuted_inst_tag_to_optimized_res, allow_pickle=True).item()
        # get the cur inst tag #
        # get the inst tag to the optimized res # get the current inst tag #
        # getthe inst tag #
        # sv_dict = {
        #     'obj_verts': obj_verts,
        #     'robot_hand_qs': robot_hand_qs,
        #     'obj_rot_euler': obj_rot_euler_angles,
        #     'obj_transl': obj_transl,
        #     'object_type': object_type,
        #     'obj_rot_quat': obj_rot_quat
        # }
        # if self.diff_contact_sequence:
        #     sv_dict.update(
        #         {
        #             'contact_data': cur_inst_contact_data
        #         }
        #     )
        #     tot_contact_data.append(cur_inst_contact_data)
        # self.data_name_to_data[cur_kine_data_fn] = sv_dict
        # data name to data #
        
        obj_pos_diff_thres = 0.10 # 10 cm 
        obj_rot_diff_thres = 0.6981317007977318 # 40 degree 
        
        for cur_inst_tag in inst_tag_to_optimized_res:
            cur_inst_optimized_res_list = inst_tag_to_optimized_res[cur_inst_tag]
            if isinstance(cur_inst_tag, tuple):
                cur_obj_inst_type = cur_inst_tag[0]
            else:
                cur_obj_inst_type = cur_inst_tag
            # print(f"cur_inst_optimized_res_list: {cur_inst_optimized_res_list}")
            for i_opt_res, cur_opt_res_fn in enumerate(cur_inst_optimized_res_list):
                print(f"[{i_opt_res}/{len(cur_inst_optimized_res_list)}] cur_opt_res_fn: {cur_opt_res_fn}")
                # cur_opt_res = np.load(cur_opt_res_fn, allow_pickle=True).item() # load the optimized res #
                # cur opt res #
                cur_opt_res_best_fn = cur_opt_res_fn.replace(".npy", "_sorted_best.npy")
                if not os.path.exists(cur_opt_res_best_fn):
                    continue
                cur_opt_res_best_dict = np.load(cur_opt_res_best_fn, allow_pickle=True).item() # best opt res fn #
                # best obj pose; best hand qtars # # best hand qs # 
                cur_best_obj_pose_diff = cur_opt_res_best_dict['best_obj_pose_diff']
                cur_best_obj_pos_diff = cur_opt_res_best_dict['best_obj_pos_diff'][0].item()
                cur_best_obj_rot_diff = cur_opt_res_best_dict['best_obj_rot_diff'][0].item() # 
                # 
                if cur_best_obj_pose_diff < obj_pos_diff_thres and cur_best_obj_rot_diff < obj_rot_diff_thres:
                    #  get the res best fn # 
                    # save the data name to the  
                    # 'optimized_obj_pose': best_obj_pose,
                    # 'optimized_hand_qtars': best_hand_qtars,
                    # 'optimized_hand_qs': best_hand_qs,
                    optimized_obj_pose = cur_opt_res_best_dict['optimized_obj_pose'][0] # get the optimized obj pose 
                    optimized_hand_qs = cur_opt_res_best_dict['optimized_hand_qs'][0] # get the optimized hand qs 
                    optimized_hand_qs = optimized_hand_qs[..., self.joint_idxes_ordering] # reorder the joint idxes #
                    # optimized hand qs #
                    # cur opt res best fn #
                    self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
                    pure_inst_tag = cur_obj_inst_type.split("_nf_")[0]
                    grab_mesh_fn = f"{pure_inst_tag}.obj"
                    grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                    
                    
                    obj_mesh = trimesh.load_mesh(grab_mesh_fn)
                    obj_verts = obj_mesh.vertices # nn_pts x 3
                    random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
                    obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
                    # tot_object_verts.append(obj_verts)
                    
                    obj_transl = optimized_obj_pose[..., :3]
                    obj_rot_quat = optimized_obj_pose[..., 3:]
                    
                    # 'obj_transl': obj_transl[::self.inv_kine_freq],
                    # 'obj_rot_quat': obj_rot_quat[::self.inv_kine_freq],
                    # 'robot_hand_qs': robot_hand_qs[::self.inv_kine_freq],
                    if cur_obj_inst_type in self.inst_tag_to_kine_traj_data:
                        
                        cur_inst_kine_data = self.inst_tag_to_kine_traj_data[cur_obj_inst_type]
                    else:
                        print(f"self.inst_tag_to_kine_traj_data: {self.inst_tag_to_kine_traj_data.keys()}")
                        cur_inst_kine_data = self.inst_tag_to_kine_traj_data[list(self.inst_tag_to_kine_traj_data.keys())[0]]
                    
                    
                    
                    sv_dict = {
                        'obj_verts': obj_verts,
                        # 'robot_hand_qs': optimized_hand_qs[::self.inv_kine_freq],
                        # 'obj_transl': obj_transl[::self.inv_kine_freq],
                        # 'obj_rot_quat': obj_rot_quat[::self.inv_kine_freq],
                        'robot_hand_qs': optimized_hand_qs, # [::self.inv_kine_freq],
                        'obj_transl': obj_transl, # [::self.inv_kine_freq],
                        'obj_rot_quat': obj_rot_quat, # [::self.inv_kine_freq],
                        'object_type': cur_obj_inst_type,
                        'kine_obj_transl': cur_inst_kine_data['obj_transl'],
                        'kine_obj_rot_quat': cur_inst_kine_data['obj_rot_quat'],
                        'kine_robot_hand_qs': cur_inst_kine_data['robot_hand_qs']
                        # inst_tag_to_kine_traj_data
                        
                    }
                    
                    # cur_inst_tag # leap_passive_active_info_ori_grab_s2_camera_browse_1_nf_300.npy # 
                    # cur_inst_tag_segs = cur_inst_tag.split("_")[6:8]
                    
                    if self.text_feature_version == 'v1':
                        cur_inst_tag_segs = cur_obj_inst_type.split("_")[3:5]
                    elif self.text_feature_version == 'v2':
                        cur_inst_tag_segs = cur_obj_inst_type.split("_")[2:5]
                    else:
                        raise ValueError(f"Invalid text feature version: {self.text_feature_version}")
                    
                    # cur inst tag segs #
                    cur_inst_motion_tag = " ".join(cur_inst_tag_segs)
                    
                    
                    if self.use_clip_glb_features:
                        text_inputs = clip.tokenize([cur_inst_motion_tag]).to(clip_device)
                        with torch.no_grad():
                            text_features = self.clip_model.encode_text(text_inputs)
                        text_features /= text_features.norm(dim=-1, keepdim=True)
                        text_features = text_features[0].cpu().numpy()
                        sv_dict['text_features'] = text_features
    
                    
                    self.data_name_to_data[cur_opt_res_best_fn] = sv_dict
                    pass
                
                
        
        
    def _preload_kine_data(self,) :
        if self.single_inst or self.debug:
            self.data_list = self.data_list[:1]

        def parse_kine_data_fn_into_object_type(kine_data_fn):
            if 'taco' in kine_data_fn:
                passive_act_pure_tag = "passive_active_info_ori_grab_s2_phone_call_1_interped_"
                
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
        
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        # obj verts; # contact data #
        tot_object_verts = []
        tot_contact_data = []
        
        # 
        print(f"Start loading kinematics data: {len(self.data_list)}")
        for i_kine, kine_fn in enumerate(self.data_list):
            print(f"[{i_kine}/{len(self.data_list)}] {kine_fn}")
            # kine saved info ## dict keys ##
            # kine_saved_info: dict_keys(['passive_meshes', 'active_meshes', 'passive_mesh_normals', 'object_transl', 'object_rot_quat', 'ts_to_allegro', 'ts_to_mano_fingers_np', 'ts_to_robot_fingers_np', 'robot_delta_states_weights_np']) 
            cur_kine_data_fn = self.data_list[i_kine]
            
            if self.load_n_data:
                self.diff_contact_sequence = False
                nn_fr = 300
                nn_hand_dof = 22
                obj_transl = np.random.randn(nn_fr, 3)
                obj_rot_quat = np.random.randn(nn_fr, 4)
                robot_hand_qs = np.random.randn(nn_fr, nn_hand_dof)
            else:
                # print(f"cur_kine_data_fn: {cur_kine_data_fn}")
                cur_kine_data = np.load(cur_kine_data_fn, allow_pickle=True).item()
                
                # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag/passive_active_info_ori_grab_s1_alarmclock_pass_1_nf_300_contact_flag.npy
                if self.diff_contact_sequence:
                    pure_kine_data_fn = cur_kine_data_fn.split("/")[-1].split(".")[0] # split #
                    pure_contact_data_fn = pure_kine_data_fn + "_contact_flag.npy"
                    full_contact_data_fn = os.path.join(self.contact_info_sv_root, pure_contact_data_fn) # contact data fn # 
                    cur_inst_contact_data = np.load(full_contact_data_fn)
                    # nn_timesteps x nn_contact_dim # 
                    ## TODO: get the contact dim here ### cur inst contact tag #
                
                obj_transl = cur_kine_data['object_transl']
                obj_rot_quat = cur_kine_data['object_rot_quat']
                robot_hand_qs = cur_kine_data['robot_delta_states_weights_np']
            
            maxx_ws = 400
            maxx_ws = min(maxx_ws, obj_transl.shape[0])
            maxx_ws = min(maxx_ws, robot_hand_qs.shape[0])
            obj_transl = obj_transl[: maxx_ws]
            obj_rot_quat = obj_rot_quat[: maxx_ws]
            robot_hand_qs = robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            
            if self.load_n_data:
                obj_rot_euler_angles = np.random.randn(nn_fr, 3)
            else:
                # transform the rot_quat # 
                obj_rot_euler_angles = []
                for i_fr in range(obj_rot_quat.shape[0]):
                    cur_rot_quat = obj_rot_quat[i_fr]
                    cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False)
                    obj_rot_euler_angles.append(cur_rot_euler)
                obj_rot_euler_angles = np.stack(obj_rot_euler_angles, axis=0)
            
            # obj_transl and obj_rot_euler_angles # 
            tot_obj_transl.append(obj_transl)
            tot_obj_rot_euler.append(obj_rot_euler_angles)
            tot_hand_qs.append(robot_hand_qs)
            
            
            if self.load_n_data:
                object_type = 'ori_grab_s9_apple_lift'
            else:
        
                object_type = parse_kine_data_fn_into_object_type(cur_kine_data_fn)
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            
            
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            tot_object_verts.append(obj_verts)
            
            # sv_dict = {
            #     'obj_verts': obj_verts,
            #     'robot_hand_qs': robot_hand_qs[::self.inv_kine_freq],
            #     'obj_rot_euler': obj_rot_euler_angles[::self.inv_kine_freq],
            #     'obj_transl': obj_transl[::self.inv_kine_freq],
            #     'object_type': object_type,
            #     'obj_rot_quat': obj_rot_quat[::self.inv_kine_freq]
            # }
            
            sv_dict = {
                'obj_verts': obj_verts,
                'robot_hand_qs': robot_hand_qs,
                'obj_rot_euler': obj_rot_euler_angles,
                'obj_transl': obj_transl,
                'object_type': object_type,
                'obj_rot_quat': obj_rot_quat
            }
            
            # diff contact sequence #
            if self.diff_contact_sequence:
                sv_dict.update(
                    {
                        'contact_data': cur_inst_contact_data
                    }
                )
                tot_contact_data.append(cur_inst_contact_data)
                
            if self.load_n_data:
                cur_inst_motion_tag = "s1 alarmclock pass"
            else:
                # cur_inst_tag # leap_passive_active_info_ori_grab_s2_camera_browse_1_nf_300.npy # 
                if self.text_feature_version == 'v1':
                    cur_inst_tag_segs = cur_kine_data_fn.split("/")[-1].split(".npy")[0].split("_")[6:8]
                elif self.text_feature_version == 'v2':
                    cur_inst_tag_segs = cur_kine_data_fn.split("/")[-1].split(".npy")[0].split("_")[5:8]
                else:
                    raise ValueError(f"Invalid text feature version: {self.text_feature_version}")
                cur_inst_motion_tag = " ".join(cur_inst_tag_segs)
            print(f"cur_inst_motion_tag = {cur_inst_motion_tag}")
            
            if self.use_clip_glb_features: # clip global features # 
                text_inputs = clip.tokenize([cur_inst_motion_tag]).to(clip_device)
                with torch.no_grad():
                    text_features = self.clip_model.encode_text(text_inputs)
                text_features /= text_features.norm(dim=-1, keepdim=True)
                text_features = text_features[0].cpu().numpy()
                sv_dict['text_features'] = text_features
            
            self.data_name_to_data[cur_kine_data_fn] = sv_dict
            
            if self.load_n_data:
                cur_inst_tag = cur_kine_data_fn
            else:
                passive_act_st_tag = "passive_active_info_"
                cur_inst_tag = cur_kine_data_fn.split("/")[-1].split(".npy")[0][len(passive_act_st_tag): ]
            self.inst_tag_to_kine_traj_data[cur_inst_tag] = {
                # 'obj_transl': obj_transl[::self.inv_kine_freq],
                # 'obj_rot_quat': obj_rot_quat[::self.inv_kine_freq],
                # 'robot_hand_qs': robot_hand_qs[::self.inv_kine_freq],
                'obj_transl': obj_transl, # [::self.inv_kine_freq],
                'obj_rot_quat': obj_rot_quat, # [::self.inv_kine_freq],
                'robot_hand_qs': robot_hand_qs # [::self.inv_kine_freq],
            }
        
        
        tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        
        
        
        if not (self.sampling and len(self.target_grab_inst_tag) > 0):
            self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
            self.std_obj_transl = np.std(tot_obj_transl, axis=0)
            self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
            self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
            self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
            self.std_hand_qs = np.std(tot_hand_qs, axis=0)
            # 
            self.avg_object_verts = np.mean(tot_object_verts, axis=0)
            self.std_object_verts = np.std(tot_object_verts, axis=0)
            
            
            self.data_statistics = {
                'avg_obj_transl': self.avg_obj_transl, 
                'std_obj_transl': self.std_obj_transl,
                'avg_obj_rot_euler': self.avg_obj_rot_euler,
                'std_obj_rot_euler': self.std_obj_rot_euler,
                'avg_obj_verts': self.avg_object_verts,
                'std_obj_verts': self.std_object_verts,
                'avg_hand_qs': self.avg_hand_qs, 
                'std_hand_qs': self.std_hand_qs
            }
            
        if self.diff_contact_sequence:
            tot_contact_data = np.concatenate(tot_contact_data, axis=0)
            avg_contact_data = np.mean(tot_contact_data, axis=0)
            std_contact_data = np.std(tot_contact_data, axis=0)
            self.avg_contact_data = avg_contact_data
            self.std_contact_data = std_contact_data
            self.data_statistics['avg_contact_data'] = avg_contact_data
            self.data_statistics['std_contact_data'] = std_contact_data
        
        
    
    def _preload_inheriting_data(self, ):
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
            
        for cur_inherit_info in self.tot_inheriting_infos:
            cur_fa_task_fn = cur_inherit_info['inherit_fr_pts_info_fn']
            cur_ch_task_fn = cur_inherit_info['to_task_pts_info_fn']
            
            ## fa and ch ##
            cur_fa_data = self._load_data_from_data_name(cur_fa_task_fn)
            cur_ch_data = self._load_data_from_data_name(cur_ch_task_fn)
            
            cur_ch_task_setting = [
                float(self.obj_name_to_idx[cur_ch_data['object_type']]) - 0.5, float(cur_ch_data['task_rot']), float(cur_ch_data['object_size_x'])
            ]
            cur_fa_task_setting = [
                float(self.obj_name_to_idx[cur_fa_data['object_type']]) - 0.5, float(cur_fa_data['task_rot']), float(cur_fa_data['object_size_x'])
            ]
            cur_inheri_data = {
                'fa_task_setting': cur_fa_task_setting, 
                'ch_task_setting' : cur_ch_task_setting
            }
            self.data_name_to_data[cur_ch_task_fn] = cur_inheri_data
    
    def _preload_data(self, ):
        
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        
        self.tot_object_type = []
        self.tot_rot = []
        self.tot_obj_size = []
        for data_nm in self.data_name_list:
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            ### training setting ###
            if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond']:
                unopt_verts = cur_data['unopt_verts']
                tot_verts = unopt_verts
                unopt_qtar_verts = cur_data['unopt_qtar_verts']
                opt_verts = cur_data['opt_verts']
                opt_qtar_verts = cur_data['opt_qtar_verts']
                
                ## unopt verts ##
                cur_clip_data = {
                    'unopt_verts': unopt_verts,
                    'unopt_qtar_verts': unopt_qtar_verts,
                    'opt_verts': opt_verts,
                    'opt_qtar_verts': opt_qtar_verts
                }
                cur_data_nm = data_nm
                self.data_name_to_data[data_nm] = cur_clip_data
            
            
            else: 
                if self.use_jointspace_seq:
                    ts_to_hand_qs = cur_data['ts_to_hand_qs']
                    # ts_to_hand_qtars = cur_data['ts_to_qtars'] 
    
                    ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
                    # ts_to_
                    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                     
                    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
                    ctl_freq_tss = sorted(ctl_freq_tss)
                    ctl_freq = 10
                    ctl_freq_tss_expanded = [ min(500 - 1, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
                    ts_to_hand_qs = {
                        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
                    }
                    # nn_ts x 
                    # tot_qs = tot_qs[None]
                    ts_keys_sorted = sorted(list(ts_to_hand_qs.keys()))
                    hand_qs_np = [
                        ts_to_hand_qs[cur_ts] for cur_ts in ts_to_hand_qs
                    ]
                    hand_qtars_np = [
                        ts_to_hand_qtars[cur_ts] for cur_ts in ts_to_hand_qtars
                    ]
                    hand_qs_np = np.stack(hand_qs_np, axis=0)
                    hand_qtars_np = np.stack(hand_qtars_np, axis=0) ## tethte qtarsnp 
                    
                    # hand_qs_np = hand_qs_np[]
                    
                    cur_data_nm = data_nm
                    
                    task_setting = {
                        'object_type': self.obj_name_to_idx[cur_data['object_type']],
                        'task_rot': cur_data['task_rot'],
                        'object_size_x': cur_data['object_size_x']
                    }
                    
                    self.tot_object_type.append(task_setting['object_type'])
                    self.tot_rot.append(task_setting['task_rot'])
                    self.tot_obj_size.append(task_setting['object_size_x']) ## get object size x ##
                    
                    
                    cur_clip_data = {
                        'tot_verts': hand_qs_np[None], 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': hand_qtars_np[None],
                        'task_setting': task_setting
                        # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                
                else:
                    # selected_frame_verts, selected_frame_qtars_verts
                    # tot_verts tot_verts_integrated_qdd_tau
                    ## convert to the interested first frame's pose ## then chnage the pose of that data ##
                    
                    tot_verts = cur_data['tot_verts']
                    
                    if self.use_static_first_frame:
                        tot_verts_first_frame = tot_verts[:, 0:1]
                        tot_verts = np.repeat(tot_verts_first_frame, tot_verts.shape[1], axis=1)
                    # print(f"tot_verts: {}")
                    
                    tot_verts_integrated_qdd_tau = cur_data['tot_qtar_verts']
                    if 'tot_qtar_verts_s2' not in cur_data:
                        tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts'].copy()
                    else:
                        tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts_s2']
                    
                    
                    # nn_ts x nn_verts x 3 #
                    # sequence length ? # # tot verts # #
                    self.nn_seq_len = tot_verts.shape[1]
                    
                    ##### inspect ######
                    mean_tot_verts = np.mean(tot_verts, axis=1)
                    mean_tot_verts = np.mean(mean_tot_verts, axis=0)
                    
                    mean_tot_verts_qdd = np.mean(tot_verts_integrated_qdd_tau, axis=1)
                    mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
                    
                    mean_tot_verts_qdd_s2 = np.mean(tot_verts_integrated_qdd_tau_s2, axis=1)
                    mean_tot_verts_qdd_s2 = np.mean(mean_tot_verts_qdd_s2, axis=0)
                    
                    print(f"mean_tot_verts: {mean_tot_verts}, mean_tot_verts_qdd: {mean_tot_verts_qdd}, mean_tot_verts_qdd_s2: {mean_tot_verts_qdd_s2}")
                    ##### inspect ######
                    
                    cur_data_nm = data_nm
                    cur_clip_data = {
                        'tot_verts': tot_verts, 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': tot_verts_integrated_qdd_tau,
                        'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                
            
            ###### not use jointspace seq ######
            if not self.use_jointspace_seq:
                init_verts = tot_verts[:, 0]
                particle_init_xs_th = torch.from_numpy(init_verts).float()
                
                if self.specified_sampled_particle_idxes_fn is not None and len(self.specified_sampled_particle_idxes_fn) > 0:
                    sampled_particle_idxes_sv_fn = self.specified_sampled_particle_idxes_fn
                else:
                    if 'allegro_flat_fivefin_yscaled_finscaled' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_finscaled_sampled_particle_idxes.npy")
                    elif 'allegro_flat_fivefin_yscaled' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_sampled_particle_idxes.npy")
                    elif 'allegro_flat_fivefin' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_sampled_particle_idxes.npy")
                    else:
                        ## al
                        ### get the particle idxes  ###
                        # get partcle init xs #
                        sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                
                if not os.path.exists(sampled_particle_idxes_sv_fn):
                    sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)
                else:
                    sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True)
                
                self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
        if self.use_jointspace_seq:
            self.tot_object_type = np.array(self.tot_object_type, dtype=np.float32)  ### (nn_instances, )
            self.tot_rot = np.array(self.tot_rot, dtype=np.float32)
            self.tot_obj_size = np.array(self.tot_obj_size, dtype=np.float32)
            self.avg_obj_type = np.mean(self.tot_object_type)
            self.avg_rot = np.mean(self.tot_rot)
            self.avg_obj_size = np.mean(self.tot_obj_size)
            self.std_obj_type = np.std(self.tot_object_type)
            self.std_rot = np.std(self.tot_rot)
            self.std_obj_size = np.std(self.tot_obj_size)
            
            self.avg_task_setting = np.array(
                [self.avg_obj_type, self.avg_rot, self.avg_obj_size], dtype=np.float32
            )
            self.std_task_setting = np.array(
                [self.std_obj_type, self.std_rot, self.std_obj_size], dtype=np.float32
            )
                
        
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")
        self.data_name_list = list(self.data_name_to_data.keys()) # data name to data # 
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_kine_v2(self, data_dict):
        # eps = 1e-6
        # obj_verts = (obj_verts - self.avg_obj_verts[None][None]) / (self.std_obj_verts[None][None] + eps)
        
        eps = 1e-6
        # inv scale data kine info #
        data_X = data_dict['X']
        data_E = data_dict['E']
        
        avg_obj_verts_th = torch.from_numpy(self.avg_object_verts).float().cuda()
        std_obj_verts_th = torch.from_numpy(self.std_object_verts).float().cuda()
        avg_hand_qs_th = torch.from_numpy(self.avg_hand_qs).float().cuda()
        std_hand_qs_th = torch.from_numpy(self.std_hand_qs).float().cuda()
        avg_obj_rot_euler_th = torch.from_numpy(self.avg_obj_rot_euler).float().cuda()
        std_obj_rot_euler_th = torch.from_numpy(self.std_obj_rot_euler).float().cuda()
        avg_obj_transl_th = torch.from_numpy(self.avg_obj_transl).float().cuda()
        std_obj_transl_th = torch.from_numpy(self.std_obj_transl).float().cuda()
        
        
        data_E = data_E[:, 0, :, :]
        dec_hand_qs = data_E[:, :, : self.nn_hands_dof]
        dec_obj_transl = data_E[:, :, self.nn_hands_dof: self.nn_hands_dof + 3]
        dec_obj_rot_euler = data_E[:, :, self.nn_hands_dof + 3: ]
        
        inv_scaled_hand_qs = (dec_hand_qs * (std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_transl = (dec_obj_transl * (std_obj_transl_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_transl_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_rot_euler = (dec_obj_rot_euler * (std_obj_rot_euler_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_rot_euler_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_verts = (data_X * (std_obj_verts_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_verts_th.unsqueeze(0).unsqueeze(0)
        
        
        if self.task_cond:
            cond_data_X = data_dict['X_cond']
            cond_data_E = data_dict['E_cond']
            cond_data_E = cond_data_E[:, 0, :, :]
            
            dec_cond_hand_qs = cond_data_E[:, :, : self.nn_hands_dof]
            dec_cond_obj_transl = cond_data_E[:, :, self.nn_hands_dof: self.nn_hands_dof + 3]
            dec_cond_obj_rot_euler = cond_data_E[:, :, self.nn_hands_dof + 3: ]
            
            inv_scaled_cond_hand_qs = (dec_cond_hand_qs * (std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
            inv_scaled_cond_obj_transl = (dec_cond_obj_transl * (std_obj_transl_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_transl_th.unsqueeze(0).unsqueeze(0)
            inv_scaled_cond_obj_rot_euler = (dec_cond_obj_rot_euler * (std_obj_rot_euler_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_rot_euler_th.unsqueeze(0).unsqueeze(0)
            
            inv_scaled_cond_obj_verts = (cond_data_X * (std_obj_verts_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_verts_th.unsqueeze(0).unsqueeze(0)
            cond_rt_dict = {
                'cond_obj_verts': inv_scaled_cond_obj_verts,
                'cond_hand_qs': inv_scaled_cond_hand_qs,
                'cond_obj_transl': inv_scaled_cond_obj_transl,
                'cond_obj_rot_euler': inv_scaled_cond_obj_rot_euler
            }
            
            # dec_obj_transl = cond_data_E[:, :, :3]
            
            
        
        # obj_verts_avg_th = torch.from_numpy(self.avg_obj_verts).float().cuda()
        # obj_verts_std_th = torch.from_numpy(self.std_obj_verts).float().cuda() ## get the avg and std object vertices # 
        # # (3,) - dim obj_verts_avg and obj_verts_std # 
        # data_E = (obj_verts_std_th.unsqueeze(0).unsqueeze(0).unsqueeze(0) + eps ) * data_E + obj_verts_avg_th.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # tot_verts = data_X
        # tot_verts_integrated_qdd_tau = data_E 
        rt_dict = {
            # 'tot_verts': data_X,
            # 'tot_verts_integrated_qdd_tau': data_E,
            # 'tot_verts_integrated_qdd_tau_s2': data_E # inv scaled #
            'obj_verts': inv_scaled_obj_verts,
            'hand_qs': inv_scaled_hand_qs,
            'obj_transl': inv_scaled_obj_transl,
            'obj_rot_euler': inv_scaled_obj_rot_euler
        }
        
        if self.task_cond:
            rt_dict.update(
                cond_rt_dict # get the cond rt dict #
            )
            
        for key in data_dict:
            if key not in rt_dict:
                rt_dict[key] = data_dict[key]
        
        return rt_dict
        
        
    
    def inv_scale_data_v2(self, data_dict, data_nm=None, notarget=False): # bsz x nn_particles x feat_dim #
        eps = 1e-6
        if self.canonicalize_obj_pts:
            
            
            if self.scale_clip_data:
                avg_tot_clip_hand_qs_th = torch.from_numpy(self.avg_tot_clip_hand_qs).float().cuda().unsqueeze(0).unsqueeze(0)
                std_tot_clip_hand_qs_th = torch.from_numpy(self.std_tot_clip_hand_qs).float().cuda().unsqueeze(0).unsqueeze(0)
                avg_tot_clip_obj_pos_th = torch.from_numpy(self.avg_tot_clip_obj_pos).float().cuda().unsqueeze(0).unsqueeze(0)
                std_tot_clip_obj_pos_th = torch.from_numpy(self.std_tot_clip_obj_pos).float().cuda().unsqueeze(0).unsqueeze(0)
                
                if 'E' in data_dict:
                    E = data_dict['E']
                    if E.size(-1) == 29:
                        pred_hand_pose, pred_obj_pos, pred_obj_ornt = E[..., :22], E[..., 22: 22 + 3], E[..., 22 + 3: ]
                        pred_hand_pose = ( pred_hand_pose * (std_tot_clip_hand_qs_th + eps) ) + avg_tot_clip_hand_qs_th
                        pred_obj_pos = ( pred_obj_pos * (std_tot_clip_obj_pos_th + eps) ) + avg_tot_clip_obj_pos_th
                        E[..., :22] = pred_hand_pose
                        E[..., 22: 22 + 3] = pred_obj_pos
                    elif E.size(-1) == 22:
                        E = (E * (std_tot_clip_hand_qs_th + eps)) + avg_tot_clip_hand_qs_th
                    elif E.size(-1) == 3:
                        E = (E * (std_tot_clip_obj_pos_th + eps)) + avg_tot_clip_obj_pos_th
                    elif E.size(-1) == 7:
                        E[..., :3] = (E[..., :3] * (std_tot_clip_obj_pos_th + eps)) + avg_tot_clip_obj_pos_th
                    data_dict['E'] = E
                else:
                    obj_pos = data_dict['obj_pos']
                    future_obj_pos = data_dict['future_obj_pos']
                    hand_pose = data_dict['hand_pose']
                    future_hand_pose = data_dict['future_hand_pose']
                    glb_hand_pose = data_dict['glb_hand_pose']
                    glb_obj_pos = data_dict['glb_obj_pos']
                    obj_pos = (obj_pos * (std_tot_clip_obj_pos_th + eps)) + avg_tot_clip_obj_pos_th
                    future_obj_pos = (future_obj_pos * (std_tot_clip_obj_pos_th + eps)) + avg_tot_clip_obj_pos_th
                    # hand_pose[..., :3] = (hand_pose[..., :3] * (std_tot_clip_hand_qs_th + eps)) + avg_tot_clip_hand_qs_th
                    hand_pose = (hand_pose * (std_tot_clip_hand_qs_th + eps)) + avg_tot_clip_hand_qs_th
                    future_hand_pose = (future_hand_pose * (std_tot_clip_hand_qs_th + eps)) + avg_tot_clip_hand_qs_th
                    glb_hand_pose = (glb_hand_pose * (std_tot_clip_hand_qs_th + eps)) + avg_tot_clip_hand_qs_th
                    glb_obj_pos = (glb_obj_pos * (std_tot_clip_obj_pos_th + eps)) + avg_tot_clip_obj_pos_th
                    data_dict.update(
                        {
                            'obj_pos': obj_pos,
                            'future_obj_pos': future_obj_pos,
                            'hand_pose': hand_pose,
                            'future_hand_pose': future_hand_pose,
                            'glb_hand_pose': glb_hand_pose,
                            'glb_obj_pos': glb_obj_pos
                        }
                    )      
            else: 
                # print(f"data_dict: {data_dict.keys()}")
                # data_dict_rt = self.scale_data_kine_new(data_dict_rt)
                if self.partial_obj_pos_info:
                    avg_obj_transl_th = torch.from_numpy(self.avg_obj_transl).float().cuda().unsqueeze(0).unsqueeze(0)
                    std_obj_transl_th = torch.from_numpy(self.std_obj_transl).float().cuda().unsqueeze(0).unsqueeze(0)
                    if 'E' in data_dict:
                        E = data_dict['E']
                        # print(f"E: {E.size()}, std_obj_transl_th: {std_obj_transl_th.size()}")
                        # avg_obj_transl_th = torch.from_numpy(self.avg_obj_transl).float().cuda()
                        # std_obj_transl_th = torch.from_numpy(self.std_obj_transl).float().cuda() 
                        E = (E * (std_obj_transl_th + eps))+ avg_obj_transl_th
                        # cur_future_obj_pos = (cur_future_obj_pos - self.avg_obj_transl[None]) /  (self.std_obj_transl[None] + eps)
                        data_dict['E'] = E 
                    else:
                        obj_pos = data_dict['obj_pos']
                        future_obj_pos = data_dict['future_obj_pos']
                        # print(f"obj_pos: {obj_pos.size()}, std_obj_transl_th: {std_obj_transl_th.size()}")
                        obj_pos = (obj_pos * (std_obj_transl_th + eps)) + avg_obj_transl_th
                        future_obj_pos = (future_obj_pos * (std_obj_transl_th + eps)) + avg_obj_transl_th # std obj transl and avg obj transl #
                        data_dict['obj_pos'] = obj_pos  
                        data_dict['future_obj_pos'] = future_obj_pos
        # data dict #
        # data_dict.update(
        #     {
        #         'hand_pose': hand_pose,
        #         'future_hand_pose': future_hand_pose,
        #         'obj_pos': obj_pos,
        #         'future_obj_pos': future_obj_pos,
        #         'glb_hand_pose': glb_hand_pose,
        #         'glb_obj_pos': glb_obj_pos,
        #         'last_frame_hand_transl': last_frame_hand_transl
        #     }
        # )
        if self.centralize_info:
            last_frame_hand_transl = data_dict['last_frame_hand_transl']
            if self.debug:
                print(f"Inv centralizing data")
            
            if 'E' in data_dict:
                E = data_dict['E']
                if self.debug:
                    print(f"Inv scaling data E: {E.size()}")
                if E.size(-1) == 29:
                    E[..., :3] = E[..., :3] + last_frame_hand_transl.unsqueeze(1)
                    E[..., 22: 22 + 3] = E[..., 22: 22 + 3] + last_frame_hand_transl.unsqueeze(1)
                elif E.size(-1) == 22:
                    E[..., :3] = E[..., :3] + last_frame_hand_transl.unsqueeze(1)
                elif E.size(-1) == 3:
                    E = E + last_frame_hand_transl.unsqueeze(1)
                elif E.size(-1) == 7:
                    E[..., :3] = E[..., :3] + last_frame_hand_transl.unsqueeze(1)
                data_dict['E'] = E
            else:
                
                # obj pos: bsz x nn_fr x 3
                obj_pos = data_dict['obj_pos']
                future_obj_pos = data_dict['future_obj_pos']
                hand_pose = data_dict['hand_pose']
                future_hand_pose = data_dict['future_hand_pose']
                glb_hand_pose = data_dict['glb_hand_pose']
                glb_obj_pos = data_dict['glb_obj_pos']
                obj_pos = obj_pos + last_frame_hand_transl.unsqueeze(1) # .unsqueeze(0)
                future_obj_pos = future_obj_pos + last_frame_hand_transl.unsqueeze(1)
                hand_pose[..., :3] = hand_pose[..., :3] + last_frame_hand_transl.unsqueeze(1)
                future_hand_pose[..., :3] = future_hand_pose[..., :3] + last_frame_hand_transl.unsqueeze(1)
                glb_hand_pose[..., :3] = glb_hand_pose[..., :3] + last_frame_hand_transl.unsqueeze(1)
                glb_obj_pos = glb_obj_pos + last_frame_hand_transl.unsqueeze(1)
                data_dict.update(
                    {
                        'obj_pos': obj_pos,
                        'future_obj_pos': future_obj_pos,
                        'hand_pose': hand_pose,
                        'future_hand_pose': future_hand_pose,
                        'glb_hand_pose': glb_hand_pose,
                        'glb_obj_pos': glb_obj_pos
                    }
                )
        
            
        return data_dict
        
        if self.kine_diff:
            rt_dict = self.inv_scale_data_kine_v2(data_dict=data_dict)
            return rt_dict
        
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        
        scaled_keys = []
        
        if self.use_jointspace_seq:
            
            data_X = data_X[:, 0]
            data_E = data_E[:, 0]
            
            if self.tracking_ctl_diff:
                # self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
                # self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
                # self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
                # self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
                
                self.avg_hand_qs_th = torch.from_numpy(self.avg_hand_qs).float().cuda() #
                self.std_hand_qs_th = torch.from_numpy(self.std_hand_qs).float().cuda() #
                self.avg_hand_qtars_th = torch.from_numpy(self.avg_hand_qtars).float().cuda() # 
                self.std_hand_qtars_th = torch.from_numpy(self.std_hand_qtars).float().cuda() #
                eps = 1e-6
                
                if not self.glb_rot_use_quat:
                    data_X = (data_X * (self.std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
                    data_E = (data_E * (self.std_hand_qtars_th.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qtars_th.unsqueeze(0).unsqueeze(0)
                
                # data_X: bsz x nn_ts x feat_dim # 
                if data_nm is not None:
                    print(f"data_nm: {data_nm[0]}")
                    tot_batch_data_transl = []
                    for cur_data_nm in data_nm:
                        cur_first_frame_obj_transl = self.data_name_to_data[cur_data_nm]['first_frame_obj_trans']
                        cur_first_frame_obj_transl = torch.from_numpy(cur_first_frame_obj_transl).float().cuda() # get the first fram obj transl 
                        tot_batch_data_transl.append(cur_first_frame_obj_transl)
                    tot_batch_data_transl  = torch.stack(tot_batch_data_transl, dim=0) #### nn_bsz x 3
                    
                    # first_frame_obj_transl = self.data_name_to_data[data_nm]['first_frame_obj_trans']
                    # first_frame_obj_transl = torch.from_numpy(first_frame_obj_transl, dtype=torch.float32).cuda()
                    data_X[..., :3] = data_X[..., :3] + tot_batch_data_transl.unsqueeze(1)
                    data_E[..., :3] = data_E[..., :3] + tot_batch_data_transl.unsqueeze(1)
                
            
            if self.diff_task_space:
                
                data_X = data_X[:, 0]
                obj_type = data_X[:, 0:1] + 0.5
                data_X = torch.cat(
                    [obj_type, data_X[:, 1:]], dim=-1
                )
                data_E = data_X.clone()
                
                # avg_task_setting_th = torch.from_numpy(self.avg_task_setting).float().cuda()
                # std_task_setting_th = torch.from_numpy(self.std_task_setting).float().cuda()
                
                # data_X = data_X * (std_task_setting_th.unsqueeze(0) + eps) + avg_task_setting_th.unsqueeze(0)
                # data_E = data_X.clone()
                
            
            rt_dict = {
                'tot_verts': data_X,
                'tot_verts_integrated_qdd_tau': data_E,
                # 'tot_verts_integrated_qdd_tau_s2': data_E # inv scaled #
            }
        else:
            th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            

            
            th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            
            th_avg_particle_xs_integrated_taus_s2 = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_xs_integrated_taus_s2=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            
            
            inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            
            data_verts, data_verts_s2 = data_E[..., :3], data_E[..., 3:]
            inv_scaled_particle_xs_integrated_taus = (data_verts * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
            
            inv_scaled_particle_xs_integrated_taus_s2 = (data_verts_s2 * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2 ## get the inv_scaled integrated taus ##
            
            ###### ======= n-scale the data ======= ######
            # data_E_inv_scaled = data_E
            # data_X_inv_scaled = data_X
            ###### ======= n-scale the data ======= ######
            
            
            rt_dict = {
                'tot_verts': inv_scaled_particle_xs,
                'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
                'tot_verts_integrated_qdd_tau_s2': inv_scaled_particle_xs_integrated_taus_s2 # inv scaled #
            }
        
        if self.training_setting == 'trajectory_translations' and (not notarget):
            # inv_scaled_particle_xs_targe
            data_X_target = data_dict['X_target']
            data_E_target = data_dict['E_target']
            inv_scaled_data_X_target = (data_X_target * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            
            data_verts_target, data_verts_s2_target = data_E_target[..., :3], data_E_target[..., 3:]
            
            inv_scaled_particle_xs_integrated_taus_target = (data_verts_target * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus #
            inv_scaled_particle_xs_integrated_taus_s2_target = (data_verts_s2_target * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus 
            
            inv_scaled_data_target_dict = {
                'tot_verts_target': inv_scaled_data_X_target,
                'tot_verts_integrated_qdd_tau_target': inv_scaled_particle_xs_integrated_taus_target,
                'tot_verts_integrated_qdd_tau_s2_target': inv_scaled_particle_xs_integrated_taus_s2_target
            }
            rt_dict.update(inv_scaled_data_target_dict)
        elif self.training_setting == 'trajectory_translations_cond' and (not notarget):
            data_X_cond = data_dict['X_cond']
            data_E_cond = data_dict['E_cond']
            inv_scaled_data_X_cond = (data_X_cond * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            data_verts_cond, data_verts_s2_cond = data_E_cond[..., :3], data_E_cond[..., 3:]
            
            inv_scaled_particle_xs_integrated_taus_cond = (data_verts_cond * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus
            inv_scaled_particle_xs_integrated_taus_s2_cond = (data_verts_s2_cond * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2
            
            inv_scaled_data_cond_dict = {
                'tot_verts_cond': inv_scaled_data_X_cond,
                'tot_verts_integrated_qdd_tau_cond': inv_scaled_particle_xs_integrated_taus_cond,
                'tot_verts_integrated_qdd_tau_s2_cond': inv_scaled_particle_xs_integrated_taus_s2_cond
            }
            rt_dict.update(inv_scaled_data_cond_dict)
        # elif self.use_jointspace_seq:
        #     data_X = 
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
            
        for key in data_dict:
            if key not in rt_dict:
                rt_dict[key] = data_dict[key]
        
        return rt_dict
    
    
    def scale_data_kine(self, data_dict, data_nm=None):
        
        # sv_dict = {
        #         'obj_verts': obj_verts,
        #         'robot_hand_qs': robot_hand_qs,
        #         'obj_rot_euler': obj_rot_euler_angles,
        #         'obj_transl': obj_transl
        #     }
        
        ## TODO: load kine data in the task conditioanl setting and scale the data here ##
        
        
        obj_verts = data_dict['obj_verts']
        robot_hand_qs = data_dict['robot_hand_qs']
        obj_rot_euler = data_dict['obj_rot_euler']
        obj_transl = data_dict['obj_transl'] 
        # object_type = data_dict['object_type']
        
        eps = 1e-6
        scaled_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        scaled_hand_qs = (robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        scaled_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        scaled_obj_transl = (obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
        
        concat_feat = np.concatenate(
            [scaled_hand_qs, scaled_obj_transl, scaled_obj_rot_euler ], axis=-1
        )
        
        if self.task_cond:
            cond_obj_verts = data_dict['cond_obj_verts']
            cond_robot_hand_qs = data_dict['cond_robot_hand_qs']
            cond_obj_rot_euler = data_dict['cond_obj_rot_euler']
            cond_obj_transl = data_dict['cond_obj_transl']
            # eps = 1e-6
            scaled_cond_obj_verts = (cond_obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
            # cond robot hand qs #
            scaled_cond_hand_qs = (cond_robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
            scaled_cond_obj_rot_euler = (cond_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
            scaled_cond_obj_transl = (cond_obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

            cond_concat_feat = np.concatenate(
                [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
            )

        
        # robot_hand_qs = data_dict['robot_hand_qs'][:, :self.nn_hands_dof] # ts x nn_qs # 
        # robot_hand_qs = robot_hand_qs[None] # 1 x ts x nn_qs # 
        # obj_verts = data_dict['obj_verts'] # ts x nn_obj_verts x 3 # 
        # obj_verts = obj_verts.transpose(1, 0, 2)[:, : robot_hand_qs.shape[1]] # nn_obj_verts x ts x 3 # 
        # # 
        # nn_pts = 512
        # rand_sampled_obj_verts_idxes = np.random.permutation(obj_verts.shape[0])[:nn_pts] # sampled idxes # 
        # obj_verts = obj_verts[rand_sampled_obj_verts_idxes] # 
        
        # avg_obj_verts_th = torc
        # eps = 1e-6
        # obj_verts = (obj_verts - self.avg_obj_verts[None][None]) / (self.std_obj_verts[None][None] + eps)
        
        rt_dict = {
            'X': scaled_obj_verts,
            'E': concat_feat[None],
        }
        if self.task_cond:
            rt_dict.update(
                {
                    'X_cond': scaled_cond_obj_verts,
                    'E_cond': cond_concat_feat[None]
                }
            )
        
        return rt_dict
    
    def _slice_tracking_kine_data(self, ):
        tot_hand_qs = []
        tot_obj_rot_eulers = []
        tot_obj_trans = []
        self.new_data_name_to_kine_info = {}
        for data_nm in self.data_name_to_data:
            # if self.slicing_data:
            pure_data_nm = data_nm.split('_STED_')[0] + ".npy"
            # else:
            #     pure_data_nm = data_nm # + ".npy"
            
            kine_traj_info = self.data_name_to_kine_info[pure_data_nm]
            
            # if self.slicing_data:
            print(f"[Slicing tracking kine data] data_nm: {data_nm}")
            sted_info = data_nm.split("/")[-1].split(".npy")[0].split('_STED_')[1]
            st_idx, ed_idx = sted_info.split("_")
            st_idx, ed_idx = int(st_idx), int(ed_idx)
            # else:
            #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0]
            
            if self.task_cond_type == 'history_future':
                slicing_idxes = self.data_name_to_data[data_nm]['slicing_idxes']
                hand_qs = kine_traj_info['hand_qs'][slicing_idxes][:, :self.nn_hands_dof]
                obj_trans = kine_traj_info['obj_trans'][slicing_idxes]
                obj_ornt = kine_traj_info['obj_ornt'][slicing_idxes]
                obj_rot_euler = kine_traj_info['obj_rot_euler'][slicing_idxes]
                obj_verts = kine_traj_info['obj_verts']
            else:
                hand_qs = kine_traj_info['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
                obj_trans = kine_traj_info['obj_trans'][st_idx: ed_idx]
                obj_ornt = kine_traj_info['obj_ornt'][st_idx: ed_idx]
                obj_rot_euler = kine_traj_info['obj_rot_euler'][st_idx: ed_idx]
                obj_verts = kine_traj_info['obj_verts']
            
            # if self.task_cond and self.task_cond_type == 'history_future':
            if self.task_cond_type == 'history_future':
                first_frame_obj_trans = self.data_name_to_data[data_nm]['first_frame_obj_trans'] # the first frametrans
            else:
                first_frame_obj_trans = obj_trans[0, :3]
            
            hand_qs[:, :3] = hand_qs[:, :3] - first_frame_obj_trans[None]
            obj_trans = obj_trans - first_frame_obj_trans[None]
            
            kine_info_dict = {
                'obj_verts': obj_verts, 
                'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
                'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
                'obj_ornt': obj_ornt ,
                'obj_rot_euler': obj_rot_euler
            }
            self.new_data_name_to_kine_info[data_nm] = kine_info_dict
            tot_hand_qs.append(hand_qs)
            tot_obj_rot_eulers.append(obj_rot_euler)
            tot_obj_trans.append(obj_trans)
        tot_obj_transl = np.concatenate(tot_obj_trans, axis=0)
        tot_obj_rot_eulers = np.concatenate(tot_obj_rot_eulers, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        # tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        self.avg_obj_rot_euler = np.mean(tot_obj_rot_eulers, axis=0)
        self.std_obj_rot_euler = np.std(tot_obj_rot_eulers, axis=0)
        # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        
        self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        # avg obj verts and the kine hand qs and #
        # self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        # self.data_statistics['std_obj_verts'] = self.std_object_verts
        self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        self.data_statistics['std_obj_transl'] = self.std_obj_transl
        self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
        
        # 
        # self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        # self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        self.data_name_to_kine_info = self.new_data_name_to_kine_info
        
        
    
    def scale_data_tracking_ctl(self, data_dict, data_nm):
        # print(f"data_nm: {data_nm}, data_dict: {data_dict.keys()}")
        # print(f"[Scale data tracking ctl] data_nm: {data_nm} glb_rot_use_quat: {self.glb_rot_use_quat}")
        particle_xs = data_dict['tot_verts']
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        
        ## NOTE: currently they are all qs and qtars ##
        eps = 1e-6
        
        if not self.glb_rot_use_quat:
            particle_xs = (particle_xs - self.avg_hand_qs[None][None]) / (self.std_hand_qs[None][None] + eps)
            particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_hand_qtars[None][None]) / (self.std_hand_qtars[None][None] + eps)
        
        # self.data_name_to_kine_info[data_inst_tag] = {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
        #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
        #     'obj_ornt': obj_ornt 
        # }
        
        assert particle_xs.shape[1] == self.slicing_ws
        
        if particle_xs.shape[1] < self.slicing_ws:
            padding_particle_xs = np.concatenate(
                [ particle_xs[:, -1:] for _ in range(self.slicing_ws - particle_xs.shape[1]) ], axis=1
            )
            particle_xs = np.concatenate(
                [ particle_xs, padding_particle_xs ], axis=1
            )
            
            padding_particle_xs_integrated_qdd_tau = np.concatenate(
                [ particle_xs_integrated_qdd_tau[:, -1:] for _ in range(self.slicing_ws - particle_xs_integrated_qdd_tau.shape[1]) ], axis=1
            )
            particle_xs_integrated_qdd_tau = np.concatenate(
                [ particle_xs_integrated_qdd_tau, padding_particle_xs_integrated_qdd_tau ], axis=1
            )

        
        
        rt_dict = {
            'X': particle_xs,
            'E': particle_xs_integrated_qdd_tau
        }
        
        if self.task_cond:
            
            # if '_STED_' in if 
            # if self.slicing_data:
            #     pure_data_nm = data_nm.split('_STED_')[0] + ".npy"
            # else:
            #     pure_data_nm = data_nm # + ".npy"
            pure_data_nm = data_nm
            
            kine_traj_info = self.data_name_to_kine_info[pure_data_nm]
            
            # if self.slicing_data:
            #     sted_info = data_nm.split("/")[-1].split(".")[0].split('_STED_')[1]
            #     st_idx, ed_idx = sted_info.split("_")
            #     st_idx, ed_idx = int(st_idx), int(ed_idx)
            # else:
            #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0]
            st_idx = 0
            ed_idx = kine_traj_info['hand_qs'].shape[0]
            
            hand_qs = kine_traj_info['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
            obj_trans = kine_traj_info['obj_trans'][st_idx: ed_idx]
            obj_ornt = kine_traj_info['obj_ornt'][st_idx: ed_idx]
            obj_rot_euler = kine_traj_info['obj_rot_euler'][st_idx: ed_idx]
            obj_verts = kine_traj_info['obj_verts']
            
            
            # first_frame_obj_trans = obj_trans[0, :3]

            ## TODO: eulers may not be a good representation ##
            
            scaled_cond_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
            # cond robot hand qs #
            
            if not self.glb_rot_use_quat:
                # scaled_cond_hand_qs = (hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
                scaled_cond_hand_qs = (hand_qs - self.avg_kine_hand_qs[None]) / (self.std_kine_hand_qs[None] + eps)
                scaled_cond_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
                scaled_cond_obj_transl = (obj_trans - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

            # E cond --- the conditional future data #  # obj trans #
            cond_concat_feat = np.concatenate(
                [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
            )
            
            assert cond_concat_feat.shape[0] == self.slicing_ws
            
            # cond concat feat --- for the concat feat #
            # cond input # 
            if cond_concat_feat.shape[0] < self.slicing_ws:
                if cond_concat_feat.shape[0] > 0:
                    padding_concat_feat = np.concatenate(
                        [ cond_concat_feat[-1:] for _ in range(self.slicing_ws - cond_concat_feat.shape[0]) ], axis=0
                    )
                    cond_concat_feat = np.concatenate(
                        [cond_concat_feat, padding_concat_feat], axis=0
                    )
                else:
                    cond_concat_feat = np.zeros((self.slicing_ws, cond_concat_feat.shape[-1]), dtype=np.float32)
                    
            
            
            rt_dict.update(
                {
                    'X_cond': scaled_cond_obj_verts,
                    'E_cond': cond_concat_feat[None]
                }
            )
            
            if self.task_cond_type == 'history_future':
                tracking_ctl_info_dict = self.data_name_to_data[data_nm] # 
                # history_info = {
                #     'history_obj_pose': history_obj_pose[None ], # 
                #     'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                #     'first_frame_obj_trans': first_frame_obj_trans,
                #     'first_frame_obj_ornt': first_frame_obj_ornt # first frae obj ornt # an the trans # 
                # }
                history_obj_pose = tracking_ctl_info_dict['history_obj_pose'] # history obj pose -- 1 x ws x nn_obj_dim 
                history_kine_qs = tracking_ctl_info_dict['history_kine_qs'][0] # history kine qs -- 1 x ws x nn_hand_qs 
                first_frame_obj_trans = tracking_ctl_info_dict['first_frame_obj_trans']
                # 'history_obj_trans': history_obj_trans[None ], 
                        # 'history_obj_rot_euler': history_obj_rot_euler[None ],
                history_obj_rot_euler = tracking_ctl_info_dict['history_obj_rot_euler'][0]
                history_obj_trans = tracking_ctl_info_dict['history_obj_trans'][0]
                
                
                if not self.glb_rot_use_quat:
                    scaled_history_kine_qs = (history_kine_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
                    scaled_history_obj_rot_euler = (history_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
                    scaled_history_obj_trans = (history_obj_trans - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
                
                
                # # nn_history_ws x 3 # # 
                # # nn_history_ws x 3 # # 
                # # nn_history_ws x 22 # # 
                # history_cond_contact_feat = np.concatenate(
                #     [history_kine_qs, history_obj_trans, history_obj_rot_euler], axis=-1 # history cond features # 
                # )
                history_cond_contact_feat = np.concatenate(
                    [scaled_history_kine_qs, scaled_history_obj_trans, scaled_history_obj_rot_euler], axis=-1
                )
                # history cond contact feat #
                
                assert history_cond_contact_feat.shape[0] == self.history_ws #  self.slicing_ws
                # print(f"history_cond_contact_feat: {history_cond_contact_feat.shape}")
                
                if history_cond_contact_feat.shape[0] < self.slicing_ws:
                    if history_cond_contact_feat.shape[0] > 0:
                        padding_history_E_cond_feat = np.zeros_like(history_cond_contact_feat[0:1])
                        padding_history_E_cond_feat = np.concatenate(
                            [ padding_history_E_cond_feat for _ in range(self.slicing_ws - history_cond_contact_feat.shape[0]) ], axis=0
                        )
                        history_cond_contact_feat = np.concatenate(
                            [ padding_history_E_cond_feat, history_cond_contact_feat ], axis=0
                        )
                    else:
                        history_cond_contact_feat = np.zeros((self.slicing_ws, history_cond_contact_feat.shape[-1]), dtype=np.float32)
                    
                    
                    
                # print(f"[After padding] history_cond_contact_feat: {history_cond_contact_feat.shape}")
                # if history_cond_contact_feat.
                history_cond_dict = {
                    'history_E_cond': history_cond_contact_feat[None]
                }
                rt_dict.update(history_cond_dict)
                
                pass
            
        return rt_dict
                
    
    def scale_data(self, data_dict, data_nm):
        
        if self.kine_diff:
            rt_dict = self.scale_data_kine(data_dict, data_nm)
            return rt_dict
        elif self.tracking_ctl_diff:
            rt_dict = self.scale_data_tracking_ctl(data_dict, data_nm)
            return rt_dict

        
        ## nn_ts x nn_particles x 3 ## ## get scaled data ##
        
        if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond']:
            unopt_xs = data_dict['unopt_verts']
            unopt_tar_xs = data_dict['unopt_qtar_verts']
            opt_xs = data_dict['opt_verts']
            opt_tar_xs = data_dict['opt_qtar_verts']
            
            eps = 1e-6
            
            # unopt_xs = (unopt_xs - self.)
            unopt_xs = (unopt_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            unopt_tar_xs = (unopt_tar_xs - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
            
            opt_xs = (opt_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            opt_tar_xs = (opt_tar_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            
            sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        
            unopt_xs = unopt_xs[sampled_particle_idxes, :, :]
            unopt_tar_xs = unopt_tar_xs[sampled_particle_idxes, :, :]
            opt_xs = opt_xs[sampled_particle_idxes, :, :]
            opt_tar_xs = opt_tar_xs[sampled_particle_idxes, :, :]
            
            unopt_E = np.concatenate(
                [unopt_tar_xs, unopt_tar_xs], axis=-1
            )    
            opt_E = np.concatenate(
                [opt_tar_xs, opt_tar_xs], axis=-1
            )
            if self.training_setting == 'trajectory_translations_cond':
                rt_dict = {
                    'X_cond': unopt_xs,
                    'E_cond': unopt_E,
                    'X': opt_xs,
                    'E': opt_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
            else:
                rt_dict = {
                    'X': unopt_xs,
                    'E': unopt_E,
                    'X_target': opt_xs,
                    'E_target': opt_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
        
        else:
            if self.use_jointspace_seq:
                
                if self.diff_task_translations:
                    fa_task_setting = data_dict['fa_task_setting']
                    task_setting = np.array(fa_task_setting, dtype=np.float32)
                    
                    ch_task_setting = data_dict['ch_task_setting']
                    ch_task_setting = [ch_task_setting[0] - 0.5, ch_task_setting[1], ch_task_setting[2]]
                    ch_task_setting = np.array(ch_task_setting, dtype=np.float32)
                    particle_xs = ch_task_setting[None][None ]
                    particle_xs_integrated_qdd_tau = particle_xs
                else:
                    
                    particle_xs = data_dict['tot_verts']
                    particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
                    
                    ## set task setting # obj_task_setting #
                    
                    # task_setting = {
                    #     'object_type': cur_data['object_type'],
                    #     'task_rot': cur_data['task_rot'],
                    #     'object_size_x': cur_data['object_size_x']
                    # }
                    #### [object_type, task_rot, object_size_x] ####
                    task_setting = [
                        data_dict['task_setting']['object_type'], data_dict['task_setting']['task_rot'], data_dict['task_setting']['object_size_x']
                    ]
                    task_setting = np.array(task_setting, dtype=np.float32)
                    
                    
                    ### 1) make it into the particle xs and also E ###
                    
                    if self.diff_task_space:
                        eps = 1e-6
                        task_setting_2 = [
                            float(data_dict['task_setting']['object_type']), data_dict['task_setting']['task_rot'], data_dict['task_setting']['object_size_x']
                        ]
                        task_setting_2 = np.array(task_setting_2, dtype=np.float32)
                        
                        task_setting_2 = (task_setting_2 - self.avg_task_setting) / (self.std_task_setting + eps)
                        
                        particle_xs = task_setting_2[None][None]
                        particle_xs_integrated_qdd_tau = task_setting_2[None][None]
                
                
                rt_dict = {
                    'X': particle_xs,
                    'E': particle_xs_integrated_qdd_tau,
                    'obj_task_setting': task_setting #### [object_type, task_rot, object_size_x] ####
                }
            
            else:
                eps = 1e-6
                particle_xs = data_dict['tot_verts']
                particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
                particle_xs_integrated_qdd_tau_s2 = data_dict['tot_verts_integrated_qdd_tau_s2']
                
                
                particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
                particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
                particle_xs_integrated_qdd_tau_s2 = (particle_xs_integrated_qdd_tau_s2 - self.avg_verts_qdd_tau_tot_cases_tot_ts_s2[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts_s2[None][None] + eps)
                # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
                sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
                
                particle_xs = particle_xs[sampled_particle_idxes, :, :]
                particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
                particle_xs_integrated_qdd_tau_s2 = particle_xs_integrated_qdd_tau_s2[sampled_particle_idxes, :, :]
                
                data_E = np.concatenate([particle_xs_integrated_qdd_tau, particle_xs_integrated_qdd_tau_s2], axis=-1)
            
                rt_dict = {
                    'X': particle_xs,
                    'E': data_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
        
        ### return the dict ###
        ### return the dict ###
        return rt_dict
    
    
    def centralize_data(self, data_dict):
        # data_dict_rt = {
        #     'obj_pts': cur_obj_verts, 
        #     'hand_pose': history_hand_qs,
        #     'obj_pos': history_obj_transl,
        #     'obj_ornt': history_obj_ornt,
        #     'future_hand_pose': future_hand_qs,
        #     'future_obj_pos': future_obj_transl, # obj transl #
        #     'future_obj_ornt': future_obj_ornt,
        #     'last_frame_obj_pos': history_obj_transl[-1:, ...],
        #     'factorized_history_window_info': factorized_history_window_info, # 
        #     'factorized_future_window_info': factorized_future_window_info, # 
        # }
        # data_dict_rt.update(
        #     {
        #         'glb_hand_pose': glb_hand_qs,
        #         'glb_obj_pos': glb_obj_transl,
        #         'glb_obj_ornt': glb_obj_rot_ornt
        #     }
        # )
        
        # get the last ahnd qs's positions from the data dict #
        hand_pose = data_dict['hand_pose']
        future_hand_pose = data_dict['future_hand_pose']
        obj_pos = data_dict['obj_pos']
        obj_ornt = data_dict['obj_ornt']
        future_obj_pos = data_dict['future_obj_pos'] 
        glb_hand_pose = data_dict['glb_hand_pose']  
        glb_obj_pos = data_dict['glb_obj_pos']
        glb_obj_ornt = data_dict['glb_obj_ornt']
        
        last_frame_hand_transl = hand_pose[-1, :3]
        
        if self.centralize_info:
            
            if self.debug:
                print(f"Centralizing data...")
            
            
            hand_pose[..., :3] = hand_pose[..., :3] - last_frame_hand_transl[None]
            future_hand_pose[..., :3] = future_hand_pose[..., :3] - last_frame_hand_transl[None]
            obj_pos[..., :3] = obj_pos[..., :3] - last_frame_hand_transl[None]
            future_obj_pos[..., :3] = future_obj_pos[..., :3] - last_frame_hand_transl[None]
            glb_hand_pose[..., :3] = glb_hand_pose[..., :3] -   last_frame_hand_transl[None]
            glb_obj_pos[..., :3] = glb_obj_pos[..., :3] - last_frame_hand_transl[None] # 
            
        
        glb_hand_pose = np.concatenate([glb_hand_pose[::self.glb_feat_per_skip], glb_hand_pose[-1:]], axis=0)
        glb_obj_pos = np.concatenate([glb_obj_pos[::self.glb_feat_per_skip], glb_obj_pos[-1:]], axis=0)
        glb_obj_ornt = np.concatenate([glb_obj_ornt[::self.glb_feat_per_skip], glb_obj_ornt[-1:]], axis=0)
        # history_window_size #
        
        hand_pose = hand_pose[-self.history_window_size: ]
        obj_pos = obj_pos[-self.history_window_size: ]
        obj_ornt = obj_ornt[-self.history_window_size: ]
        
        data_dict.update(
            {
                'hand_pose': hand_pose,
                'future_hand_pose': future_hand_pose,
                'obj_pos': obj_pos,
                'obj_ornt': obj_ornt,
                'future_obj_pos': future_obj_pos,
                'glb_hand_pose': glb_hand_pose,
                'glb_obj_pos': glb_obj_pos,
                'glb_obj_ornt': glb_obj_ornt,
                'last_frame_hand_transl': last_frame_hand_transl,
            }
        )
        
        
        
        return data_dict
    
    
    
    def scale_data_kine_new(self, data_dict, data_nm=None):
        
        # data_dict_rt = {
        #     'obj_pts': cur_obj_verts, 
        #     'hand_pose': history_hand_qs,
        #     'obj_pos': history_obj_transl,
        #     'obj_ornt': history_obj_ornt,
        #     'future_hand_pose': future_hand_qs,
        #     'future_obj_pos': future_obj_transl,
        #     'future_obj_ornt': future_obj_ornt,
        #     'last_frame_obj_pos': history_obj_transl[-1:, ...]
        # }
        
        # # data dict rt #
        
        # if self.w_glb_traj_feat_cond:
        #     glb_hand_qs = cur_hand_qs[::10]
        #     glb_obj_transl = cur_obj_transl[::10]
        #     glb_obj_rot_ornt = cur_obj_rot_quat[::10]
            
        #     # print(f"[Debug] glb_hand_qs: {glb_hand_qs.shape}, glb_obj_transl: {glb_obj_transl.shape}, glb_obj_rot_ornt: {glb_obj_rot_ornt.shape}")
        #     data_dict_rt.update(
        #         {
        #             'glb_hand_pose': glb_hand_qs,
        #             'glb_obj_pos': glb_obj_transl,
        #             'glb_obj_ornt': glb_obj_rot_ornt
        #         }
        #     )
            
        eps = 1e-6
        obj_pts = data_dict['obj_pts'] ## nn_pts x 3 ## 
        canonicalized_obj_pts = (obj_pts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        
        rt_dict = { }
        rt_dict.update(data_dict)
        
        #  avg_tot_clip_hand_qs, std_tot_clip_hand_qs, avg_tot_clip_obj_pos, std_tot_clip_obj_pos
        if self.scale_clip_data: # if we scale the clip data #
            if self.debug:
                print(f"Scaling clip data...")
            # three types of information -- history object pos, history hand info, future obj pos, future hand info #
            hand_pose = data_dict['hand_pose']
            future_hand_pose = data_dict['future_hand_pose']
            obj_pos = data_dict['obj_pos']
            future_obj_pos = data_dict['future_obj_pos']
            glb_hand_pose = data_dict['glb_hand_pose']
            glb_obj_pos = data_dict['glb_obj_pos']
            
            hand_pose = ( hand_pose - self.avg_tot_clip_hand_qs[None] ) / (self.std_tot_clip_hand_qs[None] + eps)
            future_hand_pose = ( future_hand_pose - self.avg_tot_clip_hand_qs[None] ) / (self.std_tot_clip_hand_qs[None] + eps)
            obj_pos = ( obj_pos - self.avg_tot_clip_obj_pos[None] ) / (self.std_tot_clip_obj_pos[None] + eps)
            future_obj_pos = ( future_obj_pos - self.avg_tot_clip_obj_pos[None] ) / (self.std_tot_clip_obj_pos[None] + eps)
            glb_hand_pose = ( glb_hand_pose - self.avg_tot_clip_hand_qs[None] ) / (self.std_tot_clip_hand_qs[None] + eps)
            glb_obj_pos = ( glb_obj_pos - self.avg_tot_clip_obj_pos[None] ) / (self.std_tot_clip_obj_pos[None] + eps)
            rt_dict.update(
                {
                    'hand_pose': hand_pose,
                    'future_hand_pose': future_hand_pose,
                    'obj_pos': obj_pos,
                    'future_obj_pos': future_obj_pos,
                    'glb_hand_pose': glb_hand_pose,
                    'glb_obj_pos': glb_obj_pos
                }
            )
            # 
        else:
            
            # # obj_pos, future_obj_pos # #
            # avg_obj_transl, std_obj_transl
            if 'obj_pos' in data_dict:
                cur_obj_pos = data_dict['obj_pos']
                cur_future_obj_pos = data_dict['future_obj_pos']
                
                
                
                
                cur_obj_pos = (cur_obj_pos - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
                
                
                
                cur_future_obj_pos = (cur_future_obj_pos - self.avg_obj_transl[None]) /  (self.std_obj_transl[None] + eps)
                
                rt_dict.update(
                    {
                        'obj_pos': cur_obj_pos,
                        'future_obj_pos': cur_future_obj_pos
                    }
                )
                

        rt_dict.update(
            {
                'obj_pts': canonicalized_obj_pts
            }
        )
        
        
        # obj_pts = data_dict['obj_pts']
        # hand_pose = data_dict['hand_pose'] # history qs #
        # obj_pos = data_dict['obj_pos']
        # obj_ornt = data_dict['obj_ornt']
        # future_hand_pose = data_dict['future_hand_pose']
        # future_obj_pos = data_dict['future_obj_pos']
        # future_obj_ornt = data
        
        
        # obj_verts = data_dict['obj_verts']
        # robot_hand_qs = data_dict['robot_hand_qs']
        # obj_rot_euler = data_dict['obj_rot_euler']
        # obj_transl = data_dict['obj_transl'] 
        # # object_type = data_dict['object_type']
        
        # eps = 1e-6
        # scaled_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        # scaled_hand_qs = (robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        # scaled_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        # scaled_obj_transl = (obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
        
        # concat_feat = np.concatenate(
        #     [scaled_hand_qs, scaled_obj_transl, scaled_obj_rot_euler ], axis=-1
        # )
        
        # if self.task_cond:
        #     cond_obj_verts = data_dict['cond_obj_verts']
        #     cond_robot_hand_qs = data_dict['cond_robot_hand_qs']
        #     cond_obj_rot_euler = data_dict['cond_obj_rot_euler']
        #     cond_obj_transl = data_dict['cond_obj_transl']
        #     # eps = 1e-6
        #     scaled_cond_obj_verts = (cond_obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        #     # cond robot hand qs #
        #     scaled_cond_hand_qs = (cond_robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        #     scaled_cond_obj_rot_euler = (cond_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        #     scaled_cond_obj_transl = (cond_obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

        #     cond_concat_feat = np.concatenate(
        #         [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
        #     )

        
        # rt_dict = {
        #     'X': scaled_obj_verts,
        #     'E': concat_feat[None],
        # }
        # if self.task_cond:
        #     rt_dict.update(
        #         {
        #             'X_cond': scaled_cond_obj_verts,
        #             'E_cond': cond_concat_feat[None]
        #         }
        #     )
        
        # return rt_dict
    
        return rt_dict
    
    
    
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if isinstance(data_dict_np[key], str):
                data_dict_th[key] = data_dict_np[key]
            elif key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
        
        
        return data_dict_th
    
    
    
    
    def __getitem__(self, index):
        
        
        
        cur_data_info = self.data_with_window_info_list[index]
        data_nm = cur_data_info['data_nm']
        cur_data_nm = data_nm
        history_window_info = cur_data_info['history_window_idxes']
        future_window_info = cur_data_info['future_window_idxes']
        
        factorized_history_window_info = history_window_info.astype(np.float32) / float(self.num_frames)
        # use the frame 1000 to rescale the window #
        self.rescaled_num_frames = 1000
        factorized_history_window_info = (factorized_history_window_info * self.rescaled_num_frames).astype(np.int32) # get the rescaled num frames #
        
        factorized_future_window_info = future_window_info.astype(np.float32) / float(self.num_frames)
        factorized_future_window_info = (factorized_future_window_info * self.rescaled_num_frames).astype(np.int32) 
        
        
        cur_data_dict = self.data_name_to_data[data_nm]
        cur_obj_transl = cur_data_dict['obj_transl']
        # cur_obj_rot_euler = cur_data_dict['obj_rot_euler']
        cur_obj_verts = cur_data_dict['obj_verts']
        cur_hand_qs = cur_data_dict['robot_hand_qs']
        cur_obj_rot_quat = cur_data_dict['obj_rot_quat']
        
        # cur obj rot quat # # # # rot quat #
        
        history_obj_transl = cur_obj_transl[history_window_info]
        history_obj_ornt = cur_obj_rot_quat[history_window_info]
        history_hand_qs = cur_hand_qs[history_window_info]
        future_obj_transl = cur_obj_transl[future_window_info]
        future_obj_ornt = cur_obj_rot_quat[future_window_info]
        future_hand_qs = cur_hand_qs[future_window_info]
        
        # if self.canonicalize_features:
        #     cur_last_frame_obj_transl = history_obj_transl[-1:, ...]
        #     history_obj_transl = history_obj_transl - cur_last_frame_obj_transl
        #     history_hand_qs[..., :3] = history_hand_qs[..., :3] - cur_last_frame_obj_transl
        #     future_obj_transl = future_obj_transl - cur_last_frame_obj_transl
        #     future_hand_qs[..., :3] = future_hand_qs[..., :3] - cur_last_frame_obj_transl
        
        # obj_pos, future_obj_pos #
        
        data_dict_rt = {
            'obj_pts': cur_obj_verts, 
            'hand_pose': history_hand_qs,
            'obj_pos': history_obj_transl,
            'obj_ornt': history_obj_ornt,
            'future_hand_pose': future_hand_qs,
            'future_obj_pos': future_obj_transl,
            'future_obj_ornt': future_obj_ornt,
            'last_frame_obj_pos': history_obj_transl[-1:, ...],
            'factorized_history_window_info': factorized_history_window_info,
            'factorized_future_window_info': factorized_future_window_info,
        }
        
        if self.diff_contact_sequence:
            cur_seq_contact_data = cur_data_info['contact_data']
            cur_seq_history_contact = cur_seq_contact_data[history_window_info]
            cur_seq_future_contact = cur_seq_contact_data[future_window_info]

            
            # history_contact, future_contact #
            data_dict_rt.update(
                {
                    'history_contact': cur_seq_history_contact,
                    'future_contact': cur_seq_future_contact
                }
            )
        
        # masked future cond -- randomly mask out future hand pose trajectory and the future object pose trajectory #
        if self.w_masked_future_cond:
            if self.debug:
                print(f"[Debug] Masked future cond")
            masked_future_hand_qs = future_hand_qs.copy()
            masked_future_obj_transl = future_obj_transl.copy()
            masked_future_obj_ornt = future_obj_ornt.copy() # # get the masked future cond # 
            # three conditions #
            # two conditions --- the contact condition is a little bit heterogeneous so we should train a separate model to deal with contact conditions #
            selected_masking_idx = np.random.randint(0, 2, size=(1,))[0].item() # gethtejselected masking idxes # 
            if selected_masking_idx == 0: # mask out object features
                masked_future_obj_transl[:, :] = 0.0
                masked_future_obj_ornt[:, :] = 0.0 
            elif selected_masking_idx == 1:
                masked_future_hand_qs[:, :] = 0.0
            data_dict_rt.update( # get hte masked iputs # 
                {
                    'masked_future_hand_qs': masked_future_hand_qs,
                    'masked_future_obj_transl': masked_future_obj_transl,
                    'masked_future_obj_ornt': masked_future_obj_ornt
                }
            )
        
        
        # 
        # w glb traj feat cond # # history # # # # too many reasons that can lead to that behaviours # # partial hand info? # # get the data statistics for the partial hand info # # history # data statistics for partialhand trajecotires, for obj pos, for obj ornt, # obj pos # should record and canonicalize when slicing #
        
        if self.w_glb_traj_feat_cond:
            
            if 'kine_obj_transl' in cur_data_dict:
                glb_obj_transl = cur_data_dict['kine_obj_transl'][::10]
                glb_obj_rot_ornt = cur_data_dict['kine_obj_rot_quat'][::10]
                glb_hand_qs = cur_data_dict['kine_robot_hand_qs'][::10]
            else:
                # for the task description? #
                glb_hand_qs = cur_hand_qs[::10]
                glb_obj_transl = cur_obj_transl[::10]
                glb_obj_rot_ornt = cur_obj_rot_quat[::10]
            
            # print(f"[Debug] glb_hand_qs: {glb_hand_qs.shape}, glb_obj_transl: {glb_obj_transl.shape}, glb_obj_rot_ornt: {glb_obj_rot_ornt.shape}")
            data_dict_rt.update(
                {
                    'glb_hand_pose': glb_hand_qs,
                    'glb_obj_pos': glb_obj_transl,
                    'glb_obj_ornt': glb_obj_rot_ornt
                }
            )
            if self.use_clip_glb_features:
                data_dict_rt.update(
                    {
                        'text_features': cur_data_dict['text_features']
                    }
                )
            if self.debug:
                print(f"cur_hand_qs: {cur_hand_qs.shape}. glb_hand_qs: {glb_hand_qs.shape}, glb_obj_transl: {glb_obj_transl.shape}, glb_obj_rot_ornt: {glb_obj_rot_ornt.shape}")
                print(f"glb_hand_qs[-1]: {glb_hand_qs[-1]}")
        # glb obj pos #
        
        # if self.centralize_info:
        data_dict_rt = self.centralize_data(data_dict_rt)
        
        if self.canonicalize_obj_pts:
            data_dict_rt = self.scale_data_kine_new(data_dict_rt)
        
        
        cur_data_scaled_th = self.data_dict_to_th(data_dict_rt)
        
        
        
        cur_data_scaled_th.update(
            {'data_nm': cur_data_nm}
        )
        return cur_data_scaled_th




class Uni_Manip_3D_PC_V9_Dataset(torch.utils.data.Dataset):
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        self.debug = self.cfg.debug
        
        # dt and task.dt #
        self.dt = cfg.task.dt
        self.nn_timesteps = cfg.task.nn_timesteps
        # 
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        self.multi_inst = cfg.dataset_3d_pc.multi_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst
        self.specified_hand_type = cfg.dataset_3d_pc.specified_hand_type 
        
        self.specified_object_type = cfg.dataset_3d_pc.specified_object_type
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages
        self.use_static_first_frame = cfg.dataset_3d_pc.use_static_first_frame
        self.use_shadow_test_data = cfg.sampling.use_shadow_test_data
        self.sampling = cfg.sampling.sampling
        
        # self.use_allegro_test_data = cfg.sampling.use_allegro_test_data
        self.specified_test_subfolder = cfg.sampling.specified_test_subfolder
        self.specified_statistics_info_fn = cfg.training.specified_statistics_info_fn
        self.specified_sampled_particle_idxes_fn = cfg.training.specified_sampled_particle_idxes_fn
        
        self.training_setting = cfg.training.setting
        self.use_jointspace_seq = cfg.training.use_jointspace_seq
        
        # 
        self.task_cond = cfg.training.task_cond # 
        self.diff_task_space = cfg.training.diff_task_space
        self.diff_task_translations = cfg.training.diff_task_translations
        
        self.kine_diff = cfg.training.kine_diff
        self.tracking_ctl_diff = cfg.training.tracking_ctl_diff
        
        
        
        ''' for sampling '''
        self.target_grab_inst_tag = cfg.sampling.target_grab_inst_tag
        self.target_grab_inst_opt_fn = cfg.sampling.target_grab_inst_opt_fn
        
        self.w_glb_traj_feat_cond = cfg.training.w_glb_traj_feat_cond
        self.canonicalize_features = cfg.dataset.canonicalize_features
        
        ''' for training and the training data '''
        self.grab_inst_tag_to_optimized_res_fn = cfg.training.grab_inst_tag_to_optimized_res_fn
        self.taco_inst_tag_to_optimized_res_fn = cfg.training.taco_inst_tag_to_optimized_res_fn
        if len(self.taco_inst_tag_to_optimized_res_fn) > 0 and os.path.exists(self.taco_inst_tag_to_optimized_res_fn):
            self.grab_inst_tag_to_optimized_res_fn = [self.grab_inst_tag_to_optimized_res_fn, self.taco_inst_tag_to_optimized_res_fn]    
      
    
    
        try:
            self.use_taco_data = cfg.training.use_taco_data
        except:
            self.use_taco_data = False
    
        try:
            self.glb_rot_use_quat = cfg.training.glb_rot_use_quat
        except:
            self.glb_rot_use_quat = False
        self.succ_rew_threshold = cfg.training.succ_rew_threshold # 
        
        
        try:
            self.task_cond_type = cfg.training.task_cond_type
        except:
            self.task_cond_type = "future"
        
        try:
            self.slicing_ws = cfg.training.slicing_ws
        except:
            self.slicing_ws = 30
            pass
        
        ### TODO: a slicing ws with an additional history window ws for tracking ###
        ### trajs obtained via closed loop planning? ###
        
        try:
            self.history_ws = cfg.training.history_ws
        except:
            self.history_ws = self.slicing_ws
        
        
        try:
            self.use_kine_obj_pos_canonicalization = cfg.training.use_kine_obj_pos_canonicalization
        except:
            self.use_kine_obj_pos_canonicalization = False
        
        
        try:
            self.exp_additional_tag = cfg.training.exp_additional_tag
        except:
            self.exp_additional_tag = ''
    
        #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy #
        # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag #
        try:
            self.taco_interped_fr_grab_tag = cfg.training.taco_interped_fr_grab_tag
        except:
            self.taco_interped_fr_grab_tag = "ori_grab_s2_phone_call_1"
        
        try:
            self.taco_interped_data_sv_additional_tag = cfg.training.taco_interped_data_sv_additional_tag
        except:
            self.taco_interped_data_sv_additional_tag = 'v2'
        
        try:
            self.num_frames = cfg.dataset.num_frames
        except:
            self.num_frames = 150
        
        valid_data_statistics = None 
        
        try:
            self.task_inherit_info_fn = cfg.training.task_inherit_info_fn
        except:
            self.task_inherit_info_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy"
        
        try:
            self.obj_type_to_kinematics_traj_dict_fn = cfg.dataset_3d_pc.obj_type_to_kinematics_traj_dict_fn
        except:
            self.obj_type_to_kinematics_traj_dict_fn = ''
        
        try:
            self.canonicalize_obj_pts = cfg.dataset_3d_pc.canonicalize_obj_pts
        except:
            self.canonicalize_obj_pts = False
            
        try:
            self.forcasting_window_size = cfg.dataset_3d_pc.forcasting_window_size
        except:
            self.forcasting_window_size = 30
        
        self.maxx_nn_pts = 512
        
        # got the self.num_frames #
        
        # 1-> sclae thj
        # try:
        #     self.step_size = 
        # self.step_size = self.slicing_ws // 2
        
        # if self.step_size == 0:
        #     self.step_size = 1
        
        self.step_size = 1
        
        try:
            self.slicing_data = cfg.training.slicing_data
        except:
            self.slicing_data = False
        
        
        
        # diff_task_translations and diff_task_space and task_cond #
        self.obj_name_to_idx = {
            'box': 0,
            'cylinder': 1
        }
        
        self.nn_hands_dof = 22
        
        exp_tags = ["tds_exp_2"]
        
        
        self.sim_platform = cfg.dataset_3d_pc.sim_platform
        
        self.data_statistics_info_fn = cfg.dataset_3d_pc.data_statistics_info_fn
        
        self.statistics_info_fn = cfg.dataset_3d_pc.statistics_info_fn
        
        print(f"statistics_info_fn: {self.statistics_info_fn}")
        
        self.tot_inheriting_infos = []

        self.hybrid_dataset = False

        # self.tracking_save_info_fn = "/cephfs/yilaa/data/GRAB_Tracking/data"
        self.tracking_save_info_fn = cfg.dataset_3d_pc.tracking_save_info_fn 
        self.tracking_info_st_tag = "passive_active_info_"
        
        self.target_grab_data_nm = None
        
        
        ## TODO: 1) add canonicalize_obj_pts; 2) add obj_type_to_kinematics_traj_dict_fn
        ## 
        if len(self.obj_type_to_kinematics_traj_dict_fn) != 0 and os.path.exists(self.obj_type_to_kinematics_traj_dict_fn):
            self.obj_type_to_kinematics_traj_dict = np.load(self.obj_type_to_kinematics_traj_dict_fn, allow_pickle=True).item()
            self.objtype_to_tracking_sv_info = self.obj_type_to_kinematics_traj_dict
            self.data_list = [
                self.objtype_to_tracking_sv_info[obj_type] for obj_type in self.objtype_to_tracking_sv_info
            ]
        else:
            passive_act_info_tag = 'passive_active_info_ori_grab'
            # tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
            tracking_save_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
            print("Start enumerating retargeted tracking info")
            tracking_save_info = os.listdir(tracking_save_info_fn)
            tracking_save_info = [
                fn for fn in tracking_save_info if fn.endswith(".npy") and fn[: len(passive_act_info_tag)] == passive_act_info_tag
            ]
            
            # if self.num_frames == 150:
            #     non_nf_tag = '_nf_'
            #     tracking_save_info = [
            #         fn for fn in tracking_save_info if non_nf_tag not in fn
            #         ]
            
            nf_tag = f'_nf_{self.num_frames}'
            tracking_save_info = [
                fn for fn in tracking_save_info if nf_tag in fn
            ]
            
            # tracking_save_info = tracking_save_info[:10]
            
            passive_act_pure_tag = "passive_active_info_"
            self.objtype_to_tracking_sv_info = {}
            for cur_sv_info in tracking_save_info:
                cur_objtype = cur_sv_info.split(".")[0]
                cur_objtype = cur_objtype.split("_nf_")[0]
                cur_objtype = cur_objtype[len(passive_act_pure_tag): ]
                self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)

                # cur_objtype = "ori_grab" + cur_objtype # cur obj type # # cur ob type #
                # self.objtype_to_tracking_sv_info[cur_objtype] = os.path.join(tracking_save_info_fn, cur_sv_info)

            tracking_save_info = [
                os.path.join(tracking_save_info_fn, fn) for fn in tracking_save_info
            ]
            self.data_list = tracking_save_info
        
        self.data_name_to_data = {}
        
        self._preload_kine_data()
        
        # # _preload_kine_data # 
        
        # # data_name_to_data #
        # # sv_dict = {
        # #     'obj_verts': obj_verts,
        # #     'robot_hand_qs': robot_hand_qs,
        # #     'obj_rot_euler': obj_rot_euler_angles,
        # #     'obj_transl': obj_transl,
        # #     'object_type': object_type,
        # # }
        # ### TODO: add the task description ###
        # forcasting_window_size = self.forcasting_window_size #  30
        # self.data_with_window_info_list = []
        # for data_nm in self.data_name_to_data:
        #     cur_val_dict = self.data_name_to_data[data_nm]
        #     cur_obj_transl = cur_val_dict['obj_transl']
        #     print(f"cur_obj_transl: {cur_obj_transl.shape}")
        #     for idx in range(cur_obj_transl.shape[0]):
        #         history_last_idx = idx - forcasting_window_size + 1 
        #         future_st_idx = idx + 1
        #         future_ed_idx = idx + forcasting_window_size
        #         history_window_idxes = range(history_last_idx, idx + 1)
        #         future_window_idxes = range(future_st_idx, future_ed_idx + 1)
        #         history_window_idxes  = np.array(history_window_idxes, dtype=np.int32)
        #         future_window_idxes = np.array(future_window_idxes, dtype=np.int32)
        #         history_window_idxes = np.clip(history_window_idxes, 0, cur_obj_transl.shape[0] - 1)
        #         future_window_idxes = np.clip(future_window_idxes, 0, cur_obj_transl.shape[0] - 1)
        #         self.data_with_window_info_list.append(
        #             {
        #                 'data_nm': data_nm,
        #                 'history_window_idxes': history_window_idxes,
        #                 'future_window_idxes': future_window_idxes
        #             }
        #         )
        #     pass
        
        
        
        
        

    
    
    def __len__(self):
        
        return len(self.data_list)
        
        ### the lenght of the target ###
        if self.sampling and self.target_grab_data_nm is not None and self.slicing_data and len(self.target_grab_inst_tag) > 0:
            return len(self.tot_target_data_nm)
                # tot_target_data_nm #
        
        if self.kine_diff and self.task_cond:
            return len(self.tot_data_dict_list)
        else:
            return len(self.data_name_to_data)
        # return len(self.data_name_list) 
    
    
    def _save_kine_data_statistics(self, ):
        # data_statistics
        if len(self.exp_additional_tag) ==0:
            data_stats_sv_fn = f"data_statistics_kinematics_diff.npy"
        else:
            data_stats_sv_fn = f"data_statistics_kinematics_diff_{self.exp_additional_tag}.npy"
        data_stats_sv_fn = os.path.join(f"assets", data_stats_sv_fn)
        np.save(data_stats_sv_fn, self.data_statistics)
        print(f"[Kine Diff] Data statistics saved to {data_stats_sv_fn}")
    
    def _save_data_statistics(self, ):
        ##### data stats sv fn #####
        if len(self.exp_additional_tag) == 0:
            data_stats_sv_fn = f"data_statistics_ws_{self.slicing_ws}_step_{self.step_size}.npy"
        else:
            # exp_additional_tag
            data_stats_sv_fn = f"data_statistics_ws_{self.slicing_ws}_step_{self.step_size}_{self.exp_additional_tag}.npy"
        
        data_stats_sv_fn = os.path.join(f"assets", data_stats_sv_fn)
        np.save(data_stats_sv_fn, self.data_statistics)
        print(f"Data statistics saved to {data_stats_sv_fn}") 
        
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ## ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    
    
    def _load_single_tracking_kine_info(self, data_inst_tag):
        if isinstance(data_inst_tag, str):
            
            kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
            kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
            # get he kinemati info file # # load #  # load from that # 
            cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
            
            hand_qs = cur_kine_data['robot_delta_states_weights_np']
            maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            hand_qs = hand_qs[:maxx_ws]
            
            obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
            
            # then segment the data_inst_tag to get the mesh file name #
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{data_inst_tag}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            while len(to_sample_fr_idxes) < self.maxx_nn_pts:
                to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
        elif isinstance(data_inst_tag, tuple): # obj
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
                obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws] # obj verts #
            elif 'taco' in obj_type:
                #  passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_031_v2.npy
                # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
                traj_kine_info = f'passive_active_info_{self.taco_interped_fr_grab_tag}_interped_taco_20230928_031_{self.taco_interped_data_sv_additional_tag}.npy'
                taco_kine_sv_root = '/cephfs/xueyi/data/TACO_Tracking_PK/data'
                traj_kine_info = os.path.join(taco_kine_sv_root, traj_kine_info) # get hejkineinfo s 
                traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
                hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
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
        
        
        kine_obj_rot_euler_angles = []
        for i_fr in range(obj_ornt.shape[0]):
            cur_rot_quat = obj_ornt[i_fr] # dkine obj rot quat #
            cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
            kine_obj_rot_euler_angles.append(cur_rot_euler)
        kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
        kine_info_dict = {
            'obj_verts': obj_verts, 
            'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
            'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
            'obj_ornt': obj_ornt ,
            'obj_rot_euler': kine_obj_rot_euler_angles
        }
        return kine_info_dict
    
    
    # 
    # load the tracking kine info #
    ### data name to kine info ###
    def _load_tracking_kine_info(self,):
        # self.maxx_kine_nn_ts = 300
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        # add the positional encodings into the model encodings? #
        # add the positional encodings into the model encoding? #
        # each is encoded positionals is encodied into the model #
        # give the number of timesteps and the timesteps #
        # give the total number of tiemsteps # tracking controls #
        # domain randomizations an the control alignmnet #
        # the relative positions of the frame sequence in the total trajectory to track # #
        # the relative positions # # relative # # # #
        
        ## maxx kine nn ts setting ##
        for i_inst, data_inst_tag in enumerate(self.data_inst_tag_list):
            print(f"[Loading tracking kine info] {i_inst}/{len(self.data_inst_tag_list)}: {data_inst_tag}")
            kine_info_dict = self._load_single_tracking_kine_info(data_inst_tag)
            # if isinstance(data_inst_tag, str):
            #     kine_info_fn = f"{self.tracking_info_st_tag}{data_inst_tag}.npy"
            #     kine_info_fn = os.path.join(self.tracking_save_info_fn, kine_info_fn)
            #     # get he kinemati info file # # load #  # load from that # 
            #     cur_kine_data = np.load(kine_info_fn, allow_pickle=True).item()
                
            #     hand_qs = cur_kine_data['robot_delta_states_weights_np']
            #     maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            #     hand_qs = hand_qs[:maxx_ws]
                
            #     obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
            #     obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
            #     # then segment the data_inst_tag to get the mesh file name #
            #     self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            #     grab_mesh_fn = f"{data_inst_tag}.obj"
            #     grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
            #     # get the object mesh #
            #     obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            #     obj_verts = obj_mesh.vertices # nn_pts x 3
            #     # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            #     to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            #     while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            #         to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            #     random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            #     random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            #     obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            # elif isinstance(data_inst_tag, tuple): # obj
            #     obj_type, traj_obj_type = data_inst_tag
            #     traj_kine_info  = f"{self.tracking_info_st_tag}{traj_obj_type}.npy"
            #     traj_kine_info = os.path.join(self.tracking_save_info_fn, traj_kine_info)
            #     traj_kine_info_data = np.load(traj_kine_info, allow_pickle=True).item()
                
            #     hand_qs = traj_kine_info_data['robot_delta_states_weights_np']
            #     maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
            #     hand_qs = hand_qs[:maxx_ws]
                
            #     obj_trans = traj_kine_info_data['object_transl'][:maxx_ws] # nn_ts x 3 
            #     obj_ornt = traj_kine_info_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
                
            #     # then segment the data_inst_tag to get the mesh file name #
            #     self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            #     grab_mesh_fn = f"{obj_type}.obj"
            #     grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
                
            #     # get the object mesh #
            #     obj_mesh = trimesh.load(grab_mesh_fn, force='mesh')
            #     obj_verts = obj_mesh.vertices # nn_pts x 3
            #     # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            #     # random_sampled_idxes = np.random.sample()
            #     to_sample_fr_idxes = list(range(obj_verts.shape[0]))
            #     while len(to_sample_fr_idxes) < self.maxx_nn_pts:
            #         to_sample_fr_idxes += list(range(obj_verts.shape[0]))
            #     random_sampled_idxes = random.sample(to_sample_fr_idxes, self.maxx_nn_pts)
            #     random_sampled_idxes = np.array(random_sampled_idxes, dtype=np.int32)
            #     obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws] # obj verts #
            #     # grab_mesh_fn = f"{data_inst_tag}.obj"
            #     # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            # else: #
            #     raise ValueError(f"Unrecognized grab_data_inst_type: {type(data_inst_tag)}")
                
            # # hand_qs = cur_kine_data['robot_delta_states_weights_np'][:maxx_ws] # nn_ts x 22
            
            # kine_obj_rot_euler_angles = []
            # for i_fr in range(obj_ornt.shape[0]):
            #     cur_rot_quat = obj_ornt[i_fr] # dkine obj rot quat #
            #     cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True) # get the rotation euler 
            #     kine_obj_rot_euler_angles.append(cur_rot_euler)
            # kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            
            # the task conditional settings #
            self.data_name_to_kine_info[self.data_list[i_inst]] = kine_info_dict
            # { # data list to the obj verts and the ahand qs # 
            #     'obj_verts': obj_verts, 
            #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
            #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
            #     'obj_ornt': obj_ornt ,
            #     'obj_rot_euler': kine_obj_rot_euler_angles
            # }
            obj_trans, kine_obj_rot_euler_angles, hand_qs, obj_verts = kine_info_dict['obj_trans'], kine_info_dict['obj_rot_euler'], kine_info_dict['hand_qs'], kine_info_dict['obj_verts']

            tot_obj_transl.append(obj_trans)
            tot_obj_rot_euler.append(kine_obj_rot_euler_angles)
            tot_hand_qs.append(hand_qs)
            tot_object_verts.append(obj_verts)
        
        
        tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
        self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
        # avg hand qs and the std hand qs #?
        ## TODO: for the kinematics target data --- we should save them using a differnet name? #
        # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        # avg kine hand qs #
        self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        
        
        # 
        self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        
        # avg obj verts and the kine hand qs and #
        self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        self.data_statistics['std_obj_verts'] = self.std_object_verts
        self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        self.data_statistics['std_obj_transl'] = self.std_obj_transl
        self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
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
        
    
    # preload single tracking data #
    # cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(target_data_fn)
    def _preload_single_tracking_ctl_data(self, data_fn, add_to_dict=True):
        
        print(f"loading from {data_fn}")
        cur_data = np.load(data_fn, allow_pickle=True).item()
        if self.use_jointspace_seq:
            if self.sim_platform == 'isaac':
                optimized_obj_pose = cur_data['optimized_obj_pose']
                optimized_hand_qtars = cur_data['optimized_hand_qtars']
                optimized_hand_qs = cur_data['optimized_hand_qs']
                # # TODO: use top-k other than using the best evaluated res ? # 
                hand_qs_np = optimized_hand_qs[0] 
                hand_qtars_np = optimized_hand_qtars[0] # 
                
                if self.glb_rot_use_quat:
                    hand_glb_rot_np = hand_qs_np[..., 3:6]
                    hand_glb_qtar_np = hand_qtars_np[..., 3:6]
                    hand_glb_rot_th = torch.from_numpy(hand_glb_rot_np)
                    hand_glb_tar_rot_th = torch.from_numpy(hand_glb_qtar_np)
                    hand_glb_quat_th = quat_from_euler_xyz(hand_glb_rot_th[..., 0], hand_glb_rot_th[..., 1], hand_glb_rot_th[..., 2])
                    hand_glb_tar_quat_th = quat_from_euler_xyz(hand_glb_tar_rot_th[..., 0], hand_glb_tar_rot_th[..., 1], hand_glb_tar_rot_th[..., 2])
                    hand_glb_rot_np = hand_glb_quat_th.numpy()
                    hand_glb_qtar_np = hand_glb_tar_quat_th.numpy()
                    
                    hand_qs_np = np.concatenate(
                        [ hand_qs_np[..., :3], hand_glb_rot_np, hand_qs_np[..., 6:] ], axis=-1
                    )
                    hand_qtars_np = np.concatenate(
                        [ hand_qtars_np[..., :3], hand_glb_qtar_np, hand_qtars_np[..., 6:] ], axis=-1
                    )
                    # hand_qs_np[..., 3:6] = hand_glb_rot_np
                    # hand_qtars_np[..., 3:6] = hand_glb_qtar_np
                # obj_pose_np = cu
            else:
                ts_to_hand_qs = cur_data['ts_to_hand_qs'] 
                ts_to_hand_qtars = cur_data['ts_to_qtars']
                if self.slicing_data:
                    sorted_ts = sorted(list(ts_to_hand_qs.keys()))
                    hand_qs_np = [
                        ts_to_hand_qs[i_ts] for i_ts in sorted_ts
                    ]
                    hand_qtars_np = [
                        ts_to_hand_qtars[i_ts] for i_ts in sorted_ts
                    ]
                else:
                    if 'ts_to_optimized_q_tars_wcontrolfreq' in cur_data:
                        ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
                    else:
                        ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_hand_tot_ctl_qtars']
                    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                        
                    max_hand_qs_kd = max(list(ts_to_hand_qs.keys()))
                    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
                    ctl_freq_tss = sorted(ctl_freq_tss) # ctl freq tss #
                    ctl_freq = 10 # 
                    ctl_freq_tss_expanded = [ min(max_hand_qs_kd, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
                    ts_to_hand_qs = {
                        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
                    }
                    hand_qs_np = [
                        ts_to_hand_qs[cur_ts] for cur_ts in ctl_freq_tss
                    ]
                    hand_qtars_np = [ # 
                        ts_to_hand_qtars[cur_ts] for cur_ts in ctl_freq_tss
                    ]
                hand_qs_np = np.stack(hand_qs_np, axis=0)
                hand_qtars_np = np.stack(hand_qtars_np, axis=0) 
                
            # cur_data_nm = data_nm #
            
            # tot_data_hand_qs.append(hand_qs_np)
            # tot_data_hand_qtars.append(hand_qtars_np)

            cur_clip_data = {
                'tot_verts': hand_qs_np[None],  # hand qtars np; tot verts # 
                # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
                'tot_verts_integrated_qdd_tau': hand_qtars_np[None], # 
                # 'task_setting': task_setting, # rotation euler angles? # 
                # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            } # grab inst tag #
            
            if self.task_cond_type == 'history_future':
                obj_pose_np = optimized_obj_pose[0]
                cur_clip_data.update(
                    {
                        'tot_obj_pose': obj_pose_np[None]
                    }
                )
                
            
            # cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
            # using the data fn as the data name # 
            if add_to_dict:
                self.data_name_to_data[data_fn] = cur_clip_data
        else:
            raise NotImplementedError(f"Unrecognized use_jointspace_seq: {self.use_jointspace_seq}")
        return cur_clip_data, hand_qs_np, hand_qtars_np
        
    
    def _preload_mocap_tracking_ctl_data(self,): # 
        # self.data_list #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        if self.single_inst: # 
            self.data_list = self.data_list[:1]
            self.data_inst_tag_list = self.data_inst_tag_list[:1]
        elif self.multi_inst:
            self.data_list = self.data_list[:10]
            self.data_inst_tag_list = self.data_inst_tag_list[:10]
        # tot_expanded_passive #
        forbid_data_inst_tags = ["ori_grab_s2_phone_call_1", "ori_grab_s2_phone_pass_1"]
        
        for i_data_inst, data_fn in enumerate(self.data_list):
            
            excluded = False 
            for cur_forbid_inst_tag in forbid_data_inst_tags:
                if cur_forbid_inst_tag in data_fn:
                    excluded = True
                    break
            if excluded: ## # excluded ##
                continue
            
            print(f"loading from {data_fn}")
            # preload the tracking #
            cur_clip_data, hand_qs_np, hand_qtars_np = self._preload_single_tracking_ctl_data(data_fn)
            tot_data_hand_qs.append(hand_qs_np)
            tot_data_hand_qtars.append(hand_qtars_np)
            self.data_name_to_data[data_fn] = cur_clip_data
            # cur_data = np.load(data_fn, allow_pickle=True).item()
            
            # if self.use_jointspace_seq:
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
                    
            #     # cur_data_nm = data_nm
                
            #     tot_data_hand_qs.append(hand_qs_np)
            #     tot_data_hand_qtars.append(hand_qtars_np)
                

            #     cur_clip_data = {
            #         'tot_verts': hand_qs_np[None],  # hand qtars np; tot verts # 
            #         # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd, 
            #         'tot_verts_integrated_qdd_tau': hand_qtars_np[None], # 
            #         # 'task_setting': task_setting, # rotation euler angles? # 
            #         # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            #     } # grab inst tag #
            #     cur_grab_inst_tag = self.data_inst_tag_list[i_data_inst]
            #     # using the data fn as the data name # 
            #     self.data_name_to_data[data_fn] = cur_clip_data
            # else:
            #     raise NotImplementedError(f"Unrecognized use_jointspace_seq: {self.use_jointspace_seq}")
        
        tot_data_hand_qs = np.concatenate(tot_data_hand_qs, axis=0)
        tot_data_hand_qtars = np.concatenate(tot_data_hand_qtars, axis=0)
        self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
        self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
        self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
        self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
        
        ### add to data statistics ###
        ### TODO: we should put object stats here --- obj stats should be calculated from the tracked trajectories ###
        self.data_statistics['avg_hand_qs'] = self.avg_hand_qs
        self.data_statistics['std_hand_qs'] = self.std_hand_qs
        self.data_statistics['avg_hand_qtars'] = self.avg_hand_qtars
        self.data_statistics['std_hand_qtars'] = self.std_hand_qtars
        
    
    
    def _get_obj_slicing_rot_trans(self, data_nm, st_idx, ed_idx):
        
        if self.task_cond_type == 'history_future':
            obj_pose = self.data_name_to_data[data_nm]['tot_obj_pose'][0]
            #  get the kine obj pose states # history future #
            
            # sliced_obj_pose = obj_pose[st_idx: ed_idx]
            sliced_obj_pose = obj_pose[st_idx - 1: ed_idx - 1 ]
            sliced_obj_trans, sliced_obj_ornt = sliced_obj_pose[:, :3], sliced_obj_pose[:, 3:]
            
            if self.use_kine_obj_pos_canonicalization:
                kine_info_dict = self.data_name_to_kine_info[data_nm]
                obj_trans = kine_info_dict['obj_trans']
                obj_ornt = kine_info_dict['obj_ornt']
                sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx - 1: ed_idx - 1], obj_ornt[st_idx - 1: ed_idx - 1]
                
        else: # sliced obj trans and the ornt #
            kine_info_dict = self.data_name_to_kine_info[data_nm]
            obj_trans = kine_info_dict['obj_trans']
            obj_ornt = kine_info_dict['obj_ornt']
            sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx: ed_idx], obj_ornt[st_idx: ed_idx]
        return sliced_obj_trans, sliced_obj_ornt
        # {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
        #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
        #     'obj_ornt': obj_ornt ,
        #     'obj_rot_euler': kine_obj_rot_euler_angles
        # }
    
    
    def _slicing_single_mocap_tracking_ctl_data(self, grab_opt_data_fn, cur_data_dict, kine_data_dict, add_to_dict=True):
        kine_qs = cur_data_dict['tot_verts'][0]
        q_tars = cur_data_dict['tot_verts_integrated_qdd_tau'][0]
        if self.task_cond_type == 'history_future':
            obj_pose = cur_data_dict['tot_obj_pose'][0]
        obj_trans = kine_data_dict['obj_trans']
        obj_ornt = kine_data_dict['obj_ornt']
        
        # slice # 
        print(f"kine_qs: {kine_qs.shape}, q_tars: {q_tars.shape}, obj_trans: {obj_trans.shape}, obj_ornt: {obj_ornt.shape}")
        
        slicing_st_idx = 0
        if self.task_cond_type == 'history_future':
            slicing_st_idx = 1
        
        self.tot_target_data_nm = []
        for i_slice in range(slicing_st_idx, kine_qs.shape[0] - self.slicing_ws, self.step_size):
            st_idx = i_slice # the start index #
            ed_idx = i_slice + self.slicing_ws
            
            slicing_idxes = list(range(st_idx, ed_idx))
            slicing_idxes = np.array(slicing_idxes, dtype=np.int32) # get the slicing idxes # 
            slicing_idxes = np.clip(slicing_idxes, a_min=0, a_max=kine_qs.shape[0] - 1)
            
            # ed_idx = min(i_slice + self.slicing_ws, kine_qs.shape[0])
            
            
            
            # sliced_obj_trans, sliced_obj_ornt = obj_trans[st_idx: ed_idx], obj_ornt[st_idx: ed_idx]
            sliced_obj_trans, sliced_obj_ornt = obj_trans[slicing_idxes], obj_ornt[slicing_idxes]
            
            # ge the slicing data obj trans #
            if self.task_cond_type == 'history_future':
                sliced_obj_pose = obj_pose[st_idx - 1: ed_idx - 1 ]
                first_frame_obj_trans = sliced_obj_pose[0, :3]
                first_frame_obj_ornt = sliced_obj_pose[0, 3:]
            else:
                first_frame_obj_trans = sliced_obj_trans[0, :3] # first farme obj trans #
            
            # cur_slice_kine_qs = kine_qs[st_idx: ed_idx]
            # cur_slice_q_tars = q_tars[st_idx: ed_idx]
            
            cur_slice_kine_qs = kine_qs[slicing_idxes]
            cur_slice_q_tars = q_tars[slicing_idxes]
            
            #### NOTE: A simple canonicalization ####
            cur_slice_kine_qs[:, :3] = cur_slice_kine_qs[:, :3] - first_frame_obj_trans[ None]
            cur_slice_q_tars[:, :3] = cur_slice_q_tars[:, :3] - first_frame_obj_trans[ None]
            #### NOTE: A simple canonicalization ####

            cur_slice_data = {
                'tot_verts': cur_slice_kine_qs[None],
                'tot_verts_integrated_qdd_tau': cur_slice_q_tars[None]
            }
            
            
            # TODO: change all the jugdement logic to this # 
            # 
            if  self.task_cond_type == 'history_future': # history future #
                history_st_idx = st_idx - self.slicing_ws
                # history_st_idx = max(0, history_st_idx)
                # history_ed_idx = st_idx # + 1
                # history_ed_idx = min(kine_qs.shape[0], history_ed_idx)
                # history_st_idx = max(0, history_st_idx)
                history_ed_idx = st_idx # + 1
                # history_ed_idx = min(kine_qs.shape[0], history_ed_idx) # hsitory idxes #
                history_idxes = list(range(history_st_idx, history_ed_idx))
                history_idxes = np.array(history_idxes, dtype=np.int32)
                history_idxes = np.clip(history_idxes, a_min=0, a_max=kine_qs.shape[0] - 1)
                history_kine_qs = kine_qs[history_idxes]
                tot_obj_pose = cur_data_dict['tot_obj_pose'][0]
                history_obj_pose = tot_obj_pose[history_idxes]
                history_obj_pose[:, :3] = history_obj_pose[:, :3] - first_frame_obj_trans[ None]
                history_kine_qs[:, :3] = history_kine_qs[:, :3] - first_frame_obj_trans[ None] # minus the first frame obj trans #
                
                ## the obj eulers may not be a good representation ## ## a representation ## ## # a representation #
                ## TODO: the obj eulers may not be a good representation ## a good representation ##
                # ### from the obj quaternion ornt to the obj euler rotations ##### rotations #####
                history_obj_trans, history_obj_ornt = history_obj_pose[:, :3], history_obj_pose[:, 3:]
                history_obj_rot_euler = []
                for ii_fr in range(history_obj_ornt.shape[0]):
                    cur_fr_obj_ornt = history_obj_ornt[ii_fr]
                    cur_fr_obj_rot_euler = R.from_quat(cur_fr_obj_ornt).as_euler('xyz', degrees=False) # as the rot eulers # 
                    history_obj_rot_euler.append(cur_fr_obj_rot_euler)
                history_obj_rot_euler = np.stack(history_obj_rot_euler, axis= 0) 
                
                # add the history information #
                history_info = {
                    'history_obj_pose': history_obj_pose[None ],
                    'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                    'first_frame_obj_trans': first_frame_obj_trans,
                    'first_frame_obj_ornt': first_frame_obj_ornt, # first frae obj ornt # an the trans # 
                    'history_obj_trans': history_obj_trans[None ], 
                    'history_obj_rot_euler': history_obj_rot_euler[None ],
                }
                cur_slice_data.update(history_info )
                # have he st_idx ? # 
            
            # cur_slice_data_nm = f"{cur_data_fn}_sted_{st_idx}_{ed_idx}"
            cur_slice_data_nm = grab_opt_data_fn.replace(".npy", f"_STED_{st_idx}_{ed_idx}.npy")
            
            if add_to_dict:
                self.data_name_to_data[cur_slice_data_nm] = cur_slice_data #  
            
            self.tot_target_data_nm.append(cur_slice_data_nm)
            if add_to_dict:
                self.data_name_list.append(cur_slice_data_nm)
            
            # slicing tracking kienmatics data ###
            # print(f"[Slicing tracking kine data] data_nm: {data_nm}")
            # sted_info = data_nm.split("/")[-1].split(".npy")[0].split('_STED_')[1]
            # st_idx, ed_idx = sted_info.split("_")
            # st_idx, ed_idx = int(st_idx), int(ed_idx)
            # # else:
            # #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0] # sliced hand qs #
            
            # sliced_hand_qs = kine_data_dict['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
            # sliced_obj_trans = kine_data_dict['obj_trans'][st_idx: ed_idx]
            # sliced_obj_ornt = kine_data_dict['obj_ornt'][st_idx: ed_idx]
            # sliced_obj_rot_euler = kine_data_dict['obj_rot_euler'][st_idx: ed_idx]
            
            # slicing_idxes
            
            sliced_hand_qs = kine_data_dict['hand_qs'][slicing_idxes][:, :self.nn_hands_dof]
            sliced_obj_trans = kine_data_dict['obj_trans'][slicing_idxes]
            sliced_obj_ornt = kine_data_dict['obj_ornt'][slicing_idxes]
            sliced_obj_rot_euler = kine_data_dict['obj_rot_euler'][slicing_idxes]
            
            obj_verts = kine_data_dict['obj_verts']
            
            first_frame_obj_trans = sliced_obj_trans[0, :3]
            sliced_hand_qs[:, :3] = sliced_hand_qs[:, :3] - first_frame_obj_trans[None]
            sliced_obj_trans = sliced_obj_trans - first_frame_obj_trans[None]
            
            kine_info_dict = {
                'obj_verts': obj_verts, 
                'hand_qs': sliced_hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
                'obj_trans': sliced_obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
                'obj_ornt': sliced_obj_ornt ,
                'obj_rot_euler': sliced_obj_rot_euler
            }
            if add_to_dict:
                self.data_name_to_kine_info[cur_slice_data_nm] = kine_info_dict
                
    
    # slice mocap ctl data #
    def _slicing_mocap_tracking_ctl_data(self, ):
        ## slice the tracking ctl data ## # tracking ctl # use_kine_obj_pos_canonicalization # kine obj pos canonicalization #
        tot_data_hand_qs = []
        tot_data_hand_qtars = []
        
        self.all_slices_data_inst_tag_list = []
        self.all_slices_data_name_to_data = {}
        for cur_data_fn in self.data_name_to_data:
            cur_data = self.data_name_to_data[cur_data_fn]
            kine_qs =cur_data['tot_verts'][0]
            q_tars = cur_data['tot_verts_integrated_qdd_tau'][0]
            
            slicing_st_idx = 0 
            if self.task_cond_type == 'history_future':
                slicing_st_idx = 1
        
            slicing_ed_idx = kine_qs.shape[0] - self.slicing_ws
            if slicing_ed_idx < slicing_st_idx + 1:
                slicing_ed_idx = slicing_st_idx + 1 ## increase the slicing ed idx 
            # step size = 30
            # print(f"kine_qs: {kine_qs.shape}, q_tars: {q_tars.shape}")
            for i_slice in range(slicing_st_idx, slicing_ed_idx, self.step_size):
                st_idx = i_slice
                ed_idx = i_slice + self.slicing_ws
                slicing_idxes = list(range(st_idx, ed_idx))
                slicing_idxes = np.array(slicing_idxes, dtype=np.int32) # 
                slicing_idxes = np.clip(slicing_idxes, a_min=0, a_max=kine_qs.shape[0] - 1) #
                # ed_idx = min(i_slice + self.slicing_ws, kine_qs.shape[0])
                
                # task cond should not ## obj slicing rot trans #
                sliced_obj_trans, sliced_obj_ornt = self._get_obj_slicing_rot_trans(cur_data_fn, st_idx, ed_idx)
                
                if sliced_obj_trans.shape[0] == 0:
                    continue
                
                first_frame_obj_trans = sliced_obj_trans[0, :3]
                first_frame_obj_ornt = sliced_obj_ornt[0, :]
                
                # cur_slice_kine_qs = kine_qs[st_idx: ed_idx]
                # cur_slice_q_tars = q_tars[st_idx: ed_idx]
                
                cur_slice_kine_qs = kine_qs[slicing_idxes]
                cur_slice_q_tars = q_tars[slicing_idxes]
                
                #### NOTE: A simple canonicalization ####
                cur_slice_kine_qs[:, :3] = cur_slice_kine_qs[:, :3] - first_frame_obj_trans[ None]
                cur_slice_q_tars[:, :3] = cur_slice_q_tars[:, :3] - first_frame_obj_trans[ None]
                #### NOTE: A simple canonicalization ####
                
                cur_slice_data = {
                    'tot_verts': cur_slice_kine_qs[None],
                    'tot_verts_integrated_qdd_tau': cur_slice_q_tars[None],
                    'slicing_idxes': slicing_idxes,
                }
                
                # TODO: change all the jugdement logic to this # 
                if  self.task_cond_type == 'history_future':
                    # history_st_idx = st_idx - self.slicing_ws
                    history_st_idx = st_idx - self.history_ws
                    # history_st_idx = max(0, history_st_idx)
                    history_ed_idx = st_idx # + 1
                    # history_ed_idx = min(kine_qs.shape[0], history_ed_idx)
                    history_idxes = list(range(history_st_idx, history_ed_idx))
                    history_idxes = np.array(history_idxes, dtype=np.int32)
                    history_idxes = np.clip(history_idxes, a_min=0, a_max=kine_qs.shape[0] - 1) # 
                    # history_kine_qs = kine_qs[history_st_idx : history_ed_idx]
                    history_kine_qs = kine_qs[history_idxes]
                    tot_obj_pose = cur_data['tot_obj_pose'][0]
                    # history_obj_pose = tot_obj_pose[history_st_idx: history_ed_idx]
                    history_obj_pose = tot_obj_pose[history_idxes]
                    history_obj_pose[:, :3] = history_obj_pose[:, :3] - first_frame_obj_trans[ None]
                    history_kine_qs[:, :3] = history_kine_qs[:, :3] - first_frame_obj_trans[ None] # minus the first frame obj trans # # j
                    
                    ## the obj eulers may not be a good representation ## ## a representation ## ## # a representation #
                    ## TODO: the obj eulers may not be a good representation ## a good representation ##
                    # ### from the obj quaternion ornt to the obj euler rotations ##### rotations #####
                    history_obj_trans, history_obj_ornt = history_obj_pose[:, :3], history_obj_pose[:, 3:]
                    history_obj_rot_euler = []
                    for ii_fr in range(history_obj_ornt.shape[0]):
                        cur_fr_obj_ornt = history_obj_ornt[ii_fr]
                        cur_fr_obj_rot_euler = R.from_quat(cur_fr_obj_ornt).as_euler('xyz', degrees=False) # as the rot eulers # 
                        history_obj_rot_euler.append(cur_fr_obj_rot_euler)
                    history_obj_rot_euler = np.stack(history_obj_rot_euler, axis= 0) 
                    
                    # add the history information #
                    history_info = {
                        'history_obj_pose': history_obj_pose[None ],
                        'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                        'first_frame_obj_trans': first_frame_obj_trans,
                        'first_frame_obj_ornt': first_frame_obj_ornt, # first frae obj ornt # an the trans # 
                        'history_obj_trans': history_obj_trans[None ], 
                        'history_obj_rot_euler': history_obj_rot_euler[None ],
                        'history_idxes': history_idxes
                    }
                    cur_slice_data.update(history_info )
                    
                    
                
                # cur_slice_data_nm = f"{cur_data_fn}_sted_{st_idx}_{ed_idx}"
                cur_slice_data_nm = cur_data_fn.replace(".npy", f"_STED_{st_idx}_{ed_idx}.npy")
                self.all_slices_data_name_to_data[cur_slice_data_nm] = cur_slice_data #  
                
                self.all_slices_data_inst_tag_list.append(cur_slice_data_nm)
                tot_data_hand_qs.append(cur_slice_kine_qs)
                tot_data_hand_qtars.append(cur_slice_q_tars)

        tot_data_hand_qs = np.concatenate(tot_data_hand_qs, axis=0)
        tot_data_hand_qtars = np.concatenate(tot_data_hand_qtars, axis=0)
        self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
        self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
        self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
        self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
        
        ### add to data statistics ###
        self.data_statistics['avg_hand_qs'] = self.avg_hand_qs
        self.data_statistics['std_hand_qs'] = self.std_hand_qs
        self.data_statistics['avg_hand_qtars'] = self.avg_hand_qtars
        self.data_statistics['std_hand_qtars'] = self.std_hand_qtars
        
        
        self.data_name_list = self.all_slices_data_inst_tag_list
        self.data_name_to_data = self.all_slices_data_name_to_data
    
    
    
    def _preload_kine_taskcond_data(self, ):
        if self.single_inst:
            self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
        
        
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            kine_data_tag = "passive_active_info_"
            kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
            return kine_object_type
           
        
        def load_traj_info(kine_dict_fn, maxx_ws):
            kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            kine_obj_transl = kine_dict['object_transl']
            kine_obj_rot_quat = kine_dict['object_rot_quat']
            kine_robot_hand_qs = kine_dict['robot_delta_states_weights_np']
            kine_obj_transl = kine_obj_transl[: maxx_ws]
            kine_obj_rot_quat = kine_obj_rot_quat[: maxx_ws]
            kine_robot_hand_qs = kine_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            kine_obj_rot_euler_angles = []
            for i_fr in range(kine_obj_rot_quat.shape[0]):
                cur_rot_quat = kine_obj_rot_quat[i_fr] #
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
                kine_obj_rot_euler_angles.append(cur_rot_euler)
            kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            return kine_obj_transl, kine_obj_rot_euler_angles, kine_robot_hand_qs
            
        def load_obj_pcs(kine_dict_fn, nn_sampled_pts):
            object_type = parse_kine_data_fn_into_object_type(kine_dict_fn)
            # kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            # kine_obj_transl = k
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes][:nn_sampled_pts]
            # tot_object_verts.append(obj_verts)/
            return obj_verts
        
        # task inheriting dict info ## inheritng dict info #
        maxx_ws = 149
        for i_data, data_dict in enumerate(self.task_inheriting_dict_info):
            # cur_inheriting_dict = {
            #     'fa_objtype': cur_parent_objtype, # 
            #     'fa_trajtype': cur_parent_trajtype, # 
            #     'ch_objtype': cur_child_objtype, # 
            #     'ch_trajtype': cur_child_trajtype
            # }
            cur_fa_objtype = data_dict['fa_objtype']
            cur_fa_trajtype = data_dict['fa_trajtype']
            cur_ch_objtype = data_dict['ch_objtype']
            cur_ch_trajtype = data_dict['ch_trajtype']
            
            ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_trajtype]
            fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_trajtype]
            
            ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_objtype]
            fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_objtype]
            
            ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
            fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
            
            ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
            fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
            
            
            
            # 
            sv_dict = {
                'obj_verts': fa_obj_verts,
                'robot_hand_qs': fa_robot_hand_qs,
                'obj_rot_euler': fa_obj_rot_euler,
                'obj_transl': fa_obj_transl
            }
            cond_sv_dict = {
                'cond_obj_verts': ch_obj_verts,
                'cond_robot_hand_qs': ch_robot_hand_qs,
                'cond_obj_rot_euler': ch_obj_rot_euler,
                'cond_obj_transl': ch_obj_transl
            }
            sv_dict.update(cond_sv_dict)
            
            
            for key in sv_dict:
                print(f"key: {key}, val: {sv_dict[key].shape }")
            
            # self.data_name_to_data[cur_kine_data_fn] = sv_dict
            self.tot_data_dict_list.append(sv_dict)
            
            # object_type = parse_kine_data_fn_into_object_type(cur_kine_data_fn)
            # self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            # grab_mesh_fn = f"{object_type}.obj"
            # grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            
            # obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            # obj_verts = obj_mesh.vertices # nn_pts x 3
            # random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            # obj_verts = obj_verts[random_sampled_idxes][:maxx_ws]
            # tot_object_verts.append(obj_verts)
            
            # ch_kine_traj_dict = np.load(ch_kine_traj_dict_fn, allow_pickle=True).item()
            # fa_kine_traj_dict = np.load(fa_kine_traj_dict_fn, allow_pickle=True).item()
            
            # ch_obj_transl = ch_kine_traj_dict['object_transl']
            # ch_obj_rot_quat = ch_kine_traj_dict['object_rot_quat']
            # ch_robot_hand_qs = ch_kine_traj_dict['robot_delta_states_weights_np']
            # maxx_ws = 150
            # ch_obj_transl = ch_obj_transl[: maxx_ws]
            # ch_obj_rot_quat = ch_obj_rot_quat[: maxx_ws]
            # ch_robot_hand_qs = ch_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            
            # ch_obj_rot_euler_angles = []
            # for i_fr in range(ch_obj_rot_quat.shape[0]):
            #     cur_rot_quat = ch_obj_rot_quat[i_fr]
            #     cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=True) # get the rotation euler 
            #     ch_obj_rot_euler_angles.append(cur_rot_euler)
            # ch_obj_rot_euler_angles = np.stack(ch_obj_rot_euler_angles, axis=0)
    
    
    def _preload_kine_target_taskcond_data(self, ):
        # if self.single_inst:
        #     self.task_inheriting_dict_info = self.task_inheriting_dict_info[:1]
        
        # tot_obj_transl = []
        # tot_obj_rot_euler = []
        # tot_hand_qs = []
        # tot_obj_verts = []
        
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            # kine_data_fn = kine_data_fn.
            kine_data_tag = "passive_active_info_"
            kine_object_type = kine_data_fn.split("/")[-1].split(".")[0][len(kine_data_tag): ]
            return kine_object_type
           
        
        def load_traj_info(kine_dict_fn, maxx_ws):
            kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            kine_obj_transl = kine_dict['object_transl']
            kine_obj_rot_quat = kine_dict['object_rot_quat']
            kine_robot_hand_qs = kine_dict['robot_delta_states_weights_np']
            kine_obj_transl = kine_obj_transl[: maxx_ws]
            kine_obj_rot_quat = kine_obj_rot_quat[: maxx_ws]
            kine_robot_hand_qs = kine_robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            kine_obj_rot_euler_angles = []
            for i_fr in range(kine_obj_rot_quat.shape[0]):
                cur_rot_quat = kine_obj_rot_quat[i_fr] # dkine obj rot quat #
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False) # get the rotation euler 
                kine_obj_rot_euler_angles.append(cur_rot_euler)
            kine_obj_rot_euler_angles = np.stack(kine_obj_rot_euler_angles, axis=0)
            
            return kine_obj_transl, kine_obj_rot_euler_angles, kine_robot_hand_qs
            
        def load_obj_pcs(kine_dict_fn, nn_sampled_pts):
            object_type = parse_kine_data_fn_into_object_type(kine_dict_fn)
            # kine_dict = np.load(kine_dict_fn, allow_pickle=True).item()
            # kine_obj_transl = k
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            # get the object mesh #
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes][:nn_sampled_pts]
            # tot_object_verts.append(obj_verts)/
            return obj_verts
        
        
        maxx_ws = 149
        
        ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        
        ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[self.target_grab_inst_tag]
        
        ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
        fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
        
        ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
        fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
        
    # 
        sv_dict = {
            'obj_verts': fa_obj_verts,
            'robot_hand_qs': fa_robot_hand_qs,
            'obj_rot_euler': fa_obj_rot_euler,
            'obj_transl': fa_obj_transl
        }
        cond_sv_dict = {
            'cond_obj_verts': ch_obj_verts,
            'cond_robot_hand_qs': ch_robot_hand_qs,
            'cond_obj_rot_euler': ch_obj_rot_euler,
            'cond_obj_transl': ch_obj_transl
        }
        sv_dict.update(cond_sv_dict)
        
        
        self.tot_data_dict_list = []
        self.tot_data_dict_list.append(sv_dict)
        
        # for i_data, data_dict in enumerate(self.task_inheriting_dict_info):
        #     # cur_inheriting_dict = {
        #     #     'fa_objtype': cur_parent_objtype, # 
        #     #     'fa_trajtype': cur_parent_trajtype, # 
        #     #     'ch_objtype': cur_child_objtype, # 
        #     #     'ch_trajtype': cur_child_trajtype
        #     # }
        #     cur_fa_objtype = data_dict['fa_objtype']
        #     cur_fa_trajtype = data_dict['fa_trajtype']
        #     cur_ch_objtype = data_dict['ch_objtype']
        #     cur_ch_trajtype = data_dict['ch_trajtype']
            
        #     ch_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_trajtype]
        #     fa_kine_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_trajtype]
            
        #     ch_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_ch_objtype]
        #     fa_obj_traj_dict_fn = self.objtype_to_tracking_sv_info[cur_fa_objtype]
            
        #     ch_obj_transl, ch_obj_rot_euler, ch_robot_hand_qs = load_traj_info(ch_obj_traj_dict_fn, maxx_ws=maxx_ws)
        #     fa_obj_transl, fa_obj_rot_euler, fa_robot_hand_qs = load_traj_info(fa_obj_traj_dict_fn, maxx_ws=maxx_ws)
            
        #     ch_obj_verts = load_obj_pcs(ch_kine_traj_dict_fn, maxx_ws)
        #     fa_obj_verts = load_obj_pcs(fa_kine_traj_dict_fn, maxx_ws)
            
            
            
        #     # 
        #     sv_dict = {
        #         'obj_verts': fa_obj_verts,
        #         'robot_hand_qs': fa_robot_hand_qs,
        #         'obj_rot_euler': fa_obj_rot_euler,
        #         'obj_transl': fa_obj_transl
        #     }
        #     cond_sv_dict = {
        #         'cond_obj_verts': ch_obj_verts,
        #         'cond_robot_hand_qs': ch_robot_hand_qs,
        #         'cond_obj_rot_euler': ch_obj_rot_euler,
        #         'cond_obj_transl': ch_obj_transl
        #     }
        #     sv_dict.update(cond_sv_dict)
            
            
        #     for key in sv_dict:
        #         print(f"key: {key}, val: {sv_dict[key].shape }")
            
        #     # self.data_name_to_data[cur_kine_data_fn] = sv_dict
        #     self.tot_data_dict_list.append(sv_dict)
            
    

    
    def _preload_kine_data(self,) :
        if self.single_inst or self.debug:
            self.data_list = self.data_list[:1]
       
        def parse_kine_data_fn_into_object_type(kine_data_fn):
            if 'taco' in kine_data_fn:
                passive_act_pure_tag = "passive_active_info_ori_grab_s2_phone_call_1_interped_"
                
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
           
        # tot_expanded_passve_meshes = [] # preload kine data #
        
        tot_obj_transl = []
        tot_obj_rot_euler = []
        tot_hand_qs = []
        
        tot_object_verts = []
        
        print(f"Start loading kinematics data: {len(self.data_list)}")
        for i_kine, kine_fn in enumerate(self.data_list):
            print(f"[{i_kine}/{len(self.data_list)}] {kine_fn}")
            # kine_saved_info: dict_keys(['passive_meshes', 'active_meshes', 'passive_mesh_normals', 'object_transl', 'object_rot_quat', 'ts_to_allegro', 'ts_to_mano_fingers_np', 'ts_to_robot_fingers_np', 'robot_delta_states_weights_np']) 
            cur_kine_data_fn = self.data_list[i_kine]
            cur_kine_data = np.load(cur_kine_data_fn, allow_pickle=True).item()
            
            obj_transl = cur_kine_data['object_transl']
            obj_rot_quat = cur_kine_data['object_rot_quat']
            robot_hand_qs = cur_kine_data['robot_delta_states_weights_np']
            
            maxx_ws = 400
            maxx_ws = min(maxx_ws, obj_transl.shape[0])
            maxx_ws = min(maxx_ws, robot_hand_qs.shape[0])
            obj_transl = obj_transl[: maxx_ws]
            obj_rot_quat = obj_rot_quat[: maxx_ws]
            robot_hand_qs = robot_hand_qs[: maxx_ws][:, :self.nn_hands_dof]
            
            # transform the rot_quat # 
            obj_rot_euler_angles = []
            for i_fr in range(obj_rot_quat.shape[0]):
                cur_rot_quat = obj_rot_quat[i_fr]
                cur_rot_euler = R.from_quat(cur_rot_quat).as_euler('xyz', degrees=False)
                obj_rot_euler_angles.append(cur_rot_euler)
            obj_rot_euler_angles = np.stack(obj_rot_euler_angles, axis=0)
            
            # obj_transl and obj_rot_euler_angles # 
            tot_obj_transl.append(obj_transl)
            tot_obj_rot_euler.append(obj_rot_euler_angles)
            tot_hand_qs.append(robot_hand_qs)
        
            object_type = parse_kine_data_fn_into_object_type(cur_kine_data_fn)
            self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
            grab_mesh_fn = f"{object_type}.obj"
            grab_mesh_fn = os.path.join(self.grab_obj_mesh_sv_folder, grab_mesh_fn)
            
            
            
            obj_mesh = trimesh.load_mesh(grab_mesh_fn)
            obj_verts = obj_mesh.vertices # nn_pts x 3
            random_sampled_idxes = np.random.permutation(obj_verts.shape[0])[: self.maxx_nn_pts]
            obj_verts = obj_verts[random_sampled_idxes] # [:maxx_ws]
            tot_object_verts.append(obj_verts)
            
            sv_dict = {
                'obj_verts': obj_verts,
                'robot_hand_qs': robot_hand_qs,
                'obj_rot_euler': obj_rot_euler_angles,
                'obj_transl': obj_transl,
                'object_type': object_type,
                'obj_rot_quat': obj_rot_quat
            }
            self.data_name_to_data[cur_kine_data_fn] = sv_dict
            
            
            
        
        tot_obj_transl = np.concatenate(tot_obj_transl, axis=0)
        tot_obj_rot_euler = np.concatenate(tot_obj_rot_euler, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        
        if not (self.sampling and len(self.target_grab_inst_tag) > 0):
            self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
            self.std_obj_transl = np.std(tot_obj_transl, axis=0)
            self.avg_obj_rot_euler = np.mean(tot_obj_rot_euler, axis=0)
            self.std_obj_rot_euler = np.std(tot_obj_rot_euler, axis=0)
            self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
            self.std_hand_qs = np.std(tot_hand_qs, axis=0)
            # 
            self.avg_object_verts = np.mean(tot_object_verts, axis=0)
            self.std_object_verts = np.std(tot_object_verts, axis=0)
            
            
            self.data_statistics = {
                'avg_obj_transl': self.avg_obj_transl, 
                'std_obj_transl': self.std_obj_transl,
                'avg_obj_rot_euler': self.avg_obj_rot_euler,
                'std_obj_rot_euler': self.std_obj_rot_euler,
                'avg_obj_verts': self.avg_object_verts,
                'std_obj_verts': self.std_object_verts,
                'avg_hand_qs': self.avg_hand_qs, 
                'std_hand_qs': self.std_hand_qs
            }
        
        
    
    def _preload_inheriting_data(self, ):
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
            
        for cur_inherit_info in self.tot_inheriting_infos:
            cur_fa_task_fn = cur_inherit_info['inherit_fr_pts_info_fn']
            cur_ch_task_fn = cur_inherit_info['to_task_pts_info_fn']
            
            ## fa and ch ##
            cur_fa_data = self._load_data_from_data_name(cur_fa_task_fn)
            cur_ch_data = self._load_data_from_data_name(cur_ch_task_fn)
            
            cur_ch_task_setting = [
                float(self.obj_name_to_idx[cur_ch_data['object_type']]) - 0.5, float(cur_ch_data['task_rot']), float(cur_ch_data['object_size_x'])
            ]
            cur_fa_task_setting = [
                float(self.obj_name_to_idx[cur_fa_data['object_type']]) - 0.5, float(cur_fa_data['task_rot']), float(cur_fa_data['object_size_x'])
            ]
            cur_inheri_data = {
                'fa_task_setting': cur_fa_task_setting, 
                'ch_task_setting' : cur_ch_task_setting
            }
            self.data_name_to_data[cur_ch_task_fn] = cur_inheri_data
    
    def _preload_data(self, ):
        
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        
        self.tot_object_type = []
        self.tot_rot = []
        self.tot_obj_size = []
        for data_nm in self.data_name_list:
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            ### training setting ###
            if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond']:
                unopt_verts = cur_data['unopt_verts']
                tot_verts = unopt_verts
                unopt_qtar_verts = cur_data['unopt_qtar_verts']
                opt_verts = cur_data['opt_verts']
                opt_qtar_verts = cur_data['opt_qtar_verts']
                
                ## unopt verts ##
                cur_clip_data = {
                    'unopt_verts': unopt_verts,
                    'unopt_qtar_verts': unopt_qtar_verts,
                    'opt_verts': opt_verts,
                    'opt_qtar_verts': opt_qtar_verts
                }
                cur_data_nm = data_nm
                self.data_name_to_data[data_nm] = cur_clip_data
            
            
            else: 
                if self.use_jointspace_seq:
                    ts_to_hand_qs = cur_data['ts_to_hand_qs']
                    # ts_to_hand_qtars = cur_data['ts_to_qtars'] 
    
                    ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
                    # ts_to_
                    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
                     
                    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
                    ctl_freq_tss = sorted(ctl_freq_tss)
                    ctl_freq = 10
                    ctl_freq_tss_expanded = [ min(500 - 1, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
                    ts_to_hand_qs = {
                        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss)) # cu
                    }
                    # nn_ts x 
                    # tot_qs = tot_qs[None]
                    ts_keys_sorted = sorted(list(ts_to_hand_qs.keys()))
                    hand_qs_np = [
                        ts_to_hand_qs[cur_ts] for cur_ts in ts_to_hand_qs
                    ]
                    hand_qtars_np = [
                        ts_to_hand_qtars[cur_ts] for cur_ts in ts_to_hand_qtars
                    ]
                    hand_qs_np = np.stack(hand_qs_np, axis=0)
                    hand_qtars_np = np.stack(hand_qtars_np, axis=0) ## tethte qtarsnp 
                    
                    # hand_qs_np = hand_qs_np[]
                    
                    cur_data_nm = data_nm
                    
                    task_setting = {
                        'object_type': self.obj_name_to_idx[cur_data['object_type']],
                        'task_rot': cur_data['task_rot'],
                        'object_size_x': cur_data['object_size_x']
                    }
                    
                    self.tot_object_type.append(task_setting['object_type'])
                    self.tot_rot.append(task_setting['task_rot'])
                    self.tot_obj_size.append(task_setting['object_size_x']) ## get object size x ##
                    
                    
                    cur_clip_data = {
                        'tot_verts': hand_qs_np[None], 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': hand_qtars_np[None],
                        'task_setting': task_setting
                        # 'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                
                else:
                    # selected_frame_verts, selected_frame_qtars_verts
                    # tot_verts tot_verts_integrated_qdd_tau
                    ## convert to the interested first frame's pose ## then chnage the pose of that data ##
                    
                    tot_verts = cur_data['tot_verts']
                    
                    if self.use_static_first_frame:
                        tot_verts_first_frame = tot_verts[:, 0:1]
                        tot_verts = np.repeat(tot_verts_first_frame, tot_verts.shape[1], axis=1)
                    # print(f"tot_verts: {}")
                    
                    tot_verts_integrated_qdd_tau = cur_data['tot_qtar_verts']
                    if 'tot_qtar_verts_s2' not in cur_data:
                        tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts'].copy()
                    else:
                        tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts_s2']
                    
                    
                    # nn_ts x nn_verts x 3 #
                    # sequence length ? # # tot verts # #
                    self.nn_seq_len = tot_verts.shape[1]
                    
                    ##### inspect ######
                    mean_tot_verts = np.mean(tot_verts, axis=1)
                    mean_tot_verts = np.mean(mean_tot_verts, axis=0)
                    
                    mean_tot_verts_qdd = np.mean(tot_verts_integrated_qdd_tau, axis=1)
                    mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
                    
                    mean_tot_verts_qdd_s2 = np.mean(tot_verts_integrated_qdd_tau_s2, axis=1)
                    mean_tot_verts_qdd_s2 = np.mean(mean_tot_verts_qdd_s2, axis=0)
                    
                    print(f"mean_tot_verts: {mean_tot_verts}, mean_tot_verts_qdd: {mean_tot_verts_qdd}, mean_tot_verts_qdd_s2: {mean_tot_verts_qdd_s2}")
                    ##### inspect ######
                    
                    cur_data_nm = data_nm
                    cur_clip_data = {
                        'tot_verts': tot_verts, 
                        # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                        'tot_verts_integrated_qdd_tau': tot_verts_integrated_qdd_tau,
                        'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
                    }
                    self.data_name_to_data[cur_data_nm] = cur_clip_data
                
            
            ###### not use jointspace seq ######
            if not self.use_jointspace_seq:
                init_verts = tot_verts[:, 0]
                particle_init_xs_th = torch.from_numpy(init_verts).float()
                
                if self.specified_sampled_particle_idxes_fn is not None and len(self.specified_sampled_particle_idxes_fn) > 0:
                    sampled_particle_idxes_sv_fn = self.specified_sampled_particle_idxes_fn
                else:
                    if 'allegro_flat_fivefin_yscaled_finscaled' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_finscaled_sampled_particle_idxes.npy")
                    elif 'allegro_flat_fivefin_yscaled' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_sampled_particle_idxes.npy")
                    elif 'allegro_flat_fivefin' in data_nm:
                        sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_sampled_particle_idxes.npy")
                    else:
                        ## al
                        ### get the particle idxes  ###
                        # get partcle init xs #
                        sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
                
                if not os.path.exists(sampled_particle_idxes_sv_fn):
                    sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                    np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)
                else:
                    sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True)
                
                self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
        if self.use_jointspace_seq:
            self.tot_object_type = np.array(self.tot_object_type, dtype=np.float32)  ### (nn_instances, )
            self.tot_rot = np.array(self.tot_rot, dtype=np.float32)
            self.tot_obj_size = np.array(self.tot_obj_size, dtype=np.float32)
            self.avg_obj_type = np.mean(self.tot_object_type)
            self.avg_rot = np.mean(self.tot_rot)
            self.avg_obj_size = np.mean(self.tot_obj_size)
            self.std_obj_type = np.std(self.tot_object_type)
            self.std_rot = np.std(self.tot_rot)
            self.std_obj_size = np.std(self.tot_obj_size)
            
            self.avg_task_setting = np.array(
                [self.avg_obj_type, self.avg_rot, self.avg_obj_size], dtype=np.float32
            )
            self.std_task_setting = np.array(
                [self.std_obj_type, self.std_rot, self.std_obj_size], dtype=np.float32
            )
                
        
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")
        self.data_name_list = list(self.data_name_to_data.keys()) # data name to data # 
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_kine_v2(self, data_dict):
        # eps = 1e-6
        # obj_verts = (obj_verts - self.avg_obj_verts[None][None]) / (self.std_obj_verts[None][None] + eps)
        
        eps = 1e-6
        # inv scale data kine info #
        data_X = data_dict['X']
        data_E = data_dict['E']
        
        avg_obj_verts_th = torch.from_numpy(self.avg_object_verts).float().cuda()
        std_obj_verts_th = torch.from_numpy(self.std_object_verts).float().cuda()
        avg_hand_qs_th = torch.from_numpy(self.avg_hand_qs).float().cuda()
        std_hand_qs_th = torch.from_numpy(self.std_hand_qs).float().cuda()
        avg_obj_rot_euler_th = torch.from_numpy(self.avg_obj_rot_euler).float().cuda()
        std_obj_rot_euler_th = torch.from_numpy(self.std_obj_rot_euler).float().cuda()
        avg_obj_transl_th = torch.from_numpy(self.avg_obj_transl).float().cuda()
        std_obj_transl_th = torch.from_numpy(self.std_obj_transl).float().cuda()
        
        
        data_E = data_E[:, 0, :, :]
        dec_hand_qs = data_E[:, :, : self.nn_hands_dof]
        dec_obj_transl = data_E[:, :, self.nn_hands_dof: self.nn_hands_dof + 3]
        dec_obj_rot_euler = data_E[:, :, self.nn_hands_dof + 3: ]
        
        inv_scaled_hand_qs = (dec_hand_qs * (std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_transl = (dec_obj_transl * (std_obj_transl_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_transl_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_rot_euler = (dec_obj_rot_euler * (std_obj_rot_euler_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_rot_euler_th.unsqueeze(0).unsqueeze(0)
        inv_scaled_obj_verts = (data_X * (std_obj_verts_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_verts_th.unsqueeze(0).unsqueeze(0)
        
        
        if self.task_cond:
            cond_data_X = data_dict['X_cond']
            cond_data_E = data_dict['E_cond']
            cond_data_E = cond_data_E[:, 0, :, :]
            
            dec_cond_hand_qs = cond_data_E[:, :, : self.nn_hands_dof]
            dec_cond_obj_transl = cond_data_E[:, :, self.nn_hands_dof: self.nn_hands_dof + 3]
            dec_cond_obj_rot_euler = cond_data_E[:, :, self.nn_hands_dof + 3: ]
            
            inv_scaled_cond_hand_qs = (dec_cond_hand_qs * (std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
            inv_scaled_cond_obj_transl = (dec_cond_obj_transl * (std_obj_transl_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_transl_th.unsqueeze(0).unsqueeze(0)
            inv_scaled_cond_obj_rot_euler = (dec_cond_obj_rot_euler * (std_obj_rot_euler_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_rot_euler_th.unsqueeze(0).unsqueeze(0)
            
            inv_scaled_cond_obj_verts = (cond_data_X * (std_obj_verts_th.unsqueeze(0).unsqueeze(0) + eps)) + avg_obj_verts_th.unsqueeze(0).unsqueeze(0)
            cond_rt_dict = {
                'cond_obj_verts': inv_scaled_cond_obj_verts,
                'cond_hand_qs': inv_scaled_cond_hand_qs,
                'cond_obj_transl': inv_scaled_cond_obj_transl,
                'cond_obj_rot_euler': inv_scaled_cond_obj_rot_euler
            }
            
            # dec_obj_transl = cond_data_E[:, :, :3]
            
            
        
        # obj_verts_avg_th = torch.from_numpy(self.avg_obj_verts).float().cuda()
        # obj_verts_std_th = torch.from_numpy(self.std_obj_verts).float().cuda() ## get the avg and std object vertices # 
        # # (3,) - dim obj_verts_avg and obj_verts_std # 
        # data_E = (obj_verts_std_th.unsqueeze(0).unsqueeze(0).unsqueeze(0) + eps ) * data_E + obj_verts_avg_th.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        
        # tot_verts = data_X
        # tot_verts_integrated_qdd_tau = data_E 
        rt_dict = {
            # 'tot_verts': data_X,
            # 'tot_verts_integrated_qdd_tau': data_E,
            # 'tot_verts_integrated_qdd_tau_s2': data_E # inv scaled #
            'obj_verts': inv_scaled_obj_verts,
            'hand_qs': inv_scaled_hand_qs,
            'obj_transl': inv_scaled_obj_transl,
            'obj_rot_euler': inv_scaled_obj_rot_euler
        }
        
        if self.task_cond:
            rt_dict.update(
                cond_rt_dict # get the cond rt dict #
            )
            
        for key in data_dict:
            if key not in rt_dict:
                rt_dict[key] = data_dict[key]
        
        return rt_dict
        
        
    
    def inv_scale_data_v2(self, data_dict, data_nm=None, notarget=False): # bsz x nn_particles x feat_dim #
        
        return data_dict
        
        if self.kine_diff:
            rt_dict = self.inv_scale_data_kine_v2(data_dict=data_dict)
            return rt_dict
        
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        
        scaled_keys = []
        
        if self.use_jointspace_seq:
            
            data_X = data_X[:, 0]
            data_E = data_E[:, 0]
            
            if self.tracking_ctl_diff:
                # self.avg_hand_qs = np.mean(tot_data_hand_qs, axis=0)
                # self.std_hand_qs = np.std(tot_data_hand_qs, axis=0)
                # self.avg_hand_qtars = np.mean(tot_data_hand_qtars, axis=0)
                # self.std_hand_qtars = np.std(tot_data_hand_qtars, axis=0)
                
                self.avg_hand_qs_th = torch.from_numpy(self.avg_hand_qs).float().cuda() #
                self.std_hand_qs_th = torch.from_numpy(self.std_hand_qs).float().cuda() #
                self.avg_hand_qtars_th = torch.from_numpy(self.avg_hand_qtars).float().cuda() # 
                self.std_hand_qtars_th = torch.from_numpy(self.std_hand_qtars).float().cuda() #
                eps = 1e-6
                
                if not self.glb_rot_use_quat:
                    data_X = (data_X * (self.std_hand_qs_th.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qs_th.unsqueeze(0).unsqueeze(0)
                    data_E = (data_E * (self.std_hand_qtars_th.unsqueeze(0).unsqueeze(0) + eps)) + self.avg_hand_qtars_th.unsqueeze(0).unsqueeze(0)
                
                # data_X: bsz x nn_ts x feat_dim # 
                if data_nm is not None:
                    print(f"data_nm: {data_nm[0]}")
                    tot_batch_data_transl = []
                    for cur_data_nm in data_nm:
                        cur_first_frame_obj_transl = self.data_name_to_data[cur_data_nm]['first_frame_obj_trans']
                        cur_first_frame_obj_transl = torch.from_numpy(cur_first_frame_obj_transl).float().cuda() # get the first fram obj transl 
                        tot_batch_data_transl.append(cur_first_frame_obj_transl)
                    tot_batch_data_transl  = torch.stack(tot_batch_data_transl, dim=0) #### nn_bsz x 3
                    
                    # first_frame_obj_transl = self.data_name_to_data[data_nm]['first_frame_obj_trans']
                    # first_frame_obj_transl = torch.from_numpy(first_frame_obj_transl, dtype=torch.float32).cuda()
                    data_X[..., :3] = data_X[..., :3] + tot_batch_data_transl.unsqueeze(1)
                    data_E[..., :3] = data_E[..., :3] + tot_batch_data_transl.unsqueeze(1)
                
            
            if self.diff_task_space:
                
                data_X = data_X[:, 0]
                obj_type = data_X[:, 0:1] + 0.5
                data_X = torch.cat(
                    [obj_type, data_X[:, 1:]], dim=-1
                )
                data_E = data_X.clone()
                
                # avg_task_setting_th = torch.from_numpy(self.avg_task_setting).float().cuda()
                # std_task_setting_th = torch.from_numpy(self.std_task_setting).float().cuda()
                
                # data_X = data_X * (std_task_setting_th.unsqueeze(0) + eps) + avg_task_setting_th.unsqueeze(0)
                # data_E = data_X.clone()
                
            
            rt_dict = {
                'tot_verts': data_X,
                'tot_verts_integrated_qdd_tau': data_E,
                # 'tot_verts_integrated_qdd_tau_s2': data_E # inv scaled #
            }
        else:
            th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            

            
            th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            
            th_avg_particle_xs_integrated_taus_s2 = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            th_std_particle_xs_integrated_taus_s2=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
            
            
            inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            
            data_verts, data_verts_s2 = data_E[..., :3], data_E[..., 3:]
            inv_scaled_particle_xs_integrated_taus = (data_verts * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
            
            inv_scaled_particle_xs_integrated_taus_s2 = (data_verts_s2 * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2 ## get the inv_scaled integrated taus ##
            
            ###### ======= n-scale the data ======= ######
            # data_E_inv_scaled = data_E
            # data_X_inv_scaled = data_X
            ###### ======= n-scale the data ======= ######
            
            
            rt_dict = {
                'tot_verts': inv_scaled_particle_xs,
                'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
                'tot_verts_integrated_qdd_tau_s2': inv_scaled_particle_xs_integrated_taus_s2 # inv scaled #
            }
        
        if self.training_setting == 'trajectory_translations' and (not notarget):
            # inv_scaled_particle_xs_targe
            data_X_target = data_dict['X_target']
            data_E_target = data_dict['E_target']
            inv_scaled_data_X_target = (data_X_target * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            
            data_verts_target, data_verts_s2_target = data_E_target[..., :3], data_E_target[..., 3:]
            
            inv_scaled_particle_xs_integrated_taus_target = (data_verts_target * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus #
            inv_scaled_particle_xs_integrated_taus_s2_target = (data_verts_s2_target * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus 
            
            inv_scaled_data_target_dict = {
                'tot_verts_target': inv_scaled_data_X_target,
                'tot_verts_integrated_qdd_tau_target': inv_scaled_particle_xs_integrated_taus_target,
                'tot_verts_integrated_qdd_tau_s2_target': inv_scaled_particle_xs_integrated_taus_s2_target
            }
            rt_dict.update(inv_scaled_data_target_dict)
        elif self.training_setting == 'trajectory_translations_cond' and (not notarget):
            data_X_cond = data_dict['X_cond']
            data_E_cond = data_dict['E_cond']
            inv_scaled_data_X_cond = (data_X_cond * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
            data_verts_cond, data_verts_s2_cond = data_E_cond[..., :3], data_E_cond[..., 3:]
            
            inv_scaled_particle_xs_integrated_taus_cond = (data_verts_cond * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus
            inv_scaled_particle_xs_integrated_taus_s2_cond = (data_verts_s2_cond * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2
            
            inv_scaled_data_cond_dict = {
                'tot_verts_cond': inv_scaled_data_X_cond,
                'tot_verts_integrated_qdd_tau_cond': inv_scaled_particle_xs_integrated_taus_cond,
                'tot_verts_integrated_qdd_tau_s2_cond': inv_scaled_particle_xs_integrated_taus_s2_cond
            }
            rt_dict.update(inv_scaled_data_cond_dict)
        # elif self.use_jointspace_seq:
        #     data_X = 
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
            
        for key in data_dict:
            if key not in rt_dict:
                rt_dict[key] = data_dict[key]
        
        return rt_dict
    
    
    def scale_data_kine(self, data_dict, data_nm=None):
        
        # sv_dict = {
        #         'obj_verts': obj_verts,
        #         'robot_hand_qs': robot_hand_qs,
        #         'obj_rot_euler': obj_rot_euler_angles,
        #         'obj_transl': obj_transl
        #     }
        
        ## TODO: load kine data in the task conditioanl setting and scale the data here ##
        
        
        obj_verts = data_dict['obj_verts']
        robot_hand_qs = data_dict['robot_hand_qs']
        obj_rot_euler = data_dict['obj_rot_euler']
        obj_transl = data_dict['obj_transl'] 
        # object_type = data_dict['object_type']
        
        eps = 1e-6
        scaled_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        scaled_hand_qs = (robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        scaled_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        scaled_obj_transl = (obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
        
        concat_feat = np.concatenate(
            [scaled_hand_qs, scaled_obj_transl, scaled_obj_rot_euler ], axis=-1
        )
        
        if self.task_cond:
            cond_obj_verts = data_dict['cond_obj_verts']
            cond_robot_hand_qs = data_dict['cond_robot_hand_qs']
            cond_obj_rot_euler = data_dict['cond_obj_rot_euler']
            cond_obj_transl = data_dict['cond_obj_transl']
            # eps = 1e-6
            scaled_cond_obj_verts = (cond_obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
            # cond robot hand qs #
            scaled_cond_hand_qs = (cond_robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
            scaled_cond_obj_rot_euler = (cond_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
            scaled_cond_obj_transl = (cond_obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

            cond_concat_feat = np.concatenate(
                [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
            )

        
        # robot_hand_qs = data_dict['robot_hand_qs'][:, :self.nn_hands_dof] # ts x nn_qs # 
        # robot_hand_qs = robot_hand_qs[None] # 1 x ts x nn_qs # 
        # obj_verts = data_dict['obj_verts'] # ts x nn_obj_verts x 3 # 
        # obj_verts = obj_verts.transpose(1, 0, 2)[:, : robot_hand_qs.shape[1]] # nn_obj_verts x ts x 3 # 
        # # 
        # nn_pts = 512
        # rand_sampled_obj_verts_idxes = np.random.permutation(obj_verts.shape[0])[:nn_pts] # sampled idxes # 
        # obj_verts = obj_verts[rand_sampled_obj_verts_idxes] # 
        
        # avg_obj_verts_th = torc
        # eps = 1e-6
        # obj_verts = (obj_verts - self.avg_obj_verts[None][None]) / (self.std_obj_verts[None][None] + eps)
        
        rt_dict = {
            'X': scaled_obj_verts,
            'E': concat_feat[None],
        }
        if self.task_cond:
            rt_dict.update(
                {
                    'X_cond': scaled_cond_obj_verts,
                    'E_cond': cond_concat_feat[None]
                }
            )
        
        return rt_dict
    
    def _slice_tracking_kine_data(self, ):
        tot_hand_qs = []
        tot_obj_rot_eulers = []
        tot_obj_trans = []
        self.new_data_name_to_kine_info = {}
        for data_nm in self.data_name_to_data:
            # if self.slicing_data:
            pure_data_nm = data_nm.split('_STED_')[0] + ".npy"
            # else:
            #     pure_data_nm = data_nm # + ".npy"
            
            kine_traj_info = self.data_name_to_kine_info[pure_data_nm]
            
            # if self.slicing_data:
            print(f"[Slicing tracking kine data] data_nm: {data_nm}")
            sted_info = data_nm.split("/")[-1].split(".npy")[0].split('_STED_')[1]
            st_idx, ed_idx = sted_info.split("_")
            st_idx, ed_idx = int(st_idx), int(ed_idx)
            # else:
            #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0]
            
            if self.task_cond_type == 'history_future':
                slicing_idxes = self.data_name_to_data[data_nm]['slicing_idxes']
                hand_qs = kine_traj_info['hand_qs'][slicing_idxes][:, :self.nn_hands_dof]
                obj_trans = kine_traj_info['obj_trans'][slicing_idxes]
                obj_ornt = kine_traj_info['obj_ornt'][slicing_idxes]
                obj_rot_euler = kine_traj_info['obj_rot_euler'][slicing_idxes]
                obj_verts = kine_traj_info['obj_verts']
            else:
                hand_qs = kine_traj_info['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
                obj_trans = kine_traj_info['obj_trans'][st_idx: ed_idx]
                obj_ornt = kine_traj_info['obj_ornt'][st_idx: ed_idx]
                obj_rot_euler = kine_traj_info['obj_rot_euler'][st_idx: ed_idx]
                obj_verts = kine_traj_info['obj_verts']
            
            # if self.task_cond and self.task_cond_type == 'history_future':
            if self.task_cond_type == 'history_future':
                first_frame_obj_trans = self.data_name_to_data[data_nm]['first_frame_obj_trans'] # the first frametrans
            else:
                first_frame_obj_trans = obj_trans[0, :3]
            
            hand_qs[:, :3] = hand_qs[:, :3] - first_frame_obj_trans[None]
            obj_trans = obj_trans - first_frame_obj_trans[None]
            
            kine_info_dict = {
                'obj_verts': obj_verts, 
                'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
                'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
                'obj_ornt': obj_ornt ,
                'obj_rot_euler': obj_rot_euler
            }
            self.new_data_name_to_kine_info[data_nm] = kine_info_dict
            tot_hand_qs.append(hand_qs)
            tot_obj_rot_eulers.append(obj_rot_euler)
            tot_obj_trans.append(obj_trans)
        tot_obj_transl = np.concatenate(tot_obj_trans, axis=0)
        tot_obj_rot_eulers = np.concatenate(tot_obj_rot_eulers, axis=0)
        tot_hand_qs = np.concatenate(tot_hand_qs, axis=0) # 
        # tot_object_verts = np.concatenate(tot_object_verts, axis=0)
        
        self.avg_obj_transl = np.mean(tot_obj_transl, axis=0)
        self.std_obj_transl = np.std(tot_obj_transl, axis=0)
        self.avg_obj_rot_euler = np.mean(tot_obj_rot_eulers, axis=0)
        self.std_obj_rot_euler = np.std(tot_obj_rot_eulers, axis=0)
        # self.avg_hand_qs = np.mean(tot_hand_qs, axis=0)
        # self.std_hand_qs = np.std(tot_hand_qs, axis=0)
        
        self.avg_kine_hand_qs = np.mean(tot_hand_qs, axis=0)
        self.std_kine_hand_qs = np.std(tot_hand_qs, axis=0)
        
        # avg obj verts and the kine hand qs and #
        # self.data_statistics['avg_obj_verts'] = self.avg_object_verts
        # self.data_statistics['std_obj_verts'] = self.std_object_verts
        self.data_statistics['avg_kine_hand_qs'] = self.avg_kine_hand_qs
        self.data_statistics['std_kine_hand_qs'] = self.std_kine_hand_qs
        self.data_statistics['avg_obj_transl'] = self.avg_obj_transl
        self.data_statistics['std_obj_transl'] = self.std_obj_transl
        self.data_statistics['avg_obj_rot_euler'] = self.avg_obj_rot_euler
        self.data_statistics['std_obj_rot_euler'] = self.std_obj_rot_euler
        
        
        # 
        # self.avg_object_verts = np.mean(tot_object_verts, axis=0)
        # self.std_object_verts = np.std(tot_object_verts, axis=0) # the std objectverts #
        self.data_name_to_kine_info = self.new_data_name_to_kine_info
        
        
    
    def scale_data_tracking_ctl(self, data_dict, data_nm):
        # print(f"data_nm: {data_nm}, data_dict: {data_dict.keys()}")
        # print(f"[Scale data tracking ctl] data_nm: {data_nm} glb_rot_use_quat: {self.glb_rot_use_quat}")
        particle_xs = data_dict['tot_verts']
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        
        ## NOTE: currently they are all qs and qtars ##
        eps = 1e-6
        
        if not self.glb_rot_use_quat:
            particle_xs = (particle_xs - self.avg_hand_qs[None][None]) / (self.std_hand_qs[None][None] + eps)
            particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_hand_qtars[None][None]) / (self.std_hand_qtars[None][None] + eps)
        
        # self.data_name_to_kine_info[data_inst_tag] = {
        #     'obj_verts': obj_verts, 
        #     'hand_qs': hand_qs, # if the task cond ws x nn_pts x 3 --- as the objet pts input #
        #     'obj_trans': obj_trans, # obj verts; obj trans; # --- ws x 3; ws x 2; ws x 22; with the object verts #
        #     'obj_ornt': obj_ornt 
        # }
        
        assert particle_xs.shape[1] == self.slicing_ws
        
        if particle_xs.shape[1] < self.slicing_ws:
            padding_particle_xs = np.concatenate(
                [ particle_xs[:, -1:] for _ in range(self.slicing_ws - particle_xs.shape[1]) ], axis=1
            )
            particle_xs = np.concatenate(
                [ particle_xs, padding_particle_xs ], axis=1
            )
            
            padding_particle_xs_integrated_qdd_tau = np.concatenate(
                [ particle_xs_integrated_qdd_tau[:, -1:] for _ in range(self.slicing_ws - particle_xs_integrated_qdd_tau.shape[1]) ], axis=1
            )
            particle_xs_integrated_qdd_tau = np.concatenate(
                [ particle_xs_integrated_qdd_tau, padding_particle_xs_integrated_qdd_tau ], axis=1
            )

        
        
        rt_dict = {
            'X': particle_xs,
            'E': particle_xs_integrated_qdd_tau
        }
        
        if self.task_cond:
            
            # if '_STED_' in if 
            # if self.slicing_data:
            #     pure_data_nm = data_nm.split('_STED_')[0] + ".npy"
            # else:
            #     pure_data_nm = data_nm # + ".npy"
            pure_data_nm = data_nm
            
            kine_traj_info = self.data_name_to_kine_info[pure_data_nm]
            
            # if self.slicing_data:
            #     sted_info = data_nm.split("/")[-1].split(".")[0].split('_STED_')[1]
            #     st_idx, ed_idx = sted_info.split("_")
            #     st_idx, ed_idx = int(st_idx), int(ed_idx)
            # else:
            #     st_idx = 0
            #     ed_idx = kine_traj_info['hand_qs'].shape[0]
            st_idx = 0
            ed_idx = kine_traj_info['hand_qs'].shape[0]
            
            hand_qs = kine_traj_info['hand_qs'][st_idx: ed_idx][:, :self.nn_hands_dof]
            obj_trans = kine_traj_info['obj_trans'][st_idx: ed_idx]
            obj_ornt = kine_traj_info['obj_ornt'][st_idx: ed_idx]
            obj_rot_euler = kine_traj_info['obj_rot_euler'][st_idx: ed_idx]
            obj_verts = kine_traj_info['obj_verts']
            
            
            # first_frame_obj_trans = obj_trans[0, :3]

            ## TODO: eulers may not be a good representation ##
            
            scaled_cond_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
            # cond robot hand qs #
            
            if not self.glb_rot_use_quat:
                # scaled_cond_hand_qs = (hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
                scaled_cond_hand_qs = (hand_qs - self.avg_kine_hand_qs[None]) / (self.std_kine_hand_qs[None] + eps)
                scaled_cond_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
                scaled_cond_obj_transl = (obj_trans - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

            # E cond --- the conditional future data #  # obj trans #
            cond_concat_feat = np.concatenate(
                [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
            )
            
            assert cond_concat_feat.shape[0] == self.slicing_ws
            
            # cond concat feat --- for the concat feat #
            # cond input # 
            if cond_concat_feat.shape[0] < self.slicing_ws:
                if cond_concat_feat.shape[0] > 0:
                    padding_concat_feat = np.concatenate(
                        [ cond_concat_feat[-1:] for _ in range(self.slicing_ws - cond_concat_feat.shape[0]) ], axis=0
                    )
                    cond_concat_feat = np.concatenate(
                        [cond_concat_feat, padding_concat_feat], axis=0
                    )
                else:
                    cond_concat_feat = np.zeros((self.slicing_ws, cond_concat_feat.shape[-1]), dtype=np.float32)
                    
            
            
            rt_dict.update(
                {
                    'X_cond': scaled_cond_obj_verts,
                    'E_cond': cond_concat_feat[None]
                }
            )
            
            if self.task_cond_type == 'history_future':
                tracking_ctl_info_dict = self.data_name_to_data[data_nm] # 
                # history_info = {
                #     'history_obj_pose': history_obj_pose[None ], # 
                #     'history_kine_qs': history_kine_qs[None ], #  obj pose and the kinematrics qs #
                #     'first_frame_obj_trans': first_frame_obj_trans,
                #     'first_frame_obj_ornt': first_frame_obj_ornt # first frae obj ornt # an the trans # 
                # }
                history_obj_pose = tracking_ctl_info_dict['history_obj_pose'] # history obj pose -- 1 x ws x nn_obj_dim 
                history_kine_qs = tracking_ctl_info_dict['history_kine_qs'][0] # history kine qs -- 1 x ws x nn_hand_qs 
                first_frame_obj_trans = tracking_ctl_info_dict['first_frame_obj_trans']
                # 'history_obj_trans': history_obj_trans[None ], 
                        # 'history_obj_rot_euler': history_obj_rot_euler[None ],
                history_obj_rot_euler = tracking_ctl_info_dict['history_obj_rot_euler'][0]
                history_obj_trans = tracking_ctl_info_dict['history_obj_trans'][0]
                
                
                if not self.glb_rot_use_quat:
                    scaled_history_kine_qs = (history_kine_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
                    scaled_history_obj_rot_euler = (history_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
                    scaled_history_obj_trans = (history_obj_trans - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
                
                
                # # nn_history_ws x 3 # # 
                # # nn_history_ws x 3 # # 
                # # nn_history_ws x 22 # # 
                # history_cond_contact_feat = np.concatenate(
                #     [history_kine_qs, history_obj_trans, history_obj_rot_euler], axis=-1 # history cond features # 
                # )
                history_cond_contact_feat = np.concatenate(
                    [scaled_history_kine_qs, scaled_history_obj_trans, scaled_history_obj_rot_euler], axis=-1
                )
                # history cond contact feat #
                
                assert history_cond_contact_feat.shape[0] == self.history_ws #  self.slicing_ws
                # print(f"history_cond_contact_feat: {history_cond_contact_feat.shape}")
                
                if history_cond_contact_feat.shape[0] < self.slicing_ws:
                    if history_cond_contact_feat.shape[0] > 0:
                        padding_history_E_cond_feat = np.zeros_like(history_cond_contact_feat[0:1])
                        padding_history_E_cond_feat = np.concatenate(
                            [ padding_history_E_cond_feat for _ in range(self.slicing_ws - history_cond_contact_feat.shape[0]) ], axis=0
                        )
                        history_cond_contact_feat = np.concatenate(
                            [ padding_history_E_cond_feat, history_cond_contact_feat ], axis=0
                        )
                    else:
                        history_cond_contact_feat = np.zeros((self.slicing_ws, history_cond_contact_feat.shape[-1]), dtype=np.float32)
                    
                    
                    
                # print(f"[After padding] history_cond_contact_feat: {history_cond_contact_feat.shape}")
                # if history_cond_contact_feat.
                history_cond_dict = {
                    'history_E_cond': history_cond_contact_feat[None]
                }
                rt_dict.update(history_cond_dict)
                
                pass
            
        return rt_dict
                
    
    def scale_data(self, data_dict, data_nm):
        
        if self.kine_diff:
            rt_dict = self.scale_data_kine(data_dict, data_nm)
            return rt_dict
        elif self.tracking_ctl_diff:
            rt_dict = self.scale_data_tracking_ctl(data_dict, data_nm)
            return rt_dict

        
        ## nn_ts x nn_particles x 3 ## ## get scaled data ##
        
        if self.training_setting in ['trajectory_translations', 'trajectory_translations_cond']:
            unopt_xs = data_dict['unopt_verts']
            unopt_tar_xs = data_dict['unopt_qtar_verts']
            opt_xs = data_dict['opt_verts']
            opt_tar_xs = data_dict['opt_qtar_verts']
            
            eps = 1e-6
            
            # unopt_xs = (unopt_xs - self.)
            unopt_xs = (unopt_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            unopt_tar_xs = (unopt_tar_xs - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
            
            opt_xs = (opt_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            opt_tar_xs = (opt_tar_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
            
            sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        
            unopt_xs = unopt_xs[sampled_particle_idxes, :, :]
            unopt_tar_xs = unopt_tar_xs[sampled_particle_idxes, :, :]
            opt_xs = opt_xs[sampled_particle_idxes, :, :]
            opt_tar_xs = opt_tar_xs[sampled_particle_idxes, :, :]
            
            unopt_E = np.concatenate(
                [unopt_tar_xs, unopt_tar_xs], axis=-1
            )    
            opt_E = np.concatenate(
                [opt_tar_xs, opt_tar_xs], axis=-1
            )
            if self.training_setting == 'trajectory_translations_cond':
                rt_dict = {
                    'X_cond': unopt_xs,
                    'E_cond': unopt_E,
                    'X': opt_xs,
                    'E': opt_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
            else:
                rt_dict = {
                    'X': unopt_xs,
                    'E': unopt_E,
                    'X_target': opt_xs,
                    'E_target': opt_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
        
        else:
            if self.use_jointspace_seq:
                
                if self.diff_task_translations:
                    fa_task_setting = data_dict['fa_task_setting']
                    task_setting = np.array(fa_task_setting, dtype=np.float32)
                    
                    ch_task_setting = data_dict['ch_task_setting']
                    ch_task_setting = [ch_task_setting[0] - 0.5, ch_task_setting[1], ch_task_setting[2]]
                    ch_task_setting = np.array(ch_task_setting, dtype=np.float32)
                    particle_xs = ch_task_setting[None][None ]
                    particle_xs_integrated_qdd_tau = particle_xs
                else:
                    
                    particle_xs = data_dict['tot_verts']
                    particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
                    
                    ## set task setting # obj_task_setting #
                    
                    # task_setting = {
                    #     'object_type': cur_data['object_type'],
                    #     'task_rot': cur_data['task_rot'],
                    #     'object_size_x': cur_data['object_size_x']
                    # }
                    #### [object_type, task_rot, object_size_x] ####
                    task_setting = [
                        data_dict['task_setting']['object_type'], data_dict['task_setting']['task_rot'], data_dict['task_setting']['object_size_x']
                    ]
                    task_setting = np.array(task_setting, dtype=np.float32)
                    
                    
                    ### 1) make it into the particle xs and also E ###
                    
                    if self.diff_task_space:
                        eps = 1e-6
                        task_setting_2 = [
                            float(data_dict['task_setting']['object_type']), data_dict['task_setting']['task_rot'], data_dict['task_setting']['object_size_x']
                        ]
                        task_setting_2 = np.array(task_setting_2, dtype=np.float32)
                        
                        task_setting_2 = (task_setting_2 - self.avg_task_setting) / (self.std_task_setting + eps)
                        
                        particle_xs = task_setting_2[None][None]
                        particle_xs_integrated_qdd_tau = task_setting_2[None][None]
                
                
                rt_dict = {
                    'X': particle_xs,
                    'E': particle_xs_integrated_qdd_tau,
                    'obj_task_setting': task_setting #### [object_type, task_rot, object_size_x] ####
                }
            
            else:
                eps = 1e-6
                particle_xs = data_dict['tot_verts']
                particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
                particle_xs_integrated_qdd_tau_s2 = data_dict['tot_verts_integrated_qdd_tau_s2']
                
                
                particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
                particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
                particle_xs_integrated_qdd_tau_s2 = (particle_xs_integrated_qdd_tau_s2 - self.avg_verts_qdd_tau_tot_cases_tot_ts_s2[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts_s2[None][None] + eps)
                # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
                sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
                
                particle_xs = particle_xs[sampled_particle_idxes, :, :]
                particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
                particle_xs_integrated_qdd_tau_s2 = particle_xs_integrated_qdd_tau_s2[sampled_particle_idxes, :, :]
                
                data_E = np.concatenate([particle_xs_integrated_qdd_tau, particle_xs_integrated_qdd_tau_s2], axis=-1)
            
                rt_dict = {
                    'X': particle_xs,
                    'E': data_E,
                    'sampled_idxes': sampled_particle_idxes,
                }
        
        ### return the dict ###
        ### return the dict ###
        return rt_dict
    
    
    def scale_data_kine_new(self, data_dict, data_nm=None):
        
        # data_dict_rt = {
        #     'obj_pts': cur_obj_verts, 
        #     'hand_pose': history_hand_qs,
        #     'obj_pos': history_obj_transl,
        #     'obj_ornt': history_obj_ornt,
        #     'future_hand_pose': future_hand_qs,
        #     'future_obj_pos': future_obj_transl,
        #     'future_obj_ornt': future_obj_ornt,
        #     'last_frame_obj_pos': history_obj_transl[-1:, ...]
        # }
        
        # # data dict rt #
        
        # if self.w_glb_traj_feat_cond:
        #     glb_hand_qs = cur_hand_qs[::10]
        #     glb_obj_transl = cur_obj_transl[::10]
        #     glb_obj_rot_ornt = cur_obj_rot_quat[::10]
            
        #     # print(f"[Debug] glb_hand_qs: {glb_hand_qs.shape}, glb_obj_transl: {glb_obj_transl.shape}, glb_obj_rot_ornt: {glb_obj_rot_ornt.shape}")
        #     data_dict_rt.update(
        #         {
        #             'glb_hand_pose': glb_hand_qs,
        #             'glb_obj_pos': glb_obj_transl,
        #             'glb_obj_ornt': glb_obj_rot_ornt
        #         }
        #     )
            
        eps = 1e-6
        obj_pts = data_dict['obj_pts'] ## nn_pts x 3 ## 
        canonicalized_obj_pts = (obj_pts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        
        rt_dict = { }
        rt_dict.update(data_dict)
         
        rt_dict.update(
            {
                'obj_pts': canonicalized_obj_pts
            }
        )
        # obj_pts = data_dict['obj_pts']
        # hand_pose = data_dict['hand_pose'] # history qs #
        # obj_pos = data_dict['obj_pos']
        # obj_ornt = data_dict['obj_ornt']
        # future_hand_pose = data_dict['future_hand_pose']
        # future_obj_pos = data_dict['future_obj_pos']
        # future_obj_ornt = data
        
        
        # obj_verts = data_dict['obj_verts']
        # robot_hand_qs = data_dict['robot_hand_qs']
        # obj_rot_euler = data_dict['obj_rot_euler']
        # obj_transl = data_dict['obj_transl'] 
        # # object_type = data_dict['object_type']
        
        # eps = 1e-6
        # scaled_obj_verts = (obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        # scaled_hand_qs = (robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        # scaled_obj_rot_euler = (obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        # scaled_obj_transl = (obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)
        
        # concat_feat = np.concatenate(
        #     [scaled_hand_qs, scaled_obj_transl, scaled_obj_rot_euler ], axis=-1
        # )
        
        # if self.task_cond:
        #     cond_obj_verts = data_dict['cond_obj_verts']
        #     cond_robot_hand_qs = data_dict['cond_robot_hand_qs']
        #     cond_obj_rot_euler = data_dict['cond_obj_rot_euler']
        #     cond_obj_transl = data_dict['cond_obj_transl']
        #     # eps = 1e-6
        #     scaled_cond_obj_verts = (cond_obj_verts - self.avg_object_verts[None]) / (self.std_object_verts[None] + eps)
        #     # cond robot hand qs #
        #     scaled_cond_hand_qs = (cond_robot_hand_qs - self.avg_hand_qs[None]) / (self.std_hand_qs[None] + eps)
        #     scaled_cond_obj_rot_euler = (cond_obj_rot_euler - self.avg_obj_rot_euler[None]) / (self.std_obj_rot_euler[None] + eps)
        #     scaled_cond_obj_transl = (cond_obj_transl - self.avg_obj_transl[None]) / (self.std_obj_transl[None] + eps)

        #     cond_concat_feat = np.concatenate(
        #         [scaled_cond_hand_qs, scaled_cond_obj_transl, scaled_cond_obj_rot_euler ], axis=-1
        #     )

        
        # rt_dict = {
        #     'X': scaled_obj_verts,
        #     'E': concat_feat[None],
        # }
        # if self.task_cond:
        #     rt_dict.update(
        #         {
        #             'X_cond': scaled_cond_obj_verts,
        #             'E_cond': cond_concat_feat[None]
        #         }
        #     )
        
        # return rt_dict
    
        return rt_dict
    
    
    #### data_dict_to_th ####
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if isinstance(data_dict_np[key], str):
                data_dict_th[key] = data_dict_np[key]
            elif key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
        
        
        return data_dict_th
    
    
    
    
    def __getitem__(self, index):
        
        # cur data info #
        cur_data_info = self.data_list[index]
        data_nm = cur_data_info # [ 'data_nm']
        cur_data_nm = data_nm
        # history_window_info = cur_data_info['history_window_idxes']
        # future_window_info = cur_data_info['future_window_idxes'] # history #
        
        # factorized_history_window_info = history_window_info.astype(np.float32) / float(self.num_frames)
        # use the frame 1000 to rescale the window # # number of saequneces # 
        # actually it is the # # factorized history widnow info #
        # self.rescaled_num_frames = 1000
        # factorized_history_window_info = (factorized_history_window_info * self.rescaled_num_frames).astype(np.int32) # get the rescaled num frames #
        
        
        # get the obj tracking #
        # get the obj tracking # structural latnet space --- #
        
        
        # history window info #
        cur_data_dict = self.data_name_to_data[data_nm]
        cur_obj_transl = cur_data_dict['obj_transl'][::5]
        cur_obj_rot_euler = cur_data_dict['obj_rot_euler'][::5]
        cur_obj_verts = cur_data_dict['obj_verts']
        cur_hand_qs = cur_data_dict['robot_hand_qs'][::5]
        cur_obj_rot_quat = cur_data_dict['obj_rot_quat'][::5]
        
        # # history window info #
        # history_obj_transl = cur_obj_transl[history_window_info]
        # history_obj_ornt = cur_obj_rot_quat[history_window_info]
        # history_hand_qs = cur_hand_qs[history_window_info]
        # future_obj_transl = cur_obj_transl[future_window_info]
        # future_obj_ornt = cur_obj_rot_quat[future_window_info]
        # future_hand_qs = cur_hand_qs[future_window_info]
        
        # if self.canonicalize_features:
        #     cur_last_frame_obj_transl = history_obj_transl[-1:, ...]
        #     history_obj_transl = history_obj_transl - cur_last_frame_obj_transl
        #     history_hand_qs[..., :3] = history_hand_qs[..., :3] - cur_last_frame_obj_transl
        #     future_obj_transl = future_obj_transl - cur_last_frame_obj_transl
        #     future_hand_qs[..., :3] = future_hand_qs[..., :3] - cur_last_frame_obj_transl
        
        data_dict_rt = {
            'obj_pts': cur_obj_verts, 
            'hand_pose': cur_hand_qs,
            'obj_pos': cur_obj_transl,
            'obj_ornt': cur_obj_rot_quat,
            # 'future_hand_pose': future_hand_qs,
            # 'future_obj_pos': future_obj_transl,
            # 'future_obj_ornt': future_obj_ornt,
            # 'last_frame_obj_pos': history_obj_transl[-1:, ...],
            # 'factorized_history_window_info': factorized_history_window_info # the factorized history info #
        }
        
        
        
        # if self.w_glb_traj_feat_cond:
        #     glb_hand_qs = cur_hand_qs[::10]
        #     glb_obj_transl = cur_obj_transl[::10]
        #     glb_obj_rot_ornt = cur_obj_rot_quat[::10]
            
        #     # print(f"[Debug] glb_hand_qs: {glb_hand_qs.shape}, glb_obj_transl: {glb_obj_transl.shape}, glb_obj_rot_ornt: {glb_obj_rot_ornt.shape}")
        #     data_dict_rt.update(
        #         {
        #             'glb_hand_pose': glb_hand_qs,
        #             'glb_obj_pos': glb_obj_transl,
        #             'glb_obj_ornt': glb_obj_rot_ornt
        #         }
        #     )

        
        # history_obj_rot_euler = history_window_info['history_obj_rot_euler']
        
        # if self.kine_diff and self.task_cond:
        #     cur_data = self.tot_data_dict_list[index] 
        #     cur_data_nm = index
        # else:
        #     cur_data_nm = self.data_name_list[index]
        #     if self.sampling and len(self.target_grab_inst_tag) > 0 and self.target_grab_data_nm is not None: 
        #         if self.slicing_data: ### slicing data ###
        #             cur_data_nm = self.tot_target_data_nm[index % len(self.tot_target_data_nm)]
        #         else:
        #             cur_data_nm = self.target_grab_data_nm
        #     if cur_data_nm not in self.data_name_to_data:
        #         cur_data = self._load_data_from_data_name(cur_data_nm)
        #         self.data_name_to_data[cur_data_nm] = cur_data
        #         # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        #     else:
        #         cur_data = self.data_name_to_data[cur_data_nm]
        
        
        # ## TODO: data selecting, data parsing, and data scaling
        # # if self.use_target_data:
        # #     cur_data_scaled = self.get_target_data() # 
        # # else:
        
        
        # # print(f"cur_data_nm: {cur_data_nm}, cur_data: {cur_data.keys()}")
        
        # if self.sampling and self.use_shadow_test_data:
        #     cur_data_scaled = self.target_data_dict
        # else:
        #     cur_data_scaled = self.scale_data(cur_data, cur_data_nm)
        
        # # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # # self.data_name_to_statistics[cur_data_nm] = { ## data scaled ##
        # #     'std': cur_data_std,
        # #     'avg': cur_data_avg
        # # }
        
        # if self.canonicalize_obj_pts:
        #     data_dict_rt = self.scale_data_kine_new(data_dict_rt)
        
        
        cur_data_scaled_th = self.data_dict_to_th(data_dict_rt)
        
        # for cur_key in cur_data_scaled_th:
        #     cur_data_th = cur_data_scaled_th[cur_key]
        #     print(f"cur_key: {cur_key}, cur_data_th: {cur_data_th.size()}")
        
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ") 
        cur_data_scaled_th.update(
            {'data_nm': cur_data_nm}
        )
        return cur_data_scaled_th




def collect_fn_pc_v7_dataset(batch):
    rt_val = {}
    first_batch = batch[0]
    for key in first_batch:
        if key == 'data_nm':
            rt_val[key] = [sample[key] for sample in batch]
        elif key == 'object_type':
            rt_val[key] = [sample[key] for sample in batch]
        else:
            try:
                rt_val[key] = torch.stack([sample[key] for sample in batch], dim=0)
            except:
                rt_val[key] = torch.stack([batch[0][key] for sample in batch], dim=0)
    return rt_val






class Uni_Manip_3D_PC_V7_Cond_Dataset(torch.utils.data.Dataset):
    def bak__init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt # 
        self.nn_timesteps = cfg.task.nn_timesteps
        # #
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst
        self.specified_hand_type = cfg.dataset_3d_pc.specified_hand_type 
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages
        self.use_static_first_frame = cfg.dataset_3d_pc.use_static_first_frame
        self.use_shadow_test_data = cfg.sampling.use_shadow_test_data
        self.sampling = cfg.sampling.sampling
        
        self.debug = cfg.debug
        
        # self.use_allegro_test_data = cfg.sampling.use_allegro_test_data
        self.specified_test_subfolder = cfg.sampling.specified_test_subfolder
        
        exp_tags = ["tds_exp_2"]

        ## root_data_folder ##
        self.data_list = []
        self.valid_subfolders = []
        
        # if self.all_clips_inst: # get the all clips instances # #
        self.data_inst_fn = f"save_info_v6.npy"
        print(f"data_inst_fn changed to {self.data_inst_fn} with all_clips_inst: {self.all_clips_inst}")
        
        ### exp tag ###
        for exp_tag in exp_tags: ## 
            cur_data_folder = os.path.join(self.data_folder, exp_tag)
            tmp_data_list = os.listdir(cur_data_folder)
            
            print(f"specified_test_subfolder: {self.specified_test_subfolder}, full_specified_test_subfolder: {os.path.join(cur_data_folder, self.specified_test_subfolder)}")
            
            if self.specified_test_subfolder is not None and len(self.specified_test_subfolder) > 0 and os.path.exists(os.path.join(cur_data_folder, self.specified_test_subfolder)):
                print(f"[here] specified_test_subfolder: {self.specified_test_subfolder}")
                tmp_data_list = [self.specified_test_subfolder]
            
            for cur_subfolder in tmp_data_list: # getting data ## specified test fn ##
                
                if self.specified_hand_type is not None:
                    if self.specified_hand_type == 'allegro_flat_fivefin_yscaled_finscaled':
                        if self.specified_hand_type not in cur_subfolder:
                            continue
                    elif self.specified_hand_type == 'allegro_flat_fivefin_yscaled':
                        if self.specified_hand_type not in cur_subfolder or 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder:
                            continue
                    elif self.specified_hand_type == 'allegro_flat_fivefin':
                        if self.specified_hand_type not in cur_subfolder or 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder or 'allegro_flat_fivefin_yscaled' in cur_subfolder:
                            continue
                    elif self.specified_hand_type == 'allegro':
                        if 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder or 'allegro_flat_fivefin_yscaled' in cur_subfolder or 'allegro_flat_fivefin' in cur_subfolder:
                            continue
                    else:
                        raise ValueError(f"Unrecognized specified_hand_type: {self.specified_hand_type}")
                
                # cur_subfolder_angle = cur_subfolder.split("_")
                ####### ====== for positiive angles ====== #######
                # for i_split in range(len(cur_subfolder_angle)):
                #     if cur_subfolder_angle[i_split] == "objgoalrot":
                #         cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                #         break
                # if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                #     continue
                ####### ====== for positiive angles ====== #######
                
                
                inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                if os.path.isdir(inst_folder):
                    ####### ======= get the instance statistics info fn ====== ######
                    save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
                    save_cur_inst_statistics_info_fn = os.path.join(inst_folder, save_cur_inst_statistics_info_fn)

                    if not os.path.exists(save_cur_inst_statistics_info_fn):
                        continue
                    
                    cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                    if os.path.exists(cur_inst_file):
                        
                        self.data_list.append(cur_inst_file)
                        self.valid_subfolders.append(cur_subfolder)
        
        # if self.all_clips_inst:
        #     valid_data_list_sv_fn = f"valid_data_statistics_v4.npy" 
        # elif self.single_inst: 
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3.npy" 
        # else:
        #     # valid_data_list_sv_fn = f"valid_data_statistics_v3_all.npy" ## solve the task better? ##
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3_positive_angles.npy"
        # # valid_data_list_sv_fn = f"valid_data_statistics_v3.npy"
        
        ####### ======= get the single inst ======= #######
        if self.single_inst: # save info v6 statistics #
            valid_data_list_sv_fn = "save_info_v6_statistics.npy"
            valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], self.valid_subfolders[0], valid_data_list_sv_fn)
        else:
            if self.specified_hand_type == "allegro":
                valid_data_list_sv_fn = "save_info_v6_statistics_allegro.npy"
                valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
            else:
                valid_data_list_sv_fn = "save_info_v6_statistics.npy"
                # valid_data_list_sv_fn = "/cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy"
                valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
        valid_data_list_sv_fn = "/cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy"
        print(f"valid_data_list_sv_fn: {valid_data_list_sv_fn}")
        valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        
        self.avg_verts_tot_cases_tot_ts = valid_data_statistics['avg_verts']
        self.std_verts_tot_cases_tot_ts = valid_data_statistics['std_verts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['avg_qtar_verts']
        self.std_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['std_qtar_verts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts_s2 = valid_data_statistics['avg_qtar_verts_s2']
        self.std_verts_qdd_tau_tot_cases_tot_ts_s2 = valid_data_statistics['std_qtar_verts_s2']
        
        
        
        self.maxx_nn_pts = 512
        
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        
        self._preload_data()
        self.data_name_to_statistics = {}
    
    
    
    def __init__(self, data_folder, cfg):
        self.data_folder = data_folder
        self.cfg = cfg
        
        
        
        #### dt and nn_timesteps ###
        self.dt = cfg.task.dt # 
        self.nn_timesteps = cfg.task.nn_timesteps
        
        
        self.use_target_data = cfg.task.use_target_data
        self.sample_wconstraints = cfg.task.sample_wconstraints
        
        self.single_inst = cfg.dataset_3d_pc.single_inst
        ### whether to test the all_clips_inst ###
        self.all_clips_inst = cfg.dataset_3d_pc.all_clips_inst
        self.specified_hand_type = cfg.dataset_3d_pc.specified_hand_type 
        
        self.specified_object_type = cfg.dataset_3d_pc.specified_object_type
        
        self.sampled_particle_idxes = None
        
        self.nn_stages = cfg.dataset_3d_pc.nn_stages
        self.use_static_first_frame = cfg.dataset_3d_pc.use_static_first_frame
        self.use_shadow_test_data = cfg.sampling.use_shadow_test_data
        self.sampling = cfg.sampling.sampling
        
        # self.use_allegro_test_data = cfg.sampling.use_allegro_test_data
        self.specified_test_subfolder = cfg.sampling.specified_test_subfolder
        self.specified_statistics_info_fn = cfg.training.specified_statistics_info_fn
        self.specified_sampled_particle_idxes_fn = cfg.training.specified_sampled_particle_idxes_fn
        
        
        ## get the task training settings  ###
        self.training_setting = cfg.training.setting ## training setting ## 
        
        
        self.debug = cfg.debug
        
        # self.specified_object_type = 
        
        exp_tags = ["tds_exp_2"]

        ## root_data_folder ##
        self.data_list = []
        self.valid_subfolders = []
        
        # if self.all_clips_inst:
        self.data_inst_fn = f"save_info_v6.npy"
        print(f"data_inst_fn changed to {self.data_inst_fn} with all_clips_inst: {self.all_clips_inst}")
        
        ### exp tag ###
        for exp_tag in exp_tags:
            cur_data_folder = os.path.join(self.data_folder, exp_tag)
            tmp_data_list = os.listdir(cur_data_folder)
            
            print(f"specified_test_subfolder: {self.specified_test_subfolder}, full_specified_test_subfolder: {os.path.join(cur_data_folder, self.specified_test_subfolder)}")
            
            if self.specified_test_subfolder is not None and len(self.specified_test_subfolder) > 0 and os.path.exists(os.path.join(cur_data_folder, self.specified_test_subfolder)):
                print(f"[here] specified_test_subfolder: {self.specified_test_subfolder}")
                tmp_data_list = [self.specified_test_subfolder]
            
            for cur_subfolder in tmp_data_list: # getting data ## specified test fn ##
                
                if self.specified_hand_type is not None:
                    if self.specified_hand_type == 'allegro_flat_fivefin_yscaled_finscaled':
                        if self.specified_hand_type not in cur_subfolder:
                            continue
                    elif self.specified_hand_type == 'allegro_flat_fivefin_yscaled':
                        if self.specified_hand_type not in cur_subfolder or 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder:
                            continue
                    elif self.specified_hand_type == 'allegro_flat_fivefin':
                        if self.specified_hand_type not in cur_subfolder or 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder or 'allegro_flat_fivefin_yscaled' in cur_subfolder:
                            continue
                    elif self.specified_hand_type == 'allegro':
                        if 'allegro_flat_fivefin_yscaled_finscaled' in cur_subfolder or 'allegro_flat_fivefin_yscaled' in cur_subfolder or 'allegro_flat_fivefin' in cur_subfolder:
                            continue
                    else:
                        raise ValueError(f"Unrecognized specified_hand_type: {self.specified_hand_type}")
                
                if self.specified_object_type is not None:
                    if f"objtype_{self.specified_object_type}" not in cur_subfolder:
                        continue
                
                # cur_subfolder_angle = cur_subfolder.split("_")
                ####### ====== for positiive angles ====== #######
                # for i_split in range(len(cur_subfolder_angle)):
                #     if cur_subfolder_angle[i_split] == "objgoalrot":
                #         cur_subfolder_angle = float(cur_subfolder_angle[i_split + 1])
                #         break
                # if isinstance(cur_subfolder_angle, list) or cur_subfolder_angle <= 0.0:
                #     continue
                ####### ====== for positiive angles ====== #######
                
                
                inst_folder = os.path.join(cur_data_folder, cur_subfolder)
                if os.path.isdir(inst_folder):
                    ####### ======= get the instance statistics info fn ====== ######
                    save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
                    save_cur_inst_statistics_info_fn = os.path.join(inst_folder, save_cur_inst_statistics_info_fn)

                    if not os.path.exists(save_cur_inst_statistics_info_fn):
                        continue
                    
                    cur_inst_file = os.path.join(inst_folder, self.data_inst_fn)
                    if os.path.exists(cur_inst_file):
                        
                        self.data_list.append(cur_inst_file)
                        self.valid_subfolders.append(cur_subfolder)
        
        # if self.all_clips_inst:
        #     valid_data_list_sv_fn = f"valid_data_statistics_v4.npy" 
        # elif self.single_inst: 
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3.npy" 
        # else:
        #     # valid_data_list_sv_fn = f"valid_data_statistics_v3_all.npy" ## solve the task better? ##
        #     valid_data_list_sv_fn = f"valid_data_statistics_v3_positive_angles.npy"
        # # valid_data_list_sv_fn = f"valid_data_statistics_v3.npy"
        # the 
        if (not self.single_inst) and self.specified_statistics_info_fn is not None and len(self.specified_statistics_info_fn) > 0 and os.path.exists(self.specified_statistics_info_fn):
            valid_data_list_sv_fn = self.specified_statistics_info_fn
        else:
            ####### ======= get the single inst ======= #######
            if self.single_inst: # save info v6 statistics #
                valid_data_list_sv_fn = "save_info_v6_statistics.npy"
                valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], self.valid_subfolders[0], valid_data_list_sv_fn)
            else:
                valid_data_list_sv_fn = "save_info_v6_statistics.npy"
                # valid_data_list_sv_fn = "/cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy"
                valid_data_list_sv_fn = os.path.join(self.data_folder, exp_tags[0], valid_data_list_sv_fn)
            
            ####### ====== only use the allegro statistics ====== #######
            valid_data_list_sv_fn = "/cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy"
            ####### ====== only use the allegro statistics ====== #######
        
        print(f"valid_data_list_sv_fn: {valid_data_list_sv_fn}")
        valid_data_statistics = np.load(valid_data_list_sv_fn, allow_pickle=True).item()
        
        
        self.avg_verts_tot_cases_tot_ts = valid_data_statistics['avg_verts']
        self.std_verts_tot_cases_tot_ts = valid_data_statistics['std_verts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['avg_qtar_verts']
        self.std_verts_qdd_tau_tot_cases_tot_ts = valid_data_statistics['std_qtar_verts']
        self.avg_verts_qdd_tau_tot_cases_tot_ts_s2 = valid_data_statistics['avg_qtar_verts_s2']
        self.std_verts_qdd_tau_tot_cases_tot_ts_s2 = valid_data_statistics['std_qtar_verts_s2']
        
        
        
        self.maxx_nn_pts = 512
        
        # self.maxx nn pts ##
        
        self.data_name_list = [fn for fn in self.data_list]
        self.data_name_to_data = {}
        self.data_name_to_fps_idxes = {}
        
        ## get data ##
        self._preload_data()
        self.data_name_to_statistics = {}
    
    
    
    def __len__(self):
        # data_name_to_data, data_name_to_fps_idxes #
        return len(self.data_name_to_data)
        # return len(self.data_name_list)
    
    
    
    def _load_data_from_data_name(self, data_nm):
        cur_data_fn = data_nm # + ".npy" # laod data from data nmae ##
        # cur_data_fn = os.path.join(self.data_folder, cur_data_fn, "checkpoints",self.ckpt_nm)
        cur_data = np.load(cur_data_fn, allow_pickle=True).item()
        return cur_data
    
    ''' Shadow '''
    def load_target_data(self, target_data_fn):
        
        eps = 1e-6
        target_pts = np.load(target_data_fn, allow_pickle=True).item()
        target_pts = target_pts['ts_to_optimized_pts_integrated_w_tau'] ## get the optimized pts #
        first_frame_pts = target_pts[0] # nn_pts x 3
        # tot_verts = 
        first_frame_pts = first_frame_pts[:, None]
        target_pts = np.repeat(first_frame_pts, self.nn_seq_len, axis=1) 
        ### scale the data ### 
        target_pts = (target_pts - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
        # particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
        # target_pts: nn_pts x nn_frames x 3 #
        data_E = np.concatenate(
            [target_pts, target_pts], axis=-1 ## get the target features
        )
        ## TODO: copy to the relative assets folder ##
        sampled_pts_idxes_fn = "/root/diffsim/tiny-differentiable-simulator/python/examples/assets/shadow_sampled_pts_idxes.npy"
        sampled_idxes = np.load(sampled_pts_idxes_fn) ## the int32 array ## 
        data_dict ={
            'X': target_pts,
            'E': data_E,
            'sampled_idxes': sampled_idxes
        }
        self.target_data_dict = data_dict
        return data_dict
    
    ## dataset 3d -> load the target first frame data ##
    ## write the gudied sampling processes ##
    ## get the results ##
    ## a further guided sampling ##
    
    def _parse_task_info_from_data_name(self, data_nm):
        # allegro_bottle_5_pds_allegro_flat_fivefin_yscaled_finscaled__ctlfreq_10_taskstage5_objtype_box_objm0.2_objsxyz_0.02_0.02_0.382_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.5_0.37_objgoalrot_0.2_0_0_objgoalrot2_0.1_0_0_tar_
        cur_data_folder = data_nm.split("/")[-2] # # data-nm 
        data_nm_segs = cur_data_folder.split("_")
        for i_seg, cur_nm in enumerate(data_nm_segs):
            if cur_nm[:4] == "objm":
                obj_mass = float(cur_nm[4:])
            elif cur_nm == "objsxyz":
                obj_size_x = float(data_nm_segs[i_seg + 1])
                obj_size_y = float(data_nm_segs[i_seg + 2])
                obj_size_z = float(data_nm_segs[i_seg + 3])
            elif cur_nm == "objgoalrot":
                obj_rot_x = float(data_nm_segs[i_seg + 1])
            elif cur_nm == "objgoalrot2":
                obj_rot_x_2 = float(data_nm_segs[i_seg + 1])
            # else
        task_settings = {
            'obj_m': obj_mass,
            'obj_size_x': obj_size_x,
            'obj_size_y': obj_size_y,
            'obj_size_z': obj_size_z,
            'obj_rot_x': obj_rot_x,
            'obj_rot_x_2': obj_rot_x_2
        }
        return task_settings
        
    
    def _preload_data(self, ):
        
        if self.single_inst:
            self.data_name_list = self.data_name_list[:1]
        
        # self.data_name_list #
        
        # single_clip_length = 300
        # sliding_window_length = 100
        
        for i_inst, data_nm in enumerate(self.data_name_list):
            
            if self.debug and i_inst >= 3:
                break
            
            print(f"Loading from {data_nm}")
            
            cur_data = self._load_data_from_data_name(data_nm)
            
            # selected_frame_verts, selected_frame_qtars_verts
            # tot_verts tot_verts_integrated_qdd_tau
            ## convert to the interested first frame's pose ## then chnage the pose of that data ##
            tot_verts = cur_data['tot_verts']
            
            if self.use_static_first_frame:
                tot_verts_first_frame = tot_verts[:, 0:1]
                tot_verts = np.repeat(tot_verts_first_frame, tot_verts.shape[1], axis=1)
            # print(f"tot_verts: {}")
            
            tot_verts_integrated_qdd_tau = cur_data['tot_qtar_verts']
            # tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts_s2']
            if 'tot_qtar_verts_s2' not in cur_data:
                tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts'].copy()
            else:
                tot_verts_integrated_qdd_tau_s2 = cur_data['tot_qtar_verts_s2']
            # nn_ts x nn_verts x 3 #
            # sequence length ? # # tot verts # #
            self.nn_seq_len = tot_verts.shape[1]
            
            mean_tot_verts = np.mean(tot_verts, axis=1)
            mean_tot_verts = np.mean(mean_tot_verts, axis=0)
            
            mean_tot_verts_qdd = np.mean(tot_verts_integrated_qdd_tau, axis=1)
            mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
            
            mean_tot_verts_qdd_s2 = np.mean(tot_verts_integrated_qdd_tau_s2, axis=1)
            mean_tot_verts_qdd_s2 = np.mean(mean_tot_verts_qdd_s2, axis=0)
            
            print(f"mean_tot_verts: {mean_tot_verts}, mean_tot_verts_qdd: {mean_tot_verts_qdd}, mean_tot_verts_qdd_s2: {mean_tot_verts_qdd_s2}")
            
            
            cur_data_nm = data_nm
            cur_clip_data = {
                'tot_verts': tot_verts, 
                # 'tot_verts_integrated_qd': cur_tot_verts_integrated_qd,
                'tot_verts_integrated_qdd_tau': tot_verts_integrated_qdd_tau,
                'tot_verts_integrated_qdd_tau_s2': tot_verts_integrated_qdd_tau_s2
            }
            
            cur_task_setting = self._parse_task_info_from_data_name(data_nm)
            cur_clip_data.update(cur_task_setting)
            self.data_name_to_data[cur_data_nm] = cur_clip_data
            
            init_verts = tot_verts[:, 0] # 
            particle_init_xs_th = torch.from_numpy(init_verts).float()
            
            if self.specified_sampled_particle_idxes_fn is not None and len(self.specified_sampled_particle_idxes_fn) > 0:
                sampled_particle_idxes_sv_fn = self.specified_sampled_particle_idxes_fn
            else:
                if 'allegro_flat_fivefin_yscaled_finscaled' in data_nm:
                    sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_finscaled_sampled_particle_idxes.npy")
                elif 'allegro_flat_fivefin_yscaled' in data_nm:
                    sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_yscaled_sampled_particle_idxes.npy")
                elif 'allegro_flat_fivefin' in data_nm:
                    sampled_particle_idxes_sv_fn = os.path.join("assets",f"allegro_flat_fivefin_sampled_particle_idxes.npy")
                else:
                    ## al
                    ### get the particle idxes  ###
                    # get partcle init xs #
                    sampled_particle_idxes_sv_fn = os.path.join("assets", "sampled_particle_idxes.npy")
            
            if not os.path.exists(sampled_particle_idxes_sv_fn):
                sampled_particle_idxes = farthest_point_sampling(particle_init_xs_th.unsqueeze(0), self.maxx_nn_pts).cpu().numpy()
                np.save(sampled_particle_idxes_sv_fn, sampled_particle_idxes)
            else:
                sampled_particle_idxes = np.load(sampled_particle_idxes_sv_fn, allow_pickle=True)
            
            self.data_name_to_fps_idxes[cur_data_nm] = sampled_particle_idxes
            
            
        # self.data name to fps idxes #
        print(f"Data loaded with: {self.data_name_to_data}")
        print(f"Data loaded with number: {len(self.data_name_to_data)}")
        # data name list #
        self.data_name_list = list(self.data_name_to_data.keys()) # data name to data # 
    
        
    
    
    
    def get_closest_training_data(self, data_dict):
        # print(f"getting the closest training data")
        # for each bsz --- find the cloest training data from self.data_name_to_data
        if len(self.data_name_to_data) == 0:
            cloest_training_data = {}
            return cloest_training_data
        # print(f"[2] getting the closest training data")
        
        nn_bsz = data_dict['tot_verts'].shape[0]
        cloest_training_data = { } 
        for i_sample in range(nn_bsz):
            cur_closest_sample_key = None
            minn_dist_w_training = 9999999.9
            
            # 'tot_verts_dd_tau': particle_accs_tau,
            # 'tot_verts_dd': particle_accs,
            # 'tot_verts_dd_final': particle_accs_final
            
            for cur_data_nm in self.data_name_to_data:
                cur_data_dict = self.data_name_to_data[cur_data_nm]
                
                data_key_diff = 0.0
                for key in  cur_data_dict:
                    cur_data_key_value = cur_data_dict[key]
                    cur_sample_key_value = data_dict[key][i_sample]
                    
                    cur_data_key_diff = np.linalg.norm(cur_data_key_value - cur_sample_key_value)
                    data_key_diff += cur_data_key_diff.item()
                if data_key_diff < minn_dist_w_training or cur_closest_sample_key is None:
                    cur_closest_sample_key = cur_data_nm
                    minn_dist_w_training = data_key_diff
                
                # cur_data_init_verts = cur_data_dict['init_verts']
                
                # cur_data_accs_tau = cur_data_dict['tot_verts_dd_tau']
                # cur_data_accs = cur_data_dict['tot_verts_dd']
                # cur_data_accs_final = cur_data_dict[]
            for key in data_dict:
                if key not in cloest_training_data:
                    cloest_training_data[key] = [self.data_name_to_data[cur_closest_sample_key][key]]
                else:
                    cloest_training_data[key].append(self.data_name_to_data[cur_closest_sample_key][key])
        for key in cloest_training_data:
            cloest_training_data[key] = np.stack(cloest_training_data[key], axis=0) # bsz x nn_particles x feat_dim

        return cloest_training_data
    
    
    def inv_scale_data_v2(self, data_dict): # bsz x nn_particles x feat_dim #
        data_X = data_dict['X']
        data_E = data_dict['E']
        if 'sampled_idxes' in data_dict:
            sampled_idxes = data_dict['sampled_idxes']
        else:
            sampled_idxes = None
        # data_X_inv_scaled = (data_X * 2.0 + 1.0) / 10.0
        # data_E_inv_scaled = (data_E * 2.0 + 1.0) / 2.0
        
        eps = 1e-6
        
        ## inv_scale data ##
        # bsz x nn_particles x nn_ts x 3
        # 
        # data_X_inv_scaled = (data_X * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        # data_E_inv_scaled = (data_E * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None]
        
        th_avg_particle_init_xs = torch.from_numpy(self.avg_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_init_xs = torch.from_numpy(self.std_verts_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        

        
        th_avg_particle_xs_integrated_taus = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        th_avg_particle_xs_integrated_taus_s2 = torch.from_numpy(self.avg_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        th_std_particle_xs_integrated_taus_s2=  torch.from_numpy(self.std_verts_qdd_tau_tot_cases_tot_ts_s2).float().to(data_X.device).unsqueeze(0).unsqueeze(0).unsqueeze(0) # (None, None, None, 3)
        
        
        
        inv_scaled_particle_xs = (data_X * (th_std_particle_init_xs + eps)) + th_avg_particle_init_xs
        
        data_verts, data_verts_s2 = data_E[..., :3], data_E[..., 3:]
        inv_scaled_particle_xs_integrated_taus = (data_verts * (th_std_particle_xs_integrated_taus + eps)) + th_avg_particle_xs_integrated_taus ## get the inv_scaled integrated taus ##
        
        inv_scaled_particle_xs_integrated_taus_s2 = (data_verts_s2 * (th_std_particle_xs_integrated_taus_s2 + eps)) + th_avg_particle_xs_integrated_taus_s2 ## get the inv_scaled integrated taus ##
        
        
        ###### ======= n-scale the data ======= ######
        # data_E_inv_scaled = data_E
        # data_X_inv_scaled = data_X
        ###### ======= n-scale the data ======= ######
        
        
        rt_dict = {
            'tot_verts': inv_scaled_particle_xs,
            'tot_verts_integrated_qdd_tau': inv_scaled_particle_xs_integrated_taus,
            'tot_verts_integrated_qdd_tau_s2': inv_scaled_particle_xs_integrated_taus_s2 # inv scaled #
        }
        
        if 'sampled_idxes' in data_dict:
            rt_dict['sampled_idxes'] = sampled_idxes
        
        return rt_dict
    
    
    def scale_data(self, data_dict, data_nm):
        
        ## nn_ts x nn_particles x 3 ##
        
        eps = 1e-6
        particle_xs = data_dict['tot_verts']
        particle_xs_integrated_qdd_tau = data_dict['tot_verts_integrated_qdd_tau']
        particle_xs_integrated_qdd_tau_s2 = data_dict['tot_verts_integrated_qdd_tau_s2']
        
        
        particle_xs = (particle_xs - self.avg_verts_tot_cases_tot_ts[None][None]) / (self.std_verts_tot_cases_tot_ts[None][None] + eps)
        particle_xs_integrated_qdd_tau = (particle_xs_integrated_qdd_tau - self.avg_verts_qdd_tau_tot_cases_tot_ts[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts[None][None] + eps)
        particle_xs_integrated_qdd_tau_s2 = (particle_xs_integrated_qdd_tau_s2 - self.avg_verts_qdd_tau_tot_cases_tot_ts_s2[None][None]) / (self.std_verts_qdd_tau_tot_cases_tot_ts_s2[None][None] + eps)
        # sampled_particle_idxes = np.random.permutation(particle_init_xs.shape[0])[: self.maxx_nn_pts] #
        sampled_particle_idxes = self.data_name_to_fps_idxes[data_nm] #
        
        
        particle_xs = particle_xs[sampled_particle_idxes, :, :]
        particle_xs_integrated_qdd_tau = particle_xs_integrated_qdd_tau[sampled_particle_idxes, :, :] ## get the sampled particles qdd tau ##
        particle_xs_integrated_qdd_tau_s2 = particle_xs_integrated_qdd_tau_s2[sampled_particle_idxes, :, :]
        
        data_E = np.concatenate([particle_xs_integrated_qdd_tau, particle_xs_integrated_qdd_tau_s2], axis=-1)
        
        # task_settings = {
        #     'obj_m': obj_mass,
        #     'obj_size_x': obj_size_x,
        #     'obj_size_y': obj_size_y,
        #     'obj_size_z': obj_size_z,
        #     'obj_rot_x': obj_rot_x,
        #     'obj_rot_x_2': obj_rot_x_2
        # }
        
        
        ###### ====== get and encode the task settings ======= #######
        obj_mass = data_dict['obj_m']
        obj_size_x = data_dict['obj_size_x']
        obj_size_y = data_dict['obj_size_y']
        obj_size_z = data_dict['obj_size_z']
        obj_rot_x = data_dict['obj_rot_x']
        obj_rot_x_2 = data_dict['obj_rot_x_2'] # obj rot x 2 #
        obj_task_setting = np.array(
            [obj_mass, obj_size_x, obj_size_y, obj_size_z, obj_rot_x, obj_rot_x_2], dtype=np.float32
        )
        ###### ====== get and encode the task settings ======= #######
        # obj task setting #
        
        
        return {
            'X': particle_xs,
            'E': data_E,
            'sampled_idxes': sampled_particle_idxes,
            'obj_task_setting': obj_task_setting
            # ''
        }
    
    
    
    def data_dict_to_th(self, data_dict_np):
        
        data_dict_th = {}
        for key in data_dict_np:
            if key in ['sampled_idxes']:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).long()
            else:
                data_dict_th[key] = torch.from_numpy(data_dict_np[key]).float()
            # print(f"key: {key}, data: {data_dict_th[key].size()}")
        
        return data_dict_th
    
    
    ## ### getitem ### ##
    # def get_data_via_index(self, index) --> getitem ##
    def __getitem__(self, index): # ge the data #
        # print(f"data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        cur_data_nm = self.data_name_list[index]

        if cur_data_nm not in self.data_name_to_data:
            cur_data = self._load_data_from_data_name(cur_data_nm)
            self.data_name_to_data[cur_data_nm] = cur_data
            # print(f"[2] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ")
        else: # cur_data_nm #
            cur_data = self.data_name_to_data[cur_data_nm] ### get the data name here ###
        
        ## TODO: data selecting, data parsing, and data scaling ##
        # if self.use_target_data: # use target data ###
        #     cur_data_scaled = self.get_target_data() ###
        # else:
        # use target data for the scaling #
        
        if self.sampling and self.use_shadow_test_data:
            cur_data_scaled = self.target_data_dict
        else:
            cur_data_scaled = self.scale_data(cur_data, cur_data_nm) ## scale the data
        
        # ## ## # # scale data ## ## cur data scaled ##
        # cur_data_std, cur_data_avg = cur_data_scaled['std'], cur_data_scaled['avg']
        # self.data_name_to_statistics[cur_data_nm] = { ## data scaled ##
        #     'std': cur_data_std,
        #     'avg': cur_data_avg
        # }
        
        ## cur datascaled ##
        
        cur_data_scaled_th = self.data_dict_to_th(cur_data_scaled)
        # print(f"[3] data_name_list: {self.data_name_list}, data_name_to_data: {self.data_name_to_data.keys()} ") 
        return cur_data_scaled_th
    
import torch

import pytorch_kinematics as pk

# import os.path.join as pjoin

from os.path import join as pjoin

import trimesh

import numpy as np
import os
from scipy.spatial.transform import Rotation as R

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64


def load_visual_meshes(mesh_root, visual_mesh_fn):
    mesh_fn = pjoin(mesh_root, visual_mesh_fn)
    mesh = trimesh.load(mesh_fn, force='mesh')
    vertts = mesh.vertices
    
    # .obj
    mesh_fn_root = "/".join(mesh_fn.split("/")[:-1])
    sampled_pts_fn = mesh_fn.split("/")[-1].split(".")[0] + "_sampled_pts.npy"
    sampled_pts_root = pjoin(mesh_fn_root, "sampled_pts")
    sampled_pts_fn = pjoin(sampled_pts_root, sampled_pts_fn)
    sampled_pts = np.load(sampled_pts_fn, )
    
    
    return sampled_pts

def test_pk(urdf_fn):
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    chain = chain.to(dtype=dtype, device=d)
    print(len(chain.get_joint_parameter_names()))
    
    # get the name to visual meshes #
    # links = chain.get_links()
    # for link in links:
    #     print(link)
    # linka_names = chain.get_link_names()
    N = 1000
    th_batch = torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)
    tg_batch = chain.forward_kinematics(th_batch[0:1])
    # for i in range(N):
    #     tg = chain.forward_kinematics(th_batch[i])
    print(type(tg_batch))
    print(tg_batch.keys())
    tot_meshes_verts = []
    for key in tg_batch.keys():
        # print(key, tg_batch[key].shape)
        link = chain.find_link(key)
        # print(f"key: {key}, link: {link}")
        link_visuals = link.visuals
        print(len(link_visuals))
        
        
        for cur_visual in link_visuals:
            m_offset = cur_visual.offset # .get_matrix()
            cur_visual_mesh_fn = cur_visual.geom_param
            m = tg_batch[key].get_matrix()
            pos = m[:, :3, 3].float()
            # rot = pk.matrix_to_quaternion(m[:, :3, :3])
            rot = m[:, :3, :3].float()
            pos = pos[0] # 
            rot = rot[0] # 
            print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
            if cur_visual_mesh_fn is None: # 
                continue ## ##
            
            verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
            verts = torch.from_numpy(verts).float().to(d) # 
            verts = verts.cpu()
            verts = m_offset.transform_points(verts)
            # 
            transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
            # transforme
            print(f"transformed_verts: {transformed_verts.size()}")
            tot_meshes_verts.append(transformed_verts)
    transformed_verts = torch.cat(tot_meshes_verts, dim=0)
    transformed_verts = transformed_verts.detach().cpu().numpy()
    print(transformed_verts.shape)
    return transformed_verts
    
    # transformed_verts #

# hand pk #

def hand_pk(urdf_fn, hand_qs, to_numpy=True):
    
    # allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    
    if 'allegro_hand_description_right_franka' in urdf_fn:
        # allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link', 'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_link8', 'part_studio_link', 'camera_link']
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link', 'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_link8',]
    else:
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    chain = chain.to(dtype=dtype, device=d)
    print(len(chain.get_joint_parameter_names()))
    nn_hand_dof = len(chain.get_joint_parameter_names())
    
    
    if not isinstance(hand_qs, torch.Tensor):
        
        hand_qs = torch.from_numpy(hand_qs).double().to(d)[:, :nn_hand_dof]
    
    # else:
    #     hand_qs = 
    
    # get the name to visual meshes ##
    # links = chain.get_links()
    # for link in links:
    #     print(link)
    # linka_names = chain.get_link_names()
    N = 1000 # # # #
    # th_batch = hand_qs[:16] # torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)
    th_batch = hand_qs
    tg_batch = chain.forward_kinematics(th_batch[:]) # forward the chain #
    # for i in range(N):
    #     tg = chain.forward_kinematics(th_batch[i])
    print(type(tg_batch))
    print(tg_batch.keys())
    
    link_key_to_vis_mesh = {}
    
    tot_hand_verts = []
    tot_ts_tot_link_idxes = []
    
    for i_ts in range(th_batch.size(0)):
        # if i_ts >= 5:
        #     break
        tot_meshes_verts = []
        tot_link_idxes = []
        # for key in tg_batch.keys():
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                rot = m[:, :3, :3].float()
                pos = pos[i_ts]
                rot = rot[i_ts]
                print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                verts = link_key_to_vis_mesh[key]
                # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                
                
                verts = torch.from_numpy(verts).float().to(d)
                # if to_numpy:
                verts = verts.cpu()
                verts = m_offset.transform_points(verts)
                
                if not to_numpy:
                    verts = verts.to(d) 
                # 
                if to_numpy:
                    transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                else:
                    transformed_verts = torch.matmul(verts, rot.transpose(0,1)) + pos.unsqueeze(0)
                # transform #
                print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                tot_link_idxes.append(cur_visual_link_idxes)
                
        transformed_verts = torch.cat(tot_meshes_verts, dim=0)
        if to_numpy:
            transformed_verts = transformed_verts.detach().cpu().numpy()
        tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
        # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
        # print(transformed_verts.shape)
        tot_hand_verts.append(transformed_verts)
        tot_ts_tot_link_idxes.append(tot_link_idxes)
    if to_numpy:
        tot_hand_verts = np.stack(tot_hand_verts, axis= 0)
    else:
        tot_hand_verts = torch.stack(tot_hand_verts, axis= 0)
    tot_ts_tot_link_idxes = np.stack(tot_ts_tot_link_idxes, axis=0)
    return tot_hand_verts, tot_ts_tot_link_idxes
    
    
    

def add_transformed_hand_verts(hand_type, retar_info_fn, w_arm=False):
    retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    hand_qs = retar_info['robot_delta_states_weights_np']
    if hand_type == 'allegro':
        if w_arm:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        else:
            urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    else:
        raise NotImplementedError
    tot_converted_hand_verts = hand_pk(urdf_fn, hand_qs=hand_qs)
    # retar_info['ts_to_allegro'] = tot_converted_hand_verts
    # np.save(retar_info_fn, retar_info)
    return tot_converted_hand_verts


def transform_hand_verts(hand_type, samples_fn, w_arm=False, bsz_idx=0):
    samples = np.load(samples_fn, allow_pickle=True).item()
    training_samples = samples['closest_training_data'] # samples fn # # jcloese training data # 
    print(f"training_samples: {training_samples.keys()}")
    gt_future_hand_pose = training_samples['future_hand_pose']
    samples = samples['samples']
    hand_pose = samples['hand_pose']
    print(f"hand_pose: {hand_pose.shape}, gt_future_hand_pose: {gt_future_hand_pose.shape}")
    
    
    hand_pose = hand_pose[bsz_idx]
    gt_future_hand_pose = gt_future_hand_pose[bsz_idx]
    # print(f"samples: {samples.keys()}") # get themodel #
    # return
    # hand_qs = retar_info['robot_delta_states_weights_np']
    
    if hand_type == 'allegro':
        if w_arm:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        else:
            urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    else:
        raise NotImplementedError
    tot_converted_hand_verts = hand_pk(urdf_fn, hand_qs=hand_pose)
    gt_verts = hand_pk(urdf_fn, gt_future_hand_pose)
    # retar_info['ts_to_allegro'] = tot_converted_hand_verts
    # np.save(retar_info_fn, retar_info)
    return tot_converted_hand_verts, gt_verts

def transform_hand_verts_AE_Diff(hand_type, samples_fn, w_arm=False):
    nn_hand_dof = 22
    samples = np.load(samples_fn, allow_pickle=True).item()
    training_samples = samples['closest_training_data']
    print(f"training_samples: {training_samples.keys()}")
    gt_future_hand_pose = training_samples['future_hand_pose']
    samples = samples['samples']
    print(f"samples: {samples.keys()}")
    hand_pose = samples['E']
    hand_pose = hand_pose[..., :nn_hand_dof]
    print(f"hand_pose: {hand_pose.shape}, gt_future_hand_pose: {gt_future_hand_pose.shape}")
    
    i_bsz = 0
    i_bsz = 1
    
    hand_pose = hand_pose[i_bsz]
    gt_future_hand_pose = gt_future_hand_pose[i_bsz]
    # print(f"samples: {samples.keys()}")
    # return
    # hand_qs = retar_info['robot_delta_states_weights_np']
    
    if hand_type == 'allegro':
        if w_arm:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        else:
            urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    else:
        raise NotImplementedError
    tot_converted_hand_verts = hand_pk(urdf_fn, hand_qs=hand_pose)
    gt_verts = hand_pk(urdf_fn, gt_future_hand_pose)
    return tot_converted_hand_verts, gt_verts
    

def inspect_retar_info_fn(retar_info_fn):
    retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    print(f"retar_info: {retar_info.keys()}")
    ts_to_hand_qs = retar_info["robot_delta_states_weights_np"] 
    passive_meshes = retar_info['passive_meshes']
    # ts_to_obj_qs = retar_info['ts_to_obj_qs']   
    # tot_ts_keys = list(ts_to_hand_qs.keys())
    maxx_ts = ts_to_hand_qs.shape[0] # max(tot_ts_keys) # models #
    tot_hand_qs = []
    for i_ts in range(maxx_ts):
        cur_ts_hand_qs = ts_to_hand_qs[i_ts]
        tot_hand_qs.append(cur_ts_hand_qs)
    tot_hand_qs = np.stack(tot_hand_qs, axis=0)
    
    return tot_hand_qs, passive_meshes

def parse_obj_type_from_file_name(file_name):
    pure_file_name = file_name.split("/")[-1].split(".")[0]
    file_st_flag = "passive_active_info_"
    pure_obj_type = pure_file_name[len(file_st_flag): ]
    pure_obj_type = pure_obj_type.split("_nf_")[0]
    return pure_obj_type


def load_obj_mesh(obj_type):
    obj_mesh_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
    obj_mesh_fn = os.path.join(obj_mesh_root, f"{obj_type}.obj")
    obj_mesh = trimesh.load(obj_mesh_fn, force='mesh')
    mesh_verts, mesh_faces = obj_mesh.vertices, obj_mesh.faces
    return mesh_verts
    




# def transform_hand_verts_kinematics(hand_type, samples_fn, w_arm=False, bsz_idx=0, debug=False):
#     samples = np.load(samples_fn, allow_pickle=True).item()
#     # load the hand and the object 
#     hand_pose = samples['robot_delta_states_weights_np']
#     obj_pos = samples['object_transl']
#     obj_ornt = samples['object_rot_quat']
    
#     if debug:
#         hand_pose = hand_pose[:10]
#         obj_pos = obj_pos[:10]
#         obj_ornt = obj_ornt[:10]
    
#     if hand_type == 'allegro':
#         if w_arm:
#             urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
#         else:
#             urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
#     else:
#         raise NotImplementedError
#     converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=hand_pose)
#     # gt_verts = hand_pk(urdf_fn, gt_future_hand_pose)
    
#     tot_verts_nn = converted_hand_verts.shape[1]
#     sampled_verts_idxes = np.random.permutation(tot_verts_nn)[:512]
#     converted_hand_verts = converted_hand_verts[:, sampled_verts_idxes] # the sampled verts idexes #
#     tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[:, sampled_verts_idxes]
    
#     ts_to_hand_verts_idxes = {}
#     for i_v in range(converted_hand_verts.shape[1]):
#         cur_hand_part_idx = tot_ts_tot_link_idxes[0, i_v].item()
#         if cur_hand_part_idx not in ts_to_hand_verts_idxes:
#             ts_to_hand_verts_idxes[cur_hand_part_idx] = [i_v]
#         else:
#             ts_to_hand_verts_idxes[cur_hand_part_idx].append(i_v)
#     for ts in ts_to_hand_verts_idxes:
#         ts_to_hand_verts_idxes[ts] = np.array(ts_to_hand_verts_idxes[ts], dtype=np.int32)
    
#     pure_obj_type = parse_obj_type_from_file_name(samples_fn)
#     obj_mesh_verts = load_obj_mesh(pure_obj_type)
    
#     sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
#     obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
#     sorted_body_idxes = sorted(list(ts_to_hand_verts_idxes.keys()))
#     maxx_body_idx = max(sorted_body_idxes)
    
    
#     ### tranfor mthe mesh vertices via obj transformations ###
#     transformed_obj_verts = []
#     tot_body_contact_flags = []
#     for i_ts in range(obj_pos.shape[0]):
#         cur_obj_pos = obj_pos[i_ts]
#         cur_obj_ornt = obj_ornt[i_ts]
#         # get the object transformation
#         rot_struct = R.from_quat(cur_obj_ornt)
#         rot_matrix= rot_struct.as_matrix()
#         cur_transformed_obj_verts = (rot_matrix @ obj_mesh_verts.T).T + cur_obj_pos[None]
        
#         body_contact_flag = np.zeros((maxx_body_idx + 1,), dtype=np.int32)
        
#         for i_body in sorted_body_idxes:
#             # if i_body in ts_to_hand_verts_idxes:
#             #     body_contact_flag[i_body] = 1
#             cur_body_pts_idxes = ts_to_hand_verts_idxes[i_body]
#             cur_body_pts = converted_hand_verts[i_ts, cur_body_pts_idxes]
#             # dist_to_obj = np.linalg.norm(cur_body_pts - cur_obj_pos, axis=1) # get the ji
#             dist_to_obj = np.sum(
#                 (cur_body_pts[:, None] - cur_transformed_obj_verts[None])**2, axis=-1
#             )
#             dist_to_obj = np.min(dist_to_obj, axis=1)
#             minn_dist_to_obj = np.min(dist_to_obj)
#             if minn_dist_to_obj < (0.01) ** 2:
#                 body_contact_flag[i_body] = 1
#         tot_body_contact_flags.append(body_contact_flag)
#         print(body_contact_flag)
#         # obj_transf = pk.matrix_from_quaternion(cur_obj_ornt, cur_obj_pos)
#         transformed_obj_verts.append(cur_transformed_obj_verts)
#     transformed_obj_verts = np.stack(transformed_obj_verts, axis=0)
#     tot_body_contact_flags = np.stack(tot_body_contact_flags, axis=0)
#     return converted_hand_verts, transformed_obj_verts, tot_ts_tot_link_idxes, tot_body_contact_flags


#     # print(f"samples: {samples.keys()}")
#     return 
    
#     training_samples = samples['closest_training_data'] # samples fn #
#     print(f"training_samples: {training_samples.keys()}")
#     gt_future_hand_pose = training_samples['future_hand_pose']
#     samples = samples['samples']
#     hand_pose = samples['hand_pose']
#     print(f"hand_pose: {hand_pose.shape}, gt_future_hand_pose: {gt_future_hand_pose.shape}")
    
    
#     hand_pose = hand_pose[bsz_idx]
#     gt_future_hand_pose = gt_future_hand_pose[bsz_idx]
#     # print(f"samples: {samples.keys()}")
#     # return
#     # hand_qs = retar_info['robot_delta_states_weights_np']
    
#     if hand_type == 'allegro':
#         if w_arm:
#             urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
#         else:
#             urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
#     else:
#         raise NotImplementedError
#     tot_converted_hand_verts = hand_pk(urdf_fn, hand_qs=hand_pose)
#     gt_verts = hand_pk(urdf_fn, gt_future_hand_pose)
#     # retar_info['ts_to_allegro'] = tot_converted_hand_verts
#     # np.save(retar_info_fn, retar_info)
#     return tot_converted_hand_verts, gt_verts




def transform_hand_verts_kinematics(hand_type, samples_fn , pure_obj_type, w_arm=False,  debug=False):
    samples = np.load(samples_fn, allow_pickle=True).item()
    print(f"samples: {samples.keys()}")
    # load the hand and the object 
    if 'hand_qs' in samples:
        hand_pose = samples['hand_qs']
        obj_pos = samples['obj_pos']
        obj_ornt = samples['obj_rot']
    else:
        hand_pose = samples['robot_delta_states_weights_np']
        obj_pos = samples['object_transl']
        obj_ornt = samples['object_rot_quat']
    
    if debug:
        hand_pose = hand_pose[:10]
        obj_pos = obj_pos[:10]
        obj_ornt = obj_ornt[:10]
    
    if hand_type == 'allegro':
        if w_arm:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        else:
            urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    else:
        raise NotImplementedError
    converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=hand_pose)
    # gt_verts = hand_pk(urdf_fn, gt_future_hand_pose)
    
    tot_verts_nn = converted_hand_verts.shape[1]
    sampled_verts_idxes = np.random.permutation(tot_verts_nn)[:512]
    converted_hand_verts = converted_hand_verts[:, sampled_verts_idxes] # the sampled verts idexes #
    tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[:, sampled_verts_idxes]
    
    ts_to_hand_verts_idxes = {}
    for i_v in range(converted_hand_verts.shape[1]):
        cur_hand_part_idx = tot_ts_tot_link_idxes[0, i_v].item()
        if cur_hand_part_idx not in ts_to_hand_verts_idxes:
            ts_to_hand_verts_idxes[cur_hand_part_idx] = [i_v]
        else:
            ts_to_hand_verts_idxes[cur_hand_part_idx].append(i_v)
    for ts in ts_to_hand_verts_idxes:
        ts_to_hand_verts_idxes[ts] = np.array(ts_to_hand_verts_idxes[ts], dtype=np.int32)
    
    # pure_obj_type = parse_obj_type_from_file_name(samples_fn)
    obj_mesh_verts = load_obj_mesh(pure_obj_type)
    
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    sorted_body_idxes = sorted(list(ts_to_hand_verts_idxes.keys()))
    maxx_body_idx = max(sorted_body_idxes)
    
    
    ### tranfor mthe mesh vertices via obj transformations ###
    transformed_obj_verts = []
    tot_body_contact_flags = []
    for i_ts in range(obj_pos.shape[0]):
        cur_obj_pos = obj_pos[i_ts]
        cur_obj_ornt = obj_ornt[i_ts]
        # get the object transformation # get the object transformation ##
        rot_struct = R.from_quat(cur_obj_ornt)
        rot_matrix= rot_struct.as_matrix()
        cur_transformed_obj_verts = (rot_matrix @ obj_mesh_verts.T).T + cur_obj_pos[None]
        
        body_contact_flag = np.zeros((maxx_body_idx + 1,), dtype=np.int32)
        
        for i_body in sorted_body_idxes:
            # if i_body in ts_to_hand_verts_idxes:
            #     body_contact_flag[i_body] = 1
            cur_body_pts_idxes = ts_to_hand_verts_idxes[i_body]
            cur_body_pts = converted_hand_verts[i_ts, cur_body_pts_idxes]
            # dist_to_obj = np.linalg.norm(cur_body_pts - cur_obj_pos, axis=1) # get the ji
            dist_to_obj = np.sum(
                (cur_body_pts[:, None] - cur_transformed_obj_verts[None])**2, axis=-1
            )
            dist_to_obj = np.min(dist_to_obj, axis=1)
            minn_dist_to_obj = np.min(dist_to_obj)
            if minn_dist_to_obj < (0.01) ** 2:
                body_contact_flag[i_body] = 1
        tot_body_contact_flags.append(body_contact_flag)
        print(body_contact_flag)
        # obj_transf = pk.matrix_from_quaternion(cur_obj_ornt, cur_obj_pos)
        transformed_obj_verts.append(cur_transformed_obj_verts)
    transformed_obj_verts = np.stack(transformed_obj_verts, axis=0)
    tot_body_contact_flags = np.stack(tot_body_contact_flags, axis=0)
    return converted_hand_verts, transformed_obj_verts, tot_ts_tot_link_idxes, tot_body_contact_flags


from scipy.spatial.transform import Rotation as R
import trimesh
# from isaacgymenvs.utils.test_pk import *
# from utils.test_pk import *





def load_obj_mesh(obj_type):
    obj_mesh_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
    obj_mesh_fn = os.path.join(obj_mesh_root, f"{obj_type}.obj")
    obj_mesh = trimesh.load(obj_mesh_fn, force='mesh')
    mesh_verts, mesh_faces = obj_mesh.vertices, obj_mesh.faces
    return mesh_verts
    



# traj modification opt #
def traj_modification_opt(inst_tag):
    kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
    kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
    kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
    print(f"kine_traj_data: {kine_traj_data.keys()}")
    
    
    joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # kine_obj_rot = kine_traj_data['']
    kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    # kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering]
    # it seems that convert them into zyx would be more stable than converting into the xyz #
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    diff_kine_obj_pos = kine_obj_pos[1:] - kine_obj_pos[:-1] # (nn_frames - 1) x 3 #
    diff_kine_obj_rot_euler = kine_obj_rot_euler[1:] - kine_obj_rot_euler[:-1] # (nn_frames - 1) x 3 #
    diff_kine_hand_trans = kine_hand_qs[1:, :3] - kine_hand_qs[:-1, :3] # (nn_frames - 1) x 3 #
    
    
    new_diff_kine_obj_rot_euler = []
    for i_fr in range(diff_kine_obj_rot_euler.shape[0]):
        cur_diff_kine_obj_rot_euler = diff_kine_obj_rot_euler[i_fr]
        if np.any(cur_diff_kine_obj_rot_euler >= np.pi) or np.any(cur_diff_kine_obj_rot_euler <= -np.pi):
            continue
        new_diff_kine_obj_rot_euler.append(cur_diff_kine_obj_rot_euler)
    
    new_diff_kine_obj_rot_euler = np.stack(new_diff_kine_obj_rot_euler, axis=0)
    diff_kine_obj_rot_euler = new_diff_kine_obj_rot_euler
    
    # diff_kine_obj_rot_euler = np.clip(diff_kine_obj_rot_euler, a_min=-np.pi, a_max=np.pi)
    
    
    maxx_diff_kine_obj_pos, minn_diff_kine_obj_pos = np.max(diff_kine_obj_pos, axis=0), np.min(diff_kine_obj_pos, axis=0)
    maxx_diff_kine_obj_rot_euler, minn_diff_kine_obj_rot_euler = np.max(diff_kine_obj_rot_euler, axis=0), np.min(diff_kine_obj_rot_euler, axis=0)
    maxx_diff_hand_trans, minn_diff_hand_trans = np.max(diff_kine_hand_trans, axis=0), np.min(diff_kine_hand_trans, axis=0) # 
    print(f"maxx_diff_kine_obj_pos: {maxx_diff_kine_obj_pos}, minn_diff_kine_obj_pos: {minn_diff_kine_obj_pos}")
    print(f"maxx_diff_kine_obj_rot_euler: {maxx_diff_kine_obj_rot_euler}, minn_diff_kine_obj_rot_euler: {minn_diff_kine_obj_rot_euler}")
    print(f"maxx_diff_hand_trans: {maxx_diff_hand_trans}, minn_diff_hand_trans: {minn_diff_hand_trans}")
    
    
    # maxx_diff_kine_obj_pos: [0.00189576 0.0024092  0.00845268], minn_diff_kine_obj_pos: [-0.00199135 -0.01218224 -0.00612274]
    # maxx_diff_kine_obj_rot_euler: [0.028512   0.04186282 0.06846939], minn_diff_kine_obj_rot_euler: [-0.05868825 -0.06077137 -0.03629342]
    # maxx_diff_hand_trans: [0.00316119 0.00237919 0.00523315], minn_diff_hand_trans: [-0.00560641 -0.00521771 -0.00337168]
        
    
    
    start_gen_frame = 120
    nn_continuing_frames = 180
    maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) # 
    
    maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
    minn_obj_rot_diff = np.array(minn_obj_rot_diff, dtype=np.float32) # get the obj rot diff and the maxx obj rot diff #
    # maxx_hand_trans_diff = [0.00316119 , 0.00237919 , 0.00523315]
    # minn_hand_trans_diff = [-0.00560641 , -0.00521771 , -0.00337168]
    # maxx_hand_trans_diff = np.array(maxx_hand_trans_diff, dtype=np.float32)
    # minn_hand_trans_diff = np.array(minn_hand_trans_diff, dtype=np.float32)
    
    
    nn_samples = 100
    
    tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    palm_name = 'palm_link'
    first_tip_name = 'link_3_tip'
    second_tip_name = 'link_7_tip'
    third_tip_name = 'link_11_tip'
    forth_tip_name = 'link_15_tip'
    
    
    urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        for i_fr in range(nn_continuing_frames):  # get the hand qs, obj pos, and the obj rot #
            # cur_rand_obj_delta_pos = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_pos = tot_delta_pos_rand[sample_idx, i_fr]
            # (maxx obj pos diff - minn obj pos diff) 
            # this scaling is not current yet  # from 0, 1 -- scale to the range of minn_obj_pos_diff and maxx_obj_pos_diff
            cur_rand_obj_delta_pos = cur_rand_obj_delta_pos * (maxx_obj_pos_diff - minn_obj_pos_diff) + minn_obj_pos_diff
            # cur_rand_obj_delta_rot = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_rot = tot_delta_rot_rand[sample_idx, i_fr]
            cur_rand_obj_delta_rot = cur_rand_obj_delta_rot * (maxx_obj_rot_diff - minn_obj_rot_diff) + minn_obj_rot_diff
            cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + cur_rand_obj_delta_pos
            cur_st_obj_pos = cur_st_obj_pos + cur_rand_obj_delta_pos
            cur_st_obj_rot_euler = cur_st_obj_rot_euler + cur_rand_obj_delta_rot
            cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
            cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
            
            
            continuing_hand_qs.append(cur_st_hand_qs.copy())
            continuing_obj_pos.append(cur_st_obj_pos.copy())
            continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
            
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame]
        init_obj_ornt = kine_obj_quat[start_gen_frame] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        # init obj verts # --- # 
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]):
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        # if hand_type == 'allegro':
        #     if w_arm:
        #         urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        #     else:
        
        # else:
        #     raise NotImplementedError
        
        # converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0).cuda())
        
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())

        tot_link_idxes = []
        # for key in tg_batch.keys():
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # transform #
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0) #  # 
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()

        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3) # --- rotation matrix #
        # # matmul(rot_matrix, verts..T).T = 
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        # (3,) --- the relative positions between the object center and other hand bodies #
        # 
        # (3,) --- the relative positions 
        
        # the relative positions #
        # relative positions #
        

        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        # continuing_obj_verts
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    # # 
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
        
        
        
        
            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2)
            loss_first_tip = torch.sum(diff_first_tip_center ** 2)
            loss_second_tip = torch.sum(diff_second_tip_center ** 2)
            loss_third_tip = torch.sum(diff_third_tip_center ** 2)
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2)
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :6] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip + glb_qs_reg_loss * 10
            print(f"loss: {loss.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
            
        
        # # get the init obj verts #
        # dist_palm_to_obj = torch.sum(
        #     ( palm_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_first_tip_to_obj = torch.sum(
        #     ( first_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_second_tip_to_obj = torch.sum(
        #     ( second_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_third_tip_to_obj = torch.sum(
        #     ( third_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_forth_tip_to_obj = torch.sum(
        #     ( forth_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # # dist forth tip #
        # minn_dist_palm_obj, nearest_dist_obj_vidx = torch.min(dist_palm_to_obj, )
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot
        }
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        # resampled info sv fn #
        resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")




def traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2" ):
    # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    # kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    # kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # # kine_obj_rot = kine_traj_data['']
    # kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    # kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering]
    # it seems that convert them into zyx would be more stable than converting into the xyz #
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    diff_kine_obj_pos = kine_obj_pos[1:] - kine_obj_pos[:-1] # (nn_frames - 1) x 3 #
    diff_kine_obj_rot_euler = kine_obj_rot_euler[1:] - kine_obj_rot_euler[:-1] # (nn_frames - 1) x 3 #
    diff_kine_hand_trans = kine_hand_qs[1:, :3] - kine_hand_qs[:-1, :3] # (nn_frames - 1) x 3 #
    
    
    new_diff_kine_obj_rot_euler = []
    for i_fr in range(diff_kine_obj_rot_euler.shape[0]):
        cur_diff_kine_obj_rot_euler = diff_kine_obj_rot_euler[i_fr]
        if np.any(cur_diff_kine_obj_rot_euler >= np.pi) or np.any(cur_diff_kine_obj_rot_euler <= -np.pi):
            continue
        new_diff_kine_obj_rot_euler.append(cur_diff_kine_obj_rot_euler)
    
    new_diff_kine_obj_rot_euler = np.stack(new_diff_kine_obj_rot_euler, axis=0)
    diff_kine_obj_rot_euler = new_diff_kine_obj_rot_euler
    
    # diff_kine_obj_rot_euler = np.clip(diff_kine_obj_rot_euler, a_min=-np.pi, a_max=np.pi)
    
    
    maxx_diff_kine_obj_pos, minn_diff_kine_obj_pos = np.max(diff_kine_obj_pos, axis=0), np.min(diff_kine_obj_pos, axis=0)
    maxx_diff_kine_obj_rot_euler, minn_diff_kine_obj_rot_euler = np.max(diff_kine_obj_rot_euler, axis=0), np.min(diff_kine_obj_rot_euler, axis=0)
    maxx_diff_hand_trans, minn_diff_hand_trans = np.max(diff_kine_hand_trans, axis=0), np.min(diff_kine_hand_trans, axis=0) # 
    print(f"maxx_diff_kine_obj_pos: {maxx_diff_kine_obj_pos}, minn_diff_kine_obj_pos: {minn_diff_kine_obj_pos}")
    print(f"maxx_diff_kine_obj_rot_euler: {maxx_diff_kine_obj_rot_euler}, minn_diff_kine_obj_rot_euler: {minn_diff_kine_obj_rot_euler}")
    print(f"maxx_diff_hand_trans: {maxx_diff_hand_trans}, minn_diff_hand_trans: {minn_diff_hand_trans}")
    
    
    # maxx_diff_kine_obj_pos: [0.00189576 0.0024092  0.00845268], minn_diff_kine_obj_pos: [-0.00199135 -0.01218224 -0.00612274]
    # maxx_diff_kine_obj_rot_euler: [0.028512   0.04186282 0.06846939], minn_diff_kine_obj_rot_euler: [-0.05868825 -0.06077137 -0.03629342]
    # maxx_diff_hand_trans: [0.00316119 0.00237919 0.00523315], minn_diff_hand_trans: [-0.00560641 -0.00521771 -0.00337168]
        
    
    
    start_gen_frame = 120
    nn_continuing_frames = 180
    maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) # 
    
    maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
    minn_obj_rot_diff = np.array(minn_obj_rot_diff, dtype=np.float32) # get the obj rot diff and the maxx obj rot diff #
    # maxx_hand_trans_diff = [0.00316119 , 0.00237919 , 0.00523315]
    # minn_hand_trans_diff = [-0.00560641 , -0.00521771 , -0.00337168]
    # maxx_hand_trans_diff = np.array(maxx_hand_trans_diff, dtype=np.float32)
    # minn_hand_trans_diff = np.array(minn_hand_trans_diff, dtype=np.float32)
    
    
    nn_samples = 100
    
    tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    palm_name = 'palm_link'
    first_tip_name = 'link_3_tip'
    second_tip_name = 'link_7_tip'
    third_tip_name = 'link_11_tip'
    forth_tip_name = 'link_15_tip'
    
    
    urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        for i_fr in range(nn_continuing_frames):  # get the hand qs, obj pos, and the obj rot #
            # cur_rand_obj_delta_pos = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_pos = tot_delta_pos_rand[sample_idx, i_fr]
            # (maxx obj pos diff - minn obj pos diff) 
            # this scaling is not current yet  # from 0, 1 -- scale to the range of minn_obj_pos_diff and maxx_obj_pos_diff
            cur_rand_obj_delta_pos = cur_rand_obj_delta_pos * (maxx_obj_pos_diff - minn_obj_pos_diff) + minn_obj_pos_diff
            # cur_rand_obj_delta_rot = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_rot = tot_delta_rot_rand[sample_idx, i_fr]
            cur_rand_obj_delta_rot = cur_rand_obj_delta_rot * (maxx_obj_rot_diff - minn_obj_rot_diff) + minn_obj_rot_diff
            cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + cur_rand_obj_delta_pos
            cur_st_obj_pos = cur_st_obj_pos + cur_rand_obj_delta_pos
            cur_st_obj_rot_euler = cur_st_obj_rot_euler + cur_rand_obj_delta_rot
            cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
            cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
            
            
            continuing_hand_qs.append(cur_st_hand_qs.copy())
            continuing_obj_pos.append(cur_st_obj_pos.copy())
            continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
            
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame]
        init_obj_ornt = kine_obj_quat[start_gen_frame] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        # init obj verts # --- # 
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]):
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        # if hand_type == 'allegro':
        #     if w_arm:
        #         urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        #     else:
        
        # else:
        #     raise NotImplementedError
        
        # converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0).cuda())
        
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())

        tot_link_idxes = []
        # for key in tg_batch.keys():
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # transform #
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0) #  # 
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()

        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3) # --- rotation matrix #
        # # matmul(rot_matrix, verts..T).T = 
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        # (3,) --- the relative positions between the object center and other hand bodies #
        # 
        # (3,) --- the relative positions 
        
        # the relative positions #
        # relative positions #
        

        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        # continuing_obj_verts
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    # # 
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
        
        
        
        
            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2)
            loss_first_tip = torch.sum(diff_first_tip_center ** 2)
            loss_second_tip = torch.sum(diff_second_tip_center ** 2)
            loss_third_tip = torch.sum(diff_third_tip_center ** 2)
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2)
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :6] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip + glb_qs_reg_loss * 10
            print(f"loss: {loss.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
            
        
        # # get the init obj verts #
        # dist_palm_to_obj = torch.sum(
        #     ( palm_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_first_tip_to_obj = torch.sum(
        #     ( first_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_second_tip_to_obj = torch.sum(
        #     ( second_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_third_tip_to_obj = torch.sum(
        #     ( third_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # dist_forth_tip_to_obj = torch.sum(
        #     ( forth_tip_link_verts.unsqueeze(1) - init_obj_verts.unsqueeze(0) ) ** 2, dim=-1 # nn_verts x nn_obj_verts 
        # )
        # # dist forth tip #
        # minn_dist_palm_obj, nearest_dist_obj_vidx = torch.min(dist_palm_to_obj, )
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot
        }
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        # resampled info sv fn #
        resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")


def traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2" ):
    # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    # kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    # kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # # kine_obj_rot = kine_traj_data['']
    # kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    # kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering]
    # it seems that convert them into zyx would be more stable than converting into the xyz #
    # tracked res 
    # identify the position of the palm link in the target starting modification frame #
    # identify the position of the object in the target starting modification frame #
    # identify the line from the palm link to the object #
    # and use that axis as the rotation axis #
    
    global dtype
    dtype = torch.float32 
    
    interp_frame = 120
    nn_continuing_frames = 180
    start_gen_frame = 120
    
    obj_rot_quat = kine_obj_quat[interp_frame]
    
    delta_xy = 0.002 * np.pi
    delta_rot_xyz_euler = [delta_xy, delta_xy, 0.0]
    
    
    delta_xy = 0.004 * np.pi
    delta_rot_xyz_euler = [0, -delta_xy, -delta_xy]
    
    delta_rot_xyz_euler = np.array(delta_rot_xyz_euler, dtype=np.float32)
    obj_rot_matrix = R.from_quat(obj_rot_quat).as_matrix() # .T
    # obj_rot_matrix_delta = R.from_euler('xyz', delta_rot_xyz_euler, degrees=False).as_matrix()
    # transformed_delta_rot_xyz_euler = R.from_matrix(obj_rot_matrix @ obj_rot_matrix_delta).as_euler('zyx', degrees=False)[[2,1,0]] # (3,) obj rot deltas #
    transformed_delta_rot_xyz_euler = obj_rot_matrix @ delta_rot_xyz_euler
    transformed_delta_rot_xyz_euler = delta_rot_xyz_euler
    
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    diff_kine_obj_pos = kine_obj_pos[1:] - kine_obj_pos[:-1] # (nn_frames - 1) x 3 #
    diff_kine_obj_rot_euler = kine_obj_rot_euler[1:] - kine_obj_rot_euler[:-1] # (nn_frames - 1) x 3 #
    diff_kine_hand_trans = kine_hand_qs[1:, :3] - kine_hand_qs[:-1, :3] # (nn_frames - 1) x 3 #
    
    
    new_diff_kine_obj_rot_euler = []
    for i_fr in range(diff_kine_obj_rot_euler.shape[0]):
        cur_diff_kine_obj_rot_euler = diff_kine_obj_rot_euler[i_fr]
        if np.any(cur_diff_kine_obj_rot_euler >= np.pi) or np.any(cur_diff_kine_obj_rot_euler <= -np.pi):
            continue
        new_diff_kine_obj_rot_euler.append(cur_diff_kine_obj_rot_euler)
    
    new_diff_kine_obj_rot_euler = np.stack(new_diff_kine_obj_rot_euler, axis=0)
    diff_kine_obj_rot_euler = new_diff_kine_obj_rot_euler
    
    # diff_kine_obj_rot_euler = np.clip(diff_kine_obj_rot_euler, a_min=-np.pi, a_max=np.pi)
    
    
    maxx_diff_kine_obj_pos, minn_diff_kine_obj_pos = np.max(diff_kine_obj_pos, axis=0), np.min(diff_kine_obj_pos, axis=0)
    maxx_diff_kine_obj_rot_euler, minn_diff_kine_obj_rot_euler = np.max(diff_kine_obj_rot_euler, axis=0), np.min(diff_kine_obj_rot_euler, axis=0)
    maxx_diff_hand_trans, minn_diff_hand_trans = np.max(diff_kine_hand_trans, axis=0), np.min(diff_kine_hand_trans, axis=0) # 
    print(f"maxx_diff_kine_obj_pos: {maxx_diff_kine_obj_pos}, minn_diff_kine_obj_pos: {minn_diff_kine_obj_pos}")
    print(f"maxx_diff_kine_obj_rot_euler: {maxx_diff_kine_obj_rot_euler}, minn_diff_kine_obj_rot_euler: {minn_diff_kine_obj_rot_euler}")
    print(f"maxx_diff_hand_trans: {maxx_diff_hand_trans}, minn_diff_hand_trans: {minn_diff_hand_trans}")
    
    
    # maxx_diff_kine_obj_pos: [0.00189576 0.0024092  0.00845268], minn_diff_kine_obj_pos: [-0.00199135 -0.01218224 -0.00612274]
    # maxx_diff_kine_obj_rot_euler: [0.028512   0.04186282 0.06846939], minn_diff_kine_obj_rot_euler: [-0.05868825 -0.06077137 -0.03629342]
    # maxx_diff_hand_trans: [0.00316119 0.00237919 0.00523315], minn_diff_hand_trans: [-0.00560641 -0.00521771 -0.00337168]
        
    
    
    start_gen_frame = 120
    nn_continuing_frames = 180
    # maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    # minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    # maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    # minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    # minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) # 
    
    # maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    # minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    # maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    # minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    # maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    # minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    # maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
    # minn_obj_rot_diff = np.array(minn_obj_rot_diff, dtype=np.float32) # get the obj rot diff and the maxx obj rot diff #
    # # maxx_hand_trans_diff = [0.00316119 , 0.00237919 , 0.00523315]
    # minn_hand_trans_diff = [-0.00560641 , -0.00521771 , -0.00337168]
    # maxx_hand_trans_diff = np.array(maxx_hand_trans_diff, dtype=np.float32)
    # minn_hand_trans_diff = np.array(minn_hand_trans_diff, dtype=np.float32)
    
    minn_obj_rot_diff = transformed_delta_rot_xyz_euler
    maxx_obj_rot_diff = transformed_delta_rot_xyz_euler
    
    maxx_obj_pos_diff = np.zeros_like(minn_obj_rot_diff)
    minn_obj_pos_diff  =    np.zeros_like(minn_obj_rot_diff)
    
    
    
    nn_samples = 1
    
    tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    palm_name = 'palm_link'
    first_tip_name = 'link_3_tip'
    second_tip_name = 'link_7_tip'
    third_tip_name = 'link_11_tip'
    forth_tip_name = 'link_15_tip'
    
    
    urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        for i_fr in range(nn_continuing_frames):  # get the hand qs, obj pos, and the obj rot #
            # cur_rand_obj_delta_pos = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_pos = tot_delta_pos_rand[sample_idx, i_fr]
            # (maxx obj pos diff - minn obj pos diff) 
            # this scaling is not current yet  # from 0, 1 -- scale to the range of minn_obj_pos_diff and maxx_obj_pos_diff
            cur_rand_obj_delta_pos = cur_rand_obj_delta_pos * (maxx_obj_pos_diff - minn_obj_pos_diff) + minn_obj_pos_diff
            # cur_rand_obj_delta_rot = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_rot = tot_delta_rot_rand[sample_idx, i_fr]
            cur_rand_obj_delta_rot = cur_rand_obj_delta_rot * (maxx_obj_rot_diff - minn_obj_rot_diff) + minn_obj_rot_diff
            cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + cur_rand_obj_delta_pos
            cur_st_obj_pos = cur_st_obj_pos + cur_rand_obj_delta_pos
            cur_st_obj_rot_euler = cur_st_obj_rot_euler + cur_rand_obj_delta_rot
            cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
            cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
            
            
            continuing_hand_qs.append(cur_st_hand_qs.copy())
            continuing_obj_pos.append(cur_st_obj_pos.copy())
            continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
            
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame]
        init_obj_ornt = kine_obj_quat[start_gen_frame] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        # init obj verts # --- # 
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]):
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        # if hand_type == 'allegro':
        #     if w_arm:
        #         urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        #     else:
        
        # else:
        #     raise NotImplementedError
        
        # converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0).cuda())
        
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())

        tot_link_idxes = []
        # for key in tg_batch.keys():
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # transform #
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0) #  # 
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()

        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3) # --- rotation matrix #
        # # matmul(rot_matrix, verts..T).T = 
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        # (3,) --- the relative positions between the object center and other hand bodies #
        # 
        # (3,) --- the relative positions 
        
        # the relative positions #
        # relative positions #
        

        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        # continuing_obj_verts
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs_glb = robot_qs[..., :6].detach().clone()
            robot_qs_glb[1:, :6] = 0.0
            robot_qs = torch.cat(
                [ robot_qs_glb, robot_qs[..., 6:] ], dim=-1
            )
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    # # 
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
        
        
        
        
            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2)
            loss_first_tip = torch.sum(diff_first_tip_center ** 2)
            loss_second_tip = torch.sum(diff_second_tip_center ** 2)
            loss_third_tip = torch.sum(diff_third_tip_center ** 2)
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2)
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :6] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip + glb_qs_reg_loss * 100
            print(f"loss: {loss.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot
        }
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        # resampled info sv fn #
        resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")




def traj_modifications_core_v3(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}):
    # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    # kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    # kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # # kine_obj_rot = kine_traj_data['']
    # kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    
    global dtype
    dtype = torch.float32 
    
    interp_frame = 120
    nn_continuing_frames = 180
    start_gen_frame = 120
    
    obj_rot_quat = kine_obj_quat[interp_frame]
    
    delta_mult_coef = 0.004 * np.pi
    delta_mult_coef = 0.008 * np.pi
    
    # delta_xy = 0.002 * np.pi
    # delta_rot_xyz_euler = [delta_xy, delta_xy, 0.0]
    
    
    # delta_xy = 0.004 * np.pi
    # delta_rot_xyz_euler = [0, -delta_xy, -delta_xy]
    
    # delta_rot_xyz_euler = np.array(delta_rot_xyz_euler, dtype=np.float32)
    # obj_rot_matrix = R.from_quat(obj_rot_quat).as_matrix() # .T
    # obj_rot_matrix_delta = R.from_euler('xyz', delta_rot_xyz_euler, degrees=False).as_matrix()
    # transformed_delta_rot_xyz_euler = R.from_matrix(obj_rot_matrix @ obj_rot_matrix_delta).as_euler('zyx', degrees=False)[[2,1,0]] # (3,) obj rot deltas #
    # transformed_delta_rot_xyz_euler = obj_rot_matrix @ delta_rot_xyz_euler
    # transformed_delta_rot_xyz_euler = delta_rot_xyz_euler
    
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    # diff_kine_obj_pos = kine_obj_pos[1:] - kine_obj_pos[:-1] # (nn_frames - 1) x 3 #
    # diff_kine_obj_rot_euler = kine_obj_rot_euler[1:] - kine_obj_rot_euler[:-1] # (nn_frames - 1) x 3 #
    # diff_kine_hand_trans = kine_hand_qs[1:, :3] - kine_hand_qs[:-1, :3] # (nn_frames - 1) x 3 #
    
    
    # new_diff_kine_obj_rot_euler = []
    # for i_fr in range(diff_kine_obj_rot_euler.shape[0]):
    #     cur_diff_kine_obj_rot_euler = diff_kine_obj_rot_euler[i_fr]
    #     if np.any(cur_diff_kine_obj_rot_euler >= np.pi) or np.any(cur_diff_kine_obj_rot_euler <= -np.pi):
    #         continue
    #     new_diff_kine_obj_rot_euler.append(cur_diff_kine_obj_rot_euler)
    
    # new_diff_kine_obj_rot_euler = np.stack(new_diff_kine_obj_rot_euler, axis=0)
    # diff_kine_obj_rot_euler = new_diff_kine_obj_rot_euler
    
    # # diff_kine_obj_rot_euler = np.clip(diff_kine_obj_rot_euler, a_min=-np.pi, a_max=np.pi)
    
    # maxx_diff_kine_obj_pos, minn_diff_kine_obj_pos = np.max(diff_kine_obj_pos, axis=0), np.min(diff_kine_obj_pos, axis=0)
    # maxx_diff_kine_obj_rot_euler, minn_diff_kine_obj_rot_euler = np.max(diff_kine_obj_rot_euler, axis=0), np.min(diff_kine_obj_rot_euler, axis=0)
    # maxx_diff_hand_trans, minn_diff_hand_trans = np.max(diff_kine_hand_trans, axis=0), np.min(diff_kine_hand_trans, axis=0) # 
    # print(f"maxx_diff_kine_obj_pos: {maxx_diff_kine_obj_pos}, minn_diff_kine_obj_pos: {minn_diff_kine_obj_pos}")
    # print(f"maxx_diff_kine_obj_rot_euler: {maxx_diff_kine_obj_rot_euler}, minn_diff_kine_obj_rot_euler: {minn_diff_kine_obj_rot_euler}")
    # print(f"maxx_diff_hand_trans: {maxx_diff_hand_trans}, minn_diff_hand_trans: {minn_diff_hand_trans}")
    
    
    # maxx_diff_kine_obj_pos: [0.00189576 0.0024092  0.00845268], minn_diff_kine_obj_pos: [-0.00199135 -0.01218224 -0.00612274]
    # maxx_diff_kine_obj_rot_euler: [0.028512   0.04186282 0.06846939], minn_diff_kine_obj_rot_euler: [-0.05868825 -0.06077137 -0.03629342]
    # maxx_diff_hand_trans: [0.00316119 0.00237919 0.00523315], minn_diff_hand_trans: [-0.00560641 -0.00521771 -0.00337168]
    
    
    
    
    start_gen_frame = 120
    nn_continuing_frames = 180
    # maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    # minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    # maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    # minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    # minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) # 
    
    # maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    # minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    # maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    # minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    # maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    # minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    # maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
    # minn_obj_rot_diff = np.array(minn_obj_rot_diff, dtype=np.float32) # get the obj rot diff and the maxx obj rot diff #
    # # maxx_hand_trans_diff = [0.00316119 , 0.00237919 , 0.00523315]
    # minn_hand_trans_diff = [-0.00560641 , -0.00521771 , -0.00337168]
    # maxx_hand_trans_diff = np.array(maxx_hand_trans_diff, dtype=np.float32)
    # minn_hand_trans_diff = np.array(minn_hand_trans_diff, dtype=np.float32)
    
    # minn_obj_rot_diff = transformed_delta_rot_xyz_euler
    # maxx_obj_rot_diff = transformed_delta_rot_xyz_euler
    
    # maxx_obj_pos_diff = np.zeros_like(minn_obj_rot_diff)
    # minn_obj_pos_diff = np.zeros_like(minn_obj_rot_diff)
    
    
    maxx_obj_pos_diff = np.zeros((3,), dtype=np.float32)
    minn_obj_pos_diff = np.zeros((3,), dtype=np.float32)
    
    
    nn_samples = 1
    
    tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    
    if hand_type == 'allegro':
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
        
        palm_name = 'palm_link'
        first_tip_name = 'link_3_tip'
        second_tip_name = 'link_7_tip'
        third_tip_name = 'link_11_tip'
        forth_tip_name = 'link_15_tip'
        
        
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            glb_dim = 7
    elif hand_type == 'leap':
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v3.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand.urdf"
            glb_dim = 7
            if w_franka_v2urdf:
                urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand_v2.urdf"
        
        allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
        palm_name = 'palm_lower'
        # first_tip_name = 'thumb_tip_head'
        # second_tip_name = 'index_tip_head'
        # third_tip_name = 'middle_tip_head'
        # forth_tip_name = 'ring_tip_head'
        
        first_tip_name = 'thumb_fingertip'
        second_tip_name = 'fingertip'
        third_tip_name = 'fingertip_2'
        forth_tip_name = 'fingertip_3'
        
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame]
        init_obj_ornt = kine_obj_quat[start_gen_frame] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        # init obj verts #
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        # if hand_type == 'allegro':
        #     if w_arm:
        #         urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
        #     else:
        
        # else:
        #     raise NotImplementedError
        
        # converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0).cuda())
        
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        
        
        # tg_batch # 
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3) # position palm link 
        
        # init_obj_pos #
        init_palm_pos = position_palm_link[0] # (3,)
        init_obj_pos = torch.from_numpy(init_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = init_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors # rotation vector # --- convert the rotation axis to the euler angles #
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) # normalize the vector #
        
        # 
        # tot_perpendicular_vecs = []
        # for ii_perp in range(2):
        #     cur_random_vec = np.random.randn(3)
        #     cur_random_vec = torch.from_numpy(cur_random_vec).float().cuda()
        #     dot_random_vec_w_palm_to_obj_vec = torch.sum(palm_to_obj_vec * cur_random_vec, dim=-1)
        #     proj_random_vec_to_palm_to_obj_vec = cur_random_vec - dot_random_vec_w_palm_to_obj_vec * palm_to_obj_vec # (3,)
        #     proj_random_vec_to_palm_to_obj_vec = proj_random_vec_to_palm_to_obj_vec / torch.clamp( torch.norm(proj_random_vec_to_palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 )
        #     tot_perpendicular_vecs.append(proj_random_vec_to_palm_to_obj_vec)
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        for i_fr in range(nn_continuing_frames): 
            # cur_rand_obj_delta_pos = np.random.uniform(0, 1, size=(3, ))
            cur_rand_obj_delta_pos = tot_delta_pos_rand[sample_idx, i_fr]
            # (maxx obj pos diff - minn obj pos diff)  # 
            # this scaling is not current yet  # from 0, 1 -- scale to the range of minn_obj_pos_diff and maxx_obj_pos_diff
            cur_rand_obj_delta_pos = cur_rand_obj_delta_pos * (maxx_obj_pos_diff - minn_obj_pos_diff) + minn_obj_pos_diff
            # cur_rand_obj_delta_rot = np.random.uniform(0, 1, size=(3, ))
            
            cur_rand_obj_delta_rot_vec = palm_to_obj_vec * delta_mult_coef
            
            # cur_rand_obj_delta_rot_vec = -1.0 * cur_rand_obj_delta_rot_vec
            
            cur_rand_obj_delta_rot_vec = cur_rand_obj_delta_rot_vec.detach().cpu().numpy()
            cur_rand_obj_delta_rot_euler = R.from_rotvec(cur_rand_obj_delta_rot_vec).as_euler('zyx', degrees=False)
            
            # cur_rand_obj_delta_rot = tot_delta_rot_rand[sample_idx, i_fr]
            # cur_rand_obj_delta_rot = cur_rand_obj_delta_rot * (maxx_obj_rot_diff - minn_obj_rot_diff) + minn_obj_rot_diff
            cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + cur_rand_obj_delta_pos
            cur_st_obj_pos = cur_st_obj_pos + cur_rand_obj_delta_pos
            cur_st_obj_rot_euler = cur_st_obj_rot_euler + cur_rand_obj_delta_rot_euler
            cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
            cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
            
            
            continuing_hand_qs.append(cur_st_hand_qs.copy())
            continuing_obj_pos.append(cur_st_obj_pos.copy())
            continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
            
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]): # continuing obj 
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        
        tot_link_idxes = []
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3])
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0)
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()
        
        # palm link verts #
        # tot link idxes #
        
        
        tg_batch_all = chain.forward_kinematics(torch.from_numpy(kine_hand_qs[: start_gen_frame+1]).float().cuda())
        key_name_to_before_start_pose = {}
        for i_link, key in enumerate(allegro_link_names):
            # # print(key, tg_batch[key].shape)
            # link = chain.find_link(key)
            # # print(f"key: {key}, link: {link}")
            # link_visuals = link.visuals
            # print(len(link_visuals))
            key_name_to_before_start_pose[key] = tg_batch_all[key].get_matrix()[:, :3, 3].detach().cpu().numpy().copy()
        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3)
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        # link_key_to_link_pos = {}
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs_glb = robot_qs[..., :glb_dim].detach().clone()
            robot_qs_glb[1:, :glb_dim] = 0.0
            robot_qs = torch.cat(
                [ robot_qs_glb, robot_qs[..., glb_dim:] ], dim=-1
            )
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
                
                link_key_to_link_pos[key] = np.concatenate(
                    [ key_name_to_before_start_pose[key],  link_pos.copy()], axis=0
                )
                # link_pos.copy()
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""

            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # print(f"diff_palm_center: {diff_palm_center.size()}, diff_first_tip_center: {diff_first_tip_center.size()}")
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2, dim=-1).mean()
            loss_first_tip = torch.sum(diff_first_tip_center ** 2, dim=-1).mean()
            loss_second_tip = torch.sum(diff_second_tip_center ** 2, dim=-1).mean()
            loss_third_tip = torch.sum(diff_third_tip_center ** 2, dim=-1).mean()
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2, dim=-1).mean()
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :glb_dim] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip + glb_qs_reg_loss * 100
            print(f"loss: {loss.item()}, loss_palm: {loss_palm.item()}, loss_first_tip: {loss_first_tip.item()}, loss_second_tip: {loss_second_tip.item()}, loss_third_tip: {loss_third_tip.item()}, loss_forth_tip: {loss_forth_tip.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        # kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        # kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        # kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot,
            'robot_delta_states_weights_np': sampled_hand_qs,
            'object_transl': sampled_obj_pos,
            'object_rot_quat': sampled_obj_rot,
            'link_key_to_link_pos': link_key_to_link_pos,
            # 'rotation_axis': -1.0 * palm_to_obj_vec,
            'rotation_axis':  palm_to_obj_vec,
        }
        
        
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        # resampled info sv fn #
        # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        if hand_type == 'leap':
            resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
        else:
            resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")


def traj_modifications_core_v4(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100):
    # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    # kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    # kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # # kine_obj_rot = kine_traj_data['']
    # kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    
    global dtype
    dtype = torch.float32 
    
    interp_frame = 120
    nn_continuing_frames = 180
    start_gen_frame = 120
    
    # obj_rot_quat = kine_obj_quat[interp_frame]
    
    # delta_mult_coef = 0.004 * np.pi
    # delta_mult_coef = 0.008 * np.pi
    
    # delta_xy = 0.002 * np.pi
    # delta_rot_xyz_euler = [delta_xy, delta_xy, 0.0]
    
    
    # delta_xy = 0.004 * np.pi
    # delta_rot_xyz_euler = [0, -delta_xy, -delta_xy]
    
    # delta_rot_xyz_euler = np.array(delta_rot_xyz_euler, dtype=np.float32)
    # obj_rot_matrix = R.from_quat(obj_rot_quat).as_matrix() # .T
    # obj_rot_matrix_delta = R.from_euler('xyz', delta_rot_xyz_euler, degrees=False).as_matrix()
    # transformed_delta_rot_xyz_euler = R.from_matrix(obj_rot_matrix @ obj_rot_matrix_delta).as_euler('zyx', degrees=False)[[2,1,0]] # (3,) obj rot deltas #
    # transformed_delta_rot_xyz_euler = obj_rot_matrix @ delta_rot_xyz_euler
    # transformed_delta_rot_xyz_euler = delta_rot_xyz_euler
    
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    # diff_kine_obj_pos = kine_obj_pos[1:] - kine_obj_pos[:-1] # (nn_frames - 1) x 3 #
    # diff_kine_obj_rot_euler = kine_obj_rot_euler[1:] - kine_obj_rot_euler[:-1] # (nn_frames - 1) x 3 #
    # diff_kine_hand_trans = kine_hand_qs[1:, :3] - kine_hand_qs[:-1, :3] # (nn_frames - 1) x 3 #
    
    
    # new_diff_kine_obj_rot_euler = []
    # for i_fr in range(diff_kine_obj_rot_euler.shape[0]):
    #     cur_diff_kine_obj_rot_euler = diff_kine_obj_rot_euler[i_fr]
    #     if np.any(cur_diff_kine_obj_rot_euler >= np.pi) or np.any(cur_diff_kine_obj_rot_euler <= -np.pi):
    #         continue
    #     new_diff_kine_obj_rot_euler.append(cur_diff_kine_obj_rot_euler)
    
    # new_diff_kine_obj_rot_euler = np.stack(new_diff_kine_obj_rot_euler, axis=0)
    # diff_kine_obj_rot_euler = new_diff_kine_obj_rot_euler
    
    # # diff_kine_obj_rot_euler = np.clip(diff_kine_obj_rot_euler, a_min=-np.pi, a_max=np.pi)
    
    # maxx_diff_kine_obj_pos, minn_diff_kine_obj_pos = np.max(diff_kine_obj_pos, axis=0), np.min(diff_kine_obj_pos, axis=0)
    # maxx_diff_kine_obj_rot_euler, minn_diff_kine_obj_rot_euler = np.max(diff_kine_obj_rot_euler, axis=0), np.min(diff_kine_obj_rot_euler, axis=0)
    # maxx_diff_hand_trans, minn_diff_hand_trans = np.max(diff_kine_hand_trans, axis=0), np.min(diff_kine_hand_trans, axis=0) # 
    # print(f"maxx_diff_kine_obj_pos: {maxx_diff_kine_obj_pos}, minn_diff_kine_obj_pos: {minn_diff_kine_obj_pos}")
    # print(f"maxx_diff_kine_obj_rot_euler: {maxx_diff_kine_obj_rot_euler}, minn_diff_kine_obj_rot_euler: {minn_diff_kine_obj_rot_euler}")
    # print(f"maxx_diff_hand_trans: {maxx_diff_hand_trans}, minn_diff_hand_trans: {minn_diff_hand_trans}")
    
    
    # maxx_diff_kine_obj_pos: [0.00189576 0.0024092  0.00845268], minn_diff_kine_obj_pos: [-0.00199135 -0.01218224 -0.00612274]
    # maxx_diff_kine_obj_rot_euler: [0.028512   0.04186282 0.06846939], minn_diff_kine_obj_rot_euler: [-0.05868825 -0.06077137 -0.03629342]
    # maxx_diff_hand_trans: [0.00316119 0.00237919 0.00523315], minn_diff_hand_trans: [-0.00560641 -0.00521771 -0.00337168]
    
    
    ## add 
    
    # start_gen_frame = 120
    # nn_continuing_frames = 180
    
    start_gen_frame = 100
    nn_continuing_frames = 200
    
    # # 
    
    # maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    # minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    # maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    # minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    # minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) # 
    
    # maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    # minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    # maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    # minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    # maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    # minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    # maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
    # minn_obj_rot_diff = np.array(minn_obj_rot_diff, dtype=np.float32) # get the obj rot diff and the maxx obj rot diff #
    # # maxx_hand_trans_diff = [0.00316119 , 0.00237919 , 0.00523315]
    # minn_hand_trans_diff = [-0.00560641 , -0.00521771 , -0.00337168]
    # maxx_hand_trans_diff = np.array(maxx_hand_trans_diff, dtype=np.float32)
    # minn_hand_trans_diff = np.array(minn_hand_trans_diff, dtype=np.float32)
    
    # minn_obj_rot_diff = transformed_delta_rot_xyz_euler
    # maxx_obj_rot_diff = transformed_delta_rot_xyz_euler
    
    # maxx_obj_pos_diff = np.zeros_like(minn_obj_rot_diff)
    # minn_obj_pos_diff = np.zeros_like(minn_obj_rot_diff)
    
    
    # maxx_obj_pos_diff = np.zeros((3,), dtype=np.float32)
    # minn_obj_pos_diff = np.zeros((3,), dtype=np.float32)
    
    
    # nn_samples = 100
    
    # tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    # tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    
    if hand_type == 'allegro':
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
        
        palm_name = 'palm_link'
        first_tip_name = 'link_3_tip'
        second_tip_name = 'link_7_tip'
        third_tip_name = 'link_11_tip'
        forth_tip_name = 'link_15_tip'
        
        
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            glb_dim = 7
    elif hand_type == 'leap':
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v3.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand.urdf"
            glb_dim = 7
            if w_franka_v2urdf:
                urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand_v2.urdf"
        
        allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
        palm_name = 'palm_lower'
        # first_tip_name = 'thumb_tip_head'
        # second_tip_name = 'index_tip_head'
        # third_tip_name = 'middle_tip_head'
        # forth_tip_name = 'ring_tip_head'
        
        first_tip_name = 'thumb_fingertip'
        second_tip_name = 'fingertip'
        third_tip_name = 'fingertip_2'
        forth_tip_name = 'fingertip_3'      
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame - 1]
        init_obj_ornt = kine_obj_quat[start_gen_frame - 1] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame - 1]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        
        
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3) # position palm link 
        
        # init_obj_pos #
        init_palm_pos = position_palm_link[0] # (3,)
        init_obj_pos = torch.from_numpy(init_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = init_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors # rotation vector # --- convert the rotation axis to the euler angles #
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) # normalize the vector #
        
        # 
        # tot_perpendicular_vecs = []
        # for ii_perp in range(2):
        #     cur_random_vec = np.random.randn(3)
        #     cur_random_vec = torch.from_numpy(cur_random_vec).float().cuda()
        #     dot_random_vec_w_palm_to_obj_vec = torch.sum(palm_to_obj_vec * cur_random_vec, dim=-1)
        #     proj_random_vec_to_palm_to_obj_vec = cur_random_vec - dot_random_vec_w_palm_to_obj_vec * palm_to_obj_vec # (3,)
        #     proj_random_vec_to_palm_to_obj_vec = proj_random_vec_to_palm_to_obj_vec / torch.clamp( torch.norm(proj_random_vec_to_palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 )
        #     tot_perpendicular_vecs.append(proj_random_vec_to_palm_to_obj_vec)
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame - 1].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame - 1].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame - 1].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        maxx_rotation_delta = 0.02 * np.pi
        maxx_trans_delta = 0.01 # 
        nn_fr_per_chunk = 20

        nn_chunks = nn_continuing_frames // nn_fr_per_chunk
        
        for i_chunk in range(nn_chunks):
            rnd_trans_dir = np.random.randn(3)
            if rnd_trans_dir[2] < 0:
                rnd_trans_dir[2] = 0.0001
            rnd_trans_dir = rnd_trans_dir / np.linalg.norm(rnd_trans_dir, ord=2)
            rnd_rot_dir = np.random.randn(3)
            rnd_rot_dir = rnd_rot_dir / np.linalg.norm(rnd_rot_dir, ord=2)
            
            # rnd_delta_trans = rnd_trans_dir *
            rnd_trans_scale = np.random.uniform(low=0.0, high=maxx_trans_delta, size=(3, ))
            rnd_rot_scale = np.random.uniform(low=0.0, high=maxx_rotation_delta, size=(3, ))
            rnd_delta_trans = rnd_trans_dir * rnd_trans_scale
            rnd_delta_rot = rnd_rot_dir * rnd_rot_scale
            
            for i_fr in range(nn_fr_per_chunk):
                cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + rnd_delta_trans
                cur_st_obj_pos = cur_st_obj_pos + rnd_delta_trans
                cur_st_obj_rot_euler = cur_st_obj_rot_euler + rnd_delta_rot
                cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
                cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
                
                continuing_hand_qs.append(cur_st_hand_qs.copy())
                continuing_obj_pos.append(cur_st_obj_pos.copy())
                continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
        
        
        # for i_fr in range(nn_continuing_frames): 
        #     # cur_rand_obj_delta_pos = np.random.uniform(0, 1, size=(3, ))
        #     cur_rand_obj_delta_pos = tot_delta_pos_rand[sample_idx, i_fr]
        #     # (maxx obj pos diff - minn obj pos diff)  # 
        #     # this scaling is not current yet  # from 0, 1 -- scale to the range of minn_obj_pos_diff and maxx_obj_pos_diff
        #     cur_rand_obj_delta_pos = cur_rand_obj_delta_pos * (maxx_obj_pos_diff - minn_obj_pos_diff) + minn_obj_pos_diff
        #     # cur_rand_obj_delta_rot = np.random.uniform(0, 1, size=(3, ))
            
        #     cur_rand_obj_delta_rot_vec = palm_to_obj_vec * delta_mult_coef
            
        #     # cur_rand_obj_delta_rot_vec = -1.0 * cur_rand_obj_delta_rot_vec
            
        #     cur_rand_obj_delta_rot_vec = cur_rand_obj_delta_rot_vec.detach().cpu().numpy()
        #     cur_rand_obj_delta_rot_euler = R.from_rotvec(cur_rand_obj_delta_rot_vec).as_euler('zyx', degrees=False)
            
        #     # cur_rand_obj_delta_rot = tot_delta_rot_rand[sample_idx, i_fr]
        #     # cur_rand_obj_delta_rot = cur_rand_obj_delta_rot * (maxx_obj_rot_diff - minn_obj_rot_diff) + minn_obj_rot_diff
        #     cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + cur_rand_obj_delta_pos
        #     cur_st_obj_pos = cur_st_obj_pos + cur_rand_obj_delta_pos
        #     cur_st_obj_rot_euler = cur_st_obj_rot_euler + cur_rand_obj_delta_rot_euler
        #     cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
        #     cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
            
            
        #     continuing_hand_qs.append(cur_st_hand_qs.copy())
        #     continuing_obj_pos.append(cur_st_obj_pos.copy())
        #     continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
            
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]): # continuing obj 
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        
        tot_link_idxes = []
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3])
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0)
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()
        
        # palm link verts #
        # tot link idxes #
        
        
        tg_batch_all = chain.forward_kinematics(torch.from_numpy(kine_hand_qs[: start_gen_frame+1]).float().cuda())
        key_name_to_before_start_pose = {}
        for i_link, key in enumerate(allegro_link_names):
            # # print(key, tg_batch[key].shape)
            # link = chain.find_link(key)
            # # print(f"key: {key}, link: {link}")
            # link_visuals = link.visuals
            # print(len(link_visuals))
            key_name_to_before_start_pose[key] = tg_batch_all[key].get_matrix()[:, :3, 3].detach().cpu().numpy().copy()
        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        
        # (3,)
        dist_obj_center_to_palm = torch.norm(obj_center_to_palm, p=2, dim=-1, keepdim=False).item()
        dist_filter_threshold = 0.15
        if dist_obj_center_to_palm > dist_filter_threshold:
            print(f"dist_obj_center_to_palm: {dist_obj_center_to_palm}")
            continue
        
        
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3)
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        # link_key_to_link_pos = {}
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs_glb = robot_qs[..., :glb_dim].detach().clone()
            # robot_qs_glb[1:, :glb_dim] = 0.0
            # robot_qs = torch.cat(
            #     [ robot_qs_glb, robot_qs[..., glb_dim:] ], dim=-1
            # )
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
                
                link_key_to_link_pos[key] = np.concatenate(
                    [ key_name_to_before_start_pose[key],  link_pos.copy()], axis=0
                )
                # link_pos.copy()
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""

            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # print(f"diff_palm_center: {diff_palm_center.size()}, diff_first_tip_center: {diff_first_tip_center.size()}")
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2, dim=-1).mean()
            loss_first_tip = torch.sum(diff_first_tip_center ** 2, dim=-1).mean()
            loss_second_tip = torch.sum(diff_second_tip_center ** 2, dim=-1).mean()
            loss_third_tip = torch.sum(diff_third_tip_center ** 2, dim=-1).mean()
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2, dim=-1).mean()
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip # + glb_qs_reg_loss * 100
            print(f"loss: {loss.item()}, loss_palm: {loss_palm.item()}, loss_first_tip: {loss_first_tip.item()}, loss_second_tip: {loss_second_tip.item()}, loss_third_tip: {loss_third_tip.item()}, loss_forth_tip: {loss_forth_tip.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        # kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        # kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        # kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot,
            'robot_delta_states_weights_np': sampled_hand_qs,
            'object_transl': sampled_obj_pos,
            'object_rot_quat': sampled_obj_rot,
            'link_key_to_link_pos': link_key_to_link_pos,
            # 'rotation_axis': -1.0 * palm_to_obj_vec,
            # 'rotation_axis':  palm_to_obj_vec,
        }
        
        
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        
        # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        if hand_type == 'leap':
            # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
            resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        else:
            # resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
            resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")





def traj_modifications_core_v5(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=1, w_interpolate=False, rnd_sample_axis=True, neg_palm_to_obj_vec=False, nn_continuing_frames=180, with_following_frames=False, start_gen_frame=120):
    # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    # kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    # kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # # kine_obj_rot = kine_traj_data['']
    # kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    
    global dtype
    dtype = torch.float32 
    
    # interp_frame = 120
    # nn_continuing_frames = 180
    # start_gen_frame = 120
    
    # obj_rot_quat = kine_obj_quat[interp_frame]
    
    delta_mult_coef = 0.004 * np.pi
    delta_mult_coef = 0.008 * np.pi
    
    
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    
    # start_gen_frame = 120
    # nn_continuing_frames = 180
    
    # nn_continuing_frames = nn_continuing_frames
    
    # maxx_obj_pos_diff = np.zeros((3,), dtype=np.float32)
    # minn_obj_pos_diff = np.zeros((3,), dtype=np.float32)
    
    
    # # nn_samples = 1
    
    # tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    # tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    
    if hand_type == 'allegro':
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
        
        palm_name = 'palm_link'
        first_tip_name = 'link_3_tip'
        second_tip_name = 'link_7_tip'
        third_tip_name = 'link_11_tip'
        forth_tip_name = 'link_15_tip'
        
        
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            glb_dim = 7
    elif hand_type == 'leap':
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v3.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand.urdf"
            glb_dim = 7
            if w_franka_v2urdf:
                urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand_v2.urdf"
        
        allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
        palm_name = 'palm_lower'
        # first_tip_name = 'thumb_tip_head'
        # second_tip_name = 'index_tip_head'
        # third_tip_name = 'middle_tip_head'
        # forth_tip_name = 'ring_tip_head'
        
        # first_tip_name = 'thumb_fingertip'
        # second_tip_name = 'fingertip'
        # third_tip_name = 'fingertip_2'
        # forth_tip_name = 'fingertip_3' 
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    
    # link_name_to_link_index = {
    #     key: i_idx for i_idx, key in enumerate(allegro_link_names)
    # }
    
    
    # mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot_meshes_verts = []
        # tot_link_idxes = []
        
        # link_key_to_vis_mesh = {}
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame]
        init_obj_ornt = kine_obj_quat[start_gen_frame] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        # init obj verts #
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3)
        
        
        init_palm_pos = position_palm_link[0] # (3,)
        init_obj_pos = torch.from_numpy(init_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = init_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) 
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        
        # cur_st_obj_rot_euler
        rnd_obj_rot_vec = np.random.randn(3,)
        rnd_obj_rot_vec = rnd_obj_rot_vec / np.linalg.norm(rnd_obj_rot_vec, ord=2) # (3,)-dim unit length vector
        
        if not rnd_sample_axis:
            ### use the palm to obj vec as the rotation axis ####
            rnd_obj_rot_vec = palm_to_obj_vec.detach().cpu().numpy()
            if neg_palm_to_obj_vec:
                rnd_obj_rot_vec = rnd_obj_rot_vec * (-1.0)
            else:
                rnd_obj_rot_vec = rnd_obj_rot_vec

        continuing_obj_rot = []
        
        if w_interpolate:
            rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
            for i_fr in range(nn_continuing_frames):    
                # cur_fr_obj_rot_vec = rnd_obj_rot_vec * delta_mult_coef # * (i_fr + 1)
                # cur_fr_obj_rot_euler = R.from_rotvec(cur_fr_obj_rot_vec).as_euler('zyx', degrees=False)

                cur_fr_obj_rot_euler = cur_st_obj_rot_euler + rnd_rot_euler * (i_fr + 1)

                # cur_fr_obj_rot_euler = cur_st_obj_rot_euler + (i_fr + 1) * delta_mult_coef
                cur_fr_obj_rot_struct = R.from_euler('zyx', cur_fr_obj_rot_euler, degrees=False)
                cur_st_obj_rot_quat = cur_fr_obj_rot_struct.as_quat()
                continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) # avoid the soft copy #
            
            # continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        else:
            
            rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
            final_rot_euler = rnd_rot_euler * nn_continuing_frames
            final_rot_euler = final_rot_euler + cur_st_obj_rot_euler
            final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
            final_rot_quat = final_rot_struct.as_quat() # final rot quat #
            
            # final_rot_vec = rnd_obj_rot_vec * delta_mult_coef * nn_continuing_frames
            # final_rot_vec_euler = R.from_rotvec(final_rot_vec).as_euler('zyx', degrees=False)
            # final_rot_euler = cur_st_obj_rot_euler + final_rot_vec_euler
            # final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
            # final_rot_quat = final_rot_struct.as_quat() # final rot quat #
            continuing_obj_rot = [ final_rot_quat.copy() for _ in range(nn_continuing_frames) ]
        
        continuing_hand_qs = [ cur_st_hand_qs.copy() for _ in range(nn_continuing_frames) ]
        continuing_obj_pos = [ cur_st_obj_pos.copy() for _ in range(nn_continuing_frames) ]
        # continuing_obj_rot = [ final_rot_quat.copy() for _ in range(nn_continuing_frames) ]
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        # start gen frame # interp after the start gen frame #
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        if with_following_frames:
            following_hand_qs = kine_hand_qs[start_gen_frame + 1: ]
            following_obj_pos = kine_obj_pos[start_gen_frame + 1: ]
            
            # kine_obj_rot_euler
            following_obj_rot_delta_eulers = kine_obj_rot_euler[start_gen_frame + 1: ] - kine_obj_rot_euler[start_gen_frame: -1]
            following_obj_rot = []
            for i_fr in range(following_hand_qs.shape[0]):
                if i_fr == 0:
                    prev_obj_rot_quat = continuing_obj_rot[-1].copy()
                else:
                    prev_obj_rot_quat = following_obj_rot[-1].copy()
                prev_obj_rot_euler = R.from_quat(prev_obj_rot_quat).as_euler('zyx', degrees=False)
                cur_obj_rot_euler = prev_obj_rot_euler + following_obj_rot_delta_eulers[i_fr]
                cur_obj_rot_struct = R.from_euler('zyx', cur_obj_rot_euler, degrees=False)
                cur_obj_rot_quat = cur_obj_rot_struct.as_quat()
                following_obj_rot.append(cur_obj_rot_quat.copy())
            # 
            # 
            # following_obj_rot = [ continuing_obj_rot[-1] for _ in range(following_hand_qs.shape[0]) ]
            following_obj_rot = np.stack(following_obj_rot, axis=0)
            sampled_hand_qs = np.concatenate([sampled_hand_qs, following_hand_qs], axis=0)
            sampled_obj_pos = np.concatenate([sampled_obj_pos, following_obj_pos], axis=0)
            sampled_obj_rot = np.concatenate([sampled_obj_rot, following_obj_rot], axis=0)
        
        # tg batch = chain #
        # tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        sampled_hand_qs_th = torch.from_numpy(sampled_hand_qs).float().cuda()
        tg_batch = chain.forward_kinematics(sampled_hand_qs_th) # 
        
        
        for i_link, key in enumerate(allegro_link_names):
            link = chain.find_link(key)
            link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
            
            link_key_to_link_pos[key] = link_pos.copy()
        
        print(f"Info to save -- sampled_hand_qs: {sampled_hand_qs.shape}, sampled_obj_pos: {sampled_obj_pos.shape}, sampled_obj_rot: {sampled_obj_rot.shape}")
        
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot,
            'robot_delta_states_weights_np': sampled_hand_qs,
            'object_transl': sampled_obj_pos,
            'object_rot_quat': sampled_obj_rot,
            'link_key_to_link_pos': link_key_to_link_pos,
            # 'rotation_axis': -1.0 * palm_to_obj_vec,
            'rotation_axis':  rnd_obj_rot_vec,
        }
        
        # change the maximum length so that we can make it to track the result #
        
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        if nn_samples == 1:
            if hand_type == 'leap':
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
            else:
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        else:
            if hand_type == 'leap':
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            else:
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")



def traj_modifications_core_v6(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100, start_gen_frame=100, nn_continuing_frames=200, nn_continuing_frames_s2=150, rnd_sample_axis=False, neg_palm_to_obj_vec=False):
    
    
    global dtype
    dtype = torch.float32 
    
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    
    if hand_type == 'allegro':
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
        
        palm_name = 'palm_link'
        first_tip_name = 'link_3_tip'
        second_tip_name = 'link_7_tip'
        third_tip_name = 'link_11_tip'
        forth_tip_name = 'link_15_tip'
        
        
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            glb_dim = 7
    elif hand_type == 'leap':
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v3.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand.urdf"
            glb_dim = 7
            if w_franka_v2urdf:
                urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand_v2.urdf"
        
        allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
        palm_name = 'palm_lower'
        # first_tip_name = 'thumb_tip_head'
        # second_tip_name = 'index_tip_head'
        # third_tip_name = 'middle_tip_head'
        # forth_tip_name = 'ring_tip_head'
        
        first_tip_name = 'thumb_fingertip'
        second_tip_name = 'fingertip'
        third_tip_name = 'fingertip_2'
        forth_tip_name = 'fingertip_3'      
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame - 1]
        init_obj_ornt = kine_obj_quat[start_gen_frame - 1] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame - 1]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        
        
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3) # position palm link 
        
        # init_obj_pos #
        init_palm_pos = position_palm_link[0] # (3,)
        init_obj_pos = torch.from_numpy(init_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = init_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors # rotation vector # --- convert the rotation axis to the euler angles #
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) # normalize the vector #
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame - 1].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame - 1].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame - 1].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        maxx_rotation_delta = 0.02 * np.pi
        maxx_trans_delta = 0.01 # 
        
        
        nn_fr_per_chunk = 20

        nn_chunks = nn_continuing_frames // nn_fr_per_chunk
        
        for i_chunk in range(nn_chunks):
            rnd_trans_dir = np.random.randn(3)
            if rnd_trans_dir[2] < 0:
                rnd_trans_dir[2] = 0.0001
            rnd_trans_dir = rnd_trans_dir / np.linalg.norm(rnd_trans_dir, ord=2)
            rnd_rot_dir = np.random.randn(3)
            rnd_rot_dir = rnd_rot_dir / np.linalg.norm(rnd_rot_dir, ord=2)
            
            # rnd_delta_trans = rnd_trans_dir *
            rnd_trans_scale = np.random.uniform(low=0.0, high=maxx_trans_delta, size=(3, ))
            rnd_rot_scale = np.random.uniform(low=0.0, high=maxx_rotation_delta, size=(3, ))
            rnd_delta_trans = rnd_trans_dir * rnd_trans_scale
            rnd_delta_rot = rnd_rot_dir * rnd_rot_scale
            
            for i_fr in range(nn_fr_per_chunk):
                cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + rnd_delta_trans
                cur_st_obj_pos = cur_st_obj_pos + rnd_delta_trans
                cur_st_obj_rot_euler = cur_st_obj_rot_euler + rnd_delta_rot
                cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
                cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
                
                continuing_hand_qs.append(cur_st_hand_qs.copy())
                continuing_obj_pos.append(cur_st_obj_pos.copy())
                continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
        
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]): # continuing obj 
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        
        tot_link_idxes = []
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3])
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0)
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()
        
        # palm link verts #
        # tot link idxes #
        
        
        tg_batch_all = chain.forward_kinematics(torch.from_numpy(kine_hand_qs[: start_gen_frame+1]).float().cuda())
        key_name_to_before_start_pose = {}
        for i_link, key in enumerate(allegro_link_names):
            key_name_to_before_start_pose[key] = tg_batch_all[key].get_matrix()[:, :3, 3].detach().cpu().numpy().copy()
        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        
        # (3,)
        dist_obj_center_to_palm = torch.norm(obj_center_to_palm, p=2, dim=-1, keepdim=False).item()
        dist_filter_threshold = 0.15
        if dist_obj_center_to_palm > dist_filter_threshold:
            print(f"dist_obj_center_to_palm: {dist_obj_center_to_palm}")
            continue
        
        
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3)
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        # link_key_to_link_pos = {}
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
                
                link_key_to_link_pos[key] = np.concatenate(
                    [ key_name_to_before_start_pose[key],  link_pos.copy()], axis=0
                )
                # link_pos.copy()
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""

            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # print(f"diff_palm_center: {diff_palm_center.size()}, diff_first_tip_center: {diff_first_tip_center.size()}")
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2, dim=-1).mean()
            loss_first_tip = torch.sum(diff_first_tip_center ** 2, dim=-1).mean()
            loss_second_tip = torch.sum(diff_second_tip_center ** 2, dim=-1).mean()
            loss_third_tip = torch.sum(diff_third_tip_center ** 2, dim=-1).mean()
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2, dim=-1).mean()
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip # + glb_qs_reg_loss * 100
            print(f"loss: {loss.item()}, loss_palm: {loss_palm.item()}, loss_first_tip: {loss_first_tip.item()}, loss_second_tip: {loss_second_tip.item()}, loss_third_tip: {loss_third_tip.item()}, loss_forth_tip: {loss_forth_tip.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        last_hand_qs = sampled_hand_qs[-1]
        last_obj_pos = sampled_obj_pos[-1]
        last_hand_qs_th = torch.from_numpy(last_hand_qs).float().to(d)
        tg_batch_last_frame = chain.forward_kinematics(last_hand_qs_th.unsqueeze(0)) # last hand qs #
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch_last_frame[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3)
        
        
        # get the plam link #
        init_palm_pos = position_palm_link[0] # (3,)
        last_obj_pos = torch.from_numpy(last_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = last_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) 
        
        
        last_hand_qs = sampled_hand_qs[-1].copy()
        last_obj_pos = sampled_obj_pos[-1].copy()
        last_obj_rot = sampled_obj_rot[-1].copy()
        
        last_obj_rot_euler = R.from_quat(last_obj_rot).as_euler('zyx', degrees=False)
        
        delta_mult_coef = 0.008 * np.pi
        
        # cur_st_obj_rot_euler
        rnd_obj_rot_vec = np.random.randn(3,)
        rnd_obj_rot_vec = rnd_obj_rot_vec / np.linalg.norm(rnd_obj_rot_vec, ord=2) # (3,)-dim unit length vector
        
        if not rnd_sample_axis:
            ### use the palm to obj vec as the rotation axis ####
            rnd_obj_rot_vec = palm_to_obj_vec.detach().cpu().numpy()
            if neg_palm_to_obj_vec:
                rnd_obj_rot_vec = rnd_obj_rot_vec * (-1.0)
            else:
                rnd_obj_rot_vec = rnd_obj_rot_vec
        
        
        nex_stage_obj_rot = []
        
        # if w_interpolate:
        rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
        for i_fr in range(nn_continuing_frames_s2): 
            cur_fr_obj_rot_euler = last_obj_rot_euler + rnd_rot_euler * (i_fr + 1)
            cur_fr_obj_rot_struct = R.from_euler('zyx', cur_fr_obj_rot_euler, degrees=False)
            cur_fr_obj_rot_quat = cur_fr_obj_rot_struct.as_quat()
            nex_stage_obj_rot.append(cur_fr_obj_rot_quat.copy()) # avoid the soft copy #
        # continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        # else:
            
        #     rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
        #     final_rot_euler = rnd_rot_euler * nn_continuing_frames
        #     final_rot_euler = final_rot_euler + cur_st_obj_rot_euler
        #     final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
        #     final_rot_quat = final_rot_struct.as_quat() # final rot quat #
            
        #     # final_rot_vec = rnd_obj_rot_vec * delta_mult_coef * nn_continuing_frames
        #     # final_rot_vec_euler = R.from_rotvec(final_rot_vec).as_euler('zyx', degrees=False)
        #     # final_rot_euler = cur_st_obj_rot_euler + final_rot_vec_euler
        #     # final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
        #     # final_rot_quat = final_rot_struct.as_quat() # final rot quat #
        #     continuing_obj_rot = [ final_rot_quat.copy() for _ in range(nn_continuing_frames) ]
        
        nex_stage_hand_qs = [
            last_hand_qs.copy() for _ in range(nn_continuing_frames_s2)
        ]
        nex_stage_obj_pos = [
            last_obj_pos.copy() for _ in range(nn_continuing_frames_s2)
        ]
        nex_stage_hand_qs = np.stack(nex_stage_hand_qs, axis=0)
        nex_stage_obj_pos = np.stack(nex_stage_obj_pos, axis=0)
        nex_stage_obj_rot = np.stack(nex_stage_obj_rot, axis=0)
        
        sampled_hand_qs = np.concatenate([sampled_hand_qs, nex_stage_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([sampled_obj_pos, nex_stage_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([sampled_obj_rot, nex_stage_obj_rot], axis=0)
        
        
        sampled_hand_qs_th = torch.from_numpy(sampled_hand_qs).float().cuda()
        tg_batch = chain.forward_kinematics(sampled_hand_qs_th) # 
        
        link_key_to_link_pos = {}
        
        for i_link, key in enumerate(allegro_link_names):
            link = chain.find_link(key)
            link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
            link_key_to_link_pos[key] = link_pos.copy()
        
        
        # kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        # kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        # kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot,
            'robot_delta_states_weights_np': sampled_hand_qs,
            'object_transl': sampled_obj_pos,
            'object_rot_quat': sampled_obj_rot,
            'link_key_to_link_pos': link_key_to_link_pos,
            # 'rotation_axis': -1.0 * palm_to_obj_vec,
            # 'rotation_axis':  palm_to_obj_vec,
        }
        
        
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        
        
        if nn_samples == 1:
            if hand_type == 'leap':
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
            else:
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        else:
            # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            if hand_type == 'leap':
                # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            else:
                # resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")






def traj_modifications_core_v7(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100, start_gen_frame=100, nn_continuing_frames=200, nn_continuing_frames_s2=150, rnd_sample_axis=False, neg_palm_to_obj_vec=False):
    
    
    global dtype
    dtype = torch.float32 
    
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    
    if hand_type == 'allegro':
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
        
        palm_name = 'palm_link'
        first_tip_name = 'link_3_tip'
        second_tip_name = 'link_7_tip'
        third_tip_name = 'link_11_tip'
        forth_tip_name = 'link_15_tip'
        
        
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            glb_dim = 7
    elif hand_type == 'leap':
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v3.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand.urdf"
            glb_dim = 7
            if w_franka_v2urdf:
                urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand_v2.urdf"
        
        allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
        palm_name = 'palm_lower'
        # first_tip_name = 'thumb_tip_head'
        # second_tip_name = 'index_tip_head'
        # third_tip_name = 'middle_tip_head'
        # forth_tip_name = 'ring_tip_head'
        
        first_tip_name = 'thumb_fingertip'
        second_tip_name = 'fingertip'
        third_tip_name = 'fingertip_2'
        forth_tip_name = 'fingertip_3'      
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame - 1]
        init_obj_ornt = kine_obj_quat[start_gen_frame - 1] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame - 1]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        
        
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3) # position palm link 
        
        # init_obj_pos #
        init_palm_pos = position_palm_link[0] # (3,)
        init_obj_pos = torch.from_numpy(init_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = init_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors # rotation vector # --- convert the rotation axis to the euler angles #
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) # normalize the vector #
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame - 1].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame - 1].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame - 1].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        maxx_rotation_delta = 0.02 * np.pi
        maxx_trans_delta = 0.01 # 
        
        
        nn_fr_per_chunk = 20

        nn_chunks = nn_continuing_frames // nn_fr_per_chunk
        
        for i_chunk in range(nn_chunks):
            rnd_trans_dir = np.random.randn(3)
            if rnd_trans_dir[2] < 0:
                rnd_trans_dir[2] = 0.0001
            rnd_trans_dir = rnd_trans_dir / np.linalg.norm(rnd_trans_dir, ord=2)
            rnd_rot_dir = np.random.randn(3)
            rnd_rot_dir = rnd_rot_dir / np.linalg.norm(rnd_rot_dir, ord=2)
            
            # rnd_delta_trans = rnd_trans_dir *
            rnd_trans_scale = np.random.uniform(low=0.0, high=maxx_trans_delta, size=(3, ))
            rnd_rot_scale = np.random.uniform(low=0.0, high=maxx_rotation_delta, size=(3, ))
            rnd_delta_trans = rnd_trans_dir * rnd_trans_scale
            rnd_delta_rot = rnd_rot_dir * rnd_rot_scale
            
            for i_fr in range(nn_fr_per_chunk):
                cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + rnd_delta_trans
                cur_st_obj_pos = cur_st_obj_pos + rnd_delta_trans
                cur_st_obj_rot_euler = cur_st_obj_rot_euler + rnd_delta_rot
                cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
                cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
                
                continuing_hand_qs.append(cur_st_hand_qs.copy())
                continuing_obj_pos.append(cur_st_obj_pos.copy())
                continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
        
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]): # continuing obj 
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        
        tot_link_idxes = []
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3])
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0)
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()
        

        
        tg_batch_all = chain.forward_kinematics(torch.from_numpy(kine_hand_qs[: start_gen_frame+1]).float().cuda())
        key_name_to_before_start_pose = {}
        for i_link, key in enumerate(allegro_link_names):
            key_name_to_before_start_pose[key] = tg_batch_all[key].get_matrix()[:, :3, 3].detach().cpu().numpy().copy()
        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        
        # (3,)
        dist_obj_center_to_palm = torch.norm(obj_center_to_palm, p=2, dim=-1, keepdim=False).item()
        dist_filter_threshold = 0.15
        if dist_obj_center_to_palm > dist_filter_threshold:
            print(f"dist_obj_center_to_palm: {dist_obj_center_to_palm}")
            continue
        
        
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3)
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        
        
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
                
                link_key_to_link_pos[key] = np.concatenate(
                    [ key_name_to_before_start_pose[key],  link_pos.copy()], axis=0
                )
                # link_pos.copy()
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""

            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # print(f"diff_palm_center: {diff_palm_center.size()}, diff_first_tip_center: {diff_first_tip_center.size()}")
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2, dim=-1).mean()
            loss_first_tip = torch.sum(diff_first_tip_center ** 2, dim=-1).mean()
            loss_second_tip = torch.sum(diff_second_tip_center ** 2, dim=-1).mean()
            loss_third_tip = torch.sum(diff_third_tip_center ** 2, dim=-1).mean()
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2, dim=-1).mean()
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip # + glb_qs_reg_loss * 100
            print(f"loss: {loss.item()}, loss_palm: {loss_palm.item()}, loss_first_tip: {loss_first_tip.item()}, loss_second_tip: {loss_second_tip.item()}, loss_third_tip: {loss_third_tip.item()}, loss_forth_tip: {loss_forth_tip.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        
        # ####### reorientation stage #######
        # last_hand_qs = sampled_hand_qs[-1]
        # last_obj_pos = sampled_obj_pos[-1]
        # last_hand_qs_th = torch.from_numpy(last_hand_qs).float().to(d)
        # tg_batch_last_frame = chain.forward_kinematics(last_hand_qs_th.unsqueeze(0)) # last hand qs #
        
        # palm_link_key  =  palm_name # 'palm_link'
        # transformation_matrix_palm_link = tg_batch_last_frame[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        # position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3)
        
        
        # # get the plam link #
        # init_palm_pos = position_palm_link[0] # (3,)
        # last_obj_pos = torch.from_numpy(last_obj_pos).float().cuda() # (3,) 
        # palm_to_obj_vec = last_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors
        # palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) 
        
        
        # last_hand_qs = sampled_hand_qs[-1].copy()
        # last_obj_pos = sampled_obj_pos[-1].copy()
        # last_obj_rot = sampled_obj_rot[-1].copy()
        
        # last_obj_rot_euler = R.from_quat(last_obj_rot).as_euler('zyx', degrees=False)
        
        # delta_mult_coef = 0.008 * np.pi
        
        # # cur_st_obj_rot_euler
        # rnd_obj_rot_vec = np.random.randn(3,)
        # rnd_obj_rot_vec = rnd_obj_rot_vec / np.linalg.norm(rnd_obj_rot_vec, ord=2) # (3,)-dim unit length vector
        
        # if not rnd_sample_axis:
        #     ### use the palm to obj vec as the rotation axis ####
        #     rnd_obj_rot_vec = palm_to_obj_vec.detach().cpu().numpy()
        #     if neg_palm_to_obj_vec:
        #         rnd_obj_rot_vec = rnd_obj_rot_vec * (-1.0)
        #     else:
        #         rnd_obj_rot_vec = rnd_obj_rot_vec
        
        
        # nex_stage_obj_rot = []
        
        # # if w_interpolate:
        # rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
        # for i_fr in range(nn_continuing_frames_s2): 
        #     cur_fr_obj_rot_euler = last_obj_rot_euler + rnd_rot_euler * (i_fr + 1)
        #     cur_fr_obj_rot_struct = R.from_euler('zyx', cur_fr_obj_rot_euler, degrees=False)
        #     cur_fr_obj_rot_quat = cur_fr_obj_rot_struct.as_quat()
        #     nex_stage_obj_rot.append(cur_fr_obj_rot_quat.copy()) # avoid the soft copy #
        # # continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        # # else:
            
        # #     rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
        # #     final_rot_euler = rnd_rot_euler * nn_continuing_frames
        # #     final_rot_euler = final_rot_euler + cur_st_obj_rot_euler
        # #     final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
        # #     final_rot_quat = final_rot_struct.as_quat() # final rot quat #
            
        # #     # final_rot_vec = rnd_obj_rot_vec * delta_mult_coef * nn_continuing_frames
        # #     # final_rot_vec_euler = R.from_rotvec(final_rot_vec).as_euler('zyx', degrees=False)
        # #     # final_rot_euler = cur_st_obj_rot_euler + final_rot_vec_euler
        # #     # final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
        # #     # final_rot_quat = final_rot_struct.as_quat() # final rot quat #
        # #     continuing_obj_rot = [ final_rot_quat.copy() for _ in range(nn_continuing_frames) ]
        
        # nex_stage_hand_qs = [
        #     last_hand_qs.copy() for _ in range(nn_continuing_frames_s2)
        # ]
        # nex_stage_obj_pos = [
        #     last_obj_pos.copy() for _ in range(nn_continuing_frames_s2)
        # ]
        # nex_stage_hand_qs = np.stack(nex_stage_hand_qs, axis=0)
        # nex_stage_obj_pos = np.stack(nex_stage_obj_pos, axis=0)
        # nex_stage_obj_rot = np.stack(nex_stage_obj_rot, axis=0)
        
        # sampled_hand_qs = np.concatenate([sampled_hand_qs, nex_stage_hand_qs], axis=0)
        # sampled_obj_pos = np.concatenate([sampled_obj_pos, nex_stage_obj_pos], axis=0)
        # sampled_obj_rot = np.concatenate([sampled_obj_rot, nex_stage_obj_rot], axis=0)
        # ####### reorientation stage #######
        
        
        
        
        sampled_hand_qs_th = torch.from_numpy(sampled_hand_qs).float().cuda()
        tg_batch = chain.forward_kinematics(sampled_hand_qs_th)
        
        link_key_to_link_pos = {}
        
        for i_link, key in enumerate(allegro_link_names):
            link = chain.find_link(key)
            link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
            link_key_to_link_pos[key] = link_pos.copy()
        
        
        # kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs 
        # kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 
        # kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot,
            'robot_delta_states_weights_np': sampled_hand_qs,
            'object_transl': sampled_obj_pos,
            'object_rot_quat': sampled_obj_rot,
            'link_key_to_link_pos': link_key_to_link_pos,
            # 'rotation_axis': -1.0 * palm_to_obj_vec,
            # 'rotation_axis':  palm_to_obj_vec,
        }
        
        
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        
        
        if nn_samples == 1:
            if hand_type == 'leap':
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
            else:
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        else:
            # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            if hand_type == 'leap':
                # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            else:
                # resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")


def traj_modifications_core_v8(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100, start_gen_frame=100, nn_continuing_frames=200, nn_continuing_frames_s2=150, nn_continuing_frames_s3=60, rnd_sample_axis=False, neg_palm_to_obj_vec=False):
    
    
    global dtype
    dtype = torch.float32
    
    # 
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    
    
    obj_mesh_verts = load_obj_mesh(inst_tag) # (nn_verts x 3) #
    sampled_obj_verts_idxes = np.random.permutation(obj_mesh_verts.shape[0])[:512]
    obj_mesh_verts = obj_mesh_verts[sampled_obj_verts_idxes]
    
    
    if hand_type == 'allegro':
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
        
        palm_name = 'palm_link'
        first_tip_name = 'link_3_tip'
        second_tip_name = 'link_7_tip'
        third_tip_name = 'link_11_tip'
        forth_tip_name = 'link_15_tip'
        
        
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
            glb_dim = 7
    elif hand_type == 'leap':
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v3.urdf'
        
        glb_dim = 6
        
        if w_franka:
            urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand.urdf"
            glb_dim = 7
            if w_franka_v2urdf:
                urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/leap_hand/franka_panda_leaphand_v2.urdf"
        
        allegro_link_names = ['fingertip', 'dip', 'pip', 'mcp_joint', 'fingertip_2', 'dip_2', 'pip_2', 'mcp_joint_2', 'fingertip_3', 'dip_3', 'pip_3', 'mcp_joint_3', 'thumb_fingertip', 'thumb_dip', 'thumb_pip', 'thumb_temp_base', 'palm_lower'] + \
                ['thumb_tip_head', 'index_tip_head', 'middle_tip_head', 'ring_tip_head']
        palm_name = 'palm_lower'
        # first_tip_name = 'thumb_tip_head'
        # second_tip_name = 'index_tip_head'
        # third_tip_name = 'middle_tip_head'
        # forth_tip_name = 'ring_tip_head'
        
        first_tip_name = 'thumb_fingertip'
        second_tip_name = 'fingertip'
        third_tip_name = 'fingertip_2'
        forth_tip_name = 'fingertip_3'      
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    
    
    link_name_to_link_index = {
        key: i_idx for i_idx, key in enumerate(allegro_link_names)
    }
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    
    
    for sample_idx in range(0, nn_samples):
        
        # tot meshes verts #
        tot_meshes_verts = []
        tot_link_idxes = []
        
        link_key_to_vis_mesh = {}
        
        
        ### calculating the hand qs, obj pos, obj quat ###
        init_obj_pos = kine_obj_pos[start_gen_frame - 1]
        init_obj_ornt = kine_obj_quat[start_gen_frame - 1] # (4, ) object orientation
        init_hand_qs = kine_hand_qs[start_gen_frame - 1]
        
        init_rot_struct = R.from_quat(init_obj_ornt)
        init_rot_matrix= init_rot_struct.as_matrix()
        init_obj_verts = (init_rot_matrix @ obj_mesh_verts.T).T + init_obj_pos[None]
        
        
        init_hand_qs = torch.from_numpy(init_hand_qs).float() # (nn_hand_dofs, )
        
        
        # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
        tg_batch = chain.forward_kinematics(init_hand_qs.unsqueeze(0).cuda())
        
        
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3) # position palm link 
        
        # init_obj_pos #
        init_palm_pos = position_palm_link[0] # (3,)
        init_obj_pos = torch.from_numpy(init_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = init_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors # rotation vector # --- convert the rotation axis to the euler angles #
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) # normalize the vector #
        
        
        
        cur_st_hand_qs = kine_hand_qs[start_gen_frame - 1].copy()
        cur_st_obj_pos = kine_obj_pos[start_gen_frame - 1].copy()
        cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame - 1].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        maxx_rotation_delta = 0.02 * np.pi
        maxx_trans_delta = 0.01 # 
        
        
        nn_fr_per_chunk = 20

        nn_chunks = nn_continuing_frames // nn_fr_per_chunk
        
        for i_chunk in range(nn_chunks):
            rnd_trans_dir = np.random.randn(3)
            if rnd_trans_dir[2] < 0:
                rnd_trans_dir[2] = 0.0001
            rnd_trans_dir = rnd_trans_dir / np.linalg.norm(rnd_trans_dir, ord=2)
            rnd_rot_dir = np.random.randn(3)
            rnd_rot_dir = rnd_rot_dir / np.linalg.norm(rnd_rot_dir, ord=2)
            
            # rnd_delta_trans = rnd_trans_dir *
            rnd_trans_scale = np.random.uniform(low=0.0, high=maxx_trans_delta, size=(3, ))
            rnd_rot_scale = np.random.uniform(low=0.0, high=maxx_rotation_delta, size=(3, ))
            rnd_delta_trans = rnd_trans_dir * rnd_trans_scale
            rnd_delta_rot = rnd_rot_dir * rnd_rot_scale
            
            for i_fr in range(nn_fr_per_chunk):
                cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + rnd_delta_trans
                cur_st_obj_pos = cur_st_obj_pos + rnd_delta_trans
                cur_st_obj_rot_euler = cur_st_obj_rot_euler + rnd_delta_rot
                cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
                cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
                
                continuing_hand_qs.append(cur_st_hand_qs.copy())
                continuing_obj_pos.append(cur_st_obj_pos.copy())
                continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
        
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        continuing_obj_verts = []
        continuing_obj_rot_matrix = []
        for i_fr in range(continuing_obj_pos.shape[0]): # continuing obj 
            cur_fr_obj_rot_matrix = R.from_quat(continuing_obj_rot[i_fr]).as_matrix()
            cur_fr_obj_verts = (cur_fr_obj_rot_matrix @ obj_mesh_verts.T).T + continuing_obj_pos[i_fr][None]
            continuing_obj_verts.append(cur_fr_obj_verts)
            continuing_obj_rot_matrix.append(cur_fr_obj_rot_matrix)
        continuing_obj_verts = np.stack(continuing_obj_verts, axis=0)
        continuing_obj_verts = torch.from_numpy(continuing_obj_verts).float()
        continuing_obj_rot_matrix = np.stack(continuing_obj_rot_matrix, axis=0)
        continuing_obj_rot_matrix = torch.from_numpy(continuing_obj_rot_matrix).float()
        
        
        
        tot_link_idxes = []
        for i_link, key in enumerate(allegro_link_names):
            # print(key, tg_batch[key].shape)
            link = chain.find_link(key)
            # print(f"key: {key}, link: {link}")
            link_visuals = link.visuals
            # print(len(link_visuals))
            
            for cur_visual in link_visuals:
                m_offset = cur_visual.offset # .get_matrix() #
                cur_visual_mesh_fn = cur_visual.geom_param
                m = tg_batch[key].get_matrix()
                pos = m[:, :3, 3].float()
                # rot = pk.matrix_to_quaternion(m[:, :3, :3])
                rot = m[:, :3, :3].float()
                # pos = pos[i_ts]
                # rot = rot[i_ts]
                if cur_visual_mesh_fn is None:
                    continue
                if isinstance(cur_visual_mesh_fn, tuple):
                    cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
                if key not in link_key_to_vis_mesh:
                    link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    verts = link_key_to_vis_mesh[key]
                    # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                    
                    verts = torch.from_numpy(verts).float().to(d)
                    # if to_numpy:
                    verts = verts.cpu()
                    verts = m_offset.transform_points(verts)
                    
                    # if not to_numpy:
                    verts = verts.to(d) 
                    
                    link_key_to_vis_mesh[key] = verts
                
                
                verts = link_key_to_vis_mesh[key]
                
                
                transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                # print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
                
                # visual link idxes #
                cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(1), dtype=np.int32)
                cur_visual_link_idxes = torch.from_numpy(cur_visual_link_idxes).long().to(d)
                tot_link_idxes.append(cur_visual_link_idxes)
        tot_meshes_verts = torch.cat(tot_meshes_verts, dim=1)
        tot_link_idxes = torch.cat(tot_link_idxes, dim=0)
        
        
        converted_hand_verts = tot_meshes_verts
        tot_ts_tot_link_idxes = tot_link_idxes.unsqueeze(0)
        
        
        
        if isinstance(converted_hand_verts, np.ndarray):
            converted_hand_verts = torch.from_numpy(converted_hand_verts).float()[0] # .cuda()
        else:
            converted_hand_verts = converted_hand_verts[0].detach().cpu()
        if isinstance(tot_ts_tot_link_idxes, np.ndarray):
            tot_ts_tot_link_idxes = torch.from_numpy(tot_ts_tot_link_idxes).long()[0] # .cuda()
        else:
            tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0].detach().cpu()
        

        
        tg_batch_all = chain.forward_kinematics(torch.from_numpy(kine_hand_qs[: start_gen_frame+1]).float().cuda())
        key_name_to_before_start_pose = {}
        for i_link, key in enumerate(allegro_link_names):
            key_name_to_before_start_pose[key] = tg_batch_all[key].get_matrix()[:, :3, 3].detach().cpu().numpy().copy()
        
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes.clone() #  torch.from_numpy(tot_ts_tot_link_idxes).long()
        palm_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[palm_name]]
        first_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[first_tip_name]]
        second_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[second_tip_name]]
        third_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[third_tip_name]]
        forth_tip_link_verts = converted_hand_verts[tot_ts_tot_link_idxes == link_name_to_link_index[forth_tip_name]]
        
        init_obj_verts = torch.from_numpy(init_obj_verts).float() # (nn_Verts, 3)
        
        center_palm = torch.mean(palm_link_verts, dim=0)
        center_first_tip = torch.mean(first_tip_link_verts, dim=0)
        center_second_tip = torch.mean(second_tip_link_verts, dim=0)
        center_third_tip = torch.mean(third_tip_link_verts, dim=0)
        center_forth_tip = torch.mean(forth_tip_link_verts, dim=0)
        
        center_obj_verts = torch.mean(init_obj_verts, dim=0) # (3,)
        
        obj_center_to_palm = center_palm - center_obj_verts
        
        # (3,)
        dist_obj_center_to_palm = torch.norm(obj_center_to_palm, p=2, dim=-1, keepdim=False).item()
        dist_filter_threshold = 0.15
        if dist_obj_center_to_palm > dist_filter_threshold:
            print(f"dist_obj_center_to_palm: {dist_obj_center_to_palm}")
            continue
        
        
        obj_center_to_first_tip = center_first_tip - center_obj_verts
        obj_center_to_second_tip = center_second_tip - center_obj_verts
        obj_center_to_third_tip = center_third_tip - center_obj_verts
        obj_center_to_forth_tip = center_forth_tip - center_obj_verts
        
        
        init_rot_matrix_th = torch.from_numpy(init_rot_matrix).float() # (3, 3)
        canon_obj_center_to_palm = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_palm.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_first_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_first_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_second_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_second_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_third_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_third_tip.unsqueeze(-1)
        ).squeeze(-1)
        canon_obj_center_to_forth_tip = torch.matmul(
            init_rot_matrix_th.contiguous().transpose(1, 0).contiguous(), obj_center_to_forth_tip.unsqueeze(-1)
        ).squeeze(-1)
        
        continuing_obj_center_to_palm = torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_palm.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_first_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_first_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_second_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_second_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_third_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_third_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        continuing_obj_center_to_forth_tip =  torch.matmul(
            continuing_obj_rot_matrix, canon_obj_center_to_forth_tip.unsqueeze(0).unsqueeze(-1)
        ).squeeze(-1)
        
        
        continuing_obj_verts_center = continuing_obj_verts.mean(dim=1) # nn_ts x 3 
        target_palm_center = continuing_obj_verts_center + continuing_obj_center_to_palm
        target_first_tip_center = continuing_obj_verts_center + continuing_obj_center_to_first_tip
        target_second_tip_center = continuing_obj_verts_center + continuing_obj_center_to_second_tip
        target_third_tip_center = continuing_obj_verts_center + continuing_obj_center_to_third_tip
        target_forth_tip_center = continuing_obj_verts_center + continuing_obj_center_to_forth_tip
        
        target_palm_center = target_palm_center.to(d)
        target_first_tip_center = target_first_tip_center.to(d)
        target_second_tip_center = target_second_tip_center.to(d)
        target_third_tip_center = target_third_tip_center.to(d)
        target_forth_tip_center = target_forth_tip_center.to(d)
        
        
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 100
        
        
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        
        
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            # for key in tg_batch.keys():
            for i_link, key in enumerate(allegro_link_names):
                # print(key, tg_batch[key].shape)
                link = chain.find_link(key)
                # print(f"key: {key}, link: {link}")
                link_visuals = link.visuals
                # print(len(link_visuals))
                
                link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
                
                link_key_to_link_pos[key] = np.concatenate(
                    [ key_name_to_before_start_pose[key],  link_pos.copy()], axis=0
                )
                # link_pos.copy()
                
                for cur_visual in link_visuals:
                    m_offset = cur_visual.offset # .get_matrix() #
                    cur_visual_mesh_fn = cur_visual.geom_param
                    m = tg_batch[key].get_matrix()
                    pos = m[:, :3, 3].float()
                    # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
                    rot = m[:, :3, :3].float()
                    # pos = pos[i_ts]
                    # rot = rot[i_ts]
                    # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
                    if cur_visual_mesh_fn is None:
                        continue
                    if isinstance(cur_visual_mesh_fn, tuple):
                        cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                    
                    if key not in link_key_to_vis_mesh:
                        link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        verts = link_key_to_vis_mesh[key]
                        # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                        
                        verts = torch.from_numpy(verts).float().to(d)
                        # if to_numpy:
                        verts = verts.cpu()
                        verts = m_offset.transform_points(verts)
                        
                        # if not to_numpy:
                        verts = verts.to(d) 
                        
                        link_key_to_vis_mesh[key] = verts
                    
                    
                    verts = link_key_to_vis_mesh[key]
                    
                    
                    # if to_numpy:
                    #     transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                    # else:
                    transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
                    # transform #
                    # print(f"transformed_verts: {transformed_verts.size()}")
                    tot_meshes_verts.append(transformed_verts)
                    
                    # cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
                    # tot_link_idxes.append(cur_visual_link_idxes)
                    
            transformed_verts = torch.cat(tot_meshes_verts, dim=1)
            # if to_numpy:
            #     transformed_verts = transformed_verts.detach().cpu().numpy()
            # tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
            # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
            # print(transformed_verts.shape)
            # tot_hand_verts.append(transformed_verts)
            # tot_ts_tot_link_idxes.append(tot_link_idxes)
            """"""""""""""""""""""""""    """"""""""""""""""""""""""

            
            ########## Version 1 ##########
            # robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            # robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            # print(f"robot_qs: {robot_qs.device}")
            # robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs, to_numpy=False) # robot qs --- (nn_ts, ) #
            # print(f"robot_verts_tot: {robot_verts_tot.device}, tot_ts_tot_link_idxes_th: {tot_ts_tot_link_idxes_th.device}")
            ########## Version 1 ##########
            
            
            robot_verts_tot = transformed_verts
            
            robot_verts_palm = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[palm_name]]
            robot_verts_first_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[first_tip_name]]
            robot_verts_second_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[second_tip_name]]
            robot_verts_third_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[third_tip_name]]
            robot_verts_forth_tip = robot_verts_tot[:, tot_ts_tot_link_idxes_th == link_name_to_link_index[forth_tip_name]]
            
            robot_verts_palm_center = robot_verts_palm.mean(dim=1)
            robot_verts_first_tip_center = robot_verts_first_tip.mean(dim=1)
            robot_verts_second_tip_center = robot_verts_second_tip.mean(dim=1)
            robot_verts_third_tip_center = robot_verts_third_tip.mean(dim=1)
            robot_verts_forth_tip_ceffter = robot_verts_forth_tip.mean(dim=1) # nn_ts x 3 #
            
            # 
            diff_palm_center = robot_verts_palm_center - target_palm_center
            diff_first_tip_center = robot_verts_first_tip_center - target_first_tip_center
            diff_second_tip_center = robot_verts_second_tip_center - target_second_tip_center
            diff_third_tip_center = robot_verts_third_tip_center - target_third_tip_center
            diff_forth_tip_center = robot_verts_forth_tip_ceffter - target_forth_tip_center
            
            # print(f"diff_palm_center: {diff_palm_center.size()}, diff_first_tip_center: {diff_first_tip_center.size()}")
            # calcuale losses
            loss_palm = torch.sum(diff_palm_center ** 2, dim=-1).mean()
            loss_first_tip = torch.sum(diff_first_tip_center ** 2, dim=-1).mean()
            loss_second_tip = torch.sum(diff_second_tip_center ** 2, dim=-1).mean()
            loss_third_tip = torch.sum(diff_third_tip_center ** 2, dim=-1).mean()
            loss_forth_tip = torch.sum(diff_forth_tip_center ** 2, dim=-1).mean()
            
            # # forth tip; third top ###
            ## TODO: add some regularizations ###
            
            # robot_qs_delta = robot_qs[1:, ]
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip # + glb_qs_reg_loss * 100
            print(f"loss: {loss.item()}, loss_palm: {loss_palm.item()}, loss_first_tip: {loss_first_tip.item()}, loss_second_tip: {loss_second_tip.item()}, loss_third_tip: {loss_third_tip.item()}, loss_forth_tip: {loss_forth_tip.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
            
            
        
        
        for i_iter in range(nn_iters):
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
        first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
        first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
        sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
        
        
        ####### reorientation stage #######
        last_hand_qs = sampled_hand_qs[-1]
        last_obj_pos = sampled_obj_pos[-1]
        last_hand_qs_th = torch.from_numpy(last_hand_qs).float().to(d)
        tg_batch_last_frame = chain.forward_kinematics(last_hand_qs_th.unsqueeze(0)) # last hand qs #
        
        palm_link_key  =  palm_name # 'palm_link'
        transformation_matrix_palm_link = tg_batch_last_frame[palm_link_key].get_matrix() # (nn_ts, 4, 4)
        position_palm_link = transformation_matrix_palm_link[:, :3, 3] # (nn_ts, 3)
        
        
        # get the plam link #
        init_palm_pos = position_palm_link[0] # (3,)
        last_obj_pos = torch.from_numpy(last_obj_pos).float().cuda() # (3,) 
        palm_to_obj_vec = last_obj_pos - init_palm_pos # (3,) # from the plam to the object -- the rotation vectors
        palm_to_obj_vec = palm_to_obj_vec / torch.clamp( torch.norm(palm_to_obj_vec, p=2, dim=-1, keepdim=False), min=1e-5 ) 
        
        
        last_hand_qs = sampled_hand_qs[-1].copy()
        last_obj_pos = sampled_obj_pos[-1].copy()
        last_obj_rot = sampled_obj_rot[-1].copy()
        
        last_obj_rot_euler = R.from_quat(last_obj_rot).as_euler('zyx', degrees=False)
        
        delta_mult_coef = 0.008 * np.pi
        
        # cur_st_obj_rot_euler
        rnd_obj_rot_vec = np.random.randn(3,)
        rnd_obj_rot_vec = rnd_obj_rot_vec / np.linalg.norm(rnd_obj_rot_vec, ord=2) # (3,)-dim unit length vector
        
        if not rnd_sample_axis:
            ### use the palm to obj vec as the rotation axis ####
            rnd_obj_rot_vec = palm_to_obj_vec.detach().cpu().numpy()
            if neg_palm_to_obj_vec:
                rnd_obj_rot_vec = rnd_obj_rot_vec * (-1.0)
            else:
                rnd_obj_rot_vec = rnd_obj_rot_vec
        
        
        nex_stage_obj_rot = []
        
        # if w_interpolate:
        rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
        for i_fr in range(nn_continuing_frames_s2): 
            cur_fr_obj_rot_euler = last_obj_rot_euler + rnd_rot_euler * (i_fr + 1)
            cur_fr_obj_rot_struct = R.from_euler('zyx', cur_fr_obj_rot_euler, degrees=False)
            cur_fr_obj_rot_quat = cur_fr_obj_rot_struct.as_quat()
            nex_stage_obj_rot.append(cur_fr_obj_rot_quat.copy()) # avoid the soft copy #
        # continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        # else:
            
        #     rnd_rot_euler = R.from_rotvec(rnd_obj_rot_vec  * delta_mult_coef).as_euler('zyx', degrees=False)
        #     final_rot_euler = rnd_rot_euler * nn_continuing_frames
        #     final_rot_euler = final_rot_euler + cur_st_obj_rot_euler
        #     final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
        #     final_rot_quat = final_rot_struct.as_quat() # final rot quat #
            
        #     # final_rot_vec = rnd_obj_rot_vec * delta_mult_coef * nn_continuing_frames
        #     # final_rot_vec_euler = R.from_rotvec(final_rot_vec).as_euler('zyx', degrees=False)
        #     # final_rot_euler = cur_st_obj_rot_euler + final_rot_vec_euler
        #     # final_rot_struct = R.from_euler('zyx', final_rot_euler, degrees=False)
        #     # final_rot_quat = final_rot_struct.as_quat() # final rot quat #
        #     continuing_obj_rot = [ final_rot_quat.copy() for _ in range(nn_continuing_frames) ]
        
        nex_stage_hand_qs = [
            last_hand_qs.copy() for _ in range(nn_continuing_frames_s2)
        ]
        nex_stage_obj_pos = [
            last_obj_pos.copy() for _ in range(nn_continuing_frames_s2)
        ]
        nex_stage_hand_qs = np.stack(nex_stage_hand_qs, axis=0)
        nex_stage_obj_pos = np.stack(nex_stage_obj_pos, axis=0)
        nex_stage_obj_rot = np.stack(nex_stage_obj_rot, axis=0)
        
        sampled_hand_qs = np.concatenate([sampled_hand_qs, nex_stage_hand_qs], axis=0)
        sampled_obj_pos = np.concatenate([sampled_obj_pos, nex_stage_obj_pos], axis=0)
        sampled_obj_rot = np.concatenate([sampled_obj_rot, nex_stage_obj_rot], axis=0)
        ####### reorientation stage #######
        
        
        
        # synthesize trajs for the next stage #
        cur_st_hand_qs = sampled_hand_qs[-1].copy()
        cur_st_obj_pos = sampled_obj_pos[-1].copy()
        # cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame - 1].copy()
        continuing_hand_qs = []
        continuing_obj_pos = []
        continuing_obj_rot = []
        
        # maxx_rotation_delta = 0.02 * np.pi
        maxx_trans_delta = 0.01 # 
        
        
        nn_fr_per_chunk = 20

        nn_chunks = nn_continuing_frames_s3 // nn_fr_per_chunk
        
        for i_chunk in range(nn_chunks):
            rnd_trans_dir = np.random.randn(3)
            if rnd_trans_dir[2] < 0:
                rnd_trans_dir[2] = 0.0001
            rnd_trans_dir = rnd_trans_dir / np.linalg.norm(rnd_trans_dir, ord=2)
            # rnd_rot_dir = np.random.randn(3)
            # rnd_rot_dir = rnd_rot_dir / np.linalg.norm(rnd_rot_dir, ord=2)
            
            # rnd_delta_trans = rnd_trans_dir *
            rnd_trans_scale = np.random.uniform(low=0.0, high=maxx_trans_delta, size=(3, ))
            # rnd_rot_scale = np.random.uniform(low=0.0, high=maxx_rotation_delta, size=(3, ))
            rnd_delta_trans = rnd_trans_dir * rnd_trans_scale
            # rnd_delta_rot = rnd_rot_dir * rnd_rot_scale
            
            for i_fr in range(nn_fr_per_chunk):
                # cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + rnd_delta_trans
                cur_st_obj_pos = cur_st_obj_pos + rnd_delta_trans
                # cur_st_obj_rot_euler = cur_st_obj_rot_euler + rnd_delta_rot
                # cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
                # cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
                
                continuing_hand_qs.append(cur_st_hand_qs.copy())
                continuing_obj_pos.append(cur_st_obj_pos.copy())
                continuing_obj_rot.append(sampled_obj_rot[-1].copy()) 
        
        
        continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
        continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
        continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        continuing_st_hand_qs_th = torch.from_numpy(continuing_hand_qs[-1:]).float().cuda()
        continuing_st_tg_batch = chain.forward_kinematics(continuing_st_hand_qs_th) # 
        continuing_palm_link_pos = continuing_st_tg_batch[palm_link_key].get_matrix()[0, :3, 3]
        tot_continuing_target_palm_link_pos = [ continuing_palm_link_pos.clone() ]
        for i_fr in range(continuing_obj_pos.shape[0] - 1):
            cur_delta_obj_trans = continuing_obj_pos[i_fr + 1] - continuing_obj_pos[i_fr]
            cur_delta_obj_trans = torch.from_numpy(cur_delta_obj_trans).float().cuda()
            cur_target_palm_pos = tot_continuing_target_palm_link_pos[-1] + cur_delta_obj_trans
            tot_continuing_target_palm_link_pos.append(cur_target_palm_pos.clone())
        tot_continuing_target_palm_link_pos = torch.stack(tot_continuing_target_palm_link_pos, dim=0) # nn_continuing_ts x 3 ## --- palm link pos #
        


        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float().cuda()
        
        import torch.nn as nn
        robot_hand_qs = nn.Embedding(
            num_embeddings=300, embedding_dim=60,
        ).cuda()
        torch.nn.init.zeros_(robot_hand_qs.weight)
        for i_fr in range(0,continuing_hand_qs_th.size(0)):
            if i_fr == 0:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone()
            else:
                robot_hand_qs.weight.data[i_fr, : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[i_fr, :].clone() - continuing_hand_qs_th[i_fr - 1, :].clone()
                
        
        # robot_hand_qs.weight.data[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] = continuing_hand_qs_th[:, :].clone()
        params_to_train = []
        params_to_train += list(robot_hand_qs.parameters())
        
        optimizer = torch.optim.LBFGS(params_to_train) # params jo train
        # self.robot_delta_states.weight.data[0, 24] = 1.0
        nn_iters = 50
        
        
        # avg_continuing_obj_verts = continuing_obj_verts.mean(dim=1) # (nn_ts x 3)
        
        tot_ts_tot_link_idxes_th = tot_ts_tot_link_idxes_th.cuda()
        
        
        
        
        def closer():
            optimizer.zero_grad()
            
            """"""""""""""""""""""""""    """"""""""""""""""""""""""
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            tg_batch = chain.forward_kinematics(robot_qs[:])

            tot_meshes_verts = []
            
            palm_link_pos = tg_batch[palm_link_key].get_matrix()[:, :3, 3]
            diff_palm_link_pos_w_target = torch.sum(
                ( palm_link_pos - tot_continuing_target_palm_link_pos ) ** 2, dim=-1
            )
            loss_palm = diff_palm_link_pos_w_target.mean()
            
            
            glb_qs_reg_loss = torch.sum(
                robot_hand_qs.weight[1:, :] ** 2, dim=-1
            )
            glb_qs_reg_loss = glb_qs_reg_loss.mean()
            
            loss = loss_palm 
            print(f"loss: {loss.item()}, loss_palm: {loss_palm.item()}, glb_qs_reg_loss: {glb_qs_reg_loss.item()}")
            loss.backward()
            
            return loss
        
        
        for i_iter in range(nn_iters):
            optimizer.step(closer)
        
        
        
        robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
        robot_qs = torch.cumsum(robot_qs, dim=0)
        continuing_hand_qs = robot_qs.cpu().detach().numpy()
        
        
        
        
        
        
        
        
        
        sampled_hand_qs = np.concatenate(
            [ sampled_hand_qs, continuing_hand_qs ], axis=0
        )
        sampled_obj_pos = np.concatenate(
            [ sampled_obj_pos, continuing_obj_pos ], axis=0
        )
        sampled_obj_rot = np.concatenate(
            [ sampled_obj_rot, continuing_obj_rot ], axis=0 # 
        )
        
        
        
        
        
        
        sampled_hand_qs_th = torch.from_numpy(sampled_hand_qs).float().cuda()
        tg_batch = chain.forward_kinematics(sampled_hand_qs_th)
        
        link_key_to_link_pos = {}
        link_key_to_link_pose = {}
        
        for i_link, key in enumerate(allegro_link_names):
            link = chain.find_link(key)
            link_pos = tg_batch[key].get_matrix()[:, :3, 3].detach().cpu().numpy()
            link_key_to_link_pos[key] = link_pos.copy()
            
            link_rot_mtx = tg_batch[key].get_matrix()[:, :3, :3].detach().cpu().numpy()
            cur_link_rot_quat = []
            for i_fr in range(link_rot_mtx.shape[0]):
                cur_fr_rot_mtx = link_rot_mtx[i_fr]
                cur_fr_rot_ornt = R.from_matrix(cur_fr_rot_mtx).as_quat() #
                cur_link_rot_quat.append(cur_fr_rot_ornt)
            cur_link_rot_quat = np.stack(cur_link_rot_quat, axis=0 ) # nn_frames x 4
            cur_link_pose = np.concatenate([link_pos, cur_link_rot_quat ], axis=-1)
            link_key_to_link_pose[key] = cur_link_pose
        
        # kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs 
        # kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 
        # kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 
        
        
        # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
        sampled_res_sv_dict = {
            'hand_qs': sampled_hand_qs,
            'obj_pos': sampled_obj_pos,
            'obj_rot': sampled_obj_rot,
            'robot_delta_states_weights_np': sampled_hand_qs,
            'object_transl': sampled_obj_pos,
            'object_rot_quat': sampled_obj_rot,
            'link_key_to_link_pos': link_key_to_link_pos,
            'link_key_to_link_pose': link_key_to_link_pose,
            # 'rotation_axis': -1.0 * palm_to_obj_vec,
            # 'rotation_axis':  palm_to_obj_vec,
        }
        
        
        
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
        # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v3"
        os.makedirs(resampled_info_sv_root, exist_ok=True)
        
        
        
        if nn_samples == 1:
            if hand_type == 'leap':
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
            else:
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        else:
            # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            if hand_type == 'leap':
                # resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300.npy"
                resampled_info_sv_fn =  f"leap_passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
            else:
                # resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
                resampled_info_sv_fn = f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")




def extract_kinematics_from_tracked_results(result_fn, inst_tag, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    version = 5
    version = 6
    version = 7
    version = 8
    version = 9
    version = 10
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}")
    
    
    # if os.path.exists(resampled_info_sv_root):
    #     print(f"Resampled folder {resampled_info_sv_root} already exists!")
    #     return
    
    
    
    if 'robot_delta_states_weights_np' in tracked_res:
        # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        ##### traj modifications core #####
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        # traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        ##### traj modifications core #####
        
        print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

        
        ##### traj modifications core v3 #####
        traj_modifications_core_v3(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf)
        ##### traj modifications core v3 #####

    else:
    
        time_keys = []
        for key in tracked_res:
            try:
                key_int = int(key)
            except:
                continue
            time_keys.append(key_int)
        time_keys = sorted(time_keys) # get the time keys
        tot_shadow_hand_dof_pos = []
        tot_object_pose = []
        
        for i_time in time_keys:
            cur_ts_tracked_res = tracked_res[i_time]
            cur_ts_shadow_hand_dof_pos = cur_ts_tracked_res['shadow_hand_dof_pos']
            cur_ts_object_pose = cur_ts_tracked_res['object_pose']
            tot_object_pose.append(cur_ts_object_pose) # nn_ts x nn_envs x 7
            tot_shadow_hand_dof_pos.append(cur_ts_shadow_hand_dof_pos)
        tot_object_pose = np.stack(tot_object_pose, axis=0) # nn_ts x nn_envs x 7 # # #
        tot_shadow_hand_dof_pos = np.stack(tot_shadow_hand_dof_pos, axis=0) # nn_ts x nn_envs x 22 # # #
        
        
        # kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
        # kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        # kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
        # kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
        # print(f"kine_traj_data: {kine_traj_data.keys()}") # trajector modifications #
        
        # kine_obj_pos = kine_traj_data['object_transl']
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
        # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        tot_envs_max_z = []
        for i_env in range(tot_object_pose.shape[1]):
            cur_env_maxx_z = np.max(tot_object_pose[:, i_env, 2])
            tot_envs_max_z.append(cur_env_maxx_z)
        tot_envs_max_z = np.array(tot_envs_max_z, dtype=np.float32)
        sorted_env_idxes = np.argsort(tot_envs_max_z)[::-1] # (nn_envs)
        selected_env_idx = sorted_env_idxes[1].item()
        
        cur_env_obj_pos = tot_object_pose[:, selected_env_idx, :3]
        cur_env_obj_ornt = tot_object_pose[:, selected_env_idx, 3:]
        cur_env_hand_qs = tot_shadow_hand_dof_pos[:, selected_env_idx]
        cur_env_hand_qs = cur_env_hand_qs[:, joint_idxes_ordering]
        
        # resampled_info_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v4"
        
        traj_modifications_core(cur_env_hand_qs, cur_env_obj_pos, cur_env_obj_ornt, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        


def extract_kinematics_from_tracked_results_final_tracking_orientation(result_fn, inst_tag, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False, nn_samples=1, w_interpolate=False, rnd_sample_axis=True, neg_palm_to_obj_vec=False, nn_continuing_frames=180, with_following_frames=False, start_gen_frame=120):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    version = 5
    version = 6
    version = 7
    version = 8
    version = 9
   
    version = 14
    version = 15
    version = 16
    version = 17
    version = 18
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}")
    
    
    
    
    if 'robot_delta_states_weights_np' in tracked_res:
        # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        ##### traj modifications core #####
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        # traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        ##### traj modifications core #####
        
        print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

        
        ##### traj modifications core v3 #####
        traj_modifications_core_v5(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, nn_samples=nn_samples, w_interpolate=w_interpolate, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames, with_following_frames=with_following_frames, start_gen_frame=start_gen_frame)
        ##### traj modifications core v3 #####

    else:
    
        time_keys = []
        for key in tracked_res:
            try:
                key_int = int(key)
            except:
                continue
            time_keys.append(key_int)
        time_keys = sorted(time_keys) # get the time keys
        tot_shadow_hand_dof_pos = []
        tot_object_pose = []
        
        for i_time in time_keys:
            cur_ts_tracked_res = tracked_res[i_time]
            cur_ts_shadow_hand_dof_pos = cur_ts_tracked_res['shadow_hand_dof_pos']
            cur_ts_object_pose = cur_ts_tracked_res['object_pose']
            tot_object_pose.append(cur_ts_object_pose) # nn_ts x nn_envs x 7
            tot_shadow_hand_dof_pos.append(cur_ts_shadow_hand_dof_pos)
        tot_object_pose = np.stack(tot_object_pose, axis=0) # nn_ts x nn_envs x 7 # # #
        tot_shadow_hand_dof_pos = np.stack(tot_shadow_hand_dof_pos, axis=0) # nn_ts x nn_envs x 22 # # #
        
        
        # kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
        # kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        # kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
        # kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
        # print(f"kine_traj_data: {kine_traj_data.keys()}") # trajector modifications #
        
        # kine_obj_pos = kine_traj_data['object_transl']
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
        # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        tot_envs_max_z = []
        for i_env in range(tot_object_pose.shape[1]):
            cur_env_maxx_z = np.max(tot_object_pose[:, i_env, 2])
            tot_envs_max_z.append(cur_env_maxx_z)
        tot_envs_max_z = np.array(tot_envs_max_z, dtype=np.float32)
        sorted_env_idxes = np.argsort(tot_envs_max_z)[::-1] # (nn_envs)
        selected_env_idx = sorted_env_idxes[1].item()
        
        cur_env_obj_pos = tot_object_pose[:, selected_env_idx, :3]
        cur_env_obj_ornt = tot_object_pose[:, selected_env_idx, 3:]
        cur_env_hand_qs = tot_shadow_hand_dof_pos[:, selected_env_idx]
        cur_env_hand_qs = cur_env_hand_qs[:, joint_idxes_ordering]
        
        # resampled_info_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v4"
        
        traj_modifications_core(cur_env_hand_qs, cur_env_obj_pos, cur_env_obj_ornt, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
      


def extract_kinematics_from_tracked_results_rndsample(result_fn, inst_tag, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    version = 5
    version = 6
    version = 7
    version = 8
    version = 9
    version = 10
    version = 11
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}")
    
    
    # if os.path.exists(resampled_info_sv_root):
    #     print(f"Resampled folder {resampled_info_sv_root} already exists!")
    #     return
    
    
    
    if 'robot_delta_states_weights_np' in tracked_res:
        # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        ##### traj modifications core #####
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        # traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        ##### traj modifications core #####
        
        print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

        
        ##### traj modifications core v3 #####
        traj_modifications_core_v4(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf)
        ##### traj modifications core v3 #####






def extract_kinematics_from_tracked_results_rndsample_hybrid(result_fn_a, result_fn_b, inst_tag_a, inst_tag_b, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False):
    inst_tag = inst_tag_a
    # result_fn_a -- object
    # result_fn_b -- hand
    tracked_res_a = np.load(result_fn_a, allow_pickle=True).item()
    tracked_res_b= np.load(result_fn_b, allow_pickle=True).item()
    
    version = 12
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag_a}_{inst_tag_b}_v{version}")
    
    
    
    kine_hand_qs = tracked_res_b['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
    kine_obj_pos = tracked_res_a['object_transl'] # nn_frames x 3 # #
    kine_obj_quat = tracked_res_a['object_rot_quat'] # nn_frames x 4 # #

    
    print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

    # traj modifications core v3 #
    
    nn_samples = 20
    ##### traj modifications core v3 #####
    traj_modifications_core_v4(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, nn_samples=nn_samples)
    ##### traj modifications core v3 #####

   

def extract_kinematics_from_tracked_results_three_stages(result_fn, inst_tag, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False, nn_samples=1, w_interpolate=False, rnd_sample_axis=True, neg_palm_to_obj_vec=False, nn_continuing_frames=180, with_following_frames=False, start_gen_frame=120, nn_continuing_frames_s2=150):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    version = 5
    version = 6
    version = 7
    version = 8
    version = 9
   
    version = 14
    version = 15
    version = 16
    version = 17
    version = 18
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}")
    
    
    
    
    if 'robot_delta_states_weights_np' in tracked_res:
        # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        ##### traj modifications core #####
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        # traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        ##### traj modifications core #####
        
        print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

        # kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100, start_gen_frame=100, nn_continuing_frames=200, nn_continuing_frames_s2=150, rnd_sample_axis=False, neg_palm_to_obj_vec=False
        
        link_key_to_link_pos = {}
        
        ##### traj modifications core v3 #####
        traj_modifications_core_v6(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, link_key_to_link_pos=link_key_to_link_pos, nn_samples=nn_samples, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames,  start_gen_frame=start_gen_frame, nn_continuing_frames_s2=nn_continuing_frames_s2)
        ##### traj modifications core v3 #####

    else:
    
        time_keys = []
        for key in tracked_res:
            try:
                key_int = int(key)
            except:
                continue
            time_keys.append(key_int)
        time_keys = sorted(time_keys) # get the time keys
        tot_shadow_hand_dof_pos = []
        tot_object_pose = []
        
        for i_time in time_keys:
            cur_ts_tracked_res = tracked_res[i_time]
            cur_ts_shadow_hand_dof_pos = cur_ts_tracked_res['shadow_hand_dof_pos']
            cur_ts_object_pose = cur_ts_tracked_res['object_pose']
            tot_object_pose.append(cur_ts_object_pose) # nn_ts x nn_envs x 7
            tot_shadow_hand_dof_pos.append(cur_ts_shadow_hand_dof_pos)
        tot_object_pose = np.stack(tot_object_pose, axis=0) # nn_ts x nn_envs x 7 # # #
        tot_shadow_hand_dof_pos = np.stack(tot_shadow_hand_dof_pos, axis=0) # nn_ts x nn_envs x 22 # # #
        
        
        # kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
        # kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        # kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
        # kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
        # print(f"kine_traj_data: {kine_traj_data.keys()}") # trajector modifications #
        
        # kine_obj_pos = kine_traj_data['object_transl']
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
        # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        tot_envs_max_z = []
        for i_env in range(tot_object_pose.shape[1]):
            cur_env_maxx_z = np.max(tot_object_pose[:, i_env, 2])
            tot_envs_max_z.append(cur_env_maxx_z)
        tot_envs_max_z = np.array(tot_envs_max_z, dtype=np.float32)
        sorted_env_idxes = np.argsort(tot_envs_max_z)[::-1] # (nn_envs)
        selected_env_idx = sorted_env_idxes[1].item()
        
        cur_env_obj_pos = tot_object_pose[:, selected_env_idx, :3]
        cur_env_obj_ornt = tot_object_pose[:, selected_env_idx, 3:]
        cur_env_hand_qs = tot_shadow_hand_dof_pos[:, selected_env_idx]
        cur_env_hand_qs = cur_env_hand_qs[:, joint_idxes_ordering]
        
        # resampled_info_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v4"
        
        traj_modifications_core(cur_env_hand_qs, cur_env_obj_pos, cur_env_obj_ornt, inst_tag, resampled_info_sv_root=resampled_info_sv_root)




def extract_kinematics_from_tracked_results_trans_only(result_fn, inst_tag, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False, nn_samples=1, w_interpolate=False, rnd_sample_axis=True, neg_palm_to_obj_vec=False, nn_continuing_frames=180, with_following_frames=False, start_gen_frame=120, nn_continuing_frames_s2=150):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    version = 5
    version = 6
    version = 7
    version = 8
    version = 9
   
    version = 14
    version = 15
    version = 16
    version = 17
    version = 18
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}")
    
    
    
    
    if 'robot_delta_states_weights_np' in tracked_res:
        # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        ##### traj modifications core #####
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        # traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        ##### traj modifications core #####
        
        print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

        # kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100, start_gen_frame=100, nn_continuing_frames=200, nn_continuing_frames_s2=150, rnd_sample_axis=False, neg_palm_to_obj_vec=False
        
        link_key_to_link_pos = {}
        
        ##### traj modifications core v3 #####
        traj_modifications_core_v7(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, link_key_to_link_pos=link_key_to_link_pos, nn_samples=nn_samples, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames,  start_gen_frame=start_gen_frame, nn_continuing_frames_s2=nn_continuing_frames_s2)
        ##### traj modifications core v3 #####

    else:
    
        time_keys = []
        for key in tracked_res:
            try:
                key_int = int(key)
            except:
                continue
            time_keys.append(key_int)
        time_keys = sorted(time_keys) # get the time keys
        tot_shadow_hand_dof_pos = []
        tot_object_pose = []
        
        for i_time in time_keys:
            cur_ts_tracked_res = tracked_res[i_time]
            cur_ts_shadow_hand_dof_pos = cur_ts_tracked_res['shadow_hand_dof_pos']
            cur_ts_object_pose = cur_ts_tracked_res['object_pose']
            tot_object_pose.append(cur_ts_object_pose) # nn_ts x nn_envs x 7
            tot_shadow_hand_dof_pos.append(cur_ts_shadow_hand_dof_pos)
        tot_object_pose = np.stack(tot_object_pose, axis=0) # nn_ts x nn_envs x 7 # # #
        tot_shadow_hand_dof_pos = np.stack(tot_shadow_hand_dof_pos, axis=0) # nn_ts x nn_envs x 22 # # #
        
        
        # kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
        # kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        # kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
        # kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
        # print(f"kine_traj_data: {kine_traj_data.keys()}") # trajector modifications #
        
        # kine_obj_pos = kine_traj_data['object_transl']
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
        # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        tot_envs_max_z = []
        for i_env in range(tot_object_pose.shape[1]):
            cur_env_maxx_z = np.max(tot_object_pose[:, i_env, 2])
            tot_envs_max_z.append(cur_env_maxx_z)
        tot_envs_max_z = np.array(tot_envs_max_z, dtype=np.float32)
        sorted_env_idxes = np.argsort(tot_envs_max_z)[::-1] # (nn_envs)
        selected_env_idx = sorted_env_idxes[1].item()
        
        cur_env_obj_pos = tot_object_pose[:, selected_env_idx, :3]
        cur_env_obj_ornt = tot_object_pose[:, selected_env_idx, 3:]
        cur_env_hand_qs = tot_shadow_hand_dof_pos[:, selected_env_idx]
        cur_env_hand_qs = cur_env_hand_qs[:, joint_idxes_ordering]
        
        # resampled_info_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v4"
        
        traj_modifications_core(cur_env_hand_qs, cur_env_obj_pos, cur_env_obj_ornt, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
      




def extract_kinematics_from_tracked_results_four_stages(result_fn, inst_tag, modified_data_sv_root="/data/xueyi/data", hand_type='allegro', w_franka=False, w_franka_v2urdf=False, nn_samples=1, w_interpolate=False, rnd_sample_axis=True, neg_palm_to_obj_vec=False, nn_continuing_frames=180, with_following_frames=False, start_gen_frame=120, nn_continuing_frames_s2=150, nn_continuing_frames_s3=60):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    version = 5
    version = 6
    version = 7
    version = 8
    version = 9
   
    version = 14
    version = 15
    version = 16
    version = 17
    version = 18
    # resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}"
    
    resampled_info_sv_root = os.path.join(modified_data_sv_root, f"GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v{version}")
    
    
    
    
    if 'robot_delta_states_weights_np' in tracked_res:
        # joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        # joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        # joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        ##### traj modifications core #####
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        # traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        ##### traj modifications core #####
        
        print(f"kine_hand_qs: {kine_hand_qs.shape}, kine_obj_pos: {kine_obj_pos.shape}, kine_obj_quat: {kine_obj_quat.shape}")

        # kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2", hand_type='allegro', w_franka=False , w_franka_v2urdf=False, link_key_to_link_pos={}, nn_samples=100, start_gen_frame=100, nn_continuing_frames=200, nn_continuing_frames_s2=150, rnd_sample_axis=False, neg_palm_to_obj_vec=False
        
        link_key_to_link_pos = {}
        
        ##### traj modifications core v3 #####
        traj_modifications_core_v8(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, link_key_to_link_pos=link_key_to_link_pos, nn_samples=nn_samples, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames,  start_gen_frame=start_gen_frame, nn_continuing_frames_s2=nn_continuing_frames_s2, nn_continuing_frames_s3=nn_continuing_frames_s3)
        ##### traj modifications core v3 #####

    else:
    
        time_keys = []
        for key in tracked_res:
            try:
                key_int = int(key)
            except:
                continue
            time_keys.append(key_int)
        time_keys = sorted(time_keys) # get the time keys
        tot_shadow_hand_dof_pos = []
        tot_object_pose = []
        
        for i_time in time_keys:
            cur_ts_tracked_res = tracked_res[i_time]
            cur_ts_shadow_hand_dof_pos = cur_ts_tracked_res['shadow_hand_dof_pos']
            cur_ts_object_pose = cur_ts_tracked_res['object_pose']
            tot_object_pose.append(cur_ts_object_pose) # nn_ts x nn_envs x 7
            tot_shadow_hand_dof_pos.append(cur_ts_shadow_hand_dof_pos)
        tot_object_pose = np.stack(tot_object_pose, axis=0) # nn_ts x nn_envs x 7 # # #
        tot_shadow_hand_dof_pos = np.stack(tot_shadow_hand_dof_pos, axis=0) # nn_ts x nn_envs x 22 # # #
        
        
        # kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
        # kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
        # kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
        # kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
        # print(f"kine_traj_data: {kine_traj_data.keys()}") # trajector modifications #
        
        # kine_obj_pos = kine_traj_data['object_transl']
        
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
        # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        tot_envs_max_z = []
        for i_env in range(tot_object_pose.shape[1]):
            cur_env_maxx_z = np.max(tot_object_pose[:, i_env, 2])
            tot_envs_max_z.append(cur_env_maxx_z)
        tot_envs_max_z = np.array(tot_envs_max_z, dtype=np.float32)
        sorted_env_idxes = np.argsort(tot_envs_max_z)[::-1] # (nn_envs)
        selected_env_idx = sorted_env_idxes[1].item()
        
        cur_env_obj_pos = tot_object_pose[:, selected_env_idx, :3]
        cur_env_obj_ornt = tot_object_pose[:, selected_env_idx, 3:]
        cur_env_hand_qs = tot_shadow_hand_dof_pos[:, selected_env_idx]
        cur_env_hand_qs = cur_env_hand_qs[:, joint_idxes_ordering]
        
        # resampled_info_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v4"
        
        traj_modifications_core(cur_env_hand_qs, cur_env_obj_pos, cur_env_obj_ornt, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
      
 
 

def hand_pk_for_palm_link_pose(urdf_fn, hand_qs, to_numpy=True):
    
    # allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    
    if 'allegro_hand_description_right_franka' in urdf_fn:
        # allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link', 'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_link8', 'part_studio_link', 'camera_link']
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link', 'panda_link0', 'panda_link1', 'panda_link2', 'panda_link3', 'panda_link4', 'panda_link5', 'panda_link6', 'panda_link7', 'panda_link8',]
    else:
        allegro_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    
    
    mesh_root = '/'.join(urdf_fn.split('/')[:-1])
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    chain = chain.to(dtype=dtype, device=d)
    print(len(chain.get_joint_parameter_names()))
    nn_hand_dof = len(chain.get_joint_parameter_names())
    
    
    if not isinstance(hand_qs, torch.Tensor):
        
        hand_qs = torch.from_numpy(hand_qs).double().to(d)[:, :nn_hand_dof]
    
    # else:
    #     hand_qs = 
    
    # get the name to visual meshes ##
    # links = chain.get_links()
    # for link in links:
    #     print(link)
    # linka_names = chain.get_link_names()
    N = 1000 # # # #
    # th_batch = hand_qs[:16] # torch.rand(N, len(chain.get_joint_parameter_names()), dtype=dtype, device=d)
    th_batch = hand_qs
    tg_batch = chain.forward_kinematics(th_batch[:]) # # 
    # for i in range(N):
    #     tg = chain.forward_kinematics(th_batch[i]) # 
    print(type(tg_batch))
    print(tg_batch.keys())
    
    link_key_to_vis_mesh = {}
    
    tot_hand_verts = []
    tot_ts_tot_link_idxes = []
    
    target_link = 'palm_link'
    
    # for i_ts in range(th_batch.size(0)):
    #     # if i_ts >= 5:
    #     #     break
    #     tot_meshes_verts = []
    #     tot_link_idxes = []
    #     # for key in tg_batch.keys():
        
        
    #     link = chain.find_link(target_link)
    
    # cur_kine_obj_trans: [ 0.4  -0.    0.66]
    # cur_kine_qs: (300, 22)
    # target_mat [[ 0.43  0.13 -0.89]
    # [-0.82 -0.36 -0.44]
    # [-0.38  0.92 -0.05]]
    
    m = tg_batch[target_link].get_matrix()
    pos = m[:, :3, 3].float()
    rot = m[:, :3, :3].float()
    
    frame_idx = 0
    
    init_pos = pos[frame_idx]
    init_rot = rot[frame_idx]
    
    # init_rot = init_rot.T
    
    tot_euler_xyz= []
    print(init_rot)
    print(init_pos)
    
    for i_fr in range(rot.shape[0]):
        cur_rot = rot[i_fr].T
        cur_euler = R.from_matrix(cur_rot.cpu()).as_euler('xyz', degrees=False) # [[2, 1, 0]]
        
        tot_euler_xyz.append(cur_euler)
        
    tot_euler_xyz = np.stack(tot_euler_xyz, axis=0)
    tot_pos = pos.cpu().numpy()
    
    first_six_dim = np.concatenate([
        tot_pos, tot_euler_xyz
    ], axis=-1)
    hand_qs = hand_qs.cpu().numpy()
    hand_qs[:, :6] = first_six_dim
    return hand_qs
    
    # from scipy.spatial.transform import Rotation as R

    # rot_euler_xyz = R.from_matrix(init_rot.cpu()).as_euler('xyz', degrees=False)
    
    # # rot_euler_xyz = R.from_matrix()
    
    # print(init_pos)
    # print(init_rot)
    # print(rot_euler_xyz)
    
    
        
    #     for i_link, key in enumerate(allegro_link_names):
    #         # print(key, tg_batch[key].shape)
    #         link = chain.find_link(key)
    #         # print(f"key: {key}, link: {link}")
    #         link_visuals = link.visuals
    #         print(len(link_visuals))
            
    #         for cur_visual in link_visuals:
    #             m_offset = cur_visual.offset # .get_matrix() #
    #             cur_visual_mesh_fn = cur_visual.geom_param
    #             m = tg_batch[key].get_matrix()
    #             pos = m[:, :3, 3].float()
    #             # rot = pk.matrix_to_quaternion(m[:, :3, :3]) # rot # #
    #             rot = m[:, :3, :3].float()
    #             pos = pos[i_ts]
    #             rot = rot[i_ts]
    #             print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
    #             if cur_visual_mesh_fn is None:
    #                 continue
    #             if isinstance(cur_visual_mesh_fn, tuple):
    #                 cur_visual_mesh_fn = cur_visual_mesh_fn[0]
                
    #             if key not in link_key_to_vis_mesh:
    #                 link_key_to_vis_mesh[key] = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
    #             verts = link_key_to_vis_mesh[key]
    #             # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                
                
    #             verts = torch.from_numpy(verts).float().to(d)
    #             # if to_numpy:
    #             verts = verts.cpu()
    #             verts = m_offset.transform_points(verts)
                
    #             if not to_numpy:
    #                 verts = verts.to(d) 
    #             # 
    #             if to_numpy:
    #                 transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
    #             else:
    #                 transformed_verts = torch.matmul(verts, rot.transpose(0,1)) + pos.unsqueeze(0)
    #             # transform #
    #             print(f"transformed_verts: {transformed_verts.size()}")
    #             tot_meshes_verts.append(transformed_verts)
                
    #             cur_visual_link_idxes = np.array([i_link] * transformed_verts.size(0), dtype=np.int32)
    #             tot_link_idxes.append(cur_visual_link_idxes)
                
    #     transformed_verts = torch.cat(tot_meshes_verts, dim=0)
    #     if to_numpy:
    #         transformed_verts = transformed_verts.detach().cpu().numpy()
    #     tot_link_idxes = np.concatenate(tot_link_idxes, axis=0)
    #     # tot_link_idxes = torch.from_numpy(tot_link_idxes).long()
    #     # print(transformed_verts.shape)
    #     tot_hand_verts.append(transformed_verts)
    #     tot_ts_tot_link_idxes.append(tot_link_idxes)
    # if to_numpy:
    #     tot_hand_verts = np.stack(tot_hand_verts, axis= 0)
    # else:
    #     tot_hand_verts = torch.stack(tot_hand_verts, axis= 0)
    # tot_ts_tot_link_idxes = np.stack(tot_ts_tot_link_idxes, axis=0)
    # return tot_hand_verts, tot_ts_tot_link_idxes




# CUDA_VISIBLE_DEVICES=2 python utils/test_pk.py

# python utils/test_pk.py

# 


if __name__=='__main__':
    
    
    w_franka_v2urdf = False
    result_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    inst_tag = "ori_grab_s2_duck_inspect_1"
    
    result_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_apple_eat_1_nf_300.npy"
    inst_tag = "ori_grab_s1_apple_eat_1"
    
    result_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_eat_1_nf_300.npy"
    inst_tag  = "ori_grab_s1_banana_eat_1"
    
    
    #### single tracking instance modification creation #####
    # extract_kinematics_from_tracked_results(result_fn, inst_tag)
    #### single tracking instance modification creation ####
    

    
    w_franka = False
    hand_type = 'allegro'
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data"
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_v2"
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_vrandsamples"
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2"
    #### try to use this strategy to create modifications for each kinematics data ####
    tracking_data_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    tracking_data_st_tag = "passive_active_info_"
    
    
    # w_franka = False
    # hand_type = 'leap'
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap"
    # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data"
    # tracking_data_st_tag = "leap_passive_active_info_"
    
    w_interpolate = False
    
    w_franka = True
    hand_type = 'leap'
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka"
    w_franka_v2urdf = True
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf"
    # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data"
    # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data"
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf"
    # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data"
    tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data"
    tracking_data_st_tag = "leap_passive_active_info_"
    
    
    
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v4urdf"
    # # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data"
    # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data"
    # w_franka_v2urdf = False
    # tracking_data_st_tag = "leap_passive_active_info_"
    
    ### palm to obj vec ###
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v5urdf"
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v6urdf"
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v7urdf"
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v6urdf_winterp"
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v5urdf_winterp"
    
    rnd_sample_axis = False
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v7urdf_winterp"
    w_interpolate = True
    rnd_sample_axis = True
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v8urdf"
    w_interpolate = False   
    rnd_sample_axis = False
    
    nn_continuing_frames = 180
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v9urdf" # v9 urdf 
    w_interpolate = True
    rnd_sample_axis = False
    neg_palm_to_obj_vec = True
    nn_continuing_frames = 360
    
    with_following_frames=  False
    
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v10urdf" # v9 urd
    w_interpolate = True
    rnd_sample_axis = False
    neg_palm_to_obj_vec = True
    nn_continuing_frames = 180
    with_following_frames = True
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v11urdf" # v9 urd
    nn_continuing_frames = 150
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v13urdf" 
    start_gen_frame = 240
    start_gen_frame = 150
    nn_continuing_frames = 150
    with_following_frames = False
    neg_palm_to_obj_vec = True
    rnd_sample_axis = False
    
    
    
    ### corev6, three stages ###
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v14urdf" 
    nn_continuing_frames_s2 = 150
    nn_continuing_frames = 100
    start_gen_frame = 120
    nn_samples = 1
    neg_palm_to_obj_vec = True
    rnd_sample_axis = False
    ### corev6, three stages ###
    
    ### corev6, three stages, wish for smaller global movements ###
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v15urdf" 
    nn_continuing_frames_s2 = 150
    # nn_continuing_frames = 100
    nn_continuing_frames = 40
    start_gen_frame = 120
    nn_samples = 100
    neg_palm_to_obj_vec = True
    rnd_sample_axis = False
    ### corev6, three stages, wish for smaller global movements ###
    
    
    ### corev7, translational stage only ###
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v16urdf" 
    nn_continuing_frames_s2 = 150
    # nn_continuing_frames = 100
    nn_continuing_frames = 180
    start_gen_frame = 120
    nn_samples = 100
    neg_palm_to_obj_vec = True
    rnd_sample_axis = False
    ### corev7, translational stage only ###
    
    ### corev8, translational stage only ###
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v17urdf" 
    nn_continuing_frames_s2 = 150
    # nn_continuing_frames = 100
    nn_continuing_frames = 40
    nn_continuing_frames_s3 = 60
    start_gen_frame = 120
    nn_samples = 100
    neg_palm_to_obj_vec = True
    rnd_sample_axis = False
    ### corev8, translational stage only ###
    
    ########### ############ ############ ############ ############ ############ ############ ############ ############
    
    
    
    # nn_samples = 100
    # nn_samples = 1
    
    os.makedirs(modified_data_sv_root, exist_ok=True)
    
    tot_tracking_data_files = os.listdir(tracking_data_sv_root)
    tot_tracking_data_files = [fn for fn in tot_tracking_data_files if fn[: len(tracking_data_st_tag)] == tracking_data_st_tag and fn.endswith(".npy")]
    
    # tot_tracking_data_files = ["leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"] 
    # tot_tracking_data_files = ["leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy", "leap_passive_active_info_ori_grab_s2_hammer_use_2_nf_300.npy"] + tot_tracking_data_files
    tot_tracking_data_files = ["leap_passive_active_info_ori_grab_s2_hammer_use_2_nf_300.npy", "leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"] + tot_tracking_data_files
    
    for i_inst, cur_inst_tracking_data_fn in enumerate(tot_tracking_data_files):
        cur_inst_tag = cur_inst_tracking_data_fn.split(".")[0][len(tracking_data_st_tag): ].split("_nf_300")[0]
        result_fn = os.path.join(tracking_data_sv_root, cur_inst_tracking_data_fn)
        
        # extract_kinematics_from_tracked_results(result_fn, cur_inst_tag, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf)
        
        # extract_kinematics_from_tracked_results_final_tracking_orientation(result_fn, cur_inst_tag, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, nn_samples=nn_samples, w_interpolate=w_interpolate, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames, with_following_frames=with_following_frames, start_gen_frame=start_gen_frame)
        
        # ### lifting - translation - reorientation kinematics synthesis ###
        # extract_kinematics_from_tracked_results_three_stages(result_fn, cur_inst_tag, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, nn_samples=nn_samples, w_interpolate=w_interpolate, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames, with_following_frames=with_following_frames, start_gen_frame=start_gen_frame, nn_continuing_frames_s2=nn_continuing_frames_s2)
        # ### lifting - translation - reorientation kinematics synthesis ###
        
        
        # # ### translational only kinematics synthesis ###
        # extract_kinematics_from_tracked_results_trans_only(result_fn, cur_inst_tag, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, nn_samples=nn_samples, w_interpolate=w_interpolate, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames, with_following_frames=with_following_frames, start_gen_frame=start_gen_frame, nn_continuing_frames_s2=nn_continuing_frames_s2)
        # # ### translational only kinematics synthesis ###
        
        # ### lifting - translation - reorientation - translation kinematics synthesis ###
        extract_kinematics_from_tracked_results_four_stages(result_fn, cur_inst_tag, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf, nn_samples=nn_samples, w_interpolate=w_interpolate, rnd_sample_axis=rnd_sample_axis, neg_palm_to_obj_vec=neg_palm_to_obj_vec, nn_continuing_frames=nn_continuing_frames, with_following_frames=with_following_frames, start_gen_frame=start_gen_frame, nn_continuing_frames_s2=nn_continuing_frames_s2, nn_continuing_frames_s3=nn_continuing_frames_s3)
        # ### lifting - translation - reorientation - translation kinematics synthesis ###
    ########### ############ ############ ############ ############ ############ ############ ############ ############
    
    
    exit(0)
    
    
    # modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v4urdf"
    # # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data"
    # tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data"
    w_franka_v2urdf = False
    # tracking_data_st_tag = "leap_passive_active_info_"
    
    os.makedirs(modified_data_sv_root, exist_ok=True)
    
    tot_tracking_data_files = os.listdir(tracking_data_sv_root)
    tot_tracking_data_files = [fn for fn in tot_tracking_data_files if fn[: len(tracking_data_st_tag)] == tracking_data_st_tag and fn.endswith(".npy")]
    
    tot_tracking_data_files = [fn for fn in tot_tracking_data_files if "duck" in fn]
    
    # ############ ############ ############ ############ ############ ############ ############ ############ ############
    # # tot_tracking_data_files = ["leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"] 
    # # tot_tracking_data_files = ["leap_passive_active_info_ori_grab_s2_hammer_use_2_nf_300.npy", "leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"] + tot_tracking_data_files
    # # tot_tracking_data_files = ["passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"] # + tot_tracking_data_files
    
    # for i_inst, cur_inst_tracking_data_fn in enumerate(tot_tracking_data_files):
    #     cur_inst_tag = cur_inst_tracking_data_fn.split(".")[0][len(tracking_data_st_tag): ].split("_nf_300")[0]
    #     result_fn = os.path.join(tracking_data_sv_root, cur_inst_tracking_data_fn)
    #     extract_kinematics_from_tracked_results_rndsample(result_fn, cur_inst_tag, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf)
    
    # ############ ############ ############ ############ ############ ############ ############ ############ ############
    
    modified_data_sv_root = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2_hybrid"
    os.makedirs(modified_data_sv_root, exist_ok=True)
    
    for i_inst_a, cur_inst_tracking_data_fn_a in enumerate(tot_tracking_data_files):
        for i_inst_b, cur_inst_tracking_data_fn_b in enumerate(tot_tracking_data_files):
            if i_inst_a == i_inst_b:
                continue
            cur_inst_tag_a = cur_inst_tracking_data_fn_a.split(".")[0][len(tracking_data_st_tag): ].split("_nf_300")[0]
            result_fn_a = os.path.join(tracking_data_sv_root, cur_inst_tracking_data_fn_a)
            cur_inst_tag_b = cur_inst_tracking_data_fn_b.split(".")[0][len(tracking_data_st_tag): ].split("_nf_300")[0]
            result_fn_b = os.path.join(tracking_data_sv_root, cur_inst_tracking_data_fn_b)
            extract_kinematics_from_tracked_results_rndsample_hybrid(result_fn_a, result_fn_b, f"{cur_inst_tag_a}", cur_inst_tag_b, modified_data_sv_root=modified_data_sv_root, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf)
    exit(0)
    
    
    
    
    # retar_info_fn = "/home/xymeow/xueyi/IsaacGymEnvs2/isaacgymenvs/data/statistics/GRAB_Tracking_PK_OFFSET_warm/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    # retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    # hand_qs = retar_info["robot_delta_states_weights_np"]
    # init_hand_qs = hand_qs[0]
    # init_hand_qs = torch.from_numpy(init_hand_qs).float()
    # print(init_hand_qs[:7])
    # # print(init_hand_qs)
    # exit(0)
    
    
    urdf_fn = "../assets/allegro_hand_description/urdf/allegro_hand_description_right_fly_v2.urdf"
    # urdf_fn = "../../tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
    retar_info_fn = "/home/xymeow/xueyi/IsaacGymEnvs2/isaacgymenvs/data/GRAB_Tracking_PK_OFFSET_Reduced/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    retar_info_fn = "/home/xymeow/xueyi/IsaacGymEnvs2/isaacgymenvs/data/GRAB_Tracking_PK_OFFSET_Reduced/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300_ori.npy"
    retar_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d4/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    retar_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d6_0d4/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy'
    retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    hand_qs = retar_info["robot_delta_states_weights_np"]
    # print(retar_info.keys())
    retar_info["robot_delta_states_weights_np"] = hand_pk_for_palm_link_pose(urdf_fn, hand_qs, to_numpy=True)
    np.save(retar_info_fn, retar_info)
    exit(0)
    
    result_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_knife_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_04-22-32-15/ts_to_hand_obj_obs_reset_1.npy"
    inst_tag = "ori_grab_s2_knife_pass_1"
    extract_kinematics_from_tracked_results(result_fn, inst_tag)
    exit(0)
    
    inst_tag = "ori_grab_s2_knife_pass_1"
    traj_modification_opt(inst_tag)
    exit(0)
    
    hand_type = 'allegro'
    # samples_fn = '/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled/passive_active_info_ori_grab_s2_knife_pass_1_nf_300_sample_0.npy'
    samples_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled/passive_active_info_ori_grab_s2_knife_pass_1_nf_300_sample_14.npy"
    # samples_fn = "/data/xueyi/uni_manip/expanded_kines/ori_grab_s2_knife_pass_1_forecasted_kine_traj_modifications.npy"
    samples_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled/passive_active_info_ori_grab_s2_knife_pass_1_nf_300_sample_15.npy"
    
    samples_fn="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled/passive_active_info_ori_grab_s2_knife_pass_1_nf_300_sample_55.npy"
    samples_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled/passive_active_info_ori_grab_s2_knife_pass_1_nf_300_sample_68.npy"
    
    # samples_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_knife_pass_1_nf_300.npy"
    pure_obj_type = 'ori_grab_s2_knife_pass_1'
    debug = True
    debug = False
    converted_hand_verts, transformed_obj_verts, tot_ts_tot_link_idxes, tot_body_contact_flags = transform_hand_verts_kinematics(hand_type, samples_fn , pure_obj_type, w_arm=False, debug=debug)
    # converted_hand_verts =
    sv_dict = {
        'hand_verts': converted_hand_verts, # # 
        'obj_verts': transformed_obj_verts, # # 
        'hand_body_idxes': tot_ts_tot_link_idxes,
        'body_contact_flags': tot_body_contact_flags
    }
    sv_converted_samples_fn = samples_fn.split("/")[-1].split(".")[0] + "_converted.npy"
    samples_fn_root = "/".join(samples_fn.split("/")[:-1])
    sv_converted_samples_fn = os.path.join(samples_fn_root, sv_converted_samples_fn)
    np.save(sv_converted_samples_fn, sv_dict)
    print(f"sv_converted_samples_fn: {sv_converted_samples_fn}")
    exit(0)
    
    
    hand_type = 'allegro' # 
    samples_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_airplane_lift_nf_300.npy"
    
    retargeted_info_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/"
    tot_samples_fns = os.listdir(retargeted_info_sv_root)
    tot_samples_fns = [cur_fn for cur_fn in tot_samples_fns if "passive_active_info" in cur_fn and cur_fn.endswith(".npy")]
    for cur_samples_fn in tot_samples_fns:
        samples_fn = pjoin(retargeted_info_sv_root, cur_samples_fn)
    
        w_arm = False
        bsz_idx = 0
        bsz_idx = 1 # find the bsz index #
        debug = False
        # debug = True
        # tranformed_verts, gt_verts = # transformed verts #
        converted_hand_verts, transformed_obj_verts, tot_ts_tot_link_idxes, tot_body_contact_flags = transform_hand_verts_kinematics(hand_type, samples_fn, bsz_idx=bsz_idx, debug=debug)
        sv_dict = {
            'hand_verts': converted_hand_verts, # 
            'obj_verts': transformed_obj_verts, # 
            'hand_body_idxes': tot_ts_tot_link_idxes, # link idxes # # link idxes #
            'body_contact_flags': tot_body_contact_flags
        }
        
        contact_flag_sv_folder = f"/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag"
        os.makedirs(contact_flag_sv_folder, exist_ok=True)
        contact_falg_sv_fn = samples_fn.split("/")[-1].split(".")[0] + "_contact_flag.npy"
        contact_falg_sv_fn = pjoin(contact_flag_sv_folder, contact_falg_sv_fn)
        np.save(contact_falg_sv_fn, tot_body_contact_flags)
    
    # np.save(f"sv_dict.npy", sv_dict)
    exit(0)
    
    
    hand_type = 'allegro'
    
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_/samples000070000.npy"
    # w_arm = False
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v2/samples000380001.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v3/samples000280001.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v4/samples000240001.npy"
    # tranformed_verts, gt_verts = transform_hand_verts_AE_Diff(hand_type, samples_fn, w_arm=False )
    # sv_dict = {
    #     'pred_verts': tranformed_verts,
    #     'gt_verts': gt_verts,
    # }
    # np.save(f"allegro_tracking_kine_diff_AE_Diff_traj_forcasting.npy", sv_dict)
    # exit(0) # get he modle and forward the hand joints 
    
    hand_type = 'allegro'
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_/samples000070000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_/samples000090000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_/samples000230000.npy"
    w_arm = False
    bsz_idx = 0
    bsz_idx = 1
    tranformed_verts, gt_verts = transform_hand_verts(hand_type, samples_fn, bsz_idx=bsz_idx)
    sv_dict = {
        'pred_verts': tranformed_verts,
        'gt_verts': gt_verts
    }
    np.save(f"allegro_tracking_kine_diff_AE_Diff_traj_forcasting.npy", sv_dict)
    exit(0)
    
    
    # urdf_fn = "/root/diffsim/IsaacGymEnvs2/assets/allegro_hand_description/urdf/allegro_hand_description_right_franka.urdf"
    # hand_qs = np.zeros((60,), dtype=np.float32).reshape(1, -1) # .reshape(1, -1)
    # pts = hand_pk(urdf_fn, hand_qs)
    # np.save(f"franka_w_allegro.npy" , pts)
    # exit(0)
    
    # urdf_fn = "/home/xueyi/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v2.urdf"
    # urdf_fn = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly_v2.urdf"
    # urdf_fn = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right_fly.urdf"
    # urdf_fn = "/home/xueyi/diffsim/tiny-differentiable-simulator/python/examples/rsc/leap_hand/leap_hand_right.urdf"
    
    
    retar_info_fn  = '/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_20231104_203_zrot_3.141592653589793_modifed_interped.npy'
    hand_type = 'allegro'
    w_arm = False
    w_arm = True 
    
    retar_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka/data/passive_active_info_ori_grab_s1_binoculars_offhand_1_nf_300.npy'
    retar_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka_v2/data/passive_active_info_ori_grab_s1_bowl_drink_1_nf_300.npy'
    # retar_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka_v2/data/passive_active_info_ori_grab_s1_stapler_pass_1_nf_300.npy'
    tot_converted_hand_verts = add_transformed_hand_verts(hand_type, retar_info_fn, w_arm=w_arm)
    retargeted_pts_fn = pjoin("./assets", "retargeted_pts.npy")
    np.save(retargeted_pts_fn, tot_converted_hand_verts)
    exit(0)
    
    
    urdf_fn = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
    
    # retar_info_fn = "/cephfs/yilaa/data/GRAB_Tracking/data/leap_passive_active_info_taco_20231104_016.npy"
    retar_info_fn = "/cephfs/yilaa/data/GRAB_Tracking/data/passive_active_info_ori_grab_s2_mug_pass_1.npy"
    retar_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka/data/passive_active_info_ori_grab_s1_binoculars_offhand_1_nf_300.npy'
    retar_info_hand_qs, passive_meshes = inspect_retar_info_fn(retar_info_fn)
    tot_retar_hand_verts = hand_pk(urdf_fn, retar_info_hand_qs)
    print(tot_retar_hand_verts.shape)
    tot_retar_hand_verts_sv_fn = pjoin("./assets", "tot_retar_hand_verts.npy")
    tot_retar_hand_verts = {
        'hand_verts': tot_retar_hand_verts,
        'obj_verts': passive_meshes
    }
    np.save(tot_retar_hand_verts_sv_fn, tot_retar_hand_verts)
    
    # transformed_verts = test_pk(urdf_fn=urdf_fn)
    
    # asset_root = './assets'
    # verts_sv_fn = "transformed_verts.npy"
    # verts_sv_fn = pjoin(asset_root, verts_sv_fn)
    # np.save(verts_sv_fn, transformed_verts)
    # print(f"vertices saved to {verts_sv_fn}")
    # 
    
    
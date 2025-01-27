import torch

import pytorch_kinematics as pk

# import os.path.join as pjoin

from os.path import join as pjoin

import trimesh

import numpy as np

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

def hand_pk(urdf_fn, hand_qs):
    
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
    
    hand_qs = torch.from_numpy(hand_qs).double().to(d)[:, :nn_hand_dof]
    
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
    
    for i_ts in range(th_batch.size(0)):
        # if i_ts >= 5:
        #     break
        tot_meshes_verts = []
        # for key in tg_batch.keys():
        for key in allegro_link_names:
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
                verts = verts.cpu()
                verts = m_offset.transform_points(verts)
                # 
                transformed_verts = torch.matmul(verts.detach().cpu(), rot.transpose(0,1).detach().cpu()) + pos.detach().cpu().unsqueeze(0)
                # transform #
                print(f"transformed_verts: {transformed_verts.size()}")
                tot_meshes_verts.append(transformed_verts)
        transformed_verts = torch.cat(tot_meshes_verts, dim=0)
        transformed_verts = transformed_verts.detach().cpu().numpy()
        # print(transformed_verts.shape)
        tot_hand_verts.append(transformed_verts)
    tot_hand_verts = np.stack(tot_hand_verts, axis= 0)
    return tot_hand_verts
    
    
    

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
    training_samples = samples['closest_training_data']
    print(f"training_samples: {training_samples.keys()}")
    gt_future_hand_pose = training_samples['future_hand_pose']
    samples = samples['samples']
    hand_pose = samples['hand_pose']
    print(f"hand_pose: {hand_pose.shape}, gt_future_hand_pose: {gt_future_hand_pose.shape}")
    
    
    hand_pose = hand_pose[bsz_idx]
    gt_future_hand_pose = gt_future_hand_pose[bsz_idx]
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
    # retar_info['ts_to_allegro'] = tot_converted_hand_verts
    # np.save(retar_info_fn, retar_info)
    return tot_converted_hand_verts, gt_verts

def transform_hand_verts_AE_Diff(hand_type, samples_fn, w_arm=False, i_bsz=0):
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
    
    # i_bsz = 0
    # i_bsz = 1
    
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





# python utils/test_pk.py

if __name__=='__main__':
    hand_type = 'allegro'
    
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_/samples000070000.npy"
    # w_arm = False
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v2/samples000380001.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v3/samples000280001.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v4/samples000240001.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2/samples003540000.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws100_v2/samples002160000.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_histindex_v2/samples005520001.npy"
    # # samples-fn #
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws300_v2/samples000220000.npy"
    # # samples fn #
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_v2_partialhand_condhandonly_/samples005040002.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_task_cond_ws60_stedgoalcond_/samples005400001.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_centralize_partialhandinfo_/samples000240000.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_centralize_handobjinfo_/samples000890001.npy"
    # tranformed_verts, gt_verts = transform_hand_verts_AE_Diff(hand_type, samples_fn, w_arm=False )
    # sv_dict = {
    #     'pred_verts': tranformed_verts,
    #     'gt_verts': gt_verts,
    # }
    # np.save(f"allegro_tracking_kine_diff_AE_Diff_traj_forcasting.npy", sv_dict)
    # exit(0)

    hand_type = 'allegro' # allegro and the allegro #
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_/samples000070000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_/samples000090000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_/samples000230000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws100_histindex_/samples000580000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws120_histindex_/samples000410000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws130_histindex_/samples000370000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws60_v2/samples003540000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_ws300_v2/samples000220000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_traj_forcasting_ws60_centralize_handobjinfo_/samples000030000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_traj_forcasting_ws60_centralize_handobjinfo_/samples000130000.npy"
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_traj_forcasting_ws60_centralize_partialobjposinfo_wglbfeat_wclipglbfeat_/samples000180000.npy"
    
    
    # 40 #
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
    
    
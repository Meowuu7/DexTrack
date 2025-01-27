import torch

import pytorch_kinematics as pk

# import os.path.join as pjoin

from os.path import join as pjoin

import trimesh

import numpy as np

from scipy.spatial.transform import Rotation as R
import os

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

def build_chain(urdf_fn):
    chain = pk.build_chain_from_urdf(open(urdf_fn).read()) # build chain from urdf #
    # chain = chain.to(dtype=dtype, device=d)
    # chain = chain.to(dtype=torch.float32).cuda() # to chaiin and the cuda #
    chain = chain.to(dtype=torch.float32, device=d)
    print(f"chain device: {d }")
    
    # self.mesh_root = '/'.join(model_path.split('/')[:-1])
    # self.chain = chain ## get chain ##
    
    # nn_hand_dof = len(self.chain.get_joint_parameter_names())
    # self.nn_hand_dof = nn_hand_dof
    return chain


# calculate the forward kinematics for posed verts #
def calculate_forward_kinematics_posed_verts(urdf_fn, hand_qs, hand_type='allegro'):
    chain = build_chain(urdf_fn=urdf_fn)
    #
    hand_qs_th = torch.from_numpy(hand_qs).float().to(d)
    tg_batch = chain.forward_kinematics(hand_qs_th) # nn_ts x ahdn qs 
    
    if hand_type == 'allegro':
        robot_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']

    link_key_to_vis_mesh = {}
    
    # mesh_root = ""
    # get the mehsh root # 
    mesh_root = "/".join(urdf_fn.split("/")[:-1])
    
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
    
    tot_ts_link_sampled_pts = []
    
    tot_link_sampled_pts = []
    for key in robot_link_names:
        link = chain.find_link(key)
        link_visuals = link.visuals
        
        for cur_visual in link_visuals:
            m_offset = cur_visual.offset
            cur_visual_mesh_fn = cur_visual.geom_param
            m = tg_batch[key].get_matrix()
            pos = m[:, :3, 3].float()
            # rot = pk.matrix_to_quaternion(m[:, :3, :3]) #
            rot = m[:, :3, :3].float()
            
            # pos: nn_ts x 3
            # rot: nn_ts x 3 x 3
            
            # pos = pos[i_ts]
            # rot = rot[i_ts]
            # print(f"pos: {pos}, rot: {rot}, visual_mesh: {cur_visual_mesh_fn}")
            if cur_visual_mesh_fn is None:
                continue
            if isinstance(cur_visual_mesh_fn, tuple):
                cur_visual_mesh_fn = cur_visual_mesh_fn[0]
            
            if key not in link_key_to_vis_mesh:
                verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
                verts = torch.from_numpy(verts).float().cuda() # 
                
                verts = m_offset.transform_points(verts.cpu())
                link_key_to_vis_mesh[key] = verts.cuda() #
                
            verts = link_key_to_vis_mesh[key]
            # verts = load_visual_meshes(mesh_root, cur_visual_mesh_fn)
            # print(f"verts: {verts.size()}, pos: {pos.size()}, rot: {rot.size()}")
            transformed_verts = torch.matmul(verts.unsqueeze(0), rot.transpose(1,2)) + pos.unsqueeze(1)
            # .detach().cpu().unsqueeze(0)
            
            tot_link_sampled_pts.append(transformed_verts)
            
    tot_link_sampled_pts = torch.cat(tot_link_sampled_pts, dim=1) # 
    # tot_ts_link_sampled_pts.append(tot_link_sampled_pts)
    # tot_ts_link_sampled_pts = torch.stack(tot_ts_link_sampled_pts, dim=0)
    # 
    # nn_ts x (nn_verts) x 3 #
    tot_link_sampled_pts_np = tot_link_sampled_pts.detach().cpu().numpy()
    
    return tot_link_sampled_pts_np


def inspect_differences_between_qs(sampled_qs, training_qs):
    print(sampled_qs.shape, training_qs.shape)
    sampled_glb_qs = sampled_qs[:, :3]
    training_glb_qs = training_qs[:, :3]
    sampled_glb_rot_qs = sampled_qs[:, 3:6]
    training_glb_rot_qs = training_qs[:, 3:6]
    sampled_glb_finger_qs = sampled_qs[:, 6: ]
    training_glb_finger_qs = training_qs[:, 6: ]
    diff_glb_trans_qs = np.abs(sampled_glb_qs - training_glb_qs).mean()
    diff_glb_rot_qs = np.abs(sampled_glb_rot_qs - training_glb_rot_qs).mean()
    diff_glb_finger_qs = np.abs(sampled_glb_finger_qs - training_glb_finger_qs).mean()
    print(f"diff_glb_trans_qs: {diff_glb_trans_qs}, diff_glb_rot_qs: {diff_glb_rot_qs}, diff_glb_finger_qs: {diff_glb_finger_qs}")
    return diff_glb_trans_qs, diff_glb_rot_qs, diff_glb_finger_qs

def load_optimized_info(subj_tag='s2'):
    optimized_info_fn = "/cephfs/xueyi/uni_manip/tds_rl_exp_ctlfreq_10_rew_v2new_pkretar_/logs_PPO/statistics/task_setting_to_optimized_res.npy"
    optimized_info = np.load(optimized_info_fn, allow_pickle=True).item()
    # ge tthe optmized infosj# 
    succ_rew_threshold = 50.0
    subj_tag = "_" + subj_tag + "_"
    tot_inst_ctl_seq = []
    for grab_inst_tag in optimized_info:
        cur_opt_res_fn = optimized_info[grab_inst_tag][0][0]
        cur_opt_res_rew_val = optimized_info[grab_inst_tag][0][1] # 
        
        if cur_opt_res_rew_val < succ_rew_threshold:
            continue
        last_folder_fn = cur_opt_res_fn.split("/")[-2]
        if subj_tag not in last_folder_fn:
            continue
        # subj tag in last_folder_fn # 
        cur_data = np.load(cur_opt_res_fn, allow_pickle=True).item()
        if 'ts_to_optimized_q_tars_wcontrolfreq' in cur_data:
            ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_optimized_q_tars_wcontrolfreq']
        else:
            ts_to_optimized_q_tars_wcontrolfreq = cur_data['ts_to_hand_tot_ctl_qtars']
        # ts
        tot_ts_keys = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
        sorted_ts = sorted(tot_ts_keys)
        cur_inst_ctl_seq = []
        for cur_ts in sorted_ts:
            cur_ts_qtar = ts_to_optimized_q_tars_wcontrolfreq[cur_ts]
            cur_inst_ctl_seq.append(cur_ts_qtar)
        cur_inst_ctl_seq = np.stack(cur_inst_ctl_seq, axis=0) ### nn_ts x nn_qtar_dof
        # cur_inst_ctl_seq 
        tot_inst_ctl_seq.append(cur_inst_ctl_seq)
    tot_inst_ctl_seq = np.stack(tot_inst_ctl_seq, axis=0) ## get the tot inst ctl seq ## ## nn_instances x nn_ts x nn_qtar_odf # 
    return tot_inst_ctl_seq

def find_nearest_ctl_seq(samples_fn, tot_optimized_qs):
    hand_qs_samples = np.load(samples_fn, allow_pickle=True).item()
    optimized_qtars = hand_qs_samples['samples']
    optimized_qtars = optimized_qtars['tot_verts_integrated_qdd_tau'] # nn_bsz x nn_ts x nn_qs
    training_qtars = hand_qs_samples['closest_training_data']
    training_qtars = training_qtars['tot_verts_integrated_qdd_tau']
    for i_bsz in range(optimized_qtars.shape[0]):
        cur_bsz_optimized_qtars = optimized_qtars[i_bsz] # nn_ts x nn_qs # 
        cur_bsz_training_qtars = training_qtars[i_bsz]
        
        
        dist_qs = np.sum(
            (cur_bsz_optimized_qtars[None] - tot_optimized_qs) ** 2, axis=-1
        )
        dist_qs = np.mean(dist_qs, axis=1)
        minn_dist_qs = np.argmin(dist_qs)
        minn_dist_qs = minn_dist_qs.item() # minn_dist_qs -- # 
        # get the minn dist qs # 
        minn_dist_val = dist_qs[minn_dist_qs] 
        print(f"i_bsz: {i_bsz}, minn_dist_qs: {minn_dist_qs}, minn_dist_val: {minn_dist_val}")
        
        
        dist_qs = np.sum(
            (cur_bsz_training_qtars[None] - tot_optimized_qs) ** 2, axis=-1
        )
        dist_qs = np.mean(dist_qs, axis=1)
        minn_dist_qs = np.argmin(dist_qs)
        minn_dist_qs = minn_dist_qs.item() # minn_dist_qs -- # 
        # get the minn dist qs # 
        minn_dist_val = dist_qs[minn_dist_qs] 
        print(f"[training sample] i_bsz: {i_bsz}, minn_dist_qs: {minn_dist_qs}, minn_dist_val: {minn_dist_val}")

def compare_res_w_best_res():
    merged_traj='/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0_merged.npy'
    gt_traj='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-23-46-55/ts_to_hand_obj_obs_reset_1_sorted.npy'
    gt_info = np.load(gt_traj, allow_pickle=True).item()
    gt_qtars = gt_info['optimized_hand_qtars']
    gt_qtars = gt_qtars[0]
    merged_info = np.load(merged_traj, allow_pickle=True).item()
    merged_training_qtars = merged_info['closest_training_data'][0]
    
    # gt_qtars = gt_qtars[0]
    print(gt_qtars.shape, merged_training_qtars.shape)
    cur_bsz = min(gt_qtars.shape[0], merged_training_qtars.shape[0])
    gt_qtars = gt_qtars[1: cur_bsz + 1]
    merged_training_qtars = merged_training_qtars[0: cur_bsz ]
    diff_gt_qtars_training_qtars = np.abs(gt_qtars - merged_training_qtars).sum(axis=-1).mean()
    print(f"diff_gt_qtars_training_qtars: {diff_gt_qtars_training_qtars}")

def inspect_samples(samples_fn):
    samples = np.load(samples_fn, allow_pickle=True).item()
    print(samples.keys())    
    samples = samples['samples']
    print(samples.keys())
    for key in samples:
        val = samples[key]
        if isinstance(val, np.ndarray):
            print(f"key: {key}, val: {val.shape}")



# transformed_dict = convert_kinematics_samples(samples_fn)
def convert_kinematics_samples(samples_fn):
    
    urdf_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc"
    urdf_fn = "allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
    urdf_fn = os.path.join(urdf_root, urdf_fn)
    
    
    
    samples = np.load(samples_fn, allow_pickle=True).item()
    samples = samples['samples']
    
    nn_max_insts = 10
    nn_max_insts = min(nn_max_insts, samples['hand_qs'].shape[0])
    
    hand_qs = samples['hand_qs'][:nn_max_insts]
    # [0] ## hand qs from the samples -- nn_bsz x nn_ts x nn_hand_dofs # 
    cond_hand_qs = samples['cond_hand_qs'][:nn_max_insts] if 'cond_hand_qs' in samples else samples['hand_qs'][:nn_max_insts]
    obj_verts = samples['obj_verts']
    cond_obj_verts = samples['cond_obj_verts'] if 'cond_obj_verts' in samples else samples['obj_verts']
    obj_transl = samples['obj_transl'][:nn_max_insts]
    obj_rot_euler  =    samples['obj_rot_euler'][:nn_max_insts]
    cond_obj_transl = samples['cond_obj_transl'][:nn_max_insts]  if 'cond_obj_transl' in samples else samples['obj_transl'][:nn_max_insts]
    cond_obj_rot_euler = samples['cond_obj_rot_euler'][:nn_max_insts]  if 'cond_obj_rot_euler' in samples else samples['obj_rot_euler'][:nn_max_insts]
    
    nn_bsz = hand_qs.shape[0]
    
    tot_hand_qs_pts_np = []
    tot_cond_hand_qs_pts_np = []
    tot_transformed_obj_verts = []
    tot_transformed_cond_obj_verts = []
    
    for i_bsz in range(nn_bsz): # 
        
        cur_bsz_obj_transl = obj_transl[i_bsz]
        cur_bsz_obj_rot_euler = obj_rot_euler[i_bsz]
        cur_bsz_cond_obj_transl = cond_obj_transl[i_bsz]
        cur_bsz_cond_obj_rot_euler = cond_obj_rot_euler[i_bsz]
        
        cur_hand_qs = hand_qs[i_bsz]
        cur_cond_hand_qs = cond_hand_qs[i_bsz]
        hand_qs_pts_np = calculate_forward_kinematics_posed_verts(urdf_fn, cur_hand_qs, hand_type='allegro')
        cond_hand_qs_pts_np = calculate_forward_kinematics_posed_verts(urdf_fn, cur_cond_hand_qs, hand_type='allegro')
        
        
        tot_hand_qs_pts_np.append(hand_qs_pts_np)
        tot_cond_hand_qs_pts_np.append(cond_hand_qs_pts_np)
        
        try:
            
                # hand_qs_pts_np, cond_hand_qs_pts_np #
            obj_verts  = obj_verts[i_bsz]
            cond_obj_verts   =  cond_obj_verts[i_bsz]
            # objverts, cond obj verts #
            # obj verts, cond obj verts #
            transformed_obj_verts = []
            transformed_cond_obj_verts = []
            ### get total obj rot matrix and transform the object verts? #
            
            
            for i_fr in range(cur_bsz_obj_rot_euler.shape[0]):
                cur_obj_rot = cur_bsz_obj_rot_euler[i_fr]
                cur_obj_rot_mtx = R.from_euler('xyz', cur_obj_rot).as_matrix()
                cur_obj_transl = cur_bsz_obj_transl[i_fr]
                cur_obj_verts = obj_verts # [i_fr]
                # transformed_obj_verts.append(curob)
                cur_transformed_obj_verts = np.matmul(cur_obj_rot_mtx, cur_obj_verts.T).T + cur_obj_transl[None]
                transformed_obj_verts.append(cur_transformed_obj_verts)
                
                # transformed_cond_obj_verts = [] #
                # for i_fr in range(cur_cond_obj_rot_euler.shape[0]):
                cur_cond_obj_rot = cur_bsz_cond_obj_rot_euler[i_fr]
                cur_cond_obj_rot_mtx = R.from_euler('xyz', cur_cond_obj_rot, degrees=True).as_matrix()
                cur_cond_obj_transl = cur_bsz_cond_obj_transl[i_fr]
                cur_cond_obj_verts = cond_obj_verts # [i_fr]
                # transformed_obj_verts.append(curob)
                cur_transformed_cond_obj_verts = np.matmul(cur_cond_obj_rot_mtx, cur_cond_obj_verts.T).T + cur_cond_obj_transl[None]
                transformed_cond_obj_verts.append(cur_transformed_cond_obj_verts) # get the transformed cond obj verts 
            
            transformed_obj_verts = np.stack(transformed_obj_verts, axis=0)
            transformed_cond_obj_verts = np.stack(transformed_cond_obj_verts, axis=0)
            
            tot_transformed_obj_verts.append(transformed_obj_verts)
            tot_transformed_cond_obj_verts.append(transformed_cond_obj_verts)
        except:
            continue
    tot_hand_qs_pts_np = np.stack(tot_hand_qs_pts_np, axis=0)
    tot_cond_hand_qs_pts_np = np.stack(tot_cond_hand_qs_pts_np, axis=0)
    if len(tot_transformed_obj_verts) > 0:
        try:
            tot_transformed_obj_verts = np.stack(tot_transformed_obj_verts, axis=0)
            tot_transformed_cond_obj_verts = np.stack(tot_transformed_cond_obj_verts, axis=0)
        except:
            tot_transformed_obj_verts = []
            tot_transformed_cond_obj_verts = []
    transformed_dict = {
        'tot_hand_qs_pts_np': tot_hand_qs_pts_np,
        'tot_cond_hand_qs_pts_np': tot_cond_hand_qs_pts_np,
        'tot_transformed_obj_verts': tot_transformed_obj_verts,
        'tot_transformed_cond_obj_verts': tot_transformed_cond_obj_verts
    }
    return transformed_dict
    
    ## tot link sampled pts np ##
    # tot_link_sampled_pts_np = calculate_forward_kinematics_posed_verts(urdf_fn, hand_qs, hand_type='allegro') #
    # tot_link_training_sampled_pts_np = calculate_forward_kinematics_posed_verts(urdf_fn, training_hand_qtars, hand_type='allegro') #

def save_sampled_parent_task_info(samples_fn, object_type):
    # object type should be  in the format of ori_grab_xxx or taco_xxx 
    samples = np.load(samples_fn, allow_pickle=True).item() # 
    samples = samples['samples']
    
    
    # samples = np.load(samples_fn, allow_pickle=True).item()
    # samples = samples['samples']
    hand_qs = samples['hand_qs'][0] # nn_bsz x nn_ts x nn_hand_dofs #
    cond_hand_qs = samples['cond_hand_qs'][0] # [0]
    obj_verts = samples['obj_verts']
    cond_obj_verts = samples['cond_obj_verts']
    obj_transl = samples['obj_transl'][0]
    obj_rot_euler  =    samples['obj_rot_euler'][0]
    cond_obj_transl = samples['cond_obj_transl'][0]
    cond_obj_rot_euler = samples['cond_obj_rot_euler'][0]
    
    obj_rot_quat = []
    for i_fr in range(obj_rot_euler.shape[0]):
        cur_fr_obj_rot_euler = obj_rot_euler[i_fr]
        cur_fr_obj_rot_struct = R.from_euler('xyz', cur_fr_obj_rot_euler)
        cur_fr_obj_rot_quat =  cur_fr_obj_rot_struct.as_quat()
        obj_rot_quat.append(cur_fr_obj_rot_quat)
    obj_rot_quat = np.stack(obj_rot_quat, axis=0)
    # object_type = "ori_grab_s2_train_lift"
    # hand_qs = cur_kine_data['robot_delta_states_weights_np'] # weights -- kinematics qs #
    # maxx_ws = min(hand_qs.shape[0], self.maxx_kine_nn_ts)
    # hand_qs = hand_qs[:maxx_ws]
    
    # obj_trans = cur_kine_data['object_transl'][:maxx_ws] # nn_ts x 3
    # obj_ornt = cur_kine_data['object_rot_quat'][:maxx_ws] # nn_ts x 4
    
    # # then segment the data_inst_tag to get the mesh file name #
    # self.grab_obj_mesh_sv_folder = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
    # grab_mesh_fn = f"{cur_object_type}.obj"
    kinematics_info_sv_dict = {
        'object_type': object_type,
        'robot_delta_states_weights_np': hand_qs,
        'object_transl': obj_transl,
        'object_rot_quat': obj_rot_quat
    }
    kinematics_info_sv_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK_Sampled"
    kinematics_info_sv_fn = f"{object_type}_sampled_kinematics_info.npy" #
    kinematics_info_sv_fn = os.path.join(kinematics_info_sv_folder, kinematics_info_sv_fn)
    np.save(kinematics_info_sv_fn, kinematics_info_sv_dict)
    print(f"kinematics_info_sv_fn: {kinematics_info_sv_fn}")
    
    



# python utils/translate_samples.py 





if __name__=='__main__':
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_/samples_ep_0_batch_0.npy"
    # #
    
    # object_type = "ori_grab_s2_train_lift" # ori grab train lift
    # save_sampled_parent_task_info(samples_fn, object_type)
    # exit(0)
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_/samples_ep_0_batch_0.npy"
    # # 
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_/samples_ep_0_batch_0.npy"
    # inspect_samples(samples_fn)
    # exit(0)
    
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_/samples_ep_0_batch_0.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_/samples_ep_0_batch_0.npy"
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_newcond_sample_/samples_ep_0_batch_0.npy"
    # # inspect_samples(samples_fn)
    # samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/samples_ep_8_batch_0.npy'
    # samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_kinediffv2_samples_/samples_ep_0_batch_0.npy'
    
    # samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/samples_ep_9_batch_0.npy'
    # samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_canonv2_samples_/samples_ep_9_batch_0.npy'
    # transformed_dict = convert_kinematics_samples(samples_fn)
    # exported_samples_fn = samples_fn.split(".")[0] + "_transformed.npy"
    # np.save(exported_samples_fn, transformed_dict) 
    # print(f"exported_samples_fn: {exported_samples_fn}")
    # # exported samples fn #
    # exit(0)
    
    # compare_res_w_best_res()
    # exit(0) # translate samples #
    
    # tot_optimized_qs = load_optimized_info()
    # print(f"tot_optimized_qs: {tot_optimized_qs.shape}")
    
    
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_/samples000040000.npy"
    
    # find_nearest_ctl_seq(samples_fn, tot_optimized_qs)
    # exit(0)
    
    urdf_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc"
    urdf_fn = "allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
    urdf_fn = os.path.join(urdf_root, urdf_fn)
    
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_si_/samples000030000.npy"
    # hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_taskcond_diffallp_/samples000210000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_taskcond_diffallp_/samples001030000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_taskcond_diffallp_multiinst_/samples001400000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_/samples000410000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_/samples001230100.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_/samples001260000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_/samples001500000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_/samples001590000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_/samples004230000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_fixedorder_/samples000010000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_/samples000040000.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_/samples001180000.npy"
    
    
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_/samples_ep_0_batch_3.npy"
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_allts_fixedorder_/samples_ep_0_batch_0.npy"
    # can we viz the task conditions? #
# <<<<<<< HEAD
    hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_slicing_/samples000690000.npy"
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_slicing_2_/samples_ep_9_batch_1.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_slicing_2_/samples_ep_9_batch_4.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_/samples001110000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_/samples_ep_1_batch_1.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v2_/samples000500000.npy'
    # 
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_15_taskcond_history_future_v2_/samples000570000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v2_/samples000580000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v2_/samples000600000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v2_/samples_ep_1_batch_12.npy'
    ### diff_glb_trans: 0.02031529089435935, diff_glb_rot: 0.15638858557213098, diff_fingers: 0.3298040498048067
    # hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_15_taskcond_history_future_v2_/samples_ep_0_batch_2.npy'
    ### diff_glb_trans: 0.023400450590997934, diff_glb_rot: 0.1990036810748279, diff_fingers: 0.3229549671523273
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v2_/samples_ep_0_batch_0.npy'
    ### diff_glb_trans: 0.027648016912280582, diff_glb_rot: 0.08876958093605936, diff_fingers: 0.2647536469157785
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v2_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v3_/samples004320000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples000750000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    # /cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_1.npy #
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_5_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_2_taskcond_history_future_v4_/samples_ep_0_batch_0.npy'
    
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_1_histws_30_taskcond_history_future_deterministic_/samples001890000.npy'
    hand_qs_samples_fn= '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_1_histws_30_taskcond_history_future_deterministic_/samples000430000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_1_histws_30_taskcond_history_future_deterministic_/samples001920000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_trainDiff_taskcond_isaac_slicing_30_histws_30_step_1_taskcond_history_future_v4_w_tacograbinterp_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_/samples001160000.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_trainDiff_taskcond_vnew_datav2_sample_/samples_ep_0_batch_0.npy'
    hand_qs_samples_fn = '/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_canonv2_samples_/samples_ep_9_batch_0.npy'
    # the major difference is that in the pre-physics step we should re-compute actions using the learned model # # deterministic model --- one step inference # #
    # 
# ========== so i think this one step inference model is important # 
    # sampling and we can have the statistics avilablble#
#     hand_qs_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_task_ctl_diff_wscale_rew_v2_new_s2_AE_Diff_trainAE_slicing_/samples000040000.npy"
    
# >>>>>>> 53ad265dd7b4e01c1105f4f8123ceb095fac3ecf
    hand_qs_samples = np.load(hand_qs_samples_fn, allow_pickle=True).item()
    optimized_hand_qtars = hand_qs_samples['samples']
    training_hand_qtars = hand_qs_samples['closest_training_data']
    
    
    print(f"optimized_hand_qtars: {optimized_hand_qtars.keys()}")
    # quaternion ? # # optimized hand qtars #
    tot_batch_sampled_pts_np = []
    tot_batch_training_pts_np = []
    # tot batch samples pts np # tot batch training pts np #
    
    if 'tot_verts_integrated_qdd_tau' not in optimized_hand_qtars:
        tot_sampled_qtars = optimized_hand_qtars['hand_qs']
        tot_training_hand_qtars = training_hand_qtars['hand_qs']
    else:
        tot_sampled_qtars = optimized_hand_qtars['tot_verts_integrated_qdd_tau']
        tot_training_hand_qtars = training_hand_qtars['tot_verts_integrated_qdd_tau']
    
    print(f"tot_sampled_qtars: {tot_sampled_qtars.shape}, tot_training_hand_qtars: {tot_training_hand_qtars.shape}")
    # exit(0)
    
    #### version 1 of the step size and the window size ####
    window_size = tot_sampled_qtars.shape[1]
    step_size = window_size // 2
    tot_merged_sampled_qtars = []
    tot_merged_training_hand_qtars = []
    for i_bsz in range(tot_sampled_qtars.shape[0]):
        if i_bsz < tot_sampled_qtars.shape[0] - 1:
            cur_sampled_qtars = tot_sampled_qtars[i_bsz, :step_size]
            cur_training_qtars = tot_training_hand_qtars[i_bsz, :step_size]
        else: # 
            cur_sampled_qtars = tot_sampled_qtars[i_bsz]
            cur_training_qtars = tot_training_hand_qtars[i_bsz]
        tot_merged_sampled_qtars.append(cur_sampled_qtars)
        tot_merged_training_hand_qtars.append(cur_training_qtars)
    tot_merged_sampled_qtars = np.concatenate(tot_merged_sampled_qtars, axis=0)
    tot_merged_training_hand_qtars = np.concatenate(tot_merged_training_hand_qtars, axis=0)
    tot_sampled_qtars = tot_merged_sampled_qtars[None]
    tot_training_hand_qtars = tot_merged_training_hand_qtars[None] # initlaize frrom the samples foras the kinematic info? #
    #### version 1 of the step size and the window size ####
    
    merged_dict = {
        'samples': tot_sampled_qtars,
        'closest_training_data': tot_training_hand_qtars
    }
    sv_merged_dict_fn = hand_qs_samples_fn.replace(".npy", "_merged.npy")
    np.save(sv_merged_dict_fn, merged_dict)
    print(f"Merged hand qtars saved to sv_merged_dict_fn: {sv_merged_dict_fn}")
    
    tot_diff_glb_trans, tot_diff_glb_rot, tot_diff_glb_finger = [], [], []
    
    for i_bsz in range(tot_sampled_qtars.shape[0]):
        # bsz x nn_ts x nn_hand_dofs #
        sampled_qtars = tot_sampled_qtars[i_bsz] # 
        
        training_hand_qtars = tot_training_hand_qtars[i_bsz]
        
        ##### 
        diff_glb_trans_qs, diff_glb_rot_qs, diff_glb_finger_qs = inspect_differences_between_qs(sampled_qtars, training_hand_qtars)
        
        tot_diff_glb_trans.append(diff_glb_trans_qs)
        tot_diff_glb_rot.append(diff_glb_rot_qs)
        tot_diff_glb_finger.append(diff_glb_finger_qs)
        
        
        
        # sampled_qtars = np.concatenate(
        #     [
        #         training_hand_qtars[:, :3], training_hand_qtars[:, 3:6], sampled_qtars[:, 6:]
        #     ], axis=1
        # )
        
        hand_qs = sampled_qtars
        # othe #
        tot_link_sampled_pts_np = calculate_forward_kinematics_posed_verts(urdf_fn, hand_qs, hand_type='allegro')
        
        tot_link_training_sampled_pts_np = calculate_forward_kinematics_posed_verts(urdf_fn, training_hand_qtars, hand_type='allegro') # 
        
        hand_qs_samples_root = "/".join(hand_qs_samples_fn.split("/")[:-1])
        hand_qs_samples_pure_fn = hand_qs_samples_fn.split("/")[-1].split(".")[0]
        exported_pts_pure_fn = hand_qs_samples_pure_fn + "_exported_pts.npy"
        exported_pts_pure_fn = os.path.join(hand_qs_samples_root, exported_pts_pure_fn)
        
        tot_batch_sampled_pts_np.append(tot_link_sampled_pts_np)
        tot_batch_training_pts_np.append(tot_link_training_sampled_pts_np)

    tot_diff_glb_trans = sum(tot_diff_glb_trans) / float(len(tot_diff_glb_trans))
    tot_diff_glb_rot = sum(tot_diff_glb_rot) / float(len(tot_diff_glb_rot))
    tot_diff_glb_finger = sum(tot_diff_glb_finger) / float(len(tot_diff_glb_finger))
    print(f"diff_glb_trans: {tot_diff_glb_trans}, diff_glb_rot: {tot_diff_glb_rot}, diff_fingers: {tot_diff_glb_finger}")
    
    
    tot_batch_sampled_pts_np = np.stack(tot_batch_sampled_pts_np, axis=0)
    tot_batch_training_pts_np = np.stack(tot_batch_training_pts_np, axis=0)
    
    hand_qs_samples.update(
        {
            'exported_pts': tot_batch_sampled_pts_np[:10],
            'training_exported_pts': tot_batch_training_pts_np[:10],
        }
    )
    np.save(exported_pts_pure_fn, hand_qs_samples)
    print(f"exported_pts_pure_fn: {exported_pts_pure_fn}") # exported pts pure fn # export pts pure fn #
    

        
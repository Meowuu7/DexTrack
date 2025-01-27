


import numpy as np
from scipy.spatial.transform import Rotation as R
import torch
import pytorch_kinematics as pk
d = "cuda" if torch.cuda.is_available() else "cpu"
import os
# import numpy as np
# from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



@torch.jit.script
def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat


@torch.jit.script
def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)




def find_grasp_frame_from_mocap_data_objpose(obj_transl, obj_ornt):
    # data_dict = np.load(mocap_data_fn, allow_pickle=True).item()
    # hand_qs = data_dict['robot_delta_states_weights_np']
    # obj_transl = data_dict['object_transl'][:]
    # obj_ornt = data_dict['object_rot_quat'][: ]
    # nn_frames x 3 
    # nn_frames x 4
    eps = 1e-2
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


def test_ref_fn(ref_fn, reversed=False):
    ref_data = np.load(ref_fn , allow_pickle=True).item()
    print(ref_data.keys())
    
    obj_transl = ref_data['object_transl']
    obj_rot_quat = ref_data['object_rot_quat']
    hand_qs = ref_data['robot_delta_states_weights_np'] # the hand qs # 
    
    if reversed:
        hand_qs = hand_qs[::-1]
        obj_transl = obj_transl[::-1] # transl 
        obj_rot_quat = obj_rot_quat[::-1]
        
    
    grasp_fr =  find_grasp_frame_from_mocap_data_objpose(obj_transl, obj_rot_quat)
    # grasp fr #
    grasp_fr_pos = hand_qs[grasp_fr, :].copy()
    canonical_grasp_pos =  hand_qs[grasp_fr, :].copy()
    canonical_grasp_pos[2] = canonical_grasp_pos[2] + 0.05
    canonical_grasp_pos[6:] = 0.0
    # canonical grasp pos # 
    # ge the grasp frame # 
    # try the linear inter poloation first ? # 
    delta_pose_diff = (grasp_fr_pos - canonical_grasp_pos) / float(grasp_fr)
    tot_pre_grasp_poses = []
    for i in range(0, grasp_fr):
        cur_hand_pose = delta_pose_diff * i + canonical_grasp_pos
        tot_pre_grasp_poses.append(cur_hand_pose)
    tot_pre_grasp_poses = np.stack(tot_pre_grasp_poses, axis=0)
    # tot pre grasp pose #
    modifed_hand_qs = np.concatenate(
        [tot_pre_grasp_poses, hand_qs[grasp_fr:, :] ], axis=0
    )
    ref_data['robot_delta_states_weights_np'] = modifed_hand_qs
    ref_data['object_rot_quat'] = obj_rot_quat
    ref_data['object_transl'] = obj_transl
    
    ref_fn = ref_fn.replace(".npy", "_modifed.npy") # 
    np.save(ref_fn, ref_data)
    print(f"reference data saved to {ref_fn}")
    
    
def linear_interp_data(ori_data, nn_interp_freq):
    interped_data = []
    for i_fr in range(ori_data.shape[0] - 1):
        nex_fr = i_fr + 1
        cur_hand_q = ori_data[i_fr]
        nex_hand_q = ori_data[nex_fr] 
        for i_interp in range(nn_interp_freq):
            cur_subfr_hand_q = (nex_hand_q - cur_hand_q) * float(i_interp) / float(nn_interp_freq) + cur_hand_q
            interped_data.append(cur_subfr_hand_q)
    interped_data.append(ori_data[-1])
    interped_data = np.stack(interped_data, axis=0)
    return interped_data



def interp_slerp(ori_data, nn_interp_freq):
    ori_ts = range(0, ori_data.shape[0])
    ori_ts = list(ori_ts) 
    ## origian ts ## 
    interped_ts = []
    for i_ts in range(len(ori_ts) - 1):
        nex_i_ts = i_ts + 1
        for i_interp in range(nn_interp_freq):
            cur_substep_ts = float(i_ts) + (float(nex_i_ts) - float(i_ts)) * float(i_interp) / float(nn_interp_freq)
            interped_ts.append(cur_substep_ts)
    interped_ts.append(ori_ts[-1])
    ori_data_struct = R.from_quat(ori_data)
    slerp = Slerp(ori_ts, ori_data_struct)
    interp_rots = slerp(interped_ts)
    interp_rots = interp_rots.as_quat()
    
    return interp_rots
        
def interpolate_hand_poses(ref_fn, nn_interp_freq=4):
    ref_data = np.load(ref_fn, allow_pickle=True).item()
    hand_qs = ref_data['robot_delta_states_weights_np']
    obj_transl = ref_data['object_transl']
    obj_rot_quat = ref_data['object_rot_quat']
    # [hand_qs[0], hadn_qs[1]] --- [hand_qs[0], 0.24, 0.5, 0.75]
    # nn_interp_freq =4
    interped_hand_qs = linear_interp_data(hand_qs, nn_interp_freq)
    interped_obj_transl = linear_interp_data(obj_transl, nn_interp_freq)

    interped_rots = interp_slerp(obj_rot_quat, nn_interp_freq) # ge the rot quat and the nn interp freq # 
    
    interped_ref_data = {
        'robot_delta_states_weights_np': interped_hand_qs,
        'object_transl': interped_obj_transl,
        'object_rot_quat': interped_rots
    }
    ref_fn = ref_fn.replace(".npy", "_interped.npy")
    np.save(ref_fn, interped_ref_data)
    print(f"interped data saved to {ref_fn}")
    
    
    # tot_hand_qs = []
    # nn_interp_freq =4
    # for i_fr in range(hand_qs.shape[0] - 1):
    #     nex_fr = i_fr + 1
    #     cur_hand_q = hand_qs[i_fr]
    #     nex_hand_q = hand_qs[nex_fr] 
    #     for i_interp in range(nn_interp_freq):
    #         cur_subfr_hand_q = (nex_hand_q - cur_hand_q) * float(i_interp) / float(nn_interp_freq) + cur_hand_q
    #         tot_hand_qs.append(cur_subfr_hand_q)
    # tot_hand_qs.append(hand_qs[-1])
    # tot_hand_qs = np.stack(tot_hand_qs, axis=0)
    # 
    
def get_totest_taco_files(taco_data_root ):
    tot_taco_fns = os.listdir(taco_data_root)
    tot_taco_fns = [fn for fn in tot_taco_fns if fn.endswith(".npy")]
    modified_fn_tag, interped_fn_tag = "_modifed.npy", "_interped.npy"
    # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_028_v2_interpfr_60_interpfr2_60_nntrans_40.npy
    tot_taco_fns = [fn for fn in tot_taco_fns if modified_fn_tag not in fn and interped_fn_tag not in fn]
    taco_st_tag = "passive_active_info_taco_"
    tot_taco_fns = [fn for fn in tot_taco_fns if fn[:len(taco_st_tag)] == taco_st_tag]
    print(len(tot_taco_fns))
    return tot_taco_fns


    
def get_totest_taco_files_new(taco_data_root ):
    tot_taco_fns = os.listdir(taco_data_root)
    tot_taco_fns = [fn for fn in tot_taco_fns if fn.endswith(".npy")]
    modified_fn_tag, interped_fn_tag = "_modifed.npy", "_interped.npy"
    # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_028_v2_interpfr_60_interpfr2_60_nntrans_40.npy
    tot_taco_fns = [fn for fn in tot_taco_fns if 'v2_interpfr_60_interpfr2_60_nntrans_40' in fn and '_20231024_' in fn]
    # taco_st_tag = "passive_active_info_taco_"
    # tot_taco_fns = [fn for fn in tot_taco_fns if fn[:len(taco_st_tag)] == taco_st_tag]
    print(len(tot_taco_fns))
    return tot_taco_fns


def get_tot_grab_files_new(grab_data_root):
    # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data
    tot_fns= os.listdir(grab_data_root)
    tot_fns = [fn for fn in tot_fns if 'passive_' in fn and 'active_' in fn and '_s1_' in fn]
    print(len(tot_fns))


def inspect_saved_retar_info(retar_info_fn):
    retar_info_dict = np.load(retar_info_fn, allow_pickle=True).item()
    print(retar_info_dict.keys())
    for key in retar_info_dict:
        cur_val = retar_info_dict[key]
        print(f"key: {key}, val: {cur_val.shape}")
    # obj_transl = retar_info_dict['object_transl']

    # for i_fr in range(obj_transl.shape[0]):
    #     print(f"i_fr: {i_fr}, obj_trans: {obj_transl[i_fr]}")
        
def transform_retar_info(retar_info_dict_fn):
    retar_info_dict = np.load(retar_info_dict_fn, allow_pickle=True).item()
    for key in retar_info_dict:
        val = retar_info_dict[key]
        print(f"key: {key}, val: {val.shape}")
    hand_qs = retar_info_dict['robot_delta_states_weights_np']
    obj_transl = retar_info_dict['object_transl']
    obj_rot_quat = retar_info_dict['object_rot_quat']
    transformed_hand_qs_rot = []
    
    for i_fr in range(hand_qs.shape[0]):
        cur_qs_rot = hand_qs[i_fr, 3:6]
        # cur_qs_rot[2] = cur_qs_rot[2]  + np.pi
        
        cur_hand_trans = hand_qs[i_fr, :3]
        
        
        delta_rot_struct = R.from_euler('zyx', [ np.pi, 0, 0])
        delta_hand_rot_struct = R.from_euler('zyx', [np.pi, 0, 0])
        ##
        
        cur_hand_trans = np.matmul( delta_hand_rot_struct.as_matrix(), cur_hand_trans[:, None])[:, 0]
        
        hand_qs[i_fr, :3] =cur_hand_trans
        
        cur_qs_rot_struct =   R.from_euler('zyx', cur_qs_rot[[2,1,0]]) #  * R.from_euler('zyx', [np.pi, 0, 0])
        
        cur_qs_rot_struct = np.matmul(
            cur_qs_rot_struct.as_matrix() , delta_rot_struct.as_matrix()
        )
        
        cur_qs_rot_struct = R.from_matrix(cur_qs_rot_struct)
        
        cur_qs_rot_xyz = cur_qs_rot_struct.as_euler('xyz', degrees=False) # degrees is False -- convert the rotation struct to eulers #
        # cur_qs_rot_xyz = cur_qs_rot_zyx[[2, 1, 0]]
        
        cur_qs_rot[1] = cur_qs_rot[1] + np.pi
        while cur_qs_rot[1] > np.pi:
            cur_qs_rot[1] -= 2*np.pi
        cur_qs_rot_xyz = cur_qs_rot
        print(hand_qs[i_fr, 3:6], cur_qs_rot_xyz)
        transformed_hand_qs_rot.append(cur_qs_rot_xyz)
        
        
        cur_obj_rot_quat = obj_rot_quat[i_fr]
        
        cur_obj_rot_quat = R.from_quat(cur_obj_rot_quat)
        
        # cur_obj_rot_quat =  cur_obj_rot_quat * R.from_euler('zyx', [np.pi, 0, 0])
        
        # cur_qs_rot_struct =  R.from_euler('xyz', cur_qs_rot) #  * R.from_euler('zyx', [np.pi, 0, 0])
        
        cur_obj_rot_quat = np.matmul(
            delta_rot_struct.as_matrix(), cur_obj_rot_quat.as_matrix() # .T
        )
        
        cur_obj_rot_quat = R.from_matrix(cur_obj_rot_quat)
        
        
        cur_obj_trans = obj_transl[i_fr]
        cur_obj_trans = np.matmul( delta_rot_struct.as_matrix(), cur_obj_trans[:, None])[:, 0]
        obj_transl[i_fr] = cur_obj_trans
        
        cur_obj_rot_quat = cur_obj_rot_quat.as_quat()
        obj_rot_quat[i_fr] = cur_obj_rot_quat
    transformed_hand_qs_rot = np.stack(transformed_hand_qs_rot, axis=0)
    hand_qs[:, 3:6] = transformed_hand_qs_rot
    retar_info_dict['robot_delta_states_weights_np'] = hand_qs
    retar_info_dict['object_transl'] = obj_transl
    
    retar_info_dict['object_rot_quat'] = obj_rot_quat   
    new_sv_info_fn = f"{retar_info_dict_fn[:-4]}_transformed.npy"
    np.save(new_sv_info_fn, retar_info_dict)
    print(  f"Saved the transformed retar info to: {new_sv_info_fn}")
    
    
def get_body_poses(retar_info_fn):
    retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    hand_qs = retar_info['robot_delta_states_weights_np']
    obj_transl = retar_info['object_transl']
    
    robot_link_names = ['link_3_tip', 'link_3', 'link_2', 'link_1', 'link_0', 'link_7_tip', 'link_7', 'link_6', 'link_5', 'link_4', 'link_11_tip', 'link_11', 'link_10', 'link_9', 'link_8', 'link_15_tip', 'link_15', 'link_14', 'link_13', 'link_12', 'palm_link']
    
    hand_qs_th = torch.from_numpy(hand_qs).float().to(d)
    
    allegro_model_path = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf"
    # allegro_model_path = 
    
    chain = pk.build_chain_from_urdf(open(allegro_model_path).read())
    
    
    # chain = chain.to(dtype=dtype, device=d)
    # chain = chain.to(dtype=torch.float32).cuda() # to chaiin and the cuda #
    chain = chain.to(dtype=torch.float32, device=d)
    print(f"chain device: {d }")
    
    tg_batch = chain.forward_kinematics(hand_qs_th) ## get the robot link states ## 
    
    tot_link_sampled_pts = [ ]
    
    link_name_to_poses = {}

    for key in robot_link_names: # find the key #
        link = chain.find_link(key)
        
        m = tg_batch[key].get_matrix()
        pos = m[:, :3, 3].float() # 
        rot = m[:, :3, :3].float()
        
        rot_quat = []
        for i_fr in range(rot.size(0)):
            cur_rot_mtx = rot[i_fr]
            cur_rot_quat = R.from_matrix(cur_rot_mtx.detach().cpu().numpy()).as_quat()
            cur_rot_quat = torch.from_numpy(cur_rot_quat).float().to(d)
            rot_quat.append(cur_rot_quat)
        rot_quat = torch.stack(rot_quat, dim=0)
        # rot_quat = R.from_matrix(rot.detach().cpu().numpy()).as_quat()
        # rot_quat = torch.from_numpy(rot_quat).float().to(d)
        # #
        m = torch.cat(
            [pos, rot_quat], dim=-1
        )
        
        link_name_to_poses[key] = m.detach().cpu().numpy() # get the link name to poses #
        
    retar_info['link_name_to_poses'] = link_name_to_poses # ge tthe link name to poses 
    np.save(retar_info_fn, retar_info)
        
    print(f"Expanded retargeted info with keys {retar_info.keys()} saved to {retar_info_fn}")


def interp_two_trajs(retar_info_fn_1, retar_info_fn_2, interp_fr=60, nn_transition=100, taco_inst_tag=''):
    retar_info_1 = np.load(retar_info_fn_1, allow_pickle=True).item()
    retar_info_2 = np.load(retar_info_fn_2,allow_pickle=True).item()
    
    nn_hand_dofs = 22
    hand_qs_1 = retar_info_1['robot_delta_states_weights_np'][:, :nn_hand_dofs]
    hand_qs_2 = retar_info_2['robot_delta_states_weights_np'][:, :nn_hand_dofs]
    obj_pos_1 = retar_info_1['object_transl']
    obj_pos_2 = retar_info_2['object_transl']
    obj_rot_1 = retar_info_1['object_rot_quat']
    obj_rot_2 = retar_info_2['object_rot_quat']
    
    
    
    tot_obj_pos = []
    tot_obj_ornt = []
    tot_hand_qs = []
    tot_obj_pos.append(obj_pos_1[: interp_fr])
    tot_obj_ornt.append(obj_rot_2[: interp_fr])
    tot_hand_qs.append(hand_qs_1[: interp_fr]) # the first hand qs frames #
    
    keyfr_obj_pos_1 = obj_pos_1[interp_fr]
    keyfr_obj_pos_2 = obj_pos_2[interp_fr]
    transit_obj_ornt = obj_rot_2[interp_fr][None]
    for i_transit in range(nn_transition):
        cur_obj_pos = keyfr_obj_pos_1 + (keyfr_obj_pos_2 - keyfr_obj_pos_1) * float(i_transit) / float(nn_transition)
        tot_obj_pos.append(cur_obj_pos[None])
        tot_obj_ornt.append(transit_obj_ornt)
    
    keyfr_hand_qs_1 = hand_qs_1[interp_fr]
    keyfr_hand_qs_2 = hand_qs_2[interp_fr]
    for i_transit in range(nn_transition):
        cur_hand_qs = keyfr_hand_qs_1 + (keyfr_hand_qs_2 - keyfr_hand_qs_1) * float(i_transit) / float(nn_transition)
        tot_hand_qs.append(cur_hand_qs[None])
    
    tot_obj_pos.append(obj_pos_2[interp_fr:])
    tot_hand_qs.append(hand_qs_2[interp_fr:])
    tot_obj_ornt.append(obj_rot_2[interp_fr:])
    
    tot_obj_pos = np.concatenate(tot_obj_pos, axis=0)
    tot_hand_qs = np.concatenate(tot_hand_qs, axis=0)
    tot_obj_ornt = np.concatenate(tot_obj_ornt, axis=0)
    retar_info_dict = {
        'robot_delta_states_weights_np': tot_hand_qs,
        'object_transl': tot_obj_pos,
        'object_rot_quat': tot_obj_ornt
    }
    sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_{taco_inst_tag}.npy'
    np.save(sv_retar_info_fn, retar_info_dict)
    print(f"Saved the interpolated retargeted info to {sv_retar_info_fn}")


    
def linear_interp_data(ori_data, nn_interp_freq):
    interped_data = []
    for i_fr in range(ori_data.shape[0] - 1):
        nex_fr = i_fr + 1
        cur_hand_q = ori_data[i_fr]
        nex_hand_q = ori_data[nex_fr] 
        for i_interp in range(nn_interp_freq):
            cur_subfr_hand_q = (nex_hand_q - cur_hand_q) * float(i_interp) / float(nn_interp_freq) + cur_hand_q
            interped_data.append(cur_subfr_hand_q)
    interped_data.append(ori_data[-1])
    interped_data = np.stack(interped_data, axis=0)
    return interped_data

def interp_slerp(ori_data, nn_interp_freq):
    ori_ts = range(0, ori_data.shape[0])
    ori_ts = list(ori_ts) 
    ## origian ts ## 
    interped_ts = []
    for i_ts in range(len(ori_ts) - 1):
        nex_i_ts = i_ts + 1
        for i_interp in range(nn_interp_freq):
            cur_substep_ts = float(i_ts) + (float(nex_i_ts) - float(i_ts)) * float(i_interp) / float(nn_interp_freq)
            interped_ts.append(cur_substep_ts)
    interped_ts.append(ori_ts[-1])
    ori_data_struct = R.from_quat(ori_data)
    slerp = Slerp(ori_ts, ori_data_struct)
    interp_rots = slerp(interped_ts)
    interp_rots = interp_rots.as_quat()
    
    return interp_rots



def find_grasp_frame_from_mocap_data(mocap_data_fn):
    data_dict = np.load(mocap_data_fn, allow_pickle=True).item()
    hand_qs = data_dict['robot_delta_states_weights_np']
    obj_transl = data_dict['object_transl'][:]
    obj_ornt = data_dict['object_rot_quat'][: ]
    # 
    # nn_frames x 3 
    # nn_frames x 4
    
    eps = 1e-2
    # 
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



def interp_two_trajs_2(retar_info_fn_1, retar_info_fn_2, grasping_fr=27, grasping_fr_grab=37, interp_fr=60, interp_fr_2=60, nn_transition=100, taco_inst_tag='', grab_inst_tag='', additional_tag='v2', hand_type='allegro'):
    
    if hand_type == 'allegro':
        sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{grab_inst_tag}_interped_taco_{taco_inst_tag}_{additional_tag}.npy'
    else:
        sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/passive_active_info_{grab_inst_tag}_interped_taco_{taco_inst_tag}_{additional_tag}.npy'
        # f'/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/passive_active_info_{grab_inst_tag}_interped_taco_{taco_inst_tag}_{additional_tag}.npy'
    if os.path.exists(sv_retar_info_fn):
        return
    
    retar_info_1 = np.load(retar_info_fn_1, allow_pickle=True).item()
    retar_info_2 = np.load(retar_info_fn_2,allow_pickle=True).item()
    
    nn_hand_dofs = 22
    hand_qs_1 = retar_info_1['robot_delta_states_weights_np'][:, :nn_hand_dofs]
    hand_qs_2 = retar_info_2['robot_delta_states_weights_np'][:, :nn_hand_dofs]
    obj_pos_1 = retar_info_1['object_transl']
    obj_pos_2 = retar_info_2['object_transl']
    obj_rot_1 = retar_info_1['object_rot_quat']
    obj_rot_2 = retar_info_2['object_rot_quat']
    
    # obj_rot_1 = obj_rot_1[:, [2, 1, 0, 3]]
    
    # first_fr_rot_1_mtx = R.from_quat(obj_rot_1[0]).as_matrix()
    # first_fr_rot_2_mtx  = R.from_quat(obj_rot_2[0]).as_matrix()
    # delta_rot_mtx = np.matmul(first_fr_rot_2_mtx, first_fr_rot_1_mtx.T)
    # modified_obj_rot_1 = []
    # for i_fr in range(obj_rot_1.shape[0]):
    #     cur_fr_rot_mtx = R.from_quat(obj_rot_1[i_fr]).as_matrix()
    #     modified_mtx = np.matmul(delta_rot_mtx, cur_fr_rot_mtx)
    #     modified_quat = R.from_matrix(modified_mtx).as_quat()
    #     modified_obj_rot_1.append(modified_quat)
    # modified_obj_rot_1 = np.stack(modified_obj_rot_1, axis=0)
    # obj_rot_1 = modified_obj_rot_1
    
    
    
    tot_obj_pos = []
    tot_obj_ornt = []
    tot_hand_qs = []
    
    # tot_obj_pos.append(obj_pos_1[: interp_fr])
    # tot_obj_ornt.append(obj_rot_2[: interp_fr])
    
    # tot_obj_ornt.append(obj_rot_1[: interp_fr])
    
    tot_obj_pos.append(obj_pos_2[:grasping_fr])
    tot_obj_ornt.append(obj_rot_2[:grasping_fr])
    
    st_obj_ornt = obj_rot_2[grasping_fr]
    ed_obj_ornt = obj_rot_1[interp_fr]
    stacked_obj_rot = np.stack(
        [st_obj_ornt, ed_obj_ornt], axis=0
    )
    interped_obj_rots = interp_slerp(stacked_obj_rot, interp_fr - grasping_fr)
    interped_obj_rots = interped_obj_rots[:-1]
    tot_obj_ornt.append(interped_obj_rots)
    
    st_obj_pos = obj_pos_2[grasping_fr]
    ed_obj_pos = obj_pos_1[grasping_fr_grab] # 
    stacked_obj_pos = np.stack(
        [st_obj_pos, ed_obj_pos], axis=0
    )
    interped_obj_pos = linear_interp_data(stacked_obj_pos, grasping_fr_grab - grasping_fr)
    tot_obj_pos.append(interped_obj_pos[:-1])
    tot_obj_pos.append(obj_pos_1[grasping_fr_grab: interp_fr])
    
    
    tot_hand_qs.append(hand_qs_1[: interp_fr]) # the first hand qs frames # # grasping fr grab -- 
    
    keyfr_obj_pos_1 = obj_pos_1[interp_fr]
    keyfr_obj_pos_2 = obj_pos_2[interp_fr_2]
    # transit_obj_ornt = obj_rot_2[interp_fr_2][None]
    for i_transit in range(nn_transition):
        cur_obj_pos = keyfr_obj_pos_1 + (keyfr_obj_pos_2 - keyfr_obj_pos_1) * float(i_transit) / float(nn_transition)
        tot_obj_pos.append(cur_obj_pos[None])
        # tot_obj_ornt.append(transit_obj_ornt)
    
    keyfr_obj_ornt_1 = obj_rot_1[interp_fr]
    keyfr_obj_ornt_2 = obj_rot_2[interp_fr_2]
    stacked_obj_rot = np.stack(
        [keyfr_obj_ornt_1, keyfr_obj_ornt_2], axis=0
    )
    interped_obj_rots = interp_slerp(stacked_obj_rot, nn_transition)
    interped_obj_rots = interped_obj_rots[:-1]
    tot_obj_ornt.append(interped_obj_rots)
    
    keyfr_hand_qs_1 = hand_qs_1[interp_fr]
    keyfr_hand_qs_2 = hand_qs_2[interp_fr_2]
    for i_transit in range(nn_transition):
        cur_hand_qs = keyfr_hand_qs_1 + (keyfr_hand_qs_2 - keyfr_hand_qs_1) * float(i_transit) / float(nn_transition)
        tot_hand_qs.append(cur_hand_qs[None])
    
    tot_obj_pos.append(obj_pos_2[interp_fr_2:])
    tot_hand_qs.append(hand_qs_2[interp_fr_2:])
    tot_obj_ornt.append(obj_rot_2[interp_fr_2:])
    
    tot_obj_pos = np.concatenate(tot_obj_pos, axis=0)
    tot_hand_qs = np.concatenate(tot_hand_qs, axis=0)
    tot_obj_ornt = np.concatenate(tot_obj_ornt, axis=0)
    retar_info_dict = {
        'robot_delta_states_weights_np': tot_hand_qs,
        'object_transl': tot_obj_pos,
        'object_rot_quat': tot_obj_ornt
    }
    # sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_{taco_inst_tag}_v2.npy'
    # save the retar info with the grab_inst_tag and taco_inst_tag #
    # sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{grab_inst_tag}_interped_taco_{taco_inst_tag}_{additional_tag}.npy'
    np.save(sv_retar_info_fn, retar_info_dict)
    print(f"Saved the interpolated retargeted info to {sv_retar_info_fn}")


def get_all_taco_instancees(taco_data_root, all_taco_trajs=False):
    
    if all_taco_trajs:
        tot_taco_subfolders = os.listdir(taco_data_root)
        tot_taco_subfolders = [ fn for fn in tot_taco_subfolders if os.path.isdir(os.path.join(taco_data_root, fn)) ]
        tot_inst_idxes = []
        for cur_subfolder in tot_taco_subfolders:
            cur_taco_data_root = f"{taco_data_root}/{cur_subfolder}"
            cur_taco_fns = os.listdir(cur_taco_data_root)
            cur_taco_fns = [fn for fn in cur_taco_fns if fn.endswith(".pkl")]
            cur_taco_inst_idxes = [fn.split(".")[0][6:]  for fn in cur_taco_fns]
            # right #
            cur_taco_inst_idxes = [ fn for fn in cur_taco_inst_idxes ] # fn not in cur_taco_inst_idxes #
            tot_inst_idxes = tot_inst_idxes + cur_taco_inst_idxes # ij
    else:
        taco_data_root = f"{taco_data_root}/20231104"
        tot_data_fn = os.listdir(taco_data_root)
        tot_data_fn = [fn for fn in tot_data_fn if fn.endswith(".pkl")]
        new_tot_inst_idxes = [fn.split(".")[0][6:] for fn in tot_data_fn]
        tot_inst_idxes = [fn for fn in new_tot_inst_idxes]
    return tot_inst_idxes
    pass



def interp_two_trajs_v1(retar_info_fn_1, retar_info_fn_2, interp_fr=60, nn_transition=100, taco_inst_tag='', hand_type='allegro'):
    retar_info_1 = np.load(retar_info_fn_1, allow_pickle=True).item()
    retar_info_2 = np.load(retar_info_fn_2,allow_pickle=True).item()
    
    nn_hand_dofs = 22
    hand_qs_1 = retar_info_1['robot_delta_states_weights_np'][:, :nn_hand_dofs]
    hand_qs_2 = retar_info_2['robot_delta_states_weights_np'][:, :nn_hand_dofs]
    obj_pos_1 = retar_info_1['object_transl']
    obj_pos_2 = retar_info_2['object_transl']
    obj_rot_1 = retar_info_1['object_rot_quat']
    obj_rot_2 = retar_info_2['object_rot_quat']
    
    
    
    tot_obj_pos = []
    tot_obj_ornt = []
    tot_hand_qs = []
    tot_obj_pos.append(obj_pos_1[: interp_fr])
    tot_obj_ornt.append(obj_rot_1[: interp_fr])
    tot_hand_qs.append(hand_qs_1[: interp_fr]) # the first hand qs frames #
    
    keyfr_obj_pos_1 = obj_pos_1[interp_fr]
    keyfr_obj_pos_2 = obj_pos_2[interp_fr]
    # transit_obj_ornt = obj_rot_2[interp_fr][None]
    for i_transit in range(nn_transition):
        cur_obj_pos = keyfr_obj_pos_1 + (keyfr_obj_pos_2 - keyfr_obj_pos_1) * float(i_transit) / float(nn_transition)
        tot_obj_pos.append(cur_obj_pos[None])
        # tot_obj_ornt.append(transit_obj_ornt)
        
    keyfr_obj_ornt_1 = obj_rot_1[interp_fr]
    keyfr_obj_ornt_2 = obj_rot_2[interp_fr]
    cat_keyfr_obj_ornt = np.stack(
        [keyfr_obj_ornt_1, keyfr_obj_ornt_2], axis=0
    )
    interped_obj_ornt = interp_slerp(cat_keyfr_obj_ornt, nn_transition)
    interped_obj_ornt = interped_obj_ornt[:-1]
    tot_obj_ornt.append(interped_obj_ornt)
    
    keyfr_hand_qs_1 = hand_qs_1[interp_fr]
    keyfr_hand_qs_2 = hand_qs_2[interp_fr]
    for i_transit in range(nn_transition):
        cur_hand_qs = keyfr_hand_qs_1 + (keyfr_hand_qs_2 - keyfr_hand_qs_1) * float(i_transit) / float(nn_transition)
        tot_hand_qs.append(cur_hand_qs[None])
    
    tot_obj_pos.append(obj_pos_2[interp_fr:])
    tot_hand_qs.append(hand_qs_2[interp_fr:])
    tot_obj_ornt.append(obj_rot_2[interp_fr:])
    
    tot_obj_pos = np.concatenate(tot_obj_pos, axis=0)
    tot_hand_qs = np.concatenate(tot_hand_qs, axis=0)
    tot_obj_ornt = np.concatenate(tot_obj_ornt, axis=0)
    retar_info_dict = {
        'robot_delta_states_weights_np': tot_hand_qs,
        'object_transl': tot_obj_pos,
        'object_rot_quat': tot_obj_ornt
    }
    if hand_type == 'allegro':
        sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_{taco_inst_tag}_v1.npy'
    elif hand_type == 'leap':
        # /cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/leap_passive_active_info_taco_20230930_039_zrot_3.141592653589793_modifed_interped.npy
        sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/leap_passive_active_info_ori_grab_s2_phone_call_1_interped_taco_{taco_inst_tag}_v1.npy'
    else:
        raise ValueError(f"Unknown hand type: {hand_type}")
    
    np.save(sv_retar_info_fn, retar_info_dict)
    print(f"Saved the interpolated retargeted info to {sv_retar_info_fn}")



########## Main functions ##########

def interpolate_all_taco_ref_files():
    taco_data_root = "/cephfs/xueyi/data/TACO_Tracking_PK/data"
    tot_totest_taco_files = get_totest_taco_files(taco_data_root=taco_data_root)
    # tot_totest_taco_files = ["passive_active_info_taco_20231104_203_zrot_3.141592653589793.npy"]
    # tot_totest_taco_files = ["passive_active_info_taco_20231104_016_zrot_3.141592653589793.npy", "passive_active_info_taco_20231104_151_zrot_3.141592653589793.npy"]
    existing_taco_file_tags = ['taco_20231104_016', 'taco_20231104_203', 'taco_20231104_151']
    z_rot_tag = '_zrot_'
    tot_totest_taco_files =  [fn for fn in tot_totest_taco_files if z_rot_tag in fn]
    tot_filtered_totest_taco_files = []
    for fn in tot_totest_taco_files:
        already_exist = False
        for existing_taco_tag in existing_taco_file_tags:
            if existing_taco_tag in fn:
                already_exist = True
                break
        if already_exist:
            continue
        tot_filtered_totest_taco_files.append(fn)
    tot_totest_taco_files = tot_filtered_totest_taco_files
                
    
    reverse_data = True
    # reverse_data = False
    
    nn_interp_freq = 2 
    
    for cur_taco_ref_fn in tot_totest_taco_files:
        full_taco_ref_fn = os.path.join(taco_data_root, cur_taco_ref_fn)
        print(f"processing {full_taco_ref_fn}")
        modified_taco_fn = full_taco_ref_fn.replace(".npy", "_modifed.npy")
        if os.path.exists(modified_taco_fn):
            continue
        #  ref_fn = ref_fn.replace(".npy", "_interped.npy") 
        interped_taco_fn = modified_taco_fn.replace(".npy", "_interped.npy") 
        if os.path.exists(interped_taco_fn):
            continue # 
        try:
            test_ref_fn(full_taco_ref_fn, reversed=reverse_data)
        except:
            continue
        modified_ref_fn  = full_taco_ref_fn.replace(".npy", "_modifed.npy") 
        try:
            interpolate_hand_poses(modified_ref_fn, nn_interp_freq=nn_interp_freq)
        except:
            continue
        

def test_find_grasping_frame():
    
    taco_inst_tag = '20231104_151'
    retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
    taco_grasping_fr = find_grasp_frame_from_mocap_data(retar_info_fn_2)
    print(taco_grasping_fr)


def interp_grab_traj_w_taco_traj(grab_inst_tag='ori_grab_s2_phone_call_1', additional_tag='v2', all_taco_trajs=False, interp_fr=120, interp_fr_2=60, nn_transition=40, hand_type='allegro'):
    grab_inst_grasping_frame = 37
    
    taco_data_root = "/cephfs/xueyi/data/taco/processed_data"
    
    existing_taco_file_tags = ['20231104_016', '20231104_203', '20231104_151']
    
    tot_inst_idxes = get_all_taco_instancees(taco_data_root, all_taco_trajs=all_taco_trajs)
    
    tot_inst_idxes = [cur_inst_tag for cur_inst_tag in tot_inst_idxes if cur_inst_tag not in existing_taco_file_tags]
    
    
    print(f"tot_inst_idxes: {tot_inst_idxes}")
    
    for i_inst, cur_inst_idx in enumerate(tot_inst_idxes):
        
        taco_inst_tag = cur_inst_idx
        
        if hand_type == 'allegro':
            retar_info_fn_1 = f'/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_{grab_inst_tag}_nf_300.npy'
            retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
            
        else:
            retar_info_fn_1 = f'/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data/leap_passive_active_info_{grab_inst_tag}.npy'
            retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/leap_passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
        # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_20231013_084_zrot_3.141592653589793_modifed_interped.npy
        
        
        if not os.path.exists(retar_info_fn_2):
            continue
            
        if hand_type == 'allegro':
            sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_{grab_inst_tag}_interped_taco_{taco_inst_tag}_{additional_tag}.npy'
        else:
            sv_retar_info_fn = f'/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/passive_active_info_{grab_inst_tag}_interped_taco_{taco_inst_tag}_{additional_tag}.npy'
        
        if os.path.exists(sv_retar_info_fn):
            print(f"[{taco_inst_tag}] Retar info {sv_retar_info_fn} already exists")
            continue
        
        taso_grasping_fr = find_grasp_frame_from_mocap_data(retar_info_fn_2)
        
        print(f"[{i_inst}/{len(tot_inst_idxes)}] processing {taco_inst_tag} with grasp frame {taso_grasping_fr}")
        
        ##### 27 #####
        taso_grasping_fr = 27
        
        # interp_fr = 120
        # interp_fr_2 = 60
        # nn_transition = 40
        
        # interp two trajs #
        # interp two trajs #
        # additional_tag='v2' #
        interp_two_trajs_2(retar_info_fn_1, retar_info_fn_2, grasping_fr=taso_grasping_fr, grasping_fr_grab=grab_inst_grasping_frame, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition, taco_inst_tag=taco_inst_tag, grab_inst_tag=grab_inst_tag, additional_tag=additional_tag, hand_type=hand_type)
        # interp_two_trajs(retar_info_fn_1, retar_info_fn_2, interp_fr, nn_transition, taco_inst_tag)


# interp #
def interp_grab_traj_w_taco_traj_v1(hand_type='allegro'):
    # grab_inst_grasping_frame = 37
    
    taco_data_root = "/cephfs/xueyi/data/taco/processed_data"
    
    existing_taco_file_tags = []
    
    tot_inst_idxes = get_all_taco_instancees(taco_data_root, all_taco_trajs=True)
    
    tot_inst_idxes = [cur_inst_tag for cur_inst_tag in tot_inst_idxes if cur_inst_tag not in existing_taco_file_tags]
    
    for i_inst, cur_inst_idx in enumerate(tot_inst_idxes):
        
        taco_inst_tag = cur_inst_idx
        
        if hand_type == 'allegro':
            retar_info_fn_1 = f'/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_nf_300.npy'
            retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
        elif hand_type == 'leap':
            retar_info_fn_1 = f'/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data/leap_passive_active_info_ori_grab_s2_phone_call_1_nf_300.npy'
            retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK_LEAP/data/leap_passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
        else:
            raise ValueError(f"Unrecognized hand type {hand_type}")
        
        
        if not os.path.exists(retar_info_fn_2):
            continue
        
        
        # taco_grasping_fr = find_grasp_frame_from_mocap_data(retar_info_fn_2)
        
        # taco_grasping_fr = find_grasp_frame_from_mocap_data(retar_info_fn_2)
        
        
        print(f"[{i_inst}/{len(tot_inst_idxes)}] processing {taco_inst_tag}")
        
        
        taso_grasping_fr = 27
        
        interp_fr = 120
        interp_fr_2 = 60
        nn_transition = 40
        
        interp_two_trajs_v1(retar_info_fn_1, retar_info_fn_2, interp_fr=60, nn_transition=nn_transition, taco_inst_tag=taco_inst_tag, hand_type=hand_type)
        
        # interp_two_trajs_2(retar_info_fn_1, retar_info_fn_2, grasping_fr=taso_grasping_fr, grasping_fr_grab=grab_inst_grasping_frame, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition, taco_inst_tag=taco_inst_tag)
        # interp_two_trajs(retar_info_fn_1, retar_info_fn_2, interp_fr, nn_transition, taco_inst_tag)


def mv_taco_grab_interp_folders(local_folder, target_folder, exp_tag):
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_TACO_taco_20231104_001_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-23-57-47
    tot_eval_fns = os.listdir(local_folder)
    tracking_data_sv_st_tag = "tracking_"
    tot_eval_fns = [
        fn for fn in tot_eval_fns if fn[: len(tracking_data_sv_st_tag)] == tracking_data_sv_st_tag
    ]
    tot_eval_fns = [
        fn for fn in tot_eval_fns if exp_tag in fn
    ]
    for i_fn, cur_fn in enumerate(tot_eval_fns):
        cur_full_fn = os.path.join(local_folder, cur_fn)
        print(f"[{i_fn}/{len(tot_eval_fns)}] mv {cur_full_fn} {target_folder}/")
        os.system(f"mv {cur_full_fn} {target_folder}/")
        

def parse_obj_type_from_taco_grab_folder_fn(exp_folder):
    exp_folder_segs = exp_folder.split("_")
    obj_type_segs = exp_folder_segs[2:5]
    obj_type = "_".join(obj_type_segs)
    return obj_type

def get_best_sv_pth(exp_folder):
    exp_nn_folder = os.path.join(exp_folder, "nn")
    if not os.path.exists(exp_nn_folder):
        return None, None
    tot_ckpt_fns = os.listdir(exp_nn_folder)
    tot_ckpt_fns = [
        fn for fn in tot_ckpt_fns if fn.endswith(".pth")
    ]
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_TACO_taco_20230928_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-02-00-42/nn/last_tracking_TACO_taco_20230928_037_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_750_rew_-69.42215.pth
    
    # get best rew #
    rew_ckpt_fns = [
        fn for fn in tot_ckpt_fns if '_rew_' in fn
    ]
    # print(f"{rew_ckpt_fns}")
    rews = []
    for cur_rew_ckpt_fn in rew_ckpt_fns:
        try:
            cur_rew = float(cur_rew_ckpt_fn.split(".pth")[0].split("_rew_")[-1])
            rews.append(cur_rew)
        except:
            pass
    # rews = [ # get the fn split #
    #     float(fn.split(".pth")[0].split("_rew_")[-1]) for fn in rew_ckpt_fns
    # ]
    if len(rews) == 0:
        return None, None
    best_rew = max(rews)
    
    best_tracking_st_tag = "tracking_"
    tot_ckpt_fns = [
        fn for fn in tot_ckpt_fns if fn[: len(best_tracking_st_tag) ] == best_tracking_st_tag
    ]
    assert len(tot_ckpt_fns) == 1
    best_ckpt_fn = tot_ckpt_fns[0]
    best_ckpt_fn = os.path.join(exp_nn_folder, best_ckpt_fn)
    return best_ckpt_fn, best_rew



# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq

def calculate_taco_obj_type_to_optimized_res(taco_sv_folder, exclude_last_opts=False):
    tot_sv_fns = os.listdir(taco_sv_folder)
    tracking_data_sv_st_tag = "tracking_"
    tot_sv_fns = [
        fn for fn in tot_sv_fns if fn[: len(tracking_data_sv_st_tag)] == tracking_data_sv_st_tag
    ]
    
    tot_sv_fns_w_time = []
    for cur_sv_fn in tot_sv_fns:
        # parse_time_from_tracking_folder_nm 
        cur_folder_time = parse_time_from_tracking_folder_nm(cur_sv_fn)
        tot_sv_fns_w_time.append( (cur_sv_fn, cur_folder_time) )
    tot_sv_fns_w_time = sorted(tot_sv_fns_w_time, key=lambda x: x[1])
    
    if exclude_last_opts:
        tot_sv_fns_w_time = tot_sv_fns_w_time[:-8]
    
    obj_type_to_optimized_res = {}
    
    # for cur_fn in tot_sv_fns: # cur fn # taco data utils #
    for cur_fn, _ in tot_sv_fns_w_time:
        cur_obj_type, _ = parse_taco_grab_obj_type_from_folder_name(cur_fn)
        cur_full_fn = os.path.join(taco_sv_folder, cur_fn)
        print(f"processing {cur_full_fn}")
        cur_best_ckpt_fn, best_rew = get_best_sv_pth(cur_full_fn)
        if cur_best_ckpt_fn is None:
            continue
        if cur_obj_type in obj_type_to_optimized_res:
            prev_best_rew = obj_type_to_optimized_res[cur_obj_type][0]
            if best_rew > prev_best_rew:
                obj_type_to_optimized_res[cur_obj_type] = ( best_rew, cur_best_ckpt_fn )
        else:
            
            obj_type_to_optimized_res[cur_obj_type] = ( best_rew,  cur_best_ckpt_fn)
    return obj_type_to_optimized_res
        

def inspect_taco_obj_type_to_optimized_res(obj_type_to_optimized_res_sv_fn):
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_sv_fn, allow_pickle=True).item()
    cur_nn = 0
    for obj_type in obj_type_to_optimized_res:
        cur_obj_res = obj_type_to_optimized_res[obj_type]
        print(f"[{cur_nn}/{len(obj_type_to_optimized_res)}] {obj_type}: {cur_obj_res}")
        cur_nn += 1



def inspect_total_taco_data():
    taso_inst_st_flag = 'taco_'
    mesh_sv_root = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    if not os.path.exists(mesh_sv_root):
        mesh_sv_root = "/home/xueyi/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    tot_mesh_folders = os.listdir(mesh_sv_root)
    # find meshes directly #
    tot_mesh_folders = [fn for fn in tot_mesh_folders if fn[: len(taso_inst_st_flag)] == taso_inst_st_flag and 'modifed_interped' not in fn and 'modifed' not in fn]
    tot_tracking_data = tot_mesh_folders
    # tot_tracking_data = 
    print(len(tot_tracking_data))

def inspect_taco_optimized_data_folder(taco_sv_folder):
    tot_fns = os.listdir(taco_sv_folder)
    fn_tracking_st_tag = "tracking_"
    tot_fns = [
        fn for fn in tot_fns if fn[: len(fn_tracking_st_tag)] == fn_tracking_st_tag
    ]
    print(len(tot_fns))


def mv_taco_interp_optimized_res(local_folder, target_folder):
    target_statistics_folder = os.path.join(target_folder, "statistics")
    obj_type_to_optimized_res_fn = "obj_type_to_optimized_res.npy"
    obj_type_to_optimized_res_fn = os.path.join(target_statistics_folder, obj_type_to_optimized_res_fn)
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item() ### instance tag to the optimized res ## 
    
    target_maxx_ep = 1000
    
    tracking_sv_tag = 'tracking_'
    
    local_opt_fns = os.listdir(local_folder)
    local_opt_fns = [ fn for fn in local_opt_fns if fn[: len(tracking_sv_tag)] == tracking_sv_tag ]
    
    for cur_local_opt_fn in local_opt_fns:
        cur_obj_type = parse_obj_type_from_taco_grab_folder_fn(cur_local_opt_fn)
        cur_full_opt_fn = os.path.join(local_folder, cur_local_opt_fn)
        
        if cur_obj_type in obj_type_to_optimized_res:
            print(f"{cur_obj_type} already exists")
            continue
        
        print(f"processing {cur_full_opt_fn}")
        cur_full_opt_nn_folder = os.path.join(cur_full_opt_fn, "nn")
        if not os.path.exists(cur_full_opt_nn_folder):
            continue
        
        tot_tracking_ckpts = os.listdir(cur_full_opt_nn_folder)
        tot_tracking_ckpts = [fn for fn in tot_tracking_ckpts if fn.endswith(".pth")]
        rew_tag, ep_tag = "_rew_", "_ep_"
        tot_tracking_ckpts = [ fn for fn in tot_tracking_ckpts if rew_tag in fn and ep_tag in fn ]
        tot_ep_numbers = []
        for cur_ckpt_fn in tot_tracking_ckpts:
            cur_ckpt_fn_segs = cur_ckpt_fn.split(rew_tag)[0].split(ep_tag)[-1]
            cur_ckpt_fn_segs = int(cur_ckpt_fn_segs)
            tot_ep_numbers.append(cur_ckpt_fn_segs)
        maxx_ep = max(tot_ep_numbers)
        if maxx_ep < target_maxx_ep:
            print(f"maxx_ep: {maxx_ep} < {target_maxx_ep}")
            continue
        
        # cur full opt fn # 
        if cur_obj_type not in obj_type_to_optimized_res:
            command_ = f"mv {cur_full_opt_fn} {target_folder}/"
            print(command_)
            os.system(command_)
    
    # cur_obj_type = parse_obj_type_from_taco_grab_folder_fn()


def parse_taco_grab_obj_type_from_folder_name(exp_folder):
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2/tracking_taco_20230927_013_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-12-30-57
    if '_TACO_' in exp_folder:
        
        exp_folder_segs = exp_folder.split("_")
        exp_taco_obj_type_segs = exp_folder_segs[2:5]
        exp_grab_obj_type_segs = exp_folder_segs[6:]
        grab_obj_type_ed_idx = 6
        for grab_obj_type_ed_idx in range(grab_obj_type_ed_idx, len(exp_folder_segs)):
            if exp_folder_segs[grab_obj_type_ed_idx] == 'obs':
                break
        grab_obj_type_segs = exp_folder_segs[6: grab_obj_type_ed_idx]
        grab_obj_type = "_".join(grab_obj_type_segs)
        
        taco_obj_type = "_".join(exp_taco_obj_type_segs)
    else:
        exp_folder_segs = exp_folder.split("_")
        exp_taco_obj_type_segs = exp_folder_segs[1:4]
        # exp_grab_obj_type_segs = exp_folder_segs[6:]
        # grab_obj_type_ed_idx = 6
        # for grab_obj_type_ed_idx in range(grab_obj_type_ed_idx, len(exp_folder_segs)):
        #     if exp_folder_segs[grab_obj_type_ed_idx] == 'obs':
        #         break
        # grab_obj_type_segs = exp_folder_segs[6: grab_obj_type_ed_idx]
        # grab_obj_type = "_".join(grab_obj_type_segs)
        
        taco_obj_type = "_".join(exp_taco_obj_type_segs)
        grab_obj_type = "ori_grab_s2_phone_call_1"
        
    return taco_obj_type, grab_obj_type


# tracking_TACO_taco_20231104_114_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-22-09-22
def get_data_inst_tag_to_optimized_res(eval_data_folder, inst_tag_to_opt_res_fn='data_inst_tag_to_optimized_res.npy', exclude_last_saved=False):
    tot_runs_subfolder = os.listdir(eval_data_folder)
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-23-05-11
    tracking_folder_st_tag = 'tracking_'
    obs_tag = 'obs_pure_state_wref_wdelta'
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if fn[: len(tracking_folder_st_tag)] == tracking_folder_st_tag
    ]
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if obs_tag in fn
    ]
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if (not os.path.exists(os.path.join(eval_data_folder, fn, "nn")) )
    ]
    
    if exclude_last_saved:
        tot_runs_subfolders_wtime = []
        for cur_fn in tot_runs_subfolder:
            cur_fn_time = parse_time_from_tracking_folder_nm(cur_fn)
            tot_runs_subfolders_wtime.append((cur_fn, cur_fn_time))
        tot_runs_subfolders_wtime = sorted(tot_runs_subfolders_wtime, key=lambda x: x[1])
        tot_runs_subfolders_wtime = tot_runs_subfolders_wtime[:-10]
        tot_runs_subfolder = [
            iitem[0] for iitem in tot_runs_subfolders_wtime
        ] # runs subfolder # # runs subfolder #
    
    
    traj_sv_fn = "ts_to_hand_obj_obs_reset_1.npy"
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if os.path.exists(os.path.join(eval_data_folder, fn, traj_sv_fn))
    ]
    
    # exclude_last_saved # 
    
    # exlcue the last saved res # 
    # save 
    
    
    data_inst_tag_to_optimized_res = {}
    for i_test, cur_run_subfolder in enumerate(tot_runs_subfolder):
        cur_full_run_subfolder = os.path.join(eval_data_folder, cur_run_subfolder)
        # cur_obj_type = parse_obj_type_fr_folder_name(cur_full_run_subfolder)
        
        taco_obj_type, grab_obj_type = parse_taco_grab_obj_type_from_folder_name(cur_run_subfolder)
        
        # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-23-05-11/ts_to_hand_obj_obs_reset_1.npy
        cur_run_traj_sv_fn = os.path.join(cur_full_run_subfolder, traj_sv_fn)
        
        print(f"[{i_test}/{len(tot_runs_subfolder)}] cur_run_subfolder: {cur_run_subfolder}")
        
        # try:
        #     cur_cur_traj_sv_info = np.load(cur_run_traj_sv_fn, allow_pickle=True).item()
        # except:
        #     continue
        
        # if 0 not in cur_cur_traj_sv_info:
        #     continue
        # first_fr_sv_res = cur_cur_traj_sv_info[0]
        # if 'shadow_hand_dof_tars' not in first_fr_sv_res:
        #     continue
        
        # if 'optimized_hand_qtars' not in cur_cur_traj_sv_info:
        #     continue
        
        cur_grab_exp_tag = (taco_obj_type, grab_obj_type)
        if cur_grab_exp_tag not in data_inst_tag_to_optimized_res:
            data_inst_tag_to_optimized_res[cur_grab_exp_tag] = [cur_run_traj_sv_fn]
        else:
            data_inst_tag_to_optimized_res[cur_grab_exp_tag].append(cur_run_traj_sv_fn) # get the run traj sv fn #
        print(f"len(data_inst_tag_to_optimized_res): {len(data_inst_tag_to_optimized_res)}")
    
    data_inst_tag_to_optimized_res_sv_statistics_folder = "statistics"
    data_inst_tag_to_optimized_res_sv_statistics_folder = os.path.join(eval_data_folder, data_inst_tag_to_optimized_res_sv_statistics_folder)
    os.makedirs(data_inst_tag_to_optimized_res_sv_statistics_folder, exist_ok=True)
    # data_inst_tag_to_optimized_res_sv_statistics_fn = "data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_sv_statistics_fn = inst_tag_to_opt_res_fn
    data_inst_tag_to_optimized_res_sv_statistics_fn = os.path.join(data_inst_tag_to_optimized_res_sv_statistics_folder, data_inst_tag_to_optimized_res_sv_statistics_fn)
    np.save(data_inst_tag_to_optimized_res_sv_statistics_fn, data_inst_tag_to_optimized_res)
    print(f"data_inst_tag_to_optimized_res saved to {data_inst_tag_to_optimized_res_sv_statistics_fn}")
    return data_inst_tag_to_optimized_res
        
    #     cur_best_rew, best_ckpt_fn = find_best_rew(cur_full_run_subfolder)
        
    #     if best_ckpt_fn is None:
    #         continue
    #     data_inst_tag_to_optimized_res[cur_obj_type] = (cur_best_rew, best_ckpt_fn)
    # return data_inst_tag_to_optimized_res



def calculate_obj_traj_diffs(opt_obj_pose, kine_obj_pose):
    # nn_frames x 7 --- for opt obj pose and the kine obj pose # 
    tot_frame_diffs = []
    tot_data_nn_frs = min(opt_obj_pose.shape[0], kine_obj_pose.shape[0])
    
    opt_obj_pose = opt_obj_pose[:tot_data_nn_frs]
    kine_obj_pose = kine_obj_pose[:tot_data_nn_frs]
    opt_obj_pos = opt_obj_pose[:, :3]
    kine_obj_pos = kine_obj_pose[:, :3]
    opt_obj_ornt = opt_obj_pose[:, 3: ]
    kine_obj_ornt = kine_obj_pose[:, 3: ]
    # diff_obj_pos = np.linalg.norm(opt_obj_pos - kine_obj_pos, p= axis=-1)
    opt_obj_pos = torch.from_numpy(opt_obj_pos).float()
    kine_obj_pos = torch.from_numpy(kine_obj_pos).float()
    diff_obj_pos = torch.norm(opt_obj_pos - kine_obj_pos, p=2, dim=-1)
    diff_obj_pos = diff_obj_pos.mean().item()
    
    object_rot = torch.from_numpy(opt_obj_ornt).float()
    target_rot = torch.from_numpy(kine_obj_ornt).float()
    
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
    rot_dist = rot_dist.mean().item() # averge rotation angle differences # []
    
    return diff_obj_pos, rot_dist
    


# def best_optimized_res(data_optimized_res_nn, data_inst_tag, grab_inst_tag):
    
#     # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20230928_030_v2.npy
    
#     optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    
#     kinematics_data_root = "/cephfs/xueyi/data/TACO_Tracking_PK/data/"
#     cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{data_inst_tag}_v2.npy"
    
    
#     cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
#     save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        
        
#     # hand_qs = save_info['robot_delta_states_weights_np'][ : ]
#     # hand_qs = hand_qs[: , : ]
    
#     goal_obj_trans = save_info['object_transl']
#     goal_obj_rot_quat = save_info['object_rot_quat']

#     goal_obj_pose = np.concatenate(
#         [goal_obj_trans, goal_obj_rot_quat],  axis=-1
#     )
    
#     tot_optimized_obj_pose = []
#     tot_optimized_hand_qs = []
#     tot_optimized_hand_qtars = []
#     # object_pose
#     tot_ts = list(optimized_res.keys())
#     tot_ts = sorted(tot_ts)
#     for ts in tot_ts:
#         cur_optimized_obj_pose = optimized_res[ts]['object_pose']
#         tot_optimized_obj_pose.append(cur_optimized_obj_pose)
        
#         cur_optimized_hand_qs = optimized_res[ts]['shadow_hand_dof_pos']
#         cur_optimized_hand_qtars = optimized_res[ts]['shadow_hand_dof_tars']
#         tot_optimized_hand_qs.append(cur_optimized_hand_qs)
#         tot_optimized_hand_qtars.append(cur_optimized_hand_qtars)
        
        
#     tot_optimized_obj_pose = np.stack(tot_optimized_obj_pose, axis=1) # nn_envs x nn_ts x 7 #
#     tot_optimized_hand_qs = np.stack(tot_optimized_hand_qs, axis=1)
#     tot_optimized_hand_qtars = np.stack(tot_optimized_hand_qtars, axis=1)
    
    
#     tot_env_diff_obj_pos, tot_env_diff_obj_rot = [], []
#     tot_env_weighted_obj_pose_diff = []
#     w_pos, w_ornt = 1.0, 0.33
#     for i_env in range(tot_optimized_obj_pose.shape[0]):
#         cur_optimized_obj_pose = tot_optimized_obj_pose[i_env]
#         cur_diff_obj_pos, cur_diff_obj_rot = calculate_obj_traj_diffs(cur_optimized_obj_pose, goal_obj_pose)
#         tot_env_diff_obj_pos.append(cur_diff_obj_pos)
#         tot_env_diff_obj_rot.append(cur_diff_obj_rot)
#         weighted_diff_obj_pose = w_pos * cur_diff_obj_pos + w_ornt * cur_diff_obj_rot
#         tot_env_weighted_obj_pose_diff.append(weighted_diff_obj_pose)
    
#     tot_env_weighted_obj_pose_diff = np.array(tot_env_weighted_obj_pose_diff)
#     sorted_envs_idxes = np.argsort(tot_env_weighted_obj_pose_diff)
#     # top
#     new_optimized_info = {
#         'optimized_obj_pose': tot_optimized_obj_pose[sorted_envs_idxes],
#         'optimized_hand_qs': tot_optimized_hand_qs[sorted_envs_idxes],
#         'optimized_hand_qtars': tot_optimized_hand_qtars[sorted_envs_idxes],
#     }
#     return new_optimized_info
    

# def inspect_optimized_res_nn(data_optimized_res_nn, data_inst_tag):
#     optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
#     print(optimized_res.keys())
    
#     # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
#     # passive_active_info_ori_grab_s1_alarmclock_lift.npy
#     # from the data
    
#     new_optimized_info = best_optimized_res(data_optimized_res_nn, data_inst_tag[0], data_inst_tag[1])
#     optimized_res.update(new_optimized_info)
#     print(optimized_res.keys())
#     ## save to a separte file ##
#     sorted_optimized_res_fn = data_optimized_res_nn.replace(".npy", "_sorted.npy")
    
#     np.save(sorted_optimized_res_fn, optimized_res)
#     print(f"sorted_optimized_res_fn: {sorted_optimized_res_fn}")
    
#     optimized_obj_pose = optimized_res['optimized_obj_pose']
#     optimized_hand_qtars = optimized_res['optimized_hand_qtars']
#     optimized_hand_qs = optimized_res['optimized_hand_qs']
#     best_obj_pose = optimized_obj_pose[0:1]
#     best_hand_qtars = optimized_hand_qtars[0:1]
#     best_hand_qs = optimized_hand_qs[0:1]
    
#     best_ts_to_opt_res_fn = sorted_optimized_res_fn.replace(".npy", "_best.npy")
#     best_ts_to_opt_res = {
#         'optimized_obj_pose': best_obj_pose,
#         'optimized_hand_qtars': best_hand_qtars,
#         'optimized_hand_qs': best_hand_qs
#     }
#     np.save(best_ts_to_opt_res_fn, best_ts_to_opt_res)
#     print(f"Saved best ts_to_opt_res to {best_ts_to_opt_res_fn}")

    
#     # for key in optimized_res:
#     #     # cur_grab_inst_tag = key
#     #     # cur_grab_
#     #     print(optimized_res[key].keys())
#     #     break
#     #     # val = optimized_res[key]
#         # print(f"key: {key}, val: {val.shape}")

# def resave_optimized_res(data_inst_tag_to_optimized_res_fn):
#     data_inst_tag_to_res = np.load(data_inst_tag_to_optimized_res_fn, allow_pickle=True).item()
#     print(data_inst_tag_to_res.keys())
#     cur_nn = 0
#     for inst_tag in data_inst_tag_to_res:
#         print(f"[{cur_nn}/{len(data_inst_tag_to_res)}] {inst_tag}")
#         cur_inst_tag_optimized_res = data_inst_tag_to_res[inst_tag]
#         for cur_inst_tag_optimized_res_fn in cur_inst_tag_optimized_res:
#             inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag)
#         cur_nn = cur_nn + 1




def best_optimized_res(data_optimized_res_nn, data_inst_tag, grab_inst_tag, index=None, additional_tag=''):
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    
    # grab_inst_tag = ""
    # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    kinematics_data_root = "/cephfs/xueyi/data/TACO_Tracking_PK/data/"
    # cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
    if len(additional_tag) == 0:    
        kine_data_inst_tag = data_inst_tag
        # kine_data_inst_tag = "taco_20230928_031"
        # cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{data_inst_tag}_v2.npy"
        cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{kine_data_inst_tag}_v2.npy"
    else:
        kine_data_inst_tag = data_inst_tag
        # kine_data_inst_tag = "taco_20230928_031"
        # cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{data_inst_tag}_v2_{additional_tag}.npy"
        cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{kine_data_inst_tag}_v2_{additional_tag}.npy"
    cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
    save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        
        
    # hand_qs = save_info['robot_delta_states_weights_np'][ : ]
    # hand_qs = hand_qs[: , : ]
    
    goal_obj_trans = save_info['object_transl']
    goal_obj_rot_quat = save_info['object_rot_quat']

    goal_obj_pose = np.concatenate(
        [goal_obj_trans, goal_obj_rot_quat],  axis=-1
    )
    
    tot_optimized_obj_pose = []
    tot_optimized_hand_qs = []
    tot_optimized_hand_qtars = []
    # object_pose
    tot_ts = list(optimized_res.keys())
    tot_ts = [ cur_ts for cur_ts in tot_ts if isinstance(cur_ts, int) ]
    tot_ts = sorted(tot_ts)
    for ts in tot_ts:
        cur_optimized_obj_pose = optimized_res[ts]['object_pose']
        tot_optimized_obj_pose.append(cur_optimized_obj_pose)
        
        cur_optimized_hand_qs = optimized_res[ts]['shadow_hand_dof_pos']
        cur_optimized_hand_qtars = optimized_res[ts]['shadow_hand_dof_tars']
        tot_optimized_hand_qs.append(cur_optimized_hand_qs)
        tot_optimized_hand_qtars.append(cur_optimized_hand_qtars)
        
        
    tot_optimized_obj_pose = np.stack(tot_optimized_obj_pose, axis=1) # nn_envs x nn_ts x 7 #
    tot_optimized_hand_qs = np.stack(tot_optimized_hand_qs, axis=1)
    tot_optimized_hand_qtars = np.stack(tot_optimized_hand_qtars, axis=1)
    
    if index is not None:
        tot_optimized_obj_pose = tot_optimized_obj_pose[index: index + 1]
        tot_optimized_hand_qs = tot_optimized_hand_qs[index: index + 1]
        tot_optimized_hand_qtars = tot_optimized_hand_qtars[index: index + 1]
    
    tot_env_diff_obj_pos, tot_env_diff_obj_rot = [], []
    tot_env_weighted_obj_pose_diff = []
    w_pos, w_ornt = 1.0, 0.33
    for i_env in range(tot_optimized_obj_pose.shape[0]):
        cur_optimized_obj_pose = tot_optimized_obj_pose[i_env]
        cur_diff_obj_pos, cur_diff_obj_rot = calculate_obj_traj_diffs(cur_optimized_obj_pose, goal_obj_pose)
        tot_env_diff_obj_pos.append(cur_diff_obj_pos)
        tot_env_diff_obj_rot.append(cur_diff_obj_rot)
        weighted_diff_obj_pose = w_pos * cur_diff_obj_pos + w_ornt * cur_diff_obj_rot
        tot_env_weighted_obj_pose_diff.append(weighted_diff_obj_pose)
    
    tot_env_weighted_obj_pose_diff = np.array(tot_env_weighted_obj_pose_diff)
    sorted_envs_idxes = np.argsort(tot_env_weighted_obj_pose_diff)
    tot_env_diff_obj_pos = np.array(tot_env_diff_obj_pos)
    tot_env_diff_obj_rot = np.array(tot_env_diff_obj_rot)
    # top
    new_optimized_info = {
        'optimized_obj_pose': tot_optimized_obj_pose[sorted_envs_idxes],
        'optimized_hand_qs': tot_optimized_hand_qs[sorted_envs_idxes],
        'optimized_hand_qtars': tot_optimized_hand_qtars[sorted_envs_idxes],
        'obj_pose_diff': tot_env_weighted_obj_pose_diff[sorted_envs_idxes],
        'obj_pos_diff': tot_env_diff_obj_pos[sorted_envs_idxes],
        'obj_rot_diff': tot_env_diff_obj_rot[sorted_envs_idxes],
    }
    return new_optimized_info
    
    



def inspect_optimized_res_nn(data_optimized_res_nn, data_inst_tag, additional_tag=''):
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    print(optimized_res.keys())
    
    
    # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # passive_active_info_ori_grab_s1_alarmclock_lift.npy
    # from the data
    
    new_optimized_info = best_optimized_res(data_optimized_res_nn, data_inst_tag[0], data_inst_tag[1], additional_tag=additional_tag)
    optimized_res.update(new_optimized_info)
    print(optimized_res.keys())
    ## save to a separte file ##
    # sorted optimized res fn #
    sorted_optimized_res_fn = data_optimized_res_nn.replace(".npy", "_sorted.npy")
    
    np.save(sorted_optimized_res_fn, optimized_res)
    print(f"sorted_optimized_res_fn: {sorted_optimized_res_fn}")
    
    optimized_obj_pose = optimized_res['optimized_obj_pose']
    optimized_hand_qtars = optimized_res['optimized_hand_qtars']
    optimized_hand_qs = optimized_res['optimized_hand_qs']
    best_obj_pose = optimized_obj_pose[0:1]
    best_hand_qtars = optimized_hand_qtars[0:1]
    best_hand_qs = optimized_hand_qs[0:1]
    # best obj pos diff #
    # best_obj_pose_diff = optimized_res['tot_env_weighted_obj_pose_diff'][0:1]
    # best_obj_pos_diff = optimized_res['']
    
    best_obj_pose_diff_new = optimized_res['obj_pose_diff'][0:1]
    best_obj_pose_diff = best_obj_pose_diff_new
    best_obj_pos_diff_new = optimized_res['obj_pos_diff'][0:1]
    best_obj_rot_diff_new = optimized_res['obj_rot_diff'][0:1] 
    
    
    best_ts_to_opt_res_fn = sorted_optimized_res_fn.replace(".npy", "_best_vv.npy")
    best_ts_to_opt_res = {
        'optimized_obj_pose': best_obj_pose,
        'optimized_hand_qtars': best_hand_qtars,
        'optimized_hand_qs': best_hand_qs,
        'obj_pose_diff': best_obj_pose_diff,
        'best_obj_pose_diff': best_obj_pose_diff_new,
        'best_obj_pos_diff': best_obj_pos_diff_new,
        'best_obj_rot_diff': best_obj_rot_diff_new
    }
    np.save(best_ts_to_opt_res_fn, best_ts_to_opt_res)
    print(f"Saved best ts_to_opt_res to {best_ts_to_opt_res_fn}")

    
    # for key in optimized_res:
    #     # cur_grab_inst_tag = key
    #     # cur_grab_
    #     print(optimized_res[key].keys())
    #     break
    #     # val = optimized_res[key]
        # print(f"key: {key}, val: {val.shape}")



# inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag)
# runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-19-10-33/ts_to_hand_obj_obs_reset_1.npy
def resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=None, additional_tag=''):
    data_inst_tag_to_res = np.load(data_inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    print(data_inst_tag_to_res.keys()) # calculate the optimization tracking results and save them int separate files #
    if target_sub_tag is not None:
        data_inst_tag_to_res = {
            key: data_inst_tag_to_res[key] for key in data_inst_tag_to_res if target_sub_tag in key[0]
        }
    
    cur_nn = 0
    for inst_tag in data_inst_tag_to_res:
        print(f"[{cur_nn}/{len(data_inst_tag_to_res)}] {inst_tag}")
        if inst_tag[0] == 'taco_20231024_184':
            continue
        cur_inst_tag_optimized_res = data_inst_tag_to_res[inst_tag]
        for cur_inst_tag_optimized_res_fn in cur_inst_tag_optimized_res:
            inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag, additional_tag=additional_tag)
        cur_nn = cur_nn + 1



# 

def inspect_eval_data_info(eval_sv_info):
    eval_data = np.load(eval_sv_info, allow_pickle=True).item()
    print(eval_data.keys())
    optimized_obj_pose = eval_data['optimized_obj_pose'][0]
    print(optimized_obj_pose.shape)
    first_frame_obj_pose = optimized_obj_pose[0]
    print(first_frame_obj_pose)

# /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_TACO_taco_20230930_055_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-07-11-34
#  

def parse_time_from_tracking_folder_nm(folder_nm):
    folder_nm_segs = folder_nm.split("_")
    time_segs = folder_nm_segs[-1]
    time_segs = time_segs.split("-")
    time_segs = [ float(ts) for ts in time_segs ]
    ts_to_secs = [ 24 * 60 * 60, 60 * 60, 60, 1 ]
    time_secs = 0
    
    for i_ts, cur_ts in enumerate(time_segs):
        time_secs += cur_ts * ts_to_secs[i_ts]
    return time_secs
    # return time_segs

def mv_eval_data_to_target_folder(local_folder, target_folder, subj_idx=None, exclude_existing_fn=None):
    if exclude_existing_fn is not None:
        exclude_existing_data_inst_to_opt_res = np.load(exclude_existing_fn, allow_pickle=True).item()
        exclude_existing_data_inst_tags = {
            key[0]: 1 for key in exclude_existing_data_inst_to_opt_res
        }
    else:
        exclude_existing_data_inst_to_opt_res = None
        exclude_existing_data_inst_tags = {}
    
    tot_eval_fns = os.listdir(local_folder)
    tracking_data_sv_st_tag = "tracking_"
    tot_eval_fns = [
        fn for fn in tot_eval_fns if fn[: len(tracking_data_sv_st_tag)] == tracking_data_sv_st_tag
    ]
    if subj_idx is not None:
        subj_tag = f'_s{subj_idx}_'
        tot_eval_fns = [ fn for fn in tot_eval_fns if subj_tag in fn ]
    
    tot_eval_fns_w_time = []
    for cur_fn in tot_eval_fns:
        cur_time_secs = parse_time_from_tracking_folder_nm(cur_fn)
        
        
        tot_eval_fns_w_time.append((cur_fn, cur_time_secs))
        
    sorted_eval_fns = sorted(tot_eval_fns_w_time, key=lambda x: x[1])
    print(sorted_eval_fns)
    
    
    
    # for idx in # d
    # 
    
    
    for idx in range(0, len(sorted_eval_fns) - 9):
        cur_fn = sorted_eval_fns[idx][0]
        
        cur_exp_obj_name = parse_obj_type_from_taco_grab_folder_fn(cur_fn)
        
        if exclude_existing_data_inst_to_opt_res is not None:
            if cur_exp_obj_name in exclude_existing_data_inst_tags:
                continue
        
        cur_full_fn = os.path.join(local_folder, cur_fn)
        
        
        print(f"mv {cur_full_fn} {target_folder}/")
        os.system(f"mv {cur_full_fn} {target_folder}/")
    
    # for i_fn, cur_fn in enumerate(tot_eval_fns):
    #     cur_full_fn = os.path.join(local_folder, cur_fn)
    #     print(f"[{i_fn}/{len(tot_eval_fns)}] mv {cur_full_fn} {target_folder}/")
    #     os.system(f"mv {cur_full_fn} {target_folder}/")

#  test #
# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy
def inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn):
    inst_tag_to_optimized_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    for key in inst_tag_to_optimized_res:
        optimized_res = inst_tag_to_optimized_res[key]
        print(f"key: {key}, optimized_res: {optimized_res}")
    print(len(inst_tag_to_optimized_res))


# parse_obj_type_from_taco_grab_folder_fn # 

# 
def calculate_tot_taco_info(taco_retar_info_root):
    taco_retar_data = os.listdir(taco_retar_info_root)
    # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231006_146_v2.npy
    st_tag = "passive_active_info_ori_grab_s2_phone_call_1"
    taco_retar_data = [
        fn for fn in taco_retar_data if fn.endswith(".npy") and fn[: len(st_tag)] == st_tag and "taco_" in fn and "v2" in fn and "_v2.npy" in fn
    ]
    taco_inst_dict = {}
    for cur_fn in taco_retar_data:
        cur_fn_segs = cur_fn.split(".")[0].split("_")
        cur_inst_tag = cur_fn_segs[-4: -1]
        cur_inst_tag = "_".join(cur_inst_tag)
        taco_inst_dict[cur_inst_tag] = cur_fn
    print(len(taco_inst_dict))
    print(taco_inst_dict.keys())
    
    optimized_res_beta_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res_beta.npy"
    optimized_res_beta = np.load(optimized_res_beta_fn, allow_pickle=True).item()
    print(len(optimized_res_beta))
    remaining_taco_inst_dict = {}
    for cur_inst_tag in taco_inst_dict:
        if cur_inst_tag not in optimized_res_beta:
            remaining_taco_inst_dict[cur_inst_tag] = taco_inst_dict[cur_inst_tag]
            # 
    print(len(remaining_taco_inst_dict))

# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res_beta.npy
def inspect_obj_type_to_optimized_res_beta(obj_type_to_optimized_res_fn):
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item()
    tot_obj_type_keys = list(obj_type_to_optimized_res.keys())
    obj_type_key = tot_obj_type_keys[0]
    print(f"obj_type_key: {obj_type_key}")


def inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn):
    
    inst_tag_to_optimized_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    print(f"nn: {len(inst_tag_to_optimized_res)}")
    # tot_inst_tags = list(inst_tag_to_optimized_res.keys())
    # inst_tag = tot_inst_tags[0]
    # print(f"inst_tag: {inst_tag}")
    for key in inst_tag_to_optimized_res:
        print(f"key: {key}")



def modify_inst_tag_to_opt_res(inst_tag_to_opt_res_dict_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_dict_fn, allow_pickle=True).item()
    new_dict = {}
    for key in inst_tag_to_opt_res:
        if isinstance(key, tuple):
            new_key = key[0]
        else:
            new_key = key
        taco_idxes = new_key.split("_")[:2]
        taco_idxes = "_".join(['taco'] + taco_idxes)
        grab_inst_tag = "ori_grab_s2_phone_call_1"
        new_key = (taco_idxes, grab_inst_tag)
        # new_dict[new_key] = inst_tag_to_opt_res[key]
        new_dict[new_key] = inst_tag_to_opt_res[key]
        print(f"new_key: {new_key}")
    dict_folder = "/".join(inst_tag_to_opt_res_dict_fn.split("/")[:-1])
    dict_nm = inst_tag_to_opt_res_dict_fn.split("/")[-1]
    dict_nm = dict_nm.replace(".npy", "_v2.npy")
    new_dict_fn = os.path.join(dict_folder, dict_nm)
    np.save(new_dict_fn, new_dict)
    print(f"new_dict_fn: {new_dict_fn}")





def get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag=None, random_select=False):
    data_inst_tag_to_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    # print(data_inst_tag_to_res.keys())
    cur_nn = 0
    data_inst_tag_to_best_opt_res = {}
    for inst_tag in data_inst_tag_to_res:
        if target_subj_tag is not None:
            if target_subj_tag not in inst_tag[0]:
                continue
        cur_inst_opt_res_fn = data_inst_tag_to_res[inst_tag][0]
        
        # if not os.path.exists(cur_inst_opt_res_fn):
        #     changed_root_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2"
        #     pure_fn = "/".join(cur_inst_opt_res_fn.split("/")[-2:])
        #     actual_res_fn = os.path.join(changed_root_folder, pure_fn)
        # else:
        actual_res_fn = cur_inst_opt_res_fn
        
        cur_inst_opt_res_fn_sorted = actual_res_fn.replace(".npy", "_sorted.npy")
        if random_select:
            cur_inst_opt_res_sorted = np.load(cur_inst_opt_res_fn_sorted, allow_pickle=True).item() # jet the opt res fn sorted #
            # optimized_obj_pose = cur_inst_opt_res_sorted['optimized_obj_pose']
            # optimized_obj_pose_diff = cur_inst_opt_res_sorted['optimized_hand_qtars']
            obj_pos_diff = cur_inst_opt_res_sorted['obj_pos_diff'] # nn_envs
            obj_pose_diff = cur_inst_opt_res_sorted['obj_pose_diff']
            obj_rot_diff = cur_inst_opt_res_sorted['obj_rot_diff']
            rnd_selected_nn = 8
            rnd_selected_idxes = np.random.choice(len(obj_pos_diff), rnd_selected_nn, replace=False)
            obj_pos_diff = obj_pos_diff[rnd_selected_idxes]
            obj_pose_diff = obj_pose_diff[rnd_selected_idxes]
            obj_rot_diff = obj_rot_diff[rnd_selected_idxes]
            sorted_pos_diff_idxes = np.argsort(obj_pos_diff)
            obj_pose_diff = obj_pose_diff[sorted_pos_diff_idxes]
            obj_rot_diff = obj_rot_diff[sorted_pos_diff_idxes]
            obj_pos_diff = obj_pos_diff[sorted_pos_diff_idxes]
            # best_obj_pose_diff = 
            # sorted_pose_diff_idxes = np.argsort(obj_pose_diff)
            # sorted_rot_diff_idxes = np.argsort(obj_rot_diff)
            # # obj_pos_diff = obj_pos_diff[sorted_pos_diff_idxes]
            best_obj_pose_diff = obj_pose_diff[0:1]
            best_obj_pos_diff = obj_pos_diff[0:1]
            best_obj_rot_diff = obj_rot_diff[0:1]
        else:
            
            cur_inst_opt_res_fn_sorted_best = cur_inst_opt_res_fn_sorted.replace(".npy", "_best.npy")
            if not os.path.exists(cur_inst_opt_res_fn_sorted_best):
                cur_inst_opt_res_fn_sorted_best = cur_inst_opt_res_fn_sorted.replace(".npy", "_best_vv.npy")
            if not os.path.exists(cur_inst_opt_res_fn_sorted_best):
                print(f"cur_inst_opt_res_fn_sorted_best: {cur_inst_opt_res_fn_sorted_best} not exists")
                continue
            cur_inst_opt_res = np.load(cur_inst_opt_res_fn_sorted_best, allow_pickle=True).item()
            # dict_keys(['optimized_obj_pose', 'optimized_hand_qtars', 'optimized_hand_qs', 'obj_pose_diff', 'best_obj_pose_diff', 'best_obj_pos_diff', 'best_obj_rot_diff'])
            best_obj_pose_diff = cur_inst_opt_res['best_obj_pose_diff']
            best_obj_pos_diff = cur_inst_opt_res['best_obj_pos_diff']
            best_obj_rot_diff = cur_inst_opt_res['best_obj_rot_diff']
            
        data_inst_tag_to_best_opt_res[inst_tag] = {
            'obj_pos_diff': best_obj_pos_diff,
            'obj_rot_diff': best_obj_rot_diff,
            'obj_pose_diff': best_obj_pose_diff,
        }
        
        
        # print(f"[{cur_nn}/{len(data_inst_tag_to_res)}] {inst_tag}")
        # cur_inst_tag_optimized_res = data_inst_tag_to_res[inst_tag]
        # for cur_inst_tag_optimized_res_fn in cur_inst_tag_optimized_res:
        #     inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag)
        # cur_nn = cur_nn + 1
    return data_inst_tag_to_best_opt_res
    pass



def calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b):
    
    obj_type_to_optimized_res = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy"
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res, allow_pickle=True).item()
    training_objs = []
    succ_rew_thres = 50.0
    for obj_type in obj_type_to_optimized_res:
        if obj_type_to_optimized_res[obj_type][0] > succ_rew_thres:
            training_objs.append(obj_type)
    print(len(training_objs), "/", len(obj_type_to_optimized_res), len(training_objs) / len(obj_type_to_optimized_res))
    
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    data_inst_to_opt_res_b = np.load(data_inst_to_opt_res_fn_b, allow_pickle=True).item()
    pos_thres = 0.14
    pos_thres = 0.10
    ornt_thres = 0.6981317007977318
    # ornt_thres = 0.3490658503988659
    # ornt_thres = 0.17453292519943295
    # pos_thres = 0.05
    # ornt_thres = 0.6981317007977318 # 
    # 0.17453292519943295
    # 0.2617993877991494
    # 0.3490658503988659 # 20 degree
    # 0.5235987755982988 # 30 degree
    # 0.6981317007977318 # 40 degree
    tot_tot_nn = []
    tot_succ_nn = []
    # for subj_inst_idx in range(1, 11):
    tot_nn = 0
    succ_nn = 0

    #     # target_inst_tag = 's2'

    #     # target_inst_tag = 's1'
    #     # target_inst_tag = f's{subj_inst_idx}_'

    for cur_inst_tag in data_inst_to_opt_res_a:
        
        if cur_inst_tag[0] in training_objs:
            continue
        
        # if isinstance(cur_inst_tag, tuple):
        #     if target_inst_tag not in cur_inst_tag[0]: 
        #         continue
        cur_data_inst_val = data_inst_to_opt_res_a[cur_inst_tag]
        cur_a_pos_diff = cur_data_inst_val['obj_pos_diff'][0].item()
        cur_a_rot_diff = cur_data_inst_val['obj_rot_diff'][0].item()
        if cur_inst_tag in data_inst_to_opt_res_b:
            cur_b_inst_val = data_inst_to_opt_res_b[cur_inst_tag]
            cur_b_pos_diff = cur_b_inst_val['obj_pos_diff'][0].item()
            cur_b_rot_diff = cur_b_inst_val['obj_rot_diff'][0].item()
            cur_a_pose_diff = cur_a_pos_diff + cur_a_rot_diff * 0.33
            cur_b_pose_diff = cur_b_pos_diff + cur_b_rot_diff * 0.33
            if cur_b_pose_diff < cur_a_pose_diff:
                print(f"cur_inst_tag: {cur_inst_tag}, cur_a_pos_diff: {cur_a_pos_diff}, cur_a_rot_diff: {cur_a_rot_diff}, cur_b_pos_diff: {cur_b_pos_diff}, cur_b_rot_diff: {cur_b_rot_diff}")
                cur_a_pos_diff = cur_b_pos_diff
                cur_a_rot_diff = cur_b_rot_diff

        if cur_a_pos_diff < pos_thres and cur_a_rot_diff < ornt_thres: # --- get the single model #
        # if cur_a_pos_diff < pos_thres:
            succ_nn += 1
        tot_nn += 1
    tot_tot_nn.append(tot_nn)
    tot_succ_nn.append(succ_nn)
    print(f"succ_nn/tot_nn: {succ_nn}/{tot_nn}")
    tot_tot_nn = sum(tot_tot_nn)
    tot_succ_nn = sum(tot_succ_nn)
    res = float(tot_succ_nn) / float(tot_tot_nn)
    print(f"tot_succ_nn/tot_tot_nn: {tot_succ_nn}/{tot_tot_nn}, res: {res}")



def calcualte_merged_succ_info_all_thres(data_inst_to_opt_res_fn_a, target_inst_tags=None):
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    # data_inst_to_opt_res_b = np.load(data_inst_to_opt_res_fn_b, allow_pickle=True).item()
    pos_thres = 0.15
    
    tot_pos_threses = [0.10, 0.10, 0.15]
    tot_ornt_threses = [0.5235987755982988, 0.6981317007977318, 0.6981317007977318]
    # tot_ornt_threses = [0.3490658503988659, 0.6981317007977318, 0.6981317007977318]
    
    tot_succ_rates = []
    
    
    for i_idx in range(len(tot_pos_threses)):
        pos_thres = tot_pos_threses[i_idx]
        ornt_thres = tot_ornt_threses[i_idx]
        # ornt_thres = 0.3490658503988659
        # ornt_thres = 0.17453292519943295
        # pos_thres = 0.05
        # ornt_thres = 0.6981317007977318 # 
        # 0.17453292519943295
        # 0.2617993877991494
        # 0.3490658503988659 # 20 degree
        # 0.5235987755982988 # 30 degree
        # 0.6981317007977318 # 40 degree
        tot_tot_nn = []
        tot_succ_nn = []
        # for subj_inst_idx in range(1, 11):
        tot_nn = 0
        succ_nn = 0

        #     target_inst_tag = 's2'

        # target_inst_tag = 's1'
        # target_inst_tag = f's{subj_inst_idx}_'

        for cur_inst_tag in data_inst_to_opt_res_a:
            # if isinstance(cur_inst_tag, tuple):
            #     if target_inst_tag not in cur_inst_tag[0]: 
            #         continue
            if 'taco_20231024_' not in cur_inst_tag[0]:
                continue
            
            # target inst tags #
            if target_inst_tags is not None: # 
                if cur_inst_tag[0] not in target_inst_tags:
                    continue
            
            cur_data_inst_val = data_inst_to_opt_res_a[cur_inst_tag]
            cur_a_pos_diff = cur_data_inst_val['obj_pos_diff'][0].item()
            cur_a_rot_diff = cur_data_inst_val['obj_rot_diff'][0].item()
            # if cur_inst_tag in data_inst_to_opt_res_b:
            #     cur_b_inst_val = data_inst_to_opt_res_b[cur_inst_tag]
            #     cur_b_pos_diff = cur_b_inst_val['obj_pos_diff'][0].item()
            #     cur_b_rot_diff = cur_b_inst_val['obj_rot_diff'][0].item()
            #     cur_a_pose_diff = cur_a_pos_diff + cur_a_rot_diff * 0.33
            #     cur_b_pose_diff = cur_b_pos_diff + cur_b_rot_diff * 0.33
            #     if cur_b_pose_diff < cur_a_pose_diff:
            #         print(f"cur_inst_tag: {cur_inst_tag}, cur_a_pos_diff: {cur_a_pos_diff}, cur_a_rot_diff: {cur_a_rot_diff}, cur_b_pos_diff: {cur_b_pos_diff}, cur_b_rot_diff: {cur_b_rot_diff}")
            #         cur_a_pos_diff = cur_b_pos_diff
                    # cur_a_rot_diff = cur_b_rot_diff

            if cur_a_pos_diff < pos_thres and cur_a_rot_diff < ornt_thres: # --- get the single model #
            # if cur_a_pos_diff < pos_thres:
                succ_nn += 1
            tot_nn += 1
        tot_tot_nn.append(tot_nn)
        tot_succ_nn.append(succ_nn)
        print(f"succ_nn/tot_nn: {succ_nn}/{tot_nn}")
        
        
        tot_tot_nn = sum(tot_tot_nn)
        tot_succ_nn = sum(tot_succ_nn)
        res = float(tot_succ_nn) / float(tot_tot_nn)
        print(f"tot_succ_nn/tot_tot_nn: {tot_succ_nn}/{tot_tot_nn}, res: {res}")
        tot_succ_rates.append(res)
    print(tot_succ_rates)





def compare_obj_type_optimized_res(obj_type_to_opt_res_a, obj_type_to_opt_res_b):
    obj_type_to_opt_res_a = np.load(obj_type_to_opt_res_a, allow_pickle=True).item()
    obj_type_to_opt_res_b = np.load(obj_type_to_opt_res_b, allow_pickle=True).item()
    for key in obj_type_to_opt_res_a:
        if key in obj_type_to_opt_res_b:
            key_res_a = obj_type_to_opt_res_a[key][0]
            key_res_b = obj_type_to_opt_res_b[key][0]
            print(f"key: {key}, key_res_a: {key_res_a}, key_res_b: {key_res_b}")



def get_good_inst_tag_base_trajectories(inst_tag_to_optimized_res_fn, obj_type_to_ooptimized_res_fn, rew_filter_thres=95.0):
    inst_tag_to_optimized_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    obj_type_to_optimized_res = np.load(obj_type_to_ooptimized_res_fn, allow_pickle=True).item()
    # print(f"inst_tag_to_optimized_res:obj_type_to_optimized_res.keys())[0]}")
    good_inst_tag_to_optimized_res = {}
    for inst_tag in obj_type_to_optimized_res:
        
        inst_tag_tuple = (inst_tag, 'ori_grab_s2_phone_call_1')
        if inst_tag_tuple not in inst_tag_to_optimized_res:
            continue
        
        cur_obj_opt_res = obj_type_to_optimized_res[inst_tag][0]
        # if isinstance(inst_tag, tuple):
        #     cur_obj_type = inst_tag[0]
        # else:
        #     cur_obj_type = inst_tag
        # # if is                                                                                                       instance(obj_type_to_optimized_res.keys()[0], tuple):
        raw_cur_inst_tag_opt_res = inst_tag_to_optimized_res[inst_tag_tuple] # [0]
        cur_inst_tag_opt_res = []
        for cur_fn in raw_cur_inst_tag_opt_res:
            sorted_cur_fn = cur_fn.replace(".npy", "_sorted.npy")
            best_cur_fn = sorted_cur_fn.replace(".npy", "_best.npy")
            cur_inst_tag_opt_res.append(best_cur_fn)
        
        if cur_obj_opt_res >= rew_filter_thres:
            good_inst_tag_to_optimized_res[inst_tag_tuple] = cur_inst_tag_opt_res
            print(f"inst_tag: {inst_tag}, cur_obj_opt_res: {cur_obj_opt_res}, cur_inst_tag_opt_res: {cur_inst_tag_opt_res}")
        
        # # # if cur_obj_type not in obj_type_to_optimized_res:
        # # #     continue
        # # cur_obj_type_opt_res = obj_type_to_optimized_res[cur_obj_type]
        # # cur_obj_type_opt_res = cur_obj_type_opt_res[0]
        # # cur_obj_type_opt_res = cur_obj_type_opt_res['obj_pos_diff']
        # if cur_obj_type_opt_res > rew_filter_thres:
        #     good_inst_tag_to_optimized_res[inst_tag] = cur_inst_tag_opt_res
    return good_inst_tag_to_optimized_res

def inspect_eval_fns(eval_folder):
    tot_fns = os.listdir(eval_folder)
    folder_st_nm = "tracking_"
    tot_fns = [ fn for fn in tot_fns if fn[: len(folder_st_nm)] == folder_st_nm ]
    print(len(tot_fns))


def inspect_two_inst_tag_to_opt_res(inst_tag_to_opt_res_a, inst_tag_to_opt_res_b):
    inst_tag_to_opt_res_a = np.load(inst_tag_to_opt_res_a, allow_pickle=True).item()
    inst_tag_to_opt_res_b = np.load(inst_tag_to_opt_res_b, allow_pickle=True).item()
    merged_dict = {}
    merged_dict.update(inst_tag_to_opt_res_a)
    merged_dict.update(inst_tag_to_opt_res_b)
    # for key in inst_tag_to_opt_res_b:
        
        # if key in inst_tag_to_opt_res_a:
        #     key_res_a = inst_tag_to_opt_res_a[key][0]
        #     key_res_b = inst_tag_to_opt_res_b[key][0]
        #     print(f"key: {key}, key_res_a: {key_res_a}, key_res_b: {key_res_b}")
    print(len(merged_dict))
    merged_dict_sv_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta_v2.npy"
    np.save(merged_dict_sv_fn, merged_dict)
    print(f"Saved merged_dict to {merged_dict_sv_fn}")


def get_sorted_taco_optimized_fns():
    taco_inst_tag_to_rew = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy"
    taco_inst_tag_to_rew = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300/statistics/obj_type_to_optimized_res.npy"
    taco_inst_tag_to_rew = np.load(taco_inst_tag_to_rew, allow_pickle=True).item()
    
    # taco_inst_tag_to_rew_items = [
    #     (key, taco_inst_tag_to_rew[key][0]) for key in taco_inst_tag_to_rew
    # ]
    # # 
    
    taco_inst_tag_to_rew_items = [
        (key, taco_inst_tag_to_rew[key][0]) for key in taco_inst_tag_to_rew if '_s1_' in key
    ]
    
    eval_split_test_inst_tag_to_rew_items = taco_inst_tag_to_rew_items
    
    # eval_split_test_inst_tag_to_rew_items = []
    # for item in taco_inst_tag_to_rew_items:
    #     cur_key = item[0]
    #     test_taco_tag = 'taco_20231024_'
    #     if isinstance(cur_key, tuple):
    #         if test_taco_tag in cur_key[0]:
    #             eval_split_test_inst_tag_to_rew_items.append(item)
    #     else:
    #         if test_taco_tag in cur_key:
    #             eval_split_test_inst_tag_to_rew_items.append(item)
            
    
    taco_inst_tag_to_rew_items = sorted(eval_split_test_inst_tag_to_rew_items, key=lambda x: x[1], reverse=True)
    # taco_inst_tag_to_rew_items = taco_inst_tag_to_rew_items[: 10]
    nn_max = 50
    for idx in range(nn_max):
        cur_item = taco_inst_tag_to_rew_items[idx]
        print(f"cur_item: {cur_item}")
        # print(f"key: {key}, val: {val}")
    # print(taco_inst_tag_to_rew_items)

def inspect_taco_optimized_res():
    test_taco_tag = "taco_20231024_"
    taco_inst_tag_to_rew = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy"
    taco_inst_tag_to_rew = np.load( taco_inst_tag_to_rew, allow_pickle=True).item()
    taco_inst_tag_to_rew_items = [ ]
    for key in taco_inst_tag_to_rew:
        if isinstance(key, tuple):
            cur_key = key[0]
        else:
            cur_key = key
        if test_taco_tag in cur_key:
            taco_inst_tag_to_rew_items.append((key, taco_inst_tag_to_rew[key][0]))
    
    taco_inst_tag_to_rew_items = sorted(taco_inst_tag_to_rew_items, key=lambda x: x[1], reverse=True)
    for i in range(20):
        cur_item = taco_inst_tag_to_rew_items[i]
        print(f"cur_item: {cur_item}")
        # print(f"key: {key}, val: {val}")



def get_good_obj_type_to_optimized_res(opt_res_fn):
    opt_res = np.load(opt_res_fn, allow_pickle=True).item()
    


# and also clone some kinemaitcs info here #

# def copy_optimized_info

# interested_keys = ['taco_20231024_176', 'taco_20231024_045', 'taco_20231024_169', 'taco_20231024_124', 'taco_20231024_070']
# data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    

def copy_optimized_infos(interested_keys, data_inst_to_optimized_res_fn, dst_folder=None):
    # keys and the optimized res fn #
    # interested_keys = ["taco_20231104_035", "taco_20231027_008", "taco_20231020_036"]
    # data_inst_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta_v2.npy"
    
    # interested_keys = ['taco_20231024_176', 'taco_20231024_045', 'taco_20231024_169', 'taco_20231024_124', 'taco_20231024_070']
    # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    
    
        
    
    data_inst_to_optimized_res = np.load(data_inst_to_optimized_res_fn, allow_pickle=True).item()
    
    if interested_keys is None:
        interested_keys = list(data_inst_to_optimized_res.keys())
        interested_keys = [ cur_key[0] for cur_key in interested_keys ]
    
    if dst_folder is None:
        dst_folder = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res"
    os.makedirs(dst_folder, exist_ok=True)
    print(list(data_inst_to_optimized_res.keys())[0],list(data_inst_to_optimized_res.values())[0] )
    for key in interested_keys:
        if 'taco' in key:
            cur_key_tuple =    (key, 'ori_grab_s2_phone_call_1')
            kine_ref_fn = f"/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{key}_v2_interpfr_60_interpfr2_60_nntrans_40.npy"
        else:
            cur_key_tuple = (key, key)
            if '_s1_' not in key:
                continue
            # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_peel_1_nf_300.npy
            kine_ref_fn = f"/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_{key}.npy"
            
        
            
        opt_res = data_inst_to_optimized_res[cur_key_tuple][0]
        opt_folder = opt_res.split("/")[-2]
        cur_dst_folder = os.path.join(dst_folder, opt_folder)
        os.makedirs(cur_dst_folder, exist_ok=True)
        opt_res_fn = opt_res.replace(".npy", "_sorted.npy").replace(".npy", "_best.npy")
        if not os.path.exists(opt_res_fn):
            opt_res_fn = opt_res.replace(".npy", "_sorted.npy").replace(".npy", "_best_vv.npy")
        cp_cmd = f"cp {opt_res_fn} {cur_dst_folder}/"
        print(cp_cmd)
        os.system(cp_cmd)
        
        
        
        kine_sv_fn = os.path.join(cur_dst_folder, kine_ref_fn.split("/")[-1])
        cp_cmd =    f"cp {kine_ref_fn} {kine_sv_fn}"
        print(cp_cmd)
        os.system(cp_cmd)


def modify_coacd_mesh_fn(test=True):
    mesh_folder = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    vis_mesh_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
    tot_inst_mesh_folders = os.listdir(mesh_folder)
    tot_inst_mesh_folders = [
        fn for fn in tot_inst_mesh_folders if 'taco_' in fn
    ]
    for fn in tot_inst_mesh_folders:
        cur_vis_mesh_fn = f"{fn}.obj"
        cur_vis_mesh_fn = os.path.join(vis_mesh_root, cur_vis_mesh_fn)
        cur_urdf_folder = os.path.join(mesh_folder, fn, "coacd")
        cur_urdf_vis_mesh_fn = os.path.join(cur_urdf_folder, f"{fn}.obj")
        cp_cmd = f"cp {cur_vis_mesh_fn} {cur_urdf_vis_mesh_fn}"
        print(cp_cmd)
        os.system(cp_cmd)
        cur_urdf_fn = os.path.join(cur_urdf_folder, f"coacd_1.urdf")
        cur_urdf_vis_fn = os.path.join(cur_urdf_folder, f"coacd_1_vis.urdf")
        cur_urdf_content = open(cur_urdf_fn, 'r').read()
        cur_urdf_content_vis = cur_urdf_content.replace("decomposed.obj", f"{fn}.obj")
        with open(cur_urdf_vis_fn, 'w') as f:
            f.write(cur_urdf_content_vis) 
        # write the file contact to the file #
        f.close()
        print(f"vis_urdf wrote to {cur_urdf_vis_fn}")
        if test:
            break
        
    pass


def modify_coacd_mesh_fn_grab(test=True):
    mesh_folder = "/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    vis_mesh_root = "/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes"
    tot_inst_mesh_folders = os.listdir(mesh_folder)
    tot_inst_mesh_folders = [
        fn for fn in tot_inst_mesh_folders if 'ori_grab_' in fn
    ]
    for fn in tot_inst_mesh_folders:
        cur_vis_mesh_fn = f"{fn}.obj"
        cur_vis_mesh_fn = os.path.join(vis_mesh_root, cur_vis_mesh_fn)
        cur_urdf_folder = os.path.join(mesh_folder, fn, "coacd")
        cur_urdf_vis_mesh_fn = os.path.join(cur_urdf_folder, f"{fn}.obj")
        cp_cmd = f"cp {cur_vis_mesh_fn} {cur_urdf_vis_mesh_fn}"
        print(cp_cmd)
        os.system(cp_cmd)
        cur_urdf_fn = os.path.join(cur_urdf_folder, f"coacd_1.urdf")
        cur_urdf_vis_fn = os.path.join(cur_urdf_folder, f"coacd_1_vis.urdf")
        cur_urdf_content = open(cur_urdf_fn, 'r').read()
        cur_urdf_content_vis = cur_urdf_content.replace("decomposed.obj", f"{fn}.obj")
        with open(cur_urdf_vis_fn, 'w') as f:
            f.write(cur_urdf_content_vis) 
        # write the file contact to the file #
        f.close()
        print(f"vis_urdf wrote to {cur_urdf_vis_fn}")
        if test:
            break
        
    pass





 
def calculate_metrics(data_inst_tag, best_eval_info_fn, v1_eval=False):
    best_eval_info = np.load(best_eval_info_fn , allow_pickle=True).item()
    print(best_eval_info.keys()) 
    
    # dict_keys(['optimized_obj_pose', 'optimized_hand_qtars', 'optimized_hand_qs', 'obj_pose_diff', 'best_obj_pose_diff', 'best_obj_pos_diff', 'best_obj_rot_diff'])
    optimized_hand_qs = best_eval_info['optimized_hand_qs'][0]
    best_obj_pos_diff = best_eval_info['best_obj_pos_diff']
    best_obj_rot_diff = best_eval_info['best_obj_rot_diff']

    #  optimized_res = np.load(dataz  _optimized_res_nn, allow_pickle=True).item()
    
    # if '_nf_300' in data_inst_tag:
    #     kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK"
    
    # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
    grab_inst_tag = "ori_grab_s2_phone_call_1"
    additional_tag = ""
    # grab_inst_tag = ""
    # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    kinematics_data_root = "/cephfs/xueyi/data/TACO_Tracking_PK/data/"
    additional_tag = "interpfr_60_interpfr2_60_nntrans_40"
    # cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
    if len(additional_tag) == 0:    
        kine_data_inst_tag = data_inst_tag
        if v1_eval:
            kine_data_inst_tag = "taco_20230928_031"
        # cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{data_inst_tag}_v2.npy"
        cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{kine_data_inst_tag}_v2.npy"
    else:
        kine_data_inst_tag = data_inst_tag
        if v1_eval:
            kine_data_inst_tag = "taco_20230928_031"
        # cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{data_inst_tag}_v2_{additional_tag}.npy"
        cur_inst_kine_data_fn = f"passive_active_info_{grab_inst_tag}_interped_{kine_data_inst_tag}_v2_{additional_tag}.npy"
     
    # if data_inst_tag.endswith(".npy"): # data inst tag #
    #     cur_inst_kine_data_fn = data_inst_tag
    # else:
    # cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
    cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
    
    # passive #
    save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        
        
    goal_hand_qs = save_info['robot_delta_states_weights_np'][ : ]
    # hand_qs = hand_qs[: , : ]
    
    goal_obj_trans = save_info['object_transl']
    goal_obj_rot_quat = save_info['object_rot_quat']

    # goal_obj_pose = np.concatenate(
    #     [goal_obj_trans, goal_obj_rot_quat],  axis=-1
    # )
    
    clipped_ws = min(optimized_hand_qs.shape[0], goal_hand_qs.shape[0])
    optimized_hand_qs = optimized_hand_qs[:clipped_ws, :]
    goal_hand_qs = goal_hand_qs[:clipped_ws, :]
    
    opt_trans, opt_glbrot, opt_jointangles = optimized_hand_qs[..., :3], optimized_hand_qs[..., 3:6], optimized_hand_qs[..., 6:]
    goal_trans, goal_glbrot, goal_jointangles = goal_hand_qs[..., :3], goal_hand_qs[..., 3:6], goal_hand_qs[..., 6:]
    diff_opt_trans = np.mean(np.linalg.norm(
        (opt_trans - goal_trans), axis=-1, ord=2
    ))
    # between the radius differences #
    diff_opt_glb_rot = np.mean(
        np.mean(
            np.abs( (opt_glbrot - goal_glbrot) ), axis=-1
        )
    )
    diff_opt_jointangles = np.mean(
        np.mean(
            np.abs(opt_jointangles - goal_jointangles), axis=-1
        )
    )
    # weighted_hand_diff = 0.33 * diff_opt_trans + 0.33 * diff_opt_glb_rot + 0.33 * diff_opt_jointangles
    # print()
    
    # diff_opt_jointangles
    # 
    weighted_hand_diff_glb = 0.5 *  diff_opt_trans + 0.5  * diff_opt_glb_rot

    print(best_obj_pos_diff, best_obj_rot_diff,  weighted_hand_diff_glb, diff_opt_jointangles, )

    eval_metrics = {
        'hand_glb_diff': weighted_hand_diff_glb.item(),
        'hand_joint_diff': diff_opt_jointangles.item(),
        'obj_pos_diff': best_obj_pos_diff[0],
        'obj_rot_diff': best_obj_rot_diff[0]
    }
    return eval_metrics





def inspect_data_inst_tag_to_optimized_res(inst_tag_to_opt_res_fn, v1_eval=False):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    tot_eval_metrics_dict = {}
    for key in inst_tag_to_opt_res:
        val = inst_tag_to_opt_res[key]
        print(f"key: {key}, val: {val}")
        inst_tag = key[0]
        if 'taco_20231024_' not in inst_tag:
            continue
        eval_res_fn = val[0]
        best_eval_res_fn = eval_res_fn.replace(".npy", "_sorted.npy").replace(".npy", "_best.npy")
        if not os.path.exists(best_eval_res_fn):
            best_eval_res_fn = eval_res_fn.replace(".npy", "_sorted.npy").replace(".npy", "_best_vv.npy")
            # continue
        if not os.path.exists(best_eval_res_fn):
            continue
        cur_eval_metrics_dict = calculate_metrics(inst_tag, best_eval_res_fn, v1_eval=v1_eval)
        for eval_key in cur_eval_metrics_dict:
            if eval_key not in tot_eval_metrics_dict:
                tot_eval_metrics_dict[eval_key] = [ cur_eval_metrics_dict[eval_key] ]
            else:
                tot_eval_metrics_dict[eval_key].append(cur_eval_metrics_dict[eval_key])
    for eval_key in tot_eval_metrics_dict:
        tot_eval_vals=  tot_eval_metrics_dict[eval_key]
        sorted_tot_eval_vals = sorted(tot_eval_vals)
        medium_val = sorted_tot_eval_vals[len(sorted_tot_eval_vals) // 2]
        tot_eval_vals = np.array(tot_eval_vals)
        print(f"eval_key: {eval_key}, mean: {np.mean(tot_eval_vals)}, medium_val: {medium_val}")


def inspect_data_inst_tag_to_optimized_res_sorted(inst_tag_to_opt_res_fn, v1_eval=False):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    tot_eval_metrics_dict = {}
    
    tot_eval_metrics_dict_tuples = [  ]
    
    # eval_metrics = {
    #     'hand_glb_diff': weighted_hand_diff_glb.item(),
    #     'hand_joint_diff': diff_opt_jointangles.item(),
    #     'obj_pos_diff': best_obj_pos_diff[0],
    #     'obj_rot_diff': best_obj_rot_diff[0]
    # }
    
    for key in inst_tag_to_opt_res:
        val = inst_tag_to_opt_res[key]
        print(f"key: {key}, val: {val}")
        inst_tag = key[0]
        if 'taco_20231024_' not in inst_tag: # inst tag ##
            continue
        eval_res_fn = val[0]
        best_eval_res_fn = eval_res_fn.replace(".npy", "_sorted.npy").replace(".npy", "_best.npy")
        if not os.path.exists(best_eval_res_fn):
            best_eval_res_fn = eval_res_fn.replace(".npy", "_sorted.npy").replace(".npy", "_best_vv.npy")
            # continue
        if not os.path.exists(best_eval_res_fn):
            continue
        cur_eval_metrics_dict = calculate_metrics(inst_tag, best_eval_res_fn, v1_eval=v1_eval)
        
        # cur_eval_metrics_dict #
        
        # sort all the sequences using the best eval res #
        
        hand_glb_diff = cur_eval_metrics_dict['hand_glb_diff']
        hand_joint_diff = cur_eval_metrics_dict['hand_joint_diff']
        obj_pos_diff = cur_eval_metrics_dict['obj_pos_diff']
        obj_rot_diff = cur_eval_metrics_dict['obj_rot_diff']
        
        weighted_obj_pos_diff = obj_pos_diff  * 1.0 + obj_rot_diff * 0.33
        
        eval_inst_tag_w_metrics_tuple = ( inst_tag,   weighted_obj_pos_diff, obj_pos_diff, obj_rot_diff, hand_glb_diff, hand_joint_diff) # object positions, rotations error, and the hand glboal diff and the hand joint diff #
        tot_eval_metrics_dict_tuples.append(eval_inst_tag_w_metrics_tuple) # eval metrics and the tuple
        
        # for eval_key in cur_eval_metrics_dict:
        #     if eval_key not in tot_eval_metrics_dict:
        #         tot_eval_metrics_dict[eval_key] = [ cur_eval_metrics_dict[eval_key] ]
        #     else:
        #         tot_eval_metrics_dict[eval_key].append(cur_eval_metrics_dict[eval_key])
    # for eval_key in tot_eval_metrics_dict:
    #     tot_eval_vals=  tot_eval_metrics_dict[eval_key]
    #     sorted_tot_eval_vals = sorted(tot_eval_vals)
    #     medium_val = sorted_tot_eval_vals[len(sorted_tot_eval_vals) // 2]
    #     tot_eval_vals = np.array(tot_eval_vals)
    #     print(f"eval_key: {eval_key}, mean: {np.mean(tot_eval_vals)}, medium_val: {medium_val}")
    sorted_tot_eval_metrics_dict_tuples = sorted(tot_eval_metrics_dict_tuples, key=lambda x: x[1])
    print(f"tot_eval_metrics_dict_tuples: {len(sorted_tot_eval_metrics_dict_tuples)}")
    # print(sorted)
    # sorted eval metrics dict tuples #
    # sorted eval metrics 
    # 
    # eval metrics dict tuples # 
    s1_sorted_tuples = sorted_tot_eval_metrics_dict_tuples[30:]
    s2_sorted_tuples = sorted_tot_eval_metrics_dict_tuples[40:] 
    s3_sorted_tuples = sorted_tot_eval_metrics_dict_tuples[80:]
    
    def get_medium_values(sorted_tuples):
        first_val_list = [cur_val[2] for cur_val in sorted_tuples]
        second_val_list = [ cur_val[3] for cur_val in sorted_tuples ]
        third_val_list = [cur_val[4] for cur_val in sorted_tuples]
        forth_val_list = [cur_val[5] for cur_val in sorted_tuples]
        first_val_list = sorted(first_val_list)
        second_val_list = sorted(second_val_list)
        third_val_list = sorted(third_val_list)
        forth_val_list = sorted(forth_val_list ) # forthval list 
        medium_first_val = first_val_list[len(first_val_list) // 2]
        second_val_list = second_val_list[len(second_val_list) // 2]
        third_val_list = third_val_list[len(third_val_list) // 2]
        forth_val_list = forth_val_list[len(forth_val_list) // 2]
        inst_tags = [cur_val[0] for cur_val in sorted_tuples]
        print(medium_first_val, second_val_list, third_val_list, forth_val_list)
        return inst_tags
    
    print("s1")
    s1_inst_tags = get_medium_values(s1_sorted_tuples)
    print("s2")
    s2_inst_tags = get_medium_values(s2_sorted_tuples)
    print("s3")
    s3_inst_tags = get_medium_values(s3_sorted_tuples)
    
    return s1_inst_tags, s2_inst_tags, s3_inst_tags # s1, s2, and s3 #
    # and we need to ave hte eval mtrics #



def inspect_kine_info_fn(kine_info_fn):
    kine_info = np.load(kine_info_fn, allow_pickle=True).item()
    print(kine_info.keys())
    for key in kine_info:
        val = kine_info[key]
        print(f"key: {key}, val: {val.shape}")




import pickle as pkl
def inpsect_taco_pkl_data(taco_pkl_data_fn):
    sv_dict = pkl.load(open(taco_pkl_data_fn, "rb"))
    print(f"sv_dict: {sv_dict.keys()}")
    pass



# python utils/taco_data_utils.py 


if __name__=='__main__':
    
    hand_type = 'leap'
    interp_grab_traj_w_taco_traj_v1(hand_type=hand_type)
    exit(0)
    
    
    # grab_inst_tag = ''
    
    grab_inst_tag = 'ori_grab_s2_phone_call_1'
    additional_tag = 'v2'
    all_taco_trajs = True
    # all_taco_trajs = False
    
    ###### original interpolation setting ######
    interp_fr=120
    interp_fr_2=60
    nn_transition=40
    additional_tag = 'v2'
    ###### original interpolation setting ######
    
    # interp_fr=60
    # interp_fr_2=60
    # nn_transition=40
    # additional_tag = f'v2_interpfr_{interp_fr}_interpfr2_{interp_fr_2}_nntrans_{nn_transition}'
    
    hand_type = 'leap'
    ### get the interp fr, interp fr 2, and the nn transitions ###
    interp_grab_traj_w_taco_traj(grab_inst_tag=grab_inst_tag, additional_tag=additional_tag, all_taco_trajs=all_taco_trajs, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition, hand_type=hand_type)
    exit(0)
    
    # get_sorted_taco_optimized_fns()
    # exit(0)
    
    # # modify_coacd_mesh_fn_grab(test=False)
    # # exit(0)
    
    
    # # interested_keys = ['taco_20231024_176', 'taco_20231024_045', 'taco_20231024_169', 'taco_20231024_124', 'taco_20231024_070']
    # # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    # # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_400_/statistics/data_inst_tag_to_optimized_res.npy'
    
    # # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    
    # interested_keys = None
    # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_400_all_/statistics/data_inst_tag_to_optimized_res.npy'
    # interested_keys= ['taco_20231104_035']
    # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_/statistics/data_inst_tag_to_optimized_res.npy'
    # interested_keys= None
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab'
    # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_v2'
    # data_inst_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta_v2.npy'
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_taco_v2'
    # copy_optimized_infos(interested_keys, data_inst_to_optimized_res_fn, dst_folder)
    # exit(0)
    
    # # taco_pkl_data_fn = "/cephfs/xueyi/data/taco/processed_data/20230919/right_20230919_002.pkl"
    # # inpsect_taco_pkl_data(taco_pkl_data_fn)
    # # exit(0)
    
    
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v1_' 
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_400_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.9sup_samples_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_400_all_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.5sup_samples_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.3sup_samples_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.1sup_samples_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_taco_generalist_direct_samples_pertrj_v3_'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_taco_generalist_direct_samples_pertrj_v4_testall_'
    # inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    # get_data_inst_tag_to_optimized_res(eval_data_folder, inst_tag_to_opt_res_fn=inst_tag_to_opt_res_fn)
    # exit(0)
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.5sup_samples_/tracking_taco_20231024_301_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-12-50-15/ts_to_hand_obj_obs_reset_1_sorted_best_vv.npy
    
    # cur_inst_tag_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_400_all_/tracking_taco_20231104_035_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-32-07/ts_to_hand_obj_obs_reset_1.npy"
    # inst_tag = ("taco_20231104_035", "ori_grab_s2_phone_call_1")
    # additional_tag = "interpfr_60_interpfr2_60_nntrans_40"
    # inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag, additional_tag=additional_tag)
    # exit(0) 
    
    
    # # data_inst_tag_to_optimized_res_fn =  '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v1_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v1_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_400_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.9sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.5sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.3sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.1sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_taco_generalist_direct_samples_pertrj_v3_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_taco_generalist_direct_samples_pertrj_v4_testall_/statistics/data_inst_tag_to_optimized_res.npy'
    
    additional_tag = "interpfr_60_interpfr2_60_nntrans_40"
    resave_optimized_res(data_inst_tag_to_optimized_res_fn, additional_tag=additional_tag)
    exit(0) 
    
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v1_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.9sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.5sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.3sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn ='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.1sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    
    # inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag=None, random_select=False)
    # eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])

    # data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    # data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    # np.save(data_inst_tag_to_best_opt_res_fn, inst_tag_to_best_opt_res) # #
    # print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(inst_tag_to_best_opt_res)}")
    # exit(0)
    
    # grab_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
    # # get_tot_grab_files_new(grab_data_root)
    # # # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data
    # # exit(0)
    
    # taco_data_fn = "/cephfs/xueyi/data/TACO_Tracking_PK/data"
    # get_totest_taco_files_new(taco_data_fn)
    # exit(0)
    
    # inspect_total_taco_data() # GRAb
    # exit(0)
    
    
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_customdamping_v2'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wstatebased_wcustomidamping'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wstatebased_wcustomidamping_realleap'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wstatebased_wcustomidamping_modelv2_realleap'
    # # taco_sv_folder = 
    # obj_type_to_optimized_res = calculate_taco_obj_type_to_optimized_res(taco_sv_folder)
    # taco_sv_statistics_folder = os.path.join(taco_sv_folder, "statistics")
    # os.makedirs(taco_sv_statistics_folder, exist_ok=True)
    # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res.npy"
    # # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res_beta.npy"
    # # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res_beta_v1.npy"
    # taco_sv_obj_type_to_optimized_res_fn = os.path.join(taco_sv_statistics_folder, taco_sv_obj_type_to_optimized_res_fn)

    # # save the optimized res #
    # np.save(taco_sv_obj_type_to_optimized_res_fn, obj_type_to_optimized_res)
    # print(f"Saved the optimized results for taco obj types to {taco_sv_obj_type_to_optimized_res_fn} with length {len(obj_type_to_optimized_res)}")
    # exit(0)
    
    
    
    # # kine_info_fn = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res/tracking_TACO_taco_20231020_036_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-21-50-08/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231020_036_v2_interpfr_60_interpfr2_60_nntrans_40.npy"
    # inspect_kine_info_fn(kine_info_fn)
    # exit(0)
    
    
    data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_rewfilter50_v1/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res.npy" # v1 iteration of TACO-GRAB training
    data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final/statistics/data_inst_tag_to_optimized_res.npy" # v2 iteration of TACO-GRAB training 
    # isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_len400_v1
    # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_len400_v1/statistics/data_inst_tag_to_optimized_res.npy" 
    # # 
    
    # 
    # 0.6751 6.37 0.1264 0.5443 21.94/50.32
    # 0.4782 4.94 0.1329 0.4228 32.44/62.36
    
    # eval_key: hand_glb_diff, mean: 0.11528140941682534, medium_val: 0.11959081515669823
    # eval_key: hand_joint_diff, mean: 0.4756938425394205, medium_val: 0.4681936800479889
    # eval_key: obj_pos_diff, mean: 0.14195056278736162, medium_val: 0.09105604588985443
    # eval_key: obj_rot_diff, mean: 0.501226585644942, medium_val: 0.48151418566703796
    # [0.05128205128205128, 0.05128205128205128, 0.6153846153846154]
        
    # eval_key: hand_glb_diff, mean: 0.2489110502677086
    # eval_key: hand_joint_diff, mean: 0.5818599126277826
    # eval_key: obj_pos_diff, mean: 0.14195056278736162
    # eval_key: obj_rot_diff, mean: 0.501226585644942
    # succ_nn/tot_nn: 2/39
    # tot_succ_nn/tot_tot_nn: 2/39, res: 0.05128205128205128
    # succ_nn/tot_nn: 2/39
    # tot_succ_nn/tot_tot_nn: 2/39, res: 0.05128205128205128
    # succ_nn/tot_nn: 24/39
    # tot_succ_nn/tot_tot_nn: 24/39, res: 0.6153846153846154
    # [0.05128205128205128, 0.05128205128205128, 0.6153846153846154] 
    
    
    # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final/statistics/data_inst_tag_to_optimized_res.npy" # 
    # eval_key: hand_glb_diff, mean: 0.2489110502677086, medium_val: 0.24613402038812637
    # eval_key: hand_joint_diff, mean: 0.5818599126277826, medium_val: 0.5770670175552368
    # eval_key: obj_pos_diff, mean: 0.14195056278736162, medium_val: 0.14105604588985443
    # eval_key: obj_rot_diff, mean: 0.501226585644942, medium_val: 0.48151418566703796
    # [0.05128205128205128, 0.05128205128205128, 0.6153846153846154]
    # data_inst_to_opt_res_fn_a  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelseveralhunderedstraj_testset_/statistics/data_inst_tag_to_optimized_res.npy"
    # eval_key: hand_glb_diff, mean: 0.15342251036705729, medium_val: 0.1298226173967123
    # eval_key: hand_joint_diff, mean: 0.4985670834974526, medium_val: 0.47717997431755066
    # eval_key: obj_pos_diff, mean: 0.04325949038482375, medium_val: 0.01994769647717476
    # eval_key: obj_rot_diff, mean: 0.5378666298062194, medium_val: 0.48536843061447144
    # [0.5490196078431373, 0.7254901960784313, 0.7973856209150327]
    data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelalltraj_testset_2_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v3_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # eval_key: hand_glb_diff, mean: 0.15430610363365582, medium_val: 0.15103782713413239
    # eval_key: hand_joint_diff, mean: 0.48556964635069855, medium_val: 0.4661847949028015 # with sup and with curriculum # 
    # eval_key: obj_pos_diff, mean: 0.04326441852388039, medium_val: 0.020951339974999428
    # eval_key: obj_rot_diff, mean: 0.5441007912158966, medium_val: 0.4953787326812744
    # [0.49019607843137253, 0.7450980392156863, 0.8104575163398693]
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v1_/statistics/data_inst_tag_to_optimized_res.npy'
    # eval_key: hand_glb_diff, mean: 0.18289726320045446, medium_val: 0.17816811986267567 # 
    # eval_key: hand_joint_diff, mean: 0.544395401976467, medium_val: 0.5437731742858887 # 
    # eval_key: obj_pos_diff, mean: 0.044306796488467776, medium_val: 0.023347297683358192 # 
    # eval_key: obj_rot_diff, mean: 0.4984648953271068, medium_val: 0.4444066286087036 # # 7.82
    # [0.4797385620915033, 0.6712418300653595, 0.7366013071895425]
    v1_eval = True
    v1_eval = False
    
    
    
    # s1_inst_tags, s2_inst_tags, s3_inst_tags = inspect_data_inst_tag_to_optimized_res_sorted(data_inst_to_opt_res_fn_a, v1_eval=v1_eval) # # 
    
    # data_inst_to_opt_res_fn_a = data_inst_to_opt_res_fn_a.replace("data_inst_tag_to_optimized_res.npy", "data_inst_tag_to_best_opt_res_all.npy")
    
    # print(f"s1-succ-rate")
    # calcualte_merged_succ_info_all_thres(data_inst_to_opt_res_fn_a, target_inst_tags=s1_inst_tags)
    # print(f"s2-succ-rate")
    # calcualte_merged_succ_info_all_thres(data_inst_to_opt_res_fn_a, target_inst_tags=s2_inst_tags)
    # print(f"s3-succ-rate")
    # calcualte_merged_succ_info_all_thres(data_inst_to_opt_res_fn_a, target_inst_tags=s3_inst_tags)
    
    # s1
    # 0.0233775582164526 0.5787567496299744 0.1481525357812643 0.47031089663505554
    # s2
    # 0.02467368356883526 0.6026294827461243 0.14556516334414482 0.47095319628715515
    # s3
    # 0.08066685497760773 0.6508540511131287 0.15137944743037224 0.4683372974395752
    # s1-succ-rate
    # succ_nn/tot_nn: 45/123
    # tot_succ_nn/tot_tot_nn: 45/123, res: 0.36585365853658536
    # succ_nn/tot_nn: 84/123
    # tot_succ_nn/tot_tot_nn: 84/123, res: 0.6829268292682927
    # succ_nn/tot_nn: 94/123
    # tot_succ_nn/tot_tot_nn: 94/123, res: 0.7642276422764228
    # [0.36585365853658536, 0.6829268292682927, 0.7642276422764228]
    # s2-succ-rate
    # succ_nn/tot_nn: 35/113
    # tot_succ_nn/tot_tot_nn: 35/113, res: 0.30973451327433627
    # succ_nn/tot_nn: 74/113
    # tot_succ_nn/tot_tot_nn: 74/113, res: 0.6548672566371682
    # succ_nn/tot_nn: 84/113
    # tot_succ_nn/tot_tot_nn: 84/113, res: 0.7433628318584071
    # [0.30973451327433627, 0.6548672566371682, 0.7433628318584071]
    # s3-succ-rate
    # succ_nn/tot_nn: 1/73
    # tot_succ_nn/tot_tot_nn: 1/73, res: 0.0136986301369863
    # succ_nn/tot_nn: 34/73
    # tot_succ_nn/tot_tot_nn: 34/73, res: 0.4657534246575342
    # succ_nn/tot_nn: 44/73
    # tot_succ_nn/tot_tot_nn: 44/73, res: 0.6027397260273972
    # [0.0136986301369863, 0.4657534246575342, 0.6027397260273972]
    
    
    # eval_key: hand_glb_diff, mean: 0.10293323398732088, medium_val: 0.09114746376872063
    # eval_key: hand_joint_diff, mean: 0.4737170913250618, medium_val: 0.46545761823654175
    # eval_key: obj_pos_diff, mean: 0.03829042381588735, medium_val: 0.023972854018211365
    # eval_key: obj_rot_diff, mean: 0.46983588550215455, medium_val: 0.4140978157520294
    # succ_nn/tot_nn: 104/153
    # tot_succ_nn/tot_tot_nn: 104/153, res: 0.6797385620915033
    # succ_nn/tot_nn: 123/153
    # tot_succ_nn/tot_tot_nn: 123/153, res: 0.803921568627451
    # succ_nn/tot_nn: 129/153
    # tot_succ_nn/tot_tot_nn: 129/153, res: 0.8431372549019608
    # [0.6797385620915033, 0.803921568627451, 0.8431372549019608]
    # 72.97 # 0.9
    
    # eval_key: hand_glb_diff, mean: 0.09838628492916361, medium_val: 0.10038124211132526
    # eval_key: hand_joint_diff, mean: 0.4543358634244527, medium_val: 0.44866544008255005
    # eval_key: obj_pos_diff, mean: 0.038733966298947195, medium_val: 0.023448247462511063
    # eval_key: obj_rot_diff, mean: 0.44811938393133266, medium_val: 0.40851378440856934
    # succ_nn/tot_nn: 94/151
    # tot_succ_nn/tot_tot_nn: 94/151, res: 0.6225165562913907
    # succ_nn/tot_nn: 121/151
    # tot_succ_nn/tot_tot_nn: 121/151, res: 0.8013245033112583
    # succ_nn/tot_nn: 129/151
    # tot_succ_nn/tot_tot_nn: 129/151, res: 0.8543046357615894
    # [0.6225165562913907, 0.8013245033112583, 0.8543046357615894]
    # 67.25 #
    
    # eval_key: hand_glb_diff, mean: 0.12794745504076965, medium_val: 0.1264638714492321
    # eval_key: hand_joint_diff, mean: 0.494531315526152, medium_val: 0.4825315773487091
    
    # eval_key: obj_pos_diff, mean: 0.044387086720686725, medium_val: 0.024772871285676956
    # eval_key: obj_rot_diff, mean: 0.5174598407511618, medium_val: 0.4680233895778656 # 
    # succ_nn/tot_nn: 88/153
    # tot_succ_nn/tot_tot_nn: 88/153, res: 0.5751633986928104
    # succ_nn/tot_nn: 116/153
    # tot_succ_nn/tot_tot_nn: 116/153, res: 0.7581699346405228
    # succ_nn/tot_nn: 128/153
    # tot_succ_nn/tot_tot_nn: 128/153, res: 0.8366013071895425
    # [0.5751633986928104, 0.7581699346405228, 0.8366013071895425]
    # 62.52 # 0.3
    # 57.69 # 0.0
    
    # 
    # 33.98/46.41
    # 0.0543870867
    # 0.5174598407511618
    # mainly the object reward without the one in the reward #
    # DGrasp --- seems that we should not use the hand metrics as well #
    # DGrasp #
    #
    
    # 0.385620915033, 0.451
    # 0.05042381588735
    # 0.50215455
    
    
    # 200+ instances that have the supervisions # # no supervisions #
    # 0.74 - 0.73 (0.9) - 0.71 (0.7) - 0.68 (0.5) - 0.63 (0.3) - 0.57 (0.0)
    
    data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.9sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.5sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.3sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.1sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    
    
    # # # # # #
    inspect_data_inst_tag_to_optimized_res(data_inst_to_opt_res_fn_a, v1_eval=v1_eval)
    data_inst_to_opt_res_fn_a = data_inst_to_opt_res_fn_a.replace("data_inst_tag_to_optimized_res.npy", "data_inst_tag_to_best_opt_res_all.npy")
    # calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_a)
    calcualte_merged_succ_info_all_thres(data_inst_to_opt_res_fn_a)
    
    
    exit(0)
    
    
    
    # modify_coacd_mesh_fn(test=False)
    # exit(0)
    
    copy_optimized_infos()
    exit(0)
    
    # grab_inst_tag = ''
    
    # grab_inst_tag = 'ori_grab_s2_phone_call_1'
    # additional_tag = 'v2'
    # all_taco_trajs = True
    # # all_taco_trajs = False
    
    # ###### original interpolation setting ######
    # interp_fr=120
    # interp_fr_2=60
    # nn_transition=40
    # additional_tag = 'v2'
    # ###### original interpolation setting ######
    
    # interp_fr=60
    # interp_fr_2=60
    # nn_transition=40
    # additional_tag = f'v2_interpfr_{interp_fr}_interpfr2_{interp_fr_2}_nntrans_{nn_transition}'
    
    # hand_type = 'leap'
    # ### get the interp fr, interp fr 2, and the nn transitions ###
    # interp_grab_traj_w_taco_traj(grab_inst_tag=grab_inst_tag, additional_tag=additional_tag, all_taco_trajs=all_taco_trajs, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition, hand_type=hand_type)
    # exit(0)
    
    # inspect_taco_optimized_res()
    # exit(0)
    
    
    # # sorted_taco_optimized_fns = ""
    # get_sorted_taco_optimized_fns()
    # exit(0)
    
    # inst_tag_to_opt_res_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta.npy"
    # inst_tag_to_opt_res_b = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_remaining_temp_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_two_inst_tag_to_opt_res(inst_tag_to_opt_res_a, inst_tag_to_opt_res_b)
    # exit(0)
    
    
    # eval_folder= '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_remaining_temp_eval'
    # inspect_eval_fns(eval_folder)
    # exit(0)
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res.npy"
    # obj_type_to_ooptimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_samples_reopt_v5/statistics/obj_type_to_optimized_res.npy'
    # rew_filter_thres = 95.0
    # good_inst_tag_to_optimized_res = get_good_inst_tag_base_trajectories(inst_tag_to_optimized_res_fn, obj_type_to_ooptimized_res_fn, rew_filter_thres=rew_filter_thres)
    # good_inst_tag_to_optimized_res_sv_fn = inst_tag_to_optimized_res_fn.replace(".npy", f"_rew_{rew_filter_thres}.npy")
    # np.save(good_inst_tag_to_optimized_res_sv_fn, good_inst_tag_to_optimized_res)
    # print(f"Saved good_inst_tag_to_optimized_res to {good_inst_tag_to_optimized_res_sv_fn}")
    # exit(0)
    
    # obj_type_to_opt_res_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_samples_reopt_v5/statistics/obj_type_to_optimized_res.npy'
    # obj_type_to_opt_res_b = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy'
    # compare_obj_type_optimized_res(obj_type_to_opt_res_a, obj_type_to_opt_res_b)
    # exit(0)
    
    
    # # # 
    # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_rewfilter50_v1/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_best_opt_res_all.npy" # v1 iteration of TACO-GRAB training
    # # data_inst_to_opt_res_fn_a = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final/statistics/data_inst_tag_to_best_opt_res_all.npy" # v2 iteration of TACO-GRAB training 
    # data_inst_to_opt_res_fn_a  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelseveralhunderedstraj_testset_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # data_inst_to_opt_res_fn_a  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelalltraj_testset_2_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_a)
    # exit(0)
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta.npy" # isnt tag to the opt res #
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_rewfilter50_v1/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelseveralhunderedstraj_testset_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelalltraj_testset_2_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_v1_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag=None, random_select=False)
    # eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    # # if len(target_subj_tag) > 0:
    # #     data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    # # else:
    # #     if random_select:
    # #         data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
    # #     else:
    # data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    # data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    # np.save(data_inst_tag_to_best_opt_res_fn, inst_tag_to_best_opt_res) # #
    # print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(inst_tag_to_best_opt_res)}")
    # exit(0)

    
    
    # inst_tag_to_opt_res_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_opt_res_dict_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_samples_reopt/statistics/obj_type_to_optimized_res.npy'
    # modify_inst_tag_to_opt_res(inst_tag_to_opt_res_dict_fn)
    # exit(0)
    
    
    
    
    # 742 samples in total -- does such trjectories optimized res jwithout the i 
    
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn)
    # exit(0)
    
    # obj_type_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res_beta.npy"
    # inspect_obj_type_to_optimized_res_beta(obj_type_to_optimized_res_fn)
    # exit(0)
    
    # taco_retar_info_root = "/cephfs/xueyi/data/TACO_Tracking_PK/data"
    # calculate_tot_taco_info(taco_retar_info_root)
    # exit(0)
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn)
    # exit(0)
    
    
    ### evaluate for ###
    # local_folder = "./runs"
    # target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2"
    # subj_idx = None
    # exclude_existing_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    
    # mv_eval_data_to_target_folder(local_folder, target_folder, subj_idx, exclude_existing_fn)
    # exit(0)
    
    
    # local_folder = "./runs"
    # target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_reopt"
    # # not with diffusion samples re ope # jparse the time and then move the reope results #
    # subj_idx = 8
    
    # mv_eval_data_to_target_folder(local_folder, target_folder, subj_idx)
    # exit(0)
    
    
    # local_folder = "./runs"
    # target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2"
    
    
    # bound loss coef # # # bound loss coef ##
    # eval_sv_info = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-23-46-56/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
    # inspect_eval_data_info(eval_sv_info)
    # # exit(0)
    

    
    # # # # # data inst tag to opt #
    # # # # # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn =  "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2/statistics/data_inst_tag_to_optimized_res_v2.npy"
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v4/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v6/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_rewfilter50_v1/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_generalist_direct_samples_customizeddamping_pertrj_v2_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelalltraj_testset_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelalltraj_testset_2_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_remaining_temp_eval/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelseveralhunderedstraj_testset_/statistics/data_inst_tag_to_optimized_res.npy'
    additional_tag = "interpfr_60_interpfr2_60_nntrans_40"
    resave_optimized_res(data_inst_tag_to_optimized_res_fn, additional_tag=additional_tag)
    exit(0) 
    
    # # # eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval"
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v6/tracking_taco_20231104_003_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-22-14-16/ts_to_hand_obj_obs_reset_1.npy
    eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2"
    eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2"
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v4'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v6'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_remaining_temp_eval'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7'
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final/tracking_taco_20231104_118_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-15-56-54/ts_to_hand_obj_obs_reset_1.npy
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_rewfilter50_v1/tracking_taco_20231027_023_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-03-20-12/ts_to_hand_obj_obs_reset_1.npy
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v7_final'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_rewfilter50_v1'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_generalist_direct_samples_customizeddamping_pertrj_v2_'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelalltraj_testset_'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_generalist_direct_samples_customizeddamping_pertrj_frmodelseveralhunderedstraj_testset_'
    eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_' 
    inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res_beta.npy"
    inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    get_data_inst_tag_to_optimized_res(eval_data_folder, inst_tag_to_opt_res_fn=inst_tag_to_opt_res_fn)
    exit(0)

    # # # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn =  "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2/statistics/data_inst_tag_to_optimized_res_v2.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5/statistics/data_inst_tag_to_optimized_res.npy"
    # additional_tag = "interpfr_60_interpfr2_60_nntrans_40"
    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, additional_tag=additional_tag)
    # exit(0)
    
    # # eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval"
    # eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2"
    # eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2"
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v2'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval'
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5' # 
    # eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v5'
    # inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    # # inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res_beta.npy"
    # # inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    # exclude_last_saved = False
    # # exclude_last_saved = True
    # get_data_inst_tag_to_optimized_res(eval_data_folder, inst_tag_to_opt_res_fn=inst_tag_to_opt_res_fn, exclude_last_saved=exclude_last_saved)
    # exit(0)
 
    # local_folder = "./runs" # ver - 1 or the ver - final? #
    # target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq"
    # mv_taco_interp_optimized_res(local_folder, target_folder) # mv the optimized res to the target folder #
    # exit(0)
    
    # inspect_total_taco_data() # GRAb
    # exit(0)
    
    
    # # # obj_type_to_optimized_res_sv_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res.npy"
    # obj_type_to_optimized_res_sv_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq/statistics/obj_type_to_optimized_res_beta.npy"
    # inspect_taco_obj_type_to_optimized_res(obj_type_to_optimized_res_sv_fn)
    # exit(0)
    
    # # taco_sv_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq"
    # # inspect_taco_optimized_data_folder(taco_sv_folder)
    # # exit(0) 
    

    # taco_sv_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq"
    # obj_type_to_optimized_res = calculate_taco_obj_type_to_optimized_res(taco_sv_folder)
    # taco_sv_statistics_folder = os.path.join(taco_sv_folder, "statistics")
    # os.makedirs(taco_sv_statistics_folder, exist_ok=True)
    # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res.npy"
    # taco_sv_obj_type_to_optimized_res_fn = os.path.join(taco_sv_statistics_folder, taco_sv_obj_type_to_optimized_res_fn)

    # # taco obj type to opt res #
    # taco_sv_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq"
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_remaining_temp'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_samples_reopt'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_samples_reopt_v5'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_customdamping'
    # taco_sv_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_customdamping_v2'
    # obj_type_to_optimized_res = calculate_taco_obj_type_to_optimized_res(taco_sv_folder)
    # taco_sv_statistics_folder = os.path.join(taco_sv_folder, "statistics")
    # os.makedirs(taco_sv_statistics_folder, exist_ok=True)
    # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res.npy"
    # # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res_beta.npy"
    # # taco_sv_obj_type_to_optimized_res_fn = "obj_type_to_optimized_res_beta_v1.npy"
    # taco_sv_obj_type_to_optimized_res_fn = os.path.join(taco_sv_statistics_folder, taco_sv_obj_type_to_optimized_res_fn)

    # # save the optimized res #
    # np.save(taco_sv_obj_type_to_optimized_res_fn, obj_type_to_optimized_res)
    # print(f"Saved the optimized results for taco obj types to {taco_sv_obj_type_to_optimized_res_fn} with length {len(obj_type_to_optimized_res)}")
    # exit(0)
    
    
    
    # local_folder = "./runs"
    # target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq"
    # exp_tag = "INTERPSEQ"
    # mv_taco_grab_interp_folders(local_folder, target_folder, exp_tag)
    # exit(0)
    
    # interp_grab_traj_w_taco_traj_v1()
    # exit(0)
    
    # grab_inst_tag = ''
    
    grab_inst_tag = 'ori_grab_s2_phone_call_1'
    additional_tag = 'v2'
    all_taco_trajs = True
    # all_taco_trajs = False
    
    ###### original interpolation setting ######
    interp_fr=120
    interp_fr_2=60
    nn_transition=40
    additional_tag = 'v2'
    ###### original interpolation setting ######
    
    interp_fr=60
    interp_fr_2=60
    nn_transition=40
    additional_tag = f'v2_interpfr_{interp_fr}_interpfr2_{interp_fr_2}_nntrans_{nn_transition}'
    
    ### get the interp fr, interp fr 2, and the nn transitions ###
    interp_grab_traj_w_taco_traj(grab_inst_tag=grab_inst_tag, additional_tag=additional_tag, all_taco_trajs=all_taco_trajs, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition)
    exit(0)
    
    # test_find_grasping_frame()
    # exit(0)
    
    interpolate_all_taco_ref_files()
    exit(0)
    
    
    taco_inst_tag = "20231104_203"
    taco_inst_tag = "20231104_151"
    
    grab_inst_grasping_frame = 37
    
    taco_data_root = "/cephfs/xueyi/data/taco/processed_data/20231104"
    tot_inst_idxes = get_all_taco_instancees(taco_data_root)
    for i_inst, cur_inst_idx in enumerate(tot_inst_idxes):
        
        taco_inst_tag = cur_inst_idx
        
        retar_info_fn_1 = f'/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_nf_300.npy'
        retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
        
        taso_grasping_fr = find_grasp_frame_from_mocap_data(retar_info_fn_2)
        
        interp_fr = 120
        interp_fr_2 = 60
        nn_transition = 40
        
        interp_two_trajs_2(retar_info_fn_1, retar_info_fn_2, grasping_fr=taso_grasping_fr, grasping_fr_grab=grab_inst_grasping_frame, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition, taco_inst_tag=taco_inst_tag)
        # interp_two_trajs(retar_info_fn_1, retar_info_fn_2, interp_fr, nn_transition, taco_inst_tag)
        
    exit(0)
    
    
    # for each taco sequence, find the grasping frame #
    
    retar_info_fn_1 = f'/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_nf_300.npy'
    retar_info_fn_2 = f'/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_{taco_inst_tag}_zrot_3.141592653589793_modifed_interped.npy'
    interp_fr = 120
    interp_fr_2 = 60
    nn_transition = 40
    interp_two_trajs_2(retar_info_fn_1, retar_info_fn_2, interp_fr=interp_fr, interp_fr_2=interp_fr_2, nn_transition=nn_transition, taco_inst_tag=taco_inst_tag)
    # interp_two_trajs(retar_info_fn_1, retar_info_fn_2, interp_fr, nn_transition, taco_inst_tag)
    exit(0)
    
    taco_inst_idx = 203
    retar_info_fn = f"/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_20231104_{taco_inst_idx}_modifed_interped.npy"
    # inspect_saved_retar_info(retar_info_fn)
    # transform_retar_info(retar_info_fn)
    ## transform_retar_info ##
    retar_info_fn = f"{retar_info_fn[:-4]}_transformed.npy"
    
    retar_info_fn = "/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_taco_20231104_203_zrot_3.141592653589793_modifed_interped.npy"
    get_body_poses(retar_info_fn)
    # #


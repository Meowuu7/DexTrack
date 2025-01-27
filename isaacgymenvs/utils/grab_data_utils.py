import numpy as np
import os
import torch

##### 1) claculate the inst type to optimized res ######
##### 2) calcualte the object type to optimized res ######


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


def parse_obj_type_fr_folder_name(folder_nm):
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_hammer_use_1_obs_pure_state_wref_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t1r1_rewfingerdist_0.5_rewdeltahandpose_0.0_10-17-19-17
    folder_st_tag = "tracking_"
    remains_folder_nm = folder_nm.split("/")[-1][len(folder_st_tag): ]
    
    folder_nm_segs = remains_folder_nm.split("_")
    st_idx = 0
    for ed_idx in range(st_idx, len(folder_nm_segs)):
        cur_seg = folder_nm_segs[ed_idx]
        if cur_seg == 'obs' or cur_seg == 'leap':
            break
    obj_type = folder_nm_segs[st_idx: ed_idx]
    obj_type = "_".join(obj_type)
    return obj_type

def find_best_rew(folder_nm):
    nn_folder = os.path.join(folder_nm, "nn")
    if not os.path.exists(nn_folder):
        return -9999.9, None
    tot_ckpts = os.listdir(nn_folder)
    tot_ckpts = [
        fn for fn in tot_ckpts if fn.endswith(".pth")
    ]
    tot_rews = []
    cur_best_rew = -9999.9
    best_ckpt_fn = None
    for cur_ckpt_fn in tot_ckpts:
        # cur_full_ckpt_fn = os.path.join(nn_folder, cur_ckpt_fn)
        cur_ckpt_pure_fn = cur_ckpt_fn.split(".pth")[0]
        cur_ckpt_pure_fn_segs = cur_ckpt_pure_fn.split("_")
        try:
            if len(cur_ckpt_pure_fn_segs[-1]) == 0:
                cur_rew = float(cur_ckpt_pure_fn_segs[-2])
            else:
                cur_rew = float(cur_ckpt_pure_fn_segs[-1])
        except:
            print(cur_ckpt_pure_fn_segs)
            continue
        tot_rews.append(cur_rew)
        if cur_rew > cur_best_rew:
            cur_best_rew = cur_rew
            best_ckpt_fn = os.path.join(nn_folder, cur_ckpt_fn)
    # maxx_rew = max(tot_rews)
    # 
    return cur_best_rew, best_ckpt_fn
    


def get_obj_type_to_optimized_res(optimized_root_folder):
    tot_folders = os.listdir(optimized_root_folder)
    tracking_st_tag = 'tracking_'
    tot_folders = [
        fn for fn in tot_folders if fn[: len(tracking_st_tag)] == tracking_st_tag
    ]
    obj_type_to_optimized_res = {}
    for i_exp, cur_folder in enumerate(tot_folders):
        print(f"[{i_exp}/{len(tot_folders)}] cur_folder: {cur_folder}")
        cur_full_folder = os.path.join(optimized_root_folder, cur_folder)
        cur_obj_type = parse_obj_type_fr_folder_name(cur_full_folder)
        
        cur_best_rew, best_ckpt_fn = find_best_rew(cur_full_folder)
        if best_ckpt_fn is None:
            continue
        obj_type_to_optimized_res[cur_obj_type] = (cur_best_rew, best_ckpt_fn)
    return obj_type_to_optimized_res

def calculate_optimized_res(optimized_data_sv_root):
    # optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab' # args.optimized_data_sv_root

    ## get hte optimized res ##
    tracking_data_statistics_folder = os.path.join(optimized_data_sv_root, "statistics")
    os.makedirs(tracking_data_statistics_folder, exist_ok=True)
    obj_type_to_optimized_res_sv_fn = "obj_type_to_optimized_res.npy"
    obj_type_to_optimized_res_sv_fn = os.path.join(tracking_data_statistics_folder, obj_type_to_optimized_res_sv_fn) 
    # if not os.path.exists(obj_type_to_optimized_res_sv_fn):
    obj_type_to_optimized_res = get_obj_type_to_optimized_res(optimized_data_sv_root) # get the tracking data sv root #
    print(f"obj_type_to_optimized_res: {obj_type_to_optimized_res}")
    np.save(obj_type_to_optimized_res_sv_fn, obj_type_to_optimized_res) # save the optimized res
    print(f"obj_type_to_optimized_res saved to {obj_type_to_optimized_res_sv_fn}")


def inspect_optimized_res(grab_inst_tag):
    tracking_data_statistics_folder = os.path.join(optimized_data_sv_root, "statistics")
    obj_type_to_optimized_res_sv_fn = "obj_type_to_optimized_res.npy"
    obj_type_to_optimized_res_sv_fn = os.path.join(tracking_data_statistics_folder, obj_type_to_optimized_res_sv_fn) 
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_sv_fn, allow_pickle=True).item()
    cur_inst_optimized_res = obj_type_to_optimized_res[grab_inst_tag]
    print(cur_inst_optimized_res)
    
def mv_eval_data_to_eval_folder(eval_folder, forbid_tag=None):
    # 
    # 
    tracking_folder_st_tag = "tracking_"
    local_runs_folder = './runs'
    tot_runs_subfolder = os.listdir(local_runs_folder)
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if fn[: len(tracking_folder_st_tag)] == tracking_folder_st_tag
    ]
    for cur_run_subfolder in tot_runs_subfolder:
        cur_full_run_subfolder = os.path.join(local_runs_folder, cur_run_subfolder)
        run_subfolder_nn_folder = os.path.join(cur_full_run_subfolder, "nn")
        if forbid_tag is not None and forbid_tag in cur_run_subfolder:
            continue
        if not os.path.exists(run_subfolder_nn_folder): # 
            print(f"mv {cur_full_run_subfolder} {eval_folder}/")
            os.system(f"mv {cur_full_run_subfolder} {eval_folder}/")
    pass

def mv_training_data_to_folder(training_data_target_folder, nf=150):
    tracking_folder_st_tag = "tracking_"
    local_runs_folder = './runs'
    tot_runs_subfolder = os.listdir(local_runs_folder)
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if fn[: len(tracking_folder_st_tag)] == tracking_folder_st_tag
    ]
    for cur_run_subfolder in tot_runs_subfolder:
        cur_full_run_subfolder = os.path.join(local_runs_folder, cur_run_subfolder)
        run_subfolder_nn_folder = os.path.join(cur_full_run_subfolder, "nn")
        if not os.path.exists(run_subfolder_nn_folder):
            continue
        if nf == 150:
            if '_nf_' in cur_run_subfolder:
                continue
        else:
            nf_tag = f'_nf_{nf}_'
            if nf_tag not in cur_run_subfolder:
                continue
        print(f"mv {cur_full_run_subfolder} {training_data_target_folder}/")
        os.system(f"mv {cur_full_run_subfolder} {training_data_target_folder}/")
    pass

def get_data_inst_tag_to_optimized_res(eval_data_folder):
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
    traj_sv_fn = "ts_to_hand_obj_obs_reset_1.npy"
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if os.path.exists(os.path.join(eval_data_folder, fn, traj_sv_fn))
    ]
    data_inst_tag_to_optimized_res = {}
    for i_test, cur_run_subfolder in enumerate(tot_runs_subfolder):
        cur_full_run_subfolder = os.path.join(eval_data_folder, cur_run_subfolder)
        cur_obj_type = parse_obj_type_fr_folder_name(cur_full_run_subfolder)
        
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
        
        cur_grab_exp_tag = (cur_obj_type, cur_obj_type)
        if cur_grab_exp_tag not in data_inst_tag_to_optimized_res:
            data_inst_tag_to_optimized_res[cur_grab_exp_tag] = [cur_run_traj_sv_fn]
        else:
            data_inst_tag_to_optimized_res[cur_grab_exp_tag].append(cur_run_traj_sv_fn) # get the run traj sv fn #
        print(f"len(data_inst_tag_to_optimized_res): {len(data_inst_tag_to_optimized_res)}")
        # 
    
    data_inst_tag_to_optimized_res_sv_statistics_folder = "statistics"
    data_inst_tag_to_optimized_res_sv_statistics_folder = os.path.join(eval_data_folder, data_inst_tag_to_optimized_res_sv_statistics_folder)
    os.makedirs(data_inst_tag_to_optimized_res_sv_statistics_folder, exist_ok=True)
    data_inst_tag_to_optimized_res_sv_statistics_fn = "data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_sv_statistics_fn = os.path.join(data_inst_tag_to_optimized_res_sv_statistics_folder, data_inst_tag_to_optimized_res_sv_statistics_fn)
    np.save(data_inst_tag_to_optimized_res_sv_statistics_fn, data_inst_tag_to_optimized_res)
    print(f"data_inst_tag_to_optimized_res saved to {data_inst_tag_to_optimized_res_sv_statistics_fn}")
    return data_inst_tag_to_optimized_res
        
    #     cur_best_rew, best_ckpt_fn = find_best_rew(cur_full_run_subfolder)
        
    #     if best_ckpt_fn is None:
    #         continue
    #     data_inst_tag_to_optimized_res[cur_obj_type] = (cur_best_rew, best_ckpt_fn)
    # return data_inst_tag_to_optimized_res



# parse_obj_type_traj_type_fr_folder_name

def get_data_inst_tag_to_optimized_res_obj_traj(eval_data_folder):
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
    traj_sv_fn = "ts_to_hand_obj_obs_reset_1.npy"
    tot_runs_subfolder = [
        fn for fn in tot_runs_subfolder if os.path.exists(os.path.join(eval_data_folder, fn, traj_sv_fn))
    ]
    data_inst_tag_to_optimized_res = {}
    for i_test, cur_run_subfolder in enumerate(tot_runs_subfolder):
        cur_full_run_subfolder = os.path.join(eval_data_folder, cur_run_subfolder)
        cur_obj_type, cur_traj_type = parse_obj_type_traj_type_fr_folder_name(cur_full_run_subfolder)
        
        # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-23-05-11/ts_to_hand_obj_obs_reset_1.npy
        cur_run_traj_sv_fn = os.path.join(cur_full_run_subfolder, traj_sv_fn)
        
        print(f"[{i_test}/{len(tot_runs_subfolder)}] cur_run_subfolder: {cur_run_subfolder}")
        
        
        
        cur_grab_exp_tag = (cur_obj_type, cur_traj_type)
        if cur_grab_exp_tag not in data_inst_tag_to_optimized_res:
            data_inst_tag_to_optimized_res[cur_grab_exp_tag] = [cur_run_traj_sv_fn]
        else:
            data_inst_tag_to_optimized_res[cur_grab_exp_tag].append(cur_run_traj_sv_fn) # get the run traj sv fn #
        print(f"len(data_inst_tag_to_optimized_res): {len(data_inst_tag_to_optimized_res)}")
        # 
    
    data_inst_tag_to_optimized_res_sv_statistics_folder = "statistics"
    data_inst_tag_to_optimized_res_sv_statistics_folder = os.path.join(eval_data_folder, data_inst_tag_to_optimized_res_sv_statistics_folder)
    os.makedirs(data_inst_tag_to_optimized_res_sv_statistics_folder, exist_ok=True)
    data_inst_tag_to_optimized_res_sv_statistics_fn = "data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_sv_statistics_fn = os.path.join(data_inst_tag_to_optimized_res_sv_statistics_folder, data_inst_tag_to_optimized_res_sv_statistics_fn)
    np.save(data_inst_tag_to_optimized_res_sv_statistics_fn, data_inst_tag_to_optimized_res)
    print(f"data_inst_tag_to_optimized_res saved to {data_inst_tag_to_optimized_res_sv_statistics_fn}")
    return data_inst_tag_to_optimized_res
        


def inspect_data_inst_tag_to_optimized_res(data_inst_tag_to_optimized_res_fn):
    data_inst_tag_to_res = np.load(data_inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    res_nn_to_exp_nn = {}
    for inst_tag in data_inst_tag_to_res:
        cur_inst_tag_optimized_res = data_inst_tag_to_res[inst_tag]
        print(inst_tag, len(cur_inst_tag_optimized_res))
        cur_nn_res = len(cur_inst_tag_optimized_res)
        if cur_nn_res not in res_nn_to_exp_nn:
            res_nn_to_exp_nn[cur_nn_res] = 1
        else:
            res_nn_to_exp_nn[cur_nn_res] += 1
    print(res_nn_to_exp_nn)




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
    
    # # to the rotation angle in arc # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wstatebased_wcustomidamping_realleap/statistics/obj_type_to_optimized_res.npy #
    rot_dist = rot_dist.mean().item() # averge rotation angle differences # []
    
    return diff_obj_pos, rot_dist
    
    # for i_fr in range(tot_data_nn_frs):
    #     cur_fr_opt_obj_pose = opt_obj_pose[i_fr]
    #     cur_fr_kine_obj_pose = kine_obj_pose[i_fr]
    #     cur_fr_opt_obj_pos, cur_fr_opt_obj_ornt = cur_fr_opt_obj_pose[:3], cur_fr_opt_obj_pose[3: ]
    #     cur_fr_kine_obj_pos, cur_fr_kine_obj_ornt = cur_fr_kine_obj_pose[:3], cur_fr_kine_obj_pose[3: ] # kine obj pose # 
    #     cur_fr_diff_pos = np.linalg.norm(cur_fr_opt_obj_pos - cur_fr_kine_obj_pos)
        
    #     quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    #     rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
    
        
        
    #     cur_fr_diff_ornt = np.linalg.norm(cur_fr_opt_obj_ornt - cur_fr_kine_obj_ornt)
    

def best_optimized_res(data_optimized_res_nn, data_inst_tag, index=None, downsample=False):
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    
    if '_nf_300' in data_inst_tag:
        # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    else:
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
    
    
    
    if data_inst_tag.endswith(".npy"):
        cur_inst_kine_data_fn = data_inst_tag
    else:
        cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
        cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
    
    save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        
        
    # hand_qs = save_info['robot_delta_states_weights_np'][ : ]
    # hand_qs = hand_qs[: , : ]
    
    goal_obj_trans = save_info['object_transl']
    goal_obj_rot_quat = save_info['object_rot_quat']
    
    if downsample:
        idxes = [ ii for ii in range(goal_obj_trans.shape[0]) if ii % 2 == 0 ]
        idxes = np.array(idxes, dtype=np.int32)
        goal_obj_trans = goal_obj_trans[idxes]
        goal_obj_rot_quat = goal_obj_rot_quat[idxes]
        

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
    
    tot_obs_buf = []
    
    for ts in tot_ts:
        cur_optimized_obj_pose = optimized_res[ts]['object_pose']
        tot_optimized_obj_pose.append(cur_optimized_obj_pose)
        
        cur_optimized_hand_qs = optimized_res[ts]['shadow_hand_dof_pos']
        cur_optimized_hand_qtars = optimized_res[ts]['shadow_hand_dof_tars']
        tot_optimized_hand_qs.append(cur_optimized_hand_qs)
        tot_optimized_hand_qtars.append(cur_optimized_hand_qtars)
        tot_obs_buf.append(optimized_res[ts]['observations'])
        
    tot_optimized_obj_pose = np.stack(tot_optimized_obj_pose, axis=1) # nn_envs x nn_ts x 7 #
    tot_optimized_hand_qs = np.stack(tot_optimized_hand_qs, axis=1)
    tot_optimized_hand_qtars = np.stack(tot_optimized_hand_qtars, axis=1)
    tot_obs_buf = np.stack(tot_obs_buf, axis=1)
    
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
    # obj_pos_diff of the object position errors #
    new_optimized_info = {
        'optimized_obj_pose': tot_optimized_obj_pose[sorted_envs_idxes],
        'optimized_hand_qs': tot_optimized_hand_qs[sorted_envs_idxes],
        'optimized_hand_qtars': tot_optimized_hand_qtars[sorted_envs_idxes],
        'obj_pose_diff': tot_env_weighted_obj_pose_diff[sorted_envs_idxes],
        'obj_pos_diff': tot_env_diff_obj_pos[sorted_envs_idxes],
        'obj_rot_diff': tot_env_diff_obj_rot[sorted_envs_idxes],
        'obs_buf': tot_obs_buf
    }
    return new_optimized_info
    
    
# runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-35-23/ts_to_hand_obj_obs_reset_1.npy
# obj_code_to_best_opt_res = inspect_optimized_res_nn_w_object_codes(data_optimized_res_nn)
def inspect_optimized_res_nn_w_object_codes(data_optimized_res_nn):
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    env_object_codes = optimized_res['env_object_codes'] # 
    obj_codes_to_tot_optimized_res = {}
    # object_code_list, env_object_codes
    for i_env, env_obj_code in enumerate(env_object_codes):
        ## ge tthe env object poses #
        cur_env_obj_pose = []
        cur_env_hand_qs = []
        cur_env_hand_qtars = []
        tot_ts = list(optimized_res.keys())
        tot_ts = [ key for key in tot_ts if key != 'env_object_codes' and key != 'object_code_list' ]
        tot_ts = [ int(key)  for key in tot_ts ]
        tot_ts = sorted(tot_ts)
        for i_ts in tot_ts:
            cur_ts_res = optimized_res[i_ts] # optimized_res
            cur_ts_object_pose = cur_ts_res['object_pose']
            cur_ts_hand_dof_pos = cur_ts_res['shadow_hand_dof_pos']
            cur_ts_hand_dof_tars = cur_ts_res['shadow_hand_dof_tars']
            cur_env_obj_pose.append(cur_ts_object_pose[i_env])
            cur_env_hand_qs.append(cur_ts_hand_dof_pos[i_env])
            cur_env_hand_qtars.append(cur_ts_hand_dof_tars[i_env])
        cur_env_obj_pose = np.array(cur_env_obj_pose)
        cur_env_hand_qs = np.array(cur_env_hand_qs)
        cur_env_hand_qtars = np.array(cur_env_hand_qtars)
        if env_obj_code not in obj_codes_to_tot_optimized_res:
            obj_codes_to_tot_optimized_res[env_obj_code] = {
                'optimized_obj_pose': [cur_env_obj_pose],
                'optimized_hand_qs': [cur_env_hand_qs],
                'optimized_hand_qtars': [cur_env_hand_qtars]
            }
        else:
            obj_codes_to_tot_optimized_res[env_obj_code]['optimized_obj_pose'].append(cur_env_obj_pose)
            obj_codes_to_tot_optimized_res[env_obj_code]['optimized_hand_qs'].append(cur_env_hand_qs)
            obj_codes_to_tot_optimized_res[env_obj_code]['optimized_hand_qtars'].append(cur_env_hand_qtars)
    
    obj_code_to_best_opt_res = {}
    for obj_code in obj_codes_to_tot_optimized_res:
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
        cur_inst_kine_data_fn = f"passive_active_info_{obj_code}.npy"
        cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
        save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        goal_obj_trans = save_info['object_transl']
        goal_obj_rot_quat = save_info['object_rot_quat']
        goal_obj_pose = np.concatenate(
            [goal_obj_trans, goal_obj_rot_quat],  axis=-1
        )
        
        
        cur_obj_optimized_obj_pose = np.stack(obj_codes_to_tot_optimized_res[obj_code]['optimized_obj_pose'], axis=0)
        cur_obj_optimized_hand_qs = np.stack(obj_codes_to_tot_optimized_res[obj_code]['optimized_hand_qs'], axis=0)
        cur_obj_optimized_hand_qtars = np.stack(obj_codes_to_tot_optimized_res[obj_code]['optimized_hand_qtars'], axis=0)
        
        tot_env_diff_obj_pos, tot_env_diff_obj_rot = [], []
        tot_env_weighted_obj_pose_diff = []
        # w_pos, w_ornt = 1.0, 0.33
        w_pos, w_ornt = 1.0, 0.0
        for i_env in range(cur_obj_optimized_obj_pose.shape[0]):
            cur_optimized_obj_pose = cur_obj_optimized_obj_pose[i_env]
            cur_diff_obj_pos, cur_diff_obj_rot = calculate_obj_traj_diffs(cur_optimized_obj_pose, goal_obj_pose)
            tot_env_diff_obj_pos.append(cur_diff_obj_pos)
            tot_env_diff_obj_rot.append(cur_diff_obj_rot)
            weighted_diff_obj_pose = w_pos * cur_diff_obj_pos + w_ornt * cur_diff_obj_rot
            tot_env_weighted_obj_pose_diff.append(weighted_diff_obj_pose)
        tot_env_weighted_obj_pose_diff = np.array(tot_env_weighted_obj_pose_diff)
        sorted_envs_idxes = np.argsort(tot_env_weighted_obj_pose_diff)
        tot_env_diff_obj_pos = np.array(tot_env_diff_obj_pos)
        tot_env_diff_obj_rot = np.array(tot_env_diff_obj_rot)
        
        best_env_idx = sorted_envs_idxes[0].item()
        best_obj_pos_diff = tot_env_diff_obj_pos[best_env_idx].item()
        best_obj_rot_diff = tot_env_diff_obj_rot[best_env_idx].item()
        best_obj_pose_diff = tot_env_weighted_obj_pose_diff[best_env_idx].item()
        
        obj_code_to_best_opt_res[obj_code] = {
            'obj_pos_diff': best_obj_pos_diff,
            'obj_rot_diff': best_obj_rot_diff,
            'obj_pose_diff': best_obj_pose_diff,
        }
    return obj_code_to_best_opt_res

    
        
        
        
# /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_flute_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-16-28-36/ts_to_hand_obj_obs_reset_1.npy
def inspect_optimized_res_nn(data_optimized_res_nn, data_inst_tag, downsample=False):
    
    sorted_optimized_res_fn = data_optimized_res_nn.replace(".npy", "_sorted.npy")
    # if os.path.exists(sorted_optimized_res_fn):
    #     return
    
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    print(optimized_res.keys())
    
    # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # passive_active_info_ori_grab_s1_alarmclock_lift.npy
    # from the data
    
    new_optimized_info = best_optimized_res(data_optimized_res_nn, data_inst_tag[0], downsample=downsample)
    optimized_res.update(new_optimized_info)
    print(optimized_res.keys())
    ## save to a separte file ##
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
    first_obs_buf = optimized_res['obs_buf'][0:1]
    
    
    best_ts_to_opt_res_fn = sorted_optimized_res_fn.replace(".npy", "_best.npy")
    best_ts_to_opt_res = {
        'optimized_obj_pose': best_obj_pose,
        'optimized_hand_qtars': best_hand_qtars,
        'optimized_hand_qs': best_hand_qs,
        'obj_pose_diff': best_obj_pose_diff,
        'best_obj_pose_diff': best_obj_pose_diff_new,
        'best_obj_pos_diff': best_obj_pos_diff_new,
        'best_obj_rot_diff': best_obj_rot_diff_new,
        'obs_buf': first_obs_buf
    }
    print(f"best_obj_pos: {best_obj_pos_diff_new}, best_obj_rot: {best_obj_rot_diff_new}")
    np.save(best_ts_to_opt_res_fn, best_ts_to_opt_res)
    print(f"Saved best ts_to_opt_res to {best_ts_to_opt_res_fn}")

    
    # for key in optimized_res:
    #     # cur_grab_inst_tag = key
    #     # cur_grab_
    #     print(optimized_res[key].keys())
    #     break
    #     # val = optimized_res[key]
        # print(f"key: {key}, val: {val.shape}")



def resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=None, downsample=False):
    data_inst_tag_to_res = np.load(data_inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    print(data_inst_tag_to_res.keys())
    
    if target_sub_tag is not None:
        data_inst_tag_to_res = {
            key: data_inst_tag_to_res[key] for key in data_inst_tag_to_res if target_sub_tag in key[0]
        }
    
    # 
    cur_nn = 0
    for inst_tag in data_inst_tag_to_res:
        print(f"[{cur_nn}/{len(data_inst_tag_to_res)}] {inst_tag}")
        cur_inst_tag_optimized_res = data_inst_tag_to_res[inst_tag]
        for cur_inst_tag_optimized_res_fn in cur_inst_tag_optimized_res:
            if not os.path.exists(cur_inst_tag_optimized_res_fn):
                changed_root_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2"
                pure_fn = "/".join(cur_inst_tag_optimized_res_fn.split("/")[-2:])
                actual_res_fn = os.path.join(changed_root_folder, pure_fn)
            else: #
                actual_res_fn = cur_inst_tag_optimized_res_fn
                # cur_inst_tag_optimized_res_fn = cur_inst_tag_optimized_res_fn.replace(changed_root_folder, "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab")
            try:
                inspect_optimized_res_nn(actual_res_fn, inst_tag, downsample=downsample)
            except:
                pass
        cur_nn = cur_nn + 1


def inspect_new_eval_res(new_eval_res_sv_dict_fn):
    sv_dict = np.load( new_eval_res_sv_dict_fn, allow_pickle=True).item()
    print(sv_dict.keys())
    
    for cur_ts in sv_dict:
        cur_ts_dict = sv_dict[cur_ts]
        print(cur_ts_dict.keys())
        
def mv_evaled_data_to_eval_folder(target_eval_folder, forbid_subj_idx=None):
    local_eval_folder = "./runs"
    tot_eval_fns = os.listdir(local_eval_folder)
    tracking_fn_tag = "tracking_"
    tot_eval_fns = [
        fn for fn in tot_eval_fns if fn[: len(tracking_fn_tag)] == tracking_fn_tag
    ]
    if forbid_subj_idx is not None:
        forbid_subj_tag = f"_s{forbid_subj_idx}_"
        tot_eval_fns = [ fn for fn in tot_eval_fns if forbid_subj_tag not in fn]
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s1_alarmclock_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-00-11-57/ts_to_hand_obj_obs_reset_1.npy
    for cur_eval_fn in tot_eval_fns:
        cur_full_eval_fn = os.path.join(local_eval_folder, cur_eval_fn)
        
        print(f"mv {cur_full_eval_fn} {target_eval_folder}/")
        os.system(f"mv {cur_full_eval_fn} {target_eval_folder}/")
        
        # cur_eval_dict_fn = "ts_to_hand_obj_obs_reset_1.npy"
        # cur_eval_dict_fn = os.path.join(cur_full_eval_fn, cur_eval_dict_fn)
        # 

# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy
def inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn):
    inst_tag_to_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    for cur_inst_tag in inst_tag_to_res:
        print(cur_inst_tag)



def inspect_data_statistics(data_statistics_fn):
    data_statistics = np.load(data_statistics_fn, allow_pickle=True).item()
    print(data_statistics.keys())




def inspect_data_instance_to_optimized_res(data_inst_to_opt_res_fn, target_grab_inst_tag=None):
    data_inst_to_opt_res = np.load(data_inst_to_opt_res_fn, allow_pickle=True).item()
    for key in data_inst_to_opt_res:
        
        if target_grab_inst_tag is not None:
            if key[0] == target_grab_inst_tag:
                cur_val = data_inst_to_opt_res[key]
                print(f"key: {key}, cur_val: {cur_val}")
        
        # if "s2" in key[0]:
        #     cur_val = data_inst_to_opt_res[key]
        #     print(f"key: {key}, cur_val: {cur_val}")


def parse_obj_type_traj_type_fr_folder_name(folder_nm):
    exp_folder_st_tag = 'tracking_'
    remains_folder_nm = folder_nm[len(exp_folder_st_tag): ] ## get the remainng folder name ##
    folder_nm_segs = remains_folder_nm.split("_")
    obj_type_st_idx  = 0
    obj_type_ed_idx = 0
    for obj_type_ed_idx in range(0, len(folder_nm_segs)):
        if folder_nm_segs[obj_type_ed_idx] == 'OPTFR':
            break
    obj_type_segs = folder_nm_segs[obj_type_st_idx: obj_type_ed_idx]
    obj_type = "_".join(obj_type_segs)
    traj_type_st_idx = obj_type_ed_idx + 1
    traj_type_ed_idx = traj_type_st_idx
    for traj_type_ed_idx in range(traj_type_ed_idx, len(folder_nm_segs)):
        if folder_nm_segs[traj_type_ed_idx] == 'obs':
            break
    traj_type = folder_nm_segs[traj_type_st_idx: traj_type_ed_idx]
    traj_type = "_".join(traj_type)
    return obj_type, traj_type

# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-18-56-44
def calculate_OPTFR_exp_to_optimized_res(exp_root_folder):
    tot_exp_folders = os.listdir(exp_root_folder)
    exp_folder_st_tag = "tracking_"
    tot_exp_folders = [
        fn for fn in tot_exp_folders if fn[: len(exp_folder_st_tag)] == exp_folder_st_tag
    ]
    
    exp_config_to_tracking_ckpt = {}
    for i_exp, cur_exp_folder in enumerate(tot_exp_folders):
        obj_type, traj_type = parse_obj_type_traj_type_fr_folder_name(cur_exp_folder)
        print(f"[{i_exp}/{len(tot_exp_folders)}] obj_type: {obj_type}, traj_type: {traj_type}")
        
        cur_exp_full_folder = os.path.join(exp_root_folder,cur_exp_folder)
        cur_exp_full_nn_folder = os.path.join(cur_exp_full_folder, "nn")
        if not os.path.exists(cur_exp_full_nn_folder):
            print(f"cur_exp_full_nn_folder: {cur_exp_full_nn_folder} not exists")
            continue
        # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-18-56-44/nn/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
        tot_tracking_ckpts = os.listdir(cur_exp_full_nn_folder)
        tot_tracking_ckpts = [ fn for fn in tot_tracking_ckpts if fn.endswith(".pth") ]
        tot_tracking_ckpts = [ fn for fn in tot_tracking_ckpts if fn[: len(exp_folder_st_tag)] == exp_folder_st_tag ]
        if len(tot_tracking_ckpts) != 1:
            continue
        assert len(tot_tracking_ckpts) == 1
        cur_exp_tracking_ckpt = tot_tracking_ckpts[0]
        cur_exp_tracking_ckpt = os.path.join(cur_exp_full_nn_folder, cur_exp_tracking_ckpt)
        exp_config_to_tracking_ckpt[(obj_type, traj_type)] = cur_exp_tracking_ckpt
    exp_statistics_folder = os.path.join(exp_root_folder, "statistics")
    os.makedirs(exp_statistics_folder, exist_ok=True)
    obj_type_to_optimized_res_sv_fn = "obj_type_to_optimized_res.npy"
    obj_type_to_optimized_res_sv_fn = os.path.join(exp_statistics_folder, obj_type_to_optimized_res_sv_fn)
    np.save(obj_type_to_optimized_res_sv_fn, exp_config_to_tracking_ckpt) 
    print(f"obj_type_to_optimized_res saved to {obj_type_to_optimized_res_sv_fn}") 
    
    return exp_config_to_tracking_ckpt


def inspect_exp_tag_to_optimized_res(exp_tag_to_optimized_res_fn):
    exp_tag_to_optimized_res = np.load(exp_tag_to_optimized_res_fn, allow_pickle=True).item()
    for key in exp_tag_to_optimized_res:
        cur_val = exp_tag_to_optimized_res[key]
        print(f"key: {key}, cur_val: {cur_val}")


def inspect_optimized_res(optimized_res_fn):
    tot_optimized_res = np.load(optimized_res_fn, allow_pickle=True).item()
    tot_keys_nn = len(tot_optimized_res)
    
    for key in tot_optimized_res:
        cur_val = tot_optimized_res[key]
        print(f"key: {key}, cur_val: {cur_val}")
        break
    tot_evaluated_grab_inst_fns = [key[0] for key in tot_optimized_res]
    for subj in range(1, 11):
        cur_subj_res_tag = f"_s{subj}_"
        cur_subj_grab_inst_fns = [ key for key in tot_evaluated_grab_inst_fns if cur_subj_res_tag in key  ]
        print(f"subj: {subj}, len(cur_subj_grab_inst_fns): {len(cur_subj_grab_inst_fns)}")
        
def inspect_inst_tag_to_optimized_res(inst_to_optimize_res_fn):
    inst_tag_to_optimized_res = np.load(inst_to_optimize_res_fn, allow_pickle=True).item()
    for inst_tag in inst_tag_to_optimized_res:
        cur_val = inst_tag_to_optimized_res[inst_tag]
        print(f"inst_tag: {inst_tag}, cur_val: {cur_val}")
   
     
# tracking_ori_grab_s4_apple_eat_1_OPTFR_ori_grab_s4_stapler_staple_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_26-02-16-52

def get_exp_info_from_folder_nm(exp_folder_fn):
    tracking_folder_fn = "tracking_"
    remains_folder_fn = exp_folder_fn[len(tracking_folder_fn): ]
    remains_folder_segs = remains_folder_fn.split("_") ## get the remains folder name ## 
    st_cur_obj_type = 0
    if '_OPTFR_' in remains_folder_fn:
        for ed_cur_obj_type in range(st_cur_obj_type, len(remains_folder_segs)):
            if remains_folder_segs[ed_cur_obj_type] == 'OPTFR':
                break
        st_fa_obj_type = ed_cur_obj_type + 1
        for ed_fa_obj_type in range(st_fa_obj_type, len(remains_folder_segs)):
            if remains_folder_segs[ed_fa_obj_type] == 'obs':
                break
        cur_obj_type = "_".join(remains_folder_segs[st_cur_obj_type: ed_cur_obj_type])
        fa_obj_type = "_".join(remains_folder_segs[st_fa_obj_type: ed_fa_obj_type])
    else:
        for ed_cur_obj_type in range(st_cur_obj_type, len(remains_folder_segs)):
            if remains_folder_segs[ed_cur_obj_type] == 'obs':
                break
        cur_obj_type = "_".join(remains_folder_segs[st_cur_obj_type: ed_cur_obj_type])
        fa_obj_type = cur_obj_type
    
    return cur_obj_type, fa_obj_type

def inspect_exp_info_to_optimized_res(exp_root_folder, subj_idx=None):
    tot_exp_folders = os.listdir(exp_root_folder)
    tracking_folder_fn = "tracking_"
    tot_exp_folders = [
        fn for fn in tot_exp_folders if fn[: len(tracking_folder_fn)] == tracking_folder_fn
    ]
    if subj_idx is not None:
        subj_tag = f"_s{subj_idx}_"
        tot_exp_folders = [
            fn for fn in tot_exp_folders if subj_tag in fn
        ]
    exp_info_to_optimized_res = { }
    for cur_exp_folder in tot_exp_folders:
        cur_obj_type, fa_obj_type = get_exp_info_from_folder_nm(cur_exp_folder)
        cur_full_exp_folder = os.path.join(exp_root_folder, cur_exp_folder)
        cur_full_exp_folder_nn = os.path.join(cur_full_exp_folder, "nn") 
        # get the ful exp folder # 
        if not os.path.exists(cur_full_exp_folder_nn):
            continue
        best_rew, best_ckpt_fn = find_best_rew(cur_full_exp_folder) 
        ### get the best rew and the ckpt fn ###
        exp_info_to_optimized_res[ ( cur_obj_type, fa_obj_type ) ] = (best_rew, best_ckpt_fn)
    statistics_folder = os.path.join(exp_root_folder, "statistics")
    os.makedirs(statistics_folder, exist_ok=True) ## statistics folder ## 
    exp_info_to_optimized_res_fn = os.path.join(statistics_folder, "exp_info_to_optimized_res.npy")
    np.save( exp_info_to_optimized_res_fn, exp_info_to_optimized_res ) # exp info to optimized res #
    return exp_info_to_optimized_res 


# def inspect_exp_info_to_optimized_res(exp_root_folder):
#     tot_exp_folders = os.listdir(exp_root_folder)
#     tracking_folder_fn = "tracking_"
#     tot_exp_folders = [
#         fn for fn in tot_exp_folders if fn[: len(tracking_folder_fn)] == tracking_folder_fn
#     ]
#     exp_info_to_optimized_res = { }
#     for cur_exp_folder in tot_exp_folders:
#         cur_obj_type, fa_obj_type = get_exp_info_from_folder_nm(cur_exp_folder)
#         cur_full_exp_folder = os.path.join(exp_root_folder, cur_exp_folder)
#         cur_full_exp_folder_nn = os.path.join(cur_full_exp_folder, "nn") 
#         # get the ful exp folder # 
#         if not os.path.exists(cur_full_exp_folder_nn):
#             continue
#         best_rew, best_ckpt_fn = find_best_rew(cur_full_exp_folder) 
#         ### get the best rew and the ckpt fn ###
#         exp_info_to_optimized_res[ ( cur_obj_type, fa_obj_type ) ] = (best_rew, best_ckpt_fn)
#     statistics_folder = os.path.join(exp_root_folder, "statistics")
#     os.makedirs(statistics_folder, exist_ok=True) ## statistics folder ## 
#     exp_info_to_optimized_res_fn = os.path.join(statistics_folder, "exp_info_to_optimized_res.npy")
#     np.save( exp_info_to_optimized_res_fn, exp_info_to_optimized_res ) # exp info to optimized res #
#     return exp_info_to_optimized_res 



def inspect_optfr_with_selfexp(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=False):
    exp_info_to_opt_res = np.load(exp_info_to_opt_res_fn, allow_pickle=True).item()
    self_exp_info_to_opt_res = np.load(self_exp_res_to_opt_res_fn, allow_pickle=True).item()
    
    inst_tag_to_opt_res = {}
    # k * n? since it should not be a deterministic mapping here #
    for exp_info in exp_info_to_opt_res:
        cur_inst_tag = exp_info[0]
        cur_inst_fa_tag = exp_info[1]
        print(f"exp_info: {exp_info}")
        if cur_inst_tag not in inst_tag_to_opt_res:
            
            inst_tag_to_opt_res[cur_inst_tag] = (exp_info_to_opt_res[exp_info][0], cur_inst_fa_tag) # rew and the father task tag #
        else:
            cur_best_rew = exp_info_to_opt_res[exp_info][0]
            prev_best_rew = inst_tag_to_opt_res[cur_inst_tag][0] ### best and the optimized res 
            if cur_best_rew > prev_best_rew:
                inst_tag_to_opt_res[cur_inst_tag] = ( exp_info_to_opt_res[exp_info][0], cur_inst_fa_tag )
                
            
            # inst_tag_to_opt_res[cur_inst_tag] = max(inst_tag_to_opt_res[cur_inst_tag], exp_info_to_opt_res[exp_info][0])
    # print(inst_tag_to_opt_res.keys())
    ### ###
    
    child_task_to_fa_task = {} 
    better_nn = 0
    tot_nn = 0
    for key in self_exp_info_to_opt_res:
        if key in inst_tag_to_opt_res: ## inst tag to opt res ##
            
            inherited_optimized_res = inst_tag_to_opt_res[key][0]
            self_optimized_res = self_exp_info_to_opt_res[key][0]
        
            if inherited_optimized_res > self_optimized_res: ### inst tag ###
                child_task_to_fa_task[key] = inst_tag_to_opt_res[key][1] ### inst tag ###
                better_nn += 1
                print(key, self_exp_info_to_opt_res[key][0], inst_tag_to_opt_res[key]) 
            tot_nn += 1
            
            # print(key, self_exp_info_to_opt_res[key][0], inst_tag_to_opt_res[key]) 
    print(f"better_nn/tot_nn: {better_nn}/{tot_nn}")
    
    if save_res:
        exp_info_to_opt_res_sv_folder = "/".join(exp_info_to_opt_res_fn.split("/")[:-1])
        child_task_to_fa_task_fn = "child_task_to_fa_task.npy"
        child_task_to_fa_task_fn = os.path.join(exp_info_to_opt_res_sv_folder, child_task_to_fa_task_fn)  
        np.save(child_task_to_fa_task_fn, child_task_to_fa_task) # 
        print(f"Child task to father task dict saved to {child_task_to_fa_task_fn}") 
    
    # is trajectory translation a good strategy #
    ## inspect thew selfexp ##
    
    ## chld ttask ##
    # child_to_fa_task = {}
    # for key in self_exp_info_to_opt_res: #
    # find the best task #
    # find the best task to solve #
    # a task planner? #
    # hierachical policy? #
    
    ###





def inspect_optfr_with_selfexp_list(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=False):
    exp_info_to_opt_res = np.load(exp_info_to_opt_res_fn, allow_pickle=True).item()
    self_exp_info_to_opt_res = np.load(self_exp_res_to_opt_res_fn, allow_pickle=True).item()
    
    inst_tag_to_opt_res = {}
    # k * n? since it should not be a deterministic mapping here #
    for exp_info in exp_info_to_opt_res:
        cur_inst_tag = exp_info[0]
        cur_inst_fa_tag = exp_info[1]
        print(f"exp_info: {exp_info}")
        if cur_inst_tag not in inst_tag_to_opt_res:
            inst_tag_to_opt_res[cur_inst_tag] = [(exp_info_to_opt_res[exp_info][0], cur_inst_fa_tag)]  
        else:
            cur_best_rew = exp_info_to_opt_res[exp_info][0]
            # prev_best_rew = inst_tag_to_opt_res[cur_inst_tag][0] ### best and the optimized res 
            # exp info #
            rew_w_fa_tag = ( cur_best_rew,   cur_inst_fa_tag)
            
            inst_tag_to_opt_res[cur_inst_tag].append(rew_w_fa_tag) # 
            
            # if cur_best_rew > prev_best_rew:
            #     inst_tag_to_opt_res[cur_inst_tag] = ( exp_info_to_opt_res[exp_info][0], cur_inst_fa_tag )
                
            
            # inst_tag_to_opt_res[cur_inst_tag] = max(inst_tag_to_opt_res[cur_inst_tag], exp_info_to_opt_res[exp_info][0])
    # print(inst_tag_to_opt_res.keys())
    ### ###
    
    child_task_to_fa_task = {} 
    better_nn = 0
    tot_nn = 0
    for key in self_exp_info_to_opt_res:
        if key in inst_tag_to_opt_res: ## inst tag to opt res ##
            self_optimized_res = self_exp_info_to_opt_res[key][0]
            tot_fa_infos = inst_tag_to_opt_res[key]
            for cur_fa_info in tot_fa_infos:
                cur_fa_rew, cur_fa_tag = cur_fa_info
                if cur_fa_rew > self_optimized_res:
                    if key not in child_task_to_fa_task:
                        better_nn += 1
                        child_task_to_fa_task[key] = [cur_fa_tag]
                    else:
                        child_task_to_fa_task[key].append(cur_fa_tag)
            
            
            # inherited_optimized_res = inst_tag_to_opt_res[key][0]
            # 
        
            # if inherited_optimized_res > self_optimized_res: ### inst tag ###
            #     child_task_to_fa_task[key] = inst_tag_to_opt_res[key][1] ### inst tag ###
            #     better_nn += 1
            #     print(key, self_exp_info_to_opt_res[key][0], inst_tag_to_opt_res[key]) 
            tot_nn += 1
            
            # print(key, self_exp_info_to_opt_res[key][0], inst_tag_to_opt_res[key]) 
    print(f"better_nn/tot_nn: {better_nn}/{tot_nn}")
    
    if save_res: # exp info to opt res #
        exp_info_to_opt_res_sv_folder = "/".join(exp_info_to_opt_res_fn.split("/")[:-1])
        child_task_to_fa_task_fn = "child_task_to_fa_task_list.npy" # child task to father task #
        child_task_to_fa_task_fn = os.path.join(exp_info_to_opt_res_sv_folder, child_task_to_fa_task_fn)  
        np.save(child_task_to_fa_task_fn, child_task_to_fa_task)
        print(f"Child task to father task dict saved to {child_task_to_fa_task_fn}") 





def inspect_opt_res(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=False):
    exp_info_to_opt_res = np.load(exp_info_to_opt_res_fn, allow_pickle=True).item()
    self_exp_info_to_opt_res = np.load(self_exp_res_to_opt_res_fn, allow_pickle=True).item()
    
    inst_tag_to_opt_res = {}
    
    tot_nn = 0
    better_nn = 0
    
    succ_rew = 15.0
    self_tot_nn = 0
    self_succ_nn = 0
    target_subj_tag = '_s9'
    for key in self_exp_info_to_opt_res:
        if target_subj_tag in key:
            cur_self_opt_rew = self_exp_info_to_opt_res[key][0]
            if cur_self_opt_rew > succ_rew:
                self_succ_nn += 1
            self_tot_nn += 1
            
    print(f"self_succ_nn/self_tot_nn: {self_succ_nn}/{self_tot_nn}")
    
    other_succ_nn = 0
    other_tot_nn = 0
    for key in exp_info_to_opt_res:
        if target_subj_tag in key:
            cur_self_opt_rew = exp_info_to_opt_res[key][0]
            if cur_self_opt_rew > succ_rew:
                other_succ_nn += 1
            other_tot_nn += 1
            
    print(f"other_succ_nn/other_tot_nn: {other_succ_nn}/{other_tot_nn}")
    
    
    
    for key in exp_info_to_opt_res:
        if key in self_exp_info_to_opt_res:
            cur_self_opt_res = self_exp_info_to_opt_res[key][0]
            cur_other_opt_res = exp_info_to_opt_res[key][0]
            if cur_other_opt_res > cur_self_opt_res:
                better_nn += 1
            tot_nn += 1
            
    print(f"better_nn/tot_nn: {better_nn}/{tot_nn}")
    
    




def inspect_child_task_to_fa_task(ch_to_fa_task_sv_fn):
    ch_to_fa_task = np.load(ch_to_fa_task_sv_fn, allow_pickle=True ).item()
    for key in ch_to_fa_task:
        print(key, ch_to_fa_task[key])
        # 
        
def inspect_grab_cross_obj_diff(grab_obj_name_idx_dict_fn):
    grab_obj_nm_idx = np.load(grab_obj_name_idx_dict_fn, allow_pickle=True).item()
    for key in grab_obj_nm_idx: 
        val = grab_obj_nm_idx[key]
        print(f"key: {key}, val: {val}")

# <<<<<<< HEAD
def merge_child_task_to_fa_task(local_dict_fn, glb_dict_fn):
    glb_dict = np.load(glb_dict_fn, allow_pickle=True).item()
    local_dict = np.load(local_dict_fn, allow_pickle=True).item()
    for key in local_dict:
        glb_dict[key] = local_dict[key]
    print(glb_dict.keys())
    np.save(glb_dict_fn, glb_dict)


# # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples
# =======
# >>>>>>> c6fbd593bf99644e2c520da3efc0c713832c4f66

def test_ckpt(ckpt_fn):
    ckpt_dict = torch.load(ckpt_fn, map_location='cpu')
    print(ckpt_dict.keys())
    

def merge_child_task_to_fa_task(local_dict_fn, glb_dict_fn):
    glb_dict = np.load(glb_dict_fn, allow_pickle=True).item()
    local_dict = np.load(local_dict_fn, allow_pickle=True).item()
    for key in local_dict:
        glb_dict[key] = local_dict[key]
    print(glb_dict.keys())
    np.save(glb_dict_fn, glb_dict)


def inspect_obj_feat_sample(obj_feat_sample_fn):
    obj_feat_samples =  np.load(obj_feat_sample_fn, allow_pickle=True).item()
    sample_key = 'samples'
    obj_feat_samples = obj_feat_samples[sample_key]
    for key in obj_feat_samples:
        cur_val = obj_feat_samples[key]
        # print(f"key: {key}, cur_val: {type(cur_val)}")
        if isinstance(cur_val, np.ndarray):
            print(f"key: {key}, cur_val: {cur_val.shape}")
        else:
            print(f"key: {key}, cur_val: {cur_val}")
# the feat dim is 256 here # 


def parse_object_info_from_kine_fn(kine_fn):
    
    kine_fn = kine_fn.split("/")[-1]
    
    if 'taco' in kine_fn:
        tracking_info_st_tag = "passive_active_info_ori_grab_s2_phone_call_1_interped_"
        inst_tag_fn = kine_fn.split(".")[0]
        inst_tag_fn = inst_tag_fn[len(tracking_info_st_tag): ]
        inst_tag_splits = inst_tag_fn.split("_")
        inst_tag_name = inst_tag_splits[0: 3]
        inst_tag = "_".join(inst_tag_name)
    else:
    
        tracking_info_st_tag = "passive_active_info_"
        inst_tag = kine_fn.split(".")[0][len(tracking_info_st_tag): ]
    return inst_tag

def summarize_traj_features(obj_sample_folder, nn_bsz):
    ep_idx = 0
    batch_st_idx = 0
    batch_ed_idx = nn_bsz
    obj_type_to_obj_feat = {}
    for cur_batch_idx in range(batch_st_idx, batch_ed_idx):
        cur_sample_batch_fn = f"samples_ep_{ep_idx}_batch_{cur_batch_idx}.npy"
        cur_sample_batch_full_fn = os.path.join(obj_sample_folder, cur_sample_batch_fn)
        cur_sample_batch = np.load(cur_sample_batch_full_fn, allow_pickle=True).item()
        print(f"cur_sample_batch: {cur_sample_batch.keys()}")
        tot_data_nm = cur_sample_batch['data_nm']
        feat_feat = cur_sample_batch['samples']['feat_feat']
        feat_feat = np.max(feat_feat, axis=0)
        print(f"feat_feat: {feat_feat.shape}, tot_data_nm: {len(tot_data_nm)}")
        for i_data, cur_data_nm in enumerate(tot_data_nm):
            cur_inst_tag = parse_object_info_from_kine_fn(cur_data_nm)
            cur_inst_feat_feat = feat_feat[i_data]
            obj_type_to_obj_feat[cur_inst_tag]=cur_inst_feat_feat
        
        # cur_sample_batch = cur_sample_batch['samples']
        # 
        # object_type = cur_sample_batch['object_type'][0][0]
        # pts_feat = cur_sample_batch['pts_feat']
        # for i_obj, cur_obj_type in enumerate(object_type):
        #     cur_obj_pts_feat = pts_feat[i_obj]
        #     obj_type_to_obj_feat[cur_obj_type] = cur_obj_pts_feat
    obj_type_to_obj_feat_fn = f"inst_tag_to_obj_feat.npy"
    obj_type_to_obj_feat_fn = os.path.join(obj_sample_folder, obj_type_to_obj_feat_fn)
    np.save(obj_type_to_obj_feat_fn, obj_type_to_obj_feat) 
    print(f"obj_type_to_obj_feat: {obj_type_to_obj_feat_fn}")



def summarize_obj_features(obj_sample_folder, nn_bsz):
    ep_idx = 0
    batch_st_idx = 0
    batch_ed_idx = nn_bsz
    obj_type_to_obj_feat = {}
    for cur_batch_idx in range(batch_st_idx, batch_ed_idx):
        cur_sample_batch_fn = f"samples_ep_{ep_idx}_batch_{cur_batch_idx}.npy"
        cur_sample_batch_full_fn = os.path.join(obj_sample_folder, cur_sample_batch_fn)
        cur_sample_batch = np.load(cur_sample_batch_full_fn, allow_pickle=True).item()
        cur_sample_batch = cur_sample_batch['samples']
        # object_type #
        # object_type = cur_sample_batch['object_type'][0][0]
        
        object_type = cur_sample_batch['data_nm'][0][0]
        # print(object_type)
        # cur_inst_tag = parse_object_info_from_kine_fn(cur_data_nm)
        object_type = [ parse_object_info_from_kine_fn(cur_obj_data_nm) for cur_obj_data_nm in object_type ]
        
        pts_feat = cur_sample_batch['pts_feat']
        for i_obj, cur_obj_type in enumerate(object_type):
            cur_obj_pts_feat = pts_feat[i_obj]
            obj_type_to_obj_feat[cur_obj_type] = cur_obj_pts_feat
    obj_type_to_obj_feat_fn = f"obj_type_to_obj_feat.npy"
    obj_type_to_obj_feat_fn = os.path.join(obj_sample_folder, obj_type_to_obj_feat_fn)
    np.save(obj_type_to_obj_feat_fn, obj_type_to_obj_feat) 
    print(f"obj_type_to_obj_feat: {obj_type_to_obj_feat_fn}")


def inspect_obj_feats(obj_type_to_obj_feat_fn):
    obj_type_to_obj_feat = np.load(obj_type_to_obj_feat_fn, allow_pickle=True).item()
    for obj_type in obj_type_to_obj_feat:
        cur_obj_feat = obj_type_to_obj_feat[obj_type]
        if 'taco_' in obj_type:
            print(f"obj_type: {obj_type}, obj_feat: {cur_obj_feat.shape}")

# grab data utils #

def inspect_ckpt(ckpt_fn):
    ckpt = torch.load(ckpt_fn, map_location='cpu')
    print(ckpt.keys())
    model_weights = ckpt['model']
    print(model_weights.keys())
    



def inspect_best_optimized_res(best_opt_fn):
    best_opt_dict = np.load(best_opt_fn, allow_pickle=True).item()
    print(best_opt_dict.keys())
    



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
        
        if not os.path.exists(cur_inst_opt_res_fn):
            changed_root_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2"
            pure_fn = "/".join(cur_inst_opt_res_fn.split("/")[-2:])
            actual_res_fn = os.path.join(changed_root_folder, pure_fn)
        else:
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
                cur_inst_opt_res_fn_sorted_best=  cur_inst_opt_res_fn_sorted.replace(".npy", "_best_vv.npy")
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
        

        
        # get the best optimized results -> load the best opt res #
        # print(f"[{cur_nn}/{len(data_inst_tag_to_res)}] {inst_tag}")
        # cur_inst_tag_optimized_res = data_inst_tag_to_res[inst_tag]
        # for cur_inst_tag_optimized_res_fn in cur_inst_tag_optimized_res:
        #     inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag)
        # cur_nn = cur_nn + 1
        
    sorted_items = list(data_inst_tag_to_best_opt_res.items())
    sorted_items = sorted(sorted_items, key=lambda x: x[1]['obj_pos_diff'])
    print(sorted_items[:40])
    return data_inst_tag_to_best_opt_res
    pass




def inspect_data_inst_to_best_opt_res(data_inst_to_best_opt_res_fn, target_sub_tag=None):
    data_inst_to_best_opt_res = np.load(data_inst_to_best_opt_res_fn, allow_pickle=True).item()
    for cur_data_inst in data_inst_to_best_opt_res:
        if target_sub_tag is not None:
            if target_sub_tag not in cur_data_inst[0]:
                continue
        cur_val = data_inst_to_best_opt_res[cur_data_inst]
        print(f"cur_data_inst: {cur_data_inst}, cur_val: {cur_val}")

### ### --- test them all ? ####
### better than the origianl one ? ##
## no -- better than the jointly training results ##
def compare_best_opt_res(data_inst_to_opt_res_fn_a, data_inst_to_opt_fn_b):
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    data_inst_to_opt_res_b = np.load(data_inst_to_opt_fn_b, allow_pickle=True).item()
    tot_nn = 0
    better_nn = 0
    for inst_tag in data_inst_to_opt_res_b:
        if (inst_tag, inst_tag) in data_inst_to_opt_res_a:
            cur_val_a = data_inst_to_opt_res_a[(inst_tag, inst_tag)]
            cur_val_b = data_inst_to_opt_res_b[inst_tag]
            cmp_obj_diff_key = 'obj_pose_diff'
            # cmp_obj_diff_key = 'obj_pos_diff'
            
            # print(f"inst_tag: {inst_tag}, cur_val_a: {cur_val_a['obj_pos_diff']}, cur_val_b: {cur_val_b['obj_pos_diff']}")
            
            cur_a_pos_diff = cur_val_a[cmp_obj_diff_key][0].item()
            cur_b_pos_diff = cur_val_b[cmp_obj_diff_key]
            tot_nn += 1
            if cur_a_pos_diff < cur_b_pos_diff:
                better_nn += 1
                print(f"inst_tag: {inst_tag}, cur_val_a: {cur_val_a['obj_pos_diff']}, cur_val_b: {cur_val_b['obj_pos_diff']}")
    print(f"better_nn/tot_nn: {better_nn}/{tot_nn}")


def calcu_eval_metrics():
    five_degree_arc = 5.0 * np.pi / 180.0
    print(five_degree_arc)
    ten_degree_arc = 10.0 * np.pi / 180.0
    print(ten_degree_arc)
    fifteen_degree_arc = 15.0 * np.pi / 180.0
    print(fifteen_degree_arc)
    twenty_degree_arc = 20.0 * np.pi / 180.0
    print(twenty_degree_arc)
    thirty_degree_arc = 30.0 * np.pi / 180.0
    forty_degree_arc = 40.0 * np.pi / 180.0
    print(thirty_degree_arc)
    print(forty_degree_arc)
    
    
def calcualte_succ_info(data_inst_to_opt_res_fn):
    data_inst_to_best_opt_res = np.load(data_inst_to_opt_res_fn, allow_pickle=True).item() # opt res fn # 
    pos_thres = 0.1
    ornt_thres = 0.6981317007977318
    # pos_thres = 0.05
    # ornt_thres = 0.6981317007977318
    
    tot_nn = 0
    succ_nn = 0
    # 's2' 
    # 
    target_inst_tag = 's2'
    for cur_data_inst_tag in data_inst_to_best_opt_res:
        if isinstance(cur_data_inst_tag, tuple):
            if target_inst_tag not in cur_data_inst_tag[0]: 
                continue
        elif isinstance(cur_data_inst_tag, str):
            if target_inst_tag not in cur_data_inst_tag:
                continue
        cur_data_inst_val = data_inst_to_best_opt_res[cur_data_inst_tag]
        cur_obj_pos_diff = cur_data_inst_val['obj_pos_diff'] # [0].item()
        cur_obj_rot_diff = cur_data_inst_val['obj_rot_diff'] # [0].item()
        
        if isinstance(cur_obj_pos_diff, np.ndarray):
            cur_obj_pos_diff = cur_obj_pos_diff[0].item()
        if isinstance(cur_obj_rot_diff, np.ndarray):
            cur_obj_rot_diff = cur_obj_rot_diff[0].item()
            
        
        if cur_obj_pos_diff < pos_thres and cur_obj_rot_diff < ornt_thres:
            succ_nn += 1
        tot_nn += 1
    print(f"succ_nn/tot_nn: {succ_nn}/{tot_nn}")


### NOTE: the fn_a is the original saved info file; and the fn_b is the re-saved by the jointly optimized strategy ####
def calcualte_merged_succ_info(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b):
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    data_inst_to_opt_res_b = np.load(data_inst_to_opt_res_fn_b, allow_pickle=True).item()
    pos_thres = 0.1
    ornt_thres = 0.6981317007977318
    # pos_thres = 0.05
    # ornt_thres = 0.6981317007977318
    tot_nn = 0
    succ_nn = 0
    target_inst_tag = 's2'
    # target_inst_tag = 's9'
    for cur_inst_tag in data_inst_to_opt_res_a:
        print(f"cur_inst_tag: {cur_inst_tag}")
        # if isinstance(cur_inst_tag, tuple):
        #     if target_inst_tag not in cur_inst_tag[0]: 
        #         continue
        cur_data_inst_val = data_inst_to_opt_res_a[cur_inst_tag]
        cur_a_pos_diff = cur_data_inst_val['obj_pos_diff'][0].item()
        cur_a_rot_diff = cur_data_inst_val['obj_rot_diff'][0].item()
        
        # if cur_inst_tag[0] in data_inst_to_opt_res_b:
        #     # print(f"cur_inst_tag: {cur_inst_tag}, xxx")
        #     cur_b_inst_val = data_inst_to_opt_res_b[cur_inst_tag[0]]
        #     cur_b_pos_diff = cur_b_inst_val['obj_pos_diff']
        #     cur_b_rot_diff = cur_b_inst_val['obj_rot_diff']
        #     cur_a_pose_diff = cur_a_pos_diff + cur_a_rot_diff * 0.33
        #     cur_b_pose_diff = cur_b_pos_diff + cur_b_rot_diff * 0.33
        #     # if cur_b_pose_diff < cur_a_pose_diff:
        #     if cur_b_pos_diff < cur_a_pos_diff:
        #         print(f"cur_inst_tag: {cur_inst_tag}, cur_a_pos_diff: {cur_a_pos_diff}, cur_a_rot_diff: {cur_a_rot_diff}, cur_b_pos_diff: {cur_b_pos_diff}, cur_b_rot_diff: {cur_b_rot_diff}")
        #         cur_a_pos_diff = cur_b_pos_diff
        #         cur_a_rot_diff = cur_b_rot_diff
                
        # if cur_a_pos_diff < pos_thres and cur_a_rot_diff < ornt_thres:
        if cur_a_pos_diff < pos_thres:
            succ_nn += 1
        tot_nn += 1
    print(f"succ_nn/tot_nn: {succ_nn}/{tot_nn}")
    
def calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b):
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    data_inst_to_opt_res_b = np.load(data_inst_to_opt_res_fn_b, allow_pickle=True).item()
    pos_thres = 0.15
    
    
    
    
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
    for subj_inst_idx in range(1, 11):
        tot_nn = 0
        succ_nn = 0

        target_inst_tag = 's2'

        target_inst_tag = 's1'
        target_inst_tag = f's{subj_inst_idx}_'

        for cur_inst_tag in data_inst_to_opt_res_a:
            if isinstance(cur_inst_tag, tuple):
                if target_inst_tag not in cur_inst_tag[0]: 
                    continue
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

            # if cur_a_pos_diff < pos_thres and cur_a_rot_diff < ornt_thres: # --- get the single model #
            if cur_a_pos_diff < pos_thres:
                succ_nn += 1
            tot_nn += 1
        tot_tot_nn.append(tot_nn)
        tot_succ_nn.append(succ_nn)
        print(f"succ_nn/tot_nn: {succ_nn}/{tot_nn}")
    tot_tot_nn = sum(tot_tot_nn)
    tot_succ_nn = sum(tot_succ_nn)
    res = float(tot_succ_nn) / float(tot_tot_nn)
    print(f"tot_succ_nn/tot_tot_nn: {tot_succ_nn}/{tot_tot_nn}, res: {res}")


def inspect_sorted_info(data_inst_to_opt_res_fn_a, data_inst_tag_to_optimized_res_fn):
    data_inst_to_opt_res = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    data_inst_tag_to_optimized_res = np.load(data_inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    print(f"data_inst_tag_to_optimized_res: {data_inst_tag_to_optimized_res.keys()}")
    # obj_pos_diff , obj_rot_diff #
    data_inst_terms = [
        (key, data_inst_to_opt_res[key]['obj_pos_diff'][0].item(), data_inst_to_opt_res[key]['obj_rot_diff'][0].item(), data_inst_tag_to_optimized_res[key]) for key in data_inst_to_opt_res
    ]
    data_inst_terms = sorted(data_inst_terms, key=lambda x: x[1], reverse=False)
    print(data_inst_terms)

def calcualte_merged_succ_info_all_thres(data_inst_to_opt_res_fn_a):
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    # data_inst_to_opt_res_b = np.load(data_inst_to_opt_res_fn_b, allow_pickle=True).item()
    pos_thres = 0.15
    
    tot_pos_threses = [0.10, 0.10, 0.15]
    tot_ornt_threses = [0.3490658503988659, 0.6981317007977318, 0.6981317007977318]
    
    tot_succ_rates = []
    # tot succ rates #
    
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

        target_inst_tag = 's1'
        # target_inst_tag = f's{subj_inst_idx}_'

        for cur_inst_tag in data_inst_to_opt_res_a:
            if isinstance(cur_inst_tag, tuple):
                if target_inst_tag not in cur_inst_tag[0]: 
                    continue
            cur_data_inst_val = data_inst_to_opt_res_a[cur_inst_tag]
            # print(f"cur_data_inst_val: {cur_data_inst_val}")
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



def calcualte_merged_succ_info_v3(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b):
    data_inst_to_opt_res_a = np.load(data_inst_to_opt_res_fn_a, allow_pickle=True).item()
    data_inst_to_opt_res_b = np.load(data_inst_to_opt_res_fn_b, allow_pickle=True).item()
    pos_thres = 0.1
    ornt_thres = 0.6981317007977318
    # pos_thres = 0.05
    # ornt_thres = 0.6981317007977318
    for subj_inst_idx in range(1, 11):
        tot_nn = 0
        succ_nn = 0
        target_inst_tag = 's2'
        target_inst_tag = 's9'
        target_inst_tag = f's{subj_inst_idx}'
        for cur_inst_tag in data_inst_to_opt_res_a:
            # if isinstance(cur_inst_tag, tuple):
            if target_inst_tag not in cur_inst_tag:
                continue
            cur_data_inst_val = data_inst_to_opt_res_a[cur_inst_tag]
            cur_a_pos_diff = cur_data_inst_val['obj_pos_diff']
            cur_a_rot_diff = cur_data_inst_val['obj_rot_diff']
            # if cur_inst_tag in data_inst_to_opt_res_b:
            #     cur_b_inst_val = data_inst_to_opt_res_b[cur_inst_tag]
            #     cur_b_pos_diff = cur_b_inst_val['obj_pos_diff'][0].item()
            #     cur_b_rot_diff = cur_b_inst_val['obj_rot_diff'][0].item()
            #     cur_a_pose_diff = cur_a_pos_diff + cur_a_rot_diff * 0.33
            #     cur_b_pose_diff = cur_b_pos_diff + cur_b_rot_diff * 0.33
            #     if cur_b_pose_diff < cur_a_pose_diff:
            #         cur_a_pos_diff = cur_b_pos_diff
            #         cur_a_rot_diff = cur_b_rot_diff #
            #     print(f"cur_inst_tag: {cur_inst_tag}, cur_a_pos_diff: {cur_a_pos_diff}, cur_a_rot_diff: {cur_a_rot_diff}, cur_b_pos_diff: {cur_b_pos_diff}, cur_b_rot_diff: {cur_b_rot_diff}")
            # if cur_a_pos_diff < pos_thres and cur_a_rot_diff < ornt_thres:
            if cur_a_pos_diff < pos_thres:
                succ_nn += 1
            tot_nn += 1
        print(f"succ_nn/tot_nn: {succ_nn}/{tot_nn}")
    


def inspect_obj_type_to_optimized_res(opt_res_fn):
    opt_res = np.load(opt_res_fn, allow_pickle=True ).item()
    print(opt_res.keys())    
    for key in opt_res:
        val = opt_res[key]
        print(f"key: {key}, val: {val}")


def inpsect_data_optimized_res_w_obj_codes(data_opt_res_fn):
    data_opt_res = np.load(data_opt_res_fn, allow_pickle=True).item()
    print(data_opt_res.keys())


def inspect_sampled_res(sampled_res_fn):
    sampled_res = np.load(sampled_res_fn, allow_pickle=True).item()
    print(sampled_res.keys())
    env_obj_codes = sampled_res['env_object_codes']
    print(env_obj_codes)



# 

# ('ori_grab_s8_stapler_pass_1', 'ori_grab_s8_stapler_pass_1') 
def inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    tot_keys = inst_tag_to_opt_res.keys()
    tot_keys = list(tot_keys) # 
    print(len(tot_keys))
    first_key = tot_keys[0]
    print(f"first key: {first_key}")


#
# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy
def inspect_child_task_to_fa_task(ch_to_fa_fn):
    # ? #
    ch_to_fa_dict = np.load(ch_to_fa_fn, allow_pickle=True).item()
    tot_ch_to_fa_task_nn = len(ch_to_fa_dict)
    print(f"tot_ch_to_fa_task_nn: {tot_ch_to_fa_task_nn}")
    for ch_task in ch_to_fa_dict:
        fa_task = ch_to_fa_dict[ch_task]
        print(f"ch_task: {ch_task}, fa_task: {fa_task}")
        



# <<<<<<< HEAD
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

def move_tested_samples(samples_root, target_folder):
    tot_samples = os.listdir(samples_root)
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_generalist_x/tracking_ori_grab_s9_airplane_fly_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-22-00-06
    tracking_st_tag = "tracking_ori_grab_"
    tot_samples = [
        fn for fn in tot_samples if fn[: len(tracking_st_tag)] == tracking_st_tag
    ]
    
    tot_eval_fns_w_time = []
    for cur_fn in tot_samples:
        cur_time_secs = parse_time_from_tracking_folder_nm(cur_fn)
        tot_eval_fns_w_time.append((cur_fn, cur_time_secs))
        
    sorted_eval_fns = sorted(tot_eval_fns_w_time, key=lambda x: x[1])
    # print(sorted_eval_fns)
    
    # parse the 
    for idx in range(0, len(sorted_eval_fns) - 9):
        cur_fn = sorted_eval_fns[idx][0] # sorted evals fns #
        
        # cur_exp_obj_name = parse_obj_type_from_taco_grab_folder_fn(cur_fn)
        
        # if exclude_existing_data_inst_to_opt_res is not None:
        #     if cur_exp_obj_name in exclude_existing_data_inst_tags:
        #         continue
        
        # get full fn from the samples root and cur_fn #
        cur_full_fn = os.path.join(samples_root, cur_fn)
        
        
        print(f"mv {cur_full_fn} {target_folder}/")
        os.system(f"mv {cur_full_fn} {target_folder}/")
    
        

# =======

def inspect_child_task_to_fa_task_list(ch_to_fa_fn):
    # ? #
    ch_to_fa_dict = np.load(ch_to_fa_fn, allow_pickle=True).item()
    tot_ch_to_fa_task_nn = len(ch_to_fa_dict)
    print(f"tot_ch_to_fa_task_nn: {tot_ch_to_fa_task_nn}")
    tot_info_nn = 0
    for ch_task in ch_to_fa_dict:
        cur_fa_info_list = ch_to_fa_dict[ch_task]
        tot_info_nn += len(cur_fa_info_list)
    print(f"tot_info_nn: {tot_info_nn}")
    # for ch_task in ch_to_fa_dict:
    #     fa_task = ch_to_fa_dict[ch_task]
    #     print(f"ch_task: {ch_task}, fa_task: {fa_task}")


def inspect_saved_samples(saved_samples_fn):
    saved_samples = np.load(saved_samples_fn, allow_pickle=True ).item()
    print(saved_samples.keys())
    
    data_nm = saved_samples['data_nm']
    print(data_nm)


def save_obj_type_to_general_imit_policy_weights():
    policy_weights_fn = "./runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-00-51-23/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    obj_type_to_policy_weights = {
        'apple': policy_weights_fn
    }
    policy_folder = "/".join(policy_weights_fn.split("/")[:-1])
    obj_type_to_policy_weights_fn = os.path.join(policy_folder, "obj_type_to_policy_weights.npy")
    np.save(obj_type_to_policy_weights_fn, obj_type_to_policy_weights)
    print(f"obj_type_to_policy_weights_fn: {obj_type_to_policy_weights_fn}"  )


def inspect_eval_dict(eval_dict_fn):
    eval_dict = np.load(eval_dict_fn, allow_pickle=True).item()
    tot_keys = list(eval_dict.keys())
    print(tot_keys[0])
    for subj_idx in range(0, 11):
        cur_subj_tag = f"_s{subj_idx}"
        tot_subj_nn = 0
        for key in eval_dict:
            if cur_subj_tag in key[0]:
                tot_subj_nn += 1
                
                # cur_val = eval_dict[key]
                # print(f"key: {key}, cur_val: {cur_val}")
        print(f"subj_tag: {cur_subj_tag}, tot_subj_nn: {tot_subj_nn}")


def inspect_eval_dict_grab_nf_300(eval_dict_fn):
    eval_dict = np.load(eval_dict_fn, allow_pickle=True).item()
    tot_keys = list(eval_dict.keys())
    print(tot_keys[0])


# and use 
def inspect_taco_eval_dict(taco_eval_dict_fn):
    taco_eval_dict = np.load(taco_eval_dict_fn, allow_pickle=True).item()
    tot_eval_nns = len(taco_eval_dict)
    print(f"tot_eval_nns: {tot_eval_nns}")
    eval_root  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval"
    tot_eval_folders = os.listdir(eval_root)
    st_tag = "tracking_TACO"
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230928_045_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-14-35-12
    tot_eval_folders = [
        fn for fn in tot_eval_folders if fn[: len(st_tag)] == st_tag
    ]
    print(len(tot_eval_folders))


def get_taco_eval_dict(taco_eval_dict_fn):
    taco_eval_dict = np.load(taco_eval_dict_fn, allow_pickle=True).item()
    print(len(taco_eval_dict))

# 
# larger than 50 and also not good performed? #
# not good performed res # ? 
def compare_diff_opt_res_ori_w_samples(obj_type_to_optimized_res_fn, opt_res_fn_ori, opt_res_fn_samples, rew_threshold=50.0):
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item()
    opt_res_ori = np.load(opt_res_fn_ori, allow_pickle=True).item()
    opt_res_samples = np.load(opt_res_fn_samples, allow_pickle=True).item()
    better_nn = 0
    tot_nn = 0
    
    for key in obj_type_to_optimized_res:
        val = obj_type_to_optimized_res[key]
        key_rew = val[0]
        if key_rew < rew_threshold:
            key_tuple = (key, key)
            if key_tuple in opt_res_ori and key_tuple in opt_res_samples:
                res_ori = opt_res_ori[key_tuple]
                res_samples = opt_res_samples[key_tuple]
                res_ori_pos_diff = res_ori['obj_pos_diff'][0].item()
                res_samples_pos_diff = res_samples['obj_pos_diff'][0].item()
                if res_ori_pos_diff  > res_samples_pos_diff:
                    print(f"key: {key}, res_ori_pos_diff: {res_ori_pos_diff}, res_samples_pos_diff: {res_samples_pos_diff}")
                    better_nn += 1
                tot_nn += 1
    print(f"better_nn/tot_nn: {better_nn}/{tot_nn}")


def select_samples_fr_ori_cur_optimized_res(obj_type_to_optimized_res_fn, opt_res_fn_samples_fn):
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item()
    opt_res_fn_samples = np.load(opt_res_fn_samples_fn, allow_pickle=True).item()
    minn_thres = 50.0
    obj_type_to_opt_pos_diff = {}
    for key in opt_res_fn_samples:
        obj_type = key[0]
        obj_pos = opt_res_fn_samples[key]['obj_pos_diff'][0].item()
        obj_type_to_opt_pos_diff[obj_type] = obj_pos
    sorted_obj_type_pos_diff_items = sorted(obj_type_to_opt_pos_diff.items(), key=lambda x: -x[1])
    selected_obj_types = []
    # selected_obj_types 
    for i_obj_type in range(len(sorted_obj_type_pos_diff_items)):
        cur_obj_type, cur_obj_pos_diff = sorted_obj_type_pos_diff_items[i_obj_type]
        cur_obj_rew =   obj_type_to_optimized_res[cur_obj_type][0]
        if cur_obj_rew > minn_thres:
            selected_obj_types.append(cur_obj_type)
    print(len(selected_obj_types))
    selected_obj_types_idxes = {
        obj_type: i_obj_type for i_obj_type, obj_type in enumerate(selected_obj_types)
    }
    samples_root_folder = "/".join(opt_res_fn_samples_fn.split("/")[: -1])
    selected_obj_types_idxes_sv_fn = os.path.join(samples_root_folder, "selected_obj_types_idxes.npy")
    np.save(selected_obj_types_idxes_sv_fn, selected_obj_types_idxes)
    print(f"selected_obj_types_idxes_sv_fn: {selected_obj_types_idxes_sv_fn}")
    # 


def inspect_samples(opt_res_fn_samples):
    opt_res_samples = np.load(opt_res_fn_samples, allow_pickle=True).item()
    opt_obj_key_w_pos_diff = {}
    for key in opt_res_samples:
        obj_type = key[0]
        opt_pos_diff = opt_res_samples[key]['obj_pos_diff'][0].item()
        opt_obj_key_w_pos_diff[obj_type] = opt_pos_diff
    sorted_opt_obj_key_w_pos_diff = sorted(opt_obj_key_w_pos_diff.items(), key=lambda x: x[1])
    print(sorted_opt_obj_key_w_pos_diff[:10])
    

def inspect_obj_type_to_optimized_res(obj_type_to_optimized_res_fn):
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item()
    print(f"obj_type_to_optimized_res: {len(obj_type_to_optimized_res)}")
    tot_nn = 0
    above_thres_nn = 0
    for key in obj_type_to_optimized_res:
        cur_optimized_res = obj_type_to_optimized_res[key]
        rew = cur_optimized_res[0]
        if rew > 50:
            above_thres_nn += 1
        tot_nn += 1
    print(f"above_thres_nn/tot_nn: {above_thres_nn}/{tot_nn}")



def get_taco_meshes_fn():
    folder_fn ="/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    tot_taco_mesh_fns = os.listdir(folder_fn)
    tot_taco_mesh_fns = [
        fn for fn in tot_taco_mesh_fns if fn[: len("taco")] == "taco" and "_modifed" not in fn
    ]
    return tot_taco_mesh_fns
    

def inspect_taco_mesh_folders():
    folder_fn ="/root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem"
    tot_taco_mesh_fns = os.listdir(folder_fn)
    tot_taco_mesh_fns = [
        fn for fn in tot_taco_mesh_fns if fn[: len("taco")] == "taco" and "_modifed" not in fn
    ]
    print(len(tot_taco_mesh_fns))
    print(f"tot_taco_mesh_fns: {tot_taco_mesh_fns[:10]}")
    day_flag_to_nn = {}
    for cur_taco_fn in tot_taco_mesh_fns:
        cur_taco_segs = cur_taco_fn.split("_")
        day_flag = cur_taco_segs[1]
        if day_flag not in day_flag_to_nn:
            day_flag_to_nn[day_flag] = 1
        else:
            day_flag_to_nn[day_flag] += 1
    print(day_flag_to_nn)


def inspect_taco_optimized_res(opt_res_fn):
    opt_res=np.load(opt_res_fn, allow_pickle=True).item()
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy
    for key in opt_res:
        key_val = opt_res[key]
        if "/cephfs/xueyi/data/TACO_Tracking_PK/" not in key_val[0]:
            print(f"key: {key}, key_val: {key_val}")
    print(f"len(opt_res): {len(opt_res)}")
    return opt_res

def construct_all_taco_opt_to_optimized_res():
    
    opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_beta_v2.npy
    # opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta_v2.npy"
    real_opt_res = inspect_taco_optimized_res(opt_res_fn)
    # real opt res # 
    tot_opt_res = {}
    tot_taco_inst_tags = get_taco_meshes_fn()
    for cur_taco_inst_tag in tot_taco_inst_tags:
        cur_taco_inst_tag_tuple = (cur_taco_inst_tag, "ori_grab_s2_phone_call_1")
        if cur_taco_inst_tag_tuple in real_opt_res:
            tot_opt_res[cur_taco_inst_tag_tuple] = real_opt_res[cur_taco_inst_tag_tuple]
        else:
            # 
            kine_fn = f"/cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_{cur_taco_inst_tag}_v2_interpfr_60_interpfr2_60_nntrans_40.npy"
            print(f"cur_taco_inst_tag: {cur_taco_inst_tag}")
            tot_opt_res[cur_taco_inst_tag_tuple] = [ kine_fn ]
    tot_opt_res_sv_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_tot.npy"
    # opt_res_fn = 
    np.save(tot_opt_res_sv_fn, tot_opt_res) # tot opt res #
    print(f"tot_opt_res_sv_fn: {tot_opt_res_sv_fn}")
    print(f"len(tot_opt_res): {len(real_opt_res)}")
    # print()



def inspect_obj_type_to_latent_folders(obj_type_to_latent_folders_fn):
    obj_type_to_latent_folders = np.load(obj_type_to_latent_folders_fn, allow_pickle=True).item()
    tot_nn = 0
    for obj_type in obj_type_to_latent_folders:
        
        if 'taco' in obj_type:
            tot_nn += 1
    print(tot_nn)


def filter_inst_tag_to_all_res():
    tot_opt_res_sv_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_tot.npy"
    opt_res = np.load(tot_opt_res_sv_fn, allow_pickle=True).item()
    
    obj_type_to_latent_features = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
    obj_type_to_latent_features = np.load(obj_type_to_latent_features, allow_pickle=True).item()
    filtered_inst_tag_to_opt_res = {}
    for key in opt_res:
        if key[0] in obj_type_to_latent_features:
            filtered_inst_tag_to_opt_res[key] = opt_res[key]
    print(len(filtered_inst_tag_to_opt_res))
    filtered_inst_tag_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_tot_filtered.npy"
    np.save(filtered_inst_tag_to_opt_res_fn, filtered_inst_tag_to_opt_res)
    print(f"filtered_inst_tag_to_opt_res_fn: {filtered_inst_tag_to_opt_res_fn}")
    
    
def inspect_inst_tag_to_opt_res():
    inst_tag_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_beta_v2.npy"
    # inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn)
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    date_to_nn = {}
    for key in  inst_tag_to_opt_res:
        grab_key = key[0]
        grab_key_segs = grab_key.split("_")
        grab_key_date = grab_key_segs[1]
        if grab_key_date not in date_to_nn:
            date_to_nn[grab_key_date] = 1
        else:
            date_to_nn[grab_key_date] += 1
    print(date_to_nn)
        
        
        
def reduce_kine_infos(grab_tracking_folder):
    # /cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s1_airplane_offhand_1_nf_300.npy
    tot_fns = os.listdir(grab_tracking_folder)
    # tot_fns = [
    #     fn for fn in tot_fns if '_nf_300' in fn
    # ]
    # tot_fns = [
    #     fn for fn in tot_fns if '_nf_300' not in fn
    # ]
    
    tot_fns = [
        fn for fn in tot_fns if '_nf_300' in fn and '_s2_' in fn
    ]
    reduced_keys = [
        'object_transl', 'object_rot_quat', 'robot_delta_states_weights_np'
    ]
    # reudced_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK"
    # reduced_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    # reduced_folder = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced/data"
    # reduced_folder = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK_Reduced_300/data"
    reduced_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_Reduced/data"
    for fn in tot_fns:
        cur_full_fn = os.path.join(grab_tracking_folder, fn)
        data_dict = np.load(cur_full_fn, allow_pickle=True).item()
        print(data_dict.keys())
        reduced_data_dict = {
            key: data_dict[key] for key in reduced_keys
        }
        reduced_fn = os.path.join(reduced_folder, fn)
        np.save(reduced_fn, reduced_data_dict)
        print(f"reduced_fn: {reduced_fn}")
        



def calculate_evaluation_metrics(data_optimized_res_nn, data_inst_tag, index=None):
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    
    # if '_nf_300' in data_inst_tag:
    #     kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK"
    
    kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
    
    if data_inst_tag.endswith(".npy"):
        cur_inst_kine_data_fn = data_inst_tag
    else:
        cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
        cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
    
    # passive #
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
    # object_pose #
    tot_ts = list(optimized_res.keys())
    tot_ts = [ cur_ts for cur_ts in tot_ts if isinstance(cur_ts, int) ]
    tot_ts = sorted(tot_ts)
    for ts in tot_ts: # 
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
    # top # radius # 
    new_optimized_info = {
        'optimized_obj_pose': tot_optimized_obj_pose[sorted_envs_idxes],
        'optimized_hand_qs': tot_optimized_hand_qs[sorted_envs_idxes],
        'optimized_hand_qtars': tot_optimized_hand_qtars[sorted_envs_idxes],
        'obj_pose_diff': tot_env_weighted_obj_pose_diff[sorted_envs_idxes],
        'obj_pos_diff': tot_env_diff_obj_pos[sorted_envs_idxes],
        'obj_rot_diff': tot_env_diff_obj_rot[sorted_envs_idxes],
    }
    return new_optimized_info
 
 
 
 
def calculate_metrics(data_inst_tag, best_eval_info_fn):
    best_eval_info = np.load(best_eval_info_fn , allow_pickle=True).item()
    # print(best_eval_info.keys()) # 
    
    # dict_keys(['optimized_obj_pose', 'optimized_hand_qtars', 'optimized_hand_qs', 'obj_pose_diff', 'best_obj_pose_diff', 'best_obj_pos_diff', 'best_obj_rot_diff'])
    optimized_hand_qs = best_eval_info['optimized_hand_qs'][0]
    best_obj_pos_diff = best_eval_info['best_obj_pos_diff']
    best_obj_rot_diff = best_eval_info['best_obj_rot_diff']

    #  optimized_res = np.load(dataz  _optimized_res_nn, allow_pickle=True).item()
    
    if '_nf_300' in data_inst_tag:
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    else:
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
     
    # if data_inst_tag.endswith(".npy"): # data inst tag #
    #     cur_inst_kine_data_fn = data_inst_tag
    # else:
    cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
    cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
    
    # passive #
    save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        
        
    goal_hand_qs = save_info['robot_delta_states_weights_np'][ : ]
    # hand_qs = hand_qs[: , : ]
    
    goal_obj_trans = save_info['object_transl']
    goal_obj_rot_quat = save_info['object_rot_quat']

    # goal_obj_pose = np.concatenate(
    #     [goal_obj_trans, goal_obj_rot_quat],  axis=-1 # axis -- is the axis #
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

    # print(best_obj_pos_diff, best_obj_rot_diff,  weighted_hand_diff_glb, diff_opt_jointangles, )

    eval_metrics = {
        'hand_glb_diff': weighted_hand_diff_glb.item(),
        'hand_joint_diff': diff_opt_jointangles.item(),
        'obj_pos_diff': best_obj_pos_diff[0],
        'obj_rot_diff': best_obj_rot_diff[0]
    }
    return eval_metrics

#         cur_latent_folder = obj_type_to_latent_folders[obj_type]
#         print(f"obj_type: {obj_type}, cur_latent_folder: {cur_latent_folder}")
# 1104 
# 1104


def inspect_data_inst_tag_to_optimized_res_metrics(inst_tag_to_opt_res_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    tot_eval_metrics_dict = {}
    for key in inst_tag_to_opt_res:
        val = inst_tag_to_opt_res[key]
        # print(f"key: {key}, val: {val}")
        inst_tag = key[0]
        # print(f"inst_tag: {inst_tag}")
        # if '_s1_' not in inst_tag:
        #     continue
        eval_res_fn = val[0]
        best_eval_res_fn = eval_res_fn.replace(".npy", "_sorted.npy").replace(".npy", "_best.npy")
        # print(f"best_eval_res_fn: {best_eval_res_fn}")
        if not os.path.exists(best_eval_res_fn):
            continue
        cur_eval_metrics_dict = calculate_metrics(inst_tag, best_eval_res_fn)
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
        print(f"eval_key: {eval_key}, mean: {np.mean(tot_eval_vals)}, medium: {medium_val}")
        




def copy_optimized_infos(interested_keys, data_inst_to_optimized_res_fn, dst_folder=None, wfranka=False):
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
            # if '_s1_' not in key:
            #     continue
            if '_nf_300' in key:
                if wfranka:
                    kine_ref_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_W_Franka_v2/data/passive_active_info_{key}.npy"
                else:
                    kine_ref_fn = f"/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_{key}.npy"
            else:
                kine_ref_fn = f"/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data/passive_active_info_{key}.npy"
            # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_peel_1_nf_300.npy
            # kine_ref_fn = f"/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_{key}.npy"
            
        
        
        opt_res = data_inst_to_optimized_res[cur_key_tuple][0]
        
        opt_res_fn = opt_res.replace(".npy", "_sorted.npy").replace(".npy", "_best.npy")
        if not os.path.exists(opt_res_fn):
            opt_res_fn = opt_res.replace(".npy", "_sorted.npy").replace(".npy", "_best_vv.npy")
        
        if not os.path.exists(opt_res_fn):
            continue
        
        opt_folder = opt_res.split("/")[-2]
        cur_dst_folder = os.path.join(dst_folder, opt_folder)
        os.makedirs(cur_dst_folder, exist_ok=True)
        
        cp_cmd = f"cp {opt_res_fn} {cur_dst_folder}/"
        print(cp_cmd)
        os.system(cp_cmd)
        
        
        
        kine_sv_fn = os.path.join(cur_dst_folder, kine_ref_fn.split("/")[-1])
        cp_cmd =    f"cp {kine_ref_fn} {kine_sv_fn}"
        print(cp_cmd)
        os.system(cp_cmd)


def inspect_optimized_res_grab_long(inst_tag_to_optimized_res_fn):
    inst_tag_to_optimized_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()

    
    
def inpsect_data_inst_tag_to_best_optimized_res(inst_tag_to_best_opt_res):
    inst_tag_to_best_opt_res = np.load(inst_tag_to_best_opt_res, allow_pickle=True).item()
    print(len(inst_tag_to_best_opt_res))
    for key in inst_tag_to_best_opt_res:
        cur_opt_res_dict = inst_tag_to_best_opt_res[key]
        print(cur_opt_res_dict.keys())
        break
    
    data_inst_tag_to_opt_res = inst_tag_to_best_opt_res.items()
    data_inst_tag_to_opt_res = sorted(data_inst_tag_to_opt_res, key=lambda x: x[1]['obj_pos_diff'][0].item())
    print(data_inst_tag_to_opt_res[:10])
    
    # best optimized res #
    
def get_data_inst_tag_to_valid_optimized_res(inst_tag_to_best_opt_res, inst_tag_to_opt_res_fn, cur_root_path):
    inst_tag_to_best_opt_res = np.load(inst_tag_to_best_opt_res, allow_pickle=True).item()
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    
    for key in inst_tag_to_best_opt_res:
        cur_opt_res_dict = inst_tag_to_best_opt_res[key]
        print(cur_opt_res_dict.keys())
        break
    
    data_inst_tag_to_opt_res = inst_tag_to_best_opt_res.items()
    data_inst_tag_to_opt_res = sorted(data_inst_tag_to_opt_res, key=lambda x: x[1]['obj_pos_diff'][0].item())
    print(data_inst_tag_to_opt_res[:10])
    top_k = 50
    data_inst_tag_to_opt_res = data_inst_tag_to_opt_res[: top_k ] 
    topk_inst_tag_to_opt_res_fn = {}
    # get he topk instnaces results #
    for cur_item in data_inst_tag_to_opt_res:
        cur_inst_tag = cur_item[0]
        tuple_key = cur_inst_tag #  (cur_inst_tag, cur_inst_tag)
        cur_inst_opt_res_fn = inst_tag_to_opt_res[tuple_key][0]
        # /root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo
        # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_
        ori_root_path = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_"
        # cur_root_path = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo"
        cur_cur_inst_opt_res_fn = cur_inst_opt_res_fn.replace(ori_root_path, cur_root_path)
        # cur cur inst opt res fn # 
        cur_cur_inst_opt_res_fn = cur_cur_inst_opt_res_fn.replace(".npy", "_sorted.npy").replace(".npy", "_best.npy")
        topk_inst_tag_to_opt_res_fn[cur_inst_tag[0]] = cur_cur_inst_opt_res_fn # 
    cur_statistics_sv_path = os.path.join(cur_root_path, 'statistics')
    cur_statistics_sv_nm = f"data_inst_tag_to_optimized_res_top{top_k}.npy"
    cur_statistics_sv_fn = os.path.join(cur_statistics_sv_path, cur_statistics_sv_nm)
    np.save(cur_statistics_sv_fn, topk_inst_tag_to_opt_res_fn)
    print(f"cur_statistics_sv_fn: {cur_statistics_sv_fn}")
    

def inspect_topk_best_optimized_res_fn(best_optimized_res_fn):
    best_optimized_res = best_optimized_res_fn #  '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/data_inst_tag_to_optimized_res_top50.npy'
    best_optimized_res = np.load(best_optimized_res, allow_pickle=True).item()
    for key in best_optimized_res:
        val = best_optimized_res[key]
        print(f"key: {key}, val: {val}")    

# top 50 # 
# def #
def combine_two_dict_fn(dict_fn_a, dict_fn_b):
    new_dict_to_fn = {}
    dict_a = np.load(dict_fn_a, allow_pickle=True).item()
    dict_b = np.load(dict_fn_b , allow_pickle=True).item()
    for ins_tag in dict_a:
        new_dict_to_fn[ins_tag] = dict_a[ins_tag]
    for ins_tag in dict_b:
        new_dict_to_fn[ins_tag] = dict_b[ins_tag]
    new_dict_to_fn_sv_fn = dict_fn_a.replace(".npy", "_cbd.npy")
    np.save(new_dict_to_fn_sv_fn, new_dict_to_fn)
    print(f"new_dict_to_fn saved to {new_dict_to_fn_sv_fn}")
    
    
def inspect_obj_type_to_optimized_res(dict_fn):
    obj_type_to_optimized_res = np.load(dict_fn, allow_pickle=True).item()
    for obj_type in obj_type_to_optimized_res:
        cur_opt_res = obj_type_to_optimized_res[obj_type]
        cur_opt_rew = cur_opt_res[0]
        print(f"obj_type: {obj_type}, cur_opt_rew: {cur_opt_rew}")  
        

# datanm_to_replay_fn_dict_fn #

# 
# succ_nn/tot_nn: 37/82 --- #
# 580 / 1113 # -- about half of the training set #
# so the question is can we get better results for trajectories using special sampling strategies? ## using all #
# 

# and the rewardis negative #
# seems that it is hard to accomplish the task? #
# python utils/grab_data_utils.py

# array[0].item() -- it should be the value in the dictionary #

# 5 cm -- 0.05
# 10 cm -- 0.1

# succ_nn/tot_nn: 57/84 -- origianl # succ_nn/tot_nn: 60/84 --- add the res from optimized together #
# succ_nn/tot_nn: 34/82 -- train together #
# succ_nn/tot_nn: 28/92 -- direct diffusion samples #



def inspect_demo_best_sv_fns(best_sv_res_fn):
    best_sv_res = np.load(best_sv_res_fn, allow_pickle=True).item()
    for key in best_sv_res:
        print(key )
        


def inspect_data_inst_tag_to_opt_res(inst_tag_to_opt_res_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    print(len(inst_tag_to_opt_res))


def inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    print(len(inst_tag_to_opt_res))
    for key in inst_tag_to_opt_res:
        val =inst_tag_to_opt_res[key]
        print(f"key: {key}, val: {val}")

def inspect_best_sv_res(best_sv_res_fn):
    best_sv_res = np.load(best_sv_res_fn, allow_pickle=True).item()
    print(f"best_sv_res: {best_sv_res.keys()}")
    for key in best_sv_res:
        print(f"key: {key}, val: {best_sv_res[key]}")
    
    
def inspect_obj_type_to_opt_res(obj_type_to_opt_res_fn):
    obj_type_to_opt_res = np.load(obj_type_to_opt_res_fn, allow_pickle=True).item()
    print(f"len: {len(obj_type_to_opt_res)}")


def inspect_optimized_res(opt_res_fn):
    opt_res = np.load(opt_res_fn, allow_pickle=True).item()
    print(opt_res.keys())


def inspect_best_optimized_res(best_opt_res_fn):
    opt_res = np.load(best_opt_res_fn, allow_pickle=True).item()
    for key in opt_res:
        val = opt_res[key]
        print(f"key: {key}, val: {val}")
    


def inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn):
    inst_tag_to_optimized_res = np.load(inst_tag_to_optimized_res_fn, allow_pickle=True).item()
    print(len(inst_tag_to_optimized_res))



def get_and_save_target_inst_tag_list_fn(subj_nm):
    # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_airplane_fly_1_nf_300.npy
    retar_info_st_tag = "passive_active_info_"
    pk_retarget_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    pk_retarget_fn_list = os.listdir(pk_retarget_sv_root)
    pk_retarget_fn_list = [ fn for fn in pk_retarget_fn_list if fn.endswith(".npy") and fn[: len(retar_info_st_tag)] == retar_info_st_tag ]
    pk_retarget_fn_names = [ fn.split('.')[0][len(retar_info_st_tag): ] for fn in pk_retarget_fn_list ]
    pk_retarget_fn_names = [ fn for fn in pk_retarget_fn_names if subj_nm + "_" in fn ]
    pk_retarget_fn_dict = {
        fn: 1 for fn in pk_retarget_fn_names
    }
    print(pk_retarget_fn_dict)
    return pk_retarget_fn_dict

def get_teacher_idx_to_teacher_weight():
    exp_sv_root = "/cephfs/xueyi/uni_manip/"
    exp_sv_fn = "isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s"
    cur_teacher_idx = 0
    teacher_idx_to_wegights = {}
    teacher_idx_to_inst_tags = {}
    for cur_subj_nm in range(2, 9):
        cur_teacher_inst_tag_fn = f"../assets/inst_tag_list_s{cur_subj_nm}.npy"
        
        cur_subj_exp_sv_fn = f"{exp_sv_fn}{cur_subj_nm}"
        full_subj_exp_sv_fn= os.path.join(exp_sv_root, cur_subj_exp_sv_fn)
        tot_exp_folders = os.listdir(full_subj_exp_sv_fn)
        folder_st_tag  = "tracking_"
        tot_exp_folders = [ fn for fn in tot_exp_folders if fn[: len(folder_st_tag)] == folder_st_tag]
        exp_folder = tot_exp_folders[0]
        full_exp_sv_fn = os.path.join(full_subj_exp_sv_fn, exp_folder)
        full_exp_sv_fn = os.path.join(full_exp_sv_fn, "nn")
        tot_exp_ckpts = os.listdir(full_exp_sv_fn)
        tot_exp_ckpts = [fn for fn in tot_exp_ckpts if fn.endswith(".pth") and fn[: len(folder_st_tag)] == folder_st_tag]
        cur_exp_ckpt = tot_exp_ckpts[0]
        full_exp_ckpt = os.path.join(full_exp_sv_fn, cur_exp_ckpt) 
        teacher_idx_to_wegights[cur_teacher_idx] = full_exp_ckpt
        
        teacher_idx_to_inst_tags[cur_teacher_idx] = cur_teacher_inst_tag_fn
        
        cur_teacher_idx = cur_teacher_idx + 1
        
    teacher_idx_to_wegights[cur_teacher_idx] = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_s10/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-04-17-19/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    teacher_idx_to_inst_tags[cur_teacher_idx] = "../assets/inst_tag_list_s10.npy"
        
    return teacher_idx_to_wegights, teacher_idx_to_inst_tags



# /data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s2_
def get_teacher_idx_to_teacher_weight_v2():
    exp_sv_root = "/data/xueyi/uni_manip/"
    exp_sv_fn = "isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s"
    cur_teacher_idx = 0
    teacher_idx_to_wegights = {}
    teacher_idx_to_inst_tags = {}
    for cur_subj_nm in range(2, 11):
        cur_teacher_inst_tag_fn = f"../assets/inst_tag_list_s{cur_subj_nm}.npy"
        
        cur_subj_exp_sv_fn = f"{exp_sv_fn}{cur_subj_nm}_"
        full_subj_exp_sv_fn= os.path.join(exp_sv_root, cur_subj_exp_sv_fn)
        tot_exp_folders = os.listdir(full_subj_exp_sv_fn)
        folder_st_tag  = "tracking_"
        tot_exp_folders = [ fn for fn in tot_exp_folders if fn[: len(folder_st_tag)] == folder_st_tag]
        exp_folder = tot_exp_folders[0]
        full_exp_sv_fn = os.path.join(full_subj_exp_sv_fn, exp_folder)
        full_exp_sv_fn = os.path.join(full_exp_sv_fn, "nn")
        tot_exp_ckpts = os.listdir(full_exp_sv_fn)
        tot_exp_ckpts = [fn for fn in tot_exp_ckpts if fn.endswith(".pth") and fn[: len(folder_st_tag)] == folder_st_tag]
        cur_exp_ckpt = tot_exp_ckpts[0]
        full_exp_ckpt = os.path.join(full_exp_sv_fn, cur_exp_ckpt) 
        teacher_idx_to_wegights[cur_teacher_idx] = full_exp_ckpt
        
        teacher_idx_to_inst_tags[cur_teacher_idx] = cur_teacher_inst_tag_fn
        print(f"cur_teacher_idx: {cur_teacher_idx}, cur_exp_ckpt: {full_exp_ckpt}")
        print(f"cur_teacher_idx: {cur_teacher_idx}, cur_teacher_inst_tag_fn: {cur_teacher_inst_tag_fn}")
        cur_teacher_idx = cur_teacher_idx + 1
        
    # teacher_idx_to_wegights[cur_teacher_idx] = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_s10/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-04-17-19/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    # teacher_idx_to_inst_tags[cur_teacher_idx] = "../assets/inst_tag_list_s10.npy"        
    return teacher_idx_to_wegights, teacher_idx_to_inst_tags

def get_teacher_idx_to_teacher_weight_v3():
    # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s9_wfuturews5freq5_
    exp_sv_root = "/data/xueyi/uni_manip/"
    # exp_sv_fn = f"isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s_wfuturews5freq5_"
    cur_teacher_idx = 0
    teacher_idx_to_wegights = {}
    teacher_idx_to_inst_tags = {}
    for cur_subj_nm in range(2, 10):
        cur_teacher_inst_tag_fn = f"../assets/inst_tag_list_s{cur_subj_nm}.npy"
        
        # cur_subj_exp_sv_fn = f"{exp_sv_fn}{cur_subj_nm}_"
        cur_subj_exp_sv_fn = f"isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s{cur_subj_nm}_wfuturews5freq5_"
        full_subj_exp_sv_fn= os.path.join(exp_sv_root, cur_subj_exp_sv_fn)
        tot_exp_folders = os.listdir(full_subj_exp_sv_fn)
        folder_st_tag  = "tracking_"
        tot_exp_folders = [ fn for fn in tot_exp_folders if fn[: len(folder_st_tag)] == folder_st_tag]
        exp_folder = tot_exp_folders[0]
        full_exp_sv_fn = os.path.join(full_subj_exp_sv_fn, exp_folder)
        full_exp_sv_fn = os.path.join(full_exp_sv_fn, "nn")
        tot_exp_ckpts = os.listdir(full_exp_sv_fn)
        tot_exp_ckpts = [fn for fn in tot_exp_ckpts if fn.endswith(".pth") and fn[: len(folder_st_tag)] == folder_st_tag]
        cur_exp_ckpt = tot_exp_ckpts[0]
        full_exp_ckpt = os.path.join(full_exp_sv_fn, cur_exp_ckpt) 
        teacher_idx_to_wegights[cur_teacher_idx] = full_exp_ckpt
        
        teacher_idx_to_inst_tags[cur_teacher_idx] = cur_teacher_inst_tag_fn
        print(f"cur_teacher_idx: {cur_teacher_idx}, cur_exp_ckpt: {full_exp_ckpt}")
        print(f"cur_teacher_idx: {cur_teacher_idx}, cur_teacher_inst_tag_fn: {cur_teacher_inst_tag_fn}")
        cur_teacher_idx = cur_teacher_idx + 1
              
    return teacher_idx_to_wegights, teacher_idx_to_inst_tags


def get_teacher_idx_to_teacher_weight_v4():
    # exp_sv_root = "/data/xueyi/uni_manip/"
    exp_sv_root = "/cephfs/xueyi/uni_manip"
    cur_teacher_idx = 0
    teacher_idx_to_wegights = {}
    teacher_idx_to_inst_tags = {}
    for cur_subj_nm in range(2, 11):
        cur_teacher_inst_tag_fn = f"../assets/inst_tag_list_s{cur_subj_nm}.npy"
        
        # cur_subj_exp_sv_fn = f"{exp_sv_fn}{cur_subj_nm}_"
        # cur_subj_exp_sv_fn = f"isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_winstfeat_s{cur_subj_nm}_wfuturews5freq5_"
        cur_subj_exp_sv_fn = f"isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s{cur_subj_nm}_v3goal"
        full_subj_exp_sv_fn= os.path.join(exp_sv_root, cur_subj_exp_sv_fn)
        tot_exp_folders = os.listdir(full_subj_exp_sv_fn)
        folder_st_tag  = "tracking_"
        tot_exp_folders = [ fn for fn in tot_exp_folders if fn[: len(folder_st_tag)] == folder_st_tag]
        exp_folder = tot_exp_folders[0]
        full_exp_sv_fn = os.path.join(full_subj_exp_sv_fn, exp_folder)
        full_exp_sv_fn = os.path.join(full_exp_sv_fn, "nn")
        tot_exp_ckpts = os.listdir(full_exp_sv_fn)
        tot_exp_ckpts = [fn for fn in tot_exp_ckpts if fn.endswith(".pth") and fn[: len(folder_st_tag)] == folder_st_tag]
        cur_exp_ckpt = tot_exp_ckpts[0]
        full_exp_ckpt = os.path.join(full_exp_sv_fn, cur_exp_ckpt) 
        teacher_idx_to_wegights[cur_teacher_idx] = full_exp_ckpt
        
        teacher_idx_to_inst_tags[cur_teacher_idx] = cur_teacher_inst_tag_fn
        print(f"cur_teacher_idx: {cur_teacher_idx}, cur_exp_ckpt: {full_exp_ckpt}")
        print(f"cur_teacher_idx: {cur_teacher_idx}, cur_teacher_inst_tag_fn: {cur_teacher_inst_tag_fn}")
        cur_teacher_idx = cur_teacher_idx + 1
              
    return teacher_idx_to_wegights, teacher_idx_to_inst_tags


def inspect_optimized_res_info(opt_res_fn):
    opt_res = np.load(opt_res_fn, allow_pickle=True).item()
    for key in opt_res:
        val = opt_res[key]
        print(f"key: {key}, val: {val}")
        break
    
# 

def find_topk_nearest_trajs(cur_idx, topk=10, traj_tracking_dir="/cephfs/yilaa/data/GRAB_Tracking/data", subj_idx=2):
    # traj_tracking_dir = "/cephfs/yilaa/data/GRAB_Tracking/data"
    if subj_idx == 2 or subj_idx < 1: # find the subj idx #
        grab_diff_arr_fn = f"grab_diff_arr.npy"
    else:
        grab_diff_arr_fn = f"grab_diff_arr_s{subj_idx}.npy"
    # grab_diff_arr_fn = "grab_diff_arr.npy"
    grab_diff_arr_fn = os.path.join(traj_tracking_dir, grab_diff_arr_fn) 
    grab_diff_arr = np.load(grab_diff_arr_fn) # grab_diff_arr:  nn_seq x nn_seq
    cur_seq_diff_arr = grab_diff_arr[cur_idx]
    cur_seq_sorted_neighbours = np.argsort(cur_seq_diff_arr, axis=0) # 
    cur_seq_sorted_neighbours = cur_seq_sorted_neighbours[1: 1 + topk]
    cur_seq_sorted_neighbours = cur_seq_sorted_neighbours.tolist()
    return cur_seq_sorted_neighbours
    ## TODO: load the idx to seq name array ###
    ## TODO: get curresponding object_name and traj_name ##
        # pass

def get_inst_tag_to_nearest_info(inst_tag_to_best_opt_res_fn, inst_tag_to_opt_res_fn, subj_idx, topk=10):
    inst_tag_to_best_opt_res = np.load(inst_tag_to_best_opt_res_fn, allow_pickle=True).item()
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    # /cephfs/xueyi/data/GRAB_Tracking_PK/data/grab_data_nm_idx_dict_s6.npy
    traj_tracking_dir = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # cur_idx 
    # grab_data_nm_idx_dict #
    grab_tracking_data_root = traj_tracking_dir
    if subj_idx == 2 or subj_idx < 1:
        data_nm_idx_dict_sv_fn = "grab_data_nm_idx_dict.npy"
    else:
        data_nm_idx_dict_sv_fn = f"grab_data_nm_idx_dict_s{subj_idx}.npy"
    data_nm_idx_dict_sv_fn = os.path.join(grab_tracking_data_root, data_nm_idx_dict_sv_fn)
    data_nm_idx_dict = np.load(data_nm_idx_dict_sv_fn, allow_pickle=True).item()
    data_nm_to_idx = data_nm_idx_dict['data_nm_to_idx'] # idx to data nm # 
    idx_to_data_nm = data_nm_idx_dict['idx_to_data_nm'] # data nm to idx # 
    
    inst_tag_to_fa_opt_res = {}
    
    for inst_tag_pair in inst_tag_to_opt_res:
        cur_inst_opt_statistics = inst_tag_to_best_opt_res[inst_tag_pair]
        cur_inst_obj_pos_diff = cur_inst_opt_statistics['obj_pos_diff'][0].item()
        
        # if cur_inst_obj_pos_diff < 0.08:
        #     continue
        
        cur_grab_data_tag = inst_tag_pair[0]
        
        # cur_grab_data_tag = cur_tracking_data.split(".")[0][len(passive_active_info_tag):]
        # traj_grab_data_tag = cur_grab_data_tag
        
        if '_nf_' in cur_grab_data_tag:
            
            pure_obj_type = cur_grab_data_tag.split('_nf_')[0] #
            nf_tag = cur_grab_data_tag.split('_nf_')[1]
        else:
            pure_obj_type = cur_grab_data_tag
            nf_tag = None
        cur_idx = data_nm_to_idx[pure_obj_type]
        
        ## subject idx ##
        cur_seq_sorted_neighbours = find_topk_nearest_trajs(cur_idx, topk=topk, traj_tracking_dir=grab_tracking_data_root, subj_idx=subj_idx)
        
        
        cur_inst_fa_to_opt_res = {}
        ### i_traj and the traj_idx ###
        for i_traj, traj_idx in enumerate(cur_seq_sorted_neighbours):
            cur_obj_name = idx_to_data_nm[traj_idx] # index to data name #
            cur_traj_name = cur_obj_name
            if nf_tag is not None:
                cur_traj_name_w_nf = "_nf_".join([cur_traj_name, nf_tag])
            cur_traj_inst_tag = (cur_traj_name_w_nf, cur_traj_name_w_nf)
            if cur_traj_inst_tag not in inst_tag_to_opt_res:
                continue
            cur_traj_inst_opt_res = inst_tag_to_opt_res[cur_traj_inst_tag][0]
            # get the cur inst opt res #
            cur_inst_fa_to_opt_res[cur_traj_name_w_nf] = cur_traj_inst_opt_res
        inst_tag_to_fa_opt_res[cur_grab_data_tag] = cur_inst_fa_to_opt_res
    # inst_tag_to_fa_opt_res #
    return inst_tag_to_fa_opt_res
    
    # if isinstance(list(obj_type_to_optimized_traj.keys())[0], tuple):
    #     if (cur_obj_name, cur_traj_name) in obj_type_to_optimized_traj:
    #         cur_obj_optimized_fn = obj_type_to_optimized_traj[(cur_obj_name, cur_traj_name)][0] ### get the optimized obj type and the traj name ###
    #         ### cur obj optimized fn ###
    #         # cur_obj_optimized_fn
    #     else:
    #         continue
    # else:
    #     if cur_obj_name in obj_type_to_optimized_traj:
    #         cur_obj_optimized_fn = obj_type_to_optimized_traj[cur_obj_name] # get the optimized traj # 
    #     else:
    #         continue


def inspect_inst_tag_to_fa_to_opt_res(inst_tag_to_fa_to_opt_res_fn):
    inst_tag_to_fa_to_opt_res = np.load(inst_tag_to_fa_to_opt_res_fn, allow_pickle=True).item()
    for key in inst_tag_to_fa_to_opt_res:
        cur_fa_to_opt_res = inst_tag_to_fa_to_opt_res[key]
        print(f"key: {key}, cur_fa_to_opt_res: {cur_fa_to_opt_res}")



def inspect_inst_tag_to_weights_fn(inst_tag_to_weights_fn):
    inst_tag_to_weights = np.load(inst_tag_to_weights_fn, allow_pickle=True).item()
    for key in inst_tag_to_weights:
        val = inst_tag_to_weights[key]
        print(f"inst: {key}, weight_fn: {val}")



### with the best optimized info res ---  ji# 



def get_good_res_from_optimized_res_all(inst_tag_to_optimized_res_all_fn, pos_thres=0.08, rot_thres=0.6981317007977318):
    inst_tag_to_optimized_res_all = np.load(inst_tag_to_optimized_res_all_fn, allow_pickle=True).item()
    inst_tag_to_optimized_res = {}
    for inst_tag in inst_tag_to_optimized_res_all:
        cur_opt_res = inst_tag_to_optimized_res_all[inst_tag]
        cur_obj_pos_diff = cur_opt_res['obj_pos_diff'][0].item()
        cur_obj_rot_diff = cur_opt_res['obj_rot_diff'][0].item()
        if cur_obj_pos_diff < pos_thres and cur_obj_rot_diff < rot_thres:
            inst_tag_to_optimized_res[inst_tag] = cur_opt_res
    return inst_tag_to_optimized_res



# /root/diffsim/IsaacGymEnvs2/assets/good_inst_opt_res.npy
def inspect_inst_opt_res(inst_opt_res_fn):
    inst_opt_res = np.load(inst_opt_res_fn, allow_pickle=True).item()
    for key in inst_opt_res:
        print(key)



# /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1/statistics/obj_type_to_optimized_res.npy
def inspect_obj_type_to_optimized_res(obj_type_to_opt_res_fn, allegro_obj_type_to_opt_res_fn):
    obj_type_to_opt_res = np.load(obj_type_to_opt_res_fn, allow_pickle=True).item()
    allegro_obj_type_to_opt_res = np.load(allegro_obj_type_to_opt_res_fn, allow_pickle=True).item()
    tot_good_nn = 0
    rew_thres = 50.0
    tot_allegro_good_nn = 0
    tot_allegro_nn = 0
    for key in obj_type_to_opt_res:
        if key in allegro_obj_type_to_opt_res:
            cur_val = obj_type_to_opt_res[key]
            # print(f"key: {key}, val: {cur_val}")
            cur_rew, cur_opt_path = cur_val
            if cur_rew > rew_thres:
                tot_good_nn += 1
            # if key in allegro_obj_type_to_opt_res:
            tot_allegro_nn += 1
            cur_allegro_val = allegro_obj_type_to_opt_res[key]
            cur_allegro_rew, cur_allegro_opt_path = cur_allegro_val
            if cur_allegro_rew > rew_thres:
                tot_allegro_good_nn += 1
            print(f"key: {key}, cur_rew: {cur_rew}, cur_allegro_rew: {cur_allegro_rew}")
    print(f"tot_good_nn/tot_nn: {tot_good_nn}/{len(obj_type_to_opt_res)}, tot_allegro_good_nn/tot_nn: {tot_allegro_good_nn}/{tot_allegro_nn}")


def inspect_obj_type_to_ckpt_fn(obj_type_to_ckpt_fn):
    obj_type_to_ckpt = np.load(obj_type_to_ckpt_fn, allow_pickle=True).item()
    for key in obj_type_to_ckpt:
        val = obj_type_to_ckpt[key]
        print(f"key: {key}, val: {val}")


def inspect_teacher_idx_to_model_weights(teacher_idx_to_model_weights_fn):
    teacher_idx_to_model_weights = np.load(teacher_idx_to_model_weights_fn, allow_pickle=True).item()
    for teacher_idx in teacher_idx_to_model_weights:
        model_weight = teacher_idx_to_model_weights[teacher_idx]
        print(f"teacher_idx: {teacher_idx}, model_weight: {model_weight}")
        
    # teacher_idx_to_model_weights[8] = "../assets/inst_tag_list_s9.npy"
    # np.save(teacher_idx_to_model_weights_fn, teacher_idx_to_model_weights)
    # print(f"saved to {teacher_idx_to_model_weights_fn}")
    # teacher_idx_to_model_weights[8] = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_teacherfrom_s9/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-19-58-19/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    # np.save(teacher_idx_to_model_weights_fn, teacher_idx_to_model_weights)
    # print(f"saved to {teacher_idx_to_model_weights_fn}")

def inspect_obj_type_to_optimized_res(obj_type_to_opt_res_fn):
    obj_type_to_opt_res = np.load(obj_type_to_opt_res_fn, allow_pickle=True).item()
    print(len(obj_type_to_opt_res))
    thres = 50.0
    tot_nn = 0
    succ_nn = 0
    for key in obj_type_to_opt_res:
        cur_val = obj_type_to_opt_res[key]
        cur_rew, cur_opt_path = cur_val
        if cur_rew > thres:
            succ_nn += 1
        tot_nn += 1
    ratio = succ_nn / tot_nn
    print(f"tot_nn: {tot_nn}, succ_nn: {succ_nn}, ratio: {ratio}")
    
def get_obj_type_to_kine_fn():
    # /cephfs/xueyi/data/TACO_Tracking_PK/data/passive_active_info_ori_grab_s2_phone_call_1_interped_taco_20231005_055_v2_interpfr_60_interpfr2_60_nntrans_40.npy
    grab_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    taco_root = "/cephfs/xueyi/data/TACO_Tracking_PK/data"
    #### get the grab obj type to the trackng data sv fn #####
    # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_binoculars_lift_nf_300.npy
    grab_tracking_st_tag = "passive_active_info_"
    grab_tracking_fns = os.listdir(grab_root)
    grab_tracking_fns = [
        fn for fn in grab_tracking_fns if fn.endswith(".npy") and fn[: len(grab_tracking_st_tag)] == grab_tracking_st_tag
    ]
    inst_tag_to_tracking_fn = {}
    for fn in grab_tracking_fns:
        cur_full_fn = os.path.join(grab_root, fn)
        grab_inst_tag = fn.split(".npy")[0][len(grab_tracking_st_tag): ]
        inst_tag_to_tracking_fn[grab_inst_tag] = cur_full_fn
    
    taco_tracking_fns = os.listdir(taco_root)
    taco_tracking_st_tag = "passive_active_info_ori_grab_s2_phone_call_1_interped_"
    taco_tracking_ed_tag = "_v2_interpfr_60_interpfr2_60_nntrans_40.npy"
    taco_tracking_fns = [
        fn for fn in taco_tracking_fns if fn.endswith(taco_tracking_ed_tag) and fn[: len(taco_tracking_st_tag)] == taco_tracking_st_tag
    ]
    
    for fn in taco_tracking_fns:
        cur_full_fn = os.path.join(taco_root, fn)
        taco_inst_tag = fn[len(taco_tracking_st_tag): - len(taco_tracking_ed_tag)] 
        inst_tag_to_tracking_fn[taco_inst_tag] = cur_full_fn
    obj_type_to_kinematics_traj_dict_fn = "../assets/obj_type_to_kinematics_traj_dict.npy"
    np.save(obj_type_to_kinematics_traj_dict_fn, inst_tag_to_tracking_fn)
    print(f"inst tag to kinematics traj dict saved to {obj_type_to_kinematics_traj_dict_fn}")
    # whether to sue the canonicalized object shape #
    # whether to load the kinematics traj dict #
    
    
def merge_two_obj_type_to_optimized_res(dict_fn_1, dict_fn_2, sv_dict_fn):
    dict_1 = np.load(dict_fn_1, allow_pickle=True).item()
    dict_2 = np.load(dict_fn_2, allow_pickle=True).item()
    new_dict = {}
    for key in dict_1:
        new_dict[key] = dict_1[key]
    for key in dict_2:
        new_dict[key] = dict_2[key]
    print(len(new_dict), len(dict_1), len(dict_2))
    
    np.save(sv_dict_fn, new_dict)
    print(f"new dict saved to {sv_dict_fn}")

def test_contact_data(contact_data_fn):
    contact_data = np.load(contact_data_fn)
    print(contact_data.shape)


def inspect_inst_tag_to_best_opt_res(inst_tag_to_best_opt_res_fn):
    inst_tag_to_best_opt_res = np.load(inst_tag_to_best_opt_res_fn, allow_pickle=True).item()
    for key in inst_tag_to_best_opt_res:
        val = inst_tag_to_best_opt_res[key]
        print(f"key: {key}, val: {val}")
        break


def merge_inst_tag_to_opt_res_dict():
    tot_inst_tag_to_opt_res_dict = {}
    # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res.npy
    data_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests"
    for subj_idx in range(2, 8):
        cur_data_root = data_root + f"{subj_idx}_"
        cur_statistics_fn = os.path.join(cur_data_root, "statistics", "data_inst_tag_to_optimized_res.npy")
        cur_statistics = np.load(cur_statistics_fn, allow_pickle=True).item()
        tot_inst_tag_to_opt_res_dict.update(cur_statistics)
    merged_inst_tag_to_opt_res_dict_sv_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res_merged.npy"
    np.save(merged_inst_tag_to_opt_res_dict_sv_fn, tot_inst_tag_to_opt_res_dict)
    print(f"merged inst tag to opt res dict saved to {merged_inst_tag_to_opt_res_dict_sv_fn}")


def calculate_avg_opt_res(inst_tag_to_opt_res_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    key_to_tot_res = {}
    for cur_key in inst_tag_to_opt_res:
        cur_val = inst_tag_to_opt_res[cur_key]
        for cur_val_key in cur_val:
            if cur_val_key not in key_to_tot_res:
                key_to_tot_res[cur_val_key] = [cur_val[cur_val_key][0].item()]
            else:
                key_to_tot_res[cur_val_key].append(cur_val[cur_val_key][0].item())
                
        # try:
        #     print(f"key: {cur_key}, val: {cur_val}")
        # except:
        #     pass
    for key in key_to_tot_res:
        cur_tot_res = key_to_tot_res[key]
        cur_avg_res = np.mean(cur_tot_res)
        cur_medium_res= np.median(cur_tot_res)
        print(f"key: {key}, cur_avg_res: {cur_avg_res}, cur_median_res: {cur_medium_res}")


def inspect_teacher_idx_to_weights(teacher_idx_to_weights_fn):
    teacher_idx_to_weights = np.load(teacher_idx_to_weights_fn, allow_pickle=True).item()
    for key in teacher_idx_to_weights:
        val = teacher_idx_to_weights[key]
        print(f"key: {key}, val: {val}")


def scp_teacher_model_weights(teacher_idx_to_weights_fn):
    teacher_idx_to_weights = np.load(teacher_idx_to_weights_fn, allow_pickle=True).item()
    for key in teacher_idx_to_weights:
        cur_weight_fn = teacher_idx_to_weights[key]
        cur_weight_folder = "/".join(cur_weight_fn.split("/")[:-1])
        os.makedirs(cur_weight_folder, exist_ok=True)
        os.system(f"scp -P 31088 -r root@10.210.22.32:{cur_weight_fn} {cur_weight_folder}")


def test_model_weights_shape(model_weight_shape_fn):
    model_weights = torch.load(model_weight_shape_fn, map_location='cpu')
    print(f"model_weights: {model_weights.keys()}")
    model_weights = model_weights['model']
    for key in model_weights:
        cur_weight = model_weights[key]
        print(f"key: {key}, shape: {cur_weight.shape}")



def inspect_weight_keys_from_ckpt(model_weight_fn):
    model_weights = torch.load(model_weight_fn, map_location='cpu')
    print(model_weights.keys())
    print(f"env states:", model_weights['env_state'])



def inspect_subj_optimized_res(optimized_res_statistics_fn):
    optimized_res = np.load(optimized_res_statistics_fn, allow_pickle=True).item()
    inst_tag_to_best_opt_res = {}
    for key in optimized_res:
        val = optimized_res[key]
        # print(f"key: {key}, val: {val}")
        best_opt_res_fn = val[0].replace(".npy", "_sorted_best.npy")
        if not os.path.exists(best_opt_res_fn):
            continue
        best_opt_res = np.load(best_opt_res_fn, allow_pickle=True).item()
        # for res_key in best_opt_res:
        #     res_val = best_opt_res[res_key]
        #     print(f"res_key: {res_key}, res_val: {res_val}")
        inst_tag_to_best_opt_res[key] = best_opt_res['best_obj_pos_diff'][0].item()
        
        # break
    sorted_best_opt_res = sorted(inst_tag_to_best_opt_res.items(), key=lambda x: x[1])
    for key, val in sorted_best_opt_res:
        print(f"key: {key}, val: {val}")


def get_target_subj_inst_tags(retargeting_data_folder, subj_tag=None):
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_cup_drink_1_nf_300.npy
    p_a_st_tag = "passive_active_info_"
    retargeting_data_fns = os.listdir(retargeting_data_folder)
    retargeting_data_fns = [fn for fn in retargeting_data_fns if fn.endswith(".npy")]
    retargeting_data_fns = [fn.split("_nf_300")[0][len(p_a_st_tag): ] for fn in retargeting_data_fns]
    if subj_tag is not None:
        retargeting_data_fns = [fn for fn in retargeting_data_fns if subj_tag in fn]
    return retargeting_data_fns
    

def inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn):
    inst_tag_to_opt_res = np.load(inst_tag_to_opt_res_fn, allow_pickle=True).item()
    obj_type_to_opt_res = {}
    for key in inst_tag_to_opt_res:
        val = inst_tag_to_opt_res[key]
        print(f"key: {key}, val: {val}")
        
        inst_single_key = key[0]
        # "ori_grab_s2_piggybank_use_1_nf_300"
        try:
            inst_obj_type = inst_single_key.split("_")[3]
            obj_type_to_opt_res[inst_obj_type] = val
        except:
            pass
        
        # break
    return obj_type_to_opt_res
    

def get_retargeting_inst_tag_to_opt_res(obj_type_to_optimized_res_fn):
    obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True ).item()
    retargeted_data_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    subj_tag = "_s1_"
    retargetd_data_fns =get_target_subj_inst_tags(retargeted_data_folder, subj_tag)
    print(f"retargetd_data_fns: {retargetd_data_fns}")
    retarget_data_inst_tag_to_opt_res_fn = {}
    for inst_tag in retargetd_data_fns:
        cur_inst_tag_tuple =  (inst_tag + "_nf_300", inst_tag + "_nf_300")
        for obj_type in obj_type_to_optimized_res:
            if obj_type in inst_tag:
                retarget_data_inst_tag_to_opt_res_fn[cur_inst_tag_tuple] = obj_type_to_optimized_res[obj_type]
                break
    retarget_data_inst_tag_sv_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res_to_s1.npy"
    np.save(retarget_data_inst_tag_sv_fn, retarget_data_inst_tag_to_opt_res_fn)
    print(f"retarget data inst tag to opt res saved to {retarget_data_inst_tag_sv_fn}")
    

def test_obj_retargete_info(obj_retar_info_fn):
    obj_retar_info = np.load(obj_retar_info_fn, allow_pickle=True).item()
    print(obj_retar_info.keys())
    object_transl = obj_retar_info['object_transl']
    print(object_transl.shape)
    dt = 1/120
    obj_vel = object_transl[1:] - object_transl[:-1]
    obj_vel = obj_vel / dt
    
    diff_obj_vel = obj_vel[1:] - obj_vel[:-1]
    diff_obj_vel = diff_obj_vel/dt
    diff_obj_vel = np.sqrt(np.sum(diff_obj_vel ** 2, axis=-1)) # nn_frame #
    diff_obj_vel = np.mean(diff_obj_vel)
    print(diff_obj_vel)
    
    
    
    # retargeted_data_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    # p_a_st_tag = "passive_active_info_"
    # retargeting_data_fns = os.listdir(retargeted_data_folder)
    # retargeting_data_fns = [fn for fn in retargeting_data_fns if fn.endswith(".npy")]
    # # retargeting_data_fns = [fn.split("_nf_300")[0][len(p_a_st_tag): ] for fn in retargeting_data_fns]
    # subj_tag = '_s1_'
    # if subj_tag is not None:
    #     retargeting_data_fns = [fn for fn in retargeting_data_fns if subj_tag in fn]
        
    
    retargeted_data_folder = "/cephfs/xueyi/data/TACO_Tracking_PK/data"
    retargeting_data_fns = os.listdir(retargeted_data_folder)
    retargeting_data_fns = [fn for fn in retargeting_data_fns if fn.endswith(".npy") and "_v2_interpfr_60_interpfr2_60_nntrans_40" in fn]
    subj_tag = '_20231024_'
    if subj_tag is not None:
        retargeting_data_fns = [fn for fn in retargeting_data_fns if subj_tag in fn]
    dt = 1/60
    
    tot_obj_vels = []
    for fn in retargeting_data_fns:
        full_fn = os.path.join(retargeted_data_folder,  fn)
        cur_data = np.load(full_fn, allow_pickle=True).item()
        
        object_transl = cur_data['object_transl']
        # print(object_transl.shape)
        # dt = 1/120
        obj_vel = object_transl[1:] - object_transl[:-1]
        obj_vel = obj_vel / dt
        diff_obj_vel = obj_vel[1:] - obj_vel[:-1]
        diff_obj_vel = diff_obj_vel/dt
        diff_obj_vel = np.sqrt(np.sum(diff_obj_vel ** 2, axis=-1)) # nn_frame #
        diff_obj_vel = np.mean(diff_obj_vel)
        
        tot_obj_vels.append(diff_obj_vel)
    tot_obj_vels = np.array(tot_obj_vels)
    print(np.mean(tot_obj_vels))


def select_nearest_forecasting_res(forecasting_res_fn, inst_tag):
    ### 
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_cubesmall_inspect_1_nf_300.npy
    kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
    kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
    kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
    print(f"kine_traj_data: {kine_traj_data.keys()}")
    
    # robot_delta_states_weights_np # 
    joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
    # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    kine_hand_qs = kine_traj_data['robot_delta_states_weights_np']
    kine_obj_pos = kine_traj_data['object_transl']
    
    kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering] # nn_fraes x nn_hand_qs #
    
    
    forecast_res = np.load(forecasting_res_fn, allow_pickle=True).item()
    print(forecast_res.keys())
    tot_forecast_ts = list(forecast_res.keys())
    tot_forecast_ts_ints = []
    for cur_ts in tot_forecast_ts:
        try:
            cur_ts_int = int(cur_ts)
            tot_forecast_ts_ints.append(cur_ts_int)
        except:
            pass
    tot_forecast_ts = sorted(tot_forecast_ts_ints)
    tot_ts_forecast_hand_qs = []
    tot_ts_forecast_obj_pos = []
    for cur_ts in tot_forecast_ts:
        cur_ts_forecast_data = forecast_res[cur_ts]
        print(cur_ts_forecast_data.keys())
        cur_ts_forcast_hand_dof_pos = cur_ts_forecast_data['forcast_hand_dof_pos']
        cur_ts_forcast_obj_pos = cur_ts_forecast_data['forcast_obj_pos']
        print(f"cur_ts_forcast_hand_dof_pos: {cur_ts_forcast_hand_dof_pos.shape}, cur_ts_forcast_obj_pos: {cur_ts_forcast_obj_pos.shape}")
        tot_ts_forecast_hand_qs.append(cur_ts_forcast_hand_dof_pos)
        tot_ts_forecast_obj_pos.append(cur_ts_forcast_obj_pos)
        # break
    tot_ts_forecast_hand_qs = np.stack(tot_ts_forecast_hand_qs, axis=1)
    tot_ts_forecast_obj_pos =   np.stack(tot_ts_forecast_obj_pos, axis=1) # get hte forecast obj pos # 
    # ts forecast obj pos #
    ## nn_envs x n_fraemes x nn_hand_qs # 
    ## nn_envs x nn_frames x 3 # 
    diff_forecast_hand_qs = np.sum(
        (tot_ts_forecast_hand_qs - kine_hand_qs[None]) ** 2, axis=-1
    )
    diff_forecast_hand_qs = np.sqrt(diff_forecast_hand_qs) # nn_envs x nn_frames 
    diff_forecast_hand_qs = np.sum(diff_forecast_hand_qs, axis=1) # nn_ens
    diff_forecast_obj_pos = np.sum(
        (tot_ts_forecast_obj_pos - kine_obj_pos[None]) ** 2, axis=-1
    )
    diff_forecast_obj_pos = np.sqrt(diff_forecast_obj_pos) # nn_envs x nn_frames
    diff_forecast_obj_pos = np.sum(diff_forecast_obj_pos, axis=1) # nn_envs
    diff_forecast_diff = diff_forecast_hand_qs + diff_forecast_obj_pos # nn_envs # # nn_envs # 
    sorted_diff_forecasted_idxes = np.argsort(diff_forecast_diff, axis=0) #
    best_idxes = sorted_diff_forecasted_idxes[: 10] # 
    best_forecasted_hand_qs = tot_ts_forecast_hand_qs[best_idxes]
    best_forecasted_obj_pos = tot_ts_forecast_obj_pos[best_idxes] # 
    
    
    sv_dict = {
        'hand_qs': best_forecasted_hand_qs,
        'obj_pos': best_forecasted_obj_pos
    }
    sv_fn = "/data/xueyi/uni_manip/expanded_kines"
    os.makedirs(sv_fn , exist_ok=True)
    
    sv_fn = os.path.join(sv_fn, f"{inst_tag}_forecasted_kine_traj.npy")
    np.save(sv_fn, sv_dict)
    print(  f"saved to {sv_fn}" )
    
    
    
    
    return kine_traj_fn


def select_nearest_forecasting_res_other_insts(forecasting_res_fn, inst_tag):
    ### 
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_cubesmall_inspect_1_nf_300.npy
    kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
    kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
    kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
    print(f"kine_traj_data: {kine_traj_data.keys()}")
    
    # robot_delta_states_weights_np # 
    joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
    # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    kine_hand_qs = kine_traj_data['robot_delta_states_weights_np']
    kine_obj_pos = kine_traj_data['object_transl']
    
    kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering] # nn_fraes x nn_hand_qs #
    
    
    tot_kine_traj_data_fn = os.listdir(kine_traj_folder)
    tot_kine_traj_data_fn = [
        fn for fn in tot_kine_traj_data_fn if fn.endswith("_nf_300.npy") and "ori_grab" in fn and inst_tag not in fn
    ]
    
    
    neis = test_nearest_trajectories()
    tot_kine_traj_data_fn = [
        f"passive_active_info_{cur_nei}_nf_300.npy" for cur_nei in neis
    ]
    
    
    tot_kine_hand_qs = []
    tot_kine_obj_transl = []
    for cur_kien_traj_fn in tot_kine_traj_data_fn:
        cur_full_kine_traj_fn = os.path.join(kine_traj_folder, cur_kien_traj_fn)
        cur_kine_traj_data = np.load(cur_full_kine_traj_fn, allow_pickle=True).item()
        cur_kine_hand_qs = cur_kine_traj_data['robot_delta_states_weights_np']
        cur_kine_obj_pos = cur_kine_traj_data['object_transl']
        cur_kine_hand_qs = cur_kine_hand_qs[..., joint_idxes_inversed_ordering]
        tot_kine_hand_qs.append(cur_kine_hand_qs)
        tot_kine_obj_transl.append(cur_kine_obj_pos)
    tot_kine_hand_qs = np.stack(tot_kine_hand_qs, axis=0)
    tot_kine_obj_transl = np.stack(tot_kine_obj_transl, axis=0)
    
    
    
    sv_dict = {
        'hand_qs': tot_kine_hand_qs,
        'obj_pos': tot_kine_obj_transl
    }
    sv_fn = "/data/xueyi/uni_manip/expanded_kines"
    os.makedirs(sv_fn , exist_ok=True)
    
    sv_fn = os.path.join(sv_fn, f"{inst_tag}_forecasted_kine_traj_others.npy")
    np.save(sv_fn, sv_dict)
    print(  f"saved to {sv_fn}" )
    
    return kine_traj_fn



def select_nearest_forecasting_res_traj_modifications(forecasting_res_fn, inst_tag):
    ### 
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_cubesmall_inspect_1_nf_300.npy
    kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
    kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
    kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
    print(f"kine_traj_data: {kine_traj_data.keys()}")
    
    # robot_delta_states_weights_np # 
    joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
    # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    kine_hand_qs = kine_traj_data['robot_delta_states_weights_np']
    kine_obj_pos = kine_traj_data['object_transl']
    
    kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering] # nn_fraes x nn_hand_qs #
    
    
    # tot_kine_traj_data_fn = os.listdir(kine_traj_folder)
    # tot_kine_traj_data_fn = [
    #     fn for fn in tot_kine_traj_data_fn if fn.endswith("_nf_300.npy") and "ori_grab" in fn and inst_tag not in fn
    # ]
    
    vidx = 3
    # vidx = 2
    vidx = 4
    
    # neis = test_nearest_trajectories()
    # tot_kine_traj_data_fn = [
    #     f"passive_active_info_{cur_nei}_nf_300.npy" for cur_nei in neis
    # ]
    
    # resampled_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
    # resampled_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2"
    resampled_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v{vidx}"
    resampled_fns = os.listdir(resampled_root)
    resampled_fns = [fn for fn in resampled_fns if fn.endswith(".npy")]
    resampled_fns = [fn for fn in resampled_fns if inst_tag in fn]
    
    
    tot_kine_hand_qs = []
    tot_kine_obj_transl = []
    
    for cur_kien_traj_fn in resampled_fns:
        cur_full_kine_traj_fn = os.path.join(resampled_root, cur_kien_traj_fn)
        cur_kine_traj_data = np.load(cur_full_kine_traj_fn, allow_pickle=True).item()
        cur_kine_hand_qs = cur_kine_traj_data['hand_qs']
        cur_kine_obj_pos = cur_kine_traj_data['obj_pos']
        cur_kine_hand_qs = cur_kine_hand_qs[..., joint_idxes_inversed_ordering]
        tot_kine_hand_qs.append(cur_kine_hand_qs)
        tot_kine_obj_transl.append(cur_kine_obj_pos)
    tot_kine_hand_qs = np.stack(tot_kine_hand_qs, axis=0)
    tot_kine_obj_transl = np.stack(tot_kine_obj_transl, axis=0)
    
    tot_nn = tot_kine_hand_qs.shape[0]
    
    
    sv_dict = {
        'hand_qs': tot_kine_hand_qs,
        'obj_pos': tot_kine_obj_transl
    }
    sv_fn = "/data/xueyi/uni_manip/expanded_kines"
    os.makedirs(sv_fn , exist_ok=True)
    
    sv_fn = os.path.join(sv_fn, f"{inst_tag}_forecasted_kine_traj_modifications_nn{tot_nn}_v{vidx}.npy")
    np.save(sv_fn, sv_dict)
    print(  f"saved to {sv_fn}" )
    
    return kine_traj_fn


def test_nearest_trajectories():
    inst_nm_idx_dict_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data/grab_data_nm_idx_dict.npy"
    inst_diff_arr_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data/grab_diff_arr.npy"
    inst_nm_idx_dict = np.load(inst_nm_idx_dict_fn, allow_pickle=True).item()
    inst_diff_arr = np.load(inst_diff_arr_fn)
    print(f"inst_diff_arr: {inst_diff_arr.shape}")
    # for nm in inst_nm_idx_dict
    print(inst_nm_idx_dict)
    idx_to_data_nm = inst_nm_idx_dict['idx_to_data_nm']
    data_nm_to_idx = inst_nm_idx_dict['data_nm_to_idx']
    target_inst_tag = "ori_grab_s2_knife_pass_1"
    target_inst_idx = data_nm_to_idx[target_inst_tag]
    cloest_inst_idxes = inst_diff_arr[target_inst_idx]
    cloest_inst_idxes = np.argsort(cloest_inst_idxes, )[:11]
    cloest_inst_idxes = cloest_inst_idxes.tolist()
    cloest_inst_idxes =  [ int(nei_idx) for nei_idx in cloest_inst_idxes ]
    neis = []
    for nei_idx in cloest_inst_idxes:
        cur_inst_tag = idx_to_data_nm[nei_idx]
        print(f"cur_inst_tag: {cur_inst_tag}")
        # break
        neis.append(cur_inst_tag)
    return neis



os.system(f"export PYTHONPATH=/root/diffsim/IsaacGymEnvs2")
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
    
    minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) 
    
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
    
    
    for sample_idx in range(0, nn_samples):
    
        
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
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        # else:
        #     raise NotImplementedError
        converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0))
        converted_hand_verts = converted_hand_verts[0]
        tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0]

        
        
        tot_ts_tot_link_idxes_th = torch.from_numpy(tot_ts_tot_link_idxes).long()
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
        
        
        # (3,) --- the relative positions between the object center and other hand bodies #
        # 
        # (3,) --- the relative positions 
        
        # the relative positions #
        # relative positions #
        
        # relative positions #
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float()
        
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
        
        
        def closer():
            optimizer.zero_grad()
            
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs) # robot qs --- (nn_ts, ) #
            
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
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip
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
        
        resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")





def traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2" ):
    
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
    
    minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) 
    
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
    
    
    for sample_idx in range(0, nn_samples):
    
        
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
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        # else:
        #     raise NotImplementedError
        converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0))
        converted_hand_verts = converted_hand_verts[0]
        tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0]

        
        
        tot_ts_tot_link_idxes_th = torch.from_numpy(tot_ts_tot_link_idxes).long()
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
        
        
        # (3,) --- the relative positions between the object center and other hand bodies #
        # 
        # (3,) --- the relative positions 
        
        # the relative positions #
        # relative positions #
        
        # relative positions #
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float()
        
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
        
        
        def closer():
            optimizer.zero_grad()
            
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs) # robot qs --- (nn_ts, ) #
            
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
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip
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
        
        resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")




def traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root="/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2" ):
    
    # kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering] # kine hand qs #
    
    
    # 
    
    interp_frame = 120
    nn_continuing_frames = 180
    start_gen_frame = 120
    
    obj_rot_quat = kine_obj_quat[interp_frame]
    
    delta_xy = 0.02 * np.pi
    delta_rot_xyz_euler = [delta_xy, delta_xy, 0.0]
    obj_rot_matrix = R.from_quat(obj_rot_quat).as_matrix()
    obj_rot_matrix_delta = R.from_euler('xyz', delta_rot_xyz_euler, degrees=False).as_matrix()
    transformed_delta_rot_xyz_euler = R.from_matrix(obj_rot_matrix @ obj_rot_matrix_delta).as_euler('zyx')[[2,1,0]] # (3,) obj rot deltas #
     
    
    kine_obj_rot_euler = []
    for i_fr in range(kine_obj_quat.shape[0]):
        cur_obj_quat = kine_obj_quat[i_fr]
        cur_obj_rot_struct = R.from_quat(cur_obj_quat)
        cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
        kine_obj_rot_euler.append(cur_obj_rot_euler)
    kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
    diff_kine_obj_pos = kine_obj_pos[1:] - kine_obj_pos[:-1] # (nn_frames - 1) x 3 #
    diff_kine_obj_rot_euler = kine_obj_rot_euler[1:] - kine_obj_rot_euler[:-1] # (nn_frames - 1) x 3 #
    diff_kine_hand_trans = kine_hand_qs[1:, :3] - kine_hand_qs[:-1, :3] # (nn_frames - 1) x 3 # # 
    
    
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
        
    
    
    # start_gen_frame = 120
    # nn_continuing_frames = 180
    # maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    # minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    # maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    # minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    # minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) 
    
    # maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    # minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    # maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    # minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    # maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    # minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    # # maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
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
    
    
    for sample_idx in range(0, nn_samples):
    
        
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
        urdf_fn = '/root/diffsim/tiny-differentiable-simulator/python/examples/rsc/allegro_hand_description/urdf/allegro_hand_description_right_fly_ball_v2_nd_v2.urdf'
        # else:
        #     raise NotImplementedError
        converted_hand_verts, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=init_hand_qs.unsqueeze(0))
        converted_hand_verts = converted_hand_verts[0]
        tot_ts_tot_link_idxes = tot_ts_tot_link_idxes[0]

        
        
        tot_ts_tot_link_idxes_th = torch.from_numpy(tot_ts_tot_link_idxes).long()
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
        
        
        # (3,) --- the relative positions between the object center and other hand bodies #
        # 
        # (3,) --- the relative positions 
        
        # the relative positions #
        # relative positions #
        
        # relative positions #
        
        continuing_hand_qs_th = torch.from_numpy(continuing_hand_qs).float()
        
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
        
        
        def closer():
            optimizer.zero_grad()
            
            
            robot_qs = robot_hand_qs.weight[: continuing_hand_qs_th.size(0), : continuing_hand_qs_th.size(1)] # robot hand qs -- nn_ts, nn_hand_qs #
            
            robot_qs = torch.cumsum(robot_qs, dim=0) # tot_ts x nn_hand_dofs #
            
            robot_verts_tot, tot_ts_tot_link_idxes = hand_pk(urdf_fn, hand_qs=robot_qs) # robot qs --- (nn_ts, ) #
            
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
            loss = loss_palm + loss_first_tip + loss_second_tip + loss_third_tip + loss_forth_tip
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
        
        resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
        resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
        np.save(resampled_info_sv_fn, sampled_res_sv_dict)
        print(f"saved to {resampled_info_sv_fn}")
        
        



## TODO: a different trajectory modification core? ##

def extract_kinematics_from_tracked_results(result_fn, inst_tag):
    tracked_res = np.load(result_fn, allow_pickle=True).item()
    
    resampled_info_sv_root = f"/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_{inst_tag}_v5"
    
    if 'robot_delta_states_weights_np' in tracked_res:
        joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
        joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
        joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering)
        # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
        
        kine_hand_qs = tracked_res['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs # #
        kine_obj_pos = tracked_res['object_transl'] # nn_frames x 3 # #
        kine_obj_quat = tracked_res['object_rot_quat'] # nn_frames x 4 # #

        
        # traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
        traj_modifications_core_v2(kine_hand_qs, kine_obj_pos, kine_obj_quat, inst_tag, resampled_info_sv_root=resampled_info_sv_root)
    else:
        time_keys = []
        for key in tracked_res:
            try:
                key_int = int(key)
            except:
                continue
            time_keys.append(key_int)
        time_keys = sorted(time_keys) # get the time keys # extract the kinematics from the tracking results #
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
        
    




def traj_modification(inst_tag):
    kine_traj_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    kine_traj_fn = f"passive_active_info_{inst_tag}_nf_300.npy"
    kine_traj_fn = os.path.join(kine_traj_folder, kine_traj_fn)
    kine_traj_data = np.load(kine_traj_fn, allow_pickle=True).item()
    print(f"kine_traj_data: {kine_traj_data.keys()}") # trajector modifications #
    
    
    joint_idxes_ordering = [_ for _ in range(10)] + [_ + 14 for _ in range(0, 8)] + [10, 11, 12, 13]
    joint_idxes_ordering = np.array(joint_idxes_ordering).astype(np.int32)
    joint_idxes_inversed_ordering = np.argsort(joint_idxes_ordering) # get the joint inversed ordering #
    # self.joint_idxes_inversed_ordering = joint_idxes_inversed_ordering
    
    kine_hand_qs = kine_traj_data['robot_delta_states_weights_np'] # nn_frames x nn_hand_dofs #
    kine_obj_pos = kine_traj_data['object_transl'] # nn_frames x 3 #
    # kine_obj_rot = kine_traj_data['']
    kine_obj_quat = kine_traj_data['object_rot_quat'] # nn_frames x 4 #
    
    
    resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2"
    traj_modifications_core(kine_hand_qs, kine_obj_pos, kine_obj_quat, resampled_info_sv_root=resampled_info_sv_root)

    
    # # kine_hand_qs = kine_hand_qs[..., joint_idxes_inversed_ordering]
    # # it seems that convert them into zyx would be more stable than converting into the xyz #
    
    # kine_obj_rot_euler = []
    # for i_fr in range(kine_obj_quat.shape[0]):
    #     cur_obj_quat = kine_obj_quat[i_fr]
    #     cur_obj_rot_struct = R.from_quat(cur_obj_quat)
    #     cur_obj_rot_euler = cur_obj_rot_struct.as_euler('zyx')
    #     kine_obj_rot_euler.append(cur_obj_rot_euler)
    # kine_obj_rot_euler = np.stack(kine_obj_rot_euler, axis=0) # nn_frames x 3 #
    
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
    
    
    # # maxx_diff_kine_obj_pos: [0.00189576 0.0024092  0.00845268], minn_diff_kine_obj_pos: [-0.00199135 -0.01218224 -0.00612274]
    # # maxx_diff_kine_obj_rot_euler: [0.028512   0.04186282 0.06846939], minn_diff_kine_obj_rot_euler: [-0.05868825 -0.06077137 -0.03629342]
    # # maxx_diff_hand_trans: [0.00316119 0.00237919 0.00523315], minn_diff_hand_trans: [-0.00560641 -0.00521771 -0.00337168]
        
    
    
    # start_gen_frame = 120
    # nn_continuing_frames = 180
    # maxx_obj_pos_diff = [0.00189576, 0.0024092 , 0.00845268]
    # minn_obj_pos_diff = [-0.00199135 , -0.01218224 , -0.00612274]
    
    # maxx_obj_pos_diff = [max(maxx_obj_pos_diff)] * 3
    # minn_obj_pos_diff = [min(minn_obj_pos_diff)] * 3 
    
    # minn_obj_pos_diff[2] = max(minn_obj_pos_diff[2], 0.0) 
    
    # maxx_obj_pos_diff = np.array(maxx_obj_pos_diff, dtype=np.float32)
    # minn_obj_pos_diff = np.array(minn_obj_pos_diff, dtype=np.float32) # (3, )
    # maxx_obj_rot_diff = [0.028512 ,  0.04186282 , 0.06846939]
    # minn_obj_rot_diff = [-0.05868825 , -0.06077137 , -0.03629342]
    
    # maxx_obj_rot_diff = [max(maxx_obj_rot_diff)] * 3
    # minn_obj_rot_diff = [min(minn_obj_rot_diff)] * 3
    
    # maxx_obj_rot_diff = np.array(maxx_obj_rot_diff, dtype=np.float32)
    # minn_obj_rot_diff = np.array(minn_obj_rot_diff, dtype=np.float32) # get the obj rot diff and the maxx obj rot diff #
    # # maxx_hand_trans_diff = [0.00316119 , 0.00237919 , 0.00523315]
    # # minn_hand_trans_diff = [-0.00560641 , -0.00521771 , -0.00337168]
    # # maxx_hand_trans_diff = np.array(maxx_hand_trans_diff, dtype=np.float32)
    # # minn_hand_trans_diff = np.array(minn_hand_trans_diff, dtype=np.float32)
    
    
    # nn_samples = 100
    
    # tot_delta_pos_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    # tot_delta_rot_rand = np.random.uniform(0, 1, size=(nn_samples, nn_continuing_frames, 3))
    
    
    # for sample_idx in range(0, nn_samples):
    
        
    #     cur_st_hand_qs = kine_hand_qs[start_gen_frame].copy()
    #     cur_st_obj_pos = kine_obj_pos[start_gen_frame].copy()
    #     cur_st_obj_rot_euler = kine_obj_rot_euler[start_gen_frame].copy()
    #     continuing_hand_qs = []
    #     continuing_obj_pos = []
    #     continuing_obj_rot = []
    #     for i_fr in range(nn_continuing_frames):  # get the hand qs, obj pos, and the obj rot #
    #         # cur_rand_obj_delta_pos = np.random.uniform(0, 1, size=(3, ))
    #         cur_rand_obj_delta_pos = tot_delta_pos_rand[sample_idx, i_fr]
    #         # (maxx obj pos diff - minn obj pos diff) 
    #         # this scaling is not current yet  # from 0, 1 -- scale to the range of minn_obj_pos_diff and maxx_obj_pos_diff
    #         cur_rand_obj_delta_pos = cur_rand_obj_delta_pos * (maxx_obj_pos_diff - minn_obj_pos_diff) + minn_obj_pos_diff
    #         # cur_rand_obj_delta_rot = np.random.uniform(0, 1, size=(3, ))
    #         cur_rand_obj_delta_rot = tot_delta_rot_rand[sample_idx, i_fr]
    #         cur_rand_obj_delta_rot = cur_rand_obj_delta_rot * (maxx_obj_rot_diff - minn_obj_rot_diff) + minn_obj_rot_diff
    #         cur_st_hand_qs[:3] = cur_st_hand_qs[:3] + cur_rand_obj_delta_pos
    #         cur_st_obj_pos = cur_st_obj_pos + cur_rand_obj_delta_pos
    #         cur_st_obj_rot_euler = cur_st_obj_rot_euler + cur_rand_obj_delta_rot
    #         cur_st_obj_rot_struct = R.from_euler('zyx', cur_st_obj_rot_euler, degrees=False)
    #         cur_st_obj_rot_quat = cur_st_obj_rot_struct.as_quat()
            
            
    #         continuing_hand_qs.append(cur_st_hand_qs.copy())
    #         continuing_obj_pos.append(cur_st_obj_pos.copy())
    #         continuing_obj_rot.append(cur_st_obj_rot_quat.copy()) 
            
        
    #     continuing_hand_qs = np.stack(continuing_hand_qs, axis=0)
    #     continuing_obj_pos = np.stack(continuing_obj_pos, axis=0)
    #     continuing_obj_rot = np.stack(continuing_obj_rot, axis=0)
        
        
        
    #     first_frames_hand_qs = kine_hand_qs[:start_gen_frame + 1]
    #     first_frames_obj_pos = kine_obj_pos[:start_gen_frame + 1]
    #     first_frames_obj_rot = kine_obj_quat[:start_gen_frame + 1]
        
    #     sampled_hand_qs = np.concatenate([first_frames_hand_qs, continuing_hand_qs], axis=0)
    #     sampled_obj_pos = np.concatenate([first_frames_obj_pos, continuing_obj_pos], axis=0)
    #     sampled_obj_rot = np.concatenate([first_frames_obj_rot, continuing_obj_rot], axis=0)
        
    #     # sampled res sv dict #
    #     # palm pose should be simulator to the orjbecvt pose, the orignetation should not be changed too much; contacts hould be maintained #
    #     # palm pos should be similar to the object pos #
    #     sampled_res_sv_dict = {
    #         'hand_qs': sampled_hand_qs,
    #         'obj_pos': sampled_obj_pos,
    #         'obj_rot': sampled_obj_rot
    #     }
        
    #     # resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled"
    #     resampled_info_sv_root =  "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled_v2"
    #     os.makedirs(resampled_info_sv_root, exist_ok=True)
        
    #     resampled_info_sv_fn =  f"passive_active_info_{inst_tag}_nf_300_sample_{sample_idx}.npy"
    #     resampled_info_sv_fn = os.path.join(resampled_info_sv_root, resampled_info_sv_fn)
    #     np.save(resampled_info_sv_fn, sampled_res_sv_dict)
    #     print(f"saved to {resampled_info_sv_fn}")



def compare_tracked_ori_trajs(tracked_fn, ori_kine_fn):
    tracked_data = np.load(tracked_fn, allow_pickle=True).item()
    kine_data = np.load(ori_kine_fn, allow_pickle=True).item()
    print(tracked_data.keys(), tracked_data[1].keys())
    print(kine_data.keys())
    
    
    tot_tracked_obj_pose = []
    # object_pose
    tracked_keys = tracked_data.keys()
    tracked_keys_int = []
    for key in tracked_keys:
        try:
            key_int = int(key)
        except:
            continue
        tracked_keys_int.append(key_int)
    tracked_keys_int = sorted(tracked_keys_int)
    for key_int in tracked_keys_int:
        cur_tracked_data = tracked_data[key_int]
        cur_tracked_obj_pose = cur_tracked_data['object_pose']
        tot_tracked_obj_pose.append(cur_tracked_obj_pose)
    
    tot_tracked_obj_pose = np.stack(tot_tracked_obj_pose, axis=0)
    
    ori_kine_obj_pos = kine_data['obj_pos']
    ori_kine_obj_ornt = kine_data['obj_rot']
    
    tracked_obj_pos = tot_tracked_obj_pose[..., :3]
    tracked_obj_ornt = tot_tracked_obj_pose[..., 3: ]
    
    minn_ws = min(tracked_obj_pos.shape[0], ori_kine_obj_pos.shape[0])
    
    ori_kine_obj_pos = ori_kine_obj_pos[: minn_ws]
    ori_kine_obj_ornt = ori_kine_obj_ornt[: minn_ws]
    ori_kine_obj_ornt = torch.from_numpy(ori_kine_obj_ornt).float()
    
    tot_tracked_diffs = []
    
    for i_env in range(tracked_obj_pos.shape[1]):
        
        cur_env_tracked_obj_pos = tracked_obj_pos[: , i_env][:minn_ws]
        cur_env_tracked_obj_ornt = tracked_obj_ornt[:, i_env][:minn_ws]
        
        diff_obj_pos = np.sum((cur_env_tracked_obj_pos - ori_kine_obj_pos) ** 2, axis=-1)
        diff_obj_pos = np.sqrt(diff_obj_pos)
        diff_obj_pos = np.mean(diff_obj_pos).item()
        
        # tot_rot_diff = []
        # for i_fr in range(tracked_obj_ornt.shape[0]):
        #     cur_tracked_obj_ornt = torch.from_numpy(tracked_obj_ornt).float()
            
            
        cur_env_tracked_obj_ornt = torch.from_numpy(cur_env_tracked_obj_ornt).float()
        
            
        quat_diff = quat_mul(cur_env_tracked_obj_ornt, quat_conjugate(ori_kine_obj_ornt))
        rot_dist = 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 0:3], p=2, dim=-1), max=1.0))
        
        
        rot_dist = torch.mean(rot_dist).item()
        
        tot_tracked_diffs.append((diff_obj_pos, rot_dist))
    tot_tracked_diffs = sorted(tot_tracked_diffs, key=lambda ii: ii[0])
    first_tot_tracked_diffs = tot_tracked_diffs[0]
    
    

    print(f"diff_obj_pos: {first_tot_tracked_diffs[0]}, rot_dist: {first_tot_tracked_diffs[1]}")
    
    pass



def test_retargeted_info(retar_info_fn):
    cur_retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    print(cur_retar_info.keys())
    object_transl = cur_retar_info['object_transl']
    for i_fr in range(object_transl.shape[0]):
        cur_obj_transl = object_transl[i_fr]
        print(f"i_fr: {i_fr}, object_transl: {cur_obj_transl}")




def test_rotations():
    # -1.5707
    from scipy.spatial.transform import Rotation as R
    rot_xyz = [0 ,-1.5707, -1.5707]
    rot_xyz = [-1.5707, -1.5707, 0]
    # rot_xyz = [1.57079632679, 0 ,0]
    # rot_xyz = [0, 0 , 0]
    rot_xyz = np.array(rot_xyz, dtype=np.float32)
    rot_quat = R.from_euler('zyx', rot_xyz, degrees=False).as_quat()
    print(rot_quat)
    pass




def test_aaa():
    retar_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_0d4_0d5_warm/data/passive_active_info_ori_grab_s2_hammer_lift_nf_300.npy"
    retar_info = np.load(retar_info_fn, allow_pickle=True ).item()
    print(retar_info.keys())



# try to train the model #

def test_retargeted_res():
    retargeted_res_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2/data/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    retargeted_res = np.load(retargeted_res_fn, allow_pickle=True).item() # 
    print(retargeted_res.keys())
    robot_delta_states_weights_np = retargeted_res['robot_delta_states_weights_np']
    print(robot_delta_states_weights_np.shape)
    for i_fr in range(robot_delta_states_weights_np.shape[0]):
        print(robot_delta_states_weights_np[i_fr, 7:])


def test_random_values():
    rand_values = torch.randn((10, ))
    rand_values_scale = 0.02
    rand_values = rand_values * rand_values_scale
    print(rand_values)



def summarize_all_instances(tot_inst_folder, new_inst_folder, hand_type):
    # modified_kinematics_data_leap_wfranka
    # /data/xueyi/data/modified_kinematics_data_leap_wfranka/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_banana_eat_1_v7
    # /data/xueyi/data/modified_kinematics_data_leap_wfranka/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_banana_eat_1_v7/leap_passive_active_info_ori_grab_s2_banana_eat_1_nf_300.npy
    if hand_type == 'leap':
        inst_tag_st_flag = "leap_passive_active_info_ori_grab"
    elif hand_type == 'allegro':
        inst_tag_st_flag = "passive_active_info_ori_grab"
    else:
        raise ValueError("hand type not supported")
    tot_inst_folders = os.listdir(tot_inst_folder)
    folder_st_flag = "GRAB_Tracking_PK_reduced_300_resampled_"
    
    for cur_inst_folder in tot_inst_folders:
        cur_full_folder = os.path.join(tot_inst_folder, cur_inst_folder)
    pass


def test_obj_reorient_vel(resampled_obj_kine_seq_fn):
    # sampled_res_sv_dict = {
    #     'hand_qs': sampled_hand_qs,
    #     'obj_pos': sampled_obj_pos,
    #     'obj_rot': sampled_obj_rot,
    #     'robot_delta_states_weights_np': sampled_hand_qs,
    #     'object_transl': sampled_obj_pos,
    #     'object_rot_quat': sampled_obj_rot,
    #     'link_key_to_link_pos': link_key_to_link_pos
    # }
    # we should be careful 
    obj_kine_info = np.load(resampled_obj_kine_seq_fn, allow_pickle=True).item() # 
    obj_rot_quat = obj_kine_info['object_rot_quat']
    rot_af_frame = 120
    obj_reorent_quat = obj_rot_quat[rot_af_frame: ]
    obj_rot_euler_angle = []
    for i_fr in range(obj_reorent_quat.shape[0]):
        cur_obj_quat = obj_reorent_quat[i_fr]
        cur_obj_euler = R.from_quat(cur_obj_quat).as_euler('xyz', degrees=False) # [[2, 1, 0]]
        obj_rot_euler_angle.append(cur_obj_euler)
    obj_rot_euler_angle = np.stack(obj_rot_euler_angle, axis=0)
    obj_rot_euler_delta_angles = obj_rot_euler_angle[1: ] - obj_rot_euler_angle[: -1]
    avg_rot_deltas = np.mean(obj_rot_euler_delta_angles, axis=0)
    print(avg_rot_deltas)
    cur_st_rot_angle = obj_rot_euler_angle[0]
    for i_fr in range(obj_rot_euler_delta_angles.shape[0]):
        cur_accumulated_rot_angle = cur_st_rot_angle + avg_rot_deltas * i_fr
        print(f"i_fr: {i_fr}, actual_rot_euler: {obj_rot_euler_angle[i_fr]}, accumulated: {cur_accumulated_rot_angle}")
    # rotation axis and the angular velocities #
    pass


def test_reorient_dict(reorient_dict_fn):
    reorient_dict = np.load(reorient_dict_fn, allow_pickle=True).item()
    print(f"reorient_dict: {reorient_dict.keys()}")


def best_optimized_res_reorient(data_optimized_res_nn, data_inst_tag, index=None, downsample=False):
    optimized_res = np.load(data_optimized_res_nn, allow_pickle=True).item()
    
    if '_nf_300' in data_inst_tag:
        # kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    else:
        kinematics_data_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
    
    # /data/xueyi/data/modified_kinematics_data_leap_wfranka/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_duck_inspect_1_v7/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy
    # /data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_duck_inspect_1_v8
    pure_data_inst_tag = data_inst_tag.split("_nf_300")[0]
    
    # kinematics_data_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/"
    # kinematics_data_folder = f"GRAB_Tracking_PK_reduced_300_resampled_{pure_data_inst_tag}_v8"
    
    kinematics_data_root = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf/"
    kinematics_data_folder = f"GRAB_Tracking_PK_reduced_300_resampled_{pure_data_inst_tag}_v9"
    kinematics_data_fn = f"leap_passive_active_info_{data_inst_tag}.npy"
    
    kinematics_data_fn = os.path.join(kinematics_data_root, kinematics_data_folder, kinematics_data_fn)
    
    save_info = np.load(kinematics_data_fn, allow_pickle=True).item()
    
    # if data_inst_tag.endswith(".npy"):
    #     cur_inst_kine_data_fn = data_inst_tag
    # else:
    #     cur_inst_kine_data_fn = f"passive_active_info_{data_inst_tag}.npy"
    #     cur_inst_kine_data_fn = os.path.join(kinematics_data_root, cur_inst_kine_data_fn)
    
    # save_info = np.load(cur_inst_kine_data_fn, allow_pickle=True).item()
        
        
    # hand_qs = save_info['robot_delta_states_weights_np'][ : ]
    # hand_qs = hand_qs[: , : ]
    
    goal_obj_trans = save_info['object_transl']
    goal_obj_rot_quat = save_info['object_rot_quat']
    
    # if downsample:
    #     idxes = [ ii for ii in range(goal_obj_trans.shape[0]) if ii % 2 == 0 ]
    #     idxes = np.array(idxes, dtype=np.int32)
    #     goal_obj_trans = goal_obj_trans[idxes]
    #     goal_obj_rot_quat = goal_obj_rot_quat[idxes]
        

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
    
    tot_obs_buf = []
    
    for ts in tot_ts:
        cur_optimized_obj_pose = optimized_res[ts]['object_pose']
        tot_optimized_obj_pose.append(cur_optimized_obj_pose)
        
        cur_optimized_hand_qs = optimized_res[ts]['shadow_hand_dof_pos']
        cur_optimized_hand_qtars = optimized_res[ts]['shadow_hand_dof_tars']
        tot_optimized_hand_qs.append(cur_optimized_hand_qs)
        tot_optimized_hand_qtars.append(cur_optimized_hand_qtars)
        tot_obs_buf.append(optimized_res[ts]['observations'])
        
    tot_optimized_obj_pose = np.stack(tot_optimized_obj_pose, axis=1) # nn_envs x nn_ts x 7 #
    tot_optimized_hand_qs = np.stack(tot_optimized_hand_qs, axis=1)
    tot_optimized_hand_qtars = np.stack(tot_optimized_hand_qtars, axis=1)
    tot_obs_buf = np.stack(tot_obs_buf, axis=1)
    
    if index is not None:
        tot_optimized_obj_pose = tot_optimized_obj_pose[index: index + 1]
        tot_optimized_hand_qs = tot_optimized_hand_qs[index: index + 1]
        tot_optimized_hand_qtars = tot_optimized_hand_qtars[index: index + 1]
    
    tot_env_diff_obj_pos, tot_env_diff_obj_rot = [], []
    tot_env_weighted_obj_pose_diff = []
    w_pos, w_ornt = 0.01, 0.33
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
    # obj_pos_diff of the object position errors #
    new_optimized_info = {
        'optimized_obj_pose': tot_optimized_obj_pose[sorted_envs_idxes][:1],
        'optimized_hand_qs': tot_optimized_hand_qs[sorted_envs_idxes][:1],
        'optimized_hand_qtars': tot_optimized_hand_qtars[sorted_envs_idxes][:1],
        'obj_pose_diff': tot_env_weighted_obj_pose_diff[sorted_envs_idxes][:1],
        'obj_pos_diff': tot_env_diff_obj_pos[sorted_envs_idxes][:1],
        'obj_rot_diff': tot_env_diff_obj_rot[sorted_envs_idxes][:1],
        # 'obs_buf': tot_obs_buf
    }
    return new_optimized_info
    
    
def inspect_teacher_weights():
    teacher_weights_fn = "/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights_v4.npy"
    teacher_weights_dict = np.load(teacher_weights_fn, allow_pickle=True).item()
    for idxx in teacher_weights_dict:
        teacher_weight_fn = teacher_weights_dict[idxx]
        print(f"idxx: {idxx}, teacher_weight_fn: {teacher_weight_fn}")



# /data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_mug_drink_2_v8/leap_passive_active_info_ori_grab_s1_mug_drink_2_nf_300.npy
def resave_retargeted_res(root_folder, inst_starting_tag = "leap_passive_active_info_ori_grab_"):
    tot_inst_folders = os.listdir(root_folder)
    tot_inst_folders = [
        fn for fn in tot_inst_folders if os.path.isdir(os.path.join(root_folder, fn))
    ]
    # inst_starting_tag = "leap_passive_active_info_ori_grab_"
    for cur_inst_folder in tot_inst_folders:
        cur_full_inst_folder = os.path.join(root_folder, cur_inst_folder)
        tot_inst_fns = os.listdir(cur_full_inst_folder)
        tot_inst_fns = [
            fn for fn in tot_inst_fns if fn[: len(inst_starting_tag)] == inst_starting_tag and fn.endswith(".npy")
        ]
        retargeted_inst_fn = tot_inst_fns[0]
        retargeted_inst_fn = os.path.join(
            root_folder, cur_inst_folder, retargeted_inst_fn
        )
        retargeted_inst_info = np.load(retargeted_inst_fn, allow_pickle=True).item()
        print(f"retargeted_inst_info: {retargeted_inst_info.keys()}")
        to_remove_inst_tag = "rotation_axis"
        removed_retargeted_inst_info = {
            key: retargeted_inst_info[key] for key in retargeted_inst_info if key != to_remove_inst_tag
        }
        removed_retargeted_inst_info_sv_fn = retargeted_inst_fn.replace(".npy", "_removed.npy")
        np.save(removed_retargeted_inst_info_sv_fn, removed_retargeted_inst_info)
        print(f"saved to {removed_retargeted_inst_info_sv_fn}")
        # break


def test_saved_experiences():
    saved_experience_fn = "experience_buffer_sv_0.npy"
    saved_dict = np.load(saved_experience_fn, allow_pickle=True).item()
    print(f"saved_dict: {saved_dict.keys()}")
    for key in saved_dict:
        cur_val = saved_dict[key]
        print(f"key: {key}, cur_val: {cur_val.shape}")



def test_real_play_file():
    # real_play_file_fn = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/selected_res/ts_to_real_play_info_1.npy"
    # real_play_file_fn = "selected_res_elephant_from_apple/ts_to_real_play_info_1.npy"
    real_play_file_fn = "selected_res_waterbottle_pass_1_new/ts_to_real_play_info_1.npy"
    out_action_fn = "tmp_output_res/tot_output_actions.npy"
    real_play_dict = np.load(real_play_file_fn, allow_pickle=True).item()
    
    # out_action_dict = np.load(out_action_fn, allow_pickle=True) # .item() #
    out_action_dict = np.load(out_action_fn) 
    print(f"out_actions_dict: {out_action_dict.shape}, {out_action_dict.shape[0] // 23}") # 
    
    for i_fr in range(out_action_dict.shape[0]):
        cur_fr_out_action_dict = out_action_dict[i_fr]
        print(cur_fr_out_action_dict)
    
    # for key in real_play_dict: # 
    #     cur_val = real_play_dict[key] # 
    #     if isinstance(cur_val, dict): # 
    #         for sub_key in cur_val: # 
    #             sub_val = cur_val[sub_key] # 
    #             print(f"key: {key}, sub_key: {sub_key}, sub_val: {sub_val.shape}") # 
    #     else: # 
    #         print(f"key: {key}, cur_val: {cur_val.shape}")
    #     # key: 215, real_arm_pos, sub_val: (7,)
    #     # key: 215, sub_key: real_leap_pos_to_sim, sub_val: (16,)
    #     # key: 215, sub_key: real_object_pose, sub_val: (7,)
    #     # key: 215, sub_key: cur_step_already_execute_actions, sub_val: (23,)
    #     # key: 215, sub_key: sim_hand_qpos, sub_val: (23,)
    #     # key: 215, sub_key: sim_fingertip_pos, sub_val: (4, 3)
    #     # key: 215, sub_key: sim_object_pose, sub_val: (7,)
    #     # set those values 
    tot_keys = list( real_play_dict.keys())
    tot_keys = sorted(tot_keys)
    first_key = tot_keys[0]
    
    first_ts_real_play_dict = real_play_dict[first_key]
    for key in first_ts_real_play_dict:
        cur_val = first_ts_real_play_dict[key]
        cur_val = torch.from_numpy(cur_val).float()
        print(f"key: {key}, cur_val: {cur_val}")


def test_presaved_experiences_buf():
    exp_buf_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s9_duck_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_27-20-20-37/experience_buffer_sv_0.npy"
    exp_buf = np.load(exp_buf_fn, allow_pickle=True).item()
    print(f"exp_buf: {exp_buf.keys()}")


def compare_two_opt_res(opt_res_a_fn, opt_res_b_fn):
    opt_res_a = np.load(opt_res_a_fn, allow_pickle=True).item()
    opt_res_b = np.load(opt_res_b_fn, allow_pickle=True).item() # 
    print(f"opt_res_a: {opt_res_a.keys()}")
    print(f"opt_res_b: {opt_res_b.keys()}")
    tot_a_better_than_b_keys = []
    tot_a_pos_diff = []
    tot_b_pos_diff = []
    for key in opt_res_b:
        if key not in opt_res_a:
            continue
        cur_res_b = opt_res_b[key]
        # print(f"key: {key}, cur_res_b: {cur_res_b.keys()}")
        cur_res_obj_pos_diff_key  = "obj_pos_diff"
        cur_res_b_obj_pos_diff = cur_res_b[cur_res_obj_pos_diff_key]
        print(f"key: {key}, cur_res_b: {cur_res_b_obj_pos_diff}")
        cur_res_a = opt_res_a[key]
        cur_res_a_obj_pos_diff = cur_res_a[cur_res_obj_pos_diff_key]
        
        if cur_res_a_obj_pos_diff < cur_res_b_obj_pos_diff:
            tot_a_better_than_b_keys.append(key[0])
        
        tot_a_pos_diff.append(cur_res_a_obj_pos_diff)
        tot_b_pos_diff.append(cur_res_b_obj_pos_diff)
        
    print(tot_a_better_than_b_keys)
    print(len(tot_a_better_than_b_keys))
    avg_a_pos_diff = np.mean(tot_a_pos_diff)
    avg_b_pos_diff = np.mean(tot_b_pos_diff)
    print(f"avg_a_pos_diff: {avg_a_pos_diff}, avg_b_pos_diff: {avg_b_pos_diff}")
    tot_a_pos_diff = sorted(tot_a_pos_diff)
    tot_b_pos_diff = sorted(tot_b_pos_diff)
    avg_a_pos_diff = tot_a_pos_diff[len(tot_a_pos_diff) // 2]
    avg_b_pos_diff = tot_b_pos_diff[len(tot_b_pos_diff) // 2]
    print(f"avg_a_pos_diff: {avg_a_pos_diff}, avg_b_pos_diff: {avg_b_pos_diff}")
    
    print(f"tot_a_pos_diff: {tot_a_pos_diff[:10]}")
    print(f"tot_b_pos_diff: {tot_b_pos_diff[:10]}")

def centralize_experience_replay_buffer(tot_folder, target_experience_folder):
    target_inst_tags = ['ori_grab_s9_banana_eat_1_nf_300', 'ori_grab_s9_cylindersmall_pass_1_nf_300', 'ori_grab_s9_stapler_lift_nf_300', 'ori_grab_s9_cubelarge_lift_nf_300', 'ori_grab_s9_doorknob_lift_nf_300', 'ori_grab_s9_waterbottle_open_1_nf_300', 'ori_grab_s9_pyramidlarge_pass_1_nf_300', 'ori_grab_s9_cylinderlarge_inspect_1_nf_300', 'ori_grab_s9_flute_play_1_nf_300', 'ori_grab_s9_cylinderlarge_lift_nf_300', 'ori_grab_s9_cubemedium_lift_nf_300', 'ori_grab_s9_piggybank_lift_Retake_nf_300', 'ori_grab_s9_pyramidmedium_pass_1_nf_300', 'ori_grab_s9_spheremedium_pass_1_nf_300', 'ori_grab_s9_eyeglasses_wear_1_nf_300', 'ori_grab_s9_flashlight_on_1_nf_300', 'ori_grab_s9_stanfordbunny_inspect_1_nf_300', 'ori_grab_s9_gamecontroller_pass_1_nf_300', 'ori_grab_s9_lightbulb_pass_1_nf_300', 'ori_grab_s9_duck_inspect_1_nf_300', 'ori_grab_s9_camera_lift_nf_300', 'ori_grab_s9_hammer_use_3_nf_300', 'ori_grab_s9_torussmall_inspect_1_nf_300', 'ori_grab_s9_cylindermedium_inspect_1_nf_300', 'ori_grab_s9_spheresmall_pass_1_nf_300', 'ori_grab_s9_watch_set_2_nf_300', 'ori_grab_s9_torusmedium_pass_1_nf_300', 'ori_grab_s9_torusmedium_inspect_1_nf_300', 'ori_grab_s9_elephant_lift_nf_300', 'ori_grab_s9_flashlight_pass_1_nf_300', 'ori_grab_s9_lightbulb_screw_1_nf_300', 'ori_grab_s9_mouse_pass_1_nf_300', 'ori_grab_s9_binoculars_pass_1_nf_300', 'ori_grab_s9_stanfordbunny_pass_1_nf_300', 'ori_grab_s9_hammer_use_1_nf_300', 'ori_grab_s9_alarmclock_lift_nf_300', 'ori_grab_s9_stamp_pass_1_nf_300', 'ori_grab_s9_torussmall_lift_nf_300', 'ori_grab_s9_apple_eat_1_nf_300', 'ori_grab_s9_pyramidmedium_lift_Retake_nf_300', 'ori_grab_s9_flute_pass_1_nf_300', 'ori_grab_s9_duck_pass_1_nf_300', 'ori_grab_s9_cubelarge_pass_1_nf_300', 'ori_grab_s9_stamp_stamp_1_nf_300', 'ori_grab_s9_mouse_use_1_nf_300', 'ori_grab_s9_alarmclock_pass_1_nf_300', 'ori_grab_s9_waterbottle_pour_1_nf_300', 'ori_grab_s9_cubesmall_inspect_1_nf_300', 'ori_grab_s9_phone_pass_1_nf_300', 'ori_grab_s9_flashlight_lift_nf_300', 'ori_grab_s9_watch_set_1_nf_300', 'ori_grab_s9_spheremedium_inspect_1_nf_300', 'ori_grab_s9_hand_inspect_1_nf_300', 'ori_grab_s9_hand_pass_1_nf_300']
    tot_subfolders = os.listdir(tot_folder)
    experience_replay_buffer_fn = "experience_buffer_sv_0.npy"
    tot_subfolders = [
        cur_sub_fn for cur_sub_fn in tot_subfolders if os.path.isdir(os.path.join(tot_folder, cur_sub_fn)) and os.path.exists(os.path.join(tot_folder, cur_sub_fn, experience_replay_buffer_fn))
    ]
    prefix_folder = "tracking_"
    for cur_sub_folder in tot_subfolders:
        cur_inst_name = cur_sub_folder.split("_obs_pure_")[0][len(prefix_folder): ]
        if cur_inst_name in target_inst_tags:
            cur_full_sub_folder = os.path.join(tot_folder, cur_sub_folder)
            cur_experience_replay_buffer_fn = os.path.join(cur_full_sub_folder, experience_replay_buffer_fn)
            target_exp_replay_buffer_fn = f"experience_buffer_sv_{cur_inst_name}.npy"
            target_exp_replay_buffer_fn = os.path.join(target_experience_folder, target_exp_replay_buffer_fn)
            commands = f"cp {cur_experience_replay_buffer_fn} {target_exp_replay_buffer_fn}"
            print(commands)
            os.system(commands)


def test_retargeted_res(retargeted_res_fn):
    retargeted_res = np.load(retargeted_res_fn , allow_pickle=True).item()
    print(retargeted_res.keys())
    pass

def test_real_play_info(real_play_info_fn):
    real_play_info = np.load(real_play_info_fn, allow_pickle=True).item()
    print(real_play_info.keys())
    test_ts = 3
    test_ts_real_play_info = real_play_info[test_ts]
    print(test_ts_real_play_info.keys())
    for key in test_ts_real_play_info:
        cur_val = test_ts_real_play_info[key]
        print(f"key: {key}, val: {cur_val.shape}")
    tot_abs_diff_real_w_tar = []
    tot_abs_diff_sim_w_tar = []
    tot_abs_diff = []
    # compare the difference between the real play info and the info in the sim
    for ts in real_play_info:
        # dict_keys(['real_arm_pos', 'real_leap_pos_to_sim', 'real_object_pose', 'cur_step_already_execute_actions', 'sim_hand_qpos', 'sim_fingertip_pos', 'sim_object_pose'])
        cur_ts_real_play_info = real_play_info[ts]
        real_arm_pos = cur_ts_real_play_info['real_arm_pos']
        real_leap_pos_to_sim = cur_ts_real_play_info['real_leap_pos_to_sim']
        sim_hand_qpos = cur_ts_real_play_info['sim_hand_qpos']
        sim_arm_qpos = sim_hand_qpos[: 7]
        sim_leap_hand_pos = sim_hand_qpos[7: ]
        # print(real_arm_pos - sim_arm_qpos)
        # print(sim_arm_qpos)
        # print(real_leap_pos_to_sim - sim_leap_hand_pos)
        executed_targets = cur_ts_real_play_info['cur_step_already_execute_actions']
        executed_targets_fingers = executed_targets[7: ]
        # executed_targets_fingers = 
        diff_real_with_targets = real_leap_pos_to_sim - executed_targets_fingers
        tot_abs_diff_real_w_tar.append(np.abs(diff_real_with_targets))
        tot_abs_diff_sim_w_tar.append(np.abs(sim_leap_hand_pos - executed_targets_fingers))
        # real - tar - sim + tar = real - tar
        tot_abs_diff.append(np.abs(real_leap_pos_to_sim - sim_leap_hand_pos))
    tot_abs_diff = np.stack(tot_abs_diff, axis=0)
    # and what about the closed loop differences? #
    avg_abs_diff = np.mean(tot_abs_diff, axis=0)
    print(avg_abs_diff)
    tot_abs_diff_real_w_tar = np.stack(tot_abs_diff_real_w_tar, axis=0)
    tot_abs_diff_sim_w_tar = np.stack(tot_abs_diff_sim_w_tar, axis=0)
    avg_abs_diff_real_w_tar = np.mean(tot_abs_diff_real_w_tar, axis=0)
    avg_abs_diff_sim_w_tar = np.mean(tot_abs_diff_sim_w_tar, axis=0)
    print(avg_abs_diff_real_w_tar)
    print(avg_abs_diff_sim_w_tar)

    # from these informa

def inspect_ts_to_reset_infos(ts_to_hand_obj_info_fn):
    ts_to_hand_obj_info = np.load(ts_to_hand_obj_info_fn, allow_pickle=True).item()
    tot_ts = list(ts_to_hand_obj_info.keys())
    tot_ts = [cur_ts for cur_ts in tot_ts if isinstance(cur_ts, int)]
    tot_ts = sorted(tot_ts)
    first_ts = tot_ts[0]
    first_ts_info = ts_to_hand_obj_info[first_ts]
    # for key in first_ts_info:
    #     val = first_ts_info[key]
    #     print(f"key: {key}, val: {val.shape}")
    first_shadow_hand_pos = first_ts_info['shadow_hand_dof_pos']
    first_obj_pose = first_ts_info['object_pose']
    print(first_shadow_hand_pos[1])
    print(first_obj_pose[1])
    
    first_shadow_hand_pos = torch.from_numpy(first_shadow_hand_pos).float()
    first_obj_pose = torch.from_numpy(first_obj_pose).float()
    print(first_shadow_hand_pos[1])
    print(first_obj_pose[1])
    

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
    

def random_sample_hand_obj_traj(kine_info_dict_fn, sv_samples_folder):
    kine_info_dict = np.load(kine_info_dict_fn, allow_pickle=True).item() # kine info dict fn 
    print(kine_info_dict.keys())
    # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy
    object_transl = kine_info_dict['object_transl']
    object_rot_quat = kine_info_dict['object_rot_quat']
    robot_delta_states_weights_np = kine_info_dict['robot_delta_states_weights_np']
    
    st_sample_fr = 100 # assume that the object has at least already been lifted up from the table #
    # sample a direction and a scale #
    # maximum rotation delta #
    maxx_rotation_delta = 0.08 * np.pi
    maxx_trans_delta = 0.04 # 
    
    maxx_rotation_delta = 0.04 * np.pi
    maxx_trans_delta = 0.02 # 
    
    maxx_rotation_delta = 0.02 * np.pi
    maxx_trans_delta = 0.01 # 
    nn_fr_per_chunk = 20
    nn_chunks = (object_transl.shape[0] - st_sample_fr) // nn_fr_per_chunk # number of chunks
    cur_st_obj_transl = object_transl[st_sample_fr - 1].copy()  # start object transl
    cur_st_obj_rot_quat = object_rot_quat[st_sample_fr - 1].copy() # 
    cur_st_robot_states = robot_delta_states_weights_np[st_sample_fr - 1].copy() # 
    
    
    cur_st_obj_rot_euler = R.from_quat(cur_st_obj_rot_quat).as_euler('xyz', degrees=False)
    
    
    random_sampled_obj_transl = []
    random_sampled_obj_rot_quat = []
    random_sampled_robot_states = []
    for i_chunk in range(nn_chunks):
        # i_chunk # 
        # start from sampling the transition and rotation direction #
        # then you should use that velocity to motivate the object and the hand #
        # then you should use that velocities to motivate the object and the hand #
        # the main question is how to change the hand dofs #
        ### random sampling transition and rotation directions ###
        rnd_trans_dir = np.random.randn(3)
        rnd_trans_dir = rnd_trans_dir / np.linalg.norm(rnd_trans_dir, ord=2)
        rnd_rot_dir = np.random.randn(3)
        rnd_rot_dir = rnd_rot_dir / np.linalg.norm(rnd_rot_dir, ord=2)
        
        # rnd_delta_trans = rnd_trans_dir *
        rnd_trans_scale = np.random.uniform(low=0.0, high=maxx_trans_delta, size=(3, ))
        rnd_rot_scale = np.random.uniform(low=0.0, high=maxx_rotation_delta, size=(3, ))
        rnd_delta_trans = rnd_trans_dir * rnd_trans_scale
        rnd_delta_rot = rnd_rot_dir * rnd_rot_scale
        
        for i_fr in range(nn_fr_per_chunk):
            # cur_st_obj_transl = cur_st_obj_transl + rnd_delta_trans 
            # quat_from_euler_xyz # quat_mul # 
            # rnd_delta_rot_th = torch.from_numpy(rnd_delta_rot).float() 
            # cur_rot_quat_th = torch.from_numpy(cur_st_obj_rot_quat).float()
            # cur_delta_rot_quat = quat_from_euler_xyz(rnd_delta_rot_th[0], rnd_delta_rot_th[1], rnd_delta_rot_th[2])
            # cur_delta_rot_quat = quat_from_euler_xyz(rnd_delta_rot_th[2], rnd_delta_rot_th[1], rnd_delta_rot_th[0])
            # cur_rot_quat_th = quat_mul(cur_rot_quat_th, cur_delta_rot_quat)
            # cur_rot_quat_th = quat_mul(cur_delta_rot_quat, cur_rot_quat_th)
            # cur_rot_quat_th = cur_rot_quat_th / torch.norm(cur_rot_quat_th, dim=-1, keepdim=True)
            # cur_st_obj_rot_quat = cur_rot_quat_th.numpy()
            
            # delta_rot_matrix = R.from_quat(cur_delta_rot_quat.numpy()).as_matrix()
            # delta_rot_matrix_at_delta_trans = delta_rot_matrix @ (cur_st_robot_states[:3] - (cur_st_obj_transl - rnd_delta_trans)) + cur_st_obj_transl
            # delta_rot_matrix = R.from_euler('zyx', rnd_delta_rot[[2,1,0]], degrees=False).as_matrix()
            delta_rot_matrix = R.from_euler('xyz', rnd_delta_rot, degrees=False).as_matrix()
            
            # cur_st_robot_states[:3] = cur_st_robot_states[:3] + rnd_delta_trans
            # cur_st_robot_states[:3] = delta_rot_matrix_at_delta_trans
            cur_st_robot_states[:3] = delta_rot_matrix @ cur_st_robot_states[:3] + rnd_delta_trans
            cur_st_robot_states[3:6] = cur_st_robot_states[3:6] + rnd_delta_rot # rnd delta rot #  # rnd delta rot #
            
            
            cur_st_obj_transl = delta_rot_matrix @ cur_st_obj_transl + rnd_delta_trans 
            
            cur_st_obj_rot_euler = cur_st_obj_rot_euler + rnd_delta_rot # [[2,1,0]]
            cur_st_obj_rot_quat = R.from_euler('xyz', cur_st_obj_rot_euler, degrees=False).as_quat()
            
            random_sampled_obj_transl.append(cur_st_obj_transl.copy())
            random_sampled_obj_rot_quat.append(cur_st_obj_rot_quat.copy())
            random_sampled_robot_states.append(cur_st_robot_states.copy())
    random_sampled_obj_transl = np.stack(random_sampled_obj_transl, axis=0)
    random_sampled_obj_rot_quat = np.stack(random_sampled_obj_rot_quat, axis=0)
    random_sampled_robot_states = np.stack(random_sampled_robot_states, axis=0)
    
    sampled_obj_transl = np.concatenate([object_transl[: st_sample_fr], random_sampled_obj_transl], axis=0)
    sampled_obj_rot_quat = np.concatenate([object_rot_quat[: st_sample_fr], random_sampled_obj_rot_quat], axis=0)
    sampled_robot_states = np.concatenate([robot_delta_states_weights_np[: st_sample_fr], random_sampled_robot_states], axis=0)
    sampled_dict = {
        'object_transl': sampled_obj_transl,
        'object_rot_quat': sampled_obj_rot_quat,
        'robot_delta_states_weights_np': sampled_robot_states
    }
    sampled_info_sv_fn = kine_info_dict_fn.split("/")[-1].replace(".npy", "_sampled.npy")
    sampled_info_sv_fn = os.path.join(sv_samples_folder, sampled_info_sv_fn)
    np.save(sampled_info_sv_fn, sampled_dict)
    print(f"saved to {sampled_info_sv_fn}")
    pass

def test_free_hand_replay_fn(freehand_folder='freehand_replay_res'):
    # free_hand_replay_fn = "freehand_replay_res/ts_to_real_play_info_1.npy"
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/freehand_replay_res2
    free_hand_replay_fn = os.path.join(freehand_folder, "ts_to_real_play_info_1.npy")
    free_hand_replay_dict = np.load(free_hand_replay_fn, allow_pickle=True).item()
    print(free_hand_replay_dict.keys())
    tot_tses = list(free_hand_replay_dict.keys())
    tot_tses = [cur_ts for cur_ts in tot_tses if isinstance(cur_ts, int)]
    tot_tses = sorted(tot_tses)
    
    tot_arm_with_hand_pos = []
    for cur_ts in tot_tses:
        cur_ts_replay_dict = free_hand_replay_dict[cur_ts]
        cur_ts_arm_pos = cur_ts_replay_dict['real_arm_pos']
        cur_ts_hand_pos_to_sim = cur_ts_replay_dict['real_leap_pos_to_sim']
        cur_ts_arm_with_hand_pos = np.concatenate([cur_ts_arm_pos, cur_ts_hand_pos_to_sim], axis=0)
        tot_arm_with_hand_pos.append(cur_ts_arm_with_hand_pos)
    tot_arm_with_hand_pos = np.stack(tot_arm_with_hand_pos, axis=0)
    return tot_arm_with_hand_pos
    
    test_ts = 1
    test_free_hand_replay_dict = free_hand_replay_dict[test_ts]
    print(f"test_free_hand_replay_dict: {test_free_hand_replay_dict.keys()}")
    for cur_key in test_free_hand_replay_dict:
        cur_val = test_free_hand_replay_dict[cur_key]
        print(f"key: {cur_key}, val: {cur_val.shape}")

def test_ts_to_obj_obs_rest_fn():
    real_tot_arm_with_hand_pos = test_free_hand_replay_fn()
    ts_to_obj_obs_reset_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-29-47/ts_to_hand_obj_obs_reset_1.npy"
    ts_to_obj_obs_reset = np.load(ts_to_obj_obs_reset_fn, allow_pickle=True).item()
    print(ts_to_obj_obs_reset.keys())
    # for key in ts_to_obj_obs_reset:
    
    tot_tses = list(ts_to_obj_obs_reset.keys())
    tot_tses = [cur_ts for cur_ts in tot_tses if isinstance(cur_ts, int)]
    tot_tses = sorted(tot_tses)
    
    tot_env_tot_hand_states = []
    for cur_ts in tot_tses:
        cur_ts_obj_obs_reset_dict = ts_to_obj_obs_reset[cur_ts]
        cur_ts_hand_states = cur_ts_obj_obs_reset_dict['shadow_hand_dof_pos']
        tot_env_tot_hand_states.append(cur_ts_hand_states)
    
    tot_env_tot_hand_states = np.stack(tot_env_tot_hand_states, axis=1) # nn_envs x nn_ts x nn_dofs
    minn_ts = min(tot_env_tot_hand_states.shape[1], real_tot_arm_with_hand_pos.shape[0])
    real_tot_arm_with_hand_pos = real_tot_arm_with_hand_pos[: minn_ts]
    tot_env_tot_hand_states = tot_env_tot_hand_states[:, : minn_ts]
    
    diff_tot_env_tot_hand_states = tot_env_tot_hand_states - real_tot_arm_with_hand_pos[None]
    diff_tot_env_tot_hand_states = np.abs(diff_tot_env_tot_hand_states)
    diff_tot_env_tot_hand_states = np.sum(diff_tot_env_tot_hand_states, axis=-1)
    diff_tot_env_tot_hand_states = np.mean(diff_tot_env_tot_hand_states, axis=1)
    minn, minn_idx = np.min(diff_tot_env_tot_hand_states), np.argmin(diff_tot_env_tot_hand_states)
    print(f"minn: {minn}, minn_idx: {minn_idx}")
    
    minn_env_tot_hand_states = tot_env_tot_hand_states[minn_idx]
    # minn_env_tot_hand_states --- nn_ts x 
    # 2004 ; p = 210; d = 20
    # for cur_ts in range(minn_env_tot_hand_states.shape[0]):
    #     cur_ts_hand_states = minn_env_tot_hand_states[cur_ts]
    #     print(cur_ts_hand_states)
    #     print(real_tot_arm_with_hand_pos[cur_ts])
    #     print()
    
    # test_ts = 1
    # test_free_hand_replay_dict = ts_to_obj_obs_reset[test_ts]
    # print(f"test_free_hand_replay_dict: {test_free_hand_replay_dict.keys()}")
    
    # for cur_key in test_free_hand_replay_dict:
    #     cur_val = test_free_hand_replay_dict[cur_key]
    #     print(f"key: {cur_key}, val: {cur_val.shape}")
    
    
def insepct_inst_tag_list_file():
    inst_tag_list_file_fn = "/root/diffsim/IsaacGymEnvs2/assets/inst_tag_list_s9.npy"
    inst_tag_list = np.load(inst_tag_list_file_fn, allow_pickle=True).item()
    for key in inst_tag_list:
        val = inst_tag_list[key]
        print(f"key: {key}, val: {val}")

def get_inst_tag_list_by_obj(obj_type_nm='duck'):
    tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    # passive_active_info_ori_grab_s1_alarmclock_lift_nf_300.npy
    inst_st_tag = "passive_active_info_"
    tot_inst_fns = os.listdir(tracking_data_sv_root)
    tot_inst_fns = [
        fn for fn in tot_inst_fns if fn.endswith(".npy") and fn[: len(inst_st_tag)] == inst_st_tag
    ]
    tot_inst_tags = [
        fn.split(".")[0][len(inst_st_tag): ] for fn in tot_inst_fns
    ]
    tot_inst_tags = [
        fn for fn in tot_inst_tags if obj_type_nm in fn
    ]
    tot_inst_tags_dict = {
        cur_inst_tag : 1 for cur_inst_tag in tot_inst_tags
    }
    tot_inst_tag_sv_fn = f"inst_tag_list_obj_{obj_type_nm}.npy"
    tot_inst_tag_sv_fn = os.path.join("../assets", tot_inst_tag_sv_fn)
    np.save(tot_inst_tag_sv_fn, tot_inst_tags_dict)
    print(f"inst tags saved to {tot_inst_tag_sv_fn}")
    


def inspect_multiple_inst_traj(multiple_inst_traj_fn):
    multiple_inst_traj = np.load(multiple_inst_traj_fn, allow_pickle=True).item()
    print(multiple_inst_traj.keys()) 
    for key in multiple_inst_traj: # nn_instances x nn_traj_len x nn_dofs (for hand) / nn_pose_dim (for obj)
        cur_val = multiple_inst_traj[key]
        print(f"key: {key}, val: {cur_val.shape}")
        # 


def compose_multiple_kine_trajs_hybrid(modified_kine_root_folder):
    # /data/xueyi/data/modified_kinematics_data_vrandsamplesv2/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_duck_inspect_1_v11/passive_active_info_ori_grab_s1_duck_inspect_1_nf_300_sample_4.npy
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2"
    # /data/xueyi/data/modified_kinematics_data_vrandsamplesv2_hybrid/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s5_duck_inspect_1_ori_grab_s1_duck_pass_1_v12
    tot_subfolders = os.listdir(modified_kine_root_folder)
    subfolder_st_tag = "GRAB_Tracking_PK_reduced_300_resampled_"
    tot_subfolders = [
        fn for fn in tot_subfolders if fn[: len(subfolder_st_tag)] == subfolder_st_tag and os.path.isdir(os.path.join(modified_kine_root_folder, fn))
    ]
    # tot_subfolders = [
    #     "_".join(fn[len(subfolder_st_tag): ].split("_")[:-1]) for fn in tot_subfolders # get all the subfolder --- inst tags 
    # ]
    maxx_ws = 300
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_peel_1_nf_300.npy
    # ori_tracking_data_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    for cur_subfolder in tot_subfolders:
        
        # cur_inst_subfoldr = f"GRAB_Tracking_PK_reduced_300_resampled_{cur_inst_tag}_v11"
        cur_full_inst_subfolder = os.path.join(modified_kine_root_folder, cur_subfolder)
        
        
        # cur_inst_tracking_data_fn = f"passive_active_info_{cur_inst_tag}_nf_300.npy"
        # cur_inst_tracking_data_fn = os.path.join(ori_tracking_data_sv_root, cur_inst_tracking_data_fn) # trackking data fn 
        # cur_inst_tracking_data = np.load(cur_inst_tracking_data_fn , allow_pickle=True).item()
        
        # cur_inst_tracking_kine_hand_qs = cur_inst_tracking_data['robot_delta_states_weights_np'] # nn_ts x nn_hand_qs #
        
        cur_inst_sampled_hand_qs = []
        cur_inst_sampled_obj_pos = []
        cur_inst_sampled_obj_rot = []
        cur_inst_sampled_base_hand_qs = []
        
        # /data/xueyi/data/modified_kinematics_data_vrandsamplesv2/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_duck_inspect_1_v11/passive_active_info_ori_grab_s1_duck_inspect_1_nf_300_sample_11.npy
        tot_sampled_instances = os.listdir(cur_full_inst_subfolder)
        tot_sampled_instances = [fn for fn in tot_sampled_instances if fn.endswith(".npy") and "_merged_samples" not in fn]
        for cur_sampled_inst_fn in tot_sampled_instances:
            cur_full_sampled_inst_fn = os.path.join(cur_full_inst_subfolder, cur_sampled_inst_fn) # cur sampled inst fn #
            # print(f"")
            cur_full_sampled_inst = np.load(cur_full_sampled_inst_fn, allow_pickle=True).item() # get thesampled inst tag #
            # dict_keys(['hand_qs', 'obj_pos', 'obj_rot', 'robot_delta_states_weights_np', 'object_transl', 'object_rot_quat', 'link_key_to_link_pos'])
            # print(f"cur_full_sampled_inst: {cur_full_sampled_inst.keys()}")
            cur_sampled_hand_qs = cur_full_sampled_inst['robot_delta_states_weights_np']
            cur_sampled_obj_pos = cur_full_sampled_inst['object_transl']
            cur_sampled_obj_ornt = cur_full_sampled_inst['object_rot_quat']
            # nn_ts x nn_dofs # 
            # if use_ori_traj_as_preoptres:
            #     cur_sampled_base_hand_qs= cur_inst_tracking_kine_hand_qs
            # else:
            cur_sampled_base_hand_qs = cur_sampled_hand_qs
            
            cur_inst_sampled_hand_qs.append(cur_sampled_hand_qs[: maxx_ws])
            cur_inst_sampled_obj_pos.append(cur_sampled_obj_pos[: maxx_ws])
            cur_inst_sampled_obj_rot.append(cur_sampled_obj_ornt[: maxx_ws])
            cur_inst_sampled_base_hand_qs.append(cur_sampled_base_hand_qs[: maxx_ws])
        
        cur_inst_sampled_hand_qs = np.stack(cur_inst_sampled_hand_qs, axis=0)
        cur_inst_sampled_obj_pos = np.stack(cur_inst_sampled_obj_pos, axis=0)
        cur_inst_sampled_obj_rot = np.stack(cur_inst_sampled_obj_rot, axis=0)
        cur_inst_sampled_base_hand_qs = np.stack(cur_inst_sampled_base_hand_qs, axis=0)
        
        print(f"cur_inst_sampled_hand_qs: {cur_inst_sampled_hand_qs.shape}, cur_inst_sampled_obj_pos: {cur_inst_sampled_obj_pos.shape}, cur_inst_sampled_obj_rot: {cur_inst_sampled_obj_rot.shape}, cur_inst_sampled_base_hand_qs: {cur_inst_sampled_base_hand_qs.shape}")
        
        # if use_ori_traj_as_preoptres:
        #     cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples.npy"
        # else:
        #     cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples_v22.npy"
        cur_inst_merged_info_sv_fn = f"merged_samples_v22.npy"
        cur_inst_merged_info_sv_fn = os.path.join(cur_full_inst_subfolder, cur_inst_merged_info_sv_fn)
        cur_inst_merged_samples_dict = {
            'hand_qs': cur_inst_sampled_hand_qs,
            'obj_pos': cur_inst_sampled_obj_pos,
            'obj_rot': cur_inst_sampled_obj_rot,
            'preopt_res': cur_inst_sampled_base_hand_qs,
        }
        np.save(cur_inst_merged_info_sv_fn, cur_inst_merged_samples_dict)
        print(f"saved to {cur_inst_merged_info_sv_fn}")
        # print(cur_inst_tracking_data.keys())
      

def compose_multiple_kine_trajs(use_ori_traj_as_preoptres=True):
    # /data/xueyi/data/modified_kinematics_data_vrandsamplesv2/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_duck_inspect_1_v11/passive_active_info_ori_grab_s1_duck_inspect_1_nf_300_sample_4.npy
    modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2"
    tot_subfolders = os.listdir(modified_kine_root_folder)
    subfolder_st_tag = "GRAB_Tracking_PK_reduced_300_resampled_"
    tot_subfolders = [
        fn for fn in tot_subfolders if fn[: len(subfolder_st_tag)] == subfolder_st_tag and os.path.isdir(os.path.join(modified_kine_root_folder, fn))
    ]
    tot_subfolders = [
        "_".join(fn[len(subfolder_st_tag): ].split("_")[:-1]) for fn in tot_subfolders # get all the subfolder --- inst tags 
    ]
    maxx_ws = 300
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_peel_1_nf_300.npy
    ori_tracking_data_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    for cur_inst_tag in tot_subfolders:
        cur_inst_subfoldr = f"GRAB_Tracking_PK_reduced_300_resampled_{cur_inst_tag}_v11"
        cur_full_inst_subfolder = os.path.join(modified_kine_root_folder, cur_inst_subfoldr)
        
        
        cur_inst_tracking_data_fn = f"passive_active_info_{cur_inst_tag}_nf_300.npy"
        cur_inst_tracking_data_fn = os.path.join(ori_tracking_data_sv_root, cur_inst_tracking_data_fn) # trackking data fn 
        cur_inst_tracking_data = np.load(cur_inst_tracking_data_fn , allow_pickle=True).item()
        
        cur_inst_tracking_kine_hand_qs = cur_inst_tracking_data['robot_delta_states_weights_np'] # nn_ts x nn_hand_qs #
        
        cur_inst_sampled_hand_qs = []
        cur_inst_sampled_obj_pos = []
        cur_inst_sampled_obj_rot = []
        cur_inst_sampled_base_hand_qs = []
        
        # /data/xueyi/data/modified_kinematics_data_vrandsamplesv2/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_duck_inspect_1_v11/passive_active_info_ori_grab_s1_duck_inspect_1_nf_300_sample_11.npy
        tot_sampled_instances = os.listdir(cur_full_inst_subfolder)
        tot_sampled_instances = [fn for fn in tot_sampled_instances if fn.endswith(".npy") and "_merged_samples" not in fn]
        for cur_sampled_inst_fn in tot_sampled_instances:
            cur_full_sampled_inst_fn = os.path.join(cur_full_inst_subfolder, cur_sampled_inst_fn) # cur sampled inst fn #
            # print(f"")
            cur_full_sampled_inst = np.load(cur_full_sampled_inst_fn, allow_pickle=True).item() # get thesampled inst tag #
            # dict_keys(['hand_qs', 'obj_pos', 'obj_rot', 'robot_delta_states_weights_np', 'object_transl', 'object_rot_quat', 'link_key_to_link_pos'])
            # print(f"cur_full_sampled_inst: {cur_full_sampled_inst.keys()}")
            cur_sampled_hand_qs = cur_full_sampled_inst['robot_delta_states_weights_np']
            cur_sampled_obj_pos = cur_full_sampled_inst['object_transl']
            cur_sampled_obj_ornt = cur_full_sampled_inst['object_rot_quat']
            # nn_ts x nn_dofs # 
            if use_ori_traj_as_preoptres:
                cur_sampled_base_hand_qs= cur_inst_tracking_kine_hand_qs
            else:
                cur_sampled_base_hand_qs = cur_sampled_hand_qs
            
            cur_inst_sampled_hand_qs.append(cur_sampled_hand_qs[: maxx_ws])
            cur_inst_sampled_obj_pos.append(cur_sampled_obj_pos[: maxx_ws])
            cur_inst_sampled_obj_rot.append(cur_sampled_obj_ornt[: maxx_ws])
            cur_inst_sampled_base_hand_qs.append(cur_sampled_base_hand_qs[: maxx_ws])
        
        cur_inst_sampled_hand_qs = np.stack(cur_inst_sampled_hand_qs, axis=0)
        cur_inst_sampled_obj_pos = np.stack(cur_inst_sampled_obj_pos, axis=0)
        cur_inst_sampled_obj_rot = np.stack(cur_inst_sampled_obj_rot, axis=0)
        cur_inst_sampled_base_hand_qs = np.stack(cur_inst_sampled_base_hand_qs, axis=0)
        
        print(f"cur_inst_sampled_hand_qs: {cur_inst_sampled_hand_qs.shape}, cur_inst_sampled_obj_pos: {cur_inst_sampled_obj_pos.shape}, cur_inst_sampled_obj_rot: {cur_inst_sampled_obj_rot.shape}, cur_inst_sampled_base_hand_qs: {cur_inst_sampled_base_hand_qs.shape}")
        
        if use_ori_traj_as_preoptres:
            cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples.npy"
        else:
            cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples_v22.npy"
        cur_inst_merged_info_sv_fn = os.path.join(cur_full_inst_subfolder, cur_inst_merged_info_sv_fn)
        cur_inst_merged_samples_dict = {
            'hand_qs': cur_inst_sampled_hand_qs,
            'obj_pos': cur_inst_sampled_obj_pos,
            'obj_rot': cur_inst_sampled_obj_rot,
            'preopt_res': cur_inst_sampled_base_hand_qs,
        }
        np.save(cur_inst_merged_info_sv_fn, cur_inst_merged_samples_dict)
        print(f"saved to {cur_inst_merged_info_sv_fn}")
        # print(cur_inst_tracking_data.keys())
       


def compose_tot_multiple_kine_trajs_hybrid(modified_kine_root_folder):
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2"
    tot_subfolders = os.listdir(modified_kine_root_folder)
    subfolder_st_tag = "GRAB_Tracking_PK_reduced_300_resampled_"
    tot_subfolders = [
        fn for fn in tot_subfolders if fn[: len(subfolder_st_tag)] == subfolder_st_tag and os.path.isdir(os.path.join(modified_kine_root_folder, fn))
    ]
    # tot_subfolders = [
    #     "_".join(fn[len(subfolder_st_tag): ].split("_")[:-1]) for fn in tot_subfolders # get all the subfolder --- inst tags 
    # ]
    # maxx_ws = 300
    tot_tot_sampled_instances_fn = []
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_peel_1_nf_300.npy
    # ori_tracking_data_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    for cur_subfolder in tot_subfolders:
        # cur_inst_subfoldr = f"GRAB_Tracking_PK_reduced_300_resampled_{cur_inst_tag}_v11" # inst subfolder
        cur_full_inst_subfolder = os.path.join(modified_kine_root_folder, cur_subfolder)
        tot_sampled_instances = os.listdir(cur_full_inst_subfolder)
        tot_sampled_instances = [fn for fn in tot_sampled_instances if fn.endswith(".npy") and "merged_samples" in fn]
        
        tot_tot_sampled_instances_fn.append(os.path.join(cur_full_inst_subfolder, tot_sampled_instances[0]))
    
    tot_tot_sampled_dict = {}
    
    for cur_sampled_fn in tot_tot_sampled_instances_fn:
        
        cur_sampled_dict = np.load(cur_sampled_fn, allow_pickle=True).item()
        for key in cur_sampled_dict:
            if key not in tot_tot_sampled_dict:
                tot_tot_sampled_dict[key] = []
            tot_tot_sampled_dict[key].append(cur_sampled_dict[key])
        
        
    for key in tot_tot_sampled_dict:
        tot_tot_sampled_dict[key] = np.concatenate(tot_tot_sampled_dict[key], axis=0)
    
    tot_sample_saved_fn = os.path.join(modified_kine_root_folder, "tot_tot_sampled_dict.npy")
    np.save(tot_sample_saved_fn, tot_tot_sampled_dict)
    print(f"saved to {tot_sample_saved_fn}")

def compose_tot_multiple_kine_trajs(use_ori_traj_as_preoptres=True):
    modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2"
    tot_subfolders = os.listdir(modified_kine_root_folder)
    subfolder_st_tag = "GRAB_Tracking_PK_reduced_300_resampled_"
    tot_subfolders = [
        fn for fn in tot_subfolders if fn[: len(subfolder_st_tag)] == subfolder_st_tag and os.path.isdir(os.path.join(modified_kine_root_folder, fn))
    ]
    tot_subfolders = [
        "_".join(fn[len(subfolder_st_tag): ].split("_")[:-1]) for fn in tot_subfolders # get all the subfolder --- inst tags 
    ]
    maxx_ws = 300
    tot_tot_sampled_instances_fn = []
    # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_banana_peel_1_nf_300.npy
    ori_tracking_data_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    for cur_inst_tag in tot_subfolders:
        cur_inst_subfoldr = f"GRAB_Tracking_PK_reduced_300_resampled_{cur_inst_tag}_v11" # inst subfolder
        cur_full_inst_subfolder = os.path.join(modified_kine_root_folder, cur_inst_subfoldr)
        tot_sampled_instances = os.listdir(cur_full_inst_subfolder)
        tot_sampled_instances = [fn for fn in tot_sampled_instances if fn.endswith(".npy") and "_merged_samples" in fn]
        
        tot_tot_sampled_instances_fn.append(os.path.join(cur_full_inst_subfolder, tot_sampled_instances[0]))
    
    tot_tot_sampled_dict = {}
    
    for cur_sampled_fn in tot_tot_sampled_instances_fn:
        
        cur_sampled_dict = np.load(cur_sampled_fn, allow_pickle=True).item()
        for key in cur_sampled_dict:
            if key not in tot_tot_sampled_dict:
                tot_tot_sampled_dict[key] = []
            tot_tot_sampled_dict[key].append(cur_sampled_dict[key])
        
        
    for key in tot_tot_sampled_dict:
        tot_tot_sampled_dict[key] = np.concatenate(tot_tot_sampled_dict[key], axis=0)
    
    tot_sample_saved_fn = os.path.join(modified_kine_root_folder, "tot_tot_sampled_dict.npy")
    np.save(tot_sample_saved_fn, tot_tot_sampled_dict)
    print(f"saved to {tot_sample_saved_fn}")
        
        
        # cur_inst_tracking_data_fn = f"passive_active_info_{cur_inst_tag}_nf_300.npy"
        # cur_inst_tracking_data_fn = os.path.join(ori_tracking_data_sv_root, cur_inst_tracking_data_fn) # trackking data fn 
        # cur_inst_tracking_data = np.load(cur_inst_tracking_data_fn , allow_pickle=True).item()
        
        # cur_inst_tracking_kine_hand_qs = cur_inst_tracking_data['robot_delta_states_weights_np'] # nn_ts x nn_hand_qs #
        
        # cur_inst_sampled_hand_qs = []
        # cur_inst_sampled_obj_pos = []
        # cur_inst_sampled_obj_rot = []
        # cur_inst_sampled_base_hand_qs = []
        
        # # /data/xueyi/data/modified_kinematics_data_vrandsamplesv2/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_duck_inspect_1_v11/passive_active_info_ori_grab_s1_duck_inspect_1_nf_300_sample_11.npy
        # tot_sampled_instances = os.listdir(cur_full_inst_subfolder)
        # tot_sampled_instances = [fn for fn in tot_sampled_instances if fn.endswith(".npy") and "_merged_samples" not in fn]
        # for cur_sampled_inst_fn in tot_sampled_instances:
        #     cur_full_sampled_inst_fn = os.path.join(cur_full_inst_subfolder, cur_sampled_inst_fn) # cur sampled inst fn #
        #     # print(f"")
        #     cur_full_sampled_inst = np.load(cur_full_sampled_inst_fn, allow_pickle=True).item() # get thesampled inst tag #
        #     # dict_keys(['hand_qs', 'obj_pos', 'obj_rot', 'robot_delta_states_weights_np', 'object_transl', 'object_rot_quat', 'link_key_to_link_pos'])
        #     # print(f"cur_full_sampled_inst: {cur_full_sampled_inst.keys()}")
        #     cur_sampled_hand_qs = cur_full_sampled_inst['robot_delta_states_weights_np']
        #     cur_sampled_obj_pos = cur_full_sampled_inst['object_transl']
        #     cur_sampled_obj_ornt = cur_full_sampled_inst['object_rot_quat']
        #     # nn_ts x nn_dofs # 
        #     if use_ori_traj_as_preoptres:
        #         cur_sampled_base_hand_qs= cur_inst_tracking_kine_hand_qs
        #     else:
        #         cur_sampled_base_hand_qs = cur_sampled_hand_qs
            
        #     cur_inst_sampled_hand_qs.append(cur_sampled_hand_qs[: maxx_ws])
        #     cur_inst_sampled_obj_pos.append(cur_sampled_obj_pos[: maxx_ws])
        #     cur_inst_sampled_obj_rot.append(cur_sampled_obj_ornt[: maxx_ws])
        #     cur_inst_sampled_base_hand_qs.append(cur_sampled_base_hand_qs[: maxx_ws])
        
        # cur_inst_sampled_hand_qs = np.stack(cur_inst_sampled_hand_qs, axis=0)
        # cur_inst_sampled_obj_pos = np.stack(cur_inst_sampled_obj_pos, axis=0)
        # cur_inst_sampled_obj_rot = np.stack(cur_inst_sampled_obj_rot, axis=0)
        # cur_inst_sampled_base_hand_qs = np.stack(cur_inst_sampled_base_hand_qs, axis=0)
        
        # print(f"cur_inst_sampled_hand_qs: {cur_inst_sampled_hand_qs.shape}, cur_inst_sampled_obj_pos: {cur_inst_sampled_obj_pos.shape}, cur_inst_sampled_obj_rot: {cur_inst_sampled_obj_rot.shape}, cur_inst_sampled_base_hand_qs: {cur_inst_sampled_base_hand_qs.shape}")
        
        # if use_ori_traj_as_preoptres:
        #     cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples.npy"
        # else:
        #     cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples_v22.npy"
        # cur_inst_merged_info_sv_fn = os.path.join(cur_full_inst_subfolder, cur_inst_merged_info_sv_fn)
        # cur_inst_merged_samples_dict = {
        #     'hand_qs': cur_inst_sampled_hand_qs,
        #     'obj_pos': cur_inst_sampled_obj_pos,
        #     'obj_rot': cur_inst_sampled_obj_rot,
        #     'preopt_res': cur_inst_sampled_base_hand_qs,
        # }
        # np.save(cur_inst_merged_info_sv_fn, cur_inst_merged_samples_dict)
        # print(f"saved to {cur_inst_merged_info_sv_fn}")
        # # print(cur_inst_tracking_data.keys())


def test_real_saved_selected_res():
    # real_selected_res_fn = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/selectedres_lightbulb_reorient/ts_to_real_play_info_4.npy"
    real_selected_res_fn = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/selected_res_cubesmall_inspect_1/ts_to_real_play_info_1.npy"
    real_selected_res = np.load(real_selected_res_fn, allow_pickle=True).item()
    print(real_selected_res.keys())
    st_res_key = 1
    real_selected_res_cur_ts = real_selected_res[st_res_key]
    for key in real_selected_res_cur_ts:
        val = real_selected_res_cur_ts[key]
        val = torch.from_numpy(val)
        print(f"key: {key}, val: {val}")

def get_new_pk_retar_res():
    pk_retar_res_fn = "/cephfs/xueyi/data/TACO_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data/leap_passive_active_info_ori_grab_taco_20231104_075_nf_300.npy"
    pk_retar_res_fn = "/cephfs/xueyi/data/TACO_Tracking_PK_LEAP_OFFSET_0d4_0d5_warm_v2_v2urdf/data/leap_passive_active_info_ori_grab_taco_20231103_118_nf_300.npy"
    pk_retar_res = np.load(pk_retar_res_fn, allow_pickle=True).item()
    to_remove_keys = ['link_key_to_link_pos']
    new_pk_retar_res = {
        key : pk_retar_res[key] for key in pk_retar_res if key not in to_remove_keys
    }
    new_pk_retar_res_fn = pk_retar_res_fn.replace(".npy", "_removed.npy")
    np.save(new_pk_retar_res_fn, new_pk_retar_res) 
    print(f"saved to {new_pk_retar_res_fn}")
    pass


def save_replay_obj_type_to_fn_dict():
    obj_type_to_fn_dict = {
        'ori_grab_s1_watch_set_2': "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_watch_set_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-16-56/ts_to_hand_obj_obs_reset_1.npy",
        "ori_grab_s2_hammer_use_2": "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_hammer_use_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-12-48/ts_to_hand_obj_obs_reset_1.npy",
        "ori_grab_s1_waterbottle_pour_1": "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_waterbottle_pour_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-21-11/ts_to_hand_obj_obs_reset_1.npy",
        "ori_grab_s2_apple_eat_1": "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_apple_eat_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-27-57/ts_to_hand_obj_obs_reset_1.npy",
        "ori_grab_s1_lightbulb_pass_1": "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_lightbulb_pass_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-23-30-18/ts_to_hand_obj_obs_reset_1.npy"
    }
    np.save("../assets/obj_type_to_fn_dict.npy", obj_type_to_fn_dict)


def inspect_grab_test_res():
    # grab_test_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.6024_rot_0.6024_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-11-22-45/ts_to_hand_obj_obs_reset_1.npy" # object conditions 
    # grab_test_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.6024_rot_0.6024_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-11-50-12/ts_to_hand_obj_obs_reset_1.npy" # hand conditions
    # grab_test_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.6024_rot_0.6024_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-16-44-33/ts_to_hand_obj_obs_reset_1.npy" # hand condition
    # grab_test_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.6024_rot_0.6024_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-17-05-40/ts_to_hand_obj_obs_reset_1.npy"
    grab_test_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.6024_rot_0.6024_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_07-22-49-17/ts_to_hand_obj_obs_reset_1.npy"
    
    grab_test_dict = np.load(grab_test_res_fn, allow_pickle=True).item()
    print(f"grab_test_dict: {grab_test_dict.keys()}")
    test_ts = 1
    test_ts_grab_test_dict = grab_test_dict[test_ts]
    for key in test_ts_grab_test_dict:
        cur_val = test_ts_grab_test_dict[key]
        print(f"key: {key}, val: {cur_val.shape}")
    tot_ts_pos_diff = []
    tot_test_ts = list(grab_test_dict.keys())
    tot_test_ts = [ int(cur_ts) for cur_ts in tot_test_ts if isinstance(cur_ts, int)]
    for cur_ts in tot_test_ts:
        cur_ts_grab_test_dict = grab_test_dict[cur_ts]
        # cur_ts #
        
        ref_goal_pose = cur_ts_grab_test_dict['goal_pose_ref_np']
        actual_goal_pose = cur_ts_grab_test_dict['object_pose']
        ref_goal_pos = ref_goal_pose[..., :3]
        actual_goal_pos = actual_goal_pose[..., :3] # get the actual goal pos
        
        # ref_goal_pose = cur_ts_grab_test_dict['next_ref_np']
        # actual_goal_pose = cur_ts_grab_test_dict['shadow_hand_dof_pos']
        # ref_goal_pos = ref_goal_pose
        # actual_goal_pos = actual_goal_pose
        
        ref_actual_diff_pos = np.linalg.norm(ref_goal_pos - actual_goal_pos, axis=-1) # (nn_envs, )
        
        
        
        ref_actual_diff_pos = np.mean(ref_actual_diff_pos)
        tot_ts_pos_diff.append(ref_actual_diff_pos.item()) # tot ts pos diff 

    avg_tot_pos_diff =  np.mean(tot_ts_pos_diff)
    print(f"avg_tot_pos_diff: {avg_tot_pos_diff}")
    
    # obj hand #
    # 0.024918535742622726, 1.8419551689887923 # obj 
    # 0.02890013746748483, 1.517915809134576 # hand 
    
    # 0.10622643416619793, 2.111735911871677 # hand cond
    # 0.052452342581655466 1.9737175261994269 # obj cond
    
    # 0.052452342581655466,  1.9737175261994269


def compose_all_duck_sequences():
    # /cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_eyeglasses_wear_1_nf_300.npy
    tracking_data_sv_root = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/"
    tot_data_fn = os.listdir(tracking_data_sv_root)
    tot_data_fn = [ fn for fn in tot_data_fn if fn.endswith(".npy") and "duck" in fn ]
    tot_hand_qs = []
    tot_obj_pos = []
    tot_obj_ornt = []
    for cur_fn in tot_data_fn:
        cur_full_fn = os.path.join(tracking_data_sv_root, cur_fn)
        cur_data_dict =     np.load(cur_full_fn, allow_pickle=True).item()
        cur_hand_qs = cur_data_dict['robot_delta_states_weights_np']
        cur_obj_pos = cur_data_dict['object_transl']
        cur_obj_ornt = cur_data_dict['object_rot_quat']
        tot_hand_qs.append(cur_hand_qs)
        tot_obj_pos.append(cur_obj_pos)
        tot_obj_ornt.append(cur_obj_ornt)
    tot_hand_qs = np.stack(tot_hand_qs, axis=0)
    tot_obj_pos = np.stack(tot_obj_pos, axis=0)
    tot_obj_ornt = np.stack(tot_obj_ornt, axis=0)
    tot_data_dict = {
        'hand_qs': tot_hand_qs,
        'obj_pos': tot_obj_pos,
        'obj_rot': tot_obj_ornt,
        'preopt_res': tot_hand_qs
    }
    new_target_sample_fn = "./data/new_tot_duck_data.npy"
    np.save(new_target_sample_fn, tot_data_dict)
    pass

def test_inst_res():
    tot_rnd_samples_fn = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2/tot_tot_sampled_dict.npy"
    tot_rnd_samples_fn = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s4_duck_inspect_1_v11/passive_active_info_ori_grab_s4_duck_inspect_1_nf_300_merged_samples_v22.npy"
    target_sample_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s4_duck_inspect_1_nf_300.npy"
    tot_rnd_samples_dict = np.load(tot_rnd_samples_fn, allow_pickle=True).item()
    print(f"tot_rnd_samples_dict: {tot_rnd_samples_dict.keys()}")
    
    target_sample_dict = np.load(target_sample_fn, allow_pickle=True).item()
    # print(f"target_sample_dict: {target_sample_dict.keys()}")
    for key in target_sample_dict:
        target_val = target_sample_dict[key]
        print(f"key: {key}, val: {target_val.shape}")

    for key in tot_rnd_samples_dict:
        cur_val = tot_rnd_samples_dict[key]
        print(f"key: {key}, val: {cur_val.shape}")
    
    target_hand_qs = target_sample_dict['robot_delta_states_weights_np']
    target_obj_pos = target_sample_dict['object_transl']
    target_obj_ornt= target_sample_dict['object_rot_quat']
    tot_rnd_samples_dict['hand_qs'] = np.concatenate(
        [ 
            target_hand_qs[None], tot_rnd_samples_dict['hand_qs']
        ], axis=0
    )
    tot_rnd_samples_dict['obj_pos'] = np.concatenate(
        [ 
            target_obj_pos[None], tot_rnd_samples_dict['obj_pos']
        ], axis=0
    )
    tot_rnd_samples_dict['obj_rot'] = np.concatenate(
        [ 
            target_obj_ornt[None], tot_rnd_samples_dict['obj_rot']
        ], axis=0
    )
    tot_rnd_samples_dict['preopt_res'] = np.concatenate(
        [ 
            target_hand_qs[None], tot_rnd_samples_dict['preopt_res']
        ], axis=0
    )
    new_target_sample_fn = "./data/new_tot_samples.npy"
    np.save(new_target_sample_fn, tot_rnd_samples_dict)
    print(f"saved to {new_target_sample_fn}")
    
  

def compose_tot_multiple_kine_trajs_reornt_seqs(modified_kine_root_folder, use_ori_traj_as_preoptres=True):
    # /data/xueyi/data/modified_kinematics_data_leap_wfranka_v7urdf_winterp/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s1_airplane_pass_1_v18/leap_passive_active_info_ori_grab_s1_airplane_pass_1_nf_300_sample_3.npy
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v7urdf_winterp"
    tot_subfolders = os.listdir(modified_kine_root_folder)
    subfolder_st_tag = "GRAB_Tracking_PK_reduced_300_resampled_"
    tot_subfolders = [
        fn for fn in tot_subfolders if fn[: len(subfolder_st_tag)] == subfolder_st_tag and os.path.isdir(os.path.join(modified_kine_root_folder, fn))
    ]
    tot_subfolders = [
        "_".join(fn[len(subfolder_st_tag): ].split("_")[:-1]) for fn in tot_subfolders # get all the subfolder --- inst tags 
    ]
    maxx_ws = 450
    # tot_tot_sampled_instances_fn = []
    # ori_tracking_data_sv_root = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data"
    for cur_inst_tag in tot_subfolders:
        cur_inst_subfoldr = f"GRAB_Tracking_PK_reduced_300_resampled_{cur_inst_tag}_v18" # inst subfolder
        cur_full_inst_subfolder = os.path.join(modified_kine_root_folder, cur_inst_subfoldr)
        tot_sampled_instances = os.listdir(cur_full_inst_subfolder)
        
        tot_sampled_instances = [
            fn for fn in tot_sampled_instances if fn.endswith(".npy") and "_merged_samples" not in fn
        ]
        
        cur_inst_sampled_hand_qs = []
        cur_inst_sampled_obj_pos = []
        cur_inst_sampled_obj_rot = []
        cur_inst_sampled_base_hand_qs = []
        
        for cur_sampled_inst_fn in tot_sampled_instances:
            cur_full_sampled_inst_fn = os.path.join(cur_full_inst_subfolder, cur_sampled_inst_fn)
            cur_full_sampled_inst = np.load(cur_full_sampled_inst_fn, allow_pickle=True).item()
            cur_sampled_hand_qs = cur_full_sampled_inst['robot_delta_states_weights_np']
            cur_sampled_obj_pos = cur_full_sampled_inst['object_transl']
            cur_sampled_obj_ornt = cur_full_sampled_inst['object_rot_quat']
            
            cur_inst_sampled_hand_qs.append(cur_sampled_hand_qs[: maxx_ws])
            cur_inst_sampled_obj_pos.append(cur_sampled_obj_pos[: maxx_ws])
            cur_inst_sampled_obj_rot.append(cur_sampled_obj_ornt[: maxx_ws])
            cur_inst_sampled_base_hand_qs.append(cur_sampled_hand_qs[: maxx_ws])
        
        cur_inst_sampled_hand_qs = np.stack(cur_inst_sampled_hand_qs, axis=0)
        cur_inst_sampled_obj_pos = np.stack(cur_inst_sampled_obj_pos, axis=0)
        cur_inst_sampled_obj_rot = np.stack(cur_inst_sampled_obj_rot, axis=0)
        cur_inst_sampled_base_hand_qs = np.stack(cur_inst_sampled_base_hand_qs, axis=0)
        
        print(f"cur_inst_sampled_hand_qs: {cur_inst_sampled_hand_qs.shape}, cur_inst_sampled_obj_pos: {cur_inst_sampled_obj_pos.shape}, cur_inst_sampled_obj_rot: {cur_inst_sampled_obj_rot.shape}, cur_inst_sampled_base_hand_qs: {cur_inst_sampled_base_hand_qs.shape}")
        
        if use_ori_traj_as_preoptres:
            cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples.npy"
        else:
            cur_inst_merged_info_sv_fn = f"passive_active_info_{cur_inst_tag}_nf_300_merged_samples_v22.npy"
        cur_inst_merged_info_sv_fn = os.path.join(cur_full_inst_subfolder, cur_inst_merged_info_sv_fn)
        cur_inst_merged_samples_dict = {
            'hand_qs': cur_inst_sampled_hand_qs,
            'obj_pos': cur_inst_sampled_obj_pos,
            'obj_rot': cur_inst_sampled_obj_rot,
            'preopt_res': cur_inst_sampled_base_hand_qs,
        }
        np.save(cur_inst_merged_info_sv_fn, cur_inst_merged_samples_dict)
        print(f"saved to {cur_inst_merged_info_sv_fn}")
            
        
    #     tot_sampled_instances = [fn for fn in tot_sampled_instances if fn.endswith(".npy") and "_merged_samples" in fn]
        
    #     tot_tot_sampled_instances_fn.append(os.path.join(cur_full_inst_subfolder, tot_sampled_instances[0]))
    
    # tot_tot_sampled_dict = {}
    
    # for cur_sampled_fn in tot_tot_sampled_instances_fn:
        
    #     cur_sampled_dict = np.load(cur_sampled_fn, allow_pickle=True).item()
    #     for key in cur_sampled_dict:
    #         if key not in tot_tot_sampled_dict:
    #             tot_tot_sampled_dict[key] = []
    #         tot_tot_sampled_dict[key].append(cur_sampled_dict[key])
        
        
    # for key in tot_tot_sampled_dict:
    #     tot_tot_sampled_dict[key] = np.concatenate(tot_tot_sampled_dict[key], axis=0)
    
    # tot_sample_saved_fn = os.path.join(modified_kine_root_folder, "tot_tot_sampled_dict.npy")
    # np.save(tot_sample_saved_fn, tot_tot_sampled_dict)
    # print(f"saved to {tot_sample_saved_fn}")
  
def inspect_ts_to_sv_res(ts_to_sv_res_dict_fn):
    ts_to_sv_dict = np.load(ts_to_sv_res_dict_fn, allow_pickle=True).item()
    print(ts_to_sv_dict.keys())
    test_ts = 0
    test_ts_sv_dict = ts_to_sv_dict[test_ts]
    print(test_ts_sv_dict.keys()) 
    # 
    # ['shadow_hand_dof_pos', 'shadow_hand_dof_tars', 'next_ref_np', 'object_pose', 'goal_pose_ref_np', 'progress_buf_np']
    # 
    # ['shadow_hand_dof_pos', 'shadow_hand_dof_tars', 'next_ref_np', 'object_pose', 'goal_pose_ref_np', 'progress_buf_np']
    # shadow hand dof pose #
    # shaodw hand dof pose #
    # TODO: the problem is that even with continuous tiemsteps, you still cannot guarantee that there is no discontinuity in the hand pose #
    # nn_envs x nn_ts #
    tot_ts = ts_to_sv_dict.keys()
    tot_ts = [cur_ts for cur_ts in tot_ts if isinstance(cur_ts, int)]
    tot_ts = sorted(tot_ts)
    
    key_to_tot_ts_res = {}
    
    for cur_ts in tot_ts:
        cur_ts_sv_dict = ts_to_sv_dict[cur_ts]
        
        for key in cur_ts_sv_dict:
            if key not in key_to_tot_ts_res:
                key_to_tot_ts_res[key] = []
            key_to_tot_ts_res[key].append(cur_ts_sv_dict[key]) 
            
    # key to tot ts res #
    for key in key_to_tot_ts_res:
        key_to_tot_ts_res[key] = np.stack(key_to_tot_ts_res[key], axis=1) # nn_envs x nn_ts x nn_feature_dim #
    
    anchor_key = 'shadow_hand_dof_pos'
    cur_key_res = key_to_tot_ts_res[key] # anchor key
    nn_envs = cur_key_res.shape[0] # fro mthe nn_env
    
    progress_buf_np_key = 'progress_buf_np'
    
    maxx_ws = 310
    
    tot_env_key_to_res_list = []
    
    for i_env in range(nn_envs):
        cur_st_idx = 0
        for cur_st_idx in range(key_to_tot_ts_res[progress_buf_np_key].shape[1]):
            cur_idx_progress_id = key_to_tot_ts_res[progress_buf_np_key][i_env, cur_st_idx].item()
            if cur_idx_progress_id == 0:
                break
        # cur st idx # cur st idx #
        cur_ed_idx = cur_st_idx + maxx_ws
        cur_env_key_to_res = {}
        for key in key_to_tot_ts_res:
            cur_key_cur_obj_res = key_to_tot_ts_res[key][i_env, cur_st_idx: cur_ed_idx] # nn_ts x nn_feature_dim
            cur_env_key_to_res[key] = cur_key_cur_obj_res
        # cur_env_key_to_res_fn = f"./data/cur_env_key_to_res_{i_env}.npy" # cur env key to res fn;  #
        tot_key_res = cur_env_key_to_res.keys()
        tot_env_key_to_res_list.append(cur_env_key_to_res)
        pass
    
    # cur_env_key_to_res #
    # cur env key to res #
    # tot key res #
    tot_tot_key_to_res = {}
    for cur_env_key_to_res in tot_env_key_to_res_list:
        for key in cur_env_key_to_res:
            if key not in tot_tot_key_to_res:
                tot_tot_key_to_res[key] = []
            tot_tot_key_to_res[key].append(cur_env_key_to_res[key])
    
    for key in tot_tot_key_to_res:
        tot_tot_key_to_res[key] = np.stack(tot_tot_key_to_res[key], axis=0) # nn_envs x nn_tot_ts x nn_features
        if len(tot_tot_key_to_res[key].shape) == 3:
            tot_tot_key_to_res[key] = np.transpose(tot_tot_key_to_res[key], (1, 0, 2)) # nn_tot_ts x nn_envs x nn_features
        else: # to to key to res #
            tot_tot_key_to_res[key] = np.transpose(tot_tot_key_to_res[key], (1, 0)) # nn_tot_ts x nn_envs
    # 
    
    # for key in key_to_tot_ts_res:
    #     # key to the tot ts res
    #     pass
    processed_res_sv_dict_fn = ts_to_sv_res_dict_fn.replace(".npy", "_processed.npy")
    np.save(processed_res_sv_dict_fn, tot_tot_key_to_res)
    print(f"saved to {processed_res_sv_dict_fn}")
    
    
    # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-18-19-42/ts_to_hand_obj_obs_reset_5700_processed.npy 
    # # # _processed # # # _processed # # #
    # # # _processed # # # _processed # # #

def test_experience_buffer(exp_buffer_fn):
    exp_buffer_dict = np.load(exp_buffer_fn, allow_pickle=True).item()
    print(exp_buffer_dict.keys())
    mus = exp_buffer_dict['mus']
    print(mus.shape)
    # print()
    # 300 100 23
    print(mus[100, 10])
  
import pytorch_kinematics as pk

d = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float64

def compute_link_key_to_link_pose(sample_fn, hand_type='allegro', w_franka=False, w_franka_v2urdf=False):
    sample_dict = np.load(sample_fn, allow_pickle=True).item() # 
    print(f"sample_dict: {sample_dict.keys()}")
    hand_qs = sample_dict['hand_qs'] # hand qs # # hand qs #
    print(f"hand_qs: {hand_qs.shape}")
    
    
    
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
        
        first_tip_name = 'thumb_fingertip'
        second_tip_name = 'fingertip'
        third_tip_name = 'fingertip_2'
        forth_tip_name = 'fingertip_3'      
    else:
        raise ValueError(f"hand type {hand_type} not implemented")
    
    chain = pk.build_chain_from_urdf(open(urdf_fn).read())
    chain = chain.to(dtype=dtype, device=d)
    # get chain #
    hand_qs_th = torch.from_numpy(hand_qs).double().to(d)
    tg_batch = chain.forward_kinematics(hand_qs_th[:]) # nn_ts x nn_hand_dof 
    # convert the rot mtx to or
    link_key_to_link_pose = {}
    for key in tg_batch:
        cur_matrix = tg_batch[key].get_matrix()
        rot_mtx = cur_matrix[:, :3, :3].float()
        trans_mtx =cur_matrix[:, :3, 3].float()
        
        tot_rot_quat = []
        rot_mtx_np = rot_mtx.detach().cpu().numpy()
        for i_ts in range(rot_mtx_np.shape[0]):
            cur_rot_mtx = rot_mtx_np[i_ts]
            cur_rot_quat = R.from_matrix(cur_rot_mtx).as_quat()
            tot_rot_quat.append(cur_rot_quat)
        tot_rot_quat = np.stack(tot_rot_quat, axis=0)
        trans_np = trans_mtx.detach().cpu().numpy()
        trans_rot_np = np.concatenate(
            [ trans_np, tot_rot_quat ], axis=-1
        )
        link_key_to_link_pose[key] = trans_rot_np
        
    sample_dict['link_key_to_link_pose'] = link_key_to_link_pose
    sample_w_pose_sv_fn = sample_fn.replace(".npy", "_w_pose.npy")
    np.save(sample_w_pose_sv_fn, sample_dict)
    print(f"saved to {sample_w_pose_sv_fn}")
        

def search_for_trans_model(ckpt_sv_root):
    import yaml
    tot_ckpt_sv_fns = os.listdir(ckpt_sv_root)
    tot_ckpt_sv_fns = [fn for fn in tot_ckpt_sv_fns if os.path.isdir(os.path.join(ckpt_sv_root, fn))]
    target_tag = '/data/xueyi/data/modified_kinematics_data_leap_wfranka_v16urdf'
    for cur_ckpt_sv_fn in tot_ckpt_sv_fns:
        cur_full_folder = os.path.join(ckpt_sv_root, cur_ckpt_sv_fn)
        cur_config_file = os.path.join(cur_full_folder, "config.yaml")
        if not os.path.exists(cur_config_file):
            continue
        with open(cur_config_file, "r") as rf:
            
            data = yaml.safe_load(rf)
            multiple_kine_source_trajs_fn = data['task']['env']['multiple_kine_source_trajs_fn']
            print(multiple_kine_source_trajs_fn)
            if multiple_kine_source_trajs_fn[: len(target_tag) ] == target_tag:
                print(f"Found! Ckpt root: {cur_full_folder}")
                break
    

def construct_teacher_idx_to_model_weights_ornt_trans():
    teacher_index_to_weights_fn = "../assets/teacher_idx_to_model_weights_ornt_trans.npy"
    teacher_index_to_weights = {
        0: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_ori_grab_s2_hammer_use_2_nf_300_reornt_hodistFalse_mt_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-20-25-12/nn/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth',
        1: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_ori_grab_s2_hammer_use_2_nf_300_reornt_hodistFalse_mt_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-01-39-29/nn/last_tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_800_rew_157.52585.pth'
    }    
    np.save(teacher_index_to_weights_fn, teacher_index_to_weights)
    print(f"teacher index to weights saved to {teacher_index_to_weights_fn}")
    
def inspect_teacher_inst_tags(teacher_inst_tag_fn):
    teacher_inst_tag_dict = np.load(teacher_inst_tag_fn, allow_pickle=True).item()
    print(teacher_inst_tag_dict)


def construct_traj_idx_to_exp_sv_folder():
    traj_idx_to_experience_sv_folder = {
        0: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-31-01',
        1: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-37-58',
        2: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-47-38',
        3: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-56-28',
        4: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-02-58',
        5: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-08-22',
        6: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-16-16',
        7: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-22-16'
    }
    traj_idx_to_experience_sv_folder_sv_fn = "../assets/traj_idx_to_experience_sv_folder.npy"
    np.save(traj_idx_to_experience_sv_folder_sv_fn, traj_idx_to_experience_sv_folder)
    print(f"traj_idx_toxxx saved to {traj_idx_to_experience_sv_folder_sv_fn}")
    return

def construct_traj_idx_to_experiences_sv_dict():
    # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-11-16-43/ts_to_hand_obj_obs_reset_1.npy
    # traj_idx_to_demo_dict_fn = {
    #     0: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-31-01',
    #     1: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-37-58',
    #     2: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-47-38',
    #     3: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-17-56-28',
    #     4: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-02-58',
    #     5: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-08-22',
    #     6: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-16-16',
    #     7: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-18-22-16'
    # }
    
    traj_idx_to_demo_dict_fn = {
        0: '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-11-16-43'
    }
    
    
    # version_idx = 0
    
    version_idx = 1
    
    traj_idx_to_demo_dict_fn = {
        cur_idx: os.path.join(traj_idx_to_demo_dict_fn[cur_idx], "ts_to_hand_obj_obs_reset_1.npy") for cur_idx in traj_idx_to_demo_dict_fn
    }
    traj_idx_to_demo_dict_fn_sv_fn = f"../assets/traj_idx_to_demo_dict_{version_idx}.npy"
    np.save(traj_idx_to_demo_dict_fn_sv_fn, traj_idx_to_demo_dict_fn)


def test_real_reaplay_info(real_replay_fn):
    real_replay_info_dict = np.load(real_replay_fn, allow_pickle=True).item()
    print(real_replay_info_dict.keys())
    test_ts = 1
    real_replay_info_dict = real_replay_info_dict[test_ts]
    print(real_replay_info_dict.keys())
    for key in real_replay_info_dict:
        cur_val = real_replay_info_dict[key]
        cur_val = torch.from_numpy(cur_val).float()
        print(f"key: {key}, val: {cur_val}")


# 0.08726646259971647
# 0.17453292519943295
# 0.2617993877991494
# 0.3490658503988659 # 20 degree
# 0.5235987755982988 # 30 degree
# 0.6981317007977318 # 40 degree




if __name__=='__main__':
    
    real_replay_fn = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/selected_res_hammer_reorient_sample_9/ts_to_real_play_info_1.npy"
    test_real_reaplay_info(real_replay_fn)
    exit(0)
    
    construct_traj_idx_to_experiences_sv_dict()
    exit(0)
    
    construct_traj_idx_to_exp_sv_folder()
    exit(0)
    
    teacher_inst_tag_fn = '../assets/inst_tag_list_s2.npy'
    inspect_teacher_inst_tags(teacher_inst_tag_fn)
    exit(0)
    
    
    construct_teacher_idx_to_model_weights_ornt_trans()
    exit(0)
    
    ckpt_sv_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_ori_grab_s2_hammer_use_2_nf_300_reornt_hodistFalse_mt_wovelFalse/"
    search_for_trans_model(ckpt_sv_root)
    exit(0)
    
    # sample_fn = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v15urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_hammer_use_2_v18/leap_passive_active_info_ori_grab_s2_hammer_use_2_nf_300_sample_0.npy"
    # hand_type = 'leap'
    # w_franka = True
    # w_franka_v2urdf = True
    # compute_link_key_to_link_pose(sample_fn, hand_type=hand_type, w_franka=w_franka, w_franka_v2urdf=w_franka_v2urdf)
    # # compute_link_key_to_link_pose(sample_fn, hand_type='allegro', w_franka=False, w_franka_v2urdf=False):
    # exit(0)
    
    # exp_buffer_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_14-23-47-16/./experience_buffer_sv_0.npy"
    # test_experience_buffer(exp_buffer_fn)
    # exit(0)
    
    # ts_to_sv_res_dict_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_leap_v2urdf_wcustomdamping_v3goal_tsFalse_woftFalse_woftrotvelFalse_dofspeedvel1_ctlfreqinv3_nkinebiasTrue_drFalsev34_rewv7_franka_ori_grab_s2_hammer_use_2_nf_300_table0.5_reornt_hodistFalse_wovelFalse/tracking_ori_grab_s2_apple_lift_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-18-19-42/ts_to_hand_obj_obs_reset_5700.npy"
    # inspect_ts_to_sv_res(ts_to_sv_res_dict_fn)
    # exit(0)
    
    
    # /data/xueyi/data/modified_kinematics_data_leap_wfranka_v15urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_hammer_use_2_v18/passive_active_info_ori_grab_s2_hammer_use_2_nf_300_merged_samples_v22.npy
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v7urdf_winterp"
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v15urdf"
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v16urdf"
    modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v17urdf"
    use_ori_traj_as_preoptres = False
    compose_tot_multiple_kine_trajs_reornt_seqs(modified_kine_root_folder, use_ori_traj_as_preoptres=use_ori_traj_as_preoptres)
    exit(0)
    
    
    use_ori_traj_as_preoptres = False
    compose_tot_multiple_kine_trajs_reornt_seqs(use_ori_traj_as_preoptres=use_ori_traj_as_preoptres)
    exit(0)
    
    
    root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf"
    resave_retargeted_res(root_folder)
    exit(0)
    
    # compose_all_duck_sequences()
    # exit(0)
    
    # test_inst_res()
    # exit(0)
    
    inspect_grab_test_res()
    exit(0)
    
    # save_replay_obj_type_to_fn_dict()
    # exit(0)
    
    # get_new_pk_retar_res()
    # exit(0)
    
    test_real_saved_selected_res()
    exit(0)
    
    # should compare two trajs from different instanfces -- and synthesize their corresponding trajectories #
    
    # compose_tot_multiple_kine_trajs()
    # exit(0)
    
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2_hybrid"
    # compose_multiple_kine_trajs_hybrid(modified_kine_root_folder)
    # exit(0)
    
    # modified_kine_root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamplesv2_hybrid"
    # compose_tot_multiple_kine_trajs_hybrid(modified_kine_root_folder)
    # exit(0)
    
    # use_ori_traj_as_preoptres = True
    # use_ori_traj_as_preoptres = False
    # compose_multiple_kine_trajs(use_ori_traj_as_preoptres=use_ori_traj_as_preoptres)
    # exit(0)
    
    # multiple_inst_traj_fn = '/data/xueyi/uni_manip/expanded_kines/ori_grab_s2_knife_pass_1_forecasted_kine_traj_modifications_nn100_v3.npy'
    # inspect_multiple_inst_traj(multiple_inst_traj_fn)
    # exit(0)
    
    
    # obj_type_nm = 'duck'
    # get_inst_tag_list_by_obj(obj_type_nm)
    # exit(0)
    
    # insepct_inst_tag_list_file()
    # exit(0)
    
    # test_ts_to_obj_obs_rest_fn()
    # exit(0)
    
    # test_free_hand_replay_fn(freehand_folder='freehand_replay_res'):
    # freehand_folder = 'freehand_replay_res'
    # freehand_folder = 'freehand_replay_res2'
    # test_free_hand_replay_fn(freehand_folder=freehand_folder)
    # exit(0)
    
    # root_folder = "/data/xueyi/data/modified_kinematics_data_vrandsamples"
    # inst_starting_tag = "passive_active_info_ori_grab_"
    # resave_retargeted_res(root_folder, inst_starting_tag=inst_starting_tag)
    # exit(0)
    
    # kine_info_dict_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    # sv_samples_folder = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_samples"
    # random_sample_hand_obj_traj(kine_info_dict_fn, sv_samples_folder)
    # exit(0)
    
    # kine_info_dict_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    # random_sample_hand_obj_traj(kine_info_dict_fn)
    # exit(0)
    
    # ts_to_hand_obj_info_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-15-11-10/ts_to_hand_obj_obs_reset_1.npy"
    # inspect_ts_to_reset_infos(ts_to_hand_obj_info_fn)
    # exit(0)
    
    # # real_play_info_fn = "freehand_replay_res/ts_to_real_play_info_2.npy"
    # real_play_info_fn = "freehand_replay_res2/ts_to_real_play_info_9.npy"
    # test_real_play_info(real_play_info_fn)
    # exit(0)
    
    # retargeted_res_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_wrbpos/data/passive_active_info_ori_grab_s9_hammer_use_2_nf_300.npy"
    # test_retargeted_res(retargeted_res_fn)
    # exit(0)
    
    # tot_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_v2_"
    # target_experience_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_v2_sv_experiences"
    # centralize_experience_replay_buffer(tot_folder, target_experience_folder)
    # exit(0)
    
    # 
    # # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_/tracking_ori_grab_s9_hammer_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-12-35-56/ts_to_hand_obj_obs_reset_1_sorted_best.npy
    # # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_/statistics/data_inst_tag_to_best_opt_res_all.npy
    # # specialist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # specialist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # generalist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # specialist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # specialist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_v2_/statistics/data_inst_tag_to_best_opt_res_all.npy" ##### Good results #######
    # # specialist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_fromteacher_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # generalist_test_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_generalist_s1_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # compare_two_opt_res(specialist_test_opt_res_fn, generalist_test_opt_res_fn)
    # # compare_two_opt_res(generalist_test_opt_res_fn, specialist_test_opt_res_fn)
    # exit(0)
    
    # # # # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_/statistics/data_inst_tag_to_optimized_res.npy
    # # # # # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_' 
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_generalist_s1_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_v2_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_fromteacher_'
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_generalist_s2_'
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)
    
    # adding demonstrations to the demonstation buffers --- #
    # cannot get effective reorientation policies #
    
    # # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_generalist_s1_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_v2_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_fromteacher_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_generalist_s2_/statistics/data_inst_tag_to_optimized_res.npy"
    downsample = False
    # downsample = True
    target_inst_tag = None
    # target_inst_tag = "s1_" # if one is better than another --- then it is a better one and we can use that as the demonstrations #
    resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_inst_tag, downsample=downsample)
    exit(0)
    
    
    # inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialist_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_newv2_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_/statistics/data_inst_tag_to_optimized_res.npy"
    # target subj tag #
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_generalist_s1_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_s9specialisttune_s1_v2_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_fromteacher_/statistics/data_inst_tag_to_optimized_res.npy"
    target_subj_tag = ''
    
    random_select = False
    
    # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/statistics/data_inst_tag_to_best_opt_res_all.npy
    
    data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    if len(target_subj_tag) > 0:
        data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    else:
        if random_select:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
        else:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res)
    print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
    exit(0)
    
    
    # test_presaved_experiences_buf()
    # exit(0)
    
    # # tmp_output_res/tot_output_actions.npy #
    # out_action_fn = "tmp_output_res/tot_output_actions.npy"
    
    # test_real_play_file()
    # exit(0)
    
    # test_saved_experiences()
    # exit(0)
    
    # root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf"
    # root_folder = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf"
    # resave_retargeted_res(root_folder)
    # exit(0)
    
    # # inspect_teacher_weights()
    # # exit(0)
    
    # data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_24-03-15-27/ts_to_hand_obj_obs_reset_1.npy"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-21-31-24/ts_to_hand_obj_obs_reset_1.npy"
    # data_inst_tag = "ori_grab_s2_duck_inspect_1_nf_300"
    data_inst_tag = "ori_grab_s2_apple_eat_1_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_watch_set_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-21-40-50/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s2_duck_inspect_1_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_watch_set_2_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_25-21-40-50/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s1_watch_set_2_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s1_waterbottle_pass_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_28-21-03-59/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s1_waterbottle_pass_1_nf_300"
    data_optimized_res_nn= "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/tracking_ori_grab_s2_duck_inspect_1_nf_300_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-02-09-28/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s2_duck_inspect_1_nf_300"
    new_res_save_info = best_optimized_res_reorient(data_optimized_res_nn, data_inst_tag, index=None, downsample=False)
    new_res_save_fn = data_optimized_res_nn.replace(".npy", "_best.npy")
    np.save(new_res_save_fn, new_res_save_info)
    print(f"best res saved to {new_res_save_fn}")
    exit(0)
    
    # reorient_dict_fn = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_duck_inspect_1_v8/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    # reorient_dict_fn = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s5_watch_pass_1_v9/leap_passive_active_info_ori_grab_s5_watch_pass_1_nf_300.npy"
    # test_reorient_dict(reorient_dict_fn)
    # exit(0)
    
    
    resampled_obj_kine_seq_fn = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v2urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s2_duck_inspect_1_v8/leap_passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    resampled_obj_kine_seq_fn = "/data/xueyi/data/modified_kinematics_data_leap_wfranka_v3urdf/GRAB_Tracking_PK_reduced_300_resampled_ori_grab_s5_watch_pass_1_v9/leap_passive_active_info_ori_grab_s5_watch_pass_1_nf_300.npy"
    test_obj_reorient_vel(resampled_obj_kine_seq_fn)
    exit(0)
    
    
    
    test_random_values()
    exit(0)
    
    test_retargeted_res()
    exit(0)
    
    # result_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy"
    # inst_tag = "ori_grab_s2_duck_inspect_1"
    # extract_kinematics_from_tracked_results(result_fn, inst_tag)
    # exit(0)
    
    # test_aaa()
    # exit(0)
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal_franka/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-15-21-36/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal_franka/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-17-42-57/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal_franka/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-17-42-57/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_supv1_0.0_ctlinv_1_all_v3goal_franka_ori_grab_s2_elephant_inspect_1_nf_300/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_06-21-40-24/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
    
    # retar_info_fn = '/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET_warm/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy'
    # retar_info_fn = '/home/xymeow/xueyi/IsaacGymEnvs2/isaacgymenvs/data/GRAB_Tracking_PK_OFFSET_warm/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy'
    # retar_info = np.load(retar_info_fn, allow_pickle=True).item()
    # hand_qs = 'robot_delta_states_weights_np'
    # hand_qs = retar_info[hand_qs]
    # print(hand_qs[0])
    # exit(0)
    
    # test_rotations()
    # exit(0)
    
    # retar_info_fn = './data/GRAB_Tracking_PK_OFFSET_Reduced/data/passive_active_info_ori_grab_s2_duck_inspect_1_nf_300.npy'
    # test_retargeted_info(retar_info_fn)
    # exit(0)
    
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data"
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK_OFFSET/data"
    # reduce_kine_infos(grab_tracking_folder)
    # exit(0)

    
    # result_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_knife_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_04-22-32-15/ts_to_hand_obj_obs_reset_1.npy"
    # inst_tag = "ori_grab_s2_knife_pass_1"
    # extract_kinematics_from_tracked_results(result_fn, inst_tag)
    # exit(0)
    
    # tracked_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_knife_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-11-22-02/ts_to_hand_obj_obs_reset_1.npy"
    # tracked_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_knife_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_03-11-42-52/ts_to_hand_obj_obs_reset_1.npy"
    # ori_kine_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300_resampled/passive_active_info_ori_grab_s2_knife_pass_1_nf_300_sample_14.npy"
    # compare_tracked_ori_trajs(tracked_fn, ori_kine_fn)
    # exit(0)
    
    # teacher_model_version_idx = 4
    # teacher_idx_to_wegights, teacher_idx_to_inst_tags  =  get_teacher_idx_to_teacher_weight_v4()
    # teacher_idx_to_wegights_sv_fn  = os.path.join("../assets", f"teacher_idx_to_wegights_v{teacher_model_version_idx}.npy")
    # np.save(teacher_idx_to_wegights_sv_fn, teacher_idx_to_wegights)
    # teacher_idx_to_inst_tags_sv_fn = os.path.join("../assets", f"teacher_idx_to_inst_tags_v{teacher_model_version_idx}.npy")
    # np.save(teacher_idx_to_inst_tags_sv_fn, teacher_idx_to_inst_tags)
    # print(f"teacher_idx_to_wegights_sv_fn: {teacher_idx_to_wegights_sv_fn}")
    # print(f"teacher_idx_to_inst_tags_sv_fn: {teacher_idx_to_inst_tags_sv_fn}")
    # exit(0)
    
    
    # inst_tag = "ori_grab_s2_knife_pass_1"
    # traj_modification_opt(inst_tag)
    # exit(0)
    
    # select nearest # # #
    
    forecasting_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_knife_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-12-52-23/ts_to_hand_obj_obs_reset_1.npy"
    inst_tag = "ori_grab_s2_knife_pass_1"
    select_nearest_forecasting_res_traj_modifications(forecasting_res_fn, inst_tag)
    exit(0)
    
    inst_tag = "ori_grab_s2_knife_pass_1"
    traj_modification(inst_tag)
    exit(0)
    
    # test_nearest_trajectories()
    # exit(0)
    
    forecasting_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_knife_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-12-52-23/ts_to_hand_obj_obs_reset_1.npy"
    inst_tag = "ori_grab_s2_knife_pass_1"
    # select_nearest_forecasting_res(forecasting_res_fn, inst_tag)
    select_nearest_forecasting_res_other_insts(forecasting_res_fn, inst_tag)
    exit(0)
    
    
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_flute_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-16-28-36/ts_to_hand_obj_obs_reset_1.npy"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/tracking_ori_grab_s2_flute_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-27-52/ts_to_hand_obj_obs_reset_1_sorted.npy"
    data_inst_tag = "ori_grab_s2_flute_pass_1_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_torussmall_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-20-12-52/ts_to_hand_obj_obs_reset_1.npy"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/tracking_ori_grab_s2_torussmall_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-02-56/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s2_torussmall_lift_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_cylindermedium_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-20-24-46/ts_to_hand_obj_obs_reset_1.npy"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/tracking_ori_grab_s2_cylindermedium_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-39-52/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s2_cylindermedium_lift_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_toothbrush_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-23-59/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s2_toothbrush_lift_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_toothpaste_squeeze_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-29-43/ts_to_hand_obj_obs_reset_1.npy"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/tracking_ori_grab_s2_toothpaste_squeeze_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-22-50-15/ts_to_hand_obj_obs_reset_1.npy"
    data_inst_tag = "ori_grab_s2_toothpaste_squeeze_1_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_flute_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_21-12-34-59/ts_to_hand_obj_obs_reset_1.npy"
    data_optimized_res_nn= "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/tracking_ori_grab_s2_flute_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_31-23-27-52/ts_to_hand_obj_obs_reset_1.npy"
    
    data_inst_tag = "ori_grab_s2_flute_pass_1_nf_300"
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_hammer_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-01-18-30/ts_to_hand_obj_obs_reset_1.npy" # best_obj_pos: [0.09693158], best_obj_rot: [0.68539172]
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_hammer_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-20-15-36/ts_to_hand_obj_obs_reset_1.npy" # best_obj_pos: [0.12394252], best_obj_rot: [0.5679754]
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_hammer_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-20-24-26/ts_to_hand_obj_obs_reset_1.npy" # best_obj_pos: [0.12394252], best_obj_rot: [0.5679754]
    data_optimized_res_nn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_wforecasting_wworldmodel_/tracking_ori_grab_s2_hammer_use_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_30-01-09-42/ts_to_hand_obj_obs_reset_1.npy" # best_obj_pos: [0.09723868], best_obj_rot: [0.68571061]
    data_inst_tag = "ori_grab_s2_hammer_use_2_nf_300"
    
    data_inst_tag = (data_inst_tag, data_inst_tag)
    inspect_optimized_res_nn(data_optimized_res_nn, data_inst_tag, downsample=False)
    exit(0)
    
    
    # flow #  # inst tag and the optimized res #inst tag # and inst tag #
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v2_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    
    calcualte_merged_succ_info_all_thres(inst_tag_to_optimized_res_fn)
    inst_tag_to_optimized_res_fn = inst_tag_to_optimized_res_fn.replace("data_inst_tag_to_best_opt_res_all.npy", "data_inst_tag_to_optimized_res.npy")
    inspect_data_inst_tag_to_optimized_res_metrics(inst_tag_to_optimized_res_fn)
    exit(0)
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_/statistics/data_inst_tag_to_optimized_res.npy"

    inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/statistics/data_inst_tag_to_optimized_res.npy"
    # target subj tag #
    target_subj_tag = ''
    
    random_select = False
    
    # /data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/statistics/data_inst_tag_to_best_opt_res_all.npy
    
    data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    if len(target_subj_tag) > 0:
        data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    else:
        if random_select:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
        else:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res)
    print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
    exit(0)
    
    # obj_retar_info_fn = "/data/xueyi/data/GRAB_Tracking_PK_reduced_300/data/passive_active_info_ori_grab_s1_alarmclock_lift_nf_300.npy"
    # test_obj_retargete_info(obj_retar_info_fn)
    # exit(0)
    
    # inst_tag_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res_to_s1.npy"
    # # inst_tag_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/obj_type_to_optimized_res.npy"
    # obj_type_to_opt_res = inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn)
    # exit(0)
    
    # obj_type_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/obj_type_to_optimized_res.npy"
    # get_retargeting_inst_tag_to_opt_res(obj_type_to_opt_res_fn)
    # exit(0)
    
    # # /data/xueyi/data/GRAB_Tracking_PK_reduced_300/data
    
    # inst_tag_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res.npy"
    # obj_type_to_opt_res = inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn)
    # obj_type_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/obj_type_to_optimized_res.npy"
    # np.save(obj_type_to_opt_res_fn, obj_type_to_opt_res)
    # print(f"obj type to opt res saved to {obj_type_to_opt_res_fn}")
    # exit(0)
    
    
    # optimized_res_statistics_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inspect_subj_optimized_res(optimized_res_statistics_fn)
    # exit(0)
    
    # model_weight_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_wforecasting_v2_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_10-11-35-45/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    # inspect_weight_keys_from_ckpt(model_weight_fn)
    # exit(0)
    
    # # model_weight_shape_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_train_freq60_substep2_nresume_ninstfeat_v2_wohistory_nfuture_s9_/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_05-08-52-28/nn/tracking_ori_grab_s2_apple_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    # # test_model_weights_shape(model_weight_shape_fn)
    # # exit(0)
    
    # teacher_idx_to_weights_fn = "/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights_v3.npy"
    # scp_teacher_model_weights(teacher_idx_to_weights_fn)
    # exit(0)
    
    # teacher_idx_to_weights_fn = "/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights_v2.npy"
    # inspect_teacher_idx_to_weights(teacher_idx_to_weights_fn=teacher_idx_to_weights_fn)
    # exit(0)
    
    # key: obj_pos_diff, cur_avg_res: 0.2577854196836309, cur_median_res: 0.22121629863977432
    # key: obj_rot_diff, cur_avg_res: 0.7331847264544993, cur_median_res: 0.7805410921573639
    # key: obj_pose_diff, cur_avg_res: 0.4997363794136155, cur_median_res: 0.509513952434063
    
    # inst_tag_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_woptresall_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # # inst_tag_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_wooptres_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_opt_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_wooptres_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # calculate_avg_opt_res(inst_tag_to_opt_res_fn)
    # exit(0)
    
    # # # merge_inst_tag_to_opt_res_dict()
    # # exit(0)
    
    
    # inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests6_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inspect_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn)
    # exit(0)
    
    
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests6_"
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_"
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests3_"
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests4_"
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests7_"
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_woptresall_"
    # grab_eval_data_folder = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_wooptres_"
    # grab_eval_data_folder = '/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_' 
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)
    
    
    # # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_/statistics/data_inst_tag_to_optimized_res.npy"
    # # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests6_/statistics/data_inst_tag_to_optimized_res.npy"
    # # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests3_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests4_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests7_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_woptresall_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_wooptres_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/statistics/data_inst_tag_to_optimized_res.npy"
    # downsample = False
    # # downsample = True
    # target_inst_tag = None
    # # target_inst_tag = "s1_" # 
    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_inst_tag, downsample=downsample)
    # exit(0)
    
    # target_subj_tag = ''
    # random_select = False
    # inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests6_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_woptresall_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_wooptres_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_openlooptest_trajfromnn_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    # eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    # if len(target_subj_tag) > 0:
    #     data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    # else:
    #     if random_select:
    #         data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
    #     else: # 
    #         data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    # data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    # np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res)
    # print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
    # exit(0)
    
    # # teacher_model_version_idx = 2
    # teacher_model_version_idx = 3
    # teacher_idx_to_wegights, teacher_idx_to_inst_tags  = get_teacher_idx_to_teacher_weight_v2()
    teacher_idx_to_wegights, teacher_idx_to_inst_tags  =  get_teacher_idx_to_teacher_weight_v3()
    teacher_idx_to_wegights_sv_fn  = os.path.join("../assets", f"teacher_idx_to_wegights_v{teacher_model_version_idx}.npy")
    np.save(teacher_idx_to_wegights_sv_fn, teacher_idx_to_wegights)
    teacher_idx_to_inst_tags_sv_fn = os.path.join("../assets", f"teacher_idx_to_inst_tags_v{teacher_model_version_idx}.npy")
    np.save(teacher_idx_to_inst_tags_sv_fn, teacher_idx_to_inst_tags)
    print(f"teacher_idx_to_wegights_sv_fn: {teacher_idx_to_wegights_sv_fn}")
    print(f"teacher_idx_to_inst_tags_sv_fn: {teacher_idx_to_inst_tags_sv_fn}")
    exit(0)
    
    # contact_data_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300_contactflag/passive_active_info_ori_grab_s1_airplane_offhand_1_nf_300_contact_flag.npy"
    # test_contact_data(contact_data_fn)
    # exit(0)
    
    # ckpt_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_newtrain_all_/tracking_ori_grab_s6_torusmedium_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_13-05-21-03/nn/last_tracking_ori_grab_s6_torusmedium_inspect_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_ep_50_rew_23.300459.pth"
    # loaded_ckpt = torch.load(ckpt_fn, map_location='cpu')['model']
    # print(loaded_ckpt.keys())
    # for key in loaded_ckpt:
    #     print(f"key: {key}, val: {loaded_ckpt[key].shape}")
    # exit(0)
    
    # dict_fn_1 = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95/statistics/obj_type_to_optimized_res.npy"
    # dict_fn_2 = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95_remains/statistics/obj_type_to_optimized_res.npy"
    # sv_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95_remains/statistics/obj_type_to_optimized_res_merged.npy"
    # dict_fn_1 = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1/statistics/obj_type_to_optimized_res.npy"
    # dict_fn_2 = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remains/statistics/obj_type_to_optimized_res.npy"
    # sv_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remains/statistics/obj_type_to_optimized_res_merged.npy"
    # dict_fn_1 = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remainsv2/statistics/obj_type_to_optimized_res.npy"
    # dict_fn_2 = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remains/statistics/obj_type_to_optimized_res_merged.npy"
    # sv_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remains/statistics/obj_type_to_optimized_res_mergedv2.npy"
    # merge_two_obj_type_to_optimized_res(dict_fn_1, dict_fn_2, sv_dict_fn)
    # exit(0)
    # # obj_type_to_kinematics_traj_dict_fn # 
    
    # get_obj_type_to_kine_fn()
    # exit(0)
    
    
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_"
    # # 0.43665158371040724, 193/442
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1"
    # # 0.4117647058823529, 56/136
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4"
    # # 0.4666666666666667, 77/165
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95"
    # train_root  ="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95_remains"
    # # 
    # train_root  ="/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remains"
    # train_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remainsv2"
    # train_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95_remainsv2"
    # obj_type_to_opt_res_fn = os.path.join(train_root, "statistics", "obj_type_to_optimized_res.npy")
    # inspect_obj_type_to_optimized_res(obj_type_to_opt_res_fn)
    # exit(0)
    
    
    # train_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_totsamples_tests2_forcastv3_partialhandobjcond_wclipglbfeatsv4_woptresall_"
    # calculate_optimized_res(train_root)
    # exit(0)
    
    # teacher_idx_to_model_weights_fn = "/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights.npy"
    # teacher_idx_to_model_weights_fn = "/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_inst_tags.npy"
    # inspect_teacher_idx_to_model_weights(teacher_idx_to_model_weights_fn)
    # exit(0)
    

    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data"
    # # reduce_kine_infos(grab_tracking_folder)

    # obj_type_to_ckpt_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_wrandomize_onlymass_/statistics/obj_type_to_optimized_res.npy"
    # inspect_obj_type_to_ckpt_fn(obj_type_to_ckpt_fn)

    # exit(0)
    
#     best_sv_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_best_opt_res_all.npy"
#     best_sv_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_optimized_res_top50.npy"
#     best_sv_res_fn = "../assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_optimized_res_top50.npy"
#     inspect_demo_best_sv_fns(best_sv_res_fn) 
#     exit(0)
    
    # obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1/statistics/obj_type_to_optimized_res.npy"
    # allegro_obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_newtrain_all_/statistics/obj_type_to_optimized_res.npy"
    # allegro_obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_/statistics/obj_type_to_optimized_res.npy"
    # inspect_obj_type_to_optimized_res(obj_type_to_opt_res_fn, allegro_obj_type_to_opt_res_fn)
    # exit(0)

    
#     # /root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_best_opt_res_all.npy
    
#     data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
#     data_inst_tag_to_optimized_res_fn =  "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wfranka_/statistics/data_inst_tag_to_optimized_res.npy"
#     interested_keys = None
#     dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_wfranka'
    

    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1"
    # # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_modelv2_realleap_wvel_stiffv4_'
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_newtrain_all_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95_remains"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remains"
    # train_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1_remainsv2"
    # train_root = "/data/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p100_d4_f0d95_remainsv3"
    # calculate_optimized_res(train_root)
    # exit(0)
    
    # inst_opt_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/good_inst_opt_res.npy"
    # inspect_inst_opt_res(inst_opt_res_fn)
    # exit(0)
    

#     copy_optimized_infos(interested_keys, data_inst_tag_to_optimized_res_fn, dst_folder=dst_folder)
#     exit(0)
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_leap_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_leap_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inspect_sorted_info(inst_tag_to_optimized_res_fn, data_inst_tag_to_optimized_res_fn)
    # exit(0)
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_nsup_downsample_moreinv_freq120_pertrain_leap_fix_ #
    
    # # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data"
    # reduce_kine_infos(grab_tracking_folder)
    # exit(0)
    
    
    # obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1/statistics/obj_type_to_optimized_res.npy"
    # allegro_obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_newtrain_all_/statistics/obj_type_to_optimized_res.npy"
    # allegro_obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_/statistics/obj_type_to_optimized_res.npy"
    # obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_nsup_downsample_moreinv_freq120_pertrain_leap_fix_/statistics/obj_type_to_optimized_res.npy"
    # obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_/statistics/obj_type_to_optimized_res.npy"
    # obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1/statistics/obj_type_to_optimized_res.npy"
    
    # inspect_obj_type_to_optimized_res(obj_type_to_opt_res_fn, allegro_obj_type_to_opt_res_fn)
    # exit(0)
    
    
    
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1"
    # # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_modelv2_realleap_wvel_stiffv4_'
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_newtrain_all_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_nsup_downsample_moreinv_freq120_pertrain_leap_fix_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_leap_fix_p3_d0.1"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_wfranka_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_wrandomize_"
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_"
    # calculate_optimized_res(train_root)
    # exit(0)
    
    # inst_opt_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/good_inst_opt_res.npy"
    # inspect_inst_opt_res(inst_opt_res_fn)
    # exit(0)
    

    # # log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s10_
    # subj_tag_idx_to_eval_root_folder = {
    #     2: "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s2new_"
    # }

    # # for subj_idx in range(3, 11):
    # for subj_idx in range(3, 9):
    #     if subj_idx == 7:
    #         continue
        
    #     subj_tag_idx_to_eval_root_folder[subj_idx] = f"/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s{subj_idx}_" 
    
    # tot_good_res_dict = {}
    # for subj_idx in subj_tag_idx_to_eval_root_folder:
    #     cur_inst_eval_folder = subj_tag_idx_to_eval_root_folder[subj_idx]
    #     # 
    #     # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s8_/statistics/data_inst_tag_to_best_opt_res_all.npy
    #     cur_inst_to_opt_res_all_fn = os.path.join(cur_inst_eval_folder, "statistics", "data_inst_tag_to_best_opt_res_all.npy")
    #     cur_good_inst_res = get_good_res_from_optimized_res_all(cur_inst_to_opt_res_all_fn)
    #     tot_good_res_dict.update(cur_good_inst_res)
    # print(tot_good_res_dict)
    # good_res_sv_dict_fn = f"../assets/good_inst_opt_res.npy"
    # np.save(good_res_sv_dict_fn, tot_good_res_dict)
    # exit(0)
    
    
    
    
    # inst_tag_to_weights_fn = "/root/diffsim/IsaacGymEnvs2/assets/teacher_idx_to_wegights.npy"
    # inspect_inst_tag_to_weights_fn(inst_tag_to_weights_fn)
    # exit(0)
    
    # inst_tag_fa_to_opt_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/inst_tag_to_fa_opt_res_top10_s10.npy"
    # inspect_inst_tag_to_fa_to_opt_res(inst_tag_fa_to_opt_res_fn)
    # exit(0)
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"
    # topk = 10
    # subj_idx = 10
    # inst_tag_to_fa_opt_res = get_inst_tag_to_nearest_info(inst_tag_to_optimized_res_fn, data_inst_tag_to_optimized_res_fn, subj_idx, topk=topk)
    # # inst_tag_to_fa_opt_res # 
    # inst_tag_to_fa_opt_res_sv_fn = f"../assets/inst_tag_to_fa_opt_res_top{topk}_s{subj_idx}.npy"
    # np.save(inst_tag_to_fa_opt_res_sv_fn, inst_tag_to_fa_opt_res)
    # print(f'inst_tag_to_fa_opt_res_sv_fn: {inst_tag_to_fa_opt_res_sv_fn}')
    # # inst tag to nearest info #
    # exit(0)
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inspect_optimized_res_info(data_inst_tag_to_optimized_res_fn)
    # exit(0)
    
    #
    
    # teacher_idx_to_wegights, teacher_idx_to_inst_tags = get_teacher_idx_to_teacher_weight()
    teacher_idx_to_wegights, teacher_idx_to_inst_tags  = get_teacher_idx_to_teacher_weight_v2()
    teacher_idx_to_wegights_sv_fn  = os.path.join("../assets", "teacher_idx_to_wegights.npy")
    np.save(teacher_idx_to_wegights_sv_fn, teacher_idx_to_wegights)
    teacher_idx_to_inst_tags_sv_fn = os.path.join("../assets", "teacher_idx_to_inst_tags.npy")
    np.save(teacher_idx_to_inst_tags_sv_fn, teacher_idx_to_inst_tags)
    print(f"teacher_idx_to_wegights_sv_fn: {teacher_idx_to_wegights_sv_fn}")
    print(f"teacher_idx_to_inst_tags_sv_fn: {teacher_idx_to_inst_tags_sv_fn}")
    exit(0)
    
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_150_customdamping_leap'
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_nsup_downsample_moreinv_freq60_pertrain_'
    # train_root = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_new_nsup_downsample_moreinv_freq60_pertrain_fix_'
    # train_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_allegro_wcustomdamping_freq60_pertrain_wrandomize_onlymass_"
    
    # calculate_optimized_res(train_root)
    # exit(0)
    
    # # subj_nm = 's10'
    # for idx in range(1, 10):
    #     subj_nm = f"s{idx}"
    #     fn_dict = get_and_save_target_inst_tag_list_fn(subj_nm)
    #     sv_root = "../assets"
    #     sv_fn = f"inst_tag_list_{subj_nm}.npy"
    #     sv_fn = os.path.join(sv_root, sv_fn)
    #     np.save(sv_fn, fn_dict)
    #     print(f"sv_fn: {sv_fn}")
    # exit(0)
    
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_allsubjs_/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn)
    # exit(0)
    
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inspect_best_optimized_res(inst_tag_to_optimized_res_fn) #
    # exit(0)
    

    
    # optimized_data_sv_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_"
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_ #
    # optimized_data_sv_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_"
    # calculate_optimized_res(optimized_data_sv_root)
    # exit(0)
    
    # grab_eval_data_folder  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_"

    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wfranka_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_v2_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_v3_"

    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)
    
    
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_allsubjs_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs1_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs2_fix_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_ep3000_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.001_newfix_frommoreenvs_teachers10new_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v2_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v3_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"

    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_/statistics/data_inst_tag_to_optimized_res.npy"
    # downsample = False
    # # downsample = True
    # target_inst_tag = None
    # # target_inst_tag = "s1_"
    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_inst_tag, downsample=downsample)
    # exit(0)

    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_leap_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wfranka_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_v2_/statistics/data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_v3_/statistics/data_inst_tag_to_optimized_res.npy"
    downsample = False
    # downsample = True
    target_inst_tag = None
    # target_inst_tag = "s1_"
    resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_inst_tag, downsample=downsample)
    exit(0)

    
    # log_path='/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s10_
    subj_tag_idx_to_eval_root_folder = {
        2: "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s2new_"
    }

    # # for subj_idx in range(3, 11):
    # for subj_idx in range(3, 9):
    #     if subj_idx == 7:
    #         continue
        
    #     subj_tag_idx_to_eval_root_folder[subj_idx] = f"/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s{subj_idx}_" 
        
    # # # for cur_subj_idx in subj_tag_idx_to_eval_root_folder:
    # # #     cur_eval_folder = subj_tag_idx_to_eval_root_folder[cur_subj_idx]
    # # #     get_data_inst_tag_to_optimized_res(cur_eval_folder) # 
    
    
    # # for cur_subj_idx in subj_tag_idx_to_eval_root_folder:
    # #     cur_eval_folder = subj_tag_idx_to_eval_root_folder[cur_subj_idx]
    # #     cur_inst_tag_to_optimized_res_fn = os.path.join(cur_eval_folder, 'statistics', 'data_inst_tag_to_optimized_res.npy')
    # #     downsample = False
    # #     target_inst_tag = None
    # #     # resave optimized res #
    # #     resave_optimized_res(cur_inst_tag_to_optimized_res_fn, target_inst_tag, downsample=downsample)
        
    # # exit(0)
    
    # for subj_idx in subj_tag_idx_to_eval_root_folder:
    #     cur_eval_folder = subj_tag_idx_to_eval_root_folder[subj_idx]
    #     inst_tag_to_optimized_res_fn = os.path.join(cur_eval_folder, 'statistics', 'data_inst_tag_to_optimized_res.npy')
    #     # resave_optimized_res(cur_inst_tag_to_optimized_res_fn, target_subj_tag, random_select=random_select)
        
    #     data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    #     eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    #     if len(target_subj_tag) > 0:
    #         data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    #     else:
    #         if random_select:
    #             data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
    #         else:
    #             data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    #     data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    #     np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res)
    #     print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
           
    
    
    # # grab_eval_data_folder  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_allsubjs_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs1_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs2_fix_"
    
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_final_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_ep3000_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.001_newfix_frommoreenvs_teachers10new_s1subj_"
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v2_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v3_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s10_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_leap_"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_"
    
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)
    
    
    # get the data tag to optimized res # # get the optimized res #
    
    
    # opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/tracking_ori_grab_s1_toothpaste_lift_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-17-43-44/ts_to_hand_obj_obs_reset_1.npy"
    # opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/tracking_ori_grab_s1_cylindermedium_pass_1_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_08-21-24-26/ts_to_hand_obj_obs_reset_1.npy"
    # inspect_optimized_res(opt_res_fn)
    # exit(0)
    
    # # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_allsubjs_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs1_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn =  "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_ep3000_/statistics/data_inst_tag_to_optimized_res.npy"
    
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.001_newfix_frommoreenvs_teachers10new_s1subj_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v2_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v3_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"

    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_/statistics/data_inst_tag_to_optimized_res.npy"

    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_persamples_wrandomize_onlymass_/statistics/data_inst_tag_to_optimized_res.npy"
    # target subj tag #
    target_subj_tag = ''
    
    random_select = False
    # random_select = True
    
    # for subj_idx in subj_tag_idx_to_eval_root_folder:
    #     cur_eval_folder = subj_tag_idx_to_eval_root_folder[subj_idx]
    #     inst_tag_to_optimized_res_fn = os.path.join(cur_eval_folder, 'statistics', 'data_inst_tag_to_optimized_res.npy')
    #     # resave_optimized_res(cur_inst_tag_to_optimized_res_fn, target_subj_tag, random_select=random_select)
        
    #     data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    #     eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    #     if len(target_subj_tag) > 0:
    #         data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    #     else:
    #         if random_select:
    #             data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
    #         else:
    #             data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    #     data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    #     np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res)
    #     print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
            
    
    # ########## Singel optimized info ##########
    data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    if len(target_subj_tag) > 0:
        data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    else:
        if random_select:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
        else:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res)
    print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
    exit(0)
    
    
    
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/statistics/data_inst_tag_to_optimized_res.npy'
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_/statistics/data_inst_tag_to_optimized_res.npy'
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs1_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs1_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"
    
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inspect_sorted_info(inst_tag_to_optimized_res_fn, data_inst_tag_to_optimized_res_fn)
    # exit(0)
    
    
    
    
    # optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2'
    # optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping'
    # # optimized_data_sv_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_"
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_ #
    # calculate_optimized_res(optimized_data_sv_root)
    # exit(0)
    
    
    
    # obj_type_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping/statistics/obj_type_to_optimized_res.npy"
    # inspect_obj_type_to_opt_res(obj_type_to_opt_res_fn)
    # exit(0)
    
    # best_sv_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/tracking_ori_grab_s1_banana_peel_2_nf_300_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-19-58-08/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
    # inspect_best_sv_res(best_sv_res_fn)
    # exit(0)
    
    
    # data_inst_tag_to_opt_res = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_opt_res = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inspect_inst_tag_to_opt_res(data_inst_tag_to_opt_res)
    # exit(0)
    
    # inst_tag_to_opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_300_v12/statistics/data_inst_tag_to_optimized_res.npy'
    # inspect_data_inst_tag_to_opt_res(inst_tag_to_opt_res_fn)
    # exit(0)
    
    # best_sv_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # best_sv_res_fn = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_optimized_res_top50.npy"
    # inspect_demo_best_sv_fns(best_sv_res_fn) 
    # exit(0)
    
    
    # # /root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_best_opt_res_all.npy
    
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # interested_keys = None
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_taco_400_demo'
    
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_optimized_res.npy"
    # interested_keys = None
    # # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_taco_400_demo'
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_baseline'
    
    # copy_optimized_infos(interested_keys, data_inst_tag_to_optimized_res_fn, dst_folder=dst_folder)
    # exit(0)
    
    
    # # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_LEAP_PK/data"
    # reduce_kine_infos(grab_tracking_folder)
    # exit(0)
    
    
    # 
    # dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_realleap_wovel/statistics/obj_type_to_optimized_res.npy"
    # dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_realleap/statistics/obj_type_to_optimized_res.npy"
    # inspect_obj_type_to_optimized_res(dict_fn)
    # exit(0)
    
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_realleap_wovel'
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_realleap'
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wstatebased_wcustomidamping_modelv2_realleap'
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wstatebased_wcustomidamping_modelv2_realleap_wvel_stiffv4_'
    # calculate_optimized_res(train_root)
    # exit(0)
    

    # dict_fn_a = '../assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_optimized_res_top50.npy'
    # dict_fn_b = '../assets/optimized_res_taco_400_demo/statistics/data_inst_tag_to_optimized_res_top50.npy'
    # combine_two_dict_fn(dict_fn_a, dict_fn_b)
    # exit(0)
    
    # the inst tag to best opt res fn for the taco dataset #
    # 
    
    # # best_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/data_inst_tag_to_optimized_res_top50.npy'
    # best_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo/statistics/data_inst_tag_to_optimized_res_top50.npy'
    # inspect_topk_best_optimized_res_fn(best_optimized_res_fn)
    # exit(0)


    # combined_ob
    

    # inst_tag_to_best_opt_res = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inst_tag_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # cur_root_path = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo"
    
    # inst_tag_to_best_opt_res = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inst_tag_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # cur_root_path = "/root/diffsim/IsaacGymEnvs2/assets/optimized_res_taco_400_demo"
    # get_data_inst_tag_to_valid_optimized_res(inst_tag_to_best_opt_res, inst_tag_to_opt_res_fn, cur_root_path)
    # exit(0)
    
    
    # inst_tag_to_best_opt_res = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    
    # inpsect_data_inst_tag_to_best_optimized_res(inst_tag_to_best_opt_res)
    # exit(0)
    
    
    
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # inspect_optimized_res_grab_long(inst_tag_to_optimized_res_fn)
    # exit(0)
    
    
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v3_/statistics/data_inst_tag_to_optimized_res.npy'
    # # data #
    # interested_keys = None
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_leap'
    # data isnt tag to optimzed # inst tag to optimized res # # optimized res #
    
    
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_taco_generalist_direct_samples_pertrj_v4_testall_/statistics/data_inst_tag_to_optimized_res.npy'
    # interested_keys = None
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_taco_leap'
    
    
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # interested_keys = None
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_300_demo'
    
    
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v4_stiffness20_v2_/statistics/data_inst_tag_to_optimized_res.npy"
    # interested_keys = None
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_taco_400_demo'
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_stiff_res'
    # dst_folder = '/root/diffsim/IsaacGymEnvs2/assets/optimized_res_grab_stiff_res_v3'
    
    # copy_optimized_infos(interested_keys, data_inst_tag_to_optimized_res_fn, dst_folder=dst_folder)
    # exit(0)
    
    
    # # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2_eval"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_taco_generalist_direct_samples_pertrj_v3_"
    # grab_eval_data_folder =  "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_"
    # grab_eval_data_folder  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_"
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)
    
    
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/statistics/data_inst_tag_to_optimized_res.npy"
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_allsubjs_/statistics/data_inst_tag_to_optimized_res.npy
    # target_inst_tag = None
    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_inst_tag)
    # exit(0)
    
    #### grab eval data folder ####
    # # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv1_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv1_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_/' # v4_300 #
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv3_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v3_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_' # 
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v4_stiffness20_v2_' # stiffness20_v2_ #
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder) #
    # exit(0) # exit(0) #
    
    # get data inst tag to optimized res #
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v3_/tracking_ori_grab_s1_torussmall_lift_leap_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_22-14-16-08/ts_to_hand_obj_obs_reset_1.npy
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_ # # realv3_ ##  real v3 ##
    
    # # # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained/statistics/data_inst_tag_to_optimized_res.npy'
    # # # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv1_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v3_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_v4_stiffness20_v2_/statistics/data_inst_tag_to_optimized_res.npy'
    # target_inst_tag = 's1'
    # target_inst_tag = None
    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=target_inst_tag)
    # exit(0)
    
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping'
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wstatebased_wcustomidamping_realleap'
    # train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_150_customdamping_leap'
    # calculate_optimized_res(train_root)
    # exit(0)
    
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv1_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_genesamples_0.9sup_samples_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_400_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_final_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_/statistics/data_inst_tag_to_optimized_res.npy'
    # # 
    # target_subj_tag = ''
    # random_select = False
    # data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    # eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    # if len(target_subj_tag) > 0:
    #     data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    # else:
    #     if random_select:
    #         data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
    #     else:
    #         data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    # data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    # np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res) # 
    # print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
    # exit(0)
    
    
    
    
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v13/statistics/data_inst_tag_to_best_opt_res_all.npy' 
    # [0.43252595155709345, 0.5813148788927336, 0.6851211072664359]
    # eval_key: hand_glb_diff, mean: 0.10011871187444563
    # eval_key: hand_joint_diff, mean: 0.48641870881634197
    # eval_key: obj_pos_diff, mean: 0.09177578062397139
    # eval_key: obj_rot_diff, mean: 0.42611799534709977 # 
    
    # eval_key: hand_glb_diff, mean: 0.17798857953464395, medium: 0.12256485968828201
    # eval_key: hand_joint_diff, mean: 0.5350225772275481, medium: 0.5218476057052612
    # eval_key: obj_pos_diff, mean: 0.11146851163357496, medium: 0.07806023955345154
    # eval_key: obj_rot_diff, mean: 0.48813456699771934, medium: 0.34438350796699524
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v12/statistics/data_inst_tag_to_best_opt_res_all.npy' 
    # [0.47126436781609193, 0.6551724137931034, 0.7241379310344828]
    # eval_key: hand_glb_diff, mean: 0.1311615438992693
    # eval_key: hand_joint_diff, mean: 0.516579905289343
    # eval_key: obj_pos_diff, mean: 0.08651978972678383
    # eval_key: obj_rot_diff, mean: 0.4307191177838664 # 43.01/61.52
    # eval_key: hand_glb_diff, mean: 0.1311615438992693, medium: 0.11180009320378304
    # eval_key: hand_joint_diff, mean: 0.516579905289343, medium: 0.5047686696052551
    # eval_key: obj_pos_diff, mean: 0.08651978972678383, medium: 0.045269615948200226
    # eval_key: obj_rot_diff, mean: 0.4307191177838664, medium: 0.3302718698978424
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_best_opt_res_all.npy' 
    # [0.39792387543252594, 0.5813148788927336, 0.6678200692041523]
    # eval_key: hand_glb_diff, mean: 0.15484016026830535
    # eval_key: hand_joint_diff, mean: 0.5335653240995846
    # eval_key: obj_pos_diff, mean: 0.08554807287806795 # and the taco
    # eval_key: obj_rot_diff, mean: 0.4158117506310515
    # eval_key: hand_glb_diff, mean: 0.15484016026830535, medium: 0.14834869280457497
    # eval_key: hand_joint_diff, mean: 0.5335653240995846, medium: 0.5264893174171448
    # eval_key: obj_pos_diff, mean: 0.08554807287806795, medium: 0.049710292369127274
    # eval_key: obj_rot_diff, mean: 0.4158117506310515, medium: 0.341532826423645
    
    # isaacgym_rl_exp_grab_generalist_direct_samples_taco_grab_v2_v5
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v13/statistics/data_inst_tag_to_best_opt_res_all.npy' 
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained/statistics/data_inst_tag_to_best_opt_res_all.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv1_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv3_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # to use  # OMniGrasp reward #
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_300_v12/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_new_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_/statistics/data_inst_tag_to_best_opt_res_all.npy'
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping_samples_realv3_/statistics/data_inst_tag_to_optimized_res.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_downsample_freq60_subs1_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    
    ### s10 ###
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_s10onlytrain_s10_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_s1subj_final_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_allsubj_final_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_newfix_s1subj_ep3000_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.001_newfix_frommoreenvs_teachers10new_s1subj_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v2_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.0005_newfix_frommoreenvs_teachers10_s1subj_v3_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_realgrab_gensamplesv4_300_sup0.000_earlyterminate_sample_/statistics/data_inst_tag_to_best_opt_res_all.npy"
    # [0.35977011494252873, 0.5494252873563219, 0.7126436781609196]
    # eval_key: hand_glb_diff, mean: 0.1242386272187805, medium: 0.17229154095053673
    # eval_key: hand_joint_diff, mean: 0.5138634036498508, medium: 0.5641828751564026
    
    
    # eval_key: obj_pos_diff, mean: 0.09271082007486758, medium: 0.066922510266304016
    # eval_key: obj_rot_diff, mean: 0.43351146146312525, medium: 0.44048333764076233
    
    
    # DGRASP #
    # [0.3482758620689655, 0.5494252873563219, 0.735632183908046]
    # eval_key: hand_glb_diff, mean: 0.13414728739459453, medium: 0.13723480373620987
    # eval_key: hand_joint_diff, mean: 0.5152642141128408, medium: 0.603957211971283
    # eval_key: obj_pos_diff, mean: 0.0862584309893307, medium: 0.06752850905060768
    # eval_key: obj_rot_diff, mean: 0.42872774888943327, medium: 0.44938764572143555
    
    # interesting demos? #
    
    ## wati for the DGRa's results (using genesamplev3) ##  no randominesses ## # multiple envornments; change the camer view and interatct with different objects? #
    
    # PPO ours reward
    # [0.38788927335640137, 0.5505190311418685, 0.6058823529411765] # 0.5813 6.03 0.1730 0.5439 36.18/56.07
    # eval_key: hand_glb_diff, mean: 0.09165994453126158, medium: 0.10763877302408218
    # eval_key: hand_joint_diff, mean: 0.5027805625022143, medium: 0.58998644948005676
    # eval_key: obj_pos_diff, mean: 0.07644877166503333, medium: 0.061110482066869736
    # eval_key: obj_rot_diff, mean: 0.41904766399456167, medium: 0.3945021986961365
    
    # merged info #
    # merged info # # # # # # # # # # merged info #
    
    calcualte_merged_succ_info_all_thres(inst_tag_to_optimized_res_fn)
    inst_tag_to_optimized_res_fn = inst_tag_to_optimized_res_fn.replace("data_inst_tag_to_best_opt_res_all.npy", "data_inst_tag_to_optimized_res.npy")
    inspect_data_inst_tag_to_optimized_res_metrics(inst_tag_to_optimized_res_fn)
    exit(0)
    
    # the succ rates # # 
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_best_opt_res_all.npy' 
    inspect_data_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn)
    exit(0)
    
    
    # data_inst_tag = "ori_grab_s1_apple_eat_1"
    # best_eval_info_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s1_apple_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_19-22-51-41/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
    # inspect_best_eval_info(data_inst_tag, best_eval_info_fn)
    # exit(0)
    
    
    # best_eval_info_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/tracking_TACO_taco_20230928_044_INTERPSEQ_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_interpfr_60_interpfr2_60_nntrans_40_04-03-33-41/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
    # inspect_best_eval_info(best_eval_info_fn)
    # exit(0)
    
    
    # grab_tracking_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    # reduce_kine_infos(grab_tracking_folder)
    # exit(0)
    
    # reduce the kinematic infos #
    # reduce the kinematic infos #
    # rotation and the translation errors #
    # wrist distances and the angle erros and the joint errors #
    
    # # compare the optimized 
    # ori_inst_tag_to_best_opt_res = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_s10.npy'
    # cur_inst_tag_to_best_opt_res = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_all_v12/statistics/data_inst_tag_to_best_opt_res_all.npy'
    
    
    train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_150_customdamping_leap'
    train_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_300_train_wcustomidamping'
    calculate_optimized_res(train_root)
    exit(0)
    
    # inspect_inst_tag_to_opt_res()
    # exit(0)
    
    # filter_inst_tag_to_all_res()
    # exit(0)
    
    # # obj_type_to_latent_features = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
    # # inspect_obj_type_to_latent_folders(obj_type_to_latent_features)
    # # exit(0)
    
    # construct_all_taco_opt_to_optimized_res()
    # exit(0)
    
    # opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res_tot.npy"
    # inspect_taco_optimized_res(opt_res_fn)
    # exit(0)
    
    
    
    # inspect_taco_mesh_folders()
    # exit(0)
    
    # obj_type_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy"
    # obj_type_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy'
    # obj_type_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res.npy'
    # inspect_obj_type_to_optimized_res(obj_type_to_optimized_res_fn)
    # exit(0)
    
    
    
    # opt_res_fn_samples = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # inspect_samples(opt_res_fn_samples)
    # exit(0)
    
    
    # obj_type_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy"
    # ## iteration 1 ##
    # data_inst_to_opt_res_fn_ori = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_samples = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # select_samples_fr_ori_cur_optimized_res(obj_type_to_optimized_res_fn, data_inst_to_opt_res_fn_samples)
    # exit(0)
    
    
    
    # obj_type_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy"
    # # ## iteration 1 ##
    # data_inst_to_opt_res_fn_ori = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_samples = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_best_opt_res_all.npy'
    
    
    # data_inst_to_opt_res_fn_ori = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_samples = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_all_v12/statistics/data_inst_tag_to_best_opt_res_all.npy'
    
    # compare_diff_opt_res_ori_w_samples(obj_type_to_optimized_res_fn, data_inst_to_opt_res_fn_ori, data_inst_to_opt_res_fn_samples)
    # exit(0)
    
    
    # taco_eval_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta_v1.npy"
    # taco_eval_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40/statistics/obj_type_to_optimized_res_beta.npy"
    # get_taco_eval_dict(taco_eval_dict_fn)
    # exit(0)
    

    # eval_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy"
    
    # inspect_eval_dict_grab_nf_300(eval_dict_fn)
    # exit(0)    

    # data_inst_tag_to_optimized_res_fn = './runs/statistics/data_inst_tag_to_optimized_res.npy'
    # # target_inst_tag = 's2'
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_optimized_res.npy"

    # tot_eval_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_taco_eval_dict(tot_eval_dict_fn)
    # exit(0)    
    
    
    # eval_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_eval_dict_grab_nf_300(eval_dict_fn)
    # exit(0)    

    data_inst_tag_to_optimized_res_fn = './runs/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v12/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v13/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_300_v12/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_all_v12/statistics/data_inst_tag_to_optimized_res.npy'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained/statistics/data_inst_tag_to_optimized_res.npy'
    # # target_inst_tag = 's2'

    target_inst_tag = 's1'

    
    # target_inst_tag = None
    

    resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=target_inst_tag)
    exit(0)
    
    
    # grab_eval_data_folder= './runs'
    # # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v12'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v13'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_300_v12'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_leap_grab_generalist_direct_samples_pertrj_'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_all_v12/'
    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained'
    get_data_inst_tag_to_optimized_res(grab_eval_data_folder) 
    exit(0)

    # samples_root = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_generalist_x"
    # target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_samples_init_specialist"
    # move_tested_samples(samples_root, target_folder)
    # exit(0)



    # # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_generalist_test/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_interpfr_60_interpfr2_60_nntrans_40_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_optimized_res.npy'
    # target_inst_tag = 's10'
    # target_inst_tag = None
    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=target_inst_tag)
    # exit(0)
    
    
    # # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_'
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/'
    # grab_eval_data_folder = './runs'
    # grab_eval_data_folder = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs_generalist_test"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v4"
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/tracking_ori_grab_s10_waterbottle_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-20-21-14/ts_to_hand_obj_obs_reset_1.npy
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10'
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v5'
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11'
    # # inst_tag_to_opt_res_fn = "data_inst_tag_to_optimized_res.npy"
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)
    
    ## eval dict fn ## #
    # eval_dict_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_eval_dict(eval_dict_fn)
    # exit(0)
    
    # # /root/diffsim/tiny-differentiable-simulator/python/examples/rsc/objs/meshes/
    # exp_info_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/exp_info_to_optimized_res.npy"
    # # exp_info_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_samples_init_specialist/statistics/obj_type_to_optimized_res.npy"
    # # exp_info_to_opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_weights_init_specialist/statistics/obj_type_to_optimized_res.npy'
    # self_exp_res_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy"  
    
    # # inspect_opt_res(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn)
    # save_res = True 
    # # # inspect_optfr_with_selfexp(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=save_res )
    # inspect_optfr_with_selfexp_list(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=save_res )
    # exit(0)
    
    # # optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2'
    # optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_samples_init_specialist'
    # optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_weights_init_specialist'
    # calculate_optimized_res(optimized_data_sv_root)
    # exit(0)
    
    # # waht about the succ rate on the TACO trajectories ? # 
    # what about the succ rate on the TACO trajectories #
    # how does it perform on TACO trajectories? #
    # save_obj_type_to_general_imit_policy_weights()
    # exit(0)
    
    
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_"
    # summarize_traj_features(samples_fn)
    # exit(0)
    
    # saved_samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_v3data_sample_/samples_ep_0_batch_0.npy"
    # inspect_saved_samples(saved_samples_fn)
    # exit(0)
    
    # ./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-17-06-19/ts_to_hand_obj_obs_reset_1.npy# 
    
    
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__05-23-03-00/ts_to_hand_obj_obs_reset_1.npy"
    # data_inst_tag = "/cephfs/xueyi/data/GRAB_Tracking_PK_Sampled/ori_grab_s2_train_lift_sampled_kinematics_info.npy"
    # data_inst_tag = [data_inst_tag]
    # inspect_optimized_res_nn(data_optimized_res_nn, data_inst_tag)
    # exit(0)
    
    
    # ch_to_fa_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task_list.npy"
    # inspect_child_task_to_fa_task_list(ch_to_fa_fn)
    # exit(0)
    
    # # cophfs # #
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy
    # def inspect_child_task_to_fa_task(ch_to_fa_fn):
    

    
    # ch_to_fa_fn  = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy"
    # inspect_child_task_to_fa_task(ch_to_fa_fn)
    # exit(0) # 
        
    # # inst tag #
    # inst_tag_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # inst_tag_to_opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2/statistics/data_inst_tag_to_optimized_res.npy'
    # inspect_inst_tag_to_opt_res(inst_tag_to_opt_res_fn)
    # exit(0)
    
    
    # sampled_res_fn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-18-11-57/ts_to_hand_obj_obs_reset_1.npy"
    # inspect_sampled_res(sampled_res_fn)
    # exit(0)
    

    # local_dict_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/child_task_to_fa_task.npy'
    # glb_dict_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy'
    # merge_child_task_to_fa_task(local_dict_fn, glb_dict_fn)
    # exit(0)

    # calcu eval metrics #
    # calcu_eval_metrics() # 
    # exit(0)
    
    # # #
    # data_optimized_res_nn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_123_runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__03-22-56-23/ts_to_hand_obj_obs_reset_1.npy" # find data optimized res nn # 
    # inpsect_data_optimized_res_w_obj_codes(data_optimized_res_nn)
    # exit(0)
    
    
    # opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy'
    # opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    # inspect_obj_type_to_optimized_res(opt_res_fn)
    # exit(0)
    

#     # # data_inst_to_opt_res_fn_a =  '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
#     data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res.npy'
#     data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_s9.npy'
#     data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_s10.npy'
#     data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/statistics/obj_code_to_best_opt_res.npy"
#     # data_inst_to_opt_res_fn_b = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
#     ##### joint training with the demonstration -- test result on training trajectories ######
#     data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-12-41-38/statistics/obj_code_to_best_opt_res.npy"
#     data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__03-22-56-23/statistics/obj_code_to_best_opt_res.npy'
#     # data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-13-30-08/statistics/obj_code_to_best_opt_res.npy"
#     data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-01-08-35/ts_to_hand_obj_obs_reset_1.npy'
#     data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-01-08-35/statistics/obj_code_to_best_opt_res.npy'
#     data_inst_to_opt_res_fn_b = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_best_opt_res_s10.npy'
#     data_inst_to_opt_res_fn_a = data_inst_to_opt_res_fn_b
#     calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b)

    # # # # data_inst_to_opt_res_fn_a =  '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
    # # # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res.npy'
    # # # # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_s9.npy'
    # # data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/statistics/obj_code_to_best_opt_res.npy"
    # data_inst_to_opt_res_fn_b = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
    
    

#     # ### using only the succ trajs during the training ? #####
#     ##### joint training with the demonstration -- test result on training trajectories ######
#     data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-12-41-38/statistics/obj_code_to_best_opt_res.npy"
#     data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__03-22-56-23/statistics/obj_code_to_best_opt_res.npy'
#     data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all.npy'



    ##### joint training with the demonstration -- test result on training trajectories ######
    data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-12-41-38/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__03-22-56-23/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_best_opt_res_all.npy' # without quality (after increased to the current one)
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_all_v12/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v12/statistics/data_inst_tag_to_best_opt_res_all.npy' # all version 
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_best_opt_res_all.npy' # 
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v13/statistics/data_inst_tag_to_best_opt_res_all.npy' # without diversity (first xxx training res)
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v4/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_a = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__09-14-58-35/statistics/obj_code_to_best_opt_res.npy'
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all_rndselect.npy'
    # data_inst_to_opt_res_fn_a = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_best_opt_res_all.npy'
    # data_inst_to_opt_res_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-13-30-08/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-01-08-35/ts_to_hand_obj_obs_reset_1.npy'
    data_inst_to_opt_res_fn_b = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-01-08-35/statistics/obj_code_to_best_opt_res.npy'

    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__05-21-54-08/statistics/obj_code_to_best_opt_res.npy'

    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__05-21-54-08/statistics/obj_code_to_best_opt_res.npy'

    data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__05-23-07-10/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-15-06/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-19-50/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-31-23/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-17-06-19/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-22-47-09/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-00-49-54/statistics/obj_code_to_best_opt_res.npy"
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-03-44-52/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-12-50-15/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b  = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-03-27/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-12-21/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-20-29/statistics/obj_code_to_best_opt_res.npy'
    # data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-20-29/statistics/obj_code_to_best_opt_res.npy'
    # data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-50-10/statistics/obj_code_to_best_opt_res.npy'
    # data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-17-10-12/statistics/obj_code_to_best_opt_res.npy'
    # data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-19-39-53/statistics/obj_code_to_best_opt_res.npy'
    # # data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-22-35-04/statistics/obj_code_to_best_opt_res.npy'

    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__08-00-58-23/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__08-04-21-37/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__08-13-39-24/statistics/obj_code_to_best_opt_res.npy'
    data_inst_to_opt_res_fn_b = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__09-14-58-35/statistics/obj_code_to_best_opt_res.npy'
    
    
    # data 
    # data_inst_to_opt_res_fn_b = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-00-49-54/ts_to_hand_obj_obs_reset_1.npy" # find succ info ##
    ##### merged succ info ##### merged succ info ##### merged succ info ##### merged succ info #####
    # calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b)
    calcualte_merged_succ_info_v2(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_a)

    # calcualte_merged_succ_info(data_inst_to_opt_res_fn_a, data_inst_to_opt_res_fn_b)
    # calcualte_merged_succ_info_v3(data_inst_to_opt_res_fn_b, data_inst_to_opt_res_fn_b)
    exit(0)
    
    # data_inst_to_opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
    # # data_inst_to_opt_res_fn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/statistics/obj_code_to_best_opt_res.npy"
    # # data_inst_to_opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res.npy'
    # calcualte_succ_info(data_inst_to_opt_res_fn)
    # exit(0)
    
    # data_inst_to_opt_res_fn_a = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
    # data_inst_to_opt_fn_b = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/statistics/obj_code_to_best_opt_res.npy"
    # compare_best_opt_res(data_inst_to_opt_res_fn_a, data_inst_to_opt_fn_b)
    # exit(0)
    
    
    # # inst_tag_to_opt_res_fn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-35-23/statistics/obj_code_to_best_opt_res.npy"
    # inst_tag_to_opt_res_fn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/statistics/obj_code_to_best_opt_res.npy"
    # inspect_data_inst_to_best_opt_res(inst_tag_to_opt_res_fn)
    # exit(0)
    
    
    # # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-35-23/ts_to_hand_obj_obs_reset_1.npy"
    # # # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/ts_to_hand_obj_obs_reset_1.npy"
    # # # ##### joint training with the demonstration -- test result on training trajectories ######
    # # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-12-41-38/ts_to_hand_obj_obs_reset_1.npy" 
    # # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-13-30-08/ts_to_hand_obj_obs_reset_1.npy"
    # # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-23-48-11/ts_to_hand_obj_obs_reset_1.npy"
    # # ##### joint training with the demonstration -- test result on training trajectories ######
    # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-12-41-38/ts_to_hand_obj_obs_reset_1.npy" 
    # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__02-13-30-08/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__03-22-56-23/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-01-08-35/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__05-21-54-08/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__05-23-07-10/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-15-06/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-19-50/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-31-23/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-17-06-19/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-22-47-09/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-00-23-01/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-00-49-54/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-03-44-52/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-12-50-15/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-03-27/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-12-21/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-20-29/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-13-50-10/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-17-10-12/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-19-39-53/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-22-35-04/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-22-47-46/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-17-10-12/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-19-39-53/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-22-35-04/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__07-22-47-46/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__08-00-58-23/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__08-04-21-37/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__08-13-39-24/ts_to_hand_obj_obs_reset_1.npy"
    # data_optimized_res_nn = "./runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__09-14-58-35/ts_to_hand_obj_obs_reset_1.npy"
    # # data_optimized_res_nn = './runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__06-15-31-23/statistics/obj_code_to_best_opt_res.npy'
    # obj_code_to_best_opt_res = inspect_optimized_res_nn_w_object_codes(data_optimized_res_nn)
    # stats_root = "/".join(data_optimized_res_nn.split("/")[:-1])
    # stats_folder = os.path.join(stats_root, "statistics")
    # os.makedirs(stats_folder, exist_ok=True)
    # obj_code_to_best_opt_res_fn = "obj_code_to_best_opt_res.npy"
    # obj_code_to_best_opt_res_fn = os.path.join(stats_folder, obj_code_to_best_opt_res_fn)
    # np.save(obj_code_to_best_opt_res_fn, obj_code_to_best_opt_res)
    # print(f"Saved obj_code_to_best_opt_res to {obj_code_to_best_opt_res_fn}")
    # exit(0)
    
    # target_sub_tag = 's2'
    # data_inst_to_best_opt_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy'
    # inspect_data_inst_to_best_opt_res(data_inst_to_best_opt_res_fn, target_sub_tag)
    # exit(0)
    
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_best_opt_res.npy
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_optimized_res.npy'
    # # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    
    # # target_subj_tag = 's2'
    # # target_subj_tag = 's9'
    # target_subj_tag = 's10'
    # data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag)
    # eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    # data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    # # # # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_optimized_res.npy'
    # inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v2/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v4/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v10/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v11/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v12/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_gene_v13/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_grab_all_v12/statistics/data_inst_tag_to_optimized_res.npy'
    inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_succonlytrained/statistics/data_inst_tag_to_optimized_res.npy'
    # target subj tag #
    target_subj_tag = ''
    
    random_select = False # 
    # random_select = True # true # # true # #
    
    data_inst_tag_to_best_opt_res = get_obj_inst_tag_to_best_opt_res(inst_tag_to_optimized_res_fn, target_subj_tag if len(target_subj_tag) > 0 else None, random_select=random_select)
    eval_statistics_sv_folder = "/".join(inst_tag_to_optimized_res_fn.split("/")[:-1])
    if len(target_subj_tag) > 0:
        data_inst_tag_to_best_opt_res_fn = f"data_inst_tag_to_best_opt_res_{target_subj_tag}.npy"
    else:
        if random_select:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all_rndselect.npy"
        else:
            data_inst_tag_to_best_opt_res_fn = "data_inst_tag_to_best_opt_res_all.npy"

    data_inst_tag_to_best_opt_res_fn = os.path.join(eval_statistics_sv_folder, data_inst_tag_to_best_opt_res_fn)
    np.save(data_inst_tag_to_best_opt_res_fn, data_inst_tag_to_best_opt_res) # 
    print(f"Saved data_inst_tag_to_best_opt_res to {data_inst_tag_to_best_opt_res_fn} with length {len(data_inst_tag_to_best_opt_res)}")
    exit(0)
    
    # training dataset --- do we really have such differnet to optimize data? # # not necessary to train them better but you can get some results #
    # # best_opt_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/tracking_ori_grab_s9_eyeglasses_wear_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_slicing_hist_pred_deter_30-15-30-29/ts_to_hand_obj_obs_reset_1_sorted_best.npy"
    # inspect_best_optimized_res(best_opt_fn)
    # exit(0)
    
    # 
    # obj_code_to_best_opt_res = inspect_optimized_res_nn_w_object_codes(data_optimized_res_nn)
    
    # ckpt_fn = 'runs/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_01-17-16-22/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth'
    # inspect_ckpt(ckpt_fn)
    # exit(0)
    
    
    # data_optimized_res_nn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-12-22-54/ts_to_hand_obj_obs_reset_1.npy'
    # data_inst_tag  = 'ori_grab_s1_cubesmall_offhand_1'
    # ### 
    # inspect_optimized_res_nn(data_optimized_res_nn, data_inst_tag,)
    
    # # data_inst_tag_to_opt
    
    # exit(0)
    
    # reward: -8.94 steps: 149.0
    # av reward: -13.87421875 av steps: 149.0
    # ori_grab_s5_watch_pass_1 -- best_pose_diff: 0.12948617503046989, obj_pos_diff: 0.035858746618032455, obj_rot_diff: 0.2837194800376892
    # ori_grab_s1_cubesmall_offhand_1 -- best_pose_diff: 0.12488595161587, obj_pos_diff: 0.012958609499037266, obj_rot_diff: 0.3391737639904022
    # /root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem/ori_grab_s1_cubesmall_offhand_1/coacd/decomposed.obj
    # /root/diffsim/UniDexGrasp/dexgrasp_policy/assets/meshdatav3_scaled/sem/ori_grab_s5_watch_pass_1/coacd/decomposed.obj
    # data_optimized_res_nn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-02-16-09/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-02-21-47/ts_to_hand_obj_obs_reset_1.npy'
    # data_optimized_res_nn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-02-25-58/ts_to_hand_obj_obs_reset_1.npy' # 1000, obj
    # # data_optimized_res_nn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-02-31-24/ts_to_hand_obj_obs_reset_1.npy' # 1000, hand, obj 
    # # av reward: -8.2059228515625 av steps: 149.0
    # # runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__01-11-11-34/ts_to_hand_obj_obs_reset_1.npy
    # data_inst_tag = 'ori_grab_s5_watch_pass_1'
    # data_inst_tag = 'ori_grab_s1_cubesmall_offhand_1'
    # new_optimized_info = best_optimized_res(data_optimized_res_nn, data_inst_tag, index=0)
    # obj_pose_diff = new_optimized_info['obj_pose_diff']
    # obj_pos_diff = new_optimized_info['obj_pos_diff']
    # obj_rot_diff = new_optimized_info['obj_rot_diff']
    # best_pose_diff = obj_pose_diff[0]
    # best_pos_diff, best_rot_diff = obj_pos_diff[0], obj_rot_diff[0]
    # print(f"best_pose_diff: {best_pose_diff}, obj_pos_diff: {best_pos_diff}, obj_rot_diff: {best_rot_diff}")
    # exit(0)
    
    # obj_type_to_obj_feat_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/obj_type_to_obj_feat.npy"
    # obj_type_to_obj_feat_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_/obj_type_to_obj_feat.npy"
    # inspect_obj_feats(obj_type_to_obj_feat_fn)
    # exit(0)
    
    # # obj_sample_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_"
    # # nn_bsz = 5
    # obj_sample_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnewv2_wtacodata_samples_"
    # nn_bsz = 12
    # # summarize_obj_features(obj_sample_folder, nn_bsz)
    # summarize_traj_features(obj_sample_folder, nn_bsz)
    # exit(0)
    
    # # 0-4 #
    # obj_feat_sample_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_trainAE_vnew_/samples_ep_9_batch_3.npy" 
    # inspect_obj_feat_sample(obj_feat_sample_fn)
    # exit(0)
    
    # ckpt_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_11-03-03-21/nn/tracking_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth"
    # test_ckpt(ckpt_fn)
    # exit(0)
# >>>>>>> c6fbd593bf99644e2c520da3efc0c713832c4f66
    
    # grab_obj_name_idx_dict_fn = "../assets/grab_obj_name_idx_dict.npy"
    # inspect_grab_cross_obj_diff(grab_obj_name_idx_dict_fn)
    # exit(0)
    
    # ch_to_fa_task_sv_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/child_task_to_fa_task.npy"
    # inspect_child_task_to_fa_task(ch_to_fa_task_sv_fn)
    # exit(0)
    
    # # exp_info_to_opt_res_fn = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/exp_info_to_optimized_res.npy"
    # # exp_info_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/exp_info_to_optimized_res.npy"
    # exp_info_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/exp_info_to_optimized_res.npy" # exp info #
    # self_exp_res_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res_s1_s10.npy"  
    # save_res = True 
    # # inspect_optfr_with_selfexp(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=save_res )
    # inspect_optfr_with_selfexp_list(exp_info_to_opt_res_fn, self_exp_res_to_opt_res_fn, save_res=save_res )
    # exit(0)
    
# <<<<<<< HEAD
#     subj_idx = 5
#     exp_root_folder = "./runs" # inspect exp info # # inspect the exp info # inspect the exp info # # inspect the exp info # # inspect the exp info #
#     exp_info_to_optimized_res = inspect_exp_info_to_optimized_res(exp_root_folder, subj_idx=subj_idx) 


    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2/statistics/exp_info_to_optimized_res.npy
    # subj_idx = 2
    # subj_idx = None
    # # exp_root_folder = "./runs" # inspect exp info # # inspect the exp info # inspect the exp info # # inspect the exp info # # inspect the exp info #
    # exp_root_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2"
    # exp_root_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2"
    # exp_info_to_optimized_res = inspect_exp_info_to_optimized_res(exp_root_folder, subj_idx=subj_idx) 
    
    # exit(0)
    
    
    # inst_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_inst_tag_to_optimized_res(inst_to_optimized_res_fn)
    # exit(0)
    
    
    # optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_optimized_res(optimized_res_fn)
    # exit(0)
    
    ### a direct traj optimized res fn ###
    ### a direct traj optimized res fn ###
    # exp_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # exp_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/obj_type_to_optimized_traj.npy"
    # inspect_exp_tag_to_optimized_res(exp_tag_to_optimized_res_fn)
    # exit(0)
    
    # ### save object type to optimized res ### ### 
    # # exp_root_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR"
    # exp_root_folder  = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2'
    # # 
    # calculate_OPTFR_exp_to_optimized_res(exp_root_folder)
    # exit(0)
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-17-38-45
    # # target_grab_inst_tag = 'ori_grab_s2_airplane_fly_1'
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-18-56-44/nn/tracking_ori_grab_s2_airplane_fly_1_OPTFR_ori_grab_s2_elephant_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5.pth
    # target_grab_inst_tag = 'ori_grab_s2_apple_lift'
    # target_grab_inst_tag = 'ori_grab_s2_elephant_inspect_1'
    # data_inst_to_opt_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_data_instance_to_optimized_res(data_inst_to_opt_res_fn, target_grab_inst_tag=target_grab_inst_tag)
    # exit(0)
    
    # data_statistics_fn = "/root/diffsim/softzoo/softzoo/diffusion/assets/data_statistics_ws_30_step_15.npy"
    # inspect_data_statistics(data_statistics_fn)
    # exit(0)
    
    # # ## inpsect optimized res nn ##
    # cur_inst_tag_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s8_hammer_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-37-40/ts_to_hand_obj_obs_reset_1.npy'
    # inst_tag = ['ori_grab_s8_hammer_lift', 'ori_grab_s8_hammer_lift']
    
    # cur_inst_tag_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_apple_lift_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-59-53/ts_to_hand_obj_obs_reset_1.npy'
    # inst_tag = ['ori_grab_s8_apple_lift', 'ori_grab_s8_apple_lift']
    
    # cur_inst_tag_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_eat_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-19-52-28/ts_to_hand_obj_obs_reset_1.npy'
    # inst_tag = ['ori_grab_s8_banana_eat_1', 'ori_grab_s8_banana_eat_1']
    
    # # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-20-51-58/ts_to_hand_obj_obs_reset_1.npy
    # cur_inst_tag_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/tracking_ori_grab_s8_banana_peel_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-20-51-58/ts_to_hand_obj_obs_reset_1.npy'
    # inst_tag = ['ori_grab_s8_banana_peel_1', 'ori_grab_s8_banana_peel_1']
    
    # inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag)
    # exit(0)
    
    # inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # inspect_inst_tag_to_optimized_res(inst_tag_to_optimized_res_fn)
    # exit(0)
    
    # target_eval_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2"
    # forbid_subj_idx = 7
    # mv_evaled_data_to_eval_folder(target_eval_folder, forbid_subj_idx)
    # exit(0)
    
    # new_eval_res_sv_dict_fn ="/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s1_alarmclock_pass_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-00-11-57/ts_to_hand_obj_obs_reset_1.npy"
    # new_eval_res_sv_dict_fn  = "/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s7_airplane_lift_Retake_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_20-14-31-52/ts_to_hand_obj_obs_reset_1.npy"
    # inspect_new_eval_res(new_eval_res_sv_dict_fn)
    # exit(0)
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res.npy
    # obj_type_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab/statistics/obj_type_to_optimized_res.npy"
    # obj_type_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # obj_type_to_res = np.load(obj_type_to_optimized_res_fn, allow_pickle=True).item()
    # print(obj_type_to_res.keys())
    # exit(0)
    
    # sv_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s1_alarmclock_offhand_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-19-05-48/ts_to_hand_obj_obs_reset_1.npy"
    # data_inst_tag = "ori_grab_s1_alarmclock_offhand_1"
    # sv_res = np.load(sv_res_fn, allow_pickle=True).item()
    # new_optimized_res = best_optimized_res(sv_res_fn, data_inst_tag)
    # print(f"new_optimized_res: {new_optimized_res.keys()}")
    # exit(0)
    
    # sv_res = np.load(sv_res_fn, allow_pickle=True).item()
    # for key in sv_res:
    #     cur_val = sv_res[key]
    #     print(cur_val.keys())
    #     break
    # # print(sv_res.keys())
    # exit(0)
    
    
    # # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/data_inst_tag_to_optimized_res.npy"
    # # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval/statistics/data_inst_tag_to_optimized_res.npy"
    
    
    # cur_inst_tag_optimized_res_fn = 'runs/tracking_ori_grab_s8_watch_lift_obs_pure_state_wref_wdelta_hand_allegro_density_500_trans_0.5_rot_0.5_goalcond_False_kinebais_wdelta_rewhandpos_dist__04-19-10-33/ts_to_hand_obj_obs_reset_1.npy'
    # inst_tag = 'ori_grab_s9_waterbottle_pour_1'
    # inst_tag = (inst_tag, inst_tag)
    # inspect_optimized_res_nn(cur_inst_tag_optimized_res_fn, inst_tag)
    # exit(0)
    
    
    # data_inst_tag_to_optimized_res_fn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy"
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    # # target_inst_tag = 's2'
    
    # data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval/statistics/data_inst_tag_to_optimized_res.npy'
    # data_inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_optimized_res.npy'



    data_inst_tag_to_optimized_res_fn = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/statistics/data_inst_tag_to_optimized_res.npy'
    target_inst_tag = 's10'
    
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_optimized_res.npy'
    target_inst_tag = 's10'
    # target_inst_tag = 's9'
    # target inst tag = 's9'
    data_inst_tag_to_optimized_res_fn = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_generalist_direct_samples_v4/statistics/data_inst_tag_to_optimized_res.npy'
    target_inst_tag = None

    resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=target_inst_tag)
    exit(0)

    # target_inst_tag = 's10'

    
    # target_inst_tag = None
    

    # resave_optimized_res(data_inst_tag_to_optimized_res_fn, target_sub_tag=target_inst_tag)
    # exit(0)

    
    data_optimized_res_nn = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s2_duck_inspect_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_12-23-05-11/ts_to_hand_obj_obs_reset_1.npy"
    # inspect_optimized_res_nn(data_optimized_res_nn)
    # exit(0)
    
    grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval"
    grab_eval_data_folder = os.path.join(grab_eval_data_folder, "statistics")
    
    data_inst_tag_to_optimized_res_fn  = "data_inst_tag_to_optimized_res.npy"
    data_inst_tag_to_optimized_res_fn = os.path.join(grab_eval_data_folder, data_inst_tag_to_optimized_res_fn)
    
    # inspect_data_inst_tag_to_optimized_res(data_inst_tag_to_optimized_res_fn)
    # exit(0)
    
    
    
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval"
    # grab_eval_data_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2"
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples'
    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300_eval'

    grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_'
    grab_eval_data_folder= '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/'
    grab_eval_data_folder = '/root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs'
    get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    exit(0)

    # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_'
    # grab_eval_data_folder= '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_/'
    # # grab_eval_data_folder = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_taco_grab_interpseq_eval_v2'
    # get_data_inst_tag_to_optimized_res(grab_eval_data_folder)
    # exit(0)

    #### calculating obj_type_to_optimized_res ####
    ## for calculating obj_type_to_optimized_res ##
    optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab'
    optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_reopt'
    optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300'
    optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_wdiffusion_samples_ws30_wohist_datav2_kineobjposcanon_repot'
    optimized_data_sv_root = '/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_OPTFR_v2'
    calculate_optimized_res(optimized_data_sv_root)
    exit(0)
    
    grab_inst_tag = "ori_grab_s3_pyramidlarge_inspect_1"
    # inspect_optimized_res(grab_inst_tag)
    # exit(0)
    
    training_data_target_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_nf_300"
    nf = 300
    # mv_training_data_to_folder(training_data_target_folder, nf=nf)
    # exit(0)
    
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_phone_call_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_18-00-33-11
    forbid_tag = "ori_grab_s2_phone_call_1"
    eval_folder = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval"
    mv_eval_data_to_eval_folder(eval_folder=eval_folder, forbid_tag=forbid_tag)
    exit(0)
    
# obj type to optimized res #
    # else:
    #     obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_sv_fn, allow_pickle=True).item() # true and the res sv fn #
    
        
        
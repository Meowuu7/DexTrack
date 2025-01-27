
import os
import numpy as np 
import argparse


def parse_obj_type_fr_folder_name(folder_nm):
    # /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_hammer_use_1_obs_pure_state_wref_density_500.0_trans_1.0_rot_1.0_goalcond_False_kinebias_t1r1_rewfingerdist_0.5_rewdeltahandpose_0.0_10-17-19-17
    folder_st_tag = "tracking_"
    remains_folder_nm = folder_nm.split("/")[-1][len(folder_st_tag): ]
    
    folder_nm_segs = remains_folder_nm.split("_")
    st_idx = 0
    for ed_idx in range(st_idx, len(folder_nm_segs)):
        cur_seg = folder_nm_segs[ed_idx]
        if cur_seg == 'obs':
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
    maxx_rew = max(tot_rews)
    # 
    return cur_best_rew, best_ckpt_fn

def get_obj_type_to_optimized_res(optimized_root_folder, prev_best_res_dict=None):
    tot_folders = os.listdir(optimized_root_folder)
    tracking_st_tag = 'tracking_'
    tot_folders = [
        fn for fn in tot_folders if fn[: len(tracking_st_tag)] == tracking_st_tag
    ]
    obj_type_to_optimized_res = {}
    for cur_folder in tot_folders:
        cur_full_folder = os.path.join(optimized_root_folder, cur_folder)
        cur_obj_type = parse_obj_type_fr_folder_name(cur_full_folder)
        
        cur_best_rew, best_ckpt_fn = find_best_rew(cur_full_folder)
        if prev_best_res_dict is not None:
            if cur_obj_type in prev_best_res_dict:
                prev_best_rew, prev_best_ckpt_fn = prev_best_res_dict[cur_obj_type]
                if prev_best_rew > cur_best_rew:
                    cur_best_rew = prev_best_rew
                    best_ckpt_fn = prev_best_ckpt_fn
            
        if best_ckpt_fn is None:
            continue
        obj_type_to_optimized_res[cur_obj_type] = (cur_best_rew, best_ckpt_fn)
    return obj_type_to_optimized_res


# parse the obj type from  

def find_and_save_optimized_data(optimized_data_sv_root):
    tracking_data_statistics_folder = os.path.join(optimized_data_sv_root, "statistics")
    os.makedirs(tracking_data_statistics_folder, exist_ok=True)
    obj_type_to_optimized_res_sv_fn = "obj_type_to_optimized_res.npy"
    obj_type_to_optimized_res_sv_fn = os.path.join(tracking_data_statistics_folder, obj_type_to_optimized_res_sv_fn) 
    if not os.path.exists(obj_type_to_optimized_res_sv_fn):
        obj_type_to_optimized_res = get_obj_type_to_optimized_res(optimized_data_sv_root) # get the tracking data sv root #
        print(f"obj_type_to_optimized_res: {obj_type_to_optimized_res}")
        np.save(obj_type_to_optimized_res_sv_fn, obj_type_to_optimized_res) # save the optimized res
        print(f"obj_type_to_optimized_res saved to {obj_type_to_optimized_res_sv_fn}")
    # obj type to optimized res #
    else:
        obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_sv_fn, allow_pickle=True).item() # true and the res sv fn #
        ### get the 
        obj_type_to_optimized_res = get_obj_type_to_optimized_res(optimized_data_sv_root, prev_best_res_dict=obj_type_to_optimized_res)
        np.save(obj_type_to_optimized_res_sv_fn, obj_type_to_optimized_res) # get the obj type to optimized res # 
        print(f"obj_type_to_optimized_res updated and saved to {obj_type_to_optimized_res_sv_fn}")
    
def parse_time_from_folder_name(folder_nm):
    pure_folder_name = folder_nm.split("/")[-1] # get the jpure folder name # 
    print(f"[parsing time] {pure_folder_name}")
    pure_folder_name_segs = pure_folder_name.split("_") # 
    try:
        date_tag = pure_folder_name_segs[-1]
        date_tag_segs=  date_tag.split("-")
        date_tag_day  = date_tag_segs[0]
        date_tag_day = int(date_tag_day) # int day #
    except:
        date_tag_day = None
    # date tag da
    return date_tag_day
    
# find_and_save_test_files(optimized_data_sv_root)
def find_and_save_test_files(optimized_data_sv_root):
    tot_folders = os.listdir(optimized_data_sv_root)
    tracking_st_tag = 'tracking_'
    tot_folders = [ # 
        fn for fn in tot_folders if fn[: len(tracking_st_tag)] == tracking_st_tag
    ]
    obj_type_to_optimized_res = {}
    for cur_folder in tot_folders:
        cur_full_folder = os.path.join(optimized_data_sv_root, cur_folder)
        cur_obj_type = parse_obj_type_fr_folder_name(cur_full_folder)
        cur_date_day = parse_time_from_folder_name(cur_full_folder)
        if cur_date_day is None:
            continue
        # get thejfl
        if cur_date_day == 12 or cur_date_day == 13:
            # 
            opt_res_fn = "ts_to_hand_obj_obs_reset_1.npy"
            opt_res_fn = os.path.join(cur_full_folder, opt_res_fn)
            if os.path.exists(opt_res_fn):
                obj_type_to_optimized_res[cur_obj_type] = opt_res_fn
            
    return obj_type_to_optimized_res
        
        # cur_best_rew, best_ckpt_fn = find_best_rew(cur_full_folder)
        # if prev_best_res_dict is not None:
        #     if cur_obj_type in prev_best_res_dict:
        #         prev_best_rew, prev_best_ckpt_fn = prev_best_res_dict[cur_obj_type]
        #         if prev_best_rew > cur_best_rew:
        #             cur_best_rew = prev_best_rew
        #             best_ckpt_fn = prev_best_ckpt_fn
            
        # if best_ckpt_fn is None:
        #     continue
        # obj_type_to_optimized_res[cur_obj_type] = (cur_best_rew, best_ckpt_fn)


# /root/diffsim/IsaacGymEnvs2/isaacgymenvs/runs/tracking_ori_grab_s2_cylindermedium_inspect_1_obs_pure_state_wref_density_500.0_trans_5.0_rot_1.0_goalcond_False_kinebias_t5r1_rewfingerdist_0.5_rewdeltahandpose_0.0_10-08-16-18

# python utils/isaac_eval_utils.py --optimized_data_sv_root=/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    ##### pool settings ####
    parser.add_argument("--optimized_data_sv_root", type=str, default='')
    
    args = parser.parse_args()
    
    # find_and_save_optimized_data(args.optimized_data_sv_root)
    
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/statistics/obj_type_to_optimized_traj.npy
    ## TODO: optimized data sv root #
    obj_type_to_optimized_traj = find_and_save_test_files(args.optimized_data_sv_root)
    
    print(f"obj_type_to_optimized_traj: {len(obj_type_to_optimized_traj)}")
    # statistics # parsing the date time from the folder #
    statistics_folder = os.path.join(args.optimized_data_sv_root, "statistics")
    os.makedirs(statistics_folder, exist_ok=True)
    obj_type_to_optimized_traj_sv_fn = os.path.join(statistics_folder, "obj_type_to_optimized_traj.npy") 
    np.save(obj_type_to_optimized_traj_sv_fn, obj_type_to_optimized_traj) 
    print(f"obj_type_to_optimized_traj saved to {obj_type_to_optimized_traj_sv_fn}")
    
    # optimized_data_sv_root = args.optimized_data_sv_root
    
    # tracking_data_statistics_folder = os.path.join(optimized_data_sv_root, "statistics")
    # os.makedirs(tracking_data_statistics_folder, exist_ok=True)
    # obj_type_to_optimized_res_sv_fn = "obj_type_to_optimized_res.npy"
    # obj_type_to_optimized_res_sv_fn = os.path.join(tracking_data_statistics_folder, obj_type_to_optimized_res_sv_fn) 
    # if not os.path.exists(obj_type_to_optimized_res_sv_fn):
    #     obj_type_to_optimized_res = get_obj_type_to_optimized_res(optimized_data_sv_root) # get the tracking data sv root #
    #     print(f"obj_type_to_optimized_res: {obj_type_to_optimized_res}")
    #     np.save(obj_type_to_optimized_res_sv_fn, obj_type_to_optimized_res) # save the optimized res
    #     print(f"obj_type_to_optimized_res saved to {obj_type_to_optimized_res_sv_fn}")
    # # obj type to optimized res #
    # else:
    #     obj_type_to_optimized_res = np.load(obj_type_to_optimized_res_sv_fn, allow_pickle=True).item() # true and the res sv fn #
    #     ### get the 
    #     obj_type_to_optimized_res = get_obj_type_to_optimized_res(optimized_data_sv_root, prev_best_res_dict=obj_type_to_optimized_res)
    #     np.save(obj_type_to_optimized_res_sv_fn, obj_type_to_optimized_res) # get the obj type to optimized res # 
    #     print(f"obj_type_to_optimized_res updated and saved to {obj_type_to_optimized_res_sv_fn}")
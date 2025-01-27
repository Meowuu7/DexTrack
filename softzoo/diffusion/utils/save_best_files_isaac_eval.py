import os
import numpy as np


def save_best_files(eval_root):
    tot_fns = os.listdir(eval_root)
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval/tracking_ori_grab_s1_airplane_offhand_1_obs_pure_state_wref_wdelta_density_500.0_trans_0.5_rot_0.5_goalcond_False_kinebias_t0.5r0.5f20_rfd_0.3_rh_0.5_17-18-04-10/ts_to_hand_obj_obs_reset_1_sorted.npy
    for fn in tot_fns:
        cur_full_fn = os.path.join(eval_root, fn)
        ts_to_opt_res_fn = "ts_to_hand_obj_obs_reset_1_sorted.npy"
        ts_to_opt_res_fn = os.path.join(cur_full_fn, ts_to_opt_res_fn)
        if not os.path.exists(ts_to_opt_res_fn):
            continue
        ts_to_opt_res = np.load(ts_to_opt_res_fn, allow_pickle=True).item()
        optimized_obj_pose = ts_to_opt_res['optimized_obj_pose']
        optimized_hand_qtars = ts_to_opt_res['optimized_hand_qtars']
        optimized_hand_qs = ts_to_opt_res['optimized_hand_qs']
        best_obj_pose = optimized_obj_pose[0:1]
        best_hand_qtars = optimized_hand_qtars[0:1]
        best_hand_qs = optimized_hand_qs[0:1]
        
        best_ts_to_opt_res_fn = ts_to_opt_res_fn.replace(".npy", "_best.npy")
        best_ts_to_opt_res = {
            'optimized_obj_pose': best_obj_pose,
            'optimized_hand_qtars': best_hand_qtars,
            'optimized_hand_qs': best_hand_qs
        }
        np.save(best_ts_to_opt_res_fn, best_ts_to_opt_res)
        print(f"Saved best ts_to_opt_res to {best_ts_to_opt_res_fn}")
        

def inspect_passive_act_info(passive_act_info_fn):
    sv_dict = np.load(passive_act_info_fn, allow_pickle=True).item()
    print(sv_dict.keys())    

def resave_kine_info(ori_data_folder, target_folder, nf=None):
    # /cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_cylinderlarge_lift.npy
    tot_passive_active_info_fns = os.listdir(ori_data_folder) 
    passive_active_info_st_tag = "passive_active_info_"
    tot_passive_active_info_fns = [fn for fn in tot_passive_active_info_fns if fn[: len(passive_active_info_st_tag)] == passive_active_info_st_tag]
    if nf is None:
        not_frame_tag  = "_nf_"
        tot_passive_active_info_fns = [fn for fn in tot_passive_active_info_fns if not_frame_tag not in fn]
    else:
        frame_tag  = f'_nf_{nf}_'
        tot_passive_active_info_fns = [fn for fn in tot_passive_active_info_fns if frame_tag in fn]
    
    for i_fn, fn in enumerate(tot_passive_active_info_fns):
        print(f"{i_fn}/{len(tot_passive_active_info_fns)} {fn}")
        ori_fn = os.path.join(ori_data_folder, fn)
        target_fn = os.path.join(target_folder, fn)
        sv_dict = np.load(ori_fn, allow_pickle=True).item()
        interested_keys = ['object_transl', 'object_rot_quat', 'robot_delta_states_weights_np']
        new_sv_dict = {}
        for key in interested_keys:
            new_sv_dict[key] = sv_dict[key]
        np.save(target_fn, new_sv_dict)
        print(f"Resaved {fn} to {target_fn}")
        # new_sv_dict = {
        #     'object_transl': object_transl,
            
        # }
        # np.save(target_fn, sv_dict)
        # print(f"Resaved {fn} to {target_fn}")
    
    
    

#  python utils/save_best_files_isaac_eval.py
if __name__ == "__main__":
    
    ori_data_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK/data"
    target_folder = "/cephfs/xueyi/data/GRAB_Tracking_PK_reduced/data"
    nf = None
    resave_kine_info(ori_data_folder, target_folder, nf=nf)
    exit(0)
    
    passive_act_info_fn = "/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s1_airplane_lift.npy"
    inspect_passive_act_info(passive_act_info_fn)
    exit(0)
    
    eval_root = "/cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval"
    save_best_files(eval_root)

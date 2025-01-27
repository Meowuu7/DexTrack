

import os
import numpy as np


def test_data_statistics(data_folder):
    # tot_data_fn = 
    tmp_data_list = os.listdir(data_folder)
    tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(data_folder, fn))]
    ckpt_nm = "ckpt_best.npy"
    
    data_list = []
    for fn in tmp_data_list:
        cur_data_ckpt_folder = os.path.join(data_folder, fn, "checkpoints")
        if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
            best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, ckpt_nm)
            if os.path.exists(best_ckpt_data_fn):
                data_list.append(fn)
                
    tot_maxx_link_rot_acc = []
    tot_minn_link_rot_acc = []
    tot_maxx_link_trans_acc = []
    tot_minn_link_trans_acc = []
    
    for data_nm in data_list:
        cur_data_folder_name = os.path.join(data_folder, data_nm)
        cur_data_ckpt_folder = os.path.join(cur_data_folder_name, "checkpoints")
        cur_data_ckpt_fn = os.path.join(cur_data_ckpt_folder, ckpt_nm)
        cur_data = np.load(cur_data_ckpt_fn, allow_pickle=True).item() 
        ## get teh current data ##
        cur_links_rot_acc = cur_data['link_rotational_accs'] ##
        cur_links_trans_acc = cur_data['link_translational_accs']
        
        cur_links_rot_acc = np.transpose(cur_links_rot_acc, (1, 0 ,2))
        cur_links_trans_acc = np.transpose(cur_links_trans_acc, (1, 0, 2))
        
        maxx_link_rot_acc = np.max(cur_links_rot_acc, axis=0)
        minn_link_rot_acc = np.min(cur_links_rot_acc, axis=0)
        maxx_link_rot_acc = np.max(maxx_link_rot_acc).item()
        minn_link_rot_acc = np.min(minn_link_rot_acc).item()
        
        maxx_link_trans_acc = np.max(cur_links_trans_acc, axis=0)
        minn_link_trans_acc = np.min(cur_links_trans_acc, axis=0)
        maxx_link_trans_acc = np.max(maxx_link_trans_acc).item()
        minn_link_trans_acc = np.min(minn_link_trans_acc).item() ## minn link trans acc ##
        
        tot_maxx_link_rot_acc.append(maxx_link_rot_acc)
        tot_minn_link_rot_acc.append(minn_link_rot_acc)
        tot_maxx_link_trans_acc.append(maxx_link_trans_acc)
        tot_minn_link_trans_acc.append(minn_link_trans_acc)
        
        print(f"maxx_link_rot_acc: {maxx_link_rot_acc}, minn_link_rot_acc: {minn_link_rot_acc}, maxx_link_trans_acc: {maxx_link_trans_acc}, minn_link_trans_acc: {minn_link_trans_acc}")
    maxx_link_rot_acc = max(tot_maxx_link_rot_acc)
    minn_link_rot_acc = min(tot_minn_link_rot_acc)
    
    maxx_link_trans_acc = max(tot_maxx_link_trans_acc)
    minn_link_trans_acc = min(tot_minn_link_trans_acc)
    print(f"maxx_link_rot_acc: {maxx_link_rot_acc}, minn_link_rot_acc: {minn_link_rot_acc}")
    print(f"minn_link_rot_acc: {maxx_link_trans_acc}, minn_link_trans_acc: {minn_link_trans_acc}")


def get_valid_data(data_folder, data_task_err_thres, data_trans_constraints_thres):
    ## from this one to the data with optimied res ## -- checkpoint best and the checkpoint last?
    tmp_data_list = os.listdir(data_folder)
    
    tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(data_folder, fn))]
    
    ckpt_nm = "ckpt_best_diff.npy"
    logs_data_fn = "log.txt"
    
    data_list = []
    for fn in tmp_data_list:
        cur_data_ckpt_folder = os.path.join(data_folder, fn, "checkpoints")
        if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
            
            best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, ckpt_nm)
            if os.path.exists(best_ckpt_data_fn):
                log_data_fn = os.path.join(data_folder, fn, logs_data_fn) ## get the logtsdata
                with open(log_data_fn, "r") as f:
                    log_data = f.readlines()
                log_data = [line.strip() for line in log_data]  
                last_log = log_data[-1]
                log_item = last_log.split(", ")
                goal_reaching_w_trans_reg = log_item[2]
                goal_reaching_trans_items = goal_reaching_w_trans_reg.split(" ")
                goal_reaching_loss = goal_reaching_trans_items[1]
                trans_reg = goal_reaching_trans_items[3]
                goal_reaching_loss = float(goal_reaching_loss)
                trans_reg = float(trans_reg)
                
                if goal_reaching_loss <= data_task_err_thres and trans_reg <= data_trans_constraints_thres:
                    data_list.append(fn) 
    
    print(f"total valid data: {len(data_list)}")
                
    valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
    valid_data_list_sv_fn = os.path.join(data_folder, valid_data_list_sv_fn)
    with open(valid_data_list_sv_fn, "w") as wf:
        for cur_fn in data_list:
            wf.write(f"{cur_fn}\n")
        wf.close()


def test_saved_ckpt(data_folder):
    tmp_data_list = os.listdir(data_folder)
    tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(data_folder, fn))] ## get the data folder 
    for fn in tmp_data_list:
        cur_data_folder = os.path.join(data_folder, fn)
        cur_ckpt_folder = os.path.join(cur_data_folder, "checkpoints")
        cur_ckpt_best_fn = os.path.join(cur_ckpt_folder, "ckpt_best_diff.npy")
        cur_ckpt_best = np.load(cur_ckpt_best_fn, allow_pickle=True).item()
        for key in cur_ckpt_best:
            key_value = cur_ckpt_best[key]
            print(f"key: {key}, key_value: {key_value.shape}")
        
        particle_link_idxes = cur_ckpt_best["particle_link_idx"]
        maxx_link_idx = np.max(particle_link_idxes)
        minn_link_idx = np.min(particle_link_idxes)
        print(f"maxx_link_idx: {maxx_link_idx}, minn_link_idx: {minn_link_idx}") 
        break




def get_data_pc_statistics(data_folder, valid_data_list_fn, data_task_err_thres, data_trans_constraints_thres):
    valid_data_list_fn = os.path.join(data_folder, valid_data_list_fn)
    with open(valid_data_list_fn, "r") as f:
        valid_data_list = f.readlines()
    valid_data_list = [fn.strip() for fn in valid_data_list]
    
    ckpt_nm = "ckpt_best_diff.npy"
    
    tot_maxx_link_rot_acc = []
    tot_minn_link_rot_acc = []
    # tot_maxx_link
    
    tot_particle_accs = []
    
    tot_particle_init_xs = []
    
    ## particles ## 
    
    maxx_nn_particles = 21000
    
    
    
    for cur_data_fn in valid_data_list:
        cur_data_ckpt_folder = os.path.join(data_folder, cur_data_fn, "checkpoints")
        cur_data_ckpt_fn = os.path.join(cur_data_ckpt_folder, ckpt_nm)
        cur_data = np.load(cur_data_ckpt_fn, allow_pickle=True).item()
        particle_accs = cur_data['particle_accs']
        particle_xs = cur_data['particle_xs']
        init_particle_xs = particle_xs[0]
        
        if init_particle_xs.shape[0] < maxx_nn_particles:
            rnd_sampled_particle_idxes = np.random.permutation(init_particle_xs.shape[0])[:maxx_nn_particles - init_particle_xs.shape[0]]
            rnd_sampled_particle_idxes = np.array(rnd_sampled_particle_idxes, dtype=np.int32)
            sampled_particle_xs = init_particle_xs[rnd_sampled_particle_idxes]
            sampled_particle_accs = particle_accs[:, rnd_sampled_particle_idxes]
            init_particle_xs = np.concatenate(
                [init_particle_xs, sampled_particle_xs], axis=0
            )
            particle_accs = np.concatenate(
                [particle_accs, sampled_particle_accs], axis=1
            )
            
        
        print(f"cur_data_fn: {cur_data_fn}")
        print(f"particle_accs: {particle_accs.shape}")
        
        
        tot_particle_init_xs.append(init_particle_xs)
        tot_particle_accs.append(particle_accs)
    
    tot_particle_accs = np.stack(tot_particle_accs, axis=0)
    tot_particle_init_xs = np.stack(tot_particle_init_xs, axis=0)
    
    ### nn_inits x nn_particles x 3 ##
    nn_insts, nn_particles = tot_particle_init_xs.shape[0], tot_particle_init_xs.shape[1]   
    tot_particle_accs = tot_particle_accs.transpose(0, 2, 1, 3)
    ## nn_ints 
    tot_particle_accs = tot_particle_accs.reshape(nn_insts, nn_particles, -1)
    
    maxx_init_xs = np.max(tot_particle_init_xs, axis=0)
    maxx_init_xs = np.max(maxx_init_xs, axis=0)
    
    minn_init_xs = np.min(tot_particle_init_xs, axis=0)
    minn_init_xs = np.min(minn_init_xs, axis=0)
    
    maxx_accs = np.max(tot_particle_accs, axis=0)
    maxx_accs = np.max(maxx_accs, axis=0)
    
    minn_accs = np.min(tot_particle_accs, axis=0)
    minn_accs = np.min(minn_accs, axis=0)
    
    flatten_particle_init_xs = tot_particle_init_xs.reshape(-1, tot_particle_init_xs.shape[-1])
    flatten_particle_accs = tot_particle_accs.reshape(-1, tot_particle_accs.shape[-1])
    avg_particle_init_xs, std_particle_init_xs = np.mean(flatten_particle_init_xs, axis=0), np.std(flatten_particle_init_xs, axis=0)
    avg_particle_accs, std_particle_accs = np.mean(flatten_particle_accs, axis=0), np.std(flatten_particle_accs, axis=0)
    
    print(f"maxx_init_xs: {maxx_init_xs}, minn_init_xs: {minn_init_xs}")
    print(f"maxx_accs: {maxx_accs}, minn_accs: {minn_accs}")
        # cur_links
    print(f"avg_particle_init_xs: {avg_particle_init_xs}, std_particle_init_xs: {std_particle_init_xs}")
    print(f"avg_particle_accs: {avg_particle_accs}, std_particle_accs: {std_particle_accs}")
    
    valid_data_statistics_fn = f"valid_data_statistics_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.npy"
    valid_data_statistics = {
        'maxx_init_xs': maxx_init_xs, 
        'minn_init_xs': minn_init_xs, 
        'maxx_accs': maxx_accs,
        'minn_accs': minn_accs,
        'avg_particle_init_xs': avg_particle_init_xs,
        'std_particle_init_xs': std_particle_init_xs,
        'avg_particle_accs': avg_particle_accs,
        'std_particle_accs': std_particle_accs
    }
    valid_data_statistics_fn = os.path.join(data_folder, valid_data_statistics_fn)
    np.save(valid_data_statistics_fn, valid_data_statistics) ## save the statistics ##
    
    ## try to save the statistics somewhere ? ##
    ## try to 
    
    # avg_particle_init_xs: [0.5002394  0.47571445], std_particle_init_xs: [0.16231899 0.01446833]
# avg_particle_accs: [ 0.          0.          3.498389   -1.0269269   2.2556274  -0.7489343
#   0.9322632  -0.49433887 -0.42477483 -0.20292756 -1.6678565   0.09148569
#  -2.7241378   0.3607354  -3.3360033   0.4294857  -3.3864303   0.39413443
#  -2.5081034   0.25277254], std_particle_accs: [0.         0.         2.5464897  1.1182752  1.4573069  0.68999374
#  0.50242215 0.26281303 0.9713965  0.39495984 1.6394814  0.76621413
#  2.1391368  1.0337377  2.3263674  1.0681769  2.4005644  1.0920538
#  2.5788157  1.4876144 ]


def get_valid_data_v2(root_data_folder, data_task_err_thres, data_trans_constraints_thres):
    ## from this one to the data with optimied res ## -- checkpoint best and the checkpoint last?
    
    
    exp_tags = ["expv4_projected", "expv4_projected_task_0", "expv4_projected_task_2"]
    
    for exp_tag in exp_tags:
        data_folder = os.path.join(root_data_folder, exp_tag)
        
        tmp_data_list = os.listdir(data_folder)
        tmp_data_list = [fn for fn in tmp_data_list if os.path.isdir(os.path.join(data_folder, fn))]
        
        ckpt_nm = "ckpt_best_diff.npy"
        logs_data_fn = "log.txt"
        
        data_list = []
        for fn in tmp_data_list:
            cur_data_ckpt_folder = os.path.join(data_folder, fn, "checkpoints")
            if os.path.exists(cur_data_ckpt_folder) and os.path.isdir(cur_data_ckpt_folder):
                
                best_ckpt_data_fn = os.path.join(cur_data_ckpt_folder, ckpt_nm)
                if os.path.exists(best_ckpt_data_fn):
                    log_data_fn = os.path.join(data_folder, fn, logs_data_fn) ## get the logtsdata
                    with open(log_data_fn, "r") as f:
                        log_data = f.readlines()
                    log_data = [line.strip() for line in log_data]  
                    last_log = log_data[-1]
                    log_item = last_log.split(", ")
                    goal_reaching_w_trans_reg = log_item[2]
                    goal_reaching_trans_items = goal_reaching_w_trans_reg.split(" ")
                    goal_reaching_loss = goal_reaching_trans_items[1]
                    trans_reg = goal_reaching_trans_items[3]
                    goal_reaching_loss = float(goal_reaching_loss)
                    trans_reg = float(trans_reg)
                    
                    if goal_reaching_loss <= data_task_err_thres and trans_reg <= data_trans_constraints_thres:
                        data_list.append(fn) 
        
        print(f"total valid data: {len(data_list)}")
                    
        valid_data_list_sv_fn = f"valid_data_list_taskerrthres{data_task_err_thres}_transconsthres{data_trans_constraints_thres}.txt"
        valid_data_list_sv_fn = os.path.join(data_folder, valid_data_list_sv_fn)
        with open(valid_data_list_sv_fn, "w") as wf:
            for cur_fn in data_list:
                wf.write(f"{cur_fn}\n")
            wf.close()



def convert_data_dict(sampled_data_dict_fn):
    # /data/xueyi/uni_manip/exp/eval_v2_/eval_v2_/sampled_pcd_wact_dict.npy
    sampled_data_dict = np.load(sampled_data_dict_fn, allow_pickle=True).item()
    sampled_data_dict_np = {
        key: sampled_data_dict[key].detach().cpu().numpy() for key in sampled_data_dict
    }
    sampled_data_dict_folder = "/data/xueyi/uni_manip/exp/eval_v2_/eval_v2_"
    sampled_data_dict_np_fn = "sampled_pcd_wact_dict_np.npy"
    sampled_data_dict_np_fn = os.path.join(sampled_data_dict_folder, sampled_data_dict_np_fn)
    np.save(sampled_data_dict_np_fn, sampled_data_dict_np)
    print(f"Sampled data saved to {sampled_data_dict_np_fn}")
    
    
def get_3d_pc_statistics(data_folder):
    # exp_tag #
    saved_info_fn = "saved_info_accs.npy"
    tot_folders = os.listdir(data_folder)
    tot_folders = [fn for fn in tot_folders if os.path.isdir(os.path.join(data_folder, fn))]
    tot_init_pts = []
    tot_pts_accs_tau = []
    tot_pts_accs = []
    tot_pts_accs_final = []
    for cur_fn in tot_folders:
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        cur_init_pts  = saved_info['robot_init_visual_pts']
        cur_pts_accs_tau = saved_info['tot_pts_accs_tau']
        cur_pts_accs = saved_info['tot_pts_accs']
        cur_pts_accs_final = saved_info['tot_pts_accs_final']
        tot_init_pts.append(cur_init_pts)
        
        cur_pts_accs_tau = cur_pts_accs_tau.transpose(1, 0, 2)
        cur_pts_accs_tau = cur_pts_accs_tau.reshape(cur_pts_accs_tau.shape[0], -1) 
        
        cur_pts_accs = cur_pts_accs.transpose(1, 0, 2)
        cur_pts_accs = cur_pts_accs.reshape(cur_pts_accs.shape[0], -1) 
        
        cur_pts_accs_final = cur_pts_accs_final.transpose(1, 0, 2)
        cur_pts_accs_final = cur_pts_accs_final.reshape(cur_pts_accs_final.shape[0], -1) 
        
        tot_pts_accs_tau.append(cur_pts_accs_tau)
        tot_pts_accs.append(cur_pts_accs)
        tot_pts_accs_final.append(cur_pts_accs_final)
    tot_init_pts = np.concatenate(tot_init_pts, axis=0)
    tot_pts_accs_tau = np.concatenate(tot_pts_accs_tau, axis=0)
    tot_pts_accs = np.concatenate(tot_pts_accs, axis=0)
    tot_pts_accs_final = np.concatenate(tot_pts_accs_final, axis=0)
    
    avg_init_pts = np.mean(tot_init_pts, axis=0)
    std_init_pts = np.std(tot_init_pts, axis=0)
    avg_tot_pts_accs_tau = np.mean(tot_pts_accs_tau, axis=0)
    std_tot_pts_accs_tau = np.std(tot_pts_accs_tau, axis=0)
    avg_tot_pts_accs = np.mean(tot_pts_accs, axis=0)
    std_tot_pts_accs = np.std(tot_pts_accs, axis=0)
    avg_tot_pts_accs_final = np.mean(tot_pts_accs_final, axis=0)

    std_tot_pts_accs_final = np.std(tot_pts_accs_final, axis=0)
    
    statistics = {
        'robot_init_visual_pts_avg': avg_init_pts,
        'robot_init_visual_pts_std': std_init_pts,
        'pts_accs_tau_avg': avg_tot_pts_accs_tau,
        'pts_accs_tau_std': std_tot_pts_accs_tau,
        'pts_accs_avg': avg_tot_pts_accs,
        'pts_accs_std': std_tot_pts_accs,
        'pts_accs_final_avg': avg_tot_pts_accs_final,
        'pts_accs_final_std': std_tot_pts_accs_final
    }
    statistics_sv_fn = os.path.join(data_folder, "valid_data_statistics.npy")
    np.save(statistics_sv_fn, statistics)
    print(f"Statistics saved to {statistics_sv_fn}")
    
    pass


def get_3d_pc_statistics_v2(data_folder):
    # exp_tag #
    saved_info_fn = "saved_info_accs_v2.npy"
    tot_folders = os.listdir(data_folder)
    
    # tot_folders = [fn for fn in tot_folders if os.path.isdir(os.path.join(data_folder, fn))]
    
    # tot_folders = ["allegro_bouncing_ball_task_0_trail6_"]
    # tot_folders = ["allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_"]
    # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
    tot_folders = ["allegro_bouncing_ball_task_0_trail6_"]
    
    tot_init_pts = []
    tot_pts_accs_tau = []
    tot_pts_accs = []
    tot_pts_accs_final = []
    for cur_fn in tot_folders:
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        if not os.path.exists(cur_fn_full):
            continue
        print(f"cur_fn_full: {cur_fn_full}")
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        cur_init_pts  = saved_info['tot_verts'][0]
        cur_pts_accs_tau = saved_info['tot_verts_dd_tau']
        cur_pts_accs = saved_info['tot_verts_dd']
        cur_pts_accs_final = saved_info['tot_verts_dd_final']
        tot_init_pts.append(cur_init_pts)
        
        cur_pts_accs_tau = cur_pts_accs_tau.transpose(1, 0, 2)
        cur_pts_accs_tau = cur_pts_accs_tau.reshape(cur_pts_accs_tau.shape[0], -1) 
        
        cur_pts_accs = cur_pts_accs.transpose(1, 0, 2)
        cur_pts_accs = cur_pts_accs.reshape(cur_pts_accs.shape[0], -1) 
        
        cur_pts_accs_final = cur_pts_accs_final.transpose(1, 0, 2)
        cur_pts_accs_final = cur_pts_accs_final.reshape(cur_pts_accs_final.shape[0], -1) 
        
        tot_pts_accs_tau.append(cur_pts_accs_tau)
        tot_pts_accs.append(cur_pts_accs)
        tot_pts_accs_final.append(cur_pts_accs_final)
    tot_init_pts = np.concatenate(tot_init_pts, axis=0)
    tot_pts_accs_tau = np.concatenate(tot_pts_accs_tau, axis=0)
    tot_pts_accs = np.concatenate(tot_pts_accs, axis=0)
    tot_pts_accs_final = np.concatenate(tot_pts_accs_final, axis=0)
    
    avg_init_pts = np.mean(tot_init_pts, axis=0)
    std_init_pts = np.std(tot_init_pts, axis=0)
    avg_tot_pts_accs_tau = np.mean(tot_pts_accs_tau, axis=0)
    std_tot_pts_accs_tau = np.std(tot_pts_accs_tau, axis=0)
    avg_tot_pts_accs = np.mean(tot_pts_accs, axis=0)
    std_tot_pts_accs = np.std(tot_pts_accs, axis=0)
    avg_tot_pts_accs_final = np.mean(tot_pts_accs_final, axis=0)

    std_tot_pts_accs_final = np.std(tot_pts_accs_final, axis=0)
    
    statistics = {
        'robot_init_visual_pts_avg': avg_init_pts,
        'robot_init_visual_pts_std': std_init_pts,
        'pts_accs_tau_avg': avg_tot_pts_accs_tau,
        'pts_accs_tau_std': std_tot_pts_accs_tau,
        'pts_accs_avg': avg_tot_pts_accs,
        'pts_accs_std': std_tot_pts_accs,
        'pts_accs_final_avg': avg_tot_pts_accs_final,
        'pts_accs_final_std': std_tot_pts_accs_final
    }
    statistics_sv_fn = os.path.join(data_folder, "valid_data_statistics_v2.npy")
    np.save(statistics_sv_fn, statistics)
    print(f"Statistics saved to {statistics_sv_fn}")
    
    pass



def get_3d_pc_statistics_v3(data_folder):
    # exp_tag #
    saved_info_fn = "saved_info_accs_v3.npy"
    tot_folders = os.listdir(data_folder)
    
    # tot_folders = [fn for fn in tot_folders if os.path.isdir(os.path.join(data_folder, fn))]
    
    # tot_folders = ["allegro_bouncing_ball_task_0_trail6_"]
    # tot_folders = ["allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_"]
    # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
    # tot_folders = ["allegro_bouncing_ball_task_0_trail6_"]
    
    # tot_init_pts = []
    # tot_pts_accs_tau = []
    # tot_pts_accs = []
    # tot_pts_accs_final = []
    
    verts_tot_cases_tot_ts = []
    verts_qdd_tau_tot_cases_tot_ts = []
    # tot vertices in each timestep # and the related informaiton # # state-action pairs #
    for cur_fn in tot_folders:
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        if not os.path.exists(cur_fn_full): 
            continue
        print(f"cur_fn_full: {cur_fn_full}")
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        
        # tot_verts tot_verts_integrated_qdd_tau
        tot_verts = saved_info['tot_verts']
        tot_verts_integrated_qdd_tau = saved_info['tot_verts_integrated_qdd_tau'] # 
        # nn_ts x nn_verts x 3 
        
        print(f"tot_verts: {tot_verts.shape}, tot_verts_integrated_qdd_tau: {tot_verts_integrated_qdd_tau.shape}")
        
        # 
        # start position to 0 #
        tot_verts_expanded = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
        tot_verts_integrated_qdd_tau_expanded = tot_verts_integrated_qdd_tau.reshape(tot_verts_integrated_qdd_tau.shape[0] * tot_verts_integrated_qdd_tau.shape[1], -1) ## get
        verts_tot_cases_tot_ts.append(tot_verts_expanded)
        verts_qdd_tau_tot_cases_tot_ts.append(tot_verts_integrated_qdd_tau_expanded) ## integrated 
        
        # # cur_init_pts  = saved_info['tot_verts'][0]
        # # cur_pts_accs_tau = saved_info['tot_verts_dd_tau']
        # # cur_pts_accs = saved_info['tot_verts_dd']
        # # cur_pts_accs_final = saved_info['tot_verts_dd_final']
        # # tot_init_pts.append(cur_init_pts)
        
        # cur_pts_accs_tau = cur_pts_accs_tau.transpose(1, 0, 2)
        # cur_pts_accs_tau = cur_pts_accs_tau.reshape(cur_pts_accs_tau.shape[0], -1) 
        
        # cur_pts_accs = cur_pts_accs.transpose(1, 0, 2)
        # cur_pts_accs = cur_pts_accs.reshape(cur_pts_accs.shape[0], -1) 
        
        # cur_pts_accs_final = cur_pts_accs_final.transpose(1, 0, 2)
        # cur_pts_accs_final = cur_pts_accs_final.reshape(cur_pts_accs_final.shape[0], -1) 
        
        # tot_pts_accs_tau.append(cur_pts_accs_tau)
        # tot_pts_accs.append(cur_pts_accs)
        # tot_pts_accs_final.append(cur_pts_accs_final)
    
    verts_tot_cases_tot_ts = np.concatenate(verts_tot_cases_tot_ts, axis=0)
    verts_qdd_tau_tot_cases_tot_ts = np.concatenate(verts_qdd_tau_tot_cases_tot_ts, axis=0)     

    avg_verts_tot_cases_tot_ts = np.mean(verts_tot_cases_tot_ts, axis=0)
    std_verts_tot_cases_tot_ts = np.std(verts_tot_cases_tot_ts, axis=0)
    
    avg_verts_qdd_tau_tot_cases_tot_ts = np.mean(verts_qdd_tau_tot_cases_tot_ts, axis=0)
    std_verts_qdd_tau_tot_cases_tot_ts = np.std(verts_qdd_tau_tot_cases_tot_ts, axis=0)
    
    
    # tot_init_pts = np.concatenate(tot_init_pts, axis=0)
    # tot_pts_accs_tau = np.concatenate(tot_pts_accs_tau, axis=0)
    # tot_pts_accs = np.concatenate(tot_pts_accs, axis=0)
    # tot_pts_accs_final = np.concatenate(tot_pts_accs_final, axis=0)
    
    # avg_init_pts = np.mean(tot_init_pts, axis=0)
    # std_init_pts = np.std(tot_init_pts, axis=0)
    # avg_tot_pts_accs_tau = np.mean(tot_pts_accs_tau, axis=0)
    # std_tot_pts_accs_tau = np.std(tot_pts_accs_tau, axis=0)
    # avg_tot_pts_accs = np.mean(tot_pts_accs, axis=0)
    # std_tot_pts_accs = np.std(tot_pts_accs, axis=0)
    # avg_tot_pts_accs_final = np.mean(tot_pts_accs_final, axis=0)

    # std_tot_pts_accs_final = np.std(tot_pts_accs_final, axis=0)
    
    statistics = {
        'avg_verts_tot_cases_tot_ts': avg_verts_tot_cases_tot_ts,
        'std_verts_tot_cases_tot_ts': std_verts_tot_cases_tot_ts,
        'avg_verts_qdd_tau_tot_cases_tot_ts': avg_verts_qdd_tau_tot_cases_tot_ts,
        'std_verts_qdd_tau_tot_cases_tot_ts': std_verts_qdd_tau_tot_cases_tot_ts,
        # 'pts_accs_avg': avg_tot_pts_accs,
        # 'pts_accs_std': std_tot_pts_accs,
        # 'pts_accs_final_avg': avg_tot_pts_accs_final,
        # 'pts_accs_final_std': std_tot_pts_accs_final
    }
    # validat data statistics ##
    statistics_sv_fn = os.path.join(data_folder, f"valid_data_statistics_v3_all.npy")
    np.save(statistics_sv_fn, statistics)
    print(f"Statistics saved to {statistics_sv_fn}")
    
    pass

# the test data v3 and the test data #



### get 3d pc and statistics ###

def get_3d_pc_statistics_v4(data_folder, nn_stages=5): 
    # exp_tag #
    
    saved_info_fn = f"saved_info_accs_v4_nstages_{nn_stages}.npy" # saved #
    tot_folders = os.listdir(data_folder)
    
    # tot_folders = [fn for fn in tot_folders if os.path.isdir(os.path.join(data_folder, fn))]
    
    # tot_folders = ["allegro_bouncing_ball_task_0_trail6_"]
    # tot_folders = ["allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_"]
    # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
    # tot_folders = ["allegro_bouncing_ball_task_0_trail6_"] # 
    
    # tot_init_pts = []
    # tot_pts_accs_tau = []
    # tot_pts_accs = []
    # tot_pts_accs_final = []
    
    verts_tot_cases_tot_ts = []
    verts_qdd_tau_tot_cases_tot_ts = []
    
    verts_qd_tot_cases_tot_ts = [] # verts got by forwarding the qs integrated by qds #
    
    
    single_clip_length = 300
    sliding_window_length = 100
    
    # 1) centralize verts by setting the hand root to the original point #
    # 2) centralize 
    
    # tot vertices in each timeste #
    for cur_fn in tot_folders: # #
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        if not os.path.exists(cur_fn_full): 
            continue
        print(f"cur_fn_full: {cur_fn_full}")
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        
        # tot_verts tot_verts_integrated_qdd_tau
        tot_verts = saved_info['tot_verts'] #
        tot_verts_integrated_qdd_tau = saved_info['tot_verts_integrated_qdd_tau'] #
        # nn_ts x nn_verts x 3 #
        tot_verts_integrated_qd = saved_info['tot_verts_integrated_qd'] #
        
        print(f"tot_verts: {tot_verts.shape}, tot_verts_integrated_qdd_tau: {tot_verts_integrated_qdd_tau.shape}") # 
        
        # tot verts integrated with qd #
        # tot verts integrated with qdd_tau #
        # tot verts integrated with qdd_tau # 
        # integrated with qdd taus # # 
        # get the 
        
        
        
        # get the clip #
        for i_starting_ts in range(0, tot_verts.shape[1] - single_clip_length, sliding_window_length):
            # get the starting ts and the ending ts #
            cur_ending_ts = i_starting_ts + single_clip_length
            cur_tot_verts = tot_verts[:, i_starting_ts:cur_ending_ts]
            cur_tot_verts_integrated_qd = tot_verts_integrated_qd[:, i_starting_ts: cur_ending_ts]
            cur_tot_verts_integrated_qdd_tau = tot_verts_integrated_qdd_tau[:, i_starting_ts: cur_ending_ts] 
            
            
            first_fr_cur_tot_verts = cur_tot_verts[:, 0]
            ### TODO: another cnetralization strategy  ? ##
            first_fr_verts_offset = first_fr_cur_tot_verts[0] ## (3,) - shape of the offset tensor #
            
            cur_tot_verts = cur_tot_verts - first_fr_verts_offset[None][None] ## the cur_tot_verts - (1,1,3)
            cur_tot_verts_integrated_qd = cur_tot_verts_integrated_qd - first_fr_verts_offset[None][None] 
            cur_tot_verts_integrated_qdd_tau = cur_tot_verts_integrated_qdd_tau - first_fr_verts_offset[None][None] 
            
            
            cur_tot_verts_expanded = cur_tot_verts.reshape(cur_tot_verts.shape[0] * cur_tot_verts.shape[1], -1)
            cur_tot_verts_integrated_qd_expanded = cur_tot_verts_integrated_qd.reshape(cur_tot_verts_integrated_qd.shape[0] * cur_tot_verts_integrated_qd.shape[1], -1)
            cur_tot_verts_integrated_qdd_tau_expanded = cur_tot_verts_integrated_qdd_tau.reshape(cur_tot_verts_integrated_qdd_tau.shape[0] * cur_tot_verts_integrated_qdd_tau.shape[1], -1)
            
            verts_tot_cases_tot_ts.append(cur_tot_verts_expanded)
            verts_qd_tot_cases_tot_ts.append(cur_tot_verts_integrated_qd_expanded)
            verts_qdd_tau_tot_cases_tot_ts.append(cur_tot_verts_integrated_qdd_tau_expanded)
            #  get verts, verts_qd, verts_qdd_tau #
            

    
    verts_tot_cases_tot_ts = np.concatenate(verts_tot_cases_tot_ts, axis=0)
    verts_qdd_tau_tot_cases_tot_ts = np.concatenate(verts_qdd_tau_tot_cases_tot_ts, axis=0)     

    avg_verts_tot_cases_tot_ts = np.mean(verts_tot_cases_tot_ts, axis=0)
    std_verts_tot_cases_tot_ts = np.std(verts_tot_cases_tot_ts, axis=0)
    
    avg_verts_qdd_tau_tot_cases_tot_ts = np.mean(verts_qdd_tau_tot_cases_tot_ts, axis=0)
    std_verts_qdd_tau_tot_cases_tot_ts = np.std(verts_qdd_tau_tot_cases_tot_ts, axis=0)
    
    
    statistics = {
        'avg_verts_tot_cases_tot_ts': avg_verts_tot_cases_tot_ts,
        'std_verts_tot_cases_tot_ts': std_verts_tot_cases_tot_ts,
        'avg_verts_qdd_tau_tot_cases_tot_ts': avg_verts_qdd_tau_tot_cases_tot_ts,
        'std_verts_qdd_tau_tot_cases_tot_ts': std_verts_qdd_tau_tot_cases_tot_ts,
        # 'pts_accs_avg': avg_tot_pts_accs,
        # 'pts_accs_std': std_tot_pts_accs,
        # 'pts_accs_final_avg': avg_tot_pts_accs_final,
        # 'pts_accs_final_std': std_tot_pts_accs_final # to
    }
    # validate data statistics #
    # validate data statistics #
    statistics_sv_fn = os.path.join(data_folder, f"valid_data_statistics_v4.npy")
    np.save(statistics_sv_fn, statistics)
    print(f"Statistics saved to {statistics_sv_fn}")
    
    pass


def get_best_loss_from_logs(logs_fn):
    with open(logs_fn, "r") as f:
        log_data = f.readlines()
    log_data = [line.strip() for line in log_data if "[#Iter" in line]  
    last_log = log_data[-1]
    log_item = last_log.split(", ")
    print(f"log_item: {log_item}")
    # [#Iter = 13] loss = 1.240745, best_loss: 0.16034677448691737,
    best_loss_item = log_item[1]
    best_loss_items = best_loss_item.split(": ")
    best_loss = best_loss_items[1]
    if best_loss[-3:] == "nan" or best_loss[-3:] == "inf":
        best_loss = 1000.0
    else:
        best_loss = float(best_loss)
    return best_loss


def get_3d_pc_statistics_v5(data_folder, nn_stages=5): 
    # exp_tag #
    
    saved_info_fn = "save_info_v5.npy"
    
    # saved_info_fn = f"saved_info_accs_v5_nstages_{nn_stages}.npy" # saved #
    tot_folders = os.listdir(data_folder)
    
    
    single_clip_length = 300
    sliding_window_length = 100
    
    tot_expanded_verts = []
    tot_expanded_qtar_verts = []
    
    loss_thres = 0.3
    
    # 1) centralize verts by setting the hand root to the original point #
    # 2) centralize 
    
    # tot vertices in each timeste #
    for cur_fn in tot_folders:
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        if not os.path.exists(cur_fn_full): 
            continue
        print(f"cur_fn_full: {cur_fn_full}")
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        
        save_cur_inst_statistics_info_fn = "save_info_v5_statistics.npy"
        save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        
        if os.path.exists(save_cur_inst_statistics_info_fn):
            os.system(f"rm {save_cur_inst_statistics_info_fn}")
        
        logs_fn = os.path.join(data_folder, cur_fn, "logs.txt")
        best_loss = get_best_loss_from_logs(logs_fn) #
        if best_loss > loss_thres:
            continue
        
        tot_verts = saved_info['tot_verts']
        tot_qtar_verts = saved_info['tot_qtar_verts']
        
        mean_tot_verts = np.mean(tot_verts, axis=1)
        mean_tot_verts = np.mean(mean_tot_verts, axis=0)
        
        mean_tot_verts_qdd = np.mean(tot_qtar_verts, axis=1)
        mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
        
        if np.any(np.isnan(mean_tot_verts)) or np.any(np.isnan(mean_tot_verts_qdd)):
            continue
        
        
        cur_inst_tot_expanded_verts = []
        cur_inst_tot_expanded_qtar_verts = []
        
        # get the clip #
        for i_starting_ts in range(0, tot_verts.shape[1] - single_clip_length, sliding_window_length):
            # get the starting ts and the ending ts #
            cur_ending_ts = i_starting_ts + single_clip_length
            cur_tot_verts = tot_verts[:, i_starting_ts:cur_ending_ts]
            cur_tot_qtar_verts = tot_qtar_verts[:, i_starting_ts: cur_ending_ts]
            
            first_fr_verts_offset = cur_tot_verts[:, 0]
            first_fr_verts_offset = first_fr_verts_offset[0]
            
            cur_tot_verts = cur_tot_verts - first_fr_verts_offset[None][None]
            cur_tot_qtar_verts = cur_tot_qtar_verts - first_fr_verts_offset[None][None]
            
            cur_expanded_verts = cur_tot_verts.reshape(cur_tot_verts.shape[0] * cur_tot_verts.shape[1], -1)
            cur_expanded_qtar_verts = cur_tot_qtar_verts.reshape(cur_tot_qtar_verts.shape[0] * cur_tot_qtar_verts.shape[1], -1)
            
            tot_expanded_verts.append(cur_expanded_verts)
            tot_expanded_qtar_verts.append(cur_expanded_qtar_verts)
            
            cur_inst_tot_expanded_verts.append(cur_expanded_verts)
            cur_inst_tot_expanded_qtar_verts.append(cur_expanded_qtar_verts)
        
        cur_inst_tot_expanded_qtar_verts = np.concatenate(cur_inst_tot_expanded_qtar_verts, axis=0)
        cur_inst_tot_expanded_verts = np.concatenate(cur_inst_tot_expanded_verts, axis=0)
        
        avg_cur_inst_tot_expanded_verts = np.mean(cur_inst_tot_expanded_verts, axis=0)
        std_cur_inst_tot_expanded_verts = np.std(cur_inst_tot_expanded_verts, axis=0)
        avg_cur_inst_tot_expanded_qtar_verts = np.mean(cur_inst_tot_expanded_qtar_verts, axis=0)
        std_cur_inst_tot_expanded_qtar_verts = np.std(cur_inst_tot_expanded_qtar_verts, axis=0)
        
        save_cur_inst_statistics_info = {
            'avg_verts': avg_cur_inst_tot_expanded_verts,
            'std_verts': std_cur_inst_tot_expanded_verts,
            'avg_qtar_verts': avg_cur_inst_tot_expanded_qtar_verts,
            'std_qtar_verts': std_cur_inst_tot_expanded_qtar_verts
        }
        
        save_cur_inst_statistics_info_fn = "save_info_v5_statistics.npy"
        save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
        
            
    tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0)
    
    avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    
    statistics = {
        'avg_verts': avg_tot_expanded_verts,
        'std_verts': std_tot_expanded_verts,
        'avg_qtar_verts': avg_tot_expanded_qtar_verts,
        'std_qtar_verts': std_tot_expanded_qtar_verts
    }
    save_cur_inst_statistics_info_fn = "save_info_v5_statistics.npy"
    save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    np.save(save_cur_inst_statistics_info_fn, statistics)
    




def get_3d_pc_statistics_v6(data_folder, nn_stages=5): 
    # exp_tag #
    ## get the version index ? ##
    ## use a separate model to depict the version index ? ##
    saved_info_fn = "save_info_v6.npy"
    
    
    # saved_info_fn = f"saved_info_accs_v5_nstages_{nn_stages}.npy" # saved #
    tot_folders = os.listdir(data_folder)
    # single clip length ## 
    
    
    
    
    # single_clip_length = 300
    # sliding_window_length = 100
    
    tot_expanded_verts = []
    tot_expanded_qtar_verts = []
    
    loss_thres = 0.3
    
    # 1) centralize verts by setting the hand root to the original point #
    # 2) centralize verts by setting the hand root to the origianl point #
    
    # tot vertices in each timeste #
    for cur_fn in tot_folders: # cur fn in the tot folders # #
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        if not os.path.exists(cur_fn_full): 
            continue
        print(f"cur_fn_full: {cur_fn_full}")
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        
        save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
        save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        
        if os.path.exists(save_cur_inst_statistics_info_fn):
            os.system(f"rm {save_cur_inst_statistics_info_fn}")
            
        cost_from_saved_info = saved_info['cost']
        best_loss = cost_from_saved_info
        
        if np.isnan(best_loss):
            continue
        
        if best_loss > loss_thres:
            continue
        
        print(f"best_loss: {best_loss}")
        # selected_frame_verts, selected_frame_qtars_verts
        selected_frame_verts = saved_info['selected_frame_verts']
        selected_frame_qtars_verts = saved_info['selected_frame_qtars_verts'] # 
        
        ## nn_obj_pts x nn_frames x 3 ##
        ## nn_obj_pts x nn_frames x 3 ##
        avg_selected_verts = np.mean(selected_frame_verts, axis=1)
        avg_selected_verts = np.mean(avg_selected_verts, axis=0)
        
        avg_selected_qtars_verts = np.mean(selected_frame_qtars_verts, axis=1)
        avg_selected_qtars_verts = np.mean(avg_selected_qtars_verts, axis=0)
        
        if np.any(np.isnan(avg_selected_verts)) or np.any(np.isnan(avg_selected_qtars_verts)):
            continue
        
        cur_expanded_selected_frames_verts = selected_frame_verts.reshape(selected_frame_verts.shape[0] * selected_frame_verts.shape[1], -1)
        cur_expanded_selected_frames_qtars_verts = selected_frame_qtars_verts.reshape(selected_frame_qtars_verts.shape[0] * selected_frame_qtars_verts.shape[1], -1) 
        
        avg_verts = np.mean(cur_expanded_selected_frames_verts, axis=0)
        std_verts = np.std(cur_expanded_selected_frames_verts, axis=0)
        avg_qtars_verts = np.mean(cur_expanded_selected_frames_qtars_verts, axis=0)
        std_qtars_verts = np.std(cur_expanded_selected_frames_qtars_verts, axis=0)
        
        tot_expanded_verts.append(cur_expanded_selected_frames_verts)
        tot_expanded_qtar_verts.append(cur_expanded_selected_frames_qtars_verts) 
        
        
        save_cur_inst_statistics_info = {
            'avg_verts': avg_verts,
            'std_verts': std_verts,
            'avg_qtar_verts': avg_qtars_verts,
            'std_qtar_verts': std_qtars_verts
        }
        
        # save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
        # save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
    
    tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0) 
    
    avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    
    statistics = {
        'avg_verts': avg_tot_expanded_verts,
        'std_verts': std_tot_expanded_verts,
        'avg_qtar_verts': avg_tot_expanded_qtar_verts,
        'std_qtar_verts': std_tot_expanded_qtar_verts
    }
    
    save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    np.save(save_cur_inst_statistics_info_fn, statistics)
    
    
    #     logs_fn = os.path.join(data_folder, cur_fn, "logs.txt")
    #     best_loss = get_best_loss_from_logs(logs_fn) #
    #     if best_loss > loss_thres:
    #         continue
        
    #     tot_verts = saved_info['tot_verts']
    #     tot_qtar_verts = saved_info['tot_qtar_verts']
        
    #     mean_tot_verts = np.mean(tot_verts, axis=1)
    #     mean_tot_verts = np.mean(mean_tot_verts, axis=0)
        
    #     mean_tot_verts_qdd = np.mean(tot_qtar_verts, axis=1)
    #     mean_tot_verts_qdd = np.mean(mean_tot_verts_qdd, axis=0)
        
    #     if np.any(np.isnan(mean_tot_verts)) or np.any(np.isnan(mean_tot_verts_qdd)):
    #         continue
        
        
    #     cur_inst_tot_expanded_verts = []
    #     cur_inst_tot_expanded_qtar_verts = []
        
    #     # get the clip #
    #     for i_starting_ts in range(0, tot_verts.shape[1] - single_clip_length, sliding_window_length):
    #         # get the starting ts and the ending ts #
    #         cur_ending_ts = i_starting_ts + single_clip_length
    #         cur_tot_verts = tot_verts[:, i_starting_ts:cur_ending_ts]
    #         cur_tot_qtar_verts = tot_qtar_verts[:, i_starting_ts: cur_ending_ts]
            
    #         first_fr_verts_offset = cur_tot_verts[:, 0]
    #         first_fr_verts_offset = first_fr_verts_offset[0]
            
    #         cur_tot_verts = cur_tot_verts - first_fr_verts_offset[None][None]
    #         cur_tot_qtar_verts = cur_tot_qtar_verts - first_fr_verts_offset[None][None]
            
    #         cur_expanded_verts = cur_tot_verts.reshape(cur_tot_verts.shape[0] * cur_tot_verts.shape[1], -1)
    #         cur_expanded_qtar_verts = cur_tot_qtar_verts.reshape(cur_tot_qtar_verts.shape[0] * cur_tot_qtar_verts.shape[1], -1)
            
    #         tot_expanded_verts.append(cur_expanded_verts)
    #         tot_expanded_qtar_verts.append(cur_expanded_qtar_verts)
            
    #         cur_inst_tot_expanded_verts.append(cur_expanded_verts)
    #         cur_inst_tot_expanded_qtar_verts.append(cur_expanded_qtar_verts)
        
    #     cur_inst_tot_expanded_qtar_verts = np.concatenate(cur_inst_tot_expanded_qtar_verts, axis=0)
    #     cur_inst_tot_expanded_verts = np.concatenate(cur_inst_tot_expanded_verts, axis=0)
        
    #     avg_cur_inst_tot_expanded_verts = np.mean(cur_inst_tot_expanded_verts, axis=0)
    #     std_cur_inst_tot_expanded_verts = np.std(cur_inst_tot_expanded_verts, axis=0)
    #     avg_cur_inst_tot_expanded_qtar_verts = np.mean(cur_inst_tot_expanded_qtar_verts, axis=0)
    #     std_cur_inst_tot_expanded_qtar_verts = np.std(cur_inst_tot_expanded_qtar_verts, axis=0)
        
    #     save_cur_inst_statistics_info = {
    #         'avg_verts': avg_cur_inst_tot_expanded_verts,
    #         'std_verts': std_cur_inst_tot_expanded_verts,
    #         'avg_qtar_verts': avg_cur_inst_tot_expanded_qtar_verts,
    #         'std_qtar_verts': std_cur_inst_tot_expanded_qtar_verts
    #     }
        
    #     save_cur_inst_statistics_info_fn = "save_info_v5_statistics.npy"
    #     save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
    #     np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
        
            
    # tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    # tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0)
    
    # avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    # std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    # avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    # std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    
    # statistics = {
    #     'avg_verts': avg_tot_expanded_verts,
    #     'std_verts': std_tot_expanded_verts,
    #     'avg_qtar_verts': avg_tot_expanded_qtar_verts,
    #     'std_qtar_verts': std_tot_expanded_qtar_verts
    # }
    # save_cur_inst_statistics_info_fn = "save_info_v5_statistics.npy"
    # save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    # np.save(save_cur_inst_statistics_info_fn, statistics)
    



def get_3d_pc_statistics_v6_v2(data_folder, specified_hand_type=None, specified_object_type= None, specified_folder=None, tot_sv_statistics_fn=None, nn_stages=5): 
    # exp_tag #
    ## get the version index ? ##
    ## use a separate model to depict the version index ? ##
    saved_info_fn = "save_info_v6.npy"
    
    
    # saved_info_fn = f"saved_info_accs_v5_nstages_{nn_stages}.npy" # saved #
    tot_folders = os.listdir(data_folder)
    # single clip length ## 
    
    if specified_folder is not None:
        tot_folders = [specified_folder]
    
    
    tot_expanded_verts = []
    tot_expanded_qtar_verts = []
    tot_expanded_qtar_verts_s2 = []
    
    loss_thres = 0.3
    
    # 1) centralize verts by setting the hand root to the original point #
    # 2) centralize verts by setting the hand root to the origianl point #
    
    # tot vertices in each timeste #
    for cur_fn in tot_folders: # cur fn in the tot folders # #
        
        if specified_hand_type is not None:
            if specified_hand_type == 'allegro_flat_fivefin_yscaled_finscaled':
                if specified_hand_type not in cur_fn:
                    continue
            elif specified_hand_type == 'allegro_flat_fivefin_yscaled':
                if specified_hand_type not in cur_fn or 'allegro_flat_fivefin_yscaled_finscaled' in cur_fn:
                    continue
            elif specified_hand_type == 'allegro_flat_fivefin':
                if specified_hand_type not in cur_fn or 'allegro_flat_fivefin_yscaled_finscaled' in cur_fn or 'allegro_flat_fivefin_yscaled' in cur_fn:
                    continue
            elif specified_hand_type == 'allegro':
                if 'allegro_flat_fivefin_yscaled_finscaled' in cur_fn or 'allegro_flat_fivefin_yscaled' in cur_fn or 'allegro_flat_fivefin' in cur_fn:
                    continue
            else:
                raise ValueError(f"Unrecognized specified_hand_type: {specified_hand_type}")
        
        if specified_object_type is not None:
            if f"objtype_{specified_object_type}" not in cur_fn:
                continue
        
        
        cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        
        
        if not os.path.exists(cur_fn_full): 
            continue
        
            
        print(f"cur_fn_full: {cur_fn_full}")
        saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        
        save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
        save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        
        if os.path.exists(save_cur_inst_statistics_info_fn):
            os.system(f"rm {save_cur_inst_statistics_info_fn}")
            
        if 'cost' not in saved_info:
            continue
            
        if 'tot_verts' not in saved_info or 'tot_qtar_verts' not in saved_info or 'tot_qtar_verts_s2'  not in saved_info:
            continue
        
        cost_from_saved_info = saved_info['cost']
        best_loss = cost_from_saved_info
        
        if np.isnan(best_loss):
            continue
        
        if best_loss > loss_thres:
            continue
        
        print(f"best_loss: {best_loss}")
        
        tot_verts = saved_info['tot_verts']
        tot_qtar_verts = saved_info['tot_qtar_verts']
        tot_qtar_verts_s2 = saved_info['tot_qtar_verts_s2']
        
        def calcu_avg_verts(cur_verts):
            avg_cur_verts = np.mean(cur_verts, axis=1)
            avg_cur_verts = np.mean(avg_cur_verts, axis=0)
            return avg_cur_verts
        
        avg_tot_verts = calcu_avg_verts(tot_verts)
        avg_tot_qtar_verts = calcu_avg_verts(tot_qtar_verts)
        avg_tot_qtar_verts_s2 = calcu_avg_verts(tot_qtar_verts_s2)


        # avg_tot_verts = np.mean(tot_verts, axis=1)
        # avg_tot_verts = np.mean(avg_tot_verts, axis=0)
        
        # avg_tot_qtar_verts = np.mean(tot_qtar_verts, axis=1)
        # avg_tot_qtar_verts = np.mean(avg_tot_qtar_verts, axis=0)
        
        
        # # selected_frame_verts, selected_frame_qtars_verts
        # selected_frame_verts = saved_info['selected_frame_verts']
        # selected_frame_qtars_verts = saved_info['selected_frame_qtars_verts'] # 
        
        # ## nn_obj_pts x nn_frames x 3 ##
        # ## nn_obj_pts x nn_frames x 3 ##
        # avg_selected_verts = np.mean(selected_frame_verts, axis=1)
        # avg_selected_verts = np.mean(avg_selected_verts, axis=0)
        
        # avg_selected_qtars_verts = np.mean(selected_frame_qtars_verts, axis=1)
        # avg_selected_qtars_verts = np.mean(avg_selected_qtars_verts, axis=0)
        
        if np.any(np.isnan(avg_tot_verts)) or np.any(np.isnan(avg_tot_qtar_verts)) or np.any(np.isnan(avg_tot_qtar_verts_s2)):
            continue
        
        
        cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
        tot_expanded_verts.append(cur_expanded_tot_verts)
        tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
        tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
        
        # cur_expanded_selected_frames_verts = selected_frame_verts.reshape(selected_frame_verts.shape[0] * selected_frame_verts.shape[1], -1)
        # cur_expanded_selected_frames_qtars_verts = selected_frame_qtars_verts.reshape(selected_frame_qtars_verts.shape[0] * selected_frame_qtars_verts.shape[1], -1) 
        
        avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
        std_verts = np.std(cur_expanded_tot_verts, axis=0)
        avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
        std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0)
        avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0)
        std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0)
        
        # tot_expanded_verts.append(cur_expanded_selected_frames_verts)
        # tot_expanded_qtar_verts.append(cur_expanded_selected_frames_qtars_verts) 
        
        
        save_cur_inst_statistics_info = {
            'avg_verts': avg_verts,
            'std_verts': std_verts,
            'avg_qtar_verts': avg_qtars_verts,
            'std_qtar_verts': std_qtars_verts,
            'avg_qtar_verts_s2': avg_qtars_verts_s2,
            'std_qtar_verts_s2': std_qtars_verts_s2
        }
        
        # save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
        # save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
    
    tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0) 
    tot_expanded_qtar_verts_s2 = np.concatenate(tot_expanded_qtar_verts_s2, axis=0) 
    
    avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    avg_tot_expanded_qtar_verts_s2 = np.mean(tot_expanded_qtar_verts_s2, axis=0)
    std_tot_expanded_qtar_verts_s2 = np.std(tot_expanded_qtar_verts_s2, axis=0)
    
    statistics = {
        'avg_verts': avg_tot_expanded_verts,
        'std_verts': std_tot_expanded_verts,
        'avg_qtar_verts': avg_tot_expanded_qtar_verts,
        'std_qtar_verts': std_tot_expanded_qtar_verts,
        'avg_qtar_verts_s2': avg_tot_expanded_qtar_verts_s2,
        'std_qtar_verts_s2': std_tot_expanded_qtar_verts_s2
    }
    
    if tot_sv_statistics_fn is None:
        save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    else:
        save_cur_inst_statistics_info_fn = tot_sv_statistics_fn
    save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    np.save(save_cur_inst_statistics_info_fn, statistics)
    
    
    



# data folder #
def get_3d_pc_statistics_v7_v2(data_folder, specified_hand_type=None, specified_object_type= None, specified_folder=None, tot_sv_statistics_fn=None, nn_stages=5): 
    # exp_tag #
    
    
    # /cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/task_with_pts_info.npy
    task_with_pts_info_fn = "task_with_pts_info.npy"
    cur_folder_pts_info = os.path.join(data_folder, task_with_pts_info_fn)
    
    tot_expanded_verts = []
    tot_expanded_qtar_verts = []
    tot_expanded_qtar_verts_s2 = []
    
    tot_pts_info_fn_list = []

    def calcu_avg_verts(cur_verts):
        avg_cur_verts = np.mean(cur_verts, axis=1)
        avg_cur_verts = np.mean(avg_cur_verts, axis=0)
        return avg_cur_verts
    
    if os.path.exists(cur_folder_pts_info):
        
        cur_pts_info = np.load(cur_folder_pts_info, allow_pickle=True).item()
        tot_verts = cur_pts_info['tot_verts']
        tot_qtar_verts = cur_pts_info['tot_qtar_verts']
        tot_qtar_verts_s2 = tot_qtar_verts.copy()
        
        # tot_verts = saved_info['tot_verts']
        # tot_qtar_verts = saved_info['tot_qtar_verts']
        # tot_qtar_verts_s2 = saved_info['tot_qtar_verts_s2']
        
        avg_tot_verts = calcu_avg_verts(tot_verts)
        avg_tot_qtar_verts = calcu_avg_verts(tot_qtar_verts)
        avg_tot_qtar_verts_s2 = calcu_avg_verts(tot_qtar_verts_s2)

        # if np.any(np.isnan(avg_tot_verts)) or np.any(np.isnan(avg_tot_qtar_verts)) or np.any(np.isnan(avg_tot_qtar_verts_s2)):
        #     continue
        
        cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
        tot_expanded_verts.append(cur_expanded_tot_verts)
        tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
        tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
        avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
        std_verts = np.std(cur_expanded_tot_verts, axis=0)
        avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
        std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0)
        avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0)
        std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0)
        
        save_cur_inst_statistics_info = {
            'avg_verts': avg_verts,
            'std_verts': std_verts,
            'avg_qtar_verts': avg_qtars_verts,
            'std_qtar_verts': std_qtars_verts,
            'avg_qtar_verts_s2': avg_qtars_verts_s2,
            'std_qtar_verts_s2': std_qtars_verts_s2
        }
        cur_inst_statistics_file_fn = [cur_folder_pts_info]
        save_cur_inst_statistics_info.update(
            {
                'file_list': cur_inst_statistics_file_fn
            }
        )
        save_cur_inst_statistics_info_fn = "save_info_v6_statistics_single.npy"
        save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
        np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
    
        tot_pts_info_fn_list.append(cur_folder_pts_info)
    
    tot_subfolders = os.listdir(data_folder)
    tot_subfolders = [  fn for fn in tot_subfolders if os.path.isdir(os.path.join(data_folder, fn)) ]
    for i_subfolder, cur_subfolder in enumerate(tot_subfolders):
        cur_full_subfolder = os.path.join(data_folder, cur_subfolder)
        cur_folder_pts_info_fn = os.path.join(cur_full_subfolder, task_with_pts_info_fn)
        if not os.path.exists(cur_folder_pts_info_fn):
            continue
        cur_folder_pts_info = np.load(cur_folder_pts_info_fn, allow_pickle=True).item()
        tot_verts = cur_folder_pts_info['tot_verts']
        tot_qtar_verts = cur_folder_pts_info['tot_qtar_verts']
        tot_qtar_verts_s2 = tot_qtar_verts.copy()
          
        cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
        tot_expanded_verts.append(cur_expanded_tot_verts)
        tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
        tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
        avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
        std_verts = np.std(cur_expanded_tot_verts, axis=0)
        avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
        std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0)
        avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0)
        std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0)
        
        save_cur_inst_statistics_info = {
            'avg_verts': avg_verts,
            'std_verts': std_verts,
            'avg_qtar_verts': avg_qtars_verts,
            'std_qtar_verts': std_qtars_verts,
            'avg_qtar_verts_s2': avg_qtars_verts_s2,
            'std_qtar_verts_s2': std_qtars_verts_s2
        }
        
        cur_inst_statistics_file_fn = [cur_folder_pts_info_fn]
        save_cur_inst_statistics_info.update(
            {
                'file_list': cur_inst_statistics_file_fn
            }
        )
        save_cur_inst_statistics_info_fn = "save_info_v6_statistics_single.npy"
        save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_subfolder, save_cur_inst_statistics_info_fn)
        np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)

        tot_pts_info_fn_list.append(cur_folder_pts_info_fn)
        print(f"statistics saved to {save_cur_inst_statistics_info_fn}")
    
    tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0) 
    tot_expanded_qtar_verts_s2 = np.concatenate(tot_expanded_qtar_verts_s2, axis=0) 
    
    avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    avg_tot_expanded_qtar_verts_s2 = np.mean(tot_expanded_qtar_verts_s2, axis=0)
    std_tot_expanded_qtar_verts_s2 = np.std(tot_expanded_qtar_verts_s2, axis=0)
    
    statistics = {
        'avg_verts': avg_tot_expanded_verts,
        'std_verts': std_tot_expanded_verts,
        'avg_qtar_verts': avg_tot_expanded_qtar_verts,
        'std_qtar_verts': std_tot_expanded_qtar_verts,
        'avg_qtar_verts_s2': avg_tot_expanded_qtar_verts_s2,
        'std_qtar_verts_s2': std_tot_expanded_qtar_verts_s2,
        'file_list': tot_pts_info_fn_list,
    }
    save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    np.save(save_cur_inst_statistics_info_fn, statistics)

    print(f"Statistics saved to {save_cur_inst_statistics_info_fn}")
        
    
    # ## get the version index ? ##
    # ## use a separate model to depict the version index ? ##
    # saved_info_fn = "save_info_v6.npy"
    
    
    # # saved_info_fn = f"saved_info_accs_v5_nstages_{nn_stages}.npy" # saved #
    # tot_folders = os.listdir(data_folder)
    # # single clip length ## 
    
    # if specified_folder is not None:
    #     tot_folders = [specified_folder]
    
    
    # tot_expanded_verts = []
    # tot_expanded_qtar_verts = []
    # tot_expanded_qtar_verts_s2 = []
    
    # loss_thres = 0.3
    
    # # 1) centralize verts by setting the hand root to the original point #
    # # 2) centralize verts by setting the hand root to the origianl point #
    
    # # tot vertices in each timeste #
    # for cur_fn in tot_folders: # cur fn in the tot folders # #
        
    #     if specified_hand_type is not None:
    #         if specified_hand_type == 'allegro_flat_fivefin_yscaled_finscaled':
    #             if specified_hand_type not in cur_fn:
    #                 continue
    #         elif specified_hand_type == 'allegro_flat_fivefin_yscaled':
    #             if specified_hand_type not in cur_fn or 'allegro_flat_fivefin_yscaled_finscaled' in cur_fn:
    #                 continue
    #         elif specified_hand_type == 'allegro_flat_fivefin':
    #             if specified_hand_type not in cur_fn or 'allegro_flat_fivefin_yscaled_finscaled' in cur_fn or 'allegro_flat_fivefin_yscaled' in cur_fn:
    #                 continue
    #         elif specified_hand_type == 'allegro':
    #             if 'allegro_flat_fivefin_yscaled_finscaled' in cur_fn or 'allegro_flat_fivefin_yscaled' in cur_fn or 'allegro_flat_fivefin' in cur_fn:
    #                 continue
    #         else:
    #             raise ValueError(f"Unrecognized specified_hand_type: {specified_hand_type}")
        
    #     if specified_object_type is not None:
    #         if f"objtype_{specified_object_type}" not in cur_fn:
    #             continue
        
        
    #     cur_fn_full = os.path.join(data_folder, cur_fn, saved_info_fn)
        
        
        
    #     if not os.path.exists(cur_fn_full): 
    #         continue
        
            
    #     print(f"cur_fn_full: {cur_fn_full}")
    #     saved_info = np.load(cur_fn_full, allow_pickle=True).item()
        
    #     save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    #     save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
        
    #     if os.path.exists(save_cur_inst_statistics_info_fn):
    #         os.system(f"rm {save_cur_inst_statistics_info_fn}")
            
    #     if 'cost' not in saved_info:
    #         continue
            
    #     if 'tot_verts' not in saved_info or 'tot_qtar_verts' not in saved_info or 'tot_qtar_verts_s2'  not in saved_info:
    #         continue
        
    #     cost_from_saved_info = saved_info['cost']
    #     best_loss = cost_from_saved_info
        
    #     if np.isnan(best_loss):
    #         continue
        
    #     if best_loss > loss_thres:
    #         continue
        
    #     print(f"best_loss: {best_loss}")
        
    #     tot_verts = saved_info['tot_verts']
    #     tot_qtar_verts = saved_info['tot_qtar_verts']
    #     tot_qtar_verts_s2 = saved_info['tot_qtar_verts_s2']
        
    #     def calcu_avg_verts(cur_verts):
    #         avg_cur_verts = np.mean(cur_verts, axis=1)
    #         avg_cur_verts = np.mean(avg_cur_verts, axis=0)
    #         return avg_cur_verts
        
    #     avg_tot_verts = calcu_avg_verts(tot_verts)
    #     avg_tot_qtar_verts = calcu_avg_verts(tot_qtar_verts)
    #     avg_tot_qtar_verts_s2 = calcu_avg_verts(tot_qtar_verts_s2)


    #     # avg_tot_verts = np.mean(tot_verts, axis=1)
    #     # avg_tot_verts = np.mean(avg_tot_verts, axis=0)
        
    #     # avg_tot_qtar_verts = np.mean(tot_qtar_verts, axis=1)
    #     # avg_tot_qtar_verts = np.mean(avg_tot_qtar_verts, axis=0)
        
        
    #     # # selected_frame_verts, selected_frame_qtars_verts
    #     # selected_frame_verts = saved_info['selected_frame_verts']
    #     # selected_frame_qtars_verts = saved_info['selected_frame_qtars_verts'] # 
        
    #     # ## nn_obj_pts x nn_frames x 3 ##
    #     # ## nn_obj_pts x nn_frames x 3 ##
    #     # avg_selected_verts = np.mean(selected_frame_verts, axis=1)
    #     # avg_selected_verts = np.mean(avg_selected_verts, axis=0)
        
    #     # avg_selected_qtars_verts = np.mean(selected_frame_qtars_verts, axis=1)
    #     # avg_selected_qtars_verts = np.mean(avg_selected_qtars_verts, axis=0)
        
    #     if np.any(np.isnan(avg_tot_verts)) or np.any(np.isnan(avg_tot_qtar_verts)) or np.any(np.isnan(avg_tot_qtar_verts_s2)):
    #         continue
        
        
    #     cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
    #     cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
    #     cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
    #     tot_expanded_verts.append(cur_expanded_tot_verts)
    #     tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
    #     tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
        
    #     # cur_expanded_selected_frames_verts = selected_frame_verts.reshape(selected_frame_verts.shape[0] * selected_frame_verts.shape[1], -1)
    #     # cur_expanded_selected_frames_qtars_verts = selected_frame_qtars_verts.reshape(selected_frame_qtars_verts.shape[0] * selected_frame_qtars_verts.shape[1], -1) 
        
    #     avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
    #     std_verts = np.std(cur_expanded_tot_verts, axis=0)
    #     avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
    #     std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0)
    #     avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0)
    #     std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0)
        
    #     # tot_expanded_verts.append(cur_expanded_selected_frames_verts)
    #     # tot_expanded_qtar_verts.append(cur_expanded_selected_frames_qtars_verts) 
        
        
    #     save_cur_inst_statistics_info = {
    #         'avg_verts': avg_verts,
    #         'std_verts': std_verts,
    #         'avg_qtar_verts': avg_qtars_verts,
    #         'std_qtar_verts': std_qtars_verts,
    #         'avg_qtar_verts_s2': avg_qtars_verts_s2,
    #         'std_qtar_verts_s2': std_qtars_verts_s2
    #     }
        
    #     # save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    #     # save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_fn, save_cur_inst_statistics_info_fn)
    #     np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
    
    # tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    # tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0) 
    # tot_expanded_qtar_verts_s2 = np.concatenate(tot_expanded_qtar_verts_s2, axis=0) 
    
    # avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    # std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    # avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    # std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    # avg_tot_expanded_qtar_verts_s2 = np.mean(tot_expanded_qtar_verts_s2, axis=0)
    # std_tot_expanded_qtar_verts_s2 = np.std(tot_expanded_qtar_verts_s2, axis=0)
    
    # statistics = {
    #     'avg_verts': avg_tot_expanded_verts,
    #     'std_verts': std_tot_expanded_verts,
    #     'avg_qtar_verts': avg_tot_expanded_qtar_verts,
    #     'std_qtar_verts': std_tot_expanded_qtar_verts,
    #     'avg_qtar_verts_s2': avg_tot_expanded_qtar_verts_s2,
    #     'std_qtar_verts_s2': std_tot_expanded_qtar_verts_s2
    # }
    
    # if tot_sv_statistics_fn is None:
    #     save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    # else:
    #     save_cur_inst_statistics_info_fn = tot_sv_statistics_fn
    # save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    # np.save(save_cur_inst_statistics_info_fn, statistics)



def get_pts_fr_qs_qtars(qs_qtars_fn):
    qs_qtars = np.load(qs_qtars_fn, allow_pickle=True).item()
    
    ts_to_hand_qs = qs_qtars['ts_to_hand_qs'] 
    ts_to_hand_qtars = qs_qtars['ts_to_qtars'] 
    
    ts_to_optimized_q_tars_wcontrolfreq = qs_qtars['ts_to_optimized_q_tars_wcontrolfreq']
    # ts_to_
    ts_to_hand_qtars = ts_to_optimized_q_tars_wcontrolfreq
    
    # ts to 
    ctl_freq_tss = list(ts_to_optimized_q_tars_wcontrolfreq.keys())
    ctl_freq_tss = sorted(ctl_freq_tss)
    ctl_freq = 10
    ctl_freq_tss_expanded = [ min(500 - 1, i_ts * ctl_freq) for i_ts in ctl_freq_tss ]
    ts_to_hand_qs = {
        ctl_freq_tss[i_ts]: ts_to_hand_qs[ ctl_freq_tss_expanded[i_ts]  ] for i_ts in range(len(ctl_freq_tss))
    }
    
    
    computed_info = compute_pts_v2(ts_to_hand_qs, ts_to_hand_qtars)

## get the 3d paired data? with the un-optimized data ##
def get_3d_pc_statistics_v7_cond_v2(data_folder, specified_hand_type=None, specified_object_type= None, specified_folder=None, tot_sv_statistics_fn=None, nn_stages=5): 
    # exp_tag #
    
    # /cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/task_with_pts_info.npy
    task_with_pts_info_fn = "task_with_pts_info.npy" # task with pts info #
    cur_folder_pts_info = os.path.join(data_folder, task_with_pts_info_fn)
    
    tot_expanded_verts = []
    tot_expanded_qtar_verts = []
    tot_expanded_qtar_verts_s2 = []
    
    tot_pts_info_fn_list = []

    def calcu_avg_verts(cur_verts):
        avg_cur_verts = np.mean(cur_verts, axis=1)
        avg_cur_verts = np.mean(avg_cur_verts, axis=0)
        return avg_cur_verts
    
    # get the first optimized pts info ##
    
    ### HACK ###
    # task_optimized_
    
    ## learn the optimization path? ##
    
    tot_subfolders = os.listdir(data_folder)
    tot_subfolders = [fn for fn in tot_subfolders if os.path.isdir(os.path.join(data_folder, fn))]
    
    # /cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/frtask_objx0.07_objrot_0.30000000000000004_objtype_cylinder_/best_res_tau_730_rnk_0.npy
    for cur_folder in tot_subfolders:
        cur_full_folder = os.path.join(data_folder, cur_folder)
        cur_setting_optimized_info = os.listdir(cur_full_folder)
        cur_setting_optimized_info = [fn for fn in cur_setting_optimized_info if "best_res_tau" in fn]
        cur_setting_optimized_info_fn_to_steps = {}
        for cur_optimized_info_fn in cur_setting_optimized_info:
            cur_info_optimized_steps = cur_optimized_info_fn.split("_")[3]
            cur_info_optimized_steps = int(cur_info_optimized_steps)
            cur_setting_optimized_info_fn_to_steps[cur_optimized_info_fn] = cur_info_optimized_steps
            # cur setting # 
        sorted_optimized_info_w_steps = sorted(
            cur_setting_optimized_info_fn_to_steps.items(), key=lambda ii: ii[1], 
        )
        first_optimized_info_fn = sorted_optimized_info_w_steps[0][0]
        last_optimized_info_fn = sorted_optimized_info_w_steps[-1][0] 
        first_optimized_info_full_fn = os.path.join(cur_full_folder, first_optimized_info_fn)
        last_optimized_info_full_fn = os.path.join(cur_full_folder, last_optimized_info_fn)
        

        
        first_optimized_info = np.load(first_optimized_info_full_fn, allow_pickle=True).item()
        last_optimized_info = np.load(last_optimized_info_full_fn, allow_pickle=True).item()
        
        first_tot_verts = first_optimized_info['tot_verts']
        first_tot_qtars_verts = first_optimized_info['tot_qtar_verts']
        
        last_tot_verts = last_optimized_info['tot_verts']
        last_tot_qtars_verts = last_optimized_info['tot_qtar_verts']
        
        saved_info = { # optimized info fn # # use the same statistics #
            'unopt_verts': first_tot_verts,
            'unopt_qtar_verts': first_tot_qtars_verts,
            'opt_verts': last_tot_verts,
            'opt_qtar_verts': last_tot_qtars_verts, 
        }
        saved_info_fn = f"optimization_info.npy"
        saved_info_fn = os.path.join(cur_full_folder, saved_info_fn)
        saved_info_fn = np.save(saved_info_fn, saved_info)
        print(f"optimization info for folder {cur_full_folder} saved to {saved_info_fn}")
            
    
    # if os.path.exists(cur_folder_pts_info):
        
    #     cur_pts_info = np.load(cur_folder_pts_info, allow_pickle=True).item()
    #     tot_verts = cur_pts_info['tot_verts']
    #     tot_qtar_verts = cur_pts_info['tot_qtar_verts']
    #     tot_qtar_verts_s2 = tot_qtar_verts.copy()
        
    #     # tot_verts = saved_info['tot_verts'] # tot verts #
    #     # tot_qtar_verts = saved_info['tot_qtar_verts']
    #     # tot_qtar_verts_s2 = saved_info['tot_qtar_verts_s2']
        
    #     # avg_tot_verts = calcu_avg_verts(tot_verts) # avg tot verts #
    #     # avg_tot_qtar_verts = calcu_avg_verts(tot_qtar_verts) # avg tot verts #
    #     # avg_tot_qtar_verts_s2 = calcu_avg_verts(tot_qtar_verts_s2)
        
    #     #### avg tot qtar verts s2 ####

    #     # if np.any(np.isnan(avg_tot_verts)) or np.any(np.isnan(avg_tot_qtar_verts)) or np.any(np.isnan(avg_tot_qtar_verts_s2)):
    #     #     continue ##########
        
    #     cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
    #     cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
    #     cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
    #     tot_expanded_verts.append(cur_expanded_tot_verts)
    #     tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
    #     tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
    #     avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
    #     std_verts = np.std(cur_expanded_tot_verts, axis=0)
    #     avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
    #     std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0)
    #     avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0)
    #     std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0)
        
    #     save_cur_inst_statistics_info = {
    #         'avg_verts': avg_verts,
    #         'std_verts': std_verts,
    #         'avg_qtar_verts': avg_qtars_verts,
    #         'std_qtar_verts': std_qtars_verts,
    #         'avg_qtar_verts_s2': avg_qtars_verts_s2,
    #         'std_qtar_verts_s2': std_qtars_verts_s2
    #     }
    #     cur_inst_statistics_file_fn = [cur_folder_pts_info]
    #     save_cur_inst_statistics_info.update(
    #         {
    #             'file_list': cur_inst_statistics_file_fn
    #         }
    #     )
    #     save_cur_inst_statistics_info_fn = "save_info_v6_statistics_single.npy"
    #     save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    #     np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)
    
    #     tot_pts_info_fn_list.append(cur_folder_pts_info)
    
    # tot_subfolders = os.listdir(data_folder)
    # tot_subfolders = [  fn for fn in tot_subfolders if os.path.isdir(os.path.join(data_folder, fn)) ]
    # for i_subfolder, cur_subfolder in enumerate(tot_subfolders):
    #     cur_full_subfolder = os.path.join(data_folder, cur_subfolder)
    #     cur_folder_pts_info_fn = os.path.join(cur_full_subfolder, task_with_pts_info_fn)
    #     if not os.path.exists(cur_folder_pts_info_fn):
    #         continue
    #     cur_folder_pts_info = np.load(cur_folder_pts_info_fn, allow_pickle=True).item()
    #     tot_verts = cur_folder_pts_info['tot_verts']
    #     tot_qtar_verts = cur_folder_pts_info['tot_qtar_verts']
    #     tot_qtar_verts_s2 = tot_qtar_verts.copy()
          
    #     cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
    #     cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
    #     cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
    #     tot_expanded_verts.append(cur_expanded_tot_verts)
    #     tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
    #     tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
    #     avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
    #     std_verts = np.std(cur_expanded_tot_verts, axis=0)
    #     avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
    #     std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0)
    #     avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0)
    #     std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0)
        
    #     save_cur_inst_statistics_info = {
    #         'avg_verts': avg_verts,
    #         'std_verts': std_verts,
    #         'avg_qtar_verts': avg_qtars_verts,
    #         'std_qtar_verts': std_qtars_verts,
    #         'avg_qtar_verts_s2': avg_qtars_verts_s2,
    #         'std_qtar_verts_s2': std_qtars_verts_s2
    #     }
        
    #     cur_inst_statistics_file_fn = [cur_folder_pts_info_fn]
    #     save_cur_inst_statistics_info.update(
    #         {
    #             'file_list': cur_inst_statistics_file_fn
    #         }
    #     )
    #     save_cur_inst_statistics_info_fn = "save_info_v6_statistics_single.npy"
    #     save_cur_inst_statistics_info_fn = os.path.join(data_folder, cur_subfolder, save_cur_inst_statistics_info_fn)
    #     np.save(save_cur_inst_statistics_info_fn, save_cur_inst_statistics_info)

    #     tot_pts_info_fn_list.append(cur_folder_pts_info_fn)
    #     print(f"statistics saved to {save_cur_inst_statistics_info_fn}")
    
    # tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    # tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0) 
    # tot_expanded_qtar_verts_s2 = np.concatenate(tot_expanded_qtar_verts_s2, axis=0) 
    
    # avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    # std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    # avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    # std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    # avg_tot_expanded_qtar_verts_s2 = np.mean(tot_expanded_qtar_verts_s2, axis=0)
    # std_tot_expanded_qtar_verts_s2 = np.std(tot_expanded_qtar_verts_s2, axis=0)
    
    # statistics = {
    #     'avg_verts': avg_tot_expanded_verts,
    #     'std_verts': std_tot_expanded_verts,
    #     'avg_qtar_verts': avg_tot_expanded_qtar_verts,
    #     'std_qtar_verts': std_tot_expanded_qtar_verts,
    #     'avg_qtar_verts_s2': avg_tot_expanded_qtar_verts_s2,
    #     'std_qtar_verts_s2': std_tot_expanded_qtar_verts_s2,
    #     'file_list': tot_pts_info_fn_list,
    # }
    # save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    # save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    # np.save(save_cur_inst_statistics_info_fn, statistics)

    # print(f"Statistics saved to {save_cur_inst_statistics_info_fn}")



# get the statistics #
# # #    # # #
def get_3d_pc_statistics_v7_v3(data_folder, specified_hand_type=None, specified_object_type= None, specified_folder=None, tot_sv_statistics_fn=None, nn_stages=5): 
    # exp_tag #
    
    tot_pts_info_fns = os.listdir(data_folder)
    tot_pts_info_fns = [fn for fn in tot_pts_info_fns if "_pts.npy" in fn and "_to_child_pts.npy" not in fn and "to_task_pts.npy" not in fn]
    print(f"tot_pts_info_fns: {tot_pts_info_fns}")
    
    
    tot_expanded_verts = []
    tot_expanded_qtar_verts = []
    tot_expanded_qtar_verts_s2 = []
    
    tot_pts_info_fn_list = []
    
    for i_pts_info, cur_pts_info_fn in enumerate(tot_pts_info_fns):
        cur_full_pts_info_fn = os.path.join(data_folder, cur_pts_info_fn)
        cur_pts_info = np.load(cur_full_pts_info_fn, allow_pickle=True).item()
        tot_verts = cur_pts_info['tot_verts']
        tot_qtar_verts = cur_pts_info['tot_qtar_verts']
        tot_qtar_verts_s2 = tot_qtar_verts.copy()
          
        cur_expanded_tot_verts = tot_verts.reshape(tot_verts.shape[0] * tot_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts = tot_qtar_verts.reshape(tot_qtar_verts.shape[0] * tot_qtar_verts.shape[1], -1)
        cur_expanded_tot_qtar_verts_s2 = tot_qtar_verts_s2.reshape(tot_qtar_verts_s2.shape[0] * tot_qtar_verts_s2.shape[1], -1)
        
        tot_expanded_verts.append(cur_expanded_tot_verts)
        tot_expanded_qtar_verts.append(cur_expanded_tot_qtar_verts)
        tot_expanded_qtar_verts_s2.append(cur_expanded_tot_qtar_verts_s2)
        
        tot_pts_info_fn_list.append(cur_full_pts_info_fn)
        # avg_verts = np.mean(cur_expanded_tot_verts, axis=0)
        # std_verts = np.std(cur_expanded_tot_verts, axis=0)
        # avg_qtars_verts = np.mean(cur_expanded_tot_qtar_verts, axis=0)
        # std_qtars_verts = np.std(cur_expanded_tot_qtar_verts, axis=0) # 
        # avg_qtars_verts_s2 = np.mean(cur_expanded_tot_qtar_verts_s2, axis=0) # std verts ## std verts ##
        # std_qtars_verts_s2 = np.std(cur_expanded_tot_qtar_verts_s2, axis=0) # std verts ## std verts ##
    # inherit fr pts info #
    inherit_fr_pts_info_fn = os.path.join(data_folder, f"best_fa_to_task_pts.npy")
    to_task_pts_info_fn = os.path.join(data_folder, f"best_task_pts.npy")
    inheriting_info = {
        'inherit_fr_pts_info_fn': inherit_fr_pts_info_fn,
        'to_task_pts_info_fn': to_task_pts_info_fn
    }
    
    tot_expanded_verts = np.concatenate(tot_expanded_verts, axis=0)
    tot_expanded_qtar_verts = np.concatenate(tot_expanded_qtar_verts, axis=0) 
    tot_expanded_qtar_verts_s2 = np.concatenate(tot_expanded_qtar_verts_s2, axis=0) 
    
    # calculate the verts #
    # calculate the verts #
    avg_tot_expanded_verts = np.mean(tot_expanded_verts, axis=0)
    std_tot_expanded_verts = np.std(tot_expanded_verts, axis=0)
    avg_tot_expanded_qtar_verts = np.mean(tot_expanded_qtar_verts, axis=0)
    std_tot_expanded_qtar_verts = np.std(tot_expanded_qtar_verts, axis=0)
    avg_tot_expanded_qtar_verts_s2 = np.mean(tot_expanded_qtar_verts_s2, axis=0)
    std_tot_expanded_qtar_verts_s2 = np.std(tot_expanded_qtar_verts_s2, axis=0)
    
    statistics = {
        'avg_verts': avg_tot_expanded_verts,
        'std_verts': std_tot_expanded_verts,
        'avg_qtar_verts': avg_tot_expanded_qtar_verts,
        'std_qtar_verts': std_tot_expanded_qtar_verts,
        'avg_qtar_verts_s2': avg_tot_expanded_qtar_verts_s2,
        'std_qtar_verts_s2': std_tot_expanded_qtar_verts_s2,
        'file_list': tot_pts_info_fn_list,
        'inheriting_info': inheriting_info,
    }
    save_cur_inst_statistics_info_fn = "save_info_v6_statistics.npy"
    save_cur_inst_statistics_info_fn = os.path.join(data_folder, save_cur_inst_statistics_info_fn)
    np.save(save_cur_inst_statistics_info_fn, statistics)
    
    
def inspect_statistics_file(statistics_fn):
    statistics = np.load(statistics_fn, allow_pickle=True).item()
    inherting_info = statistics['inheriting_info']
    print(inherting_info)
    inherit_fr_pts_info_fn=  inherting_info['inherit_fr_pts_info_fn']
    to_task_pts_info_fn = inherting_info['to_task_pts_info_fn']
    inherit_fr_pts_info = np.load(inherit_fr_pts_info_fn, allow_pickle=True).item()
    to_task_pts_info = np.load(to_task_pts_info_fn, allow_pickle=True).item()
    
    print(inherit_fr_pts_info.keys())

    task_setting_keys = [
        'object_type', 'task_rot', 'object_size_x'
    ]
    for task_key in task_setting_keys:
        task_val = inherit_fr_pts_info[task_key]
        print(f"[father task] task_key: {task_key}, task_val: {task_val}")
    
    for task_key in task_setting_keys:
        task_val = to_task_pts_info[task_key]
        print(f"[child task] task_key: {task_key}, task_val: {task_val}")
    
    obj_transformation_keys = [
        'ball_trans_err', 'ball_ornt_err'
    ]
    for obj_key in obj_transformation_keys:
        obj_val = inherit_fr_pts_info[obj_key]
        print(f"[father task] obj_key: {obj_key}, obj_val: {obj_val}")
    
    for obj_key in obj_transformation_keys:
        obj_val = to_task_pts_info[obj_key]
        print(f"[child task] obj_key: {obj_key}, obj_val: {obj_val}")


def calculate_single_traj_hand_obj_diff(pred_qs, gt_qs):
    pred_hand_qs = pred_qs[..., :-7]
    pred_obj_pos = pred_qs[..., -7:-4]
    pred_obj_ornt = pred_qs[..., -4:]
    gt_hand_qs = gt_qs[..., :-7]
    gt_obj_pos = gt_qs[..., -7:-4]
    gt_obj_ornt = gt_qs[..., -4:]
    
    diff_pred_gt_hand_qs = np.sum((pred_hand_qs - gt_hand_qs) ** 2, axis=-1)
    diff_pred_gt_hand_qs = np.mean(diff_pred_gt_hand_qs).item()
    
    diff_pred_gt_obj_pos = np.sum((pred_obj_pos - gt_obj_pos) ** 2, axis=-1)
    diff_pred_gt_obj_pos = np.mean(diff_pred_gt_obj_pos).item()
    
    diff_pred_gt_obj_ornt = np.sum((pred_obj_ornt - gt_obj_ornt) ** 2, axis=-1)
    diff_pred_gt_obj_ornt = np.mean(diff_pred_gt_obj_ornt).item()
    
    return diff_pred_gt_hand_qs, diff_pred_gt_obj_pos, diff_pred_gt_obj_ornt

def calculate_single_traj_hand_obj_diff_from_file(samples_fn):
    samples = np.load(samples_fn, allow_pickle=True).item()
    samples_key = 'samples'
    gt_key = None
    for key in samples.keys():
        if key != samples_key:
            gt_key = key
            break
    gt_data = samples[gt_key]
    samples_data = samples[samples_key]
    print(f"gt_data: {gt_data.keys()}, samples_data: {samples_data.keys()}")
    
    gt_hand_pose = gt_data['future_hand_pose']
    gt_obj_pos = gt_data['future_obj_pos']
    gt_obj_ornt = gt_data['future_obj_ornt']
    
    samples_hand_pose = samples_data['hand_pose']
    samples_obj_pos = samples_data['obj_pos']
    samples_obj_ornt = samples_data['obj_ornt']
    
    gt_hand_qs = np.concatenate([gt_hand_pose, gt_obj_pos, gt_obj_ornt], axis=-1)
    samples_hand_qs = np.concatenate([samples_hand_pose, samples_obj_pos, samples_obj_ornt], axis=-1)
    diff_hand_qs, diff_obj_pos, diff_obj_ornt = calculate_single_traj_hand_obj_diff(samples_hand_qs, gt_hand_qs)
    print(f"diff_hand_qs: {diff_hand_qs}, diff_obj_pos: {diff_obj_pos}, diff_obj_ornt: {diff_obj_ornt}")
    return diff_hand_qs, diff_obj_pos, diff_obj_ornt
    
def calculate_single_traj_AE_sample_diff(ae_sample_fn):
    samples = np.load(ae_sample_fn, allow_pickle=True).item()
    print(f"samples: {samples.keys()}")
    samples_key = 'samples'
    gt_key = None
    for key in samples.keys():
        if key != samples_key:
            gt_key = key
            break
    gt_data = samples[gt_key]
    samples_data = samples[samples_key]
    print(f"gt_data: {gt_data.keys()}, samples_data: {samples_data.keys()}")
    gt_hand_pose = gt_data['future_hand_pose']
    gt_obj_pos = gt_data['future_obj_pos']
    gt_obj_ornt = gt_data['future_obj_ornt']
    
    # samples_hand_pose = samples_data['hand_pose']
    # samples_obj_pos = samples_data['obj_pos']
    # samples_obj_ornt = samples_data['obj_ornt']
    
    samples_hand_qs = samples_data['E']
    
    gt_hand_qs = np.concatenate([gt_hand_pose, gt_obj_pos, gt_obj_ornt], axis=-1)
    # samples_hand_qs = np.concatenate([samples_hand_pose, samples_obj_pos, samples_obj_ornt], axis=-1)
    # samples_hand_qs = 
    diff_hand_qs, diff_obj_pos, diff_obj_ornt = calculate_single_traj_hand_obj_diff(samples_hand_qs, gt_hand_qs)
    print(f"diff_hand_qs: {diff_hand_qs}, diff_obj_pos: {diff_obj_pos}, diff_obj_ornt: {diff_obj_ornt}")
    return diff_hand_qs, diff_obj_pos, diff_obj_ornt

def calculate_single_traj_Diff_sample(diff_sample_fn):
    # trai nthe diff samples # diff samples -- no conditions -- just the diff field #
    return calculate_single_traj_AE_sample_diff(diff_sample_fn)
    # samples = np.load(diff_sample_fn, allow_pickle=True).item()
    # print(f"samples: {samples.keys()}")
    # samples_key = 'samples'
    # gt_key = None
    # for key in samples.keys():
    #     if key != samples_key:
    #         gt_key = key
    #         break
    # gt_data = samples[gt_key]
    # samples_data = samples[samples_key]
    # print(f"gt_data: {gt_data.keys()}, samples_data: {samples_data.keys()}")


def test_samples_data(samples_fn):
    samples = np.load(samples_fn, allow_pickle=True).item()
    print(  samples.keys()  )
    training_samples = samples['closest_training_data']
    samples_samples = samples['samples']
    for key in samples_samples:
        cur_val = samples_samples[key]
        try:
            print(f"key: {key}, val: {cur_val.shape}")
        except:
            pass
    for key in training_samples:
        cur_val = training_samples[key]
        try:
            print(f"key: {key}, val: {cur_val.shape}")
        except:
            pass


# test the differences between samples and the gt ones #


if __name__ == "__main__":
    
    samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_ws60_centralize_handobjinfo_/samples000890001.npy"
    
    test_samples_data(samples_fn)
    exit(0)
    

    # diff_sample_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_Diff_v2/samples000380001.npy"
    # calculate_single_traj_Diff_sample(diff_sample_fn)
    # exit(0)
    
    
    # ae_sample_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE/samples000740000.npy"
    # ae_sample_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_AEDiff_AE_v2/samples000210000.npy"
    # calculate_single_traj_AE_sample_diff(ae_sample_fn)
    # exit(0)
    
    samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_"
    samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_"
    samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_"
    samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws100_"
    samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws120_"
    samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_histindex_"
    # samples_folder = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_skippedglb_ws60_"
    tot_samples_fn = os.listdir(samples_folder)
    tot_samples_fn = [fn for fn in tot_samples_fn if "samples" in fn and fn.endswith(".npy")]
    tot_samples_fn = sorted(tot_samples_fn)
    # sample_ep_to_hand_qs_diff
    hand_qs_diff_list = []
    obj_pos_diff_list = []
    obj_ornt_diff_list=[]
    for sample_fn in tot_samples_fn:
        full_sample_fn = os.path.join(samples_folder, sample_fn)
        diff_hand_qs, diff_obj_pos, diff_obj_ornt = calculate_single_traj_hand_obj_diff_from_file(full_sample_fn)
        samples_key = "samples"
        sample_ep = sample_fn.split(".npy")[0][len(samples_key):]
        sample_ep = int(sample_ep)
        hand_qs_diff_list.append(diff_hand_qs)
        obj_pos_diff_list.append(diff_obj_pos)
        obj_ornt_diff_list.append(diff_obj_ornt)
        
    hand_qs_diff_list = np.array(hand_qs_diff_list)
    obj_pos_diff_list = np.array(obj_pos_diff_list)
    obj_ornt_diff_list = np.array(obj_ornt_diff_list)
    print(f"hand_qs_diff_list: {hand_qs_diff_list}")
    print(f"obj_pos_diff_list: {obj_pos_diff_list}")
    print(f"obj_ornt_diff_list: {obj_ornt_diff_list}")
    
    # samples_fn = "/cephfs/xueyi/uni_manip/tds_diffusion_exp/allegro_tracking_kine_diff_AE_Diff_traj_forcasting_fixed_/samples000930000.npy"
    # calculate_single_traj_hand_obj_diff_from_file(samples_fn)
    exit(0)
    
    # statistics_fn = "/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported/save_info_v6_statistics.npy"
    # inspect_statistics_file(statistics_fn)
    # exit(0)
    
    data_folder = "/cephfs/yilaa/uni_manip/tds_rl_exp/./logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_box_wsmallsigma_svres__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_box_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported"
    
    data_folder = "/cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_targets_cylinder_wsmallsigma_svres_v2__rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_/exported"
    get_3d_pc_statistics_v7_v3(data_folder)
    exit(0)
    
    # get 3d cp statistics #
    data_folder = "/cephfs/yilaa/uni_manip/tds_rl_exp/logs_PPO/allegro_bottle_ctlfreq_10_bt_nenv_16_allsmallsigmas_statewdiff_wobjtype_rewv2_obstype_state_wdiff_actlim_0.01_taskstage5_objtype_cylinder_objm0.39_objsxyz_0.08_0.08_0.382_objinitxyz_0.22_0.2_0.2_objgoalrot_0.4_0_0_objgoalrot2_0.4_0_0_" 
    
    # get_3d_pc_statistics_v7_v2(data_folder=data_folder)
    
    get_3d_pc_statistics_v7_cond_v2(data_folder=data_folder)
    exit(0)
    
    # cp /cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics.npy /cephfs/yilaa/uni_manip/tds_exp_2/save_info_v6_statistics_allegro.npy 
    data_folder = "/cephfs/yilaa/uni_manip/tds_exp_2"
    
    # specified_folder = "allegro_bottle_5_pds_allegro_flat_fivefin_yscaled_finscaled__ctlfreq_10_taskstage5_objtype_box_objm0.2_objsxyz_0.02_0.02_0.382_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.5_0.37_objgoalrot_0.1_0_0_objgoalrot2_0.1_0_0_tar_"
    
    # specified_folder = "allegro_bottle_5_pds_wgravity_v3_allegro_flat_fivefin_yscaled__ctlfreq_10_taskstage5_objtype_box_objm0.2_objsxyz_0.02_0.02_0.382_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.5_0.37_objgoalrot_0.1_0_0_objgoalrot2_0.1_0_0_tar_"
    
    # specified_folder = "allegro_bottle_5_pds_wgravity_v3_allegro_flat_fivefin__ctlfreq_10_taskstage5_objtype_box_objm0.2_objsxyz_0.02_0.02_0.382_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.5_0.37_objgoalrot_0.1_0_0_objgoalrot2_0.1_0_0_tar_"
    
    specified_folder = None
    
    specified_hand_type = 'allegro'
    specified_object_type = 'lotsballs'
    tot_sv_statistics_fn = 'allegro_lotsballs_save_info_v6_statistics.npy'
    
    get_3d_pc_statistics_v6_v2(data_folder, specified_hand_type=specified_hand_type, specified_object_type=specified_object_type, specified_folder=specified_folder, tot_sv_statistics_fn=tot_sv_statistics_fn, nn_stages=5)
    exit(0)
    
    # 
    # python utils/test_data.py
    data_folder = "/data/xueyi/uni_manip/tds_exp_2"
    get_3d_pc_statistics_v6(data_folder, nn_stages=5)
    exit(0)
    
    # # 
    # /data/xueyi/uni_manip/tds_exp_2/allegro_bouncing_bottle_5_test_cppad_massx10_taskc_m0d29_
    # /data/xueyi/uni_manip/tds_exp_2/allegro_bottle_5_taskstage2_objm0.39_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.2_0.37_objgoalrot_2.827433
    # 
    data_folder = "/data/xueyi/uni_manip/tds_exp_2"
    
    get_3d_pc_statistics_v4(data_folder, nn_stages=5)
    # get_3d_pc_statistics_v2(data_folder)
    exit(0)
    
    get_3d_pc_statistics_v3(data_folder)
    exit(0)
    
    
    # sampled_data_dict_fn = "/data/xueyi/uni_manip/exp/eval_v2_/eval_v2_/sampled_pcd_wact_dict.npy"
    # convert_data_dict(sampled_data_dict_fn) # simulator and the sims # 
    # exit(0)
    
    # data_folder = "/data2/xueyi/uni_manip/exp_v2/exp"
    # test_data_statistics(data_folder)
    # python utils/test_data.py
    
    # data_folder = "/data/xueyi/softzoo/expv4"
    # test_saved_ckpt(data_folder)
    # exit(0)
        
    data_folder = "/data/xueyi/softzoo/expv4"
    data_task_err_thres = 0.03
    data_trans_constraints_thres = 0.01
    # get_valid_data(data_folder, data_task_err_thres, data_trans_constraints_thres)
    
    root_data_folder =  "/data/xueyi/softzoo"
    get_valid_data_v2(root_data_folder, data_task_err_thres, data_trans_constraints_thres)
    
    

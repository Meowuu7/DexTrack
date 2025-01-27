import numpy as np
import taichi as ti
import os 
import wandb
# import cma



from softzoo.engine.uni_manip.sim_engine.uni_manip_2d_projected import UniManip2D


dim = 2
quality = 1
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

RIGID_SPHERE = 0
RIGID_CUBE = 1
RIGID_SDF = 2



def get_manipulator_infos(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=0.1, base_x_len=0.1):
    # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8]
    # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    
    # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
    # tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
    
    
    tot_nn_links_one_side = []
    tot_len_one_side = []
    link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages // 2 - 1)
    st_nn_link_one_side = 1
    for i in range(nn_stages):
        cur_link_len = st_len_one_side + (i // 2) * link_len_one_side_interval
        
        tot_len_one_side.append(cur_link_len)
        tot_nn_links_one_side.append(st_nn_link_one_side)
        
        if i % 2 == 0:
            st_nn_link_one_side = st_nn_link_one_side * 2
    
    print("tot_nn_links_one_side: ", tot_nn_links_one_side)
    print(f"tot_len_one_side: {tot_len_one_side}")
    return tot_nn_links_one_side, tot_len_one_side




def get_graph_connectivity_info(nn_link_one_side):
    nn_tot_links = nn_link_one_side * 2 + 1
    connectivity_arr = np.zeros((nn_tot_links, nn_tot_links), dtype=np.float32)
    
    connectivity_arr[0, 0] = 1.0
    
    for i_link in range(1, nn_link_one_side + 1):
        cur_link_parent = i_link - 1
        connectivity_arr[cur_link_parent, i_link] = 1.0

    for i_link in range(nn_link_one_side + 1, nn_tot_links):

        if i_link == nn_link_one_side + 1:
            cur_link_parent = 0
        else:
            cur_link_parent = i_link - 1
        connectivity_arr[cur_link_parent, i_link] = 1.0
        
    return connectivity_arr

def get_constraint_evolution_path(n_links, link_idx_to_original_parent): # link to origianl parnet ## # cross link-num; intra link-number ## intra link-number ##
    ## original E ## constriant ## 
    ## similarity between such connectivities ##
    # original_E: n_links x n_links
    ## \alpha scheduled path ##
    ## link_idx_to_original_parent: a dict with link_idx to original_parent mapping ##
    alpha_path = [1.0 - 0.1 * i for i in range(11)] ## alpha --> on other links; (1 - alpha) --> on the current link ##
    nn_evolve_levels = 10
    tot_scaled_E = []
    for i_alpha, cur_alpha in enumerate(alpha_path):
        # current_start_E = np.copy(original_E)
        # current_start_E = np.zeros_like(original_E, dtype=np.float32) ## get the current original E
        current_start_E = np.zeros((n_links, n_links), dtype=np.float32) ## initialize current start_E
        ## === Initialize current_start_E === ##
        for i_link in range(current_start_E.shape[0]):
            
            cur_link_parent = link_idx_to_original_parent[i_link] ## get the parent
            if cur_link_parent == i_link:
                current_start_E[i_link, i_link] = 1.0
            else:
                current_start_E[i_link, i_link] = 1.0 - cur_alpha # 
                current_start_E[i_link, cur_link_parent] = cur_alpha 
                ## get the current start E ##
        
        
            ### after getting current_start_E --> should evolve starting from current_start_E ###
            # for i_level in range(nn_evolve_levels):
                # current evolve level #
        for i_level in range(nn_evolve_levels):
            # # (cur_alpha / nn_evolve_levels) * i_level -> that should seperate to others #
            # # (cur_alpha / )
            for i_link in range(current_start_E.shape[0]):
                cur_evolve_E = np.copy(current_start_E) ## from the current evolve E 
                if i_link == 0 or cur_link_parent == 0: ## the current link is the root link 
                    continue
                ### then you should design the evolution path that starts from the cur_evolve_E ###
                cur_ranking = [cur_link_parent - i_i for i_i in range(cur_link_parent + 1)] # the ranking for the parent indexes #
                ### get 
                ori_parent_final_level_alpha = cur_alpha / float(cur_link_parent + 1)
                other_parent_final_level_alpha = cur_alpha - ori_parent_final_level_alpha ## final parent alpha sum
                other_parent_cur_level_alpha = (float(i_level) / float(nn_evolve_levels - 1)) * other_parent_final_level_alpha  ### the othe 
                
                # cur_other_parent_weight_sum = (cur_alpha / nn_evolve_levels) * i_level ## other parent weight sum ##
                cur_other_parent_weight = other_parent_cur_level_alpha / float(cur_link_parent ) ## from the current link parent for getting other parent weight sum ##
                # cur_evolve_E[cur_link_parent, i_link] = cur_other_parent_weight ## set the other 
                cur_evolve_E[: cur_link_parent, i_link] = cur_other_parent_weight
                cur_evolve_E[cur_link_parent, i_link] = cur_alpha - other_parent_cur_level_alpha ## get other parent cur level alpha ## 
            tot_scaled_E.append(cur_evolve_E)
    return tot_scaled_E


def get_constraint_evolution_path_curri_v2(n_links, link_idx_to_original_parent):
    ## similarity between such connectivities ##
    # original_E: n_links x n_links
    ## \alpha scheduled path ##
    ## link_idx_to_original_parent: a dict with link_idx to original_parent mapping ##
    alpha_path = [1.0 - 0.1 * i for i in range(11)] ## alpha --> on other links; (1 - alpha) --> on the current link ##
    nn_evolve_levels = 10
    tot_scaled_E = []
    for i_alpha, cur_alpha in enumerate(alpha_path):
        # current_start_E = np.copy(original_E)
        # current_start_E = np.zeros_like(original_E, dtype=np.float32) ## get the current original E
        current_start_E = np.zeros((n_links, n_links), dtype=np.float32) ## initialize current start_E
        ## === Initialize current_start_E === ##
        for i_link in range(current_start_E.shape[0]):
            
            cur_link_parent = link_idx_to_original_parent[i_link] ## get the parent
            if cur_link_parent == i_link:
                current_start_E[i_link, i_link] = 1.0
            else:
                current_start_E[i_link, i_link] = 1.0 - cur_alpha # 
                current_start_E[cur_link_parent, i_link] = cur_alpha 
                ## get the current start E ##
        
        tot_scaled_E.append(current_start_E)
        #     ### after getting current_start_E --> should evolve starting from current_start_E ###
        #     # for i_level in range(nn_evolve_levels):
        #         # current evolve level #
        # for i_level in range(nn_evolve_levels):
        #     # # (cur_alpha / nn_evolve_levels) * i_level -> that should seperate to others #
        #     # # (cur_alpha / )
        #     for i_link in range(current_start_E.shape[0]):
        #         cur_evolve_E = np.copy(current_start_E) ## from the current evolve E 
        #         if i_link == 0 or cur_link_parent == 0: ## the current link is the root link 
        #             continue
        #         ### then you should design the evolution path that starts from the cur_evolve_E ###
        #         cur_ranking = [cur_link_parent - i_i for i_i in range(cur_link_parent + 1)] # the ranking for the parent indexes #
        #         ### get 
        #         ori_parent_final_level_alpha = cur_alpha / float(cur_link_parent + 1)
        #         other_parent_final_level_alpha = cur_alpha - ori_parent_final_level_alpha ## final parent alpha sum
        #         other_parent_cur_level_alpha = (float(i_level) / float(nn_evolve_levels - 1)) * other_parent_final_level_alpha  ### the othe 
                
        #         # cur_other_parent_weight_sum = (cur_alpha / nn_evolve_levels) * i_level ## other parent weight sum ##
        #         cur_other_parent_weight = other_parent_cur_level_alpha / float(cur_link_parent ) ## from the current link parent for getting other parent weight sum ##
        #         # cur_evolve_E[cur_link_parent, i_link] = cur_other_parent_weight ## set the other 
        #         cur_evolve_E[: cur_link_parent, i_link] = cur_other_parent_weight
        #         cur_evolve_E[cur_link_parent, i_link] = cur_alpha - other_parent_cur_level_alpha ## get other parent cur level alpha ## 
        #     tot_scaled_E.append(cur_evolve_E)
    return tot_scaled_E


def get_regular_link_idx_to_parent_idx(nn_links_one_side): 
    link_idx_to_parent_idx = {0: 0}
    for i_link in range(1, nn_links_one_side + 1):
        cur_link_parent = i_link - 1
        link_idx_to_parent_idx[i_link] = cur_link_parent
        # connectivity_arr[cur_link_parent, i_link] = 1.0

    for i_link in range(nn_links_one_side + 1, nn_links_one_side * 2 + 1):

        if i_link == nn_links_one_side + 1:
            cur_link_parent = 0
        else:
            cur_link_parent = i_link - 1
        link_idx_to_parent_idx[i_link] = cur_link_parent
        # connectivity_arr[cur_link_parent, i_link] = 1.0
    return link_idx_to_parent_idx
    
# all five #
## all five ##

# translate # # rotate # scale # ## 
## translate # rotate # scale
## data used for trainig the constraint distribution model #
## constraint path for optimization #


def main_a(cfg):
    
    
    PROJ_ROOT_FOLDER = cfg.run.root_dir
    
    ''' Generate necessary links here  '''    
    fixed_y_len = cfg.uni_manip.fixed_y_len
    base_x_len = cfg.uni_manip.base_x_len
    base_xys_tag = f"baseX_{base_x_len}_Y_{fixed_y_len}"
    st_len_one_side = cfg.uni_manip.st_len_one_side
    ed_len_one_side = cfg.uni_manip.ed_len_one_side
    nn_stages = cfg.uni_manip.nn_stages
    
    
    # tot_nn_links_one_side, tot_len_one_side = get_manipulator_infos(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)

    ''' General settings '''
    use_wandb = cfg.general.use_wandb
    
    uni_manip_2d_nn_particles = n_particles * 3
    
    
    goal_x = cfg.task.goal_x
    goal_y = cfg.task.goal_y
    
    if use_wandb:
        wandb.init(project="uni_manip_2d")

    
    nn_timesteps = cfg.task.nn_timesteps
    dt = cfg.task.dt
    dim = cfg.task.dim
    
    nn_maximum_links = 10 * 2 + 1
    
    nn_links = cfg.uni_manip.nn_links 
    nn_links_one_side = (nn_links - 1) // 2
    nn_joints = nn_links - 1
    
    len_one_side = 0.26666666666666666
    
    link_idx_to_parent_idx = get_regular_link_idx_to_parent_idx(nn_links_one_side) ## nn_links_one_side ##
    tot_E = get_constraint_evolution_path_curri_v2(nn_links, link_idx_to_parent_idx)
    
    prev_saved_best_ckpt_fn = None
    proj_link_relations = {
        idx: [idx] for idx in range(nn_links)
    }
    inherit_from_prev = cfg.optimization.inherit_from_prev
    
    additional_exp_tag = f"curri_v2__nreg_inherit_{inherit_from_prev}_seed_{cfg.seed}_contact_spring_d_{cfg.sim.contact_spring_d}_damping_{cfg.sim.contact_damping_d}"
    
    tot_E = list(reversed(tot_E))
    
    st_idx = cfg.run.start_inst_idx
    
    for i_inst in range(st_idx, len(tot_E)):
        # print(f"[inst {i_inst}] nn_links_one_side: {nn_links_one_side}, len_one_side: {len_one_side}")
        print(f"[inst {i_inst}]")
        print(f"{tot_E[i_inst]}")
        
        # /data/xueyi/softzoo/assets/obj_info_n_links_5_childlinklen_0.26666666666666666_baseX_0.1_Y_0.05.npy
        # len_one_side = 0.26666666666666666
        exp_tag = f"iinst_{i_inst}_nlinks_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{additional_exp_tag}_"
        
        obj_info_fn = f"assets/obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{base_xys_tag}.npy"
        
        obj_info_fn = os.path.join(PROJ_ROOT_FOLDER, obj_info_fn)
        
        obj_info = np.load(obj_info_fn, allow_pickle=True).item()
        
        uni_manip_2d_nn_particles = obj_info['particles_xs'].shape[0]
        link_joint_pos = obj_info['link_joint_pos']
        
        link_joint_dir = obj_info['link_joint_dir']
        link_parent_idx = obj_info['link_parent_idx']
        
        # cur_transformation_penalty_term = tot_transformation_penalty_loss[i_inst]
        
        uni_manip_2d = UniManip2D(uni_manip_2d_nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=exp_tag, cur_transformation_penalty_term=1.0, cfg=cfg)
        
        print(f"Calculating the graph connectivity info with nn_links: {nn_links_one_side}")
        # cur_connectivity_info = get_graph_connectivity_info(10)
        sub_cur_connectivity_info = tot_E[i_inst] # get_graph_connectivity_info(nn_links_one_side)
        cur_connectivity_info = np.zeros((nn_maximum_links, nn_maximum_links), dtype=np.float32)
        # cur_connectivity_info
        cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]] = sub_cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]]
        uni_manip_2d.set_graph_connectivity(cur_connectivity_info)
        

        connectivity_graph_info = uni_manip_2d.graph.graph_A.to_numpy()
        print(f"connectivity_graph_info: {connectivity_graph_info}")
        

        print(f"Start optimization!")
        uni_manip_2d.set_goal(goal_x, goal_y)
        uni_manip_2d.initialize()
        
        
        if (prev_saved_best_ckpt_fn is not None) and inherit_from_prev:
            ckpt_five_link_sv_ckpt_fn = prev_saved_best_ckpt_fn

            print(f"Loading from {ckpt_five_link_sv_ckpt_fn}")
            uni_manip_2d.load_proj_optimized_info(ckpt_five_link_sv_ckpt_fn, proj_link_relations)
        
        
        prev_saved_best_ckpt_fn = uni_manip_2d.optimize_with_planning(n_terminate=False)

        print(f"i_inst: {i_inst}, best saved to {prev_saved_best_ckpt_fn}")

        
        
    



from hydra import compose, initialize
# from omegaconf import OmegaConf
def main():
    with initialize(version_base="1.3", config_path="../cfgs", job_name="test_app"):
        cfg = compose(config_name="config")


    cfg.run.exp_folder_tag = f"expv4_projected_task_{cfg.task.task_idx}" # "expv4_projected"
    
    if cfg.dummy:
        cfg.run.exp_folder_tag = f"expv4_projected_dummy_task_{cfg.task.task_idx}"
    
    random_seed = cfg.seed
    ti_device_memory_fraction = 0.8
    ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=cfg.seed)
    
    
    main_a(cfg=cfg)
    


if __name__=='__main__':
    
    main()
    exit(0) 





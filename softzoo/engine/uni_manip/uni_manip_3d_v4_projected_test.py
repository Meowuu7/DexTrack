import numpy as np
import taichi as ti
import os 
import wandb
# import cma



from softzoo.engine.uni_manip.sim_engine.uni_manip_2d_projected import UniManip2D
from softzoo.engine.uni_manip.sim_engine.uni_manip_3d_projected import UniManip3D
import softzoo.engine.uni_manip.sim_engine.dyn_model_act_v2 as dyn_model_act
from softzoo.engine.uni_manip.sim_engine.graph_3d import Graph
import random
import torch

# dim = 2
quality = 1
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

RIGID_SPHERE = 0
RIGID_CUBE = 1
RIGID_SDF = 2

def set_seed_everywhere(seed=0):
    # env.action_space.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

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


# connectivity_arr = get_graph_connectivity_info_via_parent_index(link_parent_index)
def get_graph_connectivity_info_via_parent_index(link_parent_index):
    connectivity_arr = np.zeros((link_parent_index.shape[0], link_parent_index.shape[0]), dtype=np.float32)
    
    for i_link in range(link_parent_index.shape[0]):
        cur_link_parent_link = link_parent_index[i_link]
        connectivity_arr[cur_link_parent_link, i_link] = 1.0 ## 
    return connectivity_arr


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

def get_constraint_evolution_path(n_links, link_idx_to_original_parent): #
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


# all five #
#

# translate # # rotate # scale # ## 
## translate # rotate # scale
## data used for trainig the constraint distribution model #
## constraint path for optimization #

# a more powerful sims and a jmor e


def main_a(cfg):
    
    
    PROJ_ROOT_FOLDER = cfg.run.root_dir
    
    ''' Generate necessary links here  '''    
    fixed_y_len = cfg.uni_manip.fixed_y_len
    base_x_len = cfg.uni_manip.base_x_len
    base_xys_tag = f"baseX_{base_x_len}_Y_{fixed_y_len}"
    # st_len_one_side = cfg.uni_manip.st_len_one_side
    # ed_len_one_side = cfg.uni_manip.ed_len_one_side
    # nn_stages = cfg.uni_manip.nn_stages
    
    
    # tot_nn_links_one_side, tot_len_one_side = get_manipulator_infos(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
    
    ## 
    
    
    urdf_fn = cfg.uni_manip.urdf_fn ##
    print(f"urdf_fn: {urdf_fn}") ## get the urdf file ##
    
    robot_agent = dyn_model_act.RobotAgent(urdf_fn) # 
    
    init_vertices, init_faces, joint_idxes = robot_agent.active_robot.get_init_visual_pts(expanded_pts=False, joint_idxes=[])
    
    init_vertices = init_vertices.detach().cpu().numpy()
    init_faces = init_faces.detach().cpu().numpy()
    joint_idxes = joint_idxes.detach().cpu().numpy() ## 
    # (self, nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=None, cur_transformation_penalty_term=1000, cfg=None): #
    
    # uni_manip_2d_nn_particles
    nn_particles = init_vertices.shape[0]
    ### and get the connectivity infors ##
    ## get he roobt link and the child links ##
    # maxx_joint_idx = np.max(joint_idxes).item()
    
    
    
    link_name_to_link_idxes = robot_agent.active_robot.link_name_to_link_idxes ## 
    link_idx_to_link_name = {v: k for k, v in link_name_to_link_idxes.items()}
    link_name_to_link_struct = robot_agent.active_robot.link_name_to_link_struct ## get the link struct 
    
    link_idxes_list = list(link_name_to_link_idxes.values())
    maxx_link_idx, minn_link_idx = max(link_idxes_list), min(link_idxes_list)
    print(f"maxx_link_idx: {maxx_link_idx}, minn_link_idx: {minn_link_idx}")
    
    
    nn_links = maxx_link_idx + 1
    nn_joints = nn_links - 1
    nn_timesteps = cfg.task.nn_timesteps
    dim = 3
    dt = cfg.task.dt
    
    link_joint_pos = np.zeros((nn_links, 3), dtype=np.float32)
    link_joint_dir = np.zeros((nn_links, 3), dtype=np.float32)
    link_parent_idx = np.zeros((nn_links), dtype=np.int32)
    visited_child = {}
    for link_idx in range(nn_links):
        
        if link_idx in link_idx_to_link_name:
            cur_link_name = link_idx_to_link_name[link_idx]
            cur_link_struct = link_name_to_link_struct[cur_link_name]
            
            print(link_idx, cur_link_name)
            
            if cur_link_struct.joint is not None:
                for joint_nm in cur_link_struct.joint:
                    cur_joint_struct = cur_link_struct.joint[joint_nm]
                    cur_joint_child_nm = cur_joint_struct.child_link
                    cur_joint_parent_nm = cur_joint_struct.parent_link
                    assert cur_joint_parent_nm == cur_link_name
                    cur_joint_child_idx = link_name_to_link_idxes[cur_joint_child_nm]
                    
                    link_joint_pos[cur_joint_child_idx] = cur_joint_struct.origin_xyz.detach().cpu().numpy()
                    link_joint_dir[cur_joint_child_idx] = cur_joint_struct.axis_xyz.detach().cpu().numpy()
                    link_parent_idx[cur_joint_child_idx] = link_idx
                    visited_child[cur_joint_child_idx] = 1
            # for joint_nm in cur_link_struct.joint:
            #     if cur_link_struct.joint[joint_nm].child_link == 
            # link_joint_pos[link_idx] = cur_link_struct.joint.origin_xyz
            # link_joint_dir[link_idx] = cur_link_struct.joint.axis_xyz
            # # parent_link = cur_link_struct.
            # parent_link_name = cur_link_struct.joint.parent_link
            # parent_link_idx = link_name_to_link_idxes[parent_link_name]
            # link_parent_idx[link_idx] = parent_link_idx # pass the parent link idx
    for link_idx in range(nn_links):
        if link_idx not in visited_child:
            print(f"not in visited child: {link_idx}")
            link_parent_idx[link_idx] = link_idx
    
    for link_idx in range(nn_links):
        cur_link_parent_idx = link_parent_idx[link_idx]
        assert cur_link_parent_idx <= link_idx
        link_joint_pos[link_idx] = link_joint_pos[link_idx] + link_joint_pos[cur_link_parent_idx]
    
    
    ## TODO: a new config file for 3-D ##
    connectivity_arr = get_graph_connectivity_info_via_parent_index(link_parent_idx)
    prev_saved_best_ckpt_fn =  cfg.run.prev_saved_best_ckpt_fn
    if len(prev_saved_best_ckpt_fn) == 0:
        prev_saved_best_ckpt_fn = None
    
    # st_idx = cfg.run.start_inst_idx
    
    additional_exp_tag = f"task_{cfg.task.task_idx}_seed_{cfg.seed}_contact_spring_d_{cfg.sim.contact_spring_d}_damping_{cfg.sim.contact_damping_d}"
    
    exp_tag = f"iinst_{0}_nlinks_{nn_links}_{additional_exp_tag}_"
    
    
    # obj_info_fn = f"assets/obj_info_n_links_{nn_links}_{base_xys_tag}.npy"
    
    
    uni_manip_3d = UniManip3D(nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, init_vertices, joint_idxes, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=exp_tag, cur_transformation_penalty_term=1.0, cfg=cfg)
    
    maxx_nn_links = Graph.get_graph_nn_links()
    # print(f"Calculating the graph connectivity info with nn_links: {nn_links_one_side}")
    # cur_connectivity_info = get_graph_connectivity_info(10) #
    # sub_cur_connectivity_info = get_graph_connectivity_info(nn_links_one_side)
    cur_connectivity_info = np.zeros((maxx_nn_links, maxx_nn_links), dtype=np.float32)
    cur_connectivity_info[:connectivity_arr.shape[0], :connectivity_arr.shape[1]] = connectivity_arr[:connectivity_arr.shape[0], :connectivity_arr.shape[1]]
    uni_manip_3d.set_graph_connectivity(cur_connectivity_info)
    

    connectivity_graph_info = uni_manip_3d.graph.graph_A.to_numpy()
    print(f"connectivity_graph_info: {connectivity_graph_info}")
    

    print(f"Start optimization!")
    uni_manip_3d.set_goal(cfg.task.goal_x, cfg.task.goal_y, cfg.task.goal_z)
    uni_manip_3d.initialize()
    
    
    # if (prev_saved_best_ckpt_fn is not None) and inherit_from_prev:
    #     ckpt_five_link_sv_ckpt_fn = prev_saved_best_ckpt_fn

    #     print(f"Loading from {ckpt_five_link_sv_ckpt_fn}")
    #     uni_manip_2d.load_proj_optimized_info(ckpt_five_link_sv_ckpt_fn, proj_link_relations)
    
    ### uni_manip_3d ###
    prev_saved_best_ckpt_fn = uni_manip_3d.optimize_with_planning(n_terminate=True)

    # print(f"i_inst: {i_inst}, best saved to {prev_saved_best_ckpt_fn}")



    
    
    
    # ''' Start optimization '''
    # for i_inst in range(st_idx, len(tot_len_one_side)):
        
    #     nn_links_one_side = tot_nn_links_one_side[i_inst]
    #     len_one_side = tot_len_one_side[i_inst]
    #     proj_link_relations = tot_proj_link_relations[i_inst]
    #     nn_links = nn_links_one_side * 2 + 1
    #     nn_joints = nn_links - 1 ## 
        
    #     print(f"[inst {i_inst}] nn_links_one_side: {nn_links_one_side}, len_one_side: {len_one_side}")
        
        
        
    #     exp_tag = f"iinst_{i_inst}_nlinks_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{additional_exp_tag}_"
        
        
    #     obj_info_fn = f"assets/obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{base_xys_tag}.npy"
        
    #     obj_info_fn = os.path.join(PROJ_ROOT_FOLDER, obj_info_fn)
        
    #     obj_info = np.load(obj_info_fn, allow_pickle=True).item()
        
    #     uni_manip_2d_nn_particles = obj_info['particles_xs'].shape[0]
    #     link_joint_pos = obj_info['link_joint_pos']
        
    #     link_joint_dir = obj_info['link_joint_dir']
    #     link_parent_idx = obj_info['link_parent_idx']
        
    #     cur_transformation_penalty_term = tot_transformation_penalty_loss[i_inst]
        
    #     uni_manip_2d = UniManip2D(uni_manip_2d_nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=exp_tag, cur_transformation_penalty_term=cur_transformation_penalty_term, cfg=cfg)
        
    #     print(f"Calculating the graph connectivity info with nn_links: {nn_links_one_side}")
    #     # cur_connectivity_info = get_graph_connectivity_info(10)
    #     sub_cur_connectivity_info = get_graph_connectivity_info(nn_links_one_side)
    #     cur_connectivity_info = np.zeros((nn_maximum_links, nn_maximum_links), dtype=np.float32)
    #     # cur_connectivity_info
    #     cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]] = sub_cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]]
    #     uni_manip_2d.set_graph_connectivity(cur_connectivity_info)
        

    #     connectivity_graph_info = uni_manip_2d.graph.graph_A.to_numpy()
    #     print(f"connectivity_graph_info: {connectivity_graph_info}")
        

    #     print(f"Start optimization!")
    #     uni_manip_2d.set_goal(goal_x, goal_y)
    #     uni_manip_2d.initialize()
        
        
    #     if (prev_saved_best_ckpt_fn is not None) and inherit_from_prev:
    #         ckpt_five_link_sv_ckpt_fn = prev_saved_best_ckpt_fn

    #         print(f"Loading from {ckpt_five_link_sv_ckpt_fn}")
    #         uni_manip_2d.load_proj_optimized_info(ckpt_five_link_sv_ckpt_fn, proj_link_relations)
        
        
    #     prev_saved_best_ckpt_fn = uni_manip_2d.optimize_with_planning(n_terminate=False)

    #     print(f"i_inst: {i_inst}, best saved to {prev_saved_best_ckpt_fn}")



    
    ''' General settings '''
    # use_wandb = cfg.general.use_wandb
    
    # uni_manip_2d_nn_particles = n_particles * 3
    
    
    # goal_x = cfg.task.goal_x
    # goal_y = cfg.task.goal_y
    
    # if use_wandb:
    #     wandb.init(project="uni_manip_2d")

    
    # # nn_timesteps = cfg.task.nn_timesteps
    # # dt = cfg.task.dt
    # # dim = cfg.task.dim
    
    # nn_maximum_links = 10 * 2 + 1
    
    
    # ''' Define the manipulator transformation process ''' 
    # # tot_transformation_penalty_loss = [1000, 1000, 1000, 1000, 1000, 3000]
    
    # tot_transformation_penalty_loss = [100, 100, 100, 10, 10, 10]
    
    # tot_proj_link_relations = [
    #     {0: [0], 1: [1, 2], 2: [3, 4]}, 
    #     {ii: [ii] for ii in range(tot_nn_links_one_side[1] * 2 + 1)},
    #     {ii: [ii * 2 - 1, ii * 2] if ii > 0 else [0] for ii in range(tot_nn_links_one_side[2] * 2 + 1) },
    #     {ii: [ii] for ii in range(tot_nn_links_one_side[3] * 2 + 1)},
    #     {ii: [ii * 2 - 1, ii * 2] if ii > 0 else [0] for ii in range(tot_nn_links_one_side[4] * 2 + 1) },
    #     {ii: [ii] for ii in range(tot_nn_links_one_side[5] * 2 + 1)},
    # ]
    
    
    # tot_nn_links_one_side = list(reversed(tot_nn_links_one_side))
    # tot_len_one_side = list(reversed(tot_len_one_side))
    # tot_proj_link_relations = list(reversed(tot_proj_link_relations))
    # tot_transformation_penalty_loss = list(reversed(tot_transformation_penalty_loss))
    
    # prev_saved_best_ckpt_fn =  cfg.run.prev_saved_best_ckpt_fn
    # if len(prev_saved_best_ckpt_fn) == 0:
    #     prev_saved_best_ckpt_fn = None


    # st_idx = cfg.run.start_inst_idx

    # inherit_from_prev = cfg.optimization.inherit_from_prev
    
    # # v4 --- kd = 0.1; act-reaching-dist-thres = 1e-3
    # additional_exp_tag = f"task_{cfg.task.task_idx}_inherit_{inherit_from_prev}_seed_{cfg.seed}_contact_spring_d_{cfg.sim.contact_spring_d}_damping_{cfg.sim.contact_damping_d}"
    
    
    # ''' Start optimization '''
    # for i_inst in range(st_idx, len(tot_len_one_side)):
        
    #     nn_links_one_side = tot_nn_links_one_side[i_inst]
    #     len_one_side = tot_len_one_side[i_inst]
    #     proj_link_relations = tot_proj_link_relations[i_inst]
    #     nn_links = nn_links_one_side * 2 + 1
    #     nn_joints = nn_links - 1
        
    #     print(f"[inst {i_inst}] nn_links_one_side: {nn_links_one_side}, len_one_side: {len_one_side}")
        
        
        
    #     exp_tag = f"iinst_{i_inst}_nlinks_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{additional_exp_tag}_"
        
        
    #     obj_info_fn = f"assets/obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{base_xys_tag}.npy"
        
    #     obj_info_fn = os.path.join(PROJ_ROOT_FOLDER, obj_info_fn)
        
    #     obj_info = np.load(obj_info_fn, allow_pickle=True).item()
        
    #     uni_manip_2d_nn_particles = obj_info['particles_xs'].shape[0]
    #     link_joint_pos = obj_info['link_joint_pos']
        
    #     link_joint_dir = obj_info['link_joint_dir']
    #     link_parent_idx = obj_info['link_parent_idx']
        
    #     cur_transformation_penalty_term = tot_transformation_penalty_loss[i_inst]
        
    #     uni_manip_2d = UniManip2D(uni_manip_2d_nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=exp_tag, cur_transformation_penalty_term=cur_transformation_penalty_term, cfg=cfg)
        
    #     print(f"Calculating the graph connectivity info with nn_links: {nn_links_one_side}")
    #     # cur_connectivity_info = get_graph_connectivity_info(10)
    #     sub_cur_connectivity_info = get_graph_connectivity_info(nn_links_one_side)
    #     cur_connectivity_info = np.zeros((nn_maximum_links, nn_maximum_links), dtype=np.float32)
    #     # cur_connectivity_info
    #     cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]] = sub_cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]]
    #     uni_manip_2d.set_graph_connectivity(cur_connectivity_info)
        

    #     connectivity_graph_info = uni_manip_2d.graph.graph_A.to_numpy()
    #     print(f"connectivity_graph_info: {connectivity_graph_info}")
        

    #     print(f"Start optimization!")
    #     uni_manip_2d.set_goal(goal_x, goal_y)
    #     uni_manip_2d.initialize()
        
        
    #     if (prev_saved_best_ckpt_fn is not None) and inherit_from_prev:
    #         ckpt_five_link_sv_ckpt_fn = prev_saved_best_ckpt_fn

    #         print(f"Loading from {ckpt_five_link_sv_ckpt_fn}")
    #         uni_manip_2d.load_proj_optimized_info(ckpt_five_link_sv_ckpt_fn, proj_link_relations)
        
        
    #     prev_saved_best_ckpt_fn = uni_manip_2d.optimize_with_planning(n_terminate=False)

    #     print(f"i_inst: {i_inst}, best saved to {prev_saved_best_ckpt_fn}")





from hydra import compose, initialize
# from omegaconf import OmegaConf
def main():
    # with initialize(version_base="1.3", config_path="../cfgs", job_name="test_app"):
    #     cfg = compose(config_name="config")
    #### get the config ####
    with initialize(version_base="1.3", config_path="../cfgs", job_name="test_app"):
        cfg = compose(config_name="config_3d")


    ## get the config 3d ##
    cfg.run.exp_folder_tag = f"expv4_projected_task_{cfg.task.task_idx}"
    
    random_seed = cfg.seed
    
    ti_device_memory_fraction = 0.8
    ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=cfg.seed)
    
    set_seed_everywhere(cfg.seed)
    
    main_a(cfg=cfg)
    


if __name__=='__main__':
    
    main()
    exit(0) 
    
    



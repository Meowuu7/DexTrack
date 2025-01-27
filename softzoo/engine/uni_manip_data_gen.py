import numpy as np
import torch as th

import os

dim = 2
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3


def generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len=0.1, base_x_len=0.1):
    per_link_len = len_one_side / float(nn_links_one_side)
    
    
    
    particle_density = n_particles / (base_x_len * fixed_y_len)
    base_link_n_particles = particle_density * (base_x_len * fixed_y_len)
    child_link_n_particles = particle_density * (per_link_len * fixed_y_len)
    
    base_link_n_particles = int(base_link_n_particles)
    child_link_n_particles = int(child_link_n_particles)
    
    ## 
    rnd_xys = np.random.rand(base_link_n_particles, dim) * np.array([base_x_len, fixed_y_len], dtype=np.float32)[None, :]
    base_link_xys = np.array([0.45, 0.45], dtype=np.float32)[None, :] + rnd_xys ## get the random initialized xys ##
    base_link_particles_link_idx = np.zeros((base_link_n_particles, ), dtype=np.int32) # the bae link particles idxes 
    
    link_xys = [base_link_xys]
    link_particle_link_idx = [base_link_particles_link_idx]
    link_joint_pos = [np.array([0.0, 0.0], dtype=np.float32)]
    link_parent_idx = [-1]
    
    
    child_link_idx = 1
    joint_x = 0.45
    joint_y = 0.45
    for i_link in range(nn_links_one_side):
        cur_link_st_x = joint_x - per_link_len
        rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, fixed_y_len], dtype=np.float32)
        cur_link_xys = np.array([cur_link_st_x, joint_y], dtype=np.float32)[None, :] + rnd_xys
        cur_link_idxes = np.ones((child_link_n_particles, ), dtype=np.int32) * child_link_idx
        cur_link_joint_pos = np.array([joint_x, joint_y], dtype=np.float32)
        cur_link_parent_idx = child_link_idx - 1
        
        link_xys.append(cur_link_xys)
        link_particle_link_idx.append(cur_link_idxes)
        link_joint_pos.append(cur_link_joint_pos)
        link_parent_idx.append(cur_link_parent_idx)
        
        joint_x -= per_link_len
        child_link_idx += 1
    
    joint_x = 0.55
    joint_y = 0.45
    for i_link in range(nn_links_one_side):
        cur_link_st_x = joint_x # - per_link_len
        rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, fixed_y_len], dtype=np.float32)
        cur_link_xys = np.array([cur_link_st_x, joint_y], dtype=np.float32)[None, :] + rnd_xys
        cur_link_idxes = np.ones((child_link_n_particles, ), dtype=np.int32) * child_link_idx
        cur_link_joint_pos = np.array([joint_x, joint_y], dtype=np.float32)
        cur_link_parent_idx = child_link_idx - 1 if i_link > 0 else 0
        
        link_xys.append(cur_link_xys)
        link_particle_link_idx.append(cur_link_idxes)
        link_joint_pos.append(cur_link_joint_pos)
        link_parent_idx.append(cur_link_parent_idx)
        
        joint_x += per_link_len
        child_link_idx += 1
    
    obj_particles = np.concatenate(link_xys, axis=0)
    particle_link_idxes = np.concatenate(link_particle_link_idx, axis=0)
    link_joint_pos = np.stack(link_joint_pos, axis=0)
    link_parent_idx = np.array(link_parent_idx, dtype=np.int32)
    
    link_joint_dir = [[1.0, 0.0] for _ in range(nn_links_one_side * 2 + 1)]
    link_joint_dir = np.array(link_joint_dir, dtype=np.float32)
    
    ## link joint dir ##
    obj_info = {
        'particles_xs': obj_particles,
        'particle_link_idxes': particle_link_idxes,
        'link_joint_pos': link_joint_pos,
        'link_joint_dir': link_joint_dir,
        'link_parent_idx': link_parent_idx
    }
    
    # asset_root_folder = os.path.join(PROJ_ROOT_FOLDER, "assets")
    # os.makedirs(asset_root_folder, exist_ok=True)
    
    # obj_info_sv_fn = os.path.join(PROJ_ROOT_FOLDER, f"assets", f"obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_baseX_{base_x_len}_Y_{fixed_y_len}.npy")
    # np.save(obj_info_sv_fn, obj_info)
    # print(f"Object information saved to {obj_info_sv_fn}")
    
    
# def test_link_gen():
#     # generate_test_links_general(dim, nn_links_one_side, len_one_side):
#     tot_nn_links_one_side = [1, 2, 2, 4, 4, 8] # link one side #
#     tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
#     dim = 2
#     for nn_links_one_side, len_one_side in zip(tot_nn_links_one_side, tot_len_one_side):
#         generate_test_links_general(dim, nn_links_one_side, len_one_side)

# def link_gen_general_v2(st_len_one_side, ed_len_one_side, nn_stages):
#     # generate_test_links_general(dim, nn_links_one_side, len_one_side):
#     # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8] # link one side #
#     # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    
#     link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
#     tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
    
#     tot_nn_links_one_side = []
#     tot_len_one_side = []
#     st_nn_link_one_side = 1
#     for i_stage in range(len(tot_len_one_side_unqie)):
#         tot_nn_links_one_side.append(st_nn_link_one_side)
#         tot_len_one_side.append(tot_len_one_side_unqie[i_stage // 2])
        
#         if i_stage % 2 == 0:
#             st_nn_link_one_side *= 2
    
#     print("tot_nn_links_one_side: ", tot_nn_links_one_side)
#     print(f"tot_len_one_side: {tot_len_one_side}")
    
    
#     dim = 2
#     for nn_links_one_side, len_one_side in zip(tot_nn_links_one_side, tot_len_one_side):
#         generate_test_links_general(dim, nn_links_one_side, len_one_side)



def get_graph_connectivity_info(nn_link_one_side):
    nn_tot_links = nn_link_one_side * 2 + 1
    connectivity_arr = np.zeros((nn_tot_links, nn_tot_links), dtype=np.float32) ## 
    
    connectivity_arr[0, 0] = 1.0
    
    ## 
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

def link_gen_general_v3(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=0.1, base_x_len=0.1):
    # generate_test_links_general(dim, nn_links_one_side, len_one_side):
    # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8] # link one side #
    # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    
    # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
    # tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
    
    ### tot_len_one_side_unique #
    # tot_len_one_side = []
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
    # tot_nn_links_one_side = []
    # tot_len_one_side = []
    # st_nn_link_one_side = 1
    # for i_stage in range(len(tot_len_one_side_unqie)):
    #     tot_nn_links_one_side.append(st_nn_link_one_side)
    #     tot_len_one_side.append(tot_len_one_side_unqie[i_stage // 2])
        
    #     if i_stage % 2 == 0:
    #         st_nn_link_one_side *= 2
    
    print("tot_nn_links_one_side: ", tot_nn_links_one_side)
    print(f"tot_len_one_side: {tot_len_one_side}")
    
    maxx_nn_links_one_side = 10
    
    for i_inst, (cur_nn_link_one_side, cur_tot_len_one_side) in enumerate(zip(tot_nn_links_one_side, tot_len_one_side)):
        cur_inst_connectivity_arr = get_graph_connectivity_info(cur_nn_link_one_side)
        per_link_len = cur_tot_len_one_side / float(cur_nn_link_one_side) ## get the nn_links one side ##
        ## get node attributes ##
        node_Xs = []
        for i_link in range(cur_nn_link_one_side * 2 + 1):
            if i_link == 0:
                cur_link_x = 0.1
                cur_link_y = 0.1
            else:
                cur_link_x = per_link_len
                cur_link_y = fixed_y_len
            node_Xs.append([cur_link_x, cur_link_y])
        node_Xs = np.array(node_Xs).astype(np.float32)
        
        if cur_nn_link_one_side < maxx_nn_links_one_side:
            full_connectivity_arr = np.eye(maxx_nn_links_one_side * 2 + 1, dtype=np.float32)
            full_connectivity_arr[: cur_inst_connectivity_arr.shape[0], : cur_inst_connectivity_arr.shape[1]] = cur_inst_connectivity_arr[:, :] ## get the inst connectivity arrs ##
            
            reamaining_node_arrs = np.zeros((maxx_nn_links_one_side * 2 + 1 - node_Xs.shape[0], 2)).astype(np.float32) ## remaining node features ##
            
            cur_inst_connectivity_arr = full_connectivity_arr
            node_Xs = np.concatenate([node_Xs, reamaining_node_arrs], axis=0)
        
        data_dict = {
            'X': node_Xs,
            'E': cur_inst_connectivity_arr
        }
        
        cur_inst_sv_fn = f"uni_manip_nn_links_one_side_{cur_nn_link_one_side}_base_x_{base_x_len}_y_{fixed_y_len}_child_x_{per_link_len}_y_{fixed_y_len}.npy"
        cur_inst_sv_fn = os.path.join(sv_root_folder, cur_inst_sv_fn)
        np.save(cur_inst_sv_fn, data_dict) ## 
        print(f"Instance {i_inst} saved to {cur_inst_sv_fn}")
            
        
    
    
    
    # dim = 2
    # for nn_links_one_side, len_one_side in zip(tot_nn_links_one_side, tot_len_one_side):
    #     generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len=fixed_y_len, base_x_len=base_x_len)

if __name__=='__main__':
    
    st_len_one_side = 0.2
    ed_len_one_side = 0.4 ## nnlinks and other relatiosns ##
    nn_stages = 6 ## fixed_y_len; ## base_x_len ##
    fixed_y_len =  0.05
    base_x_len = 0.1
    
    
    sv_root_folder  = "/data/xueyi/softzoo/data/uni_manip_2d_data"
    sv_root_folder = "/data2/xueyi/uni_manip/data"
    os.makedirs(sv_root_folder, exist_ok=True)
    ## link gen general v3 ##
    link_gen_general_v3(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)

# python softzoo/engine/uni_manip_data_gen.py 


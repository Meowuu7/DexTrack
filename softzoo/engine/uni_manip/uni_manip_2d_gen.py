import numpy as np
import os

dim = 2
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

PROJ_ROOT_FOLDER = "/data/xueyi/softzoo"


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
    
    asset_root_folder = os.path.join(PROJ_ROOT_FOLDER, "assets")
    os.makedirs(asset_root_folder, exist_ok=True)
    
    obj_info_sv_fn = os.path.join(PROJ_ROOT_FOLDER, f"assets", f"obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_baseX_{base_x_len}_Y_{fixed_y_len}.npy")
    np.save(obj_info_sv_fn, obj_info)
    print(f"Object information saved to {obj_info_sv_fn}")
    
    
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
    link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages // 2 - 1) ## assum that we have six stages in total --- then we need to get the stage length and xxx
    st_nn_link_one_side = 1
    for i in range(nn_stages):
        cur_link_len = st_len_one_side + (i // 2) * link_len_one_side_interval
        
        tot_len_one_side.append(cur_link_len)
        tot_nn_links_one_side.append(st_nn_link_one_side)
        
        if i % 2 == 0:
            st_nn_link_one_side = st_nn_link_one_side * 2
    
    
    print("tot_nn_links_one_side: ", tot_nn_links_one_side)
    print(f"tot_len_one_side: {tot_len_one_side}")
    
    
    dim = 2
    for nn_links_one_side, len_one_side in zip(tot_nn_links_one_side, tot_len_one_side):
        generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len=fixed_y_len, base_x_len=base_x_len)




def get_manipulator_infos(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=0.1, base_x_len=0.1):
    # generate_test_links_general(dim, nn_links_one_side, len_one_side):
    # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8] # link one side #
    # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    
    # link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
    # tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
    
    ### tot_len_one_side_unique #
    # tot_len_one_side = []
    tot_nn_links_one_side = []
    tot_len_one_side = []
    link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages // 2 - 1) ## assum that we have six stages in total --- then we need to get the stage length and xxx
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



if __name__=='__main__':
    
    nn_links = 5
    nn_links_one_side = (nn_links - 1) // 2
    
    dim = 2
    len_one_side = (0.26666666666666666 + 0.30000000000000004) / 2.0
    # fixed_y_len = 0.1
    # base_x_len = 0.1
    
    fixed_y_len = 0.05
    base_x_len = 0.1
    
    ## for evaluating on new manipulators ## for evaluating on new manipulators ##
    ## for new evaluations ##
    generate_test_links_general_flexy(dim, nn_links_one_side, len_one_side, fixed_y_len=fixed_y_len, base_x_len=base_x_len)

# import hydra
# # @hydra.main(config_path="/root/diffsim/softzoo/cfgs", config_name="config", version_base="1.3")
# # @hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
# def main_a(cfg):
#     # PROJ_ROOT_FOLDER = "/root/diffsim/softzoo"
    
#     # PROJ_ROOT_FOLDER = "/data/xueyi/softzoo" ## set the proj root folder ## \
        
#     PROJ_ROOT_FOLDER = cfg.run.root_dir
    
#     ''' Generate necessary links here  '''
#     # fixed_y_len = 0.05
#     # base_x_len = 0.1
#     # ## get the gen general v3 ##
#     # # print(f"Start generating v3 manipulators")
#     # base_xys_tag = f"baseX_{base_x_len}_Y_{fixed_y_len}"
#     # st_len_one_side = 0.2
#     # ed_len_one_side = 0.4
#     # nn_stages = 6
    
#     fixed_y_len = cfg.uni_manip.fixed_y_len
#     base_x_len = cfg.uni_manip.base_x_len
#     base_xys_tag = f"baseX_{base_x_len}_Y_{fixed_y_len}"
#     st_len_one_side = cfg.uni_manip.st_len_one_side
#     ed_len_one_side = cfg.uni_manip.ed_len_one_side
#     nn_stages = cfg.uni_manip.nn_stages
    
#     # link_gen_general_v3(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
    
#     tot_nn_links_one_side, tot_len_one_side = get_manipulator_infos(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
#     # exit(0)


# ## Get the softzoo manipulators informations ##

# python softzoo/engine/uni_manip/uni_manip_2d_gen.py
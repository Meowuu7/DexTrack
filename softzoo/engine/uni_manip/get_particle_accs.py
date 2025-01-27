

import numpy as np
# import taichi as ti
import os 
# import wandb
# import cma
# import hydra

# ti.init(arch=ti.gpu, )  # Try to run on GPU

# random_seed = 42
# ti_device_memory_fraction = 0.8
# ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=random_seed)

## structure of the unified manipulator in 2D ##
## currently --- kinematics only ##

dim = 2
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

RIGID_SPHERE = 0
RIGID_CUBE = 1
RIGID_SDF = 2




def get_diff_particle_infos_fr_ckpt_folder(exp_folder, dt):
    ckpt_folder = os.path.join(exp_folder, "checkpoints")
    ckpt_fn = os.path.join(ckpt_folder, f"ckpt_best.npy")
    save_info = np.load(ckpt_fn, allow_pickle=True).item()
    particle_xs = save_info['particle_xs'] ## get the particle xs from the ckpt fn ##
    ### particle xs from the fn ## 
    ### particle xs: nn_timesteps x nn_particles x 2 ###
    particle_vels = []
    particle_accs = []
    for i_ts in range(particle_xs.shape[0]):
        cur_particle_xs = particle_xs[i_ts] ## cur_particle_xs # 
        if i_ts == 0:
            cur_particle_vels = np.zeros_like(cur_particle_xs)
            cur_particle_accs = np.zeros_like(cur_particle_xs)
        else:
            # x_t = x_{t - 1} + dt * v_t 
            cur_particle_vels = (particle_xs[i_ts] - particle_xs[i_ts - 1]) / dt
            # v_t = v_{t - 1} + dt * a_t 
            cur_particle_accs = (cur_particle_vels - particle_vels[-1]) / dt
        particle_vels.append(cur_particle_vels)
        particle_accs.append(cur_particle_accs) ## get the particle accs ## 
    ###### ===== get the particle vels and particle accs ===== ######
    particle_vels = np.stack(particle_vels, axis=0)
    particle_accs = np.stack(particle_accs, axis=0) ## 
    save_info['particle_vels'] = particle_vels
    save_info['particle_accs'] = particle_accs
    
    
    particle_diff_sv_fn = f"ckpt_best_diff.npy"
    particle_diff_sv_fn = os.path.join(ckpt_folder, particle_diff_sv_fn) ## ckpt_folder with the particle_diff_sv_fn ## 
    np.save(particle_diff_sv_fn, save_info) 
    print(f"particle info with diff saved to {particle_diff_sv_fn}") ## get the particle diff sv fn ##
    
    # return save_info 

## get particles accs ##
# or does we need so many particles? #
# positions, accs --- scale them to a certain range and use them for the training #

# point cloud diffusions #


def main_a(cfg):
    # PROJ_ROOT_FOLDER = "/root/diffsim/softzoo"
    
    # PROJ_ROOT_FOLDER = "/data/xueyi/softzoo" ## get the proj root folder ##
        
    PROJ_ROOT_FOLDER = cfg.run.root_dir
    
    ''' Generate necessary links here  '''
    # fixed_y_len = 0.05
    # base_x_len = 0.1
    # ## get the gen general v3 ##
    # # print(f"Start generating v3 manipulators")
    # base_xys_tag = f"baseX_{base_x_len}_Y_{fixed_y_len}"
    # st_len_one_side = 0.2
    # ed_len_one_side = 0.4
    # nn_stages = 6
    
    dt = cfg.task.dt 
    
    ###### ==== load the optimized dict; convert the particle information to the particle accs ===== #####
    
    save_root_dir = cfg.run.root_dir
    exp_tag = cfg.run.exp_folder_tag
    
    
    # expv4_projected
    # expv4_projected_task_0
    exp_tags = ["expv4_projected", "expv4_projected_task_0", "expv4_projected_task_2"]
    
    for exp_tag in exp_tags:
        print(f"current exp_tag: {exp_tag}")
        
        cur_save_root_dir = os.path.join(save_root_dir, exp_tag)
        tot_optimized_folders = os.listdir(cur_save_root_dir)
        
        tot_optimized_folders = [fn for fn in tot_optimized_folders if not (("curri" in fn) and ("v2" not in fn))]
    
        for cur_folder in tot_optimized_folders:
            cur_optimized_folder = os.path.join(cur_save_root_dir, cur_folder) ## get the optimized folders ##
            # particle_diff_sv_info = 
            ### ==== calculate particle diff mvovments and save them ==== ###
            try: # 
                get_diff_particle_infos_fr_ckpt_folder(cur_optimized_folder, dt=dt)
            except:
                pass
    
    exit(0)
    
    
    
from hydra import compose, initialize
from omegaconf import OmegaConf
# @hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main():
    
    with initialize(version_base="1.3", config_path="../cfgs", job_name="test_app"):
        cfg = compose(config_name="config")
    
    random_seed = cfg.seed
    # ti_device_memory_fraction = 0.8
    # ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=cfg.seed)
    
    main_a(cfg=cfg)





if __name__=='__main__':
    
    
    
    main()
    exit(0) 
    
   
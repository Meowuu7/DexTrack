

import numpy as np
import taichi as ti
import os 
import wandb
import cma

from softzoo.engine.uni_manip.sim_engine.uni_manip_2d_maximal import UniManip2D

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


from hydra import compose, initialize

def test_sampled_pcd_w_actions(samples_fn):
    
    
    with initialize(version_base="1.3", config_path="../cfgs", job_name="test_app"):
        cfg = compose(config_name="config")
        
    ti_device_memory_fraction = 0.8
    ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=cfg.seed)
    
    
        
    cfg.run.exp_folder_tag = "expv4_testsamples_"
    
    # test manip 
    samples = np.load(samples_fn, allow_pickle=True).item()
    samples = samples['samples']
    sampled_pcs = samples['X']
    sampled_acts = samples['E']
    
    
    
    nn_samples = sampled_pcs.shape[0]
    for i_s in range(nn_samples):
        cur_sampled_pcs = sampled_pcs[i_s]
        
        nn_sampled_pts = cur_sampled_pcs.shape[0]
        nn_timesteps = 10
        dim = 2
        dt = 1e-1
        
        # the uni manip struct #
        uni_manip_struct = UniManip2D(nn_sampled_pts, 3, 2, 10, 2, dt, None, None, None, None, exp_tag="test_samples", cur_transformation_penalty_term=1000, cfg=cfg)
        
        # load_optimized_particle_accs()
        uni_manip_struct.initialize()
        uni_manip_struct.load_optimized_particle_accs(sampled_pcs[i_s], sampled_acts[i_s]) ## sampled pcs and the sampled acts ## 
        ## ### 
        uni_manip_struct.forward_stepping_particles(sv_ckpt_tag=f"best_{i_s}") ## for the forward stepping ##





if __name__=='__main__':
    samples_fn = "/data/xueyi/uni_manip/exp/test_scaled_pcd_/samples000029000.npy"
    test_sampled_pcd_w_actions(samples_fn=samples_fn) ## get the samples fn ##
    # python uni_manip/test_uni_manip_2d_v4.py
    # 
    # 


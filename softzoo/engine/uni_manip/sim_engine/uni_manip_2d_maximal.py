

import numpy as np
import taichi as ti
import os 
import wandb
import cma

from softzoo.engine.uni_manip.sim_engine.graph import Graph

# ti.init(arch=ti.gpu, )  # Try to run on GPU

# random_seed = 42
# ti_device_memory_fraction = 0.8
# ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=random_seed)



dim = 2
quality = 1
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

RIGID_SPHERE = 0
RIGID_CUBE = 1
RIGID_SDF = 2


# link idx to pcs #
# link idx to parent #
# link idx to parent #
# link idx 


@ti.data_oriented
class UniManip2D:
    def __init__(self, nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=None, cur_transformation_penalty_term=1000, cfg=None):
        # self.link_idx_to_pcs = link_idx_to_pcs # 
        # self.link_idx_to_parent = link_idx_to_parent
        # self.link_idx_to_joint_info = link_idx_to_joint_info
        # self.joint_idx_to_constraint_factor = joint_idx_to_constraint_factor # maximal 
        # self.joint idx to constraint factor = 
        self.dim = dim # self.dim = dim # # self.dim = dim ###
        # transform optimization results across different kinds of manipulators -> try to use the originally optimized transformations to initilize the following transformations
        self.nn_particles = nn_particles
        self.nn_links = nn_links
        self.nn_joints = nn_joints
        self.nn_timesteps = nn_timesteps
        
        self.cur_transformation_penalty_term = cur_transformation_penalty_term
        
        self.cfg = cfg
        
        # self.point_to_link_idx = ti.field(dtype=ti.i32, ) # joint_dirs #
        # self.joint_pos = ti.field(dtype=ti.f32, shape=(self.nn_joints, 2)) # joint_dirs #
        # self.joint_dirs = ti.field(dtype=ti.f32, shape=(self.nn_joints, 2)) # joint_dirs #
        # self.joint_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_joints, )) #
        # self.joint_dirs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_joints, )) ## TODO: should be normalized to the 1-norm vector in the initialization and the following time-stepping processes as well ##
        
        ''' Making the data folder for saving checkpoints '''
        # save_root_dir = "/data2/xueyi/uni_manip/exp"
        # save_root_dir = "/data/xueyi/uni_manip/exp"
    
        
        self.use_wandb = self.cfg.general.use_wandb
        
        CODE_ROOT_FOLDER = self.cfg.run.proj_dir
        PROJ_ROOT_FOLDER = self.cfg.run.root_dir
        exp_folder_tag = self.cfg.run.exp_folder_tag
        # save_root_dir = os.path.join(PROJ_ROOT_FOLDER, "exp")
        save_root_dir = os.path.join(PROJ_ROOT_FOLDER, exp_folder_tag)
        os.makedirs(save_root_dir, exist_ok=True)


        if exp_tag is not None:
            self.save_folder = os.path.join(save_root_dir, f"n_links_{self.nn_links}_tag_{exp_tag}")
        else:
            self.save_folder = os.path.join(save_root_dir, f"n_links_{self.nn_links}")
        os.makedirs(self.save_folder, exist_ok=True)
        
        self.ckpt_sv_folder = os.path.join(self.save_folder, "checkpoints")
        os.makedirs(self.ckpt_sv_folder, exist_ok=True)
        
        self.log_sv_fn = os.path.join(self.save_folder, "log.txt") ## get the logging file ##
        
        ''' Simulation parameters '''
        self.dt = dt
        # self.link_rotational_lr = 0.001
        # self.link_translational_lr = 0.001 ## graspit! ##
        self.link_rotational_lr = self.cfg.optimization.link_rotational_lr
        self.link_translational_lr = self.cfg.optimization.link_translational_lr
        
        
        ''' Get initial articulated object information -- at the rest pose state '''
        if obj_info_fn is not None:
            self.obj_info_fn = obj_info_fn
            self.obj_info = np.load(self.obj_info_fn, allow_pickle=True).item()
            self.ori_particles_xs = self.obj_info['particles_xs']
            self.particle_link_idxes = self.obj_info['particle_link_idxes']
            
            expanded_ori_particles_xs = self.ori_particles_xs[None, :, :]
            expanded_ori_particles_xs = np.repeat(expanded_ori_particles_xs, self.nn_timesteps, axis=0)
            self.expanded_ori_particles_xs = expanded_ori_particles_xs
            
        else:
            self.expanded_ori_particles_xs = None
            self.particle_link_idxes = None
        
        ## particle_vels, link_rotational_dot_mtx ##
        
        ''' Particle States '''
        self.particle_xs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_particles), needs_grad=True)

        self.particle_vels = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_particles), needs_grad=True)
        self.particle_link_idx = ti.field(dtype=ti.i32, shape=(nn_particles, ))
        
        if self.expanded_ori_particles_xs is not None:
            self.particle_xs.from_numpy(self.expanded_ori_particles_xs)
            self.particle_link_idx.from_numpy(self.particle_link_idxes) 
            
        ''' Particle velocitieis and the actions ''' 
        ##### particle_vels, particle_accs #####
        self.particle_vels =  ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_particles), needs_grad=True)
        self.particle_accs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_particles), needs_grad=True)
        
        
        ''' The graph connectivity matrix '''
        self.graph = Graph()
        ## TODO: set the graph connectivity array; ##
        ## TODO: normalize the graph connectivity array; ##
        
        
        
        
        ''' Link joint infos '''
        self.link_joint_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        self.link_joint_dir = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        self.link_parent_idx = ti.field(dtype=ti.i32, shape=(self.nn_links))
        
        if link_joint_pos is not None:
            self.link_joint_pos.from_numpy(link_joint_pos)
            self.link_joint_dir.from_numpy(link_joint_dir)
            self.link_parent_idx.from_numpy(link_parent_idx)
            
        ## selflink ##
        
        # link_translational_vels, link_rotational_vels #
        
        
        ''' Link SE(3) States '''
        if self.dim == 2: ## rotation angle and the joint ##
            # rotation angle # ## about the joint angle acclerations --- joint angle acceleraions; joint angles ## joint angle accelerations ## joint accelerations ## joint torque; --> body joint/torques --> such information added to each particle ##
            ## just what the internal forces can provide ##
            
            ## joint accelerations ## 
            self.link_rotational_vecs = ti.Vector.field(1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
            self.link_rotational_vels = ti.Vector.field(1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        else:
            # quaternion #
            self.link_rotational_vecs = ti.Vector.field(self.dim + 1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
            self.link_rotational_vels = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.link_rotational_mtx = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.link_rotational_dot_mtx = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        
        self.link_translational_vecs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.link_translational_vels = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        
        self.proj_link_translational_vecs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.proj_link_translational_vels = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        
        ''' Link SE(3) Accs '''
        if self.dim == 2:
            self.link_rotational_accs = ti.Vector.field(1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        else:
            # x, y, z rotations 
            self.link_rotational_accs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.link_translational_accs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.proj_link_translational_accs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        
        ''' Losses '''
        self.transformation_penalty_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.transformation_acc_reg_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        self.goal_reaching_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True) # goal reaching loss #
        self.task_irr_goal_reaching_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True) # goal reaching loss #
        self.tot_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
        
        ''' Task related variables -- Goals '''
        self.goal_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=()) ## single gaol reaching
        
        
        ### TODO: initialize the rigid body states, vels, and accs ###
        ''' Task-related -- rigid body manipulations '''
        self.rigid_type = RIGID_SPHERE
        ## for the rigid body linear accs, angular accs, angular vels, linear vels, positions, and the orientations ##
        self.rigid_body_linear_forces = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid body forces ##
        self.rigid_body_torques = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_linear_accs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_angular_accs = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_linear_vels = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_angular_vels = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_linear_states = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_angular_states = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_inv_inertia = ti.field(dtype=ti.f32, shape=())
        self.rigid_body_goal_trans = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=())
        
        ##### Initialized during the __init__ process #####
        self.rigid_body_mass = ti.field(dtype=ti.f32, shape=()) ## rigid body mass ##
        self.rigid_body_mass[None] = 1.0
        self.rigid_body_center = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=()) ## rigid body center ##
        self.rigid_body_center[None] = ti.Vector([0.7, 0.3])
        self.rigid_body_radius = ti.field(dtype=ti.f32, shape=())
        self.rigid_body_radius[None] = 0.05 # ## 
        self.rigid_body_inertia = ti.Matrix.field(n=self.dim, m=self.dim, dtype=ti.f32, shape=()) ## rigid body inertia ##
        self.rigid_body_inertia[None] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) ## version 1 of the body inertia ##
        # self.rigid_body_inv_inertia[None] = ti.Matrix([[1.0, 0.0], [0.0, 1.0]]) ## 
        self.rigid_body_inv_inertia[None] = 1.0 / 50.0
        self.rigid_body_goal_trans[None] = ti.Vector([-0.3, -0.2])
        
        ## it seems that we can not always find a good way for enforcing the rigid body constraints ##
        ## 
        
        
        
        # self.rigid_type = RIGID_CUBE
        
        ### TODO: add a general SDF representation ###
        
        # self.rigid_type = RIGID_SDF ## rigid sdf ## # rigid sdf ## ## rigid sdf ##
        
        self.rigid_body_sdf_field = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))
        
        rigid_body_sdf_np = "softzoo/assets/102_obj.npy"
        rigid_body_sdf_np = os.path.join(CODE_ROOT_FOLDER, rigid_body_sdf_np)

        rigid_body_sdf_np = np.load(rigid_body_sdf_np, allow_pickle=True)
        self.rigid_body_sdf_field.from_numpy(rigid_body_sdf_np)
        
        
        # [0.28125  0.359375] [0.71875   0.5703125]

        rigid_body_pcs_fn = "softzoo/assets/102_obj_pts.npy"
        
        rigid_body_pcs_fn = os.path.join(CODE_ROOT_FOLDER, rigid_body_pcs_fn)

        rigid_body_pcs = np.load(rigid_body_pcs_fn, allow_pickle=True)
        minn_pcs, maxx_pcs = np.min(rigid_body_pcs, axis=0), np.max(rigid_body_pcs, axis=0)
        center_pcs = (minn_pcs + maxx_pcs) / 2.0
        extent_pcs = maxx_pcs - minn_pcs
        extent_pcs = np.max(extent_pcs)
        scale_pcs = 0.15 / extent_pcs
        # for each pos; ( rigid_rot_inv @ (pos - (rigid_center + rigid_trans)) ) / scale + center_pcs -> use that to query the sdf values
        self.rigid_body_center_pcs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=())
        self.rigid_body_center_pcs[None] = ti.Vector(center_pcs.tolist())
        self.rigid_body_scale = ti.field(dtype=ti.f32, shape=())
        self.rigid_body_scale[None] = scale_pcs
        
        nn_rigid_pcs = rigid_body_pcs.shape[0]
        self.nn_rigid_pcs = nn_rigid_pcs
        self.rigid_pcs_input = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(nn_rigid_pcs))
        self.rigid_pcs_input.from_numpy(rigid_body_pcs)
        self.rigid_pcs_transform = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(nn_rigid_pcs))
        
        self.pos_sdf_values = ti.field(dtype=ti.f32, shape=())
          
        
        ''' Contact-related penalty coefficients '''
        self.contact_depth_coef = ti.field(dtype=ti.f32, shape=())
        self.contact_damping_coef = ti.field(dtype=ti.f32, shape=())
        self.contact_depth_coef[None] = self.cfg.sim.contact_spring_d
        self.contact_damping_coef[None] =  self.cfg.sim.contact_damping_d
        
        
        self.vis = self.cfg.general.vis
        # print(self.vis: )
        print(f"vis: {self.vis}")
        
        if self.vis:
            # self.vis = vis
            self.gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
        # else:
        #     self.vis = False
    
    ## get inference for the processing technique? ##
    ## get inference 
    def set_graph_connectivity(self, graph_connectivity_arr):
        self.graph.set_graph_connectivity(graph_connectivity_arr)
    
    
    
    @ti.kernel
    def set_goal(self, goal_pos_x: ti.f32, goal_pos_y: ti.f32):
        self.goal_pos[None] = ti.Vector([goal_pos_x, goal_pos_y])
        
        
    @ti.kernel
    def initialize(self,): # 
        
        # print(f"self.ori_particles_xs: {self.ori_particles_xs.shape}")
        # expanded_ori_particles_xs = self.ori_particles_xs[None, :, :]
        # expanded_ori_particles_xs = np.repeat(expanded_ori_particles_xs, self.nn_timesteps, axis=0)
        # self.particle_xs.from_numpy(self.expanded_ori_particles_xs) # 
        
        # self.particle_link_idx.from_numpy(self.particle_link_idxes) # 
        
        # if self.dim == 2:
        self.link_rotational_vecs.fill(0.0)
        self.link_rotational_vels.fill(0.0)
        self.link_rotational_mtx.fill(ti.Matrix([[1.0, 0.0], [0.0, 1.0]]))
        self.link_translational_vecs.fill(0.0)
        self.link_translational_vels.fill(0.0)
        
        self.link_rotational_accs.fill(0.0)
        self.link_translational_accs.fill(0.0)
        self.proj_link_translational_accs.fill(0.0)
        self.proj_link_translational_vels.fill(0.0)
        self.proj_link_translational_vecs.fill(0.0)
        
        self.link_rotational_vecs.grad.fill(0.0)
        self.link_rotational_vels.grad.fill(0.0)
        self.link_rotational_mtx.grad.fill(0.0)
        self.link_translational_vecs.grad.fill(0.0)
        self.link_translational_vels.grad.fill(0.0)
        self.link_rotational_accs.grad.fill(0.0)
        self.link_translational_accs.grad.fill(0.0)
        self.proj_link_translational_accs.grad.fill(0.0) ## proj link translational ####
        
        self.proj_link_translational_vels.fill(0.0)
        self.proj_link_translational_vecs.fill(0.0)
        self.proj_link_translational_vels.grad.fill(0.0)
        self.proj_link_translational_vecs.grad.fill(0.0)
        # else:
        #     raise NotImplementedError    
        
        ##### particle_vels, particle_accs #####
        self.particle_vels.fill(0.0)
        self.particle_accs.fill(0.0)
        self.particle_vels.grad.fill(0.0)
        self.particle_accs.grad.fill(0.0)
        
        ''' clear grads '''
        self.particle_xs.grad.fill(0.0)
        
        ''' Loss '''
        self.transformation_penalty_loss.fill(0.0)
        self.transformation_acc_reg_loss.fill(0.0)
        self.goal_reaching_loss.fill(9999999.0)
        self.task_irr_goal_reaching_loss.fill(9999999.0)
        self.tot_loss.fill(0.0)
        
        self.transformation_penalty_loss.grad.fill(0.0)
        self.transformation_acc_reg_loss.grad.fill(0.0)
        self.goal_reaching_loss.grad.fill(0.0) # 
        self.task_irr_goal_reaching_loss.grad.fill(0.0)
        self.tot_loss.grad.fill(0.0)
        
        ''' Rigid body related states, vels, and accs '''
        self.rigid_body_linear_forces.fill(0.0)
        self.rigid_body_linear_vels.fill(0.0)
        self.rigid_body_linear_states.fill(0.0)
        self.rigid_body_angular_states.fill(0.0)
        self.rigid_body_torques.fill(0.0)
        self.rigid_body_linear_accs.fill(0.0)
        self.rigid_body_angular_accs.fill(0.0)
        self.rigid_body_angular_vels.fill(0.0)
        
        self.rigid_body_linear_forces.grad.fill(0.0)
        self.rigid_body_linear_vels.grad.fill(0.0)
        self.rigid_body_linear_states.grad.fill(0.0)
        self.rigid_body_angular_states.grad.fill(0.0)
        self.rigid_body_torques.grad.fill(0.0)
        self.rigid_body_linear_accs.grad.fill(0.0)
        self.rigid_body_angular_accs.grad.fill(0.0)
        self.rigid_body_angular_vels.grad.fill(0.0)
        
        pass
    
    
    def load_optimized_particle_accs(self, particle_init_xs_np, particle_accs_np):
        
        #### scale the particle acs ### 
        valid_manip_statistic_info_fn = "/data/xueyi/softzoo/expv4/valid_data_statistics_taskerrthres0.03_transconsthres0.01.npy"
        valid_manip_statistics = np.load(valid_manip_statistic_info_fn, allow_pickle=True).item()
        self.avg_particle_init_xs = valid_manip_statistics['avg_particle_init_xs']
        self.std_particle_init_xs = valid_manip_statistics['std_particle_init_xs']
        self.avg_particle_accs = valid_manip_statistics['avg_particle_accs']
        self.std_particle_accs = valid_manip_statistics['std_particle_accs'] ## 
        
        ###### inversely scale the data ######
        # (init_pos - self.avg_particle_init_xs[None]) / (self.std_particle_init_xs[None] + 1e-6)
        particle_init_xs_np = (particle_init_xs_np * (self.std_particle_init_xs[None] + 1e-6)) + self.avg_particle_init_xs[None]
        avg_particle_init_xs = np.mean(particle_init_xs_np, axis=0)
        print(f"avg_particle_init_xs: {avg_particle_init_xs}, particle_accs_np: {particle_accs_np.shape}")
        # particle_acts = (particle_acts - self.avg_particle_accs[None]) / (self.std_particle_accs[None] + 1e-6)
        particle_accs_np = (particle_accs_np * (self.std_particle_accs[None] + 1e-6)) + self.avg_particle_accs[None] ## 
        ### get the particle accs and the particle init xs ####
        ### ### 
        nn_particles = particle_accs_np.shape[0]
        particle_accs_np = particle_accs_np.reshape(nn_particles, self.nn_timesteps, -1)
        particle_accs_np = np.transpose(particle_accs_np, (1, 0, 2))
        
        expanded_particle_init_xs = particle_init_xs_np[None, :, :]
        
        # expanded_ori_particles_xs = expanded_particle_init_xs[None]
        expanded_ori_particles_xs = np.repeat(expanded_particle_init_xs, self.nn_timesteps, axis=0)
        
        ### get the particle xs and the particle accs ###
        self.particle_xs.from_numpy(expanded_ori_particles_xs)
        self.particle_accs.from_numpy(particle_accs_np)
        # 
    # 
    
    @ti.kernel
    def initialize_grads(self, ):
        self.particle_xs.grad.fill(0.0)
        self.particle_vels.grad.fill(0.0)
        self.particle_accs.grad.fill(0.0) ## get partciel accs and the vels ###
        
        self.link_rotational_vecs.fill(0.0)
        self.link_rotational_vels.fill(0.0)
        self.link_translational_vecs.fill(0.0)
        self.link_translational_vels.fill(0.0)
        
        
        
        
        self.link_rotational_vecs.grad.fill(0.0)
        self.link_rotational_vels.grad.fill(0.0)
        self.link_rotational_mtx.grad.fill(0.0)
        self.link_translational_vecs.grad.fill(0.0)
        self.link_translational_vels.grad.fill(0.0)
        self.link_rotational_accs.grad.fill(0.0)
        self.link_translational_accs.grad.fill(0.0)
        self.proj_link_translational_accs.grad.fill(0.0) ## get the proj link tarnslational ##

        self.proj_link_translational_accs.fill(0.0)
        
        self.proj_link_translational_vels.fill(0.0)
        self.proj_link_translational_vecs.fill(0.0)
        self.proj_link_translational_vels.grad.fill(0.0)
        self.proj_link_translational_vecs.grad.fill(0.0)
 
        self.transformation_penalty_loss.grad.fill(0.0)
        self.transformation_acc_reg_loss.grad.fill(0.0)
        self.goal_reaching_loss.grad.fill(0.0)
        self.task_irr_goal_reaching_loss.grad.fill(0.0)
        
        self.tot_loss.grad.fill(0.0)
        
        
        ''' For the passvie objects, the states, vels, and accs together with their grads should be set to zeros '''
        self.rigid_body_linear_forces.fill(0.0)
        self.rigid_body_linear_vels.fill(0.0)
        self.rigid_body_linear_states.fill(0.0)
        self.rigid_body_angular_states.fill(0.0)
        self.rigid_body_torques.fill(0.0)
        self.rigid_body_linear_accs.fill(0.0)
        self.rigid_body_angular_accs.fill(0.0)
        self.rigid_body_angular_vels.fill(0.0)
        
        
        
        self.rigid_body_linear_forces.grad.fill(0.0)
        self.rigid_body_linear_vels.grad.fill(0.0)
        self.rigid_body_linear_states.grad.fill(0.0)
        self.rigid_body_angular_states.grad.fill(0.0)
        self.rigid_body_torques.grad.fill(0.0)
        self.rigid_body_linear_accs.grad.fill(0.0)
        self.rigid_body_angular_accs.grad.fill(0.0)
        self.rigid_body_angular_vels.grad.fill(0.0)
        
    
    
    @ti.kernel
    def initialize_loss(self, ):
        self.transformation_penalty_loss.fill(0.0)
        self.transformation_acc_reg_loss.fill(0.0)
        self.goal_reaching_loss.fill(9999999.0)
        self.task_irr_goal_reaching_loss.fill(9999999.0)
        self.tot_loss.fill(0.0)
        
        self.transformation_penalty_loss.grad.fill(0.0)
        self.transformation_acc_reg_loss.grad.fill(0.0)
        self.goal_reaching_loss.grad.fill(0.0)
        self.tot_loss.grad.fill(0.0)
        self.task_irr_goal_reaching_loss.grad.fill(0.0)
    
    ## 
    ## get link transformation penalties ##
    
    ## use the number of samled pts as the input, actuate such points -- - useing the point accs and at last conduct the foreard timestepping for getting the results ##
    
    
    @ti.kernel
    def proj_link_transformation_matrices_per_link(self, s: ti.i32, i_link: ti.i32):
        for j_link in range(i_link):
            # j_link_rotational_mtx_dot = self.link_rotational_dot_mtx[s, j_link]
            j_link_rotational_vec = self.link_rotational_vecs[s, j_link]
            j_link_rotational_vel = self.link_rotational_vels[s, j_link] ## get the rotational velocity #
            j_link_rotational_acc = self.link_rotational_accs[s, j_link] ## get he link rotational acc

            j_link_rotational_mtx_ddot = ti.Matrix([[-ti.sin(j_link_rotational_vec[0]), -ti.cos(j_link_rotational_vec[0])], [ti.cos(j_link_rotational_vec[0]), -ti.sin(j_link_rotational_vec[0])]]) * j_link_rotational_acc[0] + ti.Matrix([[-ti.cos(j_link_rotational_vec[0]), ti.sin(j_link_rotational_vec[0])], [-ti.sin(j_link_rotational_vec[0]), -ti.cos(j_link_rotational_vec[0])]]) * j_link_rotational_vel[0] * j_link_rotational_vel[0]


            i_link_rotational_vec = self.link_rotational_vecs[s, i_link]
            i_link_rotational_vel = self.link_rotational_vels[s, i_link] ## get the rotational velocity #
            i_link_rotational_acc = self.link_rotational_accs[s, i_link] ## get he link rotational acc

            i_link_rotational_mtx_ddot = ti.Matrix([[-ti.sin(i_link_rotational_vec[0]), -ti.cos(i_link_rotational_vec[0])], [ti.cos(i_link_rotational_vec[0]), -ti.sin(i_link_rotational_vec[0])]]) * i_link_rotational_acc[0] + ti.Matrix([[-ti.cos(i_link_rotational_vec[0]), ti.sin(i_link_rotational_vec[0])], [-ti.sin(i_link_rotational_vec[0]), -ti.cos(i_link_rotational_vec[0])]]) * i_link_rotational_vel[0] * i_link_rotational_vel[0]
   
            j_link_translational_acc = self.link_translational_accs[s, j_link] 
            # j_to_c_translational_acc = (j_link_rotational_mtx_ddot - i_link_rotational_mtx_ddot) @ self.link_joint_pos[j_link] + j_link_translational_acc
            j_to_c_translational_acc = (j_link_rotational_mtx_ddot - i_link_rotational_mtx_ddot) @ self.link_joint_pos[i_link] + j_link_translational_acc
            if self.graph.graph_A[j_link, i_link] > 0.1:
                self.proj_link_translational_accs[s, i_link] += j_to_c_translational_acc * self.graph.graph_A[j_link, i_link] ## accumulate the proj _translation accs ##
            
    @ti.kernel
    def proj_link_transformation_matrices_self_link(self, s: ti.i32, i_link: ti.i32):
        i_link_translational_accs = self.link_translational_accs[s, i_link]
        self.proj_link_translational_accs[s, i_link] += self.graph.graph_A[i_link, i_link] * i_link_translational_accs
    
    @ti.kernel
    def pass_proj_link_transformation_matrices(self, s: ti.i32, i_link: ti.i32):
        self.link_translational_accs[s, i_link] = self.proj_link_translational_accs[s, i_link]
        
        
    @ti.kernel
    def proj_link_transformation_matrices_per_link_vels(self, s: ti.i32, i_link: ti.i32):
        for j_link in range(i_link):
            # j_link_rotational_mtx_dot = self.link_rotational_dot_mtx[s, j_link]
            j_link_rotational_vec = self.link_rotational_vecs[s, j_link]
            j_link_rotational_vel = self.link_rotational_vels[s, j_link] ## get the rotational velocity #
            # j_link_rotational_acc = self.link_rotational_accs[s, j_link] ## get he link rotational acc

            j_link_rotational_mtx_dot = ti.Matrix([[-ti.sin(j_link_rotational_vec[0]), -ti.cos(j_link_rotational_vec[0])], [ti.cos(j_link_rotational_vec[0]), -ti.sin(j_link_rotational_vec[0])]]) * j_link_rotational_vel[0] # + ti.Matrix([[-ti.cos(j_link_rotational_vec[0]), ti.sin(j_link_rotational_vec[0])], [-ti.sin(j_link_rotational_vec[0]), -ti.cos(j_link_rotational_vec[0])]]) * j_link_rotational_vel[0] * j_link_rotational_vel[0]

        
            i_link_rotational_vec = self.link_rotational_vecs[s, i_link]
            i_link_rotational_vel = self.link_rotational_vels[s, i_link] ## get the rotational velocity #
            # i_link_rotational_acc = self.link_rotational_accs[s, i_link] ## get he link rotational acc

            i_link_rotational_mtx_dot = ti.Matrix([[-ti.sin(i_link_rotational_vec[0]), -ti.cos(i_link_rotational_vec[0])], [ti.cos(i_link_rotational_vec[0]), -ti.sin(i_link_rotational_vec[0])]]) * i_link_rotational_vel[0]  # + ti.Matrix([[-ti.cos(i_link_rotational_vec[0]), ti.sin(i_link_rotational_vec[0])], [-ti.sin(i_link_rotational_vec[0]), -ti.cos(i_link_rotational_vec[0])]]) * i_link_rotational_vel[0] * i_link_rotational_vel[0]
   
            j_link_translational_vel = self.link_translational_vels[s, j_link] 
            # j_to_c_translational_acc = (j_link_rotational_mtx_ddot - i_link_rotational_mtx_ddot) @ self.link_joint_pos[j_link] + j_link_translational_acc
            j_to_c_translational_vel = (j_link_rotational_mtx_dot - i_link_rotational_mtx_dot) @ self.link_joint_pos[i_link] + j_link_translational_vel
            self.proj_link_translational_vels[s, i_link] += j_to_c_translational_vel * self.graph.graph_A[j_link, i_link] ## accumulate the proj _translation accs ##
            
    @ti.kernel
    def proj_link_transformation_matrices_self_link_vels(self, s: ti.i32, i_link: ti.i32):
        # i_link_translational_accs = self.link_translational_accs[s, i_link]
        # self.proj_link_translational_accs[s, i_link] += self.graph.graph_A[i_link, i_link] * i_link_translational_accs
    
        # i_link_translational_vels = self.link_translational_vels[s, i_link]
        self.proj_link_translational_vels[s, i_link] += self.graph.graph_A[i_link, i_link] * self.link_translational_vels[s, i_link]
        # self.projlinktr
    
    @ti.kernel
    def pass_proj_link_transformation_matrices_vels(self, s: ti.i32, i_link: ti.i32):
        # self.link_translational_accs[s, i_link] = self.proj_link_translational_accs[s, i_link]
        self.link_translational_vels[s, i_link] = self.proj_link_translational_vels[s, i_link]
        
        
    @ti.kernel
    def proj_link_transformation_matrices_per_link_vecs(self, s: ti.i32, i_link: ti.i32):
        for j_link in range(i_link):
            # j_link_rotational_mtx_dot = self.link_rotational_dot_mtx[s, j_link]
            j_link_rotational_vec = self.link_rotational_vecs[s, j_link]
            # j_link_rotational_vel = self.link_rotational_vels[s, j_link] ## get the rotational velocity #
            # j_link_rotational_acc = self.link_rotational_accs[s, j_link] ## get he link rotational acc

            j_link_rotational_mtx = ti.Matrix([[ti.cos(j_link_rotational_vec[0]), -ti.sin(j_link_rotational_vec[0])], [ti.sin(j_link_rotational_vec[0]), ti.cos(j_link_rotational_vec[0])]])  # * j_link_rotational_vel[0] # + ti.Matrix([[-ti.cos(j_link_rotational_vec[0]), ti.sin(j_link_rotational_vec[0])], [-ti.sin(j_link_rotational_vec[0]), -ti.cos(j_link_rotational_vec[0])]]) * j_link_rotational_vel[0] * j_link_rotational_vel[0]

        
            i_link_rotational_vec = self.link_rotational_vecs[s, i_link]
            # i_link_rotational_vel = self.link_rotational_vels[s, i_link] ## get the rotational velocity #
            # i_link_rotational_acc = self.link_rotational_accs[s, i_link] ## get he link rotational acc

            i_link_rotational_mtx = ti.Matrix([[ti.cos(i_link_rotational_vec[0]), -ti.sin(i_link_rotational_vec[0])], [ti.sin(i_link_rotational_vec[0]), ti.cos(i_link_rotational_vec[0])]])    # + ti.Matrix([[-ti.cos(i_link_rotational_vec[0]), ti.sin(i_link_rotational_vec[0])], [-ti.sin(i_link_rotational_vec[0]), -ti.cos(i_link_rotational_vec[0])]]) * i_link_rotational_vel[0] * i_link_rotational_vel[0]
   
            j_link_translational_vec = self.link_translational_vecs[s, j_link] 
            # j_to_c_translational_acc = (j_link_rotational_mtx_ddot - i_link_rotational_mtx_ddot) @ self.link_joint_pos[j_link] + j_link_translational_acc
            j_to_c_translational_vec = (j_link_rotational_mtx - i_link_rotational_mtx) @ self.link_joint_pos[i_link] + j_link_translational_vec
            if self.graph.graph_A[j_link, i_link] > 0.1:
                self.proj_link_translational_vecs[s, i_link] += j_to_c_translational_vec * self.graph.graph_A[j_link, i_link] ## accumulate the proj _translation accs ##
            
    @ti.kernel
    def proj_link_transformation_matrices_self_link_vecs(self, s: ti.i32, i_link: ti.i32):
        # i_link_translational_accs = self.link_translational_accs[s, i_link]
        # self.proj_link_translational_accs[s, i_link] += self.graph.graph_A[i_link, i_link] * i_link_translational_accs
    
        # i_link_translational_vecs = self.link_translational_vecs[s, i_link]
        self.proj_link_translational_vecs[s, i_link] += self.graph.graph_A[i_link, i_link] * self.link_translational_vecs[s, i_link]
        # self.projlinktr
    
    @ti.kernel
    def pass_proj_link_transformation_matrices_vecs(self, s: ti.i32, i_link: ti.i32):
        # self.link_translational_accs[s, i_link] = self.proj_link_translational_accs[s, i_link]
        self.link_translational_vecs[s, i_link] = self.proj_link_translational_vecs[s, i_link]
            
    
    
    def project_link_transformation_matrices(self, s: ti.i32):
        # for i_link in 
        ## ## tarns accs ## 
        for i_link in range(self.nn_links):
            self.proj_link_transformation_matrices_self_link(s, i_link)
            self.proj_link_transformation_matrices_per_link(s, i_link)
            self.pass_proj_link_transformation_matrices(s, i_link)
            
            # pass
            
    def project_link_transformation_matrices_backward(self, s: ti.i32):
        for i_link in range(self.nn_links - 1, -1, -1):
            self.pass_proj_link_transformation_matrices.grad(s, i_link)
            self.proj_link_transformation_matrices_per_link.grad(s, i_link)
            self.proj_link_transformation_matrices_self_link.grad(s, i_link)
            
    
    def project_link_transformation_matrices_vels(self, s: ti.i32):
        # for i_link in 
        ## ## tarns accs ## 
        for i_link in range(self.nn_links):
            self.proj_link_transformation_matrices_self_link_vels(s, i_link)
            self.proj_link_transformation_matrices_per_link_vels(s, i_link)
            self.pass_proj_link_transformation_matrices_vels(s, i_link)
            
            # pass
            
    def project_link_transformation_matrices_vels_backward(self, s: ti.i32):
        for i_link in range(self.nn_links - 1, -1, -1):
            self.pass_proj_link_transformation_matrices_vels.grad(s, i_link)
            self.proj_link_transformation_matrices_per_link_vels.grad(s, i_link)
            self.proj_link_transformation_matrices_self_link_vels.grad(s, i_link)
            
    
    def project_link_transformation_matrices_vecs(self, s: ti.i32):
        # for i_link in 
        ## ## tarns accs ## 
        for i_link in range(self.nn_links):
            self.proj_link_transformation_matrices_self_link_vecs(s, i_link)
            self.proj_link_transformation_matrices_per_link_vecs(s, i_link)
            self.pass_proj_link_transformation_matrices_vecs(s, i_link)
            
            # pass
    
    ## goal reaching ##
    def project_link_transformation_matrices_vecs_backward(self, s: ti.i32):
        for i_link in range(self.nn_links - 1, -1, -1):
            self.pass_proj_link_transformation_matrices_vecs.grad(s, i_link)
            self.proj_link_transformation_matrices_per_link_vecs.grad(s, i_link)
            self.proj_link_transformation_matrices_self_link_vecs.grad(s, i_link)
            
        # pass
    
    @ti.kernel
    def get_rotational_transformation_matrices(self, s: ti.i32):
        for link in range(self.nn_links):
            if s > 0:
                rotational_vels = self.link_rotational_vels[s - 1, link] + self.dt * self.link_rotational_accs[s, link]
                rotational_vecs = self.link_rotational_vecs[s - 1, link] + self.dt * self.link_rotational_vels[s - 1, link] + self.dt * self.dt * self.link_rotational_accs[s, link]
                self.link_rotational_vels[s, link] = rotational_vels
                self.link_rotational_vecs[s, link] = rotational_vecs
                # self.link_rotational_mtx[s, link] = ti.Matrix([[ti.cos(rotational_vecs), -ti.sin(rotational_vecs)], [ti.sin(rotational_vecs), ti.cos(rotational_vecs)]])
                self.link_rotational_mtx[s, link] = ti.Matrix([[ti.cos(rotational_vecs[0]), -ti.sin(rotational_vecs[0])], [ti.sin(rotational_vecs[0]), ti.cos(rotational_vecs[0])]])
                
                self.link_rotational_dot_mtx[s, link] = ti.Matrix([[-ti.sin(rotational_vecs[0]), -ti.cos(rotational_vecs[0])], [ti.cos(rotational_vecs[0]), -ti.sin(rotational_vecs[0])]]) * rotational_vels[0]
                

    ## get the translational transformation matrices ## ## get the translational matrices ## ## ## translational matrices ## ## translatonal matrices ## 
    @ti.kernel
    def get_translational_transformation_matrices(self, s: ti.i32):
        for link in range(self.nn_links):

            # if s > 0:
            # rotational_vels = self.link_rotational_vels[s - 1, link] + self.dt * self.link_rotational_accs[s, link]
            # rotational_vecs = self.link_rotational_vecs[s - 1, link] + self.dt * self.link_rotational_vels[s - 1, link] + self.dt * self.dt * self.link_rotational_accs[s, link]
            # self.link_rotational_vels[s, link] = rotational_vels
            # self.link_rotational_vecs[s, link] = rotational_vecs
            # # self.link_rotational_mtx[s, link] = ti.Matrix([[ti.cos(rotational_vecs), -ti.sin(rotational_vecs)], [ti.sin(rotational_vecs), ti.cos(rotational_vecs)]])
            # self.link_rotational_mtx[s, link] = ti.Matrix([[ti.cos(rotational_vecs[0]), -ti.sin(rotational_vecs[0])], [ti.sin(rotational_vecs[0]), ti.cos(rotational_vecs[0])]])
            
            # self.link_rotational_dot_mtx[s, link] = ti.Matrix([[-ti.sin(rotational_vecs[0]), -ti.cos(rotational_vecs[0])], [ti.cos(rotational_vecs[0]), -ti.sin(rotational_vecs[0])]]) * rotational_vels[0]
            
            translational_vels = self.link_translational_vels[s - 1, link] + self.dt * self.link_translational_accs[s, link]
            translational_vecs = self.link_translational_vecs[s - 1, link] + self.dt * self.link_translational_vels[s - 1, link] + self.dt * self.dt * self.link_translational_accs[s, link]
            self.link_translational_vels[s, link] = translational_vels
            self.link_translational_vecs[s, link] = translational_vecs

    
    
    @ti.kernel
    def get_transformation_matrices(self, s: ti.i32):
        # v_{t + 1} = v_{t} + dt * a_{t + 1}
        # \delta x_{t + 1} = x_{t} + dt * v_{t + 1}
        # \delta x_{t + 1} = x_{t} + dt * v_{t} + dt * dt * a_{t + 1}
        for link in range(self.nn_links):
            if s > 0:
                rotational_vels = self.link_rotational_vels[s - 1, link] + self.dt * self.link_rotational_accs[s, link]
                rotational_vecs = self.link_rotational_vecs[s - 1, link] + self.dt * self.link_rotational_vels[s - 1, link] + self.dt * self.dt * self.link_rotational_accs[s, link]
                self.link_rotational_vels[s, link] = rotational_vels
                self.link_rotational_vecs[s, link] = rotational_vecs
                # self.link_rotational_mtx[s, link] = ti.Matrix([[ti.cos(rotational_vecs), -ti.sin(rotational_vecs)], [ti.sin(rotational_vecs), ti.cos(rotational_vecs)]])
                self.link_rotational_mtx[s, link] = ti.Matrix([[ti.cos(rotational_vecs[0]), -ti.sin(rotational_vecs[0])], [ti.sin(rotational_vecs[0]), ti.cos(rotational_vecs[0])]])
                
                self.link_rotational_dot_mtx[s, link] = ti.Matrix([[-ti.sin(rotational_vecs[0]), -ti.cos(rotational_vecs[0])], [ti.cos(rotational_vecs[0]), -ti.sin(rotational_vecs[0])]]) * rotational_vels[0]
                
                # 
                translational_vels = self.link_translational_vels[s - 1, link] + self.dt * self.link_translational_accs[s, link]
                translational_vecs = self.link_translational_vecs[s - 1, link] + self.dt * self.link_translational_vels[s - 1, link] + self.dt * self.dt * self.link_translational_accs[s, link]
                self.link_translational_vels[s, link] = translational_vels
                self.link_translational_vecs[s, link] = translational_vecs
                
    def get_transformation_matrices_wproj(self, s: ti.i32):

        
        if s > 0:
            self.get_rotational_transformation_matrices(s)
        
        self.project_link_transformation_matrices(s)
        
        if s > 0:
            self.get_translational_transformation_matrices(s)
        
        
        self.project_link_transformation_matrices_vels(s)
        self.project_link_transformation_matrices_vecs(s)
        
        # link_translational_accs_tot = np.sum(self.link_translational_accs.to_numpy())
        # print(f"link_translational_accs_tot: {link_translational_accs_tot}")
        
        
        
    
    def get_transformation_matrices_wproj_grad(self, s: ti.i32):
        
        
        self.project_link_transformation_matrices_vecs_backward(s)
        
        self.project_link_transformation_matrices_vels_backward(s)
        
        if s > 0:
            self.get_translational_transformation_matrices.grad(s)
        
        self.project_link_transformation_matrices_backward(s)
        
        if s > 0:
            self.get_rotational_transformation_matrices.grad(s)

    
    @ti.kernel
    def transform_particles(self, s: ti.i32):
        for p in range(self.nn_particles):
            ori_particle_pos = self.particle_xs[0, p]
            link_idx = self.particle_link_idx[p]
            link_rot_mtx = self.link_rotational_mtx[s, link_idx]
            link_trans_vec = self.link_translational_vecs[s, link_idx]
            cur_particle_pos = link_rot_mtx @ ori_particle_pos + link_trans_vec
            self.particle_xs[s, p] = cur_particle_pos
            
            link_rot_mtx_dot = self.link_rotational_dot_mtx[s, link_idx]
            link_trans_vel = self.link_translational_vels[s, link_idx]
            cur_particle_vel = link_rot_mtx_dot @ ori_particle_pos + link_trans_vel
            
            self.particle_vels[s, p] = cur_particle_vel
    
    @ti.func
    def get_sdf_sphere(self, pos):
        rel_center_to_pos = pos  # - self.rigid_body_center[None]
        sqr_rel_center_to_pos = rel_center_to_pos ** 2
        sqr_rel_center_to_pos_all = sqr_rel_center_to_pos[0] + sqr_rel_center_to_pos[1]
        dist_center_to_pos = ti.sqrt(sqr_rel_center_to_pos_all)
        pos_sdf = dist_center_to_pos - self.rigid_body_radius[None]
        return pos_sdf
        
    @ti.func
    def get_sdf_cube(self, pos):
        rel_center_to_pos = pos # - self.rigid_body_center[None]
        abs_delta_x = ti.abs(rel_center_to_pos[0])
        abs_delta_y = ti.abs(rel_center_to_pos[1])
        sdf_x = abs_delta_x - self.rigid_body_radius[None]
        sdf_y = abs_delta_y - self.rigid_body_radius[None]
        pos_sdf = ti.max(sdf_x, sdf_y)
        return pos_sdf

    @ti.func
    def get_sdf_rigid_sdf(self, pos):
        # rel_center_to_pos = pos - (self.rigid_body_center[None] + self.)
        scaled_pos = (pos / self.rigid_body_scale[None]) + self.rigid_body_center_pcs[None]
        scaled_pos_grid = (scaled_pos * n_grid).cast(ti.i32)
        scaled_pos_grid = ti.Vector([ti.max(0, ti.min(scaled_pos_grid[0], n_grid - 1)), ti.max(0, ti.min(scaled_pos_grid[1], n_grid - 1))])
        pos_sdf = self.rigid_body_sdf_field[scaled_pos_grid[0], scaled_pos_grid[1]]
        return pos_sdf
        

    @ti.func
    def get_sdf(self, pos):
        
        if self.rigid_type == RIGID_SPHERE:
            self.pos_sdf_values[None] = self.get_sdf_sphere(pos=pos)
            # return self.get_sdf_sphere(pos=pos)
        elif self.rigid_type == RIGID_CUBE:
            # return self.get_sdf_cube(pos=pos)
            self.pos_sdf_values[None] = self.get_sdf_cube(pos=pos)
        elif self.rigid_type == RIGID_SDF:
            self.pos_sdf_values[None] = self.get_sdf_rigid_sdf(pos=pos)
            # return self.get_sdf_rigid_sdf(pos=pos)
        return self.pos_sdf_values[None]
        # return dist_center_to_pos - self.rigid_body_radius[None]
        # return pos.norm() - self.rigid_body_radius[None]
    
    
    ### TODO: this function should only be called when s > 0 ###
    ### TODO: note that sometimes (seemingly so) the judggement s > 0 cannot be well processed ###
    @ti.kernel
    def calculate_contact_forces(self, s: ti.i32):
        for p in range(self.nn_particles):
            cur_particle_pos = self.particle_xs[s, p]

            cur_particle_vel = self.particle_vels[s, p] ## get the particle velocity ##
     
            
            # if s > 0:
            prev_rigid_body_angular_state = self.rigid_body_angular_states[s - 1][0]
            prev_rigid_body_rot_mtx = ti.Matrix([[ti.cos(prev_rigid_body_angular_state), -ti.sin(prev_rigid_body_angular_state)], [ti.sin(prev_rigid_body_angular_state), ti.cos(prev_rigid_body_angular_state)]])
            prev_rigid_body_linear_state = self.rigid_body_linear_states[s - 1] ## lnear state

            
            inversely_transformed_pos = prev_rigid_body_rot_mtx.transpose() @ (cur_particle_pos - prev_rigid_body_linear_state - self.rigid_body_center[None])
            
            # cur_particle_sdf = self.get_sdf(cur_particle_pos) ## get the particle positions ##
            
            ###### Calculate the sdf inline ######
            # rel_center_to_pos = cur_particle_pos - self.rigid_body_center[None] ## cetner to the pos ##
            # sqr_rel_center_to_pos = rel_center_to_pos ** 2
            # sqr_rel_center_to_pos_all = sqr_rel_center_to_pos[0] + sqr_rel_center_to_pos[1]
            # dist_center_to_pos = ti.sqrt(sqr_rel_center_to_pos_all)
            # cur_particle_sdf = dist_center_to_pos - self.rigid_body_radius[None]
            ###### Calculate the sdf inline ######
            
            
            ###### Calculate the sdf via function ######
            # cur_particle_sdf = self.get_sdf(cur_particle_pos, s)
            cur_particle_sdf = self.get_sdf(inversely_transformed_pos)
            ###### Calculate the sdf via function ######
            
            
            cur_particle_sdf = ti.min(cur_particle_sdf, cur_particle_sdf - cur_particle_sdf)
            
            # contact_force
            # joint_contact_force = ti.Vector([0.0, 0.0])
            # joint_torque = ti.Vector([0.0])
            
            # cur_particle_sdf = min(cur_particle_sdf, 0.0)
            
            
            # ## calculate the penalty based forces --- the penetration depth component ##
            center_to_pos = inversely_transformed_pos # - self.rigid_body_center[None] ## 
            center_to_pos = center_to_pos.normalized() ## get the direction
            contact_depth_penalty_forces = cur_particle_sdf * center_to_pos * self.contact_depth_coef[None] ## k * sd(p) * penetration_depth_coef
            ## calcualte the damping force ##
            inversely_transformed_vel =  (prev_rigid_body_rot_mtx.transpose() @ cur_particle_vel) * (-1.0 * cur_particle_sdf)
            contact_damping_force = inversely_transformed_vel * self.contact_damping_coef[None] # a big issue --- contact modeling ... #
            
            contact_force = contact_depth_penalty_forces + contact_damping_force
            joint_contact_force = prev_rigid_body_rot_mtx @ contact_force
            
            
            transformed_center_to_pos = prev_rigid_body_rot_mtx @ inversely_transformed_pos # - self.rigid_body_center[None]
            joint_torque = ti.math.cross(transformed_center_to_pos, joint_contact_force)
                
            
            
            self.rigid_body_linear_forces[s] += joint_contact_force # / self.rigid_body_mass[None]
            self.rigid_body_torques[s] += joint_torque # / self.rigid_body_inertia[None] ### TODO: checkt this !!
            
            
            
            
    ## TODO: should only be called when s > 0 ##
    @ti.kernel
    def transform_rigid_body(self, s: ti.i32):
        cur_rigid_body_linear_force = self.rigid_body_linear_forces[s]
        cur_rigid_body_torque = self.rigid_body_torques[s]
        
        # rigid_body_inv_inertia #
        rigid_body_linear_acc = cur_rigid_body_linear_force / self.rigid_body_mass[None]
        rigid_body_angular_acc = self.rigid_body_inv_inertia[None] * cur_rigid_body_torque
        
        # get the accs and the torques 
        # if s > 0:
        rigid_body_linear_vel = self.rigid_body_linear_vels[s - 1] + self.dt * rigid_body_linear_acc
        rigid_body_angular_vel = self.rigid_body_angular_vels[s - 1] + self.dt * rigid_body_angular_acc
        rigid_body_linear_state = self.rigid_body_linear_states[s - 1] + self.dt * rigid_body_linear_vel
        rigid_body_angular_state = self.rigid_body_angular_states[s - 1] + self.dt * rigid_body_angular_vel
        self.rigid_body_linear_vels[s] = rigid_body_linear_vel
        self.rigid_body_angular_vels[s] = rigid_body_angular_vel
        self.rigid_body_linear_states[s] = rigid_body_linear_state
        self.rigid_body_angular_states[s] = rigid_body_angular_state
            
        
        
            
    ### TODO: transform the rigid object ###

    @ti.kernel
    def get_link_transformation_penalties(self, s: ti.i32):
        # tot_transformation_penalties = 0.0
        # pass
        # tot_penalties = []
        for link in range(1, self.nn_links):
            # if link in self.link_idx_to_joint_info:
            # if self.link_parent_idx[link] >= 0:
            # if link > 0:
                joint_pos = self.link_joint_pos[link]
                joint_dir = self.link_joint_dir[link]
                parent_link_idx = self.link_parent_idx[link] ## TODO: not sure whether we can use the link here to anchor the dict
                parent_rot_mtx = self.link_rotational_mtx[s, parent_link_idx]
                parent_trans_vec = self.link_translational_vecs[s, parent_link_idx]
                
                child_rot_mtx = self.link_rotational_mtx[s, link]
                child_trans_vec = self.link_translational_vecs[s, link] 
                
                # print(f"joint_pos: {joint_pos}, link: {link}")
                
                joint_pos_transformed_by_parent = parent_rot_mtx @ joint_pos + parent_trans_vec
                joint_pos_transformed_by_child = child_rot_mtx @ joint_pos + child_trans_vec
                
                
                # the boundary of the shape # # a function for defining the the boundary of the shape #
                diff_joint_pos = (joint_pos_transformed_by_child - joint_pos_transformed_by_parent) ** 2
                # print(f'diff_joint_pos: {diff_joint_pos}')
                diff_joint_pos_loss = diff_joint_pos[0] + diff_joint_pos[1]
                # print(f"diff_joint_pos_loss: {diff_joint_pos_loss}, link: {link}")
                # tot_penalties.append()
                self.transformation_penalty_loss[None] += diff_joint_pos_loss
        
    
    @ti.kernel
    def get_link_acc_reg_penalties(self, s: ti.i32):
        pass
        # for link in range(self.nn_links):
        #     cur_link_rot_acc = self.link_rotational_accs[s, link] # 
        #     cur_link_trans_acc = self.link_translational_accs[s, link]
        #     cur_link_rot_acc_reg = (cur_link_rot_acc ** 2).norm()
        #     cur_link_trans_acc_reg = (cur_link_trans_acc ** 2).norm() 
        #     self.transformation_acc_reg_loss[None] += (cur_link_rot_acc_reg + cur_link_trans_acc_reg)
    
    @ti.kernel
    ## get the goal reaching task objective ##
    def get_goal_reaching_loss(self, s: ti.i32):
        # for p in range(self.nn_particles):
        #     cur_particle_pos = self.particle_xs[s, p]
        #     goal_pos = self.goal_pos[None]
        #     cur_particle_goal_diff = (cur_particle_pos - goal_pos).norm() ** 2
            # self.goal_reaching_loss[None] = ti.min(cur_particle_goal_diff, self.goal_reaching_loss[None])
        # rigid_goal_trans = self.rigid_body_goal_trans[None]
        rigid_goal_trans = ti.Vector([-0.3, -0.2])
        rigid_trans = self.rigid_body_linear_states[s]
        rigid_trans_diff = (rigid_trans - rigid_goal_trans) ** 2
        
        rigid_trans_diff_all = rigid_trans_diff[0] + rigid_trans_diff[1]
        self.goal_reaching_loss[None] = rigid_trans_diff_all # goal reaching loss ##
        
    
    
    @ti.kernel
    def get_task_irre_goal_reaching_loss(self, s: ti.i32):
        for p in range(self.nn_particles):
            cur_particle_pos = self.particle_xs[s, p]
            goal_pos = self.rigid_body_center[None] # + self.rigid_body_linear_states[s]
            cur_particle_goal_diff = (cur_particle_pos - goal_pos).norm() ** 2
            self.task_irr_goal_reaching_loss[None] = ti.min(cur_particle_goal_diff, self.task_irr_goal_reaching_loss[None])
    
    @ti.kernel
    def get_goal_reaching_loss_backward(self, s: ti.i32):
        for p in range(self.nn_particles):
            cur_particle_pos = self.particle_xs[s, p]
            goal_pos = self.rigid_body_center[None]  # + self.rigid_body_linear_states[s]
            cur_particle_goal_diff = (cur_particle_pos - goal_pos).norm() ** 2 # loss = (x - y) ** 2; d loss = 2 * (x - y) * dx
            if ti.abs(cur_particle_goal_diff - self.task_irr_goal_reaching_loss[None]) < 1e-3:
            # if ti.abs(cur_particle_goal_diff - self.task_irr_goal_reaching_loss[None]) < 1e-2:
            # if ti.abs(cur_particle_goal_diff - self.task_irr_goal_reaching_loss[None]) < 1e-1:
                self.particle_xs.grad[s, p] += 2 * (cur_particle_pos - goal_pos) * self.task_irr_goal_reaching_loss.grad[None]
            # self.goal_reaching_loss[None] = ti.min(cur_particle_goal_diff, self.goal_reaching_loss[None])
    
    
    @ti.kernel
    def get_total_loss_v2(self):
        # self.tot_loss[None] += self.transformation_penalty_loss[None] * self.cur_transformation_penalty_term + self.transformation_acc_reg_loss[None] * 0.01 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]
        # if self.ii_iter < 2000:
        # self.tot_loss[None] += self.transformation_acc_reg_loss[None] * 0.01 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]
        # else:
        self.tot_loss[None] +=  self.goal_reaching_loss[None] #
    
    
    @ti.kernel
    def get_total_loss(self):
        self.tot_loss[None] += self.transformation_penalty_loss[None] * self.cur_transformation_penalty_term + self.transformation_acc_reg_loss[None] * 0.01 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]

        # self.tot_loss[None] += self.transformation_acc_reg_loss[None] * 0.01 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]
        ##### acc reg loss ##### ## actuate the acc reg losses in the dirversed spaces ######
        # self.tot_loss[None] += self.transformation_acc_reg_loss[None] * 10000.0 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]


    
    @ti.kernel
    def get_transformed_rigid_pcs(self, s: ti.i32):
        for p in range(self.nn_rigid_pcs):
        
            rigid_rot = self.rigid_body_angular_states[s][0]
            rigid_rot_mtx = ti.Matrix([[ti.cos(rigid_rot), -ti.sin(rigid_rot)], [ti.sin(rigid_rot), ti.cos(rigid_rot)]])

            rigid_pcs_center = self.rigid_body_center_pcs[None]
            cur_ori_pc = self.rigid_pcs_input[p]
            
            transformed_pc = rigid_rot_mtx @ ( (cur_ori_pc - rigid_pcs_center) * self.rigid_body_scale[None] ) + self.rigid_body_center[None] + self.rigid_body_linear_states[s]
            self.rigid_pcs_transform[p] = transformed_pc
        
        
    
    def forward_stepping(self,):
        for s in range(self.nn_timesteps):
            
            # self.get_transformation_matrices_wproj(s)
            self.get_transformation_matrices(s)
            
            self.transform_particles(s)
            
            if s > 0:
                self.calculate_contact_forces(s=s)
                self.transform_rigid_body(s=s)
            
            if self.vis:
                self.gui.circles(
                    self.particle_xs.to_numpy()[s],
                    radius=1.5, # 
                    # palette=[0x068587, 0xED553B, 0xEEEEF0],
                    # palette_indices=material,
                )
                
                if self.rigid_type == RIGID_SDF:
                    self.get_transformed_rigid_pcs(s)
                    self.gui.circles(
                        self.rigid_pcs_transform.to_numpy(),
                        radius=1.5,
                    )
                elif self.rigid_type == RIGID_CUBE:
                    self.gui.rect(self.rigid_body_center[None].to_numpy() + self.rigid_body_linear_states.to_numpy()[s], size=[self.rigid_body_radius[None] * 2, self.rigid_body_radius[None] * 2], color=0x068587)
                elif self.rigid_type == RIGID_SPHERE:
                    self.gui.circle(self.rigid_body_center[None].to_numpy() + self.rigid_body_linear_states.to_numpy()[s], radius=self.rigid_body_radius[None], color=0x068587)
                else:
                    raise NotImplementedError
                self.gui.show()
            
        
        for s in range(self.nn_timesteps):
            self.get_link_transformation_penalties(s) # link transformation penalties for each state
            self.get_link_acc_reg_penalties(s) #
        self.get_goal_reaching_loss(self.nn_timesteps - 1) # goal reaching
        self.get_task_irre_goal_reaching_loss(self.nn_timesteps - 1)
    
    
    def backward_stepping(self):
        self.get_goal_reaching_loss_backward(self.nn_timesteps - 1)
        self.get_goal_reaching_loss.grad(self.nn_timesteps - 1)
        #### check the rigid body trans grad ---- it has values ####
        # rigit_body_trans_grad=  np.sum(self.rigid_body_linear_states.grad.to_numpy())
        # print(f"rigit_body_trans_grad: {rigit_body_trans_grad}")
        #### check the rigid body trans grad ---- it has values ####
        # self.get_goal_reaching_loss_backward(self.nn_timesteps - 1)
        for s in range(self.nn_timesteps - 1, -1, -1):
            # self.get_goal_reaching_loss.grad(s)
            self.get_link_acc_reg_penalties.grad(s)
            self.get_link_transformation_penalties.grad(s)
            
        # particle_xs_grad = np.sum(self.particle_xs.grad.to_numpy())
        # print(f"After goal reaching loss backward: particle_xs_grad = {particle_xs_grad}")
        
        for s in range(self.nn_timesteps - 1, -1, -1):
            
            if s > 0:
                self.transform_rigid_body.grad(s=s)
                self.calculate_contact_forces.grad(s=s)
                
            
            self.transform_particles.grad(s)
            
            
            # rotational_accs_grad = np.sum(self.link_rotational_accs.grad.to_numpy())
            # translational_accs_grad = np.sum(self.link_translational_accs.grad.to_numpy())
            # rotation_mtx_grad = np.sum(self.link_rotational_mtx.grad.to_numpy())
            # translational_vecs_grad = np.sum(self.link_translational_vecs.grad.to_numpy())
            # print(f"After {s}-step's particles transformation backward: rotational_accs_grad = {rotational_accs_grad}, translational_accs_grad: {translational_accs_grad}, rotation_mtx_grad = {rotation_mtx_grad}, translational_vecs_grad = {translational_vecs_grad}")
            
        
            self.get_transformation_matrices.grad(s)
            # self.get_transformation_matrices_wproj_grad(s)
            
        
    
    @ti.kernel
    def update_accs_per_timestep(self, s: ti.i32):
        for link in range(self.nn_links):
            self.link_rotational_accs[s, link] = self.link_rotational_accs[s, link] - self.link_rotational_lr * self.link_rotational_accs.grad[s, link]
            self.link_translational_accs[s, link] = self.link_translational_accs[s, link] - self.link_translational_lr * self.link_translational_accs.grad[s, link]
    
    
    def update_accs(self,):
        for s in range(self.nn_timesteps):
            self.update_accs_per_timestep(s)
            
    
    
    @ti.kernel
    def update_particle_states(self, s: ti.i32):
        ### s should be larger than 0 ###
        for p in range(self.nn_particles):
            
            cur_particle_accs = self.particle_accs[s, p]
            prev_particle_vels = self.particle_vels[s - 1, p]
            prev_particle_vecs = self.particle_xs[s - 1, p] ### get the particle accs ###
            cur_particle_vels = prev_particle_vels + self.dt * cur_particle_accs
            cur_particle_xs = prev_particle_vecs + self.dt * prev_particle_vels + self.dt * self.dt * cur_particle_accs
            # 
            self.particle_xs[s, p ] = cur_particle_xs
            self.particle_vels[s, p] = cur_particle_vels
            
            
            # cur_particle_pos = self.particle_xs[s, p]
            # goal_pos = self.rigid_body_center[None] # + self.rigid_body_linear_states[s]
            # cur_particle_goal_diff = (cur_particle_pos - goal_pos).norm() ** 2
            # self.task_irr_goal_reaching_loss[None] = ti.min(cur_particle_goal_diff, self.task_irr_goal_reaching_loss[None])
    
    
    
    def forward_stepping_particles(self, sv_ckpt_tag):
        
        for i_s in range(self.nn_timesteps):
            if i_s > 0:
                ## update the particle states
                self.update_particle_states(i_s) 
            if i_s > 0:
                self.calculate_contact_forces(s=i_s)
                self.transform_rigid_body(s=i_s)
            
        ## save the checkpoints ###
        self.save_checkpoints(0, sv_ckpt_tag)
        
        ## get the ch
        
        ## now we have the checkpoins ###
        
        
        pass
        
    
    
    def optimize_iter(self, ):
        # idfferent nodes 
        # 
        # TODO: initialize the particles, links, joints, and goal positions #
        self.initialize_loss()
        self.initialize_grads()
        
        self.forward_stepping()
        
        
        if self.ii_iter  < 2000:    
            self.get_total_loss()
        else:
            # self.get_total_loss_v2()
            self.get_total_loss()
        
        # print(f"tot_loss: {self.tot_loss[None]}, transformation_penalty_loss: {self.transformation_penalty_loss[None]}")
        
        self.tot_loss.grad[None] = 1.0
        
        if self.use_wandb:
            wandb.log({
                "tot_loss": self.tot_loss.to_numpy().item(), 
                "transformation_penalty_loss": self.transformation_penalty_loss.to_numpy().item(),
                "goal_reaching_loss": self.goal_reaching_loss.to_numpy().item()
            })
        
        # self.get_total_loss.grad()
        
        if self.ii_iter  < 2000:    
            self.get_total_loss.grad()
        else:
            # self.get_total_loss_v2.grad()
            self.get_total_loss.grad()
        
        
        self.backward_stepping()
        


        self.update_accs()
        
    def forward_get_loss(self,):
        self.initialize_loss()
        self.initialize_grads()
        self.forward_stepping()
        self.get_total_loss()
        
        goal_reaching_loss_np = self.goal_reaching_loss.to_numpy().item()
        
        return goal_reaching_loss_np
    
    def forward_get_loss_fr_list(self, x):
        link_rotational_accs_flatten = x[: self.nn_rotational_acc_vars]
        link_translational_accs_flatten = x[self.nn_rotational_acc_vars: ]
        link_rotational_accs_flatten = np.array(link_rotational_accs_flatten, dtype=np.float32)
        link_rotational_accs_np = link_rotational_accs_flatten.reshape(self.link_rotational_accs_shape)
        
        link_translational_accs_flatten = np.array(link_translational_accs_flatten, dtype=np.float32)
        link_translational_accs_np = link_translational_accs_flatten.reshape(self.link_translational_accs_shape)
        
        self.link_rotational_accs.from_numpy(link_rotational_accs_np)
        self.link_translational_accs.from_numpy(link_translational_accs_np)
        
        x_loss = self.forward_get_loss()
        return x_loss
    
        
    
    ## for the gloal opt ##
    def optimize_iter_cmaes(self, ): 
        
        sigma = 0.1
        # sigma = 0.05
        
        # total_states = 
        
        # es = cma.CMAEvolutionStrategy(trans_states.tolist(), sigma)
        
        # for _ in range(6):
        solutions = self.es.ask()
        
        
        tot_res = []
        for x in solutions:
            
            x_loss = self.forward_get_loss_fr_list(x)
            
            tot_res.append(x_loss)
        
        
        self.es.tell(solutions, tot_res)
        self.es.logger.add()  
        self.es.disp()
    
        pass
    
    def optimize_cma(self, ):
        
        self.link_rotational_accs_shape = self.link_rotational_accs.to_numpy().shape
        self.link_translational_accs_shape = self.link_translational_accs.to_numpy().shape
        
        self.nn_rotational_acc_vars = self.link_rotational_accs.to_numpy().flatten().shape[0]
        
        optimize_variables = self.link_rotational_accs.to_numpy().flatten().tolist() + self.link_translational_accs.to_numpy().flatten().tolist()
        
        
        
        sigma = 1.0
        self.es = cma.CMAEvolutionStrategy(optimize_variables, sigma)
        
        ii_iter = 0
        while True:
            self.optimize_iter_cmaes()
            
            x_opt = self.es.result_pretty()[0]
            
            x_loss = self.forward_get_loss_fr_list(x_opt)
            
            ## get loss fr list ##
            print(f"ii_iter: {ii_iter}, tot_loss: {self.tot_loss[None]}, gaol_reaching: {self.goal_reaching_loss[None]} transformation_penalty_loss: {self.transformation_penalty_loss[None]}")
            
            ii_iter += 1
            

    
    # best_ckpt_sv_fn = optimize(nn_tot_iters=40000, n_terminate=False) ## n_terminate ##
    def optimize(self, nn_tot_iters=40000, n_terminate=False):
        ii_iter = 0
        best_loss = 9999999.0
        best_ckpt_sv_fn = ""
        while True:
            ii_iter += 1
            self.optimize_iter() ## optimize for each iteration ##
            
            goal_reaching_loss = self.goal_reaching_loss.to_numpy().item()
            if goal_reaching_loss < best_loss:
                best_loss = goal_reaching_loss
                best_ckpt_sv_fn = self.save_checkpoints(ii_iter, ckpt_sv_fn="ckpt_best")
                # best_ckpt_sv_fn = os.path.joi
                
            
            
            if ii_iter % 1000 == 0:
                print(f"ii_iter: {ii_iter}, tot_loss: {self.tot_loss[None]}, gaol_reaching: {self.goal_reaching_loss[None]} transformation_penalty_loss: {self.transformation_penalty_loss[None]}")
                
                if save_ckpts:
                    # ckpt_fn = os.path.join(self.ckpt_sv_folder, f"ckpt_{ii_iter}.npy") #
                    # np.save(ckpt_fn, {"particle_xs": self.particle_xs.to_numpy()}) #
                    
                    self.save_checkpoints(ii_iter) # save the per iter #
            
            if (not n_terminate) and (ii_iter >= nn_tot_iters):
                break
        return best_ckpt_sv_fn

    def optimize_with_planning(self, nn_tot_iters=40000, n_terminate=False):
        ii_iter = 0
        best_loss = 9999999.0
        best_ckpt_sv_fn = ""
        while True:
            ii_iter += 1
            self.ii_iter = ii_iter
            self.optimize_iter() ## optimize for each iteration ##
            
            goal_reaching_loss = self.goal_reaching_loss.to_numpy().item()
            if goal_reaching_loss < best_loss:
                best_loss = goal_reaching_loss
                best_ckpt_sv_fn = self.save_checkpoints(ii_iter, ckpt_sv_fn="ckpt_best")
            
            if ii_iter % 1000 == 0:
                print(f"ii_iter: {ii_iter}, tot_loss: {self.tot_loss[None]}, gaol_reaching: {self.goal_reaching_loss[None]} transformation_penalty_loss: {self.transformation_penalty_loss[None]}, task_irr_goal_reaching_loss: {self.task_irr_goal_reaching_loss[None]}")
                
                logging_dir = f"ii_iter: {ii_iter}, tot_loss: {self.tot_loss[None]}, gaol_reaching: {self.goal_reaching_loss[None]} transformation_penalty_loss: {self.transformation_penalty_loss[None]}, task_irr_goal_reaching_loss: {self.task_irr_goal_reaching_loss[None]}"
                ### log to the file ###
                with open(self.log_sv_fn, "a") as f:   
                    f.write(f"{logging_dir}\n")
                f.close()
                ### log to the file ###
                
                if self.cfg.general.save_ckpts:
                    # ckpt_fn = os.path.join(self.ckpt_sv_folder, f"ckpt_{ii_iter}.npy") #
                    # np.save(ckpt_fn, {"particle_xs": self.particle_xs.to_numpy()}) #
                    
                    self.save_checkpoints(ii_iter) # save the per iter #
            if ii_iter % 1000 == 0:
                last_ckpt_sv_fn = self.save_checkpoints(ii_iter, ckpt_sv_fn="ckpt_last")
            
            if (not n_terminate) and (ii_iter >= nn_tot_iters) and (goal_reaching_loss < 1e-2):
                break
            elif (not n_terminate) and (ii_iter >= nn_tot_iters * 10):
                break
        return best_ckpt_sv_fn
            
    
    # 
    def save_checkpoints(self, ii_iter, ckpt_sv_fn=None):
        
        
        save_info = {
            'particle_xs': self.particle_xs.to_numpy(),
            'particle_link_idx': self.particle_link_idx.to_numpy(),
            'link_rotational_accs': self.link_rotational_accs.to_numpy(),
            'link_translational_accs': self.link_translational_accs.to_numpy(),
            'rigid_linear_translations': self.rigid_body_linear_states.to_numpy(),
            'link_translational_vels': self.link_translational_vels.to_numpy(),
            'link_rotational_vels': self.link_rotational_vels.to_numpy(),
            'link_translational_vecs': self.link_translational_vecs.to_numpy(),
            'link_rotational_vecs': self.link_rotational_vecs.to_numpy(),
            # ''
        }
        
        if self.rigid_type == RIGID_SDF:
            tot_rigid_body_pcs = []
            for s in range(nn_timesteps):
                self.get_transformed_rigid_pcs(s)
                tot_rigid_body_pcs.append(self.rigid_pcs_transform.to_numpy())
            # self.get_transformed_rigid_pcs(s)
            # self.gui.circles(
            #     self.rigid_pcs_transform.to_numpy(), ## particle xs ##
            #     radius=1.5, # 
            #     # palette=[0x068587, 0xED553B, 0xEEEEF0], # get the particle xs #
            #     # palette_indices=material,
            # )
            tot_rigid_body_pcs = np.stack(tot_rigid_body_pcs, axis=0)
            save_info['rigid_body_pcs'] = tot_rigid_body_pcs
        
        if ckpt_sv_fn is not None:
            ckpt_fn = os.path.join(self.ckpt_sv_folder, f'{ckpt_sv_fn}.npy')
        else:
            ckpt_fn = os.path.join(self.ckpt_sv_folder, f'ckpt_{ii_iter}.npy')
        # print(f"ckpt_fn: {ckpt_fn}")
        np.save(ckpt_fn, save_info)
        return ckpt_fn
    
    def load_proj_optimized_info(self, ckpt_fn, proj_link_relations):
        optimized_info = np.load(ckpt_fn, allow_pickle=True).item()
        link_rotational_accs = optimized_info['link_rotational_accs']
        link_translational_accs = optimized_info['link_translational_accs']
        # proj_link_relations: { link_idx: [link_idx_1, link_idx_2], }
        cur_inst_link_rotational_accs_np = self.link_rotational_accs.to_numpy()
        cur_inst_link_translational_accs_np = self.link_translational_accs.to_numpy()
        
        for link_idx in proj_link_relations:
            cur_fr_link_idxes = proj_link_relations[link_idx]
            cur_fr_link_idxes = np.array(cur_fr_link_idxes)
            cur_fr_link_rotational_accs = link_rotational_accs[:, cur_fr_link_idxes]
            cur_proj_link_rotational_accs = np.mean(cur_fr_link_rotational_accs, axis=1)
            cur_fr_link_translational_accs = link_translational_accs[:, cur_fr_link_idxes]
            cur_proj_link_translational_accs = np.mean(cur_fr_link_translational_accs, axis=1)
            cur_inst_link_rotational_accs_np[:, link_idx] = cur_proj_link_rotational_accs
            cur_inst_link_translational_accs_np[:, link_idx] = cur_proj_link_translational_accs
        print(f"Initializing link rotational accs and translational accs from the projected values...")
        self.link_rotational_accs.from_numpy(cur_inst_link_rotational_accs_np)
        self.link_translational_accs.from_numpy(cur_inst_link_translational_accs_np)
        print(f"Initialized!")
        
            
    
        
    def forward_transform_links(self, link_idx_to_transformations): 
        raise NotImplementedError
    
    def set_link_transformations(self, link_idx_to_transformations):
        self.link_idx_to_transformations = link_idx_to_transformations
    
    def ori_init(self, n_links, link_lengths, link_masses, link_inertias):
        self.n_links = n_links
        self.link_lengths = link_lengths
        self.link_masses = link_masses
        self.link_inertias = link_inertias
        self.link_mass_centers = np.zeros((n_links, 2))
        self.link_mass_centers[:, 0] = link_lengths / 2
        self.link_mass_centers[:, 1] = 0
        self.link_mass_centers = self.link_mass_centers
        self.link_mass_moments = np.zeros((n_links, 2, 2))
        for i in range(n_links): ##
            self.link_mass_moments[i, 0, 0] = link_masses[i] * (link_lengths[i] ** 2) / 12
            self.link_mass_moments[i, 1, 1] = link_masses[i] * (link_lengths[i] ** 2) / 12
        self.link_mass_moments = self.link_mass_moments
        self.link_mass_moments_inv = np.zeros((n_links, 2, 2))
        for i in range(n_links):
            self.link_mass_moments_inv[i, 0, 0] = 1 / self.link_mass_moments[i, 0, 0]
            self.link_mass_moments_inv


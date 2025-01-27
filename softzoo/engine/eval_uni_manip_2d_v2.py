import numpy as np
import taichi as ti
import os 
import wandb
import cma

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

## get the graph structure ## 
## graph node: information about the graph node including the x-y sizes ##
## graph connectiveity information -- the A_t matrix where the graph size is ##
## graph connectivity information -- the A_t matrix where the graph size is ##

## graph connectivity information -- the A_t matrix where the graph size is N nodes ##

## graph connectivity information ##

## evaluate the manipulators ##


class GraphNode:
    def __init__(self, x_size, y_size, node_idx, node_type, node_name):
        self.x_size = x_size
        self.y_size = y_size
        self.node_idx = node_idx
        self.node_type = node_type
        self.node_name = node_name


        self.node_x = 0.0
        self.node_y = 0.0
    
    def set_node_xy(self, _x, _y):
        self.node_x = _x
        self.node_y = _y
        
class Graph:
    def __init__(self) -> None:
        # 
        # get the graph #
        self.graph_nodes = []
        self.graph_connectivity = []
        
        
        self.nn_timesteps = 10
        self.nn_links = 21 
        
        self.dim = dim
        
        ## the sum of the row should be 1 ## ## not sure whether this ## ## ## [-1, 1] -> scal to [0, 1] -> normalize ##
        self.graph_A = ti.field(dtype=ti.f32, shape=(self.nn_links, self.nn_links))
        # 
        if dim == 2:
            self.graph_node_angular_acc = ti.Vector.field(n=dim - 1, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
            self.proj_graph_node_angular_acc = ti.Vector.field(n=dim - 1, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
        else:
            self.graph_node_angular_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
            self.proj_graph_node_angular_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
            
        # graph node angular accs 
        self.graph_node_linear_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps)) 
        
        # self.proj_gra
        ## graph linear acc ## 
        self.proj_graph_node_linear_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
        
        # get hteaction simulation forward and the backward process, right? # 
        self.link_joint_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        self.link_joint_dir = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        
        self.get_directional_graph_mask()
        
        
        pass
    
    
    def get_directional_graph_mask(self, ):
        directional_graph_mask = np.ones((self.nn_links, self.nn_links), dtype=np.float32)
        for i_link in range(self.nn_links):
            directional_graph_mask[i_link, : i_link] = 0.0
        self.directional_graph_mask = directional_graph_mask
        
    
    ## we should have the graph structure ##
    def set_graph_connectivity_normalized_value(self, graph_connectivity_arr):
        #3 assum the graph connectivity arr has 
        graph_connectivity_arr = np.clip(graph_connectivity_arr, -1.0, 1.0)
        graph_connectivity_arr = (graph_connectivity_arr + 1.0) / 2.0
        graph_connectivity_arr = graph_connectivity_arr * self.directional_graph_mask
        ## should 
        graph_connectivity_arr = graph_connectivity_arr / np.clip(np.sum(graph_connectivity_arr, axis=0, keepdims=True), a_min=1e-6, a_max=None)
        
        self.graph_A.from_numpy(graph_connectivity_arr) 
        
        
    ## we should have the graph structure ##
    def set_graph_connectivity(self, graph_connectivity_arr):
        #3 assum the graph connectivity arr has 
        # graph_connectivity_arr = np.clip(graph_connectivity_arr, -1.0, 1.0)
        # graph_connectivity_arr = (graph_connectivity_arr + 1.0) / 2.0
        # graph_connectivity_arr = graph_connectivity_arr * self.directional_graph_mask
        # ## should 
        # graph_connectivity_arr = graph_connectivity_arr / np.clip(np.sum(graph_connectivity_arr, axis=0, keepdims=True), a_min=1e-6, a_max=None)
        
        self.graph_A.from_numpy(graph_connectivity_arr) 
        
    # def set_graph_link_p
    def set_link_joint_pos(self, link_joint_pos):
        self.link_joint_pos.from_numpy(link_joint_pos) ## for each link -> can be connected to the parent with one side and the child with another side ##



@ti.data_oriented
class UniManip2D:
    def __init__(self, nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=None, cur_transformation_penalty_term=1000, cfg=None):
        # self.link_idx_to_pcs = link_idx_to_pcs # 
        # self.link_idx_to_parent = link_idx_to_parent
        # self.link_idx_to_joint_info = link_idx_to_joint_info
        # self.joint_idx_to_constraint_factor = joint_idx_to_constraint_factor
        self.dim = dim
        # transform optimization results across different kinds of manipulators -> try to use the originally optimized transformations to initilize the following transformations
        self.nn_particles = nn_particles
        self.nn_links = nn_links
        self.nn_joints = nn_joints
        self.nn_timesteps = nn_timesteps
        
        self.cur_transformation_penalty_term = cur_transformation_penalty_term
        
        self.cfg = cfg
        
        #### ==== Get sampled graph structures and the projected acts ===== ##### 
        self.sampled_graph_wacts_sv_rt_dict = self.cfg.sampled_graph_wacts_sv_rt_dict ## get the sampled graphs and the acts #
        self.sampled_rt_dict = np.load(self.sampled_graph_wacts_sv_rt_dict, allow_pickle=True).item()
        
        graph_idx = 0
        self.sampled_link_acts = self.sampled_rt_dict['links_act'][graph_idx]
        self.graph_X = self.sampled_rt_dict['graph_X'][graph_idx]
        self.graph_E = self.sampled_rt_dict['graph_E'][graph_idx] ## get the graph_X, graph_E ##
        self.sampled_link_joint_pos = self.sampled_rt_dict['link_joint_pos'][graph_idx]
        
        print(self.graph_X)
        
        link_xy_threshold = 3e-2
        first_invalid_link_idx = 0
        for ii_link in range(self.graph_X.shape[0]):
            cur_x, cur_y = self.graph_X[ii_link, 0], self.graph_X[ii_link, 1]
            if cur_x < link_xy_threshold or cur_y < link_xy_threshold:
                first_invalid_link_idx = ii_link
                break
        nn_links = first_invalid_link_idx
        self.nn_links = nn_links
        
        # self.point_to_link_idx = ti.field(dtype=ti.i32, )
        # self.joint_pos = ti.field(dtype=ti.f32, shape=(self.nn_joints, 2))
        # self.joint_dirs = ti.field(dtype=ti.f32, shape=(self.nn_joints, 2)) 
        # self.joint_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_joints, ))
        # self.joint_dirs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_joints, )) ## TODO: should be normalized to the 1-norm vector in the initialization and the following time-stepping processes as well 
        
        ''' Making the data folder for saving checkpoints '''

        
        self.use_wandb = self.cfg.general.use_wandb
        
        CODE_ROOT_FOLDER = self.cfg.run.proj_dir
        PROJ_ROOT_FOLDER = self.cfg.run.root_dir
        save_root_dir = os.path.join(PROJ_ROOT_FOLDER, "exp")
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
        # self.link_translational_lr = 0.001
        self.link_rotational_lr = self.cfg.optimization.link_rotational_lr
        self.link_translational_lr = self.cfg.optimization.link_translational_lr
        
        
        ''' Get initial articulated object information -- at the rest pose state ''' # link rotational lr ##
        self.obj_info_fn = obj_info_fn
        self.obj_info = np.load(self.obj_info_fn, allow_pickle=True).item()
        self.ori_particles_xs = self.obj_info['particles_xs']
        self.particle_link_idxes = self.obj_info['particle_link_idxes']
        
        expanded_ori_particles_xs = self.ori_particles_xs[None, :, :]
        expanded_ori_particles_xs = np.repeat(expanded_ori_particles_xs, self.nn_timesteps, axis=0)
        self.expanded_ori_particles_xs = expanded_ori_particles_xs
        
        ## particle_vels, link_rotational_dot_mtx ##
        
        ''' Particle States '''
        self.particle_xs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_particles), needs_grad=True)
        self.particle_vels = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_particles), needs_grad=True)
        self.particle_link_idx = ti.field(dtype=ti.i32, shape=(nn_particles, ))
        
        self.particle_xs.from_numpy(self.expanded_ori_particles_xs)
        self.particle_link_idx.from_numpy(self.particle_link_idxes) 
        
        
        ''' The graph connectivity matrix '''
        self.graph = Graph()
        ## TODO: set the graph connectivity array; ##
        ## TODO: normalize the graph connectivity array; ##
        ## TODO: vis the graph connectivity info ###
        self.set_graph_connectivity(self.graph_E) # set hte connecitivyt informations ##
        
        
        
        
        ''' Link joint infos '''
        self.link_joint_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        self.link_joint_dir = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        self.link_parent_idx = ti.field(dtype=ti.i32, shape=(self.nn_links))
        self.link_joint_pos.from_numpy(self.sampled_link_joint_pos)
        self.link_joint_dir.from_numpy(link_joint_dir)
        # self.link_parent_idx.from_numpy(link_parent_idx)
        # general_link_parent_idx = [0] + [i - 1 for i in rang]
        link_parent_idx = [0] + [i - 1 for i in range(1, 1 + (self.nn_links - 1) // 2)] + [0] + [i - 1 for i in range(2 + (self.nn_links - 1) // 2, self.nn_links)]
        link_parent_idx = np.array(link_parent_idx, dtype=np.int32)
        self.link_parent_idx.from_numpy(link_parent_idx)
        
        
        ## selflink ##
        
        # link_translational_vels, link_rotational_vels
        # link_translational_vecs, link_rotational_vecs
        
        
        ''' Link SE(3) States '''
        if self.dim == 2:
            # rotation angle #
            self.link_rotational_vecs = ti.Vector.field(1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
            self.link_rotational_vels = ti.Vector.field(1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        else:
            # quaternion #
            self.link_rotational_vecs = ti.Vector.field(self.dim + 1, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
            self.link_rotational_vels = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.link_rotational_mtx = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        self.link_rotational_dot_mtx = ti.Matrix.field(self.dim, self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        
        self.link_translational_vecs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True) # 
        self.link_translational_vels = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True)
        
        self.proj_link_translational_vecs = ti.Vector.field(self.dim, dtype=ti.f32, shape=(self.nn_timesteps, self.nn_links), needs_grad=True) # 
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
        self.transformation_acc_reg_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True) # 
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
        self.rigid_body_torques = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid body torques ##
        self.rigid_body_linear_accs = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid body accs ##
        self.rigid_body_angular_accs = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid body angular ac ## 
        self.rigid_body_linear_vels = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True)
        self.rigid_body_angular_vels = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid body angular vels ##
        self.rigid_body_linear_states = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid bdoy positions ##
        self.rigid_body_angular_states = ti.Vector.field(n=1, dtype=ti.f32, shape=(self.nn_timesteps, ), needs_grad=True) ## rigid body orientations ##
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
        # goal reaching 
        
        
        # self.rigid_type = RIGID_CUBE
        
        ### TODO: add a general SDF representation ###
        ## scale, center; and how to calculate the sdf values ##
        
        
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
        self.contact_depth_coef = ti.field(dtype=ti.f32, shape=()) ## contact distance ##
        self.contact_damping_coef = ti.field(dtype=ti.f32, shape=())
        self.contact_depth_coef[None] = self.cfg.sim.contact_spring_d #   1e-1
        self.contact_damping_coef[None] =  self.cfg.sim.contact_damping_d #  1e-1
        
        
        self.vis = self.cfg.general.vis
        # print(self.vis: )
        print(f"vis: {self.vis}")
        
        if self.vis:
            # self.vis = vis
            self.gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
        # else:
        #     self.vis = False
    
    def set_links_rotational_translational_accs(self):
        # sampled_link_acts
        self.sampled_link_acts = np.transpose(self.sampled_link_acts, (1, 0, 2))
        sampled_link_rot_acts, sampled_link_trans_acts = self.sampled_link_acts[..., :1], self.sampled_link_acts[..., 1:]
        self.link_rotational_accs.from_numpy(sampled_link_rot_acts)
        self.link_translational_accs.from_numpy(sampled_link_trans_acts)
        
        pass
    
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
        self.proj_link_translational_accs.grad.fill(0.0) ## proj link translational s ##
        
        self.proj_link_translational_vels.fill(0.0)
        self.proj_link_translational_vecs.fill(0.0)
        self.proj_link_translational_vels.grad.fill(0.0)
        self.proj_link_translational_vecs.grad.fill(0.0)
        # else:
        #     raise NotImplementedError    
        
        
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
    
    
    # 
    
    @ti.kernel
    def initialize_grads(self, ):
        self.particle_xs.grad.fill(0.0)
        
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
        # v_{t + 1} = v_{t} + dt * a_{t + 1} #
        # \delta x_{t + 1} = x_{t} + dt * v_{t + 1} #
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
            
            link_rot_mtx_dot = self.link_rotational_dot_mtx[s, link_idx] ## link ##
            link_trans_vel = self.link_translational_vels[s, link_idx]
            cur_particle_vel = link_rot_mtx_dot @ ori_particle_pos + link_trans_vel ## add the link transformation velocities ##
            
            self.particle_vels[s, p] = cur_particle_vel ## get the particle's velocity 
    
    @ti.func
    def get_sdf_sphere(self, pos):
        rel_center_to_pos = pos  # - self.rigid_body_center[None] ## cetner to the pos ##
        sqr_rel_center_to_pos = rel_center_to_pos ** 2
        sqr_rel_center_to_pos_all = sqr_rel_center_to_pos[0] + sqr_rel_center_to_pos[1]
        dist_center_to_pos = ti.sqrt(sqr_rel_center_to_pos_all)
        pos_sdf = dist_center_to_pos - self.rigid_body_radius[None]
        return pos_sdf
        
    @ti.func
    def get_sdf_cube(self, pos):
        rel_center_to_pos = pos # - self.rigid_body_center[None] ## cetner to the pos ##
        abs_delta_x = ti.abs(rel_center_to_pos[0])
        abs_delta_y = ti.abs(rel_center_to_pos[1])
        sdf_x = abs_delta_x - self.rigid_body_radius[None]
        sdf_y = abs_delta_y - self.rigid_body_radius[None]
        pos_sdf = ti.max(sdf_x, sdf_y)
        return pos_sdf

    @ti.func
    def get_sdf_rigid_sdf(self, pos):
        # rel_center_to_pos = pos - (self.rigid_body_center[None] + self.)
        scaled_pos = (pos / self.rigid_body_scale[None]) + self.rigid_body_center_pcs[None] ## to the sdf coordinate
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
        # else:
        #     raise NotImplementedError
        return self.pos_sdf_values[None]
        # get the center to pos #
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
    
    ## calculate # penetration depth and the penetration velocity -> calculate the contact forces for the object ##
    
    
    # the goal is reaching a goal #
    # goal is reaching a goal #
    # def transform_particles
                
    
    
    
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
        # self.tot_loss[None] += self.transformation_penalty_loss[None] * self.cur_transformation_penalty_term + self.transformation_acc_reg_loss[None] * 0.01 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]

        self.tot_loss[None] += self.transformation_acc_reg_loss[None] * 0.01 + self.goal_reaching_loss[None]  + self.task_irr_goal_reaching_loss[None]


    
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
            
            self.get_transformation_matrices_wproj(s)
            
            self.transform_particles(s)
            
            if s > 0:
                self.calculate_contact_forces(s=s)
                self.transform_rigid_body(s=s)
            
            if self.vis:
                self.gui.circles(
                    self.particle_xs.to_numpy()[s], ## particle xs ##
                    radius=1.5, # 
                    # palette=[0x068587, 0xED553B, 0xEEEEF0], # get the particle xs #
                    # palette_indices=material,
                )
                
                if self.rigid_type == RIGID_SDF:
                    self.get_transformed_rigid_pcs(s)
                    self.gui.circles(
                        self.rigid_pcs_transform.to_numpy(), ## particle xs ##
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
        self.get_goal_reaching_loss(self.nn_timesteps - 1) # goal reaching loss for each timestep
        self.get_task_irre_goal_reaching_loss(self.nn_timesteps - 1)
    
    
    def backward_stepping(self):
        self.get_goal_reaching_loss_backward(self.nn_timesteps - 1) # for particle xs ## 
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
            
        
            # self.get_transformation_matrices.grad(s)
            self.get_transformation_matrices_wproj_grad(s)
            # rotational_accs_grad = np.sum(self.link_rotational_accs.grad.to_numpy())
            # translational_accs_grad = np.sum(self.link_translational_accs.grad.to_numpy())
            # print(f"After {s}-step's transformation backward: rotational_accs_grad = {rotational_accs_grad}, translational_accs_grad: {translational_accs_grad}")
        
    
    @ti.kernel
    def update_accs_per_timestep(self, s: ti.i32): ## update accs per timesttep ##
        for link in range(self.nn_links):
            self.link_rotational_accs[s, link] = self.link_rotational_accs[s, link] - self.link_rotational_lr * self.link_rotational_accs.grad[s, link]
            self.link_translational_accs[s, link] = self.link_translational_accs[s, link] - self.link_translational_lr * self.link_translational_accs.grad[s, link]
    
    
    def update_accs(self,):
        for s in range(self.nn_timesteps):
            self.update_accs_per_timestep(s)
    
    
    def evaluate_iter(self, ): # --- for nodes --- we want to model the distriution --- the structural and the information of such K nodes --- x length; y length; and the connectivities between 
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
        
        # self.tot_loss.grad[None] = 1.0
        
        # if self.use_wandb:
        #     wandb.log({
        #         "tot_loss": self.tot_loss.to_numpy().item(), 
        #         "transformation_penalty_loss": self.transformation_penalty_loss.to_numpy().item(),
        #         "goal_reaching_loss": self.goal_reaching_loss.to_numpy().item()
        #     })
        
        # # self.get_total_loss.grad()
        
        # if self.ii_iter  < 2000:    
        #     self.get_total_loss.grad()
        # else:
        #     # self.get_total_loss_v2.grad()
        #     self.get_total_loss.grad()
        
        # # goal_reaching_loss_grad = np.sum(self.goal_reaching_loss.grad.to_numpy())
        # # task_irr_goal_reaching_loss_grad = np.sum(self.task_irr_goal_reaching_loss.grad.to_numpy())
        # # print(f"After tot_loss grad: goal_reaching_loss_grad = {goal_reaching_loss_grad}, transformation_penalty_loss_grad: {self.transformation_penalty_loss.grad[None]}, task_irr_goal_reaching_loss_grad: {task_irr_goal_reaching_loss_grad}")
        
        # self.backward_stepping()
        
        # self.update_accs()
    
    
    
    def optimize_iter(self, ): # --- for nodes --- we want to model the distriution --- the structural and the information of such K nodes --- x length; y length; and the connectivities between 
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
        
        # goal_reaching_loss_grad = np.sum(self.goal_reaching_loss.grad.to_numpy())
        # task_irr_goal_reaching_loss_grad = np.sum(self.task_irr_goal_reaching_loss.grad.to_numpy())
        # print(f"After tot_loss grad: goal_reaching_loss_grad = {goal_reaching_loss_grad}, transformation_penalty_loss_grad: {self.transformation_penalty_loss.grad[None]}, task_irr_goal_reaching_loss_grad: {task_irr_goal_reaching_loss_grad}")
        
        self.backward_stepping()
        
# <<<<<<< HEAD
#         # rigid_body_linear_state_grad = np.sum(self.rigid_body_linear_states.grad.to_numpy())
#         # rigid_body_linear_forces_grad = np.sum(self.rigid_body_linear_forces.grad.to_numpy())
#         # rigid_body_torques_grad = np.sum(self.rigid_body_torques.grad.to_numpy())
#         # rotational_accs_grad = np.sum(self.link_rotational_accs.grad.to_numpy())
#         # translational_accs_grad = np.sum(self.link_translational_accs.grad.to_numpy())
#         # print(f"rigid_body_linear_state_grad: {rigid_body_linear_state_grad}")
#         # print(f"rigid_body_linear_forces_grad: {rigid_body_linear_forces_grad}, rigid_body_torques_grad: {rigid_body_torques_grad}")
        
#         # print(f"After back_stepping: rotational_accs_grad = {rotational_accs_grad}, translational_accs_grad: {translational_accs_grad}")
# =======
        
#         ## 
        
#         # if self.ii_iter  >= 2000:    
#         #     # rigid_body_linear_state_grad = np.sum(self.rigid_body_linear_states.grad.to_numpy())
#         #     # rigid_body_linear_forces_grad = np.sum(self.rigid_body_linear_forces.grad.to_numpy())
#         #     # rigid_body_torques_grad = np.sum(self.rigid_body_torques.grad.to_numpy())
#         #     rotational_accs_grad = np.sum(self.link_rotational_accs.grad.to_numpy())
#         #     translational_accs_grad = np.sum(self.link_translational_accs.grad.to_numpy())
#         #     # print(f"rigid_body_linear_state_grad: {rigid_body_linear_state_grad}")
#         #     # print(f"rigid_body_linear_forces_grad: {rigid_body_linear_forces_grad}, rigid_body_torques_grad: {rigid_body_torques_grad}")
#         #     print(f"After back_stepping: rotational_accs_grad = {rotational_accs_grad}, translational_accs_grad: {translational_accs_grad}")
# >>>>>>> d0fd77bbe6dbba4bb50509f06bc66a0bccb51fd4
        
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
            # link_rotational_accs_flatten = x[: self.nn_rotational_acc_vars]
            # link_translational_accs_flatten = x[self.nn_rotational_acc_vars: ]
            # link_rotational_accs_flatten = np.array(link_rotational_accs_flatten, dtype=np.float32)
            # link_rotational_accs_np = link_rotational_accs_flatten.reshape(self.link_rotational_accs_shape)
            
            # link_translational_accs_flatten = np.array(link_translational_accs_flatten, dtype=np.float32)
            # link_translational_accs_np = link_translational_accs_flatten.reshape(self.link_translational_accs_shape)
            
            # self.link_rotational_accs.from_numpy(link_rotational_accs_np)
            # self.link_translational_accs.from_numpy(link_translational_accs_np)
            
            # x_loss = self.forward_get_loss()
            
            x_loss = self.forward_get_loss_fr_list(x)
            
            tot_res.append(x_loss)
        
        
        self.es.tell(solutions, tot_res)
        self.es.logger.add()  
        self.es.disp()
        
        ## xopt, es ##
        # x_opt, es = cma.fmin2(cmaes_func, trans_states.tolist(), sigma)
        
    
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
            
            
            ## not save the checkpoint ##
            # if we use an optimization strateg? #
            # if ii_iter % 1000 == 0:
            #     self.save_checkpoints(ii_iter) 
            
            
            # if ii_iter >= 100:
            #     break
            # # print(f"Getting ")
            
            
        
    
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

    
    def eval_res(self, nn_tot_iters=40000, n_terminate=False):
        ii_iter = 0
        best_loss = 9999999.0
        best_ckpt_sv_fn = ""
        # while True:
        
        # ii_iter += 1
        self.ii_iter = ii_iter
        self.evaluate_iter() ## optimize for each iteration ##
        
        goal_reaching_loss = self.goal_reaching_loss.to_numpy().item()
        # if goal_reaching_loss < best_loss:
        #     best_loss = goal_reaching_loss
        best_ckpt_sv_fn = self.save_checkpoints(ii_iter, ckpt_sv_fn="ckpt_best")
        print(f"saved to best_ckpt_sv_fn: {best_ckpt_sv_fn}")
        
        
        if ii_iter % 1000 == 0:
            print(f"ii_iter: {ii_iter}, tot_loss: {self.tot_loss[None]}, gaol_reaching: {self.goal_reaching_loss[None]} transformation_penalty_loss: {self.transformation_penalty_loss[None]}, task_irr_goal_reaching_loss: {self.task_irr_goal_reaching_loss[None]}")
            
            logging_dir = f"ii_iter: {ii_iter}, tot_loss: {self.tot_loss[None]}, gaol_reaching: {self.goal_reaching_loss[None]} transformation_penalty_loss: {self.transformation_penalty_loss[None]}, task_irr_goal_reaching_loss: {self.task_irr_goal_reaching_loss[None]}"
            ### log to the file ###
            with open(self.log_sv_fn, "a") as f:   
                f.write(f"{logging_dir}\n")
            f.close()
            ### log to the file ###
            
        #     if self.cfg.general.save_ckpts:
        #         # ckpt_fn = os.path.join(self.ckpt_sv_folder, f"ckpt_{ii_iter}.npy") #
        #         # np.save(ckpt_fn, {"particle_xs": self.particle_xs.to_numpy()}) #
                
        #         self.save_checkpoints(ii_iter) # save the per iter #
        # if ii_iter % 1000 == 0:
        #     last_ckpt_sv_fn = self.save_checkpoints(ii_iter, ckpt_sv_fn="ckpt_last")
        
        # if (not n_terminate) and (ii_iter >= nn_tot_iters) and (goal_reaching_loss < 1e-2):
        #     break
        # elif (not n_terminate) and (ii_iter >= nn_tot_iters * 10):
        #     break
        # return best_ckpt_sv_fn
            
    
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
            'link_translational_vecs': self.link_translational_vecs.to_numpy(), ## save the useufl ckpts ##
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
        
            

        
    def forward_transform_links(self, link_idx_to_transformations): # can complete in the taichi forward process
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

### as a simple test you can generate one simple manipulator and test the forward and backward processes ###
### as a simple test you can generate one simple manipulator and test the forward and backward processes ###

def generate_test_links(dim=2):
    # three links #
    n_links = 3
    ### the base link ###
    rnd_xys = np.random.rand(n_particles, dim) * 0.1 ## nn_particles x 2 ## 
    base_link_xys = np.array([0.45, 0.45], dtype=np.float32)[None, :] + rnd_xys ## get the random initialized xys ##
    rnd_xys = np.random.rand(n_particles, dim) * np.array([0.2, 0.1], dtype=np.float32) ## nn_particles x 2 ##
    right_link_xys = np.array([0.55, 0.45], dtype=np.float32)[None, :] + rnd_xys
    rnd_xys = np.random.rand(n_particles, dim) * np.array([0.2, 0.1], dtype=np.float32) 
    left_link_xys = np.array([0.25, 0.45], dtype=np.float32)[None, :] + rnd_xys

    obj_particles = np.concatenate(
        [base_link_xys, left_link_xys, right_link_xys], axis=0
    )
    particle_link_idxes = np.zeros((n_particles * 3, ), dtype=np.int32)
    particle_link_idxes[n_particles: 2 * n_particles] = 1
    particle_link_idxes[2 * n_particles:] = 2
    
    # particles_xs, particle_link_idxes #
    obj_info = {
        'particles_xs': obj_particles,
        'particle_link_idxes': particle_link_idxes
    }
    save_res_dir = "assets"
    if not os.path.exists(save_res_dir):
        os.makedirs(save_res_dir, exist_ok=True)
    obj_info_fn = os.path.join(save_res_dir, "obj_info.npy")
    np.save(obj_info_fn, obj_info)
    print(f'Object information file saved to {obj_info_fn}')

def generate_test_links_2(dim=2):
    n_links = 5
    rnd_xys = np.random.rand(n_particles, dim) * 0.1 ## nn_particles x 2 ##
    base_link_xys = np.array([0.45, 0.45], dtype=np.float32)[None, :] + rnd_xys ## get the random initialized xys ##
    base_link_particles_link_idx = np.zeros((n_particles, ), dtype=np.int32)
    
    rnd_xys = np.random.rand(n_particles // 2, dim) * np.array([0.1, 0.1], dtype=np.float32)
    right_link_1_xys = np.array([0.55, 0.45], dtype=np.float32)[None, :] + rnd_xys
    right_link_1_particles_link_idx = np.ones((n_particles // 2, ), dtype=np.int32) * 3
    
    rnd_xys = np.random.rand(n_particles // 2, dim) * np.array([0.1, 0.1], dtype=np.float32)
    right_link_2_xys = np.array([0.65, 0.45], dtype=np.float32)[None, :] + rnd_xys
    right_link_2_particles_link_idx = np.ones((n_particles // 2, ), dtype=np.int32) * 4
    
    rnd_xys = np.random.rand(n_particles // 2, dim) * np.array([0.1, 0.1], dtype=np.float32)
    left_link_1_xys = np.array([0.35, 0.45], dtype=np.float32)[None, :] + rnd_xys
    left_link_1_particles_link_idx = np.ones((n_particles // 2, ), dtype=np.int32)
    
    rnd_xys = np.random.rand(n_particles // 2, dim) * np.array([0.1, 0.1], dtype=np.float32)
    left_link_2_xys = np.array([0.25, 0.45], dtype=np.float32)[None, :] + rnd_xys
    left_link_2_particles_link_idx = np.ones((n_particles // 2, ), dtype=np.int32) * 2
    
    obj_particles = np.concatenate(
        [base_link_xys, left_link_1_xys, left_link_2_xys, right_link_1_xys, right_link_2_xys], axis=0
    )
    particle_link_idxes = np.concatenate(
        [base_link_particles_link_idx, left_link_1_particles_link_idx, left_link_2_particles_link_idx, right_link_1_particles_link_idx, right_link_2_particles_link_idx], axis=0
    )
    link_joint_pos = [
        [0, 0],
        [0.45, 0.45],
        [0.35, 0.45],
        [0.55, 0.45],
        [0.65, 0.45]
    ]
    
    link_joint_dir = [
        [1.0, 0.0] for _ in range(n_links)
    ]
    
    link_parent_idx = [-1, 0, 1, 0, 3]
    link_joint_pos = np.array(link_joint_pos, dtype=np.float32)
    link_joint_dir = np.array(link_joint_dir, dtype=np.float32) ## 
    link_parent_idx = np.array(link_parent_idx, dtype=np.int32)
    
    obj_info = {
        'particles_xs': obj_particles,
        'particle_link_idxes': particle_link_idxes,
        'link_joint_pos': link_joint_pos,
        'link_joint_dir': link_joint_dir,
        'link_parent_idx': link_parent_idx
    }
    obj_info_sv_fn = os.path.join(f"assets", f"obj_info_n_links_{n_links}.npy")
    np.save(obj_info_sv_fn, obj_info)
    print(f"Object information saved to {obj_info_sv_fn}")
    
    
# unified manipulators space #
# unified manipulators space #
# constraints unification, joint transformations unification #
# for the root link with index 0, we do not have its parent link or joint information #
# for others, the transformation constraints should be added between the current link's transformation and the parent link's #
# for others, #

def generate_test_links_general(dim, nn_links_one_side, len_one_side):
    per_link_len = len_one_side / float(nn_links_one_side)
    
    
    
    particle_density = n_particles / (0.1 * 0.1)
    base_link_n_particles = particle_density * (0.1 * 0.1)
    child_link_n_particles = particle_density * (per_link_len * 0.1)
    
    base_link_n_particles = int(base_link_n_particles)
    child_link_n_particles = int(child_link_n_particles)
    
    ## 
    rnd_xys = np.random.rand(base_link_n_particles, dim) * 0.1 ## nn_particles x 2 ##
    base_link_xys = np.array([0.45, 0.45], dtype=np.float32)[None, :] + rnd_xys ## get the random initialized xys ##
    base_link_particles_link_idx = np.zeros((base_link_n_particles, ), dtype=np.int32)
    
    link_xys = [base_link_xys]
    link_particle_link_idx = [base_link_particles_link_idx]
    link_joint_pos = [np.array([0.0, 0.0], dtype=np.float32)]
    link_parent_idx = [-1]
    
    
    child_link_idx = 1
    joint_x = 0.45
    joint_y = 0.45
    for i_link in range(nn_links_one_side):
        cur_link_st_x = joint_x - per_link_len
        rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, 0.1], dtype=np.float32)
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
        rnd_xys = np.random.rand(child_link_n_particles, dim) * np.array([per_link_len, 0.1], dtype=np.float32)
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
    
    
    obj_info = {
        'particles_xs': obj_particles,
        'particle_link_idxes': particle_link_idxes,
        'link_joint_pos': link_joint_pos,
        'link_joint_dir': link_joint_dir,
        'link_parent_idx': link_parent_idx
    }
    obj_info_sv_fn = os.path.join(f"assets", f"obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}.npy")
    np.save(obj_info_sv_fn, obj_info)
    print(f"Object information saved to {obj_info_sv_fn}")
    

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
    
    
def test_link_gen():
    # generate_test_links_general(dim, nn_links_one_side, len_one_side):
    tot_nn_links_one_side = [1, 2, 2, 4, 4, 8] # link one side #
    tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    dim = 2
    for nn_links_one_side, len_one_side in zip(tot_nn_links_one_side, tot_len_one_side):
        generate_test_links_general(dim, nn_links_one_side, len_one_side)

def link_gen_general_v2(st_len_one_side, ed_len_one_side, nn_stages):
    # generate_test_links_general(dim, nn_links_one_side, len_one_side):
    # tot_nn_links_one_side = [1, 2, 2, 4, 4, 8] # link one side #
    # tot_len_one_side = [0.2, 0.2, 0.3, 0.3, 0.4, 0.4]
    
    link_len_one_side_interval = (ed_len_one_side - st_len_one_side) / float(nn_stages - 1)
    tot_len_one_side_unqie = [st_len_one_side + i * link_len_one_side_interval for i in range(nn_stages)]
    
    tot_nn_links_one_side = []
    tot_len_one_side = []
    st_nn_link_one_side = 1
    for i_stage in range(len(tot_len_one_side_unqie)):
        tot_nn_links_one_side.append(st_nn_link_one_side)
        tot_len_one_side.append(tot_len_one_side_unqie[i_stage // 2])
        
        if i_stage % 2 == 0:
            st_nn_link_one_side *= 2
    
    print("tot_nn_links_one_side: ", tot_nn_links_one_side)
    print(f"tot_len_one_side: {tot_len_one_side}")
    
    
    dim = 2
    for nn_links_one_side, len_one_side in zip(tot_nn_links_one_side, tot_len_one_side):
        generate_test_links_general(dim, nn_links_one_side, len_one_side)


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


## test the def link ## 
### for two d manipulators, we have one uniform manipulator formulation ###
### one root link with several side links lying on two sides ###
### one root link with several side links lying on two sides ###

## generate the graph ##
## get the graph connectivity informaton ##

## connectivity arrs ##

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




import hydra


def main_b(cfg):
    # PROJ_ROOT_FOLDER = "/root/diffsim/softzoo"
    
    # PROJ_ROOT_FOLDER = "/data/xueyi/softzoo" ## set the proj root folder ## \
        
    PROJ_ROOT_FOLDER = cfg.run.root_dir
    
    ''' Generate necessary links here  '''
    fixed_y_len = cfg.uni_manip.fixed_y_len
    base_x_len = cfg.uni_manip.base_x_len
    base_xys_tag = f"baseX_{base_x_len}_Y_{fixed_y_len}"
    st_len_one_side = cfg.uni_manip.st_len_one_side
    ed_len_one_side = cfg.uni_manip.ed_len_one_side
    nn_stages = cfg.uni_manip.nn_stages
    
    link_gen_general_v3(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
    
    tot_nn_links_one_side, tot_len_one_side = get_manipulator_infos(st_len_one_side, ed_len_one_side, nn_stages, fixed_y_len=fixed_y_len, base_x_len=base_x_len)
    # exit(0)

    
    
    # python softzoo/engine/uni_manip_2d.py
    ''' General settings '''
    vis = True # vis #
    vis = False # vis #
    
    # vis = cfg.general.vis
    # save_ckpts 
    
    save_ckpts = True
    save_ckpts = False # save itermediate ckpt
    
    # use_wandb = True
    # use_wandb = False # not use wan db 
    use_wandb = cfg.general.use_wandb
    
    uni_manip_2d_nn_particles = n_particles * 3
    
    # goal_x = 0.7
    # goal_y = 0.3 
    
    goal_x = cfg.task.goal_x
    goal_y = cfg.task.goal_y
    
    if use_wandb:
        wandb.init(project="uni_manip_2d")
    # nn_timesteps = 10
    # dim = 2
    # dt = 1e-1
    
    nn_timesteps = cfg.task.nn_timesteps
    dt = cfg.task.dt
    dim = cfg.task.dim
    
    
    ''' Define the manipulator transformation process ''' 
    tot_transformation_penalty_loss = [1000, 1000, 1000, 1000, 1000, 3000]
    
    tot_transformation_penalty_loss = [100, 100, 100, 10, 10, 10]
    
    tot_proj_link_relations = [
        {0: [0], 1: [1, 2], 2: [3, 4]}, 
        {ii: [ii] for ii in range(tot_nn_links_one_side[1] * 2 + 1)},
        {ii: [ii * 2 - 1, ii * 2] if ii > 0 else [0] for ii in range(tot_nn_links_one_side[2] * 2 + 1) },
        {ii: [ii] for ii in range(tot_nn_links_one_side[3] * 2 + 1)},
        {ii: [ii * 2 - 1, ii * 2] if ii > 0 else [0] for ii in range(tot_nn_links_one_side[4] * 2 + 1) },
        {ii: [ii] for ii in range(tot_nn_links_one_side[5] * 2 + 1)},
    ]
    
    
    tot_nn_links_one_side = list(reversed(tot_nn_links_one_side))
    tot_len_one_side = list(reversed(tot_len_one_side))
    tot_proj_link_relations = list(reversed(tot_proj_link_relations))
    tot_transformation_penalty_loss = list(reversed(tot_transformation_penalty_loss))
    
    prev_saved_best_ckpt_fn =  None


    st_idx = cfg.run.start_inst_idx # start idx #
    # st_idx = 4
    
    
    # inherit_from_prev = True
    # inherit_from_prev = False
    inherit_from_prev = cfg.optimization.inherit_from_prev
    
    # v4 --- kd = 0.1; act-reaching-dist-thres = 1e-3
    additional_exp_tag = f"actpassivgoalreaching_3_lessconstraints_v4_test_inherit_{inherit_from_prev}_seed_{cfg.seed}_contact_spring_d_{cfg.sim.contact_spring_d}_damping_{cfg.sim.contact_damping_d}"
    
    
    ''' Start optimization '''
    for i_inst in range(st_idx, len(tot_len_one_side)):
        
        nn_links_one_side = tot_nn_links_one_side[i_inst] ## total nn links one side ##
        len_one_side = tot_len_one_side[i_inst]
        proj_link_relations = tot_proj_link_relations[i_inst]
        nn_links = nn_links_one_side * 2 + 1
        nn_joints = nn_links - 1
        
        print(f"[inst {i_inst}] nn_links_one_side: {nn_links_one_side}, len_one_side: {len_one_side}")
        
        
        
        exp_tag = f"iinst_{i_inst}_nlinks_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{additional_exp_tag}_"
        
        
        obj_info_fn = f"assets/obj_info_n_links_{nn_links_one_side * 2 + 1}_childlinklen_{len_one_side}_{base_xys_tag}.npy"
        
        obj_info_fn = os.path.join(PROJ_ROOT_FOLDER, obj_info_fn)
        
        obj_info = np.load(obj_info_fn, allow_pickle=True).item() ## obj info 
        
        uni_manip_2d_nn_particles = obj_info['particles_xs'].shape[0]
        link_joint_pos = obj_info['link_joint_pos']
        
        link_joint_dir = obj_info['link_joint_dir'] # link # ## obj info ##
        link_parent_idx = obj_info['link_parent_idx'] # link #  ## 
        
        cur_transformation_penalty_term = tot_transformation_penalty_loss[i_inst]
        
        uni_manip_2d = UniManip2D(uni_manip_2d_nn_particles, nn_links, nn_joints, nn_timesteps, dim, dt, obj_info_fn, link_joint_pos, link_joint_dir, link_parent_idx, exp_tag=exp_tag, cur_transformation_penalty_term=cur_transformation_penalty_term, cfg=cfg)
        
        print(f"Calculating the graph connectivity info with nn_links: {nn_links_one_side}")
        # cur_connectivity_info = get_graph_connectivity_info(10)
        # sub_cur_connectivity_info = get_graph_connectivity_info(nn_links_one_side)
        # cur_connectivity_info = np.zeros((21, 21), dtype=np.float32)
        # cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]] = sub_cur_connectivity_info[:sub_cur_connectivity_info.shape[0], :sub_cur_connectivity_info.shape[1]]
        # uni_manip_2d.set_graph_connectivity(cur_connectivity_info)
        

        # connectivity_graph_info = uni_manip_2d.graph.graph_A.to_numpy()
        # print(f"connectivity_graph_info: {connectivity_graph_info}")
        

        print(f"Start optimization!")
        uni_manip_2d.set_goal(goal_x, goal_y)
        uni_manip_2d.initialize()
        uni_manip_2d.set_links_rotational_translational_accs()
        
        
        ## TODO: try two versions: 1) with the inherited optimized results; 2) without the inherited optimized results ##
        ## with the inherited optimized results or without the inherited results ## ## ## -- strateg
        # if (prev_saved_best_ckpt_fn is not None) and inherit_from_prev:
        #     ckpt_five_link_sv_ckpt_fn = prev_saved_best_ckpt_fn

        #     print(f"Loading from {ckpt_five_link_sv_ckpt_fn}")
        #     uni_manip_2d.load_proj_optimized_info(ckpt_five_link_sv_ckpt_fn, proj_link_relations)
        
        uni_manip_2d.eval_res()
        
        # prev_saved_best_ckpt_fn = uni_manip_2d.optimize_with_planning(n_terminate=False)

        # print(f"i_inst: {i_inst}, best saved to {prev_saved_best_ckpt_fn}")


from hydra import compose, initialize
from omegaconf import OmegaConf
# @hydra.main(config_path="cfgs", config_name="config", version_base="1.3")
def main():
    # main_a(cfg)
    
    
    with initialize(version_base="1.3", config_path="cfgs", job_name="test_app"):
        cfg = compose(config_name="eval_config")
        
    random_seed = cfg.seed
    ti_device_memory_fraction = 0.8
    ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=cfg.seed)
    
    
    # cfg = OmegaConf.to_yaml(cfg)
    
    main_b(cfg=cfg)
    


if __name__=='__main__':
    PROJ_ROOT_FOLDER = "/data2/xueyi/softzoo"
    # test_link_gen()
    # exit(0)
    
    # CUDA_VISIBLE_DEVICES=0 python softzoo/engine/uni_manip_2d_v3.py
    
    main()
    exit(0) 
    
    
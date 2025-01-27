from typing import Optional, Union, Tuple, List, Dict, Any, Callable
import numpy as np
import taichi as ti

# import taichi as ti


I_DTYPE = ti.i32
F_DTYPE = ti.f64
NORM_EPS = 1e-8

# from . import I_DTYPE, F_DTYPE
from softzoo.engine.materials import Material
from softzoo.engine.static import Static
from softzoo.engine.primitives import Primitive
from softzoo.tools.taichi_utils import TiDeviceInterface
from softzoo.tools.taichi_utils import clamp_at_zero
from softzoo.tools.general_utils import compute_lame_parameters
from softzoo.configs.item_configs.particle_group_info import get_cfg_defaults as get_particle_group_info_defaults


@ti.data_oriented
class MPMSolver:
    def __init__(self, cfg):
        # Parse configuration
        self.i_dtype = I_DTYPE
        self.f_dtype = F_DTYPE
        self.needs_grad = cfg.needs_grad
        self.use_checkpointing = cfg.use_checkpointing
        self.checkpoint_cache_device = cfg.checkpoint_cache_device
        if not self.needs_grad:
            assert not self.use_checkpointing, 'Checkpointing is not required when needs_grad = False'

        self.dim = cfg.dim
        if isinstance(cfg.padding, (list, tuple)):
            assert len(cfg.padding) == self.dim
            self.padding = ti.Vector(cfg.padding, I_DTYPE)
        else:
            self.padding = ti.Vector([cfg.padding] * self.dim, I_DTYPE)
        self.quality = cfg.quality
        self.n_grid = int(128 * self.quality) # number of grids #
        self.res = tuple([self.n_grid] * self.dim)

        self.dx, self.inv_dx = 1. / self.n_grid, float(self.n_grid) # dx and inv_dx
        self.p_vol = self.dx ** self.dim # grid cell volume

        self.max_num_particles = cfg.max_num_particles
        self.max_substeps = cfg.max_substeps
        self.max_substeps_local = cfg.max_substeps_local
        self.default_dt = cfg.default_dt
        self.max_actuation = cfg.max_actuation
        self.proj_coef = cfg.proj_coef
        self.cfl_max = 0.9 # set to 0 to disable

        # Miscellaneous
        self.sim_t = 0. # simulation time
        self.current_s = 0 # indexing of the current substep
        
        self.cfg = cfg

        self.particle_group_info = { # particles with the same id share the same particle group information
            0: get_particle_group_info_defaults()
        }
        self.particle_group_info[0].E_0 = cfg.E_0 # default Young's modulus
        self.particle_group_info[0].nu_0 = cfg.nu_0 # default Poisson's ratio
        mu_0, lambd_0 = compute_lame_parameters(cfg.E_0, cfg.nu_0) # default Lame parameters
        self.mu_0 = mu_0
        self.lambd_0 = lambd_0
        self.particle_group_info[0].mu_0 = mu_0
        self.particle_group_info[0].lambd_0 = lambd_0
        self.particle_group_info[0].p_rho_0 = cfg.p_rho_0 # particle density

        self.gravity = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=())
        self.n_particles = ti.field(self.i_dtype, shape=())
        
        self.grid_process_static = [] # grid operation for static object like world boundary or surfaces
        self.static_component_info = [] # information of static component; used for rendering
        self.collider_occupancy_kernel = dict() # used for setting occupancy in grid_m from collider kernel
        self.primitives = []

        self.set_gravity(cfg.gravity)

        # Instantiate particles
        self.x = ti.Vector.field(self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # particle position
        self.v = ti.Vector.field(self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # particle velocity
        self.F = ti.Matrix.field(self.dim, self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # deformation gradient
        self.F_tmp = ti.Matrix.field(self.dim, self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad)
        self.C = ti.Matrix.field(self.dim, self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # affine velocity field
        self.Jp = ti.field(dtype=self.f_dtype)

        self.U = ti.Matrix.field(self.dim, self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # U in SVD
        self.sig = ti.Matrix.field(self.dim, self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # sigma in SVD
        self.V = ti.Matrix.field(self.dim, self.dim, dtype=self.f_dtype, needs_grad=self.needs_grad) # V in SVD
        
        self.actuation = ti.field(dtype=self.f_dtype, needs_grad=self.needs_grad) # particle-wise actuation
        self.particle_id_to_act_buffer_id = None # a mapping from particle id to action buffer id (memory allocated at initialize)
        self.act_buffer = ti.field(dtype=self.f_dtype, needs_grad=self.needs_grad) # buffer to store lower-dimensional (actual) actuation
        if cfg.use_dynamic_field:
            act_buffer_alloc = ti.root.dense(ti.i, self.max_substeps_local).dynamic(ti.j, self.max_actuation)
        else:
            act_buffer_alloc = ti.root.dense(ti.ij, (self.max_substeps_local, self.max_actuation))
        act_buffer_alloc.place(self.act_buffer)
        if self.needs_grad:
            act_buffer_alloc.place(self.act_buffer.grad)

        self.mu = ti.field(dtype=self.f_dtype, needs_grad=self.needs_grad) # particle-wise Lame parameter 1 (no E, nu to avoid unsafe division)
        self.lambd = ti.field(dtype=self.f_dtype, needs_grad=self.needs_grad) # particle-wise Lame parameter 2
        self.p_rho = ti.field(dtype=self.f_dtype, needs_grad=self.needs_grad) # particle density

        self.material = ti.field(dtype=self.i_dtype) # material type
        self.particle_ids = ti.field(dtype=self.i_dtype) # particle id

        chunk_size = self.max_num_particles // 2**10 if self.dim == 2 else self.max_num_particles // 2**7

        if cfg.use_dynamic_field:
            self.particle = ti.root.dense(ti.i, self.max_substeps_local + 1).dynamic(ti.j, self.max_num_particles, chunk_size)
        else:
            self.particle = ti.root.dense(ti.ij, (self.max_substeps_local + 1, self.max_num_particles))
        if self.needs_grad:
            self.particle.place(self.x, self.x.grad, self.v, self.v.grad, self.F, self.F.grad, self.C, self.C.grad,
                                self.Jp, self.Jp.grad, self.actuation, self.actuation.grad, self.F_tmp, self.F_tmp.grad,
                                self.U, self.U.grad, self.sig, self.sig.grad, self.V, self.V.grad)
        else:
            self.particle.place(self.x, self.v, self.F, self.C, self.Jp, self.actuation, self.F_tmp, self.U, self.sig, self.V)

        if cfg.use_dynamic_field:
            self.particle_time_invariant = ti.root.dynamic(ti.i, self.max_num_particles, chunk_size)
        else:
            self.particle_time_invariant = ti.root.dense(ti.i, self.max_num_particles)
        if self.needs_grad:
            self.particle_time_invariant.place(self.mu, self.mu.grad, self.lambd, self.lambd.grad, self.p_rho, self.p_rho.grad)
        else:
            self.particle_time_invariant.place(self.mu, self.lambd, self.p_rho)
        self.particle_time_invariant.place(self.material, self.particle_ids)

        # Instantiate grids #
        grid_shape = (self.max_substeps_local + 1,) + self.res
        timestep_shape = (self.max_substeps_local + 1, )
        self.grid_shape = grid_shape
        self.grid_v_in = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad) # grid momentum before grid_op
        self.grid_m = ti.field(dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad) # grid mass
        self.grid_v_out = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad) # grid velocity after grid_op
        # self.grid_v_out_proj = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad) 
        
        ## set up the grid actuation forces ##
        # grid act forces 
        self.grid_act_forces = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad)
        self.grid_act_forces_out = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad)
        ###### articulated rigid body -> should set the particle to group (body) correspondence relations ######
        ## number of joints, and joint positions ##
        self.joint_nn = 1
        self.body_nn = 2
        self.joint_pos_shape = (self.joint_nn, )
        ## ``constant'' variables for storing joints ##
        self.joint_pos = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=self.joint_pos_shape, needs_grad=False)
        self.joint_dir = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=self.joint_pos_shape, needs_grad=False)
        self.joint_particle_index = ti.Vector.field(2, dtype=self.i_dtype, shape=(self.joint_nn, ), needs_grad=False) # 
        ## particle index to body index ## ## body to the the joint index ##
        self.particle_body_index = ti.field(dtype=self.i_dtype, shape=(self.max_num_particles, ), needs_grad=False)
        self.body_parent_joint_index = ti.field(dtype=self.i_dtype, shape=(self.body_nn, ), needs_grad=False ) ## body to the joint index ##
        ### TODO: grid max particle weight should be reset to zeros befoe each simulation cycle ###
        ##### we should set the weight and body index in the p2g process #####
        # grid_max_particle_weight, grid_body_index # 
        self.grid_max_particle_weight = ti.field(dtype=self.f_dtype, shape=grid_shape[1:], needs_grad=False) ## does not need grad ## 
        self.grid_body_index = ti.field(dtype=self.i_dtype, shape=grid_shape[1:], needs_grad=False) ## does not need grad ## # 


        self.grid_sum_angular_vel = ti.field(dtype=self.f_dtype, shape=timestep_shape, needs_grad=self.needs_grad)
        self.grid_sum_angular_m = ti.field(dtype=self.f_dtype, shape=timestep_shape, needs_grad=self.needs_grad)
        self.grid_avg_angular_vel = ti.field(dtype=self.f_dtype, shape=timestep_shape, needs_grad=self.needs_grad)
        
        
        # Functions for material-wise computation
        self.material_models = dict()
        self.active_materials = dict(ids=[], members=[]) # to avoid compile all material models, which can be slow
        self.base_active_materials = cfg.base_active_materials

    ########################################
    ###          Initialization          ###
    ########################################
    def initialize(self):
        # NOTE: call this after everything in the scene is added to the solver

        # Get material models (base active materials + existing particle material)
        base_active_material_ids = [getattr(Material, v).value for v in self.base_active_materials]
        material_ids = -np.ones((self.n_particles[None]), dtype=np.int)
        self.get_material(material_ids)
        active_material_ids = np.unique(material_ids).tolist()
        active_material_ids = list(set(active_material_ids).union(base_active_material_ids))
        active_materials = [v for v in Material.members() if v.value in active_material_ids]
        for material in active_materials:
            compute_F_tmp, p2g = material.get_material_model(self)
            self.material_models[material.value] = { # Only use independent p2g to save memory and compile time
                'compute_F_tmp': compute_F_tmp,
                'p2g': p2g,
            }
        self.active_materials['members'] = active_materials
        self.active_materials['ids'] = active_material_ids

        # Set up actuators
        if not (hasattr(self, 'no_actuators_setup') and getattr(self, 'no_actuators_setup')): # NOTE: allow to be done outside
            self.n_groups = len(self.particle_group_info)
            self.n_actuators = sum([v.active for v in self.particle_group_info.values()])
            assert self.n_actuators <= self.max_actuation, 'Exceed maximal number of actuators'
            
            self.particle_id_to_act_buffer_id = ti.field(dtype=self.i_dtype, shape=(self.n_groups,))
            for k, v in self.particle_group_info.items():
                if v.active:
                    self.particle_id_to_act_buffer_id[k] = k
                else: # act buffer id being -1 to indicate passive (non-active)
                    self.particle_id_to_act_buffer_id[k] = -1
        
        # Gradient checkpointing
        if self.use_checkpointing: # checkpoint cache ##
            self.instantiate_checkpoint_cache()
           
     
    def initialize_particles(self):
        ### only support 2D currently ###
        particle_pos = [
            [0.1, 0.1], [0.2, 0.1], [0.2, 0.4], [0.2, 0.5], [0.5, 0.4], [0.5, 0.5]
        ]
        joint_particle_index = [
            [2, 3]
        ]
        particle_body_index = [
            0, 0, 0, 0, 1, 1
        ]
        body_parent_joint_index = [
            -1, 0
        ]
        self.joint_particle_index.from_numpy(np.array(joint_particle_index))
        self.particle_body_index.from_numpy(np.array(particle_body_index))
        self.body_parent_joint_index.from_numpy(np.array(body_parent_joint_index))
        self.n_particles[None] = len(particle_pos)
        for i, pos in enumerate(particle_pos):
            # self.x[0, i].from_numpy(np.array(pos))
            self.x[0, i] = ti.Vector(pos)
            # self.v[0, i].from_numpy(np.zeros(2))
            self.v[0, i] = ti.Vector([0, 0])
            # self.F[0, i].from_numpy(np.eye(2))
            self.F[0, i] = ti.Matrix([[1, 0], [0, 1]])
            # self.C[0, i].from_numpy(np.eye(2))
            self.C[0, i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[0, i] = 1.
            self.actuation[0, i] = 0.
            self.mu.fill(0.)
            self.lambd.fill(0.)
            self.p_rho.fill(1.)
            self.material.fill(0)
            self.particle_ids.fill(0)
            
    @staticmethod
    def get_particles_dense_nn(xx_nn):
        particle_pos = [
            [0.1, 0.1], [0.2, 0.1], [0.2, 0.4], [0.2, 0.5], [0.5, 0.4], [0.5, 0.5]
        ]
        joint_particle_index = [
            [2, 3]
        ]
        particle_body_index = [
            0, 0, 0, 0, 1, 1
        ]
        body_parent_joint_index = [
            -1, 0
        ]
        
        particle_xx_nn = xx_nn
        particle_dx = (0.5 - 0.2) / float(particle_xx_nn)
        particle_yy_nn = (0.5 - 0.4) / particle_dx
        particle_yy_nn = int(particle_yy_nn)
        particle_xs = np.linspace(0.2 + particle_dx, 0.5, particle_xx_nn)
        particle_ys = np.linspace(0.4 + particle_dx, 0.5, particle_yy_nn)
        # 
        
        nn = len(particle_xs) * len(particle_ys) + len(particle_pos)
        return nn - 2
            
    def initialize_particles_dense(self, xx_nn):
        ### only support 2D currently ###
        # particle_pos = [
        #     [0.1, 0.1], [0.2, 0.1], [0.2, 0.4], [0.2, 0.5], [0.5, 0.4], [0.5, 0.5]
        # ]
        # joint_particle_index = [
        #     [2, 3]
        # ]
        # particle_body_index = [
        #     0, 0, 0, 0, 1, 1
        # ]
        # body_parent_joint_index = [
        #     -1, 0
        # ]
        
        particle_pos = [
            [0.2, 0.4], [0.2, 0.5], [0.5, 0.4], [0.5, 0.5]
        ]
        joint_particle_index = [
            [0, 1]
        ]
        particle_body_index = [
            1, 1, 1, 1
        ]
        body_parent_joint_index = [
            -1, 0
        ]
        self.joint_particle_index.from_numpy(np.array(joint_particle_index))
        
        self.body_parent_joint_index.from_numpy(np.array(body_parent_joint_index))
        
        for i, pos in enumerate(particle_pos):
            # self.x[0, i].from_numpy(np.array(pos))
            self.x[0, i] = ti.Vector(pos)
            # self.v[0, i].from_numpy(np.zeros(2))
            self.v[0, i] = ti.Vector([0, 0])
            # self.F[0, i].from_numpy(np.eye(2))
            self.F[0, i] = ti.Matrix([[1, 0], [0, 1]])
            # self.C[0, i].from_numpy(np.eye(2))
            self.C[0, i] = ti.Matrix([[1, 0], [0, 1]])
            self.Jp[0, i] = 1.
            self.actuation[0, i] = 0.
            self.mu.fill(self.mu_0)
            self.lambd.fill(self.lambd_0)
            self.p_rho.fill(self.cfg.p_rho_0)
            self.material.fill(0)
            self.particle_ids.fill(0)
        
        particle_xx_nn = xx_nn
        particle_dx = (0.5 - 0.2) / float(particle_xx_nn)
        particle_yy_nn = (0.5 - 0.4) / particle_dx
        particle_yy_nn = int(particle_yy_nn)
        particle_xs = np.linspace(0.2 + particle_dx, 0.5, particle_xx_nn)
        particle_ys = np.linspace(0.4 + particle_dx, 0.5, particle_yy_nn)
        # 
        cur_n = len(particle_pos)
        for i_x, x in enumerate(particle_xs):
            for i_y, y in enumerate(particle_ys):
                pos = [x, y]
                self.x[0, cur_n] = ti.Vector(pos)
                self.v[0, cur_n] = ti.Vector([0, 0])
                
                # self.F[0, i].from_numpy(np.eye(2))
                self.F[0, cur_n] = ti.Matrix([[1, 0], [0, 1]])
                # self.C[0, i].from_numpy(np.eye(2))
                self.C[0, cur_n] = ti.Matrix([[1, 0], [0, 1]])
                self.Jp[0, cur_n] = 1.
                self.actuation[0, cur_n] = 0.
                particle_body_index.append(1)
                cur_n += 1
                
            # self.mu.fill(0.)
            # self.lambd.fill(0.)
            # self.p_rho.fill(1.)
            # self.material.fill(0)
            # self.particle_ids.fill(0)
        self.particle_body_index.from_numpy(np.array(particle_body_index))
        self.n_particles[None] = len(particle_body_index)
        return cur_n
        
        
            
    @ti.kernel
    def initialize_actuation_forces(self, s: I_DTYPE):
        
        # # grid_act_forces # 
        for I in ti.grouped(ti.ndrange(*self.res)): 
            # self.grid_act_forces[s, I] = ti.Vector.zero(self.f_dtype, self.dim)
            self.grid_act_forces[s, I] = ti.Vector([2, -2])
        
    def initialize_actuation_forces_all(self):
        for s in range(self.max_substeps_local + 1):
            self.initialize_actuation_forces(s)
        

    def set_gravity(self, g: Union[Tuple, List]):
        assert len(g) == self.dim
        self.gravity[None] = g

    def add(self,
            item: Any,
            item_cfg: Optional[Any] = None):
        if isinstance(item, str):
            cls_name, type_name = item.split('.')
            item = getattr(globals()[cls_name], type_name)

        if Static.is_member(item) or Primitive.is_member(item):
            item.add(solver=self, cfg=item_cfg)
        else:
            raise ValueError(f'Unrecognized item {item}')

    def get_latest_s(self, consider_cycle=True):
        latest_s = self.current_s
        if consider_cycle:
            latest_s = self.get_cyclic_s(latest_s)
        return latest_s
    
    def get_cyclic_s(self, s):
        return s % self.max_substeps_local # substeps local # 

    @property
    def current_s_local(self):
        return self.get_cyclic_s(self.current_s)
    ########################################

    ########################################
    ###            Simulation            ###
    ########################################
    def step(self,
             action=None,
             frame_dt: Optional[float] = 8e-3):
        n_substeps = np.ceil(frame_dt / self.default_dt)
        dt = frame_dt / n_substeps

        frame_time_left = frame_dt
        while frame_time_left > 0:
            frame_time_left -= dt
            if self.use_checkpointing:
                self.record_action_buffer(action, dt)
            self.substep(action, dt)
            if not self.needs_grad:
                self.current_s = self.current_s % self.max_substeps
            # assert self.current_s < self.max_substeps, f'Current substep {self.current_s} exceeding maximal substeps {self.max_substeps}'

    def step_new(self,
             action=None,
             frame_dt: Optional[float] = 8e-3):
        n_substeps = np.ceil(frame_dt / self.default_dt)
        dt = frame_dt / n_substeps

        frame_time_left = frame_dt
        while frame_time_left > 0:
            frame_time_left -= dt
            # if self.use_checkpointing:
            #     self.record_action_buffer(action, dt)
            self.substep(None, dt)
            if not self.needs_grad:
                self.current_s = self.current_s % self.max_substeps
            # assert self.current_s < self.max_substeps, f'Current substep {self.current_s} exceeding maximal substeps {self.max_substeps}'

    


    @ti.ad.grad_replaced
    def substep(self, action: Any, dt: float, recompute: Optional[bool] = False):
        s = self.current_s_local

        self.clear_grid(s)
        # self.grid_sum_angular_vel.fill(0.0)
        # self.grid_sum_angular_m.fill(0.0)

        # if action is not None:
        #     self.set_act_buffer(s, action) 
        # self.sync_actuation_with_buffer(s) # 
        self.compute_F_tmp(s, dt) # compute F tmp # 
        self.svd(s)
        self.p2g(s, dt) # 
        for primitive in self.primitives:
            if primitive.is_rigid:
                primitive.forward_kinematics(s, dt)
                
        # update_joint_position_and_direction
        # project_grid_actuation_forces
        self.update_joint_position_and_direction(s)
        self.project_grid_actuation_forces(s)
        
        
        
        ''' Grid OP with projection '''
        # self.grid_op(s, dt)
        
        # # grid_avg_angular_vel = self.grid_sum_angular_vel[None] / (self.grid_sum_angular_m[None] + 1e-6)
        # # self.grid_avg_angular_vel[None] = grid_avg_angular_vel
        
        # self.grid_average_angular_vel(s)
        
        # self.grid_angular_proj(s, dt)
        ''' Grid OP with projection '''
        
        self.grid_op_nproj(s, dt=dt)
        
        
        
        self.g2p(s, dt)
        
        ## and the similar angular velocities ##

        self.current_s += 1
        if not recompute:
            self.sim_t += dt

            # if self.current_s_local == 0: 
            #     if self.use_checkpointing:
            #         self.sim_to_checkpoint_cache()
            #     else:
            #         self.copy_substep(self.max_substeps_local, 0) # cycle the first step

    @ti.ad.grad_for(substep)
    def substep_grad(self, action: Any, dt: float, pre_grad_fn: Optional[Callable] = None,
                     compute_grad: Optional[bool] = True):
        # if self.current_s_local == 0 and self.use_checkpointing:
        #     self.sim_from_checkpoint_cache()

        if pre_grad_fn is not None:
            pre_grad_fn()

        self.current_s -= 1
        s = self.current_s_local

        if compute_grad:
            self.g2p.grad(s, dt)
            grad_v_out_grad = self.grid_v_out.grad.to_numpy()
            # print(f"s: {s}, grad_v_out_grad: {np.sum(grad_v_out_grad)}")
            
            self.grid_angular_proj.grad(s, dt)
            self.grid_average_angular_vel.grad(s)
            
            self.grid_op.grad(s, dt)
            grid_act_forces_out_grad = self.grid_act_forces_out.grad.to_numpy()
            # print(f"s: {s}, grid_act_forces_out_grad: {np.sum(grid_act_forces_out_grad)}")
            self.project_grid_actuation_forces.grad(s)
            grid_act_forces_grad = self.grid_act_forces.grad.to_numpy()
            # print(f"s: {s}, grid_act_forces_grad: {np.sum(grid_act_forces_grad)}")
            self.update_joint_position_and_direction.grad(s)
            for i in range(len(self.primitives) - 1, -1, -1):
                primitive = self.primitives[i]
                if primitive.is_rigid:
                    primitive.forward_kinematics.grad(s, dt)
            
            self.p2g.grad(s, dt)
            self.svd_grad(s)
            self.compute_F_tmp.grad(s, dt)
            # self.sync_actuation_with_buffer.grad(s)

    ########################################

    ########################################
    ###             MPM                  ###
    ########################################
    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))
    
    @ti.kernel
    def compute_F_tmp(self, s: I_DTYPE, dt: F_DTYPE):
        for p in range(self.n_particles[None]):
            p_mass = self.p_vol * self.p_rho[p]

            if p_mass > 0: # NOTE: allow zero-mass particle
                for material_id in ti.static(self.active_materials['ids']):
                    if self.material[p] == material_id:
                        self.material_models[material_id]['compute_F_tmp'](p, s, dt, self.F, self.F_tmp, self.Jp, self.C)

    @ti.kernel
    def p2g(self, s: I_DTYPE, dt: F_DTYPE):
        for p in range(self.n_particles[None]):
            p_mass = self.p_vol * self.p_rho[p] # volume * density #

            if p_mass > 0: # NOTE: allow zero-mass particle
                next_s = s + 1 # NOTE: j
                stress = ti.Matrix.zero(F_DTYPE, self.dim, self.dim)
                for material_id in ti.static(self.active_materials['ids']):
                    if self.material[p] == material_id:
                        stress = self.material_models[material_id]['p2g'](p, s, next_s, dt, p_mass, self.x, self.v, self.F, self.C, self.Jp,
                                                                          self.F_tmp, self.U, self.sig, self.V, self.actuation, self.mu,
                                                                          self.lambd, self.particle_ids, self.grid_v_in, self.grid_m)
                        
                # To grid (maximally share the code across materials to save compile time and memory)
                mass = p_mass
                affine = stress + mass * self.C[s, p] # the affine matrix = stress + mass * C # stress + mass * C #

                base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
                fx = self.x[s, p] * self.inv_dx - base.cast(F_DTYPE)
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(F_DTYPE) - fx) * self.dx
                    weight = ti.cast(1., F_DTYPE)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    self.grid_v_in[s, base + offset] += weight * (mass * self.v[s, p] + affine @ dpos)
                    self.grid_m[s, base + offset] += weight * mass ## weight and mass ##
                    
                    if weight * mass > self.grid_max_particle_weight[base + offset]:
                        self.grid_max_particle_weight[base + offset] = weight * mass
                        self.grid_body_index[base + offset] = self.particle_body_index[p]
                    
                ## get grid m and grid v in ##
    # update_joint_position_and_direction
    # project_grid_actuation_forces
    @ti.kernel
    def update_joint_position_and_direction(self, s: I_DTYPE):
        for j in range(self.joint_nn):
            body_index = self.joint_particle_index[j][0]
            joint_index = self.joint_particle_index[j][1]
            joint_pos = self.x[s, joint_index]
            joint_dir = self.x[s, body_index]- self.x[s, joint_index]
            self.joint_pos[j] = joint_pos + 0.5 * joint_dir # get joint positions # 
            
            self.joint_pos[j] = ti.Vector([0.2, 0.45])
            self.joint_dir[j] = joint_dir.normalized() # get joint directions #
                
    ##### strategy 1: greedy local force projection #####
    @ti.kernel
    def project_grid_actuation_forces(self, s: I_DTYPE):
        
        for I in ti.grouped(ti.ndrange(*self.res)):
            
            grid_body_index = self.grid_body_index[I]
            # for the joint index and joint position ? #
            body_parent_joint_index = self.body_parent_joint_index[grid_body_index]
            
            proj_act_force = ti.Vector.zero(F_DTYPE, self.dim)
            if body_parent_joint_index > -1: # joint index #
                joint_pos = self.joint_pos[body_parent_joint_index]
                # ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE) #
                grid_pos = I.cast(F_DTYPE) * self.dx
                joint_dir = self.joint_dir[body_parent_joint_index]
                joint_pos_to_grid = grid_pos - joint_pos
                
                joint_pos_to_grid_len = ti.sqrt(joint_pos_to_grid[0] ** 2 + joint_pos_to_grid[1] ** 2)
                
                ## ## ## ## 
                # proj_joint_pos_to_grid_to_joint_dir = ti.math.dot(joint_pos_to_grid, joint_dir) * joint_dir
                # joint_pos_to_grid = joint_pos_to_grid - proj_joint_pos_to_grid_to_joint_dir
                
                
                joint_pos_to_grid = joint_pos_to_grid.normalized()
                
                cur_grid_act_force = self.grid_act_forces[s, I]
                mult_joint_pos_to_grid_w_act_force = joint_pos_to_grid * cur_grid_act_force
                dot_joint_pos_to_grid_w_act_force = mult_joint_pos_to_grid_w_act_force[0] + mult_joint_pos_to_grid_w_act_force[1]
                proj_act_force = cur_grid_act_force - dot_joint_pos_to_grid_w_act_force * joint_pos_to_grid
                
                # just for fun # # just for fun # #
                ### project the actuation force for teh same angular velocity ###
                proj_act_force_dir = proj_act_force.normalized()
                
                proj_act_force_dir = ti.Vector([joint_pos_to_grid[1], -joint_pos_to_grid[0]], F_DTYPE)
                proj_act_force_len = joint_pos_to_grid_len * 2000 * self.grid_m[s, I]
                
                proj_act_force = (self.proj_coef) * (proj_act_force_dir * proj_act_force_len) + (1.0 - self.proj_coef) * cur_grid_act_force * 0.01
                
                # proj_act_force = ti.Vector([2.0, -2.0]) * 10
                ### project the actuation force for teh same angular velocity ###
                
                ####### get proj_act_force --- strategy 1 #######
                # joint_dir = joint_dir.normalized()
                # joint_act_proj_direction = ti.math.cross(joint_dir, joint_pos_to_grid)
                
                # cur_grid_act_force = self.grid_act_forces[s, I]
                # # dot_grid_act_force_w_proj_dir = ti.math.dot(cur_grid_act_force, joint_act_proj_direction)
                
                # dot_grid_act_force_w_proj_dir = cur_grid_act_force * joint_act_proj_direction
                # dot_grid_act_force_w_proj_dir = dot_grid_act_force_w_proj_dir[0] + dot_grid_act_force_w_proj_dir[1]
                
                # proj_act_force = dot_grid_act_force_w_proj_dir * joint_act_proj_direction
                ####### get proj_act_force --- strategy 1 #######
                
            # else:
            #     proj_act_force = ti.Vector.zero(F_DTYPE, self.dim)
            self.grid_act_forces_out[s, I] = proj_act_force
            # self.grid_act_forces_out[s, I] = self.grid_act_forces[s, I]

    @ti.kernel
    def grid_op(self, s: I_DTYPE, dt: F_DTYPE):
        for I in ti.grouped(ti.ndrange(*self.res)): # this timestep
            
            # Apply gravity # 
            v_out = ti.Vector.zero(F_DTYPE, self.dim)
            if self.grid_m[s, I] > 0: # no need for epsilon here #
                v_out = (1 / self.grid_m[s, I]) * self.grid_v_in[s, I] # momentum to velocity #
                v_out += dt * self.gravity[None]  ## no gravity 
                # v_out #
                
                
                v_out += dt * self.grid_act_forces_out[s, I] / self.grid_m[s, I] # actuation force
            
            

            # # Apply collider of static item #
            # for i in ti.static(range(len(self.grid_process_static))):
            #     v_out, _ = self.grid_process_static[i](dt, I, v_out, self.grid_m[s, I], s)
                
            # # Apply collider of rigid body TODO: handle collision to static component
            # for i in ti.static(range(len(self.primitives))):
            #     if ti.static(self.primitives[i].is_rigid):
            #         v_out, _ = self.primitives[i].collide(dt, I, v_out, self.grid_m[s, I], s, self.gravity[None])
                    
            # proj the v_out ##
            grid_body_index = self.grid_body_index[I]
            # for the joint index and joint position ? #
            body_parent_joint_index = self.body_parent_joint_index[grid_body_index]
            
            if body_parent_joint_index > -1:
                joint_pos = self.joint_pos[body_parent_joint_index]
                # ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
                grid_pos = I.cast(F_DTYPE) * self.dx
                joint_dir = self.joint_dir[body_parent_joint_index]
                
                
                
                joint_pos_to_grid = grid_pos - joint_pos
                
                len_joint_pos_to_grid = ti.sqrt(joint_pos_to_grid[0] ** 2 + joint_pos_to_grid[1] ** 2)
                
                joint_pos_to_grid = joint_pos_to_grid.normalized()
                mult_joint_pos_to_grid_w_vout = joint_pos_to_grid * v_out
                dot_joint_pos_to_grid_w_vout = mult_joint_pos_to_grid_w_vout[0] + mult_joint_pos_to_grid_w_vout[1]
                v_out = v_out - dot_joint_pos_to_grid_w_vout * joint_pos_to_grid
                
                v_out_len = ti.sqrt(v_out[0] ** 2 + v_out[1] ** 2)
                
                angular_out = v_out_len / (len_joint_pos_to_grid + 1e-6)
                self.grid_sum_angular_vel[s] += angular_out * self.grid_m[s, I]
                self.grid_sum_angular_m[s] += self.grid_m[s, I]                
                
                ######## project v_out -- strategy 1 ########
                # proj_joint_pos_to_grid_to_joint_dir = ti.math.dot(joint_pos_to_grid, joint_dir) * joint_dir
                # joint_pos_to_grid = joint_pos_to_grid - proj_joint_pos_to_grid_to_joint_dir
                
                # joint_pos_to_grid = joint_pos_to_grid.normalized()
                # joint_dir = joint_dir.normalized()
                # joint_act_proj_direction = ti.math.cross(joint_dir, joint_pos_to_grid)
                # # print("voutt", v_out, joint_act_proj_direction)
                # # dot_v_out_w_proj_dir = ti.math.dot(v_out, joint_act_proj_direction)
                # dot_v_out_w_proj_dir = v_out * joint_act_proj_direction
                # # dot_v_out_w_proj_dir 
                # dot_v_out_w_proj_dir = dot_v_out_w_proj_dir[0] + dot_v_out_w_proj_dir[1]
                # # ti.math.su
                # v_out = dot_v_out_w_proj_dir * joint_act_proj_direction
                ######## project v_out -- strategy 1 ########
            # proj the v_out ##
            else:
                v_out = ti.Vector.zero(F_DTYPE, self.dim) # fixed grid point #


            # if ti.static(self.cfl_max > 0): # ctl_max #
            #     v_allowed = self.dx * self.cfl_max / dt
            #     v_out = ti.min(ti.max(v_out, -v_allowed), v_allowed)

            # self.grid_v_out[s, I] = v_out
    
    @ti.kernel
    def grid_op_nproj(self, s: I_DTYPE, dt: F_DTYPE):
        for I in ti.grouped(ti.ndrange(*self.res)): # this timestep
            
            # Apply gravity # 
            v_out = ti.Vector.zero(F_DTYPE, self.dim)
            if self.grid_m[s, I] > 0: # no need for epsilon here #
                v_out = (1 / self.grid_m[s, I]) * self.grid_v_in[s, I] # momentum to velocity #
                v_out += dt * self.gravity[None]  ## no gravity 
                # v_out #
                
                
                v_out += dt * self.grid_act_forces_out[s, I] / self.grid_m[s, I] # actuation force
            
            

            # # Apply collider of static item #
            # for i in ti.static(range(len(self.grid_process_static))):
            #     v_out, _ = self.grid_process_static[i](dt, I, v_out, self.grid_m[s, I], s)
                
            # # Apply collider of rigid body TODO: handle collision to static component
            # for i in ti.static(range(len(self.primitives))):
            #     if ti.static(self.primitives[i].is_rigid):
            #         v_out, _ = self.primitives[i].collide(dt, I, v_out, self.grid_m[s, I], s, self.gravity[None])
                    
            ''' Ori: project the output velocity '''
            # # proj the v_out ##
            # grid_body_index = self.grid_body_index[I]
            # # for the joint index and joint position ? #
            # body_parent_joint_index = self.body_parent_joint_index[grid_body_index]
            
            # if body_parent_joint_index > -1:
            #     joint_pos = self.joint_pos[body_parent_joint_index]
            #     # ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
            #     grid_pos = I.cast(F_DTYPE) * self.dx
            #     joint_dir = self.joint_dir[body_parent_joint_index]
                
                
                
            #     joint_pos_to_grid = grid_pos - joint_pos
                
            #     len_joint_pos_to_grid = ti.sqrt(joint_pos_to_grid[0] ** 2 + joint_pos_to_grid[1] ** 2)
                
            #     joint_pos_to_grid = joint_pos_to_grid.normalized()
            #     mult_joint_pos_to_grid_w_vout = joint_pos_to_grid * v_out
            #     dot_joint_pos_to_grid_w_vout = mult_joint_pos_to_grid_w_vout[0] + mult_joint_pos_to_grid_w_vout[1]
            #     v_out = v_out - dot_joint_pos_to_grid_w_vout * joint_pos_to_grid
                
            #     v_out_len = ti.sqrt(v_out[0] ** 2 + v_out[1] ** 2)
                
            #     angular_out = v_out_len / (len_joint_pos_to_grid + 1e-6)
            #     self.grid_sum_angular_vel[s] += angular_out * self.grid_m[s, I]
            #     self.grid_sum_angular_m[s] += self.grid_m[s, I]                
                
            #     ######## project v_out -- strategy 1 ########
            #     # proj_joint_pos_to_grid_to_joint_dir = ti.math.dot(joint_pos_to_grid, joint_dir) * joint_dir
            #     # joint_pos_to_grid = joint_pos_to_grid - proj_joint_pos_to_grid_to_joint_dir
                
            #     # joint_pos_to_grid = joint_pos_to_grid.normalized()
            #     # joint_dir = joint_dir.normalized()
            #     # joint_act_proj_direction = ti.math.cross(joint_dir, joint_pos_to_grid)
            #     # # print("voutt", v_out, joint_act_proj_direction)
            #     # # dot_v_out_w_proj_dir = ti.math.dot(v_out, joint_act_proj_direction)
            #     # dot_v_out_w_proj_dir = v_out * joint_act_proj_direction
            #     # # dot_v_out_w_proj_dir 
            #     # dot_v_out_w_proj_dir = dot_v_out_w_proj_dir[0] + dot_v_out_w_proj_dir[1]
            #     # # ti.math.su
            #     # v_out = dot_v_out_w_proj_dir * joint_act_proj_direction
            #     ######## project v_out -- strategy 1 ########
            # # proj the v_out ##
            # else:
            #     v_out = ti.Vector.zero(F_DTYPE, self.dim) # fixed grid point #
            ''' Ori: project the output velocity '''

            # if ti.static(self.cfl_max > 0): # ctl_max #
            #     v_allowed = self.dx * self.cfl_max / dt
            #     v_out = ti.min(ti.max(v_out, -v_allowed), v_allowed)

            self.grid_v_out[s, I] = v_out
    
    
    @ti.kernel
    def grid_average_angular_vel(self, s: I_DTYPE):
        grid_avg_angular_vel = self.grid_sum_angular_vel[s] / (self.grid_sum_angular_m[s] + 1e-6)
        grid_avg_angular_vel = ti.min(100, grid_avg_angular_vel)
        self.grid_avg_angular_vel[s] = grid_avg_angular_vel
        
    
    @ti.kernel
    def grid_angular_proj(self, s: I_DTYPE, dt: F_DTYPE):
        for I in ti.grouped(ti.ndrange(*self.res)): # this timestep
            
            # # Apply gravity # 
            v_out = ti.Vector.zero(F_DTYPE, self.dim)
            if self.grid_m[s, I] > 0: # no need for epsilon here #
                v_out = (1 / self.grid_m[s, I]) * self.grid_v_in[s, I] # momentum to velocity #
                v_out += dt * self.gravity[None]  ## no gravity 
                # v_out #
                
                
                v_out += dt * self.grid_act_forces_out[s, I] / self.grid_m[s, I] # actuation force
            
            

            # # Apply collider of static item #
            # for i in ti.static(range(len(self.grid_process_static))):
            #     v_out, _ = self.grid_process_static[i](dt, I, v_out, self.grid_m[s, I], s)
                
            # # Apply collider of rigid body TODO: handle collision to static component
            # for i in ti.static(range(len(self.primitives))):
            #     if ti.static(self.primitives[i].is_rigid):
            #         v_out, _ = self.primitives[i].collide(dt, I, v_out, self.grid_m[s, I], s, self.gravity[None])
            ## proj the v_out ##
            # proj the v_out ##
            grid_body_index = self.grid_body_index[I]
            # for the joint index and joint position ? #
            body_parent_joint_index = self.body_parent_joint_index[grid_body_index]
            
            if body_parent_joint_index > -1:
                # joint_pos = self.joint_pos[body_parent_joint_index]
                # # ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
                # grid_pos = I.cast(F_DTYPE) * self.dx
                # joint_dir = self.joint_dir[body_parent_joint_index]
                
                
                
                # joint_pos_to_grid = grid_pos - joint_pos
                # len_joint_pos_to_grid = ti.sqrt(joint_pos_to_grid[0] ** 2 + joint_pos_to_grid[1] ** 2)
                
                # v_out = len_joint_pos_to_grid * self.grid_avg_angular_vel[s]
                
                
                # joint_pos_to_grid = joint_pos_to_grid.normalized()
                # mult_joint_pos_to_grid_w_vout = joint_pos_to_grid * v_out
                # dot_joint_pos_to_grid_w_vout = mult_joint_pos_to_grid_w_vout[0] + mult_joint_pos_to_grid_w_vout[1]
                # v_out = v_out - dot_joint_pos_to_grid_w_vout * joint_pos_to_grid
                
                
                joint_pos = self.joint_pos[body_parent_joint_index]
                # ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
                grid_pos = I.cast(F_DTYPE) * self.dx
                joint_dir = self.joint_dir[body_parent_joint_index]
                
                
                
                joint_pos_to_grid = grid_pos - joint_pos
                ## joint pos to grid ##
                
                len_joint_pos_to_grid = ti.sqrt(joint_pos_to_grid[0] ** 2 + joint_pos_to_grid[1] ** 2)
                
                joint_pos_to_grid = joint_pos_to_grid.normalized()
                mult_joint_pos_to_grid_w_vout = joint_pos_to_grid * v_out
                dot_joint_pos_to_grid_w_vout = mult_joint_pos_to_grid_w_vout[0] + mult_joint_pos_to_grid_w_vout[1]
                v_out = v_out - dot_joint_pos_to_grid_w_vout * joint_pos_to_grid
                
                v_out_dir = v_out.normalized()
                
                v_out_dir = ti.Vector([joint_pos_to_grid[1], -joint_pos_to_grid[0]])
                v_out = v_out_dir * self.grid_avg_angular_vel[s] * len_joint_pos_to_grid ## v out ad v out dir ##
                
                # v_out = v_out_dir * 100 * len_joint_pos_to_grid
                # v_out_len = ti.sqrt(v_out[0] ** 2 + v_out[1] ** 2)
                
                # angular_out = v_out_len / (len_joint_pos_to_grid + 1e-6)
                # self.grid_sum_angular_vel[s] += angular_out * self.grid_m[s, I]
                # self.grid_sum_angular_m[s] += self.grid_m[s, I]                
                
                
                
                
                # angular_out = v_out / (len_joint_pos_to_grid + 1e-6)
                # self.grid_sum_angular_vel[None] += angular_out * self.grid_m[s, I]
                # self.grid_sum_angular_m[None] += self.grid_m[s, I]                
                
                ######## project v_out -- strategy 1 ########
                # proj_joint_pos_to_grid_to_joint_dir = ti.math.dot(joint_pos_to_grid, joint_dir) * joint_dir
                # joint_pos_to_grid = joint_pos_to_grid - proj_joint_pos_to_grid_to_joint_dir
                
                # joint_pos_to_grid = joint_pos_to_grid.normalized()
                # joint_dir = joint_dir.normalized()
                # joint_act_proj_direction = ti.math.cross(joint_dir, joint_pos_to_grid)
                # # print("voutt", v_out, joint_act_proj_direction)
                # # dot_v_out_w_proj_dir = ti.math.dot(v_out, joint_act_proj_direction)
                # dot_v_out_w_proj_dir = v_out * joint_act_proj_direction
                # # dot_v_out_w_proj_dir 
                # dot_v_out_w_proj_dir = dot_v_out_w_proj_dir[0] + dot_v_out_w_proj_dir[1]
                # # ti.math.su
                # v_out = dot_v_out_w_proj_dir * joint_act_proj_direction
                ######## project v_out -- strategy 1 ########
            # proj the v_out ##
            else:
                v_out = ti.Vector.zero(F_DTYPE, self.dim) # fixed grid point #


            # if ti.static(self.cfl_max > 0): # ctl_max #
            #     v_allowed = self.dx * self.cfl_max / dt
            #     v_out = ti.min(ti.max(v_out, -v_allowed), v_allowed)

            self.grid_v_out[s, I] = v_out
    
    

    @ti.kernel
    def g2p(self, s: I_DTYPE, dt: F_DTYPE):
        for p in range(self.n_particles[None]):
            p_mass = self.p_vol * self.p_rho[p]

            if p_mass > 0: # NOTE: allow zero-mass particle # # allow zero-mass particles ##
                ### allow zero mass ### # # ## allow ## allow ##
                
                ## but ## actuation 
                ## virtual work; F (\delta x
                # # actuation force space -> for the force field #
                # actuation 
                # mpm --- if we simply rely on the mpm and only actuate 
                # 
                # a systemetic representation of the grid to particle operation ##
                # grid to particle representations #
                # grid-and-particle representations #
                ## allow ## ## controlling, projective controling 
                next_s = s + 1
  
                base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
                fx = self.x[s, p] * self.inv_dx - base.cast(F_DTYPE)
                w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]

                new_v = ti.Vector.zero(F_DTYPE, self.dim)
                new_C = ti.Matrix.zero(F_DTYPE, self.dim, self.dim)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(F_DTYPE) - fx
                    g_v = self.grid_v_out[s, base + offset]
                    weight = ti.cast(1., F_DTYPE)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos) # 
                
                ## if we ## 
                ## ##
                ## if we proejct v directly? --- not a simulation process, right? ## ##
                joint_pos = self.joint_pos[0]
                joint_pos_to_particle_pos = self.x[s, p] - joint_pos # get the joint pos to particle pos vector ##
                joint_pos_to_particle_pos_len = ti.sqrt(joint_pos_to_particle_pos[0] ** 2 + joint_pos_to_particle_pos[1] ** 2)
                
                dir_joint_pos_to_particle_pos = joint_pos_to_particle_pos.normalized()
                new_v_dir = ti.Vector([dir_joint_pos_to_particle_pos[1], -dir_joint_pos_to_particle_pos[0]])
                new_v_len = ti.sqrt(new_v[0] ** 2 + new_v[1] ** 2) ## get the new v length ##
                
                
                ###### project new _v ####
                # new_v = new_v_dir * new_v_len ## no more ##
                ###### project new _v ####
                
                # new_v_len = joint_pos_to_particle_pos_len * 100 # self.grid_avg_angular_vel[s] 
                # new_v = new_v_dir * new_v_len
                
                # new_v = new_v_dir * joint_pos_to_particle_pos_len * self.grid_avg_angular_vel[s] # ## get the
                
                
                self.v[next_s, p], self.C[next_s, p] = new_v, new_C
                
                self.x[next_s, p] = self.x[s, p] + dt * self.v[next_s, p]

    @ti.kernel
    def svd(self, s: I_DTYPE):
        for p in range(self.n_particles[None]):
            self.U[s, p], self.sig[s, p], self.V[s, p] = ti.svd(self.F_tmp[s, p], dt=self.f_dtype)

    @ti.kernel
    def svd_grad(self, s: I_DTYPE):
        for p in range(self.n_particles[None]):
            u_p = self.U[s, p]
            sig_p = self.sig[s, p]
            v_p = self.V[s, p]
            gu_p = self.U.grad[s, p]
            gsig_p = self.sig.grad[s, p]
            gv_p = self.V.grad[s, p]

            ut_p = u_p.transpose()
            vt_p = v_p.transpose()
            sig_term = u_p @ gsig_p @ vt_p

            s_ = ti.Vector.zero(self.f_dtype, self.dim)
            if ti.static(self.dim == 2):
                s_ = ti.Vector([sig_p[0, 0], sig_p[1, 1]]) ** 2
            else:
                s_ = ti.Vector([sig_p[0, 0], sig_p[1, 1], sig_p[2, 2]]) ** 2
            F = ti.Matrix.zero(self.f_dtype, self.dim, self.dim)
            for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
                if i == j: F[i, j] = 0
                else: F[i, j] = 1. / clamp_at_zero(s_[j] - s_[i])
            u_term = u_p @ ((F * (ut_p @ gu_p - gu_p.transpose() @ u_p)) @ sig_p) @ vt_p
            v_term = u_p @ (sig_p @ ((F * (vt_p @ gv_p - gv_p.transpose() @ v_p)) @ vt_p))

            self.F_tmp.grad[s, p] += (u_term + v_term + sig_term)

    ########################################

    ########################################
    ###         Reset functions          ###
    ########################################
    @ti.kernel
    def clear_grid(self, s: I_DTYPE):
        zero = ti.Vector.zero(F_DTYPE, self.dim)
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.grid_v_in[s, I] = zero
            self.grid_v_out[s, I] = zero
            self.grid_m[s, I] = 0
            self.grid_act_forces_out[s, I] = zero
            self.grid_max_particle_weight[I] = 0
            self.grid_body_index[I] = -1
            if ti.static(self.needs_grad):
                self.grid_v_in.grad[s, I] = zero
                self.grid_v_out.grad[s, I] = zero
                self.grid_m.grad[s, I] = 0
                # grid_max_particle_weight, grid_body_index # 
                

    @ti.kernel
    def clear_particle_grad(self):
        zero_vec = ti.Vector.zero(self.f_dtype, self.dim)
        zero_mat = ti.Matrix.zero(self.f_dtype, self.dim, self.dim)
        for I in ti.grouped(self.x):
            self.x.grad[I] = zero_vec
            self.v.grad[I] = zero_vec
            self.F.grad[I] = zero_mat
            self.C.grad[I] = zero_mat
            self.Jp.grad[I] = 0.

            self.F_tmp.grad[I] = zero_mat
            self.U.grad[I] = zero_mat
            self.V.grad[I] = zero_mat
            self.sig.grad[I] = zero_mat
            
            self.actuation.grad[I] = 0.

            self.mu.grad[I[1]] = 0.
            self.lambd.grad[I[1]] = 0.
            self.p_rho.grad[I[1]] = 0.
            
            

    @ti.kernel
    def clear_act_buffer_grad(self):
        for I in ti.grouped(self.act_buffer):
            self.act_buffer.grad[I] = 0.

    def clear_primitives(self):
        for primitive in self.primitives:
            if primitive.is_rigid:
                primitive.clear()

    def clear_primitives_grad(self):
        for primitive in self.primitives:
            if primitive.is_rigid:
                primitive.clear_grad()

    def reset(self):
        self.x.fill(0.) # NOTE: reset all steps (to fix left-out particles in varying design)
        self.v.fill(0.)
        
        # ## set up the grid actuation forces ##
        # # grid act forces 
        # self.grid_act_forces = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad)
        # self.grid_act_forces_out = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=grid_shape, needs_grad=self.needs_grad)
        # ###### articulated rigid body -> should set the particle to group (body) correspondence relations ######
        # ## number of joints, and joint positions ##
        # self.joint_nn = 1
        # self.body_nn = 2
        # self.joint_pos_shape = (self.joint_nn, )
        # ## ``constant'' variables for storing joints ##
        # self.joint_pos = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=self.joint_pos_shape, needs_grad=False)
        # self.joint_dir = ti.Vector.field(self.dim, dtype=self.f_dtype, shape=self.joint_pos_shape, needs_grad=False)
        # self.joint_particle_index = ti.Vector.field(2, dtype=self.i_dtype, shape=(self.joint_nn, ), needs_grad=False) # 
        # ## particle index to body index ## ## body to the the joint index ##
        # self.particle_body_index = ti.field(dtype=self.i_dtype, shape=(self.max_num_particles, ), needs_grad=False)
        # self.body_parent_joint_index = ti.field(dtype=self.i_dtype, shape=(self.body_nn, ), needs_grad=False ) ## body to the joint index ##
        # ### TODO: grid max particle weight should be reset to zeros befoe each simulation cycle ###
        # ##### we should set the weight and body index in the p2g process #####
        # 
        # self.grid_max_particle_weight = ti.field(dtype=self.f_dtype, shape=grid_shape[1:], needs_grad=False) ## does not need grad ## 
        # self.grid_body_index = ti.field(dtype=self.i_dtype, shape=grid_shape[1:], needs_grad=False) ## does not need grad ## # 
        # # grid body index # # grid body index #

        self.act_buffer.fill(0.)
        # self.grid_act_forces.fill(0.)
        self.grid_act_forces.grad.fill(0.0)
        self.grid_act_forces_out.fill(0.)
        self.grid_max_particle_weight.fill(0.)
        
        self.grid_sum_angular_vel.fill(0.0)
        self.grid_sum_angular_m.fill(0.0)
        self.grid_avg_angular_vel.fill(0.0)
        
        # self.grid_act_forces.
        
        self.current_s = 0
        
        
        self.clear_primitives()
        if self.needs_grad:
            self.clear_particle_grad()
            self.clear_act_buffer_grad()
            self.clear_primitives_grad()

            self.grid_m.grad.fill(0.)
            self.grid_v_in.grad.fill(0.)
            self.grid_v_out.grad.fill(0.)
            
            self.grid_sum_angular_vel.grad.fill(0.0)
            self.grid_sum_angular_m.grad.fill(0.0)
            self.grid_avg_angular_vel.grad.fill(0.0)

        if hasattr(self, 'v_buffer'):
            self.v_buffer.fill(0.)
            if self.needs_grad:
                self.v_buffer.grad.fill(0.)

    ########################################

    ########################################
    ###         Checkpointing            ###
    ########################################
    def record_action_buffer(self, action, dt): # reset the action buffer ## 
        dtype = 'float64' if self.f_dtype == ti.f64 else 'float32'
        if action is None:
            action = [0.] * self.n_actuators
        action = self.ckpt_cache_device.tensor(action, dtype)
        self.checkpoint_cache['act_buffer'][self.current_s] = action
        self.checkpoint_cache['dt'][self.current_s] = dt

    def instantiate_checkpoint_cache(self):
        if self.checkpoint_cache_device == 'torch_gpu':
            ckpt_cache_device = TiDeviceInterface.TorchGPU
        elif self.checkpoint_cache_device == 'torch_cpu':
            ckpt_cache_device = TiDeviceInterface.TorchCPU
        elif self.checkpoint_cache_device == 'numpy':
            ckpt_cache_device = TiDeviceInterface.Numpy
        else:
            raise ValueError(f'Unrecognized checkpoint cache device {self.checkpoint_cache_device}')
        ckpt_cache_device.set_dtype(self.i_dtype, self.f_dtype)
        N = int(np.ceil(self.max_substeps / self.max_substeps_local))

        self.checkpoint_cache = dict(
            x=ckpt_cache_device.create_field((N, self.n_particles[None]), vec_dim=self.dim),
            v=ckpt_cache_device.create_field((N, self.n_particles[None]), vec_dim=self.dim),
            F=ckpt_cache_device.create_field((N, self.n_particles[None]), mat_dim=(self.dim, self.dim)),
            C=ckpt_cache_device.create_field((N, self.n_particles[None]), mat_dim=(self.dim, self.dim)),
            Jp=ckpt_cache_device.create_field((N, self.n_particles[None])),
            act_buffer=ckpt_cache_device.create_field((self.max_substeps, self.max_actuation)), # NOTE: cannot be recomputed, record every substeps
            dt=[None] * self.max_substeps,
        )

        self.ckpt_cache_device = ckpt_cache_device

    def sim_to_checkpoint_cache(self):
        ckpt_i = self.current_s // self.max_substeps_local
        self.save_sim_checkpoint_to_ext(
            s=0,
            x=self.checkpoint_cache['x'][ckpt_i],
            v=self.checkpoint_cache['v'][ckpt_i],
            F=self.checkpoint_cache['F'][ckpt_i],
            C=self.checkpoint_cache['C'][ckpt_i],
            Jp=self.checkpoint_cache['Jp'][ckpt_i],
        )

        self.copy_substep(self.max_substeps_local, 0) # cycle the first step
    
    def sim_from_checkpoint_cache(self):
        self.copy_substep_grad(0, self.max_substeps_local)
        self.reset_grad_till_substep(self.max_substeps_local) # reset grad from 0 ~ (max_substeps_local - 1)

        ckpt_i = self.current_s // self.max_substeps_local
        self.load_sim_checkpoint_from_ext(
            s=0,
            x=self.checkpoint_cache['x'][ckpt_i],
            v=self.checkpoint_cache['v'][ckpt_i],
            F=self.checkpoint_cache['F'][ckpt_i],
            C=self.checkpoint_cache['C'][ckpt_i],
            Jp=self.checkpoint_cache['Jp'][ckpt_i],
        )

        self.current_s -= self.max_substeps_local # reset step because we recompute forward sim
        for _ in range(self.max_substeps_local):
            s = self.current_s
            action = self.checkpoint_cache['act_buffer'][s]
            dt = self.checkpoint_cache['dt'][s]
            self.substep(action, dt, recompute=True)

    @ti.kernel
    def save_sim_checkpoint_to_ext(self, s: I_DTYPE, x: ti.types.ndarray(), v: ti.types.ndarray(), F: ti.types.ndarray(),
                                   C: ti.types.ndarray(), Jp: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for d1 in ti.static(range(self.dim)):
                x[p, d1] = self.x[s, p][d1]
                v[p, d1] = self.v[s, p][d1]
                for d2 in ti.static(range(self.dim)):
                    F[p, d1, d2] = self.F[s, p][d1, d2]
                    C[p, d1, d2] = self.C[s, p][d1, d2]
            Jp[p] = self.Jp[s, p]

    @ti.kernel
    def load_sim_checkpoint_from_ext(self, s: I_DTYPE, x: ti.types.ndarray(), v: ti.types.ndarray(), F: ti.types.ndarray(),
                                     C: ti.types.ndarray(), Jp: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            for d1 in ti.static(range(self.dim)):
                self.x[s, p][d1] = x[p, d1]
                self.v[s, p][d1] = v[p, d1]
                for d2 in ti.static(range(self.dim)):
                    self.F[s, p][d1, d2] = F[p, d1, d2]
                    self.C[s, p][d1, d2] = C[p, d1, d2]
            self.Jp[s, p] = Jp[p]

    @ti.kernel
    def copy_substep(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles[None]):
            self.x[target, p] = self.x[source, p]
            self.v[target, p] = self.v[source, p]
            self.F[target, p] = self.F[source, p]
            self.C[target, p] = self.C[source, p]
            self.Jp[target, p] = self.Jp[source, p]
            self.actuation[target, p] = self.actuation[source, p] # may not be neccessary; just for consistency

        for i in range(self.n_actuators):
            self.act_buffer[target, i] = self.act_buffer[source, i]

    @ti.kernel
    def copy_substep_grad(self, source: ti.i32, target: ti.i32):
        for p in range(self.n_particles[None]):
            self.x.grad[target, p] = self.x.grad[source, p]
            self.v.grad[target, p] = self.v.grad[source, p]
            self.F.grad[target, p] = self.F.grad[source, p]
            self.C.grad[target, p] = self.C.grad[source, p]
            self.Jp.grad[target, p] = self.Jp.grad[source, p]
            self.actuation.grad[target, p] = self.actuation.grad[source, p]
        
        for i in range(self.n_actuators):
            self.act_buffer.grad[target, i] = self.act_buffer.grad[source, i]

    @ti.kernel
    def reset_grad_till_substep(self, s_end: I_DTYPE):
        for s, p in ti.ndrange(s_end, self.n_particles[None]):
            self.x.grad[s, p].fill(0)
            self.v.grad[s, p].fill(0)
            self.F.grad[s, p].fill(0)
            self.C.grad[s, p].fill(0)
            self.Jp.grad[s, p] = ti.cast(0., F_DTYPE)
            self.F_tmp.grad[s, p].fill(0)
            self.U.grad[s, p].fill(0)
            self.V.grad[s, p].fill(0)
            self.sig.grad[s, p].fill(0)
            self.actuation.grad[s, p] = ti.cast(0., F_DTYPE)

        for s, i in ti.ndrange(s_end, self.n_actuators):
            self.act_buffer.grad[s, i] = ti.cast(0, F_DTYPE)

    ########################################

    ########################################
    ###         Helper functions         ###
    ########################################
    @ti.func
    def seed_particle(self, s, x, v, material, particle_id, p_rho, mu, lambd):
        p = ti.atomic_add(self.n_particles[None], 1)

        # Set by arguments
        self.x[s, p] = x
        self.v[s, p] = v
        self.material[p] = material
        self.particle_ids[p] = particle_id

        # Set by default value
        self.F[s, p] = ti.Matrix.identity(F_DTYPE, self.dim)
        self.C[s, p] = ti.Matrix.zero(F_DTYPE, self.dim, self.dim)
        
        if material == Material.Sand.value:
            self.Jp[s, p] = 0
        else:
            self.Jp[s, p] = 1
    
        # Set by particle group information
        self.p_rho[p] = p_rho
        self.mu[p] = mu
        self.lambd[p] = lambd

    @ti.func
    def seed_nonoverlap_particle(self, s, x, v, material, particle_id, p_rho, mu, lambd):
        base = ti.floor(x * self.inv_dx - 0.5).cast(I_DTYPE)
        n_occupied = ti.cast(0, I_DTYPE)
        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
            if self.grid_m[s, base + offset] > 0:
                n_occupied += 1

        if n_occupied == 0: # only spawn particle at non-occupied grid cell
            self.seed_particle(s, x, v, material, particle_id, p_rho, mu, lambd)

    @ti.kernel
    def set_particle_occupancy(self, s: I_DTYPE, p_start: I_DTYPE, p_end: I_DTYPE):
        for p in range(p_start, p_end):
            base = ti.floor(self.x[s, p] * self.inv_dx - 0.5).cast(I_DTYPE)
            for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.dim)))):
                self.grid_m[s, base + offset] += 1

    def set_collider_occupancy(self, s: int, static_i: int):
        # TODO: need to add rigid-body collider here
        if static_i not in self.collider_occupancy_kernel.keys():
            @ti.kernel
            def _set_collider_occupancy(_s: I_DTYPE):
                for _I in ti.grouped(ti.ndrange(*self.res)):
                    v_in = ti.Vector.zero(self.f_dtype, self.dim)
                    _, signed_dist = self.grid_process_static[static_i](1., _I, v_in, 1., _s)

                    if signed_dist < 0:
                        self.grid_m[_s, _I] += 1
        
            kernel = _set_collider_occupancy
        else: # avoid create duplicate kernel for the same static_i
            kernel = self.collider_occupancy_kernel[static_i]

        kernel(s)

    @ti.kernel
    def get_material(self, material: ti.types.ndarray()):
        for p in range(self.n_particles[None]):
            material[p] = self.material[p]

    @ti.kernel
    def get_act_buffer(self, s: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.n_actuators):
            ext_arr[p] = self.act_buffer[s, p]

    @ti.kernel
    def set_act_buffer(self, s: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.n_actuators):
            self.act_buffer[s, p] = ext_arr[p]

    @ti.kernel
    def sync_actuation_with_buffer(self, s: I_DTYPE):
        for p in range(self.n_particles[None]):
            id = self.particle_ids[p]
            self.actuation[s, p] = 0.

            act_buffer_id = self.particle_id_to_act_buffer_id[id]
            if act_buffer_id != -1:
                self.actuation[s, p] = self.act_buffer[s, act_buffer_id]


    # def show(self):
    
    def get_rotation_loss(self, ):
        target_point_pos = self.x[self.current_s, 4] # 
        x_target_point_pos = target_point_pos[0]
        diff_x_target_point_pos_w_target = (0.1 - x_target_point_pos) ** 2
        grad_x_target_point_pos = -2 * (0.1 - x_target_point_pos)
        self.x.grad[self.current_s, 4][0] += grad_x_target_point_pos
        return diff_x_target_point_pos_w_target
    
    @ti.kernel
    def update_act_force_field_t(self, s: I_DTYPE):
        for I in ti.grouped(ti.ndrange(*self.res)): #
            upd_force = self.grid_act_forces[s, I] - 10* self.grid_act_forces.grad[s, I]
            # upd_force = upd_force.normalized() * (2.0 * ti.math.sqrt(2))
            self.grid_act_forces[s, I] = upd_force
        
    
    def update_act_force_field(self):
        for i_s in range(self.max_substeps_local):
            self.update_act_force_field_t(i_s)

    ########################################

if __name__=='__main__':
    # test mpm solver #
    
    from softzoo.engine.taichi_sim import TaichiSim
    from softzoo.configs.config import get_cfg_defaults
    # from ..configs.item_configs.design_space import get_cfg_defaults as get_design_space_cfg_defaults
    from softzoo.tools.general_utils import merge_cfg, set_cfg_attr
    # from ..engine import I_DTYPE, F_DTYPE


    # @ti.data_oriented
    # class BaseEnv(Env):
    #     def __init__(self, cfg_file: CN, out_dir: str, device: Optional[str] = 'numpy',
    #                 cfg_kwargs: Optional[Dict] = dict(), ti_device_memory_fraction: Optional[float] = None):
    #         if ti_device_memory_fraction is not None:
    #             ti.init(arch=ti.cuda, device_memory_fraction=ti_device_memory_fraction)

    #         # Get environment configuration
    #         default_cfg = get_cfg_defaults()
    #         cfg = merge_cfg(default_cfg, cfg_file)
    #         for k, v in cfg_kwargs.items():
    #             set_cfg_attr(cfg, k, v)
    
    # cfg = MPMConfig()
    ti_device_memory_fraction = 0.8
    ti.init(arch=ti.cuda, device_memory_fraction=ti_device_memory_fraction)
    cfg_file = "/home/xueyi/diffsim/softzoo/softzoo/configs/env_configs/ground.yaml"
    default_cfg = get_cfg_defaults()
    cfg = merge_cfg(default_cfg, cfg_file)
    # for k, v in cfg_kwargs.items():
    #     set_cfg_attr(cfg, k, v)
    
    print(cfg)
    ### set the gravity ###
    cfg.gravity = [0, 0, 0]
    cfg.dim = 2
    cfg.needs_grad = True
    cfg.use_checkpointing = False
    cfg.checkpoint_cache_device = 'numpy'
    
    
    xx_nn = 2
    # xx_nn = 20
    tot_dense_particles = MPMSolver.get_particles_dense_nn(xx_nn=xx_nn)
    
    cfg.SIMULATOR.max_num_particles = tot_dense_particles
    print(f"tot_dense_particles: {cfg.SIMULATOR.max_num_particles}")
    cfg.SIMULATOR.max_substeps_local = 80
    cfg.SIMULATOR.dim = 2
    cfg.SIMULATOR.padding = [2, 2]
    cfg.SIMULATOR.gravity = [0, 0]
    cfg.SIMULATOR.proj_coef = 1.0
    cfg.SIMULATOR.proj_coef = 0.0
    cfg.SIMULATOR.default_dt = 0.0005
    cfg.SIMULATOR.p_rho_0 = 1.0
    cfg.SIMULATOR.E_0 = 1.0
    cfg.SIMULATOR.nu_0 = 0.02
    mpm_solver = MPMSolver(cfg=cfg.SIMULATOR)
    mpm_solver.initialize_particles_dense(xx_nn=xx_nn)
    mpm_solver.initialize_actuation_forces_all()
    
    gui = ti.GUI("MPM88")
    cur_ts = 0
    
    #  python softzoo/engine/mpm_solver_v2.py
    
    while gui.running and not gui.get_event(gui.ESCAPE):
        # for s in range(50):
        #     substep()
        
        # def step(self, action, frame_dt):
        # assert self.is_initialized, 'Should call initialize() first before doing anything.'
            
        # Step solver
        mpm_solver.step_new()
        
        
        
        gui.clear(0x112F41) ## having set six particles ##
        gui.circles(mpm_solver.x.to_numpy()[cur_ts], radius=7, color=0x068587)
        gui.show()
        
        cur_ts = (cur_ts + 1) % (mpm_solver.grid_shape[0])
        
    
    
    
    
    
    ### TODO: sim the dense particles ##
    
    
    cfg.SIMULATOR.max_num_particles = 6
    cfg.SIMULATOR.dim = 2
    cfg.SIMULATOR.padding = [2, 2]
    cfg.SIMULATOR.gravity = [0, 0]
    mpm_solver = MPMSolver(cfg=cfg.SIMULATOR)
    mpm_solver.initialize_particles()
    mpm_solver.initialize_actuation_forces_all()
    ## six particles ##
    ## define the forward field and see the projection ## forces_all ##
    
    gui = ti.GUI("MPM88")
    cur_ts = 0
    # while gui.running and not gui.get_event(gui.ESCAPE):
    #     # for s in range(50):
    #     #     substep()
        
    #     # def step(self, action, frame_dt):
    #     # assert self.is_initialized, 'Should call initialize() first before doing anything.'
            
    #     # Step solver
    #     mpm_solver.step_new()
        
        
        
    #     gui.clear(0x112F41) ## having set six particles ##
    #     gui.circles(mpm_solver.x.to_numpy()[cur_ts], radius=7, color=0x068587)
    #     gui.show()
        
    #     cur_ts = (cur_ts + 1) % (mpm_solver.grid_shape[0])
        
    
    # ### for optimization ###
    # while gui.running and not gui.get_event(gui.ESCAPE):
    #     for i_s in range(mpm_solver.max_substeps_local):
    #         mpm_solver.substep(None, mpm_solver.default_dt)
            
    #         gui.clear(0x112F41) ## having set six particles ##
    #         gui.circles(mpm_solver.x.to_numpy()[i_s], radius=7, color=0x068587)
    #         gui.show()
        
    #     # get loss and optimize #
    #     diff_loss = mpm_solver.get_rotation_loss()
    #     print(f"diff_loss: {diff_loss}")
    #     for i_s in range(mpm_solver.max_substeps_local):
    #         # mpm_solver.substep(None, mpm_solver.default_dt)
    #         mpm_solver.substep_grad(None, mpm_solver.default_dt)
        
    #     summ_force_grad = mpm_solver.grid_act_forces.grad.to_numpy()
    #     print(f"summ_force_grad: {np.sum(summ_force_grad)}")
        
    #     print(f"After grad")
    #     mpm_solver.update_act_force_field()
    #     print(f"After update act force field")
    #     mpm_solver.reset()
    #     mpm_solver.initialize_particles()
    #     print(f"Reset")
        
    while True:
        for i_s in range(mpm_solver.max_substeps_local):
            mpm_solver.substep(None, mpm_solver.default_dt)
            
            # gui.clear(0x112F41) ## having set six particles ##
            # gui.circles(mpm_solver.x.to_numpy()[i_s], radius=7, color=0x068587)
            # gui.show()
        
        # get loss and optimize #
        diff_loss = mpm_solver.get_rotation_loss()
        print(f"diff_loss: {diff_loss}")
        for i_s in range(mpm_solver.max_substeps_local):
            # mpm_solver.substep(None, mpm_solver.default_dt)
            mpm_solver.substep_grad(None, mpm_solver.default_dt)
        
        summ_force_grad = mpm_solver.grid_act_forces.grad.to_numpy()
        print(f"summ_force_grad: {np.sum(summ_force_grad)}")
        
        print(f"After grad")
        mpm_solver.update_act_force_field()
        print(f"After update act force field")
        mpm_solver.reset() # reset the particles and the forces ## # forces # # forces # # 
        #hwo to parameterize tthe motions of the particles --- actuations # actuations 
        # 
        mpm_solver.initialize_particles()
        print(f"Reset")
        
        # this # 
        # grid force field #
        # constraints formulation 
        # affine transformation --- affine transformation ---
        # 

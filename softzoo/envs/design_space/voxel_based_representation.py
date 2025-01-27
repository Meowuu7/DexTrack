from yacs.config import CfgNode as CN
import numpy as np
from attrdict import AttrDict
import taichi as ti

from .design_representation import DesignRepresentation
from ...engine import I_DTYPE, F_DTYPE
from ...engine.taichi_sim import TaichiSim
from ...tools.general_utils import load_mesh, compute_lame_parameters
from ...tools.voxelizer import Voxelizer
from ...engine.materials import Material


@ti.data_oriented
class VoxelBasedRepresentation(DesignRepresentation):
    def __init__(self, sim: TaichiSim, cfg: CN):
        super().__init__(sim, cfg)

        # Arguments
        self.voxel_resolution = self.cfg.voxel_resolution
        if self.cfg.base_shape.type == 'Primitive.Box':
            self.scale = self.cfg.base_shape.size
        elif self.cfg.base_shape.type == 'Primitive.Mesh':
            self.scale = self.cfg.base_shape.scale
        else:
            raise NotImplementedError
        
        assert self.sim.solver.dim == 3, 'Voxel based representation can only be used in 3-dimensional world'
        
    def build(self):
        i_dtype_np = np.int32 if I_DTYPE == ti.i32 else np.int64

        if self.cfg.base_shape.type == 'Primitive.Mesh' and self.cfg.base_shape.file_path is not None:
            # Load mesh TODO: more general function to load mesh
            triangles = load_mesh(self.cfg.base_shape.file_path)
            
            # Scale mesh to fit a 1x1x1 cube
            triangles_fl = np.concatenate(np.split(triangles, 3, axis=-1), axis=0)
            triangles_fl_min = triangles_fl.min(0)
            max_edge_length = (triangles_fl.max(0) - triangles_fl_min).max()
            triangles_fl = (triangles_fl - triangles_fl_min) / max_edge_length
            triangles = np.concatenate(np.split(triangles_fl, 3, axis=0), axis=1)
        
        # Compute bounding boxes of the mesh and compute voxel resolution for setting up prescribed voxel-based representation
        # NOTE: voxel resolution (describe robot shape) is not the same as voxelizer resolution (for base mesh voxelization)
        if isinstance(self.voxel_resolution, (int, float)):
            self.voxel_resolution = (self.voxel_resolution,) * self.sim.solver.dim
        voxelizer_res = list(self.voxel_resolution)
        for i in range(len(voxelizer_res)): # make 2**N
            r = 1
            while (r * 2) < voxelizer_res[i]:
                r = r * 2
            voxelizer_res[i] = r if r == voxelizer_res[i] else r * 2
        voxelizer_dx = 1. / max(voxelizer_res)
        self.voxelizer_super_sample = 1
        self.voxel_resolution = voxelizer_res

        # Find a list of non-used particle ids for robot body and append to particle_group_info
        item_particle_ids = [v.particle_id for v in self.sim.cfg.ENVIRONMENT.ITEMS if hasattr(v, 'particle_id')]
        existing_nonrobot_particle_ids = list(self.sim.solver.particle_group_info.keys()) + item_particle_ids
        max_voxels = np.prod(self.voxel_resolution)
        robot_particle_ids = []
        id = 0
        while True:
            if id not in existing_nonrobot_particle_ids:
                robot_particle_ids.append(id)
                self.sim.solver.particle_group_info[id] = self.cfg.base_shape.particle_info
            id += 1
            if len(robot_particle_ids) >= max_voxels:
                break
        self.robot_particle_ids = ti.field(I_DTYPE, shape=(max_voxels,))
        self.robot_particle_ids.from_numpy(np.array(robot_particle_ids, dtype=i_dtype_np))

        robot_particle_ids_np = np.array(robot_particle_ids)
        assert (np.unique(robot_particle_ids_np[1:] - robot_particle_ids_np[:-1]) == 1).all(), f'Robot particle IDs are not consecutive'

        # An indicator to differentiate robot and non-robot particle id NOTE: need to be built after matter
        max_particle_group_id = max(existing_nonrobot_particle_ids + robot_particle_ids)
        self.particle_id_to_design_id = ti.Vector.field(self.sim.solver.dim, dtype=I_DTYPE)
        ti.root.dynamic(ti.i, max_particle_group_id + 1).place(self.particle_id_to_design_id)
        vsr_res = self.voxel_resolution
        @ti.kernel
        def _get_voxel_indexing():
            for I in ti.grouped(ti.ndrange(*vsr_res)): # set voxel indexing for robot
                idx = I[0] * vsr_res[1] * vsr_res[2] + I[1] * vsr_res[2] + I[2]
                id = self.robot_particle_ids[idx]
                self.particle_id_to_design_id[id] = I
        _get_voxel_indexing()
        for id in existing_nonrobot_particle_ids: # set invalid indexing for non-robot
            self.particle_id_to_design_id[id] = [-1] * self.sim.solver.dim

        # Append to external items in sim to allow sim to set up semantic map
        for id in robot_particle_ids:
            self.sim.external_items.append(CN(dict(
                particle_id=id,
                semantic_id=self.semantic_id,
            )))

        # Voxelize mesh
        self.voxelizer = Voxelizer(res=voxelizer_res, # NOTE: will be modified as 2**N
                                   dx=voxelizer_dx,
                                   precision=self.sim.solver.f_dtype,
                                   padding=(0,) * self.sim.solver.dim, # no padding since this is local voxel grid
                                   super_sample=self.voxelizer_super_sample)
        if self.cfg.base_shape.type == 'Primitive.Mesh' and self.cfg.base_shape.file_path is not None:
            self.voxelizer.voxelize(triangles)
        else:
            self.voxelizer.fill_all(1)

        sample_density_scale = (self.sim.solver.n_grid / max(self.voxel_resolution))**self.sim.solver.dim * np.prod(self.scale)
        self.sample_density = max(1, int(self.sample_density * sample_density_scale)) # compensate for voxelizer resolution wrt. mpm solver resolution

        self.p_start = self.sim.solver.n_particles[None]
        self.seed_particles_from_voxels(self.sim.solver.current_s)
        self.p_end = self.sim.solver.n_particles[None]
        assert self.sim.solver.n_particles[None] <= self.sim.solver.max_num_particles, \
            f'Exceeding maximal number of particles {self.sim.solver.max_num_particles} when adding {self.p_end - self.p_start} to {self.p_start}'

        # Handle actuators TODO: passive actuators
        self.sim.solver.sync_actuation_with_buffer = self.sync_actuation_with_buffer
        self.sim.solver.use_actuator_to_specify_muscle_direction = True
        if self.particle_info.muscle_direction is not None:
            self.sim.solver.base_muscle_direction_matrix = np.diag(self.particle_info.muscle_direction)
        else:
            self.sim.solver.base_muscle_direction_matrix = np.zeros((self.sim.solver.dim, self.sim.solver.dim))
        self.sim.solver.get_muscle_direction_matrix = self.get_muscle_direction_matrix
        
        # Buffer
        self.n_particles = self.compute_n_particles()
        self.buffer = AttrDict(dict(
            geometry=ti.field(F_DTYPE, shape=self.voxel_resolution, needs_grad=self.sim.solver.needs_grad),
            softness=ti.field(F_DTYPE, shape=self.voxel_resolution, needs_grad=self.sim.solver.needs_grad),
            actuator=ti.field(F_DTYPE, shape=(self.sim.solver.n_actuators,) + tuple(self.voxel_resolution), needs_grad=self.sim.solver.needs_grad),
        ))

    def update_renderer(self, renderer):
        robot_particle_ids = self.robot_particle_ids.to_numpy()
        ref_robot_id = robot_particle_ids[0]
        ref_robot_count = self.p_end - self.p_start
        new_particle_id_range = dict()
        if hasattr(renderer, 'particle_id_range'): # GGUI
            for k, v in renderer.particle_id_range.items():
                if k not in robot_particle_ids:
                    new_particle_id_range[k] = v
                elif k == ref_robot_id:
                    new_particle_id_range[k] = (self.p_start, ref_robot_count)
            renderer.particle_id_range = new_particle_id_range
        else: # GL
            pass

    def initialize(self):
        self.buffer['geometry'].fill(1.)
        self.buffer['softness'].fill(1.)
        self.buffer['actuator'].fill(1. / self.n_actuators)

    def reset(self):
        for v in self.buffer.values():
            if self.sim.solver.needs_grad:
                v.grad.fill(0.)
        self.n_active_particles = self.compute_n_active_particles()
            
    @ti.ad.grad_replaced
    def set_design(self, design):
        if 'geometry' in design.keys():
            self.sim.device.from_ext(self.buffer['geometry'], design['geometry'])
        if 'softness' in design.keys():
            self.sim.device.from_ext(self.buffer['softness'], design['softness'])
        if 'actuator' in design.keys():
            self.sim.device.from_ext(self.buffer['actuator'], design['actuator'])
        self.set_design_kernel()
        if 'actuator_direction' in design.keys():
            self.sim.device.from_ext(self.sim.solver.muscle_direction, design['actuator_direction']) # TODO no grad
        
    @ti.ad.grad_for(set_design)
    def set_design_grad(self, design):
        self.set_design_kernel.grad()

    def get_grad(self):
        grad = dict()
        for k, v in self.buffer.items():
            grad[k] = self.sim.device.to_ext(v.grad)
        return grad

    @ti.kernel
    def seed_particles_from_voxels(self, s: I_DTYPE):
        for i, j, k in self.voxelizer.voxels:
            if self.voxelizer.voxels[i, j, k] > 0:
                for l in range(self.sample_density + 1):
                    ss = self.sample_density / self.voxelizer_super_sample**self.sim.solver.dim
                    if ti.random() + l < ss:
                        x = ti.Vector([
                            ti.random() + i,
                            ti.random() + j,
                            ti.random() + k
                        ], dt=F_DTYPE) * (self.voxelizer.dx / self.voxelizer_super_sample) * \
                            ti.Vector(self.scale, F_DTYPE) + ti.Vector(self.initial_position, F_DTYPE)
                        particle_id = self.robot_particle_ids[i*self.voxelizer.res[1]*self.voxelizer.res[2] + j*self.voxelizer.res[2] + k] # NOTE: this will cause non-contiguous particle id
                        v = ti.Vector(self.initial_velocity, F_DTYPE)
                        self.sim.solver.seed_nonoverlap_particle(s, x, v, self.material.value, particle_id,
                                                                self.particle_info.p_rho_0, self.particle_info.mu_0,
                                                                self.particle_info.lambd_0)

    @ti.kernel
    def set_design_kernel(self):
        for p in range(self.sim.solver.n_particles[None]):
            # Convert particle id to design group id
            id = self.sim.solver.particle_ids[p]
            design_id = self.particle_id_to_design_id[id]
            
            # Set particle-wise design parameter
            if self.is_robot(id):
                # update density
                self.sim.solver.p_rho[p] = self.particle_info.p_rho_0 * self.buffer['geometry'][design_id]
                if self.sim.solver.p_rho[p] < self.p_rho_lower_bound: # NOTE: may cause discontinuity
                    self.sim.solver.x[0, p] = [0., 0., 0.]
                    self.sim.solver.p_rho[p] = 0
                    
                # update softness (both mu and lambd is linearly proportional to E)
                self.sim.solver.mu[p] = self.particle_info.mu_0 * self.buffer['softness'][design_id]
                self.sim.solver.lambd[p] = self.particle_info.lambd_0 * self.buffer['softness'][design_id]

    @ti.kernel
    def sync_actuation_with_buffer(self, s: I_DTYPE):
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            design_id = self.particle_id_to_design_id[id]
            
            self.sim.solver.actuation[s, p] = 0.
            if self.is_robot(id) and self.sim.solver.p_rho[p] > 0:
                for i in ti.static(range(self.sim.solver.n_actuators)):
                    self.sim.solver.actuation[s, p] += self.buffer['actuator'][i, design_id] * self.sim.solver.act_buffer[s, i]

    def get_actuator(self, s: int, keep_mask=False):
        s_local = self.sim.solver.get_cyclic_s(s)
        mask = self.get_particle_mask()
        actuator = self.sim.device.create_f_tensor((self.n_particles, self.sim.solver.n_actuators))
        self.get_actuator_kernel(s_local, actuator)
        if keep_mask:
            return actuator, mask
        else:
            return actuator[mask.to(bool)]

    @ti.kernel
    def get_actuator_kernel(self, s_local: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.p_start, self.p_end):
            p_ext = p - self.p_start
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id) and (self.sim.solver.p_rho[p] > 0):
                design_id = self.particle_id_to_design_id[id]
                for i in ti.static(range(self.sim.solver.n_actuators)):
                    ext_arr[p_ext, i] = self.buffer['actuator'][i, design_id]

    @ti.func
    def get_muscle_direction_matrix(self, p: I_DTYPE, id: I_DTYPE):
        id = self.sim.solver.particle_ids[p]
        design_id = self.particle_id_to_design_id[id]
        AAt = ti.Matrix.zero(F_DTYPE, self.sim.solver.dim, self.sim.solver.dim)
        # don't need to check if particle is robot since this function is used in muscle material model and the particle must be robot
        for i in ti.static(range(self.sim.solver.n_actuators)):
            # NOTE: interpolating muscle direction matrix may not be totally reasonable
            AAt += self.buffer['actuator'][i, design_id] * self.sim.solver.muscle_direction[i]

        return AAt

    @ti.func
    def is_robot(self, id: I_DTYPE):
        design_id = self.particle_id_to_design_id[id]
        return design_id[0] != -1

    @property
    def size(self):
        return self.voxel_resolution

    @property
    def needs_to_be_built_at_env_init(self):
        return True

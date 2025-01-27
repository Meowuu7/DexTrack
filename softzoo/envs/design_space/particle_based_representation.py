from yacs.config import CfgNode as CN
from attrdict import AttrDict
import numpy as np
import taichi as ti

from .design_representation import DesignRepresentation
from ...engine import I_DTYPE, F_DTYPE
from ...engine.taichi_sim import TaichiSim



@ti.data_oriented
class ParticleBasedRepresentation(DesignRepresentation):
    def __init__(self, sim: TaichiSim, cfg: CN):
        super().__init__(sim, cfg)

        # Check if the particle id for robot has no conflict
        item_particle_ids = [v.particle_id for v in self.sim.cfg.ENVIRONMENT.ITEMS if hasattr(v, 'particle_id')]
        existing_particle_ids = list(self.sim.solver.particle_group_info.keys()) + item_particle_ids
        assert self.cfg.base_shape.particle_id not in existing_particle_ids, \
            'Assigned particle ID for robot is already taken. Please specify a new ID without conflict'

        # Add to environment item list and let TaichiSim takes care of the rest
        self.p_start = self.sim.solver.n_particles[None]
        self.sim.solver.add(self.cfg.base_shape.type, self.cfg.base_shape)
        self.p_end = self.sim.solver.n_particles[None]

        # Handle actuators
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
            geometry=ti.field(F_DTYPE, shape=(self.n_particles,), needs_grad=self.sim.solver.needs_grad),
            softness=ti.field(F_DTYPE, shape=(self.n_particles,), needs_grad=self.sim.solver.needs_grad),
            actuator=ti.field(F_DTYPE, shape=(self.n_actuators, self.n_particles), needs_grad=self.sim.solver.needs_grad),
        ))

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
    def set_design_kernel(self):
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            
            # Set particle-wise design parameter
            if self.is_robot(id):
                p_design_space = p - self.p_start

                # update density
                self.sim.solver.p_rho[p] = self.particle_info.p_rho_0 * self.buffer['geometry'][p_design_space]
                if self.sim.solver.p_rho[p] < self.p_rho_lower_bound: # NOTE: may cause discontinuity
                    self.sim.solver.x[0, p] = [0., 0., 0.]
                    self.sim.solver.p_rho[p] = 0
                    
                # update softness (both mu and lambd is linearly proportional to E)
                self.sim.solver.mu[p] = self.particle_info.mu_0 * self.buffer['softness'][p_design_space]
                self.sim.solver.lambd[p] = self.particle_info.lambd_0 * self.buffer['softness'][p_design_space]

    @ti.kernel
    def sync_actuation_with_buffer(self, s: I_DTYPE):
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]

            self.sim.solver.actuation[s, p] = 0.
            if self.is_robot(id) and self.sim.solver.p_rho[p] > 0:
                p_design_space = p - self.p_start
                for i in ti.static(range(self.sim.solver.n_actuators)):
                    self.sim.solver.actuation[s, p] += self.buffer['actuator'][i, p_design_space] * self.sim.solver.act_buffer[s, i]

    @ti.func
    def is_robot(self, id: I_DTYPE):
        return id == self.cfg.base_shape.particle_id

    @ti.func
    def get_muscle_direction_matrix(self, p: I_DTYPE, id: I_DTYPE):
        p_design_space = p - self.p_start
        AAt = ti.Matrix.zero(F_DTYPE, self.sim.solver.dim, self.sim.solver.dim)
        # don't need to check if particle is robot since this function is used in muscle material model and the particle must be robot
        for i in ti.static(range(self.sim.solver.n_actuators)):
            # NOTE: interpolating muscle direction matrix may not be totally reasonable
            AAt += self.buffer['actuator'][i, p_design_space] * self.sim.solver.muscle_direction[i]

        return AAt

    @property
    def size(self):
        return self.n_particles

    ### HACK solution of setting v
    def instantiate_v_buffer(self):
        self.sim.solver.v_buffer = ti.Vector.field(self.sim.solver.dim, F_DTYPE,
                                                   shape=(self.sim.solver.max_substeps, self.n_actuators),
                                                   needs_grad=True)

    @ti.kernel
    def set_v_buffer(self, s: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.n_actuators):
            for d in ti.static(range(self.sim.solver.dim)):
                self.sim.solver.v_buffer[s, p][d] = ext_arr[p, d]

    @ti.kernel
    def add_v_with_buffer(self, s: I_DTYPE, s_local: I_DTYPE):
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]

            if self.is_robot(id) and self.sim.solver.p_rho[p] > 0:
                p_design_space = p - self.p_start
                for i in ti.static(range(self.sim.solver.n_actuators)):
                    for d in ti.static(range(self.sim.solver.dim)):
                        self.sim.solver.v[s_local, p][d] += self.buffer['actuator'][i, p_design_space] * self.sim.solver.v_buffer[s, i][d]
    
    def get_v_buffer_grad(self, s: int):
        ext_arr = self.sim.device.create_f_tensor((self.n_actuators, self.sim.solver.dim))
        self.get_v_buffer_grad_kernel(s, ext_arr)
        return ext_arr

    @ti.kernel
    def get_v_buffer_grad_kernel(self, s: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.n_actuators):
            for d in ti.static(range(self.sim.solver.dim)):
                ext_arr[p, d] = self.sim.solver.v_buffer.grad[s, p][d]
    ##########

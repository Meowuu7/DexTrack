import numpy as np
import taichi as ti

from .primitive_base import PrimitiveBase
from .. import I_DTYPE, F_DTYPE, NORM_EPS
from ..materials import Material


@ti.data_oriented
class Box(PrimitiveBase):
    def __init__(self, solver, cfg):
        super().__init__(solver, cfg)
        assert len(self.size) == self.solver.dim and len(self.size) == self.solver.dim

        # Instantiate shape in MPM
        if not self.is_rigid:
            vol = 1
            for i in range(self.solver.dim):
                vol = vol * self.size[i]

            num_new_particles = int(self.sample_density * vol / self.solver.p_vol + 1)
            assert self.solver.n_particles[
                None] + num_new_particles <= self.solver.max_num_particles, \
                    f'Cannot add additional {num_new_particles} particles on top of {self.solver.n_particles[None]} particles'
            
            self.source_bound = ti.Vector.field(self.solver.dim, dtype=F_DTYPE, shape=2)
            for i in range(self.solver.dim):
                self.source_bound[0][i] = self.initial_position[i] - self.size[i] / 2.
                self.source_bound[1][i] = self.size[i]

            self.seed(self.solver.current_s, num_new_particles, self.material.value, self.particle_id)

    @ti.kernel
    def seed(self, s: I_DTYPE, num_new_particles: I_DTYPE, material: I_DTYPE, particle_id: I_DTYPE):
        for p in range(self.solver.n_particles[None],
                       self.solver.n_particles[None] + num_new_particles):
            x = ti.Vector.zero(F_DTYPE, self.solver.dim)
            for k in ti.static(range(self.solver.dim)):
                x[k] = self.source_bound[0][k] + ti.random() * self.source_bound[1][k]
            v = ti.Vector(self.initial_velocity, F_DTYPE)
            self.solver.seed_nonoverlap_particle(s, x, v, material, particle_id, self.particle_info.p_rho_0,
                                                 self.particle_info.mu_0, self.particle_info.lambd_0)

    def _inv_inertia(self):
        dim = self.solver.dim
        size = self.cfg.size
        density = self.cfg.density

        vol = np.prod(size)
        mass = density * vol

        if dim == 2:
            v1 = 1 / (1 / 12 * mass * (size[0]**2 + size[1]**2))
            v2 = 1 / mass
            inv_inertia = ti.Matrix([
                [v1,  0.,  0.],
                [0.,  v2,  0.],
                [0.,  0.,  v2],
            ])
        else:
            v1 = 1 / (1 / 12 * mass * (size[1]**2 + size[2]**2))
            v2 = 1 / (1 / 12 * mass * (size[0]**2 + size[2]**2))
            v3 = 1 / (1 / 12 * mass * (size[0]**2 + size[1]**2))
            v4 = 1. / mass
            inv_inertia = ti.Matrix([
                [v1,  0.,  0.,  0.,  0.,  0.],
                [0.,  v2,  0.,  0.,  0.,  0.],
                [0.,  0.,  v3,  0.,  0.,  0.],
                [0.,  0.,  0.,  v4,  0.,  0.],
                [0.,  0.,  0.,  0.,  v4,  0.],
                [0.,  0.,  0.,  0.,  0.,  v4],
            ])
    
        return inv_inertia

    @ti.func
    def _sdf(self, s, grid_pos):
        q = ti.abs(grid_pos) - ti.Vector(self.size, F_DTYPE) / 2.
        out = max(q, 0.0).norm(NORM_EPS)
        if ti.static(self.solver.dim == 2):
            out += min(max(q[0], q[1]), 0.0)
        else:
            out += min(max(q[0], max(q[1], q[2])), 0.0)
        return out

    # TODO: use analytical normal

import math
import taichi as ti

from .primitive_base import PrimitiveBase
from .. import I_DTYPE, F_DTYPE, NORM_EPS
from ..materials import Material


@ti.data_oriented
class Sphere(PrimitiveBase):
    def __init__(self, solver, cfg):
        super().__init__(solver, cfg)
        
        # Instantiate shape in MPM
        if not self.is_rigid:
            if self.solver.dim == 2:
                vol = math.pi * self.radius**2
            else:
                vol = 3. / 4 * math.pi * self.radius**3

            num_new_particles = int(self.sample_density * vol / self.solver.p_vol + 1)
            assert self.solver.n_particles[
                None] + num_new_particles <= self.solver.max_num_particles

            self.source_bound = ti.Vector.field(self.solver.dim, dtype=F_DTYPE, shape=2)
            self.source_bound[0] = self.initial_position
            self.source_bound[1] = [self.radius] * self.solver.dim

            self.seed(self.solver.current_s, num_new_particles, self.material.value, self.particle_id)

    @ti.kernel
    def seed(self, s: I_DTYPE, num_new_particles: I_DTYPE, material: I_DTYPE, particle_id: I_DTYPE):
        for p in range(self.solver.n_particles[None],
                       self.solver.n_particles[None] + num_new_particles):
            x = self.source_bound[0] + self.random_point_in_unit_sphere() * self.source_bound[1]
            v = ti.Vector(self.initial_velocity, F_DTYPE)
            self.solver.seed_nonoverlap_particle(s, x, v, material, particle_id, self.particle_info.p_rho_0,
                                                 self.particle_info.mu_0, self.particle_info.lambd_0)

    @ti.func
    def random_point_in_unit_sphere(self):
        ret = ti.Vector.zero(F_DTYPE, n=self.solver.dim)
        while True:
            for i in ti.static(range(self.solver.dim)):
                ret[i] = ti.random(F_DTYPE) * 2 - 1
            if ret.norm_sqr() <= 1:
                break
        return ret

    def _inv_inertia(self):
        dim = self.solver.dim
        radius = self.cfg.radius
        density = self.cfg.density

        if dim == 2:
            vol = 1. / 2. * math.pi * radius**dim
        else:
            vol = 4. / 3. * math.pi * radius**dim
        mass = density * vol
        
        # solid sphere
        if dim == 2:
            v1 = 1 / (1 / 2 * mass * radius**2)
            v2 = 1. / mass
            inv_inertia = ti.Matrix([
                [v1,  0.,  0.],
                [0.,  v2,  0.],
                [0.,  0.,  v2],
            ])
        else:
            v1 = 1 / (2 / 5 * mass * radius**2)
            v2 = 1. / mass
            inv_inertia = ti.Matrix([
                [v1,  0.,  0.,  0.,  0.,  0.],
                [0.,  v1,  0.,  0.,  0.,  0.],
                [0.,  0.,  v1,  0.,  0.,  0.],
                [0.,  0.,  0.,  v2,  0.,  0.],
                [0.,  0.,  0.,  0.,  v2,  0.],
                [0.,  0.,  0.,  0.,  0.,  v2],
            ])

        return inv_inertia

    @ti.func
    def _sdf(self, s, grid_pos):
        return (grid_pos).norm(NORM_EPS) - self.radius

    @ti.func
    def _normal(self, s, grid_pos):
        return (grid_pos).normalized(NORM_EPS)

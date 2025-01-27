import taichi as ti

from .primitive_base import PrimitiveBase
from ...tools.voxelizer import Voxelizer
from .. import I_DTYPE, F_DTYPE
from ...tools.general_utils import load_mesh


@ti.data_oriented
class Mesh(PrimitiveBase):
    def __init__(self, solver, cfg):
        super().__init__(solver, cfg)
        assert self.solver.dim == 3

        self.grid_size = self.solver.n_grid # NOTE: grid size is a different thing in taichi_element implementation

        self.voxelizer = Voxelizer(res=self.solver.res,
                                   dx=self.solver.dx,
                                   precision=self.solver.f_dtype,
                                   padding=self.solver.padding,
                                   super_sample=self.voxelizer_super_sample)
        
        # Instantiate shape in MPM
        if not self.is_rigid:
            triangles = load_mesh(self.file_path, self.scale, offset=self.offset)
            self.voxelizer.voxelize(triangles)

            self.source_bound = ti.Vector.field(self.solver.dim, dtype=F_DTYPE, shape=1)
            for i in range(self.solver.dim):
                self.source_bound[0][i] = self.initial_position[i] - self.offset[i]

            self.seed(self.solver.current_s, -1, self.material.value, self.particle_id)
            ti.sync() # NOTE: not sure why we need this; it seems like it's ok without this

    @ti.kernel
    def seed(self, s: I_DTYPE, num_new_particles: I_DTYPE, material: I_DTYPE, particle_id: I_DTYPE):
        for i, j, k in self.voxelizer.voxels:
            inside = 1
            if ti.static(False): # NOTE: not working
                for d in ti.static(range(3)):
                    inside = inside and -self.grid_size // 2 + self.solver.padding <= i \
                                and i < self.grid_size // 2 - self.solver.padding

            if inside and self.voxelizer.voxels[i, j, k] > 0:
                for l in range(self.sample_density + 1):
                    ss = self.sample_density / self.voxelizer_super_sample**self.solver.dim
                    if ti.random() + l < ss:
                        x = ti.Vector([
                            ti.random() + i,
                            ti.random() + j,
                            ti.random() + k
                        ], dt=F_DTYPE) * (self.solver.dx / self.voxelizer_super_sample
                              ) + self.source_bound[0]

                        v = ti.Vector(self.initial_velocity, F_DTYPE)
                        self.solver.seed_nonoverlap_particle(s, x, v, material, particle_id, self.particle_info.p_rho_0,
                                                             self.particle_info.mu_0, self.particle_info.lambd_0)

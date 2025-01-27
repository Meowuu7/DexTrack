from yacs.config import CfgNode as CN
from attrdict import AttrDict
import taichi as ti
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ...engine import F_DTYPE, I_DTYPE, NORM_EPS
from ...engine.materials import Material
from ...engine.taichi_sim import TaichiSim
from ...tools.general_utils import compute_lame_parameters


@ti.data_oriented
class DesignRepresentation:
    def __init__(self, sim: TaichiSim, cfg: CN):
        self.sim = sim
        self.cfg = cfg

        # Shared arguments
        self.material = getattr(Material, self.cfg.base_shape.material)
        self.sample_density = self.cfg.base_shape.sample_density
        self.initial_position = self.cfg.base_shape.initial_position
        self.initial_velocity = self.cfg.base_shape.initial_velocity
        self.particle_info = self.cfg.base_shape.particle_info
        mu_0, lambd_0 = compute_lame_parameters(self.particle_info.E_0, self.particle_info.nu_0)
        self.particle_info.mu_0 = self.cfg.base_shape.particle_info.mu_0 = mu_0
        self.particle_info.lambd_0 = self.cfg.base_shape.particle_info.lambd_0 = lambd_0
        self.semantic_id = self.cfg.base_shape.semantic_id

        self.n_actuators = self.cfg.n_actuators
        self.p_rho_lower_bound = self.cfg.p_rho_lower_bound_mul * self.particle_info.p_rho_0
        self.initial_principle_direction = self.cfg.initial_principle_direction

        # Check and overwrite actuators setting of solver
        assert self.cfg.n_actuators <= self.sim.solver.max_actuation
        self.sim.solver.no_actuators_setup = True
        self.sim.solver.n_actuators = cfg.n_actuators

        # Buffer
        self.buffer = AttrDict(dict( # to be defined
            geometry=None,
            softness=None,
            actuator=None,
        ))

        # Use to compute muscle stiffness
        self.sim.solver.base_muscle_stiffness = self.particle_info.E_0
        self.sim.solver.base_muscle_mu = mu_0

        # For extracting particles data
        self.p_start = None # to be specified in child class
        self.p_end = None

    def initialize(self):
        pass

    def reset(self):
        raise NotImplementedError

    @ti.ad.grad_replaced
    def set_design(self, design):
        raise NotImplementedError

    @ti.ad.grad_for(set_design)
    def set_design_grad(self, design):
        raise NotImplementedError

    def get_grad(self):
        raise NotImplementedError

    @ti.func
    def is_robot(self, id: I_DTYPE):
        raise NotImplementedError

    @property
    def size(self):
        raise NotImplementedError

    ### Basic helper functions
    def get_x(self, s: int, keep_mask=False):
        s_local = self.sim.solver.get_cyclic_s(s)
        mask = self.get_particle_mask()
        x = self.sim.device.create_f_tensor((self.n_particles, self.sim.solver.dim))
        self.get_x_kernel(s_local, x)
        if keep_mask:
            return x, mask
        else:
            return x[mask.to(bool)]

    def get_v(self, s: int, keep_mask=False):
        s_local = self.sim.solver.get_cyclic_s(s)
        mask = self.get_particle_mask()
        v = self.sim.device.create_f_tensor((self.n_particles, self.sim.solver.dim))
        self.get_v_kernel(s_local, v)
        if keep_mask:
            return v, mask
        else:
            return v[mask.to(bool)]

    @ti.kernel
    def compute_n_particles(self) -> int:
        n_particles = 0
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id):
                n_particles += 1
        return n_particles

    @ti.kernel
    def compute_n_active_particles(self) -> int:
        n_particles = 0
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id) and (self.sim.solver.p_rho[p] > 0):
                n_particles += 1
        return n_particles

    @ti.kernel
    def get_x_kernel(self, s_local: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.p_start, self.p_end):
            p_ext = p - self.p_start
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id) and (self.sim.solver.p_rho[p] > 0):
                for d in ti.static(range(self.sim.solver.dim)):
                    ext_arr[p_ext, d] = self.sim.solver.x[s_local, p][d]

    @ti.kernel
    def get_v_kernel(self, s_local: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.p_start, self.p_end):
            p_ext = p - self.p_start
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id) and (self.sim.solver.p_rho[p] > 0):
                for d in ti.static(range(self.sim.solver.dim)):
                    ext_arr[p_ext, d] = self.sim.solver.v[s_local, p][d]

    @ti.kernel
    def transform_x(self, s_local: I_DTYPE, transform: ti.types.ndarray()):
        for p in range(self.p_start, self.p_end):
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id) and (self.sim.solver.p_rho[p] > 0):
                for d in ti.static(range(self.sim.solver.dim)): # NOTE: only support translation now
                    self.sim.solver.x[s_local, p][d] += transform[d]

    @ti.kernel
    def add_to_v(self, s_local: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.p_start, self.p_end):
            p_ext = p - self.p_start
            id = self.sim.solver.particle_ids[p]
            if self.is_robot(id) and self.sim.solver.p_rho[p] > 0:
                for d in ti.static(range(self.sim.solver.dim)):
                    self.sim.solver.v[s_local, p][d] += ext_arr[p_ext, d]

    def get_particle_mask(self):
        mask = self.sim.device.create_i_tensor((self.n_particles,))
        self.get_particle_mask_kernel(mask)
        return mask

    @ti.kernel
    def get_particle_mask_kernel(self, ext_arr: ti.types.ndarray()):
        for p in range(self.p_start, self.p_end):
            p_ext = p - self.p_start
            id = self.sim.solver.particle_ids[p]
            ext_arr[p_ext] = ti.cast(self.is_robot(id) and (self.sim.solver.p_rho[p] > 0), I_DTYPE)
    ########

    ### Functions for computing robot orientation
    def reset_orientation_data(self):
        s = self.sim.solver.current_s
        assert s == 0, 'This function should be called at s = 0, before reset and after set_design'
        x, mask = self.get_x(s, keep_mask=True)
        mask = mask.to(bool)
        masked_x = x[mask]
        if self.initial_principle_direction is None:
            masked_x_np = masked_x.data.cpu().numpy()
            pca = PCA(n_components=self.sim.solver.dim)
            pca.fit(masked_x_np)
            initial_principle_direction = pca.components_[0]
        else:
            initial_principle_direction = self.initial_principle_direction
        initial_principle_direction = self.sim.device.tensor(initial_principle_direction)

        masked_x_centered = x - masked_x.mean(0)
        masked_x_centered[~mask, :] = 0. # NOTE: setting as centers; not affecting proj.argmin or proj.argmax
        proj = masked_x_centered @ initial_principle_direction

        min_p, max_p = proj.argmin().item(), proj.argmax().item()
        min_p += self.p_start # NOTE: handle case where robot is instantiated later (p_start != 0 like vbr)
        max_p += self.p_start
        if not hasattr(self, 'orientation_data'):
            self.orientation_data = dict()
            self.orientation_data['orientation'] = ti.Vector.field(self.sim.solver.dim, dtype=F_DTYPE, shape=(), needs_grad=self.sim.solver.needs_grad)
        self.orientation_data['min_p'] = min_p
        self.orientation_data['max_p'] = max_p

    def get_orientation(self, s): # not using grad_replaced here otherwise no return value
        s_local = self.sim.solver.get_cyclic_s(s)
        self.compute_orientation_kernel(s_local, self.orientation_data['min_p'], self.orientation_data['max_p'])
        orientation = self.orientation_data['orientation'][None]
        return orientation

    def get_orientation_grad(self, s):
        s_local = self.sim.solver.get_cyclic_s(s)
        self.compute_orientation_kernel.grad(s_local, self.orientation_data['min_p'], self.orientation_data['max_p'])

    @ti.kernel
    def compute_orientation_kernel(self, s_local: I_DTYPE, min_p: I_DTYPE, max_p: I_DTYPE):
        self.orientation_data['orientation'][None] = (self.sim.solver.x[s_local, max_p] - self.sim.solver.x[s_local, min_p]).normalized(NORM_EPS)

    def reset_orientation_grad(self):
        self.orientation_data['orientation'].grad.fill(0.)

    def get_v_avg(self, s, mode=0):
        v, mask = self.get_v(s, keep_mask=True)
        mask = mask.to(bool)
        v = v[mask]
        if mode == 0:
            v_avg = v.mean(0)
        elif mode == 1:
            orientation = self.get_orientation(s)
            orientation = self.sim.device.tensor(orientation)
            proj_v = v @ orientation
            v_avg = proj_v.mean() * orientation
        else:
            sm1 = max(s - 1, 0)
            U = self.sim.apply('get', 'U', s=sm1) # NOTE: since svd is computed for new F
            V = self.sim.apply('get', 'V', s=sm1)
            R_T = V @ U.transpose(2, 1) # R = UV^T
            v_invR = (R_T[mask,...] @ v[...,None])[...,0]

            orientation = self.get_orientation(s)
            orientation = self.sim.device.tensor(orientation)
            proj_v_invR = v_invR @ orientation
            v_avg = proj_v_invR.mean() * orientation
        return v_avg
    ##############

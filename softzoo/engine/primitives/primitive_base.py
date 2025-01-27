import taichi as ti

from .. import I_DTYPE, F_DTYPE, NORM_EPS
from ..materials import Material
from ...tools.general_utils import merge_cfg, compute_lame_parameters
from ...tools.taichi_utils import qmul3d, qrot2d, qrot3d, w2quat2d, w2quat3d, inv_trans2d, inv_trans3d, cross2d
from ...configs.item_configs.primitives import get_cfg_defaults


@ti.data_oriented
class PrimitiveBase:
    def __init__(self, solver, cfg):
        # Get configuration (in-place update)
        default_cfg = get_cfg_defaults(cfg.type)
        merge_cfg(default_cfg, cfg, replace=True)
        
        cfg.particle_info.mu_0, cfg.particle_info.lambd_0 = compute_lame_parameters(
            cfg.particle_info.E_0, cfg.particle_info.nu_0) # recompute lame parameter

        solver.particle_group_info[cfg.particle_id] = cfg.particle_info # append to particle group information in the solver

        self.cfg = cfg

        # Get basic attribute
        self.solver = solver
        for k, v in self.cfg.items():
            setattr(self, k, v)
        self.is_rigid = self.material == 'Rigid' # rigid body is not modelled by MPM and thus treated differently

        if not self.is_rigid:
            self.material = getattr(Material, self.cfg.material)

        # Check configuration
        assert len(self.initial_position) == self.solver.dim
        assert len(self.initial_rotation) == 4
        assert len(self.initial_velocity) == self.solver.dim

        # Initialization for rigid body
        if self.is_rigid:
            assert not self.solver.use_checkpointing, 'Checkpointing not yet support for rigid body'
            
            # Allocate tensors for rigid-body object state
            self.inv_inertia = self._inv_inertia()

            self.position = ti.Vector.field(solver.dim, F_DTYPE, needs_grad=solver.needs_grad) # positon of the primitive
            self.rotation = ti.Vector.field(4, F_DTYPE, needs_grad=solver.needs_grad) # quaternion for storing rotation; NOTE: use 4-dim quaternion for both 2D and 3D
            if self.solver.dim == 2:
                self.w_dim = 1 # angular velocity or torque
                self.v_dim = 2 # linear velocity or force
                
                self.qrot = qrot2d
                self.w2quat = w2quat2d
                self.qmul = qmul3d
                self.inv_trans = inv_trans2d
            else:
                self.w_dim = 3 # angular velocity or torque
                self.v_dim = 3 # linear velocity or force
                
                self.qrot = qrot3d
                self.w2quat = w2quat3d
                self.qmul = qmul3d
                self.inv_trans = inv_trans3d
                
            assert len(self.initial_twist) == (self.w_dim + self.v_dim)
            assert len(self.initial_wrench) == (self.w_dim + self.v_dim)
                
            self.wrench = ti.Vector.field(self.w_dim + self.v_dim, F_DTYPE, needs_grad=solver.needs_grad)
            self.twist = ti.Vector.field(self.w_dim + self.v_dim, F_DTYPE, needs_grad=solver.needs_grad)
            
            ti.root.dense(ti.i, (solver.max_substeps,)).place(self.position, self.rotation,
                                                              self.wrench, self.twist)
            if solver.needs_grad:
                ti.root.dense(ti.i, (solver.max_substeps,)).place(self.position.grad, self.rotation.grad,
                                                                  self.wrench.grad, self.twist.grad)
                
            # Initialize rigid-body state
            self.initialize_rigid_body_state(self.solver.current_s)

    @ti.kernel
    def initialize_rigid_body_state(self, s: I_DTYPE):
        self.position[s] = self.initial_position
        self.rotation[s] = self.initial_rotation
        self.twist[s] = self.initial_twist
        self.wrench[s] = self.initial_wrench

    @ti.kernel
    def forward_kinematics(self, s: I_DTYPE, dt: F_DTYPE):
        # Rigid-body motion
        xyz_limit = ti.Vector([0., 1.], dt=F_DTYPE) # TODO: not sure if we keep this here
        next_s = (s + 1) % self.solver.max_substeps

        self.twist[next_s] = self.twist[s] + self.inv_inertia @ self.wrench[s] * dt
        twist = self.twist[next_s]
        
        v = ti.Vector.zero(F_DTYPE, self.v_dim)
        w = ti.Vector.zero(F_DTYPE, self.w_dim)
        if ti.static(self.solver.dim == 2):
            v = ti.Vector([twist[1], twist[2]])
            w = ti.Vector([twist[0]])
        else:
            v = ti.Vector([twist[3], twist[4], twist[5]])
            w = ti.Vector([twist[0], twist[1], twist[2]])

        self.position[next_s] = max(min(self.position[s] + v * dt, xyz_limit[1]), xyz_limit[0])
        self.rotation[next_s] = self.qmul(self.w2quat(w * dt, F_DTYPE), self.rotation[s])

        if ti.static(not self.solver.needs_grad): # NOTE: reusing cache[s+1] requires reset since we do atomic add in collide function
            self.twist[s] = ti.Vector.zero(F_DTYPE, self.w_dim + self.v_dim)
            self.wrench[s] = ti.Vector.zero(F_DTYPE, self.w_dim + self.v_dim)

    @ti.func
    def collide(self, dt, I, grid_v_I, grid_m_I, s, gravity):
        dt = ti.cast(dt, F_DTYPE)
        dx = ti.cast(self.solver.dx, F_DTYPE)
        collider_grid_m_I = ti.cast(self.solver.p_vol * self.density, F_DTYPE) # NOTE: resemble a small region of object occupying the grid cell

        force = ti.Vector.zero(F_DTYPE, self.v_dim) # Used to compute soft-to-rigid effect
        torque = ti.Vector.zero(F_DTYPE, self.w_dim)
        v_out = grid_v_I

        # Compute SDF
        grid_pos = I * dx
        signed_dist = self.sdf(s, grid_pos)
        
        # Compute collision
        influence = min(ti.exp(-signed_dist * self.softness), 1)

        if grid_m_I > 0 and ((self.softness > 0 and influence > 0.1) or signed_dist <= 0.0):
            # Compute collider velocity at this grid cell
            collider_v_at_grid = ti.Vector.zero(F_DTYPE, self.solver.dim)
            v = ti.Vector.zero(F_DTYPE, self.solver.dim) # linear velocity of rigid body CoM
            wv = ti.Vector.zero(F_DTYPE, self.solver.dim) # point velocity from angular velocity of rigid body
            if ti.static(self.solver.dim == 2):
                v = ti.Vector([self.twist[s][1], self.twist[s][2]], dt=F_DTYPE)
                wv = cross2d(self.twist[s][0], grid_pos - self.position[s])
            else:
                v = ti.Vector([self.twist[s][3], self.twist[s][4], self.twist[s][5]], dt=F_DTYPE)
                w = ti.Vector([self.twist[s][0], self.twist[s][1], self.twist[s][2]], dt=F_DTYPE)
                wv = w.cross(grid_pos - self.position[s])
            collider_v_at_grid = v + wv
            
            # Compute the actual v_in of the grid cell by considering collider velocity (relative velocity of grid cell and collider at this grid cell)
            collider_v_in = grid_v_I - collider_v_at_grid
            
            # Compute velocity as if the collider is (relatively) static; similar to static collider
            n = self.normal(s, grid_pos)
            normal_component = collider_v_in.dot(n)

            collider_v_t = collider_v_in - min(normal_component, 0) * n # normal_component * n # NOTE: not using min(normal_component, 0) * n
            collider_v_t_norm = collider_v_t.norm(NORM_EPS)

            if normal_component < 0 and collider_v_t_norm > 1e-30:
                collider_v_t_normalized = collider_v_t / collider_v_t_norm
                collider_v_t = collider_v_t_normalized * max(0, \
                    collider_v_t_norm + normal_component * ti.cast(self.friction, F_DTYPE))
            
            # Compute final v_out for the grid cell
            v_out = collider_v_at_grid # original grid cell velocity of collider
            v_out += collider_v_in * (1 - influence) # grid cell velocity act upon the collider
            v_out += collider_v_t * influence * (1 - influence) # friction from rigid body to soft body
            v_out += n * influence * influence # normal force that "expel" particle out of rigid body # NOTE: defy physics
            
            # Update wrench based on impact (grid force/velocity applied on rigid collider)
            safe_grid_m_I = min(grid_m_I / collider_grid_m_I, 1) * collider_grid_m_I # NOTE HACK: to prevent highly-compressed/density grid cell that yields super large force on rigid body
            force += -(v_out - grid_v_I) * safe_grid_m_I / dt # force applied on the collider from Newton's 3rd law
            torque += (grid_pos - self.position[s]).cross(force)

        # Apply gravity TODO: need to check if this should affect v_out, e.g., when interacting with soft robot
        if signed_dist <= 0.0:
            force += gravity * collider_grid_m_I

        # Update wrench
        next_s = (s + 1) % self.solver.max_substeps
        if ti.static(self.solver.dim == 2):
            self.wrench[next_s] += ti.Vector([torque[0], force[0], force[1]], dt=F_DTYPE)
        else:
            self.wrench[next_s] += ti.Vector([torque[0], torque[1], torque[2], force[0], force[1], force[2]], dt=F_DTYPE)

        return v_out, signed_dist
    
    @ti.kernel
    def clear(self):
        for s in self.position:
            self.position[s] = ti.Vector.zero(F_DTYPE, self.solver.dim)
            self.rotation[s] = ti.Vector.zero(F_DTYPE, 4)
            self.twist[s] = ti.Vector.zero(F_DTYPE, self.w_dim + self.v_dim)
            self.wrench[s] = ti.Vector.zero(F_DTYPE, self.w_dim + self.v_dim)

    @ti.kernel
    def clear_grad(self):
        for s in self.position:
            self.position.grad[s] = ti.Vector.zero(F_DTYPE, self.solver.dim)
            self.rotation.grad[s] = ti.Vector.zero(F_DTYPE, 4)
            self.twist.grad[s] = ti.Vector.zero(F_DTYPE, self.w_dim + self.v_dim)
            self.wrench.grad[s] = ti.Vector.zero(F_DTYPE, self.w_dim + self.v_dim)

    @ti.func
    def sdf(self, s, grid_pos):
        grid_pos = self.inv_trans(grid_pos, self.position[s], self.rotation[s])
        return self._sdf(s, grid_pos)

    @ti.func
    def normal(self, s, grid_pos):
        grid_pos = self.inv_trans(grid_pos, self.position[s], self.rotation[s])
        return self.qrot(self.rotation[s], self._normal(s, grid_pos))

    #### To be implemented in children class
    @ti.kernel
    def seed(self, s: I_DTYPE, num_new_particles: I_DTYPE, material: I_DTYPE, particle_id: I_DTYPE):
        raise NotImplementedError

    @ti.kernel
    def _inv_inertia(self):
        raise NotImplementedError

    @ti.func
    def _sdf(self, s, grid_pos):
        raise NotImplementedError

    @ti.func
    def _normal(self, s, grid_pos):
        # NOTE: better use analytical form
        d = ti.cast(1e-4, F_DTYPE)
        n = ti.Vector.zero(F_DTYPE, self.solver.dim)
        for i in ti.static(range(self.solver.dim)):
            inc = grid_pos
            dec = grid_pos
            inc[i] += d
            dec[i] -= d
            n[i] = (0.5 / d) * (self._sdf(s, inc) - self._sdf(s, dec))
        return n / n.norm(NORM_EPS)

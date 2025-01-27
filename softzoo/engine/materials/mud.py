from typing import Any
import taichi as ti

from .. import I_DTYPE, F_DTYPE
from . import default


def get_material_model(solver: Any):
    water_E = 1e3
    hardening_coef = 1. # bring down multiplier from 10 to 1 in the exponential for numerical stability
    theta_c = 2.5e-2 # critical compression
    theta_s = 4.5e-3 # critical stretch
    Jp_lower_bound = 0.0 # NOTE: to avoid too large h, which causes nan in backward
    fluid_ratio = 0.5 # interpolation ratio between fluid and granular material

    dim = solver.dim
    dx = solver.dx
    inv_dx = solver.inv_dx
    p_vol = solver.p_vol

    compute_F_tmp, p2g = default.get_material_model(solver)

    @ti.func
    def p2g(
        p: I_DTYPE,
        s: I_DTYPE,
        next_s: I_DTYPE,
        dt: F_DTYPE,
        p_mass: F_DTYPE,
        x: ti.template(),
        v: ti.template(),
        F: ti.template(),
        C: ti.template(),
        Jp: ti.template(),
        F_tmp: ti.template(),
        U: ti.template(),
        sig: ti.template(),
        V: ti.template(),
        actuation: ti.template(),
        mu_: ti.template(),
        lambd_: ti.template(),
        particle_ids: ti.template(),
        grid_v_in: ti.template(),
        grid_m: ti.template(),
    ) -> ti.template():
        # Get Lame parameters
        h = ti.exp(hardening_coef * (1.0 - Jp[s, p]))
        mu = mu_[p] * h
        lambd = lambd_[p] * h

        # Compute determinant (fluid)
        J_fluid = ti.cast(1., F_DTYPE)
        new_sig_fluid = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig_fluid[d, d] = sig[s, p][d, d]
            J_fluid *= new_sig_fluid[d, d]
        
        new_Jp_fluid = J_fluid

        # Compute deformation gradient (fluid)
        new_F_fluid = ti.Matrix.identity(F_DTYPE, dim)
        sqrtJ = ti.sqrt(J_fluid) # TODO need pow(x, 1/3) for 3-dim
        new_F_fluid[0, 0] = sqrtJ
        new_F_fluid[1, 1] = sqrtJ

        # Compute stress (fluid)
        stress_fluid = ti.Matrix.identity(F_DTYPE, dim) * (J_fluid - 1) * ti.cast(water_E, F_DTYPE)

        # Compute determinant (plastic granular)
        J_granular = ti.cast(1., F_DTYPE)
        new_sig_granular = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig_granular[d, d] = ti.min(ti.max(sig[s, p][d, d], 1 - theta_c), 1 + theta_s)  # Plasticity
            J_granular *= new_sig_granular[d, d]

        new_Jp_granular = Jp[s, p]
        for d in ti.static(range(dim)):
            new_Jp_granular *= sig[s, p][d, d] / (new_sig_granular[d, d] + 1e-8)

        # Compute deformation gradient (plastic granular)
        new_F_granular = U[s, p] @ new_sig_granular @ V[s, p].transpose()

        # Update deformation gradient and plastic determinant
        new_Jp = ti.cast(fluid_ratio, F_DTYPE) * new_Jp_fluid + ti.cast(1. - fluid_ratio, F_DTYPE) * new_Jp_granular
        new_F = ti.cast(fluid_ratio, F_DTYPE) * new_F_fluid + ti.cast(1. - fluid_ratio, F_DTYPE) * new_F_granular
        Jp[next_s, p] = ti.max(new_Jp, Jp_lower_bound)
        F[next_s, p] = new_F

        # Compute cauchy (fluid)
        stress_granular = 2 * mu * (new_F_granular - U[s, p] @ V[s, p].transpose()) @ new_F_granular.transpose() + \
                ti.Matrix.identity(F_DTYPE, dim) * lambd * J_granular * (J_granular - 1)
        
        # Compute stress (momentum)
        stress = ti.cast(fluid_ratio, F_DTYPE) * stress_fluid + ti.cast(1. - fluid_ratio, F_DTYPE) * stress_granular
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress
            
    return compute_F_tmp, p2g

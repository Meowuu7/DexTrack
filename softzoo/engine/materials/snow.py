from typing import Any
import taichi as ti

from .. import I_DTYPE, F_DTYPE
from . import default


def get_material_model(solver: Any):
    # Reference: http://alexey.stomakhin.com/research/siggraph2013_snow.pdf

    hardening_coef = 1. # bring down multiplier from 10 to 1 in the exponential for numerical stability
    theta_c = 2.5e-2 # critical compression
    theta_s = 4.5e-3 # critical stretch
    Jp_lower_bound = 0.9 # NOTE: to avoid too large h, which causes nan in backward

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

        # Compute determinant
        J = ti.cast(1., F_DTYPE)
        new_sig = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig[d, d] = ti.min(ti.max(sig[s, p][d, d], 1 - theta_c), 1 + theta_s)  # Plasticity
            J *= new_sig[d, d]

        new_Jp = Jp[s, p]
        for d in ti.static(range(dim)):
            new_Jp *= sig[s, p][d, d] / (new_sig[d, d] + 1e-8)
        Jp[next_s, p] = ti.max(new_Jp, Jp_lower_bound)

        # Compute deformation gradient
        new_F = U[s, p] @ new_sig @ V[s, p].transpose()
        F[next_s, p] = new_F

        # Compute stress
        stress = 2 * mu * (new_F - U[s, p] @ V[s, p].transpose()) @ new_F.transpose() + \
                ti.Matrix.identity(F_DTYPE, dim) * lambd * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress
            
    return compute_F_tmp, p2g

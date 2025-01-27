from typing import Any
import taichi as ti

from .. import I_DTYPE, F_DTYPE, NORM_EPS
from . import default


def get_material_model(solver: Any):
    yield_stress = 200.

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
        h = ti.cast(1., F_DTYPE)
        mu = mu_[p] * h
        lambd = lambd_[p] * h

        # Compute deformation gradient (von mises)
        epsilon = ti.Vector.zero(F_DTYPE, dim)
        for d in ti.static(range(dim)):
            epsilon[d] = ti.log(sig[s, p][d, d])
        epsilon_hat = epsilon - (epsilon.sum() / ti.cast(dim, F_DTYPE))
        epsilon_hat_norm = epsilon_hat.norm(NORM_EPS)
        delta_gamma = epsilon_hat_norm - ti.cast(yield_stress, F_DTYPE) / (2 * mu)

        epsilon -= (delta_gamma / epsilon_hat_norm) * epsilon_hat
        epsilon_exp = ti.exp(epsilon)
        mask = ti.cast(epsilon_hat_norm > 0, F_DTYPE) # NOTE: perhaps global data cannot be used in condition in ti func
        new_sig = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig[d, d] = mask * epsilon_exp[d] + (1 - mask) * sig[s, p][d, d]

        new_F = U[s, p] @ new_sig @ V[s, p].transpose()
        F[next_s, p] = new_F

        # Compute determinant
        J = new_F.determinant()
        Jp[next_s, p] = J

        # Compute stress
        stress = 2 * mu * (new_F - U[s, p] @ V[s, p].transpose()) @ new_F.transpose() + \
                ti.Matrix.identity(F_DTYPE, dim) * lambd * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress
            
    return compute_F_tmp, p2g

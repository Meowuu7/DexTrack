from typing import Any
import math
import taichi as ti

from .. import I_DTYPE, F_DTYPE, NORM_EPS
from . import default


def get_material_model(solver: Any):
    friction_angle = math.radians(45)
    sin_phi = math.sin(friction_angle)
    alpha = math.sqrt(2 / 3) * 2 * sin_phi / (3 - sin_phi) # sand parameters

    dim = solver.dim
    dx = solver.dx
    inv_dx = solver.inv_dx
    p_vol = solver.p_vol

    compute_F_tmp, p2g = default.get_material_model(solver)

    @ti.func
    def sand_projection(sigma, Jp_old, mu, lambd):
        sigma_out = ti.Matrix.zero(F_DTYPE, dim, dim)
        epsilon = ti.Vector.zero(F_DTYPE, dim)
        for i in ti.static(range(dim)):
            epsilon[i] = ti.log(ti.max(ti.abs(sigma[i, i]), 1e-4))
            sigma_out[i, i] = ti.cast(1, F_DTYPE)
        tr = epsilon.sum() + Jp_old
        epsilon_hat = epsilon - tr / dim
        epsilon_hat_norm = epsilon_hat.norm(eps=NORM_EPS) # NOTE: always use safe-guard eps in norm

        Jp = ti.max(tr, 0.0)

        mask = ti.cast(tr < 0.0, F_DTYPE)
        delta_gamma = epsilon_hat_norm + (dim * lambd + 2 * mu) / (2 * mu) * tr * alpha
        sig_vec = ti.exp(epsilon - ti.max(0, delta_gamma) / epsilon_hat_norm * epsilon_hat)

        sigma_out = ti.Matrix.zero(F_DTYPE, dim, dim)
        for i in ti.static(range(dim)):
            sigma_out[i, i] = mask * sig_vec[i] + (1. - mask)

        return sigma_out, Jp

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
        h = ti.cast(1., F_DTYPE) # NOTE: we are using mu_0 and lambd_0 for sand
        mu = mu_[p] * h
        lambd = lambd_[p] * h

        # Compute determinant
        new_sig, new_Jp = sand_projection(sig[s, p], Jp[s, p], mu, lambd)

        Jp[next_s, p] = new_Jp

        # Compute deformation gradient
        new_F = U[s, p] @ new_sig @ V[s, p].transpose()
        F[next_s, p] = new_F

        # Compute stress
        stress = ti.Matrix.zero(F_DTYPE, dim, dim)
        log_sig_sum = ti.cast(0.0, F_DTYPE)
        center = ti.Matrix.zero(F_DTYPE, dim, dim)
        for i in ti.static(range(dim)):
            log_sig_sum += ti.log(new_sig[i, i])
            center[i, i] = 2.0 * mu * ti.log(new_sig[i, i]) * (1 / new_sig[i, i])
        for i in ti.static(range(dim)):
            center[i, i] += lambd * log_sig_sum * (1 / new_sig[i, i])
        stress = U[s, p] @ center @ V[s, p].transpose() @ new_F.transpose()
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress

    return compute_F_tmp, p2g

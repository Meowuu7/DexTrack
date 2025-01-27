from typing import Any
import taichi as ti

from .. import I_DTYPE, F_DTYPE


def get_material_model(solver: Any):
    dim = solver.dim
    dx = solver.dx
    inv_dx = solver.inv_dx
    p_vol = solver.p_vol

    @ti.func
    def compute_F_tmp(
        p: I_DTYPE,
        s: I_DTYPE,
        dt: F_DTYPE,
        F: ti.template(),
        F_tmp: ti.template(),
        Jp: ti.template(),
        C: ti.template(),
    ):
        new_F_tmp = F[s, p]
        new_F_tmp = (ti.Matrix.identity(F_DTYPE, dim) + dt * C[s, p]) @ new_F_tmp
        F_tmp[s, p] = new_F_tmp

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
        h = ti.cast(1., F_DTYPE) # hardening coefficient
        mu = mu_[p] * h
        lambd = lambd_[p] * h

        # Compute determinant
        J = ti.cast(1., F_DTYPE)
        new_sig = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig[d, d] = sig[s, p][d, d]
            J *= new_sig[d, d]

        new_Jp = Jp[s, p]
        for d in ti.static(range(dim)):
            new_Jp *= sig[s, p][d, d] / (new_sig[d, d] + 1e-8)

        Jp[next_s, p] = new_Jp

        # Compute deformation gradient
        new_F = U[s, p] @ new_sig @ V[s, p].transpose()
        F[next_s, p] = new_F

        # Compute stress
        stress = 2 * mu * (new_F - U[s, p] @ V[s, p].transpose()) @ new_F.transpose() + \
                ti.Matrix.identity(F_DTYPE, dim) * lambd * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress

    return compute_F_tmp, p2g

from typing import Any
import taichi as ti

from .. import I_DTYPE, F_DTYPE
from . import default


def get_material_model(solver: Any):
    water_E = 1e3

    dim = solver.dim
    dx = solver.dx
    inv_dx = solver.inv_dx
    p_vol = solver.p_vol

    compute_F_tmp, p2g = default.get_material_model(solver)

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
        new_F_tmp = ti.Matrix.identity(F_DTYPE, dim)
        new_F_tmp[0, 0] = Jp[s, p]
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
        # Compute determinant
        J = ti.cast(1., F_DTYPE)
        new_sig = ti.Matrix.zero(F_DTYPE, dim, dim)
        for d in ti.static(range(dim)):
            new_sig[d, d] = sig[s, p][d, d]
            J *= new_sig[d, d]

        Jp[next_s, p] = J

        # Compute deformation gradient
        new_F = ti.Matrix.identity(F_DTYPE, dim)
        sqrtJ = ti.sqrt(J) # TODO need pow(x, 1/3) for 3-dim
        new_F[0, 0] = sqrtJ
        new_F[1, 1] = sqrtJ

        F[next_s, p] = new_F

        # Compute stress
        stress = ti.Matrix.identity(F_DTYPE, dim) * (J - 1) * ti.cast(water_E, F_DTYPE)
        stress = (-dt * p_vol * 4 * inv_dx**2) * stress

        return stress
            
    return compute_F_tmp, p2g

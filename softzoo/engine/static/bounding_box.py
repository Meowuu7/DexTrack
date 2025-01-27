import taichi as ti

from ...configs.item_configs.static import get_cfg_defaults
from ...tools.general_utils import merge_cfg


def add(solver, cfg):
    default_cfg = get_cfg_defaults(cfg.type)
    merge_cfg(default_cfg, cfg, replace=True)

    solver.static_component_info.append(cfg)
    
    @ti.func
    def collide(dt, I, grid_v_I, grid_m_I, s):
        grid_v_out_I = grid_v_I
        signed_dist = 0.
        for d in ti.static(range(solver.dim)):
            if (I[d] < solver.padding[d]) or (I[d] >= (solver.res[d] - solver.padding[d])): # used for non-overlapping initialization
                signed_dist -= 1.

            if I[d] < solver.padding[d] and grid_v_I[d] < 0:
                grid_v_out_I[d] = 0  # Boundary conditions
            if I[d] >= (solver.res[d] - solver.padding[d]) and grid_v_I[d] > 0:
                grid_v_out_I[d] = 0

        return grid_v_out_I, signed_dist # NOTE: signed distance is dummy here

    solver.grid_process_static.append(collide)

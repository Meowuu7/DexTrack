import enum
import math
import taichi as ti

from .. import NORM_EPS, F_DTYPE
from ...configs.item_configs.static import get_cfg_defaults
from ...tools.general_utils import merge_cfg


class Surface(enum.Enum):
    Sticky = 0
    Slip = 1
    Separate = 2
    PseudoPassiveDynamics = 3


def add(solver, cfg):
    # Parse configuration
    default_cfg = get_cfg_defaults(cfg.type)
    merge_cfg(default_cfg, cfg, replace=True)

    point = cfg.point
    normal = cfg.normal
    surface = cfg.surface
    friction = cfg.friction
    pseudo_passive_velocity = cfg.pseudo_passive_velocity

    # Preprocess
    cls_name, type_name = surface.split('.')
    surface = getattr(globals()[cls_name], type_name).value

    if surface == Surface.Sticky.value:
        assert friction == 0.0, 'Friction must be 0 on sticky surfaces.'
        
    normal_scale = 1.0 / math.sqrt(sum([x**2 for x in cfg.normal]))
    normal = list(normal_scale * x for x in normal)
    
    solver.static_component_info.append({
        'type': cfg.type,
        'points': [point],
        'normals': [normal],
        'item_id': cfg.item_id,
        'semantic_id': cfg.semantic_id,
    })

    # Define surface collider
    @ti.func
    def collide(dt, I, grid_v_I, grid_m_I, s):
        offset = I * solver.dx - ti.Vector(point)
        n = ti.Vector(normal)
        grid_v_out_I = grid_v_I
        signed_dist = offset.dot(n)
        if signed_dist < 0:
            if ti.static(surface == Surface.Sticky.value):
                grid_v_out_I = ti.Vector.zero(F_DTYPE, solver.dim)
            else:
                v = grid_v_I
                normal_component = n.dot(v)

                if ti.static(surface == Surface.Slip.value):
                    # Project out all normal component
                    v = v - n * normal_component
                elif ti.static(surface == Surface.PseudoPassiveDynamics.value):
                    ppv = ti.Vector(pseudo_passive_velocity, F_DTYPE)
                    ppv = ppv - n * min(n.dot(ppv), ti.cast(0, F_DTYPE))
                    v = v - n * normal_component + ppv
                else:
                    # Project out only inward normal component
                    v = v - n * min(normal_component, 0)

                if normal_component < 0 and v.norm() > 1e-30:
                    # Apply friction here
                    v = v.normalized(NORM_EPS) * max(
                        0,
                        v.norm(NORM_EPS) + normal_component * friction)

                grid_v_out_I = v
        return grid_v_out_I, signed_dist

    solver.grid_process_static.append(collide)

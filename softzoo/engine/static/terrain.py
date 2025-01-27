import noise
import numpy as np
import taichi as ti

from .flat_surface import Surface
from .. import I_DTYPE, F_DTYPE, NORM_EPS
from ...configs.item_configs.static import get_cfg_defaults
from ...tools.general_utils import merge_cfg


def add(solver, cfg):
    # Parse configuration
    default_cfg = get_cfg_defaults(cfg.type)
    merge_cfg(default_cfg, cfg, replace=True)

    surface = cfg.surface
    cls_name, type_name = surface.split('.')
    surface = getattr(globals()[cls_name], type_name).value

    friction = cfg.friction
    pseudo_passive_velocity = cfg.pseudo_passive_velocity
    min_height = cfg.min_height
    max_height = cfg.max_height
    semantic_id = cfg.semantic_id

    resolution = cfg.resolution
    scale = cfg.scale
    octaves = cfg.octaves
    persistence = cfg.persistence
    lacunarity = cfg.lacunarity
    repeat = cfg.repeat

    # Get points representation with Perlin noise
    f_dtype_np = np.float64 if F_DTYPE == ti.f64 else np.float32

    def norm_fn(_x, _new_min=0., _new_max=1., _x_min=None, _x_max=None): # normalize surface to min/max height
        if _x_min is None: _x_min = _x.min()
        if _x_max is None: _x_max = _x.max()
        _out = (_x - _x_min) / (_x_max - _x_min)
        _out = _out * (_new_max - _new_min) + _new_min
        return _out, (_x_min, _x_max)

    padded_mul = 3 # NOTE: need to be an odd number
    assert padded_mul % 2 == 1
    if solver.dim == 2: # make global for in-place modification
        surface_np = np.zeros((resolution, 2), dtype=f_dtype_np)
        surface_np_padded = np.zeros((resolution * padded_mul, 2), dtype=f_dtype_np)
    else:
        surface_np = np.zeros((resolution, resolution, 3), dtype=f_dtype_np)
        surface_np_padded = np.zeros((resolution * padded_mul, resolution * padded_mul, 3), dtype=f_dtype_np)

    def generate_surface():
        base = np.random.randint(0, repeat) # TODO: not sure if this is the best parameters to be randomized
        if solver.dim == 2:
            for i in range(resolution):
                surface_np[i, 0] = i
                surface_np[i, 1] = noise.pnoise1(i/scale,
                                                 octaves=octaves,
                                                 persistence=persistence,
                                                 lacunarity=lacunarity,
                                                 repeat=repeat,
                                                 base=base)
            
            padding_x_frac = solver.padding[0] / solver.n_grid
            surface_np[..., 0], (x_min, x_max) = norm_fn(surface_np[..., 0], padding_x_frac, 1. - padding_x_frac)
            surface_np[..., 1], (y_min, y_max) = norm_fn(surface_np[..., 1], min_height, max_height)

            half_padded_mul = (padded_mul - 1) // 2
            for i in range(padded_mul * resolution):
                non_padded_i = i - half_padded_mul * resolution
                surface_np_padded[i, 0] = non_padded_i
                surface_np_padded[i, 1] = noise.pnoise1(non_padded_i/scale,
                                                        octaves=octaves,
                                                        persistence=persistence,
                                                        lacunarity=lacunarity,
                                                        repeat=repeat,
                                                        base=base)
            surface_np_padded[..., 0] = norm_fn(surface_np_padded[..., 0], padding_x_frac, 1. - padding_x_frac, x_min, x_max)
            surface_np_padded[..., 1] = norm_fn(surface_np_padded[..., 1], min_height, max_height, y_min, y_max)
            assert np.allclose(surface_np_padded[((padded_mul - 1) // 2) * resolution : ((padded_mul - 1) // 2 + 1) * resolution], surface_np)
        else:
            for i in range(resolution):
                for j in range(resolution):
                    surface_np[i, j, 0] = i
                    surface_np[i, j, 2] = j
                    surface_np[i, j, 1] = noise.pnoise2(i/scale, # height is y-component in mpm
                                                        j/scale,
                                                        octaves=octaves,
                                                        persistence=persistence,
                                                        lacunarity=lacunarity,
                                                        repeatx=repeat,
                                                        repeaty=repeat,
                                                        base=base)

            padding_x_frac = solver.padding[0] / solver.n_grid
            padding_z_frac = solver.padding[2] / solver.n_grid
            surface_np[..., 0], (x_min, x_max) = norm_fn(surface_np[..., 0], padding_x_frac, 1. - padding_x_frac)
            surface_np[..., 1], (y_min, y_max) = norm_fn(surface_np[..., 1], min_height, max_height)
            surface_np[..., 2], (z_min, z_max) = norm_fn(surface_np[..., 2], padding_z_frac, 1. - padding_z_frac)

            half_padded_mul = (padded_mul - 1) // 2
            for i in range(padded_mul * resolution):
                for j in range(padded_mul * resolution):
                    non_padded_i = i - half_padded_mul * resolution
                    non_padded_j = j - half_padded_mul * resolution
                    surface_np_padded[i, j, 0] = non_padded_i
                    surface_np_padded[i, j, 2] = non_padded_j
                    surface_np_padded[i, j, 1] = noise.pnoise2(non_padded_i/scale, # height is y-component in mpm
                                                               non_padded_j/scale,
                                                               octaves=octaves,
                                                               persistence=persistence,
                                                               lacunarity=lacunarity,
                                                               repeatx=repeat,
                                                               repeaty=repeat,
                                                               base=base)
            surface_np_padded[..., 0], _ = norm_fn(surface_np_padded[..., 0], padding_x_frac, 1. - padding_x_frac, x_min, x_max)
            surface_np_padded[..., 1], _ = norm_fn(surface_np_padded[..., 1], min_height, max_height, y_min, y_max)
            surface_np_padded[..., 2], _ = norm_fn(surface_np_padded[..., 2], padding_z_frac, 1. - padding_z_frac, z_min, z_max)
            assert np.allclose(surface_np_padded[((padded_mul - 1) // 2) * resolution : ((padded_mul - 1) // 2 + 1) * resolution,((padded_mul - 1) // 2) * resolution : ((padded_mul - 1) // 2 + 1) * resolution], surface_np)

    # Compute surface normal
    @ti.kernel
    def polysurface_points_to_normals(points: ti.template(), normals: ti.template()):
        for I in ti.grouped(points):
            if ti.static(solver.dim == 3):
                tan_right = points[I[0], ti.min(I[1] + 1, points.shape[1] - 1)] - points[I]
                if I[1] + 1 >= points.shape[1]: # at the right boundary
                    tan_left = points[I[0], ti.max(0, I[1] - 1)] - points[I]
                    tan_right = -tan_left

                tan_down = points[ti.min(I[0] + 1, points.shape[0] - 1), I[1]] - points[I] # NOTE: [0, 0] is at upper-left corner
                if I[0] + 1 >= points.shape[0]: # at the bottom boundary
                    tan_up = points[ti.max(I[0] - 1, 0), I[1]] - points[I]
                    tan_down = -tan_up
                
                normals[I] = tan_right.cross(tan_down) # left down also looks okay
            else: # 2D
                tan = points[I + 1] - points[I]
                normals[I] = ti.Vector([-tan[1], tan[0]])

            normals[I] = normals[I].normalized() # NOTE: using normals[I] / normals[I].norm(NORM_EPS) will generate incorrect results

    polysurface_points = ti.Vector.field(solver.dim, dtype=F_DTYPE, shape=[resolution] * (solver.dim - 1))
    polysurface_normals = ti.Vector.field(polysurface_points.n, dtype=F_DTYPE, shape=[resolution] * (solver.dim - 1))
    
    def generate_polysurface(gen_surface=True):
        if gen_surface:
            generate_surface() # in-place modification
        polysurface_points.from_numpy(surface_np)
        polysurface_points_to_normals(polysurface_points, polysurface_normals)

    generate_polysurface()
    
    solver.static_component_info.append({
        'type': cfg.type,
        'points': surface_np, # NOTE: this is modified in-place
        'points_padded': surface_np_padded, # NOTE: this is modified in-place
        'padded_mul': padded_mul,
        'item_id': cfg.item_id,
        'semantic_id': semantic_id,
        'reset_fn': generate_polysurface,
        'polysurface_points': polysurface_points,
        'polysurface_normals': polysurface_normals,
        'resolution': resolution,
    })

    # Define surface collider
    @ti.func
    def collide(dt, I, grid_v_I, grid_m_I, s):
        dx = ti.cast(solver.dx, F_DTYPE)
        
        points_I_frac = ti.Vector.zero(F_DTYPE, solver.dim - 1)
        # NOTE: I * self.dx ranges from 0 to 1
        if ti.static(solver.dim == 2):
            padding_frac_x = solver.padding[0] / solver.n_grid
            points_I_frac[0] = (ti.cast(I[0], F_DTYPE) * dx - padding_frac_x) / (1 - 2 * padding_frac_x)
            points_I_frac[0] = points_I_frac[0] * ti.cast(polysurface_points.shape[0], F_DTYPE)
        else:
            padding_frac_x = solver.padding[0] / solver.n_grid
            padding_frac_z = solver.padding[2] / solver.n_grid
            points_I_frac[0] = (ti.cast(I[0], F_DTYPE) * dx - padding_frac_x) / (1 - 2 * padding_frac_x)
            points_I_frac[1] = (ti.cast(I[2], F_DTYPE) * dx - padding_frac_z) / (1 - 2 * padding_frac_z)
            
            points_I_frac[0] = points_I_frac[0] * ti.cast(polysurface_points.shape[0], F_DTYPE)
            points_I_frac[1] = points_I_frac[1] * ti.cast(polysurface_points.shape[1], F_DTYPE)
        points_I = points_I_frac.cast(int)
        point = polysurface_points[points_I]
        offset = I * dx - point

        n = polysurface_normals[points_I] # normalized
        
        grid_v_out_I = grid_v_I
        signed_dist = offset.dot(n)
        if signed_dist < cfg.signed_dist_thresh:
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
                    ppv = ppv - n * ti.min(n.dot(ppv), ti.cast(0, F_DTYPE))
                    v = v - n * normal_component + ppv
                else:
                    # Project out only inward normal component
                    v = v - n * ti.min(normal_component, ti.cast(0, F_DTYPE))
                
                if normal_component < 0 and v.norm() > 1e-30:
                    # Apply friction here
                    v = v.normalized(NORM_EPS) * ti.max(
                        ti.cast(0, F_DTYPE),
                        v.norm(NORM_EPS) + normal_component * ti.cast(friction, F_DTYPE))

                grid_v_out_I = v

        return grid_v_out_I, signed_dist

    solver.grid_process_static.append(collide)

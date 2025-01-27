from yacs.config import CfgNode as CN

from .particle_group_info import get_cfg_defaults as get_cfg_defaults_particle_group_info
from .primitives import get_cfg_defaults as get_cfg_defaults_primitives


__C = CN()

### Default
__C.DEFAULT = CN()
__C.DEFAULT.n_actuators = 10
__C.DEFAULT.p_rho_lower_bound_mul = 0.1
__C.DEFAULT.initial_principle_direction = None

__C.DEFAULT.base_shape = CN()
__C.DEFAULT.base_shape.material = 'SimpleMuscle'
__C.DEFAULT.base_shape.sample_density = 4
__C.DEFAULT.base_shape.particle_info = get_cfg_defaults_particle_group_info()
__C.DEFAULT.base_shape.particle_info.muscle_direction = [0., 1., 0.]
__C.DEFAULT.base_shape.initial_position = [0.1, 0.1, 0.1]
__C.DEFAULT.base_shape.initial_velocity = [0., 0., 0.]
__C.DEFAULT.base_shape.semantic_id = 1

### Voxel-based representation
__C.VBR = CN()
__C.VBR.voxel_resolution = [16, 16, 16]

__C.VBR.base_shape = get_cfg_defaults_primitives('Box')
__C.VBR.base_shape.type = 'Primitive.Box'
__C.VBR.base_shape.size = [0.1, 0.1, 0.1]
__C.VBR.base_shape.particle_id = 1
__C.VBR.base_shape.material = 'SimpleMuscle'
__C.VBR.base_shape.particle_info.muscle_direction = [0., 1., 0.]

### Particle-based representation
__C.PBR = CN()
__C.PBR.base_shape = get_cfg_defaults_primitives('Box')
__C.PBR.base_shape.type = 'Primitive.Box'
__C.PBR.base_shape.size = [0.1, 0.1, 0.1]
__C.PBR.base_shape.particle_id = 1
__C.PBR.base_shape.material = 'SimpleMuscle'
__C.PBR.base_shape.particle_info.muscle_direction = [0., 1., 0.]

### Dummy representation
__C.DUMMY = CN()


def get_cfg_defaults(item_type='DEFAULT', pbr_successor_type=None, vbr_successor_type=None):
    if pbr_successor_type is not None:
        __C.PBR.base_shape = get_cfg_defaults_primitives(pbr_successor_type)
        __C.PBR.base_shape.particle_id = 1
        __C.PBR.base_shape.material = 'SimpleMuscle'
        __C.PBR.base_shape.particle_info.muscle_direction = [0., 1., 0.]

    if vbr_successor_type is not None:
        __C.VBR.base_shape = get_cfg_defaults_primitives(vbr_successor_type)
        __C.VBR.base_shape.particle_id = 1
        __C.VBR.base_shape.material = 'SimpleMuscle'
        __C.VBR.base_shape.particle_info.muscle_direction = [0., 1., 0.]

    item_type = item_type.upper()
    other_cfg = getattr(__C, item_type).clone()

    cfg = __C.DEFAULT.clone()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(other_cfg)
    cfg.set_new_allowed(False)

    return cfg

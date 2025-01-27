from yacs.config import CfgNode as CN

from .particle_group_info import get_cfg_defaults as get_cfg_defaults_particle_group_info


__C = CN()

### Default
__C.DEFAULT = CN()
__C.DEFAULT.type = 'Primitive.Unidentified'
__C.DEFAULT.spawn_order = 0
__C.DEFAULT.particle_id = 0
__C.DEFAULT.particle_info = get_cfg_defaults_particle_group_info()
__C.DEFAULT.semantic_id = 0
__C.DEFAULT.item_id = -1 # item id will be assigned dynamically

__C.DEFAULT.material = 'Elastic'
__C.DEFAULT.sample_density = 1
__C.DEFAULT.density = 1.0 # for rigid-body
__C.DEFAULT.friction = 0.9
__C.DEFAULT.softness = 0.

__C.DEFAULT.initial_position = [0.3, 0.3]
__C.DEFAULT.initial_rotation = [1., 0., 0., 0.]
__C.DEFAULT.initial_velocity = [0., 0.] # for particles
__C.DEFAULT.initial_twist = [0., 0., 0.] # for rigid-body, 3-dim vector in 2D, 6-dim vector in 3D
__C.DEFAULT.initial_wrench = [0., 0., 0.]

### Box
__C.BOX = CN()
__C.BOX.type = 'Primitive.Box'
__C.BOX.size = [0.1, 0.1]

### Sphere
__C.SPHERE = CN()
__C.SPHERE.type = 'Primitive.Sphere'
__C.SPHERE.radius = 0.1

### Mesh
__C.MESH = CN()
__C.MESH.type = 'Primitive.Mesh'
__C.MESH.file_path = None
__C.MESH.scale = [1., 1., 1.]
__C.MESH.offset = [0.5, 0.5, 0.5] # Offset when loading the mesh; make sure triangles lie in [0, 0, 0] and [1, 1, 1]
__C.MESH.voxelizer_super_sample = 2

### Ellipsoid
__C.ELLIPSOID = CN()
__C.ELLIPSOID.type = 'Primitive.Ellipsoid'
__C.ELLIPSOID.radius = [0.1, 0.1, 0.1]


def get_cfg_defaults(item_type='DEFAULT'):
    item_type = item_type.replace('Primitive.', '').upper()
    other_cfg = getattr(__C, item_type).clone()

    cfg = __C.DEFAULT.clone()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(other_cfg)
    cfg.set_new_allowed(False)

    return cfg

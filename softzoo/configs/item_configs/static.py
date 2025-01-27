from yacs.config import CfgNode as CN


__C = CN()

# Default configuration
__C.DEFAULT = CN()
__C.DEFAULT.type = 'Unidentified'
__C.DEFAULT.semantic_id = 0
__C.DEFAULT.item_id = -1 # item id will be assigned dynamically

# Bounding box
__C.BOUNDINGBOX = CN()
__C.BOUNDINGBOX.type = 'Static.BoundingBox'

# Flat surface
__C.FLATSURFACE = CN()
__C.FLATSURFACE.type = 'Stataic.FlatSurface'
__C.FLATSURFACE.point = [0., 0.01]
__C.FLATSURFACE.normal = [0., 1.]
__C.FLATSURFACE.surface = 'Surface.Slip'
__C.FLATSURFACE.friction = 0.0
__C.FLATSURFACE.pseudo_passive_velocity = [1., 0.]

# Terrain
__C.TERRAIN = CN()
__C.TERRAIN.type = 'Static.Terrain'
__C.TERRAIN.surface = 'Surface.Slip'
__C.TERRAIN.friction = 0.0
__C.TERRAIN.pseudo_passive_velocity = [1., 0.]
__C.TERRAIN.min_height = 0.1
__C.TERRAIN.max_height = 0.2
__C.TERRAIN.resolution = 512
__C.TERRAIN.scale = 1000. # larger scale for simpler terrain
__C.TERRAIN.octaves = 6
__C.TERRAIN.persistence = 0.5
__C.TERRAIN.lacunarity = 2.0
__C.TERRAIN.repeat = 1024
__C.TERRAIN.signed_dist_thresh = 0.0


def get_cfg_defaults(item_type='DEFAULT'):
    item_type = item_type.replace('Static.', '').upper()
    other_cfg = getattr(__C, item_type).clone()

    cfg = __C.DEFAULT.clone()
    cfg.set_new_allowed(True)
    cfg.merge_from_other_cfg(other_cfg)
    cfg.set_new_allowed(False)

    return cfg

from yacs.config import CfgNode as CN


__C = CN()

### Simulator
__C.SIMULATOR = CN()
__C.SIMULATOR.use_dynamic_field = True
__C.SIMULATOR.needs_grad = True
__C.SIMULATOR.use_checkpointing = True
__C.SIMULATOR.dim = 3 # world dimension
__C.SIMULATOR.padding = [0, 0, 0]
__C.SIMULATOR.quality = 1. # determine grid size
__C.SIMULATOR.max_substeps = 8192
__C.SIMULATOR.max_substeps_local = 8192
__C.SIMULATOR.checkpoint_cache_device = 'torch_cpu'
__C.SIMULATOR.max_num_particles = 2**20
__C.SIMULATOR.default_dt = 0.00015625 # time interval of a substep
__C.SIMULATOR.gravity = (0., -9.8, 0.)
__C.SIMULATOR.p_rho_0 = 1e3 # default particle density (mass / volume)
__C.SIMULATOR.E_0 = 1e5 # default Young's modulus
__C.SIMULATOR.nu_0 = 0.2 # default Poisson's ratio
__C.SIMULATOR.max_actuation = 10 # maximal number of actuation
__C.SIMULATOR.base_active_materials = [] # will be appended dynamically by items' materials

### Environment
__C.ENVIRONMENT = CN()
__C.ENVIRONMENT.frame_dt = 8e-3
__C.ENVIRONMENT.use_semantic_occupancy = True
__C.ENVIRONMENT.use_renderer = True
__C.ENVIRONMENT.actuation_strength = 0.5
__C.ENVIRONMENT.observation_space = ['time']

__C.ENVIRONMENT.design_space = 'pbr'
__C.ENVIRONMENT.design_space_config = CN(new_allowed=True)

__C.ENVIRONMENT.objective = None
__C.ENVIRONMENT.objective_config = CN(new_allowed=True)

__C.ENVIRONMENT.ITEMS = list()

__C.ENVIRONMENT.CUSTOM = CN(new_allowed=True)

### Renderer
__C.RENDERER = CN()
__C.RENDERER.type = 'gui' # gui / ggui / ray_tracing

__C.RENDERER.GUI = CN()
__C.RENDERER.GUI.fps = 30
__C.RENDERER.GUI.title = 'GUI Renderer'
__C.RENDERER.GUI.res = 512
__C.RENDERER.GUI.background_color = 0x112F41
__C.RENDERER.GUI.particle_colors = (0x068587, 0xED553B, 0xEEEEF0, 0xFFFF00, 0x000000, 0xFF0000, 0x808080)
__C.RENDERER.GUI.static_component_color = 0x000000
__C.RENDERER.GUI.rigid_body_color = 0x000000
__C.RENDERER.GUI.circle_radius = 1.5

__C.RENDERER.GGUI = CN()
__C.RENDERER.GGUI.title = 'GGUI Renderer'
__C.RENDERER.GGUI.tmp_fpath = '/tmp/tmp.png'
__C.RENDERER.GGUI.offscreen_rendering = True
__C.RENDERER.GGUI.save_to_video = True
__C.RENDERER.GGUI.fps = 30
__C.RENDERER.GGUI.res = (640, 480)
__C.RENDERER.GGUI.ambient_light = (0.5, 0.5, 0.5)
__C.RENDERER.GGUI.camera_position = [0.15, 0.75, -0.45]
__C.RENDERER.GGUI.camera_lookat = [0.425, 0.309, 0.17]
__C.RENDERER.GGUI.camera_fov = 55
__C.RENDERER.GGUI.particle_radius = 0.01
__C.RENDERER.GGUI.particle_colors = [
    [0.023, 0.522, 0.490, 0.2], # water
    [0.925, 0.329, 0.231, 1.0], # elastic
    [0.933, 0.933, 0.941, 1.], # snow
    [0.94901961, 0.82352941, 0.6627451, 1.0], # sand
    [0.0, 0.0, 0.0, 1.0], # stationary
    [1.0, 0.0, 0.0, 1.0], # simple muscle
    [0.945, 0.761, 0.490, 1.0], # fake rigid
    [1.0, 0.0, 0.0, 1.0], # diffaqua muscle
    [0.388, 0.349, 0.321, 1.0], # mud
    [0.353, 0.263, 0.196, 1.0], # plasticine
]
__C.RENDERER.GGUI.particle_coloring_mode = ['material', 'particle_id', 'particle_density', 'actuation'][0]
__C.RENDERER.GGUI.ground_surface_cmap = 'Greys'
__C.RENDERER.GGUI.meshify_particle_ids = []
__C.RENDERER.GGUI.meshification_colors = []
__C.RENDERER.GGUI.background = None # set to None for not using a skybox
__C.RENDERER.GGUI.background_color = (0., 0., 0.)

__C.RENDERER.GL = CN()
__C.RENDERER.GL.save_to_video = True
__C.RENDERER.GL.dump_data = False
__C.RENDERER.GL.fps = 30
__C.RENDERER.GL.res = (640, 480)
__C.RENDERER.GL.camera_position = [0.15, 0.75, -0.45]
__C.RENDERER.GL.camera_lookat = [0.425, 0.309, 0.17]
__C.RENDERER.GL.camera_fov = 50
__C.RENDERER.GL.draw_plane = False
__C.RENDERER.GL.light_position = [0.5, 5.0, 0.5]
__C.RENDERER.GL.light_lookat = [0.5, 0.5, 0.49]
__C.RENDERER.GL.light_fov = 50
__C.RENDERER.GL.ground_surface_cmap = 'texture/soil1_512x512.png'
__C.RENDERER.GL.ground_surface_brightness_increase = 0
__C.RENDERER.GL.background = 'skybox/vary_sky.jpg' # set to None for not using a skybox
__C.RENDERER.GL.background_brightness_increase = 0
__C.RENDERER.GL.tile_texture = False
__C.RENDERER.GL.msaa_samples = 8
__C.RENDERER.GL.anisotropy_scale = 1.0 # 1.0 (default) 0.5 (mud) 0.1 (snow/plasticine)
__C.RENDERER.GL.smoothing = 0.5
__C.RENDERER.GL.rendering_scale = 2.0
__C.RENDERER.GL.fluid_rest_distance = 0.0125
__C.RENDERER.GL.gl_color_gamma = 3.5
__C.RENDERER.GL.bodies_info = CN()
__C.RENDERER.GL.bodies_info.Water = CN()
__C.RENDERER.GL.bodies_info.Water.draw_density = False
__C.RENDERER.GL.bodies_info.Water.draw_diffuse = False
__C.RENDERER.GL.bodies_info.Water.draw_ellipsoids = True
__C.RENDERER.GL.bodies_info.Water.draw_points = False
__C.RENDERER.GL.bodies_info.Water.needs_smoothing = True
__C.RENDERER.GL.bodies_info.Water.particle_color = [0.023, 0.522, 0.490, 0.2]
__C.RENDERER.GL.bodies_info.Water.particle_radius = 0.0015
__C.RENDERER.GL.bodies_info.Water.anisotropy_scale = 1.0
__C.RENDERER.GL.bodies_info.Elastic = CN()
__C.RENDERER.GL.bodies_info.Elastic.draw_density = False
__C.RENDERER.GL.bodies_info.Elastic.draw_diffuse = False
__C.RENDERER.GL.bodies_info.Elastic.draw_ellipsoids = True
__C.RENDERER.GL.bodies_info.Elastic.draw_points = False
__C.RENDERER.GL.bodies_info.Elastic.needs_smoothing = True
__C.RENDERER.GL.bodies_info.Elastic.particle_color = [0.925, 0.329, 0.231, 1.0]
__C.RENDERER.GL.bodies_info.Elastic.particle_radius = 0.001
__C.RENDERER.GL.bodies_info.Elastic.anisotropy_scale = 1.2
__C.RENDERER.GL.bodies_info.Snow = CN()
__C.RENDERER.GL.bodies_info.Snow.draw_density = False
__C.RENDERER.GL.bodies_info.Snow.draw_diffuse = False
__C.RENDERER.GL.bodies_info.Snow.draw_ellipsoids = True
__C.RENDERER.GL.bodies_info.Snow.draw_points = False
__C.RENDERER.GL.bodies_info.Snow.needs_smoothing = True
__C.RENDERER.GL.bodies_info.Snow.particle_color = [0.933, 0.933, 0.941, 0.95]
__C.RENDERER.GL.bodies_info.Snow.particle_radius = 0.001
__C.RENDERER.GL.bodies_info.Snow.anisotropy_scale = 1.0
__C.RENDERER.GL.bodies_info.Sand = CN()
__C.RENDERER.GL.bodies_info.Sand.draw_density = False
__C.RENDERER.GL.bodies_info.Sand.draw_diffuse = False
__C.RENDERER.GL.bodies_info.Sand.draw_ellipsoids = False
__C.RENDERER.GL.bodies_info.Sand.draw_points = True
__C.RENDERER.GL.bodies_info.Sand.needs_smoothing = False
__C.RENDERER.GL.bodies_info.Sand.particle_color = [0.94901961, 0.82352941, 0.6627451, 1.0]
__C.RENDERER.GL.bodies_info.Sand.particle_radius = 0.003
__C.RENDERER.GL.bodies_info.Sand.anisotropy_scale = 1.0
__C.RENDERER.GL.bodies_info.Stationary = CN()
__C.RENDERER.GL.bodies_info.SimpleMuscle = CN()
__C.RENDERER.GL.bodies_info.DiffAquaMuscle = CN()
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.draw_density = False
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.draw_diffuse = False
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.draw_ellipsoids = False # True
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.draw_points = True # False
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.needs_smoothing = False
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.particle_color = [0.7, 0.0, 0.0, 0.95]
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.particle_radius = 0.006
__C.RENDERER.GL.bodies_info.DiffAquaMuscle.anisotropy_scale = 1.0
__C.RENDERER.GL.bodies_info.Mud = CN()
__C.RENDERER.GL.bodies_info.Mud.draw_density = False
__C.RENDERER.GL.bodies_info.Mud.draw_diffuse = False
__C.RENDERER.GL.bodies_info.Mud.draw_ellipsoids = True
__C.RENDERER.GL.bodies_info.Mud.draw_points = False
__C.RENDERER.GL.bodies_info.Mud.needs_smoothing = True
__C.RENDERER.GL.bodies_info.Mud.particle_color = [0.388, 0.349, 0.321, 0.8]
__C.RENDERER.GL.bodies_info.Mud.particle_radius = 0.001
__C.RENDERER.GL.bodies_info.Mud.anisotropy_scale = 1.0
__C.RENDERER.GL.bodies_info.Plasticine = CN()
__C.RENDERER.GL.bodies_info.Plasticine.draw_density = False
__C.RENDERER.GL.bodies_info.Plasticine.draw_diffuse = False
__C.RENDERER.GL.bodies_info.Plasticine.draw_ellipsoids = True
__C.RENDERER.GL.bodies_info.Plasticine.draw_points = False
__C.RENDERER.GL.bodies_info.Plasticine.needs_smoothing = True
__C.RENDERER.GL.bodies_info.Plasticine.particle_color = [0.278, 0.153, 0.102, 1.0] # [0.353, 0.263, 0.196, 1.0]
__C.RENDERER.GL.bodies_info.Plasticine.particle_radius = 0.003
__C.RENDERER.GL.bodies_info.Plasticine.anisotropy_scale = 0.5


def get_cfg_defaults():
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()

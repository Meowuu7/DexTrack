from typing import Optional, Dict
from yacs.config import CfgNode as CN
import numpy as np
import taichi as ti
from gym import Env, spaces

from ..engine.taichi_sim import TaichiSim
from ..configs.config import get_cfg_defaults
from ..configs.item_configs.design_space import get_cfg_defaults as get_design_space_cfg_defaults
from ..tools.general_utils import merge_cfg, set_cfg_attr
from ..engine import I_DTYPE, F_DTYPE


@ti.data_oriented
class BaseEnv(Env):
    def __init__(self, cfg_file: CN, out_dir: str, device: Optional[str] = 'numpy',
                 cfg_kwargs: Optional[Dict] = dict(), ti_device_memory_fraction: Optional[float] = None):
        if ti_device_memory_fraction is not None:
            ti.init(arch=ti.cuda, device_memory_fraction=ti_device_memory_fraction)

        # Get environment configuration
        default_cfg = get_cfg_defaults()
        cfg = merge_cfg(default_cfg, cfg_file)
        for k, v in cfg_kwargs.items():
            set_cfg_attr(cfg, k, v)

        # Instantiate Taichi simulation
        self.sim = TaichiSim(cfg, device)
        
        f_dtype_np = np.float32 if self.sim.solver.f_dtype == ti.f32 else np.float64
        
        # Instantiate design representation
        pbr_successor_type = None
        if cfg.ENVIRONMENT.design_space == 'pbr':
            pbr_successor_type = cfg.ENVIRONMENT.design_space_config.base_shape.type.split('.')[-1]
        default_design_space_cfg = get_design_space_cfg_defaults(cfg.ENVIRONMENT.design_space, pbr_successor_type)
        cfg.ENVIRONMENT.design_space_config = merge_cfg(
            default_design_space_cfg, cfg.ENVIRONMENT.design_space_config)

        if cfg.ENVIRONMENT.design_space == 'vbr':
            from .design_space.voxel_based_representation import VoxelBasedRepresentation
            design_space_cls = VoxelBasedRepresentation
        elif cfg.ENVIRONMENT.design_space == 'pbr':
            from .design_space.particle_based_representation import ParticleBasedRepresentation
            design_space_cls = ParticleBasedRepresentation
        elif cfg.ENVIRONMENT.design_space == 'dummy':
            from .design_space.dummy_representation import DummyRepresentation
            design_space_cls = DummyRepresentation
        else:
            raise ValueError(f'Unrecognized design space API {cfg.design_space}')
        self.design_space = design_space_cls(self.sim, cfg.ENVIRONMENT.design_space_config)
        
        # Instantiate renderer
        if cfg.ENVIRONMENT.use_renderer:
            if cfg.RENDERER.type == 'gui':
                from ..engine.renderer.gui_renderer import GUIRenderer
                self.renderer = GUIRenderer(self.sim, out_dir, cfg.RENDERER.GUI)
            elif cfg.RENDERER.type == 'ggui':
                from ..engine.renderer.ggui_renderer import GGUIRenderer
                self.renderer = GGUIRenderer(self.sim, out_dir, cfg.RENDERER.GGUI)
            elif cfg.RENDERER.type == 'gl':
                from ..engine.renderer.gl_renderer import GLRenderer
                self.renderer = GLRenderer(self.sim, out_dir, cfg.RENDERER.GL)
            else:
                raise ValueError(f'Unrecognized renderer type {cfg.RENDERER.type}')

            self.renderer.actuation_strength = cfg.ENVIRONMENT.actuation_strength
            self.renderer.is_robot = self.design_space.is_robot

        # Environment attributes
        self.frame_dt = self.sim.frame_dt
        self.max_steps = self.sim.max_steps
        self.substep_dt = self.sim.substep_dt
        self.cfg = cfg
        self.out_dir = out_dir

        # Instantiate objective
        objective = cfg.ENVIRONMENT.get('objective', None)
        objective_config = cfg.ENVIRONMENT.get('objective_config', dict())
        if objective == 'trajectory_following':
            from .objectives.traj_following import TrajectoryFollowing
            objective_cls = TrajectoryFollowing
        elif objective == 'move_in_circles':
            from .objectives.move_in_circles import MoveInCircles
            objective_cls = MoveInCircles
        elif objective == 'move_forward':
            from .objectives.move_forward import MoveForward
            objective_cls = MoveForward
        elif objective in [None, 'None']:
            from .objectives.dummy import Dummy
            objective_cls = Dummy
        else:
            raise ValueError(f'Unrecognized objective {objective}')
        self.objective = objective_cls(self, objective_config)

        obs_spaces = dict()
        if self.objective.obs_shape is not None:
            obs_spaces['objective'] = spaces.Box(low=-np.inf, high=np.inf, shape=self.objective.obs_shape, dtype=f_dtype_np)
        
        # Environment spec.
        if not isinstance(self.cfg.ENVIRONMENT.observation_space, list):
            self.cfg.ENVIRONMENT.observation_space = [self.cfg.ENVIRONMENT.observation_space]
        
        if 'time' in self.cfg.ENVIRONMENT.observation_space: # time
            obs_spaces['time'] = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=f_dtype_np)
        if 'com' in self.cfg.ENVIRONMENT.observation_space: # center of mass (position + velocity)
            obs_spaces['com'] = spaces.Box(low=-np.inf, high=np.inf, shape=(self.sim.solver.dim * 2,), dtype=f_dtype_np)

        for k in self.cfg.ENVIRONMENT.observation_space:
            if k not in obs_spaces.keys():
                raise ValueError(f'Missing definition of observation space {k}')

        self.observation_space = spaces.Dict(obs_spaces)

        self.action_space = spaces.Box(low=-cfg.ENVIRONMENT.actuation_strength,
                                       high=cfg.ENVIRONMENT.actuation_strength,
                                       shape=(self.design_space.n_actuators,))

    def initialize(self):
        if getattr(self.design_space, 'needs_to_be_built_at_env_init', False):
            self.design_space.build()
        self.sim.initialize()
        if self.cfg.ENVIRONMENT.use_renderer:
            self.renderer.initialize()
        if hasattr(self, 'renderer') and hasattr(self.design_space, 'update_renderer'): # HACK handle particle_ids_range in voxel-based representation
            self.design_space.update_renderer(self.renderer)
        self.design_space.initialize()

        print(f'N particles: {self.sim.solver.n_particles[None]}')
        print(f'M substeps: {self.sim.solver.max_substeps}')
        print(f'M substeps (local): {self.sim.solver.max_substeps_local}')

    def reset(self, design=None):
        self.frame_idx = 0
        self.sim.reset()
        self.design_space.reset()
        if self.cfg.ENVIRONMENT.use_renderer:
            self.renderer.reset()
        if design is not None:
            self.design_space.set_design(design)
        self.design_space.reset_orientation_data()
        obs = None

        return obs

    def step(self, action):
        self.sim.step(action, self.frame_dt)
        obs = self.get_obs()
        reward = self.get_reward()
        done = self.exceed_max_steps() or self.objective.get_done()
        info = dict()
        self.frame_idx += 1
        
        return obs, reward, done, info

    def render(self):
        assert self.has_renderer
        self.renderer.render()

    def close(self):
        if hasattr(self, 'renderer'):
            self.renderer.close()

    def get_obs(self):
        obs = dict()
        s = self.sim.solver.current_s
        if 'time' in self.observation_space.spaces.keys():
            obs['time'] = self.sim.device.tensor([self.sim.solver.sim_t])
        if 'com' in self.observation_space.spaces.keys():
            robot_x = self.design_space.get_x(s)
            if s == 0:
                robot_v_avg = self.design_space.get_v_avg(s, mode=0) # TODO HACK: orientation data is set after reset
            else:
                robot_v_avg = self.design_space.get_v_avg(s, mode=1) # NOTE: mode 1 uses projected v
            obs['com'] = self.sim.device.cat([robot_x.mean(0), robot_v_avg])
        if 'objective' in self.observation_space.spaces.keys():
            obs['objective'] = self.objective.get_obs(s)
        
        return obs

    def get_reward(self):
        s = self.sim.solver.current_s
        rew = self.objective.get_reward(s)
        return rew

    def exceed_max_steps(self):
        return not (self.frame_idx < self.max_steps)

    @property
    def has_renderer(self):
        return self.cfg.ENVIRONMENT.use_renderer

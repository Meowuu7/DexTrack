from fcntl import F_DUPFD
from typing import Optional, Dict
from yacs.config import CfgNode as CN
import os
import numpy as np
import torch
import taichi as ti

from . import ENV_CONFIGS_DIR
from .base_env import BaseEnv
from ..engine import I_DTYPE, F_DTYPE
from ..engine.materials import Material
from ..configs.item_configs.particle_group_info import get_cfg_defaults as get_particle_group_info_cfg_defaults


@ti.data_oriented
class AquaticEnvironment(BaseEnv):
    def __init__(self, 
                 cfg_file: str,
                 out_dir: str, 
                 device: Optional[str] = 'numpy',
                 cfg_kwargs: Optional[Dict] = dict(),
                 ti_device_memory_fraction: Optional[float] = None,
                 initialize: Optional[bool] = False):
        cfg_file = os.path.join(ENV_CONFIGS_DIR, cfg_file)
        super().__init__(cfg_file, out_dir, device, cfg_kwargs, ti_device_memory_fraction)
        if initialize:
            self.initialize()

    def initialize(self):
        # Instantiate water particles
        matter_item_cfg = CN()
        matter_item_cfg.type = 'Primitive.Box'
        matter_item_cfg.material = Material.Water.name
        matter_item_cfg.particle_id = self.cfg.ENVIRONMENT.CUSTOM.matter_id
        matter_item_cfg.semantic_id = self.cfg.ENVIRONMENT.CUSTOM.matter_semantic_id
        matter_item_cfg.initial_position = [0.5, self.cfg.ENVIRONMENT.CUSTOM.matter_depth/2., 0.5]
        matter_item_cfg.initial_velocity = [0., 0., 0.]
        matter_item_cfg.particle_info = get_particle_group_info_cfg_defaults()
        matter_item_cfg.particle_info.p_rho_0 = self.cfg.ENVIRONMENT.CUSTOM.matter_density
        matter_item_cfg.sample_density = self.cfg.ENVIRONMENT.CUSTOM.matter_sample_density
        matter_item_cfg.size = [
            (self.sim.solver.n_grid - 2 * self.sim.solver.padding[0]) / self.sim.solver.n_grid,
            self.cfg.ENVIRONMENT.CUSTOM.matter_depth,
            (self.sim.solver.n_grid - 2 * self.sim.solver.padding[2]) / self.sim.solver.n_grid,
        ]
        self.matter_p_start = self.sim.solver.n_particles[None]
        self.sim.solver.add(matter_item_cfg.type, matter_item_cfg)
        self.matter_p_end = self.sim.solver.n_particles[None]

        # Default initialization
        super().initialize()

        particle_ids = self.sim.device.clone(self.sim.apply('get', 'particle_ids'))
        self.matter_mask = (particle_ids == self.cfg.ENVIRONMENT.CUSTOM.matter_id).float()[:,None]

        assert Material.Water in self.sim.solver.active_materials['members'], \
            'Material `Water` must be activated in aquatic environment'

        # Get polysurface of terrain
        terrain_info = [v for v in self.sim.solver.static_component_info if v['type'] == 'Static.Terrain']
        assert len(terrain_info) == 1, f'Only allow 1 terrain. There is {len(terrain_info)} terrain now'
        self.terrain_info = terrain_info[0]

        # Reset water based on terrain
        reset_s = 0
        self.reset_water(reset_s)
        new_x = self.sim.device.clone(self.sim.apply('get', 'x', s=reset_s))
        self.sim.initial_state['x'] = self.sim.initial_state['x'] * (1 - self.matter_mask) + new_x * self.matter_mask # only reset matter x

    def reset(self, design=None):
        reset_s = 0

        # Default reset
        obs = super().reset(design)

        # Randomize terrain
        if self.cfg.ENVIRONMENT.CUSTOM.randomize_terrain:
            self.terrain_info['reset_fn']()
        
        # Place robot
        self.sim.solver.grid_m.fill(0.)
        self.set_robot_occupancy(reset_s)

        # Reset water
        if self.cfg.ENVIRONMENT.CUSTOM.randomize_terrain or design is not None: # NOTE: only when terrain or robot design is reset
            self.reset_water(reset_s)
            new_x = self.sim.device.clone(self.sim.apply('get', 'x', s=reset_s))
            self.sim.initial_state['x'] = self.sim.initial_state['x'] * (1 - self.matter_mask) + new_x * self.matter_mask # only reset matter x

        self.sim.solver.grid_m.fill(0.) # NOTE: to restore effect from set_robot_occupancy

        # Reset objective
        self.objective.reset()

        # Get observation
        obs = self.get_obs()

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)

        return obs, reward, done, info

    def get_obs(self):
        return super().get_obs()

    @ti.kernel
    def set_robot_occupancy(self, s: I_DTYPE):
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            if self.design_space.is_robot(id):
                base = ti.floor(self.sim.solver.x[s, p] * self.sim.solver.inv_dx - 0.5).cast(I_DTYPE)
                for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.sim.solver.dim)))):
                    self.sim.solver.grid_m[s, base + offset] += 1

    @ti.kernel
    def reset_water(self, s: I_DTYPE):
        for p in range(self.matter_p_start, self.matter_p_end):
            surface_I_x = ti.cast(ti.random() * (self.terrain_info['polysurface_points'].shape[0] - 1), I_DTYPE)
            surface_I_z = ti.cast(ti.random() * (self.terrain_info['polysurface_points'].shape[1] - 1), I_DTYPE)
            surface_point = self.terrain_info['polysurface_points'][surface_I_x, surface_I_z]
            surface_point_ip1 = self.terrain_info['polysurface_points'][surface_I_x + 1, surface_I_z]
            surface_point_jp1 = self.terrain_info['polysurface_points'][surface_I_x, surface_I_z + 1]
            di = (surface_point_ip1 - surface_point) * ti.random()
            dj = (surface_point_jp1 - surface_point) * ti.random()
            min_height = ti.max(surface_point[1], surface_point_ip1[1], surface_point_jp1[1])

            new_x = surface_point + di + dj
            new_x[1] = ti.max(ti.random() * self.cfg.ENVIRONMENT.CUSTOM.matter_depth, min_height)

            base = ti.floor(new_x * self.sim.solver.inv_dx - 0.5).cast(I_DTYPE)
            if self.sim.solver.grid_m[s, base] > 0: # NOTE: avoid placing particle at robot body
                self.sim.solver.x[s, p] = ti.Vector.zero(F_DTYPE, self.sim.solver.dim)
            else:
                self.sim.solver.x[s, p] = new_x

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
class ManipulationEnvironment(BaseEnv):
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
        # Get data of matter (before super initialize since we are instantiating matters)
        if self.cfg.ENVIRONMENT.CUSTOM.has_matter_on_ground:
            # get material models
            self.matter_materials = []
            for matter_material in self.cfg.ENVIRONMENT.CUSTOM.matter_materials:
                matter_material = getattr(Material, matter_material)
                self.matter_materials.append(matter_material)

            # initialize particles for matter
            self.matter_id = self.cfg.ENVIRONMENT.CUSTOM.matter_id
            matter_item_cfg = CN()
            matter_item_cfg.type = 'Primitive.Box'
            matter_item_cfg.material = self.matter_materials[0].name
            matter_item_cfg.particle_id = self.matter_id
            matter_item_cfg.semantic_id = self.cfg.ENVIRONMENT.CUSTOM.matter_semantic_id
            matter_item_cfg.initial_position = [0.5, 0.5, 0.5] # initialize at the center of a 1-cube
            matter_item_cfg.initial_velocity = self.cfg.ENVIRONMENT.CUSTOM.get('matter_initial_velocity', [0., 0., 0.])
            matter_item_cfg.sample_density = self.cfg.ENVIRONMENT.CUSTOM.matter_sample_density
            matter_item_cfg.particle_info = get_particle_group_info_cfg_defaults()
            matter_item_cfg.particle_info.E_0 = self.cfg.ENVIRONMENT.CUSTOM.matter_youngs_modulus
            self.padding = self.cfg.ENVIRONMENT.CUSTOM.get('matter_padding', self.sim.solver.padding)
            matter_item_cfg.size = [
                (self.sim.solver.n_grid - 2*self.padding[0]) / self.sim.solver.n_grid,
                self.cfg.ENVIRONMENT.CUSTOM.matter_thickness,
                (self.sim.solver.n_grid - 2*self.padding[2]) / self.sim.solver.n_grid,
            ]
            self.matter_p_start = self.sim.solver.n_particles[None]
            self.sim.solver.add(matter_item_cfg.type, matter_item_cfg)
            self.matter_p_end = self.sim.solver.n_particles[None]

        super().initialize()

        if self.cfg.ENVIRONMENT.CUSTOM.has_matter_on_ground:
            for matter_material in self.matter_materials:
                assert matter_material in self.sim.solver.active_materials['members'], \
                        f'Matter material {matter_material.name} is not active'

        # Get polysurface of terrain
        terrain_info = [v for v in self.sim.solver.static_component_info if v['type'] == 'Static.Terrain']
        assert len(terrain_info) == 1, f'Only allow 1 terrain. There is {len(terrain_info)} terrain now'
        self.terrain_info = terrain_info[0]

    def reset(self, design=None):
        reset_s = 0

        # Default reset
        obs = super().reset(design)

        # Randomize terrain
        if self.cfg.ENVIRONMENT.CUSTOM.randomize_terrain:
            self.terrain_info['reset_fn']()

        # Randomly place matter on the ground
        if self.cfg.ENVIRONMENT.CUSTOM.has_matter_on_ground:
            matter_material_id = np.random.choice(self.matter_materials).value
            self.reset_matter(reset_s, matter_material_id)

        # Put robot on ground
        max_surface_height = self.get_max_surface_height_under_robot(reset_s)
        if self.cfg.ENVIRONMENT.CUSTOM.has_matter_on_ground:
            max_surface_height += self.cfg.ENVIRONMENT.CUSTOM.matter_thickness
        robot_x = self.design_space.get_x(reset_s)
        offset = self.sim.device.tensor([0., max_surface_height + self.design_space.initial_position[1] - robot_x.min(0)[0][1], 0.])
        self.design_space.transform_x(reset_s, offset)

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
    def get_max_surface_height_under_robot(self, s: I_DTYPE) -> F_DTYPE:
        # Get maximal surface height under robot
        max_surface_height = ti.cast(0., F_DTYPE)
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            if self.design_space.is_robot(id) and self.sim.solver.p_rho[p] > 0:
                res_mul = self.terrain_info['resolution'] / self.sim.solver.n_grid # conversion from mpm resolution to terrain surface resolution
                I = ti.floor((self.sim.solver.x[s, p] * self.sim.solver.inv_dx - 0.5) * res_mul).cast(I_DTYPE)
                i = ti.min(ti.max(I[0], 0), self.terrain_info['resolution'] - 1)
                j = ti.min(ti.max(I[2], 0), self.terrain_info['resolution'] - 1)
                surface_point = self.terrain_info['polysurface_points'][i, j]
                ti.atomic_max(max_surface_height, surface_point[1])

        return max_surface_height

    @ti.kernel
    def reset_matter(self, s: I_DTYPE, matter_material_id: I_DTYPE):
        for p in range(self.sim.solver.n_particles[None]):
            id = self.sim.solver.particle_ids[p]
            if id == self.matter_id:
                # set material
                self.sim.solver.material[p] = matter_material_id

                # set position
                padding_frac_x = self.sim.solver.padding[0] / self.sim.solver.n_grid
                padding_frac_z = self.sim.solver.padding[2] / self.sim.solver.n_grid
                matter_padding_frac_x = self.padding[0] / self.sim.solver.n_grid
                matter_padding_frac_z = self.padding[2] / self.sim.solver.n_grid
                
                base = self.sim.solver.x[s, p]
                base[0] = (base[0] - padding_frac_x) / (1 - 2 * padding_frac_x)
                base[2] = (base[2] - padding_frac_z) / (1 - 2 * padding_frac_z)

                terrain_grid_x = ti.cast(base[0] * self.terrain_info['resolution'], I_DTYPE)
                terrain_grid_z = ti.cast(base[2] * self.terrain_info['resolution'], I_DTYPE)

                surface_point = self.terrain_info['polysurface_points'][terrain_grid_x, terrain_grid_z]
                surface_normal = self.terrain_info['polysurface_normals'][terrain_grid_x, terrain_grid_z]

                dk = surface_normal * self.cfg.ENVIRONMENT.CUSTOM.matter_thickness
                rand_dk_mul = (p % 10) / 10.
                self.sim.solver.x[s, p] = surface_point + dk * rand_dk_mul

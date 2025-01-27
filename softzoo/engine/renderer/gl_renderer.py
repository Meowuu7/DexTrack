import os
import pickle
from yacs.config import CfgNode as CN
from itertools import groupby
import numpy as np
from skimage.io import imread
from skimage.transform import resize
import cv2
import taichi as ti

from .gl_renderer_src import flex_renderer
from . import BaseRenderer
from .. import I_DTYPE
from ..taichi_sim import TaichiSim
from ..materials import Material
from ...tools.general_utils import get_video_writer, surface_to_mesh, compute_camera_angle


@ti.data_oriented
class GLRenderer(BaseRenderer):
    MATERIAL_IDS_NEED_SMOOTHING = [Material.Water.value, Material.Mud.value]

    def __init__(self, sim: TaichiSim, out_dir: str, cfg: CN):
        super().__init__(sim, out_dir, cfg)
        assert self.sim.solver.dim == 3, 'GL renderer only support 3D'

        self.res = self.cfg.res
        self.camera_position = np.array(self.cfg.camera_position)
        self.camera_lookat = np.array(self.cfg.camera_lookat)
        self.camera_angle = compute_camera_angle(self.cfg.camera_position, self.cfg.camera_lookat)
        self.lights = []
        self.draw_plane = self.cfg.draw_plane
        self.camera_fov = self.cfg.camera_fov / 180.0 * np.pi
        self.light_position = np.array(self.cfg.light_position)
        self.light_lookat = np.array(self.cfg.light_lookat)
        self.light_fov = self.cfg.light_fov
        self.bodies_info = self.cfg.bodies_info

        self._msaa_samples = self.cfg.msaa_samples
        self._anisotropy_scale = self.cfg.anisotropy_scale
        self._smoothing = self.cfg.smoothing
        self._rendering_scale = self.cfg.rendering_scale
        self._fluid_rest_distance = self.cfg.fluid_rest_distance
        self._gl_color_gamma = self.cfg.gl_color_gamma
        self._global_draw_density = False
        self._global_draw_diffuse = False
        self._global_draw_ellipsoids = True
        self._global_draw_points = False

        flex_renderer.init(self.res[0], self.res[1], self._msaa_samples, self.camera_fov)
        
        self.save_to_video = self.cfg.save_to_video
        self.ground_surface_cmap = self.cfg.ground_surface_cmap
        self.ground_surface_brightness_increase = self.cfg.ground_surface_brightness_increase

        self.tile_texture = self.cfg.tile_texture
        self.dump_data = self.cfg.dump_data
        
        self.count = 0
        self.frame_num = 0
        self.meshes = dict()

        bg_img = imread(os.path.join(os.path.dirname(__file__), '../../assets', self.cfg.background))
        bg_img = (resize(bg_img, self.res[::-1]) * 255).astype(np.uint8)
        bg_img = increase_brightness(bg_img, self.cfg.background_brightness_increase)
        self.bg_img = bg_img

    def initialize(self):
        # Get body info
        self.n_particles = self.sim.solver.n_particles[None]

        material = self.sim.apply('get', 'material')
        
        bodies_color = []
        bodies_n_particles = []
        bodies_particle_radius = []
        bodies_particle_offset = []
        bodies_needs_smoothing = []
        bodies_draw_density = []
        bodies_draw_diffuse = []
        bodies_draw_ellipsoids = []
        bodies_draw_points = []
        bodies_anisotropy_scale = []
        for k, g in groupby(enumerate(material), lambda _x: _x[1]):
            g = list(g)
            g_id = int(g[0][1])
            material_cls = [v for v in Material.members() if v.value == g_id][0]
            body_info = getattr(self.bodies_info, material_cls.name)
            bodies_color.append(to_gl_color(body_info.particle_color))
            bodies_n_particles.append(len(g))
            bodies_particle_radius.append(body_info.particle_radius)
            bodies_particle_offset.append(g[0][0])
            bodies_needs_smoothing.append(body_info.needs_smoothing)
            bodies_draw_density.append(body_info.draw_density)
            bodies_draw_diffuse.append(body_info.draw_diffuse)
            bodies_draw_ellipsoids.append(body_info.draw_ellipsoids)
            bodies_draw_points.append(body_info.draw_points)
            bodies_anisotropy_scale.append(body_info.anisotropy_scale)
        n_bodies = len(bodies_color)

        assert np.sum(bodies_n_particles) == self.n_particles

        if False: # HACK: need to draw robot body first otherwise cannot see the robot
            if k == 7:
                for v in ['bodies_n_particles', 'bodies_particle_radius', 'bodies_particle_offset', 'bodies_color',
                          'bodies_needs_smoothing', 'bodies_draw_density', 'bodies_draw_diffuse', 'bodies_draw_ellipsoids',
                          'bodies_draw_points', 'bodies_anisotropy_scale']:
                    exec(f'{v} = {v}[::-1]') # NOTE: this doesn't work; need to change the next part manually

        if hasattr(self, 'objective_render_init'): # HACK: to allow objective class
            inp_data = [bodies_n_particles, bodies_particle_radius, bodies_particle_offset, bodies_color, bodies_needs_smoothing, bodies_draw_density, bodies_draw_diffuse, bodies_draw_ellipsoids, bodies_draw_points, bodies_anisotropy_scale]
            n_bodies = self.objective_render_init(inp_data, n_bodies) # in-place

        bodies_n_particles = np.array(bodies_n_particles)
        bodies_particle_radius = np.array(bodies_particle_radius)
        bodies_particle_offset = np.array(bodies_particle_offset)
        bodies_color = np.array(bodies_color).flatten()
        bodies_needs_smoothing = np.array(bodies_needs_smoothing)
        bodies_draw_density = np.array(bodies_draw_density)
        bodies_draw_diffuse = np.array(bodies_draw_diffuse)
        bodies_draw_ellipsoids = np.array(bodies_draw_ellipsoids)
        bodies_draw_points = np.array(bodies_draw_points)
        bodies_anisotropy_scale = np.array(bodies_anisotropy_scale)

        flex_renderer.create_scene(
            bodies_particle_radius,
            self._anisotropy_scale,
            self._smoothing,
            self._gl_color_gamma,
            self._fluid_rest_distance * self._rendering_scale,
            self.light_position * self._rendering_scale,
            self.light_lookat * self._rendering_scale,
            self.light_fov,
            bodies_n_particles,
            bodies_particle_offset,
            bodies_color,
            bodies_needs_smoothing,
            n_bodies,
            self._global_draw_density,
            self._global_draw_diffuse,
            self._global_draw_ellipsoids,
            self._global_draw_points,
            self.draw_plane,
            bodies_draw_density,
            bodies_draw_diffuse,
            bodies_draw_ellipsoids,
            bodies_draw_points,
            bodies_anisotropy_scale,
        )

        flex_renderer.set_camera_params(self.camera_position * self._rendering_scale, self.camera_angle)

    def reset(self):
        # Reset dump data
        if self.dump_data:
            self.dump_data_dir = os.path.join(self.out_dir, f'Ep_{self.count:04d}')
            os.makedirs(self.dump_data_dir, exist_ok=True)
            self.frame_num = 0

        # Reset video writer
        if hasattr(self, 'video_writer') and self.video_writer.warmStarted:
            self.close_video()
            self.count += 1
        if self.save_to_video:
            self.open_video()

        # Cache terrain plot
        for v in self.sim.solver.static_component_info:
            if v['type'] == 'Static.FlatSurface':
                surface = np.zeros((2, 2, 3))
                normal = v['normals'][0]
                surface[0, 0] = np.array([0., 0., 0.])
                surface[0, 1] = np.array([0., -normal[2], 1.])
                surface[1, 0] = np.array([1., -normal[0], 0.])
                surface[1, 1] = np.array([1., -(normal[0] + normal[2]), 1.])
                surface += np.array(v['points'][0])
                self.add_surface_to_meshes(v['item_id'], surface, normal, cmap=self.ground_surface_cmap)
                assert False
            elif v['type'] == 'Static.Terrain':
                surface = v['points_padded']
                padded_mul = v['padded_mul']
                normal = self.sim.solver.static_component_info[1]['polysurface_normals'].to_numpy()
                normal = np.tile(normal, (padded_mul, padded_mul, 1)).reshape(-1, 3)
                self.add_surface_to_meshes(v['item_id'], surface, normal, cmap=self.ground_surface_cmap, padded_mul=padded_mul)

    def render(self):
        # Particles
        s_local = self.sim.solver.current_s_local
        inp = np.ones((self.n_particles, 4))
        self.get_x(s_local, inp)
        if hasattr(self, 'objective_render'):
            inp = self.objective_render(inp)
        flex_renderer.set_positions(inp.flatten() * self._rendering_scale)

        # Meshes
        flex_renderer.clear_meshes()
        
        for v in self.sim.solver.static_component_info:
            if v['item_id'] in self.meshes.keys():
                mesh = self.meshes[v['item_id']]
                vertices = mesh['vertices']
                vertex_normals = mesh['vertex_normals']
                indices = mesh['indices']
                colors = mesh['colors']
                flex_renderer.add_mesh(vertices.flatten() * self._rendering_scale,
                                       vertices.shape[0],
                                       vertex_normals.flatten(),
                                       vertex_normals.shape[0],
                                       indices,
                                       indices.shape[0],
                                       colors.flatten(),
                                       colors.shape[0])

        # Render
        img, depth = flex_renderer.render()
        img = np.flip(img.reshape([self.res[1], self.res[0], 4]), 0)[:, :, :3]

        # depth = depth.reshape([self.res[1], self.res[0]])
        bg_mask = np.all(img == 0, axis=-1) # assume backgroud color is 0
        img = np.where(np.stack([bg_mask]*3, axis=-1), self.bg_img, img)

        if self.save_to_video:
            self.video_writer.writeFrame(img)
        else:
            cv2.imshow('GL Renderer', img[:,:,::-1])
            cv2.waitKey(1)

        if self.dump_data:
            data = dict()
            data['mesh'] = mesh
            x = self.sim.apply('get', 'x', s=s_local).numpy()
            mat = self.sim.apply('get', 'material').numpy()
            p_rho = self.sim.apply('get', 'p_rho').numpy()
            mask = p_rho > 0
            data['particle_position'] = x[mask]
            data['particle_material'] = mat[mask]
            data['material_mapping'] = {v.value: v.name for v in Material.members()}
            data_fpath = os.path.join(self.dump_data_dir, f'{self.frame_num:03d}.pkl')
            with open(data_fpath, 'wb') as f:
                pickle.dump(data, f)
            self.frame_num += 1
    
    def add_surface_to_meshes(self, item_id, surface, normal, cmap='terrain', padded_mul=1):
        # Convert surface to vertices and indices NOTE: mesh is of fixed shape
        vertices, indices = surface_to_mesh(surface)

        if cmap[:9] == 'textures/':
            texture_fpath = os.path.join(os.path.dirname(__file__), '../../assets/', cmap)
            img = imread(texture_fpath)
            img = increase_brightness(img, self.ground_surface_brightness_increase)
            if img.shape[:2] != surface[...,1].shape:
                if self.tile_texture:
                    img = np.tile(img, (padded_mul, padded_mul, 1))
                else:
                    img = resize(img / 255., (img.shape[0] * padded_mul, img.shape[1] * padded_mul)) * 255.
            assert img.shape[:2] == surface[...,1].shape, \
                f'Texture size {img.shape[:2]} is inconsistent with surface size {surface[...,1].shape}'
            colors = np.ones(surface[...,1].shape + (4,))
            colors[..., :3] = img / 255.
            colors = colors.reshape(-1, 4)
        else:
            cmap = matplotlib.cm.get_cmap(cmap)
            norm = matplotlib.colors.Normalize(vmin=surface[...,1].min(), vmax=surface[...,1].max())
            colors = cmap(norm(surface[...,1])).reshape(-1, 4)

        # Get data container
        if item_id not in self.meshes.keys():
            self.meshes[item_id] = {
                'vertices': None,
                'vertex_normals': None,
                'indices': None,
                'per_vertex_color': None
            }

        self.meshes[item_id]['vertices'] = vertices
        self.meshes[item_id]['vertex_normals'] = normal
        self.meshes[item_id]['indices'] = indices.flatten()
        self.meshes[item_id]['colors'] = colors

    @ti.kernel
    def get_x(self, s_local: I_DTYPE, ext_arr: ti.types.ndarray()):
        for p in range(self.sim.solver.n_particles[None]):
            for d in ti.static(range(self.sim.solver.dim)):
                ext_arr[p, d] = self.sim.solver.x[s_local, p][d]

    def close(self):
        if hasattr(self, 'video_writer') and self.video_writer.warmStarted:
            self.close_video()

    def open_video(self):
        video_path = os.path.join(self.out_dir, f'Ep_{self.count:04d}.mp4')
        self.video_writer = get_video_writer(video_path, self.cfg.fps)

    def close_video(self):
        self.video_writer.close()


def to_gl_color(color):
    return [color[0], color[1], color[2], 1. - color[3]]


def increase_brightness(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    if value >= 0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = -value
        v[v < lim] = 0
        v[v >= lim] -= -value

    final_hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

    return img

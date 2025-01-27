import os
from yacs.config import CfgNode as CN
from collections import namedtuple
import matplotlib
import numpy as np
import taichi as ti
from skimage.io import imread
from skimage.transform import resize
import torch
import open3d as o3d

from . import BaseRenderer
from .. import I_DTYPE
from ..taichi_sim import TaichiSim
from ...tools.general_utils import get_video_writer, surface_to_mesh, cartesian_np, pcd_to_mesh


@ti.data_oriented
class GGUIRenderer(BaseRenderer):
    MeshificationInfo = namedtuple('MeshificationInfo', ['mesh_key', 'update_vertices_kernel', 'update_triangles_kernel'])

    def __init__(self, sim: TaichiSim, out_dir: str, cfg: CN):
        super().__init__(sim, out_dir, cfg)
        assert self.sim.solver.dim == 3, 'GGUI renderer only support 3D'

        # Setup renderer
        self.window = ti.ui.Window(self.cfg.title, self.cfg.res, vsync=True, show_window=not self.cfg.offscreen_rendering)
        self.canvas = self.window.get_canvas()
        self.canvas.set_background_color(self.cfg.background_color)
        self.scene = ti.ui.Scene()
        self.camera = ti.ui.Camera()
        self.set_camera(position=self.cfg.camera_position,
                        lookat=self.cfg.camera_lookat,
                        fov=self.cfg.camera_fov)
        
        # Parameters for rendering
        self.ambient_light = self.cfg.ambient_light
        self.particle_coloring_mode = self.cfg.particle_coloring_mode
        self.ground_surface_cmap = self.cfg.ground_surface_cmap
        self.meshify_particle_ids = self.cfg.meshify_particle_ids
        self.meshification_colors = self.cfg.meshification_colors
        self.save_to_video = self.cfg.save_to_video
        assert len(self.meshify_particle_ids) == len(self.meshification_colors), 'Meshification colors must be of the same length as meshify_particle_ids'

        # Other attributes
        self.f_dtype = ti.f32 # GGUI only take f32
        self.i_dtype = ti.i32
        self.f_dtype_np = np.float32
        self.i_dtype_np = np.int32
        self.f_dtype_torch = torch.float32
        self.i_dtype_torch = torch.int32

        self.count = 0
        self.meshes = dict()

    def initialize(self):
        # Get particle color for rendering
        self.particle_colors = ti.Vector.field(4, dtype=self.f_dtype)
        if self.particle_coloring_mode in ['material', 'actuation']:
            ti.root.dense(ti.i, len(self.cfg.particle_colors)).place(self.particle_colors)
            self.particle_colors.from_numpy(np.array(self.cfg.particle_colors, dtype=self.f_dtype_np))
        elif self.particle_coloring_mode in ['particle_id', 'particle_density', 'E']: # based on particle id
            group_ids = self.sim.solver.particle_group_info.keys()
            max_group_id = max(max(group_ids), len(group_ids))
            if max_group_id > 1:
                ti.root.dynamic(ti.i, max_group_id).place(self.particle_colors)
            else:
                ti.root.dense(ti.i, max_group_id).place(self.particle_colors)
            for k in self.sim.solver.particle_group_info.keys():
                self.particle_colors[k] = list(np.random.uniform(0., 1., (3,))) + [1.]
        else:
            raise ValueError(f'Unrecognized particle coloring mode {self.particle_coloring_mode}')

        # Cache for particles information
        if self.sim.solver.n_particles[None] > 0:
            self.particles_info = dict(
                position=ti.Vector.field(self.sim.solver.dim, dtype=self.f_dtype, shape=(self.sim.solver.n_particles[None])),
                material=ti.field(dtype=self.i_dtype, shape=(self.sim.solver.n_particles[None])),
                color=ti.Vector.field(4, dtype=self.f_dtype, shape=(self.sim.solver.n_particles[None])),
            )

            particle_ids = self.sim.apply('get', 'particle_ids')
            p_range = self.sim.device.tensor(range(self.sim.solver.n_particles[None]), dtype='int32')
            self.particle_id_range = dict() # {ID: (offset, count)}
            for id in particle_ids.unique():
                id = int(id)
                mask = particle_ids == id
                # pid_idcs = torch.where(mask)[0]
                # assert ((pid_idcs[1:] - pid_idcs[:-1]).unique() == 1).all(), f'Non-contiguous particle ids {pid_idcs}'
                count = int(mask.sum())
                offset = int(p_range[mask][0])
                self.particle_id_range[id] = (offset, count)

        # Data required for meshification
        if len(self.meshify_particle_ids) > 0:
            self.meshification_helpers = dict()

        # Load skybox
        if self.cfg.background is not None:
            assert not self.cfg.offscreen_rendering, 'If using background image, offscreen rendering should be turned off'
            bg_fpath = os.path.join(os.path.dirname(__file__), '../../assets/', self.cfg.background)
            img = imread(bg_fpath)
            img = resize(img, self.window.get_window_shape()[::-1])
            img = (img * 255).astype(np.uint8)
            self.background_img = img

    def reset(self):
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
                self.add_surface_to_meshes(v['item_id'], surface, cmap=self.ground_surface_cmap)
            elif v['type'] == 'Static.Terrain':
                surface = v['points_padded']
                padded_mul = v['padded_mul']
                self.add_surface_to_meshes(v['item_id'], surface, cmap=self.ground_surface_cmap, padded_mul=padded_mul)
    
    def render(self):
        # Setup scene
        if not self.save_to_video:
            self.update_camera()
        self.scene.set_camera(self.camera)

        self.scene.ambient_light(self.ambient_light)
        self.scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
        self.scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

        # Snapshot simulation and visualize particles
        if self.sim.solver.n_particles[None] > 0:
            s = self.sim.solver.get_cyclic_s(self.sim.solver.current_s - 1)

            particle_info_updated = False
            if len(self.meshify_particle_ids) > 0:
                points = self.sim.apply('get', 'x', s=s).to(self.f_dtype_torch)

            for id, (index_offset, index_count) in self.particle_id_range.items():
                if id in self.meshify_particle_ids:
                    # Convert point cloud to mesh
                    points_at_id = points[index_offset:index_offset+index_count]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_at_id.cpu().numpy())
                    pcd.estimate_normals()

                    voxel_size = min(points_at_id.max(0)[0] - points_at_id.min(0)[0]).item() / 10
                    mesh = pcd_to_mesh(pcd, voxel_size=voxel_size)
                    
                    verts = np.asarray(mesh.vertices)
                    triangles = np.asarray(mesh.triangles)

                    # Update mesh in taichi field
                    self.update_particle_mesh(id, verts, triangles)
                    mesh_key = self.meshification_helpers[id].mesh_key

                    # Put taichi mesh in scene
                    n_verts = verts.shape[0]
                    n_triangles = triangles.shape[0]
                    self.scene.mesh(**self.meshes[mesh_key],
                                    vertex_count=n_verts,
                                    index_count=n_triangles*3)
                else:
                    if not particle_info_updated:
                        self.update_particle_info(s)
                        particle_info_updated = True
                    self.scene.particles(self.particles_info['position'],
                                         per_vertex_color=self.particles_info['color'],
                                         radius=self.cfg.particle_radius,
                                         index_offset=index_offset,
                                         index_count=index_count)

        # Visualize ground / terrain
        for v in self.sim.solver.static_component_info:
            if v['type'] in ['Static.FlatSurface', 'Static.Terrain']:
                self.scene.mesh(**self.meshes[v['item_id']])
            else:
                continue

        # Visualize rigid body
        for v in self.sim.solver.primitives:
            if not v.is_rigid: # using particle visualization
                continue

            s = self.sim.solver.current_s_local
            if v.type == 'Primitive.Sphere':
                position = ti.Vector.field(self.sim.solver.dim, dtype=self.f_dtype, shape=(1,)) # TODO: slow
                position[0] = v.position[s]
                radius = v.radius
                self.scene.particles(position, radius=radius)
            elif v.type == 'Primitive.Box':
                position = v.position[s] # TODO
            else:
                raise ValueError(f'Cannot visualize {v.cfg.type}')

        # Plot frame and write to video stream
        try:
            self.canvas.scene(self.scene)
            if self.save_to_video:
                img = self.window.get_image_buffer_as_numpy()
                img = (img.transpose(1, 0, 2)[::-1] * 255).astype(np.uint8)
                if self.cfg.background is not None:
                    depth = self.window.get_depth_buffer_as_numpy()
                    depth = depth.transpose(1, 0)[::-1]
                    sky_mask = (depth == 0)[..., None]
                    img[...,:3] = (1 - sky_mask) * img[...,:3] + sky_mask * self.background_img
                self.video_writer.writeFrame(img)
            else:
                self.window.show()
        except:
            pass # HACK

    def update_camera(self, t=None, rotate=False):
        self.camera.track_user_inputs(self.window, movement_speed=0.03, hold_key=ti.ui.RMB)

        speed = 1e-2
        if self.window.is_pressed(ti.ui.UP):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.DOWN):
            camera_dir = np.array(self.camera.curr_lookat - self.camera.curr_position)
            camera_dir[1] = 0
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) - camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) - camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.RIGHT):
            camera_dir = np.cross(np.array(self.camera.curr_lookat - self.camera.curr_position), np.array([0, 1, 0]))
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.LEFT):
            camera_dir = np.cross(np.array(self.camera.curr_lookat - self.camera.curr_position), np.array([0, 1, 0]))
            camera_dir /= np.linalg.norm(camera_dir)
            new_camera_pos = np.array(self.camera.curr_position) - camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) - camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.TAB):
            camera_dir = np.array([0, 1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)
        elif self.window.is_pressed(ti.ui.CAPSLOCK):
            camera_dir = np.array([0, -1, 0])
            new_camera_pos = np.array(self.camera.curr_position) + camera_dir * speed
            new_camera_lookat = np.array(self.camera.curr_lookat) + camera_dir * speed
            self.camera.position(*new_camera_pos)
            self.camera.lookat(*new_camera_lookat)

        # rotate
        if rotate and t is not None:
            speed = 7.5e-4
            xz_radius = np.linalg.norm([self.camera_vec[0], self.camera_vec[2]])
            rad = speed * np.pi * t + self.camera_init_xz_rad
            x = xz_radius * np.sin(rad)
            z = xz_radius * np.cos(rad)
            new_camera_pos = np.array([
                    x + self.camera_lookat[0],
                    self.camera_pos[1],
                    z + self.camera_lookat[2]]) 
            self.camera.position(*new_camera_pos)

    def close(self):
        if hasattr(self, 'video_writer') and self.video_writer.warmStarted:
            self.close_video()
        self.window.destroy()

    def set_camera(self, position=None, lookat=None, fov=None):
        if position is not None:
            self.camera.position(*position)
        if lookat is not None:
            self.camera.lookat(*lookat)
        if fov is not None:
            self.camera.fov(fov)
        self.camera.up(0, 1, 0)

    @ti.kernel
    def update_particle_info(self, s: I_DTYPE):
        for p in range(self.sim.solver.n_particles[None]):
            self.particles_info['position'][p] = ti.cast(self.sim.solver.x[s, p], self.f_dtype)
            self.particles_info['material'][p] = self.sim.solver.material[p]
            if ti.static(self.particle_coloring_mode == 'material'):
                self.particles_info['color'][p] = self.particle_colors[self.sim.solver.material[p]]
            elif ti.static(self.particle_coloring_mode == 'actuation'):
                id = self.sim.solver.particle_ids[p]
                if self.is_robot(id):
                    act = self.sim.solver.actuation[s, p] / ti.cast(self.actuation_strength, self.f_dtype) # HACK: normalization factor
                    act = act / 2. # -0.5~0.5
                    self.particles_info['color'][p] = ti.Vector([0.5 - act, 0.5 - ti.abs(act), 0.5 + act, 0.5],
                                                                self.f_dtype)
                else:
                    self.particles_info['color'][p] = self.particle_colors[self.sim.solver.material[p]]
            elif ti.static(self.particle_coloring_mode == 'particle_id'):
                self.particles_info['color'][p] = self.particle_colors[self.sim.solver.particle_ids[p]]
            elif ti.static(self.particle_coloring_mode == 'E'):
                # use mu as a proxy of E
                mu_0 = ti.cast(self.sim.solver.particle_group_info[0].mu_0, self.f_dtype)
                self.particles_info['color'][p][0] = (ti.cast(self.sim.solver.mu[p], self.f_dtype) / mu_0)
                self.particles_info['color'][p][1] = (ti.cast(self.sim.solver.mu[p], self.f_dtype) / mu_0)
                self.particles_info['color'][p][2] = (ti.cast(self.sim.solver.mu[p], self.f_dtype) / mu_0)
                self.particles_info['color'][p][3] = 1. - (ti.cast(self.sim.solver.mu[p], self.f_dtype) / mu_0)
            else: # particle density
                p_rho_0 = ti.cast(self.sim.solver.particle_group_info[0].p_rho_0, self.f_dtype)
                self.particles_info['color'][p][0] = (ti.cast(self.sim.solver.p_rho[p], self.f_dtype) / p_rho_0)
                self.particles_info['color'][p][1] = (ti.cast(self.sim.solver.p_rho[p], self.f_dtype) / p_rho_0)
                self.particles_info['color'][p][2] = (ti.cast(self.sim.solver.p_rho[p], self.f_dtype) / p_rho_0)
                self.particles_info['color'][p][3] = 1. - (ti.cast(self.sim.solver.p_rho[p], self.f_dtype) / p_rho_0)

    def open_video(self):
        video_path = os.path.join(self.out_dir, f'Ep_{self.count:04d}.mp4')
        self.video_writer = get_video_writer(video_path, self.cfg.fps)

    def close_video(self):
        self.video_writer.close()

    def add_surface_to_meshes(self, item_id, surface, cmap='terrain', padded_mul=1):
        # Convert surface to vertices and indices NOTE: mesh is of fixed shape
        vertices, indices = surface_to_mesh(surface)

        if cmap[:9] == 'textures/':
            texture_fpath = os.path.join(os.path.dirname(__file__), '../../assets/', cmap)
            img = imread(texture_fpath)
            if not img.shape == surface.shape: # if texture not already the correct size (untiled)
                img = np.tile(img, (padded_mul, padded_mul, 1))
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
                'vertices': ti.Vector.field(self.sim.solver.dim, dtype=self.f_dtype, shape=(len(vertices))),
                'indices': ti.field(dtype=self.i_dtype, shape=(np.prod(indices.shape))),
                'per_vertex_color': ti.Vector.field(4, dtype=self.f_dtype, shape=(len(colors)))
            }

        vertices_ti = self.meshes[item_id]['vertices']
        indices_ti = self.meshes[item_id]['indices']
        colors_ti = self.meshes[item_id]['per_vertex_color']

        # To taichi
        vertices_ti.from_numpy(vertices.astype(self.f_dtype_np))
        indices_ti.from_numpy(indices.flatten().astype(self.i_dtype_np))
        colors_ti.from_numpy(colors.astype(self.f_dtype_np)) 

    def update_particle_mesh(self, id, verts, triangles):
        if id not in self.meshification_helpers.keys(): # dynamically instantiating data
            mesh_key = -id # use negative id to differntiate from item id used in static component

            max_n_vertices = 20000 # HACK: hardcoded
            max_n_triangles = 50000
            self.meshes[mesh_key] = { # NOTE: doesn't work with dynamic snode
                'vertices': ti.Vector.field(self.sim.solver.dim, dtype=self.f_dtype, shape=(max_n_vertices)),
                'indices': ti.field(dtype=self.i_dtype, shape=(max_n_triangles * 3)),
                'per_vertex_color': ti.Vector.field(4, dtype=self.f_dtype, shape=(max_n_vertices))
            }

            color = self.meshification_colors[self.meshify_particle_ids.index(id)]

            @ti.kernel # NOTE: need to be called right after kernel instantiation otherwise `mesh_key` will change
            def update_vertices_kernel(n_verts: I_DTYPE, verts: ti.types.ndarray()):
                for i in range(n_verts):
                    for d in ti.static(range(3)):
                        self.meshes[mesh_key]['vertices'][i][d] = verts[i, d]
                    self.meshes[mesh_key]['per_vertex_color'][i] = ti.Vector(color, self.f_dtype)

            @ti.kernel
            def update_triangles_kernel(n_tri: I_DTYPE, triangles: ti.types.ndarray()):
                for i in range(n_tri):
                    for d in ti.static(range(3)):
                        self.meshes[mesh_key]['indices'][i * 3 + d] = triangles[i, d]

            self.meshification_helpers[id] = self.MeshificationInfo(
                mesh_key=mesh_key,
                update_vertices_kernel=update_vertices_kernel,
                update_triangles_kernel=update_triangles_kernel,
            )

        meshification_helper = self.meshification_helpers[id]
        mesh_key = meshification_helper.mesh_key

        n_verts = verts.shape[0]
        n_triangles = triangles.shape[0]
        assert n_verts < self.meshes[mesh_key]['vertices'].shape[0]
        assert n_triangles < self.meshes[mesh_key]['indices'].shape[0]

        meshification_helper.update_vertices_kernel(n_verts, verts.astype(self.f_dtype_np))
        meshification_helper.update_triangles_kernel(n_triangles, triangles.astype(self.i_dtype_np))

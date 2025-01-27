import os
from yacs.config import CfgNode as CN
import numpy as np
import cv2
import taichi as ti

from . import BaseRenderer
from ..taichi_sim import TaichiSim
from ...tools.general_utils import get_video_writer, qrot2d
from ...tools.taichi_utils import TiDeviceInterface


class GUIRenderer(BaseRenderer):
    def __init__(self, sim: TaichiSim, out_dir: str, cfg: CN):
        super().__init__(sim, out_dir, cfg)
        assert self.sim.solver.dim == 2, 'GUI renderer only support 2D'
        
        self.count = 0
        self.particles_info = dict()
        self.gui = ti.GUI(self.cfg.title,
                          res=self.cfg.res,
                          background_color=self.cfg.background_color,
                          show_gui=False)
            
    def reset(self):
        if hasattr(self, 'video_writer') and self.video_writer.warmStarted:
            self.close_video()
            self.count += 1
        self.open_video()
    
    def render(self):
        # Snapshot simulation
        self.update_particle_info()
        colors = np.array(self.cfg.particle_colors, dtype=np.uint32)
        self.gui.circles(self.particles_info['position'],
                         radius=self.cfg.circle_radius,
                         color=colors[self.particles_info['material']])

        # Visualize ground / terrain
        for v in self.sim.solver.static_component_info:
            if v['type'] in ['Static.FlatSurface', 'Static.Terrain']:
                if v['type'] == 'Static.FlatSurface':
                    point, normal = v['points'][0], v['normals'][0]
                    tangent = np.array([normal[1], normal[0]])
                    points = np.array([point, np.array(point) + tangent])
                else:
                    points = np.array(v['points'])

                for i in range(points.shape[0] - 1):
                    x_i = points[i, 0]
                    y_i = points[i, 1]
                    x_ip1 = points[i + 1, 0]
                    y_ip1 = points[i + 1, 1]
                    self.gui.line((x_i, y_i), (x_ip1, y_ip1), radius=2, color=self.cfg.static_component_color)
            elif v['type'] == 'Static.Cave':
                occupancy = cv2.resize(v['grid_mass'].to_numpy(), self.gui.res, interpolation=cv2.INTER_NEAREST)
                img = self.gui.get_image()
                mask = occupancy != 0
                masked_occupancy = occupancy[mask]
                img[mask] = np.stack([(masked_occupancy / occupancy.max())] * 3 +
                    [np.ones_like(masked_occupancy)], axis=-1)
                self.gui.set_image(img)
            else:
                continue

        # Visualize rigid body
        for v in self.sim.solver.primitives:
            if not v.is_rigid: # using particle visualization
                continue

            s = self.sim.solver.get_latest_s()
            if v.type == 'Primitive.Sphere':
                position = v.position[s]
                radius = v.radius * self.cfg.res # in pixel
                self.gui.circle(position, self.cfg.rigid_body_color, radius)
            elif v.type == 'Primitive.Box':
                position = v.position[s]
                q2d = [v.rotation[s][0], v.rotation[s][3]]
                p1 = np.array(qrot2d(q2d, [v.size[0] / 2., v.size[1] / 2.])) + position
                p2 = np.array(qrot2d(q2d, [v.size[0] / 2., -v.size[1] / 2.])) + position
                p3 = np.array(qrot2d(q2d, [-v.size[0] / 2., -v.size[1] / 2.])) + position
                p4 = np.array(qrot2d(q2d, [-v.size[0] / 2., v.size[1] / 2.])) + position
                self.gui.line(p1, p2, radius=2, color=self.cfg.rigid_body_color)
                self.gui.line(p2, p3, radius=2, color=self.cfg.rigid_body_color)
                self.gui.line(p3, p4, radius=2, color=self.cfg.rigid_body_color)
                self.gui.line(p4, p1, radius=2, color=self.cfg.rigid_body_color)
            elif v.type == 'Primitive.Capsule':
                import pdb; pdb.set_trace() # DEBUG
            else:
                raise ValueError(f'Cannot visualize {v.cfg.type}')
        
        # Plot frame and write to video stream
        self.gui.core.update()
        img = self.gui.get_image()
        img = img.transpose(1, 0, 2)[::-1] # otherwise flipped
        img = (img * 255).astype(np.uint8)
        self.video_writer.writeFrame(img)
        self.gui.clear()
        
    def close(self):
        if self.video_writer.warmStarted:
            self.close_video()
        self.gui.close()

    def update_particle_info(self):
        position = self.sim.apply('get', 'x', s=self.sim.solver.get_latest_s())
        material = self.sim.apply('get', 'material')
        if self.sim.device != TiDeviceInterface.Numpy:
            position = position.cpu().numpy()
            material = material.cpu().numpy()

        self.particles_info = {
            'position': position,
            'material': material,
        }

    def open_video(self):
        video_path = os.path.join(self.out_dir, f'Ep_{self.count:04d}.mp4')
        self.video_writer = get_video_writer(video_path, self.cfg.fps)

    def close_video(self):
        self.video_writer.close()

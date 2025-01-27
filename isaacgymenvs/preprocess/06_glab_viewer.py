import sys
# sys.path.append('home/mingxian/project/IsaacGymEnvs/isaacgymenvs')
sys.path.append('.')

import collections
import contextlib
import numpy as np
import pyglet

from pyglet.gl import *

from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree, SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion
from poselib.poselib.core.rotation3d import quat_identity
import torch

BLACK = (0, 0, 0)
WHITE = (1, 1, 1)
RED = (1, 0.2, 0.2)
YELLOW = (1, 1, 0.2)
ORANGE = (1, 0.7, 0.2)
GREEN = (0.2, 0.9, 0.2)
BLUE = (0.2, 0.3, 0.9)
COLORS = (WHITE, RED, YELLOW, GREEN, BLUE, ORANGE)


@contextlib.contextmanager
def gl_context(scale=None, translate=None, rotate=None, mat=None):
    glPushMatrix()
    if mat is not None:
        glMultMatrixf(vec(*mat))
    if translate is not None:
        glTranslatef(*translate)
    if rotate is not None:
        glRotatef(*rotate)
    if scale is not None:
        glScalef(*scale)
    yield
    glPopMatrix()


def vec(*args):
    return (GLfloat * len(args))(*args)


def sphere_vertices(n=2):
    idx = [[0, 1, 2], [0, 5, 1], [0, 2, 4], [0, 4, 5],
           [3, 2, 1], [3, 4, 2], [3, 5, 4], [3, 1, 5]]
    vtx = list(np.array([
        [ 1, 0, 0], [0,  1, 0], [0, 0,  1],
        [-1, 0, 0], [0, -1, 0], [0, 0, -1]], 'f'))
    for _ in range(n):
        idx_ = []
        for ui, vi, wi in idx:
            u, v, w = vtx[ui], vtx[vi], vtx[wi]
            d, e, f = u + v, v + w, w + u
            di = len(vtx)
            vtx.append(d / np.linalg.norm(d))
            ei = len(vtx)
            vtx.append(e / np.linalg.norm(e))
            fi = len(vtx)
            vtx.append(f / np.linalg.norm(f))
            idx_.append([ui, di, fi])
            idx_.append([vi, ei, di])
            idx_.append([wi, fi, ei])
            idx_.append([di, ei, fi])
        idx = idx_
    vtx = np.array(vtx, 'f').flatten()
    return np.array(idx).flatten(), vtx, vtx


def quat_from_euler_xyz(roll, pitch, yaw):
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qx, qy, qz, qw], dim=-1)

def quat_mul(a, b):
    assert a.shape == b.shape
    shape = a.shape
    a = a.reshape(-1, 4)
    b = b.reshape(-1, 4)

    x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
    x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    ww = (z1 + x1) * (x2 + y2)
    yy = (w1 - y1) * (w2 + z2)
    zz = (w1 + y1) * (w2 - z2)
    xx = ww + yy + zz
    qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
    w = qq - ww + (z1 - y1) * (y2 - z2)
    x = qq - xx + (x1 + w1) * (x2 + w2)
    y = qq - yy + (w1 - x1) * (y2 + z2)
    z = qq - zz + (z1 + y1) * (w2 - x2)

    quat = torch.stack([x, y, z, w], dim=-1).view(shape)

    return quat

def quat_conjugate(a):
    shape = a.shape
    a = a.reshape(-1, 4)
    return torch.cat((-a[:, :3], a[:, -1:]), dim=-1).view(shape)


# >>>>>>>> [BEGIN mano] >>>>>>>>
import numpy as np

# npath='/home/mingxian/project/grab/full/grab/s2/apple_lift.npz'
npath='/home/mingxian/project/grab/grab_github_unzip/grab/s1/apple_lift.npz'
whole_body = np.load(npath, allow_pickle=True)
# gg['rhand']
rhand = whole_body['rhand'].flatten()[0]
rhands = rhand['params']
rhands['global_orient']
rhands['transl']
rhands['fullpose']


object_ = whole_body['object'].flatten()[0]
objects = object_['params']
objects['transl']
objects['global_orient']
objects_transl = torch.tensor(objects['transl']).float() +1

table_ = whole_body['table'].flatten()[0]
tables = table_['params']
tables['transl']
tables['global_orient']
tables_transl = torch.tensor(tables['transl']).float() +1



bodys = whole_body['object'].flatten()[0]['params'] 
import torch
node_names = ['right_wrist', 
              'right_thumb1', 'right_thumb2', 'right_thumb3',
              'right_index1', 'right_index2', 'right_index3', 
              'right_middle1', 'right_middle2', 'right_middle3',
              'right_ring1', 'right_ring2', 'right_ring3', 
              'right_pinky1', 'right_pinky2', 'right_pinky3', 
            ]

parent_indices = torch.tensor([-1,  
                       0,  1,  2,  
                       0,  4,  5,  
                       0,  7,  8,  
                       0, 10, 11,
                       0, 13, 14
                       ])

# s2 joint global location
global_translation = torch.tensor([
        [ 0.0943,  0.0076,  0.0056],

        [ 0.0097,  0.0034,  0.0260],
        [-0.0236,  0.0041,  0.0297],
        [-0.0469,  0.0016,  0.0297],

        [ 0.0012,  0.0055,  0.0013],
        [-0.0304,  0.0054, -0.0028],
        [-0.0546,  0.0029, -0.0071],

        [ 0.0260, -0.0033, -0.0391],
        [ 0.0096, -0.0044, -0.0511],
        [-0.0068, -0.0061, -0.0625],

        [ 0.0126,  0.0021, -0.0224],
        [-0.0163,  0.0033, -0.0275],
        [-0.0400,  0.0011, -0.0345],

        [ 0.0725, -0.0068,  0.0337],
        [ 0.0553, -0.0062,  0.0597],
        [ 0.0344, -0.0110,  0.0766]
    ]
)

local_translation = torch.zeros_like(global_translation)
local_translation[1:] = global_translation[1:] - global_translation[parent_indices[1:]]

# fullpose = torch.tensor(rhands['fullpose']).float().reshape(-1, 15, 3)
# fullpose = fullpose[0]
# root_ori = torch.tensor(rhands['global_orient']).float()
# root_ori = root_ori[0]
# root_trans = torch.tensor(rhands['transl']).float() 
# root_trans = root_trans[0]
# q0 = quat_from_euler_xyz(fullpose[:,0], fullpose[:,1], fullpose[:,2])
# q0_r = quat_from_euler_xyz(*root_ori)
# q0_init = torch.zeros_like(q0_r)
# q0_all = torch.cat([q0_init.unsqueeze(0), q0])

fullpose = torch.tensor(rhands['fullpose']).float().reshape(-1, 15, 3)
root_ori = torch.tensor(rhands['global_orient']).float()
root_global_trans = torch.tensor(rhands['transl']).float() 
j_local_rots = quat_from_euler_xyz(fullpose[:,:,0], fullpose[:,:,1], fullpose[:,:,2])
root_rots = quat_from_euler_xyz(root_ori[:,0], root_ori[:,1], root_ori[:,2])
all_local_rots = torch.cat([root_rots.unsqueeze(1), j_local_rots], dim=1)


nt = SkeletonTree(node_names, parent_indices, local_translation=local_translation)
# nt._local_translation
zero_pose = SkeletonState.zero_pose(nt)
z_pos = SkeletonState.from_rotation_and_root_translation(skeleton_tree=nt,
            r=quat_identity([nt.num_joints]),
            t=global_translation[0]+1,
            is_local=True,
        )
# z_pos.root_translation()
grab_pts = zero_pose.global_translation.numpy()
grab_pts2 = z_pos.global_translation.numpy()

motion_poses = SkeletonState.from_rotation_and_root_translation(skeleton_tree=nt,
            r=all_local_rots, t=root_global_trans+1, is_local=True,)


motion_poses.global_translation
# (global_translation.numpy() == grab_pts + global_translation[0].numpy())
# (global_translation == grab_pts2)
# grab_pts *= 1

# >>>>>>>> [END mano] >>>>>>>>




class PoseIterator:
    def __init__(self, poses):
        self.poses = poses
        self.cur = 0
        self.end = len(self.poses)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur < self.end:
            result = self.poses.global_translation[self.cur]
            self.cur += 1
            return result
        else:
            self.poses.global_translation[-1]
            # raise StopIteration

class PoseIterator:
    def __init__(self, poses):
        self.poses = poses
        self.cur = 0
        self.end = len(self.poses)

    def __iter__(self):
        return self

    def __next__(self):
        if self.cur < self.end:
            hand_pts = self.poses.global_translation[self.cur]
            object_pts = objects_transl[self.cur:self.cur+1]
            table_pts = tables_transl[self.cur::self.cur+1] 

            result = torch.cat([hand_pts, object_pts, table_pts], dim=0)
            self.cur += 1
            return result
        else:
            self.poses.global_translation[-1]
            # raise StopIteration
        
class Viewer(pyglet.window.Window):
    def __init__(self, c3d_reader=None, trace=None, paused=False):
        if pyglet.version > '1.3':
            display = pyglet.canvas.get_display()
        else:
            platform = pyglet.window.get_platform()
            display = platform.get_default_display()
        screen = display.get_default_screen()
        try:
            config = screen.get_best_config(Config(
                alpha_size=8,
                depth_size=24,
                double_buffer=True,
                sample_buffers=1,
                samples=4))
        except pyglet.window.NoSuchConfigException:
            config = screen.get_best_config(Config())

        super(Viewer, self).__init__(
            width=800, height=450, resizable=True, vsync=False, config=config)

        self._frames = PoseIterator(motion_poses)
        # self._frame_rate = c3d_reader.header.frame_rate
        self._frame_rate = 120

        self._maxlen = 16
        # self._trails = [[] for _ in range(c3d_reader.point_used)]
        self._trails = [[] for _ in range(self._frames.end)]
        self._reset_trails()

        self.trace = trace
        self.paused = paused

        self.zoom = 5
        self.ty = 0
        self.tz = -1
        self.ry = 30
        self.rz = 30

        #self.fps = pyglet.clock.ClockDisplay()

        self.on_resize(self.width, self.height)

        glEnable(GL_BLEND)
        glEnable(GL_COLOR_MATERIAL)
        glEnable(GL_CULL_FACE)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_NORMALIZE)
        glEnable(GL_POLYGON_SMOOTH)

        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glDepthFunc(GL_LEQUAL)
        glCullFace(GL_BACK)
        glFrontFace(GL_CCW)
        glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST)
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST)
        glShadeModel(GL_SMOOTH)

        glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.2, 0.2, 0.2, 1.0))
        glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(1.0, 1.0, 1.0, 1.0))
        glLightfv(GL_LIGHT0, GL_POSITION, vec(3.0, 3.0, 10.0, 1.0))
        glEnable(GL_LIGHT0)

        BLK = [100, 100, 100] * 6
        WHT = [150, 150, 150] * 6
        N = 10
        z = 0
        vtx = []
        for i in range(N, -N, -1):
            for j in range(-N, N, 1):
                vtx.extend((j,   i, z, j, i-1, z, j+1, i,   z,
                            j+1, i, z, j, i-1, z, j+1, i-1, z))

        self.floor = pyglet.graphics.vertex_list(
            len(vtx) // 3,
            ('v3f/static', vtx),
            ('c3B/static', ((BLK + WHT) * N + (WHT + BLK) * N) * N),
            ('n3i/static', [0, 0, 1] * (len(vtx) // 3)))

        idx, vtx, nrm = sphere_vertices()
        self.sphere = pyglet.graphics.vertex_list_indexed(
            len(vtx) // 3, idx, ('v3f/static', vtx), ('n3f/static', nrm))

    def on_mouse_scroll(self, x, y, dx, dy):
        if dy == 0: return
        self.zoom *= 1.1 ** (-1 if dy < 0 else 1)

    def on_mouse_drag(self, x, y, dx, dy, buttons, modifiers):
        if buttons == pyglet.window.mouse.LEFT:
            # pan
            self.ty += 0.03 * dx
            self.tz += 0.03 * dy
        else:
            # roll
            self.ry += 0.2 * -dy
            self.rz += 0.2 * dx
        #print('z', self.zoom, 't', self.ty, self.tz, 'r', self.ry, self.rz)

    def on_resize(self, width, height):
        glViewport(0, 0, width, height)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glu.gluPerspective(45, float(width) / height, 1, 100)

    def on_key_press(self, key, modifiers):
        k = pyglet.window.key
        if key == k.ESCAPE:
            pyglet.app.exit()
        elif key == k.SPACE:
            self.paused = False if self.paused else True
        elif key == k.PLUS or key == k.EQUAL:
            self._maxlen *= 2
            self._reset_trails()
        elif key == k.UNDERSCORE or key == k.MINUS:
            self._maxlen = max(1, self._maxlen / 2)
            self._reset_trails()
        elif key == k.RIGHT:
            skip = int(self._frame_rate)
            if modifiers:
                skip *= 10
            [self._next_frame() for _ in range(skip)]

    def on_draw(self):
        self.clear()

        # http://njoubert.com/teaching/cs184_fa08/section/sec09_camera.pdf
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(1, 0, 0, 0, 0, 0, 0, 0, 1)
        glTranslatef(-self.zoom, 0, 0)
        glTranslatef(0, self.ty, self.tz)
        glRotatef(self.ry, 0, 1, 0)
        glRotatef(self.rz, 0, 0, 1)

        self.floor.draw(GL_TRIANGLES)

        hand_idx = motion_poses.global_translation.shape[1]
        object_idx = hand_idx + 1 
        for t, trail in enumerate(self._trails):
            if t < hand_idx:
                glColor4f(*(COLORS[0 % len(COLORS)] + (0.7, )))
            elif t == hand_idx:
                glColor4f(*(COLORS[1 % len(COLORS)] + (0.7, )))
            elif t == hand_idx+1:
                glColor4f(*(COLORS[2 % len(COLORS)] + (0.7, )))
            else:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            point = None
            glBegin(GL_LINES)
            for point in trail:
                glVertex3f(*point)
            glEnd()
            with gl_context(translate=point, scale=(0.02, 0.02, 0.02)):
                self.sphere.draw(GL_TRIANGLES)

    def _reset_trails(self):
        self._trails = [collections.deque(t, self._maxlen) for t in self._trails]

    def _next_frame(self):
        try:
            return next(self._frames)
        except StopIteration:
            pyglet.app.exit()

    def update(self, dt):
        if self.paused:
            return
        # for trail, point in zip(self._trails, self._next_frame()[1]):
        for trail, point in zip(self._trails, self._next_frame()):
            # if point[3] > -1 or not len(trail):
            #     trail.append(point[:3] / 1000.)
            # else:
            #     trail.append(trail[-1])
            # trail.append(point[:3] / 1000.)
            trail.append(point[:3])

    def mainloop(self):
        pyglet.clock.schedule_interval(self.update, 0.05 / self._frame_rate)
        pyglet.app.run()


if __name__ == '__main__':
    a = Viewer()
    a.mainloop()
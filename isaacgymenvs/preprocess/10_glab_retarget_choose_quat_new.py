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
c_scale = 0.005


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
objects_transl = torch.tensor(objects['transl']).float()

table_ = whole_body['table'].flatten()[0]
tables = table_['params']
tables['transl']
tables['global_orient']
tables_transl = torch.tensor(tables['transl']).float()



bodys = whole_body['object'].flatten()[0]['params'] 
import torch
node_names = ['right_wrist', 
              'right_thumb1', 'right_thumb2', 'right_thumb3',
              'right_index1', 'right_index2', 'right_index3', 
              'right_middle1', 'right_middle2', 'right_middle3',
              'right_ring1', 'right_ring2', 'right_ring3', 
              'right_pinky1', 'right_pinky2', 'right_pinky3', 
            ]



# 从grab joint mapping到shadow hand joints
joint_mapping = {
    # 'robot0:ffproximal'
    'right_index1': 'robot0:ffknuckle',
    'right_index2': 'robot0:ffmiddle', 
    'right_index3': 'robot0:ffdistal',

    # 'robot0:mfproximal'
    'right_middle1': 'robot0:mfknuckle',
    'right_middle2': 'robot0:mfmiddle', 
    'right_middle3': 'robot0:mfdistal',

    # 'robot0:rfproximal'
    'right_ring1': 'robot0:rfknuckle', 
    'right_ring2': 'robot0:rfmiddle',
    'right_ring3': 'robot0:rfdistal',

    # 'robot0:lfmetacarpal', 'robot0:lfproximal'
    'right_pinky1': 'robot0:lfknuckle',
    'right_pinky2': 'robot0:lfmiddle',
    'right_pinky3': 'robot0:lfdistal',

    # 'robot0:thproximal', 'robot0:thmiddle'
    'right_thumb1': 'robot0:thbase',
    'right_thumb2': 'robot0:thhub',
    'right_thumb3': 'robot0:thdistal',

    # 'right_wrist': 'robot0:palm'
    'right_wrist': 'robot0:hand mount'
}

#  



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


grab_tree = SkeletonTree(node_names, parent_indices, local_translation=local_translation)
# nt._local_translation
grab_zero_pose = SkeletonState.zero_pose(grab_tree)
# z_pos = SkeletonState.from_rotation_and_root_translation(skeleton_tree=nt,
#             r=quat_identity([nt.num_joints]),
#             t=global_translation[0],
#             is_local=True,
#         )
# z_pos.root_translation()
grab_pts = grab_zero_pose.global_translation.numpy()
# grab_pts2 = grab_zero_pose.global_translation.numpy()

grab_motion_poses = SkeletonState.from_rotation_and_root_translation(skeleton_tree=grab_tree,
            r=all_local_rots, t=root_global_trans, is_local=True)


grab_motion_poses.global_translation
# (global_translation.numpy() == grab_pts + global_translation[0].numpy())
# (global_translation == grab_pts2)
# grab_pts *= 1

# >>>>>>>> [END mano] >>>>>>>>




# >>>>>>>> [BEGIN 凑 retarget需要的参数] >>>>>>>>
shadow_hand_name_list = ['robot0:hand mount', 'robot0:palm', 'robot0:ffknuckle', 
 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
 'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
 'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 
 'robot0:rfmiddle', 'robot0:rfdistal', 'robot0:lfmetacarpal', 
 'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 
 'robot0:lfdistal', 'robot0:thbase', 'robot0:thproximal', 
 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
sh_name_dict = {shadow_hand_name_list[i]:i for i in range(len(shadow_hand_name_list))}

## >>>>>>>> [获取tar_hand] >>>>>>>
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/open_ai_assets/hand/shadow_hand.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24.xml'

sh_tree = SkeletonTree.from_mjcf(mjcf_path)
sh_zero_pose = SkeletonState.zero_pose(sh_tree)
sh_init_pts = sh_zero_pose.global_translation.numpy()


## <<<<<<<< [获取tar_hand] <<<<<<<<



# <<<<<<<< [END 凑 retarget需要的参数] <<<<<<<<








# >>>>>>>> [create src tpose] >>>>>>>>
source_tpose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=grab_tree,
    r=quat_identity([grab_tree.num_joints]),
    t=torch.zeros(3, dtype=grab_tree.local_translation.dtype),
    is_local=True,
)
# <<<<<<<< [create src tpose] <<<<<<<<



tar_local_rot = quat_identity([sh_tree.num_joints])
tar_local_rot[sh_name_dict['robot0:thbase']] = torch.tensor([0, 0.382499, 0, 0.923956]).float()

sh_rot_pose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=sh_tree,
    r=tar_local_rot,
    t=torch.zeros(3, dtype=sh_tree.local_translation.dtype),
    is_local=True,
)

sh_rot_pts = sh_rot_pose.global_translation.numpy()

## 调节shadow hand的t-pose，使他和grab hand对齐
# scale_to_target_skeleton = 1.18
# tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2-0.05, 0]).float())
# 'robot0:palm', 'robot0:ffknuckle', 
#  'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
#  'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
#  'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 
#  'robot0:rfmiddle', 'robot0:rfdistal', 'robot0:lfmetacarpal', 
#  'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 
#  'robot0:lfdistal', 'robot0:thbase', 'robot0:thproximal', 
#  'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
# tar_local_rot[sh_name_dict['robot0:thbase']] = torch.tensor([0, 0.382499, 0, 0.923956]).float()
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())
# tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())

# target_tpose = SkeletonState.from_rotation_and_root_translation(
#     skeleton_tree=shadow_hand_skeleton_tree,
#     r=tar_local_rot,
#     t=torch.zeros(3, dtype=shadow_hand_skeleton_tree.local_translation.dtype),
#     is_local=True,
# )







# src_hand_t_pose = source_tpose.global_translation.numpy() * scale_to_target_skeleton
# tar_hand_t_pose = target_tpose.global_translation.numpy() 



grab_tpose_pts = grab_pts

pts = [
    grab_tpose_pts,
    sh_init_pts,
    sh_rot_pts,
    ]
pts_vis = np.concatenate(pts, axis=0)
# pts [grab_tpose, sh_init, sh_rot]
# sh_rot是把mjcf中的local rotation加上的结果


# 0 gray
# 1 red 
# 2 yellow 
# 3 green
# 4 blue

# grab_motion_poses.retarget_to(
#     joint_mapping=joint_mapping,
#       source_tpose_local_rotation=quat_identity([1]),
# #     source_tpose_root_translation=-global_translation[0],
#       target_skeleton_tree=t,
# #     target_tpose_local_rotation=
#       target_tpose_root_translation=torch.zeros_like(global_translation[0]).numpy(),
#        rotation_to_target_skeleton=quat_identity([1]),
# #     scale_to_target_skeleton=
# )




 
class Viewer(pyglet.window.Window):
    def __init__(self, trace=None, paused=False):
        self.display_buf = [1, 1, 1, 0, 0, 0]
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

        # self._frame_rate = c3d_reader.header.frame_rate
        self._frame_rate = 120

        self._maxlen = 16
        # self._trails = [[] for _ in range(c3d_reader.point_used)]
        self._trails = [[] for _ in range(150)]
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

        hand_idx = grab_motion_poses.global_translation.shape[1]
        object_idx = hand_idx + 1 
        display_buf = self.display_buf
        for t, trail in enumerate(self._trails):
            if t < 16:
                if display_buf[0] == 1:
                    glColor4f(*(COLORS[0 % len(COLORS)] + (0.7, )))
                else:
                    continue
            elif t < 16+24:
                if display_buf[1] == 1:
                    glColor4f(*(COLORS[1 % len(COLORS)] + (0.7, )))
                else:
                    continue
            elif t < 40+24:
                if display_buf[2] == 1:
                    glColor4f(*(COLORS[2 % len(COLORS)] + (0.7, )))
                else:
                    continue
            # elif t < 56+16:
            #     if display_buf[3] == 1:
            #         glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            #     else:
            #         continue
            # elif t < 72+24:
            #     if display_buf[4] == 1:
            #         glColor4f(*(COLORS[4 % len(COLORS)] + (0.7, )))
            #     else:
            #         continue
            # elif t < 96+24:
            #     if display_buf[5] == 1:
            #         glColor4f(*(COLORS[5 % len(COLORS)] + (0.7, )))
            #     else:
            #         continue


            else:
                continue
                glColor4f(*(COLORS[5 % len(COLORS)] + (0.7, )))
            point = None
            glBegin(GL_LINES)
            for point in trail:
                glVertex3f(*point)
            glEnd()
            with gl_context(translate=point, scale=(c_scale, c_scale, c_scale)):
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
        # for trail, point in zip(self._trails, self._next_frame()):
        for trail, point in zip(self._trails, pts_vis):
            trail.append(point[:3])
            # if point[3] > -1 or not len(trail):
            #     trail.append(point[:3] / 1000.)
            # else:
            #     trail.append(trail[-1])
            # trail.append(point[:3] / 1000.)
            

    def mainloop(self):
        pyglet.clock.schedule_interval(self.update, 0.05 / self._frame_rate)
        pyglet.app.run()



class MyViewer(Viewer):
    def __init__(self, trace=None, paused=False):
        super(MyViewer, self).__init__()
        self.display_buf = [1, 1, 1, 0, 0, 0]

        # self.pts_vis = pts_vis

        self.tar_local_rot = quat_identity([sh_tree.num_joints])

        # 调节shadow hand的t-pose，使他和grab hand对齐
        self.scale_to_target_skeleton = 1.18
        self.tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2-0.16, 0]).float())
        # 'robot0:palm', 'robot0:ffknuckle', 
        #  'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
        #  'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
        #  'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 
        #  'robot0:rfmiddle', 'robot0:rfdistal', 'robot0:lfmetacarpal', 
        #  'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 
        #  'robot0:lfdistal', 'robot0:thbase', 'robot0:thproximal', 
        #  'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
        # tar_local_rot[sh_name_dict['robot0:thbase']] = torch.tensor([0, 0.382499, 0, 0.923956]).float()


        self.tar_local_rot[sh_name_dict['robot0:lfknuckle']] = quat_from_euler_xyz(*torch.tensor([0, -0.349, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:thbase']] = quat_from_euler_xyz(*torch.tensor([0, 0.349, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:ffknuckle']] = quat_from_euler_xyz(*torch.tensor([0, 0.349, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:rfknuckle']] = quat_from_euler_xyz(*torch.tensor([0, -0.349/2, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:lfmetacarpal']] = quat_from_euler_xyz(*torch.tensor([0.15, 0, 0.15]).float())

        

    def update(self, dt):
        if self.paused:
            return
        

        # target_tpose = SkeletonState.from_rotation_and_root_translation(
        #     skeleton_tree=shadow_hand_skeleton_tree,
        #     r=self.tar_local_rot,
        #     t=torch.zeros(3, dtype=shadow_hand_skeleton_tree.local_translation.dtype),
        #     is_local=True,
        # )
        # tar_hand_t_pose = target_tpose.global_translation.numpy()  
        # src_hand_t_pose = source_tpose.global_translation.numpy() * self.scale_to_target_skeleton

        # pts = [
        #     src_first_frame+disp, 
        #     src_hand_zero_pose+disp,
        #     correct_sh_pts+disp,
        #     src_hand_t_pose+disp,
        #     tar_hand_t_pose+disp,
        #     shadow_hand_init_pts+disp,
        # ]
        # pts_vis = np.concatenate(pts, axis=0)

        # pts_vis = self.pts_vis
        for trail, point in zip(self._trails, pts_vis):
            trail.append(point[:3])


if __name__ == '__main__':
    a = MyViewer()
    a.mainloop()
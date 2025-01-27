import os
from c3d.scripts.c3d_viewer import *
import torch


# >>>>>>>>> skeleton tree >>>>>>>>>
import os
import sys
# sys.path.append('home/mingxian/project/IsaacGymEnvs/isaacgymenvs')
sys.path.append('.')
from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonTree, SkeletonState
from poselib.poselib.visualization.common import plot_skeleton_state, plot_skeleton_motion


# >>>>>>>>>>>>>>>>>> hyper-params >>>>>>>>>>>>>>>>>>
sphere_size = 0.005
sphere_size = 0.01
# >>>>>>>>>>>>>>>>>> hyper-params >>>>>>>>>>>>>>>>>>


asset_path = '/home/mingxian/project/IsaacGymEnvs/assets/amp/motions/amp_humanoid_run.npy'
curr_motion = SkeletonMotion.from_file(asset_path)
curr_motion.skeleton_tree
# curr_motion.fps


mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/open_ai_assets/hand/shadow_hand.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24.xml'



# writer = c3d.Writer()
fileroot = '/home/mingxian/project/grab/mocap__s2/s2'
filename_ = 'camera_lift.c3d' 
filelist = ['hammer_lift.c3d' ]

filename = os.path.join(fileroot, filename_)
reader = c3d.Reader(open(filename, 'rb')) 
frames = list(reader.read_frames())
frames_len = len(frames)

'''
frame_id, points, analogs = frames[i]
pts = points[:, :3] # 前3为xyz
reader.point_rate: 120.0
reader.point_labels # 每个点对应的名称
'''

point_labels = list(reader.point_labels)

# point_labels.index('_00177_Body:RIHAND                                              ')
# point_labels.index('_00177_Body:RTHM3                                               ')
# point_labels.index('_00177_Body:RPNK3                                               ')
# point_labels.index('_00177_Body:RTHM1                                               ')
# point_labels.index('_00177_Body:RTHM2                                               ')

prefix = point_labels[0][:12]
rhand_name_list = [
    'RIHAND', 'ROHAND', 
    'RTHM1', 'RTHM2', 'RTHM3', 'RTHM4',
    'RIDX1', 'RIDX2', 'RIDX3',
    'RMID1', 'RMID2', 'RMID3',
    'RRNG1', 'RRNG2', 'RRNG3',
    'RPNK1', 'RPNK2', 'RPNK3',
    ]
table_name_list = ['Table:Table1', 'Table:Table2', 'Table:Table3', 'Table:Table4']
table_key_name_dict = {i: "{:<64}".format(i) for i in table_name_list}
table_key_idx_dict = {names: point_labels.index(table_key_name_dict[names]) for names in table_name_list}
table_key_idx = list(table_key_idx_dict.values())

# "{:<52}".format(map_name_list[0])
# len = 64 in total
# prefix + "{:<52}".format(rhand_name_list[0]) 

# 52
rhand_key_name_dict = {i: prefix + "{:<52}".format(i) for i in rhand_name_list}
rhand_key_idx_dict = {names: point_labels.index(rhand_key_name_dict[names]) for names in rhand_name_list}
rhand_key_idx = list(rhand_key_idx_dict.values())

rth_name_list = ['RTHM1', 'RTHM2', 'RTHM3', 'RTHM4']
rihand_name_list = ['RIHAND', 'ROHAND',]
ridx_name_list = ['RIDX1', 'RIDX2', 'RIDX3',]
rmid_name_list = ['RMID1', 'RMID2', 'RMID3',]
rrng_name_list = ['RRNG1', 'RRNG2', 'RRNG3',]
rpnk_name_list = ['RPNK1', 'RPNK2', 'RPNK3',]

rth_kidx = list(({names: point_labels.index(rhand_key_name_dict[names]) for names in rth_name_list}).values())
rihand_kidx = list(({names: point_labels.index(rhand_key_name_dict[names]) for names in rihand_name_list}).values())
ridx_kidx = list(({names: point_labels.index(rhand_key_name_dict[names]) for names in ridx_name_list}).values())
rmid_kidx = list(({names: point_labels.index(rhand_key_name_dict[names]) for names in rmid_name_list}).values())
rrng_kidx = list(({names: point_labels.index(rhand_key_name_dict[names]) for names in rrng_name_list}).values())
rpnk_kidx = list(({names: point_labels.index(rhand_key_name_dict[names]) for names in rpnk_name_list}).values())


object_key_idx = [i for i in range(99, 106+1)]
ground_key_idx = [i for i in range(111, 115+1)]
gravity_key_idx = [i for i in range(107, 110+1)]






















t = SkeletonTree.from_mjcf(mjcf_path)
# t.node_names
t._local_translation
zero_pose = SkeletonState.zero_pose(t)
sh_pts = zero_pose.global_translation.numpy()
sh_pts *= 1



shadow_hand_name_list = ['robot0:hand mount', 'robot0:palm', 'robot0:ffknuckle', 
 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
 'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
 'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 
 'robot0:rfmiddle', 'robot0:rfdistal', 'robot0:lfmetacarpal', 
 'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 
 'robot0:lfdistal', 'robot0:thbase', 'robot0:thproximal', 
 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
sh_name_dict = {shadow_hand_name_list[i]:i for i in range(len(shadow_hand_name_list))}

sh_th_name = ['robot0:thbase', 'robot0:thproximal', 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
sh_ff_name = ['robot0:ffknuckle', 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', ]
# sh_ff_name = ['robot0:ffknuckle', 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', ]
sh_th_idx = [sh_name_dict[names] for names in sh_th_name]
sh_ff_idx = [sh_name_dict[names] for names in sh_ff_name]
sh_idx = [i for i in range(len(shadow_hand_name_list))]


[sh_name_dict[names] for names in sh_th_name]

local_rot = zero_pose.local_rotation.clone()
# local_rot | x, y, z, w


# from utils.torch_utils import quat_from_euler_xyz, quat_mul

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

# >>>>>>>>>>> 调整 shadow_hand 的方向 >>>>>>>>>>>
# local_rot[0] = torch.tensor([1, 0, 0, 0]).to(local_rot.device)[:]
# x: -90
# local_rot[0] = torch.tensor([0.7071, -0.7071, 0, 0]).to(local_rot.device)[:]
q1 = quat_from_euler_xyz(*torch.tensor([torch.pi/2, 0, 0]).float())
# z: -90
# local_rot[1] = torch.tensor([0.7071, 0, 0, -0.7071]).to(local_rot.device)[:]
q2 = quat_from_euler_xyz(*torch.tensor([0, 0, -torch.pi/2,]).float())
# 把shadow_hand的方向和grab方向对齐
# quat_mul(q2, q1)

# local_rot[0] = torch.tensor([0, 0, -0.5, 0.5]).to(local_rot.device)[:]
# local_rot[0] = quat_mul(q2, q1)
# q1 = quat_from_euler_xyz(*torch.tensor([torch.pi/2, -torch.pi, -torch.pi/2,]).float())
# q2 = quat_from_euler_xyz(*torch.tensor([0, 0, 0.,]).float())
local_rot[0] = quat_mul(q2, q1)
# >>>>>>>>>>> 调整 shadow_hand 的方向 >>>>>>>>>>>

retarget_pose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=t,
    r=local_rot,
    t=zero_pose.root_translation,
    is_local=True
)

sh_pts = retarget_pose.global_translation.numpy()
sh_pts *= 2
# plot_skeleton_state(zero_pose)


# >>>>>>>>> skeleton tree >>>>>>>>>






# >>>>>>>> mx_debug >>>>>>>> 借鉴从fbx构造skeleton motion
# fbx_path = '/home/mingxian/project/IsaacGymEnvs/isaacgymenvs/poselib/data/01_01_cmu.fbx'
# SkeletonMotion.from_fbx(fbx_path)
# debug = 10

# 使用skeletionState来构造skeletonMotion
# r=local_rotation
# torch.Size([2752, 39, 4])
# t=root_translation
# torch.Size([2752, 3])

# skeleton_state = SkeletonState.from_rotation_and_root_translation(
#             skeleton_tree, r=local_rotation, t=root_translation, is_local=True
#         )
# cls.from_skeleton_state(
#             skeleton_state=skeleton_state, fps=fps
#         )
# >>>>>>>> mx_debug >>>>>>>> 借鉴从fbx构造skeleton motion

class MyViewer(Viewer):
    def __init__(self, c3d_reader, trace=None, paused=False):
        super(MyViewer, self).__init__(c3d_reader, trace=trace, paused=paused)
        self.pts = self._next_frame()[1]
        debug = 10
        self.scales = 500.
        global sh_pts
        sh_pts[:,] += self.pts[41,:3]/self.scales 
        sh_pts[:,] += np.array([1, 0, 0.1]).astype(np.float32).reshape(-1, ) 

        debug = 10


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

        # input('g')
        for t, trail in enumerate(self._trails):
            if not t in rhand_key_idx:
            # if not t in rhand_key_idx+table_key_idx+object_key_idx+gravity_key_idx+ground_key_idx:
            # if False:
                continue
            
            # glColor4f(*(COLORS[t % len(COLORS)] + (0.7, )))
            if t in rth_kidx: 
                glColor4f(*(COLORS[0 % len(COLORS)] + (0.7, )))
            elif t in rihand_kidx:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            elif t in ridx_kidx:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            elif t in rmid_kidx:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            elif t in rrng_kidx:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            elif t in rpnk_kidx:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
            elif t in table_key_idx:
                glColor4f(*(COLORS[6 % len(COLORS)] + (0.9, )))
            elif t in object_key_idx:
                glColor4f(*(COLORS[2 % len(COLORS)] + (0.8, )))
            elif t in gravity_key_idx:
                glColor4f(*(COLORS[3 % len(COLORS)] + (0.8, )))
            elif t in ground_key_idx:
                glColor4f(*(COLORS[0 % len(COLORS)] + (0.6, ))) 
            else:
                glColor4f(*(COLORS[t % len(COLORS)] + (0.7, )))
            

            point = None
            glBegin(GL_LINES)
            for point in trail:
                glVertex3f(*point)
            glEnd()

            with gl_context(translate=point, scale=(sphere_size, sphere_size, sphere_size)):
                self.sphere.draw(GL_TRIANGLES)

        # >>>>>>>>>>>>> draw shadow_hand_pts

        glColor4f(*(COLORS[4 % len(COLORS)] + (0.8, )))
        valid_idx = set(sh_idx) 
        # valid_idx = valid_idx - set(sh_th_idx) - set(sh_ff_idx) 
        valid_idx = valid_idx - set(sh_ff_idx) 
        valid_idx = sh_th_idx 
        for idx, point in enumerate(sh_pts):
            # if not (idx in valid_idx):
            #     continue
            glColor4f(*(COLORS[4 % len(COLORS)] + (0.8, )))
            if idx in sh_ff_idx:
                glColor4f(*(COLORS[5 % len(COLORS)] + (0.8, )))
            elif idx in sh_th_idx:
                glColor4f(*(COLORS[1 % len(COLORS)] + (0.8, )))
            else:
                pass
            with gl_context(translate=point, scale=(sphere_size, sphere_size, sphere_size)):
                self.sphere.draw(GL_TRIANGLES) 

        # if len(trail) > 3:
        #     glColor4f(*(COLORS[4 % len(COLORS)] + (0.8, )))
        #     point = None
        #     # pts = [[1,2,3.], [3,4,5.]]
        #     glBegin(GL_POINT)

        #     # glVertex3f(*trail[0])
        #     # glVertex3f(*trail[1])

        #     for point in sh_pts:
        #         glVertex3f(*point)
        #     glEnd()
        #     # with gl_context(translate=point, scale=(0.02, 0.02, 0.02)):
        #     #     self.sphere.draw(GL_TRIANGLES)
    


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
        elif key == k.Z:
            print('gg')
            exit


    def update(self, dt):
        if self.paused:
            return
        # for trail, point in zip(self._trails, self._next_frame()[1]):
        for trail, point in zip(self._trails, self.pts):
            if point[3] > -1 or not len(trail):
                # trail.append(point[:3] / 600.)
                trail.append(point[:3] / self.scales)
                # trail.append(point[:3] / 1.)
            else:
                trail.append(trail[-1])

    def mainloop(self):
        pyglet.clock.schedule_interval(self.update, 100 / self._frame_rate)
        pyglet.app.run()

if __name__ == '__main__':



    debug = 10


    MyViewer(c3d.Reader(open(filename, 'rb'))).mainloop()
    exit()
    
    for filename in filelist:
        filename = os.path.join(fileroot, filename)
        try:
            viewer = Viewer(c3d.Reader(open(filename, 'rb')))

            # viewer.on_key_press()
            viewer.mainloop()
        except StopIteration:
            pass

    pass
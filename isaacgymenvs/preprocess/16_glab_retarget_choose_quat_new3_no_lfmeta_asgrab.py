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
c_scale = 0.003


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




def quat_pos(x):
    """
    make all the real part of the quaternion positive
    """
    q = x
    z = (q[..., 3:] < 0).float()
    q = (1 - 2 * z) * q
    return q

def quat_abs(x):
    """
    quaternion norm (unit quaternion represents a 3D rotation, which has norm of 1)
    """
    x = x.norm(p=2, dim=-1)
    return x

def quat_unit(x):
    """
    normalized quaternion with norm of 1
    """
    norm = quat_abs(x).unsqueeze(-1)
    return x / (norm.clamp(min=1e-9))

def quat_normalize(q):
    """
    Construct 3D rotation from quaternion (the quaternion needs not to be normalized).
    """
    q = quat_unit(quat_pos(q))  # normalized to positive and unit quaternion
    return q

def quat_from_rotation_matrix(m):
    """
    Construct a 3D rotation from a valid 3x3 rotation matrices.
    Reference can be found here:
    http://www.cg.info.hiroshima-cu.ac.jp/~miyazaki/knowledge/teche52.html

    :param m: 3x3 orthogonal rotation matrices.
    :type m: Tensor

    :rtype: Tensor
    """
    m = m.unsqueeze(0)
    diag0 = m[..., 0, 0]
    diag1 = m[..., 1, 1]
    diag2 = m[..., 2, 2]

    # Math stuff.
    w = (((diag0 + diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    x = (((diag0 - diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    y = (((-diag0 + diag1 - diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5
    z = (((-diag0 - diag1 + diag2 + 1.0) / 4.0).clamp(0.0, None)) ** 0.5

    # Only modify quaternions where w > x, y, z.
    c0 = (w >= x) & (w >= y) & (w >= z)
    x[c0] *= (m[..., 2, 1][c0] - m[..., 1, 2][c0]).sign()
    y[c0] *= (m[..., 0, 2][c0] - m[..., 2, 0][c0]).sign()
    z[c0] *= (m[..., 1, 0][c0] - m[..., 0, 1][c0]).sign()

    # Only modify quaternions where x > w, y, z
    c1 = (x >= w) & (x >= y) & (x >= z)
    w[c1] *= (m[..., 2, 1][c1] - m[..., 1, 2][c1]).sign()
    y[c1] *= (m[..., 1, 0][c1] + m[..., 0, 1][c1]).sign()
    z[c1] *= (m[..., 0, 2][c1] + m[..., 2, 0][c1]).sign()

    # Only modify quaternions where y > w, x, z.
    c2 = (y >= w) & (y >= x) & (y >= z)
    w[c2] *= (m[..., 0, 2][c2] - m[..., 2, 0][c2]).sign()
    x[c2] *= (m[..., 1, 0][c2] + m[..., 0, 1][c2]).sign()
    z[c2] *= (m[..., 2, 1][c2] + m[..., 1, 2][c2]).sign()

    # Only modify quaternions where z > w, x, y.
    c3 = (z >= w) & (z >= x) & (z >= y)
    w[c3] *= (m[..., 1, 0][c3] - m[..., 0, 1][c3]).sign()
    x[c3] *= (m[..., 2, 0][c3] + m[..., 0, 2][c3]).sign()
    y[c3] *= (m[..., 2, 1][c3] + m[..., 1, 2][c3]).sign()

    return quat_normalize(torch.stack([x, y, z, w], dim=-1)).squeeze(0)


def batch_rodrigues(rot_vecs, epsilon: float = 1e-8):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''
    assert len(rot_vecs.shape) == 2, (
        f'Expects an array of size Bx3, but received {rot_vecs.shape}')

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device
    dtype = rot_vecs.dtype

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True, p=2)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat



def _index_from_letter(letter: str):
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
):
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])



def matrix_to_euler_angles(matrix, convention: str):
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


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
rh_fullpose48 = torch.tensor(np.concatenate([rhands['global_orient'], rhands['fullpose']], axis=1)).float().view(-1, 16, 3) 
rh_lr_quat = quat_from_rotation_matrix(batch_rodrigues(rh_fullpose48.view(-1, 3))).view(-1, 16, 4)
rh_gt = torch.tensor(rhands['transl']).float()


object_ = whole_body['object'].flatten()[0]
objects = object_['params']
objects['transl']
objects['global_orient']
objects_transl = torch.tensor(objects['transl']).float()
objects_gr = torch.tensor(objects['global_orient']).float()
obj_gr_quat = quat_from_rotation_matrix(batch_rodrigues(objects_gr))
obj_gt = torch.tensor(objects['global_orient']).float()


table_ = whole_body['table'].flatten()[0]
tables = table_['params']
tables['transl']
tables['global_orient']
tables_transl = torch.tensor(tables['transl']).float()
tables_gr = torch.tensor(tables['global_orient']).float()
tab_gr_quat = quat_from_rotation_matrix(batch_rodrigues(tables_gr))
tab_gt = torch.tensor(tables['global_orient']).float()


bodys = whole_body['object'].flatten()[0]['params'] 
import torch
node_names = ['right_wrist', 
              'right_index1', 'right_index2', 'right_index3',
              'right_middle1', 'right_middle2', 'right_middle3',
              'right_pinky1', 'right_pinky2', 'right_pinky3',
              'right_ring1', 'right_ring2', 'right_ring3', 
              'right_thumb1', 'right_thumb2', 'right_thumb3', 
            ]
grab_name_dict = {node_names[i]:i for i in range(len(node_names))}






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
         [ 0.0957,  0.0064,  0.0062],
         [ 0.0076,  0.0012,  0.0269],
         [-0.0251,  0.0052,  0.0291],
         [-0.0473,  0.0039,  0.0290],
         [ 0.0010,  0.0049,  0.0028],
         [-0.0302,  0.0068, -0.0028],
         [-0.0531,  0.0055, -0.0067],
         [ 0.0269, -0.0036, -0.0370],
         [ 0.0099, -0.0035, -0.0495],
         [-0.0060, -0.0042, -0.0599],
         [ 0.0139,  0.0024, -0.0205],
         [-0.0144,  0.0045, -0.0256],
         [-0.0379,  0.0028, -0.0332],
         [ 0.0716, -0.0091,  0.0320],
         [ 0.0519, -0.0082,  0.0557],
         [ 0.0297, -0.0137,  0.0702]
    ]
)

local_translation = torch.zeros_like(global_translation)
local_translation[1:] = global_translation[1:] - global_translation[parent_indices[1:]]

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
            r=rh_lr_quat, t=rh_gt, is_local=True)


grab_motion_poses.global_translation

# >>>>>>>> [END mano] >>>>>>>>


# grab_local_transl = SkeletonState.zero_pose(grab_motion_poses.skeleton_tree).local_translation

tmp_local_rot = quat_identity([grab_motion_poses.skeleton_tree.num_joints])
# tmp_local_rot[grab_name_dict['robot0:thbase']] = torch.tensor([0, 0.382499, 0, 0.923956]).float()
tmp_local_rot[grab_name_dict['right_wrist']] = quat_from_euler_xyz(*torch.tensor([0, torch.pi/2, 0]).float())

grab_rot_pose = SkeletonState.from_rotation_and_root_translation(
    skeleton_tree=grab_motion_poses.skeleton_tree,
    r=tmp_local_rot,
    t=torch.zeros(3, dtype=grab_motion_poses.skeleton_tree.local_translation.dtype),
    is_local=True,
)



grab_local_rot = grab_rot_pose.local_rotation
grab_local_transl = grab_rot_pose.local_translation
debug = 10

grab_rot_pts = grab_rot_pose.global_translation

# >>>>>>>> [BEGIN 凑 retarget需要的参数] >>>>>>>>
shadow_hand_name_list = ['robot0:hand mount', 'robot0:palm', 'robot0:ffknuckle', 
 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
 'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
 'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 
 'robot0:rfmiddle', 'robot0:rfdistal', 
 'robot0:lfmetacarpal', 
 'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 
 'robot0:lfdistal', 'robot0:thbase', 'robot0:thproximal', 
 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
sh_name_dict = {shadow_hand_name_list[i]:i for i in range(len(shadow_hand_name_list))}

## >>>>>>>> [获取tar_hand] >>>>>>>
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/open_ai_assets/hand/shadow_hand.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24_retarget.xml'
# mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24_retarget_grab.xml'
mjcf_path = '/home/mingxian/project/IsaacGymEnvs/assets/hand/mjcf/shadow_hand_24_retarget_v2.xml'


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





sh_local_rot = quat_identity([sh_tree.num_joints])
sh_local_rot[sh_name_dict['robot0:hand mount']] = quat_mul(quat_from_euler_xyz(*torch.tensor([0, 0, 0.025]).float()), sh_local_rot[sh_name_dict['robot0:hand mount']])
sh_local_rot[sh_name_dict['robot0:hand mount']] = quat_mul(quat_from_euler_xyz(*torch.tensor([0, -0.15, 0]).float()), sh_local_rot[sh_name_dict['robot0:hand mount']]) 
sh_local_rot[sh_name_dict['robot0:thbase']]= torch.tensor([0, 0.382499, 0, 0.923956]).float()
sh_local_rot[sh_name_dict['robot0:lfknuckle']] = quat_from_euler_xyz(*torch.tensor([0, -0.45, 0]).float())
sh_local_rot[sh_name_dict['robot0:ffknuckle']] = quat_from_euler_xyz(*torch.tensor([0, 0.2, 0]).float())
sh_local_rot[sh_name_dict['robot0:rfknuckle']] = quat_from_euler_xyz(*torch.tensor([0, -0.1, 0]).float())
sh_local_rot[sh_name_dict['robot0:thhub']] = quat_from_euler_xyz(*torch.tensor([0, -0.2, 0]).float())
sh_local_rot[sh_name_dict['robot0:thbase']] = quat_mul(quat_from_euler_xyz(*torch.tensor([0, 0.2, 0]).float()), sh_local_rot[sh_name_dict['robot0:thbase']])









grab_tpose_pts = grab_pts

pts = [
    grab_tpose_pts,
    sh_init_pts,
    sh_rot_pts,
    grab_rot_pts,
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


class GRABBVH():
    def __init__(self, pa) -> None:
        pass

    def output(self):

        grab_tree.node_names
        grab_tree.local_translation        
        infos = r'HIERARCHY\n'
        suf = ''
        for i in range(len(grab_tree.num_joints)):
            ch_idx = i
            pa_idx = grab_tree.parent_indices[ch_idx]
            if pa_idx == -1:
                tmp_infos = r'ROOT %s\
                aa s'
                # '
            else:
                pass
            infos += tmp_infos
            

 
class Viewer(pyglet.window.Window):
    def __init__(self, trace=None, paused=False):
        self.display_buf = [1, 0, 0, 1, 0, 0]
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
            # if t == grab_name_dict['right_index1']:
            # if t == 13:
            #     # 'right_thumb1','right_index1','right_middle1','right_ring1','right_pinky1'
            #     continue
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
            elif t < 64+24:
                if display_buf[3] == 1:
                    glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))
                else:
                    continue
            elif t < 88+16:
                if display_buf[4] == 1:
                    glColor4f(*(COLORS[4 % len(COLORS)] + (0.7, )))
                else:
                    continue
            # elif t < 96+24:
            #     if display_buf[5] == 1:
            #         glColor4f(*(COLORS[5 % len(COLORS)] + (0.7, )))
            #     else:
            #         continue
            else:
                continue
                glColor4f(*(COLORS[5 % len(COLORS)] + (0.7, )))
            
            # visualization tese for thumb
            # if t == 13:
            #     glColor4f(*(COLORS[1 % len(COLORS)] + (0.7, )))
            # elif t == 14:
            #     glColor4f(*(COLORS[2 % len(COLORS)] + (0.7, )))
            # elif t == 15:
            #     glColor4f(*(COLORS[3 % len(COLORS)] + (0.7, )))

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
        self.display_buf = [1, 0, 0, 1, 0, 0]
        # self.display_buf = [1, 0, 0, 0, 0, 0]

        # self.pts_vis = pts_vis

        self.tar_local_rot = quat_identity([sh_tree.num_joints])

        # 调节shadow hand的t-pose，使他和grab hand对齐
        self.scale_to_target_skeleton = 1.07

        # self.tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_from_euler_xyz(*torch.tensor([0, -torch.pi/2, 0]).float())

        # self.tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_from_euler_xyz(*torch.tensor([0, 0, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_mul(quat_from_euler_xyz(*torch.tensor([0, 0, 0.025]).float()), self.tar_local_rot[sh_name_dict['robot0:hand mount']])
        self.tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_mul(quat_from_euler_xyz(*torch.tensor([0, -0.15, 0]).float()), self.tar_local_rot[sh_name_dict['robot0:hand mount']]) 
        self.tar_local_rot[sh_name_dict['robot0:thbase']]= torch.tensor([0, 0.382499, 0, 0.923956]).float()


        # self.tar_local_rot[sh_name_dict['robot0:thbase']]= quat_mul(quat_from_euler_xyz(*torch.tensor([0, -0.15, 0]).float()), self.tar_local_rot[sh_name_dict['robot0:thbase']]) 
        # self.tar_local_rot[sh_name_dict['robot0:thbase']]= quat_mul(
        #     quat_from_euler_xyz(*torch.tensor([0, 0.349, 0]).float()), 
        #     torch.tensor([0, 0.382499, 0, 0.923956]).float()) 

        # self.tar_local_rot[sh_name_dict['robot0:hand mount']] = quat_mul(,quat_from_euler_xyz(*torch.tensor([0.10, 0, 0]).float()), self.tar_local_rot[sh_name_dict['robot0:hand mount']])
        # 'robot0:palm', 'robot0:ffknuckle', 
        #  'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
        #  'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 
        #  'robot0:mfdistal', 'robot0:rfknuckle', 'robot0:rfproximal', 
        #  'robot0:rfmiddle', 'robot0:rfdistal', 'robot0:lfmetacarpal', 
        #  'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 
        #  'robot0:lfdistal', 'robot0:thbase', 'robot0:thproximal', 
        #  'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal']
        # tar_local_rot[sh_name_dict['robot0:thbase']] = torch.tensor([0, 0.382499, 0, 0.923956]).float()


        self.tar_local_rot[sh_name_dict['robot0:lfknuckle']] = quat_from_euler_xyz(*torch.tensor([0, -0.45, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:ffknuckle']] = quat_from_euler_xyz(*torch.tensor([0, 0.2, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:rfknuckle']] = quat_from_euler_xyz(*torch.tensor([0, -0.1, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:thhub']] = quat_from_euler_xyz(*torch.tensor([0, -0.2, 0]).float())
        self.tar_local_rot[sh_name_dict['robot0:thbase']] = quat_mul(quat_from_euler_xyz(*torch.tensor([0, 0.2, 0]).float()), self.tar_local_rot[sh_name_dict['robot0:thbase']])

        self.tar_local_rot[sh_name_dict['robot0:lfmetacarpal']] = quat_from_euler_xyz(*torch.tensor([0, -0.05, 0]).float())

    def update(self, dt):
        if self.paused:
            return
        

        sh_tpose = SkeletonState.from_rotation_and_root_translation(
            skeleton_tree=sh_tree,
            r=self.tar_local_rot,
            t=torch.zeros(3, dtype=sh_tree.local_translation.dtype),
            is_local=True,
        )
        sh_tpose_pts = sh_tpose.global_translation.numpy()  

        src_hand_tpose = grab_tpose_pts * self.scale_to_target_skeleton
        grab_rot_pts_ = grab_rot_pts* self.scale_to_target_skeleton

        pts = [
            # src_hand_tpose,
            grab_rot_pts_,
            sh_init_pts,
            sh_rot_pts,
            sh_tpose_pts,
        ]
        pts_vis = np.concatenate(pts, axis=0)

        # pts_vis = self.pts_vis
        for trail, point in zip(self._trails, pts_vis):
            trail.append(point[:3])


# grab_rot_pts_[grab_name_dict['right_index1']]
# sh_tpose_pts[sh_name_dict['robot0:ffknuckle']]
# sh_tpose_pts[sh_name_dict['robot0:ffknuckle']] - sh_tpose_pts[sh_name_dict['robot0:palm']]
# grab_rot_pts_[grab_name_dict['right_index1']].detach().numpy() - sh_tpose_pts[sh_name_dict['robot0:palm']]

# shadow_hand_name_list = ['robot0:hand mount', 'robot0:palm', 
# 'robot0:ffknuckle', 'robot0:ffproximal', 'robot0:ffmiddle', 'robot0:ffdistal', 
#  'robot0:mfknuckle', 'robot0:mfproximal', 'robot0:mfmiddle', 'robot0:mfdistal',
#  'robot0:rfknuckle', 'robot0:rfproximal', 'robot0:rfmiddle', 'robot0:rfdistal', 
#  'robot0:lfmetacarpal', 'robot0:lfknuckle', 'robot0:lfproximal', 'robot0:lfmiddle', 'robot0:lfdistal', 
# 'robot0:thbase', 'robot0:thproximal', 'robot0:thhub', 'robot0:thmiddle', 'robot0:thdistal'
# ]

# node_names = ['right_wrist', 
#               'right_thumb1', 'right_thumb2', 'right_thumb3',
#               'right_index1', 'right_index2', 'right_index3', 
#               'right_middle1', 'right_middle2', 'right_middle3',
#               'right_ring1', 'right_ring2', 'right_ring3', 
#               'right_pinky1', 'right_pinky2', 'right_pinky3', 
#             ]

# gmp_lr=grab_motion_poses.local_rotation


# fullpose = torch.tensor(rhands['fullpose']).float().reshape(-1, 15, 3)


# quat_mul(quat_conjugate(quat_from_euler_xyz(*fullpose[1000][1])), quat_from_euler_xyz(*fullpose[1000][2]))


# quat_from_rotation_matrix(batch_rodrigues(fullpose[1000][1].reshape(1, -1)))
# quat_from_rotation_matrix(batch_rodrigues(fullpose[1000][:]))

# quat_conjugate(quat_from_rotation_matrix(batch_rodrigues(fullpose[0][:])))
# quat_mul(
#     quat_conjugate(quat_from_rotation_matrix(batch_rodrigues(fullpose[0][:]))), 
#     quat_from_rotation_matrix(batch_rodrigues(fullpose[2000][:])))


# go = torch.tensor(rhands['global_orient']).float()

# quat_mul(
#     quat_from_rotation_matrix(batch_rodrigues(fullpose[0][:])),
#     quat_from_rotation_matrix(batch_rodrigues(go[0:1][:]))
#     )

# quat_mul(
#     quat_from_rotation_matrix(batch_rodrigues(fullpose[0][0])),
#     quat_from_rotation_matrix(batch_rodrigues(go[0][:]))
#     )

# matrix_to_euler_angles(batch_rodrigues(fullpose[0][:]), 'ZYX')

if __name__ == '__main__':



    import pyrender
    import trimesh
    import smplx
    import os

    class Mesh(trimesh.Trimesh):

        def __init__(self,
                    filename=None,
                    vertices=None,
                    faces=None,
                    vc=None,
                    fc=None,
                    vscale=None,
                    process = False,
                    visual = None,
                    wireframe=False,
                    smooth = False,
                    **kwargs):

            self.wireframe = wireframe
            self.smooth = smooth

            if filename is not None:
                mesh = trimesh.load(filename, process = process)
                vertices = mesh.vertices
                faces= mesh.faces
                visual = mesh.visual
            if vscale is not None:
                vertices = vertices*vscale

            if faces is None:
                mesh = points2sphere(vertices)
                vertices = mesh.vertices
                faces = mesh.faces
                visual = mesh.visual

            super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

            if vc is not None:
                self.set_vertex_colors(vc)
            if fc is not None:
                self.set_face_colors(fc)

        def rot_verts(self, vertices, rxyz):
            return np.array(vertices * rxyz.T)

        def colors_like(self,color, array, ids):

            color = np.array(color)

            if color.max() <= 1.:
                color = color * 255
            color = color.astype(np.int8)

            n_color = color.shape[0]
            n_ids = ids.shape[0]

            new_color = np.array(array)
            if n_color <= 4:
                new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
            else:
                new_color[ids, :] = color

            return new_color

        def set_vertex_colors(self,vc, vertex_ids = None):

            all_ids = np.arange(self.vertices.shape[0])
            if vertex_ids is None:
                vertex_ids = all_ids

            vertex_ids = all_ids[vertex_ids]
            new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
            self.visual.vertex_colors[:] = new_vc

        def set_face_colors(self,fc, face_ids = None):

            if face_ids is None:
                face_ids = np.arange(self.faces.shape[0])

            new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
            self.visual.face_colors[:] = new_fc

        @staticmethod
        def concatenate_meshes(meshes):
            return trimesh.util.concatenate(meshes)
    
    grab_path = '/home/mingxian/project/grab/grab_github_unzip/grab'
    vtemp_path = 'tools/subject_meshes/male/s2_rhand.ply'
    rh_mesh = os.path.join(grab_path, '..', vtemp_path)
    rh_vtemp = np.array(Mesh(filename=rh_mesh).vertices)

    # rh_m = smplx.create(model_path='/home/mingxian/project/grab/grab_github_unzip/models',
    #                     model_type='mano',
    #                     is_rhand = True,
    #                     v_template = rh_vtemp,
    #                     num_pca_comps=24,
    #                     flat_hand_mean=True,
    #                     batch_size=1034)

    # rh_m = smplx.create(model_path='/home/mingxian/project/grab/grab_github_unzip/models',
    #                     model_type='mano',
    #                     is_rhand = True,
    #                     v_template = rh_vtemp,
    #                     num_pca_comps=24,
    #                     flat_hand_mean=True)
    
    rh_m = smplx.create(model_path='/home/mingxian/project/grab/grab_github_unzip/models',
                        model_type='mano',
                        is_rhand = True,
                        num_pca_comps=24,
                        flat_hand_mean=True)
    rh_m.parents
    
    # outputs = rh_m()
    fms = 1230
    # outputs = rh_m.forward(right_hand_pose=torch.tensor(rhands['fullpose'][fms]), global_orient =torch.tensor(rhands['global_orient'][fms]).view(1, -1))
    # outputs = rh_m.forward(hand_pose=torch.tensor(rhands['hand_pose'][fms]), global_orient =torch.tensor(rhands['global_orient'][fms]).view(1, -1))
    outputs = rh_m.forward(hand_pose=torch.tensor(rhands['hand_pose'][fms:fms+1]))
    # outputs = rh_m.forward()

    outputs.hand_pose
    outputs.global_orient
    outputs.joints
    outputs.transl
    outputs.vertices
    # Mesh(filename=rh_mesh).export('tmp.obj')
    tmp_obj = Mesh(filename=rh_mesh) 
    # Mesh(vertices=outputs.vertices.squeeze(0).detach().numpy(), faces=tmp_obj.faces).export('debug.obj')
    tmps_mesh = Mesh(vertices=outputs.vertices.squeeze(0).detach().numpy(), faces=tmp_obj.faces)
    vertex_colors = np.ones([outputs.vertices.squeeze(0).detach().numpy().shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    # tri_mesh = trimesh.Trimesh(vertices, rh_m.faces,
    #                             vertex_colors=vertex_colors)
    tri_mesh = trimesh.Trimesh(outputs.vertices.squeeze(0).detach().numpy(), faces=tmp_obj.faces,
                                vertex_colors=vertex_colors)

    mesh = pyrender.Mesh.from_trimesh(tri_mesh)

    scene = pyrender.Scene()
    scene.add(mesh)

    # if plot_joints:
    out_j = outputs.joints.detach().numpy()
    sm = trimesh.creation.uv_sphere(radius=0.005)
    sm.visual.vertex_colors = [0.9, 0.1, 0.1, 1.0]
    tfs = np.tile(np.eye(4), (out_j.shape[1], 1, 1))
    tfs[:, :3, 3] = out_j[0]
    joints_pcl = pyrender.Mesh.from_trimesh(sm, poses=tfs)
    scene.add(joints_pcl)

    # pyrender.Viewer(scene, use_raymond_lighting=True)

    
    a = MyViewer()
    a.mainloop()
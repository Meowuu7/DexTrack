from typing import Union, Optional, Any
import os
import enum
from yacs.config import CfgNode as CN
import numpy as np
from skvideo.io import FFmpegWriter
from plyfile import PlyData
import open3d as o3d
import skimage
import time
import trimesh
from torch.utils.tensorboard import SummaryWriter
import pprint
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def merge_cfg(base_cfg: CN, cfg: Union[CN, str], replace: Optional[bool] = False):
    assert isinstance(base_cfg, CN)
    assert isinstance(cfg, (CN, str))
    
    cfg_out = base_cfg.clone()

    if isinstance(cfg, str):
        assert os.path.exists(cfg)
        cfg_out.merge_from_file(cfg)
    else:
        cfg_out.merge_from_other_cfg(cfg)

    base_types = (int, float, str, type(None))
    def _dict_to_cfg(_cfg): # recursively convert dict to CN
        if isinstance(_cfg, (list, tuple)):
            for _i, _v in enumerate(_cfg):
                if not isinstance(_v, base_types):
                    _dict_to_cfg(_v)
                if isinstance(_v, dict):
                    _cfg[_i] = CN(_v)
        else:
            for _k, _v in _cfg.items():
                if not isinstance(_v, base_types):
                    _dict_to_cfg(_v)
                if isinstance(_v, dict):
                    _cfg[_k] = CN(_v)
    
    _dict_to_cfg(cfg_out)

    if replace:
        cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(cfg_out)
        cfg.set_new_allowed(False)
    
    return cfg_out


def set_cfg_attr(cfg: CN, key: str, val: Any):
    fields = key.split('.')
    pointer = cfg
    for x in fields[:-1]:
        pointer = getattr(pointer, x)
    setattr(pointer, fields[-1], val)


def compute_lame_parameters(E: float, nu: float):
    # Compute Lame parameters (mu, lambda) from Young's modulus (E) and Poisson's ratio (nu)
    mu = E / (2 * (1 + nu))
    lambd = E * nu / ((1 + nu) * (1 - 2 * nu))
    return mu, lambd


class Enum(enum.Enum):
    @classmethod
    def members(cls):
        return cls.__members__.values()

    @classmethod
    def is_member(cls, inp):
        return (inp in cls.__members__) or (inp.name in cls.__members__)


def get_video_writer(filename: str, rate: Optional[int] = 30):
    rate = str(rate)
    return FFmpegWriter(filename,
                        inputdict={'-r': rate},
                        outputdict={# '-vcodec': 'h265', #'mpeg4',
                                    '-pix_fmt': 'yuv420p',
                                    '-r': rate})


def recursive_getattr(obj, name):
    out = obj
    for v in name.split('.'):
        assert hasattr(out, v), f'{out} has no attribute {v}'
        out = getattr(out, v)
        
    return out


def qrot2d(rot, v):
    """ Apply rotation with 2D quaternion. """
    return [rot[0]*v[0]-rot[1]*v[1], rot[1]*v[0] + rot[0]*v[1]]


def surface_to_mesh(surface: np.ndarray):
    vertices = []
    indices = []
    grid_size = surface.shape
    for i in range(grid_size[0]):
        for j in range(grid_size[1]):
            vertices.append(surface[i, j])
            if i < (grid_size[0] - 1) and j < (grid_size[1] - 1):
                index = i * grid_size[0] + j
                a = index
                b = index + 1
                c = index + grid_size[0] + 1
                d = index + grid_size[0]
                indices.append((a, b, c))
                indices.append((a, c, d))
    vertices, indices = np.array(vertices), np.array(indices)

    return vertices, indices


def load_mesh(fn, scale=1, offset=(0, 0, 0)):
    if isinstance(scale, (int, float)):
        scale = (scale, scale, scale)
    print(f'loading {fn}')
    if os.path.splitext(fn) == '.ply':
        plydata = PlyData.read(fn)
        x = plydata['vertex']['x']
        y = plydata['vertex']['y']
        z = plydata['vertex']['z']
        elements = plydata['face']
        vertex_indices = elements['vertex_indices']
    else:
        mesh = trimesh.load(fn, skip_materials=True)
        vertices = np.array(mesh.vertices)
        vertex_indices = np.array(mesh.faces)
        x, y, z = vertices[:,0], vertices[:,1], vertices[:,2]
    num_tris = len(vertex_indices)
    triangles = np.zeros((num_tris, 9), dtype=np.float32)

    for i, face in enumerate(vertex_indices):
        assert len(face) == 3
        for d in range(3):
            triangles[i, d * 3 + 0] = x[face[d]] * scale[0] + offset[0]
            triangles[i, d * 3 + 1] = y[face[d]] * scale[1] + offset[1]
            triangles[i, d * 3 + 2] = z[face[d]] * scale[2] + offset[2]

    return triangles


def load_points_from_mesh(fn, scale=1, offset=(0, 0, 0), num_points=5000):
    mesh = o3d.io.read_triangle_mesh(fn)
    pc = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pc.points)
    points = points * np.array(scale) + np.array(offset)

    return points


def scale_triangle_wrt_1cube(triangles, scale=np.ones((3,))):
    # Scale mesh to fit a 1x1x1 cube and scale
    triangles_fl = np.concatenate(np.split(triangles, 3, axis=-1), axis=0)
    triangles_fl_min = triangles_fl.min(0)
    max_edge_length = (triangles_fl.max(0) - triangles_fl_min).max()
    triangles_fl = (triangles_fl - triangles_fl_min) / max_edge_length
    triangles_fl *= scale
    triangles = np.concatenate(np.split(triangles_fl, 3, axis=0), axis=1)
    return triangles


def cartesian_np(coord_list):
    idcs_list = list(range(len(coord_list) + 1))
    idcs_list = [idcs_list[1], idcs_list[0]] + idcs_list[2:]
    out = np.stack(np.meshgrid(*coord_list), -1).transpose(*idcs_list)
    return out


def fig2arr(fig):
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return data


def pcd_to_mesh(pcd, strategy='voxel', voxel_size=None):
    """ Convert Open3d PCD object to Open3d Mesh object. """
    if strategy == 'bpa':
        distances = pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(distances)
        radius = 3 * avg_dist

        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector([radius, radius * 2]))
        
        mesh = mesh.simplify_quadric_decimation(100000)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
    elif strategy == 'poisson':
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8, width=0, scale=1.1, linear_fit=False)[0]
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
    elif strategy == 'voxel':
        # convert point cloud to voxel grid
        if voxel_size is None:
            pts = np.asarray(pcd.points)
            voxel_size = min(pts.max(0) - pts.min(0)) / 10 # 5
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=voxel_size)

        # convert voxel grid indices to voxel grid (fake) sdf
        padding = 3
        voxels = voxel_grid.get_voxels()
        mesh = o3d.geometry.TriangleMesh()
        if len(voxels) > 0:
            indices = np.stack(list(vx.grid_index for vx in voxels))
            indices_shape = indices.max(0) + 1
            voxel_grid_shape = indices_shape + 2 * padding
            voxels_np = np.ones(voxel_grid_shape)
            for idx in indices:
                idx += padding
                voxels_np[idx[0], idx[1], idx[2]] = -1.

            # convert voxel sdf to mesh (in voxel grid coordinate)
            verts, faces, normals, values = skimage.measure.marching_cubes(voxels_np)
            
            # normalize to original coordinate
            voxel_grid_extents = voxel_grid.get_max_bound() - voxel_grid.get_min_bound()
            verts = (verts - padding) / indices_shape * voxel_grid_extents + voxel_grid.get_min_bound()

            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
    else:
        raise ValueError(f'Unrecognized strategy {strategy} to convert point cloud to mesh')

    return mesh


def save_pcd_to_mesh(filepath, pcd):
    mesh = pcd_to_mesh(pcd)
    o3d.io.write_triangle_mesh(filepath, mesh)


class Logger:
    def __init__(self, logdir, write_mode='w', with_tensorboard=False, suffix=""):
        self._text_writer = open(os.path.join(logdir, f'results{suffix}.txt'),
                                 write_mode)
        if with_tensorboard:
            self._tf_writer = SummaryWriter(logdir)
        self._logdir = logdir
        self._scalars = {'default': {}}
        self._timer = {'default': {}}
        self._accum_timer = {'default': {}}

        self._deque_maxlen = 100

    def write(self, iter, group='default', use_scientific_notation=True):
        text = [f'iter#{iter}']
        for name, value in self._scalars[group].items():
            if hasattr(self, '_tf_writer'):
                self._tf_writer.add_scalar(f'{group}/{name}', value, iter)
            if use_scientific_notation:
                text.append(f'{name}: {value:.4e}')
            else:
                text.append(f'{name}: {value}')

        for name, value_list in self._accum_timer[group].items():
            value = np.mean(value_list)
            if hasattr(self, '_tf_writer'):
                self._tf_writer.add_scalar(f'{group}/{name}', value, iter)
            if use_scientific_notation:
                text.append(f't/{name}: {value:.4e}')
            else:
                text.append(f't/{name}: {value}')

        self.print('  '.join(text))

    def reset(self, group='default'):
        for name in self._accum_timer[group].keys():
            self._accum_timer[group][name] = deque(maxlen=self._deque_maxlen)

    def print(self, data):
        if not isinstance(data, str):
            data = pprint.pformat(data)
        self._text_writer.write(data + '\n')
        self._text_writer.flush()
        print(data)

    def scalar(self, name, value, group=None):
        group = 'default' if group is None else group
        self._scalars[group][name] = float(value)

    def tic(self, name, group=None):
        group = 'default' if group is None else group
        self._timer[group][name] = [time.time()]

    def toc(self, name, group=None):
        group = 'default' if group is None else group
        assert len(
            self._timer[group][name]) == 1, f'Should call tic({name}) first'
        self._timer[group][name].append(time.time())

        if not name in self._accum_timer[group].keys():
            self._accum_timer[group][name] = deque(maxlen=self._deque_maxlen)
        tic, toc = self._timer[group][name]
        self._accum_timer[group][name].append(toc - tic)

    def create_group(self, group):
        self._scalars[group] = {}
        self._timer[group] = {}
        self._accum_timer[group] = {}

    def close(self):
        self._text_writer.close()
        if hasattr(self, '_tf_writer'):
            self._tf_writer.close()


def extract_part_pca(pcd, return_part_colors=False, within_part_clustering=True):
    colors = np.asarray(pcd.colors)
    points = np.asarray(pcd.points)
    points_std = StandardScaler().fit_transform(points)
    unique_colors = np.unique(colors, axis=0)
    all_part_pc = dict()
    all_part_pc_pca = dict()
    all_part_pc_std = dict()
    for i, unique_color in enumerate(unique_colors):
        mask = np.all(colors == unique_color[None,:], axis=1)
        masked_points = points[mask]
        masked_points_std = points_std[mask]
        if within_part_clustering: # HACK
            within_part_pcd = o3d.geometry.PointCloud()
            within_part_pcd.points = o3d.utility.Vector3dVector(masked_points)
            eps = max(masked_points.max(0) - masked_points.min(0)) / np.power(masked_points.shape[0], 1/3)
            within_part_labels = np.array(within_part_pcd.cluster_dbscan(eps=eps, min_points=10))
            unique_within_part_labels = np.unique(within_part_labels)
            n_within_part_labels = len(unique_within_part_labels)
            if -1 in unique_within_part_labels:
                n_within_part_labels -= 1
            if n_within_part_labels > 1:
                within_part_mask = within_part_labels == 0
                all_part_pc_pca[i] = masked_points[within_part_mask]
                all_part_pc_std[i] = masked_points_std[within_part_mask]
            else:
                all_part_pc_pca[i] = masked_points
                all_part_pc_std[i] = masked_points_std
            all_part_pc[i] = masked_points
        else:
            all_part_pc[i] = masked_points
            all_part_pc_pca[i] = masked_points
            all_part_pc_std[i] = masked_points_std

    all_part_pca_components = dict()
    all_part_pca_singular_values = dict()
    for k, part_pc_pca in all_part_pc_pca.items():
        pca = PCA(n_components=3)
        pca.fit(part_pc_pca)
        all_part_pca_components[k] = pca.components_
        all_part_pca_singular_values[k] = pca.singular_values_

    if return_part_colors:
        return all_part_pca_components, all_part_pca_singular_values, all_part_pc, unique_colors
    else:
        return all_part_pca_components, all_part_pca_singular_values, all_part_pc


def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))
    
    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])

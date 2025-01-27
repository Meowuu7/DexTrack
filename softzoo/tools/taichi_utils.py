import pstats
import numpy as np
import torch
import taichi as ti

from .general_utils import Enum
from ..engine import NORM_EPS


@ti.func
def clamp_at_zero(x):
    if x >= 0:
        x = ti.max(x, 1e-6)
    else:
        x = ti.min(x, -1e-6)
    return x


class TiDeviceInterface(Enum):
    """ A helper class to handle external array that interfaces with Taichi. """
    Numpy = 0
    TorchCPU = 1
    TorchGPU = 2
    Taichi = 3
    
    def set_dtype(self, i_dtype_ti, f_dtype_ti):
        if self == TiDeviceInterface.Numpy:
            _mapping = {
                ti.i32: np.int32,
                ti.i64: np.int64,
                ti.f32: np.float32,
                ti.f64: np.float64,
            }
        elif self in [TiDeviceInterface.TorchCPU, TiDeviceInterface.TorchGPU]:
            _mapping = {
                ti.i32: torch.int32,
                ti.i64: torch.int64,
                ti.f32: torch.float32,
                ti.f64: torch.float64,
            }
        elif self == TiDeviceInterface.Taichi:
            _mapping = {
                ti.i32: ti.i32,
                ti.i64: ti.i64,
                ti.f32: ti.f32,
                ti.f64: ti.f64,
            }
        self._i_dtype = _mapping[i_dtype_ti]
        self._f_dtype = _mapping[f_dtype_ti]
        
    def create_field(self, shape, dtype='f', vec_dim=None, mat_dim=None):
        dtype = self.i_dtype if dtype == 'i' else self.f_dtype
        if self == TiDeviceInterface.Taichi:
            if vec_dim is not None:
                tensor = ti.Vector.field(vec_dim, dtype, shape=shape)
            elif mat_dim is not None:
                tensor = ti.Matrix.field(*mat_dim, dtype, shape=shape)
            else:
                tensor = ti.field(dtype, shape=shape)
        else:
            if vec_dim is not None:
                shape = list(shape) + [vec_dim]
            elif mat_dim is not None:
                shape = list(shape) + list(mat_dim)
            if dtype == 'i':
                tensor = self.create_i_tensor(shape)
            else:
                tensor = self.create_f_tensor(shape)

        return tensor

    def create_i_tensor(self, shape):
        if self == TiDeviceInterface.Numpy:
            tensor = np.zeros(shape, dtype=self.i_dtype)
        elif self == TiDeviceInterface.TorchCPU:
            if self.i_dtype == torch.int32:
                tensor = torch.IntTensor(*shape)
            else:
                tensor = torch.LongTensor(*shape)
            tensor.fill_(0)
        elif self == TiDeviceInterface.TorchGPU:
            if self.i_dtype == torch.int32:
                tensor = torch.cuda.IntTensor(*shape)
            else:
                tensor = torch.cuda.LongTensor(*shape)
            tensor.fill_(0)

        return tensor

    def create_f_tensor(self, shape):
        if self == TiDeviceInterface.Numpy:
            tensor = np.zeros(shape, dtype=self.f_dtype)
        elif self == TiDeviceInterface.TorchCPU:
            if self.f_dtype == torch.float32:
                tensor = torch.FloatTensor(*shape)
            else:
                tensor = torch.DoubleTensor(*shape)
            tensor.fill_(0)
        elif self == TiDeviceInterface.TorchGPU:
            if self.f_dtype == torch.float32:
                tensor = torch.cuda.FloatTensor(*shape)
            else:
                tensor = torch.cuda.DoubleTensor(*shape)
            tensor.fill_(0)

        return tensor

    def tensor(self, tensor, dtype='float64'):
        if isinstance(tensor, torch.Tensor):
            if dtype == 'int32':
                tensor = tensor.to(torch.int32)
            elif dtype == 'int64':
                tensor = tensor.to(torch.int64)
            elif dtype == 'float32':
                tensor = tensor.to(torch.float32)
            else:
                tensor = tensor.to(torch.float64)
            
            if self == TiDeviceInterface.TorchGPU:
                tensor = tensor.cuda()
        else:
            if self == TiDeviceInterface.Numpy:
                tensor = np.array(tensor, dtype=dtype)
            elif self == TiDeviceInterface.TorchCPU:
                if dtype == 'int32':
                    tensor = torch.IntTensor(tensor)
                elif dtype == 'int64':
                    tensor = torch.LongTensor(tensor)
                elif dtype == 'float32':
                    tensor = torch.FloatTensor(tensor)
                else:
                    tensor = torch.DoubleTensor(tensor)
            elif self == TiDeviceInterface.TorchGPU:
                if dtype == 'int32':
                    tensor = torch.cuda.IntTensor(tensor)
                elif dtype == 'int64':
                    tensor = torch.cuda.LongTensor(tensor)
                elif dtype == 'float32':
                    tensor = torch.cuda.FloatTensor(tensor)
                else:
                    tensor = torch.cuda.DoubleTensor(tensor)

        return tensor

    def stack(self, tensor_list, **kwargs):
        if self in [TiDeviceInterface.TorchCPU, TiDeviceInterface.TorchGPU]:
            out = torch.stack(tensor_list, **kwargs)
        else:
            raise NotImplementedError

        return out

    def cat(self, tensor_list, **kwargs):
        if self in [TiDeviceInterface.TorchCPU, TiDeviceInterface.TorchGPU]:
            out = torch.cat(tensor_list, **kwargs)
        else:
            raise NotImplementedError

        return out

    def clone(self, tensor):
        if self in [TiDeviceInterface.TorchCPU, TiDeviceInterface.TorchGPU]:
            out = tensor.clone()
        elif self == TiDeviceInterface.Numpy:
            out = tensor.copy()

        return out

    def from_ext(self, tensor_ti, tensor_ext):
        if self in [TiDeviceInterface.TorchCPU, TiDeviceInterface.TorchGPU]:
            tensor_ti.from_torch(tensor_ext)
        elif self == TiDeviceInterface.Numpy:
            tensor_ti.from_numpy(tensor_ext)

    def to_ext(self, tensor_ti):
        # NOTE: make sure avoid doing this with dynamic taichi tensor
        if self == TiDeviceInterface.TorchCPU:
            out = tensor_ti.to_torch(device='cpu')
        elif self == TiDeviceInterface.TorchGPU:
            out = tensor_ti.to_torch(device='cuda')
        elif self == TiDeviceInterface.Numpy:
            out = tensor_ti.to_numpy()

        return out

    @property
    def i_dtype(self):
        return self._i_dtype

    @property
    def f_dtype(self):
        return self._f_dtype


@ti.func
def cross2d(w, v):
    """ Cross product in 2D space, where w is a scalar and v is a 2-dim vector. """
    return ti.Vector([-w * v[1], w * v[0]])


@ti.func
def cross2d_scalar(a, b):
    """ Cross product in 2D that only returns the magnitude of the z value. """
    return a[0] * b[1] - a[1] * b[0]


@ti.func
def inside_ccw(p, a, b, c):
    """ Used in voxelizer. """
    return cross2d_scalar(a - p, b - p) >= 0 and cross2d_scalar(
        b - p, c - p) >= 0 and cross2d_scalar(c - p, a - p) >= 0


@ti.func
def qrot2d(rot, v):
    """ Apply rotation with 2D quaternion. """
    return ti.Vector([rot[0]*v[0]-rot[1]*v[1], rot[1]*v[0] + rot[0]*v[1]])


@ti.func
def qrot3d(rot, v):
    """ Apply rotation with 3D quaternion. """
    qvec = ti.Vector([rot[1], rot[2], rot[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (rot[0] * uv + uuv)


@ti.func
def qmul3d(q, r):
    """ Quaternion multiplication in 3D. """
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z])
    return out / ti.sqrt(out.dot(out)) # normalize it to prevent some unknown NaN problems.


@ti.func
def w2quat2d(axis_angle, dtype):
    """ 2D angle to quaternion. """
    w = axis_angle.norm(NORM_EPS)
    out = ti.Vector.zero(dt=dtype, n=4)
    out[0] = 1.
    if w > 1e-9:
        v = (axis_angle/w) * ti.sin(w/2)
        out[0] = ti.cos(w/2)
        out[1] = 0.
        out[2] = 0.
        out[3] = v[0] # MPM rotates along z axis in 2D
    return out


@ti.func
def w2quat3d(axis_angle, dtype):
    """ 3D angle to quaternion. """
    w = axis_angle.norm(NORM_EPS)
    out = ti.Vector.zero(dt=dtype, n=4)
    out[0] = 1.
    if w > 1e-9:
        v = (axis_angle/w) * ti.sin(w/2)
        out[0] = ti.cos(w/2)
        out[1] = v[0]
        out[2] = v[1]
        out[3] = v[2]
    return out


@ti.func
def inv_trans2d(pos, position, rotation):
    """ Apply 2D transformation. """
    inv_quat = ti.Vector([rotation[0], -rotation[3]]).normalized(NORM_EPS)
    return qrot2d(inv_quat, pos - position)


@ti.func
def inv_trans3d(pos, position, rotation):
    """ Apply 3D transformation. """
    inv_quat = ti.Vector([rotation[0], -rotation[1], -rotation[2], -rotation[3]]).normalized(NORM_EPS)
    return qrot3d(inv_quat, pos - position)

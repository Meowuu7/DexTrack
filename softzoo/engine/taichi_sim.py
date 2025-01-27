from typing import Optional, Any
from yacs.config import CfgNode as CN
import taichi as ti

from ..tools.general_utils import Enum, recursive_getattr
from ..tools.taichi_utils import TiDeviceInterface
from .mpm_solver import MPMSolver


class TensorType(Enum):
    Scalar = 0
    Vector = 1
    Matrix = 2
    TimedScalar = 3
    TimedVector = 4
    TimedMatrix = 5
    ActBuffer = 6
    VBuffer = 7


@ti.data_oriented
class TaichiSim:
    def __init__(self,
                 cfg: CN,
                 device: Optional[str] = 'numpy',
                 solver: Optional[MPMSolver] = None):
        # Instantiate solver for simulation
        self.solver = MPMSolver(cfg.SIMULATOR) if solver is None else solver    
        self.is_initialized = False
        self.cfg = cfg

        # Set up data interface between taichi and external array
        if device == 'numpy':
            device = TiDeviceInterface.Numpy
        elif device == 'torch_cpu':
            device = TiDeviceInterface.TorchCPU
        elif device == 'torch_gpu':
            device = TiDeviceInterface.TorchGPU
        else:
            raise ValueError(f'Unrecognized device {device}')

        device.set_dtype(self.solver.i_dtype, self.solver.f_dtype)
        self.device = device
        
        # Store get/set kernel and associated data container
        self.interface = dict()

        # Interface to upper-level environment
        self.needs_grad = self.solver.needs_grad
        self.frame_dt = cfg.ENVIRONMENT.frame_dt
        n_substeps = int(self.frame_dt / self.solver.default_dt) + 1
        self.substep_dt = self.frame_dt / n_substeps
        if self.needs_grad:
            self.max_steps = self.solver.max_substeps // n_substeps
        else:
            self.max_steps = float('Inf')
            
        # External access before initialization for dynamic setup
        self.external_items = []

    def initialize(self):
        # NOTE: should be called after all particle instantiation; then number of particles is fixed.
        # Add item from config
        cfg = self.cfg
        cfg.ENVIRONMENT.ITEMS = sorted(cfg.ENVIRONMENT.ITEMS, key=lambda x: x.get('spawn_order', 0))
        n_items = len(cfg.ENVIRONMENT.ITEMS)
        p_start, p_end, static_i = 0, 0, 0
        current_spawn_order = 0
        for i, item in enumerate(cfg.ENVIRONMENT.ITEMS):
            item.item_id = i
            self.solver.add(item.type, item) # NOTE: in-place update from default config

            # NOTE: make sure particles/collider with lower spawn order don't overlap with those with higher order
            if item.type[:7] == 'Static.':
                self.solver.set_collider_occupancy(self.solver.current_s, static_i)
                static_i += 1
            
            if i < (n_items - 1) and cfg.ENVIRONMENT.ITEMS[i + 1].get('spawn_order', current_spawn_order) > current_spawn_order:
                p_end = self.solver.n_particles[None]
                # set particle occupancy 
                self.solver.set_particle_occupancy(self.solver.current_s, p_start, p_end)
                p_start = p_end
            current_spawn_order = item.get('spawn_order', current_spawn_order)
        
        # Set up semantic grid (used for environmental sensing)
        self.use_semantic_occupancy = cfg.ENVIRONMENT.use_semantic_occupancy
        if self.use_semantic_occupancy: # unique semantic index and semantic occupancy ##
            semantic_ids = dict() # get unique semantic ids and associated particle ids
            for i, v in enumerate(cfg.ENVIRONMENT.ITEMS + self.external_items):
                if v.semantic_id not in semantic_ids.keys():
                    semantic_ids[v.semantic_id] = dict(particle_id=[], static_component_id=[])
                if hasattr(v, 'particle_id'): # primitive
                    semantic_ids[v.semantic_id]['particle_id'].append(v.particle_id)
                else: # static component
                    static_component_id = [_i for _i, _v in enumerate(self.solver.static_component_info) if _v['item_id'] == i][0]
                    semantic_ids[v.semantic_id]['static_component_id'].append(static_component_id)
                    
            self.semantic_ids = ti.field(dtype=self.solver.i_dtype, shape=(self.solver.n_particles[None],))
            self.semantic_ids.fill(-1)
            self.grid_semantic = dict() # create a dictionary of (id, grid_semantic) and set up their update kernel
            self.grid_semantic_kernel = dict()
            for k, v in semantic_ids.items():
                # semantics of the grid? # 
                self.grid_semantic[k] = ti.field(dtype=self.solver.i_dtype, shape=self.solver.res) # TODO: can have a more efficient way to store this

                kernels = self.instantiate_grid_semantic_kernel(k, v)
                self.grid_semantic_kernel[k] = { # setup kernel
                    'particle': kernels[0],
                    'static_component': kernels[1],
                    'grid_access': kernels[2],
                }

        # Initialize solver
        self.solver.initialize()
        self.initial_state = dict()
        self.update_initial_state()
        
        self.is_initialized = True

    def reset(self):
        assert self.is_initialized, 'Should call initialize() first before doing anything.'

        self.solver.sim_t = 0.
        self.solver.current_s = 0
        self.solver.reset()

        for k, v in self.initial_state.items(): # set initial state
            self.apply('set', k, s=0, ext_arr=v)
        
    def step(self, action, frame_dt):
        assert self.is_initialized, 'Should call initialize() first before doing anything.'
            
        # Step solver
        self.solver.step(action, frame_dt)

    def get_obs(self):
        obs = None
        if self.use_semantic_occupancy:
            obs = self.get_semantic_occupancy()
            obs = self.device.stack(list(obs.values()), dim=-1)

        return obs

    def apply(self, mode: str, tensor_field_name: str, **kernel_kwargs):
        assert mode in ['get', 'set']
        
        key = (mode, tensor_field_name)
        if key not in self.interface.keys():
            self.interface[key] = self.access_function(mode, tensor_field_name)
        kernel, data = self.interface[key]

        if mode == 'get': # NOTE: data is pass-by-reference and thus if taichi tensor changed, the "get" data will also change
            kernel_kwargs.update(dict(ext_arr=data))
            kernel(**kernel_kwargs)
            return data
        else:
            kernel(**kernel_kwargs)

    def update_initial_state(self, x=None, v=None, F=None, C=None, Jp=None):
        if x is None:
            self.initial_state['x'] = self.device.clone(self.apply('get', 'x', s=0))
        if v is None:
            self.initial_state['v'] = self.device.clone(self.apply('get', 'v', s=0))
        if F is None:
            self.initial_state['F'] = self.device.clone(self.apply('get', 'F', s=0))
        if C is None:
            self.initial_state['C'] = self.device.clone(self.apply('get', 'C', s=0))
        if Jp is None:
            self.initial_state['Jp'] = self.device.clone(self.apply('get', 'Jp', s=0))

    def access_function(self, mode: str, tensor_field_name: str):
        n_particles = self.solver.n_particles[None]
        dim = self.solver.dim
        n_actuators = self.solver.n_actuators

        # Determine tensor type and construct data container
        if tensor_field_name in ['material', 'particle_ids']:
            tensor_type = TensorType.Scalar
            data_container = self.device.create_i_tensor((n_particles,))
        elif tensor_field_name in ['mu', 'lambd', 'p_rho', 'mu.grad', 'lambd.grad', 'p_rho.grad']:
            tensor_type = TensorType.Scalar
            data_container = self.device.create_f_tensor((n_particles,))
        elif tensor_field_name in ['F_tmp', 'U', 'sig', 'V', 'F_tmp.grad', 'U.grad', 'sig.grad', 'V.grad']:
            tensor_type = TensorType.TimedMatrix
            data_container = self.device.create_f_tensor((n_particles, dim, dim))
        elif tensor_field_name in ['Jp', 'actuation', 'Jp.grad', 'actuation.grad']:
            tensor_type = TensorType.TimedScalar
            data_container = self.device.create_f_tensor((n_particles,))
        elif tensor_field_name in ['x', 'v', 'x.grad', 'v.grad']:
            tensor_type = TensorType.TimedVector
            data_container = self.device.create_f_tensor((n_particles, dim))
        elif tensor_field_name in ['F', 'C', 'F.grad', 'C.grad']:
            tensor_type = TensorType.TimedMatrix
            data_container = self.device.create_f_tensor((n_particles, dim, dim))
        elif tensor_field_name in ['act_buffer', 'act_buffer.grad']:
            tensor_type = TensorType.ActBuffer
            data_container = self.device.create_f_tensor((n_actuators,))
        elif tensor_field_name in ['v_buffer', 'v_buffer.grad']:
            tensor_type = TensorType.VBuffer
            data_container = self.device.create_f_tensor((n_actuators, self.solver.dim))
        else:
            raise ValueError(f'Unrecognized tensor field name {tensor_field_name}')

        # Instantiate kernel to get/set tensor field in the solver
        if mode == 'get':
            kernel = self.instantiate_get_tensor_field_kernel(tensor_field_name, tensor_type)
        else:
            kernel = self.instantiate_set_tensor_field_kernel(tensor_field_name, tensor_type)
            data_container = None # not used for set
        
        return kernel, data_container

    def instantiate_get_tensor_field_kernel(self, tensor_field_name: str, tensor_type: TensorType):
        assert TensorType.is_member(tensor_type), f'Unrecognized tensor type {tensor_type}'
        tensor_field = recursive_getattr(self.solver, tensor_field_name)

        dim = self.solver.dim
        i_dtype = self.solver.i_dtype
        if tensor_type == TensorType.Scalar:
            @ti.kernel
            def get_tensor_field(ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    ext_arr[p] = tensor_field[p]
        elif tensor_type == TensorType.Vector:
            @ti.kernel
            def get_tensor_field(ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d in ti.static(range(dim)):
                        ext_arr[p, d] = tensor_field[p][d]
        elif tensor_type == TensorType.Matrix:
            @ti.kernel
            def get_tensor_field(ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d1 in ti.static(range(dim)):
                        for d2 in ti.static(range(dim)):
                            ext_arr[p, d1, d2] = tensor_field[p][d1, d2]
        elif tensor_type == TensorType.TimedScalar:
            @ti.kernel
            def get_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    ext_arr[p] = tensor_field[s, p]
        elif tensor_type == TensorType.TimedVector:
            @ti.kernel
            def get_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d in ti.static(range(dim)):
                        ext_arr[p, d] = tensor_field[s, p][d]
        elif tensor_type == TensorType.TimedMatrix:
            @ti.kernel
            def get_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d1 in ti.static(range(dim)):
                        for d2 in ti.static(range(dim)):
                            ext_arr[p, d1, d2] = tensor_field[s, p][d1, d2]
        elif tensor_type == TensorType.ActBuffer:
            @ti.kernel
            def get_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_actuators):
                    ext_arr[p] = tensor_field[s, p]
        elif tensor_type == TensorType.VBuffer:
            @ti.kernel
            def get_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_actuators):
                    for d in ti.static(range(self.solver.dim)):
                        ext_arr[p, d] = tensor_field[s, p][d]

        return get_tensor_field
    
    def instantiate_set_tensor_field_kernel(self, tensor_field_name: str, tensor_type: TensorType):
        assert TensorType.is_member(tensor_type), f'Unrecognized tensor type {tensor_type}'
        tensor_field = recursive_getattr(self.solver, tensor_field_name)

        dim = self.solver.dim
        i_dtype = self.solver.i_dtype
        if tensor_type == TensorType.Scalar:
            @ti.kernel
            def set_tensor_field(ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    tensor_field[p] = ext_arr[p]
        elif tensor_type == TensorType.Vector:
            @ti.kernel
            def set_tensor_field(ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d in ti.static(range(dim)):
                        tensor_field[p][d] = ext_arr[p, d]
        elif tensor_type == TensorType.Matrix:
            @ti.kernel
            def set_tensor_field(ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d1 in ti.static(range(dim)):
                        for d2 in ti.static(range(dim)):
                            tensor_field[p][d1, d2] = ext_arr[p, d1, d2]
        elif tensor_type == TensorType.TimedScalar:
            @ti.kernel
            def set_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    tensor_field[s, p] = ext_arr[p]
        elif tensor_type == TensorType.TimedVector:
            @ti.kernel
            def set_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d in ti.static(range(dim)):
                        tensor_field[s, p][d] = ext_arr[p, d]
        elif tensor_type == TensorType.TimedMatrix:
            @ti.kernel
            def set_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_particles[None]):
                    for d1 in ti.static(range(dim)):
                        for d2 in ti.static(range(dim)):
                            tensor_field[s, p][d1, d2] = ext_arr[p, d1, d2]
        elif tensor_type == TensorType.ActBuffer:
            @ti.kernel
            def set_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_actuators):
                    tensor_field[s, p] = ext_arr[p]
        elif tensor_type == TensorType.VBuffer:
            @ti.kernel
            def set_tensor_field(s: i_dtype, ext_arr: ti.types.ndarray()):
                for p in range(self.solver.n_actuators):
                    for d in ti.static(range(self.solver.dim)):
                        tensor_field[s, p][d] = ext_arr[p, d]

        return set_tensor_field

    def get_semantic_occupancy(self, s=None):
        s = self.solver.get_cyclic_s(self.solver.current_s if s is None else s)

        semantic_occupancy = dict()
        for semantic_id, kernels in self.grid_semantic_kernel.items():
            self.grid_semantic[semantic_id].fill(0) # NOTE: reset
            
            if kernels['particle'] is not None:
                kernels['particle'](s)
            if kernels['static_component'] is not None:
                kernels['static_component'](s)
            
            # data = self.device.create_i_tensor(self.grid_semantic[semantic_id].shape)
            data = kernels['grid_access'][1]
            kernels['grid_access'][0](data)
            semantic_occupancy[semantic_id] = data # need to further do `.transpose(1, 0)[::-1]` otherwise flipped
            
        return semantic_occupancy

    def instantiate_grid_semantic_kernel(self, semantic_id, association):
        grid = self.grid_semantic[semantic_id]
        
        # Get update kernel based on particles
        particle_ids = association['particle_id'] # NOTE: must be of different variable name from `static_component_ids`
        if len(particle_ids) == 0:
            particle_update_kernel = None
        else:
            particle_id_to_semantic_id = ti.field(self.solver.i_dtype)
            ti.root.dynamic(ti.i, max(particle_ids) + 1).place(particle_id_to_semantic_id)
            for pid in particle_ids:
                particle_id_to_semantic_id[pid] = semantic_id
            @ti.kernel
            def fill_in_semantic_id():
                for p in range(self.solver.n_particles[None]):
                    particle_id = self.solver.particle_ids[p]
                    self.semantic_ids[p] = particle_id_to_semantic_id[particle_id]
            fill_in_semantic_id()
            del particle_id_to_semantic_id
            
            @ti.kernel
            def particle_update_kernel(s: self.solver.i_dtype):
                for p in range(self.solver.n_particles[None]):
                    if self.semantic_ids[p] == semantic_id:
                        base = ti.floor(self.solver.x[s, p] * self.solver.inv_dx - 0.5).cast(self.solver.i_dtype)
                        for offset in ti.static(ti.grouped(ti.ndrange(*((3, ) * self.solver.dim)))):
                            ti.atomic_max(grid[base + offset], 1) # mimic or
                                
        # Get update kernel based on static component
        static_component_ids = association['static_component_id']
        if len(static_component_ids) == 0:
            static_component_update_kernel = None
        else:
            @ti.kernel
            def static_component_update_kernel(s: self.solver.i_dtype):
                for I in ti.grouped(grid):
                    v_in = ti.Vector.zero(self.solver.f_dtype, self.solver.dim)
                    for id in ti.static(static_component_ids): # this can be slow if len(static_component_ids) is large
                        _, signed_dist = self.solver.grid_process_static[id](1., I, v_in, 1., s)
                        if signed_dist < 0:
                            ti.atomic_max(grid[I], 1)
        
        # Get kernel to access semantic grid
        grid_access_data = self.device.create_i_tensor(grid.shape)
        @ti.kernel
        def grid_access_kernel(ext_arr: ti.types.ndarray()):
            for I in ti.grouped(grid):
                ext_arr[I] = grid[I]

        return particle_update_kernel, static_component_update_kernel, (grid_access_kernel, grid_access_data)

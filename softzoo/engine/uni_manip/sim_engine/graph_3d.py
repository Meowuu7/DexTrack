import numpy as np
import taichi as ti
import os 
import wandb
import cma

# ti.init(arch=ti.gpu, )  # Try to run on GPU

# random_seed = 42
# ti_device_memory_fraction = 0.8
# ti.init(arch=ti.gpu, device_memory_fraction=ti_device_memory_fraction, random_seed=random_seed)

## structure of the unified manipulator in 2D ##
## currently --- kinematics only ##

dim = 2
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

RIGID_SPHERE = 0
RIGID_CUBE = 1
RIGID_SDF = 2

## get the graph structure ## 
## graph node: information about the graph node including the x-y sizes ##
## graph connectiveity information -- the A_t matrix where the graph size is ##
## graph connectivity information -- the A_t matrix where the graph size is ##



class GraphNode:
    def __init__(self, x_size, y_size, node_idx, node_type, node_name):
        self.x_size = x_size
        self.y_size = y_size
        self.node_idx = node_idx
        self.node_type = node_type
        self.node_name = node_name


        self.node_x = 0.0
        self.node_y = 0.0
    
    def set_node_xy(self, _x, _y):
        self.node_x = _x
        self.node_y = _y
        
class Graph:
    def __init__(self) -> None:
        # 
        # get the graph #
        self.graph_nodes = []
        self.graph_connectivity = []
        
        
        self.nn_timesteps = 10
        self.nn_links = 40
        
        self.dim = dim
        
        ## the sum of the row should be 1 ## ## not sure whether this ## ## ## [-1, 1] -> scal to [0, 1] -> normalize ##
        self.graph_A = ti.field(dtype=ti.f32, shape=(self.nn_links, self.nn_links))
        # 
        if dim == 2:
            self.graph_node_angular_acc = ti.Vector.field(n=dim - 1, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
            self.proj_graph_node_angular_acc = ti.Vector.field(n=dim - 1, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
        else:
            self.graph_node_angular_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
            self.proj_graph_node_angular_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
            
        # graph node angular accs 
        self.graph_node_linear_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps)) 
        
        # self.proj_gra
        ## graph linear acc ## 
        self.proj_graph_node_linear_acc = ti.Vector.field(n=dim, dtype=ti.f32, shape=(self.nn_links, self.nn_timesteps))
        
        # get hteaction simulation forward and the backward process, right? # 
        self.link_joint_pos = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        self.link_joint_dir = ti.Vector.field(n=self.dim, dtype=ti.f32, shape=(self.nn_links))
        
        self.get_directional_graph_mask()
        
        
        pass
    
    @staticmethod
    def get_graph_nn_links():
        return 40
    
    def get_directional_graph_mask(self, ):
        directional_graph_mask = np.ones((self.nn_links, self.nn_links), dtype=np.float32)
        for i_link in range(self.nn_links):
            directional_graph_mask[i_link, : i_link] = 0.0
        self.directional_graph_mask = directional_graph_mask
        
    
    ## we should have the graph structure ##
    def set_graph_connectivity_normalized_value(self, graph_connectivity_arr):
        #3 assum the graph connectivity arr has 
        graph_connectivity_arr = np.clip(graph_connectivity_arr, -1.0, 1.0)
        graph_connectivity_arr = (graph_connectivity_arr + 1.0) / 2.0
        graph_connectivity_arr = graph_connectivity_arr * self.directional_graph_mask
        ## should 
        graph_connectivity_arr = graph_connectivity_arr / np.clip(np.sum(graph_connectivity_arr, axis=0, keepdims=True), a_min=1e-6, a_max=None)
        
        self.graph_A.from_numpy(graph_connectivity_arr) 
        
        
    ## we should have the graph structure ##
    def set_graph_connectivity(self, graph_connectivity_arr):
        #3 assum the graph connectivity arr has 
        # graph_connectivity_arr = np.clip(graph_connectivity_arr, -1.0, 1.0)
        # graph_connectivity_arr = (graph_connectivity_arr + 1.0) / 2.0
        # graph_connectivity_arr = graph_connectivity_arr * self.directional_graph_mask
        # ## should 
        # graph_connectivity_arr = graph_connectivity_arr / np.clip(np.sum(graph_connectivity_arr, axis=0, keepdims=True), a_min=1e-6, a_max=None)
        
        self.graph_A.from_numpy(graph_connectivity_arr) 
        
    # def set_graph_link_p
    def set_link_joint_pos(self, link_joint_pos):
        self.link_joint_pos.from_numpy(link_joint_pos) ## for each link -> can be connected to the parent with one side and the child with another side ##

import taichi as ti
import numpy as np

ti.init(arch=ti.gpu)  # Try to run on GPU
quality = 1  # Use a larger value for higher-res simulations
n_particles, n_grid = 9000 * quality**2, 128 * quality

n_particles = n_particles // 3

dx, inv_dx = 1 / n_grid, float(n_grid)
dt = 1e-4 / quality # 
dt = 1e-5 / quality # 
dt = 1e-6 / quality # 

frame_dt = 2e-3
frame_dt = 1e-4

p_vol, p_rho = (dx * 0.5) ** 2, 1
p_mass = p_vol * p_rho # 
# E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
E, nu = 1e5, 0.2   # Young's modulus and Poisson's ratio
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
x = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # position
v = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)  # velocity
C = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # affine velocity field
F = ti.Matrix.field(2, 2, dtype=ti.f32, shape=n_particles)  # deformation gradient 
# deformation and the deformation gradient # from the deformation to the deformation gradient #
# [ [\partial f_x / \partial x, \partial f_x / \partial y], [ \partial f_y / \partial x, \partial f_y / \partial y ] ]
material = ti.field(dtype=int, shape=n_particles)  # material id 
is_actuated = ti.field(dtype=int, shape=n_particles)  # material id 
Jp = ti.field(dtype=ti.f32, shape=n_particles)  # plastic deformation
grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(n_grid, n_grid))  # grid node momentum/velocity
grid_m = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))  # grid node mass
grid_is_actuated = ti.field(dtype=ti.f32, shape=(n_grid, n_grid))  # grid node mass


## local 

# joint_pos = ti.Vector([0.4, 0.37], dt=ti.f32) ## jointacts ##
# change the deformation gradient for 
joint_pos_xy = [0.4, 0.37]

gravity = 50
gravity = 9.8
gravity = 0


actuated_boundary_thres = 1e-2
actuated_boundary_thres = 2e-2

add_stress_force = True
# add_stress_force = False

## the strategy of adding joint forces ##
# ti.f32 = ti.f32 ##

# ti.f32 = ti.f32

@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
        grid_is_actuated[i, j] = 0
    for p in x:  # Particle state update and scatter to grid (P2G)
        base = (x[p] * inv_dx - 0.5).cast(int) # base = (x[p] * inv_dx - 0.5).cast(int) 
        fx = x[p] * inv_dx - base.cast(ti.f32)
        # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        # F[p]: deformation gradient update
        F[p] = (ti.Matrix.identity(ti.f32, 2) + dt * C[p]) @ F[p] # update F[p] #
        # h: Hardening coefficient: snow gets harder when compressed # h: 
        h = ti.exp(10 * (1.0 - Jp[p]))
        if material[p] == 1:  # jelly, make it softer
            h = 0.3
        mu, la = mu_0 * h, lambda_0 * h
        if material[p] == 0:  # liquid
            mu = 0.0
        U, sig, V = ti.svd(F[p]) # # SVD and the SVD # # SVD and the SVD # # avoid zero eigenvalues # 
        # Avoid zero eigenvalues because of numerical errors # numerical errors #
        for d in ti.static(range(2)): # sig[d, d] -- decomposed singular value at dimension - d #
            sig[d, d] = ti.max(sig[d, d], 1e-6) 
        J = 1.0 # 
        ## udate J via plastici y # 
        for d in ti.static(range(2)): 
            new_sig = sig[d, d]
            if material[p] == 2:  # Snow
                new_sig = ti.min(ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3)  # Plasticity #
                # new_sig
            Jp[p] *= sig[d, d] / new_sig
            sig[d, d] = new_sig
            J *= new_sig
        if material[p] == 0: # reset deformation gradient to avoid numerical instability # plasticity #
            # Reset deformation gradient to avoid numerical instability
            F[p] = ti.Matrix.identity(ti.f32, 2) * ti.sqrt(J)
        elif material[p] == 2:
            # Reconstruct elastic deformation gradient after plasticity # plasticity # plasticity #
            F[p] = U @ sig @ V.transpose()
        # stress tensor -- a 2 x 2 tensor #
        # stress tensor -- a 2 x 2 tensor # 
        # 2 mu (F - U @ V^T) @ F^T + lambda J (J - 1) #
        # for non plasticity materials, stress = 2 * mu * (F - U @ V^T) @ F^T #
        # the stress tensor is related to the deformation gradient tensor #
        # the deformation gradient tensor is updated at every iteration # # every iteration #
        # to actuate the particle, a direct way is changing the deformation gradient #
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(ti.f32, 2) * la * J * (
            J - 1
        )
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = p_mass * C[p]
        if add_stress_force:
            affine = stress + affine
        # else: # else # # else #
            
        # Loop over 3x3 grid node neighborhood
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset.cast(ti.f32) - fx) * dx
            weight = w[i][0] * w[j][1]
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass # 
            if is_actuated[p] == 1:
                grid_is_actuated[base + offset] = 1
            
    ### grid operations ###
    for i, j in grid_m:
        if grid_m[i, j] > 0:  # No need for epsilon here
            grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
             
            # grid_v[i, j][1] -= dt * gravity # gravity # # momentum to velocity # 
            
            
            # grid_v[i, j] = grid_v[i, j] + dt * ti.Vector([50, -50])  # uniform linear force field 
            
            # joint_pos = ti.Vector([0.5, 0.47], dt=ti.f32)
            joint_pos = ti.Vector(joint_pos_xy, dt=ti.f32)
            grid_pos = ti.Vector([i, j], dt=ti.i32).cast(ti.f32) * dx
            # # joint_to_grid_p = grid_pos - ti.Vector([0.05, 0.05], dt=ti.f32)
            joint_to_grid_p = grid_pos - joint_pos
            # joint_to_grid_p[0] = joint_to_grid_p[0] - 0.05
            # joint_to_grid_p[1] = joint_to_grid_p[1] - 0.05
            # joint_to_grid_p = ti.Vector([i * dx - 0, j * dx - 0], dt=ti.f32)
            joint_to_grid_p_len = ti.sqrt(joint_to_grid_p[0] ** 2 + joint_to_grid_p[1] ** 2)
            
            ''' Rotational velocity field '''
            #### method 1 --> using the rotational velocity field ####
            # if joint_to_grid_p_len > 1e-5:
            #     joint_to_grid_p_dir = joint_to_grid_p / (joint_to_grid_p_len)
                
            #     # joint_to_grid_p_dir = joint_to_grid_p.normalized()
            #     vel_dir = ti.Vector([joint_to_grid_p_dir[1], -joint_to_grid_p_dir[0]])
            #     vel_len = joint_to_grid_p_len * 1
            #     vel = vel_dir * 100 * joint_to_grid_p_len
                
            #     # grid_v[i, j] = grid_v[i, j] + dt * vel
                
            #     # print(vel, dt) ##
            #     # grid_v[i, j] =  dt * vel #
                
            #     # grid_v[i, j] = grid_v[i, j]
            #     grid_v[i, j] = vel #  dt * vel
            #     # grid_v # grid_v[i, j] = vel ## add the grid_v ##
                
            #     # use the zero grid vs ##
                
            #     # dot_grid_v_vel_dir = grid_v[i, j][0]  * vel_dir[0] + grid_v[i, j][1] * vel_dir[1]
            #     # proj_grid_v =  dot_grid_v_vel_dir * vel_dir
            #     # # print(vel, proj_grid_v, dt)
            #     # grid_v[i, j] = proj_grid_v +  dt * vel
            #     # ## grid_v[i, j] = proj_grid_v
            # else:
            #     grid_v[i, j] = ti.Vector([0, 0])
            #### method 1 --> using the rotational velocity field ####
            ''' Rotational velocity field '''    
            
            ''' Joint constraints ''' # joint # joint and the velocity field ## 
            #### method 2 --> using the rotational velocity field ####
            if joint_to_grid_p_len < 1e-2 or grid_is_actuated[i, j] == 1: ## 
                if joint_to_grid_p_len > 1e-5 and grid_is_actuated[i, j] == 1:
                    joint_to_grid_p_dir = joint_to_grid_p / (joint_to_grid_p_len)
                    vel_dir = ti.Vector([joint_to_grid_p_dir[1], -joint_to_grid_p_dir[0]])
                    vel_len = joint_to_grid_p_len * 1
                    vel = vel_dir * 100 * joint_to_grid_p_len
                    grid_v[i, j] = vel
                else:
                    grid_v[i, j] = ti.Vector([0, 0])
            ''' Joint constraints '''
            
            
            # grid ## grid and the grid ## 
            if i < 3 and grid_v[i, j][0] < 0: # boundary conditions #
                grid_v[i, j][0] = 0  # Boundary conditions # 
            if i > n_grid - 3 and grid_v[i, j][0] > 0: # # 
                grid_v[i, j][0] = 0
            if j < 3 and grid_v[i, j][1] < 0: # ##
                grid_v[i, j][1] = 0
            if j > n_grid - 3 and grid_v[i, j][1] > 0:
                grid_v[i, j][1] = 0 # G2P --- G2P # ## G2P #

    for p in x:  # grid to particle (G2P) # grid to particle (G2P) # # G2P -- G2P ##
        base = (x[p] * inv_dx - 0.5).cast(int)
        base[0] = ti.max(0, ti.min(base[0], n_grid - 1))
        base[1] = ti.max(0, ti.min(base[1], n_grid - 1))
        fx = x[p] * inv_dx - base.cast(ti.f32)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(ti.f32, 2)
        new_C = ti.Matrix.zero(ti.f32, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)): # 
            # loop over 3x3 grid node neighborhood # 
            dpos = ti.Vector([i, j]).cast(ti.f32) - fx
            g_v = grid_v[base + ti.Vector([i, j])]
            weight = w[i][0] * w[j][1]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos) ## 
        
        # new_v, new_C # -- new_v new_C -- # add to the particle # # vnn #
        # 
        
        v[p], C[p] = new_v, new_C # 
        x[p] += dt * v[p]  # advection 
        # print(f"x[p]: {x[p]}, v[p]: {v[p]}, p: {p}") #

# n particles # # #
# n particles # # # 
# # 

group_size = n_particles #  n_particles // 3 # n_particles // 3 #

## group size = number of particles ##
##

@ti.kernel
def initialize():
    for i in range(n_particles):
        rnd_x = ti.random()
        rnd_y = ti.random()
        # x[i] = [
        #     # ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size + 1),
        #     # ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size + 1),
        #     ti.random() * 0.2 + 0.3 + 0.10 * ( 1),
        #     ti.random() * 0.2 + 0.05 + 0.32 * ( 1),
        # ]
        x[i] = [
            # ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size + 1),
            # ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size + 1),
            rnd_x * 0.2 + 0.3 + 0.10 * ( 1),
            rnd_y * 0.2 + 0.05 + 0.32 * ( 1), # 
        ]
        
        if rnd_x < actuated_boundary_thres and rnd_y < actuated_boundary_thres:
            is_actuated[i] = 1
        elif rnd_x > 1 - actuated_boundary_thres and rnd_y > 1 - actuated_boundary_thres:
            is_actuated[i] = 1
        elif rnd_x < actuated_boundary_thres and rnd_y > 1 - actuated_boundary_thres:
            is_actuated[i] = 1
        elif rnd_x > 1 - actuated_boundary_thres and rnd_y < actuated_boundary_thres:
            is_actuated[i] = 1 # whether the particle can be actuated #
        
        material[i] = 1 #  i // group_size  # 0: fluid 1: jelly 2: snow
        
        # material[i] = 1 #
        ''' method 1 --- the zero initial velocity field '''
        v[i] = ti.Matrix([0, 0])
        ''' method 1 --- the zero initial velocity field '''
        
        # v[i] = ti.Mate
        # v[i] = 
        # v[i] =
        # x[i]
        
        ''' method 2 -- the rotational initial velocity field '''
        cur_particle_pos = x[i] ## the i-th particle's position ##
        ## from the particle pos to the particle's position ##
        ## cur_particle_pos ##
        joint_pos = ti.Vector(joint_pos_xy, dt=ti.f32)
        joint_to_particle_p = cur_particle_pos - joint_pos ## joint pos to the particle pos ## 
        joint_to_particle_p_len = ti.sqrt(joint_to_particle_p[0] ** 2 + joint_to_particle_p[1] ** 2)
        if joint_to_particle_p_len > 1e-5:
            joint_to_partilce_dir = joint_to_particle_p.normalized()
            vel_dir = ti.Vector([joint_to_partilce_dir[1], -joint_to_partilce_dir[0]])
            ## get the velocity direction ##
            vel = vel_dir * 100 * joint_to_particle_p_len
            v[i] = vel 
        ''' method 2 -- the rotational initial velocity field ''' ## 
        
        
        F[i] = ti.Matrix([[1, 0], [0, 1]])
        Jp[i] = 1


def main():
    initialize() ## initialize -- initialization ##
    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(frame_dt // dt)):
            substep()
        gui.circles(
            x.to_numpy(),
            radius=1.5, # 
            palette=[0x068587, 0xED553B, 0xEEEEF0],
            palette_indices=material,
        )
        ### circles ### # circles #
        gui.circles(
            np.array([joint_pos_xy], dtype=np.float32),
            radius=7, #
            # palette=[0x068587, 0xED553B, 0xEEEEF0],
            # palette_indices=material,
        )
        # Change to gui.show(f'{frame:06d}.png') to write images to disk # show the image in the gui #
        gui.show() # change the gui show #
        ## transferred to the gird space ##


# # fro mthe grid velocities to the control information in the reduced dimensional systems ### ? # How to control that in them? # how to control ## control a soft robot # in the control a soft robot # soft robot; soft robot # # # # how to control that in them # # from # #
# python softzoo/engine/mpm99_ours.py 
# controlling an object ## 
# transforming the control signal from one space to another space ## 
# from one soft scenario to antoher one ? # 
## from one soft scenario to antoher ## # from one soft # actuate the soft # 
# from one soft #  from one soft scenariors to another # 
## disspation of the energy ## # disspation # 
# each grid can be actuated ## each grid can be actuated ##
# should add an actuation force on the grid ### # apt # should add an actuation ## should be a good ### should be a good res and the res and the res and the rest ###
## should add an actuation ## should be a good ### should be a good res and the res and the res and the rest ## 

# python softzoo/engine/mpm99_ours_jointact.py
if __name__ == "__main__":
    main()

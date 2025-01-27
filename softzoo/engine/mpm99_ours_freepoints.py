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

reference_xs = ti.Vector.field(2, dtype=ti.f32, shape=n_particles)
goal_reaching_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
delta_vel_energy_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
delta_rot_acc_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
tot_loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)


## ## the center of the 
## rotation quaternions ##
## translation vectors ##
## constraints between them ##
# unit vector -> a rot quat #

dim = 2

nn_frames = 10

## 
## [ [ cos(a), sin(a) ]  [ -sin(a), cos(a) ] ] ##


# rot_quats, rot_vec, delta_rot_vec, rot_mtx, trans_ts, delta_trans_ts, xs
rot_quats = ti.Vector.field(dim + 1, dtype=ti.f32, shape=(nn_frames, ), needs_grad=True)

## TODO: initialize the rot_vec, trans_ts, xs ## 
rot_vec = ti.Vector.field(dim, dtype=ti.f32, shape=(nn_frames, ), needs_grad=True) ## get nn_frames ## 
delta_rot_vec = ti.Vector.field(dim, dtype=ti.f32, shape=(nn_frames, ), needs_grad=True) ## get the delta rot vec ##
rot_mtx = ti.Matrix.field(dim, dim, dtype=ti.f32, shape=(nn_frames, ), needs_grad=True)
trans_ts = ti.Vector.field(dim, dtype=ti.f32, shape=(nn_frames, ), needs_grad=True) 
delta_trans_ts = ti.Vector.field(dim, dtype=ti.f32, shape=(nn_frames, ), needs_grad=True) 
xs = ti.Vector.field(2, dtype=ti.f32, shape=(nn_frames, n_particles), needs_grad=True)
# vs = ti.Vector.fie
## trans ts ## 

# j


## upate the partciels positions ## 



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

@ti.kernel
def calculate_rot_matrices(s: ti.i32):
    cur_delta_rot_vec = delta_rot_vec[s] ## 2-dim taichi vectors # 
    prev_rot_vec = rot_vec[s - 1] ## 2-dim taichi vector ## 
    cur_rot_vec = prev_rot_vec + cur_delta_rot_vec ## cur delt rot vec ##
    len_cur_rot_vec = cur_rot_vec.norm() ## vector ##
    if len_cur_rot_vec > 1e-5:
        cur_rot_vec = cur_rot_vec.normalized()
    rot_vec[s] = cur_rot_vec
    cur_rot_mtx = ti.Matrix([ [ cur_rot_vec[0], -cur_rot_vec[1] ], [ cur_rot_vec[1], cur_rot_vec[0] ] ], dt=ti.f32) ## the rotation matrix ## 
    trans_ts[s] = trans_ts[s - 1] + delta_trans_ts[s]
    rot_mtx[s] = cur_rot_mtx ## get the rot mtx ## 
    
    # rot @ jt + trans = jt #
    joint_pos = ti.Vector(joint_pos_xy, dt=ti.f32)
    trans_ts[s] = (ti.Matrix.identity(dt=ti.f32, n=2) - cur_rot_mtx) @ joint_pos
    


@ti.kernel
def substep_lag(s: ti.i32): ## 
    for p in range(n_particles):
        init_p_pos = xs[0, p] ##
        cur_rot_mtx = rot_mtx[s]
        cur_trans_t = trans_ts[s]
        # print(f"cur_rot_mtx: {cur_rot_mtx}, init_p_pos: {init_p_pos}, cur_trans_t: {cur_trans_t}")
        cur_p_pos = cur_rot_mtx @ init_p_pos + cur_trans_t ## 3 - dim 
        xs[s, p] = cur_p_pos
        # 
        

        
        
    

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

## substep -> give the delta rotation at this step, and the delta; the rotation matrix at the next step is \Delta R R and the translation at the next step is \Delta R \times t + \Delta t ##
## energy function -> 1) constraints for the rotation matrix and the transformation matrix -> the joint constraints; 2) minimizing the kinematics energy 0.5 * \Delta v; and 3) reaching the goal
## energy function 

## angular velocity ##

## how to get the angular velocity from rotation matrices at two timesteps? ##
## rotation parameterization -> the quaternion ##
## changing rate of the rotation matrix --> angular velocity ##
## changing rate of the translational vector --> linear velocity ##
## or the coverage of the particle? ##
## get the 

## TODO: remember to reset the goal reaching loss to zero ahead of each optimization iteration ##
## TODO: change x to a time varying field --> x is of the shape [tot_timesteps, n_particles, 2] ##
## TODO: joint constraint lloss ## 


@ti.kernel
def clear_losses():
    goal_reaching_loss.fill(0.)
    # if goal_reaching_loss.grad != None:
    # try:
    goal_reaching_loss.grad.fill(0.)
    # except:
    #     pass
    delta_vel_energy_loss.fill(0.)
    # if delta_vel_energy_loss.grad is not None:
    delta_vel_energy_loss.grad.fill(0.) 
    delta_rot_acc_loss.fill(0.)
    # if delta_rot_acc_loss.grad is not None:
    delta_rot_acc_loss.grad.fill(0.) ## get the acc rot loss ##
    ## delta vel energy loss ##
    tot_loss.fill(0.)
    # if tot_loss.grad is not None:
    tot_loss.grad.fill(0.)
    

@ti.kernel
def clear_grad():
    for fr in range(nn_frames):
        rot_quats.grad[fr] = [0, 0, 0]
        rot_vec.grad[fr] = [0, 0]
        delta_rot_vec.grad[fr] = [0, 0]
        rot_mtx.grad[fr] = [ [0, 0], [0, 0] ]
        trans_ts.grad[fr] = [0, 0]
        delta_trans_ts[fr] = [0, 0]
        
        
    

@ti.kernel
def get_diff_particle_pos_loss(s: ti.i32):
    # tot_particle_pos_diff = 0.0
    for p in x: # p in the x #
        cur_particle_pos = xs[s, p] # 
        ref_particle_pos = reference_xs[p]
        diff_particle_pos = (cur_particle_pos - ref_particle_pos).norm()
        # tot_particle_pos_diff = tot_particle_pos_diff + diff_particle_pos
        goal_reaching_loss[None] += diff_particle_pos
        tot_loss[None] += diff_particle_pos
    # for p in the x domain # 


## get the kinematics energies 
@ti.kernel
def get_diff_particle_v_reg(s: ti.i32):
    for p in x: ## 
        cur_p_v = v[s, p]
        prev_p_v = v[s - 1, p]
        # 
        finite_diff_vs = (cur_p_v - prev_p_v) / dt # piece dt or frame dt? particle_p_v and them 
        cur_finite_diff_v_energy = 0.5 * (finite_diff_vs ** 2)
        delta_vel_energy_loss[None] += cur_finite_diff_v_energy
        
@ti.kernel
def get_acc_delta_rot_vec(s: ti.i32):
    cur_delta_rot_vec = delta_rot_vec[s]
    prev_delta_rot_vec = delta_rot_vec[s - 1] ##3 get theprev delta rot vec ## 
    diff_prev_to_cur_rot_vec = cur_delta_rot_vec - prev_delta_rot_vec ## get the delta rot vectors ## 
    # diff_prev_to_cur_rot = ti.sum(diff_prev_to_cur_rot_vec ** 2)
    diff_prev_to_cur_rot = diff_prev_to_cur_rot_vec.norm() ** 2
    delta_vel_energy_loss[None] += diff_prev_to_cur_rot ## differences between energy losses ##
    
    tot_loss[None] += diff_prev_to_cur_rot
    
    


### get loss ###
def get_loss(s):
    # changing rate of the rotation matrix #
    # goal -> minimize the difference between the target particle positions and the current particle positions #
    get_diff_particle_pos_loss(s)
    # get_diff_particle_v_reg(s=s) ## get 
    if s  > 0:
        get_acc_delta_rot_vec(s=s)
    ## get the total loss ##
    # 
    # 

## optimize #
## get the joints ##
# the joint_pos and the constraints -- a two link system ? ##

## get the loss and get the loss ##
## then update the rotation velociities ## 
## then update the rotation 
 
    
def optimize_iter(vis):
    
    
    
    clear_losses()
    clear_grad()
    xs.grad.fill(0)
    
    for s in range(nn_frames): ## get the
        # print(f"s: {s}")
        if s > 0:
            calculate_rot_matrices(s)


        substep_lag(s)
        
        if vis:
            gui.circles(
                reference_xs.to_numpy(),
                radius=1.5, # 
                # palette=[0x068587, 0xED553B, 0xEEEEF0],
                # palette_indices=material,
            )
            gui.circles(
                xs.to_numpy()[s],
                radius=1.5, # 
                palette=[0x068587, 0xED553B, 0xEEEEF0],
                palette_indices=material,
            )
            ## accumulated errors 
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

        # get_loss(s=s) ## get_loss ##
        
        get_diff_particle_pos_loss(s=s)
        # if s  > 0:
        #     get_acc_delta_rot_vec(s=s)
        
    # for s in range(nn_frames):
    #     print(f"delta_rot_vec.grad[s]: {delta_rot_vec.grad[s]}")
        # delta_rot_vec[s] = delta_rot_vec[s] - upd_lr * delta_rot_vec.grad[s] ## get the grad ## 
        ## get the rot _vec
        
    
    print(f"tot_loss: {tot_loss}")
    # tot_loss.backward() ## get loss ##
    tot_loss.grad[None] =  1.0 ## get the loss ##
    
    for s in range(nn_frames - 1, -1, -1):
        
        # if s > 0:
        #     get_acc_delta_rot_vec.grad(s)
        # print(f"[1] delta_rot_vec.grad[s]: {delta_rot_vec.grad[s]}")
        
        get_diff_particle_pos_loss.grad(s)
        # print(f"[2] delta_rot_vec.grad[s]: {delta_rot_vec.grad[s]}")
        
        # get_loss.grad(s=s) ## get the grad ##
        substep_lag.grad(s)
        # print(f"[3] delta_rot_vec.grad[s]: {delta_rot_vec.grad[s]}")
        if s > 0:
            calculate_rot_matrices.grad(s)
        # print(f"[4] delta_rot_vec.grad[s]: {delta_rot_vec.grad[s]}")
    
    upd_lr = 0.001
    for s in range(nn_frames):
        # print(f"delta_rot_vec.grad[s]: {delta_rot_vec.grad[s]}, delta_trans_ts.grad[s]: {delta_trans_ts.grad[s]}")
        delta_rot_vec[s] = delta_rot_vec[s] - upd_lr * delta_rot_vec.grad[s] ## get the grad ## 
        ## get the rot _vec
        # trans_ts[s] = trans_ts[s] - upd_lr * transts
        delta_trans_ts[s] = delta_trans_ts[s] - upd_lr * delta_trans_ts.grad[s]



def run_and_vis_main(vis):
    ## init transformation quantities ##
    init_transformation_quantities()
    initialize()
    
    while True: # optimize iter ## ## optimize the iter ## ## 
        optimize_iter(vis)
    # the optimize_iter #
    



group_size = n_particles #  n_particles // 3 # n_particles // 3 #

## group size = number of particles ##

# @ti.kernel

### init the two-dimensional transformations for each frame ###
@ti.kernel
def init_transformation_quantities():
    
    for fr in range(nn_frames):
        rot_vec[fr] = [1, 0] ## 
        delta_rot_vec[fr] = [0, 0]
        rot_mtx[fr] = [ [1, 0], [0, 1] ]
        trans_ts[fr] = [0, 0] ## 
        delta_trans_ts[fr] = [0, 0]
        
        rot_vec.grad[fr] = [0, 0]
        delta_rot_vec.grad[fr] = [0, 0]
        rot_mtx.grad[fr] = [ [0, 0], [0, 0] ]
        trans_ts.grad[fr] = [0, 0]
        delta_trans_ts.grad[fr] = [0, 0]

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
        xs[0, i] = [ # 
            # ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size + 1),
            # ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size + 1),
            rnd_x * 0.2 + 0.3 + 0.10 * ( 1),
            rnd_y * 0.2 + 0.05 + 0.32 * ( 1), # 
        ]
        
        
        # joint pos = 
        joint_pos = ti.Vector(joint_pos_xy, dt=ti.f32)
        # get the joint pos
        # tar_rot_angle = ti.math.pi / 2.0
        tar_rot_angle = -1. * ti.math.pi  # -  ti.math.pi / 2.0
        tar_cos = ti.math.cos(tar_rot_angle)
        tar_sin = ti.math.sin(tar_rot_angle)
        tar_rot_mtx = ti.Matrix([[tar_cos, -tar_sin], [tar_sin, tar_cos]], dt=ti.f32)
        # rot @ jt + trans = jt --> trans = (I - rot) @ jt
        tar_trans = (ti.Matrix.identity(dt=ti.f32, n=2) - tar_rot_mtx) @ joint_pos
        cur_reference_x = tar_rot_mtx @ x[i] + tar_trans
        reference_xs[i] = cur_reference_x ## get the reference xs here ##
        
    
        if rnd_x < actuated_boundary_thres and rnd_y < actuated_boundary_thres:
            is_actuated[i] = 1
        elif rnd_x > 1 - actuated_boundary_thres and rnd_y > 1 - actuated_boundary_thres:
            is_actuated[i] = 1
        elif rnd_x < actuated_boundary_thres and rnd_y > 1 - actuated_boundary_thres:
            is_actuated[i] = 1
        elif rnd_x > 1 - actuated_boundary_thres and rnd_y < actuated_boundary_thres:
            is_actuated[i] = 1 # whether the particle can be actuated #
        
        material[i] = 1 #  i // group_size  # 0: fluid 1: jelly 2: snow # 
        
        # material[i] = 1 #
        ''' method 1 --- the zero initial velocity field '''
        v[i] = ti.Matrix([0, 0])
        ''' method 1 --- the zero initial velocity field '''
        
        
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
    # ti.GUI --- get the taichi gui here #
    ## if we use the taichi gui ## ## 
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

## TODO: save the checkpoints ## 
# TODO: hard joint constraints and the soft joint constraints ? ##


# hard joint constraints for link motions #
# task # 
# per-point densely reaching the goal positions #
# reaching a sparse goal #
# 



# python softzoo/engine/mpm99_ours_freepoints.py
if __name__ == "__main__":
    # main()

    # joints as the hard constraints while 
    vis = True    
    # vis = False
    
    if vis:
        gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    
    
    run_and_vis_main(vis=vis)
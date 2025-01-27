



export goal_cond=True


# export cuda_idx=7

export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-08-48-29/nn/last_Humanoid_ep_1850_rew_443.01654.pth

export w_obj_ornt=False

# export w_obj_ornt=True

export goal_cond=False

export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-15-58-08/nn/last_Humanoid_ep_850_rew_69.35681.pth
export checkpoint=runs/Humanoid_03-16-04-34/nn/last_Humanoid_ep_10000_rew_195.45563.pth
# 
export checkpoint=runs/Humanoid_04-04-57-39/nn/last_Humanoid_ep_2850_rew_79.1641.pth
export checkpoint=runs/Humanoid_04-05-43-25/nn/Humanoid.pth
# export 

export checkpoint="runs/grasp_ori_grab_s2_train_lift_objdensity_500_07-15-05-31/nn/last_grasp_ori_grab_s2_train_lift_objdensity_500_ep_1400_rew_315.84476.pth"

export object_name='ori_grab_s2_train_lift'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'

# export object_name=''
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'

# export object_name=''


export exp_dir='.'
# export exp_dir='/cephfs/xueyi/exp/IsaacGymEnvs/isaacgymenvs'

export tag=grasp_${object_name}

# full_experiment_name, name #
export train_name=grasp_${object_name}
export full_experiment_name=${train_name}


export obs_type='pure_state'

export use_canonical_state=True

export use_fingertips=False

export use_fingertips=True

export glb_trans_vel_scale=10
export glb_rot_vel_scale=5

export glb_trans_vel_scale=1
export glb_rot_vel_scale=1


export checkpoint='/home/xueyi/IsaacGymEnvs/assets/retar_data/grasp_ori_grab_s2_train_lift_objdensity_500_usetip_True.pth'


export rigid_obj_density=500

# full_experiment_name, name #
export train_name=grasp_${object_name}_objdensity_${rigid_obj_density}_usetip_${use_fingertips}_glbtrans${glb_trans_vel_scale}_glbrot${glb_rot_vel_scale}
export full_experiment_name=${train_name}
# 






# CUDA_VISIBLE_DEVICES=4 python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False force_render=False headless=True   task.env.numEnvs=10240 train.params.config.minibatch_size=10240  task.env.useRelativeControl=True train.params.config.max_epochs=10000  task.env.goal_cond=True task.env.w_obj_ornt=True checkpoint=runs/Humanoid_03-08-30-37/nn/Humanoid.pth

# bash scripts/run_grasp_capturevideo.sh

##### train using the prev_state control mode #####
python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=True force_render=True headless=False   task.env.numEnvs=4096 train.params.config.minibatch_size=4096  task.env.useRelativeControl=True train.params.config.max_epochs=100000  task.env.w_obj_ornt=${w_obj_ornt}   task.env.goal_cond=${goal_cond}  checkpoint=${checkpoint} task.env.object_name=${object_name} tag=${tag} exp_dir=${exp_dir}  train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name} task.env.observationType=${obs_type} task.env.use_canonical_state=${use_canonical_state}  task.env.glb_trans_vel_scale=${glb_trans_vel_scale} task.env.glb_rot_vel_scale=${glb_rot_vel_scale}   task.env.use_fingertips=${use_fingertips}  task.env.rigid_obj_density=${rigid_obj_density} 

# task.env.w_obj_ornt=True  

# task.env.goal_cond=True 




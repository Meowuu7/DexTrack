export goal_cond=True


export cuda_idx=7

# export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-08-48-29/nn/last_Humanoid_ep_1850_rew_443.01654.pth
# export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-15-58-08/nn/last_Humanoid_ep_850_rew_69.35681.pth

export checkpoint=runs/Humanoid_03-19-11-03/nn/last_Humanoid_ep_3200_rew_86.94377.pth



export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'

#### /home/xueyi/IsaacGymEnvs/assets/datasetv4.1/sem/taco_20231104_016 # 
export object_name='taco_20231104_016'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'


export checkpoint=''



export rigid_obj_density=100


export rigid_obj_density=500



export object_name=car

export tag=grasp_${object_name}




export checkpoint='runs/grasp_car_objdensity_100_07-09-19-00/nn/grasp_car_objdensity_100.pth'

export object_name=''
export checkpoint='runs/grasp__objdensity_500_08-05-44-50/nn/grasp__objdensity_500.pth'



export object_name=''
export checkpoint=''
export real_obj_name='phone_call_1'
export real_obj_name='apple_lift'
export real_obj_name='hammer_use_1'
export real_obj_name='taco_20231104_203'

# full_experiment_name, name #
export train_name=grasp_${object_name}_${real_obj_name}_objdensity_${rigid_obj_density}
export full_experiment_name=${train_name}

export exp_dir='.'



# bash scripts/run_grasp_shadow_capturevideo.sh
# global movements of the hand and the object # 
# track the using the kinematics bias, control the hand #

##### train using the prev_state control mode #####
# CUDA_VISIBLE_DEVICES=${cuda_idx} 

python train.py task=ShadowHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=True  force_render=True headless=False    task.env.numEnvs=1024 train.params.config.minibatch_size=1024  train.params.config.max_epochs=10000  checkpoint=${checkpoint} tag=${tag}  train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name}  exp_dir=${exp_dir}
#  task.env.useRelativeControl=True
#  task.env.object_name=${object_name} 

# task.env.rigid_obj_density=${rigid_obj_density}

# task.env.w_obj_ornt=True 
# task.env.goal_cond=True 

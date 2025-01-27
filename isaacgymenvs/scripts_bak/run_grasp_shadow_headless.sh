export goal_cond=True


export cuda_idx=3

# export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-08-48-29/nn/last_Humanoid_ep_1850_rew_443.01654.pth
# export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-15-58-08/nn/last_Humanoid_ep_850_rew_69.35681.pth

export checkpoint=runs/Humanoid_03-19-11-03/nn/last_Humanoid_ep_3200_rew_86.94377.pth



export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'

export object_name=''
#### /home/xueyi/IsaacGymEnvs/assets/datasetv4.1/sem/taco_20231104_016 # 
# export object_name='taco_20231104_016'
# # export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'


export checkpoint=''



export rigid_obj_density=100

export rigid_obj_density=500


# export object_name=car

export tag=grasp_${object_name}


# full_experiment_name, name #
export train_name=grasp_${object_name}_objdensity_${rigid_obj_density}
export full_experiment_name=${train_name}


# export object_name=""


# bash scripts/run_grasp_shadow_headless.sh


##### train using the prev_state control mode #####
CUDA_VISIBLE_DEVICES=${cuda_idx} python train.py task=ShadowHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False  force_render=False headless=True    task.env.numEnvs=10240 train.params.config.minibatch_size=10240  train.params.config.max_epochs=10000  checkpoint=${checkpoint} tag=${tag}  train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name}   task.env.object_name=${object_name}
#  task.env.useRelativeControl=True
#  task.env.object_name=${object_name} 

# task.env.rigid_obj_density=${rigid_obj_density}

# task.env.w_obj_ornt=True 
# task.env.goal_cond=True 

export goal_cond=True



export numEnvs=4096
export minibatch_size=4096


export cuda_idx=6

# export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-08-48-29/nn/last_Humanoid_ep_1850_rew_443.01654.pth
# export checkpoint=/home/xueyi/IsaacGymEnvs/isaacgymenvs/runs/Humanoid_03-15-58-08/nn/last_Humanoid_ep_850_rew_69.35681.pth

export checkpoint=runs/Humanoid_03-19-11-03/nn/last_Humanoid_ep_3200_rew_86.94377.pth



export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'

# #### /home/xueyi/IsaacGymEnvs/assets/datasetv4.1/sem/taco_20231104_016 # 
# export object_name='taco_20231104_016'
# # export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'




export checkpoint=''



export rigid_obj_density=500
# export rigid_obj_density=100

export use_fingertips=False

export use_fingertips=True


export tag=grasp_${object_name}





# observationType
export obs_type='pure_state'


#### canonical states ####
export use_canonical_state=True
export use_canonical_state=False

export additiona_tag=""
export additiona_tag="new"
export additiona_tag="ncanon"
export use_relative_control=True



export use_relative_bias_control=False
export use_relative_bias_control=True
export use_relative_control=False
export use_canonical_state=True
export additiona_tag="biasctl"

#### translation and rotation scales ####
export glb_trans_vel_scale=10
export glb_rot_vel_scale=5

export glb_trans_vel_scale=1
export glb_rot_vel_scale=1
#### translation and rotation scales ####


# full_experiment_name, name #
export train_name=grasp_${object_name}_objdensity_${rigid_obj_density}_usetip_${use_fingertips}_glbtrans${glb_trans_vel_scale}_glbrot${glb_rot_vel_scale}_${additiona_tag}
export full_experiment_name=${train_name}
export test=False # True

# test = False #

#### Test setting ####
# export checkpoint='runs/grasp_ori_grab_s2_train_lift_objdensity_500_usetip_True_08-01-26-47/nn/grasp_ori_grab_s2_train_lift_objdensity_500_usetip_True.pth'
# export test=True
#### Test setting ####

export cuda_idx=1


## TODO: xxx ###

# bash scripts/run_grasp_headless.sh


##### train using the prev_state control mode #####
CUDA_VISIBLE_DEVICES=${cuda_idx} python train.py task=AllegroHandGrasp train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=False  force_render=False headless=True    task.env.numEnvs=${numEnvs} train.params.config.minibatch_size=${minibatch_size}  task.env.useRelativeControl=${use_relative_control} train.params.config.max_epochs=10000  checkpoint=${checkpoint} task.env.object_name=${object_name} tag=${tag}  train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name} task.env.rigid_obj_density=${rigid_obj_density} task.env.observationType=${obs_type} task.env.use_canonical_state=${use_canonical_state} task.env.use_fingertips=${use_fingertips} task.env.glb_trans_vel_scale=${glb_trans_vel_scale} task.env.glb_rot_vel_scale=${glb_rot_vel_scale}   test=${test}  task.env.use_relative_bias_control=${use_relative_bias_control}

# task.env.w_obj_ornt=True 
# task.env.goal_cond=True 

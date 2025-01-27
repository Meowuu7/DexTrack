
export numEnvs=1024
export minibatch_size=1024

export numEnvs=4096
export minibatch_size=4096


export mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy'


export checkpoint=runs/Humanoid_03-09-34-46/nn/Humanoid.pth 

export checkpoint=runs/Humanoid_03-16-01-13/nn/Humanoid.pth

export checkpoint=runs/Humanoid_03-16-01-13/nn/Humanoid.pth

export checkpoint=runs/Humanoid_03-18-43-48/nn/Humanoid.pth

export checkpoint=runs/Humanoid_03-18-43-48/nn/last_Humanoid_ep_7950_rew_171.98253.pth
# runs/Humanoid_03-18-43-48/nn/last_Humanoid_ep_7950_rew_171.98253.pth
export checkpoint=/home/xueyi/IsaacGymEnvs/assets/retar_data/last_Humanoid_ep_4400_rew_160.85417.pth
export checkpoint=/home/xueyi/IsaacGymEnvs/assets/retar_data/last_Humanoid_ep_5000_rew_140.40802.pth

export checkpoint=runs/Humanoid_04-14-59-01/nn/Humanoid.pth



### Object task setting ###
export object_name='ori_grab_s2_headphones_lift'
export mocap_sv_info_fn='../assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift_pkretar.npy'
### Object task setting ###

### Object task setting ###
export object_name='ori_grab_s2_flashlight_on_2'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_flashlight_on_2.npy'
### Object task setting ###

### Object task setting ###
export object_name='ori_grab_s2_hand_inspect_1'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_hand_inspect_1.npy'
### Object task setting ###


### Object task setting ###
export object_name='ori_grab_s2_headphones_lift'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_headphones_lift.npy'
export checkpoint=../assets/retar_data/last_Humanoid_ep_10000_rew__167.42_.pth
### Object task setting ###

### Object task setting ###
export object_name='ori_grab_s2_flashlight_on_2'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_flashlight_on_2.npy'
export checkpoint=../assets/retar_data/last_Humanoid_flashlight_on_2.pth
### Object task setting ###

### Object task setting ###
export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'
export checkpoint=../assets/retar_data/tracking_ori_grab_s2_train_lift.pth
### Object task setting ###


### Object task setting ###
export object_name='ori_grab_s2_train_lift'
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'
export checkpoint='runs/tracking_ori_grab_s2_train_lift_05-13-20-04/nn/last_tracking_ori_grab_s2_train_lift_ep_10000_rew__106.25_.pth'
export checkpoint='runs/tracking_ori_grab_s2_train_lift_06-07-25-07/nn/tracking_ori_grab_s2_train_lift.pth'
export checkpoint=''
export goal_cond=True
export dt=0.0166
export kinematics_only=True
export use_kinematics_bias=True
### Object task setting ###



# export kinematics_only=True
export object_name='ori_grab_s2_train_lift'
# export checkpoint='runs/tracking_ori_grab_s2_train_lift_07-10-24-57/nn/last_tracking_ori_grab_s2_train_lift_ep_1900_rew_138.87497.pth'
export checkpoint=''
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'
export goal_cond=False # 
export test=False # True
# export dt=0.00833  
export dt=0.0166 # 1 / 60 s 
export use_kinematics_bias=True
export kinematics_only=False





export use_relative_control=True
export use_relative_control=False


####  
export use_kinematics_bias=False
export kinematics_only=False
export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'




# export mocap_sv_info_fn='/home/xueyi/IsaacGymEnvs/assets/retar_data/passive_active_info_ori_grab_s2_train_lift.npy'

# export object_name='ori_grab_s2_train_lift'
# export mocap_sv_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK/data/passive_active_info_ori_grab_s2_train_lift.npy'


# full_experiment_name #
export train_name=tracking_${object_name}
export full_experiment_name=${train_name}


export exp_dir='.'
export tag=tracking_${object_name}



# export checkpoint=''

# 

# bash scripts/run_tracking_capturevideo.sh



python train.py task=AllegroHandTracking train=HumanoidPPO sim_device='cuda:0' rl_device='cuda:0'  capture_video=True force_render=True headless=False   task.env.numEnvs=${numEnvs} train.params.config.minibatch_size=${minibatch_size}  task.env.useRelativeControl=${use_relative_control} train.params.config.max_epochs=100000 task.env.mocap_sv_info_fn=${mocap_sv_info_fn} checkpoint=${checkpoint} task.env.object_name=${object_name} tag=${tag} exp_dir=${exp_dir} train.params.config.name=${train_name} train.params.config.full_experiment_name=${full_experiment_name} task.env.goal_cond=${goal_cond} task.env.kinematics_only=${kinematics_only} task.env.use_kinematics_bias=${use_kinematics_bias}

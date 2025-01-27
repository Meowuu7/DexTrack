###### Sim parameters #######
export dt=0.0166
# export substeps=4
export substeps=2
###### Sim parameters #######

export num_frames=300

export tracking_save_info_fn='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data'
export tracking_data_sv_root='/cephfs/xueyi/data/GRAB_Tracking_PK_reduced_300/data'

export grab_inst_tag_to_optimized_res_fn='./data/statistics/data_inst_tag_to_optimized_res.npy'
export taco_inst_tag_to_optimized_res_fn=''
export object_type_to_latent_feature_fn='./data/statistics/obj_type_to_obj_feat.npy'
export inst_tag_to_latent_feature_fn='./data/statistics/inst_tag_to_obj_feat.npy'
export mocap_sv_info_fn='./data/GRAB_Tracking_PK_OFFSET_Reduced/data/passive_active_info_ori_grab_s2_apple_lift_nf_300.npy'


export hand_pose_guidance_glb_trans_coef=0.6
export hand_pose_guidance_glb_rot_coef=0.1
export hand_pose_guidance_fingerpose_coef=0.1

export rew_finger_obj_dist_coef=0.3
export rew_delta_hand_pose_coef=0.5

export glb_trans_vel_scale=0.5
export glb_rot_vel_scale=0.5
export dofSpeedScale=20

export numEnvs=40000
export minibatch_size=40000


export hand_type=allegro


##### multiple instnace training setting #####
export subj_idx=2
export subj_nm=s${subj_idx}
export target_inst_tag_list_fn=assets/inst_tag_list_s${subj_idx}.npy
##### multiple instnace training setting #####


##### single instnace training setting #####
export subj_nm=''
export generalist_tune_all_instnaces=False
export data_inst_flag='ori_grab_s2_duck_inspect_1_nf_300' 
export target_inst_tag_list_fn=''
export test_inst_tag=${data_inst_flag}
export single_instance_training=True # the learning model would not print additional infos in this setting
export numEnvs=22000
export minibatch_size=22000

export single_instance_state_based_train=True
##### single instnace training setting #####


export log_path=./logs/isaacgym_rl_exp_grab_train_${hand_type}_






export st_idx=4


# script for 
# multiple instances, 
# use kinematic bias,
# allegro hand



# bash scripts/run_tracking_headless_grab_whltrain_v1.sh




python train_pool.py --hand_type=${hand_type} --numEnvs=${numEnvs} --minibatch_size=${minibatch_size} \
    --test=False --use_kinematics_bias=True \
    --hand_pose_guidance_glb_trans_coef=${hand_pose_guidance_glb_trans_coef} --hand_pose_guidance_glb_rot_coef=${hand_pose_guidance_glb_rot_coef} --hand_pose_guidance_fingerpose_coef=${hand_pose_guidance_fingerpose_coef} \
    --dt=${dt} --glb_trans_vel_scale=${glb_trans_vel_scale} --glb_rot_vel_scale=${glb_rot_vel_scale} --rew_finger_obj_dist_coef=${rew_finger_obj_dist_coef} --rew_delta_hand_pose_coef=${rew_delta_hand_pose_coef} \
    --num_frames=${num_frames} --tracking_data_sv_root=${tracking_data_sv_root} --subj_nm=${subj_nm}  --st_idx=${st_idx} --dofSpeedScale=${dofSpeedScale}  --data_inst_flag=${data_inst_flag} \
    --checkpoint='' --max_epochs=1000 --test_inst_tag=${test_inst_tag} --test_optimized_res='' \
    --preload_experiences_tf=False --preload_experiences_path='' --single_instance_training=${single_instance_training} --generalist_tune_all_instnaces=${generalist_tune_all_instnaces} \
    --log_path=${log_path} --grab_inst_tag_to_optimized_res_fn=${grab_inst_tag_to_optimized_res_fn} --taco_inst_tag_to_optimized_res_fn=${taco_inst_tag_to_optimized_res_fn} \
    --supervised_loss_coef=0 --inst_tag_to_latent_feature_fn=${inst_tag_to_latent_feature_fn} --object_type_to_latent_feature_fn=${object_type_to_latent_feature_fn} --maxx_inst_nn=1 \
    --tracking_save_info_fn=${tracking_save_info_fn} --tracking_info_st_tag=passive_active_info_ --use_local_canonical_state=True  --dataset_type=grab --train_on_all_trajs=True  \
    --controlFrequencyInv=1 --target_inst_tag_list_fn=${target_inst_tag_list_fn} --single_instance_state_based_train=${single_instance_state_based_train}

# TODO: set this one in the with arm setting
# --w_franka=${w_franka} 


# TODO: add the obj ornt and rot reward scheduling setting 
#  --include_obj_rot_in_obs=${include_obj_rot_in_obs}
 
# TODO: notice that the control frequency inv is 1 in the default setting --- but in the with arm setting it should be 3


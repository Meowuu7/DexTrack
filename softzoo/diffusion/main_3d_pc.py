# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""



import os
import json
from utils.fixseed import fixseed
# from utils.parser_util import train_args
# from utils import dist_util
# from train.training_loop import TrainLoop
# from train.training_loop_ours import TrainLoop as TrainLoop_Ours ### trainer ours ###

# from training_loop import TrainLoop
# from training_loop_act import TrainLoop
# from training_loop_pc import TrainLoop
from training_loop_3d_pc import TrainLoop

from training_loop_3d_deterministic import TrainLoop as TrainLoopDeterministic

## TODO: the training loss function in the diffusion model
## TODO: simple method for running the model


from dataset.get_data import get_dataset_loader_3d_pc, get_dataset_loader_3d_v3_pc, get_dataset_loader_3d_v5_pc, get_dataset_loader_3d_v6_pc, get_dataset_loader_3d_v7_pc
from model_util import create_model_and_diffusion_3d_pc, create_model_deterministic
# from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform


import shutil
from hydra import compose, initialize
import argparse



def main(pre_args):
    
    with initialize(version_base="1.3", config_path="cfgs", job_name="test_app"):
        # if os.path.exists("/cephfs/xueyi/backup"):
        #     cfg = compose(config_name="K2_config_3d_k8s")
        # elif os.path.exists("/root/diffsim/softzoo"):
        #     cfg = compose(config_name="config_3d_k8s")
        # else:
        #     raise ValueError("Please run this code on the k8s cluster")
        cfg = compose(config_name="K2_config_3d_k8s")
        # else:
        #     cfg = compose(config_name="config")
    args = cfg
    
    args.sampling.sampling = pre_args.sampling
    args.training.resume_checkpoint_pc = pre_args.resume_checkpoint
    args.sampling.use_shadow_test_data = pre_args.use_shadow_test_data
    args.sampling.specified_test_subfolder = pre_args.specified_test_subfolder
    args.training.task_cond = pre_args.task_cond
    args.training.save_interval = pre_args.save_interval
    args.dataset_3d_pc.statistics_info_fn = pre_args.statistics_info_fn
    args.dataset_3d_pc.single_inst = pre_args.single_inst
    args.training.setting = pre_args.training_setting
    args.sampling.use_t = pre_args.use_t
    args.training.batch_size = pre_args.batch_size
    args.training.use_jointspace_seq = pre_args.training_use_jointspace_seq
    args.training.diff_task_translations = pre_args.diff_task_translations
    args.training.diff_task_space = pre_args.diff_task_space
    args.training.kine_diff = pre_args.kine_diff # kine diff #
    args.training.concat_two_dims = pre_args.concat_two_dims
    args.training.tracking_ctl_diff = pre_args.tracking_ctl_diff
    # target_grab_inst_tag: ''
#   target_grab_inst_opt_fn: ''
    args.sampling.target_grab_inst_tag = pre_args.target_grab_inst_tag
    args.sampling.target_grab_inst_opt_fn = pre_args.target_grab_inst_opt_fn
    args.training.AE_Diff = pre_args.AE_Diff
    args.training.train_AE = pre_args.train_AE
    args.training.train_Diff = pre_args.train_Diff
    args.training.cond_diff_allparams = pre_args.cond_diff_allparams
    args.training.succ_rew_threshold = pre_args.succ_rew_threshold
    args.training.slicing_data = pre_args.slicing_data
    args.training.slicing_ws = pre_args.slicing_ws
    args.training.grab_inst_tag_to_opt_stat_fn = pre_args.grab_inst_tag_to_opt_stat_fn
    
    
    args.training.grab_inst_tag_to_optimized_res_fn = pre_args.grab_inst_tag_to_optimized_res_fn
    args.training.taco_inst_tag_to_optimized_res_fn = pre_args.taco_inst_tag_to_optimized_res_fn
    
    # if len(pre_args.taco_inst_tag_to_optimized_res_fn) > 0:
    #     args.training.grab_inst_tag_to_optimized_res_fn = [args.training.grab_inst_tag_to_optimized_res_fn,  pre_args.taco_inst_tag_to_optimized_res_fn]
    
    args.dataset_3d_pc.multi_inst = pre_args.multi_inst
    args.dataset_3d_pc.sim_platform = pre_args.sim_platform
    
    args.training.task_cond_type = pre_args.task_cond_type
    args.training.debug = pre_args.debug
    args.training.history_ws = pre_args.history_ws
    
    # taco_interped_fr_grab_tag, taco_interped_data_sv_additional_tag
    args.training.taco_interped_fr_grab_tag = pre_args.taco_interped_fr_grab_tag
    args.training.taco_interped_data_sv_additional_tag = pre_args.taco_interped_data_sv_additional_tag

    args.training.exp_additional_tag = pre_args.exp_additional_tag
    args.training.sub_task_cond_type = pre_args.sub_task_cond_type 
    args.training.task_inherit_info_fn = pre_args.task_inherit_info_fn 
    args.training.glb_rot_use_quat = pre_args.glb_rot_use_quat 
    print(f"glb_rot_use_quat: {args.training.glb_rot_use_quat}") 
    args.training.use_kine_obj_pos_canonicalization = pre_args.use_kine_obj_pos_canonicalization 
    print(f"use_kine_obj_pos_canonicalization: {args.training.use_kine_obj_pos_canonicalization}") 
    args.dataset_3d_pc.data_statistics_info_fn = pre_args.data_statistics_info_fn # 
    # # /cephfs/
    # /cephfs/xueyi/uni_manip/isaacgym_rl_exp_grab_eval_v2/statistics/data_inst_tag_to_best_opt_res_all_rndselect.npy #
    args.training.kine_diff_version = pre_args.kine_diff_version 
    print(f"Using kinematic diff with model version: {args.training.kine_diff_version}")
    args.training.use_taco_data = pre_args.use_taco_data
    args.training.w_glb_traj_feat_cond = pre_args.w_glb_traj_feat_cond
    args.dataset.canonicalize_features = pre_args.canonicalize_features
    # obj_type_to_kinematics_traj_dict_fn, canonicalize_obj_pts
    args.dataset_3d_pc.obj_type_to_kinematics_traj_dict_fn = pre_args.obj_type_to_kinematics_traj_dict_fn
    args.dataset_3d_pc.canonicalize_obj_pts = pre_args.canonicalize_obj_pts
    args.dataset_3d_pc.forcasting_window_size = pre_args.forcasting_window_size
    args.training.w_history_window_index = pre_args.w_history_window_index
    
    args.training.partial_hand_info = pre_args.partial_hand_info
    args.training.partial_obj_info = pre_args.partial_obj_info
    
    args.training.diff_contact_sequence = pre_args.diff_contact_sequence
    args.training.w_masked_future_cond = pre_args.w_masked_future_cond
    args.training.st_ed_state_cond = pre_args.st_ed_state_cond
    args.training.partial_obj_pos_info = pre_args.partial_obj_pos_info
    args.training.centralize_info = pre_args.centralize_info
    args.training.glb_feat_per_skip = pre_args.glb_feat_per_skip
    args.dataset_3d_pc.history_window_size = pre_args.history_window_size
    # scale_clip_data # # scaled clip data #
    args.dataset_3d_pc.scale_clip_data = pre_args.scale_clip_data
    # hist_cond_partial_hand_info, hist_cond_partial_obj_info, hist_cond_partial_obj_pos_info
    args.training.hist_cond_partial_hand_info = pre_args.hist_cond_partial_hand_info
    args.training.hist_cond_partial_obj_info = pre_args.hist_cond_partial_obj_info
    args.training.hist_cond_partial_obj_pos_info = pre_args.hist_cond_partial_obj_pos_info
    args.training.inv_kine_freq = pre_args.inv_kine_freq
    # load_excuted_optimized_res, excuted_inst_tag_to_optimized_res
    args.training.load_excuted_optimized_res = pre_args.load_excuted_optimized_res
    args.training.excuted_inst_tag_to_optimized_res = pre_args.excuted_inst_tag_to_optimized_res
    
    args.training.text_feature_version = pre_args.text_feature_version
    
    
    args.dataset.num_frames = pre_args.num_frames
    
    # use_clip_glb_features, clip_feat_dim
    args.training.use_clip_glb_features = pre_args.use_clip_glb_features
    args.training.clip_feat_dim = pre_args.clip_feat_dim

    args.training.inv_forecasting_freq = pre_args.inv_forecasting_freq

    args.debug = pre_args.debug
    if len(pre_args.exp_tag) > 0:
        args.exp_tag = pre_args.exp_tag
    
    fixseed(cfg.seed)


    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    else:
        os.makedirs(args.save_dir, exist_ok=True) 
        exp_tag = args.exp_tag
        args.save_dir = os.path.join(args.save_dir, exp_tag)
        os.makedirs(args.save_dir, exist_ok=True)
        


    # shutil.copyfile(src, dst)
    config_path = "cfgs/config.yaml"
    dst_config_folder = args.save_dir
    shutil.copy(config_path, dst_config_folder)
    
    
    # dist_util.setup_dist(args.device)
    
    print("creating data loader...")
    #### getthe dataset and the num frames ####
    
    
    if args.dataset_3d_pc.data_tag == "v6": # 
        data = get_dataset_loader_3d_v6_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
        
    elif args.dataset_3d_pc.data_tag == "v7":
        data = get_dataset_loader_3d_v7_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
        
    elif args.dataset_3d_pc.data_tag == "v5":
        
        print(f"getting dataset for model with arch: {args.model.model_arch}")
        data = get_dataset_loader_3d_v5_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
        
    elif args.model.model_arch == "transformer_v2":
        
        print(f"getting dataset for model with arch: {args.model.model_arch}")
        data = get_dataset_loader_3d_v3_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    
    else:
        data = get_dataset_loader_3d_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)

    
    print("creating model and diffusion...")
    
    
    if pre_args.use_deterministic:
        model = create_model_deterministic(args)
        model.cuda()
        
        diffusion = None
        
        ##### whether to evaluate the data #####
        if args.sampling.sampling:
            TrainLoopDeterministic(args, model, diffusion, data).eval_loop()
        else:
            TrainLoopDeterministic(args, model, diffusion, data).run_loop()
        
    else:
        
        # model, diffusion = create_model_and_diffusion(args, data)
        model, diffusion = create_model_and_diffusion_3d_pc(args)
        model.cuda()
    
    

        #### data fitting model ####
        # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
        # print(f"Start training loops for rep_type {args.rep_type}")
        
        print(f"sampling: {args.sampling.sampling}")
        if args.sampling.sampling:
            
            if args.sampling.use_shadow_test_data:
                shadow_test_data_fn = args.sampling.shadow_test_data_fn 
                data.dataset.load_target_data(shadow_test_data_fn)
                ### TODO: the sampling function ##
                TrainLoop(args, model, diffusion, data).predict_from_shadow_target_data()
            else:
                ### evaluate loop ###
                sampling_t = args.sampling.use_t
                if sampling_t >= 1000:
                    sampling_t = None
                TrainLoop(args, model, diffusion, data).eval_loop(use_t=sampling_t)
        else:
            TrainLoop(args, model, diffusion, data).run_loop() 
    



# slicing_ws: 30
#   slicing_data: False



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    # parser.add_argument("--a_rot_task_x_angle_factor", type=float, default=0.5, help="Initial learning rate for cos, linear is 0.001")
    # parser.add_argument("--lr_fract", type=int, default=1000, help="Decay factor for exp")
    # parser.add_argument("--tau_grad_clip", type=float, default=0.001, help="Decay factor for exp")
    parser.add_argument("--sampling", default=False, action='store_true', help="Decay factor for exp")
    parser.add_argument("--use_shadow_test_data", default=False, action='store_true', help="Decay factor for exp")
    parser.add_argument("--task_cond", default=False, action='store_true', help="Decay factor for exp")
    # parser.add_argument("--selecting_res_taus", default=False, action='store_true', help="Whether this script is used to select taus")
    parser.add_argument("--resume_checkpoint", type=str, default='', help="Render mode for the environment")
    # specified_test_subfolder
    parser.add_argument("--specified_test_subfolder", type=str, default='', help="Render mode for the environment")
    parser.add_argument("--exp_tag", type=str, default='', help="Render mode for the environment")
    parser.add_argument("--debug", default=False, action='store_true', help="debug flag")
    # save_interval
    parser.add_argument("--save_interval", type=int, default=20000, help="Render mode for the environment")
    # statistics_info_fn
    parser.add_argument("--statistics_info_fn", type=str, default='', help="Render mode for the environment")
    # single_inst
    parser.add_argument("--single_inst", default=False, action='store_true', help="single_inst flag")
    # setting
    parser.add_argument("--training_setting", type=str, default='regular_training', help="Render mode for the environment")
    ### get use_t ###
    parser.add_argument("--use_t", type=int, default=200, help="Render mode for the environment")
    # batch_size
    parser.add_argument("--batch_size", type=int, default=16, help="Render mode for the environment")
    # use_jointspace_seq --- whether to use that
    parser.add_argument("--training_use_jointspace_seq", default=False, action='store_true', help="single_inst flag")
    # parser.add_argument("--best_taus_fn", type=str, default='/data/xueyi/uni_manip/tds_rl_exp/logs_PPO/test_allegro_bottle_v2__2024-06-14-05-06-49_seed3407_noisesigma0.3_lr0.0005_envallegro_env_bottle_v2_mujoco_net256_256_objrottarx_0.2', help="Render mode for the environment")
    
    # diff_task_translations #
    # diff_task_space # 
    parser.add_argument("--diff_task_space", default=False, action='store_true')
    parser.add_argument("--diff_task_translations", default=False, action='store_true')
    # kine_diff
    parser.add_argument("--kine_diff", default=False, action='store_true')
    # concat_two_dims --- whether to concat two dims #
    parser.add_argument("--concat_two_dims", default=False, action='store_true') ## 
    # tracking_ctl_diff
    parser.add_argument("--tracking_ctl_diff", default=False, action='store_true') ## 
    parser.add_argument("--AE_Diff", default=False, action='store_true') ## 
    parser.add_argument("--train_AE", default=False, action='store_true') ## 
    parser.add_argument("--train_Diff", default=False, action='store_true') ## 
    # target_grab_inst_tag: ''
    # target_grab_inst_opt_fn: ''
    parser.add_argument("--target_grab_inst_tag", type=str, default='', help="Render mode for the environment")
    ### get use_t ###
    parser.add_argument("--target_grab_inst_opt_fn", type=str, default='', help="Render mode for the environment")
    ### get use_t ###
    
    # cond_diff_allparams
    parser.add_argument("--cond_diff_allparams", default=False, action='store_true') 
    # succ_rew_threshold    
    parser.add_argument("--succ_rew_threshold", type=float, default=50.0, help="Render mode for the environment")
    # multi_inst
    parser.add_argument("--multi_inst", default=False, action='store_true') 
    parser.add_argument("--slicing_ws", type=int, default=30)
    parser.add_argument("--slicing_data", default=False, action='store_true') 
    # grab_inst_tag_to_opt_stat_fn
    parser.add_argument("--grab_inst_tag_to_opt_stat_fn", type=str, default='')
    # sim_platform
    parser.add_argument("--sim_platform", type=str, default='pybullet')
    # grab_inst_tag_to_optimized_res_fn
    parser.add_argument("--grab_inst_tag_to_optimized_res_fn", type=str, default='')
    # task_cond_type
    parser.add_argument("--task_cond_type", type=str, default='future')
    # debug
    # history_ws
    parser.add_argument("--history_ws", type=int, default=30)
    parser.add_argument("--use_deterministic", default=False, action='store_true') 
    parser.add_argument("--taco_inst_tag_to_optimized_res_fn", type=str, default='')
    # 
    parser.add_argument("--taco_interped_fr_grab_tag", type=str, default='ori_grab_s2_phone_call_1')
    parser.add_argument("--taco_interped_data_sv_additional_tag", type=str, default='v2')
    # parser.add_argument("--debug", default=False, action='store_true') 
    # exp_additional_tag
    parser.add_argument("--exp_additional_tag", type=str, default='')
    # sub_task_cond_type
    parser.add_argument("--sub_task_cond_type", type=str, default='full')
    # task_inherit_info_fn
    parser.add_argument("--task_inherit_info_fn", type=str, default='')
    # glb_rot_use_quat
    parser.add_argument("--glb_rot_use_quat", default=False, action='store_true') 
    # use_kine_obj_pos_canonicalization
    parser.add_argument("--use_kine_obj_pos_canonicalization", default=False, action='store_true') 
    # data_statistics_info_fn
    parser.add_argument("--data_statistics_info_fn", type=str, default='')
    # kine_diff_version
    parser.add_argument("--kine_diff_version", type=str, default='v1')
    # use_taco_data
    parser.add_argument("--use_taco_data", default=False, action='store_true') 
    # num_frames
    parser.add_argument("--num_frames", type=int, default=150) 
    parser.add_argument("--maxx_inst_nn", type=int, default=2000) 
    # w_glb_traj_feat_cond
    parser.add_argument("--w_glb_traj_feat_cond", default=False, action='store_true') 
    # canonicalize_features
    parser.add_argument("--canonicalize_features", default=False, action='store_true') 
    # obj_type_to_kinematics_traj_dict_fn, canonicalize_obj_pts
    parser.add_argument("--obj_type_to_kinematics_traj_dict_fn", type=str, default='')
    parser.add_argument("--canonicalize_obj_pts", default=False, action='store_true') 
    # forcasting_window_size
    parser.add_argument("--forcasting_window_size", type=int, default=30)
    # w_history_window_index
    parser.add_argument("--w_history_window_index", default=False, action='store_true') 
    # partial_hand_info, partial_obj_info
    parser.add_argument("--partial_hand_info", default=False, action='store_true') 
    parser.add_argument("--partial_obj_info", default=False, action='store_true') 
    # diff_contact_sequence
    parser.add_argument("--diff_contact_sequence", default=False, action='store_true') 
    # w_masked_future_cond
    parser.add_argument("--w_masked_future_cond", default=False, action='store_true') 
    # st_ed_state_cond
    parser.add_argument("--st_ed_state_cond", default=False, action='store_true') 
    # partial_obj_pos_info
    parser.add_argument("--partial_obj_pos_info", default=False, action='store_true') 
    # centralize_info
    parser.add_argument("--centralize_info", default=False, action='store_true') 
    # glb_feat_per_skip
    parser.add_argument("--glb_feat_per_skip", type=int, default=1)
    # history_window_size
    parser.add_argument("--history_window_size", type=int, default=30)
    # scale_clip_data
    parser.add_argument("--scale_clip_data", default=False, action='store_true') 
    # hist_cond_partial_hand_info, hist_cond_partial_obj_info, hist_cond_partial_obj_pos_info
    parser.add_argument("--hist_cond_partial_hand_info", default=False, action='store_true') 
    parser.add_argument("--hist_cond_partial_obj_info", default=False, action='store_true') 
    parser.add_argument("--hist_cond_partial_obj_pos_info", default=False, action='store_true') 
    # load_excuted_optimized_res, excuted_inst_tag_to_optimized_res
    parser.add_argument("--load_excuted_optimized_res", default=False, action='store_true') 
    # excuted_inst_tag_to_optimized_res
    parser.add_argument("--excuted_inst_tag_to_optimized_res", type=str, default='')
    # inv_kine_freq
    parser.add_argument("--inv_kine_freq", type=int, default=1)
    # use_clip_glb_features, clip_feat_dim
    parser.add_argument("--use_clip_glb_features", default=False, action='store_true') 
    # clip_feat_dim
    parser.add_argument("--clip_feat_dim", type=int, default=512)
    # text_feature_version
    parser.add_argument("--text_feature_version", type=str, default='v1')
    # inv_forecasting_freq
    parser.add_argument("--inv_forecasting_freq", type=int, default=1)
    args = parser.parse_args()
    
    main(args)

# # continuous ? #
# Diffusion part ### diffusion part #
# how to sample from the model using the timestep #
# how to sample from the mode lus ingthe timestep #


# CUDA_VISIBLE_DEVICES=3 python main_3d_pc.py --resume_checkpoint="/data/xueyi/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v2_v1_pts_512_singleinst_v3_/model000090000.pt" --sampling 

# CUDA_VISIBLE_DEVICES=7 python main_3d_pc.py --resume_checkpoint="/data/xueyi/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v2_datav5_pts_512_allinstclipts_v4_allclipssingleinst_/model000038000.pt" --sampling 

# /data/xueyi/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v2_datav5_pts_512_allinstclipts_v4_allclipssingleinst_v6datda_/model000030000.pt

# CUDA_VISIBLE_DEVICES=2 python main_3d_pc.py --resume_checkpoint="/data/xueyi/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v2_datav5_pts_512_allinstclipts_v4_allclipssingleinst_v6datda_/model000030000.pt" --sampling 



##### get the 3d pc main #####
# CUDA_VISIBLE_DEVICES=4 python main_3d_pc.py --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v3_datav5_pts_512_v7datav_allinst_allegrobox_largebsz_/model000020000.pt

# CUDA_VISIBLE_DEVICES=4 python main_3d_pc.py  --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v3_datav5_pts_512_v7datav_allinst_allegrobox_largebsz_/model000020000.pt  --sampling --use_shadow_test_data 

#### use the shadow test data ####
## main 3d pc ##
####### resume checkpoints #######
# CUDA_VISIBLE_DEVICES=2 python main_3d_pc.py  --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v3_datav5_pts_512_v7datav_singleinst_allegrobox_static_first_frame_/model000020000.pt  --sampling --use_shadow_test_data 


####### resume checkpoints #######
# CUDA_VISIBLE_DEVICES=0 python main_3d_pc.py  --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v3_datav5_pts_512_v7datav_singleinst_allegrobox_static_first_frame_flat_fivefin_/model000100000.pt  --sampling --use_shadow_test_data 

# CUDA_VISIBLE_DEVICES=0 python main_3d_pc.py  --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v3_datav5_pts_512_v7datav_allinst_allegrobox_largebsz_/model000140000.pt  --sampling  --specified_test_subfolder="allegro_bottle_5_pds_wgravity_v3__ctlfreq_10_taskstage5_objtype_box_objm0.2_objsxyz_0.02_0.02_0.382_objinitxyz_0.2_0.2_0.2_objgoalxyz_0.2_0.5_0.37_objgoalrot_0.1_0_0_objgoalrot2_0.1_0_0_tar_"

# CUDA_VISIBLE_DEVICES=4 python main_3d_pc.py --exp_tag=test_scale_ori_transformer_v3_datav5_pts_512_v7datav_allinst_allegrobox_wtaskcond_  --task_cond --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/test_scale_ori_transformer_v3_datav5_pts_512_v7datav_allinst_allegrobox_largebsz_/model000240000.pt

# # CUDA_VISIBLE_DEVICES=6 python main_3d_pc.py  --task_cond --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_lotsballs_allinst_/model000020000.pt --exp_tag=allegro_lotsballs_singleinst_taskcond_v2_  --save_interval=200
# --exp_tag=allegro_lotsballs_allinst_taskcond_ --debug
# --exp_tag=allegro_lotsballs_singleinst_taskcond_

# cuda visible devices = 0
# CUDA_VISIBLE_DEVICES=0 python main_3d_pc.py  --task_cond  --resume_checkpoint=/cephfs/yilaa/uni_manip/tds_diffusion_exp/allegro_lotsballs_singleinst_/model000160000.pt --exp_tag=allegro_lotsballs_singleinst_taskcond_v2_ --save_interval=100


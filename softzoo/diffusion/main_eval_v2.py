# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""



import os
import json
from utils.fixseed import fixseed



from eval_loop_combined_v2 import EvalLoopCombined

from dataset.get_data import get_dataset_loader_act, get_dataset_loader, get_dataset_loader_pc, get_dataset_loader_segs
from model_util import create_model_and_diffusion_act, create_model_and_diffusion, create_model_and_diffusion_pc, create_model_and_diffusion_segs


import shutil
from hydra import compose, initialize
def main():
    
    if os.path.exists("/root/diffsim/softzoo"):
        config_file_name = "config_k8s_eval"
    else:
        config_file_name = "config"
    
    with initialize(version_base="1.3", config_path="cfgs", job_name="test_app"):
        if os.path.exists("/root/diffsim/softzoo"):
            cfg = compose(config_name=config_file_name)
        else:
            cfg = compose(config_name=config_file_name)

    
    args = cfg


    fixseed(cfg.seed)


    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    else: ## get the svae dir and the exp_tag #
        os.makedirs(args.save_dir, exist_ok=True) 
        exp_tag = args.exp_tag
        args.save_dir = os.path.join(args.save_dir, exp_tag)
        os.makedirs(args.save_dir, exist_ok=True)


    config_path = f"cfgs/{config_file_name}.yaml"
    dst_config_folder = args.save_dir
    shutil.copy(config_path, dst_config_folder)
    
    
    print("creating data loader...")
    data = get_dataset_loader_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    
    data_segs = get_dataset_loader_segs(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    
    # data_graph = get_dataset_loader(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)

    print("creating model and diffusion...")
    
    
    model, diffusion = create_model_and_diffusion_pc(args)
    model.cuda()
    
    model_segs, diffusion_segs = create_model_and_diffusion_segs(args)
    model_segs.cuda()
    
    # model_graph, diffusion_graph = create_model_and_diffusion(args=args.diffusion)
    # model_graph.cuda()
    
    
    model_dict = {
        'model_graph': model, 
        'model_act': model,
        'model_segs': model_segs,
    }
    diffusion_dict = {
        'diffusion_graph': diffusion, 
        'diffusion_act': diffusion,
        'diffusion_segs': diffusion_segs,
    }
    data_dict = {
        'data_graph': data, 
        'data_act': data,
        'data_segs': data_segs,
    }
    
    
    sv_dict_fn = "/data/xueyi/uni_manip/exp/eval_v2_/eval_v2_/sampled_pcd_wact_dict.npy"
    data_segs.dataset.calibrate_data_dict_from_sv_dict(sv_dict_fn)
    
    #### ==== evaluate segmentations from the point clouds ==== ####
    # EvalLoopCombined(args, model_dict, diffusion_dict, data_dict).evaluate_segs_from_pcd_run_loop()
    
    ## get a new trajectory for the manipulator from the dataset? ##
    ## the sampled trajectories are unconstrained from ##
    ## evaluate from pcd run loop ##
    
    ### 
    # EvalLoopCombined(args, model_dict, diffusion_dict, data_dict).evaluate_from_pcd_run_loop()
    
    
    use_t = args.sampling.use_t
    args.exp_tag = f"{args.exp_tag}_use_t_{use_t}_"
    
    ###### ===== Get args and cfgs ===== ######
    # EvalLoopCombined(args, model_dict, diffusion_dict, data_dict).evaluate_from_pcd_wconstraints_run_loop()
    
    transfer_pcd_use_t = args.sampling.transfer_pcd_use_t
    args.exp_tag = f"{args.exp_tag}_transfer_pcds_use_t_{transfer_pcd_use_t}_"
    ###### ===== Translate the manipulation trajectories ===== ######
    EvalLoopCombined(args, model_dict, diffusion_dict, data_dict).evaluate_transfer_pcds_run_loop()



if __name__ == "__main__":
    main() 




# CUDA_VISIBLE_DEVICES=5  python main_eval_v2.py


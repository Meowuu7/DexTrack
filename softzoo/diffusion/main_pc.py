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
from training_loop_pc import TrainLoop


## TODO: the training loss function in the diffusion model
## TODO: simple method for running the model


from dataset.get_data import get_dataset_loader_pc
from model_util import create_model_and_diffusion_pc
# from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


import shutil
from hydra import compose, initialize
def main():
    
    with initialize(version_base="1.3", config_path="cfgs", job_name="test_app"):
        if os.path.exists("/root/diffsim/softzoo"):
            cfg = compose(config_name="config_k8s")
        else:
            cfg = compose(config_name="config")
    args = cfg
    

    
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
    #### getthe dataset and the num frames ## ## get_dataset loader ##
    data = get_dataset_loader_pc(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)

    print("creating model and diffusion...")
    
    # model, diffusion = create_model_and_diffusion(args, data)  ## get the
    model, diffusion = create_model_and_diffusion_pc(args)
    model.cuda()
    

    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    # print(f"Start training loops for rep_type {args.rep_type}")
    TrainLoop(args, model, diffusion, data).run_loop() 

# 
if __name__ == "__main__":
    main()

# CUDA_VISIBLE_DEVICES=0  python main_pc.py
# 

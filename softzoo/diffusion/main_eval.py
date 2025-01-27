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
from eval_loop_combined import EvalLoopCombined


## TODO: construct and get the data loader 
    ## TODO: remember to normalize the values for the node statistics and the edge statistics ## get the jata dict 
    ## TODO: store them in the dict form  ## store them 
## TODO: the training loss function in the diffusion model
## TODO: simple method for running the model


from dataset.get_data import get_dataset_loader_act, get_dataset_loader
from model_util import create_model_and_diffusion_act, create_model_and_diffusion
# from train.train_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform  # required for the eval operation


import shutil
from hydra import compose, initialize
def main():
    
    with initialize(version_base="1.3", config_path="cfgs", job_name="test_app"):
        cfg = compose(config_name="config_eval")
    args = cfg
    
    ## add the save dir ##
    
    # args = train_args() 
    fixseed(cfg.seed)
    
    #
    # train_platform_type = eval(args.train_platform_type)
    # train_platform = train_platform_type(args.save_dir)
    # train_platform.report_args(args, name='Args') # train platform

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    
    else:
        os.makedirs(args.save_dir, exist_ok=True) 
        exp_tag = args.exp_tag
        args.save_dir = os.path.join(args.save_dir, exp_tag)
        os.makedirs(args.save_dir, exist_ok=True)
        
        
    # args_path = os.path.join(args.save_dir, 'args.json')
    # with open(args_path, 'w') as fw:
    #     json.dump(vars(args), fw, indent=4, sort_keys=True)
    
    # shutil.copyfile(src, dst)
    config_path = "cfgs/config.yaml"
    dst_config_folder = args.save_dir
    shutil.copy(config_path, dst_config_folder)
    

    # dist 
    # dist_util.setup_dist(args.device)
    # 

    print("creating data loader...")
    #### getthe dataset and the num frames ## ## get_dataset loader ##
    data = get_dataset_loader_act(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)
    
    data_graph = get_dataset_loader(name=args.dataset.dataset_type, batch_size=args.training.batch_size, num_frames=args.dataset.num_frames, args=args)

    print("creating model and diffusion...")
    
    # model, diffusion = create_model_and_diffusion(args, data)  ## get the
    model, diffusion = create_model_and_diffusion_act(args)
    model.cuda()
    
    model_graph, diffusion_graph = create_model_and_diffusion(args=args)
    model_graph.cuda()
    
    

    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    # print(f"Start training loops for rep_type {args.rep_type}")
    # EvalLoopCombined(args, model, diffusion, data).run_loop() 
    
    ### eval graph loops --- run the evaluation ##
    # __init__(self, args, model_graph, diffusion_graph, model_act, diffusion_act, data, data_act):
    EvalLoopCombined(args, model_graph, diffusion_graph, model, diffusion, data_graph, data).evaluate_run_loop()


# main() #
if __name__ == "__main__":
    main() 
    # the repo and the trained ckpts for the graph and the act # 

# CUDA_VISIBLE_DEVICES=1  python main_eval.py
# 

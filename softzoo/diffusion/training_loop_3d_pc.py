# import copy
# import functools
import os
import time
from types import SimpleNamespace
import numpy as np

import blobfile as bf
import torch
from torch.optim import AdamW

from diffusion import logger
# from utils import dist_util
# from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, UniformSampler
from tqdm import tqdm
from diffusion.resample import create_named_schedule_sampler
from utils import dist_util
# from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper
# from eval import eval_humanml, eval_humanact12_uestc

### TODO: cosntruct and get the data loder ###
# from data_loaders.get_data import get_dataset_loader


# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
## intial log loss scale ##
INITIAL_LOG_LOSS_SCALE = 20.0



class TrainLoop:
    def __init__(self, args, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset.dataset_type
        self.model = model
        self.diffusion = diffusion
        
        


        self.data = data
        self.batch_size = args.training.batch_size
        self.microbatch = args.training.batch_size
        self.lr = args.training.lr
        self.log_interval = args.training.log_interval
        self.save_interval = args.training.save_interval
        self.resume_checkpoint = args.training.resume_checkpoint_pc
        self.use_fp16 = False  
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.training.weight_decay
        self.lr_anneal_steps = args.training.lr_anneal_steps
        self.training_setting = args.training.setting
        
        
        ## if we have task cond the nwe should add ##
        self.kine_diff  = args.training.kine_diff
        ## if we have task_cond then weshould ##
        self.task_cond = args.training.task_cond
        
        self.AE_Diff = args.training.AE_Diff
        self.train_AE = args.training.train_AE
        self.train_Diff = args.training.train_Diff
        # 
        self.cond_diff_allparams = args.training.cond_diff_allparams
        
        try: 
            self.sub_task_cond_type = args.training.sub_task_cond_type
        except:
            self.sub_task_cond_type = 'full'
            
            
        # partial_hand_info = cfg.training.partial_hand_info
        # partial_obj_info = cfg.training.partial_obj_info
        
        self.partial_hand_info = args.training.partial_hand_info
        self.partial_obj_info = args.training.partial_obj_info
        self.partial_obj_pos_info = args.training.partial_obj_pos_info
        
        
        
        # w_glb_traj_feat_cond # 
        self.w_glb_traj_feat_cond = args.training.w_glb_traj_feat_cond # with glb traj feat cond #
        self.debug = args.training.debug
        
        try:
            self.w_masked_future_cond = args.training.w_masked_future_cond
        except:
            self.w_masked_future_cond = False
        
        try:    
            self.w_history_window_index = args.training.w_history_window_index
        except:
            self.w_history_window_index = False 
            
        try:
            self.diff_contact_sequence = args.training.diff_contact_sequence
        except:
            self.diff_contact_sequence = False
            
        print(f"self.w_history_window_index: {self.w_history_window_index}")
        #### getting related configs ####

        self.step = 0
        # self.resume_step = 0
        self.resume_step = False
        self.global_batch = self.batch_size
        self.num_steps = args.training.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        
        print(f"num_epochs: {self.num_epochs}, num_steps: {self.num_steps}")

        self.sync_cuda = torch.cuda.is_available()
        
        ## === resume the checkpoint === ##
        self._load_and_sync_parameters() 


        self.save_dir = args.save_dir
        
        
        # if len(self.resume_checkpoint) > 0: # resume of the checkpoint # # #
        #     ckpt_folder = "/".join(self.resume_checkpoint.split("/")[:-1])
        #     self.save_dir = ckpt_folder
        print(f"save_dirP {self.save_dir}")
        # self.overwrite = args.overwrite


        # 
        if self.training_setting in ['trajectory_forcasting', 'trajectory_encdec']:
            trainable_params = []
            trainable_params += list(self.model.parameters())
            self.opt = AdamW(
                trainable_params, lr=self.lr, weight_decay=self.weight_decay
            ) 
        else:
            if self.task_cond:
                
                if self.AE_Diff:
                    # trainable_params = []
                    # # trainable_params += list(self.model.cond_input_process_pc.parameters())
                    # # trainable_params += list(self.model.cond_input_process_feat.parameters())
                    # trainable_params += list(self.model.parameters())
                    
                    # ### if we only have the trainable parameters ### #
                    trainable_params = []
                    
                    # trainable_params += list(self.model.positional_encoder_feat.parameters())
                    # trainable_params += list(self.model.transformer_encoder_feat.parameters())
                    # trainable_params += list(self.model.time_embedder.parameters())
                    # trainable_params += list(self.model.transformer_with_timesteps_encoder_feat.parameters())
                    # trainable_params += list(self.model.pc_latent_processing.parameters())
                    # trainable_params += list(self.model.feat_latent_processing.parameters())
                    if self.kine_diff:
                        trainable_params += list(self.model.cond_input_process_pc.parameters())
                        trainable_params += list(self.model.cond_input_process_feat.parameters())
                        trainable_params += list(self.model.positional_encoder_feat.parameters())
                        trainable_params += list(self.model.transformer_encoder_feat.parameters())
                        trainable_params += list(self.model.positional_encoder_time.parameters())
                        trainable_params += list(self.model.time_embedder.parameters())
                        trainable_params += list(self.model.transformer_with_timesteps_encoder_feat.parameters())
                        trainable_params += list(self.model.pc_latent_processing.parameters())
                        trainable_params += list(self.model.feat_latent_processing.parameters())
                        try:
                            trainable_params += list(self.model.cond_input_process_feat_hist.parameters())
                            trainable_params += list(self.model.cond_input_positional_encoder_hist.parameters())
                            trainable_params += list(self.model.cond_input_transformer_encoder_hist.parameters())
                        except:
                            pass
                        try:
                            trainable_params += list(self.model.cond_input_position_indicating_encoder_hist.parameters())
                        except:
                            pass
                    else:
                        
                        # trainable_params += list(self.model.cond_input_process_pc.parameters())
                        # trainable_params += list(self.model.cond_input_process_feat.parameters())
                        trainable_params += list(self.model.positional_encoder.parameters())
                        trainable_params += list(self.model.transformer_encoder.parameters())
                        trainable_params += list(self.model.time_embedder.parameters())
                        trainable_params += list(self.model.transformer_with_timesteps_encoder.parameters())
                        
                        
                        if self.sub_task_cond_type in ['full', 'full_woornt']:
                            trainable_params += list(self.model.cond_input_process_pc.parameters())
                            trainable_params += list(self.model.cond_input_process_feat.parameters())
                            trainable_params += list(self.model.cond_positional_encoder.parameters())
                            trainable_params += list(self.model.cond_transformer_encoder.parameters())
                            trainable_params += list(self.model.history_cond_input_process_feat.parameters())
                            trainable_params += list(self.model.history_cond_positional_encoder.parameters())
                            trainable_params += list(self.model.history_cond_transformer_encoder.parameters())
                        elif self.sub_task_cond_type == 'full_wohistory':
                            trainable_params += list(self.model.future_cond_input_process_feat.parameters())
                            trainable_params += list(self.model.cond_input_process_pc.parameters())
                            trainable_params += list(self.model.future_cond_positional_encoder.parameters())
                            trainable_params += list(self.model.future_cond_transformer_encoder.parameters())
                        elif self.sub_task_cond_type == 'hand_pose_traj_wpc':
                            trainable_params += list(self.model.cond_input_process_pc.parameters())
                            trainable_params += list(self.model.cond_input_process_feat.parameters())
                            trainable_params += list(self.model.cond_positional_encoder.parameters())
                            trainable_params += list(self.model.cond_transformer_encoder.parameters())
                            trainable_params += list(self.model.future_cond_input_process_feat.parameters())
                            trainable_params += list(self.model.future_cond_positional_encoder.parameters())
                            trainable_params += list(self.model.future_cond_transformer_encoder.parameters())
                        else:
                            trainable_params += list(self.model.parameters())
                            
                        
                        # trainable_params += list(self.model.parameters())
                    
                    self.opt = AdamW(
                        trainable_params, lr=self.lr, weight_decay=self.weight_decay
                    ) 
                    # self.args.training.tracking_ctl_diff #
                elif self.args.training.tracking_ctl_diff:
                    # cond diff all parameters? #
                    
                    # 
                    if self.cond_diff_allparams:
                        trainable_params = []
                        trainable_params += list(self.model.parameters())
                        self.opt = AdamW(
                            trainable_params, lr=self.lr, weight_decay=self.weight_decay
                        ) # cond 
                    else:
                        self.opt = AdamW(
                            list(self.model.cond_processing.parameters()) + list(self.model.transformer_encoder_cond.parameters()), lr=self.lr, weight_decay=self.weight_decay
                        )
                else:
                    self.opt = AdamW(
                        self.model.cond_processing.parameters(), lr=self.lr, weight_decay=self.weight_decay
                    )
                    
            else:
                if self.AE_Diff and self.train_Diff and (not self.train_AE):
                    # if self.train_Diff:
                    if self.kine_diff:
                        trainable_params = []
                        trainable_params += list(self.model.positional_encoder_feat.parameters())
                        trainable_params += list(self.model.transformer_encoder_feat.parameters())
                        trainable_params += list(self.model.time_embedder.parameters())
                        trainable_params += list(self.model.transformer_with_timesteps_encoder_feat.parameters())
                        trainable_params += list(self.model.pc_latent_processing.parameters())
                        trainable_params += list(self.model.feat_latent_processing.parameters())
                    else:
                        trainable_params = []
                        trainable_params += list(self.model.positional_encoder.parameters())
                        trainable_params += list(self.model.transformer_encoder.parameters())
                        trainable_params += list(self.model.time_embedder.parameters())
                        trainable_params += list(self.model.transformer_with_timesteps_encoder.parameters())
                    
                    self.opt = AdamW(
                        trainable_params, lr=self.lr, weight_decay=self.weight_decay
                    )
                else: #
                    print(f"Adding all parameters into the optimizer")
                    self.opt = AdamW(
                        self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
                    )
        
        
        # get the device #
        
        self.device = torch.device('cuda')

        if not self.training_setting in ['trajectory_forcasting', 'trajectory_encdec']:
            self.schedule_sampler_type = 'uniform'
            self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
            self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False  # if self.args.nprocs == 1 else True # # if self.args.nprocs # # self.args.nprocs #
        self.ddp_model = model
        
    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict:
            # print()
            # if self.args.resume_diff:
            print(f"k: {k}")
            if k in model_dict:
                if ori_dict[k].shape == model_dict[k].shape:
                    v = ori_dict[k]
                    part_dict[k] = v
                    tot_params_n += 1
                # v = ori_dict[k]
                # part_dict[k] = v
                # tot_params_n += 1
            # else: # training loop 3d pc ## ##
            #     if k in model_dict and "denoising" not in k:
            #         v = ori_dict[k]
            #         part_dict[k] = v
            #         tot_params_n += 1
        model_dict.update(part_dict)
        model.load_state_dict(model_dict)
        print(f"Resume glb-backbone finished!! Total number of parameters: {tot_params_n}.")
    
    
    
    def _load_and_sync_parameters_cond(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            state_dicts = dist_util.load_state_dict(
                                resume_checkpoint, map_location=dist_util.dev()
                            )
            if self.args.diff_basejtsrel:
                model_dict = self.model.state_dict()
                    
                model_dict.update(state_dicts)
                self.model.load_state_dict(model_dict)
                
                if self.args.finetune_with_cond_jtsobj:
                    # cond_joints_offset_input_process <- joints_offset_input_process; cond_sequence_pos_encoder <- sequence_pos_encoder; cond_seqTransEncoder <- seqTransEncoder
                    self.model.cond_joints_offset_input_process.load_state_dict(self.model.joints_offset_input_process.state_dict())
                    self.model.cond_sequence_pos_encoder.load_state_dict(self.model.sequence_pos_encoder.state_dict())
                    self.model.cond_seqTransEncoder.load_state_dict(self.model.seqTransEncoder.state_dict())
                
            else:
                raise ValueError(f"Must have diff_basejtsrel setting, others not implemented yet!")


    def _load_and_sync_parameters(self):
        resume_checkpoint = find_resume_checkpoint() or self.resume_checkpoint

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
            
            
            self.safe_load_ckpt(self.model, 
                                    dist_util.load_state_dict(
                                        resume_checkpoint, map_location=dist_util.dev()
                                    )
                                )

    def _load_optimizer_state(self): ## load optimizer state ##
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:09}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

  

    def run_loop(self):
        
        ### run loop ###
        for epoch in range(self.num_epochs):
            for batch in tqdm(self.data): # 
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                
                for k in batch: 
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                    elif isinstance(batch[k], list):
                        batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
                    else:
                        batch[k] = batch[k]
                
                
                self.run_step(batch) 
                if self.step % self.log_interval == 0:
                    loss_dict = logger.get_current().name2val
                    print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, loss_dict["loss"]))
                    
                ## how to train a good policy ##
                if self.step > 0 and self.step % self.save_interval == 0:
                    self.save() # 

                    # predict from data #
                    self.model.eval()
                    tot_samples = self.predict_from_data(batch)
                    
                    
                    notarget = False
                    if self.training_setting == 'trajectory_translations':
                        notarget = True 
                    print(f"tot_samples: {tot_samples.keys()}")
                    
                    for key in batch:
                        if key not in tot_samples:
                            tot_samples[key] = batch[key]
                    
                    
                    # scaled samples #
                    tot_samples = self.data.dataset.inv_scale_data_v2(tot_samples, data_nm=batch['data_nm'],  notarget=notarget)
                    tot_samples = {
                        key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in tot_samples.items() 
                    }
                    
                    ## tot training loop 3d pc ##
                    print(f"tot_samples: {tot_samples.keys()}")
                    
                    input_batch = batch
                    inv_sacled_input_batch = self.data.dataset.inv_scale_data_v2(input_batch, data_nm=batch['data_nm'])
                    inv_sacled_input_batch = {
                        key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in inv_sacled_input_batch.items()
                    }
                    
                    
                    tot_samples_sv_dict = {
                        'samples': tot_samples,
                        'closest_training_data': inv_sacled_input_batch
                    }
                    
                    
                    self.save_samples(tot_samples_sv_dict)

                self.step += 1



    def predict_from_shadow_target_data(self, ):
        i_batch = 0
        for batch in tqdm(self.data): 
            if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                break
            
            for k in batch: 
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
                elif isinstance(batch[k], list):
                    batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
                else:
                    batch[k] = batch[k]

            self.model.eval() 
            
            ###### TODO: predict from the data  ######
            
            tot_samples = self.predict_from_data_from_shadow_target(use_t=None)
            tot_samples = self.data.dataset.inv_scale_data_v2(tot_samples) # inv scale the data
            
            tot_samples = {
                key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in tot_samples.items()
            }  # tot samples ##
            
            tot_samples_sv_dict = {
                'samples': tot_samples,
            }
            
            sv_samples_fn = f"samples_ep_{0}_batch_{i_batch}.npy" # b
            sv_samples_fn = os.path.join(self.save_dir, sv_samples_fn)
            np.save(sv_samples_fn, tot_samples_sv_dict)
            
            
            i_batch += 1
            break
            


    def eval_loop(self, use_t=None):
        
        for epoch in range(10):
            i_batch = 0
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                
                for k in batch: 
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                    elif isinstance(batch[k], list):
                        batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
                    else:
                        batch[k] = batch[k]

                self.model.eval() 
                
                
                tot_samples = self.predict_from_data(batch, use_t=use_t)
                notarget = False
                if self.training_setting == 'trajectory_translations':
                    notarget = True 
                
                
                tot_samples = self.data.dataset.inv_scale_data_v2(tot_samples, data_nm=batch['data_nm'], notarget=notarget) # inv scale the data
                
                
                tot_samples = { 
                    key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in tot_samples.items()
                } 
                
                input_batch = batch
                inv_sacled_input_batch = self.data.dataset.inv_scale_data_v2(input_batch, data_nm=batch['data_nm'])
                inv_sacled_input_batch = {
                    key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in inv_sacled_input_batch.items()
                }
                
                # inv scale the data #
                
                tot_samples_sv_dict = {
                    'samples': tot_samples,
                    'closest_training_data': inv_sacled_input_batch
                }
                
                if 'data_nm' in batch:
                    tot_samples_sv_dict['data_nm'] = batch['data_nm']
                
                sv_samples_fn = f"samples_ep_{epoch}_batch_{i_batch}.npy"
                sv_samples_fn = os.path.join(self.save_dir, sv_samples_fn)
                print(f"Samples saved to {sv_samples_fn}")
                np.save(sv_samples_fn, tot_samples_sv_dict)
                
                
                i_batch += 1
                
                self.step += 1


    def evaluate(self):
        if not self.args.eval_during_training:
            return


    def run_step(self, batch): 
        
        
        # if self.training_setting == 'trajectory_forcasting_diffusion_masked_futurecond':
        #     self.forward_backward_trajectory_forcasting_diffusion_masked_futurecond(batch=batch)
        if self.training_setting == 'trajectory_forcasting_diffusion':
            # forward backward trajectory diffusion #
            if self.w_masked_future_cond:
                self.forward_backward_trajectory_forcasting_diffusion_masked_futurecond(batch=batch)
            else:
                self.forward_backward_trajectory_forcasting_diffusion(batch=batch)
        
        elif self.training_setting == 'trajectory_encdec' :
            self.forward_backward_traj_forcasting_pc_traj_encdec(batch=batch)   
        
        elif self.training_setting == 'trajectory_forcasting':
            self.forward_backward_traj_forcasting(batch=batch)
        elif self.training_setting == 'trajectory_translations':
            self.forward_backward_traj_translations(batch)
        elif self.training_setting == 'trajectory_translations_cond':
            self.forward_backward_traj_translations_cond(batch)
        else: # change the training setting 
            self.forward_backward(batch)
        # self.mp_trainer.optimize(self.opt)
        self.opt.step()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch):
        # self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        
        nn_bsz = batch['X'].shape[0]
        
        interested_keys = ['X', 'E', 'X_cond', 'E_cond', 'obj_task_setting', 'history_E_cond']
        interested_keys = [key for key in interested_keys if key in batch]
        
        calculate_loss_keys = None 
        
        if self.kine_diff:
            calculate_loss_keys = ['X']
        
        # i in range #
        for i in range(0, nn_bsz, self.microbatch):
            # print(f"batch_device: {batch['base_pts'].device}") ## base pts device ### # device # #
            # Eliminates the microbatch feature 
            assert i == 0
            assert self.microbatch == self.batch_size
            # micro = batch
            micro = {
                key: batch[key] for key in interested_keys
            }
            # micro_cond = cond # 
            ## micro-batch # base_pts; base_pts #
            last_batch = (i + self.microbatch) >= nn_bsz
            t, weights = self.schedule_sampler.sample(nn_bsz, self.device)

            training_losses_func = self.diffusion.training_losses
            
            if self.AE_Diff:
                if self.train_AE: # train AE and train Diff #
                    if self.kine_diff:
                        training_losses_func = self.diffusion.training_losses_AE
                    else:
                        training_losses_func = self.diffusion.training_losses_CtlTraj_AE
                else:
                    if self.kine_diff:
                        training_losses_func = self.diffusion.training_losses_AE_Diff
                    else:
                        training_losses_func = self.diffusion.training_losses_CtlTraj_AE_Diff # 

            # print(f"self.model: {self.model}, diffusion: {self.diffusion}") # autoencoding #
            losses = training_losses_func( ## get the losses from the diffusion model ##
                self.model,
                micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                t,  # [bs](int) sampled timesteps
                model_kwargs={'y': batch},
                calculate_loss_keys=calculate_loss_keys
                # dataset=self.data.dataset
            )
            # loss aware sampler #
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            
            loss = (losses["loss"] * weights).mean()
            
            # if self.args.nprocs > 1:
            #     torch.distributed.barrier()
            #     dist_util.reduce_mean(loss, self.args.nprocs)
                
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # self.mp_trainer.backward(loss)
            
            loss.backward()
            # self.opt.step()
    
    # forward_backward_trajectory_forcasting_diffusion_masked_futurecond
    def forward_backward_trajectory_forcasting_diffusion_masked_futurecond(self, batch):
        # self.mp_trainer.zero_grad()
        
        if self.debug:
            print(f"forward_backward_trajectory_forcasting_diffusion_masked_futurecond")
        
        self.opt.zero_grad()
        
        obj_pts = batch['obj_pts']
        
        # hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim #
        # obj_pos = batch['obj_pos']
        # obj_ornt = batch['obj_ornt']
        
        hand_pose = batch['masked_future_hand_qs']
        obj_pos = batch['masked_future_obj_transl']
        obj_ornt = batch['masked_future_obj_ornt']
        
        # expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous()
        
        future_hand_pose = batch['future_hand_pose']
        future_obj_pos = batch['future_obj_pos']
        future_obj_ornt = batch['future_obj_ornt']
        
        
        # tot_pose_feat = torch.cat(
        #     [ hand_pose, obj_pos, obj_ornt ], dim=-1
        # )
        
        if self.w_glb_traj_feat_cond: # masked future cond #
            # glb traj feat cond #
            glb_hand_pose = batch['glb_hand_pose']
            glb_obj_pos = batch['glb_obj_pos']
            glb_obj_ornt = batch['glb_obj_ornt']
            expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, glb_obj_pos.size(1), 1, 1).contiguous()
            tot_glb_pose_feat = torch.cat(
                [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1
            )
        else:
            tot_glb_pose_feat = None
            expanded_glb_obj_pts = None
        
        # two conditions: 1) whole trajectory; and 2) the history trajectory # # calculate the loss yes #
        calculate_loss_keys = None
        calculate_loss_keys = ['E']
        nn_bsz = obj_pts.shape[0]
        for i in range(0, nn_bsz, self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            
            X = obj_pts # nn bsz x nn pts x 3 
            # E is the hand pose and the obj pos and ornt 
            
            if self.diff_contact_sequence:
                future_contact_seq = batch['future_contact'] # nn_future_ws x nn_contact_dim #
                E = future_contact_seq
            else:
                if self.partial_hand_info:
                    E = future_hand_pose
                elif self.partial_obj_info:
                    E = torch.cat(
                        [ future_obj_pos, future_obj_ornt ], dim=-1
                    )
                elif self.partial_obj_pos_info:
                    E = future_obj_pos
                else:
                    
                    E = torch.cat(
                        [ future_hand_pose, future_obj_pos, future_obj_ornt ], dim=-1
                    )
                    
            
            
            
            micro = {'X': X, 'E': E}
            
            # history_E_cond, X_cond, E_cond
            if self.task_cond:
                X_cond = obj_pts
                E_cond = torch.cat(
                    [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1
                )
                history_E_cond = torch.cat(
                    [ hand_pose, obj_pos, obj_ornt ], dim=-1
                )
                if self.partial_hand_info:
                    history_E_cond[..., hand_pose.size(-1): ] = 0.0
                elif self.partial_obj_info:
                    history_E_cond[..., : hand_pose.size(-1)] = 0.0 # 
                elif self.partial_obj_pos_info:
                    history_E_cond[..., : hand_pose.size(-1)] = 0.0
                    history_E_cond[..., -4: ] = 0.0
                    
                ## NOTE ##
                ###### only use one single frame for the condition #####
                history_E_cond = history_E_cond[:, -1:, :]
                    
                micro.update(
                    {
                        'X_cond': X_cond, 'E_cond': E_cond, 'history_E_cond': history_E_cond
                    }
                )
                
                if self.debug:
                    for key in micro:
                        print(f"key: {key}, shape: {micro[key].shape}")
                
                if self.w_history_window_index:
                    if self.w_masked_future_cond:
                        factorized_history_window_info = batch['factorized_future_window_info']
                    else:
                        factorized_history_window_info = batch['factorized_history_window_info']
                    
                    ## NOTE ##
                    factorized_history_window_info = factorized_history_window_info[:, -1: ]
                    
                    
                    micro.update(
                        {
                            'history_E_window_idx': factorized_history_window_info
                        }
                    )
            
            
            t, weights = self.schedule_sampler.sample(nn_bsz, self.device)
            
            training_losses_func = self.diffusion.training_losses
            
            if self.AE_Diff:
                if self.train_AE: # train AE and train Diff #
                    # if self.kine_diff:
                    training_losses_func = self.diffusion.training_losses_AE
                    # else:
                        
                    #     training_losses_func = self.diffusion.training_losses_CtlTraj_AE
                else:
                    # if self.kine_diff:
                    if self.debug:
                        print(f"Training diff")
                    training_losses_func = self.diffusion.training_losses_AE_Diff
                    # else:
                    #     training_losses_func = self.diffusion.training_losses_CtlTraj_AE_Diff # 

            # print(f"self.model: {self.model}, diffusion: {self.diffusion}") # autoencoding #
            losses = training_losses_func( ## get the losses from the diffusion model ##
                self.model,
                micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                t,  # [bs](int) sampled timesteps
                model_kwargs={'y': batch},
                calculate_loss_keys=calculate_loss_keys
            )
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            
            loss = (losses["loss"] * weights).mean()
            
            # if self.args.nprocs > 1:
            #     torch.distributed.barrier()
            #     dist_util.reduce_mean(loss, self.args.nprocs)
                
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # self.mp_trainer.backward(loss)
            
            loss.backward()
            # self.opt.step()
        
    
    
    def forward_backward_trajectory_forcasting_diffusion(self, batch):
        # self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        
        obj_pts = batch['obj_pts'] # object points --- nn_bsz x nn_pts x 3 # 
        hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim #  # 
        obj_pos = batch['obj_pos']
        obj_ornt = batch['obj_ornt']
        expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous() 
        
        future_hand_pose = batch['future_hand_pose']
        future_obj_pos = batch['future_obj_pos']
        future_obj_ornt = batch['future_obj_ornt']
        
        
        if self.debug:
            for   key in batch:
                val = batch[key]
                try:    
                    print(f"key: {key}, shape: {val.size()}")
                except:
                    pass
        # tot_pose_feat = torch.cat(
        #     [ hand_pose, obj_pos, obj_ornt ], dim=-1
        # )
        
        if self.w_glb_traj_feat_cond: 
            # glb traj feat cond #
            glb_hand_pose = batch['glb_hand_pose']
            glb_obj_pos = batch['glb_obj_pos']
            glb_obj_ornt = batch['glb_obj_ornt']
            # expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, glb_obj_pos.size(1), 1, 1).contiguous()
            # tot_glb_pose_feat = torch.cat(
            #     [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1 # get the glbjhand pose 
            # )
        else:
            tot_glb_pose_feat = None
            expanded_glb_obj_pts = None
        
        # two conditions: 1) whole trajectory; and 2) the history trajectory #
        calculate_loss_keys = None
        calculate_loss_keys = ['E']
        nn_bsz = obj_pts.shape[0]
        for i in range(0, nn_bsz, self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            
            X = obj_pts # nn bsz x nn pts x 3 
            # E is the hand pose and the obj pos and ornt 
            
            if self.diff_contact_sequence:
                future_contact_seq = batch['future_contact'] # nn_future_ws x nn_contact_dim
                E = future_contact_seq
            else:
                if self.partial_hand_info:
                    E = future_hand_pose
                elif self.partial_obj_info:
                    E = torch.cat(
                        [ future_obj_pos, future_obj_ornt ], dim=-1
                    )
                elif self.partial_obj_pos_info:
                    if self.debug:
                        print(f"Partial obj pos info")
                    E = future_obj_pos
                else:
                    
                    E = torch.cat(
                        [ future_hand_pose, future_obj_pos, future_obj_ornt ], dim=-1
                    )
                    
            
            
            
            micro = {'X': X, 'E': E}
            
            # history_E_cond, X_cond, E_cond
            if self.task_cond:
                X_cond = obj_pts
                E_cond = torch.cat(
                    [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1
                )
                
                if self.partial_hand_info:
                    history_E_cond = hand_pose
                elif self.partial_obj_info:
                    history_E_cond = torch.cat(
                        [ obj_pos, obj_ornt ], dim=-1
                    )
                elif self.partial_obj_pos_info:
                    history_E_cond = obj_pos
                else:
                    history_E_cond = torch.cat(
                        [ hand_pose, obj_pos, obj_ornt ], dim=-1
                    )
                    
                ## NOTE ##
                ###### only use one single frame for the condition #####
                # history_E_cond = history_E_cond[:, -1:, :]
                
                micro.update(
                    {
                        'X_cond': X_cond, 'E_cond': E_cond, 'history_E_cond': history_E_cond
                    }
                )
                
                if self.w_history_window_index:
                    factorized_history_window_info = batch['factorized_history_window_info']
                    
                    
                    ## NOTE ##
                    # factorized_history_window_info = 
                    # factorized_history_window_info = factorized_history_window_info[:, -1:]
                    
                    micro.update(
                        {
                            'history_E_window_idx': factorized_history_window_info
                        }
                    )
                if self.debug:
                    for key in micro:
                        print(f"key: {key}, shape: {micro[key].shape}")
                
            if self.debug:
                # print(f"[shape to features] E:  {E.size()}")
                for key in micro:
                    try:
                        print(f"key: {key} val: {micro[key].size()}")
                    except:
                        continue
                
            
            t, weights = self.schedule_sampler.sample(nn_bsz, self.device)
            
            training_losses_func = self.diffusion.training_losses
            
            if self.AE_Diff:
                if self.train_AE: # train AE and train Diff #
                    # if self.kine_diff:
                    training_losses_func = self.diffusion.training_losses_AE
                    # else:
                        
                    #     training_losses_func = self.diffusion.training_losses_CtlTraj_AE
                else:
                    # if self.kine_diff:
                    if self.debug:
                        print(f"Training diff")
                    training_losses_func = self.diffusion.training_losses_AE_Diff #
                    # else:
                    #     training_losses_func = self.diffusion.training_losses_CtlTraj_AE_Diff # 

            # print(f"self.model: {self.model}, diffusion: {self.diffusion}") # autoencoding #
            losses = training_losses_func( ## get the losses from the diffusion model ##
                self.model,
                micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                t,  # [bs](int) sampled timesteps
                model_kwargs={'y': batch},
                calculate_loss_keys=calculate_loss_keys
            )
            
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            
            loss = (losses["loss"] * weights).mean()
            
            # if self.args.nprocs > 1:
            #     torch.distributed.barrier()
            #     dist_util.reduce_mean(loss, self.args.nprocs)
                
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            # self.mp_trainer.backward(loss)
            
            loss.backward()
            # self.opt.step()
        
    
        # 
    
    def forward_backward_traj_translations(self, batch):
        # self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        
        nn_bsz = batch['X'].shape[0]
        
        interested_keys = ['X', 'E', 'X_target', 'E_target', 'obj_task_setting']
        interested_keys = [key for key in interested_keys if key in batch]
        
        for i in range(0, nn_bsz, self.microbatch):
            # print(f"batch_device: {batch['base_pts'].device}") ## base pts device 
            # Eliminates the microbatch feature 
            assert i == 0
            assert self.microbatch == self.batch_size
            # micro = batch
            micro = {
                key: batch[key] for key in interested_keys
            }
            # micro_cond = cond # 
            ## micro-batch # base_pts; base_pts #
            last_batch = (i + self.microbatch) >= nn_bsz
            t, weights = self.schedule_sampler.sample(nn_bsz, self.device)

            # print(f"self.model: {self.model}, diffusion: {self.diffusion}")
            losses = self.diffusion.training_losses_traj_translations(
                self.model,
                micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                t,  # [bs](int) sampled timesteps
                model_kwargs={'y': batch},
                # dataset=self.data.dataset
            )
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            loss = (losses["loss"] * weights).mean()
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss.backward()
      
    
    #### trajs translations conds ####
    def forward_backward_traj_translations_cond(self, batch):
        # self.mp_trainer.zero_grad() # traj translations cond #
        self.opt.zero_grad()
        
        nn_bsz = batch['X'].shape[0]
        
        interested_keys = ['X', 'E', 'X_target', 'E_target', 'obj_task_setting', 'X_cond', 'E_cond', 'history_E_cond']
        interested_keys = [key for key in interested_keys if key in batch]
        
        for i in range(0, nn_bsz, self.microbatch):
            # print(f"batch_device: {batch['base_pts'].device}") ## base pts device 
            # Eliminates the microbatch feature 
            assert i == 0
            assert self.microbatch == self.batch_size
            # micro = batch
            micro = {
                key: batch[key] for key in interested_keys
            }
            # micro_cond = cond #
            ## micro-batch ##
            last_batch = (i + self.microbatch) >= nn_bsz
            t, weights = self.schedule_sampler.sample(nn_bsz, self.device)

            # print(f"self.model: {self.model}, diffusion: {self.diffusion}")
            losses = self.diffusion.training_losses_traj_translations_cond(
                self.model,
                micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                t,  # [bs](int) sampled timesteps
                model_kwargs={'y': batch},
                # dataset=self.data.dataset
            )
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )
            
            loss = (losses["loss"] * weights).mean()
            
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )
            loss.backward()
      
    
    def predict_traj_forcasting_pc_traj_encdec(self, batch):
        # self.mp_trainer.zero_grad()
        
        
        # obj pts #
        obj_pts = batch['obj_pts'] # object points --- nn_bsz x nn_pts x 3 ## # hand pose #
        hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim ## # hand pose #
        obj_pos = batch['obj_pos']
        obj_ornt = batch['obj_ornt']
        # expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous()
        
        
        # if self.w_history_window_index:
        #     history_window_index = batch['factorized_history_window_info']
        # else:
        #     history_window_index = None
        
        tot_pose_feat = torch.cat(
            [ hand_pose, obj_pos, obj_ornt ], dim=-1
        )
        
        
        rt_val_dict = self.model(obj_pts, tot_pose_feat, )
        
        return rt_val_dict
    
      
    
    def forward_backward_traj_forcasting_pc_traj_encdec(self, batch):
        # self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        
        # w_history_window_index #
        
        obj_pts = batch['obj_pts'] # object points --- nn_bsz x nn_pts x j3 # 
        # expanded_obj_pts = 
        hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim #  # 
        obj_pos = batch['obj_pos']
        obj_ornt = batch['obj_ornt']
        
        
        # if self.w_history_window_index:
        #     history_window_index = batch['factorized_history_window_info']
        # else:
        #     history_window_index = None
        
        # expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous() # expanded 
        # 
        
        tot_pose_feat = torch.cat(
            [ hand_pose, obj_pos, obj_ornt ], dim=-1
        )
        
        # if self.w_glb_traj_feat_cond: 
            
        #     glb_hand_pose = batch['glb_hand_pose']
        #     glb_obj_pos = batch['glb_obj_pos']
        #     glb_obj_ornt = batch['glb_obj_ornt']
        #     expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, glb_obj_pos.size(1), 1, 1).contiguous()
        #     tot_glb_pose_feat = torch.cat(
        #         [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1 # get the glbjhand pose 
        #     )
        # else:
        #     tot_glb_pose_feat = None
        #     expanded_glb_obj_pts = None
        
        
        # if self.w_history_window_index:
        #     # history_window_index
        #     rt_val_dict = self.model(obj_pts, tot_pose_feat, )
        # else:
        rt_val_dict = self.model(obj_pts, tot_pose_feat, )
        
        
        pred_hand_pose = rt_val_dict['hand_pose']
        pred_obj_pos = rt_val_dict['obj_pos']
        pred_obj_ornt = rt_val_dict['obj_ornt']
        
        decoded_pts = rt_val_dict['decoded_pts']
        
        
        
        # print(f"pred_hand_pose: {pred_hand_pose.size()}, pred_obj_pos: {pred_obj_pos.size()}, pred_obj_ornt: {pred_obj_ornt.size()}")
        
        # future_hand_pose = batch['future_hand_pose']
        # future_obj_pos = batch['future_obj_pos']
        # future_obj_ornt = batch['future_obj_ornt']
        
        # print(f"future_hand_pose: {future_hand_pose.size()}, future_obj_pos: {future_obj_pos.size()}, future_obj_ornt: {future_obj_ornt.size()}")
        
        loss_hand_pose = torch.mean(
           torch.sum( (pred_hand_pose - hand_pose) ** 2, dim=-1), 
        )
        loss_obj_pos = torch.mean(
            torch.sum ( (obj_pos  - pred_obj_pos) ** 2, dim=-1 )
        )
        loss_obj_ornt = torch.mean(
            torch.sum( (pred_obj_ornt - obj_ornt) ** 2, dim=-1 )
        )
        
        
        diff_pred_obj_pts_w_ori_pts = torch.sum(
            (decoded_pts.unsqueeze(1) - obj_pts.unsqueeze(2)) ** 2, dim=-1
        )
        minn_pred_to_gt, _ = torch.min(diff_pred_obj_pts_w_ori_pts, dim=-1)
        minn_gt_to_pred, _ = torch.min(diff_pred_obj_pts_w_ori_pts, dim=-2) # get the 
        loss_pts = (torch.mean(minn_pred_to_gt, dim=-1) + torch.mean(minn_gt_to_pred, dim=-1)) / 2
        loss_pts = loss_pts / 2.0
        loss_pts = loss_pts.mean()
        
        
        loss = loss_hand_pose + loss_obj_pos + loss_obj_ornt + loss_pts
        
        losses = {}
        losses['loss'] = loss
        
        
        log_loss_dict_ndiffusion(
          losses
        )
        # self.mp_trainer.backward(loss) # 
        
        loss.backward()
        
    
      
    def forward_backward_traj_forcasting(self, batch):
        # self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        
        
        
        obj_pts = batch['obj_pts'] # object points --- nn_bsz x nn_pts x 3 #
        hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim
        obj_pos = batch['obj_pos']
        obj_ornt = batch['obj_ornt']
        
        
        if self.w_history_window_index:
            history_window_index = batch['factorized_history_window_info']
        else:
            history_window_index = None
        
        expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous() #  
        
        #### partial hand info ####
        if self.partial_hand_info:
            tot_pose_feat = hand_pose
        elif self.partial_obj_info:
            tot_pose_feat = torch.cat(
                [ obj_pos, obj_ornt ], dim=-1
            )
        elif self.partial_obj_pos_info:
            tot_pose_feat = obj_pos
        else:
            tot_pose_feat = torch.cat(
                [ hand_pose, obj_pos, obj_ornt ], dim=-1
            )
        
        # tot_pose_feat = torch.cat(
        #     [ hand_pose, obj_pos, obj_ornt ], dim=-1
        # )
        
        if self.w_glb_traj_feat_cond: 
            
            if 'text_features' in batch:
                if self.debug:
                    print(f"Using text features")
                tot_glb_pose_feat = batch['text_features']
                expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, 1, 1, 1).contiguous()
            else:
            
                glb_hand_pose = batch['glb_hand_pose']
                glb_obj_pos = batch['glb_obj_pos']
                glb_obj_ornt = batch['glb_obj_ornt']
                expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, glb_obj_pos.size(1), 1, 1).contiguous()
                tot_glb_pose_feat = torch.cat(
                    [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1
                )
        else:
            tot_glb_pose_feat = None
            expanded_glb_obj_pts = None
            
        
        
        if self.debug:
            print(f"expanded_obj_pts: {expanded_obj_pts.size()}, tot_pose_feat: {tot_pose_feat.size()}, tot_glb_pose_feat: {tot_glb_pose_feat.size()}")
        if self.w_history_window_index:
            # history_window_index
            rt_val_dict = self.model(expanded_obj_pts, tot_pose_feat, tot_feat_feat=tot_glb_pose_feat, tot_obj_pts=expanded_glb_obj_pts, history_window_index=history_window_index)
        else:
            rt_val_dict = self.model(expanded_obj_pts, tot_pose_feat, tot_feat_feat=tot_glb_pose_feat, tot_obj_pts=expanded_glb_obj_pts)
        
        
        pred_hand_pose = rt_val_dict['hand_pose']
        pred_obj_pos = rt_val_dict['obj_pos']
        pred_obj_ornt = rt_val_dict['obj_ornt']
        
        # print(f"pred_hand_pose: {pred_hand_pose.size()}, pred_obj_pos: {pred_obj_pos.size()}, pred_obj_ornt: {pred_obj_ornt.size()}")
        
        future_hand_pose = batch['future_hand_pose'] # future hand pose #
        future_obj_pos = batch['future_obj_pos']
        future_obj_ornt = batch['future_obj_ornt']
        
        
        
        
        # print(f"future_hand_pose: {future_hand_pose.size()}, future_obj_pos: {future_obj_pos.size()}, future_obj_ornt: {future_obj_ornt.size()}")
        
        loss_hand_pose = torch.mean(
           torch.sum( (pred_hand_pose - future_hand_pose) ** 2, dim=-1 ), 
        )
        loss_obj_pos = torch.mean(
            torch.sum ( (future_obj_pos  - pred_obj_pos) ** 2, dim=-1 )
        )
        loss_obj_ornt = torch.mean(
            torch.sum( (pred_obj_ornt - future_obj_ornt) ** 2, dim=-1 )
        )
        
        ### get the partial hand info ###
        if self.partial_hand_info:
            loss = loss_hand_pose
        elif self.partial_obj_info:
            loss = loss_hand_pose + loss_obj_ornt
        elif self.partial_obj_pos_info:
            loss = loss_obj_pos
        else:
            loss = loss_hand_pose + loss_obj_pos + loss_obj_ornt
        
        losses = {}
        losses['loss'] = loss
        
        
        log_loss_dict_ndiffusion(
          losses
        )
        
        loss.backward()
        
        
        
    
            
    def predict_single_step(self, batch, use_t=None):
        # self.mp_trainer.zero_grad()
        # use_t is not None #
        # tot_samples = []
        # tot_targets = []
        
        # tot_dec_disp_e_along_normals = []
        # tot_dec_disp_e_vt_normals = []
        # tot_pred_joints_quant = []
        tot_samples = {key: [] for key in batch}
        
        nn_bsz = batch['X'].shape[0]
        interest_keys = ['X', 'E', 'X_cond', 'E_cond', 'obj_task_setting', 'history_E_cond']
        interest_keys = [key for key in interest_keys if key in batch]
        # add object type #
        for i in range(0, nn_bsz, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            # rhand_joints = micro['rhand_joints']
            # micro_cond = cond # add the object features 
            # last_batch = (i + self.microbatch) >= nn_bsz
            # t, weights = self.schedule_sampler.sample(micro['X'].shape[0], dist_util.dev())
            
            shape = {
                key: micro[key].shape for key in interest_keys
            }
            
            sample_fn = self.diffusion.p_sample_loop
            
            if self.AE_Diff:
                if self.train_AE:
                    sample_fn = self.diffusion.p_sample_loop_AE
                else:
                    sample_fn = self.diffusion.p_sample_loop_AE_Diff
            
            
            if self.AE_Diff and self.train_AE:
                # ret_encoded_feat
                samples = sample_fn(
                    self.ddp_model, 
                    shape,
                    noise=None,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                    progress=True,
                    use_t=use_t,
                    data=micro if (self.AE_Diff or (use_t is not None) or (self.task_cond or self.training_setting == 'trajectory_translations_cond')) else None,
                    ret_encoded_feat=True
                )
            else:
                samples = sample_fn(
                    self.ddp_model, 
                    shape,
                    noise=None,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                    progress=True,
                    use_t=use_t,
                    data=micro if (self.AE_Diff or (use_t is not None) or (self.task_cond or self.training_setting == 'trajectory_translations_cond')) else None
                )

            for key in samples:
                if key in tot_samples:
                    tot_samples[key].append(samples[key])
                else:
                    tot_samples[key] = [ samples[key] ]
            for key in micro:
                if key not in samples:
                    tot_samples[key].append(micro[key])

        print(f"tot_samples: {tot_samples.keys()}")
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
                print(f"key: {key}, shape: {tot_samples[key].shape}")
            except:
                continue
        return tot_samples

    

    # predict_single_step_from_shadow_target
    
    def predict_single_step_from_shadow_target(self, batch, use_t=None):
        
        
        tot_samples = {key: [] for key in batch}

        
        nn_bsz = batch['X'].shape[0] # batch #
        interest_keys = ['X', 'E']
        # 
        for i in range(0, nn_bsz, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            # ## micro batch ##
            # rhand_joints = micro['rhand_joints'] #
            # micro_cond = cond # micro cond and cond ##
            ## predict single step ##
            last_batch = (i + self.microbatch) >= nn_bsz
            t, weights = self.schedule_sampler.sample(micro['X'].shape[0], dist_util.dev())
            

            # shape = {
            #     key: micro[key].shape for key in micro
            # }
            
            shape = {
                key: micro[key].shape for key in interest_keys
            }
            
            ## sample from the model --- the target sample should be in the sahpe of micro['X'].shape ##
            # shape = micro['X'].shape

            sample_fn = self.diffusion.p_sample_loop_pcdguided
            samples = sample_fn(
                self.ddp_model, 
                shape,
                noise=None,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                progress=True,
                use_t=use_t,
                data=micro #  if use_t is not None else None
            )

            for key in samples:
                tot_samples[key].append(samples[key])
            for key in micro:
                if key not in interest_keys:
                    tot_samples[key].append(micro[key])

            
            
            
        print(f"tot_samples: {tot_samples.keys()}")
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
            except:
                continue
        return tot_samples



    # training loop #
    def predict_traj_forcasting(self, batch):
        # self.mp_trainer.zero_grad()
        
        
        # obj pts #
        obj_pts = batch['obj_pts'] # object points --- nn_bsz x nn_pts x 3 ## # hand pose #
        hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim ## # hand pose #
        obj_pos = batch['obj_pos']
        obj_ornt = batch['obj_ornt']
        expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous()
        
        
        if self.w_history_window_index:
            history_window_index = batch['factorized_history_window_info']
        else:
            history_window_index = None
        
        # tot_pose_feat = torch.cat(
        #     [ hand_pose, obj_pos, obj_ornt ], dim=-1
        # )
        
        #### partial hand info ####
        if self.partial_hand_info:
            tot_pose_feat = hand_pose
        elif self.partial_obj_info:
            tot_pose_feat = torch.cat(
                [ obj_pos, obj_ornt ], dim=-1
            )
        elif self.partial_obj_pos_info:
            tot_pose_feat = obj_pos
        else:
            tot_pose_feat = torch.cat(
                [ hand_pose, obj_pos, obj_ornt ], dim=-1
            )
        
        
        if self.w_glb_traj_feat_cond:
            
            if 'text_features' in batch:
                if self.debug:
                    print(f"Using text features")
                tot_glb_pose_feat = batch['text_features']
                expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, 1, 1, 1).contiguous()
            else:
                # glb hand pose #
                glb_hand_pose = batch['glb_hand_pose']
                glb_obj_pos = batch['glb_obj_pos']
                glb_obj_ornt = batch['glb_obj_ornt']
                expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, glb_obj_pos.size(1), 1, 1).contiguous()
                tot_glb_pose_feat = torch.cat(
                    [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1 # get the glbjhand pose ## glb hand pose # glb hand pose #
                )
        else:
            tot_glb_pose_feat = None
            expanded_glb_obj_pts = None
            
        if self.w_history_window_index:
            # history_window_index
            rt_val_dict = self.model(expanded_obj_pts, tot_pose_feat, tot_feat_feat=tot_glb_pose_feat, tot_obj_pts=expanded_glb_obj_pts, history_window_index=history_window_index)
        else:
            rt_val_dict = self.model(expanded_obj_pts, tot_pose_feat, tot_feat_feat=tot_glb_pose_feat, tot_obj_pts=expanded_glb_obj_pts)
        
        
        ## rt val dict #### rt val dict ## ## expanded obj pts ##
        # rt_val_dict = self.model(expanded_obj_pts, tot_pose_feat, tot_feat_feat=tot_glb_pose_feat, tot_obj_pts=expanded_glb_obj_pts)
        
        ## rt val dict = self.model() ## ## rt val dict ##
        
        
        # # loss # loss # loss #
        # obj_pts = batch['obj_pts'] # object points # --- nn_bsz x nn_pts x 3 #
        # # expanded_obj_pts #
        # hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim #
        # obj_pos = batch['obj_pos']
        # obj_ornt = batch['obj_ornt']
        # expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous() # expanded 
        # # 
        # tot_pose_feat = torch.cat(
        #     [ hand_pose, obj_pos, obj_ornt ], dim=-1
        # )
        # # rt val dict #
        # rt_val_dict = self.model(expanded_obj_pts, tot_pose_feat)
        return rt_val_dict
    
    
    def predict_traj_forcasting_diffusion(self, batch, use_t=None ):
        # self.mp_trainer.zero_grad()
        
        obj_pts = batch['obj_pts'] # object points --- nn_bsz x nn_pts x j3 # 
        # expanded_obj_pts = 
        hand_pose = batch['hand_pose'] # nn_bsz x nn_ts x nn_hand_pos_dim #  # 
        obj_pos = batch['obj_pos']
        obj_ornt = batch['obj_ornt']
        expanded_obj_pts = obj_pts.unsqueeze(1).repeat(1, hand_pose.size(1), 1, 1).contiguous() # expanded 
        # 
        
        future_hand_pose = batch['future_hand_pose']
        future_obj_pos = batch['future_obj_pos']
        future_obj_ornt = batch['future_obj_ornt']
        
        
        tot_pose_feat = torch.cat(
            [ hand_pose, obj_pos, obj_ornt ], dim=-1
        )
        
        if self.w_glb_traj_feat_cond: 
            
            glb_hand_pose = batch['glb_hand_pose']
            glb_obj_pos = batch['glb_obj_pos']
            glb_obj_ornt = batch['glb_obj_ornt']
            expanded_glb_obj_pts = obj_pts.unsqueeze(1).repeat(1, glb_obj_pos.size(1), 1, 1).contiguous()
            tot_glb_pose_feat = torch.cat(
                [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1 # get the glbjhand pose 
            )
        else:
            tot_glb_pose_feat = None
            expanded_glb_obj_pts = None
            
        tot_samples = {}
        
        # diff_contact_sequence #
        
        # two conditions: 1) whole trajectory; and 2) the history trajectory #
        calculate_loss_keys = None
        nn_bsz = obj_pts.shape[0]
        for i in range(0, nn_bsz, self.microbatch):
            assert i == 0
            assert self.microbatch == self.batch_size
            
            X = obj_pts # nn bsz x nn pts x 3 
            # E is the hand pose and the obj pos and ornt 
            
            
            
            if self.diff_contact_sequence:
                future_contact_seq = batch['future_contact'] # nn_future_ws x nn_contact_dim
                E = future_contact_seq
            else:
                if self.partial_hand_info:
                    E = future_hand_pose
                elif self.partial_obj_info:
                    E = torch.cat(
                        [ future_obj_pos, future_obj_ornt ], dim=-1
                    )
                elif self.partial_obj_pos_info:
                    
                    E = future_obj_pos
                else:
                    
                    E = torch.cat(
                        [ future_hand_pose, future_obj_pos, future_obj_ornt ], dim=-1
                    )
                # E = torch.cat(
                #     [ future_hand_pose, future_obj_pos, future_obj_ornt ], dim=-1
                # )
            
            
            micro = {'X': X, 'E': E}
            # t, weights = self.schedule_sampler.sample(nn_bsz, self.device)
            
            ###### task cond ######
            if self.task_cond:
                # task_cond #
                X_cond = obj_pts
                E_cond = torch.cat(
                    [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1
                )
                history_E_cond = torch.cat(
                    [ hand_pose, obj_pos, obj_ornt ], dim=-1
                )
                
                # E_cond = torch.cat(
                #     [ glb_hand_pose, glb_obj_pos, glb_obj_ornt ], dim=-1
                # )
                
                if self.partial_hand_info:
                    history_E_cond = hand_pose
                elif self.partial_obj_info:
                    history_E_cond = torch.cat(
                        [ obj_pos, obj_ornt ], dim=-1
                    )
                elif self.partial_obj_pos_info:
                    history_E_cond = obj_pos
                else:
                    history_E_cond = torch.cat(
                        [ hand_pose, obj_pos, obj_ornt ], dim=-1
                    )
                
                micro.update(
                    {
                        'X_cond': X_cond, 'E_cond': E_cond, 'history_E_cond': history_E_cond
                    }
                )
                
                if self.w_history_window_index:
                    factorized_history_window_info = batch['factorized_history_window_info']
                    micro.update(
                        {
                            'history_E_window_idx': factorized_history_window_info
                        }
                    )
            
            shape = {
                key: micro[key].shape for key in micro
            }
            
            
            if self.AE_Diff:
                if self.train_AE:
                    sample_fn = self.diffusion.p_sample_loop_AE
                else:
                    sample_fn = self.diffusion.p_sample_loop_AE_Diff
            
            
            if self.AE_Diff and self.train_AE:
                # ret_encoded_feat
                samples = sample_fn(
                    self.ddp_model, 
                    shape,
                    noise=None,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                    progress=True,
                    use_t=use_t,
                    data=micro if (self.AE_Diff or (use_t is not None) or (self.task_cond or self.training_setting == 'trajectory_translations_cond')) else None,
                    ret_encoded_feat=True
                )
            else:
                samples = sample_fn(
                    self.ddp_model, 
                    shape,
                    noise=None,
                    clip_denoised=False,
                    denoised_fn=None,
                    cond_fn=None,
                    model_kwargs=None,
                    progress=True,
                    use_t=use_t,
                    data=micro if (self.AE_Diff or (use_t is not None) or (self.task_cond or self.training_setting == 'trajectory_translations_cond')) else None
                )

            for key in samples:
                if key in tot_samples:
                    tot_samples[key].append(samples[key])
                else:
                    tot_samples[key] = [ samples[key] ]
            for key in micro:
                if key not in samples:
                    tot_samples[key].append(micro[key])

        print(f"tot_samples: {tot_samples.keys()}")
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
                print(f"key: {key}, shape: {tot_samples[key].shape}")
            except:
                continue
        return tot_samples
    
    
    
    def predict_from_data(self, batch, use_t=None):

        
        tot_samples = {}
        
        # for batch in tqdm(self.data): # from the data #
            
        for k in batch: # to the torch.Tensor #
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
            elif isinstance(batch[k], list):
                # batch[k] = [ subval.to(self.device) for subval in batch[k] ]
                batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
            else:
                batch[k] = batch[k]
        
        # call the function predict from data #
        use_t = use_t 
        
        ## data predict single step ## 
        if self.training_setting == 'trajectory_forcasting_diffusion':
            samples = self.predict_traj_forcasting_diffusion(batch, use_t=use_t)
        elif self.training_setting == 'trajectory_encdec':
            samples = self.predict_traj_forcasting_pc_traj_encdec(batch)
        elif self.training_setting == 'trajectory_forcasting':
            samples = self.predict_traj_forcasting(batch)
        else:
            samples = self.predict_single_step(batch, use_t=use_t)
        
        for key in samples:
            if key not in tot_samples:
                tot_samples[key] = [samples[key]]
            else:
                tot_samples[key].append(samples[key])
            # 
            # break
            
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0)
            except:
                pass
        return tot_samples

    
    
    
    def predict_from_data_from_shadow_target(self, use_t=None):

        
        tot_samples = {}
        
        for batch in tqdm(self.data):
            
            for k in batch: # to the torch.Tensor #
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
                elif isinstance(batch[k], list):
                    # batch[k] = [ subval.to(self.device) for subval in batch[k] ]
                    batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
                else:
                    batch[k] = batch[k]
            
            # call the function predict from data #
            use_t = use_t
            
            ## data predict single step ## # data predict single step ##
            samples = self.predict_single_step_from_shadow_target(batch, use_t=use_t)
            
            for key in samples:
                if key not in tot_samples:
                    tot_samples[key] = [samples[key]]
                else:
                    tot_samples[key].append(samples[key])
            
            break
            
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0)
            except:
                pass
        return tot_samples
        

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def prev_ckpt_file_name(self):
        prev_step = self.step + self.resume_step - self.save_interval
        return f"model{(prev_step):09d}.pt" #
    
    def prev_opt_file_name(self):
        prev_step = self.step + self.resume_step - self.save_interval
        return f"opt{(prev_step):09d}.pt" #
        
    
    # def prev_samples_file_name(self):
    #     prev_step = self.step + self.resume_step - self.save_interval
    #     return f"model{(prev_step):09d}.pt" #
        

    def ckpt_file_name(self):
        return f"model{(self.step+self.resume_step):09d}.pt"

    def samples_file_name(self):
        return f"samples{(self.step+self.resume_step):09d}.npy"

    def save_samples(self, samples):
        
        sv_samples_fn = self.samples_file_name()
        sv_samples_fn = os.path.join(self.save_dir, sv_samples_fn)
        logger.log(f"saving samples to {sv_samples_fn}...")
        
        np.save(sv_samples_fn, samples)
        
        
    def save(self):
        def save_checkpoint(params):
            # if self.args.finetune_with_cond:  # 
            #     state_dict = self.mp_trainer.model.state_dict()
            # else:
            #     state_dict = self.mp_trainer.master_params_to_state_dict(params)

            state_dict = self.model.state_dict()
            # Do not save CLIP weights
            clip_weights = [e for e in state_dict.keys() if e.startswith('clip_model.')]
            for e in clip_weights:
                del state_dict[e]

            
            # logger.log(f"saving model...") #
            
            filename = self.ckpt_file_name()
            
            prev_filename = self.prev_ckpt_file_name()
            prev_model_sv_fn = os.path.join(self.save_dir, prev_filename)
            if os.path.exists(prev_model_sv_fn):
                os.system(f"rm {prev_model_sv_fn}")
                logger.log(f"removing previous model from {prev_model_sv_fn}") # remove the previously saved model # 
            
            
            model_sv_fn = os.path.join(self.save_dir, filename)
            logger.log(f"saving model to {model_sv_fn}...")
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(None)
        
        
        prev_opt_name = self.prev_opt_file_name() # prev #
        prev_opt_file_name = os.path.join(self.save_dir, prev_opt_name)
        if os.path.exists(prev_opt_file_name):
            os.system(f"rm {prev_opt_file_name}")
            logger.log(f"removing previous opt from {prev_opt_file_name}")

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
            
    ## use the smaples to anchor cloest result ## # use the samples to anchor cloes result #
    ## 


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


def log_loss_dict_ndiffusion( losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
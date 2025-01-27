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
        
        ### TODO: you can jsut pass `None` as the diffusion here ###
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
        
        # self.
        self.kine_diff  = args.training.kine_diff
        ## if we have task_cond then weshould ## ##
        self.task_cond = args.training.task_cond
        
        self.AE_Diff = args.training.AE_Diff
        self.train_AE = args.training.train_AE
        self.train_Diff = args.training.train_Diff
        # 
        self.cond_diff_allparams = args.training.cond_diff_allparams
        
        
        #### getting related configs #### ##

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
        
        
        
        print(f"save_dirP {self.save_dir}")
        
        
        
        
        trainable_params = []
        trainable_params += list(self.model.parameters())
        self.opt = AdamW(
            trainable_params, lr=self.lr, weight_decay=self.weight_decay
        )


        # if self.task_cond:
            
        #     if self.AE_Diff:
        #         # trainable_params = []
        #         # # trainable_params += list(self.model.cond_input_process_pc.parameters())
        #         # # trainable_params += list(self.model.cond_input_process_feat.parameters())
        #         # trainable_params += list(self.model.parameters())
                
        #         # ### if we only have the trainable parameters ### #
        #         trainable_params = []
                
        #         # trainable_params += list(self.model.positional_encoder_feat.parameters())
        #         # trainable_params += list(self.model.transformer_encoder_feat.parameters())
        #         # trainable_params += list(self.model.time_embedder.parameters())
        #         # trainable_params += list(self.model.transformer_with_timesteps_encoder_feat.parameters())
        #         # trainable_params += list(self.model.pc_latent_processing.parameters())
        #         # trainable_params += list(self.model.feat_latent_processing.parameters())
                
        #         trainable_params += list(self.model.cond_input_process_pc.parameters())
        #         trainable_params += list(self.model.cond_input_process_feat.parameters())
        #         trainable_params += list(self.model.positional_encoder.parameters())
        #         trainable_params += list(self.model.transformer_encoder.parameters())
        #         trainable_params += list(self.model.time_embedder.parameters())
        #         trainable_params += list(self.model.transformer_with_timesteps_encoder.parameters())
                
        #         self.opt = AdamW(
        #             trainable_params, lr=self.lr, weight_decay=self.weight_decay
        #         ) 
        #         # self.args.training.tracking_ctl_diff #
        #     elif self.args.training.tracking_ctl_diff:
        #         # cond diff all parameters? #
                
        #         # 
        #         if self.cond_diff_allparams:
        #             trainable_params = []
        #             trainable_params += list(self.model.parameters())
        #             self.opt = AdamW(
        #                 trainable_params, lr=self.lr, weight_decay=self.weight_decay
        #             ) # cond 
        #         else:
        #             self.opt = AdamW(
        #                 list(self.model.cond_processing.parameters()) + list(self.model.transformer_encoder_cond.parameters()), lr=self.lr, weight_decay=self.weight_decay
        #             )
        #     else:
        #         self.opt = AdamW(
        #             self.model.cond_processing.parameters(), lr=self.lr, weight_decay=self.weight_decay
        #         )
                
        # else:
        #     if self.AE_Diff and self.train_Diff and (not self.train_AE):
        #         # if self.train_Diff:
        #         if self.kine_diff:
        #             trainable_params = []
        #             trainable_params += list(self.model.positional_encoder_feat.parameters())
        #             trainable_params += list(self.model.transformer_encoder_feat.parameters())
        #             trainable_params += list(self.model.time_embedder.parameters())
        #             trainable_params += list(self.model.transformer_with_timesteps_encoder_feat.parameters())
        #             trainable_params += list(self.model.pc_latent_processing.parameters())
        #             trainable_params += list(self.model.feat_latent_processing.parameters())
        #         else:
        #             trainable_params = []
        #             trainable_params += list(self.model.positional_encoder.parameters())
        #             trainable_params += list(self.model.transformer_encoder.parameters())
        #             trainable_params += list(self.model.time_embedder.parameters())
        #             trainable_params += list(self.model.transformer_with_timesteps_encoder.parameters())
                
        #         self.opt = AdamW(
        #             trainable_params, lr=self.lr, weight_decay=self.weight_decay
        #         )
        #     else: #
        #         self.opt = AdamW(
        #             self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        #         )
        
        # # ## TODO: the training resume settings ##
        # # if self.resume_step and not args.not_load_opt:
        # #     self._load_optimizer_state()

        # self.device = torch device #
        self.device = torch.device('cuda')

        self.schedule_sampler_type = 'uniform'
        # self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False  # if self.args.nprocs == 1 else True # # if self.args.nprocs # # self.args.nprocs #
        self.ddp_model = model
        
    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict:
            print(f"k: {k}")
            if k in model_dict:
                if ori_dict[k].shape == model_dict[k].shape:
                    v = ori_dict[k]
                    part_dict[k] = v
                    tot_params_n += 1

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
            
            ### load the ckpt ###
            self.safe_load_ckpt(self.model, 
                                    dist_util.load_state_dict(
                                        resume_checkpoint, map_location=dist_util.dev()
                                    )
                                )

    def _load_optimizer_state(self):
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
        
        
        for epoch in range(self.num_epochs):
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
                
                self.run_step(batch) # run step 
                if self.step % self.log_interval == 0:
                    loss_dict = logger.get_current().name2val
                    print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, loss_dict["loss"]))
                    
                ## how to train a good policy ##
                if self.step > 0 and self.step % self.save_interval == 0:
                    self.save()

                    # predict from data #
                    self.model.eval()
                    tot_samples = self.predict_from_data(batch)
                    
                    
                    notarget = False
                    if self.training_setting == 'trajectory_translations':
                        notarget = True 
                    print(f"tot_samples: {tot_samples.keys()}")
                    
                    
                    tot_samples['E'] = tot_samples['E'].unsqueeze(1).unsqueeze(1)
                    tot_samples['X'] = tot_samples['E'].clone()
                    
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

    # the next is predict from the use t's data ## predict from the use t data # # predict from # 
    # perturb the existing data ## target data # shadow data ##
    def predict_from_shadow_target_data(self, ): # # 
        i_batch = 0
        for batch in tqdm(self.data): ## get the batch ## ## 
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
                
                # samples are related to batch #
                tot_samples = self.predict_from_data(batch, use_t=use_t)
                notarget = False
                if self.training_setting == 'trajectory_translations':
                    notarget = True 
                
                # tot_samples #
                
                ## ## 
                ## get the total samples ##
                tot_samples['E'] = tot_samples['E'].unsqueeze(1).unsqueeze(1)
                tot_samples['X'] = tot_samples['E'].clone()
                
                tot_samples = self.data.dataset.inv_scale_data_v2(tot_samples, data_nm=batch['data_nm'], notarget=notarget) # inv scale the data
                
                ## resave the tot samples ##
                tot_samples = {
                    key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in tot_samples.items()
                } 
                
                ## reseave the tot samples ##
                input_batch = batch
                inv_sacled_input_batch = self.data.dataset.inv_scale_data_v2(input_batch, data_nm=batch['data_nm'])
                inv_sacled_input_batch = {
                    key: val.cpu().detach().numpy() if isinstance(val, torch.Tensor) else val for key, val in inv_sacled_input_batch.items()
                }
                
                
                tot_samples_sv_dict = {
                    'samples': tot_samples, #samples and samples ##
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
        
        ### training setting ###
        if self.training_setting == 'trajectory_translations':
            self.forward_backward_traj_translations(batch)
        elif self.training_setting == 'trajectory_translations_cond':
            self.forward_backward_traj_translations_cond(batch)
        else:
            self.forward_backward(batch)
        # self.mp_trainer.optimize(self.opt)
        self.opt.step()
        self._anneal_lr()
        self.log_step()


    # forward backward # forward backward #
    def forward_backward(self, batch):
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
            # micro = batch # micro batch #
            micro = {
                key: batch[key] for key in interested_keys
            }
            
            task_cond = {
                'X': micro['X_cond'], 'E': micro['E_cond'], 'history_E': micro['history_E_cond']
            }
            
            last_batch = (i + self.microbatch) >= nn_bsz
            
            decoded_acts = self.model(task_cond) # nn_bsz x nn_hand_dofs #
            # decoded acts #
            
            gt_acts = micro['E'] # # micro E --- nn_bsz x nn_pts x nn_ts x nn_hand_dof ##
            gt_acts = gt_acts.contiguous().view(gt_acts.size(0), decoded_acts.size(-1)).contiguous() # gt acts for decoding #
            
            decoded_loss = torch.sum(
                (gt_acts - decoded_acts) ** 2, dim=-1 ## nn_bszx 
            )
            decoded_loss = decoded_loss.mean() # get the average loss across all batches #
            
            
            # # t, weights = self.schedule_sampler.sample(nn_bsz, self.device)

            # # training_losses_func = self.diffusion.training_losses
            
            # # if self.AE_Diff:
            # #     if self.train_AE: # train AE and train Diff #
            # #         if self.kine_diff:
            # #             training_losses_func = self.diffusion.training_losses_AE
            # #         else:
            # #             training_losses_func = self.diffusion.training_losses_CtlTraj_AE
            # #     else:
            # #         if self.kine_diff:
            # #             training_losses_func = self.diffusion.training_losses_AE_Diff
            # #         else:
            # #             training_losses_func = self.diffusion.training_losses_CtlTraj_AE_Diff # 

            # # print(f"self.model: {self.model}, diffusion: {self.diffusion}") # autoencoding #
            # # losses = training_losses_func( ## get the losses from the diffusion model ##
            # #     self.model,
            # #     micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
            # #     t,  # [bs](int) sampled timesteps
            # #     model_kwargs={'y': batch},
            # #     calculate_loss_keys=calculate_loss_keys
            # #     # dataset=self.data.dataset
            # # )
            # # loss aware sampler #
            # if isinstance(self.schedule_sampler, LossAwareSampler):
            #     self.schedule_sampler.update_with_local_losses(
            #         t, losses["loss"].detach()
            #     )
            
            losses = {}
            losses['loss'] = decoded_loss
            log_loss_dict(
                {k: v for k, v in losses.items()}
            )
            
            
            loss = decoded_loss
            
            
            loss.backward()
    
    
    #### -> samples wiht the key 'E' <-> the sampled/decoded next step actions ####
    def predict_single_step(self, batch, use_t=None):
        
        
        tot_samples = {key: [] for key in batch}
        
        nn_bsz = batch['X'].shape[0] # batch #
        interest_keys = ['X', 'E', 'X_cond', 'E_cond', 'obj_task_setting', 'history_E_cond']
        interest_keys = [key for key in interest_keys if key in batch]
        # 
        
        samples = {
            'E': []
        }
        
        for i in range(0, nn_bsz, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            
            task_cond = {
                'X': micro['X_cond'], 'E': micro['E_cond'], 'history_E': micro['history_E_cond']
            }
            
            decoded_acts = self.model(task_cond) # nn_bsz x nn_hand_dofs #
            # decoded acts #
            
            # gt_acts = micro['E'] # # micro E --- nn_bsz x nn_pts x nn_ts x nn_hand_dof ##
            # gt_acts = gt_acts.contiguous().view(gt_acts.size(0), decoded_acts.size(-1)).contiguous() # gt acts for decoding #
            
            # decoded_loss = torch.sum(
            #     (gt_acts - decoded_acts) ** 2, dim=-1 ## nn_bszx 
            # )
            # decoded_loss = decoded_loss.mean() # get the average loss across all batches #
            
            # shape = {
            #     key: micro[key].shape for key in interest_keys
            # }
            
            # ## sample from the model --- the target sample should be in the sahpe of micro['X'].shape ##
            # # shape = micro['X'].shape
            
            # # 

            # sample_fn = self.diffusion.p_sample_loop
            
            # if self.AE_Diff:
            #     if self.train_AE:
            #         sample_fn = self.diffusion.p_sample_loop_AE
            #     else: # 
            #         sample_fn = self.diffusion.p_sample_loop_AE_Diff
            
            # samples = sample_fn(
            #     self.ddp_model, 
            #     shape,
            #     noise=None,
            #     clip_denoised=False,
            #     denoised_fn=None,
            #     cond_fn=None,
            #     model_kwargs=None,
            #     progress=True,
            #     use_t=use_t,
            #     data=micro if (self.AE_Diff or (use_t is not None) or (self.task_cond or self.training_setting == 'trajectory_translations_cond')) else None
            # )

            # for key in samples:
            #     tot_samples[key].append(samples[key])
            # for key in micro:
            #     if key not in samples:
            #         tot_samples[key].append(micro[key])
            samples['E'].append(decoded_acts)
            
            for key in samples:
                if key not in tot_samples:
                    tot_samples[key] = samples[key]
                else:
                    tot_samples[key] += samples[key]
            

            
        print(f"tot_samples: {tot_samples.keys()}")
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
                print(f"key: {key}, shape: {tot_samples[key].shape}")
            except:
                continue
        
        tot_samples['X'] = tot_samples['E']    
        
        return tot_samples


        
    
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


    
    
    def predict_from_data(self, batch, use_t=None):

        
        tot_samples = {}
        
        
        for k in batch:
            if isinstance(batch[k], torch.Tensor):
                batch[k] = batch[k].to(self.device)
            elif isinstance(batch[k], list):
                # batch[k] = [ subval.to(self.device) for subval in batch[k] ]
                batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
            else:
                batch[k] = batch[k]
        
        
        use_t = use_t
        
        
        samples = self.predict_single_step(batch, use_t=use_t)
        
        for key in samples:
            if key not in tot_samples:
                tot_samples[key] = [samples[key]]
            else:
                tot_samples[key].append(samples[key])
            ## got total samples ##
            
        for key in tot_samples:
            try:
                tot_samples[key] = torch.cat(tot_samples[key], dim=0)
            except:
                pass
        return tot_samples

    
    ### predict from data ###
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
            
            
            # if os.path.exists(prev_model_sv_fn):
            #     os.system(f"rm {prev_model_sv_fn}")
            #     logger.log(f"removing previous model from {prev_model_sv_fn}") # remove the previously saved model # 
            
            
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


def log_loss_dict(losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        # for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
        #     quartile = int(4 * sub_t / diffusion.num_timesteps)
        #     logger.logkv_mean(f"{key}_q{quartile}", sub_loss)

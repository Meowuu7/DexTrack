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

# to the real values # 
# to the real values #


class TrainLoop:
    def __init__(self, args, model, diffusion, data):
        self.args = args
        self.dataset = args.dataset.dataset_type ## train the dataset-type #
        # self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion


        self.data = data
        self.batch_size = args.training.batch_size
        self.microbatch = args.training.batch_size
        self.lr = args.training.lr
        self.log_interval = args.training.log_interval
        self.save_interval = args.training.save_interval
        self.resume_checkpoint = args.training.resumd_checkooit_pc
        self.use_fp16 = False  
        self.fp16_scale_growth = 1e-3
        self.weight_decay = args.training.weight_decay
        self.lr_anneal_steps = args.training.lr_anneal_steps
        #### getting related configs ####

        self.step = 0
        # self.resume_step = 0
        self.resume_step = False
        self.global_batch = self.batch_size # 
        self.num_steps = args.training.num_steps # 
        self.num_epochs = self.num_steps // len(self.data) + 1
        
        print(f"num_epochs: {self.num_epochs}, num_steps: {self.num_steps}")

        self.sync_cuda = torch.cuda.is_available()
        
        ## === resume the checkpoint === ##
        self._load_and_sync_parameters() 


        self.save_dir = args.save_dir
        # self.overwrite = args.overwrite

        self.opt = AdamW( ## 
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        # ## TODO: the training resume settings ##
        # if self.resume_step and not args.not_load_opt:
        #     self._load_optimizer_state()


        self.device = torch.device('cuda')

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None

        self.use_ddp = False  # if self.args.nprocs == 1 else True
        self.ddp_model = model
        
    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict:
            # if self.args.resume_diff:
            if k in model_dict:
                v = ori_dict[k]
                part_dict[k] = v
                tot_params_n += 1
            # else:
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
        
        ### run loop ###
        for epoch in range(self.num_epochs):
            # print(f'Starting epoch {epoch}')
            for batch in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
                    break
                
                # batch cur
                for k in batch: 
                    # print(f"k: {k}, shape: {batch[k].size()}")
                    if isinstance(batch[k], torch.Tensor):
                        batch[k] = batch[k].to(self.device)
                    elif isinstance(batch[k], list):
                        batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
                    else:
                        batch[k] = batch[k]
                
                ## run the step ## ## run the step ##
                self.run_step(batch)
                
                if self.step % self.log_interval == 0:
                    loss_dict = logger.get_current().name2val
                    print('step[{}]: loss[{:0.5f}]'.format(self.step+self.resume_step, loss_dict["loss"]))
                    for k in loss_dict:
                        v = loss_dict[k]
                        if k in ['rel_pred_loss', 'dist_pred_loss', 'dec_e_along_normals_loss', 'dec_e_vt_normals_loss', 'joints_pred_loss', 'jts_pred_loss', 'jts_latent_denoising_loss', 'basejtsrel_pred_loss', 'basejtsrel_latent_denoising_loss', 'basejtse_along_normals_pred_loss', 'basejtse_vt_normals_pred_loss', 'basejtse_latent_denoising_loss', "KL_loss", "avg_joints_pred_loss", "basejtrel_denoising_loss", "avgjts_denoising_loss"]: ## avg_joints_pred_loss # avg joints pred loss # 
                            print(f"\t{k}: {loss_dict[k].mean().item() if isinstance(loss_dict[k], torch.Tensor) else loss_dict[k]}")
                            
                        ## ##
                        # if k in ['step', 'samples'] or '_q' in k: # step samples #
                        #     continue
                        # else:
                        #     self.train_platform.report_scalar(name=k, value=v, iteration=self.step, group_name='Loss') #


                if self.step % self.save_interval == 0:
                    self.save()
                    # if self.args.nprocs > 1:
                    #     self.model.module.eval()
                    # else:
                    #     self.model.eval()
                    # self.evaluate()
                    # if self.args.nprocs > 1:
                    #     self.model.module.train()
                    # else:
                    #     self.model.train()
                    
                    self.model.eval()
                    tot_samples = self.predict_from_data()
                    
                    closest_training_data = self.data.dataset.get_closest_training_data(tot_samples['X'])
                    closest_training_data = {
                        'X': closest_training_data
                    }
                    
                    tot_samples_np = {key: val.cpu().detach().numpy() for key, val in tot_samples.items()} # project to the numpy array #
                    
                    #### get the tot samples sv dict ####
                    ## get the tot samples sv dict ##
                    tot_samples_sv_dict = {
                        'samples': tot_samples_np,
                        'closest_training_data': closest_training_data
                    }
                    
                    ### TODO: the data scaling for the action data ###

                    
                    self.save_samples(tot_samples_sv_dict)

                self.step += 1


    def evaluate(self):
        if not self.args.eval_during_training:
            return


    def run_step(self, batch): 
        self.forward_backward(batch)
        # self.mp_trainer.optimize(self.opt)
        self.opt.step()
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch): # 
        # self.mp_trainer.zero_grad()
        self.opt.zero_grad()
        
        nn_bsz = batch['X'].shape[0]
        
        for i in range(0, nn_bsz, self.microbatch):
            # print(f"batch_device: {batch['base_pts'].device}") ## base pts device 
            # Eliminates the microbatch feature 
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            # micro_cond = cond # 
            ## micro-batch # base_pts; base_pts #
            last_batch = (i + self.microbatch) >= nn_bsz
            t, weights = self.schedule_sampler.sample(nn_bsz, self.device)

            # print(f"self.model: {self.model}, diffusion: {self.diffusion}")
            losses = self.diffusion.training_losses( ## get the losses from the diffusion model ##
                self.model,
                micro,  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
                t,  # [bs](int) sampled timesteps
                model_kwargs={'y': batch},
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
            
    def predict_single_step(self, batch, use_t=None):
        # self.mp_trainer.zero_grad()
        # use_t is not Noen 
        # tot_samples = []
        # tot_targets = [] # predict single step ##
        
        # tot_dec_disp_e_along_normals = []
        # tot_dec_disp_e_vt_normals = []
        # tot_pred_joints_quant = []
        tot_samples = {key: [] for key in batch}

        
        nn_bsz = batch['X'].shape[0]
        # 
        for i in range(0, nn_bsz, self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            # ## micro batch ##
            # rhand_joints = micro['rhand_joints']
            # micro_cond = cond # micro_cond and cond ##
            ## micro-batch ##
            last_batch = (i + self.microbatch) >= nn_bsz
            t, weights = self.schedule_sampler.sample(micro['X'].shape[0], dist_util.dev())
            if use_t is not None:
                t = torch.zeros_like(t) + use_t

            shape = {
                key: micro[key].shape for key in micro
            }
            
            ## sample from the model --- the target sample should be in the sahpe of micro['X'].shape ##
            # shape = micro['X'].shape

            sample_fn = self.diffusion.p_sample_loop
            samples = sample_fn(
                self.ddp_model, 
                shape,
                noise=None,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                progress=True
            )

            for key in samples:
                tot_samples[key].append(samples[key])

            ## tot # primal space actions distribution; actions jdistribtuion; 
            ## for a sinlge manpulator ---- segs distribution ##
            ## for a segmented manipulator ---- construct the graph and the constraints distributions ## 
            ## transformations across different manipulator morphologies ##
            ## correspondences ##
            ## cnnect manipulators with different structures ##
            ## optimize using different constraints ##
            ## they are all maniulators that can accomplish the same task ##
            ## no so many sequneces currently ##
            ## its an action optimizatin pprocess ##
            ## ##
            # tot_samples['X'].append(samples)

        for key in tot_samples:
            tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
        return tot_samples


    ### predict from data ###
    def predict_from_data(self):

        
        tot_samples = {}
        
        for batch in tqdm(self.data): # batch data #
            
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
                elif isinstance(batch[k], list):
                    # batch[k] = [subval.to(self.device) for subval in batch[k]]
                    batch[k] = [subval.to(self.device) if isinstance(subval, torch.Tensor) else subval for subval in batch[k]]
                else:
                    batch[k] = batch[k]
            
            
            use_t = None
            
            ## data predict single step ##
            samples = self.predict_single_step(batch, use_t=use_t)
            
            for key in samples:
                if key not in tot_samples:
                    tot_samples[key] = [samples[key]]
                else:
                    tot_samples[key].append(samples[key])
            
        for key in tot_samples:
            tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## get the tot_samples ##
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

            # logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            model_sv_fn = os.path.join(self.save_dir, filename)
            logger.log(f"saving model to {model_sv_fn}...")
            with bf.BlobFile(bf.join(self.save_dir, filename), "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint(None)

        with bf.BlobFile(
            bf.join(self.save_dir, f"opt{(self.step+self.resume_step):09d}.pt"),
            "wb",
        ) as f:
            torch.save(self.opt.state_dict(), f)
            
    ## use the smaples to anchor cloest result --- then 


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

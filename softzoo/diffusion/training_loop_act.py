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
        # if self.args.nprocs > 1:   # train loop -> eval loop --- get the two diffusions with their corresponding models --- operate in that way # operate in that way # # And evaluate via such two parts of models ##
        # evaluateloop # -- args, model_graph, diffusion_graph, model_act, diffusion_act, data, data_act ## --- 
        #     self.cond_mode = model.module.cond_mode
        # else:
        #     self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.training.batch_size
        self.microbatch = args.training.batch_size
        self.lr = args.training.lr
        self.log_interval = args.training.log_interval
        self.save_interval = args.training.save_interval
        self.resume_checkpoint = args.training.resume_checkpoint
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

        # if self.args.finetune_with_cond: 
        #     self._load_and_sync_parameters_cond() 
        #     print(f"Setting trans linear layer to zero for conditioning...")
        #     self.model.set_trans_linear_layer_to_zero()
        # else: # finetune_with_cond
        #     self._load_and_sync_parameters()
        
        # self.mp_trainer = MixedPrecisionTrainer(
        #     model=self.model, # 
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=self.fp16_scale_growth,
        #     args=args,
        # )

        self.save_dir = args.save_dir
        # self.overwrite = args.overwrite

        self.opt = AdamW( ## 
            self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        ## TODO: the training resume settings ##
        if self.resume_step and not args.not_load_opt:
            self._load_optimizer_state()


        self.device = torch.device('cuda')

        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion)
        self.eval_wrapper, self.eval_data, self.eval_gt_data = None, None, None
        # if args.dataset in ['kit', 'humanml', 'motion_ours'] and args.eval_during_training:
        #     mm_num_samples = 0  # mm is super slow hence we won't run it during training
        #     mm_num_repeats = 0  # mm is super slow hence we won't run it during training
        #     gen_loader = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                     split=args.eval_split,
        #                                     hml_mode='eval')

        #     self.eval_gt_data = get_dataset_loader(name=args.dataset, batch_size=args.eval_batch_size, num_frames=None,
        #                                            split=args.eval_split,
        #                                            hml_mode='gt')
        #     self.eval_wrapper = EvaluatorMDMWrapper(args.dataset, dist_util.dev())
        #     self.eval_data = {
        #         'test': lambda: eval_humanml.get_mdm_loader(
        #             model, diffusion, args.eval_batch_size,
        #             gen_loader, mm_num_samples, mm_num_repeats, gen_loader.dataset.opt.max_motion_length,
        #             args.eval_num_samples, scale=1.,
        #         )
        #     } ## use_ddp ##
        self.use_ddp = False  # if self.args.nprocs == 1 else True
        self.ddp_model = model
        
    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        for k in ori_dict:
            if self.args.resume_diff:
                if k in model_dict:
                    v = ori_dict[k]
                    part_dict[k] = v
                    tot_params_n += 1
            else:
                if k in model_dict and "denoising" not in k:
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
            
            # self.model.load_state_dict(
            #     dist_util.load_state_dict(
            #         resume_checkpoint, map_location=dist_util.dev()
            #     )
            # )
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
                    
                    # tot_samples_np_inv_scaled = self.data.dataset.inv_scale_data(tot_samples_np) ## inv scaled the data ##
                    # closest_training_data = self.data.dataset.get_closest_training_data(tot_samples_np_inv_scaled)
                    
                    # tot_samples_sv_dict = {
                    #     'samples': tot_samples_np,
                    #     'closest_training_data': closest_training_data
                    # }
                    
                    self.save_samples(tot_samples_sv_dict)
                    # # Run for a finite amount of time in integration tests. # 
                    # if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    #     return
                self.step += 1
            # if not (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            #     break
        # Save the last checkpoint if it wasn't already saved. #
        # if (self.step - 1) % self.save_interval != 0: # saved #
        #     self.save()
        #     self.evaluate()

    def evaluate(self):
        if not self.args.eval_during_training:
            return
        # start_eval = time.time()
        # if self.eval_wrapper is not None:
        #     print('Running evaluation loop: [Should take about 90 min]')
        #     log_file = os.path.join(self.save_dir, f'eval_humanml_{(self.step + self.resume_step):09d}.log')
        #     diversity_times = 300
        #     mm_num_times = 0  # mm is super slow hence we won't run it during training
        #     eval_dict = eval_humanml.evaluation(
        #         self.eval_wrapper, self.eval_gt_data, self.eval_data, log_file,
        #         replication_times=self.args.eval_rep_times, diversity_times=diversity_times, mm_num_times=mm_num_times, run_mm=False)
        #     print(eval_dict)
        #     for k, v in eval_dict.items():
        #         if k.startswith('R_precision'):
        #             for i in range(len(v)):
        #                 self.train_platform.report_scalar(name=f'top{i + 1}_' + k, value=v[i],
        #                                                   iteration=self.step + self.resume_step,
        #                                                   group_name='Eval')
        #         else:
        #             self.train_platform.report_scalar(name=k, value=v, iteration=self.step + self.resume_step,
        #                                               group_name='Eval')

        # elif self.dataset in ['humanact12', 'uestc']:
        #     eval_args = SimpleNamespace(num_seeds=self.args.eval_rep_times, num_samples=self.args.eval_num_samples,
        #                                 batch_size=self.args.eval_batch_size, device=self.device, guidance_param = 1,
        #                                 dataset=self.dataset, unconstrained=self.args.unconstrained,
        #                                 model_path=os.path.join(self.save_dir, self.ckpt_file_name()))
        #     eval_dict = eval_humanact12_uestc.evaluate(eval_args, model=self.model, diffusion=self.diffusion, data=self.data.dataset)
        #     print(f'Evaluation results on {self.dataset}: {sorted(eval_dict["feats"].items())}')
        #     for k, v in eval_dict["feats"].items():
        #         if 'unconstrained' not in k:
        #             self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval')
        #         else:
        #             self.train_platform.report_scalar(name=k, value=np.array(v).astype(float).mean(), iteration=self.step, group_name='Eval Unconstrained')

        # end_eval = time.time()
        # print(f'Evaluation time: {round(end_eval-start_eval)/60}min')


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
                micro['X'],  # [bs, ch, image_size, image_size] ## then you should sample the res from it ##
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

            # shape = {
            #     key: micro[key].shape for key in micro
            # }
            
            ## sample from the model --- the target sample should be in the sahpe of micro['X'].shape ##
            shape = micro['X'].shape

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
                # model_kwargs=micro,
                # skip_timesteps=0, 
                # # init_image=micro,
                # progress=True,
                # dump_steps=None,
                # # const_noise=False,
                # const_noise=self.args.const_noise,
                # st_timestep=use_t,
            )
            
            # for key in samples:
            #     tot_samples[key].append(samples[key])

            tot_samples['X'].append(samples)
            # sample either as joints or as relative positions for each base pts #
            # tot_samples.append(samples['sampled_rhand_joints'])
            # tot_targets.append(micro['rhand_joints'])
            
            # if 'e_disp_rel_to_base_along_normals' in samples:
            #     tot_dec_disp_e_along_normals.append(samples['e_disp_rel_to_base_along_normals'])
            #     tot_dec_disp_e_vt_normals.append(samples['e_disp_rel_to_baes_vt_normals'])
            # if 'pred_joint_quants' in samples:
            #     tot_pred_joints_quant.append(samples['pred_joint_quants'])

        for key in tot_samples:
            tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## 
        return tot_samples
        # model_output = torch.cat(tot_samples, dim=0)
        # # model_output = tot_samples
        # target = torch.cat(tot_targets, dim=0)
        
        # if len(tot_dec_disp_e_along_normals) > 0:
        #     tot_dec_disp_e_along_normals = torch.cat(tot_dec_disp_e_along_normals, dim=0) 
        #     tot_dec_disp_e_vt_normals = torch.cat(tot_dec_disp_e_vt_normals, dim=0) ### tot_dec_disp_e_vt_normals #
        
        # if len(tot_pred_joints_quant) > 0:
        #     tot_pred_joints_quant = torch.cat(tot_pred_joints_quant, dim=0)
        
        # # print(f"Returning with model_output; {model_output.size()}, target: {target.size()}")
        # print(f"Returning with target: {target.size()}")
        
        # if isinstance(tot_pred_joints_quant, torch.Tensor):
        #     return model_output, target, tot_pred_joints_quant
        # elif isinstance(tot_dec_disp_e_along_normals, torch.Tensor):
        #     return model_output, target, tot_dec_disp_e_along_normals, tot_dec_disp_e_vt_normals
        # else:
        #     return model_output, target
        
        # return  model_output, target

    ### predict from data ###
    def predict_from_data(self):

        # for epoch 
        # for epoch in range(self.num_epochs): # 
        # print(f'Starting epoch {epoch}') # the 
        
        ## ==== a single pass for a single sequence ==== ##
        # tot_model_outputs = []
        # tot_targets = []
        # tot_st_idxes = []
        # tot_ed_idxes = []
        # tot_pert_verts = []
        # tot_verts = []
        # tot_dec_disp_e_along_normals = []
        # tot_dec_disp_e_vt_normals = []
        # ## motion; cond; data ##
        # tot_pred_joints_quant = []
        
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
            
            # motion = motion.to(self.device)
            # cond['y'] = {key: val.to(self.device) if torch.is_tensor(val) else val for key, val in cond['y'].items()}
            # st_idxes = cond['y']['st_idx'] # st_idxes
            # ed_idxes = cond['y']['ed_idx'] # ed_idxes
            
            # pert_verts = cond['y']['pert_verts']
            # verts = cond['y']['verts']
            
            # if 'avg_joints' in cond['y']:
            #     avg_joints = cond['y']['avg_joints']
            #     std_joints = cond['y']['std_joints']
            # else:
            #     avg_joints = None
            #     std_joints = None
                
            # st_idxes = batch['st_idx']
            # ed_idxes = batch['ed_idx']
            # pert_verts = batch['pert_verts']
            # verts = batch['verts']
            
            # # tot pert verts
            # tot_pert_verts.append(pert_verts)
            # tot_verts.append(verts)
            
            ## generative denoising -> we want to use it for the denoising task ##
            # std_joints: bsz x 1
            # avg_joints: bsz x 1 x 3 --> mean of joints for each batch 
        
            ## predict_single_step ##
            # model_output, target = self.predict_single_step(batch, use_t=1) ### trainingjloop ours
            # use_t = self.args.use_t
            use_t = None
            
            ## data predict single step ##
            samples = self.predict_single_step(batch, use_t=use_t)
            
            for key in samples:
                if key not in tot_samples:
                    tot_samples[key] = [samples[key]]
                else:
                    tot_samples[key].append(samples[key])
            
            # tot_samples['X'].append(samples)
            
            # #### diff baes jts e ##
            # if len(tot_pred_outputs) == 3:
            #     model_output, target, pred_joints_quant = tot_pred_outputs
            #     tot_pred_joints_quant.append(pred_joints_quant)
            # elif self.args.diff_basejtse: 
            #     model_output, target, dec_disp_e_along_normals, dec_disp_e_vt_normals = tot_pred_outputs
            # else:
            #     model_output, target = tot_pred_outputs[:2]
            
            # # model output; target #
            # ## model_output: ([6, 21, 3, 60]), target: torch.Size([6, 21, 3, 60])
            # # if avg_joints is not None:
            # #     ### model_output, target ###
            # #     model_output = (model_output * std_joints.unsqueeze(-1).unsqueeze(-1)) + avg_joints.unsqueeze(-1)
            # #     target = (target * std_joints.unsqueeze(-1).unsqueeze(-1)) + avg_joints.unsqueeze(-1)
            
            # # 10 -> the output sequence is still a little bit noisy #
            # # 100 -> 60 
            # # the difficulty of predicting base pts rel position information #
            # # the difficulty of the prediction problem # base pts rel information p
            # ## predicting base pts relative positions to the base_pts predictions ## ### base pts predictions ## wu le ##
            
            # if self.args.diff_basejtse: 
            #     tot_dec_disp_e_along_normals.append(dec_disp_e_along_normals)
            #     tot_dec_disp_e_vt_normals.append(dec_disp_e_vt_normals)
                
            
            
            
            # tot_st_idxes.append(st_idxes)
            # tot_ed_idxes.append(ed_idxes)
            # tot_targets.append(target)
            # tot_model_outputs.append(model_output)
            # tot_model_outputs.extend(model_output)
            # tot_model_outputs = tot_model_outputs + model_output
        
        for key in tot_samples:
            tot_samples[key] = torch.cat(tot_samples[key], dim=0) ## get the tot_samples ##
        return tot_samples
        
        # tot_st_idxes = torch.cat(tot_st_idxes, dim=0)
        # tot_ed_idxes = torch.cat(tot_ed_idxes, dim=0)
        # tot_targets = torch.cat(tot_targets, dim=0)
        # tot_model_outputs = torch.cat(tot_model_outputs, dim=0)
        
        # if self.args.diff_basejtse: 
        #     tot_dec_disp_e_along_normals = torch.cat(tot_dec_disp_e_along_normals, dim=0)
        #     tot_dec_disp_e_vt_normals = torch.cat(tot_dec_disp_e_vt_normals, dim=0)
        
        # if len(tot_pred_joints_quant) > 0:
        #     tot_pred_joints_quant = torch.cat(tot_pred_joints_quant, dim=0)
        
        # tot_pert_verts = torch.cat(tot_pert_verts, dim=0)
        # tot_verts = torch.cat(tot_verts, dim=0)
        
        # if isinstance(tot_pred_joints_quant, torch.Tensor):
        #     return  tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts, tot_pred_joints_quant
        
        # elif self.args.diff_basejtse: 
        #     return tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts, tot_dec_disp_e_along_normals, tot_dec_disp_e_vt_normals
        # else:
        #     return tot_targets, tot_model_outputs, tot_st_idxes, tot_ed_idxes, tot_pert_verts, tot_verts
            

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

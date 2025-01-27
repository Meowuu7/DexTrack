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


class EvalLoopCombined:
    def __init__(self, args, model_graph, diffusion_graph, model_act, diffusion_act, data, data_act):
        
        self.args = args 
        
        # self.dataset = 
        
        # self.args = args
        self.dataset = args.dataset.dataset_type
        # self.train_platform = train_platform
        # self.model = model
        # self.diffusion = diffusion
        self.model_graph = model_graph
        self.diffusion_graph = diffusion_graph
        self.model_act = model_act
        self.diffusion_act = diffusion_act ## get the diffusions and models
        # if self.args.nprocs > 1:   # train loop -> eval loop --- get the two diffusions with their corresponding models --- operate in that way # operate in that way # # And evaluate via such two parts of models ##
        # evaluateloop # -- args, model_graph, diffusion_graph, model_act, diffusion_act, data, data_act ## --- 
        #     self.cond_mode = model.module.cond_mode
        # else:
        #     self.cond_mode = model.cond_mode
        #### ==== data and data_act ==== ####
        self.data = data
        self.data_act = data_act
        
        self.batch_size = args.training.batch_size
        self.microbatch = args.training.batch_size
        self.lr = args.training.lr
        self.log_interval = args.training.log_interval
        self.save_interval = args.training.save_interval
        self.resume_checkpoint = args.training.resume_checkpoint # resume checkpoint # 
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
        
        self.resume_checkpoint_graph = args.training.resume_checkpoint_graph
        self.resume_checkpoint_act = args.training.resume_checkpoint_act ## resume rl 
        
        print(f"num_epochs: {self.num_epochs}, num_steps: {self.num_steps}")

        self.sync_cuda = torch.cuda.is_available()
        
        ## === resume the checkpoint === ##
        # self._load_and_sync_parameters() # one for the graph
        self._load_and_sync_parameters_graph()
        self._load_and_sync_parameters_act()

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
        self.exp_tag = args.exp_tag
        
        self.save_dir = os.path.join(self.save_dir, self.exp_tag)
        os.makedirs(self.save_dir, exist_ok=True)
        
        # self.overwrite = args.overwrite
        
        ## gett he optimizer ##

        self.opt = AdamW( # adamw #
            self.model_act.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        
        # ## TODO: the training resume settings ##
        # if self.resume_step and not args.not_load_opt:
        #     self._load_optimizer_state()


        self.device = torch.device('cuda')

        self.schedule_sampler_type = 'uniform'
        ## get the schedule sampler ##
        self.schedule_sampler = create_named_schedule_sampler(self.schedule_sampler_type, diffusion_act)
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
        # self.ddp_model = model
        
    def safe_load_ckpt(self, model, state_dicts):
        ori_dict = state_dicts
        part_dict = dict()
        model_dict = model.state_dict()
        tot_params_n = 0
        print(f"model_dict: {model_dict.keys()}")
        for k in ori_dict:
            print(f"k: {k}")
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
            
    def _load_and_sync_parameters_graph(self):
        resume_checkpoint = self.resume_checkpoint_graph

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
  
            self.safe_load_ckpt(self.model_graph, 
                                    dist_util.load_state_dict(
                                        resume_checkpoint, map_location=dist_util.dev()
                                    )
                                )
    
    def _load_and_sync_parameters_act(self):
        resume_checkpoint = self.resume_checkpoint_act

        if resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(resume_checkpoint)
            logger.log(f"loading model from checkpoint: {resume_checkpoint}...")
  
            self.safe_load_ckpt(self.model_act, 
                                    dist_util.load_state_dict(
                                        resume_checkpoint, map_location=dist_util.dev()
                                    )
                                )
    
    ## laod the optimizer statet ##
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
    ## predict 
    
    def predict_acts(self, batch):
        nn_bsz = batch['X'].shape[0]
        
        tot_samples = {key: [] for key in batch}
        ## for the micro and the batch adn the micro and the micro and the graph ad the p_sample loops # 
        for i in range(0, nn_bsz, self.microbatch):
            micro = batch
            shape = micro['X'].shape
            # shape = { key: micro[key].shape for key in micro }
            # shape = { key: micnro }
            
            sample_fn = self.diffusion_act.p_sample_loop
            
            samples = sample_fn(
                self.model_act,
                shape,
                noise=None,
                clip_denoised=False,
                denoised_fn=None,
                cond_fn=None,
                model_kwargs=None,
                progress=True
            )
            
            # for key in samples: # get the samples key #
            tot_samples['X'].append(samples)
            ## tot_samples ##
        for key in tot_samples: #  get the tot_samples
            tot_samples[key] = torch.cat(tot_samples[key], dim=0)
        return tot_samples
        
    
    def predict_graph_structure(self, batch):
        tot_samples = {key: [] for key in batch}
        
        nn_bsz = batch['X'].shape[0]
        
        for i in range(0, nn_bsz, self.microbatch):
            # assert i == 
            micro = batch
            shape = micro['X'].shape
            shape = { key: micro[key].shape for key in micro }
            
            sample_fn = self.diffusion_graph.p_sample_loop
            
            samples = sample_fn(
                self.model_graph,
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
        for key in tot_samples:
            tot_samples[key] = torch.cat(tot_samples[key], dim=0)
        return tot_samples ## for each kkey in the tot_samples --> bsz x shape ##
    
    def proj_graph_struct(self, graph_struct):
        graph_X = graph_struct['X']
        graph_E = graph_struct['E']
        nx, ny = graph_E.size(0), graph_E.size(1)
        if len(graph_E.size()) > 2:
            graph_E = graph_E.squeeze(-1)
        for i_x in range(nx):
            for i_y in range(ny):
                if i_x > i_y:
                    graph_E[i_x, i_y] = 0.0
                ### TODO: add the graph connectivity values thresholding ###
        for i_y in range(ny):
            adj_values_to_y = graph_E[: i_y + 1, i_y] ##
            if torch.sum(adj_values_to_y).item() == 0.0:
                graph_E[i_y, i_y] = 1.0
        graph_E = graph_E / torch.clamp(torch.sum(graph_E, dim=0, keepdim=True), min=1e-6)
        ## get the graph connectivities ## 
        rt_dict = {
            'X': graph_X, 
            'E': graph_E
        }
                    
        return rt_dict
        
    
    
    ### project_act_via_graph_struct(self, links_act, graph_struct):
    def project_act_via_graph_struct(self, links_act, graph_struct):
        # get the graph sturcutres here 
        ## graph structures ## 
        graph_X =  graph_struct['X']
        graph_E = graph_struct['E']
        # graph_E #
        graph_E = ((graph_E * 2.0) + 1.0) / 2.0 ## scale the data ##
        graph_X = (graph_X * 2.0 + 1.0) / 10.0 ## getthe graph X ##
        
        graph_struct = {
            'X': graph_X,
            'E': graph_E
        }
        
        ### graph E should behave like a graph structure ##
        ### directional graph ###
        ### E[parent_idx, child_idx] = parent-child connection ###
        ### a normalized ### 
        ### if without any connections to others ---  then it should be self-conected -- at least it should has the connection to itself ##
        ## TODO: set the threshold for the graph connectivity values and the graph scale values --- elements should degenerate to zero if smaller than their corresponding values ##
        
        proj_dict = self.proj_graph_struct(graph_struct)
        graph_X, graph_E = proj_dict['X'], proj_dict['E'] ## get the proj dicts ##  graph x and graph e #
        # links_act -> get he rotationsla and translational states for each link at each frame using the links_act #
        # get the links act ## 
        ## nn-graph-nodes x nn-timesteps x nn-links x nn-act0dimensions ##
        # # get the 
        # nn_nodes, nn_ts = links_act.size(0), links_act.size(1) ## get the
        links_act = links_act.contiguous().transpose(1, 0).contiguous()
        nn_ts, nn_nodes = links_act.size(0), links_act.size(1)
        links_act_rot, links_act_trans = links_act[..., 0], links_act[..., 1:] ## links act rot; lnks act trans ##
        proj_links_act_trans = torch.zeros_like(links_act_trans)
        links_vel_rot, links_vel_trans = torch.zeros_like(links_act_rot), torch.zeros_like(links_act_trans) ## links act rot and the trans # 
        proj_links_vel_trans = torch.zeros_like(links_vel_trans) 
        links_vec_rot, links_vec_trans = torch.zeros_like(links_vel_rot), torch.zeros_like(links_vel_trans) ## 
        proj_links_vec_trans = torch.zeros_like(links_vec_trans) ### links_vel_trans ##
        dt = 1e-1 
        # tranforma the rotational accs to vels and vecs ##
        for i_ts in range(1, nn_ts): 
            ## 
            ## 
            ## 1) get the roatioanl transformations --- forward for the tranfsoramtiosn
            ## 2) project for the translational transformations and the translational velocities and the translational actions
            ## 3) 
            
            ## 1) get the rotational accelerations, velocities, and vectors ##
            ## 2) project for the accs ## --> however such accs cannot be used directly for integrating ##
            cur_ts_links_rot_accs = links_act_rot[i_ts] # act rot #
            # cur ts link rot accs # 
            # accs ## 
            cur_ts_links_rot_vels = links_vel_rot[i_ts - 1] + dt * cur_ts_links_rot_accs
            # link_vec_rot + dt * (link_rot_vel[t - 1] + dt * link_rot_accs[t]) )
            cur_ts_links_rot_vecs = links_vec_rot[i_ts - 1] + dt * links_vel_rot[i_ts - 1] + dt * dt * links_act_rot[i_ts]
            
            links_vel_rot[i_ts] = cur_ts_links_rot_vels
            links_vec_rot[i_ts] = cur_ts_links_rot_vecs ## pass the links_vel, links_vec ##
            
            pass
        ## links joint pos ##
        ## get links joint acts from graph Xs ##
        link_joint_pos = [] ### TODO: get links pos via the graph X information ### ## get the links ###
        ## links joint positions ## links and the graphs #### and the graph, the graph ## # 
        ## ## --- for graph 
        ## ## --- graph with nodes ## ## 
        ## xx nodes; 
        ## for the joint pos ##
        ## for the joint pos ## 
        ##  threhold is 0.03 ##
        ## at first, get the valid number of links ##
        ## then, get the valid number of links ## 
        ## then use that to find the graph links ##
        ## only the first xxx is valid --- for others --- you can set then to random values ## 
        dist_threshold = 0.03
        first_invalid_link_idx = 0
        
        
        
        link_joint_pos.append(torch.zeros((2,), dtype=torch.float32, device=links_act.device))
        
        for i_link in range(graph_X.size(0)):
            cur_link_x, cur_link_y = graph_X[i_link, 0], graph_X[i_link, 1] ## 
            if cur_link_x < dist_threshold or cur_link_y < dist_threshold:
                first_invalid_link_idx = i_link
                break
        nn_links_one_side = (first_invalid_link_idx - 1) // 2 ### nn_links_one_side, base_link, nn_links_one_side
        ### get the links joint pos ###
        st_joint_pos = torch.tensor([0.5 - graph_X[0, 0] / 2.0, 0.45], dtype=torch.float32, device=links_act.device)
        for i_link in range(1, nn_links_one_side + 1):
            link_joint_pos.append(st_joint_pos.clone())
            cur_link_x, cur_link_y = graph_X[i_link, 0], graph_X[i_link, 1]
            st_joint_pos[0] = st_joint_pos[0] - cur_link_x ## st joint pos ##
        st_joint_pos = torch.tensor([0.5 + graph_X[0, 0] / 2.0, 0.45], dtype=torch.float32, device=links_act.device)
        for i_link in range(nn_links_one_side + 1, nn_links_one_side * 2 + 1):
            link_joint_pos.append(st_joint_pos.clone())
            cur_link_x, cur_link_y = graph_X[i_link, 0], graph_X[i_link, 1]
            st_joint_pos[0] = st_joint_pos[0] + cur_link_x
        
        nn_valid_nodes = nn_links_one_side * 2 + 1
        
        print(f"link_joint_pos: {first_invalid_link_idx, link_joint_pos}")
        print(f"graph_X")
        print(graph_X)
        print(f"graph_E")
        print(graph_E)
        # get joint poses ## 
        link_joint_pos = torch.stack(link_joint_pos, dim=0) ## get joint poses for each link ## 
            
        
        for i_ts in range(nn_ts):
            for i_link in range(nn_valid_nodes):
                proj_links_act_trans[i_ts, i_link] = links_act_trans[i_ts, i_link] * graph_E[i_link, i_link]
                for j_link in range(i_link):
                    j_link_rot_vec = links_vec_rot[i_ts, j_link]
                    j_link_rot_vel = links_vel_rot[i_ts, j_link]
                    j_link_rot_acc = links_act_rot[i_ts, j_link]
                    
                    j_link_rot_mtx_ddot = torch.tensor(
                        [ [ -torch.sin(j_link_rot_vec), -torch.cos(j_link_rot_vec) ], [ torch.cos(j_link_rot_vec), -torch.sin(j_link_rot_vec) ] ], dtype=torch.float32, device=links_act.device
                    ) * j_link_rot_acc + torch.tensor(
                        [ [ -torch.cos(j_link_rot_vec), torch.sin(j_link_rot_vec) ], [ -torch.sin(j_link_rot_vec), -torch.cos(j_link_rot_vec) ] ], dtype=torch.float32, device=links_act.device
                    ) * j_link_rot_vel * j_link_rot_vel ## get the j_link_rot_vels 
                    
                    i_link_rot_vec = links_vec_rot[i_ts, i_link]
                    i_link_rot_vel = links_vel_rot[i_ts, i_link]
                    i_link_rot_acc = links_act_rot[i_ts, i_link]
                    
                    i_link_rot_mtx_ddot = torch.tensor(
                        [ [ -torch.sin(i_link_rot_vec), -torch.cos(i_link_rot_vec) ], [ torch.cos(i_link_rot_vec), -torch.sin(i_link_rot_vec) ] ], dtype=torch.float32, device=links_act.device
                    ) * i_link_rot_acc + torch.tensor(
                        [ [ -torch.cos(i_link_rot_vec), torch.sin(i_link_rot_vec) ], [ -torch.sin(i_link_rot_vec), -torch.cos(i_link_rot_vec) ] ], dtype=torch.float32, device=links_act.device
                    ) * i_link_rot_vel * i_link_rot_vel ## get the j_link_rot_vels 
                    
                    j_link_trans_acc = links_act_trans[i_ts, j_link]
                    j_to_c_trans_acc = torch.matmul(j_link_rot_mtx_ddot - i_link_rot_mtx_ddot, link_joint_pos[i_link].unsqueeze(-1)).squeeze(-1) + j_link_trans_acc
                    if graph_E[j_link, i_link] > 0.0:
                        proj_links_act_trans[i_ts, i_link] += j_to_c_trans_acc * graph_E[j_link, i_link]
                links_act_trans[i_ts, i_link] = proj_links_act_trans[i_ts, i_link] ## get the act trasn ##
        
        links_act[..., 1:] = links_act_trans.clone()
        links_act = links_act.contiguous().transpose(1, 0).contiguous()
        # links_act ## links act ## ## links act #### 
        ## links act ##  ## 
        ### TODO: correctness check for the above code ###
        ##3 TODO: ave the projected acts and the graph structure -- 
        
        # nn_nodes x nn_dim 
        
        expanded_link_joint_pose = torch.cat(
            [link_joint_pos, torch.zeros((nn_nodes - link_joint_pos.size(0), 2), dtype=torch.float32, device=link_joint_pos.device)], dim=0
        )
        
        
        rt_dict = {
            'links_act': links_act,
            'graph_X': graph_X,
            'graph_E':  graph_E,
            'link_joint_pos': expanded_link_joint_pose
        }
        
        return rt_dict
        
        
    
    ## evaluate for a loop ##
    def evaluate_run_loop(self):
        
        sampled_projected_rt_dict = {
            'links_act': [], 'graph_X': [], 'graph_E': [], 'link_joint_pos': []
        }
        
        for batch_graph, batch_act in tqdm(zip(self.data, self.data_act)):
            ## sample for the graph structure
            # predict from the data
            ## TODO: add the predicted actions ##
            sampled_graph_struct = self.predict_graph_structure(batch_graph) ## get the sampled graphs ##
            ## sampled graph struct ##
            ## get the sampled graph structut ##
            sampled_act = self.predict_acts(batch_act)
            
            # tot_projected_links_act = []
            
            tot_projected_rt_dict = {
                'links_act': [], 'graph_X': [], 'graph_E': [], 'link_joint_pos': []
            }
            
            for i_sample in range(sampled_graph_struct['X'].size(0)):
                # cur_graph_struct = sampled_graph_struct[i_sample]
                cur_graph_struct = {
                    'X': sampled_graph_struct['X'][i_sample], ## X sampled graph struct ##
                    'E': sampled_graph_struct['E'][i_sample]
                }
                cur_sampled_act = sampled_act['X'][i_sample] # and 
                
                cur_sampled_act = self.data_act.dataset.inv_scale_data(cur_sampled_act) ## inv scale the data ##
                
                ## get the projected graph acts and the graph struct  
                # projected_links_act = self.project_act_via_graph_struct(cur_sampled_act, cur_graph_struct)
                projected_rt_dict = self.project_act_via_graph_struct(cur_sampled_act, cur_graph_struct)
                # cur_act = batch_act[i_sample] ## get the current act ##
                ## cur_act and the batch_act ##
                # tot_projected_links_act.append(projected_links_act)
                
                for key in projected_rt_dict:
                    tot_projected_rt_dict[key].append(projected_rt_dict[key])
            for key in tot_projected_rt_dict:
                tot_projected_rt_dict[key] = torch.stack(tot_projected_rt_dict[key], dim=0) ## get the total proejcted rt dicts
            
            for key in tot_projected_rt_dict:
                sampled_projected_rt_dict[key].append(tot_projected_rt_dict[key])
            # tot_projected_links_act = torch.stack(tot_projected_links_act, dim=0) ## nn_samples x xxxxxx
            
        for key in sampled_projected_rt_dict:
            sampled_projected_rt_dict[key] = torch.cat(sampled_projected_rt_dict[key], dim=0) ## get projected rt_dict
            # pass
        
        ## TODO: save the link joint infors
        ####### ======== Save the sampled rt_dict in `.npy` file format ========= #######
        sampled_projected_rt_dict_np = {} # save proj # # save proj # # just for eval and for draw a demo #?
        for key in sampled_projected_rt_dict:
            sampled_projected_rt_dict_np[key] = sampled_projected_rt_dict[key].detach().cpu().numpy()
        save_projected_rt_dict_fn = os.path.join(self.save_dir, "sampled_projected_rt_dict.npy")
        np.save(save_projected_rt_dict_fn, sampled_projected_rt_dict_np)
        print(f"Sampled information saved to {save_projected_rt_dict_fn}")
            
        return sampled_projected_rt_dict

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

                ### 

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
        # start_eval = time.time() ## time.time() ## --- get the evaluation starting time ##
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
        # target = torch.cat(tot_targets, dim=0) # target = torch.cat() # target and the forces #
        # target # 
        # for a specific manipulator? # two-D ###
        # the same sahpe but with differnet constraints? #
        # asmpling from the manipulation trajectory mdoel #
        # we have a # with one manipulatin trajectories # # record each #
        # jgive non shape === get its trajectories #
        # given the manipualtor consider the translate process #
        # in two-D; and simple manipulations ## # in two-D # ##
        
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

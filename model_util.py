
from diffusion.respace import SpacedDiffusion, SpacedDiffusionAct, space_timesteps, SpacedDiffusionPC, SpacedDiffusionSegs, SpacedDiffusion3DPC # as SpacedDiffusion
# SpacedDiffusion_OursV5
# from diffusion.respace_ours import SpacedDiffusionV5 as SpacedDiffusion_OursV5
# import .gaussian_diffusion as gd
from diffusion import gaussian_diffusion as gd
from diffusion import gaussian_diffusion_act as gd_act
from diffusion import gaussian_diffusion_pc as gd_pc
from diffusion import gaussian_diffusion_3d_pc as gd_3d_pc
from diffusion import gaussian_diffusion_segs as gd_segs
# from diffusion.respace import SpacedDiffusion, space_timesteps
from models.transformer_model import GraphTransformer, MLP_Net, MLP_Act_Net, MLP_Net_PC, MLP_Net_Segs, Transformer_Net_PC_Seq, MLP_Net_PC_Only, Transformer_Net_PC_Seq_V2, Transformer_Net_PC_Seq_V3, Transformer_Net_PC_Seq_V3_wcond, Transformer_Net_PC_Seq_V3_wcond_V2, Transformer_Net_PC_Seq_V3_wtaskcond_V2, Transformer_Net_PC_Seq_V3_KineDiff, Transformer_Net_PC_Seq_V3_KineDiff_AE, Transformer_Net_PC_Seq_V3_AE
import torch.nn as nn


from models.transformer_model import Transformer_Net_PC_Seq_V4


def create_model_and_diffusion_act(args):
    ## 
    
    model = create_model_act(args)
    diffusion_model = create_gaussian_diffusion_act(args.diffusion)
    return model, diffusion_model


def create_model_and_diffusion(args):
    ## 
    
    model = create_model(args)
    diffusion_model = create_gaussian_diffusion(args.diffusion)
    return model, diffusion_model

def create_model_and_diffusion_pc(args):
    ## 
    
    model = create_model_pc(args)
    diffusion_model = create_gaussian_diffusion_pc(args.diffusion)
    return model, diffusion_model


def create_model_deterministic(args):
    
    cfg = args
    
    if cfg.training.diff_task_space:
        input_dims = {
            'X': 3,
            'feat': 3
        }
    elif cfg.training.kine_diff:
        concat_two_dims = cfg.training.concat_two_dims
        input_dims = {
            'X': 3,
            'feat': 22 + 3 + 3,
            'concat_two_dims': concat_two_dims
        }
    else:
        input_dims = {
            'X': 22,
            'feat': 22,
        }
    
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    
    
    output_dims = {
        'X': input_dims['X'],
        'feat': input_dims['feat']
    }
    
    #### the args and the cfg ####
    if cfg.training.tracking_ctl_diff:
        cond_task_type = 'tracking'
    else:
        cond_task_type = 'rotation'
    
    
    model = Transformer_Net_PC_Seq_V4(
        n_layers=args.model.n_layers_3d_pc,
        input_dims=input_dims,
        hidden_mlp_dims=hidden_mlp_dims,
        output_dims=output_dims,
        act_fn_in=nn.ReLU(),
        act_fn_out=nn.ReLU(),
        traj_cond=True,
        task_cond_type=cfg.training.task_cond_type,
        debug=cfg.training.debug
    )
    ### having returned the model ###
    return model

## create model and diffusion ##
def create_model_and_diffusion_3d_pc(args):
    ## # creat model ad the diffusion #
    if args.model.model_arch == "transformer_v2":
        model = create_model_3d_pc_transformer_v2(args)
    elif args.model.model_arch == "transformer_v3":
        # if args.training.use_jointspace
        if args.training.use_jointspace_seq:
            if args.training.task_cond:
                model = create_model_3d_jointspace_transformer_v3_taskcondv2(args)
            else:
                model = create_model_3d_jointspace_transformer_v3(args)
        elif args.training.task_cond: # training with the task conditions #
            model = create_model_3d_pc_transformer_v3_taskcond(args)
        elif args.training.setting == 'trajectory_translations_cond':
            model = create_model_3d_pc_transformer_v3_trajtranscond(args)
        else:
            model = create_model_3d_pc_transformer_v3(args)
    elif args.model.model_arch == "transformer":
        model = create_model_3d_pc_transformer(args)
    else:
        model = create_model_3d_pc(args)
        # pass
    
    # model = create_model_3d_pc(args)
    # model = create_model_3d_pc_only(args)
    # model = create_model_3d_pc_transformer(args)
    diffusion_model = create_gaussian_diffusion_3d_pc(args.diffusion)
    return model, diffusion_model
    

def create_model_and_diffusion_segs(args):
    ## 
    
    model = create_model_segs(args)
    diffusion_model = create_gaussian_diffusion_segs(args.diffusion)
    return model, diffusion_model
    


def create_model(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    
    input_dims = {
        'X': cfg.model.input_dims.X,
        'E': cfg.model.input_dims.E, 
        'y': cfg.model.input_dims.y ## get the model X E and y ### we have a label for each graph, represented as the y? ##
    }
    
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims.X,
        'E': cfg.model.hidden_mlp_dims.E,
        'y': cfg.model.hidden_mlp_dims.y
    }
    
    hidden_dims = {
        'dx': cfg.model.hidden_dims.dx,
        'de': cfg.model.hidden_dims.de,
        'dy': cfg.model.hidden_dims.dy,
        'dim_ffX': cfg.model.hidden_dims.dim_ffX,
        'dim_ffE': cfg.model.hidden_dims.dim_ffE,
        'n_head': cfg.model.hidden_dims.n_head,
    }
    ### get 
    ## TODO: the output dim should be set as the same to the input_dims, right ? ## 
    output_dims = { ## 
        'X': cfg.model.output_dims.X,
        'E': cfg.model.output_dims.E,
        'y': cfg.model.output_dims.y
    }
    
    
    # model = GraphTransformer(n_layers=cfg.model.n_layers,
    #                          input_dims=input_dims,
    #                          hidden_mlp_dims=hidden_mlp_dims,
    #                          hidden_dims=hidden_dims,
    #                          output_dims=output_dims,
    #                          act_fn_in=nn.ReLU(),
    #                          act_fn_out=nn.ReLU())
    # MLP_Net
    model = MLP_Net(n_layers=cfg.model.n_layers,
                             input_dims=input_dims,
                             hidden_mlp_dims=hidden_mlp_dims,
                             hidden_dims=hidden_dims,
                             output_dims=output_dims,
                             act_fn_in=nn.ReLU(),
                             act_fn_out=nn.ReLU())
    return model



def create_model_act(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    input_dim = cfg.model.input_dims_act.X 
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_act.X, 
        'y': cfg.model.hidden_mlp_dims_act.y
    }
    output_dim = cfg.model.output_dims_act.X
    
    model = MLP_Act_Net(n_layers=cfg.model.n_layers_act,
                        input_dim=input_dim,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dim=output_dim,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    return model




def create_model_pc(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    input_dims = {
        'X': cfg.model.input_dims_pc.pos,
        'feat': cfg.model.input_dims_pc.feat,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_pc.X, 
        'feat': cfg.model.hidden_mlp_dims_pc.feat,
        't': cfg.model.hidden_mlp_dims_pc.t
    }
    output_dims = {
        'X': cfg.model.output_dims_pc.X,
        'feat': cfg.model.output_dims_pc.feat
    }
    
    model = MLP_Net_PC(n_layers=cfg.model.n_layers_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    return model



def create_model_3d_pc(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    input_dims = {
        'X': cfg.model.input_dims_3d_pc.pos,
        'feat': cfg.model.input_dims_3d_pc.feat,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat': cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': cfg.model.output_dims_3d_pc.X,
        'feat': cfg.model.output_dims_3d_pc.feat
    }
    
    model = MLP_Net_PC(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model

def create_model_3d_pc_only(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    input_dims = {
        'X': cfg.model.input_dims_3d_pc.pos,
        'feat': cfg.model.input_dims_3d_pc.feat,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat': cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': cfg.model.output_dims_3d_pc.X,
        'feat': cfg.model.output_dims_3d_pc.feat
    }
    
    model = MLP_Net_PC_Only(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model

def create_model_3d_pc_transformer(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    input_dims = {
        'X': cfg.model.input_dims_3d_pc.pos,
        'feat': cfg.model.input_dims_3d_pc.feat,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': cfg.model.output_dims_3d_pc.X,
        'feat': cfg.model.output_dims_3d_pc.feat
    }
    
    model = Transformer_Net_PC_Seq(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model


def create_model_3d_pc_transformer_v2(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    print(f"creating model 3d pc transformer v2...")
    
    input_dims = {
        'X': cfg.model.input_dims_3d_pc.pos,
        'feat': cfg.model.input_dims_3d_pc.feat,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': cfg.model.output_dims_3d_pc.X,
        'feat': cfg.model.output_dims_3d_pc.feat
    }
    
    model = Transformer_Net_PC_Seq_V2(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model

def create_model_3d_pc_transformer_v3(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    print(f"creating model 3d pc transformer v3...")
    
    input_dims = {
        'X': 3,
        'feat': 6,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': 3,
        'feat': 6
    }
    
    model = Transformer_Net_PC_Seq_V3(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model


#### model forthe joint space state-action sequences ####
def create_model_3d_jointspace_transformer_v3(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    
    print(f"creating model 3d pc transformer v3...")
    # input_dims = {
    #     'X': 22, ### the number of tiemsteps and with theaaa
    #     'feat': 22,
    # }
    
    if cfg.training.diff_task_space:
        input_dims = {
            'X': 3,
            'feat': 3
        }
    elif cfg.training.kine_diff:
        concat_two_dims = cfg.training.concat_two_dims
        input_dims = {
            # 'X': 22,
            # 'feat': 3,
            # 'concat_two_dims': concat_two_dims
            'X': 3,
            'feat': 22 + 3 + 3,
            'concat_two_dims': concat_two_dims
        }
    else:
        input_dims = {
            'X': 22,
            'feat': 22,
        }
    
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    # trainin two dims #
    output_dims = {
        'X': input_dims['X'],
        'feat': input_dims['feat']
    }
    
    if cfg.training.kine_diff:
        if cfg.training.AE_Diff:
            model = Transformer_Net_PC_Seq_V3_KineDiff_AE(
                n_layers=cfg.model.n_layers_3d_pc,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU()
            )
        else:
            model = Transformer_Net_PC_Seq_V3_KineDiff(
                n_layers=cfg.model.n_layers_3d_pc,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU()
            )
    else:
        # AE Diff for training the autoencoder and the diffusion separately #
        if cfg.training.AE_Diff:
            model = Transformer_Net_PC_Seq_V3_AE(
                n_layers=cfg.model.n_layers_3d_pc,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(),
                debug=cfg.training.debug
            )
        else:
            model = Transformer_Net_PC_Seq_V3(n_layers=cfg.model.n_layers_3d_pc,
                            input_dims=input_dims,
                            hidden_mlp_dims=hidden_mlp_dims,
                            output_dims=output_dims,
                            act_fn_in=nn.ReLU(),
                            act_fn_out=nn.ReLU())

    return model




#### model forthe joint space state-action sequences ####
def create_model_3d_jointspace_transformer_v3_taskcondv2(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    
    print(f"creating model 3d pc transformer v3...")
    
    if cfg.training.diff_task_space:
        input_dims = {
            'X': 3,
            'feat': 3
        }
    elif cfg.training.kine_diff:
        concat_two_dims = cfg.training.concat_two_dims
        input_dims = {
            'X': 3,
            'feat': 22 + 3 + 3,
            'concat_two_dims': concat_two_dims
        }
    else:
        input_dims = {
            'X': 22, ### the number of tiemsteps and with theaaa
            'feat': 22,
        }
    
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    
    
    output_dims = {
        'X': input_dims['X'],
        'feat': input_dims['feat']
    }
    
    # cond_task_type # 
    if cfg.training.tracking_ctl_diff:
        cond_task_type = 'tracking'
    else:
        cond_task_type = 'rotation'
    
    
    try: 
        sub_task_cond_type = cfg.training.sub_task_cond_type
    except:
        sub_task_cond_type = 'full'
    
    
    if cfg.training.kine_diff:
        # if cfg.training.AE_Diff:
        #     model = Transformer_Net_PC_Seq_V3_KineDiff_AE(
        #         n_layers=cfg.model.n_layers_3d_pc,
        #         input_dims=input_dims,
        #         hidden_mlp_dims=hidden_mlp_dims,
        #         output_dims=output_dims,
        #         act_fn_in=nn.ReLU(),
        #         act_fn_out=nn.ReLU()
        #     )
        # else:
        assert (cfg.training.AE_Diff)
        model = Transformer_Net_PC_Seq_V3_KineDiff_AE(
            n_layers=cfg.model.n_layers_3d_pc,
            input_dims=input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            output_dims=output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
            traj_cond=True
        )
    else:
        if cfg.training.AE_Diff:
            model = Transformer_Net_PC_Seq_V3_AE(
                n_layers=cfg.model.n_layers_3d_pc,
                input_dims=input_dims,
                hidden_mlp_dims=hidden_mlp_dims,
                output_dims=output_dims,
                act_fn_in=nn.ReLU(),
                act_fn_out=nn.ReLU(),
                traj_cond=True,
                task_cond_type=cfg.training.task_cond_type,
                sub_task_cond_type=sub_task_cond_type,
                debug=cfg.training.debug
            )
        else:
            model = Transformer_Net_PC_Seq_V3_wtaskcond_V2(n_layers=cfg.model.n_layers_3d_pc,
                                input_dims=input_dims,
                                hidden_mlp_dims=hidden_mlp_dims,
                                output_dims=output_dims,
                                act_fn_in=nn.ReLU(),
                                act_fn_out=nn.ReLU(),
                                cond_task_type=cond_task_type) # cond task type #
            
    
    return model


def create_model_3d_pc_transformer_v3_taskcond(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    print(f"creating model 3d pc transformer v3...")
    
    input_dims = {
        'X': 3,
        'feat': 6,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': 3,
        'feat': 6
    }
    
    model = Transformer_Net_PC_Seq_V3_wcond(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model





def create_model_3d_pc_transformer_v3_trajtranscond(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    print(f"creating model 3d pc transformer v3...")
    
    input_dims = {
        'X': 3,
        'feat': 6,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_3d_pc.X, 
        'feat':  256, # cfg.model.hidden_mlp_dims_3d_pc.feat,
        't': cfg.model.hidden_mlp_dims_3d_pc.t
    }
    output_dims = {
        'X': 3,
        'feat': 6
    }
    
    model = Transformer_Net_PC_Seq_V3_wcond_V2(n_layers=cfg.model.n_layers_3d_pc,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    
    return model



def create_model_segs(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    ## E: n x n connection relations -- [-1, 1] --> the real values are [0, 1] --- the connectivity relations between different links ##
    ## ## ## ##
    
    input_dims = {
        'X': cfg.model.input_dims_segs.pos,
        'segs': cfg.model.input_dims_segs.segs,
    }
    hidden_mlp_dims = {
        'X': cfg.model.hidden_mlp_dims_segs.X, 
        'segs': cfg.model.hidden_mlp_dims_segs.segs,
        't': cfg.model.hidden_mlp_dims_segs.t
    }
    output_dims = {
        'X': cfg.model.output_dims_segs.X,
        'segs': cfg.model.output_dims_segs.segs
    }
    
    model = MLP_Net_Segs(n_layers=cfg.model.n_layers_segs,
                        input_dims=input_dims,
                        hidden_mlp_dims=hidden_mlp_dims,
                        output_dims=output_dims,
                        act_fn_in=nn.ReLU(),
                        act_fn_out=nn.ReLU())
    
    return model





def create_model_jointspace(cfg):
    ## in the 2-dim setting ##
    ## for the graph node --- link-x-len, link-y-len, 
    ## x-len and y-len should be set to zeros if there is no such links here ##
    ## --- and the vlaues should be normalized ##
    ## how to construct the graph using graph nodes? or how to compose them to the manipulator? --- some thing should be fixed --- such as the linking strategies ##
    ## for two-d --- first half is the left part of links while the second half is the second part of links ###
    ## they are connected via the joints --- a point in the link ##
    ## for each node in the two-d --- treating each link as the capsule or the rectangle -- 
    ## then each link is represented as the rectangle --- the x len and the y len of the rectangle ##
    
    
    input_dims = {
        'X': cfg.model.jointspace_input_dims.X,
        'E': cfg.model.jointspace_input_dims.E, 
        'y': cfg.model.jointspace_input_dims.y 
    }
    
    hidden_mlp_dims = {
        'X': cfg.model.jointspace_hidden_mlp_dims.X,
        'E': cfg.model.jointspace_hidden_mlp_dims.E,
        'y': cfg.model.jointspace_hidden_mlp_dims.y
    }
    
    hidden_dims = {
        'dx': cfg.model.jointspace_hidden_dims.dx,
        'de': cfg.model.jointspace_hidden_dims.de,
        'dy': cfg.model.jointspace_hidden_dims.dy,
        # 'dim_ffX': cfg.model.hidden_dims.dim_ffX,
        # 'dim_ffE': cfg.model.hidden_dims.dim_ffE,
        'n_head': cfg.model.jointspace_hidden_dims.n_head,
    }
    
    output_dims = {
        'X': cfg.model.jointspace_output_dims.X,
        'E': cfg.model.jointspace_output_dims.E,
        'y': cfg.model.jointspace_output_dims.y
    }
    
    model = MLP_Net(n_layers=cfg.model.n_layers,
                             input_dims=input_dims,
                             hidden_mlp_dims=hidden_mlp_dims,
                             hidden_dims=hidden_dims,
                             output_dims=output_dims,
                             act_fn_in=nn.ReLU(),
                             act_fn_out=nn.ReLU())
    return model





def create_gaussian_diffusion(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    betas = gd.get_named_beta_schedule(args.noise_schedule, steps)
    loss_type = gd.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]
    
    return SpacedDiffusion(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not predict_xstart else gd.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
        # lambda_vel=args.lambda_vel,
        # lambda_rcxyz=args.lambda_rcxyz,
        # lambda_fc=args.lambda_fc,
        # denoising_stra=args.denoising_stra,
        # inter_optim=args.inter_optim,
        # args=args,
    )


def create_gaussian_diffusion_act(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    betas = gd_act.get_named_beta_schedule(args.noise_schedule, steps)
    loss_type = gd_act.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusionAct(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_act.ModelMeanType.EPSILON if not predict_xstart else gd_act.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_act.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd_act.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_act.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )
    
def create_gaussian_diffusion_pc(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    betas = gd_pc.get_named_beta_schedule(args.noise_schedule, steps)
    loss_type = gd_pc.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusionPC(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_pc.ModelMeanType.EPSILON if not predict_xstart else gd_pc.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_pc.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd_pc.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_pc.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )


def create_gaussian_diffusion_segs(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    ## gd_segs for diffusion segs ##
    # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    betas = gd_segs.get_named_beta_schedule(args.noise_schedule, steps)
    loss_type = gd_segs.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusionSegs(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_segs.ModelMeanType.EPSILON if not predict_xstart else gd_segs.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_segs.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd_segs.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_segs.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )



def create_gaussian_diffusion_3d_pc(args):
    predict_xstart = True
    steps = 1000
    scale_beta = 1.
    timestep_respacing = ''
    learn_sigma = False
    rescale_timesteps = False

    # betas = gd.get_named_beta_schedule(args.noise_schedule, steps, scale_beta)
    betas = gd_3d_pc.get_named_beta_schedule(args.noise_schedule, steps)
    loss_type = gd_3d_pc.LossType.MSE

    if not timestep_respacing:
        timestep_respacing = [steps]

    return SpacedDiffusion3DPC(
        use_timesteps=space_timesteps(steps, timestep_respacing),
        betas=betas,
        model_mean_type=(
            gd_3d_pc.ModelMeanType.EPSILON if not predict_xstart else gd_3d_pc.ModelMeanType.START_X
        ),
        model_var_type=(
            (
                gd_3d_pc.ModelVarType.FIXED_LARGE
                if not args.sigma_small
                else gd_3d_pc.ModelVarType.FIXED_SMALL
            )
            if not learn_sigma
            else gd_3d_pc.ModelVarType.LEARNED_RANGE
        ),
        loss_type=loss_type,
        rescale_timesteps=rescale_timesteps,
    )

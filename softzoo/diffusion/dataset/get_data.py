from dataset.dataset import Uni_Manip_Dataset
from dataset.dataset import Uni_Manip_Act_Dataset
from dataset.dataset import Uni_Manip_PC_Dataset
from dataset.dataset import Uni_Manip_PCSeg_Dataset
from dataset.dataset import Uni_Manip_3D_PC_Dataset
from dataset.dataset import Uni_Manip_3D_PC_V3_Dataset
from dataset.dataset import Uni_Manip_3D_PC_V5_Dataset
from dataset.dataset import Uni_Manip_3D_PC_V6_Dataset
from dataset.dataset import Uni_Manip_3D_PC_V7_Dataset
from dataset.dataset import Uni_Manip_3D_PC_V7_Cond_Dataset
from dataset.dataset import collect_fn_pc_v7_dataset
from dataset.dataset import Uni_Manip_3D_PC_V8_Dataset
from dataset.dataset import Uni_Manip_3D_PC_V9_Dataset
from torch.utils.data import DataLoader

def get_dataset_loader_act(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_act.data_folder # get the data_folder config ##
    uni_manip_dataset = Uni_Manip_Act_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader

def get_dataset_loader_pc(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_act.data_folder # get the data_folder config ##
    uni_manip_dataset = Uni_Manip_PC_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader


def get_dataset_loader_3d_pc(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_3d_pc.data_folder # get the data_folder config ##
    uni_manip_dataset = Uni_Manip_3D_PC_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader

def get_dataset_loader_3d_v3_pc(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_3d_pc.data_folder # get the data_folder config ##
    
    
    uni_manip_dataset = Uni_Manip_3D_PC_V3_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader


def get_dataset_loader_3d_v5_pc(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_3d_pc.data_folder # get the data_folder config ##
    
    
    uni_manip_dataset = Uni_Manip_3D_PC_V5_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader


def get_dataset_loader_3d_v6_pc(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg):
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_3d_pc.data_folder # get the data_folder config ##
    
    uni_manip_dataset = Uni_Manip_3D_PC_V6_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader

def get_dataset_loader_3d_v7_pc(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg):
    ## data folder ##
    # data_folder = args.dataset.data_folder
    data_folder = args.dataset_3d_pc.data_folder
    if args.training.setting in ['trajectory_forcasting', 'trajectory_forcasting_diffusion']:
        uni_manip_dataset = Uni_Manip_3D_PC_V8_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    elif args.training.setting == 'trajectory_encdec':
        uni_manip_dataset = Uni_Manip_3D_PC_V9_Dataset(data_folder, args)
    elif args.training.task_cond and not args.training.use_jointspace_seq: 
        uni_manip_dataset = Uni_Manip_3D_PC_V7_Cond_Dataset(data_folder, args) 
    else:
        uni_manip_dataset = Uni_Manip_3D_PC_V7_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    shuffle = True if not args.sampling.sampling else False
    print(f"Shuffle: {shuffle}")
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, collate_fn=collect_fn_pc_v7_dataset)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader

def get_dataset_loader_segs(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    # data_folder = args.dataset.data_folder # get the data_folder config ##
    data_folder = args.dataset_segs.data_folder # get the data_folder config ##
    uni_manip_dataset = Uni_Manip_PCSeg_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader

# python -m train.train_mdm --save_dir save/my_humanml_trans_enc_512 --dataset motion_ours
def get_dataset_loader(name, batch_size, num_frames, split='train', hml_mode='train', args=None):
    # __init__(self, data_folder, cfg): ## #
    ## data folder ##
    data_folder = args.dataset.data_folder # get the data_folder config ##
    uni_manip_dataset = Uni_Manip_Dataset(data_folder, args) ## get the manipulator dataset # ## # get the manipulator dataset ### # dataset # # dataset # # dataset # #
    ## dataloader ## --- loader ##
    loader = DataLoader(uni_manip_dataset, batch_size=batch_size, shuffle=True, num_workers=8)
    ## loader --- loader ## # loader # # loader # # loader # # loader # # # # # #
    return loader
    
    dataset = get_dataset(name, num_frames, split, hml_mode, args=args)
    collate = get_collate_fn(name, hml_mode, args=args)
    
    if args is not None and name in ["motion_ours"] and len(args.single_seq_path) > 0:
        shuffle_loader = False
        drop_last = False
    else:
        shuffle_loader = True
        drop_last = True

    num_workers = 8
    num_workers = 16 # 
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle_loader,
        num_workers=num_workers, drop_last=drop_last, collate_fn=collate
    )

    return loader
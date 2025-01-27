import os
from yacs.config import CfgNode as CN

from ..taichi_sim import TaichiSim


class BaseRenderer:
    def __init__(self, sim: TaichiSim, out_dir: str, cfg: CN):
        self.sim = sim
        self.out_dir = out_dir
        self.cfg = cfg
        
        if not os.path.isdir(self.out_dir):
            os.makedirs(self.out_dir)

    def initialize(self):
        pass

    def reset(self):
        raise NotImplementedError # to be defined in children class
    
    def render(self):
        raise NotImplementedError # to be defined in children class

    def close(self):
        raise NotImplementedError # to be defined in children class

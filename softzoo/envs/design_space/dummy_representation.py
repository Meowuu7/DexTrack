from yacs.config import CfgNode as CN
import taichi as ti

from .design_representation import DesignRepresentation
from ...engine import I_DTYPE
from ...engine.taichi_sim import TaichiSim


class DummyRepresentation(DesignRepresentation):
    def __init__(self, sim: TaichiSim, cfg: CN):
        super().__init__(sim, cfg)
        self.sim.solver.no_actuators_setup = False

    def initialize(self):
        pass

    def reset(self):
        pass

    @ti.ad.grad_replaced
    def set_design(self, design):
        pass

    @ti.ad.grad_for(set_design)
    def set_design_grad(self, design):
        pass

    def get_grad(self):
        return dict()

    @ti.func
    def is_robot(self, id: I_DTYPE):
        return 0

    @property
    def size(self):
        return 0

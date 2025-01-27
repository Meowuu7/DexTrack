from yacs.config import CfgNode as CN

from ...tools.general_utils import compute_lame_parameters


__C = CN()

E_0, nu_0 = 1e5, 0.2
mu_0, lambd_0 = compute_lame_parameters(E_0, nu_0)

__C.E_0 = E_0
__C.nu_0 = nu_0
__C.mu_0 = mu_0
__C.lambd_0 = lambd_0
__C.p_rho_0 = 1e3
__C.muscle_direction = None # only used in SimpleMuscle
__C.active = False
__C.max_coef_restitution = 0.1


def get_cfg_defaults():
    return __C.clone()

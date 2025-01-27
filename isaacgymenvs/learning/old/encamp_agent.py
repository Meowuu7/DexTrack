# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import copy
from datetime import datetime
from gym import spaces
import numpy as np
import os
import time
import yaml

from rl_games.algos_torch import a2c_continuous
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import central_value
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.common import a2c_common
from rl_games.common import datasets
from rl_games.common import schedulers
from rl_games.common import vecenv

import torch
from torch import optim

from . import amp_datasets as amp_datasets

from tensorboardX import SummaryWriter


class ENCAMPAgent(a2c_continuous.A2CAgent):
    def __init__(self, base_name, params):
        
        a2c_continuous.A2CAgent.__init__(self, base_name, params)

        self.model.a2c_network.mu_body.weight.requires_grad=False
        self.model.a2c_network.mu_body.bias.requires_grad=False
        self.model.a2c_network.mu_body.weight.requires_grad=False
        self.model.a2c_network.mu_body.weight.requires_grad=False
        # self.optimizer = optim.Adam(self.model.parameters(), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), float(self.last_lr), eps=1e-08, weight_decay=self.weight_decay)
        return

    def restore_from_partial(self, fnb, fnh):
        checkpoint_body = torch_ext.load_checkpoint(fnb)
        checkpoint_hand = torch_ext.load_checkpoint(fnh)

        model_state_dict = self.model.state_dict()

        model_state_dict['a2c_network.mu_body.weight'] = checkpoint_body['model']['a2c_network.mu.weight']
        model_state_dict['a2c_network.mu_body.bias'] = checkpoint_body['model']['a2c_network.mu.bias']

        model_state_dict['a2c_network.mu_hand.weight'] = checkpoint_hand['model']['a2c_network.mu.weight']
        model_state_dict['a2c_network.mu_hand.bias'] = checkpoint_hand['model']['a2c_network.mu.bias']
        
        # self.model.load_state_dict(checkpoint_body['model'])
        # self.model.load_state_dict(checkpoint_hand['model'])

        self.model.load_state_dict(model_state_dict) 
        pass

    def restore(self, fn):
        return super().restore(fn)
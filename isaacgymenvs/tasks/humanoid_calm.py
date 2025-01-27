# Copyright (c) 2018-2022, NVIDIA Corporation
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

from enum import Enum
import numpy as np
import torch

from tasks.humanoid_amp import HumanoidAMP
from tasks.humanoid_amp import *


class HumanoidCALM(HumanoidAMP):
    
    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        self.cfg = cfg

        # numAMPEncObsSteps

        super().__init__(cfg=self.cfg, 
                         rl_device=rl_device, 
                         sim_device=sim_device, 
                         graphics_device_id=graphics_device_id, 
                         headless=headless, 
                         virtual_screen_capture=virtual_screen_capture, 
                         force_render=force_render,
                        )

        self.num_enc_amp_obs = self._num_amp_obs_enc_steps * self._num_amp_obs_per_step
        self._enc_amp_obs_space = spaces.Box(np.ones(self.num_enc_amp_obs) * -np.Inf, np.ones(self.num_enc_amp_obs) * np.Inf)

        return

    # debug 
    @property
    def task_obs_size(self):
        return 0 
    


    @property
    def amp_observation_space(self):
        return self._amp_obs_space
    
    @property
    def enc_amp_observation_space(self):
        return self._enc_amp_obs_space

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

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class PointNetModule(nn.Module):
    ## then the observation to the env is xxx ##
    ## the observation to the policy network is about 1024? ###
    def __init__(self, output_latent_dim=256):
        super().__init__()
        
        self.pc_input_feat_dim = 3
        self.output_latent_dim = output_latent_dim
        
        # 213 #
        # self.pts_feats_encoding_net= nn.Sequential(
        #     nn.Linear(self.pc_input_feat_dim, 128), nn.ReLU(),
        #     nn.Linear(128, 256), nn.ReLU(),
        #     nn.Linear(256, 213),
        #     # nn.ReLU(),
        #     # nn.Linear(512, 1024)
        # )

        self.pts_feats_encoding_net= nn.Sequential(
            nn.Linear(self.pc_input_feat_dim, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(), 
            nn.Linear(512, self.output_latent_dim) # output latent dim #
        )
    def forward(self, pc_input):
        # bsz x nn_flatten_dim
        bsz = pc_input.size(0)
        pc =pc_input.contiguous().view(bsz, -1, self.pc_input_feat_dim).contiguous()
        pc_feats = self.pts_feats_encoding_net(pc) # 
        pc_feats, _ = torch.max(pc_feats, dim=1) ## bsz x nn_latent_dim #
        # pc_input = pc_input.permute(0, 2, 1)
        # pc_feats = self.pts_feats_encoding_net(pc_input)
        # pc_feats = torch.max(pc_feats, 2)[0]
        return pc_feats


class VisionPPOBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            
            ## add a pointnet here ## is continuous #
            
            ### add a pointnet here ###
            
            # if self.is_continuous:
            #     if (not self.space_config['learn_sigma']):
            #         actions_num = kwargs.get('actions_num')
            #         sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            #         self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
            #         sigma_init(self.sigma)
                    
            # amp_input_shape = kwargs.get('amp_input_shape')
            # self._build_disc(amp_input_shape)
            
            ##### Construct the pointnet module #####
            output_latent_dim = 1493
            self.pc_net = PointNetModule(output_latent_dim=output_latent_dim)
            # 213

            return

        def load(self, params): # s
            super().load(params) 

            ## TODO: 
            # self._disc_units = params['disc']['units']
            # self._disc_activation = params['disc']['activation']
            # self._disc_initializer = params['disc']['initializer']
            return

        # def eval_critic(self, obs):
        #     c_out = self.critic_cnn(obs)
        #     c_out = c_out.contiguous().view(c_out.size(0), -1)
        #     c_out = self.critic_mlp(c_out)              
        #     value = self.value_act(self.value(c_out))
        #     return value

        # def eval_disc(self, amp_obs):
        #     disc_mlp_out = self._disc_mlp(amp_obs)
        #     disc_logits = self._disc_logits(disc_mlp_out)
        #     return disc_logits

        # def get_disc_logit_weights(self):
        #     return torch.flatten(self._disc_logits.weight)

        # def get_disc_weights(self):
        #     weights = []
        #     for m in self._disc_mlp.modules():
        #         if isinstance(m, nn.Linear):
        #             weights.append(torch.flatten(m.weight))

        #     weights.append(torch.flatten(self._disc_logits.weight))
        #     return weights

        # def _build_disc(self, input_shape): # add the amp network ? #
        #     self._disc_mlp = nn.Sequential()

        #     mlp_args = {
        #         'input_size' : input_shape[0], 
        #         'units' : self._disc_units, 
        #         'activation' : self._disc_activation, 
        #         'dense_func' : torch.nn.Linear
        #     }
        #     self._disc_mlp = self._build_mlp(**mlp_args)
            
        #     mlp_out_size = self._disc_units[-1]
        #     self._disc_logits = torch.nn.Linear(mlp_out_size, 1)

        #     ## init factory ## # load the expert ? # # register network; 
        #     mlp_init = self.init_factory.create(**self._disc_initializer)
        #     for m in self._disc_mlp.modules():
        #         if isinstance(m, nn.Linear):
        #             mlp_init(m.weight)
        #             if getattr(m, "bias", None) is not None:
        #                 torch.nn.init.zeros_(m.bias) 

        #     torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        #     torch.nn.init.zeros_(self._disc_logits.bias) 

            # return

    def build(self, name, **kwargs):
        net = VisionPPOBuilder.Network(self.params, **kwargs)
        return net
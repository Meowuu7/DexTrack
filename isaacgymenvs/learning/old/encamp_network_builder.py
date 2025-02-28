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

# from learning import amp_network_builder

DISC_LOGIT_INIT_SCALE = 1.0

class ENCAMPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            super().__init__(params, **kwargs)

            # if self.is_continuous:
            #     if not self.space_config['learn_sigma']:
            #         actions_num = kwargs.get('actions_num')
            #         sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            #         self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
            #         sigma_init(self.sigma)
                    
            # amp_input_shape = kwargs.get('amp_input_shape')
            # self._build_disc(amp_input_shape)

            input_shape = kwargs.pop('input_shape')
            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)


            actions_num = kwargs.pop('actions_num')
            in_mlp_shape = mlp_input_shape
            if len(self.units) == 0:
                out_size = mlp_input_shape
            else:
                out_size = self.units[-1]

            # self.mu = torch.nn.Linear(out_size, actions_num)
            # self.mu_act = self.activations_factory.create(self.space_config['mu_activation']) 
            # mu_init = self.init_factory.create(**self.space_config['mu_init'])
            # self.sigma_act = self.activations_factory.create(self.space_config['sigma_activation']) 
            # sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            # self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)


            actions_num_body = 28
            # actions_num_hand = 24 # wrj1 + wrj0
            actions_num_hand = 22
            self.mu_body = torch.nn.Linear(512, actions_num_body)
            self.mu_act_body = self.activations_factory.create(self.space_config['mu_activation']) 
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act_body = self.activations_factory.create(self.space_config['sigma_activation']) 
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            self.sigma_body = nn.Parameter(torch.zeros(actions_num_body, requires_grad=True, dtype=torch.float32), requires_grad=True)

            self.mu_hand = torch.nn.Linear(100, actions_num_hand)
            self.mu_act_hand = self.activations_factory.create(self.space_config['mu_activation']) 
            mu_init = self.init_factory.create(**self.space_config['mu_init'])
            self.sigma_act_hand = self.activations_factory.create(self.space_config['sigma_activation']) 
            sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
            self.sigma_hand = nn.Parameter(torch.zeros(actions_num_hand, requires_grad=True, dtype=torch.float32), requires_grad=True)


            mu_init(self.mu_body.weight)
            mu_init(self.mu_hand.weight)
            sigma_init(self.sigma_body)
            sigma_init(self.sigma_hand)
            return

        def load(self, params):
            super().load(params)

            # self._disc_units = params['disc']['units']
            # self._disc_activation = params['disc']['activation']
            # self._disc_initializer = params['disc']['initializer']
            return

        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)

            actor_outputs = self.eval_actor(obs)
            value = self.eval_critic(obs)

            output = actor_outputs + (value, states)

            return output

        def eval_actor(self, obs):
            a_out = self.actor_cnn(obs)
            a_out = a_out.contiguous().view(a_out.size(0), -1)
            a_out = self.actor_mlp(a_out)

            # h_size = a_out.shape[-1]//2
            latent_body = a_out[...,:512]
            latent_hand = a_out[...,512:]


            mu_body = self.mu_act_body(self.mu_body(latent_body))
            if self.space_config['fixed_sigma']:
                sigma_body = mu_body * 0.0 + self.sigma_act_body(self.sigma[:28])
            else:
                sigma_body = self.sigma_act_body(self.sigma_body(latent_body))

            mu_hand = self.mu_act_hand(self.mu_hand(latent_hand))
            if self.space_config['fixed_sigma']:
                sigma_hand = mu_hand * 0.0 + self.sigma_act_hand(self.sigma[28:])
            else:
                sigma_hand = self.sigma_act_hand(self.sigma_hand(latent_hand)) 


            # mu_body, sigma_body = self.actor_mlp_body(latent_body)
            # mu_hand, sigma_hand = self.actor_mlp_hand(latent_hand)

            mu = torch.cat([mu_body, mu_hand], dim=-1)
            sigma = torch.cat([sigma_body, sigma_hand], dim=-1)
            return mu, sigma
                     

        def eval_critic(self, obs):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

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

        # def _build_disc(self, input_shape):
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

        #     mlp_init = self.init_factory.create(**self._disc_initializer)
        #     for m in self._disc_mlp.modules():
        #         if isinstance(m, nn.Linear):
        #             mlp_init(m.weight)
        #             if getattr(m, "bias", None) is not None:
        #                 torch.nn.init.zeros_(m.bias) 

        #     torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
        #     torch.nn.init.zeros_(self._disc_logits.bias) 

        #     return

    def build(self, name, **kwargs):
        net = ENCAMPBuilder.Network(self.params, **kwargs)
        return net

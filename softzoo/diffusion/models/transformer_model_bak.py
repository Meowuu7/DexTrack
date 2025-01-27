import math

import torch
import torch.nn as nn
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm
from torch.nn import functional as F
from torch import Tensor

from models import utils
from diffusion import diffusion_utils
from models.layers import Xtoy, Etoy, masked_softmax
import numpy as np


# 
def batched_index_select(values, indices, dim = 1):
  value_dims = values.shape[(dim + 1):]
  values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
  indices = indices[(..., *((None,) * len(value_dims)))]
  indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
  value_expand_len = len(indices_shape) - (dim + 1)
  values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

  value_expand_shape = [-1] * len(values.shape)
  expand_slice = slice(dim, (dim + value_expand_len))
  value_expand_shape[expand_slice] = indices.shape[expand_slice]
  values = values.expand(*value_expand_shape)

  dim += value_expand_len
  return values.gather(dim, indices)


class XEyTransformerLayer(nn.Module):
    """ Transformer that updates node, edge and global features
        d_x: node features
        d_e: edge features
        dz : global features
        n_head: the number of heads in the multi_head_attention
        dim_feedforward: the dimension of the feedforward network model after self-attention
        dropout: dropout probablility. 0 to disable
        layer_norm_eps: eps value in layer normalizations. # layer normalization ## layer norm epxs # 
    """
    def __init__(self, dx: int, de: int, dy: int, n_head: int, dim_ffX: int = 2048,
                 dim_ffE: int = 128, dim_ffy: int = 2048, dropout: float = 0.1,
                 layer_norm_eps: float = 1e-5, device=None, dtype=None) -> None:
        kw = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = NodeEdgeBlock(dx, de, dy, n_head, **kw)

        self.linX1 = Linear(dx, dim_ffX, **kw)
        self.linX2 = Linear(dim_ffX, dx, **kw)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, **kw)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, **kw)
        self.linE2 = Linear(dim_ffE, de, **kw)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, **kw)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, **kw)
        self.lin_y2 = Linear(dim_ffy, dy, **kw)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, **kw)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

        self.activation = F.relu

    def forward(self, X: Tensor, E: Tensor, y, node_mask: Tensor):
        """ Pass the input through the encoder layer.
            X: (bs, n, d)
            E: (bs, n, n, d)
            y: (bs, dy)
            node_mask: (bs, n) Mask for the src keys per batch (optional)
            Output: newX, newE, new_y with the same shape.
        """

        newX, newE, new_y = self.self_attn(X, E, y, node_mask=node_mask)

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(self.activation(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(self.activation(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(self.activation(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """ Self attention layer that also updates the representations on the edges. """
    def __init__(self, dx, de, dy, n_head, **kwargs):
        super().__init__()
        assert dx % n_head == 0, f"dx: {dx} -- nhead: {n_head}"
        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Attention
        self.q = Linear(dx, dx) #
        self.k = Linear(dx, dx)
        self.v = Linear(dx, dx)

        # FiLM E to X
        self.e_add = Linear(de, dx)
        self.e_mul = Linear(de, dx)

        # FiLM y to E
        self.y_e_mul = Linear(dy, dx)           # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx) # from the global feature to the node feature

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx) # 
        self.y_x_add = Linear(dy, dx)

        # Process y
        self.y_y = Linear(dy, dy) # process the y features #
        self.x_y = Xtoy(dx, dy) 
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx) # output layer for  the node features 
        self.e_out = Linear(dx, de) # output for edge featwures --- de is the edge feature dimension? #
        self.y_out = nn.Sequential(nn.Linear(dy, dy), nn.ReLU(), nn.Linear(dy, dy))

    def forward(self, X, E, y, node_mask):
        """
        :param X: bs, n, d        node features
        :param E: bs, n, n, d     edge features
        :param y: bs, dz           global features
        :param node_mask: bs, n
        :return: newX, newE, new_y with the same shape.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)        # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)           # bs, n, 1, 1 # emasks1 j
        e_mask2 = x_mask.unsqueeze(1)           # bs, 1, n, 1

        # 1. Map X to keys and queries ## map the features X to the qs anqueries #
        Q = self.q(X) * x_mask           # (bs, n, dx)
        K = self.k(X) * x_mask           # (bs, n, dx)
        diffusion_utils.assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)                              # (bs, 1, n, n_head, df)
        K = K.unsqueeze(1)                              # (bs, n, 1, n head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        diffusion_utils.assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2                        # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2                  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)                      # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, de
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2      # bs, n, n, de
        diffusion_utils.assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)    # bs, 1, n, 1
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        V = self.v(X) * x_mask                        # bs, n, dx
        V = V.reshape((V.size(0), V.size(1), self.n_head, self.df))
        V = V.unsqueeze(1)                                     # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * V
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)            # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        diffusion_utils.assert_correctly_masked(newX, x_mask)

        # Process y based on X axnd E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)               # bs, dy

        return newX, newE, new_y


class MLP_Act_Net(nn.Module):
    def __init__(self, n_layers:  int, input_dim: int, hidden_mlp_dims: dict, output_dim: int, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_mlp_dims = hidden_mlp_dims
        self.output_dim = output_dim
        
        act_data_embedding_mlps_list = []
        for i_layer in range(n_layers - 1):
            if i_layer == 0:
                cur_mlp_layer = nn.Linear(self.input_dim, hidden_mlp_dims['X'])
            else:
                cur_mlp_layer = nn.Linear(self.hidden_mlp_dims['X'], self.hidden_mlp_dims['X'])
            act_data_embedding_mlps_list.append(cur_mlp_layer)
            act_data_embedding_mlps_list.append(act_fn_in)
        act_data_embedding_mlps_list.append(nn.Linear(self.hidden_mlp_dims['X'], self.hidden_mlp_dims['X']))
        
        
        y_data_embedding_mlps_list = []
        for i_layer in range(n_layers - 1):
            if i_layer == 0:
                cur_mlp_layer = nn.Linear(1, hidden_mlp_dims['y'])
            else:
                cur_mlp_layer = nn.Linear(self.hidden_mlp_dims['y'], self.hidden_mlp_dims['y'])
            y_data_embedding_mlps_list.append(cur_mlp_layer)
            y_data_embedding_mlps_list.append(act_fn_in)
        y_data_embedding_mlps_list.append(nn.Linear(self.hidden_mlp_dims['y'], self.hidden_mlp_dims['y']))
        
        self.act_data_embedding_mlps = nn.Sequential(*act_data_embedding_mlps_list)
        self.y_data_embedding_mlps = nn.Sequential(*y_data_embedding_mlps_list)
        
        self.concate_hidden_dims = self.hidden_mlp_dims['X'] + self.hidden_mlp_dims['y']
        
        
        self.act_data_output_mlps = nn.Sequential(
            nn.Linear(self.concate_hidden_dims, self.hidden_mlp_dims['X']), act_fn_out, 
            nn.Linear(self.hidden_mlp_dims['X'], self.output_dim)
        )
    
    def forward(self, X, y, node_mask):
        bsz = X.size(0)
        
        if y.dtype == torch.int32 or y.dtype == torch.long:
            y = y.float() / 1000.0

        
        nn_nodes = X.size(1)
        # nn_nodes_feats = X.size(2)
        nn_ts, nn_act_dim = X.size(2), X.size(3) ## get the X sizes and the dimensions ##
        
        flatten_X = X.contiguous().view(bsz, -1).contiguous()
        # flatten_E = E.contiguous().view(bsz, -1).contiguous()
        flatten_y = y.contiguous().view(bsz, -1).contiguous()
        
        X_embedding = self.act_data_embedding_mlps(flatten_X)
        # E_embedding = self.E_feat_embedding_mlps(flatten_E)
        y_embedding = self.y_data_embedding_mlps(flatten_y)
        
        concat_embedding = torch.cat(
            [X_embedding,  y_embedding], dim=-1
        )
        
        X_out = self.act_data_output_mlps(concat_embedding)
        # E_out = self.E_output_layer(concat_embedding)
        
        X_out = X_out.contiguous().view(bsz, nn_nodes, -1).contiguous()
        # E_out = E_out.contiguous().view(bsz, nn_nodes, nn_nodes, 1).contiguous()
        
        ## nn_nodes x timestaps x act_dim ##
        X_out = X_out.contiguous().view(bsz, nn_nodes, nn_ts, nn_act_dim).contiguous()
        return X_out ## Xout in the original dimensions ##
        
        
        # X is the mlp and y is the mlp #

class MLP_Net(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        ####### X E y feature input dimensions #######
        # self.X_feat_in_dim = 21 * 2
        # self.E_feat_in_dim = 21 * 21 ## E features
        # self.y_feat_in_dim = 1 ## y features in dim
        ####### X E y feature input dimensions #######
        
        self.X_feat_in_dim = input_dims['X']
        self.E_feat_in_dim = input_dims['E']
        self.y_feat_in_dim = input_dims['y']
        
        
        
        # self.X_feat_embedding_mlps = nn.Sequential(
        #     [nn.Linear(self.X_feat_in_dim, hidden_mlp_dims['X']), act_fn_in] * (n_layers - 1)
        # )
        X_feat_embedding_mlps_list = []
        for i_layer in range(n_layers - 1):
            if i_layer == 0:
                cur_mlp_layer = nn.Linear(self.X_feat_in_dim, hidden_mlp_dims['X'])
            else:
                cur_mlp_layer = nn.Linear(hidden_mlp_dims['X'], hidden_mlp_dims['X'])
            X_feat_embedding_mlps_list.append(cur_mlp_layer)
            X_feat_embedding_mlps_list.append(act_fn_in)
        X_feat_embedding_mlps_list.append(nn.Linear(hidden_mlp_dims['X'], hidden_mlp_dims['X']))
        
        E_feat_embedding_mlp_list = []
        for i_layer in range(n_layers - 1):
            if i_layer == 0:
                cur_mlp_layer = nn.Linear(self.E_feat_in_dim, hidden_mlp_dims['E'])
            else:
                cur_mlp_layer = nn.Linear(hidden_mlp_dims['E'], hidden_mlp_dims['E'])
            E_feat_embedding_mlp_list.append(cur_mlp_layer)
            E_feat_embedding_mlp_list.append(act_fn_in)
        E_feat_embedding_mlp_list.append(nn.Linear(hidden_mlp_dims['E'], hidden_mlp_dims['E']))
        
        y_feat_embedding_mlp_list = []
        for i_layer in range(n_layers - 1):
            if i_layer == 0:
                cur_mlp_layer = nn.Linear(self.y_feat_in_dim, hidden_mlp_dims['y'])
            else:
                cur_mlp_layer = nn.Linear(hidden_mlp_dims['y'], hidden_mlp_dims['y'])
            y_feat_embedding_mlp_list.append(cur_mlp_layer)
            y_feat_embedding_mlp_list.append(act_fn_in)
        y_feat_embedding_mlp_list.append(nn.Linear(hidden_mlp_dims['y'], hidden_mlp_dims['y']))
        
        ### Get the X_feat, E_feat, and y_feat embedding mlps ###
        ### Get the X_feat, E_feat, and y_feat embedding mlps ###
        self.X_feat_embedding_mlps = nn.Sequential(
            *X_feat_embedding_mlps_list
        )
        self.E_feat_embedding_mlps = nn.Sequential(
            *E_feat_embedding_mlp_list
        )
        self.y_feat_embedding_mlps = nn.Sequential(
            *y_feat_embedding_mlp_list
        )
        
        
        ## bsz x [x_embeddings, e_embeddings, y_embeddings] ##
        concat_embedding_dim = hidden_mlp_dims['X'] + hidden_mlp_dims['E'] + hidden_mlp_dims['y']
        # X_output_mlp_list = []
        self.X_output_layer = nn.Sequential(
            nn.Linear(concat_embedding_dim, hidden_mlp_dims['X']), act_fn_out, nn.Linear(hidden_mlp_dims['X'], self.X_feat_in_dim)
        )
        self.E_output_layer = nn.Sequential(
            nn.Linear(concat_embedding_dim, hidden_mlp_dims['E']), act_fn_out, nn.Linear(hidden_mlp_dims['E'], self.E_feat_in_dim)
        )
        
    def forward(self, X, E, y, node_mask):
        bsz = X.size(0)
        
        if y.dtype == torch.int32 or y.dtype == torch.long:
            y = y.float() / 1000.0

        
        nn_nodes = X.size(1)
        nn_nodes_feats = X.size(2)
        
        flatten_X = X.contiguous().view(bsz, -1).contiguous()
        flatten_E = E.contiguous().view(bsz, -1).contiguous()
        flatten_y = y.contiguous().view(bsz, -1).contiguous()
        
        X_embedding = self.X_feat_embedding_mlps(flatten_X)
        E_embedding = self.E_feat_embedding_mlps(flatten_E)
        y_embedding = self.y_feat_embedding_mlps(flatten_y)
        
        concat_embedding = torch.cat(
            [X_embedding, E_embedding, y_embedding], dim=-1
        )
        
        X_out = self.X_output_layer(concat_embedding)
        E_out = self.E_output_layer(concat_embedding)
        
        X_out = X_out.contiguous().view(bsz, nn_nodes, -1).contiguous()
        E_out = E_out.contiguous().view(bsz, nn_nodes, nn_nodes, 1).contiguous()
        
        return utils.PlaceHolder(X=X_out, E=E_out, y=y)
        

        
### mlp net for processing point cloud data ###
class MLP_Net_PC(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        self.glb_feat_trans_dim = 2048 # concate the pos hidden and the feat hidden --- then the glb trans glock # 
        self.t_in_dim = 1
        self.t_hidden_dim = hidden_mlp_dims['t'] ## hidden dim for time sequence ## ## cocate wtih the glb feature ##
        
        self.decoder_in_dim = self.glb_feat_trans_dim + self.pos_in_dim + self.feat_in_dim + self.t_hidden_dim ## decoder input dim ## for each point
        
        self.pos_out_dim = output_dims['X']
        self.feat_out_dim = output_dims['feat']
        
        self.pos_embedding_mlps = nn.Sequential(
            * [nn.Linear(self.pos_in_dim, self.pos_hidden_dim)] + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim)]
        )
        self.feat_embedding_mlps = nn.Sequential(
            * [nn.Linear(self.feat_in_dim, self.feat_hidden_dim)] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)]
        )
        self.glb_feat_trans_mlps = nn.Sequential(
            * [nn.Linear(self.pos_hidden_dim + self.feat_hidden_dim, self.glb_feat_trans_dim), act_fn_in] + [nn.Linear(self.glb_feat_trans_dim, self.glb_feat_trans_dim)]
        ) ## a two-layer mlps for getting the trasformed glb features
        self.t_trans_mlps = nn.Sequential(
            * [nn.Linear(self.t_in_dim, self.t_hidden_dim), act_fn_in] + [nn.Linear(self.t_hidden_dim, self.t_hidden_dim)]
        ) ## two-layer for t transformation
        
        self.pos_decoder_mlps = nn.Sequential(
            * [nn.Linear(self.decoder_in_dim, self.pos_hidden_dim), act_fn_in] + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim)] * (n_layers - 1) + [nn.Linear(self.pos_hidden_dim, self.pos_out_dim)]
        )
        self.feat_decoder_mlps = nn.Sequential(
            * [nn.Linear(self.decoder_in_dim, self.feat_hidden_dim), act_fn_in] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_out_dim)] # 
        ) # get feature decoder #
        
        
        
        # self.X_feat_in_dim = 21 * 2
        # self.E_feat_in_dim = 21 * 21 ## E features
        # self.y_feat_in_dim = 1 ## y features in dim
        
        
        # # self.X_feat_embedding_mlps = nn.Sequential( ## x feat j
        # #     [nn.Linear(self.X_feat_in_dim, hidden_mlp_dims['X']), act_fn_in] * (n_layers - 1)
        # # )
        # X_feat_embedding_mlps_list = []
        # for i_layer in range(n_layers - 1): # 
        #     if i_layer == 0:
        #         cur_mlp_layer = nn.Linear(self.X_feat_in_dim, hidden_mlp_dims['X'])
        #     else:
        #         cur_mlp_layer = nn.Linear(hidden_mlp_dims['X'], hidden_mlp_dims['X'])
        #     X_feat_embedding_mlps_list.append(cur_mlp_layer)
        #     X_feat_embedding_mlps_list.append(act_fn_in)
        # X_feat_embedding_mlps_list.append(nn.Linear(hidden_mlp_dims['X'], hidden_mlp_dims['X']))
        
        # E_feat_embedding_mlp_list = []
        # for i_layer in range(n_layers - 1):
        #     if i_layer == 0:
        #         cur_mlp_layer = nn.Linear(self.E_feat_in_dim, hidden_mlp_dims['E'])
        #     else:
        #         cur_mlp_layer = nn.Linear(hidden_mlp_dims['E'], hidden_mlp_dims['E'])
        #     E_feat_embedding_mlp_list.append(cur_mlp_layer)
        #     E_feat_embedding_mlp_list.append(act_fn_in)
        # E_feat_embedding_mlp_list.append(nn.Linear(hidden_mlp_dims['E'], hidden_mlp_dims['E']))
        
        # y_feat_embedding_mlp_list = []
        # for i_layer in range(n_layers - 1):
        #     if i_layer == 0:
        #         cur_mlp_layer = nn.Linear(self.y_feat_in_dim, hidden_mlp_dims['y'])
        #     else:
        #         cur_mlp_layer = nn.Linear(hidden_mlp_dims['y'], hidden_mlp_dims['y'])
        #     y_feat_embedding_mlp_list.append(cur_mlp_layer)
        #     y_feat_embedding_mlp_list.append(act_fn_in)
        # y_feat_embedding_mlp_list.append(nn.Linear(hidden_mlp_dims['y'], hidden_mlp_dims['y']))
        
        # ### Get the X_feat, E_feat, and y_feat embedding mlps ###
        # self.X_feat_embedding_mlps = nn.Sequential(
        #     *X_feat_embedding_mlps_list
        # )
        # self.E_feat_embedding_mlps = nn.Sequential(
        #     *E_feat_embedding_mlp_list
        # )
        # self.y_feat_embedding_mlps = nn.Sequential(
        #     *y_feat_embedding_mlp_list
        # )
        
        
        # ## bsz x [x_embeddings, e_embeddings, y_embeddings] ##
        # concat_embedding_dim = hidden_mlp_dims['X'] + hidden_mlp_dims['E'] + hidden_mlp_dims['y']
        # # X_output_mlp_list = []
        # self.X_output_layer = nn.Sequential(
        #     nn.Linear(concat_embedding_dim, hidden_mlp_dims['X']), act_fn_out, nn.Linear(hidden_mlp_dims['X'], self.X_feat_in_dim)
        # )
        # self.E_output_layer = nn.Sequential(
        #     nn.Linear(concat_embedding_dim, hidden_mlp_dims['E']), act_fn_out, nn.Linear(hidden_mlp_dims['E'], self.E_feat_in_dim)
        # )
        
        # 
        
    def forward(self, X, feat, y, node_mask=None, cond=None):
        bsz = X.size(0)
        
        
        if len(X.size()) == 4:
            X = X[:, 0]
            feat = feat[:, 0]
            additional_dim = True 
        else:
            additional_dim = False
        
        if y.dtype == torch.int32 or y.dtype == torch.long:
            y = y.float() / 1000.0

        
        
        # X : bsz x nn_points  x 3
        # feat: bsz x nn_points  x (T x per_time_act_dim)
        
        nn_points = X.size(1)
        
        pos_embedding = self.pos_embedding_mlps(X) ## bsz x nn_points x pos_hidden_dim ##
        feat_flatten = feat.contiguous().view(bsz, nn_points, -1).contiguous() ## bsz x nn_points x (act_in_dim)
        feat_embedding = self.feat_embedding_mlps(feat_flatten) ## bsz x nn_points x fea_hidden_dim ##
        pos_feat_embedding = torch.cat([pos_embedding, feat_embedding], dim=-1) ## bsz x nn_points x (pos_hidden_dim + feat_hidden_dim) ## # 
        glb_pos_feat_embedding, _ = torch.max(pos_feat_embedding, dim=1) ## bsz x hidden dims ## 
        glb_pos_feat_embedding = self.glb_feat_trans_mlps(glb_pos_feat_embedding) ## bsz x hidden dim ##
        
        t_embedding = self.t_trans_mlps(y) ## bsz x t_embeddings ##
        glb_pos_feat_t_embedding = torch.cat(
            [glb_pos_feat_embedding, t_embedding], dim=-1
        )
        
        glb_pos_feat_t_embedding_expanded = glb_pos_feat_t_embedding.unsqueeze(1).repeat(1, nn_points, 1) ## bsz x nn_points x hidden_dim ##
        # print(f"glb_pos_feat_t_embedding_expanded: {glb_pos_feat_t_embedding_expanded.size()}, feat_flatten: {feat_flatten.size()}, X: {X.size()}")
        decoder_in_feats = torch.cat(
            [glb_pos_feat_t_embedding_expanded, feat_flatten, X], dim=-1 ## bsz x nn_points x (hidden_dim + hidden_dim) ##
        )

        
        x_out  = self.pos_decoder_mlps(decoder_in_feats) ## bsz x nn_points x 3 ##
        feat_out = self.feat_decoder_mlps(decoder_in_feats) ## bsz x nn_points x feat_out_dim ##
        
        
        if additional_dim:
            x_out = x_out.unsqueeze(1)
            feat_out = feat_out.unsqueeze(1)
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        
        # nn_nodes = X.size(1)
        # nn_nodes_feats = X.size(2)
        
        # flatten_X = X.contiguous().view(bsz, -1).contiguous()
        # flatten_E = E.contiguous().view(bsz, -1).contiguous()
        # flatten_y = y.contiguous().view(bsz, -1).contiguous()
        
        # X_embedding = self.X_feat_embedding_mlps(flatten_X)
        # E_embedding = self.E_feat_embedding_mlps(flatten_E)
        # y_embedding = self.y_feat_embedding_mlps(flatten_y)
        
        # concat_embedding = torch.cat(
        #     [X_embedding, E_embedding, y_embedding], dim=-1
        # )
        
        # X_out = self.X_output_layer(concat_embedding)
        # E_out = self.E_output_layer(concat_embedding)
        
        # X_out = X_out.contiguous().view(bsz, nn_nodes, -1).contiguous()
        # E_out = E_out.contiguous().view(bsz, nn_nodes, nn_nodes, 1).contiguous()
        
        
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)


# 

### mlp net for processing point cloud data ###
class MLP_Net_PC_Only(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        self.glb_feat_trans_dim = 2048 # concate the pos hidden and the feat hidden --- then the glb trans glock # 
        self.t_in_dim = 1
        self.t_hidden_dim = hidden_mlp_dims['t'] ## hidden dim for time sequence ## ## cocate wtih the glb feature ##
        
        self.decoder_in_dim = self.glb_feat_trans_dim + self.t_hidden_dim +  self.pos_hidden_dim ## decoder input dim ## for each point
        
        self.pos_out_dim = output_dims['X']
        self.feat_out_dim = output_dims['feat']
        
        # self.time # feat # # feat 
        
        self.pos_embedding_mlps = nn.Sequential(
            * [nn.Linear(self.pos_in_dim, self.pos_hidden_dim)] + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim)]
        )
        self.pos_embedding_feat_mlps = nn.Sequential(
            * [nn.Linear(self.pos_in_dim, self.feat_hidden_dim)] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)]
        )
        self.feat_embedding_mlps = nn.Sequential(
            * [nn.Linear(self.feat_in_dim, self.feat_hidden_dim)] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)]
        )
        self.glb_feat_trans_mlps = nn.Sequential(
            * [nn.Linear(self.pos_hidden_dim + self.feat_hidden_dim, self.glb_feat_trans_dim), act_fn_in] + [nn.Linear(self.glb_feat_trans_dim, self.glb_feat_trans_dim)]
        ) ## a two-layer mlps for getting the trasformed glb features ##
        self.t_trans_mlps = nn.Sequential(
            * [nn.Linear(self.t_in_dim, self.t_hidden_dim), act_fn_in] + [nn.Linear(self.t_hidden_dim, self.t_hidden_dim)]
        ) ## two-layer for t transformation
        
        self.time_embedder = TimestepEmbedderV2(
            latent_dim=self.t_hidden_dim, max_len=5000
        )
        
        self.pos_decoder_mlps = nn.Sequential(
            * [nn.Linear(self.decoder_in_dim, self.pos_hidden_dim), act_fn_in] + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim)] * (n_layers - 1) + [nn.Linear(self.pos_hidden_dim, self.pos_out_dim)]
        )
        self.feat_decoder_mlps = nn.Sequential(
            * [nn.Linear(self.decoder_in_dim, self.feat_hidden_dim), act_fn_in] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_out_dim)] # 
        ) # get feature decoder #
        
        
        
    def forward(self, X, feat, y, node_mask):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # X : bsz x nn_points  x 3
        # feat: bsz x nn_points  x (T x per_time_act_dim)
        
        nn_points = X.size(1)
        
        pos_embedding = self.pos_embedding_mlps(X) ## bsz x nn_points x pos_hidden_dim ##
        feat_flatten = feat.contiguous().view(bsz, nn_points, -1).contiguous() ## bsz x nn_points x (act_in_dim)
        feat_embedding = self.feat_embedding_mlps(feat_flatten) ## bsz x nn_points x fea_hidden_dim ##
        
        feat_pos_embedding = self.pos_embedding_feat_mlps(X)
        pos_feat_embedding = torch.cat([pos_embedding, feat_pos_embedding], dim=-1) ## bsz x nn_points x (pos_hidden_dim + feat_hidden_dim) ## # 
        glb_pos_feat_embedding, _ = torch.max(pos_feat_embedding, dim=1) ## bsz x hidden dims ## 
        glb_pos_feat_embedding = self.glb_feat_trans_mlps(glb_pos_feat_embedding) ## bsz x hidden dim ##
        
        # t_embedding = self.t_trans_mlps(y) ## bsz x t_embeddings ##
        
        t_embedding = self.time_embedder(y.squeeze(-1))
        
        # print(f"t_embedding: {t_embedding.size()}, y: {y.size()}")
        
        glb_pos_feat_t_embedding = torch.cat(
            [glb_pos_feat_embedding, t_embedding.squeeze(1)], dim=-1
        )
        
        glb_pos_feat_t_embedding_expanded = glb_pos_feat_t_embedding.unsqueeze(1).repeat(1, nn_points, 1) ## bsz x nn_points x hidden_dim ##
        # print(f"glb_pos_feat_t_embedding_expanded: {glb_pos_feat_t_embedding_expanded.size()}, feat_flatten: {feat_flatten.size()}, X: {X.size()}")
        decoder_in_feats = torch.cat(
            [glb_pos_feat_t_embedding_expanded, pos_embedding], dim=-1 ## bsz x nn_points x (hidden_dim + hidden_dim) ##
        )

        
        x_out  = self.pos_decoder_mlps(decoder_in_feats) ## bsz x nn_points x 3 ##
        # feat_out = self.feat_decoder_mlps(decoder_in_feats) ## bsz x nn_points x feat_out_dim ##
        
        
        return utils.PlaceHolder(X=x_out, E=feat, y=y)


### mlp net for processing point cloud data ###
class MLP_Net_Segs(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information ## X feat X feat ##
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['segs'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['segs'] # 1024
        
        self.glb_feat_trans_dim = 2048 # concate the pos hidden and the feat hidden --- then the glb trans glock # 
        self.t_in_dim = 1
        self.t_hidden_dim = hidden_mlp_dims['t'] ## hidden dim for time sequence ## ## cocate wtih the glb feature ##
        
        self.decoder_in_dim = self.glb_feat_trans_dim + self.pos_in_dim + self.feat_in_dim + self.t_hidden_dim ## decoder input dim ## for each point
        
        self.pos_out_dim = output_dims['X']
        self.feat_out_dim = output_dims['segs']
        
        self.pos_embedding_mlps = nn.Sequential(
            * [nn.Linear(self.pos_in_dim, self.pos_hidden_dim)] + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim)]
        )
        self.feat_embedding_mlps = nn.Sequential(
            * [nn.Linear(self.feat_in_dim, self.feat_hidden_dim)] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim), act_fn_in] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)]
        )
        self.glb_feat_trans_mlps = nn.Sequential(
            * [nn.Linear(self.pos_hidden_dim + self.feat_hidden_dim, self.glb_feat_trans_dim), act_fn_in] + [nn.Linear(self.glb_feat_trans_dim, self.glb_feat_trans_dim)]
        ) ## a two-layer mlps for getting the trasformed glb features
        self.t_trans_mlps = nn.Sequential(
            * [nn.Linear(self.t_in_dim, self.t_hidden_dim), act_fn_in] + [nn.Linear(self.t_hidden_dim, self.t_hidden_dim)]
        ) ## two-layer for t transformation
        
        self.pos_decoder_mlps = nn.Sequential(
            * [nn.Linear(self.decoder_in_dim, self.pos_hidden_dim), act_fn_in] + [nn.Linear(self.pos_hidden_dim, self.pos_hidden_dim)] * (n_layers - 1) + [nn.Linear(self.pos_hidden_dim, self.pos_out_dim)]
        )
        self.feat_decoder_mlps = nn.Sequential(
            * [nn.Linear(self.decoder_in_dim, self.feat_hidden_dim), act_fn_in] + [nn.Linear(self.feat_hidden_dim, self.feat_hidden_dim)] * (n_layers - 1) + [nn.Linear(self.feat_hidden_dim, self.feat_out_dim)] # 
        )
        
        
        
    def forward(self, X, feat, y, node_mask):
        bsz = X.size(0)
        
        if y.dtype == torch.int32 or y.dtype == torch.long:
            y = y.float() / 1000.0

        # feat: bsz x nn_points  x (T x per_time_act_dim)
        
        nn_points = X.size(1)
        
        pos_embedding = self.pos_embedding_mlps(X) ## bsz x nn_points x pos_hidden_dim ##
        feat_flatten = feat.contiguous().view(bsz, nn_points, -1).contiguous() ## bsz x nn_points x (act_in_dim)
        feat_embedding = self.feat_embedding_mlps(feat_flatten) ## bsz x nn_points x fea_hidden_dim ##
        pos_feat_embedding = torch.cat([pos_embedding, feat_embedding], dim=-1) ## bsz x nn_points x (pos_hidden_dim + feat_hidden_dim) ## # 
        glb_pos_feat_embedding, _ = torch.max(pos_feat_embedding, dim=1) ## bsz x hidden dims ## 
        glb_pos_feat_embedding = self.glb_feat_trans_mlps(glb_pos_feat_embedding) ## bsz x hidden dim ##
        
        t_embedding = self.t_trans_mlps(y) ## bsz x t_embeddings ##
        glb_pos_feat_t_embedding = torch.cat(
            [glb_pos_feat_embedding, t_embedding], dim=-1
        )
        
        glb_pos_feat_t_embedding_expanded = glb_pos_feat_t_embedding.unsqueeze(1).repeat(1, nn_points, 1) ## bsz x nn_points x hidden_dim ##
        # print(f"glb_pos_feat_t_embedding_expanded: {glb_pos_feat_t_embedding_expanded.size()}, feat_flatten: {feat_flatten.size()}, X: {X.size()}")
        decoder_in_feats = torch.cat(
            [glb_pos_feat_t_embedding_expanded, feat_flatten, X], dim=-1 ## bsz x nn_points x (hidden_dim + hidden_dim) ##
        )

        
        x_out  = self.pos_decoder_mlps(decoder_in_feats) ## bsz x nn_points x 3 ##
        feat_out = self.feat_decoder_mlps(decoder_in_feats) ## bsz x nn_points x feat_out_dim ##
        
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)
           
         
        

## get the graph transformer ##
class GraphTransformer(nn.Module):
    """
    n_layers : int -- number of layers
    dims : dict -- contains dimensions for each feature type
    input_dims: X = xx, E = xx, y = xx ## does we need y here ? -- it can be set as a dummy vector ##
    hidden_mlp_dims: X = xx, E = xx, 
    """
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict, hidden_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        # 
        self.n_layers = n_layers
        self.out_dim_X = output_dims['X']
        self.out_dim_E = output_dims['E']
        self.out_dim_y = output_dims['y']

        self.mlp_in_X = nn.Sequential(nn.Linear(input_dims['X'], hidden_mlp_dims['X']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['X'], hidden_dims['dx']), act_fn_in)

        self.mlp_in_E = nn.Sequential(nn.Linear(input_dims['E'], hidden_mlp_dims['E']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['E'], hidden_dims['de']), act_fn_in)

        self.mlp_in_y = nn.Sequential(nn.Linear(input_dims['y'], hidden_mlp_dims['y']), act_fn_in,
                                      nn.Linear(hidden_mlp_dims['y'], hidden_dims['dy']), act_fn_in)

        self.tf_layers = nn.ModuleList([XEyTransformerLayer(dx=hidden_dims['dx'],
                                                            de=hidden_dims['de'],
                                                            dy=hidden_dims['dy'],
                                                            n_head=hidden_dims['n_head'],
                                                            dim_ffX=hidden_dims['dim_ffX'],
                                                            dim_ffE=hidden_dims['dim_ffE'])
                                        for i in range(n_layers)])

        self.mlp_out_X = nn.Sequential(nn.Linear(hidden_dims['dx'], hidden_mlp_dims['X']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['X'], output_dims['X']))

        self.mlp_out_E = nn.Sequential(nn.Linear(hidden_dims['de'], hidden_mlp_dims['E']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['E'], output_dims['E']))

        self.mlp_out_y = nn.Sequential(nn.Linear(hidden_dims['dy'], hidden_mlp_dims['y']), act_fn_out,
                                       nn.Linear(hidden_mlp_dims['y'], output_dims['y']))

    def forward(self, X, E, y, node_mask):
        bs, n = X.shape[0], X.shape[1]
        
        if y.dtype == torch.int32 or y.dtype == torch.long:
            y = y.float() / 1000.0

        # print(f"y: {y}")
        diag_mask = torch.eye(n)
        diag_mask = ~diag_mask.type_as(E).bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        X_to_out = X[..., :self.out_dim_X]
        E_to_out = E[..., :self.out_dim_E]
        y_to_out = y[..., :self.out_dim_y]
        
        # print(f"E: {E.size()}, weight_1: {self.mlp_in_E[0].weight.size()}, weight_2: {self.mlp_in_E[2].weight.size()}")
    
        new_E = self.mlp_in_E(E)
        new_E = (new_E + new_E.transpose(1, 2)) / 2
        # print("types:", X.dtype, new_E.dtype, y.dtype, node_mask.dtype)
        after_in = utils.PlaceHolder(X=self.mlp_in_X(X), E=new_E, y=self.mlp_in_y(y)).mask(node_mask)
        X, E, y = after_in.X, after_in.E, after_in.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask)

        X = self.mlp_out_X(X)
        E = self.mlp_out_E(E)
        y = self.mlp_out_y(y)

        X = (X + X_to_out)
        E = (E + E_to_out) * diag_mask
        y = y + y_to_out

        E = 1/2 * (E + torch.transpose(E, 1, 2))

        return utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask)




### mlp net for processing point cloud data ###
class Transformer_Net_PC_Seq(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        # trnsformer pc seq #
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        self.per_point_input_dim = 12 # 
        self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False)
        
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer ### # InputProcessObjBaseV2
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    
        
        
        
    def forward(self, X, feat, y, node_mask):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # bsz x nn
        # X : bsz x 

        # X : bsz x nn_points x (T x per_time_act_dim)
        tot_n_feats = feat.size(-1)
        np = feat.size(1)
        nt = tot_n_feats // self.per_point_input_dim_acc
        per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
        per_point_feat = torch.cat(
            [
                X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
            ], dim=-1
        )
        
        per_point_embedding = self.input_process(per_point_feat)
        per_point_embedding = self.positional_encoder(per_point_embedding)
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
        time_embedding = self.time_embedder(y_expanded)
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
        feat_out = per_point_output[:, :, :, 3:]
        feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        x_out = per_point_output[:, :, 0, :3]
        
        
        
        # # feat: bsz x nn_points  x (T x per_time_act_dim)
        
        # nn_points = X.size(1)
        
        # pos_embedding = self.pos_embedding_mlps(X) ## bsz x nn_points x pos_hidden_dim ##
        # feat_flatten = feat.contiguous().view(bsz, nn_points, -1).contiguous() ## bsz x nn_points x (act_in_dim)
        # feat_embedding = self.feat_embedding_mlps(feat_flatten) ## bsz x nn_points x fea_hidden_dim ##
        # pos_feat_embedding = torch.cat([pos_embedding, feat_embedding], dim=-1) ## bsz x nn_points x (pos_hidden_dim + feat_hidden_dim) ## # 
        # glb_pos_feat_embedding, _ = torch.max(pos_feat_embedding, dim=1) ## bsz x hidden dims ## 
        # glb_pos_feat_embedding = self.glb_feat_trans_mlps(glb_pos_feat_embedding) ## bsz x hidden dim ##
        
        # t_embedding = self.t_trans_mlps(y) ## bsz x t_embeddings ##
        # glb_pos_feat_t_embedding = torch.cat(
        #     [glb_pos_feat_embedding, t_embedding], dim=-1
        # )
        
        # glb_pos_feat_t_embedding_expanded = glb_pos_feat_t_embedding.unsqueeze(1).repeat(1, nn_points, 1) ## bsz x nn_points x hidden_dim ##
        # # print(f"glb_pos_feat_t_embedding_expanded: {glb_pos_feat_t_embedding_expanded.size()}, feat_flatten: {feat_flatten.size()}, X: {X.size()}")
        # decoder_in_feats = torch.cat(
        #     [glb_pos_feat_t_embedding_expanded, feat_flatten, X], dim=-1 ## bsz x nn_points x (hidden_dim + hidden_dim) ##
        # )

        
        # x_out  = self.pos_decoder_mlps(decoder_in_feats) ## bsz x nn_points x 3 ##
        # feat_out = self.feat_decoder_mlps(decoder_in_feats) ## bsz x nn_points x feat_out_dim ##
        
    
        
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)





### mlp net for processing point cloud data ###
class Transformer_Net_PC_Seq_V2(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        # trnsformer pc seq #
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        self.per_point_input_dim = 6 # 
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers # 
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer #### InputProcessObjBaseV2
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    
        
        
        
    def forward(self, X, feat, y, node_mask):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # bsz x nn
        # X : bsz x 
        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        bsz, np, nt, _ = feat.size()
        per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        per_point_feat = torch.cat(
            [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        )
        

        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
        
        
        
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        
        per_point_embedding = self.input_process(per_point_feat)
        per_point_embedding = self.positional_encoder(per_point_embedding)
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
        time_embedding = self.time_embedder(y_expanded)
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
        # input dim and the output feat dim = 3 #
        x_out = per_point_output[:, :, :, :3] # get the x_out forjthe x # bsz x np x nt x 3 
        feat_out = per_point_output[:, :, :, 3:] # get the feat out for the feat # bsz x np x nt x feat_dim #
        
        
        # feat_out = per_point_output[:, :, :, 3:]
        # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # x_out = per_point_output[:, :, 0, :3]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)



### mlp net for processing point cloud data ###
class Transformer_Net_PC_Seq_V3(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        # trnsformer pc seq #
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        # self.per_point_input_dim = 9 # acc # # 
        self.per_point_input_dim = self.pos_in_dim + self.feat_in_dim 
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers # 
        # input process obj base -> what's that ->  a point with features encoding network #
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
        ### # a token embedding ##
        ### # token embeddings ## # 
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer ####
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    
        
        
        
    def forward(self, X, feat, y, node_mask=None, cond=None):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # bsz x nn
        # X : bsz x 
        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        bsz, np, nt, _ = feat.size()
        per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        

        
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        per_point_feat = torch.cat(
            [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        )
        

        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
        
        
        
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        
        per_point_embedding = self.input_process(per_point_feat)
        per_point_embedding = self.positional_encoder(per_point_embedding)
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
        time_embedding = self.time_embedder(y_expanded)
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
        # input dim and the output feat dim = 3 #
        x_out = per_point_output[:, :, :, :self.pos_in_dim] # get the x_out forjthe x # bsz x np x nt x 3 
        feat_out = per_point_output[:, :, :, self.pos_in_dim:] # get the feat out for the feat # bsz x np x nt x feat_dim #
        # task translations as the optimizations # #
        # task translations as the optimizations # #
        
        
        # feat_out = per_point_output[:, :, :, 3:]
        # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # x_out = per_point_output[:, :, 0, :3]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)





class Transformer_Net_PC_Seq_V3_AE(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, task_cond_type='future', sub_task_cond_type='full', debug=False, glb_rot_use_quat=False ):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat']
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        self.traj_cond = traj_cond
        
        self.task_cond_type = task_cond_type
        
        # self.per_point_input_dim = 9 # acc # # 
        self.per_point_input_dim = self.pos_in_dim + self.feat_in_dim 
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers # 
        
        self.debug = debug
        self.sub_task_cond_type = sub_task_cond_type
        self.glb_rot_use_quat = glb_rot_use_quat
        
        
        ### per point --- input process for the per point features ###
        ###### input process ######
        # input process obj base -> what's that ->  a point with features encoding network #
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
        
        if self.traj_cond:
            # input process obj base pc # history conditioning -> the object shape; object starting pose #
            # model_1 -> input process obj base PC; 
            # model_2 -> object pose...
            # 
            
            if self.sub_task_cond_type == 'obj_shape_pose':
                if self.debug:
                    print(f"Constructing networks for sub_task_cond_type: {self.sub_task_cond_type}")
                # bsz x nn_latent_dim #
                self.cond_input_process_pc = InputProcessObjBasePC( 3, self.latent_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True) 
                self.cond_input_process_feat = InputProcessObjBaseV5( 3 + 3, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                
            elif self.sub_task_cond_type == 'obj_shape_hand_pose':
                if self.debug:
                    print(f"Constructing networks for sub_task_cond_type: {self.sub_task_cond_type}")
                self.cond_input_process_pc = InputProcessObjBasePC( 3, self.latent_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.cond_input_process_feat = InputProcessObjBaseV5( 3 + 3 + self.feat_in_dim, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                
            elif self.sub_task_cond_type == 'obj_shape':
                if self.debug:
                    print(f"Constructing networks for sub_task_cond_type: {self.sub_task_cond_type}")
                self.cond_input_process_pc = InputProcessObjBasePC( 3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True) 
            
            elif self.sub_task_cond_type == 'hand_pose_traj':
                self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim , self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                # positional encoding for the time sequence #
                # self.cond_positonal_encoder #
                ### Encoding layer ####
                cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.cond_transformer_encoder = nn.TransformerEncoder(cond_transformer_encoder_layer, # basejtsrel_seqTrans
                                                            num_layers=self.num_layers)
                
                self.future_cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.future_cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                future_cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.future_cond_transformer_encoder = nn.TransformerEncoder(future_cond_transformer_encoder_layer, # basejtsrel_seqTrans
                                                            num_layers=self.num_layers)
            
            elif self.sub_task_cond_type == 'hand_pose_traj_wpc':
                self.cond_input_process_pc = InputProcessObjBasePC( 3, self.latent_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True) 
                self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim , self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                # positional encoding for the time sequence #
                # self.cond_positonal_encoder #
                ### Encoding layer ####
                cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.cond_transformer_encoder = nn.TransformerEncoder(cond_transformer_encoder_layer, # basejtsrel_seqTrans
                                                            num_layers=self.num_layers)
                
                
                self.future_cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.future_cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                future_cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.future_cond_transformer_encoder = nn.TransformerEncoder(future_cond_transformer_encoder_layer, # basejtsrel_seqTrans
                                                            num_layers=self.num_layers)
            
            elif self.sub_task_cond_type == 'full_wohistory':
                # future trajectories with the desired hand joint positions and the object poses #
                # hand_glb_cond
                self.future_cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim + 3 + 3, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                # object point cloud shape processing #
                self.cond_input_process_pc = InputProcessObjBasePC( 3, self.latent_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.future_cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                future_cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                self.future_cond_transformer_encoder = nn.TransformerEncoder(future_cond_transformer_encoder_layer,
                                                            num_layers=self.num_layers)
            
            elif self.sub_task_cond_type in ['full', 'full_woornt']:
                self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True) 
                self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim + 3 + 3, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                self.cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                # positional encoding for the time sequence #
                # self.cond_positonal_encoder #
                ### Encoding layer ####
                cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)

                self.cond_transformer_encoder = nn.TransformerEncoder(cond_transformer_encoder_layer, # basejtsrel_seqTrans
                                                            num_layers=self.num_layers)
                
                # history_cond_input_process_feat # 
                if self.task_cond_type == 'history_future':
                    self.history_cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim + 3 + 3, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
                    self.history_cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
                    history_cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
                    self.history_cond_transformer_encoder = nn.TransformerEncoder(history_cond_transformer_encoder_layer,
                                                            num_layers=self.num_layers)
            else:
                raise ValueError(f"Invalid sub_task_cond_type: {self.sub_task_cond_type}")
            
                
            pass
        
        ### # a token embedding ##
        ### # token embeddings ## # 
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer ####
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    
        
    def encode(self, X, feat):
        # encode the feature #
        if self.debug:
            print(f"glb_rot_use_quat: {self.glb_rot_use_quat}, pts_feat: {X.size()}, feat_feat: {feat.size()}")
        
        per_point_feat = torch.cat(
            [X, feat], dim=-1 ### n_bsz x np x nt x 6 ## 
        )
        x_pts_feat, x_glb_feat = self.input_process(per_point_feat, rt_glb=True, permute_dim=False )
        # x_pts_feat: bsz x ws x nn_latent_dim #
        # x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        # encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim 
        # 
        tot_encoded_feats = {
            'pts_feat': x_pts_feat,
            'feat_feat': x_pts_feat # get all of the encoded featres #
        }
        return tot_encoded_feats
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        # pts feat #
        pts_feat = pts_feat.contiguous().permute(1, 0, 2).contiguous() # nn_ts x nn_bsz x latnet_dim
        
        per_point_output = self.output_process(pts_feat, None) # nb x np x nt x latent_dim 
        
        # input dim and the output feat dim = 3 #
        x_out = per_point_output[:, :, :, :self.pos_in_dim] # get the x_out forjthe x # bsz x np x nt x 3 
        feat_out = per_point_output[:, :, :, self.pos_in_dim:] 
        
        # utils.PlaceHolder(X=x_out, E=feat_out, y=y)
        tot_decoded_feats = {
            'X': x_out,
            'feat': feat_out,
        }
        return tot_decoded_feats
        
        # decoded_pts = self.output_process(pts_feat)
        # decoded_feat = self.output_process_feat(encoded_feat) #
        # tot_decoded_feats = {
        #     'X': decoded_pts,
        #     'feat': decoded_feat
        # }
        # return tot_decoded_feats
        
        
    def forward(self, pts_feat, feat_feat, y, node_mask=None, cond=None):
        # bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        # bsz, np, nt, _ = feat.size()
        # per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        

        
        # per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        # per_point_feat = torch.cat(
        #     [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        # )
        

        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        nn_ts = pts_feat.size(1)
        
        if self.debug:
            print(f"sub_task_cond_type: {self.sub_task_cond_type}, pts_feat: {pts_feat.size()}, feat_feat: {feat_feat.size()}, y: {y.size()}")
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        
        # per_point_embedding = self.input_process(per_point_feat)
        np = 1
        per_point_embedding = pts_feat.contiguous().permute(1, 0, 2).contiguous() # nn_ts x nn_bsz x latnet_dim # 
        
        if self.traj_cond:
            cond_X = cond['X']
            cond_E = cond['E'] #get the pts and features # 
            
            if self.sub_task_cond_type == 'obj_shape_pose':
                cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
                cond_glb_feat = cond_glb_feat[:, 0, :] # ---> get all the glb feat ## 
                cond_prev_obj_pose = cond['history_E'][..., -1: , -6: ] # nn_bsz x nn_points x 1 x 3 
                cond_history_obj_pose_feat = self.cond_input_process_feat(cond_prev_obj_pose) ## nn_ts x nn_bsz x nn_latent_dim
                # 
                expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_history_obj_pose_feat.size(0), 1, 1) # 
                cat_cond_feat = torch.cat(
                    [ expanded_cond_pts_feat, cond_history_obj_pose_feat ], dim=-1 # conditional input features
                )
                ## nn_ts x nn-bsz x nn_latnet_dim
                cat_cond_feat = cat_cond_feat.contiguous().repeat(per_point_embedding.size(0), 1, 1).contiguous() 
                per_point_embedding = per_point_embedding + cat_cond_feat
                
            elif self.sub_task_cond_type == 'obj_shape_hand_pose':
                cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
                cond_glb_feat = cond_glb_feat[:, 0, :]
                cond_prev_obj_pose = cond['history_E'][..., -1: , : ] # nn_bsz x nn_points x 1 x 3
                cond_history_obj_pose_feat = self.cond_input_process_feat(cond_prev_obj_pose) ## nn_ts x nn_bsz x nn_latent_dim
                expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_history_obj_pose_feat.size(0), 1, 1) # 
                cat_cond_feat = torch.cat(
                    [ expanded_cond_pts_feat, cond_history_obj_pose_feat ], dim=-1 # conditional input features
                )
                ## nn_ts x nn-bsz x nn_latnet_dim
                cat_cond_feat = cat_cond_feat.contiguous().repeat(per_point_embedding.size(0), 1, 1).contiguous() 
                per_point_embedding = per_point_embedding + cat_cond_feat
            
            elif self.sub_task_cond_type == 'hand_pose_traj':
                hand_pose_traj = cond['history_E'][..., :, : -6]
                cond_history_feat = self.cond_input_process_feat(hand_pose_traj)
                cond_history_feat = self.cond_positional_encoder(cond_history_feat)
                cond_history_feat = self.cond_transformer_encoder(cond_history_feat) # nn_ts x nn_bsz x nn_latent_dim
                cond_history_feat = cond_history_feat[-1:, :, :].contiguous().repeat(per_point_embedding.size(0), 1, 1).contiguous()
                per_point_embedding = per_point_embedding + cond_history_feat
                
                future_feat = self.future_cond_input_process_feat(cond['E'][..., :-6])
                cat_future_cond_feat = future_feat
                cat_future_cond_feat = self.future_cond_positional_encoder(cat_future_cond_feat)
                cat_future_cond_feat = self.future_cond_transformer_encoder(cat_future_cond_feat)
                per_point_embedding = per_point_embedding + cat_future_cond_feat
                
            elif self.sub_task_cond_type == 'hand_pose_traj_wpc':
                hand_pose_traj = cond['history_E'][..., :, : -6]
                cond_history_feat = self.cond_input_process_feat(hand_pose_traj)
                cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
                cond_glb_feat = cond_glb_feat[:, 0, :] # cond input process pc #
                expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_history_feat.size(0), 1, 1)
                cat_cond_feat = torch.cat( [ expanded_cond_pts_feat, cond_history_feat ], dim=-1 ) # nn_ts x nn_bsz x nn_latent_dim  ## nn_ts ## 
                cat_cond_feat = self.cond_positional_encoder(cat_cond_feat)
                cat_cond_feat = self.cond_transformer_encoder(cat_cond_feat)
                cat_cond_feat = cat_cond_feat[-1:, :, :].contiguous().repeat(per_point_embedding.size(0), 1, 1).contiguous()
                per_point_embedding = per_point_embedding + cat_cond_feat
                
                future_feat = self.future_cond_input_process_feat(cond['E'][..., :-6])
                expanded_future_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(future_feat.size(0), 1, 1)
                cat_future_cond_feat = torch.cat( [ expanded_future_cond_pts_feat, future_feat ], dim=-1 )
                cat_future_cond_feat = self.future_cond_positional_encoder(cat_future_cond_feat)
                cat_future_cond_feat = self.future_cond_transformer_encoder(cat_future_cond_feat)
                per_point_embedding = per_point_embedding + cat_future_cond_feat
                
            
            elif self.sub_task_cond_type == 'obj_shape':
                cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
                cond_glb_feat = cond_glb_feat[:, 0, :] # cond input process pc #
                expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(per_point_embedding.size(0), 1, 1)
                cat_cond_feat = expanded_cond_pts_feat
                # cat_cond_feat = cat_cond_feat.contiguous().repeat(per_point_embedding.size(0), 1, 1).contiguous()
                per_point_embedding = per_point_embedding + cat_cond_feat
                # obj shape #
            
            elif self.sub_task_cond_type == 'full_wohistory':
                cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
                cond_glb_feat = cond_glb_feat[:, 0, :]
                cond_encoded_feat = self.future_cond_input_process_feat(cond_E)
                expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1) # expanded cond pts features #
                cat_cond_feat = torch.cat(
                    [ expanded_cond_pts_feat, cond_encoded_feat ], dim=-1
                )
                cat_cond_feat = self.future_cond_positional_encoder(cat_cond_feat)
                cat_cond_feat = self.future_cond_transformer_encoder(cat_cond_feat)
                per_point_embedding = per_point_embedding + cat_cond_feat 
            
            elif self.sub_task_cond_type in [ 'full', 'full_woornt']:
                
                cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
                cond_glb_feat = cond_glb_feat[:, 0, :] # cond input process pc #
                
                if self.sub_task_cond_type == 'full_woornt':
                    cond_E = torch.cat(
                        [ cond_E[..., :-3], torch.zeros_like(cond_E[..., -3:]) ], dim=-1
                    )
                
                cond_encoded_feat = self.cond_input_process_feat(cond_E)
                # print(f"cond_encoded_feat: {cond_encoded_feat.size()}, cond_glb_feat: {cond_glb_feat.size()}")
                expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
                cat_cond_feat_obj_embedding = torch.cat(
                    [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
                )
                cat_cond_feat_obj_embedding = self.cond_positional_encoder(cat_cond_feat_obj_embedding)
                cat_cond_feat_obj_embedding = self.cond_transformer_encoder(cat_cond_feat_obj_embedding)
                # cat_cond_feat_obj_embedding = cat_cond_feat_obj_embedding[0:1].repeat(nn_ts, 1, 1)
                if per_point_embedding.size(0) == cat_cond_feat_obj_embedding.size(0):
                    # print(f"per_point_embedding: {per_point_embedding.size()}, cat_cond_feat_obj_embedding: {cat_cond_feat_obj_embedding.size()}")
                    per_point_embedding = per_point_embedding + cat_cond_feat_obj_embedding
                else:
                    nn_control_freq = 10
                    feat_ts_idxes = torch.arange(0, nn_ts, dtype=torch.long, device=cat_cond_feat_obj_embedding.device)
                    feat_ts_idxes = feat_ts_idxes * nn_control_freq
                    feat_ts_idxes = torch.clamp(feat_ts_idxes, min=0, max=nn_ts - 1) # get the feat ts idxes #
                    cat_cond_feat_obj_embedding = cat_cond_feat_obj_embedding[feat_ts_idxes]
                    per_point_embedding = per_point_embedding + cat_cond_feat_obj_embedding
                
                if self.task_cond_type == 'history_future':
                    if self.debug: # history E #
                        print(f"Debug mode with task_cond_type {self.task_cond_type} with cond: {cond.keys()}")
                    cond_history_E = cond['history_E']
                    
                    if self.sub_task_cond_type == 'full_woornt':
                        cond_history_E = torch.cat( # cond_history_E #
                            [ cond_history_E[..., :-3], torch.zeros_like(cond_history_E[..., -3:]) ], dim=-1
                        )
                    
                    history_encoded_feat = self.history_cond_input_process_feat(cond_history_E)
                    # bsz x # 
                    expanded_cond_history_pts_feat = cond_glb_feat.unsqueeze(0).repeat(history_encoded_feat.size(0), 1, 1)
                    
                    cat_history_cond_feat_obj_embedding = torch.cat(
                        [expanded_cond_history_pts_feat, history_encoded_feat],  dim=-1
                    )
                    
                    cat_history_cond_feat_obj_embedding = self.history_cond_positional_encoder(cat_history_cond_feat_obj_embedding)
                    cat_history_cond_feat_obj_embedding = self.history_cond_transformer_encoder(cat_history_cond_feat_obj_embedding)
                    # nn_ts x 1 x 1 # 
                    added_to_emebedding_history_cond_feats = cat_history_cond_feat_obj_embedding[0: 1].repeat(per_point_embedding.size(0), 1, 1).contiguous()
                    per_point_embedding = per_point_embedding + added_to_emebedding_history_cond_feats # add the history cond feats to the per-point-embeddings #
            else:
                raise ValueError(f"Invalid sub_task_cond_type: {self.sub_task_cond_type}")
            
        
        
        per_point_embedding = self.positional_encoder(per_point_embedding)
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
        time_embedding = self.time_embedder(y_expanded)
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        # per point output #
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = per_point_output.contiguous().permute(1, 0, 2).contiguous() 
        x_out = per_point_output
        feat_out = per_point_output
        
        # # per_point_output = self.output_process(per_point_output, per_point_feat)
        # per_point_output = self.output_process(per_point_output, None)
        
        # # input dim and the output feat dim = 3 #
        # x_out = per_point_output[:, :, :, :self.pos_in_dim] # get the x_out forjthe x # bsz x np x nt x 3 
        # feat_out = per_point_output[:, :, :, self.pos_in_dim:] # get the feat out for the feat # bsz x np x nt x feat_dim #
        # # task translations as the optimizations # # per-point output #
        
        
        # # feat_out = per_point_output[:, :, :, 3:]
        # # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # # x_out = per_point_output[:, :, 0, :3]
        # x_out = x_out[:, 0]
        # feat_out = feat_out[:, 0]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)




class Transformer_Net_PC_Seq_V4(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, task_cond_type='future', debug=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat']  
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        self.traj_cond = traj_cond
        
        self.task_cond_type = task_cond_type
        
        # self.per_point_input_dim = 9
        self.per_point_input_dim = self.pos_in_dim + self.feat_in_dim 
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        
        self.debug = debug
        
        
        # cond input process of pc #
        # feat in dim --> with the object trans and the object rot euler # 
        # feat hidden dim # feat hidden dim # input process obj base 
        
        
        #### Input process layer for encoding the target trajectory ####
        # self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        #### Input process layer for encoding the target trajectory ####
        
        #### find the PC and BaseV5 ####
        
        
        self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True) 
        # what's the input processing module here ? #
        self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim + 3 + 3, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
        self.cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout) # positional encoding for the time sequence #
        # positional encoding for the time sequence #
        # self.cond_positonal_encoder #
        # traj cond # and the process pc, feat #
        # traj cond # traj cond # process feat #
        ### Encoding layer ####
        cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.cond_transformer_encoder = nn.TransformerEncoder(cond_transformer_encoder_layer, # cond transformer 
                                                    num_layers=self.num_layers)
        # input process obj base #
        
        # history_cond_input_process_feat, 
        # if self.task_cond_type == 'history_future': # history input process base #
        self.history_cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim + 3 + 3, self.feat_hidden_dim // 2, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
        self.history_cond_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        history_cond_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                    nhead=self.num_heads,
                                                    dim_feedforward=self.ff_size,
                                                    dropout=self.dropout,
                                                    activation=self.activation)
        self.history_cond_transformer_encoder = nn.TransformerEncoder(history_cond_transformer_encoder_layer,
                                                num_layers=self.num_layers)
        
        ### latent_dim x latent_dim ###
        ### bsz x latnet_dim ### 
        ### -> decode the encoded features to the joint actions ###
        ### like a policy network which has one single output frame ###
        ### sigmoid for the output ? ###
        ### history cond transformer encoder ###
        
        self.output_process = OutputProcessObjBaseRawV8(self.latent_dim + self.latent_dim, self.feat_in_dim)
        
        
        ######## positional encoder ########
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        # transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                 nhead=self.num_heads,
        #                                                 dim_feedforward=self.ff_size,
        #                                                 dropout=self.dropout,
        #                                                 activation=self.activation)

        # self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
        #                                             num_layers=self.num_layers)
        
        # self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        # transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                 nhead=self.num_heads,
        #                                                 dim_feedforward=self.ff_size,
        #                                                 dropout=self.dropout,
        #                                                 activation=self.activation)

        # self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
        #                                             num_layers=self.num_layers)
        
        # self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
        ######## positional encoder ########
    
        
    # def encode(self, X, feat):
    #     # encode the feature #
    #     per_point_feat = torch.cat(
    #         [X, feat], dim=-1 ### n_bsz x np x nt x 6 ## 
    #     )
    #     x_pts_feat, x_glb_feat = self.input_process(per_point_feat, rt_glb=True, permute_dim=False )
    #     # x_pts_feat: bsz x ws x nn_latent_dim #
    #     # x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
    #     # encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim 
    #     # 
    #     tot_encoded_feats = {
    #         'pts_feat': x_pts_feat,
    #         'feat_feat': x_pts_feat # get all of the encoded featres #
    #     }
    #     return tot_encoded_feats
    
    # def decode(self, tot_latent_feats): 
    #     pts_feat = tot_latent_feats['pts_feat']
    #     encoded_feat = tot_latent_feats['feat_feat']
    #     # pts feat #
    #     pts_feat = pts_feat.contiguous().permute(1, 0, 2).contiguous() # nn_ts x nn_bsz x latnet_dim
        
    #     per_point_output = self.output_process(pts_feat, None) # nb x np x nt x latent_dim 
        
    #     # input dim and the output feat dim = 3 #
    #     x_out = per_point_output[:, :, :, :self.pos_in_dim] # get the x_out forjthe x # bsz x np x nt x 3 
    #     feat_out = per_point_output[:, :, :, self.pos_in_dim:] 
        
    #     # utils.PlaceHolder(X=x_out, E=feat_out, y=y)
    #     tot_decoded_feats = {
    #         'X': x_out,
    #         'feat': feat_out,
    #     }
    #     return tot_decoded_feats
        
    #     # decoded_pts = self.output_process(pts_feat)
    #     # decoded_feat = self.output_process_feat(encoded_feat) #
    #     # tot_decoded_feats = {
    #     #     'X': decoded_pts,
    #     #     'feat': decoded_feat
    #     # }
    #     # return tot_decoded_feats
        
    
    ####### forward pass #######
    def forward(self,  cond=None):
        # bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        # bsz, np, nt, _ = feat.size()
        # per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        

        
        # per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        # per_point_feat = torch.cat(
        #     [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        # )
        

        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        # nn_ts = pts_feat.size(1)
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        
        # per_point_embedding = self.input_process(per_point_feat)
        # np = 1
        
        
        cond_X = cond['X']
        cond_E = cond['E']
        
        # nn_ts = cond_E.size(1) ## nn_bsz x nn_ts x (nn_hand_dofs + 3 + 3)
        
        
        cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
        cond_glb_feat = cond_glb_feat[:, 0, :]
        cond_encoded_feat = self.cond_input_process_feat(cond_E)
        
        # cond encoded feat --- nn_ts x nn_bsz x nn_latent_dim #
        
        # print(f"cond_encoded_feat: {cond_encoded_feat.size()}, cond_glb_feat: {cond_glb_feat.size()}")
        #### conditional global fatures ####
        expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
        cat_cond_feat_obj_embedding = torch.cat(
            [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
        )
        cat_cond_feat_obj_embedding = self.cond_positional_encoder(cat_cond_feat_obj_embedding) ## positional encoder ##
        cat_cond_feat_obj_embedding = self.cond_transformer_encoder(cat_cond_feat_obj_embedding) ## cond transformer encoder; cond transformer encoders ##
        # cat cond feat obj embedding # 
        
        ## nn_bsz x nn_latent_dim ##
        cond_feat_embedding = cat_cond_feat_obj_embedding[0]
        
        cond_history_E = cond['history_E']
        
        ### history encoded feat: nn_history_ts x nn_latent_dim ###
        history_encoded_feat = self.history_cond_input_process_feat(cond_history_E)
        
        ## history_encoded_feat: nn_ts x nn_bsz x nn_latent-dim ## history cond input process feat 3
        expanded_cond_history_pts_feat = cond_glb_feat.unsqueeze(0).repeat(history_encoded_feat.size(0), 1, 1)
        
        cat_history_cond_feat_obj_embedding = torch.cat(
            [expanded_cond_history_pts_feat, history_encoded_feat],  dim=-1
        )
        
        cat_history_cond_feat_obj_embedding = self.history_cond_positional_encoder(cat_history_cond_feat_obj_embedding)
        cat_history_cond_feat_obj_embedding = self.history_cond_transformer_encoder(cat_history_cond_feat_obj_embedding)
        
        
        
        history_feat = cat_history_cond_feat_obj_embedding[0] ## nn_bsz x nn_latent_dim ##
        
        # [ cond feat embedding, history feat ] #
        embedding_features = torch.cat(
            [ cond_feat_embedding, history_feat ], dim=-1
        )
        
        decoded_feats = self.output_process(embedding_features)
        
        return decoded_feats
        
        
        # # nn_ts x 1 x 1 # 
        # added_to_emebedding_history_cond_feats = cat_history_cond_feat_obj_embedding[0: 1].repeat(per_point_embedding.size(0), 1, 1).contiguous()
        # per_point_embedding = per_point_embedding + added_to_emebedding_history_cond_feats # add the history cond feats to the per-point-embeddings #
            
            
        
        
        
        # # # cat_cond_feat_obj_embedding = cat_cond_feat_obj_embedding[0:1].repeat(nn_ts, 1, 1)
        # # if per_point_embedding.size(0) == cat_cond_feat_obj_embedding.size(0):
        # #     # print(f"per_point_embedding: {per_point_embedding.size()}, cat_cond_feat_obj_embedding: {cat_cond_feat_obj_embedding.size()}")
        # #     per_point_embedding = per_point_embedding + cat_cond_feat_obj_embedding
        # # else:
        # #     nn_control_freq = 10
        # #     feat_ts_idxes = torch.arange(0, nn_ts, dtype=torch.long, device=cat_cond_feat_obj_embedding.device)
        # #     feat_ts_idxes = feat_ts_idxes * nn_control_freq
        # #     feat_ts_idxes = torch.clamp(feat_ts_idxes, min=0, max=nn_ts - 1) # get the feat ts idxes #
        # #     cat_cond_feat_obj_embedding = cat_cond_feat_obj_embedding[feat_ts_idxes]
        # #     per_point_embedding = per_point_embedding + cat_cond_feat_obj_embedding
        
        # if self.task_cond_type == 'history_future':
        #     if self.debug: # history E #
        #         print(f"Debug mode with task_cond_type {self.task_cond_type} with cond: {cond.keys()}")
        #     cond_history_E = cond['history_E']
        #     history_encoded_feat = self.history_cond_input_process_feat(cond_history_E)
        #     # bsz x # 
        #     expanded_cond_history_pts_feat = cond_glb_feat.unsqueeze(0).repeat(history_encoded_feat.size(0), 1, 1)
            
        #     cat_history_cond_feat_obj_embedding = torch.cat(
        #         [expanded_cond_history_pts_feat, history_encoded_feat],  dim=-1
        #     )
            
        #     cat_history_cond_feat_obj_embedding = self.history_cond_positional_encoder(cat_history_cond_feat_obj_embedding)
        #     cat_history_cond_feat_obj_embedding = self.history_cond_transformer_encoder(cat_history_cond_feat_obj_embedding)
        #     # nn_ts x 1 x 1 # 
        #     added_to_emebedding_history_cond_feats = cat_history_cond_feat_obj_embedding[0: 1].repeat(per_point_embedding.size(0), 1, 1).contiguous()
        #     per_point_embedding = per_point_embedding + added_to_emebedding_history_cond_feats # add the history cond feats to the per-point-embeddings #
                
            
        
        
        # per_point_embedding = self.positional_encoder(per_point_embedding)
        # per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        # y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
        # time_embedding = self.time_embedder(y_expanded)
        # per_point_embedding_with_timesteps = torch.cat(
        #     [time_embedding, per_point_embedding], dim=0
        # )
        
        # # per point output #
        # per_point_output = self.transformer_with_timesteps_encoder(
        #     per_point_embedding_with_timesteps
        # )[1:]
        
        # per_point_output = per_point_output.contiguous().permute(1, 0, 2).contiguous() 
        # x_out = per_point_output
        # feat_out = per_point_output
        
        # # per_point_output = self.output_process(per_point_output, per_point_feat)
        # per_point_output = self.output_process(per_point_output, None)
        
        # # input dim and the output feat dim = 3 #
        # x_out = per_point_output[:, :, :, :self.pos_in_dim] # get the x_out forjthe x # bsz x np x nt x 3 
        # feat_out = per_point_output[:, :, :, self.pos_in_dim:] # get the feat out for the feat # bsz x np x nt x feat_dim #
        # # task translations as the optimizations # # per-point output #
        
        
        # # feat_out = per_point_output[:, :, :, 3:]
        # # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # # x_out = per_point_output[:, :, 0, :3]
        # x_out = x_out[:, 0]
        # feat_out = feat_out[:, 0]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)



class Transformer_Net_PC_Seq_V3_wcond(nn.Module): 
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        # self.per_point_input_dim = 9 # acc # # 
        self.per_point_input_dim = self.pos_in_dim + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers # 
        
        
        self.task_configs_dim = 6
        self.cond_latent_dim = self.latent_dim
        self.cond_processing = InputProcessObjBaseCondsV5(self.task_configs_dim, self.cond_latent_dim)
        
        # input process obj base -> what's that ->  a point with features encoding network #
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
        ### # a token embedding ##
        ### # token embeddings ## # 
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer #### InputProcessObjBaseV2
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    

        
        ### ### 
        
        # # input conditions? # #
        # # input conditions? # #
        
    def forward(self, X, feat, y, cond):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # cond : bsz x nn_feat_dim
        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        bsz, np, nt, _ = feat.size()
        per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        per_point_feat = torch.cat(
            [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        )
        
        cond_encoding = self.cond_processing(cond)
        
        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}, cond: {cond.size()}")
        
        per_point_embedding = self.input_process(per_point_feat)
        per_point_embedding = self.positional_encoder(per_point_embedding)
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        # bsz x np x nt x cond_encoding_dim #
        
        cond_encoding = cond_encoding.unsqueeze(1).unsqueeze(1).repeat(1, np, nt, 1)
        cond_encoding = cond_encoding.contiguous().permute(2, 0, 1, 3).contiguous()
        cond_encoding = cond_encoding.contiguous().view(nt, bsz * np, -1).contiguous()
        
        # print(f"per_point_embedding: {per_point_embedding.size()}, cond_encoding: {cond_encoding.size()}    ")
        per_point_embedding = per_point_embedding + cond_encoding ## cond embedding ##
        
        y_expanded = y.squeeze(-1).repeat(np)
        time_embedding = self.time_embedder(y_expanded)
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
        # input dim and the output feat dim = 3 #
        x_out = per_point_output[:, :, :, : self.pos_in_dim] # bsz x np x nt x 3 
        feat_out = per_point_output[:, :, :, self.pos_in_dim :]
        
        
        # feat_out = per_point_output[:, :, :, 3:]
        # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # x_out = per_point_output[:, :, 0, :3]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)

# 
class Transformer_Net_PC_Seq_V3_wtaskcond_V2(nn.Module): 
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), cond_task_type='rotation'):
        super().__init__()
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024
        
        # self.per_point_input_dim = 9 # acc # # acc # # act adn the act #
        self.per_point_input_dim = self.pos_in_dim + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers # 
        
        self.cond_task_type = cond_task_type
        
        ## obj task cond or othe conditions # for each ws --- the input to the transformer model? #
        ## task transformer ## --- simple pooling along the task dimension? # 
        ## 
        # 
        # self.task_configs_dim = 6
        if self.cond_task_type == 'rotation':
            self.task_configs_dim = 3
            self.cond_latent_dim = self.latent_dim
            self.cond_processing = InputProcessObjBaseCondsV6(self.task_configs_dim, self.cond_latent_dim)
        elif self.cond_task_type == 'tracking':
            ## construct the tracking task codnitions ##
            self.hand_qs_input_dim = 22
            self.obj_trans_input_dim = 3
            self.obj_ornt_input_dim = 4
            self.cond_processing = InputProcessObjBaseCondsV7(self.hand_qs_input_dim, self.obj_trans_input_dim, self.obj_ornt_input_dim, latent_dim=self.latent_dim) # get the cond process layer 
            self.cond_positional_encoding = PositionalEncoding(self.latent_dim, self.dropout) # cond 
            transformer_encoder_layer_cond = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
            self.transformer_encoder_cond = nn.TransformerEncoder(transformer_encoder_layer_cond, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers) # ge tthe tranfromer layer 
        
            pass
        else:
            raise ValueError
        
        
        # input process obj base -> what's that ->  a point with features encoding network #
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
        ### # a token embedding ##
        ### # token embeddings ## # 
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer #### InputProcessObjBaseV2
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    

        
        ### ### 
        
        # # input conditions? # #
        # # input conditions? # #
        
    def forward(self, X, feat, y, cond):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # cond : bsz x nn_feat_dim
        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        # if 
        
        bsz, np, nt, _ = feat.size()
        per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        per_point_feat = torch.cat(
            [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        )
        
        if self.cond_task_type == 'rotation':
        
            cond_encoding = self.cond_processing(cond)
            
            # bsz x np x nt x cond_encoding_dim #
        
            cond_encoding = cond_encoding.unsqueeze(1).unsqueeze(1).repeat(1, np, nt, 1)
            cond_encoding = cond_encoding.contiguous().permute(2, 0, 1, 3).contiguous()
            cond_encoding = cond_encoding.contiguous().view(nt, bsz * np, -1).contiguous()
        elif self.cond_task_type == 'tracking':
            cond_encoding = self.cond_processing(cond) # bsz x nn_ws x latnet_dim 
            cond_encoding = cond_encoding.permute(1, 0, 2) # bsz x nnws x latendim # # 
            cond_encoding = self.cond_positional_encoding(cond_encoding)
            # cond_encoding # transformer encoder cond # 
            # make them into the all zero input #
            cond_encoding = self.transformer_encoder_cond(cond_encoding) # nn_ws x bsz x latnetdim 
            cond_encoding = cond_encoding.unsqueeze(2).repeat(1, 1, np, 1).contiguous() # get the best cond encoding #
            cond_encoding = cond_encoding[0:1]
            cond_encoding = cond_encoding.view(cond_encoding.size(0), bsz * np, -1).contiguous() # get the cond encoding layer
            # print(f"cond_encoding: {cond_encoding.size()}")
        else:
            raise ValueError
            
        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}, cond: {cond.size()}")
            
            
        per_point_embedding = self.input_process(per_point_feat)
        per_point_embedding = self.positional_encoder(per_point_embedding)
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        
        # print(f"per_point_embedding: {per_point_embedding.size()}, cond_encoding: {cond_encoding.size()}    ")
        per_point_embedding = per_point_embedding + cond_encoding ## cond embedding ##
        
        y_expanded = y.squeeze(-1).repeat(np)
        time_embedding = self.time_embedder(y_expanded)
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
        # input dim and the output feat dim = 3 #
        x_out = per_point_output[:, :, :, : self.pos_in_dim] # bsz x np x nt x 3 
        feat_out = per_point_output[:, :, :, self.pos_in_dim :]
        
        
        # feat_out = per_point_output[:, :, :, 3:]
        # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # x_out = per_point_output[:, :, 0, :3]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)



class Transformer_Net_PC_Seq_V3_wcond_V2(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        ## bsz x nn_particles ## # wcondv2 #
        ## bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) ##
        
        ## trnsformer pc seq ##
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information ##
        
        ### TODO: modify such dims ###
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X'] # 1024 ##
        self.feat_hidden_dim = hidden_mlp_dims['feat'] # 1024 # # # get the feat hidden dim ##
        
        # self.per_point_input_dim = 9 # acc ##
        self.per_point_input_dim = self.pos_in_dim + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers # 
        # input process obj base -> what's that ->  a point with features encoding network #
        self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
        self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            
        ### Encoding layer ####
        transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        # transformer encoder # 
        
        ### timesteps embedding layer ###
        self.time_embedder = TimestepEmbedder(self.latent_dim, self.positional_encoder)
        
        
        ##### add the conditional input process processing layer #####
        ##### add the conditional transformer encoder layer #####
        ##### add the conditional timestep embedder layer #####
        self.input_process_cond = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        self.positional_encoder_cond = PositionalEncoding(self.latent_dim, self.dropout)
        # positional encoder cond # 
        
        
        
        #### encoder layers with the timesteps cond ####
        transformer_encoder_layer_cond = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.transformer_encoder_cond = nn.TransformerEncoder(transformer_encoder_layer_cond, num_layers=self.num_layers)
        #### encoder layers with the timesteps cond ####
        
        #### feature converter cond ####
        self.feature_converter_cond = nn.Linear(self.latent_dim, self.latent_dim)
        torch.nn.init.zeros_(self.feature_converter_cond.weight)
        torch.nn.init.zeros_(self.feature_converter_cond.bias) 
        #### feature converter cond ####
        
        transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                        nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)

        self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, # basejtsrel_seqTrans
                                                    num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseRawV5(self.latent_dim, self.per_point_input_dim)
    
    
    
    def forward(self, X, feat, y, X_cond, feat_cond, node_mask):
        # X_cond and the feat_cond are the conditions #
        bsz = X.size(0)
        
        # forward the process #
        # if y.dtype == torch.int32 or y.dtype == torch.long: ##
        #     y = y.float() / 1000.0

        # bsz x nn x 
        # X : bsz x 
        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        bsz, np, nt, _ = feat.size()
        per_point_feat_accs = feat # per point feat accs # #
        # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        
        # per point x; per point feat #
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
        per_point_feat = torch.cat(
            [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
        )
        
        ### per point x cond ###
        per_point_x_cond = X_cond
        per_point_feat_cond = torch.cat(
            [per_point_x_cond, feat_cond], dim=-1
        )
        
        
        
        # # X : bsz x nn_points x (T x per_time_act_dim)
        # tot_n_feats = feat.size(-1) # 
        # np = feat.size(1)
        # nt = tot_n_feats // self.per_point_input_dim_acc
        # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
        # per point embedding #
        
        # per_point_feat = torch.cat(
        #     [
        #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
        #     ], dim=-1
        # )
        
        per_point_embedding_cond = self.input_process_cond(per_point_feat_cond)
        per_point_embedding_cond = self.positional_encoder_cond(per_point_embedding_cond)
        # transformer encoder cond #
        per_point_embedding_cond = self.transformer_encoder_cond(per_point_embedding_cond) 
        
        per_point_embedding_cond = self.feature_converter_cond(per_point_embedding_cond)
        
        
        per_point_embedding = self.input_process(per_point_feat)
        per_point_embedding = self.positional_encoder(per_point_embedding) # positional encoder #
        per_point_embedding = self.transformer_encoder(per_point_embedding)
        
        
        per_point_embedding = per_point_embedding + per_point_embedding_cond
        
        # y
        y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
        time_embedding = self.time_embedder(y_expanded)
        
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, per_point_embedding], dim=0
        )
        
        # per point output 
        per_point_output = self.transformer_with_timesteps_encoder(
            per_point_embedding_with_timesteps
        )[1:]
        
        per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
        # input dim and the output feat dim = 3 #
        x_out = per_point_output[:, :, :, :3] # get the x_out forjthe x # bsz x np x nt x 3 
        feat_out = per_point_output[:, :, :, 3:] # get the feat out for the feat # bsz x np x nt x feat_dim #
        
        
        # feat_out = per_point_output[:, :, :, 3:]
        # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
        # x_out = per_point_output[:, :, 0, :3]
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)

  
        
        
    # def forward(self, X, feat, y, node_mask):
    #     bsz = X.size(0)
        
    #     # forward the process #
    #     # if y.dtype == torch.int32 or y.dtype == torch.long: ##
    #     #     y = y.float() / 1000.0

    #     # bsz x nn x 
    #     # X : bsz x 
        
    #     # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
    #     bsz, np, nt, _ = feat.size()
    #     per_point_feat_accs = feat # per point feat accs # #
    #     # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        
    #     # per point x; per point feat #
    #     per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point x #
        
    #     per_point_feat = torch.cat(
    #         [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ## 
    #     )
        
        
    #     ### 
    #     # # X : bsz x nn_points x (T x per_time_act_dim)
    #     # tot_n_feats = feat.size(-1) # 
    #     # np = feat.size(1)
    #     # nt = tot_n_feats // self.per_point_input_dim_acc
    #     # per_point_feat_accs = feat.contiguous().view(bsz, -1, nt, self.per_point_input_dim_acc).contiguous()
        
    #     # per_point_feat = torch.cat(
    #     #     [
    #     #         X.unsqueeze(2).repeat(1, 1, nt, 1), per_point_feat_accs
    #     #     ], dim=-1
    #     # )
        
    #     per_point_embedding = self.input_process(per_point_feat)
    #     per_point_embedding = self.positional_encoder(per_point_embedding) # positional encoder #
    #     per_point_embedding = self.transformer_encoder(per_point_embedding)
        
    #     # y
    #     y_expanded = y.squeeze(-1).repeat(np) # .unsqueeze(0).repeat(nt, np)
    #     time_embedding = self.time_embedder(y_expanded)
    #     per_point_embedding_with_timesteps = torch.cat(
    #         [time_embedding, per_point_embedding], dim=0
    #     )
        
    #     # per point output 
    #     per_point_output = self.transformer_with_timesteps_encoder(
    #         per_point_embedding_with_timesteps
    #     )[1:]
        
    #     per_point_output = self.output_process(per_point_output, per_point_feat) # nb x np x nt x latent_dim 
        
    #     # input dim and the output feat dim = 3 #
    #     x_out = per_point_output[:, :, :, :3] # get the x_out forjthe x # bsz x np x nt x 3 
    #     feat_out = per_point_output[:, :, :, 3:] # get the feat out for the feat # bsz x np x nt x feat_dim #
        
        
    #     # feat_out = per_point_output[:, :, :, 3:]
    #     # feat_out = feat_out.contiguous().view(bsz, np, -1).contiguous()
    #     # x_out = per_point_output[:, :, 0, :3]
        
    #     return utils.PlaceHolder(X=x_out, E=feat_out, y=y)



### mlp net for processing point cloud data ###
class Transformer_Net_PC_Seq_V3_KineDiff(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU()):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #
        
        # pos in dim; feat in dim #
        
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        
        # input process obj base -> what's that ->  a point with features encoding network #
        # self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        
       
        
        # ### feature input jdim ## feature hidden dim ##
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        # # perpoint glb feature # 
        
        ## TODO: modify the feat_in_dim ##
        ###### ==== input process pc ===== #####
        self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # bsz x ws x 1 x (feat_dim)
        self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        ### positional 
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout) ##     
        
        ### Encoding layer ###
        # transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
        #                                                 nhead=self.num_heads,
        #                                                 dim_feedforward=self.ff_size,
        #                                                 dropout=self.dropout,
        #                                                 activation=self.activation)

        # self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, # basejtsrel_seqTrans
        #                                             num_layers=self.num_layers)
        
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        # 
        self.concat_latent_dim = self.latent_dim + self.latent_dim # if self.concat_two_dims else self.latent_dim
        
        ### timesteps embedding layer ###
        self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout)
        self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        # transformer_encoder_layer_with_timesteps = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim,
        #                                                 nhead=self.num_heads,
        #                                                 dim_feedforward=self.ff_size,
        #                                                 dropout=self.dropout,
        #                                                 activation=self.activation)

        # self.transformer_with_timesteps_encoder = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps, num_layers=self.num_layers)
        
        # latnet dim # 
        transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        # per point input dim #
        # 
        pc_dec_input_dim = self.concat_latent_dim
        # pc_dec_input_dim = 
        self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        self.output_process_feat = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.feat_in_dim)
        
        
        
    def forward(self, X, feat, y, node_mask=None, cond=None):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        # bsz x nn
        # X : bsz x 
        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        # transormer model #
        bsz, np, nt, _ = feat.size() # continuous #
        per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        

        # per point x #
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous()
        
        # per_point_feat = torch.cat(
        #     [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ###
        # )
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}") # bsz x latent_dim # #
        obj_pts_embedding, obj_glb_embedding = self.input_process_pc(per_point_x) # bsz x latent_dim
        obj_glb_embedding = obj_glb_embedding.contiguous().transpose(0, 1).contiguous() # (n_bsz x np) x nt x embedding_dim
        
        feat_embedding = self.input_process_feat(per_point_feat_accs) # nt x (n_bsz x np) x embedding_dim
        feat_embedding = self.positional_encoder_feat(feat_embedding) # get the positional embedding
        feat_embedding = self.transformer_encoder_feat(feat_embedding) # nt x (n_bsz x np) x embedding_dim 
        
        # 
        expanded_obj_glb_embedding = obj_glb_embedding.repeat(nt, 1, 1).contiguous() ## nt x (n_bsz x np) x embedding_dim
        
        
        # print(f"feat_embedding: {feat_embedding.size()}, expanded_obj_glb_embedding: {expanded_obj_glb_embedding.size()}")
        cat_feat_obj_embedding = torch.cat(
            [feat_embedding, expanded_obj_glb_embedding], dim=-1 ## get the feat obj embedding
        )
        # cat_feat_obj_embedding = self.positional_encoder_feat(cat_feat_obj_embedding)
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        # y_expanded_feat = y.squeeze(-1).repeat(np)
        # time_embedding_feat = self.time_embedder(y_expanded_feat)
        
        # nt x (bsz x np) x encoded_dim
        dec_output_x_feat = time_embedding + per_point_feat_output
        # 
        per_point_output = self.output_process(dec_output_x_feat, per_point_x, per_point_feat_accs, obj_glb_embedding.contiguous().transpose(0, 1).contiguous(), obj_pts_embedding) # nb x np x nt x latent_dim 
        per_point_output_feat = self.output_process_feat(per_point_feat_output, per_point_feat_accs)
        
        x_out = per_point_output
        feat_out = per_point_output_feat
        
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)




# create the v2 model for kinematics diff? #
class Transformer_Net_PC_Seq_V3_KineDiff_AE(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #


        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        
        # #
        # input process obj base -> what's that ->  a point with features encoding network #
        # self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 
        


        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        ### Encoders ###
        # conditions -> add the conditions #
        # encoders like such two input processing part #
        # then the concate features are fed to the transformer encoder layer #
        # then the features are decoded back to get the output ## after that get the output #
        
        ## TODO: modify the feat_in_dim ##
        ###### ==== input process pc ===== #####
        self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # bsz x ws x 1 x (feat_dim)
        self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        if self.traj_cond:
            # ge the input process pc # 
            self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            pass
        
        ### positional 
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout)
        
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=self.latent_dim,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        
        self.concat_latent_dim = self.latent_dim + self.latent_dim
        
        ### Encoders ###
        
        
        ### timesteps embedding layer ###
        self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout)
        self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        
        
        transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        # [noised point feature, noised feature feature] -> feed to the transformer encoder # 
        # treat it as the denoiser? #
        # denoied late 
        # 
        self.pc_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        self.feat_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        
        pc_dec_input_dim = self.concat_latent_dim
        # pc_dec_input_dim = 
        self.output_process = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        # self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        self.output_process_feat = OutputProcessObjBaseRawV5_V2(self.latent_dim, self.feat_in_dim)
        # 

    def encode(self, X, feat):
        # encode the feature #
        # X, feat #
        x_pts_feat, x_glb_feat = self.input_process_pc(X) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model 
        # 
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    # # decode the feature # # 
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        tot_decoded_feats = {
            'X': decoded_pts,
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    def forward(self, pts_feat, feat_feat, y, node_mask=None, cond=None):
        # pts_feat # bsz x latent_dim 
        # feat_feat: nn_ts x bsz x latent_dim 
        
        
        
        nn_ts, tot_bsz = feat_feat.size()[:2]
        expanded_pts_feat = pts_feat.unsqueeze(0).repeat(nn_ts, 1, 1)
        cat_feat_obj_embedding = torch.cat(
            [expanded_pts_feat, feat_feat], dim=-1
        )
        
        if self.traj_cond:
            cond_X = cond['X']
            cond_E = cond['E'] #get the pts and features # 
            cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
            cond_glb_feat = cond_glb_feat[:, 0, :]
            cond_encoded_feat = self.cond_input_process_feat(cond_E)
            expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
            
            cat_cond_feat_obj_embedding = torch.cat(
                [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
            )
            cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding
            
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        # per_point_feat_output : nn_ts x bsz x latent_dim #
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        decoded_pts_feat = per_point_feat_output[-1]
        decoded_pts_feat = self.pc_latent_processing(decoded_pts_feat) # 
        per_point_feat_output = self.feat_latent_processing(per_point_feat_output)
        # decoded_feat = {
        #     'pts_feat': decoded_pts_feat,
        #     'feat_feat': per_point_feat_output
        # }
        
        decoded_feat = utils.PlaceHolder(X=decoded_pts_feat, E=per_point_feat_output, y=y)
        
        return decoded_feat # return the decoded features #
    
    
    
    
        
    def forward_bak(self, X, feat, y, node_mask=None, cond=None):
        bsz = X.size(0)
        
        # if y.dtype == torch.int32 or y.dtype == torch.long:
        #     y = y.float() / 1000.0

        
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}")
        
        # transormer model #
        bsz, np, nt, _ = feat.size() # continuous #
        per_point_feat_accs = feat # .contiguous().transpose(0, 2, 1, 3).contiguous() ## get the per point feat accs 
        

        # per point x #
        per_point_x = X # .contiguous().transpose(0, 2, 1, 3).contiguous()
        
        # per_point_feat = torch.cat(
        #     [per_point_x, per_point_feat_accs], dim=-1 ### n_bsz x np x nt x 6 ###
        # )
        # print(f"X: {X.size()}, feat: {feat.size()}, y: {y.size()}") # bsz x latent_dim # #
        obj_pts_embedding, obj_glb_embedding = self.input_process_pc(per_point_x) # bsz x latent_dim
        obj_glb_embedding = obj_glb_embedding.contiguous().transpose(0, 1).contiguous() # (n_bsz x np) x nt x embedding_dim
        
        feat_embedding = self.input_process_feat(per_point_feat_accs) # nt x (n_bsz x np) x embedding_dim
        feat_embedding = self.positional_encoder_feat(feat_embedding) # get the positional embedding
        feat_embedding = self.transformer_encoder_feat(feat_embedding) # nt x (n_bsz x np) x embedding_dim 
        
        # 
        expanded_obj_glb_embedding = obj_glb_embedding.repeat(nt, 1, 1).contiguous() ## nt x (n_bsz x np) x embedding_dim
        
        
        # print(f"feat_embedding: {feat_embedding.size()}, expanded_obj_glb_embedding: {expanded_obj_glb_embedding.size()}")
        cat_feat_obj_embedding = torch.cat(
            [feat_embedding, expanded_obj_glb_embedding], dim=-1 ## get the feat obj embedding
        )
        # cat_feat_obj_embedding = self.positional_encoder_feat(cat_feat_obj_embedding)
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        # y_expanded_feat = y.squeeze(-1).repeat(np)
        # time_embedding_feat = self.time_embedder(y_expanded_feat)
        
        # nt x (bsz x np) x encoded_dim
        dec_output_x_feat = time_embedding + per_point_feat_output
        # 
        per_point_output = self.output_process(dec_output_x_feat, per_point_x, per_point_feat_accs, obj_glb_embedding.contiguous().transpose(0, 1).contiguous(), obj_pts_embedding) # nb x np x nt x latent_dim 
        per_point_output_feat = self.output_process_feat(per_point_feat_output, per_point_feat_accs)
        
        x_out = per_point_output
        feat_out = per_point_output_feat
        
        
        return utils.PlaceHolder(X=x_out, E=feat_out, y=y)





class Transformer_Net_PC_Seq_V3_KineDiff_AE_V2(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #


        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        
        # 
        # input process obj base -> what's that ->  a point with features encoding network #
        # self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 

        
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        ### Encoders ###
        # conditions -> add the conditions #
        # encoders like such two input processing part #
        # then the concate features are fed to the transformer encoder layer #
        # then the features are decoded back to get the output ## after that get the output #
        
        ## TODO: modify the feat_in_dim ##
        ###### ==== input process pc ===== #####
        self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # bsz x ws x 1 x (feat_dim) # input #
        self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        self.input_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout) # positional encodings and the dropout #
        input_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers)
        ######### input pc and the feature processing #########
        
        
        if self.traj_cond:
            # ge the input process pc #  # trajectory encoding --- with the AE jencoding s 
            self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            
            self.cond_input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            cond_input_transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.cond_input_transformer_encoder = nn.TransformerEncoder(cond_input_transformer_encoder_layer, num_layers=self.num_layers)
            
            pass
        
        ### positional 
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout)
        
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=self.latent_dim ,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        
        self.concat_latent_dim = self.latent_dim + self.latent_dim
        
        ### Encoders ###
        
        
        ### timesteps embedding layer ###
        self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout)
        self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        
        
        transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        # [noised point feature, noised feature feature] -> feed to the transformer encoder # 
        # treat it as the denoiser? #
        # denoied late 
        # 
        self.pc_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        self.feat_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        
        pc_dec_input_dim = self.concat_latent_dim
        

        self.output_process_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout)
        
        output_process_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.output_process_transformer_encoder = nn.TransformerEncoder(output_process_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        self.output_process = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        # self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        self.output_process_feat = OutputProcessObjBaseRawV5_V2(self.latent_dim, self.feat_in_dim)
    

    def encode(self, X, feat):
        # encode the feature #
        # X, feat #
        x_pts_feat, x_glb_feat = self.input_process_pc(X) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model  # nt x np x embedding_dim #
        
        expanded_x_glb_feat = x_glb_feat.unsqueeze(0).contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() ## nt x np x ebmedding_dim 
        # encoded_feat = torch.cat([encoded_feat, expanded_x_glb_feat], dim=-1) # nt x np x (2 * embedding_dim) #
        encoded_feat = self.input_positional_encoder(encoded_feat)
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nt x np x (2 * embedding_dim) #
        
        self.nt = encoded_feat.size(0) # nt of the encoded features #
        #
        encoded_feat = encoded_feat[0:1]
        # # encoded_feat
        # encoded_feat = self.input_positional_encoder(encoded_feat ) # nt x np x embedding_dim # 
        # # encode the features #
        
        
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        
        # output_process_positional_encoder # # 
        encoded_feat = encoded_feat.contiguous().repeat(self.nt, 1, 1).contiguous() 
        # encoded_feat # 
        encoded_feat = self.output_process_positional_encoder(encoded_feat)
        encoded_feat = self.output_process_transformer_encoder(encoded_feat)
        # 
        
        
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        tot_decoded_feats = {
            'X': decoded_pts,
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    
    def forward(self, pts_feat, feat_feat, y, node_mask=None, cond=None):
        # pts_feat # bsz x latent_dim 
        # feat_feat: nn_ts x bsz x latent_dim 
        
        nn_ts, tot_bsz = feat_feat.size()[:2]
        expanded_pts_feat = pts_feat.unsqueeze(0).repeat(nn_ts, 1, 1)
        cat_feat_obj_embedding = torch.cat(
            [expanded_pts_feat, feat_feat], dim=-1
        )
        
        if self.traj_cond:
            cond_X = cond['X']
            cond_E = cond['E'] #get the pts and features # 
            cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
            cond_glb_feat = cond_glb_feat[:, 0, :]
            cond_encoded_feat = self.cond_input_process_feat(cond_E)
            
            cond_encoded_feat = self.cond_input_positional_encoder(cond_encoded_feat)
            cond_encoded_feat = self.cond_input_transformer_encoder(cond_encoded_feat) # get the cond encoded features ## 
            cond_encoded_feat = cond_encoded_feat[0:1]
            
            expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
            
            cat_cond_feat_obj_embedding = torch.cat(
                [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
            )
            cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding
            
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        # perpoint feat output #
        # per_point_feat_output : nn_ts x bsz x latent_dim #
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        decoded_pts_feat = per_point_feat_output[-1]
        decoded_pts_feat = self.pc_latent_processing(decoded_pts_feat) # 
        per_point_feat_output = self.feat_latent_processing(per_point_feat_output)
        # decoded_feat = {
        #     'pts_feat': decoded_pts_feat,
        #     'feat_feat': per_point_feat_output
        # }
        
        decoded_feat = utils.PlaceHolder(X=decoded_pts_feat, E=per_point_feat_output, y=y)
        
        return decoded_feat
    




class Transformer_Net_PC_Seq_V3_KineDiff_AE_V3(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, w_glb_traj_feat_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #

        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        # self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9 # add the pos in dim #
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        self.w_glb_traj_feat_cond = w_glb_traj_feat_cond
        
        # 
        # input process obj base -> what's that ->  a point with features encoding network #
        # self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 

        
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        ### Encoders ###
        # conditions -> add the conditions #
        # encoders like such two input processing part #
        # then the concate features are fed to the transformer encoder layer #
        # then the features are decoded back to get the output ## after that get the output #
        
        # add an input process fdeaureblockil # add an input foundmenatl #
        
        ## TODO: modify the feat_in_dim ##
        ###### ==== input process pc ===== #####
        # self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # # bsz x ws x 1 x (feat_dim) # input #
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        # input base # # input process obj base # #
        self.input_process = InputProcessObjBaseV7(self.feat_in_dim, self.latent_dim)
        
        
        
        self.input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout) # positional encodings and the dropout # # get the positional encoders #
        # add the positional encoder -> feed to the transformer encoder #
        
        input_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers) # after we've got features after the transformer encoder # 
        
        
        if self.w_glb_traj_feat_cond:
            self.input_process_glb_traj = InputProcessObjBaseV7(self.feat_in_dim, self.latent_dim)
            self.input_positional_encoder_glb_traj = PositionalEncoding(self.latent_dim, self.dropout)
            input_transformer_encoder_layer_glb_traj = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.input_transformer_encoder_glb_traj = nn.TransformerEncoder(input_transformer_encoder_layer_glb_traj, num_layers=self.num_layers)

            tot_feat_in_dim = self.latent_dim * 2
        else:
            tot_feat_in_dim = self.latent_dim
        
        
        
        #### ocpy the encoded features into multiple copys ####
        #### add the positional encodeing again ####
        self.positional_encoder_feat = PositionalEncoding(tot_feat_in_dim, self.dropout)
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=tot_feat_in_dim ,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseV7(self.feat_in_dim, tot_feat_in_dim)
        
        ##### add a output block which uses the output feature to predict their corresponding hand pose sequence and the object pose sequence #####
        
        
        
        # self.input_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout) # positional encodings and the dropout #
        # input_transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim , 
        #     nhead=self.num_heads,
        #     dim_feedforward=self.ff_size,
        #     dropout=self.dropout,
        #     activation=self.activation
        # )
        # self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers)
        ######### input pc and the feature processing #########
        
        ####### trajectory conditional input encoders #######
        # if self.traj_cond:
        #     # ge the input process pc #  # trajectory encoding --- with the AE jencoding s 
        #     self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
        #     self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            
        #     self.cond_input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        #     cond_input_transformer_encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=self.latent_dim , 
        #         nhead=self.num_heads,
        #         dim_feedforward=self.ff_size,
        #         dropout=self.dropout,
        #         activation=self.activation
        #     )
        #     self.cond_input_transformer_encoder = nn.TransformerEncoder(cond_input_transformer_encoder_layer, num_layers=self.num_layers)
            
        #     pass
        ####### trajectory conditional input encoders #######
        
        ### positional ###
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        
        
        # self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout)
        
        # transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim ,
        #     nhead=self.num_heads,
        #     dim_feedforward=self.ff_size,
        #     dropout=self.dropout,
        #     activation=self.activation
        # )
        # self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        
        # self.concat_latent_dim = self.latent_dim + self.latent_dim
        
        # ### Encoders ###
        
        
        # ### timesteps embedding layer ###
        # self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout)
        # self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        
        
        # transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
        #                                                 dim_feedforward=self.ff_size,
        #                                                 dropout=self.dropout,
        #                                                 activation=self.activation)
        # self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        # # [noised point feature, noised feature feature] -> feed to the transformer encoder # 
        # # treat it as the denoiser? #
        # # denoied late 
        # # 
        # self.pc_latent_processing = nn.Sequential(
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.latent_dim), 
        #     # nn.ReLU(),
        # )
        
        # self.feat_latent_processing = nn.Sequential(
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.latent_dim), 
        #     # nn.ReLU(),
        # )
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        
        # pc_dec_input_dim = self.concat_latent_dim
        

        # self.output_process_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout)
        
        # output_process_transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim , 
        #     nhead=self.num_heads,
        #     dim_feedforward=self.ff_size,
        #     dropout=self.dropout,
        #     activation=self.activation
        # )
        # self.output_process_transformer_encoder = nn.TransformerEncoder(output_process_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        # self.output_process = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        # # self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        # self.output_process_feat = OutputProcessObjBaseRawV5_V2(self.latent_dim, self.feat_in_dim)
    

    def encode(self, X, feat):
        # encode the feature #
        # X, feat #
        x_pts_feat, x_glb_feat = self.input_process_pc(X) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model  # nt x np x embedding_dim #
        
        expanded_x_glb_feat = x_glb_feat.unsqueeze(0).contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() ## nt x np x ebmedding_dim 
        # encoded_feat = torch.cat([encoded_feat, expanded_x_glb_feat], dim=-1) # nt x np x (2 * embedding_dim) #
        encoded_feat = self.input_positional_encoder(encoded_feat)
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nt x np x (2 * embedding_dim) #
        
        self.nt = encoded_feat.size(0) # nt of the encoded features #
        #
        encoded_feat = encoded_feat[0:1]
        # # encoded_feat
        # encoded_feat = self.input_positional_encoder(encoded_feat ) # nt x np x embedding_dim # 
        # # encode the features #
        
        
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        
        # output_process_positional_encoder # # 
        encoded_feat = encoded_feat.contiguous().repeat(self.nt, 1, 1).contiguous() 
        # encoded_feat # 
        encoded_feat = self.output_process_positional_encoder(encoded_feat)
        encoded_feat = self.output_process_transformer_encoder(encoded_feat)
        # 
        
        
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        tot_decoded_feats = {
            'X': decoded_pts, # 
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    
    
    def forward(self, pts_feat, feat_feat, tot_feat_feat=None, tot_obj_pts=None, y=None, node_mask=None, cond=None):
        # forward of the model #
        # whether we need the canonicalization in this model?
        
        # pts_feat -- bsz x nn_ts x nn_pts x 3 # pts feat #
        # feat_feat -- bsz x nn_ts x (nn_hand_pose_dim + nn_obj_pos_dim + nn_obj_ornt_dim) #
        x_hand_pose, x_obj_pos, x_obj_ornt = feat_feat[..., : self.feat_in_dim], feat_feat[..., self.feat_in_dim: self.feat_in_dim + 3], feat_feat[..., self.feat_in_dim + 3: self.feat_in_dim + 7]
        # get the x hand pose and x obj jornt # 
        encoded_feat = self.input_process(x_hand_pose, x_obj_pos, x_obj_ornt, pts_feat) # nn_bsz x nn_ts x latne_dim #
        encoded_feat = encoded_feat.contiguous().transpose(1, 0).contiguous() 
        encoded_feat = self.input_positional_encoder(encoded_feat) # nn_ts x nn_bsz x latent_dim # positional encoder #
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nn_ts x nn_bsz x latent_dim # transformer encoder #
        # input transformer encoder #
        last_encoded_feat = encoded_feat[-1:, :, :] # get the last encoded #
        ## NOTE: assume the input conditional window is with the same length as the output windo
        # print(f"last_encoded_feat: {last_encoded_feat.size()}, encoded_feat: {encoded_feat.size()}")
        expanded_encoded_feat = last_encoded_feat.contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() # nn_ts x nn_bsz x latnet_dim #
        # 
        
        if self.w_glb_traj_feat_cond:
            glb_hand_pose, glb_obj_pos, glb_obj_ornt = tot_feat_feat[..., : self.feat_in_dim], tot_feat_feat[..., self.feat_in_dim: self.feat_in_dim + 3], tot_feat_feat[..., self.feat_in_dim + 3: self.feat_in_dim + 7]
            glb_encoded_feat = self.input_process_glb_traj(glb_hand_pose, glb_obj_pos, glb_obj_ornt, tot_obj_pts)
            glb_encoded_feat = glb_encoded_feat.contiguous().transpose(1, 0).contiguous() 
            glb_encoded_feat = self.input_positional_encoder_glb_traj(glb_encoded_feat) # nn_ts x nn_bsz x latent_dim #
            # iput process positional encoder glb traj # 
            glb_encoded_feat = self.input_transformer_encoder_glb_traj(glb_encoded_feat)
            last_glb_encoded_feat = glb_encoded_feat[-1:, :, :] # 
            expanded_glb_encoded_feat = last_glb_encoded_feat.contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() # nn_ts x nn_bsz x latnet_dim 
            expanded_encoded_feat = torch.cat([ expanded_encoded_feat, expanded_glb_encoded_feat ], dim=-1) # 2*latent_dim --- enocded features #
            
            
            
        
        # print(f"expanded_encoded_feat: {expanded_encoded_feat.size()}") # positional encoder feature #
        expanded_encoded_feat = self.positional_encoder_feat(expanded_encoded_feat)
        decoded_feat = self.transformer_encoder_feat(expanded_encoded_feat)
        # joint_pos, obj_pos, obj_ornt = self.decode(decoded_feat)
        
        decoded_feat = decoded_feat.contiguous().transpose(1, 0).contiguous() 
        
        x_hand_pose, x_obj_pos, x_obj_ornt = self.output_process(decoded_feat )
        
        # rt val dict #
        rt_val_dict = {
            'hand_pose': x_hand_pose,
            'obj_pos': x_obj_pos, 
            'obj_ornt': x_obj_ornt
        }
        return rt_val_dict
        
        
        
        # # pts_feat # bsz x latent_dim # bsz x latent_dim # # latent_dim # #
        # # feat_feat: nn_ts x bsz x latent_dim 
        
        
        # nn_ts, tot_bsz = feat_feat.size()[:2]
        # expanded_pts_feat = pts_feat.unsqueeze(0).repeat(nn_ts, 1, 1)
        # cat_feat_obj_embedding = torch.cat(
        #     [expanded_pts_feat, feat_feat], dim=-1
        # )
        
        # if self.traj_cond:
        #     cond_X = cond['X']
        #     cond_E = cond['E'] #get the pts and features # 
        #     cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
        #     cond_glb_feat = cond_glb_feat[:, 0, :]
        #     cond_encoded_feat = self.cond_input_process_feat(cond_E)
            
        #     cond_encoded_feat = self.cond_input_positional_encoder(cond_encoded_feat)
        #     cond_encoded_feat = self.cond_input_transformer_encoder(cond_encoded_feat) # get the cond encoded features ## 
        #     cond_encoded_feat = cond_encoded_feat[0:1]
            
        #     expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
            
        #     cat_cond_feat_obj_embedding = torch.cat(
        #         [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
        #     )
        #     cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding
            
        
        # y_expanded = y.squeeze(-1) # nt x 1 #  
        # # print(f"y_expanded: {y_expanded.size()}, ")
        # time_embedding = self.time_embedder(y_expanded)
        # # print(f"time_embedding: {time_embedding.size()}")
        
        # per_point_embedding_with_timesteps = torch.cat(
        #     [time_embedding, cat_feat_obj_embedding], dim=0
        # )
        # # perpoint feat output #
        # # per_point_feat_output : nn_ts x bsz x latent_dim #
        # per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
        #     per_point_embedding_with_timesteps
        # )[1:]
        
        # decoded_pts_feat = per_point_feat_output[-1]
        # decoded_pts_feat = self.pc_latent_processing(decoded_pts_feat) # 
        # per_point_feat_output = self.feat_latent_processing(per_point_feat_output)
        # # decoded_feat = {
        # #     'pts_feat': decoded_pts_feat,
        # #     'feat_feat': per_point_feat_output
        # # }
        
        # decoded_feat = utils.PlaceHolder(X=decoded_pts_feat, E=per_point_feat_output, y=y)
        
        # return decoded_feat
    





class Transformer_Net_PC_Seq_V3_KineDiff_AE_V4(nn.Module):
    # whether to add the model with the time step conditions #
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, w_timestep_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual)
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat']
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        
        # whether to add the timestep conditioning #
        self.w_timestep_cond = w_timestep_cond
        # maxx_timestep_cond, maxx_input_traj_length #
        self.maxx_timestep_cond = 1000
        self.maxx_input_traj_length = 500
        
        
        ###### ==== input process pc ===== #####
        self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # bsz x ws x 1 x (feat_dim) 
        self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        self.input_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout) # positional encodings and the dropout #
        input_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers)
        ######### input pc and the feature processing #########
        
        
        if self.traj_cond:
            # sparse planing using the conditional diffusion #
            # ge the input process pc #  # trajectory encoding --- with the AE encoding s 
            self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            
            self.cond_input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
            cond_input_transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.cond_input_transformer_encoder = nn.TransformerEncoder(cond_input_transformer_encoder_layer, num_layers=self.num_layers)
            
            # encode them into a single features and then use the modle to predict all the future features #
            
            self.cond_input_process_feat_hist = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            self.cond_input_positional_encoder_hist = PositionalEncoding(self.latent_dim, self.dropout)
            cond_input_transformer_encoder_layer_hist = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.cond_input_transformer_encoder_hist = nn.TransformerEncoder(cond_input_transformer_encoder_layer_hist, num_layers=self.num_layers)
            
            pass
        
        ### positional 
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout)
        
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=self.latent_dim ,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        
        self.concat_latent_dim = self.latent_dim + self.latent_dim
        
        ### Encoders ###
        
        
        ### timesteps embedding layer ###
        self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout)
        self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        
        
        transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        # [noised point feature, noised feature feature] -> feed to the transformer encoder # 
        # treat it as the denoiser? #
        # denoied late 
        # 
        self.pc_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        self.feat_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        
        pc_dec_input_dim = self.concat_latent_dim
        

        self.output_process_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout)
        
        output_process_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.output_process_transformer_encoder = nn.TransformerEncoder(output_process_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        self.output_process = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        # self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        self.output_process_feat = OutputProcessObjBaseRawV5_V2(self.latent_dim, self.feat_in_dim)
    

    def encode(self, X, feat):
        x_pts_feat, x_glb_feat = self.input_process_pc(X)
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        if len(feat.size()) == 3:
            feat = feat.unsqueeze(1) # nbsz x np x nt x nf #
        
        # input process features # 
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model  # nt x np x embedding_dim #
        
        # expanded_x_glb_feat = x_glb_feat.unsqueeze(0).contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() ## nt x np x ebmedding_dim 
        # encoded_feat = torch.cat([encoded_feat, expanded_x_glb_feat], dim=-1) # nt x np x (2 * embedding_dim) #
        encoded_feat = self.input_positional_encoder(encoded_feat)
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nt x np x (2 * embedding_dim) #
        
        # encoded jfeatures #
        self.nt = encoded_feat.size(0) # nt of the encoded features #
        # # encoded # # # encoded # # #
        encoded_feat = encoded_feat[-1:]
        # # encoded # # # encoded # # #
        # encoded_feat = self.input_positional_encoder(encoded_feat ) # nt x np x embedding_dim # 
        # # encode the features # # # positional encoder # #
        
        
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        
        # output_process_positional_encoder # # 
        encoded_feat = encoded_feat.contiguous().repeat(self.nt, 1, 1).contiguous() 
        # encoded_feat # 
        encoded_feat = self.output_process_positional_encoder(encoded_feat)
        encoded_feat = self.output_process_transformer_encoder(encoded_feat)
        # 
        
        
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        if len(decoded_feat.size()) == 4:
            # print(f"decoded_feat: {decoded_feat.size()}")
            decoded_feat = decoded_feat[:, 0, :, :]
        tot_decoded_feats = {
            'X': decoded_pts,
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    # traj conditions #
    def forward(self, pts_feat, feat_feat, y, node_mask=None, cond=None):
        # pts_feat # bsz x latent_dim 
        # feat_feat: nn_ts x bsz x latent_dim 
        
        nn_ts, tot_bsz = feat_feat.size()[:2]
        expanded_pts_feat = pts_feat.unsqueeze(0).repeat(nn_ts, 1, 1)
        cat_feat_obj_embedding = torch.cat(
            [expanded_pts_feat, feat_feat], dim=-1
        )
        
        if self.traj_cond:
            cond_X = cond['X']
            cond_E = cond['E'] #get the pts and features # 
            cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
            cond_glb_feat = cond_glb_feat[:, 0, :]
            if len(cond_E.size()) == 3:
                cond_E = cond_E.unsqueeze(1)
            cond_encoded_feat = self.cond_input_process_feat(cond_E)
            
            cond_encoded_feat = self.cond_input_positional_encoder(cond_encoded_feat)
            cond_encoded_feat = self.cond_input_transformer_encoder(cond_encoded_feat) # get the cond encoded features ## 
            cond_encoded_feat = cond_encoded_feat[0:1]
            
            expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
            
            cat_cond_feat_obj_embedding = torch.cat(
                [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
            )
            cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding
            
            
            ##### Two input conditions #####
            cond_E_hist = cond['history_E']
            if len(cond_E_hist.size()) == 3:
                cond_E_hist = cond_E_hist.unsqueeze(1)
            cond_encoded_feat_hist = self.cond_input_process_feat_hist(cond_E_hist)
            cond_encoded_feat_hist = self.cond_input_positional_encoder_hist(cond_encoded_feat_hist)
            cond_encoded_feat_hist = self.cond_input_transformer_encoder_hist(cond_encoded_feat_hist) # get the cond encoded features ##
            cond_encoded_feat_hist = cond_encoded_feat_hist[0:1]
            expanded_conf_pts_feat_hist = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat_hist.size(0), 1, 1)
            cat_cond_feat_obj_embedding_hist = torch.cat(
                [ expanded_conf_pts_feat_hist, cond_encoded_feat_hist], dim=-1
            )
            cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding_hist
            
            
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        # perpoint feat output #
        # per_point_feat_output : nn_ts x bsz x latent_dim #
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        decoded_pts_feat = per_point_feat_output[-1]
        decoded_pts_feat = self.pc_latent_processing(decoded_pts_feat)
        per_point_feat_output = self.feat_latent_processing(per_point_feat_output)
        # decoded_feat = {
        #     'pts_feat': decoded_pts_feat,
        #     'feat_feat': per_point_feat_output
        # }
        
        decoded_feat = utils.PlaceHolder(X=decoded_pts_feat, E=per_point_feat_output, y=y)
        
        return decoded_feat
    



class Transformer_Net_PC_Seq_V3_KineDiff_AE_V6(nn.Module):
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, w_glb_traj_feat_cond=False, w_timestep_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual) #

        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat'] ## get the act sequence related input dimension 
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        # self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9 # add the pos in dim #
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        self.w_glb_traj_feat_cond = w_glb_traj_feat_cond
        # self.w_timestep_cond = w_timestep_cond # timestetp cond #
        # 
        # whether to add the timestep conditioning #
        self.w_timestep_cond = w_timestep_cond
        # maxx_timestep_cond, maxx_input_traj_length #
        self.maxx_timestep_cond = 1000
        self.maxx_input_traj_length = 500
        
        # input process obj base -> what's that ->  a point with features encoding network #
        # self.input_process = InputProcessObjBaseV5( self.per_point_input_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False ) 

        
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        ### Encoders ###
        # conditions -> add the conditions #
        # encoders like such two input processing part #
        # then the concate features are fed to the transformer encoder layer #
        # then the features are decoded back to get the output ## after that get the output #
        
        # add an input process fdeaureblockil # add an input foundmenatl #
        
        ## TODO: modify the feat_in_dim ##
        ###### ==== input process pc ===== #####
        # self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # # bsz x ws x 1 x (feat_dim) # input #
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        
        # input base # # input process obj base # #
        self.input_process = InputProcessObjBaseV7(self.feat_in_dim, self.latent_dim)
        
        
        
        self.input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length) # positional encodings and the dropout # # get the positional encoders #
        # add the positional encoder -> feed to the transformer encoder #
        
        self.input_position_indicating_encoder =  PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_timestep_cond) # timestep cond #
        
        input_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers) # after we've got features after the transformer encoder # 
        
        
        if self.w_glb_traj_feat_cond:
            self.input_process_glb_traj = InputProcessObjBaseV7(self.feat_in_dim, self.latent_dim)
            self.input_positional_encoder_glb_traj = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
            input_transformer_encoder_layer_glb_traj = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.input_transformer_encoder_glb_traj = nn.TransformerEncoder(input_transformer_encoder_layer_glb_traj, num_layers=self.num_layers)

            tot_feat_in_dim = self.latent_dim * 2
        else:
            tot_feat_in_dim = self.latent_dim
        
        
        
        #### ocpy the encoded features into multiple copys ####
        #### add the positional encodeing again ####
        self.positional_encoder_feat = PositionalEncoding(tot_feat_in_dim, self.dropout, max_len=self.maxx_input_traj_length)
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=tot_feat_in_dim ,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseV7(self.feat_in_dim, tot_feat_in_dim)
        
        ##### add a output block which uses the output feature to predict their corresponding hand pose sequence and the object pose sequence #####
        
        
    def encode(self, X, feat):
        # encode the feature #
        # X, feat #
        x_pts_feat, x_glb_feat = self.input_process_pc(X) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model  # nt x np x embedding_dim #
        
        expanded_x_glb_feat = x_glb_feat.unsqueeze(0).contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() ## nt x np x ebmedding_dim 
        # encoded_feat = torch.cat([encoded_feat, expanded_x_glb_feat], dim=-1) # nt x np x (2 * embedding_dim) #
        encoded_feat = self.input_positional_encoder(encoded_feat)
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nt x np x (2 * embedding_dim) #
        
        self.nt = encoded_feat.size(0) # nt of the encoded features #
        #
        encoded_feat = encoded_feat[0:1]
        # # encoded_feat
        # encoded_feat = self.input_positional_encoder(encoded_feat ) # nt x np x embedding_dim # 
        # # encode the features #
        
        
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        
        # output_process_positional_encoder # # 
        encoded_feat = encoded_feat.contiguous().repeat(self.nt, 1, 1).contiguous() 
        # encoded_feat # 
        encoded_feat = self.output_process_positional_encoder(encoded_feat)
        encoded_feat = self.output_process_transformer_encoder(encoded_feat)
        # 
        
        
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        tot_decoded_feats = {
            'X': decoded_pts, # 
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    
    
    def forward(self, pts_feat, feat_feat, tot_feat_feat=None, tot_obj_pts=None, y=None, node_mask=None, cond=None, history_window_index=None):
        # forward of the model #
        # whether we need the canonicalization in this model?
        # pts feat and feat jeat --- but how to add the conditions here ? #
        
        
        assert history_window_index is not None
        # pts_feat -- bsz x nn_ts x nn_pts x 3 # pts feat #
        # feat_feat -- bsz x nn_ts x (nn_hand_pose_dim + nn_obj_pos_dim + nn_obj_ornt_dim) #
        x_hand_pose, x_obj_pos, x_obj_ornt = feat_feat[..., : self.feat_in_dim], feat_feat[..., self.feat_in_dim: self.feat_in_dim + 3], feat_feat[..., self.feat_in_dim + 3: self.feat_in_dim + 7]
        # get the x hand pose and x obj jornt # 
        encoded_feat = self.input_process(x_hand_pose, x_obj_pos, x_obj_ornt, pts_feat) # nn_bsz x nn_ts x latne_dim #
        encoded_feat = encoded_feat.contiguous().transpose(1, 0).contiguous() 
        encoded_feat = self.input_positional_encoder(encoded_feat) # nn_ts x nn_bsz x latent_dim # positional encoder #
        
        # get the history window index #
        history_window_index = history_window_index.long()
        encoded_feat = self.input_position_indicating_encoder.forward_batch_selection(encoded_feat, time_index=history_window_index)
        
        # 
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nn_ts x nn_bsz x latent_dim # transformer encoder #
        # input transformer encoder #
        last_encoded_feat = encoded_feat[-1:, :, :] # get the last encoded #
        ## NOTE: assume the input conditional window is with the same length as the output windo
        # print(f"last_encoded_feat: {last_encoded_feat.size()}, encoded_feat: {encoded_feat.size()}")
        expanded_encoded_feat = last_encoded_feat.contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() # nn_ts x nn_bsz x latnet_dim #
        # 
        
        if self.w_glb_traj_feat_cond:
            glb_hand_pose, glb_obj_pos, glb_obj_ornt = tot_feat_feat[..., : self.feat_in_dim], tot_feat_feat[..., self.feat_in_dim: self.feat_in_dim + 3], tot_feat_feat[..., self.feat_in_dim + 3: self.feat_in_dim + 7]
            glb_encoded_feat = self.input_process_glb_traj(glb_hand_pose, glb_obj_pos, glb_obj_ornt, tot_obj_pts)
            glb_encoded_feat = glb_encoded_feat.contiguous().transpose(1, 0).contiguous() 
            glb_encoded_feat = self.input_positional_encoder_glb_traj(glb_encoded_feat) # nn_ts x nn_bsz x latent_dim #
            # iput process positional encoder glb traj # 
            glb_encoded_feat = self.input_transformer_encoder_glb_traj(glb_encoded_feat)
            last_glb_encoded_feat = glb_encoded_feat[-1:, :, :] # 
            expanded_glb_encoded_feat = last_glb_encoded_feat.contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() # nn_ts x nn_bsz x latnet_dim 
            expanded_encoded_feat = torch.cat([ expanded_encoded_feat, expanded_glb_encoded_feat ], dim=-1) # 2*latent_dim --- enocded features #
            
            
            
        
        # print(f"expanded_encoded_feat: {expanded_encoded_feat.size()}") # positional encoder feature #
        expanded_encoded_feat = self.positional_encoder_feat(expanded_encoded_feat)
        decoded_feat = self.transformer_encoder_feat(expanded_encoded_feat)
        # joint_pos, obj_pos, obj_ornt = self.decode(decoded_feat)
        
        decoded_feat = decoded_feat.contiguous().transpose(1, 0).contiguous() 
        
        x_hand_pose, x_obj_pos, x_obj_ornt = self.output_process(decoded_feat )
        
        # rt val dict #
        rt_val_dict = {
            'hand_pose': x_hand_pose,
            'obj_pos': x_obj_pos, 
            'obj_ornt': x_obj_ornt
        }
        return rt_val_dict
        
    



class Transformer_Net_PC_Seq_V3_KineDiff_AE_V5(nn.Module):
    # whether to add the model with the time step conditions #
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, w_timestep_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual)
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat']
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        
        # whether to add the timestep conditioning #
        self.w_timestep_cond = w_timestep_cond
        # maxx_timestep_cond, maxx_input_traj_length #
        self.maxx_timestep_cond = 1000
        self.maxx_input_traj_length = 500
        
        
        ###### ==== input process pc ===== ##### #
        self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        # bsz x ws x 1 x (feat_dim) 
        self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        self.input_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length) # positional encodings and the dropout #
        
        
        
        input_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        ######### input pc and the feature processing #########
        
        
        if self.traj_cond:
            # sparse planing using the conditional diffusion #
            # ge the input process pc #  # trajectory encoding --- with the AE encoding s 
            self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            
            self.cond_input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
            cond_input_transformer_encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.cond_input_transformer_encoder = nn.TransformerEncoder(cond_input_transformer_encoder_layer, num_layers=self.num_layers)
            
            # encode them into a single features and then use the modle to predict all the future features #
            
            self.cond_input_process_feat_hist = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            self.cond_input_positional_encoder_hist = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
            cond_input_transformer_encoder_layer_hist = nn.TransformerEncoderLayer(
                d_model=self.latent_dim , 
                nhead=self.num_heads,
                dim_feedforward=self.ff_size,
                dropout=self.dropout,
                activation=self.activation
            )
            self.cond_input_transformer_encoder_hist = nn.TransformerEncoder(cond_input_transformer_encoder_layer_hist, num_layers=self.num_layers)
            
            pass
        
        ### positional 
        # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=self.latent_dim ,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        
        self.concat_latent_dim = self.latent_dim + self.latent_dim
        
        ### Encoders ###
        
        
        ### timesteps embedding layer ###
        self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        
        
        transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
                                                        dim_feedforward=self.ff_size,
                                                        dropout=self.dropout,
                                                        activation=self.activation)
        self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        # [noised point feature, noised feature feature] -> feed to the transformer encoder # 
        # treat it as the denoiser? #
        # denoied late 
        # 
        self.pc_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        self.feat_latent_processing = nn.Sequential(
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
            nn.Linear(self.concat_latent_dim, self.latent_dim), 
            # nn.ReLU(),
        )
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        
        pc_dec_input_dim = self.concat_latent_dim
        

        self.output_process_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        
        output_process_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.output_process_transformer_encoder = nn.TransformerEncoder(output_process_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        self.output_process = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        # self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        self.output_process_feat = OutputProcessObjBaseRawV5_V2(self.latent_dim, self.feat_in_dim)
    

    def encode(self, X, feat):
        # encode the feature #
        # X, feat #
        x_pts_feat, x_glb_feat = self.input_process_pc(X) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        if len(feat.size()) == 3:
            feat = feat.unsqueeze(1) # nbsz x np x nt x nf #
        
        # input process features # 
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model  # nt x np x embedding_dim #
        
        # expanded_x_glb_feat = x_glb_feat.unsqueeze(0).contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() ## nt x np x ebmedding_dim 
        # encoded_feat = torch.cat([encoded_feat, expanded_x_glb_feat], dim=-1) # nt x np x (2 * embedding_dim) #
        encoded_feat = self.input_positional_encoder(encoded_feat)
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nt x np x (2 * embedding_dim) #
        
        # encoded jfeatures #
        self.nt = encoded_feat.size(0) # nt of the encoded features #
        # # encoded # # # encoded # # #
        encoded_feat = encoded_feat[-1:]
        # # encoded # # # encoded # # #
        # encoded_feat = self.input_positional_encoder(encoded_feat ) # nt x np x embedding_dim # 
        # # encode the features # # # positional encoder # #
        
        
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        
        # output_process_positional_encoder # # 
        encoded_feat = encoded_feat.contiguous().repeat(self.nt, 1, 1).contiguous() 
        # encoded_feat # 
        encoded_feat = self.output_process_positional_encoder(encoded_feat)
        encoded_feat = self.output_process_transformer_encoder(encoded_feat)
        # 
        
        
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        if len(decoded_feat.size()) == 4:
            # print(f"decoded_feat: {decoded_feat.size()}")
            decoded_feat = decoded_feat[:, 0, :, :]
        tot_decoded_feats = {
            'X': decoded_pts,
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    
    def forward(self, pts_feat, feat_feat, y, node_mask=None, cond=None):
        # pts_feat # bsz x latent_dim 
        # feat_feat: nn_ts x bsz x latent_dim 
        
        nn_ts, tot_bsz = feat_feat.size()[:2]
        expanded_pts_feat = pts_feat.unsqueeze(0).repeat(nn_ts, 1, 1)
        cat_feat_obj_embedding = torch.cat(
            [expanded_pts_feat, feat_feat], dim=-1
        )
        
        if self.traj_cond:
            cond_X = cond['X']
            cond_E = cond['E'] #get the pts and features # 
            cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
            cond_glb_feat = cond_glb_feat[:, 0, :]
            if len(cond_E.size()) == 3:
                cond_E = cond_E.unsqueeze(1)
            cond_encoded_feat = self.cond_input_process_feat(cond_E)
            
            cond_encoded_feat = self.cond_input_positional_encoder(cond_encoded_feat)
            cond_encoded_feat = self.cond_input_transformer_encoder(cond_encoded_feat) # get the cond encoded features ## 
            cond_encoded_feat = cond_encoded_feat[0:1]
            
            expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
            
            cat_cond_feat_obj_embedding = torch.cat(
                [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
            )
            cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding
            
            
            ##### Two input conditions #####
            cond_E_hist = cond['history_E']
            if len(cond_E_hist.size()) == 3:
                cond_E_hist = cond_E_hist.unsqueeze(1)
            cond_encoded_feat_hist = self.cond_input_process_feat_hist(cond_E_hist)
            cond_encoded_feat_hist = self.cond_input_positional_encoder_hist(cond_encoded_feat_hist)
            cond_encoded_feat_hist = self.cond_input_transformer_encoder_hist(cond_encoded_feat_hist) # get the cond encoded features ##
            cond_encoded_feat_hist = cond_encoded_feat_hist[0:1]
            expanded_conf_pts_feat_hist = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat_hist.size(0), 1, 1)
            cat_cond_feat_obj_embedding_hist = torch.cat(
                [ expanded_conf_pts_feat_hist, cond_encoded_feat_hist], dim=-1
            )
            cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding_hist
            
            
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        # perpoint feat output #
        # per_point_feat_output : nn_ts x bsz x latent_dim #
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        decoded_pts_feat = per_point_feat_output[-1]
        decoded_pts_feat = self.pc_latent_processing(decoded_pts_feat)
        per_point_feat_output = self.feat_latent_processing(per_point_feat_output)
        # decoded_feat = {
        #     'pts_feat': decoded_pts_feat,
        #     'feat_feat': per_point_feat_output
        # }
        
        decoded_feat = utils.PlaceHolder(X=decoded_pts_feat, E=per_point_feat_output, y=y)
        
        return decoded_feat
    



class Transformer_Net_PC_Seq_V3_KineDiff_AE_V7(nn.Module):
    # whether to add the model with the time step conditions #
    def __init__(self, n_layers: int, input_dims: dict, hidden_mlp_dims: dict,
                 output_dims: dict, act_fn_in: nn.ReLU(), act_fn_out: nn.ReLU(), traj_cond=False, w_timestep_cond=False):
        super().__init__()
        
        # bsz x nn_particles x nn_ts x (dim_state + dim_acc_tau + dim_acc + dim_acc_actual)
        ## transform the X matrix -- bsz x nn_nodes x 2 ---> bsz x (nn_nodes x 2) ---> bsz x hidden_dim ---> bsz x (nn_nodes x 2) ---> for predicting the nodes information
        self.pos_in_dim = input_dims['X']
        self.feat_in_dim = input_dims['feat']
        self.pos_hidden_dim = hidden_mlp_dims['X']
        self.feat_hidden_dim = hidden_mlp_dims['feat']
        # self.concat_two_dims = input_dims['concat_two_dims']
        
        # self.per_point_input_dim = 9 # get the jase pc #
        self.per_point_input_dim = self.pos_in_dim # + self.feat_in_dim
        # self.per_point_input_dim_acc = 9
        self.latent_dim = self.feat_hidden_dim
        self.num_heads = 4
        self.ff_size = self.latent_dim
        self.dropout = 0.0
        self.activation = 'relu'
        self.num_layers = n_layers
        self.traj_cond = traj_cond
        
        # whether to add the timestep conditioning #
        self.w_timestep_cond = w_timestep_cond
        # maxx_timestep_cond, maxx_input_traj_length #
        self.maxx_timestep_cond = 1000
        self.maxx_input_traj_length = 500
        
        
        ###### ==== input process pc ===== ##### #
        self.input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
        
        # input base # # input process obj base # #
        self.input_process = InputProcessObjBaseV7(self.feat_in_dim, self.latent_dim)
        
        
        
        
        self.input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length) # positional encodings and the dropout # # get the positional encoders #
        # add the positional encoder -> feed to the transformer encoder #
        
        input_transformer_encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_dim , 
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers) # after we've got features after the transformer encoder # 
        
        
        # if self.w_glb_traj_feat_cond:
        #     self.input_process_glb_traj = InputProcessObjBaseV7(self.feat_in_dim, self.latent_dim)
        #     self.input_positional_encoder_glb_traj = PositionalEncoding(self.latent_dim, self.dropout)
        #     input_transformer_encoder_layer_glb_traj = nn.TransformerEncoderLayer(
        #         d_model=self.latent_dim , 
        #         nhead=self.num_heads,
        #         dim_feedforward=self.ff_size,
        #         dropout=self.dropout,
        #         activation=self.activation
        #     )
        #     self.input_transformer_encoder_glb_traj = nn.TransformerEncoder(input_transformer_encoder_layer_glb_traj, num_layers=self.num_layers)

        #     tot_feat_in_dim = self.latent_dim * 2
        # else:
        #     tot_feat_in_dim = self.latent_dim
        
        tot_feat_in_dim = self.latent_dim
        
        #### ocpy the encoded features into multiple copys ####
        #### add the positional encodeing again ####
        self.positional_encoder_feat = PositionalEncoding(tot_feat_in_dim, self.dropout, max_len=self.maxx_input_traj_length)
        transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
            d_model=tot_feat_in_dim ,
            nhead=self.num_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout,
            activation=self.activation
        )
        self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        self.output_process = OutputProcessObjBaseV7(self.feat_in_dim, tot_feat_in_dim)
        
        
        self.output_process_pc = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        
        
        
        
        
        # # bsz x ws x 1 x (feat_dim) 
        # self.input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False )
        
        # self.input_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length) # positional encodings and the dropout #
        
        
        
        # input_transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim , 
        #     nhead=self.num_heads,
        #     dim_feedforward=self.ff_size,
        #     dropout=self.dropout,
        #     activation=self.activation
        # )
        # self.input_transformer_encoder = nn.TransformerEncoder(input_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        # ######### input pc and the feature processing #########
        
        
        # if self.traj_cond:
        #     # sparse planing using the conditional diffusion #
        #     # ge the input process pc #  # trajectory encoding --- with the AE encoding s 
        #     self.cond_input_process_pc = InputProcessObjBasePC(3, self.latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
        #     self.cond_input_process_feat = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
            
        #     self.cond_input_positional_encoder = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        #     cond_input_transformer_encoder_layer = nn.TransformerEncoderLayer(
        #         d_model=self.latent_dim , 
        #         nhead=self.num_heads,
        #         dim_feedforward=self.ff_size,
        #         dropout=self.dropout,
        #         activation=self.activation
        #     )
        #     self.cond_input_transformer_encoder = nn.TransformerEncoder(cond_input_transformer_encoder_layer, num_layers=self.num_layers)
            
        #     # encode them into a single features and then use the modle to predict all the future features #
            
        #     self.cond_input_process_feat_hist = InputProcessObjBaseV5( self.feat_in_dim, self.feat_hidden_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=True)
        #     self.cond_input_positional_encoder_hist = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        #     cond_input_transformer_encoder_layer_hist = nn.TransformerEncoderLayer(
        #         d_model=self.latent_dim , 
        #         nhead=self.num_heads,
        #         dim_feedforward=self.ff_size,
        #         dropout=self.dropout,
        #         activation=self.activation
        #     )
        #     self.cond_input_transformer_encoder_hist = nn.TransformerEncoder(cond_input_transformer_encoder_layer_hist, num_layers=self.num_layers)
            
        #     pass
        
        # ### positional 
        # # self.positional_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        # self.positional_encoder_feat = PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        
        # transformer_encoder_layer_feat = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim ,
        #     nhead=self.num_heads,
        #     dim_feedforward=self.ff_size,
        #     dropout=self.dropout,
        #     activation=self.activation
        # )
        # self.transformer_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_feat, num_layers=self.num_layers)
        
        
        # self.concat_latent_dim = self.latent_dim + self.latent_dim
        
        # ### Encoders ###
        
        
        # ### timesteps embedding layer ###
        # self.positional_encoder_time = PositionalEncoding(self.concat_latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        # self.time_embedder = TimestepEmbedder(self.concat_latent_dim, self.positional_encoder_time)
        
        
        
        # transformer_encoder_layer_with_timesteps_feat = nn.TransformerEncoderLayer(d_model=self.concat_latent_dim, nhead=self.num_heads,
        #                                                 dim_feedforward=self.ff_size,
        #                                                 dropout=self.dropout,
        #                                                 activation=self.activation)
        # self.transformer_with_timesteps_encoder_feat = nn.TransformerEncoder(transformer_encoder_layer_with_timesteps_feat, num_layers=self.num_layers)
        
        # # [noised point feature, noised feature feature] -> feed to the transformer encoder # 
        # # treat it as the denoiser? #
        # # denoied late 
        # # 
        # self.pc_latent_processing = nn.Sequential(
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.latent_dim), 
        #     # nn.ReLU(),
        # )
        
        # self.feat_latent_processing = nn.Sequential(
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.concat_latent_dim), nn.ReLU(),
        #     nn.Linear(self.concat_latent_dim, self.latent_dim), 
        #     # nn.ReLU(),
        # )
        
        
        # self.output_process = OutputProcessObjBaseRawV5(self.concat_latent_dim, self.per_point_input_dim)
        
        
        # pc_dec_input_dim = self.concat_latent_dim
        

        # self.output_process_positional_encoder =  PositionalEncoding(self.latent_dim, self.dropout, max_len=self.maxx_input_traj_length)
        
        # output_process_transformer_encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.latent_dim , 
        #     nhead=self.num_heads,
        #     dim_feedforward=self.ff_size,
        #     dropout=self.dropout,
        #     activation=self.activation
        # )
        # self.output_process_transformer_encoder = nn.TransformerEncoder(output_process_transformer_encoder_layer, num_layers=self.num_layers)
        
        
        
        # self.output_process = OutputProcessObjBaseRawPC_V2(self.latent_dim, 128)
        # self.output_process = OutputProcessObjBaseRawPC(self.concat_latent_dim, self.per_point_input_dim)
        # self.output_process_feat = OutputProcessObjBaseRawV5_V2(self.latent_dim, self.feat_in_dim)
    

    def encode(self, X, feat):
        # encode the feature #
        # X, feat #
        x_pts_feat, x_glb_feat = self.input_process_pc(X) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        if len(feat.size()) == 3:
            feat = feat.unsqueeze(1) # nbsz x np x nt x nf #
        
        # input process features # 
        encoded_feat = self.input_process_feat(feat) # nt x (n_bsz x np) x embedding_dim # v2 model  # nt x np x embedding_dim #
        
        # expanded_x_glb_feat = x_glb_feat.unsqueeze(0).contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() ## nt x np x ebmedding_dim 
        # encoded_feat = torch.cat([encoded_feat, expanded_x_glb_feat], dim=-1) # nt x np x (2 * embedding_dim) #
        encoded_feat = self.input_positional_encoder(encoded_feat)
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nt x np x (2 * embedding_dim) #
        
        # encoded jfeatures #
        self.nt = encoded_feat.size(0) # nt of the encoded features #
        # # encoded # # # encoded # # #
        encoded_feat = encoded_feat[-1:]
        # # encoded # # # encoded # # #
        # encoded_feat = self.input_positional_encoder(encoded_feat ) # nt x np x embedding_dim # 
        # # encode the features # # # positional encoder # #
        
        
        tot_encoded_feats = {
            'pts_feat': x_glb_feat,
            'feat_feat': encoded_feat
        }
        return tot_encoded_feats
    
    
    def decode(self, tot_latent_feats): 
        pts_feat = tot_latent_feats['pts_feat']
        encoded_feat = tot_latent_feats['feat_feat']
        
        # output_process_positional_encoder # # 
        encoded_feat = encoded_feat.contiguous().repeat(self.nt, 1, 1).contiguous() 
        # encoded_feat # 
        encoded_feat = self.output_process_positional_encoder(encoded_feat)
        encoded_feat = self.output_process_transformer_encoder(encoded_feat)
        # 
        
        
        decoded_pts = self.output_process(pts_feat)
        decoded_feat = self.output_process_feat(encoded_feat) #
        if len(decoded_feat.size()) == 4:
            # print(f"decoded_feat: {decoded_feat.size()}")
            decoded_feat = decoded_feat[:, 0, :, :]
        tot_decoded_feats = {
            'X': decoded_pts,
            'feat': decoded_feat
        }
        return tot_decoded_feats
    
    
    def forward(self, pts_feat, feat_feat, y=None, node_mask=None, cond=None):
        # pts_feat # bsz x latent_dim 
        # feat_feat: nn_ts x bsz x latent_dim 
        
        x_pts_feat, x_glb_feat = self.input_process_pc(pts_feat) #
        x_glb_feat = x_glb_feat[:, 0, :] # bsz x latent_dim
        
        decoded_pts = self.output_process_pc(x_glb_feat)
        
        # nn_ts, tot_bsz = feat_feat.size()[:2]
        # expanded_pts_feat = pts_feat.unsqueeze(0).repeat(nn_ts, 1, 1)
        # cat_feat_obj_embedding = torch.cat(
        #     [expanded_pts_feat, feat_feat], dim=-1
        # )
        
        # pts_feat -- bsz x nn_ts x nn_pts x 3 # pts feat #
        # feat_feat -- bsz x nn_ts x (nn_hand_pose_dim + nn_obj_pos_dim + nn_obj_ornt_dim) #
        x_hand_pose, x_obj_pos, x_obj_ornt = feat_feat[..., : self.feat_in_dim], feat_feat[..., self.feat_in_dim: self.feat_in_dim + 3], feat_feat[..., self.feat_in_dim + 3: self.feat_in_dim + 7]
        # get the x hand pose and x obj jornt # 
        expanded_pts = pts_feat.unsqueeze(1).repeat(1, x_hand_pose.size(1), 1, 1)
        encoded_feat = self.input_process(x_hand_pose, x_obj_pos, x_obj_ornt, expanded_pts) # nn_bsz x nn_ts x latne_dim #
        encoded_feat = encoded_feat.contiguous().transpose(1, 0).contiguous() 
        encoded_feat = self.input_positional_encoder(encoded_feat) # nn_ts x nn_bsz x latent_dim # positional encoder #
        encoded_feat = self.input_transformer_encoder(encoded_feat) # nn_ts x nn_bsz x latent_dim # transformer encoder #
        # input transformer encoder #
        last_encoded_feat = encoded_feat[-1:, :, :] # get the last encoded #
        ## NOTE: assume the input conditional window is with the same length as the output windo
        # print(f"last_encoded_feat: {last_encoded_feat.size()}, encoded_feat: {encoded_feat.size()}")
        expanded_encoded_feat = last_encoded_feat.contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() # nn_ts x nn_bsz x latnet_dim #
        # 
        
        # if self.w_glb_traj_feat_cond:
        #     glb_hand_pose, glb_obj_pos, glb_obj_ornt = tot_feat_feat[..., : self.feat_in_dim], tot_feat_feat[..., self.feat_in_dim: self.feat_in_dim + 3], tot_feat_feat[..., self.feat_in_dim + 3: self.feat_in_dim + 7]
        #     glb_encoded_feat = self.input_process_glb_traj(glb_hand_pose, glb_obj_pos, glb_obj_ornt, tot_obj_pts)
        #     glb_encoded_feat = glb_encoded_feat.contiguous().transpose(1, 0).contiguous() 
        #     glb_encoded_feat = self.input_positional_encoder_glb_traj(glb_encoded_feat) # nn_ts x nn_bsz x latent_dim #
        #     # iput process positional encoder glb traj # 
        #     glb_encoded_feat = self.input_transformer_encoder_glb_traj(glb_encoded_feat)
        #     last_glb_encoded_feat = glb_encoded_feat[-1:, :, :] # 
        #     expanded_glb_encoded_feat = last_glb_encoded_feat.contiguous().repeat(encoded_feat.size(0), 1, 1).contiguous() # nn_ts x nn_bsz x latnet_dim 
        #     expanded_encoded_feat = torch.cat([ expanded_encoded_feat, expanded_glb_encoded_feat ], dim=-1) # 2*latent_dim --- enocded features #
            
            
            
        
        # print(f"expanded_encoded_feat: {expanded_encoded_feat.size()}") # positional encoder feature #
        expanded_encoded_feat = self.positional_encoder_feat(expanded_encoded_feat)
        decoded_feat = self.transformer_encoder_feat(expanded_encoded_feat)
        # joint_pos, obj_pos, obj_ornt = self.decode(decoded_feat)
        
        decoded_feat = decoded_feat.contiguous().transpose(1, 0).contiguous() 
        
        x_hand_pose, x_obj_pos, x_obj_ornt = self.output_process(decoded_feat )
        
        # rt val dict #
        rt_val_dict = {
            'hand_pose': x_hand_pose,
            'obj_pos': x_obj_pos, 
            'obj_ornt': x_obj_ornt,
            'decoded_pts': decoded_pts,
            'encoded_pts_feat': x_glb_feat,
            'encoded_traj_feat': last_encoded_feat,
        }
        return rt_val_dict
    
    
    
        
        # if self.traj_cond:
        #     cond_X = cond['X']
        #     cond_E = cond['E'] #get the pts and features # 
        #     cond_pts_feat, cond_glb_feat = self.cond_input_process_pc(cond_X)
        #     cond_glb_feat = cond_glb_feat[:, 0, :]
        #     if len(cond_E.size()) == 3:
        #         cond_E = cond_E.unsqueeze(1)
        #     cond_encoded_feat = self.cond_input_process_feat(cond_E)
            
        #     cond_encoded_feat = self.cond_input_positional_encoder(cond_encoded_feat)
        #     cond_encoded_feat = self.cond_input_transformer_encoder(cond_encoded_feat) # get the cond encoded features ## 
        #     cond_encoded_feat = cond_encoded_feat[0:1]
            
        #     expanded_cond_pts_feat = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat.size(0), 1, 1)
            
        #     cat_cond_feat_obj_embedding = torch.cat(
        #         [expanded_cond_pts_feat, cond_encoded_feat], dim=-1
        #     )
        #     cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding
            
            
        #     ##### Two input conditions #####
        #     cond_E_hist = cond['history_E']
        #     if len(cond_E_hist.size()) == 3:
        #         cond_E_hist = cond_E_hist.unsqueeze(1)
        #     cond_encoded_feat_hist = self.cond_input_process_feat_hist(cond_E_hist)
        #     cond_encoded_feat_hist = self.cond_input_positional_encoder_hist(cond_encoded_feat_hist)
        #     cond_encoded_feat_hist = self.cond_input_transformer_encoder_hist(cond_encoded_feat_hist) # get the cond encoded features ##
        #     cond_encoded_feat_hist = cond_encoded_feat_hist[0:1]
        #     expanded_conf_pts_feat_hist = cond_glb_feat.unsqueeze(0).repeat(cond_encoded_feat_hist.size(0), 1, 1)
        #     cat_cond_feat_obj_embedding_hist = torch.cat(
        #         [ expanded_conf_pts_feat_hist, cond_encoded_feat_hist], dim=-1
        #     )
        #     cat_feat_obj_embedding = cat_feat_obj_embedding + cat_cond_feat_obj_embedding_hist
            
            
        
        y_expanded = y.squeeze(-1) # nt x 1 #  
        # print(f"y_expanded: {y_expanded.size()}, ")
        time_embedding = self.time_embedder(y_expanded)
        # print(f"time_embedding: {time_embedding.size()}")
        
        per_point_embedding_with_timesteps = torch.cat(
            [time_embedding, cat_feat_obj_embedding], dim=0
        )
        # perpoint feat output #
        # per_point_feat_output : nn_ts x bsz x latent_dim #
        per_point_feat_output = self.transformer_with_timesteps_encoder_feat(
            per_point_embedding_with_timesteps
        )[1:]
        
        decoded_pts_feat = per_point_feat_output[-1]
        decoded_pts_feat = self.pc_latent_processing(decoded_pts_feat)
        per_point_feat_output = self.feat_latent_processing(per_point_feat_output)
        # decoded_feat = {
        #     'pts_feat': decoded_pts_feat,
        #     'feat_feat': per_point_feat_output
        # }
        
        decoded_feat = utils.PlaceHolder(X=decoded_pts_feat, E=per_point_feat_output, y=y)
        
        return decoded_feat
   



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        # not used in the final model 
        # get the x.shape #
        # but we want it has #
        # pe: 
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)
    
    def forward_batch_selection(self, x, time_index):
        # 
        # self.pe : num_total_indexes x latent_dim # 
        # time_index: bsz x nn_ts # 
        # batched_index_select(values, indices, dim = 1):
        maxx_time_index = torch.max(time_index).item()
        minn_time_index = torch.min(time_index).item() # j
        # print(f"pe: {self.pe.size()}, maxx_time_index: {maxx_time_index}, minn_time_index: {minn_time_index}, ")
        selected_positional_encodings = batched_index_select(self.pe, time_index, dim=0)[:, :, 0] # bsz x nn_ts x latent_dim #
        x = x + selected_positional_encodings.contiguous().transpose(1, 0).contiguous()
        return self.dropout(x)
        # pass


class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)

class TimestepEmbedderV2(nn.Module):
    def __init__(self, latent_dim, max_len=5000):
        super().__init__()
        self.latent_dim = latent_dim
        # self.sequence_pos_encoder = sequence_pos_encoder
        
        pe = torch.zeros(max_len, latent_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.pe[timesteps]) # .permute(1, 0, 2)




# InputProcessObjBaseV7( input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False )
# OutputProcessObjBaseV7
class InputProcessObjBaseV7(nn.Module):
    def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=False): 
        super().__init__()
        # self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        # nnbsz  x nn_ts x nn_pts
        
        self.feats_encoding_net = nn.Sequential(
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim), # 
        )
        
        self.obj_pos_encoding_net = nn.Sequential(
            nn.Linear(3, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.obj_ornt_encoding_net = nn.Sequential(
            nn.Linear(4, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        ### nn_bsz x nn_ts x latent_dim
        self.obj_pts_encoding_net = nn.Sequential(
            nn.Linear(3, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        self.obj_pts_glb_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        nn_final_latent_dim = self.latent_dim * 4
        self.tot_feats_net = nn.Sequential(
            nn.Linear(nn_final_latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        
        self.init_weights()

    def init_weights(self, ):
        for i_module, module in enumerate(self.feats_encoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        
        for i_module, module in enumerate(self.obj_pos_encoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        
        for i_module, module in enumerate(self.obj_ornt_encoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        
        for i_module, module in enumerate(self.obj_pts_encoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
                
        for i_module, module in enumerate(self.obj_pts_glb_encoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
                
        for i_module, module in enumerate(self.tot_feats_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
    
    # x obj ornt; x obj pos #
    def forward(self, x_hand_pos, x_obj_pos, x_obj_ornt, x_pts):
        x_hand_feat = self.feats_encoding_net(x_hand_pos)
        x_obj_pos_feat = self.obj_pos_encoding_net(x_obj_pos)
        x_obj_ornt_feat = self.obj_ornt_encoding_net(x_obj_ornt)
        x_pts_feat = self.obj_pts_encoding_net(x_pts)
        x_glb_pts_feat, _ = torch.max(x_pts_feat, dim=-2) 
        x_glb_pts_feat = self.obj_pts_glb_encoding_net(x_glb_pts_feat) # nn_bsz x nn_ts x nn_feat_dim #
        x_tot_feat = torch.cat(
            [ x_hand_feat, x_obj_pos_feat, x_obj_ornt_feat, x_glb_pts_feat ], dim=-1
        )
        x_tot_feat = self.tot_feats_net(x_tot_feat)
        return x_tot_feat # tot feat #
    
    
    
    
    

class OutputProcessObjBaseV7(nn.Module):
    def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=False): 
        super().__init__()
        # self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        # nnbsz  x nn_ts x nn_pts 
        
        self.hand_pose_decoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, input_feats)
        )
        
        self.obj_pos_decoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 3)
        )
        
        self.obj_ornt_decoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, 4)
        )
        
        
        self.init_weights()

    def init_weights(self, ):
        for i_module, module in enumerate(self.hand_pose_decoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.obj_pos_decoding_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        
        for module in self.obj_ornt_decoding_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        
                    
    def forward(self, x_feat):
        
        x_hand_pose = self.hand_pose_decoding_net(x_feat) 
        x_obj_pos = self.obj_pos_decoding_net(x_feat)
        x_obj_ornt = self.obj_ornt_decoding_net(x_feat)
        
        x_obj_ornt = x_obj_ornt / torch.clamp(torch.norm(x_obj_ornt, p=2, dim=-1, keepdim=True), min=1e-6)
        
        return x_hand_pose, x_obj_pos, x_obj_ornt #d
        
        #
        # x_hand_feat = self.feats_encoding_net(x_hand_pos)
        # x_obj_pos_feat = self.obj_pos_encodin_net(x_obj_pos)
        # x_obj_ornt_feat = self.obj_ornt_encoding_net(x_obj_ornt)
        # x_pts_feat = self.obj_pts_encoding_net(x_pts)
        # x_glb_pts_feat, _ = torch.max(x_pts_feat, dim=-2) 
        # x_glb_pts_feat = self.obj_pts_glb_encoding_net(x_glb_pts_feat) # nn_bsz x nn_ts x nn_feat_dim #
        # x_tot_feat = torch.cat(
        #     [ x_hand_feat, x_obj_pos_feat, x_obj_ornt_feat, x_glb_pts_feat ], dim=-1
        # )
        # x_tot_feat = self.tot_feats_net(x_tot_feat)
        # return x_tot_feat # get the x_tot_feat #
    
    
    
    
  

# InputProcessObjBaseV5( input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False )
class InputProcessObjBaseV5(nn.Module):
    def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=False): 
        super().__init__()
        # self.data_rep = data_rep
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )

        self.without_glb = without_glb
        self.only_with_glb = only_with_glb
        
        self.zero_init = zero_init 
        
        if self.without_glb:
            self.pts_glb_feats_encoding_net = nn.Sequential(
                nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        else:
            self.pts_glb_feats_encoding_net = nn.Sequential(
                nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
                nn.Linear(self.latent_dim, self.latent_dim),
            )
        
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        self.init_weights()

    def init_weights(self, ):
        for i_module, module in enumerate(self.pts_feats_encoding_net):
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.glb_feats_encoding_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for i_module, module in enumerate(self.pts_glb_feats_encoding_net):
            if isinstance(module, nn.Linear):
                if i_module < len(self.pts_glb_feats_encoding_net) - 1 or (not self.zero_init):
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.zeros_(module.weight)
    
    # transformer model #
    def forward(self, x, rt_glb=False, permute_dim=True):
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3) # x.size()
        # bsz, nf, nnb = x.size()[:3]
        
        
        bsz, np, nt, nf = x.size() # bas z x np x nt x nf #
        x_pts_emb = self.pts_feats_encoding_net(
            x
        )
        x_glb_emb, _ = torch.max(x_pts_emb, dim=1, keepdim=True)
        x_glb_emb = self.glb_feats_encoding_net(x_glb_emb)
        x_pts_emb = torch.cat(
            [x_pts_emb, x_glb_emb.repeat(1, np, 1, 1)], dim=-1
        )
        x_pts_emb = self.pts_glb_feats_encoding_net(x_pts_emb)
        if permute_dim:
            x_pts_emb = x_pts_emb.permute(2, 0, 1, 3).contiguous()
            x_pts_emb = x_pts_emb.view(x_pts_emb.size(0), bsz * np, -1).contiguous()
        else:
            x_pts_emb = x_pts_emb[:, 0, :, :].contiguous()
            
        
        # nt x (bsz x np) x latent_dim #
        
        if rt_glb:
            return x_pts_emb, x_glb_emb
        else:
        
            return x_pts_emb
  
  
# give it more conditional features #
# give it more #
# pool to get the global features #
# use the global features as the additional condition to predict the state at each timestep #


# InputProcessObjBaseV5( input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
class InputProcessObjBasePC(nn.Module):
    def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False, zero_init=False): 
        super().__init__()
        # self.data_rep = data_rep #
        self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        self.latent_dim = latent_dim
        
        self.zero_init = zero_init ## whether to zero-init the last layer #
        
        self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
            nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.glb_feats_encoding_net = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim),
        )
        
        self.init_weights() 

    def init_weights(self, ):
        for i_module, module in enumerate(self.pts_feats_encoding_net):
            if isinstance(module, nn.Linear):
                # if i_module < len(self.pts_feats_encoding_net) - 1 or (not self.zero_init):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
                # else:
                #     torch.nn.init.zeros_(module.bias)
                #     torch.nn.init.zeros_(module.weight)
        for module in self.glb_feats_encoding_net:
            if isinstance(module, nn.Linear):
                if i_module < len(self.glb_feats_encoding_net) - 1 or (not self.zero_init):
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.xavier_uniform_(module.weight)
                else:
                    torch.nn.init.zeros_(module.bias)
                    torch.nn.init.zeros_(module.weight)

        
    def forward(self, x, rt_glb=False): # decode relative positions and others ## # cond on the points for points output #
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3) # x.size()
        # bsz, nf, nnb = x.size()[:3]


        # bsz, np, nt, nf = x.size()
        # encoded features; # transformed features -- sequential #
        # x pts embedding # perpoint feature # perpoint feature glb #
        x_pts_emb = self.pts_feats_encoding_net(
            x
        )
        x_glb_emb, _ = torch.max(x_pts_emb, dim=1, keepdim=True)
        x_glb_emb = self.glb_feats_encoding_net(x_glb_emb)
        # x_pts_emb = torch.cat(
        #     [x_pts_emb, x_glb_emb.repeat(1, np, 1, 1)], dim=-1
        # )
        # x_pts_emb = self.pts_glb_feats_encoding_net(x_pts_emb)
        # x_pts_emb = x_pts_emb.permute(2, 0, 1, 3).contiguous()
        # x_pts_emb = x_pts_emb.view(x_pts_emb.size(0), bsz * np, -1).contiguous()
        
        return x_pts_emb, x_glb_emb
        
        
      

# InputProcessObjBaseV5( input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
class InputProcessObjBaseCondsV5(nn.Module):
    # def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False): 
    ### number of the input features: object mass, 
    def __init__(self, input_feat_dim=5, latent_dim=128): 
        super().__init__()
        # self.data_rep = data_rep
        self.input_feat_dim = input_feat_dim # 
        self.latent_dim = latent_dim
        
        self.cond_feat_encoding_net = nn.Sequential(
            nn.Linear(self.input_feat_dim, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim), 
        )
        
        self.init_weights() 
        
        # initilaize the bais to zeros
        
        
        # self.input_feats = input_feats # 21 * 3 + 3 + 3 --> for each joint + 3 pos + 3 normals #
        # self.latent_dim = latent_dim
        
        # self.pts_feats_encoding_net = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
        # self.glb_feats_encoding_net = nn.Sequential(
        #     nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )

        # self.without_glb = without_glb
        # self.only_with_glb = only_with_glb
        
        # if self.without_glb:
        #     self.pts_glb_feats_encoding_net = nn.Sequential(
        #         nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(), 
        #         nn.Linear(self.latent_dim, self.latent_dim),
        #     )
        # else:
        #     self.pts_glb_feats_encoding_net = nn.Sequential(
        #         nn.Linear(self.latent_dim * 2, self.latent_dim), nn.ReLU(), 
        #         nn.Linear(self.latent_dim, self.latent_dim),
        #     )
        
        # self.embedding_pn_blk = nn.Sequential( # nnb --> 21
        #     nn.Linear(input_feats, self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim, self.latent_dim),
        # )
        
    def init_weights(self, ):
        for module in self.cond_feat_encoding_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.zeros_(module.weight) ## nit w### TODO: check whether it is a good strategy #
        
    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3) # x.size()
        # bsz, nf, nnb = x.size()[:3]
        
        # x: bsz x nn_features
        # x_feat: bsz x nn_out_features
        
        x_feat = self.cond_feat_encoding_net(x) ## 
        
        # # # why separate them? # #
        # bsz, np, nt, nf = x.size()
        # # 
        
        # x_pts_emb = self.pts_feats_encoding_net(
        #     x
        # )
        # x_glb_emb, _ = torch.max(x_pts_emb, dim=1, keepdim=True)
        # x_glb_emb = self.glb_feats_encoding_net(x_glb_emb)
        # x_pts_emb = torch.cat(
        #     [x_pts_emb, x_glb_emb.repeat(1, np, 1, 1)], dim=-1
        # )
        # x_pts_emb = self.pts_glb_feats_encoding_net(x_pts_emb)
        # x_pts_emb = x_pts_emb.permute(2, 0, 1, 3).contiguous()
        # x_pts_emb = x_pts_emb.view(x_pts_emb.size(0), bsz * np, -1).contiguous()
        
        return x_feat



# InputProcessObjBaseV5( input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
class InputProcessObjBaseCondsV6(nn.Module):
    # def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False): 
    ### number of the input features: object mass,  # nn ws x xxx 
    def __init__(self, input_feat_dim=5, latent_dim=128): 
        super().__init__()
        # self.data_rep = data_rep
        self.input_feat_dim = input_feat_dim # 
        self.latent_dim = latent_dim
        self.nn_obj_type = 2 ## 
        
        #### 1) encode the object type indicator into the latent vector ####
        #### 2) with the encoded factor and the translated fator
        self.obj_type_embedding_layer = nn.Embedding(
            num_embeddings=self.nn_obj_type * 2, embedding_dim=self.latent_dim
        )
        
        
        self.cond_feat_encoding_net = nn.Sequential(
            nn.Linear(self.input_feat_dim - 1, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim), 
        )
        
        self.cond_feat_processing_net =  nn.Sequential(
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim + self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim) ## get the latent dim ##
        )
        
        self.init_weights() 
        
        
    def init_weights(self, ):
        ### only the last layer needs to bbe zero ##
        for module in self.cond_feat_encoding_net:
            if isinstance(module, nn.Linear):
                # torch.nn.init.zeros_(module.bias)
                # torch.nn.init.zeros_(module.weight) ## nit w### TODO: check whether it is a good strategy #
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.cond_feat_processing_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.zeros_(module.weight) ## bias and the weight ## 
        
    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3) # x.size()
        # bsz, nf, nnb = x.size()[:3]
        
        # x: bsz x nn_features
        # x_feat: bsz x nn_out_features
        
        # object type embedding # # 
        x_obj_type_indicator = x[..., 0].long()
        x_obj_type_embedding = self.obj_type_embedding_layer(x_obj_type_indicator)
        
        
        x_feat = self.cond_feat_encoding_net(x[..., 1:]) ## 
        x_feat = torch.cat(
            [ x_obj_type_embedding, x_feat ], dim=-1
        )
        
        # get the xfeat #
        x_feat = self.cond_feat_processing_net(x_feat)
        
        
        return x_feat
        



# InputProcessObjBaseV5( input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False)
class InputProcessObjBaseCondsV7(nn.Module):
    # def __init__(self, input_feats, latent_dim, layernorm=True, without_glb=False, only_with_glb=False): 
    ### number of the input features: object mass,  # nn ws x xxx 
    def __init__(self,  hand_qs_input_dim=22, obj_tarns_input_dim=3, obj_ornt_input_dim=4, latent_dim=256): 
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hand_qs_input_dim  = hand_qs_input_dim
        self.obj_trans_input_dim = obj_tarns_input_dim
        self.obj_ornt_input_dim = obj_ornt_input_dim
        
        self.obj_pc_input_processing_layer = nn.Sequential(
            nn.Linear(3, self.latent_dim), nn.ReLU(), 
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        ### glb embedding is a latent-dim vecotr ##
        self.obj_pc_glb_processing_layer = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )
        
        self.hand_qs_encoding_net = nn.Sequential(
            nn.Linear(self.hand_qs_input_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim) 
        )
        self.obj_trans_encoding_net = nn.Sequential(
            nn.Linear(self.obj_trans_input_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim) 
        )
        self.obj_ornt_encoding_net = nn.Sequential(
            nn.Linear(self.obj_ornt_input_dim, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim) 
        )
        # 
        self.cond_feat_processing_net =  nn.Sequential(
            nn.Linear(self.latent_dim * 4, self.latent_dim), nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim) ## get the latent dim ## 
        )
        
        ## get the hand qs, obj trans, obj ornt input networks ##
        # self.pc_encoded_feat + self.hand_qs_encoded_feat + self.obj_trans_encoded_feat + self.obj_ornt_encoded_feat # 
        
        # # self.data_rep = data_rep
        # self.input_feat_dim = input_feat_dim # 
        # self.latent_dim = latent_dim
        # self.nn_obj_type = 2 ## 
        
        # #### 1) encode the object type indicator into the latent vector ####
        # #### 2) with the encoded factor and the translated fator
        # self.obj_type_embedding_layer = nn.Embedding(
        #     num_embeddings=self.nn_obj_type * 2, embedding_dim=self.latent_dim
        # )
        
        
        # self.cond_feat_encoding_net = nn.Sequential(
        #     nn.Linear(self.input_feat_dim - 1, self.latent_dim), nn.ReLU(), 
        #     nn.Linear(self.latent_dim, self.latent_dim), 
        # )
        
        # self.cond_feat_processing_net =  nn.Sequential(
        #     nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim + self.latent_dim), nn.ReLU(),
        #     nn.Linear(self.latent_dim + self.latent_dim, self.latent_dim) ## get the latent dim ##
        # )
        
        self.init_weights() 
        
        
    def init_weights(self, ):
        for module in self.obj_pc_input_processing_layer:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.obj_pc_glb_processing_layer:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.hand_qs_encoding_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.obj_trans_encoding_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.obj_ornt_encoding_net: # obj ornt #
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.xavier_uniform_(module.weight)
        for module in self.cond_feat_processing_net:
            if isinstance(module, nn.Linear):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.zeros_(module.weight)
        
        
        
    def forward(self, x): # decode relative positions and others ##
        # bs, nframes, njoints, nfeats = x.shape #
        # bsz x nf x nnj x (3 + nnb x (3 + 3))  # bsz x nf x nnb x (latent_dim)
        # x: bsz x nf x nnb x (3 + 3 + 21 * 3) # x.size()
        # bsz, nf, nnb = x.size()[:3]
        
        # x: bsz x nn_features
        # x_feat: bsz x nn_out_features
        
        # x: bsz x ws x nn_feaures --- 
        
        x_hand_qs = x[..., : self.hand_qs_input_dim]
        x_obj_trans = x[..., self.hand_qs_input_dim: self.hand_qs_input_dim + self.obj_trans_input_dim]
        x_obj_ornt = x[..., self.hand_qs_input_dim + self.obj_trans_input_dim: self.hand_qs_input_dim + self.obj_trans_input_dim + self.obj_ornt_input_dim]
        x_obj_pts = x[..., self.hand_qs_input_dim + self.obj_trans_input_dim + self.obj_ornt_input_dim:  ] # 

        # 
        x_obj_pts = x_obj_pts[:, 0, : ] # get the first frame obj pts # 
        x_obj_pts = x_obj_pts.view(x_obj_pts.size(0), -1) # get the obj pts # # 
        # print(f"x_obj_pts: {x_obj_pts.size()}")
        x_obj_pts = x_obj_pts.view(x_obj_pts.size(0), -1, 3).contiguous() # get the obj pts contiguous # 
        # xobj pts 
        
        x_obj_pts_embedding = self.obj_pc_input_processing_layer(x_obj_pts) ## bsz x nn_ws x latent_dim
        x_obj_pts_embedding, _ = torch.max(x_obj_pts_embedding, dim=1) #
        x_obj_pts_embedding = self.obj_pc_glb_processing_layer(x_obj_pts_embedding) # obj ps embeddings # bsz x nn_ws x latent_dim  # 
        x_obj_pts_embedding = x_obj_pts_embedding.unsqueeze(1).repeat(1, x_hand_qs.size(1), 1) # bsz x nn_ws x latent_dim
        
        x_hand_qs_embedding = self.hand_qs_encoding_net(x_hand_qs) # hand qs embeddings # bsz x nn_ws x latent_dim
        x_obj_trans_embedding = self.obj_trans_encoding_net(x_obj_trans)
        x_obj_ornt_embedding = self.obj_ornt_encoding_net(x_obj_ornt) # ge the obj trans and the ornt # 
        
        ## x_obj_pts_embedding: bsz x nn_ws x latnet_dim 3 
        # print(f"")
        x_tot_embedding = torch.cat(
            [x_hand_qs_embedding, x_obj_trans_embedding, x_obj_ornt_embedding, x_obj_pts_embedding], dim=-1
        )
        cond_embedding = self.cond_feat_processing_net(
            x_tot_embedding
        )
        # x
        
        # object type embedding # # 
        # x_obj_type_indicator = x[..., 0].long()
        # x_obj_type_embedding = self.obj_type_embedding_layer(x_obj_type_indicator)
        
        
        # x_feat = self.cond_feat_encoding_net(x[..., 1:]) ## 
        # x_feat = torch.cat(
        #     [ x_obj_type_embedding, x_feat ], dim=-1
        # ) 
        
        # ws x bsz x xx #
        
        # # get the xfeat #
        # x_feat = self.cond_feat_processing_net(x_feat)
        
        
        return cond_embedding
        
       
  

class OutputProcessObjBaseRawV5(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # self.data_rep = data_rep
        self.latent_dim = latent_dim
        # self.njoints = njoints
        # self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        # self.v5_out_not_cond_base = v5_out_not_cond_base
        
        # if self.not_cond_base:
        self.rel_dec_cond_dim = self.latent_dim
        self.dist_dec_cond_dim = self.latent_dim
        # else:
        #     self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3
        #     self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        # self.use_anchors = use_anchors
        # self.nn_keypoints = nn_keypoints
        # if self.use_anchors:
        #     self.nn_keypoints = 
        
        # self.rel_dec_blk = nn.Sequential(
        #     nn.Linear(self.rel_dec_cond_dim,  3,),
        # )
        
        # self.out_objbase_v5_bundle_out = out_objbase_v5_bundle_out
        
        # if self.out_objbase_v5_bundle_out:
        #     if self.v5_out_not_cond_base:
        #         self.rel_dec_blk = nn.Sequential(
        #             nn.Linear(self.latent_dim,  self.latent_dim // 2), nn.ReLU(),
        #             nn.Linear(self.latent_dim // 2, self.nn_keypoints * 3),
        #         )
        #     else:
        #         self.rel_dec_blk = nn.Sequential(
        #             nn.Linear(self.latent_dim + 3 + 3,  self.latent_dim // 2), nn.ReLU(),
        #             nn.Linear(self.latent_dim // 2, self.nn_keypoints * 3),
        #         )
        # else:
        self.output_layer = nn.Sequential(
            nn.Linear(self.rel_dec_cond_dim,  output_dim),
        )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # # )
        # self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        # self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
        #   self.dist_dec_cond_dim, 1 * self.nn_keypoints
        # )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, input_x=None): #  
        # nframes, bs, d = output.shape
        
        # # bsz, nframes, nnj, nnb = x['rel_base_pts_to_rhand_joints'].shape[:4] # pert_rel_base_pts_to_rhand_joints
        # bsz, nframes, nnj, nnb = x['pert_rel_base_pts_to_rhand_joints'].shape[:4] # bsz x nf x nnj x nnb x 3  # nf x nnb x 3 --> noisy input for denoised values #
        # # forward the samole # base_pts, base_normals, # 
        # # base_pts = x['base_pts'] # bsz x nnb x 3
        # base_pts = x['normed_base_pts'] # bsz x nnb x 3
        # base_normals = x['base_normals'] # bsz x nnb x 3
        # # rel_base_pts_to_rhand_joints = x['rel_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb x 3
        # # dist_base_pts_to_rhand_joints = x['dist_base_pts_to_rhand_joints'] # bsz x ws x nnj x nnb
        # ## 
        # output: bsz x nf x nnj x latent_dim
        
        if input_x is None:
            nb = output.size(1)
            np = 1
            nt = output.size(0)
            nf = output.size(-1)
        else:
            nb, np, nt, nf = input_x.size()[:4]
        
        # nt x nb x np x feat_dim #
        # 
        output = output.view(nt, nb, np, -1) # nframes x bsz x nnb x latent_dim 
        output = output.permute(1, 2, 0, 3) # nb x np x nt x latent_dim 
        
        # if self.out_objbase_v5_bundle_out:
        #     if self.v5_out_not_cond_base:
        #         output_exp = output
        #     else: # otuptu_exp for rel_dec_blk
        #         base_pts_exp = base_pts.unsqueeze(1).repeat(1, nframes, 1, 1)
        #         base_normals_exp = base_normals.unsqueeze(1).repeat(1, nframes, 1, 1)
        #         output_exp = torch.cat( # with input noisy data # ############### denoised latents for each base pts ###
        #             [output, base_pts_exp, base_normals_exp], dim=-1
        #         )
        #     dec_rel = self.rel_dec_blk(output_exp)
        #     dec_rel = dec_rel.view(bsz, nframes, nnb, nnj, 3).permute(0, 1, 3, 2, 4).contiguous()
        # else:
        # output = output.permute(1, 0, 2)
        # output = output.view(bsz, nframes, nnj, -1).contiguous() # bsz x nf x nnj x (decoded_latent_dim) # 
        # output = output.unsqueeze(2).repeat(1, 1, nnj, 1, 1).contiguous()
        # bsz x nnframes x d #  # 
        # output = output.permute(1, 0, 2).contiguous().unsqueeze(2).unsqueeze(2).repeat(1, 1, nnj, nnb, 1).contiguous()
        # base_pts_exp = base_pts.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # base_normals_exp = base_normals.unsqueeze(1).unsqueeze(1).repeat(1, nframes, nnj, 1, 1)
        # bsz x nnframes x nnb x (d + 3 + 3) # --> base normals ##
        
        # if self.not_cond_base:
        #     output_exp = output
        # else:
        # output_exp = torch.cat( # with input noisy data
        #     [output, base_pts_exp, base_normals_exp, x['pert_rel_base_pts_to_rhand_joints']], dim=-1
        # )
        out = self.output_layer(output) #  nb x np x nt x outputd_dim 
        # dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # decoded rel, decoded distances #
        # out = {
        #   'dec_rel': dec_rel,
        # #   'dec_dist': dec_dist.squeeze(-1),
        # }
        return out ## output




class OutputProcessObjBaseRawPC(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # self.data_rep = data_rep 
        self.latent_dim = latent_dim
        # self.njoints = njoints
        # self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        # self.v5_out_not_cond_base = v5_out_not_cond_base
        
        # if self.not_cond_base:
        self.rel_dec_cond_dim = self.latent_dim + 3 # + self.latent_dim // 2
        # self.dist_dec_cond_dim = self.latent_dim
        # else:
        #     self.rel_dec_cond_dim = self.latent_dim + 3 + 3 + 3
        #     self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        
        # self.use_anchors = use_anchors
        # self.nn_keypoints = nn_keypoints
        # if self.use_anchors:
        #     self.nn_keypoints = 
        
        # self.rel_dec_blk = nn.Sequential(
        #     nn.Linear(self.rel_dec_cond_dim,  3,),
        # )
        
        # self.out_objbase_v5_bundle_out = out_objbase_v5_bundle_out
        
        # if self.out_objbase_v5_bundle_out:
        #     if self.v5_out_not_cond_base:
        #         self.rel_dec_blk = nn.Sequential(
        #             nn.Linear(self.latent_dim,  self.latent_dim // 2), nn.ReLU(),
        #             nn.Linear(self.latent_dim // 2, self.nn_keypoints * 3),
        #         )
        #     else:
        #         self.rel_dec_blk = nn.Sequential(
        #             nn.Linear(self.latent_dim + 3 + 3,  self.latent_dim // 2), nn.ReLU(),
        #             nn.Linear(self.latent_dim // 2, self.nn_keypoints * 3),
        #         )
        # else:
        self.output_layer = nn.Sequential(
            # nn.Linear(self.rel_dec_cond_dim,  output_dim),
            nn.Linear(self.rel_dec_cond_dim,  self.rel_dec_cond_dim), nn.ReLU(),
            nn.Linear(self.rel_dec_cond_dim,  self.rel_dec_cond_dim), nn.ReLU(),
            nn.Linear(self.rel_dec_cond_dim,  output_dim),
        )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # # )
        # self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        # self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
        #   self.dist_dec_cond_dim, 1 * self.nn_keypoints
        # )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, input_x, input_feat, input_x_feat_glb, input_x_feat_pts): #  
        # nframes, bs, d = output.shape
        
        # input x # 
        # output: bsz x ws x feature #
        
        nb, np, nt, nf = input_feat.size()[:4]
        
        output = output.view(nt, nb, np, -1) # nframes x bsz x nnb x latent_dim 
        output = output.permute(1, 2, 0, 3) # nb x np x nt x latent_dim 
        
        # bsz x 1 x feat_di
        
        input_x_feat_glb_expanded = input_x_feat_glb.repeat(1, input_x.size(1), 1)
        
        output = output[:, :, 0, :] 
        # input_x: bsz x nn_pts x 3 
        output = output.contiguous().repeat(1, input_x.size(1), 1)
        
        # cat_x = torch.cat(
        #     [input_x, output, input_x_feat_glb_expanded], dim=-1
        # )
        
        cat_x = torch.cat(
            [input_x, input_x_feat_pts, input_x_feat_glb_expanded], dim=-1
        )
        
        out = self.output_layer(cat_x) #  nb x np x nt x outputd_dim 
        # dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # decoded rel, decoded distances #
        # out = {
        #   'dec_rel': dec_rel,
        # #   'dec_dist': dec_dist.squeeze(-1),
        # }
        return out ## output
   



class OutputProcessObjBaseRawPC_V2(nn.Module):
    def __init__(self, latent_dim, nn_pts=128):
        super().__init__()
        # self.data_rep = data_rep 
        self.latent_dim = latent_dim
        # self.njoints = njoints
        # self.not_cond_base = not_cond_base ## not cond base ##
        # self.nfeats = nfeats # dec cond on latent code and base pts, base normals #
        
        # self.v5_out_not_cond_base = v5_out_not_cond_base
        
        # if self.not_cond_base:
        self.rel_dec_cond_dim = self.latent_dim # + 3 # + self.latent_dim // 2
        
        self.dec_output_dim = nn_pts * 3
        self.nn_pts = nn_pts
        
        # 
        self.output_layer = nn.Sequential(
            nn.Linear(self.rel_dec_cond_dim,  self.rel_dec_cond_dim), nn.ReLU(),
            nn.Linear(self.rel_dec_cond_dim,  self.rel_dec_cond_dim), nn.ReLU(),
            nn.Linear(self.rel_dec_cond_dim,  self.dec_output_dim),
        )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # # )
        # self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        # self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
        #   self.dist_dec_cond_dim, 1 * self.nn_keypoints
        # )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output_feat): 
        # nframes, bs, d = output.shape
        
        # input x # 
        # output: bsz x ws x feature #
        
        output_pts = self.output_layer(output_feat)
        output_pts = output_pts.view(output_pts.size(0), self.nn_pts, 3)
        return output_pts


class OutputProcessObjBaseRawV5_V2(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # self.data_rep = data_rep
        self.latent_dim = latent_dim
        
        
        self.rel_dec_cond_dim = self.latent_dim
        self.dist_dec_cond_dim = self.latent_dim
        
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.rel_dec_cond_dim,  output_dim),
        )
        
        
    def forward(self, output): #  
        # nframes, bs, d = output.shape
        
        # nt x nb x np x latnet_dim # ---- necessary to get the input x here # only to provide the shape information #
        # nb, np, nt, nf = input_x.size()[:4]
        
        nt = output.size(0)
        nb = output.size(1)
        np = 1
        
        output = output.view(nt, nb, np, -1) # nframes x bsz x nnb x latent_dim 
        output = output.permute(1, 2, 0, 3) # nb x np x nt x latent_dim 
        
        
        out = self.output_layer(output) #  nb x np x nt x outputd_dim 
        # dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        
        return out




#### output process V8 ####
class OutputProcessObjBaseRawV8(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super().__init__()
        # self.data_rep = data_rep
        self.latent_dim = latent_dim
        
        
        
        self.rel_dec_cond_dim = self.latent_dim
        self.dist_dec_cond_dim = self.latent_dim
        
        
        
        self.output_layer = nn.Sequential(
            nn.Linear(self.rel_dec_cond_dim,  output_dim),
        )
        
        # self.rel_dec_blk = nn.Linear( # rel_dec_blk -> output relative positions #
        #   self.rel_dec_cond_dim, 3 * 21
        # # )
        # self.dist_dec_cond_dim = self.latent_dim + 3 + 3
        # self.dist_dec_blk = nn.Linear( # dist_dec_blk -> output relative distances #
        #   self.dist_dec_cond_dim, 1 * self.nn_keypoints
        # )
        # self.poseFinal = nn.Linear(self.latent_dim, self.input_feats)
        # if self.data_rep == 'rot_vel':
        #     self.velFinal = nn.Linear(self.latent_dim, self.input_feats)

    def forward(self, output, input_x=None):
        # nframes, bs, d = output.shape #
        
        ## output: nn_bsz x nn_latent_dim ##
        ## output: nn_bsz x nn_latent_dim ##
        ## nn_bsz x nn_hand_actions ##
        decoded_feats = self.output_layer(output) 
        
        return decoded_feats
        
        
        # if input_x is None:
        #     nb = output.size(1)
        #     np = 1
        #     nt = output.size(0)
        #     nf = output.size(-1)
        # else:
        #     nb, np, nt, nf = input_x.size()[:4]
        
        # # nt x nb x np x feat_dim #
        # # 
        # output = output.view(nt, nb, np, -1) # nframes x bsz x nnb x latent_dim 
        # output = output.permute(1, 2, 0, 3) # nb x np x nt x latent_dim 
        
        
        
        # out = self.output_layer(output) #  nb x np x nt x outputd_dim 
        # # dec_rel = dec_rel.contiguous().view(bsz, nframes, nnj, nnb, 3).contiguous() # bsz x nnframes x nnb x nnj x 3 #
        
        # # decoded rel, decoded distances #
        # # out = {
        # #   'dec_rel': dec_rel,
        # # #   'dec_dist': dec_dist.squeeze(-1),
        # # }
        # return out ## output

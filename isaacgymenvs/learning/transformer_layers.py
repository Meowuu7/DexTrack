
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

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
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

class InputProcess(nn.Module):
    def __init__(self, nn_latents=256):
        super().__init__()
        # input is nn_bsz x (flatten_dim), where flatten_dim = history_length x per_history_feature
        self.hand_qpos_in_dim = 23
        self.obj_pos_in_dim = 3
        self.obj_ornt_in_dim = 4
        
        self.hand_qpos_processing_layers = nn.Sequential(
            nn.Linear(self.hand_qpos_in_dim, nn_latents // 2), nn.ReLU(),
            nn.Linear(nn_latents // 2, nn_latents)
        )
        self.obj_pos_processing_layers = nn.Sequential(
            nn.Linear(self.obj_pos_in_dim, nn_latents // 2), nn.ReLU(),
            nn.Linear(nn_latents // 2, nn_latents),
        )
        self.obj_ornt_processing_layers = nn.Sequential(
            nn.Linear(self.obj_ornt_in_dim, nn_latents // 2), nn.ReLU(),
            nn.Linear(nn_latents // 2, nn_latents)
        )
        
        self.cat_feats_processing_layers = nn.Sequential(
            nn.Linear(3 * nn_latents, 2 * nn_latents), nn.ReLU(),
            nn.Linear(2 * nn_latents, nn_latents), 
        )
        
        self._init_models() # 
        
    def _init_models(self, ):
        for layer in self.hand_qpos_processing_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
        for layer in self.obj_pos_processing_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        for layer in self.obj_ornt_processing_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
        for layer in self.cat_feats_processing_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
    
    
    def forward(self, in_feats, nn_history=20):
        expanded_in_feats = in_feats.contiguous().view(in_feats.size(0), nn_history, -1).contiguous()
        hand_qpos, obj_pos, obj_ornt = expanded_in_feats[..., : self.hand_qpos_in_dim], expanded_in_feats[..., self.hand_qpos_in_dim: self.hand_qpos_in_dim + self.obj_pos_in_dim], expanded_in_feats[..., self.hand_qpos_in_dim + self.obj_pos_in_dim: ]
        
        hand_qpos_feats, obj_pos_feats, obj_ornt_feats = self.hand_qpos_processing_layers(hand_qpos), self.obj_pos_processing_layers(obj_pos), self.obj_ornt_processing_layers(obj_ornt)
        hand_obj_cat_feats = torch.cat(
            [ hand_qpos_feats, obj_pos_feats, obj_ornt_feats ], dim=-1
        )
        cat_feats = self.cat_feats_processing_layers(hand_obj_cat_feats)
        return cat_feats


class InputProcessV2(nn.Module):
    def __init__(self, nn_latents=256):
        super().__init__()
        # input is nn_bsz x (flatten_dim), where flatten_dim = history_length x per_history_feature
        self.hand_qpos_in_dim = 23
        self.obj_pos_in_dim = 3
        self.obj_ornt_in_dim = 4
        
        self.in_feat_dim = self.hand_qpos_in_dim + self.obj_pos_in_dim + self.obj_ornt_in_dim
        
        # self.hand_qpos_processing_layers = nn.Sequential(
        #     nn.Linear(self.hand_qpos_in_dim, nn_latents // 2), nn.ReLU(),
        #     nn.Linear(nn_latents // 2, nn_latents)
        # )
        # self.obj_pos_processing_layers = nn.Sequential(
        #     nn.Linear(self.obj_pos_in_dim, nn_latents // 2), nn.ReLU(),
        #     nn.Linear(nn_latents // 2, nn_latents),
        # )
        # self.obj_ornt_processing_layers = nn.Sequential(
        #     nn.Linear(self.obj_ornt_in_dim, nn_latents // 2), nn.ReLU(),
        #     nn.Linear(nn_latents // 2, nn_latents)
        # )
        
        self.feats_processing_layers = nn.Sequential(
            nn.Linear(self.in_feat_dim, nn_latents // 2), nn.ReLU(),
            nn.Linear(nn_latents // 2, nn_latents), 
        )
        
        self._init_models() # 
        
    def _init_models(self, ):
        for layer in self.feats_processing_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
                
        # for layer in self.obj_pos_processing_layers:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_uniform_(layer.weight)
        #         torch.nn.init.zeros_(layer.bias)
        
        # for layer in self.obj_ornt_processing_layers:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_uniform_(layer.weight)
        #         torch.nn.init.zeros_(layer.bias)
        
        # for layer in self.cat_feats_processing_layers:
        #     if isinstance(layer, nn.Linear):
        #         torch.nn.init.xavier_uniform_(layer.weight)
        #         torch.nn.init.zeros_(layer.bias)
    
    
    def forward(self, in_feats, nn_history=20):
        expanded_in_feats = in_feats.contiguous().view(in_feats.size(0), nn_history, -1).contiguous()
        cat_feats = self.feats_processing_layers(expanded_in_feats)
        
        # expanded_in_feats = in_feats.contiguous().view(in_feats.size(0), nn_history, -1).contiguous()
        # hand_qpos, obj_pos, obj_ornt = expanded_in_feats[..., : self.hand_qpos_in_dim], expanded_in_feats[..., self.hand_qpos_in_dim: self.hand_qpos_in_dim + self.obj_pos_in_dim], expanded_in_feats[..., self.hand_qpos_in_dim + self.obj_pos_in_dim: ]
        
        # hand_qpos_feats, obj_pos_feats, obj_ornt_feats = self.hand_qpos_processing_layers(hand_qpos), self.obj_pos_processing_layers(obj_pos), self.obj_ornt_processing_layers(obj_ornt)
        # hand_obj_cat_feats = torch.cat(
        #     [ hand_qpos_feats, obj_pos_feats, obj_ornt_feats ], dim=-1
        # )
        # cat_feats = self.cat_feats_processing_layers(hand_obj_cat_feats)
        return cat_feats


class OutputProcess(nn.Module):
    def __init__(self, nn_latents=256):
        super().__init__()
        # input is nn_bsz x (flatten_dim), where flatten_dim = history_length x per_history_feature
        self.hand_qpos_in_dim = 23
        
        self.output_layers = nn.Sequential(
            nn.Linear(nn_latents, nn_latents // 2), nn.ReLU(), 
            nn.Linear(nn_latents // 2, self.hand_qpos_in_dim)
        )
        
        self._init_models() # 
    
    def _init_models(self, ):
        for layer in self.output_layers:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                torch.nn.init.zeros_(layer.bias)
        
    def forward(self, in_feats):
        out_feats = self.output_layers(in_feats)
        return out_feats


class TransformerFeatureProcessing(nn.Module):
    def __init__(self, nn_latents=256, dropout=0.0):
        super().__init__()
        self.dropout = dropout
        self.num_heads = 1
        self.ff_size = nn_latents # 256 # 
        # self.ff_size = 2048
        self.ff_size = nn_latents
        self.activation = 'relu'
        self.num_layers = 4
        normalize_before =  False
        # self.input_process = InputProcess(nn_latents=nn_latents)
        
        self.input_process = InputProcessV2(nn_latents=nn_latents)
        
        self.max_len = 25
        
        self.positional_encoding = PositionalEncoding(nn_latents, self.dropout, max_len=self.max_len)
        
        in_feat_seq_transformer_layers = nn.TransformerEncoderLayer(d_model=nn_latents,
                                                            nhead=self.num_heads,
                                                            dim_feedforward=self.ff_size,
                                                            dropout=self.dropout,
                                                            activation=self.activation)
        self.in_feat_transformer_encoder = nn.TransformerEncoder(in_feat_seq_transformer_layers,
                                                        num_layers=self.num_layers)
        
        # # Transformer # #
        # self.real_basejtsrel_to_joints_embed_timestep = TimestepEmbedder(self.latent_dim, self.real_basejtsrel_to_joints_sequence_pos_encoder) # 
        
        # # # # # # if mlp is ok # # then there is no need to use other architectures #
        
        # self.real_basejtsrel_to_joints_sequence_pos_denoising_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        
        self.out_positional_encoding = PositionalEncoding(nn_latents, self.dropout, max_len=self.max_len)
        self.out_transformer_layers = nn.TransformerEncoderLayer(d_model=nn_latents,
                                                                nhead=self.num_heads,
                                                                dim_feedforward=self.ff_size,
                                                                dropout=self.dropout,
                                                                activation=self.activation)
        self.out_transformer_decoder = nn.TransformerEncoder(self.out_transformer_layers,
                                                        num_layers=self.num_layers)
        
        self.output_process = OutputProcess(nn_latents=nn_latents)
        
        
        self.feature_output_layers = nn.Sequential(
            nn.Linear(nn_latents, nn_latents), nn.ReLU(),
            nn.Linear(nn_latents, 230)
        )
        
        self.process_net = nn.Sequential(
            nn.Linear(600, 8192), nn.ReLU(),
            nn.Linear(8192, 4096), nn.ReLU(),
            nn.Linear(4096, 2048), nn.ReLU(),
            nn.Linear(2048, 1024), nn.ReLU(),
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 230),
        )
        
    def forward(self, in_feats, nn_history=20, nn_future=10):
        
        # [1] processed_in_feats: torch.Size([30, 20, 512])
        # [2] processed_in_feats: torch.Size([20, 30, 512])
        # [3] processed_in_feats: torch.Size([20, 30, 512])
        
        
        # output_feats = self.process_net(in_feats)
        # return output_feats
        
        processed_in_feats = self.input_process(in_feats, nn_history=nn_history)
        # print(f"[1] processed_in_feats: {processed_in_feats.size()}")
        processed_in_feats = processed_in_feats.contiguous().transpose(1, 0).contiguous()
        # print(f"[2] processed_in_feats: {processed_in_feats.size()}")
        # processed_in_feats = self.positional_encoding(processed_in_feats)
        processed_in_feats = self.in_feat_transformer_encoder(processed_in_feats)
        # print(f"[3] processed_in_feats: {processed_in_feats.size()}")
        # processed_in_feats = processed_in_feats[-1:] # 1 x nn_bsz x nn_latnets
        
        processed_in_feats = torch.mean(processed_in_feats, dim=0, keepdim=True)
        
        output_feats = self.feature_output_layers(processed_in_feats[0])
        
        
        
        
        # processed_in_feats = processed_in_feats.repeat(nn_future, 1, 1).contiguous()
        # processed_in_feats = self.out_positional_encoding(processed_in_feats)
        # processed_in_feats = self.out_transformer_decoder(processed_in_feats) # nn_future x nn_bsz x nn_latents
        
        # processed_in_feats = processed_in_feats.contiguous().transpose(1, 0).contiguous() # nn_bsz x nn_future x nn_latnets
        # output_feats = self.output_process(processed_in_feats)
        
        # output_feats = output_feats.contiguous().view(output_feats.size(0), -1).contiguous()
        # # print(f"output_feats: {output_feats.size()}")
        
        
        return output_feats






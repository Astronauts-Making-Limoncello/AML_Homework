import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention
from SpatioTemporalSelfAttention import SpatioTemporalSelfAttention
from MLP import MLP
from utils.init_layer import conv_init, bn_init, ln_init, fc_init

from rich import print

# taken as is from CVPR'23 reference paper https://github.com/zhenhuat/STCFormer
class SpatioTemporalDecoderBlock(nn.Module):
  
    def __init__(
        self, 
        in_features: int, out_features: int, num_joints: int,
        num_frames: int, num_frames_out: int,
        num_heads: int, use_skip_connection: bool,
        skip_connection_weight: float, 
        dropout: float
    ):
        super().__init__()

        self.num_frames_out = num_frames_out

        self.use_skip_connection = use_skip_connection
        self.skip_connection_weight = skip_connection_weight

        self.spatio_temporal_self_attention = SpatioTemporalSelfAttention(
            in_features=in_features, out_features=in_features, num_joints=num_joints,
            num_frames=num_frames, num_frames_out=num_frames_out, num_heads=num_heads
        )

        self.dropout_1 = nn.Dropout(p=dropout)

        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.spatio_temporal_cross_attention = SpatioTemporalCrossAttention(
            in_features=in_features, out_features=in_features, num_joints=num_joints,
            num_frames=num_frames, num_frames_out=num_frames_out,
            num_heads=num_heads
        )

        self.dropout_2 = nn.Dropout(p=dropout)

        self.layer_norm_2 = nn.LayerNorm(in_features)

        self.mlp = MLP(in_features, out_features)

        self.layer_norm_3 = nn.LayerNorm(out_features)

        self.dropout_3 = nn.Dropout(p=dropout)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.LayerNorm):
                ln_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(
        self, x: Tensor, memory: Tensor, 
        mask_s_self_attn: Tensor, mask_t_self_attn: Tensor,
        mask_s_cross_attn: Tensor, mask_t_cross_attn: Tensor
    ):
        #  decoder_input.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        # encoder_output.shape: batch_size, temporal_dim, spatial_dim, feature_dim

        x_ = self.layer_norm_1(x)
        # x = x + self.dropout_1(self.spatio_temporal_self_attention(x_, x_, x_, mask_s_self_attn, mask_t_self_attn))
        x = self.dropout_1(self.spatio_temporal_self_attention(x_, x_, x_, mask_s_self_attn, mask_t_self_attn))
        x_ = self.layer_norm_2(x)
        # x = x + self.dropout_2(self.spatio_temporal_cross_attention(x_, memory, memory, mask_s_cross_attn, mask_t_cross_attn))
        x = self.dropout_2(self.spatio_temporal_cross_attention(x_, memory, memory, mask_s_cross_attn, mask_t_cross_attn))
        x_ = self.layer_norm_3(x)
        # x = x + self.dropout_3(self.mlp(x_))
        x = self.dropout_3(self.mlp(x_))
        
        return x
    
    





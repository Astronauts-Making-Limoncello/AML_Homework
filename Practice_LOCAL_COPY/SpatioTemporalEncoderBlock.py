import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from SpatioTemporalSelfAttention import SpatioTemporalSelfAttention
from MLP import MLP
from utils.init_layer import conv_init, bn_init, ln_init, fc_init

from rich import print

# taken as is from CVPR'23 reference paper https://github.com/zhenhuat/STCFormer
class SpatioTemporalEncoderBlock(nn.Module):
  
    def __init__(
        self, 
        in_features: int, out_features: int, num_joints: int,
        num_frames: int, num_frames_out: int,
        num_heads: int, use_skip_connection: bool,
        dropout: float
    ):
        super().__init__()

        self.use_skip_connection = use_skip_connection

        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.spatio_temporal_cross_attention = SpatioTemporalSelfAttention(
            in_features=in_features, out_features=out_features, num_joints=num_joints,
            num_frames=num_frames, num_frames_out=num_frames_out,
            num_heads=num_heads
        )

        self.dropout_1 = nn.Dropout(p=dropout)

        self.layer_norm_2 = nn.LayerNorm(out_features)

        self.mlp = MLP(out_features, out_features)

        self.dropout_2 = nn.Dropout(p=dropout)

            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.LayerNorm):
                ln_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(self, encoder_input: Tensor, mask_s: Optional[Tensor]=None, mask_t: Optional[Tensor]=None):
        # encoder_input.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        
        enc_inp_for_skip_connection = encoder_input

        encoder_input = self.layer_norm_1(encoder_input)

        encoder_input = self.spatio_temporal_cross_attention(
            q=encoder_input, k=encoder_input, v=encoder_input, 
            mask_s=mask_s, mask_t=mask_t
        )

        encoder_input = self.dropout_1(encoder_input)

        if self.use_skip_connection:
            encoder_input = encoder_input + enc_inp_for_skip_connection

        enc_inp_for_skip_connection = encoder_input

        encoder_input = self.layer_norm_2(encoder_input)

        encoder_output = self.mlp(encoder_input)

        encoder_output = self.dropout_2(encoder_output)

        if self.use_skip_connection:
            encoder_output = encoder_output + enc_inp_for_skip_connection

        return encoder_output
    
    





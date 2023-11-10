import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention
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

        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.spatio_temporal_cross_attention = SpatioTemporalCrossAttention(
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
            
            
    def forward(self, decoder_input: Tensor, encoder_output: Tensor, mask_s: Optional[Tensor]=None, mask_t: Optional[Tensor]=None):
        #  decoder_input.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        # encoder_output.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        
        dec_inp_for_skip_connection = decoder_input * self.skip_connection_weight

        decoder_input = self.layer_norm_1(decoder_input)

        decoder_input = self.spatio_temporal_cross_attention(
            q=decoder_input, k=encoder_output, v=encoder_output, 
            mask_s=mask_s, mask_t=mask_t
        )

        decoder_input = self.dropout_1(decoder_input)

        if self.use_skip_connection:
            decoder_input = decoder_input + dec_inp_for_skip_connection

        dec_inp_for_skip_connection = decoder_input * self.skip_connection_weight

        decoder_input = self.layer_norm_2(decoder_input)

        decoder_input = self.mlp(decoder_input)

        decoder_input = self.dropout_2(decoder_input)

        if self.use_skip_connection:
            decoder_input = decoder_input + dec_inp_for_skip_connection

        return decoder_input
    
    





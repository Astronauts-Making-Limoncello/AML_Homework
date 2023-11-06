import torch
import torch.nn as nn

from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention
from MLP import MLP
from utils.init_layer import conv_init, bn_init, ln_init, fc_init

from rich import print

# taken as is from CVPR'23 reference paper https://github.com/zhenhuat/STCFormer
class SpatioTemporalBlock(nn.Module):
  
    def __init__(
        self, 
        in_features, out_features, num_joints,
        num_frames, num_frames_out,
        num_heads, use_skip_connection
    ):
        super().__init__()

        self.use_skip_connection = use_skip_connection

        self.layer_norm_1 = nn.LayerNorm(in_features)

        self.spatio_temporal_cross_attention = SpatioTemporalCrossAttention(
            in_features=in_features, out_features=out_features, num_joints=num_joints,
            num_frames=num_frames, num_frames_out=num_frames_out,
            num_heads=num_heads
        )

        self.layer_norm_2 = nn.LayerNorm(out_features)

        self.mlp = MLP(out_features, out_features)
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.LayerNorm):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(self, x, encoder_output, mask_s, mask_t):
        #              x.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        # encoder_output.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        
        x_prime = x

        x = self.layer_norm_1(x)

        x = self.spatio_temporal_cross_attention(
            x, encoder_output, mask_s=mask_s, mask_t=mask_t
        )

        if self.use_skip_connection:
            x = x + x_prime

        x_prime = x

        x = self.layer_norm_2(x)

        x = self.mlp(x)

        if self.use_skip_connection:
            x = x + x_prime

        return x
    
    





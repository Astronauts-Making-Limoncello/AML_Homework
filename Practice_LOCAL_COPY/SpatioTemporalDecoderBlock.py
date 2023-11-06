import torch
import torch.nn as nn

from SpatioTemporalCrossAttention import SpatioTemporalCrossAttention

from rich import print

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

class SpatioTemporalDecoderBlock(nn.Module):
  
    def __init__(
        self, 
        in_features, out_features, num_joints,
        num_frames, num_frames_out,
        num_heads, use_skip_connection
    ):
        super().__init__()

        self.layer_norm = nn.LayerNorm(in_features)

        self.spatio_temporal_cross_attention = SpatioTemporalCrossAttention(
            in_features=in_features, out_features=out_features, num_joints=num_joints,
            num_frames=num_frames, num_frames_out=num_frames_out,
            num_heads=num_heads
        )

        self.use_skip_connection = use_skip_connection
        
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(self, x, encoder_output, mask_s, mask_t):
        #              x.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        # encoder_output.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        
        x_prime = x

        x = self.layer_norm(x)

        x = self.spatio_temporal_cross_attention(
            x, encoder_output, mask_s=mask_s, mask_t=mask_t
        )

        if self.use_skip_connection:
            x = x + x_prime

        return x
    
    





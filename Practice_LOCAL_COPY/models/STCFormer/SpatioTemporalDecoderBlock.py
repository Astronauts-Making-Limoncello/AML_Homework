import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from models.STCFormer.SpatioTemporalCrossAttention import SpatioTemporalCrossAttention
from models.STCFormer.SpatioTemporalSelfAttention import SpatioTemporalSelfAttention
from models.STCFormer.MLP import MLP
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
        """
        Initializes a new instance of the class.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            num_joints (int): The number of joints.
            num_frames (int): The number of frames.
            num_frames_out (int): The number of output frames.
            num_heads (int): The number of attention heads.
            use_skip_connection (bool): Whether to use skip connections.
            skip_connection_weight (float): The weight of the skip connection.
            dropout (float): The dropout rate.

        Returns:
            None
        """
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
        """
        Performs the forward pass of the model.

        Args:
            x (Tensor): The input tensor of shape (batch_size, temporal_dim, spatial_dim, feature_dim).
            memory (Tensor): The memory tensor of shape (batch_size, temporal_dim, spatial_dim, feature_dim).
            mask_s_self_attn (Tensor): The self-attention mask for the spatial dimension of shape (batch_size, temporal_dim, spatial_dim, spatial_dim).
            mask_t_self_attn (Tensor): The self-attention mask for the temporal dimension of shape (batch_size, temporal_dim, temporal_dim, temporal_dim).
            mask_s_cross_attn (Tensor): The cross-attention mask for the spatial dimension of shape (batch_size, temporal_dim, spatial_dim, spatial_dim).
            mask_t_cross_attn (Tensor): The cross-attention mask for the temporal dimension of shape (batch_size, temporal_dim, temporal_dim, temporal_dim).

        Returns:
            Tensor: The output tensor after the forward pass of shape (batch_size, temporal_dim, spatial_dim, feature_dim).
        """
        #  decoder_input.shape: batch_size, temporal_dim, spatial_dim, feature_dim
        # encoder_output.shape: batch_size, temporal_dim, spatial_dim, feature_dim

        x_ = self.layer_norm_1(x)
        x  = self.dropout_1(self.spatio_temporal_self_attention(x_, x_, x_, mask_s_self_attn, mask_t_self_attn))
        x_ = self.layer_norm_2(x)
        x  = self.dropout_2(self.spatio_temporal_cross_attention(x_, memory, memory, mask_s_cross_attn, mask_t_cross_attn))
        x_ = self.layer_norm_3(x)
        x  = self.dropout_3(self.mlp(x_))
        
        return x
    
    





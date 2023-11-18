import torch
import torch.nn as nn
from torch.nn.functional import softmax
from torch import Tensor
from typing import Optional
from utils.masking import causal_mask

from rich import print

# Adapted from original idea in Tang et. al 2023, "3D Human Pose Estimation with Spatio-Temporal Criss-Cross Attention"
# Paper: https://ieeexplore.ieee.org/document/10204803
# GitHub open-source implementation: https://github.com/zhenhuat/STCFormer

class SpatioTemporalCrossAttention(nn.Module):
    def __init__(
        self, in_features: int, out_features: int,
        num_frames: int, num_frames_out: int,
        num_joints: int, num_heads: int
    ):
        """
        Initializes a new instance of the class.

        Args:
            in_features (int): The number of input features.
            out_features (int): The number of output features.
            num_frames (int): The number of frames.
            num_frames_out (int): The number of output frames.
            num_joints (int): The number of joints.
            num_heads (int): The number of heads.

        Returns:
            None
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.num_frames = num_frames
        self.num_frames_out = num_frames_out
        self.num_joints = num_joints
        
        self.num_heads = num_heads
        assert out_features % num_heads == 0, "out_features % num_heads must be 0"
        self.hidden_features = out_features // num_heads

        assert self.hidden_features % 2 == 0, "hidden_features should be exact multiple of 2"
        self.hidden_features_s = self.hidden_features // 2
        self.hidden_features_t = self.hidden_features // 2

        self.Wq = nn.Linear(in_features, out_features)
        self.Wk = nn.Linear(in_features, out_features)
        self.Wv = nn.Linear(in_features, out_features)

        # maps temporal dimension of the encoder output to the temporal dimension of the decoder input
        # in order to have the same sequence length, needed during multi-dimensional mat mul operations
        self.temporal_explosion_s = nn.Conv2d(
            in_channels=self.num_frames, out_channels=self.num_frames_out, 
            kernel_size=1
        )
        
        self.temporal_explosion_t = nn.Conv2d(
            in_channels=self.num_frames, out_channels=self.num_frames_out, 
            kernel_size=1
        )

    def forward(
        self, q: Tensor, k: Tensor, v: Tensor, mask_s: Tensor, mask_t: Tensor,
    ):
        # x.shape             : b, num_frames_out, num_joints, in_features
        # encoder_output.shape: b, num_frames    , num_joints, in_features
        
        # x and encoder_output always follow this generic shape: b, t, s, f
        # b --> batch size              (default: 256)
        # h --> head dimension          (default: 4 or 8)
        # t --> temporal size/dimension (default: 10 in encoder, 25 in decoder)
        # s --> spacial size/dimension  (default: 22 joints)
        # f --> feature size/dimension  (default: 3, the x, y and z coordinates of the joints)

        assert q.shape[0] == k.shape[0] == v.shape[0], f"q, k and v should have the same batch size. Got instead {q.shape[0]}, {k.shape[0]} and {v.shape[0]}"

        batch_size = q.shape[0]

        # applying q, k and v transformations
        q = self.Wq(q)
        k = self.Wk(k)
        v = self.Wv(v)
        # q.shape      : b, num_frames_out, num_joints, out_features
        # k and v shape: b, num_frames    , num_joints, out_features

        temporal_size = q.shape[1]

        # splitting into heads
        # q = q.reshape(batch_size, self.num_heads, self.num_frames_out, self.num_joints, self.hidden_features)
        # using -1 instead of self.num_frames_out in order to be flexible to whatever number of frames
        # this flexibility is needed in autoregressive paradigms
        # NOTE this could be achieved by requiring num_frames_out in the forward pass,
        # NOTE but for now this approach works ok, since there's this comment explaining it :)
        # q = q.reshape(batch_size, self.num_heads,                  -1, self.num_joints, self.hidden_features)
        q = q.reshape(batch_size, self.num_heads, self.num_frames_out, self.num_joints, self.hidden_features)
        k = k.reshape(batch_size, self.num_heads, self.num_frames    , self.num_joints, self.hidden_features)
        v = v.reshape(batch_size, self.num_heads, self.num_frames    , self.num_joints, self.hidden_features)
        # q.shape      : b, num_heads, num_frames_out, num_joints, hidden_features
        # k and v shape: b, num_heads, num_frames    , num_joints, hidden_features


        # dividing q, k and v into spatial (_s) and temporal (_t) portions
        # generic shape of the transformation: (b, h, t, s, hidden_features) --> (b, h, t, s, hidden_features/2)
        q_s, q_t = torch.chunk(q, chunks=2, dim=-1)
        k_s, k_t = torch.chunk(k, chunks=2, dim=-1)
        v_s, v_t = torch.chunk(v, chunks=2, dim=-1)
        
        # to handle different sequence lengths between encoder_output and x
        k_s = k_s.permute((0, 2, 3, 1, 4))
        # generic shape of transformation: (b, h, t, s, hidden_features_s) --> (b, t, s, h, hidden_features_s)
        k_s = k_s.reshape(batch_size, self.num_frames, self.num_joints, self.num_heads * self.hidden_features//2)
        # generic shape of transformation: (b, t, s, h, hidden_features_s) --> (b, t, s, out_features/2)
        k_s = self.temporal_explosion_s(k_s)
        # (b, num_frames_out, num_joints, out_features/2)
        k_s = k_s.reshape(batch_size, self.num_frames_out, self.num_joints, self.num_heads, self.hidden_features_s)
        k_s = k_s.permute(0, 3, 1, 2, 4)
        # generic shape of transformation: (b, h, t, s, hidden_feature_s)

        # to handle different sequence lengths between encoder_output and x
        v_s = v_s.permute((0, 2, 3, 1, 4))
        # generic shape of transformation: (b, h, t, s, hidden_features_s) --> (b, t, s, h, hidden_features_s)
        v_s = v_s.reshape(batch_size, self.num_frames, self.num_joints, self.num_heads * self.hidden_features//2)
        # generic shape of transformation: (b, t, s, h, hidden_features_s) --> (b, t, s, out_features/2)
        v_s = self.temporal_explosion_s(v_s)
        # (b, num_frames_out, num_joints, out_features/2)
        v_s = v_s.reshape(batch_size, self.num_frames_out, self.num_joints, self.num_heads, self.hidden_features_s)
        v_s = v_s.permute(0, 3, 1, 2, 4)
        # generic shape of transformation: (b, h, t, s, hidden_feature_s)

        # generic shape of the transformation: (b, h, t, s, hidden_features_s) @ (b, h, t, hidden_features_s, s) --> (b, h, t, s, s)
        attn_s = torch.einsum("bhtsc,bhtdu->bhtsu", q_s, k_s.transpose(3, 4))
        # attn_s.shape: (b, num_frames_out, num_joints, num_joints)

        # masking, if mask_s provided
        if mask_s is not None:
            attn_s = attn_s.masked_fill_(mask_s == 0, -1e9)
            # print(attn_s)

        # dimension of interest (i.e. temporal or spatial) needs to be the -2 dimension
        # so, gotta permute temporal q, k and v to reflect this 
        q_t = q_t.permute(0, 1, 3, 2, 4)
        # (b, num_heads, num_frames_out, num_joints, hidden_features_t) --> (b, num_heads, num_joints, num_frames_out, hidden_features_t)
        k_t = k_t.permute(0, 1, 3, 2, 4)
        v_t = v_t.permute(0, 1, 3, 2, 4)
        # (b, num_heads, num_frames    , num_joints, hidden_features_t) --> (b, num_heads, num_joints, num_frames, hidden_features_t)
        
  
        k_t = k_t.permute(0, 1, 3, 2, 4)

        k_t = k_t.reshape((-1, self.num_frames, self.num_joints, self.hidden_features_t))

        k_t = self.temporal_explosion_t(k_t)
 
        k_t = k_t.reshape((batch_size, self.num_heads, self.num_frames_out, self.num_joints, self.hidden_features_t))

        k_t = k_t.permute(0, 1, 3, 2, 4)

        attn_t = torch.einsum("bhstf,bhsug->bhstg", q_t, k_t.transpose(3, 4))
        # (b, num_heads, num_joints, num_frames_out, hidden_features_t) 
        # @ 
        # (b, num_heads, num_joints, hidden_features_t, num_frames) 
        # =
        # (b, num_heads, num_joints, num_frames_out, num_frames)

        if mask_t is not None:
            attn_t = attn_t.masked_fill_(mask_t == 0, -1e9)
        
        attn_s = softmax(attn_s, dim=-1)
        attn_t = softmax(attn_t, dim=-1)

        x_s = torch.einsum("bhtss,bhtuf->bhtsf", attn_s, v_s)
        # generic shape of the transformation: (b, h, t, s, s) @ (b, h, t, s, hidden_features_s) --> (b, h, t, s, hidden_features_s)

        x_t = torch.einsum("bhstf,bhsug->bhstg", attn_t, v_t)
        # generic shape of the transformation: (b, h, s, t, t) @ (b, h, s, t, hidden_features_t) --> (b, h, s, t, hidden_features_t)
        x_t = x_t.permute(0, 1, 3, 2, 4)
        # generic shape of the transformation: (b, h, s, t, hidden_features_t) --> (b, h, t, s, hidden_features_t)

        # in order to handle autoregression evaluation, gotta be able to handle temporal size in a dynamic way 
        # in fact, in autoregression, temporal dimension is not fixed, but it grows predicted token after predicted token 
        
        # x_t is the one suffering from the growing temporal size problem in autoregression,
        # so concatenating x_s as is to x_t, in autoregression, will not work, because of the different temporal size
        # here's how we decided to solve this:
        #   - create two x_s chunks, one up to the temporal size and one from temporal size to the end
        #   - concatenate x_s_up_to_temporal_size with x_t as is, because the temporal size matches
        #   - concatenate x_s_after_temporal_size with zero valued features
        #   - reunite the two x_s chunks
        # the last choice is because after temporal size, temporal attention has not been computed
        # so, we set its vector to zero
        # we decided to explicit all steps to be as transparent and clear as possible

        # get up to temporal size chunk for x_s
        x_s_up_to_temporal_size = x_s[:, :, :temporal_size, :, :]
        
        # concatenate temporal attention features up to temporal size to x_s
        x_up_to_temporal_size = torch.cat((x_s_up_to_temporal_size, x_t), dim=-1)

        # get after temporal size chunk for x_s
        x_s_after_temporal_size = x_s[:, :, temporal_size:, :, :]

        # concatenate temporal attention features after temporal size to x_s
        x_after_temporal_size = torch.cat((x_s[:, :, temporal_size:, :, :], torch.zeros_like(x_s_after_temporal_size)), dim=-1)

        # merge the two x chunks
        x = torch.cat((x_up_to_temporal_size, x_after_temporal_size), dim=2)

        # x = torch.cat((x_s, x_t), dim=-1)
        # (b, num_heads, num_frames_out, num_joints, hidden_features_s), (b, num_heads, num_frames_out, num_joints, hidden_features_t) --> (b, num_heads, num_frames_out, num_joints, hidden_features)

        x = x.permute(0, 2, 3, 1, 4)
        # (b, num_heads, num_frames_out, num_joints, hidden_features) --> (b, num_frames_out, num_joints, num_heads, hidden_features)

        x = x.reshape(batch_size, self.num_frames_out, self.num_joints, self.num_heads*self.hidden_features)
        # (b, num_frames_out, num_joints, out_features)

        return x
        # (b, num_frames_out, num_joints, out_features)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))
    
    num_joints = 22
    num_frames = 10
    num_frames_out = 25

    in_features = 3
    out_features = 128
    num_heads = 4
    
    spatio_temporal_cross_attention = SpatioTemporalCrossAttention(
        in_features=in_features, out_features=out_features,
        num_frames=num_frames, num_frames_out=num_frames_out,
        num_joints=num_joints,
        num_heads=num_heads
    ).to(device)
    
    batch_size = 256

    encoder_output = torch.rand((batch_size, num_frames, num_joints, in_features)).to(device)
    tgt = torch.rand((batch_size, num_frames_out, num_joints, in_features)).to(device)
    # alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
    # seq_len is num_frames_out in decoder, so gotta use num_frames_out
    # first three dimensions set to 1 are for: batch, heads and time
    decoder_mask_s = causal_mask((1, 1, 1, num_joints, num_joints)).to(device)

    # since we are in the decoder, mask must have same shape of attn_t
    # REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
    # so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
    # first three dimensions set to 1 are for: batch, heads and space
    decoder_mask_t = causal_mask((1, 1, 1, num_frames_out, num_frames)).to(device)

    st_cross_attn = spatio_temporal_cross_attention(
        q=tgt, k=encoder_output, v=encoder_output, 
        mask_s=decoder_mask_s, mask_t=decoder_mask_t
    )

    print(f"st_cross_attn.shape: {st_cross_attn.shape}")

if __name__ == "__main__":
    main()



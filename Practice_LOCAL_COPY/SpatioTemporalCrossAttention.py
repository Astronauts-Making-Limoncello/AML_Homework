import torch
import torch.nn as nn

from torch.nn.functional import softmax

from pos_embed import Pos_Embed

from rich import print

def causal_mask(mask_shape):
  mask = torch.triu(torch.ones(mask_shape), diagonal=1).type(torch.int)
  return mask == 0

# Adapted from original idea in Tang et. al 2023, "3D Human Pose Estimation with Spatio-Temporal Criss-Cross Attention"
# Paper: https://ieeexplore.ieee.org/document/10204803
# GitHub open-source implementation: https://github.com/zhenhuat/STCFormer

class SpatioTemporalCrossAttention(nn.Module):
    def __init__(
        self, in_features, out_features,
        num_frames, num_frames_out,
        num_joints, num_heads
    ):
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
        self.temporal_explosion = nn.Conv2d(
            in_channels=self.num_frames, out_channels=self.num_frames_out, 
            kernel_size=1
        )

    def forward(self, x, encoder_output, mask_s, mask_t):
        # x.shape             : b, num_frames_out, num_joints, in_features
        # encoder_output.shape: b, num_frames    , num_joints, in_features
        
        # x and encoder_output always follow this generic shape: b, t, s, f
        # b --> batch size              (default: 256)
        # h --> head dimension          (default: 4 or 8)
        # t --> temporal size/dimension (default: 10 in encoder, 25 in decoder)
        # s --> spacial size/dimension  (default: 22 joints)
        # f --> feature size/dimension  (default: 3, the x, y and z coordinates of the joints)

        assert x.shape[0] == encoder_output.shape[0], "x and encoder_output should have the same batch size"

        batch_size = x.shape[0]

        # applying q, k and v transformations
        q = self.Wq(x)
        k = self.Wk(encoder_output)
        v = self.Wv(encoder_output)
        # q.shape      : b, num_frames_out, num_joints, out_features
        # k and v shape: b, num_frames    , num_joints, out_features

        # splitting into heads
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
        k_s = self.temporal_explosion(k_s)
        # (b, num_frames_out, num_joints, out_features/2)
        k_s = k_s.reshape(batch_size, self.num_frames_out, self.num_joints, self.num_heads, self.hidden_features_s)
        k_s = k_s.permute(0, 3, 1, 2, 4)
        # generic shape of transformation: (b, h, t, s, hidden_feature_s)

        # to handle different sequence lengths between encoder_output and x
        v_s = v_s.permute((0, 2, 3, 1, 4))
        # generic shape of transformation: (b, h, t, s, hidden_features_s) --> (b, t, s, h, hidden_features_s)
        v_s = v_s.reshape(batch_size, self.num_frames, self.num_joints, self.num_heads * self.hidden_features//2)
        # generic shape of transformation: (b, t, s, h, hidden_features_s) --> (b, t, s, out_features/2)
        v_s = self.temporal_explosion(v_s)
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

        # dimension of interest (i.e. temporal or spatial) needs to be the -2 dimension
        # so, gotta permute temporal q, k and v to reflect this 
        q_t = q_t.permute(0, 1, 3, 2, 4)
        # (b, num_heads, num_frames_out, num_joints, hidden_features_t) --> (b, num_heads, num_joints, num_frames_out, hidden_features_t)
        k_t = k_t.permute(0, 1, 3, 2, 4)
        v_t = v_t.permute(0, 1, 3, 2, 4)
        # (b, num_heads, num_frames    , num_joints, hidden_features_t) --> (b, num_heads, num_joints, num_frames, hidden_features_t)
        

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

        x = torch.cat((x_s, x_t), dim=-1)
        # (b, num_heads, num_frames_out, num_joints, hidden_features_s), (b, num_heads, num_frames_out, num_joints, hidden_features_t) --> (b, num_heads, num_frames_out, num_joints, hidden_features)

        x = x.permute(0, 2, 3, 1, 4)
        # (b, num_heads, num_frames_out, num_joints, hidden_features) --> (b, num_frames_out, num_joints, num_heads, hidden_features)

        x = x.reshape(batch_size, self.num_frames_out, self.num_joints, self.num_heads*self.hidden_features)
        # (b, num_frames_out, num_joints, out_features)

        return x
        # (b, num_frames_out, num_joints, out_features)



if __name__ == "__main__":
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
        x=tgt, encoder_output=encoder_output, 
        mask_s=decoder_mask_s, mask_t=decoder_mask_t
    )

    print(f"st_cross_attn.shape: {st_cross_attn.shape}")



import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from models.STCFormer.SpatioTemporalEncoder import SpatioTemporalEncoder
from models.STCFormer.SpatioTemporalDecoder import SpatioTemporalDecoder
from utils.init_layer import conv_init, bn_init, ln_init, fc_init
from utils.masking import causal_mask

from rich import print

class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self, 
        st_encoder: SpatioTemporalEncoder, st_decoder: SpatioTemporalDecoder,
        num_frames: int, num_joints: int, 
        in_features_encoder: int, in_features_decoder: int
    ):
        """
        Initializes the class with the given parameters.

        Args:
            st_encoder (SpatioTemporalEncoder): The spatio-temporal encoder.
            st_decoder (SpatioTemporalDecoder): The spatio-temporal decoder.
            num_frames (int): The number of frames.
            num_joints (int): The number of joints.
            in_features_encoder (int): The number of input features for the encoder.
            in_features_decoder (int): The number of input features for the decoder.
        """
        super().__init__()

        self.st_encoder = st_encoder
        self.st_decoder = st_decoder

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.in_features_encoder = in_features_encoder
        self.in_features_decoder = in_features_decoder

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
        self, 
        src: Tensor, tgt: Tensor, 
        tgt_mask_s_self_attn: Tensor, tgt_mask_t_self_attn: Tensor,
        tgt_mask_s_cross_attn: Tensor, tgt_mask_t_cross_attn: Tensor,
        src_mask_s: Optional[Tensor]=None, src_mask_t: Optional[Tensor]=None,
    ):
        """
        Forward pass of the SpatioTemporalTransformer model.
        
        Args:
            src (Tensor): The input source tensor.
            tgt (Tensor): The input target tensor.
            tgt_mask_s_self_attn (Tensor): The self-attention mask for the source target tensor.
            tgt_mask_t_self_attn (Tensor): The self-attention mask for the temporal target tensor.
            tgt_mask_s_cross_attn (Tensor): The cross-attention mask for the source target tensor.
            tgt_mask_t_cross_attn (Tensor): The cross-attention mask for the temporal target tensor.
            src_mask_s (Optional[Tensor]): The optional self-attention mask for the source tensor.
            src_mask_t (Optional[Tensor]): The optional self-attention mask for the temporal tensor.
        
        Returns:
            Tensor: The output tensor of the decoder.
        """

        encoder_output = self.st_encoder.forward(
            encoder_input=src, mask_s=src_mask_s, mask_t=src_mask_t
        )

        decoder_output = self.st_decoder.forward(
            decoder_input=tgt, encoder_output=encoder_output, 
            mask_s_self_attn=tgt_mask_s_self_attn, mask_t_self_attn=tgt_mask_t_self_attn,
            mask_s_cross_attn=tgt_mask_s_cross_attn, mask_t_cross_attn=tgt_mask_t_cross_attn
        )

        return decoder_output


def main():
    """
    Generate the function comment for the given function body.

    :param None:
    :return str: The function comment in Markdown format.
    """
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

    # temporal
    num_frames = 10
    num_frames_out = 25
    
    # spatial 
    num_joints = 22
    in_features_encoder = 3
    in_features_decoder = 4
    hidden_features = 128
    out_features_encoder = 4
    out_features_decoder = 4

    # model
    num_heads = 8
    use_skip_connection = True
    skip_connection_weight = 1.
    dropout = 0.1

    num_encoder_blocks = 3

    st_encoder = SpatioTemporalEncoder(
        in_features_encoder, hidden_features, out_features_encoder, num_joints,
        num_frames, num_frames, # it's encoder, so num_frames_out == num_frames
        num_heads, use_skip_connection,
        num_encoder_blocks,
        dropout=dropout
    )

    encoder_mask_s = None
    encoder_mask_t = None

    num_decoder_blocks = 3

    st_decoder = SpatioTemporalDecoder(
        decoder_input_in_features=in_features_decoder, 
        encoder_output_in_features=out_features_encoder,
        hidden_features=hidden_features,
        out_features=out_features_decoder, num_joints=num_joints, 
        num_frames=num_frames, num_frames_out=num_frames_out, 
        num_heads=num_heads, use_skip_connection=use_skip_connection, skip_connection_weight=skip_connection_weight,
        num_decoder_blocks=num_decoder_blocks,
        dropout=dropout
    )

    # alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
    # seq_len is num_frames_out in decoder, so gotta use num_frames_out
    # first three dimensions set to 1 are for: batch, heads and time
    decoder_mask_s_self_attention = causal_mask((1, 1, 1, num_joints, num_joints)).to(device)
    decoder_mask_s_cross_attention = causal_mask((1, 1, 1, num_joints, num_joints)).to(device)
    # decoder_mask_s = causal_mask((batch_size, num_heads, num_frames_out, num_joints, num_joints)).to(device)

    # since we are in the decoder, mask must have same shape of attn_t
    # REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
    # so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
    # first three dimensions set to 1 are for: batch, heads and space
    # decoder_mask_t_cross_attention = causal_mask((1, 1, 1, num_frames_out, num_frames)).to(device)
    decoder_mask_t_cross_attention = causal_mask((1, 1, 1, num_frames_out, num_frames_out)).to(device)
    decoder_mask_t_self_attention  = causal_mask((1, 1, 1, num_frames_out, num_frames_out)).to(device)
    # decoder_mask_t = causal_mask((batch_size, num_heads, num_joints, num_frames_out, num_frames)).to(device)

    st_transformer = SpatioTemporalTransformer(
        st_encoder=st_encoder, st_decoder=st_decoder,
        num_frames=num_frames, num_joints=num_joints, 
        in_features_encoder=in_features_encoder, in_features_decoder=in_features_decoder, 
    ).to(device)

    batch_size = 256
    # src = torch.rand((batch_size, in_features, num_frames, num_joints)).to(device)
    src = torch.rand((batch_size, num_frames    , num_joints, in_features_encoder)).to(device)
    tgt = torch.rand((batch_size, num_frames_out, num_joints, in_features_decoder)).to(device)

    decoder_output = st_transformer.forward(
        src=src, tgt=tgt,
        src_mask_s=encoder_mask_s, src_mask_t=encoder_mask_t,
        tgt_mask_s_self_attn=decoder_mask_s_self_attention, tgt_mask_t_self_attn=decoder_mask_t_self_attention,
        tgt_mask_s_cross_attn=decoder_mask_s_cross_attention, tgt_mask_t_cross_attn=decoder_mask_t_cross_attention
    )
    print(f"decoder_output: {decoder_output.shape}")

if __name__ == "__main__":

    main()
    





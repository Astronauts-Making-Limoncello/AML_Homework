import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from SpatioTemporalDecoderBlock import SpatioTemporalDecoderBlock
from utils.init_layer import conv_init, fc_init, bn_init, ln_init
from utils.masking import causal_mask

from rich import print


class SpatioTemporalDecoder(nn.Module):
  
    def __init__(
        self, 
        decoder_input_in_features: int, encoder_output_in_features: int,
        hidden_features: int, out_features: int, num_joints: int,
        num_frames: int, num_frames_out: int,
        num_heads: int, use_skip_connection: bool,
        skip_connection_weight: float,
        num_decoder_blocks: int, dropout: float
    ):
        """
        Initializes the SpatioTemporalDecoder model.

        Args:
            decoder_input_in_features (int): The number of input features for the decoder.
            encoder_output_in_features (int): The number of input features for the encoder output.
            hidden_features (int): The number of hidden features in the model.
            out_features (int): The number of output features.
            num_joints (int): The number of joints in the model.
            num_frames (int): The number of input frames.
            num_frames_out (int): The number of output frames.
            num_heads (int): The number of attention heads in the model.
            use_skip_connection (bool): Whether to use skip connections.
            skip_connection_weight (float): The weight of the skip connections.
            num_decoder_blocks (int): The number of decoder blocks in the model.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super().__init__()

        self.hidden_features = hidden_features
        self.num_frames_out = num_frames_out
        self.num_joints = num_joints

        self.decoder_input_fc_in  = nn.Linear(decoder_input_in_features, self.hidden_features)
        self.encoder_output_fc_in = nn.Linear(encoder_output_in_features, self.hidden_features)

        self.decoder_blocks = nn.Sequential()
        for decoder_block_id in range(num_decoder_blocks):
            
            self.decoder_blocks.add_module(
                f"decoder_block_{decoder_block_id}",
                SpatioTemporalDecoderBlock(
                    in_features=hidden_features, out_features=hidden_features, num_joints=num_joints, 
                    num_frames=num_frames, num_frames_out=num_frames_out, 
                    num_heads=num_heads, use_skip_connection=use_skip_connection,
                    skip_connection_weight=skip_connection_weight, 
                    dropout=dropout
                )
            )

        self.x_fc_out = nn.Linear(hidden_features, out_features)
            
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
            decoder_input: Tensor, encoder_output: Tensor, 
            mask_s_self_attn: Tensor, mask_t_self_attn: Tensor,
            mask_s_cross_attn: Tensor, mask_t_cross_attn: Tensor
        ):
        """
        Applies the forward pass of the model.

        Args:
            decoder_input (Tensor): The input tensor to the decoder. Shape: (batch_size, temporal_size, num_joints, hidden_features).
            encoder_output (Tensor): The output tensor from the encoder. Shape: (batch_size, temporal_size, num_joints, hidden_features).
            mask_s_self_attn (Tensor): The self-attention mask for the source sequence. Shape: (batch_size, temporal_size, temporal_size).
            mask_t_self_attn (Tensor): The self-attention mask for the target sequence. Shape: (batch_size, num_frames_out, num_frames_out).
            mask_s_cross_attn (Tensor): The cross-attention mask for the source sequence. Shape: (batch_size, temporal_size, num_frames_out).
            mask_t_cross_attn (Tensor): The cross-attention mask for the target sequence. Shape: (batch_size, num_frames_out, temporal_size).

        Returns:
            Tensor: The output tensor from the model. Shape: (batch_size, num_frames_out, num_joints, hidden_features).
        """

        decoder_input  = self.decoder_input_fc_in(decoder_input)
        encoder_output = self.encoder_output_fc_in(encoder_output)

        if (decoder_input.shape[1] != self.num_frames_out):

            batch_size, temporal_size, _, _ = decoder_input.shape

            temporal_padding = torch.zeros(
                (batch_size, self.num_frames_out-temporal_size, self.num_joints, self.hidden_features)
            ).to(decoder_input.device)

            decoder_input = torch.cat((decoder_input, temporal_padding), dim=1)

        for decoder_block in self.decoder_blocks:
            x = decoder_block.forward(
                decoder_input, encoder_output, 
                mask_s_self_attn=mask_s_self_attn, mask_t_self_attn=mask_t_self_attn,
                mask_s_cross_attn=mask_s_cross_attn, mask_t_cross_attn=mask_t_cross_attn
            )
            


        x = self.x_fc_out(x)

        return x


def main():
    """
    Main function to execute the program.
    
    This function initializes the device based on the availability of a CUDA-capable GPU.
    It then prints the type of device being used.
    
    The function sets the values for the following variables:
    - num_joints: The number of joints.
    - num_frames: The number of frames.
    - num_frames_out: The number of output frames.
    - in_features: The number of input features.
    - hidden_features: The number of hidden features.
    - out_features: The number of output features.
    - num_heads: The number of attention heads.
    - use_skip_connection: A boolean value indicating whether to use skip connections.
    - num_decoder_blocks: The number of decoder blocks.
    
    The function initializes an instance of the SpatioTemporalDecoder class with the above parameters.
    
    The function sets the value for the batch size.
    
    The function initializes a random tensor for the decoder input with the shape (batch_size, num_frames_out, num_joints, in_features).
    
    The function initializes a random tensor for the encoder output with the shape (batch_size, num_frames, num_joints, in_features).
    
    The function creates a causal mask for the spatial dimension with the shape (1, 1, 1, num_joints, num_joints).
    
    The function creates a causal mask for the temporal dimension with the shape (1, 1, 1, num_frames_out, num_frames).
    
    The function calls the forward method of the spatio_temporal_decoder instance with the decoder input, encoder output, spatial mask, and temporal mask as arguments.
    
    The function prints the shape of the output tensor.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))
    
    num_joints = 22
    num_frames = 10
    num_frames_out = 25

    in_features = 3
    hidden_features = 128
    out_features = 3

    num_heads = 4
    use_skip_connection = True, 
    num_decoder_blocks = 2

    spatio_temporal_decoder = SpatioTemporalDecoder(
        in_features, hidden_features, out_features, num_joints,
        num_frames, num_frames_out,
        num_heads, use_skip_connection,
        num_decoder_blocks
    )

    batch_size = 256
    # decoder input must come with in_features, because it comes directly from the
    # dataset!
    decoder_input  = torch.rand((batch_size, num_frames_out, num_joints, in_features)) 
    # encoder output must come already mapped to hidden features, because it is
    # supposed to come out of the encoder
    encoder_output = torch.rand((batch_size, num_frames    , num_joints, in_features))

    # alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
    # seq_len is num_frames_out in decoder, so gotta use num_frames_out
    # first three dimensions set to 1 are for: batch, heads and space
    mask_s = causal_mask((1, 1, 1, num_joints, num_joints))
    
    # since we are in the decoder, mask must have same shape of attn_t
    # REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
    # so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
    # first three dimensions set to 1 are for: batch, heads and space
    mask_t = causal_mask((1, 1, 1, num_frames_out, num_frames))

    x = spatio_temporal_decoder.forward(decoder_input, encoder_output, mask_s, mask_t)

    print(f"x.shape: {x.shape}")


if __name__ == "__main__":
    main()
    
    





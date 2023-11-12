import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional

from SpatioTemporalEncoderBlock import SpatioTemporalEncoderBlock
from utils.init_layer import conv_init, fc_init, bn_init, ln_init
from utils.masking import causal_mask

from rich import print



class SpatioTemporalEncoder(nn.Module):
  
    def __init__(
        self, 
        in_features: int, hidden_features: int, out_features: int, num_joints: int,
        num_frames: int, num_frames_out: int,
        num_heads: int, use_skip_connection: bool,
        num_encoder_blocks: int,
        dropout: float
    ):
        """
        Initializes the SpatioTemporalEncoder model.

        Args:
            in_features (int): The number of input features.
            hidden_features (int): The number of hidden features.
            out_features (int): The number of output features.
            num_joints (int): The number of joints.
            num_frames (int): The number of frames.
            num_frames_out (int): The number of output frames.
            num_heads (int): The number of attention heads.
            use_skip_connection (bool): Whether to use skip connection.
            num_encoder_blocks (int): The number of encoder blocks.
            dropout (float): The dropout rate.

        Returns:
            None
        """
        super().__init__()

        self.encoder_input_fc_in = nn.Linear(in_features, hidden_features)

        self.encoder_blocks = nn.Sequential()
        for encoder_block_id in range(num_encoder_blocks):
            
            self.encoder_blocks.add_module(
                f"encoder_block_{encoder_block_id}",
                SpatioTemporalEncoderBlock(
                    in_features=hidden_features, out_features=hidden_features, num_joints=num_joints, 
                    num_frames=num_frames, num_frames_out=num_frames_out, 
                    num_heads=num_heads, use_skip_connection=use_skip_connection,
                    dropout=dropout
                )
            )

        self.encoder_output_fc_out = nn.Linear(hidden_features, out_features)
            
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
        """
        Forward pass of the model.
       
        Args:
           encoder_input (Tensor): The input tensor to the encoder.
           mask_s (Optional[Tensor], optional): The source mask tensor. Defaults to None.
           mask_t (Optional[Tensor], optional): The target mask tensor. Defaults to None.
       
        Returns:
           Tensor: The output tensor from the encoder.
        """
       
        encoder_input = self.encoder_input_fc_in(encoder_input)

        for encoder_block in self.encoder_blocks:
            encoder_output = encoder_block.forward(
                encoder_input, mask_s=mask_s, mask_t=mask_t
            )
            
        encoder_output = self.encoder_output_fc_out(encoder_output)

        return encoder_output


def main():
    """
    The main function that executes the entire program.

    This function initializes the device to be used for computation. It checks if a CUDA-enabled GPU is available and sets the device accordingly. It then prints the device being used and its type.

    The function sets the values for the number of joints, number of frames, and number of output frames. It also sets the input features, hidden features, and output features.

    The function initializes the number of heads, the use of skip connections, and the number of encoder blocks.

    The function creates an instance of the SpatioTemporalEncoder class, passing the input features, hidden features, output features, number of joints, number of frames, and number of output frames as arguments.

    The function sets the batch size and generates random encoder input of shape (batch_size, num_frames, num_joints, in_features).

    The function initializes the mask_s and mask_t variables. These variables can be used to apply masks to the attention mechanism of the encoder.

    The function calls the forward method of the spatio_temporal_encoder instance, passing the encoder input, mask_s, and mask_t as arguments. The result is stored in the x variable.

    The function prints the shape of x.

    Parameters:
    None

    Returns:
    None
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
    num_encoder_blocks = 2

    spatio_temporal_encoder = SpatioTemporalEncoder(
        in_features, hidden_features, out_features, num_joints,
        num_frames, num_frames, # encoder, so num_frames_out == num_frames
        num_heads, use_skip_connection,
        num_encoder_blocks
    )

    batch_size = 256
    # encoder input must come with in_features, because it comes directly from the
    # dataset!
    encoder_input = torch.rand((batch_size, num_frames, num_joints, in_features)) 

    # alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
    # seq_len is num_frames_out in encoder, so gotta use num_frames_out
    # first three dimensions set to 1 are for: batch, heads and space
    mask_s = None
    # mask_s = causal_mask((1, 1, 1, num_joints, num_joints))
    
    # since we are in the encoder, mask must have same shape of attn_t
    # REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
    # so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
    # first three dimensions set to 1 are for: batch, heads and space
    mask_t = None
    # mask_t = causal_mask((1, 1, 1, num_frames_out, num_frames))

    x = spatio_temporal_encoder.forward(encoder_input, mask_s, mask_t)

    print(f"x.shape: {x.shape}")


if __name__ == "__main__":
    main()
    
    





import torch
import torch.nn as nn

from SpatioTemporalDecoderBlock import SpatioTemporalDecoderBlock
from utils.init_layer import conv_init, fc_init, bn_init, ln_init

from rich import print



class SpatioTemporalDecoder(nn.Module):
  
    def __init__(
        self, 
        in_features, hidden_features, out_features, num_joints,
        num_frames, num_frames_out,
        num_heads, use_skip_connection,
        num_decoder_blocks
    ):
        super().__init__()

        self.num_joints = num_joints

        self.x_fc_in              = nn.Linear(in_features, hidden_features)
        self.encoder_output_fc_in = nn.Linear(in_features, hidden_features)

        self.decoder_blocks = nn.Sequential()
        for decoder_block_id in range(num_decoder_blocks):
            
            self.decoder_blocks.add_module(
                f"decoder_block_{decoder_block_id}",
                SpatioTemporalDecoderBlock(
                    in_features=hidden_features, out_features=hidden_features, num_joints=num_joints, 
                    num_frames=num_frames, num_frames_out=num_frames_out, 
                    num_heads=num_heads, use_skip_connection=use_skip_connection
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
            
            
    def forward(self, x, encoder_output, mask_s, mask_t):

        x = self.x_fc_in(x)
        encoder_output = self.encoder_output_fc_in(encoder_output)

        for decoder_block in self.decoder_blocks:
            x = decoder_block.forward(
                x, encoder_output, mask_s=mask_s, mask_t=mask_t
            )
            


        x = self.x_fc_out(x)

        return x
    
def causal_mask(mask_shape):
    mask = torch.triu(torch.ones(mask_shape), diagonal=1).type(torch.int)
    return mask == 0

def main():
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
    # encoder output must come already mapped to hidden features, because it is
    # supposed to come out of the encoder
    encoder_output = torch.rand((batch_size, num_frames    , num_joints, in_features))
    # decoder input must come with in_features, because it comes directly from the
    # dataset!
    x              = torch.rand((batch_size, num_frames_out, num_joints, in_features)) 

    # alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
    # seq_len is num_frames_out in decoder, so gotta use num_frames_out
    # first three dimensions set to 1 are for: batch, heads and space
    mask_s = causal_mask((1, 1, 1, num_joints, num_joints))
    
    # since we are in the decoder, mask must have same shape of attn_t
    # REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
    # so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
    # first three dimensions set to 1 are for: batch, heads and space
    mask_t = causal_mask((1, 1, 1, num_frames_out, num_frames))

    x = spatio_temporal_decoder.forward(x, encoder_output, mask_s, mask_t)

    print(f"x.shape: {x.shape}")


if __name__ == "__main__":
    main()
    
    





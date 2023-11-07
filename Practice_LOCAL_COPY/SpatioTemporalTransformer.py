import torch
import torch.nn as nn

from SpatioTemporalEncoder import SpatioTemporalEncoder
from SpatioTemporalDecoder import SpatioTemporalDecoder

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

def causal_mask(mask_shape):
    mask = torch.triu(torch.ones(mask_shape), diagonal=1).type(torch.int)
    return mask == 0

class SpatioTemporalTransformer(nn.Module):
    def __init__(
        self, 
        st_encoder: SpatioTemporalEncoder, st_decoder: SpatioTemporalDecoder,
        num_frames, num_joints, in_features
    ):
        super().__init__()

        self.st_encoder = st_encoder
        self.st_decoder = st_decoder

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.in_features = in_features

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
        

    def forward(self, src, tgt, tgt_mask_s, tgt_mask_t):

        # print(f"\[SpatioTemporalTransformer.forward] src.shape: {src.shape}")
        # print(f"\[SpatioTemporalTransformer.forward] tgt.shape: {tgt.shape}")

        encoder_output = self.st_encoder.forward(
            encoder_input=src, mask_s=None, mask_t=None
        )
        encoder_output = encoder_output.view(-1, self.num_frames, self.num_joints, self.in_features)
        # print(f"\[SpatioTemporalTransformer.forward] encoder_output.shape: {encoder_output.shape}")

        decoder_output = self.st_decoder.forward(
            decoder_input=tgt, encoder_output=encoder_output, 
            mask_s=tgt_mask_s, mask_t=tgt_mask_t
        )
        # print(f"\[SpatioTemporalTransformer.forward] decoder_output.shape: {decoder_output.shape}")

        return decoder_output


def main():
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

    # temporal
    num_frames = 10
    num_frames_out = 25
    
    # spatial 
    num_joints = 22
    in_features = 3
    hidden_features = 128
    out_features = 3

    # model
    num_heads = 8
    use_skip_connection = True

    num_encoder_blocks = 3

    st_encoder = SpatioTemporalEncoder(
        in_features, hidden_features, out_features, num_joints,
        num_frames, num_frames, # it's encoder, so num_frames_out == num_frames
        num_heads, use_skip_connection,
        num_encoder_blocks
    )

    num_decoder_blocks = 3

    st_decoder = SpatioTemporalDecoder(
        in_features, hidden_features, out_features, num_joints,
        num_frames, num_frames_out,
        num_heads, use_skip_connection,
        num_decoder_blocks
    )

    # alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
    # seq_len is num_frames_out in decoder, so gotta use num_frames_out
    # first three dimensions set to 1 are for: batch, heads and time
    decoder_mask_s = causal_mask((1, 1, 1, num_joints, num_joints)).to(device)

    # since we are in the decoder, mask must have same shape of attn_t
    # REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
    # so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
    # first three dimensions set to 1 are for: batch, heads and time
    decoder_mask_t = causal_mask((1, 1, 1, num_frames_out, num_frames)).to(device)

    st_transformer = SpatioTemporalTransformer(
        st_encoder=st_encoder, st_decoder=st_decoder,
        num_frames=num_frames, num_joints=num_joints, in_features=in_features, 
    ).to(device)

    batch_size = 256
    # src = torch.rand((batch_size, in_features, num_frames, num_joints)).to(device)
    src = torch.rand((batch_size, num_frames    , num_joints, in_features)).to(device)
    tgt = torch.rand((batch_size, num_frames_out, num_joints, in_features)).to(device)

    decoder_output = st_transformer.forward(
        src=src, tgt=tgt,
        tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
    )
    print(f"decoder_output: {decoder_output.shape}")

if __name__ == "__main__":

    main()
    





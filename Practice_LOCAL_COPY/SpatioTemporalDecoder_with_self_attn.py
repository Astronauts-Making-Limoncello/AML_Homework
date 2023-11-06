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

class SpatioTemporalDecoder(nn.Module):
  
    def __init__(
        self, 
        in_features, hidden_features, out_features,
        spatio_temporal_self_attention_config,
        spatio_temporal_cross_attention_config
    ):
        super().__init__()

        assert len(spatio_temporal_self_attention_config) == len(spatio_temporal_cross_attention_config), "spatio-temporal self and cross attention configs must have the same number of configurations"

        self.x_fc_in              = nn.Linear(in_features, hidden_features)
        self.encoder_output_fc_in = nn.Linear(in_features, hidden_features)

        self.x_fc_out = nn.Linear(spatio_temporal_cross_attention_config[-1][1], out_features)

        self.blocks = nn.ModuleList()
        layer_id = 0
        
        for (
            self_attn_in_features, self_attn_out_feat, self_attn_num_frames, self_attn_num_frames_out, self_attn_num_heads
        ), (
            cross_attn_in_features, cross_attn_out_feat, cross_attn_num_frames, cross_attn_num_frames_out, cross_attn_num_heads
        ) in zip(
            spatio_temporal_self_attention_config, spatio_temporal_cross_attention_config
        ):
            
            self_attn_layer = SpatioTemporalCrossAttention(
                in_features=self_attn_in_features, out_features=self_attn_out_feat,
                # self-attention does NOT need temporal dimension explosion, because it 
                # uses decoder input (x) for q, k and v.
                # So, setting temporal explosion to explode to same temporal dimensionality
                num_frames=self_attn_num_frames_out, num_frames_out=self_attn_num_frames_out,
            )
            self.blocks.add_module(f"self_attn_{layer_id}", self_attn_layer)
            
            cross_attn_layer = SpatioTemporalCrossAttention(
                in_features=cross_attn_in_features, out_features=cross_attn_out_feat,
                num_frames=cross_attn_num_frames, num_frames_out=cross_attn_num_frames_out,
            )
            self.blocks.add_module(f"cross_attn_{layer_id}", cross_attn_layer)

            layer_id += 1
            
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)
            
            
    def forward(self, x, encoder_output):

        # print(f"\[SpatioTemporalDecoder] x.shape             : {x.shape}")
        # print(f"\[SpatioTemporalDecoder] encoder_output.shape: {encoder_output.shape}")

        x = self.x_fc_in(x)
        encoder_output = self.encoder_output_fc_in(encoder_output)

        # print(f"\[SpatioTemporalDecoder] x.shape             : {x.shape}")
        # print(f"\[SpatioTemporalDecoder] encoder_output.shape: {encoder_output.shape}")

        for layer_name, layer in self.blocks._modules.items():
            
            if layer_name.startswith("self_attn_"):
                # h = x
                # if self-attention, then q, k and v come from the decoder input (x)
                x = layer(x, x)
                # x += h
            
            elif layer_name.startswith("cross_attn_"):
                # h = x
                # if cross-attention, q comes from decoder input (x), k and v come from encoder_output
                x = layer(x, encoder_output)
                # x += h

        x = self.x_fc_out(x)

        return x

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))
    
    num_joints = 22
    num_frames = 10
    num_frames_out = 25

    in_features = 3
    hidden_features = 128
    out_features = 3

    # spatial_features_in, spatial_features_out, temporal_features_in, temporal_features_out, num_heads
    spatio_temporal_self_attention_config = [
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
    ]
    
    spatio_temporal_cross_attention_config = [
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
       [hidden_features, hidden_features, num_frames, num_frames_out, 1],
    ]

    spatio_temporal_transformer = SpatioTemporalDecoder(
        in_features, hidden_features, out_features,
        spatio_temporal_self_attention_config=spatio_temporal_self_attention_config,
        spatio_temporal_cross_attention_config=spatio_temporal_cross_attention_config
        # encoder_output_out_features=hidden_features,
        # out_features=out_features
    )

    batch_size = 256
    # encoder output must come already mapped to hidden features, because it is
    # supposed to come out of the encoder
    encoder_output = torch.rand((batch_size, num_frames    , num_joints, in_features))
    # decoder input must come with in_features, because it comes directly from the
    # dataset!
    x              = torch.rand((batch_size, num_frames_out, num_joints, in_features)) 

    x = spatio_temporal_transformer.forward(x, encoder_output)

    print(f"x.shape: {x.shape}")


if __name__ == "__main__":
    main()
    
    





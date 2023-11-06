import torch
import torch.nn as nn

from SpatioTemporalAttention import SpatioTemporalAttention

from rich import print

# Taken exactly as is from assignment original code.
# Uses STABlock (refactored to SpatioTemporalAttention for naming consistency )as encoding blocks, 
# which perform spatio-temporal self-attention
# Just removed the *_out layers, because these are used to expand the temporal dimension
# from num_frames to num_frames_out, which is not needed in our custom model.
# In fact, our custom model uses a dedicated cross-spatio-temporal-attention decoder to expand the temporal dimension.

def conv_init(conv):
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

DEFAULT_CONFIG = [
    [16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16], 
    [16, 16, 16], [16, 16, 16], [16, 16, 16], [16,  3, 16]    
]

class SpatioTemporalEncoder(nn.Module):
    def __init__(
        self, num_joints, num_frames, num_frames_out, num_heads, num_channels, out_features,
        kernel_size, len_parts=1, use_pes=True, config=DEFAULT_CONFIG, num_persons=1, 
        att_drop=0, dropout=0, dropout2d=0
    ):
        super().__init__()

        self.num_frames = num_frames
        self.num_joints = num_joints
        self.num_channels = num_channels
        self.num_persons = num_persons
        self.len_parts = len_parts
        in_channels = config[0][0]
        self.out_channels = config[-1][1]

        num_frames = num_frames // len_parts
        num_joints = num_joints * len_parts
        
        self.input_map = nn.Sequential(
            nn.Conv2d(num_channels, in_channels, 1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1)
        )

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(
                SpatioTemporalAttention(
                    in_channels, out_channels, qkv_dim, 
                    num_frames=num_frames, 
                    num_joints=num_joints, 
                    num_heads=num_heads,
                    kernel_size=kernel_size,
                    use_pes=use_pes,
                    att_drop=att_drop
                )
            )   
        
        # to map encoder output to desired out_feature dimensionality
        self.fc_out = nn.Linear(num_channels, out_features)
        
        # this should be needed at decoding time only, because that's where
        # we want to explode the temporal dimension
        # self.conv_out = nn.Conv1d(num_frames, num_frames_out, 1, stride=1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (before reshape): {x.shape}")
        x = x.reshape(-1, self.num_frames, self.num_joints, self.num_channels, self.num_persons).permute(0, 3, 1, 2, 4).contiguous()
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, in_features, num_frames, num_joints, 1) 
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after  reshape): {x.shape}")
        N, C, T, V, M = x.shape
        
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (before permute and view): {x.shape}")
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        # x.shape: (batch_size, in_features, num_frames, num_joints, 1) --> (batch_size, in_features, num_frames, num_joints)
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after  permute and view): {x.shape}")
        
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (before view): {x.shape}")
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, in_features, num_frames, num_joints) 
        # it seems strange, it's probably needed to handle some special case or something among those lines?
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after  view): {x.shape}")
        
        x = self.input_map(x)
        # x.shape: (batch_size, in_channels, num_frames, num_joints)
        # this probably maps original number of input features (3, i.e. x, y and z coordinates of the joints)
        # to a needed number of features/channels, specified in in_channels of class builder
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after self.input_map): {x.shape}")

        for i, block in enumerate(self.blocks):
            # print("-----------")
            # print(f"\[SpatioTemporalTransformerEncoder.forward] block {i}, x.shape (before block): {x.shape}")
            x = block(x)
            # print(f"\[SpatioTemporalTransformerEncoder.forward] block {i}, x.shape (after  block): {x.shape}")
            # print("-----------")
            # print("\n\n")

        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (before reshape): {x.shape}")
        # x = x.reshape(-1, self.num_frames, self.num_joints*self.num_channels)
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, num_frames, in_features * num_joints)
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after  reshape): {x.shape}")
        x = x.reshape(-1, self.num_frames, self.num_joints, self.num_channels)
        
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (before self.conv_out): {x.shape}")
        # expands num_frames dimension from num_frames input frames to num_frames_out output frames
        # it should NOT be needed during the encoding stage...
        # x = self.conv_out(x)
        # x.shape: (batch_size, num_frames_out, in_features * num_joints)
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after  self.conv_out): {x.shape}")

        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (before self.fc_out): {x.shape}")
        # it should be NOT needed during the encoding stage...
        # out = self.fc_out(x)  
        # x.shape: (batch_size, num_frames_out, in_features * num_joints)
        # print(f"\[SpatioTemporalTransformerEncoder.forward] x.shape (after  self.fc_out): {x.shape}")

        x = self.fc_out(x)
        
        return x

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))
    
    num_joints = 22
    num_frames = 10
    num_frames_out = 25

    config = [
        [16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16], 
        [16, 16, 16], [16, 16, 16], [16, 16, 16], [16,  3, 16]    
    ]

    in_features = 3
    out_features = 3
    
    spatio_temporal_transformer_encoder = SpatioTemporalEncoder(
        num_joints=num_joints, num_frames=num_frames, num_frames_out=num_frames_out, 
        num_heads=8, num_channels=in_features, kernel_size=[3, 3], config=config, 
        out_features = out_features
    ).to(device)
    
    batch_size = 256
    x = torch.rand((256, 3, num_frames, num_joints)).to(device)

    print(f"\[SpatioTemporalTransformerEncoder.main] x.shape             : {x.shape}")
    encoder_output = spatio_temporal_transformer_encoder(x)
    print(f"\[SpatioTemporalTransformerEncoder.main] encoder_output.shape: {encoder_output.shape}")




import torch.nn as nn
from .sta_block import STA_Block

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


class Model(nn.Module):
    def __init__(self, num_joints, 
                 num_frames, num_frames_out, num_heads, num_channels, 
                 kernel_size, len_parts=1, use_pes=True, config=None, num_persons=1,
                 att_drop=0, dropout=0, dropout2d=0):
        super().__init__()

        config = [
            [16,16,  16], [16,16,  16], 
            [16,16,  16], [16,16,  16],
            [16,  16,  16], [16,16,  16], 
            [16,16,  16], 
            [16,3,  16]    
        ]

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
            nn.LeakyReLU(0.1))

        self.blocks = nn.ModuleList()
        for index, (in_channels, out_channels, qkv_dim) in enumerate(config):
            self.blocks.append(STA_Block(in_channels, out_channels, qkv_dim, 
                                         num_frames=num_frames, 
                                         num_joints=num_joints, 
                                         num_heads=num_heads,
                                         kernel_size=kernel_size,
                                         use_pes=use_pes,
                                         att_drop=att_drop))   
        self.fc_out = nn.Linear(66, 66)
        self.conv_out = nn.Conv1d(num_frames, num_frames_out, 1, stride=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
            elif isinstance(m, nn.Linear):
                fc_init(m)

    def forward(self, x):
        # print(f"\[STTformer.forward] x.shape (before reshape): {x.shape}")
        x = x.reshape(-1, self.num_frames, self.num_joints, self.num_channels, self.num_persons).permute(0, 3, 1, 2, 4).contiguous()
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, in_features, num_frames, num_joints, 1) 
        # print(f"\[STTformer.forward] x.shape (after  reshape): {x.shape}")
        N, C, T, V, M = x.shape
        
        # print(f"\[STTformer.forward] x.shape (before permute and view): {x.shape}")
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        # x.shape: (batch_size, in_features, num_frames, num_joints, 1) --> (batch_size, in_features, num_frames, num_joints)
        # print(f"\[STTformer.forward] x.shape (after  permute and view): {x.shape}")
        
        # print(f"\[STTformer.forward] x.shape (before view): {x.shape}")
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, in_features, num_frames, num_joints) 
        # it seems strange, it's probably needed to handle some special case or something among those lines?
        # print(f"\[STTformer.forward] x.shape (after  view): {x.shape}")
        
        x = self.input_map(x)
        # x.shape: (batch_size, in_channels, num_frames, num_joints)
        # this probably maps original number of input features (3, i.e. x, y and z coordinates of the joints)
        # to a needed number of features/channels, specified in in_channels of class builder
        # print(f"\[STTformer.forward] x.shape (after self.input_map): {x.shape}")

        
        
        for i, block in enumerate(self.blocks):
            # print(f"\[STTformer.forward] block {i}, x.shape (before block): {x.shape}")
            x = block(x)
            # print(f"\[STTformer.forward] block {i}, x.shape (after  block): {x.shape}")

        # print(f"\[STTformer.forward] x.shape (before reshape): {x.shape}")
        x = x.reshape(-1, self.num_frames, self.num_joints*self.num_channels)
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, num_frames, in_features * num_joints)
        # print(f"\[STTformer.forward] x.shape (after  reshape): {x.shape}")
        
        # print(f"\[STTformer.forward] x.shape (before self.conv_out): {x.shape}")
        # expands num_frames dimension from num_frames input frames to num_frames_out output frames
        x = self.conv_out(x)
        # x.shape: (batch_size, num_frames_out, in_features * num_joints)
        # print(f"\[STTformer.forward] x.shape (after  self.conv_out): {x.shape}")

        # print(f"\[STTformer.forward] x.shape (before self.fc_out): {x.shape}")
        out = self.fc_out(x)  
        # x.shape: (batch_size, num_frames_out, in_features * num_joints)
        # print(f"\[STTformer.forward] x.shape (after  self.fc_out): {x.shape}")
        
        return out
import torch
import torch.nn as nn

from models.STCFormer.SpatioTemporalAttention import SpatioTemporalAttention

from rich import print

# Taken exactly as is from assignment original code.
# Uses STABlock (refactored to SpatioTemporalAttention for naming consistency )as encoding blocks, 
# which perform spatio-temporal self-attention
# Just removed the *_out layers, because these are used to expand the temporal dimension
# from num_frames to num_frames_out, which is not needed in our custom model.
# In fact, our custom model uses a dedicated cross-spatio-temporal-attention decoder to expand the temporal dimension.

def conv_init(conv):
    """
    Initialize the given convolutional layer using the Kaiming normal initialization method.

    Parameters:
        conv (nn.Module): The convolutional layer to be initialized.

    Returns:
        None
    """
    nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    # nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    """
    Initialize the given BatchNorm layer with the specified scale.

    Parameters:
        bn (nn.BatchNorm2d): The BatchNorm layer to initialize.
        scale (float): The scale value for the weight initialization.

    Returns:
        None
    """
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def fc_init(fc):
    """
    Initialize the given fully connected layer.

    Parameters:
        fc (nn.Linear): The fully connected layer to initialize.

    Returns:
        None
    """
    nn.init.xavier_normal_(fc.weight)
    nn.init.constant_(fc.bias, 0)

DEFAULT_CONFIG = [
    [16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16], 
    [16, 16, 16], [16, 16, 16], [16, 16, 16], [16,  3, 16]    
]

class STAGEncoder(nn.Module):
    def __init__(
        self, num_joints, num_frames, num_frames_out, num_heads, num_channels, out_features,
        kernel_size, len_parts=1, use_pes=True, config=DEFAULT_CONFIG, num_persons=1, 
        att_drop=0, dropout=0, dropout2d=0
    ):
        """
        Initializes the object with the given parameters.

        Args:
            num_joints (int): The number of joints.
            num_frames (int): The number of frames.
            num_frames_out (int): The number of output frames.
            num_heads (int): The number of attention heads.
            num_channels (int): The number of channels.
            out_features (int): The number of output features.
            kernel_size (int): The size of the kernel.
            len_parts (int, optional): The length of the parts. Defaults to 1.
            use_pes (bool, optional): Whether to use positional encoding. Defaults to True.
            config (dict, optional): The configuration. Defaults to DEFAULT_CONFIG.
            num_persons (int, optional): The number of persons. Defaults to 1.
            att_drop (float, optional): The attention dropout. Defaults to 0.
            dropout (float, optional): The dropout rate. Defaults to 0.
            dropout2d (float, optional): The 2D dropout rate. Defaults to 0.

        Returns:
            None
        """
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

    def forward(self, encoder_input, mask_s, mask_t):
        """
        Forward pass of the SpatioTemporalTransformerEncoder module.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features, num_frames, num_joints).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_frames, num_joints, num_channels).
        """

        x = encoder_input
        
        x = x.reshape(-1, self.num_frames, self.num_joints, self.num_channels, self.num_persons).permute(0, 3, 1, 2, 4).contiguous()
        N, C, T, V, M = x.shape
        
        x = x.permute(0, 4, 1, 2, 3).contiguous().view(N * M, C, T, V)
        
        x = x.view(x.size(0), x.size(1), T // self.len_parts, V * self.len_parts)
        # x.shape: (batch_size, in_features, num_frames, num_joints) --> (batch_size, in_features, num_frames, num_joints) 
        
        x = self.input_map(x)
        # x.shape: (batch_size, in_channels, num_frames, num_joints)
        # this probably maps original number of input features (3, i.e. x, y and z coordinates of the joints)
        # to a needed number of features/channels, specified in in_channels of class builder

        for i, block in enumerate(self.blocks):
            x = block(x)
        x = x.reshape(-1, self.num_frames, self.num_joints, self.num_channels)
        x = self.fc_out(x)
        
        return x
    
def main():
    """
    Initializes the main function.

    This function sets up the device based on the availability of CUDA.
    It also initializes the number of joints, frames, and output frames.
    The configuration for the SpatioTemporalEncoder is specified.
    The input and output features are set.
    The SpatioTemporalEncoder is initialized and moved to the device.
    The batch size is set and the input tensor is generated.
    The shape of the input tensor is printed.
    Finally, the input tensor is passed through the SpatioTemporalEncoder and the shape of the output is printed.

    Returns:
        None
    """
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
    
    spatio_temporal_transformer_encoder = STAGEncoder(
        num_joints=num_joints, num_frames=num_frames, num_frames_out=num_frames_out, 
        num_heads=8, num_channels=in_features, kernel_size=[3, 3], config=config, 
        out_features = out_features
    ).to(device)
    
    batch_size = 256
    x = torch.rand((256, 3, num_frames, num_joints)).to(device)

    print(f"\[SpatioTemporalTransformerEncoder.main] x.shape             : {x.shape}")
    encoder_output = spatio_temporal_transformer_encoder(x)
    print(f"\[SpatioTemporalTransformerEncoder.main] encoder_output.shape: {encoder_output.shape}")


if __name__ == "__main__":
    main()



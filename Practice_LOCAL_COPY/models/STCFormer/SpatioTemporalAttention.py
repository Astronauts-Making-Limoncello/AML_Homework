import torch
import torch.nn as nn
from pos_embed import Pos_Embed

from rich import print

# Taken exactly as is from assignment original code, just renamed the class to uniform names of custom model modules

class SpatioTemporalAttention(nn.Module):
    def __init__(self, in_channels, out_channels, qkv_dim,
                 num_frames, num_joints, num_heads,
                 kernel_size, use_pes=True, att_drop=0):
        """
        Initializes the object.

        Args:
            in_channels (int): The number of input channels.
            out_channels (int): The number of output channels.
            qkv_dim (int): The dimensionality of the query, key, and value vectors.
            num_frames (int): The number of frames.
            num_joints (int): The number of joints.
            num_heads (int): The number of attention heads.
            kernel_size (tuple): The size of the kernel.
            use_pes (bool, optional): Whether to use position embeddings. Defaults to True.
            att_drop (float, optional): The dropout rate for attention weights. Defaults to 0.

        Returns:
            None
        """
        super().__init__()
        self.qkv_dim = qkv_dim
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.use_pes = use_pes
        pads = int((kernel_size[1] - 1) / 2)
        padt = int((kernel_size[0] - 1) / 2)
        
        # Spatio-Temporal Tuples Attention
        if self.use_pes: self.pes = Pos_Embed(in_channels, num_frames, num_joints)
        self.to_qkvs = nn.Conv2d(in_channels, 2 * num_heads * qkv_dim, 1, bias=True)
        self.alphas = nn.Parameter(torch.ones(1, num_heads, 1, 1), requires_grad=True)
        self.att0s = nn.Parameter(torch.ones(1, num_heads, num_joints, num_joints) / num_joints, requires_grad=True)
        self.out_nets = nn.Sequential(nn.Conv2d(in_channels * num_heads, out_channels, (1, kernel_size[1]), padding=(0, pads)), nn.BatchNorm2d(out_channels))
        self.ff_net = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))

        # Inter-Frame Feature Aggregation
        self.out_nett = nn.Sequential(nn.Conv2d(out_channels, out_channels, (kernel_size[0], 1), padding=(padt, 0)), nn.BatchNorm2d(out_channels))

        if in_channels != out_channels:
            self.ress = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
            self.rest = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.ress = lambda x: x
            self.rest = lambda x: x

        self.tan = nn.Tanh()
        self.relu = nn.LeakyReLU(0.1)
        self.drop = nn.Dropout(att_drop)

    def forward(self, x):
        """
        Forward pass of the SpatioTemporalAttention module.

        Args:
            x (torch.Tensor): Input tensor of shape (N, C, T, V).

        Returns:
            torch.Tensor: Tensor of shape (N, C, T, V) representing the output of the module.
        """

        N, C, T, V = x.size()
        # Spatio-Temporal Tuples Attention
        xs = self.pes(x) + x if self.use_pes else x
        q, k = torch.chunk(self.to_qkvs(xs).view(N, 2 * self.num_heads, self.qkv_dim, T, V), 2, dim=1)
       
        attention = self.tan(torch.einsum('nhctu,nhctv->nhuv', [q, k]) / (self.qkv_dim * T)) * self.alphas
        attention = attention + self.att0s.repeat(N, 1, 1, 1)
        attention = self.drop(attention)
        
        xs = torch.einsum('nctu,nhuv->nhctv', [x, attention])
        xs = xs.contiguous().view(N, self.num_heads * self.in_channels, T, V)
    
        x_ress = self.ress(x)
        xs = self.relu(self.out_nets(xs) + x_ress)
        xs = self.relu(self.ff_net(xs) + x_ress)

        # Inter-Frame Feature Aggregation
        xt = self.relu(self.out_nett(xs) + self.rest(xs))

        return xt
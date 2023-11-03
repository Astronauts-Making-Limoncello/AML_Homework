import torch
from torch import nn

from GraphAttentionLayer import GraphAttentionLayer

class ConvSTGATBlock(nn.Module):
  
  def __init__(
    self, 
    GAT_in_features, GAT_out_features, GAT_n_heads, # GAT layer config params
    conv_3D_in_channels, conv_3D_out_channels, conv_3D_kernel_size, # conv3d config params
    conv_2D_in_channels, conv_2D_out_channels, conv_2D_kernel_size, # conv2d config params
    concat: bool = True, 
    dropout: float = 0.4, 
    leaky_relu_slope: float = 0.2,
  ):
    super(ConvSTGATBlock, self).__init__()

    self.GAT_layer = GraphAttentionLayer(
      in_features=GAT_in_features, out_features=GAT_out_features,
      n_heads=GAT_n_heads, concat=concat, 
      dropout=dropout, leaky_relu_slope=leaky_relu_slope
    )

    self.conv_3D = nn.Conv3d(
      in_channels=conv_3D_in_channels, out_channels=conv_3D_out_channels, 
      kernel_size=conv_3D_kernel_size
    )

    self.conv_2D = nn.Conv2d(
      in_channels=conv_2D_in_channels, out_channels=conv_2D_out_channels, 
      kernel_size=conv_2D_kernel_size
    )

  def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
    h_prime = h

    print(f"[STGAT, forward] h_prime.shape (before GAT layer): {h_prime.shape}")
    h_prime = self.GAT_layer.forward(h_prime, adj_mat)
    print(f"[STGAT, forward] h_prime.shape (after  GAT layer): {h_prime.shape}")

    h_prime = self.conv_2D(h_prime)
    print(f"[STGAT, forward] h_prime.shape (after    conv_2D): {h_prime.shape}")

    h_prime = torch.unsqueeze(h_prime, 1) # batch, 1, time, joints, coordinates
    print(f"[STGAT, forward] h_prime.shape (after  unsqueeze): {h_prime.shape}")

    h_prime = self.conv_3D(h_prime)
    print(f"[STGAT, forward] h_prime.shape (after    conv_3D): {h_prime.shape}")

    h_prime = torch.squeeze(h_prime, 1) # batch, 1, time, joints, coordinates
    print(f"[STGAT, forward] h_prime.shape (after    squeeze): {h_prime.shape}")

    return h_prime
  




def main():

  connect = [
    (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10),
    (0, 1), (0, 6), (6, 17), (17, 18), 
    (18, 19), (19, 20), (20, 21), (21, 22),
    (1, 25), (25, 26), (26, 27), (27, 28), 
    (28, 29), (29, 30), (24, 25), (24, 17),
    (24, 14), (14, 15)
  ]
  
  n_nodes = 32 # N (also called joints at the end of the notebook)

  adj_mat = torch.zeros(n_nodes, n_nodes, dtype=torch.int)
  for edge in connect:
    adj_mat[edge[0], edge[1]] = 1
    adj_mat[edge[1], edge[0]] = 1  # If the graph is undirected

  STGAT_block = ConvSTGATBlock(
    GAT_in_features=3, GAT_out_features=128, GAT_n_heads=1,
    conv_3D_in_channels=1, conv_3D_out_channels=1, conv_3D_kernel_size=2,
    conv_2D_in_channels=10, conv_2D_out_channels=25, conv_2D_kernel_size=3
  )

  x = torch.rand((256, 10, 32, 3))
  print(f"[ConvSTGATBlock, main] x.shape (init): {x.shape}")

  x = STGAT_block.forward(x, adj_mat)

  # print(f"[ConvSTGATBlock, main] x.shape (after forward): {x.shape}")




if __name__ == "__main__":
    main()


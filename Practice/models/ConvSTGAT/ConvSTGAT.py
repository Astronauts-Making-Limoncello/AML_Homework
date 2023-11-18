import torch
from torch import nn

from ConvSTGATBlock import ConvSTGATBlock

class STGAT(nn.Module):
  
  def __init__(self, stgat_blocks: list[nn.Module]):
    super(STGAT, self).__init__()

    self.stgat = nn.ModuleList(stgat_blocks)


  def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
    h_prime = h

    for block_id, block in enumerate(self.stgat):
      print(f"[STGAT, forward] block {block_id}")

      h_prime = block.forward(
        h_prime, 
        adj_mat if block_id == 0 else torch.ones(h_prime.shape[-2], h_prime.shape[-2])
      )

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

  STGAT_block_0 = ConvSTGATBlock(
    GAT_in_features=3, GAT_out_features=128, GAT_n_heads=1,
    conv_2D_in_channels=10, conv_2D_out_channels=16, conv_2D_kernel_size=4,
    conv_3D_in_channels=1, conv_3D_out_channels=1, conv_3D_kernel_size=2,
  )
  
  STGAT_block_1 = ConvSTGATBlock(
    GAT_in_features=124, GAT_out_features=64, GAT_n_heads=1,
    conv_2D_in_channels=15, conv_2D_out_channels=21, conv_2D_kernel_size=5,
    conv_3D_in_channels=1, conv_3D_out_channels=1, conv_3D_kernel_size=3,
  )
  
  STGAT_block_2 = ConvSTGATBlock(
    GAT_in_features=61, GAT_out_features=7, GAT_n_heads=1,
    conv_2D_in_channels=19, conv_2D_out_channels=25, conv_2D_kernel_size=2,
    conv_3D_in_channels=1, conv_3D_out_channels=1, conv_3D_kernel_size=2,
  )

  x = torch.rand((256, 10, 32, 3))
  print(f"[STGAT, main] x.shape (init): {x.shape}")

  # stgat = STGAT([STGAT_block_0, STGAT_block_1, STGAT_block_2])
  stgat = STGAT([STGAT_block_0, STGAT_block_1])

  stgat.forward(x, adj_mat)


if __name__ == "__main__":
  main()


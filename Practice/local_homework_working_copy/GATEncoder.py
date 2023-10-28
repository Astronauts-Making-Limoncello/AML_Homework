import torch
from torch import nn

from GraphAttentionLayer import GraphAttentionLayer

class GATEncoder(nn.Module):
  
  def __init__(
    self, 
    GAT_config: list,
    channel_config: list,
    concat: bool = True, 
    dropout: float = 0.4, 
    leaky_relu_slope: float = 0.2,
  ):
    super(GATEncoder, self).__init__()

    if len(GAT_config) != len(channel_config):
      raise ValueError("GAT_config len different than channel_config len")

    self.GAT_encoder = nn.ModuleList()

    for i, (in_features, out_features, n_heads) in enumerate(GAT_config):
      # in_features --> F in GAT paper
      # out_features --> F' in GAT paper
      
      self.GAT_encoder.append(
        GraphAttentionLayer(
          in_features=in_features,
          out_features=out_features,
          n_heads=n_heads, concat=concat,
          dropout=dropout, leaky_relu_slope=leaky_relu_slope
        )
      )

      break

      self.GAT_encoder.append(
        nn.Conv2d(
          in_channels=channel_config[i][0], 
          out_channels=channel_config[i][1], 
          kernel_size=1
        )
      )

  def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):

    h_prime = h

    for gat_enc_layer in self.GAT_encoder:

      if isinstance(gat_enc_layer, GraphAttentionLayer):
        h_prime = gat_enc_layer.forward(h=h_prime, adj_mat=adj_mat)
        print(f"h_prime.shape: {h_prime.shape}")
      
      elif isinstance(gat_enc_layer, nn.Conv2d):
        h_prime = gat_enc_layer.forward(h_prime)
      
      else:
        raise ValueError(f"Unsupported layer type: {type(gat_enc_layer)}")


    return h_prime
  




def main():
  batch_size = 256
  in_features = 3 # F
  out_features = 128 # F'
  n_nodes = 32 # N (also called joints at the end of the notebook)
  n_input = 10 # number of frames in the input
  n_output = 25 # number of frames in the output (to predict)
  n_latent = 15 # number of frames in the latent representation

  h = torch.rand(size=(batch_size, n_input, n_nodes, in_features))

  connect = [
    (1, 2), (2, 3), (3, 4), (4, 5),
    (6, 7), (7, 8), (8, 9), (9, 10),
    (0, 1), (0, 6), (6, 17), (17, 18), 
    (18, 19), (19, 20), (20, 21), (21, 22),
    (1, 25), (25, 26), (26, 27), (27, 28), 
    (28, 29), (29, 30), (24, 25), (24, 17),
    (24, 14), (14, 15)
  ]
  adj_mat = torch.zeros(n_nodes, n_nodes, dtype=torch.int)
  for edge in connect:
    adj_mat[edge[0], edge[1]] = 1
    adj_mat[edge[1], edge[0]] = 1  # If the graph is undirected

  # configuration shapes for GAT and conv layers.
  # for GAT we can decide the F-->F' mapping dimensions, as per original paper
  # for conv we use the channels to "index" the number of predicted frames, similar to STABlock class

  # (in_features, out_features, n_heads) for each GAT layer
  # make sure that out_features % n_heads == 0
  GAT_config = [
    (in_features, 20, 4),
    (20, 40, 8),
    (40 , out_features, 2)
  ]

  # in_channels and out_channels for each conv2D layer
  channel_config = [
    [n_input, 2],
    [2, 4],
    [4, n_latent]
  ]
  
  enc = GATEncoder(GAT_config, channel_config)

  h_prime = enc.forward(h, adj_mat)

  print(f"h_prime.shape: {h_prime.shape}")


if __name__ == "__main__":
    main()


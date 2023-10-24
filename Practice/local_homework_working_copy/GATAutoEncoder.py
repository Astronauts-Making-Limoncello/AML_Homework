import torch
from torch import nn

from GATEncoder import GATEncoder
from GATDecoder import GATDecoder

class GATAutoEncoder(nn.Module):
  
  def __init__(self, enc: GATEncoder, dec: GATDecoder):
    super(GATAutoEncoder, self).__init__()

    self.enc = enc
    self.dec = dec

  def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):

    h_latent = self.enc(h, adj_mat)
    print(f"h_latent.shape: {h_latent.shape}")
    h_tilde= self.dec(h_latent, adj_mat)

    return h_tilde


def main():
  batch_size = 256
  in_features = 3 # F
  lat_features = 128 # F'
  out_features = 3 # F
  n_nodes = 32 # N (also called joints at the end of the notebook)
  n_input = 10
  n_latent = 15
  n_output = 25

  h = torch.rand(size=(batch_size, n_input, n_nodes, in_features))

  connect = [
    (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (9, 10), (0, 1), 
    (0, 6), (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (1, 25), 
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (24, 25), (24, 17), 
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
  GAT_config_enc = [
    (in_features, 20, 4),
    (20, 40, 8),
    (40 , lat_features, 2)
  ]

  # in_channels and out_channels for each conv2D layer
  channel_config_enc = [
    [n_input, 2],
    [2, 4],
    [4, n_latent]
  ]
  
  enc = GATEncoder(GAT_config_enc, channel_config_enc)

  # (in_features, out_features, n_heads) for each GAT layer
  # make sure that out_features % n_heads == 0
  GAT_config_dec = [
    (lat_features, 20, 4),
    (20, 40, 8),
    (40 , out_features, 1)
  ]

  # in_channels and out_channels for each conv2D layer
  channel_config_dec = [
    [n_latent, 2],
    [2, 4],
    [4, n_output]
  ]
  
  dec = GATDecoder(GAT_config_dec, channel_config_dec)  

  autoenc = GATAutoEncoder(enc, dec)

  h_tilde = autoenc.forward(h, adj_mat)

  print(f"h_tilde.shape: {h_tilde.shape}")


if __name__ == "__main__":
  main()
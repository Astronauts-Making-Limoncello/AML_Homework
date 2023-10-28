import torch
from torch import nn
import torch.nn.functional as F

################################
### GAT LAYER DEFINITION ###
################################

# taken from official paper implementation BUT added support for batch dimension

class GraphAttentionLayer(nn.Module):

 def __init__(self, in_features: int, out_features: int, n_heads: int, concat: bool = False, dropout: float = 0.4, leaky_relu_slope: float = 0.2):
    super(GraphAttentionLayer, self).__init__()

    self.n_heads = n_heads # Number of attention heads
    self.concat = concat # wether to concatenate the final attention heads
    self.dropout = dropout # Dropout rate

    if concat: # concatenating the attention heads
      self.out_features = out_features # Number of output features per node
      assert out_features % n_heads == 0 # Ensure that out_features is a multiple of n_heads
      self.n_hidden = out_features // n_heads
    else: # averaging output over the attention heads (Used in the main paper)
      self.n_hidden = out_features

    # A shared linear transformation, parametrized by a weight matrix W is applied to every node
    # Initialize the weight matrix W
    self.W = nn.Parameter(torch.empty(size=(in_features, self.n_hidden * n_heads)))

    # Initialize the attention weights a
    self.a = nn.Parameter(torch.empty(size=(n_heads, 2 * self.n_hidden, 1)))

    self.leakyrelu = nn.LeakyReLU(leaky_relu_slope) # LeakyReLU activation function
    self.softmax = nn.Softmax(dim=1) # softmax activation function to the attention coefficients

    self.reset_parameters() # Reset the parameters


 def reset_parameters(self):

    nn.init.xavier_normal_(self.W)
    nn.init.xavier_normal_(self.a)

 def _get_attention_scores(self, h_transformed: torch.Tensor):

    # print(f"[_get_attention_scores] a.shape: {self.a.shape}") # n_heads, n_nodes, 1
    source_scores = torch.matmul(h_transformed, self.a[:, :self.n_hidden, :])
    # print(f"[_get_attention_scores] source_scores.shape: {source_scores.shape}") # batch_size, n_input (temporal dim), n_heads, n_nodes (spacial dimension), 1
    target_scores = torch.matmul(h_transformed, self.a[:, self.n_hidden:, :])
    # print(f"[_get_attention_scores] target_scores.shape: {target_scores.shape}") # batch_size, n_input (temporal dim), n_heads, n_nodes (spacial dimension), 1

    # broadcast add
    # (n_heads, n_nodes, 1) + (n_heads, 1, n_nodes) = (n_heads, n_nodes, n_nodes)
    e = source_scores + target_scores.mT
    # print(f"[_get_attention_scores] e.shape: {e.shape}") # batch, n_input (temporal dim), n_heads, n_nodes (spacial dim), n_nodes (spacial dim)
    return self.leakyrelu(e)

 def forward(self, h: torch.Tensor, adj_mat: torch.Tensor):
    batch_size, n_input, n_nodes, in_features = h.shape

    # print(f"h.shape: {h.shape}")
    # print(f"self.W.shape: {self.W.shape}")


    # Apply linear transformation to node feature -> W h
    # print(f"[GATLayer] h.shape: {h.shape}") # batch_size, n_input (temporal dim), n_nodes (spacial dim/skeleton joints), in_features (F in original GAT paper)
    # print(f"[GATLayer] W.shape: {self.W.shape}") # in_features (F in original GAT paper), out_features (F' in original GAT paper)
    h_transformed = torch.einsum("btjs,sh->btjh", h, self.W)
    # b --> batch dimension
    # t --> temporal dimension (n_input)
    # j --> spacial dimension (n_nodes)
    # s --> dimensionality of spacial dimension (in_features, F in original GAT paper)
    # h --> dimensionality of spacial dimension (out_features, F' in original GAT paper)
    # print(f"[GATLayer] h_transformed.shape (einsum): {h_transformed.shape}") # batch_size, n_input (temporal dim), n_nodes (spacial dim/skeleton joints), out_features
    h_transformed = F.dropout(h_transformed, self.dropout, training=self.training)
    
    # splitting the heads by reshaping the tensor and putting heads dim first
    h_transformed = h_transformed.view(batch_size, n_input, n_nodes, self.n_heads, self.n_hidden)
    # print(f"[GATLayer] h_transformed (view).shape: {h_transformed.shape}") 
    h_transformed = h_transformed.permute(0, 1, 3, 2, 4)
    # print(f"[GATLayer] h_transformed (permute).shape: {h_transformed.shape}")

    # getting the attention scores
    # output shape (n_heads, n_nodes, n_nodes)
    e = self._get_attention_scores(h_transformed)

    # Set the attention score for non-existent edges to -9e15 (MASKING NON-EXISTENT EDGES)
    connectivity_mask = -9e16 * torch.ones_like(e)
    e = torch.where(adj_mat > 0, e, connectivity_mask) # masked attention scores
    # print(f"[GATLayer] e.shape: {e.shape}") # batch, n_input (temporal dim), n_heads, n_nodes (spacial dim), n_nodes (spacial dim)

    # attention coefficients are computed as a softmax over the rows
    # for each column j in the attention score matrix e
    attention = F.softmax(e, dim=-1)
    attention = F.dropout(attention, self.dropout, training=self.training)

    # final node embeddings are computed as a weighted average of the features of its neighbors
    h_prime = torch.matmul(attention, h_transformed)
    # print(f"[GATLayer] h_prime.shape: {h_prime.shape}") # batch_size, n_input (temporal dim), n_heads, n_nodes (spacial dim), n_hidden

    # concatenating/averaging the attention heads
    # output shape (n_nodes, out_features)
    if self.concat:
      # h_prime = h_prime.permute(0, 2, 1, 3).contiguous().view(batch_size, n_nodes, self.out_features)
      h_prime = h_prime.view(batch_size, n_input, n_nodes, self.out_features)
      # print(f"h_prime.shape: {h_prime.shape}")
    else:
      # TODO add support for batch size here, this version doesn't support it!
      h_prime = h_prime.mean(dim=0)

    # print(f"[GATLayer] h_prime.shape: {h_prime.shape}") # batch_size, n_input (temporal dim), n_nodes (spacial dim), out_features
    return h_prime
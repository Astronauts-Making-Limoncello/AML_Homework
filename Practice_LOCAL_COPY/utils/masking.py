import torch

def causal_mask(mask_shape):
  mask = torch.triu(torch.ones(mask_shape), diagonal=1).type(torch.int)
  
  return mask == 0
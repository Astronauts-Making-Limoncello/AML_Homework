# https://www.comet.com/site/blog/explainable-ai-for-transformers/

import torch
from torch.nn.functional import softmax

from rich import print
from rich.pretty import pprint

import seaborn as sns
import matplotlib.pyplot as plt

import os

TRAIN_ID = f"2023_11_07_18_22_25"
CKPT_PATH = f"../checkpoints/{TRAIN_ID}/h36m_3d_25frames_ckpt_best_val_loss.pt"
imgs_dir = f"../images/{TRAIN_ID}"

ckpt = torch.load(CKPT_PATH)

model = dict(ckpt["model_state_dict"])

encoder_q = []
encoder_k = []

decoder_q = []
decoder_k = []

for layer_name in model.keys():
  if "encoder" in layer_name:
    
    if "Wq.weight" in layer_name:
      encoder_q.append(layer_name)

    if "Wk.weight" in layer_name:
      encoder_k.append(layer_name)
  
  if "decoder" in layer_name:
    
    if "Wq.weight" in layer_name:
      decoder_q.append(layer_name)

    if "Wk.weight" in layer_name:
      decoder_k.append(layer_name)

os.makedirs(imgs_dir) if not os.path.exists(imgs_dir) else None

for block_id, (q_matrix_name, k_matrix_name) in enumerate(zip(encoder_q, encoder_k)):

  print(block_id)

  q = model[q_matrix_name]
  k = model[k_matrix_name]

  q_s, q_t = torch.chunk(q, chunks=2, dim=-1)
  k_s, k_t = torch.chunk(k, chunks=2, dim=-1)

  attn_s = torch.matmul(q_s, k_s.T)
  attn_s = softmax(attn_s, dim=-1)
  attn_s = attn_s.cpu()

  heatmap = sns.heatmap(data=attn_s, cmap="Blues")
  
  figure = heatmap.get_figure()    
  figure.savefig(f"{imgs_dir}/attn_s_encoder_block_{block_id}.png", dpi=200)

  plt.clf()
  
  attn_t = torch.matmul(q_t, k_t.T)
  attn_t = softmax(attn_t, dim=-1)
  attn_t = attn_t.cpu()

  heatmap = sns.heatmap(data=attn_t, cmap="Blues")
  
  figure = heatmap.get_figure()    
  figure.savefig(f"{imgs_dir}/attn_t_encoder_block_{block_id}.png", dpi=200)

  plt.clf()

  heatmap = sns.heatmap(data=(attn_t - attn_s), cmap="Blues")
  
  figure = heatmap.get_figure()    
  figure.savefig(f"{imgs_dir}/attn_t_minus_s_encoder_block_{block_id}.png", dpi=200)

  plt.clf()




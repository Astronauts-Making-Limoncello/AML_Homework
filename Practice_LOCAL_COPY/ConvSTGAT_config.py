import numpy as np
import torch

# # Arguments to setup the datasets
datas = 'h36m' # dataset name
path = './data/h3.6m/h3.6m/dataset'
input_n=10 # number of frames to train on (default=10)
output_n=25 # number of frames to predict on
input_dim=3 # dimensions of the input coordinates(default=3)
skip_rate=1 # # skip rate of frames
joints_to_consider_n=22

#FLAGS FOR THE TRAINING
mode='train' #choose either train or test mode

batch_size_test=8
model_path= './checkpoints/' # path to the model checkpoint file

actions_to_consider_test='all' # actions to test on.
model_name = datas+'_3d_'+str(output_n)+'frames_ckpt' #the model name to save/load

#FLAGS FOR THE VISUALIZATION
actions_to_consider_viz='all' # actions to visualize
visualize_from='test'
n_viz=2

# actions_to_consider_train = ["walking"]
actions_to_consider_train = None

# human skeleton

dim_used = np.array(
  [
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92
  ]
)

# every joint has three coordinates
n_joints = len(dim_used)//3

# all coordinate triples, for all joints, are put together in one big list (dim_used)
# so, to get the joint ids that are actually used
# we need to divide all values by the integer part of three
# in fact, joint 2 has its coordinates in positions 6, 7 and 8, 
# joint 5 has its coordinates in positions 15, 16 and 17, etc.
# we understood this by looking at dim_used and the "joint_to_ignore" array below
# noticing that values in joint_to_ignore * 3 were absent from dim_used.
joint_to_consider_ids = set(dim_used//3)

# we mapped the original joint number to the number of used joints.
# for example, if we have the following used joints: 0, 1, 4, 5, 9
# we want that 0 --> 0, 1 --> 1, 4 --> 2, 5 --> 3, 9 --> 4, etc.
# in order to create a dense adjacency matrix, when representing the skeleton as a graph 
joints_to_consider_ids_remapped = {
    k: v for k, v in zip(joint_to_consider_ids, range(len(joint_to_consider_ids))) 
}

# print(f"joint_to_consider_ids: {joint_to_consider_ids}")
# print(f"joints_to_consider_ids_remapped: {joints_to_consider_ids_remapped}")
# print(f"len(joint_to_consider_ids): {len(joint_to_consider_ids)}")

# pairs of joints that are connected to actually create a skeleton
# uses original IDs, NOT the ones in joints_to_consider_ids_remapped
connect = [
  (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (9, 10), (0, 1), 
  (0, 6), (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (1, 25), 
  (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (24, 25), (24, 17), 
  (24, 14), (14, 15)
]

# mapping connections from original IDs to our IDs
connect = [c for c in connect if c[0] in joint_to_consider_ids and c[1] in joint_to_consider_ids]

# creating the actual adjacency matrix
adj_mat = torch.zeros(n_joints, n_joints, dtype=torch.int)
for edge in connect:
  adj_mat[joints_to_consider_ids_remapped[edge[0]], joints_to_consider_ids_remapped[edge[1]]] = 1
  adj_mat[joints_to_consider_ids_remapped[edge[1]], joints_to_consider_ids_remapped[edge[0]]] = 1  # If the graph is undirected


batch_size = 64
lim_n_batches_percent = 0.25

# *_dim --> number of features --> spacial dimensionality
# n_*   --> number of frames   --> temporal dimensionality

latent_dim = 1 # features in the latent space
latent_n = 5 # number of frames in the latent space

# we are autoencoding, so input dims must be the same as the output ones
output_dim = input_dim

heads_concat = True

# (input_dim, output_dim, n_heads) for each GAT layer
# make sure that output_dim % n_heads == 0
GAT_config_enc = [
  (input_dim, 200, 4),
  (200, 400, 8),
  (400 , latent_dim, 1)
]

# in_channels and out_channels for each conv2D layer
channel_config_enc = [
  [input_n, 20],
  [20, 40],
  [40, latent_n]
]

# (input_dim, output_dim, n_heads) for each GAT layer
# make sure that output_dim % n_heads == 0
# GAT_config_dec = [
#   (input_dim, 128, 4),
#   (128, 256, 8),
#   (256, 128, 4),
#   (128, output_dim, 1)
# ]
GAT_config_dec = [
  (input_dim, 256, 4),
  (256, 512, 8),
  (512, 256, 4),
  (256, output_dim, 1)
]

# in_channels and out_channels for each conv2D layer
# channel_config_dec = [
#   [input_n, 192],
#   [192, 384],
#   [384, 192],
#   [192, 96],
#   [96, output_n]
# ]
channel_config_dec = [
  [input_n, 384],
  [384, 768],
  [768, 384],
  [384, output_n]
]

# Arguments to setup the optimizer
lr=1e-03 # learning rate

use_scheduler=False # use LR scheduler
scheduler_verbose=True

milestones=[5]   # the epochs after which the learning rate is adjusted by gamma
step_size = 10
gamma=0.5 #gamma correction to the learning rate, after reaching the milestone epochs

# weight_decay=1e-05 # weight decay (L2 penalty)
weight_decay=0 # weight decay (L2 penalty)
clip_grad=1.0 # select max norm to clip gradients

scheduler = None

n_epochs = 41
log_step = 999999999999999

save_and_plot = True # save the model and plot the loss. Change to True if you want to save the model and plot the loss

# joints at same loc
joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
joint_equal = np.array([13, 19, 22, 13, 27, 30])
index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))







# returns a dict containing all current parameters from training config
def get_config_dict():
  return {
  "input_n": input_n, 
  "output_n": output_n, 
  "input_dim": input_dim, 
  "skip_rate": skip_rate,
  "joints_to_consider_n": joints_to_consider_n,
  "batch_size": batch_size,
  "lim_n_batches_percent": lim_n_batches_percent,

  "latent_dim": latent_dim,
  "latent_n": latent_n,
  "output_dim": output_dim,
  "GAT_config_enc": GAT_config_enc,
  "channel_config_enc": channel_config_enc,
  "GAT_config_dec": GAT_config_dec,
  "channel_config_dec": channel_config_dec,
  "heads_concat": heads_concat,
  
  "lr": lr,
  "use_scheduler": use_scheduler,
  "milestones": milestones,
  "step_size": step_size,
  "gamma": gamma,
  "weight_decay": weight_decay,
  "clip_grad": clip_grad,
  
  "n_epochs": n_epochs,
}
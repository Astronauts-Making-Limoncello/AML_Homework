import numpy as np

# # Arguments to setup the datasets
datas = 'h36m' # dataset name
path = './data/h3.6m/h3.6m/dataset'
input_n=10 # number of frames to train on (default=10)
output_n=25 # number of frames to predict on
input_dim=3 # dimensions of the input coordinates(default=3)
skip_rate=1 # # skip rate of frames
joints_to_consider=22


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

# actions_to_consider_train = ["smoking"]
actions_to_consider_train = None

batch_size = 256
lim_n_batches_percent = 0.025

num_heads_encoder = 8 # encoder
num_heads_decoder = 1 # decoder
kernel_size = [3,3]
att_drop=0

# temporal
num_frames = 10
num_frames_out = 25

# spatial 
num_joints = 22
in_features = 3
hidden_features = 128
out_features = 3


config = [
  # [16, 16, 16], [16, 16, 16], [16, 16, 16], [16, 16, 16], 
  # [16, 16, 16], [16, 16, 16], [16, 16, 16], 
  [16,  out_features, 16]    
]

use_skip_connection_decoder = False
num_decoder_blocks = 3

# Arguments to setup the optimizer
lr=3e-3 # learning rate

use_scheduler=False 
milestones=[10]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.33 #gamma correction to the learning rate, after reaching the milestone epochs
step_size=10

weight_decay=1e-8 # weight decay (L2 penalty)

momentum=0.3
nesterov=True

clip_grad=None # select max norm to clip gradients

n_epochs = 6
log_step = 99999
log_epoch = 1 

dim_used = np.array(
  [
    6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92
  ]
)

save_and_plot = True # save the model and plot the loss. Change to True if you want to save the model and plot the loss

# joints at same loc
joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
joint_equal = np.array([13, 19, 22, 13, 27, 30])
index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))





def get_config_dict():
  return {
    "batch_size": batch_size,
    "lim_n_batches_percent": lim_n_batches_percent,
    "encoder_num_heads": num_heads_encoder,
    "decoder_num_heads": num_heads_decoder,
    "kernel_size": kernel_size,
    "att_drop": att_drop,
    "st_encoder_config": config,
    "use_skip_connection_decoder": use_skip_connection_decoder,
    "num_decoder_blocks": num_decoder_blocks,
    "num_frames": num_frames,
    "num_frames_out": num_frames_out,
    "in_features": in_features,
    "out_features": out_features,
    "hidden_features": hidden_features,
    "lr": lr,
    "use_scheduler": use_scheduler,
    "milestones": milestones,
    "gamma": gamma,
    "step_size": step_size,
    "weight_decay": weight_decay,
    "momentum": momentum,
    "nesterov": nesterov,
    "clip_grad": clip_grad,
    "n_epochs": n_epochs,
}
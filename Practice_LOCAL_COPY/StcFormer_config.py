import numpy as np

datas = 'h36m' # dataset name
path = './data/h3.6m/h3.6m/dataset'
input_n=10 # number of frames to train on (default=10)
output_n=25 # number of frames to predict on
input_dim=3 # dimensions of the input coordinates(default=3)
skip_rate=1 # # skip rate of frames
joints_to_consider=22

actions_to_consider_test='all' # actions to test on.
# actions_to_consider_train = ["smoking"]
actions_to_consider_train = None

batch_size_test = batch_size = 256
lim_n_batches_percent = 0.999

# temporal dimensionalities
num_frames = 10
num_frames_out = 25

# spatial dimensionalities
num_joints = 22
in_features = 3
hidden_features = 32
out_features = 3

use_skip_connection_encoder = False
num_heads_encoder = 1
num_encoder_blocks = 2

encoder_mask_s = None
encoder_mask_t = None

use_skip_connection_decoder = False
num_heads_decoder = 1
num_decoder_blocks = 2

lr=8e-3 
weight_decay=1e-5 
amsgrad=True
momentum=0.3
nesterov=True
clip_grad=None # select max norm to clip gradients

use_scheduler=False 
milestones=[40, 120, 240]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.5 #gamma correction to the learning rate, after reaching the milestone epochs
step_size=10
n_epochs = 51
log_step = 99999
log_epoch = 1 

save_and_plot = True # save the model and plot the loss. Change to True if you want to save the model and plot the loss

dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                  26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                  46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                  75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

# returns a dict containing all current parameters from training config
def get_config_dict():
  return {
    "batch_size": batch_size,
    "lim_n_batches_percent": lim_n_batches_percent,
    "encoder_num_heads": num_heads_encoder,
    "decoder_num_heads": num_heads_decoder,
    "use_skip_connection_decoder": use_skip_connection_decoder,
    "use_skip_connection_encoder": use_skip_connection_encoder,
    "num_encoder_blocks": num_encoder_blocks,
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
    "amsgrad": amsgrad,
    "momentum": momentum,
    "nesterov": nesterov,
    "clip_grad": clip_grad,
    "n_epochs": n_epochs,
 }
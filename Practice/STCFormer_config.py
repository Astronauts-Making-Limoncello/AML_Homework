import numpy as np

# # Arguments to setup the datasets
datas = 'h36m' # dataset name
path = './data/h3.6m/h3.6m/dataset'
input_n=10 # number of frames to train on (default=10)
n_special_tokens = 2 # start_of_sequence and end_of_sequence tokens
output_n=25 # number of frames to predict on
input_dim=3 # dimensions of the input coordinates(default=3)
skip_rate=1 # # skip rate of frames
joints_to_consider=22

actions_to_consider_test='all' # actions to test on.

# actions_to_consider_train = ["smoking"]
actions_to_consider_train = None

batch_size = 256
batch_size_val = batch_size
batch_size_test = batch_size
lim_n_batches_percent = 0.1

# temporal dimensionalities
num_frames = 10
num_frames_out = 25

# spatial dimensionalities
num_joints = 22
in_features = 3
in_features_encoder = in_features
in_features_decoder = in_features
hidden_features = 64
out_features = 3
out_features_encoder = out_features
out_features_decoder = out_features

autoregression = True

# start of sequence (SOS) special token 
use_start_of_sequence_token = True
# TODO add support for EOS special stoken
# end of sequence (EOS) special token
use_end_of_sequence_token   = False


use_start_of_sequence_token_in_encoder = False
use_end_of_sequence_token_in_encoder = False
use_special_tokens_encoder = (
  use_start_of_sequence_token_in_encoder or use_end_of_sequence_token_in_encoder
)

use_start_of_sequence_token_in_decoder = False
use_end_of_sequence_token_in_decoder = False
use_special_tokens_decoder = (
  use_start_of_sequence_token_in_decoder or use_end_of_sequence_token_in_decoder
)

# total number of special tokens used
# used to pad feature vectors to store vector representations of special tokens
# NOTE to guarantee all possible combinations working, gotta hardcode the length of the vector representing the 
# special tokens
# then each special token will have its vector, like (0, 1) or (1, 0), while normal tokens will have the all-zero vector
num_special_tokens_encoder = 2
num_special_tokens_decoder = 2

# start of sequence token represented by the vector (0, 1)
#   end of sequence token represented by the vector (1, 0)
#        all other tokens represented by the vector (0, 0)
# dataset has in_feature-dimensional vectors
# so, gotta att num_special_tokens (2) dimensions features 
# to in_features in order to accomodate the concatenation of these vectors
in_features_encoder += num_special_tokens_encoder if (use_special_tokens_encoder) else 0
in_features_decoder += num_special_tokens_decoder if (use_special_tokens_decoder) else 0

out_features_encoder += num_special_tokens_encoder if (use_special_tokens_encoder) else 0
out_features_decoder += num_special_tokens_decoder if (use_special_tokens_decoder) else 0

# offset to apply to in_feature to index each special token
# special tokens are identified by num_special_tokens{encoder,decoder} one-hot encoding vectors
# so we keep track of what index in the one-hot encoding vector is actually indexing what token
# NOTE read comments in num_special_tokens_{encoder,decoder} to have a better understanding
#                 start of sequence is in position in_features + 0
#                   end of sequence is in position in_features + 1
#   other future special token will be in position in_features + 2
# another future special token will be in position in_features + 3
# etc.
start_of_sequence_token_offset = 1
end_of_sequence_token_offset   = 0
# other_token_offset = 2
# another_token_offset = 3
# etc.

use_skip_connection_encoder = True
num_heads_encoder = 1
num_encoder_blocks = 1
dropout_encoder = 0.0

encoder_mask_s = None
encoder_mask_t = None

use_skip_connection_decoder = False
skip_connection_weight_decoder = 1
num_heads_decoder = 1
num_decoder_blocks = 1
dropout_decoder = 0.0

lr=1e-3
weight_decay=1e-5 
amsgrad=True
momentum=0.3
nesterov=True
clip_grad=None # select max norm to clip gradients

use_scheduler=False 
milestones=[40, 120, 240]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.5 #gamma correction to the learning rate, after reaching the milestone epochs
step_size=10

n_epochs = 4
log_step = 99999
log_epoch = 1 

model_path= './checkpoints/' # path to the model checkpoint file
model_name = datas+'_3d_'+str(output_n)+'frames_ckpt' #the model name to save/load

epoch_offset = 0
resume_from_ckpt = None

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
  "autoregression": autoregression,
  "use_start_of_sequence_token": use_start_of_sequence_token,
  "use_end_of_sequence_token": use_end_of_sequence_token, 
  "use_start_of_sequence_token_in_encoder": use_start_of_sequence_token_in_encoder,
  "use_end_of_sequence_token_in_encoder": use_end_of_sequence_token_in_encoder,
  "use_special_tokens_encoder": use_special_tokens_encoder, 
  "use_start_of_sequence_token_in_decoder": use_start_of_sequence_token_in_decoder, 
  "use_end_of_sequence_token_in_decoder": use_end_of_sequence_token_in_decoder, 
  "use_special_tokens_decoder": use_special_tokens_decoder,  
  "num_special_tokens_encoder": num_special_tokens_encoder, 
  "num_special_tokens_decoder": num_special_tokens_decoder, 
  "start_of_sequence_token_offset": start_of_sequence_token_offset, 
  "end_of_sequence_token_offset": end_of_sequence_token_offset,   
  "encoder_num_heads": num_heads_encoder,
  "decoder_num_heads": num_heads_decoder,
  "use_skip_connection_decoder": use_skip_connection_decoder,
  "skip_connection_weight_decoder": skip_connection_weight_decoder,
  "use_skip_connection_encoder": use_skip_connection_encoder,
  "num_encoder_blocks": num_encoder_blocks,
  "num_decoder_blocks": num_decoder_blocks,
  "num_frames": num_frames,
  "num_frames_out": num_frames_out,
  "in_features_encoder": in_features_encoder,
  "in_features_decoder": in_features_decoder,
  "out_features_encoder": out_features_encoder,
  "out_features_decoder": out_features_decoder,
  "dropout_encoder": dropout_encoder,
  "dropout_decoder": dropout_decoder,
  "hidden_features": hidden_features,
  "checkpoint_path": resume_from_ckpt,
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
from utils import h36motion3d as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.autograd
import torch
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
from utils.loss_funcs import *
from utils.data_utils import define_actions
from utils.h36_3d_viz import visualize
import time

import torch.nn.functional as F

import datetime
import os
import wandb

from rich import print

### --- DEVICE --- ###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

### --- DEVICE --- ###

############################################################

### --- DATA --- ###

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

actions_to_consider_train = ["smoking"]
# actions_to_consider_train = None

if actions_to_consider_train is not None:
  print(f"[b][#FF0000]WARNING: training only on these actions --> {actions_to_consider_train}")

# Load Data
dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=0, actions=actions_to_consider_train)
vald_dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=1, actions=actions_to_consider_train)

batch_size_test = batch_size = 256
lim_n_batches_percent = 0.999

data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)#
vald_loader = DataLoader(vald_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

### --- DATA --- ###

############################################################

### --- MODEL --- ###

from SpatioTemporalEncoder import SpatioTemporalEncoder
from SpatioTemporalDecoder import SpatioTemporalDecoder
from SpatioTemporalTransformer import SpatioTemporalTransformer
from utils.masking import causal_mask

# temporal dimensionalities
num_frames = 10
num_frames_out = 25

# spatial dimensionalities
num_joints = 22
in_features = 3
in_features_encoder = in_features
in_features_decoder = in_features
hidden_features = 32
out_features = 3
out_features_encoder = out_features
out_features_decoder = out_features

autoregression = True

if autoregression:

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
  
  use_start_of_sequence_token_in_decoder = True
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

use_skip_connection_encoder = False
num_heads_encoder = 1
num_encoder_blocks = 2

st_encoder = SpatioTemporalEncoder(
    in_features_encoder, hidden_features, out_features_encoder, num_joints,
    num_frames, num_frames, # it's encoder, so num_frames_out == num_frames
    num_heads_encoder, use_skip_connection_encoder,
    num_encoder_blocks
)

encoder_mask_s = None
encoder_mask_t = None

use_skip_connection_decoder = False
num_heads_decoder = 1
num_decoder_blocks = 2

st_decoder = SpatioTemporalDecoder(
    decoder_input_in_features=in_features_decoder, 
    encoder_output_in_features=out_features_encoder, 
    hidden_features=hidden_features, out_features=out_features_decoder, 
    num_joints=num_joints,
    num_frames=num_frames, num_frames_out=num_frames_out,
    num_heads=num_heads_decoder, use_skip_connection=use_skip_connection_decoder,
    num_decoder_blocks=num_decoder_blocks
)

# alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
# seq_len is num_frames_out in decoder, so gotta use num_frames_out
decoder_mask_s = causal_mask((1, 1, 1, num_joints, num_joints)).to(device)

# since we are in the decoder, mask must have same shape of attn_t
# REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
# so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
decoder_mask_t = causal_mask((1, 1, 1, num_frames_out, num_frames)).to(device)

model = st_transformer = SpatioTemporalTransformer(
    st_encoder=st_encoder, st_decoder=st_decoder,
    num_frames=num_frames, num_joints=num_joints, 
    in_features_encoder=in_features_encoder, in_features_decoder=in_features_decoder
).to(device)

print(f"Number of trainable parameters: [b][#6495ED]{sum(p.numel() for p in model.parameters() if p.requires_grad)}")

### --- MODEL --- ###

############################################################

### --- OPTIMIZER --- ###

lr=8e-3 
weight_decay=1e-5 
amsgrad=True
momentum=0.3
nesterov=True
clip_grad=None # select max norm to clip gradients


optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay, amsgrad=amsgrad)
# optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

### --- OPTIMIZER --- ###

############################################################

### --- LR SCHEDULER --- ###
use_scheduler=False 
milestones=[40, 120, 240]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.5 #gamma correction to the learning rate, after reaching the milestone epochs
step_size=10

if use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma, verbose=True)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)

### --- LR SCHEDULER --- ###

############################################################

### --- TRAINING CONFIG --- ###

n_epochs = 3
log_step = 99999
log_epoch = 1 

train_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(f"Train ID: [bold][#FF1493]{train_id}")
model_path= './checkpoints/' # path to the model checkpoint file
model_name = datas+'_3d_'+str(output_n)+'frames_ckpt' #the model name to save/load
ckpt_dir = f"{model_path}{train_id}"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None

epoch_offset = 0
resume_from_ckpt = None
# resume_from_ckpt = f"{model_path}/2023_11_06_08_30_03/h36m_3d_25frames_ckpt_best_val_loss.pt"
if resume_from_ckpt is not None:
  print(f"[b][#008000]RESUMING FROM CHECKPOINT {resume_from_ckpt}...")
  
  ckpt = torch.load(resume_from_ckpt)

  model.load_state_dict(ckpt["model_state_dict"])
  print(f"[b][#008000]MODEL STATE DICT LOADED")
  optimizer.load_state_dict(ckpt["optimizer_state_dict"])
  print(f"[b][#008000]OPTIMIZER STATE DICT LOADED")

  epoch_offset = ckpt["epoch"] - 1

train_config = {
  "batch_size": batch_size,
  "lim_n_batches_percent": lim_n_batches_percent,
  "autoregression": autoregression,
  "encoder_num_heads": num_heads_encoder,
  "decoder_num_heads": num_heads_decoder,
  "use_skip_connection_decoder": use_skip_connection_decoder,
  "use_skip_connection_encoder": use_skip_connection_encoder,
  "num_encoder_blocks": num_encoder_blocks,
  "num_decoder_blocks": num_decoder_blocks,
  "decoder_mask_s": decoder_mask_s,
  "decoder_mask_t": decoder_mask_t,
  "num_frames": num_frames,
  "num_frames_out": num_frames_out,
  "in_features_encoder": in_features_encoder,
  "in_features_decoder": in_features_decoder,
  "out_features_encoder": out_features_encoder,
  "out_features_decoder": out_features_decoder,
  "hidden_features": hidden_features,
  "checkpoint_path": resume_from_ckpt,
  "checkpoint_epoch": ckpt["epoch"] if resume_from_ckpt is not None else -69,
  "model": model,
  "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
  "lr": lr,
  "use_scheduler": use_scheduler,
  "milestones": milestones,
  "gamma": gamma,
  "step_size": step_size,
  "weight_decay": weight_decay,
  "amsgrad": amsgrad,
  "momentum": momentum,
  "nesterov": nesterov,
  "optimizer": optimizer,
  "scheduler": scheduler if use_scheduler else None,
  "clip_grad": clip_grad,
  "n_epochs": n_epochs,
  "train_id": train_id
}

### --- TRAINING CONFIG --- ###

############################################################

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

progress_bar = Progress(
  TextColumn("[progress.description]{task.description}"),
  TextColumn("[progress.percentage]{task.percentage:>3.2f}%"),
  BarColumn(),
  MofNCompleteColumn(),
  TextColumn("•"),
  TimeElapsedColumn(),
  TextColumn("•"),
  TimeRemainingColumn(),
  TextColumn("[#00008B]{task.speed} it/s"),
  SpinnerColumn(),
)

progress_bar.start()

def train(data_loader,vald_loader, path_to_save_model=None):

  n_train_batches = int(len(data_loader) * lim_n_batches_percent) + 1 
  n_val_batches = int(len(vald_loader) * lim_n_batches_percent) + 1 

  epoch_task = progress_bar.add_task("[bold][#B22222]Epoch progress...", total=n_epochs-1)  
  train_task = progress_bar.add_task("[bold][#6495ED]Train batches progress...", total=n_train_batches)  
  val_task = progress_bar.add_task("[bold][#008080]Val batches progress...", total=n_val_batches)  

  # wandb.init(
  #   project="Custom-model-SpatioTemporalTransformer",
  #   config=train_config
  # )

  train_loss = []
  val_loss = []
  train_loss_best = 100000000
  val_loss_best   = 100000000

  dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

  for epoch in range(n_epochs-1):
      
      progress_bar.reset(task_id=train_task)
      progress_bar.reset(task_id=val_task)

      running_loss=0
      n=0
      model.train()
      for cnt,batch in list(enumerate(data_loader))[:n_train_batches]:
          batch_size_current_batch = batch.shape[0]
          batch=batch.float().to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim

          # NOTE: encoders based on their code require the permute
          # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          # NOTE: encoders based on our code do NOT require the permute
          sequences_train = batch[:, 0:input_n, dim_used].view(-1, input_n, num_joints, 3)
          # print(f"\[main.train] sequences_train.shape: {sequences_train.shape}")
          # sequences_train.shape: batch_size, input_n, num_joints, in_features

          sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)
          # print(f"\[main.train] sequences_gt.shape: {sequences_gt.shape}")

          if autoregression:

            # working on a copy of sequences_gt so as we keep the original untoched for the loss computation
            # and to keep compatibility for when autoregression=False
            sequences_gt_autoregression = batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

            # adding num_special_tokens extra features
            # to accomodate encoding of special tokens (start of sequence, end of sequence)
            in_features_pad_tgt = torch.zeros(
              batch_size_current_batch, output_n, num_joints, num_special_tokens_decoder
            ).to(device)
            sequences_gt_autoregression = torch.cat((sequences_gt_autoregression, in_features_pad_tgt), dim=-1)
            # print(f"sequences_gt_autoregression.shape: {sequences_gt_autoregression.shape}")

            if use_start_of_sequence_token_in_decoder:
              
              # removing last element of gt sequence, in order to shift gt sequence to right by one position
              # by means of adding the start of sequence token
              sequences_gt_autoregression = sequences_gt_autoregression[:, :-1, :, :]
              # print(f"sequences_gt_autoregression.shape: {sequences_gt_autoregression.shape}")

              # creating the start of sequence token
              # see autoregression section comments of report for full detail on shapes, etc.
              start_of_sequence_token_decoder = torch.zeros(
                (batch_size_current_batch, 1, num_joints, in_features_decoder)
              ).to(device)
              # print(f"start_of_sequence_token_decoder.shape: {start_of_sequence_token_decoder.shape}")
              
              # concatenating sequences_gt_autoregression to start_of_sequence_token along the temporal dimension
              # effectively shifts the output by 1 position, by means of adding the
              # start of sequence token
              sequences_gt_autoregression = torch.cat(
                tensors=(start_of_sequence_token_decoder, sequences_gt_autoregression),
                dim=1 
              )
              # print(f"sequences_gt_autoregression.shape: {sequences_gt_autoregression.shape}")

              # feature vector of special token (which is concatenated to feature vector of data!)
              # is a one-hot encoding vector, where each special token type has its own dimension set to 1
              # so, setting to 1 the dimension of the start_of_sequence special token to 1
              sequences_gt_autoregression[:, 0, :, in_features+start_of_sequence_token_offset] = 1.

          optimizer.zero_grad()
          sequences_predict=model(
            src=sequences_train, 
            tgt=sequences_gt_autoregression if autoregression else sequences_gt, 
            src_mask_s=encoder_mask_s, src_mask_t=encoder_mask_t,
            tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
          )
          # print(f"\[main.train] sequences_predict.shape: {sequences_predict.shape}")

          if autoregression:
            # special token num_special_token_{encoder,decoder}-long vectors 
            # get appended to the end of the in_feature-long feature vector
            # so, using slicing, we can separate feature vector from special token encoding vectors
             
            sequences_predict_special_tokens =           sequences_predict[:, :, :, in_features:]
            sequences_gt_special_tokens      = sequences_gt_autoregression[:, :, :, in_features:]
            # print(f"\[main.train] sequences_predict_special_tokens.shape: {sequences_predict_special_tokens.shape}")
            # print(f"\[main.train] sequences_gt_special_tokens.shape     : {sequences_gt_special_tokens.shape}")
            
            sequences_predict = sequences_predict[:, :, :, :in_features]
            # print(f"\[main.train] sequences_predict.shape: {sequences_predict.shape}")


          loss = mpjpe_error(sequences_predict,sequences_gt)
          
          if autoregression:
            loss_2 = torch.nn.functional.cross_entropy(
              input=sequences_predict_special_tokens.reshape(
                batch_size_current_batch, num_frames_out * num_joints, num_special_tokens_decoder
              ), 
              target=sequences_gt_special_tokens.reshape(
                batch_size_current_batch, num_frames_out * num_joints, num_special_tokens_decoder
              ),

            )

          if autoregression:
            loss += loss_2


        #   if cnt % log_step == 0:
        #     print('[Epoch: %d, Iteration: %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

          loss.backward()
          if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

          optimizer.step()
          running_loss += loss*batch_dim

          if running_loss/n < train_loss_best:
            train_loss_best = running_loss/n

            torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': train_loss,
              'val_loss': val_loss
              }, f"{ckpt_dir}/{model_name}_best_train_loss.pt")

          progress_bar.update(task_id=train_task, advance=1)
          progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))


      train_loss.append(running_loss.detach().cpu()/n)

      continue

      model.eval()
      with torch.no_grad():
          running_loss=0
          n=0
          for cnt,batch in list(enumerate(vald_loader))[:n_val_batches]:
              batch=batch.float().to(device)
              batch_dim=batch.shape[0]
              n+=batch_dim

              # TODO encoders based on their code require the permute
              # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
              # TODO encoders based on our code do NOT require the permute
              sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3)
              sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

              # sequences_predict=model(sequences_train).view(-1, output_n, joints_to_consider, 3)
              sequences_predict=model(
                src=sequences_train, tgt=sequences_gt,
                tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
              )
              # print(f"\[main.train] sequences_predict.shape: {sequences_predict.shape}")
              sequences_predict=sequences_predict.view(-1, output_n, joints_to_consider, 3)
              # print(f"\[main.train] sequences_predict.shape: {sequences_predict.shape}")
              
              
              
              
              loss=mpjpe_error(sequences_predict,sequences_gt)

            #   if cnt % log_step == 0:
            #             print('[Epoch: %d, Iteration: %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))
              running_loss+=loss*batch_dim

              progress_bar.update(task_id=val_task, advance=1)
              progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))


          if running_loss/n < val_loss_best:
            val_loss_best = running_loss/n

            torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': train_loss,
              'val_loss': val_loss
              }, f"{ckpt_dir}/{model_name}_best_val_loss.pt")


          val_loss.append(running_loss.detach().cpu()/n)



      if use_scheduler:
        scheduler.step()

      # save and plot model every 5 epochs
      '''
      Insert your code below. Use the argument path_to_save_model to save the model to the path specified.
      '''
      if save_and_plot and epoch%log_epoch==0:
          torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }, f"{ckpt_dir}/{model_name}_epoch_{str(epoch + 1)}.pt")

      # wandb.log({
      #     "epoch": epoch + epoch_offset,
      #     "loss/train": train_loss[-1],
      #     "loss/val": val_loss[-1]
      # })

      if epoch == 0:
        print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b] | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b]")
      else:
        # print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:.3f}[/#6495ED][/b] ([#6495ED][b]{(train_loss[-1] - train_loss[-2]):.3f}[/#6495ED][/b]), best: {train_loss_best:.3f} | val loss: [b][#008080]{val_loss[-1]:.3f}[/#008080][/b] ([b][#008080]{(val_loss[-1] - val_loss[-2]):.3f}[/#008080][/b]), best: {val_loss_best:.3f}")
        print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b], best (step): {train_loss_best:07.3f} | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b], best: {val_loss_best:07.3f}")


  # wandb.log({
  #   "best_loss/train": np.array(train_loss).min(),
  #   "best_loss_epoch/train": np.array(train_loss).argmin() + epoch_offset,
  #   "best_loss/val": np.array(val_loss).min(),
  #   "best_loss_in_step/train": train_loss_best,
  #   "best_loss_in_step/val": val_loss_best
  # })


      

save_and_plot = True # save the model and plot the loss. Change to True if you want to save the model and plot the loss

train(data_loader,vald_loader, path_to_save_model=model_path)

print("\n\n\n\n\n TODO ADD AUTOREGRESSION SUPPORT TO TEST TOO!!! \n\n\n")
exit()

def test(ckpt_path=None):

    action_loss_dict = {}

    # model.load_state_dict(torch.load(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    # print('model loaded')
    model.eval()
    accum_loss=0
    n_batches=0 # number of batches for all the sequences

    actions=define_actions(actions_to_consider_test)
    actions_task = progress_bar.add_task("[bold][#9400D3]Action test batches progress...", total=len(actions))


    dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                      26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                      46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                      75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
    # joints at same loc
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31])
    index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    joint_equal = np.array([13, 19, 22, 13, 27, 30])
    index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))
    totalll=0
    counter=0
    for action in actions:
      running_loss=0
      
      n=0
      dataset_test = datasets.Datasets(path,input_n,output_n,skip_rate, split=2,actions=[action], use_progress_bar=False)
      #print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))
      test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
      n_test_batches = int(len(test_loader) * lim_n_batches_percent) + 1   
      
      for cnt,batch in list(enumerate(test_loader))[:n_test_batches]:
        with torch.no_grad():

          batch=batch.to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim


          all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]

          # TODO encoders based on their code require the permute
          # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          # TODO encoders based on our code do NOT require the permute
          sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3)
          # sequences_gt=batch[:, input_n:input_n+output_n, :]
          sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

          # print(f"sequences_train.shape  : {sequences_train.shape}")
          # print(f"sequences_gt.shape     : {sequences_gt.shape}")
          running_time = time.time()

          sequences_predict=model(
            src=sequences_train, tgt=sequences_gt,
            tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
          ).view(-1, output_n, joints_to_consider, 3)
          # print(f"sequences_predict.shape: {sequences_predict.shape}")
          #sequences_predict = model(sequences_train)
          totalll += time.time()-running_time
          counter += 1
          sequences_predict=sequences_predict.contiguous().view(-1,output_n,len(dim_used))
          # print(f"sequences_predict.shape: {sequences_predict.shape}")

          all_joints_seq[:,:,dim_used] = sequences_predict
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")

          all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")

          all_joints_seq = all_joints_seq.view(-1,output_n,32,3)
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")
          # sequences_gt = sequences_gt.view(-1,output_n,32,3)
          # gotta retake it from the batch, in order to get the original number of joints (32)
          sequences_gt=batch[:, input_n:input_n+output_n, :]
          # print(f"sequences_gt.shape     : {sequences_gt.shape}")


          loss=mpjpe_error(all_joints_seq,sequences_gt)

          running_loss+=loss*batch_dim
          accum_loss+=loss*batch_dim

      #print('loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/n))
    #   print(str(action),': ', str(np.round((running_loss/n).item(),1)))
      action_loss_dict[f"loss/test/{action}"] = np.round((running_loss/n).item(),1)

      n_batches+=n

      progress_bar.advance(task_id=actions_task, advance=1)
    
    
    # print('Average: '+str(np.round((accum_loss/n_batches).item(),1)))
    # print('Prediction time: ', totalll/counter)

    action_loss_dict["loss/test"] = np.round((accum_loss/n_batches).item(),1)
    # wandb.log(action_loss_dict)

# ckpt_path = './checkpoints/h36m_3d_25frames_ckpt_epoch_0.pt' # Change the epoch according to the validation curve
ckpt_path = f"{ckpt_dir}/{model_name}_best_val_loss.pt"
test(ckpt_path)
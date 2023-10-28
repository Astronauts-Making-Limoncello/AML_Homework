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

import wandb

import os

from GATEncoder import GATEncoder
from GATDecoder import GATDecoder
from GATAutoEncoder import GATAutoEncoder

# Use GPU if available, otherwise stick with cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

# *_dim --> number of features --> spacial dimensionality
# n_*   --> number of frames   --> temporal dimensionality

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

actions_to_consider_train = ["walking"]
# actions_to_consider_train = None

if actions_to_consider_train is not None:
  print(f"ACTIONS_TO_CONSIDER_TRAIN: {actions_to_consider_train}")

# Load Data
print('Loading Train Dataset...')
dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=0, actions=actions_to_consider_train)
print('Loading Validation Dataset...')
vald_dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=1, actions=actions_to_consider_train)

batch_size=256
lim_n_batches_percent = 0.1

print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)#

print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
vald_loader = DataLoader(vald_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

from models.sttr.sttformer import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device: %s'%device)

n_heads = 2

# model = Model(num_joints=joints_to_consider_n,
#                  num_frames=input_n, num_frames_out=output_n, num_heads=n_heads,
#                  num_channels=3, kernel_size=[3,3], use_pes=True).to(device)

# *_dim --> number of features --> spacial dimensionality
# n_*   --> number of frames   --> temporal dimensionality

latent_dim = 128 # features in the latent space
latent_n = 15 # number of frames in the latent space

# we are autoencoding, so input dims must be the same as the output ones
output_dim = input_dim

# (input_dim, output_dim, n_heads) for each GAT layer
# make sure that output_dim % n_heads == 0
GAT_config_enc = [
  (input_dim, 20, 4),
  (20, 40, 8),
  (40 , latent_dim, 2)
]

# in_channels and out_channels for each conv2D layer
channel_config_enc = [
  [input_n, 2],
  [2, 4],
  [4, latent_n]
]

enc = GATEncoder(GAT_config_enc, channel_config_enc)

# (input_dim, output_dim, n_heads) for each GAT layer
# make sure that output_dim % n_heads == 0
GAT_config_dec = [
  (latent_dim, 20, 4),
  (20, 40, 8),
  (40 , output_dim, 1)
]

# in_channels and out_channels for each conv2D layer
channel_config_dec = [
  [latent_n, 2],
  [2, 4],
  [4, output_n]
]

dec = GATDecoder(GAT_config_dec, channel_config_dec)  

model = GATAutoEncoder(enc, dec).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# Arguments to setup the optimizer
lr=1e-01 # learning rate
use_scheduler=True # use MultiStepLR scheduler
milestones=[10,30]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.1 #gamma correction to the learning rate, after reaching the milestone epochs
weight_decay=1e-05 # weight decay (L2 penalty)
optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay)

if use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

clip_grad=None # select max norm to clip gradients
# Argument for training
# n_epochs=41
# log_step = 200
n_epochs=3
log_step = 1

train_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
ckpt_dir = f"{model_path}{train_id}"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None

train_config = {
  "input_n": input_n, 
  "output_n": output_n, 
  "input_dim": input_dim, 
  "skip_rate": skip_rate,
  "joints_to_consider_n": joints_to_consider_n,
  "batch_size": batch_size,
  "lim_n_batches_percent": lim_n_batches_percent,
  # "adj_mat": adj_mat,
  
  "n_heads": n_heads,
  "latent_dim": latent_dim,
  "latent_n": latent_n,
  "output_dim": output_dim,
  "GAT_config_enc": GAT_config_enc,
  "channel_config_enc": channel_config_enc,
  "GAT_config_dec": GAT_config_dec,
  "channel_config_dec": channel_config_dec,
  "model": str(model),
  "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
  
  "lr": lr,
  "use_scheduler": use_scheduler,
  "milestones": milestones,
  "gamma": gamma,
  "weight_decay": weight_decay,
  "optimizer": str(optimizer),
  "scheduler": str(scheduler),
  "clip_grad": clip_grad,
  
  "n_epochs": n_epochs,
  "train_id": train_id
}

def train(data_loader,vald_loader, path_to_save_model=None):

  # wandb.init(
  #   project="Custom model",
  #   config=train_config
  # )

  n_train_batches = int(len(data_loader) * lim_n_batches_percent) + 1 
  n_val_batches = int(len(vald_loader) * lim_n_batches_percent) + 1 

  train_loss = []
  val_loss = []
  val_loss_best = 1000

  dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])
  
  n_joints = len(dim_used)//3
  joint_to_consider_ids = set(dim_used//3)
  joints_to_consider_ids_remapped = {
      k: v for k, v in zip(joint_to_consider_ids, range(len(joint_to_consider_ids))) 
  }
  print(f"joint_to_consider_ids: {joint_to_consider_ids}")
  print(f"joints_to_consider_ids_remapped: {joints_to_consider_ids_remapped}")
  print(f"len(joint_to_consider_ids): {len(joint_to_consider_ids)}")

  connect = [
    (1, 2), (2, 3), (3, 4), (4, 5), (6, 7), (7, 8), (8, 9), (9, 10), (0, 1), 
    (0, 6), (6, 17), (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (1, 25), 
    (25, 26), (26, 27), (27, 28), (28, 29), (29, 30), (24, 25), (24, 17), 
    (24, 14), (14, 15)
  ]
  connect = [c for c in connect if c[0] in joint_to_consider_ids and c[1] in joint_to_consider_ids]
  adj_mat = torch.zeros(n_joints, n_joints, dtype=torch.int).to(device)
  for edge in connect:
    adj_mat[joints_to_consider_ids_remapped[edge[0]], joints_to_consider_ids_remapped[edge[1]]] = 1
    adj_mat[joints_to_consider_ids_remapped[edge[1]], joints_to_consider_ids_remapped[edge[0]]] = 1  # If the graph is undirected
  
  print(f"len(dim_used): {len(dim_used)}")
  print(f"len(dim_used)//3: {len(dim_used)//3}")
  print(f"n_joints: {n_joints}")

  for epoch in range(n_epochs-1):
      running_loss=0
      n=0
      model.train()
      for cnt,batch in list(enumerate(data_loader))[:n_train_batches]:
          batch=batch.float().to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim

          sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3)
          # print(f"[train] sequences_train.shape: {sequences_train.shape}") # batch_size, n_input (temporal dim), n_nodes (spacial dim/skeleton joints), in_features
          sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

          optimizer.zero_grad()
          
          sequences_predict=model.forward(sequences_train, adj_mat)
          # print(f"[train] sequences_predict.shape: {sequences_predict.shape}") # batch_size, n_output (temporal dim), n_nodes (spacial dim/skeleton joints), out_features

          loss=mpjpe_error(sequences_predict,sequences_gt)


          if cnt % log_step == 0:
            print('[Epoch: %d, Iteration: %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

          loss.backward()
          if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

          optimizer.step()
          running_loss += loss*batch_dim

      train_loss.append(running_loss.detach().cpu()/n)
      model.eval()
      with torch.no_grad():
          running_loss=0
          n=0
          for cnt,batch in list(enumerate(vald_loader))[:n_val_batches]:
              batch=batch.float().to(device)
              batch_dim=batch.shape[0]
              n+=batch_dim

              sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3)
              sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

              sequences_predict=model(sequences_train, adj_mat)
              
              loss=mpjpe_error(sequences_predict,sequences_gt)

              if cnt % log_step == 0:
                print('[Epoch: %d, Iteration: %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))
              running_loss+=loss*batch_dim
          val_loss.append(running_loss.detach().cpu()/n)
          if running_loss/n < val_loss_best:
            val_loss_best = running_loss/n

            torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': train_loss,
              'val_loss': val_loss
            }, f"{ckpt_dir}/{model_name}_best_val_loss.pt")



      if use_scheduler:
        scheduler.step()

      # save and plot model every 5 epochs
      '''
      Insert your code below. Use the argument path_to_save_model to save the model to the path specified.
      '''
      if save_and_plot and epoch%5==0:
          torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }, f"{ckpt_dir}/{model_name}_epoch_{str(epoch + 1)}.pt")

      # wandb.log({
      #     "epoch": epoch,
      #     "loss/train": train_loss[-1],
      #     "loss/val": val_loss[-1]
      # })

save_and_plot = True # save the model and plot the loss. Change to True if you want to save the model and plot the loss

# launch training
train(data_loader,vald_loader, path_to_save_model=model_path)

exit()
# TODO implement changes before using it!
def test(ckpt_path=None):
    # model.load_state_dict(torch.load(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    print('model loaded')
    model.eval()
    accum_loss=0
    n_batches=0 # number of batches for all the sequences
    actions=define_actions(actions_to_consider_test)
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
      dataset_test = datasets.Datasets(path,input_n,output_n,skip_rate, split=2,actions=[action])
      #print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))

      test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
      for cnt,batch in enumerate(test_loader):
        with torch.no_grad():

          batch=batch.to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim


          all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]

          sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          sequences_gt=batch[:, input_n:input_n+output_n, :]


          running_time = time.time()
          sequences_predict=model(sequences_train).view(-1, output_n, joints_to_consider_n, 3)
          #sequences_predict = model(sequences_train)
          totalll += time.time()-running_time
          counter += 1
          sequences_predict=sequences_predict.contiguous().view(-1,output_n,len(dim_used))

          all_joints_seq[:,:,dim_used] = sequences_predict


          all_joints_seq[:,:,index_to_ignore] = all_joints_seq[:,:,index_to_equal]

          loss=mpjpe_error(all_joints_seq.view(-1,output_n,32,3),sequences_gt.view(-1,output_n,32,3))
          running_loss+=loss*batch_dim
          accum_loss+=loss*batch_dim

      #print('loss at test subject for action : '+str(action)+ ' is: '+ str(running_loss/n))
      print(str(action),': ', str(np.round((running_loss/n).item(),1)))
      n_batches+=n
    print('Average: '+str(np.round((accum_loss/n_batches).item(),1)))
    print('Prediction time: ', totalll/counter)

ckpt_path = f"{ckpt_dir}/{model_name}_best_val_loss.pt"
test(ckpt_path)
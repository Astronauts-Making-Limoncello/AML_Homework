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

import rich
from rich import print
from utils.progress import get_progress_bar

from models.ConvGAT.GATEncoder import GATEncoder
from models.ConvGAT.GATDecoder import GATDecoder
from models.ConvGAT.GATAutoEncoder import GATAutoEncoder

import ConvSTGAT_config as conf

progress_bar = get_progress_bar()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {torch.cuda.get_device_name(device)}")

# *_dim --> number of features --> spacial dimensionality
# n_*   --> number of frames   --> temporal dimensionality

if conf.actions_to_consider_train is not None:
  print(f"WARNING: NOT using all actions, training only on {conf.actions_to_consider_train}")

# Load Data
print('Loading Train Dataset...')
dataset = datasets.Datasets(
  conf.path,conf.input_n,conf.output_n,conf.skip_rate, split=0, 
  actions=conf.actions_to_consider_train
)
print('Loading Validation Dataset...')
vald_dataset = datasets.Datasets(
  conf.path,conf.input_n,conf.output_n,conf.skip_rate, split=1, 
  actions=conf.actions_to_consider_train
)

data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0, pin_memory=True)#

vald_loader = DataLoader(vald_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0, pin_memory=True)

enc = GATEncoder(conf.GAT_config_enc, conf.channel_config_enc, conf.heads_concat)

dec = GATDecoder(conf.GAT_config_dec, conf.channel_config_dec, conf.heads_concat)  

# model = GATAutoEncoder(enc, dec).to(device)
model = dec.to(device)

print('Num trainable params: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer=optim.Adam(model.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

if conf.use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=conf.gamma, verbose=conf.scheduler_verbose)  
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=scheduler_verbose)

train_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
ckpt_dir = f"{conf.model_path}{train_id}"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None

train_config = {
  
  "model": str(model),
  "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
  "optimizer": str(optimizer),
  "scheduler": str(scheduler),
  "train_id": train_id
}.update(conf.get_config_dict())

def train(data_loader,vald_loader, path_to_save_model=None):

  wandb.init(
    project="Custom-model-ConvSTGAT",
    config=train_config
  )
  wandb.watch(model)

  n_train_batches = int(len(data_loader) * conf.lim_n_batches_percent) + 1 
  n_val_batches = int(len(vald_loader) * conf.lim_n_batches_percent) + 1 

  epoch_task = progress_bar.add_task("[bold][#B22222]Epoch progress...", total=conf.n_epochs-1)  
  train_task = progress_bar.add_task("[bold][#6495ED]Train batches progress...", total=n_train_batches)  
  val_task = progress_bar.add_task("[bold][#008080]Val batches progress...", total=n_val_batches)  

  progress_bar.start()

  train_loss = []
  val_loss = []
  train_loss_best = 10000000
  val_loss_best   = 10000000

  adj_mat = conf.adj_mat.to(device)

  for epoch in range(conf.n_epochs-1):
      running_loss=0
      n=0

      progress_bar.reset(task_id=train_task)
      progress_bar.reset(task_id=val_task)

      model.train()

      for cnt,batch in list(enumerate(data_loader))[:n_train_batches]:
        batch=batch.float().to(device)
        
        batch_dim=batch.shape[0]
        
        n+=batch_dim

        sequences_train=batch[:, 0:conf.input_n, conf.dim_used].view(
          -1, conf.input_n, len(conf.dim_used)//3, 3
        )
        sequences_gt=batch[
          :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
        ].view(-1, conf.output_n, len(conf.dim_used)//3, 3)

        optimizer.zero_grad()
        
        sequences_predict=model.forward(sequences_train, adj_mat)

        loss=mpjpe_error(sequences_predict,sequences_gt)

        if (cnt + 1)% conf.log_step == 0:
          print('[Epoch: %d, Iteration: %5d]  training loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))

        loss.backward()

        if conf.clip_grad is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), conf.clip_grad)

        optimizer.step()
        
        running_loss += loss*batch_dim

        if running_loss/n < train_loss_best:
          train_loss_best = running_loss/n

        progress_bar.update(task_id=train_task, advance=1)
        progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))

      train_loss.append(running_loss.detach().cpu()/n)
      
      model.eval()

      with torch.no_grad():
          
          running_loss=0
          
          n=0
          
          for cnt,batch in list(enumerate(vald_loader))[:n_val_batches]:
            batch=batch.float().to(device)
            
            batch_dim=batch.shape[0]
            
            n+=batch_dim

            sequences_train=batch[:, 0:conf.input_n, conf.dim_used].view(
              -1, conf.input_n, len(conf.dim_used)//3, 3
            )
            sequences_gt=batch[
              :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
            ].view(-1, conf.output_n, len(conf.dim_used)//3, 3)

            sequences_predict=model(sequences_train, adj_mat)
            
            loss=mpjpe_error(sequences_predict,sequences_gt)

            if (cnt + 1)% conf.log_step == 0:
              print('[Epoch: %d, Iteration: %5d]  validation loss: %.3f' %(epoch + 1, cnt + 1, loss.item()))
            
            running_loss+=loss*batch_dim

            progress_bar.update(task_id=val_task, advance=1)
            progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))
          
          val_loss.append(running_loss.detach().cpu()/n)
          
          if running_loss/n < val_loss_best:
            val_loss_best = running_loss/n

            torch.save({
              'epoch': epoch + 1,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(),
              'train_loss': train_loss,
              'val_loss': val_loss
            }, f"{ckpt_dir}/{conf.model_name}_best_val_loss.pt")

      if conf.use_scheduler:
        scheduler.step()

      if conf.save_and_plot and epoch%5==0:
        torch.save(
          {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
          }, 
          f"{ckpt_dir}/{conf.model_name}_epoch_{str(epoch + 1)}.pt"
        )

      wandb.log({
          "epoch": epoch,
          "loss/train": train_loss[-1],
          "loss/val": val_loss[-1]
      })
      
      if epoch == 0:
        rich.print(f"epoch: [bold][#B22222]{epoch + 1}[/#B22222][/b], train loss: [bold][#6495ED]{train_loss[-1]:.3f}[/#6495ED][/b], val loss: [b][#008080]{val_loss[-1]:.3f}[/#008080][/b]")
      else:
        rich.print(f"epoch: [bold][#B22222]{epoch + 1}[/#B22222][/b], train loss: [bold][#6495ED]{train_loss[-1]:.3f}[/#6495ED][/b] ([#6495ED][b]{(train_loss[-1] - train_loss[-2]):.3f}[/#6495ED][/b]), val loss: [b][#008080]{val_loss[-1]:.3f}[/#008080][/b] ([b][#008080]{(val_loss[-1] - val_loss[-2]):.3f}[/#008080][/b])")

  wandb.log({
    "best_loss/train": train_loss_best,
    "best_loss/val": val_loss_best,
  })     


# launch training
train(data_loader, vald_loader, path_to_save_model=conf.model_path)

# TODO implement changes before using it!
def test(ckpt_path=None):
    
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])

    model.eval()
    
    accum_loss=0
    
    n_batches=0 # number of batches for all the sequences
    
    actions=define_actions(conf.actions_to_consider_test)
    
    counter=0
    
    for action in actions:
      running_loss=0
      n=0
      
      dataset_test = datasets.Datasets(conf.path, conf.input_n, conf.output_n, conf.skip_rate, split=2, actions=[action])

      test_loader = DataLoader(dataset_test, batch_size=conf.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
      
      for cnt,batch in enumerate(test_loader):
        with torch.no_grad():

          batch=batch.to(device)
          
          batch_dim=batch.shape[0]
          
          n+=batch_dim

          all_joints_seq=batch.clone()[:, conf.input_n:conf.input_n+conf.output_n, :]

          sequences_train=batch[:, 0:conf.input_n, conf.dim_used].view(
            -1, conf.input_n, len(conf.dim_used)//3, 3
          ).permute(0,3,1,2)
          sequences_gt=batch[:, conf.input_n:conf.input_n+conf.output_n, :]
          
          sequences_predict=model(sequences_train).view(-1, conf.output_n, conf.joints_to_consider_n, 3)

          counter += 1

          sequences_predict=sequences_predict.contiguous().view(-1, conf.output_n, len(conf.dim_used))

          all_joints_seq[:, :, conf.dim_used] = sequences_predict

          all_joints_seq[:, :, conf.index_to_ignore] = all_joints_seq[:, :, conf.index_to_equal]

          loss=mpjpe_error(all_joints_seq.view(-1, conf.output_n, 32, 3), sequences_gt.view(-1, conf.output_n, 32, 3))
          running_loss+=loss*batch_dim
          accum_loss+=loss*batch_dim

      n_batches+=n

ckpt_path = f"{ckpt_dir}/{conf.model_name}_best_val_loss.pt"
test(ckpt_path)
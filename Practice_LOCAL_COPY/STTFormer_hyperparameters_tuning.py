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

import warnings; warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

USE_PROGRESS_BARS = True
if USE_PROGRESS_BARS:
  from utils.progress import get_progress_bar
  progress_bar = get_progress_bar()

USE_RICH_PRINT = True
if USE_RICH_PRINT:
  from rich import print



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

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

# Load Data
print('Loading Train Dataset...')
dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=0)
print('Loading Validation Dataset...')
vald_dataset = datasets.Datasets(path,input_n,output_n,skip_rate, split=1)

batch_size=256
lim_n_batches_percent = 0.01

print('>>> Training dataset length: {:d}'.format(dataset.__len__()))
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)#

print('>>> Validation dataset length: {:d}'.format(vald_dataset.__len__()))
vald_loader = DataLoader(vald_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

from models.sttr.sttformer import Model

n_heads = 8
kernel_size = [3,3]
att_drop=0

model = Model(
  num_joints=joints_to_consider,
  num_frames=input_n, num_frames_out=output_n, num_heads=n_heads,
  num_channels=3, kernel_size=kernel_size, use_pes=True,
  att_drop=att_drop
).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

# Arguments to setup the optimizer
lr=0.2 # learning rate

use_scheduler=True # use MultiStepLR scheduler
milestones=[10,30]   # the epochs after which the learning rate is adjusted by gamma
gamma=0.1 #gamma correction to the learning rate, after reaching the milestone epochs
step_size=30

weight_decay=0.00000001 # weight decay (L2 penalty)

momentum=0.3
nesterov=True

optimizer=optim.Adam(model.parameters(),lr=lr,weight_decay=weight_decay, amsgrad=False)
# optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

if use_scheduler:
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

clip_grad=None # select max norm to clip gradients
# Argument for training
# n_epochs=41
# log_step = 200

n_epochs = 41
log_step = 99999
log_epoch = 1 

train_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
ckpt_dir = f"{model_path}{train_id}"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None


train_config = {
  "batch_size": batch_size,
  "lim_n_batches_percent": lim_n_batches_percent,
  "n_heads": n_heads,
  "kernel_size": kernel_size,
  "att_drop": att_drop,
  "model": model,
  "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
  "lr": lr,
  "use_scheduler": use_scheduler,
  "milestones": milestones,
  "gamma": gamma,
  "step_size": step_size,
  "weight_decay": weight_decay,
  "momentum": momentum,
  "nesterov": nesterov,
  "optimizer": optimizer,
  "scheduler": scheduler if use_scheduler else None,
  "clip_grad": clip_grad,
  "n_epochs": n_epochs,
  "train_id": train_id
}

def train(data_loader,vald_loader, path_to_save_model=None):

  n_train_batches = int(len(data_loader) * lim_n_batches_percent) + 1 
  n_val_batches = int(len(vald_loader) * lim_n_batches_percent) + 1 

  if USE_PROGRESS_BARS:

    epoch_task = progress_bar.add_task("[bold][#B22222]Epoch progress...", total=n_epochs-1)  
    train_task = progress_bar.add_task("[bold][#6495ED]Train batches progress...", total=n_train_batches)  
    val_task = progress_bar.add_task("[bold][#008080]Val batches progress...", total=n_val_batches)  

    progress_bar.start()

  wandb.init(
    project="Spatio-Temporal Transformer hyperparameters-fine-tuning",
    config=train_config
  )

  train_loss = []
  val_loss = []
  train_loss_best = 100000000
  val_loss_best   = 100000000

  dim_used = np.array([6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30, 31, 32, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
                    46, 47, 51, 52, 53, 54, 55, 56, 57, 58, 59, 63, 64, 65, 66, 67, 68,
                    75, 76, 77, 78, 79, 80, 81, 82, 83, 87, 88, 89, 90, 91, 92])

  for epoch in range(n_epochs-1):
      
      if USE_PROGRESS_BARS:
        progress_bar.reset(task_id=train_task)
        progress_bar.reset(task_id=val_task)

      running_loss=0
      n=0
      model.train()
      for cnt,batch in list(enumerate(data_loader))[:n_train_batches]:
          batch=batch.float().to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim

          sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

          optimizer.zero_grad()
          sequences_predict=model(sequences_train).view(-1, output_n, joints_to_consider, 3)

          loss=mpjpe_error(sequences_predict,sequences_gt)

          loss.backward()
          if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(),clip_grad)

          optimizer.step()
          running_loss += loss*batch_dim

          if running_loss/n < train_loss_best:
            train_loss_best = running_loss/n

            torch.save(
              {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
              }, 
              f"{ckpt_dir}/{model_name}_best_train_loss.pt"
            )

          if USE_PROGRESS_BARS:
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


              sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
              sequences_gt=batch[:, input_n:input_n+output_n, dim_used].view(-1,output_n,len(dim_used)//3,3)

              sequences_predict=model(sequences_train).view(-1, output_n, joints_to_consider, 3)
              loss=mpjpe_error(sequences_predict,sequences_gt)

              running_loss+=loss*batch_dim

              if USE_PROGRESS_BARS:
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
              }, f"{ckpt_dir}/{model_name}_best_val_loss.pt")





      if use_scheduler:
        scheduler.step()

      if save_and_plot and epoch%log_epoch==0:
          torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss
            }, f"{ckpt_dir}/{model_name}_epoch_{str(epoch + 1)}.pt")

      wandb.log({
          "epoch": epoch,
          "loss/train": train_loss[-1],
          "loss/val": val_loss[-1]
      })

      if epoch == 0:
        print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b] | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b]")
      else:
        print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b], best (step): {train_loss_best:07.3f} | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b], best: {val_loss_best:07.3f}")

      wandb.log({
        "best_loss/train": np.array(train_loss).min(),
        "best_loss_epoch/train": np.array(train_loss).argmin(),
        "best_loss/val": np.array(val_loss).min(),
        "best_loss_in_step/train": train_loss_best,
        "best_loss_in_step/val": val_loss_best
      })

  final_epoch_print = f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b], best (step): {train_loss_best:07.3f} | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b], best: {val_loss_best:07.3f}"

  return final_epoch_print

save_and_plot = False # save the model and plot the loss. Change to True if you want to save the model and plot the loss

final_epoch_print = train(data_loader,vald_loader, path_to_save_model=model_path)

def test(ckpt_path, final_epoch_print):

    action_loss_dict = {}

    # model.load_state_dict(torch.load(ckpt_path))
    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    # print('model loaded')
    model.eval()
    accum_loss=0
    n_batches=0 # number of batches for all the sequences

    actions=define_actions(actions_to_consider_test)

    if USE_PROGRESS_BARS:
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
      dataset_test = datasets.Datasets(path,input_n,output_n,skip_rate, split=2,actions=[action])
      #print('>>> test action for sequences: {:d}'.format(dataset_test.__len__()))
      test_loader = DataLoader(dataset_test, batch_size=batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
      n_test_batches = int(len(test_loader) * lim_n_batches_percent) + 1   
      
      for cnt,batch in list(enumerate(test_loader))[:n_test_batches]:
        with torch.no_grad():

          batch=batch.to(device)
          batch_dim=batch.shape[0]
          n+=batch_dim


          all_joints_seq=batch.clone()[:, input_n:input_n+output_n,:]

          sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          sequences_gt=batch[:, input_n:input_n+output_n, :]


          running_time = time.time()
          sequences_predict=model(sequences_train).view(-1, output_n, joints_to_consider, 3)
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
    #   print(str(action),': ', str(np.round((running_loss/n).item(),1)))
      action_loss_dict[f"loss/test/{action}"] = np.round((running_loss/n).item(),1)

      n_batches+=n

      if USE_PROGRESS_BARS: 
        progress_bar.advance(task_id=actions_task, advance=1)
    
    
    # print('Average: '+str(np.round((accum_loss/n_batches).item(),1)))
    # print('Prediction time: ', totalll/counter)

    action_loss_dict["loss/test"] = np.round((accum_loss/n_batches).item(),1)
    action_loss_dict["best_loss/test"] = np.round((accum_loss/n_batches).item(),1)
    wandb.log(action_loss_dict)

    print(f"{final_epoch_print} | test loss (avg of all actions): {np.round((running_loss/n).item(),1)}")

# ckpt_path = './checkpoints/h36m_3d_25frames_ckpt_epoch_0.pt' # Change the epoch according to the validation curve
ckpt_path = f"{ckpt_dir}/{model_name}_best_val_loss.pt"
# test(ckpt_path)
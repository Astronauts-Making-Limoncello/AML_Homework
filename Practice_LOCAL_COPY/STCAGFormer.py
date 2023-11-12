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

import rich
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn, TimeRemainingColumn, SpinnerColumn

from utils.progress import get_progress_bar

import STCAGFormer_config as conf 

from utils.masking import causal_mask

progress_bar = get_progress_bar()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

if conf.actions_to_consider_train is not None:
  print(f"WARNING: training only on these actions --> {conf.actions_to_consider_train}")

# Load Data
dataset = datasets.Datasets(conf.path, conf.input_n, conf.output_n, conf.skip_rate, split=0, actions=conf.actions_to_consider_train)
vald_dataset = datasets.Datasets(conf.path,conf.input_n,conf.output_n,conf.skip_rate, split=1, actions=conf.actions_to_consider_train)

data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0, pin_memory=True)
vald_loader = DataLoader(vald_dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0, pin_memory=True)

from models.STCAGFormer.STAGEncoder import STAGEncoder
from models.STCFormer.SpatioTemporalDecoder import SpatioTemporalDecoder
from models.STCFormer.SpatioTemporalTransformer import SpatioTemporalTransformer

st_encoder = STAGEncoder(
  num_joints=conf.num_joints, num_frames=conf.num_frames, num_frames_out=conf.num_frames_out,
  num_heads=conf.num_heads_encoder, num_channels=conf.in_features, out_features=conf.out_features,
  kernel_size=[3, 3], config=conf.config
)

st_decoder = SpatioTemporalDecoder(
  conf.in_features, conf.hidden_features, conf.out_features, conf.num_joints,
  conf.num_frames, conf.num_frames_out,
  conf.num_heads_decoder, conf.use_skip_connection_decoder,
  conf.num_decoder_blocks
)

# alternative POV (referencing this Transformer implementaiton https://github.com/hkproj/pytorch-transformer): 
# seq_len is num_frames_out in decoder, so gotta use num_frames_out
decoder_mask_s = causal_mask((1, 1, 1, conf.num_joints, conf.num_joints)).to(device)

# since we are in the decoder, mask must have same shape of attn_t
# REMEMBER, NO temporal explosion applied to {q,k,v}_t, because multi-dimensional mat muls to produce attn_t do NOT require it
# so, we have to mask a (num_frames_out, num_frames) dimensional matrix  
decoder_mask_t = causal_mask((1, 1, 1, conf.num_frames_out, conf.num_frames)).to(device)

model = st_transformer = SpatioTemporalTransformer(
    st_encoder=st_encoder, st_decoder=st_decoder,
    num_frames=conf.num_frames, num_joints=conf.num_joints, in_features=conf.in_features
).to(device)

print('total number of parameters of the network is: '+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

optimizer=optim.Adam(model.parameters(),lr=conf.lr,weight_decay=conf.weight_decay, amsgrad=False)
# optimizer=optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)

if conf.use_scheduler:
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=conf.gamma, verbose=True)

train_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
ckpt_dir = f"{conf.model_path}{train_id}"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None

epoch_offset = 0
resume_from_ckpt = None

if resume_from_ckpt is not None:
  print(f"RESUMING FROM CHECKPOINT {resume_from_ckpt}...")
  
  ckpt = torch.load(resume_from_ckpt)

  model.load_state_dict(ckpt["model_state_dict"])
  print(f"MODEL STATE DICT LOADED")
  optimizer.load_state_dict(ckpt["optimizer_state_dict"])
  print(f"OPTIMIZER STATE DICT LOADED")

  epoch_offset = ckpt["epoch"] - 1

train_config = conf.get_config_dict().update(
  {
    "decoder_mask_s": decoder_mask_s,
    "decoder_mask_t": decoder_mask_t,
    "checkpoint_path": resume_from_ckpt,
    "checkpoint_epoch": ckpt["epoch"] if resume_from_ckpt is not None else -69,
    "model": model,
    "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    "optimizer": optimizer,
    "scheduler": scheduler if conf.use_scheduler else None,
    "train_id": train_id
  }
)

progress_bar.start()

def train(data_loader,vald_loader, path_to_save_model=None):

  n_train_batches = int(len(data_loader) * conf.lim_n_batches_percent) + 1 
  n_val_batches = int(len(vald_loader) * conf.lim_n_batches_percent) + 1 

  epoch_task = progress_bar.add_task("[bold][#B22222]Epoch progress...", total=conf.n_epochs-1)  
  train_task = progress_bar.add_task("[bold][#6495ED]Train batches progress...", total=n_train_batches)  
  val_task = progress_bar.add_task("[bold][#008080]Val batches progress...", total=n_val_batches)  

  wandb.init(
    project="Custom-model-SpatioTemporalTransformer-their-encoder",
    config=train_config
  )

  train_loss = []
  val_loss = []
  train_loss_best = 100000000
  val_loss_best   = 100000000

  for epoch in range(conf.n_epochs-1):
      
      progress_bar.reset(task_id=train_task)
      progress_bar.reset(task_id=val_task)

      running_loss=0
      
      n=0
      
      model.train()
      
      for cnt,batch in list(enumerate(data_loader))[:n_train_batches]:
          
          batch=batch.float().to(device)
          
          batch_dim=batch.shape[0]
          
          n+=batch_dim

          # TODO encoders based on their code require the permute
          # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          # TODO encoders based on our code do NOT require the permute
          sequences_train=batch[:, 0:conf.input_n, conf.dim_used].view(
            -1,conf.input_n,len(conf.dim_used)//3,3
          )
          sequences_gt=batch[
            :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
          ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

          optimizer.zero_grad()
          
          sequences_predict=model(
            src=sequences_train, tgt=sequences_gt, 
            tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
          )
          
          sequences_predict=sequences_predict.view(-1, conf.output_n, conf.joints_to_consider, 3)

          loss=mpjpe_error(sequences_predict,sequences_gt)

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

              # TODO encoders based on their code require the permute
              # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
              # TODO encoders based on our code do NOT require the permute
              sequences_train=batch[
                :, 0:conf.input_n, conf.dim_used
              ].view(-1,conf.input_n,len(conf.dim_used)//3,3)
              sequences_gt=batch[
                :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
              ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

              # sequences_predict=model(sequences_train).view(-1, output_n, joints_to_consider, 3)
              sequences_predict=model(
                src=sequences_train, tgt=sequences_gt,
                tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
              )
              sequences_predict=sequences_predict.view(
                -1, conf.output_n, conf.joints_to_consider, 3
              )

              loss=mpjpe_error(sequences_predict,sequences_gt)

              running_loss+=loss*batch_dim

              progress_bar.update(task_id=val_task, advance=1)
              progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))


          val_loss.append(running_loss.detach().cpu()/n)
          if running_loss/n < val_loss_best:
            val_loss_best = running_loss/n

            torch.save(
              {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
              }, 
              f"{ckpt_dir}/{conf.model_name}_best_val_loss.pt"
            )

      if conf.use_scheduler:
        scheduler.step()

      if conf.save_and_plot and epoch%conf.log_epoch==0:
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
        "epoch": epoch + epoch_offset,
        "loss/train": train_loss[-1],
        "loss/val": val_loss[-1]
      })

      if epoch == 0:
        print(f"epoch: {epoch + 1}, train loss: {train_loss[-1]:.3f}, val loss: {val_loss[-1]:.3f}")
      else:
        print(f"epoch: {epoch + 1}, train loss: {train_loss[-1]:.3f} ({(train_loss[-1] - train_loss[-2]):.3f}), val loss: {val_loss[-1]:.3f} ({(val_loss[-1] - val_loss[-2]):.3f})")


  wandb.log({
    "best_loss/train": np.array(train_loss).min(),
    "best_loss_epoch/train": np.array(train_loss).argmin() + epoch_offset,
    "best_loss/val": np.array(val_loss).min(),
    "best_loss_epoch/val": np.array(val_loss).argmin() + epoch_offset
  })


train(data_loader,vald_loader, path_to_save_model=conf.model_path)

def test(ckpt_path=None):

    action_loss_dict = {}

    model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
    
    model.eval()
    
    accum_loss=0
    
    n_batches=0 # number of batches for all the sequences

    actions=define_actions(conf.actions_to_consider_test)
    actions_task = progress_bar.add_task("[bold][#9400D3]Action test batches progress...", total=len(actions))

    counter=0
    
    for action in actions:
      
      running_loss=0
      
      n=0
      
      dataset_test = datasets.Datasets(conf.path,conf.input_n,conf.output_n,conf.skip_rate, split=2,actions=[action], use_progress_bar=False)

      test_loader = DataLoader(dataset_test, batch_size=conf.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
      
      n_test_batches = int(len(test_loader) * conf.lim_n_batches_percent) + 1   
      
      for cnt,batch in list(enumerate(test_loader))[:n_test_batches]:
        
        with torch.no_grad():

          batch=batch.to(device)
          
          batch_dim=batch.shape[0]
          
          n+=batch_dim

          all_joints_seq=batch.clone()[:, conf.input_n:conf.input_n+conf.output_n,:]

          # TODO encoders based on their code require the permute
          # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          # TODO encoders based on our code do NOT require the permute
          sequences_train=batch[:, 0:conf.input_n, conf.dim_used].view(
            -1,conf.input_n,len(conf.dim_used)//3,3
          )
          # sequences_gt=batch[:, input_n:input_n+output_n, :]
          sequences_gt=batch[
            :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
          ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

          sequences_predict=model(
            src=sequences_train, tgt=sequences_gt,
            tgt_mask_s=decoder_mask_s, tgt_mask_t=decoder_mask_t
          ).view(-1, conf.output_n, conf.joints_to_consider, 3)
          # print(f"sequences_predict.shape: {sequences_predict.shape}")
          #sequences_predict = model(sequences_train)
          counter += 1
          sequences_predict=sequences_predict.contiguous().view(
            -1, conf.output_n, len(conf.dim_used)
          )
          # print(f"sequences_predict.shape: {sequences_predict.shape}")

          all_joints_seq[:,:,conf.dim_used] = sequences_predict
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")

          all_joints_seq[:,:,conf.index_to_ignore] = all_joints_seq[:,:,conf.index_to_equal]
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")

          all_joints_seq = all_joints_seq.view(-1,conf.output_n,32,3)
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")
          # sequences_gt = sequences_gt.view(-1,output_n,32,3)
          # gotta retake it from the batch, in order to get the original number of joints (32)
          sequences_gt=batch[
            :, conf.input_n:conf.input_n+conf.output_n, :
          ]
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
    wandb.log(action_loss_dict)

# ckpt_path = './checkpoints/h36m_3d_25frames_ckpt_epoch_0.pt' # Change the epoch according to the validation curve
ckpt_path = f"{ckpt_dir}/{conf.model_name}_best_val_loss.pt"
test(ckpt_path)
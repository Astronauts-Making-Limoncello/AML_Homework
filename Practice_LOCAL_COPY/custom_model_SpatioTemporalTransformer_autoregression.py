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

from models.STCFormer.SpatioTemporalEncoder import SpatioTemporalEncoder
from models.STCFormer.SpatioTemporalDecoder import SpatioTemporalDecoder
from models.STCFormer.SpatioTemporalTransformer import SpatioTemporalTransformer
from utils.masking import causal_mask

import SpatioTemporalTransformer_autoregression_config as conf

from utils.progress import get_progress_bar

### --- DEVICE --- ###

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device,  '- Type:', torch.cuda.get_device_name(0))

### --- DEVICE --- ###

############################################################

### --- DATA --- ###

if conf.actions_to_consider_train is not None:
  print(f"[b][#FF0000]WARNING: training only on these actions --> {conf.actions_to_consider_train}")

# Load Data
dataset = datasets.Datasets(conf.path, conf.input_n, conf.output_n, conf.skip_rate, split=0, actions=conf.actions_to_consider_train)
vald_dataset = datasets.Datasets(conf.path, conf.input_n, conf.output_n, conf.skip_rate, split=1, actions=conf.actions_to_consider_train)

data_loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=0, pin_memory=True)#
vald_loader = DataLoader(vald_dataset, batch_size=conf.batch_size_val, shuffle=True, num_workers=0, pin_memory=True)

### --- DATA --- ###

############################################################

### --- MODEL --- ###

st_encoder = SpatioTemporalEncoder(
  conf.in_features_encoder, conf.hidden_features, conf.out_features_encoder, conf.num_joints,
  conf.num_frames, conf.num_frames, # it's encoder, so num_frames_out == num_frames
  conf.num_heads_encoder, conf.use_skip_connection_encoder,
  conf.num_encoder_blocks, 
  conf.dropout_encoder
)

st_decoder = SpatioTemporalDecoder(
  decoder_input_in_features=conf.in_features_decoder, 
  encoder_output_in_features=conf.out_features_encoder, 
  hidden_features=conf.hidden_features, out_features=conf.out_features_decoder, 
  num_joints=conf.num_joints,
  num_frames=conf.num_frames, num_frames_out=conf.num_frames_out,
  num_heads=conf.num_heads_decoder, use_skip_connection=conf.use_skip_connection_decoder,
  skip_connection_weight=conf.skip_connection_weight_decoder,
  num_decoder_blocks=conf.num_decoder_blocks, dropout=conf.dropout_decoder
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
    num_frames=conf.num_frames, num_joints=conf.num_joints, 
    in_features_encoder=conf.in_features_encoder, in_features_decoder=conf.in_features_decoder
).to(device)

print(f"Number of trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

### --- MODEL --- ###

############################################################

### --- OPTIMIZER --- ###

optimizer=optim.Adam(model.parameters(),lr=conf.lr,weight_decay=conf.weight_decay, amsgrad=conf.amsgrad)

### --- OPTIMIZER --- ###

############################################################

### --- LR SCHEDULER --- ###

if conf.use_scheduler:
  scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=conf.gamma, verbose=True)
  # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)

### --- LR SCHEDULER --- ###

############################################################

### --- TRAINING CONFIG --- ###

train_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
print(f"Train ID: {train_id}")

ckpt_dir = f"{conf.model_path}{train_id}"
os.makedirs(ckpt_dir) if not os.path.exists(ckpt_dir) else None


# resume_from_ckpt = f"{model_path}/2023_11_06_08_30_03/h36m_3d_25frames_ckpt_best_val_loss.pt"
if conf.resume_from_ckpt is not None:
  print(f"Resuming from checkpoint {conf.resume_from_ckpt}...")
  
  ckpt = torch.load(conf.resume_from_ckpt)

  model.load_state_dict(ckpt["model_state_dict"])
  print(f"Model state dict loaded!")
  optimizer.load_state_dict(ckpt["optimizer_state_dict"])
  print(f"Optimizer state dict loaded!")

  epoch_offset = ckpt["epoch"] - 1

train_config = conf.get_config_dict().update(
  {
    "decoder_mask_s": decoder_mask_s,
    "decoder_mask_t": decoder_mask_t,
    "checkpoint_epoch": ckpt["epoch"] if conf.resume_from_ckpt is not None else -69,
    "model": model,
    "num_trainable_parameters": str(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    "optimizer": optimizer,
    "scheduler": scheduler if conf.use_scheduler else None,
    "train_id": train_id
  }
)

### --- TRAINING CONFIG --- ###

############################################################

progress_bar = get_progress_bar()

progress_bar.start()

def train(data_loader,vald_loader, path_to_save_model=None):

  n_train_batches = int(len(data_loader) * conf.lim_n_batches_percent) + 1 
  n_val_batches = int(len(vald_loader) * conf.lim_n_batches_percent) + 1 

  epoch_task = progress_bar.add_task("[bold][#B22222]Epoch progress...", total=conf.n_epochs-1)  
  train_task = progress_bar.add_task("[bold][#6495ED]Train batches progress...", total=n_train_batches)  
  val_task = progress_bar.add_task("[bold][#008080]Val batches progress...", total=n_val_batches)  

  wandb.init(
    project="Custom-model-SpatioTemporalTransformer",
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
      
        batch_size_current_batch = batch.shape[0]
        
        batch=batch.float().to(device)
        
        batch_dim=batch.shape[0]
        
        n+=batch_dim

        # NOTE: encoders based on their code require the permute
        # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
        # NOTE: encoders based on our code do NOT require the permute
        sequences_train = batch[:, 0:conf.input_n, conf.dim_used].view(-1, conf.input_n, conf.num_joints, 3)
        # sequences_train.shape: batch_size, input_n, num_joints, in_features

        sequences_gt=batch[
          :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
        ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

        # working on a copy of sequences_gt so as we keep the original untoched for the loss computation
        sequences_gt_autoreg = batch[
          :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
        ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

        # adding num_special_tokens extra features
        # to accomodate encoding of special tokens (start of sequence, end of sequence)
        in_features_pad_tgt = torch.zeros(
          batch_size_current_batch, conf.output_n, conf.num_joints, conf.num_special_tokens_decoder
        ).to(device)
        sequences_gt_autoreg = torch.cat((sequences_gt_autoreg, in_features_pad_tgt), dim=-1)

        # removing last element of gt sequence, in order to shift gt sequence to right by one position
        # by means of adding the start of sequence token
        sequences_gt_autoreg = sequences_gt_autoreg[:, :-1, :, :]
        # print(f"sequences_gt_autoreg.shape: {sequences_gt_autoreg.shape}")

        # creating the start of sequence token
        # see autoreg section comments of report for full detail on shapes, etc.
        start_of_sequence_token_decoder = torch.zeros(
          (batch_size_current_batch, 1, conf.num_joints, conf.in_features_decoder)
        ).to(device)
        # print(f"start_of_sequence_token_decoder.shape: {start_of_sequence_token_decoder.shape}")
            
        # concatenating sequences_gt_autoreg to start_of_sequence_token along the temporal dimension
        # effectively shifts the output by 1 position, by means of adding the
        # start of sequence token
        sequences_gt_autoreg = torch.cat(
          tensors=(start_of_sequence_token_decoder, sequences_gt_autoreg),
          dim=1 
        )

        # feature vector of special token (which is concatenated to feature vector of data!)
        # is a one-hot encoding vector, where each special token type has its own dimension set to 1
        # so, setting to 1 the dimension of the start_of_sequence special token to 1
        sequences_gt_autoreg[:, 0, :, conf.in_features + conf.start_of_sequence_token_offset] = 1.

        optimizer.zero_grad()

        decoder_mask_s_self_attn = causal_mask(
          (batch_size_current_batch, conf.num_heads_decoder, conf.num_frames_out, conf.num_joints, conf.num_joints)
        ).to(device)
        decoder_mask_s_cross_attn = causal_mask(
          (batch_size_current_batch, conf.num_heads_decoder, conf.num_frames_out, conf.num_joints, conf.num_joints)
        ).to(device)
        
        decoder_mask_t_cross_attention = causal_mask(
          (batch_size_current_batch, conf.num_heads_decoder, conf.num_joints, conf.num_frames_out, conf.num_frames_out)
        ).to(device)
        decoder_mask_t_self_attention = causal_mask(
          (batch_size_current_batch, conf.num_heads_decoder, conf.num_joints, conf.num_frames_out, conf.num_frames_out)
        ).to(device)

        sequences_predict=model(
          src=sequences_train, 
          tgt=sequences_gt_autoreg, 
          src_mask_s=conf.encoder_mask_s, src_mask_t=conf.encoder_mask_t,
          tgt_mask_s_self_attn=decoder_mask_s_self_attn, tgt_mask_s_cross_attn=decoder_mask_s_cross_attn, 
          tgt_mask_t_self_attn=decoder_mask_t_self_attention, tgt_mask_t_cross_attn=decoder_mask_t_cross_attention
        )

        # special token num_special_token_{encoder,decoder}-long vectors 
        # get appended to the end of the in_feature-long feature vector
        # so, using slicing, we can separate feature vector from special token encoding vectors
          
        sequences_predict_special_tokens =           sequences_predict[:, :, :, conf.in_features:]
        sequences_gt_special_tokens      = sequences_gt_autoreg[:, :, :, conf.in_features:]
        
        sequences_predict = sequences_predict[:, :, :, :conf.in_features]

        loss_mpjpe = mpjpe_error(sequences_predict,sequences_gt)
        
        # computing l_2 loss between features of special tokens
        # it will be used as second term of the training loss
        loss_special_tokens = l2_error(
          sequences_predict_special_tokens, sequences_gt_special_tokens, 
          conf.num_special_tokens_decoder
        )

        loss = loss_mpjpe + loss_special_tokens
        loss.backward()
        
        if conf.clip_grad is not None:
          torch.nn.utils.clip_grad_norm_(model.parameters(), conf.clip_grad)

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
            f"{ckpt_dir}/{conf.model_name}_best_train_loss.pt"
          )

        progress_bar.update(task_id=train_task, advance=1)
        progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))

      train_loss.append(running_loss.detach().cpu()/n)
      
      model.eval()
    
      with torch.no_grad():
          
          running_loss=0
          
          n=0
          
          for cnt,batch in list(enumerate(vald_loader))[:n_val_batches]:
            
            batch_size_current_batch = batch.shape[0]
            
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

            start_of_sequence_token_decoder = torch.zeros(
              (batch_size_current_batch, 1, conf.num_joints, conf.in_features_decoder)
            ).to(device)
            
            # setting the decoder output to the start of sequence token
            tgt = start_of_sequence_token_decoder

            for frame_out in range(conf.num_frames_out):
              decoder_mask_s_autoreg_self_attn = causal_mask(
                (batch_size_current_batch, conf.num_heads_decoder, conf.num_frames_out, conf.num_joints, conf.num_joints)
              ).to(device)
              decoder_mask_s_autoreg_cross_attn = causal_mask(
                (batch_size_current_batch, conf.num_heads_decoder, conf.num_frames_out, conf.num_joints, conf.num_joints)
              ).to(device)

              decoder_mask_t_autoreg_cross_attn = causal_mask(
                (batch_size_current_batch, conf.num_heads_decoder, conf.num_joints, conf.num_frames_out, conf.num_frames_out)
              ).to(device)
              decoder_mask_t_autoreg_self_attn = causal_mask(
                (batch_size_current_batch, conf.num_heads_decoder, conf.num_joints, conf.num_frames_out, conf.num_frames_out)
              ).to(device)

              decoder_output = model(
                src=sequences_train, tgt=tgt, 
                tgt_mask_s_self_attn=decoder_mask_s_autoreg_self_attn, 
                tgt_mask_t_self_attn=decoder_mask_t_autoreg_self_attn,
                tgt_mask_s_cross_attn=decoder_mask_s_autoreg_cross_attn, 
                tgt_mask_t_cross_attn=decoder_mask_t_autoreg_cross_attn
              )

              tgt = torch.cat((tgt, decoder_output[:, -1:, :, :]), dim=1)

              # working on a copy of sequences_gt so as we keep the original untoched for the loss computation
              # and to keep compatibility for when autoreg=False
              sequences_gt_autoreg = batch[
                :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
              ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

              # adding num_special_tokens extra features
              # to accomodate encoding of special tokens (start of sequence, end of sequence)
              in_features_pad_tgt = torch.zeros(
                batch_size_current_batch, conf.output_n, conf.num_joints, conf.num_special_tokens_decoder
              ).to(device)
              sequences_gt_autoreg = torch.cat((sequences_gt_autoreg, in_features_pad_tgt), dim=-1)

            # removing start of sequence special token from the autoregressive predictions
            sequences_predict = tgt[:, 1:, :, :]

            sequences_predict_special_tokens =           sequences_predict[:, :, :, conf.in_features:]
            sequences_gt_special_tokens      = sequences_gt_autoreg[:, :, :, conf.in_features:]
        
            sequences_predict = sequences_predict[:, :, :, :conf.in_features]

            loss_mpjpe = mpjpe_error(sequences_predict,sequences_gt)

            # computing l_2 loss between features of special tokens
            # it will be used as second term of the training loss
            loss_special_tokens = l2_error(
              sequences_predict_special_tokens, sequences_gt_special_tokens, 
              conf.num_special_tokens_decoder
            )

            loss = loss_mpjpe + loss_special_tokens

            running_loss+=loss*batch_dim

            progress_bar.update(task_id=val_task, advance=1)
            progress_bar.update(task_id=epoch_task, advance=1/(n_train_batches + n_val_batches))

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


          val_loss.append(running_loss.detach().cpu()/n)



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
        print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b] | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b]")
      else:
        print(f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b], best (step): {train_loss_best:07.3f} | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b], best: {val_loss_best:07.3f}")


  wandb.log({
    "best_loss/train": np.array(train_loss).min(),
    "best_loss_epoch/train": np.array(train_loss).argmin() + epoch_offset,
    "best_loss/val": np.array(val_loss).min(),
    "best_loss_in_step/train": train_loss_best,
    "best_loss_in_step/val": val_loss_best
  })

  final_epoch_print = f"epoch: [bold][#B22222]{(epoch + 1):04}[/#B22222][/b] | train loss: [bold][#6495ED]{train_loss[-1]:07.3f}[/#6495ED][/b], best (step): {train_loss_best:07.3f} | val loss: [b][#008080]{val_loss[-1]:07.3f}[/#008080][/b], best: {val_loss_best:07.3f}"

  return final_epoch_print

final_epoch_print = train(data_loader,vald_loader, path_to_save_model=conf.model_path)

def test(ckpt_path, final_epoch_print):

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

      dataset_test = datasets.Datasets(
        conf.path, conf.input_n, conf.output_n, conf.skip_rate, split=2,actions=[action], use_progress_bar=False
      )
      test_loader = DataLoader(dataset_test, batch_size=conf.batch_size_test, shuffle=False, num_workers=0, pin_memory=True)
      n_test_batches = int(len(test_loader) * conf.lim_n_batches_percent) + 1   
      
      for cnt,batch in list(enumerate(test_loader))[:n_test_batches]:
        
        with torch.no_grad():
          
          batch_size_current_batch = batch.shape[0]
          
          batch=batch.to(device)
          
          batch_dim=batch.shape[0]
          
          n+=batch_dim

          all_joints_seq=batch.clone()[:, conf.input_n:conf.input_n+conf.output_n,:]

          # TODO encoders based on their code require the permute
          # sequences_train=batch[:, 0:input_n, dim_used].view(-1,input_n,len(dim_used)//3,3).permute(0,3,1,2)
          # TODO encoders based on our code do NOT require the permute
          sequences_train=batch[:, 0:conf.input_n, conf.dim_used].view(
            -1, conf.input_n, len(conf.dim_used)//3, 3
          )
          sequences_gt=batch[
            :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
          ].view(-1, conf.output_n, len(conf.dim_used)//3, 3)

          start_of_sequence_token_decoder = torch.zeros(
            (batch_size_current_batch, 1, conf.num_joints, conf.in_features_decoder)
          ).to(device)
          
          # setting the decoder output to the start of sequence token
          tgt = start_of_sequence_token_decoder

          for frame_out in range(conf.num_frames_out):
            decoder_mask_s_autoreg_self_attn = causal_mask(
              (batch_size_current_batch, conf.num_heads_decoder, conf.num_frames_out, conf.num_joints, conf.num_joints)
            ).to(device)
            decoder_mask_s_autoreg_cross_attn = causal_mask(
              (batch_size_current_batch, conf.num_heads_decoder, conf.num_frames_out, conf.num_joints, conf.num_joints)
            ).to(device)

            decoder_mask_t_autoreg_cross_attn = causal_mask(
              (batch_size_current_batch, conf.num_heads_decoder, conf.num_joints, conf.num_frames_out, conf.num_frames_out)
            ).to(device)
            decoder_mask_t_autoreg_self_attn = causal_mask(
              (batch_size_current_batch, conf.num_heads_decoder, conf.num_joints, conf.num_frames_out, conf.num_frames_out)
            ).to(device)

            decoder_output = model(
              src=sequences_train, tgt=tgt, 
              tgt_mask_s_self_attn=decoder_mask_s_autoreg_self_attn, 
              tgt_mask_t_self_attn=decoder_mask_t_autoreg_self_attn,
              tgt_mask_s_cross_attn=decoder_mask_s_autoreg_cross_attn, 
              tgt_mask_t_cross_attn=decoder_mask_t_autoreg_cross_attn
            )

            tgt = torch.cat((tgt, decoder_output[:, -1:, :, :]), dim=1)

            # working on a copy of sequences_gt so as we keep the original untoched for the loss computation
            # and to keep compatibility for when autoreg=False
            sequences_gt_autoreg = batch[
              :, conf.input_n:conf.input_n+conf.output_n, conf.dim_used
            ].view(-1,conf.output_n,len(conf.dim_used)//3,3)

            # adding num_special_tokens extra features
            # to accomodate encoding of special tokens (start of sequence, end of sequence)
            in_features_pad_tgt = torch.zeros(
              batch_size_current_batch, conf.output_n, conf.num_joints, conf.num_special_tokens_decoder
            ).to(device)
            sequences_gt_autoreg = torch.cat((sequences_gt_autoreg, in_features_pad_tgt), dim=-1)

          # removing start of sequence special token from the autoregressive predictions
          sequences_predict = tgt[:, 1:, :, :]

          sequences_predict_special_tokens =           sequences_predict[:, :, :, conf.in_features:]
          sequences_gt_special_tokens      = sequences_gt_autoreg[:, :, :, conf.in_features:]
      
          sequences_predict = sequences_predict[:, :, :, :conf.in_features]

          # computing l_2 loss between features of special tokens
          # it will be used as second term of the training loss
          loss_special_tokens = l2_error(
            sequences_predict_special_tokens, sequences_gt_special_tokens, 
            conf.num_special_tokens_decoder
          )

          sequences_predict=sequences_predict.contiguous().view(-1,conf.output_n,len(conf.dim_used))
          # print(f"sequences_predict.shape: {sequences_predict.shape}")
          # print(f"all_joints_seq.shape: {all_joints_seq.shape}")
          all_joints_seq[:,:,conf.dim_used] = sequences_predict
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")

          all_joints_seq[:,:,conf.index_to_ignore] = all_joints_seq[:,:,conf.index_to_equal]
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")

          all_joints_seq = all_joints_seq.view(-1,conf.output_n,32,3)
          # print(f"all_joints_seq.shape   : {all_joints_seq.shape}")
          # sequences_gt = sequences_gt.view(-1,output_n,32,3)
          # gotta retake it from the batch, in order to get the original number of joints (32)
          sequences_gt=batch[:, conf.input_n:conf.input_n+conf.output_n, :]
          # print(f"sequences_gt.shape     : {sequences_gt.shape}")


          loss_mpjpe = mpjpe_error(all_joints_seq,sequences_gt)
          
          loss = loss_mpjpe + loss_special_tokens

          running_loss+=loss*batch_dim
          accum_loss+=loss*batch_dim


      action_loss_dict[f"loss/test/{action}"] = np.round((running_loss/n).item(),1)

      n_batches+=n

      progress_bar.advance(task_id=actions_task, advance=1)
    

    print(f"{final_epoch_print} | test loss (avg of all actions): {np.round((running_loss/n).item(),1)}")
    

    action_loss_dict["loss/test"] = np.round((accum_loss/n_batches).item(),1)
    action_loss_dict["best_loss/test"] = np.round((accum_loss/n_batches).item(),1)
    wandb.log(action_loss_dict)

ckpt_path = f"{ckpt_dir}/{conf.model_name}_best_val_loss.pt"
test(ckpt_path, final_epoch_print)
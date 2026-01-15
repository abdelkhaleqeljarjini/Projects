import os
import gc
import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from Tokenizer import RLLMTokenizer
from sklearn.model_selection import train_test_split
from Dataloader import TrainDataset
from Model import RegressionLLM
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

L_RATE = 1e-5
MAX_LEN = 12
MAX_LEN_E = 512
NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_CORES = 0
dropout = 0.2
num_heads = 12
n_blocks = 2
d = 768

my_path = "path"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Data uploading and split
df = pd.read_csv(my_path+"file.csv")
df = df.sort_values(by=['user_id','timestamp'],ignore_index=True)

# test data will contain seen and unseen users, we neep to keep the whole samples for some users in the validation and test
user_train, user_test = train_test_split(df.user_id.unique(),test_size=0.1,shuffle=True,random_state=42)
user_val, user_test = train_test_split(user_test,test_size=0.5,shuffle=True,random_state=42)
df_train = df[df['user_id'].isin(user_train)].reset_index(drop=True)
df_test = df[df['user_id'].isin(user_test)].reset_index(drop=True)
df_val = df[df['user_id'].isin(user_val)].reset_index(drop=True)
train, test = train_test_split(df_train,test_size=0.1,shuffle=True,random_state=42)
val, test = train_test_split(test,test_size=0.5,shuffle=True,random_state=42)
df_train = train.sort_values(by=['user_id','timestamp'],ignore_index=True)
df_test = pd.concat([test,df_test],axis=0, ignore_index=True)
df_test = df_test.sort_values(by=['user_id','timestamp'],ignore_index=True)
df_val = pd.concat([val,df_val],axis=0, ignore_index=True)
df_val = df_val.sort_values(by=['user_id','timestamp'],ignore_index=True)

#Training loop
seed_val = 1024

rd.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

train = df_train.reset_index(drop=True)
val = df_val.reset_index(drop=True)

train_dataset = DatasetTS(train,MODEL_NAME)
val_dataset = DatasetTS(val,MODEL_NAME)

epoch_val_loss=[]
epoch_train_loss=[]
epoch=0
while epoch<NUM_EPOCHS:
    print('======== Epoch {:} / {:} ========'.format(epoch+1, NUM_EPOCHS))

    if epoch==0:
        model.to(device)
        optimizer = AdamW(model.parameters(),
                         lr=L_RATE,
                         eps=1e-8)
        scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2)
    else:
        model.to(device)

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,num_workers=NUM_CORES,collate_fn=dynamic_collate_fn)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,num_workers=NUM_CORES,collate_fn=dynamic_collate_fn)
    print('Training...')
    model.train()
    torch.set_grad_enabled(True)
    total_train_loss = []

    for i, batch in enumerate(train_dataloader):

        batch = [tensor.to(device) for tensor in batch]
        input_ids = batch[0]
        attention_mask = batch[1]
        input_y = batch[2]
        labels = batch[3]
        pad_mask_ts = batch[4]

        outputs = model(
            idx_e=input_ids,
            attention_mask=attention_mask,
            idx=input_y,
            targets=labels,
            pad_mask_ts=pad_mask_ts
            )
        loss = outputs['loss']
        total_train_loss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    scheduler.step()
    print('Validation...')
    gc.collect()
    torch.set_grad_enabled(False)
    model.eval()
    total_val_loss = []
    labels_list=[]
    preds_list=[]
    for j,val_batch in enumerate(val_dataloader):

        batch = [tensor.to(device) for tensor in batch]
        input_ids = batch[0]
        attention_mask = batch[1]
        input_y = batch[2]
        labels = batch[3]
        pad_mask_ts = batch[4]

        outputs = model(
            idx_e=input_ids,
            attention_mask=attention_mask,
            idx=input_y,
            targets=labels,
            pad_mask_ts=pad_mask_ts
            )
        loss = outputs['loss'].detach().cpu().numpy()
        total_val_loss.append(loss.item())

    current_train_loss = sum(total_train_loss)/len(total_train_loss)
    current_val_loss = sum(total_val_loss)/len(total_val_loss)
    print('train loss',current_train_loss)
    print('val_loss',current_val_loss)
    epoch_train_loss.append(current_train_loss)
    checkpoints = model.state_dict()
    if current_val_loss < min(epoch_val_loss,default=0):
      model_name = my_path + 'Best_model_weights.pth'
      torch.save(checkpoints, model_name)
    epoch_val_loss.append(current_val_loss)
    model_name = my_path + 'model_weights.pth'
    torch.save(checkpoints, model_name)
    epoch+=1




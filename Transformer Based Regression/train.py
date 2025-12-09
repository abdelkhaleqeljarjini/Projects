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
from train_data_loader import TrainDataset
from Model import DecoderRLLM

L_RATE = 1e-5
MAX_LEN = 5
MAX_LEN_E = 2000
NUM_EPOCHS = 100
BATCH_SIZE = 32
NUM_CORES = 0

my_path = "path"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(my_path+"file.csv")
df['x'] = df.iloc[:,[0,2,3,4]].astype(str).agg('[SEP]'.join,axis=1)

train, test, y_train, y_test = train_test_split(df[['x','valence']], df[['valence']],
                                                    test_size=0.2, random_state=42)
test, val, y_test, y_val = train_test_split(test[['x','valence']], test[['valence']],
                                                    test_size=0.5, random_state=42)

# %%time

seed_val = 1024

torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

train = train.reset_index(drop=True)
val = val.reset_index(drop=True)

train_dataset = TrainDataset(train)
val_dataset = TrainDataset(val)

epoch_val_loss=[]
epoch_train_loss=[]
epochs_without_improvement = 0
epoch=0
model = DecoderRLLM(30522,768)
while epochs_without_improvement<20 and epoch<NUM_EPOCHS:
    print('======== Epoch {:} / {:} ========'.format(epoch+1, NUM_EPOCHS))

    if epoch==0:
        model.to(device)
        optimizer = AdamW(model.parameters(),
                         lr=L_RATE,
                         eps=1e-8)
    else:
        model.to(device)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=BATCH_SIZE,
                                               shuffle=True,num_workers=NUM_CORES)
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=BATCH_SIZE,
                                               shuffle=True,num_workers=NUM_CORES)
    print('Training...')
    model.train()
    torch.set_grad_enabled(True)
    total_train_loss = []

    for i, batch in enumerate(train_dataloader):

        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        token_type_ids = batch[2].to(device)
        input_y = batch[3].to(device)
        labels = batch[4].to(device)

        outputs = model(
            idx=input_y,
            idx_e=input_ids,
            attention_mask=attention_mask,
            targets=labels,)
        loss = outputs['loss']
        total_train_loss.append(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    print('Validation...')
    gc.collect()
    torch.set_grad_enabled(False)
    model.eval()
    total_val_loss = []
    labels_list=[]
    preds_list=[]
    for j,val_batch in enumerate(val_dataloader):

        input_ids = val_batch[0].to(device)
        attention_mask = val_batch[1].to(device)
        token_type_ids = val_batch[2].to(device)
        labels = val_batch[4].to(device)
        input_y = val_batch[3].to(device)

        outputs = model(
            idx=input_y,
            idx_e=input_ids,
            attention_mask=attention_mask,
            targets=labels)
        loss = outputs['loss']
        total_val_loss.append(loss.item())
        val_preds = outputs['logits'].detach().cpu().numpy()
        val_labels = labels.detach().cpu().numpy()
        labels_list.extend(val_labels)
        preds_list.extend(val_preds)
    current_train_loss = sum(total_train_loss)/len(total_train_loss)
    current_val_loss = sum(total_val_loss)/len(total_val_loss)
    print('train loss',current_train_loss)
    print('val_loss',current_val_loss)
    epoch_train_loss.append(current_train_loss)
    if current_val_loss < min(epoch_val_loss,default=0):
      model_name = my_path + 'Best_model_weights'+'.pth'
      torch.save(model.state_dict(), model_name)
    epoch_val_loss.append(current_val_loss)
    model_name = my_path + 'model_weights'+'.pth'
    torch.save(model.state_dict(), model_name)

    epoch+=1

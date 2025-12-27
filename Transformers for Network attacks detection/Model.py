import os
import random
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import gc


#Hyperparameters
L_RATE = 1e-3
MAX_LEN = 512
NUM_EPOCHS = 100
BATCH_SIZE = 128
MODEL_NAME = 'bert-base-uncased' # For tokenizer
dropout = 0.2
NUM_CORES = os.cpu_count()

#Data path
path = '' # Math for the data files

#Device
device  = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Training & validation data
df_gl_train = pd.read_csv(path)
df_gl_val = pd.read_csv(path)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Data Loader
class TrainDataset(Dataset):

    def __init__(self,dataframe):
        self.df = dataframe

    def __getitem__(self,index):

        text = self.df.iloc[index]['x']
        #tokenizing the data
        encoded_dict = tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = MAX_LEN,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        #Extracting the model inputs, already tensors
        padded_tokens_list = encoded_dict['input_ids'].squeeze()
        att_mask = encoded_dict['attention_mask'].squeeze()
        target = torch.tensor(self.df.iloc[index]['Label_trsf'],dtype=torch.long)
        sample = (padded_tokens_list,att_mask,target)
        return sample

    def __len__(self):
        return len(self.df)

#Multi Head Attention
class MultiHA(nn.Module):
  def __init__(self, d, hs,causal_attention = False ,max_len=2000):
    super().__init__()
    self.causal_attention = causal_attention
    self.hs = hs
    self.d = d
    self.num_heads = d//hs # % of division should be zero
    self.query = nn.Linear(d, d, bias=False)
    self.key = nn.Linear(d, d, bias=False)
    self.value = nn.Linear(d, d, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.proj  = nn.Linear(d,d)

  def forward(self, x,y,pad_mask =None):
    batch,seq_len,emb_size = x.shape
    q = self.resize(self.query(y), batch, y.shape[1], self.num_heads, self.hs)
    k = self.resize(self.key(x), batch, seq_len, self.num_heads, self.hs)
    v = self.resize(self.value(x), batch, seq_len, self.num_heads, self.hs)
    att = q@k.transpose(-2,-1)
    att = att*self.hs**-0.5
    if pad_mask is not None:
      att = att.masked_fill(pad_mask[:, None, None, :]==0,float('-inf'))
    att = self.dropout(F.softmax(att, dim=-1))@v
    att = self.dropout(self.proj(att.transpose(1,2).reshape(batch,seq_len,self.d)))
    return att

  @staticmethod
  def resize(x, batch, seq_len, num_heads, hs):
    return x.view(batch,seq_len, num_heads, hs ).transpose(1,2)

#Feed Forward block 
class FeedForward(nn.Module):

  def __init__(self,d):
    super().__init__()
    self.ffw = nn.Sequential(
        nn.Linear(d,d),
        nn.GELU(),
        nn.Linear(d,d),
        nn.Dropout(dropout)
        )

  def forward(self,x):
    return self.ffw(x)

#Model
class Anomaly(nn.Module):
  def __init__(self,vs,d,num_heads,max_len=2000,num_cls=2):
    super().__init__()
    self.embedding = nn.Embedding(vs,d)
    self.pos_embedding = nn.Embedding(max_len,d) # Learned positional embeddings.
    self.ah = MultiHA(d=d,hs = d//num_heads,max_len=max_len)
    self.ln1 =nn.LayerNorm(d)
    self.ln2 =nn.LayerNorm(d)
    self.ffwd = FeedForward(d)
    self.lo = nn.Linear(d,num_cls)
    self.dropout = nn.Dropout(p=0.1)

  def forward(self,idx,pad_mask=None,targets=None):
    batch,seq_len = idx.shape
    x = self.embedding(idx) + self.pos_embedding(torch.arange(seq_len,device=device))
    x = self.ln1(x + self.ah(x,x,pad_mask))
    x = self.ln2(x + self.ffwd(x))
    logits = self.lo(x.sum(dim = 1))
    if targets is None:
      loss =  None
    else:
      cross__entropy = nn.CrossEntropyLoss(weight=torch.tensor([0.6,2.97],device=device,))
      loss = cross__entropy(logits,targets)
    return {'logits': logits, 'loss':loss}


model = Anomaly(vs=tokenizer.vocab_size,d=192,num_heads=6,max_len=MAX_LEN)

# Training loop
seed_val = 1024
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)

train = df_gl_train.reset_index(drop=True)
val = df_gl_val.reset_index(drop=True)

train_dataset = TrainDataset(train)
val_dataset = TrainDataset(val)

epoch_val_loss=[]
epoch_train_loss=[]
epochs_without_improvement = 0
epoch=0
while epochs_without_improvement<20 and epoch<NUM_EPOCHS:
    print('======== Epoch {:} / {:} ========'.format(epoch+1, NUM_EPOCHS))

    if epoch==0:

        model.to(device)
        optimizer = AdamW(model.parameters(),
                         lr=L_RATE,
                         eps=1e-8)
        scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2)
    else:
        model.to(device)

    train_dataloader = DataLoader(train_dataset,batch_size=BATCH_SIZE,
                                               shuffle=True,num_workers=NUM_CORES)
    val_dataloader = DataLoader(val_dataset,batch_size=BATCH_SIZE,
                                               shuffle=True,num_workers=NUM_CORES)
    print('Training...')
    model.train()
    torch.set_grad_enabled(True)
    total_train_loss = []

    for i, batch in enumerate(train_dataloader):
        if i%100==0:print(f'Batch: {i}')
        input_ids = batch[0].to(device)
        attention_mask = batch[1].to(device)
        targets = batch[2].to(device)

        outputs = model(
            idx=input_ids,
            pad_mask=attention_mask,
            targets=targets
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

        input_ids = val_batch[0].to(device)
        attention_mask = val_batch[1].to(device)
        targets = val_batch[2].to(device)

        outputs = model(
            idx=input_ids,
            pad_mask=attention_mask,
            targets=targets
        )
        loss = outputs['loss'].detach().cpu().numpy()
        total_val_loss.append(loss.item())
        gc.collect()
        if j%160==0: break
    current_train_loss = sum(total_train_loss)/len(total_train_loss)
    current_val_loss = sum(total_val_loss)/len(total_val_loss)
    print('train loss',current_train_loss)
    print('val_loss',current_val_loss)
    epoch_train_loss.append(current_train_loss)
    if current_val_loss < min(epoch_val_loss,default=0):
      model_name = 'Best_model_weights.pth'
      torch.save(model.state_dict(), model_name)
    epoch_val_loss.append(current_val_loss)
    model_name = 'model_weights.pth'
    torch.save(model.state_dict(), model_name)
    epoch+=1  

#Test

df_gl_test = pd.read_csv(path)
df_gl_test = df_gl_test.reset_index(drop=True)
test_loader = DataLoader(TrainDataset(df_gl_test),batch_size=BATCH_SIZE,
                                          shuffle=True,num_workers=NUM_CORES)
test_out = {} # to be able to save the pres at the end
for i, batch in enumerate(test_loader):
  if i%200==0:print(f'Batch: {i}')
  input_ids = batch[0].to(device)
  attention_mask = batch[1].to(device)
  targets = batch[2]
  outputs = model(
      idx=input_ids,
      pad_mask=attention_mask,)
  if i==0:
    test_out['preds'] = outputs['logits'].detach().cpu()
    test_out['labels'] =  targets
  else:
    test_out['preds'] = torch.cat((test_out['preds'],outputs['logits'].detach().cpu()), dim=0)
    test_out['labels'] = torch.cat((test_out['labels'],targets), dim=0)

preds = torch.argmax(test_out['preds'], dim=1).numpy() # getting the predicted class

# Confusion matrix
cm = confusion_matrix(test_out['labels'], preds)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #For normalization
labels = np.array(['Normal','Attack'])
cm_display = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labels)
cm_display.plot(values_format=".4f",cmap='viridis')

plt.show()

import torch
import pandas as pd
import torch
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from Tokenizer import RLLMTokenizer

MODEL_NAME = 'bert-base-uncased'
MAX_LEN = 5
MAX_LEN_E = 2000
BATCH_SIZE = 32

class DatasetTS(Dataset,AutoTokenizer,RLLMTokenizer):
  def __init__(self,df,MODEL_NAME,SPACE=100):
    self.df = df
    self.x_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    self.tokenizer = RLLMTokenizer(SPACE)

  def __getitem__(self,i):

    df = self.df.iloc[i]
    enc = self.x_tokenizer.encode_plus(df.text,return_tensors='pt',padding='max_length', max_length=MAX_LEN_E)
    ids = enc['input_ids'].squeeze(0)
    pad_mask = enc['attention_mask'].squeeze(0)
    encoded_y = self.tokenizer.encode_plus(df.valence,df.arousal)
    input_y = encoded_y[:-1]
    target = encoded_y[1:]
    user_id = df['user_id']
    mask_t = torch.zeros((BATCH_SIZE,))
    return [ids, pad_mask, input_y, target, mask_t, int(user_id) ]

  def __len__(self):
    return len(self.df)
# custome collate function to be able to create attention mask for inter user essays
def dynamic_collate_fn(batch):
  pad_mask_t =[]
  batch_size = len(batch)
  users = list(set([t[5]  for t in batch]))
  b = [[] for _ in users]
  for t in batch:
    index  = users.index(t[5])
    b[index].append(t[:5])
  prev_user_seq = 0
  for u in b:
    user_seq = len(u)
    for i, inp in enumerate(u):
      u[i][4] = torch.cat((torch.zeros((prev_user_seq,),dtype=torch.long),torch.ones((user_seq ,),dtype=torch.long),torch.zeros((batch_size-user_seq-prev_user_seq,),dtype=torch.long)))
    prev_user_seq += user_seq
  batch = []
  for t in b:
    batch += t
  return torch.stack([t[0] for t in batch]),torch.stack([t[1] for t in batch]),torch.stack([t[2] for t in batch]),torch.stack([t[3] for t in batch]),torch.stack([t[4] for t in batch])

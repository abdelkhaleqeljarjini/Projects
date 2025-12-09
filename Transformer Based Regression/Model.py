import os
import torch
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MaskedAttentionHead(nn.Module):
  def __init__(self, d, hs,max_len=6):
    super().__init__()
    self.hs = hs
    self.query = nn.Linear(d, hs, bias=False)
    self.key = nn.Linear(d, hs, bias=False)
    self.value = nn.Linear(d, hs, bias=False)
    self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len)))

  def forward(self, x):
    batch,seq_len,emb_size = x.shape
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    att = q@k.transpose(-2,-1)
    att = att*self.hs**-0.5
    att = att.masked_fill(self.tril[:seq_len,:seq_len]==0,float('-inf'))
    att = F.softmax(att, dim=-1)@v
    return att

class CrossAttentionHead(nn.Module):
  def __init__(self, d, hs):
    super().__init__()
    self.hs = hs
    self.query = nn.Linear(d, hs, bias=False)
    self.key = nn.Linear(d, hs, bias=False)
    self.value = nn.Linear(d, hs, bias=False)

  def forward(self, x,y):
    q = self.query(y)
    k = self.key(x)
    v = self.value(x)
    att = q@k.transpose(-2,-1)
    att = att*self.hs**-0.5
    att = F.softmax(att, dim=-1)@v
    return att

class SelfAttentionHead(nn.Module):
  def __init__(self, d, hs):
    super().__init__()
    self.hs = hs
    self.query = nn.Linear(d, hs, bias=False)
    self.key = nn.Linear(d, hs, bias=False)
    self.value = nn.Linear(d, hs, bias=False)

  def forward(self, x, attention_mask):
    q = self.query(x)
    k = self.key(x)
    v = self.value(x)
    att = q@k.transpose(-2,-1)
    att = att*self.hs**-0.5
    att = att.masked_fill(attention_mask[:, None, :]==0,float('-inf'))
    att = F.softmax(att, dim=-1)@v
    return att
  
class FeedForward(nn.Module):

  def __init__(self,d):
    super().__init__()
    self.ffw = nn.Sequential(
        nn.Linear(d,d),
        nn.GELU(),
        nn.Linear(d,d),
        nn.GELU())

  def forward(self,x):
    return self.ffw(x)

class EncoderRLLM(nn.Module):

  def __init__(self, vs, d,max_len=2000):
    super().__init__()
    self.max_len = max_len
    self.embedding = nn.Embedding(vs,d)
    self.pos_embedding = nn.Embedding(max_len,d) # Learned positional embeddings.
    self.ah = SelfAttentionHead(d,d)
    self.ln1 =nn.LayerNorm(d)
    self.ln2 =nn.LayerNorm(d)
    self.ffwd = FeedForward(d)


  def forward(self, idx, attention_mask):
    batch,seq_len = idx.shape
    x = self.embedding(idx) + self.pos_embedding(torch.arange(seq_len,device= device))
    x = self.ln1(self.ah(x,attention_mask))
    x = self.ln2(self.ffwd(x))
    return x
  
class DecoderRLLM(nn.Module):

  def __init__(self, vse, d, vs=212, max_len=5,max_len_enc=2000):
    super().__init__()
    self.max_len = max_len
    self.encoder = EncoderRLLM(vse,d,max_len_enc)
    self.cross_ah = CrossAttentionHead(d,d)
    self.embedding = nn.Embedding(vs,d)
    self.pos_embedding = nn.Embedding(max_len,d) # Learned positional embeddings.
    self.ah = MaskedAttentionHead(d,d)
    self.ln1 = nn.LayerNorm(d)
    self.ln2 = nn.LayerNorm(d)
    self.ln3 = nn.LayerNorm(d)
    self.ffwd = FeedForward(d)
    self.linear_head = nn.Linear(d,vs)

  def forward(self, idx=None,idx_e=None, targets=None,attention_mask=None):
    batch,seq_len = idx.shape
    encoder_out = self.encoder(idx_e,attention_mask)
    x = self.embedding(idx) + self.pos_embedding(torch.arange(seq_len,device= device))
    x = self.ln1(self.ah(x))
    x = self.ln2(self.cross_ah(encoder_out,x))
    x = self.ln3(self.ffwd(x))
    logits = self.linear_head(x) # ( batch, vocab size )
    b,seq,cl=logits.shape
    if targets is None:
      loss = None
    else:
      loss = F.cross_entropy(logits.view(b*seq,cl),targets.view(b*seq))
    return {'logits': logits, 'loss':loss}

  def generate(self, idx_e, idx,attention_mask=None, num_out_tokens=5):
    for _ in range(num_out_tokens):
      # if idx is None:
      #   idx = torch.full((1,1),455).to(device)
      idx_trunc = idx[:,-self.max_len:]
      out = self(idx=idx,idx_e=idx_e, attention_mask=attention_mask)
      probs = F.softmax(out['logits'][:,-1,:], dim=-1)
      id_next = torch.multinomial(probs,num_samples=1)
      idx = torch.cat((idx,id_next),dim=1)

    return idx

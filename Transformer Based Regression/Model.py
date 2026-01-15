import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#Multihead attention class
class MultiHA(nn.Module):
  def __init__(self, d, hs,dropout,max_len=None,causal_attention = False):
    super().__init__()
    self.causal_attention = causal_attention
    self.hs = hs
    self.d = d
    self.num_heads = d//hs # % of division should be zero
    self.query = nn.Linear(d, d, bias=False)
    self.key = nn.Linear(d, d, bias=False)
    self.value = nn.Linear(d, d, bias=False)
    self.dropout = nn.Dropout(dropout)
    self.proj = nn.Linear(d,d)
    if causal_attention:
      self.register_buffer('tril',torch.tril(torch.ones(max_len,max_len)))

  def forward(self, x,y,pad_mask =None, time_attention=None):
    batch,seq_len,emb_size = x.shape
    q = self.resize(self.query(y), batch, y.shape[1], self.num_heads, self.hs)
    k = self.resize(self.key(x), batch, seq_len, self.num_heads, self.hs)
    v = self.resize(self.value(x), batch, seq_len, self.num_heads, self.hs)
    att = q@k.transpose(-2,-1)
    att = att*self.hs**-0.5
    if self.causal_attention:
      att = att.masked_fill(self.tril[:seq_len,:seq_len]==0,float('-inf'))
    if pad_mask is not None:
      att = att.masked_fill(pad_mask[:, None, None, :]==0,float('-inf'))
    if time_attention is not None:
      att = att.masked_fill(time_attention[None, None, :, :]==0,float('-inf'))
    att = self.dropout(F.softmax(att, dim=-1))@v
    att = att.transpose(1,2).reshape(batch,y.shape[1],self.d)
    return self.dropout(self.proj(att))

  @staticmethod
  def resize(x, batch, seq_len, num_heads, hs):
    return x.view(batch,seq_len, num_heads, hs ).transpose(1,2)
  
class FeedForward(nn.Module):

  def __init__(self,d,DROPOUT):
    super().__init__()
    self.ffw = nn.Sequential(
        nn.Linear(d,2*d),
        nn.GELU(),
        nn.Linear(2*d,d),
        nn.Dropout(DROPOUT)
        )

  def forward(self,x):
    return self.ffw(x)

#Encoder for the input text
class EncoderRLLM(nn.Module):

  def __init__(self, d,num_heads,dropout):
    super().__init__()
    self.self_ah = MultiHA(d=d,hs=d//num_heads,dropout=dropout)
    self.ln1 =nn.LayerNorm(d)
    self.ln2 =nn.LayerNorm(d)
    self.ffwd = FeedForward(d,dropout)

  def forward(self, t_emb, attention_mask):
    t_emb = self.ln1(t_emb + self.self_ah(t_emb,t_emb,attention_mask))
    t_emb = self.ln2(t_emb + self.ffwd(t_emb))
    return t_emb
  
#Transformer block
class Block(nn.Module):
  def __init__(self,d,num_heads,dropout):
    super().__init__()
    self.encoder = EncoderRLLM(d=d,num_heads=num_heads,dropout=dropout)
    self.time_ah = MultiHA(d=d,hs=d//num_heads,causal_attention=True,max_len=BATCH_SIZE,dropout=dropout)
    self.cross_ah = MultiHA(d=d,hs=d//num_heads,dropout=dropout)
    self.self_ah = MultiHA(d=d,hs=d//num_heads, causal_attention=True,max_len= MAX_LEN,dropout=dropout)
    self.ln1 = nn.LayerNorm(d)
    self.ln2 = nn.LayerNorm(d)
    self.ln3 = nn.LayerNorm(d)
    self.ln4 = nn.LayerNorm(d)
    self.ffwd = FeedForward(d,dropout)

  def forward(self,args):
    t_emb = args['t_emb']
    y = args['y']
    pad_mask = args['pad_mask'] #padding mask for the text input
    pad_mask_ts = args['pad_mask_ts'] #mask for theattention between essays 
    encoder = self.encoder(t_emb,pad_mask)
    B,_,d = encoder.shape
    x = encoder.mean(1).view(1,B,d)
    x = self.ln1(x + self.time_ah(x,x,time_attention=pad_mask_ts))
    y = self.ln2(y + self.self_ah(y,y))
    y = self.ln3(y + self.cross_ah(x.view(B,1,d),y))
    y = self.ln4(y + self.ffwd(y))
    return {'t_emb':encoder,'y': y,'pad_mask_ts':pad_mask_ts,'pad_mask':pad_mask}

#Autoregressive Generative Regression Model
class RegressionLLM(nn.Module):

  def __init__(self, vse, d, vs, max_len,max_len_enc,num_heads,dropout,n_blocks):
    super().__init__()
    self.max_len = max_len
    self.embedding_dec = nn.Embedding(vs,d)
    self.pos_embedding_dec = nn.Embedding(max_len,d) # Learned positional embeddings.
    self.embedding_enc = nn.Embedding(vse,d)
    self.pos_embedding_enc = nn.Embedding(max_len_enc,d) # Learned positional embeddings.
    self.blocks = nn.Sequential(*[Block(d,num_heads,dropout) for _ in range(n_blocks)])
    self.out_head = nn.Linear(d,vs)

  def forward(self, idx=None,idxe=None, targets=None,attention_mask=None, pad_mask_ts=None):
    batch,seq_len_dec = idx.shape
    batch,seq_len_enc = idxe.shape
    y = self.embedding_dec(idx) + self.pos_embedding_dec(torch.arange(seq_len_dec,device= device))
    t_emb = self.embedding_enc(idxe) + self.pos_embedding_enc(torch.arange(seq_len_enc,device= device))
    y = self.blocks({'t_emb':t_emb,'y': y,'pad_mask_ts':pad_mask_ts,'pad_mask':attention_mask})['y']
    logits = self.out_head(y) # ( batch, seq, vocab size )
    b,seq,vs=logits.shape
    if targets is None:
      loss = None
    else:
      loss = F.cross_entropy(logits.view(b*seq,vs),targets.view(b*seq))
    return {'logits': logits, 'loss':loss}

  def generate(self, idx, idxe,attention_mask=None, pad_mask_ts=None, num_out_tokens=11):
    
    for _ in range(num_out_tokens):
      batch,seq_len_dec = idx.shape
      idx_trunc = idx[:,-self.max_len:]
      out = self(idx=idx,idxe=idxe, attention_mask=attention_mask,pad_mask_ts=pad_mask_ts)
      probs = F.softmax(out['logits'][:,-1,:], dim=-1)
      id_next = torch.multinomial(probs,num_samples=1)
      idx = torch.cat((idx,id_next),dim=1)
      
    return idx


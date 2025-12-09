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

x_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer = RLLMTokenizer(100)

class TrainDataset(Dataset):

    def __init__(self,dataframe):
        self.df = dataframe

    def __getitem__(self,index):

        # batch = []
        # for index in idx:
        text = self.df.loc[index,'x']
        #tokenizing the data
        encoded_dict = x_tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length = MAX_LEN_E,
            return_attention_mask=True,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        #Extracting the model inputs, already tensors
        padded_tokens_list = encoded_dict['input_ids'].squeeze()
        att_mask = encoded_dict['attention_mask'].squeeze()
        token_type_ids = encoded_dict['token_type_ids'].squeeze()
        #transforming target label to tensors
        # target = self.df.loc[index,'valence']
        encoded_y = tokenizer.encode(self.df.loc[index,'valence'])
        # target = torch.tensor(self.df.loc[index,'valence'],dtype=torch.float32)
        input_y = encoded_y[:-1]
        target = encoded_y[1:]
        sample = (padded_tokens_list,att_mask,token_type_ids,input_y,target)
        return sample

    def __len__(self):
        return len(self.df)
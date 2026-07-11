import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch
import json
from torch.optim import Adam
import numpy as np


## Read dictionaries
with open('data/snips_processed/word2idx.json','r') as f:
    word2idx = json.load(f)

with open('data/snips_processed/idx2intent.json','r') as f:
    idx2intent = json.load(f)

with open('data/snips_processed/slot2idx.json','r') as f:
    slot2idx = json.load(f)
    idx2slot = {val:key for key,val in json.load(f).items()}


idx2word = {val:key for key,val in word2idx.items()}

### Read train data

with open('data/snips_processed/train.jsonl','r') as f:
    train_data = [json.loads(line) for line in f]



class SnipsDataset(Dataset):
    def __init__(self,data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self,idx):
        ex = self.data[idx]
        token_ids = torch.tensor(ex['token_ids'],dtype=torch.long)
        slot_ids = torch.tensor(ex['slot_ids'],dtype=torch.long)
        intent_id = torch.tensor(ex['intent_id'],dtype=torch.long)
        return token_ids,slot_ids,intent_id
        
def collate_fn(batch):
    token_ids = [value[0] for value in batch]
    slot_ids = [value[1] for value in batch]

    padded_token = pad_sequence(token_ids,batch_first=True,padding_value = word2idx['PAD'])
    padded_slot = pad_sequence(slot_ids,batch_first=True,padding_value=slot2idx['PAD'])

    return {'token_ids' : padded_token ,
            'slot_ids' : padded_slot ,
            'intent_ids': torch.stack([value[2] for value in batch])}





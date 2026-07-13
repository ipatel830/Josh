import json
import torch
from torchcrf import CRF
import torch.nn as nn
from typing import Tuple, Dict, Optional, List



def _load_dictionary(path : str)-> tuple:
    with open(path,'r') as f:
        name = json.load(f)
        name_inverse = {val:key for key,val in name.items()}
    return name, name_inverse


## NLU MODEL ARCHITECTURE

class JointIntentSlotModel(nn.Module):
    def __init__(self, vocab_size, num_slots, num_intents, emb_dim=100, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.slot_head = nn.Linear(hidden_dim * 2, num_slots)
        self.intent_head = nn.Linear(hidden_dim * 2, num_intents)
        self.crf = CRF(num_slots,batch_first=True)
    def forward(self, token_ids):
        embedded = self.embedding(token_ids)              # [batch, seq_len, emb_dim]
        lstm_out, (h_n, c_n) = self.lstm(embedded)        # lstm_out: [batch, seq_len, hidden_dim*2]
        slot_logits = self.slot_head(lstm_out)             # [batch, seq_len, num_slots]

        # for intent: concat final forward + backward hidden states
        intent_input = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden_dim*2]
        intent_logits = self.intent_head(intent_input)       # [batch, num_intents]

        return slot_logits, intent_logits
    def slot_loss(self,slot_logits,slot_ids,mask):
        return -self.crf(slot_logits,slot_ids,mask=mask,reduction='mean')
    
    def decode_slots(self,slot_logits,mask):
        return self.crf.decode(slot_logits,mask=mask)



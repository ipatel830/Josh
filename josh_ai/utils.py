import json
import torch
from torch.utils.data import Dataset
from torchcrf import CRF
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torchaudio.transforms as T
from typing import Tuple, Dict, Optional, List



def _load_dictionary(path : str)-> tuple:
    with open(path,'r') as f:
        name = json.load(f)
        name_inverse = {val:key for key,val in name.items()}
    return name, name_inverse

def _load_data(path) -> List:
    with open(path,'r') as f:
        data = [json.loads(line) for line in f]
    return data



## SPEECH TO MEL-SPECTROGRAM
def STM(audio,sample_rate):
    mel_transform = nn.Sequential(
        T.MelSpectrogram(sample_rate=sample_rate,  
        n_fft=400,
        hop_length=160,
        n_mels=80,
        normalized=True,
        norm='slaney'),
        T.AmplitudeToDB()
    )
    return mel_transform(audio)


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



class SpecAugment(nn.Module):
    def __init__(
        self,
        num_freq_masks: int = 2,
        freq_mask_param: int = 27,
        num_time_masks: int = 2,
        time_mask_param: int = 100,
        time_mask_ratio: float = 0.05,
    ):
        super().__init__()
        self.num_freq_masks  = num_freq_masks
        self.num_time_masks  = num_time_masks
        self.time_mask_ratio = time_mask_ratio
 
        self.freq_maskers = nn.ModuleList([
            T.FrequencyMasking(freq_mask_param=freq_mask_param)
            for _ in range(num_freq_masks)
        ])
        self.time_maskers = nn.ModuleList([
            T.TimeMasking(time_mask_param=time_mask_param,
                          p=time_mask_ratio)
            for _ in range(num_time_masks)
        ])
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if not self.training:
            return x
 
        x = x.permute(0, 2, 1)  
 
        for masker in self.freq_maskers:
            x = masker(x)
 
        for masker in self.time_maskers:
            x = masker(x)
 
        x = x.permute(0, 2, 1)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self,d_model,max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len,d_model)
        position = torch.arange(0,max_len).unsqueeze(1).float()
        div_term = torch.pow(10000.0,torch.arange(0,d_model,2).float() / d_model)

        pe[:, 0::2] = torch.sin(position/div_term)
        pe[:, 1::2] = torch.cos(position/div_term)

        self.register_buffer('pe',pe.unsqueeze(0))

    def forward(self,x):
        return x + self.pe[:, :x.size(1), :]



## STT MODEL ARCHITECTURE
class S2T(nn.Module):
    def __init__(self,n_mels=80,vocab_size=31):
        super().__init__()
        self.pos_encoding = PositionalEncoding(d_model=512)
        self.spec_aug = SpecAugment()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_mels, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.1),
            nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1), 
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.1),
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=512,
            nhead=8,
            dim_feedforward=2048,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=4)
        self.fc = nn.Linear(512,vocab_size)
    def forward(self,x):
        x = x.float()
        x = self.spec_aug(x)
        x = x.permute(0,2,1)
        x = self.cnn(x)
        x = x.permute(0,2,1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.fc(x)
        return x
    


def ctc_collapse(token_ids: list[int], blank_id: int) -> list[int]:

    result = []
    prev   = None
    for tid in token_ids:
        if tid == blank_id:
            prev = None   
            continue
        if tid != prev:
            result.append(tid)
        prev = tid
    return result
 
 
def ctc_greedy_decode(logits: torch.Tensor, blank_id: int) -> list[int]:

    ids = torch.argmax(logits, dim=-1).tolist()
    return ctc_collapse(ids, blank_id)
 
 

def ids_to_text(token_ids: list[int], tokenizer) -> str:

    text = tokenizer.decode(token_ids)
    return text.replace("|", " ").strip().lower()
 
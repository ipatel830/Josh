import torchaudio.transforms as T
import torchaudio as taudio
import torch.nn as nn
from torch.utils.data import Dataset
import torch
import os
from torch.nn.utils.rnn import pad_sequence
import boto3


class process_data(Dataset):
    def __init__(self,root_folder,train=None):
        self.root_folder = root_folder
        self.train = train
        self.length_data = 0
        self.no_audio = 0
        if train: self.saved_dir_name = 'processed_train'
        else: self.saved_dir_name = 'processed_test'
        os.makedirs(self.saved_dir_name, exist_ok=True) ### make directory if it doesnt exist##
        self.preprocess()

    def preprocess(self):
        samplerate = 16000
        mel_transform = nn.Sequential(
            T.MelSpectrogram(sample_rate=samplerate,  
            n_fft=400,
            hop_length=160,
            n_mels=80,
            normalized=True,
            norm='slaney'),
            T.AmplitudeToDB()
        )

        for dirpath, dirnames, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.endswith('.flac'):
                    full_path = os.path.join(dirpath, filename)
                    idx = filename.split('.')[0]
                    audio, _ = taudio.load(uri=full_path)
                    mel_spec = mel_transform(audio)
                    mel_spec = mel_spec.squeeze(0)
                    torch.save({'audio':mel_spec},f'{self.saved_dir_name}/{idx}.pt')
                    del audio, mel_spec
                    self.length_data+=1

        for dirpath, dirnames, filenames in os.walk(self.root_folder):
            for filename in filenames:
                if filename.endswith('.trans.txt'):
                    full_path = os.path.join(dirpath, filename)
                    with open(full_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            parts = line.split(' ')
                            idx = parts[0]
                            text = ' '.join(parts[1:])

                            #check .pt file exists with mel spectogram
                            pt_path = f"{self.saved_dir_name}/{idx}.pt"
                            if os.path.exists(pt_path):
                                sample = torch.load(pt_path,weights_only=True)
                                sample['text'] = text
                                torch.save(sample,pt_path)
                                del sample
                            else:
                                print(f"Warning: no audio found for {idx}")
                                self.no_audio +=1


    def tokenize(self,tokenizer):
        files = [f for f in os.listdir(self.saved_dir_name) if f.endswith('.pt')]
        for fname in files:
            pt_path = f'{self.saved_dir_name}/{fname}'
            sample = torch.load(pt_path,weights_only=True)
            if 'text' in sample:
                text = sample['text'].replace(' ', '|')
                label_ids = tokenizer(text).input_ids
                sample['labels'] = torch.tensor(label_ids, dtype=torch.long)
                torch.save(sample, pt_path)
                del sample
                
        return self.length_data,self.no_audio


class LibriSpeechDataset(Dataset):
    def __init__(self, processed_dir):
        self.files = [
            os.path.join(processed_dir, f)
            for f in os.listdir(processed_dir)
            if f.endswith('.pt')
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data   = torch.load(self.files[idx], weights_only=True)
        sample = data['audio']
        data['audio'] = (sample - sample.mean()) / (sample.std() + 1e-8)
        return data

def collate_fn(batch):
    audios = [item["audio"].T for item in batch] 
    labels = [item["labels"] for item in batch]

    audios_padded = pad_sequence(audios, batch_first=True) 
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=0) 

    input_lengths  = torch.tensor([a.shape[0] for a in audios])
    target_lengths = torch.tensor([l.shape[0] for l in labels])

    return {
        "audio":          audios_padded,
        "labels":         labels_padded,
        "input_lengths":  input_lengths,
        "target_lengths": target_lengths
    }

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

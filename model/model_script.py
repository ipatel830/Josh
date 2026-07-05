

# import dependencies after running environment.yml script #######
import os
import sys
import logging
import subprocess
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2CTCTokenizer 
from torchaudio.models.decoder import ctc_decoder
from jiwer import wer,cer
import boto3

## append directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR,'../data'))
sys.path.append(os.path.join(BASE_DIR,'../model'))


from dataset import collate_fn,LibriSpeechDataset,PositionalEncoding,SpecAugment
from evaluation import evaluate_batch




logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s %(message)s',
                    handlers = [
                        logging.FileHandler(os.path.join(BASE_DIR,'training.log')),
                        logging.StreamHandler()
                        ]
                    )
log = logging.getLogger(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
log.info(f"Using device: {device}")


def verify_checkpoint(path):
    """Verify checkpoint file is not corrupted."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=True)
        required = ['epoch', 'model_state', 'optimizer_state',
                    'scheduler_state', 'best_wer', 'loss_arr', 'wer_arr', 'cer_arr']
        for key in required:
            assert key in ckpt, f"Missing key: {key}"
        log.info(f"Checkpoint verified — epoch {ckpt['epoch']}, best WER {ckpt['best_wer']:.4f}")
        return True
    except Exception as e:
        log.error(f"Checkpoint verification failed: {e}")
        return False

data_dir = os.path.join(BASE_DIR,'../data')
lm_path = os.path.join(BASE_DIR,'lm','4-gram.arpa')


tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(os.path.join(data_dir,'wav2vec2_tokenizer'))

full_dataset = LibriSpeechDataset(os.path.join(data_dir, 'processed_train/'))

train_size = int(0.9 * len(full_dataset))
val_size   = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(
    full_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)

train_dataloader = DataLoader(
    train_dataset, 
    batch_size=80, 
    shuffle=True, 
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)

val_dataloader = DataLoader(
    val_dataset, 
    batch_size=32, 
    shuffle=False,  
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
)


batch = next(iter(train_dataloader))
log.info(f"Audio:          {batch['audio'].shape}")           # (B, T, n_mels)
log.info(f"Labels:         {batch['labels'].shape}")          # (B, L)
log.info(f"Input lengths:  {batch['input_lengths']}")         # real audio lengths
log.info(f"Target lengths: {batch['target_lengths']}")        # real label lengths


def get_conv_output_length(length, layers):
    for kernel, stride, padding in layers:
        length = (length + 2 * padding - kernel) // stride + 1
    return length


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


vocab_size = tokenizer.vocab_size
vocab = [None] * len(tokenizer)
for token, idx in tokenizer.get_vocab().items():
    vocab[idx] = token
vocab[tokenizer.pad_token_id] = "-"


#### Use a beam decoder + 4-gram model for better path prediction ########
decoder = ctc_decoder(lexicon=None,
                      tokens=vocab,
                      lm=lm_path,
                      beam_size=50,
                      blank_token="-",
                      sil_token="|")

###### Initialize model with parameters and send to device and .compile #######

device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = S2T(n_mels=80,vocab_size=vocab_size).to(device)

blank_id = tokenizer.pad_token_id
ctc_loss = nn.CTCLoss(blank=blank_id,zero_infinity=True)
optimizer = torch.optim.Adam(params=model.parameters(),lr=3e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3)

# resume in case of instance failure
if os.path.exists('checkpoint.pt'):
    ckpt = torch.load('checkpoint.pt', map_location=device,weights_only=True)
    model.load_state_dict(ckpt['model_state'])
    optimizer.load_state_dict(ckpt['optimizer_state'])
    scheduler.load_state_dict(ckpt['scheduler_state'])
    start_epoch = ckpt['epoch'] + 1
    best_wer    = ckpt['best_wer']
    loss_arr    = ckpt['loss_arr']
    wer_arr     = ckpt['wer_arr']
    cer_arr     = ckpt['cer_arr']
    log.info(f"Resumed from epoch {start_epoch}")
else:
    start_epoch = 0
    loss_arr  = []
    wer_arr, cer_arr = [], []
    best_wer  = float('inf')

model = torch.compile(model)

nepochs   = 300
conv_params = [(3, 2, 1), (3, 2, 1), (3, 2, 1)]


for epoch in range(start_epoch,nepochs):

    ########TRAIN###########
    model.train()
    all_predictions   = []
    all_ground_truths = []
    batch_loss        = 0

    for batch in train_dataloader:
        optimizer.zero_grad()

        x      = batch['audio'].float().to(device)
        logits = model(x)

        log_probs     = torch.nn.functional.log_softmax(logits, dim=-1)
        log_probs_ctc = log_probs.permute(1, 0, 2)              # [T, B, vocab]

        input_lengths = torch.tensor([
            get_conv_output_length(l.item(), conv_params)
            for l in batch['input_lengths']
        ]).to(device)

        loss = ctc_loss(
            log_probs_ctc,
            batch['labels'].to(device),
            input_lengths,
            batch['target_lengths'].to(device),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        batch_loss += loss.item()

        preds, gts = evaluate_batch(logits, batch['labels'], tokenizer)
        all_predictions.extend(preds)
        all_ground_truths.extend(gts)

    if all_predictions:
        log.info(f"Sample pred: {all_predictions[0][:80]}")
        log.info(f"Sample true: {all_ground_truths[0][:80]}")

        #### save following values in case of EC2 instance failures

    torch.save({
        'epoch':           epoch,
        'model_state':     model.state_dict(),
        'optimizer_state': optimizer.state_dict(),
        'scheduler_state': scheduler.state_dict(),
        'best_wer':        best_wer,
        'loss_arr':        loss_arr,
        'wer_arr':         wer_arr,
        'cer_arr':         cer_arr,
    }, 'checkpoint.pt')

    avg_loss = batch_loss / len(train_dataloader)
    paired   = [(p, g) for p, g in zip(all_predictions, all_ground_truths) if g.strip()]
    wer_val  = wer([g for _, g in paired], [p for p, _ in paired])
    cer_val  = cer([g for _, g in paired], [p for p, _ in paired])

    loss_arr.append(avg_loss)
    wer_arr.append(wer_val)
    cer_arr.append(cer_val)
    scheduler.step(avg_loss)


    ####### VALIDATION every 5 epochs ###########
    if (epoch + 1) % 5 == 0:
        model.eval()
        beam_preds, beam_gts = [], []

        with torch.no_grad():
            for batch in val_dataloader:
                x      = batch['audio'].float().to(device)   
                logits = model(x)
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)  
                results   = decoder(log_probs.cpu())

                for hyps in results:
                    tokens = decoder.idxs_to_tokens(hyps[0].tokens)
                    text = "".join(tokens).replace("|", " ").replace("-", "").strip().lower()
                    beam_preds.append(text)

                _, gts = evaluate_batch(logits, batch['labels'], tokenizer)
                beam_gts.extend(gts)

        paired_beam = [(p, g) for p, g in zip(beam_preds, beam_gts) if g.strip()]
        beam_wer    = wer([g for _, g in paired_beam], [p for p, _ in paired_beam])
        log.info(f"  Beam WER (val): {beam_wer:.4f}")

        if beam_wer < best_wer:                              
            best_wer = beam_wer
            torch.save(model.state_dict(), "best_model.pt")
            log.info("  -> Model saved")

        model.train()

        with open('metrics.json','w') as f:
            json.dump({
                'loss':loss_arr,
                'wer' : wer_arr,
                'cer' : cer_arr,
            }, f)

    log.info("*" * 41)
    log.info(f"Epoch:     {epoch+1}/{nepochs}")
    log.info(f"Loss:      {avg_loss:.4f}")
    log.info(f"WER:       {wer_val:.4f}")
    log.info(f"CER:       {cer_val:.4f}")
    log.info(f"Best WER:  {best_wer:.4f}")
    log.info("_" * 41)


log.info("Training complete - everything is saved...")



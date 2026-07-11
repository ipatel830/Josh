import json
import time
import numpy as np
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam


## Read dictionaries
with open('data/snips_processed/word2idx.json','r') as f:
    word2idx = json.load(f)

with open('data/snips_processed/intent2idx.json','r') as f:
    intent2idx = json.load(f)
    idx2intent = {val:key for key,val in intent2idx.items()}

with open('data/snips_processed/slot2idx.json','r') as f:
    slot2idx = json.load(f)
    idx2slot = {val:key for key,val in json.load(f).items()}


idx2word = {val:key for key,val in word2idx.items()}

### Read train,test,validate data

with open('data/snips_processed/train.jsonl','r') as f:
    train_data = [json.loads(line) for line in f]

with open('data/snips_processed/test.jsonl','r') as f:
    test_data = [json.loads(line) for line in f]

with open('data/snips_processed/valid.jsonl','r') as f:
    valid_data = [json.loads(line) for line in f]



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



## Create datasets class
train_dataset = SnipsDataset(train_data)
valid_dataset = SnipsDataset(valid_data)
test_dataset = SnipsDataset(test_data)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)




## bidirectional LSTM with embedding
class JointIntentSlotModel(nn.Module):
    def __init__(self, vocab_size, num_slots, num_intents, emb_dim=100, hidden_dim=128, pad_idx=0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.slot_head = nn.Linear(hidden_dim * 2, num_slots)
        self.intent_head = nn.Linear(hidden_dim * 2, num_intents)

    def forward(self, token_ids):
        embedded = self.embedding(token_ids)              # [batch, seq_len, emb_dim]
        lstm_out, (h_n, c_n) = self.lstm(embedded)         # lstm_out: [batch, seq_len, hidden_dim*2]

        slot_logits = self.slot_head(lstm_out)             # [batch, seq_len, num_slots]

        # for intent: concat final forward + backward hidden states
        intent_input = torch.cat((h_n[-2], h_n[-1]), dim=1)  # [batch, hidden_dim*2]
        intent_logits = self.intent_head(intent_input)       # [batch, num_intents]

        return slot_logits, intent_logits
    

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = JointIntentSlotModel(vocab_size=len(word2idx), num_slots=len(slot2idx), num_intents=len(intent2idx), pad_idx=word2idx['PAD']).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

intent_loss_fn = nn.CrossEntropyLoss()
slot_loss_fn = nn.CrossEntropyLoss(ignore_index=slot2idx['PAD'])


num_epochs = 30
slot_loss_weight = 1.0
intent_loss_weight = 1.0
pad_idx = slot2idx['PAD']

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    epoch_start = time.time()

    for batch in train_loader:
        token_ids = batch['token_ids'].to(device)
        slot_ids = batch['slot_ids'].to(device)
        intent_ids = batch['intent_ids'].to(device)

        optimizer.zero_grad()

        slot_logits, intent_logits = model(token_ids)

        slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1)), slot_ids.view(-1))
        intent_loss = intent_loss_fn(intent_logits, intent_ids)

        loss = slot_loss_weight * slot_loss + intent_loss_weight * intent_loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    print(f'[Epoch {epoch}/{num_epochs}] train_loss={avg_train_loss:.4f} time={epoch_time:.1f}s', flush=True)

    if epoch % 5 == 0:
        model.eval()
        total_val_loss = 0.0
        correct_intents = 0
        total_examples = 0

        all_slot_preds = []
        all_slot_labels = []

        with torch.no_grad():
            for batch in valid_loader:
                token_ids = batch['token_ids'].to(device)
                slot_ids = batch['slot_ids'].to(device)
                intent_ids = batch['intent_ids'].to(device)

                slot_logits, intent_logits = model(token_ids)

                slot_loss = slot_loss_fn(slot_logits.view(-1, slot_logits.size(-1)), slot_ids.view(-1))
                intent_loss = intent_loss_fn(intent_logits, intent_ids)
                val_loss = slot_loss_weight * slot_loss + intent_loss_weight * intent_loss
                total_val_loss += val_loss.item()

                intent_preds = torch.argmax(intent_logits, dim=1)
                correct_intents += (intent_preds == intent_ids).sum().item()
                total_examples += intent_ids.size(0)

                slot_preds = torch.argmax(slot_logits, dim=-1)  # [batch, seq_len]

                # flatten and drop PAD positions before collecting for metrics
                mask = slot_ids != pad_idx
                all_slot_preds.extend(slot_preds[mask].cpu().tolist())
                all_slot_labels.extend(slot_ids[mask].cpu().tolist())

        avg_val_loss = total_val_loss / len(valid_loader)
        intent_acc = correct_intents / total_examples
        slot_acc = sum(p == l for p, l in zip(all_slot_preds, all_slot_labels)) / len(all_slot_labels)

        print(f'  [Validation @ epoch {epoch}] val_loss={avg_val_loss:.4f} intent_acc={intent_acc:.4f} slot_acc={slot_acc:.4f}', flush=True)

        # per-slot-class precision/recall/f1, skipping PAD entirely since it's excluded from the arrays above
        target_names = [idx2slot[i] for i in sorted(set(all_slot_labels))]
        report = classification_report(
            all_slot_labels, all_slot_preds,
            labels=sorted(set(all_slot_labels)),
            target_names=target_names,
            zero_division=0
        )
        print(f'  [Slot classification report @ epoch {epoch}]\n{report}', flush=True)
import json
import time
import numpy as np
import data.utils as d
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.optim import Adam


## Read dictionaries
train_data_path = 'data/snips_processed/train.jsonl'
valid_data_path = 'data/snips_processed/valid.jsonl'
intent_path =  'data/snips_processed/intent2idx.json'
slot_path = 'data/snips_processed/slot2idx.json'
word_path ='data/snips_processed/word2idx.json'
train_data = d._load_data(path=train_data_path)
valid_data = d._load_data(path=valid_data_path)
intent2idx,idx2intent = d._load_dictionary(path=intent_path)
slot2idx,idx2slot = d._load_dictionary(path=slot_path)
word2idx,idx2word = d._load_dictionary(path=word_path)


train_dataset = d.SnipsDataset(train_data)
valid_dataset = d.SnipsDataset(valid_data)


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=d.make_collate_fn(word2idx['PAD'],slot2idx['PAD']))
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, collate_fn=d.make_collate_fn(word2idx['PAD'],slot2idx['PAD']))


## bidirectional LSTM with embedding

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = d.JointIntentSlotModel(vocab_size=len(word2idx), num_slots=len(slot2idx), num_intents=len(intent2idx), pad_idx=word2idx['PAD']).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

intent_loss_fn = nn.CrossEntropyLoss()
slot_loss_fn = nn.CrossEntropyLoss(ignore_index=slot2idx['PAD'])

print(f'Using device: {device}')
print(next(model.parameters()).device)


num_epochs = 30
slot_loss_weight = 1.0
intent_loss_weight = 1.0
pad_idx = slot2idx['PAD']
metrics_history = []
best_val_loss = float('inf')

for epoch in range(1, num_epochs + 1):
    model.train()
    total_train_loss = 0.0
    epoch_start = time.time()

    for batch in train_loader:
        token_ids = batch['token_ids'].to(device)
        slot_ids = batch['slot_ids'].to(device)
        intent_ids = batch['intent_ids'].to(device)

        mask = (slot_ids != pad_idx)

        optimizer.zero_grad()

        slot_logits, intent_logits = model(token_ids)

        slot_loss = model.slot_loss(slot_logits, slot_ids, mask)
        intent_loss = intent_loss_fn(intent_logits, intent_ids)

        loss = slot_loss_weight * slot_loss + intent_loss_weight * intent_loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    epoch_time = time.time() - epoch_start
    print(f'[Epoch {epoch}/{num_epochs}] train_loss={avg_train_loss:.4f} time={epoch_time:.1f}s', flush=True)
    epoch_record = {'epoch':epoch,'avg_train_loss':avg_train_loss,'epoch_time_s': epoch_time}
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
                mask = (slot_ids != pad_idx)

                slot_logits, intent_logits = model(token_ids)

                slot_loss = model.slot_loss(slot_logits, slot_ids, mask)
                intent_loss = intent_loss_fn(intent_logits, intent_ids)
                val_loss = slot_loss_weight * slot_loss + intent_loss_weight * intent_loss
                total_val_loss += val_loss.item()

                intent_preds = torch.argmax(intent_logits, dim=1)
                correct_intents += (intent_preds == intent_ids).sum().item()
                total_examples += intent_ids.size(0)

                # CRF decode returns a list (len=batch) of variable-length tag-id lists,
                # already trimmed to each example's real (unpadded) length
                decoded_batch = model.decode_slots(slot_logits, mask)
                for i, decoded_seq in enumerate(decoded_batch):
                    true_len = mask[i].sum().item()
                    true_seq = slot_ids[i, :true_len].cpu().tolist()
                    all_slot_preds.extend(decoded_seq)
                    all_slot_labels.extend(true_seq)
        avg_val_loss = total_val_loss / len(valid_loader)
        intent_acc = correct_intents / total_examples
        slot_acc = sum(p == l for p, l in zip(all_slot_preds, all_slot_labels)) / len(all_slot_labels)
        print(f'  [Validation @ epoch {epoch}] val_loss={avg_val_loss:.4f} intent_acc={intent_acc:.4f} slot_acc={slot_acc:.4f}', flush=True)
        epoch_record.update({'val_loss':avg_val_loss,'intent_acc':intent_acc,'slot_acc':slot_acc,
                             'classification_report':classification_report(all_slot_labels,
                                                   all_slot_preds,
                                                   labels=sorted(set(all_slot_labels)),
                                                   target_names=[idx2slot[i] for i in sorted(set(all_slot_labels))],
                                                   zero_division=0,
                                                   output_dict=True)
                             })
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'nlu_crf_best.pt')
            print(f'  [Checkpoint] New best model saved at epoch {epoch} (val_loss={avg_val_loss:.4f})', flush=True)   
    metrics_history.append(epoch_record)

metrics_log_path = 'nlu_metrics/metrics.json'
with open(metrics_log_path, 'w') as f:
    json.dump(metrics_history, f, indent=2)
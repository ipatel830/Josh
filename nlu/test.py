import json
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import data.utils as d



data_path = 'data/snips_processed/test.jsonl'
intent_path =  'data/snips_processed/intent2idx.json'
slot_path = 'data/snips_processed/slot2idx.json'
word_path ='data/snips_processed/word2idx.json'
test_data = d._load_data(path=data_path)
intent2idx,idx2intent = d._load_dictionary(path=intent_path)
slot2idx,idx2slot = d._load_dictionary(path=slot_path)
word2idx,idx2word = d._load_dictionary(path=word_path)


device = ('cuda' if torch.cuda.is_available() else 'cpu')
model = d.JointIntentSlotModel(vocab_size=len(word2idx), num_slots=len(slot2idx), num_intents=len(intent2idx), pad_idx=word2idx['PAD'])
state_dict = torch.load('nlu_crf_best.pt',weights_only=True)
model.load_state_dict(state_dict)



model.to(device)
model.eval()

test_dataset = d.SnipsDataset(test_data)
collate = d.make_collate_fn(token_pad_idx=word2idx['PAD'], slot_pad_idx=slot2idx['PAD'])
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate)

pad_idx = slot2idx['PAD']
correct_intents = 0
total_examples = 0
all_slot_preds = []
all_slot_labels = []


with torch.no_grad():
    for batch in test_loader:
        token_ids = batch['token_ids'].to(device)
        slot_ids = batch['slot_ids'].to(device)
        intent_ids = batch['intent_ids'].to(device)
        mask = (slot_ids != pad_idx)

        slot_logits, intent_logits = model(token_ids)

        intent_preds = torch.argmax(intent_logits, dim=1)
        correct_intents += (intent_preds == intent_ids).sum().item()
        total_examples += intent_ids.size(0)

        decoded_batch = model.decode_slots(slot_logits, mask)
        for i, decoded_seq in enumerate(decoded_batch):
            true_len = mask[i].sum().item()
            true_seq = slot_ids[i, :true_len].cpu().tolist()
            all_slot_preds.extend(decoded_seq)
            all_slot_labels.extend(true_seq)

intent_acc = correct_intents / total_examples
slot_acc = sum(p == l for p, l in zip(all_slot_preds, all_slot_labels)) / len(all_slot_labels)

print(f'[TEST] intent_acc={intent_acc:.4f} slot_acc={slot_acc:.4f}')

report = classification_report(
    all_slot_labels, all_slot_preds,
    labels=sorted(set(all_slot_labels)),
    target_names=[idx2slot[i] for i in sorted(set(all_slot_labels))],
    zero_division=0
)
print(report)
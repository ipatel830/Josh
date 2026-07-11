import json
from collections import Counter

# folders = ['train','test','valid']

# for folder in folders:
#     examples = []
#     with open(f'snips/{folder}','r') as f:
#         for line in f:
#             line = line.strip()
#             if not line: continue
#             seq_part, intent = line.split(' <=> ')
#             tokens,slot_tags = [],[]
#             for pair in seq_part.split(' '):
#                 word,tag = pair.rsplit(':',1)
#                 tokens.append(word)
#                 slot_tags.append(tag)
#             assert len(tokens) == len(slot_tags)
#             examples.append({'tokens':tokens,'slot_tags':slot_tags,'intent':intent})
#         # write to folder
#         with open(f"snips_processed/{folder}.jsonl","w") as out:
#             for ex in examples:
#                 out.write(json.dumps(ex)+'\n')

#     print(f"{folder}: {len(examples)} examples")



### create slot_tags -> idx dictionary

# with open('snips/vocab.slot') as f:
#     slot2idx = {tag.strip(): i for i,tag in enumerate(f)}
#     slot2idx['PAD'] = len(slot2idx)
# with open('snips_processed/slot2idx.json','w') as f:
#     json.dump(slot2idx,f)

# ### create intent -> idx dictionary

# with open('snips/vocab.intent') as f:
#     intent2idx = {intent.strip(): i for i,intent in enumerate(f)}

# with open('snips_processed/intent2idx.json','w') as f:
#     json.dump(intent2idx,f)

# ### inverse idx -> intent
# idx2intent = {idx : intent for intent,idx in intent2idx.items()}

# with open('snips_processed/idx2intent.json','w') as f:
#     json.dump(idx2intent,f)

### create token dictionary (word tokenization)


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f]
    
def tokens_to_ids(tokens,word2idx):
    return [word2idx.get(tok.lower(),word2idx['UNK']) for tok in tokens]

def slot_to_ids(slot_tags, slot2idx):
    return [slot2idx[tag] for tag in slot_tags]  # KeyError if tag missing, not silent None

with open('snips_processed/word2idx.json') as f:
    word2idx = json.load(f)
with open('snips_processed/slot2idx.json') as f:
    slot2idx = json.load(f)
with open('snips_processed/intent2idx.json') as f:
    intent2idx = json.load(f)

for file in ['train.jsonl', 'valid.jsonl', 'test.jsonl']:
    data = load_jsonl(f'snips_processed/{file}')

    for ex in data:
        ex['token_ids'] = tokens_to_ids(ex['tokens'], word2idx)
        ex['slot_ids'] = slot_to_ids(ex['slot_tags'], slot2idx)
        ex['intent_id'] = intent2idx[ex['intent']]

    with open(f'snips_processed/{file}', 'w') as f:
        for ex in data:
            f.write(json.dumps(ex) + '\n')

    print(f'{file}: done, {len(data)} examples')

# word_counts = Counter()
# for ex in train_data:
#     for tok in ex['tokens']:
#         word_counts[tok.lower()] += 1

# min_freq = 2
# word2idx = {'PAD':0,'UNK':1}
# for word,count in word_counts.items():
    # if count >= min_freq:
        # word2idx[word] = len(word2idx)

# print(f'vocab size {len(word2idx)}')


# with open('snips_processed/word2idx.json','w') as f:
#     json.dump(word2idx,f)

# for ex in train_data:
#     ex['tokens_ids'] = tokens_to_ids(ex['tokens'],word2idx)

# with open('snips_processed/train.jsonl','w') as f:
#     for ex in train_data:
#         f.write(json.dumps(ex)+'\n')

    
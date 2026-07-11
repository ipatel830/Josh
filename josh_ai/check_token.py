import json
from transformers import Wav2Vec2CTCTokenizer 




stt_tokenizer = '../STT/data/wav2vec2_tokenizer'
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(stt_tokenizer)

print(tokenizer.vocab_size)

with open(f'{stt_tokenizer}/vocab.json') as f:
    raw_vocab = json.load(f)

vocab_size = len(raw_vocab)  # 30, guaranteed to match your checkpoint

vocab = [None] * vocab_size
for token, idx in raw_vocab.items():
    vocab[idx] = token
vocab[tokenizer.pad_token_id] = ""
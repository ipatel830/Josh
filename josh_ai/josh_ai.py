import utils as u
import torch
import torchaudio
from pyctcdecode import build_ctcdecoder
from transformers import Wav2Vec2CTCTokenizer 



class VoiceAssistantPipeline:
    def __init__(self, stt_path,stt_tokenizer, nlu_path, kenlm_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(stt_tokenizer)
        vocab = [None] * len(self.tokenizer)
        for token, idx in self.tokenizer.get_vocab().items():
            vocab[idx] = token
        vocab[self.tokenizer.pad_token_id] = ""
        vocab_size = self.tokenizer.vocab_size
        self.decoder = build_ctcdecoder(
            labels=vocab,
            kenlm_model_path=kenlm_path,
            alpha=0.5,
            beta=1.0,
        )
        # --- STT side ---
        self.stt_model = u.S2T(n_mels=80,vocab_size=vocab_size)
        self.stt_model.load_state_dict(torch.load(stt_path,map_location=self.device))
        self.stt_model.to(self.device)
        self.stt_model.eval()
        # --- NLU side ---
        self.word2idx, _ = u._load_dictionary('nlu/data/snips_processed/word2idx.json')
        self.slot2idx, self.idx2slot = u._load_dictionary('nlu/data/snips_processed/slot2idx.json')
        self.intent2idx, self.idx2intent = u._load_dictionary('nlu/data/snips_processed/intent2idx.json')
        self.nlu_model = u.JointIntentSlotModel(vocab_size=len(self.word2idx), 
                                                num_slots=len(self.slot2idx),
                                                num_intents=len(self.intent2idx),
                                                pad_idx=self.word2idx['PAD'])
        self.nlu_model.load_state_dict(torch.load(nlu_path, map_location=self.device))
        self.nlu_model.to(self.device).eval()

    def transcribe(self, audio_path) -> str:
        audio, sample_rate = torchaudio.load(audio_path)
        if sample_rate != 16000:
            audio = torchaudio.functional.resample(audio,sample_rate,16000)
            sample_rate = 16000
        mel_spec = u.STM(audio,sample_rate)
        with torch.no_grad():
            logits = self.stt_model(mel_spec.float().to(self.device))
        log_probs = torch.nn.functional.log_softmax(logits,dim=-1)
        log_probs_np = log_probs.squeeze(0).detach().cpu().numpy()
        text = self.decoder.decode(log_probs_np)
        return text.strip().lower()

    def understand(self, text: str) -> dict:
        tokens = text.strip().split()
        if not tokens:
            return {'intent':None,'slots':{}}
        
        token_ids = [self.word2idx.get(tok.lower(),self.word2idx['UNK']) for tok in tokens]
        input_tensor = torch.tensor([token_ids],dtype=torch.long).to(self.device)
        mask = torch.ones_like(input_tensor,dtype=torch.bool)

        with torch.no_grad():
            slot_logits, intent_logits = self.nlu_model(input_tensor)
            intent_id = torch.argmax(intent_logits,dim=1).item()
            intent = self.idx2intent[intent_id]

            slot_id_seq = self.nlu_model.decode_slots(slot_logits,mask)[0]
            slot_tags = [self.idx2slot[i] for i in slot_id_seq]

        slots = {}
        current_slot_name = None
        current_slot_words = []
        
        for tok,tag in zip(tokens,slot_tags):
            if tag.startswith('B-'):
                if current_slot_name:
                    slots[current_slot_name] = ' '.join(current_slot_words)
                current_slot_name = tag[2:]
                current_slot_words = [tok]
            elif tag.startswith('I-') and current_slot_name == tag[2:]:
                current_slot_words.append(tok)
            else:
                if current_slot_name:
                    slots[current_slot_name] = ' '.join(current_slot_words)
                current_slot_name = None
                current_slot_words = []
        if current_slot_name:
            slots[current_slot_name] = ' '.join(current_slot_words)

        return {'intent': intent, 'slots': slots}

    def run(self, audio) -> dict:
        text = self.transcribe(audio)
        result = self.understand(text)
        result['transcription'] = text
        return result
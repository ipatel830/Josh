import utils as u
import json
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class VoiceAssistantPipeline:
    def __init__(self, whisper_path, nlu_path):

        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        self.whisper_processor = WhisperProcessor.from_pretrained(whisper_path)
        self.whisper_model = WhisperForConditionalGeneration.from_pretrained(whisper_path)
        self.whisper_model.to(self.device)
        self.whisper_model.eval()

        # --- NLU side ---
        self.word2idx, _ = u._load_dictionary('models/nlu_dependencies/word2idx.json')
        self.slot2idx, self.idx2slot = u._load_dictionary('models/nlu_dependencies/slot2idx.json')
        self.intent2idx, self.idx2intent = u._load_dictionary('models/nlu_dependencies/intent2idx.json')
        self.nlu_model = u.JointIntentSlotModel(vocab_size=len(self.word2idx), 
                                                num_slots=len(self.slot2idx),
                                                num_intents=len(self.intent2idx),
                                                pad_idx=self.word2idx['PAD'])
        self.nlu_model.load_state_dict(torch.load(nlu_path, map_location=self.device))
        self.nlu_model.to(self.device).eval()

    def transcribe(self, audio_path) -> str:
        audio, sample_rate = torchaudio.load(audio_path)

        if audio.shape[0] > 1:
            audio = audio.mean(dim=0, keepdim=True)

        target_sr = 16000
        if sample_rate != target_sr:
            audio = torchaudio.functional.resample(audio, sample_rate, target_sr)
            sample_rate = target_sr

        audio_np = audio.squeeze(0).numpy()

        inputs = self.whisper_processor(audio_np, sampling_rate=target_sr, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)

        with torch.no_grad():
            predicted_ids = self.whisper_model.generate(input_features, num_beams=1)  # num_beams=1 = greedy

        text = self.whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
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

    def run(self, audio):
        text = self.transcribe(audio)
        result = self.understand(text)
        result['transcription'] = text
        return result
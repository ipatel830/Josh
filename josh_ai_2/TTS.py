import json
import os
import random
import torch
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    WhisperFeatureExtractor,
)
from datasets import load_dataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SpeechT5 models...")
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(DEVICE)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(DEVICE)


whisper_fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")


embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
NUM_SPEAKERS = len(embeddings_dataset)


def random_speaker_embedding():
    idx = random.randint(0, NUM_SPEAKERS - 1)
    return torch.tensor(embeddings_dataset[idx]["xvector"]).unsqueeze(0).to(DEVICE)


def build_target_string(tokens, slot_tags, intent):

    target = f"<intent>{intent}"
    current_slot = None
    current_words = []

    for tok, tag in zip(tokens, slot_tags):
        if tag.startswith("B-"):
            if current_slot:
                target += f"<{current_slot}>{' '.join(current_words)}"
            current_slot = tag[2:]
            current_words = [tok]
        elif tag.startswith("I-") and current_slot == tag[2:]:
            current_words.append(tok)
        else:
            if current_slot:
                target += f"<{current_slot}>{' '.join(current_words)}"
            current_slot = None
            current_words = []

    if current_slot:
        target += f"<{current_slot}>{' '.join(current_words)}"

    return target


def synthesize_split(jsonl_path, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True)

    with open(jsonl_path, "r") as f:
        examples = [json.loads(line) for line in f]

    if limit:
        examples = examples[:limit]

    for i, ex in enumerate(examples):
        tokens = ex["tokens"]
        slot_tags = ex["slot_tags"]
        intent = ex["intent"]

        sentence = " ".join(tokens)
        target_string = build_target_string(tokens, slot_tags, intent)

        speaker_embedding = random_speaker_embedding()
        inputs = tts_processor(text=sentence, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            speech = tts_model.generate_speech(
                inputs["input_ids"], speaker_embedding, vocoder=vocoder
            )

        speech_np = speech.cpu().numpy()


        mel_features = whisper_fe(
            speech_np, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)  # shape: [n_mels, 3000] -- Whisper's fixed 30s window

        sample_dir = os.path.join(out_dir, str(i))
        os.makedirs(sample_dir, exist_ok=True)

        torch.save(
            {
                "mel_spec": mel_features,
                "target_string": target_string,
                "text": sentence,
            },
            os.path.join(sample_dir, "sample.pt"),
        )

        if i % 100 == 0:
            print(f"  [{out_dir}] {i}/{len(examples)} synthesized")

    print(f"Done: {len(examples)} examples -> {out_dir}")


if __name__ == "__main__":
    synthesize_split("nlu/data/snips_processed/train.jsonl", "synthetic_data/train")
    synthesize_split("nlu/data/snips_processed/valid.jsonl", "synthetic_data/valid")
    synthesize_split("nlu/data/snips_processed/test.jsonl", "synthetic_data/test")
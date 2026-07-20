import json
import os
import random
import numpy as np
import torch
from transformers import (
    SpeechT5Processor,
    SpeechT5ForTextToSpeech,
    SpeechT5HifiGan,
    WhisperFeatureExtractor,
)
from datasets import load_dataset



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading SpeechT5 models...")
tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

whisper_fe = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny")

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
NUM_SPEAKERS = len(embeddings_dataset)


def random_speaker_embedding():
    idx = random.randint(0, NUM_SPEAKERS - 1)
    return torch.tensor(embeddings_dataset[idx]["xvector"]).unsqueeze(0).to(device)


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


def add_gaussian_noise(speech, snr_db):

    signal_power = np.mean(speech ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), speech.shape)
    return speech + noise


def apply_simple_reverb(speech, decay=0.3, delay_samples=800):
    reverb = np.zeros_like(speech)
    if delay_samples < len(speech):
        # splice speech after delay_samples and set it equal to the decay of that delayed sample
        reverb[delay_samples:] = speech[:-delay_samples] * decay
    return speech + reverb


def random_gain(speech, min_db=-6, max_db=6):
    gain_db = random.uniform(min_db, max_db)
    gain_factor = 10 ** (gain_db / 20)
    return speech * gain_factor


def augment_audio(speech_np):
    if random.random() < 0.7:
        snr = random.uniform(5, 25)  # random noise severity
        speech_np = add_gaussian_noise(speech_np, snr_db=snr)

    if random.random() < 0.4:
        speech_np = apply_simple_reverb(speech_np)

    if random.random() < 0.5:
        speech_np = random_gain(speech_np)

    max_val = np.max(np.abs(speech_np))
    if max_val > 1.0:
        speech_np = speech_np / max_val

    return speech_np.astype(np.float32)


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
        inputs = tts_processor(text=sentence, return_tensors="pt").to(device)

        with torch.no_grad():
            speech = tts_model.generate_speech(
                inputs["input_ids"], speaker_embedding, vocoder=vocoder
            )

        speech_np = speech.cpu().numpy()
        speech_np = augment_audio(speech_np)


        mel_features = whisper_fe(
            speech_np, sampling_rate=16000, return_tensors="pt"
        ).input_features.squeeze(0)  # shape: [n_mels, 3000]


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
    synthesize_split("snips_processed/train.jsonl", "synthetic_data/train")
    synthesize_split("snips_processed/valid.jsonl", "synthetic_data/valid")
    synthesize_split("snips_processed/test.jsonl", "synthetic_data/test")
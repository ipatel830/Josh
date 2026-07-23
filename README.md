# Josh — On-Device Voice Assistant

A small-footprint, Alexa-style voice assistant designed to run within tight memory
constraints (target: ~2GB or less), covering the full pipeline from raw audio to
actionable intent.

## Project Goals

- Fully local, offline-capable speech understanding — no cloud APIs required at runtime.
- Small enough to run on resource-constrained hardware (originally targeting
  50–250MB RAM class devices, currently validated within a 2GB Docker budget).
- Understand open-vocabulary commands ("play *bohemian rhapsody* by *queen*"),
  not just a fixed set of pre-defined phrases.

---

## Phase 1: Two-Stage Pipeline (STT → NLU)

The initial architecture split the problem into two independently trained models.

### NLU: Joint Intent + Slot Filling

- **Data:** SNIPS (7 intents, 72 BIO slot tags, ~14k utterances), parsed from
  raw `word:TAG` format into structured JSONL with token/slot/intent fields.
- **Model:** Embedding → BiLSTM → two heads (intent classification, per-token
  slot tagging), with a CRF layer added on top of the slot head to enforce
  valid BIO tag transitions.
- **Result:** ~0.98 intent accuracy, ~0.95 slot accuracy on held-out test data.
  Rare slot types (low-support classes) remained the weakest area, as expected
  given data scarcity rather than a modeling flaw.

### STT: Custom CTC Model

- **Data:** LibriSpeech, mel-spectrogram features (80 mel bins), trained with
  a CNN + Transformer encoder + CTC output head.
- **Decoding:** Initially KenLM + beam search (via `pyctcdecode`), later
  compared against greedy CTC decoding.
- **Key finding:** The model performed well on LibriSpeech-domain audio (clean,
  narrated audiobook speech) but generalized poorly to real microphone input —
  casual commands, different acoustic conditions, different speaking style.
  This was a genuine domain-mismatch problem, not a bug, and became the
  motivating reason for Phase 2.

### Deployment (Phase 1)

- Packaged into a Docker container with a FastAPI HTTP interface
  (`/transcribe` endpoint accepting uploaded audio).
- Significant effort went into trimming dependencies (dropping `pyctcdecode`/
  KenLM's >1GB language model, using CPU-only PyTorch, multi-stage Docker
  build) to fit a strict memory budget.

---

## Phase 2: Pivot to Whisper + Rethinking the Architecture

Given the domain-mismatch problem, the custom CTC STT model was replaced with
**Whisper-tiny** (OpenAI, MIT licensed), pretrained on a much larger and more
diverse real-world audio dataset. This immediately improved transcription
quality on real, casual speech without any additional training.

This also opened the door to a bigger architectural question: rather than
maintaining two separately trained models (STT text output → separate NLU
model), could a single end-to-end model go directly from audio to structured
intent/slot output?

### Why Not Pure Classification Heads

An initial idea — pooling Whisper's encoder output into fixed classification
heads (similar to the Fluent Speech Commands task structure) — was considered
and rejected. Fluent Speech Commands uses a closed vocabulary (fixed action/
object/location categories), which cannot represent open-ended slot values
like arbitrary artist or playlist names. This ruled out a pure classification-
head approach for this project's actual requirements.

### Current Direction: Encoder-Decoder Structured Generation

The adopted approach keeps Whisper's full encoder-decoder structure, but
**changes what the decoder is trained to output**: instead of plain
transcription, it generates a structured string directly, e.g.:

```
<intent>PlayMusic<track>bohemian rhapsody<artist>queen
```

This preserves open-vocabulary slot filling (since the decoder can still
generate arbitrary words) while collapsing STT and NLU into a single trained
model — removing the need for a separate NLU model and separate tokenization/
decoding logic entirely.

**Planned split for the deployed device:**
- Mel-spectrogram computation happens on-device (lightweight, no model needed).
- The encoder + decoder (fine-tuned) handle everything from mel-spec input to
  final structured output.

### Training Data for the Unified Model

Since no existing dataset pairs audio with this exact structured output, a
synthetic data pipeline was built:

- SNIPS text/intent/slot labels are converted into the structured target
  string format.
- Audio is synthesized per utterance using HuggingFace's SpeechT5 TTS, with
  **randomized speaker embeddings** for voice diversity.
- Synthesized audio is converted to mel-spectrograms using Whisper's own
  `WhisperFeatureExtractor`, guaranteeing exact preprocessing alignment with
  the encoder.
- **Augmentation** (randomized Gaussian noise at varying SNR, simple reverb,
  random gain) is applied to synthetic clips to reduce the gap between clean
  TTS audio and real-world acoustic conditions.
- Output is saved per-sample (`mel_spec`, `target_string`, `text`) for
  straightforward loading during training.

This is a known compromise: synthetic TTS audio, even augmented, has a
domain ceiling that real recorded speech does not share. Mixing in real
recorded utterances alongside the synthetic set remains a planned next step.

---

## Known Limitations / Open Items

- The unified encoder-decoder model has not yet been trained or evaluated —
  the current state is data pipeline and architecture design, not results.
- Synthetic-only training data will likely underperform on truly novel real
  voices/environments until real recorded audio is incorporated.
- Rare slot types remain a weak point inherited from the underlying SNIPS
  data distribution, independent of architecture changes.
- Quantization / ONNX export for the final deployed model has not yet been
  applied, though the memory budget has comfortable headroom at present.
- Fluent Speech Commands dataset licensing is ambiguous (academic-research
  language on the official page vs. a permissive license on a derived model
  checkpoint) — not currently used as training data due to this uncertainty.

## Next Steps

1. Fine-tune Whisper's decoder (encoder frozen initially) on the synthetic
   structured-output dataset.
2. Evaluate against a held-out synthetic test set, then against real
   recorded audio to measure the synthetic-to-real generalization gap.
3. Incorporate real recorded speech into training if the gap proves large.
4. Build the on-device mel-spectrogram computation to match training
   exactly, removing any dependency on `WhisperFeatureExtractor` at
   inference time.
5. Re-run the full Docker memory/latency benchmark once the unified model
   replaces the two-stage pipeline.

import josh_ai as j_ai
import torch


stt_path = '../STT/best_model.pt'
stt_tokenizer = '../STT/data/wav2vec2_tokenizer'
nlu_path = '../nlu/nlu_crf_best.pt'
kenlm_path = '4-gram.arpa'
device = ('cuda' if torch.cuda.is_available() else 'cpu')


assistant = j_ai.VoiceAssistantPipeline(stt_path,stt_tokenizer,nlu_path,kenlm_path,device=device)


audio_path = '/data/mahi.m4a'

output = assistant.run(audio_path)
print(output)

print(output['transcription'])

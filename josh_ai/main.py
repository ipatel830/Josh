import josh_ai as j_ai


stt_path = '../STT/best_model.pt'
nlu_path = '../nlu/nlu_crf_best.pt'
kenlm_path = '../STT/model/4-gram.arpa'

assistant = j_ai.VoiceAssistantPipeline(stt_path,nlu_path,kenlm_path)


audio_path = 'data/ishan.m4a'

output = assistant.run(audio_path)
print(output)

# print(output['transcription'])

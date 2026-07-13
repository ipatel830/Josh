import josh_ai as j_ai


whisper_path = 'models/whisper-tiny-local/'
nlu_path = 'models/nlu_model.pt'

assistant = j_ai.VoiceAssistantPipeline(whisper_path,nlu_path)


audio_path = '../josh_test_data/mahi.m4a'

output = assistant.run(audio_path)
print(output)

# print(output['transcription'])

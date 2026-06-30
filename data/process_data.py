
from dataset import process_data
from transformers import Wav2Vec2CTCTokenizer
import torchaudio.datasets as dataset


### download all data
dataset.LIBRISPEECH(root='./LibriSpeech/',url='train-clean-100',download=True)

dataset.LIBRISPEECH(root='./LibriSpeech/',url='train-clean-360',download=True)

dataset.LIBRISPEECH(root='./LibriSpeech/',url='test-clean',download=True)

### combine 100 and 360 datasets, process, and tokenize train and test

processed = process_data(root_folder='LibriSpeech/train-clean-100',train=True)
processed = process_data(root_folder='LibriSpeech/train-clean-360',train=True)

tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("./wav2vec2_tokenizer")
data_len,missing_audio  = processed.tokenize(tokenizer)


process_test = process_data(root_folder='LibriSpeech/test-clean',train=False)
data_len,missing_audio = process_test.tokenize(tokenizer)




print(f"Length of test data: {data_len}")
print(f"Missing test audio: {missing_audio}")


with open('data_processing_output.txt','w') as f:
    f.write(f'Length of dataset: {data_len}')
    f.write(f"Missing audio: {missing_audio}")
    f.close()






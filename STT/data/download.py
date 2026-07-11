import torchaudio.datasets as dataset
import boto3
import os
from config import bucket_name

### download all data
dataset.LIBRISPEECH(root='./LibriSpeech/',url='train-clean-100',download=True)

dataset.LIBRISPEECH(root='./LibriSpeech/',url='train-clean-360',download=True)

dataset.LIBRISPEECH(root='./LibriSpeech/',url='test-clean',download=True)

os.system(f'aws s3 sync ./LibriSpeech/ s3://{bucket_name}/LibriSpeech/')
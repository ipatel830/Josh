import os
import boto3
from dataset import process_data
from transformers import Wav2Vec2CTCTokenizer
from config import bucket_name

BUCKET = bucket_name
s3     = boto3.client('s3')

def upload_folder_to_s3(local_folder, bucket, s3_prefix):
    files = []
    for dirpath, _, filenames in os.walk(local_folder):
        for filename in filenames:
            files.append(os.path.join(dirpath, filename))

    print(f"Uploading {len(files)} files to s3://{bucket}/{s3_prefix}")
    for i, local_path in enumerate(files):
        s3_key = os.path.join(
            s3_prefix,
            os.path.relpath(local_path, local_folder)
        )
        s3.upload_file(local_path, bucket, s3_key)
        if (i + 1) % 500 == 0:
            print(f"  uploaded {i+1}/{len(files)}")
    print(f"Upload complete: {len(files)} files")

# tokenizer
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained('./wav2vec2_tokenizer')

#preprocess both train splits into the same output folder 
print("Processing train-clean-100...")
train_100 = process_data(root_folder='LibriSpeech/train-clean-100', train=True)


print("Processing train-clean-360...")
train_360 = process_data(root_folder='LibriSpeech/train-clean-360', train=True)
data_len,missing_audio = train_360.tokenize(tokenizer)

#  preprocess test 
print("Processing test-clean...")
test = process_data(root_folder='LibriSpeech/test-clean', train=False)
test_len,test_missing = test.tokenize(tokenizer)

# upload to S3
print("Uploading processed train to S3...")
upload_folder_to_s3('processed_train/', BUCKET, 'processed_train/')

print("Uploading processed test to S3...")
upload_folder_to_s3('processed_test/', BUCKET, 'processed_test/')

# ── save and upload summary 

summary = (
    f"Train dataset length: {data_len}\n"
    f"Train missing audio:  {missing_audio}\n"
    f"Test dataset length:  {test_len}\n"
    f"Test missing audio:   {test_missing}\n"
)
print(summary)
with open('data_processing_output.txt', 'w') as f:
    f.write(summary)

s3.upload_file('data_processing_output.txt', BUCKET, 'logs/data_processing_output.txt')
print("Done.")
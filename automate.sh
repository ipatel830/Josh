#!/bin/bash
set -e  

#  miniconda 
if [ ! -d "$HOME/miniconda" ]; then
    echo "Installing miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    rm ~/miniconda.sh
else
    echo "Miniconda already installed, skipping..."
fi

# add conda to PATH for this script session
export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"

# pull repo
if [ -d "Josh" ]; then
    echo "Repo exists, pulling latest..."
    cd Josh && git pull
else
    echo "Cloning repo..."
    git clone https://github.com/ipatel830/Josh.git && cd Josh
fi

# conda environment 
if conda env list | grep -q "josh_environment"; then
    echo "Environment exists, skipping create..."
else
    echo "Creating conda environment..."
    conda env create -f environment.yml -n josh_environment
fi

# activate conda env
CONDA_RUN="conda run -n josh_environment"

# pull processed data from S3 
echo "Pulling processed data from S3..."
aws s3 sync s3://your-bucket-name/processed_data/ ./data/processed_train/ --quiet

# pull LM file if not already there
if [ ! -f "./model/lm/4-gram.arpa" ]; then
    echo "Pulling language model from S3..."
    mkdir -p ./model/lm
    aws s3 cp s3://your-bucket-name/lm/4-gram.arpa ./model/lm/4-gram.arpa
fi

# run training 
echo "Starting training..."
cd model/
$CONDA_RUN nohup python train.py > training.log 2>&1 &
echo $! > train.pid
echo "Training started with PID $(cat train.pid)"
echo "Monitor with: tail -f Josh/model/training.log"


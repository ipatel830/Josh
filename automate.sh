#!/bin/bash
set -e

# ── miniconda ─────────────────────────────────────────────────────────────────
if [ ! -d "$HOME/miniconda" ]; then
    echo "Installing miniconda..."
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
    bash ~/miniconda.sh -b -p $HOME/miniconda
    rm ~/miniconda.sh
else
    echo "Miniconda already installed, skipping..."
fi

export PATH="$HOME/miniconda/bin:$PATH"
source "$HOME/miniconda/etc/profile.d/conda.sh"

#  pull repo
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

CONDA_RUN="conda run -n josh_environment"

#  download and process data 
$CONDA_RUN python data/download.py || { echo "download.py failed"; exit 1; }

echo "Processing and uploading data to S3..."
$CONDA_RUN python data/process_data.py || { echo "process_data.py failed"; exit 1; }

#  run training
echo "Starting training..."
cd model/
$CONDA_RUN nohup python model_script.py > stdout.log 2>&1 &
echo $! > train.pid
echo "Training started with PID $(cat train.pid)"
echo "Monitor training log: tail -f Josh/model/training.log"
echo "Monitor stdout:       tail -f Josh/model/stdout.log"
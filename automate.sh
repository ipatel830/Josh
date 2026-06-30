#!/bin/bash

if [ -d "Josh" ]; then
  cd Josh && git pull
else
  git clone https://github.com/ipatel830/Josh.git && cd Josh
fi

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
~/miniconda/bin/conda init bash
source ~/.bashrc


conda env create -f environment.yml -n josh_environment
conda activate josh_environment

cd data 


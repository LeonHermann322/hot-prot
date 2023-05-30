#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --mail-type ALL
#SBATCH --mail-user hoangan.nguyen@student.hpi.de
#SBATCH --time=100:0:0
#SBATCH --partition=sorcery

eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/hoangan.nguyen/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/hoangan.nguyen/anaconda3/envs/hotprot/lib"
srun python applications/vae_train.py

#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --mem=64G
#SBATCH --gpus=2
#SBATCH --gpus-per-node=2
#SBATCH --mail-type ALL
#SBATCH --mail-user hoangan.nguyen@student.hpi.de
#SBATCH --partition=sorcery
#SBATCH --time=5-00:00:00
#SBATCH --exclude=ac922-[01-02],a6k5-01



eval "$(conda shell.bash hook)"
conda activate hotprot
export PYTHONPATH="/hpi/fs00/home/hoangan.nguyen/hot-prot"
export LD_LIBRARY_PATH="/hpi/fs00/home/hoangan.nguyen/anaconda3/envs/hotprot/lib"
srun python lightning/test.py

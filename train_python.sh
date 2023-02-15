#!/bin/bash
#SBATCH -A renard-molecule2022
#SBATCH --partition sorcery
#SBATCH --mem=64G
#SBATCH --gpus=1
#SBATCH --mail-type ALL
#SBATCH --mail-user hoangan.nguyen@student.hpi.de
#SBATCH --time=70:0:0

eval "$(conda shell.bash hook)"
conda activate hotshot
srun python uni_prot_train.py
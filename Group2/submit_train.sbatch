#!/bin/bash
#
# Train the CNN/MLP regressors on a finished EQ-SANS cylinder campaign.
# Runs on a mesonet A100 (train_regressor.py picks up CUDA automatically).
#
# Usage:
#   sbatch submit_train.sbatch                                          # smoke defaults
#   sbatch submit_train.sbatch submit_full_manifest.json dataset_campaign 25
#
#SBATCH --job-name=eqsans_train_%j
#SBATCH --error=eqsans_train_%j.err
#SBATCH --output=eqsans_train_%j.out
#SBATCH --nodes 1
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time 01:00:00
#SBATCH --account m26216
set -euo pipefail

export MAMBA_EXE='/projects/m26216/INSTALL/bin/micromamba'
export MAMBA_ROOT_PREFIX='/projects/m26216/INSTALL/micromamba/'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate panraid

# Known-good absolute path, confirmed via `ls -ld ~/Group2` earlier in this
# project -- not relying on SLURM's default working directory on this
# cluster a second time (see make_run_script.py's own history for why).
cd /home/georgeyumnam/Group2

MANIFEST=${1:-submit_smoke_manifest.json}
DATA=${2:-dataset_smoke}
EPOCHS=${3:-25}

echo "=== $(date)  manifest=$MANIFEST  data=$DATA  epochs=$EPOCHS  node=$(hostname) ==="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
python3 train_regressor.py --manifest "$MANIFEST" --data "$DATA" --epochs "$EPOCHS"

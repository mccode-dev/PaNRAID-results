#!/bin/bash
#
# Classical regressors (random forest, extra trees, gradient boosting, XGBoost,
# SVR, kNN, ridge) on the finished EQ-SANS campaign, on the SAME split as the
# CNN/MLP. CPU-only work; --gres=gpu:1 is kept only because every other script
# on this partition that is known to run here requests it.
#
# Usage (from ~/Group2 on Juliet):
#   sbatch submit_classical.sbatch
#   sbatch submit_classical.sbatch submit_full_manifest.json dataset_campaign eqsans_regressor_checkpoint.pt
#
#SBATCH --job-name=eqsans_classical_%j
#SBATCH --error=eqsans_classical_%j.err
#SBATCH --output=eqsans_classical_%j.out
#SBATCH --nodes 1
#SBATCH --partition=mesonet
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time 01:00:00
#SBATCH --account m26216
set -euo pipefail

export MAMBA_EXE='/projects/m26216/INSTALL/bin/micromamba'
export MAMBA_ROOT_PREFIX='/projects/m26216/INSTALL/micromamba/'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX" 2> /dev/null)"
micromamba activate panraid

cd /home/georgeyumnam/Group2

MANIFEST=${1:-submit_full_manifest.json}
DATA=${2:-dataset_campaign}
CKPT=${3:-eqsans_regressor_checkpoint.pt}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "=== $(date)  manifest=$MANIFEST  data=$DATA  ckpt=$CKPT  node=$(hostname)  cpus=$SLURM_CPUS_PER_TASK ==="
python3 -c "import sklearn; print('scikit-learn', sklearn.__version__)"
python3 -c "import xgboost; print('xgboost', xgboost.__version__)" || echo "xgboost missing: XGBoost will be skipped"
ls train_regressor.py train_classical.py

python3 train_classical.py --manifest "$MANIFEST" --data "$DATA" --nn-checkpoint "$CKPT" \
    --jobs "$SLURM_CPUS_PER_TASK"
echo "=== done $(date) ==="
ls -la classical_*.png classical_results.json

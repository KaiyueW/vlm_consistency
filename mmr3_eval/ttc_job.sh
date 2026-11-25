#!/bin/bash
# Submit with:   sbatch ttc_job.sh <model> <task> [limit]
#   sbatch ttc_job.sh llava logiqa 3        # smoke test, 3 items
#   sbatch ttc_job.sh llava logiqa          # the whole task
# Log: mmr3_eval/slurm-<jobid>.out   (watch it with: tail -f slurm-<jobid>.out)
# <changed> new file
#SBATCH --account=st-amuham01-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --job-name=ttc
#SBATCH --output=/scratch/st-amuham01-1/kwang67/vlm_consistency/mmr3_eval/slurm-%j.out

MODEL=${1:?usage: sbatch ttc_job.sh <model> <task> [limit]}
TASK=${2:?usage: sbatch ttc_job.sh <model> <task> [limit]}
LIMIT=${3:-}

source /scratch/st-amuham01-1/kwang67/vlm_consistency/env.sh
cd /scratch/st-amuham01-1/kwang67/vlm_consistency/mmr3_eval

echo "node: $(hostname)  start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv

ARGS="--model $MODEL --task $TASK --dtype bfloat16"
[ -n "$LIMIT" ] && ARGS="$ARGS --limit $LIMIT"

python run_ttc.py $ARGS
echo "end: $(date)"
#!/bin/bash

#SBATCH --job-name=table_bert

### Logging
#SBATCH --output=/checkpoint/%u/shared/table_bert/logs/%x-%j.out
#SBATCH --error=/checkpoint/%u/shared/table_bert/logs/%x-%j.err
#SBATCH --mail-user=pengcheng@fb.com
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --time 72:00:00

### Resources (note:gpu==tasks per node, otherwise chaos)
#SBATCH --gres=gpu:8
#SBATCH --mem=256GB
#SBATCH --cpus-per-task=8
###SBATCH --constraint=bldg1
###SBATCH --constraint=volta32gb

CHECKPOINT_DIR=${1:-"/checkpoint/$USER/shared/table_bert/checkpoints/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"}
mkdir -p ${CHECKPOINT_DIR}

chmod 777 scripts/wrapper.sh

echo "Starting distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"
srun --label scripts/wrapper.sh $CHECKPOINT_DIR

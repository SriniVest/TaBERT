#!/bin/bash

#SBATCH --job-name=table_bert

### Logging
#SBATCH --output=/checkpoint/%u/shared/table_bert/logs/%x-%j.out
#SBATCH --error=/checkpoint/%u/shared/table_bert/logs/%x-%j.err
#SBATCH --open-mode=append
#SBATCH --signal=USR1@120
##SBATCH --mail-user=pengcheng@fb.com
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time 72:00:00

### Resources (note:gpu==tasks per node, otherwise chaos)
#SBATCH --gres=gpu:8
#SBATCH --mem=200GB
#SBATCH --cpus-per-task=8
#SBATCH --constraint=volta32gb

if [ -z "${SLURM_JOB_ID}" ]; then
  CHECKPOINT_DIR=${1:-"/tmp/table_bert"}
else
  CHECKPOINT_DIR=${1:-"/checkpoint/$USER/shared/table_bert/checkpoints/run_${SLURM_JOB_NAME}_${SLURM_JOB_ID}"}
fi

# Module init
module purge
module load anaconda3
module load cuda/10.0 cudnn

# source deactivate
source activate /private/home/"$USER"/.conda/envs/pytorch

module load NCCL/2.4.8-1-cuda.10.0

BASEDIR=$PWD

mkdir -p ${CHECKPOINT_DIR}

>&2 echo "Starting distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"

>&2 echo "Running job ${SLURM_JOB_ID} on ${SLURM_NNODES} nodes: ${SLURM_NODELIST}"
>&2 echo "Node: ${SLURMD_NODENAME}"
>&2 echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
>&2 echo "Checkpoint dir: ${CHECKPOINT_DIR}"
>&2 echo "CUDA_LAUNCH_BLOCKING: ${CUDA_LAUNCH_BLOCKING}"

BERT_MODEL=bert-base-uncased
#DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_bindata0829_1101201723756_h5
DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_bindata0829_1108200356422
#DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_bindata0829_1108200356422_sample
BATCH_SIZE=16
LEARNING_RATE=3e-5
EPS=1e-8
WEIGHT_DECAY=0.0
MASTER_PORT=19533
CLIP_NORM=1.0
MAX_EPOCH=-1

export PYTHONPATH="$PYTHONPATH":"$BASEDIR"
PYTHON=/private/home/"$USER"/.conda/envs/pytorch/bin/python

srun --label ${PYTHON} -u train.py \
    --task vanilla \
    --data-dir ${DATASET_PATH} \
    --output-dir ${CHECKPOINT_DIR} \
    --table-bert-extra-config ${TABLE_BERT_EXTRA_CONFIG} \
    --train-batch-size ${BATCH_SIZE} \
    --learning-rate ${LEARNING_RATE} \
    --max-epoch ${MAX_EPOCH} \
    --adam-eps ${EPS} \
    --weight-decay ${WEIGHT_DECAY} \
    --master-port ${MASTER_PORT} \
    --fp16 \
    --clip-norm ${CLIP_NORM} \
    --empty-cache-freq 128

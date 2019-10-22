#!/bin/bash

#SBATCH --job-name=table_bert

### Logging
#SBATCH --output=/checkpoint/%u/shared/table_bert/logs/%x-%j.out
#SBATCH --error=/checkpoint/%u/shared/table_bert/logs/%x-%j.err
#SBATCH --mail-user=pengcheng@fb.com
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --time 72:00:00

### Resources (note:gpu==tasks per node, otherwise chaos)
#SBATCH --gres=gpu:8
#SBATCH --mem=360GB
#SBATCH --cpus-per-task=8
###SBATCH --constraint=bldg1
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
source activate /private/home/"$USER"/.conda/envs/table_bert

# load NCCL 2.4.7?
export NCCL_ROOT_DIR=/public/apps/NCCL/2.4.7-1
export LD_LIBRARY_PATH=/public/apps/NCCL/2.4.7-1/lib:${LD_LIBRARY_PATH}
export LIBRARY_PATH=/public/apps/NCCL/2.4.7-1/lib:${LIBRARY_PATH}
# bug fix for early version of NCCL
export NCCL_LL_THRESHOLD=0

BASEDIR=$PWD

mkdir -p ${CHECKPOINT_DIR}

>&2 echo "Starting distributed job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"

>&2 echo "Running job ${SLURM_JOB_ID} on ${SLURM_NNODES} nodes: ${SLURM_NODELIST}"
>&2 echo "Node: ${SLURMD_NODENAME}"
>&2 echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
>&2 echo "Checkpoint dir: ${CHECKPOINT_DIR}"

BERT_MODEL=bert-large-uncased
DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_bindata_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_token_ctxsmpl_nearest_nomaxlen_epoch15
BATCH_SIZE=8
LEARNING_RATE=3e-5
EPS=1e-8
EPOCHS=10

export MASTER_PORT=19533
export PYTHONPATH="$PYTHONPATH":"$BASEDIR"
PYTHON=/private/home/"$USER"/.conda/envs/table_bert/bin/python

srun --label ${PYTHON} -m model.train \
    --data_dir ${DATASET_PATH} \
    --output_dir ${CHECKPOINT_DIR} \
    --bert_model ${BERT_MODEL} \
    --do_lower_case \
    --train_batch_size=${BATCH_SIZE} \
    --learning_rate=${LEARNING_RATE} \
    --eps=${EPS} \
    --epochs ${EPOCHS} \
    --master_port ${MASTER_PORT} \
    --fp16

# --no_init \
# --config_file=config.bert_base.json \
# --ddp_backend apex
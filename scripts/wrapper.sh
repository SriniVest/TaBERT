#!/bin/bash

# Usage: ./wrapper.sh <CHECKPOINT_DIR>

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

if [ -z "$SLURM_JOB_ID" ]; then
  CHECKPOINT_DIR=${1:-"/tmp/table_bert"}
  mkdir -p ${CHECKPOINT_DIR}
else
  CHECKPOINT_DIR=$1
fi

BASEDIR=$PWD

echo "Kicking off $SOURCE with params $PARAMS."

export MASTER_PORT=19533

echo "Running job $SLURM_JOB_ID on $SLURM_NNODES nodes: $SLURM_NODELIST"
echo "Node: $SLURMD_NODENAME"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Checkpoint dir: $CHECKPOINT_DIR"

# DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/table_bert_data_0610_epoch150_sampled_val_ctx128
# DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/table_bert_data_0614_epoch10_sampled_val_ctx128
# DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/table_bert_data_sampled_val_ctx128_epoch20
# DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_data_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_token_ctxsmpl_concate_and_enumerate_epoch20
# DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_data_ctx128_pmsk_col0.4_pmsk_ctx0.15_tbmsk_column_ctxsmpl_nearest_epoch20
DATASET_PATH=/private/home/pengcheng/Research/datasets/table_bert/tb_data_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_token_ctxsmpl_nearest_tbfirst_epoch20
EPOCHS=10

export PYTHONPATH="$PYTHONPATH":"$BASEDIR"

NCCL_LL_THRESHOLD=0 /private/home/"$USER"/.conda/envs/table_bert/bin/python -m model.train \
    --train_data ${DATASET_PATH}/train \
    --dev_data ${DATASET_PATH}/dev \
    --output_dir ${CHECKPOINT_DIR} \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --train_batch_size=4 \
    --epochs ${EPOCHS} \
    --master_port ${MASTER_PORT} \
    --fp16

# --no_init \
# --config_file=config.bert_base.json \
# --ddp_backend apex

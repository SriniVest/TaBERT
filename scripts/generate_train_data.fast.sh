#!/bin/bash

#SBATCH --job-name=table_bert

### Logging
#SBATCH --output=/checkpoint/%u/shared/table_bert/logs/generate-data-%x-%j.out
#SBATCH --error=/checkpoint/%u/shared/table_bert/logs/generate-data-%x-%j.err
##SBATCH --mail-user=pengcheng@fb.com
#SBATCH --mail-type=END,FAIL,REQUEUE

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=20
#SBATCH --time 24:00:00

### Resources (note:gpu==tasks per node, otherwise chaos)
#SBATCH --mem=250GB
#SBATCH --cpus-per-task=1

# Module init
module purge
module load anaconda3

# source deactivate
source activate /private/home/"$USER"/.conda/envs/pytorch

export PYTHONPATH="$PYTHONPATH":"$BASEDIR"
PYTHON=/private/home/"$USER"/.conda/envs/pytorch/bin/python

# default parameters
train_corpus=/private/home/pengcheng/Research/datasets/table_data/tables.wiki_and_common_crawl.0829.jsonl
max_predictions_per_seq=200
masked_column_prob=0.2
masked_context_prob=0.15
max_context_len=128
table_mask_strategy=column_token
context_sample_strategy=concate_and_enumerate
cell_input_template="'"'column(value)(type)'"'"
column_delimiter=[SEP]
epochs=15

COMMIT_HASH=`git rev-parse --short=8 HEAD`
timestamp=`date '+%m%d%H%M%S'`

timestamp=$(python -c 'import datetime; print(datetime.datetime.utcnow().strftime("%m%d%H%M%S%f")[:-3])')
work_dir_name=tb_bindata0829_${COMMIT_HASH}_${timestamp}

work_dir=/private/home/pengcheng/Research/datasets/table_bert/${work_dir_name}

mkdir -p ${work_dir}

srun --label ${PYTHON} -m utils.prepare_bert_training_data_fast \
    --output_dir ${work_dir} \
    --train_corpus ${train_corpus} \
    --base_model_name bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate ${epochs} \
    --max_context_len ${max_context_len} \
    --table_mask_strategy ${table_mask_strategy} \
    --context_sample_strategy ${context_sample_strategy} \
    --masked_column_prob ${masked_column_prob} \
    --masked_context_prob ${masked_context_prob} \
    --max_predictions_per_seq ${max_predictions_per_seq} \
    --cell_input_template ${cell_input_template} \
    --column_delimiter ${column_delimiter}

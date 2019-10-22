#!/usr/bin/env bash

#SBATCH --job-name=table_bert_gen_data

### Logging
#SBATCH --output=/checkpoint/%u/shared/table_bert/logs/%x-%j.out
#SBATCH --error=/checkpoint/%u/shared/table_bert/logs/%x-%j.err

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --time 24:00:00

### Resources (note:gpu==tasks per node, otherwise chaos)
#SBATCH --mem=490GB
#SBATCH --cpus-per-task=80

set -e

redis-cli config set save ""
echo CONFIG SET maxmemory 400gb | redis-cli
echo CONFIG SET stop-writes-on-bgsave-error no | redis-cli

# default parameters
max_predictions_per_seq=200
masked_column_prob=0.2
masked_context_prob=0.15
max_context_len=128
table_mask_strategy=column_token
context_sample_strategy=concate_and_enumerate
context_sample_strategy=nearest
epochs=15

work_dir_name=tb_bindata0829_ctx${max_context_len}_pmsk_col${masked_column_prob}_pmsk_ctx${masked_context_prob}_tbmsk_${table_mask_strategy}_ctxsmpl_${context_sample_strategy}_nomaxlen_epoch${epochs}

work_dir=/private/home/pengcheng/Research/datasets/table_bert/${work_dir_name}

mkdir -p ${work_dir}

#     --train_corpus /private/home/pengcheng/Research/datasets/table_bert/table.wiki_and_common_crawl.jsonl \3
#     --train_corpus wiki.dump.sample.jsonl \
# Previous Training Corpus: --train_corpus /private/home/pengcheng/Research/datasets/table_bert/table.wiki_and_common_crawl.jsonl \

echo "Job Id: ${SLURM_JOB_ID}" > ${work_dir}/job_info.log

python -m model.prepare_training_data \
    --output_dir ${work_dir} \
    --train_corpus /private/home/pengcheng/Research/datasets/table_data/tables.wiki_and_common_crawl.0829.jsonl \
    --bert_model bert-base-uncased \
    --do_lower_case \
    --epochs_to_generate ${epochs} \
    --max_context_len ${max_context_len} \
    --table_mask_strategy ${table_mask_strategy} \
    --context_sample_strategy ${context_sample_strategy} \
    --masked_column_prob ${masked_column_prob} \
    --masked_context_prob ${masked_context_prob} \
    --max_predictions_per_seq ${max_predictions_per_seq} 2>>${work_dir}/err.log

redis-cli FLUSHALL
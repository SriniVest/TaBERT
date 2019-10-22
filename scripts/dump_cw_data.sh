#!/bin/bash

#SBATCH --job-name=data_preprocessing

### Logging
#SBATCH --output=/checkpoint/%u/logs/%x-%j.out
#SBATCH --error=/checkpoint/%u/logs/%x-%j.err

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 2:00:00

#SBATCH --mem=256GB
#SBATCH --cpus-per-task=40

#SBATCH --partition=priority
#SBATCH --comment="intern checkout 08/30"

CW_DATA_PATH=/private/home/pengcheng/Research/data/common_crawl
OUTPUT_FILE=/private/home/pengcheng/Research/datasets/table_data/common_crawl.dump.0829.jsonl
FILTER="[0-1][0-9].tar.gz"

echo "Slurm Job ID: "$SLURM_JOB_ID >${OUTPUT_FILE}.err

PYTHON=/private/home/"$USER"/.conda/envs/table_bert/bin/python

$PYTHON \
    -m data.common_crawl \
    --worker_num 40 \
    --input_file ${CW_DATA_PATH} \
    --filter ${FILTER} \
    --output_file ${OUTPUT_FILE} 2>${OUTPUT_FILE}.err
#!/bin/bash

#SBATCH --job-name=data_preprocessing

### Logging
#SBATCH --output=/checkpoint/%u/logs/%x-%j.out
#SBATCH --error=/checkpoint/%u/logs/%x-%j.err

### Node info
#SBATCH --partition=learnfair
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time 24:00:00

#SBATCH --mem=256GB
#SBATCH --cpus-per-task=80

#SBATCH --partition=priority
#SBATCH --comment="intern checkout 08/30"

JAVA_PATH=/private/home/pengcheng/Apps/jdk-12.0.1/bin/
CLASSPATH=dependency/tableBERT-1.0-SNAPSHOT-jar-with-dependencies.jar

WIKI_DUMP=/private/home/pengcheng/Research/data/wiki/enwiki-20190520-pages-articles-multistream.xml.bz2
OUTPUT_FILE=/private/home/pengcheng/Research/datasets/table_data/wiki.dump.jsonl

echo "Slurm Job ID: "$SLURM_JOB_ID >${OUTPUT_FILE}.err

PYTHON=/private/home/"$USER"/.conda/envs/table_bert/bin/python

CLASSPATH=${CLASSPATH} JAVA_PATH=${JAVA_PATH} $PYTHON \
    -m data.extract_wiki_data \
    --wiki_dump ${WIKI_DUMP} \
    --output_file ${OUTPUT_FILE} 2>${OUTPUT_FILE}.err
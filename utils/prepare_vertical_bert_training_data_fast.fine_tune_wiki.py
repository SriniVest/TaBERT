import multiprocessing
import os, sys
import subprocess
import time
import traceback
from argparse import ArgumentParser, Namespace
import logging
from multiprocessing import connection
from typing import List, Iterator, Callable

import h5py
import numpy as np
import json
import ujson
import msgpack
import signal

import gc
import torch
import zmq

from pathlib import Path

from table_bert.vertical.config import VerticalAttentionTableBertConfig
from table_bert.vertical.dataset import serialize_row_data
from table_bert.vertical.input_formatter import VerticalAttentionTableBertInputFormatter
from tqdm import tqdm, trange

from random import shuffle, choice, sample, random

from pytorch_pretrained_bert import *

from table_bert.input_formatter import VanillaTableBertInputFormatter, TableBertBertInputFormatter
from table_bert.config import TableBertConfig
from table_bert.dataset import Example, TableDatabase
from utils.prepare_vertical_bert_training_data_fast import sample_context


def generate_for_epoch(table_db: TableDatabase,
                       indices: List[int],
                       epoch_id: int,
                       epoch_file: Path,
                       input_formatter: TableBertBertInputFormatter,
                       args: Namespace,
                       train=True):
    debug_file = epoch_file.with_suffix('.sample.json') if args.is_master else None
    if debug_file:
        f_dbg = open(debug_file, 'w')

    row_data_sequences = []
    row_data_offsets = []
    mlm_data_sequences = []
    mlm_data_offsets = []

    def _save_shard():
        data = {
            'row_data_sequences': np.uint16(row_data_sequences),
            'row_data_offsets': np.uint64(row_data_offsets),
            'mlm_data_sequences': np.uint16(mlm_data_sequences),
            'mlm_data_offsets': np.uint64(mlm_data_offsets),
        }

        with h5py.File(str(epoch_file), 'w') as f:
            for key, val in data.items():
                f.create_dataset(key, data=val)

        del row_data_sequences[:]
        del row_data_offsets[:]
        del mlm_data_sequences[:]
        del mlm_data_offsets[:]

    for example_idx in tqdm(indices, desc=f"Generating dataset {epoch_file}", file=sys.stdout):
        example = table_db[example_idx]

        if train and epoch_id >= args.use_common_crawl_for_num_epochs and example.source == 'common_crawl':
            continue

        try:
            instances = input_formatter.get_pretraining_instances_from_example(example, sample_context)

            for instance in instances:
                if debug_file and random() <= 0.05:
                    f_dbg.write(json.dumps(instance) + os.linesep)

                input_formatter.remove_unecessary_instance_entries(instance)

                table_data = []
                for row_inst in instance['rows']:
                    row_data = serialize_row_data(row_inst, config=input_formatter.config)
                    table_data.extend(row_data)

                row_data_offsets.append([
                    instance['table_size'][0],  # row_num
                    instance['table_size'][1],  # column_num
                    len(row_data_sequences),  # start index
                    len(row_data_sequences) + len(table_data)  # end index
                ])
                row_data_sequences.extend(table_data)

                s1 = len(mlm_data_sequences)
                mlm_data = []

                mlm_data.extend(instance['masked_context_token_positions'])
                s2 = s1 + len(mlm_data)

                mlm_data.extend(instance['masked_context_token_label_ids'])
                s3 = s1 + len(mlm_data)

                mlm_data.extend(instance['masked_column_token_column_ids'])
                s4 = s1 + len(mlm_data)

                mlm_data.extend(instance['masked_column_token_label_ids'])
                s5 = s1 + len(mlm_data)

                mlm_data_offsets.append([s1, s2, s3, s4, s5])
                mlm_data_sequences.extend(mlm_data)
        except:
            # raise
            typ, value, tb = sys.exc_info()
            print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)
            print(example.serialize(), file=sys.stderr)
            print('*' * 50 + 'Stack Trace' + '*' * 50, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)

            sys.stderr.flush()

    _save_shard()


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of preprocess to pregenerate")
    parser.add_argument('--no_wiki_tables_from_common_crawl', action='store_true', default=False)
    parser.add_argument('--global_rank', type=int, default=os.environ.get('SLURM_PROCID', 0))
    parser.add_argument('--world_size', type=int, default=os.environ.get('SLURM_NTASKS', 1))

    # fine tune arguments
    parser.add_argument('--use_common_crawl_for_num_epochs', type=int, default=4)

    VerticalAttentionTableBertConfig.add_args(parser)

    args = parser.parse_args()
    args.is_master = args.global_rank == 0

    logger = logging.getLogger('DataGenerator')
    handler = logging.StreamHandler(sys.stderr)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    logger.info(f'Rank {args.global_rank} out of {args.world_size}')
    sys.stderr.flush()

    table_bert_config = VerticalAttentionTableBertConfig.from_dict(vars(args))
    tokenizer = BertTokenizer.from_pretrained(table_bert_config.base_model_name)
    input_formatter = VerticalAttentionTableBertInputFormatter(table_bert_config, tokenizer)

    total_tables_num = int(subprocess.check_output(f"wc -l {args.train_corpus}", shell=True).split()[0])
    dev_table_num = min(int(total_tables_num * 0.1), 100000)
    train_table_num = total_tables_num - dev_table_num

    # seed the RNG to make sure each process follows the same spliting
    rng = np.random.RandomState(seed=5783287)

    corpus_table_indices = list(range(total_tables_num))
    rng.shuffle(corpus_table_indices)
    dev_table_indices = corpus_table_indices[:dev_table_num]
    train_table_indices = corpus_table_indices[dev_table_num:]

    local_dev_table_indices = dev_table_indices[args.global_rank::args.world_size]
    local_train_table_indices = train_table_indices[args.global_rank::args.world_size]
    local_indices = local_dev_table_indices + local_train_table_indices

    logger.info(f'total tables: {total_tables_num}')
    logger.debug(f'local dev table indices: {local_dev_table_indices[:1000]}')
    logger.debug(f'local train table indices: {local_train_table_indices[:1000]}')

    with TableDatabase.from_jsonl(args.train_corpus, backend='memory', tokenizer=tokenizer, indices=local_indices) as table_db:
        local_indices = {idx for idx in local_indices if idx in table_db}
        local_dev_table_indices = [idx for idx in local_dev_table_indices if idx in local_indices]
        local_train_table_indices = [idx for idx in local_train_table_indices if idx in local_indices]

        args.output_dir.mkdir(exist_ok=True, parents=True)
        print(f'Num tables to be processed by local worker: {len(table_db)}', file=sys.stdout)

        if args.is_master:
            with (args.output_dir / 'config.json').open('w') as f:
                json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

        (args.output_dir / 'train').mkdir(exist_ok=True)
        (args.output_dir / 'dev').mkdir(exist_ok=True)

        # generate dev data first
        dev_file = args.output_dir / 'dev' / f'epoch_0.shard{args.global_rank}.h5'
        generate_for_epoch(table_db, local_dev_table_indices, 0, dev_file, input_formatter, args, train=False)

        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            gc.collect()
            epoch_filename = args.output_dir / 'train' / f"epoch_{epoch}.shard{args.global_rank}.h5"
            generate_for_epoch(table_db, local_train_table_indices, epoch, epoch_filename, input_formatter, args)


if __name__ == '__main__':
    main()

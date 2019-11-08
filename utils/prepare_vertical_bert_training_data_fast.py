import multiprocessing
import os, sys
import subprocess
import time
import traceback
from argparse import ArgumentParser, Namespace
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
from tqdm import tqdm, trange

from random import shuffle, choice, sample, random

from pytorch_pretrained_bert import *

from table_bert.input_formatter import VanillaTableBertInputFormatter, TableBertBertInputFormatter
from table_bert.config import TableBertConfig
from table_bert.dataset import Example, TableDatabase


def sample_context(example: Example, max_context_length: int, context_sample_strategy: str = 'nearest') -> Iterator:
    if context_sample_strategy == 'nearest':
        selected_context = []

        context_before, context_after = example.context[0], example.context[1]
        context_src = 'before'
        if not context_before:
            context = context_after
            context_src = 'after'
        elif not context_after:
            context = context_before
        elif random() < 0.5:
            context = context_after
            context_src = 'after'
        else:
            context = context_before

        if context_src == 'before':
            for i in reversed(range(0, len(context))):
                sent = context[i]
                selected_context = sent + selected_context

                if len(selected_context) > max_context_length:
                    selected_context = selected_context[-max_context_length:]  # only keep context close to the table
                    break
        elif context_src == 'after':
            for i in range(0, len(context)):
                sent = context[i]
                selected_context = selected_context + sent

                if len(selected_context) > max_context_length:
                    selected_context = selected_context[:max_context_length]  # only keep context close to the table
                    break

        if selected_context:
            yield selected_context
    elif context_sample_strategy == 'concate_and_enumerate':
        # concatenate the context before and after, select a random chunk of text
        all_context = example.context[0] + example.context[1]
        selected_context = []
        for i in range(len(all_context)):
            sent = all_context[i]
            selected_context.extend(sent)
            if len(selected_context) > max_context_length:
                selected_context = selected_context[:max_context_length]

                if selected_context:
                    yield selected_context
                selected_context = []

        if selected_context:
            yield selected_context
    else:
        raise RuntimeError('Unknown context sample strategy')


def generate_for_epoch(table_db: TableDatabase,
                       indices: List[int],
                       epoch_file: Path,
                       input_formatter: TableBertBertInputFormatter,
                       args: Namespace):
    debug_file = epoch_file.with_suffix('.sample.json') if args.is_master else None
    if debug_file:
        f_dbg = open(debug_file, 'w')

    sequences = []
    segment_a_lengths = []
    sequence_offsets = []
    masked_lm_positions = []
    masked_lm_label_ids = []
    masked_lm_offsets = []

    def _save():
        data = {
            'sequences': np.uint16(sequences),
            'segment_a_lengths': np.uint16(segment_a_lengths),
            'sequence_offsets': np.int64(sequence_offsets),
            'masked_lm_positions': np.uint16(masked_lm_positions),
            'masked_lm_label_ids': np.uint16(masked_lm_label_ids),
            'masked_lm_offsets': np.int64(masked_lm_offsets)
        }

        with h5py.File(str(epoch_file), 'w') as f:
            for key, val in data.items():
                f.create_dataset(key, data=val)

        del sequences[:]
        del segment_a_lengths[:]
        del sequence_offsets[:]
        del masked_lm_positions[:]
        del masked_lm_label_ids[:]
        del masked_lm_offsets[:]

    for example_idx in tqdm(indices, desc=f"Generating dataset {epoch_file}", file=sys.stdout):
        example = table_db[example_idx]
        try:
            instances = input_formatter.get_pretraining_instances_from_example(example, sample_context)

            for instance in instances:
                if debug_file and random() <= 0.05:
                    f_dbg.write(json.dumps(instance) + os.linesep)

                input_formatter.remove_unecessary_instance_entries(instance)

                cur_pos = len(sequences)
                sequence_len = len(instance['token_ids'])
                sequences.extend(instance['token_ids'])
                segment_a_lengths.append(instance['segment_a_length'])
                sequence_offsets.append([cur_pos, cur_pos + sequence_len])

                cur_pos = len(masked_lm_positions)
                lm_mask_len = len(instance['masked_lm_positions'])
                masked_lm_positions.extend(instance['masked_lm_positions'])
                masked_lm_label_ids.extend(instance['masked_lm_label_ids'])
                masked_lm_offsets.append([cur_pos, cur_pos + lm_mask_len])
        except:
            # raise
            typ, value, tb = sys.exc_info()
            print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)
            print(example.serialize(), file=sys.stderr)
            print('*' * 50 + 'Stack Trace' + '*' * 50, file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            # print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)

            sys.stderr.flush()

    _save()


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of preprocess to pregenerate")
    parser.add_argument('--no_wiki_tables_from_common_crawl', action='store_true', default=False)
    parser.add_argument('--global_rank', type=int, default=os.environ.get('SLURM_PROCID', 0))
    parser.add_argument('--world_size', type=int, default=os.environ.get('SLURM_NTASKS', 1))

    VerticalAttentionTableBertConfig.add_args(parser)

    args = parser.parse_args()
    args.is_master = args.global_rank == 0

    print(f'Rank {args.global_rank} out of {args.world_size}', file=sys.stderr)
    sys.stderr.flush()

    table_bert_config = TableBertConfig.from_dict(vars(args))
    input_formatter = VanillaTableBertInputFormatter(table_bert_config)
    tokenizer = BertTokenizer.from_pretrained(args.base_model_name, do_lower_case=args.do_lower_case)

    total_tables_num = int(subprocess.check_output(f"wc -l {args.train_corpus}", shell=True).split()[0])
    dev_table_num = min(int(total_tables_num * 0.1), 100000)
    train_table_num = total_tables_num - dev_table_num

    # seed the RNG to make sure each process follows the same spliting
    np.random.seed(1992)

    corpus_table_indices = list(range(total_tables_num))
    np.random.shuffle(corpus_table_indices)
    dev_table_indices = corpus_table_indices[:dev_table_num]
    train_table_indices = corpus_table_indices[dev_table_num:]

    local_dev_table_indices = dev_table_indices[args.global_rank::args.world_size]
    local_train_table_indices = train_table_indices[args.global_rank::args.world_size]
    local_indices = local_dev_table_indices + local_train_table_indices

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
        generate_for_epoch(table_db, local_dev_table_indices, dev_file, input_formatter, args)

        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            gc.collect()
            epoch_filename = args.output_dir / 'train' / f"epoch_{epoch}.shard{args.global_rank}.h5"
            generate_for_epoch(table_db, local_train_table_indices, epoch_filename, input_formatter, args)


if __name__ == '__main__':
    main()

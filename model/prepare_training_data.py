import json
import os, sys
from argparse import ArgumentParser
from typing import List, Dict

import numpy as np

import shelve
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm, trange

from random import shuffle, choice, sample, random

from pytorch_pretrained_bert import *

from model.table import Example


class TableDatabase:
    def __init__(self, reduce_memory=False):
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.document_shelf_filepath = self.working_dir / 'shelf.db'
            self.document_shelf = shelve.open(str(self.document_shelf_filepath),
                                              flag='n', protocol=-1)
            self.documents = None
        else:
            self.documents = []
            self.document_shelf = None
            self.document_shelf_filepath = None
            self.temp_dir = None
        self.table_ids = []
        self.doc_cumsum = None
        self.cumsum_max = None
        self.reduce_memory = reduce_memory

    def add_table(self, table):
        if not table:
            return
        if self.reduce_memory:
            self.document_shelf[table.guid] = table
        else:
            self.documents.append(table)
        self.table_ids.append(table.guid)

    def __len__(self):
        return len(self.table_ids)

    def __getitem__(self, item):
        if self.reduce_memory:
            return self.document_shelf[str(item)]
        else:
            return self.documents[item]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        if self.document_shelf is not None:
            self.document_shelf.close()
        if self.temp_dir is not None:
            self.temp_dir.cleanup()


def create_instance_from_document(doc_database: TableDatabase, example_idx: int,
                                  masked_context_token_prob: float, mask_column_token_prob: float,
                                  max_context_length: int,
                                  max_sequence_length: int,
                                  max_predictions_per_seq: int,
                                  column_delimiter: str,
                                  vocab_list: list) -> List[Dict]:
    example = doc_database[example_idx]
    # Account for [CLS], [SEP], [SEP]

    context = example.context  # assume it's a list of tokenized sentence
    selected_context = []
    for i in reversed(range(0, len(context))):
        sent = context[i]
        selected_context = sent + selected_context

        if len(selected_context) > max_context_length:
            selected_context = selected_context[-max_context_length:]  # only keep context close to the table

    assert len(selected_context) > 0

    tokens_a = ['[CLS]'] + selected_context + ['[SEP]']
    # segment_ids = [0] * len(sequence)
    context_cand_indices = list(range(1, len(tokens_a) - 1))

    tokens_b = []
    column_cand_indices = []

    max_table_token_length = max_sequence_length - len(tokens_a) - 1  # account for ending [SEP]
    col_start_idx = len(tokens_a)
    for col_id, column in enumerate(example.header):
        col_tokens = list(column.name_tokens)
        col_name_indices = list(range(col_start_idx, col_start_idx + len(column.name_tokens)))

        col_tokens += ['('] + [column.type] + [')']
        col_type_idx = col_start_idx + len(column.name_tokens) + 1

        col_tokens += ['('] + column.sample_value_tokens[:5] + [')']
        col_tokens += [column_delimiter]

        _col_cand_indices = col_name_indices + [col_type_idx]

        tokens_b += col_tokens
        column_cand_indices.extend(_col_cand_indices)

        if len(tokens_b) >= max_table_token_length:
            tokens_b = tokens_b[:max_table_token_length]
            column_cand_indices = [idx for idx in column_cand_indices if idx < max_sequence_length - 1]

            break

        col_start_idx += len(col_tokens)

    del tokens_b[-1]  # remove last delimiter
    sequence = tokens_a + tokens_b + ['[SEP]']
    segment_ids = [0] * len(tokens_a) + [1] * len(tokens_b) + [1]

    masked_sequence, masked_lm_positions, masked_lm_labels = create_masked_lm_predictions(sequence, context_cand_indices, column_cand_indices,
                                                                                          masked_context_token_prob, mask_column_token_prob,
                                                                                          max_predictions_per_seq, vocab_list)

    instance = {
        "tokens": masked_sequence,
        "segment_ids": segment_ids,
        "masked_lm_positions": masked_lm_positions,
        "masked_lm_labels": masked_lm_labels
    }

    return [instance]


def create_masked_lm_predictions(tokens, context_indices, column_indices,
                                 masked_context_token_prob, mask_column_token_prob,
                                 max_predictions_per_seq, vocab_list):
    """Creates the predictions for the masked LM objective. This is mostly copied from the Google BERT repo, but
    with several refactors to clean it up and remove a lot of unnecessary variables."""

    # mask `mask_column_token_prob` of tokens in columns
    # mask `masked_lm_prob` of tokens in NL context

    num_column_tokens_to_mask = min(max_predictions_per_seq,
                                    max(2, int(round(len(column_indices) * mask_column_token_prob))))
    max_context_token_to_mask = max_predictions_per_seq - num_column_tokens_to_mask
    num_context_tokens_to_mask = min(max_context_token_to_mask,
                                     max(1, int(round(len(context_indices) * masked_context_token_prob))))

    shuffle(column_indices)
    masked_column_token_indices = sorted(sample(column_indices, num_column_tokens_to_mask))

    if num_context_tokens_to_mask:
        shuffle(context_indices)
        masked_context_token_indices = sorted(sample(context_indices, num_context_tokens_to_mask))
        masked_indices = sorted(masked_context_token_indices + masked_column_token_indices)
    else:
        masked_indices = masked_column_token_indices

    masked_token_labels = []

    for index in masked_indices:
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = "[MASK]"
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_token_labels.append(tokens[index])
        # Once we've saved the true label for that token, we can overwrite it with the masked version
        tokens[index] = masked_token

    return tokens, masked_indices, masked_token_labels


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=True,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual", "bert-base-chinese"])
    parser.add_argument("--do_lower_case", action="store_true")

    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--max_context_len", type=int, default=256)
    parser.add_argument("--masked_context_prob", type=float, default=0.15,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--masked_column_prob", type=float, default=0.20,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--column_delimiter", type=str, default='[SEP]', help='Column delimiter')

    args = parser.parse_args()

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())

    with TableDatabase(reduce_memory=args.reduce_memory) as table_db:
        with args.train_corpus.open() as f:
            for lind_id, line in enumerate(tqdm(f, desc="Loading Dataset", unit=" entries")):
                entry = json.loads(line)
                example = Example.from_dict(entry, tokenizer, suffix=lind_id)

                # TODO: move this to data pre-processing
                if any(len(col.name.split(' ')) > 10 for col in example.header):
                    continue

                table_db.add_table(example)

        args.output_dir.mkdir(exist_ok=True)

        # generate train and dev split
        example_indices = list(range(len(table_db)))
        shuffle(example_indices)
        dev_size = int(len(table_db) * 0.1)
        train_indices = example_indices[:-dev_size]
        dev_indices = example_indices[-dev_size:]

        (args.output_dir / 'train').mkdir()
        (args.output_dir / 'dev').mkdir()

        def _create_instances(_idx):
            return create_instance_from_document(
                        table_db, _idx,
                        max_context_length=args.max_context_len, max_sequence_length=args.max_seq_len,
                        max_predictions_per_seq=args.max_predictions_per_seq,
                        masked_context_token_prob=args.masked_context_prob, mask_column_token_prob=args.masked_column_prob,
                        column_delimiter=args.column_delimiter, vocab_list=vocab_list)

        def _generate_for_epoch(_indices, _epoch_file, _metrics_file):
            num_instances = 0
            with _epoch_file.open('w') as f:
                for example_idx in tqdm(_indices, desc="Document"):
                    doc_instances = _create_instances(example_idx)
                    doc_instances = [json.dumps(instance) for instance in doc_instances]

                    for instance in doc_instances:
                        f.write(instance + '\n')
                        num_instances += 1

            with _metrics_file.open('w') as f:
                metrics = {
                    "num_training_examples": num_instances,
                    "max_seq_len": args.max_seq_len
                }
                f.write(json.dumps(metrics))

        # generate dev data first
        dev_file = args.output_dir / 'dev' / 'epoch_0.json'
        dev_metrics_file = args.output_dir / 'dev' / "epoch_0_metrics.json"
        _generate_for_epoch(dev_indices, dev_file, dev_metrics_file)

        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            epoch_filename = args.output_dir / 'train' / f"epoch_{epoch}.json"
            metrics_file = args.output_dir / 'train' / f"epoch_{epoch}_metrics.json"
            _generate_for_epoch(train_indices, epoch_filename, metrics_file)


if __name__ == '__main__':
    main()

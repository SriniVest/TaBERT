import json
import multiprocessing
import os, sys
import time
import traceback
from argparse import ArgumentParser, Namespace
from multiprocessing import connection
from types import SimpleNamespace
from typing import List, Dict

import gc
import zmq

import numpy as np

import shelve
from pathlib import Path
from tempfile import TemporaryDirectory
from tqdm import tqdm, trange

from random import shuffle, choice, sample, random

from pytorch_pretrained_bert import *

from model.dataset import Example


TRAIN_INSTANCE_QUEUE_ADDRESS = 'tcp://127.0.0.1:5560'
EXAMPLE_QUEUE_ADDRESS = 'tcp://127.0.0.1:5561'


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

    @staticmethod
    def __load_process(file, job_queue, num_workers):
        cnt = 0
        with file.open() as f:
            for line in f:
                job_queue.put(line)
                cnt += 1
                # if cnt % 10000 == 0:
                #     print(f'read {cnt} examples')
                #     sys.stdout.flush()

        for _ in range(num_workers):
            job_queue.put(None)

    @staticmethod
    def __example_worker_process(job_queue, example_queue, tokenizer):
        job = job_queue.get()
        cnt = 0
        while job:
            example = Example.from_dict(json.loads(job), tokenizer, suffix=None)
            data = example.serialize()
            example_queue.put(data)

            job = job_queue.get()

            cnt += 1
            # if cnt % 1000 == 0:
            #     print(f'[__example_worker_process] read {cnt} examples')
            #     sys.stdout.flush()

        example_queue.put(None)

    @classmethod
    def from_jsonl_queue(cls, file_path: Path, tokenizer: BertTokenizer) -> 'TableDatabase':
        file_path = Path(file_path)
        db = cls()

        job_queue = multiprocessing.Queue(maxsize=10000)
        example_queue = multiprocessing.Queue(maxsize=10000)
        num_workers = multiprocessing.cpu_count()

        reader = multiprocessing.Process(target=cls.__load_process, args=(file_path, job_queue, num_workers), daemon=True)
        reader.start()

        workers = []
        for _ in range(num_workers):
            worker = multiprocessing.Process(target=cls.__example_worker_process,
                                             args=(job_queue, example_queue, tokenizer),
                                             daemon=True)
            worker.start()
            workers.append(worker)

        stop_count = 0
        cnt = 0
        with tqdm(desc="Loading Dataset", unit=" entries", file=sys.stdout) as pbar:
            while True:
                data = example_queue.get()

                if data is None:
                    stop_count += 1
                    if stop_count == num_workers: break
                    else: continue
                else: pbar.update(1)

                example = Example.from_serialized(data)
                # TODO: move this to data pre-processing
                if any(len(col.name.split(' ')) > 10 for col in example.header):
                    continue

                db.add_table(example)
                cnt += 1
                if cnt % 10000 == 0:
                    print(f'load {cnt} examples')
                    sys.stdout.flush()

                # print(f'[Main] {example_queue.qsize()}')
                sys.stdout.flush()

        reader.join()
        for worker in workers:
            worker.join()

        return db

    @staticmethod
    def __load_process_zmq(file, num_workers):
        context = zmq.Context()
        job_sender = context.socket(zmq.PUSH)
        job_sender.setsockopt(zmq.LINGER, -1)
        job_sender.bind("tcp://127.0.0.1:5557")

        cnt = 0
        with file.open() as f:
            for line in f:
                job_sender.send_string(line)
                # if cnt % 10000 == 0:
                #     print(f'read {cnt} examples')
                #     sys.stdout.flush()

        for _ in range(num_workers):
            job_sender.send_string('')

        time.sleep(600)

    @staticmethod
    def __example_worker_process_zmq(tokenizer):
        context = zmq.Context()
        job_receiver = context.socket(zmq.PULL)
        job_receiver.setsockopt(zmq.LINGER, -1)
        job_receiver.connect("tcp://127.0.0.1:5557")

        example_sender = context.socket(zmq.PUSH)
        example_sender.setsockopt(zmq.LINGER, -1)
        example_sender.connect("tcp://127.0.0.1:5558")

        cnt = 0
        while True:
            job = job_receiver.recv_string()
            if job:
                example = Example.from_dict(json.loads(job), tokenizer, suffix=None)
                data = example.serialize()
                example_sender.send_pyobj(data)
            else:
                example_sender.send_pyobj(None)

            cnt += 1
            # if cnt % 1000 == 0:
            #     print(f'[__example_worker_process] read {cnt} examples')
            #     sys.stdout.flush()

    @classmethod
    def from_jsonl(cls, file_path: Path, tokenizer: BertTokenizer) -> 'TableDatabase':
        file_path = Path(file_path)
        db = cls()

        num_workers = multiprocessing.cpu_count()

        reader = multiprocessing.Process(target=cls.__load_process_zmq, args=(file_path, num_workers),
                                         daemon=True)

        context = zmq.Context()
        example_receiver = context.socket(zmq.PULL)
        example_receiver.setsockopt(zmq.LINGER, -1)
        example_receiver.bind("tcp://*:5558")

        workers = []
        for _ in range(num_workers):
            worker = multiprocessing.Process(target=cls.__example_worker_process_zmq,
                                             args=(tokenizer,),
                                             daemon=True)
            worker.start()
            workers.append(worker)

        reader.start()

        stop_count = 0
        with tqdm(desc="Loading Dataset", unit=" entries", file=sys.stdout) as pbar:
            while True:
                data = example_receiver.recv_pyobj()

                if data is None:
                    stop_count += 1
                    print(f'{stop_count} worker stoped!')
                    if stop_count == num_workers:
                        break
                    else:
                        continue
                else:
                    pbar.update(1)

                example = Example.from_serialized(data)
                # TODO: move this to data pre-processing
                if any(len(col.name.split(' ')) > 10 for col in example.header):
                    continue

                db.add_table(example)

                # print(f'[Main] {example_queue.qsize()}')
                sys.stdout.flush()

        # reader.join()
        # for worker in workers:
        #     worker.join()
        reader.terminate()
        for worker in workers:
            worker.terminate()

        return db

    def add_table(self, table):
        if not table:
            return
        if self.reduce_memory:
            self.document_shelf[table.guid] = table
        else:
            self.documents.append(table)
        self.table_ids.append(table.uuid)

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


def create_training_instances_from_example(example: Example,
                                           masked_context_token_prob: float, mask_column_token_prob: float,
                                           max_context_length: int,
                                           max_sequence_length: int,
                                           max_predictions_per_seq: int,
                                           column_delimiter: str,
                                           tokenizer: BertTokenizer,
                                           vocab_list: list) -> List[Dict]:
    # Account for [CLS], [SEP], [SEP]

    context_before, context_after = example.context[0], example.context[1]
    if not context_before:
        context = context_after[::-1]
    elif not context_after:
        context = context_before
    elif random() < 0.5:
        context = context_after[::-1]
    else:
        context = context_before

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

        col_values = example.column_data[col_id]
        # print(col_values)
        col_values = [val for val in col_values if val is not None and len(val) > 0]
        sampled_value = choice(col_values)
        # print('chosen value', sampled_value)
        sampled_value_tokens = tokenizer.tokenize(sampled_value)

        col_tokens += ['('] + sampled_value_tokens[:5] + [')']
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
        "masked_lm_labels": masked_lm_labels,
        "source": example.source
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


def generate_train_instance_from_example(args: Namespace):
    context = zmq.Context()
    example_receiver = context.socket(zmq.PULL)
    # example_receiver.setsockopt(zmq.LINGER, -1)
    example_receiver.connect(EXAMPLE_QUEUE_ADDRESS)

    instance_sender = context.socket(zmq.PUSH)
    # instance_sender.setsockopt(zmq.LINGER, -1)
    instance_sender.connect(TRAIN_INSTANCE_QUEUE_ADDRESS)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    vocab_list = list(tokenizer.vocab.keys())

    # print('started queues')

    while True:
        job = example_receiver.recv_pyobj()
        if job:
            example = Example.from_serialized(job)
            # print('get one example')
            try:
                instances = create_training_instances_from_example(
                    example,
                    max_context_length=args.max_context_len,
                    max_sequence_length=args.max_seq_len,
                    max_predictions_per_seq=args.max_predictions_per_seq,
                    masked_context_token_prob=args.masked_context_prob,
                    mask_column_token_prob=args.masked_column_prob,
                    column_delimiter=args.column_delimiter,
                    tokenizer=tokenizer,
                    vocab_list=vocab_list
                )
                for instance in instances:
                    instance_sender.send_pyobj(json.dumps(instance))
            except:
                typ, value, tb = sys.exc_info()
                print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)
                print(example.serialize(), file=sys.stderr)
                print('*' * 50 + 'Stack Trace' + '*' * 50, file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                print('*' * 50 + 'Exception' + '*' * 50, file=sys.stderr)

                sys.stderr.flush()
        else:
            instance_sender.send_pyobj(None)

    while True:
        time.sleep(10)


def write_instance_to_file(output_file: Path, num_workers: int, stat_send: connection.Connection):
    context = zmq.Context()
    instance_receiver = context.socket(zmq.PULL)
    # instance_receiver.setsockopt(zmq.LINGER, -1)
    instance_receiver.bind(TRAIN_INSTANCE_QUEUE_ADDRESS)

    finished_worker_num = 0
    num_instances = 0
    with output_file.open('w') as f:
        while True:
            data = instance_receiver.recv_pyobj()
            if data is not None:
                f.write(data + os.linesep)
                num_instances += 1
                # print('write one example')
            else:
                # print('one worker finished')
                finished_worker_num += 1
                if finished_worker_num == num_workers:
                    break

    stat_send.send(num_instances)
    instance_receiver.close()


def generate_for_epoch(table_db: TableDatabase,
                       indices: List[int],
                       epoch_file: Path,
                       metrics_file: Path,
                       args: Namespace):
    print(f'Generating {epoch_file}', file=sys.stderr)

    # initialize job sender
    context = zmq.Context()
    example_sender = context.socket(zmq.PUSH)
    # example_sender.setsockopt(zmq.LINGER, -1)
    example_sender.bind(EXAMPLE_QUEUE_ADDRESS)

    stat_recv, stat_send = multiprocessing.Pipe()
    num_workers = multiprocessing.cpu_count() - 2

    instance_writer_process = multiprocessing.Process(target=write_instance_to_file,
                                                      args=(epoch_file, num_workers, stat_send))
    instance_writer_process.start()

    workers = []
    for _ in range(num_workers):
        worker_process = multiprocessing.Process(target=generate_train_instance_from_example,
                                                 args=(args, ),
                                                 daemon=True)
        worker_process.start()
        workers.append(worker_process)

    for example_idx in tqdm(indices, desc="Document", file=sys.stdout):
        example = table_db[example_idx]
        example_sender.send_pyobj(example.serialize())
        # print('send one example')

    for _ in range(num_workers):
        example_sender.send_pyobj(None)

    num_instances = stat_recv.recv()
    print('num instanances:', num_instances)
    instance_writer_process.join()

    for worker in workers:
        worker.terminate()

    with metrics_file.open('w') as f:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": args.max_seq_len
        }
        f.write(json.dumps(metrics))

    example_sender.close()


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

    parser.add_argument('--no_wiki_tables_from_common_crawl', action='store_true', default=False)

    args = parser.parse_args()

    global tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    with TableDatabase.from_jsonl(args.train_corpus, tokenizer=tokenizer) as table_db:
        args.output_dir.mkdir(exist_ok=True, parents=True)

        # generate train and dev split
        example_indices = list(range(len(table_db)))
        shuffle(example_indices)
        dev_size = min(int(len(table_db) * 0.1), 500000)
        train_indices = example_indices[:-dev_size]
        dev_indices = example_indices[-dev_size:]

        (args.output_dir / 'train').mkdir(exist_ok=True)
        (args.output_dir / 'dev').mkdir(exist_ok=True)

        # generate dev data first
        dev_file = args.output_dir / 'dev' / 'epoch_0.json'
        dev_metrics_file = args.output_dir / 'dev' / "epoch_0_metrics.json"
        generate_for_epoch(table_db, dev_indices, dev_file, dev_metrics_file, args)

        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            gc.collect()
            epoch_filename = args.output_dir / 'train' / f"epoch_{epoch}.json"
            metrics_file = args.output_dir / 'train' / f"epoch_{epoch}_metrics.json"
            generate_for_epoch(table_db, train_indices, epoch_filename, metrics_file, args)


if __name__ == '__main__':
    main()

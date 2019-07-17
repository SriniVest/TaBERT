import json
import logging
import math
import multiprocessing
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Iterator

import numpy as np
import redis
import torch
import zmq
from pytorch_pretrained_bert import BertTokenizer
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
from tqdm import tqdm


class DistributedSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        indices = torch.randperm(len(self.dataset), generator=g).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class TableDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, num_data_epochs, reduce_memory=False, multi_gpu=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.data_epoch = self.epoch = epoch
        # self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        examples = []

        indices = []
        if multi_gpu:
            logging.info(f'Load a sub-sample of the whole dataset')
            num_shards = torch.distributed.get_world_size()
            local_shard_id = torch.distributed.get_rank()

            dataset_size = int(subprocess.check_output(f"/usr/bin/wc -l {data_file}", shell=True).split()[0])
            shard_size = dataset_size // num_shards

            logging.info(f'dataset_size={dataset_size}, shard_size={shard_size}')

            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(dataset_size, generator=g).tolist()

            # make it evenly divisible
            indices = indices[:shard_size * num_shards]
            assert len(indices) == shard_size * num_shards

            # subsample
            indices = indices[local_shard_id:len(indices):num_shards]
            assert len(indices) == shard_size

            indices = set(indices)

        logging.info(f"Loading examples from {training_path} for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples", file=sys.stdout)):
                if (not multi_gpu) or (multi_gpu and i in indices):
                    line = line.strip()
                    example = json.loads(line)
                    examples.append(example)

        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        return self.examples[item]

    @staticmethod
    def collate(examples, tokenizer):
        batch_size = len(examples)
        max_len = max(len(e['tokens']) for e in examples)

        input_array = np.zeros((batch_size, max_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_len), dtype=np.bool)
        lm_label_array = np.full((batch_size, max_len), dtype=np.int, fill_value=-1)

        for e_id, example in enumerate(examples):
            token_ids = tokenizer.convert_tokens_to_ids(example['tokens'])
            masked_label_ids = tokenizer.convert_tokens_to_ids(example['masked_lm_labels'])
            segment_ids = example['segment_ids']
            masked_lm_positions = example['masked_lm_positions']

            input_array[e_id, :len(token_ids)] = token_ids
            mask_array[e_id, :len(token_ids)] = 1
            segment_array[e_id, :len(segment_ids)] = segment_ids
            lm_label_array[e_id, masked_lm_positions] = masked_label_ids

        return (torch.tensor(input_array.astype(np.int64)),
                torch.tensor(mask_array.astype(np.int64)),
                torch.tensor(segment_array.astype(np.int64)),
                torch.tensor(lm_label_array.astype(np.int64)))


class Column(object):
    def __init__(self, name, type, sample_value=None, **kwargs):
        self.name = name
        self.type = type
        self.sample_value = sample_value

        self.fields = []
        for key, val in kwargs.items():
            self.fields.append(key)
            setattr(self, key, val)

    def to_dict(self):
        data = {
            'name': self.name,
            'type': self.type,
            'sample_value': self.sample_value,
        }

        for key in self.fields:
            data[key] = getattr(self, key)

        return data


class Example(object):
    def __init__(self, uuid, header, context, column_data=None, **kwargs):
        self.uuid = uuid
        self.header = header
        self.context = context
        self.column_data = column_data

        for key, val in kwargs.items():
            setattr(self, key, val)

    def serialize(self):
        example = {
            'uuid': self.uuid,
            'source': self.source,
            'context': self.context,
            'column_data': self.column_data,
            'header': [x.to_dict() for x in self.header]
        }

        return example

    @classmethod
    def from_serialized(cls, data) -> 'Example':
        header = [Column(**x) for x in data['header']]
        data['header'] = header
        return Example(**data)

    @classmethod
    def from_dict(cls, entry: Dict, tokenizer: Optional[BertTokenizer], suffix) -> 'Example':
        def _get_data_source():
            return 'common_crawl' if 'context_before' in entry else 'wiki'

        source = _get_data_source()

        header_entry = entry['header'] if source == 'wiki' else entry['table']['header']
        header = []
        column_data = []
        for col in header_entry:
            sample_value = col['sample_value']['value']
            if tokenizer:
                name_tokens = tokenizer.tokenize(col['name'])
            else: name_tokens = None
            column = Column(col['name'],
                            col['type'],
                            sample_value,
                            name_tokens=name_tokens)
            header.append(column)

        if source == 'wiki':
            for row in entry['data'][1:]:
                for col_id, (tag, cell_val) in enumerate(row):
                    if col_id >= len(column_data):
                        column_data.append([])

                    column_data[col_id].append(cell_val)
        else:
            for row in entry['table']['rows']:
                for col_id, (cell_val) in enumerate(row):
                    if col_id >= len(column_data):
                        column_data.append([])

                    column_data[col_id].append(cell_val)

        context_before = []
        context_after = []

        if source == 'wiki':
            for para in entry['context']:
                for sent in para:
                    if tokenizer:
                        sent = tokenizer.tokenize(sent)

                    context_before.append(sent)

            caption = entry['caption']
            if caption:
                if tokenizer:
                    caption = tokenizer.tokenize(entry['caption'])

                context_before.append(caption)

            uuid = f"{source}_{entry['id']}_{'_'.join(entry['title'])}"
        else:
            for sent in entry['context_before']:
                if tokenizer:
                    sent = tokenizer.tokenize(sent)
                context_before.append(sent)

            for sent in entry['context_after']:
                if tokenizer:
                    sent = tokenizer.tokenize(sent)
                context_after.append(sent)

            uuid = entry['uuid']

        return cls(uuid, header,
                   [context_before, context_after],
                   column_data=column_data,
                   source=source)


class TableDatabase:
    def __init__(self):
        self.restore_client()
        self.client.flushall(asynchronous=False)
        self._cur_index = multiprocessing.Value('i', 0)

    def restore_client(self):
        self.client = redis.Redis(host='localhost', port=6379, db=0)

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

        while True:
            job_sender.send_string('')
            time.sleep(0.1)

    @staticmethod
    def __example_worker_process_zmq(tokenizer, db):
        context = zmq.Context()
        job_receiver = context.socket(zmq.PULL)
        job_receiver.setsockopt(zmq.LINGER, -1)
        job_receiver.connect("tcp://127.0.0.1:5557")

        cache_client = redis.Redis(host='localhost', port=6379, db=0)
        buffer_size = 20000

        def _add_to_cache():
            if buffer:
                with db._cur_index.get_lock():
                    index_end = db._cur_index.value + len(buffer)
                    db._cur_index.value = index_end
                index_start = index_end - len(buffer)
                values = {str(i): val for i, val in zip(range(index_start, index_end), buffer)}
                cache_client.mset(values)
                del buffer[:]

        cnt = 0
        buffer = []
        while True:
            job = job_receiver.recv_string()
            if job:
                cnt += 1
                example = Example.from_dict(json.loads(job), tokenizer, suffix=None)

                # TODO: move this to data pre-processing
                if any(len(col.name.split(' ')) > 10 for col in example.header):
                    continue

                data = example.serialize()
                buffer.append(json.dumps(data))

                if len(buffer) >= buffer_size:
                    _add_to_cache()
            else:
                job_receiver.close()
                _add_to_cache()
                break

            cnt += 1

    @classmethod
    def from_jsonl(cls, file_path: Path, tokenizer: Optional[BertTokenizer] = None) -> 'TableDatabase':
        file_path = Path(file_path)
        db = cls()
        num_workers = multiprocessing.cpu_count() - 5

        reader = multiprocessing.Process(target=cls.__load_process_zmq, args=(file_path, num_workers),
                                         daemon=True)

        workers = []
        for _ in range(num_workers):
            worker = multiprocessing.Process(target=cls.__example_worker_process_zmq,
                                             args=(tokenizer, db),
                                             daemon=True)
            worker.start()
            workers.append(worker)

        reader.start()

        stop_count = 0
        db_size = 0
        with tqdm(desc="Loading Dataset", unit=" entries", file=sys.stdout) as pbar:
            while True:
                cur_db_size = len(db)
                pbar.update(cur_db_size - db_size)
                db_size = cur_db_size

                all_worker_finished = all(not w.is_alive() for w in workers)
                if all_worker_finished:
                    print(f'all workers stoped!')
                    break

                time.sleep(5)

        for worker in workers:
            worker.join()
        reader.terminate()

        return db

    def __len__(self):
        return self._cur_index.value

    def __getitem__(self, item) -> Example:
        result = self.client.get(str(item))
        if result is None:
            raise IndexError(item)

        example = Example.from_serialized(json.loads(result))

        return example

    def __iter__(self) -> Iterator[Example]:
        for i in range(len(self)):
            yield self[i]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, traceback):
        print('Flushing all entries in cache')
        self.client.flushall()

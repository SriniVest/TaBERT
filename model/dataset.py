import json
import logging
import math
import subprocess
import sys
from collections import OrderedDict
from typing import Dict

import numpy as np
import torch
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
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
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

        for key, val in kwargs.items():
            setattr(self, key, val)


class Example(object):
    def __init__(self, guid, header, context, data=None):
        self.guid = guid
        self.header = header
        self.context = context
        self.data = data

    @classmethod
    def from_dict(cls, entry: Dict, tokenizer, suffix) -> 'Example':
        header = []
        data = OrderedDict()
        for col_data in entry['header']:
            sample_value = col_data['sample_value']['value']
            column = Column(col_data['name'],
                            col_data['type'],
                            sample_value,
                            name_tokens=tokenizer.tokenize(col_data['name']),
                            type_tokens=tokenizer.tokenize(col_data['type']),
                            sample_value_tokens=tokenizer.tokenize(sample_value))
            header.append(column)

        for row in entry['data'][1:]:
            for col_id, (tag, cell_val) in enumerate(row):
                col_name = header[col_id].name
                data.setdefault(col_name, []).append(cell_val)

        context = []
        for para in entry['context']:
            for sent in para:
                tokenized_sent = tokenizer.tokenize(sent)
                context.append(tokenized_sent)

        if entry['caption']:
            caption = tokenizer.tokenize(entry['caption'])
            context.append(caption)

        guid = f"{entry['id']}_{'_'.join(entry['title'])}_{suffix}"

        return cls(guid, header, context, data=data)
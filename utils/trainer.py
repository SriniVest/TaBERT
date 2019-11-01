import contextlib
import sys
from argparse import Namespace
from itertools import chain
from typing import Dict
import logging

import torch
import torch.nn as nn

from fairseq import optim, distributed_utils
from fairseq.optim import lr_scheduler


class Trainer(object):
    def __init__(self, model: nn.Module, args: Namespace):
        self.model = model
        self.args = args
        self._num_updates = 0
        self.cuda = not self.args.cpu and torch.cuda.is_available()
        self.logger = logging.getLogger()

        if args.fp16:
            self.model = self.model.half()

        self.build_optimizer()

    def build_optimizer(self):
        params = list(
            filter(
                lambda p: p.requires_grad,
                self.model.parameters(),
            )
        )

        if self.args.fp16:
            if self.cuda and torch.cuda.get_device_capability(0)[0] < 7:
                print('| WARNING: your device does NOT support faster training with --fp16, '
                      'please switch to FP32 which is likely to be faster')
            if self.args.memory_efficient_fp16:
                self.optimizer = optim.MemoryEfficientFP16Optimizer.build_optimizer(self.args, params)
            else:
                self.optimizer = optim.FP16Optimizer.build_optimizer(self.args, params)
        else:
            if self.cuda and torch.cuda.get_device_capability(0)[0] >= 7:
                print('| NOTICE: your device may support faster training with --fp16')
            self.optimizer = optim.build_optimizer(self.args, params)

        self.lr_scheduler = lr_scheduler.build_lr_scheduler(self.args, self.optimizer)
        self.lr_scheduler.step_update(0)

    def prepare_sample(self, sample: Dict):
        def _apply_func(x):
            if torch.is_tensor(x):
                if self.cuda:
                    x = x.cuda()
                if self.args.fp16 and x.dtype is torch.float32:
                    x = x.half()

            return x

        return {
            key: _apply_func(val)
            for key, val
            in sample.items()
        }

    def train_step(self, samples):
        self.optimizer.zero_grad()
        logging_outputs = []

        for i, sample in enumerate(samples):
            sample = self.prepare_sample(sample)

            def maybe_no_sync():
                """
                Whenever *samples* contains more than one mini-batch, we
                want to accumulate gradients locally and only call
                all-reduce in the last backwards pass.
                """
                if (
                    self.args.world_size > 1
                    and hasattr(self.model, 'no_sync')
                    and i < len(samples) - 1
                ):
                    return self.model.no_sync()
                else:
                    return contextlib.ExitStack()  # dummy contextmanager

            with maybe_no_sync():
                # forward and backward
                loss = self.model(**sample)
                logging_output = {'sample_size': sample['sample_size']}
                self.optimizer.backward(loss)

                logging_outputs.append(logging_output)

        # gather logging outputs from all replicas
        if self.args.world_size > 1:
            logging_outputs = distributed_utils.all_gather_list(logging_outputs)
            logging_outputs = list(chain.from_iterable(logging_outputs))

        sample_size = sum(x['sample_size'] for x in logging_outputs)
        self.optimizer.multiply_grads(self.args.world_size / sample_size)

        # clip grads
        if self.args.clip_norm > 0.:
            grad_norm = self.optimizer.clip_grad_norm(self.args.clip_norm)

        # take an optimization step
        self.optimizer.step()
        self.take_one_step()

    def take_one_step(self):
        self._num_updates += 1
        self.lr_scheduler.step_update(self._num_updates)
        if self._num_updates >= self.lr_scheduler.total_num_update:
            logging.warning('Reached max num of updates')
            # exit(0)

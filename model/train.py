import re
import shutil
import sys

import torch
import logging
import random

import torch.nn as nn
import torch.distributed as dist
from fairseq.data import GroupedIterator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import json
import numpy as np
from functools import partial

from model.comm import init_distributed_mode
from model.dataset import TableDataset
from model.evaluator import Evaluator
from model.trainer import Trainer
from model.util import parse_arg, init_logger


def main():
    args = parse_arg()

    init_distributed_mode(args)
    logger = init_logger(args)

    train_data_dir = args.data_dir / 'train'
    dev_data_dir = args.data_dir / 'dev'

    if args.is_master:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        with (args.output_dir / 'config.json').open('w') as f:
            json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

        # copy the table bert config file to the working directory
        shutil.copy(args.data_dir / 'config.json', args.output_dir / 'tb_config.json')

    assert args.data_dir.is_dir(), \
        "--data_dir should point to the folder of files made by pregenerate_training_data.py!"

    epoch_stat_files = list(train_data_dir.glob('epoch_*.metrics.json'))
    epoch_ids = [
        int(re.search(r'epoch_(\d+).metrics.json', str(f)).group(1))
        for f in epoch_stat_files
    ]
    max_epoch = max(epoch_ids)

    if args.cpu:
        device = torch.device('cpu')
    else:
        device = torch.cuda.current_device()

    logger.info("device: {} gpu_id: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.local_rank, bool(args.multi_gpu), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    real_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not args.cpu:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logger.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    # Prepare model
    if args.no_init:
        assert args.config_file is not None
        model = BertForMaskedLM(BertConfig.from_json_file(args.config_file))
    else:
        model = BertForMaskedLM.from_pretrained(args.bert_model)

    if args.fp16:
        model = model.half()

    model = model.to(device)
    if args.multi_gpu:
        if args.ddp_backend == 'pytorch':
            model = nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
                device_ids=[args.local_rank], output_device=args.local_rank
            )
        else:
            import apex
            model = apex.parallel.DistributedDataParallel(model, delay_allreduce=True)

    trainer = Trainer(model, args)

    logger.info("***** Running training *****")
    logger.info(f"  Current config: {args}")

    model.train()

    # we also partitation the dev set for every local process
    dev_set = TableDataset(epoch=0, training_path=dev_data_dir, tokenizer=tokenizer,
                           reduce_memory=args.reduce_memory, multi_gpu=args.multi_gpu)

    evaluator = Evaluator(batch_size=args.train_batch_size * 4, args=args)

    for epoch in range(max_epoch + 1):  # inclusive
        epoch_dataset = TableDataset(epoch=epoch, training_path=train_data_dir, tokenizer=tokenizer,
                                     reduce_memory=args.reduce_memory,
                                     multi_gpu=args.multi_gpu)

        train_sampler = RandomSampler(epoch_dataset)

        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=real_batch_size,
                                      num_workers=0,
                                      collate_fn=epoch_dataset.collate)

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", file=sys.stdout,
                  disable=not args.is_master) as pbar:
            samples_iter = GroupedIterator(iter(train_dataloader), args.gradient_accumulation_steps)

            for step, samples in enumerate(samples_iter):
                trainer.train_step(samples)

                pbar.update(len(samples))
                # pbar.set_postfix_str(', '.join(f"{k} {v}" for k, v in trainer.stat()))

            logger.info(f'Epoch {epoch} finished.')

            if args.is_master:
                # Save a trained model
                logger.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = args.output_dir / f"pytorch_model_epoch{epoch:02d}.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))

            # perform validation
            logger.info("** ** * Perform validation ** ** * ")
            dev_results = evaluator.evaluate(model.module if args.multi_gpu else model, dev_set)

            if args.is_master:
                logger.info('** ** * Validation Results ** ** * ')
                logger.info(f'Epoch {epoch} Validation Results: {dev_results}')

            # flush logging information to disk
            sys.stderr.flush()


if __name__ == '__main__':
    main()

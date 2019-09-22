import re
import sys

import torch
import random

import torch.nn as nn
from fairseq.data import GroupedIterator
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

import json
import numpy as np

from model.vanilla_table_bert import VanillaTableBert
from utils.comm import init_distributed_mode
from utils.config import TableBertConfig
from utils.dataset import TableDataset
from utils.evaluator import Evaluator
from utils.trainer import Trainer
from utils.util import parse_arg, init_logger


def main():
    args = parse_arg()

    init_distributed_mode(args)
    logger = init_logger(args)

    train_data_dir = args.data_dir / 'train'
    dev_data_dir = args.data_dir / 'dev'
    table_bert_config = TableBertConfig.from_file(
        args.data_dir / 'config.json', base_model_name=args.base_model_name)

    if args.is_master:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        with (args.output_dir / 'train_config.json').open('w') as f:
            json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

        # copy the table bert config file to the working directory
        # shutil.copy(args.data_dir / 'config.json', args.output_dir / 'tb_config.json')
        # save table BERT config
        table_bert_config.save(args.data_dir / 'tb_config.json')

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

    # Prepare model
    if args.no_init:
        raise NotImplementedError
    else:
        model = VanillaTableBert(table_bert_config)

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

        model_ptr = model.module
    else:
        model_ptr = model

    trainer = Trainer(model, args)

    logger.info("***** Running training *****")
    logger.info(f"  Current config: {args}")

    model.train()

    # we also partitation the dev set for every local process
    dev_set = TableDataset(epoch=0, training_path=dev_data_dir, tokenizer=model_ptr.tokenizer,
                           multi_gpu=args.multi_gpu)

    evaluator = Evaluator(batch_size=args.train_batch_size * 4, args=args)

    for epoch in range(max_epoch + 1):  # inclusive
        epoch_dataset = TableDataset(epoch=epoch, training_path=train_data_dir, tokenizer=model_ptr.tokenizer,
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
                model_to_save = model_ptr  # Only save the model it-self
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

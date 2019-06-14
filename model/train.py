import sys

import torch
import logging
import random

import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm

from pytorch_pretrained_bert.modeling import BertForMaskedLM
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam

import json
import numpy as np
from functools import partial

from model.comm import init_distributed_mode
from model.dataset import TableDataset
from model.evaluator import Evaluator
from model.util import parse_arg


logging.getLogger().setLevel(logging.DEBUG)


def main():
    args = parse_arg()

    with (args.output_dir / 'config.json').open('w') as f:
        json.dump(vars(args), f, indent=2, sort_keys=True, default=str)

    init_distributed_mode(args)

    assert args.train_data.is_dir(), \
        "--train_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.train_data / f"epoch_{i}.json"
        metrics_file = args.train_data / f"epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            print(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            print("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    device = torch.cuda.current_device()
    logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, args.local_rank, bool(args.multi_gpu), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.multi_gpu:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model = model.to(device)
    if args.multi_gpu:
        model = nn.parallel.DistributedDataParallel(model,
                                                    find_unused_parameters=True,
                                                    device_ids=[args.local_rank], output_device=args.local_rank)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=args.learning_rate,
                         warmup=args.warmup_proportion,
                         t_total=num_train_optimization_steps)

    global_step = 0

    # setup logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[Node {args.node_id} | Rank {args.global_rank} | %(asctime)s] %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)

    logging.info("***** Running training *****")
    logging.info(f"  Num examples = {total_train_examples}")
    logging.info("  Batch size = %d", args.train_batch_size)
    logging.info("  Num steps = %d", num_train_optimization_steps)
    model.train()

    # we also partitation the dev set for every local process
    dev_set = TableDataset(epoch=0, training_path=args.dev_data, tokenizer=tokenizer, num_data_epochs=1,
                           reduce_memory=args.reduce_memory, multi_gpu=args.multi_gpu)

    evaluator = Evaluator(batch_size=args.train_batch_size * 4, args=args)

    for epoch in range(args.epochs):
        epoch_dataset = TableDataset(epoch=epoch, training_path=args.train_data, tokenizer=tokenizer,
                                     num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory,
                                     multi_gpu=args.multi_gpu)

        train_sampler = RandomSampler(epoch_dataset)

        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=0,
                                      collate_fn=partial(epoch_dataset.collate, tokenizer=tokenizer))
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}", file=sys.stdout) as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, segment_ids, lm_label_ids = batch
                loss = model(input_ids, segment_ids, input_mask, lm_label_ids)

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            logging.info(f'Epoch {epoch} finished.')

            if args.is_master:
                # Save a trained model
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                output_model_file = args.output_dir / f"pytorch_model_epoch{epoch:02d}.bin"
                torch.save(model_to_save.state_dict(), str(output_model_file))

            # perform validation
            logging.info("** ** * Perform validation ** ** * ")
            dev_results = evaluator.evaluate(model.module if args.multi_gpu else model, dev_set)

            if args.is_master:
                logging.info('** ** * Validation Results ** ** * ')
                logging.info(f'Epoch {epoch} Validation Results: {dev_results}')

            # flush logging information to disk
            sys.stderr.flush()

            # if args.multi_gpu:
            #     # let other processes wait for the main process to finish validation
            #     logging.info("Syncing over all processes")
            #     torch.distributed.barrier()
            #     logging.info("Syncing over all processes done!")


if __name__ == '__main__':
    main()

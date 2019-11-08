import logging
from argparse import ArgumentParser
from pathlib import Path
import socket

import torch

from fairseq.options import eval_str_list
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule


def parse_arg():
    parser = ArgumentParser()
    parser.add_argument('--task',
                        type=str,
                        default='vanilla',
                        choices=['vanilla', 'vertical_attention'])
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument("--cpu",
                        action='store_true',
                        help="Whether not to use CUDA when available")

    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)

    parser.add_argument("--base-model-name", type=str, required=False,
                        help="Bert pre-trained table_bert selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
                        default='bert-base-uncased')
    parser.add_argument('--no-init', action='store_true', default=False)
    # parser.add_argument('--config-file', type=Path, help='table_bert config file if do not use pre-trained BERT table_bert.')

    # distributed training
    parser.add_argument("--ddp-backend", type=str, default='pytorch', choices=['pytorch', 'apex'])
    parser.add_argument("--local_rank", "--local-rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--master-port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")
    parser.add_argument("--debug-slurm", action='store_true',
                        help="Debug multi-GPU / multi-node within a SLURM job")

    # training details
    parser.add_argument("--train-batch-size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--max-epoch", default=-1, type=int)
    # parser.add_argument("--total-num-update", type=int, default=1000000, help="Number of steps to train for")
    parser.add_argument('--gradient-accumulation-steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--lr-scheduler", type=str, default='polynomial_decay', help='Learning rate scheduler')
    parser.add_argument("--optimizer", type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--lr', '--learning-rate', default='0.00005', type=eval_str_list,
                        metavar='LR_1,LR_2,...,LR_N',
                        help='learning rate for the first N epochs; all epochs >N using LR_N'
                             ' (note: this may be interpreted differently depending on --lr-scheduler)')
    parser.add_argument('--clip-norm', default=0., type=float, help='clip gradient')
    parser.add_argument('--empty-cache-freq', default=0, type=int,
                        help='how often to clear the PyTorch CUDA cache (0 to disable)')
    parser.add_argument('--save-checkpoint-every-niter', default=10000, type=int)

    FairseqAdam.add_args(parser)
    PolynomialDecaySchedule.add_args(parser)

    # FP16 training
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--memory-efficient-fp16',
                        action='store_true',
                        help='Use memory efficient fp16')
    parser.add_argument('--threshold-loss-scale', type=float, default=None)
    parser.add_argument('--fp16-init-scale', type=float, default=128)
    # parser.add_argument('--fp16-scale-window', type=int, default=0)
    parser.add_argument('--fp16-scale-tolerance', type=float, default=0.0)
    parser.add_argument('--min-loss-scale', default=1e-4, type=float, metavar='D',
                        help='minimum FP16 loss scale, after which training is stopped')

    args = parser.parse_args()

    return args


def init_logger(args):
    # setup logger
    logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(f"[{socket.gethostname()} | Node {args.node_id} | Rank {args.global_rank} | %(asctime)s] %(message)s",
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler.setFormatter(formatter)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    return logger

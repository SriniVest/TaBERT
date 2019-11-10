import logging
from argparse import ArgumentParser
from pathlib import Path
import socket

import torch

from fairseq.options import eval_str_list
from fairseq.optim.adam import FairseqAdam
from fairseq.optim.lr_scheduler.polynomial_decay_schedule import PolynomialDecaySchedule


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

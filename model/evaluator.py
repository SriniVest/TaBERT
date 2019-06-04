from functools import partial

import gc
import torch
import torch.nn as nn
import logging
from torch.utils.data import Dataset, DataLoader, SequentialSampler

from tqdm import tqdm
from typing import Dict
import math


class Evaluator(object):
    def __init__(self, args):
        self.args = args
        self.batch_size = args.train_batch_size * 4
        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device('cpu')

    def evaluate(self, model: nn.Module, dataset: Dataset) -> Dict:
        gc.collect()

        was_training = model.training
        model.eval()

        cum_loss = num_slots = 0.
        with torch.no_grad():
            data_loader = DataLoader(dataset,
                                     batch_size=self.batch_size, sampler=SequentialSampler(dataset),
                                     collate_fn=partial(dataset.collate, tokenizer=dataset.tokenizer))

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='sum')

            with tqdm(total=len(data_loader), desc=f"Evaluation") as pbar:
                for step, batch in enumerate(data_loader):
                    batch = tuple(t.to(self.device) for t in batch)
                    input_ids, input_mask, segment_ids, lm_label_ids = batch
                    # (batch_size, max_seq_len, vocab_size)
                    predictions = model(input_ids, segment_ids, input_mask)

                    loss_sum = loss_fct(predictions.view(-1, predictions.size(-1)), lm_label_ids.view(-1))
                    cum_loss += loss_sum.item()
                    num_slots += (lm_label_ids != -1).sum().item()

                    pbar.update(1)

        if was_training:
            model.train()

        ppl = math.exp(cum_loss / num_slots)

        return {'ppl': ppl}

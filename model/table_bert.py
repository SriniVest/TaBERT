from pathlib import Path
from typing import Union, Dict

import torch
from pytorch_pretrained_bert import BertForPreTraining, BertForMaskedLM, BertTokenizer
from torch import nn as nn

from utils.config import TableBertConfig


MAX_BERT_INPUT_LENGTH = 512
NEGATIVE_NUMBER = -1e8
CONFIG_NAME = 'bert_config.json'
WEIGHTS_NAME = 'pytorch_model.bin'


class TableBertModel(nn.Module):
    def __init__(
        self,
        config: TableBertConfig,
        bert_model: BertForPreTraining = None,
        **kwargs
    ):
        nn.Module.__init__(self)
        bert_model = bert_model or BertForMaskedLM.from_pretrained(config.base_model_name)
        self._bert_model = bert_model
        self.tokenizer = BertTokenizer.from_pretrained(config.base_model_name)
        self.config = config

    @property
    def bert(self):
        return self._bert_model.bert

    @property
    def bert_config(self):
        return self.bert.config

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def output_size(self):
        return self.bert.config.hidden_size

    @classmethod
    def load(
        cls,
        model_path: Union[str, Path],
        config_file: Union[str, Path],
        **override_config: Dict
    ):
        if model_path and isinstance(model_path, str):
            model_path = Path(model_path)
        if isinstance(config_file, str):
            config_file = Path(config_file)

        if model_path:
            state_dict = torch.load(str(model_path), map_location='cpu')
        else:
            state_dict = None

        config = TableBertConfig.from_file(config_file, **override_config)

        # old model format
        if '_bert_model' not in state_dict:
            bert_model = BertForMaskedLM.from_pretrained(
                config.base_model_name,
                state_dict=state_dict
            )
            model = cls(config, bert_model=bert_model)
        else:
            model = cls(config)
            model.load_state_dict(state_dict)

        return model

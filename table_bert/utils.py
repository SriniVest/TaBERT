#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging

try:
    from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertForPreTraining, BertConfig, BertModel
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    logging.warning('You are using the old verion of `pytorch_pretrained_bert`')
except ImportError:
    from transformers.tokenization_bert import BertTokenizer
    from transformers.modeling_bert import BertForMaskedLM, BertForPreTraining, BertModel
    from transformers.configuration_bert import BertConfig

    hf_flag = 'new'

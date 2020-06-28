import logging
from enum import Enum


class TransformerVersion(Enum):
    PYTORCH_PRETRAINED_BERT = 0
    TRANSFORMERS = 1


TRANSFORMER_VERSION = None


try:
    from pytorch_pretrained_bert.modeling import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertConfig,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead, BertLayerNorm, gelu
    )
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    TRANSFORMER_VERSION = TransformerVersion.PYTORCH_PRETRAINED_BERT
    logging.warning('You are using the old version of `pytorch_pretrained_bert`')
except ImportError:
    from transformers.tokenization_bert import BertTokenizer
    from transformers.modeling_bert import (
        BertForMaskedLM, BertForPreTraining, BertModel,
        BertSelfOutput, BertIntermediate, BertOutput,
        BertLMPredictionHead, BertLayerNorm, gelu
    )
    from transformers.configuration_bert import BertConfig

    hf_flag = 'new'
    TRANSFORMER_VERSION = TransformerVersion.TRANSFORMERS


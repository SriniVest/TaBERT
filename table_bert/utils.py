import logging

try:
    from transformers.tokenization_bert import BertTokenizer
    from transformers.modeling_bert import BertForMaskedLM, BertForPreTraining
    from transformers.configuration_bert import BertConfig

    hf_flag = 'new'
except ImportError:
    logging.warning('Are you using the old verion of `pytorch_pretrained_bert`? Trying to load it...')
    from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertForPreTraining, BertConfig, BertModel
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'

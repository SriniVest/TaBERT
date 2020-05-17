import logging

try:
    from pytorch_pretrained_bert.modeling import BertForMaskedLM, BertForPreTraining, BertConfig, BertModel
    from pytorch_pretrained_bert.tokenization import BertTokenizer

    hf_flag = 'old'
    logging.warning('Are you using the old verion of `pytorch_pretrained_bert`')
except ImportError:
    from transformers.tokenization_bert import BertTokenizer
    from transformers.modeling_bert import BertForMaskedLM, BertForPreTraining, BertModel
    from transformers.configuration_bert import BertConfig

    hf_flag = 'new'

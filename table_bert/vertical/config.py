from argparse import ArgumentParser

from pytorch_pretrained_bert import BertForMaskedLM

from table_bert.config import TableBertConfig


class VerticalAttentionTableBertConfig(TableBertConfig):
    def __init__(self, **kwargs):
        TableBertConfig.__init__(self, **kwargs)

        self.num_vertical_attention_heads = kwargs.get('num_vertical_attention_heads', 6)
        self.num_vertical_layers = kwargs.get('num_vertical_layers', 3)
        self.sample_row_num = kwargs.get('sample_row_num', 3)
        self.table_mask_strategy = kwargs.get('table_mask_strategy', 'column')
        self.predict_cell_tokens = kwargs.get('predict_cell_tokens', False)

        bert_config = BertForMaskedLM.from_pretrained(self.base_model_name).config
        for k, v in vars(bert_config).items():
            setattr(self, k, v)

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        TableBertConfig.add_args(parser)

        parser.add_argument("--num_vertical_attention_heads", type=int, default=6)
        parser.add_argument("--num_vertical_layers", type=int, default=3)
        parser.add_argument("--sample_row_num", type=int, default=3)
        parser.add_argument("--predict_cell_tokens", action='store_true', default=False)

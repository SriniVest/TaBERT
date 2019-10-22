import json
from argparse import ArgumentParser
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Union

from pytorch_pretrained_bert import BertTokenizer


class TableBertConfig(SimpleNamespace):
    def __init__(
        self,
        base_model_name: str = 'bert-base-uncased',
        column_delimiter: str = '[SEP]',
        context_first: bool = True,
        cell_input_template: str = 'column (value) (type)',
        column_representation: str = 'mean_pool',
        max_cell_len: int = 5,
        max_sequence_len: int = 512,
        max_context_len: int = 256,
        masked_context_prob: float = 0.15,
        masked_column_prob: float = 0.2,
        max_predictions_per_seq: int = 100,
        context_sample_strategy: str = 'nearest',
        table_mask_strategy: str = 'column_token',
        do_lower_case: bool = True,
        **kwargs
    ):
        super(TableBertConfig, self).__init__()

        self.base_model_name = base_model_name
        self.column_delimiter = column_delimiter
        self.context_first = context_first
        self.column_representation = column_representation

        self.max_cell_len = max_cell_len
        self.max_sequence_len = max_sequence_len
        self.max_context_len = max_context_len

        self.do_lower_case = do_lower_case

        tokenizer = BertTokenizer.from_pretrained(self.base_model_name)
        self.cell_input_template = tokenizer.tokenize(cell_input_template)

        self.masked_context_prob = masked_context_prob
        self.masked_column_prob = masked_column_prob
        self.max_predictions_per_seq = max_predictions_per_seq
        self.context_sample_strategy = context_sample_strategy
        self.table_mask_strategy = table_mask_strategy

    @classmethod
    def add_args(cls, parser: ArgumentParser):
        parser.add_argument('--base_model_name', type=str, default='bert-base-uncased')

        parser.add_argument('--context_first', dest='context_first', action='store_true')
        parser.add_argument('--table_first', dest='context_first', action='store_false')
        parser.set_defaults(context_first=True)

        parser.add_argument("--column_delimiter", type=str, default='[SEP]', help='Column delimiter')
        parser.add_argument("--cell_input_template", type=str, default='column|value|type', help='Cell representation')
        parser.add_argument("--column_representation", type=str, default='mean_pool', help='Column representation')

        # training specifications
        parser.add_argument("--max_sequence_len", type=int, default=512)
        parser.add_argument("--max_context_len", type=int, default=256)
        parser.add_argument("--max_cell_len", type=int, default=5)

        parser.add_argument("--masked_context_prob", type=float, default=0.15,
                            help="Probability of masking each token for the LM task")
        parser.add_argument("--masked_column_prob", type=float, default=0.20,
                            help="Probability of masking each token for the LM task")
        parser.add_argument("--max_predictions_per_seq", type=int, default=100,
                            help="Maximum number of tokens to mask in each sequence")

        parser.add_argument('--context_sample_strategy', type=str, default='nearest',
                            choices=['nearest', 'concate_and_enumerate'])
        parser.add_argument('--table_mask_strategy', type=str, default='column_token',
                            choices=['column', 'column_token'])

        parser.add_argument("--do_lower_case", action="store_true")
        parser.set_defaults(do_lower_case=True)

        return parser

    @classmethod
    def from_file(cls, file_path: Union[str, Path], **override_args):
        if isinstance(file_path, str):
            file_path = Path(file_path)

        args = json.load(file_path.open())
        override_args = override_args or dict()
        args.update(override_args)
        default_config = TableBertConfig()
        config_dict = {}
        for key, default_val in vars(default_config).items():
            val = args.get(key, default_val)
            config_dict[key] = val

        # backward compatibility
        if 'column_item_delimiter' in args:
            column_item_delimiter = args['column_item_delimiter']
            cell_input_template = 'column'
            use_value = args.get('use_sample_value', True)
            use_type = args.get('use_type_text', True)

            if use_value:
                cell_input_template += column_item_delimiter + 'value'
            if use_type:
                cell_input_template += column_item_delimiter + 'type'

            config_dict['cell_input_template'] = cell_input_template

        config = cls(**config_dict)

        return config

    @classmethod
    def from_dict(cls, args: Dict):
        return cls(**args)

    def save(self, file_path: Path):
        json.dump(vars(self), file_path.open('w'), indent=2)

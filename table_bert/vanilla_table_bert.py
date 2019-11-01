from typing import List, Any
import numpy as np

import torch
from pytorch_pretrained_bert import BertForPreTraining
from torch_scatter import scatter_max, scatter_mean

from table_bert.table_bert import TableBertModel
from table_bert.config import TableBertConfig
from table_bert.table import Table
from table_bert.input_formatter import VanillaTableBertInputFormatter


class VanillaTableBert(TableBertModel):
    def __init__(
        self,
        config: TableBertConfig,
        bert_model: BertForPreTraining = None,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, bert_model=bert_model, **kwargs)
        self.input_formatter = VanillaTableBertInputFormatter(self.config)

    def forward(self, *input: Any, **kwargs: Any):
        return self._bert_model(*input, **kwargs)

    def encode_context_and_table(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        context_token_indices: torch.Tensor,
        context_token_mask: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_mask: torch.Tensor,
        **kwargs
    ):

        # print('input_ids', input_ids.size(), file=sys.stderr)
        # print('segment_ids', segment_ids.size(), file=sys.stderr)
        # print('attention_mask', attention_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.size(), file=sys.stderr)
        # print('column_token_mask', column_token_mask.sum(dim=-1), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.size(), file=sys.stderr)
        # print('column_token_to_column_id', column_token_to_column_id.sum(dim=-1), file=sys.stderr)
        # print('column_mask', column_mask.size(), file=sys.stderr)

        # try:
        sequence_output, _ = self.bert(input_ids, segment_ids, attention_mask, output_all_encoded_layers=False)
        # except:
        #     print('!!!!!Exception!!!!!')
        #     datum = (input_ids, segment_ids, attention_mask, question_token_mask,
        #              column_token_mask, column_token_to_column_id, column_mask)
        #     torch.save(datum, 'debug.tensors.bin')
        #     raise

        # gather column representations
        # (batch_size, max_seq_len, encoding_size)
        flattened_column_encoding = sequence_output
        # (batch_size, max_column_size, encoding_size)
        column_encoding = self.get_column_representation(
            flattened_column_encoding,
            column_token_to_column_id,
            column_token_mask,
            column_mask,
            aggregator=self.config.column_representation
        )

        # (batch_size, context_len, encoding_size)
        context_encoding = torch.gather(
            sequence_output,
            dim=1,
            index=context_token_indices.unsqueeze(-1).expand(-1, -1, sequence_output.size(-1)),
        )
        context_encoding = context_encoding * context_token_mask.unsqueeze(-1)

        return context_encoding, column_encoding

    @staticmethod
    def get_column_representation(
        flattened_column_encoding: torch.Tensor,
        column_token_to_column_id: torch.Tensor,
        column_token_mask: torch.Tensor,
        column_mask: torch.Tensor,
        aggregator: str = 'mean_pool'
    ) -> torch.Tensor:
        """
        Args:
            flattened_column_encoding: (batch_size, total_column_token_num, encoding_size)
            column_token_to_column_id: (batch_size, total_column_token_num + 1)
            column_mask: (batch_size, max_column_num)
            aggregator: ['mean_pool', 'max_pool', 'first_token']
        Returns:
            column_encoding: (batch_size, max_column_num, encoding_size)
        """

        if aggregator.startswith('max_pool'):
            agg_func = scatter_max
            flattened_column_encoding[column_token_mask == 0] = float('-inf')
        elif aggregator.startswith('mean_pool') or aggregator.startswith('first_token'):
            agg_func = scatter_mean
        else:
            raise ValueError(f'Unknown column representation method {aggregator}')

        max_column_num = column_mask.size(-1)
        # column_token_to_column_id: (batch_size, max_column_num)
        # (batch_size, max_column_size + 1, encoding_size)
        result = agg_func(flattened_column_encoding,
                          column_token_to_column_id.unsqueeze(-1).expand(-1, -1, flattened_column_encoding.size(-1)),
                          dim=1,
                          dim_size=max_column_num + 1)

        # remove the last "garbage collection" entry, mask out padding columns
        result = result[:, :-1] * column_mask.unsqueeze(-1)

        if aggregator == 'max_pool':
            column_encoding = result[0]
        else:
            column_encoding = result

        return column_encoding

    def to_tensor_dict(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        table_specific_tensors=True
    ):
        instances = []
        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            instance = self.input_formatter.get_input(context, table)
            instances.append(instance)

        batch_size = len(contexts)
        max_sequence_len = max(len(x['tokens']) for x in instances)

        # basic tensors
        input_array = np.zeros((batch_size, max_sequence_len), dtype=np.int)
        mask_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)
        segment_array = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

        # table specific tensors
        if table_specific_tensors:
            max_column_num = max(len(x['column_spans']) for x in instances)
            max_context_len = max(x['context_length'] for x in instances)

            context_token_indices = np.zeros((batch_size, max_context_len), dtype=np.int)
            context_mask = np.zeros((batch_size, max_context_len), dtype=np.bool)
            column_token_mask = np.zeros((batch_size, max_sequence_len), dtype=np.bool)

            # we initialize the mapping with the id of last column as the "garbage collection" entry for reduce ops
            column_token_to_column_id = np.zeros((batch_size, max_sequence_len), dtype=np.int)
            column_token_to_column_id.fill(max_column_num)

            column_mask = np.zeros((batch_size, max_column_num), dtype=np.bool)

            column_span = 'whole_span'
            if 'column_name' in self.config.column_representation:
                column_span = 'column_name'
            elif 'first_token' in self.config.column_representation:
                column_span = 'first_token'

        for i, instance in enumerate(instances):
            token_ids = self.tokenizer.convert_tokens_to_ids(instance['tokens'])

            input_array[i, :len(token_ids)] = token_ids
            segment_array[i, instance['segment_a_length']: ] = 1
            mask_array[i, :len(token_ids)] = 1.

            if table_specific_tensors:
                context_token_indices[i, :instance['context_length']] = list(range(*instance['context_span'])) #instance['context_token_indices']
                context_mask[i, :instance['context_length']] = 1.

                header = tables[i].header
                for col_id, column in enumerate(header):
                    if col_id < len(instance['column_spans']):
                        col_start, col_end = instance['column_spans'][col_id][column_span]

                        column_token_to_column_id[i, col_start: col_end] = col_id
                        column_token_mask[i, col_start: col_end] = 1.
                        column_mask[i, col_id] = 1.

        tensor_dict = {
            'input_ids': torch.tensor(input_array.astype(np.int64)),
            'segment_ids': torch.tensor(segment_array.astype(np.int64)),
            'attention_mask': torch.tensor(mask_array, dtype=torch.float32),
        }

        if table_specific_tensors:
            tensor_dict.update({
                'context_token_indices': torch.tensor(context_token_indices.astype(np.int64)),
                'context_token_mask': torch.tensor(context_mask, dtype=torch.float32),
                'column_token_to_column_id': torch.tensor(column_token_to_column_id.astype(np.int64)),
                'column_token_mask': torch.tensor(column_token_mask, dtype=torch.float32),
                'column_mask': torch.tensor(column_mask, dtype=torch.float32)
            })

        # for instance in instances:
        #     print(instance)

        return tensor_dict, instances

    def encode(self, contexts: List[List[str]], tables: List[Table]):
        tensor_dict, instances = self.to_tensor_dict(contexts, tables)
        device = next(self.parameters()).device

        for key in tensor_dict.keys():
            tensor_dict[key] = tensor_dict[key].to(device)

        context_encoding, column_encoding = self.encode_context_and_table(
            **tensor_dict)

        info = {
            'tensor_dict': tensor_dict,
            'instances': instances
        }

        return context_encoding, column_encoding, info

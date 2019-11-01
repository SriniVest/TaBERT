import math
import numpy as np

import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertConfig, BertForPreTraining
from pytorch_pretrained_bert.modeling import BertSelfOutput

from table_bert.table import Column
from table_bert.vanilla_table_bert import VanillaTableBert, VanillaTableBertInputFormatter, TableBertConfig
from table_bert.table import *


class VerticalEmbeddingLayer(nn.Module):
    def forward(self, hidden_states: torch.Tensor):
        return hidden_states


class VerticalTableBertLayer(nn.Module):
    def __init__(self, config: TableBertConfig, bert_config: BertConfig):
        super(VerticalTableBertLayer, self).__init__()

        self.self_attention = VerticalSelfAttention(config, bert_config)
        self.self_output = BertSelfOutput(bert_config)

    def forward(self, hidden_states, attention_mask):
        self_attention_output = self.self_attention(hidden_states, attention_mask)
        output = self.self_output(self_attention_output, hidden_states)

        return output


class VerticalSelfAttention(nn.Module):
    def __init__(self, config: TableBertConfig, bert_config: BertConfig):
        super(VerticalSelfAttention, self).__init__()

        if bert_config.hidden_size % config.num_vertical_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (bert_config.hidden_size, config.num_vertical_attention_heads))

        self.num_attention_heads = config.num_vertical_attention_heads
        self.attention_head_size = int(bert_config.hidden_size / config.num_vertical_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_linear = nn.Linear(bert_config.hidden_size, self.all_head_size)
        self.key_linear = nn.Linear(bert_config.hidden_size, self.all_head_size)
        self.value_linear = nn.Linear(bert_config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(bert_config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # (batch_size, max_row_num, max_sequence_len, num_attention_heads, attention_head_size)
        x = x.view(*new_x_shape)
        # (batch_size, max_sequence_len, num_attention_heads, max_row_num, attention_head_size)
        x = x.permute(0, 2, 3, 1, 4)

        return x

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor):
        mixed_query_layer = self.query_linear(hidden_states)
        mixed_key_layer = self.key_linear(hidden_states)
        mixed_value_layer = self.value_linear(hidden_states)

        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, attention_head_size)
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, max_row_num)
        attention_probs = torch.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # ([batch_size, max_sequence_len], num_attention_heads, max_row_num, attention_head_size)
        context_layer = torch.matmul(attention_probs, value_layer)
        # (batch_size, max_row_num, max_sequence_len, num_attention_heads, attention_head_size)
        context_layer = context_layer.permute(0, 3, 1, 2, 4).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer


class VerticalAttentionTableBert(VanillaTableBert):
    def __init__(
        self,
        config: TableBertConfig,
        bert_model: BertForPreTraining = None,
        **kwargs
    ):
        super(VanillaTableBert, self).__init__(config, bert_model=bert_model, **kwargs)
        self.input_formatter = VanillaTableBertInputFormatter(self.config)

        self.config.num_vertical_attention_heads = 6
        self.config.num_vertical_layers = 2

        self.vertical_embedding_layer = VerticalEmbeddingLayer()
        self.vertical_transformer_layers = nn.ModuleList([
            VerticalTableBertLayer(self.config, self.bert_config)
            for _ in range(self.config.num_vertical_layers)
        ])

    def to_tensor_dict(
        self,
        contexts: List[List[str]],
        tables: List[Table],
        table_specific_tensors=True
    ):
        batch_size = len(contexts)
        row_to_table_map = []
        example_first_row_id = []
        max_row_num = max(len(table) for table in tables)

        row_contexts = []
        row_tables = []

        for e_id, (context, table) in enumerate(zip(contexts, tables)):
            example_first_row_id.append(len(row_contexts))

            for row_id, row in enumerate(table.data):
                new_header = []
                for column in table.header:
                    new_col = Column(
                        column.name, column.type, row[column.name],
                        name_tokens=column.name_tokens, sample_value_tokens=row[column.name]
                    )
                    new_header.append(new_col)

                row_tb = Table(row_id, new_header)

                row_tables.append(row_tb)
                row_contexts.append(context)
                row_to_table_map.append(e_id)

        tensor_dict, row_instances = VanillaTableBert.to_tensor_dict(self, row_contexts, row_tables)

        column_mask = tensor_dict['column_mask']
        context_token_mask = tensor_dict['context_token_mask']
        table_mask = np.zeros((batch_size, max_row_num, column_mask.size(-1)), dtype=np.float32)
        context_mask = np.zeros((batch_size, max_row_num, context_token_mask.size(-1)), dtype=np.float32)
        for e_id, table in enumerate(tables):
            first_row_flattened_pos = example_first_row_id[e_id]
            table_mask[e_id, :len(table)] = column_mask[first_row_flattened_pos]
            context_mask[e_id, :len(table)] = context_token_mask[first_row_flattened_pos]

        tensor_dict['table_mask'] = torch.from_numpy(table_mask).to(self.device)
        tensor_dict['context_mask'] = torch.from_numpy(context_mask).to(self.device)

        # (total_row_num)
        tensor_dict['row_to_table_map'] = torch.tensor(row_to_table_map, dtype=torch.long, device=self.device)
        tensor_dict['example_first_row_id'] = torch.tensor(example_first_row_id, dtype=torch.long, device=self.device)
        tensor_dict['table_sizes'] = [len(table) for table in tables]

        return tensor_dict

    def unpack_flattened_encoding(self, row_context_encoding, row_cell_encoding, tensor_dict):
        all_row_num = row_context_encoding.size(0)
        scatter_indices = np.zeros(all_row_num, dtype=np.int64)
        table_sizes = tensor_dict['table_sizes']
        table_num = len(table_sizes)
        max_row_num = max(table_sizes)
        cum_size = 0
        for table_id, tb_size in enumerate(table_sizes):
            scatter_indices[cum_size: cum_size + tb_size] = list(
                range(table_id * max_row_num, table_id * max_row_num + tb_size))
            cum_size += tb_size

        new_zeros = row_context_encoding.new_zeros

        table_encoding = new_zeros(
            table_num * max_row_num,
            row_cell_encoding.size(-2),  # max_column_num
            row_cell_encoding.size(-1)   # encoding_size
        )
        table_encoding[scatter_indices] = row_cell_encoding
        table_encoding = table_encoding.view(table_num, max_row_num, row_cell_encoding.size(-2), row_cell_encoding.size(-1))

        context_encoding = new_zeros(
            table_num * max_row_num,
            row_context_encoding.size(-2),
            row_context_encoding.size(-1)
        )
        context_encoding[scatter_indices] = row_context_encoding
        context_encoding = context_encoding.view(table_num, max_row_num, row_context_encoding.size(-2), row_context_encoding.size(-1))

        return context_encoding, table_encoding

    def encode(self, contexts: List[List[str]], tables: List[Table]):
        row_tensor_dict = self.to_tensor_dict(contexts, tables)

        for key in row_tensor_dict.keys():
            if torch.is_tensor(row_tensor_dict[key]):
                row_tensor_dict[key] = row_tensor_dict[key].to(self.device)

        # (total_row_num, sequence_len, ...)
        row_context_encoding, row_cell_encoding = self.encode_context_and_table(
            **row_tensor_dict)

        # (batch_size, max_row_num, max_context_len, encoding_size)
        # (batch_size, max_row_num, max_column_num, encoding_size)
        context_encoding, table_encoding = self.unpack_flattened_encoding(
            row_context_encoding, row_cell_encoding,
            row_tensor_dict
        )

        # (batch_size, max_row_num, sequence_len, encoding_size)
        hidden_states = torch.cat([context_encoding, table_encoding], dim=2)

        unpacked_context_mask = row_tensor_dict['context_mask']
        unpacked_table_mask = row_tensor_dict['table_mask']

        # (batch_size, max_row_num, sequence_len)
        attention_mask = torch.cat(
            [unpacked_context_mask, unpacked_table_mask],
            dim=-1
        )
        # (batch_size, sequence_len, 1, max_row_num, 1)
        attention_mask = attention_mask.transpose(-1, -2)[:, :, None, :, None]
        attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        attention_mask = (1.0 - attention_mask) * -10000.0

        hidden_states = self.vertical_embedding_layer(hidden_states)
        vertical_layer_outputs = []
        for vertical_layer in self.vertical_transformer_layers:
            hidden_states = vertical_layer(hidden_states, attention_mask=attention_mask)
            vertical_layer_outputs.append(hidden_states)

        last_hidden_states = vertical_layer_outputs[-1]
        last_context_encoding = last_hidden_states[:, :, :context_encoding.size(2), :]
        last_table_encoding = last_hidden_states[:, :, context_encoding.size(2):, :]

        # mean-pool last encoding
        table_sizes = torch.tensor([len(table) for table in tables], device=self.device, dtype=torch.float32)
        mean_pooled_context_encoding = (last_context_encoding * unpacked_context_mask.unsqueeze(-1)).sum(dim=1) / table_sizes[:, None, None]
        mean_pooled_table_encoding = (last_table_encoding * unpacked_table_mask.unsqueeze(-1)).sum(dim=1) / table_sizes[:, None, None]

        mean_pooled_context_mask = unpacked_context_mask[:, 0, :]
        context_encoding = {
            'value': mean_pooled_context_encoding,
            'mask': mean_pooled_context_mask,
        }

        mean_pooled_table_mask = row_tensor_dict['column_mask'][row_tensor_dict['example_first_row_id']]
        table_encoding = {
            'value': mean_pooled_table_encoding,
            'mask': mean_pooled_table_mask,
        }

        info = {
            'tensor_dict': row_tensor_dict,
        }

        return context_encoding, table_encoding, info

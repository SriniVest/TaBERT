from table_bert.table import Column
from table_bert.vanilla_table_bert import *


class ContentBasedTableBert(VanillaTableBert):
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
        table_mask = np.zeros((batch_size, max_row_num, column_mask.size(-1)), dtype=np.float32)
        for e_id, table in enumerate(tables):
            first_row_flattened_pos = example_first_row_id[e_id]
            table_mask[e_id, :len(table)] = column_mask[first_row_flattened_pos]

        tensor_dict['table_mask'] = torch.from_numpy(table_mask).to(self.device)

        # (total_row_num)
        tensor_dict['row_to_table_map'] = torch.tensor(row_to_table_map, dtype=torch.long, device=self.device)
        tensor_dict['example_first_row_id'] = torch.tensor(example_first_row_id, dtype=torch.long, device=self.device)

        return tensor_dict

    def flattened_row_encoding_to_table_encoding(self, row_encoding, table_sizes):
        table_num = len(table_sizes)
        max_row_num = max(table_sizes)

        scatter_indices = np.zeros(row_encoding.size(0), dtype=np.int64)
        cum_size = 0
        for table_id, tb_size in enumerate(table_sizes):
            scatter_indices[cum_size: cum_size + tb_size] = list(
                range(table_id * max_row_num, table_id * max_row_num + tb_size))
            cum_size += tb_size

        scatter_indices = torch.from_numpy(scatter_indices).to(self.device)

        out = row_encoding.new_zeros(
            table_num * max_row_num,
            row_encoding.size(-2),  # max_column_num
            row_encoding.size(-1)   # encoding_size
        )
        out[scatter_indices] = row_encoding

        table_encoding = out.view(table_num, max_row_num, row_encoding.size(-2), row_encoding.size(-1))

        return table_encoding

    def encode(self, contexts: List[List[str]], tables: List[Table]):
        row_tensor_dict = self.to_tensor_dict(contexts, tables)

        for key in row_tensor_dict.keys():
            row_tensor_dict[key] = row_tensor_dict[key].to(self.device)

        # (total_row_num, sequence_len, ...)
        row_context_encoding, row_encoding = self.encode_context_and_table(
            **row_tensor_dict)

        # (batch_size, context_len, ...)
        context_encoding = scatter_mean(row_context_encoding, index=row_tensor_dict['row_to_table_map'], dim=0)

        # # (batch_size, context_len, ...)
        # context_encoding, max_row_id = scatter_max(row_context_encoding, index=row_tensor_dict['row_to_table_map'],
        #                                            dim=0)

        example_first_row_indices = row_tensor_dict['example_first_row_id']
        # (batch_size, context_len)
        context_mask = row_tensor_dict['context_token_mask'][example_first_row_indices]

        context_encoding = {
            'value': context_encoding,
            'mask': context_mask,
        }

        # (batch_size, row_num, column_num, encoding_size)
        table_encoding_var = self.flattened_row_encoding_to_table_encoding(
            row_encoding, [len(table) for table in tables])
        # max_row_num = table_encoding_var.size(1)
        # table_column_mask = row_tensor_dict['column_mask'][example_first_row_indices]

        table_encoding = {
            'value': table_encoding_var,
            'mask': row_tensor_dict['table_mask'],
            'column_mask': row_tensor_dict['column_mask']
            #  'row_encoding': row_encoding,
            #  'row_encoding_mask': row_tensor_dict['column_mask']
        }

        info = {
            'tensor_dict': row_tensor_dict,
        }

        return context_encoding, table_encoding, info

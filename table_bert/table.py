from typing import List, Dict, Any
import pandas as pd


class Column(object):
    def __init__(self, name, type, sample_value=None, **kwargs):
        self.name = name
        self.type = type
        self.sample_value = sample_value

        self.fields = []
        for key, val in kwargs.items():
            self.fields.append(key)
            setattr(self, key, val)

    def to_dict(self):
        data = {
            'name': self.name,
            'type': self.type,
            'sample_value': self.sample_value,
        }

        for key in self.fields:
            data[key] = getattr(self, key)

        return data


class Table(object):
    def __init__(self, id, header, data=None, **kwargs):
        self.id = id
        self.header = header
        self.header_index = {column.name: column for column in header}
        self.data: List[Any] = data
        self.fields = []

        for key, val in kwargs.items():
            self.fields.append(key)
            setattr(self, key, val)

    def with_rows(self, rows):
        extra_fields = {f: getattr(self, f) for f in self.fields}

        return Table(self.id, self.header, data=rows, **extra_fields)

    def get_column(self, column_name):
        return self.header_index[column_name]

    def __len__(self):
        return len(self.data)

    @property
    def as_row_list(self):
        if isinstance(self.data[0], dict):
            return [
                row[column.name]
                for row in self.data
                for column in self.header
            ]

        return self.data

    def to_data_frame(self, tokenizer=None):
        row_data = self.as_row_list
        columns = [column.name for column in self.header]

        if tokenizer:
            row_data = [
                [
                    ' '.join(tokenizer.tokenize(str(cell)))
                    for cell in row
                ]
                for row in row_data
            ]

            columns = [' '.join(tokenizer.tokenize(str(column))) for column in columns]

        df = pd.DataFrame(row_data, columns=columns)

        return df

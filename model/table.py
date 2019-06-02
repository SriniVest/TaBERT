from typing import Dict


class Column(object):
    def __init__(self, name, type, sample_value=None, **kwargs):
        self.name = name
        self.type = type
        self.sample_value = sample_value

        for key, val in kwargs.items():
            setattr(self, key, val)


class Example(object):
    def __init__(self, guid, header, context):
        self.guid = guid
        self.header = header
        self.context = context

    @classmethod
    def from_dict(cls, entry: Dict, tokenizer, suffix) -> 'Example':
        header = []
        for col_data in entry['header']:
            sample_value = col_data['sample_value']['value']
            column = Column(col_data['name'],
                            col_data['type'],
                            sample_value,
                            name_tokens=tokenizer.tokenize(col_data['name']),
                            type_tokens=tokenizer.tokenize(col_data['type']),
                            sample_value_tokens=tokenizer.tokenize(sample_value))
            header.append(column)

        context = []
        for para in entry['context']:
            for sent in para:
                tokenized_sent = tokenizer.tokenize(sent)
                context.append(tokenized_sent)

        if entry['caption']:
            caption = tokenizer.tokenize(entry['caption'])
            context.append(caption)

        guid = f"{entry['id']}_{'_'.join(entry['title'])}_{suffix}"

        return cls(guid, header, context)

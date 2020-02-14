# TaBERT: Learning Contextual Representations for Natural Language Utterances and Structured Tables

## Installation

First, install the conda environment `pytorch` with supporting libraries.

```bash
bash scripts/setup_env.sh
```

Once the conda environment is created, install `TaBERT` using the following command:

```bash
conda activate pytorch
pip install --editable .
```

## Using a Pre-trained Model

First, load the model from a checkpoint folder

```python
from table_bert import TableBertModel

model = TableBertModel.load(
    'path/to/pretrained/model/checkpoint.bin',
    config_file='path/to/pretrained/model/config.json'
)
```

To produce representations of natural language text and and its associated table:
```python
from table_bert import Table, Column

table = Table(
    id='List of countries by GDP (PPP)',
    header=[
        Column('Nation', 'text'),
        Column('Gross Domestic Product', 'real')
    ],
    data=[
        ['United States', '21,439,453'],
        ['China', '27,308,857'],
        ['European Union', '22,774,165'],
    ]
).tokenize(model.tokenizer)

# visualize table in IPython notebook:
# display(table.to_data_frame(), detokenize=True)

context = 'show me countries ranked by GDP'

# model takes batched, tokenized inputs
context_encoding, column_encoding, info_dict = model.encode(
    contexts=[model.tokenizer.tokenize(context)],
    tables=[table]
)
```

For the returned tuple `context_encoding` and `column_encoding` are PyTorch tensors representing utterances and table columns. `info_dict` contains useful information (e.g., context/table masks, original inputs to BERT) for downstream usage.

```
context_encoding.shape
>>> torch.Size([1, 7, 768])

column_encoding.shape
>>> torch.Size([1, 2, 768])
```
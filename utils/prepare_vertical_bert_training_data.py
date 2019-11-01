from table_bert.content_based_table_bert import VerticalAttentionTableBert
from table_bert.vertical.config import VerticalAttentionTableBertConfig
from table_bert.vertical.dataset import serialize_row_data
from table_bert.vertical.input_formatter import VerticalAttentionTableBertInputFormatter
from utils.prepare_training_data import *


def write_instance_to_file(
        output_file: Path,
        num_workers: int,
        stat_send: connection.Connection,
        shard_size: int = 3000000
):
    context = zmq.Context()
    instance_receiver = context.socket(zmq.PULL)
    instance_receiver.bind(TRAIN_INSTANCE_QUEUE_ADDRESS)

    finished_worker_num = 0
    num_instances = 0
    shard_id = 0

    row_data_sequences = []
    row_data_offsets = []
    mlm_data_sequences = []
    mlm_data_offsets = []

    def _save_shard():
        nonlocal shard_id

        data = {
            'row_data_sequences': np.uint16(row_data_sequences),
            'row_data_offsets': np.uint64(row_data_offsets),
            'mlm_data_sequences': np.uint16(mlm_data_sequences),
            'mlm_data_offsets': np.uint64(mlm_data_offsets),
        }

        tgt_file = output_file.with_name(output_file.name + f'.shard{shard_id}.bin')
        torch.save(data, str(tgt_file), pickle_protocol=4)

        shard_id += 1
        del row_data_sequences[:]
        del row_data_offsets[:]
        del mlm_data_sequences[:]
        del mlm_data_offsets[:]

    while True:
        data = instance_receiver.recv_pyobj()
        if data is not None:
            data = msgpack.unpackb(data, raw=False)

            table_data = []
            for row_inst in data['rows']:
                row_data = serialize_row_data(row_inst)
                table_data.extend(row_data)

            row_data_offsets.append([
                data['table_size'][0],  # row_num
                data['table_size'][1],  # column_num
                len(row_data_sequences),  # start index
                len(row_data_sequences) + len(table_data)  # end index
            ])
            row_data_sequences.extend(table_data)

            s1 = len(mlm_data_sequences)
            mlm_data = []

            mlm_data.extend(data['masked_context_token_positions'])
            s2 = s1 + len(mlm_data)

            mlm_data.extend(data['masked_context_token_label_ids'])
            s3 = s1 + len(mlm_data)

            mlm_data.extend(data['masked_column_token_column_ids'])
            s4 = s1 + len(mlm_data)

            mlm_data.extend(data['masked_column_token_label_ids'])
            s5 = s1 + len(mlm_data)

            mlm_data_offsets.append([s1, s2, s3, s4, s5])
            mlm_data_sequences.extend(mlm_data)

            num_instances += 1

            if num_instances > 0 and num_instances % shard_size == 0:
                _save_shard()
        else:
            finished_worker_num += 1
            if finished_worker_num == num_workers:
                break

    if len(row_data_sequences) > 0:
        _save_shard()

    stat_send.send((num_instances, shard_id))
    instance_receiver.close()
    context.destroy()


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs_to_generate", type=int, default=3,
                        help="Number of epochs of preprocess to pregenerate")

    VerticalAttentionTableBertConfig.add_args(parser)
    args = parser.parse_args()

    table_bert_config = VerticalAttentionTableBertConfig.from_dict(vars(args))
    input_formatter = VerticalAttentionTableBertInputFormatter(table_bert_config)
    tokenizer = input_formatter.tokenizer

    with TableDatabase.from_jsonl(args.train_corpus, tokenizer=tokenizer) as table_db:
        args.output_dir.mkdir(exist_ok=True, parents=True)
        print(f'Num entries in database: {len(table_db)}', file=sys.stderr)

        # generate train and dev split
        example_indices = list(range(len(table_db)))
        shuffle(example_indices)
        dev_size = min(int(len(table_db) * 0.1), 100000)
        train_indices = example_indices[:-dev_size]
        dev_indices = example_indices[-dev_size:]

        # with (args.output_dir / 'config.json').open('w') as f:
        #     json.dump(vars(args), f, indent=2, sort_keys=True, default=str)
        table_bert_config.save(args.output_dir / 'config.json')

        (args.output_dir / 'train').mkdir(exist_ok=True)
        (args.output_dir / 'dev').mkdir(exist_ok=True)

        # generate dev preprocess first
        dev_file = args.output_dir / 'dev' / 'epoch_0'
        dev_metrics_file = args.output_dir / 'dev' / "epoch_0.metrics.json"
        generate_for_epoch(
            table_db,
            dev_indices, dev_file, dev_metrics_file,
            input_formatter,
            write_instance_to_file,
            args)

        for epoch in trange(args.epochs_to_generate, desc='Epoch'):
            gc.collect()
            epoch_filename = args.output_dir / 'train' / f"epoch_{epoch}"
            metrics_file = args.output_dir / 'train' / f"epoch_{epoch}.metrics.json"
            generate_for_epoch(
                table_db,
                train_indices, epoch_filename, metrics_file,
                input_formatter,
                write_instance_to_file,
                args
            )


if __name__ == '__main__':
    main()

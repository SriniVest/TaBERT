import json
import sh
from pathlib import Path
from typing import Dict
import unicodedata
import re


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    value = str(unicodedata.normalize('NFKD', value).encode('ascii', 'ignore'))
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)

    return value


def submit(args: Dict, nodes=4, constraint=None):
    sbatch_script = open('scripts/submit_tb_job.v2.sh').read()

    for arg, val in args.items():
        sbatch_script = sbatch_script.replace('${' + arg + '}', str(val))

    job_dir = Path('scripts/runs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    job_file_name = 'table_bert_' + '_'.join(slugify(str(x)[-32:]) for x in args.values()) + '.sh'
    job_file = job_dir / job_file_name
    job_file.write_text(sbatch_script)

    cmd_arg = []
    cmd_arg.extend(['--nodes', nodes])
    #cmd_arg.extend(['--partition', 'priority'])
    #cmd_arg.extend(['--time', '1:00:00'])
    #cmd_arg.extend(['--comment', '"urgent debug"'])
    if constraint:
        cmd_arg.extend(['--constraint', constraint])
        # cmd_arg.extend(['--comment', '"intern checkout 08/30"'])

    cmd_arg.append(str(job_file))

    cmd = sh.sbatch(*cmd_arg)
    print(job_file, '>>>', cmd)


if __name__ == '__main__':
    dataset_base_dir = Path('/private/home/pengcheng/Research/datasets/table_bert/')
    for dataset_name in [
       # 'tb_bindata0829_a7ee5a6c_1119183730869',
       # 'tb_bindata0829_a7ee5a6c_1119184004683',
       # 'tb_bindata0829_a7ee5a6c_1119184004689',
       # 'tb_bindata0829_a7ee5a6c_1119184004266',
       'tb_bindata0829_9f1338bc_1214231423821',
       'tb_bindata0829_9f1338bc_1214212818515'
    ]:
        for bert_model in ['bert-base-uncased']:
           for batch_size in [16]:
                for lr in ['2e-5', '4e-5']:  # 1e-5, 2e-5, 3e-5
                    for eps in [1e-8]:  # 1e-6,
                        for weight_decay in [0.01]:
                            for clip_norm in [1.0]:
                                    data_set_path = dataset_base_dir / dataset_name
                                    extra_config = {'base_model_name': bert_model}
                                    args = {
                                        'DATASET_PATH': str(data_set_path),
                                        'BATCH_SIZE': batch_size,
                                        'BERT_MODEL': bert_model,
                                        'LEARNING_RATE': lr,
                                        'EPS': eps,
                                        'WEIGHT_DECAY': weight_decay,
                                        'MAX_EPOCH': 10,
                                        'CLIP_NORM': clip_norm,
                                        'TABLE_BERT_EXTRA_CONFIG': "'" + json.dumps(extra_config) + "'"
                                    }

                                    nodes = 4
                                    constraint = None
                                    if bert_model == 'bert-large-uncased':
                                        nodes = 8

                                    submit(args, nodes=nodes, constraint=constraint)

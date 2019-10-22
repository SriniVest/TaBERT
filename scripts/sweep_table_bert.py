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
    sbatch_script = open('scripts/submit_tb_job.sh').read()

    for arg, val in args.items():
        sbatch_script = sbatch_script.replace('${' + arg + '}', str(val))

    job_dir = Path('scripts/runs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    job_file_name = 'table_bert_' + '_'.join(slugify(str(x)[-32:]) for x in args.values()) + '.sh'
    job_file = job_dir / job_file_name
    job_file.write_text(sbatch_script)

    cmd_arg = []
    cmd_arg.extend(['--nodes', nodes])
    if constraint:
        cmd_arg.extend(['--constraint', constraint])
        cmd_arg.extend(['--partition', 'priority'])
        cmd_arg.extend(['--comment', '"intern checkout 08/30"'])

    cmd_arg.append(str(job_file))

    cmd = sh.sbatch(*cmd_arg)
    print(job_file, '>>>', cmd)


if __name__ == '__main__':
    dataset_base_dir = Path('/private/home/pengcheng/Research/datasets/table_bert/')
    for dataset_name in [
        # 'tb_bindata_ctx128_pmsk_col0.1_pmsk_ctx0.15_tbmsk_column_ctxsmpl_concate_and_enumerate_nomaxlen_epoch15',
        # 'tb_bindata_ctx128_pmsk_col0.1_pmsk_ctx0.15_tbmsk_column_ctxsmpl_nearest_nomaxlen_epoch15',
        # 'tb_bindata_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_ctxsmpl_concate_and_enumerate_nomaxlen_epoch15',
        'tb_bindata0829_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_ctxsmpl_concate_and_enumerate_nomaxlen_epoch15',
        'tb_bindata0829_ctx256_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_ctxsmpl_concate_and_enumerate_nomaxlen_epoch15'
        # 'tb_bindata_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_ctxsmpl_nearest_nomaxlen_epoch15',
        # 'tb_bindata_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_token_ctxsmpl_concate_and_enumerate_nomaxlen_epoch15',
        # 'tb_bindata_ctx128_pmsk_col0.2_pmsk_ctx0.15_tbmsk_column_token_ctxsmpl_nearest_nomaxlen_epoch15',
        # 'tb_bindata_ctx128_pmsk_col0.3_pmsk_ctx0.15_tbmsk_column_ctxsmpl_concate_and_enumerate_nomaxlen_epoch15',
        # 'tb_bindata_ctx128_pmsk_col0.3_pmsk_ctx0.15_tbmsk_column_ctxsmpl_nearest_nomaxlen_epoch15'
    ]:
        for bert_model in ['bert-base-uncased']:
            for batch_size in [8]:
                for lr in [2e-5]:  # 1e-5, 3e-5
                    for eps in [1e-8, 1e-6]:  # 1e-6,
                        # if eps == 1e-8 and lr == 3e-5:
                        #     continue

                        data_set_path = dataset_base_dir / dataset_name
                        args = {
                            'DATASET_PATH': str(data_set_path),
                            'BATCH_SIZE': batch_size,
                            'BERT_MODEL': bert_model,
                            'LEARNING_RATE': lr,
                            'EPS': eps
                        }

                        nodes = 4
                        constraint = None
                        if bert_model == 'bert-large-uncased':
                            nodes = 8
                            constraint = 'volta32gb'

                        submit(args, nodes=nodes, constraint=constraint)

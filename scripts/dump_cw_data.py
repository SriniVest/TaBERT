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


def submit(args: Dict, nodes=1, constraint=None):
    sbatch_script = open('scripts/dump_cw_data.sh').read()

    for arg, val in args.items():
        sbatch_script = sbatch_script.replace('${' + arg + '}', str(val))

    job_dir = Path('scripts/runs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    job_file_name = 'table_bert_' + '_'.join(slugify(str(x)[-32:]) for x in args.values()) + '.sh'
    job_file = job_dir / job_file_name
    job_file.write_text(sbatch_script)

    cmd_arg = []
    # cmd_arg.extend(['--nodes', nodes])
    # cmd_arg.extend(['--partition', 'priority'])
    # cmd_arg.extend(['--comment', '"intern checkout 08/30"'])
    if constraint:
        cmd_arg.extend(['--constraint', constraint])

    cmd_arg.append(str(job_file))

    cmd = sh.sbatch(*cmd_arg)
    print(job_file, '>>>', cmd)


if __name__ == '__main__':
    dump_base_filename = Path('/private/home/pengcheng/Research/datasets/table_data/common_crawl.dump.0829.jsonl')
    for part in range(6):
        for chunk in ['0-1', '2-3', '4-5', '6-7', '8-9']:
            i, j = chunk.split('-')
            args = {
                'OUTPUT_FILE': str(dump_base_filename.with_suffix(f'.{part}{i}-{part}{j}.jsonl')),
                'FILTER': f'{part}[{chunk}].tar.gz'
            }

            submit(args)

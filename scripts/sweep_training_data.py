import sh, re
from pathlib import Path
from typing import Dict


def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    import unicodedata
    value = str(value)
    value = str(unicodedata.normalize('NFKD', value).encode('ascii', 'ignore'))
    value = re.sub('[^\w\s-]', '', value).strip().lower()
    value = re.sub('[-\s]+', '-', value)

    return value


def submit(args: Dict):
    sbatch_script = open('scripts/generate_train_data.sh').read()

    for arg, val in args.items():
        sbatch_script = sbatch_script.replace('${' + arg + '}', str(val))

    job_dir = Path('scripts/runs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    job_file_name = 'gen_data_' + '_'.join(slugify(x) for x in args.values()) + '.sh'
    job_file = job_dir / job_file_name
    job_file.write_text(sbatch_script)

    # cmd = sh.sbatch(str(job_file))
    cmd = sh.bash(str(job_file))
    print(cmd)


for max_context_len in [128, 256]:
    for table_mask_strategy in ['column']:  # ['column', 'column_token']:
        for context_sample_strategy in ['concate_and_enumerate']:  # ['nearest', 'concate_and_enumerate']:
            for masked_column_prob in [0.2]:  # [0.1, 0.2, 0.3]:
                for column_delimiter in ['[SEP]', '[unused0]']:
                    for cell_input_template in ["'column(value)(type)'", "'column:value(type)'", "'column|value|type'", "'column|type|value'"]:
                        setting = {
                            'max_context_len': max_context_len,
                            'table_mask_strategy': table_mask_strategy,
                            'context_sample_strategy': context_sample_strategy,
                            'masked_column_prob': masked_column_prob,
                            'column_delimiter': column_delimiter,
                            'cell_input_template': cell_input_template
                        }

                        submit(setting)

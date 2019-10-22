import sh
from pathlib import Path
from typing import Dict

# params = {
#     'table_mask_strategy': ['column', 'column_token'],
#     'context_sample_strategy': ['nearest', 'concate_and_enumerate'],
# }


def submit(args: Dict):
    sbatch_script = open('scripts/generate_train_data.sh').read()

    for arg, val in args.items():
        sbatch_script = sbatch_script.replace('${' + arg + '}', str(val))

    job_dir = Path('scripts/runs/')
    job_dir.mkdir(parents=True, exist_ok=True)

    job_file_name = 'gen_data_' + '_'.join(str(x) for x in args.values()) + '.sh'
    job_file = job_dir / job_file_name
    job_file.write_text(sbatch_script)

    cmd = sh.sbatch(str(job_file))
    print(cmd)


for max_context_len in [128, 256]:
    for table_mask_strategy in ['column']:  # ['column', 'column_token']:
        for context_sample_strategy in ['concate_and_enumerate']:  # ['nearest', 'concate_and_enumerate']:
            for masked_column_prob in [0.2]:  # [0.1, 0.2, 0.3]:
                # if table_mask_strategy == 'column_token' and masked_column_prob != 0.2:
                #     continue

                setting = {
                    'max_context_len': max_context_len,
                    'table_mask_strategy': table_mask_strategy,
                    'context_sample_strategy': context_sample_strategy,
                    'masked_column_prob': masked_column_prob
                }

                submit(setting)

"""Validate or run the paper's litmus tests from any working directory."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

import yaml

from framework.litmus_validation import validate_config
from framework.run_experiment import Experiment


ROOT = Path(__file__).resolve().parent


def prepare(mode):
    manifest = json.loads((ROOT / 'snapshot.json').read_text())
    for path, expected in manifest['files'].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'File checksum mismatch: {path}')
    batch = yaml.safe_load((ROOT / f'experiments/batches/{mode}.yaml').read_text())
    if batch['tests'] != manifest['tests']:
        raise ValueError('Batch test list differs from manifest')
    prepared = []
    for path in batch['tests']:
        config = yaml.safe_load((ROOT / path).read_text())
        validate_config(config)
        experiment = Experiment(str(ROOT / path), batch['iters'])
        # Pass validated configuration and generated CUDA directly to the runner.
        prepared.append((path, Experiment.from_prepared(
            config, experiment.name, experiment.source_code, batch['iters'])))
    return batch, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('short', 'full'))
    parser.add_argument('--check', action='store_true', help='CPU validation/rendering only; never submit jobs')
    parser.add_argument('--output', type=Path, help='New output directory (must not already exist)')
    args = parser.parse_args()
    batch, prepared = prepare(args.mode)
    print(f'Validated and rendered {len(prepared)} tests; {batch["iters"]} iterations, '
          f'{batch["max_wall_time"]} seconds per test, at most {batch["max_retry"]} attempts.')
    if args.check:
        return
    for executable in ('sbatch', 'squeue', 'scancel', 'conda'):
        if shutil.which(executable) is None:
            parser.error(f'{executable} is required; run on a configured SLURM submission host')
    if not os.environ.get('CONDA_PREFIX') or not os.environ.get('CONDA_PATH'):
        parser.error('Activate the supplied conda environment and set CONDA_PATH to the conda installation')
    os.environ.setdefault('CONDA_ENV', os.environ['CONDA_PREFIX'])
    if args.output:
        output = args.output.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / 'output').mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix=f'{args.mode}-', dir=ROOT / 'output'))
    shutil.copy(ROOT / 'snapshot.json', output / 'snapshot.json')
    (output / 'batch.yaml').write_text(yaml.safe_dump(batch, sort_keys=False))
    # The allocation limit is separate from the runner's all-attempt wall budget.
    script = (ROOT / 'framework/experiment_job.sh').read_text()
    script = script.replace('#SBATCH --time=00:15:00', '#SBATCH --time=01:00:00')
    dispatch = output / 'experiment_job.sh'
    dispatch.write_text(script)
    print(f'Results: {output}', flush=True)
    for source, experiment in prepared:
        saved_input = output / 'inputs' / source
        saved_input.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / source, saved_input)
        experiment.dispatch_script = str(dispatch)
        experiment.run(str(output), max_retry=batch['max_retry'], quiet=True,
                       max_wall_time=batch['max_wall_time'])
    print(f'Campaign finished. Inspect final_results.yaml and attempt logs under {output}.')


if __name__ == '__main__':
    main()

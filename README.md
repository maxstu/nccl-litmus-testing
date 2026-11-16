# NCCL Litmus Testing

A DSL, compiler, runner and collection of litmus tests for exploring NCCL concurrency behaviour.
The initial release contains the 147-test software snapshot used in our operational-semantics follow-up work.
Contributions of tests, tooling improvements and reproducible observations are welcome.

Experimental results are distributed as release assets, not stored in Git.
You can analyse downloaded evidence locally; running the tests requires a GPU cluster.

## Install the Python dependencies

From this directory, use Python 3.10 or later:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Analyse the released evidence (no GPUs required)

Download `nccl-litmus-testing-v1.0.0-evidence.zip` from the
[v1.0.0 release](https://github.com/maxstu/nccl-litmus-testing/releases/tag/v1.0.0)
and extract it into this directory.
It supplies `evidence/results.json`; that directory is ignored by Git.
Alternatively, download the complete artefact ZIP from the same release for source and evidence together.
Use the matching `v1.0.0` source when reproducing these results.

```sh
python analyse.py
# To inspect the test inventory without downloading evidence:
python analyse.py --inventory-only
```

The script reads `evidence/results.json` and writes CSV, Markdown and LaTeX tables to `output/analysis/`.
The evidence includes structured observations and attempt outcomes for the paper, but not raw logs.
See the release notes and generated analysis for the recorded results.

| Output | Contents |
| --- | --- |
| `litmus-suite.*` | Test counts by topic. |
| `campaign-outcomes.*` | The paper's four outcome categories. |
| `campaign-outcomes-detailed.*` | Detailed outcomes, including stalled attempts. |
| `campaign-completion.*` | Iteration and wall-time stopping conditions. |
| `audit.json` | Per-test classifications and supporting observations. |

## Check the test configurations (no GPUs required)

```sh
python run.py short --check
python run.py full --check
```

These commands verify file checksums, validate all 147 configurations and generate CUDA sources.
They use the Python environment above and do not submit jobs or execute GPU code.

## Set up a cluster environment

Requirements:

- Linux x86-64 SLURM cluster with three compatible NVIDIA GPUs on one node.
- A compatible NVIDIA driver and conda.
- Shared artefact/output paths between the submission host and compute node.
- Compute-node access to a dynamically assigned TCP port on the submission host, using its hostname.

```sh
conda env create -f config/env.yaml
conda activate nccl
export CONDA_PATH="$(conda info --base)"
export CONDA_ENV="$CONDA_PREFIX"

# Set these if required by your cluster:
export SBATCH_PARTITION=your_gpu_partition
export SBATCH_ACCOUNT=your_allocation
```

The conda environment provides CUDA 13.3 development tools, NCCL 2.32.3.1,
the host compiler and Python/C++ dependencies; a separate CUDA toolkit is unnecessary.
The harness uses `sbatch`, `squeue`, `scancel` and `srun`, and requests three GPUs for each job.
Compilation and execution take place on the allocated GPU node.

## Run the experiments

```sh
python run.py short
python run.py full
```

| Mode | Iteration target per test | Wall-time budget per test | Maximum attempts |
| --- | ---: | ---: | ---: |
| `short` | 100 | 10 minutes | 3 |
| `full` | 1,000,000 | 3,600 seconds | 3 |

The launcher runs tests sequentially and must remain running; use `tmux` for long campaigns.
Each invocation creates an output directory; use `--output /shared/path/new-directory` to choose its location.
A test stops at the first limit reached, with wall time including queueing, compilation and retries,
so short mode may finish without executing on a busy cluster.
The supplied one-hour evidence trials use a 10,000-attempt cap rather than these presets' three attempts.
If the launcher is forcibly killed, inspect `squeue` and cancel remaining jobs with `scancel JOB_ID`.

## Inspect or analyse your results

Pass the output directory printed by `run.py`:

```sh
python analyse.py --local-runs output/YOUR-RUN --output output/analysis
```

Each test's run directory contains generated CUDA, attempt logs and `final_results.yaml`.
Analysis accepts any iteration count, including short runs, and writes tables and a per-test audit.
Completion is assessed against the run’s configured target: 100 iterations for short mode or one million for full mode.
Partial runs still appear in the audit, with their stopping condition.
Wait for `run.py` to finish before analysing its output.

## Files and DSL

| Path | Contents |
| --- | --- |
| `run.py`, `analyse.py` | Execution and analysis entry points. |
| `experiments/<topic>/*.yaml` | The 147 tests. |
| `experiments/batches/` | Short/full test lists and execution limits. |
| `evidence/results.json` | Downloaded release evidence; excluded from Git. |
| `config/env.yaml` | Cluster software environment. |
| `snapshot.json` | Test list and file checksums. |
| `framework/compiler/` | DSL parser and CUDA generator. |
| `framework/templates/` | CUDA/NCCL runtime template. |
| `framework/run_experiment.py`, `framework/experiment_job.sh` | Submission and execution harness. |
| `framework/result_analysis.py`, `framework/litmus_validation.py` | Result interpretation and input validation. |
| `framework/protobuf/` | Observation protocol and bindings. |

Tests specify the program, GPU resources, communicators, initial buffers and expected outcomes in YAML.
The DSL uses zero-based identifiers: `thread` defines a host thread, `group` wraps NCCL group calls,
and `send<0,1>[0](0,0,float,buffer_size)` sends rank 0's buffer 0 to rank 1 using communicator 0 and stream 0.
`sync(rank,stream)` waits for stream completion; `barrier({0,1})` synchronises host threads.
See the test files for examples and the compiler for the full syntax.

## Contributing

Open an issue or pull request with a small test, a tooling change or a reproducible observation.
For GPU observations, include the test, NCCL/CUDA versions, hardware, execution settings and relevant logs.
Distinguish expected outcomes and model predictions from observed GPU behaviour.
CPU validation and CUDA source generation do not establish GPU correctness.
Avoid committing generated sources, logs or results; attach evidence separately.

`snapshot.json` records the initial cohort and checksums used by the validation commands.
Changes to tracked snapshot files must deliberately update their checksums;
new tests must also update the manifest and both batch selections.
Keep test expectations unchanged unless the proposed change explains the scientific reason.

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).

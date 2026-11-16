from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from framework.protobuf import experiment_pb2
from tqdm import tqdm
from typing import Union
import argparse
import itertools
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import yaml
import zmq

def compile_dsl(*args, **kwargs):
    """Load the local-only compiler when preparing an experiment."""
    from framework.compiler import compile_dsl as compile_program
    return compile_program(*args, **kwargs)

class RunCancelled(BaseException):
    """Explicit cancellation must escape the ordinary attempt retry handler."""

class WallTimeLimit(BaseException):
    """The per-test wall-time budget expired, independently of execution stalls."""

def validate_max_wall_time(value):
    if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 1):
        raise ValueError("max_wall_time must be a positive integer number of seconds")
    return value

def check_cancelled():
    cancel_file = os.environ.get("NCCL_CANCEL_FILE")
    if cancel_file and Path(cancel_file).exists():
        raise RunCancelled("Run cancellation requested")

def on_exit(finalizer_func):
    """
    A decorator that ensures a finalizer function is called 
    when the decorated method exits (via return, break, or exception).
    """
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            finally:
                finalizer_func(self) 
        return wrapper
    return decorator

class JobStatus(Enum):
    READY = "R"
    PENDING = "PD"
    CONFIGURING = "CF"
    COMPLETING = "CG"
    CANCELLED = "CA"
    MISSING = "?"

@dataclass(frozen=True)
class ExitStatus(ABC):
    @abstractmethod
    def message(self) -> str:
        return "Not implemented for base class."

@dataclass(frozen=True)
class ExitSuccess(ExitStatus):
    def message(self) -> str:
        return "Experiment finished successfully."

@dataclass(frozen=True)
class ExitTimeout(ExitStatus):
    seconds: int
    def message(self) -> str:
        return f"Job was timed out after {self.seconds} seconds."

@dataclass(frozen=True)
class ExitMissing(ExitStatus):
    def message(self) -> str:
        return "Job disappeared from queue (cancelled by user or slurm issue)."

@dataclass(frozen=True)
class ExitError(ExitStatus):
    reason: str
    def message(self) -> str:
        return f"Got an unexpected error (likely programmer error): {self.reason}"

@dataclass
class JobResult():
    status: ExitStatus
    results: dict
    
class BadExperimentConfig(Exception):
    pass

class UnexpectedMessageException(Exception):
    pass 

@dataclass
class Results():
    progress: int
    observations: dict
    errors: list

class SlurmJob:
    def __init__(self, dispatch_script, output_dir, progress, monitor_address, attempt, gpu_count=3):
        self.dispatch_script = dispatch_script
        self.output_dir = output_dir
        self.progress = progress
        self.monitor_address = monitor_address
        self.attempt = attempt
        self.gpu_count = gpu_count
        self.job_id = None
        self.successful = False
        self.log = None

    def __enter__(self):
        check_cancelled()
        self.log = Path(self.output_dir) / f"attempt_{self.attempt}.log"

        command = subprocess.run(
            ["sbatch", 
             f"--output={self.log}", 
             f"--gres=gpu:{self.gpu_count}",
             self.dispatch_script, str(self.output_dir), 
             str(self.progress), self.monitor_address], 
            capture_output=True, text=True
        )

        if command.returncode != 0:
            raise RuntimeError(f"Error submitting job: {command.stderr}")
        
        job_id_match = re.search(r'\d+', command.stdout)
        if not job_id_match:
            raise RuntimeError(f"Could not parse Job ID from: {command.stdout}")

        self.job_id = job_id_match.group()
        journal = os.environ.get("NCCL_JOB_JOURNAL")
        if journal:
            try:
                with open(journal, "a") as f:
                    f.write(json.dumps({"job_id": self.job_id,
                                        "test_id": Path(self.output_dir).name,
                                        "attempt": self.attempt}) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                check_cancelled()
            except BaseException:
                subprocess.run(["scancel", self.job_id])
                raise
        print(f"Job {self.job_id} submitted successfully. Logging to {self.log}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.job_id and not self.successful:
            print(f"Cancelling SLURM job {self.job_id}")
            subprocess.run(["scancel", self.job_id])

    def success(self):
        self.successful = True

class ExperimentMonitor():
    deadline = None

    def check_deadline(self):
        check_cancelled()
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise WallTimeLimit("Per-test wall-time limit reached")

    def poll(self, milliseconds):
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                raise WallTimeLimit("Per-test wall-time limit reached")
            milliseconds = min(milliseconds, max(0, int(remaining * 1000)))
        result = self.poller.poll(milliseconds)
        if self.deadline is not None and time.monotonic() >= self.deadline:
            raise WallTimeLimit("Per-test wall-time limit reached")
        return result
    context = None
    socket = None
    poller = None
    results = None
    progress_bar = None

    @staticmethod
    def _state_to_hashable(state_pb) -> tuple:
        ranks = []
        for rank in state_pb.ranks:
            buffers = []
            for buf in rank.buffers:
                elements = []
                for elem in buf.elements:
                    count_val = elem.count if elem.HasField('count') else None
                    elements.append((elem.value, count_val))
                buffers.append(tuple(elements))
            ranks.append(tuple(buffers))
        return tuple(ranks)
    
    def __init__(self, start_progress=0) -> None:
        self.phase = "handshake"
        self.context = zmq.Context()
        self.poller = zmq.Poller()
        self.socket = self.context.socket(zmq.PAIR)
        self.socket.bind(f"tcp://*:0")
        self.poller.register(self.socket, zmq.POLLIN)
        self.results = Results(start_progress, Counter(), [])

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar:
            self.progress_bar.close()
        self.socket.close(linger=0)
        self.context.term()

    def get_socket_address(self) -> str:
        endpoint = self.socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')
        bound_port = endpoint.rsplit(':', 1)[-1]
        host_ip = socket.gethostbyname(socket.gethostname())
        return f"tcp://{host_ip}:{bound_port}"

    def job_status(self, job_id) -> JobStatus:
        result = subprocess.run(
            ['squeue', '-j', str(job_id), '-h', '-o', '%t'], 
            capture_output=True, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(f"squeue failed: {result.stderr.strip()}")
        status_str = result.stdout.strip()
        
        if not status_str:
            return JobStatus.MISSING
        
        try:
            return JobStatus(status_str)
        except ValueError:
            raise RuntimeError(f"Unrecognised SLURM state: {status_str}")

    def render_bar(current, total, length=40):
        # Calculate mechanics
        fraction = current / total
        arrow = int(fraction * length) * '█'
        padding = (length - len(arrow)) * '-'
        percent = f"{100 * fraction:.1f}%"
        
        # Render to terminal
        sys.stdout.write(f"\r|{arrow}{padding}| {current}/{total} ({percent})")
        sys.stdout.flush()

    def handshake(self, job_id: int) -> bool:
        print("Waiting for CUDA job handshake... ")

        check_freq = 1 # second
        missing_limit = 60 # seconds
        missing_count = 0

        # Wait for job to appear in the queue, which may normally be slightly delayed
        while self.job_status(job_id) == JobStatus.MISSING and \
              missing_count * check_freq < missing_limit:
            missing_count += 1
            self.check_deadline()
            time.sleep(check_freq)

        # Wait for slurm to start running the job
        while self.job_status(job_id) in (JobStatus.PENDING, JobStatus.CONFIGURING):
            self.check_deadline()
            time.sleep(check_freq)

        # Cold CUDA compilation and shared-filesystem reads can exceed a minute.
        # This startup allowance is separate from the execution-stall timeout.
        for _ in range(300):
            self.check_deadline()
            socks = dict(self.poll(1000))
            if self.socket in socks:
                msg = self.socket.recv_string()
                if msg == "READY":
                    self.socket.send_string("OK")
                    print("Handshake complete.")
                    return True
                break
            if self.job_status(job_id) == JobStatus.MISSING:
                break

        # If we drop down here, it either timed out or got a bad message
        if self.job_status(job_id) == JobStatus.MISSING:
            print("Failed (Job aborted or went missing).")
        else:
            print("Failed (Timeout waiting for READY).")
            
        return False

    def watch(self, job_id: str, timeout: int, iters: int, quiet: bool = False) -> JobResult:
        if not self.handshake(job_id):
            return JobResult(ExitError("Handshake failed"), self.results)
        self.phase = "startup"
        
        print(f"Monitoring Job {job_id}...")
        if not quiet:
            self.progress_bar = tqdm(total=iters, desc=f"Experiment Progress", unit="iter")
            self.progress_bar.update(self.results.progress)
        else:
            print(f"Starting at iteration {self.results.progress}/{iters}")

        running = False
        warmup_iters = 3
        started = time.monotonic()
        while True:
            self.check_deadline()
            socks = dict(self.poll(1000 * timeout))
            if self.socket in socks:
                message = experiment_pb2.Message()
                message.ParseFromString(self.socket.recv())
                if not message.HasField("update"):
                    return JobResult(ExitError("Missing update"), self.results)
                progress = message.update.iteration
                if progress < self.results.progress or progress > iters:
                    return JobResult(ExitError("Invalid iteration checkpoint"), self.results)
                if message.HasField("result"):
                    count = sum(obs.count for obs in message.result.observations)
                    if any(obs.count <= 0 for obs in message.result.observations) or \
                            count + len(message.result.errors) != progress - self.results.progress:
                        return JobResult(ExitError("Result count does not match checkpoint"), self.results)
                    for obs in message.result.observations:
                        self.results.observations[self._state_to_hashable(obs.state)] += obs.count
                    self.results.errors.extend(message.result.errors)
                    delta = progress - self.results.progress
                    self.results.progress = progress
                    if self.progress_bar:
                        self.progress_bar.update(delta)
                # Heartbeats demonstrate liveness, but cannot commit unsent results.
                if message.update.status == experiment_pb2.Update.WAITING:
                    self.phase = "execution"
                if message.update.status == experiment_pb2.Update.RUNNING:
                    running = True
                    self.phase = "execution"
                if message.update.status == experiment_pb2.Update.COMPLETE:
                    if self.results.progress != iters:
                        return JobResult(ExitError("Completion without all results"), self.results)
                    return JobResult(ExitSuccess(), self.results)
                continue
            state = self.job_status(job_id)
            if state in (JobStatus.MISSING, JobStatus.COMPLETING, JobStatus.CANCELLED):
                return JobResult(ExitMissing(), self.results)
            if state != JobStatus.READY:
                return JobResult(ExitError(f"Unexpected execution state: {state.value}"), self.results)
            if not running and warmup_iters > 0:
                warmup_iters -= 1
                continue
            return JobResult(ExitTimeout(int(time.monotonic() - started)), self.results)

class Experiment():
    name = None
    config = None
    source_code = None
    output_dir = None
    template = None
    dispatch_script = str(Path(__file__).resolve().with_name("experiment_job.sh"))
    iters = None
    results = None
    execution_status = "not_started"
    attempts = None
    preserve_final = False
    deadline = None
    started_at = None
    max_wall_time = None

    def source_path(self) -> str:
        return Path(self.output_dir)/"experiment.cu"

    def __init__(self, config_file: str, iters: int) -> None:
        from jinja2 import Environment, FileSystemLoader

        # Load the config from specified file
        if not os.path.isfile(config_file):
            raise FileNotFoundError(f"Could not find experiment file: {config_file}")

        with open(config_file, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                raise BadExperimentConfig(f"Not a valid yaml file: {config_file}")

        self.name = Path(config_file).stem
        self.input_path = Path(config_file).resolve()
        self.results = Results(0, Counter(), [])

        # Create CUDA source code
        program = self.config['program']
        required_gpus = self.config['required_gpus']
        buffers_per_gpu = self.config['buffers_per_gpu']
        buffer_size = self.config['buffer_size']
        
        if 'streams_per_gpu' not in self.config:
             raise ValueError(f"No stream count specified for {self.name} (missing 'streams_per_gpu' in YAML).")
        streams_per_gpu = self.config['streams_per_gpu']
        # CUDA commands and positional oracles require validated allocations even
        # for direct Experiment construction outside the normal preparation path.
        from framework.compiler.parser import program as parse_program
        from framework.compiler.dsl_ast import Copy, Fill
        from framework.compiler.events import Event, Wait, StreamWait, event_layout
        ast = parse_program.parse(program)
        if ('events_per_gpu' in self.config or 'elementwise_results' in self.config
                or any(isinstance(command, (Copy, Fill, Event, Wait, StreamWait))
                       for thread in ast.threads for command in thread.commands)):
            from framework.litmus_validation import validate_config, validate_elementwise_results
            validate_config(self.config)

        initial_values_raw = self.config['initial_values']
        initial_values = str(initial_values_raw).replace("[", "{").replace("]", "}")
        
        # Safely attempt to get config_iters
        config_iters = self.config.get('iterations')

        # Prioritize CLI `iters`, then YAML `config_iters`
        if iters is not None:
            self.iters = iters
        elif config_iters is not None:
            self.iters = config_iters
        else:
            raise ValueError(f"No iteration count specified for {self.name} (neither in CLI nor YAML).")
        
        # Compile the litmus and fill the template
        litmus_test = compile_dsl(program, streams_per_gpu, self.config.get('communicators'))
        
        communicators = self.config.get('communicators')
        if not communicators:
             raise ValueError(f"No communicator layout specified for {self.name} (missing 'communicators' in YAML).")
        
        num_cliques = len(communicators)

        template = self.template or Environment(loader=FileSystemLoader(str(Path(__file__).resolve().parent / "templates"))).get_template("experiment.cu.j2")
        self.source_code = template.render(
            program=litmus_test, 
            iterations=self.iters, 
            required_gpus=required_gpus, 
            buffer_size=buffer_size, 
            buffers_per_gpu=buffers_per_gpu, 
            streams_per_gpu=streams_per_gpu, 
            initial_values=initial_values,
            communicators=communicators,
            num_cliques=num_cliques,
            event_layout=event_layout(ast),
            elementwise_results=(validate_elementwise_results(self.config)
                                 if 'elementwise_results' in self.config else None),
        ) 
        # Template-only control lines can leave indented blank lines in saved
        # translations. Keep generated source clean without changing CUDA code.
        self.source_code = '\n'.join(line.rstrip() for line in self.source_code.splitlines()) + '\n'

    @classmethod
    def from_prepared(cls, config: dict, name: str, source: str, iters: int):
        """Consume locally generated CUDA without a compiler or template remotely."""
        if Path(name).name != name or name in ("", ".", ".."):
            raise ValueError("Prepared experiment name must be a directory basename")
        if not isinstance(iters, int) or isinstance(iters, bool) or iters < 1:
            raise ValueError("Iteration count must be a positive integer")
        experiment = cls.__new__(cls)
        experiment.config = config
        experiment.name = name
        experiment.source_code = source
        experiment.iters = iters
        experiment.results = Results(0, Counter(), [])
        return experiment

    @on_exit(lambda self: self.save_final_results())
    def run(self, output_dir: str, max_retry: int, quiet: bool = False, resume: bool = False, max_wall_time: int | None = None) -> None:
        if hasattr(self, 'input_path'):
            from framework.experiment_catalog import ensure_runnable
            ensure_runnable(self.input_path)
        self.output_dir = Path(output_dir)/self.name

        # Resume logic
        if resume and (self.output_dir / "final_results.yaml").exists():
            self.preserve_final = True
            print(f"Skipping {self.name} (already completed).")
            return

        self.max_wall_time = validate_max_wall_time(max_wall_time)
        self.started_at = time.monotonic()
        self.deadline = self.started_at + max_wall_time if max_wall_time is not None else None
        self.preserve_final = False
        self.attempts = []
        self.execution_status = "running"

        # Prepare output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for item in self.output_dir.iterdir():
            if item.name == "final_results.yaml":
                continue # Keep results if we are somehow here, though resume should have caught it
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()

        # Copy protobuf files to the work directory
        shutil.copy(Path(__file__).resolve().parent / "protobuf" / "experiment.pb.cc", self.output_dir)
        shutil.copy(Path(__file__).resolve().parent / "protobuf" / "experiment.pb.h", self.output_dir)

        with self.source_path().open(mode='w') as f:
            f.write(self.source_code)

        print(f"Running {self.name}")

        for attempt in range(max_retry):
            if self.deadline is not None and time.monotonic() >= self.deadline:
                self.execution_status = "wall_time_exhausted"
                return
            try:
                check_cancelled()
            except RunCancelled:
                self.execution_status = "cancelled"
                raise
            print(f"Attempt {attempt+1} / {max_retry}")
            results = self.run_attempt(attempt, quiet)

            if not results:
                if self.execution_status == "wall_time_exhausted":
                    return
                continue

            self.results.observations.update(results.observations)
            self.results.errors.extend(results.errors)
            self.results.progress = results.progress

            if self.execution_status == "wall_time_exhausted":
                return

            print(f"Cumulative progress: {self.results.progress}/{self.iters}")

            if self.results.progress >= self.iters:
                self.execution_status = "completed"
                print(f"Experiment {self.name} completed successfully.")
                return

        self.execution_status = ("wall_time_exhausted" if self.deadline is not None
                                 and time.monotonic() >= self.deadline else "retry_exhausted")

    def run_attempt(self, attempt: int, quiet: bool = False):
        monitor = None
        gpu_count = max(3, self.config.get('required_gpus', 3))
        outcome = {"attempt": attempt, "phase": "submission", "outcome": "error",
                   "requested_gpus": gpu_count}
        try:
            with ExperimentMonitor(self.results.progress) as monitor, SlurmJob(
                    self.dispatch_script, self.output_dir, self.results.progress,
                    monitor.get_socket_address(), attempt, gpu_count=gpu_count) as job:
                outcome["job_id"] = job.job_id
                monitor.deadline = self.deadline
                job_result = monitor.watch(job.job_id, 10, self.iters, quiet)
                outcome.update(phase=monitor.phase, outcome=type(job_result.status).__name__,
                               message=job_result.status.message())
                print(job_result.status.message())
                if isinstance(job_result.status, ExitSuccess):
                    job.success()
                return job_result.results
        except WallTimeLimit as exc:
            self.execution_status = "wall_time_exhausted"
            outcome.update(outcome="WallTimeLimit", phase=monitor.phase if monitor else "submission",
                           message=str(exc))
            return monitor.results if monitor else None
        except RunCancelled:
            self.execution_status = "cancelled"
            outcome.update(outcome="cancelled", phase=monitor.phase if monitor else "submission")
            if monitor:
                self.results.observations.update(monitor.results.observations)
                self.results.errors.extend(monitor.results.errors)
                self.results.progress = monitor.results.progress
            raise
        except Exception as exc:
            outcome.update(message=str(exc), phase=monitor.phase if monitor else "submission")
            print(f"Attempt failed with exception: {exc}")
            return monitor.results if monitor else None
        finally:
            if self.attempts is None:
                self.attempts = []
            self.attempts.append(outcome)

    class _Conformance(Enum):
        ACCEPTED = "yes"
        REJECTED = "no"
        SPEC_MALFORMED = "malformed conformance spec in yaml"
        SPEC_MISSING = "ignored (no spec given)"
        WILDCARD = "ignored (wildcard)"

    def _check_conformance(self, state_tuple, acceptable_results):
        if not acceptable_results:
            return self._Conformance.SPEC_MISSING
        outcomes = [self._check_single_outcome(state_tuple, [spec]) for spec in acceptable_results]
        for accepted in (self._Conformance.WILDCARD, self._Conformance.ACCEPTED):
            if accepted in outcomes:
                return accepted
        if self._Conformance.SPEC_MALFORMED in outcomes:
            return self._Conformance.SPEC_MALFORMED
        return self._Conformance.REJECTED

    def _check_single_outcome(self, state_tuple: tuple, acceptable_results: list | None) -> _Conformance:
        if not acceptable_results:
            return self._Conformance.SPEC_MISSING
        
        for spec in acceptable_results:
            if spec == "*":
                return self._Conformance.WILDCARD
            
            if len(state_tuple) != len(spec):
                return self._Conformance.SPEC_MALFORMED
                
            for r in range(len(state_tuple)):
                if spec[r] == "*":
                    continue # spec says the data in this rank is irrelevant
                
                if isinstance(spec[r], list) and len(state_tuple[r]) != len(spec[r]):
                    return self._Conformance.SPEC_MALFORMED
                elif not isinstance(spec[r], list) and spec[r] != "*":
                     return self._Conformance.SPEC_MALFORMED
                    
                for b in range(len(state_tuple[r])):
                    if spec[r][b] == "*":
                        continue # spec says the data in this buffer is irrelevant

                    match spec[r][b]:
                        case str() if spec[r][b] == "*":
                            continue # wildcard
                        case float(val) | int(val):
                            allowed_vals = {float(val)}
                        case str(val):
                            try:
                                allowed_vals = {float(val)}
                            except ValueError:
                                return self._Conformance.SPEC_MALFORMED
                        case list(buf):
                            try:
                                allowed_vals = {float(x) for x in buf}
                            except ValueError:
                                return self._Conformance.SPEC_MALFORMED
                        case _:
                            return self._Conformance.SPEC_MALFORMED

                    # state_tuple[r][b] is a tuple of (value, count) 
                    buffer_vals = set(elem[0] for elem in state_tuple[r][b])
                    
                    if not buffer_vals.issubset(allowed_vals):
                        return self._Conformance.REJECTED
                    
        return self._Conformance.ACCEPTED

    def _format_state(self, state_tuple: tuple) -> str:
        lines = ["{"]
        for r_idx, r in enumerate(state_tuple):
            buf_strs = []
            for b in r:
                el_strs = []
                for val, count in b:
                    # If count is None, it was a Unique element. Otherwise, Run/Counter.
                    if count is None:
                        el_strs.append(f"{val}")
                    else:
                        el_strs.append(f"({val}: {count})")
                buf_strs.append("[" + ", ".join(el_strs) + "]")
            
            rank_str = f"    Rank {r_idx}: [" + ", ".join(buf_strs) + "]"
            lines.append(rank_str)
        lines.append("}")
        return "\n".join(lines)

    def save_final_results(self):
        if self.preserve_final or not self.output_dir:
            return
        def multiline_presenter(dumper, data):
            if '\n' in data:
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)
        
        yaml.add_representer(str, multiline_presenter)

        acceptable_results = self.config.get('acceptable_results')
        expected_deadlock = self.config.get('expected_deadlock', False)
        expected_ub = self.config.get('expected_undefined_behaviour',
                                      self.config.get('expected_undefined_behavior', False))

        formatted_observations = []
        for state_tuple, count in self.results.observations.most_common():
            obs_dict = {
                "count": count, 
                "state": self._format_state(state_tuple)
            } 
            
            # Check if the result was expected
            obs_dict["conforms"] = self._check_conformance(state_tuple, acceptable_results).value
                
            formatted_observations.append(obs_dict)

        # Overall experiment conformance
        if expected_deadlock:
            if self.results.progress == 0 and any(
                    a["outcome"] == "ExitTimeout" and a["phase"] == "execution"
                    for a in (self.attempts or [])):
                experiment_conforms = "yes (deadlocked as expected)"
            else:
                experiment_conforms = ("no (completed but deadlock was expected)"
                                       if self.results.progress else
                                       "inconclusive (no execution deadlock evidence)")
        elif expected_ub:
            experiment_conforms = ("yes (undefined behaviour as expected)"
                                   if any(a["phase"] == "execution" for a in (self.attempts or []))
                                   else "inconclusive (test execution not established)")
        else:
            if self.results.progress < self.iters:
                experiment_conforms = "no (deadlocked or failed prematurely)"
            elif self.results.errors:
                experiment_conforms = "no (runtime errors reported)"
            elif not acceptable_results:
                experiment_conforms = ("observed (no outcome contract)"
                                       if self.config.get('observation_only', False)
                                       else "ignored (no spec given)")
            else:
                # Check if all observations conform
                all_obs_conform = all(obs["conforms"] == "yes" for obs in formatted_observations)
                if all_obs_conform:
                    experiment_conforms = "yes"
                else:
                    experiment_conforms = "no (some observations do not conform)"

        final_data = {
            "name": self.name,
            "execution_status": self.execution_status,
            "attempts": self.attempts or [],
            "max_wall_time": self.max_wall_time,
            "elapsed_seconds": (time.monotonic() - self.started_at
                                if self.started_at is not None else None),
            "total_iterations": self.results.progress,
            "expected_deadlock": expected_deadlock,
            "expected_undefined_behaviour": expected_ub,
            "observation_only": self.config.get('observation_only', False),
            "experiment_conforms": experiment_conforms,
            "observations": formatted_observations,
            "errors": self.results.errors
        }
        
        result_file = Path(self.output_dir) / "final_results.yaml"
        
        temporary = result_file.with_suffix(".yaml.tmp")
        with open(temporary, 'w') as f:
            yaml.dump(final_data, f, sort_keys=False)
        temporary.replace(result_file)

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run NCCL experiments and litmus tests.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Run a single experiment:
    python3 -m framework.run_experiment experiments/p2p/simple.yaml

  Run the experiments/batches/hand_picked.yaml batch:
    python3 -m framework.run_experiment

  Run a batch of experiments using a configuration file:
    python3 -m framework.run_experiment --config batch_run.yaml

  Run in the background with quiet mode and resume capability:
    nohup python3 -m framework.run_experiment --config batch_run.yaml -q -r > session.log 2>&1 &

Configuration File Format (YAML):
  output: "./results"   # Optional: Output directory
  iters: 1000           # Optional: Number of iterations per test
  quiet: true           # Optional: Suppress interactive progress bars
  resume: true          # Optional: Skip already finished tests
  tests:                # Required: Nonempty list of tests to run
    - experiments/p2p/simple.yaml
    - experiments/dependencies/p2p_causality.yaml
"""
    )

    parser.add_argument("path", nargs="?", default=None, help="Path to a single experiment YAML file. If omitted, the hand-picked batch is run.")
    parser.add_argument("-o", "--output", type=str, default=None, help="Directory to store results (default: ./output)")
    parser.add_argument("-i", "--iters", type=int, default=None, help="Number of iterations for each experiment, overriding values in experiment YAMLs.")
    parser.add_argument("-c", "--config", type=str, default=None, help="Path to a batch configuration YAML file.")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress interactive progress bars (recommended for background runs).")
    parser.add_argument("-r", "--resume", action="store_true", help="Skip experiments that already have a 'final_results.yaml' in the output directory.")

    args = parser.parse_args(argv)

    if args.iters is not None and args.iters < 1:
        parser.error('--iters must be positive')

    default_batch = not args.config and not args.path
    if default_batch:
        args.config = str(Path(__file__).resolve().parents[1] / 'experiments/batches/hand_picked.yaml')

    # Default values
    output_dir = "./output"
    iterations = None
    tests_to_run = []
    quiet = args.quiet
    resume = args.resume
    max_retry = 3
    max_wall_time = None

    if args.config:
        if not os.path.isfile(args.config):
            print(f"Error: Configuration file {args.config} not found.")
            sys.exit(1)
        with open(args.config, 'r') as f:
            try:
                run_config = yaml.safe_load(f)
                if not isinstance(run_config, dict):
                    raise ValueError("Config file must be a dictionary")
                
                if not isinstance(run_config.get('tests'), list) or not run_config['tests']:
                    raise ValueError('Explicit batch must contain a nonempty tests list')

                # Check for conflicts
                conflicts = []
                if args.path and 'tests' in run_config:
                    conflicts.append("test path(s)")
                if args.output and 'output' in run_config and not default_batch:
                    conflicts.append("output directory")
                if args.quiet and 'quiet' in run_config:
                    conflicts.append("quiet mode")
                if args.resume and 'resume' in run_config:
                    conflicts.append("resume mode")
                
                if conflicts:
                    print(f"Error: Conflict detected! The following options were specified both on the command line and in the config file: {', '.join(conflicts)}.")
                    sys.exit(1)

                output_dir = run_config.get('output', output_dir)
                iterations = run_config.get('iters', iterations)
                tests_to_run = run_config.get('tests', [])
                quiet = run_config.get('quiet', quiet)
                resume = run_config.get('resume', resume)
                max_retry = run_config.get("max_retry", 3)
                if isinstance(max_retry, bool) or not isinstance(max_retry, int) or max_retry < 1:
                    raise ValueError("max_retry must be a positive integer")
                max_wall_time = validate_max_wall_time(run_config.get("max_wall_time"))
            except (yaml.YAMLError, ValueError) as exc:
                print(f"Error parsing configuration file: {exc}")
                sys.exit(1)

    # CLI values override/set if not already set by config (and no conflict)
    if args.output:
        output_dir = args.output
    if args.iters:
        iterations = args.iters
    if args.path:
        tests_to_run = [args.path]

    # Run the experiments
    for test_path in tests_to_run:
        print(f"--- Processing {test_path} ---")
        try:
            experiment = Experiment(test_path, iterations)
            experiment.run(output_dir, max_retry=max_retry, quiet=quiet, resume=resume,
                           max_wall_time=max_wall_time)
        except (ValueError, FileNotFoundError) as e:
            print(f"Skipping {test_path}: {e}")

    print("All done!")


if __name__ == "__main__":
    main()

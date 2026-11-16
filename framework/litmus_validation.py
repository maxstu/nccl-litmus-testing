"""CPU-only resource validation and CUDA source rendering for litmus tests.

Run from the repository root with ``python -m framework.litmus_validation``.
Optional paths select individual YAML files or suite directories. With no paths,
all litmus YAMLs under experiments/ are checked, excluding batches.

Checks preserve intentional operation mismatches, aliases, races and deadlocks.
They do not establish semantic predictions or GPU behaviour. Rendering never
calls Experiment.run(), compiles CUDA, or submits work to the cluster.
"""

import argparse
import math
from pathlib import Path
import sys

import yaml

from framework.compiler.dsl_ast import (
    Barrier, AllGather, AllToAll, BufferSize, BufferType, Collective, Gather, Group,
    PointToPoint, ReduceScatter, Rooted, Scatter, Sync, Copy, Fill, Skip, RootRole, PeerRole, Broadcast, Reduce,
)
from framework.compiler.events import Event, Wait, StreamWait, event_layout
from framework.compiler.parser import program
from framework.run_experiment import Experiment


def require(condition, message):
    if not condition:
        raise ValueError(message)


def is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def is_number(value):
    return is_integer(value) or (isinstance(value, float) and math.isfinite(value))


def check_index(value, size, label):
    require(is_integer(value) and 0 <= value < size, f"{label}: index {value!r} outside [0,{size})")


def check_matrix(value, ranks, buffers, label, check_entry, allow_rank_wildcard=False):
    require(isinstance(value, list) and len(value) == ranks, f"{label}: expected {ranks} rank rows")
    for rank, row in enumerate(value):
        if allow_rank_wildcard and row == "*":
            continue
        require(isinstance(row, list) and len(row) == buffers, f"{label}[{rank}]: expected {buffers} buffers")
        for buffer, entry in enumerate(row):
            require(check_entry(entry), f"{label}[{rank}][{buffer}]: invalid entry {entry!r}")


def is_outcome_entry(entry):
    return entry == "*" or is_number(entry) or (
        isinstance(entry, list) and bool(entry) and all(is_number(value) for value in entry)
    )


def validate_elementwise_results(config):
    """Validate an optional exact final state alongside its value-set contract."""
    exact = config.get("elementwise_results")
    if exact is None:
        return
    ranks, buffers, capacity = (config[key] for key in
                                ("required_gpus", "buffers_per_gpu", "buffer_size"))
    check_matrix(exact, ranks, buffers, "elementwise_results",
                 lambda values: (is_number(values) and abs(values) <= 3.4028234e38)
                 or (isinstance(values, list) and len(values) == capacity
                     and all(is_number(value) and abs(value) <= 3.4028234e38 for value in values)))
    exact = [[values if isinstance(values, list) else [values] * capacity for values in row]
             for row in exact]
    expected_ub = config.get("expected_undefined_behaviour",
                             config.get("expected_undefined_behavior", False))
    require(not (config.get("observation_only") or config.get("expected_deadlock") or expected_ub),
            "elementwise_results: requires an ordinary completion/value test")
    outcomes = config.get("acceptable_results")
    require(isinstance(outcomes, list) and len(outcomes) == 1,
            "elementwise_results: requires one matching acceptable_results state")
    check_matrix(outcomes[0], ranks, buffers, "acceptable_results[0]", is_outcome_entry)
    for rank, row in enumerate(exact):
        for buffer, values in enumerate(row):
            allowed = outcomes[0][rank][buffer]
            require(allowed != "*", "elementwise_results: wildcard outcome is not an exact oracle")
            allowed = allowed if isinstance(allowed, list) else [allowed]
            require(set(allowed) == set(values),
                    "elementwise_results: values disagree with acceptable_results")
    return exact


def validate_config(config):
    """Validate allocations and references, deliberately permitting mismatches.

    Missing participants, endpoint mismatches and aliases can be litmus subjects,
    so this checks resource safety without enforcing operation matching. Buffer
    counts use NCCL per-participant capacities, not mathematical helper domains.
    """
    require(isinstance(config, dict), "configuration must be a mapping")
    for key in ("required_gpus", "buffers_per_gpu", "streams_per_gpu", "buffer_size"):
        value = config.get(key)
        require(is_integer(value) and value > 0, f"{key}: expected a positive integer")
    ranks = config["required_gpus"]
    buffers = config["buffers_per_gpu"]
    streams = config["streams_per_gpu"]
    capacity = config["buffer_size"]
    require("events_per_gpu" not in config, "events_per_gpu is obsolete; global event IDs are allocated from the program")
    communicators = config.get("communicators")
    require(isinstance(communicators, list) and bool(communicators), "communicators: expected a nonempty list")
    for index, clique in enumerate(communicators):
        require(isinstance(clique, list) and bool(clique), f"communicators[{index}]: empty/non-list clique")
        for rank in clique:
            check_index(rank, ranks, f"communicators[{index}]")
        require(len(set(clique)) == len(clique), f"communicators[{index}]: duplicate rank")
    check_matrix(config.get("initial_values"), ranks, buffers, "initial_values", is_number)

    if "iterations" in config:
        require(is_integer(config["iterations"]) and config["iterations"] > 0,
                "iterations: expected a positive integer")
    for flag in ("expected_deadlock", "expected_undefined_behaviour",
                 "expected_undefined_behavior", "observation_only"):
        require(flag not in config or isinstance(config[flag], bool), f"{flag}: expected a boolean")
    expected_ub = config.get("expected_undefined_behaviour",
                             config.get("expected_undefined_behavior", False))
    exceptional = config.get("expected_deadlock", False) or expected_ub
    outcomes = config.get("acceptable_results")
    if outcomes is not None:
        require(isinstance(outcomes, list), "acceptable_results: expected a list")
        for index, outcome in enumerate(outcomes):
            check_matrix(outcome, ranks, buffers, f"acceptable_results[{index}]",
                         is_outcome_entry, allow_rank_wildcard=True)
    observation_only = config.get("observation_only", False)
    require(not observation_only or not (outcomes or config.get("expected_deadlock", False)
                                        or config.get("expected_undefined_behaviour", False)
                                        or config.get("expected_undefined_behavior", False)),
            "observation_only: cannot also specify acceptable results, deadlock, or undefined behaviour")
    require(observation_only or exceptional or bool(outcomes),
            "test requires acceptable_results, expected deadlock/undefined behaviour, or observation_only")
    validate_elementwise_results(config)

    source = config.get("program")
    require(isinstance(source, str) and bool(source.strip()), "program: expected DSL source")
    ast = program.parse(source)
    commands = []
    for thread in ast.threads:
        for node in thread.commands:
            commands.extend(node.commands if isinstance(node, Group) else [node])
    for index, command in enumerate(commands):
        if isinstance(command, (Barrier, Skip)):
            continue  # Participant validation is performed by the program parser.
        label = f"command {index + 1}"
        require(isinstance(command, (PointToPoint, Collective, Sync, Copy, Fill, Event, Wait, StreamWait)),
                f"{label}: unsupported command")
        check_index(command.rank, ranks, f"{label} rank")
        if isinstance(command, (Sync, Event, Wait, StreamWait)):
            check_index(command.stream, streams, f"{label} stream")
            if isinstance(command, (Event, Wait, StreamWait)):
                require(is_integer(command.event) and command.event >= 0, f"{label}: invalid event ID")
            continue
        if isinstance(command, (Copy, Fill)):
            check_index(command.stream, streams, f"{label} stream")
            require(command.meta.kind == BufferType.Float, f"{label}: CUDA buffer commands require float")
            count = capacity if command.meta.size == BufferSize.BufferSize else command.meta.size
            require(is_integer(count) and 0 < count <= capacity, f"{label}: invalid CUDA element count")
            if isinstance(command, Fill):
                check_index(command.buffer, buffers, f"{label} buffer")
                require(is_number(command.value) and abs(command.value) <= 3.4028234e38,
                        f"{label}: fill value must be finite and representable as float")
                require(is_integer(command.delay_cycles) and 0 <= command.delay_cycles <= 100000000,
                        f"{label}: fill delay must be in [0,100000000] clock cycles")
            else:
                check_index(command.source_buffer, buffers, f"{label} source buffer")
                check_index(command.destination_buffer, buffers, f"{label} destination buffer")
                require(command.source_buffer != command.destination_buffer,
                        f"{label}: copy requires distinct buffers")
            continue
        check_index(command.channel.stream, streams, f"{label} stream")
        check_index(command.channel.communicator, len(communicators), f"{label} communicator")
        clique = communicators[command.channel.communicator]
        require(command.rank in clique, f"{label}: issuing rank absent from communicator")
        require(isinstance(command.meta.kind, BufferType), f"{label}: unsupported datatype")
        count = capacity if command.meta.size == BufferSize.BufferSize else command.meta.size
        require(is_integer(count) and count > 0, f"{label}: count must be a positive integer")
        if isinstance(command, PointToPoint):
            check_index(command.peer, ranks, f"{label} peer")
            require(command.peer in clique, f"{label}: peer absent from communicator")
            check_index(command.buffer, buffers, f"{label} buffer")
            require(count <= capacity, f"{label}: count exceeds buffer capacity")
        else:
            if isinstance(command, RootRole):
                require(command.rank == command.root, f"{label}: root endpoint must issue on the root")
            if isinstance(command, PeerRole):
                require(command.rank != command.root, f"{label}: peer must differ from root")
            for field in ("source_buffer", "destination_buffer"):
                inactive = isinstance(command, PeerRole) and (
                    (field == "source_buffer" and isinstance(command, (Broadcast, Scatter))) or
                    (field == "destination_buffer" and isinstance(command, (Gather, Reduce))))
                if inactive:
                    require(getattr(command, field) is None, f"{label}: peer has an inactive buffer")
                else:
                    check_index(getattr(command, field), buffers, f"{label} {field}")
            if isinstance(command, Rooted):
                check_index(command.root, ranks, f"{label} root")
                require(command.root in clique, f"{label}: root absent from communicator")
            source_count = count
            destination_count = count
            if isinstance(command, (ReduceScatter, AllToAll)) or (
                isinstance(command, Scatter) and command.rank == command.root
            ):
                source_count *= len(clique)
            if isinstance(command, (AllGather, AllToAll)) or (
                isinstance(command, Gather) and command.rank == command.root
            ):
                destination_count *= len(clique)
            require(source_count <= capacity, f"{label}: source count exceeds buffer capacity")
            require(destination_count <= capacity, f"{label}: destination count exceeds buffer capacity")
    return len(commands)


def validate_file(path):
    path = Path(path)
    config = yaml.safe_load(path.read_text())
    commands = validate_config(config)
    experiment = Experiment(str(path), iters=1)
    require(bool(experiment.source_code.strip()), "rendered CUDA source is empty")
    return commands


def selected_paths(paths):
    """Expand directory selections without treating batch files as litmus tests."""
    files = set()
    for path in paths:
        path = Path(path)
        if path.is_dir():
            files.update(candidate for candidate in path.rglob("*.yaml")
                         if "batches" not in candidate.relative_to(path).parts)
            files.update(candidate for candidate in path.rglob("*.yml")
                         if "batches" not in candidate.relative_to(path).parts)
        else:
            files.add(path)
    return sorted(files)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path,
                        help="YAML files or suite directories (default: experiments/)")
    args = parser.parse_args(argv)
    paths = selected_paths(args.paths or [Path("experiments")])
    if not paths:
        print("No litmus YAML files selected.", file=sys.stderr)
        return 1
    failures = 0
    for path in paths:
        try:
            count = validate_file(path)
            print(f"PASS {path}: {count} commands; resources checked and source rendered")
        except Exception as error:
            failures += 1
            print(f"FAIL {path}: {error}", file=sys.stderr)
    print(f"CPU checks: {len(paths) - failures}/{len(paths)} files passed; GPU behaviour remains unvalidated.")
    return int(bool(failures))


if __name__ == "__main__":
    sys.exit(main())

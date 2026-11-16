"""Shared result classification for research and standalone artefact analysis."""
import ast
import hashlib
import re

from framework.run_experiment import Experiment


CATEGORIES = (
    'All runs complete with expected outcomes (1M iterations)',
    'All runs deadlocked (expected)',
    'Expected deadlock, got results',
    'Expected UB, observed deadlock',
    'Expected UB, completed with consistent results (1M iterations)',
    'Unexpected deadlock',
    'Unexpected completed outcomes',
    'Mixed completion and deadlock',
    'Incomplete/not run yet',
)
INPUT_KEYS = (
    'required_gpus', 'buffers_per_gpu', 'buffer_size', 'streams_per_gpu',
    'communicators', 'initial_values', 'elementwise_results',
)

PAPER_CATEGORIES = CATEGORIES[:5] + (
    'Expected UB, completed with varied outcomes (1M iterations)',
    'Expected UB, resource termination without observations',
    'Expected UB, observed deadlock and resource termination',
) + CATEGORIES[5:8] + ('Observation-only', 'Pending')


def current_records(records, iteration_target=1_000_000, wall_time_target=3600):
    """Select the shortest newest whole-run suffix reaching either campaign limit.

    A recorded wall-budget exhaustion ends selection even without observations.
    Whole records cannot be split into chronological observations from histograms.
    """
    selected = []
    for record in sorted(records, key=lambda r: r['run'], reverse=True):
        selected.append(record)
        if (sum(r['iterations'] for r in selected) >= iteration_target
                or (completion_status([record], iteration_target) == 'wall_time_exhausted'
                    and record['max_wall_time'] >= wall_time_target)):
            break
    return list(reversed(selected))


def completion_status(records, iteration_target=1_000_000):
    """Report the evidenced stopping condition independently of conformance."""
    if sum(r['iterations'] for r in records) >= iteration_target:
        return 'iteration_target_reached'
    if not records:
        return 'incomplete'
    latest = max(records, key=lambda r: r['run'])
    if (latest.get('execution_status') == 'wall_time_exhausted'
            and isinstance(latest.get('max_wall_time'), (int, float))
            and latest['max_wall_time'] > 0
            and isinstance(latest.get('elapsed_seconds'), (int, float))
            and latest['elapsed_seconds'] >= latest['max_wall_time']
            and any(a.get('outcome') == 'WallTimeLimit' for a in latest['attempts'])):
        return 'wall_time_exhausted'
    if (latest.get('execution_status') == 'retry_exhausted'
            and isinstance(latest.get('max_retry'), int)
            and latest['max_retry'] > 0
            and len(latest['attempts']) >= latest['max_retry']):
        return 'retry_exhausted'
    return 'incomplete'


def classify_paper(config, records, iteration_target=1_000_000):
    if config.get('observation_only'):
        return 'Observation-only'
    ub = config.get('expected_undefined_behaviour',
                    config.get('expected_undefined_behavior', False))
    if ub and not config.get('expected_deadlock'):
        if any(r['deadlocks'] for r in records) and any(
                r.get('resource_termination') for r in records):
            return PAPER_CATEGORIES[7]
        if not sum(r['iterations'] for r in records) and any(
                r.get('resource_termination') for r in records):
            return PAPER_CATEGORIES[6]
        if (sum(r['iterations'] for r in records) >= iteration_target
                and not any(r['deadlocks'] or r['runtime_errors'] for r in records)
                and len({s for r in records for s in r['states']}) > 1):
            return PAPER_CATEGORIES[5]
    category = classify(config, records, iteration_target)
    return 'Pending' if category == CATEGORIES[8] else category


def signature(config):
    # Ignore annotations and outcome contracts, but preserve the executable inputs
    # and any elementwise instrumentation.
    program = re.sub(r'/\*.*?\*/|//[^\n]*', '', config['program'], flags=re.S)
    return [config.get(key) for key in INPUT_KEYS] + [''.join(program.split())]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def value_conforms(state, expected):
    ranks = [ast.literal_eval(line.split(':', 1)[1].strip())
             for line in state.splitlines() if line.strip().startswith('Rank ')]
    state_tuple = tuple(tuple(tuple((float(v), 1) for v in buf) for buf in rank)
                        for rank in ranks)
    # Use the harness's interpretation of scalar/list/wildcard outcomes.
    checker = Experiment.__new__(Experiment)
    result = checker._check_conformance(state_tuple, expected)
    return result in (checker._Conformance.ACCEPTED, checker._Conformance.WILDCARD)


def classify(config, records, iteration_target=1_000_000):
    deadlock = bool(config.get('expected_deadlock'))
    ub = bool(config.get('expected_undefined_behaviour',
                         config.get('expected_undefined_behavior', False))) and not deadlock
    expected = config.get('acceptable_results')
    if config.get('observation_only') or not (deadlock or ub or expected):
        return CATEGORIES[8]
    iterations = sum(r['iterations'] for r in records)
    stalls = sum(r['deadlocks'] for r in records)
    if stalls and iterations:
        return CATEGORIES[7]
    if stalls:
        return CATEGORIES[1 if deadlock else 3 if ub else 5]
    if iterations and deadlock:
        return CATEGORIES[2]
    if any(r['unexpected_values'] for r in records):
        return CATEGORIES[6]
    if iterations < iteration_target or any(r['runtime_errors'] for r in records):
        return CATEGORIES[8]
    states = {state for r in records for state in r['states']}
    if ub:
        return CATEGORIES[4] if len(states) == 1 else CATEGORIES[8]
    return CATEGORIES[0]


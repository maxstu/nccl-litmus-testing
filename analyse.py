"""Regenerate suite and outcome tables from anonymous saved evidence, without GPUs."""

import argparse
import ast
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

import yaml

from framework.result_analysis import current_records, classify_paper, completion_status, signature, value_conforms, PAPER_CATEGORIES

ROOT = Path(__file__).resolve().parent
QUESTIONS = {
    'Command matching': 'Which calls can form one operation?',
    'Grouping and host ownership': 'When can one thread’s calls progress together?',
    'P2P ordering': 'Which send matches which receive?',
    'Dependencies and synchronisation': 'When can a later operation safely consume a result?',
    'Communicator interaction': 'How do operations interact between communicators?',
    'Buffer aliasing': 'When do accesses conflict?',
    'Collective values': 'How are contributions combined and distributed?',
}


SUMMARY_CATEGORIES = (
    'Completed with results consistent with expectations (non-UB)',
    'Expected UB, observed consistent results',
    'Expected UB, observed corruption of output buffers',
    'Expected UB/deadlock, terminated abnormally',
)


def summary_category(config, records, iteration_target=1_000_000):
    """Classify completed observations, retaining attempt failures in the audit.

    Do not force incomplete or contradictory evidence into the four paper rows.
    Corruption requires nonuniform buffers in this uniform-buffer cohort, or
    values outside the initial-value domain in a send/receive-only program.
    Mere variation between whole-buffer outcomes is insufficient.
    """
    ub = config.get('expected_undefined_behaviour',
                    config.get('expected_undefined_behavior', False))
    deadlock = config.get('expected_deadlock', False)
    if config.get('observation_only'):
        return 'Unclassified: observation-only'
    iterations = sum(r['iterations'] for r in records)
    if not iterations:
        if (ub or deadlock) and completion_status(records, iteration_target) == 'wall_time_exhausted' and any(
                r['deadlocks'] or r.get('resource_termination') or
                any(a['phase'] == 'execution' for a in r['attempts']) for r in records):
            return SUMMARY_CATEGORIES[3]
        return 'Unclassified: incomplete or unexpected termination'
    if deadlock or any(r['unexpected_values'] for r in records):
        return 'Unclassified: unexpected completed outcomes'
    if iterations < iteration_target or any(r['runtime_errors'] for r in records):
        return 'Unclassified: incomplete observations or runtime errors'
    if not ub:
        return SUMMARY_CATEGORIES[0] if config.get('acceptable_results') else 'Unclassified: missing oracle'
    states = {s for r in records for s in r['states']}
    if len(states) == 1:
        return SUMMARY_CATEGORIES[1]
    # Evidence states list distinct values per buffer unless elementwise instrumentation is enabled.
    if not config.get('elementwise_results'):
        buffers = [ast.literal_eval(line.split(':', 1)[1].strip())
                   for state in states for line in state.splitlines()
                   if line.strip().startswith('Rank ')]
        initial = config['initial_values']
        uniform_inputs = all(isinstance(v, (int, float)) for rank in initial for v in rank)
        if uniform_inputs and any(len(set(buf)) > 1 for rank in buffers for buf in rank):
            return SUMMARY_CATEGORIES[2]
        commands = re.findall(r'([A-Za-z_]+)\s*<', config['program'])
        if uniform_inputs and commands and set(commands) <= {'send', 'recv'}:
            domain = {v for rank in initial for v in rank}
            if any(v not in domain for rank in buffers for buf in rank for v in buf):
                return SUMMARY_CATEGORIES[2]
    return 'Unclassified: varied UB outcomes without established corruption'


def local_results(root, runs):
    """Read launcher outputs in caller-supplied chronological order."""
    snapshot = json.loads((root / 'snapshot.json').read_text())
    tests = {path: {'path': path, 'records': []} for path in snapshot['tests']}
    if len({run.resolve() for run in runs}) != len(runs):
        raise ValueError('Duplicate local run directory')
    limits = set()
    for number, run in enumerate(runs, 1):
        if not (run / 'batch.yaml').is_file():
            raise ValueError(f'Not a launcher output directory: {run}')
        batch = yaml.safe_load((run / 'batch.yaml').read_text())
        limits.add((batch['iters'], batch['max_wall_time']))
        if len(limits) != 1:
            raise ValueError('Local runs must have the same iteration and wall-time targets')
        for path, test in tests.items():
            result_path = run / Path(path).stem / 'final_results.yaml'
            if not result_path.exists():
                continue
            original = yaml.safe_load((run / 'inputs' / path).read_text())
            result = yaml.safe_load(result_path.read_text())
            attempts = result.get('attempts', [])
            test['records'].append({
                'run': f'run-{number:06d}', 'input_signature': signature(original),
                'iterations': result['total_iterations'],
                'execution_status': result.get('execution_status'),
                'max_wall_time': result.get('max_wall_time', batch.get('max_wall_time')),
                'elapsed_seconds': result.get('elapsed_seconds'),
                'max_retry': batch.get('max_retry', 3),
                'observations': [{'state': o['state'], 'count': o['count']}
                                 for o in result.get('observations', [])],
                'runtime_errors': len(result.get('errors', [])),
                'attempts': [{'phase': a.get('phase'), 'outcome': a.get('outcome')} for a in attempts],
                'resource_termination': any(a.get('phase') == 'execution' for a in attempts) and any(
                    re.search(r'oom_kill|Out Of Memory', log.read_text(errors='replace'))
                    for log in result_path.parent.glob('attempt_*.log')),
            })
    if not any(test['records'] for test in tests.values()):
        raise ValueError('No final results found in supplied launcher outputs')
    iterations, wall_time = limits.pop()
    return {'schema_version': 1, 'iteration_target': iterations,
            'wall_time_target': wall_time, 'tests': list(tests.values())}


def analyse(root, evidence=None, payload=None):
    snapshot = json.loads((root / 'snapshot.json').read_text())
    for path, expected in snapshot['files'].items():
        if hashlib.sha256((root / path).read_bytes()).hexdigest() != expected:
            raise ValueError(f'File checksum mismatch: {path}')
    supplied = evidence is not None or payload is not None
    if payload is None:
        payload = json.loads(evidence.read_text()) if evidence else {'schema_version': 1, 'tests': []}
    if payload['schema_version'] != 1:
        raise ValueError('Unsupported evidence schema')
    iteration_target = payload.get('iteration_target', 1_000_000)
    wall_time_target = payload.get('wall_time_target', 3600)
    if any(type(v) is not int or v <= 0 for v in (iteration_target, wall_time_target)):
        raise ValueError('Analysis targets must be positive integers')
    def category_label(label):
        return label.replace('(1M iterations)', f'({iteration_target:,} iterations)') if iteration_target != 1_000_000 else label
    inputs = {}
    for test in payload['tests']:
        if test['path'] in inputs or test['path'] not in snapshot['tests']:
            raise ValueError('Duplicate or unknown evidence test')
        inputs[test['path']] = test
    tests = []
    for path in snapshot['tests']:
        config = yaml.safe_load((root / path).read_text())
        source = inputs.get(path, {'records': []})
        records, seen = [], set()
        for record in source['records']:
            if record['run'] in seen:
                raise ValueError(f'Duplicate run/test evidence: {path}')
            seen.add(record['run'])
            if record['input_signature'] != signature(config):
                raise ValueError(f'Evidence inputs or instrumentation differ from test: {path}')
            observations = record['observations']
            counts = [o['count'] for o in observations] + [record['runtime_errors'], record['iterations']]
            if any(type(count) is not int or count < 0 for count in counts):
                raise ValueError('Invalid observation counts')
            if sum(o['count'] for o in observations) + record['runtime_errors'] != record['iterations']:
                raise ValueError('Observations/errors do not account for received iterations')
            expected = config.get('acceptable_results')
            check_values = expected and not config.get('expected_deadlock') and not config.get(
                'expected_undefined_behaviour', config.get('expected_undefined_behavior', False))
            records.append({**record,
                'deadlocks': sum(a['phase'] == 'execution' and a['outcome'] == 'ExitTimeout'
                                 for a in record['attempts']),
                'states': sorted({o['state'] for o in observations if o['count']}),
                'unexpected_values': sum(o['count'] for o in observations
                                         if check_values and not value_conforms(o['state'], expected)),
            })
        selected = current_records(records, iteration_target, wall_time_target)
        tests.append({'path': path, 'category': category_label(classify_paper(config, selected, iteration_target)),
                      'summary_category': summary_category(config, selected, iteration_target),
                      'completion_status': completion_status(selected, iteration_target),
                      'evidence': selected})
    counts = Counter(t['category'] for t in tests)
    summary = Counter(t['summary_category'] for t in tests)
    return {'iteration_target': iteration_target, 'wall_time_target': wall_time_target,
            'total': len(tests), 'evidence_supplied': supplied,
            'evidence_sha256': hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest() if supplied else None,
            'families': dict(Counter(snapshot['families'].values())),
            'completion_counts': dict(Counter(t['completion_status'] for t in tests)),
            'summary_counts': {c: summary[c] for c in SUMMARY_CATEGORIES},
            'unclassified_counts': {c: n for c, n in summary.items() if c not in SUMMARY_CATEGORIES},
            'counts': {category_label(category): counts[category_label(category)] for category in PAPER_CATEGORIES}, 'tests': tests}


def table(output, name, headers, rows):
    with (output / f'{name}.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)
    markdown = ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join('---' for _ in headers) + ' |']
    markdown.extend('| ' + ' | '.join(map(str, row)) + ' |' for row in rows)
    (output / f'{name}.md').write_text('\n'.join(markdown) + '\n')
    # These table values contain no LaTeX-special characters except the count header.
    def escape(value):
        return str(value).replace('&', r'\&').replace('#', r'\#').replace('_', r'\_')
    columns = (r'@{}p{0.35\linewidth}p{\dimexpr0.55\linewidth-4\tabcolsep\relax}p{0.10\linewidth}@{}'
               if len(headers) == 3 else
               r'@{}p{0.82\linewidth}p{\dimexpr0.18\linewidth-2\tabcolsep\relax}@{}')
    lines = [r'\begin{tabular}{' + columns + '}', r'\toprule',
             ' & '.join(map(escape, headers)) + r' \\', r'\midrule']
    lines.extend(' & '.join(map(escape, row)) + r' \\' for row in rows)
    lines.extend([r'\bottomrule', r'\end{tabular}'])
    (output / f'{name}.tex').write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'evidence/results.json')
    parser.add_argument('--inventory-only', action='store_true', help='No results required; do not emit an outcome table')
    parser.add_argument('--local-runs', type=Path, nargs='+',
                        help='Analyse fresh run.py outputs instead of imported evidence, oldest first')
    parser.add_argument('--output', type=Path, default=ROOT / 'output/analysis')
    args = parser.parse_args()
    if args.inventory_only and args.local_runs:
        parser.error('--inventory-only and --local-runs are mutually exclusive')
    if not args.inventory_only and not args.local_runs and not args.evidence.is_file():
        parser.error('No saved evidence supplied; import results first, or use --inventory-only')
    payload = local_results(ROOT, args.local_runs) if args.local_runs else None
    report = analyse(ROOT, None if args.inventory_only or payload else args.evidence, payload)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2) + '\n')
    table(args.output, 'litmus-suite', ['Concept tested', 'Questions of the form...', '# Tests'],
          [(family, question, report['families'][family]) for family, question in QUESTIONS.items()]
          + [('Total', '', report['total'])])
    if not args.inventory_only:
        table(args.output, 'campaign-completion', ['Stopping condition', '# Tests'],
              list(report['completion_counts'].items()) + [('Total', report['total'])])
        table(args.output, 'campaign-outcomes', ['Recorded outcome', '# Tests'],
              list(report['summary_counts'].items()) + list(report['unclassified_counts'].items())
              + [('Total', report['total'])])
        table(args.output, 'campaign-outcomes-detailed', ['Recorded outcome', '# Tests'],
              list(report['counts'].items()) + [('Total', report['total'])])
    print(f'Wrote tables and per-test audit to {args.output}')


if __name__ == '__main__':
    main()

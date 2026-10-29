"""Run paired initial states across all packaged policies and floor coefficients."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import datetime
import json
import os
from pathlib import Path
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from rollout import NativeTrial, PACKAGES, ROOT, sha


def make_plan():
    tests = [{'suite': 'nominal', 'mu': 1., 'seed': 0, 'jitter': False}]
    tests += [{'suite': 'clean', 'mu': 1., 'seed': 6000 + i, 'jitter': True} for i in range(64)]
    for mu in [.5, .6, .7, .8, .9, 1.1, 1.2, 1.3, 1.5]:
        tests += [{'suite': f'mu_{mu:.1f}', 'mu': mu, 'seed': 6000 + i, 'jitter': True} for i in range(16)]
    rng = np.random.default_rng(20261029)
    tests += [{'suite': 'friction_only_dr', 'mu': float(rng.uniform(.5, 1.5)),
               'seed': 7000 + i, 'jitter': True} for i in range(32)]
    assert len(tests) == 241
    return tests


def run_candidate(directory):
    trial = NativeTrial(directory)
    for index, test in enumerate(make_plan()):
        folder = ROOT / 'results' / directory / test['suite'] / str(test['seed'])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / 'evaluation.json'
        if path.exists():
            existing = json.loads(path.read_text())
            assert existing['evaluator_sha256'] == sha(ROOT / 'rollout.py')
            assert existing['policy_sha256'] == trial.contract['policy_sha256']
            assert existing['floor_friction'] == test['mu']
            continue
        result, poses = trial.run(mu=test['mu'], seed=test['seed'], jitter=test['jitter'],
                                  capture=test['suite'] == 'nominal')
        result.update(suite=test['suite'], evaluator_sha256=sha(ROOT / 'rollout.py'))
        if poses is not None:
            np.savez_compressed(folder / 'rollout.npz', **poses)
            result['rollout_sha256'] = sha(folder / 'rollout.npz')
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(result, indent=2) + '\n')
        temp.replace(path)
        if index % 16 == 0:
            print('PROGRESS', directory, index + 1, '/', 241,
                  'clip', result['clip_success'], 'hold', result['clip_and_hold_success'], flush=True)
    return directory


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    entries = json.loads((PACKAGES / 'selection.json').read_text())['entries']
    directories = [e['directory'] for e in entries]
    priorities = ['fixed/mu_0.7', 'fixed/mu_0.8', 'fixed/mu_1.3', 'dr_shared/rank1_s45_i20000']
    directories = priorities + [d for d in directories if d not in priorities]
    plan = {'created_at': datetime.datetime.now().astimezone().isoformat(), 'directories': directories,
            'trials_per_candidate': 241, 'total_trials': 241 * len(directories), 'tests': make_plan(),
            'runtime_dr': False, 'initial_jitter': 'Same amplitudes as training, independent NumPy seed stream',
            'friction_dr_scope': 'Floor friction only, U(0.5,1.5), constant during each trial; no mass/COM/push/observation DR',
            'physics': 'Standalone MuJoCo mj_step', 'clip_s': 5.92, 'post_clip_hold_s': 2.,
            'fall': {'height_m': .35, 'tilt_deg': 75., 'duration_s': .2},
            'evaluator_sha256': sha(ROOT / 'rollout.py'), 'workers': args.workers}
    (ROOT / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    start = time.monotonic()
    done = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_candidate, d): d for d in directories}
        for future in as_completed(futures):
            done.append(future.result())
            (ROOT / 'status.json').write_text(json.dumps({'completed_candidates': done,
                  'wall_seconds': time.monotonic() - start, 'total_candidates': len(directories),
                  'updated_at': datetime.datetime.now().astimezone().isoformat()}, indent=2) + '\n')
            print('CANDIDATE_COMPLETE', done[-1], len(done), '/', len(directories), flush=True)
    print('ALL_COMPLETE', round(time.monotonic() - start, 2), flush=True)


if __name__ == '__main__':
    main()

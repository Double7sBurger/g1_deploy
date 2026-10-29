"""Collect selected CR7 checkpoints and their unchanged training evidence."""

import hashlib
import json
from pathlib import Path
import shutil
import statistics
import tarfile

ROOT = Path(__file__).resolve().parent.parent
DEST = Path('/home/henry/workspace/g1_deploy/policies/cr7_friction_20261029')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rank(candidate):
    suites = candidate['suites'].values()
    rates = [s['success_rate'] for s in suites]
    return min(rates), statistics.mean(rates), -statistics.mean(s['body_error_m'] for s in suites)


def main():
    summary = json.loads((ROOT / 'summary.json').read_text())
    choices = []
    for group in summary['best_by_group']:
        if group['mode'] == 'old_implicit200':
            c = group['best']
            mu = int(c['arm'].removeprefix('cr7-mu')) / 10
            choices.append((f'fixed/mu_{mu:.1f}', c))
    for arm in ('shared', 'ground'):
        candidates = sorted((c for c in summary['candidates'] if c['mode'] == 'old_explicit400'
                             and c['arm'] == f'cr7-{arm}' and c['status'] == 'COMPLETE'),
                            key=rank, reverse=True)
        for i, c in enumerate(candidates[:3], 1):
            choices.append((f'dr_{arm}/rank{i}_s{c["training_seed"]}_i{c["iteration"]}', c))
    assert len(choices) == 15
    DEST.mkdir(parents=True, exist_ok=True)
    (DEST / '.gitignore').write_text('!**/motion.npz\n')
    entries = []
    for relative, c in choices:
        assert c['status'] == 'COMPLETE'
        target = DEST / relative
        source, checkpoint = Path(c['source_run']), Path(c['checkpoint'])
        files = {checkpoint.name: checkpoint, 'motion.npz': source / 'reference.npz',
                 'motion.json': source / 'reference.json'}
        for name in ('sysid_parameters.json', 'experiment.json', 'source_manifest.json', 'training_summary.json'):
            files['training/' + name] = source / name
        for p in (checkpoint.parent / 'params').glob('*.yaml'):
            files['training/params/' + p.name] = p
        for collection in ('results', 'validation'):
            for p in (ROOT / collection / c['id']).glob('*/*/evaluation.json'):
                files['evaluations/' + collection + '/' + str(p.relative_to(ROOT / collection / c['id']))] = p
        holdout = ROOT / 'validation' / c['id'] / 'summary.json'
        if holdout.exists():
            files['evaluations/holdout_summary.json'] = holdout
        for name, p in files.items():
            assert p.is_file(), p
            out = target / name
            out.parent.mkdir(parents=True, exist_ok=True)
            if out.exists():
                assert sha(out) == sha(p), out
            else:
                shutil.copy2(p, out)
            assert sha(out) == sha(p)
        assert sha(target / checkpoint.name) == c['checkpoint_sha256']
        assert sha(target / 'motion.npz') == c['motion_sha256']
        assert sha(target / 'training/sysid_parameters.json') == c['sysid_sha256']
        (target / 'selection.json').write_text(json.dumps(c, indent=2) + '\n')
        entries.append({'directory': relative, 'candidate': c})
    sources = DEST / 'training_sources'
    sources.mkdir(exist_ok=True)
    for mode in ('old_implicit200', 'old_explicit400'):
        experiment = Path(next(e['candidate']['source_experiment'] for e in entries if e['candidate']['mode'] == mode))
        project = experiment / 'snapshot/project'
        paths = list((project / 'source/motion_tracking/motion_tracking').rglob('*.py'))
        paths += list((project / 'scripts').rglob('*.py'))
        with tarfile.open(sources / (mode + '.tar.gz'), 'w:gz') as archive:
            for p in sorted(set(paths)):
                archive.add(p, arcname=str(p.relative_to(project)))
        (sources / (mode + '.json')).write_text(json.dumps({
            'source_experiment': str(experiment),
            'files_sha256': {str(p.relative_to(project)): sha(p) for p in sorted(set(paths))},
            'archive_scope': 'Frozen project Python sources; Isaac Lab, dependencies and USD assets are external.',
        }, indent=2) + '\n')
    (DEST / 'selection.json').write_text(json.dumps({
        'protocol': 'fall-only-v1', 'selection_scope': summary['scope'],
        'ranking': 'min(clean, robust) full survival; mean survival rate; mean body error while alive',
        'initial_corrective_steps_allowed': True, 'hardware_validated': False,
        'entries': entries,
    }, indent=2) + '\n')
    print(json.dumps([{'directory': e['directory'], 'id': e['candidate']['id']} for e in entries], indent=2))


if __name__ == '__main__':
    main()

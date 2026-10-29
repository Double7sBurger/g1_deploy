"""Package the two selected seeds from each completed walk friction experiment."""

import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT.parent / 'newton_sysid_friction8_20261028'
SOURCE = EXPERIMENT / 'partial_results/selected'
DEST = Path('/home/henry/workspace/g1_deploy/policies/walk_sysid_friction_20261029')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        assert sha(source) == sha(target), target
    else:
        shutil.copy2(source, target)
    assert sha(source) == sha(target)


def main():
    entries = []
    for arm in ('shared', 'ground'):
        selected = json.loads((SOURCE / f'walk-{arm}/selection.json').read_text())
        assert selected['all_eight_seeds_complete']
        for rank, choice in enumerate(selected['best_two'], 1):
            source = SOURCE / f'walk-{arm}/best_{rank}'
            relative = f'dr_{arm}/rank{rank}_s{choice["seed"]}_i{choice["iteration"]}'
            folder = DEST / relative
            checkpoint = source / f'model_{choice["iteration"]}.pt'
            experiment = json.loads((source / 'experiment.json').read_text())
            assert experiment['arm'] == choice['motion'] and experiment['seed'] == choice['seed']
            for target, original in [(checkpoint.name, checkpoint), ('motion.npz', source / 'reference.npz'),
                                     ('motion.json', source / 'reference.json')]:
                copy(original, folder / target)
            for name in ('experiment.json', 'source_manifest.json', 'training_summary.json', 'sysid_parameters.json'):
                copy(source / name, folder / 'training' / name)
            for path in (source / 'params').glob('*.yaml'):
                copy(path, folder / 'training/params' / path.name)
            evaluations = list((source / 'evaluations').glob('*/*/evaluation.json'))
            assert len(evaluations) == 4
            for path in evaluations:
                data = json.loads(path.read_text())
                assert data['checkpoint_sha256'] == choice['checkpoint_sha256']
                assert data['motion_sha256'] == choice['motion_sha256']
                assert data['sysid_sha256'] == experiment['sysid_sha256']
                assert data['full_clip_success_rate'] == 1.0
                copy(path, folder / 'evaluations/selection' / path.relative_to(source / 'evaluations'))
            candidate = dict(choice, mode='old_explicit400', arm=choice['motion'],
                             training_seed=choice['seed'], checkpoint=str(checkpoint),
                             sysid_sha256=experiment['sysid_sha256'], source_experiment=str(EXPERIMENT),
                             source_run=str(source))
            for target, field in [(checkpoint.name, 'checkpoint_sha256'), ('motion.npz', 'motion_sha256'),
                                  ('training/sysid_parameters.json', 'sysid_sha256')]:
                assert sha(folder / target) == candidate[field]
            (folder / 'selection.json').write_text(json.dumps(candidate, indent=2) + '\n')
            entries.append(dict(directory=relative, candidate=candidate))
        copy(SOURCE / f'walk-{arm}/selection.json', DEST / 'original_selections' / f'walk-{arm}.json')
    (DEST / '.gitignore').write_text('!**/motion.npz\n!native_mujoco_validation/videos/**/*.mp4\n')
    (DEST / 'selection.json').write_text(json.dumps({
        'protocol': 'shared-dr-v1', 'selection_scope': 'Eight seeds per arm; three late checkpoints per seed; two distinct seeds per arm',
        'hardware_validated': False, 'entries': entries,
    }, indent=2) + '\n')
    old_sources = DEST.parent / 'cr7_friction_20261029/training_sources'
    for suffix in ('.tar.gz', '.json'):
        copy(old_sources / ('old_explicit400' + suffix), DEST / 'training_sources' / ('old_explicit400' + suffix))
    print(json.dumps([e['directory'] for e in entries], indent=2))


if __name__ == '__main__':
    main()

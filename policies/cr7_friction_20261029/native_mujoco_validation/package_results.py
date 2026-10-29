"""Attach audited native MuJoCo evidence to the existing deployment collection."""

from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path(__file__).resolve().parent
PACKAGES = Path('/home/henry/workspace/g1_deploy/policies/cr7_friction_20261029')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


summary = json.loads((ROOT / 'summary.json').read_text())
assert summary['completed_trials'] == 3615 and summary['pending_trials'] == 0
plan = json.loads((ROOT / 'plan.json').read_text())
failures = Counter()
for p in (ROOT / 'results').rglob('evaluation.json'):
    r = json.loads(p.read_text())
    model_dir = ROOT / 'models' / r['mode']
    assert r['model_sha256'] == sha(model_dir / 'model.mjb')
    assert r['evaluator_sha256'] == sha(ROOT / 'rollout.py')
    f = r['numerical_failure']
    if f:
        key = ('after_clip' if f['time_s'] > 5.92 else 'during_clip') + '_' + (
            'after_fall' if r['fall_time_s'] is not None and f['time_s'] > r['fall_time_s'] else 'before_fall')
        failures[key] += 1
fresh_checks = []
for name, original in [('mu13_seed7011', 'fixed/mu_1.3/friction_only_dr/7011'),
                       ('mu12_nominal', 'fixed/mu_1.2/nominal/0')]:
    a = json.loads((ROOT / 'finite_check' / name / 'rollout.json').read_text())
    b = json.loads((ROOT / 'results' / original / 'evaluation.json').read_text())
    assert a['numerical_failure'] == b['numerical_failure']
    assert a['initial_state'] == b['initial_state']
    fresh_checks.append({'directory': a['directory'], 'seed': a['seed'], 'failure': a['numerical_failure'],
                         'original_seed': b['seed'], 'initial_jitter': a['initial_jitter'],
                         'reproduced_in_fresh_process': True})
models = {m: json.loads((ROOT / 'models' / m / 'model.json').read_text()) for m in ('implicit200', 'explicit400')}
videos = json.loads((ROOT / 'videos_verified.json').read_text())
assert len(videos) == 5
for v in videos:
    assert sha(Path(v['video'])) == v['sha256']
    v['encoded_jump_landing_contact_sheet_reviewed'] = True
write(ROOT / 'videos_verified.json', videos)
priority_quality = {}
for directory in ('fixed/mu_0.7', 'fixed/mu_0.8'):
    trials = [json.loads(p.read_text()) for p in (ROOT / 'results' / directory).rglob('evaluation.json')]
    assert len(trials) == 241 and all(r['clip_and_hold_success'] for r in trials)
    priority_quality[directory] = {'successful_trials': 241, 'total_trials': 241,
        'max_actual_joint_limit_excess_deg': max(r['max_actual_joint_limit_excess_deg'] for r in trials)}
audit = {'completed_at': datetime.datetime.now().astimezone().isoformat(), 'candidates': 15,
         'trials': 3615, 'pending': 0, 'native_mujoco_version': '3.11.0',
         'frozen_model_sha256': {k: v['model_sha256'] for k, v in models.items()},
         'frozen_model_fk_error_max_m': max(v['fk_max_body_error_m'] for v in models.values()),
         'model_gravity_and_control_timing_verified': True,
         'floor_contact_coefficients_verified_each_trial': True,
         'numerical_failure_breakdown': dict(failures), 'fresh_process_checks': fresh_checks,
         'priority_candidates': priority_quality, 'encoded_videos_verified': 5,
         'batch_wall_seconds': json.loads((ROOT / 'completed_original_status.json').read_text())['wall_seconds'],
         'evaluator_sha256': sha(ROOT / 'rollout.py'),
         'diagnostic_versions_excluded': ['diagnostic_v1', 'diagnostic_v2'], 'hardware_validated': False}
write(ROOT / 'AUDIT.json', audit)
venv = '/home/henry/workspace/isaac-spinning-up/logs/osmo/newton_sysid_20261028/validation/venv/bin/python'
repro = f'''原生 MuJoCo 闭环复现（本机已有依赖）：
{venv} {ROOT}/rollout.py --directory fixed/mu_0.7 --mu 1.0 --output {ROOT}/reproduce_mu07
{venv} {ROOT}/rollout.py --directory fixed/mu_0.8 --mu 0.5 --seed 6000 --jitter --output {ROOT}/reproduce_mu08_low_friction

完整3615条矩阵（已存在且hash一致的记录会跳过）：
{venv} {ROOT}/batch.py --workers 4
python {ROOT}/summarize.py

可视化：
{venv} {ROOT}/render.py fixed/mu_0.7

MuJoCo 3.11.0；模型位于models/implicit200与explicit400/model.mjb。
模型源为官方Isaac6.2 g1_29dof.usd，无Dex3；只在一次性模型导出时使用Newton的USD转换器。
独立rollout进程仅调用mujoco.mj_step，断言没有导入Newton/Isaac Lab。
运行脚本当前记录本机路径；迁移时需调整DEPLOY/PACKAGES/ROOT及可视化ASSET路径。
policy/motion/contract仍使用所在部署包。不要将本仿真中的bias/延时实现再叠加到真实硬件控制。
原始结果归档trial_results.tar.gz内含results/下3615份evaluation.json及15条nominal轨迹，运行时不依赖诊断版本。
所有试验5.92秒CR7+2秒末帧hold；此处friction_only_dr仅随机地面，不包括质量/COM/编码器/推扰/噪声DR。
'''
(ROOT / 'REPRODUCE.txt').write_text(repro)
dest = PACKAGES / 'native_mujoco_validation'
dest.mkdir(exist_ok=True)
for name in ('summary.json', 'results.csv', 'report.html', 'plan.json', 'jump_quality.json', 'AUDIT.json',
             'REPRODUCE.txt', 'videos_verified.json', 'rollout.py', 'batch.py', 'summarize.py',
             'export_model.py', 'render.py', 'verify_videos.py', 'jump_quality.py', 'package_results.py'):
    shutil.copy2(ROOT / name, dest / name)
shutil.copytree(ROOT / 'models', dest / 'models', dirs_exist_ok=True)
shutil.copytree(ROOT / 'videos', dest / 'videos', dirs_exist_ok=True)
with tarfile.open(dest / 'trial_results.tar.gz', 'w:gz') as archive:
    for p in sorted((ROOT / 'results').rglob('*')):
        if p.is_file():
            archive.add(p, arcname=str(p.relative_to(ROOT)))
for row in summary['candidates']:
    folder = PACKAGES / row['directory']
    out = folder / 'evaluations/native_mujoco'
    out.mkdir(exist_ok=True)
    write(out / 'summary.json', row)
    manifest = json.loads((folder / 'manifest.json').read_text())
    manifest['files_sha256']['evaluations/native_mujoco/summary.json'] = sha(out / 'summary.json')
    write(folder / 'manifest.json', manifest)
ignore = PACKAGES / '.gitignore'
rule = '!native_mujoco_validation/videos/**/*.mp4\n'
if rule not in ignore.read_text():
    ignore.write_text(ignore.read_text() + rule)
readme = PACKAGES / 'README.md'
text = readme.read_text()
text = text.replace('本次是离线导出/接口验证，不是新的真机或 MuJoCo 物理验证。部分评估仍记录实际关节超限，',
                    '导出阶段完成离线接口验证；后续原生 MuJoCo 验证见下节，尚无真机验证。部分评估仍记录实际关节超限，')
marker = '\n## 后续原生 MuJoCo 验证（2026-10-29）\n'
if marker in text:
    text = text.split(marker)[0]
text += marker + '''
已完成独立 `mujoco.mj_step` 闭环测试：15个候选 × 241条 = 3615条，均覆盖完整5.92秒动作及2秒收尾。
模型为同一官方无手G1 USD转换的物理模型，保留旧SysID；使用当前部署runner，没有用旧Newton记录替代物理运行。

μ=0.7 与 μ=0.8 均 241/241 通过，包括μ=1的64次初态扰动、九个其他固定μ各16次，以及地面U(0.5,1.5)的32次。
这里随机DR仅改变地面摩擦，未加入之前完整DR评估里的质量/COM/编码器/推扰/观测噪声。
共享DR第一名在μ=1下49/64通过、地面DR第一名36/64；其nominal起跳双脚离地峰值仅约8cm，参考为48.5cm。
μ=0.7/0.8该峰值约41.6/40.3cm。起跳幅度只是额外诊断，没有改变只按摔倒计分的要求。

两优先候选在241条中的最大实际关节超限分别10.05°/11.33°，未通过裁剪动作隐去。
未测试真机、双Dex3附加质量或go/go2切换过程。

- [完整报告与五段左右对比视频](native_mujoco_validation/report.html)
- [起跳高度对照](native_mujoco_validation/jump_quality.json)
- [审计与复现命令](native_mujoco_validation/REPRODUCE.txt)
- `native_mujoco_validation/trial_results.tar.gz`：所有原始评分与nominal物理轨迹。
- 各候选的 `evaluations/native_mujoco/summary.json`：对应结果。
'''
readme.write_text(text)
hashes = [f'{sha(p)}  {p.relative_to(PACKAGES)}' for p in sorted(PACKAGES.rglob('*'))
          if p.is_file() and p.name != 'SHA256SUMS']
(PACKAGES / 'SHA256SUMS').write_text('\n'.join(hashes) + '\n')
print(json.dumps(audit, indent=2, ensure_ascii=False))

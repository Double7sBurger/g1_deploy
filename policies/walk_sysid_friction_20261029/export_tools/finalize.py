"""Attach verified walk rollouts and training provenance to deployment bundles."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REVIEW = ROOT / 'native_mujoco_validation'
DEST = Path('/home/henry/workspace/g1_deploy/policies/walk_sysid_friction_20261029')
PYTHON = ROOT.parent / 'newton_sysid_20261028/validation/venv/bin/python'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')


def main():
    summary = json.loads((REVIEW / 'summary.json').read_text())
    assert summary['completed_trials'] == 964 and summary['pending_trials'] == 0
    entries = json.loads((DEST / 'selection.json').read_text())['entries']
    by_directory = {c['directory']: c for c in summary['candidates']}
    assert len(entries) == len(by_directory) == 4
    checks = []
    for entry in entries:
        folder = DEST / entry['directory']
        candidate = entry['candidate']
        contract = json.loads((folder / 'contract.json').read_text())
        validation = json.loads((folder / 'validation.json').read_text())
        for file, key in [(Path(candidate['checkpoint']).name, 'checkpoint_sha256'),
                          ('policy.pt', 'policy_sha256'), ('motion.npz', 'motion_sha256')]:
            assert sha(folder / file) == contract[key]
        assert validation['checkpoint_sha256'] == contract['checkpoint_sha256']
        assert validation['policy_sha256'] == contract['policy_sha256']
        assert validation['saved_training_pd_gains_verified']
        assert validation['eager_cpu_vs_jit_cpu_max_abs_error'] == 0
        scores = {}
        for suite in ('clean', 'robust'):
            records = list((folder / 'evaluations/selection' / suite).glob('*/evaluation.json'))
            assert len(records) == 2
            success = 0
            for file in records:
                data = json.loads(file.read_text())
                assert data['checkpoint_sha256'] == contract['checkpoint_sha256']
                assert data['motion_sha256'] == contract['motion_sha256']
                assert data['start_frames'][:32] == [0] * 32
                assert data['survived_steps'][:32] == [399] * 32
                success += sum(data['success'][:32])
            assert success == 64
            scores[suite] = dict(full_clip_success=success, trials=64)
        write_json(folder / 'evaluations/native_mujoco/summary.json', by_directory[entry['directory']])
        checks.append(dict(directory=entry['directory'], original_selection=scores,
                           policy_sha256=contract['policy_sha256'], checkpoint_sha256=contract['checkpoint_sha256']))
    video_checks = []
    for video in sorted((REVIEW / 'videos').rglob('*.mp4')):
        sidecar = json.loads(video.with_suffix('.json').read_text())
        assert sha(video) == sidecar['video_sha256']
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=codec_name,width,height,nb_frames,r_frame_rate:format=duration',
            '-of', 'json', str(video)]))
        stream = probe['streams'][0]
        assert (stream['width'], stream['height'], stream['nb_frames'], stream['r_frame_rate']) == (1280, 800, '250', '25/1')
        assert abs(float(probe['format']['duration']) - 10.0) < .01
        subprocess.run(['ffmpeg', '-v', 'error', '-i', str(video), '-f', 'null', '-'], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        video_checks.append(dict(file=str(video.relative_to(REVIEW)), sha256=sha(video), ffprobe=probe))
    assert len(video_checks) == 4
    write_json(REVIEW / 'videos_verified.json', video_checks)
    target = DEST / 'native_mujoco_validation'
    target.mkdir(exist_ok=True)
    for name in ('plan.json', 'status.json', 'summary.json', 'results.csv', 'report.html',
                 'rollout.py', 'batch.py', 'summarize.py', 'render.py', 'videos_verified.json', 'walk_fk_validation.json'):
        shutil.copy2(REVIEW / name, target / name)
    for name in ('models', 'videos'):
        shutil.copytree(REVIEW / name, target / name, dirs_exist_ok=True)
    raw = list((REVIEW / 'results').rglob('evaluation.json'))
    poses = list((REVIEW / 'results').rglob('rollout.npz'))
    assert len(raw) == 964 and len(poses) == 4
    with tarfile.open(target / 'trial_results.tar.gz', 'w:gz') as archive:
        for file in sorted(raw + poses):
            archive.add(file, arcname=str(file.relative_to(REVIEW)))
    with tarfile.open(target / 'trial_results.tar.gz') as archive:
        assert len(archive.getnames()) == 968
    tools = DEST / 'export_tools'
    tools.mkdir(exist_ok=True)
    for name in ('prepare.py', 'export_verify.py', 'finalize.py', 'validation_old_explicit400.json'):
        shutil.copy2(ROOT / name, tools / name)
    lines = [
        '# Walk：旧 SysID＋摩擦 DR 部署包', '',
        '包含共享摩擦与地面独立摩擦两组各两个不同训练 seed 的候选。每组原先训练 8 个 seeds，',
        '从每个 seed 的 20k／25k／29999 checkpoint 中筛选。目录 rank 保留原 Newton 筛选顺序。',
        '本次未重新训练，旧版 policies/mimic/walk、mimic_dr/walk 与 mimic_dr_contact_dr/walk 保持不变。', '',
        '| 目录 | 原 Newton clean / 完整 DR | MuJoCo 地面 μ=1 | MuJoCo 地面 U(0.5,1.5) | MuJoCo 全部测试 |',
        '|---|---|---|---|---|',
    ]
    for entry in entries:
        row = by_directory[entry['directory']]
        s = row['suites']
        total = sum(v['clip_and_hold_success'] for v in s.values())
        lines.append(f'| `{entry["directory"]}` | 64/64；64/64 | {s["clean"]["clip_and_hold_success"]}/64 | {s["friction_only_dr"]["clip_and_hold_success"]}/32 | {total}/241 |')
    lines += [
        '', '共享组两份与地面独立组第一名均 241/241；地面独立组第二名 239/241，其余两次摔倒发生在固定摩擦测试。',
        '964 次中无数值异常。地面独立组第二名实际关节位置最大超限约 26.76°，其他三份约 4.52–5.48°；',
        '上述均包含扰动试验，不能把不摔倒直接当作硬件稳定保证。优先使用各组 rank1，第二名保留作对照。',
        '', '每包包括 `policy.pt`、`contract.json`、`motion.npz/json`、原始训练 checkpoint、',
        '`training/params/env.yaml`、`agent.yaml`、SysID 参数、原始训练命令、原筛选及本次 MuJoCo 验证结果。',
        '', '在现有部署命令中使用对应目录，例如：', '',
        '```text', '--tracking policies/walk_sysid_friction_20261029/dr_shared/rank1_s49_i25000', '```', '',
        '这是有限时长的 mimic walk 动作，不是可通过速度命令无限行走的 locomotion policy。',
        '参考动作 400 帧、50Hz，最后一帧 7.98 秒；保留原始起止姿态与过渡，没有改成新的直立起步版本。',
        '', '训练使用官方 Isaac 6.2 G1 29DOF 无手资产、旧 PhysX 辨识的 26 关节参数，',
        'Newton MJWarp 400Hz 显式 SysIdDCMotor、50Hz policy、4096 环境、30000 次更新。',
        '动作变化惩罚 -0.01，脚滑奖励权重 0；摩擦范围均为 0.5–1.5。各组精确采样规则见原始 env.yaml。',
        '共享与独立摩擦是训练方案；本次 MuJoCo 验证统一机器人摩擦 1、地面 priority 1。',
        '', '原 Newton 筛选使用 shared-dr-v1 跟踪终止规则与完整 DR；MuJoCo 采用独立原生 mj_step、',
        '现有部署观测和动作接口，完整 7.98 秒动作＋2 秒末帧保持。初始纠偏步行不算失败，',
        'torso 高度低于 0.35m 或倾角大于 75°持续 0.2 秒算摔倒；数值异常另列。',
        '每份 241 次测试：参考初态 1 次、μ=1 初态扰动 64 次、其他 9 个固定摩擦各 16 次、',
        '地面随机摩擦 32 次。MuJoCo 随机评估只有地面摩擦和初态扰动，不含其他运行时 DR。',
        '', '[完整 MuJoCo 报告与四段左右对比视频](native_mujoco_validation/report.html)。',
        '各视频使用固定参考初态、地面 μ=1，左 reference、右 policy，下方显示位置误差随时间变化。',
        '误差经过任务的平移／朝向对齐；脚滑和实际关节超限另存 JSON，不作为本次不摔倒判据。',
        '', '导出已逐一比较冻结训练观测／动作／目标角与现有 G1TrackingPolicyRunner；',
        '每份还用 256 个随机输入核对 eager 与 TorchScript 一致性。没有修改部署 loop 或保护条件。',
        '硬件端使用真实 encoder，不要再次叠加本仿真中的 SysID bias 或延时。',
        '模型无 Dex3；本次没有验证双 Dex3、真机、go/go2 切换或长期站立。',
        '', '复现命令见 [REPRODUCE.txt](REPRODUCE.txt)，文件摘要见 `SHA256SUMS`。',
    ]
    (DEST / 'README.md').write_text('\n'.join(lines) + '\n')
    commands = [
        '本机导出与部署接口验证（需要已冻结的训练环境）：',
        f'python {ROOT.parent}/newton_sysid_friction8_20261028/run_local.py {ROOT}/export_verify.py --mode old_explicit400',
        '', '原生 MuJoCo 验证（已有匹配记录跳过）：',
        f'{PYTHON} {REVIEW}/batch.py --workers 4',
        f'python {REVIEW}/summarize.py',
        '', '示例单条播放：',
        f'{PYTHON} {REVIEW}/rollout.py --directory dr_shared/rank1_s49_i25000 --mu 1.0 --output /tmp/walk_native_replay',
        '', '每份训练命令见 training/experiment.json，最终环境配置见 training/params/env.yaml。',
        '训练 Python 源码归档不含 Isaac Lab 依赖和 USD。MuJoCo MJB 已包含物理模型，重放不依赖 Newton。',
        '原始 964 条评估和 4 条 nominal 轨迹在 native_mujoco_validation/trial_results.tar.gz。',
        '脚本使用本机绝对路径；迁移后调整 DEPLOY/PACKAGES 与渲染 ASSET 路径。',
    ]
    (DEST / 'REPRODUCE.txt').write_text('\n'.join(commands) + '\n')
    for entry in entries:
        folder = DEST / entry['directory']
        write_json(folder / 'manifest.json', {'files_sha256': {
            str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name != 'manifest.json'}, 'hardware_validated': False})
    write_json(DEST / 'AUDIT.json', dict(packages=checks, native_mujoco_completed=964,
        numerical_failures=summary['numerical_failures_including_post_clip'],
        raw_evaluation_count=len(raw), nominal_trajectory_count=len(poses), verified_videos=len(video_checks),
        hardware_validated=False, unchanged_deployment_code=True))
    files = [p for p in sorted(DEST.rglob('*')) if p.is_file() and p.name != 'SHA256SUMS']
    checksums = ''.join(f'{sha(p)}  {p.relative_to(DEST)}\n' for p in files)
    (DEST / 'SHA256SUMS').write_text(checksums)
    subprocess.run(['sha256sum', '--check', '--quiet', 'SHA256SUMS'], cwd=DEST, check=True)
    print(f'Verified {len(files)} files; {len(entries)} packages; {len(raw)} native MuJoCo trials.')


if __name__ == '__main__':
    main()

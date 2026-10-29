"""Write a portable index, training notes, and checksums for the CR7 bundles."""

import csv
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent
DEST = Path('/home/henry/workspace/g1_deploy/policies/cr7_friction_20261029')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def main():
    selection = json.loads((DEST / 'selection.json').read_text())
    table, rows, maxima = [], [], dict(observation=0., action=0., target=0.)
    for entry in selection['entries']:
        c, relative = entry['candidate'], entry['directory']
        folder = DEST / relative
        contract = json.loads((folder / 'contract.json').read_text())
        experiment = json.loads((folder / 'training/experiment.json').read_text())
        contract['task'] = experiment['command'][experiment['command'].index('--task') + 1]
        write_json(folder / 'contract.json', contract)
        validation = json.loads((folder / 'validation.json').read_text())
        assert validation['checkpoint_sha256'] == c['checkpoint_sha256'] == sha(folder / contract['checkpoint'])
        assert validation['policy_sha256'] == contract['policy_sha256'] == sha(folder / 'policy.pt')
        assert contract['motion_sha256'] == c['motion_sha256'] == sha(folder / 'motion.npz')
        assert contract['sysid_sha256'] == c['sysid_sha256'] == sha(folder / 'training/sysid_parameters.json')
        assert validation['eager_cpu_vs_jit_cpu_max_abs_error'] == 0
        for key in maxima:
            maxima[key] = max(maxima[key], validation['max_abs_error'][key])
        clean, robust = (c['suites'][s] for s in ('clean', 'robust'))
        table.append(f'| `{relative}` | {c["training_seed"]} | {c["iteration"]} | '
                     f'{clean["success_count"]}/64 ({clean["success_rate"]:.2%}) | '
                     f'{robust["success_count"]}/64 ({robust["success_rate"]:.2%}) |')
        rows.append(dict(directory=relative, seed=c['training_seed'], iteration=c['iteration'],
                         clean_success=clean['success_count'], robust_success=robust['success_count'],
                         clean_body_error_cm=100 * clean['body_error_m'], robust_body_error_cm=100 * robust['body_error_m'],
                         clean_contact_slip_m_s=clean['mean_contact_slip_speed_m_s'],
                         robust_contact_slip_m_s=robust['mean_contact_slip_speed_m_s'],
                         clean_max_actual_limit_excess_deg=clean['max_joint_limit_excess_deg'],
                         robust_max_actual_limit_excess_deg=robust['max_joint_limit_excess_deg'],
                         checkpoint_sha256=c['checkpoint_sha256']))
        files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*'))
                 if p.is_file() and p.name != 'manifest.json'}
        write_json(folder / 'manifest.json', {'candidate_id': c['id'], 'files_sha256': files,
                                              'hardware_validated': False})
    with (DEST / 'index.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    status = {}
    refit = ROOT.parent / 'newton_refit_fixedfriction_20261028'
    for mode in ('explicit400', 'implicit200'):
        raw = json.loads((refit / mode / 'manual_status_latest.json').read_text())
        tasks = [t for g in raw['workflow']['groups'] for t in g['tasks'] if t['name'].startswith('train-')]
        counts = {state: sum(t['status'] == state for t in tasks) for state in sorted({t['status'] for t in tasks})}
        status[mode] = {'queried_at': raw['queried_at'], 'counts': counts,
                        'completed': [t['name'] for t in tasks if t['status'] == 'COMPLETED'],
                        'failed': [t['name'] for t in tasks if t['status'] == 'FAILED'],
                        'workflow': f'g1-refit-{"exp400" if mode == "explicit400" else "imp200"}-friction-1'}
    write_json(DEST / 'new_sysid_status.json', status)
    sources = DEST / 'training_sources'
    for name in ('prepare.py', 'export_verify.py', 'finalize.py'):
        shutil.copy2(Path(__file__).with_name(name), sources / name)
    fixed = ROOT.parent / 'newton_allfive_fixedfriction_20261028'
    dr = ROOT.parent / 'newton_sysid_friction8_20261028'
    reproduce = f'''本机导出和接口校验复现（需要原冻结训练目录及 pinned venv）：
python {fixed}/run_local.py {ROOT}/deploy_export/export_verify.py --mode old_implicit200
python {dr}/run_local.py {ROOT}/deploy_export/export_verify.py --mode old_explicit400

原始训练命令：各包 training/experiment.json 的 command 数组。
原始最终配置：training/params/env.yaml 与 agent.yaml。
匹配的项目 Python 源码：training_sources/old_implicit200.tar.gz、old_explicit400.tar.gz。
这些源码包不包含 Isaac Lab、CUDA 依赖、USD 资产；版本/原始路径见 experiment.json 和 source_manifest.json。
完整 fall-only 重评命令与协议：{ROOT}/REPRODUCE.txt。
各包 evaluations 保存本候选筛选与已有独立验证的原始评分；未运行真机。
'''
    (DEST / 'REPRODUCE.txt').write_text(reproduce)
    readme = '''# CR7 固定摩擦与摩擦 DR 部署候选（2026-10-29）

共 15 个独立部署包：固定地面摩擦 0.5–1.3 每档一个最佳 checkpoint，共享摩擦 DR 和地面单独 DR 各三个。
所有包均是 **旧 PhysX SysID 参数 + Newton 训练**，没有混入新 Newton SysID 的 checkpoint。
起步时走几步、短暂跟踪偏差不直接判失败；以下排名采用全程只按摔倒终止的已完成评估。

优先顺序：`fixed/mu_0.7`，然后 `fixed/mu_0.8`。这两包在独立种子 2031/2032 的 clean 和 DR 测试中均为 64/64。
`fixed/mu_1.3` 为另一候选，独立验证分别为 61/64 和 60/64。

## 文件索引

| 相对本目录的部署路径 | 训练 seed | checkpoint 迭代 | clean 全程不摔 | DR 全程不摔 |
|---|---:|---:|---:|---:|
''' + '\n'.join(table) + '''

DR 按 checkpoint 排名前三，没有要求三个不同 seed。地面 DR 第 1、3 名来自 seed 48 的不同训练阶段；全部原始模型文件互不相同。
只纳入四批评估都数值有效的候选；异常仿真没有伪记作 0% 成功率。完整筛选范围/候选来源见 `selection.json`。

## 现有 g1_deploy 的使用方式

在原来的 `scripts/run_policy_loop.py` 启动命令中，把 tracking 参数指向整个包目录，例如：

```text
--tracking policies/cr7_friction_20261029/fixed/mu_0.7
```

其他候选只需替换此路径，例如 `policies/cr7_friction_20261029/dr_shared/rank1_s45_i20000`。
沿用已经确认的机器人网络、IMU 安装角、等待阶段和站立 policy 参数。本次没有修改 loop 或其保护阈值。
特别是 loop 原有的 `--tracking_max_tilt_error` 默认仍为 0.8 rad；它与这里离线 fall-only 评分不是同一个判断。
每包的 `contract.json` 已包含 29 关节映射、Kp/Kd、默认姿态、action scale 和 50 Hz 控制周期。
传给 `--tracking` 的是目录；`model_*.pt` 是训练 checkpoint，`policy.pt` 才是导出的 actor。

每个候选包包含：

- `policy.pt`、`contract.json`、`motion.npz`、`motion.json`：现有 tracking runner 的部署文件。
- `model_*.pt`：对应的原始训练 checkpoint。
- `training/params/env.yaml`、`agent.yaml`：保存的完整训练配置。
- `training/experiment.json`：训练任务、命令、seed、版本、摩擦设置。
- `training/sysid_parameters.json`：对应的五项关节参数；其余 training JSON 记录来源和训练汇总。
- `selection.json`、`evaluations/`：选模分数、脚滑/实际超限统计、原始评估及已有的独立验证。
- `validation.json`、`manifest.json`：离线部署接口检查及文件 SHA256。

## 三组训练设置

| 项目 | fixed | dr_shared | dr_ground |
|---|---|---|---|
| 引擎 | Newton MJWarp | Newton MJWarp | Newton MJWarp |
| 物理步长 / PD | 5 ms，原生隐式 PD | 2.5 ms，显式 SysID PD | 2.5 ms，显式 SysID PD |
| policy | 50 Hz | 50 Hz | 50 Hz |
| 摩擦 | 地面固定指定 μ；机器人 1 | 每环境一个 U(0.5,1.5)，机器人全部碰撞形状共享；地面 0 | 地面每环境 U(0.5,1.5)；机器人 1；地面 priority=1 |
| SysID | 旧五参数全用，延迟向上取整到 5 ms | 同旧五参数，2.5 ms 延迟精度 | 同左 |
| action rate 惩罚 | -0.01 | -0.01 | -0.01 |
| foot slip 惩罚 | 0 | 0 | 0 |

共有设置：官方 Isaac 6.2 `g1_29dof.usd`、无 Dex3/灵巧手、29 个身体关节；4096 环境，训练预算 30000 次更新。
这些候选从 20000、25000、29999 中选择，不保证最终迭代总是最好。参考动作均为同一份 297 帧 / 50 Hz 的 grounded CR7（完整片段 5.92 s）。
训练保留 torso 质量 log-uniform ×[0.8,1.25]、观测噪声、初始状态扰动、10–15 s 间隔 XY 速度推扰 ±0.5 m/s。
训练没有新增 COM/随机编码器 DR；固定的 SysID encoder bias 与随机编码器 DR 是两回事。
原始奖励、终止项及所有细节以各包原封不动的 env.yaml 与匹配源码为准；本次只重新选择/导出，没有重新训练或改训练终止项。

## 评分和验证范围

摔倒定义：torso 高度低于 0.35 m，或相对竖直倾角超过 75°，连续 0.2 s。
关闭原 anchor/末端跟踪误差及 500°/s 终止。小幅起步调整不触发这些摔倒条件。
每列分数来自 seed 2026/2027 各 32 个 frame-0 完整起点试验，总计 64；中间帧起点只作补充，不混入主分数。
排名先比较 clean/DR 两列中较低的全程成功率，再比较两列平均值，最后比较存活期间的身体误差。

clean 统一地面 μ=1，仍保留初始状态扰动；不是在每包自己的训练 μ 上测。
DR 评估包含质量、COM、编码器、观测噪声与 1–3 s 六维速度推扰；fixed 在地面 U(0.5,1.5) 下测，
两组旧 explicit400 DR 候选均沿用原先的“脚部共享 U(0.5,1.5)、地面 0”测试协议。
因此 fixed 和 DR 两大类的数值不是严格仅改变摩擦的对照，详见逐候选 evaluations。

已逐包验证 256 组输入上的 CPU eager / TorchScript 输出完全一致；再对照匹配的冻结 Newton 环境，
在参考帧 0、74、148、222、295 加关节扰动，比较部署 obs、action、encoder 目标。
全部通过；PD 增益逐关节与保存的训练 env.yaml 核对。验证结果见各包 validation.json。
硬件应直接输入 encoder 反馈、输出 encoder 目标；**不要再次添加 SysID bias 或软件延时**。
这些项已经描述真实硬件的特性，重复施加会改变策略的输入输出关系。

本次是离线导出/接口验证，不是新的真机或 MuJoCo 物理验证。部分评估仍记录实际关节超限，
没有通过裁剪 action/目标来隐藏；例如 μ=0.7 的筛选最大实际超限为 clean 9.46° / DR 13.44°。
`index.csv` 给出每包身体误差、接触滑速和实际超限峰值，完整记录在 evaluations。

## 新 Newton SysID

状态与查询时间见 `new_sysid_status.json`。这批仍在训练，最新完成不等于已经筛选验证。
之前只完成评估的新 400 Hz μ=0.5 两个最终 checkpoint，全程不摔仍为 0/64；不能据此断言新参数所有摩擦档都不行。
本目录先提供用户指定的旧实验候选，不将新完成但未评估模型标为最佳。

复现入口见 `REPRODUCE.txt`；保存的两份训练源码压缩包与导出脚本在 `training_sources/`。
'''
    (DEST / 'README.md').write_text(readme)
    audit = {'created_at': datetime.datetime.now().astimezone().isoformat(), 'bundles': len(rows),
             'groups': {'fixed': 9, 'shared_dr': 3, 'ground_dr': 3}, 'max_abs_interface_error': maxima,
             'cpu_eager_jit_max_abs_error': 0., 'hardware_validated': False,
             'deployment_code_changed': False, 'source_checkpoints_and_configs_sha256_verified': True}
    write_json(DEST / 'AUDIT.json', audit)
    lines = [f'{sha(p)}  {p.relative_to(DEST)}' for p in sorted(DEST.rglob('*'))
             if p.is_file() and p.name != 'SHA256SUMS']
    (DEST / 'SHA256SUMS').write_text('\n'.join(lines) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()

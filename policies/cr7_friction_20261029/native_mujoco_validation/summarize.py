"""Audit standalone MuJoCo trials and publish per-policy, per-friction outcomes."""

import csv
import datetime
import hashlib
import html
import json
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
PACKAGES = Path('/home/henry/workspace/g1_deploy/policies/cr7_friction_20261029')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    plan = json.loads((ROOT / 'plan.json').read_text())
    complete, invalid, rows = 0, 0, []
    for directory in plan['directories']:
        package = PACKAGES / directory
        contract = json.loads((package / 'contract.json').read_text())
        suites = {}
        for test in plan['tests']:
            suite = suites.setdefault(test['suite'], {'expected': 0, 'reports': []})
            suite['expected'] += 1
            p = ROOT / 'results' / directory / test['suite'] / str(test['seed']) / 'evaluation.json'
            if not p.exists():
                continue
            r = json.loads(p.read_text())
            assert r['evaluator_sha256'] == plan['evaluator_sha256'] == sha(ROOT / 'rollout.py')
            assert r['policy_sha256'] == contract['policy_sha256'] == sha(package / 'policy.pt')
            assert r['floor_friction'] == test['mu']
            assert abs(r['observed_floor_contact_friction_min'] - test['mu']) < 1e-6
            assert abs(r['observed_floor_contact_friction_max'] - test['mu']) < 1e-6
            assert r['physics'].startswith('Standalone native MuJoCo mj_step')
            if r['numerical_failure'] is None:
                assert abs(r['simulated_s'] - 7.92) < 1e-7
                assert all(v == 0 for k, v in r['warning_counts'].items() if 'BAD' in k)
            else:
                invalid += 1
            assert r['clip_numerically_valid'] == (r['numerical_failure'] is None or r['numerical_failure']['time_s'] > 5.92 + 1e-8)
            if 'rollout_sha256' in r:
                assert sha(p.with_name('rollout.npz')) == r['rollout_sha256']
            suite['reports'].append(r)
            complete += 1
        summaries = {}
        for name, s in suites.items():
            data = s['reports']
            if not data:
                continue
            valid = [r for r in data if r['clip_numerically_valid']]
            hold_valid = [r for r in data if r['numerical_failure'] is None]
            good = [r for r in valid if r['clip_success']]
            summaries[name] = {
                'expected': s['expected'], 'completed': len(data), 'clip_valid': len(valid),
                'clip_invalid': len(data) - len(valid), 'hold_invalid': len(data) - len(hold_valid),
                'clip_success': sum(r['clip_success'] for r in valid),
                'clip_success_rate_valid_only': mean(r['clip_success'] for r in valid) if valid else None,
                'clip_and_hold_success': sum(r['clip_and_hold_success'] for r in hold_valid),
                'clip_and_hold_rate_valid_only': mean(r['clip_and_hold_success'] for r in hold_valid) if hold_valid else None,
                'mean_body_error_m_successful_clips': mean(r['clip_mean_body_error_m'] for r in good) if good else None,
                'max_actual_joint_limit_excess_deg_all_valid_rollouts': max((r['max_actual_joint_limit_excess_deg'] for r in hold_valid), default=None),
                'mean_contact_foot_origin_speed_m_s_successful_clips': mean(r['mean_contact_foot_origin_speed_m_s'] for r in good
                   if r['mean_contact_foot_origin_speed_m_s'] is not None) if good else None,
                'failure_examples': [dict(seed=r['seed'], floor_friction=r['floor_friction'],
                    fall_time_s=r['fall_time_s'], numerical_failure=r['numerical_failure'])
                    for r in data if not r['clip_and_hold_success']][:8],
            }
        c = json.loads((package / 'selection.json').read_text())
        rows.append({'directory': directory, 'training_seed': c['training_seed'], 'iteration': c['iteration'],
                     'mode': 'implicit200' if directory.startswith('fixed/') else 'explicit400',
                     'checkpoint_sha256': c['checkpoint_sha256'], 'suites': summaries})
    result = {'updated_at': datetime.datetime.now().astimezone().isoformat(), 'protocol': plan,
              'completed_trials': complete, 'numerical_failures_including_post_clip': invalid,
              'pending_trials': plan['total_trials'] - complete, 'hardware_validated': False, 'candidates': rows}
    (ROOT / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    fields = ['directory', 'suite', 'expected', 'completed', 'clip_valid', 'clip_invalid', 'hold_invalid',
              'clip_success', 'clip_and_hold_success', 'mean_body_error_m_successful_clips']
    with (ROOT / 'results.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for name, s in row['suites'].items():
                writer.writerow({k: dict(directory=row['directory'], suite=name, **s).get(k) for k in fields})
    def cell(s, hold=False):
        if not s:
            return 'pending'
        bad = s['hold_invalid'] if hold else s['clip_invalid']
        good = s['clip_and_hold_success'] if hold else s['clip_success']
        text = f'{good}/{s["completed"]}'
        if bad:
            text += f'；数值异常 {bad}'
        return text
    content = '''<!doctype html><html lang="zh"><meta charset="utf-8"><title>CR7 原生 MuJoCo 验证</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 24px;color:#17202a}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}th,td{border:1px solid #cbd5e1;padding:8px;text-align:left}th{background:#edf2f7}code{background:#f1f5f9}video{width:48%;max-width:640px}p{line-height:1.6}a{color:#1565c0}</style>
<h1>CR7：独立原生 MuJoCo 闭环验证</h1>
<p>物理过程由 <code>mujoco.mj_step</code> 直接计算；测试进程不导入 Newton / Isaac Lab。使用现有 g1_deploy 的 actor、观测拼接与动作输出。
官方 Isaac 6.2 g1_29dof.usd（无 Dex3）一次性转换为 MuJoCo 物理模型，保留旧 SysID、惯量、碰撞和对应执行器设置。
这不是 g1_deploy 默认 Unitree MJCF 的测试，也不是把旧 Newton 轨迹重新渲染。</p>
<p>每条完整 CR7 5.92 秒，再保持末参考姿态 2 秒；持续 0.2 秒 torso 高度&lt;0.35m 或倾角&gt;75°记摔倒，其他跟踪误差不终止。
不自动重置，允许观察摔倒后的继续运动。表内为确认通过/总试验；数值异常另列，不等同于摔倒。
JSON另附仅数值有效试验的成功率，避免隐去异常造成误读。</p>
<p>所有试验机器人摩擦=1，地面 priority=1，实测 contact.friction 逐条核验。clean 有初态扰动但无运行时 DR。
friction_only_dr 仅地面每条 U(0.5,1.5)，不是之前包含质量/COM/编码器/推扰/噪声的完整 DR 评估。</p>
'''
    content += f'<p>完成 {complete}/{plan["total_trials"]} 条；含收尾阶段的数值异常 {invalid} 条。{html.escape(result["updated_at"])}</p>'
    content += '<h2>统一地面摩擦 1.0 与地面摩擦 DR</h2><table><tr><th>候选</th><th>精确参考初态：动作+收尾</th><th>clean 完整动作</th><th>clean 动作+收尾</th><th>摩擦 DR 完整动作</th><th>摩擦 DR 动作+收尾</th></tr>'
    for r in rows:
        s = r['suites']
        values = [cell(s.get('nominal'), True), cell(s.get('clean')), cell(s.get('clean'), True),
                  cell(s.get('friction_only_dr')), cell(s.get('friction_only_dr'), True)]
        content += '<tr><td>' + html.escape(r['directory']) + '</td>' + ''.join('<td>' + x + '</td>' for x in values) + '</tr>'
    content += '</table><h2>固定地面摩擦：完整动作+2秒收尾</h2><p>每格16条相同初态扰动。1.0列单独使用上表64条。</p><table><tr><th>候选</th>'
    mus = [.5, .6, .7, .8, .9, 1., 1.1, 1.2, 1.3, 1.5]
    content += ''.join(f'<th>{mu:.1f}</th>' for mu in mus) + '</tr>'
    for r in rows:
        content += '<tr><td>' + html.escape(r['directory']) + '</td>'
        for mu in mus:
            s = r['suites'].get('clean' if mu == 1. else f'mu_{mu:.1f}')
            color = '#fff'
            if s and s['completed']:
                score = s['clip_and_hold_success'] / s['completed']
                color = f'hsl({120*score:.0f},55%,88%)'
            content += f'<td style="background:{color}">{cell(s, True)}</td>'
        content += '</tr>'
    content += '</table>'
    if (ROOT / 'jump_quality.json').exists():
        jump = json.loads((ROOT / 'jump_quality.json').read_text())
        content += '<h2>起跳质量：不摔不等于完成相同幅度</h2><p>固定精确参考初态、摩擦1.0。在参考起跳窗口内，计算两脚物理碰撞盒的最低点离地高度，取较低脚，再取时间峰值。这里只作诊断，没有添加成功门槛。</p>'
        content += f'<p>参考双脚离地峰值 {jump["reference"]["peak_both_feet_clearance_m"]*100:.1f} cm，双脚同时高于2cm的时间 {jump["reference"]["both_feet_above_2cm_duration_s"]:.2f} s。</p>'
        content += '<table><tr><th>候选</th><th>双脚离地峰值(cm)</th><th>双脚高于2cm时长(s)</th><th>平均身体误差(cm)</th><th>最大身体误差(cm)</th></tr>'
        for j in jump['candidates']:
            content += f'<tr><td>{html.escape(j["directory"])}</td><td>{j["peak_both_feet_clearance_m"]*100:.1f}</td><td>{j["both_feet_above_2cm_duration_s"]:.2f}</td><td>{j["clip_mean_body_error_cm"]:.2f}</td><td>{j["clip_peak_body_error_cm"]:.2f}</td></tr>'
        content += '</table>'
    content += '<h2>对比视频</h2><p>左 reference，右本次原生 MuJoCo 运行；下方随时间的身体误差。固定精确参考初态，不从多次试验中挑成功条。</p>'
    for p in sorted((ROOT / 'videos').rglob('*.mp4')):
        relative = p.relative_to(ROOT)
        content += f'<p>{html.escape(str(relative.parent))}</p><video controls preload="metadata" src="{relative}"></video>'
    content += '''<h2>范围与局限</h2><p>这些是已从 Newton 评估挑出的15个 checkpoint，并非重新搜索全部训练模型。
固定摩擦组使用200Hz隐式PD，旧摩擦DR组使用400Hz显式DCMotor；都以50Hz运行policy。
额外的2秒站立不代表长期站立能力；未覆盖 go/go2 切换过程或真实传感器/通信时延。
模型无Dex3，实际装双Dex3的机器人存在质量分布差异。关节实际超限和滑脚另记，未改变或裁剪部署输出，不将不摔率当真机保证。
本次没有重新训练、修改硬件控制逻辑或发送真机命令。</p>
<p>逐条记录在 results/；模型、源代码、计划与汇总均在本目录。JSON中误差均有明确单位，脚滑统计是受力脚部原点速度，采样50Hz。
成功动作的误差用于辅助比较；数值失败与跌倒后指标不混入该均值。</p></html>'''
    (ROOT / 'report.html').write_text(content)
    print(json.dumps({'completed': complete, 'pending': result['pending_trials'], 'numerical_failures': invalid,
                     'candidates': [{'directory': r['directory'], 'clean': cell(r['suites'].get('clean'), True),
                                    'friction_dr': cell(r['suites'].get('friction_only_dr'), True)} for r in rows]}, indent=2))


if __name__ == '__main__':
    main()

"""Render standalone native MuJoCo trajectories against the reference."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('MUJOCO_GL', 'egl')
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


ROOT = Path(__file__).resolve().parent
ASSET = Path('/home/henry/workspace/isaac-spinning-up/logs/motions/asap_cr7_20261021/review/scene.xml')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    args = parser.parse_args()
    package = Path('/home/henry/workspace/g1_deploy/policies/walk_sysid_friction_20261029') / args.directory
    candidate = json.loads((package / 'selection.json').read_text())
    folder = ROOT / 'results' / args.directory / 'nominal/0'
    report = json.loads((folder / 'evaluation.json').read_text())
    assert hashlib.sha256((folder / 'rollout.npz').read_bytes()).hexdigest() == report['rollout_sha256']
    assert hashlib.sha256((package / 'policy.pt').read_bytes()).hexdigest() == report['policy_sha256']
    policy = np.load(folder / 'rollout.npz')
    reference = np.load(package / 'motion.npz')
    names = policy['body_names'].tolist()
    ref_names = json.loads((package / 'motion.json').read_text())['body_names']
    order = [ref_names.index(name) for name in names]
    n = len(policy['body_pos_w'])
    errors = policy['error_body_pos'] * 100
    assert np.isfinite(errors).all()
    pelvis = names.index('pelvis')
    indices = np.minimum(np.arange(n), len(reference['joint_pos']) - 1)
    positions = reference['body_pos_w'][indices][:, order].copy()
    rotations = Rotation.from_quat(reference['body_quat_w'][indices][:, order][..., [1, 2, 3, 0]].reshape(-1, 4))
    ref_root = Rotation.from_quat(reference['body_quat_w'][0, order[pelvis]][[1, 2, 3, 0]])
    policy_root = Rotation.from_quat(policy['body_quat_w'][0, pelvis][[1, 2, 3, 0]])
    alignment = Rotation.from_euler('Z', policy_root.as_euler('ZYX')[0] - ref_root.as_euler('ZYX')[0])
    origin = positions[0, pelvis].copy()
    destination = policy['body_pos_w'][0, pelvis].copy()
    origin[2] = destination[2] = 0
    positions = alignment.apply((positions - origin).reshape(-1, 3)).reshape(positions.shape) + destination
    quaternions = (alignment * rotations).as_quat().reshape(n, len(names), 4)[..., [3, 0, 1, 2]]
    model = mujoco.MjModel.from_xml_path(str(ASSET))
    data = mujoco.MjData(model)
    mids = model.body_mocapid[[model.body(name).id for name in names]]
    assert (mids >= 0).all()
    camera = mujoco.MjvCamera()
    camera.distance, camera.elevation, camera.azimuth = 4.0, -10, 35
    font_path = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    font = ImageFont.truetype(font_path, 24, index=2)
    small = ImageFont.truetype(font_path, 19, index=2)
    out = ROOT / 'videos' / args.directory
    out.mkdir(parents=True, exist_ok=True)
    video = out / 'native_mujoco_vs_reference.mp4'
    encoder = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', '1280x800', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '19', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
    samples = []
    sample_frames = set(int(x) // 2 * 2 for x in np.linspace(0, n - 1, 6))
    ymax = max(10, int(np.ceil(errors.max() / 5)) * 5)
    try:
        with mujoco.Renderer(model, height=504, width=600) as renderer:
            for frame in range(0, n, 2):
                canvas = Image.new('RGB', (1280, 800), (18, 25, 34))
                draw = ImageDraw.Draw(canvas)
                label = f"Walk · {candidate['arm']} · seed {candidate['training_seed']} · {candidate['iteration']} · MuJoCo · μ=1.0"
                draw.text((18, 8), label, font=font, fill='white')
                draw.text((1260, 10), f'{frame / 50:.2f} / 9.98 s · 原速', font=small, anchor='rt', fill='white')
                camera.lookat[:] = [*((positions[frame, pelvis, :2] + policy['body_pos_w'][frame, pelvis, :2]) / 2), .95]
                for column, (pos, quat, title) in enumerate([
                    (positions[frame], quaternions[frame], '左：参考动作'),
                    (policy['body_pos_w'][frame], policy['body_quat_w'][frame], '右：原生 MuJoCo 闭环策略')]):
                    data.mocap_pos[mids], data.mocap_quat[mids] = pos, quat
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    canvas.paste(Image.fromarray(renderer.render()).resize((640, 538)), (column * 640, 64))
                    draw.text((column * 640 + 18, 38), title, font=small, fill='white')
                index = min(frame, len(errors) - 1)
                draw.text((18, 610), f'身体位置误差（任务位置/朝向对齐）：{errors[index]:.2f} cm', font=small, fill=(80, 205, 255))
                if report['numerical_failure']:
                    status = f"数值异常 {report['numerical_failure']['time_s']:.2f}s"
                elif report['fall_time_s'] is not None:
                    status = f"摔倒时刻 {report['fall_time_s']:.2f}s · 继续观察"
                else:
                    status = '完整动作 + 2s 收尾未摔倒'
                draw.text((1260, 610), status, font=small, anchor='rt', fill=(200, 220, 200))
                x0, x1, y0, y1 = 70, 1220, 650, 748
                for tick in np.linspace(0, ymax, 3):
                    y = y1 - tick / ymax * (y1 - y0)
                    draw.line([(x0, y), (x1, y)], fill=(65, 75, 85))
                    draw.text((x0 - 10, y), f'{tick:g}', font=small, anchor='rm', fill=(185, 195, 205))
                for tick in range(10):
                    x = x0 + tick / 9.98 * (x1 - x0)
                    draw.text((x, y1 + 8), str(tick), font=small, anchor='mt', fill=(185, 195, 205))
                draw.text((x1, y1 + 8), '时间(s)', font=small, anchor='rt', fill=(185, 195, 205))
                points = [(x0 + i / 499 * (x1 - x0), y1 - errors[i] / ymax * (y1 - y0)) for i in range(index + 1)]
                if len(points) > 1:
                    draw.line(points, fill=(80, 205, 255), width=3)
                encoder.stdin.write(np.asarray(canvas).tobytes())
                if frame in sample_frames:
                    samples.append(canvas.resize((640, 400)))
    finally:
        encoder.stdin.close()
        assert encoder.wait() == 0
    sheet = Image.new('RGB', (1920, 800))
    for i, sample in enumerate(samples):
        sheet.paste(sample, ((i % 3) * 640, (i // 3) * 400))
    sheet.save(video.with_suffix('.jpg'), quality=95)
    video.with_suffix('.json').write_text(json.dumps({
        'directory': args.directory, 'evaluation': str(folder / 'evaluation.json'),
        'video_sha256': hashlib.sha256(video.read_bytes()).hexdigest(),
        'pose_sha256': report['rollout_sha256'], 'physics': report['physics'],
        'policy_sha256': report['policy_sha256'], 'floor_friction': report['floor_friction'],
        'clip_success': report['clip_success'], 'clip_and_hold_success': report['clip_and_hold_success'],
        'video_duration_s': len(range(0, n, 2)) / 25,
        'selection': 'Deterministic exact-reference initial state, no success-based trial selection',
        'reference_render_alignment': 'One initial XY/yaw transform; no height offset; last reference frame held after 7.98s',
    }, indent=2) + '\n')
    print(video, flush=True)


if __name__ == '__main__':
    main()

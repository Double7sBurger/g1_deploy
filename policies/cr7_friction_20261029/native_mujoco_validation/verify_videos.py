"""Verify encoded videos and extract frames covering takeoff and landing."""

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
reports = []
for video in sorted((ROOT / 'videos').rglob('*.mp4')):
    source = json.loads(video.with_suffix('.json').read_text())
    assert hashlib.sha256(video.read_bytes()).hexdigest() == source['video_sha256']
    probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=codec_name,width,height,r_frame_rate,nb_frames:format=duration',
        '-of', 'json', str(video)], text=True))
    stream = probe['streams'][0]
    assert stream['codec_name'] == 'h264' and stream['width'] == 1280 and stream['height'] == 800
    assert stream['r_frame_rate'] == '25/1' and int(stream['nb_frames']) == 199
    assert abs(float(probe['format']['duration']) - 7.96) < 1e-6
    sheet = video.with_name('encoded_jump_and_landing.jpg')
    select = '+'.join(f'eq(n,{i})' for i in (0, 40, 55, 64, 90, 198))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(video),
        '-vf', f"select='{select}',scale=640:400,tile=3x2", '-frames:v', '1', str(sheet)], check=True)
    reports.append({'video': str(video), 'sha256': source['video_sha256'], 'probe': probe,
                    'contact_sheet': str(sheet), 'physics': source['physics'], 'floor_friction': source['floor_friction']})
(ROOT / 'videos_verified.json').write_text(json.dumps(reports, indent=2) + '\n')
print('Verified', len(reports), 'encoded native MuJoCo videos.')

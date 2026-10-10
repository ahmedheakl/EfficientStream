"""Download the test set's videos and transcode them for the phone: H.264, long side <= 1280 (the latest frame uses
896), trimmed to the last question + 2 s, no audio, the source frame rate kept. Writes videos/<name>.mp4 and
manifest.json (the test set with the local file names, durations and frame rates)."""
import json, os, subprocess, zipfile, hashlib
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
from huggingface_hub import hf_hub_download
B = os.environ.get('ES_BENCH', 'es-bench').rstrip('/') + '/'
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
ts = json.load(open(B + 'testset.json'))
os.makedirs(B + 'src', exist_ok=True); os.makedirs(B + 'videos', exist_ok=True)
end = {}
for q in ts: end[q['video']] = max(end.get(q['video'], 0), q['time'])

def name(v): return hashlib.md5(v.encode()).hexdigest()[:10] + '.mp4'

def source(v):
    dst = B + 'src/' + name(v)
    if os.path.exists(dst): return dst
    bench, rest = v.split('/', 1)
    if bench == 'streambench':
        cls, f = rest.split('/', 1)
        if cls == 'Movie':
            with zipfile.ZipFile(B + 'raw/sb/Movie.zip') as z: data = z.read(f'Movie/{f}')
        else:
            url = f'https://huggingface.co/datasets/Barry-12138/StreamBench_v0.3/resolve/main/{cls}.zip'
            with RemoteZip(url, headers={'Authorization': f'Bearer {tok}'}) as z: data = z.read(f'{cls}/{f}')
        open(dst, 'wb').write(data)
    else:
        repo = {'ovobench': 'freesky/ovobench', 'streamingbench': 'freesky/streamingbench'}[bench]
        p = hf_hub_download(repo, 'videos/' + rest, repo_type='dataset', cache_dir=B + 'hfcache')
        os.symlink(os.path.realpath(p), dst)
    return dst

def work(v):
    src = source(v)
    out = B + 'videos/' + name(v)
    if not os.path.exists(out):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-t', str(end[v] + 2), '-an',
                        '-vf', "scale='if(gte(iw,ih),min(1280,iw),-2)':'if(gte(iw,ih),-2,min(1280,ih))'",
                        '-c:v', 'libx264', '-crf', '18', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                        '-fps_mode', 'passthrough', '-movflags', '+faststart', out + '.tmp.mp4'], check=True)
        os.rename(out + '.tmp.mp4', out)
    pr = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                                    'stream=width,height,avg_frame_rate:format=duration', '-of', 'json', out],
                                   capture_output=True, text=True).stdout)
    s = pr['streams'][0]; a, b = map(int, s['avg_frame_rate'].split('/'))
    return v, dict(file=name(v), width=s['width'], height=s['height'], fps=a / b,
                   duration=float(pr['format']['duration']))

with ThreadPoolExecutor(6) as ex:
    info = dict(ex.map(work, sorted(end)))
for q in ts: q.update(info[q['video']])
json.dump(ts, open(B + 'manifest.json', 'w'), indent=1)
print(len(info), 'videos;', sum(os.path.getsize(B + 'videos/' + i['file']) for i in info.values()) / 1e6, 'MB')

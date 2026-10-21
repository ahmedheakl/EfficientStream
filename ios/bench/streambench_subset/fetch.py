"""Downloads the subset's StreamBench videos (Ego / WebVideo from the dataset's zips on the Hub, Movie from the local
Movie.zip) and transcodes them for the phone: H.264, long side <= 640 (the model sees 416 x 224), up to the last
question + 2 s, no audio, the source frame rate. Writes videos/<name>.mp4 and manifest.json (the phone's
QwenStreamBench format)."""
import json, os, subprocess, zipfile, hashlib
from concurrent.futures import ThreadPoolExecutor
from remotezip import RemoteZip
B = os.environ.get('ES_BENCH', 'es-bench').rstrip('/') + '/'
S = B + 'sb_subset/'
tok = open(os.path.expanduser('~/.cache/huggingface/token')).read().strip()
subset = json.load(open(S + 'subset.json'))
os.makedirs(S + 'src', exist_ok=True)
os.makedirs(S + 'videos', exist_ok=True)


def name(v):
    return hashlib.md5(v.encode()).hexdigest()[:10] + '.mp4'


def source(v):
    dst = S + 'src/' + name(v)
    if os.path.exists(dst):
        return dst
    cls, f = v.split('/', 1)[1].split('/', 1)
    if cls == 'Movie':
        with zipfile.ZipFile(B + 'raw/sb/Movie.zip') as z:
            data = z.read(f'Movie/{f}')
    else:
        url = f'https://huggingface.co/datasets/Barry-12138/StreamBench_v0.3/resolve/main/{cls}.zip'
        with RemoteZip(url, headers={'Authorization': f'Bearer {tok}'}) as z:
            data = z.read(f'{cls}/{f}')
    open(dst + '.tmp', 'wb').write(data)
    os.rename(dst + '.tmp', dst)
    return dst


def work(v):
    src = source(v['source'])
    out = S + 'videos/' + name(v['source'])
    end = max(q['time'] for q in v['questions']) + 2
    if not os.path.exists(out):
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-i', src, '-t', str(end), '-an',
                        '-vf', "scale='if(gte(iw,ih),min(640,iw),-2)':'if(gte(iw,ih),-2,min(640,ih))'",
                        '-c:v', 'libx264', '-crf', '21', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', out + '.tmp.mp4'], check=True)
        os.rename(out + '.tmp.mp4', out)
    return dict(video=name(v['source']), source=v['source'], category=v['category'], questions=v['questions'])


with ThreadPoolExecutor(4) as ex:
    manifest = list(ex.map(work, subset))
json.dump(manifest, open(S + 'manifest.json', 'w'), indent=1)
print(len(manifest), 'videos;', sum(os.path.getsize(S + 'videos/' + m['video']) for m in manifest) / 1e6, 'MB')

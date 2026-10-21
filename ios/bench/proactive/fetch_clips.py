"""Downloads the items' OVO-Bench videos and cuts each item's clip for the phone: [start, end] of the source, H.264, long
side <= 960, no audio, source frame rate. Writes clips/pb_###.mp4 and adds the clip's duration and the event's time in
the clip (event = onset - start) to items.json."""
import json, os, subprocess
from concurrent.futures import ThreadPoolExecutor
from huggingface_hub import hf_hub_download
B = os.environ.get('ES_BENCH', 'es-bench').rstrip('/') + '/'
P = B + 'proactive_bench/'
items = json.load(open(P + 'items.json'))
os.makedirs(P + 'clips', exist_ok=True)


def source(v):
    return hf_hub_download('freesky/ovobench', 'videos/' + v.split('/', 1)[1], repo_type='dataset',
                           cache_dir=B + 'hfcache')


def cut(it):
    out = P + 'clips/' + it['clip']
    if not os.path.exists(out):
        src = source(it['video'])
        subprocess.run(['ffmpeg', '-v', 'error', '-y', '-ss', f"{it['start']:.3f}", '-i', src,
                        '-t', f"{it['end'] - it['start']:.3f}", '-an',
                        '-vf', "scale='if(gte(iw,ih),min(960,iw),-2)':'if(gte(iw,ih),-2,min(960,ih))'",
                        '-c:v', 'libx264', '-crf', '20', '-preset', 'medium', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', out + '.tmp.mp4'], check=True)
        os.rename(out + '.tmp.mp4', out)
    d = float(subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', out],
                             capture_output=True, text=True).stdout)
    return it['id'], d


# one download per video first (several items share a video), then the cuts
with ThreadPoolExecutor(4) as ex:
    list(ex.map(source, sorted({i['video'] for i in items})))
with ThreadPoolExecutor(6) as ex:
    dur = dict(ex.map(cut, items))
for it in items:
    it['duration'] = dur[it['id']]
    it['event'] = it['onset'] - it['start']
json.dump(items, open(P + 'items.json', 'w'), indent=1)
print(len(items), 'clips;', sum(os.path.getsize(P + 'clips/' + i['clip']) for i in items) / 1e6, 'MB;',
      'short clips:', [i['id'] for i in items if i['duration'] < i['event'] + 2])

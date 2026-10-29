"""Resumable parallel range downloader for the Optomec 316L dataset (scidb.cn caps ~70 KB/s per connection).

Usage: python download.py <url_list.txt> <out_root> [--only denoise|microscope|xlsx] [--workers 32]
Each file gets split into 8 MB chunks; finished chunks are logged in <file>.done so reruns resume.
"""
import os, sys, time, threading, urllib.parse, urllib.request, argparse
from concurrent.futures import ThreadPoolExecutor

CHUNK = 8 * 2**20


def target(url, root):
    p = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlparse(url).query)['path'][0]).lstrip('/')
    p = p.replace('V2/', '', 1)
    if p.startswith('coaxial_image/'):  # coaxial_image/DED_SS316L_0001/denoise.zip
        _, track, name = p.split('/')
        return os.path.join(root, 'coaxial', track + '_' + name)
    return os.path.join(root, p.replace('Zipped Microscope', 'microscope_zips'))


def size_of(url, tries=40):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
            with urllib.request.urlopen(req, timeout=60) as r:
                cr = r.headers['Content-Range']  # some files come back 200 without it
                return int(cr.split('/')[-1]) if cr else int(r.headers['Content-Length'])
        except Exception as e:
            err = e
            time.sleep(min(300, 5 * 2 ** t))
    raise RuntimeError(f'size failed: {err}')


def get_range(url, a, b, tries=40):
    for t in range(tries):
        try:
            req = urllib.request.Request(url, headers={'Range': f'bytes={a}-{b}'})
            with urllib.request.urlopen(req, timeout=120) as r:
                data = r.read()
            if len(data) == b - a + 1:
                return data
        except Exception as e:
            err = e
            time.sleep(min(300, 5 * 2 ** t))  # backoff: server answers 429 (per-IP rate limit) when hammered
    raise RuntimeError(f'failed {a}-{b}: {err}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('urls'); ap.add_argument('root')
    ap.add_argument('--only', default=None); ap.add_argument('--workers', type=int, default=32)
    a = ap.parse_args()
    urls = [u.strip().replace(' ', '%20') for u in open(a.urls) if u.strip()]
    if a.only == 'denoise':
        urls = [u for u in urls if 'coaxial_image/' in u and u.endswith('denoise.zip')]
    elif a.only == 'microscope':
        urls = sorted([u for u in urls if 'Microscope' in u])
    elif a.only == 'all':  # microscope first (biggest value), then coaxial denoise
        urls = sorted([u for u in urls if 'Microscope' in u]) + [u for u in urls if 'coaxial_image/' in u and u.endswith('denoise.zip')]
    elif a.only == 'xlsx':
        urls = [u for u in urls if u.endswith('.xlsx')]

    jobs, lock = [], threading.Lock()
    for u in urls:
        out = target(u, a.root)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        if os.path.exists(out) and not os.path.exists(out + '.done'):
            continue  # already complete
        n = size_of(u)
        if n == 0:
            print('EMPTY on server', out, flush=True); continue
        done = set(int(x) for x in open(out + '.done')) if os.path.exists(out + '.done') else set()
        if not os.path.exists(out) or os.path.getsize(out) != n:
            with open(out, 'wb') as f:
                f.truncate(n)
            done = set()
        open(out + '.done', 'a').close()
        for i, s in enumerate(range(0, n, CHUNK)):
            if i not in done:
                jobs.append((u, out, i, s, min(s + CHUNK, n) - 1))
    left = {}
    for j in jobs:
        left[j[1]] = left.get(j[1], 0) + 1
    print(f'{len(jobs)} chunks over {len(left)} files', flush=True)

    def work(j):
        u, out, i, s, e = j
        data = get_range(u, s, e)
        with lock:
            with open(out, 'r+b') as f:
                f.seek(s); f.write(data)
            with open(out + '.done', 'a') as f:
                f.write(f'{i}\n')
            left[out] -= 1
            if left[out] == 0:
                os.remove(out + '.done')
                print('DONE', out, flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(work, jobs))  # a chunk that fails 20x aborts; rerun resumes


if __name__ == '__main__':
    main()

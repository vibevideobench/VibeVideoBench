#!/usr/bin/env python3
"""Download the exact wheels pinned in requirements.lock (CPython 3.12, manylinux x86_64).

Each wheel is chosen from PyPI by the sha256 recorded in the lock, fetched with pdl (parallel ranged
download, the proxy throttles each connection) and verified again after download.

  python3 fetch_wheels.py requirements.lock OUT_DIR PDL_PATH
"""
import hashlib
import json
import re
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OK_TAGS = re.compile(r'-(py3|py2\.py3|cp3\d+)-(none|abi3|cp312)-(any|[\w.]*manylinux[\w.]*x86_64[\w.]*)\.whl$')


def parse_lock(text):
    """{name: (version, {sha256, ...})} from a `uv pip compile --generate-hashes` file."""
    pins, current = {}, None
    for line in text.splitlines():
        m = re.match(r'^([A-Za-z0-9_.-]+)==([^ \\]+)', line)
        if m:
            current = m.group(1).lower()
            pins[current] = (m.group(2), set())
        for h in re.findall(r'--hash=sha256:([0-9a-f]{64})', line):
            pins[current][1].add(h)
    return pins


def pick(name, version, hashes):
    with urllib.request.urlopen(f'https://pypi.org/pypi/{name}/{version}/json', timeout=120) as r:
        files = json.load(r)['urls']
    for f in files:
        if f['packagetype'] == 'bdist_wheel' and f['digests']['sha256'] in hashes and OK_TAGS.search(f['filename']):
            return f['filename'], f['url'], f['digests']['sha256']
    raise SystemExit(f'no compatible wheel for {name}=={version}')


def fetch(item, out, pdl):
    filename, url, digest = item
    path = out / filename
    if not path.exists():
        subprocess.run([sys.executable, pdl, url, str(path), '-n', '8'], check=True)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        path.unlink()
        raise SystemExit(f'sha256 mismatch: {filename}')
    return filename


def main():
    lock, out, pdl = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    out.mkdir(parents=True, exist_ok=True)
    items = [pick(n, v, h) for n, (v, h) in parse_lock(lock.read_text()).items()]
    with ThreadPoolExecutor(6) as pool:
        for name in pool.map(lambda i: fetch(i, out, pdl), items):
            print('ok', name)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""pdl: parallel ranged HTTP download (stdlib only).

The proxy on this machine throttles every connection (~70 KB/s) but not the total, so big files
(models, footage, fonts) download much faster split into N concurrent Range requests.

  pdl URL OUT [-n 16]          # resumes nothing; writes OUT atomically (OUT.part -> OUT)
  pdl --skip-existing URL OUT  # do nothing if OUT already exists and is non-empty
"""
import argparse
import os
import shutil
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

UA = {"User-Agent": "pdl/1.0"}


def resolve(url):
    """Follow redirects; return (final_url, size or None, accepts_ranges)."""
    req = urllib.request.Request(url, headers={**UA, "Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        final = r.geturl()
        cr = r.headers.get("Content-Range")  # bytes 0-0/12345
        if r.status == 206 and cr and "/" in cr and cr.rsplit("/", 1)[1].isdigit():
            return final, int(cr.rsplit("/", 1)[1]), True
        n = r.headers.get("Content-Length")
        return final, int(n) if n and n.isdigit() else None, False


def fetch_range(url, path, start, end, tries=8):
    for k in range(tries):
        try:
            got = 0
            req = urllib.request.Request(url, headers={**UA, "Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "r+b") as f:
                f.seek(start)
                while True:
                    b = r.read(1 << 16)
                    if not b:
                        break
                    f.write(b)
                    got += len(b)
            if got == end - start + 1:
                return got
            start += got  # short read: continue from where it stopped
        except Exception as e:  # noqa: BLE001 - retry on any network error
            if k == tries - 1:
                raise
            time.sleep(2 + 2 * k)
    raise RuntimeError(f"range {start}-{end} incomplete")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("-n", "--connections", type=int, default=16)
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    if a.skip_existing and os.path.exists(a.out) and os.path.getsize(a.out) > 0:
        print(f"pdl: exists {a.out}")
        return
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    part = a.out + ".part"
    t0 = time.time()
    final, size, ranged = resolve(a.url)
    if not ranged or not size:
        with urllib.request.urlopen(urllib.request.Request(final, headers=UA), timeout=300) as r, open(part, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    else:
        with open(part, "wb") as f:
            f.truncate(size)
        n = max(1, min(a.connections, size // (256 << 10) or 1))
        step = -(-size // n)
        spans = [(i, min(i + step, size) - 1) for i in range(0, size, step)]
        with ThreadPoolExecutor(len(spans)) as ex:
            list(ex.map(lambda s: fetch_range(final, part, *s), spans))
    os.replace(part, a.out)
    sz = os.path.getsize(a.out)
    dt = time.time() - t0
    print(f"pdl: {a.out}  {sz / 1e6:.1f} MB in {dt:.0f}s ({sz / 1e6 / max(dt, 1e-3):.2f} MB/s)")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Extract the font files we ship (static TTF/OTF + license) from the downloaded release zips.

  python3 -I extract_fonts.py ZIP_DIR FONT_DIR
"""
import fnmatch
import sys
import zipfile
from pathlib import Path

# zip name -> (target folder, member patterns)
SELECT = {
    'smiley-sans-v2.0.1.zip': ('smiley-sans', ['SmileySans-Oblique.ttf', 'LICENSE*']),
    'Inter-4.1.zip': ('inter', ['extras/ttf/Inter-*.ttf', 'LICENSE.txt']),
    'JetBrainsMono-2.304.zip': ('jetbrains-mono', ['fonts/ttf/JetBrainsMono-*.ttf', 'OFL.txt']),
}


def main():
    zips, fonts = Path(sys.argv[1]), Path(sys.argv[2])
    for name, (folder, patterns) in SELECT.items():
        dest = fonts / folder
        dest.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zips / name) as z:
            members = [m for m in z.namelist() if not m.endswith('/')
                       and any(fnmatch.fnmatch(m, p) or fnmatch.fnmatch(m.rsplit('/', 1)[-1], p) for p in patterns)]
            if not members:
                raise SystemExit(f'nothing matched in {name}')
            for m in members:
                (dest / m.rsplit('/', 1)[-1]).write_bytes(z.read(m))
        print(folder, len(members), 'files')


if __name__ == '__main__':
    main()

"""Check the pinned Python packages and print the toolchain versions."""
import subprocess
import sys

import cairocffi
import cv2
import numpy
import PIL
from importlib.metadata import version

expected = {'numpy': '2.5.4', 'pillow': '12.3.0', 'opencv-python-headless': '4.14.0.94',
            'cairocffi': '1.7.1', 'playwright': '1.62.0', 'uv': '0.13.1'}
wrong = {k: version(k) for k, v in expected.items() if version(k) != v}
if wrong or sys.version_info[:2] != (3, 12):
    sys.exit(f'unexpected versions: {wrong} python={sys.version}')
assert numpy.__version__ and PIL.__version__ and cv2.__version__ and cairocffi.cairo_version()
node = subprocess.check_output(['node', '--version'], text=True).strip()
chrome = subprocess.check_output(['/opt/chromium/chrome', '--version'], text=True).strip()
print(f'python {sys.version.split()[0]}, cairo {cairocffi.cairo_version_string()}, node {node}, {chrome}')

"""Playwright (Python): fonts, GSAP and Three.js (+addons) from /opt/vendor, and whether WebGL works.

Pages are served from a fake origin through page.route, so ES modules load without a web server.
Tries the default headless shell and full Chrome (CHROME_BIN), each with and without
--enable-unsafe-swiftshader, and reports which combinations get a WebGL context.
"""
import json
import mimetypes
import os
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOTS = {'/vendor/': Path('/opt/vendor'), '/': Path('/smoke')}


def serve(route):
    path = route.request.url.split('http://smoke.local', 1)[1].split('?')[0]
    for prefix, root in ROOTS.items():
        if path.startswith(prefix):
            file = root / path[len(prefix):]
            if file.is_file():
                return route.fulfill(body=file.read_bytes(),
                                     content_type=mimetypes.guess_type(file.name)[0] or 'application/octet-stream')
    route.fulfill(status=404)


results = {}
with sync_playwright() as p:
    for name, kwargs in [('headless-shell', {}), ('chrome', {'executable_path': os.environ['CHROME_BIN']})]:
        for flag in ([], ['--enable-unsafe-swiftshader']):
            browser = p.chromium.launch(args=['--disable-dev-shm-usage', *flag], **kwargs)
            page = browser.new_page(viewport={'width': 900, 'height': 560})
            page.route('http://smoke.local/**', serve)
            page.goto('http://smoke.local/page.html')
            page.wait_for_function('document.title.startsWith("{")', timeout=30000)
            info = json.loads(page.title())
            label = name + (' +swiftshader-flag' if flag else '')
            results[label] = info['webgl']
            if name == 'chrome' and (flag or info['webgl']) and not Path('browser.png').exists():
                page.screenshot(path='browser.png')
            browser.close()
assert info['gsap'] == '3.12.5' and info['three'] == '160' and info['addons'], info
if not any(results.values()):
    raise SystemExit(f'no WebGL in any configuration: {results}')
print('webgl: ' + '; '.join(f'{k}={"yes" if v else "NO"}' for k, v in results.items()) + f' | renderer {next(v for v in results.values() if v)}')

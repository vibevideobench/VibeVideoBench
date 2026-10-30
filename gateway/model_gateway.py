#!/usr/bin/env python3
"""Per-case model gateway for sandboxed agent runs (host side; the provider key never enters a container).

Each agent harness inside the sandbox talks to this gateway instead of the provider, with a per-case token:

  POST /case/<system>/<case>/v1/responses                                   (codex harness, OpenAI Responses)
  POST /case/<system>/<case>/v1beta/models/<model>:streamGenerateContent    (antigravity harness, Gemini)
  POST /case/<system>/<case>/v1/messages[/count_tokens]                     (claude harness, Anthropic Messages)
  POST /case/<system>/<case>/v1/chat/completions                            (dsh, kimi (Kimi Code), zcode (ZCode): OpenAI
                                                                            Chat Completions)
  GET  /health                                                              gateway status

The gateway forwards to <upstream><path> of the system with the provider key added.

Each system names its provider in systems.json: "upstream" (API base URL, without the /v1 part the harness sends),
"key_file" (relative paths are taken from the directory of systems.json) and, for a KEY=value env file such as
keys.env, "key_var" or a list "key_vars"; without key_var(s) the whole file is the key. --upstream / --key-file apply
to systems that name none. With several keys the first one with credit is used; a key answered with an out-of-credit
error is skipped (the same request is retried on the next key) and tried again after --exhausted-recheck seconds.
<runs-dir>/key_state.json shows which keys are set aside (names only): schedule.py stops new launches of a system
whose keys are all set aside.

Optional per system: "proxy" (HTTP proxy URL for this upstream; otherwise the environment's http(s)_proxy, if any,
else direct), "upstream_model" (send this model id upstream instead of the one the harness names), "strip_path_prefix" (e.g. "/v1":
removed from the harness's path before it is appended, for an upstream base that already holds the API version), "extra_body"
(fields added to every request).

A case is served only if <runs-dir>/<system>/<case>/ has auth.json (its token) and run.json (its deadline).
The gateway checks the token, the deadline and the system's model/effort, blocks paid image/audio generation,
spaces request starts, and records each request, the raw streamed response (key redacted) and a usage record
under <case>/native_api/<index>/.
"""
import argparse
import base64
import copy
import hashlib
import http.client
import http.server
import json
import os
from pathlib import Path
import re
import signal
import socket
import threading
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--runs-dir', type=Path, required=True, help='holds <system>/<case>/{auth.json,run.json}')
ap.add_argument('--systems', type=Path, required=True,
                help='JSON {system: {"model", "effort", "harness": codex|antigravity|claude|dsh|kimi|zcode, "upstream", '
                     '"key_file", "key_var"|"key_vars", ["proxy", "upstream_model", "strip_path_prefix", "extra_body"]}}')
ap.add_argument('--key-file', type=Path, help='key of the systems without their own key_file')
ap.add_argument('--port', type=int, default=18885)
ap.add_argument('--upstream', help='API base URL of the systems without their own "upstream"')
ap.add_argument('--gap', type=float, default=2.05,
                help='minimum seconds between two request starts to the same upstream model (~30 RPM). Per model, '
                     'not per upstream, so parallel runs of different models do not queue behind each other. The wait '
                     'is recorded as queued_s (not counted as model time)')
ap.add_argument('--tcp-keepalive', type=int, default=30,
                help='seconds between TCP keepalive probes on upstream connections (0 = off)')
ap.add_argument('--read-timeout', type=float, default=1200,
                help='seconds without upstream bytes before a request is abandoned (always bounded by the case deadline). '
                     'Long, because Gemini sends a function call only once it is complete, so writing a large file '
                     'is silent for minutes')
ap.add_argument('--gemini-gap', type=float, default=6.1, help='extra spacing between antigravity requests')
ap.add_argument('--rate-limit-wait', type=float, default=240,
                help='on an upstream HTTP 429 rate limit (not out-of-credit), back off and retry here while the request has '
                     'been in the gateway less than this many seconds; that whole time is recorded in queued_s (below the '
                     'agents\' 300 s stream idle timeouts). Some harnesses give up after 2 retries of their own')
ap.add_argument('--exhausted-recheck', type=float, default=1800,
                help='seconds before a key out of credit is tried again')
ap.add_argument('--only', type=lambda x: x.split(','),
                help='serve only these systems (comma-separated): a second gateway on the same --runs-dir, which leaves the '
                     'other systems\' entries of key_state.json to the first')
A = ap.parse_args()

SYSTEMS = json.loads(A.systems.read_text())
SYSTEMS = {s: c for s, c in SYSTEMS.items() if not s.startswith('_') and isinstance(c, dict)}   # "_comment" etc.
if A.only:
    SYSTEMS = {s: SYSTEMS[s] for s in A.only}


def key_names(cfg):
    return cfg.get('key_vars') or ([cfg['key_var']] if cfg.get('key_var') else ['key_file'])


def key_path(cfg):
    if cfg.get('key_file'):
        path = Path(cfg['key_file']).expanduser()
        return path if path.is_absolute() else A.systems.resolve().parent / path
    return A.key_file


def load_keys(cfg):
    path = key_path(cfg)
    if not path:
        raise ValueError('no key_file in systems.json and no --key-file')
    if not (cfg.get('upstream') or A.upstream):
        raise ValueError('no upstream in systems.json and no --upstream')
    text = path.read_text()
    if key_names(cfg) == ['key_file']:
        return [text.strip()]
    found = [re.search(r'^\s*(?:export\s+)?' + re.escape(n) + r'\s*=\s*[\'"]?([^\'"\s]+)', text, re.M)
             for n in key_names(cfg)]
    missing = [n for n, m in zip(key_names(cfg), found) if not m]
    if missing:
        raise ValueError(f'{missing} not in {path}')
    return [m.group(1) for m in found]


KEYS, KEY_ERRORS = {}, {}   # system -> keys; system -> why its keys did not load
KEY_LOCK = threading.Lock()
EXHAUSTED = {}   # (system, key index) -> (when it was set aside, when to try it again, why)
SECRET_LOCK = threading.Lock()
SECRETS = []     # every key this gateway has used, longest first, redacted from what it writes


def remember(*values):
    with SECRET_LOCK:
        new = {v for v in values if isinstance(v, str) and v and v not in SECRETS}
        if new:
            SECRETS[:] = sorted(set(SECRETS) | new, key=len, reverse=True)


def credentials(system):
    """The system's keys, loaded on first use and retried while missing: that system is refused, the others are
    still served."""
    with KEY_LOCK:
        if system not in KEYS:
            try:
                KEYS[system] = load_keys(SYSTEMS[system])
                KEY_ERRORS.pop(system, None)
                remember(*KEYS[system])
            except Exception as exc:
                KEY_ERRORS[system] = f'{type(exc).__name__}: {exc}'
        return KEYS.get(system)


def pick_key(ids, skip=()):
    """Index (into ids, see key_ids) of the first key not set aside for exhausted credit, else of the one that is
    available again soonest; None if all are in skip."""
    with KEY_LOCK:
        now = time.time()
        free = [i for i in range(len(ids)) if i not in skip]
        again = lambda i: EXHAUSTED.get(ids[i], (0, 0))[1]
        ok = [i for i in free if again(i) <= now]
        return (ok or sorted(free, key=again) or [None])[0]


def key_ids(system, keys):
    return [(system, i) for i in range(len(keys))]


def mark_exhausted(kid, until=None, why='credit'):
    with KEY_LOCK:
        now = time.time()
        EXHAUSTED[kid] = (now, min(until or now + A.exhausted_recheck, now + 8 * 86400), why)
        save_key_state()


def save_key_state():
    """<runs-dir>/key_state.json for schedule.py: per system with a key set aside, its key names and which are set
    aside since / until when. The caller holds KEY_LOCK."""
    now, state = time.time(), {}
    for s, cfg in SYSTEMS.items():
        names = key_names(cfg)
        ids = dict(zip(names, key_ids(s, names)))
        marks = {n: EXHAUSTED[k] for n, k in ids.items() if k in EXHAUSTED}
        if marks:
            state[s] = {'keys': names, 'exhausted_at': {n: m[0] for n, m in marks.items()},
                        'available_at': {n: m[1] for n, m in marks.items()}, 'why': {n: m[2] for n, m in marks.items()}}
    if A.only:     # another gateway serves the other systems
        try:
            state = {**{s: v for s, v in read(A.runs_dir / 'key_state.json')['systems'].items() if s not in SYSTEMS}, **state}
        except (OSError, ValueError, KeyError):
            pass
    save(A.runs_dir / 'key_state.json', {'updated': now, 'recheck_after_s': A.exhausted_recheck, 'systems': state})


def keepalive(sock):
    """Some network paths drop a connection after a few minutes without bytes (e.g. while Gemini generates a large
    function call it only streams when complete); TCP keepalive probes keep the idle flow alive."""
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, A.tcp_keepalive)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, A.tcp_keepalive)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 10)


class KeepAliveHTTPConnection(http.client.HTTPConnection):
    def connect(self):
        super().connect()
        keepalive(self.sock)


class KeepAliveHTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        super().connect()
        keepalive(self.sock)


class KeepAliveHTTPHandler(urllib.request.HTTPHandler):
    def http_open(self, req):
        return self.do_open(KeepAliveHTTPConnection, req)


class KeepAliveHTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(KeepAliveHTTPSConnection, req, context=self._context)


OPENERS = {}
OPENER_LOCK = threading.Lock()


def opener_for(proxy):
    """An opener through proxy (a URL), or with proxy None through the environment's http(s)_proxy (direct if unset)."""
    with OPENER_LOCK:
        if proxy not in OPENERS:
            OPENERS[proxy] = urllib.request.build_opener(
                urllib.request.ProxyHandler({'http': proxy, 'https': proxy} if proxy else None),
                *([KeepAliveHTTPHandler(), KeepAliveHTTPSHandler()] if A.tcp_keepalive else []))
        return OPENERS[proxy]


def open_retrying(opener, req, timeout, row=None):
    """opener.open, again when the proxy refused the tunnel (nothing reached the upstream yet).

    Every attempt opens a fresh copy of req: urllib's ProxyHandler rewrites the Request it opens, and reopening the
    same object can send it, Authorization header included, as plain HTTP inside the tunnel."""
    for attempt in range(5):
        sent = copy.copy(req)
        sent.headers, sent.unredirected_hdrs = dict(req.headers), dict(req.unredirected_hdrs)
        if getattr(sent, '_tunnel_host', None) or sent.type != urllib.parse.urlsplit(sent.full_url).scheme:
            raise urllib.error.URLError('request object was already opened; refusing to resend it')
        try:
            return opener.open(sent, timeout=timeout)
        except urllib.error.HTTPError:
            raise
        except urllib.error.URLError as exc:
            if 'Tunnel connection failed' not in str(exc.reason) or attempt == 4:
                raise
            if row is not None:
                row['tunnel_retries'] = row.get('tunnel_retries', 0) + 1
            time.sleep(0.3)


def out_of_balance(text):
    """An upstream 429 that is the account's balance, not a rate limit: no point waiting it out in this request
    (e.g. "credit balance exhausted"; GLM: code 1113 余额不足或无可用资源包, as JSON escapes or not)."""
    return 'credit' in text.lower() or '余额不足' in text or '\\u4f59\\u989d\\u4e0d\\u8db3' in text.lower()


SHRINK_CASES = set()    # (system, case) an upstream refused as too large: their images go as JPEG from then on
SHRINK_CACHE = {}       # sha1 of a data URL -> its JPEG data URL (or None: keep it), so history re-encodes identically


def shrink_images(body, row, min_chars=256 * 1024):
    """Re-encode large lossless data-URL images (codex's view_image sends rendered frames as PNG, often 3-7 MB each) as
    JPEG quality 95, 4:4:4, same pixels: ~13x smaller. For an upstream whose request size cap is far below the API's
    own (some OpenAI-compatible endpoints answer 413 above ~8-12 MB, where the OpenAI API takes 50 MB). Images with
    transparency stay as they are. Needs Pillow (without it nothing is re-encoded). Records what it did on the row;
    True if the body got smaller."""
    try:
        from PIL import Image
    except ImportError:
        row['images_reencoded_error'] = 'Pillow not installed'
        return False
    import io
    done = row.get('images_reencoded') or {'images': 0, 'chars_before': 0, 'chars_after': 0}

    def jpeg(url):
        digest = hashlib.sha1(url.encode()).hexdigest()
        if digest not in SHRINK_CACHE:
            out = None
            try:
                im = Image.open(io.BytesIO(base64.b64decode(url.split(',', 1)[1])))
                if 'A' not in im.getbands() or im.getchannel('A').getextrema()[0] == 255:
                    buf = io.BytesIO()
                    im.convert('RGB').save(buf, 'JPEG', quality=95, subsampling=0)
                    out = 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()
                    out = out if len(out) < len(url) else None
            except Exception:
                out = None
            if len(SHRINK_CACHE) > 2000:
                SHRINK_CACHE.clear()
            SHRINK_CACHE[digest] = out
        return SHRINK_CACHE[digest]

    def walk(x):
        items = x.items() if isinstance(x, dict) else enumerate(x) if isinstance(x, list) else ()
        for k, v in list(items):
            if (isinstance(v, str) and len(v) >= min_chars and v.startswith('data:image/')
                    and not v.startswith('data:image/jpeg')):
                new = jpeg(v)
                if new:
                    x[k] = new
                    done['images'] += 1
                    done['chars_before'] += len(v)
                    done['chars_after'] += len(new)
            elif isinstance(v, (dict, list)):
                walk(v)
    before = done['images']
    walk(body)
    if done['images'] > before:
        row['images_reencoded'] = done
    return done['images'] > before


HISTORY_LOCK = threading.Lock()
AFFINITY_LOCK = threading.Lock()
RATE_LOCK = threading.Lock()
NEXT = {}        # (upstream, model) -> earliest next request start
GROUP_NEXT = {}
EXCLUDED_HEADERS = {'authorization', 'x-goog-api-key', 'x-api-key', 'host', 'content-length', 'connection',
                    'transfer-encoding', 'accept-encoding', 'cookie', 'proxy-authorization'}


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f'.{os.getpid()}.{time.time_ns()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def affinity_key(C, original, renew=False):
    """Some API aggregators pin a conversation (its prompt_cache_key) to one upstream channel; if that channel is
    disabled they answer 403 'channel affinity ... disabled' forever, so such a conversation gets a fresh routing key."""
    with AFFINITY_LOCK:
        path = C / 'gateway_affinity.json'
        keys = read(path) if path.exists() else {}
        if renew:
            n = int(keys[original].rpartition('-r')[2]) + 1 if original in keys else 1
            keys[original] = f'{original}-r{n}'
            save(path, keys)
        return keys.get(original, original)


def redact(text):
    for key in list(SECRETS):
        text = text.replace(key, '[REDACTED]')
    return text


def redact_bytes(data):
    for key in list(SECRETS):
        data = data.replace(key.encode(), b'[REDACTED]')
    return data


def check_body(cfg, path, body):
    """Return (status, message) to reject, or None. May adapt a Gemini thinkingBudget to thinkingLevel."""
    if cfg['harness'] == 'codex':
        if path != '/v1/responses':
            return 403, 'Only selected model inference is provided'
        if body.get('model') != cfg['model'] or body.get('reasoning', {}).get('effort') != cfg['effort']:
            return 400, 'Unexpected selected model/effort'
        if any(t.get('type') == 'image_generation' for t in body.get('tools', [])):
            return 403, 'Paid image generation is not enabled'
        return None
    if cfg['harness'] == 'claude':
        if path not in ['/v1/messages', '/v1/messages/count_tokens']:
            return 403, 'Only selected model inference is provided'
        if body.get('model') != cfg['model']:
            return 400, 'Unexpected selected model'
        return None
    if cfg['harness'] in ('kimi', 'zcode'):     # OpenAI Chat Completions; the effort, when sent, is the system's
        if path != '/v1/chat/completions':
            return 403, 'Only selected model inference is provided'
        if body.get('model') != cfg['model'] or body.get('reasoning_effort', cfg['effort']) != cfg['effort']:
            return 400, 'Unexpected selected model/effort'
        return None
    if cfg['harness'] == 'dsh':
        if path != '/v1/chat/completions':
            return 403, 'Only selected model inference is provided'
        if body.get('model') != cfg['model'] or body.get('reasoning_effort', cfg['effort']) != cfg['effort']:
            return 400, 'Unexpected selected model/effort'
        if cfg['effort'] == 'off' and ((body.get('thinking') or {}).get('type') != 'disabled' or 'reasoning_effort' in body):
            return 400, 'Expected thinking disabled'
        return None
    if path not in [f"/v1beta/models/{cfg['model']}:streamGenerateContent",
                    f"/v1beta/models/{cfg['model']}:generateContent"]:
        return 403, 'Only selected model inference is provided'
    gen = body.setdefault('generationConfig', {})
    if any(str(x).upper() in ['IMAGE', 'AUDIO'] for x in gen.get('responseModalities', [])):
        return 403, 'Paid media generation is not enabled'
    thinking = gen.setdefault('thinkingConfig', {})
    level = cfg['effort'].upper()
    if 'thinkingBudget' in thinking:
        body['_adaptation'] = {'removed_thinkingBudget': thinking.pop('thinkingBudget'), 'thinkingLevel': level}
        thinking['thinkingLevel'] = level
    if thinking.get('thinkingLevel') != level:
        return 400, f'Expected {level} effort'
    return None


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

    def log_message(self, *args):
        pass

    def reply(self, status, obj):
        raw = json.dumps(obj).encode()
        try:
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(raw)))
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(raw)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        self.close_connection = True

    def do_GET(self):
        if urllib.parse.urlsplit(self.path).path != '/health':
            return self.reply(404, {'error': 'not found'})
        for s in SYSTEMS:
            credentials(s)
        self.reply(200, {'ok': True, 'pid': os.getpid(), 'runs_dir': str(A.runs_dir), 'systems': SYSTEMS,
                         'unavailable': {s: KEY_ERRORS[s] for s in SYSTEMS if s in KEY_ERRORS},
                         'env_proxy': os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')})

    def do_POST(self):
        try:
            self.post()
        except Exception:
            traceback.print_exc()
            try:
                self.reply(502, {'error': 'gateway internal error'})
            except Exception:
                pass
            self.close_connection = True

    def post(self):
        parts = urllib.parse.urlsplit(self.path)
        bits = parts.path.split('/')
        if len(bits) < 6 or bits[1] != 'case' or bits[2] not in SYSTEMS:
            return self.reply(404, {'error': 'unknown route or system'})
        system, case = bits[2:4]
        if not case or not all(c.isalnum() or c in '_-' for c in case):
            return self.reply(400, {'error': 'bad case id'})
        cfg = SYSTEMS[system]
        C = A.runs_dir / system / case
        if not (C / 'auth.json').exists() or not (C / 'run.json').exists():
            return self.reply(404, {'error': 'case not prepared'})
        supplied = [self.headers.get('Authorization', '').removeprefix('Bearer '), self.headers.get('x-goog-api-key', ''),
                    self.headers.get('x-api-key', '')]
        supplied += urllib.parse.parse_qs(parts.query).get('key', [])
        if read(C / 'auth.json')['token'] not in supplied:
            return self.reply(403, {'error': 'bad token'})
        run = read(C / 'run.json')
        # the model and effort this run was started with (run_case records them): a system's can change between runs,
        # and runs already going keep theirs
        cfg = dict(cfg, effort=run.get('effort') or cfg['effort'], model=run.get('model') or cfg['model'])
        deadline = run['deadline']
        if time.time() >= deadline:
            return self.reply(408, {'error': 'Case deadline reached'})
        keys = credentials(system)
        if not keys:
            return self.reply(503, {'error': f'credentials of {system} unavailable: {KEY_ERRORS.get(system)}'})
        ids = key_ids(system, keys)
        names = key_names(cfg)
        path = '/' + '/'.join(bits[4:])
        try:
            body = json.loads(self.rfile.read(int(self.headers.get('Content-Length', '0'))))
        except ValueError:
            return self.reply(400, {'error': 'bad json'})
        rejected = check_body(cfg, path, body)
        adaptation = body.pop('_adaptation', None)
        if rejected:
            return self.reply(rejected[0], {'error': rejected[1]})
        # Optional per system (absent = unchanged): serve it through another upstream model id than the one its
        # harness names, and add request fields (e.g. a provider's tool-call streaming flag, so long tool-call
        # arguments stream instead of arriving as one chunk after the harness's stream-idle timeout).
        if cfg.get('upstream_model') and 'model' in body:
            body['model'] = cfg['upstream_model']
        if cfg.get('extra_body'):
            body.update(cfg['extra_body'])
        # A run directory can be replaced while a request of the previous attempt is still streaming (aborted and
        # relaunched case), so number requests after the highest existing directory and tag each with its run.
        run_started = run.get('started')
        with HISTORY_LOCK:
            history = C / 'native_api'
            history.mkdir(parents=True, exist_ok=True)
            index = 1 + max([int(d.name) for d in history.iterdir() if d.name.isdigit()], default=-1)
            while True:
                directory = history / f'{index:05}'
                try:
                    directory.mkdir()
                    break
                except FileExistsError:
                    index += 1
            save(directory / 'request.json', body)
        pairs = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k.lower() not in ['key', 'api_key', 'apikey']]
        upstream_base = (cfg.get('upstream') or A.upstream).rstrip('/')
        strip = cfg.get('strip_path_prefix')        # e.g. "/v1" for an upstream whose base already holds the version
        url = upstream_base + (path[len(strip):] if strip and path.startswith(strip + '/') else path) + ('?' + urllib.parse.urlencode(pairs) if pairs else '')
        opener = opener_for(cfg.get('proxy'))
        headers = {k: v for k, v in self.headers.items() if k.lower() not in EXCLUDED_HEADERS}
        headers['Content-Type'] = 'application/json'

        def use_key(index):
            for name in [k for k in headers if k.lower() in ('authorization', 'x-api-key', 'x-goog-api-key')]:
                del headers[name]       # what the previous key of this request set
            key = keys[index]
            if cfg['harness'] == 'claude':
                headers['x-api-key'] = key
            elif cfg['harness'] == 'antigravity':
                headers['x-goog-api-key'] = key
            else:
                headers['Authorization'] = 'Bearer ' + key

        row = {'system': system, 'case_id': case, 'index': index, 'path': path, 'started': time.time(),
               'requested_model': cfg['model'], 'adaptation': adaptation, 'upstream_request_sent': False, 'raw_usage': None}
        if cfg['harness'] == 'claude':
            row.update(sent_effort=(body.get('output_config') or {}).get('effort'),
                       anthropic_beta=self.headers.get('anthropic-beta'))
        sent = False
        try:
            with RATE_LOCK:     # book a start slot; wait outside the lock so one queue never blocks another
                now = time.time()
                start = max(now, NEXT.get((upstream_base, cfg['model']), 0), GROUP_NEXT.get(system, 0))
                NEXT[(upstream_base, cfg['model'])] = start + A.gap
                GROUP_NEXT[system] = start + (A.gemini_gap if cfg['harness'] == 'antigravity' else 0)
            row['queued_s'] = round(start - now, 3)
            if start > now:
                time.sleep(min(start - now, max(0, deadline - now)))
            if time.time() >= deadline:
                raise TimeoutError('Case deadline reached while queued')
            original = body.get('prompt_cache_key')
            attempt, spent = 0, set()
            if (system, case) in SHRINK_CASES:     # refused as too large before: skip the 413 round trip
                shrink_images(body, row)
            while True:
                if original:
                    body['prompt_cache_key'] = affinity_key(C, original)
                index = pick_key(ids, spent)
                if len(keys) > 1:
                    row['key'] = names[index]
                use_key(index)
                req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers)
                row.update(upstream_request_sent=True, upstream_started=time.time())
                try:
                    upstream = open_retrying(opener, req, max(.1, min(A.read_timeout, deadline - time.time())), row)
                    break
                except urllib.error.HTTPError as exc:
                    exc.text = exc.read().decode(errors='replace')
                    if ((exc.code == 403 and 'has been suspended' in exc.text) or
                            (exc.code == 500 and 'failed to exchange JWT for access token' in exc.text)) \
                            and row.get('suspended_retries', 0) < 8 and time.time() < deadline - 15:
                        # an aggregator's key pool with one suspended provider key behind it: refused unprocessed,
                        # the next attempt usually gets another key
                        row['suspended_retries'] = row.get('suspended_retries', 0) + 1
                        time.sleep(min(2 * row['suspended_retries'], 10))
                        row['queued_s'] = round(time.time() - row['started'], 3)    # refused attempts + backoff: not
                        continue                                                    # model time (run_case credits it)
                    if exc.code == 413 and not row.get('images_reencoded') and shrink_images(body, row):
                        # over the upstream's request size cap only because of PNG frames: resend them as JPEG
                        SHRINK_CASES.add((system, case))
                        row['reencoded_after_413'] = True
                        continue
                    limited = exc.code == 429 and not out_of_balance(exc.text)
                    waited = time.time() - row['started']    # all gateway time so far: queue, rejected attempts, backoff
                    if limited and waited < A.rate_limit_wait and time.time() < deadline - 5:
                        pause = min(5 * 2 ** row.get('rate_limit_retries', 0), 30, A.rate_limit_wait - waited,
                                    max(0, deadline - 5 - time.time()))
                        row.update(rate_limit_retries=row.get('rate_limit_retries', 0) + 1,
                                   rate_limit_wait_s=round(row.get('rate_limit_wait_s', 0) + pause, 3),
                                   queued_s=round(waited + pause, 3), last_rate_limit=redact(exc.text)[:300])
                        time.sleep(pause)
                        continue
                    if exc.code in (402, 429) and out_of_balance(exc.text):
                        # this key is out of credit (for every case of the system): set it aside, try the next one
                        mark_exhausted(ids[index])
                        spent.add(index)
                        if len(keys) > 1:
                            row.setdefault('key_failovers', []).append({'key': names[index], 'why': 'credit',
                                                                        'error': redact(exc.text)[:300]})
                            if pick_key(ids, spent) is not None:
                                continue
                        row['credit_exhausted'] = True
                        raise
                    attempt += 1
                    if not (original and exc.code == 403 and 'channel affinity' in exc.text and attempt < 3):
                        raise
                    renewed = affinity_key(C, original, renew=True)
                    row.setdefault('affinity_rekeys', []).append({'from': body['prompt_cache_key'], 'to': renewed,
                                                                  'error': redact(exc.text)[:500]})
            if original and body['prompt_cache_key'] != original:
                row['prompt_cache_key_sent'] = body['prompt_cache_key']
            with upstream as r, (directory / 'response.sse').open('wb') as log:
                row['http_status'] = r.status
                self.connection.settimeout(10)
                self.send_response(r.status)
                self.send_header('Content-Type', r.headers.get('Content-Type', 'text/event-stream'))
                self.send_header('Connection', 'close')
                self.end_headers()
                sent = True
                client = True
                whole = [] if 'json' in r.headers.get('Content-Type', '') else None
                for line in r:
                    if whole is not None:
                        whole.append(line)
                    log.write(redact_bytes(line))
                    log.flush()
                    if client:
                        try:
                            self.wfile.write(line)
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError, TimeoutError):
                            row['client_disconnected'] = True
                            client = False
                    if line.startswith(b'data:'):
                        try:
                            e = json.loads(line[5:])
                        except ValueError:
                            continue
                        if e.get('type') in ['response.completed', 'response.failed', 'response.incomplete']:
                            resp = e.get('response', {})
                            row.update(terminal_event=e['type'], raw_usage=resp.get('usage'),
                                       returned_model=resp.get('model'), model_error=resp.get('error'))
                        if cfg['harness'] == 'claude':
                            if e.get('type') == 'message_start':
                                row.update(raw_usage=e['message'].get('usage'), returned_model=e['message'].get('model'))
                            elif e.get('type') == 'message_delta':
                                row['raw_usage'] = {**(row.get('raw_usage') or {}), **(e.get('usage') or {})}
                                if (e.get('delta') or {}).get('stop_reason'):
                                    row['finish_reason'] = e['delta']['stop_reason']
                            elif e.get('type') == 'error':
                                row['model_error'] = e.get('error')
                            elif e.get('type') == 'message_stop':
                                row['terminal_event'] = 'message_stop'
                            continue
                        if cfg['harness'] in ('dsh', 'kimi', 'zcode'):
                            if e.get('usage'):
                                row['raw_usage'] = e['usage']
                            if e.get('error'):
                                row['model_error'] = e['error']
                            if e.get('model'):
                                row['returned_model'] = e['model']
                            for choice in e.get('choices') or []:
                                if choice.get('finish_reason'):
                                    row['finish_reason'] = choice['finish_reason']
                            continue
                        if e.get('usageMetadata'):
                            row['raw_usage'] = e['usageMetadata']
                        if e.get('modelVersion'):
                            row['returned_model'] = e['modelVersion']
                        for cand in e.get('candidates', []):
                            if cand.get('finishReason'):
                                row['finish_reason'] = cand['finishReason']
                    if time.time() >= deadline:
                        raise TimeoutError('Case deadline reached during response')
                if whole:  # a non-streamed JSON reply (e.g. Anthropic count_tokens)
                    try:
                        j = json.loads(b''.join(whole))
                        row.update(raw_usage=j.get('usage') or row.get('raw_usage'), model_error=j.get('error'),
                                   finish_reason=j.get('stop_reason') or (j.get('choices') or [{}])[0].get('finish_reason')
                                   or ('json_reply' if r.status == 200 and not j.get('error') else None))
                    except ValueError:
                        pass
        except urllib.error.HTTPError as exc:
            row.update(http_status=exc.code, error=redact(getattr(exc, 'text', None) or exc.read().decode(errors='replace'))[:6000],
                       retry_after=exc.headers.get('Retry-After'))
        except Exception as exc:
            row['error'] = redact(str(exc))[:2000]
        row['finished'] = time.time()
        row['api_complete'] = (row.get('http_status') == 200 and not row.get('error') and
                               bool(row.get('finish_reason') or row.get('terminal_event') == 'response.completed'))
        if read(C / 'run.json').get('started') == run_started:
            save(directory / 'record.json', row)
        else:
            print(f'dropping record of {system}/{case} #{index}: its run directory was replaced', flush=True)
        if not sent:
            self.reply(row.get('http_status') or 502, {'error': row.get('error', 'Provider transport failed')})
        self.close_connection = True


if __name__ == '__main__':
    A.runs_dir.mkdir(parents=True, exist_ok=True)
    server = http.server.ThreadingHTTPServer(('127.0.0.1', A.port), Handler)
    # SIGTERM drains: stop listening (a new gateway can take the port at once) and let requests already accepted finish
    # and write their records, instead of cutting every stream (unrecorded, so uncredited; some harnesses even take a
    # cut reply as final). Streams end by the case deadline at the latest; SIGKILL still stops at once.
    server.daemon_threads = False
    signal.signal(signal.SIGTERM, lambda *_: threading.Thread(target=server.shutdown).start())
    save(A.runs_dir / f'gateway_process_{A.port}.json', {'pid': os.getpid(), 'port': A.port, 'started': time.time(),
                                                        'systems': SYSTEMS, 'upstream': A.upstream})
    env_proxy = os.environ.get('https_proxy') or os.environ.get('HTTPS_PROXY')
    print(f'model gateway on 127.0.0.1:{A.port}, runs {A.runs_dir}, proxy {env_proxy or "none (direct)"}', flush=True)
    for name, cfg in SYSTEMS.items():
        if credentials(name) is None:
            print(f'{name}: refused until its credentials load ({KEY_ERRORS[name]})', flush=True)
        else:
            print(f"{name}: {cfg['harness']} {cfg['model']} via {cfg.get('upstream') or A.upstream}, "
                  f"{len(KEYS[name])} key(s)", flush=True)
    if A.only:      # the other gateway rewrites key_state.json with its systems only: put ours back
        def keep_key_state():
            while True:
                time.sleep(60)
                with KEY_LOCK:
                    save_key_state()
        threading.Thread(target=keep_key_state, daemon=True, name='key_state').start()
    server.serve_forever()
    print(f'SIGTERM: stopped listening on {A.port}; draining requests in flight', flush=True)
    server.server_close()     # closes the listening socket, then joins the request threads
    print('drained', flush=True)

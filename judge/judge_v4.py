#!/usr/bin/env python3
"""Rubric judging, phase 1 (L2/L3): two independent full-modality judges. Each sees the task prompt the agent got
(<tasks>/<task>/prompt.txt), the task's rubric, and the whole MP4 with its audio, and returns
per check item pass/fail/unknown with timestamped evidence and per quality dimension a 1-5 score or unknown with an
observation for each of its 4 observation points. The judges and their endpoints are set in judge/config.json
("judges": name -> one endpoint or a list tried in rotation; default gemini4_argon and gemini38_flash).
The judge is never told which system made the film. A run is judged once its terminal.json says it ended with a video
and its task's rubric file parses; a finished (judge, run, rubric) is never judged again, a re-run case (new
terminal.json start) or a changed rubric is.

Runs that ended without a video go to a trajectory pre-check (text only, "precheck" endpoint): ended normally -> every
judge gets an all-zero result (scoring no_video_zero); ended abnormally (infrastructure) -> precheck.json says so and
the run gets no result until the case is run again.

Layout (judge_results/):
  <judge>/<system>/<task>/result.json            final validated result + hashes
  <judge>/<system>/<task>/attempt_N/             request text + media reference, raw SSE (key redacted), parsed reply
  <judge>/<system>/<task>/failed.json            last failure (retried by the watch loop)
  _media/<system>/<task>/media.json              probe, sha256 of the original and of what was sent (original or proxy)
  _precheck/<system>/<task>/precheck.json        no-video trajectory check
  _state/daemon_<host>.json                      live counters

  python3 -B judge_v4.py run [--runs runs] [--out judge_results] [--judges gemini4_argon,gemini38_flash] [--watch]
  python3 -B judge_v4.py one <judge> <system> <task>        (single job, foreground)
"""
import argparse
import base64
import collections
import concurrent.futures as cf
import hashlib
import json
import os
import random
import re
import socket
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.connection import HTTPConnection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                     # repository root: keys.env, runs/, data/ and the result folders live here


# VVBENCH_CONFIG_DIR (set by `vvbench.py --config-dir`): a directory outside the repository holding judge.json and keys.env
CONFIG_DIR = Path(os.environ['VVBENCH_CONFIG_DIR']).expanduser() if os.environ.get('VVBENCH_CONFIG_DIR') else None
CONFIG_PATH = CONFIG_DIR / 'judge.json' if CONFIG_DIR else HERE / 'config.json'


def _config():
    return json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}


CFG = _config()
PATHS = {'runs': 'runs', 'tasks': 'data/tasks', 'rubrics': 'data/rubrics', 'tags': 'data/tags', 'quiz': 'data/quiz',
         'judge_results': 'judge_results', 'l1_results': 'l1_results', 'quiz_results': 'quiz_results',
         **CFG.get('paths', {})}


def path(name):
    """A configured path; relative paths are relative to the repository root."""
    p = Path(PATHS[name]).expanduser()
    return p if p.is_absolute() else ROOT / p


FFMPEG, FFPROBE = CFG.get('ffmpeg', 'ffmpeg'), CFG.get('ffprobe', 'ffprobe')
RUNS = path('runs')                    # <system>/<task>/{terminal.json, collected/, native_api/, workspace/}
TASKS = path('tasks')                  # <task>/{task.json, prompt.txt, inputs/}
RUBRICS = path('rubrics')              # <task>.json
OUT = path('judge_results')
JUDGES = ['gemini4_argon', 'gemini38_flash']
INLINE_LIMIT = 34_000_000            # raw bytes of the inline MP4; base64 adds 4/3 and the body must stay < 50 MB
MAX_BODY = 50_000_000
TRANSIENT = {408, 409, 429, 500, 502, 503, 504, 520, 522, 524}
DIMS = ['VQ', 'VA', 'CL', 'TM', 'AQ']
VERDICTS = {'pass', 'fail', 'unknown', 'not_applicable'}
ITEM_KEEP = ['id', 'title', 'goal_id', 'criticality', 'evaluator', 'evidence_type', 'pass_condition', 'fail_condition',
             'unknown_condition', 'na_condition', 'applicability', 'observation_scope', 'object_relation',
             'allowed_variations', 'boundary_notes', 'evidence', 'process_evidence', 'required_when_applicable',
             'required_card_meanings', 'required_display_mapping']
HOST = socket.gethostname().split('.')[0]
_KEYS = None


def key(var):
    """An API key by variable name: from the repository's keys.env (KEY=value lines), else the environment."""
    global _KEYS
    if _KEYS is None:
        _KEYS = {}
        f = Path(CFG.get('keys_file', 'keys.env')).expanduser()
        f = f if f.is_absolute() else (CONFIG_DIR or ROOT) / f
        if f.exists():
            for line in f.read_text().splitlines():
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    _KEYS[k.strip()] = v.strip().strip('"\'')
    v = _KEYS.get(var) or os.environ.get(var)
    if not v:
        raise RuntimeError(f'API key {var} is not set (keys.env or the environment)')
    return v


def endpoints(name):
    """The endpoint(s) of a configured role (a judge name, precheck, quiz, gpt, qwen): [{base_url, model, key_var}]."""
    eps = CFG.get('judges', {}).get(name) or CFG.get(name)
    if not eps:
        raise RuntimeError(f'no endpoint for {name!r} in {CONFIG_PATH} (copy judge/config.example.json and edit it)')
    return eps if isinstance(eps, list) else [eps]


def configure(a):
    """Point the module at the folders given on the command line."""
    global RUNS, OUT
    if getattr(a, 'runs', None):
        RUNS = Path(a.runs).resolve()
    if getattr(a, 'out', None):
        OUT = Path(a.out).resolve()


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.tmp{os.getpid()}.{threading.get_ident()}')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    os.replace(tmp, path)


def load(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 22), b''):
            h.update(b)
    return h.hexdigest()


# ---------------------------------------------------------------- stats shared with the progress page
STATS = collections.defaultdict(collections.Counter)
ERRORS = collections.deque(maxlen=60)
SLOCK = threading.Lock()


def stat(judge, k, n=1):
    with SLOCK:
        STATS[judge][k] += n


def err(judge, where, msg):
    with SLOCK:
        ERRORS.append({'at': now_iso(), 'judge': judge, 'where': where, 'error': str(msg)[:400]})


# ---------------------------------------------------------------- transport (Gemini native API over SSE)
class KeepAlive(HTTPAdapter):
    def init_poolmanager(self, *a, **kw):
        opts = list(HTTPConnection.default_socket_options) + [(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)]
        for name, value in [('TCP_KEEPIDLE', 60), ('TCP_KEEPINTVL', 20), ('TCP_KEEPCNT', 6)]:
            if hasattr(socket, name):
                opts.append((socket.IPPROTO_TCP, getattr(socket, name), value))
        kw['socket_options'] = opts
        return super().init_poolmanager(*a, **kw)


class Transient(RuntimeError):
    pass


def part_record(p):
    if 'text' in p:
        return {'text': p['text']}
    d = p['inlineData']
    return {'inlineData': {'mimeType': d['mimeType'], 'media_ref': p.get('_ref')}, 'videoMetadata': p.get('videoMetadata')}


def gemini_url(ep):
    return ep['base_url'].rstrip('/') + f"/models/{ep['model']}:streamGenerateContent?alt=sse"


def gemini_call(parts, out, ep, judge, thinking='HIGH', max_tokens=65536):
    """One attempt; the request text is saved verbatim, the media by reference (path + sha256), the SSE raw."""
    out.mkdir(parents=True, exist_ok=True)
    k, model = key(ep['key_var']), ep['model']
    body = {'contents': [{'role': 'user', 'parts': [{x: v for x, v in p.items() if x != '_ref'} for p in parts]}],
            'generationConfig': {'thinkingConfig': {'thinkingLevel': thinking, 'includeThoughts': True},
                                 'maxOutputTokens': max_tokens, 'responseMimeType': 'application/json',
                                 'mediaResolution': 'MEDIA_RESOLUTION_HIGH'}}
    raw = json.dumps(body, ensure_ascii=False).encode()
    if len(raw) > MAX_BODY:
        raise ValueError(f'request body {len(raw)} bytes > {MAX_BODY}')
    save(out / 'request.json', {'url': gemini_url(ep), 'model': model, 'body_sha256': hashlib.sha256(raw).hexdigest(),
                                'payload_bytes': len(raw), 'generationConfig': body['generationConfig'],
                                'parts': [part_record(p) for p in parts]})
    del body
    rec = {'model': model, 'prepared': time.time(), 'host': HOST}
    events = []
    stat(judge, 'calls')
    try:
        rec['started'] = time.time()
        with requests.Session() as s:
            s.mount('http://', KeepAlive())
            s.mount('https://', KeepAlive())
            with s.post(gemini_url(ep), data=raw, stream=True, timeout=(60, 900),
                        headers={'Content-Type': 'application/json', 'x-goog-api-key': k}) as r:
                del raw
                rec['http_status'] = r.status_code
                if r.status_code != 200:
                    msg = f'HTTP {r.status_code}: {r.text[:1500]}'.replace(k, '[REDACTED]')
                    stat(judge, f'http_{r.status_code}')
                    raise (Transient if r.status_code in TRANSIENT else RuntimeError)(msg)
                with (out / 'response.sse').open('wb') as f:
                    for line in r.iter_lines(chunk_size=8192):
                        f.write(line.replace(k.encode(), b'[REDACTED]') + b'\n')
                        if line.startswith(b'data:'):
                            data = line[5:].strip()
                            if data and data != b'[DONE]':
                                events.append(json.loads(data))
        content, finish = [], None
        for e in events:
            if e.get('error'):
                raise Transient('upstream SSE error: ' + str(e['error'])[:1500])
            if e.get('usageMetadata'):
                rec['usage'] = e['usageMetadata']
            if e.get('modelVersion'):
                rec['returned_model'] = e['modelVersion']
            for c in e.get('candidates', []):
                content.extend(c.get('content', {}).get('parts', []))
                finish = c.get('finishReason', finish)
        rec['finish_reason'] = finish
        save(out / 'model_content.json', [p for p in content if not p.get('thought')])
        text = ''.join(p.get('text', '') for p in content if not p.get('thought')).strip()
        (out / 'final_text.txt').write_text(text)
        if finish != 'STOP':
            if finish is None and not text:
                raise Transient('stream ended without a finishReason or text')
            raise ValueError(f'incomplete output: finishReason={finish}')
        return json.loads(re.sub(r'^```(?:json)?\s*|\s*```$', '', text))
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
        rec['error'] = repr(exc)[:2000]
        stat(judge, 'net_errors')
        raise Transient(rec['error']) from None
    except Exception as exc:
        rec['error'] = str(exc).replace(k, '[REDACTED]')[:3000]
        raise
    finally:
        rec['finished'] = time.time()
        save(out / 'request_record.json', rec)


def rel(d):
    try:
        return str(d.relative_to(OUT))
    except ValueError:
        return str(d)


def call_retry(parts, out, eps, judge, validate, attempts=10, **kw):
    """Transient API errors back off and retry (rotating endpoints when there are several); invalid replies retry up
    to 3 times. Each attempt gets its own directory, numbered after any earlier ones. Returns (reply, dir, model)."""
    bad, last = 0, None
    done = max([int(p.name.split('_')[1]) for p in out.glob('attempt_*') if p.name.split('_')[1].isdigit()], default=0)
    for i in range(attempts):
        d = out / f'attempt_{done + i + 1}'
        ep = eps[(done + i) % len(eps)]
        try:
            reply = gemini_call(parts, d, ep, judge, **kw)
            validate(reply)
            return reply, d, ep['model']
        except Transient as exc:
            last = exc
            stat(judge, 'transient')
            err(judge, rel(d), exc)
            if i == attempts - 1:
                raise
            time.sleep(min(300, 15 * 2 ** min(i, 5)) * random.uniform(0.7, 1.3))
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            last = exc
            (d / 'invalid_reply.txt').write_text(repr(exc)[:3000])
            stat(judge, 'invalid')
            err(judge, rel(d), 'invalid: ' + repr(exc)[:300])
            bad += 1
            if bad >= 3 or i == attempts - 1:
                raise
    raise RuntimeError(f'gave up: {last!r}')


# ---------------------------------------------------------------- media
def run_cmd(args, timeout=3600):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def probe(video):
    p = run_cmd([FFPROBE, '-v', 'error', '-show_streams', '-show_format', '-of', 'json', str(video)], 600)
    if p.returncode:
        raise RuntimeError('ffprobe failed: ' + p.stderr[-800:])
    info = json.loads(p.stdout)
    vs = [s for s in info['streams'] if s['codec_type'] == 'video']
    if not vs:
        raise RuntimeError('no video stream')
    v = vs[0]
    a = [s for s in info['streams'] if s['codec_type'] == 'audio']
    num, den = (int(x) for x in v.get('avg_frame_rate', '0/1').split('/'))
    dur = float(info['format'].get('duration') or v.get('duration') or 0)
    return {'duration_s': dur, 'width': v['width'], 'height': v['height'], 'fps': round(num / den, 3) if den else None,
            'vcodec': v.get('codec_name'), 'pix_fmt': v.get('pix_fmt'), 'audio_streams': len(a),
            'acodec': a[0].get('codec_name') if a else None, 'bytes': int(info['format']['size'])}


class MediaBusy(RuntimeError):
    pass


class Unplayable(RuntimeError):
    """The delivered file cannot be decoded (usually the deadline cut the render: ftyp but no moov)."""


def claim(path, ttl=7200):
    """Atomic claim on a piece of work (also across hosts sharing the result folder): True if this process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, json.dumps({'host': HOST, 'pid': os.getpid(), 'at': time.time()}).encode())
        os.close(fd)
        return True
    except FileExistsError:
        try:
            if time.time() - path.stat().st_mtime > ttl:
                path.unlink(missing_ok=True)
        except OSError:
            pass
        return False


ENC = threading.BoundedSemaphore(int(os.environ.get('JUDGE_ENC', '8')))
MLOCKS = collections.defaultdict(threading.Lock)
MLOCK = threading.Lock()


def media_for(system, task, video):
    """What the full-modality judges get: the original MP4 when it is H.264/yuv420p, AAC/MP3 (or silent) and fits one
    inline request; otherwise a full-duration proxy keeping every second and the original audio track (longest edge
    1280 px CRF 26, then 960/30, 720/34, 640/38 at 15 fps until it fits)."""
    d = OUT / '_media' / system / task
    with MLOCK:
        lock = MLOCKS[(system, task)]
    with lock:
        st = os.stat(video)
        held = False
        m = load(d / 'media.json')
        if m and m['video'] == str(video) and m['video_bytes'] == st.st_size and m['video_mtime'] == st.st_mtime \
                and Path(m['sent']).exists() and 'mono' not in str(m.get('proxy')):
            return m
        if not claim(d / 'media.lock'):
            raise MediaBusy(f'another host is preparing media for {system}/{task}')
        held = True
        try:
            return _encode(d, video, st, system, task)
        finally:
            if held:
                (d / 'media.lock').unlink(missing_ok=True)


def _encode(d, video, st, system, task):
        try:
            pr = probe(video)
        except Exception as exc:
            save(d / 'unplayable.json', {'video': str(video), 'bytes': st.st_size, 'at': now_iso(),
                                         'error': str(exc)[:1500]})
            raise Unplayable(str(exc)[:500]) from None
        (d / 'unplayable.json').unlink(missing_ok=True)   # it decodes now; a record of an earlier file must not stay
        m = {'video': str(video), 'video_bytes': st.st_size, 'video_mtime': st.st_mtime, 'video_sha256': sha256_file(video),
             'probe': pr, 'at': now_iso()}
        if (pr['bytes'] <= INLINE_LIMIT and pr['vcodec'] == 'h264' and pr['pix_fmt'] in ('yuv420p', 'yuvj420p')
                and pr['acodec'] in (None, 'aac', 'mp3')):
            m.update(source='original', sent=str(video), sent_sha256=m['video_sha256'], sent_bytes=pr['bytes'])
        else:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / 'judge_proxy.mp4'
            tmp = d / f'judge_proxy.{HOST}.{os.getpid()}.mp4'   # readers must never see a half-written proxy
            with ENC:
                for edge, crf, fps in [(1280, 26, None), (960, 30, None), (720, 34, None), (640, 38, 15)]:
                    scale = f"scale='if(gt(iw,ih),min({edge},iw),-2)':'if(gt(iw,ih),-2,min({edge},ih))'"
                    vf = scale + (f',fps={fps}' if fps else '')
                    p = run_cmd([FFMPEG, '-v', 'error', '-y', '-threads', '4', '-i', str(video), '-map', '0:v:0',
                                 '-map', '0:a:0?', '-vf', vf, '-c:v', 'libx264', '-preset', 'medium', '-crf', str(crf),
                                 '-pix_fmt', 'yuv420p', *AUDIO_ARGS(pr), '-movflags', '+faststart', str(tmp)], 7200)
                    if p.returncode:
                        tmp.unlink(missing_ok=True)
                        raise RuntimeError('proxy encode failed: ' + p.stderr[-800:])
                    if tmp.stat().st_size <= INLINE_LIMIT:
                        break
                else:
                    size = tmp.stat().st_size
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError(f'proxy still {size} bytes at 640 px / 15 fps')
            os.replace(tmp, dst)
            m.update(source='proxy', sent=str(dst), sent_sha256=sha256_file(dst), sent_bytes=dst.stat().st_size,
                     proxy={'edge': edge, 'crf': crf, 'fps': fps, 'audio': ' '.join(AUDIO_ARGS(pr))}, sent_probe=probe(dst))
        save(d / 'media.json', m)
        return m


def AUDIO_ARGS(pr):
    """Proxies keep the original audio track: copied when it is AAC/MP3, else AAC 192k with its channels."""
    return ['-c:a', 'copy'] if pr['acodec'] in ('aac', 'mp3') else ['-c:a', 'aac', '-b:a', '192k']


def judge_fps(dur):
    """Frames the judge samples per second: 4 for normal lengths, fewer for long films so one request stays well inside
    the context window (about 600 sampled frames at most)."""
    return 4 if dur <= 150 else 2 if dur <= 300 else 1 if dur <= 600 else 0.5


# ---------------------------------------------------------------- rubric and prompt
def rubric_file(task):
    return RUBRICS / f'{task}.json'


def ready_rubrics():
    """task -> sha256 of every rubric file that parses and has check items and the five dimensions."""
    out = {}
    for p in RUBRICS.glob('*.json'):
        try:
            raw = p.read_bytes()
            d = json.loads(raw)
            if not d.get('task_rubric') or [q.get('id') for q in d.get('quality_rubric', [])] != DIMS:
                continue
            out[p.stem] = hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError):
            continue
    return out


def judge_view(d):
    """The rubric as the judge sees it: the task context (task_profile, core_goals, evidence_policy,
    allowed_variations), the check items' ITEM_KEEP fields (no basis quotes: the full prompt is given instead), the
    quality dimensions without their modality tags, and the program checks' ids and titles."""
    items = [{k: it[k] for k in ITEM_KEEP if k in it and it[k] not in (None, '', [], {})} for it in d['task_rubric']]
    quality = []
    for q in d['quality_rubric']:
        q = {k: v for k, v in q.items() if k not in ('modality',)}
        q['observation_items'] = [{k: v for k, v in o.items() if k != 'modality'} for o in q.get('observation_items', [])]
        quality.append(q)
    ctx = {k: d[k] for k in ('task_profile', 'core_goals', 'evidence_policy', 'allowed_variations') if d.get(k)}
    programs = [{'id': p.get('id'), 'title': p.get('title')} for p in d.get('program_check_proposals') or []]
    return ctx, items, quality, programs


def prompt_text(task):
    return (TASKS / task / 'prompt.txt').read_text().strip()


def fmt_time(dur):
    return f'{dur:.1f} s'


INTRO = '''You are a strict, careful video reviewer. Below is a finished video that was delivered for the task brief below. It is the complete delivery: {dur} long, {sound}. Check it against the task's rubric item by item and score the five quality dimensions, using only what you actually see and hear in the video, with timestamps.

## Rules
1. Judge only the delivered video. Watch and listen to the whole video before reaching any conclusion. Timestamps are seconds from the start of the video (0.0 to {end:.1f}).
2. Self-declarations on screen or in the narration ("✓ shown", "requirement met", shot labels, captions that claim something happened) are not evidence. Only what the picture and sound actually show counts.
3. This is the entire delivered video, not an excerpt. Each check item gets exactly one verdict:
   - pass: within the item's applicability you actually observed the pass_condition being met;
   - fail: you checked the whole applicable range and confirmed the omission or the violation. Content that never appears in the video is a fail; a video shorter than required is treated the same way;
   - unknown: the evidence is not available in the video, is unclear or inaudible, needs a precision you cannot get from viewing (exact frames, exact held durations), or the item can only be checked from delivery files, the project, source code or metadata. Follow the item's own unknown_condition.
   Never turn "I did not notice it" into fail without checking the whole applicable range, and never turn uncertainty into pass. not_applicable is allowed only when the item's na_condition or applicability explicitly excludes this video; say why.
4. Judge each item only by its own conditions. Do not merge related items, and do not let the overall impression of the video loosen or tighten a verdict.
5. Delivery-file checks (resolution, frame rate, duration, container, editable project, asset files) are done by programs and are listed only for reference under program checks: do not judge them. Where an item's evidence mentions program checks, project files, logs, OCR or ASR, judge only the part that can be seen or heard in the video.
6. Follow the task brief and the rubric only; do not add requirements. If the rubric conflicts with the task brief, follow the brief and record the conflict in prompt_rubric_conflicts.
7. Quality dimensions VQ, VA, CL, TM, AQ: give each an integer score 1-5, or "unknown". 1, 3 and 5 have task-specific anchors; 2 and 4 lie between adjacent anchors and you must say which anchor the video is closer to and why. Write an observation for each of the dimension's 4 observation points (they are not scored separately; they are the basis of the dimension score). If the evidence is insufficient, the score is "unknown": never use 3 or another middle score in place of unknown. Score each dimension independently; there is no total. A missing audio track is an observed fact (it is not missing evidence).{audio_rule}

## Task brief (the instructions the producer received)
{prompt}

## Rubric context
{ctx}

## Check items ({n} items; criticality core = core requirement, detail = presentation detail)
{items}

## Program checks (done by programs; listed for reference only, do not judge)
{programs}

## Quality dimensions
{quality}
'''

AUDIO_RULE = {True: '', False: ' This video has no audio track: there is no narration, music or sound effect. Text on screen is only text on screen; items that require something to be said or heard fail if it is not heard.'}

TAIL = '''## Output
Output exactly one JSON object and nothing else, with this structure:
{{
 "items": [{{"id": "<check item id>", "verdict": "pass | fail | unknown | not_applicable",
            "observation": "what you actually saw/heard that bears on this item (1-3 sentences)",
            "reason": "why this verdict follows from the item's pass/fail/unknown conditions (1-2 sentences)",
            "evidence": [{{"start_s": 12.0, "end_s": 18.5, "channel": "visual | audio", "what": "what is shown or heard in this span"}}]}}],
 "dimensions": [{{"id": "VQ", "score": "1-5 integer or \\"unknown\\"",
                 "closer_to": "only for 2 or 4: which anchor it is closer to and why",
                 "observations": [{{"id": "VQ1", "observation": "...", "time_ranges": [[start_s, end_s]]}}],
                 "strengths": [{{"text": "...", "start_s": 0.0, "end_s": 0.0}}],
                 "defects": [{{"text": "...", "start_s": 0.0, "end_s": 0.0, "severity": "minor | moderate | severe"}}],
                 "reason": "how the anchors and observations give this score"}}],
 "prompt_rubric_conflicts": [{{"item_id": "...", "note": "..."}}],
 "overall_note": "two or three sentences describing what the video shows and how (description only, no score)"
}}
Requirements:
- items: exactly {n} entries, one for each of these ids: {ids}.
- Every pass and every fail needs at least one evidence span with real timestamps from this video (for a fail because something is absent, give the span(s) you checked). channel is visual for picture/on-screen text/subtitles, audio for narration/music/sound.
- dimensions: exactly 5 entries in the order VQ, VA, CL, TM, AQ; observations for each dimension use exactly its 4 observation point ids ({obs_ids}).
- Be concise: observation and reason are short and factual. Write in English.
'''


def build_parts(task, rubric, media):
    ctx, items, quality, programs = judge_view(rubric)
    pr = media['probe']
    dur, audio = pr['duration_s'], pr['audio_streams'] > 0
    obs_ids = '; '.join(f"{q['id']}: " + ', '.join(o['id'] for o in q.get('observation_items', [])) for q in quality)
    ids = [it['id'] for it in items]
    intro = INTRO.format(dur=fmt_time(dur), sound='with an audio track' if audio else 'with NO audio track', end=dur,
                         audio_rule=AUDIO_RULE[audio], prompt=prompt_text(task),
                         ctx=json.dumps(ctx, ensure_ascii=False, indent=1), n=len(items),
                         items=json.dumps(items, ensure_ascii=False, indent=1),
                         programs=json.dumps(programs, ensure_ascii=False), quality=json.dumps(quality, ensure_ascii=False, indent=1))
    fps = judge_fps(dur)
    raw = Path(media['sent']).read_bytes()
    video = {'inlineData': {'mimeType': 'video/mp4', 'data': base64.b64encode(raw).decode()}, 'videoMetadata': {'fps': fps},
             '_ref': {'path': media['sent'], 'sha256': media['sent_sha256'], 'bytes': len(raw), 'source': media['source']}}
    del raw
    parts = [{'text': intro},
             {'text': f"## The delivered video ({fmt_time(dur)}, {'with' if audio else 'NO'} audio track; sampled at {fps} frames per second)"},
             video,
             {'text': TAIL.format(n=len(ids), ids=', '.join(ids), obs_ids=obs_ids)}]
    want_obs = {q['id']: [o['id'] for o in q.get('observation_items', [])] for q in quality}
    return parts, ids, want_obs, fps


def num(x):
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        return float(x)
    if isinstance(x, str):
        m = re.fullmatch(r'\s*(?:(\d+):)?(\d+(?:\.\d+)?)\s*s?\s*', x)
        if m:
            return (int(m.group(1)) * 60 if m.group(1) else 0) + float(m.group(2))
    raise ValueError(f'bad time {x!r}')


def validator(ids, want_obs, dur):
    def check(reply):
        assert isinstance(reply, dict), 'reply is not an object'
        items = reply.get('items')
        assert isinstance(items, list), 'items missing'
        got = [i.get('id') for i in items]
        missing, extra = sorted(set(ids) - set(got)), sorted(set(got) - set(ids))
        assert not missing and not extra and len(got) == len(ids), f'item ids: missing {missing[:8]} extra {extra[:8]} n={len(got)}/{len(ids)}'
        for i in items:
            v = str(i.get('verdict', '')).strip().lower().replace(' ', '_').replace('n/a', 'not_applicable')
            assert v in VERDICTS, f'bad verdict {i.get("verdict")!r} for {i.get("id")}'
            i['verdict'] = v
            ev = i.get('evidence') or []
            assert isinstance(ev, list), f'evidence of {i["id"]} is not a list'
            for e in ev:
                e['start_s'], e['end_s'] = num(e.get('start_s', 0)), num(e.get('end_s', e.get('start_s', 0)))
        dims = reply.get('dimensions')
        assert isinstance(dims, list) and [d.get('id') for d in dims] == DIMS, f'dimensions {[d.get("id") for d in dims or []]}'
        for d in dims:
            s = d.get('score')
            if isinstance(s, str) and s.strip().isdigit():
                s = int(s.strip())
            if isinstance(s, float) and s.is_integer():
                s = int(s)
            if isinstance(s, str):
                s = s.strip().lower()
            assert s == 'unknown' or (isinstance(s, int) and 1 <= s <= 5), f'bad score {d.get("score")!r} for {d["id"]}'
            d['score'] = s
            obs = [o.get('id') for o in d.get('observations') or []]
            assert sorted(obs) == sorted(want_obs[d['id']]), f'{d["id"]} observations {obs} != {want_obs[d["id"]]}'
    return check


# ---------------------------------------------------------------- runs
def scan_runs():
    """Every ended run under RUNS (<system>/<task>/terminal.json): system, task, run start (identity), status, video
    path or None."""
    out = []
    for system in systems():
        for C in sorted((RUNS / system).iterdir()):
            if C.name.startswith('_'):
                continue
            t = load(C / 'terminal.json')
            if not t:
                continue
            v = t.get('video')
            vp = C / 'collected' / 'video.mp4'
            video = str(vp) if v and vp.exists() else (str(v) if isinstance(v, str) and v and Path(v).exists() else None)
            out.append({'system': system, 'task': C.name, 'run_started': str(t.get('started')), 'status': t.get('status'),
                        'elapsed_s': t.get('elapsed_s'), 'budget_s': t.get('budget_seconds'),
                        'technical_pass': t.get('technical_pass'), 'video': video, 'run_dir': str(C)})
    return out


def systems():
    """The systems with a folder under RUNS, in name order."""
    try:
        return sorted(p.name for p in RUNS.iterdir() if p.is_dir() and not p.name.startswith(('_', '.')))
    except OSError:
        return []


def result_current(path, run, rubric_sha, media_sha=None):
    r = load(path)
    if not r or r.get('run_started') != run['run_started'] or r.get('rubric_sha256') != rubric_sha:
        return False
    return not (media_sha and r.get('sent_sha256') and r['sent_sha256'] != media_sha)


_MSHA = {}


def unplayable(system, task, video=None):
    """The "cannot be decoded" record of this run's collected file. With video given, a record that is not about that
    file (other path or size, or older than the file) is ignored: it was left by an earlier attempt of the case."""
    p = OUT / '_media' / system / task / 'unplayable.json'
    un = load(p)
    if un and video:
        try:
            st = os.stat(video)
            if un.get('video') != str(video) or un.get('bytes') != st.st_size or p.stat().st_mtime < st.st_mtime:
                return None
        except OSError:
            pass
    return un


def media_sha(system, task):
    """sent_sha256 of the prepared media, cached by media.json's mtime (the daemon scans every run every pass)."""
    p = OUT / '_media' / system / task / 'media.json'
    try:
        mt = p.stat().st_mtime
    except OSError:
        return None
    hit = _MSHA.get((system, task))
    if hit and hit[0] == mt:
        return hit[1]
    sha = (load(p) or {}).get('sent_sha256')
    _MSHA[(system, task)] = (mt, sha)
    return sha


def retire(path):
    """Keep a result that no longer matches its run or rubric next to the new one."""
    if path.exists():
        os.replace(path, path.with_name(f'result.superseded.{int(time.time())}.json'))


def judge_job(judge, run, rubric_sha):
    rdir = OUT / judge / run['system'] / run['task']
    res = rdir / 'result.json'
    if result_current(res, run, rubric_sha):
        return 'skip'
    stat(judge, 'started')
    t0 = time.time()
    try:
        raw = rubric_file(run['task']).read_bytes()
        if hashlib.sha256(raw).hexdigest() != rubric_sha:
            return 'rubric-changed'
        rubric = json.loads(raw)
        media = media_for(run['system'], run['task'], run['video'])
        parts, ids, want_obs, fps = build_parts(run['task'], rubric, media)
        dur = media['probe']['duration_s']
        reply, att, model = call_retry(parts, rdir, endpoints(judge), judge, validator(ids, want_obs, dur))
        del parts
        rec = load(att / 'request_record.json', {})
        retire(res)
        save(res, {'judge': judge, 'judge_model': model, 'returned_model': rec.get('returned_model'),
                   'phase': 1, 'scoring': 'judged', 'judged_at': now_iso(), 'attempt': att.name,
                   'system': run['system'], 'task': run['task'], 'run_started': run['run_started'],
                   'run_status': run['status'], 'technical_pass': run['technical_pass'],
                   'video': run['video'], 'video_sha256': media['video_sha256'], 'media_source': media['source'],
                   'sent_sha256': media['sent_sha256'], 'sample_fps': fps, 'duration_s': dur,
                   'audio_streams': media['probe']['audio_streams'],
                   'rubric_file': str(rubric_file(run['task'])), 'rubric_sha256': rubric_sha,
                   'prompt_sha256': hashlib.sha256(prompt_text(run['task']).encode()).hexdigest(),
                   'usage': rec.get('usage'), 'seconds': round(time.time() - t0, 1),
                   'items': reply['items'], 'dimensions': reply['dimensions'],
                   'prompt_rubric_conflicts': reply.get('prompt_rubric_conflicts') or [],
                   'overall_note': reply.get('overall_note'),
                   'note': 'not calibrated against human review; no weighted total'})
        (rdir / 'failed.json').unlink(missing_ok=True)
        stat(judge, 'done')
        return 'done'
    except Exception as exc:
        if isinstance(exc, MediaBusy):
            stat(judge, 'media_busy')
            return 'media-busy'
        if isinstance(exc, Unplayable):
            stat(judge, 'unplayable')
            return 'unplayable'
        stat(judge, 'failed')
        err(judge, f"{run['system']}/{run['task']}", repr(exc)[:400])
        prev = load(rdir / 'failed.json') or {}
        rounds = prev.get('rounds', 0) + 1 if prev.get('run_started') == run['run_started'] else 1
        save(rdir / 'failed.json', {'at': now_iso(), 'run_started': run['run_started'], 'rubric_sha256': rubric_sha,
                                    'rounds': rounds, 'error': repr(exc)[:3000], 'trace': traceback.format_exc()[-3000:]})
        return 'failed'


# ---------------------------------------------------------------- no-video pre-check
PRE_INTRO = '''You are auditing one run of an autonomous coding agent. The agent had a wall-clock budget to produce a video (an MP4 in its workspace) for a task, using its own native harness and tools inside a sandbox. The run ended WITHOUT a delivered video. Decide whether the run ended normally or abnormally.

- normal: the outcome is the agent's own doing. Examples: it worked until the time budget ran out; it decided it was done, gave up, or wrote a blocker/failure report while its API calls were working; it got stuck in its own bugs or loops; its own tool calls failed because of its own mistakes (including requests it made too large itself); the model refused or was stopped by the provider's content filter on an ordinary turn, with no API fault just before.
- abnormal: infrastructure cut the run short or made the harness stop. Any API / interface fault that led to the end counts. Examples: a refusal or content-filter stop on the request right after an API fault, cut stream or read timeout (e.g. when the harness asks the model to resume a cut response); the agent wrote a blocker note or stopped because API faults kept cutting its work; the last API call hung until the deadline with next to no output; the API/gateway returned errors, empty replies, cut-off streams or network_error and the harness took that as the final answer; the upstream switched backends and returned a non-answer marked final; tools that should exist were missing; the sandbox or harness crashed or was killed; quota/credit/rate limit failures ended it; the run ended long before the budget for reasons outside the agent.

Facts about the run (the gateway records every model call; "kind" is how each call ended):
{facts}

The last model calls (oldest first):
{calls}

The final visible messages of the conversation (oldest first, truncated):
{tail}

Output exactly one JSON object: {{"verdict": "normal | abnormal", "category": "short label, e.g. budget_exhausted, gave_up, blocker_report, model_loop, upstream_error_taken_as_final, stream_cut, empty_reply, tools_missing, sandbox_crash, quota", "time_lost_s": <seconds of budget left when it ended, from the facts>, "evidence": "2-4 sentences citing the specific calls/messages", "confidence": "high | medium | low"}}'''


def _texts(o, out, depth=0):
    if depth > 12:
        return
    if isinstance(o, dict):
        if o.get('thought') is True or o.get('type') in ('reasoning', 'thinking', 'redacted_thinking', 'summary_text', 'reasoning_text'):
            return
        for k, v in o.items():
            if k in ('text', 'content', 'output') and isinstance(v, str):
                out.append(v)
            elif k not in ('reasoning_content', 'thinking', 'summary', 'instructions', 'tools', 'signature', 'data',
                           'inlineData', 'image_url', 'thoughtSignature', 'encrypted_content'):
                _texts(v, out, depth + 1)
    elif isinstance(o, list):
        for v in o:
            _texts(v, out, depth + 1)


def conv_tail(req, n=14, cut=1500):
    """Last n messages of the conversation in a gateway request.json, harness independent (messages / input / contents)."""
    try:
        body = json.loads(Path(req).read_text())
    except (OSError, ValueError):
        return []
    body = body.get('body', body) if isinstance(body, dict) else body
    msgs = None
    for k in ('messages', 'input', 'contents'):
        if isinstance(body, dict) and isinstance(body.get(k), list):
            msgs = body[k]
            break
    if not msgs:
        return []
    out = []
    for m in msgs[-n:]:
        role = m.get('role') or m.get('type') or '?'
        bits = []
        _texts(m, bits)
        calls = []
        for c in (m.get('tool_calls') or []):
            f = c.get('function', {})
            calls.append(f"{f.get('name')}({str(f.get('arguments'))[:300]})")
        if m.get('type') in ('function_call', 'custom_tool_call'):
            calls.append(f"{m.get('name')}({str(m.get('arguments') or m.get('input'))[:300]})")
        for p in m.get('parts', []) if isinstance(m.get('parts'), list) else []:
            if p.get('functionCall'):
                calls.append(f"{p['functionCall'].get('name')}({json.dumps(p['functionCall'].get('args'))[:300]})")
        if isinstance(m.get('content'), list):
            for p in m['content']:
                if isinstance(p, dict) and p.get('type') == 'tool_use':
                    calls.append(f"{p.get('name')}({json.dumps(p.get('input'))[:300]})")
        txt = ' '.join(b for b in bits if b.strip())
        if len(txt) > cut:
            txt = txt[:cut // 3] + ' […] ' + txt[-(2 * cut) // 3:]
        out.append(f"[{role}] {txt}" + (f"  TOOL CALLS: {'; '.join(calls)}" if calls else ''))
    return out


FIN = re.compile(rb'"(?:finish_reason|stop_reason|finishReason)"\s*:\s*"([A-Za-z_]+)"')
RESP = re.compile(rb'"type"\s*:\s*"response\.(completed|incomplete|failed)"')
OUTTOK = re.compile(rb'"(?:output_tokens|completion_tokens|candidatesTokenCount)"\s*:\s*(\d+)')
FAULTS = ('network_error', 'sensitive', 'incomplete', 'failed', 'SAFETY', 'RECITATION', 'OTHER', 'PROHIBITED_CONTENT',
          'MALFORMED_FUNCTION_CALL', 'refusal')


def kind(r):
    """How one gateway-recorded model call (native_api/<n>/{record.json, response.sse}) ended: ok-<stop reason>, or a
    fault the harness may have taken for the model's final answer (http error, fault finish reason, cut-off stream,
    empty reply)."""
    try:
        d = json.load(open(r + '/record.json'))
        b = open(r + '/response.sse', 'rb').read()
    except (OSError, ValueError):
        return 'unreadable', None
    if d.get('http_status') != 200:
        return f"http-{d.get('http_status')}", None
    fin = [x.decode() for x in FIN.findall(b)]
    resp = [x.decode() for x in RESP.findall(b)]
    tok = max([int(x) for x in OUTTOK.findall(b)] or [-1])
    end = (fin or resp or [None])[-1]
    if end in FAULTS:
        return f'fault-{end}', tok
    if not d.get('api_complete') or end is None:
        return 'fault-cut-off', tok
    if tok <= 0 and len(b) < 2000:
        return 'fault-empty', tok
    return 'ok-' + end, tok


SKIP_TYPES = {'summary_text', 'reasoning_text', 'thinking', 'redacted_thinking'}


def _reply_texts(o, out):
    if isinstance(o, dict):
        if o.get('thought') is True or (isinstance(o.get('type'), str) and o['type'] in SKIP_TYPES):
            return
        for k, v in o.items():
            if k in ('text', 'content') and isinstance(v, str):
                out.append(v)
            elif k not in ('reasoning_content', 'thinking', 'summary', 'instructions', 'input', 'tools', 'arguments',
                           'partial_json', 'args', 'functionCall'):
                _reply_texts(v, out)
    elif isinstance(o, list):
        for v in o:
            _reply_texts(v, out)


def sse_reply(b):
    """The visible reply text of a raw SSE response (no thinking / reasoning / summaries)."""
    out, lines = [], [l[5:].strip() for l in b.split(b'\n') if l.startswith(b'data:')]
    for l in lines or [b]:
        try:
            _reply_texts(json.loads(l), out)
        except ValueError:
            pass
    # the Responses API repeats each finished text in several events: keep the first copy of each
    seen, uniq = set(), []
    for t in out:
        if len(t) > 40 and t in seen:
            continue
        seen.add(t); uniq.append(t)
    return ''.join(uniq)


def precheck_facts(run):
    C = Path(run['run_dir'])
    t = load(C / 'terminal.json', {})
    calls = sorted(C.glob('native_api/*'))
    kinds = [kind(str(r))[0] for r in calls]
    facts = {k: t.get(k) for k in ('status', 'elapsed_s', 'budget_seconds', 'wall_timeout', 'api_failure_credit_s',
                                   'queue_credit_s', 'technical_pass', 'video', 'stall', 'harness', 'segments')}
    facts['harness'] = t.get('harness')
    facts['model_calls'] = len(calls)
    facts['call_kinds_count'] = dict(collections.Counter(kinds))
    usage = t.get('usage') or {}
    facts['usage'] = {k: usage.get(k) for k in ('requests', 'errors', 'output_tokens_including_reasoning') if k in usage}
    try:
        facts['budget_left_s'] = round(float(t.get('budget_seconds') or 0) - float(t.get('elapsed_s') or 0))
    except (TypeError, ValueError):
        pass
    err_tail = ''
    try:
        err_tail = (C / 'stderr.log').read_text(errors='replace')[-1500:]
    except OSError:
        pass
    facts['harness_stderr_tail'] = err_tail
    un = unplayable(run['system'], run['task'], run['video']) if run['video'] else None
    if un:
        facts['delivered_file_unplayable'] = {'bytes': un.get('bytes'), 'ffprobe_error': str(un.get('error'))[:300],
                                              'note': 'a file was collected but cannot be decoded, usually because the '
                                                      'deadline cut the render before the moov atom was written'}
    lines = []
    for r, k in list(zip(calls, kinds))[-8:]:
        rec = load(r / 'record.json', {})
        try:
            txt = sse_reply((r / 'response.sse').read_bytes()).strip()
        except OSError:
            txt = ''
        lines.append(f"call {r.name}: kind={k} http={rec.get('http_status')} finish={rec.get('finish_reason')} "
                     f"error={str(rec.get('error') or '')[:200]!r} started={rec.get('started')} finished={rec.get('finished')} "
                     f"reply_tail={txt[-500:]!r}")
    tail = conv_tail(calls[-1] / 'request.json') if calls else []
    return facts, lines, tail


def precheck_job(run):
    d = OUT / '_precheck' / run['system'] / run['task']
    pc = load(d / 'precheck.json')
    if pc and pc.get('run_started') == run['run_started']:
        return pc
    stat('precheck', 'started')
    try:
        facts, lines, tail = precheck_facts(run)
        text = PRE_INTRO.format(facts=json.dumps(facts, ensure_ascii=False, indent=1, default=str),
                                calls='\n'.join(lines) or '(none)', tail='\n'.join(tail) or '(none)')

        def check(r):
            assert r.get('verdict') in ('normal', 'abnormal'), f"bad verdict {r.get('verdict')!r}"
        reply, att, model = call_retry([{'text': text}], d, endpoints('precheck'), 'precheck', check, attempts=6, thinking='HIGH',
                                       max_tokens=16384)
        pc = {'system': run['system'], 'task': run['task'], 'run_started': run['run_started'], 'checked_at': now_iso(),
              'model': model, 'attempt': att.name, 'facts': facts, **reply}
        save(d / 'precheck.json', pc)
        stat('precheck', 'done')
        return pc
    except Exception as exc:
        stat('precheck', 'failed')
        err('precheck', f"{run['system']}/{run['task']}", repr(exc)[:400])
        save(d / 'failed.json', {'at': now_iso(), 'error': repr(exc)[:3000], 'trace': traceback.format_exc()[-3000:]})
        return None


def zero_result(judge, run, rubric_sha, pc, why='no video was delivered'):
    """Run ended normally without a usable video: every check item fails and every dimension scores 0."""
    rubric = json.loads(rubric_file(run['task']).read_bytes())
    res = OUT / judge / run['system'] / run['task'] / 'result.json'
    if result_current(res, run, rubric_sha):
        return
    retire(res)
    save(res, {'judge': judge, 'phase': 2 if judge == 'omni' else 1, 'scoring': 'no_video_zero', 'judged_at': now_iso(),
               'system': run['system'], 'task': run['task'], 'run_started': run['run_started'], 'run_status': run['status'],
               'technical_pass': run['technical_pass'], 'video': None, 'rubric_file': str(rubric_file(run['task'])),
               'rubric_sha256': rubric_sha, 'precheck': {k: pc.get(k) for k in ('verdict', 'category', 'evidence', 'confidence', 'model')},
               'why': why,
               'items': [{'id': it['id'], 'verdict': 'fail', 'observation': why,
                          'reason': 'the run ended normally and delivered no usable video (trajectory pre-check)',
                          'evidence': []}
                         for it in rubric['task_rubric']],
               'dimensions': [{'id': q, 'score': 0, 'reason': why} for q in DIMS],
               'note': 'no usable video: all items fail, all dimensions 0'})
    stat(judge, 'zero')


# ---------------------------------------------------------------- daemon
def write_state(judges, pending, t0):
    with SLOCK:
        s = {'host': HOST, 'pid': os.getpid(), 'started': t0, 'updated': time.time(), 'judges': judges,
             'stats': {k: dict(v) for k, v in STATS.items()}, 'pending': dict(pending), 'errors': list(ERRORS)[-40:]}
    save(OUT / '_state' / f'daemon_{HOST}.json', s)


def shard_of(a, run):
    if not a.shard:
        return True
    i, n = (int(x) for x in a.shard.split('/'))
    return int(hashlib.sha256(f"{run['system']}/{run['task']}".encode()).hexdigest(), 16) % n == i


def media_pass(a):
    runs = [r for r in scan_runs() if r['video'] and shard_of(a, r)]
    runs.sort(key=lambda r: -os.path.getsize(r['video']) if os.path.exists(r['video']) else 0)
    t0 = time.time()
    done = busy = fail = 0
    def one(run):
        return media_for(run['system'], run['task'], run['video'])
    with cf.ThreadPoolExecutor(a.workers) as ex:
        futs = {ex.submit(one, r): r for r in runs}
        for f in cf.as_completed(futs):
            r = futs[f]
            try:
                f.result(); done += 1
            except MediaBusy:
                busy += 1
            except Exception as exc:
                fail += 1
                err('media', f"{r['system']}/{r['task']}", repr(exc)[:300])
                print('FAIL', r['system'], r['task'], repr(exc)[:200], flush=True)
            if (done + busy + fail) % 25 == 0:
                print(f'{done + busy + fail}/{len(runs)} ready {done} busy {busy} fail {fail} '
                      f'{(time.time() - t0) / 60:.1f} min', flush=True)
    print('media pass done', done, 'busy', busy, 'fail', fail, f'{(time.time() - t0) / 60:.1f} min')


def daemon(a):
    judges = a.judges.split(',')
    stop = threading.Event()
    live = {'pending': {}}

    def heartbeat():
        while not stop.wait(15):
            write_state(judges, live['pending'], t0)
    threading.Thread(target=heartbeat, daemon=True).start()
    pools = {j: cf.ThreadPoolExecutor(a.workers, thread_name_prefix=j) for j in judges}
    prepool = cf.ThreadPoolExecutor(a.precheck_workers, thread_name_prefix='pre')
    inflight = {}            # (judge, system, task) -> (future, run_started, rubric_sha)
    pre_inflight = {}
    failed_at = {}
    t0 = time.time()
    last_scan = 0
    settled = set()          # (judge, system, task, run_started, rubric_sha, media_sha) already finished
    while True:
        if time.time() - last_scan >= a.scan_every:
            last_scan = time.time()
            rub = ready_rubrics()
            runs = scan_runs()
            order = sorted(runs, key=lambda r: (r['task'] not in rub, r['task'], r['system']))
            for run in order:
                if not shard_of(a, run):
                    continue
                sha = rub.get(run['task'])
                un = unplayable(run['system'], run['task'], run['video']) if run['video'] else None
                if not run['video'] or un:
                    if not a.precheck:
                        continue
                    k = (run['system'], run['task'])
                    cur = pre_inflight.get(k)
                    if not cur or cur[1] != run['run_started']:
                        pre_inflight[k] = (prepool.submit(precheck_job, run), run['run_started'])
                    elif cur[0].done() and sha:
                        pc = cur[0].result()
                        if pc and pc.get('verdict') == 'normal':
                            why = ('the delivered file cannot be decoded (the render was cut off): '
                                   + str(un.get('error'))[:200]) if un else 'no video was delivered'
                            for j in judges:
                                zero_result(j, run, sha, pc, why)
                    continue
                if not sha:
                    continue
                for j in judges:
                    k = (j, run['system'], run['task'])
                    cur = inflight.get(k)
                    if cur and not cur[0].done() and cur[1:] == (run['run_started'], sha):
                        continue
                    msha = media_sha(run['system'], run['task'])
                    ident = (j, run['system'], run['task'], run['run_started'], sha, msha)
                    if ident in settled:
                        continue
                    rdir = OUT / j / run['system'] / run['task']
                    if result_current(rdir / 'result.json', run, sha, msha):
                        settled.add(ident)
                        continue
                    f = load(rdir / 'failed.json')
                    if f and f.get('run_started') == run['run_started']:
                        if f.get('rounds', 1) >= a.max_rounds:
                            continue
                        if time.time() - failed_at.setdefault(k, time.time()) < a.retry_failed:
                            continue
                    failed_at.pop(k, None)
                    inflight[k] = (pools[j].submit(judge_job, j, run, sha), run['run_started'], sha)
        pending = collections.Counter(k[0] for k, v in inflight.items() if not v[0].done())
        pending['precheck'] = sum(not v[0].done() for v in pre_inflight.values())
        live['pending'] = dict(pending)
        write_state(judges, pending, t0)
        if not a.watch and not any(pending.values()):
            break
        time.sleep(20)
    stop.set()
    write_state(judges, {}, t0)


def common_args(p):
    p.add_argument('--runs', help=f'run folder holding <system>/<task>/ (default {PATHS["runs"]}, from the repo root)')
    p.add_argument('--out', help=f'result folder (default {PATHS["judge_results"]})')


def main():
    ap = argparse.ArgumentParser(description='Rubric judging, phase 1: full-modality judges (see the module docstring).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='judge every ended run (daemon with --watch)')
    common_args(r)
    r.add_argument('--judges', default=','.join(JUDGES))
    r.add_argument('--workers', type=int, default=16, help='judge calls in flight per judge')
    r.add_argument('--precheck-workers', type=int, default=8)
    r.add_argument('--no-precheck', dest='precheck', action='store_false')
    r.add_argument('--watch', action='store_true')
    r.add_argument('--scan-every', type=int, default=120)
    r.add_argument('--retry-failed', type=int, default=900)
    r.add_argument('--max-rounds', type=int, default=4)
    r.add_argument('--shard', default='')
    m = sub.add_parser('media', help='prepare the judge media (original or proxy) of every run ahead of judging')
    common_args(m)
    m.add_argument('--workers', type=int, default=8)
    m.add_argument('--shard', default='')
    o = sub.add_parser('one', help='judge one run in the foreground')
    common_args(o)
    o.add_argument('judge')
    o.add_argument('system')
    o.add_argument('task')
    p = sub.add_parser('precheck-one', help='run the no-video pre-check of one run')
    common_args(p)
    p.add_argument('system')
    p.add_argument('task')
    a = ap.parse_args()
    configure(a)
    if a.cmd == 'run':
        daemon(a)
    elif a.cmd == 'media':
        media_pass(a)
    elif a.cmd == 'one':
        run = next(x for x in scan_runs() if x['system'] == a.system and x['task'] == a.task)
        sha = hashlib.sha256(rubric_file(a.task).read_bytes()).hexdigest()
        print(judge_job(a.judge, run, sha), dict(STATS), list(ERRORS)[-3:])
    else:
        run = next(x for x in scan_runs() if x['system'] == a.system and x['task'] == a.task)
        print(json.dumps(precheck_job(run), ensure_ascii=False, indent=1, default=str)[:3000], list(ERRORS)[-3:])


if __name__ == '__main__':
    main()

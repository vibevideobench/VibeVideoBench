#!/usr/bin/env python3
"""L1 = the rubric's program checks ("P items", program_check_proposals) of every task.

The run judged for (system, task) is the one under <runs>/<system>/<task> whose terminal.json exists. Two stages,
both cached per run start:
  measure  facts.json: ffprobe (container, streams, rates, packet count); one full decode of video + audio with
           blackdetect / freezedetect (decode errors, decoded frames); audio peak / RMS; speech by silero VAD (the
           one bundled with faster-whisper); narration language by Whisper language ID (faster-whisper, CPU; model
           "whisper_model" in judge/config.json, default small) over <= 30 s windows of speech; the collector's
           delivery record (project, timeline, notes, T01-T07) and a bounded listing of the workspace.
  judge    result.json: per P item pass / fail / unknown (/ not_applicable) with a reason.
           Edu tasks (ids PA*, ZH*) P01-P08 (decodable MP4, 1920x1080, 30 fps, duration, audible audio, narration
           coverage, narration language, black / freeze) by the item's own rule, in code. Everything else (edu P09
           deliverables, and the other tasks' free-text proposals, most of them task specific) by GPT (judge_v4_p2's
           "gpt" endpoint), given only the facts: pass / fail only when the facts establish it, unknown when they do
           not cover what the item measures (event timing, content, project internals...).
           A run without a video: every item fail ("no video", as the rubric judges score no video 0).
Layout: l1_results/<system>/<task>/{facts.json, result.json, gpt/part_N/attempt_M/}, l1_results/summary.json.
Needs the packages in judge/requirements.txt (faster-whisper, numpy) and ffmpeg / ffprobe.
  l1_judge.py run [--runs runs] [--out l1_results] [--watch SECONDS] [--measure-workers 4] [--judge-workers 8]
                  [--set edu|main]
  l1_judge.py one <system> <task>          measure + judge one run, foreground
  l1_judge.py report                       pass rates per set and system -> l1_results/summary.json
"""
import argparse
import collections
import concurrent.futures as cf
import hashlib
import json
import math
import multiprocessing
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_v4 as jv   # noqa: E402  (configuration: paths, ffmpeg, keys)

OUT = jv.path('l1_results')
WHISPER = jv.CFG.get('whisper_model', 'small')     # a faster-whisper model name or a local model folder
FFMPEG, FFPROBE = jv.FFMPEG, jv.FFPROBE
EDU_PREFIXES = ('PA', 'ZH')
FACTS_VERSION, RULES_VERSION = 1, 1
GPT_ITEMS = 50
SKIP_DIRS = {'node_modules', '.git', '__pycache__', '.cache', '.venv', 'venv', 'site-packages', 'dist-packages', '.npm',
             '.local', '.config', '.mozilla', '.pki'}
EDU_RULES = {'P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08'}


def load(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(v, ensure_ascii=False, indent=1) + '\n')
    os.replace(tmp, p)


def sha(b):
    return hashlib.sha256(b).hexdigest()


def task_set(task):
    return 'edu' if task.startswith(EDU_PREFIXES) else 'main'


def run_identity(C):
    t = load(C / 'terminal.json') or {}
    vp = C / 'collected' / 'video.mp4'
    v = t.get('video')
    video = str(vp) if v and vp.exists() else (v if isinstance(v, str) and v and os.path.exists(v) else None)
    return str(t.get('started')), video, t


# ---------------------------------------------------------------- measure
def sh(cmd, timeout, binary=False):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, **({} if binary else {'text': True, 'errors': 'replace'}))


def rate(r):
    try:
        n, m = r.split('/')
        return float(n) / float(m) if float(m) else None
    except (ValueError, AttributeError):
        return None


def probe(video):
    j = json.loads(sh([FFPROBE, '-v', 'error', '-print_format', 'json', '-show_format', '-show_streams', video], 300).stdout)
    fmt = j.get('format', {})
    vs = [s for s in j.get('streams', []) if s.get('codec_type') == 'video' and not (s.get('disposition') or {}).get('attached_pic')]
    aus = [s for s in j.get('streams', []) if s.get('codec_type') == 'audio']
    v = vs[0] if vs else {}
    pk = sh([FFPROBE, '-v', 'error', '-select_streams', 'v:0', '-count_packets', '-show_entries', 'stream=nb_read_packets',
             '-of', 'csv=p=0', video], 900).stdout.strip() if vs else ''
    return {'format_name': fmt.get('format_name'), 'major_brand': (fmt.get('tags') or {}).get('major_brand'),
            'duration_s': float(fmt['duration']) if fmt.get('duration') else None, 'bytes': int(fmt.get('size') or 0),
            'video': {'codec': v.get('codec_name'), 'width': v.get('width'), 'height': v.get('height'),
                      'avg_fps': rate(v.get('avg_frame_rate')), 'r_fps': rate(v.get('r_frame_rate')),
                      'avg_frame_rate': v.get('avg_frame_rate'), 'pix_fmt': v.get('pix_fmt'),
                      'duration_s': float(v['duration']) if v.get('duration') else None,
                      'packets': int(pk) if pk.isdigit() else None} if vs else None,
            'video_streams': len(vs),
            'audio': [{'codec': a.get('codec_name'), 'sample_rate': int(a.get('sample_rate') or 0), 'channels': a.get('channels'),
                       'duration_s': float(a['duration']) if a.get('duration') else None} for a in aus]}


def decode(video, duration):
    """One full decode of video + audio: [error]/[fatal] log lines, decoded frames, black and freeze intervals."""
    p = sh([FFMPEG, '-nostdin', '-hide_banner', '-loglevel', 'level+info', '-threads', '4', '-i', video, '-map', '0:v:0',
            '-map', '0:a?', '-vf', 'blackdetect=d=0.5:pix_th=0.06,freezedetect=n=0.001:d=2', '-f', 'null', '-'],
           max(900, 4 * (duration or 0)))
    err, black, freeze, cur = [], [], [], None
    for line in p.stderr.replace('\r', '\n').split('\n'):
        if line.startswith('[error]') or line.startswith('[fatal]'):
            err.append(line[:300])
        m = re.search(r'black_start:([\d.]+) black_end:([\d.]+)', line)
        if m:
            black.append([float(m.group(1)), float(m.group(2))])
        m = re.search(r'freeze_start: ([\d.]+)', line)
        if m:
            cur = float(m.group(1))
        m = re.search(r'freeze_end: ([\d.]+)', line)
        if m and cur is not None:
            freeze.append([cur, float(m.group(1))])
            cur = None
    if cur is not None:
        freeze.append([cur, duration or cur])
    frames = re.findall(r'frame=\s*(\d+)', p.stderr)
    return {'returncode': p.returncode, 'errors': len(err), 'first_errors': err[:8],
            'decoded_frames': int(frames[-1]) if frames else None, 'black': black, 'freeze': freeze}


_MODEL = None


def whisper():
    global _MODEL
    if _MODEL is None:
        from faster_whisper import WhisperModel
        _MODEL = WhisperModel(str(WHISPER), device='cpu', compute_type='int8', cpu_threads=4)
    return _MODEL


def audio_facts(video, duration):
    import numpy as np
    from faster_whisper.vad import VadOptions, get_speech_timestamps
    pcm = sh([FFMPEG, '-nostdin', '-v', 'error', '-i', video, '-map', '0:a:0', '-ac', '1', '-ar', '16000', '-f', 's16le', '-'],
             max(600, 2 * (duration or 0)), binary=True).stdout
    x = np.frombuffer(pcm, np.int16).astype(np.float32) / 32768.0
    if not len(x):
        return {'samples': 0}
    peak = float(np.abs(x).max())
    rms = float(np.sqrt(np.mean(x ** 2)))
    segs = get_speech_timestamps(x, VadOptions(threshold=0.5, min_silence_duration_ms=500, speech_pad_ms=100))
    sp = [[s['start'] / 16000, s['end'] / 16000] for s in segs]
    D = duration or len(x) / 16000
    speech_s = sum(e - s for s, e in sp)
    edges = [0.0] + [v for s in sp for v in s] + [D]
    silences = [[edges[i], edges[i + 1]] for i in range(0, len(edges), 2) if edges[i + 1] > edges[i]]
    inner = [[max(a, 5.0), min(b, D - 5.0)] for a, b in silences if min(b, D - 5.0) > max(a, 5.0)]
    # language: Whisper language ID on consecutive <= 30 s windows of speech only
    windows, buf, n, t0 = [], [], 0, None
    for s, e in sp:
        a, b = int(s * 16000), int(e * 16000)
        while a < b:
            take = min(b - a, 30 * 16000 - n)
            buf.append(x[a:a + take]); n += take
            t0 = s if t0 is None else t0
            a += take
            if n >= 30 * 16000:
                windows.append((t0, a / 16000, np.concatenate(buf))); buf, n, t0 = [], 0, None
    if n >= 16000:
        windows.append((t0, sp[-1][1], np.concatenate(buf)))
    langs, share = [], collections.Counter()
    for a, b, w in windows[:60]:
        lang, prob, _ = whisper().detect_language(audio=w, language_detection_segments=1)
        sec = len(w) / 16000
        langs.append({'start': round(a, 2), 'end': round(b, 2), 'speech_s': round(sec, 2), 'lang': lang, 'prob': round(prob, 3)})
        share[lang] += sec
    tot = sum(share.values())
    return {'samples': len(x), 'peak_dbfs': round(20 * math.log10(peak), 2) if peak > 0 else None,
            'rms_dbfs': round(20 * math.log10(rms), 2) if rms > 0 else None,
            'speech': {'segments': len(sp), 'speech_s': round(speech_s, 2), 'ratio': round(speech_s / D, 4) if D else None,
                       'longest_silence_s': round(max((b - a for a, b in silences), default=0), 2),
                       'longest_silence_excluding_first_last_5s': round(max((b - a for a, b in inner), default=0), 2),
                       'silences_over_20s_excluding_edges': [[round(a, 1), round(b, 1)] for a, b in inner if b - a > 20],
                       'first_speech_s': round(sp[0][0], 2) if sp else None, 'last_speech_end_s': round(sp[-1][1], 2) if sp else None,
                       'intervals': [[round(a, 2), round(b, 2)] for a, b in sp]},
            'language': {'windows': langs, 'seconds_by_language': {k: round(v, 1) for k, v in share.most_common()},
                         'share_by_language': {k: round(v / tot, 4) for k, v in share.most_common()} if tot else {},
                         'model': f'faster-whisper {WHISPER} (int8, CPU), language ID per <=30 s window of VAD speech'}}


def workspace_listing(W, cap=400):
    rows, big, total, by_ext = [], [], 0, collections.Counter()
    for root, dirs, files in os.walk(W):
        dirs[:] = sorted(x for x in dirs if x not in SKIP_DIRS)
        rel = os.path.relpath(root, W)
        total += len(files)
        by_ext.update(os.path.splitext(f)[1].lower() or '(none)' for f in files)
        if len(files) > 150:
            big.append({'dir': rel, 'files': len(files), 'examples': sorted(files)[:3]})
            continue
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.lstat(p)
            except OSError:
                continue
            rows.append((rel.count(os.sep) + (rel != '.'), os.path.normpath(os.path.join(rel, f)), st.st_size,
                         os.path.islink(p)))
    rows.sort()
    return {'files_total': total, 'by_extension': dict(by_ext.most_common(25)), 'large_dirs': big[:30],
            'listing_truncated': len(rows) > cap,
            'files': [{'path': p, 'bytes': b, **({'symlink': True} if l else {})} for _, p, b, l in rows[:cap]]}


def measure(job):
    system, task, ts, C = job
    started, video, term = run_identity(C)
    f = {'version': FACTS_VERSION, 'system': system, 'task': task, 'task_set': ts, 'run_dir': str(C), 'run_started': started,
         'status': term.get('status'), 'video': video, 'measured_at': time.time()}
    t0 = time.time()
    try:
        dl = load(C / 'collected' / 'delivery.json') or {}
        f['delivery_record'] = {k: dl.get(k) for k in ['selected_video', 'selection_rule', 'project', 'timeline',
                                                        'declared_project_entry', 'source_and_notes_files', 'mp4_candidates',
                                                        'technical_pass']}
        f['delivery_record']['checks'] = [{k: c.get(k) for k in ['id', 'name', 'verdict', 'evidence']} for c in dl.get('checks') or []]
        f['workspace'] = workspace_listing(C / 'workspace')
        if video:
            f['probe'] = probe(video)
            D = f['probe']['duration_s']
            f['decode'] = decode(video, D)
            f['audio_facts'] = audio_facts(video, D) if f['probe']['audio'] else None
        f['measure_s'] = round(time.time() - t0, 1)
        save(OUT / system / task / 'facts.json', f)
        return system, task, 'measured', None
    except Exception as exc:
        save(OUT / system / task / 'facts_failed.json', {'run_started': started, 'at': time.time(), 'error': repr(exc)[:2000],
                                                        'trace': traceback.format_exc()[-4000:]})
        return system, task, 'failed', repr(exc)[:300]


# ---------------------------------------------------------------- judge
def edu_rule(pid, item, f, task_json):
    """Verdict and reason for edu P01-P08, each by its own pass_condition."""
    pr, dec, au = f['probe'], f['decode'], f.get('audio_facts') or {}
    v, D = pr.get('video') or {}, pr.get('duration_s')
    if pid == 'P01':
        mp4 = 'mp4' in (pr.get('format_name') or '') and (pr.get('major_brand') or '').strip() != 'qt' and f['video'].lower().endswith('.mp4')
        ok = mp4 and dec['errors'] == 0 and dec['returncode'] == 0
        return ('pass' if ok else 'fail'), f"format {pr.get('format_name')} brand {pr.get('major_brand')}; full decode: {dec['errors']} error lines, exit {dec['returncode']}" + (f"; first: {dec['first_errors'][:2]}" if dec['errors'] else '')
    if pid == 'P02':
        return ('pass' if (v.get('width'), v.get('height')) == (1920, 1080) else 'fail'), f"{v.get('width')}x{v.get('height')}"
    if pid == 'P03':
        fps = v.get('avg_fps')
        return ('unknown', 'no average frame rate') if fps is None else (('pass' if abs(fps - 30) <= 0.01 else 'fail'), f"average frame rate {v.get('avg_frame_rate')} = {fps:.4f}")
    if pid == 'P04':
        m = re.search(r'\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]', item.get('pass_condition') or '')
        lo, hi = (float(m.group(1)), float(m.group(2))) if m else (task_json['delivery']['duration_s'][0] - 0.5, task_json['delivery']['duration_s'][1] + 0.5)
        return ('unknown', 'no container duration') if D is None else (('pass' if lo <= D <= hi else 'fail'), f'container duration {D:.2f} s, allowed [{lo}, {hi}]')
    if pid == 'P05':
        if not pr.get('audio'):
            return 'fail', 'no audio stream'
        pk = au.get('peak_dbfs')
        return ('pass' if pk is not None and pk > -50 else 'fail'), f'audio stream present; peak {pk} dBFS (needs > -50)'
    sp = au.get('speech') if au else None
    if pid == 'P06':
        if not pr.get('audio'):
            return 'fail', 'no audio stream'
        ok = sp['ratio'] is not None and sp['ratio'] >= 0.5 and sp['longest_silence_excluding_first_last_5s'] <= 20
        return ('pass' if ok else 'fail'), (f"VAD speech {sp['speech_s']} s of {D:.1f} s = {sp['ratio']:.1%}; longest no-speech stretch "
                                            f"(first/last 5 s excluded) {sp['longest_silence_excluding_first_last_5s']} s (needs >= 50% and <= 20 s)")
    if pid == 'P07':
        want = 'zh' if f['task'].startswith('ZH') else 'en'
        if not pr.get('audio') or not sp or not sp['speech_s']:
            return 'fail', 'no narration detected'
        share = (au['language']['share_by_language'] or {}).get(want, 0.0)
        return ('pass' if share >= 0.9 else 'fail'), f"Whisper language ID over {sp['speech_s']} s of speech: {want} {share:.1%} (needs >= 90%); all: {au['language']['share_by_language']}"
    if pid == 'P08':
        longest_black = max((b - a for a, b in dec['black']), default=0)
        speech = sp['intervals'] if sp else []

        def outside_speech(a, b):
            covered = sum(max(0, min(b, e) - max(a, s)) for s, e in speech)
            return max(0, (b - a) - covered)
        frozen = sum(outside_speech(a, b) for a, b in dec['freeze'])
        share = frozen / D if D else 0
        ok = longest_black <= 10 and share < 0.25
        return ('pass' if ok else 'fail'), (f'longest black stretch {longest_black:.1f} s (needs <= 10); frozen picture without narration '
                                            f'{frozen:.1f} s = {share:.1%} of the film (needs < 25%; freezes while narration plays not counted)')
    raise KeyError(pid)


GPT_INTRO = '''You are the L1 program-check judge of a video-generation benchmark. An AI agent was asked to make a video for the task below. You cannot see or hear the video. You get FACTS measured by programs on the delivered files (ffprobe, a full ffmpeg decode with black/freeze detection, audio levels, voice-activity detection, Whisper language identification, the delivery collector's record and a listing of the agent's workspace) and the task's program-check items.

For every item decide strictly from the FACTS:
- "pass": the facts establish the item's pass condition.
- "fail": the facts establish that the condition is violated.
- "unknown": the facts do not cover what the item measures (e.g. event timing, onsets, on-screen content, wording, the inside of the editable project beyond which files exist), or the item needs a human/independent observation. Do not guess; do not pass an item because nothing contradicts it.
- "not_applicable": only when the item's own applicability condition clearly excludes this delivery.
Apply numeric conditions exactly as written (tolerances included). Give a short reason citing the facts used.

Answer with JSON only: {"items": [{"id": "<item id>", "verdict": "pass|fail|unknown|not_applicable", "reason": "<= 300 chars"}]} with exactly one entry per item, in the given order.'''


def compact_facts(f):
    g = json.loads(json.dumps(f))
    au = g.get('audio_facts') or {}
    if au.get('speech'):
        iv = au['speech'].pop('intervals')
        au['speech']['first_intervals_s'] = iv[:40]
    if au.get('language'):
        au['language']['windows'] = au['language']['windows'][:40]
    for k in ['run_dir', 'measured_at', 'version', 'measure_s']:
        g.pop(k, None)
    return g


def item_view(x):
    keep = ['id', 'title', 'name', 'method', 'pass_condition', 'fail_rule', 'fail_condition', 'unknown_rule', 'unknown_condition',
            'applicability', 'inputs', 'coverage_limit']
    return {k: (x[k][:900] if isinstance(x[k], str) else x[k]) for k in keep if x.get(k) not in (None, '', [])}


def gpt_items(system, task, items, f, task_json, rdir):
    import judge_v4_p2 as p2
    ep = p2.gpt()
    out = []
    parts = [items[i:i + GPT_ITEMS] for i in range(0, len(items), GPT_ITEMS)]
    for n, part in enumerate(parts):
        text = (GPT_INTRO + '\n\n## Task delivery requirements (task.json)\n' +
                json.dumps({'title': task_json.get('title'), 'delivery': task_json.get('delivery'), 'audio': task_json.get('audio')}, ensure_ascii=False) +
                '\n\n## FACTS\n' + json.dumps(compact_facts(f), ensure_ascii=False) +
                '\n\n## Items\n' + json.dumps([item_view(x) for x in part], ensure_ascii=False))
        want = [x['id'] for x in part]

        def once(i, text=text, want=want, n=n):
            body = {'store': False, 'reasoning': {'effort': 'medium'}, 'max_output_tokens': 40000,
                    'input': [{'role': 'user', 'content': [{'type': 'input_text', 'text': text}]}]}
            reply, _ = p2.post_json('responses', body, rdir / 'gpt' / f'part_{n}' / f'attempt_{i + 1}', ep, f'l1 {system}/{task}')
            r = p2.parse_json(reply)
            got = {x['id']: x for x in r['items']}
            assert [i for i in want if i not in got] == [], f'missing items {[i for i in want if i not in got][:5]}'
            res = []
            for i in want:
                vd = got[i].get('verdict')
                assert vd in ('pass', 'fail', 'unknown', 'not_applicable'), vd
                res.append({'id': i, 'verdict': vd, 'reason': str(got[i].get('reason', ''))[:600], 'by': 'gpt'})
            return res
        out += p2.with_retry(once, f'l1 {system}/{task} part {n}')
    return out


def judge(job):
    system, task, ts, C = job
    rdir = OUT / system / task
    started, video, _ = run_identity(C)
    rub_raw = jv.rubric_file(task).read_bytes()
    items = json.loads(rub_raw).get('program_check_proposals') or []
    task_json = load(jv.TASKS / task / 'task.json') or {}
    f = load(rdir / 'facts.json') if video else None
    if video and (not f or f.get('run_started') != started or f.get('version') != FACTS_VERSION):
        return system, task, 'no-facts', None
    fsha = sha(json.dumps(f, sort_keys=True).encode()) if f else None
    old = load(rdir / 'result.json')
    if old and old.get('run_started') == started and old.get('facts_sha256') == fsha and old.get('rubric_sha256') == sha(rub_raw) \
            and old.get('rules_version') == RULES_VERSION:
        return system, task, 'current', None
    try:
        res = []
        if not video:
            res = [{'id': x.get('id'), 'verdict': 'fail', 'reason': 'no video delivered (no video = 0)', 'by': 'no_video'}
                   for x in items]
        else:
            rest = []
            for x in items:
                pid = (x.get('id') or '').split('-')[-1]
                if ts == 'edu' and pid in EDU_RULES:
                    vd, why = edu_rule(pid, x, f, task_json)
                    res.append({'id': x['id'], 'verdict': vd, 'reason': why, 'by': 'rule'})
                else:
                    rest.append(x)
            if rest:
                res += gpt_items(system, task, rest, f, task_json, rdir)
            order = {x.get('id'): i for i, x in enumerate(items)}
            res.sort(key=lambda r: order.get(r['id'], 1e9))
        c = collections.Counter(r['verdict'] for r in res)
        save(rdir / 'result.json', {'system': system, 'task': task, 'task_set': ts, 'run_dir': str(C), 'run_started': started,
                                    'no_video': not video, 'facts_sha256': fsha, 'rubric_sha256': sha(rub_raw),
                                    'rules_version': RULES_VERSION, 'judged_at': time.time(), 'counts': dict(c), 'items': res})
        (rdir / 'failed.json').unlink(missing_ok=True)
        return system, task, 'judged', dict(c)
    except Exception as exc:
        save(rdir / 'failed.json', {'run_started': started, 'at': time.time(), 'error': repr(exc)[:2000], 'trace': traceback.format_exc()[-4000:]})
        return system, task, 'failed', repr(exc)[:300]


# ---------------------------------------------------------------- driver
def ended_runs():
    """(system, task, run dir) of every run under jv.RUNS that has a terminal.json."""
    return [(s, C.name, C) for s in jv.systems() for C in sorted((jv.RUNS / s).iterdir())
            if not C.name.startswith('_') and (C / 'terminal.json').exists()]


def jobs(only_set=None):
    return [(s, t, task_set(t), C) for s, t, C in ended_runs() if only_set is None or task_set(t) == only_set]


def needs_measure(job):
    s, t, ts, C = job
    started, video, _ = run_identity(C)
    if not video:
        return False
    f = load(OUT / s / t / 'facts.json')
    if f and f.get('run_started') == started and f.get('version') == FACTS_VERSION:
        return False
    bad = load(OUT / s / t / 'facts_failed.json')
    return not (bad and bad.get('run_started') == started and time.time() - bad.get('at', 0) < 3600)


def log(*a):
    print(time.strftime('%F %T'), *a, flush=True)


def one_pass(a):
    js = jobs(a.set)
    todo = [j for j in js if needs_measure(j)]
    log(f'{len(js)} ended runs, {len(todo)} to measure')
    stats = collections.Counter()
    # fork: the workers inherit the folders set on the command line
    with cf.ProcessPoolExecutor(a.measure_workers, mp_context=multiprocessing.get_context('fork')) as mp, \
            cf.ThreadPoolExecutor(a.judge_workers) as tp:
        jf = [tp.submit(judge, j) for j in js if not needs_measure(j)]
        mf = {mp.submit(measure, j): j for j in todo}
        for fut in cf.as_completed(mf):
            s, t, st, why = fut.result()
            stats['measure_' + st] += 1
            if st == 'failed':
                log('measure failed', s, t, why)
            else:
                jf.append(tp.submit(judge, mf[fut]))
        for fut in cf.as_completed(jf):
            s, t, st, info = fut.result()
            stats['judge_' + st] += 1
            if st in ('judged', 'failed'):
                log('judge', st, s, t, info)
    log('pass done', dict(stats))
    report(quiet=True)


def report(quiet=False):
    rows = collections.defaultdict(lambda: collections.Counter())
    per_task = collections.defaultdict(list)
    for s, t, ts, C in jobs():
        r = load(OUT / s / t / 'result.json')
        k = (ts, s)
        if not r or r.get('run_started') != run_identity(C)[0]:
            rows[k]['runs_without_result'] += 1
            continue
        rows[k]['runs'] += 1
        rows[k]['no_video'] += r['no_video']
        rows[k].update(r['counts'])
        dec = [x for x in r['items'] if x['verdict'] in ('pass', 'fail')]
        per_task[k].append(sum(x['verdict'] == 'pass' for x in r['items']) / len(r['items']) if r['items'] else None)
    summary = {}
    for k, c in sorted(rows.items()):
        n = sum(c[v] for v in ['pass', 'fail', 'unknown', 'not_applicable'])
        pt = [x for x in per_task[k] if x is not None]
        summary[f'{k[0]}/{k[1]}'] = {**c, 'items': n, 'pass_rate_all_items': round(c['pass'] / n, 4) if n else None,
                                     'pass_rate_decided': round(c['pass'] / (c['pass'] + c['fail']), 4) if c['pass'] + c['fail'] else None,
                                     'unknown_rate': round(c['unknown'] / n, 4) if n else None,
                                     'mean_per_run_pass_share': round(sum(pt) / len(pt), 4) if pt else None}
    save(OUT / 'summary.json', {'updated': time.time(), 'by_set_system': summary})
    if not quiet:
        for k, v in summary.items():
            print(f"{k:32} runs {v.get('runs', 0):3} (no video {v.get('no_video', 0):3}, pending {v.get('runs_without_result', 0):3}) "
                  f"pass {v.get('pass', 0):5} fail {v.get('fail', 0):5} unknown {v.get('unknown', 0):5}  "
                  f"pass/decided {v['pass_rate_decided']}  per-run pass share {v['mean_per_run_pass_share']}")


def main():
    global OUT
    ap = argparse.ArgumentParser(description='L1 program checks (see the module docstring).')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def paths(p):
        p.add_argument('--runs', help=f'run folder holding <system>/<task>/ (default {jv.PATHS["runs"]})')
        p.add_argument('--out', help=f'result folder (default {jv.PATHS["l1_results"]})')
    r = sub.add_parser('run', help='measure and judge every ended run')
    paths(r)
    r.add_argument('--watch', type=int, default=0, help='repeat every N seconds (0: one pass)')
    r.add_argument('--measure-workers', type=int, default=4)
    r.add_argument('--judge-workers', type=int, default=8)
    r.add_argument('--set', choices=['edu', 'main'])
    o = sub.add_parser('one', help='measure and judge one run in the foreground')
    paths(o)
    o.add_argument('system')
    o.add_argument('task')
    paths(sub.add_parser('report', help='pass rates per set and system'))
    a = ap.parse_args()
    if a.runs:
        jv.RUNS = Path(a.runs).resolve()
    if a.out:
        OUT = Path(a.out).resolve()
    if a.cmd == 'report':
        return report()
    if a.cmd == 'one':
        C = jv.RUNS / a.system / a.task
        assert (C / 'terminal.json').exists(), f'{C} has not ended (no terminal.json)'
        j = (a.system, a.task, task_set(a.task), C)
        print(measure(j) if needs_measure(j) else 'facts current')
        print(judge(j))
        print(json.dumps(load(OUT / a.system / a.task / 'result.json'), ensure_ascii=False, indent=1)[:6000])
        return
    while True:
        try:
            one_pass(a)
        except Exception:
            log('pass failed', traceback.format_exc()[-2000:])
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == '__main__':
    main()

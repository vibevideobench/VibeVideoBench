#!/usr/bin/env python3
"""Rubric judging, phase 2: the "omni" judge group = a vision model on sampled frames ("gpt" endpoint in
judge/config.json, OpenAI Responses API) + an omni model on the whole MP4 for sound and sound-with-picture ("qwen"
endpoint, OpenAI-compatible chat completions), split by the task's modality tags and merged into one result per film.

Routing (<tags>/<task>.json: {"tags": {item or observation point id: visual | audio | audio_visual | artifact}}):
  item visual        -> GPT        item audio / audio_visual -> Qwen omni        item artifact -> neither (program check)
  dimensions VQ VA CL TM -> GPT    dimension AQ -> Qwen
  observation points follow their own tag: the audio-tagged points of GPT's dimensions are observed by Qwen first and
  handed to GPT as reference for that dimension's score (the only thing the two judges share).
GPT gets frames (<= 1280 px, the last frame included, no audio) in one Responses API request per slice of items and
is told that sampling cannot prove unsampled instants; Qwen gets a small full-duration proxy that keeps the original
audio.

  python3 -B judge_v4_p2.py run [--runs runs] [--out judge_results] [--watch] [--workers 16] [--shard i/n]
  python3 -B judge_v4_p2.py one <system> <task>
"""
import argparse
import base64
import collections
import concurrent.futures as cf
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_v4 as jv
import requests

TAGS = jv.path('tags')
JUDGE = 'omni'
GPT_DIMS = ['VQ', 'VA', 'CL', 'TM']
QWEN_DIMS = ['AQ']
BODY_LIMIT = 9_300_000           # 9.9 MB of frames goes through, 12.2 MB comes back 413, so one request holds this much
GPT_ITEMS = 55                   # check items per GPT call (most tasks have fewer; the largest has about 100)
QWEN_ITEMS = 35                  # check items per Qwen call (each call re-sends the video; the most audio items in one task is about 60)
QWEN_LIMIT = 12_000_000          # raw bytes of the MP4 sent to Qwen (base64 adds 4/3; ~39 MB bodies are reset)
VERDICTS = jv.VERDICTS


def gpt():
    return jv.endpoints('gpt')[0]


def qwen():
    return jv.endpoints('qwen')[0]


def tags_file(task):
    return TAGS / f'{task}.json'


def tags_for(task):
    d = jv.load(tags_file(task))
    return d.get('tags') if d else None


# ---------------------------------------------------------------- media for each judge
def frames_for(system, task, video, duration):
    """Sampled JPEG frames plus the last frame, cached per (plan, task). Returns [(t_seconds, path)]."""
    edge, q, target = frame_plan(duration)
    fps = max(min(target / max(duration, 0.1), 4.0), 0.02)
    d = jv.OUT / '_frames' / system / task / f'e{edge}q{q}_fps_{fps:.4f}'
    done = d / 'frames.json'
    have = jv.load(done)
    if have and all(Path(p).exists() for _, p in have):
        return [(t, p) for t, p in have]
    if not jv.claim(d / 'frames.lock'):
        raise jv.MediaBusy(f'another worker is extracting frames for {system}/{task}')
    try:
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob('f_*.jpg'):
            f.unlink()
        scale = f"scale='if(gt(iw,ih),min({edge},iw),-2)':'if(gt(iw,ih),-2,min({edge},ih))'"
        with jv.ENC:
            p = jv.run_cmd([jv.FFMPEG, '-v', 'error', '-y', '-threads', '4', '-i', str(video), '-map', '0:v:0',
                            '-vf', f'fps={fps:.5f},{scale}', '-q:v', str(q), str(d / 'f_%05d.jpg')], 3600)
            if p.returncode:
                raise RuntimeError('frame extract failed: ' + p.stderr[-600:])
            q2 = jv.run_cmd([jv.FFMPEG, '-v', 'error', '-y', '-sseof', '-0.4', '-i', str(video), '-map', '0:v:0',
                             '-vf', scale, '-q:v', str(q), '-frames:v', '1', str(d / 'last.jpg')], 600)
        out = [(round(i / fps, 2), str(f)) for i, f in enumerate(sorted(d.glob('f_*.jpg')))]
        if q2.returncode == 0 and (d / 'last.jpg').exists():
            out.append((round(duration, 2), str(d / 'last.jpg')))
        if not out:
            raise RuntimeError('no frames extracted')
        jv.save(done, out)
        return out
    finally:
        (d / 'frames.lock').unlink(missing_ok=True)


def qwen_media(system, task, video, probe):
    """Small full-duration proxy for Qwen: the original audio track, picture small enough that one request fits."""
    d = jv.OUT / '_qwen_media' / system / task
    m = jv.load(d / 'media.json')
    if m and Path(m['sent']).exists() and m.get('video') == str(video) and m.get('video_bytes') == os.path.getsize(video):
        return m
    if not jv.claim(d / 'media.lock'):
        raise jv.MediaBusy(f'another worker is preparing qwen media for {system}/{task}')
    try:
        d.mkdir(parents=True, exist_ok=True)
        dst = d / 'qwen_proxy.mp4'
        tmp = d / f'qwen_proxy.{jv.HOST}.{os.getpid()}.mp4'
        with jv.ENC:
            for edge, crf, fps in [(480, 30, 2), (384, 32, 2), (320, 34, 1)]:
                scale = f"scale='if(gt(iw,ih),min({edge},iw),-2)':'if(gt(iw,ih),-2,min({edge},ih))'"
                p = jv.run_cmd([jv.FFMPEG, '-v', 'error', '-y', '-threads', '4', '-i', str(video), '-map', '0:v:0',
                                '-map', '0:a:0?', '-vf', f'{scale},fps={fps}', '-c:v', 'libx264', '-preset', 'veryfast',
                                '-crf', str(crf), '-pix_fmt', 'yuv420p', '-c:a', 'aac', '-b:a', '96k',
                                '-movflags', '+faststart', str(tmp)], 3600)
                if p.returncode:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError('qwen proxy failed: ' + p.stderr[-600:])
                if tmp.stat().st_size <= QWEN_LIMIT:
                    break
            else:
                size = tmp.stat().st_size
                tmp.unlink(missing_ok=True)
                raise RuntimeError(f'qwen proxy still {size} bytes at 320 px / 1 fps')
        os.replace(tmp, dst)
        m = {'video': str(video), 'video_bytes': os.path.getsize(video), 'sent': str(dst),
             'sent_bytes': dst.stat().st_size, 'sent_sha256': jv.sha256_file(dst), 'at': jv.now_iso(),
             'proxy': {'edge': edge, 'crf': crf, 'fps': fps, 'audio': 'aac 96k, original channels'},
             'audio_streams': probe['audio_streams'], 'duration_s': probe['duration_s']}
        jv.save(d / 'media.json', m)
        return m
    finally:
        (d / 'media.lock').unlink(missing_ok=True)


# ---------------------------------------------------------------- transport
def post_json(api, body, out, ep, label, timeout=(60, 1800), stream=False):
    """One OpenAI-compatible call (api 'responses' or 'chat') to endpoint ep; saves the request (media by reference)
    and the raw response. Returns (text, response id) for 'responses', the text for 'chat'."""
    out.mkdir(parents=True, exist_ok=True)
    k, model = jv.key(ep['key_var']), ep['model']
    url = ep['base_url'].rstrip('/') + ('/responses' if api == 'responses' else '/chat/completions')
    body = dict(body, model=model)
    raw = json.dumps(body, ensure_ascii=False).encode()
    jv.save(out / 'request.json', {'url': url, 'model': model, 'label': label, 'payload_bytes': len(raw),
                                   'body_sha256': hashlib.sha256(raw).hexdigest(),
                                   'body': redact_media(body)})
    rec = {'model': model, 'label': label, 'url': url, 'payload_bytes': len(raw), 'started': time.time(), 'host': jv.HOST}
    jv.stat(JUDGE, 'calls')
    try:
        with requests.Session() as s:
            s.mount('http://', jv.KeepAlive())
            s.mount('https://', jv.KeepAlive())
            with s.post(url, data=raw, stream=stream, timeout=timeout,
                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + k}) as r:
                rec['http_status'] = r.status_code
                if stream:
                    chunks = []
                    with (out / 'response.sse').open('wb') as f:
                        for line in r.iter_lines(chunk_size=8192):
                            f.write(line.replace(k.encode(), b'[REDACTED]') + b'\n')
                            if line.startswith(b'data:') and line[5:].strip() not in (b'', b'[DONE]'):
                                chunks.append(json.loads(line[5:]))
                    if r.status_code != 200:
                        raise fault(r.status_code, json.dumps(chunks)[:1200])
                    text, usage, finish = '', None, None
                    for c in chunks:
                        usage = c.get('usage') or usage
                        for ch in c.get('choices', []):
                            text += (ch.get('delta', {}) or {}).get('content') or ''
                            finish = ch.get('finish_reason') or finish
                    rec.update(usage=usage, finish_reason=finish)
                    (out / 'final_text.txt').write_text(text)
                    if finish not in ('stop', None) or not text.strip():
                        raise jv.Transient(f'bad streamed reply: finish={finish} chars={len(text)}')
                    return text
                body_text = r.text
                (out / 'response.json').write_text(body_text.replace(k, '[REDACTED]'))
                if r.status_code != 200:
                    raise fault(r.status_code, body_text[:1200])
                d = r.json()
                rec.update(usage=d.get('usage'), returned_model=d.get('model'), response_id=d.get('id'),
                           status=d.get('status'), incomplete=d.get('incomplete_details'))
                if api == 'responses':
                    text = ''.join(c.get('text', '') for o in d.get('output', []) if o.get('type') == 'message'
                                   for c in o.get('content', []))
                    if d.get('status') and d['status'] != 'completed':
                        raise jv.Transient(f"response status {d['status']} {d.get('incomplete_details')}")
                else:
                    ch = d['choices'][0]
                    text = ch['message'].get('content') or ''
                    rec['finish_reason'] = ch.get('finish_reason')
                    if ch.get('finish_reason') not in ('stop', None):
                        raise jv.Transient(f"finish_reason {ch.get('finish_reason')}")
                (out / 'final_text.txt').write_text(text)
                rec['response_id'] = d.get('id')
                return (text, d.get('id')) if api == 'responses' else text
    except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError) as exc:
        rec['error'] = repr(exc)[:1500]
        jv.stat(JUDGE, 'net_errors')
        raise jv.Transient(rec['error']) from None
    except Exception as exc:
        rec['error'] = str(exc).replace(k, '[REDACTED]')[:2000]
        raise
    finally:
        rec['finished'] = time.time()
        jv.save(out / 'request_record.json', rec)


def fault(status, text):
    jv.stat(JUDGE, f'http_{status}')
    msg = f'HTTP {status}: {text}'
    return (jv.Transient if status in jv.TRANSIENT or status == 413 else RuntimeError)(msg)


def redact_media(body):
    """The request as saved: base64 media replaced by its size and sha so attempts stay readable."""
    def walk(o):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if isinstance(v, str) and v.startswith('data:') and len(v) > 400:
                    out[k] = f'<{len(v)} chars, sha256 {hashlib.sha256(v.encode()).hexdigest()[:16]}>'
                else:
                    out[k] = walk(v)
            return out
        if isinstance(o, list):
            return [walk(v) for v in o]
        return o
    return walk(body)


def parse_json(text):
    t = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    try:
        return json.loads(t)
    except ValueError:
        m = re.search(r'\{.*\}', t, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def with_retry(fn, label, attempts=8):
    """Retry transient upstream failures and unparsable replies; the caller's fn does one attempt."""
    bad = 0
    for i in range(attempts):
        try:
            return fn(i)
        except jv.Transient as exc:
            jv.stat(JUDGE, 'transient')
            jv.err(JUDGE, label, exc)
            if i == attempts - 1:
                raise
            time.sleep(min(240, 10 * 2 ** min(i, 5)) * random.uniform(0.7, 1.3))
        except (ValueError, KeyError, TypeError, AssertionError) as exc:
            jv.stat(JUDGE, 'invalid')
            jv.err(JUDGE, label, 'invalid: ' + repr(exc)[:300])
            bad += 1
            if bad >= 3 or i == attempts - 1:
                raise
    raise RuntimeError('unreachable')


# ---------------------------------------------------------------- prompts
RULES = '''## Rules
1. Judge only the delivered video, from what you can {sense} in it. Timestamps are seconds from the start (0.0 to {end:.1f}).
2. Self-declarations in the video (on-screen "✓ done", "requirement met", shot labels, narration claiming something was shown) are not evidence; only what is actually shown or heard counts.
3. Each check item gets exactly one verdict:
   - pass: within the item's applicability you actually observed the pass_condition being met;
   - fail: you checked the whole applicable range and confirmed the omission or violation. Content that never appears is a fail, and a video shorter than required is treated the same way;
   - unknown: the evidence is missing, unclear, needs precision your input cannot give, or the item can only be checked from delivery files, the project or metadata. Follow the item's unknown_condition. Never turn "I did not notice it" into fail without checking the applicable range, and never turn uncertainty into pass. not_applicable only when the item's na_condition or applicability excludes this video.
4. Judge each item by its own conditions alone; do not merge items and do not let the overall impression loosen or tighten a verdict.
5. Delivery-file checks (resolution, frame rate, duration, container, editable project, assets) are done by programs and are not yours to judge.
6. Follow the task brief and the rubric only; add no requirements of your own. If the rubric conflicts with the brief, follow the brief and say so in prompt_rubric_conflicts.
{extra}'''

GPT_SENSE = 'see'
GPT_EXTRA = '''7. You receive frames sampled from the video at one fixed interval, in time order, with their timestamps, and no audio. The sampling is evidence, not a handicap: consecutive frames one interval apart do show what is on screen through the whole film, so judge from them normally, and say which timestamps you used. A requirement that lasts longer than a couple of intervals (a shot being held, an element staying on screen, a layout surviving a section, an order of events, something never appearing) is decided by the frames: answer pass or fail, not unknown. Reserve unknown for what is genuinely below the sampling interval or outside the picture: a single-frame event, an exact frame count or exact duration tighter than the interval, a fast motion or transition between two frames, the smoothness of an animation, and anything that must be heard.
8. Subtitles and all on-screen text count as picture, so they are yours to judge.
9. When only part of an item needs sound, judge the part you can see and say so; answer unknown only if the item cannot be decided without hearing it (the reviewer who has the audio judges those items).'''

QWEN_SENSE = 'see and hear'
QWEN_EXTRA = '''7. You receive the complete video with its original audio track. You judge the items that need sound: whether narration exists, its language and content, pronunciation, music, sound effects, mixing, and whether sound and picture line up (sound-and-picture items are yours because you can see the picture too).
8. The picture you receive is a small, low-frame-rate copy: it is good enough to tell what is on screen and when, but not for fine visual detail. If an item needs fine visual detail, answer unknown and say so.
9. If the video has no audio track at all, that is an observed fact, not missing evidence: items that require something to be heard are a fail.'''

ITEM_SCHEMA = '''{"items": [{"id": "<check item id>", "verdict": "pass | fail | unknown | not_applicable",
   "observation": "what you actually {sensed} that bears on this item (1-3 sentences)",
   "reason": "why this verdict follows from the item's conditions (1-2 sentences)",
   "evidence": [{"start_s": 0.0, "end_s": 0.0, "channel": "visual | audio", "what": "what is shown or heard in this span"}]}]'''

DIM_SCHEMA = '''{"dimensions": [{"id": "<dimension id>", "score": "1-5 integer or \\"unknown\\"",
   "closer_to": "only for a score of 2 or 4: which anchor it is closer to and why",
   "observations": [{"id": "<observation point id>", "observation": "what you observed for this point", "time_ranges": [[0.0, 0.0]]}],
   "strengths": [{"text": "...", "start_s": 0.0, "end_s": 0.0}],
   "defects": [{"text": "...", "start_s": 0.0, "end_s": 0.0, "severity": "minor | moderate | severe"}],
   "reason": "how the anchors and your observations give this score"}],
 "overall_note": "two or three sentences describing what the video shows and how (description only, no score)"'''

CONFLICTS = ' "prompt_rubric_conflicts": [{"item_id": "...", "note": "..."}]'

DIM_RULES = '''## Scoring the quality dimensions
Give each dimension an integer 1-5 or "unknown". 1, 3 and 5 have task-specific anchors; 2 and 4 lie between adjacent anchors, and when you give one you must say which anchor the video is closer to and why. Write an observation for each listed observation point: they are not scored separately, they are the basis of the score. If the evidence is insufficient, the score is "unknown" - never a middle score in its place. If the video has no audio track at all, that is not missing evidence: score the sound dimension by its anchors (normally the 1 anchor), not "unknown" and not "not_applicable". Score each dimension independently; there is no total.'''


def brief_block(task, rubric):
    ctx, _, _, _ = jv.judge_view(rubric)
    return (f"## Task brief (the instructions the producer received)\n{jv.prompt_text(task)}\n\n"
            f"## Rubric context\n{json.dumps(ctx, ensure_ascii=False, indent=1)}\n")


def items_for(rubric, tags, kinds):
    return [{k: it[k] for k in jv.ITEM_KEEP if k in it and it[k] not in (None, '', [], {})}
            for it in rubric['task_rubric'] if tags.get(it['id']) in kinds]


def dims_for(rubric, ids, tags, keep):
    """The dimensions this judge scores, each carrying only the observation points tagged for it."""
    out = []
    for q in rubric['quality_rubric']:
        if q['id'] not in ids:
            continue
        q = {k: v for k, v in q.items() if k != 'modality'}
        q['observation_items'] = [{k: v for k, v in o.items() if k != 'modality'}
                                  for o in q.get('observation_items', []) if keep is None or tags.get(o['id']) in keep]
        out.append(q)
    return out


def ask(head, sensed, items, dims, part=None, ref=''):
    """One judging request: the rules and brief, a slice of the check items, and (on the last slice) the dimensions."""
    ids = [i['id'] for i in items]
    dim_ids = [d['id'] for d in dims]
    t = head
    if items:
        t += (f"\n## Check items for you ({len(items)} items"
              + (f", part {part[0]} of {part[1]} of your items" if part and part[1] > 1 else '') + ")\n"
              + json.dumps(items, ensure_ascii=False, indent=1) + '\n')
    if dims:
        t += '\n' + DIM_RULES + ref + f"\n\n## Quality dimensions for you\n{json.dumps(dims, ensure_ascii=False, indent=1)}\n"
    schema = ([ITEM_SCHEMA.replace('{sensed}', sensed)] if items else []) + ([DIM_SCHEMA] if dims else [])
    t += ("\n## Output\nOutput exactly one JSON object and nothing else, with these keys:\n"
          + ',\n'.join(schema) + ',\n' + CONFLICTS + '}\n')
    if items:
        t += (f"items must have exactly {len(ids)} entries, with these ids: {', '.join(ids)}. Every pass and every fail "
              "needs at least one evidence span with real timestamps (for something absent, the span you checked). "
              "channel is audio for anything heard, visual for picture and on-screen text.\n")
    if dims:
        t += (f"dimensions must be exactly these ids in this order: {', '.join(dim_ids)}, each with an observation for "
              "exactly its listed observation points.\n")
    t += "prompt_rubric_conflicts is [] unless the rubric contradicts the brief. Be concise and factual, in English."
    return t, ids, dim_ids, {d['id']: [o['id'] for o in d['observation_items']] for d in dims}


def validator(ids, dim_ids, want_obs):
    def check(reply):
        assert isinstance(reply, dict), 'reply is not an object'
        if ids:
            by = {}
            for i in reply.get('items') or []:                  # unrequested ids and repeats are dropped
                by.setdefault(i.get('id'), i)
            missing = [i for i in ids if i not in by]
            assert not missing, f'item ids: {len(missing)} of {len(ids)} missing, e.g. {missing[:6]}'
            reply['items'] = [by[i] for i in ids]
            for i in reply['items']:
                v = str(i.get('verdict', '')).strip().lower().replace(' ', '_').replace('n/a', 'not_applicable')
                assert v in VERDICTS, f'bad verdict {i.get("verdict")!r} for {i.get("id")}'
                i['verdict'] = v
                for e in i.get('evidence') or []:
                    e['start_s'], e['end_s'] = jv.num(e.get('start_s', 0)), jv.num(e.get('end_s', e.get('start_s', 0)))
        if dim_ids:
            dims = reply.get('dimensions') or []
            assert [d.get('id') for d in dims] == dim_ids, f'dimensions {[d.get("id") for d in dims]} != {dim_ids}'
            for d in dims:
                s = d.get('score')
                if isinstance(s, str) and s.strip().isdigit():
                    s = int(s.strip())
                if isinstance(s, float) and s.is_integer():
                    s = int(s)
                if isinstance(s, str):
                    s = s.strip().lower().replace(' ', '_').replace('n/a', 'not_applicable')
                assert s in ('unknown', 'not_applicable') or (isinstance(s, int) and 1 <= s <= 5), \
                    f'bad score {d.get("score")!r} for {d["id"]}'
                d['score'] = s
                by = {o.get('id'): o for o in d.get('observations') or []}
                missing = [i for i in want_obs[d['id']] if i not in by]
                assert not missing, f'{d["id"]}: no observation for {missing}'
                d['observations'] = [by[i] for i in want_obs[d['id']]]      # extras dropped, order fixed
    return check


def chunks_of(items, n):
    return [items[i:i + n] for i in range(0, len(items), n)] or [[]]


# ---------------------------------------------------------------- Qwen: sound and sound-with-picture
def qwen_refused(rubric, tags, why):
    """What the audio judge would have returned, when it refused to look at the file at all."""
    items = [{'id': it['id'], 'verdict': 'unknown', 'judge_source': 'audio_judge_refused',
              'observation': 'the audio judge refused the file: ' + why,
              'reason': 'no model heard this video, so a sound requirement cannot be decided either way',
              'evidence': []} for it in items_for(rubric, tags, ('audio', 'audio_visual'))]
    dims = [{'id': d['id'], 'score': 'unknown', 'observations': [], 'judge_source': 'audio_judge_refused',
             'closer_to': 'the audio judge refused the file: ' + why}
            for d in dims_for(rubric, QWEN_DIMS, tags, None)]
    return items, dims, {}, [], 'the audio judge refused this file (content filter); sound was not reviewed'


def qwen_pass(system, task, rubric, tags, media, dur, rdir):
    audio_items = items_for(rubric, tags, ('audio', 'audio_visual'))
    head = (f"You are a strict, careful video reviewer. You are sent a finished video delivered for the task brief "
            f"below ({dur:.1f} s, {'with' if media['audio_streams'] else 'NO'} audio track). Check it against the part "
            f"of the rubric given to you.\n\n"
            + RULES.format(sense=QWEN_SENSE, end=dur, extra=QWEN_EXTRA) + '\n\n' + brief_block(task, rubric))
    aq = dims_for(rubric, QWEN_DIMS, tags, None)
    extra = [d for d in dims_for(rubric, GPT_DIMS, tags, ('audio', 'audio_visual')) if d['observation_items']]
    ref = ('' if not extra else
           f"\nFor {', '.join(d['id'] for d in extra)} you write only the observations for the listed sound-related "
           f"observation points and score \"unknown\": the reviewer who sees the full-quality picture scores those "
           f"dimensions, using your observations.\n")
    b64 = 'data:video/mp4;base64,' + base64.b64encode(Path(media['sent']).read_bytes()).decode()
    cs = chunks_of(audio_items, QWEN_ITEMS)
    items, dims, conflicts, note = [], [], [], None
    for n, chunk in enumerate(cs, 1):
        last = n == len(cs)
        text, ids, dim_ids, want = ask(head, 'saw or heard', chunk, aq + extra if last else [],
                                       part=(n, len(cs)), ref=ref)
        check = validator(ids, dim_ids, want)
        label = f'{system}/{task} qwen {n}/{len(cs)}'

        def once(i, text=text, check=check, label=label, n=n):
            body = {'stream': True, 'stream_options': {'include_usage': True}, 'modalities': ['text'],
                    'messages': [{'role': 'user', 'content': [{'type': 'video_url', 'video_url': {'url': b64}},
                                                              {'type': 'text', 'text': text}]}]}
            reply = parse_json(post_json('chat', body, rdir / 'qwen' / f'part_{n}' / f'attempt_{i + 1}', qwen(),
                                         label, stream=True))
            check(reply)
            return reply
        reply = with_retry(once, label)
        items += reply.get('items') or []
        conflicts += reply.get('prompt_rubric_conflicts') or []
        if last:
            dims = reply['dimensions']
            note = reply.get('overall_note')
    aud_obs = {d['id']: d.get('observations') or [] for d in dims if d['id'] not in QWEN_DIMS}
    return items, [d for d in dims if d['id'] in QWEN_DIMS], aud_obs, conflicts, note


# ---------------------------------------------------------------- GPT: picture
def frame_plan(dur):
    """Longest edge, JPEG quality and how many frames to sample: all the frames go in one request (store=False, so
    there is no chaining), and the body ceiling is about 10 MB, so long films trade pixels for coverage."""
    if dur <= 90:
        return 1280, 3, int(max(24, min(90, dur * 3)))
    if dur <= 240:
        return 1024, 4, 170
    if dur <= 600:
        return 896, 5, 230
    return 768, 5, 300


def gap(frames):
    """The sampling interval the judge is told about: what one step in the frame list costs in seconds."""
    d = [b[0] - a[0] for a, b in zip(frames, frames[1:])]
    return statistics.median(d) if d else 0.0


def pick_frames(frames, budget):
    """Keep as many frames as the body budget allows, evenly spread, always including the last one."""
    while len(frames) > 2:
        size = sum(os.path.getsize(p) * 4 // 3 + 80 for _, p in frames)
        if size <= budget:
            break
        keep = max(2, int(len(frames) * budget / size))
        step = len(frames) / keep
        frames = [frames[int(i * step)] for i in range(keep - 1)] + [frames[-1]]
    return frames


def gpt_pass(system, task, rubric, tags, frames, dur, rdir, aud_obs):
    visual_items = items_for(rubric, tags, ('visual',))
    head = (f"You are a strict, careful video reviewer. The frames you are sent come from one finished video "
            f"({dur:.1f} s) delivered for the task brief below. Check it against the part of the rubric given to you."
            f"\n\n" + RULES.format(sense=GPT_SENSE, end=dur, extra=GPT_EXTRA) + '\n\n' + brief_block(task, rubric))
    dims = dims_for(rubric, GPT_DIMS, tags, ('visual',))
    ref = ('' if not aud_obs else
           "\nThe reviewer who heard the audio observed the sound-related observation points of these dimensions; take "
           "their observations as given (you cannot hear the video), use them together with your own for the score, and "
           "do not write those points yourself:\n" + json.dumps(aud_obs, ensure_ascii=False, indent=1) + '\n')
    cs = chunks_of(visual_items, GPT_ITEMS)
    items, out_dims, conflicts, note, used = [], [], [], None, frames[:2]
    for n, chunk in enumerate(cs, 1):
        last = n == len(cs)
        text, ids, dim_ids, want = ask(head, 'saw', chunk, dims if last else [], part=(n, len(cs)), ref=ref)
        check = validator(ids, dim_ids, want)
        label = f'{system}/{task} gpt {n}/{len(cs)}'
        keep = pick_frames(frames, BODY_LIMIT - len(text.encode()) - 4000)
        used = keep if len(keep) > len(used) else used
        content = [{'type': 'input_text', 'text':
                    f"## The delivered video, as {len(keep)} frames sampled in time order, one every "
                    f"{gap(keep):.2f} s\nTimestamps in seconds: "
                    + ', '.join(f'{t:g}' for t, _ in keep) + f'. The last frame is the end of the video ({dur:.1f} s).'}]
        content += [{'type': 'input_image', 'detail': 'high',
                     'image_url': 'data:image/jpeg;base64,' + base64.b64encode(Path(p).read_bytes()).decode()}
                    for _, p in keep]
        content.append({'type': 'input_text', 'text': text})

        def once(i, content=content, check=check, label=label, n=n):
            body = {'store': False, 'reasoning': {'effort': 'high'}, 'max_output_tokens': 60000,
                    'input': [{'role': 'user', 'content': content}]}
            out, _ = post_json('responses', body, rdir / 'gpt' / f'part_{n}' / f'attempt_{i + 1}', gpt(), label)
            reply = parse_json(out)
            check(reply)
            return reply
        reply = with_retry(once, label)
        items += reply.get('items') or []
        conflicts += reply.get('prompt_rubric_conflicts') or []
        if last:
            out_dims = reply['dimensions']
            note = reply.get('overall_note')
    return items, out_dims, conflicts, note, used


# ---------------------------------------------------------------- one film
def judge_one(run, rubric_sha):
    system, task = run['system'], run['task']
    rdir = jv.OUT / JUDGE / system / task
    res = rdir / 'result.json'
    tags = tags_for(task)
    g_model, q_model = gpt()['model'], qwen()['model']
    if not tags:
        return 'no-tags'
    jv.stat(JUDGE, 'started')
    t0 = time.time()
    try:
        raw = jv.rubric_file(task).read_bytes()
        if hashlib.sha256(raw).hexdigest() != rubric_sha:
            return 'rubric-changed'
        rubric = json.loads(raw)
        media = jv.media_for(system, task, run['video'])       # probe + the phase-1 proxy (also validates the file)
        dur = media['probe']['duration_s']
        qm = qwen_media(system, task, media['sent'], media['probe'])
        frames = frames_for(system, task, media['sent'], dur)
        try:
            q_items, aq_dims, aud_obs, q_conf, q_note = qwen_pass(system, task, rubric, tags, qm, dur, rdir)
            refused = None
        except RuntimeError as exc:
            # the omni provider's content filter may reject the file itself ("data_inspection_failed"); re-encoding
            # does not change its mind, so the sound items stay unjudged instead of failing the whole film
            if 'data_inspection_failed' not in str(exc):
                raise
            refused = str(exc)[:300]
            q_items, aq_dims, aud_obs, q_conf, q_note = qwen_refused(rubric, tags, refused)
        g_items, g_dims, g_conf, g_note, used = gpt_pass(system, task, rubric, tags, frames, dur, rdir, aud_obs)
        by_id = {}
        for it in q_items:
            by_id[it['id']] = dict(it, judge_source=it.get('judge_source') or q_model)
        for it in g_items:
            by_id[it['id']] = dict(it, judge_source=g_model)
        for it in rubric['task_rubric']:
            if it['id'] in by_id:
                continue
            kind = tags.get(it['id'])
            by_id[it['id']] = {'id': it['id'], 'verdict': 'unknown', 'judge_source': 'program_check_not_run',
                              'observation': 'not judged by a viewing model',
                              'reason': ('this item is a delivery-file / project check (modality artifact); the program '
                                         'checks were not run in this round' if kind == 'artifact' else
                                         f'no modality tag routed this item to a judge (tag {kind!r})'), 'evidence': []}
        items = [by_id[it['id']] for it in rubric['task_rubric']]
        # the sound-related observation points of GPT's dimensions were observed by Qwen and fed to GPT as reference
        dims = {d['id']: dict(d, judge_source=g_model,
                              **({'audio_observations': aud_obs[d['id']]} if aud_obs.get(d['id']) else {}))
                for d in g_dims}
        dims.update({d['id']: dict(d, judge_source=d.get('judge_source') or q_model)
                     for d in aq_dims})
        dims = [dims[d] for d in jv.DIMS if d in dims]
        missing = [d for d in jv.DIMS if d not in {x['id'] for x in dims}]
        assert not missing, f'dimensions missing after merge: {missing}'
        counts = collections.Counter(i['judge_source'] for i in items)
        jv.retire(res)
        jv.save(res, {'judge': JUDGE, 'phase': 2, 'scoring': 'judged', 'judged_at': jv.now_iso(),
                      'judge_models': {'visual': g_model, 'audio': q_model}, 'routing': 'modality_tags',
                      'system': system, 'task': task, 'run_started': run['run_started'], 'run_status': run['status'],
                      'technical_pass': run['technical_pass'], 'video': run['video'],
                      'video_sha256': media['video_sha256'], 'duration_s': dur,
                      'audio_streams': media['probe']['audio_streams'],
                      'gpt_input': {'frames': len(used), 'extracted': len(frames), 'detail': 'high',
                                    'plan': dict(zip(('max_edge_px', 'jpeg_q', 'target_frames'), frame_plan(dur))),
                                    'first_s': used[0][0], 'last_s': used[-1][0]},
                      'qwen_input': {'sent_sha256': qm['sent_sha256'], 'bytes': qm['sent_bytes'], 'proxy': qm['proxy']},
                      'rubric_file': str(jv.rubric_file(task)), 'rubric_sha256': rubric_sha,
                      'tags_file': str(tags_file(task)),
                      'prompt_sha256': hashlib.sha256(jv.prompt_text(task).encode()).hexdigest(),
                      'items_by_source': dict(counts), 'seconds': round(time.time() - t0, 1),
                      'items': items, 'dimensions': dims,
                      'prompt_rubric_conflicts': (q_conf or []) + (g_conf or []),
                      'overall_note': {'visual': g_note, 'audio': q_note},
                      **({'audio_judge_refused': refused} if refused else {}),
                      'note': 'not calibrated against human review; no weighted total; artifact items need program checks'
                              + ('; the audio judge refused this file (content filter), so the sound items are unknown'
                                 if refused else '')})
        (rdir / 'failed.json').unlink(missing_ok=True)
        jv.stat(JUDGE, 'done')
        return 'done'
    except Exception as exc:
        if isinstance(exc, jv.MediaBusy):
            jv.stat(JUDGE, 'media_busy')
            return 'media-busy'
        if isinstance(exc, jv.Unplayable):
            jv.stat(JUDGE, 'unplayable')
            return 'unplayable'
        jv.stat(JUDGE, 'failed')
        jv.err(JUDGE, f'{system}/{task}', repr(exc)[:400])
        prev = jv.load(rdir / 'failed.json') or {}
        rounds = prev.get('rounds', 0) + 1 if prev.get('run_started') == run['run_started'] else 1
        jv.save(rdir / 'failed.json', {'at': jv.now_iso(), 'run_started': run['run_started'], 'rubric_sha256': rubric_sha,
                                       'rounds': rounds, 'error': repr(exc)[:3000],
                                       'trace': traceback.format_exc()[-3000:]})
        return 'failed'


# ---------------------------------------------------------------- daemon
def daemon(a):
    pool = cf.ThreadPoolExecutor(a.workers, thread_name_prefix='omni')
    inflight = {}
    failed_at = {}
    settled = set()
    t0 = time.time()
    stop = threading.Event()
    live = {'pending': {}}

    def heartbeat():
        while not stop.wait(15):
            jv.write_state([JUDGE], live['pending'], t0)
    threading.Thread(target=heartbeat, daemon=True).start()
    last_scan = 0
    while True:
        if time.time() - last_scan >= a.scan_every:
            last_scan = time.time()
            rub = jv.ready_rubrics()
            runs = jv.scan_runs()
            for run in sorted(runs, key=lambda r: (r['task'], r['system'])):
                if not jv.shard_of(a, run):
                    continue
                sha = rub.get(run['task'])
                if not sha or not tags_for(run['task']):
                    continue
                un = jv.unplayable(run['system'], run['task'], run['video']) if run['video'] else None
                if not run['video'] or un:
                    pc = jv.load(jv.OUT / '_precheck' / run['system'] / run['task'] / 'precheck.json')
                    if pc and pc.get('run_started') == run['run_started'] and pc.get('verdict') == 'normal':
                        why = ('the delivered file cannot be decoded (the render was cut off): '
                               + str(un.get('error'))[:200]) if un else 'no video was delivered'
                        jv.zero_result(JUDGE, run, sha, pc, why)
                    continue
                k = (run['system'], run['task'])
                cur = inflight.get(k)
                if cur and not cur[0].done() and cur[1:] == (run['run_started'], sha):
                    continue
                ident = (k, run['run_started'], sha)
                if ident in settled:
                    continue
                if jv.result_current(jv.OUT / JUDGE / run['system'] / run['task'] / 'result.json', run, sha):
                    settled.add(ident)
                    continue
                f = jv.load(jv.OUT / JUDGE / run['system'] / run['task'] / 'failed.json')
                if f and f.get('run_started') == run['run_started']:
                    if f.get('rounds', 1) >= a.max_rounds:
                        continue
                    if time.time() - failed_at.setdefault(k, time.time()) < a.retry_failed:
                        continue
                failed_at.pop(k, None)
                inflight[k] = (pool.submit(judge_one, run, sha), run['run_started'], sha)
        live['pending'] = {JUDGE: sum(1 for v in inflight.values() if not v[0].done())}
        jv.write_state([JUDGE], live['pending'], t0)
        if not a.watch and not live['pending'][JUDGE]:
            break
        time.sleep(20)
    stop.set()


def main():
    ap = argparse.ArgumentParser(description='Rubric judging, phase 2: the omni judge group (see the module docstring).')
    sub = ap.add_subparsers(dest='cmd', required=True)
    r = sub.add_parser('run', help='judge every ended run (daemon with --watch)')
    jv.common_args(r)
    r.add_argument('--workers', type=int, default=16)
    r.add_argument('--watch', action='store_true')
    r.add_argument('--scan-every', type=int, default=120)
    r.add_argument('--retry-failed', type=int, default=900)
    r.add_argument('--max-rounds', type=int, default=4)
    r.add_argument('--shard', default='')
    o = sub.add_parser('one', help='judge one run in the foreground')
    jv.common_args(o)
    o.add_argument('system')
    o.add_argument('task')
    a = ap.parse_args()
    jv.configure(a)
    if a.cmd == 'run':
        daemon(a)
    else:
        run = next(x for x in jv.scan_runs() if x['system'] == a.system and x['task'] == a.task)
        sha = hashlib.sha256(jv.rubric_file(a.task).read_bytes()).hexdigest()
        print(judge_one(run, sha), dict(jv.STATS), list(jv.ERRORS)[-3:])


if __name__ == '__main__':
    main()

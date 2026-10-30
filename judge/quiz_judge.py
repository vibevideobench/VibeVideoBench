#!/usr/bin/env python3
"""Grounded quiz for the edu tasks (<quiz>/<ID>.json, an auxiliary metric next to L1 / L2 / L3). Protocol from the quiz
files: the grader answers from the finished video only; options A–E, E = "the video does not cover it"; an A–D answer
must come with a timestamp and the video's own words (narration or on-screen text), checked against the video
afterwards; an answer whose evidence does not check out counts as E. Score = correct answers / questions.
  answer   the "quiz" endpoint of judge/config.json (a Gemini model; judge_v4's video path: original MP4 or its proxy,
           inline) gets the questions without answers / basis / validation and returns {choice, time, quote, source}
           per question
  check    narration quotes: fuzzy match against a faster-whisper transcript (CPU) within +-20 s of the given time;
           on-screen quotes: GPT (judge_v4_p2's "gpt" endpoint) looks at frames at t-1 / t / t+1 and says whether the
           text is there; either check passing verifies the answer
  score    correct = verified answer == key; a run without a video scores 0
Layout: quiz_results/<system>/<task>/{transcript.json, result.json, gemini/attempt_N, gpt/...}; quiz_results/summary.json.
The judge media cache is judge_v4's (<judge_results>/_media), shared with the rubric judges.
  quiz_judge.py run [--runs runs] [--out quiz_results] [--judge-results judge_results] [--watch SECONDS] [--workers 4]
  quiz_judge.py one <system> <task>      quiz_judge.py report
"""
import argparse
import base64
import collections
import concurrent.futures as cf
import difflib
import json
import re
import subprocess
import sys
import time
import traceback
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_v4 as jv      # noqa: E402
import judge_v4_p2 as p2   # noqa: E402

QUIZ = jv.path('quiz')
WHISPER = jv.CFG.get('whisper_model', 'small')     # a faster-whisper model name or a local model folder
OUT = jv.path('quiz_results')
JUDGE = 'quiz'
VERSION = 1
NARR_MIN = 0.6       # share of the quote's characters found in order in the transcript window

INTRO = '''You are taking a quiz about an educational explainer video. Watch and listen to the whole video below. Answer every question ONLY from what this video itself says (narration) or shows (on-screen text) -- not from your own knowledge and not from what a good lecture "would" say.

Each question has options A-E; E means the video does not cover it. Choose A, B, C or D only if the video itself states it, and then give:
  "time": the second (number, from the start of the video) where it is said or shown,
  "quote": the exact words of the narration or of the on-screen text at that moment, verbatim, in the video's language (one short sentence or phrase, at most 40 words / 60 characters),
  "source": "narration" or "screen".
If you cannot point to such a moment, answer E (and leave time / quote empty). Your quotes are checked against the video afterwards; an answer whose quote is not there counts as E.

The video is {dur:.1f} seconds long. Return JSON only: {{"answers": [{{"id": "Q01", "choice": "A", "time": 12.5, "quote": "...", "source": "narration"}}, ...]}} with exactly one entry per question, in the given order.

## Questions
{questions}'''


def load(p, default=None):
    return jv.load(p, default)


def questions(task):
    q = json.load(open(QUIZ / f'{task}.json'))['questions']
    view = [{'id': x['id'], 'question': x['question'], 'options': x['options']} for x in q]
    return q, view


def norm(s):
    s = unicodedata.normalize('NFKC', str(s)).lower()
    return ''.join(ch for ch in s if ch.isalnum())


def coverage(quote, text):
    """Share of the quote's characters that appear, in order, in text (difflib matching blocks)."""
    a, b = norm(quote), norm(text)
    if not a or not b:
        return 0.0
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return sum(x.size for x in m.get_matching_blocks()) / len(a)


_W = None


def whisper():
    global _W
    if _W is None:
        from faster_whisper import WhisperModel
        _W = WhisperModel(str(WHISPER), device='cpu', compute_type='int8', cpu_threads=4, num_workers=4)
    return _W


def transcript(system, task, video, started):
    p = OUT / system / task / 'transcript.json'
    t = load(p)
    if t and t.get('run_started') == started and t.get('video') == video:
        return t
    lang = 'zh' if task.startswith('ZH') else 'en'
    segs, info = whisper().transcribe(video, language=lang, vad_filter=True, beam_size=1, condition_on_previous_text=False)
    rows = [{'start': round(s.start, 2), 'end': round(s.end, 2), 'text': s.text.strip()} for s in segs]
    t = {'run_started': started, 'video': video, 'language': lang, 'model': f'faster-whisper {WHISPER} int8 CPU, VAD on',
         'duration_s': round(info.duration, 2), 'segments': rows}
    jv.save(p, t)
    return t


def frames_b64(video, t, dur):
    out = []
    for x in [t - 1, t, t + 1]:
        x = min(max(0.0, x), max(0.0, dur - 0.05))
        jpg = subprocess.run([jv.FFMPEG, '-nostdin', '-v', 'error', '-ss', f'{x:.2f}', '-i', video, '-frames:v', '1',
                              '-vf', "scale='min(1280,iw)':-2", '-q:v', '4', '-f', 'image2', '-c:v', 'mjpeg', '-'],
                             capture_output=True, timeout=120).stdout
        if jpg:
            out.append((round(x, 2), base64.b64encode(jpg).decode()))
    return out


def check_screen(system, task, video, dur, claims, rdir):
    """claims: [(qid, time, quote)] -> {qid: (bool, reason)} by GPT looking at frames t-1 / t / t+1."""
    if not claims:
        return {}
    content = [{'type': 'input_text', 'text':
                'For each claim below you get up to three frames of a video (about 1 s apart) around the claimed time. '
                'Decide whether the quoted text is visibly on screen in at least one of the frames -- verbatim or with only '
                'trivial differences (case, punctuation, line breaks, a formula typeset differently). Do not accept text '
                'that only means something similar. Return JSON only: {"claims": [{"id": "...", "on_screen": true|false, '
                '"reason": "<= 150 chars"}]}, one entry per claim.'}]
    for qid, t, quote in claims:
        content.append({'type': 'input_text', 'text': f'## Claim {qid}: at {t:.1f} s the screen shows: "{quote}"'})
        for x, b in frames_b64(video, t, dur):
            content.append({'type': 'input_text', 'text': f'frame at {x:.2f} s'})
            content.append({'type': 'input_image', 'image_url': 'data:image/jpeg;base64,' + b, 'detail': 'high'})
    want = [c[0] for c in claims]

    def once(i):
        body = {'store': False, 'reasoning': {'effort': 'low'}, 'max_output_tokens': 8000,
                'input': [{'role': 'user', 'content': content}]}
        reply, _ = p2.post_json('responses', body, rdir / 'gpt' / f'attempt_{i + 1}', p2.gpt(), f'quiz screen {system}/{task}')
        r = {x['id']: x for x in p2.parse_json(reply)['claims']}
        assert all(q in r for q in want), 'missing claims'
        return {q: (bool(r[q].get('on_screen')), str(r[q].get('reason', ''))[:300]) for q in want}
    return p2.with_retry(once, f'quiz screen {system}/{task}')


def grade(system, task, ts, C):
    rdir = OUT / system / task
    t = load(C / 'terminal.json') or {}
    started = str(t.get('started'))
    vp = C / 'collected' / 'video.mp4'
    video = str(vp) if t.get('video') and vp.exists() else None
    old = load(rdir / 'result.json')
    if old and old.get('run_started') == started and old.get('version') == VERSION:
        return system, task, 'current', None
    qs, view = questions(task)
    key = {x['id']: x['answer'] for x in qs}
    try:
        unplayable = None
        if video:
            try:
                media = jv.media_for(system, task, video)
            except jv.Unplayable as exc:      # e.g. a render cut off by the deadline (no moov): scored as no video
                unplayable = str(exc)[:300]
        if not video or unplayable:
            res = {'no_video': True, 'unplayable': unplayable,
                   'answers': [{'id': q, 'choice': None, 'key': key[q], 'counted_as': 'E', 'correct': False} for q in key]}
        else:
            dur = media['probe']['duration_s']
            raw = Path(media['sent']).read_bytes()
            parts = [{'text': INTRO.format(dur=dur, questions=json.dumps(view, ensure_ascii=False, indent=1))},
                     {'inlineData': {'mimeType': 'video/mp4', 'data': base64.b64encode(raw).decode()},
                      'videoMetadata': {'fps': jv.judge_fps(dur)},
                      '_ref': {'path': media['sent'], 'sha256': media['sent_sha256'], 'bytes': len(raw), 'source': media['source']}}]
            del raw

            def validate(r):
                got = {a['id']: a for a in r['answers']}
                assert all(q in got for q in key), 'missing answers'
                for q in key:
                    a = got[q]
                    assert a.get('choice') in ('A', 'B', 'C', 'D', 'E'), a.get('choice')
                    if a['choice'] != 'E':
                        float(a.get('time'))
                        assert str(a.get('quote') or '').strip(), 'A-D without a quote'
            reply, att, model = jv.call_retry(parts, rdir / 'gemini', jv.endpoints('quiz'), JUDGE, validate, attempts=6, thinking='HIGH', max_tokens=32768)
            del parts
            got = {a['id']: a for a in reply['answers']}
            tr = transcript(system, task, video, started)
            screen = []
            answers = []
            for q in key:
                a = got[q]
                row = {'id': q, 'choice': a['choice'], 'key': key[q]}
                if a['choice'] != 'E':
                    tt, quote = float(a['time']), str(a['quote'])
                    window = ' '.join(s['text'] for s in tr['segments'] if s['end'] >= tt - 20 and s['start'] <= tt + 20)
                    cov = coverage(quote, window)
                    row.update(time=tt, quote=quote, source=a.get('source'), narration_coverage=round(cov, 3),
                               narration_ok=cov >= NARR_MIN)
                    if not row['narration_ok']:
                        screen.append((q, tt, quote))
                answers.append(row)
            sc = check_screen(system, task, video, dur, screen, rdir)
            for row in answers:
                if row['choice'] == 'E':
                    row.update(verified=False, counted_as='E')
                else:
                    ok_s, why = sc.get(row['id'], (False, None))
                    if row['id'] in sc:
                        row.update(screen_ok=ok_s, screen_reason=why)
                    row['verified'] = bool(row['narration_ok'] or ok_s)
                    row['counted_as'] = row['choice'] if row['verified'] else 'E'
                row['correct'] = row['counted_as'] == row['key']
            res = {'no_video': False, 'model': model, 'attempt': att.name, 'video_duration_s': dur, 'answers': answers}
        a = res['answers']
        res.update(version=VERSION, system=system, task=task, run_started=started, run_dir=str(C), judged_at=time.time(),
                   questions=len(a), correct=sum(x['correct'] for x in a), score=round(sum(x['correct'] for x in a) / len(a), 4),
                   answered_A_D=sum(1 for x in a if x.get('choice') not in (None, 'E')),
                   verified=sum(1 for x in a if x.get('verified')),
                   correct_before_check=sum(1 for x in a if x.get('choice') and x.get('choice') == x.get('key')))
        jv.save(rdir / 'result.json', res)
        (rdir / 'failed.json').unlink(missing_ok=True)
        return system, task, 'graded', f"{res['correct']}/{res['questions']}"
    except Exception as exc:
        jv.save(rdir / 'failed.json', {'run_started': started, 'at': time.time(), 'error': repr(exc)[:2000],
                                       'trace': traceback.format_exc()[-4000:]})
        return system, task, 'failed', repr(exc)[:300]


def jobs():
    """Every ended run (terminal.json) of a task that has a quiz file."""
    return [(s, C.name, 'edu', C) for s in jv.systems() for C in sorted((jv.RUNS / s).iterdir())
            if (QUIZ / f'{C.name}.json').exists() and (C / 'terminal.json').exists()]


def report(quiet=False):
    by = collections.defaultdict(collections.Counter)
    for s, t, _, C in jobs():
        r = load(OUT / s / t / 'result.json')
        tt = load(C / 'terminal.json') or {}
        if not r or r.get('run_started') != str(tt.get('started')):
            by[s]['pending'] += 1
            continue
        c = by[s]
        c['runs'] += 1; c['no_video'] += r['no_video']; c['questions'] += r['questions']; c['correct'] += r['correct']
        c['answered_A_D'] += r['answered_A_D']; c['verified'] += r['verified']; c['correct_before_check'] += r['correct_before_check']
        c['score_sum_x1000'] += round(1000 * r['score'])
    summary = {s: {**c, 'mean_score': round(c['score_sum_x1000'] / c['runs'] / 1000, 4) if c['runs'] else None,
                   'verified_share_of_A_D': round(c['verified'] / c['answered_A_D'], 4) if c['answered_A_D'] else None}
               for s, c in sorted(by.items())}
    jv.save(OUT / 'summary.json', {'updated': time.time(), 'by_system': summary})
    if not quiet:
        for s, v in summary.items():
            print(f"{s:22} runs {v.get('runs', 0):3} (no video {v.get('no_video', 0)}, pending {v.get('pending', 0)}) "
                  f"mean score {v['mean_score']}  correct {v.get('correct', 0)}/{v.get('questions', 0)}  "
                  f"A-D verified {v['verified_share_of_A_D']}  correct before evidence check {v.get('correct_before_check', 0)}")


def main():
    global OUT
    ap = argparse.ArgumentParser(description='Grounded quiz for the edu tasks (see the module docstring).')
    sub = ap.add_subparsers(dest='cmd', required=True)

    def paths(p):
        p.add_argument('--runs', help=f'run folder holding <system>/<task>/ (default {jv.PATHS["runs"]})')
        p.add_argument('--out', help=f'result folder (default {jv.PATHS["quiz_results"]})')
        p.add_argument('--judge-results',
                       help=f'judge_v4 result folder, for its media cache (default {jv.PATHS["judge_results"]})')
    r = sub.add_parser('run', help='grade every ended run of a task with a quiz')
    paths(r)
    r.add_argument('--watch', type=int, default=0, help='repeat every N seconds (0: one pass)')
    r.add_argument('--workers', type=int, default=4)
    o = sub.add_parser('one', help='grade one run in the foreground')
    paths(o)
    o.add_argument('system')
    o.add_argument('task')
    paths(sub.add_parser('report', help='mean score per system'))
    a = ap.parse_args()
    jv.configure(argparse.Namespace(runs=a.runs, out=a.judge_results))
    if a.out:
        OUT = Path(a.out).resolve()
    if a.cmd == 'report':
        return report()
    if a.cmd == 'one':
        print(grade(a.system, a.task, 'edu', jv.RUNS / a.system / a.task))
        print(json.dumps(load(OUT / a.system / a.task / 'result.json'), ensure_ascii=False, indent=1)[:5000])
        return
    while True:
        js = [j for j in jobs()]
        with cf.ThreadPoolExecutor(a.workers) as ex:
            for f in cf.as_completed([ex.submit(grade, *j) for j in js]):
                s, t, st, info = f.result()
                if st != 'current':
                    print(time.strftime('%F %T'), st, s, t, info, flush=True)
        report(quiet=True)
        if not a.watch:
            break
        time.sleep(a.watch)


if __name__ == '__main__':
    main()

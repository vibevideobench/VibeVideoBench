#!/usr/bin/env python3
"""Summary of a benchmark run: per system, the deliverable / playable rate, the video score, the rubric item pass rate,
Strict, the L1 program checks and the edu quiz, for all tasks and for the main / edu task sets.

Inputs (read only):
  <runs>/<system>/<task>/terminal.json                 the run that counts (ended); "video", "status", "started"
  <judge-results>/<group>/<system>/<task>/result.json  judge groups gemini4_argon, gemini38_flash (judge_v4.py) and
                                                       omni (judge_v4_p2.py); scoring "judged" or "no_video_zero"
  <l1>/<system>/<task>/result.json                     L1 program checks (l1_judge.py)
  <quiz>/<system>/<task>/result.json                   edu quiz (quiz_judge.py)
  data/rubrics/<task>.json, data/tags/<task>.json       item criticality (core ...) and modality tags
A result counts only for the current run (its run_started within 1 s of terminal.json's started).

Metrics (per system; n = the number of tasks in the set; a judge group's figure first, then the mean of the groups):
  deliver     runs that delivered an MP4 / n
  playable    runs with an MP4 that at least one group scored ("judged") / n
  score0      per group: per dimension (VQ VA CL TM AQ) the mean of the 1-5 scores, every run without a scorable video
              entering as 0 (a dimension a judge left unknown / not applicable stays out); score0 = mean of the five
  score       the same over the scored videos only (no zeros)
  rate        rubric item pass rate: pass / (pass + fail) over all items of the scored videos (unknown and N/A out)
  strict      share of runs (with a result of the group, no video = not passed) whose core items have no fail and at
              least one pass (unknown / N/A allowed)
  kinds       rate split by the item's modality tag (visual / audio / audio_visual)
  l1_rate     L1 pass / (pass + fail) over the runs with a video; l1_unknown = unknown / all L1 items of those runs;
              l1_share = mean per run of its passed share of L1 items (no video = 0)
  quiz        edu tasks: mean quiz score (correct / questions; no video = 0)
Systems are ordered by score0.

Writes <out>/summary.md (one table per task set), <out>/summary.json, <out>/per_task.csv.
  python3 -B judge/report.py [--runs runs] [--judge-results judge_results] [--l1 l1_results] [--quiz quiz_results]
                             [--out report]
"""
import argparse
import collections
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import judge_v4 as jv   # noqa: E402

GROUPS = ['gemini4_argon', 'gemini38_flash', 'omni']
DIMS = jv.DIMS
KINDS = ['visual', 'audio', 'audio_visual']
SETS = {'all': 'All tasks', 'main': 'Main tasks', 'edu': 'Education tasks (PA*, ZH*)'}
TOL = 1.0


def task_set(task):
    return 'edu' if task.startswith(('PA', 'ZH')) else 'main'


def same_start(a, b):
    try:
        return abs(float(a) - float(b)) <= TOL
    except (TypeError, ValueError):
        return False


def num(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def first(dirs, name):
    return next((d / name for d in dirs if (d / name).exists()), None)


def load_inputs(a):
    rub_dirs, tag_dirs = [Path(p) for p in a.rubrics], [Path(p) for p in a.tags]
    tasks = sorted({p.stem for d in rub_dirs for p in d.glob('*.json')})
    crit, tags = {}, {}
    for t in tasks:
        r = jv.load(first(rub_dirs, f'{t}.json')) or {}
        crit[t] = {it['id']: it.get('criticality') for it in r.get('task_rubric') or []}
        tf = first(tag_dirs, f'{t}.json')
        tags[t] = (jv.load(tf) or {}).get('tags') or {} if tf else {}
    runs = {}
    for S in sorted(p for p in Path(a.runs).iterdir() if p.is_dir() and not p.name.startswith('_')):
        for C in sorted(p for p in S.iterdir() if p.is_dir() and p.name in crit):
            t = jv.load(C / 'terminal.json')
            if isinstance(t, dict):
                runs[(S.name, C.name)] = {'started': t.get('started'), 'video': bool(t.get('video')), 'status': t.get('status')}
    JR = Path(a.judge_results)
    res = {g: {} for g in GROUPS}
    for g in GROUPS:
        for (s, c), run in runs.items():
            r = jv.load(JR / g / s / c / 'result.json')
            if not isinstance(r, dict) or not same_start(r.get('run_started'), run['started']):
                continue
            items = {it.get('id'): it.get('verdict') for it in r.get('items') or []}
            dims = {d.get('id'): d.get('score') for d in r.get('dimensions') or []}
            res[g][(s, c)] = {'judged': r.get('scoring') == 'judged', 'items': items, 'dims': dims}
    l1, quiz = {}, {}
    for (s, c), run in runs.items():
        r = jv.load(Path(a.l1) / s / c / 'result.json')
        if isinstance(r, dict) and same_start(r.get('run_started'), run['started']):
            l1[(s, c)] = r
        r = jv.load(Path(a.quiz) / s / c / 'result.json')
        if isinstance(r, dict) and same_start(r.get('run_started'), run['started']):
            quiz[(s, c)] = r
    return tasks, crit, tags, runs, res, l1, quiz


def strict_case(d, crit):
    """passed / no_video / core_fail / no_pass for one group's result of one run."""
    if not d['judged']:
        return 'no_video'
    codes = [d['items'][i] for i, c in crit.items() if c == 'core' and i in d['items']]
    return 'core_fail' if 'fail' in codes else 'no_pass' if 'pass' not in codes else 'passed'


def group_stats(keys, res, crit, tags):
    """One judge group over the runs (keys) it has a current result for."""
    mine = {k: res[k] for k in keys if k in res}
    judged = [d for d in mine.values() if d['judged']]
    zero = len(mine) - len(judged)
    v = collections.Counter(x for d in judged for x in d['items'].values())
    kind = {x: collections.Counter() for x in KINDS}
    for (s, c), d in mine.items():
        if d['judged']:
            for i, x in d['items'].items():
                if tags[c].get(i) in kind:
                    kind[tags[c][i]][x] += 1
    dim, dim0 = {}, {}
    for D in DIMS:
        xs = [d['dims'][D] for d in judged if num(d['dims'].get(D))]
        dim[D] = sum(xs) / len(xs) if xs else None
        dim0[D] = sum(xs) / (len(xs) + zero) if xs or zero else None
    why = collections.Counter(strict_case(d, crit[c]) for (s, c), d in mine.items())
    pf = v['pass'] + v['fail']
    return {'runs': len(mine), 'judged': len(judged), 'zero': zero,
            'rate': 100 * v['pass'] / pf if pf else None,
            'items': {x: v[x] for x in ('pass', 'fail', 'unknown', 'not_applicable')},
            'dims': dim, 'score': mean(dim.values()) if all(x is not None for x in dim.values()) else None,
            'score0': mean(dim0.values()) if all(x is not None for x in dim0.values()) else None,
            'strict': 100 * why['passed'] / len(mine) if mine else None,
            'strict_why': {k: 100 * why[k] / len(mine) if mine else None for k in ('no_video', 'core_fail', 'no_pass')},
            'kinds': {x: (100 * kind[x]['pass'] / (kind[x]['pass'] + kind[x]['fail'])
                          if kind[x]['pass'] + kind[x]['fail'] else None) for x in KINDS}}


def avg(per, f):
    xs = [f(p) for p in per.values() if p['runs']]
    return mean(xs) if xs and all(x is not None for x in xs) else None


def summarize(tasks, crit, tags, runs, res, l1, quiz, which):
    tset = [t for t in tasks if which == 'all' or task_set(t) == which]
    n = len(tset)
    out = {}
    for s in sorted({s for s, _ in runs}):
        keys = [(s, t) for t in tset if (s, t) in runs]
        per = {g: group_stats(keys, res[g], crit, tags) for g in GROUPS}
        mp4 = [k for k in keys if runs[k]['video']]
        playable = sum(1 for k in mp4 if any(k in res[g] and res[g][k]['judged'] for g in GROUPS))
        lc, share, l1n = collections.Counter(), [], 0
        for k in keys:
            r = l1.get(k)
            if not r:
                continue
            l1n += 1
            its = r.get('items') or []
            if not r.get('no_video'):
                lc.update(x['verdict'] for x in its)
            if its:
                share.append(sum(x['verdict'] == 'pass' for x in its) / len(its))
        dec, ln = lc['pass'] + lc['fail'], sum(lc.values())
        qz = [quiz[k]['score'] for k in keys if k in quiz and num(quiz[k].get('score'))]
        out[s] = {
            'tasks': n, 'ended': len(keys), 'mp4': len(mp4), 'playable_runs': playable,
            'timed_out': sum(1 for k in keys if runs[k]['status'] == 'timed_out'),
            'deliver': 100 * len(mp4) / n if n else None, 'playable': 100 * playable / n if n else None,
            'score0': avg(per, lambda p: p['score0']), 'score': avg(per, lambda p: p['score']),
            'rate': avg(per, lambda p: p['rate']), 'strict': avg(per, lambda p: p['strict']),
            'strict_why': {w: avg(per, lambda p, w=w: p['strict_why'][w]) for w in ('no_video', 'core_fail', 'no_pass')},
            'dims': {D: avg(per, lambda p, D=D: p['dims'][D]) for D in DIMS},
            'kinds': {x: avg(per, lambda p, x=x: p['kinds'][x]) for x in KINDS},
            'judge_groups': {g: p for g, p in per.items()},
            'l1_runs': l1n, 'l1_rate': 100 * lc['pass'] / dec if dec else None,
            'l1_unknown': 100 * lc['unknown'] / ln if ln else None,
            'l1_share': 100 * mean(share) if share else None,
            'quiz_runs': len(qz), 'quiz': 100 * mean(qz) if qz else None}
    return dict(sorted(out.items(), key=lambda kv: -(kv[1]['score0'] if kv[1]['score0'] is not None else -1)))


def f(x, d=1, pct=False):
    return '–' if x is None else f'{x:.{d}f}' + ('%' if pct else '')


def markdown(summary):
    lines = ['# Benchmark summary', '']
    for which, rows in summary.items():
        if not rows:
            continue
        n = next(iter(rows.values()))['tasks']
        lines += [f'## {SETS[which]} ({n})', '',
                  '| System | Runs | Deliver | Playable | Score (no video = 0) | Score (playable) | Rubric pass | Strict | '
                  'L1 pass | Quiz |', '|---|---|---|---|---|---|---|---|---|---|']
        for s, r in rows.items():
            lines.append(f"| {s} | {r['ended']}/{r['tasks']} | {f(r['deliver'], 1, True)} | {f(r['playable'], 1, True)} | "
                         f"{f(r['score0'], 2)} | {f(r['score'], 2)} | {f(r['rate'], 1, True)} | {f(r['strict'], 1, True)} | "
                         f"{f(r['l1_rate'], 1, True)} | {f(r['quiz'], 1, True) if which != 'main' else '–'} |")
        lines += ['', '| System | ' + ' | '.join(DIMS) + ' | visual items | audio items | audio-visual items |',
                  '|---|' + '---|' * (len(DIMS) + 3)]
        for s, r in rows.items():
            lines.append(f'| {s} | ' + ' | '.join(f(r['dims'][D], 2) for D in DIMS) + ' | ' +
                         ' | '.join(f(r['kinds'][x], 1, True) for x in KINDS) + ' |')
        lines.append('')
    lines += ['Deliver: runs with an MP4 / tasks. Playable: MP4s a judge group could score / tasks. '
              'Scores: mean of the five 1-5 dimensions (VQ VA CL TM AQ); "no video = 0" counts every run without a '
              'scorable video as 0. Rubric pass: pass / (pass + fail) over the items of the scored videos. Strict: '
              'share of runs whose core items have no fail and at least one pass (no video = not passed). '
              'Every judge-based figure is computed per judge group (gemini4_argon, gemini38_flash, omni) and averaged. '
              'L1 pass: program checks, pass / (pass + fail) over runs with a video. Quiz: education tasks, mean score.', '']
    return '\n'.join(lines)


def per_task_rows(tasks, crit, runs, res, l1, quiz):
    for (s, t), run in sorted(runs.items()):
        row = {'system': s, 'task': t, 'set': task_set(t), 'status': run['status'], 'video': run['video']}
        for g in GROUPS:
            d = res[g].get((s, t))
            row[f'{g}_scoring'] = None if d is None else 'judged' if d['judged'] else 'no_video_zero'
            if d:
                v = collections.Counter(d['items'].values())
                row[f'{g}_pass'], row[f'{g}_fail'], row[f'{g}_unknown'] = v['pass'], v['fail'], v['unknown']
                row[f'{g}_strict'] = strict_case(d, crit[t]) == 'passed'
                sc = [d['dims'].get(D) for D in DIMS]
                row[f'{g}_score'] = round(mean(sc), 3) if d['judged'] and all(num(x) for x in sc) else (0 if not d['judged'] else None)
        r = l1.get((s, t))
        if r:
            v = collections.Counter(x['verdict'] for x in r.get('items') or [])
            row['l1_pass'], row['l1_fail'], row['l1_unknown'] = v['pass'], v['fail'], v['unknown']
        q = quiz.get((s, t))
        if q:
            row['quiz_score'] = q.get('score')
        yield row


def main():
    ap = argparse.ArgumentParser(description='Benchmark summary per system (see the module docstring).')
    ap.add_argument('--runs', default=str(jv.path('runs')), help='run folder holding <system>/<task>/ (default runs)')
    ap.add_argument('--judge-results', default=str(jv.path('judge_results')), help='default judge_results')
    ap.add_argument('--l1', default=str(jv.path('l1_results')), help='default l1_results')
    ap.add_argument('--quiz', default=str(jv.path('quiz_results')), help='default quiz_results')
    ap.add_argument('--rubrics', nargs='+', default=[str(jv.path('rubrics'))], help='rubric folder(s) (default data/rubrics)')
    ap.add_argument('--tags', nargs='+', default=[str(jv.path('tags'))], help='tag folder(s) (default data/tags)')
    ap.add_argument('--out', default=str(jv.ROOT / 'report'), help='output folder (default report)')
    a = ap.parse_args()
    tasks, crit, tags, runs, res, l1, quiz = load_inputs(a)
    if not runs:
        sys.exit(f'no ended runs (terminal.json) under {a.runs}')
    summary = {w: summarize(tasks, crit, tags, runs, res, l1, quiz, w) for w in SETS}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jv.save(out / 'summary.json', {'tasks': len(tasks), 'judge_groups': GROUPS, 'sets': summary})
    (out / 'summary.md').write_text(markdown(summary), encoding='utf-8')
    rows = list(per_task_rows(tasks, crit, runs, res, l1, quiz))
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(out / 'per_task.csv', 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print(f'{len(runs)} runs of {len(tasks)} tasks -> {out}/summary.md, summary.json, per_task.csv')


if __name__ == '__main__':
    main()

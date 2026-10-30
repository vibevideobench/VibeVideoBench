#!/usr/bin/env python3
"""Freeze a task directory (<ID>/{prompt.txt, task.json, inputs/}, default ../data/tasks) into an experiment's
run_case snapshot, and write the experiment's selection.json (the cases schedule.py runs, in index order).

  <experiment>/snapshot/model_inputs/<ID>/        byte copy of the task (what run_case copies to <case>/inputs)
  <experiment>/snapshot/inventory.json            per task: title, language, delivery, prompt hash, input files
  <experiment>/snapshot/evaluation/tasks/<ID>.json   the program checks the collector expects (T01-T05; T07 with an
                                                  exact frame_count; T06 when delivery.audio_required)
  <experiment>/snapshot/manifest.json             source, index.json sha256, per-task source sha256
  <experiment>/selection.json                     {"tasks": [{"task_id", ...}, ...]}

The task directory is read only. An existing snapshot is never overwritten (runs refer to it): freeze into a new
experiment instead.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

HERE = Path(__file__).resolve().parent
CHECKS = {'T01': '提交文件存在', 'T02': '完整解码', 'T03': '画幅', 'T04': '帧率', 'T05': '时长', 'T06': '必需声音的音频流存在',
          'T07': '实际解码总帧数符合精确要求'}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def save(p, v):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=1) + '\n')


def freeze(data, x, root):
    """Copy one task to root/model_inputs/<ID>; return its eval checks, inventory row, selection row, source hashes
    and problems (inputs/... paths its prompt.txt or task.json names that do not exist)."""
    tid = x['task_id']
    src = data / tid
    g = root / 'model_inputs' / tid
    shutil.copytree(src, g, symlinks=False)
    task = json.loads((g / 'task.json').read_text())
    prompt = (g / 'prompt.txt').read_bytes()
    assets = [{'path': p.relative_to(g).as_posix(), 'bytes': p.stat().st_size, 'sha256': sha(p.read_bytes())}
              for p in sorted((g / 'inputs').rglob('*')) if p.is_file()] if (g / 'inputs').is_dir() else []
    ids = ['T01', 'T02', 'T03', 'T04', 'T05'] + ['T06'] * bool(task['delivery'].get('audio_required')) + ['T07'] * ('frame_count' in task['delivery'])
    checks = {'task_id': tid, 'version': task.get('version'), 'title': task['title'],
              'evaluation': {'program_checks': [{'id': i, 'name': CHECKS[i], 'evaluator': 'program',
                                                 'implementation': 'controller/check_delivery.py'}
                                                for i in ids]}}
    language = x.get('language') or task.get('language')
    inventory = {'task_id': tid, 'title': task['title'], 'family': tid[0], 'language': language,
                 'delivery': task['delivery'], 'version': task.get('version'), 'prompt_sha256': sha(prompt),
                 'assets': assets, 'audio_required': bool(task['delivery'].get('audio_required'))}
    selection = {'task_id': tid, 'title': task['title'], 'language': language}
    source = {'task_json_sha256': sha((src / 'task.json').read_bytes())}
    problems = [f'{tid}/{f} names missing {r}' for f in ['prompt.txt', 'task.json']
                for r in re.findall(r'(?<![\w/])/?inputs/[\w./-]*[\w-]', (g / f).read_text())
                if not (g / r.lstrip('/')).exists()]
    print(tid, len(assets), 'files', flush=True)
    return checks, inventory, selection, source, problems


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('experiment', type=Path, help='experiment directory (created if missing)')
    ap.add_argument('--data', type=Path, default=HERE.parent / 'data/tasks', help='task directory (default: ../data/tasks)')
    ap.add_argument('--only', type=lambda x: x.split(','), help='comma-separated task ids (default: every task)')
    a = ap.parse_args()
    data, E = a.data.resolve(), a.experiment.resolve()
    if (data / 'index.json').exists():
        index = json.loads((data / 'index.json').read_text())['tasks']
    else:   # no index: every <ID>/task.json, sorted
        index = [{'task_id': p.parent.name} for p in sorted(data.glob('*/task.json'))]
    if a.only:
        known = {x['task_id'] for x in index}
        assert set(a.only) <= known, sorted(set(a.only) - known)
        index = [x for x in index if x['task_id'] in a.only]
    assert index, f'no tasks in {data}'
    snap = E / 'snapshot'
    assert not snap.exists(), f'{snap} exists; runs may refer to it (freeze into a new experiment)'
    stage = E / 'snapshot.partial'
    shutil.rmtree(stage, ignore_errors=True)
    inventory, selection, sources, problems = [], [], {}, []
    for x in index:
        checks, row, sel, sources[x['task_id']], bad = freeze(data, x, stage)
        save(stage / 'evaluation/tasks' / (x['task_id'] + '.json'), checks)
        inventory.append(row)
        selection.append(sel)
        problems += bad
    if problems:
        shutil.rmtree(stage)
        raise SystemExit('nothing frozen:\n' + '\n'.join(problems))
    save(stage / 'inventory.json', inventory)
    index_json = data / 'index.json'
    save(stage / 'manifest.json', {'source': str(data),
                                   'index_sha256': sha(index_json.read_bytes()) if index_json.exists() else None,
                                   'case_ids': [x['task_id'] for x in index], 'tasks': sources,
                                   'asset_files': sum(len(x['assets']) for x in inventory)})
    stage.rename(snap)
    save(E / 'selection.json', {'source': str(data), 'tasks': selection})
    print(len(selection), 'tasks frozen into', snap, flush=True)


if __name__ == '__main__':
    main()

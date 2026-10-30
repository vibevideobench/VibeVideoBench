#!/usr/bin/env python3
"""Collect a finished run's delivery: pick the submitted MP4, copy it to <case>/collected/video.mp4 and run the
program delivery checks (check_delivery.py) against the frozen task; writes <case>/collected/delivery.json.

Video selection, in order: submission.json "video"; the latest pre-deadline MP4 named in the agent's final reply;
the latest pre-deadline MP4 in the workspace. Never chosen by score.

  collect.py --case-root <experiment>/runs/<system>/<case> --snapshot <experiment>/snapshot
"""
import argparse
import json
from pathlib import Path
import shutil
import time

import check_delivery
from util import read, save, sha


def rollout_events(C):
    p = Path(C) / 'rollout.jsonl'
    for line in (p.read_text(errors='replace').splitlines() if p.exists() else []):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        if isinstance(e, dict):
            yield e


def messages(C):
    """The agent's final replies, in order, from each harness's stream (rollout.jsonl)."""
    replies = []
    for e in rollout_events(C):
        if e.get('type') == 'item.completed' and e.get('item', {}).get('type') == 'agent_message':
            replies.append(e['item'].get('text', ''))                            # Codex
        if e.get('event') == 'result' and isinstance(e.get('result'), dict):
            replies.append(e['result'].get('response', ''))                      # Antigravity
        if e.get('type') == 'result' and isinstance(e.get('result'), str):
            replies.append(e['result'])                                          # Claude Code
        if e.get('type') == 'final' and isinstance(e.get('text'), str):
            replies.append(e['text'])                                            # dsh
        if e.get('type') == 'result' and isinstance(e.get('response'), str):
            replies.append(e['response'])                                        # ZCode
        if e.get('role') == 'assistant' and isinstance(e.get('content'), str) and not e.get('tool_calls'):
            replies.append(e['content'])                                         # Kimi Code: a turn's closing reply
    return replies


def workspace_path(W, text):
    if not isinstance(text, str):
        return None
    p = Path(text)
    if p.is_absolute():
        if not str(p).startswith('/workspace/'):
            return None
        p = Path(str(p)[len('/workspace/'):])
    q = (W / p).resolve()
    return q if q.is_relative_to(W.resolve()) else None


def collect(C, snapshot):
    run = read(C / 'run.json')
    task = read(C / 'inputs/task.json')
    W = C / 'workspace'
    candidates = [p for p in W.rglob('*') if p.suffix.lower() == '.mp4' and p.is_file() and p.resolve().is_relative_to(W.resolve())
                  and p.stat().st_mtime <= run['deadline'] and p.stat().st_size > 0]
    manifest = {}
    if (W / 'submission.json').exists():
        try:
            manifest = read(W / 'submission.json')
        except (ValueError, OSError):
            pass
    explicit = workspace_path(W, manifest.get('video'))
    selected = explicit if explicit in candidates else None
    reason = 'submission.json' if selected else None
    replies = messages(C)
    final_text = replies[-1] if replies else ''
    if not selected:
        named = [p for p in candidates if str(p.relative_to(W)) in final_text or '/workspace/' + str(p.relative_to(W)) in final_text]
        if named:
            selected = max(named, key=lambda p: p.stat().st_mtime)
            reason = 'native_final_message'
    if not selected and candidates:
        selected = max(candidates, key=lambda p: p.stat().st_mtime)
        reason = 'latest_pre_deadline_mp4_fallback'
    declared_project = workspace_path(W, manifest.get('project')) or W
    # A single editable source/project file is a valid entry point. T01 checks
    # the presence of its existing containing directory, not the manifest's
    # choice of a directory vs an entry-file spelling. No model files are made.
    project = declared_project.parent if declared_project.is_file() else declared_project
    project_resolution = 'existing_entry_file_containing_directory' if project != declared_project else 'declared_directory_or_workspace'
    timeline = workspace_path(W, manifest.get('timeline'))
    if not timeline or not timeline.is_file():
        options = [p for pattern in ['*timeline*', '*storyboard*', '*分镜*', '*时间表*'] for p in W.rglob(pattern)
                   if p.is_file() and p.resolve().is_relative_to(W.resolve())
                   and p.suffix in ['.json', '.md', '.txt', '.csv', '.html']
                   and not any(x in p.parts for x in ['node_modules', '.cache', '.venv'])]
        timeline = min(options, key=lambda p: len(str(p))) if options else W / '__missing_timeline__'
    out = C / 'collected'
    out.mkdir(exist_ok=True)
    if selected:
        shutil.copy2(selected, out / 'video.mp4')
        assert sha(selected) == sha(out / 'video.mp4')
    checks = check_delivery.check(task, out / 'video.mp4', project, timeline)
    expected = {r['id'] for r in read(Path(snapshot) / 'evaluation/tasks' / (task['task_id'] + '.json'))['evaluation']['program_checks']}
    observed = {r['id']: r for r in checks}
    for missing in expected - observed.keys():
        checks.append({'id': missing, 'verdict': 'fail' if not selected else 'unknown',
                       'evidence': 'No MP4 delivered' if not selected else 'Checker did not produce this required measurement',
                       'evaluator': 'program'})
    technical_pass = all(observed.get(cid, {}).get('verdict') == 'pass' for cid in expected)
    frozen_inputs_unchanged = all(sha(C / 'inputs' / p) == h for p, h in read(C / 'prepared.json')['input_hashes'].items())
    source_files = []
    for p in W.rglob('*'):
        if not p.is_file() or p.is_symlink() or any(x in p.parts for x in ['node_modules', '.cache', '.venv', '.python', '__pycache__']):
            continue
        if p.suffix.lower() in ['.py', '.js', '.mjs', '.cjs', '.html', '.css', '.ts', '.tsx', '.jsx', '.md', '.json', '.csv']:
            source_files.append({'path': str(p.relative_to(W)), 'bytes': p.stat().st_size, 'sha256': sha(p)})
    result = {'case_id': task['task_id'], 'system': run['system'], 'completed': time.time(),
              'selected_video': str(selected) if selected else None, 'selection_rule': reason,
              'video': str(out / 'video.mp4') if selected else None,
              'video_sha256': sha(out / 'video.mp4') if selected else None,
              'video_bytes': (out / 'video.mp4').stat().st_size if selected else 0,
              'project': str(project), 'timeline': str(timeline) if timeline.is_file() else None,
              'declared_project_entry': str(declared_project), 'project_path_resolution': project_resolution,
              'native_final_reply': final_text, 'mp4_candidates': len(candidates),
              'checks': checks, 'technical_pass': technical_pass,
              'input_hashes_unchanged': frozen_inputs_unchanged,
              'checker_sha256': sha(check_delivery.__file__),
              'collection_implementation_sha256': sha(Path(__file__)),
              'source_and_notes_files': source_files,
              'limitations': ['Delivery checks do not prove content, quality, project editability or source licensing.'],
              'content_and_quality': 'not_yet_evaluated'}
    save(out / 'delivery.json', result)
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--case-root', type=Path, required=True)
    ap.add_argument('--snapshot', type=Path, required=True, help='the frozen snapshot the run used')
    a = ap.parse_args()
    r = collect(a.case_root.resolve(), a.snapshot.resolve())
    print(json.dumps({k: r[k] for k in ['case_id', 'system', 'video', 'technical_pass', 'input_hashes_unchanged']}))

#!/usr/bin/env python3
"""Run every case of <experiment>/selection.json on every system of <experiment>/experiment.json through run_case.py,
at most --per-system at a time per system (systems proceed independently, in selection order).

Restartable: runs with terminal.json are done and skipped, live controllers are adopted. A run directory without
terminal.json and without a live controller (the machine went down mid-run) is reported as an orphan and left alone:
remove it to run that case again. Once every provider key of a system is set aside for exhausted credit (the gateway's
<experiment>/runs/key_state.json), that system gets no new launches; runs that ended for that reason are listed as
cut_by_credit in queue.json (remove them and restart after a top-up). Writes <experiment>/queue.json.

  schedule.py <experiment> [--per-system 4] [--stagger 20] [--only sys1,sys2] [--after b=a,...] [--total N] [-- run_case options]

run_case options default to experiment.json "run_case_args".

--after b=a: launch system b only once system a has nothing pending or running (b takes over a's slots, so the
machine load stays the same).

--total N: slots that no system can use (a system out of credit, or one with nothing left to launch) go to the system
with the most pending cases, waiting --after systems included, as long as at most N runs are live. Every system that
can still launch keeps its own --per-system slots, so a borrowed run never delays one.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
HOST = socket.gethostname()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    temp = path.with_name(f'{path.name}.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    temp.replace(path)


def alive(C, run):
    """Is the run's controller still running? Its pid if the run is on this host, else a fresh heartbeat file
    (run_case.py touches it every 30 s; a run directory on storage shared by several machines)."""
    try:
        pid = run.get('controller_pid')
        if run.get('host', HOST) == HOST and pid and 'run_case.py' in Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace'):
            return True
    except OSError:
        pass
    try:
        return time.time() - (C / 'heartbeat').stat().st_mtime < 120
    except OSError:
        return False


def out_of_credit(C):
    """Did the run end on a request the provider refused for exhausted credit?"""
    calls = sorted((C / 'native_api').glob('*/record.json'))
    if not calls:
        return False
    last = read(calls[-1])
    text = json.dumps(last)
    return bool(last.get('credit_exhausted')) or 'credit_balance_exhausted' in text or 'credit balance exhausted' in text


def pick(state, gave_up, per_system, total, slots=None):
    """The system to launch next: an own free slot first (experiment order), else an idle slot for the most work left.
    slots: {system: N} overriding per_system for those systems."""
    S, waiting = state['systems'], state['waiting_for']
    limit = lambda k: (slots or {}).get(k, per_system)
    open_ = {k: v for k, v in S.items() if v['pending'] and k not in gave_up}
    for k, v in open_.items():
        if k not in waiting and len(v['running']) < limit(k):
            return k
    reserved = sum(max(len(v['running']), limit(k) if k in open_ and k not in waiting else 0) for k, v in S.items())
    if open_ and reserved < total:
        return max(open_, key=lambda k: len(open_[k]['pending']))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('experiment', type=Path)
    ap.add_argument('--per-system', type=int, default=4)
    ap.add_argument('--slots', default='', help='comma-separated system=N overriding --per-system for those systems')
    ap.add_argument('--stagger', type=float, default=20, help='seconds between two launches')
    ap.add_argument('--only', help='comma-separated subset of the experiment systems')
    ap.add_argument('--after', default='', help='comma-separated later=earlier system pairs')
    ap.add_argument('--total', type=int, default=0, help='lend idle slots up to this many live runs (0: off)')
    a, extra = ap.parse_known_args()     # unknown options go to run_case.py
    E = a.experiment.resolve()
    cfg = read(E / 'experiment.json')
    cases = [t['task_id'] for t in read(E / 'selection.json')['tasks']]
    extra = [x for x in extra if x != '--'] or cfg.get('run_case_args', [])
    (E / 'logs').mkdir(exist_ok=True)
    systems = a.only.split(',') if a.only else cfg['systems']
    assert set(systems) <= set(cfg['systems']), systems
    after = dict(x.split('=') for x in a.after.split(',') if x)
    slots = {k: int(v) for k, v in (x.split('=') for x in a.slots.split(',') if x)}
    assert set(slots) <= set(systems), slots
    assert all(b in systems and x in systems and systems.index(x) < systems.index(b) for b, x in after.items()), after
    procs, last_launch, gave_up = {}, 0, {}
    while True:
        state = {'scheduler_pid': os.getpid(), 'host': HOST, 'updated': time.time(), 'per_system': a.per_system,
                 'slots': slots, 'gave_up': gave_up, 'systems': {}, 'orphans': [], 'cut_by_credit': [],
                 'after': after, 'waiting_for': {}, 'total': a.total, 'scheduled_systems': systems}
        keys = read(E / 'runs/key_state.json') if (E / 'runs/key_state.json').exists() else {}
        for system in systems:
            info = (keys.get('systems') or {}).get(system)
            held = bool(info) and len(info['exhausted_at']) == len(info['keys']) and \
                all(time.time() - t < keys['recheck_after_s'] for t in info['exhausted_at'].values())
            pending, running, ended = [], [], []
            for case in cases:
                C = E / 'runs' / system / case
                p = procs.get((system, case))
                if (C / 'terminal.json').exists():
                    if p and p.poll() is None:
                        running.append(case)
                        continue
                    ended.append(case)
                    # Claude Code -p exits 0 after reporting an API error as its result: native_finished, cut all the same
                    if read(C / 'terminal.json')['status'] in ('native_error', 'native_finished') and out_of_credit(C):
                        state['cut_by_credit'].append(f'{system}/{case}')
                elif (p and p.poll() is None) or ((C / 'run.json').exists() and alive(C, read(C / 'run.json'))):
                    running.append(case)
                elif C.exists() or (p and p.returncode != 0):   # controller gone without terminal.json / failed launch
                    state['orphans'].append(f'{system}/{case}')
                else:
                    pending.append(case)
            if held and system not in gave_up:
                gave_up[system] = time.time()
                print(time.strftime('%F %T'), system, 'all keys out of credit: no further launches', flush=True)
            state['systems'][system] = {'pending': pending, 'running': running, 'ended': ended}
            first = state['systems'].get(after.get(system), {})
            if first and ((first['pending'] and after[system] not in gave_up) or first['running']):
                state['waiting_for'][system] = after[system]
        system = pick(state, gave_up, a.per_system, a.total, slots) if time.time() - last_launch >= a.stagger else None
        if system:
            pending = state['systems'][system]['pending']
            case = pending[0]
            log = (E / 'logs' / f'run_{system}_{case}.log').open('a')
            cmd = [sys.executable, '-B', str(HERE / 'run_case.py'), '--experiment', str(E), '--system', system,
                   '--case', case, *extra]
            procs[(system, case)] = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                                     start_new_session=True)
            last_launch = time.time()
            print(time.strftime('%F %T'), 'launched', system, case, flush=True)
            state['systems'][system]['running'].append(case)
            pending.remove(case)
        save(E / 'queue.json', state)
        if all((not v['pending'] or k in gave_up) and not v['running'] for k, v in state['systems'].items()):
            print(time.strftime('%F %T'), 'all runs ended; orphans:', state['orphans'],
                  '; cut by credit:', state['cut_by_credit'], flush=True)
            break
        time.sleep(5)


if __name__ == '__main__':
    main()

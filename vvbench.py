#!/usr/bin/env python3
"""One entry point for the whole benchmark: set up, check, run, judge, report.

  python3 vvbench.py init                      copy the example configs that are missing (keys.env, systems.json, ...)
  python3 vvbench.py doctor [--systems a,b]    check image, agent CLIs, keys, ffmpeg and judge packages
  python3 vvbench.py fetch-data                optional: download the task inputs not shipped with the repository
                                               (data/EXTERNAL_INPUTS.md lists them; they must be in place before run)
  python3 vvbench.py run EXP --systems a,b     freeze the tasks into EXP/, run every task on every system, judge, report
  python3 vvbench.py judge EXP                 (re)judge the ended runs of EXP (skips what is already judged)
  python3 vvbench.py report EXP                write EXP/report/ from the judge results

`run` starts the model and TTS gateways itself (unless they already serve this experiment) and stops them at the end.
Everything is restartable: run the same command again after an interruption and it continues where it stopped
(finished cases have a terminal.json, finished judgments a result.json).

Experiment layout: EXP/{snapshot/, selection.json, experiment.json, runs/<system>/<task>/, logs/,
judge_results/, l1_results/, quiz_results/, report/}.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PY = [sys.executable, '-B']
EXAMPLES = {'keys.env': 'keys.env.example', 'systems.json': 'systems.example.json',
            'controller/config.json': 'controller/config.example.json', 'judge/config.json': 'judge/config.example.json'}
MODEL_PORT, TTS_PORT = 18885, 18883
# VVBENCH_CONFIG_DIR: a directory outside the repository holding systems.json, keys.env, controller.json and judge.json
# (the controller, gateways and judges read the same variable)
CONFIG_DIR = Path(os.environ['VVBENCH_CONFIG_DIR']).expanduser() if os.environ.get('VVBENCH_CONFIG_DIR') else None
IN_CONFIG_DIR = {'keys.env': 'keys.env', 'systems.json': 'systems.json', 'controller/config.json': 'controller.json',
                 'judge/config.json': 'judge.json'}


def conf(name):
    """Path of the config file NAME (a key of EXAMPLES): in VVBENCH_CONFIG_DIR if set, else in the repository."""
    return CONFIG_DIR / IN_CONFIG_DIR[name] if CONFIG_DIR else ROOT / name


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def save(p, v):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(v, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')


def keys_env():
    """Variables of keys.env (KEY=value lines), environment variables taking precedence."""
    out = {}
    p = conf('keys.env')
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                out[k.strip()] = v.strip().strip('"\'')
    out.update({k: v for k, v in os.environ.items() if k in out or k.endswith('_API_KEY') or k == 'DASHSCOPE_TTS_URL'})
    return out


def systems():
    return {k: v for k, v in read(conf('systems.json')).items() if not k.startswith('_')}


def listening(port):
    try:
        socket.create_connection(('127.0.0.1', port), 1).close()
        return True
    except OSError:
        return False


def health(port):
    try:
        with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
                f'http://127.0.0.1:{port}/health', timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def step(msg):
    print(f'\n== {msg}', flush=True)


def sh(cmd, **kw):
    print('$', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run([str(c) for c in cmd], cwd=ROOT, **kw).returncode


# ---------------------------------------------------------------- init / doctor

def cmd_init(a):
    for dst, src in EXAMPLES.items():
        if conf(dst).exists():
            print(f'kept    {conf(dst)}')
        else:
            conf(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / src, conf(dst))
            print(f'created {conf(dst)}  (from {src})')
    print(f'\nNext: put your API keys in {conf("keys.env")}, the agent CLI directories in {conf("controller/config.json")} '
          '(see docs/harnesses.md), then: python3 vvbench.py doctor --systems <names from systems.json>')


DATASET = 'diaomuxi/vibevideobench'   # huggingface.co/datasets/diaomuxi/vibevideobench
GET_DATASET = f'huggingface-cli download {DATASET} --repo-type dataset --local-dir data'


def have_dataset():
    return (ROOT / 'data/tasks/index.json').exists()


def external_missing():
    """Task inputs listed in data/external.json that are not downloaded yet."""
    p = ROOT / 'data/external.json'
    files = read(p)['files'] if p.exists() else []
    return [f'{e["task"]}/inputs/{e["path"]}' for e in files if not (ROOT / 'data/tasks' / e['task'] / 'inputs' / e['path']).exists()]


def fetch_data():
    step('download the task inputs not shipped with the repository (data/EXTERNAL_INPUTS.md)')
    return sh([*PY, ROOT / 'data/fetch_external.py'])


def checks(names, judge=True):
    """[(ok, what, hint)] for everything a run of these systems needs."""
    missing = external_missing()
    out = [(have_dataset(), 'dataset in data/ (tasks, rubrics, tags, quiz)', f'download it: {GET_DATASET}'),
           (have_dataset() and not missing, 'external task inputs in place (data/EXTERNAL_INPUTS.md)',
            f'{len(missing)} missing, e.g. {", ".join(missing[:3])}: download them as data/EXTERNAL_INPUTS.md says '
            '(or try python3 vvbench.py fetch-data)')]
    for dst, src in EXAMPLES.items():
        out.append((conf(dst).exists(), f'{conf(dst).relative_to(ROOT) if not CONFIG_DIR else conf(dst)} exists', f'python3 vvbench.py init (copies {src})'))
    if not all(ok for ok, _, _ in out[2:]):     # without the config files nothing else can be checked
        return out
    cfg, sy, env = read(conf('controller/config.json')), systems(), keys_env()
    unknown = [n for n in names if n not in sy]
    out.append((not unknown, f'systems {",".join(names) or "(none given)"} are in systems.json',
                f'unknown: {unknown}; defined: {", ".join(sy)}'))
    denv = dict(os.environ, **({'DOCKER_HOST': cfg['docker_host']} if cfg.get('docker_host') else {}))
    image = cfg.get('image', 'vibe-l0:0.1')
    ok = shutil.which('docker') and subprocess.run(['docker', 'image', 'inspect', image], env=denv,
                                                   capture_output=True).returncode == 0
    out.append((bool(ok), f'sandbox image {image} is built', 'cd docker && ./fetch.sh && ./build.sh'))
    tools = [cfg.get('ffmpeg_dir') or '', '']
    for tool in ('ffmpeg', 'ffprobe'):
        found = any(shutil.which(tool, path=d or None) for d in tools)
        out.append((found, f'{tool} on the host', 'install ffmpeg (or set ffmpeg_dir in controller/config.json)'))
    for n in names:
        if n not in sy:
            continue
        h = sy[n].get('harness_install') or sy[n].get('harness')
        d = (cfg.get('harness') or {}).get(h)
        out.append((bool(d) and Path(d).is_dir(), f'{n}: agent CLI at controller/config.json harness.{h}',
                    f'install it and set harness.{h} (docs/harnesses.md); now: {d!r}'))
        vars_ = sy[n].get('key_vars') or ([sy[n]['key_var']] if sy[n].get('key_var') else [])
        kf = Path(sy[n].get('key_file') or 'keys.env').expanduser()
        kf = kf if kf.is_absolute() else conf('systems.json').parent / kf     # as the gateway resolves it
        if vars_ and kf.resolve() == (conf('keys.env')).resolve():
            missing = [v for v in vars_ if not env.get(v) or env[v].endswith('...')]
            out.append((not missing, f'{n}: API key {",".join(vars_)} in keys.env', f'missing: {missing}'))
        elif vars_:     # another KEY=value file
            text = kf.read_text() if kf.exists() else ''
            missing = [v for v in vars_ if f'{v}=' not in text]
            out.append((not missing, f'{n}: API key {",".join(vars_)} in {kf}', f'missing: {missing}'))
        else:           # the whole file is the key
            out.append((kf.exists() and kf.stat().st_size > 0, f'{n}: key file {kf}', 'missing or empty'))
    tts = env.get('DASHSCOPE_API_KEY', '')
    out.append((tts.startswith('sk-') and len(tts) >= 20 and not tts.endswith('...'),
                'DASHSCOPE_API_KEY in keys.env (TTS tool in the sandbox)',
                'a DashScope key (sk-...): agents use Qwen3-TTS for narration through the host TTS gateway'))
    if judge:
        jc = read(conf('judge/config.json'))
        roles = [*(jc.get('judges') or {}).values(), *(jc.get(r) for r in ('precheck', 'quiz', 'gpt', 'qwen'))]
        jvars = sorted({e['key_var'] for r in roles if r for e in (r if isinstance(r, list) else [r])})
        missing = [v for v in jvars if not env.get(v) or env[v].endswith('...')]
        out.append((not missing, f'judge keys {",".join(jvars)} in keys.env', f'missing: {missing}'))
        for mod, pkg in (('requests', 'requests'), ('numpy', 'numpy'), ('faster_whisper', 'faster-whisper')):
            ok = subprocess.run([sys.executable, '-c', f'import {mod}'], capture_output=True).returncode == 0
            out.append((ok, f'python package {pkg} (judges)', 'pip install -r judge/requirements.txt'))
    return out


def report_checks(res):
    for ok, what, hint in res:
        print(f'  {"ok  " if ok else "FAIL"}  {what}' + ('' if ok else f'\n          -> {hint}'))
    return all(ok for ok, _, _ in res)


def cmd_fetch_data(a):
    sys.exit(fetch_data())


def cmd_doctor(a):
    names = [s for s in (a.systems or '').split(',') if s]
    sys.exit(0 if report_checks(checks(names, judge=not a.no_judge)) else 1)


# ---------------------------------------------------------------- gateways

class Gateways:
    """Start the model / TTS gateways for EXP unless they already run; stop the ones started here."""

    def __init__(self, E, port, tts_port):
        self.E, self.port, self.tts_port, self.procs = E, port, tts_port, []

    def __enter__(self):
        try:
            return self._enter()
        except BaseException:            # a gateway failed to start: stop the one already started
            self.__exit__()
            raise

    def _enter(self):
        runs = (self.E / 'runs').resolve()
        runs.mkdir(parents=True, exist_ok=True)
        logs = self.E / 'logs'
        logs.mkdir(exist_ok=True)
        h = health(self.port)
        if h:
            if Path(h.get('runs_dir', '')).resolve() != runs:
                sys.exit(f'port {self.port} serves another experiment ({h.get("runs_dir")}); stop it or use --port')
            print(f'model gateway already running on :{self.port} (pid {h.get("pid")})')
        elif listening(self.port):
            sys.exit(f'port {self.port} is taken by something that is not the model gateway; use --port')
        else:
            self._start([*PY, ROOT / 'gateway/model_gateway.py', '--runs-dir', runs, '--systems', conf('systems.json'),
                         '--port', self.port], logs / 'model_gateway.log')
            self._wait(self.port, 'model gateway')
        if listening(self.tts_port):
            print(f'TTS gateway already running on :{self.tts_port}')
        else:
            url = keys_env().get('DASHSCOPE_TTS_URL')
            self._start([*PY, ROOT / 'gateway/tts_gateway.py', '--port', self.tts_port, '--log', logs / 'tts_calls.jsonl',
                         *(['--api-url', url] if url else [])], logs / 'tts_gateway.log')
            self._wait(self.tts_port, 'TTS gateway')
        return self

    def _start(self, cmd, log):
        print('$', ' '.join(str(c) for c in cmd), f'> {log}', flush=True)
        self.procs.append(subprocess.Popen([str(c) for c in cmd], cwd=ROOT, stdout=open(log, 'a'),
                                           stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True))

    def _wait(self, port, name):
        for _ in range(60):
            if listening(port):
                return
            if self.procs[-1].poll() is not None:
                sys.exit(f'{name} exited at start; see {self.E / "logs"}')
            time.sleep(0.5)
        sys.exit(f'{name} did not start listening on :{port}; see {self.E / "logs"}')

    def __exit__(self, *exc):
        for p in self.procs:
            if p.poll() is None:
                p.send_signal(signal.SIGTERM)       # the model gateway finishes in-flight requests first
        for p in self.procs:
            try:
                p.wait(timeout=120)
            except subprocess.TimeoutExpired:
                p.kill()


# ---------------------------------------------------------------- run / judge / report

def prepare(E, names, tasks, port, tts_port):
    if not (E / 'snapshot').exists():
        step(f'freeze tasks into {E}')
        cmd = [*PY, ROOT / 'controller/freeze.py', E] + (['--only', tasks] if tasks else [])
        if sh(cmd):
            sys.exit('freeze failed')
    elif tasks:
        sel = {t['task_id'] for t in read(E / 'selection.json')['tasks']}
        if sel != set(tasks.split(',')):
            sys.exit(f'{E} is already frozen with {len(sel)} tasks; use a new experiment directory for another task set')
    x = read(E / 'experiment.json') if (E / 'experiment.json').exists() else {}
    x['systems'] = list(dict.fromkeys([*x.get('systems', []), *names]))
    x['run_case_args'] = ['--port', str(port), '--tts-port', str(tts_port)]
    save(E / 'experiment.json', x)


def judge(E, names):
    runs = E / 'runs'
    jr = E / 'judge_results'
    rc = 0
    step('L1 program checks')
    rc |= sh([*PY, ROOT / 'judge/l1_judge.py', 'run', '--runs', runs, '--out', E / 'l1_results'])
    step('rubric judges, phase 1 (full-modality judges)')
    rc |= sh([*PY, ROOT / 'judge/judge_v4.py', 'run', '--runs', runs, '--out', jr])
    step('rubric judges, phase 2 (omni judge group)')
    rc |= sh([*PY, ROOT / 'judge/judge_v4_p2.py', 'run', '--runs', runs, '--out', jr])
    step('grounded quiz (education tasks)')
    rc |= sh([*PY, ROOT / 'judge/quiz_judge.py', 'run', '--runs', runs, '--out', E / 'quiz_results',
              '--judge-results', jr])
    if rc:
        print('\nsome judgments failed (see above); run `python3 vvbench.py judge', E, '` again to retry them')
    return rc


def report(E):
    step('report')
    script = ROOT / 'judge/report.py'
    if not script.exists():
        print('judge/report.py is missing; skipping the report')
        return 1
    rc = sh([*PY, script, '--runs', E / 'runs', '--judge-results', E / 'judge_results', '--l1', E / 'l1_results',
             '--quiz', E / 'quiz_results', '--out', E / 'report'])
    if not rc:
        print(f'\nreport: {E / "report/summary.md"}')
    return rc


def cmd_run(a):
    E = Path(a.experiment).resolve()
    names = [s for s in a.systems.split(',') if s]
    if not have_dataset():
        sys.exit(f'the dataset is not in data/: download it first ({GET_DATASET}), then run the same command again')
    missing = external_missing()
    if missing:
        print(f'{len(missing)} task inputs are not in place:', *missing, sep='\n  ')
        sys.exit('download them first as data/EXTERNAL_INPUTS.md says (or try `python3 vvbench.py fetch-data`), '
                 'then run the same command again')
    step('checks')
    if not report_checks(checks(names, judge=not a.no_judge)):
        sys.exit('fix the items marked FAIL, then run the same command again')
    prepare(E, names, a.tasks, a.port, a.tts_port)
    with Gateways(E, a.port, a.tts_port):
        step(f'run {len(read(E / "selection.json")["tasks"])} tasks x {len(names)} systems '
             f'(logs: {E / "logs"}, progress: {E / "queue.json"})')
        cmd = [*PY, ROOT / 'controller/schedule.py', E, '--only', ','.join(names), '--per-system', a.per_system]
        if sh(cmd):
            sys.exit('schedule.py failed')
    if a.no_judge:
        return
    judge(E, names)
    report(E)


def cmd_judge(a):
    E = Path(a.experiment).resolve()
    if not report_checks([c for c in checks([]) if c[1].startswith(('judge', 'python', 'ff', 'judge/', 'external', 'dataset'))]):
        sys.exit('fix the items marked FAIL first')
    rc = judge(E, None)
    rc |= report(E)
    sys.exit(rc)


def cmd_report(a):
    sys.exit(report(Path(a.experiment).resolve()))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0], epilog=__doc__.split('\n\n', 1)[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config-dir', help='read systems.json, keys.env, controller.json and judge.json from this directory '
                    'instead of the repository (same as VVBENCH_CONFIG_DIR); keeps keys and local paths out of the checkout')
    sub = ap.add_subparsers(dest='cmd', required=True)
    sub.add_parser('init', help='copy the example configs that are missing').set_defaults(f=cmd_init)
    sub.add_parser('fetch-data', help='optional: download the task inputs not shipped with the repository '
                   '(data/EXTERNAL_INPUTS.md)').set_defaults(f=cmd_fetch_data)
    d = sub.add_parser('doctor', help='check that everything a run needs is in place')
    d.add_argument('--systems', help='comma-separated systems of systems.json to check')
    d.add_argument('--no-judge', action='store_true', help='skip the judge checks')
    d.set_defaults(f=cmd_doctor)
    r = sub.add_parser('run', help='run, judge and report an experiment')
    r.add_argument('experiment', help='experiment directory (created on first use)')
    r.add_argument('--systems', required=True, help='comma-separated systems of systems.json')
    r.add_argument('--tasks', help='comma-separated task ids (default: all 239); fixed at the first run of EXP')
    r.add_argument('--per-system', type=int, default=4, help='parallel runs per system (each takes 8 CPUs, 32 GB)')
    r.add_argument('--port', type=int, default=MODEL_PORT, help='model gateway port')
    r.add_argument('--tts-port', type=int, default=TTS_PORT, help='TTS gateway port')
    r.add_argument('--no-judge', action='store_true', help='only run the agents')
    r.set_defaults(f=cmd_run)
    j = sub.add_parser('judge', help='judge the ended runs of an experiment, then report')
    j.add_argument('experiment')
    j.set_defaults(f=cmd_judge)
    p = sub.add_parser('report', help='write EXP/report/ from the judge results')
    p.add_argument('experiment')
    p.set_defaults(f=cmd_report)
    a = ap.parse_args()
    if a.config_dir:
        global CONFIG_DIR
        CONFIG_DIR = Path(a.config_dir).expanduser().resolve()
        os.environ['VVBENCH_CONFIG_DIR'] = str(CONFIG_DIR)     # for the controller, gateways and judges started from here
    a.f(a)


if __name__ == '__main__':
    main()

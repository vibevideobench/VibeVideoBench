#!/usr/bin/env python3
"""Use Codex or Claude Code inside the vibe-l0 sandbox on one system of systems.json, outside a benchmark run (to try a
system, debug a harness, inspect the image): the interactive TUI by default, or the agent's own arguments after --.

  agent_session.py SYSTEM [DIR] [-- AGENT ARGS...]
  agent_session.py gpt56_sol ~/proj                               # Codex TUI, ~/proj mounted at /workspace
  agent_session.py gpt56_sol ~/proj -- exec "run the tests"        # one-shot codex exec
  agent_session.py claude_opus ~/proj -- -p "summarise README"     # one-shot Claude Code
  agent_session.py claude_opus ~/proj --session ID -- --continue   # back into a session's conversation

Same protection as benchmark runs: the API key stays in the host model gateway and the container only gets a session
token. A session is a gateway "case" <sessions>/runs/<system>/<id>/ (deadline --hours ahead, every request logged in
native_api/); <id>/home is the agent's home (its config, conversations, logs), kept for --session.
A sessions gateway is started in the background on --port when nothing listens there (log: <sessions>/gateway.log). It
reads systems.json once: after changing that, stop it (its pid is in /health) and rerun.
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
import time
import urllib.request

from util import read, save

HERE = Path(__file__).resolve().parent
GATEWAY = HERE.parent / 'gateway/model_gateway.py'
PROXY_VARS = ['http_proxy', 'https_proxy', 'HTTP_PROXY', 'HTTPS_PROXY']
DIRECT = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health(port):
    try:
        with DIRECT.open(f'http://127.0.0.1:{port}/health', timeout=3) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def start_gateway(port, runs, systems, log, key_file=None):
    """The model gateway as a background process that outlives this one: its /health once it answers."""
    cmd = [sys.executable, '-B', str(GATEWAY), '--runs-dir', str(runs), '--systems', str(systems), '--port', str(port)]
    cmd += ['--key-file', str(key_file)] if key_file else []
    with open(log, 'a') as out:
        subprocess.Popen(cmd, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
    for _ in range(50):
        time.sleep(0.2)
        if (h := health(port)):
            print(f'started the sessions gateway on 127.0.0.1:{port} (pid {h.get("pid")}, log {log})', file=sys.stderr)
            return h
    sys.exit(f'the sessions gateway did not come up: see {log}')


def ensure_gateway(a, runs):
    h = health(a.port) or start_gateway(a.port, runs, a.systems, a.sessions / 'gateway.log', a.key_file)
    if Path(h['runs_dir']).resolve() != runs:
        sys.exit(f'port {a.port} serves {h["runs_dir"]}, not {runs}: pick another --port')
    if a.system not in h['systems']:
        sys.exit(f'the gateway on {a.port} (pid {h.get("pid")}) predates {a.system}: stop it, then rerun')
    if a.system in (h.get('unavailable') or {}):
        print(f'warning: {a.system}: {h["unavailable"][a.system]}', file=sys.stderr)


def codex_config(cfg, base):
    """~/.codex/config.toml: every codex entry point (TUI, exec, resume) goes to the gateway with the system's model."""
    return f'''model = "{cfg['model']}"
model_provider = "vibe"
model_reasoning_effort = "{cfg['effort']}"
approval_policy = "never"
sandbox_mode = "danger-full-access"
check_for_update_on_startup = false
project_root_markers = []

[model_providers.vibe]
name = "vibe sandbox gateway"
base_url = "{base}/v1"
env_key = "VIBE_PROXY_TOKEN"
wire_api = "responses"
request_max_retries = 4
stream_max_retries = 5
stream_idle_timeout_ms = 300000

[projects."/workspace"]
trust_level = "trusted"
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('system')
    ap.add_argument('dir', nargs='?', default='.', help='mounted at /workspace (default: current directory)')
    cd = Path(os.environ['VVBENCH_CONFIG_DIR']).expanduser() if os.environ.get('VVBENCH_CONFIG_DIR') else None
    ap.add_argument('--systems', type=Path, default=cd / 'systems.json' if cd else HERE.parent / 'systems.json')
    ap.add_argument('--config', type=Path, default=cd / 'controller.json' if cd else HERE / 'config.json')
    ap.add_argument('--sessions', type=Path, default=HERE.parent / 'sessions')
    ap.add_argument('--port', type=int, default=18892)
    ap.add_argument('--key-file', type=Path, help='passed to a newly started gateway (systems without their own key_file)')
    ap.add_argument('--hours', type=float, default=24, help='the session token stops working after this long')
    ap.add_argument('--session', help='reuse this session id (its home: settings, conversations to resume)')
    ap.add_argument('--image', help='sandbox image (default: config.json "image")')
    argv = sys.argv[1:]
    extra = argv[argv.index('--') + 1:] if '--' in argv else []
    a = ap.parse_args(argv[:argv.index('--')] if '--' in argv else argv)
    a.systems, a.sessions = a.systems.resolve(), a.sessions.resolve()
    if not a.config.exists():
        sys.exit(f'{a.config} not found: copy {HERE / "config.example.json"} to it and fill in your paths')
    config = read(a.config)
    if config.get('docker_host'):
        os.environ['DOCKER_HOST'] = config['docker_host']
    image = a.image or config.get('image', 'vibe-l0:0.1')
    cfg = read(a.systems).get(a.system)
    if not cfg or cfg['harness'] not in ('codex', 'claude'):
        sys.exit(f'{a.system}: not a codex or claude system of {a.systems}')
    install = cfg.get('harness_install') or cfg['harness']      # a system may need its own build of the CLI
    harness_dir = (config.get('harness') or {}).get(install)
    if not harness_dir or not Path(harness_dir).is_dir():
        sys.exit(f'system {a.system} uses the {cfg["harness"]} harness, but {a.config} key "harness.{cfg["harness"]}" '
                 f'({harness_dir or "unset"}) is not a directory: install the harness and set its path '
                 f'(layout and tested version: config.example.json)')
    runs = a.sessions / 'runs'
    runs.mkdir(parents=True, exist_ok=True)
    ensure_gateway(a, runs)

    sid = a.session or time.strftime('%m%d-%H%M%S-') + secrets.token_hex(2)
    C = runs / a.system / sid
    if a.session and not C.exists():
        sys.exit(f'no session {C}')
    (C / 'home').mkdir(parents=True, exist_ok=True)
    if not (C / 'auth.json').exists():
        save(C / 'auth.json', {'token': secrets.token_urlsafe(32)})
        (C / 'auth.json').chmod(0o600)
    token = read(C / 'auth.json')['token']
    now = time.time()
    save(C / 'run.json', {'started': now, 'deadline': now + a.hours * 3600, 'interactive_session': True,
                          'system': a.system, 'workspace': str(Path(a.dir).resolve())})
    base = f'http://127.0.0.1:{a.port}/case/{a.system}/{sid}'
    env = {'HOME': '/root', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'VIBE_PROXY_TOKEN': token,
           'no_proxy': '127.0.0.1,localhost', 'NO_PROXY': '127.0.0.1,localhost'}
    env.update({k: os.environ[k] for k in PROXY_VARS if os.environ.get(k)})
    if cfg['harness'] == 'codex':
        (C / 'home/.codex').mkdir(exist_ok=True)
        (C / 'home/.codex/config.toml').write_text(codex_config(cfg, base))
        mount, agent = f'{Path(harness_dir).resolve()}:/opt/native:ro', ['/opt/native/codex']
        if extra[:1] == ['exec']:
            agent += ['exec', '--skip-git-repo-check'] + extra[1:]
        else:
            agent += extra
    else:
        # every Claude Code request (main loop, subagents, background helpers) uses the system's model; the session
        # token goes as a Bearer token (ANTHROPIC_AUTH_TOKEN), which Claude Code does not ask to approve
        env.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_AUTH_TOKEN=token, IS_SANDBOX='1', DISABLE_AUTOUPDATER='1',
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS='1',
                   ANTHROPIC_MODEL=cfg['model'], ANTHROPIC_DEFAULT_OPUS_MODEL=cfg['model'],
                   ANTHROPIC_DEFAULT_SONNET_MODEL=cfg['model'], ANTHROPIC_DEFAULT_HAIKU_MODEL=cfg['model'],
                   ANTHROPIC_SMALL_FAST_MODEL=cfg['model'], CLAUDE_CODE_SUBAGENT_MODEL=cfg['model'])
        if not (C / 'home/.claude.json').exists():
            save(C / 'home/.claude.json', {'hasCompletedOnboarding': True, 'bypassPermissionsModeAccepted': True,
                                           'projects': {'/workspace': {'hasTrustDialogAccepted': True}}})
            save(C / 'home/.claude/settings.json', {'skipDangerousModePermissionPrompt': True})
        mount = f'{Path(harness_dir).resolve()}:/opt/claude:ro'
        agent = ['/opt/claude/claude', '--model', cfg['model'], '--effort', cfg['effort'],
                 '--dangerously-skip-permissions'] + extra
    tty = ['-it'] if sys.stdin.isatty() and sys.stdout.isatty() else ['-i']
    print(f'session {C}', file=sys.stderr)
    cmd = ['docker', 'run', '--rm', *tty, '--pull', 'never', '--network', 'host', '--init', '--shm-size=2g',
           '-w', '/workspace', '-v', f'{Path(a.dir).resolve()}:/workspace', '-v', f'{C / "home"}:/root', '-v', mount,
           *[x for k, v in env.items() for x in ('-e', f'{k}={v}')], image, *agent]
    sys.exit(subprocess.call(cmd))


if __name__ == '__main__':
    main()

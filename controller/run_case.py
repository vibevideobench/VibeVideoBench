#!/usr/bin/env python3
"""Run one benchmark case with a native agent in a fresh vibe-l0 container (one container per case).

prepare (copy the frozen task package, workspace links, per-case token) -> docker run -d -> docker exec the
agent under a 3600 s wall-clock budget (resume the same native conversation after an API failure) ->
docker kill on timeout -> docker diff (what the model installed) -> docker rm -> collect the delivery
(collect.py) -> terminal.json.

Layout: <experiment>/runs/<system>/<case>/. The model gateway (gateway/model_gateway.py --runs-dir
<experiment>/runs) must be serving on --port, the TTS gateway (gateway/tts_gateway.py) on --tts-port.
Host paths (harness installs, docker host, ffmpeg) come from controller/config.json (see config.example.json).
"""
import argparse
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
import collect
from util import read, save, sha, toml

HERE = Path(__file__).resolve().parent
HOST = socket.gethostname()
BEAT_S = 30        # the controller touches <run>/heartbeat this often while it lives (setup, agent and teardown)
# where each harness's host install is mounted read-only in the container (config.json "harness" gives the host dirs)
MOUNTS = {'codex': '/opt/native', 'antigravity': '/opt/agy', 'claude': '/opt/claude', 'dsh': '/opt/dsh-runtime',
          'kimi': '/opt/kimi', 'zcode': '/opt/zcode'}
FIRST_PROMPT = ('Read ENVIRONMENT.md and prompt.txt. Complete only that task autonomously using your native '
                'harness before the deadline in /run_budget.json.')
RESUME_PROMPT = ('Continue the same task in this native conversation. The same files and requirements apply. '
                 '/run_budget.json contains the original deadline; no additional task time was granted.')
NO_SUBAGENTS = ' Do not spawn subagents.'   # dsh only


def dsh_patch(model, effort, base_url):
    """dsh headless profile overlay, pointed at the benchmark gateway.

    maxTokens 131072: dsh ends the turn and exits 1 when one step hits the cap (one step of >32k reasoning ended runs
    within minutes at 32768); runs that never reach the cap are unaffected.

    effort "off" disables thinking: pi-ai's deepseek format then sends thinking={"type": "disabled"} and no
    reasoning_effort (the upstream's minimal/low still reason; "off" is not a valid reasoning_effort). dsh requires a
    model to offer some level beyond "off", hence the extra high entry, which this profile never selects."""
    disabled = ['llm-deepseek', 'llm-deepseek-account', 'deepseek-account', 'session-title-llm', 'session-log-deepseek',
                'plugin-package-inventory-deepseek', 'web-search-deepseek', 'web-deepseek-api']
    return f'''- id: agent-default-model
  config:
    provider: vibe
    model: {model}
    reasoningEffort: {effort}
- id: llm-pi-ai
  config:
    providers:
      vibe:
        displayName: benchmark gateway
        api: openai-completions
        apiKeyEnv: VIBE_PROXY_TOKEN
        baseURL: {base_url}
        reasoning: {effort}
        streamIdleTimeoutMs: 300000
        retryPolicy:
          mode: normal
          maxRetries: 2
        compat:
          thinkingFormat: deepseek
          supportsReasoningEffort: true
          requiresReasoningContentOnAssistantMessages: true
          supportsUsageInStreaming: true
          supportsStore: false
          supportsDeveloperRole: false
          maxTokensField: max_completion_tokens
        models:
          - id: {model}
            name: {model} via benchmark gateway
            contextWindow: 1000000
            maxTokens: 131072
            input: [text, image]
            reasoningEfforts:
''' + ''.join(f'              {x}: {x}\n' for x in ([effort] if effort != 'off' else ['off', 'high'])) + ''.join(f'- id: {x}\n  disabled: true\n' for x in disabled)


def kimi_config(model, effort, base_url):
    """Kimi Code config.toml: one OpenAI-compatible provider (the gateway) and the model at the system's effort."""
    return f'''default_model = "vibe-model"
default_permission_mode = "auto"

[providers.vibe]
type = "openai"
base_url = "{base_url}"
api_key_env = "VIBE_PROXY_TOKEN"

[models.vibe-model]
provider = "vibe"
model = "{model}"
max_context_size = 262144
capabilities = ["thinking", "image_in", "tool_use"]
support_efforts = ["low", "high", "max"]
default_effort = "{effort}"

[thinking]
enabled = true
effort = "{effort}"
'''


def zcode_provider_config(model, effort, base_url, token):
    """ZCode personal provider file: the gateway as an OpenAI Chat Completions provider, the model as a manual rule
    (reasoning_effort low/high/max, as ZCode's own GLM rule) selected at the system's effort."""
    rule = {'enabled': True,
            'properties': {'contextWindow': 1000000, 'supportsJsonSchemaOutput': True, 'supportsNativeWebSearch': False,
                           'supportsMidConversationSystem': False,
                           'inputFormat': {'supportsImage': True, 'supportsVideo': False, 'supportsPdf': False}},
            'optionSpecs': {'reasoningLevel': {'values': ['low', 'high', 'max'], 'map': '{"reasoning_effort": reasoningLevel}'},
                            'maxOutputTokens': {'max': 131072}}}
    return {'schemaVersion': 1, 'config': {
        'providerOrder': ['vibe'],
        'providerConfigRules': {'providerRules': [{
            'providerId': 'vibe', 'providerName': 'vibe', 'enabled': True,
            'config': {'group': 'standard-personal', 'access': {'type': 'api-key', 'apiKey': token},
                       'api': {'type': 'openai-chat-completions', 'baseUrl': base_url}, 'personalModelIds': [model]}}]},
        'modelConfigRules': {'providerModelRules': [],
                             'manualProviderModelRules': [{'providerId': 'vibe', 'modelId': model, 'config': rule}]},
        'defaultModelSelection': {'providerId': 'vibe', 'modelId': model, 'options': {'reasoningLevel': effort}}}}


def heartbeat(C):
    def beat():
        while True:
            try:
                (C / 'heartbeat').touch()
            except OSError:     # moved aside after terminal.json, just before this process exits
                pass
            time.sleep(BEAT_S)
    threading.Thread(target=beat, daemon=True).start()


def docker(*args, **kw):
    return subprocess.run(['docker', *args], capture_output=True, text=True, **kw)


CONTROL_PROMPT = ('Engineering connectivity control only. Read ENVIRONMENT.md. Run pwd and python3 -c "print(12345)" '
                  'with your native terminal. Finish with READY and the observed shell output. Do not make a video '
                  'or read benchmark tasks.')


def prepare(C, system, cfg, case, port, snapshot, guide, control=False):
    W, I = C / 'workspace', C / 'inputs'
    W.mkdir(parents=True)
    (C / 'home').mkdir()
    if cfg['harness'] == 'antigravity':
        save(C / 'home/.gemini/antigravity-cli/settings.json', {'modelProvider': 'gemini'})
    if cfg['harness'] == 'dsh':
        overlay = C / 'home/.dsh/profiles/headless/cordis.patch.yml'
        overlay.parent.mkdir(parents=True)
        overlay.write_text(dsh_patch(cfg['model'], cfg['effort'], f'http://127.0.0.1:{port}/case/{system}/{case}/v1'))
    if control:
        I.mkdir()
        (I / 'prompt.txt').write_text(CONTROL_PROMPT)
        save(I / 'task.json', {'task_id': case, 'control': True})
    else:
        shutil.copytree(snapshot / 'model_inputs' / case, I)
    for name in ['prompt.txt', 'task.json', 'inputs']:
        if (I / name).exists():
            (W / name).symlink_to('/inputs/' + name)
    shutil.copy(guide, W / 'ENVIRONMENT.md')
    agents = ('Read ENVIRONMENT.md and prompt.txt. Complete the task using your native harness '
              'and this task-scoped environment.' + (NO_SUBAGENTS if cfg['harness'] == 'dsh' else '') + '\n')
    (W / 'AGENTS.md').write_text(agents)
    if cfg['harness'] == 'claude':          # Claude Code reads CLAUDE.md, not AGENTS.md
        (W / 'CLAUDE.md').write_text(agents)
    save(C / 'auth.json', {'token': secrets.token_urlsafe(32)})
    (C / 'auth.json').chmod(0o600)
    base_url = f'http://127.0.0.1:{port}/case/{system}/{case}/v1'
    if cfg['harness'] == 'kimi':
        (C / 'home/.kimi-code').mkdir(parents=True)
        (C / 'home/.kimi-code/config.toml').write_text(kimi_config(cfg['model'], cfg['effort'], base_url))
    if cfg['harness'] == 'zcode':        # ZCode reads the key from its provider file, not from the environment
        save(C / 'home/.zcode/v2/provider_config.json',
             zcode_provider_config(cfg['model'], cfg['effort'], base_url, read(C / 'auth.json')['token']))
        save(C / 'home/.zcode/cli/config.json', {'modelStream': {'idleTimeoutMs': 600000}})
    save(C / 'prepared.json', {'system': system, 'case_id': case, 'model': cfg, 'sandbox': 'docker vibe-l0',
                               'input_hashes': {str(p.relative_to(I)): sha(p) for p in I.rglob('*') if p.is_file()},
                               'environment_guide_sha256': sha(W / 'ENVIRONMENT.md')})


def write_budget(C, row):
    # /run_budget.json is a single-file bind mount: rewrite in place so the container sees updates
    # (an atomic rename would leave the container on the old inode).
    with (C / 'deadline.json').open('w') as f:
        json.dump({'started': row['started'], 'deadline': row['deadline'], 'case_wall_seconds': row['budget_seconds'],
                   'excluded_api_outage_pause_seconds': round(row.get('excluded_pause_s', 0), 3),
                   **({'added_for_failed_api_requests_seconds': round(row['api_failure_credit_s'], 3)}
                      if row.get('api_failure_credit_cap') else {}),
                   **({'added_for_benchmark_gateway_queue_seconds': round(row['queue_credit_s'], 3)}
                      if row.get('queue_credit') else {}),
                   'policy': 'API waits and ordinary retries count. Only recorded provider-outage pauses between '
                             'native segments are excluded' + (', and time blocked on failed API requests is added '
                             'back to the deadline' if row.get('api_failure_credit_cap') else '') +
                             (', and time requests waited in the benchmark gateway\'s own rate-limit queue is added back'
                              if row.get('queue_credit') else '') + '.'}, f, indent=2)


def queue_seconds(C):
    """Time requests waited in our gateway's rate-limit queue before being sent upstream (record.json queued_s)."""
    total = 0.
    for p in (C / 'native_api').glob('*/record.json'):
        try:
            total += read(p).get('queued_s') or 0
        except ValueError:      # being written
            pass
    return total


def native_command(system, cfg, case, port, prompt, session=None):
    if cfg['harness'] == 'claude':
        return (['/opt/claude/claude', '-p', prompt, '--output-format', 'stream-json', '--verbose',
                 '--model', cfg['model'], '--effort', cfg['effort'], '--dangerously-skip-permissions'] +
                (['--resume', session] if session else []))
    if cfg['harness'] == 'dsh':
        return (['/opt/node/bin/node', '/opt/dsh-runtime/node_modules/@deepseek-ai/dsh/lib/bin.js',
                 '--profile', 'headless', '--json'] + (['--session-id', session] if session else []) +
                [prompt + NO_SUBAGENTS])
    if cfg['harness'] == 'kimi':
        return (['/opt/node/bin/node', '/opt/kimi/node_modules/@moonshot-ai/kimi-code/dist/main.mjs',   # -p runs unattended
                 '--output-format', 'stream-json'] + (['-S', session] if session else []) + ['-p', prompt])
    if cfg['harness'] == 'zcode':
        return (['/opt/node/bin/node', '/opt/zcode/zcode.cjs', '--prompt', prompt, '--cwd', '/workspace', '--mode', 'yolo',
                 '--disallowed-tools', 'AskUserQuestion EnterPlanMode ExitPlanMode', '--output-format', 'stream-json',
                 '--no-color'] + (['--resume', session] if session else []))
    if cfg['harness'] == 'antigravity':
        return (['/opt/agy/agy'] + (['--conversation', session] if session else []) +
                ['--model', 'gemini-api://' + cfg['model'], '--effort', cfg['effort'], '--dangerously-skip-permissions',
                 '--print', prompt, '--output-format', 'stream-json', '--print-timeout', '0'])   # our deadline (incl. credits) stops it, not agy's own 60m
    options = {'model_provider': 'vibe', 'model_reasoning_effort': cfg['effort'], 'model_reasoning_summary': 'none',
               'project_root_markers': [],
               'model_providers.vibe': {'name': 'Benchmark logged gateway',
                                        'base_url': f'http://127.0.0.1:{port}/case/{system}/{case}/v1',
                                        'env_key': 'VIBE_PROXY_TOKEN', 'wire_api': 'responses',
                                        'request_max_retries': 2, 'stream_max_retries': 2,
                                        'stream_idle_timeout_ms': 300000}}
    command = ['/opt/native/codex', 'exec'] + (['resume'] if session else [])
    command += ['--skip-git-repo-check', '--dangerously-bypass-approvals-and-sandbox', '--json', '--model', cfg['model']]
    for key, value in options.items():
        command += ['-c', key + '=' + toml(value)]
    return command + ([session, prompt] if session else [prompt])


def exec_env(system, cfg, case, token, port, tts_port, threads='8'):
    base = f'http://127.0.0.1:{port}/case/{system}/{case}'
    env = {'HOME': '/root', 'LANG': 'C.UTF-8', 'LC_ALL': 'C.UTF-8', 'TERM': 'dumb', 'PYTHONNOUSERSITE': '1',
           # GNU nproc reports OMP_NUM_THREADS, so this is also the core count the agent sees (--threads)
           'OMP_NUM_THREADS': threads, 'OPENBLAS_NUM_THREADS': threads, 'MKL_NUM_THREADS': threads,
           'XDG_CACHE_HOME': '/workspace/.cache', 'XDG_DATA_HOME': '/root/.local/share',
           'VIBE_PROXY_TOKEN': token, 'QWEN3_TTS_URL': f'http://127.0.0.1:{tts_port}/case/{system}/{case}/tts'}
    if cfg['harness'] == 'antigravity':
        env.update(GEMINI_API_KEY=token, GOOGLE_GEMINI_BASE_URL=base)
    if cfg['harness'] == 'claude':
        # every request (main loop, subagents, background helpers) goes to the selected model through the gateway
        env.update(ANTHROPIC_BASE_URL=base, ANTHROPIC_API_KEY=token, IS_SANDBOX='1', DISABLE_AUTOUPDATER='1',
                   CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC='1', ANTHROPIC_MODEL=cfg['model'],
                   ANTHROPIC_DEFAULT_OPUS_MODEL=cfg['model'], ANTHROPIC_DEFAULT_SONNET_MODEL=cfg['model'],
                   ANTHROPIC_DEFAULT_HAIKU_MODEL=cfg['model'], ANTHROPIC_SMALL_FAST_MODEL=cfg['model'],
                   CLAUDE_CODE_SUBAGENT_MODEL=cfg['model'],
                   # some Anthropic-compatible upstreams reject Claude Code's experimental anthropic-beta values
                   CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS='1')
    if cfg['harness'] == 'dsh':
        env.update(DSH_HOME='/root/.dsh', DSH_PERMISSION_MODE='danger-full-access', DSH_TELEMETRY_DISABLED='1',
                   NODE_OPTIONS='--max-old-space-size=4096')
    if cfg['harness'] == 'kimi':
        env.update(KIMI_CODE_HOME='/root/.kimi-code', KIMI_MODEL_THINKING_EFFORT=cfg['effort'], KIMI_DISABLE_TELEMETRY='1',
                   KIMI_CODE_NO_AUTO_UPDATE='1', KIMI_CLI_NO_AUTO_UPDATE='1', NODE_OPTIONS='--max-old-space-size=4096')
    if cfg['harness'] == 'zcode':
        # ZCode's model client ignores NO_PROXY (only ZCODE_NO_PROXY / network.noProxy): without this it sends the gateway
        # requests to docker's injected HTTP proxy
        env.update(USERPROFILE='/root', ZCODE_DATA_BASE_DIR='/root', ZCODE_NO_PROXY='127.0.0.1,localhost',
                   NODE_USE_ENV_PROXY='0', NODE_OPTIONS='--max-old-space-size=4096')
    return [x for k, v in env.items() for x in ('-e', f'{k}={v}')]


def events(C):
    """Completed native turns and the native conversation id seen so far."""
    turns, ids = 0, []
    p = C / 'rollout.jsonl'
    for line in (p.read_text().splitlines() if p.exists() else []):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        turns += e.get('type') == 'turn.completed' or (e.get('event') == 'result' and
                                                       e.get('result', {}).get('status') == 'SUCCESS')
        if e.get('type') == 'thread.started' and e.get('thread_id'):
            ids.append(e['thread_id'])
        # Claude Code stream-json: system/init carries the session, a non-error result closes the turn
        turns += e.get('type') == 'result' and e.get('subtype') == 'success' and not e.get('is_error')
        if e.get('type') == 'system' and e.get('subtype') == 'init' and e.get('session_id'):
            ids.append(e['session_id'])
        # dsh --json: a session event, and status turn_end with reason completed
        turns += (e.get('type') == 'status' and e.get('phase') == 'turn_end' and
                  (e.get('reason') or {}).get('kind') == 'completed')
        if e.get('type') == 'session' and e.get('sessionId'):
            ids.append(e['sessionId'])
        for key in ['step_update', 'result']:
            value = e.get(key, {})
            if isinstance(value, dict) and value.get('conversation_id'):
                ids.append(value['conversation_id'])
        # Kimi Code stream-json: session.resume_hint closes a successful turn and names the session
        if e.get('role') == 'meta' and e.get('type') == 'session.resume_hint' and e.get('session_id'):
            turns += 1
            ids.append(e['session_id'])
        # ZCode stream-json: every event carries sessionId (sess_...); turn.completed is counted above
        if isinstance(e.get('sessionId'), str) and e['sessionId'].startswith('sess_'):
            ids.append(e['sessionId'])
    if not ids:     # Kimi Code names its session only after a successful turn; a failed first turn left it on disk
        ids = [p.name for p in (C / 'home/.kimi-code/sessions').glob('wd_*/session_*') if p.is_dir()]
    return turns, (ids[0] if len(set(ids)) == 1 else None)


def client_error(r):
    """A 4xx that resending the same request would get again (not 408/429, nor an upstream timeout mislabelled as 400:
    'ValueError: BackendConnectionTimeout: The request timed out.')."""
    status = r.get('http_status') or 0
    return 400 <= status < 500 and status not in (408, 429) and 'BackendConnectionTimeout' not in str(r.get('error') or '')


def recent_api_failure(C, since):
    """Did the native segment end on a provider failure worth pausing for (5xx, 408/429, timeout, dropped stream)?
    Other 4xx (e.g. 413: the conversation outgrew the upstream's request size limit) and exhausted credit recur on
    every resume of the same conversation, so they end the run instead of looping through outage pauses."""
    calls = sorted((C / 'native_api').glob('*/request.json'))
    if not calls:
        return False
    if not (calls[-1].parent / 'record.json').exists():
        # Still in flight at the gateway: the native client gave up on a silent upstream first (dsh: 300 s idle timeout).
        return calls[-1].stat().st_mtime >= since
    last = read(calls[-1].parent / 'record.json')
    error = str(last.get('error') or '') + str(last.get('model_error') or '')
    # agy waits up to 30 min for the model's background tasks (a render, a server) before it exits on an API error
    window = 1900 if read(C / 'run.json').get('harness') == 'antigravity' else 45
    return (last.get('started', 0) >= since and not last.get('api_complete') and not last.get('client_disconnected')
            and time.time() - last.get('finished', 0) < window
            and not client_error(last) and not last.get('credit_exhausted')
            and not any(x in error for x in ['credit balance exhausted', 'insufficient_quota']))


def failure_end(C):
    """When the last request failed (None while it has no record): the agent was blocked from then on."""
    calls = sorted((C / 'native_api').glob('*/request.json'))
    p = calls[-1].parent / 'record.json' if calls else None
    return read(p).get('finished') if p and p.exists() else None


def union(intervals):
    out = []
    for a, b in sorted(x for x in intervals if x[1] > x[0]):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def failed_waits(C, row, records):
    """(start, end, finished) of each failed API request (5xx, 429, timeouts, dropped streams) the agent was blocked
    on: from its start until it ended or the agent sent its next request. Requests cut by the deadline are left out."""
    starts = sorted(p.stat().st_mtime for p in (C / 'native_api').glob('*/request.json'))
    out = []
    for r in records:
        if r.get('api_complete') or r.get('client_disconnected') or not r.get('finished') \
                or 'deadline' in str(r.get('error', '')) or r['finished'] >= row.get('deadline', float('inf')) - 1 \
                or client_error(r):     # client errors (e.g. an upstream without count_tokens)
            continue
        nxt = next((t for t in starts if t > r['started'] + 0.5), r['finished'])
        out.append((r['started'] + (r.get('queued_s') or 0), min(r['finished'], nxt), r['finished']))   # queue time is credited separately
    return out


def api_failure_seconds(C, row):
    """Time the agent was blocked on failed API requests (failed_waits), less recorded outage pauses."""
    records = [read(p) for p in (C / 'native_api').glob('*/record.json')]
    blocked = union([(a, b) for a, b, _ in failed_waits(C, row, records)])
    pauses = [(x['started'], x['finished']) for x in row.get('cooldown_intervals', [])]
    return sum(b - a for a, b in blocked) - sum(max(0, min(b, y) - max(a, x)) for a, b in blocked for x, y in pauses)


def usage(C):
    result = {'requests': 0, 'with_usage': 0, 'errors': 0, 'input_tokens': 0, 'cached_input_tokens': 0,
              'output_tokens_including_reasoning': 0, 'reasoning_tokens': 0, 'actual_cost': None,
              'interrupted_observations': 0, 'local_cooldown_rejections': 0}
    deadline = read(C / 'run.json').get('deadline', float('inf'))
    for p in sorted((C / 'native_api').glob('*/record.json')):
        r = read(p)
        result['requests'] += 1
        if not r.get('api_complete'):
            # cut off by the case deadline (the read timeout is bounded by it, so the error may just say "timed out")
            if r.get('client_disconnected') or 'deadline' in str(r.get('error', '')) or r.get('finished', 0) >= deadline - 1:
                result['interrupted_observations'] += 1
            else:
                result['errors'] += 1
        u = r.get('raw_usage') or {}
        if 'cache_read_input_tokens' in u or 'cache_creation_input_tokens' in u:     # Anthropic Messages
            values = [u.get('input_tokens', 0) + (u.get('cache_creation_input_tokens') or 0) +
                      (u.get('cache_read_input_tokens') or 0), u.get('cache_read_input_tokens', 0),
                      u.get('output_tokens', 0), (u.get('output_tokens_details') or {}).get('thinking_tokens', 0)]
        elif 'prompt_tokens' in u:                                                  # OpenAI Chat Completions
            values = [u.get('prompt_tokens', 0), (u.get('prompt_tokens_details') or {}).get('cached_tokens') or
                      u.get('prompt_cache_hit_tokens', 0), u.get('completion_tokens', 0),
                      (u.get('completion_tokens_details') or {}).get('reasoning_tokens', 0)]
        elif 'input_tokens' in u:
            values = [u.get('input_tokens', 0), (u.get('input_tokens_details') or {}).get('cached_tokens', 0),
                      u.get('output_tokens', 0), (u.get('output_tokens_details') or {}).get('reasoning_tokens', 0)]
        elif 'promptTokenCount' in u or 'candidatesTokenCount' in u:
            values = [u.get('promptTokenCount', 0), u.get('cachedContentTokenCount', 0),
                      u.get('candidatesTokenCount', 0) + u.get('thoughtsTokenCount', 0), u.get('thoughtsTokenCount', 0)]
        else:
            continue
        result['with_usage'] += 1
        for key, value in zip(['input_tokens', 'cached_input_tokens', 'output_tokens_including_reasoning',
                               'reasoning_tokens'], values):
            result[key] += value or 0
    return result


def load_config(path):
    """controller/config.json: host paths. Sets DOCKER_HOST (if given) and puts ffmpeg_dir first on PATH (the delivery
    checker runs ffmpeg/ffprobe on the host)."""
    if not path.exists():
        raise SystemExit(f'{path} not found: copy {HERE / "config.example.json"} to it and fill in your paths')
    config = read(path)
    if config.get('docker_host'):
        os.environ['DOCKER_HOST'] = config['docker_host']
    if config.get('ffmpeg_dir'):
        os.environ['PATH'] = config['ffmpeg_dir'] + os.pathsep + os.environ.get('PATH', '')
    return config


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--experiment', type=Path, required=True)
    ap.add_argument('--system', required=True)
    ap.add_argument('--case', required=True)
    cd = Path(os.environ['VVBENCH_CONFIG_DIR']).expanduser() if os.environ.get('VVBENCH_CONFIG_DIR') else None
    ap.add_argument('--systems', type=Path, default=cd / 'systems.json' if cd else HERE.parent / 'systems.json')
    ap.add_argument('--config', type=Path, default=cd / 'controller.json' if cd else HERE / 'config.json')
    ap.add_argument('--port', type=int, default=18885, help='model gateway port')
    ap.add_argument('--tts-port', type=int, default=18883, help='TTS gateway port')
    ap.add_argument('--budget', type=float, default=3600)
    ap.add_argument('--image', help='sandbox image (default: config.json "image")')
    ap.add_argument('--cpus', default='8')
    ap.add_argument('--memory', default='32g')
    ap.add_argument('--threads', default='8', help='OMP/OPENBLAS/MKL_NUM_THREADS in the container, which GNU nproc '
                    'reports as the core count (keep it equal to --cpus, as the guide states)')
    ap.add_argument('--max-segments', type=int, default=40)
    ap.add_argument('--max-pause', type=float, default=3 * 3600, help='give up after this much total outage pause')
    ap.add_argument('--control', action='store_true', help='connectivity control instead of the task (case id must start with _)')
    ap.add_argument('--snapshot', type=Path, help='frozen snapshot (model_inputs/, evaluation/; default <experiment>/snapshot)')
    ap.add_argument('--guide', type=Path, default=HERE / 'guide_sandbox.md', help='the workspace ENVIRONMENT.md')
    ap.add_argument('--api-failure-credit', type=float, default=1800,
                    help='add time blocked on failed API requests back to the deadline, up to this many seconds (0: off)')
    ap.add_argument('--queue-credit', action=argparse.BooleanOptionalAction, default=True,
                    help="add time requests waited in the gateway's rate-limit queue (record.json queued_s) back to the deadline")
    a = ap.parse_args()
    config = load_config(a.config)
    a.image = a.image or config.get('image', 'vibe-l0:0.1')
    a.experiment = a.experiment.resolve()     # docker -v needs absolute host paths
    a.snapshot = (a.snapshot or a.experiment / 'snapshot').resolve()
    cfg = read(a.systems)[a.system]
    install = cfg.get('harness_install') or cfg['harness']      # a system may need its own build of the CLI
    harness_dir = (config.get('harness') or {}).get(install)
    if not harness_dir or not Path(harness_dir).is_dir():
        raise SystemExit(f'system {a.system} uses the {cfg["harness"]} harness, but {a.config} key '
                         f'"harness.{install}" ({harness_dir or "unset"}) is not a directory: install the harness '
                         f'and set its path (layout and tested version: config.example.json)')
    C = a.experiment / 'runs' / a.system / a.case
    if (C / 'terminal.json').exists():
        print(f'{C} already finished')
        return
    assert not C.exists(), f'{C} exists without terminal.json; inspect or remove it before rerunning'
    assert a.control == a.case.startswith('_'), 'control case ids start with _'
    image_id = docker('image', 'inspect', a.image, '--format', '{{.Id}}').stdout.strip()
    assert image_id, f'image {a.image} not found at DOCKER_HOST={os.environ.get("DOCKER_HOST", "(default)")}: build it with docker/build.sh'
    prepare(C, a.system, cfg, a.case, a.port, a.snapshot, a.guide.resolve(), a.control)
    token = read(C / 'auth.json')['token']
    name = f'vibe-{a.experiment.name}-{a.system}-{a.case}'.lower().replace('_', '-')
    started = time.time()
    row = {'system': a.system, 'case_id': a.case, 'model': cfg['model'], 'effort': cfg['effort'],
           'harness': cfg['harness'], 'gateway_port': a.port,
           'sandbox': {'kind': 'docker', 'image': a.image, 'image_id': image_id, 'container': name, 'network': 'host',
                       'cpus': a.cpus, 'memory': a.memory, 'threads': a.threads},
           'started': started, 'deadline': started + a.budget, 'budget_seconds': a.budget, 'segments': 0,
           'excluded_pause_s': 0., 'cooldown_intervals': [], 'status': 'running', 'controller_pid': os.getpid(), 'host': HOST,
           'snapshot': str(a.snapshot), 'api_failure_credit_cap': a.api_failure_credit, 'api_failure_credit_s': 0.,
           'queue_credit': a.queue_credit, 'queue_credit_s': 0.}
    save(C / 'run.json', row)
    heartbeat(C)
    write_budget(C, row)
    docker('rm', '-f', name)
    up = docker('run', '-d', '--pull', 'never', '--name', name, '--network', 'host', '--init', '--shm-size=2g',
                '--cpus', a.cpus, '--memory', a.memory, '-w', '/workspace',
                '-v', f'{C / "workspace"}:/workspace', '-v', f'{C / "inputs"}:/inputs:ro', '-v', f'{C / "home"}:/root',
                '-v', f'{C / "deadline.json"}:/run_budget.json:ro',
                '-v', f'{Path(harness_dir).resolve()}:{MOUNTS[cfg["harness"]]}:ro',
                a.image, 'sleep', 'infinity')
    if up.returncode:
        row.update(status='sandbox_error', error=up.stderr[-2000:])
        save(C / 'run.json', row)
        save(C / 'terminal.json', row)
        raise SystemExit(up.stderr)
    status = 'native_error'
    while True:
        remaining = row['deadline'] - time.time()
        if remaining <= 0:
            status = 'timed_out'
            break
        turns_before, session = events(C)
        resume = row['segments'] > 0 and session is not None
        native = native_command(a.system, cfg, a.case, a.port, RESUME_PROMPT if resume else FIRST_PROMPT,
                                session if resume else None)
        row['segments'] += 1
        segment = {'started': time.time(), 'native_argv': native, 'resumed_conversation': session if resume else None,
                   'seconds_remaining': remaining}
        with (C / 'rollout.jsonl').open('a') as out, (C / 'stderr.log').open('a') as err:
            child = subprocess.Popen(['docker', 'exec', '-w', '/workspace',
                                      *exec_env(a.system, cfg, a.case, token, a.port, a.tts_port, a.threads), name, *native],
                                     stdout=out, stderr=err, stdin=subprocess.DEVNULL)
            save(C / 'run.json', row)
            while True:
                if a.api_failure_credit:
                    credit = min(a.api_failure_credit, api_failure_seconds(C, row))
                    if credit > row['api_failure_credit_s'] + 1:
                        row['deadline'] += credit - row['api_failure_credit_s']
                        row['api_failure_credit_s'] = credit
                        save(C / 'run.json', row)
                        write_budget(C, row)
                if a.queue_credit:
                    queued = queue_seconds(C)
                    if queued > row['queue_credit_s'] + 1:
                        row['deadline'] += queued - row['queue_credit_s']
                        row['queue_credit_s'] = queued
                        save(C / 'run.json', row)
                        write_budget(C, row)
                try:
                    code = child.wait(timeout=max(0.1, min(15, row['deadline'] - time.time())))
                    timed_out = False
                    break
                except subprocess.TimeoutExpired:
                    if time.time() < row['deadline']:
                        continue
                    row['stall'] = stall(name)  # before the kill: the container's cgroup goes with it
                    docker('kill', name)        # stops the agent itself, not just the docker exec client
                    code = child.wait(timeout=60)
                    timed_out = True
                    break
        turns_after, _ = events(C)
        segment.update(finished=time.time(), exit_code=code, timed_out=timed_out,
                       native_completed_turns=turns_after - turns_before)
        save(C / 'segments' / f"{row['segments']:03}.json", segment)
        if timed_out:
            status = 'timed_out'
            break
        if code == 0 or segment['native_completed_turns']:
            status = 'native_finished'
            break
        if (recent_api_failure(C, segment['started']) and row['segments'] < a.max_segments
                and row['excluded_pause_s'] < a.max_pause):
            # The native agent gave up after its own retries (provider outage, e.g. no_capacity). Pause between
            # segments with backoff, exclude only that recorded pause from the budget, then resume the same
            # conversation. Time inside segments, including retries, still counts (apart
            # from the idle tail after the failure, below).
            outages = len(row['cooldown_intervals'])
            pause = min(600, 30 * 2 ** outages)
            row.update(status='api_outage_pause')
            save(C / 'run.json', row)
            pause_start = time.time()
            time.sleep(pause)
            now = time.time()
            # the stretch between the failure and the native exit (agy: up to 30 min waiting for background tasks)
            # is part of the outage too, as long as it falls inside this segment
            pause_start = max(segment['started'], min(pause_start, failure_end(C) or pause_start))
            row['deadline'] += now - pause_start
            row['excluded_pause_s'] += now - pause_start
            row['cooldown_intervals'].append({'started': pause_start, 'finished': now, 'excluded_seconds': now - pause_start,
                                              'reason': 'native segment ended on provider API failure'})
            row['status'] = 'running'
            save(C / 'run.json', row)
            write_budget(C, row)
            continue
        status = 'native_error'
        break
    finish(C, row, name, status, time.time(), a.budget, a.control)


def stall(name):
    """How long the container's tasks waited for CPU / IO / memory (cgroup v2 PSI totals in seconds: 'some' = at least
    one task waiting, 'full' = all of them), its CPU use and its --cpus throttling, so far. Contention with the other
    runs and with other jobs on the host shows up here; it is recorded, not credited to the budget.
    Read it before the container is killed or removed: its cgroup goes with it."""
    try:
        cid = docker('inspect', '-f', '{{.Id}}', name).stdout.strip()
        # cgroup v2 (vibe.slice: a dockerd with --cgroup-parent vibe.slice; system.slice: a plain systemd docker) or
        # v1 + cgroupfs (e.g. inside a Kubernetes pod, with or without v1 PSI files)
        d = next((x for x in [Path(f'/sys/fs/cgroup/vibe.slice/docker-{cid}.scope'),
                              Path(f'/sys/fs/cgroup/system.slice/docker-{cid}.scope'),
                              Path(f'/sys/fs/cgroup/cpu,cpuacct/vibe.slice/{cid}'),
                              Path(f'/sys/fs/cgroup/cpu,cpuacct/docker/{cid}')] if x.exists()), None)
        if d is None:
            return {'error': f'no cgroup directory found for {cid[:12]}'}
        v1 = 'cpu,cpuacct' in str(d)
        out = {}
        for r in ['cpu', 'io', 'memory']:
            if not (d / f'{r}.pressure').exists():     # kernels without PSI: CPU use and throttling only
                continue
            for line in (d / f'{r}.pressure').read_text().splitlines():
                kind, *kv = line.split()
                out[f'{r}_{kind}_s'] = round(int(dict(x.split('=') for x in kv)['total']) / 1e6, 1)
        st = dict(line.split() for line in (d / 'cpu.stat').read_text().splitlines())
        # v1 throttled_time is summed over the CPUs the group ran on (it can exceed the wall time); nr_throttled /
        # nr_periods (both versions) say how often the --cpus quota was hit
        used, throttled = (int((d / 'cpuacct.usage').read_text()) / 1e9, int(st.get('throttled_time', 0)) / 1e9) if v1 \
            else (int(st['usage_usec']) / 1e6, int(st.get('throttled_usec', 0)) / 1e6)
        out.update(cpu_used_s=round(used, 1), cpu_throttled_s=round(throttled, 1),
                   cpu_periods=int(st.get('nr_periods', 0)), cpu_throttled_periods=int(st.get('nr_throttled', 0)),
                   cgroup=str(d).split('/sys/fs/cgroup/')[1].split('/')[0], read_at=round(time.time(), 1))
        return out
    except Exception as exc:
        return {'error': repr(exc)}


def finish(C, row, name, status, finished, budget, control=False):
    """Record what the agent installed outside the mounts, remove the container, collect and write terminal.json."""
    row['stall'] = row.get('stall') or stall(name)
    mounts = {'/opt', '/usr', '/usr/sbin', '/opt/agy', '/opt/native', '/opt/claude', '/opt/dsh-runtime', '/opt/kimi',
              '/opt/zcode', '/usr/sbin/docker-init'}
    installed = [line for line in docker('diff', name).stdout.splitlines() if line[2:] not in mounts and
                 not line[2:].startswith(('/workspace', '/root', '/inputs', '/run_budget.json', '/tmp', '/dev',
                                          '/proc', '/sys', '/run', '/etc/hostname', '/etc/hosts', '/etc/resolv.conf'))]
    (C / 'container_diff.txt').write_text('\n'.join(installed) + '\n')
    docker('rm', '-f', name)
    row.update(status=status, finished=finished,
               elapsed_s=round(min(budget, finished - row['started'] - row['excluded_pause_s']
                                   - row.get('api_failure_credit_s', 0) - row.get('queue_credit_s', 0)), 3),
               wall_timeout=status == 'timed_out', container_changed_paths=len(installed))
    save(C / 'run.json', row)
    collect_into(C, row, control)
    print(json.dumps({k: row.get(k) for k in ['system', 'case_id', 'status', 'elapsed_s', 'segments', 'video',
                                              'technical_pass']}, ensure_ascii=False), flush=True)


def collect_into(C, row, control=False):
    """Collect the delivery (collect.py) and write terminal.json."""
    try:
        if control:
            row['control_ready'] = 'READY' in (C / 'rollout.jsonl').read_text() and '12345' in (C / 'rollout.jsonl').read_text()
        else:
            delivery = collect.collect(C, row['snapshot'])
            row.update(video=delivery['video'], technical_pass=delivery['technical_pass'],
                       input_hashes_unchanged=delivery['input_hashes_unchanged'])
    except Exception as exc:
        row['collection_error'] = repr(exc)
    row['usage'] = usage(C)
    save(C / 'terminal.json', row)


if __name__ == '__main__':
    main()

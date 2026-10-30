# Agent CLIs (harnesses)

Each system is evaluated through its vendor's own agent CLI, run unmodified inside the sandbox. The CLIs are not
shipped with this repository: install them on the host, then put each install directory in `controller/config.json`
under `harness.<name>`. A directory is mounted read-only into the container at a fixed path; only the CLI of the system
being run is checked, so you only need the ones you evaluate. The CLI runs inside the Ubuntu 24.04 image, so install
the Linux x86_64 build.

The CLI never gets your API key. `run_case.py` points it at the host model gateway
(`http://127.0.0.1:<port>/case/<system>/<case>/…`) with a per-case token; the gateway adds the real key from
`keys.env`. All settings below are applied by `run_case.py` — you do not configure the CLIs yourself.

| harness | `systems.json` `"harness"` | protocol through the gateway | tested version |
|---|---|---|---|
| Codex CLI | `codex` | OpenAI Responses | codex-cli 0.147.0 (gpt6_astra: 0.162.1, see below) |
| Claude Code | `claude` | Anthropic Messages | 2.1.301 |
| Antigravity CLI | `antigravity` | Gemini `streamGenerateContent` | agy 1.4.0 |
| DeepSeek Harness | `dsh` | OpenAI Chat Completions | @deepseek-ai/dsh 0.2.1-rc.2 |
| Kimi Code | `kimi` | OpenAI Chat Completions | @moonshot-ai/kimi-code 2.2.0 |
| ZCode CLI | `zcode` | OpenAI Chat Completions | 0.17.0 (commit 3c5b2276) |

Other versions usually work but may change agent behaviour; report the version with your results.

All harnesses also get `OMP_NUM_THREADS`/`OPENBLAS_NUM_THREADS`/`MKL_NUM_THREADS` (= `--threads`, default 8),
`QWEN3_TTS_URL` (the TTS gateway), `HOME=/root` and a UTF-8 locale. Each run's `home/` directory holds the CLI's
config, logs and conversation, kept for resuming the same conversation after an API outage.

## Codex CLI — `harness.codex`

- Install: the standalone Linux release from [github.com/openai/codex/releases](https://github.com/openai/codex/releases)
  (tested: the x86_64 musl build of 0.147.0).
- Directory: contains `codex` and `codex-code-mode-host`. Mounted at `/opt/native`.
- Invocation: `codex exec --json --dangerously-bypass-approvals-and-sandbox` (the container is the sandbox), with
  `-c` overrides: `model_provider=vibe`, `model_providers.vibe={base_url=<gateway>/v1, env_key=VIBE_PROXY_TOKEN,
  wire_api=responses, …}`, `model_reasoning_effort=<effort>`, `model_reasoning_summary=none`,
  `project_root_markers=[]`. Resumes with `codex exec resume`.

- A second Codex build for `gpt6_astra`: that system's `systems.json` entry has `"harness_install": "codex_astra"`, so it
  is run with `harness.codex_astra` (tested: codex-cli 0.162.1, the build its runs used). Install it into
  its own directory the same way.

## Claude Code — `harness.claude`

- Install: the official native installer (`curl -fsSL https://claude.ai/install.sh | bash`), then use the directory
  holding the installed `claude` binary.
- Directory: contains the native `claude` binary. Mounted at `/opt/claude`.
- Environment: `ANTHROPIC_BASE_URL` (gateway), `ANTHROPIC_API_KEY` (session token); `ANTHROPIC_MODEL`,
  `ANTHROPIC_DEFAULT_{OPUS,SONNET,HAIKU}_MODEL`, `ANTHROPIC_SMALL_FAST_MODEL`, `CLAUDE_CODE_SUBAGENT_MODEL` all set to
  the system's model; `IS_SANDBOX=1`, `DISABLE_AUTOUPDATER=1`, `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1`,
  `CLAUDE_CODE_DISABLE_EXPERIMENTAL_BETAS=1`. A `CLAUDE.md` is written into the workspace.
- Invocation: `claude -p <prompt> --output-format stream-json --verbose --effort <effort>
  --dangerously-skip-permissions`. Resumes with `--resume`.

## Antigravity CLI — `harness.antigravity`

- Install: Google's installer ([antigravity.google/product/antigravity-cli](https://antigravity.google/product/antigravity-cli)),
  into a directory of its own: `curl -fsSL https://antigravity.google/cli/install.sh | bash -s -- --dir <dir>`.
  The installer takes the current release (no version pin); check it with `<dir>/agy --version`.
  No Google sign-in is needed: the gateway serves it in Gemini API-key mode.
- Directory: contains the `agy` binary. Mounted at `/opt/agy`.
- Environment: `GEMINI_API_KEY` (session token), `GOOGLE_GEMINI_BASE_URL` (gateway); writes
  `~/.gemini/antigravity-cli/settings.json` = `{"modelProvider": "gemini"}`.
- Invocation: `agy --model gemini-api://<model> --effort <effort> --dangerously-skip-permissions --print <prompt>
  --output-format stream-json --print-timeout 0`. Resumes with `--conversation`.

## DeepSeek Harness — `harness.dsh`

- Install: `npm install --prefix <dir> @deepseek-ai/dsh@0.2.1-rc.2`
  ([github.com/deepseek-ai/deepseek-harness](https://github.com/deepseek-ai/deepseek-harness), MIT; developer preview,
  so pin the version).
- Directory: the npm prefix (`<dir>/node_modules/@deepseek-ai/dsh`). Mounted at `/opt/dsh-runtime`; runs on the image's
  Node (`/opt/node`).
- Environment: `DSH_HOME=/root/.dsh`, `DSH_PERMISSION_MODE=danger-full-access`, `DSH_TELEMETRY_DISABLED=1`; writes a
  headless profile `~/.dsh/profiles/headless/cordis.patch.yml` (OpenAI-compatible provider at the gateway,
  `maxTokens` 131072, account and web-search modules disabled).
- Invocation: `dsh --profile headless --json`; the prompt asks it not to spawn sub-agents. Resumes with `--session-id`.

## Kimi Code — `harness.kimi`

- Install: `npm install --prefix <dir> @moonshot-ai/kimi-code@2.2.0`
  ([github.com/MoonshotAI/kimi-code](https://github.com/MoonshotAI/kimi-code)).
- Directory: the npm prefix (`<dir>/node_modules/@moonshot-ai/kimi-code`). Mounted at `/opt/kimi`; runs on the image's
  Node 24.
- Environment: `KIMI_CODE_HOME`, `KIMI_MODEL_THINKING_EFFORT`, telemetry and auto-update off; writes
  `~/.kimi-code/config.toml` (OpenAI-compatible provider at the gateway, `api_key_env=VIBE_PROXY_TOKEN`).
- Invocation: `kimi-code -p <prompt> --output-format stream-json`. Resumes with `-S`.

## ZCode CLI — `harness.zcode`

- Install: build the CLI from source, [github.com/zai-org/ZCode](https://github.com/zai-org/ZCode) (Apache-2.0; the
  desktop app from zcode.z.ai does not expose it). Needs Node 24.14 and pnpm 10.33 (see the repo's `mise.toml`).
  Tested at commit `3c5b2276` (2026-10-20):
  ```bash
  git clone https://github.com/zai-org/ZCode && cd ZCode && git checkout 3c5b2276
  pnpm install && pnpm --filter @zcode/cli... build
  # harness.zcode = <ZCode>/apps/zcode-cli/packages/cli/dist   (contains zcode.cjs)
  ```
- Directory: the CLI's `dist/` containing `zcode.cjs`. Mounted at `/opt/zcode`; runs on the image's Node.
- Environment: `ZCODE_DATA_BASE_DIR=/root`, `ZCODE_NO_PROXY=127.0.0.1,localhost`, `NODE_USE_ENV_PROXY=0`; writes
  `~/.zcode/v2/provider_config.json` (gateway + session token) and `~/.zcode/cli/config.json` (idle timeout).
- Invocation: `zcode --mode yolo --output-format stream-json`. Resumes with `--resume`.

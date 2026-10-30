# Installation

Everything runs on one Linux x86_64 host: the sandbox containers, the two host gateways that hold your API keys,
the controller that launches runs, and the judges. Agents never see a provider key — each container gets only a
per-case token for the local model gateway.

```
 container (vibe-l0)                      host
 ┌───────────────────────┐   127.0.0.1   ┌────────────────────────┐      ┌──────────────┐
 │ agent CLI (codex, …)  │ ────────────▶ │ gateway/model_gateway  │ ───▶ │ model API    │
 │ /opt/benchmark-tools/ │ ────────────▶ │ gateway/tts_gateway    │ ───▶ │ DashScope TTS│
 └───────────────────────┘               └────────────────────────┘      └──────────────┘
        ▲ launched by controller/run_case.py (one container per case, via schedule.py)
```

## 1. Host prerequisites

| | version | used by |
|---|---|---|
| Linux x86_64 | — | containers use `--network host` |
| Docker Engine | 24+ with BuildKit and the buildx plugin (`docker buildx version`) | sandbox image, runs |
| Python | 3.10+ | gateways, controller, judges |
| ffmpeg / ffprobe | 6+ on `PATH` | delivery check after each run, L1 checks, judge media |
| Internet access | — | image build, model APIs (an `http(s)_proxy` in the environment is honoured, not required) |

Plan for disk space: the image build downloads Node, Chromium, fonts and Python wheels (`docker/downloads/`), and
each run keeps its workspace and rendered video under `<experiment>/runs/`.

Resources per run (the benchmark's main setting): 8 CPUs, 32 GB memory, a 3600 s wall-clock budget. Size the number
of parallel runs (`schedule.py --per-system`, `--total`) to your machine.

## 2. Python packages

The gateways and the controller use the standard library only. The judges need a few packages:

```bash
python3 -m venv .venv && . .venv/bin/activate
pip install -r judge/requirements.txt     # requests, numpy, faster-whisper (L1 VAD / language ID, quiz transcripts)
pip install -r gateway/requirements.txt   # optional: Pillow
```

faster-whisper downloads its model (`whisper_model` in `judge/config.json`, default `small`) on first use; set it to
a local directory for offline hosts.

## 3. Build the sandbox image

No prebuilt image is published; build it from `docker/`:

```bash
cd docker
./fetch.sh            # downloads into docker/downloads/, checks sha256 (re-run skips finished files)
./build.sh            # docker build → vibe-l0:0.1
./sandbox.sh smoke    # acceptance test (browser, drawing, versions; TTS is skipped without the TTS gateway)
./sandbox.sh doctor   # storage, internet, and whether the gateways are listening
```

Behind a proxy, export `http_proxy`/`https_proxy` for `fetch.sh` and put the proxy in Docker's client config
(`~/.docker/config.json`, `"proxies"`) so the build receives it. `docker/rootfs/opt/ENVIRONMENT.md` lists what the
image contains.

## 4. Get the data

The tasks, rubrics, tags and quizzes are not in this repository: they are the HuggingFace dataset
[`diaomuxi/vibevideobench`](https://huggingface.co/datasets/diaomuxi/vibevideobench). Download it into `data/`, where the pipeline reads it:

```bash
huggingface-cli download diaomuxi/vibevideobench --repo-type dataset --local-dir data
```

62 inputs of 23 tasks are not in the dataset either, because their license or the people shown in them do not allow
redistribution: arXiv papers, Adobe font documentation, press images, photos of identifiable people, excerpts of ESA / NASA videos, web page
snapshots, Pexels stock photos and two brand marks. **Put them in place before you run anything**:
`data/EXTERNAL_INPUTS.md` (it comes with the dataset) lists each one with its path, source link, sha256 and, for the
video excerpts, the ffmpeg command that cuts it. `python3 vvbench.py doctor` lists what is still missing and `run`
refuses to start until everything is there.

`python3 vvbench.py fetch-data` (or `python3 data/fetch_external.py`) can do it for you: it downloads every entry of
`data/external.json` through your `http(s)_proxy`, cuts the excerpts and checks the results. The two source videos
are about 1.5 GB and 0.5 GB.

## 5. Agent CLIs

The agent CLIs are not shipped. Install the ones you want to evaluate on the host and point
`controller/config.json` at them — see [harnesses.md](harnesses.md).

## 6. Configuration

All local config files are gitignored. `python3 vvbench.py init` copies each missing one from its example; then edit it.

| file | from | holds |
|---|---|---|
| `keys.env` | `keys.env.example` | your API keys, one `NAME=value` per line |
| `systems.json` | `systems.example.json` | the systems to evaluate: model, effort, harness, upstream API, which keys |
| `controller/config.json` | `controller/config.example.json` | image tag, agent CLI directories, optional `docker_host` / `ffmpeg_dir` |
| `judge/config.json` | `judge/config.example.json` | judge models and endpoints (`base_url`, `model`, `key_var`) |
| `<experiment>/experiment.json` | `controller/experiment.example.json` | which systems an experiment runs |

`DASHSCOPE_API_KEY` in `keys.env` is used by the TTS gateway (agents may synthesise narration through it) and by the
qwen judge.

## 7. Run

```bash
python3 vvbench.py doctor --systems claude_opus      # every item should say ok
python3 vvbench.py run exp1 --systems claude_opus    # freeze, run all 239 tasks, judge, report
```

`run` creates `exp1/` on first use, starts the two host gateways (and stops them at the end), runs every task with
`controller/schedule.py`, then the four judges, then `judge/report.py`. Run the same command again after an
interruption to continue. `--tasks ID,ID` limits a new experiment to some tasks; `--no-judge` only runs the agents;
`python3 vvbench.py judge exp1` judges what is still missing and rebuilds the report.

The same steps by hand, for debugging one stage:

```bash
python3 controller/freeze.py exp1                                     # data/tasks -> exp1/snapshot, exp1/selection.json
cp controller/experiment.example.json exp1/experiment.json            # edit "systems"
python3 gateway/model_gateway.py --runs-dir exp1/runs --systems systems.json &
python3 gateway/tts_gateway.py &
python3 controller/schedule.py exp1 --per-system 4
python3 controller/run_case.py --experiment exp1 --system claude_opus --case D0130   # or one case alone
python3 judge/l1_judge.py    run --runs exp1/runs --out exp1/l1_results
python3 judge/judge_v4.py    run --runs exp1/runs --out exp1/judge_results
python3 judge/judge_v4_p2.py run --runs exp1/runs --out exp1/judge_results
python3 judge/quiz_judge.py  run --runs exp1/runs --out exp1/quiz_results --judge-results exp1/judge_results
python3 judge/report.py --runs exp1/runs --judge-results exp1/judge_results --l1 exp1/l1_results \
    --quiz exp1/quiz_results --out exp1/report
```

To use an agent interactively in the sandbox (outside a benchmark run): `docker/sandbox.sh agent <system> [DIR]`.

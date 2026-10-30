#!/usr/bin/env bash
# Everyday entry point for the vibe-l0 sandbox image.
#
#   ./sandbox.sh doctor           check Docker storage, internet through the proxy, and the host gateways
#   ./sandbox.sh shell [DIR]      interactive bash, DIR (default: current dir) mounted at /workspace
#   ./sandbox.sh run DIR CMD...   run one command with DIR mounted at /workspace
#   ./sandbox.sh smoke            acceptance test (the TTS check is skipped without a gateway on :18883)
#   ./sandbox.sh agent SYSTEM [DIR] [-- AGENT ARGS]
#                                 Codex / Claude Code on a system of ../systems.json (e.g. gpt56_sol,
#                                 claude_opus); see ../controller/agent_session.py
#
# Containers share the host network (--network host): 127.0.0.1 inside is the host, so the HTTP proxy and
# the model / TTS gateways on the host are reachable, and no bridge, NAT or iptables rule is needed.
# The host's proxy variables are passed through; the image itself holds no proxy address.
# Uses DOCKER_HOST / the default Docker. Build the image first with ./fetch.sh && ./build.sh.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
IMAGE=${VIBE_IMAGE:-vibe-l0:0.1}

ARGS=(--rm --network host --init --shm-size=2g)
for v in http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; do
  if [ -n "${!v:-}" ]; then ARGS+=(-e "$v=${!v}"); fi
done
NOPROXY=127.0.0.1,localhost${no_proxy:+,$no_proxy}
ARGS+=(-e "no_proxy=$NOPROXY" -e "NO_PROXY=$NOPROXY")

need_image() {
  docker image inspect "$IMAGE" >/dev/null 2>&1 || { echo "$IMAGE not found: build it with ./fetch.sh && ./build.sh" >&2; exit 1; }
}

doctor() {
  echo "docker:  ${DOCKER_HOST:-default socket}, server $(docker version --format '{{.Server.Version}}')"
  need_image
  echo "image:   $IMAGE $(docker image inspect "$IMAGE" --format '{{.Id}}')"
  docker run "${ARGS[@]}" "$IMAGE" bash -c '
    # A full Docker storage disk only shows up for non-root writes (root may use the reserved blocks).
    if setpriv --reuid=_apt --regid=nogroup --clear-groups mktemp -d >/dev/null 2>&1; then
      echo "storage: ok, $(df -h / | awk "NR == 2 {print \$4}") free"
    else
      echo "storage: FULL (the disk holding the Docker data-root is full)"
    fi
    code=$(curl -s -o /dev/null -m 20 -w "%{http_code}" https://pypi.org/simple/)
    echo "network: https://pypi.org -> $code (proxy: ${https_proxy:-none})"
    for p in 18885:model-gateway 18883:tts-gateway; do
      if python3 -c "import socket; socket.create_connection((\"127.0.0.1\", ${p%%:*}), 2)" 2>/dev/null; then
        echo "${p#*:}: listening on 127.0.0.1:${p%%:*}"
      else
        echo "${p#*:}: nothing on 127.0.0.1:${p%%:*}"
      fi
    done'
}

case "${1:-}" in
  doctor) doctor ;;
  shell)
    need_image
    docker run -it "${ARGS[@]}" -v "$(realpath "${2:-.}"):/workspace" "$IMAGE" bash ;;
  run)
    [ $# -ge 3 ] || { echo "usage: $0 run DIR CMD..." >&2; exit 2; }
    need_image
    dir=$(realpath "$2"); shift 2
    tty=(); if [ -t 0 ] && [ -t 1 ]; then tty=(-it); fi
    docker run "${tty[@]}" "${ARGS[@]}" -v "$dir:/workspace" "$IMAGE" "$@" ;;
  smoke)
    need_image
    out=${SMOKE_OUT:-$HERE/build/smoke-out}
    mkdir -p "$out" && rm -f "$out"/*
    docker run "${ARGS[@]}" -v "$HERE/smoke:/smoke:ro" -v "$out:/out" "$IMAGE" bash /smoke/run_smoke.sh \
      | tee "$HERE/build/smoke.log" ;;
  agent)
    shift
    need_image
    exec python3 -B "$HERE/../controller/agent_session.py" "$@" ;;
  *) sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'; exit 2 ;;
esac

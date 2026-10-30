#!/usr/bin/env bash
# Build the L0 image (run ./fetch.sh first). Host networking: the build serves Chromium from downloads/
# on 127.0.0.1:18765 and reaches the internet directly or through your HTTP proxy.
set -euo pipefail
cd "$(dirname "$0")"
TAG=${1:-vibe-l0:0.1}
# Uses DOCKER_HOST / the default Docker. Behind a proxy, set it in Docker's client config
# (~/.docker/config.json "proxies") so the build gets it as build args.
[ -f build/downloads.sha256 ] || { echo "run ./fetch.sh first" >&2; exit 1; }
docker build --network host --progress plain -t "$TAG" .
docker image inspect "$TAG" --format 'built {{index .RepoTags 0}} {{.Id}} size={{.Size}}'

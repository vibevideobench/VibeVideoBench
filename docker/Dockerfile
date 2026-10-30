# syntax=docker/dockerfile:1
# L0 image for the vibe-video benchmark: shared base capabilities only, no tech_stack wrappers.
# Build with ./build.sh after ./fetch.sh (host networking: HTTP proxy and the local Chromium mirror
# are on 127.0.0.1).
FROM ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac

ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8 LC_ALL=C.UTF-8 TZ=Asia/Shanghai

# --- apt: bootstrap curl, then everything else with a parallel .deb prefetch ---
# Package lists are kept in the image so agents can apt-get install right away.
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl
COPY build/apt-packages.txt build/apt-install.sh /tmp/build/
RUN bash /tmp/build/apt-install.sh $(grep -v '^#' /tmp/build/apt-packages.txt)

# --- fonts not packaged by Ubuntu + fontconfig defaults (Noto CJK SC first) ---
COPY downloads/fonts/ /usr/share/fonts/truetype/vibe-extra/
COPY rootfs/etc/fonts/local.conf /etc/fonts/local.conf
RUN fc-cache -f

# --- Python 3.12 venv; also sees apt-installed python3-* packages ---
COPY build/requirements.lock /tmp/build/
RUN --mount=type=bind,source=downloads/wheels,target=/wheels \
    python3.12 -m venv --system-site-packages /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir --no-index --find-links /wheels --require-hashes -r /tmp/build/requirements.lock
ENV PATH=/opt/venv/bin:$PATH

# --- Chromium for Playwright 1.62.0, served to `playwright install` from downloads/playwright ---
ENV PLAYWRIGHT_BROWSERS_PATH=/opt/ms-playwright
RUN --mount=type=bind,source=downloads/playwright,target=/mirror \
    set -eux; \
    port=$(python3 -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'); \
    python3 -m http.server $port --bind 127.0.0.1 --directory /mirror >/tmp/mirror.log 2>&1 & pid=$!; sleep 1; \
    curl -sSf --noproxy '*' -o /dev/null -r 0-0 http://127.0.0.1:$port/builds/cft/151.0.7922.34/linux64/chrome-linux64.zip; \
    PLAYWRIGHT_DOWNLOAD_HOST=http://127.0.0.1:$port playwright install --with-deps chromium; \
    kill $pid; rm -f /tmp/mirror.log; \
    ln -s /opt/ms-playwright/chromium-1234/chrome-linux64 /opt/chromium; \
    /opt/chromium/chrome --version

# --- Node.js 24 LTS + npm; global playwright matching the browser build ---
RUN --mount=type=bind,source=downloads/node,target=/dl \
    tar -xJf /dl/node-v24.22.0-linux-x64.tar.xz -C /opt && mv /opt/node-v24.22.0-linux-x64 /opt/node
ENV PATH=/opt/node/bin:$PATH NODE_PATH=/opt/node/lib/node_modules \
    npm_config_registry=https://registry.npmjs.org/ npm_config_update_notifier=false npm_config_fund=false
RUN PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=1 npm install -g playwright@1.62.0 && npm cache clean --force

# --- /opt/vendor: full three / gsap packages ---
RUN --mount=type=bind,source=downloads/npm,target=/dl \
    set -eux; mkdir -p /opt/vendor/three /opt/vendor/gsap; \
    tar -xzf /dl/three-0.160.0.tgz -C /opt/vendor/three --strip-components=1; \
    tar -xzf /dl/gsap-3.12.5.tgz -C /opt/vendor/gsap --strip-components=1; \
    cp /opt/vendor/three/build/three.module.js /opt/vendor/gsap/dist/gsap.min.js /opt/vendor/; \
    printf '%s  %s\n' \
      76dea8151bc9352aef3528b4262e249b2604f62543828328db978d060d61a495 /opt/vendor/three.module.js \
      28033e449a31ebcc396e5be8b13b63152bf03094288fb5867034321927bce087 /opt/vendor/gsap.min.js | sha256sum -c -
COPY rootfs/opt/vendor/manifest.json /opt/vendor/manifest.json

# --- TTS client (talks to the host-side gateway; no key in the image) + environment notes ---
COPY rootfs/opt/benchmark-tools/ /opt/benchmark-tools/
COPY rootfs/opt/ENVIRONMENT.md /opt/ENVIRONMENT.md
RUN chmod 755 /opt/benchmark-tools/*.py && rm -rf /tmp/build /root/.cache /root/.npm

ENV PATH=/opt/benchmark-tools:$PATH \
    CHROME_BIN=/opt/chromium/chrome \
    PUPPETEER_SKIP_DOWNLOAD=1 PUPPETEER_EXECUTABLE_PATH=/opt/chromium/chrome \
    PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_BREAK_SYSTEM_PACKAGES=1 \
    QWEN3_TTS_URL=http://127.0.0.1:18883/tts \
    NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost NODE_USE_ENV_PROXY=1
WORKDIR /workspace
LABEL org.opencontainers.image.title="vibe-l0" org.opencontainers.image.version="0.1"

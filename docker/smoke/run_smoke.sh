#!/bin/bash
# Smoke test for the L0 image. Run inside a container with this folder at /smoke and a writable /out:
#   docker run --rm --network host --init --shm-size=2g \
#     -v "$PWD/smoke:/smoke:ro" -v /some/dir:/out vibe-l0:0.1 bash /smoke/run_smoke.sh
set -uo pipefail
cd /out
pass=0; fail=0
check() {
  local name=$1; shift
  if out=$("$@" 2>&1); then
    echo "PASS $name | $(echo "$out" | tail -1)"; pass=$((pass+1))
  else
    echo "FAIL $name"; echo "$out" | tail -15 | sed 's/^/     /'; fail=$((fail+1))
  fi
}

check versions python3 /smoke/versions.py
check ffmpeg bash -c 'ffmpeg -hide_banner -encoders | grep -q libx264 && ffmpeg -hide_banner -filters | grep -q " subtitles " &&
  ffmpeg -hide_banner -filters | grep -q " drawtext " && ffmpeg -version | head -1'
check fontconfig bash -c 'fc-match sans-serif | grep -q "Noto Sans CJK SC" && fc-match serif | grep -q "Noto Serif CJK SC" &&
  fc-match monospace | grep -q "Noto Sans Mono CJK SC" && fc-match Sans | grep -q "Noto Sans CJK SC" && fc-match "sans-serif:lang=zh-cn"'
check fonts bash -c 'for f in "LXGW WenKai" "Smiley Sans" "ZCOOL KuaiLe" "ZCOOL XiaoWei" "ZCOOL QingKe HuangYou" "Ma Shan Zheng" \
  "Zhi Mang Xing" "Long Cang" "Inter" "JetBrains Mono" "WenQuanYi Micro Hei" "AR PL UKai" "Noto Color Emoji" "FontAwesome"; do
  fc-list | grep -q "$f" || { echo "missing $f"; exit 1; }; done; echo "$(fc-list | wc -l) font files"'
check python-draw python3 /smoke/draw.py
check playwright-python python3 /smoke/browser.py
check playwright-node node /smoke/browser_node.js
check apt-install bash -c 'apt-get install -y --no-install-recommends tree >/dev/null 2>&1 && tree --version'
check pip-install bash -c 'pip install -q tqdm && python -c "import tqdm; print(\"tqdm\", tqdm.__version__)"'
check npm-install bash -c 'cd /tmp && npm install --silent left-pad >/dev/null 2>&1 && node -e "console.log(require(\"/tmp/node_modules/left-pad\")(\"npm ok\", 8))"'
if python3 -c "import urllib.request as u; u.build_opener(u.ProxyHandler({})).open('http://127.0.0.1:18883/health', timeout=3)" 2>/dev/null; then
  check tts bash -c 'rm -f /out/tts.wav && qwen3_tts.py --text "你好，这是 L0 镜像的语音测试。" --language zh --output /out/tts.wav'
else
  echo "SKIP tts (no gateway on 127.0.0.1:18883)"
fi
echo "== $pass passed, $fail failed; artifacts in /out"
[ "$fail" -eq 0 ]

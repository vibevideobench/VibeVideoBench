#!/usr/bin/env bash
# Download everything the L0 image needs (except apt packages) into downloads/.
# The proxy throttles each connection, so every file goes through pdl (parallel ranged download).
# Re-running skips files that already exist; checksums are verified where upstream publishes them.
set -euo pipefail
cd "$(dirname "$0")"
PDL=build/pdl.py
D=downloads
get() { mkdir -p "$(dirname "$D/$2")"; python3 "$PDL" --skip-existing "$1" "$D/$2" -n 16; }

# --- Node 24 LTS ---
NODE=v24.22.0
get https://nodejs.org/dist/$NODE/node-$NODE-linux-x64.tar.xz node/node-$NODE-linux-x64.tar.xz
get https://nodejs.org/dist/$NODE/SHASUMS256.txt node/SHASUMS256.txt
(cd $D/node && grep " node-$NODE-linux-x64.tar.xz\$" SHASUMS256.txt | sha256sum -c -)

# --- Chromium for Playwright 1.62.0 (chromium-1234 = Chrome for Testing 151.0.7922.34) ---
# Laid out like the CDN so the build can serve it to `playwright install` via PLAYWRIGHT_DOWNLOAD_HOST.
CFT=151.0.7922.34
for z in chrome-linux64 chrome-headless-shell-linux64; do
  get https://cdn.playwright.dev/builds/cft/$CFT/linux64/$z.zip playwright/builds/cft/$CFT/linux64/$z.zip
done
get https://cdn.playwright.dev/builds/ffmpeg/1011/ffmpeg-linux.zip playwright/builds/ffmpeg/1011/ffmpeg-linux.zip

# --- /opt/vendor: full npm packages ---
get https://registry.npmjs.org/three/-/three-0.160.0.tgz npm/three-0.160.0.tgz
get https://registry.npmjs.org/gsap/-/gsap-3.12.5.tgz npm/gsap-3.12.5.tgz
sha256sum -c - <<EOF
1ee2f935c4f555814b388e87b5ef78a44856bd2e9d0feb88643a6e193fb42856  $D/npm/three-0.160.0.tgz
134ab4f04f6d98d5fd55f89b6f07726a3b9660c483d4ca31acc0efec62fd5236  $D/npm/gsap-3.12.5.tgz
EOF

# --- Python wheels pinned in build/requirements.lock ---
python3 build/fetch_wheels.py build/requirements.lock $D/wheels "$PDL"

# --- Fonts not packaged by Ubuntu (all SIL OFL) ---
LXGW=v1.522
for w in Light Regular Medium; do
  get https://github.com/lxgw/LxgwWenKai/releases/download/$LXGW/LXGWWenKai-$w.ttf fonts/lxgw-wenkai/LXGWWenKai-$w.ttf
done
get https://github.com/atelier-anchor/smiley-sans/releases/download/v2.0.1/smiley-sans-v2.0.1.zip zips/smiley-sans-v2.0.1.zip
get https://github.com/rsms/inter/releases/download/v4.1/Inter-4.1.zip zips/Inter-4.1.zip
get https://github.com/JetBrains/JetBrainsMono/releases/download/v2.304/JetBrainsMono-2.304.zip zips/JetBrainsMono-2.304.zip
GF=47d54ca1ebb14fd864695f1d39dc734f39017c33   # google/fonts commit, 2026-10-16
for p in zcoolkuaile/ZCOOLKuaiLe-Regular.ttf zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf \
         zcoolqingkehuangyou/ZCOOLQingKeHuangYou-Regular.ttf mashanzheng/MaShanZheng-Regular.ttf \
         zhimangxing/ZhiMangXing-Regular.ttf longcang/LongCang-Regular.ttf; do
  dir=${p%%/*}
  get https://raw.githubusercontent.com/google/fonts/$GF/ofl/$p fonts/google/$dir/${p##*/}
  get https://raw.githubusercontent.com/google/fonts/$GF/ofl/$dir/OFL.txt fonts/google/$dir/OFL.txt
done
python3 -I build/extract_fonts.py $D/zips $D/fonts
# The Smiley Sans release zip ships no license text (the font is OFL 1.1).
get https://raw.githubusercontent.com/atelier-anchor/smiley-sans/v2.0.1/LICENSE fonts/smiley-sans/LICENSE

# Check what was downloaded against the pinned list build/downloads.sha256 (written when it does not exist yet, or
# when it matches). LC_ALL=C: the pinned list is byte-sorted, and diff compares line by line, so a shell with a
# UTF-8 locale would otherwise sort the same files differently and the check would fail on the order alone.
(cd $D && find node playwright npm wheels fonts -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > build/downloads.sha256.new
if [ -f build/downloads.sha256 ] && ! diff -u build/downloads.sha256 build/downloads.sha256.new; then
  echo "downloads differ from the pinned build/downloads.sha256 (diff above): an upstream file changed" >&2
  exit 1
fi
mv build/downloads.sha256.new build/downloads.sha256
echo "done: $(wc -l < build/downloads.sha256) files, $(du -sh $D | cut -f1)"

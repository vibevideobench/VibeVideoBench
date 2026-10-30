# Environment (image vibe-l0:0.1)

Ubuntu 24.04 container. You are root. There is no GPU and no display server (Xvfb is installed for
programs that need one). Internet access goes through the HTTP proxy given in `http_proxy` / `https_proxy`;
apt, pip, uv, npm, curl, git, Python, Chromium and Node `fetch` (`NODE_USE_ENV_PROXY=1` is set) all use it.
`127.0.0.1` is the host, so local services on the host are reachable directly.

## Installed

- **Python 3.12** in `/opt/venv` (first on PATH as `python`, `python3`, `pip`): numpy 2.5.4, Pillow 12.3.0,
  opencv-python-headless 4.14.0.94, cairocffi 1.7.1, playwright 1.62.0, uv 0.13.1.
  The venv also sees Python packages installed with apt (`python3-*`).
- **Node.js 24.22.0** and npm 11 in `/opt/node`. Global package: playwright 1.62.0.
- **Chromium 151** (Chrome for Testing 151.0.7922.34, the build Playwright 1.62.0 expects):
  `CHROME_BIN=/opt/chromium/chrome`; the headless shell is under `/opt/ms-playwright`.
  As root, Chromium needs `--no-sandbox` (Playwright adds it by default); `--disable-dev-shm-usage` is
  safe to add. `PUPPETEER_EXECUTABLE_PATH` points to the same binary. WebGL is software-rendered.
- **ffmpeg / ffprobe 6.1.1**, ImageMagick 6 (`convert`), SoX, gcc/g++/make, pkg-config, git, curl, jq.
- Development headers for cairo, pango, libjpeg, giflib and librsvg; Mesa software OpenGL/EGL (llvmpipe).
- **/opt/vendor**: three 0.160.0 (full npm package in `/opt/vendor/three`, plus `/opt/vendor/three.module.js`)
  and GSAP 3.12.5 (full npm package in `/opt/vendor/gsap`, plus `/opt/vendor/gsap.min.js`).
  `manifest.json` lists versions and sha256.
- **Fonts** (`fc-list`; files under `/usr/share/fonts`): Noto Sans / Serif CJK SC, TC, JP, KR (all weights),
  WenQuanYi Zen Hei and Micro Hei, AR PL UKai, LXGW WenKai, Smiley Sans (得意黑), ZCOOL KuaiLe,
  ZCOOL XiaoWei, ZCOOL QingKe HuangYou, Ma Shan Zheng, Zhi Mang Xing, Long Cang, Noto Sans / Serif,
  DejaVu, Liberation, Inter, JetBrains Mono, Noto Color Emoji, Font Awesome 4 (family `FontAwesome`).
  fontconfig generic families: `sans-serif` → Noto Sans CJK SC, `serif` → Noto Serif CJK SC,
  `monospace` → Noto Sans Mono CJK SC.
  The Noto CJK files in `/usr/share/fonts/opentype/noto/` are collections (`.ttc`) whose face index 0 is
  Japanese: index 0 = JP, 1 = KR, 2 = SC, 3 = TC, 4 = HK. Simplified Chinese with Pillow is
  `ImageFont.truetype("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", size, index=2)`;
  `fc-match -f '%{file} %{index}\n' 'sans-serif:lang=zh-cn'` prints the file and index.

## Installing more

`apt-get install`, `pip install`, `npm install` and `uv` all work. The apt package lists are from the
image build date; run `apt-get update` first if a package download returns 404.

## Speech (Qwen3-TTS, cloud)

```
python3 /opt/benchmark-tools/qwen3_tts.py --text "你好，欢迎试听。" --language zh --output chinese.wav
python3 /opt/benchmark-tools/qwen3_tts.py --input narration.txt --language en --output english.wav
```

Options: `--language zh|en|auto`, `--voice` (default Cherry), `--model` (default qwen3-tts-flash).
Each call takes 1–600 characters, refuses to overwrite an existing file, writes a WAV and prints JSON
including `duration_s` and `sample_rate`. No API key is needed inside the container.

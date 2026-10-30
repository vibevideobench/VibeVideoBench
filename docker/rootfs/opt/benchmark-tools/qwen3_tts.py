#!/usr/bin/env python3
"""Qwen3 云端 TTS（容器版）：只需 Python 3 标准库，无需本地模型或 ffmpeg。

容器里没有 API Key：请求发给宿主机上的 TTS 网关（环境变量 QWEN3_TTS_URL），由网关调用百炼。

中文：python3 qwen3_tts.py --text "你好，欢迎试听。" --language zh --output chinese.wav
英文：python3 qwen3_tts.py --text "Hello, welcome to the demo." --language en --output english.wav
文件：python3 qwen3_tts.py --input narration.txt --language zh --output narration.wav
"""
import argparse
import base64
import io
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.request
import wave

GATEWAY = os.environ.get("QWEN3_TTS_URL", "http://127.0.0.1:18883/tts")
DEFAULT_MODEL = "qwen3-tts-flash"
LANGUAGES = {"zh": "Chinese", "en": "English", "auto": "Auto"}


def synthesize(text, language, voice, model):
    payload = {"text": text, "language": language, "voice": voice, "model": model}
    request = urllib.request.Request(
        GATEWAY, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    # 网关在本机，不经过 HTTP 代理。
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=180) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"TTS 网关返回 HTTP {exc.code}：{detail[:800]}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"无法连接 TTS 网关 {GATEWAY}：{exc.reason}") from None


def normalize_wav(raw):
    """修正服务端可能返回的流式 WAV 长度字段，保持 PCM 音频不变。"""
    with wave.open(io.BytesIO(raw), "rb") as source:
        channels, width, rate = source.getnchannels(), source.getsampwidth(), source.getframerate()
        pcm = source.readframes(len(raw) // (channels * width))
    if not pcm or len(pcm) % (channels * width):
        raise ValueError("返回的音频为空或存在不完整的 PCM 采样。")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(width)
        target.setframerate(rate)
        target.writeframes(pcm)
    return output.getvalue(), len(pcm) / (channels * width * rate), rate


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="直接输入朗读文本")
    source.add_argument("--input", type=Path, help="UTF-8 文本文件")
    parser.add_argument("--language", choices=LANGUAGES, default="zh", help="zh 中文，en 英文，auto 自动")
    parser.add_argument("--voice", default="Cherry", help="音色，默认 Cherry")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="默认 qwen3-tts-flash")
    parser.add_argument("--output", type=Path, required=True, help="输出 WAV 文件路径")
    args = parser.parse_args()
    text = (args.text if args.text is not None else args.input.read_text(encoding="utf-8-sig")).strip()
    if not text or len(text) > 600:
        parser.error("每次请输入 1～600 个字符；长文请按完整语义段落拆分。")
    if args.output.suffix.lower() != ".wav":
        parser.error("输出文件请使用 .wav 扩展名。")
    if args.output.exists():
        parser.error(f"输出文件已存在，请另选文件名：{args.output}")
    result = synthesize(text, LANGUAGES[args.language], args.voice, args.model)
    audio, duration, rate = normalize_wav(base64.b64decode(result["audio_b64"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as target:
        target.write(audio)
    print(json.dumps({"output": str(args.output.resolve()), "model": args.model,
                      "voice": args.voice, "language": LANGUAGES[args.language],
                      "duration_s": round(duration, 3), "sample_rate": rate,
                      "usage": result.get("usage"), "request_id": result.get("request_id")}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, KeyError, wave.Error, EOFError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        sys.exit(1)

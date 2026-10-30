#!/usr/bin/env python3
"""Host-side Qwen3-TTS gateway: the DashScope key stays on the host, containers never see it.

Containers (docker --network host) POST JSON {"text", "language", "voice", "model"} to
http://127.0.0.1:18883/tts, or to /case/<system>/<case>/tts to tag calls per case. The gateway calls
DashScope's Qwen3-TTS API, downloads the WAV and returns {"audio_b64", "request_id", "usage"}. Every call is
appended to the JSONL log. Upstream failures come back as HTTP 502 with "kind": "environment", so the controller
can count them as environment faults rather than model errors.

The DashScope key is DASHSCOPE_API_KEY from the environment, else from --key-file: a KEY=value env file holding
DASHSCOPE_API_KEY (default: keys.env at the repository root) or a file holding only the key.

The in-container client (rootfs/opt/benchmark-tools/qwen3_tts.py) gives up after 180 s, so a request must be
answered well within that (with few slots and many parallel runs the queue can otherwise grow past every
client's timeout while the gateway keeps synthesizing audio nobody waits for):
  - a request that cannot get a slot within --queue-timeout is answered 503 "busy" without calling DashScope;
  - synthesis is retried only when nothing was synthesized: DashScope's 429 rate limit, or a request that never
    reached DashScope (proxy CONNECT refused, connection reset while sending); otherwise not (each one costs quota);
  - the audio download is retried; --proxy sets the egress for both (default: the environment's http(s)_proxy).

  python3 gateway/tts_gateway.py [--key-file keys.env] [--log tts_calls.jsonl] [--proxy http://host:port] \
      [--concurrency 3] [--queue-timeout 60]
"""
import argparse
import base64
import http.server
import json
import os
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"
LANGUAGES = {"Chinese", "English", "Auto"}
ROUTE = re.compile(r"^/(?:case/([\w-]+)/([\w-]+)/)?tts$")
VOICE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MODEL = re.compile(r"^qwen3-tts[\w.-]{0,60}$")
LOG_LOCK = threading.Lock()
SEND_TRIES, DOWNLOAD_TRIES = 3, 3
KEYS_ENV = (Path(os.environ["VVBENCH_CONFIG_DIR"]).expanduser() if os.environ.get("VVBENCH_CONFIG_DIR")
            else Path(__file__).resolve().parent.parent) / "keys.env"


def read_key(path):
    """DASHSCOPE_API_KEY from the environment, else from path (a KEY=value env file, or the key alone)."""
    key, where = os.environ.get("DASHSCOPE_API_KEY", "").strip(), "environment variable DASHSCOPE_API_KEY"
    if not key:
        text, where = path.read_text(encoding="utf-8-sig"), str(path)
        found = re.search(r'^\s*(?:export\s+)?DASHSCOPE_API_KEY\s*=\s*[\'"]?([^\'"\s]+)', text, re.M)
        key = found.group(1) if found else text.strip().strip('"\'`')
    if not key.startswith("sk-") or len(key) < 20 or any(c.isspace() for c in key):
        raise ValueError(f"{where} must hold one complete DashScope API key")
    return key


def synthesize(text, language, voice, model, key, opener, give_up):
    """Retried until give_up (epoch s) only when nothing was synthesized: 429 Throttling.RateQuota (an account
    may allow only a few requests at a time) or a request that never
    reached DashScope (urllib raises a bare URLError only while sending: proxy tunnel refused, connect failed)."""
    payload = {"model": model, "input": {"text": text, "voice": voice, "language_type": language}}
    request = urllib.request.Request(
        API_URL, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
    )
    attempt = 0
    while True:
        attempt += 1
        try:
            with opener.open(request, timeout=25) as response:
                result = json.load(response)
            break
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").replace(key, "[REDACTED]")
            if exc.code != 429 or time.time() + attempt > give_up:
                raise RuntimeError(f"DashScope HTTP {exc.code}: {detail[:800]}") from None
        except OSError:
            # URLError: never sent. A read timeout may have been billed, but a synthesis costs a few hundred
            # characters and usually answers in ~2 s, while a hung one holds the client: retry it too.
            if attempt >= SEND_TRIES or time.time() + 25 > give_up:
                raise
        time.sleep(min(attempt, 5))
    if not result.get("output", {}).get("audio", {}).get("url"):
        raise RuntimeError("DashScope returned no audio url (model permission or quota?)")
    return result


def download_audio(url, opener):
    parsed = urllib.parse.urlsplit(url)
    if not parsed.hostname or not parsed.hostname.endswith(".aliyuncs.com"):
        raise RuntimeError("audio url is not on the expected aliyuncs.com domain")
    url = urllib.parse.urlunsplit(parsed._replace(scheme="https"))
    for attempt in range(DOWNLOAD_TRIES):
        try:
            with opener.open(url, timeout=15) as response:
                return response.read()
        except (urllib.error.URLError, OSError):
            time.sleep(1)
    raise RuntimeError("synthesis succeeded but the audio download failed")


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def reply(self, status, body):
        raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/health":
            return self.reply(404, {"error": "not found"})
        self.reply(200, {"status": "ok"})

    def do_POST(self):
        match = ROUTE.match(urllib.parse.urlsplit(self.path).path)
        if not match:
            return self.reply(404, {"error": "POST /tts or /case/<system>/<case>/tts"})
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            text, language = str(body["text"]).strip(), body.get("language", "Chinese")
            voice, model = body.get("voice", "Cherry"), body.get("model", "qwen3-tts-flash")
        except (ValueError, KeyError, TypeError):
            return self.reply(400, {"error": "body must be JSON with a text field", "kind": "request"})
        checks = [(1 <= len(text) <= 600, "text must be 1-600 characters"),
                  (language in LANGUAGES, f"language must be one of {sorted(LANGUAGES)}"),
                  (VOICE.match(str(voice)), "invalid voice name"),
                  (MODEL.match(str(model)), "only qwen3-tts models are provided")]
        for ok, problem in checks:
            if not ok:
                return self.reply(400, {"error": problem, "kind": "request"})
        row = {"started": time.time(), "system": match.group(1), "case": match.group(2), "chars": len(text),
               "language": language, "voice": voice, "model": model}
        got = self.server.slots.acquire(timeout=self.server.queue_timeout)
        row["queued_s"] = round(time.time() - row["started"], 3)
        try:
            if not got:
                row.update(status=503, error="TTS gateway busy: no free slot, nothing was synthesized; retry shortly")
                return self.reply(503, {"error": row["error"], "kind": "environment"})
            try:
                # leave room for the download inside the client's 180 s
                result = synthesize(text, language, voice, model, self.server.key, self.server.opener,
                                    row["started"] + 120)
                audio = download_audio(result["output"]["audio"]["url"], self.server.opener)
            finally:
                self.server.slots.release()
        except (RuntimeError, ValueError, OSError) as exc:
            row.update(status=502, error=str(exc).replace(self.server.key, "[REDACTED]"))
            self.reply(502, {"error": row["error"], "kind": "environment"})
        else:
            row.update(status=200, bytes=len(audio), request_id=result.get("request_id"), usage=result.get("usage"))
            self.reply(200, {"audio_b64": base64.b64encode(audio).decode("ascii"),
                             "request_id": result.get("request_id"), "usage": result.get("usage")})
        finally:
            row["elapsed_s"] = round(time.time() - row["started"], 3)
            with LOG_LOCK, open(self.server.log, "a", encoding="utf-8") as log:
                log.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    global API_URL
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--key-file", type=Path, default=KEYS_ENV,
                        help="keys.env with DASHSCOPE_API_KEY, or a file with the key alone (when the variable is unset)")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18883)
    parser.add_argument("--log", type=Path, default=Path("tts_calls.jsonl"))
    parser.add_argument("--concurrency", type=int, default=3, help="parallel DashScope calls (rate limit)")
    parser.add_argument("--queue-timeout", type=float, default=60, help="seconds a request may wait for a slot")
    parser.add_argument("--proxy", help="HTTP proxy for DashScope and the audio download (default: environment)")
    parser.add_argument("--api-url", default=os.environ.get("DASHSCOPE_TTS_URL", API_URL),
                        help="DashScope TTS endpoint; DashScope keys are per region, so with an international key use "
                             "https://dashscope-intl.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation "
                             "(env DASHSCOPE_TTS_URL; default: the mainland China endpoint)")
    args = parser.parse_args()
    API_URL = args.api_url
    server = http.server.ThreadingHTTPServer((args.bind, args.port), Handler)
    server.key, server.log = read_key(args.key_file), args.log
    server.slots = threading.BoundedSemaphore(args.concurrency)
    server.queue_timeout = args.queue_timeout
    server.opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": args.proxy, "https": args.proxy} if args.proxy else None))
    print(f"tts gateway on http://{args.bind}:{args.port}/tts, log {args.log}, {args.concurrency} slots, "
          f"queue timeout {args.queue_timeout:g} s, {API_URL}, proxy {args.proxy or 'from environment'}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

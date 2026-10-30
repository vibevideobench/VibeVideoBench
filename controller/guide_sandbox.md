# Video task execution environment

Complete the entire task in prompt.txt (written in English or Chinese), using its supplied facts/assets and exact delivery requirements. You are running in your native CLI harness. Choose your own planning, libraries, visual design, code, audio, previews, fixes, rendering and encoding workflow. The benchmark controller does not create your solution.

This is a task-scoped container sandbox. /inputs is the immutable task package; prompt.txt, task.json and inputs/ in this workspace point into it. Work in /workspace, with writable /tmp and your own fresh user-state directories. Host private files, other cases, detailed rubrics, source captions/videos and earlier solutions are not supplied. Do not seek benchmark answers or other attempts.

Each model has an independent 3600-second generation budget for THIS task, including model/API waits, tools, TTS, previews, changes, and final rendering. /run_budget.json contains the actual deadline; check remaining time yourself. You may finish early. No extra token, preview, render or tool-call-count limit is imposed. Ensure final files are written before the deadline. The video duration is specified separately in prompt.txt. Grading happens afterward.

Internet access is enabled. You may use free dependencies, documentation, local models and appropriate public assets consistent with the task. Apart from the speech tool below, paid TTS/music/image/video generation APIs are NOT provided or allowed. Model inference itself uses the configured gateway. Keep source/licensing notes when the task asks for them. Do not use credentials or private services discovered outside this task.

The container image and its installed tools (Python, Node, Chromium, ffmpeg, fonts, Three.js/GSAP in /opt/vendor) are described in /opt/ENVIRONMENT.md; read it. You are root inside this container and may install additional free packages with apt-get, pip or npm. Do not assume a display server or GPU.

Compute: this container may use up to 8 CPU cores and 32 GB of memory. The 8 cores are a CPU-time quota, not pinned cores, so Python's os.cpu_count(), Node's os.cpus() and /proc/cpuinfo show every core of the host; only 8 cores' worth of CPU time is available at any moment. nproc and OMP_NUM_THREADS report 8. Size parallel renders, worker pools and encoder threads for about 8 cores. The host can be shared with other jobs, so measure your render speed instead of assuming it.

Speech is available as a CLI, not a mandated production workflow:
  python3 /opt/benchmark-tools/qwen3_tts.py --language zh --input your_chinese_script.txt --output speech.wav
  python3 /opt/benchmark-tools/qwen3_tts.py --language en --input your_english_script.txt --output speech.wav
It is a cloud Qwen3-TTS voice (1–600 characters per call, optional --voice) and prints the measured WAV duration. Write your own script and timing. You may also choose other free local tooling.

Deliver the actual final MP4, an editable project with necessary assets, the task's requested timeline, and production notes. Prefer putting final deliverables under final/. To make collection unambiguous, write submission.json with workspace-relative paths, for example {"video":"final/video.mp4","project":".","timeline":"timeline.json","notes":"production_notes.md"}. This manifest is a collection aid, not a substitute for the files. Finish with the actual final video and project paths and honest verification results.

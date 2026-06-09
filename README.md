# Dog Behavior Detector

## Introduction

This is a small personal project for detecting dog behavior using RTSP-supported home security cameras, YOLO object detection, and local LLM. If suspicious behavior (e.g. destructive behavior, zoomies) is detected, it sends an alert to your Telegram account(s).

## Architecture

There are 3 main types of components: detectors, recorders, and manager. The high-level logic is as follows:

- Each **detector** (1 for each stream) grabs a frame from the RTSP stream and runs them through `yolo26n` every 1 second. It notifies the manager when a dog is detected.
- Each **recorder** (1 for each stream) saves the last 5 seconds of video footage in memory at 5 FPS.
- If a dog is detected, the **manager** grabs the relevant video snippets from the recorders, sends it to the LLM for analysis, and sends an alert via Telegram.

Other features:

- If YOLO doesn't detect a dog in any stream for 10 seconds, the LLM steps in for object detection.
- To save prefill time, frames are cropped and/or downscaled for LLM analysis.
- The manager compiles the frames back into a video file when sending you a Telegram alert.
- Telegram alerts for camera outages and errors
- A web interface for monitoring logs (port 8972)

## Usage

Copy `config.yaml.sample` to `config.yaml` and modify it, including but not limited to:

- `streams`: make an entry with keys `name` and `rtsp` (URL) for each RTSP stream
- `yolo_device`: set to Intel iGPU by default
- `llm_endpoint.openai_compatible_url` if you're not running LM Studio server on the same device
- `llm_endpoint.model`
    - If you're using a reasoning model, you will likely need to increase `max_tokens` to 2-4K or more
- `dog_description`
- Under `telegram`: 
    - `bot_token` from @BotFather
    - `chat_ids` from @userinfobot
    - `live_stream_url`, `logs_url` to your server URL

`prompt.txt` contains the prompt that will be sent to the LLM along with the video frames. Read through it and make any changes appropriate for your situation.

When you're ready, use Docker Compose to start the application. For example:

```
docker compose up --build -d
```

The web interface is hosted on port 8972 by default. You can use the following Telegram commands:
```
status - Get cameras status
last - Get most recent analysis result
logs - Get URL to view previous analysis results
live - Get URL to view live video streams
score - Set alert score threshold
sysalert - Enable/disable system alerts
mute - Temporarily mute alerts
unmute - Resume receiving alerts
```

You can also copy-paste this to BotFather.

## Other

### Disclaimer

This project was developed specifically for my home server configuration:

- i5-10210U w/ Intel iGPU, 32GB DDR4 memory
- Ubuntu Server 24.04
- [Frigate](https://frigate.video/) running on port 8971 with bundled go2rtc running on 8554
    - Cameras: 6x Aqara G100
- [LM Studio](https://lmstudio.ai/) server running on port 1234
    - Models are running on a more powerful GPU on a different device via LM Link (either RTX 5070 Ti mobile, RTX 5090 desktop, or M5 Max)
- Connected to other personal devices with Tailscale

This project was not developed with other devices in mind and might not work for you out-of-the-box. You may need to modify the source code for this to work on your device; do it at your own risk.

Tips if you do want to use this app directly:

- Find a model that works best for you. Bigger models and reasoning models may be more accurate but are slower.
- Through trial and error, find configuration and prompt that works best for your dog and home.
- Make sure to test thoroughly to minimize inaccuracies.

### Limitations

You should never use this as a primary supervision method for your dog; it's best if you use it as a backup for human errors. Expect lots of false negatives and false positives until you fine-tune the prompt and config, and even then, some of them will be inevitable due to the inherent limitations of AI.

### License

GNU General Public License (GPL)
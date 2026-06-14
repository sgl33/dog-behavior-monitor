# Dog Behavior Detector

## Introduction

This is a small personal project for detecting dog behavior using RTSP-supported home security cameras, YOLO object detection, and local LLM. If suspicious behavior (e.g. destructive behavior, zoomies) is detected, it sends an alert to your Telegram account(s).

## Architecture

There are 3 main types of components: detectors, recorders, and manager. The high-level logic is as follows:

- Each **detector** (1 for each stream) grabs a frame from the RTSP stream and runs them through YOLO every 1 second. It notifies the manager when a dog is detected.
- Each **recorder** (1 for each stream) saves the last 5 seconds of video footage in memory at 5 FPS.
- If a dog is detected, the **manager** grabs the relevant video snippets from the recorders, sends it to the LLM for analysis, and sends an alert via Telegram.

Other features:

- To minimize false positives, the same footage is sent to the LLM twice and both must be flagged before Telegram alerts are sent.
- If YOLO doesn't detect a dog in any stream for 10 seconds, the LLM steps in for object detection.
- To save prefill time, frames are cropped and/or downscaled for LLM analysis.
- The manager compiles the frames back into a video file when sending you a Telegram alert.
- Telegram alerts for camera outages and errors
- A web interface for monitoring logs (port 8972)

## Usage

Initial setup will only take 5-10 minutes, but hours of testing and fine-tuning is recommended before using it in "production". See [USAGE.md](USAGE.md) for details.

## Other

### Disclaimer

This project was developed specifically for my home server configuration:

- Intel Core i5-10210U w/ Intel UHD Graphics
- NVIDIA GeForce RTX 5070 Ti
- 32GB DDR4 memory
- Ubuntu Server 24.04
- [Frigate](https://frigate.video/) running on port 8971 with bundled go2rtc running on 8554
    - Cameras: 8x Aqara G100

This project was not developed with other devices in mind and might not work for you out-of-the-box. You may need to modify the source code for this to work on your device; do it at your own risk.

### Limitations

You should never use this as a primary supervision method for your dog; it's best if you use it as a backup for human errors. Expect lots of false negatives and false positives until you fine-tune the prompt and config, and even then, some of them will be inevitable due to the inherent limitations of AI.

### License

GNU General Public License (GPL)
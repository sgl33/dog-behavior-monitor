# How to Use

This documentation covers how to use this `dog-behavior-detector` software.

## Model Selection

### YOLO Object Detection

If you're running YOLO inference on a dedicated GPU, you probably don't need to worry about the model size. But if you're using an integrated GPU, you should carefully select a YOLO model size that supports your hardware configuration. 

`YOLO26n` or `YOLO26s` is recommended for most integrated GPUs. If you're running it on a dedicated GPU, try `YOLO26m` or `YOLO26l`. You will receive a Telegram message (if configured) if YOLO inference falls behind.

### Vision LLM

The Vision LLM model is the primary model used to analyze video clips. This model should be run **locally** to avoid high API usage costs and for privacy. 

As of July 2026, Qwen3 models are a great fit for this due to their efficient vision encoders and great instruction-following capabilities. Specifically, the following models are recommended (NVFP4 quantization on vLLM):
- RTX 5060 Ti 8GB: [Qwen3 VL 4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)
- RTX 5070 Ti 16GB: [Qwen3 VL 8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct)
- 2x RTX 5060 Ti 16GB (32GB total): [Qwen3.6 35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)
- RTX 5090 32GB: [Qwen3.6 27B](https://huggingface.co/Qwen/Qwen3.6-27B) 

### Double-pass LLM (Optional)

You can enable double-pass verification to reduce false positives. If enabled, videos flagged by the Vision LLM will go through the double-pass LLM for a second round of verification and send an alert only if both LLMs flag it.

Sample use cases:
- **Reduce false positives**: you want to reduce false positives by running it through the same model twice at non-zero temperature.
- **Save energy**: you want to run the primary model on a power-efficient GPU (e.g. RTX 5060 Ti, 180W) but want to verify alerts on a more powerful GPU (e.g. RTX 5090, 575W).
- **Cloud model**: you want to use a powerful cloud model that cannot run on your local machine to verify alerts.

### Fast & Query LLMs

The "Fast" model is currently used for background summarization tasks and triaging user queries. Its tasks are relatively simple, so any model with 16K+ model should work. It can also be the same model as your vision model.

The "Query" model is currently used when you ask a question to the Telegram bot about your dog. This model should support at least 256K context. 

For the "Fast" model, either use the "Vision" model or a [free OpenRouter model](https://openrouter.ai/openrouter/free). For the "Query" model, you may need to use a paid model if the free models are not good enough.

> Note: OpenRouter or its providers may store your prompt data and use it for training purposes, depending on the model. No image or video is sent to "Fast" or "Query" models, but if privacy is a concern, use a paid model and provider with Zero Data Retention (ZDR).

## Configuration

Copy `sample-config.yaml` to `config.yaml` and modify it.

`prompt.txt` contains the prompt that will be sent to the LLM along with the video frames. Read through it and make any changes appropriate for your situation.

## Run

Use Docker Compose to start the application. For example:

```
docker compose up --build -d
```

To stop the containers:
```
docker compose down
```

The containers will start automatically when the server restarts, unless manually stopped.

### Sharing one GPU between vLLM and YOLO (CUDA MPS)

If the YOLO detector and the vLLM server run on the **same** NVIDIA GPU, enable the
CUDA Multi-Process Service (MPS) so their kernels overlap on the SMs instead of
serializing — this avoids vLLM token-latency jitter caused by YOLO's periodic
inference bursts.

1. Install the host MPS daemon (runs as root, starts before Docker, survives reboots):
   ```
   sudo cp deploy/nvidia-mps.service /etc/systemd/system/nvidia-mps.service
   sudo systemctl daemon-reload
   sudo systemctl enable --now nvidia-mps
   ```
2. The compose files are already wired for it: both the detector (`docker-compose.yaml`)
   and vLLM (`llm/qwen3-vl-8b.yaml`) mount the MPS pipe, set `ipc: host`, and cap their
   SM share via `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (detector 4%, vLLM 96%). These caps
   are independent limits, not a partition — tune them to taste.
3. Verify both container processes show type **`M+C`** in `nvidia-smi`. Plain `C` means a
   container didn't join the daemon (check `ipc: host` and the `/tmp/nvidia-mps` mount).

If YOLO inference starts falling behind (you'll get a Telegram alert), raise the detector's
percentage; this barely affects vLLM since YOLO only bursts roughly once per `detect_interval`.

## Telegram

You can use the following Telegram commands:


You can copy-paste the following to BotFather:
```
status - Get cameras status
score - Set alert score threshold
sysalert - Enable/disable system alerts
mute - Disable behavior alerts
unmute - Enable behavior alerts
snooze - Temporarily snooze behavior alerts
```

## Tips

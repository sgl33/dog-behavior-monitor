# How to Use

This documentation covers how to use this `dog-behavior-detector` software.

## Model Selection

### YOLO Object Detection

If you're running YOLO inference on a dedicated GPU, you probably don't need to worry about the model size. But if you're using an integrated GPU, you should select a YOLO model size carefully that supports your configuration. 

Assuming the default configuration, the following is an *estimated* maximum number of camera streams that the integrated GPU can comfortably support:

| iGPU | Nano (`yolo26n`) | Small (`yolo26s`) | Medium (`yolo26m`) |
| - | - | - | - |
| Intel UHD Graphics (i5-10210U) | 8 | 5 | - |
| Intel Iris Xe Graphics | 20 | 10 | 5 |
| Intel Arc Graphics | 80 | 35 | 15 |
| AMD Radeon 780M | 50 | 25 | 12 |

This is assuming nothing else is running on the GPU. If you have other processes like Frigate motion detection, the actual numbers will be different. You will receive a Telegram message (if configured) if YOLO inference falls behind.

### Vision LLM

The Vision LLM model is the primary model used to analyze video clips. This model should be run **locally** to avoid high API usage costs (as videos take up *a lot* of input tokens) and for privacy. The model should

- Support image inputs, 32K minimum context, and be reasonably accurate.
- Be reasonably fast and fit entirely in the GPU's VRAM. 
- Not "think" before responding - the "thinking" process makes latency highly unpredictable.

As of June 2026, [Qwen3 VL 8B](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) is a great place to start. It runs well on just 12GB of VRAM at Q4 (16GB for NVFP4) and is reasonably accurate. If it's too slow or doesn't fit, try [Qwen3 VL 4B](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct); I wouldn't go use any smaller models than that. If you have some headroom, try [Qwen3.6 27B](https://huggingface.co/Qwen/Qwen3.5-27B).

If you have a NVIDIA Blackwell GPU (such as the RTX 50 series), use NVFP4 quantizations on [vLLM](https://vllm.ai/) to increase your inference speed by up to 3-4x. Note, however, that NVFP4 models require more VRAM than regular 4-bit quantizations. If you're buying a new GPU, the RTX 5070 Ti is just right for Qwen3 VL 8B and RTX 5090 for Qwen3.6 27B (both NVFP4) for sub-2s inference average.

### Fast LLM

The "Fast" model is currently used for background summarization tasks and triaging user queries. Its tasks are relatively simple, so any model with 16K+ model should work. It can also be the same model as your vision model though not recommended for maximum performance.

The "Query" model is currently used when you ask a question to the Telegram bot about your dog. This model should support at least 256K context. 

The [free OpenRouter models](https://openrouter.ai/openrouter/free) are recommended. By simply adding $10 to your account, you get 1,000 free API calls per day. For the "Query" model, you may need to use a paid model if the free models are not good enough.

> Note: OpenRouter or its providers may store your prompt data and use it for training purposes, depending on the model. No image or video is sent to "Fast" or "Query" models, but if privacy is a concern, use a paid model and provider with Zero Data Retention (ZDR).

## Configuration

Copy `sample-config.yaml` to `config.yaml` and modify it, including but not limited to:

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

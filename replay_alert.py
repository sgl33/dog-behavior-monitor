#!/usr/bin/env python3
"""Send a saved alert JSON to the vision endpoint and print the response.

Usage:
    python replay_alert.py data/alerts/20240101_120000_score7.json
    python replay_alert.py data/alerts/20240101_120000_score7.json --model my-model
    python replay_alert.py data/alerts/20240101_120000_score7.json --url http://localhost:8000/v1
"""

import argparse
import json
import sys
from pathlib import Path

import requests
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved alert JSON against the vision endpoint.")
    parser.add_argument("json_file", type=Path, help="Path to the alert .json file")
    parser.add_argument("--url", help="Override vision endpoint URL (base, without /chat/completions)")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--token", help="Override bearer token")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    llm = config.get("llm_endpoint", {})
    base_url = (args.url or llm.get("vision_url", "http://localhost:8000/v1")).rstrip("/")
    model = args.model or llm.get("vision_model", "")
    token = args.token or llm.get("vision_token")

    user_content: list[dict] = json.loads(args.json_file.read_text())

    prompt_path = Path(__file__).parent / "prompts" / "analyze_prompt.txt"
    prompt = prompt_path.read_text().format(dog_description=config.get("dog_description", ""))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]

    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens or llm.get("max_tokens", 1024),
        "enable_thinking": False,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "dog_analysis",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "description": {"type": "string"},
                        "summary": {"type": "string"},
                        "score": {"type": "integer"},
                    },
                    "required": ["description", "summary", "score"],
                    "additionalProperties": False,
                },
            },
        },
    }

    headers = {"Authorization": f"Bearer {token}"} if token else {}
    url = f"{base_url}/chat/completions"
    print(f"POST {url}  model={model}")

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        try:
            print(resp.json(), file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)

    content = resp.json()["choices"][0]["message"]["content"]
    print(content)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Send a saved alert JSON to the vision endpoint and print the response.

Usage:
    python eval.py data/alerts/20240101_120000_score7.json
    python eval.py data/eval/          # replay all, print old-vs-new score matrix
    python eval.py data/eval/ --watch  # loop forever, process only new files
    python eval.py data/alerts/20240101_120000_score7.json --model my-model
    python eval.py data/alerts/20240101_120000_score7.json --url http://localhost:8000/v1
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

import requests
import yaml


def _build_payload(user_content: list[dict], config: dict, args: argparse.Namespace) -> tuple[dict, dict, str]:
    llm = config.get("llm_endpoint", {})
    base_url = (args.url or llm.get("vision_url", "http://localhost:8000/v1")).rstrip("/")
    model = args.model or llm.get("vision_model", "")
    token = args.token or llm.get("vision_token")

    prompt_path = Path(__file__).parent / "prompts" / "analyze_prompt.txt"
    prompt = prompt_path.read_text().format(dog_description=config.get("dog_description", ""))
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": user_content},
    ]
    payload = {
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
    return payload, headers, url


def _send(payload: dict, headers: dict, url: str) -> tuple[dict, float]:
    t0 = time.monotonic()
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    elapsed = time.monotonic() - t0
    if not resp.ok:
        raise requests.HTTPError(
            f"HTTP {resp.status_code}: {resp.text}",
            response=resp,
        )
    return json.loads(resp.json()["choices"][0]["message"]["content"]), elapsed


_RESET = "\033[0m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"


def _score_color(score: int) -> str:
    if score >= 7:
        return _RED
    if score >= 4:
        return _YELLOW
    return _GREEN


def _delta_color(delta: int) -> str:
    if delta <= 1:
        return _GREEN
    if delta <= 3:
        return _YELLOW
    return _RED


def _print_matrix(matrix: list[list[int]]) -> None:
    n = len(matrix)
    BREAKS = {4, 7}

    def join_cols(parts: list[str]) -> str:
        segs = []
        for i, p in enumerate(parts):
            if i > 0:
                segs.append(" | " if i in BREAKS else "  ")
            segs.append(p)
        return "".join(segs)

    prefix = "old\\new| "
    col_raw = [f"{c:2}" for c in range(n)]
    col_colored = [f"{_score_color(c)}{c:2}{_RESET}" for c in range(n)]

    print(prefix + join_cols(col_colored))
    sep_line = "".join("+" if ch == "|" else "-" for ch in prefix + join_cols(col_raw))
    print(sep_line)

    for old, row in enumerate(matrix):
        if old in BREAKS:
            print(sep_line)
        cells = [
            f"{_delta_color(abs(old - new))}{v:2}{_RESET}" if v else "  "
            for new, v in enumerate(row)
        ]
        print(f"  {_score_color(old)}{old:2}{_RESET}   | " + join_cols(cells))


_WATCH_INTERVAL = 5  # seconds between directory polls
_HERE = Path(__file__).parent


def _state_file(directory: Path) -> Path:
    return _HERE / f".eval_processed_{directory.resolve().name}"


def _load_processed(directory: Path, model: str) -> dict[str, dict | None]:
    """Return {filename: {"old": n, "new": n} | None} for files already processed with this model."""
    state = _state_file(directory)
    if not state.exists():
        return {}
    data = json.loads(state.read_text())
    if isinstance(data, list):  # migrate old flat list → unknown model
        data = {"": {name: None for name in data}}
        _state_file(directory).write_text(json.dumps(data, indent=2))
    elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
        # migrate intermediate format {model: [filenames]}
        data = {m: {name: None for name in files} for m, files in data.items()}
        _state_file(directory).write_text(json.dumps(data, indent=2))
    return data.get(model, {})


def _save_processed(directory: Path, model: str, processed: dict[str, dict | None]) -> None:
    state = _state_file(directory)
    data = json.loads(state.read_text()) if state.exists() else {}
    if isinstance(data, list):
        data = {"": {name: None for name in data}}
    data[model] = processed
    state.write_text(json.dumps(data, indent=2))


def _process_file(path: Path, config: dict, args: argparse.Namespace) -> tuple[int, int, str, str, float]:
    m = re.search(r"_score(\d+)", path.stem)
    old_score = int(m.group(1)) if m else -1
    user_content = json.loads(path.read_text())
    payload, headers, url = _build_payload(user_content, config, args)
    result, elapsed = _send(payload, headers, url)
    return old_score, result["score"], result["summary"], result["description"], elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved alert JSON against the vision endpoint.")
    parser.add_argument("json_file", type=Path, help="Path to an alert .json file or a directory of them")
    parser.add_argument("--url", help="Override vision endpoint URL (base, without /chat/completions)")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--token", help="Override bearer token")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens")
    parser.add_argument("--watch", action="store_true", help="Loop forever, processing new JSON files as they appear (requires a directory)")
    parser.add_argument("--all", dest="run_all", action="store_true", help="Process all files, ignoring previously evaluated ones")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = args.model or config.get("llm_endpoint", {}).get("vision_model", "")

    if args.watch:
        if not args.json_file.is_dir():
            parser.error("--watch requires a directory argument")
        directory = args.json_file
        processed = _load_processed(directory, model_name)
        print(f"Watching {directory}  model={model_name}  ({len(processed)} already processed with this model)  Ctrl-C to stop")
        matrix = [[0] * 11 for _ in range(11)]
        try:
            while True:
                new_files = sorted(
                    p for p in directory.glob("*.json") if processed.get(p.name) is None
                )
                for path in new_files:
                    scores = None
                    try:
                        old_score, new_score, summary, description, elapsed = _process_file(path, config, args)
                        scores = {"old": old_score, "new": new_score}
                        arrow = f"{_delta_color(abs(old_score - new_score))} ──► {_RESET}"
                        print(f"{path.name}: {_score_color(old_score)}{old_score}{_RESET}{arrow}{_score_color(new_score)}{new_score}{_RESET} - {summary} ({description}) [{elapsed:.2f}s]")
                        if 0 <= old_score <= 10 and 0 <= new_score <= 10:
                            matrix[old_score][new_score] += 1
                    except Exception as e:
                        print(f"{path.name}  →  ERROR: {e}", file=sys.stderr)
                    processed[path.name] = scores
                    _save_processed(directory, model_name, processed)
                time.sleep(_WATCH_INTERVAL)
        except KeyboardInterrupt:
            print()
            _print_matrix(matrix)
        return

    if args.json_file.is_dir():
        directory = args.json_file
        processed = _load_processed(directory, model_name)
        all_files = sorted(directory.glob("*.json"))
        files = all_files if args.run_all else [f for f in all_files if processed.get(f.name) is None]
        if not files:
            print("No new JSON files to process (use --all to reprocess already-evaluated files).", file=sys.stderr)
            sys.exit(1)
        skipped = len(all_files) - len(files)
        if skipped:
            print(f"Skipping {skipped} already-evaluated file(s)  (--all to include them)")
        matrix = [[0] * 11 for _ in range(11)]
        for i, path in enumerate(files, 1):
            scores = None
            try:
                old_score, new_score, summary, description, elapsed = _process_file(path, config, args)
                scores = {"old": old_score, "new": new_score}
                arrow = f"{_delta_color(abs(old_score - new_score))} ──► {_RESET}"
                print(f"[{i}/{len(files)}] {path.name}: {_score_color(old_score)}{old_score}{_RESET}{arrow}{_score_color(new_score)}{new_score}{_RESET} - {summary} ({description}) [{elapsed:.2f}s]")
                if 0 <= old_score <= 10 and 0 <= new_score <= 10:
                    matrix[old_score][new_score] += 1
            except Exception as e:
                print(f"[{i}/{len(files)}] {path.name}  →  ERROR: {e}", file=sys.stderr)
            processed[path.name] = scores
            _save_processed(directory, model_name, processed)
        print()
        print(f"── Current run ({len(files)} file(s)) ──")
        _print_matrix(matrix)

        cumulative_matrix = [[0] * 11 for _ in range(11)]
        for scores in processed.values():
            if scores and 0 <= scores["old"] <= 10 and 0 <= scores["new"] <= 10:
                cumulative_matrix[scores["old"]][scores["new"]] += 1
        cumulative_total = sum(sum(row) for row in cumulative_matrix)
        if cumulative_total > sum(sum(row) for row in matrix):
            print()
            print(f"── All {cumulative_total} scored file(s) in folder ──")
            _print_matrix(cumulative_matrix)
        return

    user_content = json.loads(args.json_file.read_text())
    payload, headers, url = _build_payload(user_content, config, args)
    print(f"POST {url}  model={payload['model']}")
    t0 = time.monotonic()
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        elapsed = time.monotonic() - t0
        resp.raise_for_status()
    except requests.HTTPError as e:
        print(f"HTTP error: {e}", file=sys.stderr)
        try:
            print(resp.json(), file=sys.stderr)
        except Exception:
            pass
        sys.exit(1)
    print(f"({elapsed:.2f}s)")
    print(resp.json()["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()

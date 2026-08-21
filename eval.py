#!/usr/bin/env python3
"""Send a saved alert JSON to the vision endpoint and print the response.

Usage:
    python eval.py data/alerts/20240101_120000_score7.json
    python eval.py data/eval/          # replay all, print old-vs-new score matrix
    python eval.py data/eval/ --parallel 4  # ...with 4 clips in flight at once
    python eval.py data/eval/ --watch  # loop forever, process only new files
    python eval.py data/alerts/20240101_120000_score7.json --model my-model
    python eval.py data/alerts/20240101_120000_score7.json --url http://localhost:8000/v1
"""

import argparse
import json
import os
import re
import statistics
import sys
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml


def _reasoning_fragment(base_url: str, budget: int | None) -> dict:
    """Request-body fragment controlling reasoning, mirroring LLMClient.

    `budget is None` suppresses reasoning (the default, matching _no_reasoning).
    `budget == 0` enables reasoning with no cap; a positive `budget` applies a
    hard thinking-token cap. The switch is endpoint-specific — OpenRouter reads a
    nested `reasoning` object, vLLM reads `chat_template_kwargs` plus a top-level
    `thinking_token_budget` — and each endpoint ignores the key that doesn't
    apply rather than erroring, so the wrong one fails open.
    """
    is_openrouter = "openrouter.ai" in base_url
    if budget is None:
        if is_openrouter:
            return {"reasoning": {"effort": "none", "exclude": True}}
        return {"chat_template_kwargs": {"enable_thinking": False}}
    if is_openrouter:
        return {"reasoning": {"max_tokens": budget} if budget > 0 else {"enabled": True}}
    fragment: dict = {"chat_template_kwargs": {"enable_thinking": True}}
    if budget > 0:
        fragment["thinking_token_budget"] = budget
    return fragment


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
    # `is None` rather than `or`: temperature 0 is both a valid override and falsy,
    # and omitting the key entirely would silently fall back to the server default
    # (1.0 on vLLM) rather than to what production actually sends.
    temperature = args.temperature if args.temperature is not None else llm.get("temperature", 0.0)
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens or llm.get("max_tokens", 1024),
        "temperature": temperature,
        **_reasoning_fragment(base_url, args.reasoning),
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


_LABEL_SUFFIX = ".label.json"


def _clip_jsons(directory: Path) -> list[Path]:
    """Clip payloads in `directory`, excluding the `.label.json` sidecars ./review writes."""
    return sorted(p for p in directory.glob("*.json") if not p.name.endswith(_LABEL_SUFFIX))


def _load_label(clip: Path) -> dict | None:
    """The human label ./review wrote beside a reviewed clip, if there is one."""
    sidecar = clip.with_name(clip.stem + _LABEL_SUFFIX)
    if not sidecar.exists():
        return None
    try:
        return json.loads(sidecar.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"warning: could not read label {sidecar.name}", file=sys.stderr)
        return None


def _confusion(pairs: list[tuple[int, float]], threshold: int, tolerance: int) -> tuple[int, int, int, int]:
    """Alert-decision counts, forgiving disagreements within `tolerance`.

    A clip is an error only when model and human land on opposite sides of the
    threshold AND differ by more than `tolerance`. Forgiven clips are credited
    according to the human label. With tolerance 0 this is the strict matrix.

    The rationale for forgiving at all is label noise: a human choosing 4 vs 5 is
    near-arbitrary given the category boundaries, so a one-point straddle measures
    the labeller, not the model. The cost is a real blind spot — see the caller.
    """
    tp = fp = fn = tn = 0
    for t, n in pairs:
        t_alert, n_alert = t >= threshold, n >= threshold
        if t_alert == n_alert or abs(t - n) <= tolerance:
            tp += t_alert
            tn += not t_alert
        elif n_alert:
            fp += 1
        else:
            fn += 1
    return tp, fp, fn, tn


def _print_accuracy(pairs: list[tuple[int, float]], threshold: int, tolerance: int) -> None:
    """Alerting accuracy against human labels — the only view that measures correctness.

    The old/new matrix compares one model run against another, so it shows drift,
    not quality. These counts need `./review` labels to exist.
    """
    exact = sum(1 for t, n in pairs if round(n) == t)
    near = sum(1 for t, n in pairs if abs(t - n) <= tolerance)

    def pct(num: int, den: int) -> str:
        return f"{num / den:.1%}" if den else "n/a"

    print(f"── Against {len(pairs)} human label(s), alert threshold {threshold} ──")
    rows = [("strict", 0)] + ([(f"±{tolerance}", tolerance)] if tolerance else [])
    for label, tol in rows:
        tp, fp, fn, tn = _confusion(pairs, threshold, tol)
        print(
            f"  {label:6} {_GREEN}TP{tp:4}{_RESET} {_RED}FP{fp:4}{_RESET} "
            f"{_RED}FN{fn:4}{_RESET} {_GREEN}TN{tn:4}{_RESET}   "
            f"precision {pct(tp, tp + fp):>6}   recall {pct(tp, tp + fn):>6}"
        )
    print(f"  score within ±{tolerance}: {pct(near, len(pairs))}   exact: {pct(exact, len(pairs))}")

    # A label adjacent to the threshold is forgiven whenever the model lands just
    # across the line — the nearest opposite-side score is within tolerance of it.
    # It can still register an error if the model overshoots further.
    straddle = sum(
        1 for t, _ in pairs
        if (threshold - t if t < threshold else t - (threshold - 1)) <= tolerance
    )
    if tolerance and straddle:
        print(
            f"  {_YELLOW}note: {straddle}/{len(pairs)} label(s) sit beside the threshold, so a score "
            f"one step across the line is forgiven for them{_RESET}"
        )


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


def _write_state(state: Path, data: dict) -> None:
    """Replace `state` atomically, so an interrupt can never leave it truncated.

    `Path.write_text` opens with O_TRUNC and only then writes: dying in that
    window leaves a zero-byte file and loses every clip recorded so far. Watch
    mode calls this after every clip and is exited with Ctrl-C, so that window
    was getting hit in practice. Writing a sibling temp file and renaming means a
    reader sees either the whole old file or the whole new one — os.replace is
    atomic within a directory, which is why the temp lives beside the target
    rather than in /tmp.
    """
    tmp = state.with_name(state.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, state)


def _read_state(state: Path) -> dict:
    """Whole state file, or {} if it is missing or unreadable.

    A truncated file used to crash every subsequent run, wedging the directory
    until someone deleted it by hand. Such a file has no recoverable content, so
    treating it as empty costs a re-score of the directory — which is what it
    forced anyway — instead of a traceback.
    """
    if not state.exists():
        return {}
    try:
        return json.loads(state.read_text())
    except (OSError, json.JSONDecodeError):
        print(f"warning: ignoring unreadable state file {state.name}", file=sys.stderr)
        return {}


def _load_processed(directory: Path, model: str) -> dict[str, dict | None]:
    """Return {filename: {"old": n, "new": n} | None} for files already processed with this model."""
    state = _state_file(directory)
    data = _read_state(state)
    if isinstance(data, list):  # migrate old flat list → unknown model
        data = {"": {name: None for name in data}}
        _write_state(state, data)
    elif isinstance(data, dict) and any(isinstance(v, list) for v in data.values()):
        # migrate intermediate format {model: [filenames]}
        data = {m: {name: None for name in files} for m, files in data.items()}
        _write_state(state, data)
    return data.get(model, {})


def _save_processed(directory: Path, model: str, processed: dict[str, dict | None]) -> None:
    state = _state_file(directory)
    data = _read_state(state)
    if isinstance(data, list):
        data = {"": {name: None for name in data}}
    data[model] = processed
    _write_state(state, data)


def _process_file(
    path: Path, config: dict, args: argparse.Namespace
) -> tuple[int, int, int | None, list[int], str, str, float]:
    """Score `path`, repeating the call `args.repeat` times.

    Greedy decoding is not reproducible on this stack (prefix caching, fp8 KV,
    NVFP4 weights, batch-dependent kernel reductions), so a near-tied clip can
    land on a different score run to run. Repeating and averaging makes a
    comparison stable; `runs` keeps the spread so unstable clips stay visible.

    The mean is deliberately not rounded: it keeps sub-integer resolution, so a
    prompt change that shifts a clip from 5,5,5,6,6 to 5,5,5,5,6 is visible
    instead of being quantised away. It is outlier-sensitive by nature — one wild
    run out of five moves it by a fifth of the gap — so read it alongside the
    `~[..]` spread marker rather than on its own.
    """
    m = re.search(r"_score(\d+)", path.stem)
    old_score = int(m.group(1)) if m else -1
    label = _load_label(path)
    truth = label.get("human_score") if label else None
    user_content = json.loads(path.read_text())
    payload, headers, url = _build_payload(user_content, config, args)

    runs: list[int] = []
    results: list[dict] = []
    total = 0.0
    for _ in range(max(1, args.repeat)):
        result, elapsed = _send(payload, headers, url)
        runs.append(result["score"])
        results.append(result)
        total += elapsed
    new_score = statistics.fmean(runs)
    # Show the text from whichever run landed closest to the mean, so the
    # description on screen belongs to a real response rather than an average.
    rep = min(results, key=lambda r: abs(r["score"] - new_score))
    return old_score, new_score, truth, runs, rep["summary"], rep["description"], total


def _iter_processed(
    files: list[Path], config: dict, args: argparse.Namespace
) -> Iterator[tuple[Path, tuple | None, Exception | None]]:
    """Score `files`, yielding (path, result, error) in `files` order.

    With `--parallel N` the requests overlap, but results are still yielded in
    submission order rather than completion order, so the `[i/N]` progress, the
    matrix and the state file come out the same as a serial run. The cost is
    head-of-line blocking on printing: one slow clip holds back the lines behind
    it even though their requests have already finished.
    """
    workers = max(1, args.parallel)
    if workers == 1:  # no pool at all, so a serial run stays exactly as it was
        for path in files:
            try:
                yield path, _process_file(path, config, args), None
            except Exception as e:
                yield path, None, e
        return
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [(path, pool.submit(_process_file, path, config, args)) for path in files]
        for path, future in futures:
            try:
                yield path, future.result(), None
            except Exception as e:
                yield path, None, e


def _fmt(score: float) -> str:
    """Integers stay integers; averaged scores keep one decimal."""
    return str(int(score)) if float(score).is_integer() else f"{score:.1f}"


def _result_line(
    prefix: str, name: str, scores: dict, summary: str, description: str,
    elapsed: float, tolerance: int,
) -> str:
    old, new, truth = scores["old"], scores["new"], scores.get("truth")
    runs = scores.get("runs") or [new]
    arrow = f"{_delta_color(round(abs(old - new)))} ──► {_RESET}"
    line = f"{prefix}{name}: {_score_color(old)}{old}{_RESET}{arrow}{_score_color(round(new))}{_fmt(new)}{_RESET}"
    if len(set(runs)) > 1:
        line += f" {_YELLOW}~{sorted(set(runs))}{_RESET}"
    if truth is not None:
        ok = abs(truth - new) <= tolerance
        line += f" [{_GREEN}✓{_RESET}]" if ok else f" [{_RED}✗ truth {truth}{_RESET}]"
    return f"{line} - {summary} ({description}) [{elapsed:.2f}s]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a saved alert JSON against the vision endpoint.")
    parser.add_argument("json_file", type=Path, help="Path to an alert .json file or a directory of them")
    parser.add_argument("--url", help="Override vision endpoint URL (base, without /chat/completions)")
    parser.add_argument("--model", help="Override model name")
    parser.add_argument("--token", help="Override bearer token")
    parser.add_argument("--max-tokens", type=int, help="Override max_tokens")
    parser.add_argument("--temperature", type=float, help="Override sampling temperature (default: llm_endpoint.temperature)")
    parser.add_argument(
        "--reasoning", nargs="?", type=int, const=0, default=None, metavar="TOKEN_BUDGET",
        help="Enable reasoning; optionally cap the thinking trace at TOKEN_BUDGET tokens (omit the value for no cap)",
    )
    parser.add_argument("--watch", action="store_true", help="Loop forever, processing new JSON files as they appear (requires a directory)")
    parser.add_argument("--all", dest="run_all", action="store_true", help="Process all files, ignoring previously evaluated ones")
    parser.add_argument("--threshold", type=int, help="Alert threshold for accuracy stats (default: telegram.alert_threshold)")
    parser.add_argument("--repeat", type=int, default=1, metavar="N", help="Score each clip N times and take the median (greedy decoding is not reproducible here)")
    parser.add_argument("--tolerance", type=int, default=1, metavar="N", help="Score within +/-N of the human label counts as correct (default: 1)")
    parser.add_argument("--parallel", type=int, default=1, metavar="N", help="Score up to N clips concurrently (default: 1, serial). Per-clip timings become wall-clock under contention.")
    args = parser.parse_args()

    config_path = Path(__file__).parent / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_name = args.model or config.get("llm_endpoint", {}).get("vision_model", "")
    threshold = args.threshold or config.get("telegram", {}).get("alert_threshold", 4)

    if args.watch:
        if not args.json_file.is_dir():
            parser.error("--watch requires a directory argument")
        directory = args.json_file
        processed = _load_processed(directory, model_name)
        print(f"Watching {directory}  model={model_name}  ({len(processed)} already processed with this model)  Ctrl-C to stop")
        matrix = [[0] * 11 for _ in range(11)]
        try:
            while True:
                new_files = [p for p in _clip_jsons(directory) if processed.get(p.name) is None]
                for path, result, error in _iter_processed(new_files, config, args):
                    scores = None
                    if error is not None:
                        print(f"{path.name}  →  ERROR: {error}", file=sys.stderr)
                    else:
                        old_score, new_score, truth, runs, summary, description, elapsed = result
                        scores = {"old": old_score, "new": new_score, "truth": truth, "runs": runs}
                        print(_result_line("", path.name, scores, summary, description, elapsed, args.tolerance))
                        if 0 <= old_score <= 10 and 0 <= new_score <= 10:
                            matrix[old_score][round(new_score)] += 1
                    processed[path.name] = scores
                    _save_processed(directory, model_name, processed)
                time.sleep(_WATCH_INTERVAL)
        except KeyboardInterrupt:
            print()
            _print_matrix(matrix)
            labelled = [
                (s["truth"], s["new"]) for s in processed.values()
                if s and s.get("truth") is not None
            ]
            if labelled:
                print()
                _print_accuracy(labelled, threshold, args.tolerance)
        return

    if args.json_file.is_dir():
        directory = args.json_file
        processed = _load_processed(directory, model_name)
        all_files = _clip_jsons(directory)
        files = all_files if args.run_all else [f for f in all_files if processed.get(f.name) is None]
        if not files:
            print("No new JSON files to process (use --all to reprocess already-evaluated files).", file=sys.stderr)
            sys.exit(1)
        skipped = len(all_files) - len(files)
        if skipped:
            print(f"Skipping {skipped} already-evaluated file(s)  (--all to include them)")
        matrix = [[0] * 11 for _ in range(11)]
        for i, (path, result, error) in enumerate(_iter_processed(files, config, args), 1):
            scores = None
            if error is not None:
                print(f"[{i}/{len(files)}] {path.name}  →  ERROR: {error}", file=sys.stderr)
            else:
                old_score, new_score, truth, runs, summary, description, elapsed = result
                scores = {"old": old_score, "new": new_score, "truth": truth, "runs": runs}
                print(_result_line(f"[{i}/{len(files)}] ", path.name, scores, summary, description, elapsed, args.tolerance))
                if 0 <= old_score <= 10 and 0 <= new_score <= 10:
                    matrix[old_score][round(new_score)] += 1
            processed[path.name] = scores
            _save_processed(directory, model_name, processed)
        print()
        print(f"── Current run ({len(files)} file(s)) ──")
        _print_matrix(matrix)

        cumulative_matrix = [[0] * 11 for _ in range(11)]
        for scores in processed.values():
            if scores and 0 <= scores["old"] <= 10 and 0 <= scores["new"] <= 10:
                cumulative_matrix[scores["old"]][round(scores["new"])] += 1
        cumulative_total = sum(sum(row) for row in cumulative_matrix)
        if cumulative_total > sum(sum(row) for row in matrix):
            print()
            print(f"── All {cumulative_total} scored file(s) in folder ──")
            _print_matrix(cumulative_matrix)

        labelled = [
            (s["truth"], s["new"]) for s in processed.values()
            if s and s.get("truth") is not None
        ]
        if labelled:
            print()
            _print_accuracy(labelled, threshold, args.tolerance)

        unstable = [s for s in processed.values() if s and len(set(s.get("runs") or [])) > 1]
        if unstable:
            print(f"  {_YELLOW}{len(unstable)} clip(s) scored inconsistently across repeats{_RESET}")
        return

    user_content = json.loads(args.json_file.read_text())
    payload, headers, url = _build_payload(user_content, config, args)
    print(f"POST {url}  model={payload['model']}  temperature={payload['temperature']}")
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

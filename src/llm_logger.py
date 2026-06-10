import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

from llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SUMMARIZE_PROMPT_PATH = _PROMPTS_DIR / "summarize_prompt.txt"
_QUERY_PROMPT_PATH = _PROMPTS_DIR / "query_prompt.txt"


def _time_weighted_avg_score(entries: list[dict]) -> float:
    if len(entries) == 1:
        return float(entries[0]["score"])
    try:
        times = [datetime.fromisoformat(e["time"]) for e in entries]
    except Exception:
        return sum(e["score"] for e in entries) / len(entries)
    # Each entry's weight is the gap to the next; last entry reuses the previous gap.
    gaps = [(times[i + 1] - times[i]).total_seconds() for i in range(len(times) - 1)]
    gaps.append(gaps[-1])
    total = sum(gaps)
    return sum(e["score"] * g for e, g in zip(entries, gaps)) / total



class LLMOutputLogger:
    def __init__(
        self,
        data_dir: Path,
        llm_client: LLMClient,
        dog_name: str = "the dog",
        retention_hours: int = 48,
        query_model: str | None = None,
    ):
        self._dir = data_dir / "llm_outputs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._llm_client = llm_client
        self._dog_name = dog_name
        self._retention_hours = retention_hours
        self._query_model = query_model
        self._lock = threading.Lock()
        self._buffer: list[dict] = []
        self._prev_entries: list[dict] = []
        self._history: dict[int, list[dict]] = {}  # chat_id → [{time, user, assistant}]
        self._current_minute: datetime | None = None  # truncated to the minute
        self._last_cleanup = datetime.now()
        self._cleanup()

    def set_query_model(self, model: str | None) -> None:
        with self._lock:
            self._query_model = model

    def log(
        self,
        result_time: datetime,
        score: int,
        summary: str,
        description: str,
        inference_time: float,
        cameras: list[str],
        detected_by: str,
    ) -> None:
        entry = {
            "time": result_time.astimezone().isoformat(timespec="seconds"),
            "score": score,
            "summary": summary,
            "description": description,
            "inference_time": round(inference_time, 2),
            "cameras": cameras,
            "detected_by": detected_by,
        }
        snapshot = None
        prev_snapshot: list[dict] = []
        flushed_minute: datetime | None = None
        do_cleanup = False

        with self._lock:
            entry_minute = result_time.replace(second=0, microsecond=0)

            if self._current_minute is None:
                self._current_minute = entry_minute

            if entry_minute != self._current_minute:
                # Clock minute rolled over — flush the completed previous minute
                if self._buffer:
                    snapshot = self._buffer[:]
                    flushed_minute = self._current_minute
                    prev_snapshot = self._prev_entries[:]
                    self._prev_entries = snapshot
                else:
                    self._prev_entries = []
                self._buffer.clear()
                self._current_minute = entry_minute

            self._buffer.append(entry)

            now = datetime.now()
            if (now - self._last_cleanup) >= timedelta(hours=1):
                self._last_cleanup = now
                do_cleanup = True

        if snapshot and flushed_minute:
            threading.Thread(
                target=self._summarize_and_write,
                args=(snapshot, flushed_minute, prev_snapshot),
                daemon=True,
                name="llm-logger-summarize",
            ).start()
        if do_cleanup:
            self._cleanup()

    def _summarize_and_write(self, entries: list[dict], minute: datetime, prev_entries: list[dict] | None = None) -> None:
        period_start = entries[0]["time"]
        period_end = entries[-1]["time"]

        try:
            start_dt = datetime.fromisoformat(period_start)
            end_dt = datetime.fromisoformat(period_end)
            duration = f"{int((end_dt - start_dt).total_seconds())}s"
        except Exception:
            duration = "unknown"

        all_entries = (prev_entries or []) + entries
        obs_text = "\n".join(
            f"- [{e['time']}] score={e['score']}: {e.get('summary') or ''} — {e['description']}"
            for e in all_entries
        )
        prompt = _SUMMARIZE_PROMPT_PATH.read_text().format(
            count=len(all_entries),
            start=all_entries[0]["time"],
            end=period_end,
            duration=duration,
            observations=obs_text,
        )
        try:
            raw = (self._llm_client.summarize(prompt) or "").strip()
            summary_text = raw if raw and raw.lower() != "null" else entries[-1]["description"]
        except Exception:
            logger.exception("LLM summarization failed, falling back to last description")
            summary_text = entries[-1]["description"]

        score = round(_time_weighted_avg_score(entries), 2)
        avg_inference_seconds = round(sum(e["inference_time"] for e in entries) / len(entries), 2)
        all_cameras = sorted({c for e in entries for c in e["cameras"]})

        record = {
            "date": minute.strftime("%Y-%m-%d"),
            "time": minute.strftime("%H:%M"),
            "entry_count": len(entries),
            "peak_score": max(e["score"] for e in entries),
            "score": score,
            "avg_inference_seconds": avg_inference_seconds,
            "cameras": all_cameras,
            "summary": summary_text,
        }
        path = self._dir / f"{minute.strftime('%Y-%m-%d')}.jsonl"
        with self._lock:
            try:
                with open(path, "a") as f:
                    f.write(json.dumps(record) + "\n")
            except Exception:
                logger.exception("Failed to write LLM output summary")

    def query(self, question: str, chat_id: int | None = None) -> str:
        cutoff = (datetime.now() - timedelta(hours=self._retention_hours)).date()
        records: list[dict] = []
        for f in sorted(self._dir.glob("*.jsonl")):
            try:
                if datetime.strptime(f.stem, "%Y-%m-%d").date() < cutoff:
                    continue
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if line:
                            records.append(json.loads(line))
            except Exception:
                logger.exception("Failed to read log file %s", f)

        with self._lock:
            buffer_snapshot = self._buffer[:]

        records.sort(key=lambda r: (r.get("date", ""), r.get("time", "")))

        records_text = "\n".join(
            f"[{r['date']} {r['time']}] score={r.get('score')}/{r.get('peak_score')} "
            f"cameras={','.join(r.get('cameras', []))}: {r.get('summary', '')}"
            for r in records
        )
        if buffer_snapshot:
            buffer_lines = "\n".join(
                f"[{e['time']}] score={e['score']} cameras={','.join(e.get('cameras', []))}: {e.get('description', '')}"
                for e in buffer_snapshot
            )
            records_text = (records_text + "\n\n(current minute, not yet summarized)\n" + buffer_lines).lstrip("\n")

        if not records_text.strip():
            return "No data available."

        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        preamble = _QUERY_PROMPT_PATH.read_text().format(
            dog_name=self._dog_name, now=now_str, records=records_text
        )

        # Prune history: keep last 5 entries within the last 15 minutes
        history_cutoff = datetime.now() - timedelta(minutes=15)
        history = [
            e for e in self._history.get(chat_id or 0, [])
            if e["time"] >= history_cutoff
        ][-20:]

        # Build multi-turn messages: preamble attached to oldest turn (or current question)
        if history:
            messages = [{"role": "user", "content": preamble + "\n\n" + history[0]["user"]}]
            messages.append({"role": "assistant", "content": history[0]["assistant"]})
            for entry in history[1:]:
                messages.append({"role": "user", "content": entry["user"]})
                messages.append({"role": "assistant", "content": entry["assistant"]})
            messages.append({"role": "user", "content": question})
        else:
            messages = [{"role": "user", "content": preamble + "\n\n" + question}]

        answer = self._llm_client.summarize(
            messages=messages, max_tokens=500, model=self._query_model, query=True
        )

        if chat_id is not None:
            history.append({"time": datetime.now(), "user": question, "assistant": answer})
            self._history[chat_id] = history

        return answer

    def _cleanup(self) -> None:
        cutoff = (datetime.now() - timedelta(hours=self._retention_hours)).date()
        for f in self._dir.glob("*.jsonl"):
            try:
                if datetime.strptime(f.stem, "%Y-%m-%d").date() < cutoff:
                    f.unlink()
                    logger.info("Removed old LLM output log: %s", f.name)
            except (ValueError, OSError):
                pass

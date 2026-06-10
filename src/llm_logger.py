import json
import logging
import re
import threading
from datetime import datetime, timedelta
from pathlib import Path

from llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SUMMARIZE_PROMPT_PATH = _PROMPTS_DIR / "summarize_prompt.txt"
_QUERY_PROMPT_PATH = _PROMPTS_DIR / "query_prompt.txt"
_TRIAGE_PROMPT_PATH = _PROMPTS_DIR / "triage_prompt.txt"


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
        summary_window_minutes: int = 10,
    ):
        self._dir = data_dir / "llm_outputs"
        self._dir.mkdir(parents=True, exist_ok=True)
        self._llm_client = llm_client
        self._dog_name = dog_name
        self._retention_hours = retention_hours
        self._summary_window_minutes = summary_window_minutes
        self._lock = threading.Lock()
        self._buffer_path = self._dir / "buffer.jsonl"
        self._window_buffer: dict[datetime, list[dict]] = {}  # minute -> raw entries
        self._history: dict[int, list[dict]] = {}  # chat_id → [{time, user, assistant}]
        self._last_cleanup = datetime.now()
        self._cleanup()
        self._load_buffer()

    def _load_buffer(self) -> None:
        if not self._buffer_path.exists():
            return
        try:
            with open(self._buffer_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    e = json.loads(line)
                    w = datetime.fromisoformat(e["time"]).replace(second=0, microsecond=0)
                    self._window_buffer.setdefault(w, []).append(e)
            total = sum(len(v) for v in self._window_buffer.values())
            logger.info("Loaded %d buffered entries from disk", total)
        except Exception:
            logger.exception("Failed to load buffer file")

    def _rewrite_buffer_file(self) -> None:
        entries = sorted(
            (e for es in self._window_buffer.values() for e in es),
            key=lambda e: e["time"],
        )
        try:
            with open(self._buffer_path, "w") as f:
                for e in entries:
                    f.write(json.dumps(e) + "\n")
        except Exception:
            logger.exception("Failed to rewrite buffer file")

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
        to_flush: list[tuple[list[dict], datetime, list[dict]]] = []
        do_cleanup = False

        with self._lock:
            entry_window = result_time.replace(second=0, microsecond=0)
            self._window_buffer.setdefault(entry_window, []).append(entry)
            try:
                with open(self._buffer_path, "a") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                logger.exception("Failed to append to buffer file")

            # Windows older than summary_window_minutes ago are ready to compact.
            # Their context is whatever is still in the realtime window (windows after them).
            ready_cutoff = entry_window - timedelta(minutes=self._summary_window_minutes)
            for w in sorted(w for w in self._window_buffer if w < ready_cutoff):
                target = self._window_buffer.pop(w)
                context = sorted(
                    (e for cw, ces in self._window_buffer.items() if cw > w for e in ces),
                    key=lambda e: e["time"],
                )
                to_flush.append((target, w, context))

            now = datetime.now()
            if (now - self._last_cleanup) >= timedelta(hours=1):
                self._last_cleanup = now
                do_cleanup = True

        for target, w, context in to_flush:
            threading.Thread(
                target=self._summarize_and_write,
                args=(target, w, context),
                daemon=True,
                name="llm-logger-summarize",
            ).start()
        if do_cleanup:
            self._cleanup()

    def _summarize_and_write(self, entries: list[dict], minute: datetime, context_entries: list[dict] | None = None) -> None:
        period_start = entries[0]["time"]
        period_end = entries[-1]["time"]

        try:
            start_dt = datetime.fromisoformat(period_start)
            end_dt = datetime.fromisoformat(period_end)
            duration = f"{int((end_dt - start_dt).total_seconds())}s"
        except Exception:
            duration = "unknown"

        # entries = the minute being summarized; context_entries = the realtime window that follows
        all_entries = entries + (context_entries or [])
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
            raw = (self._llm_client.summarize(prompt, model=self._llm_client.fast_model) or "").strip()
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
            self._rewrite_buffer_file()

    _TRIAGE_FALLBACK_MINUTES = 60

    def _triage_minutes(self, question: str) -> int:
        max_minutes = self._retention_hours * 60
        try:
            prompt = _TRIAGE_PROMPT_PATH.read_text().format(
                max_minutes=max_minutes, question=question
            )
            raw = self._llm_client.summarize(prompt, model=self._llm_client.fast_model, max_tokens=1024)
            match = re.search(r"\d+", raw or "")
            if not match:
                logger.info("Query triage: no number in response, using %d min fallback", self._TRIAGE_FALLBACK_MINUTES)
                return self._TRIAGE_FALLBACK_MINUTES
            minutes = min(int(match.group()), max_minutes)
            logger.info("Query triage: %d min", minutes)
            return minutes
        except Exception:
            logger.exception("Triage failed, using %d min fallback", self._TRIAGE_FALLBACK_MINUTES)
            return self._TRIAGE_FALLBACK_MINUTES

    def query(self, question: str, chat_id: int | None = None) -> str:
        minutes = self._triage_minutes(question)
        cutoff_dt = datetime.now() - timedelta(minutes=minutes)
        cutoff_date = cutoff_dt.date()
        cutoff_hhmm = cutoff_dt.strftime("%H:%M")
        records: list[dict] = []
        for f in sorted(self._dir.glob("*.jsonl")):
            try:
                file_date = datetime.strptime(f.stem, "%Y-%m-%d").date()
                if file_date < cutoff_date:
                    continue
                with open(f) as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        r = json.loads(line)
                        if file_date == cutoff_date and r.get("time", "") < cutoff_hhmm:
                            continue
                        records.append(r)
            except Exception:
                logger.exception("Failed to read log file %s", f)

        with self._lock:
            buffer_snapshot = sorted(
                (e for entries in self._window_buffer.values() for e in entries),
                key=lambda e: e["time"],
            )

        records.sort(key=lambda r: (r.get("date", ""), r.get("time", "")))
        logger.info("Query context: %d records + %d buffer entries", len(records), len(buffer_snapshot))

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
            messages=messages, max_tokens=500, model=self._llm_client.memory_model, endpoint="memory"
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

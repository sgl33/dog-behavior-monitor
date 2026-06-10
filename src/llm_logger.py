import json
import logging
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path

from llm import LLMClient

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_SUMMARIZE_PROMPT_PATH = _PROMPTS_DIR / "summarize_prompt.txt"
_BATCH_SUMMARIZE_PROMPT_PATH = _PROMPTS_DIR / "batch_summarize_prompt.txt"
_QUERY_PROMPT_PATH = _PROMPTS_DIR / "query_prompt.txt"
_TRIAGE_PROMPT_PATH = _PROMPTS_DIR / "triage_prompt.txt"


@dataclass
class LogEntry:
    time: str
    score: int
    summary: str
    description: str
    inference_time: float
    cameras: list[str]
    detected_by: str

    @classmethod
    def from_dict(cls, d: dict) -> "LogEntry":
        return cls(
            time=d["time"],
            score=d["score"],
            summary=d.get("summary") or "",
            description=d["description"],
            inference_time=d["inference_time"],
            cameras=d["cameras"],
            detected_by=d["detected_by"],
        )


@dataclass
class SummaryRecord:
    date: str
    time: str
    entry_count: int
    peak_score: int
    score: float
    avg_inference_seconds: float
    cameras: list[str]
    summary: str

    @classmethod
    def from_dict(cls, d: dict) -> "SummaryRecord":
        return cls(
            date=d["date"],
            time=d["time"],
            entry_count=d["entry_count"],
            peak_score=d["peak_score"],
            score=d["score"],
            avg_inference_seconds=d["avg_inference_seconds"],
            cameras=d["cameras"],
            summary=d.get("summary") or "",
        )


@dataclass
class ChatTurn:
    time: datetime
    question: str
    response: str


def _time_weighted_avg_score(entries: list[LogEntry]) -> float:
    """
    Calculates time-weighted average score.
    Equation: `weighted_avg = sum(gap * score) / (len(score) * sum(gap))`
    """
    if len(entries) == 1:
        return float(entries[0].score)

    try:
        times = [datetime.fromisoformat(e.time) for e in entries]
    except Exception:
        return sum(e.score for e in entries) / len(entries)

    gaps = [
        (times[i + 1] - times[i]).total_seconds()
        for i in range(len(times) - 1)
    ]
    gaps.append(gaps[-1])
    return sum(e.score * g for e, g in zip(entries, gaps)) / sum(gaps)


class LLMOutputLogger:
    """
    Responsible for keeping track of LLM output data for querying:
    - All records for the last 10 minutes are kept individually ("buffer")
    - Every "minute" is compacted into one sentence after 10 minutes using LLM
    """

    _CHAT_HISTORY_MAX_TURNS = 20
    _CHAT_HISTORY_WINDOW_MINUTES = 15
    _SUMMARIZE_BATCH_SIZE = 5

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
        self._window_buffer: dict[datetime, list[LogEntry]] = {}  # minute -> raw entries
        self._history: dict[int, list[ChatTurn]] = {}  # chat_id → turns
        self._last_cleanup = datetime.now()
        self._cleanup()
        self._load_buffer()

    def _load_buffer(self) -> None:
        """
        Read `data/llm_outputs/buffer.jsonl` into `self._window_buffer`.
        """
        if not self._buffer_path.exists():
            return
        try:
            with open(self._buffer_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = LogEntry.from_dict(json.loads(line))
                    time_minute = datetime.fromisoformat(entry.time).replace(
                        second=0, microsecond=0
                    )
                    self._window_buffer.setdefault(time_minute, []).append(entry)
            total = sum(len(v) for v in self._window_buffer.values())
            logger.info("Loaded %d buffered entries from disk", total)
        except Exception:
            logger.exception("Failed to load buffer file")

    def _rewrite_buffer_file(self) -> None:
        """
        Write `self._window_buffer` to `data/llm_outputs/buffer.jsonl`,
        overwriting any existing data in the file.
        """
        entries = sorted(
            (e for es in self._window_buffer.values() for e in es),
            key=lambda e: e.time,
        )
        try:
            with open(self._buffer_path, "w") as f:
                for e in entries:
                    f.write(json.dumps(asdict(e)) + "\n")
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
        """
        Add a log for a single LLM inference output.

        Args:
            result_time (datetime): time of inference
            score (int): range 0-10
            summary (str): a few words
            description (str): 1-2 sentences
            inference_time (float): how long LLM inference took, in seconds
            cameras (list[str]): camera IDs the dog was detected in
            detected_by (str): `YOLO` or `LLM`
        """
        entry = LogEntry(
            time=result_time.astimezone().isoformat(timespec="seconds"),
            score=score,
            summary=summary,
            description=description,
            inference_time=round(inference_time, 2),
            cameras=cameras,
            detected_by=detected_by,
        )
        entry_window = result_time.replace(second=0, microsecond=0)

        # Append to memory and file
        with self._lock:
            self._window_buffer.setdefault(entry_window, []).append(entry)
            try:
                with open(self._buffer_path, "a") as f:
                    f.write(json.dumps(asdict(entry)) + "\n")
            except Exception:
                logger.exception("Failed to append to buffer file")

        self._compact_applicable_logs(entry_window)

    def _compact_applicable_logs(self, entry_window: datetime) -> None:
        """
        Identify logs in the buffer to compact, and compact them in batches of
        _SUMMARIZE_BATCH_SIZE using the fast LLM. Partial batches (< batch size)
        remain in the buffer until enough minutes accumulate.
        """
        batches: list[list[tuple[list[LogEntry], datetime, list[LogEntry]]]] = []
        do_cleanup = False

        with self._lock:
            ready_cutoff = entry_window - timedelta(minutes=self._summary_window_minutes)
            ready_windows = sorted(ws for ws in self._window_buffer if ws < ready_cutoff)

            # Only process complete batches; leave the remainder in the buffer
            n_complete = (len(ready_windows) // self._SUMMARIZE_BATCH_SIZE) * self._SUMMARIZE_BATCH_SIZE
            context_entries = sorted(
                (e for ws, entries in self._window_buffer.items() if ws >= ready_cutoff for e in entries),
                key=lambda e: e.time,
            )
            for i in range(0, n_complete, self._SUMMARIZE_BATCH_SIZE):
                batch = [
                    (self._window_buffer.pop(ws), ws, context_entries)
                    for ws in ready_windows[i:i + self._SUMMARIZE_BATCH_SIZE]
                ]
                batches.append(batch)

            now = datetime.now()
            if (now - self._last_cleanup) >= timedelta(hours=1):
                self._last_cleanup = now
                do_cleanup = True

        for batch in batches:
            threading.Thread(
                target=self._summarize_batch_and_write,
                args=(batch,),
                daemon=True,
                name="llm-logger-summarize",
            ).start()
        if do_cleanup:
            self._cleanup()

    def _summarize_batch_and_write(
        self,
        batch: list[tuple[list[LogEntry], datetime, list[LogEntry]]],
    ) -> None:
        context_entries = batch[0][2]

        # Build the windows block for the prompt
        window_blocks: list[str] = []
        for idx, (entries, minute, _) in enumerate(batch):
            obs = "\n".join(
                f"- [{e.time}] score={e.score}: {e.summary} — {e.description}"
                for e in entries
            )
            window_blocks.append(f"[{minute.strftime('%H:%M')} ({len(entries)} obs)]\n{obs}")
        windows_text = "\n\n".join(window_blocks)
        if context_entries:
            ctx = "\n".join(
                f"- [{e.time}] score={e.score}: {e.summary} — {e.description}"
                for e in context_entries
            )
            windows_text += f"\n\n[Context — what happened after, for reference]\n{ctx}"

        prompt = _BATCH_SUMMARIZE_PROMPT_PATH.read_text().format(
            batch_size=len(batch),
            windows=windows_text,
        )

        try:
            raw = (self._llm_client.summarize(prompt, model=self._llm_client.fast_model) or "").strip()
            parts = [p.strip() for p in raw.split("---") if p.strip()]
        except Exception:
            logger.exception("LLM batch summarization failed")
            parts = []

        records: list[tuple[SummaryRecord, datetime]] = []
        for idx, (entries, minute, _) in enumerate(batch):
            summary_text = (
                parts[idx]
                if idx < len(parts) and parts[idx] and parts[idx].lower() != "null"
                else entries[-1].description
            )
            records.append((SummaryRecord(
                date=minute.strftime("%Y-%m-%d"),
                time=minute.strftime("%H:%M"),
                entry_count=len(entries),
                peak_score=max(e.score for e in entries),
                score=round(_time_weighted_avg_score(entries), 2),
                avg_inference_seconds=round(sum(e.inference_time for e in entries) / len(entries), 2),
                cameras=sorted({c for e in entries for c in e.cameras}),
                summary=summary_text,
            ), minute))

        with self._lock:
            for record, minute in records:
                path = self._dir / f"{minute.strftime('%Y-%m-%d')}.jsonl"
                try:
                    with open(path, "a") as f:
                        f.write(json.dumps(asdict(record)) + "\n")
                except Exception:
                    logger.exception("Failed to write summary for minute %s", minute)
            self._rewrite_buffer_file()

    def _triage_minutes(self, question: str) -> int:
        """
        Get an estimate of how many minutes of data we'll need for a given
        question using the fast LLM.

        Args:
            question (str): user question

        Returns:
            int: number of minutes, range [0, self._retention_hours * 60],
                given by the fast LLM. Defaults to max if failed.
        """
        max_minutes = self._retention_hours * 60
        try:
            prompt = _TRIAGE_PROMPT_PATH.read_text().format(
                max_minutes=max_minutes, question=question
            )
            raw = self._llm_client.summarize(prompt, model=self._llm_client.fast_model, max_tokens=1024)
            match = re.search(r"\d+", raw or "")
            if not match:
                logger.info(
                    "Query triage: no number in response, using %d min fallback",
                    max_minutes
                )
                return max_minutes
            minutes = min(int(match.group()), max_minutes)
            logger.info("Query triage: %d min", minutes)
            return minutes
        except Exception:
            logger.exception(
                "Triage failed, using %d min fallback", max_minutes
            )
            return max_minutes

    def _read_summary_records(self, cutoff_date, cutoff_hhmm: str) -> list[SummaryRecord]:
        records: list[SummaryRecord] = []
        for filepath in sorted(self._dir.glob("*.jsonl")):
            try:
                file_date = datetime.strptime(filepath.stem, "%Y-%m-%d").date()
                if file_date < cutoff_date:
                    continue
                with open(filepath) as file:
                    for line in file:
                        line = line.strip()
                        if not line:
                            continue
                        record = SummaryRecord.from_dict(json.loads(line))
                        if file_date == cutoff_date and record.time < cutoff_hhmm:
                            continue
                        records.append(record)
            except Exception:
                logger.exception("Failed to read log file %s", filepath)
        records.sort(key=lambda r: (r.date, r.time))
        return records

    def _read_buffer_snapshot(self) -> list[LogEntry]:
        with self._lock:
            return sorted(
                (e for entries in self._window_buffer.values() for e in entries),
                key=lambda e: e.time,
            )

    @staticmethod
    def _stringify_records(
        records: list[SummaryRecord], buffer: list[LogEntry]
    ) -> str:
        text = "\n".join(
            f"[{r.date} {r.time}] score={r.score}/{r.peak_score} "
            f"cameras={','.join(r.cameras)}: {r.summary}"
            for r in records
        )
        if buffer:
            buffer_lines = "\n".join(
                f"[{e.time}] score={e.score} cameras={','.join(e.cameras)}: {e.description}"
                for e in buffer
            )
            text = (text + "\n\n(current minute, not yet summarized)\n" + buffer_lines).lstrip("\n")
        return text

    def _build_messages(
        self, preamble: str, question: str, chat_id: int | None
    ) -> tuple[list[dict], list[ChatTurn]]:
        history_cutoff = datetime.now() - timedelta(minutes=self._CHAT_HISTORY_WINDOW_MINUTES)
        history = [
            turn for turn in self._history.get(chat_id or 0, [])
            if turn.time >= history_cutoff
        ][-self._CHAT_HISTORY_MAX_TURNS:]
        if not history:
            return [{"role": "user", "content": preamble + "\n\n" + question}], history
        messages = [{"role": "user", "content": preamble + "\n\n" + history[0].question}]
        messages.append({"role": "assistant", "content": history[0].response})
        for turn in history[1:]:
            messages.append({"role": "user", "content": turn.question})
            messages.append({"role": "assistant", "content": turn.response})
        messages.append({"role": "user", "content": question})
        return messages, history

    def query(self, question: str, chat_id: int | None = None) -> str:

        # Estimate how much data to include
        minutes = self._triage_minutes(question)
        cutoff_dt = datetime.now() - timedelta(minutes=minutes)
        cutoff_date = cutoff_dt.date()
        cutoff_hhmm = cutoff_dt.strftime("%H:%M")
        
        # Prepare data for prompt
        records = self._read_summary_records(cutoff_date, cutoff_hhmm)
        buffer_snapshot = self._read_buffer_snapshot()
        logger.info(
            "Query context: %d records + %d buffer entries",
            len(records), len(buffer_snapshot)
        )
        records_text = self._stringify_records(records, buffer_snapshot)
        if not records_text.strip():
            return "No data available."

        # Construct prompt
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M %Z")
        preamble = _QUERY_PROMPT_PATH.read_text().format(
            dog_name=self._dog_name, now=now_str, records=records_text
        )
        messages, history = self._build_messages(preamble, question, chat_id)

        # Query LLM
        answer = self._llm_client.summarize(
            messages=messages, 
            max_tokens=500, 
            model=self._llm_client.memory_model, 
            endpoint="memory"
        )
        answer += "\n\n(Disclaimer: this response was generated using AI "
        answer += "and may be inaccurate.)"

        # Add response to history for future reference
        if chat_id is not None:
            history.append(ChatTurn(
                time=datetime.now(), question=question, response=answer
            ))
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

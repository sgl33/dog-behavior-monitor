import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from llm import LLMClient
from llm_logger import LLMOutputLogger, LogEntry, SummaryRecord

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
_QUERY_PROMPT_PATH = _PROMPTS_DIR / "query_prompt.txt"
_TRIAGE_PROMPT_PATH = _PROMPTS_DIR / "triage_prompt.txt"


@dataclass
class ChatTurn:
    time: datetime
    question: str
    response: str


class MemoryQuerier:
    """
    Answers natural-language questions about the dog's recent activity by
    reading summaries/buffer from an `LLMOutputLogger` store and querying the
    memory LLM. Keeps a short per-chat history for follow-up questions.
    """
    _CHAT_HISTORY_MAX_TURNS = 20
    _CHAT_HISTORY_WINDOW_MINUTES = 15

    def __init__(
        self,
        store: LLMOutputLogger,
        llm_client: LLMClient,
        dog_name: str = "the dog",
    ):
        self._store = store
        self._llm_client = llm_client
        self._dog_name = dog_name
        self._history: dict[int, list[ChatTurn]] = {}  # chat_id → turns

    def set_dog_name(self, name: str) -> None:
        self._dog_name = name

    def _triage_minutes(self, question: str) -> int:
        """
        Get an estimate of how many minutes of data we'll need for a given
        question using the fast LLM.

        Args:
            question (str): user question

        Returns:
            int: number of minutes, range [0, store.retention_hours * 60],
                given by the fast LLM. Defaults to max if failed.
        """
        max_minutes = self._store.retention_hours * 60
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
        records = self._store.read_summary_records(cutoff_date, cutoff_hhmm)
        buffer_snapshot = self._store.read_buffer_snapshot()
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

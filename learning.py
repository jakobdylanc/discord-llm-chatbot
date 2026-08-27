from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path

from brain import BrainSnapshot, DQNAgent, Experience, LayerSnapshot
from social import DEFAULT_REPUTATION, update_reputation

REWARD_WINDOW_SECONDS = 150
DEFAULT_COOLDOWN = 20
DEFAULT_BUDGET = 6
HEAT_WINDOW_SECONDS = 300

POSITIVE_EMOJI = frozenset({"👍", "✅", "❤️", "❤", "🔥", "😍"})
NEGATIVE_EMOJI = frozenset({"👎", "❌", "💀", "😡", "🤮"})


class Action(IntEnum):
    STAY = 0
    REPLY = 1
    REACT = 2


@dataclass(frozen=True)
class Decision:
    kind: str
    action: Action | None
    learn: bool
    reason: str


def emoji_score(emoji: str) -> int:
    name = emoji.strip()
    if name in POSITIVE_EMOJI:
        return 1
    if name in NEGATIVE_EMOJI:
        return -1
    return 0


def compute_reward(
    positives: int,
    negatives: int,
    human_reply: bool,
    deleted: bool,
    action: Action,
) -> float:
    if deleted:
        return -2.0
    if positives == 0 and negatives == 0 and not human_reply:
        return -0.2
    score = min(positives, 3) * 1.0 + negatives * -1.0
    if human_reply:
        score += 0.5
    return score


def command_sync_mode(dev_guild_id: str | None, allow_global: bool) -> str:
    if dev_guild_id:
        return "guild"
    if allow_global:
        return "global"
    return "skip"


def decide_action(
    *,
    forced: bool,
    learning: bool,
    act: bool,
    cooldown_ok: bool,
    budget_ok: bool,
    select: Callable[[], int],
) -> Decision:
    if forced:
        return Decision("reply", Action.REPLY, learn=learning, reason="forced")
    if not learning:
        return Decision("ignore", None, False, "learning_off")
    if not act:
        return Decision("ignore", None, False, "act_off")
    chosen = Action(select())
    if chosen is Action.STAY:
        return Decision("stay", Action.STAY, True, "stay")
    if not cooldown_ok:
        return Decision("ignore", None, False, "cooldown")
    if not budget_ok:
        return Decision("ignore", None, False, "over_budget")
    return Decision(chosen.name.lower(), chosen, True, "policy")


def _now() -> datetime:
    return datetime.now(UTC)


def _ts(when: datetime) -> float:
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return when.timestamp()


def _safe_guild(guild_id: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in guild_id)


class LearningStore:
    def __init__(self, data_dir: str | Path, reward_window_seconds: int = REWARD_WINDOW_SECONDS) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        try:
            self.data_dir.chmod(0o700)
        except OSError:
            pass
        self.brains_dir = self.data_dir / "llmcord-brains"
        self.brains_dir.mkdir(parents=True, exist_ok=True)
        self.reward_window_seconds = reward_window_seconds
        self._db = sqlite3.connect(self.data_dir / "llmcord.sqlite")
        self._db.row_factory = sqlite3.Row
        self._init_schema()
        self._brains: dict[str, DQNAgent] = {}

    def close(self) -> None:
        self.save()
        self._db.close()

    def _init_schema(self) -> None:
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS guild_settings (
                guild_id TEXT PRIMARY KEY,
                learning INTEGER NOT NULL DEFAULT 0,
                act INTEGER NOT NULL DEFAULT 0,
                cooldown_seconds INTEGER NOT NULL DEFAULT 20,
                unsolicited_per_hour INTEGER NOT NULL DEFAULT 6
            );
            CREATE TABLE IF NOT EXISTS budget_events (
                guild_id TEXT NOT NULL,
                spent_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS channel_events (
                channel_id TEXT NOT NULL,
                at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS channel_cooldown (
                channel_id TEXT PRIMARY KEY,
                last_unsolicited_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reputation (
                guild_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                value REAL NOT NULL,
                PRIMARY KEY (guild_id, user_id)
            );
            CREATE TABLE IF NOT EXISTS pending (
                message_id TEXT PRIMARY KEY,
                guild_id TEXT NOT NULL,
                action INTEGER NOT NULL,
                state_json TEXT NOT NULL,
                opened_at REAL NOT NULL,
                positives INTEGER NOT NULL DEFAULT 0,
                negatives INTEGER NOT NULL DEFAULT 0,
                human_reply INTEGER NOT NULL DEFAULT 0,
                deleted INTEGER NOT NULL DEFAULT 0
            );
            """
        )
        self._db.commit()

    def _settings(self, guild_id: str) -> sqlite3.Row:
        row = self._db.execute("SELECT * FROM guild_settings WHERE guild_id = ?", (guild_id,)).fetchone()
        if row is None:
            self._db.execute("INSERT INTO guild_settings (guild_id) VALUES (?)", (guild_id,))
            self._db.commit()
            row = self._db.execute("SELECT * FROM guild_settings WHERE guild_id = ?", (guild_id,)).fetchone()
        return row

    def is_learning(self, guild_id: str) -> bool:
        return bool(self._settings(guild_id)["learning"])

    def is_act(self, guild_id: str) -> bool:
        return bool(self._settings(guild_id)["act"])

    def set_learning(self, guild_id: str, enabled: bool) -> None:
        self._settings(guild_id)
        self._db.execute("UPDATE guild_settings SET learning = ? WHERE guild_id = ?", (int(enabled), guild_id))
        self._db.commit()

    def set_act(self, guild_id: str, enabled: bool) -> None:
        self._settings(guild_id)
        self._db.execute("UPDATE guild_settings SET act = ? WHERE guild_id = ?", (int(enabled), guild_id))
        self._db.commit()

    def cooldown_seconds(self, guild_id: str) -> int:
        return int(self._settings(guild_id)["cooldown_seconds"])

    def budget_per_hour(self, guild_id: str) -> int:
        return int(self._settings(guild_id)["unsolicited_per_hour"])

    def reputation(self, guild_id: str, user_id: str) -> float:
        row = self._db.execute(
            "SELECT value FROM reputation WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        ).fetchone()
        return float(row["value"]) if row else DEFAULT_REPUTATION

    def apply_reputation(self, guild_id: str, user_id: str, *, target: float) -> float:
        value = update_reputation(self.reputation(guild_id, user_id), target=target)
        self._db.execute(
            """
            INSERT INTO reputation(guild_id, user_id, value) VALUES(?,?,?)
            ON CONFLICT(guild_id, user_id) DO UPDATE SET value=excluded.value
            """,
            (guild_id, user_id, value),
        )
        self._db.commit()
        return value

    def note_channel_message(self, channel_id: str, when: datetime | None = None) -> None:
        self._db.execute("INSERT INTO channel_events(channel_id, at) VALUES(?, ?)", (channel_id, _ts(when or _now())))
        self._db.commit()

    def channel_heat(self, channel_id: str, when: datetime | None = None) -> int:
        cutoff = _ts(when or _now()) - HEAT_WINDOW_SECONDS
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM channel_events WHERE channel_id = ? AND at >= ?",
            (channel_id, cutoff),
        ).fetchone()
        return int(row["n"])

    def cooldown_ok(self, channel_id: str, guild_id: str, when: datetime | None = None) -> bool:
        row = self._db.execute(
            "SELECT last_unsolicited_at FROM channel_cooldown WHERE channel_id = ?",
            (channel_id,),
        ).fetchone()
        if row is None:
            return True
        return (_ts(when or _now()) - float(row["last_unsolicited_at"])) >= self.cooldown_seconds(guild_id)

    def mark_unsolicited(self, channel_id: str, when: datetime | None = None) -> None:
        self._db.execute(
            """
            INSERT INTO channel_cooldown(channel_id, last_unsolicited_at) VALUES(?, ?)
            ON CONFLICT(channel_id) DO UPDATE SET last_unsolicited_at=excluded.last_unsolicited_at
            """,
            (channel_id, _ts(when or _now())),
        )
        self._db.commit()

    def budget_ok(self, guild_id: str, when: datetime | None = None) -> bool:
        cutoff = _ts(when or _now()) - 3600
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM budget_events WHERE guild_id = ? AND spent_at >= ?",
            (guild_id, cutoff),
        ).fetchone()
        return int(row["n"]) < self.budget_per_hour(guild_id)

    def spend_budget(self, guild_id: str, when: datetime | None = None) -> None:
        self._db.execute("INSERT INTO budget_events(guild_id, spent_at) VALUES(?, ?)", (guild_id, _ts(when or _now())))
        self._db.commit()

    def brain_path(self, guild_id: str) -> Path:
        return self.brains_dir / f"{_safe_guild(guild_id)}.json"

    def brain(self, guild_id: str) -> DQNAgent:
        if guild_id not in self._brains:
            agent = DQNAgent()
            path = self.brain_path(guild_id)
            if path.exists():
                agent.import_weights(_snapshot_from_json(path.read_text(encoding="utf-8")))
            self._brains[guild_id] = agent
        return self._brains[guild_id]

    def save(self) -> None:
        for guild_id, agent in self._brains.items():
            self.brain_path(guild_id).write_text(_snapshot_to_json(agent.export_weights()), encoding="utf-8")

    def open_pending(
        self,
        *,
        guild_id: str,
        message_id: str,
        action: Action,
        state: list[float],
        opened_at: datetime | None = None,
    ) -> None:
        self._db.execute(
            "INSERT OR REPLACE INTO pending(message_id, guild_id, action, state_json, opened_at) VALUES(?,?,?,?,?)",
            (message_id, guild_id, int(action), json.dumps(state), _ts(opened_at or _now())),
        )
        self._db.commit()

    def note_reaction(self, message_id: str, emoji: str) -> None:
        score = emoji_score(emoji)
        if score == 0:
            return
        column = "positives" if score > 0 else "negatives"
        self._db.execute(f"UPDATE pending SET {column} = {column} + 1 WHERE message_id = ?", (message_id,))
        self._db.commit()

    def note_human_reply(self, message_id: str) -> None:
        self._db.execute("UPDATE pending SET human_reply = 1 WHERE message_id = ?", (message_id,))
        self._db.commit()

    def note_deleted(self, message_id: str) -> None:
        self._db.execute("UPDATE pending SET deleted = 1 WHERE message_id = ?", (message_id,))
        self._db.commit()

    def settle_due(self, when: datetime | None = None) -> list[Experience]:
        now_ts = _ts(when or _now())
        cutoff = now_ts - self.reward_window_seconds
        rows = self._db.execute(
            "SELECT * FROM pending WHERE deleted = 1 OR opened_at <= ?",
            (cutoff,),
        ).fetchall()
        settled: list[Experience] = []
        for row in rows:
            action = Action(int(row["action"]))
            reward = compute_reward(
                int(row["positives"]),
                int(row["negatives"]),
                bool(row["human_reply"]),
                bool(row["deleted"]),
                action,
            )
            state = json.loads(row["state_json"])
            exp = Experience(state=state, action=int(action), reward=reward, next_state=list(state), done=True)
            if self.is_learning(row["guild_id"]):
                agent = self.brain(row["guild_id"])
                agent.remember(exp)
                agent.learn()
            settled.append(exp)
            self._db.execute("DELETE FROM pending WHERE message_id = ?", (row["message_id"],))
        self._db.commit()
        if settled:
            self.save()
        return settled

    def remember_stay(self, guild_id: str, state: list[float]) -> None:
        if not self.is_learning(guild_id):
            return
        exp = Experience(state=state, action=int(Action.STAY), reward=0.0, next_state=list(state), done=True)
        agent = self.brain(guild_id)
        agent.remember(exp)
        agent.learn()
        self.save()

    def status(self, guild_id: str) -> dict:
        agent = self.brain(guild_id)
        rewards = agent.recent_rewards
        mean = sum(rewards) / len(rewards) if rewards else 0.0
        return {
            "learning": self.is_learning(guild_id),
            "act": self.is_act(guild_id),
            "topology": list(agent.online.topology),
            "epsilon": agent.epsilon,
            "steps": agent.step_count,
            "replay": len(agent.buffer),
            "last_q": list(agent.last_q),
            "histogram": list(agent.action_histogram),
            "reward_mean": mean,
            "budget_left": max(0, self.budget_per_hour(guild_id) - self._spent_last_hour(guild_id)),
        }

    def _spent_last_hour(self, guild_id: str, when: datetime | None = None) -> int:
        cutoff = _ts(when or _now()) - 3600
        row = self._db.execute(
            "SELECT COUNT(*) AS n FROM budget_events WHERE guild_id = ? AND spent_at >= ?",
            (guild_id, cutoff),
        ).fetchone()
        return int(row["n"])


def _snapshot_to_json(snapshot: BrainSnapshot) -> str:
    payload = {
        "topology": list(snapshot.topology),
        "layers": [{"weights": layer.weights, "biases": layer.biases} for layer in snapshot.layers],
        "epsilon": snapshot.epsilon,
        "step_count": snapshot.step_count,
        "buffer": [
            {
                "state": exp.state,
                "action": exp.action,
                "reward": exp.reward,
                "next_state": exp.next_state,
                "done": exp.done,
            }
            for exp in snapshot.buffer
        ],
    }
    return json.dumps(payload)


def _snapshot_from_json(text: str) -> BrainSnapshot:
    payload = json.loads(text)
    buffer = [
        Experience(
            state=item["state"],
            action=item["action"],
            reward=item["reward"],
            next_state=item["next_state"],
            done=item["done"],
        )
        for item in payload.get("buffer", [])
    ]
    return BrainSnapshot(
        topology=tuple(payload["topology"]),
        layers=[LayerSnapshot(layer["weights"], layer["biases"]) for layer in payload["layers"]],
        epsilon=payload["epsilon"],
        step_count=payload["step_count"],
        buffer=buffer,
    )

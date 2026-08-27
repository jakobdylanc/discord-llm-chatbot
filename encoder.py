from __future__ import annotations

import math
import re
from enum import Enum

_TOKEN = re.compile(r"[a-z0-9]+")

_POSITIVE = {
    "love",
    "great",
    "awesome",
    "nice",
    "good",
    "thanks",
    "thank",
    "cool",
    "amazing",
    "best",
    "happy",
    "lol",
    "lmao",
    "haha",
    "w",
    "based",
    "fire",
    "goat",
    "pog",
    "clean",
}
_NEGATIVE = {
    "hate",
    "bad",
    "awful",
    "terrible",
    "worst",
    "sucks",
    "trash",
    "angry",
    "sad",
    "annoying",
    "stupid",
    "dumb",
    "l",
    "mid",
    "cringe",
    "broken",
    "ugh",
    "wtf",
}
_POS_EMOJI = {"❤", "😂", "🔥", "👍", "😍"}
_NEG_EMOJI = {"💀", "👎", "😡", "🤮"}


class Intent(Enum):
    question = "question"
    greeting = "greeting"
    mod_request = "modRequest"
    memory_store = "memoryStore"
    persona_switch = "personaSwitch"
    rep_query = "repQuery"
    small_talk = "smallTalk"
    command = "command"
    unknown = "unknown"

    @property
    def index(self) -> int:
        return list(Intent).index(self)


_GREETINGS = ("hi", "hey", "yo", "sup", "hello")


def classify(text: str) -> Intent:
    lower = text.lower()
    if lower.startswith("!") or lower.startswith("/"):
        return Intent.command
    if "remember" in lower or "note that" in lower:
        return Intent.memory_store
    if "rep" in lower or "reputation" in lower:
        return Intent.rep_query
    if "switch" in lower or "be aviva" in lower:
        return Intent.persona_switch
    if lower.endswith("?") or lower.startswith("what") or lower.startswith("how"):
        return Intent.question
    if any(lower.startswith(prefix) for prefix in _GREETINGS):
        return Intent.greeting
    return Intent.small_talk


def sentiment(text: str) -> float:
    tokens = _TOKEN.findall(text.lower())
    if not tokens:
        return 0.0
    score = 0
    for token in tokens:
        if token in _POSITIVE:
            score += 1
        if token in _NEGATIVE:
            score -= 1
    for char in text:
        if char in _POS_EMOJI:
            score += 1
        elif char in _NEG_EMOJI:
            score -= 1
    return max(-1.0, min(1.0, score / max(len(tokens), 4)))


def encode_state(
    *,
    text: str,
    reputation: float,
    mentions_bot: bool,
    has_image: bool,
    hour: int,
    channel_heat: int,
) -> list[float]:
    vector = [0.0] * 18
    intent = classify(text)
    vector[intent.index] = 1.0
    vector[9] = max(0.0, min(1.0, reputation))
    vector[10] = min(len(text), 400) / 400
    vector[11] = 1.0 if mentions_bot else 0.0
    vector[12] = 1.0 if text.endswith("?") else 0.0
    vector[13] = 1.0 if has_image else 0.0
    vector[14] = math.sin(2 * math.pi * hour / 24)
    vector[15] = math.cos(2 * math.pi * hour / 24)
    vector[16] = min(channel_heat, 30) / 30
    vector[17] = sentiment(text)
    return vector

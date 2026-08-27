from __future__ import annotations

DECAY = 0.95
BLEND = 1.0 - DECAY
DEFAULT_REPUTATION = 0.5


def update_reputation(current: float, *, target: float) -> float:
    next_value = current * DECAY + target * BLEND
    return max(0.0, min(1.0, next_value))

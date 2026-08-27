# Goals

## Abbey stay/reply/react DQN on llmcord
status: done
- Offline loop is Current: `[18, 64, 32, 3]` linear DQN, per-guild opt-in learning (default off), delayed 150s rewards, `/learn` `/act` `/brain`, Grok 4.6 replies on mention/DM. Gate: `uv run pytest` (58), `uv run ruff check .`, dummy `--check-config`.
- Residual (not this goal): live Discord/xAI still need operator tokens in `.env`. No push to `origin` (`jakobdylanc/llmcord`).

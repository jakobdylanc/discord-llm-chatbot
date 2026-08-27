# Goals

## Abbey stay/reply/react DQN on llmcord
status: done
- Offline loop is Current: `[18, 64, 32, 3]` linear DQN, per-guild opt-in learning (default off), delayed 150s rewards, `/learn` `/act` `/brain`, Grok 4.6 replies on mention/DM. Gate: `uv run pytest` (58), `uv run ruff check .`, dummy `--check-config`.
- Residual (not this goal): live Discord/xAI still need operator tokens in `.env`. No push to `origin` (`jakobdylanc/llmcord`).

## Local WDBX memory + ABI MCP provider bridge on llmcord
status: in_progress
- Captured 2026-08-27. Scope: `backends.py` adapters (WDBX REST memory, ABI MCP JSON-RPC provider), their config validation in `settings.py`, and the `on_message` wiring. Acceptance: adapters exercised against the real `abi wdbx api serve` / `abi-mcp` binaries, not only `httpx.MockTransport`.

# Goals

## Abbey stay/reply/react DQN on llmcord
status: done
- Offline loop is Current: `[18, 64, 32, 3]` linear DQN, per-guild opt-in learning (default off), delayed 150s rewards, `/learn` `/act` `/brain`, Grok 4.6 replies on mention/DM. Gate: `uv run pytest` (58), `uv run ruff check .`, dummy `--check-config`.
- Residual (not this goal): live Discord/xAI still need operator tokens in `.env`. No push to `origin` (`jakobdylanc/llmcord`).

## Local WDBX memory + ABI MCP provider bridge on llmcord
status: done
- Captured 2026-08-27. Scope: `backends.py` adapters (WDBX REST memory, ABI MCP JSON-RPC provider), their config validation in `settings.py`, and the `on_message` wiring. Acceptance: adapters exercised against the real `abi wdbx api serve` / `abi-mcp` binaries, not only `httpx.MockTransport`.
- **Current at the adapter layer.** Both adapters were driven against the real release binaries on 2026-08-27, which is what the mocks could not prove. WDBX `remember` + `recall` round-trip on a live store with cross-scope isolation measured at semantic 0.0 (same text, other scope); ABI MCP `ping` plus all three tools (`ai_run`, `ai_complete`, `ai_learn`) return clean text. Two defects the green mock suite had passed are fixed with regression tests built from captured live payloads: recall returned **zero** rows because the default `min_score` 0.5 put the floor at 0.975 against a 0.9735 paraphrase, and the persona label `Abbey: ` leaked into every reply on the default `tool: ai_run`. Landed as 07cc9a3 + d5bdff2 on local `main`. Gate: `uv run pytest` (78), `uv run ruff check .`, `--check-config` with memory both off and on.
- **Residual, deliberately not claimed.** Never exercised through a real Discord message: the `on_message` wiring around the adapters is covered only by the source-text assertions in `tests/test_discord_wiring.py`, and live Discord/xAI still need operator tokens. Recall *quality* stays weak by construction, not fixed: on a live 5-document probe true hits span 0.20-0.85 and non-hits span -0.33-0.53, so the bands overlap and 0.35 buys 7/9 recall at ~25% false positives. The ceiling is the hashed n-gram embedding, not the constant. `ai_complete`/`ai_learn` metadata stripping is still a first-`": "` split and would break if ABI adds a metadata value containing `": "`. No push to `origin` (`jakobdylanc/llmcord`, upstream, no write access).

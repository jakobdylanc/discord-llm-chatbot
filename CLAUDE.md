# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this checkout is

A `uv` Python 3.12 Discord bot: upstream [llmcord](https://github.com/jakobdylanc/llmcord)
plus an "Abbey" per-guild DQN that decides whether to stay silent, reply, or react to
unsolicited messages. `origin` points at the **upstream** repo (`jakobdylanc/llmcord`),
not a fork of this account, so there are no push rights. Commit locally and say so.

## Commands

```bash
uv sync                                    # install (dev group is default)
uv run pytest                              # test gate
uv run ruff check .                        # lint gate
uv run python main.py --check-config       # validate config.yaml + env, exit
uv run python main.py                      # run the bot
docker compose up                          # containerized run (host network)
```

Single test: `uv run pytest tests/test_learning.py::test_policy_ignored_when_act_off`.
`pytest` is configured with `pythonpath = ["."]`, so modules import flat (`from brain import ...`).

`--check-config` needs only `DISCORD_BOT_TOKEN` set; a dummy value is enough to
exercise validation without touching Discord. `python llmcord.py` still works as a
legacy entrypoint but skips the argument parsing in `main.py`.

Ruff: line length 120, rules `E,F,I,UP,B`; `llmcord.py` is exempt from `E501` and `B008`.

## Architecture

Layered, one direction, no cycles. Only `main.py` and the tests import `llmcord`.

- **`main.py`** argparse entrypoint. Calls `llmcord.configure()`, then `llmcord.main()`.
- **`llmcord.py`** the entire Discord runtime: slash commands, event handlers,
  reply-chain walking, streaming, module-level mutable globals (`config`, `curr_model`,
  `learning_store`, `msg_nodes`).
- **`settings.py`** YAML load, `_env` resolution, validation, permission evaluation.
  Imports `backends.validate_loopback_url` so config validation and runtime enforce the
  same loopback rule.
- **`backends.py`** the two optional local sidecar adapters (`WdbxMemoryBackend` over
  REST, `AbiMcpBackend` over JSON-RPC), plus embedding and memory-gating helpers.
- **`learning.py`** SQLite store, per-guild settings, cooldown/budget, delayed-reward
  bookkeeping, and the pure `decide_action` policy function.
- **`brain.py`** dependency-free DQN: dense layers, replay buffer, agent, snapshot I/O.
- **`encoder.py`** intent classification, sentiment, and the 18-float state vector.
- **`social.py`** reputation EWMA (decay 0.95).

### `on_message` decision flow

Order matters and is asserted by tests (see Gotchas):

1. Skip bot authors; record `note_human_reply` for the referenced message.
2. Hot-reload and validate `config.yaml` in a thread, then `is_message_allowed`.
3. `encode_state(...)` builds the 18-float vector; `decide_action(...)` returns one of
   `ignore` / `stay` / `react` / `reply`.
4. `ignore`, `stay`, and `react` all **return before any provider client is built**.
   Only `reply` proceeds.
5. Walk the reply chain into `msg_nodes` (size-capped at `MAX_MESSAGE_NODES`, per-node
   `asyncio.Lock`), download text/image attachments, assemble messages newest-first,
   then send reversed.
6. Optional WDBX recall appended as a user message, then the system prompt.
7. Stream through OpenAI-compatible chunks, or one bounded ABI MCP response chunked at
   1000 chars.
8. If learning, `open_pending` on the first response message.

`forced` (a DM or a direct mention) always replies and bypasses the policy, the
cooldown, and the hourly budget. Learning and unsolicited action are **off by default
per guild** (`/learn on`, then `/act on`, admins only).

### Delayed-reward loop

`open_pending` writes the state vector and chosen action to the `pending` table.
`on_raw_reaction_add`, `note_human_reply`, and `on_raw_message_delete` accumulate
counters on that row. `settle_loop` ticks every `learning.settle_every_seconds` and
calls `settle_due()`, which settles rows older than `reward_window_seconds` (150 s
default) or already deleted, turns them into `Experience` records, and calls
`agent.remember` + `agent.learn` + `save()`. Deletion short-circuits the window.

### Persistence

`~/.abbey` by default (`config.data_path` or `ABBEY_DATA_PATH` override), chmod 0700:
`llmcord.sqlite` for settings, budget, cooldown, reputation, and pending rows;
`llmcord-brains/<guild>.json` for weights, epsilon, step count, and the replay buffer.
Guild key is `"dm"` for direct messages.

## Gotchas

- **The 18-dim contract is silent.** `encode_state` emits exactly 18 floats and
  `DQNAgent`'s default topology is `[18, 64, 32, 3]`. `import_weights` **returns without
  error** on a topology or layer-size mismatch, so changing one side alone gives every
  guild a freshly initialized brain with no warning. Change both, and expect saved JSON
  to be dead.
- **`tests/test_discord_wiring.py` asserts on the source text of `llmcord.py`**, using
  `str.index` ordering inside `on_message`. Reordering that function, or renaming the
  literals it greps for, breaks tests even when behavior is unchanged.
- **`_env` resolution is order-sensitive.** In `resolve_env`, a set env var writes to the
  bare key, and a literal key appearing *later* in the same mapping overwrites it. Always
  put `foo:` **before** `foo_env:` in `config.yaml`, or the YAML value wins over the
  environment. An unset var never clobbers an existing literal.
- **Config is reloaded and re-validated on every message.** A `config.yaml` that fails
  validation makes the bot log an exception and silently drop messages rather than crash.
- **Two httpx clients, deliberately.** `_httpx_client()` fetches Discord attachments;
  `_backend_httpx_client()` is built with `trust_env=False` so no proxy env var can
  redirect loopback backend traffic off-machine.
- **Loopback is a security invariant, not a convenience.** `validate_loopback_url` rejects
  anything that is not `http://127.0.0.1` or `http://localhost` with no credentials, path,
  query, or fragment, so configured bearer tokens cannot reach a remote host. WDBX writes
  are gated by `should_store_memory` (explicit "remember" / "note that"), recalled records
  are wrapped by `render_memory_context` as untrusted data, and `AbiMcpBackend` receives
  only the current user text, never the system prompt, reply chain, or recalled memory.
- **Slash commands do not sync by default.** `command_sync_mode` registers guild-scoped
  when `DISCORD_DEV_GUILD_ID` is set, globally only when `ABBEY_ALLOW_GLOBAL_COMMANDS=1`,
  and otherwise skips with a warning. Admin-only commands read
  `permissions.users.admin_ids`.
- **`tests/test_backends.py` mocks the WDBX and ABI wire protocols with
  `httpx.MockTransport`, so a green suite does not mean the bridge works.** The mock
  encodes an *assumption* about the service response shape. Live verification against the real
  `abi wdbx api serve` / `abi-mcp` binaries has already caught two defects the mocks
  passed: a recall floor that returned zero memories for ordinary queries, and a
  persona label leaking into replies. Re-verify live after touching `backends.py`.
- **`abi-mcp` exits on stdin EOF.** It runs a stdio MCP loop alongside the HTTP
  transport, so backgrounding it detached makes it print its listening banner and quit.
  Hold stdin open (`tail -f /dev/null | abi-mcp &`).
- **`.gitignore` is a whitelist and the `Dockerfile` `COPY` is an explicit file list.**
  A new top-level module needs an entry in both or it is invisible to git and missing
  from the image.

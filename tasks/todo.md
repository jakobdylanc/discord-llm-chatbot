# Todo

## Local WDBX memory + ABI MCP provider bridge

- [x] Live-verify `WdbxMemoryBackend` against real `abi wdbx api serve` (found: recall returned 0 rows)
- [x] Live-verify `AbiMcpBackend` against real `abi-mcp` (found: persona label leaked on `ai_run`)
- [x] Skip WDBX recall when the selected provider backend is `abi-mcp`
- [x] Commit the bridge work (07cc9a3) and the docs/ledger (d5bdff2)

### Open, not blocking the goal

- [ ] Exercise the bridge through a real Discord message (needs operator tokens in `.env`)
- [ ] Revisit the recall embedding if memory is ever relied on; threshold tuning is at its useful limit

## Decompose on_message

- [x] Extract _build_conversation, _apply_memory, _completion_chunks (da8d1de)
- [x] Replace source-text wiring tests with behavioural, mutation-checked ones (51fb1bf)
- [x] Extract _gate returning a Gate verdict (487cc40)
- [x] Extract _stream_reply and cover the streaming loop

### Open

- [x] Measure the coupling before splitting (found: `config` is rebound, so a naive split silently breaks hot-reload)
- [x] Extract `_reload_config`; `_gate` now has zero global writes
- [x] Give rebound globals a holder so importers cannot snapshot them (044b9c2)
- [x] Split into runtime.py / pipeline.py / llmcord.py

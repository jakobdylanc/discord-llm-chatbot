# Todo

## Local WDBX memory + ABI MCP provider bridge

- [x] Live-verify `WdbxMemoryBackend` against real `abi wdbx api serve` (found: recall returned 0 rows)
- [x] Live-verify `AbiMcpBackend` against real `abi-mcp` (found: persona label leaked on `ai_run`)
- [x] Skip WDBX recall when the selected provider backend is `abi-mcp`
- [x] Commit the bridge work (07cc9a3) and the docs/ledger (d5bdff2)

### Open, not blocking the goal

- [ ] Exercise the bridge through a real Discord message (needs operator tokens in `.env`)
- [ ] Replace the first-`": "` metadata split in `AbiMcpBackend.complete` with a marker anchored on `block_id=`
- [ ] Revisit the recall embedding if memory is ever relied on; threshold tuning is at its useful limit

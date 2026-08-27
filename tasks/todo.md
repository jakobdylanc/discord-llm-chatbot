# Todo

## Local WDBX memory + ABI MCP provider bridge

- [ ] Live-verify `WdbxMemoryBackend` against real `abi wdbx api serve` (mock protocol assumption is unproven)
- [ ] Live-verify `AbiMcpBackend` against real `abi-mcp` JSON-RPC
- [ ] Skip WDBX recall when the selected provider backend is `abi-mcp` (recall result is discarded; pays timeout for nothing)
- [ ] Commit the bridge work (untracked `backends.py`, `tests/test_backends.py` + 8 modified files)

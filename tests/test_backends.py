import asyncio
import json
import math

import httpx
import pytest

from backends import (
    MAX_ABI_INPUT_BYTES,
    AbiMcpBackend,
    BackendError,
    WdbxMemoryBackend,
    render_memory_context,
    scoped_embedding,
    should_store_memory,
    text_embedding,
)


def test_text_embedding_is_stable_normalized_and_case_insensitive() -> None:
    first = text_embedding("Remember Abbey likes Rust")
    second = text_embedding("remember abbey likes rust")
    assert first == second
    assert len(first) == 32
    assert math.sqrt(sum(value * value for value in first)) == pytest.approx(1.0)
    assert text_embedding("") == [1.0] + [0.0] * 31
    same_scope = sum(
        left * right
        for left, right in zip(
            scoped_embedding("one", "channel:1", "alpha"),
            scoped_embedding("one", "channel:1", "beta"),
            strict=True,
        )
    )
    other_scope = sum(
        left * right
        for left, right in zip(
            scoped_embedding("one", "channel:1", "alpha"),
            scoped_embedding("one", "channel:2", "alpha"),
            strict=True,
        )
    )
    assert same_scope > 0.9
    assert same_scope > other_scope


def test_local_backends_reject_remote_or_credentialed_urls() -> None:
    client = httpx.AsyncClient()
    try:
        with pytest.raises(ValueError, match="loopback"):
            WdbxMemoryBackend(client, base_url="https://example.com")
        with pytest.raises(ValueError, match="credentials"):
            AbiMcpBackend(client, base_url="http://user:pass@localhost:8080")
    finally:
        asyncio.run(client.aclose())


def test_wdbx_memory_writes_vector_and_scoped_payload_then_recalls_it() -> None:
    requests: list[httpx.Request] = []
    stored_value: str | None = None

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal stored_value
        requests.append(request)
        body = json.loads(request.content)
        if request.url.path == "/insert" and "vector" in body:
            return httpx.Response(200, json={"inserted": "vector", "id": "vec-7"})
        if request.url.path == "/insert" and "key" in body:
            stored_value = body["value"]
            return httpx.Response(200, json={"inserted": "kv"})
        if request.url.path == "/query" and "vector" in body:
            assert body["limit"] == 10
            return httpx.Response(200, json={"results": [{"id": "vec-7", "semantic": 0.99}]})
        if request.url.path == "/query" and body.get("key") == "llmcord:v3:llmcord:memory:vec-7":
            return httpx.Response(200, json={"value": stored_value})
        raise AssertionError(f"unexpected request: {request.method} {request.url} {body}")

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = WdbxMemoryBackend(client, base_url="http://127.0.0.1:8081", limit=3)
            vector_id = await backend.remember(
                scope="guild:1",
                author_id="2",
                role="user",
                content="remember that tests matter",
                message_id="9",
            )
            assert vector_id == "vec-7"
            memories = await backend.recall(scope="guild:1", query="what matters?")
            assert [memory.content for memory in memories] == ["remember that tests matter"]
            assert "untrusted historical data" in render_memory_context(memories)

    asyncio.run(exercise())
    kv_write = json.loads(requests[1].content)
    stored = json.loads(kv_write["value"])
    assert kv_write["key"] == "llmcord:v3:llmcord:memory:vec-7"
    assert stored["scope"] == "guild:1"
    assert stored["schema"] == "llmcord.memory.v3"
    assert stored["namespace"] == "llmcord"


def test_wdbx_recall_filters_other_scopes() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if "vector" in body:
            return httpx.Response(200, json={"results": [{"id": 4, "semantic": 0.99}]})
        assert body["key"] == "llmcord:v3:llmcord:memory:4"
        return httpx.Response(
            200,
            json={
                "value": json.dumps(
                    {
                        "schema": "llmcord.memory.v3",
                        "namespace": "llmcord",
                        "scope": "guild:other",
                        "content": "private to another scope",
                    }
                )
            },
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = WdbxMemoryBackend(client, base_url="http://localhost:8081")
            assert await backend.recall(scope="guild:mine", query="private") == []

    asyncio.run(exercise())


def test_wdbx_recall_ignores_low_relevance_candidates_without_fetching_payloads() -> None:
    requests = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requests
        requests += 1
        body = json.loads(request.content)
        assert "vector" in body
        return httpx.Response(200, json={"results": [{"id": 4, "semantic": 0.95}]})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = WdbxMemoryBackend(client, base_url="http://localhost:8081", min_score=0.5)
            assert await backend.recall(scope="guild:mine", query="unrelated") == []

    asyncio.run(exercise())
    assert requests == 1


def test_wdbx_retries_only_the_idempotent_metadata_write() -> None:
    vector_inserts = 0
    metadata_inserts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal vector_inserts, metadata_inserts
        body = json.loads(request.content)
        if "vector" in body:
            vector_inserts += 1
            return httpx.Response(200, json={"id": "vec-retry"})
        metadata_inserts += 1
        if metadata_inserts < 3:
            return httpx.Response(429, json={"error": "rate limited"})
        return httpx.Response(200, json={"inserted": "kv"})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = WdbxMemoryBackend(client, base_url="http://localhost:8081")
            assert await backend.remember(
                scope="guild:1",
                author_id="2",
                role="user",
                content="remember that retries are bounded",
                message_id="10",
            ) == "vec-retry"

    asyncio.run(exercise())
    assert vector_inserts == 1
    assert metadata_inserts == 3


def test_abi_mcp_sends_only_supplied_user_text_and_extracts_tool_text() -> None:
    captured: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "result": {"content": [{"type": "text", "text": "Abbey: local response"}]},
            },
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = AbiMcpBackend(client, base_url="http://127.0.0.1:8080", tool="ai_run")
            result = await backend.complete("Hello", "abbey-local")
            # The profile label is stripped: it is router bookkeeping, not reply text.
            assert result == "local response"

    asyncio.run(exercise())
    assert captured["method"] == "tools/call"
    assert captured["params"]["name"] == "ai_run"
    assert captured["params"]["arguments"] == {"input": "Hello"}


def test_abi_mcp_surfaces_json_rpc_errors_without_response_details() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"jsonrpc": "2.0", "id": 1, "error": {"code": -32603, "message": "Missing input"}},
        )

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = AbiMcpBackend(client, base_url="http://localhost:8080")
            with pytest.raises(BackendError, match="Missing input"):
                await backend.complete("hello", "local")

    asyncio.run(exercise())


def test_abi_mcp_bounds_unicode_input_by_encoded_bytes() -> None:
    serialized = AbiMcpBackend._bound_input("😀" * 10_000)
    assert len(serialized.encode("utf-8")) <= MAX_ABI_INPUT_BYTES
    assert serialized.startswith("😀")


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("remember that tests matter", True),
        ("<@123> please note that deploys need review", True),
        ("Could you remember that I prefer Rust?", True),
        ("do not remember this", False),
        ("what do you remember?", False),
        ("I remember that day", False),
    ],
)
def test_explicit_memory_consent(text: str, expected: bool) -> None:
    assert should_store_memory(text) is expected


def test_abi_mcp_rejects_mismatched_json_rpc_envelope() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 99, "result": {}})

    async def exercise() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            backend = AbiMcpBackend(client, base_url="http://localhost:8080")
            with pytest.raises(BackendError, match="mismatched"):
                await backend.health()

    asyncio.run(exercise())


def test_default_min_score_admits_a_real_paraphrase_hit_and_rejects_unrelated_text() -> None:
    """Regression for the live-verified recall failure.

    The MockTransport tests hand-feed `semantic` values, so they never exercised the
    floor against scores this embedding actually produces. Against a live
    `abi wdbx api serve` store the old 0.5 default put the floor at 0.975 while a
    genuine paraphrase scored 0.9735, so recall silently returned nothing.
    """
    from backends import CONTENT_WEIGHT, DEFAULT_MIN_SCORE, SCOPE_WEIGHT, scoped_embedding

    def semantic(left: str, right: str, *, scope: str = "guild:1:channel:2", other: str | None = None) -> float:
        first = scoped_embedding("llmcord", scope, left)
        second = scoped_embedding("llmcord", other or scope, right)
        return sum(a * b for a, b in zip(first, second, strict=True))

    floor = SCOPE_WEIGHT + CONTENT_WEIGHT * DEFAULT_MIN_SCORE
    stored = "Abbey runs the llmcord bridge on loopback"

    assert semantic(stored, "what does Abbey run on loopback") >= floor
    assert semantic(stored, stored) >= floor
    assert semantic(stored, "pizza recipe with extra cheese") < floor

    # Scope separation must stay far stronger than any content signal.
    cross = semantic(stored, stored, other="dm:999")
    assert cross < floor


def test_abi_tool_output_strips_metadata_blob_and_persona_label() -> None:
    """Payloads captured live from `abi-mcp` on 2026-08-27.

    ai_run carries no metadata blob but still prefixes the profile name, so the old
    `ai_complete`/`ai_learn`-only stripping let a literal "Abbey: " reach Discord on
    the default `tool: ai_run` configuration.
    """
    body = "Loopback keeps the bridge on this host."
    captured = {
        "ai_run": f"Abbey: {body}",
        "ai_complete": (
            "requested_model=abbey-local provider=local transport=in-process "
            "metadata_key=completion:30d2c49b-ed05-44ca-a666-ae52512854cf "
            f"block_id=679bf8d695697089749da74867a9c29f: Abbey: {body}"
        ),
    }
    for tool, raw in captured.items():
        def handler(_request: httpx.Request, _raw: str = raw) -> httpx.Response:
            return httpx.Response(
                200,
                json={"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": _raw}]}},
            )

        async def run(_tool: str = tool) -> str:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                backend = AbiMcpBackend(client, base_url="http://127.0.0.1:8090", tool=_tool)
                return await backend.complete("anything", "abbey-local")

        assert asyncio.run(run()) == body, tool

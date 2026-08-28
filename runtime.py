"""Shared mutable runtime state for the bot.

Everything here is looked up through `state` rather than being a module-level name
that gets rebound. That distinction is the whole point of this module: `config` is
replaced wholesale on every message by the hot-reload, so a `from ... import config`
elsewhere would capture the startup snapshot and quietly serve a stale config forever
while reload appeared to work. `state` itself is never rebound, so `state.config` is
resolved at access time and importers cannot freeze it.

`discord_bot`, `msg_nodes` and `edit_lock` are safe to import directly: each is created
once and mutated in place, never reassigned.

The singleton is called `runtime`, not `state`: the pipeline already uses `state` for a
DQN feature vector, and shadowing that would be a genuine bug rather than a style nit.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import discord
import httpx
from discord.ext import commands

if TYPE_CHECKING:
    from learning import LearningStore

MAX_MESSAGE_NODES = 500

intents = discord.Intents.default()
intents.message_content = True
discord_bot = commands.Bot(intents=intents, command_prefix=None)

edit_lock = asyncio.Lock()


@dataclass
class MsgNode:
    role: Literal["user", "assistant"] = "assistant"

    text: str | None = None
    images: list[dict[str, Any]] = field(default_factory=list)

    has_bad_attachments: bool = False
    fetch_parent_failed: bool = False

    parent_msg: discord.Message | None = None

    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


msg_nodes: dict[int, MsgNode] = {}


@dataclass
class Runtime:
    """The state that changes while the bot runs."""

    config: dict[str, Any] = field(default_factory=dict)
    config_filename: str = "config.yaml"
    curr_model: str = ""
    learning_store: LearningStore | None = None
    last_task_time: float = 0.0
    _httpx_client: httpx.AsyncClient | None = None
    _backend_httpx_client: httpx.AsyncClient | None = None

    def http(self) -> httpx.AsyncClient:
        """Client for Discord attachment downloads."""
        if self._httpx_client is None:
            self._httpx_client = httpx.AsyncClient()
        return self._httpx_client

    def backend_http(self) -> httpx.AsyncClient:
        """Client for the loopback WDBX/ABI sidecars.

        `trust_env=False` so no proxy environment variable can redirect traffic that is
        supposed to stay on this host.
        """
        if self._backend_httpx_client is None:
            self._backend_httpx_client = httpx.AsyncClient(trust_env=False)
        return self._backend_httpx_client

    async def aclose(self) -> None:
        for client in (self._httpx_client, self._backend_httpx_client):
            if client is not None:
                await client.aclose()
        self._httpx_client = self._backend_httpx_client = None


runtime = Runtime()

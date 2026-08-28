"""Behavioural tests for the on_message gate.

These replace an earlier set that sliced llmcord.py's source and asserted on
`str.index` ordering of literals. That pinned the gate to text layout: it failed any
refactor that moved the gate into a helper, and it would have passed a refactor that
kept the literals but reordered execution. What actually matters is that nothing
reaches a provider until the permission and policy gate has allowed it, so assert
that instead.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import discord
import pytest

import llmcord
from brain import DQNAgent
from encoder import encode_state
from learning import Action, LearningStore, command_sync_mode, decide_action


class _Sentinel(Exception):
    """Raised by the patched conversation builder to mark the gate as passed."""


def _message(*, is_dm: bool = False, content: str = "hello", bot: bool = False) -> SimpleNamespace:
    channel = SimpleNamespace(
        id=222,
        type=discord.ChannelType.private if is_dm else discord.ChannelType.text,
        parent_id=None,
        category_id=None,
    )
    return SimpleNamespace(
        id=333,
        content=content,
        author=SimpleNamespace(id=111, bot=bot, roles=()),
        channel=channel,
        guild=None if is_dm else SimpleNamespace(id=444),
        mentions=[],
        reference=None,
        attachments=[],
    )


@pytest.fixture
def wired(tmp_path, monkeypatch):
    """Point llmcord at a scratch learning store and a real config, and trap the gate."""
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    store = LearningStore(tmp_path)
    monkeypatch.setattr(llmcord, "learning_store", store)
    monkeypatch.setattr(llmcord, "config_filename", str(Path(__file__).resolve().parents[1] / "config.yaml"))
    monkeypatch.setattr(llmcord, "config", llmcord.load_and_validate(llmcord.config_filename))
    monkeypatch.setattr(llmcord, "curr_model", "x-ai/grok-4.6")

    reached: list[str] = []

    async def _trap(*_args, **_kwargs):
        reached.append("provider")
        raise _Sentinel

    monkeypatch.setattr(llmcord, "_build_conversation", _trap)
    try:
        yield store, reached
    finally:
        store.close()


def _run(msg) -> None:
    asyncio.run(llmcord.on_message(msg))


def test_bot_authored_messages_never_reach_the_provider(wired) -> None:
    """Uses a DM so the forced path is the one under test.

    A guild message would be stopped by the policy gate regardless, so it cannot tell
    whether the author check works. Mutation-checked: deleting the author guard fails
    this test, and does not fail a guild-channel version of it.
    """
    _store, reached = wired
    _run(_message(is_dm=True, bot=True))
    assert reached == [], "a bot-authored DM must not reach the provider"


def test_guild_message_is_dropped_before_the_provider_when_learning_is_off(wired) -> None:
    store, reached = wired
    assert store.is_learning("444") is False
    _run(_message())
    assert reached == [], "policy gate must return before any provider work"


def test_direct_message_is_forced_through_to_the_provider(wired) -> None:
    _store, reached = wired
    with pytest.raises(_Sentinel):
        _run(_message(is_dm=True))
    assert reached == ["provider"], "a DM must always reply regardless of policy"


def test_act_off_still_blocks_unsolicited_replies_when_learning_is_on(wired) -> None:
    store, reached = wired
    store.set_learning("444", True)
    assert store.is_act("444") is False
    _run(_message())
    assert reached == [], "learning on but act off must not produce unsolicited traffic"


def test_stay_decision_records_experience_without_calling_the_provider(wired) -> None:
    store, reached = wired
    store.set_learning("444", True)
    store.set_act("444", True)
    # Force the policy to choose STAY rather than depending on random weights.
    store.brain("444").select_action = lambda _state: int(Action.STAY)
    _run(_message())
    assert reached == []
    assert len(store.brain("444").buffer) == 1, "STAY must still be learned from"


def test_encode_state_width_matches_the_dqn_input_layer() -> None:
    state = encode_state(
        text="hello", reputation=0.5, mentions_bot=False, has_image=False, hour=12, channel_heat=0
    )
    assert len(state) == DQNAgent().online.topology[0], "encoder width must match the DQN input layer"
    assert len(state) == 18


def test_decide_action_is_pure_and_forced_bypasses_cooldown_and_budget() -> None:
    decision = decide_action(
        forced=True, learning=False, act=False, cooldown_ok=False, budget_ok=False, select=lambda: 0
    )
    assert decision.kind == "reply" and decision.reason == "forced"


def test_command_sync_defaults_to_skip_without_an_explicit_target() -> None:
    assert command_sync_mode(None, False) == "skip"
    assert command_sync_mode("123", False) == "guild"
    assert command_sync_mode(None, True) == "global"


def test_reaction_delete_and_settle_handlers_are_registered() -> None:
    for name in ("on_raw_reaction_add", "on_raw_message_delete", "settle_loop", "sync_app_commands"):
        assert callable(getattr(llmcord, name)), name
    commands = {c.name for c in llmcord.discord_bot.tree.get_commands()}
    assert {"model", "learn", "act", "brain"} <= commands


class _FakeSent:
    """Stands in for a discord.Message the bot just sent."""

    _next_id = 9000

    def __init__(self, **kwargs):
        type(self)._next_id += 1
        self.id = type(self)._next_id
        self.kwargs = kwargs

    async def reply(self, **kwargs):
        return _FakeSent(**kwargs)

    async def edit(self, **kwargs):
        self.kwargs = kwargs


class _FakeTyping:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return False


def _streamable(monkeypatch):
    msg = _message()
    msg.channel.typing = lambda: _FakeTyping()
    msg.reply = _FakeSent().reply
    monkeypatch.setitem(llmcord.config, "use_plain_responses", True)
    return msg


async def _chunks(pairs):
    for pair in pairs:
        yield pair


def test_stream_reply_assembles_every_chunk_including_the_first(monkeypatch, wired) -> None:
    """The loop carries a one-chunk lag; the first chunk is easy to drop.

    A regression here would silently truncate the opening of every reply, which no
    other test in this suite would notice.
    """
    msg = _streamable(monkeypatch)
    sent = asyncio.run(
        llmcord._stream_reply(
            msg, _chunks([("Hello ", None), ("there ", None), ("world", "stop")]), user_warnings=set()
        )
    )
    assert len(sent) == 1
    assert sent[0].kwargs["view"].children[0].content == "Hello there world"


def test_stream_reply_splits_when_content_exceeds_the_plaintext_limit(monkeypatch, wired) -> None:
    msg = _streamable(monkeypatch)
    block = "x" * 2500
    sent = asyncio.run(
        llmcord._stream_reply(msg, _chunks([(block, None), (block, None), ("!", "stop")]), user_warnings=set())
    )
    assert len(sent) == 2, "5001 chars must not be crammed into one 4000-char message"
    assert sum(len(m.kwargs["view"].children[0].content) for m in sent) == 5001


def test_stream_reply_releases_node_locks_when_the_provider_fails(monkeypatch, wired) -> None:
    """A mid-stream failure must not leave a message node locked forever."""
    msg = _streamable(monkeypatch)

    async def _explode():
        yield "partial", None
        raise RuntimeError("provider died")

    sent = asyncio.run(llmcord._stream_reply(msg, _explode(), user_warnings=set()))
    assert sent == [], "nothing was flushed before the failure"
    assert all(not node.lock.locked() for node in llmcord.msg_nodes.values())

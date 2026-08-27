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

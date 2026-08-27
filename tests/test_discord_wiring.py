from pathlib import Path

from encoder import encode_state
from learning import Action, command_sync_mode, decide_action


def test_on_message_skips_bots_then_encodes_and_decides_before_grok() -> None:
    src = Path("llmcord.py").read_text(encoding="utf-8")
    start = src.index("async def on_message")
    end = src.index("async def run_bot")
    body = src[start:end]
    bot_skip = body.index("new_msg.author.bot")
    encode_at = body.index("encode_state(")
    decide_at = body.index("decide_action(")
    ignore_at = body.index('decision.kind == "ignore"')
    stay_at = body.index('decision.kind == "stay"')
    grok_at = body.index("openai_client")
    assert bot_skip < encode_at < decide_at < ignore_at < grok_at
    assert stay_at < grok_at
    assert "add_reaction" in body
    stay_return = body[stay_at:grok_at]
    assert "return" in stay_return
    ignore_return = body[ignore_at:stay_at]
    assert "return" in ignore_return


def test_reaction_delete_settle_and_slash_commands_exist() -> None:
    src = Path("llmcord.py").read_text(encoding="utf-8")
    assert "async def on_raw_reaction_add" in src
    assert "learning_store.note_reaction" in src
    assert "async def on_raw_message_delete" in src
    assert "learning_store.note_deleted" in src
    assert "note_human_reply" in src
    assert "async def settle_loop" in src
    assert 'name="learn"' in src
    assert 'name="act"' in src
    assert 'name="brain"' in src
    assert "command_sync_mode" in src
    assert "copy_global_to" in src
    assert "WdbxMemoryBackend" in src
    assert "render_memory_context" in src
    assert "AbiMcpBackend" in src
    assert "completion_chunks" in src
    assert "trust_env=False" in src
    assert 'messages.append(dict(role="user", content=memory_context))' in src
    assert "load_and_validate, config_filename" in src

    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    assert "backends.py" in dockerfile


def test_shipped_policy_and_sync_functions() -> None:
    state = encode_state(
        text="hello?",
        reputation=0.5,
        mentions_bot=True,
        has_image=False,
        hour=0,
        channel_heat=0,
    )
    assert len(state) == 18
    forced = decide_action(
        forced=True,
        learning=False,
        act=False,
        cooldown_ok=False,
        budget_ok=False,
        select=lambda: 0,
    )
    assert forced.kind == "reply"
    assert forced.action is Action.REPLY
    ignored = decide_action(
        forced=False,
        learning=True,
        act=False,
        cooldown_ok=True,
        budget_ok=True,
        select=lambda: 1,
    )
    assert ignored.kind == "ignore"
    stayed = decide_action(
        forced=False,
        learning=True,
        act=True,
        cooldown_ok=True,
        budget_ok=True,
        select=lambda: 0,
    )
    assert stayed.kind == "stay"
    assert command_sync_mode(None, False) == "skip"
    assert command_sync_mode("99", False) == "guild"
    assert command_sync_mode(None, True) == "global"

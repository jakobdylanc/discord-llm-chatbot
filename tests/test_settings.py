from datetime import UTC, datetime
from pathlib import Path

import pytest

from settings import (
    ConfigError,
    coerce_ids,
    default_model,
    format_system_prompt,
    is_message_allowed,
    is_vision_model,
    resolve_env,
    split_provider_model,
    validate_config,
)


def test_resolve_env_reads_suffixed_keys_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    node = {"bot_token_env": "DISCORD_BOT_TOKEN", "status_message": "hi"}
    assert resolve_env(node) == {"bot_token": "test-token", "status_message": "hi"}


def test_resolve_env_does_not_wipe_existing_value_when_env_is_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISCORD_ADMIN_IDS", raising=False)
    node = {"admin_ids": [1, 2], "admin_ids_env": "DISCORD_ADMIN_IDS"}
    assert resolve_env(node)["admin_ids"] == [1, 2]


def test_resolve_env_prefers_set_env_over_yaml_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_ADMIN_IDS", "42, 99")
    node = {"admin_ids": [1], "admin_ids_env": "DISCORD_ADMIN_IDS"}
    assert resolve_env(node)["admin_ids"] == "42, 99"


def test_resolve_env_walks_lists_and_nested_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INNER", "yes")
    node = {"items": [{"flag_env": "INNER"}]}
    assert resolve_env(node) == {"items": [{"flag": "yes"}]}


def test_coerce_ids_parses_csv_json_and_ints() -> None:
    assert coerce_ids(None) == []
    assert coerce_ids("") == []
    assert coerce_ids("11, 22;33") == [11, 22, 33]
    assert coerce_ids([11, "22", ""]) == [11, 22]
    assert coerce_ids(7) == [7]


def test_coerce_ids_rejects_junk() -> None:
    with pytest.raises(ConfigError, match="id list"):
        coerce_ids("not-an-id")


def test_split_provider_model_strips_vision_suffix() -> None:
    assert split_provider_model("x-ai/grok-4.6:vision") == ("x-ai", "grok-4.6")
    assert split_provider_model("ollama/llama4") == ("ollama", "llama4")


def test_split_provider_model_rejects_missing_slash() -> None:
    with pytest.raises(ConfigError, match="provider/model"):
        split_provider_model("grok-4.6")


def test_is_vision_model_matches_known_tags() -> None:
    assert is_vision_model("x-ai/grok-4.6")
    assert is_vision_model("openai/gpt-5.5")
    assert not is_vision_model("openrouter/deepseek-r1")


def test_default_model_is_first_mapping_key() -> None:
    assert default_model({"x-ai/grok-4.6": {}, "openai/gpt-5.5": {}}) == "x-ai/grok-4.6"


def test_format_system_prompt_inserts_local_date_and_time() -> None:
    now = datetime(2026, 8, 27, 15, 4, 5, tzinfo=UTC)
    text = format_system_prompt("Date {date}. Time {time}.", now)
    assert "August 27 2026" in text
    assert "15:04:05" in text


def test_validate_config_requires_token_models_and_known_provider() -> None:
    with pytest.raises(ConfigError, match="bot_token"):
        validate_config({"models": {"x-ai/grok-4.6": {}}, "providers": {"x-ai": {}}})
    with pytest.raises(ConfigError, match="models"):
        validate_config({"bot_token": "t", "models": {}, "providers": {"x-ai": {}}})
    with pytest.raises(ConfigError, match="unknown provider"):
        validate_config(
            {
                "bot_token": "t",
                "models": {"missing/grok": {}},
                "providers": {"x-ai": {}},
            }
        )
    validate_config(
        {
            "bot_token": "t",
            "models": {"x-ai/grok-4.6": {}},
            "providers": {"x-ai": {"base_url": "https://api.x.ai/v1"}},
        }
    )


def _perms(*, admin=(), users=(), blocked_users=(), roles=(), blocked_roles=(), channels=(), blocked_channels=()):
    return {
        "users": {"admin_ids": list(admin), "allowed_ids": list(users), "blocked_ids": list(blocked_users)},
        "roles": {"allowed_ids": list(roles), "blocked_ids": list(blocked_roles)},
        "channels": {"allowed_ids": list(channels), "blocked_ids": list(blocked_channels)},
    }


def test_empty_allowlists_permit_everyone_in_a_guild_channel() -> None:
    assert is_message_allowed(
        user_id=1,
        role_ids=set(),
        channel_ids={10},
        is_dm=False,
        allow_dms=True,
        permissions=_perms(),
    )


def test_blocked_user_is_denied_even_if_admin_allowlist_empty() -> None:
    assert not is_message_allowed(
        user_id=1,
        role_ids=set(),
        channel_ids={10},
        is_dm=False,
        allow_dms=True,
        permissions=_perms(blocked_users=(1,)),
    )


def test_admin_bypasses_channel_allowlist_and_closed_dms() -> None:
    assert is_message_allowed(
        user_id=9,
        role_ids=set(),
        channel_ids={10},
        is_dm=False,
        allow_dms=True,
        permissions=_perms(admin=(9,), channels=(99,)),
    )
    assert is_message_allowed(
        user_id=9,
        role_ids=set(),
        channel_ids=set(),
        is_dm=True,
        allow_dms=False,
        permissions=_perms(admin=(9,)),
    )


def test_non_admin_dm_respects_allow_dms() -> None:
    assert not is_message_allowed(
        user_id=1,
        role_ids=set(),
        channel_ids=set(),
        is_dm=True,
        allow_dms=False,
        permissions=_perms(),
    )


def test_role_allowlist_grants_guild_access() -> None:
    assert is_message_allowed(
        user_id=1,
        role_ids={5},
        channel_ids={10},
        is_dm=False,
        allow_dms=True,
        permissions=_perms(roles=(5,), users=(99,)),
    )


def test_repo_config_yaml_declares_xai_default(monkeypatch: pytest.MonkeyPatch) -> None:
    from settings import get_config

    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    cfg = get_config(str(Path(__file__).resolve().parents[1] / "config.yaml"))
    assert default_model(cfg["models"]) == "x-ai/grok-4.6"
    assert cfg["providers"]["x-ai"]["api_key"] == "xai-test"
    assert cfg["bot_token"] == "test-token"
    learning = cfg["learning"]
    assert learning["default_enabled"] is False
    assert learning["default_act"] is False
    assert learning["cooldown_seconds"] == 20
    assert learning["unsolicited_per_hour"] == 6
    assert learning["reward_window_seconds"] == 150
    validate_config(cfg)

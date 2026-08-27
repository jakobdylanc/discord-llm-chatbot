from __future__ import annotations

import os
from datetime import datetime
from typing import Any

import yaml

VISION_MODEL_TAGS = (
    "chat-latest",
    "claude",
    "gemini",
    "gemma",
    "gpt-4",
    "gpt-5",
    "gpt-latest",
    "grok-4",
    "llama",
    "vision",
    "vl",
)


class ConfigError(ValueError):
    """Invalid or incomplete llmcord configuration."""


def resolve_env(node: Any) -> Any:
    """Replace `foo_env: VAR` keys with `foo: os.environ[VAR]` when VAR is set.

    Unset env vars do not overwrite a YAML value already present under `foo`.
    """
    if isinstance(node, list):
        return [resolve_env(item) for item in node]
    if not isinstance(node, dict):
        return node

    resolved: dict[str, Any] = {}
    for key, value in node.items():
        if key.endswith("_env"):
            dest = key.removesuffix("_env")
            env_val = os.environ.get(value) if isinstance(value, str) else None
            if env_val is not None and env_val != "":
                resolved[dest] = env_val
            elif dest not in resolved:
                resolved[dest] = None
        else:
            resolved[key] = resolve_env(value)
    return resolved


def coerce_ids(value: Any) -> list[int]:
    try:
        if value is None or value == "":
            return []
        if isinstance(value, int):
            return [value]
        if isinstance(value, str):
            parts = [part.strip() for part in value.replace(";", ",").split(",")]
            return [int(part) for part in parts if part]
        if isinstance(value, list):
            ids: list[int] = []
            for item in value:
                if item in (None, ""):
                    continue
                ids.append(int(item))
            return ids
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"cannot parse id list from {value!r}") from exc
    raise ConfigError(f"cannot parse id list from {type(value).__name__}")


def coerce_permission_ids(config: dict[str, Any]) -> dict[str, Any]:
    permissions = config.get("permissions")
    if not isinstance(permissions, dict):
        return config
    for group in ("users", "roles", "channels"):
        section = permissions.get(group)
        if not isinstance(section, dict):
            continue
        for key, value in list(section.items()):
            if key.endswith("_ids"):
                section[key] = coerce_ids(value)
    return config


def get_config(filename: str = "config.yaml") -> dict[str, Any]:
    with open(filename, encoding="utf-8") as file:
        loaded = yaml.safe_load(file) or {}
    if not isinstance(loaded, dict):
        raise ConfigError("config.yaml must be a mapping")
    config = resolve_env(loaded)
    coerce_permission_ids(config)
    return config


def split_provider_model(spec: str) -> tuple[str, str]:
    stripped = spec.removesuffix(":vision")
    provider, sep, model = stripped.partition("/")
    if not sep or not provider or not model:
        raise ConfigError(f"model {spec!r} must be in provider/model form")
    return provider, model


def is_vision_model(spec: str) -> bool:
    lowered = spec.lower()
    return any(tag in lowered for tag in VISION_MODEL_TAGS)


def default_model(models: dict[str, Any] | None) -> str:
    if not models:
        raise ConfigError("config.yaml models list is empty")
    return next(iter(models))


def format_system_prompt(template: str, now: datetime) -> str:
    return template.replace("{date}", now.strftime("%B %d %Y")).replace("{time}", now.strftime("%H:%M:%S %Z%z")).strip()


def validate_config(config: dict[str, Any]) -> None:
    if not config.get("bot_token"):
        raise ConfigError("bot_token is missing. Set DISCORD_BOT_TOKEN in .env or bot_token in config.yaml.")
    models = config.get("models")
    if not isinstance(models, dict) or not models:
        raise ConfigError("config.yaml models list is empty")
    providers = config.get("providers") or {}
    if not isinstance(providers, dict):
        raise ConfigError("config.yaml providers must be a mapping")
    for name in models:
        provider, _model = split_provider_model(str(name))
        if provider not in providers:
            raise ConfigError(f"model {name!r} references unknown provider {provider!r}")


def load_and_validate(filename: str = "config.yaml") -> dict[str, Any]:
    config = get_config(filename)
    validate_config(config)
    return config


def is_message_allowed(
    *,
    user_id: int,
    role_ids: set[int],
    channel_ids: set[int],
    is_dm: bool,
    allow_dms: bool,
    permissions: dict[str, Any],
) -> bool:
    users = permissions["users"]
    roles = permissions["roles"]
    channels = permissions["channels"]

    admin_ids = coerce_ids(users.get("admin_ids"))
    allowed_user_ids = coerce_ids(users.get("allowed_ids"))
    blocked_user_ids = coerce_ids(users.get("blocked_ids"))
    allowed_role_ids = coerce_ids(roles.get("allowed_ids"))
    blocked_role_ids = coerce_ids(roles.get("blocked_ids"))
    allowed_channel_ids = coerce_ids(channels.get("allowed_ids"))
    blocked_channel_ids = coerce_ids(channels.get("blocked_ids"))

    user_is_admin = user_id in admin_ids

    allow_all_users = not allowed_user_ids if is_dm else not allowed_user_ids and not allowed_role_ids
    is_good_user = (
        user_is_admin
        or allow_all_users
        or user_id in allowed_user_ids
        or any(role_id in allowed_role_ids for role_id in role_ids)
    )
    is_bad_user = (
        not is_good_user or user_id in blocked_user_ids or any(role_id in blocked_role_ids for role_id in role_ids)
    )

    allow_all_channels = not allowed_channel_ids
    if is_dm:
        is_good_channel = user_is_admin or allow_dms
    else:
        is_good_channel = user_is_admin or allow_all_channels or any(cid in allowed_channel_ids for cid in channel_ids)
    is_bad_channel = not is_good_channel or any(cid in blocked_channel_ids for cid in channel_ids)

    return not (is_bad_user or is_bad_channel)

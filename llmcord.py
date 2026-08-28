from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import discord
from discord import app_commands
from discord.app_commands import Choice
from dotenv import load_dotenv
from openai import AsyncOpenAI

from learning import Action, LearningStore, command_sync_mode, emoji_score
from pipeline import (
    _apply_memory,
    _build_conversation,
    _completion_chunks,
    _gate,
    _reload_config,
    _stream_reply,
)
from runtime import MAX_MESSAGE_NODES, MsgNode, discord_bot, msg_nodes, runtime
from settings import (
    ConfigError,
    default_model,
    format_system_prompt,
    is_vision_model,
    load_and_validate,
    split_provider_model,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)




# The ABI router returns one bounded string; chunk it to the shape the stream loop wants.





def _data_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("data_path") or os.environ.get("ABBEY_DATA_PATH")
    return Path(raw) if raw else Path.home() / ".abbey"


def _is_admin(user_id: int) -> bool:
    admin_ids = runtime.config.get("permissions", {}).get("users", {}).get("admin_ids") or []
    return user_id in admin_ids


def _guild_key(interaction_or_msg: discord.Interaction | discord.Message) -> str:
    guild = getattr(interaction_or_msg, "guild", None)
    return str(guild.id) if guild else "dm"






def configure(filename: str = "config.yaml", *, init_store: bool = True) -> dict[str, Any]:
    load_dotenv()
    runtime.config = load_and_validate(filename)
    runtime.config_filename = filename
    if not runtime.curr_model or runtime.curr_model not in runtime.config["models"]:
        runtime.curr_model = default_model(runtime.config["models"])
    status = (runtime.config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128]
    discord_bot.activity = discord.CustomActivity(name=status)
    if init_store:
        window = int((runtime.config.get("learning") or {}).get("reward_window_seconds") or 150)
        runtime.learning_store = LearningStore(_data_dir(runtime.config), reward_window_seconds=window)
    return runtime.config


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:

    if model == runtime.curr_model:
        output = f"Current model: `{runtime.curr_model}`"
    else:
        if _is_admin(interaction.user.id):
            runtime.curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:

    if curr_str == "":
        try:
            runtime.config = await asyncio.to_thread(load_and_validate, runtime.config_filename)
            if runtime.curr_model not in runtime.config["models"]:
                runtime.curr_model = default_model(runtime.config["models"])
        except (OSError, ConfigError):
            logging.exception("Failed to reload config for /model autocomplete")

    models = runtime.config.get("models") or {}
    choices = (
        [Choice(name=f"◉ {runtime.curr_model} (current)", value=runtime.curr_model)] if curr_str.lower() in runtime.curr_model.lower() else []
    )
    choices += [
        Choice(name=f"○ {model}", value=model)
        for model in models
        if model != runtime.curr_model and curr_str.lower() in model.lower()
    ]

    return choices[:25]


async def _require_admin(interaction: discord.Interaction) -> bool:
    if _is_admin(interaction.user.id):
        return True
    await interaction.response.send_message("You don't have permission to do that.", ephemeral=True)
    return False


@discord_bot.tree.command(name="learn", description="Adaptive learning for this server (default off)")
@app_commands.describe(mode="status, on, or off")
@app_commands.choices(
    mode=[
        Choice(name="status", value="status"),
        Choice(name="on", value="on"),
        Choice(name="off", value="off"),
    ]
)
async def learn_command(interaction: discord.Interaction, mode: str = "status") -> None:
    if runtime.learning_store is None:
        await interaction.response.send_message("Learning store is not initialized.", ephemeral=True)
        return
    guild_id = _guild_key(interaction)
    if mode != "status" and not await _require_admin(interaction):
        return
    if mode == "on":
        runtime.learning_store.set_learning(guild_id, True)
    elif mode == "off":
        runtime.learning_store.set_learning(guild_id, False)
    enabled = runtime.learning_store.is_learning(guild_id)
    status = runtime.learning_store.status(guild_id)
    text = (
        f"learning: **{'on' if enabled else 'off'}** · replay {status['replay']} · "
        f"steps {status['steps']} · ε {status['epsilon']:.3f}"
    )
    if not interaction.response.is_done():
        await interaction.response.send_message(text, ephemeral=True)
    else:
        await interaction.followup.send(text, ephemeral=True)


@discord_bot.tree.command(name="act", description="Allow unsolicited stay/reply/react (default off)")
@app_commands.describe(mode="status, on, or off")
@app_commands.choices(
    mode=[
        Choice(name="status", value="status"),
        Choice(name="on", value="on"),
        Choice(name="off", value="off"),
    ]
)
async def act_command(interaction: discord.Interaction, mode: str = "status") -> None:
    if runtime.learning_store is None:
        await interaction.response.send_message("Learning store is not initialized.", ephemeral=True)
        return
    guild_id = _guild_key(interaction)
    if mode != "status" and not await _require_admin(interaction):
        return
    if mode == "on":
        runtime.learning_store.set_act(guild_id, True)
    elif mode == "off":
        runtime.learning_store.set_act(guild_id, False)
    enabled = runtime.learning_store.is_act(guild_id)
    text = f"act: **{'on' if enabled else 'off'}** (unsolicited policy; mentions/DMs always reply)"
    if not interaction.response.is_done():
        await interaction.response.send_message(text, ephemeral=True)
    else:
        await interaction.followup.send(text, ephemeral=True)


@discord_bot.tree.command(name="brain", description="Inspect this server's DQN")
async def brain_command(interaction: discord.Interaction) -> None:
    if not await _require_admin(interaction):
        return
    if runtime.learning_store is None:
        await interaction.response.send_message("Learning store is not initialized.", ephemeral=True)
        return
    status = runtime.learning_store.status(_guild_key(interaction))
    hist = status["histogram"]
    text = (
        f"topology `{status['topology']}` · ε `{status['epsilon']:.3f}` · steps `{status['steps']}`\n"
        f"replay `{status['replay']}` · budget left `{status['budget_left']}`\n"
        f"last Q `{[round(q, 3) for q in status['last_q']]}`\n"
        f"actions stay/reply/react `{hist}` · reward mean `{status['reward_mean']:.3f}`\n"
        f"learning `{'on' if status['learning'] else 'off'}` · act `{'on' if status['act'] else 'off'}`"
    )
    await interaction.response.send_message(text, ephemeral=True)


async def settle_loop() -> None:
    while True:
        interval = float((runtime.config.get("learning") or {}).get("settle_every_seconds") or 5)
        await asyncio.sleep(interval)
        if runtime.learning_store is None:
            continue
        try:
            runtime.learning_store.settle_due()
        except Exception:
            logging.exception("Reward settle failed")


async def sync_app_commands() -> None:
    mode = command_sync_mode(
        os.environ.get("DISCORD_DEV_GUILD_ID"), os.environ.get("ABBEY_ALLOW_GLOBAL_COMMANDS") == "1"
    )
    if mode == "guild":
        guild = discord.Object(id=int(os.environ["DISCORD_DEV_GUILD_ID"]))
        discord_bot.tree.copy_global_to(guild=guild)
        await discord_bot.tree.sync(guild=guild)
        logging.info("Synced slash commands to guild %s", guild.id)
        return
    if mode == "global":
        await discord_bot.tree.sync()
        logging.info("Synced slash commands globally")
        return
    logging.warning("Skipping slash command sync. Set DISCORD_DEV_GUILD_ID or ABBEY_ALLOW_GLOBAL_COMMANDS=1")


@discord_bot.event
async def on_ready() -> None:
    if client_id := runtime.config.get("client_id"):
        logging.info(
            "\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id=%s&permissions=412317191168&scope=bot\n",
            client_id,
        )

    await sync_app_commands()
    asyncio.create_task(settle_loop())


@discord_bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    if runtime.learning_store is None:
        return
    if discord_bot.user and payload.user_id == discord_bot.user.id:
        return
    emoji = str(payload.emoji)
    runtime.learning_store.note_reaction(str(payload.message_id), emoji)
    score = emoji_score(emoji)
    if score != 0:
        guild_id = str(payload.guild_id) if payload.guild_id else "dm"
        runtime.learning_store.apply_reputation(guild_id, str(payload.user_id), target=1.0 if score > 0 else 0.0)


@discord_bot.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
    if runtime.learning_store is None:
        return
    runtime.learning_store.note_deleted(str(payload.message_id))
    runtime.learning_store.settle_due()




















@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:

    if new_msg.author.bot:
        return

    # Recorded before the config reload on purpose: a broken config must not stop the
    # reward loop from noticing that a human replied.
    if runtime.learning_store is not None and new_msg.reference is not None and new_msg.reference.message_id:
        runtime.learning_store.note_human_reply(str(new_msg.reference.message_id))

    if not await _reload_config():
        return

    gate = await _gate(new_msg)
    if not gate.proceed:
        return
    guild_id, state, should_learn = gate.guild_id, gate.state, gate.should_learn

    provider_slash_model = runtime.curr_model
    try:
        provider, model = split_provider_model(provider_slash_model)
    except ConfigError:
        logging.exception("Current model %r is invalid", provider_slash_model)
        return

    try:
        provider_config = runtime.config["providers"][provider]
    except KeyError:
        logging.error("Provider %r is not in config.yaml", provider)
        return

    provider_backend = provider_config.get("backend", "openai")
    openai_client = None
    if provider_backend == "openai":
        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key") or "sk-no-key-required"
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = runtime.config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = is_vision_model(provider_slash_model)

    max_text = runtime.config.get("max_text", 100000)
    max_images = runtime.config.get("max_images", 5) if accept_images else 0
    max_messages = runtime.config.get("max_messages", 25)

    messages, user_warnings = await _build_conversation(
        new_msg, max_text=max_text, max_images=max_images, max_messages=max_messages
    )

    logging.info(
        "Message received (user ID: %s, attachments: %s, conversation length: %s):\n%s",
        new_msg.author.id,
        len(new_msg.attachments),
        len(messages),
        new_msg.content,
    )

    await _apply_memory(messages, new_msg, provider_backend=provider_backend)

    if system_prompt := runtime.config.get("system_prompt"):
        now = datetime.now().astimezone()
        messages.append(dict(role="system", content=format_system_prompt(system_prompt, now)))

    chunks = _completion_chunks(
        provider_backend=provider_backend,
        provider_config=provider_config,
        openai_client=openai_client,
        openai_kwargs=dict(
            model=model,
            messages=messages[::-1],
            stream=True,
            extra_headers=extra_headers,
            extra_query=extra_query,
            extra_body=extra_body,
        ),
        user_text=new_msg.content,
        model=model,
    )
    response_msgs = await _stream_reply(new_msg, chunks, user_warnings=user_warnings)

    if should_learn and runtime.learning_store is not None and state is not None and response_msgs:
        runtime.learning_store.open_pending(
            guild_id=guild_id,
            message_id=str(response_msgs[0].id),
            action=Action.REPLY,
            state=state,
        )

    if (num_nodes := len(msg_nodes)) > MAX_MESSAGE_NODES:
        for msg_id in sorted(msg_nodes.keys())[: num_nodes - MAX_MESSAGE_NODES]:
            async with msg_nodes.setdefault(msg_id, MsgNode()).lock:
                msg_nodes.pop(msg_id, None)


async def run_bot() -> None:
    token = runtime.config.get("bot_token")
    if not token:
        raise ConfigError("bot_token is missing")
    try:
        await discord_bot.start(token)
    finally:
        if runtime.learning_store is not None:
            try:
                runtime.learning_store.save()
            except Exception:
                logging.exception("Failed to persist brains")
        await runtime.aclose()
        if not discord_bot.is_closed():
            await discord_bot.close()


def main(filename: str = "config.yaml") -> None:
    configure(filename)
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        logging.info("Shutting down")


if __name__ == "__main__":
    main()

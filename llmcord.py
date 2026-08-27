from __future__ import annotations

import asyncio
import logging
import os
from base64 import b64encode
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import discord
from discord import app_commands
from discord.app_commands import Choice
from discord.ext import commands
from discord.ui import LayoutView, TextDisplay
from dotenv import load_dotenv
from openai import AsyncOpenAI

from backends import (
    DEFAULT_MIN_SCORE,
    AbiMcpBackend,
    WdbxMemoryBackend,
    render_memory_context,
    should_store_memory,
)
from encoder import encode_state
from learning import Action, LearningStore, command_sync_mode, decide_action, emoji_score
from settings import (
    ConfigError,
    coerce_bool,
    default_model,
    format_system_prompt,
    is_message_allowed,
    is_vision_model,
    load_and_validate,
    split_provider_model,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()

STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1

MAX_MESSAGE_NODES = 500

# The ABI router returns one bounded string; chunk it to the shape the stream loop wants.
ABI_CHUNK_CHARS = 1_000

config: dict[str, Any] = {}
config_filename = "config.yaml"
curr_model = ""
last_task_time = 0.0
edit_lock = asyncio.Lock()
learning_store: LearningStore | None = None

intents = discord.Intents.default()
intents.message_content = True
discord_bot = commands.Bot(intents=intents, command_prefix=None)
httpx_client = None
backend_httpx_client = None


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


def _data_dir(cfg: dict[str, Any]) -> Path:
    raw = cfg.get("data_path") or os.environ.get("ABBEY_DATA_PATH")
    return Path(raw) if raw else Path.home() / ".abbey"


def _is_admin(user_id: int) -> bool:
    admin_ids = config.get("permissions", {}).get("users", {}).get("admin_ids") or []
    return user_id in admin_ids


def _guild_key(interaction_or_msg: discord.Interaction | discord.Message) -> str:
    guild = getattr(interaction_or_msg, "guild", None)
    return str(guild.id) if guild else "dm"


def _memory_scope(msg: discord.Message) -> str:
    return f"guild:{msg.guild.id}:channel:{msg.channel.id}" if msg.guild else f"dm:{msg.author.id}"


def _memory_backend(cfg: dict[str, Any]) -> WdbxMemoryBackend | None:
    memory = cfg.get("memory") or {}
    if not coerce_bool(memory.get("enabled", False), name="memory.enabled"):
        return None
    return WdbxMemoryBackend(
        _backend_httpx_client(),
        base_url=str(memory["base_url"]),
        token=memory.get("token"),
        limit=int(memory.get("limit", 5)),
        max_memory_chars=int(memory.get("max_memory_chars", 2_000)),
        namespace=str(memory.get("namespace", "llmcord")),
        min_score=float(memory.get("min_score", DEFAULT_MIN_SCORE)),
        timeout_seconds=float(memory.get("timeout_seconds", 3)),
    )


def configure(filename: str = "config.yaml", *, init_store: bool = True) -> dict[str, Any]:
    global config, config_filename, curr_model, learning_store
    load_dotenv()
    config = load_and_validate(filename)
    config_filename = filename
    if not curr_model or curr_model not in config["models"]:
        curr_model = default_model(config["models"])
    status = (config.get("status_message") or "github.com/jakobdylanc/llmcord")[:128]
    discord_bot.activity = discord.CustomActivity(name=status)
    if init_store:
        window = int((config.get("learning") or {}).get("reward_window_seconds") or 150)
        learning_store = LearningStore(_data_dir(config), reward_window_seconds=window)
    return config


def _httpx_client():
    global httpx_client
    if httpx_client is None:
        import httpx

        httpx_client = httpx.AsyncClient()
    return httpx_client


def _backend_httpx_client():
    global backend_httpx_client
    if backend_httpx_client is None:
        import httpx

        backend_httpx_client = httpx.AsyncClient(trust_env=False)
    return backend_httpx_client


@discord_bot.tree.command(name="model", description="View or switch the current model")
async def model_command(interaction: discord.Interaction, model: str) -> None:
    global curr_model

    if model == curr_model:
        output = f"Current model: `{curr_model}`"
    else:
        if _is_admin(interaction.user.id):
            curr_model = model
            output = f"Model switched to: `{model}`"
            logging.info(output)
        else:
            output = "You don't have permission to change the model."

    await interaction.response.send_message(output, ephemeral=(interaction.channel.type == discord.ChannelType.private))


@model_command.autocomplete("model")
async def model_autocomplete(interaction: discord.Interaction, curr_str: str) -> list[Choice[str]]:
    global config, curr_model

    if curr_str == "":
        try:
            config = await asyncio.to_thread(load_and_validate, config_filename)
            if curr_model not in config["models"]:
                curr_model = default_model(config["models"])
        except (OSError, ConfigError):
            logging.exception("Failed to reload config for /model autocomplete")

    models = config.get("models") or {}
    choices = (
        [Choice(name=f"◉ {curr_model} (current)", value=curr_model)] if curr_str.lower() in curr_model.lower() else []
    )
    choices += [
        Choice(name=f"○ {model}", value=model)
        for model in models
        if model != curr_model and curr_str.lower() in model.lower()
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
    if learning_store is None:
        await interaction.response.send_message("Learning store is not initialized.", ephemeral=True)
        return
    guild_id = _guild_key(interaction)
    if mode != "status" and not await _require_admin(interaction):
        return
    if mode == "on":
        learning_store.set_learning(guild_id, True)
    elif mode == "off":
        learning_store.set_learning(guild_id, False)
    enabled = learning_store.is_learning(guild_id)
    status = learning_store.status(guild_id)
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
    if learning_store is None:
        await interaction.response.send_message("Learning store is not initialized.", ephemeral=True)
        return
    guild_id = _guild_key(interaction)
    if mode != "status" and not await _require_admin(interaction):
        return
    if mode == "on":
        learning_store.set_act(guild_id, True)
    elif mode == "off":
        learning_store.set_act(guild_id, False)
    enabled = learning_store.is_act(guild_id)
    text = f"act: **{'on' if enabled else 'off'}** (unsolicited policy; mentions/DMs always reply)"
    if not interaction.response.is_done():
        await interaction.response.send_message(text, ephemeral=True)
    else:
        await interaction.followup.send(text, ephemeral=True)


@discord_bot.tree.command(name="brain", description="Inspect this server's DQN")
async def brain_command(interaction: discord.Interaction) -> None:
    if not await _require_admin(interaction):
        return
    if learning_store is None:
        await interaction.response.send_message("Learning store is not initialized.", ephemeral=True)
        return
    status = learning_store.status(_guild_key(interaction))
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
        interval = float((config.get("learning") or {}).get("settle_every_seconds") or 5)
        await asyncio.sleep(interval)
        if learning_store is None:
            continue
        try:
            learning_store.settle_due()
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
    if client_id := config.get("client_id"):
        logging.info(
            "\n\nBOT INVITE URL:\nhttps://discord.com/oauth2/authorize?client_id=%s&permissions=412317191168&scope=bot\n",
            client_id,
        )

    await sync_app_commands()
    asyncio.create_task(settle_loop())


@discord_bot.event
async def on_raw_reaction_add(payload: discord.RawReactionActionEvent) -> None:
    if learning_store is None:
        return
    if discord_bot.user and payload.user_id == discord_bot.user.id:
        return
    emoji = str(payload.emoji)
    learning_store.note_reaction(str(payload.message_id), emoji)
    score = emoji_score(emoji)
    if score != 0:
        guild_id = str(payload.guild_id) if payload.guild_id else "dm"
        learning_store.apply_reputation(guild_id, str(payload.user_id), target=1.0 if score > 0 else 0.0)


@discord_bot.event
async def on_raw_message_delete(payload: discord.RawMessageDeleteEvent) -> None:
    if learning_store is None:
        return
    learning_store.note_deleted(str(payload.message_id))
    learning_store.settle_due()




async def _completion_chunks(
    *,
    provider_backend: str,
    provider_config: dict[str, Any],
    openai_client: AsyncOpenAI | None,
    openai_kwargs: dict[str, Any],
    user_text: str,
    model: str,
):
    """Yield `(text, finish_reason)` pairs from whichever provider backend is selected.

    The ABI MCP router returns one bounded string rather than a stream, so it is chunked
    to the same shape the streaming loop already consumes.
    """
    if provider_backend == "abi-mcp":
        abi = AbiMcpBackend(
            _backend_httpx_client(),
            base_url=provider_config["base_url"],
            token=provider_config.get("token"),
            tool=provider_config.get("tool", "ai_run"),
            evidence_limit=int(provider_config.get("evidence_limit", 5)),
            timeout_seconds=float(provider_config.get("timeout_seconds", 30)),
        )
        text = await abi.complete(user_text, model)
        for start in range(0, len(text), ABI_CHUNK_CHARS):
            end = min(start + ABI_CHUNK_CHARS, len(text))
            yield text[start:end], "stop" if end == len(text) else None
        return

    if openai_client is None:
        raise ConfigError(f"provider backend {provider_backend!r} has no OpenAI client")
    async for chunk in await openai_client.chat.completions.create(**openai_kwargs):
        choice = chunk.choices[0] if chunk.choices else None
        if choice is not None:
            yield choice.delta.content or "", choice.finish_reason


async def _build_conversation(
    new_msg: discord.Message, *, max_text: int, max_images: int, max_messages: int
) -> tuple[list[dict[str, Any]], set[str]]:
    """Walk the reply chain newest-first into OpenAI-shaped messages plus user warnings."""
    messages = []
    user_warnings = set()
    curr_msg = new_msg

    while curr_msg is not None and len(messages) < max_messages:
        curr_node = msg_nodes.setdefault(curr_msg.id, MsgNode())

        async with curr_node.lock:
            if curr_node.text is None:
                cleaned_content = curr_msg.content.removeprefix(discord_bot.user.mention).lstrip()

                good_attachments = [
                    att
                    for att in curr_msg.attachments
                    if att.content_type and any(att.content_type.startswith(x) for x in ("text", "image"))
                ]

                attachment_responses = await asyncio.gather(*[_httpx_client().get(att.url) for att in good_attachments])

                curr_node.role = "assistant" if curr_msg.author == discord_bot.user else "user"

                curr_node.text = "\n".join(
                    ([cleaned_content] if cleaned_content else [])
                    + [
                        "\n".join(filter(None, (embed.title, embed.description, embed.footer.text)))
                        for embed in curr_msg.embeds
                    ]
                    + [
                        component.content
                        for component in curr_msg.components
                        if component.type == discord.ComponentType.text_display
                    ]
                    + [
                        resp.text
                        for att, resp in zip(good_attachments, attachment_responses, strict=True)
                        if att.content_type.startswith("text")
                    ]
                )

                curr_node.images = [
                    dict(
                        type="image_url",
                        image_url=dict(url=f"data:{att.content_type};base64,{b64encode(resp.content).decode('utf-8')}"),
                    )
                    for att, resp in zip(good_attachments, attachment_responses, strict=True)
                    if att.content_type.startswith("image")
                ]

                if curr_node.role == "user" and (curr_node.text or curr_node.images):
                    curr_node.text = f"<@{curr_msg.author.id}>: {curr_node.text}"

                curr_node.has_bad_attachments = len(curr_msg.attachments) > len(good_attachments)

                try:
                    if (
                        curr_msg.reference is None
                        and discord_bot.user.mention not in curr_msg.content
                        and (
                            prev_msg_in_channel := (
                                [m async for m in curr_msg.channel.history(before=curr_msg, limit=1)] or [None]
                            )[0]
                        )
                        and prev_msg_in_channel.type in (discord.MessageType.default, discord.MessageType.reply)
                        and prev_msg_in_channel.author
                        == (
                            discord_bot.user
                            if curr_msg.channel.type == discord.ChannelType.private
                            else curr_msg.author
                        )
                    ):
                        curr_node.parent_msg = prev_msg_in_channel
                    else:
                        is_public_thread = curr_msg.channel.type == discord.ChannelType.public_thread
                        parent_is_thread_start = (
                            is_public_thread
                            and curr_msg.reference is None
                            and curr_msg.channel.parent.type == discord.ChannelType.text
                        )

                        if (
                            parent_msg_id := curr_msg.channel.id
                            if parent_is_thread_start
                            else getattr(curr_msg.reference, "message_id", None)
                        ):
                            if parent_is_thread_start:
                                curr_node.parent_msg = (
                                    curr_msg.channel.starter_message
                                    or await curr_msg.channel.parent.fetch_message(parent_msg_id)
                                )
                            else:
                                curr_node.parent_msg = (
                                    curr_msg.reference.cached_message
                                    or await curr_msg.channel.fetch_message(parent_msg_id)
                                )

                except (discord.NotFound, discord.HTTPException):
                    logging.exception("Error fetching next message in the chain")
                    curr_node.fetch_parent_failed = True

            node_text = curr_node.text or ""
            if curr_node.images[:max_images]:
                content = [dict(type="text", text=node_text[:max_text])] + curr_node.images[:max_images]
            else:
                content = node_text[:max_text]

            if content != "":
                messages.append(dict(content=content, role=curr_node.role))

            if len(node_text) > max_text:
                user_warnings.add(f"⚠️ Max {max_text:,} characters per message")
            if len(curr_node.images) > max_images:
                user_warnings.add(
                    f"⚠️ Max {max_images} image{'' if max_images == 1 else 's'} per message"
                    if max_images > 0
                    else "⚠️ Can't see images"
                )
            if curr_node.has_bad_attachments:
                user_warnings.add("⚠️ Unsupported attachments")
            if curr_node.fetch_parent_failed or (curr_node.parent_msg is not None and len(messages) == max_messages):
                user_warnings.add(f"⚠️ Only using last {len(messages)} message{'' if len(messages) == 1 else 's'}")

            curr_msg = curr_node.parent_msg

    return messages, user_warnings


async def _apply_memory(
    messages: list[dict[str, Any]], new_msg: discord.Message, *, provider_backend: str
) -> None:
    """Append recalled WDBX context to `messages` and persist explicit remember requests."""
    try:
        memory_backend = _memory_backend(config)
    except Exception:
        logging.exception("WDBX memory configuration failed; continuing without durable memory")
        memory_backend = None

    if memory_backend is not None:
        # The ABI MCP path deliberately sends only the current user text, so it discards
        # `messages` entirely. Recalling into a list nobody reads would pay the WDBX
        # timeout on every message for nothing. Writes still run: an explicit "remember"
        # must persist regardless of which provider answers.
        if provider_backend != "abi-mcp":
            try:
                memories = await memory_backend.recall(scope=_memory_scope(new_msg), query=new_msg.content)
                if memory_context := render_memory_context(memories):
                    messages.append(dict(role="user", content=memory_context))
            except Exception:
                logging.exception("WDBX memory recall failed; continuing without recalled memory")
        if should_store_memory(new_msg.content):
            try:
                await memory_backend.remember(
                    scope=_memory_scope(new_msg),
                    author_id=str(new_msg.author.id),
                    role="user",
                    content=new_msg.content,
                    message_id=str(new_msg.id),
                )
            except Exception:
                logging.exception("WDBX memory write failed")


@discord_bot.event
async def on_message(new_msg: discord.Message) -> None:
    global last_task_time, config, curr_model

    if new_msg.author.bot:
        return

    if learning_store is not None and new_msg.reference is not None and new_msg.reference.message_id:
        learning_store.note_human_reply(str(new_msg.reference.message_id))

    is_dm = new_msg.channel.type == discord.ChannelType.private
    mentioned = bool(discord_bot.user and discord_bot.user in new_msg.mentions)
    forced = is_dm or mentioned

    role_ids = set(role.id for role in getattr(new_msg.author, "roles", ()))
    channel_ids = set(
        filter(
            None,
            (
                new_msg.channel.id,
                getattr(new_msg.channel, "parent_id", None),
                getattr(new_msg.channel, "category_id", None),
            ),
        )
    )

    try:
        config = await asyncio.to_thread(load_and_validate, config_filename)
        if curr_model not in config["models"]:
            curr_model = default_model(config["models"])
    except (OSError, ConfigError):
        logging.exception("Failed to reload config.yaml")
        return

    allow_dms = config.get("allow_dms", True)
    permissions = config["permissions"]

    if not is_message_allowed(
        user_id=new_msg.author.id,
        role_ids=role_ids,
        channel_ids=channel_ids,
        is_dm=is_dm,
        allow_dms=allow_dms,
        permissions=permissions,
    ):
        return

    guild_id = str(new_msg.guild.id) if new_msg.guild else "dm"
    state: list[float] | None = None
    should_learn = False

    if learning_store is not None:
        learning_store.note_channel_message(str(new_msg.channel.id))
        has_image = any(att.content_type and att.content_type.startswith("image") for att in new_msg.attachments)
        state = encode_state(
            text=new_msg.content,
            reputation=learning_store.reputation(guild_id, str(new_msg.author.id)),
            mentions_bot=mentioned,
            has_image=has_image,
            hour=datetime.now().astimezone().hour,
            channel_heat=learning_store.channel_heat(str(new_msg.channel.id)),
        )
        captured_state = state

        def _select() -> int:
            assert learning_store is not None
            return learning_store.brain(guild_id).select_action(captured_state)

        decision = decide_action(
            forced=forced,
            learning=learning_store.is_learning(guild_id),
            act=learning_store.is_act(guild_id),
            cooldown_ok=forced or learning_store.cooldown_ok(str(new_msg.channel.id), guild_id),
            budget_ok=forced or learning_store.budget_ok(guild_id),
            select=_select,
        )
        should_learn = decision.learn
        logging.info("policy guild=%s kind=%s reason=%s forced=%s", guild_id, decision.kind, decision.reason, forced)
        if decision.kind == "ignore":
            return
        if decision.kind == "stay":
            learning_store.remember_stay(guild_id, state)
            return
        if decision.kind == "react":
            if not forced:
                learning_store.spend_budget(guild_id)
                learning_store.mark_unsolicited(str(new_msg.channel.id))
            emoji = (config.get("learning") or {}).get("react_emoji") or "👍"
            try:
                await new_msg.add_reaction(emoji)
            except discord.HTTPException:
                logging.exception("Failed to add reaction")
            if should_learn:
                learning_store.open_pending(
                    guild_id=guild_id,
                    message_id=str(new_msg.id),
                    action=Action.REACT,
                    state=state,
                )
            return
        if decision.kind == "reply" and not forced:
            learning_store.spend_budget(guild_id)
            learning_store.mark_unsolicited(str(new_msg.channel.id))
    elif not forced:
        return

    provider_slash_model = curr_model
    try:
        provider, model = split_provider_model(provider_slash_model)
    except ConfigError:
        logging.exception("Current model %r is invalid", provider_slash_model)
        return

    try:
        provider_config = config["providers"][provider]
    except KeyError:
        logging.error("Provider %r is not in config.yaml", provider)
        return

    provider_backend = provider_config.get("backend", "openai")
    openai_client = None
    if provider_backend == "openai":
        base_url = provider_config["base_url"]
        api_key = provider_config.get("api_key") or "sk-no-key-required"
        openai_client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    model_parameters = config["models"].get(provider_slash_model, None)

    extra_headers = provider_config.get("extra_headers")
    extra_query = provider_config.get("extra_query")
    extra_body = (provider_config.get("extra_body") or {}) | (model_parameters or {}) or None

    accept_images = is_vision_model(provider_slash_model)

    max_text = config.get("max_text", 100000)
    max_images = config.get("max_images", 5) if accept_images else 0
    max_messages = config.get("max_messages", 25)

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

    if system_prompt := config.get("system_prompt"):
        now = datetime.now().astimezone()
        messages.append(dict(role="system", content=format_system_prompt(system_prompt, now)))

    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    acquired_nodes: list[MsgNode] = []

    openai_kwargs = dict(
        model=model,
        messages=messages[::-1],
        stream=True,
        extra_headers=extra_headers,
        extra_query=extra_query,
        extra_body=extra_body,
    )

    if use_plain_responses := config.get("use_plain_responses", False):
        max_message_length = 4000
    else:
        max_message_length = 4096 - len(STREAMING_INDICATOR)
        embed = discord.Embed.from_dict(
            dict(fields=[dict(name=warning, value="", inline=False) for warning in sorted(user_warnings)])
        )

    async def reply_helper(**reply_kwargs) -> None:
        reply_target = new_msg if not response_msgs else response_msgs[-1]
        response_msg = await reply_target.reply(**reply_kwargs)
        response_msgs.append(response_msg)

        node = MsgNode(parent_msg=new_msg)
        msg_nodes[response_msg.id] = node
        await node.lock.acquire()
        acquired_nodes.append(node)

    try:
        async with new_msg.channel.typing():
            chunks = _completion_chunks(
                provider_backend=provider_backend,
                provider_config=provider_config,
                openai_client=openai_client,
                openai_kwargs=openai_kwargs,
                user_text=new_msg.content,
                model=model,
            )
            async for chunk_content, chunk_finish_reason in chunks:
                if finish_reason is not None:
                    break

                finish_reason = chunk_finish_reason

                prev_content = curr_content or ""
                curr_content = chunk_content

                new_content = prev_content if finish_reason is None else (prev_content + curr_content)

                if response_contents == [] and new_content == "":
                    continue

                if (
                    start_next_msg := response_contents == []
                    or len(response_contents[-1] + new_content) > max_message_length
                ):
                    response_contents.append("")

                response_contents[-1] += new_content

                if not use_plain_responses:
                    async with edit_lock:
                        time_delta = datetime.now().timestamp() - last_task_time

                        ready_to_edit = time_delta >= EDIT_DELAY_SECONDS
                        msg_split_incoming = (
                            finish_reason is None and len(response_contents[-1] + curr_content) > max_message_length
                        )
                        is_final_edit = finish_reason is not None or msg_split_incoming
                        is_good_finish = finish_reason is not None and finish_reason.lower() in ("stop", "end_turn")

                        if start_next_msg or ready_to_edit or is_final_edit:
                            embed.description = (
                                response_contents[-1]
                                if is_final_edit
                                else (response_contents[-1] + STREAMING_INDICATOR)
                            )
                            embed.color = (
                                EMBED_COLOR_COMPLETE if msg_split_incoming or is_good_finish else EMBED_COLOR_INCOMPLETE
                            )

                            if start_next_msg:
                                await reply_helper(embed=embed, silent=True)
                            else:
                                await asyncio.sleep(max(0.0, EDIT_DELAY_SECONDS - time_delta))
                                await response_msgs[-1].edit(embed=embed)

                            last_task_time = datetime.now().timestamp()

            if use_plain_responses:
                for content in response_contents:
                    await reply_helper(view=LayoutView().add_item(TextDisplay(content=content)))

    except Exception:
        logging.exception("Error while generating response")
    finally:
        joined = "".join(response_contents)
        for node in acquired_nodes:
            node.text = joined
            if node.lock.locked():
                node.lock.release()

    if should_learn and learning_store is not None and state is not None and response_msgs:
        learning_store.open_pending(
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
    token = config.get("bot_token")
    if not token:
        raise ConfigError("bot_token is missing")
    try:
        await discord_bot.start(token)
    finally:
        if learning_store is not None:
            try:
                learning_store.save()
            except Exception:
                logging.exception("Failed to persist brains")
        client = httpx_client
        if client is not None:
            await client.aclose()
        backend_client = backend_httpx_client
        if backend_client is not None:
            await backend_client.aclose()
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

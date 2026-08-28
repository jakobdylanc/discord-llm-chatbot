"""The message pipeline: gate, assemble, augment, stream.

Split out of llmcord.py, which keeps bot wiring (commands, events, lifecycle). Shared
mutable state comes from runtime.py rather than from llmcord, so nothing here imports
the module that imports this one.
"""

from __future__ import annotations

import asyncio
import logging
from base64 import b64encode
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import discord
from discord.ui import LayoutView, TextDisplay
from openai import AsyncOpenAI

from backends import (
    DEFAULT_MIN_SCORE,
    AbiMcpBackend,
    WdbxMemoryBackend,
    render_memory_context,
    should_store_memory,
)
from encoder import encode_state
from learning import Action, decide_action
from runtime import MsgNode, discord_bot, edit_lock, msg_nodes, runtime
from settings import (
    ConfigError,
    coerce_bool,
    default_model,
    is_message_allowed,
    load_and_validate,
)

EMBED_COLOR_COMPLETE = discord.Color.dark_green()
EMBED_COLOR_INCOMPLETE = discord.Color.orange()
STREAMING_INDICATOR = " ⚪"
EDIT_DELAY_SECONDS = 1
ABI_CHUNK_CHARS = 1_000


def _memory_scope(msg: discord.Message) -> str:
    return f"guild:{msg.guild.id}:channel:{msg.channel.id}" if msg.guild else f"dm:{msg.author.id}"


def _memory_backend(cfg: dict[str, Any]) -> WdbxMemoryBackend | None:
    memory = cfg.get("memory") or {}
    if not coerce_bool(memory.get("enabled", False), name="memory.enabled"):
        return None
    return WdbxMemoryBackend(
        runtime.backend_http(),
        base_url=str(memory["base_url"]),
        token=memory.get("token"),
        limit=int(memory.get("limit", 5)),
        max_memory_chars=int(memory.get("max_memory_chars", 2_000)),
        namespace=str(memory.get("namespace", "llmcord")),
        min_score=float(memory.get("min_score", DEFAULT_MIN_SCORE)),
        timeout_seconds=float(memory.get("timeout_seconds", 3)),
    )


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
            runtime.backend_http(),
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

                attachment_responses = await asyncio.gather(*[runtime.http().get(att.url) for att in good_attachments])

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
        memory_backend = _memory_backend(runtime.config)
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


async def _stream_reply(
    new_msg: discord.Message,
    chunks: AsyncIterator[tuple[str, str | None]],
    *,
    user_warnings: set[str],
) -> list[discord.Message]:
    """Stream `chunks` into Discord, splitting and editing messages as content arrives.

    Returns the messages sent, newest last. Every reply node is locked while being
    written and released in the finally block, so a mid-stream failure still leaves the
    cache consistent and the partial text readable by a later reply-chain walk.
    """

    curr_content = finish_reason = None
    response_msgs = []
    response_contents = []
    acquired_nodes: list[MsgNode] = []

    if use_plain_responses := runtime.config.get("use_plain_responses", False):
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
                        time_delta = datetime.now().timestamp() - runtime.last_task_time

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

                            runtime.last_task_time = datetime.now().timestamp()

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

    return response_msgs


async def _reload_config() -> bool:
    """Re-read and validate config.yaml, keeping curr_model pointing at something real.

    Runs per message so edits apply without a restart. Returns False when the file no
    longer validates, in which case the caller must drop the message rather than serve
    it with a half-applied config.
    """

    try:
        runtime.config = await asyncio.to_thread(load_and_validate, runtime.config_filename)
    except (OSError, ConfigError):
        logging.exception("Failed to reload config.yaml")
        return False
    if runtime.curr_model not in runtime.config["models"]:
        runtime.curr_model = default_model(runtime.config["models"])
    return True


@dataclass(frozen=True)
class Gate:
    """The policy verdict for one incoming message.

    `proceed` is the only thing on_message has to branch on; everything the reward
    bookkeeping needs later rides along rather than being recomputed.
    """

    proceed: bool
    guild_id: str = "dm"
    state: list[float] | None = None
    should_learn: bool = False


STOP = Gate(proceed=False)


async def _gate(new_msg: discord.Message) -> Gate:
    """Decide whether this message earns a reply, applying any non-reply policy action.

    Returns STOP for every path that must not reach a provider: denied permissions and
    the ignore/stay/react verdicts. Mentions and DMs are forced through and bypass
    cooldown and budget.

    Reads global runtime state but writes none of it, so it can move to its own module
    once `config` stops being a rebound global. See _reload_config.
    """
    store = runtime.learning_store
    is_dm = new_msg.channel.type == discord.ChannelType.private
    mentioned = bool(discord_bot.user and discord_bot.user in new_msg.mentions)
    forced = is_dm or mentioned

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
    if not is_message_allowed(
        user_id=new_msg.author.id,
        role_ids={role.id for role in getattr(new_msg.author, "roles", ())},
        channel_ids=channel_ids,
        is_dm=is_dm,
        allow_dms=runtime.config.get("allow_dms", True),
        permissions=runtime.config["permissions"],
    ):
        return STOP

    guild_id = str(new_msg.guild.id) if new_msg.guild else "dm"
    if store is None:
        return Gate(proceed=True, guild_id=guild_id) if forced else STOP

    store.note_channel_message(str(new_msg.channel.id))
    state = encode_state(
        text=new_msg.content,
        reputation=store.reputation(guild_id, str(new_msg.author.id)),
        mentions_bot=mentioned,
        has_image=any(att.content_type and att.content_type.startswith("image") for att in new_msg.attachments),
        hour=datetime.now().astimezone().hour,
        channel_heat=store.channel_heat(str(new_msg.channel.id)),
    )
    decision = decide_action(
        forced=forced,
        learning=store.is_learning(guild_id),
        act=store.is_act(guild_id),
        cooldown_ok=forced or store.cooldown_ok(str(new_msg.channel.id), guild_id),
        budget_ok=forced or store.budget_ok(guild_id),
        select=lambda: store.brain(guild_id).select_action(state),
    )
    logging.info("policy guild=%s kind=%s reason=%s forced=%s", guild_id, decision.kind, decision.reason, forced)

    def _spend() -> None:
        if not forced:
            store.spend_budget(guild_id)
            store.mark_unsolicited(str(new_msg.channel.id))

    if decision.kind == "ignore":
        return STOP
    if decision.kind == "stay":
        store.remember_stay(guild_id, state)
        return STOP
    if decision.kind == "react":
        _spend()
        emoji = (runtime.config.get("learning") or {}).get("react_emoji") or "\N{THUMBS UP SIGN}"
        try:
            await new_msg.add_reaction(emoji)
        except discord.HTTPException:
            logging.exception("Failed to add reaction")
        if decision.learn:
            store.open_pending(guild_id=guild_id, message_id=str(new_msg.id), action=Action.REACT, state=state)
        return STOP
    if decision.kind == "reply":
        _spend()
    return Gate(proceed=True, guild_id=guild_id, state=state, should_learn=decision.learn)

<h1 align="center">
  llmcord
</h1>

<h3 align="center"><i>
  Talk to LLMs with your friends!
</i></h3>

<p align="center">
  <img src="https://github.com/user-attachments/assets/7791cc6b-6755-484f-a9e3-0707765b081f" alt="">
</p>

llmcord transforms Discord into a collaborative LLM frontend. It works with practically any LLM, remote or locally hosted.

## Features

### Reply-based conversations:
Just @ the bot to start a conversation and reply to continue. Build conversations with reply chains!

The reply chain is the conversation history, stored entirely in Discord. No database required.

You can:
- Branch conversations endlessly
- Continue other people's conversations
- @ the bot while replying to ANY message to include it in the conversation

Additionally:
- When DMing the bot, conversations continue automatically (no reply required). To start a fresh conversation, just @ the bot. You can still reply to continue from anywhere.
- You can branch conversations into [threads](https://support.discord.com/hc/en-us/articles/4403205878423-Threads-FAQ). Just create a thread from any message and @ the bot inside to continue.
- Back-to-back messages from the same user are automatically chained together. Just reply to the latest one and the bot will see all of them.

This checkout is a [uv](https://docs.astral.sh/uv/) Python 3.12 Abbey bot on top of llmcord. Default model is **xAI grok-4.6** (`XAI_API_KEY`). Run with `uv run python main.py`.

It transcribes Abbey's stay/reply/react DQN (`[18, 64, 32, 3]`, delayed 150 s rewards) into pure Python. Learning and unsolicited speech are **default off** (`/learn on`, `/act on`). This is not AbbeyBot (Swift), not abbey-bot (Rust), and not WDBX. Mentions and DMs still always reply.

---

### Model switching with `/model`:
![image](https://github.com/user-attachments/assets/568e2f5c-bf32-4b77-ab57-198d9120f3d2)

llmcord supports remote models from:
- [OpenRouter](https://openrouter.ai/models)
- [OpenAI](https://platform.openai.com/docs/models)
- [xAI](https://docs.x.ai/docs/models)
- [Google](https://ai.google.dev/gemini-api/docs/models)

Or run local models with:
- [LM Studio](https://lmstudio.ai)
- [Ollama](https://ollama.com)
- [vLLM](https://github.com/vllm-project/vllm)

...Or use any other OpenAI /v1/chat/completions compatible API server.

---

### And more:
- Supports image attachments when using a vision model (like gpt-5, grok-4, claude-4, etc.)
- Supports text file attachments (.txt, .py, .c, etc.)
- Customizable personality (aka system prompt)
- Distinguishes users via their Discord IDs
- Streamed responses (turns green when complete, automatically splits into separate messages when too long)
- Hot reloading config (you can change settings without restarting the bot)
- Displays helpful warnings when appropriate (like "⚠️ Only using last 25 messages" when the customizable message limit is exceeded)
- Caches message data in a size-managed (no memory leaks) and mutex-protected (no race conditions) global dictionary to maximize efficiency and minimize Discord API calls
- Fully asynchronous
- Hot-reloaded YAML config, env-backed secrets, and a `uv` project layout (`main.py` entrypoint)
- Per-guild DQN (`/learn`, `/act`, `/brain`): stay / reply / react, replay buffer, 150 s reaction rewards. Inspectable; not a hosted fine-tune.

## Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/jakobdylanc/llmcord
   cd llmcord
   ```

2. Set up `config.yaml`:

> Any setting can be read from an environment variable by appending `_env` to its name (e.g. `bot_token_env: DISCORD_BOT_TOKEN`).

### Discord settings:

| Setting | Description |
| --- | --- |
| **bot_token** | Create a new Discord bot at [discord.com/developers/applications](https://discord.com/developers/applications) and generate a token under the "Bot" tab. Also enable "MESSAGE CONTENT INTENT". |
| **client_id** | Found under the "OAuth2" tab of the Discord bot you just made. |
| **status_message** | Set a custom message that displays on the bot's Discord profile.<br /><br />**Max 128 characters.** |
| **max_text** | The maximum amount of text allowed in a single message, including text from file attachments.<br /><br />Default: `100,000` |
| **max_images** | The maximum number of image attachments allowed in a single message.<br /><br />Default: `5`<br /><br />**Only applicable when using a vision model.** |
| **max_messages** | The maximum number of messages allowed in a reply chain. When exceeded, the oldest messages are dropped.<br /><br />Default: `25` |
| **use_plain_responses** | When set to `true` the bot will use plaintext responses instead of embeds. Plaintext responses have a shorter character limit so the bot's messages may split more often.<br /><br />Default: `false`<br /><br />**Also disables streamed responses and warning messages.** |
| **allow_dms** | Set to `false` to disable direct message access.<br /><br />Default: `true` |
| **permissions** | Configure access permissions for `users`, `roles` and `channels`, each with a list of `allowed_ids` and `blocked_ids`.<br /><br />Control which `users` are admins with `admin_ids`. Admins can change the model with `/model` and DM the bot even if `allow_dms` is `false`.<br /><br />**Leave `allowed_ids` empty to allow ALL in that category.**<br /><br />**Role and channel permissions do not affect DMs.**<br /><br />**You can use [category](https://support.discord.com/hc/en-us/articles/115001580171-Channel-Categories-101) IDs to control channel permissions in groups.** |

### LLM settings:

| Setting | Description |
| --- | --- |
| **providers** | Add the LLM providers you want to use, each with a `base_url` and optional `api_key` entry. This checkout defaults to `x-ai` at `https://api.x.ai/v1` using `XAI_API_KEY`. Popular providers (`openrouter`, `openai`, `ollama`, etc.) are also included.<br /><br />**Only supports OpenAI /v1/chat/completions compatible APIs.**<br /><br />**Some providers may need `extra_headers` / `extra_query` / `extra_body` entries for extra HTTP data. See the included `azure-openai` provider for an example.** |
| **models** | Add the models you want to use in `<provider>/<model>: <parameters>` format (examples are included). When you run `/model` these models will show up as autocomplete suggestions.<br /><br />**Refer to each provider's documentation for supported parameters.**<br /><br />**The first model in your `models` list will be the default model at startup.**<br /><br />**Some vision models may need `:vision` added to the end of their name to enable image support.** |
| **system_prompt** | Write anything you want to customize the bot's behavior!<br /><br />**Leave blank for no system prompt.**<br /><br />**You can use the `{date}` and `{time}` tags in your system prompt to insert the current date and time, based on your host computer's time zone.**<br /><br />**It is recommended to include something like `"User messages are prefixed with their Discord ID as <@ID>. Use this format to mention users."` in your system prompt to help the bot understand the user message format.** |

3. Run the bot:

   **uv (preferred):**
   ```bash
   cp .env.example .env   # set DISCORD_BOT_TOKEN, DISCORD_CLIENT_ID, XAI_API_KEY
   uv sync
   uv run python main.py --check-config
   uv run python main.py
   ```

   **pip:**
   ```bash
   python -m pip install -U -r requirements.txt
   python main.py
   ```

   **Docker:**
   ```bash
   docker compose up
   ```

   `uv run pytest` and `uv run ruff check .` are the local gate. `python llmcord.py` still works.

   Admins: `/learn on` then `/act on` in a private guild before expecting unsolicited stay/reply/react. Set `DISCORD_DEV_GUILD_ID` so slash commands register guild-scoped. Brains persist under `~/.abbey/`.

## Notes

- If you're having issues, try my suggestions [here](https://github.com/jakobdylanc/llmcord/issues/19)

- PRs are welcome :)

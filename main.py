"""uv entrypoint: `uv run python main.py`"""

from __future__ import annotations

import argparse
import sys

from settings import ConfigError, default_model


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Talk to LLMs from Discord (llmcord).")
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Validate config.yaml and required env vars, then exit.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to YAML config (default: config.yaml)",
    )
    args = parser.parse_args(argv)

    from llmcord import configure
    from llmcord import main as run_bot

    try:
        config = configure(args.config, init_store=not args.check_config)
    except (ConfigError, OSError) as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 1

    if args.check_config:
        print(f"config ok (default model: {default_model(config['models'])})")
        return 0

    run_bot(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

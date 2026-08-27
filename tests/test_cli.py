from pathlib import Path

import pytest

from main import main


def test_check_config_fails_without_bot_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISCORD_BOT_TOKEN", raising=False)
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert main(["--check-config"]) == 1


def test_check_config_passes_with_required_env(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    assert main(["--check-config"]) == 0
    captured = capsys.readouterr()
    assert "x-ai/grok-4.6" in captured.out


def test_check_config_accepts_explicit_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISCORD_BOT_TOKEN", "test-token")
    path = Path(__file__).resolve().parents[1] / "config.yaml"
    assert main(["--check-config", "--config", str(path)]) == 0


def test_importing_llmcord_does_not_connect() -> None:
    import llmcord

    assert not llmcord.discord_bot.is_ready()
    assert llmcord.discord_bot.is_closed() is False

import math

import pytest

from encoder import Intent, classify, encode_state, sentiment
from social import update_reputation


def test_intent_order_is_spec_order() -> None:
    assert [i.value for i in Intent] == [
        "question",
        "greeting",
        "modRequest",
        "memoryStore",
        "personaSwitch",
        "repQuery",
        "smallTalk",
        "command",
        "unknown",
    ]


def test_classify_rules_first_match_wins() -> None:
    assert classify("/model") is Intent.command
    assert classify("!ping") is Intent.command
    assert classify("please remember this") is Intent.memory_store
    assert classify("note that Abbey is here") is Intent.memory_store
    assert classify("what's my reputation") is Intent.rep_query
    assert classify("be aviva") is Intent.persona_switch
    assert classify("what is a lock") is Intent.question
    assert classify("hello there") is Intent.greeting


def test_classify_fallthrough_is_smalltalk_not_unknown() -> None:
    assert classify("pizza") is Intent.small_talk


def test_encode_length_question_mention_and_hour() -> None:
    v = encode_state(
        text="hello?",
        reputation=0.5,
        mentions_bot=True,
        has_image=False,
        hour=0,
        channel_heat=0,
    )
    assert len(v) == 18
    assert v[11] == 1.0
    assert v[12] == 1.0
    assert v[13] == 0.0
    assert v[14] == pytest.approx(0.0)
    assert v[15] == pytest.approx(1.0)
    assert v[Intent.question.index] == 1.0
    assert sum(v[0:9]) == pytest.approx(1.0)


def test_encode_greeting_without_question_mark() -> None:
    v = encode_state(text="hello there", reputation=0.5, mentions_bot=False, has_image=False, hour=0, channel_heat=0)
    assert v[Intent.greeting.index] == 1.0
    assert v[12] == 0.0


def test_encode_heat_and_length_caps() -> None:
    v = encode_state(text="x" * 800, reputation=1.0, mentions_bot=False, has_image=True, hour=6, channel_heat=100)
    assert v[10] == 1.0
    assert v[13] == 1.0
    assert v[16] == 1.0
    assert v[14] == pytest.approx(math.sin(2 * math.pi * 6 / 24))


def test_sentiment_positive_and_negative() -> None:
    assert sentiment("love this, great job thanks") > 0
    assert sentiment("this is trash and awful") < 0
    assert sentiment("") == 0.0


def test_reputation_ema_matches_0_5_to_0_525() -> None:
    assert update_reputation(0.5, target=1.0) == pytest.approx(0.525)
    assert 0.0 <= update_reputation(0.01, target=0.0) <= 1.0

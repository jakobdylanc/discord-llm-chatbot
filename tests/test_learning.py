from datetime import UTC, datetime, timedelta

from brain import Experience
from learning import Action, LearningStore, command_sync_mode, compute_reward, decide_action, emoji_score


def test_command_sync_skips_global_without_override() -> None:
    assert command_sync_mode(None, False) == "skip"
    assert command_sync_mode("123", False) == "guild"
    assert command_sync_mode(None, True) == "global"


def test_new_guild_learning_and_act_are_off(tmp_path) -> None:
    store = LearningStore(tmp_path)
    assert store.is_learning("g1") is False
    assert store.is_act("g1") is False


def test_policy_ignored_when_act_off() -> None:
    decision = decide_action(forced=False, learning=True, act=False, cooldown_ok=True, budget_ok=True, select=lambda: 1)
    assert decision.kind == "ignore"
    assert decision.learn is False
    assert decision.reason == "act_off"


def test_mention_forces_reply_even_if_policy_would_stay() -> None:
    decision = decide_action(forced=True, learning=True, act=True, cooldown_ok=True, budget_ok=True, select=lambda: 0)
    assert decision.kind == "reply"
    assert decision.action is Action.REPLY
    assert decision.learn is True
    assert decision.reason == "forced"


def test_forced_reply_does_not_learn_when_learning_off() -> None:
    decision = decide_action(forced=True, learning=False, act=False, cooldown_ok=True, budget_ok=True, select=lambda: 1)
    assert decision.kind == "reply"
    assert decision.learn is False


def test_stay_learns_without_spending_budget() -> None:
    decision = decide_action(forced=False, learning=True, act=True, cooldown_ok=True, budget_ok=False, select=lambda: 0)
    assert decision.kind == "stay"
    assert decision.learn is True


def test_over_budget_neither_acts_nor_learns() -> None:
    decision = decide_action(forced=False, learning=True, act=True, cooldown_ok=True, budget_ok=False, select=lambda: 1)
    assert decision.kind == "ignore"
    assert decision.learn is False
    assert decision.reason == "over_budget"


def test_cooldown_neither_acts_nor_learns() -> None:
    decision = decide_action(forced=False, learning=True, act=True, cooldown_ok=False, budget_ok=True, select=lambda: 2)
    assert decision.kind == "ignore"
    assert decision.reason == "cooldown"


def test_emoji_score_maps_known_reactions() -> None:
    assert emoji_score("👍") == 1
    assert emoji_score("✅") == 1
    assert emoji_score("❤️") == 1
    assert emoji_score("🔥") == 1
    assert emoji_score("👎") == -1
    assert emoji_score("💀") == -1
    assert emoji_score("😀") == 0


def test_settle_thumbs_up_caps_at_three() -> None:
    assert compute_reward(positives=5, negatives=0, human_reply=False, deleted=False, action=Action.REPLY) == 3.0


def test_settle_silence_and_delete() -> None:
    assert compute_reward(0, 0, False, False, Action.REPLY) == -0.2
    assert compute_reward(0, 0, False, True, Action.REPLY) == -2.0
    assert compute_reward(1, 0, True, False, Action.REPLY) == 1.5


def test_budget_blocks_seventh_unsolicited_action(tmp_path) -> None:
    store = LearningStore(tmp_path)
    store.set_learning("g1", True)
    store.set_act("g1", True)
    now = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    for i in range(6):
        assert store.budget_ok("g1", now + timedelta(minutes=i))
        store.spend_budget("g1", now + timedelta(minutes=i))
    assert store.budget_ok("g1", now + timedelta(minutes=6)) is False
    later = now + timedelta(hours=1, minutes=1)
    assert store.budget_ok("g1", later) is True


def test_pending_settle_writes_experience(tmp_path) -> None:
    store = LearningStore(tmp_path)
    store.set_learning("g1", True)
    state = [0.0] * 18
    opened = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    store.open_pending(
        guild_id="g1",
        message_id="m1",
        action=Action.REPLY,
        state=state,
        opened_at=opened,
    )
    store.note_reaction("m1", "👍")
    settled = store.settle_due(opened + timedelta(seconds=151))
    assert len(settled) == 1
    assert settled[0].action == Action.REPLY
    assert settled[0].reward == 1.0
    assert len(store.brain("g1").buffer) == 1


def test_remember_stay_records_reward_zero_and_reloads(tmp_path) -> None:
    store = LearningStore(tmp_path)
    store.set_learning("g1", True)
    state = [0.0] * 18
    store.remember_stay("g1", state)
    exp = list(store.brain("g1").buffer._storage)
    assert len(exp) == 1
    assert exp[0].action == Action.STAY
    assert exp[0].reward == 0.0
    store.close()
    reloaded = LearningStore(tmp_path)
    again = list(reloaded.brain("g1").buffer._storage)
    assert len(again) == 1
    assert again[0].action == Action.STAY
    assert again[0].reward == 0.0


def test_deleted_pending_settles_minus_two_immediately(tmp_path) -> None:
    store = LearningStore(tmp_path)
    store.set_learning("g1", True)
    opened = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    store.open_pending(
        guild_id="g1",
        message_id="m-del",
        action=Action.REPLY,
        state=[0.0] * 18,
        opened_at=opened,
    )
    store.note_deleted("m-del")
    settled = store.settle_due(opened + timedelta(seconds=1))
    assert len(settled) == 1
    assert settled[0].reward == -2.0


def test_channel_cooldown_blocks_then_allows(tmp_path) -> None:
    store = LearningStore(tmp_path)
    now = datetime(2026, 8, 27, 12, 0, tzinfo=UTC)
    store.mark_unsolicited("c1", now)
    assert store.cooldown_ok("c1", "g1", now + timedelta(seconds=1)) is False
    assert store.cooldown_ok("c1", "g1", now + timedelta(seconds=20)) is True


def test_guild_brains_do_not_share_weights(tmp_path) -> None:
    store = LearningStore(tmp_path)
    a = store.brain("g1")
    b = store.brain("g2")
    state = [0.0] * 18
    nxt = [0.0] * 18
    a.remember(Experience(state, 1, 1.0, nxt, True))
    assert len(b.buffer) == 0
    store.save()
    other = LearningStore(tmp_path)
    assert len(other.brain("g1").buffer) == 1
    assert len(other.brain("g2").buffer) == 0

from dataclasses import replace

import pytest

from brain import DQNAgent, Experience, NeuralNetwork


def test_forward_rejects_wrong_width() -> None:
    net = NeuralNetwork([18, 64, 32, 3], seed=0)
    with pytest.raises(ValueError, match="width"):
        net.forward([0.0] * 17)


def test_forward_returns_output_width() -> None:
    net = NeuralNetwork([18, 64, 32, 3], seed=0)
    out = net.forward([0.0] * 18)
    assert len(out) == 3


def test_default_agent_is_18_in_three_linear_q() -> None:
    agent = DQNAgent()
    assert agent.online.topology == [18, 64, 32, 3]
    assert agent.online.output_activation == "linear"
    q = agent.q_values([0.0] * 18)
    assert len(q) == 3
    assert abs(sum(q) - 1.0) > 0.5


def test_train_drops_loss_on_one_hot_target() -> None:
    net = NeuralNetwork([4, 8, 3], seed=0)
    x = [1.0, 0.0, 0.0, 0.0]
    target = [0.0, 1.0, 0.0]
    before = (net.forward(x)[1] - 1.0) ** 2
    for _ in range(2000):
        net.train(x, target, lr=0.01)
    after = (net.forward(x)[1] - 1.0) ** 2
    assert after < before * 0.2


def test_learn_is_noop_until_batch_is_full() -> None:
    agent = DQNAgent(topology=[4, 8, 3], buffer_capacity=32, batch_size=16, seed=0)
    state = [1.0, 0.0, 0.0, 0.0]
    nxt = [0.0, 1.0, 0.0, 0.0]
    for _ in range(15):
        agent.remember(Experience(state, 1, 1.0, nxt, True))
        agent.learn()
    assert agent.step_count == 0
    agent.remember(Experience(state, 1, 1.0, nxt, True))
    agent.learn()
    assert agent.step_count == 1


def test_dqn_prefers_rewarded_action() -> None:
    agent = DQNAgent(topology=[4, 16, 3], buffer_capacity=256, batch_size=16, seed=1)
    state = [1.0, 0.0, 0.0, 0.0]
    next_state = [0.0, 1.0, 0.0, 0.0]
    for _ in range(80):
        agent.remember(Experience(state, action=1, reward=1.0, next_state=next_state, done=True))
        agent.learn()
    q = agent.q_values(state)
    assert q.index(max(q)) == 1


def test_snapshot_roundtrip_preserves_q() -> None:
    agent = DQNAgent(topology=[18, 64, 32, 3], seed=2)
    state = [0.0] * 18
    state[0] = 1.0
    before = agent.q_values(state)
    snap = agent.export_weights()
    other = DQNAgent(topology=[18, 64, 32, 3], seed=99)
    other.import_weights(snap)
    assert other.q_values(state) == pytest.approx(before)


def test_snapshot_rejects_topology_drift() -> None:
    agent = DQNAgent(topology=[18, 64, 32, 3], seed=3)
    snap = replace(agent.export_weights(), topology=(18, 32, 3))
    before = agent.export_weights()
    agent.import_weights(snap)
    assert agent.export_weights().topology == before.topology
    assert agent.export_weights().layers == before.layers

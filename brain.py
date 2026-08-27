from __future__ import annotations

import math
import random
from copy import deepcopy
from dataclasses import dataclass, field


def _clip(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return min(hi, max(lo, value))


def _relu(value: float) -> float:
    return value if value > 0.0 else 0.0


@dataclass
class DenseLayer:
    input_count: int
    output_count: int
    weights: list[float]
    biases: list[float]

    @classmethod
    def he(cls, input_count: int, output_count: int, rng: random.Random) -> DenseLayer:
        scale = math.sqrt(2.0 / input_count)
        weights = [rng.uniform(-scale, scale) for _ in range(input_count * output_count)]
        return cls(input_count, output_count, weights, [0.0] * output_count)

    def dot(self, row: int, inputs: list[float]) -> float:
        base = row * self.input_count
        total = 0.0
        for col, x in enumerate(inputs):
            total += self.weights[base + col] * x
        return total


class NeuralNetwork:
    def __init__(self, topology: list[int], seed: int | None = None, output_activation: str = "linear") -> None:
        if len(topology) < 2:
            raise ValueError("topology needs at least an input and an output layer")
        self.topology = list(topology)
        self.output_activation = output_activation
        rng = random.Random(seed)
        self.layers = [DenseLayer.he(topology[i], topology[i + 1], rng) for i in range(len(topology) - 1)]

    def copy(self) -> NeuralNetwork:
        return deepcopy(self)

    def forward(self, inputs: list[float]) -> list[float]:
        return self.forward_retaining(inputs)[0]

    def forward_retaining(self, inputs: list[float]) -> tuple[list[float], list[list[float]], list[list[float]]]:
        if len(inputs) != self.topology[0]:
            raise ValueError(f"input width {len(inputs)} != {self.topology[0]}")
        pre: list[list[float]] = []
        post: list[list[float]] = [list(inputs)]
        activation = list(inputs)
        for idx, layer in enumerate(self.layers):
            z = [layer.dot(row, activation) + layer.biases[row] for row in range(layer.output_count)]
            pre.append(z)
            is_output = idx == len(self.layers) - 1
            activation = list(z) if is_output else [_relu(v) for v in z]
            post.append(activation)
        return activation, pre, post

    def train(self, inputs: list[float], target: list[float], lr: float = 0.001) -> None:
        output, pre, post = self.forward_retaining(inputs)
        if len(target) != len(output):
            raise ValueError("target width must match output width")
        delta = [o - t for o, t in zip(output, target, strict=True)]
        for idx in range(len(self.layers) - 1, -1, -1):
            layer = self.layers[idx]
            input_to_layer = post[idx]
            next_delta = [0.0] * layer.input_count
            for row in range(layer.output_count):
                d = _clip(delta[row])
                if d == 0.0:
                    continue
                base = row * layer.input_count
                for col in range(layer.input_count):
                    next_delta[col] += layer.weights[base + col] * d
                    layer.weights[base + col] -= lr * d * input_to_layer[col]
                layer.biases[row] -= lr * d
            if idx > 0:
                z = pre[idx - 1]
                next_delta = [g if z[col] > 0 else 0.0 for col, g in enumerate(next_delta)]
            delta = next_delta


@dataclass(frozen=True)
class Experience:
    state: list[float]
    action: int
    reward: float
    next_state: list[float]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int = 10_000, rng: random.Random | None = None) -> None:
        self.capacity = capacity
        self._rng = rng or random.Random()
        self._storage: list[Experience] = []
        self._write = 0

    def __len__(self) -> int:
        return len(self._storage)

    def append(self, experience: Experience) -> None:
        if len(self._storage) < self.capacity:
            self._storage.append(experience)
            return
        self._storage[self._write] = experience
        self._write = (self._write + 1) % self.capacity

    def sample(self, size: int) -> list[Experience]:
        if not self._storage:
            return []
        return [self._storage[self._rng.randrange(len(self._storage))] for _ in range(size)]


@dataclass
class LayerSnapshot:
    weights: list[float]
    biases: list[float]


@dataclass
class BrainSnapshot:
    topology: tuple[int, ...]
    layers: list[LayerSnapshot]
    epsilon: float
    step_count: int
    buffer: list[Experience] = field(default_factory=list)


class DQNAgent:
    def __init__(
        self,
        topology: list[int] | None = None,
        buffer_capacity: int = 10_000,
        batch_size: int = 64,
        seed: int | None = None,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_min: float = 0.01,
        epsilon_decay: float = 0.995,
        target_sync_interval: int = 100,
    ) -> None:
        topology = list(topology or [18, 64, 32, 3])
        self.rng = random.Random(seed)
        self.online = NeuralNetwork(topology, seed=seed)
        self.target = self.online.copy()
        self.buffer = ReplayBuffer(buffer_capacity, rng=self.rng)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.target_sync_interval = target_sync_interval
        self.step_count = 0
        self.action_histogram = [0, 0, 0]
        self.last_q: list[float] = [0.0, 0.0, 0.0]
        self.recent_rewards: list[float] = []

    @property
    def action_count(self) -> int:
        return self.online.topology[-1]

    def q_values(self, state: list[float]) -> list[float]:
        return self.online.forward(state)

    def select_action(self, state: list[float]) -> int:
        self.last_q = self.q_values(state)
        if self.rng.random() < self.epsilon:
            action = self.rng.randrange(self.action_count)
        else:
            q = self.last_q
            action = max(range(len(q)), key=lambda i: q[i])
        if action < len(self.action_histogram):
            self.action_histogram[action] += 1
        return action

    def remember(self, experience: Experience) -> None:
        self.buffer.append(experience)
        self.recent_rewards.append(experience.reward)
        self.recent_rewards = self.recent_rewards[-50:]

    def learn(self) -> None:
        if len(self.buffer) < self.batch_size:
            return
        for exp in self.buffer.sample(self.batch_size):
            future = 0.0 if exp.done else max(self.target.forward(exp.next_state))
            target_q = exp.reward + self.gamma * future
            predicted = self.online.forward(exp.state)
            self.online.train(exp.state, self._make_target(predicted, exp.action, target_q))
        self.step_count += 1
        if self.step_count % self.target_sync_interval == 0:
            self.target = self.online.copy()
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def export_weights(self) -> BrainSnapshot:
        return BrainSnapshot(
            topology=tuple(self.online.topology),
            layers=[LayerSnapshot(list(layer.weights), list(layer.biases)) for layer in self.online.layers],
            epsilon=self.epsilon,
            step_count=self.step_count,
            buffer=list(self.buffer._storage),
        )

    def import_weights(self, snapshot: BrainSnapshot) -> None:
        if tuple(snapshot.topology) != tuple(self.online.topology):
            return
        if len(snapshot.layers) != len(self.online.layers):
            return
        for layer, saved in zip(self.online.layers, snapshot.layers, strict=True):
            if len(saved.weights) != len(layer.weights) or len(saved.biases) != len(layer.biases):
                return
        for layer, saved in zip(self.online.layers, snapshot.layers, strict=True):
            layer.weights = list(saved.weights)
            layer.biases = list(saved.biases)
        self.target = self.online.copy()
        self.epsilon = snapshot.epsilon
        self.step_count = snapshot.step_count
        for exp in snapshot.buffer:
            self.buffer.append(exp)

    @staticmethod
    def _make_target(predicted: list[float], action: int, value: float) -> list[float]:
        target = list(predicted)
        target[action] = value
        return target

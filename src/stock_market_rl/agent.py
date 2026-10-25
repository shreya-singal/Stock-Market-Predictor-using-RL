from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np

from .features import build_state_vector


class QTradingAgent:
    def __init__(
        self,
        window_size: int = 20,
        action_space: Sequence[int] = (0, 1, 2),
        learning_rate: float = 0.15,
        discount: float = 0.95,
        epsilon: float = 0.2,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        random_seed: int = 42,
    ) -> None:
        self.window_size = window_size
        self.action_space = tuple(action_space)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.random_seed = random_seed
        self.q_table: dict[tuple[float, ...], np.ndarray] = defaultdict(lambda: np.zeros(len(self.action_space)))
        np.random.seed(self.random_seed)

    def _state_key(self, state: np.ndarray | Sequence[float]) -> tuple[float, ...]:
        arr = np.asarray(state, dtype=float)
        if arr.size == 0:
            raise ValueError("State cannot be empty.")
        return tuple(np.round(arr, 4))

    def choose_action(self, state: np.ndarray | Sequence[float]) -> int:
        key = self._state_key(state)
        if np.random.random() < self.epsilon:
            return int(np.random.choice(self.action_space))
        action_values = self.q_table.get(key, np.zeros(len(self.action_space)))
        return int(self.action_space[int(np.argmax(action_values))])

    def update(self, state, action, reward, next_state, done: bool) -> None:
        state_key = self._state_key(state)
        next_key = self._state_key(next_state)
        q_values = self.q_table[state_key]
        future_value = 0.0 if done else float(np.max(self.q_table.get(next_key, np.zeros(len(self.action_space)))))
        target = reward + self.discount * future_value
        q_values[int(action)] += self.learning_rate * (target - q_values[int(action)])
        self.q_table[state_key] = q_values
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def train(
        self,
        prices: np.ndarray | list[float],
        episodes: int = 150,
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
    ) -> dict[str, float]:
        series = np.asarray(prices, dtype=float)
        if series.size < self.window_size + 2:
            raise ValueError("Price series is too short for training.")

        episode_returns: list[float] = []
        for _ in range(episodes):
            cash = float(initial_cash)
            position = 0
            entry_price = 0.0
            for idx in range(self.window_size, len(series) - 1):
                state = build_state_vector(series[: idx + 1], self.window_size)
                action = self.choose_action(state)
                current_price = float(series[idx])
                next_price = float(series[idx + 1])
                reward = 0.0

                if action == 1 and position == 0:
                    cash -= current_price * (1.0 + transaction_cost)
                    position = 1
                    entry_price = current_price
                elif action == 2 and position > 0:
                    pnl = (current_price - entry_price) * position
                    reward = pnl - current_price * transaction_cost * position
                    cash += current_price * (1.0 - transaction_cost) * position
                    position = 0
                    entry_price = 0.0
                elif position > 0:
                    reward = (current_price - entry_price) * position * 0.05

                next_state = build_state_vector(series[: idx + 2], self.window_size)
                self.update(state, action, reward, next_state, done=False)

            if position > 0:
                cash += position * series[-1] * (1.0 - transaction_cost)
                position = 0

            episode_returns.append((cash / initial_cash) - 1.0)

        return {
            "mean_return": float(np.mean(episode_returns)),
            "best_return": float(np.max(episode_returns)),
            "final_return": float(episode_returns[-1]),
        }

    def evaluate_strategy(
        self,
        prices: np.ndarray | list[float],
        initial_cash: float = 10_000.0,
        transaction_cost: float = 0.001,
    ) -> dict[str, float]:
        series = np.asarray(prices, dtype=float)
        cash = float(initial_cash)
        position = 0
        entry_price = 0.0
        trade_count = 0

        for idx in range(self.window_size, len(series) - 1):
            state = build_state_vector(series[: idx + 1], self.window_size)
            action = self.choose_action(state)
            current_price = float(series[idx])

            if action == 1 and position == 0:
                cash -= current_price * (1.0 + transaction_cost)
                position = 1
                entry_price = current_price
                trade_count += 1
            elif action == 2 and position > 0:
                cash += current_price * (1.0 - transaction_cost)
                position = 0
                entry_price = 0.0
                trade_count += 1

        if position > 0:
            cash += position * series[-1] * (1.0 - transaction_cost)

        final_value = cash
        total_return = (final_value / initial_cash) - 1.0
        return {
            "final_cash": float(final_value),
            "total_return": float(total_return),
            "trades": int(trade_count),
            "portfolio_value": float(final_value),
        }

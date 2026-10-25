import numpy as np

from stock_market_rl.agent import QTradingAgent
from stock_market_rl.evaluation import compute_metrics
from stock_market_rl.features import build_state_vector


def test_build_state_vector_shape():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0])
    state = build_state_vector(prices, window_size=3)
    assert state.shape == (3,)
    assert np.isfinite(state).all()


def test_q_agent_update_changes_q_table():
    agent = QTradingAgent(window_size=3, action_space=[0, 1, 2], learning_rate=0.1, discount=0.9, epsilon=0.0)
    state = (0.1, 0.2, -0.1)
    before = agent.q_table.get(state, np.zeros(3)).copy()
    agent.update(state, 1, 0.5, (0.2, 0.3, 0.0), done=False)
    after = agent.q_table.get(state, np.zeros(3))
    assert not np.allclose(after, before)


def test_compute_metrics_basic_return():
    prices = np.array([100.0, 105.0, 110.0, 108.0])
    metrics = compute_metrics(prices)
    assert 'total_return' in metrics
    assert metrics['total_return'] > 0


def test_pipeline_smoke():
    prices = np.array([100.0, 101.0, 102.0, 101.5, 103.0, 104.0, 103.0, 105.0])
    agent = QTradingAgent(window_size=3, action_space=[0, 1, 2], learning_rate=0.1, discount=0.9, epsilon=0.0)
    metrics = agent.evaluate_strategy(prices, initial_cash=1000.0)
    assert 'final_cash' in metrics
    assert 'total_return' in metrics

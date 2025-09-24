"""
Unit tests for the stock analysis functions.
"""

import pytest
import numpy as np
from agentic_lab.agents.trader.stock_analysis.analyzer import (
    calculate_moving_average,
    calculate_exponential_moving_average,
    calculate_rsi,
    calculate_macd,
    calculate_bollinger_bands,
    calculate_stochastic_oscillator,
    calculate_average_true_range,
    calculate_on_balance_volume,
)


@pytest.fixture
def sample_data():
    """Fixture to provide sample stock data for testing."""
    return [
        100,
        102,
        105,
        103,
        106,
        108,
        110,
        112,
        115,
        113,
        116,
        118,
        120,
        119,
        122,
    ]


def test_calculate_moving_average(sample_data):
    """Test the calculate_moving_average function."""
    result = calculate_moving_average(sample_data, 5)
    assert isinstance(result, list)
    assert len(result) == len(sample_data)
    # The first 4 values should be NaN, which are converted to None by tolist()
    assert all(x is not None for x in result[4:])
    assert result[4] == pytest.approx(103.2)
    assert result[-1] == pytest.approx(119.0)


def test_calculate_exponential_moving_average(sample_data):
    """Test the calculate_exponential_moving_average function."""
    result = calculate_exponential_moving_average(sample_data, 5)
    assert isinstance(result, list)
    assert len(result) == len(sample_data)
    assert all(isinstance(x, float) for x in result)
    assert result[0] == pytest.approx(100.0)
    assert result[-1] == pytest.approx(118.85, abs=1e-2)


def test_calculate_rsi(sample_data):
    """Test the calculate_rsi function."""
    result = calculate_rsi(sample_data, 14)
    assert isinstance(result, list)
    assert len(result) == len(sample_data)
    assert result[-1] == pytest.approx(84.375, abs=1e-2)


def test_calculate_macd(sample_data):
    """Test the calculate_macd function."""
    macd_line, signal_line, histogram = calculate_macd(sample_data)
    assert isinstance(macd_line, list)
    assert isinstance(signal_line, list)
    assert isinstance(histogram, list)
    assert len(macd_line) == len(sample_data)
    assert macd_line[-1] == pytest.approx(5.14, abs=1e-2)
    assert signal_line[-1] == pytest.approx(3.69, abs=1e-2)
    assert histogram[-1] == pytest.approx(1.45, abs=1e-2)


def test_calculate_bollinger_bands(sample_data):
    """Test the calculate_bollinger_bands function."""
    upper, middle, lower = calculate_bollinger_bands(sample_data, 5)
    assert isinstance(upper, list)
    assert isinstance(middle, list)
    assert isinstance(lower, list)
    assert len(upper) == len(sample_data)
    assert middle[4] == pytest.approx(103.2)
    assert upper[4] == pytest.approx(103.2 + 2 * np.std(sample_data[:5], ddof=1), abs=1e-2)
    assert lower[4] == pytest.approx(103.2 - 2 * np.std(sample_data[:5], ddof=1), abs=1e-2)
    assert middle[-1] == pytest.approx(119.0)
    assert upper[-1] == pytest.approx(119.0 + 2 * np.std(sample_data[-5:], ddof=1), abs=1e-2)
    assert lower[-1] == pytest.approx(119.0 - 2 * np.std(sample_data[-5:], ddof=1), abs=1e-2)


def test_calculate_stochastic_oscillator(sample_data):
    """Test the calculate_stochastic_oscillator function."""
    high = [x + 2 for x in sample_data]
    low = [x - 2 for x in sample_data]
    percent_k, percent_d = calculate_stochastic_oscillator(high, low, sample_data)
    assert isinstance(percent_k, list)
    assert isinstance(percent_d, list)
    assert len(percent_k) == len(sample_data)
    assert percent_k[-1] == pytest.approx(91.67, abs=1e-2)
    assert percent_d[-1] == pytest.approx(89.58, abs=1e-2)


def test_calculate_average_true_range(sample_data):
    """Test the calculate_average_true_range function."""
    high = [x + 2 for x in sample_data]
    low = [x - 2 for x in sample_data]
    atr = calculate_average_true_range(high, low, sample_data)
    assert isinstance(atr, list)
    assert len(atr) == len(sample_data)
    assert atr[-1] == pytest.approx(4.23, abs=1e-2)


def test_calculate_on_balance_volume(sample_data):
    """Test the calculate_on_balance_volume function."""
    volume = [1000 * (i + 1) for i in range(len(sample_data))]
    obv = calculate_on_balance_volume(sample_data, volume)
    assert isinstance(obv, list)
    assert len(obv) == len(sample_data)
    assert obv[-1] == 63000.0


def test_invalid_input_types():
    """Test functions with invalid input types."""
    with pytest.raises(TypeError):
        calculate_moving_average("not a list", 5)
    with pytest.raises(ValueError):
        calculate_moving_average([1, 2, 3], 0)
    with pytest.raises(TypeError):
        calculate_exponential_moving_average([1, "b", 3], 5)
    with pytest.raises(ValueError):
        calculate_exponential_moving_average([1, 2, 3], -1)

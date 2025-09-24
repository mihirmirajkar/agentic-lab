"""
This module provides a set of functions to calculate various technical analysis
indicators for stock data. These indicators are commonly used by traders to
analyze market trends and make decisions.
"""

from typing import List, Tuple
import numpy as np
import pandas as pd


def calculate_moving_average(data: List[float], window: int) -> List[float]:
    """
    Calculates the Simple Moving Average (SMA) for a given dataset.

    Args:
        data: A list of floats representing the price data.
        window: The number of periods to use for the moving average calculation.

    Returns:
        A list of floats representing the moving average.
    """
    if not isinstance(data, list) or not all(
        isinstance(x, (int, float)) for x in data
    ):
        raise TypeError("data must be a list of numbers.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    return pd.Series(data).rolling(window=window).mean().tolist()


def calculate_exponential_moving_average(
    data: List[float], window: int
) -> List[float]:
    """
    Calculates the Exponential Moving Average (EMA) for a given dataset.

    Args:
        data: A list of floats representing the price data.
        window: The number of periods to use for the EMA calculation.

    Returns:
        A list of floats representing the exponential moving average.
    """
    if not isinstance(data, list) or not all(
        isinstance(x, (int, float)) for x in data
    ):
        raise TypeError("data must be a list of numbers.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    return (
        pd.Series(data)
        .ewm(span=window, adjust=False)
        .mean()
        .tolist()
    )


def calculate_rsi(data: List[float], window: int = 14) -> List[float]:
    """
    Calculates the Relative Strength Index (RSI).

    Args:
        data: A list of floats representing the price data.
        window: The number of periods to use for the RSI calculation.

    Returns:
        A list of floats representing the RSI values.
    """
    if not isinstance(data, list) or not all(
        isinstance(x, (int, float)) for x in data
    ):
        raise TypeError("data must be a list of numbers.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if len(data) < window:
        return []

    series = pd.Series(data)
    delta = series.diff()

    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    no_loss_mask = (avg_loss == 0) & (avg_gain > 0)
    neutral_mask = (avg_gain == 0) & (avg_loss == 0)

    rsi = rsi.where(~no_loss_mask, 100.0)
    rsi = rsi.where(~neutral_mask, 50.0)

    return rsi.tolist()


def calculate_macd(
    data: List[float], fast_period: int = 12, slow_period: int = 26, signal_period: int = 9
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculates the Moving Average Convergence Divergence (MACD).

    Args:
        data: A list of floats representing the price data.
        fast_period: The number of periods for the fast EMA.
        slow_period: The number of periods for the slow EMA.
        signal_period: The number of periods for the signal line EMA.

    Returns:
        A tuple containing three lists: MACD line, signal line, and histogram.
    """
    if not isinstance(data, list) or not all(
        isinstance(x, (int, float)) for x in data
    ):
        raise TypeError("data must be a list of numbers.")
    if not all(
        isinstance(p, int) and p > 0
        for p in [fast_period, slow_period, signal_period]
    ):
        raise ValueError("All period arguments must be positive integers.")
    if slow_period <= fast_period:
        raise ValueError("slow_period must be greater than fast_period.")

    series = pd.Series(data)
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line.tolist(), signal_line.tolist(), histogram.tolist()


def calculate_bollinger_bands(
    data: List[float], window: int = 20, num_std_dev: int = 2
) -> Tuple[List[float], List[float], List[float]]:
    """
    Calculates the Bollinger Bands.

    Args:
        data: A list of floats representing the price data.
        window: The number of periods for the moving average.
        num_std_dev: The number of standard deviations.

    Returns:
        A tuple containing three lists: upper band, middle band (SMA), and lower band.
    """
    if not isinstance(data, list) or not all(
        isinstance(x, (int, float)) for x in data
    ):
        raise TypeError("data must be a list of numbers.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if not isinstance(num_std_dev, (int, float)) or num_std_dev <= 0:
        raise ValueError("num_std_dev must be a positive number.")

    series = pd.Series(data)
    middle_band = series.rolling(window=window).mean()
    std_dev = series.rolling(window=window).std()
    upper_band = middle_band + (std_dev * num_std_dev)
    lower_band = middle_band - (std_dev * num_std_dev)

    return upper_band.tolist(), middle_band.tolist(), lower_band.tolist()


def calculate_stochastic_oscillator(
    high: List[float],
    low: List[float],
    close: List[float],
    window: int = 14,
    k_smoothing: int = 3,
) -> Tuple[List[float], List[float]]:
    """
    Calculates the Stochastic Oscillator.

    Args:
        high: A list of floats representing the high prices.
        low: A list of floats representing the low prices.
        close: A list of floats representing the close prices.
        window: The number of periods for the oscillator calculation.
        k_smoothing: The number of periods for smoothing %K to get %D.

    Returns:
        A tuple containing two lists: %K (fast) and %D (slow).
    """
    if not all(
        isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
        for data in [high, low, close]
    ):
        raise TypeError("high, low, and close must be lists of numbers.")
    if len(high) != len(low) or len(low) != len(close):
        raise ValueError("All price lists must have the same length.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")
    if not isinstance(k_smoothing, int) or k_smoothing <= 0:
        raise ValueError("k_smoothing must be a positive integer.")

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    lowest_low = low_series.rolling(window=window).min()
    highest_high = high_series.rolling(window=window).max()

    price_range = highest_high - lowest_low
    safe_range = price_range.replace(0, np.nan)
    percent_k = 100 * (close_series - lowest_low) / safe_range
    percent_k = percent_k.where(price_range != 0, 0.0)
    percent_d = percent_k.rolling(window=k_smoothing, min_periods=1).mean()

    return percent_k.tolist(), percent_d.tolist()


def calculate_average_true_range(
    high: List[float], low: List[float], close: List[float], window: int = 14
) -> List[float]:
    """
    Calculates the Average True Range (ATR).

    Args:
        high: A list of floats representing the high prices.
        low: A list of floats representing the low prices.
        close: A list of floats representing the close prices.
        window: The number of periods for the ATR calculation.

    Returns:
        A list of floats representing the ATR values.
    """
    if not all(
        isinstance(data, list) and all(isinstance(x, (int, float)) for x in data)
        for data in [high, low, close]
    ):
        raise TypeError("high, low, and close must be lists of numbers.")
    if len(high) != len(low) or len(low) != len(close):
        raise ValueError("All price lists must have the same length.")
    if not isinstance(window, int) or window <= 0:
        raise ValueError("window must be a positive integer.")

    high_series = pd.Series(high)
    low_series = pd.Series(low)
    close_series = pd.Series(close)

    high_low = high_series - low_series
    high_close = np.abs(high_series - close_series.shift())
    low_close = np.abs(low_series - close_series.shift())

    tr = pd.DataFrame(
        {"high_low": high_low, "high_close": high_close, "low_close": low_close}
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / window, adjust=False).mean()

    return atr.tolist()


def calculate_on_balance_volume(
    close: List[float], volume: List[float]
) -> List[float]:
    """
    Calculates the On-Balance Volume (OBV).

    Args:
        close: A list of floats representing the close prices.
        volume: A list of floats representing the trading volume.

    Returns:
        A list of floats representing the OBV.
    """
    if not isinstance(close, list) or not all(
        isinstance(x, (int, float)) for x in close
    ):
        raise TypeError("close must be a list of numbers.")
    if not isinstance(volume, list) or not all(
        isinstance(x, (int, float)) for x in volume
    ):
        raise TypeError("volume must be a list of numbers.")
    if len(close) != len(volume):
        raise ValueError("close and volume lists must have the same length.")

    close_series = pd.Series(close)
    volume_series = pd.Series(volume)

    obv = [0.0] * len(close)
    for i in range(1, len(close)):
        if close[i] > close[i - 1]:
            obv[i] = obv[i - 1] + volume[i]
        elif close[i] < close[i - 1]:
            obv[i] = obv[i - 1] - volume[i]
        else:
            obv[i] = obv[i - 1]
    return obv

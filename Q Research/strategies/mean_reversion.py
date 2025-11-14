import pandas as pd

def mean_reversion_strategy(df: pd.DataFrame, short_window=20, long_window=50):
    """
    简单的均值回归策略（示例）：
    - 价格高于长期均线则卖出
    - 价格低于长期均线则买入
    """
    df = df.copy()
    df["short_ma"] = df["close"].rolling(window=short_window).mean()
    df["long_ma"] = df["close"].rolling(window=long_window).mean()

    df["signal"] = 0
    df.loc[df["short_ma"] > df["long_ma"], "signal"] = 1  # 买入
    df.loc[df["short_ma"] < df["long_ma"], "signal"] = -1  # 卖出

    return df
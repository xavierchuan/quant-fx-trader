# compute_indicators.py

import os
import pandas as pd
from loguru import logger

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
DERIVED_DATA_DIR = os.path.join(BASE_DIR, "data", "derived")
os.makedirs(DERIVED_DATA_DIR, exist_ok=True)

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # SMA
    df['SMA_20'] = df['close'].rolling(window=20).mean()

    # Bollinger Bands
    df['BB_MID'] = df['close'].rolling(window=20).mean()
    df['BB_STD'] = df['close'].rolling(window=20).std()
    df['BB_UPPER'] = df['BB_MID'] + 2 * df['BB_STD']
    df['BB_LOWER'] = df['BB_MID'] - 2 * df['BB_STD']

    # MACD
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

    # RSI
    def compute_rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    df['RSI_14'] = compute_rsi(df['close'])

    return df


if __name__ == "__main__":
    input_path = os.path.join(RAW_DATA_DIR, "EURUSD_H1.csv")
    output_path = os.path.join(DERIVED_DATA_DIR, "EURUSD_H1_with_indicators.csv")

    df = pd.read_csv(input_path, parse_dates=["time"])
    df = compute_indicators(df)
    df.to_csv(output_path, index=False)
    logger.info(f"✅ Saved with indicators to {output_path}")

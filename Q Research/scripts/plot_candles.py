# scripts/plot_candles.py

import os
import pandas as pd
import plotly.graph_objects as go

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

def plot_csv_candles(file_path, symbol, granularity):
    df = pd.read_csv(file_path, parse_dates=["time"])

    fig = go.Figure(data=[
        go.Candlestick(
            x=df["time"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"]
        )
    ])
    fig.update_layout(
        title=f"{symbol} {granularity} Candlestick Chart",
        xaxis_title="Time",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False
    )

    # 创建 charts 文件夹（如果不存在）
    os.makedirs("charts", exist_ok=True)
    output_file = f"charts/{symbol.replace('/', '')}_{granularity}.html"
    fig.write_html(output_file)
    print(f"✅ Saved chart to {output_file}")

if __name__ == "__main__":
    plot_csv_candles(os.path.join(RAW_DATA_DIR, "EURUSD_H1.csv"), "EUR/USD", "H1")

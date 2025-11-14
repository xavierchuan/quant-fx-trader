# scripts/plot_indicators.py
import os
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
DERIVED_DATA_DIR = os.path.join(BASE_DIR, "data", "derived")
CHARTS_DIR = os.path.join(BASE_DIR, "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

DATA_PATH = os.path.join(DERIVED_DATA_DIR, "EURUSD_H1_with_indicators.csv")

# 自动生成带时间戳的输出文件名
timestamp = datetime.now().strftime("%Y%m%d_%H%M")
OUTPUT_PATH = os.path.join(CHARTS_DIR, f"EURUSD_H1_with_indicators_{timestamp}.html")

df = pd.read_csv(DATA_PATH, parse_dates=["time"])

# 计算均线（若已存在将覆盖为最新计算）
if {"close"}.issubset(df.columns):
    df["SMA_20"] = df["close"].rolling(20).mean()
    df["SMA_200"] = df["close"].rolling(200).mean()

# 创建两行子图，共享x轴
fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                    vertical_spacing=0.1,
                    row_heights=[0.7, 0.3],
                    specs=[[{"type": "candlestick"}],
                           [{"secondary_y": True}]])

# 第一个子图：蜡烛图 + 均线
fig.add_trace(go.Candlestick(
    x=df['time'],
    open=df['open'],
    high=df['high'],
    low=df['low'],
    close=df['close'],
    name='Candlestick'
), row=1, col=1)

if 'SMA_20' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['SMA_20'],
        line=dict(color='blue', width=1),
        name='SMA 20'
    ), row=1, col=1)

if 'EMA_50' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['EMA_50'],
        line=dict(color='orange', width=1),
        name='EMA 50'
    ), row=1, col=1)

# 绘制 SMA 200
if 'SMA_200' in df.columns:
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['SMA_200'],
        line=dict(width=1),
        name='SMA 200'
    ), row=1, col=1)

# 金叉/死叉标记：SMA20 与 SMA200 交叉
if 'SMA_20' in df.columns and 'SMA_200' in df.columns:
    sign = np.sign(df["SMA_20"] - df["SMA_200"])
    cross = sign.diff().fillna(0).ne(0) & df["SMA_20"].notna() & df["SMA_200"].notna()
    golden = cross & (sign > 0)
    dead = cross & (sign < 0)

    # 金叉
    fig.add_trace(go.Scatter(
        x=df.loc[golden, "time"], y=df.loc[golden, "SMA_20"],
        mode="markers", name="Golden Cross",
        marker_symbol="triangle-up", marker_size=9
    ), row=1, col=1)

    # 死叉
    fig.add_trace(go.Scatter(
        x=df.loc[dead, "time"], y=df.loc[dead, "SMA_20"],
        mode="markers", name="Dead Cross",
        marker_symbol="triangle-down", marker_size=9
    ), row=1, col=1)

# 第二个子图：RSI 和 MACD
rsi_exists = 'RSI' in df.columns
macd_exists = 'MACD' in df.columns and 'MACD_signal' in df.columns and 'MACD_hist' in df.columns

if rsi_exists:
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['RSI'],
        line=dict(color='purple', width=1),
        name='RSI'
    ), row=2, col=1, secondary_y=False)

if macd_exists:
    # MACD 柱状图
    fig.add_trace(go.Bar(
        x=df['time'], y=df['MACD_hist'],
        marker_color='grey',
        name='MACD Hist'
    ), row=2, col=1, secondary_y=True)
    # MACD 线
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['MACD'],
        line=dict(color='blue', width=1),
        name='MACD'
    ), row=2, col=1, secondary_y=True)
    # MACD 信号线
    fig.add_trace(go.Scatter(
        x=df['time'], y=df['MACD_signal'],
        line=dict(color='orange', width=1, dash='dot'),
        name='MACD Signal'
    ), row=2, col=1, secondary_y=True)

# 布局设置
fig.update_layout(
    title="EUR/USD with Technical Indicators",
    xaxis_title="Time",
    yaxis_title="Price",
    xaxis_rangeslider_visible=False,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

# RSI y轴范围限制
if rsi_exists:
    fig.update_yaxes(title_text="RSI", row=2, col=1, secondary_y=False, range=[0, 100])

# MACD y轴标题
if macd_exists:
    fig.update_yaxes(title_text="MACD", row=2, col=1, secondary_y=True)

# 保存图表
fig.write_html(OUTPUT_PATH)
print(f"✅ 图表已保存至 {OUTPUT_PATH}")

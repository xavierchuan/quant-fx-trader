import math
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd
from loguru import logger

# 允许从项目根目录导入模块
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.append(PROJECT_ROOT)
sys.path.append(REPO_ROOT)

from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
from shared.utils.config import OANDA_TOKEN  # 确保这里能拿到 token

# 日志
logger.remove()
logger.add(sys.stderr, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}")

def normalize_to_oanda(symbol: str) -> str:
    """
    将用户友好的交易对格式转换为OANDA格式，例如：
    'eurusd', 'EUR-USD', 'eur_usd' -> 'EUR_USD'
    """
    s = symbol.upper().replace("-", "_").replace(" ", "").replace("/", "_")
    # 如果已经是正确格式，直接返回
    if "_" in s and len(s) == 7:
        return s
    # 尝试拆分为两部分
    if len(s) == 6:
        return s[:3] + "_" + s[3:]
    return s

def default_out_csv(project_root: str, instrument: str, granularity: str) -> str:
    """
    根据交易对和时间粒度生成默认输出路径，如：
    data/raw/EURUSD_H1.csv
    """
    fname = f"{instrument.replace('_','')}_{granularity}.csv"
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    return os.path.join(raw_dir, fname)

def _bars_per_day(granularity: str) -> float:
    """
    Rough estimate of bars per day for常见 OANDA 粒度。
    用于根据 count 推算需要拉取的天数。
    """
    granularity = granularity.upper()
    seconds_map = {
        "S5": 5,
        "S10": 10,
        "S15": 15,
        "S30": 30,
        "M1": 60,
        "M2": 120,
        "M4": 240,
        "M5": 300,
        "M10": 600,
        "M15": 900,
        "M30": 1800,
        "H1": 3600,
        "H2": 7200,
        "H3": 10800,
        "H4": 14400,
        "H6": 21600,
        "H8": 28800,
        "H12": 43200,
        "D": 86400,
        "W": 86400 * 5,
        "M": 86400 * 21,
    }
    seconds = seconds_map.get(granularity, 3600)
    if seconds <= 0:
        return 24
    return max(86400 / seconds, 1)


def get_candles(symbol="EUR_USD", granularity="H1", start_days_ago=365, target_count: int | None = None) -> pd.DataFrame:
    """
    循环抓取 OANDA 历史K线（默认过去一年），自动分页拼接。
    """
    if not OANDA_TOKEN:
        raise RuntimeError("OANDA_TOKEN 为空，请在 utils/config.py 配置或通过环境变量提供。")

    client = API(access_token=OANDA_TOKEN)

    end = datetime.now(timezone.utc)
    start = end - timedelta(days=start_days_ago)
    cur = start
    all_rows = 0
    parts = []

    logger.info(f"开始下载 {symbol} {granularity}（过去 {start_days_ago} 天）")

    # 每次抓 20 天（H1 ≈ 480 根），避免单次数据过大
    step = timedelta(days=20)

    while cur < end:
        to_ts = min(cur + step, end)
        params = {
            "granularity": granularity,
            "price": "M",
            "from": cur.isoformat(),
            "to": to_ts.isoformat(),
        }
        r = instruments.InstrumentsCandles(instrument=symbol, params=params)
        try:
            client.request(r)
        except Exception as e:
            logger.error(f"请求失败 {cur} ~ {to_ts}: {e}")
            break

        candles = r.response.get("candles", [])
        if not candles:
            logger.warning(f"区间无数据：{cur} ~ {to_ts}")
            cur = to_ts
            continue

        data = [{
            "time": c["time"],
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low":  float(c["mid"]["l"]),
            "close":float(c["mid"]["c"]),
            "volume": c["volume"]
        } for c in candles if c.get("complete")]

        if data:
            df_part = pd.DataFrame(data)
            parts.append(df_part)
            all_rows += len(df_part)
            logger.info(f"抓取区间 {cur:%Y-%m-%d} ~ {to_ts:%Y-%m-%d} 行数={len(df_part)}，累计={all_rows}")

        cur = to_ts  # 推进窗口

        if target_count and all_rows >= target_count:
            logger.info(f"已满足目标条数 {target_count}，停止抓取。")
            break

    if not parts:
        logger.warning("没有获取到任何数据。")
        return pd.DataFrame()

    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").drop_duplicates(subset=["time"]).reset_index(drop=True)
    if target_count:
        df = df.tail(target_count).reset_index(drop=True)

    logger.info(f"✅ 下载完成：总计 {len(df)} 行（{df['time'].min()} ~ {df['time'].max()}）")
    return df

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="EUR_USD")
    parser.add_argument("--granularity", default="H1")
    parser.add_argument("--days", type=int, default=365, help="向前回溯天数")
    parser.add_argument("--count", type=int, default=None, help="（可选）需要的 K 线数量，脚本会根据粒度估算天数，抓够后截断")
    parser.add_argument("--out", "--output", dest="out", default=None)
    args = parser.parse_args()

    # 规范化交易对格式
    args.symbol = normalize_to_oanda(args.symbol)

    # 自动生成输出路径（如果未指定）
    if args.out is None:
        args.out = default_out_csv(PROJECT_ROOT, args.symbol, args.granularity)

    # 确保输出目录存在
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    target_count = args.count if args.count and args.count > 0 else None
    if target_count:
        est_days = math.ceil(target_count / _bars_per_day(args.granularity)) + 5
        if est_days > args.days:
            logger.info(f"根据 count={target_count} 估算需要 {est_days} 天数据（原 days={args.days}），已自动扩展。")
            args.days = est_days

    try:
        df = get_candles(args.symbol, args.granularity, args.days, target_count=target_count)
    except Exception as e:
        logger.exception(f"下载失败：{e}")
        sys.exit(1)

    if df.empty:
        logger.warning("结果为空，未保存。")
        sys.exit(2)

    df.to_csv(args.out, index=False)
    logger.info(f"📦 已保存到：{args.out}")
    # 方便你肉眼确认
    logger.info(f"尾部预览：\n{df.tail(3).to_string(index=False)}")

if __name__ == "__main__":
    main()

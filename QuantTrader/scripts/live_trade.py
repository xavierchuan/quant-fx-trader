"""
简易实盘脚本示例（使用 OANDA）
将实时数据传入策略，策略产生信号 -> 风险检查 -> 下单执行

注意：在真实交易前请先在沙盒/模拟账户充分测试。
"""
import asyncio
import os
from dotenv import load_dotenv
from loguru import logger

from core.data.oanda import OANDADataFeed
from core.execution.oanda_handler import OANDAExecutionHandler
from core.strategy.rsi_mean_reversion import RSIMeanReversionStrategy
from core.risk.base import SimpleRiskManager
from core.data.base import MarketDataEvent

load_dotenv()

async def main():
    account_id = os.getenv('OANDA_ACCOUNT_ID')
    token = os.getenv('OANDA_TOKEN')
    env = os.getenv('OANDA_ENVIRONMENT', 'practice')
    instrument = 'EUR_USD'

    data_feed = OANDADataFeed(
        instrument=instrument,
        timeframe='H1',
        account_id=account_id,
        access_token=token,
        environment=env
    )

    strategy = RSIMeanReversionStrategy(instrument=instrument)
    risk_manager = SimpleRiskManager(max_position_size=1.0, max_portfolio_risk=0.02, max_drawdown=0.1)
    execution = OANDAExecutionHandler(account_id=account_id, access_token=token, environment=env)

    async def on_market(event: MarketDataEvent):
        signal = await strategy.on_data(event)
        if not signal:
            return
        if await risk_manager.check_signal(signal):
            await execution.process_signal(signal)

    # 订阅实时数据并运行
    await data_feed.subscribe(on_market)
    logger.info('已订阅实时数据，开始监听（按 Ctrl+C 退出）')

    try:
        while True:
            await asyncio.sleep(1)
    except asyncio.CancelledError:
        pass
    finally:
        await data_feed.unsubscribe()

if __name__ == '__main__':
    asyncio.run(main())

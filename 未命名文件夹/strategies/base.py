# strategies/base.py
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from typing import Dict, Any

class Strategy:
    """
    只负责“发信号”，不做撮合和资金结算。
    on_bar 输入 state，输出 {"action": "ENTER_LONG"/"EXIT_LONG"/"HOLD"}。
    """
    def __init__(self, **params: Any) -> None:
        self.params = params

    def on_bar(self, state: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "HOLD"}
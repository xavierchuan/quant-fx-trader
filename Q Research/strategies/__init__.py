# strategies/__init__.py
from typing import Dict, Callable, Any

# 策略注册表：name -> class
_REGISTRY: Dict[str, Callable[..., Any]] = {}

def register(name: str):
    """用作装饰器：@register('sma_atr')"""
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco

def _lazy_import_all():
    """
    懒加载：首次 load_strategy 时再导入具体策略文件，
    这样不会因为循环依赖或路径问题导致注册表是空的。
    """
    # 在这里逐个导入具体策略模块；导入发生时模块内的 @register 会把类放进 _REGISTRY
    from . import sma_atr  # noqa: F401
    from . import regime_sma  # noqa: F401
    from . import band_mean_revert  # noqa: F401
    from . import bollinger_mean_revert  # noqa: F401
    from . import ma_crossover  # noqa: F401
    from . import momentum  # noqa: F401
    from . import xgb_signal  # noqa: F401

def load_strategy(name: str, **kwargs):
    # 第一次用时尝试懒加载，填充注册表
    if not _REGISTRY:
        _lazy_import_all()
    if name not in _REGISTRY:
        # 再尝试一次（防止用户后来才添加文件）
        _lazy_import_all()
    if name not in _REGISTRY:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(_REGISTRY.keys())}")
    return _REGISTRY[name](**kwargs)

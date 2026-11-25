"""Which optional extras this test run has: toolsets mounted by installation alone change exact tool lists."""
import importlib.util

#: market_prices is mounted whenever the [market] extra is installed (it needs no setting)
MARKET_INSTALLED = importlib.util.find_spec("market_data") is not None
MARKET_TOOLS = {"market_prices"} if MARKET_INSTALLED else set()
MARKET_STATUS = "enabled" if MARKET_INSTALLED else "not installed"

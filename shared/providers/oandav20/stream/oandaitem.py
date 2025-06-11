from abc import ABC
from datetime import datetime

from mpmath import mp

_currency_multipliers = {'HUF': 149.2537313432836, 'JPY': 63.25515280739161, 'THB': 14.214641080312722,
                         'TRY': 14.214641080312722, 'CZK': 9.23951670220327, 'MXN': 8.173418621179815,
                         'ZAR': 7.107320540156361, 'SEK': 4.264392324093817, 'CNH': 2.8429282160625444,
                         'HKD': 3.1982942430703623, 'NOK': 4.264392324093817, 'DKK': 2.8429282160625444,
                         'PLN': 1.5991471215351811, 'NZD': 0.6751954513148543, 'AUD': 0.6041222459132907,
                         'CAD': 0.5685856432125089, 'GBP': 0.31982942430703626, 'SGD': 0.5330490405117271,
                         'CHF': 0.3624733475479744, 'USD': 0.39445628997867804, 'EUR': 0.3766879886282872}


def _get_instrument_multiplier(instrument: str):
    return mp.fdiv(_currency_multipliers.get(instrument[:3], 1), _currency_multipliers.get(instrument[-3:], 1))


def _str_to_datetime(timestamp):
    """Convert a timestamp string to a datetime object."""
    timestamp_truncated = timestamp[:26] + 'Z'  # Keep only up to 6 digits for microseconds
    return datetime.strptime(timestamp_truncated, "%Y-%m-%dT%H:%M:%S.%fZ")


def _compute_weighted_geometric_mid(prices):
    logarithmic_linear_combination = mp.zero
    total_liquidity = mp.zero
    for p in prices:
        liquidity = mp.mpf(p['liquidity'])
        price = p['price']
        logarithmic_linear_combination += liquidity * mp.ln(price)
        total_liquidity += liquidity
    weighted_arithmetic_mean_of_logarithms = mp.fdiv(logarithmic_linear_combination, total_liquidity)
    weighted_geometric_mean = mp.exp(weighted_arithmetic_mean_of_logarithms)
    return weighted_geometric_mean


class OandaItem(ABC):
    def __init__(self, time: datetime):
        self.time = time


class Heartbeat(OandaItem):
    def __init__(self, time: str):
        super().__init__(_str_to_datetime(time))


class Tick(OandaItem):
    def __init__(self, instrument: str, time: str, bids: list[dict], asks: list[dict]):
        super().__init__(_str_to_datetime(time))
        self.instrument = instrument
        self.bids = bids
        self.asks = asks
        self.mid = _compute_weighted_geometric_mid(asks + bids)
        self.multiplier = _get_instrument_multiplier(instrument)
        self.scaled_mid = self.multiplier * self.mid
        self.bid = max([float(bid["price"]) for bid in bids])
        self.ask = min([float(ask["price"]) for ask in asks])


def transform_tick(tick: dict) -> OandaItem:
    """
    Apply transformations to each tick here, such as extracting bid/ask data or applying custom formatting.

    :param tick: Raw tick data from the pricing stream.
    :return: Transformed tick data.
    """
    if tick['type'] != 'PRICE':
        return tick

    return Tick(instrument=tick['instrument'], time=tick['time'], bids=tick.get("bids", []), asks=tick.get("asks", []))
    # Example transformation: extract essential data or convert to a custom structure

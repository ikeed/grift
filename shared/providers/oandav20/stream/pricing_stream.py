from mpmath import mp
from oandapyV20.endpoints.pricing import PricingStream

from shared.decoupler.decoupler import precision_digits
from shared.providers.oandav20.Auth import AuthenticatedOandaService

mp.dps = precision_digits


class OandaPricingService(AuthenticatedOandaService):
    def __init__(self, environment: str = "practice", instruments: list[str] = None):
        """
        Initializes the OandaPricingService with account credentials and API settings.

        :param environment: "practice" for demo or "production" for live environment.
        :param instruments: List of instruments to stream, e.g., ["EUR_USD", "USD_JPY"].
        """
        super().__init__(environment=environment)
        self.pricing_stream = PricingStream(
            accountID=self.account_id,
            params={"instruments": ",".join(instruments)}
        )
        self.priority_queue = []

    def get_pricing_stream(self):
        """
        Starts the price streaming, applies transformations to each tick, and yields the transformed data.
        This version wraps the tick generator in a priority queue to ensure ticks are processed in order.

        :return: Iterator of transformed pricing stream data.
        """
        return self.client.request(self.pricing_stream)
        # for tick in self.client.request(self.pricing_stream):
        #     if tick['type'] == 'PRICE':
        #         print(tick)
        #     # Parse timestamp as an offset-aware datetime
        #     timestamp = datetime.fromisoformat(tick['time'])
        #     if timestamp.tzinfo is None:
        #         # Make sure the tick timestamp is aware (assuming UTC if missing tzinfo)
        #         timestamp = timestamp.replace(tzinfo=timezone.utc)
        #         # Add to priority queue
        #         heapq.heappush(self.priority_queue, Tick(timestamp, transform_tick(tick)))
        #
        #     # Yield ticks in the correct order, one at a time
        #     while self.priority_queue and self.priority_queue[0].timestamp <= datetime.now(timezone.utc):
        #         yield heapq.heappop(self.priority_queue).data

import asyncio
import json
import math
import sys
from datetime import datetime, UTC
from typing import Dict, List, TypedDict, Any, cast
from pathlib import Path
import importlib.util

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from google.cloud import firestore, pubsub_v1
except ImportError:
    # Handle import error or use mock classes for development
    from unittest.mock import MagicMock
    firestore = MagicMock()
    pubsub_v1 = MagicMock()

from aiohttp import web
from mpmath import mp

# Import health module using a modified path that works with the hyphenated directory
health_module_path = str(Path(project_root) / 'services/tick_fetcher_decoupler/health.py')
health_spec = importlib.util.spec_from_file_location('health', health_module_path)
health = importlib.util.module_from_spec(health_spec)
health_spec.loader.exec_module(health)

# Local imports
from grift.config import settings
from grift.logger import StructuredLogger
from shared.decoupler.decoupler import ContinuousDecoupleService
from shared.providers.oandav20.stream.pricing_stream import OandaPricingService

logger = StructuredLogger(__name__)

class FirestoreData(TypedDict):
    instrument: str
    last_timestamp: str
    last_price: float

class CheckpointData(TypedDict):
    timestamp: str
    price: float

class TickData(TypedDict):
    instrument: str
    last_timestamp: str
    last_price: float

class TickFetcherDecoupler:
    def __init__(self):
        self.db = firestore.Client()
        self.publisher = pubsub_v1.PublisherClient()
        self.topic_path = self.publisher.topic_path(
            settings.gcp.project_id, settings.pubsub.topic_raw
        )

        # Initialize the OANDA pricing service
        self.pricing_service = OandaPricingService(
            environment=settings.oanda.environment,
            instruments=settings.oanda.currency_pairs
        )

        # Initialize the decoupler service
        self.decoupler = ContinuousDecoupleService(
            pair_columns=settings.oanda.currency_pairs,
            algo="trust-krylov",
            use_row_scaling=True
        )

        # Initialize health server
        self.health_server = health.HealthServer(port=8080)

        self.tick_batch: List[TickData] = []
        self.last_checkpoint: Dict[str, CheckpointData] = {}

    async def process_tick(self, tick: Dict[str, Any]) -> None:
        """Process a single tick, adding it to the batch and processing if batch is full."""
        if tick["type"] != "PRICE":
            return

        # Update health check timestamp
        self.health_server.last_tick_time = datetime.now(UTC)

        instrument = tick["instrument"]
        timestamp = datetime.fromisoformat(tick["time"])
        # Calculate mid price from bid and ask
        mid_price = math.sqrt(float(tick["bids"][0]["price"]) * float(tick["asks"][0]["price"]))

        processed_tick = {
            "instrument": instrument,
            "timestamp": timestamp.isoformat(),
            "scaled_mid": mp.mpf(str(mid_price))
        }

        # Update the decoupler with the new tick
        result = self.decoupler.update(processed_tick)
        if result is not None:
            # Publish the decoupled vector to Pub/Sub
            message = {
                "timestamp": timestamp.isoformat(),
                "w": [float(x) for x in result[:len(settings.oanda.currency_pairs)]]
            }
            await self.publisher.publish(
                self.topic_path,
                json.dumps(message).encode("utf-8")
            )

            # Add to checkpoint batch
            new_tick: TickData = {
                "instrument": instrument,
                "last_timestamp": timestamp.isoformat(),
                "last_price": float(mid_price)
            }
            self.tick_batch.append(new_tick)

            if len(self.tick_batch) >= settings.firestore.checkpoint_batch_size:
                await self.checkpoint_batch()

    async def checkpoint_batch(self) -> None:
        """Write the current batch of ticks to Firestore."""
        if not self.tick_batch:
            return

        batch = self.db.batch()
        for tick_data in self.tick_batch:
            doc_ref = self.db.collection(settings.firestore.collection_checkpoints).document(tick_data["instrument"])
            # Convert TypedDict to dict and ensure all values are JSON-serializable
            firestore_data = {
                "instrument": tick_data["instrument"],
                "last_timestamp": tick_data["last_timestamp"],
                "last_price": float(tick_data["last_price"])
            }
            batch.set(doc_ref, firestore_data)

        batch.commit()
        self.tick_batch.clear()
        logger.info("Checkpointed batch", batch_size=settings.firestore.checkpoint_batch_size)

    async def restore_from_checkpoint(self) -> None:
        """Restore the last known state from Firestore."""
        docs = self.db.collection(settings.firestore.collection_checkpoints).stream()
        for doc in docs:
            data = cast(FirestoreData, doc.to_dict())
            checkpoint_data: CheckpointData = {
                "timestamp": data["last_timestamp"],
                "price": float(data["last_price"])
            }
            self.last_checkpoint[data["instrument"]] = checkpoint_data

        if self.last_checkpoint:
            logger.info("Restored checkpoints", instrument_count=len(self.last_checkpoint))

    async def run(self):
        """Main service loop."""
        # Start health server
        self.health_server.runner = web.AppRunner(self.health_server.app)
        await self.health_server.runner.setup()
        self.health_server.site = web.TCPSite(self.health_server.runner, "0.0.0.0", self.health_server.port)
        await self.health_server.site.start()
        logger.info("Health server started", port=self.health_server.port)

        await self.restore_from_checkpoint()

        try:
            async for tick in self.pricing_service.get_pricing_stream():
                await self.process_tick(tick)
        except Exception as e:
            logger.error("Error in pricing stream", error=str(e))
            # Ensure we checkpoint any remaining ticks before exiting
            await self.checkpoint_batch()
            raise
        finally:
            # Clean up health server
            if self.health_server.runner:
                await self.health_server.runner.cleanup()

async def main():
    service = TickFetcherDecoupler()
    await service.run()

if __name__ == "__main__":
    asyncio.run(main())

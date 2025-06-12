from aiohttp import web
import asyncio
from typing import Optional
from datetime import datetime, UTC

from grift.logger import StructuredLogger

logger = StructuredLogger(__name__)

class HealthServer:
    def __init__(self, port: int = 8080):
        self.port = port
        self.app = web.Application()
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.last_tick_time: Optional[datetime] = None

        # Set up routes
        self.app.router.add_get("/healthz", self.healthz)
        self.app.router.add_get("/readyz", self.readyz)
        self.app.router.add_get("/metrics", self.metrics)

    async def healthz(self, request: web.Request) -> web.Response:
        """Basic health check that always returns 200 if the server is running."""
        return web.Response(text="OK")

    async def readyz(self, request: web.Request) -> web.Response:
        """Readiness check that verifies we're receiving ticks within acceptable delay."""
        if not self.last_tick_time:
            return web.Response(text="No ticks received yet", status=503)

        now = datetime.now(UTC)
        delay = (now - self.last_tick_time).total_seconds()

        # If we haven't received a tick in more than 30 seconds, we're not ready
        if delay > 30:
            return web.Response(
                text=f"Last tick was {delay:.1f} seconds ago",
                status=503
            )
        return web.Response(text="OK")

    async def metrics(self, request: web.Request) -> web.Response:
        """Prometheus metrics endpoint."""
        now = datetime.now(UTC)
        metrics = []

        # Basic up metric
        metrics.append("tick_fetcher_up 1")

        # Last tick age in seconds
        if self.last_tick_time:
            delay = (now - self.last_tick_time).total_seconds()
            metrics.append(f"tick_fetcher_last_tick_age_seconds {delay:.1f}")

        return web.Response(text="\n".join(metrics) + "\n", content_type="text/plain")

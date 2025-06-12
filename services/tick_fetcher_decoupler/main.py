"""Main entry point for the Tick Fetcher Decoupler service."""
import sys
import logging
import os
from shared.providers.oandav20.stream.pricing_stream import OandaPricingService
from shared.decoupler.decoupler import ContinuousDecoupleService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Main entry point."""
    logger.info("Starting Tick Fetcher Decoupler service...")

    # Set up remote debugging if DEBUG_WAIT is true
    if os.getenv('DEBUG_WAIT', 'false').lower() == 'true':
        logger.info("Waiting for debugger to attach...")
        import debugpy
        debugpy.listen(('0.0.0.0', 5680))
        debugpy.wait_for_client()
        logger.info("Debugger attached!")

    try:
        stream = OandaPricingService().pricing_stream
        decoupler = ContinuousDecoupleService()
        for tick in stream.get_pricing_stream():
            # Process each tick as needed
            logger.info(f"Received tick: {tick}")
            decoupler.update(tick)
    except Exception as e:
        logger.error(f"Error in Tick Fetcher Decoupler service: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

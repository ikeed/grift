import os

import oandapyV20
from dotenv import load_dotenv


class AuthenticatedOandaService:
    def __init__(self, environment: str = "practice"):
        """
        Base service that handles authentication and environment setup for OANDA API services.

        :param environment: "practice" for demo or "production" for live environment.
        """
        # Load environment-specific settings
        load_dotenv(f".env.oanda{environment}")

        self.account_id = os.getenv("ACCOUNT_ID")
        print(f"Got account_id = {self.account_id}")
        access_token = os.getenv("ACCESS_TOKEN")

        # Map custom environment input to expected values by oandapyV20
        oanda_env = "practice" if environment == "practice" else "live"

        # Initialize the API client with the correct environment
        self.client = oandapyV20.API(access_token=access_token, environment=oanda_env)

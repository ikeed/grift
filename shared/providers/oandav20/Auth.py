import os
from google.cloud import secretmanager
import oandapyV20
from grift.config import settings
from grift.logger import StructuredLogger

logger = StructuredLogger(__name__)


class AuthenticatedOandaService:
    def __init__(self, environment: str = "practice"):
        """
        Base service that handles authentication and environment setup for OANDA API services.

        :param environment: "practice" for demo or "production" for live environment.
        """
        self.environment = environment
        secrets_client = secretmanager.SecretManagerServiceClient()

        # Define secret names based on environment
        env_prefix = f"oanda-{environment}"
        account_id_secret = f"{env_prefix}-account-id"
        token_secret = f"{env_prefix}-api-token"

        try:
            # Access secrets
            self.account_id = self._access_secret(secrets_client, account_id_secret)
            access_token = self._access_secret(secrets_client, token_secret)
            logger.info(f"Successfully retrieved OANDA credentials for {environment} environment")

            # Initialize OANDA client
            oanda_env = "practice" if environment == "practice" else "live"
            self.client = oandapyV20.API(access_token=access_token, environment=oanda_env)

        except Exception as e:
            logger.error(f"Failed to initialize OANDA service", error=str(e), environment=environment)
            raise

    def _access_secret(self, client, secret_id: str) -> str:
        """Access the latest version of a secret from Secret Manager."""
        name = f"projects/{settings.gcp.project_id}/secrets/{secret_id}/versions/latest"
        try:
            response = client.access_secret_version(request={"name": name})
            return response.payload.data.decode("UTF-8")
        except Exception as e:
            logger.error(f"Failed to access secret", secret_id=secret_id, error=str(e))
            raise

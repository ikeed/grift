import logging

from oandapyV20.endpoints.accounts import AccountInstruments

from shared.providers.oandav20.Auth import AuthenticatedOandaService

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class OandaInstrumentService(AuthenticatedOandaService):
    def __init__(self, environment: str = "practice"):
        """
        Initializes the OandaInstrumentService with account credentials and API settings.

        :param environment: "practice" for demo or "production" for live environment.
        """
        super().__init__(environment)

    def list_instruments(self):
        """
        Fetches and returns the list of instruments available in the account.

        :return: List of instrument names.
        """
        r = AccountInstruments(accountID=self.account_id)
        response = self.client.request(r)

        # Extract and return the instrument names
        forex_instruments = [instrument for instrument in response.get("instruments", []) if
                             instrument.get("type") == "CURRENCY"]
        # print(json.dumps(forex_instruments, indent=2))
        return forex_instruments

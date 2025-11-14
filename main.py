# main.py
import oandapyV20
import oandapyV20.endpoints.accounts as accounts
from shared.utils.config import OANDA_TOKEN, OANDA_ACCOUNT_ID, OANDA_URL
from shared.utils.logger import logger

def connect_oanda():
    client = oandapyV20.API(access_token=OANDA_TOKEN)
    return client

def get_account_summary(client):
    r = accounts.AccountSummary(accountID=OANDA_ACCOUNT_ID)
    client.request(r)
    return r.response

if __name__ == "__main__":
    client = connect_oanda()
    logger.info("✅ Connected to OANDA")

    summary = get_account_summary(client)
    balance = summary['account']['balance']
    currency = summary['account']['currency']
    logger.info(f"💰 Balance: {balance} {currency}")

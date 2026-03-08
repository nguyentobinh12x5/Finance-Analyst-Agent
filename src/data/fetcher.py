import requests
from loguru import logger
from src.config.settings import settings

class DataFetcher:
    def __init__(self):
        self.base_url = "https://financialmodelingprep.com/stable"

    def fetch_sp500_tickers(self):
        url = f"{self.base_url}/sp500_constituent?api_key={settings.FMP_API_KEY}"
        try: 
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            tickers = [item['symbol'] for item in data]
            return tickers
        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching SP500 tickers: {e}")
            return []   
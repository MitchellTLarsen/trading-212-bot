import requests
import time
from config import BASE_URL, HEADERS


class Trading212Client:
    """Client for the Trading 212 practice account API."""

    def __init__(self):
        self.base_url = BASE_URL
        self.headers = HEADERS
        self._last_request_time = 0
        self._min_interval = 0.2  # 5 requests/sec rate limit

    def _throttle(self):
        elapsed = time.time() - self._last_request_time
        if elapsed < self._min_interval:
            time.sleep(self._min_interval - elapsed)
        self._last_request_time = time.time()

    def _get(self, endpoint, params=None):
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        resp = requests.get(url, headers=self.headers, params=params)
        resp.raise_for_status()
        return resp.json()

    def _post(self, endpoint, data=None):
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        resp = requests.post(url, headers=self.headers, json=data)
        resp.raise_for_status()
        return resp.json()

    def _delete(self, endpoint):
        self._throttle()
        url = f"{self.base_url}{endpoint}"
        resp = requests.delete(url, headers=self.headers)
        resp.raise_for_status()
        return resp.json() if resp.content else None

    # --- Account ---

    def get_account_cash(self):
        return self._get("/equity/account/cash")

    def get_account_metadata(self):
        return self._get("/equity/account/info")

    # --- Portfolio ---

    def get_portfolio(self):
        return self._get("/equity/portfolio")

    def get_position(self, ticker):
        return self._get(f"/equity/portfolio/{ticker}")

    # --- Orders ---

    def place_market_order(self, ticker, quantity):
        return self._post("/equity/orders/market", {
            "ticker": ticker,
            "quantity": quantity,
        })

    def place_limit_order(self, ticker, quantity, limit_price, time_validity="Day"):
        return self._post("/equity/orders/limit", {
            "ticker": ticker,
            "quantity": quantity,
            "limitPrice": limit_price,
            "timeValidity": time_validity,
        })

    def place_stop_order(self, ticker, quantity, stop_price, time_validity="Day"):
        return self._post("/equity/orders/stop", {
            "ticker": ticker,
            "quantity": quantity,
            "stopPrice": stop_price,
            "timeValidity": time_validity,
        })

    def get_orders(self):
        return self._get("/equity/orders")

    def cancel_order(self, order_id):
        return self._delete(f"/equity/orders/{order_id}")

    # --- Instruments ---

    def get_instruments(self):
        return self._get("/equity/metadata/instruments")

    def search_instruments(self, query):
        instruments = self.get_instruments()
        query_lower = query.lower()
        return [i for i in instruments if query_lower in i.get("ticker", "").lower()
                or query_lower in i.get("name", "").lower()]

    # --- Historical data ---

    def get_historical_orders(self, cursor=None, limit=50):
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._get("/equity/history/orders", params=params)

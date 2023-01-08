import alpaca_trade_api as tradeapi

class tradingAPI:

    def __init__(self):

        # Set the API key and secret
        self.API_KEY = "YOUR_API_KEY"
        self.API_SECRET = "YOUR_API_SECRET"

        # Create an API client
        self.api = api = tradeapi.REST(API_KEY, API_SECRET, api_version="v2")

    def buyStock(self, symbol, qty, mtype, time_in_force):
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="buy",
            type= mtype,
            time_in_force= time_in_force
        )
        print(f"Purchased {symbol} at ${self.getStockPrice(symbol)}")

    def sellStock(self, symbol, qty, mtype, time_in_force):
        api.submit_order(
            symbol=symbol,
            qty=qty,
            side="sell",
            type= mtype,
            time_in_force= time_in_force
        )
        print(f"Sold {symbol} at ${self.getStockPrice(symbol)}")


    def getStockPrice(self, symbol):
        return api.polygon.last_trade(symbol).price
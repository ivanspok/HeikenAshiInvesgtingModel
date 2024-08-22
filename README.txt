# Current setup of the model:

# File to get historical data: 
yfinance_api.py 

# File for experimenting buy condition:
main.ipynb

# file for test trading 
test_trading.py


# Buy order ONLY in normal trading_hours
# Sell limit if tourched orders any time! (fill_outside_rth = True)

# Buy-sell methology
1. Buy if buy condition is met.
2. Place limit and stop orders.  Record they orders id.
3. If order executed based on historical orders -> Update DB.


# if buy condition is true place LIT buy order with:

 current_price (from df_stock_1m) 
 buy_trigger_price (from df_stock)

 # price should be higher than aux_price!
 price = buy_trigger_price * 1.0001
 aux_price (trigger price ) = price 

 when buy: price higher that trigger price
 when sell: price lower than trigger price
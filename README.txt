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


 # Trailing moomoo orders conception

 If current_gain more than 0.4% place Sell Trailing Stop Lmt Order with trailing_ratio = 0.15%, spread = 0.03%
 Cancel Limit-if-touched sell order
 
 Modify triling ration of Sell Trailing Stop Lmt Order based on current gain according to this table:
 
Current gain	trailing ratio	stop_price	spread	      Sell Price
0.40%	        0.00%     	    0.50%	       0.03%      	NO TRAILING ORDER SET
0.50%	        0.00%     	    0.50%	       0.03%      	NO TRAILING ORDER SET
0.5500%	      0.05%     	    0.50%	       0.03%      	0.47%
0.6000%	      0.10%     	    0.50%	       0.03%      	0.47%
0.6500%	      0.10%     	    0.55%	       0.03%      	0.52%
0.7000%	      0.10%     	    0.60%	       0.03%      	0.57%
0.7500%	      0.10%     	    0.65%	       0.03%      	0.62%
0.8000%	      0.15%     	    0.65%	       0.03%      	0.62%
0.8500%	      0.15%     	    0.70%	       0.03%      	0.67%
0.9000%	      0.20%     	    0.70%	       0.03%      	0.67%
0.9500%	      0.20%     	    0.75%	       0.03%      	0.72%
1.0000%	      0.20%     	    0.80%	       0.03%      	0.77%

Current gain	trailing ratio	stop_price	spread	      Sell Price           Action\Status
0.40%	        0.15%     	    0.50%	       0.03%      	0.25%            SET TRAILING ORDER, trailing LIT was set at the begining 0.6%
0.50%	        0.15%     	    0.50%	       0.03%      	0.35%            TRAILING ORDER SET, increase traling LIT gain to 0.65%
0.5500%	      0.15%     	    0.50%	       0.03%      	0.47%            TRAILING ORDER SET, increase traling LIT gain to 0.75%
0.6000%	      0.15%     	    0.50%	       0.03%      	0.47%            TRAILING ORDER SET, Cancel trailing LIT
0.6500%	      0.15%     	    0.55%	       0.03%      	0.52%            TRAILING ORDER SET
0.7000%	      0.15%     	    0.60%	       0.03%      	0.57%            TRAILING ORDER SET
0.7500%	      0.15%     	    0.65%	       0.03%      	0.62%            ...
0.8000%	      0.15%     	    0.65%	       0.03%      	0.62%
0.8500%	      0.15%     	    0.70%	       0.03%      	0.67%
0.9000%	      0.20%     	    0.70%	       0.03%      	0.67%
0.9500%	      0.15%     	    0.75%	       0.03%      	0.72%
1.0000%	      0.20%     	    0.80%	       0.03%      	0.77%

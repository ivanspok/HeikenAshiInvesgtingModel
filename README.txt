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
0.50%	        0.15%     	    0.50%	       0.03%      	0.35%            TRAILING ORDER SET, increase trailing LIT gain to 0.65%
0.5500%	      0.15%     	    0.50%	       0.03%      	0.47%            TRAILING ORDER SET, increase trailing LIT gain to 0.75%
0.6000%	      0.15%     	    0.50%	       0.03%      	0.47%            TRAILING ORDER SET, Cancel trailing LIT
0.6500%	      0.15%     	    0.55%	       0.03%      	0.52%            TRAILING ORDER SET
0.7000%	      0.15%     	    0.60%	       0.03%      	0.57%            TRAILING ORDER SET
0.7500%	      0.15%     	    0.65%	       0.03%      	0.62%            ...
0.8000%	      0.15%     	    0.65%	       0.03%      	0.62%
0.8500%	      0.15%     	    0.70%	       0.03%      	0.67%
0.9000%	      0.20%     	    0.70%	       0.03%      	0.67%
0.9500%	      0.15%     	    0.75%	       0.03%      	0.72%
1.0000%	      0.20%     	    0.80%	       0.03%      	0.77%


если 13:30 свеча зеланая
и 13:30 more than min(open, close of each candle from 9:30 to it)    more than close of 9:30!!!!
sum pct more than 0 

if close 12:30 MORE (OPEN OR CLOSE 9:30) AND SUM psc more than 0
candle is green 
9:30 pct less than -1% !!!

I need 12_30 condition. 
If it trigger place limit if touch order with df[close][-2] price !!!

If time 15:48 place limit order with df_1m.close[-1]


@@@
Time	Condition		
9:30	9_30		check it
9:30 - 9:47	930_47		check it
9:47 - 10:45	drowdown930	drowdown	
10:40- 11:05	DO NOT BUY		
11:05:00 - 12:00	c2	c32	drowdown
12:00  - 13:30	c42	drowdown	?
			
			
13:30 - 15:00	buy condition 13:30		
			
3:35:00 PM - 15:50	Rising before closing the market?!!!		check assumption




gain_coef !!!!  +

buy order shoud be outside market hours, good_till_cancel ++

* verify cancelation condition (shoud cancel just only when market would open) ++


during outsite hours price shoud requtest with prepost data

______________________________
during outsite hours price shoud requtest with prepost data +


* place sell orders 

if gain more than 2% limit sell order at 0.5%
if gain more than 3% limit sell order at 1%
if gain more than 4% limit sell order at 2%
if gain more than 4.5% limit sell order at 2.5%


If more than 3% fast sold it


changed in count cycle also !!!
Chainsaw Man Episode 

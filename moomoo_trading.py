import yfinance as yf
# import matplotlib

from datetime import datetime, timedelta, tzinfo
import pandas as pd
pd.options.mode.chained_assignment = None 

from personal_settings import personal_settings as ps
from datetime import date, timedelta, timezone
import os, pathlib
import numpy as np
import pickle
import functions_2 as f2
from tradingInterface import TradeInterface
from tqdm import tqdm
import time
import moomoo as ft
from moomoo_api import Moomoo_API
import global_variables as gv
import winsound
import math

import warnings
warnings.filterwarnings('error')
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

from colog.colog import colog
c = colog()
warning = colog(TextColor='orange')
alarm = colog(TextColor='red')

from currency_converter import CurrencyConverter
cr = CurrencyConverter()
rate = cr.convert(1, 'AUD', 'USD')

# Trade settings 
money_permitted_for_trade = 2200 * rate # in AUD * rate = USD
default_buy_sum = 1000 * rate # in AUD * rate = USD
min_buy_sum = 650 * rate # in AUD * rate = USD
max_buy_sum = 1400 * rate # in AUD * rate = USD
stop_trading_profit_value = -30 * rate # in AUD * rate = USD
max_stock_price = 1000 * rate # in AUD * rate = USD

# Moomoo settings
moomoo_ps = ps.Moomoo()
ip = '127.0.0.1'
port = 11111
unlock_pwd = moomoo_ps.unlock_pwd
ACC_ID = moomoo_ps.acc_id
TRD_ENV = ft.TrdEnv.REAL
MARKET = 'US.'
# ft.SysConfig.set_all_thread_daemon(True)

# Version 1.0
# stock_name_list_opt = [
#   'GOOG', 'JPM', 'XOM', 'UNH', 'AVGO', 'LLY', 'COST',
#   'CRM', 'TMO', 'NFLX', 'TXN', 'INTU', 'NKE', 'QCOM',
#   'BA', 'AMGN', 'MDT', 'PLD', 'MS', 'GS', 'LMT', 'ADI', 'TJX', 'ELV', 'C', 'CVS', 'VRTX', 'SCHW', 'LRCX',
#   'TMUS', 'ETN', 'ZTS', 'CI', 'FI', 'EQIX', 'DUK', 'MU',
#   'AON', 'ITW', 'SNPS', 'KLAC', 'CL', 'WM', 'HCA', 'MMM',
#   'CMG', 'EW', 'GM', 'MCK', 'NSC', 'PH', 'MPC', 'ROP', 
#   'MCHP', 'USB', 'CCI', 'MAR', 'MSI', 'GD', 'JCI', 'PSX', 
#   'SRE', 'ADSK', 'AJG', 'TEL', 'TT', 'PCAR', 'OXY', 'CARR',
#   'IDXX', 'GIS', 'CTAS', 'AIG', 'ANET', 'BIIB', 'SPG', 'MSCI', 'DHI'
# ]

# Version 2.0
stock_name_list_opt = [
'BA', 'ON', 'MCHP', 'ADI', 'PANW', 'DHI', 'ANET', 'AMD', 'LRCX', 'LLY',
'MU', 'TXN', 'AIG', 'WMB', 'BSX', 'NKE', 'OXY', 'TT', 'AMAT', 'ETN', 'DE',
'EL', 'FDX', 'MAR', 'GE', 'NFLX', 'NUE', 'GOOG', 'ECL', 'AVGO', 'CAT', 'SPG',
'ADSK', 'INTU', 'SLB', 'F', 'WMT', 'SBUX', 'SNPS', 'AJG', 'TMUS', 'KLAC', 'CI',
'JCI', 'GILD', 'QCOM', 'ROP', 'MO', 'WM', 'HON', 'ITW', 'GS', 'HCA', 'TJX', 'ICE',
'DXCM', 'IDXX', 'ABBV', 'CDNS', 'CMCSA', 'JNJ', 'EQIX', 'MDLZ', 'NXPI', 'MSI', 'TEL',
'LMT', 'USB', 'MRK', 'HLT', 'APD', 'CTAS', 'MNST', 'NOW', 'AMT', 'PH', 'HUM', 'ADM',
'TDG', 'EMR', 'GM', 'ADP', 'CMG', 'SCHW', 'MSCI', 'EOG', 'UNP', 'INTC', 'CME',
'MA', 'CVS', 'XOM', 'CSCO', 'WELL', 'TMO', 'MRNA', 'PLD', 'APH', 'PEP', 'CRM', 'MMM',
'MMC', 'LIN', 'GIS', 'COST', 'CSX', 'IQV', 'FI', 'MCD', 'VRTX'
]

# stock_name_list_opt = ['CME', 'AMT']

# settings for historical df from yfinance
period = '3mo'
interval = '1h' 


# settings for buy condition Version 1.0
# is_near_global_max_prt = 80
# distance_from_last_top  = 0
# last_top_ratio = 1
# RIV  = 0.25
# buy_ratio_border = 9
# bull_trend_coef = 1.12
# number_tries_to_submit_order = {}
#
# settings for buy condition Version 2.0
is_near_global_max_prt = 96
distance_from_last_top  = 0
last_top_ratio = 1
RIV  = 0.15
buy_ratio_border = 0
bull_trend_coef = 1.12
number_tries_to_submit_order = {}
#

# Stock candles
def get_historical_df(ticker='', interval='1h', period='2y', start_date=date.today(), end_date=date.today()):
    
  
    insturument = yf.Ticker(ticker)
    df = insturument.history(period=period, interval=interval)

    df = df.rename(columns={"Open": "open", "Close" :'close', "High" : 'high', "Low": 'low'})

    df['pct'] = np.where(df['open'] < df['close'],  
                         (df['close'] / df['open'] - 1) * 100,
                         -(df['open'] / df['close'] - 1) * 100
    )
    # df.index = pd.to_datetime(df['t'], unit='s', utc=True).map(lambda x: x.tz_convert('America/New_York'))
    df = f2.get_heiken_ashi_v2(df)
    df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]

    return df 

def is_near_global_max(df, i, k=400, prt=70):

  if i > k:
   gmax = max(df['close'].iloc[i - k: i])
   reference_point = df['close'].iloc[i - k]
  else:
   reference_point = df['close'].iloc[0]
   gmax = max(df['close'].iloc[0: i])
 
  if gmax == reference_point:
    result = False
  elif 100 * (df['close'].iloc[i] - reference_point) / (gmax - reference_point) > prt:
    result = True
  else:
    result = False

  return result

def number_red_candles(df, i, k=11):

  if i < k:
    number_red_candles = (df['ha_colour'][0 : i] == 'red').sum()
  else:
    number_red_candles = (df['ha_colour'][i - k : i] == 'red').sum()
  return number_red_candles

def stock_buy_condition(df, ticker):
  '''
    Parameters:

    Returns:
      condition
  '''
  condition = False
  buy_price = 0

  last_top = df['close'].iloc[0]
  last_top_time = df['close'].index[0]
  last_top_i = 0
    
  range_ = range(df.shape[0] - 200, df.shape[0])

  for i in range_:
   # last top and reverse flag
    if df['ha_colour'].iloc[i] == 'red' \
      and df['ha_colour'].iloc[i - 1] == 'green'\
      and df['ha_colour'].iloc[i - 2] == 'green'\
      and df['ha_pct'].iloc[i - 1] > 0.1 \
      and df['ha_pct'].iloc[i - 2] > 0.1:
  
      last_top = df['high'].iloc[i - 1]
      last_top_i = i - 1

  i = df.shape[0] - 1

  # dynaminc profit/loose coefficient
  if df['close'].iloc[i] / df['close'].iloc[i - 200] > bull_trend_coef: # bull trend
    gain_coef = 1.02
    lose_coef = 0.95  
  else: # bear trend
    gain_coef = 1.005
    lose_coef = 0.98

  buy_ratio = float(last_top -  df['open'].iloc[i]) / float(df['ha_pct'].iloc[i])
  if df['ha_pct'].iloc[i] > RIV \
    and last_top / df['open'].iloc[i] > last_top_ratio \
    and i - last_top_i > distance_from_last_top \
    and buy_ratio > buy_ratio_border \
    and not(is_near_global_max(df, i, k=400, prt=is_near_global_max_prt)) \
    and number_red_candles(df, i) > 6:
    
    buy_price = float(df['close'].iloc[i])
    condition = True
  
  return condition, buy_price, gain_coef, lose_coef

def sell_stock_condition(order, current_price):

  buy_price = order['buy_price']
  gain_coef = order['gain_coef']
  lose_coef = order['lose_coef']

  sell_condition = False

  if current_price / buy_price >= gain_coef:
    sell_condition = True  
  
  if current_price / buy_price <= lose_coef:
    sell_condition = True

  return sell_condition

def current_profit(df, hours=24):
  try:
    ref_time = datetime.now().astimezone() - timedelta(hours=hours)
    if df.shape[0] == 0:
      current_profit = 0
    else:
      df = df[(df['sell_time'] >= ref_time)]
      current_profit = df['profit'].sum()
  except:
    current_profit = -12345
  return float(current_profit)

def update_buy_order_based_on_platform_data(order):
    history_orders = ma.get_history_orders()
    history_order = history_orders.loc[(history_orders['order_id'] == order['buy_order_id'])]
    if history_order.shape[0] > 0 \
    and history_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL:
      order['buy_price'] = float(history_order['dealt_avg_price'].values[0])
      order['buy_sum'] = float(order['buy_price'] * history_order['qty'].values[0])
      buy_commission = ma.get_order_commission(order['buy_order_id'])             
      order['buy_commission'] = buy_commission
    return order

if __name__ == '__main__':

  alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')
  ma = Moomoo_API(ip, port, trd_env=TRD_ENV, acc_id = ACC_ID)
  ti = TradeInterface(platform='moomoo', df_name='real_trade_db', moomoo_api=ma)
  # ti = TradeInterface(platform='test', df_name='test', moomoo_api=ma)
  df = ti.load_trade_history() # load previous history
  df = df.drop_duplicates()
  ti.__save_orders__(df)

  # BUY TEST
  # order = ti.buy_order(ticker='CWPE', buy_price=1.8, buy_sum=4)
  # order['gain_coef'] = 1.05
  # order['lose_coef'] = 0.98
  # order = update_buy_order_based_on_platform_data(order)
  # df = ti.record_order(order)
  # ma.cancel_order(order['buy_order_id'])

  # SELL TEST
  # historical_orders = ma.get_history_orders()
  # historical_order = historical_orders.iloc[0]
  # order = df.iloc[0]
  # order = ti.sell_order(order, sell_price=order['buy_price']*1.05, historical_order = historical_order)

  # GET HISTORY TEST
  # historical_orders = ma.get_history_orders()
  # for index, row in historical_orders.iterrows():
  #   if row['aux_price'].values[0] > 0:  ##!!!??>
  #     print('ok')  
        

  # Moomoo trade algo:
  # 1. Check what stocks are bought based on MooMoo 
  # 2. Check that for all bought stocks placed LIMIT and STOP orders ()
      # if ticker in bought_stocks \
      # and ticker in df and status is bought 
      # and not ORDER LIMIT and STOP
      # Place LIMIT-IF-TOUCH orders and STOP ORDER 
  # 3. Check history of executed orders. 
      # If one is executed cancel other
      # Update df, csv, sql
  # 4. Calculate day's profit
  # For optimal stock list:
  # 5. Buy condition + day's profit limitation 

  while True: 
    alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')  
    # 1. Check what stocks are bought based on MooMoo (position list) and df
    positions = ma.get_positions()
    limit_if_touched_sell_orders, stop_sell_orders = ma.get_sell_orders()
    # try:
    #   if limit_if_touched_sell_orders.shape[0] == 0:
    #     limit_if_touched_sell_orders_list = []
    #   else:
    #     limit_if_touched_sell_orders_list = []
    #     for order in limit_if_touched_sell_orders:
    #       ticker = order['code'].split('.')[1]
    #       limit_if_touched_sell_orders_list.append(ticker)

    #   if stop_sell_orders.shape[0] == 0:
    #     stop_sell_orders_list = []
    #   else:
    #     stop_sell_orders_list = []
    #     for order in stop_sell_orders:
    #       ticker = order['code'].split('.')[1]
    #       stop_sell_orders_list.append(ticker)
    # except:
    #   pass
    limit_if_touched_sell_orders_list = []
    for index, row in limit_if_touched_sell_orders.iterrows():
      ticker = row['code'].split('.')[1]
      limit_if_touched_sell_orders_list.append(ticker)
    stop_sell_orders_list = []
    for index, row in stop_sell_orders.iterrows():
      ticker = row['code'].split('.')[1]
      stop_sell_orders_list.append(ticker)

    historical_orders = ma.get_history_orders()
    # Check bought stock based on df 
    if df.shape[0] > 0:
      bought_stocks = df.loc[df['status'] == 'bought']
      bought_stocks_list = bought_stocks.ticker.to_list()
    else:
      bought_stocks_list = []

    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED and STOP orders
    for code in positions:
      ticker = code.split('.')[1]
      if ticker in bought_stocks_list:
        order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]   
        qty = order['stocks_number']  # stock number should be taken from the trade 
        if order['limit_if_touched_order_id'] is None \
          and not (ticker in limit_if_touched_sell_orders_list):
          price = order['buy_price'] * order['gain_coef']  # buy price should be taken from the trade platform
          order_id = ma.place_limit_if_touched_order(ticker, price, qty)
          if not (order_id is None):
            order['limit_if_touched_order_id'] = order_id
            df = ti.record_order(order)
        else:
          sell_order = limit_if_touched_sell_orders.loc[limit_if_touched_sell_orders['order_id'] == order['limit_if_touched_order_id']]
          if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED and \
              sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING:
            alarm.print(f'{ticker} limit if touched order has not been sumbitted')
            # if order status isn't SUBMITTED after 3 times cancel the order and RESUBMITIT it?

        if order['stop_order_id'] is None \
          and not (ticker in stop_sell_orders_list):
          price = order['buy_price'] * order['lose_coef']  # buy price should be taken from the trade platform
          order_id = ma.place_stop_order(ticker, price, qty)
          if not (order_id is None):
            order['stop_order_id'] = order_id
            df = ti.record_order(order)
        else:
          sell_order = stop_sell_orders.loc[stop_sell_orders['order_id'] == order['stop_order_id']]
          if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED and \
              sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTING:
            alarm.print(f'{ticker} stop order has not been sumbitted')
            # if order status isn't SUBMITTED after 3 times cancel the order and RESUBMITIT it
      else:
        alarm.print(f'{ticker} is not in DB!')
    
    # For optimal stock list:
    # 5. Buy condition + day's profit limitation
    for ticker in stock_name_list_opt:
      print(f'Stock is {ticker}')
      if ticker in bought_stocks_list:
        order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]
      else:
        order = []
      try:
        stock_df = get_historical_df(ticker = ticker, period=period, interval=interval)
      except Exception as e:
        print(f'{e}')
        stock_df = None
      
      if not(stock_df is None):
        if not(ticker in bought_stocks_list):
          buy_condition, buy_price, gain_coef, lose_coef = stock_buy_condition(stock_df, ticker)
        else:
          buy_condition = False
        # c.print(f'buy condition is {buy_condition}', color='blue')
        c.green_red_print(buy_condition, 'buy condition')
        print(f'stock {ticker}, time: {stock_df.index[-1]} last price is {stock_df['close'].iloc[-1]:.2f}')
        current_timezone = datetime.now().astimezone().tzinfo
        time_is_correct =  (datetime.now().astimezone() - stock_df.index[-1].astimezone(current_timezone)).seconds  < 60 * 60 * 1 + 60 * 5 
        c.print(f'Time is correct condition {time_is_correct}', color='yellow')

        if buy_condition and buy_price != 0 and time_is_correct:

          avl_withdrawal_cash = ma.get_avl_withdrawal_cash()
          profit_24hours = current_profit(df)
          c.print(f'Availble withdrawal cash is {avl_withdrawal_cash}')
          c.print(f'Last 24 hours profit is {profit_24hours}')

          # Check last 24 hours profit condition and
          # Check available money for trading based on Available Funds
          security_condition = True
          if avl_withdrawal_cash < min_buy_sum:
            security_condition = False
            alarm.print('Available funds less than minumim buy sum')

          # add condtion based on money permitted for trade !!!

          if profit_24hours <= stop_trading_profit_value:
            security_condition = False
            alarm.print(f'Profit value for last 24 hours {profit_24hours} is less {stop_trading_profit_value}')

          if stock_df['close'].iloc[-1] > max_stock_price:
            security_condition = False
            alarm.print(f'Stock price {stock_df['close'].iloc[-1]} more than maximum allowed price {max_stock_price}')

          # Calculate buy_sum based on available money and min and max buy_sum condition
          if avl_withdrawal_cash < default_buy_sum:
            buy_sum = avl_withdrawal_cash
          else:
            buy_sum = default_buy_sum
          if buy_sum < min_buy_sum:
            buy_sum = 0
          if buy_sum > max_buy_sum:
            buy_sum = max_buy_sum

          # if all conditions are met place buy order
          if security_condition:
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_sum=buy_sum)
            order['gain_coef'] = gain_coef
            order['lose_coef'] = lose_coef
            order = update_buy_order_based_on_platform_data(order)
            df = ti.record_order(df, order)

        # Recheck buy order information including commission from order history
        if ticker in bought_stocks_list: 
          # 1.111 was set during buy order creation
          if order['buy_commission'] == 1.111 \
            or math.isnan(order['buy_commission']) \
            or order['buy_commission'] == None \
            or order['buy_commission'] == 0:
            order = update_buy_order_based_on_platform_data(order)
            df = ti.update_order(df, order)
        
        # Checking if sell orders have been executed
        if ticker in bought_stocks_list:
           order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]  
           historical_limit_if_touched_order = historical_orders.loc[
              historical_orders['order_id'] == order['limit_if_touched_order_id']]
           historical_stop_order = historical_orders.loc[
              historical_orders['order_id'] == order['stop_order_id']]
           # checking and update limit if touched order
           if historical_limit_if_touched_order.shape[0] > 0 \
             and  historical_limit_if_touched_order['order_status'].values[0] == ft.OrderStatus.SUBMITTED\
             and order['status'] == 'bought':
             sell_price = order['buy_price'] * order['gain_coef']
             order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_limit_if_touched_order)
             df = ti.update_order(df, order)
             # cancel stop order
             ma.cancel_order(order['stop_order_id'])
          # checking and update stop order
           if historical_stop_order.shape[0] > 0 \
            and historical_stop_order['order_status'].values[0] == ft.OrderStatus.SUBMITTED\
            and order['status'] == 'bought':
             sell_price = order['buy_price'] * order['lose_coef']
             order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_stop_order)
             df = ti.update_order(df, order)
             # cancel limit-if-touched order
             ma.cancel_order(order['limit_if_touched_order_id'])
        
            # Update df, csv, sql !!!!!!!!!!!!

    print('Waiting progress:')
    for i in tqdm(range(60)):
      time.sleep(1)


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
import sql_db
import pytz

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
money_permitted_for_trade = 3400 * rate # in AUD * rate = USD
default_buy_sum = 2200 # in USD
min_buy_sum = 800 # in USD
max_buy_sum = 1450 # in  USD
stop_trading_profit_value = -30 * rate # in AUD * rate = USD
max_stock_price = 1050 # in  USD

# Moomoo settings
moomoo_ps = ps.Moomoo()
ip = '127.0.0.1'
port = 11111
unlock_pwd = moomoo_ps.unlock_pwd
ACC_ID = moomoo_ps.acc_id
TRD_ENV = ft.TrdEnv.REAL
MARKET = 'US.'
# ft.SysConfig.set_all_thread_daemon(True)

# Version 3.0
stock_name_list_opt = [
'BA', 'OXY', 'DHI', 'ON', 'PANW', 'AMD', 'MCHP', 'BSX', 'INTU', 'HLT', 'NXPI', 'AIG', 'EL',
'LLY', 'USB', 'AMAT', 'ADI', 'ANET', 'DXCM', 'LRCX', 'EOG', 'GE', 'MU', 'PSX', 'NUE', 'AVGO',
'WMB', 'TJX', 'SNPS', 'WMT', 'KLAC', 'MAR', 'SBUX', 'ECL', 'CDNS', 'EMR', 'TT', 'IDXX', 'VLO',
'TXN', 'F', 'ABBV', 'MPC', 'CAT', 'CSCO', 'NFLX', 'JCI', 'TDG', 'MRK', 'MRNA', 'DE', 'NOW',
'TMUS', 'GM', 'WELL', 'ETN', 'ICE', 'WM', 'CME', 'PCAR', 'CTAS', 'MSI', 'GILD', 'SLB', 'CMCSA',
'ROP', 'ADM', 'LOW', 'QCOM', 'VRTX', 'MO', 'EXC', 'CI', 'JNJ', 'CL', 'STZ', 'CMG', 'MMM', 'SCHW',
'GOOG', 'PH', 'LMT', 'HON', 'PEP', 'COP', 'CRM', 'MSCI', 'UNP', 'APD', 'ADP', 'CVS', 'GS', 'HUM',
'ADSK', 'TEL', 'IBM', 'ROK', 'MNST', 'CVX', 'ITW', 'ADBE', 'PM', 'SPG', 'TGT', 'PYPL', 'APH', 'FDX',
'CSX', 'SHW', 'TFC', 'UNH',
'AMT'
]

opt_stocks_for_bear_trend = ['BA', 'INTU', 'MCHP', 'LLY', 'DHI', 'ANET', 'AIG', 'NUE', 'MAR', 'OXY', 'ON',
  'GE', 'AMAT', 'NXPI', 'SNPS', 'UNP', 'KLAC', 'BSX', 'MSI', 'CRM', 'CAT', 'ADI',
    'ETN', 'JCI', 'HLT', 'CSCO', 'WMT', 'TDG', 'TT', 'ECL', 'LOW', 'ADSK', 'TJX',
    'VRTX', 'APH', 'ABBV', 'STZ', 'SBUX', 'DE', 'MRK', 'CTAS', 'MNST', 'CME', 'MO', 'TXN', 'ITW']

# stock_name_list_opt = ['EOG']

# settings for historical df from yfinance
period = '3mo'
interval = '1h' 

# settings for buy condition Version 3.0
is_near_global_max_prt = 96
distance_from_last_top  = 0
last_top_ratio = 1
RIV  = 0.05
buy_ratio_border = 0
bull_trend_coef = 1.07
number_tries_to_submit_order = {}

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
    if ticker in opt_stocks_for_bear_trend:
      gain_coef = 1.02
    else:
      gain_coef = 1.005
    lose_coef = 0.95  
  else: # bear trend
    gain_coef = 1.005
    lose_coef = 0.95

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
    ref_time = datetime.now() - timedelta(hours=hours)
    if df.shape[0] == 0:
      current_profit = 0
    else:
      df = df[(df['sell_time'] >= ref_time)]
      current_profit = df['profit'].sum()
  except Exception as e:
    alarm.print(e)
    current_profit = 10
  return float(current_profit)

def update_buy_order_based_on_platform_data(order):
    history_orders = ma.get_history_orders()
    history_order = history_orders.loc[(history_orders['order_id'] == order['buy_order_id'])]
    if history_order.shape[0] > 0 \
    and (history_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL 
        or history_order['order_status'].values[0] == ft.OrderStatus.CANCELLED_PART):
      order['buy_price'] = float(history_order['dealt_avg_price'].values[0])
      order['buy_sum'] = float(order['buy_price'] * history_order['qty'].values[0])
      buy_commission = ma.get_order_commission(order['buy_order_id'])             
      order['buy_commission'] = buy_commission
      order['stocks_number'] = int(history_order['qty'].values[0])
      order['status'] = 'bought'

      if history_order['order_status'].values[0] == ft.OrderStatus.CANCELLED_PART:
        ma.modify_limit_if_touched_order(order, order['gain_coef'])
        ma.modify_stop_order(order, order['lose_coef'])
    return order

def test_trading_simulation(ticker, stock_df, df_test, bought_stocks_list):
      
    if not(stock_df is None):

        if not(ticker in bought_stocks_list):
          buy_condition, buy_price, gain_coef, lose_coef = stock_buy_condition(stock_df, ticker)
        else:
          buy_condition = False
        current_timezone = datetime.now().astimezone().tzinfo
        time_is_correct =  (datetime.now().astimezone() - stock_df.index[-1].astimezone(current_timezone)).seconds  < 60 * 60 * 1 + 60 * 5 

        if buy_condition and buy_price != 0 and time_is_correct:
          order = ti_test.buy_order(ticker=ticker, buy_price=buy_price, buy_sum=1000.0)
          order['gain_coef'] = gain_coef
          order['lose_coef'] = lose_coef
          df_test = ti_test.record_order(df_test, order, sim=True)

        # Checking for sell condition
        if ticker in bought_stocks_list:
          order = df_test.loc[(df_test['ticker'] == ticker) & 
                              ((df_test['status'] == 'bought') | (df_test['status'] == 'filled part') |  (df_test['status'] == 'created'))
                             ].sort_values('buy_time').iloc[-1]
          order_book_is_available = False

          try:
            stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
          except Exception as e:
            print(f'Minute data for stock {ticker} has not been received')
      
          current_stock = stock_df_1m.sort_index().iloc[-1]
          if order_book_is_available:
            # get current price from order book
            current_price = 0
          else:
            current_price = current_stock['close']
          sell_condition = sell_stock_condition(order, current_price)

          if sell_condition:
            order = ti_test.sell_order(order, current_price)
            df_test = ti_test.update_order(df_test, order, sim=True)

    return df_test

def load_orders_from_csv():
  # FUNCTION TO UPDATE times from csv files to df with correct time format
  df = pd.read_csv('db/real_trade_db.csv', index_col='index')
  df['buy_time'] = pd.to_datetime(df['buy_time'], dayfirst=True)
  df['sell_time'] = pd.to_datetime(df['sell_time'], dayfirst=True)
  # tzinfo = pytz.timezone('Australia/Melbourne')
  # buy_times_list = []
  # sell_times_list = []
  # for index, row in df2.iterrows():
  #   if type(row['buy_time']) == str and'+' in row['buy_time']:
  #     dt = datetime.strptime(row['buy_time'].split('+')[0], '%Y-%m-%d %H:%M:%S.%f')
  #     buy_times_list.append(np.datetime64(dt))
  #   else:
  #     dt = datetime.strptime('1971-01-01 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
  #     buy_times_list.append(np.datetime64(dt))
  #   if type(row['sell_time']) == str and '+' in row['sell_time']:
  #     dt = datetime.strptime(row['sell_time'].split('+')[0], '%Y-%m-%d %H:%M:%S.%f')
  #     sell_times_list.append(np.datetime64(dt))
  #   else:
  #     dt = datetime.strptime('1971-01-01 00:00:00.000000', '%Y-%m-%d %H:%M:%S.%f')
  #     sell_times_list.append(np.datetime64(dt))
  # df2['buy_time'] = buy_times_list
  # df2['sell_time'] = sell_times_list
  # df = pd.concat([df, df2])
  return df

def get_orders_list_from_moomoo_orders(orders: pd.DataFrame):
  orders_list = []
  for index, row in orders.iterrows():
    ticker = row['code'].split('.')[1]
    orders_list.append(ticker)
  return orders_list

def place_sell_order_if_it_was_not_placed(df, order, sell_orders, sell_orders_list, price, order_type):
  '''
    - type: buy | limit_if_touched | stop | trailing_LIT
  '''
  moomoo_order_id = None
  try:
    ticker = order['ticker']
    order_id_type = order_type + '_order_id'
    order_id = order[order_id_type]
  except Exception as e:
    alarm.print(e) 
    order_id = None
 
  if order_id in [None, ''] \
    and ticker not in sell_orders_list:
    if order_type in ['limit_if_touched']:
      moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty)
    if order_type in ['stop']:
      moomoo_order_id = ma.place_stop_order(ticker, price, qty)   
    if order_type in ['trailing_LIT']:
      moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty, aux_price_coef = 1.0005, remark = 'trailing_LIT')
    if not (moomoo_order_id is None):
      order[order_id_type] = moomoo_order_id
      df = ti.record_order(df, order)
  else:
    sell_order = sell_orders.loc[sell_orders['order_id'] == order[order_id_type]]
    # if sell_order.shape[0] > 0:
    try:
      if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
          and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
          and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
        alarm.print(f'{ticker} {order_type} order has not been sumbitted')
    except Exception as e:
      alarm.print(e)
  return df, order

if __name__ == '__main__':

  load_from_csv = False

  alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')
  ma = Moomoo_API(ip, port, trd_env=TRD_ENV, acc_id = ACC_ID)
  ti = TradeInterface(platform='moomoo', df_name='real_trade_db', moomoo_api=ma)

  if load_from_csv:
    df = load_orders_from_csv()
  else:
    df = ti.load_trade_history() # load previous history
  df = df.drop_duplicates()
  
  for index, row in df.iterrows():
    ti.update_order(df, row)
  
  # df.iloc[0] = [0,'DE', datetime.now(), 374.06, 751, 0, None,0,0,0,2,'bought',1.005, 0.95,1.007,0,
  # 'FA1956E877FC84A000', 'FA1956E75EBF44A000', 'FA1956E877FC84A000', None] 
  # df.loc[] = [0,'DE', datetime.now(), 374.06, 751, 0, None,0,0,0,2,'bought',1.005, 0.95,1.007,0,
  # 'FA1956DF73E03B2000', 'FA1956E75EBF44A000', 'FA1956E877FC84A000', 'FA1956EEE2AC3B2000'] 
  # df._set_value(0, 'trailing_LIT_order_id' ,'FA1956EEE2AC3B2000')
  # df.drop(index=1, inplace=True)
  ti.__save_orders__(df)

  # SQL INIT
  try:
    parent_path = pathlib.Path(__file__).parent
    folder_path = pathlib.Path.joinpath(parent_path, 'sql')
    db = sql_db.DB_connection(folder_path, 'trade.db', df)
  except Exception as e:
    alarm.print(e)
  
  # TEST TRADING
  ti_test = TradeInterface(platform='test', df_name='test') # test trading
  df_test = ti_test.load_trade_history()

  # TESTS
  if True:
    pass
    # BUY TEST
    # order = ti.buy_order(ticker='CWPE', buy_price=1.8, buy_sum=4)
    # order['gain_coef'] = 1.05
    # order['lose_coef'] = 0.98
    # order = update_buy_order_based_on_platform_data(order)
    # df = ti.record_order(order)
    # ma.cancel_order(order, type='buy')

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

    # Get current orders and they lists:
    limit_if_touched_sell_orders, stop_sell_orders, limit_if_touched_buy_orders,\
        trailing_LIT_orders = ma.get_orders()
    limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
    limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
    stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
    trailing_LIT_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_orders)

    historical_orders = ma.get_history_orders()
    # Check bought stock based on df 
    if df.shape[0] > 0:
      bought_stocks = df.loc[(df['status'] == 'bought') | (df['status'] == 'filled part') | (df['status'] == 'placed') ]
      bought_stocks_list = bought_stocks.ticker.to_list()
    else:
      bought_stocks_list = []
    
    # TEST TRADING
    if df_test.shape[0] > 0:
      bought_stocks_test = df_test.loc[(df_test['status'] == 'bought') | (df_test['status'] == 'filled part') | (df_test['status'] == 'placed')]
      bought_stocks_test_list = bought_stocks_test.ticker.to_list()
    else:
      bought_stocks_test_list = []

    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED and STOP orders
    for code in positions:
      ticker = code.split('.')[1]
      if ticker in bought_stocks_list:
        order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]   
        qty = order['stocks_number']  # stock number should be taken from the trade 
        # Checking limit_if_touched_order
        price = order['buy_price'] * order['gain_coef']
        df, order = place_sell_order_if_it_was_not_placed(df,
          order=order,
          sell_orders=limit_if_touched_sell_orders,
          sell_orders_list=limit_if_touched_sell_orders_list,
          price=price,
          order_type='limit_if_touched')
        # Checking stop_order
        price = order['buy_price'] * order['lose_coef'] 
        df, order = place_sell_order_if_it_was_not_placed(df,
          order=order,
          sell_orders=stop_sell_orders,
          sell_orders_list=stop_sell_orders_list,
          price=price,
          order_type='stop')
        # Checing trailing_LIT_order
        if order['gain_coef'] > 1.005:
          price = order['buy_price'] * order['trailing_LIT_gain_coef'] 
          df, order = place_sell_order_if_it_was_not_placed(df,
            order=order,
            sell_orders=trailing_LIT_orders,
            sell_orders_list=trailing_LIT_orders_list,
            price=price,
            order_type='trailing_LIT')
          
        if False:
          if order['limit_if_touched_order_id'] in [None, ''] \
            and not (ticker in limit_if_touched_sell_orders_list):
            price = order['buy_price'] * order['gain_coef']  # buy price should be taken from the trade platform
            order_id = ma.place_limit_if_touched_order(ticker, price, qty)
            if not (order_id is None):
              order['limit_if_touched_order_id'] = order_id
              df = ti.record_order(df, order)
          else:
            sell_order = limit_if_touched_sell_orders.loc[limit_if_touched_sell_orders['order_id'] == order['limit_if_touched_order_id']]
            if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
              and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
              and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
              alarm.print(f'{ticker} limit if touched order has not been sumbitted')
          # Checking stop_order
          if order['stop_order_id'] is None \
            and not (ticker in stop_sell_orders_list):
            price = order['buy_price'] * order['lose_coef']  # buy price should be taken from the trade platform
            order_id = ma.place_stop_order(ticker, price, qty)
            if not (order_id is None):
              order['stop_order_id'] = order_id
              df = ti.record_order(df, order)
          else:
            sell_order = stop_sell_orders.loc[stop_sell_orders['order_id'] == order['stop_order_id']]
            if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
                and sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTING \
                and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
              alarm.print(f'{ticker} stop order has not been sumbitted')
          # Checing trailing_LIT_order
          if order['gain_coef'] > 1.005:
            if order['trailing_LIT_order_id'] is None \
              and not (ticker in trailing_LIT_orders_list):
              price = order['buy_price'] * order['trailing_LIT_gain_coef']  # buy price should be taken from the trade platform
              order_id = ma.place_limit_if_touched_order(ticker, price, qty, aux_price_coef = 1.0005, remark = 'trailing_LIT')
              if not (order_id is None):
                order['trailing_LIT_order_id'] = order_id
                df = ti.record_order(df, order)
            else:
              sell_order = trailing_LIT_orders.loc[trailing_LIT_orders['order_id'] == order['trailing_LIT_order_id']]
              if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
                and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
                and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
                alarm.print(f'{ticker} trailing limit if touched order has not been sumbitted')
      else:
        alarm.print(f'{ticker} is in positional list but not in DB!')
    
    # For optimal stock list:
    # 5. Buy condition + day's profit limitation
    for ticker in stock_name_list_opt:
      print(f'Stock is {ticker}')
      # REAL TRADING
      if ticker in bought_stocks_list:
        order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]      
      else:
        order = []
      try:
        stock_df = get_historical_df(ticker = ticker, period=period, interval=interval)
        try:
          stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
          current_price = stock_df_1m['close'].iloc[-1]
        except Exception as e:
          if not(stock_df is None):
            current_price = stock_df['close'].iloc[-1]
          alarm.print(e) 
        # TEST TRADING
        # df_test = test_trading_simulation(ticker, stock_df, df_test, bought_stocks_test_list)
      except Exception as e:
        print(f'{e}')
        stock_df = None
      
      # DYNAMIC GAIN COEFFICIENT
      # if gain_coef\lose_coef doesn't match order's modify the order:
      if not(stock_df is None):
        if ticker in bought_stocks_list:
          i = stock_df.shape[0] - 1
          if stock_df['close'].iloc[i] / stock_df['close'].iloc[i - 200] > bull_trend_coef: # bull trend
            if ticker in opt_stocks_for_bear_trend:
              gain_coef = 1.02
            else:
              gain_coef = 1.005

            if ticker in ['AMT']:
              gain_coef = 1.005
            
            lose_coef = 0.95  
          else: # bear 
            gain_coef = 1.005
            lose_coef = 0.95 
          
          # Trailing LIT gain coefficient:
          current_gain = current_price / order['buy_price']
          trailing_LIT_gain_coef = order['trailing_LIT_gain_coef']
          if current_gain >= 1.0058 and current_gain < 1.007:
            trailing_LIT_gain_coef = 1.005
          if current_gain >= 1.007:
            trailing_LIT_gain_coef = current_gain - 0.001
          if current_gain >= 1.012:
            trailing_LIT_gain_coef = current_gain - 0.002

          if order['gain_coef'] != gain_coef:
            ma.unlock_trade()
            order_id = ma.modify_limit_if_touched_order(order, gain_coef)
            if order_id != order['limit_if_touched_order_id']:
              order['limit_if_touched_order_id'] = order_id
            order['gain_coef'] = gain_coef
            df = ti.update_order(df, order)
          if order['lose_coef'] != lose_coef:
            ma.unlock_trade()
            order_id = ma.modify_stop_order(order, lose_coef)
            if order_id != order['stop_order_id']:
              order['stop_order_id'] = order_id
            order['lose_coef'] = lose_coef
            df = ti.update_order(df, order)
          if order['trailing_LIT_gain_coef'] != trailing_LIT_gain_coef:
            ma.unlock_trade()
            order_id = ma.modify_limit_if_touched_order(order, trailing_LIT_gain_coef,
                                                        aux_price_coef=1.0005,
                                                        order_type='trailing_LIT' 
                                                        )  
            if order_id != order['trailing_LIT_order_id']:
              order['trailing_LIT_order_id'] = order_id
            order['trailing_LIT_gain_coef'] = trailing_LIT_gain_coef
            df = ti.update_order(df, order)          

      # BUY SECTION:
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

          us_cash = ma.get_us_cash()
          profit_24hours = current_profit(df)
          c.print(f'Availble withdrawal cash is {us_cash}')
          c.print(f'Last 24 hours profit is {profit_24hours}')

          # Check last 24 hours profit condition and
          # Check available money for trading based on Available Funds
          security_condition = True
          if us_cash < min_buy_sum:
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
          if us_cash < default_buy_sum:
            buy_sum = us_cash
          else:
            buy_sum = default_buy_sum
          if buy_sum < min_buy_sum:
            buy_sum = 0
          if buy_sum > max_buy_sum:
            buy_sum = max_buy_sum

          # if all conditions are met place buy order
          if security_condition:
            # check buy price from 1 minute data before place the BUY order:
            try:
              stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
              buy_price = stock_df_1m['close'].iloc[-1]
            except Exception as e:
              alarm.print(e)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_sum=buy_sum)
            order['gain_coef'] = gain_coef
            order['lose_coef'] = lose_coef
            if order['status'] == 'placed':
              order = update_buy_order_based_on_platform_data(order)
              df = ti.record_order(df, order)

      # CHECKING FOR CANCELATION OF THE BUY ORDER
      # Check current price and status of buy order
      # If price more or equal than price*gain_coef CANCEL the BUY ORDER
      # If status was not FILLED_PART, change status of order to 'cancel', update the order
      if not(stock_df is None):
        if ticker in bought_stocks_list:
          try:
            stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
            current_price = stock_df_1m['close'].iloc[-1]
          except Exception as e:
            print(f'Minute data for stock {ticker} has not been received')
            current_price = stock_df['close'].iloc[-1]
          if current_price >= order['buy_price'] * order['gain_coef']:
            if limit_if_touched_buy_orders.shape[0] > 0:
              limit_if_touched_buy_order = limit_if_touched_buy_orders.loc[
                limit_if_touched_buy_orders['order_id'] == order['buy_order_id']]
              if limit_if_touched_buy_order['order_status'].values[0] != ft.OrderStatus.FILLED_PART:
                # cancel limit_if_touched_sell_order and stop_order if they was placed
                if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='limit_if_touched')    
                if order['stop_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='stop')
                order['status'] = 'cancelled'
              else:
                order['status'] = 'filled part'
              # cancel buy limit order
              ma.cancel_order(order, order_type='buy')
              df = ti.update_order(df, order)
      
      # Recheck buy order information including commission from the order history
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
          historical_trailing_LIT_order = historical_orders.loc[
            historical_orders['order_id'] == order['trailing_LIT_order_id']]
          # checking and update limit if touched order
          if historical_limit_if_touched_order.shape[0] > 0 \
            and  historical_limit_if_touched_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
            and order['status'] in ['placed', 'bought', 'filled part']:
            sell_price = order['buy_price'] * order['gain_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_limit_if_touched_order)
            df = ti.update_order(df, order)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            # cancel stop order
            ma.cancel_order(order, order_type='stop')
            ma.cancel_order(order, order_type='trailing_LIT')
        # checking and update stop order
          if historical_stop_order.shape[0] > 0 \
          and historical_stop_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['placed', 'bought', 'filled part']:
            sell_price = order['buy_price'] * order['lose_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_stop_order)
            df = ti.update_order(df, order)
            # cancel limit-if-touched order
            ma.cancel_order(order, order_type='limit_if_touched')
            ma.cancel_order(order, order_type='trailing_LIT')
        # checking and update trailing limit if touched order
          if historical_trailing_LIT_order.shape[0] > 0 \
            and  historical_trailing_LIT_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
            and order['status'] in ['placed', 'bought', 'filled part']:
            sell_price = order['buy_price'] * order['trailing_LIT_gain_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_trailing_LIT_order)
            df = ti.update_order(df, order)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            # cancel stop order and limit if touched 
            ma.cancel_order(order, order_type='stop')
            ma.cancel_order(order, order_type='limit_if_touched')

    print('Waiting progress:')
    for i in tqdm(range(60)):
      time.sleep(1)



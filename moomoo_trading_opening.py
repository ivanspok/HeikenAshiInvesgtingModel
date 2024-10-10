#%% IMPORT
import yfinance as yf

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
import shutil

import warnings
warnings.filterwarnings('error')
warnings.filterwarnings(action="ignore", message="unclosed", category=ResourceWarning)

from colog.colog import colog
c = colog()
warning = colog(TextColor='purple')
alarm = colog(TextColor='red')
blue = colog(TextColor='blue')

from currency_converter import CurrencyConverter
cr = CurrencyConverter()
rate = cr.convert(1, 'AUD', 'USD')
#%% SETTINGS

# Trade settings 
default_buy_sum = 3300 # in USD should be 3300 !!!!!
min_buy_sum = 2000 # in USD  should be 2000 !!!!!
max_buy_sum = 3300 # in  USD should be 3300 !!!!!
stop_trading_profit_value = -150 * rate # in AUD * rate = USD
max_stock_price = 1050 # in  USD

order_life_time_min = 8

# Moomoo settings
moomoo_ps = ps.Moomoo()
ip = '127.0.0.1'
port = 11111
unlock_pwd = moomoo_ps.unlock_pwd
ACC_ID = moomoo_ps.acc_id
TRD_ENV = ft.TrdEnv.REAL
MARKET = 'US.'
# ft.SysConfig.set_all_thread_daemon(True)

stock_name_list = ['DXCM', 'BIIB', 'CCI', 'CVS', 'WFC', 'CSX', 'TXN', 'HD', 'DUK', 'ABBV', 'KO',
                        'PSA', 'BA', 'PYPL', 'SBUX', 'ADM', 'FDX', 'PGR', 'ADP', 'JCI', 'TEL', 'WM', 'IDXX',
                          'NUE', 'ITW', 'PCAR', 'SPG', 'BDX', 'PEP', 'KMB', 'ROK', 'DE', 'AMGN', 'MNST', 'SNPS', 
                          'CL', 'MMC', 'BLK', 'AJG', 'MDLZ', 'AMD', 'CB', 'SCHW', 'AVGO', 'WELL', 'MSCI', 'MRNA', 
                          'OXY', 'ADSK', 'APH', 'IBM', 'CHTR', 'EL', 'MDT', 'GS', 'NOC', 'ELV', 'USB', 'AEP', 'SRE',
                            'LMT', 'LIN', 'WMT', 'TT', 'PSX', 'UNP', 'MCD', 'CMG', 'ROP', 'QCOM', 'CVX', 'SO', 'NEE', 
                            'UNH', 'CDNS', 'AXP', 'MRK', 'CMCSA', 'MO', 'ORCL', 'NFLX', 'MPC', 'FTNT', 'MU', 'ECL',
                              'JPM', 'SHW', 'CSCO', 'ETN', 'GOOG', 'SYK', 'MCHP']

# settings for historical df from yfinance
period = 'max'
interval = '1m' 

# orders settings
trail_spread_coef = 0.0003
trailing_stop_limit_act_coef = 1.002
aux_price_coef = 1.0005

class Default():
  def __init__(self):
    self.gain_coef = 1.005
    self.trailing_ratio = 0.12
default = Default()

#%% FUNCTIONS

def get_historical_df(ticker='', interval='1h', period='2y', start_date=date.today(), end_date=date.today()):
    
    try:
      insturument = yf.Ticker(ticker)
      df = insturument.history(period=period, interval=interval, prepost=True)

      df = df.rename(columns={"Open": "open", "Close" :'close', "High" : 'high', "Low": 'low'})

      df['pct'] = np.where(df['open'] < df['close'],  
                          (df['close'] / df['open'] - 1) * 100,
                          -(df['open'] / df['close'] - 1) * 100
      )
      # df.index = pd.to_datetime(df['t'], unit='s', utc=True).map(lambda x: x.tz_convert('America/New_York'))
      df = f2.get_heiken_ashi_v2(df)
      df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]
      df = df.sort_index(ascending=True)
    except:
      df = pd.DataFrame()

    return df 

def more_than_pct(value1, value2, pct):
  if 100 * (value1 / value2 - 1) > pct:
    return True
  else:
    return False

def is_near_global_max(df, i, k=400, prt=70):
  result = False
  try:
    if i > k:
      gmax = max(df['close'].iloc[i - k: i])
      gmin = min(df['close'].iloc[i - k: i])
      reference_point = df['close'].iloc[i - k]
    else:
      reference_point = df['close'].iloc[0]
      gmax = max(df['close'].iloc[0: i])
      gmin = min(df['close'].iloc[0: i])
    result = 100 * (df['close'].iloc[i] - gmin) / (gmax - gmin) > prt
  except Exception as e:
    alarm.print(e)
  return result

def number_red_candles(df, i, k=11):

  if i < k:
    number_red_candles = (df['ha_colour'][0 : i] == 'red').sum()
  else:
    number_red_candles = (df['ha_colour'][i - k : i] == 'red').sum()
  return number_red_candles

def stock_buy_condition(df, df_1m, ticker):
  '''
    Parameters:

    Returns:
      condition
  '''

  # change order to limit and correct buy price!!! 

  condition = False
  buy_price = 0
  condition_type = 'No one'
    
  range_ = range(df.shape[0] - 200, df.shape[0])

  i = df.shape[0] - 1
  i_1m = df_1m.shape[0] - 1

  # c.green_red_print(df['pct'].iloc[i] > 0.12, '''df['pct'].iloc[i] > 0.12''')
  # c.green_red_print(df['close'].iloc[i] > df['close'].iloc[i  - 1], '''df['close'].iloc[i] > df['close'].iloc[i  - 1]''')
  # c.green_red_print(df_1m['pct'].iloc[i_1m - 15 : i_1m].sum() < -0.4,  '''df_1m['pct'].iloc[i_1m - 6 : i_1m].sum() < -0.4''')
  print(f'''last 15 candle sum is {df_1m['pct'].iloc[i_1m - 15 : i_1m].sum():.2f}''')
  cond_1 = df['pct'].iloc[i] > 0.12
  cond_2 = df['close'].iloc[i] > df['close'].iloc[i  - 1]
  cond_3 = df_1m['pct'].iloc[i_1m - 15 : i_1m].sum() < -0.4
  cond_4 = df_1m['pct'].iloc[i_1m - 30 : i_1m].sum() < -0.4
  cond_5 = df_1m['close'].iloc[i_1m - 15 : i_1m].max()/df_1m['close'].iloc[i_1m] > 1.004
  ha_cond = df_1m['ha_colour'].iloc[i_1m] == 'green' and df_1m['ha_colour'].iloc[i_1m - 1] == 'green'

  cond_6 = df['pct'].iloc[i - 1] > 0.25
  cond_7 = df['pct'].iloc[i] > 0.05 

  if df.index[-1].hour == 9:
    c.green_red_print(cond_1, 'cond_1')
    c.green_red_print(cond_2, 'cond_2')
    c.green_red_print(cond_3, 'cond_3')
    c.green_red_print(cond_4, 'cond_4')
    c.green_red_print(cond_5, 'cond_5')
    if cond_1 and cond_2 and ha_cond \
      and (cond_3 or cond_4 or cond_5):
      buy_price = float(df['close'].iloc[i])
      condition = True      
      condition_type = '9:30 0.4%'  
  else:
    c.green_red_print(cond_6, 'cond_6')
    c.green_red_print(cond_7, 'cond_7')
    c.green_red_print(cond_3, 'cond_3')
    c.green_red_print(cond_4, 'cond_4')
    c.green_red_print(cond_5, 'cond_5')
    if cond_6 and cond_7 and ha_cond \
      and (cond_3 or cond_4 or cond_5):
   
      buy_price = float(df['close'].iloc[i])
      condition = True
      condition_type = 'not 9:30 0.4%'

  c2 = False
  c32 = False
  if df.index[i].hour == 11 and df.index[i].minute == 30:
    # i : 11hours, i - 1 : 10hours, i - 2: 9hours, i - 3 : 15hours
    CM930 = df['close'].iloc[i - 2] 
    CM1030 = (df['close'].iloc[i - 1] + df['open'].iloc[i - 1]) / 2
    CM1130 = df['close'].iloc[i]
    # 930 candle is red; and 
    c2 = more_than_pct(CM1030, CM930, 0.21) and more_than_pct(CM1130, CM1030, 0.1) \
      and df['open'].iloc[i - 2] / df['close'].iloc[i - 2]  > 1.01 \
      and df['close'].iloc[i - 3] / df['close'].iloc[i - 2] < 1.002 \
      and df['close'].iloc[i - 3] / df['close'].iloc[i - 2] > 0.99 \
      and ha_cond
    c32 = more_than_pct(CM1130, CM1030, 0.01) and more_than_pct(CM930, CM1030, 0.01) and more_than_pct(CM1130, CM930, 0.4)\
     and df_1m['close'].iloc[i_1m - 30 : i_1m].max() / df_1m['close'].iloc[i_1m] < 1.002 \
     and ha_cond
    c.print(f'CM930 is {CM930:.2f}, CM1030 is {CM1030:.2f}, CM1130 is {CM1130:.2f}', color='blue')

  if c2 or c32:
    condition = True
    condition_type = 'c2 or c32'
  # c32 = False
  # if df.index[i].hour == 12 and df.index[i].minute == 30:
  #   # i : 12hours, i - 1 : 11hours, i - 2: 10hours, i - 3 : 9hours
  #   CM930 = df['close'].iloc[i - 3]
  #   CM1030 = (df['close'].iloc[i - 2] + df['open'].iloc[i - 2]) / 2
  #   CM1130 = (df['close'].iloc[i - 1] + df['open'].iloc[i - 1]) / 2
  #   c32 = more_than_pct(CM1130, CM1030, 0.01) and more_than_pct(CM930, CM1030, 0.01) and more_than_pct(CM1130, CM930, 0.4) # local minumum (1.6)
  c42 = False
  if df.index[i].hour == 12 and df.index[i].minute == 30:
    # i : 11hours, i - 1 : 10hours, i - 2: 9hours, i - 3 : 15hours
    CM930 = df['close'].iloc[i - 3] 
    CM1030 = (df['close'].iloc[i - 2] + df['open'].iloc[i - 2]) / 2
    CM1130 = (df['close'].iloc[i - 1] + df['open'].iloc[i - 1]) / 2
    CM1230 = df['close'].iloc[i]
    c42 = more_than_pct(CM1230, CM1130, 0.01) and more_than_pct(CM930, CM1030, 0.01) and more_than_pct(CM1230, CM930, 0.4) \
      and df_1m['close'].iloc[i_1m - 30 : i_1m].max() / df_1m['close'].iloc[i_1m] < 1.002 \
      and ha_cond
    c.print(f'CM930 is {CM930:.2f}, CM1030 is {CM1030:.2f}, CM1130 is {CM1130:.2f}', color='blue')

  if c42:
    condition = True
    condition_type = 'c42'

  return condition, buy_price, condition_type  

def stock_buy_condition_1m(df):
  '''
    Parameters:

    Returns:
      condition
  '''
  condition = False
  buy_price = 0
  if False:
    i = df.shape[0] - 1

    # if df['ha_pct'].iloc[i - 1] > 0.01 \
    #   and number_red_candles(df, i, k=8) >= 7  \
    #   and is_near_global_max(df, i, k=120, prt=58):

    #  and df['ha_colour'].iloc[i - 3] == 'red'\
    # and df['ha_colour'].iloc[i - 2] == 'red'\


    # print(f'''df['ha_pct'].iloc[i] is {df['ha_pct'].iloc[i]:.2f}, df['pct'].iloc[i-30 : i].sum() is {df['pct'].iloc[i-30 : i].sum():.2f}''')
    # print(f'i, i-60, i-120 : {df['close'].iloc[i]:.2f}, {df['close'].iloc[i- 60]:.2f}, {df['close'].iloc[i - 120]:.2f}')

    if df['ha_pct'].iloc[i] > 0 \
      and is_near_global_max(df, i, k=240, prt=20) \
      and df['ha_pct'].iloc[i] >= 0.02 \
      and df['ha_pct'].iloc[i] <= 0.05 \
      and df['pct'].iloc[i-30 : i].sum() >= 0.2 \
      and df['close'].iloc[i] > df['close'].iloc[i - 120] \
      and df['close'].iloc[i] > df['close'].iloc[i - 60] \
      and df['close'].iloc[i - 60] > df['close'].iloc[i - 120]:

          # and df['pct'].iloc[i-30 : i].sum() <= 0.6 \
    
      buy_price = float(max(df['close'][i - 4 : i]))
      condition = True
    
  return condition, buy_price

def stock_buy_condition_1230(df, df_1m):
  i_1m = df_1m.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[i_1m] == 'green' and df_1m['ha_colour'].iloc[i_1m - 1] == 'green'
  condition = False
  if df.index[-1].hour == 13 and df.index[-2].hour == 12:
    # -1: 13:30; -2: 12:30: -3: 11:30; -4: 10:30; -5: 9:30
    # candle is green
    # candle pct more than 0 
    # more taht close 9:30
    # sum pct 10:30 - 12:30  more than 0 
    # 9:30 less than -0l77
    if ha_cond \
      and df['close'].iloc[-2] > df['open'].iloc[-2] \
      and df['pct'].iloc[-2] > 0 \
      and df['close'].iloc[-2] > df['close'].iloc[-5] \
      and df['pct'].iloc[-4:-1].sum() > 0\
      and (df['pct'].iloc[-5] < -0.58 \
           or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006):
      condition = True

  return condition
      
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

def update_buy_order_based_on_platform_data(order, history_orders = None):
    try:
      if type(history_orders) != pd.DataFrame:
        if history_orders == None:
          history_orders = ma.get_history_orders()
      else:
        if history_orders.empty:
          history_orders = ma.get_history_orders()
    except Exception as e:
      alarm.print(e)

    try:
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
        # add order['buy_time']?

        if history_order['order_status'].values[0] == ft.OrderStatus.CANCELLED_PART:
          ma.modify_limit_if_touched_order(order, order['gain_coef'])
          ma.modify_stop_order(order, order['lose_coef'])
    except Exception as e:
      alarm.print(e)
    return order

def load_orders_from_csv(file_name):
  # FUNCTION TO UPDATE times from csv files to df with correct time format
  df = pd.read_csv(f'db/{file_name}.csv', index_col='Unnamed: 0')
  df['buy_time'] = pd.to_datetime(df['buy_time'], dayfirst=False)
  df['sell_time'] = pd.to_datetime(df['sell_time'], dayfirst=False)
  return df

def load_orders_from_xlsx(file_name):
  try:
    date = str(datetime.now()).split(' ')[0]
    shutil.copyfile(f'db/{file_name}.xlsx', f'db/bin/{file_name}_{date}.xlsx')
  except Exception as e:
    alarm.print(e)
  try:
    df = pd.read_excel(f'db/{file_name}.xlsx', index_col='Unnamed: 0')
  except:
    df = ti.load_trade_history()
  return df

def get_orders_list_from_moomoo_orders(orders: pd.DataFrame):
  orders_list = []
  for index, row in orders.iterrows():
    ticker = row['code'].split('.')[1]
    orders_list.append(ticker)
  return orders_list

def isNaN(num):
    return num != num

def place_sell_order_if_it_was_not_placed(df, order, sell_orders, sell_orders_list, price, order_type, current_gain=1):
  '''
    - type: buy | limit_if_touched | stop | trailing_LIT | trailing_stop_limit
  '''
  moomoo_order_id = None
  try:
    ticker = order['ticker']
    order_id_type = order_type + '_order_id'
    order_id = order[order_id_type]
    qty = order['stocks_number']
  except Exception as e:
    alarm.print(e)
    order_id = None
 
  if (order_id in [None, '', 'FAXXXX'] 
    or isNaN(order_id)) \
    and ticker not in sell_orders_list:
    if order_type == 'limit_if_touched':
      moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty)
    if order_type == 'stop':
      moomoo_order_id = ma.place_stop_order(ticker, price, qty)   
    if order_type == 'trailing_LIT':
      moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty, aux_price_coef=aux_price_coef, remark='trailing_LIT')

    if order_type == 'trailing_stop_limit':
      trail_spread = order['buy_price'] * trail_spread_coef
      trail_value = default.trailing_ratio
      moomoo_order_id = ma.place_trailing_stop_limit_order(ticker, price, qty, trail_value=trail_value, trail_spread=trail_spread)
      order['trailing_ratio'] = trail_value
      # if not (moomoo_order_id is None):
      #   ma.cancel_order(order=order, order_type='trailing_LIT')

    if order_type == 'stop_limit_sell':
      moomoo_order_id = ma.place_stop_limit_sell_order(ticker, price, qty)

    if not (moomoo_order_id is None):
      order[order_id_type] = moomoo_order_id
      df = ti.update_order(df, order)

  else:
    if sell_orders.shape[0] > 0:
      sell_order = sell_orders.loc[sell_orders['order_id'] == order[order_id_type]]
    else:
      sell_order = pd.DataFrame()
    try:
      if sell_order.shape[0] > 0:
        if sell_order['order_status'].values[0] != ft.OrderStatus.SUBMITTED \
            and sell_order['order_status'].values[0]  != ft.OrderStatus.SUBMITTING \
            and sell_order['order_status'].values[0]  != ft.OrderStatus.WAITING_SUBMIT:
          alarm.print(f'{ticker} {order_type} order has not been sumbitted')
      else:
        alarm.print(f'{ticker} {order_type} order has not been placed')
    except Exception as e:
      alarm.print(e)
      alarm.print(f'{ticker} {order_type} order has not been placed')
  return df, order

def clean_cancelled_and_failed_orders_history(df, type):
    historical_orders = ma.get_history_orders()
    placed_stocks = df.loc[(df['status'] == type)]
    for index, row in placed_stocks.iterrows():
      ticker = row['ticker']
      buy_order = historical_orders.loc[historical_orders['order_id'] == row['buy_order_id']]
      if buy_order['order_status'].values[0] in [ft.OrderStatus.CANCELLED_ALL, ft.OrderStatus.FAILED]:
        index = df.loc[(df['buy_order_id'] == row['buy_order_id']) & (df['status'] == type) & (df['ticker'] == ticker)].index
        df.drop(index=index, inplace=True)
    return df

def get_bought_and_placed_stock_list(df):
  if df.shape[0] > 0:
    bought_stocks = df.loc[(df['status'] == 'bought') | (df['status'] == 'filled part')]
    placed_stocks = df.loc[(df['status'] == 'placed')]
    bought_stocks_list = bought_stocks.ticker.to_list()
    placed_stocks_list = placed_stocks.ticker.to_list()
  else:
    bought_stocks_list = []
    placed_stocks_list = []
    bought_stocks = pd.DataFrame()
    placed_stocks = pd.DataFrame()

  return bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list


#%% MAIN
if __name__ == '__main__':

  # All parameter should be False. Change to True if you need change\fix DB
  load_from_csv = False
  load_from_xslx = True
  clean_placed_orders = True
  clean_cancelled_orders = True
  read_sql_from_df = False
  test_trading = False
  # Interface initialization
  alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')
  ma = Moomoo_API(ip, port, trd_env=TRD_ENV, acc_id = ACC_ID)
  file_name = 'real_trade_opening_db'
  ti = TradeInterface(platform='moomoo', df_name=file_name, moomoo_api=ma)
  # Load trade history
  if load_from_csv:
    df = load_orders_from_csv(file_name=file_name)
  elif load_from_xslx:
    df = load_orders_from_xlsx(file_name=file_name)
  else:
    df = ti.load_trade_history() # load previous history
  df = df.drop_duplicates()
  # Cleaning
  if clean_placed_orders:
    df = clean_cancelled_and_failed_orders_history(df, type='placed')
  if clean_cancelled_orders:
    df = clean_cancelled_and_failed_orders_history(df, type='cancelled')
  
  if df.shape[0] == 1 and pd.isna(df['ticker'].iloc[0]):
    df = df.drop(index=df.index[0])
  # Save csv/excel/df/
  ti.__save_orders__(df)
  
  # Algorithm
  if True:
    pass
    # for chosen stocks place:
    # Place stop limit buy order with 0.15% trigger and 0.2% price
    # Place limit if touch sell order for profit
    # Place stop limit sell order for loss if price drops below 0.10% with price 0%
  
  alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')  
  while True: 
    current_minute = datetime.now().astimezone().minute 
    market_is_opening = datetime.now().astimezone().hour == 23 \
    and current_minute > 18 and current_minute < 25
    if market_is_opening:
      indicators = {}
      for ticker in tqdm(stock_name_list):
        try:
          stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m')
          max = np.maximum(stock_df_1m['close'].iloc[-1060 : -1].max(), stock_df_1m['close'].iloc[-1])
          min = np.minimum(stock_df_1m['close'].iloc[-1060 : -1].min(), stock_df_1m['close'].iloc[-1])
          indicator = (stock_df_1m['close'].iloc[-1] - min ) / (max - min)
          indicators[ticker] = indicator 
        except Exception as e:
          alarm.print(e)
      indicators = dict(sorted(indicators.items(), key=lambda item: item[1]))
      selected_stock_list = [key for key in list(indicators)[:10]]
      print(f'indicators valus are {indicators}')
      print(f'selected_stock_list is {selected_stock_list}')
    else:
      selected_stock_list = ['AMD', 'GOOG', 'V', 'INTC', 'DIS', 'F', 'MA','JPM', 'SHW', 'CSCO']
    
    # 1. Check what stocks are bought based on MooMoo (position list) and df
    positions_list = ma.get_positions()

    # Get current orders and they lists:
    limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, \
    limit_if_touched_buy_orders, trailing_LIT_orders, trailing_stop_limit_orders, \
      stop_limit_buy_orders, stop_limit_sell_orders = ma.get_orders()
    
    limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
    trailing_stop_limit_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_orders)
    stop_limit_buy_orders_list = get_orders_list_from_moomoo_orders(stop_limit_buy_orders)
    stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(stop_limit_sell_orders)

    historical_orders = ma.get_history_orders()
    # Check bought stocks and placed based on df 
    bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
    
    # Recheck all placed buy orders from the order history
    for ticker in placed_stocks_list:
      try: 
        order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])].sort_values('buy_time').iloc[-1]
        # 1.111 was set during buy order creation
        if order['buy_commission'] == 1.111 \
          or math.isnan(order['buy_commission']) \
          or order['buy_commission'] == None \
          or order['buy_commission'] == 0:
          order = update_buy_order_based_on_platform_data(order, historical_orders)
          if order['status'] == 'bought':
            df = ti.update_order(df, order)
        #Recheck all placed buy orders from the orders by cancel time condition  
        if (datetime.now() - order['buy_time']).seconds / 60 > order_life_time_min:    
          if stop_limit_buy_orders.shape[0] > 0:
            stop_limit_buy_order = stop_limit_buy_orders.loc[
              stop_limit_buy_orders['order_id'] == order['buy_order_id']]
            if not stop_limit_buy_order.empty:
              if stop_limit_buy_order['order_status'].values[0] != ft.OrderStatus.FILLED_PART:
                # cancel limit_if_touched_sell_order and stop_order if they was placed
                if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='limit_if_touched')
                if order['stop_limit_sell_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='stop_limit_sell')
                if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='trailing_stop_limit')
                order['status'] = 'cancelled'
              else:
                order['status'] = 'filled part'
            # cancel buy limit order
            ma.cancel_order(order, order_type='buy')
            df = ti.update_order(df, order)
      except Exception as e:
        alarm.print(e)

    # Check statuses of all bought stocks if they not in positional list:
    try:
      # ticker in bought stocks list after confirmation of the buy order
      for ticker in bought_stocks_list:
          order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time').iloc[-1] 
          if ticker not in positions_list:
            historical_order = historical_orders.loc[(historical_orders['order_status']  == ft.OrderStatus.FILLED_ALL) &
                                                      (historical_orders['code'] == MARKET + ticker) &
                                                      (historical_orders['trd_side'] == ft.TrdSide.SELL) &
                                                      (historical_orders['qty'] == order['stocks_number'])
                                                      ].sort_values('updated_time').iloc[-1]
            order = ti.sell_order(order, sell_price=0.01, historical_order = historical_order)
            df = ti.update_order(df, order)
            if order['limit_if_touched_sell_order_id'] not in ['', None, []]:                  
              ma.cancel_order(order, order_type='limit_if_touched_sell_limit')
            if order['stop_limit_sell_order_id'] not in ['', None, []]:                  
              ma.cancel_order(order, order_type='stop_limit_sell_order')
    except Exception as e:
      alarm.print(e)

    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if needed
    for ticker in positions_list:
      # ticker = code.split('.')[1]
      try:
        if ticker in bought_stocks_list:
          order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]   
          qty = order['stocks_number']  # stock number should be taken from the trade 
        # Checking limit if touched sell order
          price = order['buy_price'] * order['gain_coef'] 
          df, order = place_sell_order_if_it_was_not_placed(df,
            order=order,
            sell_orders=limit_if_touched_sell_orders,
            sell_orders_list=limit_if_touched_sell_orders_list,
            price=price,
            order_type='limit_if_touched_sell')
        # Checking stop limit sell order
          price = order['buy_price'] * 0.9982
          df, order = place_sell_order_if_it_was_not_placed(df,
            order=order,
            sell_orders=stop_limit_sell_orders,
            sell_orders_list=stop_limit_sell_orders_list,
            price=price,
            order_type='stop_limit_sell')
        # Checking trailing_stop_limit_order
          try:
            stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
            current_price = stock_df_1m['close'].iloc[-1]
            if current_price >= order['buy_price'] * trailing_stop_limit_act_coef:
              price = order['buy_price'] * default.gain_coef
              df, order = place_sell_order_if_it_was_not_placed(df,
                order=order,
                sell_orders=trailing_stop_limit_orders,
                sell_orders_list=trailing_stop_limit_orders_list,
                price=price,
                order_type='trailing_stop_limit') 
          except Exception as e:
            alarm.print(e)    
        else:
          alarm.print(f'{ticker} is in positional list but not in DB!')
      except Exception as e:
        alarm.print(e)


      # 3.7 Checking if sell orders have been executed
      try:
        if ticker in bought_stocks_list:
          order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time').iloc[-1]
          historical_limit_if_touched_order = historical_orders.loc[
            historical_orders['order_id'] == order['limit_if_touched_order_id']]
          historical_stop_limit_sell_order = historical_orders.loc[
            historical_orders['order_id'] == order['stop_limit_sell_order_id']]
          # checking and update limit if touched order
          if historical_limit_if_touched_order.shape[0] > 0 \
            and  historical_limit_if_touched_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
            and order['status'] in ['bought', 'filled part']:
            sell_price = order['buy_price'] * order['gain_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order=historical_limit_if_touched_order)
            df = ti.update_order(df, order)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            ma.cancel_order(order, order_type='stop_limit_sell')
        # checking and update stop order
          if historical_stop_limit_sell_order.shape[0] > 0 \
          and historical_stop_limit_sell_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['bought', 'filled part']:
            sell_price = order['buy_price'] * 0.998
            order = ti.sell_order(order, sell_price=sell_price, historical_order=historical_stop_limit_sell_order)
            df = ti.update_order(df, order)
            # cancel limit-if-touched order
            ma.cancel_order(order, order_type='limit_if_touched')
      except Exception as e:
        alarm.print(e)

    # 3. For selected stock list:

    current_minute = datetime.now().astimezone().minute 
    market_is_opening = datetime.now().astimezone().hour == 23 \
      and current_minute > 25 and current_minute < 34
    # market_is_opening = True # TEST DELETE IT

    us_cash = ma.get_us_cash()
    c.print(f'Available cash is {us_cash:.2f}, min buy sum is {min_buy_sum}', color='turquoise')
    if us_cash > min_buy_sum \
      and market_is_opening:

      for ticker in selected_stock_list:
        print(f'Stock is {ticker}')
        order = []    
        
        # 3.2 Get historical data, current_price for stocks in optimal list
        try:
            stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m')
            current_price = stock_df_1m['close'].iloc[-1]
            current_timezone = datetime.now().astimezone().tzinfo
            time_is_correct =  (datetime.now().astimezone() - stock_df_1m.index[-1].astimezone(current_timezone)).seconds  < 60 * 3
        except Exception as e:
            stock_df_1m = None
            alarm.print(e)     

        # 3.4 BUY SECTION:
        if not(stock_df_1m is None):

          if not(ticker in bought_stocks_list or ticker in placed_stocks_list):

            current_timezone = datetime.now().astimezone().tzinfo
            time_is_correct =  (datetime.now().astimezone() - stock_df_1m.index[-1].astimezone(current_timezone)).seconds  <  60 * 5 
            c.print(f'Time is correct condition {time_is_correct}', color='yellow')

            if time_is_correct: # TEST DELETE It

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

              if stock_df_1m['close'].iloc[-1] > max_stock_price:
                security_condition = False
                alarm.print(f'''Stock price {stock_df_1m['close'].iloc[-1]} more than maximum allowed price {max_stock_price}''')

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
                
                buy_condition_type = 'opening'
                buy_price = stock_df_1m['close'].iloc[-1]
                
                # play sound:
                winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
                order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_condition_type=buy_condition_type, buy_sum=buy_sum)
                order['gain_coef'] = default.gain_coef

                if order['status'] == 'placed':
                  order = update_buy_order_based_on_platform_data(order)
                  df = ti.record_order(df, order)

                  # Place sell orders straigh away
                  # limit if touched sell order
                  price = order['buy_price'] * order['gain_coef'] 
                  df, order = place_sell_order_if_it_was_not_placed(df,
                    order=order,
                    sell_orders=limit_if_touched_sell_orders,
                    sell_orders_list=limit_if_touched_sell_orders_list,
                    price=price,
                    order_type='limit_if_touched')
                # stop limit sell order
                  price = order['buy_price'] * 0.9972
                  df, order = place_sell_order_if_it_was_not_placed(df,
                    order=order,
                    sell_orders=stop_limit_sell_orders,
                    sell_orders_list=stop_limit_sell_orders_list,
                    price=price,
                    order_type='stop_limit_sell')
      
    print('Waiting progress:')
    for i in tqdm(range(1)):
      time.sleep(1)
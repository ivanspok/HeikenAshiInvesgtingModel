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
money_permitted_for_trade = 3400 * rate # in AUD * rate = USD
default_buy_sum = 2200 # in USD
min_buy_sum = 800 # in USD
max_buy_sum = 2200 # in  USD
stop_trading_profit_value = -150 * rate # in AUD * rate = USD
max_stock_price = 1050 # in  USD

order_1m_life_time_min = 10
order_1h_life_time_min = 2
place_trailing_stop_order_imidiately = False

# Moomoo settings
moomoo_ps = ps.Moomoo()
ip = '127.0.0.1'
port = 11111
unlock_pwd = moomoo_ps.unlock_pwd
ACC_ID = moomoo_ps.acc_id
TRD_ENV = ft.TrdEnv.REAL
MARKET = 'US.'
# ft.SysConfig.set_all_thread_daemon(True)

# Verstion 4.0
stock_name_list_opt = ['MAR', 'MO', 'DHI', 'PYPL', 'HLT', 'BIIB', 'ZTS', 'TMUS', 'CMG', 'WELL', 'DUK',
                       'MCHP', 'MS', 'ROK', 'INTU', 'EXC', 'CI', 'STZ', 'NXPI', 'COF', 'DIS', 'EL', 'MU',
                       'NOW', 'CAT', 'HUM', 'OXY', 'PSX', 'COP', 'QCOM', 'ON', 'CVS', 'VLO', 'TJX', 'PGR',
                       'TXN', 'EOG', 'CSX', 'PM', 'MMM', 'ADM', 'GM', 'PEP', 'VZ', 'REGN', 'PCAR', 'GILD',
                       'EW', 'ANET', 'GIS', 'LIN', 'TEL', 'VRTX', 'BLK', 'ELV', 'KO', 'DE', 'KLAC', 'SBUX',
                       'PLD', 'PANW', 'CHTR', 'SNPS', 'MCK', 'TFC', 'USB', 'TDG', 'PFE', 'MRK', 'CB', 'MPC',
                       'GOOG', 'TT', 'PSA', 'CME', 'AIG', 'IDXX', 'BSX', 'WFC', 'ABT', 'TGT', 'EQIX', 'CTAS',
                       'BA', 'AON', 'SYK', 'F', 'SRE', 'JCI', 'NKE', 'MCD', 'TMO', 'SO', 'WMT', 'UNH', 'CSCO',
                      'CRM', 'PG', 'JNJ', 'HCA', 'XOM', 'ETN', 'V', 'COST', 'GS', 'FCX', 'MDT', 'NEE', 'CL',
                       'BMY', 'SPG', 'ADI', 'ABBV', 'WM', 'NUE', 'MDLZ', 'MA', 'MRNA', 'ROP', 'AEP', 'IBM',
                      'EMR', 'JPM', 'APD', 'AMD', 'T', 'INTC', 'ITW', 'AMGN', 'ADSK', 'SHW', 'NOC', 'NSC', 'SCHW', 'ICE']

# Verstion 5.0
stock_name_list_opt = ['DXCM', 'BIIB', 'CCI', 'CVS', 'WFC', 'CSX', 'TXN', 'HD', 'DUK', 'ABBV', 'KO',
                        'PSA', 'BA', 'PYPL', 'SBUX', 'ADM', 'FDX', 'PGR', 'ADP', 'JCI', 'TEL', 'WM', 'IDXX',
                          'NUE', 'ITW', 'PCAR', 'SPG', 'BDX', 'PEP', 'KMB', 'ROK', 'DE', 'AMGN', 'MNST', 'SNPS', 
                          'CL', 'MMC', 'BLK', 'AJG', 'MDLZ', 'AMD', 'CB', 'SCHW', 'AVGO', 'WELL', 'MSCI', 'MRNA', 
                          'OXY', 'ADSK', 'APH', 'IBM', 'CHTR', 'EL', 'MDT', 'GS', 'NOC', 'ELV', 'USB', 'AEP', 'SRE',
                            'LMT', 'LIN', 'WMT', 'TT', 'PSX', 'UNP', 'MCD', 'CMG', 'ROP', 'QCOM', 'CVX', 'SO', 'NEE', 
                            'UNH', 'CDNS', 'AXP', 'MRK', 'CMCSA', 'MO', 'ORCL', 'NFLX', 'MPC', 'FTNT', 'MU', 'ECL',
                              'JPM', 'SHW', 'CSCO', 'ETN', 'GOOG', 'SYK', 'MCHP']


opt_stocks_for_bear_trend = ['BA', 'INTU', 'MCHP', 'LLY', 'DHI', 'ANET', 'AIG', 'NUE', 'MAR', 'OXY', 'ON',
  'GE', 'AMAT', 'NXPI', 'SNPS', 'UNP', 'KLAC', 'BSX', 'MSI', 'CRM', 'CAT', 'ADI',
    'ETN', 'JCI', 'HLT', 'CSCO', 'WMT', 'TDG', 'TT', 'ECL', 'LOW', 'ADSK', 'TJX',
    'VRTX', 'APH', 'ABBV', 'STZ', 'SBUX', 'DE', 'MRK', 'CTAS', 'MNST', 'CME', 'MO', 'TXN', 'ITW']

# stock_name_list_opt = ['AMD']

# settings for historical df from yfinance
period = '3mo'
interval = '1h' 

# settings for buy condition Version 3.0
is_near_global_max_prt = 120  # used to be 100
distance_from_last_top  = 0
last_top_ratio = 1
RIV  = 0.15
buy_ratio_border = 0
bull_trend_coef = 1.07
number_tries_to_submit_order = {}

# orders settings
lose_coef_1m = 0.995
lose_coef_1h = 0.995
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
    df = df.sort_index(ascending=True)

    return df 

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
if False:
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
      if df['ha_colour'].iloc[i - 1] == 'red' \
        and df['ha_colour'].iloc[i - 2] == 'green'\
        and df['ha_colour'].iloc[i - 3] == 'green'\
        and df['ha_pct'].iloc[i - 2] > 0.1 \
        and df['ha_pct'].iloc[i - 3] > 0.1:
    
        last_top = df['high'].iloc[i - 1]
        last_top_i = i - 1

    i = df.shape[0] - 1

    # changed from i to i-1 as i is dynamic price!!!
    buy_ratio = float(last_top -  df['open'].iloc[i]) / float(df['ha_pct'].iloc[i])
    if df['ha_pct'].iloc[i] > RIV \
      and last_top / df['open'].iloc[i] > last_top_ratio \
      and i - last_top_i > distance_from_last_top \
      and buy_ratio > buy_ratio_border \
      and not(is_near_global_max(df, i, k=400, prt=is_near_global_max_prt)) \
      and number_red_candles(df, i) > 6:
      
      buy_price = float(df['close'].iloc[i])
      condition = True
    
    return condition, buy_price

def stock_buy_condition(df, df_1m, ticker):
  '''
    Parameters:

    Returns:
      condition
  '''

  # change order to limit and correct buy price!!! 

  condition = False
  buy_price = 0
    
  range_ = range(df.shape[0] - 200, df.shape[0])

  i = df.shape[0] - 1
  i_1m = df_1m.shape[0] - 1

  c.green_red_print(df['pct'].iloc[i] > 0.5, '''df['pct'].iloc[i] > 0.5''')
  c.green_red_print(df['close'].iloc[i] > df['close'].iloc[i  - 1], '''df['close'].iloc[i] > df['close'].iloc[i  - 1]''')
  c.green_red_print(df_1m['pct'].iloc[i_1m - 5 : i_1m].sum() > 0,  '''df_1m['pct'].iloc[i - 5 : i].sum() ''')
  c.green_red_print(df_1m['pct'].iloc[i_1m - 3 : i_1m].sum() > 0 , '''df_1m['pct'].iloc[i - 3 : i].sum()''')
  c.green_red_print((df_1m['close'].iloc[i_1m] > df_1m['close'].iloc[i_1m - 6: i_1m]).all(), '''(df_1m['close'].iloc[i] > df_1m['close'].iloc[i - 6: i]).all()''')

  if df.index[-1].hour == 9:
    if df['pct'].iloc[i] > 0.5 \
      and df['close'].iloc[i] > df['close'].iloc[i  - 1]\
      and df_1m['pct'].iloc[i_1m - 5 : i_1m].sum() > 0\
      and df_1m['pct'].iloc[i_1m - 3 : i_1m].sum() > 0\
      and (df_1m['close'].iloc[i_1m] > df_1m['close'].iloc[i_1m - 6: i_1m]).all():
      buy_price = float(df['close'].iloc[i])
      condition = True
  else:
    if df['pct'].iloc[i - 1] > 0.5 \
      and df['pct'].iloc[i] > 0 \
      and df['close'].iloc[i] > df['close'].iloc[i  - 1]\
      and df_1m['pct'].iloc[i_1m - 5 : i_1m].sum() > 0\
      and df_1m['pct'].iloc[i_1m - 3 : i_1m].sum() > 0\
      and (df_1m['close'].iloc[i_1m] > df_1m['close'].iloc[i_1m - 6: i_1m]).all():
   
      buy_price = float(df['close'].iloc[i])
      condition = True
  
  return condition, buy_price


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

def test_trading_simulation(ticker, stock_df, df_test, bought_stocks_list):
      
    if not(stock_df is None):

        if not(ticker in bought_stocks_list):
          buy_condition, buy_price = stock_buy_condition(stock_df, ticker)
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
  df = pd.read_csv('db/real_trade_db.csv', index_col='Unnamed: 0')
  df['buy_time'] = pd.to_datetime(df['buy_time'], dayfirst=False)
  df['sell_time'] = pd.to_datetime(df['sell_time'], dayfirst=False)
  return df

def load_orders_from_xlsx():
  try:
    date = str(datetime.now()).split(' ')[0]
    shutil.copyfile('db/real_trade_db.xlsx', f'db/bin/real_trade_db_{date}.xlsx')
  except Exception as e:
    alarm.print(e)
  df = pd.read_excel('db/real_trade_db.xlsx', index_col='Unnamed: 0')
  return df

def get_orders_list_from_moomoo_orders(orders: pd.DataFrame):
  orders_list = []
  for index, row in orders.iterrows():
    ticker = row['code'].split('.')[1]
    orders_list.append(ticker)
  return orders_list

def isNaN(num):
    return num != num

def place_sell_order_if_it_was_not_placed(df, order, sell_orders, sell_orders_list, price, order_type):
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
      moomoo_order_id = ma.place_trailing_stop_limit_order(ticker, price, qty, trail_value=default.trailing_ratio, trail_spread=trail_spread)
      # if not (moomoo_order_id is None):
      #   ma.cancel_order(order=order, order_type='trailing_LIT')

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
  return bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list


#%% MAIN
if __name__ == '__main__':

  # All parameter should be False. Change to True if you need change\fix DB
  load_from_csv = False
  load_from_xslx = True
  clean_placed_orders = False
  clean_cancelled_orders = True
  read_sql_from_df = False
  test_trading = False
  # Interface initialization
  alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')
  ma = Moomoo_API(ip, port, trd_env=TRD_ENV, acc_id = ACC_ID)
  ti = TradeInterface(platform='moomoo', df_name='real_trade_db', moomoo_api=ma)
  # Load trade history
  if load_from_csv:
    df = load_orders_from_csv()
  elif load_from_xslx:
    df = load_orders_from_xlsx()
  else:
    df = ti.load_trade_history() # load previous history
  df = df.drop_duplicates()
  # Cleaning
  if clean_placed_orders:
    df = clean_cancelled_and_failed_orders_history(df, type='placed')
  if clean_cancelled_orders:
    df = clean_cancelled_and_failed_orders_history(df, type='cancelled')
  # Save csv/excel/df/
  ti.__save_orders__(df)
  # Print last hours profit
  profit_24hours = current_profit(df, hours=24)
  warning.print(f'Last 24 hours profit is {profit_24hours:.2f}')
  profit_48hours = current_profit(df, hours=48)
  warning.print(f'Last 48 hours profit is {profit_48hours:.2f}')

  # SQL INIT
  try:
    parent_path = pathlib.Path(__file__).parent
    folder_path = pathlib.Path.joinpath(parent_path, 'sql')
    db = sql_db.DB_connection(folder_path, 'trade.db', df)
    if load_from_csv or load_from_xslx or read_sql_from_df:
      db.update_db_from_df(df)
  except Exception as e:
    alarm.print(e)
  
  # TEST TRADING
  if test_trading:
    ti_test = TradeInterface(platform='test', df_name='test') # test trading
    df_test = ti_test.load_trade_history()
  
  # Algorithm
  if True:
    pass  
    # Moomoo trade algo:
    # 1. Check what stocks are bought based on MooMoo 
        # Recheck all placed buy orders from the order history
        # Recheck all placed buy orders from the orders by cancel time condition
        # Check statuses of all bought stocks if they not in positional list:
    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if condition
    # 3. For optimal stock list:
      # 3.1 If counter condition:
        # Recheck for all tickers in positional list that LIMIT/trailing_LIT_order is placed if condition
        # Recalculate Trailing LIT gain coefficient:
        # Recheck all placed order information  
      # 3.2 Get historical data, current_price for bought stocks
      # 3.3 DYNAMIC GAIN COEFFICIENT
          # if gain_coef\lose_coef doesn't match order's modify the order:
      # 3.4. BUY SECTION:
      # 3.5. Recheck buy order information including commission from the order history
      # 3.6. CHECKING FOR CANCELATION OF THE BUY ORDER
          # Check current price and status of buy order
          # If price more or equal than price*gain_coef CANCEL the BUY ORDER
          # If status was not FILLED_PART, change status of order to 'cancel', update the order
      # 3.7 Checking if sell orders have been executed

  market_value = -1 # start offset
  total_market_value = -99  
  # Calculate market direction
  total_market_value_2m = 0
  total_market_value_5m = 0
  total_market_value_30m = 0
  total_market_value_60m = 0
  warning.print('Calculation of market direction:')
  for ticker in tqdm(stock_name_list_opt):
      try:
          stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
          total_market_value_2m += stock_df_1m['pct'].iloc[-2:-1].sum()
          total_market_value_5m += stock_df_1m['pct'].iloc[-5:-1].sum()
          total_market_value_30m += stock_df_1m['pct'].iloc[-30:-1].sum()
          total_market_value_60m += stock_df_1m['pct'].iloc[-60:-1].sum()
      except Exception as e:
        alarm.print(e)

  while True: 
    alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')  
    # 1. Check what stocks are bought based on MooMoo (position list) and df
    positions_list = ma.get_positions()

    # Get current orders and they lists:
    limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, limit_if_touched_buy_orders,\
        trailing_LIT_orders, trailing_stop_limit_orders = ma.get_orders()
    limit_buy_orders_list = get_orders_list_from_moomoo_orders(limit_buy_orders)
    limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_buy_orders)
    limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
    stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
    trailing_LIT_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_orders)
    trailing_stop_limit_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_orders)

    historical_orders = ma.get_history_orders()
    # Check bought stocks and placed based on df 
    bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
    
    # TEST TRADING
    if test_trading:
      if df_test.shape[0] > 0:
        bought_stocks_test = df_test.loc[(df_test['status'] == 'bought') | (df_test['status'] == 'filled part')]
        bought_stocks_test_list = bought_stocks_test.ticker.to_list()
      else:
        bought_stocks_test_list = []

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
        if (datetime.now() - order['buy_time']).seconds / 60 > order_1h_life_time_min \
          or ((datetime.now() - order['buy_time']).seconds / 60 >= order_1m_life_time_min \
               and order['buy_condition_type'] == '1m'):
          if limit_buy_orders.shape[0] > 0:
            limit_buy_order = limit_buy_orders.loc[
              limit_buy_orders['order_id'] == order['buy_order_id']]
            if not limit_buy_order.empty:
              if limit_buy_order['order_status'].values[0] != ft.OrderStatus.FILLED_PART:
                # cancel limit_if_touched_sell_order and stop_order if they was placed
                if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='limit_if_touched') 
                if order['trailing_LIT_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='trailing_LIT')   
                if order['stop_order_id'] not in ['', None, []]:                  
                    ma.cancel_order(order, order_type='stop')
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
            if order['trailing_LIT_order_id'] not in ['', None, []]:                  
              ma.cancel_order(order, order_type='trailing_LIT')    
            if order['stop_order_id'] not in ['', None, []]:                  
              ma.cancel_order(order, order_type='stop')
            if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
              ma.cancel_order(order, order_type='trailing_stop_limit')
    except Exception as e:
      alarm.print(e)

    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if needed
    for ticker in positions_list:
      # ticker = code.split('.')[1]
      try:
        if ticker in bought_stocks_list:
          order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1]   
          qty = order['stocks_number']  # stock number should be taken from the trade 

          # Checking limit_if_touched_order
          if False:
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
          # Checking trailing_LIT_order
          price = order['buy_price'] * order['trailing_LIT_gain_coef'] 
          df, order = place_sell_order_if_it_was_not_placed(df,
            order=order,
            sell_orders=trailing_LIT_orders,
            sell_orders_list=trailing_LIT_orders_list,
            price=price,
            order_type='trailing_LIT')
        # Checking trailing_stop_limit_order
          try:
            stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
            current_price = stock_df_1m['close'].iloc[-1]
            if current_price >= order['buy_price'] * trailing_stop_limit_act_coef or True:
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

    # 3. For optimal stock list:
    counter = 0
    market_value_2m = 0
    market_value_5m = 0
    market_value_30m = 0
    market_value_60m = 0
    blue.print(f'total market_value {total_market_value:.2f}, 2m is {total_market_value_2m:.2f}, 5m is {total_market_value_5m:.2f} \
30m is {total_market_value_30m:.2f}, 60m is {total_market_value_60m:.2f}')
    for ticker in stock_name_list_opt:
      print(f'Stock is {ticker}')
      order = []    
      # 3.1 If counter condition:  
      counter += 1
      if counter > 20:

        bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
        history_orders = ma.get_history_orders()
        
        # Get current orders and they lists:
        limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, limit_if_touched_buy_orders,\
            trailing_LIT_orders, trailing_stop_limit_orders = ma.get_orders()
        limit_buy_orders_list = get_orders_list_from_moomoo_orders(limit_buy_orders)
        limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_buy_orders)
        limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
        stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
        trailing_LIT_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_orders)
        trailing_stop_limit_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_orders)
        # Recheck for all tickers in positional list that LIMIT/trailing_LIT_order are placed if condition
        for ticker2 in positions_list:
          try:
            if ticker2 in bought_stocks_list:
              order = bought_stocks.loc[bought_stocks['ticker'] == ticker2].sort_values('buy_time').iloc[-1]   
              qty = order['stocks_number']  # stock number should be taken from the trade 
              # Checking trailing_stop_limit_order
              try:
                stock_df_1m = get_historical_df(ticker = ticker2, period='1d', interval='1m')
                current_price = stock_df_1m['close'].iloc[-1]
                warning.print(f'Recheking trailing stop limit order for {ticker2}, current price is {current_price:.2f}')
                if current_price >= order['buy_price'] * trailing_stop_limit_act_coef or True:
                  price = order['buy_price'] * default.gain_coef
                  df, order = place_sell_order_if_it_was_not_placed(df,
                    order=order,
                    sell_orders=trailing_stop_limit_orders,
                    sell_orders_list=trailing_stop_limit_orders_list,
                    price=price,
                    order_type='trailing_stop_limit') 
                # Modify trailing_stop_limit_order if price dropped below 0.016%
                if current_price < order['buy_price'] * 0.9984:
                  trailing_ratio = default.trailing_ratio
                  trail_spread = trail_spread_coef
                  order_id = ma.modify_trailing_stop_limit_order(order=order,
                                                        trail_value=trailing_ratio,
                                                        trail_spread=trail_spread)  
                  if order_id != order['trailing_stop_limit_order_id']:
                    order['trailing_stop_limit_order_id'] = order_id
                  order['trailing_ratio'] = trailing_ratio
                  df = ti.update_order(df, order)

                # Recalculate Trailing LIT gain coefficient:
                current_gain = current_price / order['buy_price']
                trailing_LIT_gain_coef = order['trailing_LIT_gain_coef']
                if current_gain >= 1.005 and current_gain < 1.0055:
                  trailing_LIT_gain_coef = 1.0065
                elif current_gain >= 1.0055 and current_gain <= 1.006:
                  trailing_LIT_gain_coef = 1.0075
                elif current_gain > 1.006:
                    trailing_LIT_gain_coef = 1.008

                if current_gain >= 1.006:
                  ma.cancel_order(order=order, order_type='trailing_LIT')
                elif order['trailing_LIT_gain_coef'] != trailing_LIT_gain_coef:
                    order_id = ma.modify_limit_if_touched_order(order, trailing_LIT_gain_coef,
                                                                aux_price_coef=aux_price_coef,
                                                                order_type='trailing_LIT' 
                    ) 
                    order['trailing_LIT_gain_coef'] = trailing_LIT_gain_coef
                    df = ti.update_order(df, order)      
              except Exception as e:
                alarm.print(e)           
            else:
              alarm.print(f'{ticker2} is in positional list but not in DB!')
          except Exception as e:
            alarm.print(e)
        
        # Recheck all placed order information
        for ticker3 in placed_stocks_list:
          try: 
            order = df.loc[(df['ticker'] == ticker3) & df['status'].isin(['placed'])]
            if type(order) == pd.Series or type(order) == pd.DataFrame:
              order = order.sort_values('buy_time').iloc[-1]
              # 1.111 was set during buy order creation
              if order['buy_commission'] == 1.111 \
                or math.isnan(order['buy_commission']) \
                or order['buy_commission'] == None \
                or order['buy_commission'] == 0:
                order = update_buy_order_based_on_platform_data(order, history_orders) # get new hirtory information
                if order['status'] == 'bought':
                  df = ti.update_order(df, order)
          except Exception as e:
            alarm.print(e)
        
        counter = 0

      # 3.2 Get historical data, current_price for stocks in optimal list
      try:
        stock_df = get_historical_df(ticker = ticker, period=period, interval=interval)
        try:
          stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m')
          current_price = stock_df_1m['close'].iloc[-1]

          current_timezone = datetime.now().astimezone().tzinfo
          time_is_correct =  (datetime.now().astimezone() - stock_df_1m.index[-1].astimezone(current_timezone)).seconds  < 60 * 3

          if time_is_correct:
            market_value += stock_df_1m['pct'].iloc[-2]
          else:
            market_value  += 0
          market_value_2m += stock_df_1m['pct'].iloc[-2:-1].sum()
          market_value_5m += stock_df_1m['pct'].iloc[-5:-1].sum()
          market_value_30m += stock_df_1m['pct'].iloc[-30:-1].sum()
          market_value_60m += stock_df_1m['pct'].iloc[-60:-1].sum()
          warning.print(f'market_value is {market_value:.2f},market_value_2m is {market_value_2m:.2f},  market_value_5m is {market_value_5m:.2f}')
          warning.print(f'market_value_30m is {market_value_30m:.2f}, market_value_60m is {market_value_60m:.2f}')

        except Exception as e:
          if not(stock_df is None):
            current_price = stock_df['close'].iloc[-1]
          alarm.print(e) 
        # TEST TRADING
        # df_test = test_trading_simulation(ticker, stock_df, df_test, bought_stocks_test_list)
      except Exception as e:
        print(f'{e}')
        stock_df = None
      
      # 3.3 DYNAMIC GAIN COEFFICIENT
      # if gain_coef\lose_coef doesn't match order's modify the order:
      if not(stock_df is None):
        try:
          if ticker in bought_stocks_list:
            order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time').iloc[-1]
        except Exception as e:
          alarm.print(e)
          
        if ticker in bought_stocks_list:
        # Trailing LIT gain coefficient:
          try:
            current_gain = current_price / order['buy_price']
            trailing_LIT_gain_coef = order['trailing_LIT_gain_coef']
            if current_gain >= 1.005 and current_gain < 1.0055:
              trailing_LIT_gain_coef = 1.0065
            elif current_gain >= 1.0055:
              trailing_LIT_gain_coef = 1.0075
            elif current_gain >= 1.006:
                trailing_LIT_gain_coef = 1.0075

            # Trailing stop limit order, trailing ratio
            if current_gain >= 1.0055 and current_gain < 1.008:
              trailing_ratio = 0.12
            elif current_gain >= 1.008 and  current_gain <= 1.009:
              trailing_ratio = 0.15
            elif current_gain > 1.009:
              trailing_ratio = 0.2
            else:
              trailing_ratio = default.trailing_ratio
          except Exception as e:
            alarm.print(e)

          try:
            # if order['gain_coef'] != gain_coef:
            #   order_id = ma.modify_limit_if_touched_order(order, gain_coef)
            #   if order_id != order['limit_if_touched_order_id']:
            #     order['limit_if_touched_order_id'] = order_id
            #   order['gain_coef'] = gain_coef
            #   df = ti.update_order(df, order)
            # if order['lose_coef'] != lose_coef:
            #   order_id = ma.modify_stop_order(order, lose_coef)
            #   if order_id != order['stop_order_id']:
            #     order['stop_order_id'] = order_id
            #   order['lose_coef'] = lose_coef
            #   df = ti.update_order(df, order)

            if current_gain >= 1.006:
              ma.cancel_order(order=order, order_type='trailing_LIT')
            elif order['trailing_LIT_gain_coef'] != trailing_LIT_gain_coef:
              order_id = ma.modify_limit_if_touched_order(order, trailing_LIT_gain_coef,
                                                          aux_price_coef=aux_price_coef,
                                                          order_type='trailing_LIT' 
              )  
              if order_id != order['trailing_LIT_order_id']:
                order['trailing_LIT_order_id'] = order_id
              order['trailing_LIT_gain_coef'] = trailing_LIT_gain_coef
              df = ti.update_order(df, order)    
          except Exception as e:
            alarm.print(e)  
          # trailing stop limit order modification condition 
          try:
            if order['trailing_ratio'] != trailing_ratio:
              trail_spread = order['buy_price'] * trail_spread_coef
              order_id = ma.modify_trailing_stop_limit_order(order=order,
                                                    trail_value=trailing_ratio,
                                                    trail_spread=trail_spread)  
              if order_id != order['trailing_stop_limit_order_id']:
                order['trailing_stop_limit_order_id'] = order_id
              order['trailing_ratio'] = trailing_ratio
              df = ti.update_order(df, order)
          except Exception as e:
            alarm.print(e)

      # 3.4 BUY SECTION:
      if not(stock_df is None):
        if not(ticker in bought_stocks_list or ticker in placed_stocks_list):
          buy_condition, buy_price = stock_buy_condition(stock_df, stock_df_1m, ticker)
          buy_price = min(stock_df['close'].iloc[-10:-1].min(), stock_df['open'].iloc[-10:-1].min())
          # if current_price / buy_price > 1.0025: buy_condition = False
          # buy_condition_1m, buy_price_1m = stock_buy_condition_1m(stock_df_1m)
          buy_condition_1m, buy_price_1m = False, False
        else:
          buy_condition = False
          buy_condition_1m = False
          buy_price_1m = 0
          buy_price = 0
        c.green_red_print(buy_condition, 'buy condition')
        c.green_red_print(buy_condition_1m, 'buy condition 1m')
        print(f'stock {ticker}, time: {stock_df.index[-1]} last price is {stock_df['close'].iloc[-1]:.2f}, pct is {stock_df['pct'].iloc[-1]:.2f}')
        print(f'''df_1m['pct'].iloc[i-5 : i].sum() is {stock_df_1m['pct'].iloc[-5 : -1].sum():.2f}''')
        current_timezone = datetime.now().astimezone().tzinfo
        time_is_correct =  (datetime.now().astimezone() - stock_df.index[-1].astimezone(current_timezone)).seconds  < 60 * 60 * 1 + 60 * 5 
        c.print(f'Time is correct condition {time_is_correct}', color='yellow')

        if ((buy_condition and buy_price != 0) or (buy_condition_1m and buy_price_1m != 0)) and time_is_correct:

          us_cash = ma.get_us_cash()
          profit_24hours = current_profit(df)
          c.print(f'Availble withdrawal cash is {us_cash}')
          c.print(f'Last 24 hours profit is {profit_24hours}')

          # Check last 24 hours profit condition and
          # Check available money for trading based on Available Funds
          security_condition = True
          security_condition_1m = True
          security_condition_1h = True
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

          if (total_market_value_2m < 0 and total_market_value_5m < -1) or total_market_value_30m < -10:
            security_condition_1m = False
            alarm.print(f'Market seems to be dropping')
          
          if total_market_value_30m < -10:
            security_condition_1h = False

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
            if buy_condition_1m:
              # buy_price = buy_price_1m
              buy_price = min(stock_df_1m['close'].iloc[-10:-1].min(), stock_df_1m['open'].iloc[-10:-1].min())
              buy_condition_type = '1m'
            else:
              # buy_price = min(stock_df['close'].iloc[-10:-1].min(), stock_df['open'].iloc[-10:-1].min())
              try:
                buy_price = stock_df_1m['close'].iloc[-2]
                # buy_price = min(stock_df_1m['close'].iloc[-3:-1].min(), stock_df_1m['open'].iloc[-3:-1].min())
              except:
                buy_price = stock_df['close'].iloc[-1]
              buy_condition_type = '1h'

            if (security_condition_1m and buy_condition_1m) \
              or (security_condition_1h and buy_condition):
              # play sound:
              winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
              order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_condition_type=buy_condition_type, buy_sum=buy_sum)
              order['gain_coef'] = default.gain_coef
              if buy_condition_type == '1h':
                order['lose_coef'] = lose_coef_1h
              else:
                order['lose_coef'] = lose_coef_1m
              order['tech_indicators'] = f'mv :{market_value:.2f}, mv_2m:{market_value_2m:.2f},\
  mv_5m : {market_value_5m:.2f}, mv_30m : {market_value_30m:.2f}, mv_60m: {market_value_60m:.2f}'
              if order['status'] == 'placed':
                order = update_buy_order_based_on_platform_data(order)
                df = ti.record_order(df, order)
                if place_trailing_stop_order_imidiately:
                  price = order['buy_price'] * default.gain_coef
                  df, order = place_sell_order_if_it_was_not_placed(df,
                  order=order,
                  sell_orders=trailing_stop_limit_orders,
                  sell_orders_list=trailing_stop_limit_orders_list,
                  price=price,
                  order_type='trailing_stop_limit') 

      # 3.5 Recheck placed orders information including commission from the order history
      if ticker in placed_stocks_list:
        try: 
          order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])]
          if type(order) == pd.Series or type(order) == pd.DataFrame:
            order = order.sort_values('buy_time').iloc[-1]
            # 1.111 was set during buy order creation
            if order['buy_commission'] == 1.111 \
              or math.isnan(order['buy_commission']) \
              or order['buy_commission'] == None \
              or order['buy_commission'] == 0:
              order = update_buy_order_based_on_platform_data(order) # get new hirtory information
              if order['status'] == 'bought':
                df = ti.update_order(df, order)
        except Exception as e:
          alarm.print(e)

      # 3.6 CHECKING FOR CANCELATION OF THE BUY ORDER
      # Check current price and status of buy order
      # If price more or equal than price*gain_coef CANCEL the BUY ORDER
      # If status was not FILLED_PART, change status of order to 'cancel', update the order
      if not(stock_df is None):
        if ticker in placed_stocks_list:
          try:
            order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])].sort_values('buy_time').iloc[-1]
            # stock_df_1m = get_historical_df(ticker = ticker, period='1d', interval='1m')
            # current_price = stock_df_1m['close'].iloc[-1]
          except Exception as e:
            print(f'Minute data for stock {ticker} has not been received')
            # current_price = stock_df['close'].iloc[-1]
          try:
            if current_price >= order['buy_price'] * order['gain_coef'] \
              or (datetime.now() - order['buy_time']).seconds / 60 > order_1h_life_time_min \
              or ((datetime.now() - order['buy_time']).seconds / 60 >= order_1m_life_time_min \
                   and order['buy_condition_type'] == '1m'):         
              if limit_buy_orders.shape[0] > 0:
                limit_buy_order = limit_buy_orders.loc[
                  limit_buy_orders['order_id'] == order['buy_order_id']]
                if not limit_buy_order.empty:
                  if limit_buy_order['order_status'].values[0] != ft.OrderStatus.FILLED_PART:
                    try:
                      # cancel limit_if_touched_sell_order and stop_order if they was placed
                      if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                          ma.cancel_order(order, order_type='limit_if_touched')    
                      if order['stop_order_id'] not in ['', None, []]:                  
                          ma.cancel_order(order, order_type='stop')
                      if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
                          ma.cancel_order(order, order_type='trailing_stop_limit')
                    except Exception as e:
                      alarm.print(e)
                  else:
                    order['status'] = 'filled part'
                # cancel buy limit order
                ma.cancel_order(order, order_type='buy')
                order['status'] = 'cancelled'
                df = ti.update_order(df, order)
          except Exception as e:
            print(e)
      
      # 3.7 Checking if sell orders have been executed
      try:
        if ticker in bought_stocks_list:
          order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time').iloc[-1]
          historical_limit_if_touched_order = historical_orders.loc[
            historical_orders['order_id'] == order['limit_if_touched_order_id']]
          historical_stop_order = historical_orders.loc[
            historical_orders['order_id'] == order['stop_order_id']]
          historical_trailing_LIT_order = historical_orders.loc[
            historical_orders['order_id'] == order['trailing_LIT_order_id']]
          historical_trailing_stop_limit_order = historical_orders.loc[
            historical_orders['order_id'] == order['trailing_stop_limit_order_id']]
          # checking and update limit if touched order
          if historical_limit_if_touched_order.shape[0] > 0 \
            and  historical_limit_if_touched_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
            and order['status'] in ['bought', 'filled part']:
            sell_price = order['buy_price'] * order['gain_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_limit_if_touched_order)
            df = ti.update_order(df, order)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            # cancel stop order
            ma.cancel_order(order, order_type='stop')
            ma.cancel_order(order, order_type='trailing_LIT')
            ma.cancel_order(order, order_type='trailing_stop_limit')
        # checking and update stop order
          if historical_stop_order.shape[0] > 0 \
          and historical_stop_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
          and order['status'] in ['bought', 'filled part']:
            sell_price = order['buy_price'] * order['lose_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_stop_order)
            df = ti.update_order(df, order)
            # cancel limit-if-touched order
            ma.cancel_order(order, order_type='limit_if_touched')
            ma.cancel_order(order, order_type='trailing_LIT')
            ma.cancel_order(order, order_type='trailing_stop_limit')
        # checking and update trailing LIT order
          if historical_trailing_LIT_order.shape[0] > 0 \
            and  historical_trailing_LIT_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
            and order['status'] in ['bought', 'filled part']:
            sell_price = order['buy_price'] * order['trailing_LIT_gain_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_trailing_LIT_order)
            df = ti.update_order(df, order)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            # cancel stop order and limit if touched 
            ma.cancel_order(order, order_type='stop')
            ma.cancel_order(order, order_type='limit_if_touched')
            ma.cancel_order(order, order_type='trailing_stop_limit')
        # checking and update trailing stop limit order
          if historical_trailing_stop_limit_order.shape[0] > 0 \
            and  historical_trailing_stop_limit_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
            and order['status'] in ['bought', 'filled part']:
            sell_price = order['buy_price'] * order['trailing_LIT_gain_coef']
            order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_trailing_stop_limit_order)
            df = ti.update_order(df, order)
            # play sound:
            winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
            # cancel stop order and limit if touched 
            ma.cancel_order(order, order_type='stop')
            ma.cancel_order(order, order_type='limit_if_touched')
            ma.cancel_order(order, order_type='trailing_LIT')
      except Exception as e:
        alarm.print(e)

    total_market_value = market_value
    total_market_value_2m = market_value_2m
    total_market_value_5m = market_value_5m
    total_market_value_30m = market_value_30m 
    total_market_value_60m = market_value_60m

    # Update SQL DB FROM df each full cycle!!!
    # try:
    #   db.update_db_from_df(df)
    # except Exception as e:
    #   alarm.print(e)
    print('Waiting progress:')
    for i in tqdm(range(20)):
      time.sleep(1)
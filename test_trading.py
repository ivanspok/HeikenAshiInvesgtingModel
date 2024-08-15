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

import warnings
warnings.filterwarnings('error')

from colog.colog import colog
c = colog()
warning = colog(TextColor='orange')
alarm = colog(TextColor='red')

#INIT
ti = TradeInterface(platform='test', df_name='test')

# Moomoo settings
ip = '127.0.0.1'
port = 11111
unlock_pwd = '771991'
trd_env = ft.TrdEnv.SIMULATE
# ft.SysConfig.set_all_thread_daemon(True)

# SETTINGS
stock_name_list_opt = [
  'GOOG', 'JPM', 'XOM', 'UNH', 'AVGO', 'LLY', 'COST',
  'CRM', 'TMO', 'NFLX', 'TXN', 'INTU', 'NKE', 'QCOM',
  'BA', 'AMGN', 'MDT', 'PLD', 'MS', 'GS', 'LMT', 'BKNG',
  'ADI', 'TJX', 'ELV', 'C', 'CVS', 'VRTX', 'SCHW', 'LRCX',
  'TMUS', 'ETN', 'ZTS', 'CI', 'FI', 'EQIX', 'DUK', 'MU',
  'AON', 'ITW', 'SNPS', 'KLAC', 'CL', 'WM', 'HCA', 'MMM',
  'CMG', 'EW', 'GM', 'MCK', 'NSC', 'PH', 'MPC', 'ROP', 
  'MCHP', 'USB', 'CCI', 'MAR', 'MSI', 'GD', 'JCI', 'PSX', 
  'SRE', 'ADSK', 'AJG', 'TEL', 'TT', 'PCAR', 'OXY', 'CARR',
  'IDXX', 'GIS', 'CTAS', 'AIG', 'ANET', 'BIIB', 'SPG', 'MSCI', 'DHI'
]

# stock_name_list_opt = [
#   'GOOG', 'AAPL'
# ]

period = '3mo'
interval = '1h' 

is_near_global_max_prt = 80
distance_from_last_top  = 0
last_top_ratio = 1
RIV  = 0.25
buy_ratio_border = 9
bull_trend_coef = 1.12
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

if __name__ == '__main__':

  df = ti.load_trade_history() # load previous history

  while True:
    if df.shape[0] > 0:
      bought_stocks_list = df.loc[df['status'] == 'bought'].ticker.to_list()
    else:
      bought_stocks_list = []

    for ticker in stock_name_list_opt:
      print(f'Stock is {ticker}')
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
        # if buy_condition and buy_price != 0:
          order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_sum=1000.0)
          order['gain_coef'] = gain_coef
          order['lose_coef'] = lose_coef
          df = ti.record_order(order)
          # Place moomoo simulation buy order
          try:
            trd_ctx  =  ft.OpenUSTradeContext(ip, port)
            stock_code = 'US.' + ticker
            ret, data = trd_ctx.place_order(
                                      price=1,
                                      qty=1,
                                      code=stock_code,
                                      trd_side=ft.TrdSide.BUY,
                                      trd_env=trd_env,
                                      order_type=ft.OrderType.MARKET)
            trd_ctx.close()
          except Exception as e:
            print(e)

        # Checking for sell condition
        if ticker in bought_stocks_list:
          order = df.loc[(df['ticker'] == ticker) & (df['status'] == 'bought')].sort_values('buy_time').iloc[-1]
          order_book_is_available = False

          try:
            stock_df = get_historical_df(ticker = ticker, period='1d', interval='1m')
          except Exception as e:
            print(f'Minute data for stock {ticker} has not been received')
      
          current_stock = stock_df.sort_index().iloc[-1]
          if order_book_is_available:
            # get current price from order book
            current_price = 0
          else:
            current_price = current_stock['close']
          sell_condition = sell_stock_condition(order, current_price)
          c.print(f'sell condition is {sell_condition}', color='purple')

          # sell_condition = True #!!!
          if sell_condition:
            # !!! need get data from trading platform 
            order = ti.sell_order(order, current_price)
            # order['status'] = 'test' # !!!!!
            df = ti.update_order(order)
            # Place Moomoo simulation sell order
            try:
              trd_ctx  =  ft.OpenUSTradeContext(ip, port)
              stock_code = 'US.' + ticker
              ret, data = trd_ctx.place_order(
                                        price=1,
                                        qty=1,
                                        code=stock_code,
                                        trd_side=ft.TrdSide.SELL,
                                        trd_env=trd_env,
                                        order_type=ft.OrderType.MARKET)
              trd_ctx.close()
            except Exception as e:
              print(e)

    print('Waiting progress:')
    for i in tqdm(range(180)):
      time.sleep(1)



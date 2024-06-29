import yfinance as yf
# import matplotlib

from datetime import datetime, timedelta, tzinfo
import pandas as pd
pd.options.mode.chained_assignment = None 

from personal_settings import personal_settings as ps
from datetime import date, timedelta
import os, pathlib
import numpy as np
import pickle
import functions_2 as f2
from tradingInterface import TradeInterface

import warnings
warnings.filterwarnings('error')

from colog.colog import colog
c = colog()

#INIT
ti = TradeInterface(platform='test', df_name='test')

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
  
  return condition, buy_price

if __name__ == '__main__':

  for ticker in stock_name_list_opt:
    print(f'Stock is {ticker}')
    stock_df = get_historical_df(ticker = ticker, period=period, interval=interval)
    condition, buy_price = stock_buy_condition(stock_df, ticker)
    c.print(f'condition is {condition}', color='blue')
    
    time_is_correct =  (datetime.now().astimezone() - stock_df.index[-1]).seconds / 60  < 5 * 60
    c.print(f'Time is correct condition {time_is_correct}', color='yellow')

    if condition and buy_price != 0 and time_is_correct:
      order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_sum=1000)

    # print(df_stocks_dict[stock].shape)


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
import traceback
from typing import Union, Tuple

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
tzinfo_ny = pytz.timezone('America/New_York')
#%% SETTINGS

# Trade settings 
default_buy_sum = 1000 # 3300 # in USD
min_buy_sum = 700 # 2000 # in USD
max_buy_sum = 1100 # 3300 # in  USD
stop_trading_profit_value = -200 * rate # in AUD * rate = USD
max_stock_price = 1360 # 1050 # in  USD

order_1m_life_time_min = 1
order_1h_life_time_min = 1
order_before_market_open_life_time_min = 60 #720 #1440
order_MA50_MA5_life_time_min = 15
place_trailing_stop_limit_order_imidiately = True

# Moomoo settings
moomoo_ps = ps.Moomoo()
ip = '127.0.0.1'
port = 11111
unlock_pwd = moomoo_ps.unlock_pwd
ACC_ID = moomoo_ps.acc_id
TRD_ENV = ft.TrdEnv.REAL
MARKET = 'US.'
# ft.SysConfig.set_all_thread_daemon(True)

# Version 7.0 (speed_norm100 list)
stock_name_list_opt = ['HD', 'C', 'SBUX', 'DE', 'CRM', 'SHW', 'CHTR', 'FTNT', 'IBM', 'PNC', 'ORCL', 'SPG',
                        'BA', 'CB', 'ANET', 'APD', 'ON', 'MRNA', 'WFC', 'COF', 'AMAT', 'MS', 'DHI', 'LOW', 
                        'QCOM', 'ADI', 'ETN', 'MCO', 'NSC', 'CMCSA', 'PANW', 'CAT', 'IDXX', 'INTC', 'ACN',
                          'ADBE', 'ADSK', 'APH', 'WMB', 'CSX', 'RTX', 'AIG', 'PCAR', 'XOM', 'TJX', 'CSCO',
                            'CTAS', 'JPM', 'NOW', 'NFLX', 'MRK', 'SYK', 'GS', 'TDG', 'GOOG', 'CMG', 'HCA', 
                            'BLK', 'FDX', 'PM', 'VZ', 'BSX', 'GE', 'AON', 'WELL', 'ABT', 'ABBV', 'PH', 'PG', 
                            'ICE', 'TT', 'ECL', 'HLT', 'ISRG', 'JNJ', 'LIN', 'MCD', 'V', 'DHR', 'COST', 'MA',
                              'INTU', 'LMT', 'REGN', 'ROK', 'BAC', 'GM', 'PSX', 'UPS', 'MPC', 'CARR', 'AVGO',
                             'NKE', 'AMD', 'F', 'LRCX', 'KLAC']
# Version 8.0 with exlusion
stock_name_list =[]
stock_name_list += ['GOOG','JPM','XOM','UNH','JNJ','V','AVGO','PG','LLY','MA','HD','CVX','MRK', 
                       'PEP','COST','ABBV','ADBE','KO','CRM','WMT','MCD','CSCO','BAC','PFE','TMO','ACN','NFLX','ABT','AMD','LIN','ORCL','CMCSA',
                       'TXN','DIS','WFC','DHR','PM','NEE','VZ','INTC','RTX','HON','LOW','UPS','INTU','SPGI','NKE','COP','QCOM','BMY','CAT','UNP','BA','ISRG',
                        'GE','IBM','AMGN','AMAT','MDT','SBUX','PLD','NOW','MS','DE','BLK','GS','T','LMT','AXP','SYK','ADI','TJX','ELV','MDLZ','GILD','ADP','MMC',
                        'C','AMT','CVS','VRTX','SCHW','LRCX','MO','TMUS','SLB', 'ETN', 'ZTS', 'CI', 'PYPL']

stock_name_list += ['FI','CB','SO','REGN','BSX','EQIX','BDX','PANW','DUK','EOG','MU','AON','ITW','CSX','SNPS','PGR','APD','KLAC','CME','NOC','CDNS','ICE',
                       'CL','SHW','WM','HCA','TGT','FCX','FDX','F','MMM','CMG','EW','GM','MCK','NXPI','MCO','NSC','HUM','EMR','DXCM','PNC','PH','MPC','APH',
                       'ROP','FTNT','MCHP','USB','CCI','MAR','MSI','GD','PSA','JCI','PSX','SRE','ADSK','AZO','TDG','ECL','AJG','KMB','TEL','TT','AEP','EL','PCAR',
                       'OXY','TFC','CARR','D','IDXX','GIS','ON','COF','ADM','MNST','NUE','CTAS','AIG','EXC','VLO','MRNA','ANET','WMB','O','STZ','IQV','HLT','CHTR','WELL',
                       'BIIB','SPG','MSCI','DHI','ROK']
stock_name_list_opt = stock_name_list

# stock_name_list_opt = ['INTC']
exclude_time_dist = {}

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
    self.gain_coef_speed_norm100 = 1.05
    self.gain_coef_before_market_open = 1.05
    self.gain_coef_MA50_MA5 = 1.07
    self.lose_coef_before_market_open = 1.005
    self.lose_coef_1_MA50_MA5 = 0.98
    self.lose_coef_2_MA50_MA5 = 0.97
    self.lose_coef_stopmarket_MA50_MA5 = 0.978
    self.trailing_ratio = 0.45
    self.trailing_ratio_MA50_MA5 = 0.99
default = Default()

#%% FUNCTIONS

def timedelta_minutes(order):
  delta = (datetime.now() - order['buy_time']) 
  deltatime_minutes = delta.seconds / 60 + delta.days * 24 * 60
  return deltatime_minutes
  
def get_historical_df(ticker='', interval='1h', period='2y', start_date=date.today(), end_date=date.today(), prepost=False) -> pd.DataFrame:
    df = pd.DataFrame()
    try:
      insturument = yf.Ticker(ticker)
      df = insturument.history(period=period, interval=interval, prepost=prepost)

      if not df.empty:
        df = df.rename(columns={"Open": "open", "Close" :'close', "High" : 'high', "Low": 'low'})

        df['pct'] = np.where(df['open'] < df['close'],  
                            (df['close'] / df['open'] - 1) * 100,
                            -(df['open'] / df['close'] - 1) * 100
        )
        # df.index = pd.to_datetime(df['t'], unit='s', utc=True).map(lambda x: x.tz_convert('America/New_York'))
        df = f2.get_heiken_ashi_v2(df)
        df = df[['open', 'high', 'low', 'close', 'pct', 'ha_pct', 'ha_colour']]
        df = df.sort_index(ascending=True)
        df = MA(df, k=5) # add MA5 column to the df
        df = MA(df, k=7) # add MA7 column to the df
        df = MA(df, k=20) # add MA20 column to the df
        df = MA(df, k=50) # add MA50 column to the df
    except Exception as e:
      alarm.print(traceback.format_exc())
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
    alarm.print(traceback.format_exc())
  return result

def number_red_candles(df, i, k=11):

  if i < k:
    number_red_candles = (df['ha_colour'][0 : i] == 'red').sum()
  else:
    number_red_candles = (df['ha_colour'][i - k : i] == 'red').sum()
  return number_red_candles
if False:
  def stock_buy_conditions(df, ticker):
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

def minmax(value, min, max):
  '''
  Limit the value beetween min and max values
  '''
  return np.maximum(np.minimum(value, max), min)

def MA(df, k=20):
  df[f'MA{k}'] = df['close'].rolling(window=k).mean()
  return df

def maximum(df, i, k=20):
  '''
  Return maximum price from close/open values, with k window, including last value
  '''
  if i < k:
    return np.amax([df['close'].iloc[0 : k].max(), df['open'].iloc[0 :k].max(), df['close'].iloc[i], df['open'].iloc[i]])
  else: 
    return np.amax([df['close'].iloc[i -k : i].max(), df['open'].iloc[i - k : i].max(), df['close'].iloc[i], df['open'].iloc[i]])

def mimimum(df, i, k=20):  
  '''
  Return minumum price from close/open values, with k window, including last value
  '''
  if i < k:
    return np.amin([df['close'].iloc[0 : k].min(), df['open'].iloc[0 : k].min(), df['close'].iloc[i], df['open'].iloc[i]])
  else:
    return np.amin([df['close'].iloc[i - k : i].min(), df['open'].iloc[i - k : i].min(), df['close'].iloc[i], df['open'].iloc[i]])

def norm(df, i, k=20):
  min = mimimum(df,i,k)
  max = maximum(df,i,k)
  return (df['close'].iloc[-1] - min) / (max - min + 0.0001)

def get_amps(df, p=7):
  min = []
  max = []
  amp = []
  amps_avg = []
  
  for i in range(df.shape[0]):
    if i < p:
      min.append(mimimum(df, i, p))
      max.append(maximum(df, i, p))
    else:
      min.append(mimimum(df, i, p))
      max.append(maximum(df, i, p))
    amp.append(max[i] / min[i])
    if i < 300:
      amps_avg.append(sum(amp[:i+1]) / len(amp[:i+1]))
    else:
      amps_avg.append(sum(amp[i - 300 : i + 1]) / len(amp[i - 300 : i + 1]))
  return amps_avg

def get_speed(df, p):
  speed100 = []
  for i in range(df.shape[0]):
    if i < p:
      speed100.append(df['close'].iloc[i] / df['close'].iloc[0])
    else:
      speed100.append(df['close'].iloc[i] / df['close'].iloc[i - p])
  return speed100

def stock_buy_conditions(df, df_1m, ticker):
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

  cond_1 = df['pct'].iloc[-1] > 0.2
  cond_2 = df['close'].iloc[-1] > df['close'].iloc[-2]
  cond_3 = df_1m['close'].iloc[-40 : -1].max() / df_1m['close'].iloc[-1] > 1.01
  cond_4 = df_1m['close'].iloc[-25 : -1].max() / df_1m['close'].iloc[-1] > 1.007
  cond_5 = df_1m['close'].iloc[-15 : -1].max() / df_1m['close'].iloc[-1] > 1.003

  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and  df_1m['ha_colour'].iloc[-3] == 'red' and df_1m['ha_colour'].iloc[-4] == 'red'
  

  cond_6 = df['pct'].iloc[-2] > 0.25
  cond_7 = df['pct'].iloc[-1] > 0.05 

  # c.green_red_print(ha_cond, 'ha_cond')
  # c.green_red_print(cond_3, 'cond_3')
  # c.green_red_print(cond_4, 'cond_4')
  # c.green_red_print(cond_5, 'cond_5')
  if df.index[-1].hour == 9:
    # c.green_red_print(cond_1, 'cond_1')
    # c.green_red_print(cond_2, 'cond_2')
    if cond_1 and cond_2 and ha_cond \
      and (cond_3 or cond_4 or cond_5):
      condition = True      
      condition_type = 'drowdown930'
      c.green_red_print(condition, 'buy condition drowdown930')
  else:
    c.green_red_print(cond_6, 'cond_6')
    c.green_red_print(cond_7, 'cond_7')
    if cond_6 and cond_7 and ha_cond \
      and (cond_3 or cond_4 or cond_5):
      condition = True
      condition_type = 'drowdown'
      c.green_red_print(condition, 'buy condition drowdown')

  c2 = False # bad condition
  c32 = False # bad condition
  # 930 and 1030 red candles put conditon with trying to get 0.5%? time beetween 11-30 and 11-40
  if df.index[-1].hour == 11 and df.index[-1].minute == 30:
    # -1 : 11:30,  -2 : 10:30, -3 : 9:30
    CM930 = df['close'].iloc[-3] 
    CM1030 = (df['close'].iloc[-2] + df['open'].iloc[-2]) / 2
    CM1130 = df['close'].iloc[-1]

    # 930 candle is red; and 1030 candle between -1% and 0.2% ?
    c2 = more_than_pct(CM1130, CM930, 0.21) and more_than_pct(CM1130, CM1030, 0.1) \
      and df['open'].iloc[-3] / df['close'].iloc[-3]  > 1.01 \
      and df['close'].iloc[-2] / df['close'].iloc[-1] < 1.002 \
      and df['close'].iloc[-2] / df['close'].iloc[-1] > 0.99 \
      and ha_cond and False
    c.green_red_print(more_than_pct(CM1130, CM1030, 0.01), 'more_than_pct(CM1130, CM1030, 0.01)')
    c.green_red_print(more_than_pct(CM930, CM1030, 0.01), 'more_than_pct(CM930, CM1030, 0.01)')
    c.green_red_print(more_than_pct(CM1130, CM930, 0.02), 'more_than_pct(CM1130, CM930, 0.02)')
    # and more_than_pct(CM930, CM1030, 0.01)
    c32 = more_than_pct(CM1130, CM1030, 0.01) and more_than_pct(CM1130, CM930, 0.02) \
      and more_than_pct(CM1030, CM930, 0.01)\
      and df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
      and df_1m.index[-1].minute < 42 and False
    
    c.print(f'CM930 is {CM930:.2f}, CM1030 is {CM1030:.2f}, CM1130 is {CM1130:.2f}', color='blue')
    c.green_red_print(ha_cond, 'ha_cond')

  if c2:
    condition = True
    condition_type = '1130_1'

  if c32:
    condition = True
    condition_type = '1130_2'

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
    condition_type = '1230_2'
  
  if False:
    if (df_1m.index[-1].hour == 12 and df_1m.index[-1].minute > 50) \
      or (df_1m.index[-1].hour == 13 and df_1m.index[-1].minute < 20):  
      
      max = np.maximum(df_1m['close'].iloc[i_1m - 30 : i_1m].max(), df_1m['close'].iloc[-1])

      cond_8 = df_1m['close'].iloc[-1] / df_1m['close'].iloc[-25] > 1.001
      cond_9 = df_1m['pct'].iloc[-25:-1].sum() > 0
      c.green_red_print(cond_8, 'cond_8')
      c.green_red_print(cond_9, 'cond_9')

      if cond_8 and cond_9 and ha_cond:
        condition = True
        condition_type = '1230_1330_inertia'

  if (df_1m.index[-1].hour == 12 and df_1m.index[-1].minute >= 20):
    
    min = mimimum(df_1m, 220)
    max = maximum(df_1m, 220)

    cond_8 = df_1m['close'].iloc[-1] / min > 1.0025
    cond_9 = (df_1m['close'].iloc[-1] - min) / (max - min + 0.0001) < 0.5
   
    c.green_red_print(cond_8, 'cond_8')
    c.green_red_print(cond_9, 'cond_9')
    c.print(f'''df_1m['close'].iloc[-1] / min is {df_1m['close'].iloc[-1] / min:.4f}''', color='yellow')

    if cond_8 and cond_9 and ha_cond:
      condition = True
      condition_type = '1220_1300_reverse_point'

  return condition, condition_type  

def stock_buy_condition_maxminnorm(df, df_1m, df_stats, display=False):

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False
  
  cv = df['close'].iloc[-2]
  min30  = mimimum(df, i, 30)
  min100  = mimimum(df, i, 100)
  max30  = maximum(df, i, 30)
  max100  = maximum(df, i, 100)
  # current_amp30 = max30 / min30
  current_amp100 = max100 / min100

  speed30 = df['close'].iloc[-2] / df['close'].iloc[-30]
  speed100 = df['close'].iloc[-2] / df['close'].iloc[-100]

  # norm30 = (current_amp30 - 1) / (df_stats[ticker]['amps_avg30'].iloc[i - 1] - 1)
  norm100 = (current_amp100 - 1) / (df_stats[ticker]['amps_avg100'].iloc[-1] - 1)

  cond_value_3 = max30 / cv 

  cond_1 = speed30 > 1 and speed100 < 1
  cond_2 = norm100 >= 1.7 and norm100 <= 2
  cond_3 = cond_value_3 <= 1.03
  
  
  if display:
    # warning.print(f'speed30: {speed30:.3f}, speed100: {speed100:.3f}, max30/cv: {max30/cv:.3f}, norm100: {norm100:.3f}')
    warning.print('maxminnorm parameters:')
    c.green_red_print(cond_1, f'cond_1, speed30: {speed30:.2f}, speed100: {speed100:.2f}')
    c.green_red_print(cond_2, f'cond_2, norm100: {norm100:.2f}')
    c.green_red_print(cond_3, f'max30/cv {cond_value_3:.2f}')
  
  if ha_cond \
    and cond_1 \
    and cond_2 \
    and cond_3 \
    and not (df.index[-1].hour in [15, 16]):

      condition = True
  if condition:
    c.green_red_print(condition, 'buy_condition_maxminnorm')
  return condition

def stats_calculation(stock_name_list):

  df_stats = {}
  current_hour = datetime.now().hour
  current_date = str(datetime.now().today()).split(' ')[0]
  file_name = f'stats/df_stats_{current_date}_{current_hour}H.plk'
  try:
    if not os.path.isfile(file_name):
      print('Statistic calculation')
      for ticker in tqdm(stock_name_list):
        # file_name = f'df_stock_{ticker}_period2y_interval1h.pkl'
        # path = os.path.join(parent_path, folder_saved_models, file_name)

        df = get_historical_df(ticker = ticker, period=period, interval=interval)
        if not df.empty:
          speed30 = get_speed(df, 30)
          speed100 = get_speed(df, 100)
          amps_avg30 = get_amps(df, 30)
          amps_avg100 = get_amps(df, 100)
          data =  {'amps_avg30': amps_avg30,
                  'amps_avg100': amps_avg100,
                  'speed30': speed30,
                  'speed100': speed100}
          # if not df.empty:
          df_stats[ticker] = pd.DataFrame(data, index=df.index)
          df_stats[ticker]['max100'] = df['close'].rolling(window=100).max()
          df_stats[ticker]['max100'] = df_stats[ticker]['max100'].bfill()
          df_stats[ticker]['min100'] = df['close'].rolling(window=100).min()
          df_stats[ticker]['min100'] = df_stats[ticker]['min100'].bfill()
          df_stats[ticker]['current_amp100'] = df_stats[ticker]['max100'] / df_stats[ticker]['min100']
          df_stats[ticker]['speed100_avg30'] = df_stats[ticker]['speed100'].rolling(window=30).mean()
          df_stats[ticker]['speed30_avg30'] = df_stats[ticker]['speed30'].rolling(window=30).mean()
          df_stats[ticker]['speed30_avg30'] = df_stats[ticker]['speed30_avg30'].bfill()
          df_stats[ticker]['speed100_avg30'] = df_stats[ticker]['speed100_avg30'].bfill()
          df_stats[ticker]['norm100'] = 0.0001
          index = df_stats[ticker]['amps_avg100'].index[-2]
          df_stats[ticker].loc[index, 'norm100'] = (df_stats[ticker]['current_amp100'].iloc[-2]- 1) / (df_stats[ticker]['amps_avg100'].iloc[- 301 : -2].mean() - 1)
          index = df_stats[ticker]['amps_avg100'].index[-1]
          df_stats[ticker].loc[index, 'norm100'] = (df_stats[ticker]['current_amp100'].iloc[-1]- 1) / (df_stats[ticker]['amps_avg100'].iloc[- 300 : -1].mean() - 1)

      with open(file_name, 'wb') as file:
        pickle.dump(df_stats, file)
    else:
      with open(file_name, 'rb') as file:
        df_stats = pickle.load(file)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df_stats

def stock_buy_condition_930_47(df):
  '''
    Parameters: df, interval 1m

    Returns:
      condition
  '''
  condition = False
  buy_price = 0

  current_minute = datetime.now().astimezone().minute 
  market_opening = datetime.now().astimezone().hour == 23 \
    and current_minute > 30 and current_minute <= 47

  i = df.shape[0] - 1

  max = maximum(df, 20)
  min = mimimum(df, 20)
  cond_1 = max / df['close'].iloc[-1] > 1.0085
  cond_2 = (df['ha_colour'].iloc[-4 : -2] == 'red').all()
  cond_3 = df['ha_colour'].iloc[-2] == 'green'
  cond_4 = df['ha_colour'].iloc[-1] == 'green'
  cond_5 = (df['close'].iloc[-20 : -1].min() - min) / (max - min + 0.0001) < 0.4

  if market_opening:
    warning.print('930_47 conditions parameters:')
    c.green_red_print(market_opening, 'market_opening')
    c.green_red_print(cond_1, 'cond_1')
    c.green_red_print(cond_2, 'cond_2')
    c.green_red_print(cond_3, 'cond_3')
    c.green_red_print(cond_4, 'cond_4')
    c.green_red_print(cond_5, 'cond_5')

  if market_opening \
    and cond_1 and cond_2 and cond_3 and cond_4 and cond_5:
    condition = True

  c.green_red_print(condition, 'buy_condition_930_47')
  return condition
# version 1 1230 conditon:
if False:
  def stock_buy_condition_1230(df, df_1m):
    i_1m = df_1m.shape[0] - 1
    ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
      and number_red_candles(df_1m, i_1m, k=11) > 5

    condition = False
    if df.index[-1].hour == 13 and df.index[-2].hour == 12:
      # moomoo time: -1: 14:30; -2: 13:30: -3: 12:30; -4: 11:30; -5: 10:30
      # yahho time: 
      # candle is green
      # candle pct more than 0 
      # more taht close 9:30
      # sum pct 10:30 - 12:30  more than 0 
      # 9:30 less than -0l77
      # improve ha_cond for this case!!!!!!!! 
      if ha_cond \
        and df['close'].iloc[-2] > df['open'].iloc[-2] \
        and df['pct'].iloc[-2] > 0 \
        and df['close'].iloc[-2] > df['close'].iloc[-5] \
        and df['pct'].iloc[-4:-1].sum() > 0\
        and (df['pct'].iloc[-5] < -0.58 \
            or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006):
        condition = True
    c.green_red_print(condition, 'buy_condition_1230')
    return condition

def stock_buy_condition_1230(df, df_1m, display=False):
  i_1m = df_1m.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False

  safe_cond_1 = df_1m['close'].iloc[-1] / df['low'].iloc[-2] < 1.005 # the price hasn't risen more that 0.3%

  if df.index[-1].hour == 12 and df.index[-2].hour == 11:
    # dropping during opening/closing and assuming rising after 12:30
    # moomoo time: -1: 13:30; -2: 12:30: -3: 11:30; -4: 10:30; -5: 9:30
    # yahoo time: -1: 12:30; -2: 11:30; -3: 10:30; -4: 9:30; -5: 15:30
    # previuos candle is green
    # previous candle pct more than 0 
    # previous candle more than close 10:30
    # candle pct more than 0.15
    # 9:30 or 15:30 less than -0.58 
    cond_1 = df['close'].iloc[-2] > df['open'].iloc[-2]
    cond_2 = df['pct'].iloc[-2] > 0
    cond_3 = df['close'].iloc[-2] > df['close'].iloc[-3]
    cond_4 = df['pct'].iloc[-1] > 0.15
    cond_5 = (df['pct'].iloc[-4] < -0.58 or df['close'].iloc[-5] / df['close'].iloc[-4] > 1.006)
    cond_6 = (df['pct'].iloc[-5] < -0.58 or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006)

    if display:
      warning.print('1230 conditions parameters:')
      c.green_red_print(cond_1, 'cond_1')
      c.green_red_print(cond_2, 'cond_2')
      c.green_red_print(cond_3, 'cond_3')
      c.green_red_print(cond_4, 'cond_4')
      c.green_red_print(cond_5, 'cond_5')
      c.green_red_print(cond_6, 'cond_6')

    if ha_cond \
      and safe_cond_1 \
      and df['close'].iloc[-2] > df['open'].iloc[-2] \
      and df['pct'].iloc[-2] > 0 \
      and df['close'].iloc[-2] > df['close'].iloc[-3] \
      and df['pct'].iloc[-1] > 0.15 \
      and (cond_5 or cond_6):
      condition = True

  if df.index[-1].hour == 13 and df.index[-2].hour == 12:
    # dropping during opening/closing and assuming rising after 13:30
    # moomoo time: -1: 13:30; -2: 12:30: -3: 11:30; -4: 10:30; -5: 9:30
    # yahoo time: -1: 12:30; -2: 11:30; -3: 10:30; -4: 9:30; -5: 15:30
    # previuos candle is green
    # previous candle pct more than 0 
    # previous candle more than close 9:30
    # candle sum pct 9:30 - 11:30 more than 0
    # 9:30 or 15:30 less than -0.58 
    if ha_cond \
      and safe_cond_1 \
      and df['close'].iloc[-2] > df['open'].iloc[-2] \
      and df['pct'].iloc[-2] > 0 \
      and df['close'].iloc[-2] > df['close'].iloc[-5] \
      and df['pct'].iloc[-4:-1].sum() > 0\
      and (df['pct'].iloc[-5] < -0.58 \
          or df['close'].iloc[-6] / df['close'].iloc[-5] > 1.006):
      condition = True
  if condition:
    c.green_red_print(condition, 'buy_condition_1230')
  return condition

def stock_buy_condition_speed_norm100(df, df_1m, df_stats, display=False):
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False

  cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
  cond_value_2 = df['close'].iloc[-2] - df['close'].iloc[-50 : -1].min()
  cond_value_3 = df_stats[ticker]['norm100'].iloc[-1]
  cond_value_4 = maximum(df, i, 500) / df['close'].iloc[-2]
  cond_value_5 = maximum(df, i, 7) / df['close'].iloc[-2]

  cond_1 = cond_value_1 >= 0.7 and cond_value_1 <= 1.1
  cond_2 = cond_value_2 <= 0   
  cond_3 = cond_value_3 >= 0.7
  cond_4 = cond_value_4 > 1.2
  cond_5 = cond_value_5 < 1.035
  
  if display:
    warning.print('speed_norm100 conditions parameters:')
    c.green_red_print(cond_1, f'cond_1 speed100_avg30: {cond_value_1:.2f}')
    c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
    c.green_red_print(cond_3, f'cond_3 {cond_value_3:.2f}')
    c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
    c.green_red_print(cond_5, f'cond_5 {cond_value_5:.2f}')

  if ha_cond \
    and cond_1 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and cond_5:

    condition = True
  if condition:          
    c.green_red_print(condition, 'buy_condition_speed_norm100')
  return condition
# version from 13/03/2025
def stock_buy_condition_before_market_open_old(df, df_1m, df_stats, display=False):
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False
  condition_type = 'No cond'

  if (df.index[i - 1].hour in [14, 15, 16]):

    cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
    cond_value_2 = cond_value_1
    # cond_value_3 = df['close'].iloc[-2] - df['close'].iloc[-5 : -1].min()
    cond_value_4 = df_stats[ticker]['norm100'].iloc[-1]
    cond_value_5 = maximum(df, i, 500) / df['close'].iloc[-2]
    cond_value_6 = maximum(df, i, 7) / df['close'].iloc[-2]

    cond_1 = cond_value_1 >= 0.7
    cond_2 = cond_value_2 <= 1.1
    # cond_3 = cond_value_3 <= 0
    cond_3 = True
    cond_4 = cond_value_4 >= 0.7
    
    # Buy with -0.4%
    cond_51 = cond_value_5 > 1.17
    cond_61 = cond_value_6 < 1.02
    # Buy with -0.7% 
    cond_52 = cond_value_5 > 1.05
    cond_62 = cond_value_6 < 1.03

    if display:
      warning.print('before_market_open conditions parameters:')
      c.green_red_print(cond_1, f'cond_1 {cond_value_1:.2f}')
      c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
      c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
      c.green_red_print(cond_51, f'cond_51 {cond_value_5:.2f}')
      c.green_red_print(cond_61, f'cond_61 {cond_value_6:.2f}')
      c.green_red_print(cond_52, f'cond_52 {cond_value_5:.2f}')
      c.green_red_print(cond_62, f'cond_62 {cond_value_6:.2f}')

    if cond_1 \
      and cond_2 \
      and cond_3 \
      and cond_4:

      if cond_52 \
        and cond_62:
        condition_type = 'before_market_open_2' # Buy with -0.7% 
        condition = True

      # More strict condition goes last
      if cond_51 \
        and cond_61:
        condition_type = 'before_market_open_1' # Buy with -0.4%
        condition = True
  if condition: 
    c.green_red_print(condition, condition_type)
  return condition, condition_type

def stock_buy_condition_speed_norm100(df, df_1m, df_stats, display=False):
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  condition = False

  cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
  cond_value_2 = df['close'].iloc[-2] - df['close'].iloc[-50 : -1].min()
  cond_value_3 = df_stats[ticker]['norm100'].iloc[-1]
  cond_value_4 = maximum(df, i, 500) / df['close'].iloc[-2]
  cond_value_5 = maximum(df, i, 7) / df['close'].iloc[-2]

  cond_1 = cond_value_1 >= 0.7 and cond_value_1 <= 1.1
  cond_2 = cond_value_2 <= 0   
  cond_3 = cond_value_3 >= 0.7
  cond_4 = cond_value_4 > 1.2
  cond_5 = cond_value_5 < 1.035
  
  if display:
    warning.print('speed_norm100 conditions parameters:')
    c.green_red_print(cond_1, f'cond_1 speed100_avg30: {cond_value_1:.2f}')
    c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
    c.green_red_print(cond_3, f'cond_3 {cond_value_3:.2f}')
    c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
    c.green_red_print(cond_5, f'cond_5 {cond_value_5:.2f}')

  if ha_cond \
    and cond_1 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and cond_5:

    condition = True
  if condition:          
    c.green_red_print(condition, 'buy_condition_speed_norm100')
  return condition

def stock_buy_condition_MA50_MA5_old(df, df_1m, df_stats, display=False):
  '''
  if gradient MA50 1hour > 0 \\
  and MA5 1hour - local minimum \\
  and rise no more than 0.55% \\
  and ha_cond
  buy_price: to have MA5[-1] > MA5[-2]: \\
           df['close'].iloc[-1] should more than df['close'].iloc[-6]\\ 
           and below this value on 0.5%
  sell_orders: 2 stop_limit loss orders, trailing order with modification \\ 
  order life time: 1hour \\
  profit value: trailing order with modification \\ 
  lose value:  -1% and -2% limit, and stop market if below -2.2%\\
  buy price: df_1m['close'].iloc[-1] + 0.05% during normal hours
  after-hours: NOT BUY
  '''
  condition = False
  condition_type = 'MA50_MA5'

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5

  cond_1 = (df['MA50'].iloc[-1] > [df['MA50'].iloc[-75], 
                                   df['MA50'].iloc[-50],
                                   df['MA50'].iloc[-25],
                                   df['MA50'].iloc[-5],
                                  ]).all()
  
  cond_2 = df_1m['close'].iloc[-1] >= df['close'].iloc[-6] * 0.995 \
            and df['MA5'].iloc[-2] <= df['MA5'].iloc[-3] \
            and df['MA5'].iloc[-3] <= df['MA5'].iloc[-4] \
            and df['MA5'].iloc[-4] <= df['MA5'].iloc[-5] \
            and df['MA5'].iloc[-6] <= df['MA5'].iloc[-7] \
            and df['MA5'].iloc[-7] <= df['MA5'].iloc[-8]
                                               
  # cond_2 = df['MA5'].iloc[-1] >= df['MA5'].iloc[-2] \
  #           and (df['MA5'].iloc[-2] < [df['MA5'].iloc[-3],
  #                                      df['MA5'].iloc[-4],
  #                                      df['MA5'].iloc[-5]
  #                                     ]).all()
  rise_value = df_1m['close'].iloc[-1] / df['close'].iloc[-3].min() # first version df_1m['close'].iloc[-1] / df['low'].iloc[-3].min()
  cond_3 = rise_value < 1.0055
  cond_4 = df_1m['close'].iloc[-1] / df['close'].iloc[-6] < 1.0055

  if display:
    warning.print('MA50_MA5 conditions parameters:')
    c.green_red_print(cond_1, f'cond_1')
    c.green_red_print(cond_2, f'cond_2')
    c.green_red_print(cond_3, f'cond_3')
    c.green_red_print(cond_4, f'cond_4')
    c.green_red_print(ha_cond, f'ha_cond')

  if cond_1 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and ha_cond:
    condition = True

  if condition: 
    c.green_red_print(condition, condition_type)

  return condition

def stock_buy_condition_MA50_MA5(df, df_1m, df_stats, ticker, display=False):
  '''
  if gradient MA5 1hour > 0 \\
  and MA5 1hour - MA50 1 hour > 0%
  and MA5 1hour - MA50 1 hour < 0.3%

  buy_price: to have MA5[-1] > MA5[-2]: \\
           df['close'].iloc[-1] should more than df['close'].iloc[-6]\\ 
           and below this value on 0.5%
  sell_orders: 2 stop_limit loss orders, trailing order with modification \\ 
  order life time: 1hour \\
  profit value: trailing order with modification \\ 
  lose value:  -1% and -2% limit, and stop market if below -2.2%\\
  buy price: df_1m['close'].iloc[-1] + 0.05% during normal hours
  after-hours: NOT BUY
  '''
  global market_value, total_market_value ,total_market_value_2m , \
    total_market_value_5m, total_market_value_30m, total_market_value_60m, \
    total_market_direction_60m
  
  condition = False
  condition_type = 'MA50_MA5'

  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5
  
  prediction_rise = 1.005
  grad_MA5 = df_1m['close'].iloc[-1] * prediction_rise >= df['close'].iloc[-6] \
            and df['MA5'].iloc[-2] >= df['MA5'].iloc[-3]

  deltaMA5_MA50 = df['MA5'].iloc[-1] / df['MA50'].iloc[-1]    
  deltaMA5_MA50_b3 = df['MA5'].iloc[-3] / df['MA50'].iloc[-3]    

  # deltaMA5_MA50_prediction = df['MA5'].iloc[-2] + (df_1m['close'].iloc[-1] * prediction_rise - df_1m['close'].iloc[-6]) / 5
  MA5_prediction = df['MA5'].iloc[-2] + (df_1m['close'].iloc[-1] * prediction_rise - df['close'].iloc[-6]) / 5
  MA50_prdediction = df['MA50'].iloc[-2] + (df_1m['close'].iloc[-1] * prediction_rise - df['close'].iloc[-51]) / 50
  deltaMA5_MA50_prediction = MA5_prediction / MA50_prdediction

  # cond_1 = deltaMA5_MA50 >= 1.000 and deltaMA5_MA50 <= 1.003 # first version
  cond_1 = deltaMA5_MA50_prediction >= 1.000 and deltaMA5_MA50_prediction <= 1.003
  cond_value_1 = deltaMA5_MA50_prediction
  cond_2 = deltaMA5_MA50_b3 < 0.999
  cond_value_2 = deltaMA5_MA50_b3

  MA50_max120_i = 120 - np.argmax(df['MA50'].iloc[-120:]) #0 the first and 120 is the last
  cond_3 = MA50_max120_i < 5 or MA50_max120_i > 60
  cond_value_3 = MA50_max120_i
  # np.amax([df['close'].iloc[0 : k].max(), df['open'].iloc[0 :k].max(), df['close'].iloc[i], df['open'].iloc[i]])

  cond_value_4 = df_1m['close'].iloc[-1] / df['MA50'].iloc[-1] 
  cond_4 = cond_value_4 < 1.005
  cond_5 = df_1m['MA5'].iloc[-1] > df_1m['MA20'].iloc[-1]
  cond_value_6 = df_1m['close'].iloc[-1] / mimimum(df, i, 24)
  cond_6 = cond_value_6 < 1.03 # current price hasn't risen more than 3% last 24 hours

  if deltaMA5_MA50 < 0.99 \
    or deltaMA5_MA50 > 1.01 \
    or not cond_2 \
    or (MA50_max120_i > 7 and  MA50_max120_i < 57):
    # exclude company from _list optimal list
    warning.print(f'{ticker} is excluding from current optimal stock list')
    exclude_time_dist[ticker] = datetime.now()
    stock_name_list_opt.remove(ticker)
    warning.print(f'Optimal stock name list len is {len(stock_name_list_opt)}')
    market_value = -1 # start offset
    total_market_value = -99  
    # Calculate market direction
    total_market_value_2m = 0
    total_market_value_5m = 0
    total_market_value_30m = 0
    total_market_value_60m = 0
    total_market_direction_60m = 0

  if display:
    warning.print('MA50_MA5 conditions parameters:')
    c.green_red_print(grad_MA5, f'grad_MA5')
    c.green_red_print(cond_1, f'cond_1 (deltaMA5_M50 prediction), {cond_value_1:.3f}')
    c.green_red_print(cond_2, f'cond_2 (deltaMA5_M50 3 hours before), {cond_value_2:.3f}')
    c.green_red_print(cond_3, f'cond_3 (distanse from 120 max), {cond_value_3:.3f}')
    c.green_red_print(cond_4, f'cond_4 (percantage above 1h MA50), {cond_value_4:.3f}')
    c.green_red_print(cond_5, f'cond_5 (current price more than 1m MA20)')
    c.green_red_print(cond_6, f'cond_6 (percantage above last 24 hours minimum), {cond_value_6:.3f}')
    c.green_red_print(ha_cond, f'ha_cond')

  if cond_1 \
    and grad_MA5 \
    and cond_2 \
    and cond_3 \
    and cond_4 \
    and cond_5:
    condition = True

  if condition: 
    c.green_red_print(condition, condition_type)

  return condition

# version from 13/03/2025 with MA, ha_cond and 
def stock_buy_condition_before_market_open(df, df_1m, df_stats, display=False):
  # additional condtions:
  # grad MA20(1min) > 0
  # dropping more than 5-7%
  # ha_cond
  # time before 14:00
  new_york_time = datetime.now(pytz.timezone('US/Eastern'))
  i_1m = df_1m.shape[0] - 1
  i = df.shape[0] - 1
  ha_cond = df_1m['ha_colour'].iloc[-1] == 'green' and df_1m['ha_colour'].iloc[-2] == 'green' \
    and number_red_candles(df_1m, i_1m, k=11) > 5
  # df_1m = MA(df_1m, 20)
  # df_1m = MA(df_1m, 60)
  # df = MA(df, 5)
  condition = False
  condition_type = 'No cond'

  # if not (new_york_time.hour >= 15 and new_york_time.hour <= 15):
  if True:

    max_value = df['close'][-14:].max()
    drop_value = max_value / df_1m['close'].iloc[-1]
    drop_value_24hr = df['high'][-24:].max() / df_1m['close'].iloc[-1]
    dist_from_min_24hr = df_1m['close'].iloc[-1] / df['low'][-24:].min()

    cond_value_1 = df_stats[ticker]['speed100_avg30'].iloc[-1]
    cond_value_2 = cond_value_1
    # cond_value_3 = df['close'].iloc[-2] - df['close'].iloc[-5 : -1].min()
    cond_value_4 = df_stats[ticker]['norm100'].iloc[-1]
    cond_value_5 = maximum(df, i, 500) / df['close'].iloc[-2]
    cond_value_6 = maximum(df, i, 7) / df['close'].iloc[-2]
    cond_value_7 = drop_value

    cond_1 = cond_value_1 >= 0.7
    cond_2 = cond_value_2 <= 1.1
    # cond_3 = cond_value_3 <= 0
    cond_3 = True
    cond_4 = cond_value_4 >= 0.7
    # cond_8 = (df_1m['MA20'].iloc[-1] > df_1m['MA20'].iloc[-4:-1]).all()
    cond_8 = (df_1m['MA20'].iloc[-1] > df_1m['MA20'].iloc[-4:-1]).all()
    cond_9 = df['MA5'].iloc[-1] >= df['MA5'].iloc[-2]
    # add drop value 24 hours. Should be more that 2.5% and distanse from min24hours less then 0.5%
    cond_10 = (df_1m['MA50'].iloc[-1] > df_1m['MA50'].iloc[-4:-1]).all() \
              and (df_1m['MA50'].iloc[-25] > df_1m['MA50'].iloc[-4:-1]).all() \
              and drop_value_24hr > 1.025 \
              and dist_from_min_24hr < 1.005
  
    # Buy with -0.4%
    cond_51 = cond_value_5 > 1.17
    cond_61 = cond_value_6 < 1.02
    # Buy with -0.7% 
    cond_52 = cond_value_5 > 1.05
    cond_62 = cond_value_6 < 1.03

    
    if display:
      warning.print('before_market_open conditions parameters:')
      warning.print(f'drop value is {drop_value:.3f}')
      c.green_red_print(ha_cond, f'ha_cond')
      c.green_red_print(cond_1, f'cond_1 {cond_value_1:.2f}')
      c.green_red_print(cond_2, f'cond_2 {cond_value_2:.2f}')
      c.green_red_print(cond_4, f'cond_4 {cond_value_4:.2f}')
      c.green_red_print(cond_51, f'cond_51 {cond_value_5:.2f}')
      c.green_red_print(cond_61, f'cond_61 {cond_value_6:.2f}')
      c.green_red_print(cond_52, f'cond_52 {cond_value_5:.2f}')
      c.green_red_print(cond_62, f'cond_62 {cond_value_6:.2f}')
      c.green_red_print(cond_8, f'cond_8')
      c.green_red_print(cond_9, f'cond_9')
      c.green_red_print(cond_10, f'cond_10')

    if cond_1 \
      and cond_2 \
      and cond_3 \
      and cond_4 \
      and cond_8 \
      and cond_9 \
      and ha_cond:

      if cond_52 \
        and cond_62 \
        and drop_value >= 1.07:
        condition_type = 'before_market_open_2' # Buy with -0.7% 
        condition = True

      # More strict condition goes last
      if cond_51 \
        and cond_61 \
        and drop_value >= 1.05:
        condition_type = 'before_market_open_1' # Buy with -0.4%
        condition = True

      if cond_51 and cond_61 \
        and cond_52 and cond_62 \
        and cond_10 \
        and drop_value > 1.003:
          condition_type = 'before_market_open_3' # 
          condition = True
        
  if condition: 
    c.green_red_print(condition, condition_type)
  return condition, condition_type

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
    alarm.print(traceback.format_exc())
    current_profit = 10
  return float(current_profit)

def update_buy_order_based_on_platform_data(order, history_orders = None):
    '''
    return: order
    '''
    try:
      if type(history_orders) != pd.DataFrame:
        if history_orders == None:
          history_orders = ma.get_history_orders()
      else:
        if history_orders.empty:
          history_orders = ma.get_history_orders()
    except Exception as e:
      alarm.print(traceback.format_exc())
      history_orders = ma.get_history_orders()

    try:
      if type(historical_orders) == pd.DataFrame and not historical_orders.empty:
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
          # df = ti.update_order(df, order)
          # add order['buy_time']?

          if history_order['order_status'].values[0] == ft.OrderStatus.CANCELLED_PART:
            ma.modify_limit_if_touched_order(order, order['gain_coef'])
            ma.modify_stop_order(order, order['lose_coef'])
    except Exception as e:
      alarm.print(traceback.format_exc())
    return order

def load_orders_from_csv():
  # FUNCTION TO UPDATE times from csv files to df with correct time format
  df = pd.read_csv('db/real_trade_db.csv', index_col='Unnamed: 0')
  df['buy_time'] = pd.to_datetime(df['buy_time'], dayfirst=False)
  df['sell_time'] = pd.to_datetime(df['sell_time'], dayfirst=False)
  gv.ORDERS_ID  = df['id'].max()
  if gv.ORDERS_ID == np.nan:
    gv.ORDERS_ID = 1
  return df

def load_orders_from_xlsx():
  try:
    date = str(datetime.now()).split(' ')[0]
    shutil.copyfile('db/real_trade_db.xlsx', f'db/bin/real_trade_db_{date}.xlsx')
  except Exception as e:
    alarm.print(traceback.format_exc())
  df = pd.read_excel('db/real_trade_db.xlsx', index_col='Unnamed: 0')
  gv.ORDERS_ID  = df['id'].max()
  if gv.ORDERS_ID == np.nan:
    gv.ORDERS_ID = 1
  return df

def get_orders_list_from_moomoo_orders(orders: pd.DataFrame):
  orders_list = []
  for index, row in orders.iterrows():
    ticker = row['code'].split('.')[1]
    orders_list.append(ticker)
  return orders_list

def isNaN(num):
    return num != num

def place_sell_order_if_it_was_not_placed(df, order, sell_orders, sell_orders_list,
                                          price, order_type, current_gain=1, low_trail_value=False) -> Tuple[pd.DataFrame, pd.DataFrame]:
  '''
    - type: buy | limit_if_touched | stop | trailing_LIT | trailing_stop_limit | stop_limit
  '''
  moomoo_order_id = None
  try:
    ticker = order['ticker']
    order_id_type = order_type + '_order_id'
    order_id = order[order_id_type]
    qty = order['stocks_number']
  except Exception as e:
    alarm.print(traceback.format_exc())
    order_id = None
 
  if (order_id in [None, '', 'FAXXXX'] 
    or isNaN(order_id)):
    # and ticker not in sell_orders_list:
    if order_type == 'limit_if_touched':
      moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty)
    if order_type == 'stop':
      moomoo_order_id = ma.place_stop_order(ticker, price, qty)   
    if order_type == 'stop_limit':
      moomoo_order_id = ma.place_stop_limit_sell_order(ticker, price, qty)   
    if order_type == 'trailing_LIT':
      moomoo_order_id = ma.place_limit_if_touched_order(ticker, price, qty, aux_price_coef=aux_price_coef, remark='trailing_LIT')

    if order_type == 'trailing_stop_limit':
      trail_spread = order['buy_price'] * trail_spread_coef
      if order['buy_condition_type'] == '1230':
        if current_gain >= 1.005 and current_gain < 1.01:
          trail_value = 0.15
        else:
          trail_value = 0.3
        if (datetime.now().astimezone(tzinfo_ny).hour == 16 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
          or (datetime.now().astimezone(tzinfo_ny).hour == 17 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
          or low_trail_value:
          trail_value = 0.02  
      elif order['buy_condition_type'] == '930-47':
        if current_gain >= 1.005 and current_gain < 1.01:
          trail_value = 0.12
        else:
          trail_value = default.trailing_ratio
      elif order['buy_condition_type'] == 'MA50_MA5':
          trail_value = default.trailing_ratio_MA50_MA5
      else:
        trail_value = default.trailing_ratio
      moomoo_order_id = ma.place_trailing_stop_limit_order(ticker, price, qty, trail_value=trail_value, trail_spread=trail_spread)
      order['trailing_ratio'] = trail_value
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
      alarm.print(traceback.format_exc())
      alarm.print(f'{ticker} {order_type} order has not been placed')
  return df, order

def clean_cancelled_and_failed_orders_history(df, type) -> pd.DataFrame:
    try:
      historical_orders = ma.get_history_orders()
      if not historical_orders.empty:
        placed_stocks = df.loc[(df['status'] == type)]
        for index, row in placed_stocks.iterrows():
          ticker = row['ticker']
          buy_order = historical_orders.loc[historical_orders['order_id'] == row['buy_order_id']]
          if buy_order.shape[0] > 0:
            if buy_order['order_status'].values[0] in [ft.OrderStatus.CANCELLED_ALL, ft.OrderStatus.FAILED]:
              index = df.loc[(df['buy_order_id'] == row['buy_order_id']) & (df['status'] == type) & (df['ticker'] == ticker)].index
              df.drop(index=index, inplace=True)
    except Exception as e:
      alarm.print(e)
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

def trailing_stop_limit_order_trailing_ratio_modification(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  try:
    current_gain = current_price / order['buy_price']
    trail_spread = order['buy_price'] * trail_spread_coef
    trailing_ratio = order['trailing_ratio']
    
    print(f'Ticker is {order['ticker']}, current_gain is {current_gain:.4f}')
    if current_gain >= 1.003 and order['trailing_ratio'] == default.trailing_ratio:
      trailing_ratio = 0.4
    if current_gain >= 1.004 and order['trailing_ratio'] == 0.4:
      trailing_ratio = 0.3
    if current_gain >= 1.005 and order['trailing_ratio'] == 0.3:
      trailing_ratio = 0.25
    if current_gain >=  1.006 and order['trailing_ratio'] == 0.25:
      trailing_ratio = 0.2
    if current_gain >=  1.007 and order['trailing_ratio'] == 0.2:
      trailing_ratio = 0.15

    if order['trailing_ratio'] != trailing_ratio:
      order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
      if order_id != order['trailing_stop_limit_order_id']:
        order['trailing_stop_limit_order_id'] = order_id
      order['trailing_ratio'] = trailing_ratio
      df = ti.update_order(df, order)
  except Exception as e:
    alarm.print
  return df, order

def check_buy_order_for_cancelation(df, ticker, historical_orders) -> pd.DataFrame:
  '''
   CHECKING FOR CANCELATION OF THE BUY ORDER
   Check current price and status of buy order
   If price more or equal than price*gain_coef CANCEL the BUY ORDER
   If status was not FILLED_PART, change status of order to 'cancel', update the order
  '''
  if ticker in list(set(placed_stocks_list + limit_buy_orders_list + limit_if_touched_buy_orders_list + stop_limit_buy_orders_list)):
    try:
      stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=True)
      if not stock_df_1m.empty:
        current_price = stock_df_1m['close'].iloc[-1]
      else:
        current_price = 0    
      
      order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])]
      if order.empty:
        order = df.loc[(df['ticker'] == ticker) & df['status'].isin([ 'cancelled'])]

      order_failed_or_cancelled = False
      if (ticker not in positions_list):
        code = 'US.' + ticker
        historical_order = historical_orders.loc[(historical_orders['code'] == code) \
                                    & (historical_orders['order_id'] == order['buy_order_id'].values[0])]
        if historical_order.shape[0] > 0:
          historical_order_status = historical_order['order_status'].values[0]
          if historical_order_status in [ft.OrderStatus.FAILED, ft.OrderStatus.CANCELLED_ALL]:
            order_failed_or_cancelled = True

      if not order.empty:
        order = order.sort_values('buy_time').iloc[-1]
        order_time = timedelta_minutes(order) 

        cond_1 = order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3'] \
                 and ((stock_df_1m.index[-1].hour in [11, 12, 13, 14] and stock_df_1m.index[-1].minute > 40) 
                  or order_time >= order_before_market_open_life_time_min)
        cond_2 = order['buy_condition_type'] == '1m' \
                 and order_time >= order_1m_life_time_min 
        cond_3 = order['buy_condition_type'] == 'MA50_MA5' \
                 and order_time >= order_MA50_MA5_life_time_min 
        cond_4 = not order['buy_condition_type'] in ['1m', 'before_market_open_1', 'before_market_open_2', 'before_market_open_3', 'MA50_MA5'] \
                 and order_time > order_1h_life_time_min

        if current_price >= order['buy_price'] * (order['gain_coef'] + 0.015) / 0.93  \
          or cond_1 \
          or cond_2 \
          or cond_3 \
          or cond_4 \
          or order_failed_or_cancelled:
                
          if buy_orders.shape[0] > 0:
            buy_order = limit_buy_orders.loc[
              buy_orders['order_id'] == order['buy_order_id']]
            if not buy_order.empty:
              if buy_order['order_status'].values[0] != ft.OrderStatus.FILLED_PART:
                try:
                  # cancel limit_if_touched_sell_order and stop_order if they was placed
                  if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='limit_if_touched')    
                  if order['stop_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='stop')
                  if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='trailing_stop_limit')
                  if order['trailing_LIT_order_id'] not in ['', None, []]:                  
                      ma.cancel_order(order, order_type='trailing_LIT_order')
                except Exception as e:
                  alarm.print(traceback.format_exc())
              else:
                order['status'] = 'filled part'
            # cancel buy limit order
            ma.cancel_order(order, order_type='buy')
            order['status'] = 'cancelled'
            df = ti.update_order(df, order)
    except Exception as e:
      print(e)
  return df

def check_if_sell_orders_have_been_executed(df, ticker) -> pd.DataFrame:
  '''
  Checking if sell orders have been executed
  '''
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
      historical_stop_limit_order = historical_orders.loc[
        historical_orders['order_id'] == order['stop_limit_order_id']]
      
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
        ma.cancel_order(order, order_type='stop_limit')
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
        ma.cancel_order(order, order_type='stop_limit')
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
        ma.cancel_order(order, order_type='stop_limit')
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
        ma.cancel_order(order, order_type='stop_limit')
      if historical_stop_limit_order.shape[0] > 0 \
        and  historical_stop_limit_order['order_status'].values[0] == ft.OrderStatus.FILLED_ALL\
        and order['status'] in ['bought', 'filled part']:
        sell_price = order['buy_price'] * order['lose_coef']
        order = ti.sell_order(order, sell_price=sell_price, historical_order = historical_stop_limit_order)
        df = ti.update_order(df, order)
        # play sound:
        winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
        # cancel stop order and limit if touched 
        ma.cancel_order(order, order_type='stop')
        ma.cancel_order(order, order_type='limit_if_touched')
        ma.cancel_order(order, order_type='trailing_LIT')
        ma.cancel_order(order, order_type='stop_limit')
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df

def check_buy_order_info_from_order_history(df, ticker) -> pd.DataFrame:
    try: 
      if ticker in placed_stocks_list:
        orders = df.loc[(df['ticker'] == ticker) & df['status'].isin(['placed'])].sort_values('buy_time')
        if orders.shape[0] > 0:
          order = orders.iloc[-1]
          # 1.111 was set during buy order creation
          if order['buy_commission'] == 1.111 \
            or math.isnan(order['buy_commission']) \
            or order['buy_commission'] == None \
            or order['buy_commission'] == 0:
            order = update_buy_order_based_on_platform_data(order, historical_orders)
            if order['status'] == 'bought':
              df = ti.update_order(df, order)
    except Exception as e:
      alarm.print(traceback.format_exc())
    return df

def check_sell_order_has_been_placed(df, order, ticker, order_type) -> Tuple[pd.DataFrame, pd.DataFrame]:
  '''
  return: df, order
  '''
  sell_orders_name = order_type + '_sell_orders'
  sell_orders_list_name = order_type + '_sell_orders_list'
  try:
    sell_orders = globals()[sell_orders_name]
    sell_orders_list = globals()[sell_orders_list_name]
    if order_type == 'trailing_stop_limit':
      stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=True)
      if not stock_df_1m.empty:
        current_price = stock_df_1m['close'].iloc[-1]
      else:
        current_price = 0
      if current_price >= order['buy_price'] * trailing_stop_limit_act_coef \
          or place_trailing_stop_limit_order_imidiately:
        permition = True
      else:
        permition = False
    else:
        permition = True

    if order_type in ['stop', 'stop_limit']:
      price = order['buy_price'] * order['lose_coef']
    elif order_type == 'limit_if_touched':
      price = order['buy_price'] * order['gain_coef']
    elif order_type == 'trailing_LIT':
      price = order['buy_price'] * order['trailing_LIT_gain_coef'] 
    else:
      price = order['buy_price'] * default.gain_coef

    if permition:
      df, order = place_sell_order_if_it_was_not_placed(df,
        order=order,
        sell_orders=sell_orders,
        sell_orders_list=sell_orders_list,
        price=price,
        order_type=order_type) 
      
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order

def place_traililing_stop_limit_order_at_the_end_of_trading_day(df, order, ticker, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Place limit order if end of the trading day for 1230 buy condition
  try:
    current_gain = current_price / order['buy_price']
    print(f'Stock {ticker}, current gain is {current_gain:.4f}')

    # Modification from 18/03/2025
    place_order_cond1 = current_gain > 1.0015 \
                        and stock_df_1m['MA7'].iloc[-1] < stock_df_1m['MA7'].iloc[-2]
    place_order_cond2 = current_gain > 1.0015 \
                        and stock_df_1m['ha_colour'].iloc[-1] == 'red' \
                        and stock_df_1m['ha_colour'].iloc[-2] == 'red'
    if place_order_cond1 or place_order_cond2:
      low_trail_value = True
    else:
      low_trail_value = False

    if ( (datetime.now().astimezone(tzinfo_ny).hour >= 15 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
          and current_gain > 0.998) \
            or current_gain >= 1.003 \
            or (current_gain > 0.998 and datetime.now().astimezone(tzinfo_ny).hour < 9) \
            or place_order_cond1 \
            or place_order_cond2:
      price = stock_df_1m['close'].iloc[-1]
      df, order = place_sell_order_if_it_was_not_placed(df,
        order=order,
        sell_orders=trailing_stop_limit_sell_orders,
        sell_orders_list=trailing_stop_limit_sell_orders_list,
        price=price,
        order_type='trailing_stop_limit',
        current_gain=current_gain,
        low_trail_value=low_trail_value) 
  except Exception as e:
    alarm.print(traceback.format_exc())      
  return df, order 

def modify_trailing_stop_limit_1230_order(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  # Modify order with buy_codition_type = 12:30
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    if order['buy_condition_type'] == '1230' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef

      trailing_ratio = order['trailing_ratio']
      if current_gain >= 1.005:
        trailing_ratio = 0.15

      if current_gain <= 1.005 and order['trailing_ratio'] != 0.15:
        trailing_ratio = 0.3

      if ((datetime.now().astimezone(tzinfo_ny).hour == 16 and datetime.now().astimezone(tzinfo_ny).minute > 55) \
        or (datetime.now().astimezone(tzinfo_ny).hour == 17 and datetime.now().astimezone(tzinfo_ny).minute > 55)) \
          and current_gain >= 0.998:
        trailing_ratio = 0.02
  
      if order['trailing_ratio'] != trailing_ratio:
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_trailing_stop_limit_MA50_MA5_order(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
  try:
    order_id = order['trailing_stop_limit_order_id']
    current_gain = current_price / order['buy_price']
    if order['buy_condition_type'] == 'MA50_MA5' \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      trail_spread = order['buy_price'] * trail_spread_coef
      trailing_ratio = order['trailing_ratio']


      # 0 -- 0.2 -- 0.5 -- 1 
      #   0.99  0.7    0.5

      # safe condition 
      if current_gain < 1.01 and order['trailing_ratio'] > 0.99:
        trailing_ratio = 0.99

      if current_gain >= 1.002 and order['trailing_ratio'] == 0.99:
        trailing_ratio = 0.8
      if current_gain > 1.002 and current_gain < 1.005 \
        and order['trailing_ratio'] > 0.7:
        trailing_ratio = 0.7
      if current_gain >= 1.005 and current_gain < 1.0075 \
        and order['trailing_ratio'] > 0.55:
        trailing_ratio = 0.55
      if current_gain >= 1.0075 and current_gain < 1.01 \
        and order['trailing_ratio'] > 0.35:
        trailing_ratio = 0.35
      if current_gain >= 1.01 and current_gain < 1.015 \
        and order['trailing_ratio'] > 0.32:
        trailing_ratio = 0.32
      if current_gain >= 1.015 \
        and order['trailing_ratio'] > 0.3:
        trailing_ratio = 0.3


      # if current_gain >= 1.01 and current_gain < 1.03 \
      #   and order['trailing_ratio'] > 0.7:
      #   trailing_ratio = 0.7 
      # if current_gain >= 1.03 and current_gain < 1.05 \
      #   and order['trailing_ratio'] > 0.6:
      #   trailing_ratio = 0.6
      # if current_gain >= 1.05 and current_gain < 1.06 \
      #   and order['trailing_ratio'] > 0.5:
      #   trailing_ratio = 0.5
      # if current_gain >= 1.07 \
      #   and order['trailing_ratio'] > 0.3:
      #   trailing_ratio = 0.3
  
      if order['trailing_ratio'] != trailing_ratio:
        order_id = ma.modify_trailing_stop_limit_order(order=order,
                                            trail_value=trailing_ratio,
                                            trail_spread=trail_spread)  
        if order_id != order['trailing_stop_limit_order_id']:
          order['trailing_stop_limit_order_id'] = order_id
        order['trailing_ratio'] = trailing_ratio
        df = ti.update_order(df, order)
  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def modify_stop_limit_before_market_open_order(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
 
  try:
    order_id = order['stop_limit_order_id']
    current_gain = current_price / order['buy_price']

    if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3'] \
      and not(order_id in [None, '', 'FAXXXX'] 
          or isNaN(order_id)):
      
      lose_coef = order['lose_coef']
      if current_gain < 1.01 and order['buy_condition_type'] == 'before_market_open_3' \
        and order['lose_coef'] < 1.003:
        lose_coef = 1.003
      if current_gain >= 1.01 and current_gain < 1.02 \
         and order['buy_condition_type'] == 'before_market_open_3':
         lose_coef = 1.005

      if current_gain >= 1.02 and current_gain < 1.03 and order['lose_coef'] < 1.0065:
        lose_coef = 1.0065
      elif current_gain >= 1.03 and current_gain < 1.04 \
        and order['lose_coef'] < 1.0075:
        lose_coef = 1.0075
      elif current_gain >= 1.04 and current_gain < 1.045 \
        and order['lose_coef'] < 1.0085:
        lose_coef = 1.0085
      elif current_gain >= 1.045 and current_gain < 1.05 \
        and order['lose_coef'] < 1.0099:
        lose_coef = 1.0099

      # Fast sold it gain more than 3%
      try:
        elapsed_time = datetime.now() - order['buy_time']
        elapsed_time_minutes = elapsed_time.days * 24 * 60 + elapsed_time.seconds / 60
        if current_gain > 1.03 and elapsed_time_minutes < 360:
          lose_coef = current_gain * 0.9999
      except Exception as e:
        alarm.print(traceback.format_exc())

      if order['lose_coef'] != lose_coef:
        order_id = ma.modify_stop_limit_order(order=order,lose_coef=lose_coef)  
        if order_id != order['stop_limit_order_id']:
          order['stop_limit_order_id'] = order_id
        order['lose_coef'] = lose_coef
        df = ti.update_order(df, order)

  except Exception as e:
    alarm.print(traceback.format_exc())
  return df, order   

def buy_price_based_on_condition(df, df_1m, condition_type):
  buy_price = 0
  try:
    if condition_type == '930-47':
      buy_price = df_1m['close'].iloc[-1]
    elif condition_type in ['drowdown930', 'drowdown', 'maxminnorm', 'speed_norm100']:
      # buy_price = df_1m['close'].iloc[-2] # ver. 1
      buy_price = mimimum(df_1m, 4)
    elif condition_type in ['1130_1', '1130_2']:
      buy_price = df_1m['close'].iloc[-2]
    elif condition_type == '1230':
      buy_price = df_1m['close'].iloc[-2]
    elif condition_type in ['1230_2']: 
      buy_price = min(df_1m['close'].iloc[-4:-1].min(), df_1m['open'].iloc[-4:-1].min(), df_1m['close'].iloc[-1])
    elif condition_type == '1m':
      buy_price = min(df_1m['close'].iloc[-10:-1].min(), df_1m['open'].iloc[-10:-1].min())
    elif condition_type == '1h':
      buy_price = df['close'].iloc[-2]
    elif condition_type == '1230_1330_inertia':
      buy_price = df_1m['close'].iloc[-2]
    else:
      buy_price = df_1m['close'].iloc[-2]

    # if condition_type == 'before_market_open_1':
    #    buy_price = min(df_1m['close'].iloc[-1] * 0.95, df['close'].iloc[-5 : -1].min()) # ver.1: 0.996; ver.2: 0.975; ver.3: 0.965
    # elif condition_type == 'before_market_open_2':
    #    buy_price = min(df_1m['close'].iloc[-1] * 0.938, df['close'].iloc[-5 : -1].min()) # ver.1: 0.993; ver.2: 0.965 ver3: 0.953
    if condition_type in ['before_market_open_1', 'before_market_open_2']:
      buy_price = df_1m['close'].iloc[-1] * 1.0005
    if condition_type == 'MA50_MA5':
      buy_price = df_1m['close'].iloc[-1] * 1.0003

  except Exception as e:
    alarm.print(traceback.format_exc())        
  return buy_price

def recalculate_trailing_LIT_gain_coef(df, order, current_price) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
  return df, order

def check_sell_orders_for_all_bougth_stocks():
  global df
    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if needed
  for ticker in positions_list:
    try:
      if ticker in bought_stocks_list:
        order = bought_stocks.loc[bought_stocks['ticker'] == ticker].sort_values('buy_time').iloc[-1] 
        stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=True)
        if not stock_df_1m.empty:
          current_price = stock_df_1m['close'].iloc[-1]  
          current_gain = current_price / order['buy_price']
        else:
          current_price = 0
          current_gain = 0
        if not order['buy_condition_type'] in ['1230', 'before_market_open_1', 'before_market_open_2', 'before_market_open_3', 'MA50_MA5']:
          # Checking limit_if_touched_order
          if False:
            df, order = check_sell_order_has_been_placed(df, ticker, order_type='limit_if_touched' )
          # Checking stop_order
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop')
          # Checking trailing_LIT_order
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_LIT')
          # Checking trailing_stop_limit_order
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
          # Modification of trailing stop limit order based on current gain
          df, order = trailing_stop_limit_order_trailing_ratio_modification(df, order, current_price)
          # Recalculate Trailing LIT gain coefficient:
          df, order = recalculate_trailing_LIT_gain_coef(df, order, current_price) 
        if order['buy_condition_type'] == '1230':
          # Place limit order if end of the trading day for 1230 buy condition
          df, order = place_traililing_stop_limit_order_at_the_end_of_trading_day(df, order, ticker, current_price) 
          # Modify order with buy_codition_type = 12:30
          df, order = modify_trailing_stop_limit_1230_order(df, order, current_price) 
        if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3']:
          # Place limit if touched sell order 
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
          # Place stop limit sell order if gain more that 2%:
          if order['buy_condition_type'] in ['before_market_open_1', 'before_market_open_2'] \
            and current_gain >= 1.01:  # v1. >=1.02 v2.1.005
              df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
          if order['buy_condition_type'] == 'before_market_open_3' \
            and current_gain > 1.003:
              df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
          # Modify stop limit sell order based on current gain and sell if fast rise 3%
          df, order = modify_stop_limit_before_market_open_order(df, order, current_price)

        if order['buy_condition_type'] == 'MA50_MA5':
          # Place limit if touched sell order 1
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='limit_if_touched')
          df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='stop_limit')
          if current_gain >= 1.001:  
            df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
          # Modify trailing stop limit sell order based on current gain   
          df, order = modify_trailing_stop_limit_MA50_MA5_order(df, order, current_price)
      else:
        alarm.print(f'{ticker} is in positional list but not in DB!')
    except Exception as e:
      alarm.print(traceback.format_exc())

def check_stocks_for_inclusion():

  warning.print('Checking optimal stock list for inclusion ...')
  for ticker in stock_name_list: 
    try:
      if ticker in exclude_time_dist:
        delta = datetime.now() - exclude_time_dist[ticker]
        deltatime_minutes = delta.seconds / 60 + delta.days * 24 * 60
        if deltatime_minutes >= 60 \
          and ticker not in stock_name_list_opt:
          stock_name_list_opt.append(ticker)
          exclude_time_dist.pop(ticker, None)
          warning.print(f'Stock {ticker} is including to optimal stock list')
    except Exception as e:
      alarm.print(traceback.format_exc())
    
  try:
      file_name = f'temp/exclude_time_dist.plk'
      with open(file_name, 'wb') as file:
        pickle.dump(exclude_time_dist, file)
        warning.print(f'Optimal stock name list len after inclustion check is {len(stock_name_list_opt)}')
      
      file_name = f'temp/stock_name_list_opt.plk'
      with open(file_name, 'wb') as file:
        pickle.dump(stock_name_list_opt, file)

  except Exception as e:
      alarm.print(traceback.format_exc())
#%% MAIN
if __name__ == '__main__':

  # All parameter should be False. Change to True if you need change\fix DB
  load_from_csv = False
  load_from_xslx = True
  clean_placed_orders = False
  clean_cancelled_orders = True
  read_sql_from_df = False
  test_trading = False
  skip_market_direction_calc = True
  test_buy_sim = False
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

  file_name = f'temp/exclude_time_dist.plk'
  file = pathlib.Path(file_name)
  if file.is_file():
    with open(file_name, 'rb') as file:
        exclude_time_dist  = pickle.load(file)

  file_name = f'temp/stock_name_list_opt.plk'
  file = pathlib.Path(file_name)
  if file.is_file():
    with open(file_name, 'rb') as file:
      stock_name_list_opt = pickle.load(file)

  # SQL INIT
  try:
    parent_path = pathlib.Path(__file__ ).parent
    folder_path = pathlib.Path.joinpath(parent_path, 'sql')
    db = sql_db.DB_connection(folder_path, 'trade.db', df)
    if load_from_csv or load_from_xslx or read_sql_from_df:
      db.update_db_from_df(df)
  except Exception as e:
    alarm.print(traceback.format_exc())

  us_cash = ma.get_us_cash()
  profit_24hours = current_profit(df)
  c.print(f'Availble withdrawal cash is {us_cash}')
  c.print(f'Last 24 hours profit is {profit_24hours}')

  # Algorithm !!! not up-to-date
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

      # Orders types:
      # buy_order
      # Sell orders:
      # limit_if_touched_order
      # stop_order
      # trailing_LIT_order
      # trailing_stop_limit


  market_value = -1 # start offset
  total_market_value = -99  
  # Calculate market direction
  total_market_value_2m = 0
  total_market_value_5m = 0
  total_market_value_30m = 0
  total_market_value_60m = 0
  total_market_direction_60m = 0
  if not skip_market_direction_calc:  
    warning.print('Calculation of market direction:')
    for ticker in tqdm(stock_name_list_opt):
        try:
            stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m')
            total_market_value_2m += minmax(stock_df_1m['pct'].iloc[-2:-1].sum(), -3, 3)
            total_market_value_5m += minmax(stock_df_1m['pct'].iloc[-5:-1].sum(), -3, 3)
            total_market_value_30m += minmax(stock_df_1m['pct'].iloc[-30:-1].sum(), -3, 3)
            total_market_value_60m += minmax(stock_df_1m['pct'].iloc[-60:-1].sum(), -3, 3)
            if stock_df_1m['pct'].iloc[-60:-1].sum() > 0:
              total_market_direction_60m += 1 
            else:
              total_market_direction_60m -= 1
        except Exception as e:
          alarm.print(traceback.format_exc())
    warning.print(f'Total markert direction 60m is {total_market_direction_60m}')

  while True:
    
    alarm.print('YOU ARE RUNNING REAL TRADE ACCOUNT')  
    # 1. Check what stocks are bought based on MooMoo (position list) and df
    positions_list = ma.get_positions()
    # Statistic calculations 
    df_stats = stats_calculation(stock_name_list_opt)
    # ReLoad trade history (syncronization) as maybe losing df somewhere
    if load_from_csv:
      df = load_orders_from_csv()
    elif load_from_xslx:
      df = load_orders_from_xlsx()
    else:
      df = ti.load_trade_history() # load previous history

    check_stocks_for_inclusion()

    # Get current orders and they lists:
    limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, \
    limit_if_touched_buy_orders, trailing_LIT_sell_orders, trailing_stop_limit_sell_orders, \
        stop_limit_buy_orders, stop_limit_sell_orders = ma.get_orders()
    buy_orders = pd.concat([limit_if_touched_buy_orders, limit_buy_orders])
    limit_buy_orders_list = get_orders_list_from_moomoo_orders(limit_buy_orders)
    limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_buy_orders)
    limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
    stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
    trailing_LIT_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_sell_orders)
    trailing_stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_sell_orders)
    stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(stop_limit_sell_orders)
    stop_limit_buy_orders_list = get_orders_list_from_moomoo_orders(stop_limit_buy_orders)

    historical_orders = ma.get_history_orders()
    # Check bought stocks and placed based on df 
    bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
    
    for ticker in list(set(placed_stocks_list + limit_buy_orders_list + limit_if_touched_buy_orders_list + stop_limit_buy_orders_list)):
      # Recheck all placed buy orders from the order history
      df = check_buy_order_info_from_order_history(df, ticker)
      # Recheck all placed buy orders from the orders by cancel time condition  
      df = check_buy_order_for_cancelation(df, ticker, historical_orders)
    
    for ticker in bought_stocks_list:
      # Checking if sell orders have been executed
      df = check_if_sell_orders_have_been_executed(df, ticker)

    # Check statuses of all bought stocks if they not in positional list:
    try:
      # ticker in bought stocks list after confirmation of the buy order
      for ticker in bought_stocks_list:
          order = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time').iloc[-1] 
          if ticker not in positions_list:
            historical_orders_= historical_orders.loc[(historical_orders['order_status']  == ft.OrderStatus.FILLED_ALL) &
                                                      (historical_orders['code'] == MARKET + ticker) &
                                                      (historical_orders['trd_side'] == ft.TrdSide.SELL) &
                                                      (historical_orders['qty'] == order['stocks_number'])
                                                      ]
            if historical_orders_.shape[0] > 0:
              historical_order = historical_orders_.sort_values('updated_time').iloc[-1]
              order = ti.sell_order(order, sell_price=0.01, historical_order = historical_order)
              df = ti.update_order(df, order)
              if order['trailing_LIT_order_id'] not in ['', None, []]:                  
                ma.cancel_order(order, order_type='trailing_LIT')    
              if order['stop_order_id'] not in ['', None, []]:                  
                ma.cancel_order(order, order_type='stop')
              if order['trailing_stop_limit_order_id'] not in ['', None, []]:                  
                ma.cancel_order(order, order_type='trailing_stop_limit')
              if order['limit_if_touched_order_id'] not in ['', None, []]:                  
                ma.cancel_order(order, order_type='limit_if_touched')            
    except Exception as e:
      alarm.print(traceback.format_exc())

    # 2. Check that for all bought stocks placed LIMIT_IF_TOUCHED, STOP orders, Trailing STOP LIMIT/trailing_LIT_order if needed
    check_sell_orders_for_all_bougth_stocks()

    # 3. For optimal stock list:
    counter = 0
    market_value_2m = 0
    market_value_5m = 0
    market_value_30m = 0
    market_value_60m = 0
    market_direction_60m = 0
    
    blue.print(f'total market_value {total_market_value:.2f}, 2m is {total_market_value_2m:.2f}, 5m is {total_market_value_5m:.2f},')
    blue.print(f'30m is {total_market_value_30m:.2f}, 60m is {total_market_value_60m:.2f}')
    blue.print(f'total market direction is {total_market_direction_60m:.2f}')
    us_cash = ma.get_us_cash()
    c.print(f'Available cash is {us_cash:.2f}, min buy sum is {min_buy_sum}', color='turquoise')

    current_minute = datetime.now().astimezone().minute 
    current_hour = datetime.now().astimezone().hour
    tzinfo_ny = pytz.timezone('America/New_York')
    new_york_time = datetime.now().astimezone(tzinfo_ny)
    new_york_hour = new_york_time.hour
    new_york_minute = new_york_time.minute
    new_york_week = new_york_time.weekday()
    market_time = (new_york_hour >= 10 and new_york_hour < 16) \
                  or (new_york_hour ==9 and new_york_minute >= 30) \
                  and new_york_week in [0, 1, 2 ,3, 4]                

    # market_time = (current_hour >= 23 and current_hour <= 7)
     
    if us_cash > min_buy_sum \
      or test_buy_sim:

      for ticker in stock_name_list_opt:
        print(f'Stock is {ticker}')

        order = []    
        # 3.1 If counter condition:  
        counter += 1
        if counter > 20:
          us_cash = ma.get_us_cash()
          bought_stocks, placed_stocks, bought_stocks_list, placed_stocks_list = get_bought_and_placed_stock_list(df)
          historical_orders = ma.get_history_orders()
          # Get current orders and they lists:
          limit_if_touched_sell_orders, stop_sell_orders, limit_buy_orders, \
          limit_if_touched_buy_orders, trailing_LIT_sell_orders, trailing_stop_limit_sell_orders, \
              stop_limit_buy_orders, stop_limit_sell_orders = ma.get_orders()
          buy_orders = pd.concat([limit_if_touched_buy_orders, limit_buy_orders])
          limit_buy_orders_list = get_orders_list_from_moomoo_orders(limit_buy_orders)
          limit_if_touched_buy_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_buy_orders)
          limit_if_touched_sell_orders_list = get_orders_list_from_moomoo_orders(limit_if_touched_sell_orders)
          stop_sell_orders_list = get_orders_list_from_moomoo_orders(stop_sell_orders)
          trailing_LIT_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_LIT_sell_orders)
          trailing_stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(trailing_stop_limit_sell_orders)
          stop_limit_sell_orders_list = get_orders_list_from_moomoo_orders(stop_limit_sell_orders)
          stop_limit_buy_orders_list = get_orders_list_from_moomoo_orders(stop_limit_buy_orders)

          # Recheck for all tickers in positional list that LIMIT/trailing_LIT_order are placed if condition
          # and trailing ratio modification
          check_sell_orders_for_all_bougth_stocks()
                                  
          for ticker3 in placed_stocks_list:
            # Checking if sell orders have been executed
            df = check_if_sell_orders_have_been_executed(df, ticker3)
            # Recheck all placed order information
            df = check_buy_order_info_from_order_history(df, ticker3)

          counter = 0
          if us_cash < min_buy_sum \
            and not test_buy_sim:
            break

        # 3.2 Get historical data, current_price for stocks in optimal list
        try:
          stock_df = get_historical_df(ticker = ticker, period=period, interval=interval, prepost=True)
          try:
            stock_df_1m = get_historical_df(ticker = ticker, period='max', interval='1m', prepost=True)
            current_price = stock_df_1m['close'].iloc[-1]
            current_timezone = datetime.now().astimezone().tzinfo
            time_is_correct =  (datetime.now().astimezone() - stock_df_1m.index[-1].astimezone(current_timezone)).seconds  < 60 * 3
            if time_is_correct:
              market_value += stock_df_1m['pct'].iloc[-2]
            else:
              market_value  += 0
            market_value_2m += minmax(stock_df_1m['pct'].iloc[-2:-1].sum(), -3, 3)
            market_value_5m += minmax(stock_df_1m['pct'].iloc[-5:-1].sum(), -3 ,3)
            market_value_30m += minmax(stock_df_1m['pct'].iloc[-30:-1].sum(), -3, 3)
            market_value_60m += minmax(stock_df_1m['pct'].iloc[-60:-1].sum(), -3, 3)
            if stock_df_1m['pct'].iloc[-60:-1].sum() > 0.2:
              market_direction_60m += 1 
            else:
              market_direction_60m -= 1
            # warning.print(f'market_value is {market_value:.2f},market_value_2m is {market_value_2m:.2f},  market_value_5m is {market_value_5m:.2f}')
            # warning.print(f'market_value_30m is {market_value_30m:.2f}, market_value_60m is {market_value_60m:.2f}')
          except Exception as e:
            if not(stock_df is None):
              current_price = stock_df['close'].iloc[-1]
            alarm.print(traceback.format_exc()) 
        except Exception as e:
          print(f'{e}')
          stock_df = None
        
        # Trailing stop limit order modification condition inside stocks optimcal cycle
        if ticker in bought_stocks_list:
          try:  
            orders = df.loc[(df['ticker'] == ticker) & df['status'].isin(['bought', 'filled_part'])].sort_values('buy_time')
            if orders.shape[0] > 0:
              order = orders.iloc[-1]
              # trailing stop limit order modification condition
              df, order= trailing_stop_limit_order_trailing_ratio_modification(df, order, current_price)
              df, order = modify_trailing_stop_limit_1230_order(df, order, current_price)
          except Exception as e:
            alarm.print(traceback.format_exc())
            
        # 3.4 BUY SECTION:
        try:
          if not(stock_df is None) and not(stock_df_1m is None):
            if not(ticker in bought_stocks_list or ticker in placed_stocks_list):
              buy_condition_MA50_MA5 = stock_buy_condition_MA50_MA5(stock_df, stock_df_1m, df_stats, ticker, display=True)
              # buy_condition, buy_condition_type_1h  = stock_buy_conditions(stock_df, stock_df_1m, ticker)
              buy_condition, buy_condition_type_1h = False, 'No cond'
              # buy_condition_930_47 = stock_buy_condition_930_47(stock_df_1m)
              buy_condition_930_47, buy_condition = False, False
              # buy_condition_1230 = stock_buy_condition_1230(stock_df, stock_df_1m, display=False)
              buy_condition_1230 = False
              # buy_condition_maxminnorm = stock_buy_condition_maxminnorm(stock_df, stock_df_1m, df_stats, display=False)
              buy_condition_maxminnorm = False
              # buy_condition_speed_norm100 = stock_buy_condition_speed_norm100(stock_df, stock_df_1m, df_stats, display=True)
              buy_condition_speed_norm100 = False
              # buy_condition_before_market_open, condition_type_before_market_open = \
              #   stock_buy_condition_before_market_open(stock_df, stock_df_1m, df_stats, display=False)   
              buy_condition_before_market_open, condition_type_before_market_open = False, 'No cond'      
            else:
              buy_condition = False
              buy_condition_1230 = False
              buy_condition_930_47 = False
              buy_condition_maxminnorm = False
              buy_condition_speed_norm100 = False
              buy_condition_before_market_open = False
              buy_condition_MA50_MA5 = False
            
            buy_condition_type = 'No cond'
            if buy_condition:
              buy_condition_type = buy_condition_type_1h 
            if buy_condition_1230:
              buy_condition_type = '1230'
            if buy_condition_930_47:
              buy_condition_type = '930-47'
            if buy_condition_maxminnorm:
              buy_condition_type = 'maxminnorm'
            if buy_condition_speed_norm100:
              buy_condition_type = 'speed_norm100'
            if buy_condition_before_market_open:
              buy_condition_type = condition_type_before_market_open
            if buy_condition_MA50_MA5:
              buy_condition_type = 'MA50_MA5'
            
            if buy_condition:
              c.green_red_print(buy_condition, 'buy condition')
              c.print(f'buy_condition_type is {buy_condition_type}', color = 'green')

            print(f'stock {ticker}, time: {stock_df.index[-1]} last price is {stock_df['close'].iloc[-1]:.2f}, pct is {stock_df['pct'].iloc[-1]:.2f}')
            current_timezone = datetime.now().astimezone().tzinfo
            time_is_correct =  (datetime.now().astimezone() - stock_df.index[-1].astimezone(current_timezone)).seconds  < 60 * 60 * 1 + 60 * 5 
            # c.print(f'Time is correct condition {time_is_correct}', color='yellow')

            buy_price = buy_price_based_on_condition(stock_df, stock_df_1m, buy_condition_type)

            # do_not_buy_schedule = (stock_df_1m.index[-1].hour == 10 and stock_df_1m.index[-1].minute >= 40) \
            #   or (stock_df_1m.index[-1].hour == 11 and stock_df_1m.index[-1].minute <= 5)
            # if do_not_buy_schedule: # change to 10:30???
            #   alarm.print(f'Do not buy schedule is active')
            do_not_buy_schedule = False

            if (buy_condition  \
                or buy_condition_1230 \
                or buy_condition_930_47 \
                or buy_condition_maxminnorm \
                or buy_condition_speed_norm100 \
                or buy_condition_before_market_open \
                or buy_condition_MA50_MA5)\
                and buy_price != 0 \
                and (time_is_correct or buy_condition_before_market_open)  \
                and not do_not_buy_schedule:

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
                alarm.print(f'''Stock price {stock_df['close'].iloc[-1]} more than maximum allowed price {max_stock_price}''')
              
              if total_market_value_60m <= 0 or total_market_direction_60m <= 0:
                security_condition = False
                alarm.print(f'''Total market value 60m {total_market_value_60m:.2f} or
                             total market direction 60m {total_market_direction_60m} are negative''')

              if (total_market_value_2m < 0 and total_market_value_5m < -1) or total_market_value_30m < -10:
                security_condition_1m = False
                # alarm.print(f'Market seems to be dropping')
              
              if total_market_value_30m < -15:
                security_condition_1h = False
                alarm.print(f'Market seems to be dropping')

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

                if (security_condition_1h and buy_condition) \
                  or buy_condition_1230 \
                  or buy_condition_930_47 \
                  or buy_condition_maxminnorm \
                  or buy_condition_speed_norm100 \
                  or buy_condition_before_market_open \
                  or buy_condition_MA50_MA5:
                  # play sound:
                  winsound.PlaySound('SystemHand', winsound.SND_ALIAS)
                  order = ti.buy_order(ticker=ticker, buy_price=buy_price, buy_condition_type=buy_condition_type, default=default, buy_sum=buy_sum)

                  if buy_condition_type == 'speed_norm100':
                    order['gain_coef'] = default.gain_coef_speed_norm100
                  elif buy_condition_type in ['before_market_open_1', 'before_market_open_2', 'before_market_open_3']:
                    order['gain_coef'] = default.gain_coef_before_market_open
                    order['lose_coef'] = default.lose_coef_before_market_open
                  elif buy_condition_type == 'MA50_MA5':
                    order['gain_coef'] = default.gain_coef_MA50_MA5
                    order['lose_coef'] = default.lose_coef_1_MA50_MA5
                  else:
                    order['gain_coef'] = default.gain_coef
                  
                  if buy_condition_type == '1h':
                    order['lose_coef'] = lose_coef_1h
  
                  order['tech_indicators'] = f'mv :{market_value:.2f}, mv_2m:{market_value_2m:.2f},\
      mv_5m : {market_value_5m:.2f}, mv_30m : {market_value_30m:.2f}, mv_60m: {market_value_60m:.2f}, md_60m: {market_direction_60m}'
                  if order['status'] == 'placed':
                    placed_stocks_list.append(ticker)
                    df = ti.record_order(df, order)
                    # Recheck all placed buy orders from the order history
                    df = check_buy_order_info_from_order_history(df, ticker)
                    if place_trailing_stop_limit_order_imidiately \
                      and not (order['buy_condition_type'] in ['1230', 'speed_norm100', 'before_market_open_1', 
                                                               'before_market_open_2', 'before_market_open_3', 'MA50_MA5']):
                      price = order['buy_price'] * default.gain_coef
                      # Checking trailing_stop_limit_order
                      # df, order = check_sell_order_has_been_placed(df, order, ticker, order_type='trailing_stop_limit')
                      df, order = place_sell_order_if_it_was_not_placed(df,
                      order=order,
                      sell_orders=trailing_stop_limit_sell_orders,
                      sell_orders_list=trailing_stop_limit_sell_orders_list,
                      price=price,
                      order_type='trailing_stop_limit') 
        except Exception as e:
          alarm.print(traceback.format_exc())


        # 3.5 Recheck placed orders information including commission from the order history
        df = check_buy_order_info_from_order_history(df, ticker)
        # 3.6 CHECKING FOR CANCELATION OF THE BUY ORDER
        df = check_buy_order_for_cancelation(df, ticker, historical_orders)
        # 3.7 Checking if sell orders have been executed
        df = check_if_sell_orders_have_been_executed(df, ticker)

    total_market_value = market_value
    total_market_value_2m = market_value_2m
    total_market_value_5m = market_value_5m
    total_market_value_30m = market_value_30m 
    total_market_value_60m = market_value_60m
    total_market_direction_60m = market_direction_60m

    # Update SQL DB FROM df each full cycle!!!
    # try:
    #   db.update_db_from_df(df)
    # except Exception as e:
    #   alarm.print(traceback.format_exc())
    print('Waiting progress:')
    if us_cash > min_buy_sum:
      for i in tqdm(range(25)):
       time.sleep(1)
    else:
      for i in tqdm(range(20)):
        time.sleep(1)
# %%
